"""Behavioral contracts for the L2 concentration-penalty sweep harness.

The sweep (tools/sweep_us_l2_lambda.py) must run the production two-stage
calibration per point with the right penalty in the right stage, write
resumable per-point artifacts with full provenance, and reuse a build's
target-frame checkpoint without requiring its identity to be re-derived.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.calibrate import TargetRegistry, TargetSpec
from populace.frame import EntitySchema, Frame, WeightKind, Weights


def _load_tool_module(name: str):
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_sweep_module():
    return _load_tool_module("sweep_us_l2_lambda")


def _load_builder_module():
    return _load_tool_module("build_us_fiscal_refresh_release")


def _sweep_frame_and_registry(n: int = 120) -> tuple[Frame, TargetRegistry]:
    """A household frame whose targets are easiest to hit by concentrating."""
    rng = np.random.default_rng(0)
    income = rng.lognormal(10.5, 1.0, n)
    weights = np.full(n, 1000.0)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n, dtype="int64"),
                    "person_household_id": np.arange(n, dtype="int64"),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.arange(n, dtype="int64"),
                    "household_count": np.ones(n),
                    "income": income,
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )
    registry = TargetRegistry(
        (
            TargetSpec(
                name="income",
                entity="household",
                measure="income",
                value=float((income * weights).sum()) * 1.25,
                period=2024,
                source="test",
                family="test",
            ),
            TargetSpec(
                name="population",
                entity="household",
                measure="household_count",
                value=float(weights.sum()),
                period=2024,
                source="test",
                family="test",
            ),
        ),
        country="us",
    )
    return frame, registry


def test_expand_sweep_points_runs_baseline_once_and_expands_arms() -> None:
    sweep = _load_sweep_module()

    points = sweep.expand_sweep_points(
        [0.0, 1e-3, 1e-3, 1e-2, 0.0],
        ["both", "selection-only", "refit-only"],
    )

    ids = [point["point_id"] for point in points]
    assert ids == [
        "baseline",
        "both-0.001",
        "selection-only-0.001",
        "refit-only-0.001",
        "both-0.01",
        "selection-only-0.01",
        "refit-only-0.01",
    ]
    by_id = {point["point_id"]: point for point in points}
    assert by_id["baseline"] == {
        "point_id": "baseline",
        "arm": "baseline",
        "l2_lambda": 0.0,
        "refit_l2_lambda": None,
    }
    # 'both' inherits (None); the other arms pin the complementary stage to 0.
    assert by_id["both-0.001"]["l2_lambda"] == 1e-3
    assert by_id["both-0.001"]["refit_l2_lambda"] is None
    assert by_id["selection-only-0.001"]["l2_lambda"] == 1e-3
    assert by_id["selection-only-0.001"]["refit_l2_lambda"] == 0.0
    assert by_id["refit-only-0.001"]["l2_lambda"] == 0.0
    assert by_id["refit-only-0.001"]["refit_l2_lambda"] == 1e-3


def test_expand_sweep_points_rejects_bad_input() -> None:
    sweep = _load_sweep_module()

    with pytest.raises(ValueError, match="arm"):
        sweep.expand_sweep_points([1e-3], ["bogus"])
    with pytest.raises(ValueError, match="non-negative"):
        sweep.expand_sweep_points([-1e-3], ["both"])
    with pytest.raises(ValueError, match="empty"):
        sweep.expand_sweep_points([], ["both"])


def test_run_sweep_threads_each_arm_penalty_into_the_right_stage(
    tmp_path,
) -> None:
    sweep = _load_sweep_module()
    frame, registry = _sweep_frame_and_registry()
    points = sweep.expand_sweep_points([0.0, 1e-3], ["both", "refit-only"])
    provenance = {"target_registry_version": registry.version}

    rows = sweep.run_sweep(
        frame,
        registry.to_target_set(),
        points,
        registry=registry,
        out_dir=tmp_path,
        epochs=80,
        learning_rate=0.05,
        max_weight_ratio=5.0,
        l0_refit_lambda_share=0.8,
        seed=0,
        target_loss_cap=1.0,
        provenance=provenance,
        log=lambda message: None,
    )

    assert [row["point_id"] for row in rows] == [
        "baseline",
        "both-0.001",
        "refit-only-0.001",
    ]
    # The stage-level provenance in each point's diagnostics proves the
    # penalty landed where the arm says: refit options carry the refit
    # lambda, selection_options the selection lambda.
    expected = {
        "baseline": (0.0, 0.0),
        "both-0.001": (1e-3, 1e-3),
        "refit-only-0.001": (0.0, 1e-3),
    }
    for point_id, (selection_l2, refit_l2) in expected.items():
        payload = json.loads(
            (
                tmp_path / "points" / point_id / "calibration_diagnostics.json"
            ).read_text()
        )
        assert payload["options"]["l2_lambda"] == refit_l2
        assert payload["options"]["selection_options"]["l2_lambda"] == selection_l2
        assert payload["build"]["l2_sweep"]["point_id"] == point_id
        assert (
            payload["build"]["l2_sweep"]["target_registry_version"] == registry.version
        )
        assert payload["effective_sample_size"] > 0.0

    summary = json.loads((tmp_path / "sweep_summary.json").read_text())
    assert summary["points"] == rows
    for row in rows:
        assert row["n_candidate_households"] == frame.n("household")
        assert 0 < row["n_selected_households"] <= row["n_candidate_households"]
        assert row["refit_effective_sample_size"] > 0.0
        assert row["refit_final_loss"] >= 0.0
    csv_text = (tmp_path / "sweep_summary.csv").read_text()
    assert csv_text.splitlines()[0].startswith("point_id,arm,l2_lambda")
    assert len(csv_text.splitlines()) == 1 + len(rows)


def test_run_sweep_resumes_completed_points(tmp_path) -> None:
    sweep = _load_sweep_module()
    frame, registry = _sweep_frame_and_registry()
    points = sweep.expand_sweep_points([0.0], ["both"])

    common = dict(
        out_dir=tmp_path,
        epochs=60,
        learning_rate=0.05,
        max_weight_ratio=5.0,
        l0_refit_lambda_share=0.8,
        seed=0,
        target_loss_cap=1.0,
    )
    targets = registry.to_target_set()
    first = sweep.run_sweep(frame, targets, points, **common, log=lambda message: None)

    messages: list[str] = []
    second = sweep.run_sweep(frame, targets, points, **common, log=messages.append)

    assert second == first
    assert any("skipping" in message for message in messages)


def _write_problem_bundle(directory: Path, n: int = 40) -> dict[str, np.ndarray]:
    """A tiny exported-problem bundle: 2 target rows over n households."""
    import scipy.sparse

    rng = np.random.default_rng(0)
    income = rng.lognormal(10.5, 1.0, n)
    ones = np.ones(n)
    dense = np.vstack([income, ones])
    weights = np.full(n, 1000.0)
    values = np.asarray([float((income * weights).sum()) * 1.25, float(weights.sum())])
    loss_weights = np.asarray([1.5, 0.5])

    directory.mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(
        directory / "constraint_matrix_csr.npz", scipy.sparse.csr_matrix(dense)
    )
    np.save(directory / "target_values.npy", values)
    np.save(directory / "initial_weights.npy", weights)
    np.save(directory / "target_loss_weights_production_us_fiscal.npy", loss_weights)
    (directory / "target_names.json").write_text(
        json.dumps(["income@2024", "population@2024"])
    )
    (directory / "target_rows_minimal.jsonl").write_text(
        '{"name": "income", "period": 2024}\n{"name": "population", "period": 2024}\n'
    )
    (directory / "metadata.json").write_text(
        json.dumps({"source_precalibration": "test-run"})
    )
    return {
        "dense": dense,
        "values": values,
        "weights": weights,
        "loss_weights": loss_weights,
    }


def test_load_problem_bundle_replays_the_compiled_problem(tmp_path) -> None:
    """The bundle mode must reproduce the exported matrix bit-for-bit.

    The sweep's claim to production-faithfulness in bundle mode rests on the
    synthetic frame + callable-measure targets compiling back to exactly the
    exported constraint system.
    """
    from populace.calibrate import build_constraint_matrix

    sweep = _load_sweep_module()
    bundle_dir = tmp_path / "bundle"
    arrays = _write_problem_bundle(bundle_dir)

    frame, targets, loss_weights, provenance = sweep.load_problem_bundle(bundle_dir)

    np.testing.assert_allclose(frame.weights_for("household").values, arrays["weights"])
    assert [(target.name, target.period) for target in targets] == [
        ("income", 2024),
        ("population", 2024),
    ]
    np.testing.assert_allclose(loss_weights, arrays["loss_weights"])
    bundle = provenance["problem_bundle"]
    assert bundle["n_targets"] == 2
    assert bundle["n_records"] == arrays["weights"].size
    assert bundle["metadata"] == {"source_precalibration": "test-run"}

    problem = build_constraint_matrix(frame, targets, "household")
    assert not problem.skipped
    np.testing.assert_allclose(problem.matrix.toarray(), arrays["dense"])
    np.testing.assert_allclose(problem.target_vector, arrays["values"])


def test_load_problem_bundle_initial_weights_override(tmp_path) -> None:
    """A supplied design-weight vector replaces the bundle's recorded start."""
    sweep = _load_sweep_module()
    bundle_dir = tmp_path / "bundle"
    arrays = _write_problem_bundle(bundle_dir)
    n = arrays["weights"].size
    design = np.linspace(500.0, 5000.0, n)
    override_path = tmp_path / "design_weights.npy"
    np.save(override_path, design)

    frame, _, _, provenance = sweep.load_problem_bundle(
        bundle_dir, initial_weights_path=override_path
    )

    np.testing.assert_allclose(frame.weights_for("household").values, design)
    override = provenance["problem_bundle"]["initial_weights_override"]
    assert override["path"] == str(override_path.resolve())
    assert len(override["sha256"]) == 64

    bad_path = tmp_path / "bad_weights.npy"
    np.save(bad_path, np.ones(3))
    with pytest.raises(ValueError, match="shape"):
        sweep.load_problem_bundle(bundle_dir, initial_weights_path=bad_path)


def test_run_sweep_over_a_problem_bundle_needs_no_registry(tmp_path) -> None:
    """End to end on the bundle input: registry-less diagnostics still ship."""
    sweep = _load_sweep_module()
    bundle_dir = tmp_path / "bundle"
    _write_problem_bundle(bundle_dir)
    frame, targets, loss_weights, provenance = sweep.load_problem_bundle(bundle_dir)
    points = sweep.expand_sweep_points([0.0], ["both"])

    rows = sweep.run_sweep(
        frame,
        targets,
        points,
        out_dir=tmp_path / "sweep",
        epochs=60,
        learning_rate=0.05,
        max_weight_ratio=None,
        l0_refit_lambda_share=0.3,
        seed=0,
        target_loss_cap=1.0,
        target_loss_weights=loss_weights,
        provenance=provenance,
        log=lambda message: None,
    )

    assert len(rows) == 1
    payload = json.loads(
        (
            tmp_path / "sweep" / "points" / "baseline" / "calibration_diagnostics.json"
        ).read_text()
    )
    assert payload["options"]["post_l0_refit"] is True
    assert payload["build"]["l2_sweep"]["problem_bundle"]["n_targets"] == 2
    assert payload["effective_sample_size"] > 0.0
    assert rows[0]["refit_final_loss"] >= 0.0


def test_read_target_frame_checkpoint_file_needs_no_expected_identity(
    monkeypatch,
    tmp_path,
    small_frame,
) -> None:
    """The sweep loads a build's checkpoint as-is and records its identity."""
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "US_SCHEMA", small_frame.schema)
    identity = builder._target_frame_checkpoint_identity(
        base_dataset_sha256="base-sha",
        policyengine_us_version="1.2.3",
        seed=0,
        target_period=builder.PERIOD,
        target_registry_version="registry-sha",
        congressional_district_vintage_crosswalk_sha256=None,
    )
    path = tmp_path / "target_frame_checkpoint.h5"
    builder._write_target_frame_checkpoint(
        path,
        frame=small_frame,
        identity=identity,
        compilation={"declared_targets": 1},
    )

    frame, stored_identity, stored_compilation = (
        builder._read_target_frame_checkpoint_file(path)
    )

    assert stored_identity == identity
    assert stored_compilation == {"declared_targets": 1}
    np.testing.assert_allclose(
        frame.weights_for("household").values,
        small_frame.weights_for("household").values,
    )
    pd.testing.assert_frame_equal(frame.table("person"), small_frame.table("person"))
