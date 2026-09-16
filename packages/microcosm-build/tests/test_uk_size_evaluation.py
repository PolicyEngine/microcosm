"""Synthetic contract tests for UK dataset-size evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.size_evaluation import (
    area_support_tables,
    dense_reference_deltas,
    footprint,
    frozen_vs_recomputed,
    gate_table,
    load_run,
    paired_targets,
    run_acceptance,
    summarize,
    weight_tables,
)

# The evaluation reads and writes PyTables-format H5 through pandas; the wheels
# lane's venv has no pytables, so these tests skip there like the other H5 tests.
pytest.importorskip("tables", exc_type=ModuleNotFoundError)

from microcosm.calibrate.solve import effective_sample_size


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_dir(
    tmp_path: Path,
    label: str,
    *,
    size_run: bool,
    red_rows: tuple[str, ...] = (),
) -> Path:
    """Write every documented run artifact as a small deterministic fixture."""

    run = tmp_path / label
    run.mkdir()
    reference = "reference" in label
    rows = [
        (
            "local/council-tax/E1",
            "constituency",
            "E1",
            "council_tax/band_c",
            "council_tax",
            100.0,
            0.40 if reference else 0.30,
        ),
        ("local/income/E1", "constituency", "E1", "income/pay", "income", 100.0, 0.05),
        (
            "local/employment/E2",
            "constituency",
            "E2",
            "employment/count",
            "employment",
            100.0,
            0.05 if reference else 0.10,
        ),
        (
            "local/age/E2",
            "constituency",
            "E2",
            "age/20_30",
            "age",
            100.0,
            0.03 if reference else 0.02,
        ),
        (
            "local/income/LA1",
            "local_authority",
            "LA1",
            "income/pay",
            "income",
            100.0,
            0.20,
        ),
        (
            "local/benefit/LA1",
            "local_authority",
            "LA1",
            "benefit/count",
            "benefit",
            100.0,
            0.02 if reference else 0.01,
        ),
        (
            "external:households/E1",
            "constituency",
            "E1",
            "households",
            "census_households",
            100.0,
            0.04 if reference else 0.05,
        ),
        (
            "external:households/E2",
            "constituency",
            "E2",
            "households",
            "census_households",
            100.0,
            0.10,
        ),
    ]
    national = [
        ("national/a", "national_a", 100.0, 0.10 if reference else 0.05),
        (
            "national/b",
            "national_b",
            100.0 if reference else 101.0,
            0.20 if reference else 0.30,
        ),
    ]
    diagnostics_rows = []
    targets = []
    for index, (
        target_name,
        area_type,
        area_code,
        metric,
        family,
        target,
        error,
    ) in enumerate(rows):
        estimate = target * (1 + error)
        diagnostics_rows.append(
            {
                "target_index": index,
                "area_type": area_type,
                "area_code": area_code,
                "area_index": index,
                "metric": metric,
                "metric_index": index,
                "value": target,
                "target_name": target_name,
                "family": family,
                "source_family": family,
                "source": "fixture",
                "period": 2025,
                "contract_target_id": f"contract-{index}",
                "target": target,
                "initial_estimate": target * 0.5,
                "final_estimate": estimate,
                "relative_error": error,
                "abs_relative_error": abs(error),
            }
        )
        targets.append(
            {
                "name": f"{target_name}@2025",
                "target_name": target_name,
                "period": 2025,
                "target": target,
                "initial_estimate": target * 0.5,
                "final_estimate": estimate,
                "relative_error": error,
                "target_loss_scale": target,
                "target_loss_weight": 1.0,
                "metadata": {
                    "area_type": area_type,
                    "area_code": area_code,
                    "metric": metric,
                    "family": family,
                    "contract_target_id": f"contract-{index}",
                },
            }
        )
    for offset, (name, family, target, error) in enumerate(national, start=len(rows)):
        estimate = target * (1 + error)
        targets.append(
            {
                "name": f"{name}@2025",
                "target_name": name,
                "period": 2025,
                "target": target,
                "initial_estimate": target * 0.5,
                "final_estimate": estimate,
                "relative_error": error,
                "target_loss_scale": target,
                "target_loss_weight": 1.0,
                "metadata": {
                    "family": family,
                    "contract_target_id": f"contract-{offset}",
                    "ledger_source": "fixture",
                },
            }
        )
    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics_path = run / "solve_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    calibration = {
        "schema_version": 6,
        "targets": targets,
        "uk_diagnostics": {
            "weights": {"effective_sample_size": 5.0},
            "rotated_holdout": {
                "mean_holdout_loss": 0.1,
                "worst_holdout_loss": 0.2,
                "fold_losses": [0.1, 0.2],
                "n_folds": 2,
                "report_only": True,
            },
        },
    }
    calibration_path = run / "calibration_diagnostics.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    support = pd.DataFrame(
        {
            "geography_level": ["constituency", "constituency", "local_authority"],
            "area_code": ["E1", "E2", "LA1"],
            "assigned_households": [40, 60, 70],
            "nonzero_households": [40, 60, 70],
            "nonzero_source_households": [40, 60, 40],
            "weight_sum": [100.0, 100.0, 100.0],
            "max_weight": [4.0, 4.0, 4.0],
            "effective_sample_size": [55.0, 45.0, 60.0],
            "support_below_floor": [True, True, True],
        }
    )
    support_path = run / "area_support_summary.csv"
    support.to_csv(support_path, index=False)

    gate_ids = (
        "uk_local_geography_ladder_post_calibration",
        "uk_local_area_support",
        "uk_local_target_fit",
        "uk_local_per_family_fit",
        "uk_local_weight_ratio",
        "uk_local_weight_ess",
    )
    gates = {}
    for gate_id in gate_ids:
        details = {}
        if gate_id == "uk_local_area_support":
            details = {
                "areas_checked": 3,
                "areas_failed": 3,
                "by_geography_level": {},
                "reviewed_exclusions": {"constituency/E1": {"reason": "fixture"}},
            }
        if gate_id == "uk_local_target_fit":
            details = {
                "failing_targets": {row: 0.4 for row in red_rows},
                "max_abs_relative_error": 0.4,
                "targets_checked": 8,
            }
        gates[gate_id] = {
            "criticality": "diagnostic" if reference else "release_blocking",
            "details": details,
            "failures": ["fixture"] if details.get("failing_targets") else None,
            "gate": gate_id,
            "phase": "post_calibration",
            "reason": None,
            "status": "failed" if details.get("failing_targets") else "passed",
        }
    gate_report = {
        "gates": gates,
        "shippable": False,
        "posture": "diagnostic",
        "release_candidate": False,
        "policy_sha256": "a" * 64,
        "gates_manifest_sha256": "b" * 64,
    }
    gate_path = run / "microcosm_uk_2025_local.local_gates.json"
    gate_path.write_text(json.dumps(gate_report), encoding="utf-8")

    if size_run:
        household = pd.DataFrame(
            {
                "household_id": [1, 11, 21, 2, 12, 22],
                "household_weight": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "clone_index": [0, 1, 2, 0, 1, 2],
            }
        )
    else:
        household = pd.DataFrame(
            {
                "household_id": [1, 11, 2],
                "household_weight": [5.0, 5.0, 5.0],
                "clone_index": [0, 1, 0],
            }
        )
    h5_path = run / "microcosm_uk_2025_local.h5"
    with pd.HDFStore(h5_path, mode="w") as store:
        store.put("household", household, format="table")

    dense_path = run / "dense_reference_diagnostics.csv"
    selection_path = run / "dataset_size_selection.csv"
    if size_run:
        dense_local = diagnostics.copy()
        dense_local.insert(
            0,
            "grain",
            np.where(
                dense_local["target_name"].str.startswith("external:"),
                "ladder",
                dense_local["area_type"],
            ),
        )
        dense_national = pd.DataFrame(
            [
                {
                    "grain": "national",
                    "name": f"{name}@2025",
                    "family": family,
                    "target": target,
                    "initial_estimate": target * 0.5,
                    "final_estimate": target * (1 + error / 2),
                    "relative_error": error / 2,
                    "abs_relative_error": abs(error / 2),
                }
                for name, family, target, error in national
            ]
        )
        pd.concat([dense_local, dense_national], ignore_index=True, sort=False).to_csv(
            dense_path, index=False
        )
        pd.DataFrame(
            {
                "pool_row_index": range(6),
                "household_id": household["household_id"],
                "clone_index": household["clone_index"],
                "design_weight": [1.0] * 6,
                "inclusion_probability": [0.5] * 6,
                "certainty": [True, False, False, True, False, False],
                "ht_baseline_weight": [2.0] * 6,
                "refit_weight": household["household_weight"],
            }
        ).to_csv(selection_path, index=False)

    (run / "run.log").write_text(
        "    11122.00 real     13204.25 user       179.23 sys\n"
        "         11103666176  maximum resident set size\n",
        encoding="utf-8",
    )
    registry_path = run / "scoring_registry_compiled.json"
    registry_path.write_text("{}\n", encoding="utf-8")

    output_paths = [
        diagnostics_path,
        calibration_path,
        support_path,
        gate_path,
        h5_path,
    ]
    if size_run:
        output_paths.extend([dense_path, selection_path])
    outputs = {
        path.stem: {
            "path": f"/old/machine/{path.name}",
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in output_paths
    }
    dense_summary = {
        "initial_loss": 0.03,
        "final_loss": 0.01,
        "n_nonzero": 6,
        "n_households": 6,
        "max_abs_relative_error": 0.20,
        "median_abs_relative_error": 0.04,
        "national_max_abs_relative_error": 0.15,
        "past_cap": {"past_at_final": 0},
        "weights": {"effective_sample_size": 6.0},
        "local_by_family": [],
        "national_by_family": [],
        "diagnostics_file": dense_path.name,
    }
    size_receipt = (
        {
            "method": "sampford_pps_fixed_size",
            "requested_households": 6,
            "realized_households": 6,
            "pool_households": 30,
            "seed": 2025,
            "protected_carriers": 2,
            "selection_receipt": {
                "k": 6,
                "pi_hi": 1.0,
                "seed": 2025,
                "certainty_count": 2,
                "boundary_pool_size": 4,
                "design": "sampford",
            },
            "selection_l0_lambda": 0.1,
            "selection_epochs": 10,
            "refit_epochs": 10,
            "pool_row_indices": list(range(6)),
            "inclusion_probabilities": [0.5] * 6,
            "refit_baseline": "normalized_horvitz_thompson",
            "stretch_reference": "normalized_horvitz_thompson_w_over_q",
            "dense_loss": 0.01,
            "compact_loss": 0.02,
            "max_target_scaled_change": 0.6,
            "certification": None,
            "dense_reference": dense_summary,
        }
        if size_run
        else None
    )
    manifest = {
        "git_commit": "fixture-commit",
        "identity": {"spine": {}, "ledger": {}, "runtime": {}},
        "parameters": {
            "n_clones": 15,
            "seed": 2025,
            "epochs": 10,
            "engine_blocks": 1,
            "sample_fraction": 1.0,
            "dataset_households": 6 if size_run else None,
            "selection_seed": 2025 if size_run else None,
        },
        "solve": {
            "n_targets": 10,
            "n_targets_by_kind": {"local": 6, "ladder": 2, "national": 2},
            "n_households": len(household),
            "pool_households": 30 if size_run else len(household),
            "initial_loss": 0.04,
            "final_loss": 0.02,
            "max_abs_relative_error": 0.30,
            "median_abs_relative_error": 0.08,
            "n_nonzero": len(household),
            "past_cap": {"past_at_final": 2},
            "measure_resolution": {
                "blocks": 1,
                "engine_version": "fixture",
                "mode": "fixture",
            },
            "dataset_size": size_receipt,
        },
        "weights": {
            "declared_stretch_bound": 100.0,
            "realized_max_weight_ratio_vs_design": 3.0 if size_run else 1.0,
            **(
                {"stretch_reference": "normalized_horvitz_thompson_w_over_q"}
                if size_run
                else {}
            ),
            "calibration_mass_change": {"old_total": 12.0, "new_total": 12.0},
        },
        "fit": {
            "local_by_family": [],
            "national_by_family": [],
            "rotated_holdout": calibration["uk_diagnostics"]["rotated_holdout"],
        },
        "releasable": False if size_run else True,
        **(
            {"release_posture": {"size_certification_present": False}}
            if size_run
            else {}
        ),
        "outputs": outputs,
    }
    (run / "rowwise_candidate_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run


def test_load_run_shapes_and_grains(tmp_path: Path) -> None:
    size_path = _write_run_dir(tmp_path, "candidate", size_run=True)
    dense_path = _write_run_dir(tmp_path, "dense", size_run=False)

    size = load_run(size_path, label="candidate")
    dense = load_run(dense_path, label="dense")

    assert len(size.targets) == 10
    assert size.targets["grain"].value_counts().to_dict() == {
        "constituency": 4,
        "local_authority": 2,
        "ladder": 2,
        "national": 2,
    }
    assert len(size.local_diagnostics) == 8
    assert size.dense_reference_diagnostics is not None
    assert size.selection is not None
    assert dense.dense_reference_diagnostics is None
    assert dense.selection is None


def test_paired_targets_counts_red_mapping_and_changed_target(tmp_path: Path) -> None:
    run = load_run(
        _write_run_dir(tmp_path, "candidate", size_run=True), label="candidate"
    )
    reference = load_run(
        _write_run_dir(
            tmp_path,
            "reference",
            size_run=False,
            red_rows=("council_tax/E1/council_tax/band_c",),
        ),
        label="reference",
    )

    result = paired_targets(run, reference)

    assert result["n_common"] == 10
    assert result["n_only_run"] == result["n_only_reference"] == 0
    assert result["n_target_value_changed"] == 1
    assert (result["wins"], result["ties"], result["losses"]) == (4, 3, 3)
    assert result["reference_red_rows"]["n"] == 1
    assert result["reference_red_rows"]["rows"][0]["name"] == (
        "local/council-tax/E1@2025"
    )
    assert result["reference_red_rows"]["still_red"] == 1
    assert any(row["name"] == "national/b@2025" for row in result["new_red_rows"])


def test_weight_tables_size_and_dense_spine_paths(tmp_path: Path) -> None:
    size = load_run(
        _write_run_dir(tmp_path, "candidate", size_run=True), label="candidate"
    )
    result = weight_tables(size)

    values = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assert result["summary"]["effective_sample_size"] == pytest.approx(
        effective_sample_size(values)
    )
    assert result["distinct_source_households"] == 2
    assert result["source_id_multiplier_basis"] == "inferred_from_clone_zero_ids"
    assert result["stretch"]["vs_pool_design"]["max"] == pytest.approx(3.0)
    assert result["clones_per_source"]["by_clone_count"] == {"3": 2}
    skipped = load_run(size.path, label="candidate", skip_h5=True)
    assert weight_tables(skipped)["summary"]["effective_sample_size"] == pytest.approx(
        effective_sample_size(values)
    )

    dense = load_run(_write_run_dir(tmp_path, "dense", size_run=False), label="dense")
    spine = pd.DataFrame(
        {"household_id": [1, 2, 3], "household_weight": [75.0, 75.0, 75.0]}
    )
    dense_result = weight_tables(dense, spine_weights=spine)
    assert dense_result["stretch"]["reference"] == "pool_design"
    assert dense_result["stretch"]["vs_pool_design"]["max"] == pytest.approx(1.0)
    dense.manifest["weights"]["realized_max_weight_ratio_vs_design"] = 2.0
    with pytest.raises(ValueError, match="does not match"):
        weight_tables(dense, spine_weights=spine)


def test_area_support_and_gate_tables(tmp_path: Path) -> None:
    run = load_run(
        _write_run_dir(tmp_path, "candidate", size_run=True), label="candidate"
    )
    areas = area_support_tables(run)

    assert areas["by_geography_level"]["constituency"]["breaches"] == {
        "rows_lt_50": 1,
        "ess_lt_50": 1,
        "sources_lt_50": 1,
        "any": 2,
        "codes": ["E1", "E2"],
    }
    assert "constituency/E1" in areas["reviewed_exclusions"]
    gates = gate_table(run)
    assert gates[0]["criticality"] == "release_blocking"
    del run.gates["gates"]["uk_local_weight_ess"]
    assert gate_table(run)[-1]["status"] == "absent"


def test_run_acceptance_statuses_and_bad_digest(tmp_path: Path) -> None:
    path = _write_run_dir(tmp_path, "candidate", size_run=True)
    run = load_run(path, label="candidate")
    result = run_acceptance(
        run, expected_households=6, expected_pool=30, expected_epochs=10
    )
    assert result["passed"]
    assert {row["status"] for row in result["checks"]} <= {
        "pass",
        "not_applicable",
    }

    # A refit whose baseline was trimmed (--baseline-pi-floor) names its stretch
    # reference with the _floored suffix; the acceptance accepts either name.
    manifest_path = path / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    floored_name = "normalized_horvitz_thompson_w_over_q_floored"
    manifest["weights"]["stretch_reference"] = floored_name
    manifest["solve"]["dataset_size"]["stretch_reference"] = floored_name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    floored = run_acceptance(
        load_run(path, label="candidate"),
        expected_households=6,
        expected_pool=30,
        expected_epochs=10,
    )
    stretch = next(row for row in floored["checks"] if row["id"] == "stretch_reference")
    assert stretch["status"] == "pass"
    assert stretch["observed"] == floored_name

    (path / "solve_diagnostics.csv").write_text("wrong\n", encoding="utf-8")
    bad = run_acceptance(run)
    assert not bad["passed"]
    digest = next(row for row in bad["checks"] if row["id"] == "output_sha256")
    assert digest["status"] == "fail"

    dense_path = _write_run_dir(tmp_path, "dense", size_run=False)
    dense = load_run(dense_path, label="dense")
    dense_result = run_acceptance(dense)
    assert (
        next(
            row for row in dense_result["checks"] if row["id"] == "dataset_households"
        )["status"]
        == "not_applicable"
    )


def test_footprint_parsing_and_null_fallback(tmp_path: Path) -> None:
    path = _write_run_dir(tmp_path, "candidate", size_run=True)
    run = load_run(path, label="candidate")
    assert footprint(run) == {
        "wall_seconds": 11122.0,
        "peak_rss_bytes": 11103666176,
        "h5_bytes": (path / "microcosm_uk_2025_local.h5").stat().st_size,
    }

    (path / "run.log").unlink()
    no_runtime = load_run(path, label="candidate", skip_h5=True)
    assert footprint(no_runtime)["wall_seconds"] is None
    assert footprint(no_runtime)["peak_rss_bytes"] is None


def test_dense_deltas_and_frozen_surface(tmp_path: Path) -> None:
    run = load_run(
        _write_run_dir(tmp_path, "candidate", size_run=True), label="candidate"
    )
    dense = dense_reference_deltas(run)
    assert dense is not None
    assert dense["n_common"] == 10
    assert dense["dense_loss"] == pytest.approx(0.01)
    assert dense["compact_loss"] == pytest.approx(0.02)

    surface_path = tmp_path / "surface.json"
    surface_path.write_text(
        json.dumps(
            {
                "national_rows": [
                    {"our_name": "national/a@2025", "candidate_estimate": 107.1},
                    {"our_name": "national/b@2025", "candidate_estimate": 131.3},
                ]
            }
        ),
        encoding="utf-8",
    )
    surface = frozen_vs_recomputed(run, surface_path)
    assert surface["n_rows"] == surface["n_matched"] == 2
    assert surface["max_abs_rel_divergence"] == pytest.approx(0.02)
    assert [row["name"] for row in surface["rows_over_1pct"]] == ["national/a@2025"]


def test_summarize_outcome_flags(tmp_path: Path) -> None:
    run = load_run(
        _write_run_dir(tmp_path, "candidate", size_run=True), label="candidate"
    )
    result = summarize(
        run,
        references={},
        dense_deltas=dense_reference_deltas(run),
        surface=None,
    )
    outcomes = {row["id"]: row for row in result["pre_registered_outcomes"]}
    assert outcomes["dense_reference_loss"]["flag"] == "ok"
    assert outcomes["max_target_scaled_change"]["flag"] == "red"
    assert outcomes["frozen_vs_recomputed_max"]["flag"] == "not_measured"


@pytest.mark.parametrize(
    "missing",
    [
        "rowwise_candidate_manifest.json",
        "solve_diagnostics.csv",
        "calibration_diagnostics.json",
        "area_support_summary.csv",
        "microcosm_uk_2025_local.local_gates.json",
    ],
)
def test_load_run_refuses_missing_required_file(tmp_path: Path, missing: str) -> None:
    path = _write_run_dir(tmp_path, "candidate", size_run=True)
    (path / missing).unlink()
    expected = (
        r"\*\.local_gates\.json"
        if missing.endswith(".local_gates.json")
        else re.escape(missing)
    )
    with pytest.raises(ValueError, match=expected):
        load_run(path, label="candidate")
