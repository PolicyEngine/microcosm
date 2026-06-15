import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from populace.frame import Frame, WeightKind


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_soi_component_amounts_use_source_specific_signs() -> None:
    builder = _load_builder_module()

    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "capital_gains_gross"),
        np.array([0.0, 0.0, 7.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "capital_gains_losses"),
        np.array([0.0, 0.0, 7.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "business_net_losses"),
        np.array([5.0, -0.0, -0.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 7.0]), "adjusted_gross_income"),
        np.array([-5.0, 7.0]),
    )


def test_soi_count_rows_count_positive_component_items() -> None:
    builder = _load_builder_module()

    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "capital_gains_gross",
            count=True,
        ),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "capital_gains_losses",
            count=True,
        ),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "business_net_losses",
            count=True,
        ),
        np.array([1.0, 0.0, 0.0]),
    )


def test_release_gate_failures_are_not_unconditional() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(object(),),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == []

    assert builder._release_gate_failures(
        result,
        {"dropped_target_names": ["missing"]},
    ) == ["1 fiscal targets were not materialized."]

    skipped = SimpleNamespace(target=SimpleNamespace(name="skipped"), reason="bad")
    with_skipped = SimpleNamespace(
        skipped=(skipped,),
        diagnostics=(object(),),
        initial_loss=10.0,
        final_loss=5.0,
    )
    assert builder._release_gate_failures(
        with_skipped,
        {"dropped_target_names": []},
    ) == ["1 fiscal targets were skipped by calibration."]

    worse = SimpleNamespace(
        skipped=(),
        diagnostics=(object(),),
        initial_loss=5.0,
        final_loss=10.0,
    )
    assert builder._release_gate_failures(worse, {"dropped_target_names": []}) == [
        "Calibration final loss is worse than the initial loss (10.0 > 5.0)."
    ]


def test_release_gate_failures_reject_positive_zero_support_targets() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=f"nation/irs/zero@{builder.PERIOD}",
                target=1_000.0,
                initial_estimate=0.0,
                final_estimate=0.0,
            ),
            SimpleNamespace(
                name=f"nation/irs/nonzero@{builder.PERIOD}",
                target=1_000.0,
                initial_estimate=10.0,
                final_estimate=20.0,
            ),
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == [
        "1 positive fiscal targets have zero materialized support "
        f"(examples: nation/irs/zero@{builder.PERIOD})."
    ]


def test_build_manifests_emits_policyengine_certifiable_release_manifest(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    release_id = "populace-us-2024-abcdef1-20260615"
    release_dir = tmp_path / "release" / release_id
    release_dir.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / builder.DATASET_FILENAME).write_bytes(b"h5")
    (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(b"npz")
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    (release_dir / "us_source_coverage.json").write_text("{}")

    monkeypatch.setattr(
        builder,
        "_runtime_versions",
        lambda: {
            "python": "3.14.0",
            "populace-data": "0.1.0",
            "policyengine-core": "3.26.11",
            "policyengine-us": "1.729.0",
        },
    )
    monkeypatch.setattr(
        builder,
        "_git_output",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        builder,
        "diagnostics_payload",
        lambda result, target_registry: {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "fraction_within_10pct": 1.0,
            "target_surface": {"sha256": "b" * 64, "n_targets": 1},
        },
    )

    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=f"nation/cbo/individual_income_tax@{builder.PERIOD}",
                target=1.0,
                initial_estimate=1.0,
                final_estimate=1.0,
            ),
        ),
        initial_loss=2.0,
        final_loss=1.0,
    )

    class FakeRegistry:
        version = "registry-sha"

        def __len__(self):
            return 1

    registry = FakeRegistry()

    builder._build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=registry,
        dropped={"dropped_target_names": []},
    )

    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert manifest["data_package"] == {"name": "populace-data", "version": "0.1.0"}
    assert manifest["default_datasets"] == {"national": "populace_us_2024"}
    assert manifest["build"]["built_with_model_package"] == {
        "name": "policyengine-us",
        "version": "1.729.0",
    }
    for artifact in manifest["artifacts"].values():
        assert artifact["repo_id"] == builder.REPO_ID
        assert artifact["revision"] == release_id
        assert artifact["kind"]
        assert artifact["sha256"]


def test_export_frame_drops_formula_owned_columns(monkeypatch, small_frame) -> None:
    builder = _load_builder_module()

    class FakePolicyEngineUSEngine:
        def _engine_computed_columns(self, tables, *, period):
            assert period == builder.PERIOD
            assert "income" in tables["person"]
            return {"income"}

    monkeypatch.setattr(builder, "PolicyEngineUSEngine", FakePolicyEngineUSEngine)

    stripped = builder._strip_calibration_columns(
        small_frame,
        np.array([1000.0, 2000.0]),
    )

    assert "income" not in stripped.table("person")
    assert stripped.weights_for("household").kind == WeightKind.CALIBRATED


def test_export_frame_seeds_partnership_inputs_before_formula_drop(
    monkeypatch, small_frame
) -> None:
    builder = _load_builder_module()

    person = small_frame.table("person").copy()
    person["partnership_s_corp_income"] = np.asarray([100.0, -5.0, 0.0, 40.0])
    frame = Frame(
        {"person": person, "household": small_frame.table("household").copy()},
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
    )

    class FakePolicyEngineUSEngine:
        def _engine_computed_columns(self, tables, *, period):
            assert period == builder.PERIOD
            assert "partnership_income" in tables["person"]
            assert "s_corp_income" in tables["person"]
            return {"partnership_s_corp_income"}

    monkeypatch.setattr(builder, "PolicyEngineUSEngine", FakePolicyEngineUSEngine)

    stripped = builder._drop_formula_owned_columns(frame)

    assert "partnership_s_corp_income" not in stripped.table("person")
    assert np.array_equal(
        stripped.table("person")["partnership_income"].to_numpy(),
        np.asarray([100.0, -5.0, 0.0, 40.0]),
    )
    assert np.array_equal(
        stripped.table("person")["s_corp_income"].to_numpy(),
        np.zeros(4),
    )


def test_post_export_sanity_checks_full_target_surface(monkeypatch, tmp_path) -> None:
    builder = _load_builder_module()

    class FakeWeights:
        values = np.asarray([1.0])

    class FakeFrame:
        def weights_for(self, entity):
            assert entity == "household"
            return FakeWeights()

    class FakeTarget:
        entity = "household"
        row_name = f"nation/cbo/individual_income_tax@{builder.PERIOD}"

        def __init__(self):
            self.observed = 2_000_000_000_000.0

        def achieved_value(self, frame, weights):
            assert isinstance(frame, FakeFrame)
            assert np.array_equal(weights, np.asarray([1.0]))
            return self.observed

    target = FakeTarget()

    class FakeRegistry:
        def to_target_set(self):
            return (target,)

    monkeypatch.setattr(builder, "_load_frame", lambda path: f"loaded:{path}")
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda frame: (
            FakeFrame(),
            FakeRegistry(),
            {"dropped_target_names": []},
        ),
    )

    result = SimpleNamespace(
        diagnostics=(
            SimpleNamespace(
                name=f"nation/cbo/individual_income_tax@{builder.PERIOD}",
                final_estimate=2_000_000_000_000.0,
            ),
        )
    )

    builder._assert_export_matches_calibration(tmp_path / "candidate.h5", result)

    target.observed = 1_990_000_000_000.0
    try:
        builder._assert_export_matches_calibration(tmp_path / "candidate.h5", result)
    except RuntimeError as exc:
        assert "Post-export sanity failed" in str(exc)
        assert "nation/cbo/individual_income_tax@2024 exported value" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected post-export sanity failure.")


def test_post_export_sanity_rejects_dropped_export_targets(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "_load_frame", lambda path: object())
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda frame: (object(), object(), {"dropped_target_names": ["missing"]}),
    )

    try:
        builder._assert_export_matches_calibration(
            tmp_path / "candidate.h5", SimpleNamespace(diagnostics=())
        )
    except RuntimeError as exc:
        assert "1 fiscal targets were not materialized after export" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected dropped-target post-export sanity failure.")


def test_reviewed_exclusions_are_exact_for_fiscal_refresh() -> None:
    builder = _load_builder_module()

    exclusions = builder._reviewed_exclusions(builder.DIRECT_ACTIVE_ALIASES)

    assert tuple(exclusions) == builder.REVIEWED_EXCLUDED_ALIASES


def test_reviewed_exclusions_fail_when_hard_target_surface_changes(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        builder,
        "hard_target_package_aliases",
        lambda: (*builder.DIRECT_ACTIVE_ALIASES, "new-hard-target"),
    )

    try:
        builder._reviewed_exclusions(builder.DIRECT_ACTIVE_ALIASES)
    except RuntimeError as exc:
        assert "Reviewed hard-target exclusion list is stale" in str(exc)
        assert "new-hard-target" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected stale reviewed-exclusion failure.")


def test_fiscal_target_source_provenance_covers_active_families() -> None:
    builder = _load_builder_module()

    provenance = builder._fiscal_target_source_provenance()

    assert set(provenance) == {"cbo", "irs_soi", "jct", "state_income_tax"}
    assert provenance["cbo"]["target_count"] == 1
    assert provenance["jct"]["target_count"] == 5
    assert provenance["irs_soi"]["sources"]
    assert provenance["state_income_tax"]["reference_urls"]


def test_us_release_id_guard() -> None:
    builder = _load_builder_module()

    builder._assert_us_release_id("populace-us-2024-base-commit-20260615T000000Z")

    try:
        builder._assert_us_release_id("populace-uk-2024-base-commit-20260615T000000Z")
    except ValueError as exc:
        assert "must start with 'populace-us-'" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected non-US release id to fail.")
