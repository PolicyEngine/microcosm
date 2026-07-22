import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build import FitWeightRecord
from populace.build.us_runtime import (
    US_PUF_SUPPORT_FIT_NAME,
    clone_us_frame_for_puf_support,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_support_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_us_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
            "person_marital_unit_id": np.asarray([10000, 10000, 20000], dtype="int64"),
            "employment_income_before_lsr": np.asarray(
                [50_000, 20_000, 125_000], dtype="int64"
            ),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    strata = pd.Series(["asec_2024", "asec_2024", "asec_2023"], name="stratum")
    weights = {
        "household": Weights(
            values=np.asarray([100.0, 300.0]),
            kind=WeightKind.DESIGN,
        )
    }
    return Frame(tables, US_SCHEMA, weights, strata)


def _support_donor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filing_status_code": [1.0, 2.0, 4.0, 1.0],
            "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
            "employment_income_before_lsr": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_equivalence_h5_metadata_mode_disables_all_leaf_timestamps(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")  # pandas HDF backend patched by the metadata mode
    builder = _load_support_builder_module()
    paths = [tmp_path / "first.h5", tmp_path / "second.h5"]
    for path in paths:
        with builder._without_pytables_leaf_timestamps(True):
            with pd.HDFStore(path, mode="w") as store:
                store.put(
                    "person",
                    pd.DataFrame({"value": [1.0, 2.0]}),
                    format="table",
                    data_columns=True,
                )
        import tables

        with tables.open_file(path, mode="r") as h5:
            leaves = list(h5.walk_nodes("/", classname="Leaf"))
            assert leaves
            assert all(leaf._get_obj_timestamps().ctime == 0 for leaf in leaves)
    assert paths[0].read_bytes() == paths[1].read_bytes()


_SUPPORT_FIT_KWARGS = dict(
    predictors=(
        "puf_predictor_filing_status_code",
        "puf_predictor_tax_unit_person_count",
    ),
    person_outputs=("employment_income_before_lsr",),
    tax_unit_outputs=(),
    n_estimators=4,
    seed=0,
)


class TestBaseBuildWeightsAudit:
    """The base build records and enforces the PUF-support fit's weight kind.

    This is what makes the build-level weights audit (populace #300) real for the
    actual production tool: ``impute_and_audit_us_puf_support`` runs the fit,
    records its resolved weight kind, writes the audit into the build summary, and
    aborts the build on a failing audit. Engine-free — the imputation's
    formula-owned guard degrades to its static seed without ``policyengine_us``.
    """

    def test_base_build_records_design_weight_kind_in_the_summary(self) -> None:
        builder = _load_support_builder_module()

        _imputed, weights_audit = builder.impute_and_audit_us_puf_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            _support_donor(),
            **_SUPPORT_FIT_KWARGS,
        )

        assert weights_audit["passed"] is True
        assert weights_audit["failures"] == []
        assert weights_audit["details"]["resolved_weight_kinds"] == {
            US_PUF_SUPPORT_FIT_NAME: "design"
        }

    def test_base_build_summary_json_carries_the_audit(self) -> None:
        # The audit record must survive JSON serialization the summary uses.
        builder = _load_support_builder_module()

        _imputed, weights_audit = builder.impute_and_audit_us_puf_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            _support_donor(),
            **_SUPPORT_FIT_KWARGS,
        )
        round_tripped = json.loads(json.dumps({"weights_audit": weights_audit}))

        assert round_tripped["weights_audit"]["details"]["resolved_weight_kinds"] == {
            US_PUF_SUPPORT_FIT_NAME: "design"
        }

    def test_qrf_finalization_stage_records_tail_bound_diagnostics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        builder = _load_support_builder_module()
        record = {
            "output": "non_sch_d_capital_gains",
            "quantile": 0.999,
            "bound_value": 100.0,
            "clipped_row_count": 2,
            "clipped_mass_before": 1_903.0,
            "clipped_mass_after": 700.0,
        }

        def fake_finalize(frame, _checkpoint_dir, *, tail_bound_diagnostics=None):
            assert tail_bound_diagnostics is not None
            tail_bound_diagnostics.append(record)
            return frame, "design"

        monkeypatch.setattr(
            builder,
            "finalize_primary_puf_qrf_chain",
            fake_finalize,
        )
        _frame, metadata = builder._qrf_finalization_stage(
            SimpleNamespace(checkpoint_dir=tmp_path),
            _minimal_us_frame(),
        )

        assert metadata["puf_tax_detail_tail_bounds"] == [record]
        assert json.loads(json.dumps(metadata))["puf_tax_detail_tail_bounds"] == [
            record
        ]

    def test_base_build_aborts_when_the_audit_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Prove the wiring can actually fail the build: a fit that resolves
        # unweighted (simulated by recording a "none" record) makes the helper
        # raise SystemExit naming the fit, so a release cannot ship a silently
        # unweighted support fit.
        builder = _load_support_builder_module()

        def fake_impute(expanded, donor, *, fit_records=None, **_kwargs):
            if fit_records is not None:
                fit_records.append(FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, "none"))
            return expanded

        monkeypatch.setattr(builder, "impute_us_puf_tax_detail_support", fake_impute)

        with pytest.raises(SystemExit) as exc:
            builder.impute_and_audit_us_puf_support(
                clone_us_frame_for_puf_support(_minimal_us_frame()),
                _support_donor(),
                **_SUPPORT_FIT_KWARGS,
            )

        assert US_PUF_SUPPORT_FIT_NAME in str(exc.value)
        assert "unweighted" in str(exc.value)


def test_cd_vintage_crosswalk_requires_cd_assignment() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
                "--congressional-district-vintage-crosswalk",
                "crosswalk.csv",
            ]
        )

    assert exc.value.code == 2


def test_pooled_asec_mode_rejects_base_h5_at_parse_time() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--asec-h5",
                "2024=asec_2024.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
            ]
        )

    assert exc.value.code == 2


def test_weeks_unemployed_source_override_parses() -> None:
    builder = _load_support_builder_module()

    args = builder._parse_args(
        [
            "--base-h5",
            "base.h5",
            "--puf-h5",
            "puf.h5",
            "--asec-2023-weeks-unemployed-source",
            "asecpub23csv.zip",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    assert args.asec_2023_weeks_unemployed_source == Path("asecpub23csv.zip")


def test_stage_cli_defaults_to_legacy_all_without_checkpoints() -> None:
    builder = _load_support_builder_module()

    args = builder._parse_args(
        [
            "--base-h5",
            "base.h5",
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    assert args.stage == "all"
    assert args.checkpoint_dir is None


def test_reconciled_outer_pipeline_order_is_locked() -> None:
    builder = _load_support_builder_module()
    expected = (
        "source_construction",
        "pre_clone_enrichment",
        "clone_feature_extraction",
        "primary_qrf_chain",
        "qrf_finalization",
        "capital_gain_distributions",
        "qbi_reconciliation",
        "wic_post_clone",
        "housing_assistance",
        "prior_year_income_post_clone",
        "child_support_post_clone",
        "disability_benefits_post_clone",
        "workers_compensation_post_clone",
        "weeks_unemployed_post_clone",
        "childcare_post_clone",
        "adult_care_post_clone",
        "energy_subsidy_post_clone",
        "retirement_contributions_post_clone",
        "retirement_distributions_post_clone",
        "education_inputs_post_clone",
        "congressional_district_assignment",
        "block_ladder_assignment",
        "final_export",
    )

    assert builder.PIPELINE_STEPS == expected
    assert builder.OUTER_STAGE_PIPELINE.names == expected
    assert tuple(name for name, _boundaries in builder.STAGE_BOUNDARIES) == expected


def test_source_and_preclone_stages_round_trip_design_weight_kind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    weeks_path = tmp_path / "weeks.zip"
    args = builder._parse_args(
        [
            "--base-h5",
            str(tmp_path / "base.h5"),
            "--puf-h5",
            str(tmp_path / "puf.h5"),
            "--asec-2023-weeks-unemployed-source",
            str(weeks_path),
            "--out",
            str(tmp_path / "out"),
            "--without-block-ladder",
            "--stage",
            "source_construction",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
        ]
    )
    frame = _minimal_us_frame()
    monkeypatch.setattr(
        builder,
        "_load_base_frame_from_args",
        lambda _args: (frame, {"kind": "fixture"}),
    )
    weeks = pd.DataFrame({"PERIDNUM": [1]})
    weeks.attrs["source_audit"] = {"fixture": True}
    monkeypatch.setattr(
        builder,
        "load_asec_2023_weeks_unemployed_source",
        lambda _path: weeks,
    )
    identity_transforms = (
        "derive_us_cps_carried_inputs",
        "with_us_prior_year_income_inputs",
        "with_us_relationship_inputs",
        "with_us_medicare_take_up_input",
        "with_us_eligibility_inputs",
        "with_us_pregnancy_inputs",
        "with_us_wic_claim_input",
        "with_us_child_support_inputs",
        "with_us_disability_benefits",
        "with_us_workers_compensation",
        "with_us_weeks_unemployed",
        "with_us_childcare_inputs",
        "with_us_energy_subsidy_input",
        "with_us_retirement_contribution_inputs",
        "with_us_retirement_distribution_inputs",
        "with_us_immigration_inputs",
    )
    for name in identity_transforms:
        monkeypatch.setattr(builder, name, lambda value, **_kwargs: value)
    passing_gate = SimpleNamespace(passed=True, failures=(), details={})
    for name in (
        "us_relationship_inputs_signal_gate",
        "us_medicare_take_up_signal_gate",
        "us_housing_inputs_signal_gate",
        "us_eligibility_inputs_signal_gate",
        "us_pregnancy_signal_gate",
        "us_wic_claim_signal_gate",
    ):
        monkeypatch.setattr(builder, name, lambda _frame: passing_gate)
    monkeypatch.setattr(builder, "_sha256", lambda _path: "fixture-sha256")

    builder._run_outer_stage(args)
    args.stage = "pre_clone_enrichment"
    builder._run_outer_stage(args)

    runtime = builder.StageRuntime(
        args.checkpoint_dir,
        builder.OUTER_STAGE_PIPELINE,
        run_config=builder._stage_run_config(args),
    )
    loaded = runtime.load("pre_clone_enrichment")
    assert runtime.context.completed == (
        "source_construction",
        "pre_clone_enrichment",
    )
    assert loaded.frame.weights_for("household").kind is WeightKind.DESIGN


def test_weeks_post_clone_rejects_source_content_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    weeks_path = tmp_path / "weeks.zip"
    monkeypatch.setattr(builder, "_sha256", lambda _path: "changed-sha256")
    monkeypatch.setattr(
        builder,
        "load_asec_2023_weeks_unemployed_source",
        lambda _path: pytest.fail("drifted weeks source was loaded"),
    )

    with pytest.raises(SystemExit, match="changed between pre-clone and post-clone"):
        builder._post_qrf_frame_stage(
            "weeks_unemployed_post_clone",
            SimpleNamespace(seed=0, target_year=2024),
            _minimal_us_frame(),
            {
                "source_construction": {
                    "weeks_unemployed_source_path": str(weeks_path)
                },
                "pre_clone_enrichment": {
                    "weeks_unemployed_source": {"sha256": "original-sha256"}
                },
            },
        )


def test_outer_stage_resume_rejects_changed_builder_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    args = builder._parse_args(
        [
            "--base-h5",
            str(tmp_path / "base.h5"),
            "--puf-h5",
            str(tmp_path / "puf.h5"),
            "--out",
            str(tmp_path / "out"),
            "--without-block-ladder",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
        ]
    )
    monkeypatch.setattr(
        builder,
        "_builder_code_identity",
        lambda: {"source_sha256": "first"},
    )
    first_config = builder._stage_run_config(args)
    builder.StageRuntime(
        args.checkpoint_dir,
        builder.OUTER_STAGE_PIPELINE,
        run_config=first_config,
    )
    monkeypatch.setattr(
        builder,
        "_builder_code_identity",
        lambda: {"source_sha256": "second"},
    )

    with pytest.raises(ValueError, match="run_config differs"):
        builder.StageRuntime(
            args.checkpoint_dir,
            builder.OUTER_STAGE_PIPELINE,
            run_config=builder._stage_run_config(args),
        )


def test_monolith_equivalence_observer_writes_all_boundaries_and_raw_bits(
    tmp_path: Path,
) -> None:
    import h5py

    builder = _load_support_builder_module()
    frame = _minimal_us_frame()
    observer = builder._EquivalenceBoundaryObserver(tmp_path)
    raw = pd.DataFrame(
        {
            "first": np.asarray([0.0, -0.0], dtype=np.float64),
            "second": np.asarray([1.25, np.nan], dtype=np.float64),
        }
    )
    for stage in builder.PIPELINE_STEPS:
        if stage == "primary_qrf_chain":
            observer.observe_primary_qrf(frame, raw)
        else:
            observer.observe_frame(stage, frame)
    observer.assert_complete()

    boundary_files = sorted(tmp_path.glob("*.frame.h5"))
    assert len(boundary_files) == len(builder.PIPELINE_STEPS)
    raw_files = sorted((tmp_path / "primary_qrf" / "targets").glob("*.h5"))
    assert [path.name for path in raw_files] == [
        "000__first.h5",
        "001__second.h5",
    ]
    with h5py.File(raw_files[0], mode="r") as h5:
        np.testing.assert_array_equal(
            np.asarray(h5["raw_draw_bits"], dtype=np.uint64),
            raw["first"].to_numpy().view(np.uint64),
        )


@pytest.mark.parametrize("stage", ["a", "b", "c", "d", "all"])
def test_stage_cli_accepts_declared_stage_names(stage: str) -> None:
    builder = _load_support_builder_module()

    args = builder._parse_args(
        [
            "--base-h5",
            "base.h5",
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
            "--stage",
            stage,
            "--checkpoint-dir",
            "checkpoints",
        ]
    )

    assert args.stage == stage
    assert args.checkpoint_dir == Path("checkpoints")


def test_stage_cli_rejects_unknown_stage() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
                "--without-block-ladder",
                "--stage",
                "not-a-stage",
            ]
        )

    assert exc.value.code == 2


def test_named_stage_requires_checkpoint_directory() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
                "--without-block-ladder",
                "--stage",
                "a",
            ]
        )

    assert exc.value.code == 2


def test_main_legacy_all_has_no_checkpoint_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_support_builder_module()
    args = SimpleNamespace(stage="all", checkpoint_dir=None)
    calls: list[object] = []
    sentinel_frame = object()
    monkeypatch.setattr(builder, "_parse_args", lambda _argv: args)
    monkeypatch.setattr(
        builder,
        "_run_all",
        lambda actual_args: calls.append(actual_args) or sentinel_frame,
    )
    monkeypatch.setattr(
        builder,
        "_run_staged_all",
        lambda *_args, **_kwargs: pytest.fail("legacy path entered staged runtime"),
    )

    builder.main([])

    assert calls == [args]


def test_main_checkpointed_all_dispatches_fresh_process_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    args = SimpleNamespace(
        stage="all",
        checkpoint_dir=tmp_path,
        target_year=2024,
    )
    calls: list[object] = []
    monkeypatch.setattr(builder, "_parse_args", lambda _argv: args)
    monkeypatch.setattr(
        builder,
        "_run_all",
        lambda _args: pytest.fail("checkpointed path entered monolith"),
    )
    monkeypatch.setattr(
        builder,
        "_run_staged_all",
        lambda actual_args: calls.append(actual_args),
    )

    builder.main([])

    assert calls == [args]


def test_staged_all_defaults_fresh_children_to_fixed_python_hash_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_support_builder_module()
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    environment = builder._staged_subprocess_environment()

    assert environment["PYTHONHASHSEED"] == "0"


def test_completed_staged_all_reenters_final_child_for_crash_window_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    calls: list[tuple[list[str], dict[str, str]]] = []

    class CompletedRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            self.context = SimpleNamespace(completed=builder.PIPELINE_STEPS)

    monkeypatch.setattr(builder, "StageRuntime", CompletedRuntime)
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder,
        "_stage_cli_args",
        lambda _args, stage: ["--stage", stage],
    )

    def fake_run(command, *, check, env):
        assert not check
        calls.append((command, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    builder._run_staged_all(SimpleNamespace(checkpoint_dir=tmp_path))

    assert len(calls) == 1
    assert calls[0][0][-2:] == ["--stage", "final_export"]
    assert calls[0][1]["PYTHONHASHSEED"] == "0"


def test_completed_final_stage_repairs_missing_artifacts_and_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    checkpoint = tmp_path / "021_final_export.frame.h5"
    checkpoint.write_bytes(b"valid-checkpoint")
    output = tmp_path / "output.h5"
    summary = tmp_path / "summary.json"
    metadata = {
        "output_h5": str(output),
        "output_sha256": "output-digest",
        "summary_path": str(summary),
        "summary_sha256": "summary-digest",
    }
    frame = object()
    runtime = SimpleNamespace(
        load=lambda _stage: SimpleNamespace(frame=frame, path=checkpoint),
        metadata={"final_export": metadata},
    )
    calls: list[object] = []
    monkeypatch.setattr(
        builder,
        "_export_staged_result",
        lambda _args, frame, _metadata: (
            calls.append(frame)
            or {
                "output_sha256": "output-digest",
                "summary_sha256": "summary-digest",
            }
        ),
    )

    builder._repair_completed_final_stage(
        SimpleNamespace(checkpoint_dir=tmp_path), runtime
    )

    assert calls == [frame]
    alias = tmp_path / builder.ALL_STAGE_CHECKPOINT_FILENAME
    assert alias.is_file()
    assert alias.stat().st_ino == checkpoint.stat().st_ino


def test_direct_named_stage_requires_fixed_python_hash_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_support_builder_module()
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(
        builder,
        "_parse_args",
        lambda _argv: SimpleNamespace(
            stage="source_construction", checkpoint_dir=Path("checkpoints")
        ),
    )
    monkeypatch.setattr(
        builder,
        "_run_configured_stage",
        lambda _args: pytest.fail("unseeded named stage ran"),
    )

    with pytest.raises(SystemExit, match="require an explicit PYTHONHASHSEED"):
        builder.main([])


def test_pooled_asec_mode_loads_sources_with_manifest_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    captured = {}
    sentinel_frame = object()
    asec_2023 = tmp_path / "asec_2023.h5"
    asec_2024 = tmp_path / "asec_2024.h5"

    def fake_build_pooled_asec_unit_frame(sources, *, target_year):
        captured["sources"] = tuple(sources)
        captured["target_year"] = target_year
        return sentinel_frame, {
            "target_person_population": 123.0,
            "weighted_person_population": 123.0,
        }

    monkeypatch.setattr(
        builder,
        "build_pooled_asec_unit_frame",
        fake_build_pooled_asec_unit_frame,
    )
    monkeypatch.setattr(builder, "_sha256", lambda path: f"sha:{Path(path).name}")

    args = builder._parse_args(
        [
            "--asec-h5",
            f"2023={asec_2023}",
            "--asec-h5",
            f"2024={asec_2024}",
            "--target-year",
            "2024",
            "--asec-max-households",
            "50",
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    frame, metadata = builder._load_base_frame_from_args(args)

    assert frame is sentinel_frame
    assert captured["target_year"] == 2024
    assert [
        (source.year, source.path.name, source.max_households)
        for source in captured["sources"]
    ] == [
        (2023, "asec_2023.h5", 50),
        (2024, "asec_2024.h5", 50),
    ]
    assert metadata == {
        "kind": "pooled_asec",
        "target_year": 2024,
        "sources": [
            {
                "year": 2023,
                "path": str(asec_2023.resolve()),
                "sha256": "sha:asec_2023.h5",
                "share": None,
                "max_households": 50,
            },
            {
                "year": 2024,
                "path": str(asec_2024.resolve()),
                "sha256": "sha:asec_2024.h5",
                "share": None,
                "max_households": 50,
            },
        ],
        "support_spine_spec": None,
        "metadata": {
            "target_person_population": 123.0,
            "weighted_person_population": 123.0,
        },
    }


def test_support_spine_spec_resolves_relative_years_and_shares(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    captured = {}
    sentinel_frame = object()
    spec_path = tmp_path / "support_spine.json"
    asec_2024 = tmp_path / "asec_2024.h5"
    asec_2025 = tmp_path / "asec_2025.h5"
    spec_path.write_text(
        json.dumps(
            {
                "version": 1,
                "country": "us",
                "policy": "test support-spine spec",
                "support_spine": {
                    "stage": "asec_load",
                    "method": "pool_raw_asec_years",
                    "target_year_from_build_config": True,
                    "sources": [
                        {
                            "role": "prior",
                            "survey": "CPS ASEC",
                            "source": "https://www.census.gov/programs-surveys/cps.html",
                            "source_year_offset": -1,
                            "share": 0.25,
                        },
                        {
                            "role": "current",
                            "survey": "CPS ASEC",
                            "source": "https://www.census.gov/programs-surveys/cps.html",
                            "source_year_offset": 0,
                            "share": 0.75,
                        },
                    ],
                },
            }
        )
    )

    def fake_build_pooled_asec_unit_frame(sources, *, target_year):
        captured["sources"] = tuple(sources)
        captured["target_year"] = target_year
        return sentinel_frame, {"weighted_person_population": 1.0}

    monkeypatch.setattr(
        builder,
        "build_pooled_asec_unit_frame",
        fake_build_pooled_asec_unit_frame,
    )
    monkeypatch.setattr(builder, "_sha256", lambda path: f"sha:{Path(path).name}")

    args = builder._parse_args(
        [
            "--asec-h5",
            f"2024={asec_2024}",
            "--asec-h5",
            f"2025={asec_2025}",
            "--target-year",
            "2025",
            "--support-spine-spec",
            str(spec_path),
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    frame, metadata = builder._load_base_frame_from_args(args)

    assert frame is sentinel_frame
    assert captured["target_year"] == 2025
    assert [
        (source.year, source.path.name, source.share) for source in captured["sources"]
    ] == [
        (2024, "asec_2024.h5", 0.25),
        (2025, "asec_2025.h5", 0.75),
    ]
    assert metadata["support_spine_spec"]["path"] == str(spec_path.resolve())
    assert metadata["support_spine_spec"]["sources"][0]["resolved_year"] == 2024
    assert metadata["support_spine_spec"]["sources"][1]["resolved_year"] == 2025


def test_support_spine_spec_requires_mapped_asec_year(tmp_path: Path) -> None:
    builder = _load_support_builder_module()

    args = builder._parse_args(
        [
            "--asec-h5",
            f"2024={tmp_path / 'asec_2024.h5'}",
            "--target-year",
            "2025",
            "--support-spine-spec",
            "default",
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    with pytest.raises(ValueError, match="current_asec.*2025"):
        builder._load_base_frame_from_args(args)


def test_support_spine_spec_rejects_extra_asec_year_mapping(tmp_path: Path) -> None:
    builder = _load_support_builder_module()

    args = builder._parse_args(
        [
            "--asec-h5",
            f"2023={tmp_path / 'asec_2023.h5'}",
            "--asec-h5",
            f"2024={tmp_path / 'asec_2024.h5'}",
            "--asec-h5",
            f"2022={tmp_path / 'asec_2022.h5'}",
            "--target-year",
            "2024",
            "--support-spine-spec",
            "default",
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    with pytest.raises(ValueError, match="unused --asec-h5.*2022"):
        builder._load_base_frame_from_args(args)


def test_period_specific_output_filenames_keep_default_compatibility() -> None:
    builder = _load_support_builder_module()

    assert builder._dataset_filename(2024) == "base_populace_us_2024_puf_support.h5"
    assert (
        builder._summary_filename(2024)
        == "base_populace_us_2024_puf_support.summary.json"
    )
    assert builder._dataset_filename(2025) == "base_populace_us_2025_puf_support.h5"


def test_block_ladder_is_required_unless_explicitly_opted_out() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            ["--base-h5", "base.h5", "--puf-h5", "puf.h5", "--out", "out"]
        )

    assert exc.value.code == 2


def test_block_ladder_and_opt_out_are_contradictory() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
                "--block-ladder-artifact",
                "ladder.npz",
                "--without-block-ladder",
            ]
        )

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("failing_gate", "failure_message"),
    [
        ("child_support", "PUF child-support channel is default-only"),
        ("disability_benefits", "PUF disability-benefits channel is default-only"),
        ("wic_claim", "PUF WIC-claim channel is default-only"),
        ("educator_expense", "PUF educator-expense channel is default-only"),
        ("form_4952", "PUF Form 4952 channel is default-only"),
        ("salt_refund", "PUF SALT-refund channel is default-only"),
        ("energy_subsidy", "PUF energy-subsidy channel is default-only"),
        ("weeks_unemployed", "PUF weeks-unemployed channel is default-only"),
        (
            "retirement_distributions",
            "PUF retirement-distribution channel is default-only",
        ),
    ],
)
def test_main_runs_cps_only_inputs_before_clone_and_after_puf_then_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_gate: str,
    failure_message: str,
) -> None:
    builder = _load_support_builder_module()
    child_support_calls: list[tuple[object, int, int]] = []
    disability_benefits_calls: list[tuple[object, int, int]] = []
    weeks_unemployed_calls: list[tuple[object, int, int, object]] = []
    weeks_unemployed_gate_frames: list[object] = []
    weeks_unemployed_source_loads: list[Path] = []
    educator_expense_gate_frames: list[object] = []
    form_4952_gate_frames: list[object] = []
    salt_refund_gate_frames: list[object] = []
    energy_subsidy_gate_frames: list[object] = []
    retirement_distribution_calls: list[tuple[object, int, int]] = []
    retirement_distribution_gate_frames: list[object] = []
    prior_year_income_calls: list[tuple[object, int, int]] = []
    prior_year_income_gate_frames: list[object] = []
    prior_year_income_reconciliation_frames: list[object] = []
    capital_gain_distributions_calls: list[object] = []
    qbi_reconciliation_calls: list[object] = []

    monkeypatch.setattr(
        builder,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "out": tmp_path,
                "target_year": 2024,
                "seed": 7,
                "n_estimators": 4,
                "puf_h5": tmp_path / "puf.h5",
                "acs_h5": tmp_path / "acs.h5",
                "asec_2023_weeks_unemployed_source": (tmp_path / "asecpub23csv.zip"),
            },
        )(),
    )
    monkeypatch.setattr(
        builder,
        "_load_base_frame_from_args",
        lambda args: ("raw", {"kind": "fixture"}),
    )
    monkeypatch.setattr(builder, "derive_us_cps_carried_inputs", lambda frame: "cps")

    def fake_prior_year_income(frame, *, seed, time_period):
        prior_year_income_calls.append((frame, seed, time_period))
        return "prior-year-direct" if frame == "cps" else "prior-year-puf"

    monkeypatch.setattr(
        builder,
        "with_us_prior_year_income_inputs",
        fake_prior_year_income,
    )
    monkeypatch.setattr(
        builder,
        "with_us_relationship_inputs",
        lambda frame, *, seed, time_period: "relationship-inputs",
    )
    monkeypatch.setattr(
        builder,
        "us_relationship_inputs_signal_gate",
        lambda frame: type(
            "Gate", (), {"passed": True, "failures": (), "details": {}}
        )(),
    )
    monkeypatch.setattr(
        builder,
        "with_us_medicare_take_up_input",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "us_medicare_take_up_signal_gate",
        lambda frame: type(
            "Gate", (), {"passed": True, "failures": (), "details": {}}
        )(),
    )
    housing_gate_frames: list[object] = []

    def fake_housing_inputs_signal_gate(frame):
        housing_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": frame != "relationship-inputs",
                "failures": (
                    ("housing inputs absent",) if frame == "relationship-inputs" else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_housing_inputs_signal_gate",
        fake_housing_inputs_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "load_acs_2022_rent_donor",
        lambda path: "acs-rent-donor",
    )
    monkeypatch.setattr(
        builder,
        "with_us_housing_inputs",
        lambda frame, *, seed, time_period, acs_rent_donor: "housing-direct",
    )
    monkeypatch.setattr(
        builder,
        "with_us_eligibility_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_pregnancy_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_wic_claim_input",
        lambda frame, *, seed, time_period: frame,
    )
    for gate_name in (
        "us_eligibility_inputs_signal_gate",
        "us_pregnancy_signal_gate",
    ):
        monkeypatch.setattr(
            builder,
            gate_name,
            lambda frame: type(
                "Gate", (), {"passed": True, "failures": (), "details": {}}
            )(),
        )
    monkeypatch.setattr(
        builder,
        "us_wic_claim_signal_gate",
        lambda frame: type(
            "Gate",
            (),
            {
                "passed": not (
                    failing_gate == "wic_claim" and frame == "qbi-reconciled"
                ),
                "failures": (
                    ("PUF WIC-claim channel is default-only",)
                    if failing_gate == "wic_claim" and frame == "qbi-reconciled"
                    else ()
                ),
                "details": {},
            },
        )(),
    )

    def fake_child_support(frame, *, seed, time_period):
        child_support_calls.append((frame, seed, time_period))
        return (
            "child-support-direct" if frame == "housing-direct" else "child-support-puf"
        )

    monkeypatch.setattr(builder, "with_us_child_support_inputs", fake_child_support)

    def fake_disability_benefits(frame, *, seed, time_period):
        disability_benefits_calls.append((frame, seed, time_period))
        if frame == "child-support-direct":
            return "disability-benefits-direct"
        return "disability-benefits-puf"

    monkeypatch.setattr(
        builder,
        "with_us_disability_benefits",
        fake_disability_benefits,
    )

    def fake_load_weeks_unemployed_source(path, **_kwargs):
        weeks_unemployed_source_loads.append(Path(path))
        return "weeks-source"

    def fake_with_weeks_unemployed(
        frame,
        *,
        seed,
        time_period,
        asec_2023_source,
    ):
        weeks_unemployed_calls.append((frame, seed, time_period, asec_2023_source))
        return frame

    monkeypatch.setattr(
        builder,
        "load_asec_2023_weeks_unemployed_source",
        fake_load_weeks_unemployed_source,
    )
    monkeypatch.setattr(
        builder,
        "with_us_weeks_unemployed",
        fake_with_weeks_unemployed,
    )
    monkeypatch.setattr(
        builder,
        "with_us_workers_compensation",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_childcare_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_adult_care_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_energy_subsidy_input",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_retirement_contribution_inputs",
        lambda frame, *, seed, time_period: frame,
    )

    def fake_retirement_distributions(frame, *, seed, time_period):
        retirement_distribution_calls.append((frame, seed, time_period))
        if frame == "disability-benefits-direct":
            return "retirement-distributions-direct"
        return "retirement-distributions-puf"

    monkeypatch.setattr(
        builder,
        "with_us_retirement_distribution_inputs",
        fake_retirement_distributions,
    )
    monkeypatch.setattr(
        builder,
        "with_us_immigration_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "clone_us_frame_for_puf_support",
        lambda frame: "expanded",
    )
    monkeypatch.setattr(builder, "_read_h5_arrays", lambda path: {})
    monkeypatch.setattr(builder, "puf_tax_unit_donor_from_arrays", lambda arrays: None)
    monkeypatch.setattr(
        builder,
        "impute_and_audit_us_puf_support",
        lambda expanded, donor, **kwargs: ("puf-imputed", {"passed": True}),
    )

    def fake_capital_gain_distributions(args, frame):
        capital_gain_distributions_calls.append(frame)
        return "capital-gain-distributions", {}

    monkeypatch.setattr(
        builder,
        "_capital_gain_distributions_stage",
        fake_capital_gain_distributions,
    )

    def fake_qbi_reconciliation(frame):
        qbi_reconciliation_calls.append(frame)
        return "qbi-reconciled"

    monkeypatch.setattr(
        builder,
        "with_us_qbi_input_reconciliation",
        fake_qbi_reconciliation,
    )
    monkeypatch.setattr(
        builder,
        "impute_us_housing_assistance_to_puf_support",
        lambda frame, *, seed: "housing-puf",
    )
    passing_gate = type(
        "Gate",
        (),
        {"passed": True, "failures": (), "details": {}},
    )()

    def fake_prior_year_income_gate(frame):
        prior_year_income_gate_frames.append(frame)
        return passing_gate

    monkeypatch.setattr(
        builder,
        "us_prior_year_income_signal_gate",
        fake_prior_year_income_gate,
    )

    def fake_prior_year_income_reconciliation_gate(frame):
        prior_year_income_reconciliation_frames.append(frame)
        return passing_gate

    monkeypatch.setattr(
        builder,
        "us_prior_year_income_source_reconciliation_gate",
        fake_prior_year_income_reconciliation_gate,
    )
    monkeypatch.setattr(
        builder, "us_qbi_inputs_signal_gate", lambda frame: passing_gate
    )
    monkeypatch.setattr(
        builder,
        "us_farm_business_income_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_domestic_production_ald_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_child_support_signal_gate",
        lambda frame: type(
            "Gate",
            (),
            {
                "passed": failing_gate != "child_support",
                "failures": (
                    ("PUF child-support channel is default-only",)
                    if failing_gate == "child_support"
                    else ()
                ),
                "details": {},
            },
        )(),
    )
    monkeypatch.setattr(
        builder,
        "us_disability_benefits_signal_gate",
        lambda frame: type(
            "Gate",
            (),
            {
                "passed": failing_gate != "disability_benefits",
                "failures": (
                    ("PUF disability-benefits channel is default-only",)
                    if failing_gate == "disability_benefits"
                    else ()
                ),
                "details": {},
            },
        )(),
    )
    monkeypatch.setattr(
        builder,
        "us_workers_compensation_signal_gate",
        lambda frame: type(
            "Gate", (), {"passed": True, "failures": (), "details": {}}
        )(),
    )

    def fake_weeks_unemployed_signal_gate(frame):
        weeks_unemployed_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "weeks_unemployed",
                "failures": (
                    ("PUF weeks-unemployed channel is default-only",)
                    if failing_gate == "weeks_unemployed"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_weeks_unemployed_signal_gate",
        fake_weeks_unemployed_signal_gate,
    )

    def fake_educator_expense_signal_gate(frame):
        educator_expense_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "educator_expense",
                "failures": (
                    ("PUF educator-expense channel is default-only",)
                    if failing_gate == "educator_expense"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_educator_expense_signal_gate",
        fake_educator_expense_signal_gate,
    )

    def fake_form_4952_election_signal_gate(frame):
        form_4952_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "form_4952",
                "failures": (
                    ("PUF Form 4952 channel is default-only",)
                    if failing_gate == "form_4952"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_form_4952_election_signal_gate",
        fake_form_4952_election_signal_gate,
    )

    def fake_salt_refund_income_signal_gate(frame):
        salt_refund_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "salt_refund",
                "failures": (
                    ("PUF SALT-refund channel is default-only",)
                    if failing_gate == "salt_refund"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_salt_refund_income_signal_gate",
        fake_salt_refund_income_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_capital_gain_details_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_childcare_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_adult_care_signal_gate",
        lambda frame: passing_gate,
    )

    def fake_energy_subsidy_signal_gate(frame):
        energy_subsidy_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "energy_subsidy",
                "failures": (
                    ("PUF energy-subsidy channel is default-only",)
                    if failing_gate == "energy_subsidy"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_energy_subsidy_signal_gate",
        fake_energy_subsidy_signal_gate,
    )
    monkeypatch.setattr(builder, "us_alimony_signal_gate", lambda frame: passing_gate)
    monkeypatch.setattr(
        builder,
        "us_casualty_loss_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_misc_itemized_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_retirement_contributions_signal_gate",
        lambda frame: passing_gate,
    )

    def fake_retirement_distributions_signal_gate(frame):
        retirement_distribution_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "retirement_distributions",
                "failures": (
                    ("PUF retirement-distribution channel is default-only",)
                    if failing_gate == "retirement_distributions"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_retirement_distributions_signal_gate",
        fake_retirement_distributions_signal_gate,
    )

    with pytest.raises(SystemExit, match=failure_message):
        builder.main()

    assert capital_gain_distributions_calls == ["puf-imputed"]
    assert qbi_reconciliation_calls == ["capital-gain-distributions"]
    if failing_gate == "wic_claim":
        assert child_support_calls == [("housing-direct", 7, 2024)]
        assert prior_year_income_calls == [("cps", 7, 2024)]
        assert prior_year_income_gate_frames == []
        assert prior_year_income_reconciliation_frames == []
        assert housing_gate_frames == ["relationship-inputs", "housing-direct"]
    else:
        assert child_support_calls == [
            ("housing-direct", 7, 2024),
            ("prior-year-puf", 7, 2024),
        ]
        assert prior_year_income_calls == [
            ("cps", 7, 2024),
            ("housing-puf", 7, 2024),
        ]
        assert prior_year_income_gate_frames == ["prior-year-puf"]
        assert prior_year_income_reconciliation_frames == ["prior-year-puf"]
        assert housing_gate_frames == [
            "relationship-inputs",
            "housing-direct",
            "prior-year-puf",
        ]
    expected_disability_calls = [("child-support-direct", 7, 2024)]
    if failing_gate not in {"child_support", "wic_claim"}:
        expected_disability_calls.append(("child-support-puf", 7, 2024))
    assert disability_benefits_calls == expected_disability_calls
    expected_weeks_calls = [("disability-benefits-direct", 7, 2024, "weeks-source")]
    if failing_gate not in {"wic_claim", "child_support", "disability_benefits"}:
        expected_weeks_calls.append(
            ("disability-benefits-puf", 7, 2024, "weeks-source")
        )
    assert weeks_unemployed_calls == expected_weeks_calls
    assert weeks_unemployed_source_loads == [tmp_path / "asecpub23csv.zip"]
    assert weeks_unemployed_gate_frames == (
        ["disability-benefits-puf"]
        if failing_gate not in {"wic_claim", "child_support", "disability_benefits"}
        else []
    )
    assert educator_expense_gate_frames == (
        ["disability-benefits-puf"]
        if failing_gate
        in {
            "educator_expense",
            "form_4952",
            "salt_refund",
            "energy_subsidy",
            "retirement_distributions",
        }
        else []
    )
    assert form_4952_gate_frames == (
        ["disability-benefits-puf"]
        if failing_gate
        in {"form_4952", "salt_refund", "energy_subsidy", "retirement_distributions"}
        else []
    )
    assert salt_refund_gate_frames == (
        ["disability-benefits-puf"]
        if failing_gate in {"salt_refund", "energy_subsidy", "retirement_distributions"}
        else []
    )
    assert energy_subsidy_gate_frames == (
        ["disability-benefits-puf"]
        if failing_gate in {"energy_subsidy", "retirement_distributions"}
        else []
    )
    expected_retirement_distribution_calls = [("disability-benefits-direct", 7, 2024)]
    if failing_gate == "retirement_distributions":
        expected_retirement_distribution_calls.append(
            ("disability-benefits-puf", 7, 2024)
        )
    assert retirement_distribution_calls == expected_retirement_distribution_calls
    assert retirement_distribution_gate_frames == (
        ["retirement-distributions-puf"]
        if failing_gate == "retirement_distributions"
        else []
    )


def test_main_summary_records_retirement_distribution_gate() -> None:
    builder = _load_support_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"retirement_distributions_signal": {' in source
    assert '"passed": retirement_distributions_gate.passed' in source
    assert '"failures": list(retirement_distributions_gate.failures)' in source
    assert '"details": dict(retirement_distributions_gate.details)' in source


def test_main_summary_records_weeks_unemployed_gate() -> None:
    builder = _load_support_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"weeks_unemployed_signal": {' in source
    assert '"passed": weeks_unemployed_gate.passed' in source
    assert '"failures": list(weeks_unemployed_gate.failures)' in source
    assert '"details": dict(weeks_unemployed_gate.details)' in source


def test_main_summary_records_salt_refund_income_gate() -> None:
    builder = _load_support_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"salt_refund_income_signal": {' in source
    assert '"passed": salt_refund_income_gate.passed' in source
    assert '"failures": list(salt_refund_income_gate.failures)' in source
    assert '"details": dict(salt_refund_income_gate.details)' in source


def test_main_summary_records_energy_subsidy_gate() -> None:
    builder = _load_support_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"energy_subsidy_signal": {' in source
    assert '"passed": energy_subsidy_gate.passed' in source
    assert '"failures": list(energy_subsidy_gate.failures)' in source
    assert '"details": dict(energy_subsidy_gate.details)' in source


def test_main_summary_records_prior_year_income_gate() -> None:
    builder = _load_support_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"prior_year_income_signal": {' in source
    assert '"passed": prior_year_income_gate.passed' in source
    assert '"failures": list(prior_year_income_gate.failures)' in source
    assert '"details": dict(prior_year_income_gate.details)' in source
    assert '"prior_year_income_source_reconciliation": {' in source
    assert '"passed": prior_year_income_reconciliation_gate.passed' in source


def test_stage_cli_round_trips_the_locked_run_config(tmp_path: Path) -> None:
    """Child-stage argv reconstruction preserves the locked run identity.

    ``--stage all`` locks ``_stage_run_config(parent_args)`` into the
    checkpoint context, then spawns each stage child with
    ``_stage_cli_args``-reconstructed argv; the child recomputes the config
    and refuses on any difference. base-r4 died in six seconds because a
    newly added input (``--asec-education-source``) entered the run config
    without entering the reconstruction. This round-trip pins the whole
    class: every parsed input that the run config records must survive
    reconstruction bit for bit.
    """

    builder = _load_support_builder_module()
    puf = tmp_path / "puf.h5"
    puf.write_bytes(b"puf")
    sidecar = tmp_path / "asecpub24csv.zip"
    sidecar.write_bytes(b"zip")
    argv = [
        "--asec-h5",
        f"2023={tmp_path / 'census_cps_2023.h5'}",
        "--asec-h5",
        f"2024={tmp_path / 'census_cps_2024.h5'}",
        "--puf-h5",
        str(puf),
        "--asec-education-source",
        f"2023={sidecar}",
        "--target-year",
        "2024",
        "--out",
        str(tmp_path / "out"),
        "--seed",
        "0",
        "--stage",
        "all",
        "--checkpoint-dir",
        str(tmp_path / "ck"),
        "--without-block-ladder",
    ]
    parent = builder._parse_args(argv)
    child_argv = builder._stage_cli_args(parent, "source_construction")
    child = builder._parse_args(child_argv)

    parent_config = builder._stage_run_config(parent)
    child_config = builder._stage_run_config(child)
    assert child_config == parent_config
