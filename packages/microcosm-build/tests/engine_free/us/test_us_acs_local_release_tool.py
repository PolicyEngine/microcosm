"""Unit tests for the pure parts of tools/build_us_acs_local_release.py."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.data import stored_inputs
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

# Tests that write real H5 bytes go through pandas' HDFStore, which needs
# pytables; the base wheel gate installs the shards without it.
requires_pytables = pytest.mark.skipif(
    importlib.util.find_spec("tables") is None,
    reason="requires pytables (the build environment)",
)


#: The stored-input contract's real entry points, kept before the autouse
#: fixture below replaces them, for the tests that pin the contract itself.
_REAL_REQUIRE_H5_STORED_INPUTS = stored_inputs.require_h5_stored_inputs
#: A policyengine-us stand-in for the package stage's stored-input contract.
_PACKAGE_ENGINE = stored_inputs.CertifiedEngine(
    label="policyengine-us 2.2.1",
    variables=frozenset(
        {
            "person_id",
            "person_household_id",
            "person_tax_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_marital_unit_id",
            "household_id",
            "tax_unit_id",
            "spm_unit_id",
            "family_id",
            "marital_unit_id",
            "household_weight",
            "weekly_hours_worked_before_lsr",
            "hours_worked_last_week",
            "takes_up_wic_if_eligible",
        }
    ),
)


@pytest.fixture(autouse=True)
def _stored_input_contract_passes(monkeypatch):
    """The package stage checks the calibrated H5 against the installed
    policyengine-us (microcosm#1026). The engine-free lane has none, and most
    package tests here hash placeholder bytes rather than an H5, so the
    contract passes by default. The tests that pin it restore the real reader
    against :data:`_PACKAGE_ENGINE`."""

    monkeypatch.setattr(stored_inputs, "installed_us_engine", lambda: _PACKAGE_ENGINE)
    monkeypatch.setattr(
        stored_inputs,
        "require_h5_stored_inputs",
        lambda path, *, engine: {
            "register_sha256": stored_inputs.register_sha256(),
            "registered_non_variables": [],
        },
    )


def _load_tool_module():
    root = _TEST_PATHS.repository
    path = root / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_staging_builder_module():
    root = _TEST_PATHS.repository
    path = root / "tools" / "build_us_acs_multispine_base.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_multispine_base",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_household_chunks_refuse_population_aggregates(monkeypatch, tmp_path) -> None:
    module = _load_tool_module()
    import build_us_fiscal_refresh_release as release

    fixture_spec = importlib.util.spec_from_file_location(
        "batched_materialization_fixtures",
        Path(__file__).with_name("test_us_batched_target_materialization.py"),
    )
    fixtures = importlib.util.module_from_spec(fixture_spec)
    assert fixture_spec.loader is not None
    fixture_spec.loader.exec_module(fixtures)
    aggregate = "medicaid_slcsp_state_denominator"
    ledger = fixtures._install_fake_engine(
        release,
        monkeypatch,
        reform_specs=(),
        aggregate_reads={"aggregate_probe": (aggregate,)},
    )
    monkeypatch.setattr(module, "project_input_only", lambda frame, **kw: (frame, {}))
    monkeypatch.setattr(module, "fill_reviewed_nulls", lambda *args, **kw: None)
    specs = (fixtures._variable("aggregate_total", base_variable="aggregate_probe"),)

    with pytest.raises(
        ValueError,
        match=rf"household batch 1/1 computed .*{aggregate}@2024",
    ):
        module.materialize_chunked(
            fixtures._nested_frame(),
            specs,
            hh_chunk=2,
            batch=2,
            summary_path=tmp_path / "summary.json",
        )
    assert len(ledger.simulations) == 1
    assert ledger.simulations[0].dataset is None


def test_spine_composition_reports_per_spine_weight_and_size() -> None:
    module = _load_tool_module()
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_spine": ["asec_puf", "asec_puf", "acs_2024_1yr"],
        }
    )
    persons = pd.DataFrame({"person_household_id": [1, 2, 2, 3, 3, 3]})
    weights = np.asarray([10.0, 30.0, 60.0])

    composition = module.spine_composition(households, persons, weights)

    asec = composition["asec_puf"]
    assert asec["households"] == 2
    assert asec["household_weight"] == pytest.approx(40.0)
    assert asec["household_weight_share"] == pytest.approx(0.4)
    # person weight: 10*1 + 30*2 = 70 -> 70/40 persons per household.
    assert asec["person_weight"] == pytest.approx(70.0)
    assert asec["implied_persons_per_household"] == pytest.approx(1.75)
    acs = composition["acs_2024_1yr"]
    assert acs["implied_persons_per_household"] == pytest.approx(3.0)
    overall = composition["_all"]
    assert overall["households"] == 3
    assert overall["effective_sample_size"] == pytest.approx(
        (10 + 30 + 60) ** 2 / (10**2 + 30**2 + 60**2)
    )


def test_finalize_reviewed_limitations_carries_staging_and_dedupes() -> None:
    module = _load_tool_module()
    staging_summary = {
        "base": {
            "donor_release": {
                "release_id": "populace-us-2024-buildo-sparse-rmloss100-x",
            }
        },
        "reviewed_limitations": [
            {"id": "acs_group_quarters_housing_universe", "status": "reviewed"},
            {
                "id": "cd_population_marginal_vintage_2020",
                "status": "stale_staging_copy",
            },
        ],
    }
    diagnostics = {
        "effective_sample_size": 4853.9,
        "ess_fraction": 0.003,
        "households": 1_600_000,
    }
    spine_qa = {
        "per_spine": {
            "acs_2024_1yr": {"ssi_incidence": 0.0259},
            "asec_puf": {"ssi_incidence": 0.0185},
        }
    }

    limitations = module.finalize_reviewed_limitations(
        staging_summary, diagnostics, spine_qa
    )

    by_id = {item["id"]: item for item in limitations}
    # Staging entries carried; finalize's own entry wins the id collision.
    assert "acs_group_quarters_housing_universe" in by_id
    assert by_id["cd_population_marginal_vintage_2020"]["status"] == (
        "reviewed_vintage"
    )
    # Lineage entries present, none blocking.
    for required in (
        "ssi_aged_band_collapse_inherited",
        "miscellaneous_income_loss_side_donor_defect",
        "tips_return_count_carrier_deficit_inherited",
        "low_effective_sample_size_lambda_zero",
        "donor_sparse_selection_training_set",
        "mixed_sub_puma_column_coverage",
        "acs_immigration_status_method",
        "acs_work_disability_inputs",
        "acs_local_ssi_medicaid_take_up",
    ):
        assert required in by_id, required
        assert by_id[required]["calibration_blocker"] is False
    # microcosm#1021: native ACS disability is a reviewed method; weeks worked
    # is staged only, and is_veteran is documented, not exported.
    work_disability = by_id["acs_work_disability_inputs"]
    assert work_disability["status"] == "reviewed_modeling_decision"
    assert set(work_disability["columns"]) == {"is_disabled", "is_blind"}
    assert "acs_local_work_disability_signal" in work_disability["treatment"]
    for fragment in ("SSIP > 0 under age 65", "policyengine-us#9660", "MIL == 2"):
        assert fragment in work_disability["reason"]
    # microcosm#1022: the engine-free fills are a reviewed method, and
    # Medicare is no longer listed as an engine-default take-up flag.
    fills = by_id["acs_engine_free_default_fills"]
    assert fills["status"] == "reviewed_modeling_decision"
    assert fills["calibration_blocker"] is False
    assert set(fills["columns"]) == {
        "is_snap_abawd_discretionary_exempt",
        "receives_housing_assistance",
        "takes_up_medicare_if_eligible",
    }
    assert "acs_local_take_up_signal" in fills["treatment"]
    for fragment in (
        "SERIALNO:SPORDER",
        "microcosm#975",
        "HINS3 == 1",
        "cap-based proxy, not an observed exemption assignment",
        "upper-bound propensity",
    ):
        assert fragment in fills["reason"]
    assert "Blank and invalid HINS3 counts" in fills["treatment"]
    # microcosm#1022: the ACS SSI disability criteria are a reviewed method,
    # and SSI take-up is assigned against them.
    ssi = by_id["acs_local_ssi_disability_criteria"]
    assert ssi["status"] == "reviewed_modeling_decision"
    assert ssi["calibration_blocker"] is False
    assert ssi["columns"] == ["meets_ssi_disability_criteria"]
    assert "acs_local_ssi_disability_signal" in ssi["treatment"]
    for fragment in ("PEDIS*", "SERIALNO:SPORDER", "acs_local_ssi_medicaid_take_up"):
        assert fragment in ssi["reason"]
    defaults = by_id["acs_take_up_engine_defaults"]["reason"]
    assert "EITC, ACA, Head Start" in defaults
    assert "Medicare take-up is native ACS HINS3" in defaults
    assert "acs_local_ssi_medicaid_take_up" in defaults
    for stale in ("Medicaid, SSI", "SSI, Head Start"):
        assert stale not in defaults
    # microcosm#1022: ACS SSI and Medicaid take-up are a reviewed method.
    take_up = by_id["acs_local_ssi_medicaid_take_up"]
    assert take_up["status"] == "reviewed_modeling_decision"
    assert take_up["calibration_blocker"] is False
    assert take_up["columns"] == list(_SSI_MEDICAID)
    assert "acs_local_ssi_medicaid_take_up_signal" in take_up["treatment"]
    for fragment in (
        "HINS4 == 1",
        "SSIP",
        "after the donor rows' pooled recipients",
        "count minus contribution",
        "probability residual / anchored eligible weight",
        "SERIALNO:SPORDER",
        "under-18",
    ):
        assert fragment in take_up["reason"]
    # microcosm#1060 review: the share-scaled targets are gone.
    assert "share of the frame" not in take_up["reason"]
    assert "the donor contribution" in take_up["treatment"]
    # microcosm#1020: the ACS immigration inputs are a reviewed method, not
    # an engine-default gap.
    immigration = by_id["acs_immigration_status_method"]
    assert immigration["status"] == "reviewed_modeling_decision"
    assert set(immigration["columns"]) == {
        "immigration_status_str",
        "ssn_card_type",
        "years_since_us_entry",
    }
    assert "acs_local_immigration_signal" in immigration["treatment"]
    ssi = by_id["ssi_aged_band_collapse_inherited"]
    assert ssi["measured_spine_ssi"] == {
        "acs_2024_1yr": 0.0259,
        "asec_puf": 0.0185,
    }
    donor = by_id["donor_sparse_selection_training_set"]
    assert "populace-us-2024-buildo-sparse-rmloss100-x" in donor["reason"]
    # Ids are unique after dedupe.
    assert len(by_id) == len(limitations)


def test_parse_args_enforces_stage_requirements(tmp_path: Path) -> None:
    module = _load_tool_module()
    base = [
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
    ]

    with pytest.raises(SystemExit):
        module._parse_args(["--stage", "materialize", *base])
    with pytest.raises(SystemExit):
        module._parse_args(["--stage", "calibrate", *base])
    with pytest.raises(SystemExit):
        module._parse_args(
            ["--stage", "package", *base, "--out-h5", str(tmp_path / "o.h5")]
        )

    args = module._parse_args(
        [
            "--stage",
            "all",
            *base,
            "--feed",
            str(tmp_path / "facts.jsonl"),
            "--out-h5",
            str(tmp_path / "out.h5"),
            "--out",
            str(tmp_path / "release"),
        ]
    )
    assert args.stages == ["materialize", "calibrate", "qa", "finalize", "package"]
    assert args.out_summary == tmp_path / "out.summary.json"
    assert args.gate_report == tmp_path / "ckpt" / "gate_summary.json"


def test_release_id_prefix_and_manifest_constants() -> None:
    module = _load_tool_module()
    assert module.RELEASE_ID_PREFIX == "populace-us-2024-buildo-acs-local"
    assert module.RELEASE_NAMESPACE == "buildo_acs_local"
    assert module.ARTIFACT_FILENAME == "populace_us_2024_acs_local.h5"
    assert module.HF_REPO_ID == "policyengine/populace-us"


def test_documented_staging_recipe_matches_legacy_and_release_parsers(
    tmp_path: Path,
) -> None:
    release = _load_tool_module()
    staging_builder = _load_staging_builder_module()
    recipe = shlex.split(release.LEGACY_STAGING_REFRESH_RECIPE)

    assert recipe[:3] == [
        "uv",
        "run",
        "tools/build_us_acs_multispine_base.py",
    ]
    staging_args = staging_builder._legacy._parse_args(recipe[3:])
    staging_summary = staging_args.summary or staging_args.out_h5.with_suffix(
        ".summary.json"
    )
    release_args = release._parse_args(
        [
            "--stage",
            "materialize",
            "--staging-h5",
            str(staging_args.out_h5),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--feed",
            str(tmp_path / "facts.jsonl"),
        ]
    )

    assert release._staging_summary_path(release_args) == staging_summary


def test_do_finalize_requires_calibration_summary(tmp_path: Path) -> None:
    module = _load_tool_module()
    staging = tmp_path / "staging.h5"
    staging.touch()
    (tmp_path / "staging.summary.json").write_text(json.dumps({}))
    args = module._parse_args(
        [
            "--stage",
            "finalize",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--out-h5",
            str(tmp_path / "out.h5"),
        ]
    )

    with pytest.raises(SystemExit, match="No calibration summary"):
        module.do_finalize(args)


def test_local_hours_gate_refuses_missing_source_audit() -> None:
    module = _load_tool_module()
    with pytest.raises(SystemExit, match="staging input-null audit"):
        module._require_local_hours(None, {})


def test_local_hours_failure_propagates_to_release_boundary(monkeypatch) -> None:
    from microcosm.build.gates import GateResult

    module = _load_tool_module()
    seen = []

    def failed_gate(frame, *, source_null_audit):
        seen.append((frame, source_null_audit))
        return GateResult(
            name="acs_local_hours_signal",
            passed=False,
            failures=("acs_2024_1yr: unresolved hours",),
        )

    monkeypatch.setattr(module, "acs_local_hours_signal_gate", failed_gate)
    marker = object()
    audit = [{"entity": "person", "column": "weekly_hours_worked_before_lsr"}]
    with pytest.raises(SystemExit, match="acs_2024_1yr: unresolved hours"):
        module._require_local_hours(marker, {"reviewed_engine_input_nulls": audit})
    assert seen == [(marker, audit)]


def test_reviewed_null_fill_refuses_to_default_fill_take_up(tmp_path) -> None:
    """microcosm#1019: a missing take-up cell is refused even when registered,
    because the engine default ``True`` is universal take-up."""

    from microcosm.frame import Frame

    module = _load_tool_module()
    frame = _plausible_hours_frame()
    spm_unit = frame.table("spm_unit").assign(
        takes_up_snap_if_eligible=[True, False] * 3 + [np.nan] * 2
    )
    frame = Frame(
        {
            entity: spm_unit if entity == "spm_unit" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    summary = tmp_path / "staging.summary.json"
    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {
                        "entity": "spm_unit",
                        "column": "takes_up_snap_if_eligible",
                        "missing_rows": 2,
                    }
                ]
            }
        )
    )
    with pytest.raises(
        module.DeniedDefaultFillError,
        match=r"spm_unit\.takes_up_snap_if_eligible \(2 null rows\)",
    ):
        module.fill_reviewed_nulls(frame, summary)
    assert spm_unit["takes_up_snap_if_eligible"].isna().sum() == 2


#: A current (post-#1022) materialize receipt, as run_identity.json records it.
_TAKE_UP_RECEIPT = {
    "seed": 0,
    "assigned_sha256": "a" * 64,
    "engine_free_fills": {"issue": "microcosm#1022"},
}


def test_take_up_consumers_refuse_a_checkpoint_without_the_assignment() -> None:
    module = _load_tool_module()
    for identity in (
        {},
        {"acs_local_take_up": {**_TAKE_UP_RECEIPT, "seed": "0"}},
        {"acs_local_take_up": {"seed": 0, "engine_free_fills": {}}},
    ):
        with pytest.raises(SystemExit, match="Re-run --stage materialize"):
            module._recorded_take_up(identity)
    recorded = {**_TAKE_UP_RECEIPT, "seed": 3}
    assert module._recorded_take_up({"acs_local_take_up": recorded}) == recorded


@pytest.mark.parametrize(
    "fills", [None, {}, {"issue": "microcosm#1019"}], ids=["absent", "empty", "stale"]
)
def test_take_up_consumers_refuse_a_pre_1022_checkpoint(fills) -> None:
    """A checkpoint materialized before #1022 calibrated against ACS rows
    whose three engine-free inputs sat at the engine default."""

    module = _load_tool_module()
    receipt = {"seed": 0, "assigned_sha256": "a" * 64}
    if fills is not None:
        receipt["engine_free_fills"] = fills
    with pytest.raises(SystemExit, match=r"microcosm#1022.*Re-run --stage materialize"):
        module._recorded_take_up({"acs_local_take_up": receipt})


_ENGINE_FREE_FILLS = (
    ("person", "is_snap_abawd_discretionary_exempt"),
    ("spm_unit", "receives_housing_assistance"),
    ("person", "takes_up_medicare_if_eligible"),
)


def test_engine_free_fills_are_never_default_filled() -> None:
    """microcosm#1022: each engine default biases SNAP on ACS rows."""

    module = _load_tool_module()
    for key in _ENGINE_FREE_FILLS:
        assert key in module.NEVER_DEFAULT_FILLED
        assert "microcosm#1022" in module._NEVER_DEFAULT_FILLED_REASONS[key]


@pytest.mark.parametrize("entity,column", _ENGINE_FREE_FILLS)
def test_reviewed_null_fill_refuses_to_default_fill_engine_free_inputs(
    tmp_path, entity, column
) -> None:
    from microcosm.frame import Frame

    module = _load_tool_module()
    frame = _plausible_hours_frame()
    table = frame.table(entity).assign(**{column: [True, False] * 3 + [np.nan] * 2})
    frame = Frame(
        {
            name: table if name == entity else frame.table(name)
            for name in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    summary = tmp_path / "staging.summary.json"
    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {"entity": entity, "column": column, "missing_rows": 2}
                ]
            }
        )
    )
    with pytest.raises(
        module.DeniedDefaultFillError, match=rf"{entity}\.{column} \(2 null rows\)"
    ):
        module.fill_reviewed_nulls(frame, summary)
    assert table[column].isna().sum() == 2


_IMMIGRATION_COLUMNS = (
    "immigration_status_str",
    "ssn_card_type",
    "years_since_us_entry",
)


def test_immigration_inputs_are_never_default_filled() -> None:
    """microcosm#1020: the engine defaults are a citizen with a valid SSN and
    5 years since entry, so none of the three may ever be the fill."""

    module = _load_tool_module()
    for column in _IMMIGRATION_COLUMNS:
        assert ("person", column) in module.NEVER_DEFAULT_FILLED


def test_reviewed_null_fill_refuses_to_default_fill_immigration(tmp_path) -> None:
    """A missing immigration cell is refused even when the staging register
    lists it: the default fill is how every ACS person became a citizen."""

    from microcosm.frame import Frame

    module = _load_tool_module()
    frame = _plausible_hours_frame()
    person = frame.table("person").assign(
        immigration_status_str=["CITIZEN"] * 5 + [None] * 3,
        ssn_card_type=["CITIZEN"] * 5 + [None] * 3,
        years_since_us_entry=[40.0] * 7 + [np.nan],
    )
    frame = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    summary = tmp_path / "staging.summary.json"
    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {"entity": "person", "column": column, "missing_rows": 3}
                    for column in _IMMIGRATION_COLUMNS
                ]
            }
        )
    )
    with pytest.raises(module.DeniedDefaultFillError) as exc:
        module.fill_reviewed_nulls(frame, summary)
    message = str(exc.value)
    assert "person.immigration_status_str (3 null rows)" in message
    assert "person.ssn_card_type (3 null rows)" in message
    assert "person.years_since_us_entry (1 null rows)" in message
    assert "microcosm#1020" in message
    assert "re-run staging" in message
    # Refused before anything was filled.
    assert person["ssn_card_type"].isna().sum() == 3
    assert person["years_since_us_entry"].isna().sum() == 1


def _staging_immigration_summary(**overrides) -> dict:
    """The two entries a current staging run records (microcosm#1020)."""

    summary = {
        "acs_local_immigration": {
            "issue": "microcosm#1020",
            "seed": 0,
            "assigned_sha256": "b" * 64,
        },
        "acs_local_immigration_gate": {
            "name": "acs_local_immigration_signal",
            "passed": True,
            "failures": [],
        },
    }
    summary.update(overrides)
    return summary


@pytest.mark.parametrize(
    "summary",
    [
        {},
        _staging_immigration_summary(acs_local_immigration_gate=None),
        _staging_immigration_summary(acs_local_immigration=None),
        _staging_immigration_summary(
            acs_local_immigration_gate={"passed": False, "failures": ["invented"]}
        ),
        _staging_immigration_summary(acs_local_immigration_gate={"passed": "true"}),
        _staging_immigration_summary(
            acs_local_immigration={"issue": "microcosm#1019", "assigned_sha256": "b"}
        ),
        _staging_immigration_summary(acs_local_immigration={"issue": "microcosm#1020"}),
    ],
    ids=[
        "pre-1020-staging",
        "no-gate",
        "no-receipt",
        "gate-failed",
        "gate-truthy-not-true",
        "wrong-issue",
        "no-digest",
    ],
)
def test_immigration_consumers_refuse_a_staging_run_without_the_stage(summary):
    module = _load_tool_module()
    with pytest.raises(SystemExit, match="Re-run staging"):
        module._require_local_immigration(summary)


def test_immigration_consumers_accept_a_current_staging_run() -> None:
    module = _load_tool_module()
    summary = _staging_immigration_summary()
    assert (
        module._require_local_immigration(summary) == summary["acs_local_immigration"]
    )


def test_materialize_refuses_a_pre_1020_staging_before_hashing_it(
    tmp_path, monkeypatch
) -> None:
    """The refusal comes before the staging H5 is hashed or loaded."""

    module = _load_tool_module()
    args = module._parse_args(_materialize_argv(tmp_path))
    (tmp_path / "staging.h5").write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps({"reviewed_engine_input_nulls": [{"entity": "person"}]})
    )
    monkeypatch.setattr(module, "state_admin_specs", lambda *a, **k: ([], []))
    touched = []
    monkeypatch.setattr(module, "_sha256", lambda path: touched.append(path))
    monkeypatch.setattr(
        module, "_load_staging_frame", lambda path: touched.append(path)
    )
    with pytest.raises(SystemExit, match="Re-run staging"):
        module.do_materialize(args)
    assert touched == []
    assert not (args.checkpoint_dir / "run_identity.json").exists()


def test_calibrate_refuses_a_pre_1020_staging_before_solving(
    tmp_path, monkeypatch
) -> None:
    """A checkpoint materialized before #1020 is refused before hours of
    solving, not by the consumer export at the end of the stage."""

    module = _load_tool_module()
    args = module._parse_args(
        [
            "--stage",
            "calibrate",
            "--staging-h5",
            str(tmp_path / "staging.h5"),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--out-h5",
            str(tmp_path / "out.h5"),
        ]
    )
    (tmp_path / "staging.summary.json").write_text(json.dumps({}))
    monkeypatch.setattr(
        module,
        "_verify_run_identity",
        lambda a: {"acs_local_take_up": _TAKE_UP_RECEIPT},
    )
    monkeypatch.setattr(
        module,
        "load_lean_frame",
        lambda *a, **k: pytest.fail("calibrate loaded the checkpoint"),
    )
    with pytest.raises(SystemExit, match="Re-run staging"):
        module.do_calibrate(args)


def test_disability_inputs_are_never_default_filled() -> None:
    """microcosm#1021: the engine default False removes every disability
    exemption, so neither flag may ever be the reviewed-null fill."""

    module = _load_tool_module()
    for column in ("is_disabled", "is_blind"):
        assert ("person", column) in module.NEVER_DEFAULT_FILLED
    # weeks_worked is held back from the export, never filled (#9660).
    assert ("person", "weeks_worked") not in module.NEVER_DEFAULT_FILLED


def test_reviewed_null_fill_refuses_to_default_fill_disability(tmp_path) -> None:
    from microcosm.frame import Frame

    module = _load_tool_module()
    frame = _plausible_hours_frame()
    person = frame.table("person").assign(
        is_disabled=[True, False] * 3 + [None, None],
        is_blind=[False] * 8,
    )
    frame = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    summary = tmp_path / "staging.summary.json"
    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {"entity": "person", "column": "is_disabled", "missing_rows": 2}
                ]
            }
        )
    )
    with pytest.raises(module.DeniedDefaultFillError) as exc:
        module.fill_reviewed_nulls(frame, summary)
    message = str(exc.value)
    assert "person.is_disabled (2 null rows)" in message
    assert "microcosm#1021" in message
    assert "is_blind" not in message
    assert person["is_disabled"].isna().sum() == 2


def _staging_work_disability_summary(**overrides) -> dict:
    """A current staging run's immigration and work/disability entries."""

    native = {"source": "acs_2024_1yr_native", "imputed_rows": 0}
    summary = _staging_immigration_summary(
        acs_local_work_disability={
            "issue": "microcosm#1021",
            "acs_persons": 4,
            "is_disabled": dict(native),
            "is_blind": dict(native),
        },
        acs_local_work_disability_gate={
            "name": "acs_local_work_disability_signal",
            "passed": True,
            "failures": [],
        },
    )
    summary.update(overrides)
    return summary


def _work_disability_receipt(**column_overrides) -> dict:
    receipt = _staging_work_disability_summary()["acs_local_work_disability"]
    for column, entry in column_overrides.items():
        receipt[column] = entry
    return receipt


_INCOME_TRANSFER_COLUMNS = (
    "child_support_received",
    "child_support_expense",
    "workers_compensation",
    "disability_benefits",
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "taxable_sep_distributions",
    "keogh_distributions",
    "tax_exempt_ira_distributions",
)


def _income_receipt(**overrides) -> dict:
    receipt = {
        "issue": "microcosm#1022",
        "donor_channel": "asec",
        "columns": {
            column: {"imputed_rows": 3, "unmodeled_rows": 0}
            for column in _INCOME_TRANSFER_COLUMNS
        },
    }
    receipt.update(overrides)
    return receipt


def _staging_income_summary(**overrides) -> dict:
    """A current staging run through the income transfer (microcosm#1022)."""

    summary = _staging_work_disability_summary(
        acs_local_income_transfer=_income_receipt(),
        acs_local_income_transfer_gate={
            "name": "acs_local_income_transfer_signal",
            "passed": True,
            "failures": [],
        },
    )
    summary.update(overrides)
    return summary


@pytest.mark.parametrize(
    "summary",
    [
        _staging_work_disability_summary(),
        _staging_income_summary(acs_local_income_transfer_gate=None),
        _staging_income_summary(acs_local_income_transfer=None),
        _staging_income_summary(
            acs_local_income_transfer_gate={"passed": False, "failures": ["x"]}
        ),
        _staging_income_summary(acs_local_income_transfer_gate={"passed": "true"}),
        _staging_income_summary(
            acs_local_income_transfer=_income_receipt(issue="microcosm#1021")
        ),
        _staging_income_summary(
            acs_local_income_transfer=_income_receipt(donor_channel="puf")
        ),
        _staging_income_summary(
            acs_local_income_transfer=_income_receipt(
                columns={
                    column: {"imputed_rows": 3, "unmodeled_rows": 0}
                    for column in _INCOME_TRANSFER_COLUMNS[1:]
                }
            )
        ),
        _staging_income_summary(
            acs_local_income_transfer=_income_receipt(
                columns={
                    column: {"imputed_rows": 3, "unmodeled_rows": 2}
                    for column in _INCOME_TRANSFER_COLUMNS
                }
            )
        ),
    ],
    ids=[
        "pre-1022-staging",
        "no-gate",
        "no-receipt",
        "failed-gate",
        "truthy-gate",
        "wrong-issue",
        "wrong-donor-channel",
        "missing-column",
        "unmodeled-rows",
    ],
)
def test_income_transfer_consumers_refuse_a_staging_run_without_the_pass(summary):
    module = _load_tool_module()
    with pytest.raises(SystemExit, match=r"microcosm#1022.*Re-run"):
        module._require_local_income_transfer(summary)


def test_income_transfer_consumers_accept_a_current_staging_run() -> None:
    module = _load_tool_module()
    summary = _staging_income_summary()
    assert (
        module._require_local_income_transfer(summary)
        == summary["acs_local_income_transfer"]
    )


#: The pinned full SIPP 2023 file (ssi_disability_criteria's donor pin).
_SIPP_SHA256 = "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"


def _ssi_disability_receipt(**overrides) -> dict:
    receipt = {
        "issue": "microcosm#1022",
        "column": "meets_ssi_disability_criteria",
        "method": "archived_sipp_qrf_on_acs_rows_missing_cells_only",
        "filled_rows": 9,
        "unfilled_acs_rows": 0,
        "sipp_donor": {"sha256": _SIPP_SHA256},
    }
    receipt.update(overrides)
    return receipt


def _staged_ssi_criteria(acs_true_rows: int = 0) -> dict:
    """A staging SSI receipt with ``acs_true_rows`` criteria-positive ACS persons.

    The package stage's SSI take-up release block reads this count. The
    package fixtures default to 0, a run the block lets through without any
    SSI take-up handling.
    """

    return _ssi_disability_receipt(
        outcome={"acs_true_rows": acs_true_rows, "filled_true_rows": acs_true_rows}
    )


def _staging_ssi_disability_summary(**overrides) -> dict:
    """A current staging run through the SSI disability criteria (microcosm#1022)."""

    summary = _staging_income_summary(
        acs_local_ssi_disability=_ssi_disability_receipt(),
        acs_local_ssi_disability_gate={
            "name": "acs_local_ssi_disability_signal",
            "passed": True,
            "failures": [],
        },
    )
    summary.update(overrides)
    return summary


@pytest.mark.parametrize(
    "summary",
    [
        _staging_income_summary(),
        _staging_ssi_disability_summary(acs_local_ssi_disability_gate=None),
        _staging_ssi_disability_summary(acs_local_ssi_disability=None),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability_gate={"passed": False, "failures": ["x"]}
        ),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability_gate={"passed": "true"}
        ),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability=_ssi_disability_receipt(issue="microcosm#1021")
        ),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability=_ssi_disability_receipt(column="is_disabled")
        ),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability=_ssi_disability_receipt(method="transfer")
        ),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability=_ssi_disability_receipt(filled_rows="9")
        ),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability=_ssi_disability_receipt(unfilled_acs_rows=3)
        ),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability=_ssi_disability_receipt(sipp_donor=None)
        ),
        _staging_ssi_disability_summary(
            acs_local_ssi_disability=_ssi_disability_receipt(
                sipp_donor={"sha256": "0" * 64}
            )
        ),
    ],
    ids=[
        "pre-1022-ssi-staging",
        "no-gate",
        "no-receipt",
        "failed-gate",
        "truthy-gate",
        "wrong-issue",
        "wrong-column",
        "wrong-method",
        "untyped-filled-rows",
        "unfilled-rows",
        "no-sipp-donor",
        "unpinned-sipp-donor",
    ],
)
def test_ssi_disability_consumers_refuse_a_staging_run_without_the_stage(summary):
    module = _load_tool_module()
    with pytest.raises(SystemExit, match=r"microcosm#1022.*Re-run staging"):
        module._require_local_ssi_disability(summary)


def test_ssi_disability_consumers_accept_a_current_staging_run() -> None:
    module = _load_tool_module()
    summary = _staging_ssi_disability_summary()
    assert (
        module._require_local_ssi_disability(summary)
        == summary["acs_local_ssi_disability"]
    )


def test_ssi_disability_criteria_are_never_default_filled() -> None:
    """microcosm#1022: the engine default False fails every ACS person under
    65 who is not blind, so it may never be the reviewed-null fill."""

    module = _load_tool_module()
    assert ("person", "meets_ssi_disability_criteria") in module.NEVER_DEFAULT_FILLED


def test_materialize_refuses_a_pre_ssi_disability_staging_before_hashing_it(
    tmp_path, monkeypatch
) -> None:
    """A staging run through the income transfer but without the SSI
    disability stage is refused before the staging H5 is hashed or loaded."""

    module = _load_tool_module()
    args = module._parse_args(_materialize_argv(tmp_path))
    (tmp_path / "staging.h5").write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [{"entity": "person"}],
                **_staging_income_summary(),
            }
        )
    )
    monkeypatch.setattr(module, "state_admin_specs", lambda *a, **k: ([], []))
    touched = []
    monkeypatch.setattr(module, "_sha256", lambda path: touched.append(path))
    monkeypatch.setattr(
        module, "_load_staging_frame", lambda path: touched.append(path)
    )
    with pytest.raises(SystemExit, match=r"SSI disability-criteria stage"):
        module.do_materialize(args)
    assert touched == []
    assert not (args.checkpoint_dir / "run_identity.json").exists()


def _calibrate_argv(tmp_path) -> list[str]:
    return [
        "--stage",
        "calibrate",
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
        "--out-h5",
        str(tmp_path / "out.h5"),
    ]


def test_calibrate_refuses_a_pre_ssi_disability_staging_before_solving(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = module._parse_args(_calibrate_argv(tmp_path))
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(_staging_income_summary())
    )
    monkeypatch.setattr(
        module,
        "_verify_run_identity",
        lambda a: {"acs_local_take_up": _TAKE_UP_RECEIPT},
    )
    monkeypatch.setattr(
        module,
        "load_lean_frame",
        lambda *a, **k: pytest.fail("calibrate loaded the checkpoint"),
    )
    with pytest.raises(SystemExit, match=r"SSI disability-criteria stage"):
        module.do_calibrate(args)


def test_consumer_export_refuses_a_pre_ssi_disability_staging_before_loading_it(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = module._parse_args(_calibrate_argv(tmp_path))
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(_staging_income_summary())
    )
    monkeypatch.setattr(
        module,
        "_load_staging_frame",
        lambda *a, **k: pytest.fail("the export loaded the staging frame"),
    )
    with pytest.raises(SystemExit, match=r"SSI disability-criteria stage"):
        module._write_calibrated_artifact(
            args,
            np.ones(1),
            {"acs_local_take_up": _TAKE_UP_RECEIPT},
        )


# ---------------------------------------------------------------------------
# microcosm#1022: ACS SSI and Medicaid take-up
# ---------------------------------------------------------------------------

_SSI_MEDICAID = ("takes_up_ssi_if_eligible", "takes_up_medicaid_if_eligible")


def _ssi_medicaid_fixtures():
    """The synthetic frames and stub engine of the stage's own tests."""

    spec = importlib.util.spec_from_file_location(
        "acs_local_ssi_medicaid_take_up_fixtures",
        Path(__file__).with_name("test_us_acs_local_ssi_medicaid_take_up.py"),
    )
    fixtures = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fixtures)
    return fixtures


def _ssi_medicaid_receipt(module, checkpoint_dir: Path, **overrides) -> dict:
    """A current materialize receipt whose recorded assignment file exists."""

    path = checkpoint_dir / module.ACS_SSI_MEDICAID_TAKE_UP_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"recorded assignment")
    receipt = {
        "issue": "microcosm#1022",
        "method": module.ACS_LOCAL_SSI_MEDICAID_TAKE_UP_METHOD,
        "seed": 0,
        "assigned_sha256": "b" * 64,
        "gate": {"passed": True, "failures": []},
        "assignment_file": {"name": path.name, "sha256": module._sha256(path)},
    }
    receipt.update(overrides)
    return receipt


def test_ssi_medicaid_take_up_is_never_default_filled() -> None:
    """microcosm#1022: the engine default True is universal take-up."""

    module = _load_tool_module()
    for column in _SSI_MEDICAID:
        reason = module._NEVER_DEFAULT_FILLED_REASONS[("person", column)]
        assert ("person", column) in module.NEVER_DEFAULT_FILLED
        assert "microcosm#1022" in reason
        assert "universal" in reason


@pytest.mark.parametrize("column", _SSI_MEDICAID)
def test_reviewed_null_fill_refuses_to_default_fill_ssi_medicaid_take_up(
    tmp_path, column
) -> None:
    from microcosm.frame import Frame

    module = _load_tool_module()
    frame = _plausible_hours_frame()
    person = frame.table("person").assign(**{column: [True, False] * 3 + [np.nan] * 2})
    frame = Frame(
        {
            name: person if name == "person" else frame.table(name)
            for name in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    summary = tmp_path / "staging.summary.json"
    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {"entity": "person", "column": column, "missing_rows": 2}
                ]
            }
        )
    )
    with pytest.raises(
        module.DeniedDefaultFillError, match=rf"person\.{column} \(2 null rows\)"
    ):
        module.fill_reviewed_nulls(frame, summary)
    assert person[column].isna().sum() == 2


@pytest.mark.parametrize(
    "edit",
    [
        "absent",
        "wrong_issue",
        "wrong_method",
        "untyped_seed",
        "failed_gate",
        "no_file_record",
        "missing_file",
        "changed_file",
    ],
)
def test_ssi_medicaid_consumers_refuse_a_checkpoint_without_the_assignment(
    tmp_path, edit
) -> None:
    module = _load_tool_module()
    receipt = _ssi_medicaid_receipt(module, tmp_path)
    recorded = tmp_path / module.ACS_SSI_MEDICAID_TAKE_UP_FILENAME
    if edit == "wrong_issue":
        receipt["issue"] = "microcosm#1019"
    elif edit == "wrong_method":
        receipt["method"] = "engine_default"
    elif edit == "untyped_seed":
        receipt["seed"] = "0"
    elif edit == "failed_gate":
        receipt["gate"] = {"passed": False, "failures": ["invented band miss"]}
    elif edit == "no_file_record":
        del receipt["assignment_file"]
    elif edit == "missing_file":
        recorded.unlink()
    elif edit == "changed_file":
        recorded.write_bytes(b"another assignment")
    identity = {} if edit == "absent" else {"acs_local_ssi_medicaid_take_up": receipt}
    with pytest.raises(SystemExit, match=r"microcosm#1022.*Re-run --stage materialize"):
        module._recorded_ssi_medicaid_take_up(identity, tmp_path)


def test_ssi_medicaid_consumers_accept_a_current_checkpoint(tmp_path) -> None:
    module = _load_tool_module()
    receipt = _ssi_medicaid_receipt(module, tmp_path)
    recorded, path = module._recorded_ssi_medicaid_take_up(
        {"acs_local_ssi_medicaid_take_up": receipt}, tmp_path
    )
    assert recorded == receipt
    assert path == tmp_path / module.ACS_SSI_MEDICAID_TAKE_UP_FILENAME


def test_materialize_resolves_the_take_up_counts_before_hashing_the_staging_h5(
    tmp_path, monkeypatch
) -> None:
    """A feed without the SSA band or CMS state counts stops materialize
    before the staging H5 is hashed or loaded."""

    module = _load_tool_module()
    args = module._parse_args(_materialize_argv(tmp_path))
    (tmp_path / "staging.h5").write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [{"entity": "person"}],
                **_staging_ssi_disability_summary(),
            }
        )
    )
    monkeypatch.setattr(module, "state_admin_specs", lambda *a, **k: ([], []))
    touched = []
    monkeypatch.setattr(module, "_sha256", lambda path: touched.append(path))
    monkeypatch.setattr(
        module, "_load_staging_frame", lambda path: touched.append(path)
    )

    def unavailable(feed):
        raise SystemExit(f"no SSA band counts in {Path(feed).name}")

    monkeypatch.setattr(module, "acs_local_ssi_medicaid_take_up_targets", unavailable)
    with pytest.raises(SystemExit, match="no SSA band counts in facts.jsonl"):
        module.do_materialize(args)
    assert touched == []


def test_calibrate_refuses_a_checkpoint_without_the_ssi_medicaid_assignment(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = module._parse_args(_calibrate_argv(tmp_path))
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(_staging_ssi_disability_summary())
    )
    monkeypatch.setattr(
        module,
        "_verify_run_identity",
        lambda a: {"acs_local_take_up": _TAKE_UP_RECEIPT},
    )
    monkeypatch.setattr(
        module,
        "load_lean_frame",
        lambda *a, **k: pytest.fail("calibrate loaded the checkpoint"),
    )
    with pytest.raises(SystemExit, match=r"SSI/Medicaid take-up assignment"):
        module.do_calibrate(args)


def test_consumer_export_refuses_a_checkpoint_without_the_ssi_medicaid_assignment(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = module._parse_args(_calibrate_argv(tmp_path))
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(_staging_ssi_disability_summary())
    )
    monkeypatch.setattr(
        module,
        "_load_staging_frame",
        lambda *a, **k: pytest.fail("the export loaded the staging frame"),
    )
    with pytest.raises(SystemExit, match=r"SSI/Medicaid take-up assignment"):
        module._write_calibrated_artifact(
            args,
            np.ones(1),
            {"acs_local_take_up": _TAKE_UP_RECEIPT},
        )


def test_recorded_ssi_medicaid_assignment_round_trips_through_the_checkpoint(
    tmp_path,
) -> None:
    """The export applies exactly the flags materialize recorded, and
    refuses a file or digest that does not reproduce them."""

    module = _load_tool_module()
    fixtures = _ssi_medicaid_fixtures()
    frame = fixtures._frame(n_donor=40, n_acs_households=600)
    result, receipt = fixtures._assign(frame)
    path = tmp_path / module.ACS_SSI_MEDICAID_TAKE_UP_FILENAME
    assert module._write_ssi_medicaid_take_up(result, path) == module._sha256(path)
    applied = module._with_recorded_ssi_medicaid_take_up(frame, receipt, path)
    for column in _SSI_MEDICAID:
        assert np.array_equal(
            applied.table("person")[column].to_numpy(dtype=bool),
            result.table("person")[column].to_numpy(dtype=bool),
        )
    with pytest.raises(SystemExit, match="does not reproduce its digest"):
        module._with_recorded_ssi_medicaid_take_up(
            frame, {**receipt, "assigned_sha256": "0" * 64}, path
        )
    path.write_bytes(b"not an assignment")
    with pytest.raises(SystemExit, match=r"cannot reproduce the ACS SSI/Medicaid"):
        module._with_recorded_ssi_medicaid_take_up(frame, receipt, path)


def _stub_ssi_medicaid_engine(module, monkeypatch, fixtures, calls, *, eligible=None):
    """Replace projection, the reviewed-null fill and the fiscal lane's two
    batched helpers; each helper returns the fixture's stub column."""

    import build_us_fiscal_refresh_release as release

    def classify(view):
        calls.append(("project",))
        return {"invented_formula_output"}

    monkeypatch.setattr(module, "_formula_owned_columns", classify)
    monkeypatch.setattr(
        module,
        "fill_reviewed_nulls",
        lambda frame, path, **kwargs: calls.append(("fill", path)),
    )

    def helper(column, name):
        def evaluate(view, *, maximum_microsim_batch_size):
            # The engine sees the projected view, never a formula output.
            assert "invented_formula_output" not in view.table("person")
            calls.append((name, view.n("household"), maximum_microsim_batch_size))
            values = view.table("person")[column].to_numpy()
            return (
                values if eligible is None or name != "eligible" else eligible(values)
            )

        return evaluate

    monkeypatch.setattr(
        release, "_ssi_person_uncapped_amount", helper(fixtures.UNCAPPED, "uncapped")
    )
    monkeypatch.setattr(
        release, "_medicaid_person_eligibility", helper(fixtures.ELIGIBLE, "eligible")
    )


def test_ssi_medicaid_stage_runs_both_engine_passes_in_household_chunks(
    tmp_path, monkeypatch
) -> None:
    """SSI candidates, then Medicaid eligibility, each over the projected and
    reviewed-null-filled view in --hh-chunk household batches: the donor
    households first (their pooled contribution sets the ACS residual
    targets), then the ACS households. The tool's stage matches the runtime
    stage and records its gate."""

    module = _load_tool_module()
    fixtures = _ssi_medicaid_fixtures()
    frame = fixtures._frame(n_donor=40, n_acs_households=600)
    frame = fixtures._with_person(
        frame, frame.table("person").assign(invented_formula_output=1.0)
    )
    ssi, medicaid = fixtures._targets(frame)
    calls: list = []
    _stub_ssi_medicaid_engine(module, monkeypatch, fixtures, calls)
    summary = tmp_path / "staging.summary.json"
    result, receipt = module._with_local_ssi_medicaid_take_up(
        frame,
        seed=5,
        targets={
            "ssi_band_targets": ssi,
            "medicaid_state_targets": medicaid,
            "medicaid_substitutions": [],
        },
        summary_path=summary,
        hh_chunk=100,
    )
    assert calls == [
        ("project",),
        ("fill", summary),
        ("uncapped", 40, 100),
        ("project",),
        ("fill", summary),
        ("eligible", 40, 100),
        ("project",),
        ("fill", summary),
        ("uncapped", 600, 100),
        ("project",),
        ("fill", summary),
        ("eligible", 600, 100),
    ]
    assert receipt["gate"] == {
        "name": "acs_local_ssi_medicaid_take_up_signal",
        "passed": True,
        "failures": [],
    }
    assert receipt["engine_prepass"]["hh_chunk"] == 100
    # The view is projected in place; the caller's frame keeps its columns.
    assert "invented_formula_output" in result.table("person")
    _, direct = fixtures._assign(frame)
    assert receipt["assigned_sha256"] == direct["assigned_sha256"]
    for column in _SSI_MEDICAID:
        assert result.table("person")[column].notna().all()


def test_ssi_medicaid_stage_failure_stops_materialize(tmp_path, monkeypatch) -> None:
    """An eligibility pass that finds no one eligible cannot carry the CMS
    counts: the gate fails before the materialize engine pass."""

    module = _load_tool_module()
    fixtures = _ssi_medicaid_fixtures()
    frame = fixtures._frame(n_donor=40, n_acs_households=300)
    ssi, medicaid = fixtures._targets(frame)
    _stub_ssi_medicaid_engine(
        module,
        monkeypatch,
        fixtures,
        [],
        eligible=lambda values: np.zeros(len(values), dtype=bool),
    )
    with pytest.raises(SystemExit, match="SSI/Medicaid take-up gate failed"):
        module._with_local_ssi_medicaid_take_up(
            frame,
            seed=5,
            targets={
                "ssi_band_targets": ssi,
                "medicaid_state_targets": medicaid,
                "medicaid_substitutions": [],
            },
            summary_path=tmp_path / "staging.summary.json",
            hh_chunk=100,
        )


@pytest.mark.parametrize(
    "summary",
    [
        _staging_immigration_summary(),
        _staging_work_disability_summary(acs_local_work_disability_gate=None),
        _staging_work_disability_summary(acs_local_work_disability=None),
        _staging_work_disability_summary(
            acs_local_work_disability_gate={"passed": False, "failures": ["x"]}
        ),
        _staging_work_disability_summary(
            acs_local_work_disability_gate={"passed": "true"}
        ),
        _staging_work_disability_summary(
            acs_local_work_disability={
                **_work_disability_receipt(),
                "issue": "microcosm#1020",
            }
        ),
        _staging_work_disability_summary(
            acs_local_work_disability=_work_disability_receipt(
                is_disabled={"source": "acs_transfer", "imputed_rows": 0}
            )
        ),
        _staging_work_disability_summary(
            acs_local_work_disability=_work_disability_receipt(
                is_blind={"source": "acs_2024_1yr_native", "imputed_rows": 12}
            )
        ),
        _staging_work_disability_summary(
            acs_local_work_disability=_work_disability_receipt(
                is_blind={"source": "acs_2024_1yr_native"}
            )
        ),
    ],
    ids=[
        "pre-1021-staging",
        "no-gate",
        "no-receipt",
        "gate-failed",
        "gate-truthy-not-true",
        "wrong-issue",
        "not-native",
        "imputed",
        "unrecorded-transfer",
    ],
)
def test_work_disability_consumers_refuse_a_staging_run_without_the_stage(summary):
    module = _load_tool_module()
    with pytest.raises(SystemExit, match=r"microcosm#1021.*Re-run staging"):
        module._require_local_work_disability(summary)


def test_work_disability_consumers_accept_a_current_staging_run() -> None:
    module = _load_tool_module()
    summary = _staging_work_disability_summary()
    assert (
        module._require_local_work_disability(summary)
        == summary["acs_local_work_disability"]
    )


def test_materialize_refuses_a_pre_1021_staging_before_hashing_it(
    tmp_path, monkeypatch
) -> None:
    """A staging run with the #1020 stage but not #1021 is refused before the
    staging H5 is hashed or loaded."""

    module = _load_tool_module()
    args = module._parse_args(_materialize_argv(tmp_path))
    (tmp_path / "staging.h5").write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [{"entity": "person"}],
                **_staging_immigration_summary(),
            }
        )
    )
    monkeypatch.setattr(module, "state_admin_specs", lambda *a, **k: ([], []))
    touched = []
    monkeypatch.setattr(module, "_sha256", lambda path: touched.append(path))
    monkeypatch.setattr(
        module, "_load_staging_frame", lambda path: touched.append(path)
    )
    with pytest.raises(SystemExit, match=r"microcosm#1021.*Re-run staging"):
        module.do_materialize(args)
    assert touched == []
    assert not (args.checkpoint_dir / "run_identity.json").exists()


def test_calibrate_refuses_a_pre_1021_staging_before_solving(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = module._parse_args(
        [
            "--stage",
            "calibrate",
            "--staging-h5",
            str(tmp_path / "staging.h5"),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--out-h5",
            str(tmp_path / "out.h5"),
        ]
    )
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(_staging_immigration_summary())
    )
    monkeypatch.setattr(
        module,
        "_verify_run_identity",
        lambda a: {"acs_local_take_up": _TAKE_UP_RECEIPT},
    )
    monkeypatch.setattr(
        module,
        "load_lean_frame",
        lambda *a, **k: pytest.fail("calibrate loaded the checkpoint"),
    )
    with pytest.raises(SystemExit, match=r"microcosm#1021.*Re-run staging"):
        module.do_calibrate(args)


def test_consumer_export_refuses_a_pre_1021_staging_before_loading_it(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = module._parse_args(
        [
            "--stage",
            "calibrate",
            "--staging-h5",
            str(tmp_path / "staging.h5"),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--out-h5",
            str(tmp_path / "out.h5"),
        ]
    )
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(_staging_immigration_summary())
    )
    monkeypatch.setattr(
        module,
        "_load_staging_frame",
        lambda *a, **k: pytest.fail("the export loaded the staging frame"),
    )
    with pytest.raises(SystemExit, match=r"microcosm#1021.*Re-run staging"):
        module._write_calibrated_artifact(
            args,
            np.ones(1),
            {"acs_local_take_up": _TAKE_UP_RECEIPT},
        )


def _package_evidence_args(module, tmp_path: Path, monkeypatch, *, hours_report):
    """Every package-stage input, with the finalize report's hours entry given.

    ``hours_report`` is what ``gate_summary.json`` records under
    ``acs_local_hours_signal`` (``None`` omits the key, as a report finalized
    before the gate existed would). The staging frame and the hours gate are
    stubbed: the test is about the binding, not the classification.
    """
    from microcosm.build.gates import GateResult

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [],
                "reviewed_limitations": [],
                # An uncapped staging run, so the package stage's cap check
                # (which runs first) lets these inputs reach the hours gates.
                "orchestration": {"max_households": None},
                # No criteria-positive ACS person: the SSI take-up release
                # block lets the run through.
                "acs_local_ssi_disability": _staged_ssi_criteria(),
            }
        )
    )
    out_h5 = tmp_path / "out.h5"
    out_h5.write_bytes(b"artifact")
    artifact_sha = module._sha256(out_h5)
    gates = {
        "us_puma_ladder_gate": {"passed": True, "failures": []},
        # The general hours gate, bound to these bytes: the package stage
        # requires it before it reaches the ACS local-hours re-check.
        "hours_worked_signal": {
            "passed": True,
            "failures": [],
            "detail": {},
            "artifact_sha256": artifact_sha,
        },
        "acs_local_take_up_signal": {
            "passed": True,
            "failures": [],
            "detail": {},
            "artifact_sha256": artifact_sha,
        },
        "acs_local_immigration_signal": {
            "passed": True,
            "failures": [],
            "detail": {},
            "artifact_sha256": artifact_sha,
        },
        "acs_local_work_disability_signal": {
            "passed": True,
            "failures": [],
            "detail": {},
            "artifact_sha256": artifact_sha,
        },
        "acs_local_income_transfer_signal": {
            "passed": True,
            "failures": [],
            "detail": {},
            "artifact_sha256": artifact_sha,
        },
        "acs_local_ssi_disability_signal": {
            "passed": True,
            "failures": [],
            "detail": {},
            "artifact_sha256": artifact_sha,
        },
        "acs_local_ssi_medicaid_take_up_signal": {
            "passed": True,
            "failures": [],
            "detail": {},
            "artifact_sha256": artifact_sha,
        },
    }
    if hours_report is not None:
        gates["acs_local_hours_signal"] = hours_report
    evidence = {
        "calibration_summary.json": {
            "households": 1,
            "calibration_diagnostics": {
                "status": "failed",
                "expected_schema_version": 8,
                "error_code": "validation_error",
                "message": "test fixture",
            },
        },
        "gate_summary.json": {"gates": gates, "reviewed_limitations": []},
        "run_identity.json": {
            "staging_sha256": module._sha256(staging),
            "population_cells_dropped": [],
        },
        "spine_qa.json": {
            "plain_consumption": True,
            "artifact_sha256": artifact_sha,
            "per_spine": {},
        },
        "consumer_export.json": {"staging_sha256": module._sha256(staging)},
        "held_back_columns.json": {"total": 0},
        "reviewed_null_fills.json": {"columns_filled": []},
        "materialize_rss.json": {
            "soi_mode": "totals",
            "materialize_peak_rss_gb": 1.0,
            "hh_chunk": 1,
        },
        "consumer_reviewed_null_fills.json": {"columns_filled": []},
    }
    for name, payload in evidence.items():
        (ckpt / name).write_text(json.dumps(payload))
    (tmp_path / "out.summary.json").write_text(json.dumps({"simulation_ready": True}))
    monkeypatch.setattr(module, "_load_staging_frame", lambda *_a, **_k: object())
    monkeypatch.setattr(
        module,
        "acs_local_hours_signal_gate",
        lambda frame, *, source_null_audit: GateResult(
            name="acs_local_hours_signal",
            passed=True,
            failures=(),
            details={"per_spine": {"acs_2024_1yr": {"rows": 1}}},
        ),
    )
    return module._parse_args(
        [
            "--stage",
            "package",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(out_h5),
            "--out",
            str(tmp_path / "release"),
            "--allow-dirty",
        ]
    )


@pytest.mark.parametrize(
    "hours_report",
    [None, {"passed": False, "failures": ["invented"]}, {"passed": "true"}],
    ids=["finalized-before-the-gate", "finalize-failed", "truthy-not-true"],
)
def test_package_requires_a_passing_hours_gate_in_the_finalize_report(
    tmp_path: Path, monkeypatch, hours_report
) -> None:
    module = _load_tool_module()
    args = _package_evidence_args(
        module, tmp_path, monkeypatch, hours_report=hours_report
    )
    with pytest.raises(SystemExit, match="Re-run --stage finalize"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()


def test_package_binds_the_hours_gate_to_the_packaged_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = _package_evidence_args(
        module,
        tmp_path,
        monkeypatch,
        hours_report={"passed": True, "failures": [], "detail": {}},
    )
    result = module.do_package(args)
    release_dir = Path(result["release_dir"])
    expected_sha = module._sha256(args.out_h5)
    for name in ("build_manifest.json", "gate_summary.json"):
        gate = json.loads((release_dir / name).read_text())["gates"][
            "acs_local_hours_signal"
        ]
        assert gate["passed"] is True
        assert gate["artifact_sha256"] == expected_sha
        assert gate["checked_at_stage"] == "package"
        assert gate["detail"] == {"per_spine": {"acs_2024_1yr": {"rows": 1}}}
    # The cap check that runs before the hours gates records what it passed.
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    assert build_manifest["staging_orchestration"]["max_households"] is None


def test_package_refuses_when_the_packaged_bytes_fail_the_hours_gate(
    tmp_path: Path, monkeypatch
) -> None:
    """A passing finalize entry does not stand in for the package-time re-check.

    The re-check runs on the calibrated H5 being packaged; if it fails, nothing
    ships: no manifest, no package result, no artifact at the release root.
    """
    from microcosm.build.gates import GateResult

    module = _load_tool_module()
    args = _package_evidence_args(
        module,
        tmp_path,
        monkeypatch,
        hours_report={"passed": True, "failures": [], "detail": {}},
    )
    loaded = []

    def load_frame(path, *_a, **_k):
        loaded.append(Path(path))
        return object()

    monkeypatch.setattr(module, "_load_staging_frame", load_frame)
    monkeypatch.setattr(
        module,
        "acs_local_hours_signal_gate",
        lambda frame, *, source_null_audit: GateResult(
            name="acs_local_hours_signal",
            passed=False,
            failures=("acs_2024_1yr: invented unresolved hours",),
        ),
    )
    with pytest.raises(
        SystemExit, match="Local hours coverage failed: acs_2024_1yr: invented"
    ):
        module.do_package(args)
    assert loaded == [Path(args.out_h5)]
    assert not (args.out / "package_result.json").exists()
    assert not list((args.out / "releases").rglob("*.json"))
    assert not (args.out / module.ARTIFACT_FILENAME).exists()


_UNSET = object()


def _package_args_before_evidence(
    module, tmp_path: Path, *, max_households=_UNSET, soi_mode="totals"
):
    """The package stage's inputs up to (not including) the qa/consumer evidence.

    ``max_households`` is what the staging summary records under
    ``orchestration``; the sentinel omits the block entirely. ``soi_mode`` is
    what ``materialize_rss.json`` records; the sentinel omits the file.
    """
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    if soi_mode is not _UNSET:
        (ckpt / "materialize_rss.json").write_text(json.dumps({"soi_mode": soi_mode}))
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    summary: dict = {"acs_local_ssi_disability": _staged_ssi_criteria()}
    if max_households is not _UNSET:
        summary["orchestration"] = {
            "max_households": max_households,
            "n_estimators": 32,
            "max_targets_per_fit": 8,
        }
    (tmp_path / "staging.summary.json").write_text(json.dumps(summary))
    out_h5 = tmp_path / "out.h5"
    out_h5.write_bytes(b"artifact")
    (ckpt / "calibration_summary.json").write_text(
        json.dumps(
            {
                "households": 1,
                "calibration_diagnostics": {
                    "status": "failed",
                    "expected_schema_version": 8,
                    "error_code": "validation_error",
                    "message": "test fixture",
                },
            }
        )
    )
    (ckpt / "gate_summary.json").write_text(json.dumps({"gates": {}}))
    (ckpt / "run_identity.json").write_text(
        json.dumps(
            {
                "staging_sha256": module._sha256(staging),
                "population_cells_dropped": [],
            }
        )
    )
    (tmp_path / "out.summary.json").write_text(json.dumps({"simulation_ready": True}))
    return module._parse_args(
        [
            "--stage",
            "package",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(out_h5),
            "--out",
            str(tmp_path / "release"),
            "--allow-dirty",
        ]
    )


@pytest.mark.parametrize(
    ("max_households", "message"),
    [
        (5000, r"capped at 5000 ACS household"),
        (0, r"capped at 0 ACS household"),
        (_UNSET, r"does not record orchestration\.max_households"),
    ],
    ids=["capped", "capped-at-zero", "cap-not-recorded"],
)
def test_package_refuses_a_capped_or_unattested_staging_run(
    tmp_path: Path, max_households, message
) -> None:
    """A capped smoke passes every finalize gate (the donor spine keeps every
    ladder cell populated), so the cap itself must refuse packaging, before
    any release directory exists."""

    module = _load_tool_module()
    args = _package_args_before_evidence(
        module, tmp_path, max_households=max_households
    )
    with pytest.raises(SystemExit, match=message):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not (args.out / "releases").exists(), "a refused smoke leaves no release"


def test_package_checks_the_cap_before_reading_the_evidence(tmp_path: Path) -> None:
    """An uncapped summary reaches the evidence checks; the cap check is first."""

    module = _load_tool_module()
    args = _package_args_before_evidence(module, tmp_path, max_households=None)
    with pytest.raises(SystemExit, match="spine_qa.json is missing"):
        module.do_package(args)


def test_do_package_requires_qa_and_consumer_evidence(tmp_path: Path) -> None:
    """Absent evidence must refuse packaging, never read as vacuously green."""

    module = _load_tool_module()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(
            {
                "orchestration": {"max_households": None},
                "acs_local_ssi_disability": _staged_ssi_criteria(),
            }
        )
    )
    out_h5 = tmp_path / "out.h5"
    out_h5.write_bytes(b"artifact")
    (ckpt / "calibration_summary.json").write_text(
        json.dumps(
            {
                "households": 1,
                "calibration_diagnostics": {
                    "status": "failed",
                    "expected_schema_version": 8,
                    "error_code": "validation_error",
                    "message": "test fixture",
                },
            }
        )
    )
    (ckpt / "gate_summary.json").write_text(json.dumps({"gates": {}}))
    (ckpt / "materialize_rss.json").write_text(json.dumps({"soi_mode": "totals"}))
    (ckpt / "run_identity.json").write_text(
        json.dumps(
            {
                "staging_sha256": module._sha256(staging),
                "population_cells_dropped": [],
            }
        )
    )
    args = module._parse_args(
        [
            "--stage",
            "package",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(out_h5),
            "--out",
            str(tmp_path / "release"),
            "--allow-dirty",
        ]
    )
    (tmp_path / "out.summary.json").write_text(json.dumps({"simulation_ready": True}))

    with pytest.raises(SystemExit, match="spine_qa.json is missing"):
        module.do_package(args)

    (ckpt / "spine_qa.json").write_text(
        json.dumps(
            {
                "plain_consumption": True,
                "artifact_sha256": module._sha256(out_h5),
                "per_spine": {},
            }
        )
    )
    with pytest.raises(SystemExit, match="consumer_export.json is missing"):
        module.do_package(args)


# ---------------------------------------------------------------------------
# SOI target surface: state (Build O contract) by default; totals and full opt-in
# ---------------------------------------------------------------------------


def _materialize_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--stage",
        "materialize",
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
        "--feed",
        str(tmp_path / "facts.jsonl"),
        *extra,
    ]


def test_soi_mode_defaults_to_state_and_totals_and_full_are_explicit_opt_ins(
    tmp_path: Path,
) -> None:
    """Max's ruling of 2026-09-22: the ACS local default is Build O's
    state-geography SOI contract; ``totals`` and ``full`` are reachable only by
    asking for them."""

    module = _load_tool_module()
    assert module.SOI_MODES == ("state", "totals", "full")
    assert module.DEFAULT_SOI_MODE == module.SOI_MODE_STATE == "state"
    assert module._parse_args(_materialize_argv(tmp_path)).soi_mode == "state"
    for mode in ("totals", "full"):
        assert (
            module._parse_args(_materialize_argv(tmp_path, "--soi-mode", mode)).soi_mode
            == mode
        )
    signature = inspect.signature(module.state_admin_specs)
    assert signature.parameters["soi_mode"].default == "state"
    with pytest.raises(SystemExit):
        module._parse_args(_materialize_argv(tmp_path, "--soi-mode", "ful"))


_HT2_BROAD = "irs_soi.historic_table_2.state_broad_totals.v1"
_HT2_AGI = "irs_soi.historic_table_2.state_agi_counts_and_amounts.v1"
_CD_FILE = "irs_soi.congressional_district_2022.all_returns.v1"


def _spec(
    role: str | None,
    *,
    state: bool = True,
    geography: str | None = "state",
    record_set: str | None = _HT2_BROAD,
) -> SimpleNamespace:
    metadata: dict[str, str] = {}
    if state:
        metadata["state_fips"] = "06"
    if role is not None:
        metadata["target_role"] = role
    if geography is not None:
        metadata["ledger_geography_level"] = geography
    if record_set is not None:
        metadata["ledger_layout_record_set_spec_id"] = record_set
    return SimpleNamespace(metadata=metadata)


def test_state_soi_surface_is_build_o_contract_by_record_set_and_geography() -> None:
    """``state`` keeps state-geography specs outside the district file, of any
    role (AGI-band rows included), and refuses to guess on missing metadata."""

    module = _load_tool_module()
    state = module.soi_surface_predicate("state")

    for role in ("soi_fiscal_distribution", "aca_ptc_returns", "aca_spending", None):
        assert state(_spec(role)), role
        assert state(_spec(role, record_set=_HT2_AGI)), role
    # The TY2023 congressional-district file is excluded even at state geography
    # (its <st>_total rows restate the Historic Table 2 totals a year later).
    assert not state(_spec("soi_fiscal_distribution", record_set=_CD_FILE))
    assert not state(_spec("aca_ptc_returns", record_set=_CD_FILE))
    # District geography is excluded even from Historic Table 2.
    assert not state(
        _spec("soi_fiscal_distribution", geography="congressional_district")
    )
    # No state_fips: never on the state surface.
    assert not state(_spec("soi_fiscal_distribution", state=False))
    # Missing either key: the mode cannot tell which contract the spec is in.
    assert not state(_spec("soi_fiscal_distribution", geography=None))
    assert not state(_spec("soi_fiscal_distribution", record_set=None))
    assert not state(_spec("soi_fiscal_distribution", record_set=""))


def test_soi_surface_predicate_drops_soi_fiscal_distribution_only_in_totals() -> None:
    module = _load_tool_module()
    totals = module.soi_surface_predicate("totals")
    full = module.soi_surface_predicate("full")

    band = _spec("soi_fiscal_distribution")
    assert not totals(band)
    assert full(band)
    for role in ("aca_ptc_returns", "aca_spending", None):
        assert totals(_spec(role)) and full(_spec(role)), role
    # Neither mode reaches past the state surface.
    for mode_predicate in (totals, full):
        assert not mode_predicate(_spec("aca_spending", state=False))
        assert not mode_predicate(_spec("soi_fiscal_distribution", state=False))


def test_unknown_soi_mode_is_refused_before_the_feed_is_read(tmp_path: Path) -> None:
    """A typo must never fall through to either surface (the old predicate
    treated every value other than ``full`` as totals)."""

    module = _load_tool_module()
    missing_feed = tmp_path / "never-read.jsonl"
    for call in (
        lambda: module.soi_surface_predicate("ful"),
        lambda: module.state_admin_specs(missing_feed, ["soi"], soi_mode="ful"),
        lambda: module.release_refresh_recipe("ful"),
    ):
        with pytest.raises(ValueError, match="soi_mode must be one of"):
            call()


@pytest.mark.parametrize("soi_mode", ["state", "totals", "full"])
def test_release_refresh_recipe_reproduces_its_soi_mode(
    tmp_path: Path, soi_mode: str
) -> None:
    """The recipe names the mode, so re-running it cannot drift with the
    parser default."""

    module = _load_tool_module()
    recipe = shlex.split(module.release_refresh_recipe(soi_mode))
    assert recipe[:3] == ["uv", "run", "tools/build_us_acs_local_release.py"]
    assert recipe[recipe.index("--soi-mode") + 1] == soi_mode
    args = module._parse_args(recipe[3:])
    assert args.soi_mode == soi_mode
    assert args.stages == ["materialize", "calibrate", "qa", "finalize", "package"]


@pytest.mark.parametrize(
    ("recorded", "message"),
    [
        (_UNSET, r"records soi_mode=None"),
        ("bands", r"records soi_mode='bands'"),
    ],
    ids=["not-recorded", "unknown"],
)
def test_package_refuses_a_checkpoint_without_a_known_soi_mode(
    tmp_path: Path, recorded, message
) -> None:
    module = _load_tool_module()
    args = _package_args_before_evidence(
        module, tmp_path, max_households=None, soi_mode=recorded
    )
    with pytest.raises(SystemExit, match=message):
        module.do_package(args)
    assert not (args.out / "releases").exists(), "a refusal leaves no release"


@pytest.mark.parametrize("recorded", ["state", "totals", "full"])
def test_package_records_the_materialized_soi_mode_not_the_parser_default(
    tmp_path: Path, monkeypatch, recorded: str
) -> None:
    """The package invocation passes no ``--soi-mode`` (so the parser says
    ``state``); the manifest and recipe must carry what materialize used."""

    module = _load_tool_module()
    args = _package_evidence_args(
        module,
        tmp_path,
        monkeypatch,
        hours_report={"passed": True, "failures": [], "detail": {}},
    )
    assert args.soi_mode == "state"
    (args.checkpoint_dir / "materialize_rss.json").write_text(
        json.dumps({"soi_mode": recorded, "hh_chunk": 1})
    )

    result = module.do_package(args)

    release_dir = Path(result["release_dir"])
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    release_manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert build_manifest["materialize"]["soi_mode"] == recorded
    for manifest in (build_manifest, release_manifest):
        recipe = shlex.split(manifest["refresh_recipe"]["release"])
        assert recipe[recipe.index("--soi-mode") + 1] == recorded


def test_pinned_feed_soi_surfaces_match_their_contracts() -> None:
    """On the real feed: the default ``state`` surface is Build O/P's
    3,972-spec admin contract, built only from the three Historic Table 2
    state tables at state geography; ``totals`` holds no
    ``soi_fiscal_distribution`` spec; ``full`` adds exactly those specs to
    ``totals`` and contains ``state``.

    The pinned consumer-facts feed is a 164 MB public aggregate export that
    no CI lane carries, so this runs only when ``MICROCOSM_US_CHRONICLE_FACTS``
    points at it (the convention of the #969 state-surface arm). Three
    registry compiles; about 3 GB peak RSS.
    """

    feed = os.environ.get("MICROCOSM_US_CHRONICLE_FACTS", "")
    if not feed or not Path(feed).exists():
        pytest.skip(
            "set MICROCOSM_US_CHRONICLE_FACTS to the pinned consumer-facts feed"
        )

    module = _load_tool_module()
    families = ["snap", "medicaid", "soi"]
    state_registry, _ = module.state_admin_specs(feed, families)
    totals_registry, _ = module.state_admin_specs(feed, families, soi_mode="totals")
    full_registry, _ = module.state_admin_specs(feed, families, soi_mode="full")
    state_specs = state_registry.specs
    totals_names = {spec.name for spec in totals_registry.specs}
    full_by_name = {spec.name: spec for spec in full_registry.specs}

    # Build P's ACS local contract: 102 SNAP + 51 Medicaid + 3,819 SOI admin
    # specs (4,459 with the 487 population marginals); Build O's 4,461 plus the
    # Vermont under-$1 taxable-interest pair the compiler now excludes.
    families_count = {}
    for spec in state_specs:
        families_count[spec.family] = families_count.get(spec.family, 0) + 1
    assert families_count == {"usda_snap": 102, "cms_medicaid": 51, "irs_soi": 3819}
    soi = [spec for spec in state_specs if spec.family == "irs_soi"]
    record_sets = {}
    for spec in soi:
        key = spec.metadata["ledger_layout_record_set_spec_id"]
        record_sets[key] = record_sets.get(key, 0) + 1
    assert record_sets == {
        "irs_soi.historic_table_2.state_broad_totals.v1": 2397,
        "irs_soi.historic_table_2.state_agi_counts_and_amounts.v1": 912,
        "irs_soi.historic_table_2.state_eitc.v1": 510,
    }
    assert {spec.metadata["ledger_geography_level"] for spec in soi} == {"state"}
    assert {spec.name for spec in state_specs} <= set(full_by_name)

    assert not [
        spec
        for spec in totals_registry.specs
        if spec.metadata.get("target_role") == "soi_fiscal_distribution"
    ]
    assert totals_names < set(full_by_name)
    added = [full_by_name[name] for name in set(full_by_name) - totals_names]
    assert added
    assert {(spec.family, spec.metadata.get("target_role")) for spec in added} == {
        ("irs_soi", "soi_fiscal_distribution")
    }


def _staging_frame_with_hours(weekly: list[float], last_week: list[float]):
    """A minimal US-schema staging frame carrying the two pool hours columns."""

    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    n = len(weekly)
    ids = np.arange(1, n + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "weekly_hours_worked_before_lsr": np.asarray(weekly, dtype=float),
            "hours_worked_last_week": np.asarray(last_week, dtype=float),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n, dtype=np.float64), WeightKind.DESIGN)},
    )


def _finalize_args(module, tmp_path: Path):
    staging = tmp_path / "staging.h5"
    staging.touch()
    (tmp_path / "staging.summary.json").write_text(
        json.dumps(
            {
                "reviewed_limitations": [],
                "reviewed_engine_input_nulls": [],
                # The current staging builder records its cap; an uncapped run
                # is what the package-stage tests built on this fixture need.
                "orchestration": {"max_households": None},
                # No criteria-positive ACS person, so the package stage's SSI
                # take-up release block lets these runs through.
                "acs_local_ssi_disability": _staged_ssi_criteria(),
            }
        )
    )
    # finalize hashes the calibrated H5 before loading it; tests that do not
    # load real bytes still need bytes to hash.
    (tmp_path / "out.h5").write_bytes(b"invented-finalize-artifact")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "calibration_summary.json").write_text(
        json.dumps(
            {
                "final_loss": 0.1,
                "initial_loss": 0.5,
                "mass_conserved_ratio": 1.0,
                "calibration_diagnostics": {
                    "status": "failed",
                    "expected_schema_version": 8,
                    "error_code": "validation_error",
                    "message": "test fixture",
                },
            }
        )
    )
    # Every checkpoint materialize writes records its SOI mode; package
    # refuses one that does not.
    (ckpt / "materialize_rss.json").write_text(json.dumps({"soi_mode": "totals"}))
    ladder = tmp_path / "ladder.npz"
    ladder.write_bytes(b"ladder-bytes")
    return module._parse_args(
        [
            "--stage",
            "finalize",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(tmp_path / "out.h5"),
            "--ladder",
            str(ladder),
        ]
    )


def _stub_local_hours_gate(module, monkeypatch) -> None:
    """Make the ACS local-hours classification pass; its own tests cover it."""

    from microcosm.build.gates import GateResult

    monkeypatch.setattr(
        module,
        "acs_local_hours_signal_gate",
        lambda frame, *, source_null_audit: GateResult(
            name="acs_local_hours_signal", passed=True, failures=(), details={}
        ),
    )


def _stub_local_take_up_gate(module, monkeypatch, *, passed=True) -> None:
    """Make the ACS take-up classification pass (or fail); its tests cover it."""

    from microcosm.build.gates import GateResult

    monkeypatch.setattr(
        module,
        "acs_local_take_up_signal_gate",
        lambda frame: GateResult(
            name="acs_local_take_up_signal",
            passed=passed,
            failures=() if passed else ("acs_2024_1yr: invented constant take-up",),
            details={},
        ),
    )


def _stub_local_immigration_gate(module, monkeypatch, *, passed=True) -> None:
    """Make the ACS immigration classification pass (or fail); its tests
    (test_us_acs_local_immigration.py) cover it."""

    from microcosm.build.gates import GateResult

    monkeypatch.setattr(
        module,
        "acs_local_immigration_signal_gate",
        lambda frame: GateResult(
            name="acs_local_immigration_signal",
            passed=passed,
            failures=()
            if passed
            else ("acs_2024_1yr: ssn_card_type is constant 'CITIZEN'",),
            details={},
        ),
    )


def _stub_local_work_disability_gate(
    module, monkeypatch, *, passed=True, calls=None
) -> None:
    """Make the native ACS work/disability classification pass (or fail); its
    tests (test_us_acs_local_work_disability.py) cover it."""

    from microcosm.build.gates import GateResult

    def gate(frame, *, receipt, require_weeks_worked=True):
        if calls is not None:
            calls.append((receipt, require_weeks_worked))
        return GateResult(
            name="acs_local_work_disability_signal",
            passed=passed,
            failures=() if passed else ("acs_2024_1yr: is_disabled is constant False",),
            details={},
        )

    monkeypatch.setattr(module, "acs_local_work_disability_signal_gate", gate)


def _stub_local_income_gate(module, monkeypatch, *, passed=True) -> None:
    """Make the ACS local income transfer gate pass (or fail); its tests
    (test_us_acs_local_income.py) cover it."""

    from microcosm.build.gates import GateResult

    def gate(frame, *, receipt):
        return GateResult(
            name="acs_local_income_transfer_signal",
            passed=passed,
            failures=()
            if passed
            else ("acs_2024_1yr: child_support_received has 2 missing row(s).",),
            details={},
        )

    monkeypatch.setattr(module, "acs_local_income_transfer_signal_gate", gate)


def _stub_local_ssi_disability_gate(module, monkeypatch, *, passed=True) -> None:
    """Make the ACS local SSI disability gate pass (or fail); its tests
    (test_us_acs_local_ssi_disability.py) cover it."""

    from microcosm.build.gates import GateResult

    def gate(frame, *, receipt):
        return GateResult(
            name="acs_local_ssi_disability_signal",
            passed=passed,
            failures=()
            if passed
            else ("acs_2024_1yr: meets_ssi_disability_criteria has 2 missing row(s).",),
            details={},
        )

    monkeypatch.setattr(module, "acs_local_ssi_disability_signal_gate", gate)


def _stub_local_ssi_medicaid_take_up_gate(
    module, monkeypatch, *, passed=True, receipts=None
) -> None:
    """Make the ACS SSI/Medicaid take-up gate pass (or fail); its tests
    (test_us_acs_local_ssi_medicaid_take_up.py) cover it."""

    from microcosm.build.gates import GateResult

    def gate(frame, *, receipt):
        if receipts is not None:
            receipts.append(receipt)
        return GateResult(
            name="acs_local_ssi_medicaid_take_up_signal",
            passed=passed,
            failures=()
            if passed
            else ("acs_2024_1yr: takes_up_ssi_if_eligible has 2 missing row(s).",),
            details={},
        )

    monkeypatch.setattr(module, "acs_local_ssi_medicaid_take_up_signal_gate", gate)


def _patch_finalize_collaborators(module, monkeypatch, frame=None, *, identity=True):
    """Stub the ladder gate and composition; ``frame=None`` loads real bytes."""

    import microcosm.build.us_runtime.puma_ladder as puma
    from microcosm.build.gates import GateResult

    monkeypatch.setattr(puma, "load_us_puma_ladder", lambda *a, **k: None)
    monkeypatch.setattr(
        puma,
        "us_puma_ladder_gate",
        lambda *a, **k: GateResult(
            name="us_puma_ladder", passed=True, failures=(), details={}
        ),
    )
    monkeypatch.setattr(module, "spine_composition", lambda *a, **k: {})
    _stub_local_hours_gate(module, monkeypatch)
    _stub_local_take_up_gate(module, monkeypatch)
    _stub_local_immigration_gate(module, monkeypatch)
    _stub_local_work_disability_gate(module, monkeypatch)
    _stub_local_income_gate(module, monkeypatch)
    _stub_local_ssi_disability_gate(module, monkeypatch)
    _stub_local_ssi_medicaid_take_up_gate(module, monkeypatch)
    if frame is not None:
        monkeypatch.setattr(module, "_load_staging_frame", lambda *a, **k: frame)
    if identity:
        monkeypatch.setattr(
            module,
            "_verify_run_identity",
            lambda a: {"ladder_sha256": module._sha256(a.ladder)},
        )


def _run_finalize(module, monkeypatch, args, frame=None):
    _patch_finalize_collaborators(module, monkeypatch, frame)
    with pytest.raises(SystemExit) as exc:
        module.do_finalize(args)
    report = (
        json.loads(args.gate_report.read_text()) if args.gate_report.exists() else None
    )
    return str(exc.value), report


def _write_frame_h5(path: Path, frame) -> None:
    """Write a staging frame as the tool's loader reads it (fixed format)."""

    from microcosm.frame import put_frame_table

    with pd.HDFStore(path, mode="w") as store:
        for entity in frame.entities:
            table = frame.table(entity).copy()
            if entity == "household":
                table["household_weight"] = frame.weights_for(entity).values
            put_frame_table(store, entity, table, preferred_format="fixed")


def _plausible_hours_frame():
    return _staging_frame_with_hours(
        [40.0, 38.0, 20.0, 45.0, 0.0, 0.0, 0.0, 0.0],
        [40.0, 35.0, 22.0, 40.0, 0.0, 0.0, 0.0, 5.0],
    )


def test_do_finalize_hard_fails_on_constant_forty_hours(tmp_path, monkeypatch) -> None:
    # microcosm#765: an artifact whose usual weekly hours are the engine's
    # constant-40 default must block packaging via the finalize hard gate.
    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    frame = _staging_frame_with_hours([40.0] * 8, [40.0] * 8)
    message, report = _run_finalize(module, monkeypatch, args, frame)
    assert "hours_worked_signal" in message
    assert report["gates"]["hours_worked_signal"]["passed"] is False


def test_do_finalize_hours_gate_passes_on_plausible_surface(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    weekly = [40.0, 38.0, 20.0, 45.0, 0.0, 0.0, 0.0, 0.0]
    last_week = [40.0, 35.0, 22.0, 40.0, 0.0, 0.0, 0.0, 5.0]
    frame = _staging_frame_with_hours(weekly, last_week)
    _message, report = _run_finalize(module, monkeypatch, args, frame)
    assert report["gates"]["hours_worked_signal"]["passed"] is True


def test_do_finalize_hard_fails_on_a_failed_take_up_gate(tmp_path, monkeypatch):
    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _patch_finalize_collaborators(module, monkeypatch, _plausible_hours_frame())
    _stub_local_take_up_gate(module, monkeypatch, passed=False)
    with pytest.raises(SystemExit) as exc:
        module.do_finalize(args)
    assert "acs_local_take_up_signal" in str(exc.value)
    report = json.loads(args.gate_report.read_text())
    gate = report["gates"]["acs_local_take_up_signal"]
    assert gate["passed"] is False
    assert gate["failures"] == ["acs_2024_1yr: invented constant take-up"]
    assert gate["artifact_sha256"] == module._sha256(args.out_h5)
    summary = json.loads(args.out_summary.read_text())
    assert "acs_local_take_up_signal" in summary["simulation_readiness_blockers"]


def test_do_finalize_take_up_gate_fails_on_default_filled_acs_rows(
    tmp_path, monkeypatch
):
    """The 2026-09-23 release signature: every ACS unit takes SNAP/TANF up."""

    module = _load_tool_module()
    real_gate = module.acs_local_take_up_signal_gate
    args = _finalize_args(module, tmp_path)
    frame = _take_up_gate_frame(
        spm_columns={
            "takes_up_snap_if_eligible": [True, True, True, False] + [True] * 4,
            "takes_up_tanf_if_eligible": [True, False, False, False] + [True] * 4,
        }
    )
    _patch_finalize_collaborators(module, monkeypatch, frame)
    monkeypatch.setattr(module, "acs_local_take_up_signal_gate", real_gate)
    with pytest.raises(SystemExit, match="acs_local_take_up_signal"):
        module.do_finalize(args)
    gate = json.loads(args.gate_report.read_text())["gates"]["acs_local_take_up_signal"]
    assert gate["passed"] is False
    assert all(failure.startswith("acs_2024_1yr:") for failure in gate["failures"])
    assert any(
        "takes_up_snap_if_eligible is constant" in failure
        for failure in gate["failures"]
    )


def _take_up_gate_frame(*, spm_columns=(), person_columns=()):
    """Four donor then four ACS single-person units carrying every column the
    real take-up gate reads, including the #1022 engine-free fills."""

    from microcosm.build.us_runtime.base_pool import spine_column
    from microcosm.frame import Frame

    frame = _plausible_hours_frame()
    spines = ["asec_puf"] * 4 + ["acs_2024_1yr"] * 4
    housing = [True, False, False, False, True, False, False, False]
    spm_unit = frame.table("spm_unit").assign(
        **{
            spine_column("spm_unit"): spines,
            "takes_up_snap_if_eligible": [True, True, True, False] * 2,
            "takes_up_tanf_if_eligible": [True, False, False, False] * 2,
            "receives_snap": [True] + [False] * 7,
            "takes_up_housing_assistance_if_eligible": housing,
            "receives_housing_assistance": housing,
            **dict(spm_columns),
        }
    )
    person = frame.table("person").assign(
        **{
            spine_column("person"): spines,
            "age": [30.0, 70.0, 40.0, 10.0, 30.0, 70.0, 40.0, 10.0],
            "HINS3": [np.nan] * 4 + [2, 1, 2, 2],
            "is_snap_abawd_discretionary_exempt": [True, False, False, False] * 2,
            "takes_up_medicare_if_eligible": [False, True, False, False] * 2,
            **dict(person_columns),
        }
    )
    household = frame.table("household").assign(
        **{spine_column("household"): spines, "TYPEHUGQ": [np.nan] * 4 + [1.0] * 4}
    )
    replaced = {"spm_unit": spm_unit, "person": person, "household": household}
    return Frame(
        {
            entity: replaced.get(entity, frame.table(entity))
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def test_do_finalize_take_up_gate_fails_on_engine_default_medicare(
    tmp_path, monkeypatch
):
    """microcosm#1022: every ACS person enrolled in Medicare (the engine
    default) blocks simulation readiness through the real gate."""

    module = _load_tool_module()
    real_gate = module.acs_local_take_up_signal_gate
    args = _finalize_args(module, tmp_path)
    frame = _take_up_gate_frame(
        person_columns={
            "takes_up_medicare_if_eligible": [False, True, False, False] + [True] * 4
        }
    )
    _patch_finalize_collaborators(module, monkeypatch, frame)
    monkeypatch.setattr(module, "acs_local_take_up_signal_gate", real_gate)
    with pytest.raises(SystemExit, match="acs_local_take_up_signal"):
        module.do_finalize(args)
    gate = json.loads(args.gate_report.read_text())["gates"]["acs_local_take_up_signal"]
    assert gate["passed"] is False
    assert any(
        "acs_2024_1yr: takes_up_medicare_if_eligible is constant" in failure
        for failure in gate["failures"]
    )
    assert any(
        "takes_up_medicare_if_eligible differs from ACS HINS3 == 1 on 3" in failure
        for failure in gate["failures"]
    )
    assert not any(failure.startswith("asec_puf") for failure in gate["failures"])


def test_do_finalize_hard_fails_on_a_failed_immigration_gate(tmp_path, monkeypatch):
    """microcosm#1020: a failed immigration gate blocks simulation readiness
    and is recorded bound to the evaluated bytes, like the take-up gate."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _patch_finalize_collaborators(module, monkeypatch, _plausible_hours_frame())
    _stub_local_immigration_gate(module, monkeypatch, passed=False)
    with pytest.raises(SystemExit) as exc:
        module.do_finalize(args)
    assert "acs_local_immigration_signal" in str(exc.value)
    report = json.loads(args.gate_report.read_text())
    gate = report["gates"]["acs_local_immigration_signal"]
    assert gate["passed"] is False
    assert gate["failures"] == ["acs_2024_1yr: ssn_card_type is constant 'CITIZEN'"]
    assert gate["artifact_sha256"] == module._sha256(args.out_h5)
    summary = json.loads(args.out_summary.read_text())
    assert summary["simulation_ready"] is False
    assert "acs_local_immigration_signal" in summary["simulation_readiness_blockers"]


def test_do_finalize_immigration_gate_fails_on_default_filled_acs_rows(
    tmp_path, monkeypatch
):
    """The 2026-09-23 release signature: every ACS person a CITIZEN with a
    citizen SSN card, every entry clock the engine default 5."""

    from microcosm.build.us_runtime.base_pool import spine_column
    from microcosm.frame import Frame

    module = _load_tool_module()
    real_gate = module.acs_local_immigration_signal_gate
    args = _finalize_args(module, tmp_path)
    frame = _plausible_hours_frame()
    ssn = ["CITIZEN", "CITIZEN", "OTHER_NON_CITIZEN", "NONE"] + ["CITIZEN"] * 4
    status = [
        "CITIZEN",
        "CITIZEN",
        "LEGAL_PERMANENT_RESIDENT",
        "UNDOCUMENTED",
    ] + ["CITIZEN"] * 4
    person = frame.table("person").assign(
        **{
            spine_column("person"): ["asec_puf"] * 4 + ["acs_2024_1yr"] * 4,
            "ssn_card_type": ssn,
            "immigration_status_str": status,
            "years_since_us_entry": [5.0] * 8,
            # Measured ACS citizenship: one ACS non-citizen the default fill
            # coded a citizen.
            "CIT": [np.nan] * 4 + [1, 1, 4, 5],
        }
    )
    frame = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    _patch_finalize_collaborators(module, monkeypatch, frame)
    monkeypatch.setattr(module, "acs_local_immigration_signal_gate", real_gate)
    with pytest.raises(SystemExit, match="acs_local_immigration_signal"):
        module.do_finalize(args)
    gate = json.loads(args.gate_report.read_text())["gates"][
        "acs_local_immigration_signal"
    ]
    assert gate["passed"] is False
    assert gate["artifact_sha256"] == module._sha256(args.out_h5)
    failures = gate["failures"]
    for column in ("ssn_card_type", "immigration_status_str"):
        assert f"acs_2024_1yr: {column} is constant 'CITIZEN'" in " ".join(failures)
    for spine in ("asec_puf", "acs_2024_1yr"):
        assert (
            f"{spine}: years_since_us_entry is the engine default 5 on every row."
            in failures
        )
    assert (
        "acs_2024_1yr: 1 person(s) are labelled against their measured CIT "
        "citizenship." in failures
    )
    # The donor spine's labels vary, so only its entry clock is refused.
    assert not any(
        failure.startswith("asec_puf:") and "years_since_us_entry" not in failure
        for failure in failures
    )


def test_do_finalize_hard_fails_on_a_failed_income_transfer_gate(tmp_path, monkeypatch):
    """microcosm#1022: a failed income transfer gate blocks simulation
    readiness and is recorded bound to the evaluated bytes."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _patch_finalize_collaborators(module, monkeypatch, _plausible_hours_frame())
    _stub_local_income_gate(module, monkeypatch, passed=False)
    with pytest.raises(SystemExit) as exc:
        module.do_finalize(args)
    assert "acs_local_income_transfer_signal" in str(exc.value)
    report = json.loads(args.gate_report.read_text())
    gate = report["gates"]["acs_local_income_transfer_signal"]
    assert gate["passed"] is False
    assert gate["artifact_sha256"] == module._sha256(args.out_h5)
    summary = json.loads(args.out_summary.read_text())
    assert summary["simulation_ready"] is False
    assert (
        "acs_local_income_transfer_signal" in summary["simulation_readiness_blockers"]
    )


@requires_pytables
@pytest.mark.parametrize("income_state", ["missing", "failed", "truthy", "stale"])
def test_package_requires_a_current_income_transfer_gate(
    tmp_path, monkeypatch, income_state
):
    """microcosm#1022: a report finalized before the income transfer gate, or
    failing it, or bound to other bytes, cannot package."""

    module = _load_tool_module()
    args = _package_args_with_hours(
        module,
        tmp_path,
        monkeypatch,
        gate_state="passed",
        income_state=income_state,
    )
    with pytest.raises(SystemExit, match="acs_local_income_transfer_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not list((args.out / "releases").rglob("*.json"))
    assert not (args.out / module.ARTIFACT_FILENAME).exists()


def test_do_finalize_hard_fails_on_a_failed_ssi_disability_gate(tmp_path, monkeypatch):
    """microcosm#1022: a failed SSI disability gate blocks simulation
    readiness and is recorded bound to the evaluated bytes."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _patch_finalize_collaborators(module, monkeypatch, _plausible_hours_frame())
    _stub_local_ssi_disability_gate(module, monkeypatch, passed=False)
    with pytest.raises(SystemExit) as exc:
        module.do_finalize(args)
    assert "acs_local_ssi_disability_signal" in str(exc.value)
    report = json.loads(args.gate_report.read_text())
    gate = report["gates"]["acs_local_ssi_disability_signal"]
    assert gate["passed"] is False
    assert gate["artifact_sha256"] == module._sha256(args.out_h5)
    summary = json.loads(args.out_summary.read_text())
    assert summary["simulation_ready"] is False
    assert "acs_local_ssi_disability_signal" in summary["simulation_readiness_blockers"]


@requires_pytables
@pytest.mark.parametrize(
    "ssi_disability_state", ["missing", "failed", "truthy", "stale"]
)
def test_package_requires_a_current_ssi_disability_gate(
    tmp_path, monkeypatch, ssi_disability_state
):
    """microcosm#1022: a report finalized before the SSI disability gate, or
    failing it, or bound to other bytes, cannot package."""

    module = _load_tool_module()
    args = _package_args_with_hours(
        module,
        tmp_path,
        monkeypatch,
        gate_state="passed",
        ssi_disability_state=ssi_disability_state,
    )
    with pytest.raises(SystemExit, match="acs_local_ssi_disability_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not list((args.out / "releases").rglob("*.json"))
    assert not (args.out / module.ARTIFACT_FILENAME).exists()


def test_do_finalize_hard_fails_on_a_failed_ssi_medicaid_take_up_gate(
    tmp_path, monkeypatch
):
    """microcosm#1022: a failed SSI/Medicaid take-up gate, graded against the
    materialize receipt in the run identity, blocks simulation readiness and
    is recorded bound to the evaluated bytes."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _patch_finalize_collaborators(module, monkeypatch, _plausible_hours_frame())
    receipts: list = []
    _stub_local_ssi_medicaid_take_up_gate(
        module, monkeypatch, passed=False, receipts=receipts
    )
    recorded = {"issue": "microcosm#1022", "marker": "run identity"}
    monkeypatch.setattr(
        module,
        "_verify_run_identity",
        lambda a: {
            "ladder_sha256": module._sha256(a.ladder),
            "acs_local_ssi_medicaid_take_up": recorded,
        },
    )
    with pytest.raises(SystemExit) as exc:
        module.do_finalize(args)
    assert "acs_local_ssi_medicaid_take_up_signal" in str(exc.value)
    assert receipts == [recorded]
    report = json.loads(args.gate_report.read_text())
    gate = report["gates"]["acs_local_ssi_medicaid_take_up_signal"]
    assert gate["passed"] is False
    assert gate["artifact_sha256"] == module._sha256(args.out_h5)
    summary = json.loads(args.out_summary.read_text())
    assert summary["simulation_ready"] is False
    assert (
        "acs_local_ssi_medicaid_take_up_signal"
        in summary["simulation_readiness_blockers"]
    )


@requires_pytables
@pytest.mark.parametrize("ssi_medicaid_state", ["missing", "failed", "truthy", "stale"])
def test_package_requires_a_current_ssi_medicaid_take_up_gate(
    tmp_path, monkeypatch, ssi_medicaid_state
):
    """microcosm#1022: a report finalized before the SSI/Medicaid take-up
    gate, or failing it, or bound to other bytes, cannot package."""

    module = _load_tool_module()
    args = _package_args_with_hours(
        module,
        tmp_path,
        monkeypatch,
        gate_state="passed",
        ssi_medicaid_state=ssi_medicaid_state,
    )
    with pytest.raises(SystemExit, match="acs_local_ssi_medicaid_take_up_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not list((args.out / "releases").rglob("*.json"))
    assert not (args.out / module.ARTIFACT_FILENAME).exists()


def _block_evidence_args(module, tmp_path, monkeypatch, *, acs_true_rows: int):
    args = _package_evidence_args(
        module,
        tmp_path,
        monkeypatch,
        hours_report={"passed": True, "failures": [], "detail": {}},
    )
    path = tmp_path / "staging.summary.json"
    summary = json.loads(path.read_text())
    summary["acs_local_ssi_disability"] = _staged_ssi_criteria(acs_true_rows)
    path.write_text(json.dumps(summary))
    return args


def test_package_blocks_criteria_positive_acs_rows_without_ssi_take_up(
    tmp_path, monkeypatch
):
    """microcosm#1022 (review of #1058): the criteria stage makes ACS persons
    SSI-eligible, and universal take-up would pay every one of them."""

    module = _load_tool_module()
    args = _block_evidence_args(module, tmp_path, monkeypatch, acs_true_rows=12)
    with pytest.raises(SystemExit) as exc:
        module.do_package(args)
    message = str(exc.value)
    assert "12 ACS person(s) meet meets_ssi_disability_criteria" in message
    assert "no SSI take-up handling" in message
    assert "microcosm#1022" in message
    assert "PR #1060" in message
    assert not (args.out / "releases").exists(), "a blocked release leaves nothing"


def test_package_records_the_ssi_take_up_block_when_no_acs_row_is_positive(
    tmp_path, monkeypatch
):
    module = _load_tool_module()
    args = _block_evidence_args(module, tmp_path, monkeypatch, acs_true_rows=0)
    result = module.do_package(args)
    manifest = json.loads(
        (Path(result["release_dir"]) / "build_manifest.json").read_text()
    )
    assert manifest["ssi_take_up_release_block"] == {
        "criteria_positive_acs_rows": 0,
        "filled_true_rows": 0,
        "take_up_handling": None,
    }


def test_recorded_ssi_take_up_handling_lifts_the_block(tmp_path, monkeypatch):
    module = _load_tool_module()
    handling = {"stage": "invented_ssi_take_up"}
    monkeypatch.setattr(
        module, "_recorded_ssi_take_up_handling", lambda identity, ckpt: handling
    )
    block = module._require_ssi_take_up_handling(
        {"acs_local_ssi_disability": _staged_ssi_criteria(12)}, {}, tmp_path
    )
    assert block == {
        "criteria_positive_acs_rows": 12,
        "filled_true_rows": 12,
        "take_up_handling": handling,
    }


def _with_recorded_ssi_medicaid_take_up(module, args, **overrides) -> dict:
    """Record a current SSI/Medicaid take-up receipt in the run identity."""

    path = args.checkpoint_dir / "run_identity.json"
    identity = json.loads(path.read_text())
    receipt = _ssi_medicaid_receipt(module, args.checkpoint_dir, **overrides)
    identity["acs_local_ssi_medicaid_take_up"] = receipt
    path.write_text(json.dumps(identity))
    return receipt


def test_the_ssi_medicaid_take_up_receipt_lifts_the_ssi_take_up_block(
    tmp_path, monkeypatch
):
    """microcosm#1022: with ACS SSI take-up assigned, criteria-positive ACS
    persons may ship; the manifest names the handling that let them."""

    module = _load_tool_module()
    args = _block_evidence_args(module, tmp_path, monkeypatch, acs_true_rows=12)
    receipt = _with_recorded_ssi_medicaid_take_up(module, args)

    result = module.do_package(args)

    manifest = json.loads(
        (Path(result["release_dir"]) / "build_manifest.json").read_text()
    )
    assert manifest["ssi_take_up_release_block"] == {
        "criteria_positive_acs_rows": 12,
        "filled_true_rows": 12,
        "take_up_handling": {
            "stage": "acs_local_ssi_medicaid_take_up",
            "issue": "microcosm#1022",
            "method": module.ACS_LOCAL_SSI_MEDICAID_TAKE_UP_METHOD,
            "assigned_sha256": receipt["assigned_sha256"],
            "assignment_file": receipt["assignment_file"],
            "finalize_gate": "acs_local_ssi_medicaid_take_up_signal",
        },
    }


@pytest.mark.parametrize("edit", ["failed", "changed-file"])
def test_an_invalid_ssi_medicaid_take_up_receipt_does_not_lift_the_block(
    tmp_path, monkeypatch, edit
):
    module = _load_tool_module()
    args = _block_evidence_args(module, tmp_path, monkeypatch, acs_true_rows=12)
    if edit == "failed":
        _with_recorded_ssi_medicaid_take_up(
            module, args, gate={"passed": False, "failures": ["x"]}
        )
        message = "records no passing ACS local SSI/Medicaid take-up"
    else:
        _with_recorded_ssi_medicaid_take_up(module, args)
        (args.checkpoint_dir / module.ACS_SSI_MEDICAID_TAKE_UP_FILENAME).write_bytes(
            b"edited after materialize"
        )
        message = "missing or changed since materialize"
    with pytest.raises(SystemExit, match=message):
        module.do_package(args)
    assert not (args.out / "releases").exists(), "a blocked release leaves nothing"


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        _ssi_disability_receipt(),
        _ssi_disability_receipt(outcome=None),
        _ssi_disability_receipt(outcome={"acs_true_rows": "12"}),
        _ssi_disability_receipt(outcome={"acs_true_rows": True}),
        _ssi_disability_receipt(outcome={"acs_true_rows": -1}),
    ],
    ids=["no-receipt", "no-outcome", "null-outcome", "string", "bool", "negative"],
)
def test_ssi_take_up_block_refuses_a_summary_without_the_criteria_count(
    tmp_path, receipt
):
    module = _load_tool_module()
    with pytest.raises(SystemExit, match=r"outcome\.acs_true_rows"):
        module._require_ssi_take_up_handling(
            {"acs_local_ssi_disability": receipt}, {}, tmp_path
        )


def test_do_finalize_hard_fails_on_a_failed_work_disability_gate(tmp_path, monkeypatch):
    """microcosm#1021: a failed work/disability gate blocks simulation
    readiness and is recorded bound to the evaluated bytes."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _patch_finalize_collaborators(module, monkeypatch, _plausible_hours_frame())
    _stub_local_work_disability_gate(module, monkeypatch, passed=False)
    with pytest.raises(SystemExit) as exc:
        module.do_finalize(args)
    assert "acs_local_work_disability_signal" in str(exc.value)
    report = json.loads(args.gate_report.read_text())
    gate = report["gates"]["acs_local_work_disability_signal"]
    assert gate["passed"] is False
    assert gate["failures"] == ["acs_2024_1yr: is_disabled is constant False"]
    assert gate["artifact_sha256"] == module._sha256(args.out_h5)
    summary = json.loads(args.out_summary.read_text())
    assert summary["simulation_ready"] is False
    assert (
        "acs_local_work_disability_signal" in summary["simulation_readiness_blockers"]
    )
    assert (
        "acs_local_work_disability_signal"
        in (report["gates"]["input_coverage"]["note"])
    )


def test_do_finalize_grades_the_staging_receipt_without_weeks(tmp_path, monkeypatch):
    """Finalize hands the gate the staging receipt and skips weeks_worked,
    which the consumer export holds back until policyengine-us#9660."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    summary_path = tmp_path / "staging.summary.json"
    staging = json.loads(summary_path.read_text())
    receipt = _work_disability_receipt()
    summary_path.write_text(
        json.dumps({**staging, "acs_local_work_disability": receipt})
    )
    _patch_finalize_collaborators(module, monkeypatch, _plausible_hours_frame())
    calls: list = []
    _stub_local_work_disability_gate(module, monkeypatch, calls=calls)
    with pytest.raises(SystemExit):  # consumer_ready: no export in this fixture
        module.do_finalize(args)
    assert calls == [(receipt, False)]
    gate = json.loads(args.gate_report.read_text())["gates"][
        "acs_local_work_disability_signal"
    ]
    assert gate["passed"] is True


def test_do_finalize_work_disability_gate_fails_on_imputed_acs_flags(
    tmp_path, monkeypatch
):
    """The pre-#1021 signature on the packaged bytes: ACS flags that are not
    the native difficulty items (here the engine default False on every ACS
    row) and a staging summary with no native receipt."""

    from microcosm.build.us_runtime.base_pool import spine_column
    from microcosm.frame import Frame

    module = _load_tool_module()
    real_gate = module.acs_local_work_disability_signal_gate
    args = _finalize_args(module, tmp_path)
    frame = _plausible_hours_frame()
    blank = [np.nan] * 4
    person = frame.table("person").assign(
        **{
            spine_column("person"): ["asec_puf"] * 4 + ["acs_2024_1yr"] * 4,
            "is_disabled": [False, True, False, True] + [False] * 4,
            "is_blind": [False, False, True, False] + [False] * 4,
            "AGEP": blank + [40, 30, 50, 8],
            "SSIP": blank + [0.0, 0.0, 0.0, np.nan],
            "DEAR": blank + [2, 1, 2, 2],
            "DEYE": blank + [2, 2, 1, 2],
            "DREM": blank + [2, 2, 2, 2],
            "DPHY": blank + [2, 2, 2, 2],
            "DDRS": blank + [2, 2, 2, 2],
            "DOUT": blank + [2, 2, 2, np.nan],
        }
    )
    frame = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    _patch_finalize_collaborators(module, monkeypatch, frame)
    monkeypatch.setattr(module, "acs_local_work_disability_signal_gate", real_gate)
    with pytest.raises(SystemExit, match="acs_local_work_disability_signal"):
        module.do_finalize(args)
    gate = json.loads(args.gate_report.read_text())["gates"][
        "acs_local_work_disability_signal"
    ]
    assert gate["passed"] is False
    assert gate["artifact_sha256"] == module._sha256(args.out_h5)
    failures = gate["failures"]
    for column in ("is_disabled", "is_blind"):
        assert (
            f"acs_2024_1yr: {column} is constant False; a spine with no variation "
            "carries no disability signal."
        ) in failures
    assert (
        "acs_2024_1yr: 2 person(s) carry a is_disabled that differs from their "
        "native ACS items; the ACS value must be measured, not imputed."
    ) in failures
    assert any("No acs_local_work_disability staging receipt" in f for f in failures)
    # weeks_worked is not graded at finalize, and the donor flags vary.
    assert not any("weeks_worked" in failure for failure in failures)
    assert not any(failure.startswith("asec_puf:") for failure in failures)


@requires_pytables
def test_finalize_binds_the_hours_gate_to_the_calibrated_artifact_bytes(
    tmp_path, monkeypatch
) -> None:
    """The gate entry must carry the digest of the bytes it evaluated."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _write_frame_h5(args.out_h5, _plausible_hours_frame())
    hashed = []
    original_sha = module._sha256

    def recording_sha(path):
        result = original_sha(path)
        if Path(path) == args.out_h5:
            hashed.append(result)
        return result

    monkeypatch.setattr(module, "_sha256", recording_sha)
    loads = []
    real_load = module._load_staging_frame
    monkeypatch.setattr(
        module,
        "_load_staging_frame",
        lambda p: (loads.append(len(hashed)), real_load(p))[1],
    )
    _message, report = _run_finalize(module, monkeypatch, args)
    gate = report["gates"]["hours_worked_signal"]
    assert gate["passed"] is True
    assert gate["artifact_sha256"] == original_sha(args.out_h5)
    # Hashed once before the frame was loaded, then re-checked after the gate.
    assert loads == [1]
    assert hashed == [gate["artifact_sha256"]] * 2


@requires_pytables
def test_finalize_refuses_artifact_changed_during_hours_validation(
    tmp_path, monkeypatch
) -> None:
    import microcosm.build.us_runtime.hours_worked as hours_worked

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _write_frame_h5(args.out_h5, _plausible_hours_frame())
    original_gate = hours_worked.us_hours_worked_signal_gate

    def changed_artifact(frame, **kwargs):
        result = original_gate(frame, **kwargs)
        args.out_h5.write_bytes(b"different-invented-artifact")
        return result

    monkeypatch.setattr(hours_worked, "us_hours_worked_signal_gate", changed_artifact)
    message, report = _run_finalize(module, monkeypatch, args)
    assert "changed during hours_worked_signal validation" in message
    assert report is None
    assert not args.out_summary.exists()


@requires_pytables
def test_finalize_report_round_trips_into_package(tmp_path, monkeypatch) -> None:
    """A finalize-written report must satisfy the package stage's binding."""

    module = _load_tool_module()
    args = _finalize_args(module, tmp_path)
    _write_frame_h5(args.out_h5, _plausible_hours_frame())
    artifact_sha = module._sha256(args.out_h5)
    evidence = {
        "run_identity.json": {
            "staging_sha256": module._sha256(args.staging_h5),
            "ladder_sha256": module._sha256(args.ladder),
            "population_cells_dropped": [],
        },
        "spine_qa.json": {
            "plain_consumption": True,
            "artifact_sha256": artifact_sha,
            "per_spine": {},
        },
        "consumer_export.json": {"staging_sha256": artifact_sha},
        "consumer_reviewed_null_fills.json": {"columns_filled": []},
    }
    for name, value in evidence.items():
        (args.checkpoint_dir / name).write_text(json.dumps(value))
    _patch_finalize_collaborators(module, monkeypatch, identity=False)
    module.do_finalize(args)
    report = json.loads(args.gate_report.read_text())
    assert report["gates"]["hours_worked_signal"]["artifact_sha256"] == artifact_sha
    assert json.loads(args.out_summary.read_text())["simulation_ready"] is True

    args.out = tmp_path / "release"
    args.allow_dirty = True
    result = module.do_package(args)
    shipped = json.loads(
        (Path(result["release_dir"]) / "gate_summary.json").read_text()
    )["gates"]["hours_worked_signal"]
    assert shipped["passed"] is True
    assert (
        shipped["artifact_sha256"]
        == result["root_artifact"]["sha256"]
        == module._sha256(Path(result["root_artifact"]["local_path"]))
        == artifact_sha
    )


def _package_args_with_hours(
    module,
    tmp_path,
    monkeypatch,
    *,
    gate_state,
    take_up_state="passed",
    immigration_state="passed",
    work_disability_state="passed",
    income_state="passed",
    ssi_disability_state="passed",
    ssi_medicaid_state="passed",
):
    """Real tiny H5 bytes; gate edits model stale separately resumed finalize."""

    _stub_local_hours_gate(module, monkeypatch)
    args = _finalize_args(module, tmp_path)
    args.out = tmp_path / "release"
    args.allow_dirty = True
    _write_frame_h5(args.out_h5, _plausible_hours_frame())
    artifact_sha = module._sha256(args.out_h5)
    gate = {"passed": True, "failures": [], "artifact_sha256": artifact_sha}
    if gate_state == "failed":
        gate.update(passed=False, failures=["invented hours failure"])
    elif gate_state == "truthy":
        gate["passed"] = "true"
    elif gate_state == "unbound":
        del gate["artifact_sha256"]
    elif gate_state == "stale":
        gate["artifact_sha256"] = "0" * 64
    local_gate = {
        "passed": True,
        "failures": [],
        "detail": {},
        "artifact_sha256": artifact_sha,
    }
    gates = (
        {}
        if gate_state == "missing"
        else {"hours_worked_signal": gate, "acs_local_hours_signal": local_gate}
    )
    take_up_gate = {"passed": True, "failures": [], "artifact_sha256": artifact_sha}
    if take_up_state == "failed":
        take_up_gate.update(passed=False, failures=["invented take-up failure"])
    elif take_up_state == "stale":
        take_up_gate["artifact_sha256"] = "0" * 64
    if gates and take_up_state != "missing":
        gates["acs_local_take_up_signal"] = take_up_gate
    immigration_gate = {
        "passed": True,
        "failures": [],
        "artifact_sha256": artifact_sha,
    }
    if immigration_state == "failed":
        immigration_gate.update(passed=False, failures=["invented CITIZEN spine"])
    elif immigration_state == "truthy":
        immigration_gate["passed"] = "true"
    elif immigration_state == "stale":
        immigration_gate["artifact_sha256"] = "0" * 64
    if gates and immigration_state != "missing":
        gates["acs_local_immigration_signal"] = immigration_gate
    work_disability_gate = {
        "passed": True,
        "failures": [],
        "artifact_sha256": artifact_sha,
    }
    if work_disability_state == "failed":
        work_disability_gate.update(passed=False, failures=["invented imputed flag"])
    elif work_disability_state == "truthy":
        work_disability_gate["passed"] = "true"
    elif work_disability_state == "stale":
        work_disability_gate["artifact_sha256"] = "0" * 64
    if gates and work_disability_state != "missing":
        gates["acs_local_work_disability_signal"] = work_disability_gate
    income_gate = {"passed": True, "failures": [], "artifact_sha256": artifact_sha}
    if income_state == "failed":
        income_gate.update(passed=False, failures=["invented missing income"])
    elif income_state == "truthy":
        income_gate["passed"] = "true"
    elif income_state == "stale":
        income_gate["artifact_sha256"] = "0" * 64
    if gates and income_state != "missing":
        gates["acs_local_income_transfer_signal"] = income_gate
    ssi_gate = {"passed": True, "failures": [], "artifact_sha256": artifact_sha}
    if ssi_disability_state == "failed":
        ssi_gate.update(passed=False, failures=["invented default-filled criteria"])
    elif ssi_disability_state == "truthy":
        ssi_gate["passed"] = "true"
    elif ssi_disability_state == "stale":
        ssi_gate["artifact_sha256"] = "0" * 64
    if gates and ssi_disability_state != "missing":
        gates["acs_local_ssi_disability_signal"] = ssi_gate
    ssi_medicaid_gate = {
        "passed": True,
        "failures": [],
        "artifact_sha256": artifact_sha,
    }
    if ssi_medicaid_state == "failed":
        ssi_medicaid_gate.update(passed=False, failures=["invented universal take-up"])
    elif ssi_medicaid_state == "truthy":
        ssi_medicaid_gate["passed"] = "true"
    elif ssi_medicaid_state == "stale":
        ssi_medicaid_gate["artifact_sha256"] = "0" * 64
    if gates and ssi_medicaid_state != "missing":
        gates["acs_local_ssi_medicaid_take_up_signal"] = ssi_medicaid_gate
    args.gate_report.write_text(json.dumps({"gates": gates}))
    args.out_summary.write_text(json.dumps({"simulation_ready": True}))
    evidence = {
        "run_identity.json": {
            "staging_sha256": module._sha256(args.staging_h5),
            "population_cells_dropped": [],
        },
        "spine_qa.json": {
            "plain_consumption": True,
            "artifact_sha256": artifact_sha,
            "per_spine": {},
        },
        "consumer_export.json": {"staging_sha256": artifact_sha},
        "consumer_reviewed_null_fills.json": {"columns_filled": []},
    }
    for name, value in evidence.items():
        (args.checkpoint_dir / name).write_text(json.dumps(value))
    return args


@requires_pytables
@pytest.mark.parametrize(
    "gate_state", ["missing", "failed", "truthy", "unbound", "stale"]
)
def test_package_requires_current_hours_gate_even_with_green_old_summary(
    tmp_path, monkeypatch, gate_state
):
    module = _load_tool_module()
    args = _package_args_with_hours(
        module, tmp_path, monkeypatch, gate_state=gate_state
    )
    with pytest.raises(SystemExit, match="hours_worked_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()


@requires_pytables
def test_package_accepts_passing_hours_gate_bound_to_the_packaged_bytes(
    tmp_path, monkeypatch
):
    module = _load_tool_module()
    args = _package_args_with_hours(module, tmp_path, monkeypatch, gate_state="passed")
    result = module.do_package(args)
    release_dir = Path(result["release_dir"])
    gate = json.loads((release_dir / "gate_summary.json").read_text())["gates"][
        "hours_worked_signal"
    ]
    copied_sha = module._sha256(Path(result["root_artifact"]["local_path"]))
    assert gate["passed"] is True
    assert gate["artifact_sha256"] == result["root_artifact"]["sha256"] == copied_sha
    take_up = json.loads((release_dir / "gate_summary.json").read_text())["gates"][
        "acs_local_take_up_signal"
    ]
    assert take_up["passed"] is True
    assert take_up["artifact_sha256"] == copied_sha
    immigration = json.loads((release_dir / "gate_summary.json").read_text())["gates"][
        "acs_local_immigration_signal"
    ]
    assert immigration["passed"] is True
    assert immigration["artifact_sha256"] == copied_sha
    work_disability = json.loads((release_dir / "gate_summary.json").read_text())[
        "gates"
    ]["acs_local_work_disability_signal"]
    assert work_disability["passed"] is True
    assert work_disability["artifact_sha256"] == copied_sha


@requires_pytables
@pytest.mark.parametrize(
    "work_disability_state", ["missing", "failed", "truthy", "stale"]
)
def test_package_requires_a_current_work_disability_gate(
    tmp_path, monkeypatch, work_disability_state
):
    """microcosm#1021: a report finalized before the work/disability gate, or
    failing it, or bound to other bytes, cannot package."""

    module = _load_tool_module()
    args = _package_args_with_hours(
        module,
        tmp_path,
        monkeypatch,
        gate_state="passed",
        work_disability_state=work_disability_state,
    )
    with pytest.raises(SystemExit, match="acs_local_work_disability_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not list((args.out / "releases").rglob("*.json"))
    assert not (args.out / module.ARTIFACT_FILENAME).exists()


@requires_pytables
@pytest.mark.parametrize("immigration_state", ["missing", "failed", "truthy", "stale"])
def test_package_requires_a_current_immigration_gate(
    tmp_path, monkeypatch, immigration_state
):
    """microcosm#1020: a report finalized before the immigration gate, or
    failing it, or bound to other bytes, cannot package."""

    module = _load_tool_module()
    args = _package_args_with_hours(
        module,
        tmp_path,
        monkeypatch,
        gate_state="passed",
        immigration_state=immigration_state,
    )
    with pytest.raises(SystemExit, match="acs_local_immigration_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not list((args.out / "releases").rglob("*.json"))
    assert not (args.out / module.ARTIFACT_FILENAME).exists()


@requires_pytables
@pytest.mark.parametrize("take_up_state", ["missing", "failed", "stale"])
def test_package_requires_a_current_take_up_gate(tmp_path, monkeypatch, take_up_state):
    """microcosm#1019: a report finalized before the take-up gate, or failing
    it, or bound to other bytes, cannot package."""

    module = _load_tool_module()
    args = _package_args_with_hours(
        module,
        tmp_path,
        monkeypatch,
        gate_state="passed",
        take_up_state=take_up_state,
    )
    with pytest.raises(SystemExit, match="acs_local_take_up_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not list((args.out / "releases").rglob("*.json"))


@requires_pytables
def test_package_records_the_stored_input_gate_bound_to_the_packaged_bytes(
    tmp_path, monkeypatch
):
    """microcosm#1026: the package stage grades the calibrated H5 it ships
    against the engine it records as built-with, and the gate summary and
    build manifest carry that verdict bound to the packaged bytes."""

    monkeypatch.setattr(
        stored_inputs, "require_h5_stored_inputs", _REAL_REQUIRE_H5_STORED_INPUTS
    )
    module = _load_tool_module()
    args = _package_args_with_hours(module, tmp_path, monkeypatch, gate_state="passed")
    result = module.do_package(args)

    release_dir = Path(result["release_dir"])
    for name in ("gate_summary.json", "build_manifest.json"):
        gate = json.loads((release_dir / name).read_text())["gates"]["stored_inputs"]
        assert gate == {
            "passed": True,
            "failures": [],
            "engine": "policyengine-us 2.2.1",
            "register_sha256": stored_inputs.register_sha256(),
            "registered_non_variables": [],
            "artifact_sha256": result["root_artifact"]["sha256"],
            "checked_at_stage": "package",
        }


def _write_lane_h5(path: Path, **person_columns) -> None:
    """The plausible-hours frame, written by the lane's own writer: the shared
    nullable writer, which adds the ``_populace_staging_metadata`` series."""

    from microcosm.frame import Frame

    staging = _load_staging_builder_module()
    frame = _plausible_hours_frame()
    person = frame.table("person").assign(**person_columns)
    frame = Frame(
        {
            **{entity: frame.table(entity) for entity in frame.entities},
            "person": person,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    staging._write_dataset(
        frame, path, period=2024, artifact_kind="calibrated_local_area_artifact"
    )


@requires_pytables
def test_the_contract_reads_the_lane_writer_output(tmp_path):
    """The lane writes through the shared nullable writer, whose H5 carries a
    ``_populace_staging_metadata`` series next to the entity tables. The
    contract reads that layout rather than refusing it."""

    path = tmp_path / "out.h5"
    _write_lane_h5(path, would_claim_wic=[True, False] * 4)

    tables = stored_inputs.h5_stored_tables(path)

    assert set(tables) == {
        "person",
        "household",
        "tax_unit",
        "spm_unit",
        "family",
        "marital_unit",
    }
    assert "would_claim_wic" in tables["person"]
    assert "household_weight" in tables["household"]
    import h5py

    with h5py.File(path, "r") as h5:
        assert "_populace_staging_metadata" in h5


@requires_pytables
def test_package_refuses_a_stale_stored_input_before_any_release_dir(
    tmp_path, monkeypatch
):
    """The #1026 regression in this lane: a calibrated H5 carrying the WIC
    draw under its retired name is refused by name, with both remedies, and
    no release directory is created."""

    monkeypatch.setattr(
        stored_inputs, "require_h5_stored_inputs", _REAL_REQUIRE_H5_STORED_INPUTS
    )
    module = _load_tool_module()
    args = _package_args_with_hours(module, tmp_path, monkeypatch, gate_state="passed")
    _write_lane_h5(args.out_h5, would_claim_wic=[True, False] * 4)

    with pytest.raises(SystemExit) as refusal:
        module.do_package(args)

    message = str(refusal.value)
    assert message.startswith(
        "Refusing to package: stored-input contract refused the release: "
        "stored column 'would_claim_wic' (person table)"
    )
    assert "policyengine-us 2.2.1" in message
    assert "Rename it to the live input" in message
    assert not (args.out / "releases").exists()
    assert not (args.out / "package_result.json").exists()

    _write_lane_h5(args.out_h5, takes_up_wic_if_eligible=[True, False] * 4)
    _refresh_package_evidence_bindings(module, args)
    result = module.do_package(args)
    assert Path(result["release_dir"]).is_dir()


def _refresh_package_evidence_bindings(module, args) -> None:
    """Rebind the package evidence to rewritten artifact bytes."""

    artifact_sha = module._sha256(args.out_h5)
    report = json.loads(args.gate_report.read_text())
    for gate in report["gates"].values():
        gate["artifact_sha256"] = artifact_sha
    args.gate_report.write_text(json.dumps(report))
    for name in ("spine_qa.json", "consumer_export.json"):
        path = args.checkpoint_dir / name
        evidence = json.loads(path.read_text())
        key = "artifact_sha256" if name == "spine_qa.json" else "staging_sha256"
        evidence[key] = artifact_sha
        path.write_text(json.dumps(evidence))


def test_package_refuses_without_an_importable_engine(tmp_path, monkeypatch):
    """No engine, no certification: the release would record a built-with
    engine the artifact was never checked against."""

    def missing():
        raise ImportError("No module named 'policyengine_us'")

    monkeypatch.setattr(stored_inputs, "installed_us_engine", missing)
    module = _load_tool_module()
    args = _package_args_before_evidence(module, tmp_path, max_households=None)

    with pytest.raises(SystemExit, match="policyengine-us cannot be imported"):
        module.do_package(args)
    assert not (args.out / "releases").exists()


@requires_pytables
@pytest.mark.parametrize("copy_state", ["new", "already_present", "no_copy"])
def test_package_rechecks_final_bytes_after_copy_or_reuse(
    tmp_path, monkeypatch, copy_state
):
    import shutil as real_shutil

    module = _load_tool_module()
    args = _package_args_with_hours(module, tmp_path, monkeypatch, gate_state="passed")
    root_copy = args.out / module.ARTIFACT_FILENAME
    original_sha = module._sha256
    root_hashes = []
    if copy_state == "new":

        def changed_copy(source, destination):
            result = real_shutil.copy2(source, destination)
            Path(destination).write_bytes(b"changed-during-copy")
            return result

        class _ToolShutil:
            """Rebind only the tool's own ``shutil`` name, not the global module."""

            copy2 = staticmethod(changed_copy)

            def __getattr__(self, name):
                return getattr(real_shutil, name)

        monkeypatch.setattr(module, "shutil", _ToolShutil())
    elif copy_state == "already_present":
        root_copy.parent.mkdir(parents=True)
        root_copy.write_bytes(args.out_h5.read_bytes())

        def changed_after_reuse_check(path):
            result = original_sha(path)
            if Path(path) == root_copy:
                root_hashes.append(result)
                if len(root_hashes) == 1:
                    root_copy.write_bytes(b"changed-after-reuse-check")
            return result

        monkeypatch.setattr(module, "_sha256", changed_after_reuse_check)
    else:
        # The artifact root IS the calibrated H5, so nothing is copied and only
        # the final re-hash can notice a source that changed after the gates.
        root_copy.parent.mkdir(parents=True)
        args.out_h5.rename(root_copy)
        args.out_h5 = root_copy
        releases = args.out / "releases"
        changed = []

        def changed_while_writing_sums(path):
            if not changed and releases in Path(path).parents:
                changed.append(path)
                with root_copy.open("ab") as stream:
                    stream.write(b"changed-after-the-gates")
            return original_sha(path)

        monkeypatch.setattr(module, "_sha256", changed_while_writing_sums)
    with pytest.raises(SystemExit, match="packaged H5.*hours_worked_signal"):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    if copy_state == "already_present":
        assert len(root_hashes) == 2, "the reuse check and the final re-hash"
    if copy_state == "no_copy":
        assert root_copy.exists(), "the calibrated H5 itself is never removed"
    else:
        assert not root_copy.exists(), "a refused copy is not left at the root"
