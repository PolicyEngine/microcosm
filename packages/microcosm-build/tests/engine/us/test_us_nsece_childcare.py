"""Synthetic childcare contracts for the engine/us environment."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_nsece_childcare import *


def test_real_engine_export_reload_preserves_child_attendance(tmp_path):
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    candidate = with_us_nsece_childcare_attendance(
        _frame(), _source(), seed=915, match_columns=("age",)
    )
    assert_childcare_attendance_exportable(candidate)
    # Project canonical engine inputs; keep source provenance in the checkpoint.
    tables = {e: candidate.table(e).copy() for e in candidate.entities}
    tables["person"] = tables["person"].drop(
        columns=[
            "person_source_id",
            *(f"{c}_source" for c in US_CHILDCARE_ATTENDANCE_COLUMNS),
        ]
    )
    projected = Frame(
        tables,
        candidate.schema,
        {"household": candidate.weights_for("household")},
        candidate.strata,
        mass_log=candidate.mass_log,
        metadata=candidate.metadata,
    )
    path = tmp_path / "engine.h5"
    PolicyEngineUSEngine().write_dataset(projected, path, period=2026)
    dataset = USSingleYearDataset(file_path=str(path))
    assert dataset.person[MONTH].tolist() == [0, 22]
    assert dataset.person[HOURS].tolist() == [0, 8]
    sim = Microsimulation(dataset=dataset)
    assert sim.calculate("childcare_hours_per_week", 2026).tolist() == [0, 40]


def test_native_export_explicitly_inherits_outside_baseline_and_preserves_parent(
    tmp_path,
):
    from policyengine_us.data import USSingleYearDataset

    from microcosm.build.us_runtime.childcare_attendance_stage import (
        export_native_childcare_candidate,
        inherit_outside_domain_attendance_baseline,
    )

    frame = _frame(parent_observed=False)
    tables = {e: frame.table(e).copy() for e in frame.entities}
    tables["household"]["household_weight"] = frame.weights_for("household").values
    parent = tmp_path / "parent.h5"
    USSingleYearDataset(**tables, time_period=2026).save(str(parent))
    original = parent.read_bytes()
    candidate = with_us_nsece_childcare_attendance(
        frame, _source(), seed=915, match_columns=("age",)
    )
    from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes

    filled = inherit_outside_domain_attendance_baseline(
        canonicalize_frame_string_dtypes(
            candidate, boundary="test_childcare_native_export"
        )
    )
    assert (
        filled.table("person").loc[0, f"{DAYS}_source"]
        == "inherited_engine_baseline_outside_age_0_12"
    )
    assert filled.table("person").loc[1, DAYS] == 5
    path = export_native_childcare_candidate(parent, filled, tmp_path / "candidate.h5")
    result = USSingleYearDataset(file_path=str(path))
    assert result.person[DAYS].tolist() == [0, 5]
    assert result.time_period == "2026"
    assert parent.read_bytes() == original


@pytest.mark.parametrize("repair_factor", [1.0, 2.0])
def test_registered_attendance_recipe_uses_real_transform_chain(
    monkeypatch, tmp_path, repair_factor
):
    import importlib.util

    from microcosm.build.us_runtime import childcare_attendance_stage as stage

    builder_path = REPOSITORY_ROOT / "tools" / "build_us_fiscal_refresh_release.py"
    builder_spec = importlib.util.spec_from_file_location(
        "attendance_fiscal_builder", builder_path
    )
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    frame = _asec_frame()
    frame.table("household")["household_source_id"] = "family"
    for column in (
        *builder.US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES.values(),
        "non_sch_d_capital_gains",
    ):
        frame.table("person")[column] = [1.0, 0.0]
    hh, cal = _raw()
    hh["HHC4_AGE_AT_USAGE_1"] = frame.table("person").loc[1, "age"] * 12
    hh["HH4_REGION"] = 1
    hh["HH4_PARWORK_STATUS"] = 2
    hh["HH4_ECON_INCOME_ANNUAL"] = 40000
    _care(cal, hours=3)
    source = derive_nsece_childcare(hh, cal)
    source.source_receipt["artifacts"] = stage.childcare_attendance_contract()[
        "artifacts"
    ]
    monkeypatch.setattr(stage, "load_nsece_childcare", lambda *args: source)
    result = stage.with_us_childcare_attendance_inputs(
        frame,
        household_tsv="synthetic",
        calendar_tsv="synthetic",
        asec_source_cache=None,
        seed=915,
        inherit_outside_domain_baseline=True,
    )
    assert result.table("person").loc[1, DAYS] == 1
    assert result.table("person").loc[1, HOURS] == 3
    assert (
        result.metadata["childcare_attendance_stage"]["stage"]
        == "nsece_childcare_attendance"
    )
    assert result.metadata["existing_receipt"] == "preserved"
    np.testing.assert_array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )
    assert_bound_childcare_attendance(result)
    options = dict(
        household_tsv="synthetic",
        calendar_tsv="synthetic",
        asec_source_cache=None,
        seed=915,
        inherit_outside_domain_baseline=True,
    )
    assert stage.with_us_childcare_attendance_inputs(result, **options) is result
    for changed in ({"seed": 916}, {"inherit_outside_domain_baseline": False}):
        with pytest.raises(ValueError, match="settings changed"):
            stage.with_us_childcare_attendance_inputs(result, **{**options, **changed})
    operations = result.metadata["childcare_attendance_stage"]["operations"]
    assert [op["operation"] for op in operations] == [
        "calendar_attendance",
        "fit_sibling_dependence",
        "regular_hours_schedule_bridge",
        "harmonize_asec_childcare_predictors",
        "joint_weighted_schedule_transfer",
        "inherit_engine_baseline",
    ]
    from policyengine_us.data import USSingleYearDataset

    from microcosm.build.us_runtime.l0_refit_export import load_us_frame

    tables = {e: result.table(e).copy() for e in result.entities}
    tables["household"]["household_weight"] = result.weights_for("household").values
    path = tmp_path / "final.h5"
    USSingleYearDataset(**tables, time_period=2026).save(str(path))
    evidence = stage.persist_native_childcare_receipt(path, result)
    assert evidence["retained_people"] == 2
    assert "rows" not in evidence
    assert_bound_childcare_attendance(load_us_frame(path))

    # Reusing a prepared native parent traverses these two repairs before the
    # attendance stage. Both changed and unchanged repairs must retain lineage.
    reused = builder._load_frame(path)
    before = reused.table("person")[list(US_CHILDCARE_ATTENDANCE_COLUMNS)].copy()
    ssa_targets = [
        SimpleNamespace(metadata={"target_role": role}, value=100.0 * repair_factor)
        for role in builder.US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES
    ]
    cgd_target = SimpleNamespace(
        name="irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount",
        metadata={"aged_to": "2024"},
        value=100.0 * repair_factor,
    )
    for repair, targets in (
        (builder._with_social_security_component_value_repair, ssa_targets),
        (builder._with_non_sch_d_cgd_value_repair, [cgd_target]),
    ):
        reused, repair_receipt = repair(reused, targets)
        assert repair_receipt["applied"] == (repair_factor != 1.0)
        builder._require_bound_childcare_attendance(reused)
    assert stage.with_us_childcare_attendance_inputs(reused, **options) is reused
    repaired_path = tmp_path / "repaired.h5"
    builder.PolicyEngineUSEngine().write_dataset(reused, repaired_path, period=2026)
    stage.persist_native_childcare_receipt(repaired_path, reused)
    reloaded = load_us_frame(repaired_path)
    assert_bound_childcare_attendance(reloaded)
    assert_frame_equal(
        reloaded.table("person")[list(US_CHILDCARE_ATTENDANCE_COLUMNS)], before
    )


def test_native_reload_preserves_binding_and_detects_tampering(tmp_path):
    from policyengine_us.data import USSingleYearDataset

    from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5
    from microcosm.build.us_runtime.l0_refit_export import load_us_frame

    candidate = _candidate()
    tables = {e: candidate.table(e).copy() for e in candidate.entities}
    tables["household"]["household_weight"] = candidate.weights_for("household").values
    # Native exports intentionally omit per-cell provenance columns.
    tables["person"] = tables["person"].drop(
        columns=[f"{c}_source" for c in US_CHILDCARE_ATTENDANCE_COLUMNS]
    )
    path = tmp_path / "native.h5"
    USSingleYearDataset(**tables, time_period=2026).save(str(path))
    for loader in (load_legacy_calibrated_us_h5, load_us_frame):
        with pytest.raises(ValueError, match="lack a bound receipt"):
            loader(path)
    stage._write_childcare_candidate_person_table(
        path, tables["person"], candidate.metadata
    )
    for loader in (load_legacy_calibrated_us_h5, load_us_frame):
        loaded = loader(path)
        assert_bound_childcare_attendance(loaded, require_stage=False)
    tables["person"].loc[1, DAYS] = 4
    stage._write_childcare_candidate_person_table(
        path, tables["person"], candidate.metadata
    )
    with pytest.raises(ValueError, match="differ from the source receipt"):
        restore_native_childcare_receipt(
            path, _replace(candidate, people=tables["person"], metadata={})
        )


def test_production_stage_refuses_unbound_existing_values(monkeypatch):
    monkeypatch.setattr(stage, "load_nsece_childcare", lambda *args: _source())
    frame = _asec_frame()
    frame.table("person").loc[1, DAYS] = 5
    with pytest.raises(ValueError, match="Pre-existing child attendance"):
        stage.with_us_childcare_attendance_inputs(
            frame,
            household_tsv="fake",
            calendar_tsv="fake",
            asec_source_cache=None,
            seed=915,
        )
