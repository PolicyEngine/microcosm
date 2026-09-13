"""Synthetic source-code and Frame/export contracts; no NSECE microdata in CI."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CALENDAR_BLOCKS,
    NSECE_CHILD_INDICES,
    NSECE_PROVIDER_INDICES,
    assert_childcare_attendance_exportable,
    derive_nsece_childcare,
    load_nsece_childcare,
    nsece_childcare_calendar_columns,
    nsece_childcare_household_columns,
    nsece_childcare_validation_report,
    with_us_nsece_childcare_attendance,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

MONTH, DAYS, HOURS = US_CHILDCARE_ATTENDANCE_COLUMNS


def _raw(n=1):
    household = pd.DataFrame(
        -9.0, index=range(n), columns=nsece_childcare_household_columns()
    )
    calendar = pd.DataFrame(
        -9.0, index=range(n), columns=nsece_childcare_calendar_columns()
    )
    for table in (household, calendar):
        table["HH4_METH_CASEID"] = np.arange(1, n + 1)
    household["HH4_METH_QUEXVERSION"] = 1
    household["HH4_REGION"] = 1
    household["HH4_PARWORK_STATUS"] = 2
    household["HH4_RPARENT"] = 1
    household["HH4_METH_WEIGHT"] = 100.0
    household["HH4_ECON_INCOME_ANNUAL"] = 40_000
    for child in NSECE_CHILD_INDICES:
        household[f"HHC4_METH_WEIGHT_{child}"] = np.nan
        household[f"HH4_MISSING_STATUS_CC_{child}"] = 0
    household["HHC4_AGE_AT_USAGE_1"] = 36
    household["HHC4_METH_WEIGHT_1"] = 100.0
    household["HH4_MISSING_STATUS_CC_1"] = 2
    for provider in NSECE_PROVIDER_INDICES:
        household[f"HH4_TYPEOFCARE_AGG_1_{provider}"] = -8
    household["HH4_TYPEOFCARE_AGG_1_1"] = 4
    for block in range(1, NSECE_CALENDAR_BLOCKS + 1):
        calendar[f"HH4_CHCAL_R_1_{block}"] = 0
    return household, calendar


def _care(calendar, *, row=0, day=0, hours=4, provider=1):
    start = day * 96 + 9 * 4 + 1
    columns = [f"HH4_CHCAL_R_1_{b}" for b in range(start, start + hours * 4)]
    calendar.loc[row, columns] = provider


def test_calendar_union_excludes_school_and_handles_multiple_providers():
    hh, cal = _raw()
    hh["HH4_TYPEOFCARE_AGG_1_2"] = 6  # K-8 school is not ECE attendance.
    hh["HH4_TYPEOFCARE_AGG_1_3"] = 3  # Unpaid care is still ECE.
    _care(cal, day=0, hours=4)
    _care(cal, day=1, hours=2, provider=3)
    _care(cal, day=2, hours=6, provider=2)
    before = (hh.copy(deep=True), cal.copy(deep=True))
    source = derive_nsece_childcare(hh, cal)
    child = source.children.iloc[0]
    assert child[DAYS] == 2
    assert child[HOURS] == 3
    assert child[MONTH] == 9
    assert child.ece_hours_per_week == 6
    assert child.ece_provider_count == 2
    assert source.weights.values.tolist() == [100]
    assert_frame_equal(hh, before[0])
    assert_frame_equal(cal, before[1])


@pytest.mark.parametrize(
    "status,code,expected",
    [
        (0, 0, "missing_calendar"),
        (1, 0, "partial_calendar"),
        (2, -1, "ambiguous_calendar"),
        (2, 97, "ambiguous_calendar"),
        (2, 77, "ambiguous_calendar"),
    ],
)
def test_missing_or_ambiguous_calendar_never_becomes_observed_zero(
    status, code, expected
):
    hh, cal = _raw()
    hh["HH4_MISSING_STATUS_CC_1"] = status
    cal.loc[0, "HH4_CHCAL_R_1_1"] = code
    source = derive_nsece_childcare(hh, cal)
    assert source.children.attendance_status.tolist() == [expected]
    assert source.children[list(US_CHILDCARE_ATTENDANCE_COLUMNS)].isna().all().all()


def test_complete_parental_care_is_a_weighted_zero_donor():
    hh, cal = _raw()
    source = derive_nsece_childcare(hh, cal)
    donors, weights = source.donors()
    assert donors[list(US_CHILDCARE_ATTENDANCE_COLUMNS)].to_numpy().tolist() == [
        [0, 0, 0]
    ]
    assert weights.total == 100


def test_unknown_provider_type_is_not_a_zero_schedule():
    hh, cal = _raw()
    _care(cal)
    hh["HH4_TYPEOFCARE_AGG_1_1"] = 8
    source = derive_nsece_childcare(hh, cal)
    assert source.children.attendance_status.tolist() == ["ambiguous_calendar"]


def test_age_in_months_and_calendar_join_are_not_row_order_dependent():
    hh, cal = _raw(2)
    hh.loc[1, "HHC4_AGE_AT_USAGE_1"] = 156
    _care(cal)
    source = derive_nsece_childcare(hh, cal.iloc[::-1])
    assert source.children.age.tolist() == [3, 13]
    assert source.children.attendance_status.tolist() == [
        "complete",
        "age_out_of_scope",
    ]
    assert source.children.loc[0, DAYS] == 1


def test_mismatched_or_duplicate_household_ids_raise():
    hh, cal = _raw(2)
    cal.loc[0, "HH4_METH_CASEID"] = 3
    with pytest.raises(ValueError, match="ID sets"):
        derive_nsece_childcare(hh, cal)
    cal.loc[0, "HH4_METH_CASEID"] = 2
    with pytest.raises(ValueError, match="unique"):
        derive_nsece_childcare(hh, cal)


def test_loader_refuses_unpinned_source(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text("wrong file\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_nsece_childcare(path, path)


def _frame(*, parent_observed=True):
    person = pd.DataFrame(
        {"person_id": [1, 2], "person_source_id": ["adult", "child"], "age": [30, 3]}
    )
    tables = {}
    for entity in US_SCHEMA.group_entities:
        person[f"person_{entity}_id"] = 1
        tables[entity] = pd.DataFrame({f"{entity}_id": [1]})
    if parent_observed:
        for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
            person[column] = [0.0, np.nan]
    tables["person"] = person
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([100.0]), WeightKind.DESIGN)},
        metadata={"existing_receipt": "preserved"},
    )


def _source():
    hh, cal = _raw()
    for day in range(5):
        _care(cal, day=day, hours=8)
    return derive_nsece_childcare(hh, cal)


def test_candidate_frame_preserves_structure_weights_and_checkpoint_receipts(tmp_path):
    frame = _frame()
    before = frame.table("person").copy(deep=True)
    candidate = with_us_nsece_childcare_attendance(
        frame, _source(), seed=915, match_columns=("age",)
    )
    assert_frame_equal(frame.table("person"), before)
    assert candidate.table("person")[DAYS].tolist() == [0, 5]
    assert candidate.metadata["existing_receipt"] == "preserved"
    assert candidate.metadata["nsece_childcare_attendance"]["candidate_only"] is True
    np.testing.assert_array_equal(
        candidate.weights_for("household").values, frame.weights_for("household").values
    )
    for entity in US_SCHEMA.group_entities:
        assert_frame_equal(candidate.table(entity), frame.table(entity))
    path = tmp_path / "candidate.h5"
    receipts = json.loads(json.dumps(candidate.metadata, default=dict))
    write_frame_checkpoint(path, candidate, metadata={"frame_metadata": receipts})
    stored = load_frame_checkpoint(path)
    reloaded = load_frame_checkpoint(
        path, frame_metadata=stored.metadata["frame_metadata"]
    ).frame
    # Checkpoint JSON sorts mapping keys; receipts retain the same content.
    assert json.loads(json.dumps(reloaded.metadata, default=dict)) == receipts
    assert_frame_equal(reloaded.table("person"), candidate.table("person"))


def test_candidate_export_refuses_unresolved_out_of_scope_inputs():
    candidate = with_us_nsece_childcare_attendance(
        _frame(parent_observed=False), _source(), seed=915, match_columns=("age",)
    )
    assert pd.isna(candidate.table("person").loc[0, DAYS])
    with pytest.raises(ValueError, match="unresolved"):
        assert_childcare_attendance_exportable(candidate)


@pytest.mark.requires_us
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


def test_validation_holds_out_whole_households_and_reports_exclusions(monkeypatch):
    import microcosm.build.us_runtime.nsece_childcare as module

    hh, cal = _raw(30)
    hh.loc[29, "HH4_MISSING_STATUS_CC_1"] = 0
    for row in range(29):
        _care(cal, row=row)
    source = derive_nsece_childcare(hh, cal)
    original = module.impute_us_childcare_attendance

    def checked(recipient, donor, **kwargs):
        assert set(recipient.source_household_id).isdisjoint(donor.source_household_id)
        return original(recipient, donor, **kwargs)

    monkeypatch.setattr(module, "impute_us_childcare_attendance", checked)
    report = nsece_childcare_validation_report(source)
    assert report["household_overlap"] == 0
    assert report["production_ready"] is False
    assert report["under13_complete_weight_share"] == pytest.approx(29 / 30)
    assert report["comparisons"][0]["observed"] == report["comparisons"][0]["predicted"]


@pytest.mark.parametrize(
    "code,parent,age,expected",
    [
        (51, 1, 36, 0),
        (51, 0, 36, 1),
        (52, -1, 36, None),
        (54, 1, 36, 1),
        (61, 1, 36, 1),
        (68, 1, 36, None),
        (68, 1, 84, 0),
        (70, 0, 36, None),
    ],
)
def test_gap_care_uses_parent_status_and_school_age(code, parent, age, expected):
    hh, cal = _raw()
    hh["HH4_RPARENT"] = parent
    hh["HHC4_AGE_AT_USAGE_1"] = age
    cal.loc[0, "HH4_CHCAL_R_1_1"] = code
    child = derive_nsece_childcare(hh, cal).children.iloc[0]
    if expected is None:
        assert pd.isna(child[DAYS])
    else:
        assert child[DAYS] == expected


def _asec_frame():
    frame = _frame()
    tables = {e: frame.table(e).copy() for e in frame.entities}
    tables["person"]["A_LINENO"] = [1, 2]
    tables["person"]["PEPAR1"] = [-1, 1]
    tables["person"]["PEPAR2"] = [-1, -1]
    tables["person"]["hours_worked_last_week"] = [40, 0]
    tables["person"]["PTOTVAL"] = [40000, 0]
    tables["person"]["source_year"] = 2023
    tables["household"]["state_fips"] = 25
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        metadata=frame.metadata,
    )


def test_asec_parent_work_and_region_come_from_measured_relationships():
    from microcosm.build.us_runtime.childcare_population import (
        harmonize_asec_childcare_predictors,
    )

    before = _asec_frame()
    result = harmonize_asec_childcare_predictors(before)
    assert result.table("person").parent_work_status.tolist() == [2, 2]
    assert result.table("person").region.tolist() == [1, 1]
    assert "parent_work_status" not in before.table("person")
    assert result.metadata["existing_receipt"] == "preserved"
    # A working adult without a parent link must not count as a working parent.
    before.table("person")["PEPAR1"] = -1
    result = harmonize_asec_childcare_predictors(before)
    assert result.table("person").parent_work_status.tolist() == [-1, -1]


def test_asec_dangling_parent_pointer_is_refused():
    from microcosm.build.us_runtime.childcare_population import (
        harmonize_asec_childcare_predictors,
    )

    frame = _asec_frame()
    frame.table("person").loc[1, "PEPAR1"] = 99
    with pytest.raises(ValueError, match="does not resolve"):
        harmonize_asec_childcare_predictors(frame)


def test_integer_source_ids_are_losslessly_encoded_and_restored():
    frame = _frame()
    frame.table("person")["person_source_id"] = [2**60, 2**60 + 1]
    result = with_us_nsece_childcare_attendance(
        frame, _source(), seed=915, match_columns=("age",)
    )
    pd.testing.assert_series_equal(
        result.table("person").person_source_id, frame.table("person").person_source_id
    )
    assert result.table("person")[DAYS].tolist() == [0, 5]


def test_sparse_matching_is_explicit_and_keeps_joint_schedule():
    from microcosm.build.us_runtime.nsece_childcare import (
        NSECE_CHILDCARE_FALLBACK_COLUMNS,
        NSECE_CHILDCARE_MATCH_COLUMNS,
    )

    frame = _frame()
    frame.table("person")["region"] = 4
    frame.table("person")["parent_work_status"] = 0
    frame.table("person")["income_band"] = 2
    with pytest.raises(ValueError, match="No compatible"):
        with_us_nsece_childcare_attendance(
            frame, _source(), seed=915, match_columns=NSECE_CHILDCARE_MATCH_COLUMNS
        )
    result = with_us_nsece_childcare_attendance(
        frame,
        _source(),
        seed=915,
        match_columns=NSECE_CHILDCARE_MATCH_COLUMNS,
        fallback_match_columns=NSECE_CHILDCARE_FALLBACK_COLUMNS,
    )
    assert result.table("person").loc[1, "childcare_attendance_match_level"] == "age"
    assert result.table("person").loc[1, [MONTH, DAYS, HOURS]].tolist() == [22, 5, 8]


def test_census_region_map_covers_all_states_once():
    from microcosm.calibrate.geography_constants import (
        US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION,
        US_STATE_NUMERIC_FIPS_TO_POSTAL,
    )

    assert set(US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION) == set(
        US_STATE_NUMERIC_FIPS_TO_POSTAL
    )
    assert US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION[11] == 3
    assert US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION[17] == 2
    assert US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION[6] == 4


def test_noncalendar_bridge_preserves_measured_hours_without_asserting_days_observed():
    from microcosm.build.us_runtime.nsece_childcare_bridge import (
        bridge_nsece_noncalendar_attendance,
    )

    hh, cal = _raw(2)
    for day in range(5):
        _care(cal, row=0, day=day, hours=8)
    hh.loc[1, "HH4_METH_QUEXVERSION"] = 2
    hh.loc[1, "HH4_MISSING_STATUS_CC_1"] = 0
    for kind in range(1, 10):
        hh.loc[1, f"HHC4_NPC_HRSWEEK_TOC{kind}_1"] = 0
    hh.loc[1, "HHC4_NPC_HRSWEEK_TOC4_1"] = 12
    source = derive_nsece_childcare(hh, cal)
    result = bridge_nsece_noncalendar_attendance(source, seed=915)
    child = result.children.iloc[1]
    assert child.attendance_status == "summary_bridge"
    assert child.regular_hours_per_week == 12
    assert child[DAYS] == 5
    assert child[HOURS] == pytest.approx(2.4)
    assert child.ece_hours_per_week == 12
    assert pd.isna(source.children.iloc[1][DAYS])
    assert len(result.donors()[0]) == 2
    with pytest.raises(ValueError, match="before bridge"):
        nsece_childcare_validation_report(result)


def test_zero_regular_care_does_not_erase_unmeasured_irregular_care():
    from microcosm.build.us_runtime.nsece_childcare_bridge import (
        bridge_nsece_noncalendar_attendance,
    )

    hh, cal = _raw(2)
    hh.loc[0, "HH4_TYPEOFCARE_AGG_1_1"] = 7
    _care(cal, row=0, hours=2)
    hh.loc[1, "HH4_METH_QUEXVERSION"] = 3
    hh.loc[1, "HH4_MISSING_STATUS_CC_1"] = 0
    for kind in range(1, 10):
        hh.loc[1, f"HHC4_NPC_HRSWEEK_TOC{kind}_1"] = 0
    result = bridge_nsece_noncalendar_attendance(
        derive_nsece_childcare(hh, cal), seed=915
    )
    assert result.children.iloc[1].regular_hours_per_week == 0
    assert result.children.iloc[1].irregular_hours_per_week == 2
    assert result.children.iloc[1][DAYS] == 1


def test_shared_household_rank_preserves_sibling_and_clone_coherence():
    from microcosm.build.us_runtime.childcare_attendance import (
        impute_us_childcare_attendance,
    )

    donor = pd.DataFrame(
        {
            "donor_id": ["a", "b"],
            "age": [3, 3],
            MONTH: [0, 22],
            DAYS: [0, 5],
            HOURS: [0, 8],
        }
    )
    recipients = pd.DataFrame(
        {
            "person_source_id": ["a", "b", "a", "b"],
            "childcare_source_household_id": ["family"] * 4,
            "age": [3] * 4,
        }
    )
    result = impute_us_childcare_attendance(
        recipients,
        donor,
        donor_weights=Weights(np.array([1.0, 1.0]), WeightKind.DESIGN),
        match_columns=("age",),
        seed=915,
        sibling_dependence=1,
    )
    assert result[DAYS].nunique() == 1
    assert result[HOURS].nunique() == 1


def test_source_stage_contract_pins_outputs_and_assets():
    from microcosm.build.source_manifest import SourceStageSpec
    from microcosm.build.us_runtime.childcare_attendance import (
        childcare_attendance_contract,
        childcare_income_band,
    )
    from microcosm.build.us_runtime.nsece_childcare import (
        NSECE_2024_CALENDAR_SHA256,
        NSECE_2024_HOUSEHOLD_SHA256,
    )

    contract = childcare_attendance_contract()
    assert (
        SourceStageSpec.from_mapping(contract).outputs
        == US_CHILDCARE_ATTENDANCE_COLUMNS
    )
    assert [x["sha256"] for x in contract["artifacts"]] == [
        NSECE_2024_HOUSEHOLD_SHA256,
        NSECE_2024_CALENDAR_SHA256,
    ]
    assert childcare_income_band([0, 25000, 25001, 200000, 200001]).tolist() == [
        0,
        1,
        2,
        4,
        5,
    ]


@pytest.mark.requires_us
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


def test_asec_income_sidecar_checks_identity_and_preserves_raw_missingness(
    tmp_path, monkeypatch
):
    import hashlib
    from types import SimpleNamespace

    from microcosm.build.us_runtime import childcare_population as population

    frame = _asec_frame()
    people = frame.table("person")
    people["PERIDNUM"] = ["1000000000000000000001", "1000000000000000000002"]
    people["A_AGE"] = people.age
    people["PTOTVAL"] = [np.nan, 0.0]
    raw = people[["PERIDNUM", "A_AGE", "A_LINENO", "PTOTVAL"]].copy()
    raw.loc[0, "PTOTVAL"] = 40000
    path = tmp_path / "source.csv"
    raw.to_csv(path, index=False)
    monkeypatch.setattr(
        population,
        "ASEC_EDUCATION_ASSISTANCE_ARCHIVES",
        {
            2023: SimpleNamespace(
                member=path.name,
                member_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                member_size_bytes=path.stat().st_size,
                rows=2,
            )
        },
    )
    with pytest.raises(ValueError, match="finite PTOTVAL"):
        population.harmonize_asec_childcare_predictors(frame)
    result = population.harmonize_asec_childcare_predictors(
        frame, source_cache=tmp_path
    )
    assert result.table("person").income_band.tolist() == [2, 2]
    assert pd.isna(result.table("person").loc[0, "PTOTVAL"])
    people.loc[1, "PTOTVAL"] = 1
    with pytest.raises(ValueError, match="disagrees with pinned source"):
        population.harmonize_asec_childcare_predictors(frame, source_cache=tmp_path)
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="identity mismatch"):
        population.harmonize_asec_childcare_predictors(frame, source_cache=tmp_path)


@pytest.mark.requires_us
def test_registered_attendance_recipe_uses_real_transform_chain(monkeypatch):
    from microcosm.build.us_runtime import childcare_attendance_stage as stage

    frame = _asec_frame()
    frame.table("household")["household_source_id"] = "family"
    hh, cal = _raw()
    hh["HHC4_AGE_AT_USAGE_1"] = frame.table("person").loc[1, "age"] * 12
    hh["HH4_REGION"] = 1
    hh["HH4_PARWORK_STATUS"] = 2
    hh["HH4_ECON_INCOME_ANNUAL"] = 40000
    _care(cal, hours=3)
    source = derive_nsece_childcare(hh, cal)
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


def test_bridge_retains_all_equally_near_donors():
    from microcosm.build.us_runtime.nsece_childcare_bridge import (
        bridge_nsece_noncalendar_attendance,
    )

    hh, cal = _raw(13)
    # Eleven donors have zero regular hours. Household 9 sorts last by ID
    # and has irregular care and substantial weight; truncating equal-distance
    # donors by their IDs would incorrectly erase that care distribution.
    hh.loc[8, "HH4_TYPEOFCARE_AGG_1_1"] = 7
    hh.loc[8, "HHC4_METH_WEIGHT_1"] = 1e12
    _care(cal, row=8, hours=2)
    for row in (11, 12):
        hh.loc[row, "HH4_METH_QUEXVERSION"] = 3
        hh.loc[row, "HH4_MISSING_STATUS_CC_1"] = 0
        for kind in range(1, 10):
            hh.loc[row, f"HHC4_NPC_HRSWEEK_TOC{kind}_1"] = 0
    source = derive_nsece_childcare(hh, cal)
    result = bridge_nsece_noncalendar_attendance(source, seed=915)
    bridged = result.children.loc[
        result.children.attendance_status.eq("summary_bridge")
    ]
    assert len(bridged) == 2
    assert bridged.irregular_hours_per_week.eq(2).all()
