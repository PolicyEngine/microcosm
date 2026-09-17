"""Synthetic source-code and Frame/export contracts; no NSECE microdata in CI."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import childcare_attendance_stage as stage
from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.childcare_attendance_receipt import (
    assert_bound_childcare_attendance,
    restore_native_childcare_receipt,
)
from microcosm.build.us_runtime.childcare_population import (
    harmonize_asec_childcare_predictors,
)
from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CALENDAR_BLOCKS,
    NSECE_CHILD_INDICES,
    NSECE_PROVIDER_INDICES,
    NSECEChildcareSource,
    assert_childcare_attendance_exportable,
    derive_nsece_childcare,
    load_nsece_childcare,
    nsece_childcare_calendar_columns,
    nsece_childcare_household_columns,
    nsece_childcare_validation_report,
    with_us_nsece_childcare_attendance,
)
from microcosm.build.us_runtime.release_input_coverage import (
    ReleaseInputColumn,
    ReleaseInputCoverageManifest,
    us_release_input_coverage_gate,
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


def test_calendar_bounds_preserve_measured_blocks_without_filling_ambiguous_time():
    hh, cal = _raw()
    _care(cal, hours=4)
    cal.loc[0, "HH4_CHCAL_R_1_200"] = 97
    child = derive_nsece_childcare(hh, cal).children.iloc[0]
    assert child.calendar_ece_hours_lower == 4
    assert child.calendar_ece_hours_upper == 4.25
    assert child.calendar_ece_days_lower == 1
    assert child.calendar_ece_days_upper == 2
    assert child.calendar_unknown_hours == 0.25
    assert child.attendance_status == "ambiguous_calendar"
    assert pd.isna(child[DAYS])


def test_partial_calendar_assumed_parental_zeros_remain_unresolved():
    hh, cal = _raw()
    _care(cal, hours=4)
    hh["HH4_MISSING_STATUS_CC_1"] = 1
    child = derive_nsece_childcare(hh, cal).children.iloc[0]
    assert child.calendar_ece_hours_lower == 4
    assert child.calendar_ece_hours_upper == 168
    assert child.calendar_ece_days_lower == 1
    assert child.calendar_ece_days_upper == 7
    assert child.calendar_unknown_hours == 164
    assert pd.isna(child.regular_hours_per_week)
    assert pd.isna(child[HOURS])


def test_missing_calendar_has_uninformative_bounds_not_observed_zeros():
    hh, cal = _raw()
    hh["HH4_MISSING_STATUS_CC_1"] = 0
    child = derive_nsece_childcare(hh, cal).children.iloc[0]
    assert child.calendar_ece_hours_lower == 0
    assert child.calendar_ece_hours_upper == 168
    assert child.calendar_ece_days_lower == 0
    assert child.calendar_ece_days_upper == 7
    assert pd.isna(child[DAYS])


def test_complete_calendar_bounds_equal_the_observed_schedule():
    hh, cal = _raw()
    _care(cal, hours=4)
    child = derive_nsece_childcare(hh, cal).children.iloc[0]
    assert child.calendar_ece_hours_lower == child.ece_hours_per_week
    assert child.calendar_ece_hours_upper == child.ece_hours_per_week
    assert child.calendar_ece_days_lower == child[DAYS]
    assert child.calendar_ece_days_upper == child[DAYS]
    assert child.calendar_unknown_hours == 0


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
@pytest.mark.parametrize("repair_factor", [1.0, 2.0])
def test_registered_attendance_recipe_uses_real_transform_chain(
    monkeypatch, tmp_path, repair_factor
):
    import importlib.util
    from pathlib import Path

    from microcosm.build.us_runtime import childcare_attendance_stage as stage

    builder_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "build_us_fiscal_refresh_release.py"
    )
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


def test_bridge_widens_a_thin_cell_instead_of_keeping_distant_donors():
    from microcosm.build.us_runtime.nsece_childcare_bridge import (
        bridge_nsece_noncalendar_attendance,
    )

    hh, cal = _raw(12)
    # The target's exact cell holds one donor with 1 regular hour on 1 day;
    # ten donors in another region match its 40 regular hours over 5 days.
    _care(cal, row=0, hours=1)
    for row in range(1, 11):
        hh.loc[row, "HH4_REGION"] = 2
        for day in range(5):
            _care(cal, row=row, day=day, hours=8)
    hh.loc[11, "HH4_METH_QUEXVERSION"] = 2
    hh.loc[11, "HH4_MISSING_STATUS_CC_1"] = 0
    for kind in range(1, 10):
        hh.loc[11, f"HHC4_NPC_HRSWEEK_TOC{kind}_1"] = 0
    hh.loc[11, "HHC4_NPC_HRSWEEK_TOC4_1"] = 40
    result = bridge_nsece_noncalendar_attendance(
        derive_nsece_childcare(hh, cal), seed=915
    )
    bridged = result.children.loc[
        result.children.attendance_status.eq("summary_bridge")
    ]
    assert len(bridged) == 1
    child = bridged.iloc[0]
    assert child.schedule_bridge_match == "age,parent_work_status,income_band"
    assert child[DAYS] == 5
    assert child[HOURS] == 8
    assert result.source_receipt["noncalendar_bridge"]["matching_levels"] == {
        "age,parent_work_status,income_band": 1
    }


def _replace(frame, *, people=None, metadata=None):
    tables = {e: frame.table(e).copy() for e in frame.entities}
    if people is not None:
        tables["person"] = people
    return Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        metadata=frame.metadata if metadata is None else metadata,
    )


def _candidate():
    return with_us_nsece_childcare_attendance(
        _frame(), _source(), seed=915, match_columns=("age",)
    )


@pytest.mark.parametrize(
    "column,value",
    [(DAYS, 4), ("age", 4), ("person_id", 99), ("person_household_id", 99)],
)
def test_persisted_value_or_identity_change_fails_binding(column, value):
    candidate = _candidate()
    candidate.table("person").loc[1, column] = value
    with pytest.raises(ValueError, match="differ from the source receipt"):
        assert_bound_childcare_attendance(candidate, require_stage=False)


def test_receipt_loss_and_metadata_relabeling_are_rejected():
    candidate = _candidate()
    with pytest.raises(ValueError, match="content-bound"):
        assert_bound_childcare_attendance(
            _replace(candidate, metadata={}), require_stage=False
        )
    metadata = json.loads(json.dumps(candidate.metadata, default=dict))
    metadata["nsece_childcare_attendance"]["seed"] += 1
    with pytest.raises(ValueError, match="metadata disagrees"):
        assert_bound_childcare_attendance(
            _replace(candidate, metadata=metadata), require_stage=False
        )


def test_same_transfer_is_idempotent_but_seed_and_source_refresh_are_rejected():
    source = _source()
    candidate = _candidate()
    assert (
        with_us_nsece_childcare_attendance(
            candidate, source, seed=915, match_columns=("age",)
        )
        is candidate
    )
    with pytest.raises(ValueError, match="settings changed"):
        with_us_nsece_childcare_attendance(
            candidate, source, seed=916, match_columns=("age",)
        )
    source.source_receipt["source_year"] = 2025
    with pytest.raises(ValueError, match="source or settings changed"):
        with_us_nsece_childcare_attendance(
            candidate, source, seed=915, match_columns=("age",)
        )


def test_selection_order_and_calibration_do_not_invalidate_attendance():
    candidate = _candidate()
    selected = _replace(candidate, people=candidate.table("person").iloc[::-1])
    assert_bound_childcare_attendance(selected, require_stage=False)
    tables = {e: selected.table(e) for e in selected.entities}
    calibrated = Frame(
        tables,
        selected.schema,
        {"household": Weights(np.array([200.0]), WeightKind.CALIBRATED)},
        metadata=selected.metadata,
    )
    assert_bound_childcare_attendance(calibrated, require_stage=False)


@pytest.mark.parametrize("value", [np.nan, None, "", "nan", "<NA>"])
def test_invalid_household_source_identity_fails_before_string_conversion(value):
    frame = _asec_frame()
    frame.table("household")["household_source_id"] = value
    with pytest.raises(ValueError, match="household identities"):
        harmonize_asec_childcare_predictors(frame)


def test_unresolved_household_membership_is_rejected():
    frame = _asec_frame()
    frame.table("household")["household_source_id"] = "household"
    frame.table("person")["PEPAR1"] = -1
    frame.table("person").loc[1, "person_household_id"] = 999
    with pytest.raises(ValueError, match="link does not resolve"):
        harmonize_asec_childcare_predictors(frame)


@pytest.mark.parametrize(
    "column,value",
    [(DAYS, np.nan), (DAYS, 8), (DAYS, 0), (US_CHILDCARE_ATTENDANCE_COLUMNS[0], 2.5)],
)
def test_final_boundary_rechecks_each_row_and_schedule(column, value):
    candidate = _candidate()
    candidate.table("person").loc[1, column] = value
    with pytest.raises(ValueError):
        assert_childcare_attendance_exportable(candidate)
    manifest = ReleaseInputCoverageManifest(
        reference={},
        columns=tuple(
            ReleaseInputColumn(c, "required") for c in US_CHILDCARE_ATTENDANCE_COLUMNS
        ),
    )
    gate = us_release_input_coverage_gate(
        candidate,
        SimpleNamespace(default_values=lambda columns: {c: 0 for c in columns}),
        manifest=manifest,
    )
    assert not gate.passed
    assert any("Childcare" in x or "Attendance" in x for x in gate.failures)


@pytest.mark.requires_us
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


@pytest.mark.requires_us
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


def test_default_outside_domain_policy_fails_early_and_names_the_flag(monkeypatch):
    monkeypatch.setattr(stage, "load_nsece_childcare", lambda *args: _source())
    frame = _asec_frame()
    # The adult has no observed attendance, as on every production parent.
    frame.table("person").loc[0, list(US_CHILDCARE_ATTENDANCE_COLUMNS)] = np.nan
    with pytest.raises(ValueError, match="inherit-outside-domain-baseline"):
        stage.with_us_childcare_attendance_inputs(
            frame,
            household_tsv="fake",
            calendar_tsv="fake",
            asec_source_cache=None,
            seed=915,
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("HH4_REGION", -1),
        ("HH4_PARWORK_STATUS", -8),
        ("HH4_ECON_INCOME_ANNUAL", -9),
    ],
)
def test_reserve_codes_never_become_matching_cells(column, value):
    hh, cal = _raw()
    hh[column] = value
    with pytest.raises(ValueError, match="reserve code"):
        derive_nsece_childcare(hh, cal)


def test_final_gate_cannot_accept_unbound_nondegenerate_columns():
    candidate = _replace(_candidate(), metadata={})
    manifest = ReleaseInputCoverageManifest(
        reference={},
        columns=tuple(
            ReleaseInputColumn(c, "required") for c in US_CHILDCARE_ATTENDANCE_COLUMNS
        ),
    )
    gate = us_release_input_coverage_gate(
        candidate,
        SimpleNamespace(default_values=lambda columns: {c: 0 for c in columns}),
        manifest=manifest,
    )
    assert not gate.passed
    assert any("content-bound" in x for x in gate.failures)


def test_receipt_loss_cannot_relabel_derived_values_as_observations():
    with pytest.raises(ValueError, match="lost its receipt"):
        with_us_nsece_childcare_attendance(
            _replace(_candidate(), metadata={}),
            _source(),
            seed=916,
            match_columns=("age",),
        )


def test_changed_recipe_invalidates_receipt(monkeypatch):
    from microcosm.build.us_runtime import childcare_attendance_receipt as receipts

    candidate = _candidate()
    monkeypatch.setattr(
        receipts, "attendance_recipe_identity", lambda: {"different": True}
    )
    with pytest.raises(ValueError, match="recipe changed"):
        assert_bound_childcare_attendance(candidate)
    # Read-only ingress checks content only, so released files stay loadable.
    assert_bound_childcare_attendance(candidate, require_stage=False)


def test_public_metadata_omits_person_hash_inventory():
    from microcosm.build.us_runtime.childcare_attendance_receipt import (
        ATTENDANCE_RECEIPT_KEY,
        childcare_attendance_public_metadata,
    )

    candidate = _candidate()
    public = childcare_attendance_public_metadata(candidate)
    assert "rows" not in public[ATTENDANCE_RECEIPT_KEY]
    assert public[ATTENDANCE_RECEIPT_KEY]["source_people"] == 2
    assert len(candidate.metadata[ATTENDANCE_RECEIPT_KEY]["rows"]) == 2
    # Population-sized mappings make immutable Frame metadata serialization
    # quadratic. Keep the private row inventory as a sequence of ID/hash pairs.
    assert isinstance(candidate.metadata[ATTENDANCE_RECEIPT_KEY]["rows"], tuple)


def test_retaining_a_subset_keeps_attendance_binding():
    candidate = _candidate()
    selected = _replace(candidate, people=candidate.table("person").iloc[[1]])
    summary = assert_bound_childcare_attendance(selected, require_stage=False)
    assert summary["source_people"] == 2
    assert summary["retained_people"] == 1


def test_shared_rank_joint_moment_integrates_unequal_cdfs():
    from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
        _joint_product,
    )

    a = (np.array([[0.0], [2.0]]), np.array([0.25, 1]))
    b = (np.array([[1.0], [3.0]]), np.array([0.5, 1]))
    # [0,.25): 0; [.25,.5): 2; [.5,1): 6.
    assert _joint_product(a, b) == pytest.approx([3.5])


def test_undefined_sibling_metrics_do_not_pass_validation():
    from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
        sibling_schedule_screen,
    )

    result = sibling_schedule_screen({"larger_households": {}})
    assert not result["passed"]
    assert all(not check["passed"] for check in result["checks"])


def _sibling_source():
    from microcosm.build.us_runtime.nsece_childcare import NSECEChildcareSource

    rows = []
    for household in range(100):
        size = 1 if household % 2 else 3
        for child in range(size):
            days = 5.0 if size == 1 else 0.0
            rows.append(
                {
                    "donor_id": f"h{household}:c{child}",
                    "source_household_id": f"h{household}",
                    "age": 3,
                    "childcare_household_size": size,
                    "region": 1,
                    "parent_work_status": 2,
                    "income_band": 1,
                    "attendance_status": "complete",
                    "household_weight": 1.0,
                    "child_weight": 1.0,
                    MONTH: 22.0 if days else 0.0,
                    DAYS: days,
                    HOURS: 8.0 if days else 0.0,
                }
            )
    children = pd.DataFrame(rows)
    return NSECEChildcareSource(
        children, Weights(np.ones(len(children)), WeightKind.DESIGN), {}
    )


def test_reserved_households_are_excluded_from_every_development_split():
    from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
        _household_splits,
    )

    children = _sibling_source().children
    arguments = {"seed": 271828, "validation_seed": 20260916}
    train, reserved = _household_splits(children, partition="validation", **arguments)[
        0
    ]
    heldout_ids = set(children.loc[reserved, "source_household_id"])
    assert heldout_ids
    scored = set()
    for training, evaluation in _household_splits(
        children, partition="development", **arguments
    ):
        training_ids = set(children.loc[training, "source_household_id"])
        evaluation_ids = set(children.loc[evaluation, "source_household_id"])
        assert training_ids.isdisjoint(evaluation_ids | heldout_ids)
        assert evaluation_ids.isdisjoint(heldout_ids | scored)
        scored.update(evaluation_ids)
    assert scored == set(children.loc[train, "source_household_id"])


def test_household_size_changes_actual_validation_donors_and_not_only_rho():
    from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
        assess_sibling_schedules,
    )

    source = _sibling_source()
    settings = {
        "fallback_match_columns": (),
        "validation_seed": 20260916,
        "partition": "validation",
    }
    legacy = assess_sibling_schedules(source, match_columns=("age",), **settings)
    conditioned = assess_sibling_schedules(
        source, match_columns=("age", "childcare_household_size"), **settings
    )
    assert legacy["larger_households"]["coupled"]["mean_total_days"] > 0
    assert conditioned["larger_households"]["coupled"]["mean_total_days"] == 0
    assert conditioned["matching_counts"] == {
        "age,childcare_household_size": 3
        * conditioned["larger_households"]["households"]
    }


def test_reserved_outcomes_cannot_influence_development(monkeypatch):
    from microcosm.build.us_runtime import nsece_childcare_sibling_validation as module

    source = _sibling_source()
    settings = {"validation_seed": 20260916, "partition": "development"}
    _, reserved = module._household_splits(
        source.children, seed=271828, validation_seed=20260916, partition="validation"
    )[0]
    reserved_ids = set(source.children.loc[reserved, "source_household_id"])
    original_fit = module.fit_nsece_sibling_dependence

    def checked_fit(children, **kwargs):
        assert set(children.source_household_id).isdisjoint(reserved_ids)
        return original_fit(children, **kwargs)

    monkeypatch.setattr(module, "fit_nsece_sibling_dependence", checked_fit)
    before = module.assess_sibling_schedules(source, **settings)
    source.children.loc[reserved, [MONTH, DAYS, HOURS]] = [30, 7, 24]
    assert module.assess_sibling_schedules(source, **settings) == before


def test_sibling_assessment_rejects_reconstructed_calendars():
    from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
        assess_sibling_schedules,
    )

    source = _sibling_source()
    source.children.loc[0, "attendance_status"] = "summary_bridge"
    with pytest.raises(ValueError, match="original measured"):
        assess_sibling_schedules(source)


def test_size_challenger_counts_missing_calendars_without_mutating_source(monkeypatch):
    from microcosm.build.us_runtime import nsece_childcare_sibling_validation as module

    source = _sibling_source()
    first_household = source.children.index[
        source.children.source_household_id.eq("h0")
    ]
    source.children.loc[first_household[1], "attendance_status"] = "missing_calendar"
    source.children.loc[first_household[1], [MONTH, DAYS, HOURS]] = np.nan
    source.children.loc[first_household[2], "age"] = 13
    before = source.children.copy(deep=True)
    calls = []

    def capture(candidate, **kwargs):
        calls.append((candidate.children.copy(), kwargs))
        return {}

    monkeypatch.setattr(module, "assess_sibling_schedules", capture)
    report = module.compare_household_size_matching(source, partition="development")
    assert len(calls) == 2
    revised, settings = calls[1]
    assert revised.loc[first_household, "childcare_household_size"].tolist() == [
        2,
        2,
        2,
    ]
    assert "childcare_household_size" in settings["match_columns"]
    assert settings["fallback_match_columns"][-2:] == (
        ("age", "childcare_household_size"),
        ("age",),
    )
    assert_frame_equal(source.children, before)
    assert report["production_recipe_changed"] is False


@pytest.mark.parametrize(
    "partition,validation_seed", [("unknown", 1), ("all", 1), ("validation", None)]
)
def test_sibling_partition_configuration_fails_closed(partition, validation_seed):
    from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
        _household_splits,
    )

    with pytest.raises(ValueError):
        _household_splits(
            _sibling_source().children,
            seed=1,
            validation_seed=validation_seed,
            partition=partition,
        )


@pytest.mark.parametrize(
    "irregular,shift,expected_days,expected_hours",
    [
        (False, 0, 5, 12),
        (True, -1, 4, 13),
        (True, 1, 6, 13),
    ],
)
def test_schedule_sensitivity_preserves_measured_regular_hours(
    irregular, shift, expected_days, expected_hours
):
    from microcosm.build.us_runtime.childcare_sensitivity import (
        noncalendar_sensitivity_source,
    )

    source = _source()
    source.children["attendance_status"] = "summary_bridge"
    source.children["regular_hours_per_week"] = 12.0
    source.children["irregular_hours_per_week"] = 1.0
    original = source.children.copy(deep=True)
    changed = noncalendar_sensitivity_source(
        source, irregular_hours=irregular, day_shift=shift
    )
    row = changed.children.iloc[0]
    assert row.regular_hours_per_week == 12
    assert row[DAYS] == expected_days
    assert row[DAYS] * row[HOURS] == pytest.approx(expected_hours)
    assert_frame_equal(source.children, original)
    observed = noncalendar_sensitivity_source(
        _source(), irregular_hours=False, day_shift=-1
    )
    assert_frame_equal(observed.children, _source().children)


def _paired_sensitivity_fixture():
    source = _source()
    source.children["attendance_status"] = "summary_bridge"
    source.children["regular_hours_per_week"] = 12.0
    source.children["irregular_hours_per_week"] = 1.0
    source.children[HOURS] = 13 / source.children[DAYS]
    source.children["ece_hours_per_week"] = 13.0
    child = source.children.iloc[0]
    people = pd.DataFrame(
        {"age": [child.age], **{c: [child[c]] for c in (MONTH, DAYS, HOURS)}}
    )
    for c in (MONTH, DAYS, HOURS):
        people[f"{c}_source"] = f"donor:{child.donor_id}"
    return people, source


def test_paired_sensitivity_keeps_selected_donor_and_measured_hours():
    from microcosm.build.us_runtime.childcare_sensitivity import (
        noncalendar_sensitivity_source,
        paired_noncalendar_attendance,
    )

    people, source = _paired_sensitivity_fixture()
    original = people.copy(deep=True)
    changed = noncalendar_sensitivity_source(source, day_shift=-1)
    values, report = paired_noncalendar_attendance(people, source, changed)
    assert values[0, 1] == people[DAYS].iloc[0] - 1
    assert values[0, 1] * values[0, 2] == pytest.approx(13)
    assert report["changed_people"] == 1
    assert report["measured_calendar_assignments_changed"] == 0
    assert_frame_equal(people, original)
    # Row order in the donor source has no effect on the identity lookup.
    reversed_source = NSECEChildcareSource(
        changed.children.iloc[::-1], changed.weights, changed.source_receipt
    )
    np.testing.assert_array_equal(
        values, paired_noncalendar_attendance(people, source, reversed_source)[0]
    )


@pytest.mark.parametrize(
    "problem", ["unknown_donor", "mixed_donor", "wrong_value", "observed_conflict"]
)
def test_paired_sensitivity_rejects_invalid_lineage_and_observation_conflicts(problem):
    from microcosm.build.us_runtime.childcare_sensitivity import (
        noncalendar_sensitivity_source,
        paired_noncalendar_attendance,
    )

    people, source = _paired_sensitivity_fixture()
    changed = noncalendar_sensitivity_source(source, day_shift=-1)
    if problem == "unknown_donor":
        for c in (MONTH, DAYS, HOURS):
            people[f"{c}_source"] = "donor:missing"
    elif problem == "mixed_donor":
        people[f"{HOURS}_source"] = "donor:another"
    elif problem == "wrong_value":
        people[HOURS] += 0.1
    else:
        people[f"{DAYS}_source"] = "observed"
    with pytest.raises(ValueError):
        paired_noncalendar_attendance(people, source, changed)


def test_paired_sensitivity_does_not_reassign_measured_donors_when_care_order_changes():
    from microcosm.build.us_runtime.childcare_sensitivity import (
        noncalendar_sensitivity_source,
        paired_noncalendar_attendance,
    )
    from microcosm.frame import WeightKind, Weights

    people, source = _paired_sensitivity_fixture()
    measured = source.children.copy().assign(
        donor_id="measured",
        attendance_status="complete",
        regular_hours_per_week=5.0,
        irregular_hours_per_week=0.0,
        ece_hours_per_week=5.0,
    )
    measured[HOURS] = 1.0
    combined = NSECEChildcareSource(
        pd.concat([source.children, measured], ignore_index=True),
        Weights(np.array([1.0, 1.0]), WeightKind.DESIGN),
        source.source_receipt,
    )
    observed_recipient = people.copy()
    observed_recipient[HOURS] = 1.0
    for c in (MONTH, DAYS, HOURS):
        observed_recipient[f"{c}_source"] = "donor:measured"
    outside = people.copy()
    outside["age"] = 30
    for c in (MONTH, DAYS, HOURS):
        outside[c] = 0.0
        outside[f"{c}_source"] = "inherited_engine_baseline"
    people = pd.concat([people, observed_recipient, outside], ignore_index=True)
    alternative = noncalendar_sensitivity_source(combined, day_shift=-1)
    before_order = combined.children.sort_values([DAYS, HOURS]).donor_id.tolist()
    after_order = alternative.children.sort_values([DAYS, HOURS]).donor_id.tolist()
    assert before_order == after_order[::-1]
    result, audit = paired_noncalendar_attendance(people, combined, alternative)
    assert result[0, 1] == 4
    np.testing.assert_array_equal(result[1:], people.iloc[1:][[MONTH, DAYS, HOURS]])
    assert audit["changed_people"] == 1
    assert audit["bridge_assigned_people"] == 1
