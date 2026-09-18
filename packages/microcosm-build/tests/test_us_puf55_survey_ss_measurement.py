"""Invented report-sum values and real retained-source qualification controls."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from test_us_full_puf_output_profiles import _puf59_columns
from test_us_survey_social_security import _social_security_arguments

from microcosm.build.us_runtime import full_puf_enrichment as full
from microcosm.build.us_runtime import puf55_survey_ss_measurement as measurement


def _values():
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 11, dtype=np.int64),
            "person_tax_unit_id": [10, 10, 20, 20, 20, 30, 30, 40, 50, 50],
            "tax_unit_role_input": pd.array(
                [
                    "HEAD",
                    "DEPENDENT",
                    "HEAD",
                    "SPOUSE",
                    "DEPENDENT",
                    "HEAD",
                    "DEPENDENT",
                    "HEAD",
                    "HEAD",
                    "DEPENDENT",
                ],
                dtype="string",
            ),
        }
    )
    units = pd.DataFrame(
        {
            "tax_unit_id": [10, 20, 30, 40, 50],
            "filing_status_input": pd.array(
                [
                    "SINGLE",
                    "JOINT",
                    "SURVIVING_SPOUSE",
                    "SEPARATE",
                    "HEAD_OF_HOUSEHOLD",
                ],
                dtype="string",
            ),
        }
    )
    reports = pd.DataFrame(
        {
            "social_security_source_total": [
                100,
                np.nan,
                200,
                300,
                900,
                400,
                np.nan,
                0,
                np.nan,
                800,
            ],
            "source_reporting_universe": [
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
            ],
        },
        index=pd.Index(person.person_id.to_numpy(), name="person_id"),
    )
    # These cells are deliberately unrelated to report availability. The proxy
    # neither consumes them nor turns excluded dependents into known zeros.
    for i, column in enumerate(full.SURVEY_SS_COMPONENTS):
        reports[column] = np.resize([-0.0, np.nan, 10.0 + i], len(reports))
        reports["allowed_" + column] = np.resize([True, False], len(reports))
    return person, units, reports


def test_exact_head_and_actual_joint_spouse_sum_preserves_dependents_and_components():
    person, units, reports = _values()
    before = [table.copy(deep=True) for table in (person, units, reports)]
    got, counts = measurement._measure(person, units, reports)
    np.testing.assert_array_equal(got[measurement.TOTAL], [100, 500, 400, 0, np.nan])
    assert got[measurement.KNOWN].tolist() == [True, True, True, True, False]
    assert got[measurement.ROUTE].tolist() == [full.PUF55_SURVEY_SS.value] * 4 + [
        full.PUF55_SURVEY_SS_NO_TOTAL.value
    ]
    assert counts == {
        "returns": 5,
        "included_reporters": 6,
        "excluded_dependents": 4,
        "unavailable_included_reports": 1,
        "nine_predictor_returns": 4,
        "eight_predictor_returns": 1,
    }
    for original, current in zip(before, (person, units, reports), strict=True):
        pd.testing.assert_frame_equal(original, current, check_exact=True)
    assert reports.loc[2, list(full.SURVEY_SS_COMPONENTS)].isna().all()
    assert np.signbit(reports.loc[1, full.SURVEY_SS_COMPONENTS[0]])


def test_alignment_uses_person_membership_and_return_ids_not_row_order():
    person, units, reports = _values()
    expected, counts = measurement._measure(person, units, reports)
    got, got_counts = measurement._measure(
        person.iloc[::-1],
        units.iloc[::-1],
        reports.iloc[[3, 0, 8, 2, 1, 5, 7, 6, 9, 4]],
    )
    pd.testing.assert_frame_equal(got.loc[expected.index], expected, check_exact=True)
    assert got_counts == counts


def test_missing_joint_spouse_report_routes_whole_return_without_partial_sum():
    person, units, reports = _values()
    reports.loc[4, "social_security_source_total"] = np.nan
    got, counts = measurement._measure(person, units, reports)
    assert np.isnan(got.loc[20, measurement.TOTAL])
    assert not got.loc[20, measurement.KNOWN]
    assert got.loc[20, measurement.ROUTE] == full.PUF55_SURVEY_SS_NO_TOTAL.value
    assert counts["eight_predictor_returns"] == 2
    assert reports.loc[3, "social_security_source_total"] == 200


@pytest.mark.parametrize(
    "case",
    [
        "no_head",
        "two_heads",
        "missing_joint_spouse",
        "widow_spouse",
        "single_spouse",
        "unknown_role",
        "missing_role",
        "unknown_status",
    ],
)
def test_invalid_roles_refuse_even_when_included_report_is_unavailable(case):
    person, units, reports = _values()
    reports.loc[3, "social_security_source_total"] = np.nan
    if case == "no_head":
        person.loc[0, "tax_unit_role_input"] = "DEPENDENT"
    elif case == "two_heads":
        person.loc[1, "tax_unit_role_input"] = "HEAD"
    elif case == "missing_joint_spouse":
        person.loc[3, "tax_unit_role_input"] = "DEPENDENT"
    elif case == "widow_spouse":
        person.loc[6, "tax_unit_role_input"] = "SPOUSE"
    elif case == "single_spouse":
        person.loc[1, "tax_unit_role_input"] = "SPOUSE"
    elif case == "unknown_role":
        person.loc[3, "tax_unit_role_input"] = "CO_FILER"
    elif case == "missing_role":
        person.loc[3, "tax_unit_role_input"] = pd.NA
    else:
        units.loc[1, "filing_status_input"] = "2"
    with pytest.raises(ValueError, match="PUF55_SURVEY_SS_ROLE_"):
        measurement._measure(person, units, reports)


@pytest.mark.parametrize("value", ["100", True, 100 + 0j])
def test_report_physical_values_are_not_coerced(value):
    person, units, reports = _values()
    reports["social_security_source_total"] = (
        reports.social_security_source_total.astype(object)
    )
    reports.loc[1, "social_security_source_total"] = value
    with pytest.raises(ValueError, match="REPORT_PHYSICAL_TYPE"):
        measurement._measure(person, units, reports)


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf])
def test_invalid_report_refuses_including_excluded_dependent(value):
    person, units, reports = _values()
    reports.loc[5, "social_security_source_total"] = value
    with pytest.raises(ValueError, match="REPORT_DOMAIN"):
        measurement._measure(person, units, reports)


def test_outside_universe_zero_cannot_be_a_known_report():
    person, units, reports = _values()
    reports.loc[9, "social_security_source_total"] = 0
    with pytest.raises(ValueError, match="OUTSIDE_UNIVERSE_OBSERVATION"):
        measurement._measure(person, units, reports)


def test_numeric_universe_is_not_a_boolean_knownness_mask():
    person, units, reports = _values()
    reports["source_reporting_universe"] = reports.source_reporting_universe.astype(int)
    with pytest.raises(ValueError, match="UNIVERSE_PHYSICAL_TYPE"):
        measurement._measure(person, units, reports)


def test_integer_report_cannot_silently_lose_precision_during_conversion():
    person, units, reports = _values()
    reports["social_security_source_total"] = pd.Series(
        0, index=reports.index, dtype="int64"
    )
    reports.loc[1, "social_security_source_total"] = 2**53 + 1
    with pytest.raises(ValueError, match="REPORT_PRECISION"):
        measurement._measure(person, units, reports)


def test_complete_joint_reports_refuse_sum_overflow():
    person, units, reports = _values()
    reports.loc[[3, 4], "social_security_source_total"] = np.finfo(np.float64).max
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="SUM_OVERFLOW"):
        measurement._measure(person, units, reports)


@pytest.mark.parametrize(
    "case", ["duplicate_person", "missing_report", "orphan_member"]
)
def test_incomplete_or_ambiguous_membership_is_not_a_fallback_route(case):
    person, units, reports = _values()
    if case == "duplicate_person":
        person.loc[1, "person_id"] = 1
    elif case == "missing_report":
        reports = reports.iloc[1:]
    else:
        person.loc[0, "person_tax_unit_id"] = 99
    with pytest.raises(ValueError, match="MEMBERSHIP"):
        measurement._measure(person, units, reports)


def test_explicit_eight_predictor_profile_keeps_all_55_outputs_and_distinct_graph_phase():
    profile = full.PUF55_SURVEY_SS_NO_TOTAL
    assert len(profile.targets) == 55
    assert len(profile.person_outputs) == 52
    assert profile.targets == full.PUF55_SURVEY_SS.targets
    assert profile.predictors == full.PUF59.predictors and len(profile.predictors) == 8
    assert profile.source_predictors == full.PUF59.source_predictors
    assert not set(full.SURVEY_SS_COMPONENTS) & set(profile.targets)
    assert len(full.PUF55_SURVEY_SS.predictors) == 9
    assert len(full.PUF59.targets) == 59 and len(full.FULL65.targets) == 65
    assert len({item.phase for item in full.PufOutputProfile}) == 4
    assert full.FULL65.phase == full.PHASE
    assert full.PUF55_SURVEY_SS.phase == full.PHASE + ".puf55_survey_ss"
    fits, applies = full.full_puf_train_apply_nodes(
        donor_population="donor",
        recipient_population="recipient",
        matrix_producer="matrix",
        seed=123,
        n_estimators=2,
        zero_atol=0.0,
        prefix="without_report",
        profile=profile,
    )
    assert len(fits) == len(applies) == 55
    assert tuple(node.params["target"] for node in fits) == profile.targets
    assert all(tuple(node.params["predictors"]) == profile.predictors for node in fits)
    assert all(node.params["phase"] == profile.phase for node in (*fits, *applies))


def test_eight_predictor_canonical_donor_does_not_read_missing_total_or_ss_components():
    person, units, pk, tk = _puf59_columns()
    profile = full.PUF55_SURVEY_SS_NO_TOTAL
    person = person.drop(columns=list(full.SURVEY_SS_COMPONENTS))
    pk = pk.drop(columns=list(full.SURVEY_SS_COMPONENTS))
    donor = full.canonical_full_puf_donor(
        person, units, person_known=pk, tax_unit_known=tk, profile=profile
    )
    assert tuple(donor) == (
        *profile.predictors,
        *profile.targets,
        "weight",
        *profile.donor_auxiliary_columns,
    )
    assert measurement.TOTAL not in donor


def test_caller_role_tables_and_descriptive_receipts_cannot_qualify_a_source():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        measurement.qualify_puf55_survey_ss_measurement(
            {
                "roles": _values()[0],
                "source_authenticated": True,
            }
        )


@pytest.fixture(scope="module")
def retained_source(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        arguments = _social_security_arguments(
            tmp_path_factory.mktemp("ss-measurement"), patch, ambiguous=True
        )
        yield measurement.source.prepare_authenticated_survey_population(**arguments)


def test_actual_retained_source_boundary_and_descriptive_evidence(retained_source):
    state = retained_source._checked()[2]
    before = {
        name: state.frame.table(name).copy(deep=True) for name in state.frame.entities
    }
    got = measurement.qualify_puf55_survey_ss_measurement(retained_source)
    assert got.preparation is retained_source
    assert (
        got.tax_unit.index.tolist()
        == state.frame.table("tax_unit").tax_unit_id.tolist()
    )
    evidence = json.loads(got.evidence)
    assert evidence["measurement"] == measurement.MEASUREMENT
    assert evidence["preparation_sha256"] == measurement._sha(retained_source.payload)
    assert evidence["projection_sha256"] == measurement._projection_digest(got.tax_unit)
    assert evidence["counts"]["returns"] == len(got.tax_unit)
    assert evidence["counts"]["nine_predictor_returns"] + evidence["counts"][
        "eight_predictor_returns"
    ] == len(got.tax_unit)
    assert not evidence["source_admission_issued"]
    assert not evidence["population_admission_issued"]
    assert not evidence["individual_beneficiary_assignment_claim"]
    assert not evidence["current_money_tax_units_reconstructed_here"]
    for name, original in before.items():
        pd.testing.assert_frame_equal(
            state.frame.table(name), original, check_exact=True
        )


def test_detached_source_result_mutation_is_not_source_evidence(retained_source):
    changed = []

    def callback(frame, event, value):
        if (
            event == "return"
            and frame.f_code
            is measurement.reports.qualify_current_social_security.__code__
            and value is not None
        ):
            value.person.iloc[
                0, value.person.columns.get_loc("social_security_source_total")
            ] = 999999.0
            changed.append(True)

    prior = sys.getprofile()
    try:
        sys.setprofile(callback)
        with pytest.raises(ValueError, match="SOURCE_REPORT_IDENTITY"):
            measurement.qualify_puf55_survey_ss_measurement(retained_source)
    finally:
        sys.setprofile(prior)
    assert changed == [True]


def test_final_owner_return_callback_cannot_change_detached_output(retained_source):
    changed = []

    def callback(frame, event, value):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code
            is measurement.source.AuthenticatedSurveyPopulationPreparation._checked.__code__
            and caller is not None
            and caller.f_code
            is measurement.qualify_puf55_survey_ss_measurement.__code__
            and "result" in caller.f_locals
        ):
            result = caller.f_locals["result"]
            result.tax_unit.iloc[
                0, result.tax_unit.columns.get_loc(measurement.TOTAL)
            ] = 999999.0
            changed.append(True)

    prior = sys.getprofile()
    try:
        sys.setprofile(callback)
        with pytest.raises(ValueError, match="FINAL_PROJECTION"):
            measurement.qualify_puf55_survey_ss_measurement(retained_source)
    finally:
        sys.setprofile(prior)
    assert changed == [True]


def test_changed_retained_role_cannot_be_routed_as_missing_report(retained_source):
    state = retained_source._checked()[2]
    people = state.frame.person
    original = people.tax_unit_role_input.copy(deep=True)
    try:
        people.loc[people.index[0], "tax_unit_role_input"] = "DEPENDENT"
        # Guarantee a changed role even if the first fixture person was already
        # a dependent. The pure preparation seal rejects before numerical routing.
        if people.tax_unit_role_input.equals(original):
            people.loc[people.index[0], "tax_unit_role_input"] = "HEAD"
        with pytest.raises(ValueError, match="PREPARED_FRAME_CHANGED"):
            measurement.qualify_puf55_survey_ss_measurement(retained_source)
    finally:
        people["tax_unit_role_input"] = original
