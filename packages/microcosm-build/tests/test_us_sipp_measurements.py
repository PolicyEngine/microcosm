"""Invented source measurements only: 2023 dictionary and guide contracts."""

from dataclasses import FrozenInstanceError, replace
from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext

import pytest

from microcosm.build.sipp_measurements import (
    SIPP_2023_MASTER_SHA256,
    MeasurementError,
    SippRow,
    SippSource,
    annual_earnings,
    annual_once,
    annual_retirement,
    december_households,
    select_household_targets,
)

SOURCE = SippSource("a" * 64, SIPP_2023_MASTER_SHA256, 2023, 2022)


def row(month=12, person="101", residence="100001", **values):
    return SippRow(
        {
            "SSUID": "sample-A",
            "PNUM": person,
            "MONTHCODE": str(month),
            "ERESIDENCEID": residence,
            "TAGE": "40",
            "TAGE_EHC": "39",
            "TPEARN_ALT": "100",
            "APEARN_ALT": "9",
            "RMNUMJOBS": "1",
            "AMNUMJOBS": "9",
            "RMESR": "1",
            "AMESR": "9",
            "TRETINCAMT": "0",
            "ARETINCAMT": "9",
            "WPFINWGT": "1000.25",
            **values,
        }
    )


def year(**values):
    return [row(month=m, **values) for m in range(1, 13)]


def no_job(**values):
    return year(TPEARN_ALT="", APEARN_ALT="0", RMNUMJOBS="0", RMESR="8", **values)


def test_signed_current_total_earnings_not_wages_or_december_annualization():
    rows = year()
    rows[0] = row(month=1, TPEARN_ALT="-9999")
    got = annual_earnings(rows, source=SOURCE)
    assert got.value == Decimal(-8899)
    assert got.name == "sipp_current_total_earnings_alt_annual"
    assert got.concept == "current_total_earnings"
    assert got.basis == "alternate_equal_month"
    assert got.status == "complete_published"
    assert {o.quality for o in got.observations} == {
        "component_derived_allocation_unresolved"
    }


@pytest.mark.parametrize(
    "text", ["-100000998", "-100000999", "NaN", "Infinity", "1e400", "1.5", None, False]
)
def test_missing_and_invalid_amounts_never_become_losses_or_zero(text):
    got = annual_earnings(year(TPEARN_ALT=text), source=SOURCE)
    assert got.value is None
    assert got.reasons
    assert all(o.origin != "structural_zero_no_job_v1" for o in got.observations)


def test_literal_zero_and_structural_zero_have_distinct_origins():
    published = annual_earnings(year(TPEARN_ALT="0"), source=SOURCE)
    structural = annual_earnings(no_job(), source=SOURCE)
    assert published.value == structural.value == 0
    assert published.status == "complete_published"
    assert structural.status == "complete_all_structural_zero"
    assert structural.observations[0].raw_fields[0].value == ""
    assert structural.observations[0].flag == 0
    assert (
        structural.observations[0].quality == "component_derived_allocation_unresolved"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"TPEARN_ALT": None},
        {"TPEARN_ALT": "-100000998"},
        {"APEARN_ALT": "9"},
        {"APEARN_ALT": "-99"},
        {"RMNUMJOBS": "1"},
        {"RMNUMJOBS": ""},
        {"RMESR": "1"},
        {"RMESR": "2"},
        {"RMESR": ""},
        {"AMNUMJOBS": "0"},
        {"AMNUMJOBS": "-99"},
        {"AMESR": "0"},
        {"AMESR": "10"},
        {"TAGE": "14"},
        {"TAGE": ""},
        {"TAGE": "91"},
    ],
)
def test_each_no_job_disqualifier_remains_unknown(change):
    values = dict(TPEARN_ALT="", APEARN_ALT="0", RMNUMJOBS="0", RMESR="8")
    values.update(change)
    got = annual_earnings(year(**values), source=SOURCE)
    assert got.value is None
    assert got.reasons


def test_no_job_uses_interview_age_and_explicit_contradictions_refuse():
    rows = no_job(TAGE="15", TAGE_EHC="14")
    assert annual_earnings(rows, source=SOURCE).value == 0
    rows[0] = replace(rows[0], contradictions=("active_job_source_evidence",))
    got = annual_earnings(rows, source=SOURCE)
    assert got.value is None
    assert "active_job_source_evidence" in got.observations[0].reasons


def test_published_amount_survives_no_job_employment_status():
    got = annual_earnings(year(RMESR="8"), source=SOURCE)
    assert got.value == 1200


@pytest.mark.parametrize("flag", range(1, 10))
def test_all_applicable_flags_preserved_without_promoting_flag_nine(flag):
    got = annual_earnings(year(APEARN_ALT=str(flag)), source=SOURCE)
    assert got.value == 1200
    assert {o.flag for o in got.observations} == {flag}
    assert (got.observations[0].quality == "reported") == (flag == 1)


@pytest.mark.parametrize("flag", ["0", "-99", "10", "", None])
def test_nonapplicable_amount_status_cannot_validate_numeric_value(flag):
    assert annual_earnings(year(APEARN_ALT=flag), source=SOURCE).value is None


def test_complete_mixed_year_and_missing_months():
    rows = year()
    rows[0] = no_job()[0]
    got = annual_earnings(rows, source=SOURCE)
    assert got.value == 1100
    assert got.status == "complete_with_structural_zero"
    partial = annual_earnings(rows[1:], source=SOURCE)
    assert partial.value is None
    assert "missing_month:1" in partial.reasons


@pytest.mark.parametrize(
    "alter", ["duplicate", "other_person", "missing_key", "invalid_month"]
)
def test_invalid_person_month_identity_refuses(alter):
    rows = year()
    if alter == "duplicate":
        rows.append(rows[0])
    if alter == "other_person":
        rows[0] = row(month=1, person="102")
    if alter == "missing_key":
        rows[0] = row(month=1, PNUM=None)
    if alter == "invalid_month":
        rows[0] = row(month=13)
    with pytest.raises(MeasurementError):
        annual_earnings(rows, source=SOURCE)


def test_retirement_sums_published_months_without_reconstructing_components():
    rows = [
        row(month=m, TRETINCAMT="200" if m <= 6 else "300", TLMPAMT="1200")
        for m in range(1, 13)
    ]
    got = annual_retirement(rows, source=SOURCE)
    assert got.value == 3000
    assert got.name == "sipp_broad_retirement_recode_annual"
    assert got.concept == "sipp_broad_retirement_recode"
    rows[0] = row(month=1, TRETINCAMT="", ARETINCAMT="0", TAGE_EHC="14")
    assert annual_retirement(rows, source=SOURCE).value is None


def test_retirement_zero_is_component_derived_not_reported():
    got = annual_retirement(year(), source=SOURCE)
    assert got.value == 0
    assert got.observations[0].quality == "component_derived_allocation_unresolved"


def test_annual_once_needs_agreeing_values_not_twelve_repetitions():
    rows = [
        row(month=1, TLIFEAMT="1200", ALIFEAMT="1", ELIFEYN="1", ALIFEYN="1"),
        row(month=2, TLIFEAMT="1200.0", ALIFEAMT="2", ELIFEYN="1", ALIFEYN="2"),
        row(month=3, TLIFEAMT="", ALIFEAMT="0", ELIFEYN="", ALIFEYN="0"),
    ]
    got = annual_once(rows, "TLIFEAMT", source=SOURCE)
    assert got.value == 1200
    assert got.status == "annual_once_with_missing_repetitions"
    assert got.observations[0].origin == "published_source_value"
    assert [o.flag for o in got.observations] == [1, 2, 0]
    assert len(got.observations) == 3


def test_annual_once_disagreement_and_contradictory_annual_route_refuse():
    rows = [
        row(
            month=1,
            TIRA_INC_AMT="1200",
            AIRA_INC_AMT="1",
            EIRA_INC_YN="1",
            AIRA_INC_YN="1",
        ),
        row(
            month=2,
            TIRA_INC_AMT="1300",
            AIRA_INC_AMT="1",
            EIRA_INC_YN="1",
            AIRA_INC_YN="1",
        ),
    ]
    assert annual_once(rows, "TIRA_INC_AMT", source=SOURCE).value is None
    rows[1] = row(
        month=2, TIRA_INC_AMT="", AIRA_INC_AMT="0", EIRA_INC_YN="2", AIRA_INC_YN="1"
    )
    assert annual_once(rows, "TIRA_INC_AMT", source=SOURCE).value is None


def test_december_membership_and_weights_precede_target_selection():
    rows = [
        row(person="101", WPFINWGT="1000.25"),
        row(person="102", WPFINWGT="2000.75"),
        row(person="103", residence="100002", WPFINWGT="9.5"),
    ]
    got = december_households(rows, source=SOURCE, complete_membership=True)
    assert len(got) == 2
    assert got[0].weight.value == Decimal("1500.50")
    assert got[0].members == (("sample-A", 101), ("sample-A", 102))
    selected = select_household_targets(got, (("sample-A", 102),))
    assert selected == (got[0],)
    assert (
        december_households(
            list(reversed(rows)), source=SOURCE, complete_membership=True
        )
        == got
    )
    assert got[0].person_weights[1].value == Decimal("2000.75")


def test_zero_weight_is_in_mean_and_decimal_precision_not_ambient():
    with localcontext() as ctx:
        ctx.prec = 3
        got = december_households(
            [row(person="101", WPFINWGT="0"), row(person="102", WPFINWGT="2000.75")],
            source=SOURCE,
            complete_membership=True,
        )
    assert got[0].weight.value == Decimal("1000.375")


@pytest.mark.parametrize("weight", ["", "NaN", "-999.00000000", "-1", "10000001", None])
def test_missing_or_invalid_member_weight_refuses_requested_mean(weight):
    got = december_households(
        [row(), row(person="102", WPFINWGT=weight)],
        source=SOURCE,
        complete_membership=True,
    )
    assert got[0].weight.value is None
    with pytest.raises(MeasurementError):
        got[0].weight.require_value()


@pytest.mark.parametrize(
    "change",
    [{"ERESIDENCEID": None}, {"ERESIDENCEID": "-999"}, {"PNUM": "100"}, {"SSUID": ""}],
)
def test_invalid_december_key_never_falls_back_to_ssuid(change):
    with pytest.raises(MeasurementError):
        december_households([row(**change)], source=SOURCE, complete_membership=True)


def test_duplicate_members_and_unasserted_complete_roster_refuse():
    with pytest.raises(MeasurementError):
        december_households([row(), row()], source=SOURCE, complete_membership=True)
    with pytest.raises(MeasurementError):
        december_households([row()], source=SOURCE, complete_membership=False)


@pytest.mark.parametrize(
    "second",
    [
        {"THVAL_VEH": "6000"},
        {"AHVAL_VEH": "1"},
        {"THVAL_VEH": ""},
        {"AHVAL_VEH": "-99"},
    ],
)
def test_household_copy_value_or_status_conflict_remains_unknown(second):
    first = dict(THVAL_VEH="5000", AHVAL_VEH="9")
    got = december_households(
        [row(person="101", **first), row(person="102", **(first | second))],
        source=SOURCE,
        copied_fields=("THVAL_VEH",),
        complete_membership=True,
    )
    assert got[0].measurements[0].value is None
    with pytest.raises(MeasurementError):
        got[0].measurements[0].require_value()


def test_copied_values_collapse_once_and_boat_value_not_car_count():
    fields = dict(
        TVEH_NUM="0",
        AVEH_NUM="1",
        THVAL_VEH="5000",
        AHVAL_VEH="9",
        THEQ_VEH="-9999",
        AHEQ_VEH="9",
        AVEH1VAL="1",
        AVEH2VAL="1",
        AVEH3VAL="1",
    )
    got = december_households(
        [row(person="101", **fields), row(person="102", **fields)],
        source=SOURCE,
        copied_fields=("TVEH_NUM", "THVAL_VEH", "THEQ_VEH"),
        complete_membership=True,
    )
    count, value, equity = got[0].measurements
    assert (count.value, value.value, equity.value) == (0, 5000, -9999)
    assert value.observations[0].quality == "component_derived_allocation_unresolved"
    assert count.basis == "reference_period_owned_cars_trucks_vans"


def test_input_mutation_cannot_change_results_or_source_identity():
    rows = year()
    got = annual_earnings(rows, source=SOURCE)
    rows[0].values["TPEARN_ALT"] = "99999"
    assert got.value == 1200
    assert got.observations[0].raw_fields[0].value == "100"
    with pytest.raises(FrozenInstanceError):
        got.value = Decimal(9)


@pytest.mark.parametrize(
    "source",
    [
        replace(SOURCE, reference_year=2024),
        replace(SOURCE, survey_year=2024),
        replace(SOURCE, metadata_sha256="b" * 64),
        replace(SOURCE, data_sha256="bad"),
    ],
)
def test_source_identity_mismatch_refuses(source):
    with pytest.raises(MeasurementError):
        annual_earnings(year(), source=source)


def test_target_year_transfer_and_unsupported_concepts_refuse():
    with pytest.raises(MeasurementError):
        annual_earnings(year(), source=SOURCE, target_year=2024)
    with pytest.raises(MeasurementError):
        december_households(
            [row()], source=SOURCE, target_year=2024, complete_membership=True
        )
    with pytest.raises(MeasurementError):
        annual_once(year(), "TPTOTINC", source=SOURCE)


def test_supplied_active_public_job_prevents_structural_zero():
    got = annual_earnings(no_job(EJB1_BMONTH="1", EJB1_EMONTH="12"), source=SOURCE)
    assert got.value is None
    assert "active_public_job:1" in got.observations[0].reasons


def test_supplied_other_earnings_recode_prevents_structural_zero():
    assert annual_earnings(no_job(TPEARN="10"), source=SOURCE).value is None


@pytest.mark.parametrize("status", ["10", "False", False])
def test_invalid_amount_status_on_missing_annual_repetition_is_not_hidden(status):
    rows = [
        row(month=1, TLIFEAMT="1200", ALIFEAMT="1", ELIFEYN="1", ALIFEYN="1"),
        row(month=2, TLIFEAMT="", ALIFEAMT=status, ELIFEYN="", ALIFEYN="0"),
    ]
    assert annual_once(rows, "TLIFEAMT", source=SOURCE).value is None


def test_annual_once_or_route_cannot_hide_invalid_companion_control():
    rows = [
        row(
            TLMPAMT="1200",
            ALMPAMT="1",
            ELMPTYP1YN="1",
            ELMPTYP2YN="2",
            ELMPTYP4YN="1",
            ALMPTYPYN="10",
        )
    ]
    assert annual_once(rows, "TLMPAMT", source=SOURCE).value is None


def test_invalid_lexical_type_is_retained_without_retaining_mutable_object():
    rows = year(APEARN_ALT=False)
    got = annual_earnings(rows, source=SOURCE)
    evidence = next(f for f in got.observations[0].raw_fields if f.name == "APEARN_ALT")
    assert evidence.input_type == "bool"
    assert evidence.value is None


@pytest.mark.parametrize("contradictions", [None, "active_job", ["active_job"]])
def test_malformed_contradiction_evidence_is_sanitized(contradictions):
    rows = [replace(row(), contradictions=contradictions)]
    with pytest.raises(MeasurementError, match="INVALID_CONTRADICTION_EVIDENCE"):
        annual_earnings(rows, source=SOURCE)


def test_missing_all_copies_is_unknown_without_inventing_a_conflict():
    got = december_households(
        [
            row(person="101", TVEH_NUM="", AVEH_NUM="0"),
            row(person="102", TVEH_NUM="", AVEH_NUM="0"),
        ],
        source=SOURCE,
        copied_fields=("TVEH_NUM",),
        complete_membership=True,
    )
    assert got[0].measurements[0].value is None
    assert "household_copy_conflict" not in got[0].measurements[0].reasons


def test_annual_sum_zero_from_offsetting_losses_retains_published_origin():
    rows = year(TPEARN_ALT="0")
    rows[0] = row(month=1, TPEARN_ALT="100")
    rows[1] = row(month=2, TPEARN_ALT="-100")
    got = annual_earnings(rows, source=SOURCE)
    assert got.value == 0
    assert got.status == "complete_published"


def test_all_five_annual_once_fields_have_their_own_routing_and_flags():
    cases = [
        ("TLIFEAMT", "ALIFEAMT", {"ELIFEYN": "1", "ALIFEYN": "1"}),
        (
            "TLMPAMT",
            "ALMPAMT",
            {"ELMPTYP1YN": "2", "ELMPTYP2YN": "1", "ELMPTYP4YN": "2", "ALMPTYPYN": "1"},
        ),
        (
            "TROLLAMT",
            "AROLLAMT",
            {"EROLLOVR1": "2", "AROLLOVR1": "1", "EROLLOVR2": "1", "AROLLOVR2": "1"},
        ),
        ("TIRA_INC_AMT", "AIRA_INC_AMT", {"EIRA_INC_YN": "1", "AIRA_INC_YN": "1"}),
        ("TTHR_INC_AMT", "ATHR_INC_AMT", {"ETHR_INC_YN": "1", "ATHR_INC_YN": "1"}),
    ]
    for field, flag, routes in cases:
        assert (
            annual_once(
                [row(**{field: "1200", flag: "9", **routes})], field, source=SOURCE
            ).value
            == 1200
        )


def test_recognized_source_columns_are_snapshotted_without_unrelated_payloads():
    rows = year()
    rows[0].values["unrelated_big_survey_column"] = ["not retained"]
    got = annual_earnings(rows, source=SOURCE)
    assert all(
        f.name != "unrelated_big_survey_column"
        for o in got.observations
        for f in o.raw_fields
    )


def test_target_selection_rejects_ambiguous_or_mixed_source_rosters():
    first = december_households([row()], source=SOURCE, complete_membership=True)
    other = december_households(
        [row()], source=replace(SOURCE, data_sha256="b" * 64), complete_membership=True
    )
    with pytest.raises(MeasurementError):
        select_household_targets((*first, *first), (("sample-A", 101),))
    with pytest.raises(MeasurementError):
        select_household_targets((*first, *other), (("sample-A", 101),))


@pytest.mark.parametrize("mode", ["round_up", "round_down", "trap_all", "small_bounds"])
@pytest.mark.parametrize("measure", ["household_mean", "earnings", "retirement"])
def test_arithmetic_is_independent_of_entire_caller_decimal_context(mode, measure):
    with localcontext() as context:
        if mode == "round_up":
            context.rounding = ROUND_UP
        elif mode == "round_down":
            context.rounding = ROUND_DOWN
        elif mode == "trap_all":
            for signal in context.traps:
                context.traps[signal] = True
        else:
            context.Emax = 2
            context.Emin = -2
            context.clamp = 1
        context.prec = 2
        before = context.copy()
        if measure == "household_mean":
            got = december_households(
                [
                    row(person="101", WPFINWGT="1"),
                    row(person="102", WPFINWGT="0"),
                    row(person="103", WPFINWGT="0"),
                ],
                source=SOURCE,
                complete_membership=True,
            )[0].weight.value
            expected = Decimal("0." + "3" * 80)
        elif measure == "earnings":
            rows = year(TPEARN_ALT="99999999")
            rows[0] = row(month=1, TPEARN_ALT="-99999999")
            got = annual_earnings(rows, source=SOURCE).value
            expected = Decimal("999999990")
        else:
            got = annual_retirement(year(TRETINCAMT="10000000000"), source=SOURCE).value
            expected = Decimal("120000000000")
        assert got == expected
        # Internal rounding must neither mutate settings nor raise flags outside.
        assert str(context) == str(before)
