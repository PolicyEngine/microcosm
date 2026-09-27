"""Invented literals only: native-key proposal semantics, not source authority."""

from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from microcosm.build.us_runtime import current_survey_hours as hours


def acs(age=30, **changes):
    return {
        "SERIALNO": "2024HU0000001",
        "SPORDER": "1",
        "AGEP": str(age),
        "WKHP": "40" if age >= 16 else "",
        "WKL": "1" if age >= 16 else "",
        "FWKHP": "0",
        "WAGP": "0",
        "SEMP": "0",
        **changes,
    }


def donor(number=1, positive=True, **changes):
    return {
        "PERIDNUM": str(number).zfill(22),
        "PH_SEQ": str(number),
        "A_LINENO": "1",
        "A_AGE": "15",
        "HRSWK": "20" if positive else "0",
        "WKSWORK": "10" if positive else "0",
        "WORKYN": "1" if positive else "2",
        "WTEMP": "0" if positive else "2",
        "WRK_CK": "1" if positive else "2",
        "MARSUPWT": "100",
        "I_HRSWK": "0",
        "I_WKSWK": "0",
        "I_WORKYN": "0",
        "I_WTEMP": "0",
        "FL_665": "1",
        **changes,
    }


def propose(rows, donors=(), **kwargs):
    return hours.propose_acs_usual_hours(rows, donors=donors, **kwargs)


@pytest.mark.parametrize("value", ["40", "99"])
@pytest.mark.parametrize(
    "flag,kind",
    [("0", "not_allocated"), ("1", "allocated"), ("", "allocation_unknown")],
)
def test_source_observations_and_allocation_survive(value, flag, kind):
    row = acs(WKHP=value, FWKHP=flag)
    result = hours.recode_acs_usual_hours(row)
    assert result.hours == float(value)
    assert result.provenance == "acs_native_hours_" + kind
    assert dict(result.raw) == row


def test_nonwork_is_source_completion_not_observed_zero_and_esr_is_irrelevant():
    observed = acs(ESR="3", hours_worked_last_week="0")
    assert hours.recode_acs_usual_hours(observed).hours == 40
    result = hours.recode_acs_usual_hours(acs(WKHP="", WKL="2", WAGP="500"))
    assert result.hours == 0
    assert result.provenance == "acs_source_nonwork_completion"


@pytest.mark.parametrize("age", [0, 14, 15])
def test_survey_universe_absence_is_not_zero(age):
    result = hours.recode_acs_usual_hours(acs(age))
    assert result.hours is None
    assert result.provenance == "acs_source_universe_unavailable"


@pytest.mark.parametrize(
    "changes",
    [
        {"WKHP": "0"},
        {"WKHP": "1.5"},
        {"WKHP": "b"},
        {"WKHP": "100"},
        {"FWKHP": "2"},
        {"WKL": "4"},
        {"WKL": "2"},
        {"AGEP": ""},
        {"AGEP": "15"},
        {"WAGP": "NaN"},
        {"SERIALNO": "2024HU1"},
        {"SPORDER": "0"},
        {"SPORDER": True},
    ],
)
def test_invalid_acs_literals_and_contradictions_refuse(changes):
    with pytest.raises(ValueError, match="NATIVE_HOURS_"):
        hours.recode_acs_usual_hours(acs(**changes))


def test_missing_header_is_not_a_blank_answer():
    row = acs()
    del row["WKHP"]
    with pytest.raises(ValueError, match="SOURCE_FIELDS"):
        hours.recode_acs_usual_hours(row)


def test_policy_is_explicit_and_unknown_earnings_stay_unknown():
    row = acs(14, WAGP="", SEMP="")
    with pytest.raises(ValueError, match="UNDER15_POLICY_REQUIRED"):
        propose([row])
    p = propose([row], under15_policy=hours.UNDER15_POLICY).proposals[0]
    assert p.hours == 0 and p.provenance == "under15_explicit_modeled_zero"
    assert p.unknown_earnings == ("WAGP", "SEMP")
    assert dict(p.raw)["WKHP"] == ""


@pytest.mark.parametrize("changes", [{"WAGP": "1"}, {"SEMP": "-1"}, {"SEMP": "1"}])
def test_under15_policy_refuses_earnings_signal(changes):
    with pytest.raises(ValueError, match="UNDER15_EARNINGS"):
        propose([acs(14, **changes)], under15_policy=hours.UNDER15_POLICY)


def test_age15_and_unresolved_adults_never_get_a_generic_fallback():
    with pytest.raises(ValueError, match="AGE15_DONORS_REQUIRED"):
        propose([acs(15)], donors=[donor()])
    with pytest.raises(ValueError, match="AGE15_DONORS_REQUIRED"):
        propose([acs(15)], age15_policy=hours.AGE15_POLICY)
    with pytest.raises(ValueError, match="UNRESOLVED_ADULT"):
        propose([acs(16, WKHP="")], donors=[donor()], age15_policy=hours.AGE15_POLICY)


def test_asec_temporary_followup_is_work_and_flags_remain_separate():
    row = donor(WORKYN="2", WTEMP="1", I_HRSWK="9", FL_665="3")
    p = hours.recode_asec_age15_hours(row)
    assert p.hours == 20 and p.provenance == "asec_positive_source_hours"
    result = propose([acs(15, FWKHP="")], [row], age15_policy=hours.AGE15_POLICY)
    draw = result.proposals[0]
    assert draw.provenance == "age15_weighted_empirical_draw"
    assert draw.donor_key == p.key and draw.donor_allocation_flags == p.allocation_flags
    assert draw.allocation_flags == (("FWKHP", None),)
    assert not result.source_authenticated
    assert not result.complete_donor_cohort_authenticated
    assert not result.release_qualified


@pytest.mark.parametrize(
    "changes",
    [
        {"A_AGE": "14"},
        {"A_AGE": "16"},
        {"MARSUPWT": "0"},
        {"MARSUPWT": "1.5"},
        {"HRSWK": "100"},
        {"WKSWORK": "0"},
        {"WRK_CK": "2"},
        {"WORKYN": "2", "WTEMP": "2"},
        {"I_HRSWK": "2"},
        {"FL_665": "4"},
        {"PERIDNUM": "123"},
    ],
)
def test_unqualified_asec_domains_refuse(changes):
    with pytest.raises(ValueError, match="NATIVE_HOURS_"):
        hours.recode_asec_age15_hours(donor(**changes))


def test_asec_zero_is_nonwork_completion_and_initial_no_is_not_sufficient():
    p = hours.recode_asec_age15_hours(donor(positive=False))
    assert p.hours == 0 and p.provenance == "asec_source_nonwork_completion"
    with pytest.raises(ValueError, match="ASEC_WORK_CONTRADICTION"):
        hours.recode_asec_age15_hours(donor(positive=False, WTEMP="1"))


@pytest.mark.parametrize("positive", [False, True])
def test_complete_supplement_nonresponse_is_preserved_not_rejected(positive):
    # Census2025 FL_665: 0=complete supplement nonresponse; valid source status,
    # not proof of observed hours or independently qualified donor eligibility.
    # https://api.census.gov/data/2025/cps/asec/mar/variables/FL_665.json
    row = donor(positive=positive, FL_665="0", I_HRSWK="1")
    recoded = hours.recode_asec_age15_hours(row)
    assert dict(recoded.raw)["FL_665"] == "0"
    assert dict(recoded.allocation_flags)["FL_665"] == 0
    batch = propose([acs(15)], [row], age15_policy=hours.AGE15_POLICY)
    draw = batch.proposals[0]
    assert draw.hours == (20 if positive else 0)
    assert dict(draw.donor_allocation_flags)["FL_665"] == 0
    assert draw.provenance == "age15_weighted_empirical_draw"
    assert not batch.complete_donor_cohort_authenticated
    assert not batch.release_qualified


def test_temporary_niu_is_not_an_explicit_no_against_final_work():
    result = hours.recode_asec_age15_hours(donor(WORKYN="2", WTEMP="0", WRK_CK="1"))
    assert result.hours == 20
    assert dict(result.raw)["WTEMP"] == "0"
    assert result.provenance == "asec_positive_source_hours"


def test_weighted_cdf_boundaries_preserve_actual_donor_and_provenance():
    donors = [donor(1, False, MARSUPWT="100"), donor(2, True, MARSUPWT="300")]
    for uniform, expected in [
        (0, 0),
        (np.nextafter(0.25, 0), 0),
        (0.25, 20),
        (np.nextafter(1.0, 0), 20),
    ]:
        with patch.object(hours, "native_hours_uniform", return_value=uniform):
            p = propose([acs(15)], donors, age15_policy=hours.AGE15_POLICY).proposals[0]
        assert p.hours == expected
        assert (
            p.donor_key == hours.recode_asec_age15_hours(donors[int(expected > 0)]).key
        )


def test_permutation_batches_and_nonidentity_earnings_do_not_change_draws():
    rows = [acs(15, SERIALNO=f"2024HU{i:07d}") for i in range(1, 25)]
    donors = [donor(2), donor(1, False), donor(3, I_HRSWK="1")]
    before = deepcopy((rows, donors))
    full = propose(rows, donors, age15_policy=hours.AGE15_POLICY)
    shuffled = propose(
        rows[::-1], donors[1:] + donors[:1], age15_policy=hours.AGE15_POLICY
    )
    assert full.proposals == shuffled.proposals[::-1]
    batches = tuple(
        p
        for batch in (rows[:7], rows[7:])
        for p in propose(batch, donors, age15_policy=hours.AGE15_POLICY).proposals
    )
    assert batches == full.proposals
    changed = propose(
        [{**r, "WAGP": "500", "SEMP": "-10"} for r in rows],
        donors,
        age15_policy=hours.AGE15_POLICY,
    )
    assert [(p.hours, p.donor_key) for p in changed.proposals] == [
        (p.hours, p.donor_key) for p in full.proposals
    ]
    assert (rows, donors) == before


def test_cdf_matches_reviewed_weight_unit_conversion_order():
    donors = [donor(1, False, MARSUPWT="117"), donor(2, True, MARSUPWT="192637")]
    weights = np.asarray([117, 192637], dtype=float) / 100.0
    weights /= np.max(weights)
    boundary = np.cumsum(weights / weights.sum())[0]
    for uniform, expected in ((np.nextafter(boundary, 0), 0), (boundary, 20)):
        with patch.object(hours, "native_hours_uniform", return_value=uniform):
            p = propose([acs(15)], donors, age15_policy=hours.AGE15_POLICY).proposals[0]
        assert p.hours == expected


def test_native_key_is_not_row_rank_and_zero_padding_cannot_duplicate_identity():
    a = hours.recode_acs_usual_hours(acs(SPORDER="02")).key
    b = hours.recode_acs_usual_hours(acs(SPORDER="2")).key
    assert a == b
    assert hours.native_hours_uniform(a) == hours.native_hours_uniform(b)
    assert a.household == "2024HU0000001"
    assert hours.native_hours_uniform(a) != hours.native_hours_uniform(
        replace(a, line=10)
    )
    with pytest.raises(ValueError, match="DUPLICATE_RECIPIENT_KEY"):
        propose([acs(SPORDER="02"), acs(SPORDER="2")])
    with pytest.raises(ValueError, match="KEY_TYPE"):
        replace(a, line=True)


@pytest.mark.parametrize(
    "second", [donor(), donor(2, PH_SEQ="1"), donor(2, PERIDNUM="1".zfill(22))]
)
def test_duplicate_donor_native_identities_refuse(second):
    with pytest.raises(ValueError, match="DUPLICATE_"):
        propose([acs(15)], [donor(), second], age15_policy=hours.AGE15_POLICY)


def test_maximum_hash_cannot_make_uniform_one():
    class Maximum:
        def digest(self):
            return b"\xff" * 8

    with patch.object(hours.hashlib, "blake2b", return_value=Maximum()):
        assert (
            hours.native_hours_uniform(hours.recode_acs_usual_hours(acs()).key)
            == (2**53 - 1) / 2**53
        )


def test_pure_proposal_does_not_read_files_or_claim_authority():
    with patch("builtins.open", side_effect=AssertionError("I/O forbidden")):
        result = propose([acs()])
    document = asdict(result)
    assert document["source_authenticated"] is False
    assert document["complete_donor_cohort_authenticated"] is False
    assert document["release_qualified"] is False
    assert hours.TARGET == "weekly_hours_worked_before_lsr"
