"""All-age source-code semantics on invented literals; no source/data admission."""

import pytest
from test_us_current_survey_hours import donor

from microcosm.build.us_runtime import current_asec_usual_hours as asec
from microcosm.build.us_runtime import current_survey_hours as common


@pytest.mark.parametrize("age", [15, 16, 40, 80, 85])
@pytest.mark.parametrize("hours", [1, 40, 99])
def test_preserves_source_hours_without_an_earnings_predictor(age, hours):
    row = donor(A_AGE=str(age), HRSWK=str(hours), MARSUPWT="0", FL_665="0", I_HRSWK="9")
    result = asec.recode_asec_usual_hours(row)
    assert result.hours == float(hours)
    assert dict(result.raw) == row
    assert result.provenance == "asec_positive_source_hours"
    assert dict(result.allocation_flags)["FL_665"] == 0
    assert dict(result.allocation_flags)["I_HRSWK"] == 9
    assert result.policy is None and result.donor_key is None


@pytest.mark.parametrize("age", [0, 14, 30])
def test_all_niu_remains_unresolved_and_is_not_observed_zero(age):
    row = donor(
        A_AGE=str(age), HRSWK="0", WKSWORK="0", WORKYN="0", WTEMP="0", WRK_CK="0"
    )
    result = asec.recode_asec_usual_hours(row)
    assert result.hours is None
    assert result.provenance == (
        "asec_source_universe_unavailable"
        if age < 15
        else "asec_work_history_unresolved"
    )
    assert dict(result.raw) == row


@pytest.mark.parametrize("temporary", ["0", "2"])
def test_final_nonwork_can_support_zero_even_when_temporary_answer_is_niu(temporary):
    row = donor(positive=False, A_AGE="55", WTEMP=temporary)
    result = asec.recode_asec_usual_hours(row)
    assert result.hours == 0.0
    assert result.provenance == "asec_source_nonwork_completion"
    assert dict(result.raw)["HRSWK"] == "0"


@pytest.mark.parametrize("temporary", ["0", "1"])
def test_initial_no_is_not_a_final_no(temporary):
    result = asec.recode_asec_usual_hours(
        donor(A_AGE="40", WORKYN="2", WTEMP=temporary)
    )
    assert result.hours == 20.0


@pytest.mark.parametrize(
    "changes",
    [
        {"A_AGE": "14"},
        {"A_AGE": "86"},
        {"HRSWK": "100"},
        {"HRSWK": ""},
        {"WKSWORK": "0"},
        {"WRK_CK": "2"},
        {"WORKYN": "0"},
        {"WORKYN": "2", "WTEMP": "2"},
        {"I_HRSWK": "2"},
        {"FL_665": "4"},
        {"MARSUPWT": "-1"},
        {"HRSWK": 40},
    ],
)
def test_invalid_codes_or_inconsistent_work_history_refuse(changes):
    with pytest.raises(ValueError, match="NATIVE_HOURS_"):
        asec.recode_asec_usual_hours(donor(**changes))


@pytest.mark.parametrize("positive", [False, True])
def test_age15_observation_agrees_with_donor_recoder_without_issuing_authority(
    positive,
):
    row = donor(positive=positive)
    assert asec.recode_asec_usual_hours(row) == common.recode_asec_age15_hours(row)


def test_unanswered_child_work_history_cannot_be_completed_from_an_earnings_value():
    row = donor(
        A_AGE="14",
        HRSWK="0",
        WKSWORK="0",
        WORKYN="0",
        WTEMP="0",
        WRK_CK="0",
        WSAL_VAL="1000",
    )
    result = asec.recode_asec_usual_hours(row)
    assert result.hours is None
    assert result.policy is None
