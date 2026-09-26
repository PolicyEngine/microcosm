"""Published item universes and periods; all observations are invented literals."""

import pytest

from microcosm.build.us_runtime import current_survey_person_status as status


def asec(age=20, person_type="2", **changes):
    row = {c: "0" for c in status.ASEC_COLUMNS}
    row.update(
        PERIDNUM="0000000000000000000001",
        PH_SEQ="00007",
        A_LINENO="1",
        A_AGE=str(age),
        PRPERTYP=person_type,
        PRDISFLG="2",
        A_ENRLW="2",
    )
    for item in status.ITEMS:
        row[item.asec] = "2" if person_type == "2" else "-1"
    if not 16 <= age <= 54 or person_type in ("1", "3"):
        row.update(A_ENRLW="0", A_FTPT="0", A_HSCOL="0")
    if person_type in ("1", "3"):
        row["PRDISFLG"] = "-1"
    row.update(changes)
    return row


def acs(age=20, **changes):
    row = {c: "0" for c in status.ACS_COLUMNS}
    row.update(
        SERIALNO="2024HU0000001",
        SPORDER="1",
        AGEP=str(age),
        DIS="2",
        SCH="1" if age >= 3 else "",
        SCHG="",
    )
    for item in status.ITEMS:
        row[item.acs] = "2" if age >= item.acs_min_age else ""
    row.update(changes)
    return row


@pytest.mark.parametrize("age,count", [(4, 2), (5, 5), (14, 5), (15, 6)])
def test_acs_battery_uses_item_universe_not_six_mandatory_answers(age, count):
    result = status.recode_person_status(acs(age), survey="acs")
    assert result["survey_any_applicable_difficulty"] is False
    assert result["survey_any_applicable_difficulty__complete"]
    assert len(result["survey_any_applicable_difficulty__applicable_fields"]) == count
    if age < 15:
        assert result["survey_independent_living_difficulty"] is None
        assert (
            result["survey_independent_living_difficulty__status"] == "outside_universe"
        )


@pytest.mark.parametrize(
    "age,person_type,known",
    [(14, "2", True), (40, "1", False), (40, "3", False), (40, "0", False)],
)
def test_asec_battery_uses_published_person_type_not_substituted_age(
    age, person_type, known
):
    result = status.recode_person_status(asec(age, person_type), survey="asec")
    assert result["survey_any_applicable_difficulty__known"] is known
    assert result["survey_status_original_age"] == age


@pytest.mark.parametrize("survey", ["acs", "asec"])
def test_a_partial_valid_yes_is_known_but_incomplete_and_never_statutory_blindness(
    survey,
):
    row = acs(DEYE="1", DEAR="") if survey == "acs" else asec(PEDISEYE="1", PEDISEAR="")
    result = status.recode_person_status(row, survey=survey)
    assert result["survey_vision_difficulty"] is True
    assert result["survey_any_applicable_difficulty"] is True
    assert not result["survey_any_applicable_difficulty__complete"]
    assert result["survey_any_applicable_difficulty__coherent"]
    assert result["survey_publisher_disability_recode__agrees_with_battery"] is False
    assert (
        not {"is_blind", "is_disabled", "is_full_time_college_student"} & result.keys()
    )
    assert result["survey_status_canonical_eligibility_assigned"] is False


@pytest.mark.parametrize("survey", ["acs", "asec"])
def test_missing_applicable_no_does_not_become_a_negative_battery(survey):
    row = acs(DEYE="") if survey == "acs" else asec(PEDISEYE="")
    result = status.recode_person_status(row, survey=survey)
    assert result["survey_any_applicable_difficulty"] is None
    assert not result["survey_any_applicable_difficulty__complete"]


@pytest.mark.parametrize("token", ["1", "2", "b", " "])
def test_acs_outside_universe_codes_do_not_manufacture_summary(token):
    result = status.recode_person_status(acs(4, DEYE="1", DOUT=token), survey="acs")
    assert result["survey_vision_difficulty"] is True
    assert result["survey_independent_living_difficulty"] is None
    assert result["survey_any_applicable_difficulty"] is None
    assert not result["survey_any_applicable_difficulty__coherent"]


@pytest.mark.parametrize(
    "token,expected",
    [
        ("-4", "unlabelled_code"),
        ("0", "unlabelled_code"),
        ("", "missing"),
        ("1.0", "malformed"),
        (" 1", "malformed"),
    ],
)
def test_printed_range_and_malformed_dressing_answers_are_not_named_no(token, expected):
    result = status.recode_person_status(asec(PEDISDRS=token), survey="asec")
    assert result["person_status_source_PEDISDRS__literal_status"] == expected
    assert result["survey_self_care_difficulty"] is None
    assert result["survey_any_applicable_difficulty"] is None


@pytest.mark.parametrize(
    "token,meaning",
    [
        ("00", "value_no_change"),
        ("1", "blank_no_change"),
        ("21", "blank_to_longitudinal_value"),
        ("40", "value_to_allocated_value"),
        ("50", "value_to_blank"),
        ("-1", "not_allocated"),
        ("51", "allocation_or_edit_unknown"),
    ],
)
def test_asec_edit_transition_is_preserved_without_nonzero_allocated_collapse(
    token, meaning
):
    result = status.recode_person_status(
        asec(PEDISEYE="1", PXDISEYE=token), survey="asec"
    )
    assert result["person_status_source_PXDISEYE"] == token
    assert result["person_status_source_PXDISEYE__meaning"] == meaning
    assert result["survey_vision_difficulty"] is True


@pytest.mark.parametrize("token", ["1", "2", "3", "", "9"])
def test_unlabelled_student_allocation_does_not_invent_no_or_reported_answer(token):
    result = status.recode_person_status(
        asec(A_ENRLW="1", A_FTPT="1", A_HSCOL="2", AXHSCOL=token), survey="asec"
    )
    assert (
        result["person_status_source_AXHSCOL__meaning"] == "allocation_or_edit_unknown"
    )
    assert result["survey_full_time_college_student_last_week"] is True
    assert result["survey_student_reference_period"] == status.ASEC_STUDENT_PERIOD
    assert not result["survey_student_annual_five_month_status_validated"]


@pytest.mark.parametrize(
    "enrolled,workload,level,expected",
    [
        ("0", "0", "0", None),
        ("2", "0", "0", False),
        ("1", "1", "2", True),
        ("1", "1", "1", False),
        ("1", "2", "2", False),
        ("2", "1", "2", None),
        ("1", "0", "2", None),
        ("1", "1", "", None),
    ],
)
def test_asec_student_combination_describes_only_last_week(
    enrolled, workload, level, expected
):
    result = status.recode_person_status(
        asec(A_ENRLW=enrolled, A_FTPT=workload, A_HSCOL=level), survey="asec"
    )
    assert result["survey_full_time_college_student_last_week"] is expected
    assert result["survey_college_attended_last_3_months"] is None


@pytest.mark.parametrize(
    "school,level,expected",
    [
        ("1", "", False),
        ("2", "15", True),
        ("3", "16", True),
        ("2", "14", False),
        ("1", "15", None),
        ("", "15", None),
        ("2", "bb", None),
    ],
)
def test_acs_college_attendance_has_unknown_full_time_workload(school, level, expected):
    result = status.recode_person_status(
        acs(SCH=school, SCHG=level, FSCHP="1", FSCHGP="0"), survey="acs"
    )
    assert result["survey_college_attended_last_3_months"] is expected
    assert result["survey_full_time_college_student_last_week"] is None
    assert not result["survey_student_full_time_measured"]
    assert result["survey_school_level_separate_allocation_measured"]
    assert "SCHL" not in status.ACS_COLUMNS and "FSCHGP" in status.ACS_COLUMNS
    assert result["person_status_source_FSCHP__meaning"] == "allocated"
    assert result["person_status_source_FSCHGP__meaning"] == "not_allocated"


@pytest.mark.parametrize("token", [None, True, 1, 1.0, "0" * 65])
def test_nonliteral_or_unbounded_source_values_refuse(token):
    with pytest.raises(ValueError, match="LITERAL_TYPE_OR_BOUND"):
        status.recode_person_status(asec(PEDISEYE=token), survey="asec")


@pytest.mark.parametrize("token", ["01", "02", "-1"])
def test_acs_item_width_agrees_with_literal_code_knownness(token):
    result = status.recode_person_status(acs(DEYE=token), survey="acs")
    assert result["person_status_source_DEYE__code"] is None
    assert result["survey_vision_difficulty"] is None
    assert not result["survey_vision_difficulty__known"]


@pytest.mark.parametrize("token", ["", "-1", "9"])
def test_unreadable_student_literal_is_unresolved_even_outside_age_universe(token):
    result = status.recode_person_status(asec(age=55, A_HSCOL=token), survey="asec")
    assert result["survey_full_time_college_student_last_week"] is None
    assert (
        result["survey_full_time_college_student_last_week__status"]
        == "unresolved_school_literal"
    )
