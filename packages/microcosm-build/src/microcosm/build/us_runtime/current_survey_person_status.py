"""Published person-status items, without statutory or annual eligibility aliases.

These pure recodes describe arbitrary literals. Source authority is borrowed by
``current_survey_person_status_source`` from the actual survey preparation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType

PROTOCOL = "microcosm.us.current-survey-person-status.v1"
ASEC_PERIOD = "cps_status_published_2025_may_retain_earlier_rotation_response"
ACS_PERIOD = "acs_condition_status_collected_2024"
ASEC_STUDENT_PERIOD = "preceding_survey_week_2025_not_income_year_2024"
ACS_STUDENT_PERIOD = "last_three_months_at_2024_collection"


@dataclass(frozen=True)
class DifficultyItem:
    observation: str
    asec: str
    asec_flag: str
    acs: str
    acs_flag: str
    acs_min_age: int


ITEMS = (
    DifficultyItem(
        "survey_hearing_difficulty", "PEDISEAR", "PXDISEAR", "DEAR", "FDEARP", 0
    ),
    DifficultyItem(
        "survey_vision_difficulty", "PEDISEYE", "PXDISEYE", "DEYE", "FDEYEP", 0
    ),
    DifficultyItem(
        "survey_cognitive_difficulty", "PEDISREM", "PXDISREM", "DREM", "FDREMP", 5
    ),
    DifficultyItem(
        "survey_ambulatory_difficulty", "PEDISPHY", "PXDISPHY", "DPHY", "FDPHYP", 5
    ),
    DifficultyItem(
        "survey_self_care_difficulty", "PEDISDRS", "PXDISDRS", "DDRS", "FDDRSP", 5
    ),
    DifficultyItem(
        "survey_independent_living_difficulty",
        "PEDISOUT",
        "PXDISOUT",
        "DOUT",
        "FDOUTP",
        15,
    ),
)
PX_CODES = MappingProxyType(
    {
        -1: "not_allocated",
        0: "value_no_change",
        1: "blank_no_change",
        2: "dont_know_no_change",
        3: "refused_no_change",
        10: "value_to_value",
        11: "blank_to_value",
        12: "dont_know_to_value",
        13: "refused_to_value",
        20: "value_to_longitudinal_value",
        21: "blank_to_longitudinal_value",
        22: "dont_know_to_longitudinal_value",
        23: "refused_to_longitudinal_value",
        30: "value_to_allocated_value_long",
        31: "blank_to_allocated_value_long",
        32: "dont_know_to_allocated_value_long",
        33: "refused_to_allocated_value_long",
        40: "value_to_allocated_value",
        41: "blank_to_allocated_value",
        42: "dont_know_to_allocated_value",
        43: "refused_to_allocated_value",
        50: "value_to_blank",
        52: "dont_know_to_blank",
        53: "refused_to_blank",
    }
)
AX_CODES = MappingProxyType(
    {0: "no_change_or_children_or_armed_forces", 4: "allocated"}
)
ACS_FLAG_CODES = MappingProxyType({0: "not_allocated", 1: "allocated"})
ASEC_COLUMNS = (
    "PERIDNUM",
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "PRPERTYP",
    *(i.asec for i in ITEMS),
    *(i.asec_flag for i in ITEMS),
    "PRDISFLG",
    "A_ENRLW",
    "A_FTPT",
    "A_HSCOL",
    "AXENRLW",
    "AXFTPT",
    "AXHSCOL",
)
ACS_COLUMNS = (
    "SERIALNO",
    "SPORDER",
    "AGEP",
    *(i.acs for i in ITEMS),
    *(i.acs_flag for i in ITEMS),
    "DIS",
    "FDISP",
    "SCH",
    "SCHG",
    "FSCHP",
)
RAW_COLUMNS = tuple(dict.fromkeys((*ASEC_COLUMNS, *ACS_COLUMNS)))
OBSERVATIONS = (
    *(i.observation for i in ITEMS),
    "survey_any_applicable_difficulty",
    "survey_publisher_disability_recode",
    "survey_full_time_college_student_last_week",
    "survey_college_attended_last_3_months",
)


def require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_PERSON_STATUS_" + reason)


def literal_code(token, named, *, width=2):
    """Named codes only; the header's numeric range does not define answers.

    Numeric zero padding is accepted within printed width and retained in the
    literal. ACS dictionary blank notation ``b``/``bb`` is never a source code.
    """
    require(type(token) is str and len(token) <= 64, "LITERAL_TYPE_OR_BOUND")
    if token == "":
        return None, "missing"
    if (
        re.fullmatch(r"-?[0-9]{1," + str(width) + r"}", token, re.ASCII) is None
        or len(token) > width
        or (token.startswith("-") and int(token) == 0)
    ):
        return None, "malformed"
    value = int(token)
    return (value, "named") if value in named else (None, "unlabelled_code")


def _observed(token, *, applicable, niu):
    code, status = literal_code(token, (-1, 1, 2))
    if applicable is None:
        return None, "unresolved_universe"
    if not applicable:
        return (
            (None, "outside_universe")
            if token == niu
            else (None, "contradictory_outside_universe")
        )
    if code in (1, 2):
        return code == 1, "published_yes" if code == 1 else "published_no"
    if code == -1 or (niu == "" and token == ""):
        return (
            None,
            "missing_applicable_item"
            if token == ""
            else "contradictory_niu_in_universe",
        )
    return None, status


def _asec_student(row, age, person_type):
    enrolled = literal_code(row["A_ENRLW"], (0, 1, 2), width=1)[0]
    full_time = literal_code(row["A_FTPT"], (0, 1, 2), width=1)[0]
    level = literal_code(row["A_HSCOL"], (0, 1, 2), width=1)[0]
    if not 16 <= age <= 54 or person_type in (1, 3):
        return (
            (None, "outside_universe")
            if (enrolled, full_time, level) == (0, 0, 0)
            else (None, "contradictory_outside_universe")
        )
    if person_type is None:
        return None, "unresolved_person_type"
    if None in (enrolled, full_time, level):
        return None, "unresolved_school_literal"
    if enrolled == 0:
        return (
            (None, "declared_niu")
            if full_time == level == 0
            else (None, "contradictory_school_route")
        )
    if enrolled == 2:
        return (
            (False, "published_not_enrolled_last_week")
            if full_time == level == 0
            else (None, "contradictory_school_route")
        )
    if full_time == 0 or level == 0:
        return None, "contradictory_school_route"
    return full_time == 1 and level == 2, "published_enrollment_combination_last_week"


def _acs_student(row, age):
    school = literal_code(row["SCH"], (1, 2, 3), width=1)[0]
    level = literal_code(row["SCHG"], range(1, 17))[0]
    if age < 3:
        return (
            (None, "outside_universe")
            if row["SCH"] == row["SCHG"] == ""
            else (None, "contradictory_outside_universe")
        )
    if school == 1:
        return (
            (False, "published_no_attendance_last_three_months")
            if row["SCHG"] == ""
            else (None, "contradictory_school_route")
        )
    if school in (2, 3) and level is not None:
        return level in (15, 16), "published_school_level_last_three_months"
    return None, "unresolved_school_route"


def recode_person_status(row, *, survey):
    """Return source-qualified *semantics*, never authority or eligibility.

    An applicable yes can establish the battery observation with incomplete
    other items. False requires every applicable item to be a valid no.
    Contradictory universe coordinates prevent a battery answer.
    """
    require(survey in ("acs", "asec"), "SURVEY")
    columns = ASEC_COLUMNS if survey == "asec" else ACS_COLUMNS
    require(set(columns) <= set(row), "LITERAL_COLUMNS")
    require(
        all(type(row[c]) is str and len(row[c]) <= 64 for c in columns),
        "LITERAL_TYPE_OR_BOUND",
    )
    age_token = row["A_AGE" if survey == "asec" else "AGEP"]
    age, age_status = literal_code(age_token, range(100), width=2)
    require(age_status == "named", "ORIGINAL_AGE")
    person_type = (
        literal_code(row["PRPERTYP"], (1, 2, 3), width=2)[0]
        if survey == "asec"
        else None
    )
    out = {
        "survey_status_source": survey,
        "survey_status_original_age": age,
        "survey_status_observation_year": 2025 if survey == "asec" else 2024,
        "survey_status_reference_period": ASEC_PERIOD
        if survey == "asec"
        else ACS_PERIOD,
        "survey_student_reference_period": ASEC_STUDENT_PERIOD
        if survey == "asec"
        else ACS_STUDENT_PERIOD,
        "survey_student_full_time_measured": survey == "asec",
        "survey_student_annual_five_month_status_validated": False,
        "survey_status_canonical_eligibility_assigned": False,
    }
    for column in columns:
        out["person_status_source_" + column] = row[column]
    # Preserve named codes and parse states separately, including unknown flags.
    domains = (
        {i.asec: ((-1, 1, 2), 2) for i in ITEMS}
        if survey == "asec"
        else {i.acs: ((1, 2), 1) for i in ITEMS}
    )
    if survey == "asec":
        domains.update(
            {
                "PRPERTYP": ((1, 2, 3), 2),
                "PRDISFLG": ((-1, 1, 2), 2),
                **{c: ((0, 1, 2), 1) for c in ("A_ENRLW", "A_FTPT", "A_HSCOL")},
            }
        )
        flags = {
            **{i.asec_flag: PX_CODES for i in ITEMS},
            **{c: AX_CODES for c in ("AXENRLW", "AXFTPT", "AXHSCOL")},
        }
    else:
        domains.update(
            {"DIS": ((1, 2), 1), "SCH": ((1, 2, 3), 1), "SCHG": (range(1, 17), 2)}
        )
        flags = {
            c: ACS_FLAG_CODES for c in (*(i.acs_flag for i in ITEMS), "FDISP", "FSCHP")
        }
    for column, (named, width) in domains.items():
        code, state = literal_code(row[column], named, width=width)
        out["person_status_source_" + column + "__code"] = code
        out["person_status_source_" + column + "__literal_status"] = state
    for column, labels in flags.items():
        code, state = literal_code(
            row[column], labels, width=2 if column.startswith("PX") else 1
        )
        out["person_status_source_" + column + "__code"] = code
        out["person_status_source_" + column + "__literal_status"] = state
        out["person_status_source_" + column + "__meaning"] = labels.get(
            code, "allocation_or_edit_unknown"
        )
    values, statuses, roster = [], [], []
    for item in ITEMS:
        applicable = (
            (None if person_type is None else person_type == 2)
            if survey == "asec"
            else age >= item.acs_min_age
        )
        field = item.asec if survey == "asec" else item.acs
        value, status = _observed(
            row[field], applicable=applicable, niu="-1" if survey == "asec" else ""
        )
        out[item.observation] = value
        out[item.observation + "__known"] = value is not None
        out[item.observation + "__status"] = status
        out[item.observation + "__applicable"] = applicable
        statuses.append(status)
        if applicable:
            roster.append(field)
            values.append(value)
    complete = bool(roster) and all(v is not None for v in values)
    coherent = not any(
        s.startswith("contradictory") or s == "unresolved_universe" for s in statuses
    )
    summary = (
        (True if True in values else False if complete else None)
        if coherent and roster
        else None
    )
    out["survey_any_applicable_difficulty"] = summary
    out["survey_any_applicable_difficulty__known"] = summary is not None
    out["survey_any_applicable_difficulty__complete"] = complete
    out["survey_any_applicable_difficulty__coherent"] = coherent
    out["survey_any_applicable_difficulty__applicable_fields"] = tuple(roster)
    pub = "PRDISFLG" if survey == "asec" else "DIS"
    published, status = _observed(
        row[pub],
        applicable=(None if person_type is None else person_type == 2)
        if survey == "asec"
        else True,
        niu="-1" if survey == "asec" else "",
    )
    out["survey_publisher_disability_recode"] = published
    out["survey_publisher_disability_recode__known"] = published is not None
    out["survey_publisher_disability_recode__status"] = status
    out["survey_publisher_disability_recode__agrees_with_battery"] = (
        summary == published if summary is not None and published is not None else None
    )
    asec_student = (
        _asec_student(row, age, person_type)
        if survey == "asec"
        else (None, "not_measured_by_acs")
    )
    acs_student = (
        _acs_student(row, age) if survey == "acs" else (None, "not_measured_by_asec")
    )
    for name, (value, status) in zip(
        OBSERVATIONS[-2:], (asec_student, acs_student), strict=True
    ):
        out[name] = value
        out[name + "__known"] = value is not None
        out[name + "__status"] = status
    out["survey_school_level_separate_allocation_measured"] = survey == "asec"
    return out
