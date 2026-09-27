"""Literal Census household classifications, without population/source authority.

Definitions: 2024 ACS Subject Definitions, Relationship to Householder / Own Child
and Household Type (pp. 88-91), and 2025 ASEC dictionary pp. 8-9, 21, 23.
See docs/native-primary-family-source.md for the exact official references.
Only original published relationship, marital, age and roster evidence is used.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

PROTOCOL = "microcosm.us.current-survey-primary-family.v1"
ASEC_HOUSEHOLD_FIELDS = ("H_SEQ", "H_HHTYPE", "H_LIVQRT", "HRHTYPE", "H_NUMPER")
ASEC_PERSON_FIELDS = (
    "PERIDNUM",
    "PH_SEQ",
    "A_LINENO",
    "A_EXPRRP",
    "A_AGE",
    "A_MARITL",
    "A_SPOUSE",
)
ACS_HOUSEHOLD_FIELDS = ("SERIALNO", "TYPEHUGQ", "NP")
ACS_PERSON_FIELDS = ("SERIALNO", "SPORDER", "RELSHIPP", "AGEP", "MAR")
READSETS = (
    ASEC_HOUSEHOLD_FIELDS,
    ASEC_PERSON_FIELDS,
    ACS_HOUSEHOLD_FIELDS,
    ACS_PERSON_FIELDS,
)
HOUSEHOLD_COUNT = "census_household_count"
HOUSEHOLD_POPULATION = "census_household_population"
PRIMARY = "census_married_primary_family"
CHILD_U18 = "census_married_primary_family_own_child_u18"
CHILD_U6 = "census_married_primary_family_own_child_u6"
CHILD_COUNT = "census_primary_married_own_children_u18_count"
SPOUSE_PAIRS = "census_all_resident_spouse_pairs"
SIZE_BINS = tuple(f"census_household_size_{i}" for i in range(1, 7)) + (
    "census_household_size_7_plus",
)
METRICS = (
    HOUSEHOLD_COUNT,
    HOUSEHOLD_POPULATION,
    PRIMARY,
    CHILD_U18,
    CHILD_U6,
    CHILD_COUNT,
    *SIZE_BINS,
    SPOUSE_PAIRS,
)
STATUSES = (
    "household_status",
    "primary_family_status",
    "own_children_status",
    "spouse_pairs_status",
)


def _require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_PRIMARY_FAMILY_" + reason)


def _code(token, low, high):
    if type(token) is not str or re.fullmatch(r"[0-9]{1,3}", token) is None:
        return None
    value = int(token)
    return value if low <= value <= high else None


def _empty(status):
    return {**dict.fromkeys(METRICS), **dict.fromkeys(STATUSES, status)}


def _universe(survey, household):
    if survey == "acs":
        kind = _code(household.get("TYPEHUGQ"), 1, 3)
        if kind is None:
            return "unavailable_housing_universe"
        return "occupied_housing_unit" if kind == 1 else "excluded_group_quarters"
    interview = _code(household.get("H_HHTYPE"), 1, 3)
    quarters = _code(household.get("H_LIVQRT"), 1, 12)
    kind = _code(household.get("HRHTYPE"), 0, 10)
    if interview is None or quarters is None or kind is None:
        return "unavailable_housing_universe"
    if interview != 1:
        return "excluded_noninterview"
    if kind == 0:
        return "unavailable_household_type_conflict"
    if (quarters <= 7) != (kind <= 8):
        return "unavailable_household_type_conflict"
    return "occupied_housing_unit" if quarters <= 7 else "excluded_nonhousing_unit"


def _asec_pairs(people):
    """Every nonzero pointer resolves reciprocally inside this exact household."""
    pointers, marital = {}, {}
    for line, row in people.items():
        pointer = _code(row.get("A_SPOUSE"), 0, 16)
        status = _code(row.get("A_MARITL"), 1, 7)
        if pointer is None:
            return None, "unavailable_A_SPOUSE"
        if status is None:
            return None, "unavailable_A_MARITL"
        pointers[line], marital[line] = pointer, status
    pairs = set()
    for line, pointer in pointers.items():
        if pointer == 0:
            if marital[line] in (1, 2):
                return None, "unavailable_spouse_pointer_conflict"
            continue
        if (
            pointer == line
            or pointer not in people
            or pointers[pointer] != line
            or marital[line] not in (1, 2)
            or marital[pointer] not in (1, 2)
        ):
            return None, "unavailable_spouse_pointer_conflict"
        pairs.add(tuple(sorted((line, pointer))))
    return pairs, "qualified"


def _primary(survey, people, head, relationships):
    spouse_codes = (3, 4) if survey == "asec" else (21, 23)
    spouses = [line for line, code in relationships.items() if code in spouse_codes]
    if len(spouses) > 1:
        return None, "unavailable_multiple_primary_spouses"
    if survey == "acs":
        if spouses and any(code in (22, 24) for code in relationships.values()):
            return None, "unavailable_primary_partner_conflict"
        if not spouses:
            return 0, "qualified"
        if any(
            _code(people[line].get("MAR"), 1, 5) != 1 for line in (head, spouses[0])
        ):
            return None, "unavailable_primary_marital_conflict"
        return 1, "qualified"
    head_pointer = _code(people[head].get("A_SPOUSE"), 0, 16)
    head_marital = _code(people[head].get("A_MARITL"), 1, 7)
    if head_pointer is None:
        return None, "unavailable_A_SPOUSE"
    if head_marital is None:
        return None, "unavailable_A_MARITL"
    if not spouses and head_pointer == 0 and head_marital not in (1, 2):
        return 0, "qualified"
    if (
        len(spouses) != 1
        or head_pointer != spouses[0]
        or _code(people[spouses[0]].get("A_SPOUSE"), 0, 16) != head
        or head_marital not in (1, 2)
        or _code(people[spouses[0]].get("A_MARITL"), 1, 7) not in (1, 2)
    ):
        return None, "unavailable_primary_spouse_conflict"
    return 1, "qualified"


def classify_primary_household(
    survey: str,
    household: Mapping[str, str],
    people: Sequence[Mapping[str, str]],
    *,
    qualified_head_line: int | None,
):
    """Classify one complete original household; never confer source authority.

    The source wrapper supplies the existing householder owner's result. Missing
    or contradictory evidence remains unavailable, separate from known zero and
    explicit universe exclusion. IDs stay literal strings; no floating coercion.
    """
    _require(survey in ("asec", "acs"), "SURVEY")
    _require(
        qualified_head_line is None or type(qualified_head_line) is int,
        "HEAD_LINE_TYPE",
    )
    universe = _universe(survey, household)
    if survey == "acs":
        kind = _code(household.get("TYPEHUGQ"), 1, 3)
        codes = [_code(row.get("RELSHIPP"), 20, 38) for row in people]
        if (kind == 1 and any(code in (37, 38) for code in codes)) or (
            kind in (2, 3)
            and any(code is not None and code != kind + 35 for code in codes)
        ):
            return _empty("unavailable_household_type_conflict")
    result = _empty(universe)
    if universe.startswith("excluded_"):
        result.update(dict.fromkeys(METRICS[:-1], 0))
        return result
    if universe != "occupied_housing_unit":
        return result
    count_field = "H_NUMPER" if survey == "asec" else "NP"
    count = _code(household.get(count_field), 0, 16 if survey == "asec" else 20)
    if count == 0:
        return _empty("unavailable_empty_occupied_household")
    result[HOUSEHOLD_COUNT] = 1
    size = len(people)
    if count is None or count != size:
        result.update(dict.fromkeys(STATUSES, "unavailable_complete_roster"))
        return result
    line_field, hh_field, person_hh_field = (
        ("A_LINENO", "H_SEQ", "PH_SEQ")
        if survey == "asec"
        else ("SPORDER", "SERIALNO", "SERIALNO")
    )
    by_line = {}
    identities = set()
    for row in people:
        line = _code(row.get(line_field), 1, 16 if survey == "asec" else 20)
        # PH_SEQ and H_SEQ may differ in leading zero formatting, not magnitude.
        raw_hh, person_hh = household.get(hh_field), row.get(person_hh_field)
        if survey == "asec":
            raw_hh, person_hh = _code_id(raw_hh), _code_id(person_hh)
        _require(raw_hh is not None and raw_hh == person_hh, "HOUSEHOLD_JOIN")
        _require(line is not None and line not in by_line, "PERSON_LINE_AXIS")
        identity = row.get("PERIDNUM") if survey == "asec" else (raw_hh, line)
        _require(identity is not None and identity not in identities, "PERSON_ID_AXIS")
        if survey == "asec":
            _require(_code_id(identity) is not None, "PERSON_ID_LITERAL")
            identity = _code_id(identity)
            _require(identity not in identities, "PERSON_ID_AXIS")
        identities.add(identity)
        by_line[line] = row
    result[HOUSEHOLD_POPULATION] = size
    result.update({name: int(i == min(size, 7)) for i, name in enumerate(SIZE_BINS, 1)})
    result["household_status"] = "qualified"
    if survey == "asec":
        pairs, status = _asec_pairs(by_line)
        result[SPOUSE_PAIRS] = None if pairs is None else len(pairs)
        result["spouse_pairs_status"] = status
    else:
        result["spouse_pairs_status"] = "unavailable_acs_secondary_spouse_pointer"
    role_field = "A_EXPRRP" if survey == "asec" else "RELSHIPP"
    relationships = {
        line: _code(
            row.get(role_field),
            1 if survey == "asec" else 20,
            14 if survey == "asec" else 38,
        )
        for line, row in by_line.items()
    }
    invalid = (
        None in relationships.values()
        or (survey == "asec" and 6 in relationships.values())
        or (
            survey == "acs" and any(code in (37, 38) for code in relationships.values())
        )
    )
    heads = [
        line
        for line, code in relationships.items()
        if code in ((1, 2) if survey == "asec" else (20,))
    ]
    if invalid or len(heads) != 1 or heads[0] != qualified_head_line:
        result.update(
            dict.fromkeys(
                ("primary_family_status", "own_children_status"),
                "unavailable_qualified_householder",
            )
        )
        return result
    if (
        survey == "asec"
        and relationships[heads[0]] == 2
        and any(code in (3, 4, 5, 7, 8, 9, 10) for code in relationships.values())
    ):
        result.update(
            dict.fromkeys(
                ("primary_family_status", "own_children_status"),
                "unavailable_householder_relationship_conflict",
            )
        )
        return result
    primary, status = _primary(survey, by_line, heads[0], relationships)
    # Household-type evidence is a cross-check, never the classifier authority.
    if survey == "asec" and primary is not None:
        if primary != int(_code(household.get("HRHTYPE"), 1, 8) in (1, 2)):
            primary, status = None, "unavailable_primary_household_type_conflict"
    result[PRIMARY], result["primary_family_status"] = primary, status
    if primary is None:
        result["own_children_status"] = status
        return result
    if primary == 0:
        result.update(
            {
                CHILD_U18: 0,
                CHILD_U6: 0,
                CHILD_COUNT: 0,
                "own_children_status": "not_married_primary_family",
            }
        )
        return result
    ages = []
    for line, role in relationships.items():
        if role not in ((5,) if survey == "asec" else (25, 26, 27)):
            continue
        row = by_line[line]
        age = _code(row.get("A_AGE" if survey == "asec" else "AGEP"), 0, 99)
        if age is None:
            result["own_children_status"] = "unavailable_own_child_age"
            return result
        if age >= 18:
            continue
        marital = _code(
            row.get("A_MARITL" if survey == "asec" else "MAR"),
            1,
            7 if survey == "asec" else 5,
        )
        if marital is None:
            result["own_children_status"] = "unavailable_own_child_marital_status"
            return result
        if marital == (7 if survey == "asec" else 5):
            ages.append(age)
    result.update(
        {
            CHILD_U18: int(bool(ages)),
            CHILD_U6: int(any(age < 6 for age in ages)),
            CHILD_COUNT: len(ages),
            "own_children_status": "qualified",
        }
    )
    return result


def _code_id(token):
    """Exact arbitrary-size integer identity, never an int64/float conversion."""
    if type(token) is not str or re.fullmatch(r"[0-9]+", token) is None:
        return None
    return int(token)
