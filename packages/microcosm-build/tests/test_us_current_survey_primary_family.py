"""Invented literal households; no population, model, weights, or source claims."""

import pytest

from microcosm.build.us_runtime import current_survey_primary_family as family


def person(survey, line, role, age=30, marital=None, spouse=0, household="7"):
    if survey == "asec":
        return dict(
            PERIDNUM=str(10**30 + line),
            PH_SEQ=household,
            A_LINENO=str(line),
            A_EXPRRP=str(role),
            A_AGE=str(age),
            A_MARITL=str(7 if marital is None else marital),
            A_SPOUSE=str(spouse),
        )
    return dict(
        SERIALNO=household,
        SPORDER=str(line),
        RELSHIPP=str(role),
        AGEP=str(age),
        MAR=str(5 if marital is None else marital),
    )


def classify(survey, people, *, head=1, married=True, household=None):
    raw = (
        dict(
            H_SEQ="00007",
            H_HHTYPE="1",
            H_LIVQRT="1",
            HRHTYPE="1" if married else "3",
            H_NUMPER=str(len(people)),
        )
        if survey == "asec"
        else dict(SERIALNO="7", TYPEHUGQ="1", NP=str(len(people)))
    )
    if household:
        raw.update(household)
    return family.classify_primary_household(
        survey, raw, people, qualified_head_line=head
    )


def couple(survey):
    return [
        person(survey, 1, 1 if survey == "asec" else 20, marital=1, spouse=2),
        person(survey, 2, 4 if survey == "asec" else 21, marital=1, spouse=1),
    ]


@pytest.mark.parametrize("survey", ["asec", "acs"])
@pytest.mark.parametrize(
    "age,count,u6",
    [(0, 1, 1), (5, 1, 1), (6, 1, 0), (17, 1, 0), (18, 0, 0), (30, 0, 0)],
)
def test_own_child_age_edges_and_one_primary_family(survey, age, count, u6):
    people = couple(survey) + [person(survey, 3, 5 if survey == "asec" else 25, age)]
    result = classify(survey, people)
    assert result[family.PRIMARY] == 1
    assert result[family.CHILD_COUNT] == result[family.CHILD_U18] == count
    assert result[family.CHILD_U6] == u6
    assert result[family.HOUSEHOLD_POPULATION] == 3
    assert (
        sum(result[name] for name in family.SIZE_BINS)
        == result[family.HOUSEHOLD_COUNT]
        == 1
    )


@pytest.mark.parametrize(
    "role,expected", [(25, 1), (26, 1), (27, 1), (30, 0), (32, 0), (35, 0), (36, 0)]
)
def test_acs_biological_adopted_step_and_unrelated_children(role, expected):
    result = classify("acs", couple("acs") + [person("acs", 3, role, 5)])
    assert result[family.CHILD_COUNT] == expected


@pytest.mark.parametrize(
    "survey,marital",
    [
        ("asec", 1),
        ("asec", 4),
        ("asec", 5),
        ("asec", 6),
        ("acs", 1),
        ("acs", 2),
        ("acs", 3),
        ("acs", 4),
    ],
)
def test_ever_married_child_is_not_an_own_child_under18(survey, marital):
    result = classify(
        survey,
        couple(survey)
        + [person(survey, 3, 5 if survey == "asec" else 25, 17, marital)],
    )
    assert result[family.CHILD_COUNT] == 0
    assert result["own_children_status"] == "qualified"


def test_unmarried_householder_with_married_subfamily_is_not_primary_couple():
    people = [
        person("asec", 1, 1, 70),
        person("asec", 2, 5, 35, 1, 3),
        person("asec", 3, 10, 35, 1, 2),
        person("asec", 4, 7, 4),
    ]
    result = classify("asec", people, married=False)
    assert result[family.PRIMARY] == result[family.CHILD_COUNT] == 0
    assert result[family.SPOUSE_PAIRS] == 1


def test_two_resident_couples_are_one_primary_family_and_two_pairs():
    people = couple("asec") + [
        person("asec", 3, 5, 30, 1, 4),
        person("asec", 4, 10, 30, 1, 3),
        person("asec", 5, 7, 4),
    ]
    result = classify("asec", people)
    assert result[family.PRIMARY] == 1 and result[family.SPOUSE_PAIRS] == 2
    assert result[family.CHILD_COUNT] == 0


def test_same_sex_acs_spouse_and_unavailable_secondary_pair_count():
    people = couple("acs")
    people[1]["RELSHIPP"] = "23"
    result = classify("acs", people)
    assert result[family.PRIMARY] == 1 and result[family.SPOUSE_PAIRS] is None
    assert result["spouse_pairs_status"] == "unavailable_acs_secondary_spouse_pointer"


@pytest.mark.parametrize("pointer", ["", "16", "1", "0", "2.0"])
def test_bad_primary_spouse_pointer_never_infers_marriage(pointer):
    people = couple("asec")
    people[0]["A_SPOUSE"] = pointer
    result = classify("asec", people)
    assert result[family.PRIMARY] is None and result[family.SPOUSE_PAIRS] is None


def test_bad_secondary_pointer_preserves_independently_qualified_primary():
    result = classify("asec", couple("asec") + [person("asec", 3, 10, 30, 1, 16)])
    assert result[family.PRIMARY] == 1 and result[family.SPOUSE_PAIRS] is None


@pytest.mark.parametrize("head", [None, 2, 12])
def test_actual_qualified_head_cannot_be_replaced_by_position_or_spouse(head):
    result = classify("acs", couple("acs"), head=head)
    assert result[family.PRIMARY] is None
    assert result["primary_family_status"] == "unavailable_qualified_householder"


@pytest.mark.parametrize(
    "survey,override,expected",
    [
        ("acs", {"TYPEHUGQ": "2"}, "excluded_group_quarters"),
        ("acs", {"TYPEHUGQ": "3"}, "excluded_group_quarters"),
        ("acs", {"TYPEHUGQ": ""}, "unavailable_housing_universe"),
        ("asec", {"HRHTYPE": "9", "H_LIVQRT": "11"}, "excluded_nonhousing_unit"),
        ("asec", {"HRHTYPE": "10", "H_LIVQRT": "12"}, "excluded_nonhousing_unit"),
        ("asec", {"HRHTYPE": "9"}, "unavailable_household_type_conflict"),
        ("asec", {"H_LIVQRT": "11"}, "unavailable_household_type_conflict"),
        ("asec", {"H_HHTYPE": ""}, "unavailable_housing_universe"),
        ("asec", {"H_HHTYPE": "2"}, "excluded_noninterview"),
    ],
)
def test_gq_noninterview_and_ambiguous_household_universe(survey, override, expected):
    people = couple(survey)
    if survey == "acs" and override.get("TYPEHUGQ") in ("2", "3"):
        people = [person("acs", 1, int(override["TYPEHUGQ"]) + 35)]
    result = classify(survey, people, household=override)
    assert result["household_status"] == expected
    assert result[family.HOUSEHOLD_COUNT] == (
        0 if expected.startswith("excluded") else None
    )
    assert result[family.PRIMARY] == result[family.HOUSEHOLD_COUNT]


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 6, 7, 12, 16])
def test_size_bins_partition_complete_housing_unit_households(size):
    people = [person("asec", i, 1 if i == 1 else 10) for i in range(1, size + 1)]
    result = classify("asec", people, married=False)
    assert result[family.HOUSEHOLD_POPULATION] == size
    assert [result[column] for column in family.SIZE_BINS] == [
        int(i == min(size, 7)) for i in range(1, 8)
    ]


@pytest.mark.parametrize("field,value", [("AGEP", ""), ("MAR", ""), ("MAR", "NA")])
def test_unknown_own_child_age_or_marital_evidence_is_not_zero(field, value):
    child = person("acs", 3, 26, 5)
    child[field] = value
    result = classify("acs", couple("acs") + [child])
    assert result[family.PRIMARY] == 1 and result[family.CHILD_COUNT] is None


def test_no_unrelated_child_age_or_tax_role_can_manufacture_own_child():
    child = person("acs", 3, 36, 4)
    child.update(is_tax_unit_dependent=True, PEPAR1="1", person_spm_unit_id="7")
    result = classify("acs", couple("acs") + [child])
    assert result[family.CHILD_COUNT] == 0


def test_arbitrary_size_ids_exact_line_join_and_order_invariance():
    people = couple("asec")
    raw_id = str(2**130 + 17)
    for row in people:
        row["PH_SEQ"] = raw_id
    first = classify("asec", people, household={"H_SEQ": raw_id})
    assert first == classify(
        "asec", list(reversed(people)), household={"H_SEQ": raw_id}
    )
    assert first[family.PRIMARY] == 1
    people[1]["PH_SEQ"] = str(int(raw_id) + 1)
    with pytest.raises(ValueError, match="HOUSEHOLD_JOIN"):
        classify("asec", people, household={"H_SEQ": raw_id})


def test_incomplete_roster_and_unnamed_relationship_preserve_unknown():
    result = classify("asec", couple("asec"), household={"H_NUMPER": "3"})
    assert (
        result[family.HOUSEHOLD_COUNT] == 1
        and result[family.HOUSEHOLD_POPULATION] is None
    )
    assert result[family.PRIMARY] is None
    people = couple("asec") + [person("asec", 3, 6, 5)]
    assert classify("asec", people)[family.PRIMARY] is None


def test_acs_relationship_universe_and_spouse_partner_conflicts():
    result = classify("acs", couple("acs"), household={"TYPEHUGQ": "2"})
    assert result[family.HOUSEHOLD_COUNT] is None
    result = classify("acs", [person("acs", 1, 37)])
    assert result[family.HOUSEHOLD_COUNT] is None
    result = classify("acs", couple("acs") + [person("acs", 3, 22)])
    assert result[family.PRIMARY] is None


def test_asec_reference_without_relatives_cannot_have_a_primary_spouse():
    people = couple("asec")
    people[0]["A_EXPRRP"] = "2"
    result = classify("asec", people)
    assert result[family.PRIMARY] is None
    assert result[family.SPOUSE_PAIRS] == 1
    assert (
        result["primary_family_status"]
        == "unavailable_householder_relationship_conflict"
    )
