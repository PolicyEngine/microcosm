"""The settled dependent-child spec drives shared benefit-unit construction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.concept_mapping import UnitRole
from microcosm.frame.unit_construction import (
    BenefitUnitRule,
    benefit_unit_diagnostics,
    benefit_unit_roles,
    build_benefit_units,
    units_per_household,
)
from test_support.microcosm_frame.concept_frames import concept_frames
from test_support.paths import paths_for

_RULE_PATH = (
    paths_for("microcosm-build").package
    / "src/microcosm/build/nz/benefit_unit_rule.json"
)
_POINTERS = ("partner_person_id", "parent_1_person_id", "parent_2_person_id")
_DIAGNOSTIC_KEYS = (
    "hours_independent_children",
    "independent_teen_parents",
    "younger_independent_teen_parents",
    "dependent_teen_parents",
    "non_partnered_coparent_households",
    "children_with_reference_person",
    "parentless_reference_person_heads",
)
_PROPERTY = settings(max_examples=100, deadline=None)


def _spec() -> dict:
    return json.loads(_RULE_PATH.read_text(encoding="utf-8"))


def _rule() -> BenefitUnitRule:
    return BenefitUnitRule.from_spec(_spec())


def _household(
    rows: list[dict], *, reference: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One small concept household, without donor or engine imports."""
    persons = []
    for row in rows:
        persons.append(
            {
                "person_household_id": 100,
                "usual_weekly_hours": 0,
                **dict.fromkeys(_POINTERS),
                **row,
            }
        )
    person = pd.DataFrame(persons)
    for column in _POINTERS:
        person[column] = pd.array(person[column], dtype="Int64")
    household = pd.DataFrame(
        {
            "household_id": [100],
            "reference_person_id": pd.array([reference], dtype="Int64"),
        }
    )
    return person, household


def _counts(**counts: int) -> dict[str, int]:
    return dict.fromkeys(_DIAGNOSTIC_KEYS, 0) | counts


def _assert_units(
    rows: list[dict],
    expected: list[int],
    diagnostics: dict[str, int],
    *,
    rule: BenefitUnitRule | None = None,
    reference: int = 1,
    roles: list[str] | None = None,
) -> None:
    person, household = _household(rows, reference=reference)
    rule = _rule() if rule is None else rule
    family, membership = build_benefit_units(person, household, rule)
    assert membership.tolist() == expected
    assert family[rule.id_column].tolist() == sorted(set(expected))
    if roles is not None:
        assert benefit_unit_roles(person, family, membership, rule).tolist() == roles
    assert benefit_unit_diagnostics(person, household, family, membership, rule) == (
        diagnostics
    )


@pytest.mark.parametrize(
    ("hours", "expected", "diagnostics"),
    [
        pytest.param(35, [1, 2], _counts(hours_independent_children=1), id="35-hours"),
        pytest.param(20, [1, 1], _counts(), id="20-hours"),
        pytest.param(30, [1, 2], _counts(hours_independent_children=1), id="30-hours"),
        pytest.param(29, [1, 1], _counts(), id="29-hours"),
    ],
)
def test_seventeen_year_old_hours_determine_independence(
    hours, expected, diagnostics
) -> None:
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {
                "person_id": 2,
                "age": 17,
                "parent_1_person_id": 1,
                "usual_weekly_hours": hours,
            },
        ],
        expected,
        diagnostics,
    )


def test_eighteen_year_old_is_an_adult_even_without_work() -> None:
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {"person_id": 2, "age": 18, "parent_1_person_id": 1},
        ],
        [1, 2],
        _counts(),
    )


def test_fifteen_year_old_remains_dependent_even_working_full_time() -> None:
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {
                "person_id": 2,
                "age": 15,
                "parent_1_person_id": 1,
                "usual_weekly_hours": 35,
            },
        ],
        [1, 1],
        _counts(),
    )


@pytest.mark.parametrize(
    ("age", "expected", "diagnostics"),
    [
        pytest.param(16, [1, 2, 2], _counts(independent_teen_parents=1), id="age16"),
        pytest.param(17, [1, 2, 2], _counts(independent_teen_parents=1), id="age17"),
        pytest.param(15, [1, 1, 1], _counts(dependent_teen_parents=1), id="age15"),
    ],
)
def test_teen_parent_and_baby_join_the_declared_unit(
    age, expected, diagnostics
) -> None:
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {"person_id": 2, "age": age, "parent_1_person_id": 1},
            {"person_id": 3, "age": 0, "parent_1_person_id": 2},
        ],
        expected,
        diagnostics,
    )


def test_non_partnered_coparents_place_children_with_parent_one() -> None:
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {"person_id": 2, "age": 42},
            {
                "person_id": 3,
                "age": 8,
                "parent_1_person_id": 1,
                "parent_2_person_id": 2,
            },
            {
                "person_id": 4,
                "age": 6,
                "parent_1_person_id": 1,
                "parent_2_person_id": 2,
            },
        ],
        [1, 2, 1, 1],
        _counts(non_partnered_coparent_households=1),
    )


def test_child_without_parent_joins_grandparent_reference_person() -> None:
    _assert_units(
        [{"person_id": 1, "age": 65}, {"person_id": 2, "age": 10}],
        [1, 1],
        _counts(children_with_reference_person=1),
    )


def test_parentless_child_joins_a_dependent_reference_persons_unit() -> None:
    # Amended B8: the 12-year-old reference person has a co-resident parent, so
    # they stay a dependent child in that parent's unit. The parentless
    # 8-year-old follows the reference person into the same unit.
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {"person_id": 2, "age": 12, "parent_1_person_id": 1},
            {"person_id": 3, "age": 8},
        ],
        [1, 1, 1],
        _counts(children_with_reference_person=1),
        reference=2,
        roles=["head", "dependent_child", "dependent_child"],
    )


@pytest.mark.parametrize(
    ("other_person", "diagnostics"),
    [
        pytest.param(
            {"person_id": 2, "age": 10},
            _counts(
                children_with_reference_person=1,
                parentless_reference_person_heads=1,
            ),
            id="parentless-child",
        ),
        pytest.param(
            {"person_id": 2, "age": 0, "parent_1_person_id": 1},
            _counts(
                younger_independent_teen_parents=1,
                parentless_reference_person_heads=1,
            ),
            id="reference-parent-under-16-with-baby",
        ),
    ],
)
def test_parentless_reference_person_heads_a_household_of_minors(
    other_person, diagnostics
) -> None:
    # Amended B8: a reference person is never their own caregiver, so a
    # 15-year-old reference person with no co-resident parent heads the unit.
    # A parent under 16 who heads a unit counts as a younger independent teen
    # parent.
    _assert_units(
        [{"person_id": 1, "age": 15}, other_person],
        [1, 1],
        diagnostics,
        roles=["head", "dependent_child"],
    )


@pytest.mark.parametrize(
    ("rows", "expected", "roles", "diagnostics"),
    [
        pytest.param(
            [
                {"person_id": 1, "age": 17},
                {"person_id": 2, "age": 45, "partner_person_id": 3},
                {"person_id": 3, "age": 43, "partner_person_id": 2},
            ],
            [1, 2, 2],
            ["head", "head", "partner"],
            _counts(parentless_reference_person_heads=1),
            id="parentless-17-year-old-with-adult-couple",
        ),
        pytest.param(
            [{"person_id": 1, "age": 16}, {"person_id": 2, "age": 52}],
            [1, 2],
            ["head", "head"],
            _counts(parentless_reference_person_heads=1),
            id="parentless-16-year-old-with-adult",
        ),
        pytest.param(
            [
                {"person_id": 1, "age": 15, "parent_1_person_id": 2},
                {"person_id": 2, "age": 44},
                {"person_id": 3, "age": 9},
            ],
            [2, 2, 2],
            ["dependent_child", "head", "dependent_child"],
            _counts(children_with_reference_person=1),
            id="15-year-old-with-parent-and-parentless-child",
        ),
        pytest.param(
            [
                {
                    "person_id": 1,
                    "age": 16,
                    "parent_1_person_id": 2,
                    "parent_2_person_id": 3,
                },
                {"person_id": 2, "age": 47, "partner_person_id": 3},
                {"person_id": 3, "age": 45, "partner_person_id": 2},
                {"person_id": 4, "age": 13},
            ],
            [2, 2, 2, 2],
            ["dependent_child", "head", "partner", "dependent_child"],
            _counts(children_with_reference_person=1),
            id="16-year-old-with-parents-and-parentless-child",
        ),
        pytest.param(
            [
                {"person_id": 1, "age": 17},
                {"person_id": 2, "age": 15},
                {"person_id": 3, "age": 0, "parent_1_person_id": 2},
            ],
            [1, 1, 1],
            ["head", "dependent_child", "dependent_child"],
            _counts(
                dependent_teen_parents=1,
                children_with_reference_person=1,
                parentless_reference_person_heads=1,
            ),
            id="minors-only",
        ),
    ],
)
def test_teen_householder_shapes_from_the_donor_build(
    rows, expected, roles, diagnostics
) -> None:
    """The household shapes the first real NZ transport run refused, and one more.

    Person 1 is the reference person (the Census householder) in each. The
    donor held parentless 16- and 17-year-old householders living with adults
    who are not their partners; each now heads a one-person unit. It also held
    15- and 16-year-old householders living with a parent and with a child who
    has no co-resident parent; that child now follows the householder into the
    householder's parent's unit. No refused donor household held only minors;
    that shape now builds too, headed by the parentless householder.
    """
    _assert_units(rows, expected, diagnostics, roles=roles)


def test_reference_parent_chain_without_an_adult_is_refused() -> None:
    # One refusal stays reachable with the flag off, from the shared caregiver
    # loop rather than B8's reference check. The 12-year-old reference person's
    # parent 1 is a parentless 15-year-old, whose caregiver is the reference
    # person, so neither reaches an adult. The 40-year-old is in neither chain.
    person, household = _household(
        [
            {"person_id": 1, "age": 12, "parent_1_person_id": 2},
            {"person_id": 2, "age": 15},
            {"person_id": 3, "age": 40},
        ]
    )
    with pytest.raises(ValueError, match="do not reach an adult"):
        build_benefit_units(person, household, _rule())


def _strict_reference_rule() -> BenefitUnitRule:
    """NZ's rule with the pre-amendment strict reference check turned back on."""
    spec = _spec()
    spec["child_without_coresident_parent"]["refuse_dependent_reference_person"] = True
    return BenefitUnitRule.from_spec(spec)


@pytest.mark.parametrize(
    ("rows", "reference"),
    [
        pytest.param(
            [
                {"person_id": 1, "age": 40},
                {"person_id": 2, "age": 12, "parent_1_person_id": 1},
                {"person_id": 3, "age": 8},
            ],
            2,
            id="parented-reference-with-parentless-child",
        ),
        pytest.param(
            [{"person_id": 1, "age": 15}, {"person_id": 2, "age": 10}],
            1,
            id="parentless-reference",
        ),
    ],
)
def test_strict_reference_flag_is_read_from_spec(rows, reference) -> None:
    # NZ's amended rule builds both households (see the tests above). The
    # shared strict branch, which no other test reaches, still refuses them
    # when a spec turns the flag back on.
    strict = _strict_reference_rule()
    assert strict.refuse_dependent_reference_person is True
    assert BenefitUnitRule.from_dict(strict.to_dict()) == strict
    person, household = _household(rows, reference=reference)
    with pytest.raises(ValueError, match="reference person.*dependent child"):
        build_benefit_units(person, household, strict)


@pytest.mark.parametrize(
    ("rows", "expected", "diagnostics"),
    [
        pytest.param(
            [
                {"person_id": 1, "age": 15},
                {"person_id": 2, "age": 0, "parent_1_person_id": 1},
                {"person_id": 3, "age": 50},
            ],
            [1, 1, 3],
            _counts(
                younger_independent_teen_parents=1,
                parentless_reference_person_heads=1,
            ),
            id="parentless-reference-parent-under-16",
        ),
        pytest.param(
            [
                {"person_id": 1, "age": 40},
                {"person_id": 2, "age": 15, "partner_person_id": 3},
                {"person_id": 3, "age": 16, "partner_person_id": 2},
                {"person_id": 4, "age": 0, "parent_1_person_id": 2},
            ],
            [1, 3, 3, 3],
            _counts(younger_independent_teen_parents=1),
            id="partnered-parent-under-16",
        ),
    ],
)
def test_parents_under_16_who_are_not_dependent_are_counted(
    rows, expected, diagnostics
) -> None:
    # Person 1 is the reference person. Counts are by hand: a parent under 16
    # who is not dependent, as a parentless reference person or a partner, is
    # counted below the parent-independence age, so the three teen-parent
    # counts partition parents under 18.
    _assert_units(rows, expected, diagnostics)


def test_country_spec_cannot_silently_use_an_inclusive_age_limit() -> None:
    spec = _spec()
    spec["dependent_child"]["age_limit_inclusive"] = True
    with pytest.raises(ValueError, match="exclusive"):
        BenefitUnitRule.from_spec(spec)


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        pytest.param(10, [1, 2], id="threshold10"),
        pytest.param(40, [1, 1], id="threshold40"),
    ],
)
def test_hours_threshold_is_read_from_spec(threshold, expected) -> None:
    spec = _spec()
    spec["dependent_child"]["financial_independence"]["min_usual_weekly_hours"] = (
        threshold
    )
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {
                "person_id": 2,
                "age": 17,
                "parent_1_person_id": 1,
                "usual_weekly_hours": 20,
            },
        ],
        expected,
        _counts(hours_independent_children=int(expected == [1, 2])),
        rule=BenefitUnitRule.from_spec(spec),
    )


@pytest.mark.parametrize(
    ("minimum_age", "expected"),
    [
        pytest.param(16, [1, 2], id="minimum16"),
        pytest.param(17, [1, 1], id="minimum17"),
    ],
)
def test_hours_minimum_age_is_read_from_spec(minimum_age, expected) -> None:
    spec = _spec()
    spec["dependent_child"]["financial_independence"]["min_age"] = minimum_age
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {
                "person_id": 2,
                "age": 16,
                "parent_1_person_id": 1,
                "usual_weekly_hours": 35,
            },
        ],
        expected,
        _counts(hours_independent_children=int(expected == [1, 2])),
        rule=BenefitUnitRule.from_spec(spec),
    )


@pytest.mark.parametrize(
    ("minimum_age", "expected"),
    [
        pytest.param(15, [1, 2, 2], id="minimum15"),
        pytest.param(17, [1, 1, 1], id="minimum17"),
    ],
)
def test_parent_independence_age_is_read_from_spec(minimum_age, expected) -> None:
    spec = _spec()
    spec["dependent_child"]["parent_independence_min_age"] = minimum_age
    _assert_units(
        [
            {"person_id": 1, "age": 40},
            {"person_id": 2, "age": 16, "parent_1_person_id": 1},
            {"person_id": 3, "age": 0, "parent_1_person_id": 2},
        ],
        expected,
        _counts(
            independent_teen_parents=int(expected == [1, 2, 2]),
            dependent_teen_parents=int(expected == [1, 1, 1]),
        ),
        rule=BenefitUnitRule.from_spec(spec),
    )


def test_declared_rule_round_trips_without_losing_settled_cases() -> None:
    rule = _rule()
    restored = BenefitUnitRule.from_dict(rule.to_dict())
    assert restored == rule
    assert restored.dependent_child.max_age == 17
    assert restored.dependent_child.parent_independence_min_age == 16
    assert restored.refuse_dependent_reference_person is False
    assert restored.dependent_child.financial_independence is not None
    assert restored.dependent_child.financial_independence.min_age == 16
    assert restored.dependent_child.financial_independence.min_usual_weekly_hours == 30


@st.composite
def _households_with_any_reference_age(draw) -> dict[str, pd.DataFrame]:
    tables = draw(concept_frames(max_households=5, max_members=7, min_id=-(10**6)))
    person = tables["person"]
    # Each reference person's age is drawn from 0 to 17 or from 18 to 130, so
    # dependent-age reference people, with and without a co-resident parent,
    # are drawn often. concept_frames draws ages independently of parent rank.
    for reference in tables["household"]["reference_person_id"]:
        person.loc[person["person_id"].eq(reference), "age"] = draw(
            st.integers(0, 17) | st.integers(18, 130)
        )
    return tables


def _households_without_an_adult_caregiver(
    person: pd.DataFrame, household: pd.DataFrame
) -> pd.Series:
    """Households whose reference person's parent chain never reaches an adult.

    This restates amended B8 apart from unit_construction. A candidate child is
    17 or under, has no co-resident partner, is not a parent aged 16 or over and
    does not work 30 hours a week at 16 or over. A candidate with a co-resident
    parent joins parent 1's unit; a parentless candidate who is not the
    reference person joins the reference person's unit. If a candidate
    reference person's parent 1 is a candidate, and so is each parent 1 above,
    the topmost has no co-resident parent and joins the reference person, so
    the chain never reaches an adult.
    """
    parents = set(person["parent_1_person_id"].dropna()) | set(
        person["parent_2_person_id"].dropna()
    )
    candidate = (
        person["age"].le(17)
        & person["partner_person_id"].isna()
        & ~(person["person_id"].isin(parents) & person["age"].ge(16))
        & ~(person["age"].ge(16) & person["usual_weekly_hours"].ge(30))
    )
    is_candidate = dict(zip(person["person_id"], candidate, strict=True))
    parent_1 = dict(zip(person["person_id"], person["parent_1_person_id"], strict=True))

    def without_adult(reference: int) -> bool:
        current = reference
        while is_candidate[current] and not pd.isna(parent_1[current]):
            current = parent_1[current]
        return current != reference and is_candidate[current]

    return pd.Series(
        [without_adult(item) for item in household["reference_person_id"]],
        index=household.index,
        dtype=bool,
    )


def _buildable_households(
    person: pd.DataFrame, household: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check the caregiver-loop refusal, then return the households that must build.

    One household whose reference chain reaches no adult refuses the whole
    frame; every other household must still build.
    """
    stuck = _households_without_an_adult_caregiver(person, household)
    if stuck.any():
        with pytest.raises(ValueError, match="do not reach an adult"):
            build_benefit_units(person, household, _rule())
    dropped = household.loc[stuck, "household_id"]
    person = person.loc[~person["person_household_id"].isin(dropped)]
    household = household.loc[~stuck]
    assume(not household.empty)
    return person.reset_index(drop=True), household.reset_index(drop=True)


@_PROPERTY
@given(
    tables=_households_with_any_reference_age(),
    weights=st.lists(
        st.floats(min_value=0.001, max_value=1e6, allow_nan=False),
        min_size=5,
        max_size=5,
    ),
)
def test_declared_nz_invariants_hold_for_random_households(tables, weights) -> None:
    person, household = _buildable_households(tables["person"], tables["household"])
    rule = _rule()
    family, membership = build_benefit_units(person, household, rule)
    roles = benefit_unit_roles(person, family, membership, rule)
    unit_of = pd.Series(membership.to_numpy(), index=person["person_id"])

    # Every person appears once, every constructed unit has members, and every
    # unit has a head adult. An independent teen parent is an adult in this
    # rule, and so is a reference person with no co-resident parent.
    assert membership.index.equals(person.index)
    assert membership.notna().all()
    assert membership.isin(family[rule.id_column]).all()
    assert set(membership) == set(family[rule.id_column])
    heads = roles.eq(UnitRole.HEAD.value)
    assert membership[heads].value_counts().eq(1).all()
    assert set(membership[heads]) == set(family[rule.id_column])

    partnered = person["partner_person_id"].notna()
    assert (
        membership[partnered].tolist()
        == unit_of.reindex(person.loc[partnered, "partner_person_id"]).tolist()
    )
    assert not roles[partnered].eq(UnitRole.DEPENDENT_CHILD.value).any()

    child = roles.eq(UnitRole.DEPENDENT_CHILD.value)
    assert person.loc[child, "age"].lt(18).all()
    independent_worker = person["age"].ge(16) & person["usual_weekly_hours"].ge(30)
    parents = set(person["parent_1_person_id"].dropna()) | set(
        person["parent_2_person_id"].dropna()
    )
    independent_parent = person["person_id"].isin(parents) & person["age"].ge(16)
    # Amended B8: a reference person is never their own caregiver, so one with
    # no co-resident parent heads a unit at any age.
    parentless_reference = (
        person["person_id"].isin(household["reference_person_id"])
        & person["parent_1_person_id"].isna()
    )
    excluded = independent_worker | independent_parent | parentless_reference
    assert not (child & excluded).any()
    adult = ~child
    assert (person["age"].ge(18) | partnered | excluded)[adult].all()
    with_parent = child & person["parent_1_person_id"].notna()
    assert (
        membership[with_parent].tolist()
        == unit_of.reindex(person.loc[with_parent, "parent_1_person_id"]).tolist()
    )
    without_parent = child & ~with_parent
    reference = household.set_index("household_id")["reference_person_id"]
    assert (
        membership[without_parent].tolist()
        == unit_of.reindex(
            reference.reindex(person.loc[without_parent, "person_household_id"])
        ).tolist()
    )

    # The teen-parent counts partition co-resident parents under 18, and the
    # reference heads are the parentless reference persons under 18 whom no
    # partner, hours or parent rule takes out of dependency.
    diagnostics = benefit_unit_diagnostics(person, household, family, membership, rule)
    young_parent = person["person_id"].isin(parents) & person["age"].lt(18)
    assert int(young_parent.sum()) == (
        diagnostics["independent_teen_parents"]
        + diagnostics["younger_independent_teen_parents"]
        + diagnostics["dependent_teen_parents"]
    )
    reference_teen = (
        person["person_id"].isin(household["reference_person_id"])
        & person["age"].lt(18)
        & ~partnered
        & ~independent_worker
        & ~independent_parent
        & person["parent_1_person_id"].isna()
    )
    assert heads[reference_teen].all()
    assert diagnostics["parentless_reference_person_heads"] == int(reference_teen.sum())

    unit_household = family.set_index(rule.id_column)[rule.household_column]
    assert (
        unit_household.reindex(membership).tolist()
        == person["person_household_id"].tolist()
    )

    household_weights = np.asarray(weights[: len(household)])
    order = np.argsort(household["household_id"].to_numpy())
    frame = Frame(
        schema=EntitySchema(group_entities=("household", rule.entity)),
        tables={
            "person": person.loc[:, ["person_id", "person_household_id"]].assign(
                **{rule.membership_column: membership.to_numpy()}
            ),
            "household": household.loc[:, ["household_id"]]
            .iloc[order]
            .reset_index(drop=True),
            rule.entity: family.loc[:, [rule.id_column]],
        },
        weights={
            "household": Weights(
                values=household_weights[order], kind=WeightKind.DESIGN
            )
        },
    )
    counts = units_per_household(household, family, rule).to_numpy()
    np.testing.assert_allclose(
        frame.resolve_weights(rule.entity).values.sum(),
        (household_weights * counts).sum(),
        rtol=1e-12,
    )


@_PROPERTY
@given(tables=_households_with_any_reference_age(), data=st.data())
def test_units_and_diagnostics_are_independent_of_person_row_order(
    tables, data
) -> None:
    person, household = _buildable_households(tables["person"], tables["household"])
    rule = _rule()
    family, membership = build_benefit_units(person, household, rule)
    order = data.draw(st.permutations(range(len(person))))
    shuffled = person.iloc[list(order)].reset_index(drop=True)
    shuffled_family, shuffled_membership = build_benefit_units(
        shuffled, household, rule
    )
    pd.testing.assert_frame_equal(family, shuffled_family)
    first = pd.Series(membership.to_numpy(), index=person["person_id"].to_numpy())
    second = pd.Series(
        shuffled_membership.to_numpy(), index=shuffled["person_id"].to_numpy()
    )
    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())
    assert benefit_unit_diagnostics(person, household, family, membership, rule) == (
        benefit_unit_diagnostics(
            shuffled, household, shuffled_family, shuffled_membership, rule
        )
    )
