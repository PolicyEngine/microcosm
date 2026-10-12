"""The settled dependent-child spec drives shared benefit-unit construction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
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
    "dependent_teen_parents",
    "non_partnered_coparent_households",
    "children_with_reference_person",
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
) -> None:
    person, household = _household(rows)
    rule = _rule() if rule is None else rule
    family, membership = build_benefit_units(person, household, rule)
    assert membership.tolist() == expected
    assert family[rule.id_column].tolist() == sorted(set(expected))
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


def test_dependent_reference_person_is_refused() -> None:
    person, household = _household(
        [
            {"person_id": 1, "age": 40},
            {"person_id": 2, "age": 12, "parent_1_person_id": 1},
            {"person_id": 3, "age": 8},
        ],
        reference=2,
    )
    with pytest.raises(ValueError, match="reference person.*dependent child"):
        build_benefit_units(person, household, _rule())


@pytest.mark.parametrize(
    "other_person",
    [
        pytest.param({"person_id": 2, "age": 10}, id="parentless-child"),
        pytest.param(
            {"person_id": 2, "age": 0, "parent_1_person_id": 1},
            id="dependent-reference-parent-with-baby",
        ),
    ],
)
def test_household_with_dependent_reference_and_no_adult_is_refused(
    other_person,
) -> None:
    person, household = _household([{"person_id": 1, "age": 15}, other_person])
    with pytest.raises(ValueError, match="reference person.*dependent child"):
        build_benefit_units(person, household, _rule())


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
    assert restored.refuse_dependent_reference_person is True
    assert restored.dependent_child.financial_independence is not None
    assert restored.dependent_child.financial_independence.min_age == 16
    assert restored.dependent_child.financial_independence.min_usual_weekly_hours == 30


@st.composite
def _households_with_adult_reference(draw) -> dict[str, pd.DataFrame]:
    tables = draw(concept_frames(max_households=5, max_members=7, min_id=-(10**6)))
    references = tables["household"]["reference_person_id"]
    # The refusal fixture checks dependent reference people. These generated
    # frames have an observable adult caregiver, so all must construct units.
    tables["person"].loc[tables["person"]["person_id"].isin(references), "age"] = draw(
        st.integers(18, 130)
    )
    return tables


@_PROPERTY
@given(
    tables=_households_with_adult_reference(),
    weights=st.lists(
        st.floats(min_value=0.001, max_value=1e6, allow_nan=False),
        min_size=5,
        max_size=5,
    ),
)
def test_declared_nz_invariants_hold_for_random_households(tables, weights) -> None:
    person, household = tables["person"], tables["household"]
    rule = _rule()
    family, membership = build_benefit_units(person, household, rule)
    roles = benefit_unit_roles(person, family, membership, rule)
    unit_of = pd.Series(membership.to_numpy(), index=person["person_id"])

    # Every person appears once, every constructed unit has members, and every
    # unit has a head adult. An independent teen parent is an adult in this rule.
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
    assert not (child & (independent_worker | independent_parent)).any()
    adult = ~child
    assert (person["age"].ge(18) | partnered | independent_worker | independent_parent)[
        adult
    ].all()
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
@given(tables=_households_with_adult_reference(), data=st.data())
def test_units_and_diagnostics_are_independent_of_person_row_order(
    tables, data
) -> None:
    person, household = tables["person"], tables["household"]
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
