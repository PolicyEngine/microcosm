"""Group encoding: concept and state bindings executed on built units.

``ConceptMapping.encode_groups`` executes the group rules a mapping declares,
on units a unit-construction step built from the pointers. Its invariants run
as properties over generated concept frames and unit rules:

- ``household_value`` is constant within a household;
- ``allocate_to_reference_unit`` puts each household amount on exactly the
  unit that holds the reference person, and conserves it; the per-adult
  alternative conserves it too;
- ``sum_over_members`` conserves member totals; ``any_member`` and
  ``reference_member`` read the members and the head;
- encoding is deterministic and row-aligned to the unit table;
- state bindings given as a list, iterator or generator encode, or are
  refused, exactly as the same bindings given as a tuple;
- membership ids of another integer dtype encode exactly as int64 ones, or
  are refused when they hold an id int64 cannot.

A row-by-row reference written from each rule's documented meaning checks the
vectorized executor (a differential test), and unit-composition flags are
checked against :func:`benefit_unit_attributes`, a second implementation of
the same composition.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame.adapters.axiom import axiom_concept_mapping
from microcosm.frame.concept_mapping import (
    Allocation,
    ConceptMapping,
    GroupKnobs,
    GroupMembership,
    GroupRule,
    Identity,
    InputBinding,
    InputDeclaration,
    Predicate,
    RelationshipRole,
    Role,
    Scale,
    ScaledSum,
    StateBinding,
    StateTest,
    TakeUpThreshold,
    UnitComposition,
    UnitFeature,
    UnitRole,
    transform_from_dict,
    transform_to_dict,
)
from microcosm.frame.concepts import CONCEPTS, AlignmentRelation, TemporalBasis
from microcosm.frame.unit_construction import (
    BenefitUnitRule,
    DependentChildRule,
    benefit_unit_attributes,
    benefit_unit_membership,
    build_benefit_units,
)
from test_support.microcosm_frame.concept_frames import concept_frames
from test_support.microcosm_frame.unit_rules import unit_rules

PROPERTY = settings(max_examples=120, deadline=None)
WEEK = 1 / 52
UNIT_CONCEPTS = UnitComposition(feature=UnitFeature.SOLE_PARENT).concepts


def _binding(engine_input: str, concepts, transform, rule, **change) -> InputBinding:
    return InputBinding(
        engine_input=engine_input,
        engine_entity=change.pop("engine_entity", "Family"),
        concepts=tuple(concepts),
        transform=transform,
        relation=AlignmentRelation.APPROXIMATE,
        note="evidence",
        group_rule=rule,
        **change,
    )


BINDINGS = (
    _binding(
        "earnings",
        ("fact:person.employment_income", "fact:person.interest_income"),
        ScaledSum(factor=WEEK),
        GroupRule.SUM_OVER_MEMBERS,
    ),
    _binding(
        "weeks",
        ("fact:person.weeks_worked",),
        Identity(),
        GroupRule.SUM_OVER_MEMBERS,
    ),
    _binding(
        "any_disability",
        ("fact:person.has_disability",),
        Identity(),
        GroupRule.ANY_MEMBER,
    ),
    _binding("head_age", ("fact:person.age",), Identity(), GroupRule.REFERENCE_MEMBER),
    _binding(
        "head_partnered",
        ("fact:person.partner_person_id",),
        RelationshipRole(role=Role.HAS_PARTNER),
        GroupRule.REFERENCE_MEMBER,
    ),
    _binding(
        "owner",
        ("fact:household.tenure",),
        Predicate(
            clauses=(
                ("fact:household.tenure", ("owned_outright", "owned_with_mortgage")),
            )
        ),
        GroupRule.HOUSEHOLD_VALUE,
    ),
    _binding(
        "rent",
        ("fact:household.rent",),
        Scale(factor=WEEK),
        GroupRule.ALLOCATE_TO_REFERENCE_UNIT,
    ),
    _binding(
        "mortgage",
        ("fact:household.mortgage_interest", "fact:household.mortgage_principal"),
        ScaledSum(factor=WEEK),
        GroupRule.ALLOCATE_TO_REFERENCE_UNIT,
    ),
    *(
        _binding(
            f"unit_{feature.value}",
            UNIT_CONCEPTS,
            UnitComposition(feature=feature),
            GroupRule.REFERENCE_MEMBER,
        )
        for feature in UnitFeature
    ),
    _binding(
        "child_age",
        ("fact:person.age",),
        Identity(),
        GroupRule.PERSON_ROLE,
        engine_entity="Child",
    ),
    _binding(
        "tax_unit_income",
        ("fact:person.employment_income",),
        Identity(),
        GroupRule.SUM_OVER_MEMBERS,
        engine_entity="TaxUnit",
    ),
)


def _mapping(bindings=BINDINGS) -> ConceptMapping:
    read = {concept_id for binding in bindings for concept_id in binding.reads}
    return ConceptMapping(
        engine="test",
        engine_version="0",
        entity_correspondence={"person": "Person", "household": "Household"},
        input_declaration=InputDeclaration.ENGINE_TYPED,
        bindings=tuple(bindings),
        unmapped={
            item.id: "not in this test" for item in CONCEPTS if item.id not in read
        },
    )


MAPPING = _mapping()
STATE = (
    StateBinding(
        engine_input="non_recipient",
        engine_entity="Family",
        columns=("person.receives_a", "person.receives_b"),
        test=StateTest.ANY_TRUE,
        group_rule=GroupRule.ANY_MEMBER,
        negate=True,
        note="evidence",
    ),
    StateBinding(
        engine_input="head_receives_a",
        engine_entity="Family",
        columns=("person.receives_a",),
        test=StateTest.ANY_TRUE,
        group_rule=GroupRule.REFERENCE_MEMBER,
        note="evidence",
    ),
    StateBinding(
        engine_input="in_area_1",
        engine_entity="Family",
        columns=("household.area",),
        test=StateTest.EQUALS,
        value=1,
        group_rule=GroupRule.HOUSEHOLD_VALUE,
        note="evidence",
    ),
)


@st.composite
def unit_frames(draw):
    """A concept frame with model state, its units, and its membership."""

    tables = draw(concept_frames(max_households=5, max_members=6, min_id=-(10**9)))
    rule = draw(unit_rules())
    person = tables["person"].copy()
    household = tables["household"].copy()
    for column in ("receives_a", "receives_b"):
        person[column] = np.asarray(
            draw(st.lists(st.booleans(), min_size=len(person), max_size=len(person))),
            dtype=bool,
        )
    household["area"] = np.asarray(
        draw(
            st.lists(
                st.integers(1, 4), min_size=len(household), max_size=len(household)
            )
        ),
        dtype=np.int64,
    )
    household["area_alt"] = np.asarray(
        draw(
            st.lists(
                st.integers(1, 4), min_size=len(household), max_size=len(household)
            )
        ),
        dtype=np.int64,
    )
    family, membership = build_benefit_units(person, household, rule)
    group = benefit_unit_membership(person, family, membership, rule)
    return {"person": person, "household": household}, rule, family, membership, group


def _encode(tables, group, **kwargs):
    return MAPPING.encode_groups(
        tables, {"Family": group}, state_bindings=STATE, **kwargs
    )


# --- A row-by-row reference, straight from each rule's documented meaning ----


def _reference(tables, rule, family, membership, *, allocation="reference_unit"):
    person = tables["person"].to_dict("records")
    households = {
        row["household_id"]: row for row in tables["household"].to_dict("records")
    }
    unit_of = dict(zip([row["person_id"] for row in person], membership, strict=True))
    # Roles from the unit table and the partner pointers alone, not from
    # benefit_unit_roles: the head is the unit's head, the head's partner in
    # the same unit is the partner, and every other member is a child.
    head_of_unit = dict(
        zip(family[rule.id_column], family[rule.head_column], strict=True)
    )
    partner_of = {
        row["person_id"]: None
        if pd.isna(row["partner_person_id"])
        else row["partner_person_id"]
        for row in person
    }
    roles = {}
    for row in person:
        pid = row["person_id"]
        head = head_of_unit[unit_of[pid]]
        if pid == head:
            roles[pid] = UnitRole.HEAD.value
        elif partner_of[head] == pid:
            roles[pid] = UnitRole.PARTNER.value
        else:
            roles[pid] = UnitRole.DEPENDENT_CHILD.value
    out = {name: [] for name in ("earnings", "weeks", "any_disability", "head_age")}
    out |= {"head_partnered": [], "owner": [], "rent": [], "mortgage": []}
    out |= {"non_recipient": [], "head_receives_a": [], "in_area_1": []}
    out |= {f"unit_{feature.value}": [] for feature in UnitFeature}
    for unit in family.to_dict("records"):
        members = [
            row for row in person if unit_of[row["person_id"]] == unit[rule.id_column]
        ]
        head = next(
            row for row in members if row["person_id"] == unit[rule.head_column]
        )
        home = households[unit[rule.household_column]]
        out["earnings"].append(
            sum(
                (row["employment_income"] + row["interest_income"]) * WEEK
                for row in members
            )
        )
        out["weeks"].append(sum(row["weeks_worked"] for row in members))
        out["any_disability"].append(any(row["has_disability"] for row in members))
        out["head_age"].append(head["age"])
        out["head_partnered"].append(not pd.isna(head["partner_person_id"]))
        out["owner"].append(home["tenure"] in ("owned_outright", "owned_with_mortgage"))
        holds_reference = any(
            row["person_id"] == home["reference_person_id"] for row in members
        )
        mortgage = (home["mortgage_interest"] + home["mortgage_principal"]) * WEEK
        out["mortgage"].append(mortgage if holds_reference else 0.0)
        if allocation == "reference_unit":
            out["rent"].append(home["rent"] * WEEK if holds_reference else 0.0)
        else:
            adults = {
                key: sum(
                    1
                    for row in person
                    if unit_of[row["person_id"]] == key
                    and roles[row["person_id"]] != UnitRole.DEPENDENT_CHILD.value
                )
                for key in set(unit_of.values())
            }
            in_home = [
                key
                for key, household_id in zip(
                    family[rule.id_column], family[rule.household_column], strict=True
                )
                if household_id == unit[rule.household_column]
            ]
            share = adults[unit[rule.id_column]] / sum(adults[key] for key in in_home)
            out["rent"].append(home["rent"] * WEEK * share)
        out["non_recipient"].append(
            not any(row["receives_a"] or row["receives_b"] for row in members)
        )
        out["head_receives_a"].append(bool(head["receives_a"]))
        out["in_area_1"].append(home["area"] == 1)
        partners = sum(
            roles[row["person_id"]] == UnitRole.PARTNER.value for row in members
        )
        children = sum(
            roles[row["person_id"]] == UnitRole.DEPENDENT_CHILD.value for row in members
        )
        out["unit_has_partner"].append(partners > 0)
        out["unit_has_dependent_child"].append(children >= 1)
        out["unit_two_or_more_dependent_children"].append(children >= 2)
        out["unit_sole_parent"].append(partners == 0 and children >= 1)
    return out


class TestExecution:
    @PROPERTY
    @given(case=unit_frames(), mode=st.sampled_from(list(Allocation)))
    def test_encode_groups_matches_a_row_by_row_reference(self, case, mode) -> None:
        tables, rule, family, membership, group = case
        encoded = _encode(tables, group, knobs=GroupKnobs(allocation={"rent": mode}))
        table = encoded.tables[rule.entity]
        expected = _reference(tables, rule, family, membership, allocation=mode.value)
        assert table[rule.id_column].tolist() == family[rule.id_column].tolist()
        for name, values in expected.items():
            actual = table[name].to_numpy()
            if isinstance(values[0], bool):
                assert actual.dtype == bool, name
                assert actual.tolist() == values, name
            else:
                np.testing.assert_allclose(
                    actual.astype(np.float64),
                    values,
                    rtol=1e-12,
                    atol=1e-9,
                    err_msg=name,
                )

    @PROPERTY
    @given(case=unit_frames())
    def test_household_values_are_constant_within_a_household(self, case) -> None:
        tables, rule, family, _, group = case
        table = _encode(tables, group).tables[rule.entity]
        households = family[rule.household_column].to_numpy()
        for name in ("owner", "in_area_1"):
            per_household = pd.Series(table[name].to_numpy()).groupby(households)
            assert (per_household.nunique() == 1).all(), name

    @PROPERTY
    @given(case=unit_frames(), mode=st.sampled_from(list(Allocation)))
    def test_allocation_conserves_each_household_amount(self, case, mode) -> None:
        tables, rule, family, membership, group = case
        table = _encode(
            tables, group, knobs=GroupKnobs(allocation={"rent": mode})
        ).tables[rule.entity]
        households = family[rule.household_column].to_numpy()
        household = tables["household"].set_index("household_id")
        for name, amount in (
            ("rent", household["rent"] * WEEK),
            (
                "mortgage",
                (household["mortgage_interest"] + household["mortgage_principal"])
                * WEEK,
            ),
        ):
            totals = pd.Series(table[name].to_numpy()).groupby(households).sum()
            np.testing.assert_allclose(
                totals.to_numpy(), amount.reindex(totals.index).to_numpy(), rtol=1e-12
            )
        # reference_unit puts the amount on exactly the reference person's unit
        reference_unit = pd.Series(
            membership.to_numpy(), index=tables["person"]["person_id"].to_numpy()
        ).reindex(household["reference_person_id"].astype("int64").to_numpy())
        holds = family[rule.id_column].isin(reference_unit.to_numpy()).to_numpy()
        assert (pd.Series(holds).groupby(households).sum() == 1).all()
        assert (table.loc[~holds, "mortgage"] == 0).all()
        if mode is Allocation.REFERENCE_UNIT:
            assert (table.loc[~holds, "rent"] == 0).all()

    @PROPERTY
    @given(case=unit_frames())
    def test_sums_conserve_member_totals(self, case) -> None:
        tables, rule, _, _, group = case
        table = _encode(tables, group).tables[rule.entity]
        person = tables["person"]
        np.testing.assert_allclose(
            table["earnings"].sum(),
            ((person["employment_income"] + person["interest_income"]) * WEEK).sum(),
            rtol=1e-9,
            atol=1e-6,
        )
        assert table["weeks"].dtype == np.int64
        assert table["weeks"].sum() == person["weeks_worked"].sum()

    @PROPERTY
    @given(case=unit_frames())
    def test_unit_composition_agrees_with_unit_attributes(self, case) -> None:
        tables, rule, family, membership, group = case
        table = _encode(tables, group).tables[rule.entity]
        attributes = benefit_unit_attributes(tables["person"], family, membership, rule)
        children = attributes["n_dependent_children"].to_numpy()
        assert table["unit_has_partner"].tolist() == attributes["is_couple"].tolist()
        assert table["unit_has_dependent_child"].tolist() == (children >= 1).tolist()
        assert (
            table["unit_two_or_more_dependent_children"].tolist()
            == (children >= 2).tolist()
        )
        assert (
            table["unit_sole_parent"].tolist() == attributes["is_sole_parent"].tolist()
        )
        # the head's partner pointer and the unit's partner role agree
        assert table["head_partnered"].tolist() == table["unit_has_partner"].tolist()

    @PROPERTY
    @given(case=unit_frames(), factor=st.floats(0.0, 3.0, allow_nan=False))
    def test_knobs_zero_scale_and_redirect(self, case, factor) -> None:
        tables, rule, _, _, group = case
        base = _encode(tables, group).tables[rule.entity]
        knobbed = _encode(
            tables,
            group,
            knobs=GroupKnobs(
                zeroed=frozenset({"earnings"}),
                factors={"rent": factor},
                state_columns={"household.area": "household.area_alt"},
            ),
        ).tables[rule.entity]
        assert (knobbed["earnings"] == 0).all()
        np.testing.assert_allclose(knobbed["rent"], base["rent"] * factor, rtol=1e-12)
        alternative = tables["household"].set_index("household_id")["area_alt"]
        expected = (
            alternative.reindex(group.units[group.household_column]).to_numpy() == 1
        )
        assert knobbed["in_area_1"].tolist() == expected.tolist()
        pd.testing.assert_frame_equal(
            knobbed.drop(columns=["earnings", "rent", "in_area_1"]),
            base.drop(columns=["earnings", "rent", "in_area_1"]),
        )

    @PROPERTY
    @given(
        case=unit_frames(),
        picks=st.lists(st.integers(0, len(STATE) - 1), max_size=len(STATE) + 1),
        knobs=st.sampled_from(
            [
                GroupKnobs(),
                GroupKnobs(state_columns={"household.area": "household.area_alt"}),
            ]
        ),
        kind=st.sampled_from(["list", "iterator", "generator"]),
    )
    def test_any_iterable_of_state_bindings_encodes_as_its_tuple(
        self, case, picks, knobs, kind
    ) -> None:
        tables, rule, _, _, group = case
        states = [STATE[pick] for pick in picks]
        given_as = {
            "list": list,
            "iterator": iter,
            "generator": lambda items: (item for item in items),
        }[kind]

        def outcome(bindings):
            try:
                encoded = MAPPING.encode_groups(
                    tables, {"Family": group}, state_bindings=bindings, knobs=knobs
                )
            except ValueError as error:
                return str(error)
            return encoded

        expected = outcome(tuple(states))
        got = outcome(given_as(states))
        if isinstance(expected, str):
            # a refusal (a binding given twice, or a knob no binding reads)
            # is the same refusal
            assert got == expected
            return
        assert not isinstance(got, str), got
        pd.testing.assert_frame_equal(
            got.tables[rule.entity], expected.tables[rule.entity]
        )
        assert got.deferred == expected.deferred

    @PROPERTY
    @given(case=unit_frames())
    def test_encoding_is_deterministic(self, case) -> None:
        tables, rule, _, _, group = case
        first = _encode(tables, group)
        second = _encode(tables, group)
        pd.testing.assert_frame_equal(
            first.tables[rule.entity], second.tables[rule.entity]
        )

    def test_a_one_shot_iterator_of_state_bindings_takes_a_column_knob(self) -> None:
        # The bindings are read twice: to encode, then to find the columns a
        # knob may replace. A spent iterator used to leave the second read
        # empty, so the area knob was refused as replacing a column no
        # binding reads.
        tables, rule, _, _, group = _example()
        tables = {**tables, "household": tables["household"].assign(area_alt=2)}
        knobs = GroupKnobs(state_columns={"household.area": "household.area_alt"})
        expected = MAPPING.encode_groups(
            tables, {"Family": group}, state_bindings=STATE, knobs=knobs
        )
        assert expected.tables[rule.entity]["in_area_1"].tolist() == [False, False]
        for states in (list(STATE), iter(STATE), (state for state in STATE)):
            got = MAPPING.encode_groups(
                tables, {"Family": group}, state_bindings=states, knobs=knobs
            )
            pd.testing.assert_frame_equal(
                got.tables[rule.entity], expected.tables[rule.entity]
            )
            assert got.deferred == expected.deferred

    def test_unexecutable_group_bindings_are_deferred(self) -> None:
        tables, _, _, _, group = _example()
        encoded = _encode(tables, group)
        assert {binding.engine_input for binding in encoded.deferred} == {
            "child_age",
            "tax_unit_income",
        }

    def test_modules_filter_bindings(self) -> None:
        bindings = [
            _binding(
                "a",
                ("fact:person.age",),
                Identity(),
                GroupRule.REFERENCE_MEMBER,
                module="m/a.yaml",
                canonical_input="zz:m/a#input.a",
            ),
            _binding(
                "b",
                ("fact:person.age",),
                Identity(),
                GroupRule.REFERENCE_MEMBER,
                module="m/b.yaml",
                canonical_input="zz:m/b#input.b",
            ),
        ]
        read = {"fact:person.age"}
        mapping = ConceptMapping(
            engine="axiom:zz",
            engine_version="0",
            entity_correspondence={"person": "Person", "household": "Household"},
            input_declaration=InputDeclaration.USAGE_INFERRED,
            bindings=tuple(bindings),
            unmapped={item.id: "x" for item in CONCEPTS if item.id not in read},
        )
        tables, rule, _, _, group = _example()
        encoded = mapping.encode_groups(tables, {"Family": group}, modules=["m/b.yaml"])
        assert list(encoded.tables[rule.entity].columns) == [rule.id_column, "b"]


def _example():
    rule = BenefitUnitRule(
        entity="family",
        dependent_child=DependentChildRule(max_age=17, financial_independence=None),
        unparented_child="reference_person_unit",
        split_parents="first_parent_unit",
        provenance={"max_age": "test"},
    )
    rows = [
        (10, 1, 40, 11, None, 50_000.0, True),
        (11, 1, 38, 10, None, 0.0, False),
        (12, 1, 10, None, 10, 0.0, False),
        (13, 1, 30, None, None, 26_000.0, False),
    ]
    person = pd.DataFrame(
        rows,
        columns=[
            "person_id",
            "person_household_id",
            "age",
            "partner_person_id",
            "parent_1_person_id",
            "employment_income",
            "receives_a",
        ],
    )
    person["parent_2_person_id"] = pd.array([None] * 4, dtype="Int64")
    for column in ("partner_person_id", "parent_1_person_id"):
        person[column] = pd.array(person[column].tolist(), dtype="Int64")
    person["age"] = person["age"].astype(np.int64)
    person["receives_b"] = False
    person["interest_income"] = 0.0
    person["weeks_worked"] = np.int64(0)
    person["has_disability"] = False
    household = pd.DataFrame(
        {
            "household_id": [1],
            "reference_person_id": pd.array([10], dtype="Int64"),
            "tenure": ["rented_private"],
            "rent": [52_000.0],
            "mortgage_interest": [0.0],
            "mortgage_principal": [0.0],
            "area": np.asarray([1], dtype=np.int64),
        }
    )
    family, membership = build_benefit_units(person, household, rule)
    group = benefit_unit_membership(person, family, membership, rule)
    return {"person": person, "household": household}, rule, family, membership, group


class TestRefusals:
    def test_a_membership_must_match_the_frame(self) -> None:
        tables, rule, family, membership, group = _example()
        cases = [
            (
                GroupMembership(
                    entity="family",
                    units=family,
                    person_unit=membership.iloc[:-1],
                    person_role=group.person_role.iloc[:-1],
                ),
                "align",
            ),
            (
                GroupMembership(
                    entity="family",
                    units=family,
                    person_unit=membership.replace({13: 99}),
                    person_role=group.person_role,
                ),
                "unknown family",
            ),
            (
                GroupMembership(
                    entity="family",
                    units=family,
                    person_unit=membership,
                    person_role=group.person_role.replace({"head": "partner"}),
                ),
                "exactly one head",
            ),
            (
                GroupMembership(
                    entity="family",
                    units=family.assign(family_household_id=7),
                    person_unit=membership,
                    person_role=group.person_role,
                ),
                "nest",
            ),
            (
                GroupMembership(
                    entity="family",
                    units=family.drop(columns="family_head_person_id"),
                    person_unit=membership,
                    person_role=group.person_role,
                ),
                "lacks",
            ),
        ]
        for bad, message in cases:
            with pytest.raises(ValueError, match=message):
                MAPPING.encode_groups(tables, {"Family": bad})

    def test_state_never_feeds_an_input_twice(self) -> None:
        tables, _, _, _, group = _example()
        twice = StateBinding(
            engine_input="owner",
            engine_entity="Family",
            columns=("person.receives_a",),
            test=StateTest.ANY_TRUE,
            group_rule=GroupRule.ANY_MEMBER,
            note="evidence",
        )
        with pytest.raises(ValueError, match="concept and by state"):
            MAPPING.encode_groups(tables, {"Family": group}, state_bindings=[twice])
        with pytest.raises(ValueError, match="twice by state"):
            MAPPING.encode_groups(
                tables, {"Family": group}, state_bindings=[STATE[0], STATE[0]]
            )

    @pytest.mark.parametrize(
        ("knobs", "message"),
        [
            (GroupKnobs(zeroed=frozenset({"missing"})), "did not produce"),
            (GroupKnobs(zeroed=frozenset({"owner"})), "not numeric"),
            (GroupKnobs(allocation={"earnings": "per_adult_share"}), "not allocated"),
            (
                GroupKnobs(state_columns={"household.nowhere": "household.area"}),
                "no binding reads",
            ),
        ],
    )
    def test_knobs_fail_closed(self, knobs, message) -> None:
        tables, _, _, _, group = _example()
        with pytest.raises(ValueError, match=message):
            _encode(tables, group, knobs=knobs)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"zeroed": {"a"}, "factors": {"a": 2.0}}, "both zeroed and scaled"),
            ({"factors": {"a": -1.0}}, "non-negative"),
            ({"factors": {"a": True}}, "number"),
            ({"allocation": {"a": "evenly"}}, "evenly"),
            ({"state_columns": {"household.a": "person.a"}}, "another entity"),
            ({"state_columns": {"a": "household.a"}}, "state column"),
        ],
    )
    def test_malformed_knobs_are_refused(self, change, message) -> None:
        with pytest.raises(ValueError, match=message):
            GroupKnobs(**change)

    def test_state_columns_are_checked(self) -> None:
        tables, _, _, _, group = _example()
        person = tables["person"].assign(receives_a=1)
        with pytest.raises(ValueError, match="boolean"):
            _encode({**tables, "person": person}, group)
        household = tables["household"].assign(area=["1"])
        with pytest.raises(ValueError, match="cannot equal"):
            _encode({**tables, "household": household}, group)
        with pytest.raises(ValueError, match="column 'area'"):
            _encode(
                {**tables, "household": tables["household"].drop(columns="area")}, group
            )
        person = tables["person"].assign(receives_b=[True, None, False, False])
        with pytest.raises(ValueError, match="missing values"):
            _encode({**tables, "person": person}, group)

    def test_one_input_name_is_one_computation(self) -> None:
        tables, _, _, _, group = _example()
        other_module = StateBinding(
            engine_input="owner",
            engine_entity="Family",
            columns=("person.receives_a",),
            test=StateTest.ANY_TRUE,
            group_rule=GroupRule.ANY_MEMBER,
            note="evidence",
            module="other.yaml",
        )
        with pytest.raises(ValueError, match="concept and by state"):
            MAPPING.encode_groups(
                tables, {"Family": group}, state_bindings=[other_module]
            )
        first = StateBinding.from_dict({**STATE[0].to_dict(), "module": "a.yaml"})
        same = StateBinding.from_dict({**STATE[0].to_dict(), "module": "b.yaml"})
        different = StateBinding.from_dict(
            {**STATE[0].to_dict(), "module": "b.yaml", "negate": False}
        )
        encoded = MAPPING.encode_groups(
            tables, {"Family": group}, state_bindings=[first, same]
        )
        assert encoded.tables["family"]["non_recipient"].tolist() == [False, True]
        with pytest.raises(ValueError, match="computed differently"):
            MAPPING.encode_groups(
                tables, {"Family": group}, state_bindings=[first, different]
            )

    def test_an_allocation_knob_on_state_is_refused(self) -> None:
        tables, _, _, _, group = _example()
        with pytest.raises(ValueError, match="not allocated"):
            _encode(
                tables,
                group,
                knobs=GroupKnobs(allocation={"in_area_1": "per_adult_share"}),
            )

    def test_an_unseeded_take_up_program_is_left_out(self) -> None:
        tables, rule, _, _, group = _example()
        person = tables["person"].assign(take_up_seed=0.5)
        tables = {**tables, "person": person}
        binding = _binding(
            "takes_up",
            ("fact:person.take_up_seed",),
            TakeUpThreshold(program="zz.x"),
            GroupRule.ANY_MEMBER,
        )
        mapping = _mapping([binding])
        encoded = mapping.encode_groups(
            tables, {"Family": group}, take_up_rates={"zz.x": None}
        )
        assert list(encoded.tables[rule.entity].columns) == [rule.id_column]
        everyone = mapping.encode_groups(
            tables, {"Family": group}, take_up_rates={"zz.x": 1.0}
        )
        assert everyone.tables[rule.entity]["takes_up"].all()

    def test_memberships_align_by_index_and_ids(self) -> None:
        tables, _, family, membership, group = _example()
        shuffled = tables["person"].iloc[[0, 3, 2, 1]]
        with pytest.raises(ValueError, match="align to the person table's index"):
            MAPPING.encode_groups({**tables, "person": shuffled}, {"Family": group})
        realigned = GroupMembership(
            entity="family",
            units=family,
            person_unit=membership.reindex(shuffled.index),
            person_role=group.person_role.reindex(shuffled.index),
        )
        first = MAPPING.encode_groups(tables, {"Family": group}).tables["family"]
        second = MAPPING.encode_groups(
            {**tables, "person": shuffled}, {"Family": realigned}
        ).tables["family"]
        pd.testing.assert_frame_equal(first, second)
        fractional = GroupMembership(
            entity="family",
            units=family,
            person_unit=membership.astype(np.float64) + 0.5,
            person_role=group.person_role,
        )
        with pytest.raises(ValueError, match="must be integers"):
            MAPPING.encode_groups(tables, {"Family": fractional})

    def test_a_membership_naming_an_unsigned_id_int64_cannot_hold_is_refused(
        self,
    ) -> None:
        # Person 13 becomes -1, so the units are -1 and 10. As uint64, -1 is
        # 2**64 - 1: a unit that does not exist, which an int64 comparison
        # used to read back as unit -1 and accept.
        tables, rule, _, _, _ = _example()
        person = tables["person"].assign(
            person_id=tables["person"]["person_id"].replace({13: -1})
        )
        tables = {**tables, "person": person}
        family, membership = build_benefit_units(person, tables["household"], rule)
        group = benefit_unit_membership(person, family, membership, rule)
        assert membership.tolist() == [10, 10, 10, -1]
        unsigned = GroupMembership(
            entity="family",
            units=family,
            person_unit=membership.astype(np.uint64),
            person_role=group.person_role,
        )
        with pytest.raises(ValueError, match=re.escape("['person unit ids']")):
            MAPPING.encode_groups(tables, {"Family": unsigned})
        # Likewise a frame whose person -1 is renamed 2**64 - 1: the unit's
        # head column, -1, no longer names its head.
        renamed = person.assign(person_id=person["person_id"].astype(np.uint64))
        with pytest.raises(ValueError, match=re.escape("['person.person_id']")):
            MAPPING.encode_groups({**tables, "person": renamed}, {"Family": group})

    def test_a_partner_pointer_int64_cannot_hold_is_refused(self) -> None:
        # Head 10 becomes -1 and partner 11's pointer becomes 2**64 - 1, which
        # names no person but an int64 read turns into -1: person 11 would pass
        # as head -1's partner. An int64 pointer naming no person is refused as
        # before, by the partner rule itself.
        tables, _, family, _, group = _example()
        person = tables["person"].copy()
        for column in ("person_id", "parent_1_person_id", "parent_2_person_id"):
            person[column] = person[column].replace({10: -1})
        household = tables["household"].assign(
            reference_person_id=tables["household"]["reference_person_id"].replace(
                {10: -1}
            )
        )
        units = family.assign(family_id=[-1, 13], family_head_person_id=[-1, 13])
        membership = GroupMembership(
            entity="family",
            units=units,
            person_unit=pd.Series([-1, -1, -1, 13], index=person.index, dtype=np.int64),
            person_role=group.person_role,
        )
        wrapped = person.assign(
            partner_person_id=pd.array([11, 2**64 - 1, None, None], dtype="UInt64")
        )
        with pytest.raises(ValueError, match=re.escape("['person.partner_person_id']")):
            MAPPING.encode_groups(
                {**tables, "person": wrapped, "household": household},
                {"Family": membership},
            )
        dangling = person.assign(
            partner_person_id=pd.array([11, 999, None, None], dtype="Int64")
        )
        with pytest.raises(ValueError, match="must be its head's partner"):
            MAPPING.encode_groups(
                {**tables, "person": dangling, "household": household},
                {"Family": membership},
            )

    @pytest.mark.parametrize("dtype", ["uint8", "UInt8", "int64"])
    def test_a_narrow_unsigned_household_cannot_catch_a_unit_past_its_range(
        self, dtype
    ) -> None:
        # The family nests in household 261, and only household 5 exists.
        # pandas matched unit households against a narrower unsigned index by
        # casting them down to it, so under uint8 household 261 was household
        # 5 (and UInt8 raised a TypeError); as int64 it is unknown.
        tables, _, family, _, group = _example()
        person = tables["person"].assign(person_household_id=np.int64(261))
        household = tables["household"].assign(household_id=pd.array([5], dtype=dtype))
        membership = GroupMembership(
            entity="family",
            units=family.assign(family_household_id=np.int64(261)),
            person_unit=group.person_unit,
            person_role=group.person_role,
        )
        with pytest.raises(ValueError, match="names an unknown household"):
            MAPPING.encode_groups(
                {**tables, "person": person, "household": household},
                {"Family": membership},
            )

    @PROPERTY
    @given(case=unit_frames(), dtype=st.sampled_from(["uint64", "UInt64", "Int64"]))
    def test_membership_ids_are_read_exactly_or_refused(self, case, dtype) -> None:
        tables, rule, family, membership, group = case
        columns = (rule.id_column, rule.household_column, rule.head_column)

        def retyped(values: pd.Series) -> pd.Series:
            # NumPy's cast turns a negative id into one above int64's range.
            unsigned = dtype.lower().startswith("u")
            raw = values.to_numpy().astype(np.uint64 if unsigned else np.int64)
            return pd.Series(raw, index=values.index, name=values.name).astype(dtype)

        units = family.assign(**{column: retyped(family[column]) for column in columns})
        retyped_group = GroupMembership(
            entity=group.entity,
            units=units,
            person_unit=retyped(membership),
            person_role=group.person_role,
        )
        ids = [value for column in columns for value in family[column].tolist()]
        if dtype.lower().startswith("u") and min(ids) < 0:
            with pytest.raises(ValueError, match="int64"):
                _encode(tables, retyped_group)
            return
        expected = _encode(tables, group).tables[rule.entity]
        got = _encode(tables, retyped_group).tables[rule.entity]
        assert got[rule.id_column].tolist() == expected[rule.id_column].tolist()
        pd.testing.assert_frame_equal(
            got.drop(columns=rule.id_column), expected.drop(columns=rule.id_column)
        )

    def test_roles_must_agree_with_the_partner_pointers(self) -> None:
        tables, _, family, membership, group = _example()
        # The couple's head is 10; claim the child 12 as the head's partner
        # and demote the real partner 11 to a child.
        roles = group.person_role.copy()
        assert roles.tolist() == ["head", "partner", "dependent_child", "head"]
        swapped = roles.copy()
        swapped.iloc[1] = UnitRole.DEPENDENT_CHILD.value
        swapped.iloc[2] = UnitRole.PARTNER.value
        bad = GroupMembership(
            entity="family", units=family, person_unit=membership, person_role=swapped
        )
        with pytest.raises(ValueError, match="partner must be its head's partner"):
            MAPPING.encode_groups(tables, {"Family": bad})
        demoted = roles.copy()
        demoted.iloc[1] = UnitRole.DEPENDENT_CHILD.value
        bad = GroupMembership(
            entity="family", units=family, person_unit=membership, person_role=demoted
        )
        with pytest.raises(ValueError, match="must be its partner"):
            MAPPING.encode_groups(tables, {"Family": bad})

    def test_a_child_relabelled_as_a_partner_is_refused(self) -> None:
        tables, rule, _, _, _ = _example()
        # Give the lone adult 13 a child, 14, so 13 heads a unit with no
        # partner in it: no head-side check can see a partner label there.
        person = tables["person"]
        child = person.iloc[[2]].assign(person_id=14)
        child["parent_1_person_id"] = pd.array([13], dtype="Int64")
        person = pd.concat([person, child], ignore_index=True)
        tables = {**tables, "person": person}
        family, membership = build_benefit_units(person, tables["household"], rule)
        group = benefit_unit_membership(person, family, membership, rule)
        roles = group.person_role.copy()
        assert membership.tolist() == [10, 10, 10, 13, 13]
        assert roles.tolist() == [
            "head",
            "partner",
            "dependent_child",
            "head",
            "dependent_child",
        ]
        MAPPING.encode_groups(tables, {"Family": group})
        roles.iloc[4] = UnitRole.PARTNER.value
        bad = GroupMembership(
            entity="family", units=family, person_unit=membership, person_role=roles
        )
        with pytest.raises(ValueError, match="partner must be its head's partner"):
            MAPPING.encode_groups(tables, {"Family": bad})

    def test_memberships_are_typed_and_distinct(self) -> None:
        tables, _, _, _, group = _example()
        with pytest.raises(ValueError, match="not a GroupMembership"):
            MAPPING.encode_groups(tables, {"Family": object()})
        with pytest.raises(ValueError, match="Two engine entities"):
            MAPPING.encode_groups(tables, {"Family": group, "TaxUnit": group})

    def test_a_state_binding_needs_a_membership(self) -> None:
        tables, _, _, _, group = _example()
        other = StateBinding(
            engine_input="x",
            engine_entity="TaxUnit",
            columns=("person.receives_a",),
            test=StateTest.ANY_TRUE,
            group_rule=GroupRule.ANY_MEMBER,
            note="evidence",
        )
        with pytest.raises(ValueError, match="'TaxUnit' membership"):
            MAPPING.encode_groups(tables, {"Family": group}, state_bindings=[other])

    def test_unit_composition_needs_membership_in_plain_encode(self) -> None:
        # plain encode defers every group binding, unit composition included
        tables, _, _, _, _ = _example()
        deferred = {binding.engine_input for binding in MAPPING.encode(tables).deferred}
        assert {f"unit_{feature.value}" for feature in UnitFeature} <= deferred


class TestConstruction:
    @pytest.mark.parametrize(
        ("binding", "message"),
        [
            (
                dict(engine_entity="Person"),
                "lives on an engine group entity",
            ),
            (dict(rule=GroupRule.ANY_MEMBER), "reference_member"),
            (dict(concepts=("fact:person.age",)), "unit construction reads"),
        ],
    )
    def test_unit_composition_rules(self, binding, message) -> None:
        fields = {
            "engine_input": "flag",
            "concepts": UNIT_CONCEPTS,
            "transform": UnitComposition(feature=UnitFeature.SOLE_PARENT),
            "rule": GroupRule.REFERENCE_MEMBER,
            **binding,
        }
        entity = fields.pop("engine_entity", "Family")
        rule = fields.pop("rule")
        if entity == "Person":
            rule = None
        with pytest.raises(ValueError, match=message):
            _mapping([_binding(engine_entity=entity, rule=rule, **fields)])

    @pytest.mark.parametrize(
        ("concepts", "message"),
        [
            (("fact:person.employment_income",), "two or more"),
            (("fact:person.age", "fact:person.weeks_worked"), "not an amount"),
            (
                ("fact:person.employment_income", "fact:household.rent"),
                "one entity",
            ),
        ],
    )
    def test_scaled_sums_add_amounts_on_one_entity(self, concepts, message) -> None:
        with pytest.raises(ValueError, match=message):
            _binding("x", concepts, ScaledSum(factor=WEEK), GroupRule.SUM_OVER_MEMBERS)

    @pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan")])
    def test_scaled_sum_factors_are_positive(self, factor) -> None:
        with pytest.raises(ValueError, match="positive"):
            ScaledSum(factor=factor)

    def test_a_scaled_sum_on_a_plain_input_encodes(self) -> None:
        binding = InputBinding(
            engine_input="weekly",
            engine_entity="Person",
            concepts=("fact:person.employment_income", "fact:person.interest_income"),
            transform=ScaledSum(factor=WEEK),
            relation=AlignmentRelation.APPROXIMATE,
            note="evidence",
        )
        tables, _, _, _, _ = _example()
        encoded = _mapping([binding]).encode(tables)
        person = tables["person"]
        np.testing.assert_allclose(
            encoded.tables["person"]["weekly"],
            (person["employment_income"] + person["interest_income"]) * WEEK,
        )
        assert binding.ref not in {b.ref for b in encoded.deferred}
        assert (
            "fact:person.employment_income"
            not in _mapping([binding]).invertible_concepts()
        )


class TestSerialization:
    @pytest.mark.parametrize(
        "transform",
        [ScaledSum(factor=WEEK), *(UnitComposition(feature=f) for f in UnitFeature)],
    )
    def test_new_transforms_round_trip(self, transform) -> None:
        assert transform_from_dict(
            json.loads(json.dumps(transform_to_dict(transform)))
        ) == (transform)

    @pytest.mark.parametrize(
        "data",
        [
            {"kind": "scaled_sum"},
            {"kind": "scaled_sum", "factor": "1"},
            {"kind": "scaled_sum", "factor": True},
            {"kind": "scaled_sum", "factor": 1, "x": 1},
            {"kind": "unit_composition"},
            {"kind": "unit_composition", "feature": "orphan"},
            {"kind": "unit_composition", "feature": 3},
        ],
    )
    def test_malformed_new_transforms_raise_value_error(self, data) -> None:
        with pytest.raises(ValueError):
            transform_from_dict(data)

    @pytest.mark.parametrize("binding", STATE)
    def test_state_bindings_round_trip(self, binding) -> None:
        payload = json.loads(json.dumps(binding.to_dict()))
        assert StateBinding.from_dict(payload) == binding

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"columns": []}, "distinct columns"),
            ({"columns": ["person.a", "household.b"]}, "one entity"),
            ({"columns": ["family.a"]}, "state column"),
            ({"group_rule": "household_value"}, "collapses by"),
            ({"test": "equals"}, "compares one column"),
            ({"value": 1}, "takes no value"),
            ({"note": ""}, "note"),
            ({"negate": 1}, "negate"),
            ({"test": "equals", "columns": ["person.a"], "value": 1.5}, "exact"),
            ({"group_rule": "sum_over_members"}, "collapses by"),
            ({"extra": 1}, "unexpected"),
        ],
    )
    def test_malformed_state_bindings_raise_value_error(self, change, message) -> None:
        payload = {**STATE[0].to_dict(), **change}
        with pytest.raises(ValueError, match=message):
            StateBinding.from_dict(payload)

    @settings(max_examples=200, deadline=None)
    @given(
        data=st.recursive(
            st.none() | st.booleans() | st.integers() | st.text(max_size=5),
            lambda inner: (
                st.lists(inner, max_size=3)
                | st.dictionaries(
                    st.sampled_from(
                        [
                            "engine_input",
                            "engine_entity",
                            "columns",
                            "test",
                            "group_rule",
                            "note",
                            "value",
                        ]
                    )
                    | st.text(max_size=4),
                    inner,
                    max_size=6,
                )
            ),
            max_leaves=10,
        )
    )
    def test_arbitrary_state_payloads_only_raise_value_error(self, data) -> None:
        with pytest.raises(ValueError):
            StateBinding.from_dict(data)


class TestNewZealandMapping:
    AS = "nz/statutes/social_security/accommodation_supplement/core.yaml"
    NZ_RULE = BenefitUnitRule(
        entity="family",
        dependent_child=DependentChildRule(max_age=17, financial_independence=None),
        unparented_child="reference_person_unit",
        split_parents="first_parent_unit",
        provenance={"max_age": "test"},
    )

    @PROPERTY
    @given(tables=concept_frames(max_households=4, max_members=6, min_id=-(10**9)))
    def test_every_accommodation_supplement_group_binding_executes(
        self, tables
    ) -> None:
        mapping = axiom_concept_mapping("nz")
        person, household = tables["person"], tables["household"]
        family, membership = build_benefit_units(person, household, self.NZ_RULE)
        group = benefit_unit_membership(person, family, membership, self.NZ_RULE)
        encoded = mapping.encode_groups(tables, {"Family": group}, modules=[self.AS])
        expected = {
            binding.engine_input
            for binding in mapping.bindings
            if binding.module == self.AS
        }
        table = encoded.tables["family"]
        assert set(table.columns) == {"family_id", *expected}
        assert encoded.deferred == ()
        for binding in mapping.bindings:
            if binding.module != self.AS:
                continue
            kind = table[binding.engine_input].dtype.kind
            assert kind in ("b", "f"), binding.engine_input
        rent = table["accommodation_supplement_weekly_rent_paid"].sum()
        np.testing.assert_allclose(rent, household["rent"].sum() * WEEK, rtol=1e-12)

    # Hand-computed households. Concept amounts are annual flows and these
    # inputs are weekly amounts, so every expected value below is the annual
    # figure divided by 52 weeks, worked by hand rather than read from a
    # binding's factor.
    RATES = "nz/statutes/social_security/main_benefits/rates.yaml"
    #: Each person's own weekly income from the seven income concepts.
    OWN_WEEKLY_INCOME = {
        101: 1_000.0 + 10.0,  # wages 52,000; interest 520
        102: 200.0 + 20.0,  # non-farm self-employment 10,400; dividends 1,040
        103: 0.0,
        104: 0.0,
        105: 400.0 + 50.0 + 50.0,  # wages 20,800; rent 2,600; private pension 2,600
        201: 250.0 + 50.0,  # wages 13,000; farm self-employment 2,600
        202: 0.0,
    }

    @staticmethod
    def _households() -> dict[str, pd.DataFrame]:
        """Two households, every income concept non-zero somewhere.

        Household 1 owns with a mortgage (interest 15,600 and principal 5,200
        a year): a couple, 101 (the reference person) and 102, their children
        103 and 104, and an adult boarder, 105. Household 2 rents privately at
        15,600 a year: a sole parent, 201, and their child, 202.
        """

        rows = [
            (101, 1, 41, 102, None, None),
            (102, 1, 39, 101, None, None),
            (103, 1, 9, None, 101, 102),
            (104, 1, 6, None, 101, 102),
            (105, 1, 30, None, None, None),
            (201, 2, 28, None, None, None),
            (202, 2, 3, None, 201, None),
        ]
        person = pd.DataFrame(
            rows,
            columns=[
                "person_id",
                "person_household_id",
                "age",
                "partner_person_id",
                "parent_1_person_id",
                "parent_2_person_id",
            ],
        )
        for column in ("partner_person_id", "parent_1_person_id", "parent_2_person_id"):
            person[column] = pd.array(person[column].tolist(), dtype="Int64")
        for item in CONCEPTS:
            if item.entity == "person" and item.monetary is not None:
                person[item.name] = 0.0
        annual = {
            "employment_income": {101: 52_000.0, 105: 20_800.0, 201: 13_000.0},
            "interest_income": {101: 520.0},
            "nonfarm_self_employment_income": {102: 10_400.0},
            "dividend_income": {102: 1_040.0},
            "rental_income": {105: 2_600.0},
            "private_pension_income": {105: 2_600.0},
            "farm_self_employment_income": {201: 2_600.0},
        }
        for column, amounts in annual.items():
            person[column] = person["person_id"].map(amounts).fillna(0.0)
        household = pd.DataFrame(
            {
                "household_id": [1, 2],
                "reference_person_id": pd.array([101, 201], dtype="Int64"),
                "tenure": ["owned_with_mortgage", "rented_private"],
            }
        )
        for item in CONCEPTS:
            if item.entity == "household" and item.monetary is not None:
                household[item.name] = 0.0
        household["mortgage_interest"] = [15_600.0, 0.0]
        household["mortgage_principal"] = [5_200.0, 0.0]
        household["rent"] = [0.0, 15_600.0]
        return {"person": person, "household": household}

    def test_hand_computed_households_get_the_supplement_inputs(self) -> None:
        tables = self._households()
        person, household = tables["person"], tables["household"]
        family, membership = build_benefit_units(person, household, self.NZ_RULE)
        group = benefit_unit_membership(person, family, membership, self.NZ_RULE)
        table = (
            axiom_concept_mapping("nz")
            .encode_groups(tables, {"Family": group}, modules=[self.AS])
            .tables["family"]
        )
        # The couple's family (101), the boarder's (105), the sole parent's (201).
        assert table["family_id"].tolist() == [101, 105, 201]
        flags = {
            "accommodation_supplement_in_relationship": [True, False, False],
            "accommodation_supplement_has_dependent_children": [True, False, True],
            "accommodation_supplement_has_two_or_more_dependent_children": [
                True,
                False,
                False,
            ],
            "accommodation_supplement_sole_parent": [False, False, True],
            # Every family in an owner-occupied household, the boarder's too.
            "accommodation_supplement_owns_premises_and_not_joint_owner_with_resident": [
                True,
                True,
                False,
            ],
        }
        for name, expected in flags.items():
            assert table[name].dtype == bool, name
            assert table[name].tolist() == expected, name
        income = self.OWN_WEEKLY_INCOME
        amounts = {
            # (15,600 + 5,200) / 52, on the family holding the reference person.
            "accommodation_supplement_owner_weekly_required_payments": [400.0, 0, 0],
            # 15,600 / 52.
            "accommodation_supplement_weekly_rent_paid": [0, 0, 300.0],
            # Summed over each family's members.
            "accommodation_supplement_relevant_weekly_income": [
                income[101] + income[102] + income[103] + income[104],
                income[105],
                income[201] + income[202],
            ],
        }
        assert amounts["accommodation_supplement_relevant_weekly_income"] == [
            1_230.0,
            500.0,
            300.0,
        ]
        for name, expected in amounts.items():
            assert table[name].dtype.kind == "f", name
            np.testing.assert_allclose(
                table[name], expected, rtol=1e-12, atol=1e-9, err_msg=name
            )

    def test_main_benefit_income_inputs_are_weekly_own_income(self) -> None:
        tables = self._households()
        mapping = axiom_concept_mapping("nz")
        encoded = mapping.encode(
            tables,
            shares={name: 0.5 for name in mapping.share_parameters()},
            take_up_rates={name: None for name in mapping.take_up_programs()},
        ).tables["person"]
        names = {
            "jobseeker_support_total_income",
            "main_benefit_income_test_assessable_income",
            "sole_parent_support_total_income",
            "supported_living_payment_total_income",
        }
        assert names <= {
            binding.engine_input
            for binding in mapping.bindings
            if binding.module == self.RATES
        }
        expected = [self.OWN_WEEKLY_INCOME[pid] for pid in encoded["person_id"]]
        for name in sorted(names):
            np.testing.assert_allclose(
                encoded[name], expected, rtol=1e-12, atol=1e-9, err_msg=name
            )

    def test_every_amount_conversion_is_annual_to_weekly(self) -> None:
        # The New Zealand rules this mapping feeds state amounts weekly, and
        # every amount concept is an annual flow, so each scaling divides by
        # the 52 weeks of a year. A monthly or fortnightly input would need a
        # different factor and a note saying so.
        by_id = {item.id: item for item in CONCEPTS}
        scaled = [
            binding
            for binding in axiom_concept_mapping("nz").bindings
            if isinstance(binding.transform, Scale | ScaledSum)
        ]
        assert {binding.engine_input for binding in scaled} >= {
            "accommodation_supplement_owner_weekly_required_payments",
            "accommodation_supplement_relevant_weekly_income",
            "jobseeker_support_total_income",
            "main_benefit_income_test_assessable_income",
            "sole_parent_support_total_income",
            "supported_living_payment_total_income",
        }
        for binding in scaled:
            for concept_id in binding.concepts:
                concept = by_id[concept_id]
                assert concept.period == "year", (binding.engine_input, concept_id)
                assert concept.temporal_basis is TemporalBasis.ANNUAL_FLOW
            assert "1/52" in binding.note, binding.engine_input
            assert binding.transform.factor * 52 == pytest.approx(1.0, abs=1e-15), (
                binding.engine_input
            )
