"""Benefit units built from concept pointers: the invariants, as properties.

Every property runs over generated valid concept frames
(:func:`test_support.microcosm_frame.concept_frames.concept_frames`) and
generated rules, so a unit rule is checked against every pointer shape the
schema allows rather than a few hand-picked households:

- partition: every person is in exactly one unit, and every unit has members;
- every unit has an adult (its head) and at most one partner;
- partners share a unit, and the partner role is exactly the head's partner;
- every dependent child shares a unit with a co-resident parent, or with the
  household's reference person when it has none;
- units nest in households;
- family weights are inherited: the sum of family weights equals the sum of
  household weight times units per household;
- unit ids are deterministic and independent of row order;
- ids of every integer dtype the contract accepts are kept exactly, or
  refused when ``int64`` cannot hold them, and the dtype changes no unit.
"""

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.concept_mapping import UnitRole
from microcosm.frame.concepts import validate_concept_tables
from microcosm.frame.unit_construction import (
    BENEFIT_UNIT_RULE_FORMAT,
    BenefitUnitRule,
    DependentChildRule,
    FinancialIndependenceTest,
    SplitParentsPolicy,
    UnparentedChildPlacement,
    benefit_unit_attributes,
    benefit_unit_membership,
    benefit_unit_roles,
    build_benefit_units,
    units_per_household,
)
from test_support.microcosm_frame.concept_frames import concept_frames
from test_support.microcosm_frame.unit_rules import unit_rules

PROPERTY = settings(max_examples=150, deadline=None)
#: Ids may be negative: the concept-frame contract accepts any unique integer,
#: so no unit reading may treat an id such as -1 as "absent".
FRAMES = concept_frames(max_households=5, max_members=6, min_id=-(10**9))
POINTERS = ("partner_person_id", "parent_1_person_id", "parent_2_person_id")
INT64_MAX = int(np.iinfo(np.int64).max)
#: Every integer dtype the concept-frame contract accepts for an id column;
#: pointers take the matching nullable dtype.
ID_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")


def _nullable(dtype: str) -> str:
    return "U" + dtype[1:].capitalize() if dtype.startswith("u") else dtype.capitalize()


@st.composite
def retyped_frames(draw) -> dict[str, pd.DataFrame]:
    """A valid frame whose ids and pointers use accepted integer dtypes.

    Person ids (and the pointers that name them) take one dtype and household
    ids another, so signed and unsigned tables are mixed; ids come from each
    dtype's whole range, with its edges and int64's edge mixed in.
    """

    tables = draw(concept_frames(max_households=4, max_members=4))
    person, household = tables["person"].copy(), tables["household"].copy()
    person_dtype = draw(st.sampled_from(ID_DTYPES))
    household_dtype = draw(st.sampled_from(ID_DTYPES))

    def renamed(old: list[int], dtype: str) -> dict[int, int]:
        info = np.iinfo(dtype)
        low, high = int(info.min), int(info.max)
        edges = [
            value
            for value in (low, high, 0, -1, INT64_MAX, INT64_MAX + 1)
            if low <= value <= high
        ]
        ids = st.integers(low, high) | st.sampled_from(edges)
        new = draw(st.lists(ids, min_size=len(old), max_size=len(old), unique=True))
        return dict(zip(old, new, strict=True))

    persons = renamed(person["person_id"].tolist(), person_dtype)
    households = renamed(household["household_id"].tolist(), household_dtype)
    nullable = _nullable(person_dtype)

    def ids(values: pd.Series, names: dict[int, int], dtype: str) -> np.ndarray:
        return np.array([names[value] for value in values.tolist()], dtype=dtype)

    def pointers(values: pd.Series) -> pd.api.extensions.ExtensionArray:
        return pd.array(
            [None if pd.isna(value) else persons[int(value)] for value in values],
            dtype=nullable,
        )

    person["person_id"] = ids(person["person_id"], persons, person_dtype)
    person["person_household_id"] = ids(
        person["person_household_id"], households, household_dtype
    )
    for column in POINTERS:
        person[column] = pointers(person[column])
    household["household_id"] = ids(
        household["household_id"], households, household_dtype
    )
    household["reference_person_id"] = pointers(household["reference_person_id"])
    return {"person": person, "household": household}


def _as_int64(person: pd.DataFrame, household: pd.DataFrame):
    """The same frame with int64 ids and Int64 pointers."""

    person = person.astype(
        {
            "person_id": np.int64,
            "person_household_id": np.int64,
            **dict.fromkeys(POINTERS, "Int64"),
        }
    )
    household = household.astype(
        {"household_id": np.int64, "reference_person_id": "Int64"}
    )
    return person, household


def _rows(person: pd.DataFrame, column: str) -> np.ndarray:
    ids = pd.Index(person["person_id"].to_numpy())
    values = person[column]
    out = np.full(len(person), -1, dtype=np.int64)
    present = values.notna().to_numpy()
    out[present] = ids.get_indexer(values[present].to_numpy(dtype=np.int64))
    return out


def _reference_rows(person: pd.DataFrame, household: pd.DataFrame) -> np.ndarray:
    ids = pd.Index(person["person_id"].to_numpy())
    by_household = pd.Series(
        ids.get_indexer(household["reference_person_id"].to_numpy(dtype=np.int64)),
        index=household["household_id"].to_numpy(),
    )
    return by_household.reindex(person["person_household_id"].to_numpy()).to_numpy()


class TestInvariants:
    @PROPERTY
    @given(tables=FRAMES, rule=unit_rules())
    def test_units_partition_persons_and_nest_in_households(self, tables, rule) -> None:
        person, household = tables["person"], tables["household"]
        family, membership = build_benefit_units(person, household, rule)
        ids = family[rule.id_column]
        # partition: one unit per person, every unit has members
        assert membership.index.equals(person.index)
        assert membership.notna().all()
        assert membership.isin(ids).all()
        assert set(membership) == set(ids)
        assert ids.is_unique
        # nesting: members live in the household the unit declares
        declared = pd.Series(
            family[rule.household_column].to_numpy(), index=ids.to_numpy()
        )
        assert np.array_equal(
            declared.reindex(membership.to_numpy()).to_numpy(),
            person["person_household_id"].to_numpy(),
        )
        # Frame group tables are sorted by id, with int64 structure columns
        assert ids.is_monotonic_increasing
        for column in (rule.id_column, rule.household_column, rule.head_column):
            assert family[column].dtype == np.int64
        assert membership.dtype == np.int64

    @PROPERTY
    @given(tables=FRAMES, rule=unit_rules())
    def test_every_unit_has_an_adult_and_partners_share_it(self, tables, rule) -> None:
        person, household = tables["person"], tables["household"]
        family, membership = build_benefit_units(person, household, rule)
        roles = benefit_unit_roles(person, family, membership, rule)
        heads = pd.Series(
            family[rule.head_column].to_numpy(), index=family[rule.id_column]
        )
        unit_of = pd.Series(membership.to_numpy(), index=person["person_id"].to_numpy())
        # the head is a member, the only head, and an adult
        assert np.array_equal(
            unit_of.reindex(heads.to_numpy()).to_numpy(), heads.index.to_numpy()
        )
        role_counts = pd.crosstab(membership.to_numpy(), roles.to_numpy()).reindex(
            columns=[role.value for role in UnitRole], fill_value=0
        )
        assert (role_counts[UnitRole.HEAD.value] == 1).all()
        assert (role_counts[UnitRole.PARTNER.value] <= 1).all()
        partner = _rows(person, "partner_person_id")
        paired = partner >= 0
        assert np.array_equal(
            membership.to_numpy()[paired], membership.to_numpy()[partner[paired]]
        )
        assert not roles[paired].eq(UnitRole.DEPENDENT_CHILD.value).any()
        # the partner role is held by the head's partner and nobody else
        # (row positions, so no person id can pass for a missing partner)
        head_row = pd.Index(person["person_id"].to_numpy()).get_indexer(
            heads.reindex(membership.to_numpy()).to_numpy()
        )
        assert np.array_equal(
            roles.eq(UnitRole.PARTNER.value).to_numpy(),
            partner[head_row] == np.arange(len(person)),
        )

    @PROPERTY
    @given(tables=FRAMES, rule=unit_rules())
    def test_dependent_children_live_with_a_parent_or_caregiver(
        self, tables, rule
    ) -> None:
        person, household = tables["person"], tables["household"]
        family, membership = build_benefit_units(person, household, rule)
        roles = benefit_unit_roles(person, family, membership, rule).to_numpy()
        unit = membership.to_numpy()
        parent = _rows(person, "parent_1_person_id")
        reference = _reference_rows(person, household)
        child = roles == UnitRole.DEPENDENT_CHILD.value
        with_parent = child & (parent >= 0)
        assert np.array_equal(unit[with_parent], unit[parent[with_parent]])
        without = child & (parent < 0)
        assert np.array_equal(unit[without], unit[reference[without]])
        # a dependent child meets the rule; partners and parents never do
        ages = person["age"].to_numpy()
        assert (ages[child] <= rule.dependent_child.max_age).all()
        assert person["partner_person_id"][child].isna().all()
        named = set(person["parent_1_person_id"].dropna()) | set(
            person["parent_2_person_id"].dropna()
        )
        assert not person["person_id"][child].isin(named).any()
        test = rule.dependent_child.financial_independence
        if test is not None:
            hours = person["usual_weekly_hours"].to_numpy()
            independent = (ages >= test.min_age) & (
                hours >= test.min_usual_weekly_hours
            )
            assert not (independent & child).any()
        if rule.unparented_child is UnparentedChildPlacement.OWN_UNIT:
            assert not without.any()

    @PROPERTY
    @given(
        tables=FRAMES,
        rule=unit_rules(),
        weights=st.lists(
            st.floats(min_value=0.001, max_value=1e6, allow_nan=False),
            min_size=5,
            max_size=5,
        ),
    )
    def test_family_weights_are_inherited_from_households(
        self, tables, rule, weights
    ) -> None:
        person, household = tables["person"], tables["household"]
        family, membership = build_benefit_units(person, household, rule)
        household_weights = np.asarray(weights[: len(household)], dtype=np.float64)
        order = np.argsort(household["household_id"].to_numpy())
        frame_person = person.loc[:, ["person_id", "person_household_id"]].assign(
            **{rule.membership_column: membership.to_numpy()}
        )
        frame = Frame(
            schema=EntitySchema(group_entities=("household", rule.entity)),
            tables={
                "person": frame_person,
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
        family_weights = frame.resolve_weights(rule.entity).values
        counts = units_per_household(household, family, rule).to_numpy()
        np.testing.assert_allclose(
            family_weights.sum(), (household_weights * counts).sum(), rtol=1e-12
        )
        by_household = pd.Series(household_weights, index=household["household_id"])
        np.testing.assert_array_equal(
            family_weights,
            by_household.reindex(family[rule.household_column]).to_numpy(),
        )

    @PROPERTY
    @given(
        tables=FRAMES,
        rule=unit_rules(),
        data=st.data(),
    )
    def test_units_do_not_depend_on_row_order(self, tables, rule, data) -> None:
        person, household = tables["person"], tables["household"]
        family, membership = build_benefit_units(person, household, rule)
        order = data.draw(st.permutations(range(len(person))))
        shuffled = person.iloc[list(order)].reset_index(drop=True)
        family_2, membership_2 = build_benefit_units(shuffled, household, rule)
        pd.testing.assert_frame_equal(family, family_2)
        first = pd.Series(membership.to_numpy(), index=person["person_id"].to_numpy())
        second = pd.Series(
            membership_2.to_numpy(), index=shuffled["person_id"].to_numpy()
        )
        pd.testing.assert_series_equal(first.sort_index(), second.sort_index())

    @PROPERTY
    @given(tables=FRAMES, rule=unit_rules())
    def test_attributes_count_the_roles(self, tables, rule) -> None:
        person, household = tables["person"], tables["household"]
        family, membership = build_benefit_units(person, household, rule)
        attributes = benefit_unit_attributes(person, family, membership, rule)
        roles = benefit_unit_roles(person, family, membership, rule)
        counts = (
            pd.crosstab(membership.to_numpy(), roles.to_numpy())
            .reindex(columns=[role.value for role in UnitRole], fill_value=0)
            .reindex(family[rule.id_column], fill_value=0)
        )
        children = counts[UnitRole.DEPENDENT_CHILD.value].to_numpy()
        assert np.array_equal(attributes["n_dependent_children"], children)
        assert (attributes["n_adults"].between(1, 2)).all()
        assert np.array_equal(
            attributes["n_members"], attributes["n_adults"] + children
        )
        assert set(attributes["family_type"]) <= {
            "single",
            "couple",
            "sole_parent",
            "couple_with_children",
        }
        assert (
            attributes["youngest_dependent_child_age"]
            .isna()
            .eq(attributes["n_dependent_children"] == 0)
            .all()
        )
        membership_view = benefit_unit_membership(person, family, membership, rule)
        assert membership_view.entity == rule.entity
        pd.testing.assert_series_equal(membership_view.person_role, roles)

    @PROPERTY
    @given(tables=retyped_frames(), rule=unit_rules())
    def test_ids_of_every_accepted_dtype_are_kept_exactly_or_refused(
        self, tables, rule
    ) -> None:
        person, household = tables["person"], tables["household"]
        # Python integers throughout, so no cast can make two ids agree.
        person_ids = person["person_id"].tolist()
        declared = dict(
            zip(person_ids, person["person_household_id"].tolist(), strict=True)
        )
        household_ids = household["household_id"].tolist()
        if max(person_ids + household_ids) > INT64_MAX:
            with pytest.raises(ValueError, match="int64"):
                build_benefit_units(person, household, rule)
            return
        family, membership = build_benefit_units(person, household, rule)
        # identity: a unit's id is its head's own person id, and it nests in
        # the household its head declares
        unit_ids = family[rule.id_column].tolist()
        unit_households = family[rule.household_column].tolist()
        assert unit_ids == family[rule.head_column].tolist()
        assert unit_households == [declared[head] for head in unit_ids]
        # nesting: each person's unit lies in the person's declared household
        household_of = dict(zip(unit_ids, unit_households, strict=True))
        assert [household_of[unit] for unit in membership.tolist()] == [
            declared[person_id] for person_id in person_ids
        ]
        # unit counts: every unit counted once, in its declared household
        counts = units_per_household(household, family, rule)
        expected = Counter(unit_households)
        assert counts.tolist() == [expected[value] for value in household_ids]
        assert int(counts.sum()) == len(family)
        # the dtype changes nothing: the same ids as int64 build the same units
        person_64, household_64 = _as_int64(person, household)
        family_64, membership_64 = build_benefit_units(person_64, household_64, rule)
        pd.testing.assert_frame_equal(family, family_64)
        pd.testing.assert_series_equal(membership, membership_64)
        pd.testing.assert_series_equal(
            benefit_unit_roles(person, family, membership, rule),
            benefit_unit_roles(person_64, family_64, membership_64, rule),
        )
        pd.testing.assert_frame_equal(
            benefit_unit_attributes(person, family, membership, rule),
            benefit_unit_attributes(person_64, family_64, membership_64, rule),
        )


class TestPlacement:
    RULE = BenefitUnitRule(
        entity="family",
        dependent_child=DependentChildRule(max_age=17, financial_independence=None),
        unparented_child="reference_person_unit",
        split_parents="refuse",
        provenance={"max_age": "test"},
    )

    @staticmethod
    def _frame(rows, reference) -> tuple[pd.DataFrame, pd.DataFrame]:
        person = pd.DataFrame(
            rows,
            columns=[
                "person_id",
                "person_household_id",
                "age",
                "partner_person_id",
                "parent_1_person_id",
                "parent_2_person_id",
                "usual_weekly_hours",
            ],
        )
        for column in ("partner_person_id", "parent_1_person_id", "parent_2_person_id"):
            person[column] = pd.array(person[column].tolist(), dtype="Int64")
        person["age"] = person["age"].astype(np.int64)
        household = pd.DataFrame(
            {
                "household_id": sorted(reference),
                "reference_person_id": pd.array(
                    [reference[key] for key in sorted(reference)], dtype="Int64"
                ),
            }
        )
        return person, household

    def test_a_couple_their_children_and_an_adult_child(self) -> None:
        person, household = self._frame(
            [
                (10, 1, 40, 11, None, None, 40.0),
                (11, 1, 38, 10, None, None, 0.0),
                (12, 1, 10, None, 10, 11, 0.0),
                (13, 1, 22, None, 10, None, 40.0),
            ],
            {1: 11},
        )
        family, membership = build_benefit_units(person, household, self.RULE)
        assert family["family_id"].tolist() == [11, 13]
        assert family["family_head_person_id"].tolist() == [11, 13]
        assert membership.tolist() == [11, 11, 11, 13]
        roles = benefit_unit_roles(person, family, membership, self.RULE)
        assert roles.tolist() == ["partner", "head", "dependent_child", "head"]

    def test_a_parentless_child_joins_the_reference_unit(self) -> None:
        person, household = self._frame(
            [(1, 1, 60, None, None, None, 0.0), (2, 1, 12, None, None, None, 0.0)],
            {1: 1},
        )
        _, membership = build_benefit_units(person, household, self.RULE)
        assert membership.tolist() == [1, 1]
        own = BenefitUnitRule(
            **{**self._fields(), "unparented_child": UnparentedChildPlacement.OWN_UNIT}
        )
        _, membership = build_benefit_units(person, household, own)
        assert membership.tolist() == [1, 2]

    def test_a_lone_young_reference_person_heads_their_unit(self) -> None:
        person, household = self._frame(
            [(1, 1, 17, None, None, None, 0.0), (2, 1, 15, None, None, None, 0.0)],
            {1: 1},
        )
        family, membership = build_benefit_units(person, household, self.RULE)
        assert membership.tolist() == [1, 1]
        attributes = benefit_unit_attributes(person, family, membership, self.RULE)
        assert attributes["family_type"].tolist() == ["sole_parent"]
        refuse = BenefitUnitRule(
            **{**self._fields(), "unparented_child": UnparentedChildPlacement.REFUSE}
        )
        with pytest.raises(ValueError, match="1 candidate"):
            build_benefit_units(person, household, refuse)

    def test_a_teen_parent_heads_their_own_unit(self) -> None:
        person, household = self._frame(
            [
                (1, 1, 45, None, None, None, 0.0),
                (2, 1, 16, None, 1, None, 0.0),
                (3, 1, 0, None, 2, None, 0.0),
            ],
            {1: 1},
        )
        _, membership = build_benefit_units(person, household, self.RULE)
        assert membership.tolist() == [1, 2, 2]

    def test_financial_independence_takes_a_worker_out_of_dependency(self) -> None:
        person, household = self._frame(
            [(1, 1, 45, None, None, None, 0.0), (2, 1, 17, None, 1, None, 40.0)],
            {1: 1},
        )
        _, membership = build_benefit_units(person, household, self.RULE)
        assert membership.tolist() == [1, 1]
        working = BenefitUnitRule(
            **{
                **self._fields(),
                "dependent_child": DependentChildRule(
                    max_age=17,
                    financial_independence=FinancialIndependenceTest(
                        min_age=16, min_usual_weekly_hours=30
                    ),
                ),
            }
        )
        _, membership = build_benefit_units(person, household, working)
        assert membership.tolist() == [1, 2]

    def test_parents_in_different_units_are_refused_or_resolved(self) -> None:
        person, household = self._frame(
            [
                (1, 1, 45, None, None, None, 0.0),
                (2, 1, 44, None, None, None, 0.0),
                (3, 1, 5, None, 2, 1, 0.0),
            ],
            {1: 1},
        )
        with pytest.raises(ValueError, match="different units"):
            build_benefit_units(person, household, self.RULE)
        first = BenefitUnitRule(
            **{**self._fields(), "split_parents": SplitParentsPolicy.FIRST_PARENT_UNIT}
        )
        _, membership = build_benefit_units(person, household, first)
        assert membership.tolist() == [1, 2, 2]

    def test_a_person_id_of_minus_one_is_a_person_not_a_missing_partner(
        self,
    ) -> None:
        # A sole parent and their child, whose id is -1: the child is not the
        # partner of a head who has none.
        person, household = self._frame(
            [(5, 1, 40, None, None, None, 0.0), (-1, 1, 10, None, 5, None, 0.0)],
            {1: 5},
        )
        family, membership = build_benefit_units(person, household, self.RULE)
        assert membership.tolist() == [5, 5]
        roles = benefit_unit_roles(person, family, membership, self.RULE)
        assert roles.tolist() == ["head", "dependent_child"]
        attributes = benefit_unit_attributes(person, family, membership, self.RULE)
        assert attributes["family_type"].tolist() == ["sole_parent"]
        assert attributes["n_dependent_children"].tolist() == [1]
        assert attributes["is_couple"].tolist() == [False]
        # Negative ids in every role: a couple -1 and -2 with their child -3,
        # and a lone adult -4 with no partner.
        person, household = self._frame(
            [
                (-1, 1, 40, -2, None, None, 0.0),
                (-2, 1, 38, -1, None, None, 0.0),
                (-3, 1, 10, None, -1, -2, 0.0),
                (-4, 1, 30, None, None, None, 0.0),
            ],
            {1: -2},
        )
        family, membership = build_benefit_units(person, household, self.RULE)
        assert family["family_id"].tolist() == [-4, -2]
        assert membership.tolist() == [-2, -2, -2, -4]
        roles = benefit_unit_roles(person, family, membership, self.RULE)
        assert roles.tolist() == ["partner", "head", "dependent_child", "head"]
        attributes = benefit_unit_attributes(person, family, membership, self.RULE)
        assert attributes["family_type"].tolist() == ["single", "couple_with_children"]

    @pytest.mark.parametrize(
        ("rows", "message"),
        [
            (
                [(1, 1, 30, 2, None, None, 0.0), (2, 2, 30, 1, None, None, 0.0)],
                "cross_household",
            ),
            (
                [(1, 1, 30, 2, None, None, 0.0), (2, 1, 30, None, None, None, 0.0)],
                "asymmetric",
            ),
            ([(1, 1, 30, None, 9, None, 0.0)], "dangling"),
        ],
    )
    def test_invalid_pointers_are_refused(self, rows, message) -> None:
        person, household = self._frame(rows, {1: 1, 2: 2})
        with pytest.raises(ValueError, match=message):
            build_benefit_units(person, household, self.RULE)

    def test_missing_columns_are_named(self) -> None:
        person, household = self._frame([(1, 1, 30, None, None, None, 0.0)], {1: 1})
        with pytest.raises(ValueError, match="parent_2_person_id"):
            build_benefit_units(
                person.drop(columns="parent_2_person_id"), household, self.RULE
            )

    def _fields(self) -> dict[str, object]:
        rule = self.RULE
        return {
            "entity": rule.entity,
            "dependent_child": rule.dependent_child,
            "unparented_child": rule.unparented_child,
            "split_parents": rule.split_parents,
            "provenance": rule.provenance,
        }


class TestIdRange:
    """Unit tables hold int64 ids; the contract also accepts unsigned ids."""

    RULE = TestPlacement.RULE

    @staticmethod
    def _one_adult(household_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        person = pd.DataFrame(
            {
                "person_id": np.array([1], dtype=np.int64),
                "person_household_id": np.array([household_id], dtype=np.uint64),
                "age": np.array([30], dtype=np.int64),
                **{column: pd.array([None], dtype="Int64") for column in POINTERS},
            }
        )
        household = pd.DataFrame(
            {
                "household_id": np.array([household_id], dtype=np.uint64),
                "reference_person_id": pd.array([1], dtype="Int64"),
            }
        )
        return person, household

    def test_the_largest_unsigned_id_int64_holds_is_kept_exactly(self) -> None:
        person, household = self._one_adult(INT64_MAX)
        family, membership = build_benefit_units(person, household, self.RULE)
        assert family.to_dict("list") == {
            "family_id": [1],
            "family_household_id": [INT64_MAX],
            "family_head_person_id": [1],
        }
        assert membership.tolist() == [1]
        assert units_per_household(household, family, self.RULE).tolist() == [1]

    @pytest.mark.parametrize("household_id", [INT64_MAX + 1, 2**64 - 1])
    def test_an_unsigned_household_id_int64_cannot_hold_is_refused(
        self, household_id
    ) -> None:
        # The concept-frame contract accepts this frame. Cast to int64, 2**63
        # became household -2**63: the unit nested in no household and its
        # household counted no unit.
        person, household = self._one_adult(household_id)
        assert not validate_concept_tables({"person": person, "household": household})
        columns = "['person.person_household_id', 'household.household_id']"
        with pytest.raises(ValueError, match=re.escape(columns)):
            build_benefit_units(person, household, self.RULE)

    def test_an_unsigned_pointer_int64_cannot_hold_is_refused(self) -> None:
        # 2**64 - 1 names no person here, but cast to int64 it is -1, who is
        # a person here: the pair would have passed as partners.
        person = pd.DataFrame(
            {
                "person_id": np.array([-1, 5], dtype=np.int64),
                "person_household_id": np.array([1, 1], dtype=np.int64),
                "age": np.array([40, 40], dtype=np.int64),
                "partner_person_id": pd.array([5, 2**64 - 1], dtype="UInt64"),
                "parent_1_person_id": pd.array([None, None], dtype="Int64"),
                "parent_2_person_id": pd.array([None, None], dtype="Int64"),
            }
        )
        household = pd.DataFrame(
            {
                "household_id": np.array([1], dtype=np.int64),
                "reference_person_id": pd.array([5], dtype="Int64"),
            }
        )
        with pytest.raises(ValueError, match=re.escape("['person.partner_person_id']")):
            build_benefit_units(person, household, self.RULE)

    def test_unit_readers_refuse_ids_int64_cannot_hold(self) -> None:
        person, household = TestPlacement._frame(
            [(-1, 1, 40, None, None, None, 0.0), (5, 1, 40, None, None, None, 0.0)],
            {1: 5},
        )
        family, membership = build_benefit_units(person, household, self.RULE)
        assert membership.tolist() == [-1, 5]
        # Unit 2**64 - 1 does not exist; cast to int64 it would be unit -1.
        wrapped = pd.Series(
            np.array([2**64 - 1, 5], dtype=np.uint64),
            index=membership.index,
            name=membership.name,
        )
        for read in (
            benefit_unit_roles,
            benefit_unit_attributes,
            benefit_unit_membership,
        ):
            with pytest.raises(ValueError, match="int64"):
                read(person, family, wrapped, self.RULE)
        # A float id is refused too: a cast would truncate it.
        with pytest.raises(ValueError, match="integer ids"):
            benefit_unit_roles(person, family, membership.astype(float), self.RULE)


class TestRuleSerialization:
    @PROPERTY
    @given(rule=unit_rules(refusing=True))
    def test_rules_round_trip_through_json(self, rule) -> None:
        payload = json.loads(json.dumps(rule.to_dict()))
        assert payload["format"] == BENEFIT_UNIT_RULE_FORMAT
        assert BenefitUnitRule.from_dict(payload) == rule
        assert hash(BenefitUnitRule.from_dict(payload)) == hash(rule)

    @pytest.mark.parametrize(
        "change",
        [
            {"format": "x"},
            {"entity": "household"},
            {"entity": "Family"},
            {"unparented_child": "nowhere"},
            {"split_parents": 3},
            {"provenance": {"": "x"}},
            {"dependent_child": {"max_age": 17}},
            {"dependent_child": {"max_age": True, "financial_independence": None}},
            {"dependent_child": {"max_age": 200, "financial_independence": None}},
            {
                "dependent_child": {
                    "max_age": 17,
                    "financial_independence": {
                        "min_age": 18,
                        "min_usual_weekly_hours": 30,
                    },
                }
            },
            {
                "dependent_child": {
                    "max_age": 17,
                    "financial_independence": {
                        "min_age": 16,
                        "min_usual_weekly_hours": 0,
                    },
                }
            },
            {"extra": 1},
        ],
    )
    def test_malformed_rules_raise_value_error(self, change) -> None:
        payload = {**TestPlacement.RULE.to_dict(), **change}
        with pytest.raises(ValueError):
            BenefitUnitRule.from_dict(payload)

    def test_a_test_must_be_declared_as_one(self) -> None:
        with pytest.raises(ValueError, match="declared as one"):
            DependentChildRule(
                max_age=17,
                financial_independence={"min_age": 16, "min_usual_weekly_hours": 30},
            )

    @PROPERTY
    @given(
        data=st.recursive(
            st.none() | st.booleans() | st.integers() | st.text(max_size=5),
            lambda inner: (
                st.lists(inner, max_size=3)
                | st.dictionaries(st.text(max_size=5), inner, max_size=3)
            ),
            max_leaves=8,
        )
    )
    def test_arbitrary_payloads_only_raise_value_error(self, data) -> None:
        with pytest.raises(ValueError):
            BenefitUnitRule.from_dict(data)
