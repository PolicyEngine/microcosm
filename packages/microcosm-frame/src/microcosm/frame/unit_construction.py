"""Benefit units built from concept relationship pointers.

A transported concept frame (:mod:`microcosm.frame.concepts`) has persons and
households only. A target country's rules assess many programs on a smaller
unit: a New Zealand benefit is assessed on a person, their partner and their
dependent children. This module builds that unit from the concept pointers
alone (``partner_person_id``, ``parent_1_person_id``, ``parent_2_person_id``
and the household's ``reference_person_id``), so no donor-country roster
column reaches the target engine. :mod:`microcosm.frame.units` is the
precedent for building unit structure in this package; it builds the US
units from raw CPS roster columns instead.

The rule
--------
Who counts as a dependent child is law, so it is a declared
:class:`BenefitUnitRule`, never a constant here. Its values come from the
country's RuleSpec where the RuleSpec encodes them and otherwise from the
country's spec data, with their citations. Given the rule:

1. A person is a *candidate child* when their age is at most the rule's
   ``max_age``, they have no co-resident partner, no co-resident person names
   them as a parent, and, when the rule declares a financial-independence
   test, they do not meet it. A partnered person and a parent are never
   dependent children, which keeps partners together and parents with their
   own children.
2. A candidate child with a co-resident parent is a *dependent child* of
   that parent's unit. A candidate child with none who is not the household's
   reference person is placed by the rule's ``unparented_child`` policy: in
   the unit that contains the household's reference person (the caregiver
   the pointers can name), in a unit of their own, or refused. A reference
   person is never their own caregiver, so a candidate child who is the
   reference person and has no co-resident parent heads their own unit.
3. Every person who is not a dependent child heads a unit with their
   co-resident partner, if any. These are the units' adults: every unit has
   one or two.

A child whose two co-resident parents are in different units (two parents who
are not partners of each other) is ambiguous; the rule's ``split_parents``
policy refuses the frame or places the child with ``parent_1_person_id``.
Pointers that leave the household, dangle, or break any other concept-frame
contract are refused before any unit is built
(:func:`~microcosm.frame.concepts.validate_concept_tables`), so units always
nest in households.

Identity
--------
A unit's id is its head's ``person_id``: unique, deterministic, and
independent of row order, so rebuilding from the same records reproduces the
same ids. A unit's head is one of its adults: the household's reference person
when they are one of them, otherwise the older adult of a couple, ties broken
by the smaller ``person_id`` (a reference person who is a dependent child is
never a head).
The unit table is sorted by id, as a :class:`~microcosm.frame.bundle.Frame`
group table must be, and carries the head (``<entity>_head_person_id``) and
the household the unit nests in (``<entity>_household_id``); consumers read
those columns rather than rely on the id's value. Units carry no weight of
their own: a unit takes its household's weight through its members
(:meth:`~microcosm.frame.bundle.Frame.resolve_weights`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from microcosm.frame.concept_mapping import GroupMembership, UnitRole
from microcosm.frame.concepts import (
    HOUSEHOLD_ID_COLUMN,
    PERSON_HOUSEHOLD_ID_COLUMN,
    PERSON_ID_COLUMN,
    _parsing,
    _record_fields,
    validate_concept_tables,
)

__all__ = [
    "BENEFIT_UNIT_RULE_FORMAT",
    "BenefitUnitRule",
    "DependentChildRule",
    "FinancialIndependenceTest",
    "SplitParentsPolicy",
    "UnparentedChildPlacement",
    "benefit_unit_attributes",
    "benefit_unit_membership",
    "benefit_unit_roles",
    "build_benefit_units",
    "units_per_household",
]

#: The serialized rule format :meth:`BenefitUnitRule.to_dict` writes.
BENEFIT_UNIT_RULE_FORMAT = "microcosm.benefit_unit_rule.v1"

_PERSON_ID = PERSON_ID_COLUMN
_PERSON_HOUSEHOLD_ID = PERSON_HOUSEHOLD_ID_COLUMN
_HOUSEHOLD_ID = HOUSEHOLD_ID_COLUMN
_ENTITY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_POINTERS = ("partner_person_id", "parent_1_person_id", "parent_2_person_id")


class UnparentedChildPlacement(StrEnum):
    """Where a candidate child with no co-resident parent goes.

    A candidate child who is the household's reference person heads their
    own unit whatever the placement: they cannot be their own caregiver.
    """

    #: Into the unit that contains the household's reference person.
    REFERENCE_PERSON_UNIT = "reference_person_unit"
    #: Into a unit of their own: they are not a dependent child.
    OWN_UNIT = "own_unit"
    #: Nowhere: the frame is refused.
    REFUSE = "refuse"


class SplitParentsPolicy(StrEnum):
    """What happens to a child whose two co-resident parents are not partners."""

    #: The frame is refused: the child's unit is ambiguous.
    REFUSE = "refuse"
    #: The child joins the unit of ``parent_1_person_id``.
    FIRST_PARENT_UNIT = "first_parent_unit"


@dataclass(frozen=True, kw_only=True)
class FinancialIndependenceTest:
    """When a candidate child counts as financially independent.

    A person at least :attr:`min_age` old who usually works at least
    :attr:`min_usual_weekly_hours` a week is not a dependent child. Only the
    employment arm of a statutory test can be read from concepts; receipt of
    a benefit or allowance is model state that depends on the units, so it
    cannot decide them.
    """

    min_age: int
    min_usual_weekly_hours: float

    def __post_init__(self) -> None:
        if isinstance(self.min_age, bool) or not isinstance(self.min_age, int):
            raise ValueError("A financial-independence age must be an integer.")
        hours = self.min_usual_weekly_hours
        if isinstance(hours, bool) or not isinstance(hours, int | float):
            raise ValueError("Financial-independence hours must be a number.")
        hours = float(hours)
        if not math.isfinite(hours) or not 0.0 < hours <= 168.0:
            raise ValueError(
                f"Financial-independence hours must lie on (0, 168], got {hours}."
            )
        if self.min_age < 0:
            raise ValueError("A financial-independence age cannot be negative.")
        object.__setattr__(self, "min_usual_weekly_hours", hours)


@dataclass(frozen=True, kw_only=True)
class DependentChildRule:
    """Who can be a dependent child.

    Attributes:
        max_age: The oldest age, in completed years, a dependent child can be.
        financial_independence: The test that takes a candidate out of
            dependency, or ``None`` when the rule declares none.
    """

    max_age: int
    financial_independence: FinancialIndependenceTest | None

    def __post_init__(self) -> None:
        if isinstance(self.max_age, bool) or not isinstance(self.max_age, int):
            raise ValueError("A dependent child's maximum age must be an integer.")
        if not 0 <= self.max_age <= 130:
            raise ValueError(f"max_age must lie on [0, 130], got {self.max_age}.")
        test = self.financial_independence
        if test is not None and not isinstance(test, FinancialIndependenceTest):
            raise ValueError("A financial-independence test must be declared as one.")
        if test is not None and test.min_age > self.max_age:
            raise ValueError(
                "A financial-independence age above max_age can never apply."
            )


@dataclass(frozen=True, kw_only=True)
class BenefitUnitRule:
    """How a country groups persons into benefit units.

    Attributes:
        entity: The frame group entity the units become (``family``). Its
            table carries ``<entity>_id``, ``<entity>_household_id`` and
            ``<entity>_head_person_id``; persons carry
            ``person_<entity>_id``.
        dependent_child: Who can be a dependent child.
        unparented_child: Where a candidate child with no co-resident parent
            goes.
        split_parents: What happens to a child whose co-resident parents are
            in different units.
        provenance: Where each value comes from (statute, RuleSpec rule or
            spec data, with citations). Carried verbatim for review; it does
            not change any unit.
    """

    entity: str
    dependent_child: DependentChildRule
    unparented_child: UnparentedChildPlacement
    split_parents: SplitParentsPolicy
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.entity, str) or not _ENTITY_NAME.match(self.entity):
            raise ValueError(f"A unit entity must be snake_case, got {self.entity!r}.")
        if self.entity in ("person", "household"):
            raise ValueError(f"Units cannot be named {self.entity!r}.")
        if not isinstance(self.dependent_child, DependentChildRule):
            raise ValueError("A unit rule needs a DependentChildRule.")
        object.__setattr__(
            self, "unparented_child", UnparentedChildPlacement(self.unparented_child)
        )
        object.__setattr__(
            self, "split_parents", SplitParentsPolicy(self.split_parents)
        )
        if not isinstance(self.provenance, Mapping) or not all(
            isinstance(key, str) and isinstance(text, str) and key and text
            for key, text in self.provenance.items()
        ):
            raise ValueError("A unit rule's provenance maps text to non-empty text.")
        object.__setattr__(self, "provenance", dict(self.provenance))

    def __hash__(self) -> int:
        return hash(
            (
                self.entity,
                self.dependent_child,
                self.unparented_child,
                self.split_parents,
                tuple(sorted(self.provenance.items())),
            )
        )

    @property
    def id_column(self) -> str:
        """The unit table's id column."""
        return f"{self.entity}_id"

    @property
    def household_column(self) -> str:
        """The unit table's column naming the household the unit nests in."""
        return f"{self.entity}_household_id"

    @property
    def head_column(self) -> str:
        """The unit table's column naming the unit's head."""
        return f"{self.entity}_head_person_id"

    @property
    def membership_column(self) -> str:
        """The person table's column naming each person's unit."""
        return f"person_{self.entity}_id"

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form that :meth:`from_dict` reads back unchanged."""
        test = self.dependent_child.financial_independence
        return {
            "format": BENEFIT_UNIT_RULE_FORMAT,
            "entity": self.entity,
            "dependent_child": {
                "max_age": self.dependent_child.max_age,
                "financial_independence": None
                if test is None
                else {
                    "min_age": test.min_age,
                    "min_usual_weekly_hours": test.min_usual_weekly_hours,
                },
            },
            "unparented_child": self.unparented_child.value,
            "split_parents": self.split_parents.value,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BenefitUnitRule:
        """The rule ``data`` describes (see :meth:`to_dict`).

        Raises:
            ValueError: If the format is not :data:`BENEFIT_UNIT_RULE_FORMAT`,
                or a field is missing, unexpected or malformed.
        """
        with _parsing("benefit unit rule"):
            fields = _record_fields(
                data,
                "benefit unit rule",
                required=(
                    "format",
                    "entity",
                    "dependent_child",
                    "unparented_child",
                    "split_parents",
                    "provenance",
                ),
            )
            if fields.pop("format") != BENEFIT_UNIT_RULE_FORMAT:
                raise ValueError(
                    f"A benefit unit rule must declare format "
                    f"{BENEFIT_UNIT_RULE_FORMAT!r}."
                )
            child = _record_fields(
                fields["dependent_child"],
                "dependent child rule",
                required=("max_age", "financial_independence"),
            )
            raw_test = child["financial_independence"]
            test = None
            if raw_test is not None:
                test_fields = _record_fields(
                    raw_test,
                    "financial independence test",
                    required=("min_age", "min_usual_weekly_hours"),
                )
                test = FinancialIndependenceTest(**test_fields)
            fields["dependent_child"] = DependentChildRule(
                max_age=child["max_age"], financial_independence=test
            )
            for name in ("entity", "unparented_child", "split_parents"):
                if not isinstance(fields[name], str):
                    raise ValueError(f"A benefit unit rule's {name!r} must be text.")
            if not isinstance(fields["provenance"], Mapping):
                raise ValueError(
                    "A benefit unit rule's 'provenance' must be an object."
                )
            return cls(**fields)


def build_benefit_units(
    person: pd.DataFrame,
    household: pd.DataFrame,
    rule: BenefitUnitRule,
) -> tuple[pd.DataFrame, pd.Series]:
    """Group a concept frame's persons into benefit units.

    Args:
        person: The concept frame's person table. It needs ``person_id``,
            ``person_household_id``, ``age`` and the three person pointers,
            plus ``usual_weekly_hours`` when the rule declares a
            financial-independence test.
        household: The concept frame's household table, with
            ``household_id`` and ``reference_person_id``.
        rule: The country's benefit-unit rule.

    Returns:
        The unit table, one row per unit sorted by id, with ``<entity>_id``,
        ``<entity>_household_id`` and ``<entity>_head_person_id`` (all
        ``int64``); and each person's unit id (``int64``), aligned to
        ``person``'s rows and index and named ``person_<entity>_id``.

    Raises:
        ValueError: If a needed column is missing, the frame breaks a
            concept-frame contract (pointers that dangle or leave the
            household, asymmetric partners, parent cycles, a reference person
            outside their household, bad ages or hours), or the rule refuses
            a child's placement.
    """

    rows = _Pointers.read(person, household, rule)
    dependent = _dependents(rows, rule)
    head = _heads(rows, dependent, rule)
    ids = rows.ids
    head_rows = np.unique(head)
    order = np.argsort(ids[head_rows], kind="stable")
    head_rows = head_rows[order]
    family = pd.DataFrame(
        {
            rule.id_column: ids[head_rows],
            rule.household_column: rows.household_ids[head_rows],
            rule.head_column: ids[head_rows],
        }
    )
    membership = pd.Series(
        ids[head], index=person.index, name=rule.membership_column, dtype=np.int64
    )
    return family, membership


def benefit_unit_roles(
    person: pd.DataFrame,
    family: pd.DataFrame,
    membership: pd.Series,
    rule: BenefitUnitRule,
) -> pd.Series:
    """Each person's role in their unit, read back from the built units.

    A unit is its head, the head's co-resident partner when the partner shares
    the unit, and dependent children; so the roles follow from the unit table
    and the partner pointers without re-running the rule.

    Returns:
        A string series aligned to ``person``, with values from
        :class:`~microcosm.frame.concept_mapping.UnitRole`.
    """

    ids = person[_PERSON_ID].to_numpy(dtype=np.int64)
    units = _unit_rows(family, membership, rule)
    head_of = family[rule.head_column].to_numpy(dtype=np.int64)[units]
    is_head = ids == head_of
    # Partner presence is a mask, not a sentinel id: any integer, -1
    # included, can be a person id.
    present = person["partner_person_id"].notna().to_numpy()
    partner = np.zeros(len(ids), dtype=np.int64)
    partner[present] = person["partner_person_id"][present].to_numpy(dtype=np.int64)
    head_row = pd.Index(ids).get_indexer(head_of)
    found = head_row >= 0
    head_row = np.where(found, head_row, 0)
    is_partner = ~is_head & found & present[head_row] & (partner[head_row] == ids)
    roles = np.where(
        is_head,
        UnitRole.HEAD.value,
        np.where(
            is_partner,
            UnitRole.PARTNER.value,
            UnitRole.DEPENDENT_CHILD.value,
        ),
    )
    return pd.Series(
        roles, index=person.index, name=f"{rule.entity}_role", dtype=object
    )


def benefit_unit_membership(
    person: pd.DataFrame,
    family: pd.DataFrame,
    membership: pd.Series,
    rule: BenefitUnitRule,
) -> GroupMembership:
    """The built units as :meth:`ConceptMapping.encode_groups` reads them."""

    return GroupMembership(
        entity=rule.entity,
        units=family,
        person_unit=membership,
        person_role=benefit_unit_roles(person, family, membership, rule),
    )


def benefit_unit_attributes(
    person: pd.DataFrame,
    family: pd.DataFrame,
    membership: pd.Series,
    rule: BenefitUnitRule,
) -> pd.DataFrame:
    """Composition attributes of each built unit.

    Returns:
        A table aligned to ``family``'s rows with ``<entity>_id`` and:

        - ``n_members``, ``n_adults`` and ``n_dependent_children`` (``int64``);
          the adults are the head and a partner who shares the unit;
        - ``is_couple``: the unit has two adults;
        - ``is_sole_parent``: one adult and at least one dependent child;
        - ``family_type``: ``single``, ``couple``, ``sole_parent`` or
          ``couple_with_children``;
        - ``youngest_dependent_child_age``: nullable ``Int64``, null when the
          unit has no dependent child.
    """

    roles = benefit_unit_roles(person, family, membership, rule).to_numpy()
    unit_rows = _unit_rows(family, membership, rule)
    n_units = len(family)
    is_adult = roles != UnitRole.DEPENDENT_CHILD.value
    n_members = np.bincount(unit_rows, minlength=n_units).astype(np.int64)
    n_adults = np.bincount(unit_rows[is_adult], minlength=n_units).astype(np.int64)
    n_children = n_members - n_adults
    ages = person["age"].to_numpy(dtype=np.float64)
    youngest = np.full(n_units, np.inf)
    child_rows = np.flatnonzero(~is_adult)
    np.minimum.at(youngest, unit_rows[child_rows], ages[child_rows])
    youngest_age = pd.array(
        [None if math.isinf(value) else int(value) for value in youngest],
        dtype="Int64",
    )
    is_couple = n_adults == 2
    with_children = n_children > 0
    family_type = np.where(
        is_couple,
        np.where(with_children, "couple_with_children", "couple"),
        np.where(with_children, "sole_parent", "single"),
    )
    return pd.DataFrame(
        {
            rule.id_column: family[rule.id_column].to_numpy(dtype=np.int64),
            "n_members": n_members,
            "n_adults": n_adults,
            "n_dependent_children": n_children,
            "is_couple": is_couple,
            "is_sole_parent": ~is_couple & with_children,
            "family_type": family_type.astype(object),
            "youngest_dependent_child_age": youngest_age,
        }
    )


def units_per_household(
    household: pd.DataFrame, family: pd.DataFrame, rule: BenefitUnitRule
) -> pd.Series:
    """How many units each household holds, aligned to ``household``'s rows."""

    counts = family[rule.household_column].value_counts()
    return pd.Series(
        counts.reindex(household[_HOUSEHOLD_ID].to_numpy(), fill_value=0)
        .to_numpy()
        .astype(np.int64),
        index=household.index,
        name=f"n_{rule.entity}_units",
    )


# --- Construction helpers --------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Pointers:
    """Pointer targets resolved to row positions (-1 when absent)."""

    ids: np.ndarray
    household_ids: np.ndarray
    age: np.ndarray
    hours: np.ndarray | None
    partner: np.ndarray
    parent_1: np.ndarray
    parent_2: np.ndarray
    reference: np.ndarray

    @classmethod
    def read(
        cls,
        person: pd.DataFrame,
        household: pd.DataFrame,
        rule: BenefitUnitRule,
    ) -> _Pointers:
        test = rule.dependent_child.financial_independence
        person_columns = [_PERSON_ID, _PERSON_HOUSEHOLD_ID, "age", *_POINTERS]
        if test is not None:
            person_columns.append("usual_weekly_hours")
        missing = [name for name in person_columns if name not in person.columns]
        missing += [
            f"household.{name}"
            for name in (_HOUSEHOLD_ID, "reference_person_id")
            if name not in household.columns
        ]
        if missing:
            raise ValueError(f"Building benefit units needs columns {missing}.")
        subset = {
            "person": person.loc[:, person_columns],
            "household": household.loc[:, [_HOUSEHOLD_ID, "reference_person_id"]],
        }
        violations = validate_concept_tables(subset)
        if violations:
            detail = "; ".join(
                f"{item.entity}.{item.column} {item.code}: {item.message}"
                + (f" ({item.rows} rows)" if item.rows else "")
                for item in violations
            )
            raise ValueError(f"Benefit units refuse an invalid concept frame: {detail}")
        ids = person[_PERSON_ID].to_numpy(dtype=np.int64)
        index = pd.Index(ids)

        def rows(values: pd.Series) -> np.ndarray:
            present = values.notna().to_numpy()
            found = np.full(len(values), -1, dtype=np.int64)
            if present.any():
                found[present] = index.get_indexer(
                    values[present].to_numpy(dtype=np.int64)
                )
            return found

        household_rows = pd.Index(
            household[_HOUSEHOLD_ID].to_numpy(dtype=np.int64)
        ).get_indexer(person[_PERSON_HOUSEHOLD_ID].to_numpy(dtype=np.int64))
        reference_by_household = rows(household["reference_person_id"])
        return cls(
            ids=ids,
            household_ids=person[_PERSON_HOUSEHOLD_ID].to_numpy(dtype=np.int64),
            age=person["age"].to_numpy(dtype=np.float64),
            hours=None
            if test is None
            else person["usual_weekly_hours"].to_numpy(dtype=np.float64),
            partner=rows(person["partner_person_id"]),
            parent_1=rows(person["parent_1_person_id"]),
            parent_2=rows(person["parent_2_person_id"]),
            reference=reference_by_household[household_rows],
        )


def _dependents(rows: _Pointers, rule: BenefitUnitRule) -> np.ndarray:
    """Which persons are dependent children under ``rule``."""

    n = len(rows.ids)
    named = np.concatenate(
        [rows.parent_1[rows.parent_1 >= 0], rows.parent_2[rows.parent_2 >= 0]]
    )
    is_parent = np.bincount(named, minlength=n) > 0
    child_rule = rule.dependent_child
    candidate = (rows.age <= child_rule.max_age) & (rows.partner < 0) & ~is_parent
    test = child_rule.financial_independence
    if test is not None:
        assert rows.hours is not None
        candidate &= ~(
            (rows.age >= test.min_age) & (rows.hours >= test.min_usual_weekly_hours)
        )
    has_parent = rows.parent_1 >= 0
    is_reference = rows.reference == np.arange(n)
    # A candidate with no co-resident parent who is the reference person
    # heads their own unit; every other unparented candidate needs placing.
    unparented = candidate & ~has_parent & ~is_reference
    placement = rule.unparented_child
    if placement is UnparentedChildPlacement.REFUSE and unparented.any():
        raise ValueError(
            f"The unit rule refuses {int(unparented.sum())} candidate "
            "child(ren) with no co-resident parent."
        )
    with_parent = candidate & has_parent
    if placement is UnparentedChildPlacement.REFERENCE_PERSON_UNIT:
        return with_parent | unparented
    return with_parent


def _heads(rows: _Pointers, dependent: np.ndarray, rule: BenefitUnitRule) -> np.ndarray:
    """Each person's unit head, as a person row."""

    n = len(rows.ids)
    own = np.arange(n)
    head = np.full(n, -1, dtype=np.int64)
    adult = ~dependent
    partner = rows.partner
    is_reference = rows.reference == own
    # A partnered person is never a candidate child, so both partners are
    # adults here; validation guarantees the pointers are symmetric.
    paired = adult & (partner >= 0)
    other = np.where(paired, partner, own)
    mine_first = (
        is_reference
        | (~is_reference[other] & (rows.age > rows.age[other]))
        | (
            ~is_reference[other]
            & (rows.age == rows.age[other])
            & (rows.ids <= rows.ids[other])
        )
    )
    head[adult] = np.where(mine_first, own, other)[adult]

    # A parent is never a candidate child, so every parent's head is known.
    with_parent = dependent & (rows.parent_1 >= 0)
    first = head[rows.parent_1[with_parent]]
    second_rows = rows.parent_2[with_parent]
    second = np.where(second_rows >= 0, head[np.maximum(second_rows, 0)], first)
    split = second != first
    if split.any() and rule.split_parents is SplitParentsPolicy.REFUSE:
        raise ValueError(
            f"{int(split.sum())} dependent child(ren) have co-resident parents "
            "in different units; the unit rule refuses the ambiguity."
        )
    head[with_parent] = first

    # Unparented dependent children join the unit of the household's
    # reference person, who is an adult or a dependent child with a parent,
    # so their head is already known.
    unparented = dependent & (rows.parent_1 < 0)
    head[unparented] = head[rows.reference[unparented]]
    if (head < 0).any():  # pragma: no cover - construction guarantees it
        raise AssertionError("A person was left without a benefit unit.")
    return head


def _unit_rows(
    family: pd.DataFrame, membership: pd.Series, rule: BenefitUnitRule
) -> np.ndarray:
    unit_ids = family[rule.id_column].to_numpy(dtype=np.int64)
    rows = pd.Index(unit_ids).get_indexer(membership.to_numpy(dtype=np.int64))
    if (rows < 0).any():
        raise ValueError("A person's unit is missing from the unit table.")
    return rows
