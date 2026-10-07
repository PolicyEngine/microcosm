"""Per-engine concept-to-input mappings and their coverage reports.

A :class:`ConceptMapping` says, for one rules engine, which engine inputs each
engine-neutral concept (:mod:`microcosm.frame.concepts`) feeds and how. It
is explicit: every binding names its engine input, the concept(s) it reads,
the transform between them and the relation that holds. Nothing matches
names. Every concept in the schema is either bound or listed as unmapped
with a reason, so a gap is always a stated decision.

Bindings on the engine's person and household entities execute here:
:meth:`ConceptMapping.encode` turns a concept frame into engine input
columns, and :meth:`ConceptMapping.decode` recovers every concept whose
bindings are invertible. Bindings on engine group entities (tax units,
benefit units, SPM units, families) need unit membership that a concept frame
does not carry; they declare a :class:`GroupRule`, and :meth:`encode`
reports them as deferred. :meth:`ConceptMapping.encode_groups` executes them
once a unit-construction step has built the units from the relationship
pointers (:mod:`microcosm.frame.unit_construction` builds benefit units) and
hands their :class:`GroupMembership` over. It also executes declared state
bindings (:class:`StateBinding`), which feed group inputs from model state
(receipt flags, an assigned area) rather than from concepts.

:func:`coverage_report` compares a mapping with an engine's input surface:
which inputs each concept feeds, which inputs no concept covers, and which
mapped inputs the engine does not have (always an error).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from microcosm.frame.concepts import (
    CONCEPT_BY_ID,
    CONCEPT_ENTITIES,
    CONCEPTS,
    HOUSEHOLD_ID_COLUMN,
    PERSON_HOUSEHOLD_ID_COLUMN,
    PERSON_ID_COLUMN,
    AlignmentRelation,
    CanonicalConceptKind,
    Concept,
    ConceptAlignment,
    TemporalBasis,
    Unit,
    _parsing,
    _record_fields,
    _text_fields,
    concept,
    concept_schema_sha256,
    derive_take_up_draws,
)
from microcosm.frame.schema import VariableMetadata

__all__ = [
    "AllocateToReferencePerson",
    "Allocation",
    "ConceptMappedEngine",
    "ConceptMapping",
    "CoresidentChildCount",
    "CoverageReport",
    "EncodedGroupInputs",
    "EncodedInputs",
    "Fraction",
    "GroupKnobs",
    "GroupMembership",
    "GroupRule",
    "Identity",
    "InputBinding",
    "InputDeclaration",
    "InputRef",
    "Scale",
    "Positive",
    "MAPPING_FORMAT",
    "Predicate",
    "Product",
    "Recode",
    "RelationshipRole",
    "Role",
    "ScaledSum",
    "Share",
    "StateBinding",
    "StateTest",
    "Sum",
    "TakeUpThreshold",
    "UnitComposition",
    "UnitFeature",
    "UnitRole",
    "bind",
    "coverage_report",
    "rules_engine_input_refs",
    "transform_from_dict",
    "transform_to_dict",
]

_PERSON_ID = PERSON_ID_COLUMN
_HOUSEHOLD_ID = HOUSEHOLD_ID_COLUMN
_PERSON_HOUSEHOLD_ID = PERSON_HOUSEHOLD_ID_COLUMN
_REFERENCE_PERSON = "fact:household.reference_person_id"


class InputDeclaration(StrEnum):
    """How an engine declares the inputs a mapping targets."""

    #: The engine declares each input's entity, dtype and period
    #: (PolicyEngine ``Variable`` classes). It has no structured, required
    #: field for the provision of law that defines an input; many variables
    #: carry optional free-form reference URLs.
    ENGINE_TYPED = "engine_typed"
    #: The input exists only because some compiled rule references a name it
    #: does not derive. It has no declared type and no defining provision
    #: (Axiom RuleSpec leaf inputs; TheAxiomFoundation/axiom-rules-engine#62).
    USAGE_INFERRED = "usage_inferred"


class GroupRule(StrEnum):
    """How a binding on an engine group entity collapses member values.

    :meth:`ConceptMapping.encode_groups` applies every rule except
    :attr:`PERSON_ROLE`, given the units' :class:`GroupMembership`.
    """

    #: Sum the members' values.
    SUM_OVER_MEMBERS = "sum_over_members"
    #: True when any member's value is true.
    ANY_MEMBER = "any_member"
    #: The value of the unit's reference member (its head or principal).
    REFERENCE_MEMBER = "reference_member"
    #: A household state, repeated on every unit in the household.
    HOUSEHOLD_VALUE = "household_value"
    #: A household amount, placed on the unit that contains the household's
    #: reference person; zero on the household's other units.
    ALLOCATE_TO_REFERENCE_UNIT = "allocate_to_reference_unit"
    #: The engine entity has one row per person in a role (Axiom's ``Child``);
    #: each row takes that person's own value.
    PERSON_ROLE = "person_role"


class Role(StrEnum):
    """A person's role, read from the relationship pointers."""

    #: The household's reference person.
    REFERENCE_PERSON = "reference_person"
    #: The co-resident partner of the household's reference person.
    REFERENCE_PERSON_PARTNER = "reference_person_partner"
    #: The reference person's partner, where the two are not married.
    UNMARRIED_PARTNER_OF_REFERENCE_PERSON = "unmarried_partner_of_reference_person"
    #: A parent of at least one co-resident person.
    PARENT_OF_CORESIDENT_CHILD = "parent_of_coresident_child"
    #: Has a co-resident spouse, civil partner or cohabiting partner.
    HAS_PARTNER = "has_partner"
    #: Has no co-resident partner.
    NO_PARTNER = "no_partner"


# --- Transforms -----------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    """The engine input holds the concept's value unchanged."""


@dataclass(frozen=True)
class Recode:
    """A category or boolean concept, recoded value by value.

    Attributes:
        pairs: ``(concept value, engine value)`` pairs, one per value in the
            concept's domain (``True`` and ``False`` for a boolean concept).
    """

    pairs: tuple[tuple[object, object], ...]

    def forward(self) -> dict[object, object]:
        """Concept value -> engine value."""
        return dict(self.pairs)

    @property
    def injective(self) -> bool:
        """Whether distinct concept values keep distinct engine values."""
        images = [engine for _, engine in self.pairs]
        return len(set(images)) == len(images)

    def inverse(self) -> dict[object, object]:
        """Engine value -> concept value; requires an injective recode."""
        if not self.injective:
            raise ValueError("A non-injective recode has no inverse.")
        return {engine: value for value, engine in self.pairs}


@dataclass(frozen=True)
class Share:
    """One of two complementary shares of an amount.

    Two ``Share`` bindings with the same parameter split one concept across
    two engine inputs, as a jurisdiction's law splits an amount the neutral
    layer does not (taxable and tax-exempt interest). The engine input gets
    ``value * s``, or ``value * (1 - s)`` for the complement, where the share
    ``s`` on [0, 1] is a country-pack parameter supplied at encode time. The
    pair sums back to the concept. Reads an annual flow only.
    """

    parameter: str
    complement: bool = False


@dataclass(frozen=True)
class Sum:
    """The engine input is the sum of several annual-flow amount concepts."""


@dataclass(frozen=True)
class ScaledSum:
    """The engine input is the sum of annual-flow amount concepts times a factor.

    A rule can state a total in another period than the concepts: the weekly
    sum of two annual home payments is ``(interest + principal) / 52``. One
    binding has one transform, so :class:`Sum` and :class:`Scale` cannot be
    chained; this is the two applied in that order. Not invertible.
    """

    factor: float

    def __post_init__(self) -> None:
        factor = float(self.factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"A scale factor must be finite and positive: {factor}.")
        object.__setattr__(self, "factor", factor)


@dataclass(frozen=True)
class Product:
    """The engine input is the product of two concepts (hours x weeks)."""


@dataclass(frozen=True)
class AllocateToReferencePerson:
    """A household amount, placed on the household's reference person.

    Every other member gets zero, so the household sum is preserved and the
    concept is recovered by summing over members.
    """


@dataclass(frozen=True)
class Predicate:
    """A boolean input: every clause's concept takes one of its values.

    Attributes:
        clauses: ``(concept id, allowed values)`` pairs, combined with AND.
    """

    clauses: tuple[tuple[str, tuple[object, ...]], ...]


@dataclass(frozen=True)
class RelationshipRole:
    """A boolean input flagging persons in ``role``.

    Attributes:
        role: The role flagged.
        max_child_age: For :attr:`Role.PARENT_OF_CORESIDENT_CHILD` only: count
            only children this age or younger, which makes the binding read
            age too.
    """

    role: Role
    max_child_age: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role(self.role))
        if (
            self.max_child_age is not None
            and self.role is not Role.PARENT_OF_CORESIDENT_CHILD
        ):
            raise ValueError("Only the parent role takes a child age limit.")

    @property
    def concepts(self) -> tuple[str, ...]:
        """The concepts the role reads."""
        base = _ROLE_CONCEPTS[self.role]
        return base if self.max_child_age is None else (*base, "fact:person.age")


@dataclass(frozen=True)
class Scale:
    """The engine input is the concept times a fixed factor.

    Converts an annual flow to the period a rule states it in: ``1/52``
    for a weekly amount, ``1/12`` for a monthly one. Invertible. Reads annual
    flows only: a stock has no weekly or monthly value.
    """

    factor: float

    def __post_init__(self) -> None:
        factor = float(self.factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"A scale factor must be finite and positive: {factor}.")
        object.__setattr__(self, "factor", factor)


@dataclass(frozen=True)
class Positive:
    """A boolean input: the numeric concept is greater than zero."""


@dataclass(frozen=True)
class Fraction:
    """A subset of an amount that another input already holds in full.

    Unlike a :class:`Share` pair, which splits an amount between two
    inputs, a fraction feeds an input that is part of an amount some other
    binding delivers whole (UK ISA interest inside savings interest). The
    input gets ``value * f`` for a country-pack parameter ``f`` on [0, 1].
    Reads an annual flow only.
    """

    parameter: str


@dataclass(frozen=True)
class CoresidentChildCount:
    """The number of co-resident persons who name this person as a parent.

    Attributes:
        max_age: Count only children this age or younger; ``None`` counts
            every co-resident child. A limit makes the binding read age too.
    """

    max_age: int | None = None

    @property
    def concepts(self) -> tuple[str, ...]:
        """The concepts the count reads."""
        if self.max_age is None:
            return _PARENT_CONCEPTS
        return (*_PARENT_CONCEPTS, "fact:person.age")


@dataclass(frozen=True)
class TakeUpThreshold:
    """A take-up flag: the program's draw falls below its rate.

    The draw is :func:`~microcosm.frame.concepts.derive_take_up_draws` of the
    persistent seed and :attr:`program`; the rate, aligned to the country's
    caseloads, is supplied at encode time.
    """

    program: str


class UnitRole(StrEnum):
    """A person's place in an engine unit built from the pointers."""

    #: The unit's reference member.
    HEAD = "head"
    #: The head's co-resident partner, in the head's unit.
    PARTNER = "partner"
    #: A dependent child of the unit's adults (or placed with them).
    DEPENDENT_CHILD = "dependent_child"


class UnitFeature(StrEnum):
    """A composition test on a built unit, read from its members' roles."""

    #: The unit has a partner besides its head.
    HAS_PARTNER = "has_partner"
    #: The unit has at least one dependent child.
    HAS_DEPENDENT_CHILD = "has_dependent_child"
    #: The unit has at least two dependent children.
    TWO_OR_MORE_DEPENDENT_CHILDREN = "two_or_more_dependent_children"
    #: The unit has one adult (no partner) and at least one dependent child.
    SOLE_PARENT = "sole_parent"


@dataclass(frozen=True)
class UnitComposition:
    """A boolean group input: the built unit passes a composition test.

    The unit is built from the relationship pointers, age and the household
    reference person under a declared rule (who counts as a dependent child
    is law, so it lives in the rule, not here; a rule with a
    financial-independence test also reads usual weekly hours). The value is
    the unit's own, the same for every member, so its binding takes the
    :attr:`GroupRule.REFERENCE_MEMBER` rule. It needs unit membership and runs
    only in :meth:`ConceptMapping.encode_groups`.
    """

    feature: UnitFeature

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature", UnitFeature(self.feature))

    @property
    def concepts(self) -> tuple[str, ...]:
        """The concepts unit construction reads."""
        return _UNIT_CONCEPTS


Transform = (
    Identity
    | Recode
    | Share
    | Fraction
    | Positive
    | Scale
    | Sum
    | Product
    | ScaledSum
    | AllocateToReferencePerson
    | Predicate
    | RelationshipRole
    | CoresidentChildCount
    | TakeUpThreshold
    | UnitComposition
)

#: Serialized transform kind -> transform class.
_TRANSFORM_KINDS: dict[str, type] = {
    "identity": Identity,
    "recode": Recode,
    "share": Share,
    "sum": Sum,
    "product": Product,
    "allocate_to_reference_person": AllocateToReferencePerson,
    "predicate": Predicate,
    "relationship_role": RelationshipRole,
    "coresident_child_count": CoresidentChildCount,
    "take_up_threshold": TakeUpThreshold,
    "fraction": Fraction,
    "positive": Positive,
    "scale": Scale,
    "scaled_sum": ScaledSum,
    "unit_composition": UnitComposition,
}
_KIND_BY_TRANSFORM: dict[type, str] = {
    cls: kind for kind, cls in _TRANSFORM_KINDS.items()
}

#: The serialized mapping format :meth:`ConceptMapping.to_dict` writes.
MAPPING_FORMAT = "microcosm.concept_mapping.v1"


def transform_to_dict(transform: Transform) -> dict[str, object]:
    """A JSON-ready form of ``transform``."""

    out: dict[str, object] = {"kind": _KIND_BY_TRANSFORM[type(transform)]}
    if isinstance(transform, Recode):
        out["pairs"] = [list(pair) for pair in transform.pairs]
    elif isinstance(transform, Share):
        out["parameter"] = transform.parameter
        out["complement"] = transform.complement
    elif isinstance(transform, Predicate):
        out["clauses"] = [
            [concept_id, list(values)] for concept_id, values in transform.clauses
        ]
    elif isinstance(transform, RelationshipRole):
        out["role"] = transform.role.value
        if transform.max_child_age is not None:
            out["max_child_age"] = transform.max_child_age
    elif isinstance(transform, Fraction):
        out["parameter"] = transform.parameter
    elif isinstance(transform, Scale | ScaledSum):
        out["factor"] = transform.factor
    elif isinstance(transform, UnitComposition):
        out["feature"] = transform.feature.value
    elif isinstance(transform, CoresidentChildCount) and transform.max_age is not None:
        out["max_age"] = transform.max_age
    elif isinstance(transform, TakeUpThreshold):
        out["program"] = transform.program
    return out


_TRANSFORM_FIELDS: dict[str, frozenset[str]] = {
    "recode": frozenset({"pairs"}),
    "share": frozenset({"parameter", "complement"}),
    "predicate": frozenset({"clauses"}),
    "relationship_role": frozenset({"role", "max_child_age"}),
    "fraction": frozenset({"parameter"}),
    "scale": frozenset({"factor"}),
    "scaled_sum": frozenset({"factor"}),
    "unit_composition": frozenset({"feature"}),
    "coresident_child_count": frozenset({"max_age"}),
    "take_up_threshold": frozenset({"program"}),
}


def transform_from_dict(data: Mapping[str, object]) -> Transform:
    """The transform ``data`` describes.

    Raises:
        ValueError: If ``data`` is not an object, the kind is unknown, or a
            field is missing, unexpected or of the wrong shape.
    """

    with _parsing("transform"):
        return _read_transform(data)


def _read_transform(data: object) -> Transform:
    if not isinstance(data, Mapping):
        raise ValueError(f"A transform must be an object, not {type(data).__name__}.")
    fields = dict(data)
    kind = fields.pop("kind", None)
    if kind not in _TRANSFORM_KINDS:
        raise ValueError(f"Unknown transform kind {kind!r}.")
    extra = set(fields) - _TRANSFORM_FIELDS.get(kind, frozenset())
    if extra:
        raise ValueError(f"Transform {kind!r} has unexpected fields {sorted(extra)}.")

    def required(name: str) -> object:
        if name not in fields:
            raise ValueError(f"Transform {kind!r} lacks {name!r}.")
        return fields[name]

    def text(name: str) -> str:
        value = required(name)
        if not isinstance(value, str):
            raise ValueError(f"Transform {kind!r} field {name!r} must be text.")
        return value

    def pairs(name: str) -> tuple[tuple[object, object], ...]:
        value = required(name)
        if not isinstance(value, list | tuple) or not all(
            isinstance(pair, list | tuple) and len(pair) == 2 for pair in value
        ):
            raise ValueError(f"Transform {kind!r} field {name!r} must list pairs.")
        return tuple((first, second) for first, second in value)

    def optional_age(name: str) -> int | None:
        value = fields.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(f"Transform {kind!r} field {name!r} must be an integer.")
        return value

    if kind == "recode":
        return Recode(pairs=pairs("pairs"))
    if kind == "share":
        complement = fields.get("complement", False)
        if not isinstance(complement, bool):
            raise ValueError("A share's complement must be true or false.")
        return Share(parameter=text("parameter"), complement=complement)
    if kind == "predicate":
        clauses = pairs("clauses")
        if not all(
            isinstance(concept_id, str) and isinstance(values, list | tuple)
            for concept_id, values in clauses
        ):
            raise ValueError(
                "Transform 'predicate' field 'clauses' must pair concept ids "
                "with value lists."
            )
        return Predicate(
            clauses=tuple((concept_id, tuple(values)) for concept_id, values in clauses)
        )
    if kind == "relationship_role":
        return RelationshipRole(
            role=Role(text("role")), max_child_age=optional_age("max_child_age")
        )
    if kind == "fraction":
        return Fraction(parameter=text("parameter"))
    if kind in ("scale", "scaled_sum"):
        factor = required("factor")
        if isinstance(factor, bool) or not isinstance(factor, int | float):
            raise ValueError("A scale factor must be a number.")
        return _TRANSFORM_KINDS[kind](factor=factor)
    if kind == "unit_composition":
        return UnitComposition(feature=UnitFeature(text("feature")))
    if kind == "coresident_child_count":
        return CoresidentChildCount(max_age=optional_age("max_age"))
    if kind == "take_up_threshold":
        return TakeUpThreshold(program=text("program"))
    return _TRANSFORM_KINDS[kind]()


_ROLE_CONCEPTS: dict[Role, tuple[str, ...]] = {
    Role.REFERENCE_PERSON: ("fact:household.reference_person_id",),
    Role.REFERENCE_PERSON_PARTNER: (
        "fact:household.reference_person_id",
        "fact:person.partner_person_id",
    ),
    Role.UNMARRIED_PARTNER_OF_REFERENCE_PERSON: (
        "fact:household.reference_person_id",
        "fact:person.partner_person_id",
        "fact:person.legal_marital_status",
    ),
    Role.PARENT_OF_CORESIDENT_CHILD: (
        "fact:person.parent_1_person_id",
        "fact:person.parent_2_person_id",
    ),
    Role.HAS_PARTNER: ("fact:person.partner_person_id",),
    Role.NO_PARTNER: ("fact:person.partner_person_id",),
}
_PARENT_CONCEPTS = ("fact:person.parent_1_person_id", "fact:person.parent_2_person_id")
#: What unit construction reads: the person pointers, age and the household
#: reference person (:mod:`microcosm.frame.unit_construction`).
_UNIT_CONCEPTS = (
    "fact:person.age",
    "fact:person.partner_person_id",
    *_PARENT_CONCEPTS,
    "fact:household.reference_person_id",
)


class InputRef(NamedTuple):
    """An engine input, scoped as precisely as the engine scopes it.

    PolicyEngine inputs are global names on one entity. Axiom inputs are
    scoped to a compiled RuleSpec module and an engine entity: the same name
    can be an input in one module and derived in another.
    """

    name: str
    entity: str
    module: str | None = None

    def label(self) -> str:
        """A readable one-line label."""
        if self.module is None:
            return f"{self.name} ({self.entity})"
        return f"{self.name} ({self.entity}, {self.module})"


@dataclass(frozen=True, kw_only=True)
class InputBinding:
    """How one engine input is fed from concepts.

    Attributes:
        engine_input: The engine input's name.
        engine_entity: The engine entity the input lives on.
        concepts: The concept ids the input's values are computed from.
            Allocation also reads the household reference person to place
            values; :attr:`reads` lists everything a binding reads.
        transform: How the input's values are computed from the concepts.
        relation: How the engine input relates to the concept(s), read from
            concept to input, in Chronicle's vocabulary.
        note: The evidence for the relation (the engine's own label or
            documentation, or how builds populate the input) and any
            difference in meaning. Required on every binding, which is
            stricter than Chronicle, where only exact alignments need
            evidence.
        group_rule: For an input on an engine group entity, how member values
            collapse onto the unit.
        module: For module-scoped engines (Axiom), the RuleSpec module whose
            compiled program takes the input, relative to its rulespec root.
        canonical_input: For Axiom, the engine's canonical request name,
            ``<jurisdiction>:<module target>#input.<name>``.
    """

    engine_input: str
    engine_entity: str
    concepts: tuple[str, ...]
    transform: Transform
    relation: AlignmentRelation
    note: str = ""
    group_rule: GroupRule | None = None
    module: str | None = None
    canonical_input: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", AlignmentRelation(self.relation))
        if self.group_rule is not None:
            object.__setattr__(self, "group_rule", GroupRule(self.group_rule))
        if not self.engine_input or not self.engine_entity:
            raise ValueError("A binding names its engine input and entity.")
        if self.relation is AlignmentRelation.SOURCE_LABEL:
            raise ValueError(
                f"Binding for {self.engine_input!r}: source_label asserts no "
                "semantic alignment and cannot bind a concept to an input."
            )
        if not self.concepts or len(set(self.concepts)) != len(self.concepts):
            raise ValueError(
                f"Binding for {self.engine_input!r} needs distinct concepts."
            )
        unknown = [item for item in self.concepts if item not in CONCEPT_BY_ID]
        if unknown:
            raise ValueError(
                f"Binding for {self.engine_input!r} reads Unknown concept(s) {unknown}."
            )
        if not self.note:
            raise ValueError(
                f"Binding for {self.engine_input!r} needs a note stating its "
                "evidence and any difference in meaning."
            )
        _check_transform_arity(self)

    @property
    def ref(self) -> InputRef:
        """The engine input this binding feeds."""
        return InputRef(self.engine_input, self.engine_entity, self.module)

    @property
    def reads(self) -> tuple[str, ...]:
        """Every concept the binding reads, including placement pointers.

        Allocation to the reference person, and allocation to the unit that
        contains that person, read the household reference person besides
        the amount; coverage reports and totality count that read.
        """
        places = isinstance(self.transform, AllocateToReferencePerson) or (
            self.group_rule is GroupRule.ALLOCATE_TO_REFERENCE_UNIT
        )
        if places and _REFERENCE_PERSON not in self.concepts:
            return (*self.concepts, _REFERENCE_PERSON)
        return self.concepts

    def legal_alignment(
        self, *, authority: str, legal_vintage: str
    ) -> ConceptAlignment | None:
        """This binding as a Chronicle-shaped alignment to law, if it is one.

        A binding whose engine input has a legal canonical id (every Axiom
        binding) and that reads a single concept asserts that the concept
        relates to that legal input as :attr:`relation` says, for the reason
        :attr:`note` gives. The mapping records neither who asserts it nor
        the legal vintage, so the caller supplies both. Returns None for a
        binding with no legal id or several concepts.
        """
        if self.canonical_input is None or len(self.concepts) != 1:
            return None
        alignment = ConceptAlignment(
            source_concept=self.concepts[0],
            canonical_concept=self.canonical_input,
            relation=self.relation,
            authority=authority,
            legal_vintage=legal_vintage,
            evidence_notes=self.note,
        )
        if alignment.canonical_kind is not CanonicalConceptKind.LEGAL:
            return None
        return alignment

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form, omitting absent optional fields."""
        out: dict[str, object] = {
            "engine_input": self.engine_input,
            "engine_entity": self.engine_entity,
            "concepts": list(self.concepts),
            "transform": transform_to_dict(self.transform),
            "relation": self.relation.value,
            "note": self.note,
        }
        if self.group_rule is not None:
            out["group_rule"] = self.group_rule.value
        if self.module is not None:
            out["module"] = self.module
        if self.canonical_input is not None:
            out["canonical_input"] = self.canonical_input
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> InputBinding:
        """The binding ``data`` describes (see :meth:`to_dict`).

        Raises:
            ValueError: If ``data`` is not an object, or a field is missing,
                unexpected or malformed.
        """
        with _parsing("binding"):
            fields = _record_fields(
                data,
                "binding",
                required=(
                    "engine_input",
                    "engine_entity",
                    "concepts",
                    "transform",
                    "relation",
                ),
                optional=("note", "group_rule", "module", "canonical_input"),
            )
            _text_fields(
                fields,
                "binding",
                (
                    "engine_input",
                    "engine_entity",
                    "relation",
                    "note",
                    "group_rule",
                    "module",
                    "canonical_input",
                ),
            )
            concepts = fields["concepts"]
            if not isinstance(concepts, list | tuple) or not all(
                isinstance(concept_id, str) for concept_id in concepts
            ):
                raise ValueError("A binding's 'concepts' must list concept ids.")
            fields["concepts"] = tuple(concepts)
            fields["transform"] = transform_from_dict(fields["transform"])
            return cls(**fields)

    @property
    def concept_entity(self) -> str:
        """The entity whose rows the transform produces values on."""
        if isinstance(self.transform, AllocateToReferencePerson):
            return "person"
        if isinstance(
            self.transform, RelationshipRole | CoresidentChildCount | UnitComposition
        ):
            return "person"
        entities = {concept(concept_id).entity for concept_id in self.concepts}
        if len(entities) != 1:
            raise ValueError(
                f"Binding for {self.engine_input!r} mixes concept entities."
            )
        return entities.pop()


def bind(
    engine_input: str,
    engine_entity: str,
    concepts: tuple[str, ...] | str,
    transform: Transform,
    relation: AlignmentRelation,
    note: str,
    *,
    group_rule: GroupRule | None = None,
) -> InputBinding:
    """A binding for a global-input engine, with one concept given bare."""
    return InputBinding(
        engine_input=engine_input,
        engine_entity=engine_entity,
        concepts=(concepts,) if isinstance(concepts, str) else concepts,
        transform=transform,
        relation=relation,
        note=note,
        group_rule=group_rule,
    )


#: Transforms that do arithmetic on a year's amount: a scale restates it per
#: week or month, a sum adds amounts (and a scaled sum scales the total), and
#: a share or fraction splits one. Each reads annual flows only, so none of
#: them takes a stock (a value at the reference date), a usual rate or a
#: persistent draw by accident; a binding that needs one needs a new transform.
#: The rule checks the concepts a
#: binding computes from (``concepts``), not the household reference person a
#: binding allocated to the reference unit also reads to place its value. A
#: product is outside the rule: it multiplies a usual rate (weekly hours) by
#: weeks.
_FLOW_TRANSFORMS = (Scale, Sum, ScaledSum, Share, Fraction)


def _check_transform_arity(binding: InputBinding) -> None:
    transform = binding.transform
    items = [concept(concept_id) for concept_id in binding.concepts]
    name = binding.engine_input
    single = (
        Identity,
        Recode,
        Share,
        Fraction,
        Positive,
        Scale,
        AllocateToReferencePerson,
        TakeUpThreshold,
    )
    if isinstance(transform, single) and len(items) != 1:
        raise ValueError(f"Binding for {name!r} must read exactly one concept.")
    if isinstance(transform, Identity) and items[0].unit is Unit.PERSON_ID:
        raise ValueError(
            f"Binding for {name!r}: person-id pointers are structure; they "
            "reach engines through roles, counts and engine unit membership."
        )
    if isinstance(transform, Recode):
        item = items[0]
        domain = item.domain if item.domain else (True, False)
        if item.dtype not in ("str", "bool"):
            raise ValueError(f"Binding for {name!r}: only categories recode.")
        values = [value for value, _ in transform.pairs]
        if sorted(map(repr, values)) != sorted(map(repr, domain)):
            raise ValueError(
                f"Binding for {name!r}: a recode must map exactly the "
                f"concept's domain {list(domain)}."
            )
    if isinstance(
        transform, Share | Fraction | Sum | ScaledSum | AllocateToReferencePerson
    ):
        for item in items:
            if item.monetary is None:
                raise ValueError(f"Binding for {name!r}: {item.id} is not an amount.")
    if isinstance(transform, Share | Fraction) and not transform.parameter:
        raise ValueError(f"Binding for {name!r}: a share names its parameter.")
    if isinstance(transform, Positive) and items[0].dtype not in ("int", "float"):
        raise ValueError(f"Binding for {name!r}: only numbers are compared.")
    if isinstance(transform, Scale) and items[0].dtype != "float":
        raise ValueError(
            f"Binding for {name!r}: only float concepts scale; a scaled count "
            "would not decode to whole numbers."
        )
    if isinstance(transform, Sum | ScaledSum) and len(items) < 2:
        raise ValueError(f"Binding for {name!r}: a sum reads two or more amounts.")
    if isinstance(transform, ScaledSum) and len({item.entity for item in items}) != 1:
        raise ValueError(f"Binding for {name!r}: a sum adds amounts on one entity.")
    if isinstance(transform, _FLOW_TRANSFORMS):
        for item in items:
            if item.temporal_basis is not TemporalBasis.ANNUAL_FLOW:
                raise ValueError(
                    f"Binding for {name!r}: a {_KIND_BY_TRANSFORM[type(transform)]} "
                    f"reads annual flows only, and {item.id} has temporal "
                    f"basis {item.temporal_basis.value}."
                )
    if isinstance(transform, Product) and len(items) != 2:
        raise ValueError(f"Binding for {name!r}: a product reads two concepts.")
    if isinstance(transform, AllocateToReferencePerson) and items[0].entity != (
        "household"
    ):
        raise ValueError(f"Binding for {name!r}: only household amounts allocate.")
    if isinstance(transform, Predicate):
        clause_ids = tuple(concept_id for concept_id, _ in transform.clauses)
        if clause_ids != binding.concepts:
            raise ValueError(
                f"Binding for {name!r}: a predicate's concepts are its clauses'."
            )
        for concept_id, allowed in transform.clauses:
            item = concept(concept_id)
            domain = item.domain if item.domain else (True, False)
            if item.dtype not in ("str", "bool") or not allowed:
                raise ValueError(
                    f"Binding for {name!r}: predicate clauses test categories."
                )
            if any(value not in domain for value in allowed):
                raise ValueError(
                    f"Binding for {name!r}: {concept_id} clause values must "
                    "come from its domain."
                )
    if isinstance(transform, RelationshipRole):
        if binding.concepts != transform.concepts:
            raise ValueError(
                f"Binding for {name!r}: role {transform.role.value} reads "
                f"{transform.concepts}."
            )
    if isinstance(transform, UnitComposition) and (
        binding.concepts != transform.concepts
    ):
        raise ValueError(
            f"Binding for {name!r}: a unit composition reads what unit "
            f"construction reads: {transform.concepts}."
        )
    if isinstance(transform, CoresidentChildCount) and (
        binding.concepts != transform.concepts
    ):
        raise ValueError(
            f"Binding for {name!r}: a child count reads both parents "
            f"(and age when limited): {transform.concepts}."
        )
    if isinstance(transform, TakeUpThreshold):
        if binding.concepts != ("fact:person.take_up_seed",):
            raise ValueError(f"Binding for {name!r}: take-up reads the seed.")
        if not transform.program:
            raise ValueError(f"Binding for {name!r}: take-up names its program.")


def _check_group_rule(engine: str, binding: InputBinding) -> None:
    rule = binding.group_rule
    if isinstance(binding.transform, UnitComposition):
        if rule is not GroupRule.REFERENCE_MEMBER:
            raise ValueError(
                f"{engine}: {binding.engine_input!r} is a unit's own composition; "
                "it takes the reference_member rule."
            )
        return
    items = [concept(concept_id) for concept_id in binding.concepts]
    household_only = all(item.entity == "household" for item in items)
    amounts = all(item.monetary is not None for item in items)
    if rule is GroupRule.HOUSEHOLD_VALUE and not (household_only and not amounts):
        raise ValueError(
            f"{engine}: {binding.engine_input!r} repeats household states only; "
            "a household amount repeated on every unit would be double-counted."
        )
    if rule is GroupRule.ALLOCATE_TO_REFERENCE_UNIT and not (
        household_only and amounts
    ):
        raise ValueError(
            f"{engine}: {binding.engine_input!r} allocates household amounts only."
        )
    if (
        rule
        in (
            GroupRule.SUM_OVER_MEMBERS,
            GroupRule.ANY_MEMBER,
            GroupRule.REFERENCE_MEMBER,
            GroupRule.PERSON_ROLE,
        )
        and binding.concept_entity != "person"
    ):
        raise ValueError(
            f"{engine}: {binding.engine_input!r} collapses person values only."
        )


@runtime_checkable
class ConceptMappedEngine(Protocol):
    """A rules-engine adapter that maps concepts onto its inputs.

    Deliberately separate from :class:`~microcosm.frame.rules.RulesEngine`:
    adding a method to that protocol would break every existing adapter and
    test double.
    """

    def concept_mapping(self) -> ConceptMapping:
        """Return the adapter's concept-to-input mapping."""
        ...


@dataclass(frozen=True, eq=False)
class EncodedInputs:
    """Engine input columns computed from a concept frame.

    Attributes:
        tables: ``person`` and ``household`` tables keyed by concept entity,
            each carrying the frame's id columns plus the engine inputs that
            live on the engine's person or household entity.
        deferred: Bindings on engine group entities, which need engine unit
            membership and were not executed here; group encoding
            (:meth:`ConceptMapping.encode_groups`) executes them on built
            units.
    """

    tables: Mapping[str, pd.DataFrame]
    deferred: tuple[InputBinding, ...]


class Allocation(StrEnum):
    """Where an allocated household amount goes among the household's units."""

    #: All of it on the unit that contains the household's reference person.
    REFERENCE_UNIT = "reference_unit"
    #: Shared among the household's units in proportion to their adults
    #: (heads and partners); every unit has at least one.
    PER_ADULT_SHARE = "per_adult_share"


@dataclass(frozen=True, eq=False)
class GroupMembership:
    """An engine group entity's units, and who belongs to them.

    A unit-construction step builds these from the relationship pointers
    (:func:`microcosm.frame.unit_construction.benefit_unit_membership`).
    :meth:`ConceptMapping.encode_groups` checks them against the concept
    frame before using them: membership aligned to the person index, integer
    ids, members in the household their unit declares, one head per unit and
    at most one partner, who is the head's partner by the pointers.

    Attributes:
        entity: The frame group entity the units are (``family``). The unit
            table carries ``<entity>_id``, ``<entity>_household_id`` and
            ``<entity>_head_person_id``.
        units: One row per unit.
        person_unit: Each person's unit id, aligned to the person table.
        person_role: Each person's :class:`UnitRole` value, aligned likewise.
    """

    entity: str
    units: pd.DataFrame
    person_unit: pd.Series
    person_role: pd.Series

    @property
    def id_column(self) -> str:
        """The unit table's id column."""
        return f"{self.entity}_id"

    @property
    def household_column(self) -> str:
        """The unit table's column naming the household each unit nests in."""
        return f"{self.entity}_household_id"

    @property
    def head_column(self) -> str:
        """The unit table's column naming each unit's head."""
        return f"{self.entity}_head_person_id"


@dataclass(frozen=True, kw_only=True)
class GroupKnobs:
    """Declared alternatives for how group inputs are encoded.

    Every knob names the engine inputs (or the state column) it changes, so
    a scenario is data: a country pack translates its own knob names (rent
    allocation, asset test, area column, rent factor) into these. A knob that
    names an input the call does not produce is refused.

    Attributes:
        allocation: Engine input -> where its household amount goes; inputs
            not named here go to the reference unit.
        zeroed: Numeric engine inputs set to zero (an asset test switched off
            by giving every unit no assets).
        factors: Numeric engine input -> a factor every unit's value is
            multiplied by (a stock adjustment to rents).
        state_columns: A state binding's declared column -> the column read
            instead, both ``<entity>.<column>`` (an alternative area
            assignment).
    """

    allocation: Mapping[str, Allocation] = field(default_factory=dict)
    zeroed: frozenset[str] = frozenset()
    factors: Mapping[str, float] = field(default_factory=dict)
    state_columns: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allocation",
            MappingProxyType(
                {name: Allocation(mode) for name, mode in self.allocation.items()}
            ),
        )
        object.__setattr__(self, "zeroed", frozenset(self.zeroed))
        factors = {}
        for name, factor in self.factors.items():
            if isinstance(factor, bool) or not isinstance(factor, int | float):
                raise ValueError(f"The factor for {name!r} must be a number.")
            if not math.isfinite(factor) or factor < 0:
                raise ValueError(
                    f"The factor for {name!r} must be finite and non-negative."
                )
            factors[name] = float(factor)
        object.__setattr__(self, "factors", MappingProxyType(factors))
        for declared, used in self.state_columns.items():
            for column in (declared, used):
                _state_column(column)
            if declared.split(".")[0] != used.split(".")[0]:
                raise ValueError(
                    f"State column {declared!r} cannot be read from another "
                    f"entity's {used!r}."
                )
        object.__setattr__(
            self, "state_columns", MappingProxyType(dict(self.state_columns))
        )
        overlap = self.zeroed & set(self.factors)
        if overlap:
            raise ValueError(f"Inputs {sorted(overlap)} are both zeroed and scaled.")

    def targets(self) -> frozenset[str]:
        """Every engine input a knob changes."""
        return frozenset(self.allocation) | self.zeroed | frozenset(self.factors)


class StateTest(StrEnum):
    """How a state binding reads its model-state columns."""

    #: True when any of the boolean columns is true.
    ANY_TRUE = "any_true"
    #: True when the one column equals the binding's value.
    EQUALS = "equals"


#: The group rules a state binding may take, by the state's entity.
_STATE_RULES = {
    "person": (GroupRule.ANY_MEMBER, GroupRule.REFERENCE_MEMBER),
    "household": (GroupRule.HOUSEHOLD_VALUE,),
}


@dataclass(frozen=True, kw_only=True)
class StateBinding:
    """A group engine input fed from model state rather than concepts.

    Model state is a column a graph node computed (a receipt flag assigned by
    take-up, an area assigned by geography), not a survey fact, so it has no
    concept and no place in a :class:`ConceptMapping`. The country pack
    declares these with their evidence, and
    :meth:`ConceptMapping.encode_groups` executes them alongside the
    mapping's group bindings: the test gives a boolean per person or
    household, the group rule collapses it onto the unit, and ``negate``
    flips the result (a unit with no recipient is a non-beneficiary unit).

    Attributes:
        engine_input: The engine input's name.
        engine_entity: The engine group entity the input lives on.
        columns: The state columns read, each ``<entity>.<column>`` on the
            frame's ``person`` or ``household`` table (one entity).
        test: How the columns give a boolean.
        group_rule: How the boolean collapses onto the unit.
        note: The evidence for the binding and any difference in meaning.
        value: The value :attr:`StateTest.EQUALS` compares with.
        negate: Whether the collapsed value is negated.
        module: For module-scoped engines, the RuleSpec module.
        canonical_input: For Axiom, the engine's canonical request name.
    """

    engine_input: str
    engine_entity: str
    columns: tuple[str, ...]
    test: StateTest
    group_rule: GroupRule
    note: str
    value: bool | int | str | None = None
    negate: bool = False
    module: str | None = None
    canonical_input: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "test", StateTest(self.test))
        object.__setattr__(self, "group_rule", GroupRule(self.group_rule))
        object.__setattr__(self, "columns", tuple(self.columns))
        name = self.engine_input
        if not name or not self.engine_entity:
            raise ValueError("A state binding names its engine input and entity.")
        if not self.note:
            raise ValueError(f"State binding for {name!r} needs a note.")
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError(f"State binding for {name!r} reads distinct columns.")
        entities = {_state_column(column)[0] for column in self.columns}
        if len(entities) != 1:
            raise ValueError(f"State binding for {name!r} reads columns on one entity.")
        entity = entities.pop()
        if self.group_rule not in _STATE_RULES[entity]:
            allowed = [rule.value for rule in _STATE_RULES[entity]]
            raise ValueError(
                f"State binding for {name!r}: {entity} state collapses by "
                f"{allowed}, not {self.group_rule.value!r}."
            )
        if self.test is StateTest.EQUALS:
            if len(self.columns) != 1 or self.value is None:
                raise ValueError(
                    f"State binding for {name!r}: equals compares one column "
                    "with a value."
                )
            if isinstance(self.value, float):
                raise ValueError(
                    f"State binding for {name!r} compares with an exact value."
                )
        elif self.value is not None:
            raise ValueError(f"State binding for {name!r}: any_true takes no value.")
        if not isinstance(self.negate, bool):
            raise ValueError(f"State binding for {name!r}: negate is true or false.")

    @property
    def ref(self) -> InputRef:
        """The engine input this binding feeds."""
        return InputRef(self.engine_input, self.engine_entity, self.module)

    @property
    def entity(self) -> str:
        """The frame entity whose state the binding reads."""
        return _state_column(self.columns[0])[0]

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form, omitting absent optional fields."""
        out: dict[str, object] = {
            "engine_input": self.engine_input,
            "engine_entity": self.engine_entity,
            "columns": list(self.columns),
            "test": self.test.value,
            "group_rule": self.group_rule.value,
            "note": self.note,
        }
        if self.value is not None:
            out["value"] = self.value
        if self.negate:
            out["negate"] = True
        if self.module is not None:
            out["module"] = self.module
        if self.canonical_input is not None:
            out["canonical_input"] = self.canonical_input
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StateBinding:
        """The state binding ``data`` describes (see :meth:`to_dict`).

        Raises:
            ValueError: If ``data`` is not an object, or a field is missing,
                unexpected or malformed.
        """
        with _parsing("state binding"):
            fields = _record_fields(
                data,
                "state binding",
                required=(
                    "engine_input",
                    "engine_entity",
                    "columns",
                    "test",
                    "group_rule",
                    "note",
                ),
                optional=("value", "negate", "module", "canonical_input"),
            )
            _text_fields(
                fields,
                "state binding",
                (
                    "engine_input",
                    "engine_entity",
                    "test",
                    "group_rule",
                    "note",
                    "module",
                    "canonical_input",
                ),
            )
            columns = fields["columns"]
            if not isinstance(columns, list | tuple) or not all(
                isinstance(column, str) for column in columns
            ):
                raise ValueError("A state binding's 'columns' must list columns.")
            fields["columns"] = tuple(columns)
            value = fields.get("value")
            if value is not None and not isinstance(value, bool | int | str):
                raise ValueError("A state binding's 'value' must be exact.")
            return cls(**fields)


@dataclass(frozen=True, eq=False)
class EncodedGroupInputs:
    """Engine group inputs computed from a concept frame and its units.

    Attributes:
        tables: One table per frame group entity encoded (``family``), row
            aligned to that membership's unit table, with the unit id column
            plus the engine inputs.
        deferred: Group bindings not executed: their engine entity has no
            membership here, or they take the per-person ``person_role`` rule.
    """

    tables: Mapping[str, pd.DataFrame]
    deferred: tuple[InputBinding, ...]


@dataclass(frozen=True, kw_only=True)
class ConceptMapping:
    """One engine's explicit concept-to-input mapping.

    Attributes:
        engine: The engine the mapping targets (``policyengine-us``,
            ``axiom:nz``).
        engine_version: The engine version or RuleSpec commit the mapping was
            reviewed against.
        entity_correspondence: Concept entity -> engine entity.
        input_declaration: How the engine declares its inputs.
        bindings: Every binding.
        unmapped: Concept id -> why no engine input takes it.
        structural_inputs: Engine inputs that carry structure (ids,
            membership, weights), not content; the coverage report lists them
            apart from uncovered inputs.
    """

    engine: str
    engine_version: str
    entity_correspondence: Mapping[str, str]
    input_declaration: InputDeclaration
    bindings: tuple[InputBinding, ...]
    unmapped: Mapping[str, str]
    structural_inputs: tuple[str, ...] = ()
    _by_concept: Mapping[str, tuple[InputBinding, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "input_declaration", InputDeclaration(self.input_declaration)
        )
        object.__setattr__(
            self,
            "entity_correspondence",
            MappingProxyType(dict(self.entity_correspondence)),
        )
        object.__setattr__(self, "unmapped", MappingProxyType(dict(self.unmapped)))
        if set(self.entity_correspondence) != set(CONCEPT_ENTITIES):
            raise ValueError(
                f"{self.engine}: entity_correspondence maps exactly "
                f"{list(CONCEPT_ENTITIES)}."
            )
        by_concept: dict[str, list[InputBinding]] = defaultdict(list)
        seen: set[InputRef] = set()
        for binding in self.bindings:
            if binding.ref in seen:
                raise ValueError(
                    f"{self.engine}: {binding.ref.label()} is bound twice."
                )
            seen.add(binding.ref)
            self._check_binding(binding)
            for concept_id in binding.reads:
                by_concept[concept_id].append(binding)
        for concept_id, reason in self.unmapped.items():
            concept(concept_id)
            if not reason:
                raise ValueError(
                    f"{self.engine}: unmapped {concept_id} needs a reason."
                )
            if concept_id in by_concept:
                raise ValueError(
                    f"{self.engine}: {concept_id} is both bound and unmapped."
                )
        missing = [
            item.id
            for item in CONCEPTS
            if item.id not in by_concept and item.id not in self.unmapped
        ]
        if missing:
            raise ValueError(
                f"{self.engine}: every concept must be bound or listed as "
                f"unmapped with a reason; missing {missing}."
            )
        self._check_shares()
        self._check_shared_names()
        object.__setattr__(
            self,
            "_by_concept",
            MappingProxyType({key: tuple(value) for key, value in by_concept.items()}),
        )

    def _check_binding(self, binding: InputBinding) -> None:
        person = self.entity_correspondence["person"]
        household = self.entity_correspondence["household"]
        is_axiom = self.input_declaration is InputDeclaration.USAGE_INFERRED
        if is_axiom and (binding.module is None or binding.canonical_input is None):
            raise ValueError(
                f"{self.engine}: {binding.engine_input!r} needs its RuleSpec "
                "module and canonical input name."
            )
        if binding.engine_input in self.structural_inputs:
            raise ValueError(
                f"{self.engine}: {binding.engine_input!r} is structural; no "
                "concept binds it."
            )
        if not is_axiom and (binding.module or binding.canonical_input):
            raise ValueError(
                f"{self.engine}: {binding.engine_input!r} is a global input; "
                "it has no module."
            )
        native = self.entity_correspondence[binding.concept_entity]
        on_group = binding.engine_entity not in (person, household)
        if on_group != (binding.group_rule is not None):
            raise ValueError(
                f"{self.engine}: {binding.engine_input!r} needs a group rule "
                "exactly when it lives on an engine group entity."
            )
        if not on_group and binding.engine_entity != native:
            raise ValueError(
                f"{self.engine}: {binding.engine_input!r} lives on "
                f"{binding.engine_entity!r} but its transform produces "
                f"{binding.concept_entity!r} values ({native!r})."
            )
        if on_group and isinstance(binding.transform, AllocateToReferencePerson):
            raise ValueError(
                f"{self.engine}: allocation targets the engine's person entity."
            )
        if not on_group and isinstance(binding.transform, UnitComposition):
            raise ValueError(
                f"{self.engine}: {binding.engine_input!r} tests a unit's "
                "composition, so it lives on an engine group entity."
            )
        if on_group:
            _check_group_rule(self.engine, binding)

    def _check_shared_names(self) -> None:
        """One encoded column per input name: every module computes it alike.

        Module-scoped engines can read the same input name in several
        modules. An encoded frame holds one column per name and entity, so
        every binding of that name must compute it the same way.
        """
        seen: dict[tuple[str, str], InputBinding] = {}
        for binding in self.bindings:
            key = (binding.engine_input, binding.engine_entity)
            first = seen.setdefault(key, binding)
            if (first.concepts, first.transform, first.group_rule) != (
                binding.concepts,
                binding.transform,
                binding.group_rule,
            ):
                raise ValueError(
                    f"{self.engine}: {binding.engine_input!r} on "
                    f"{binding.engine_entity!r} is computed differently in "
                    f"{first.module} and {binding.module}."
                )

    def _check_shares(self) -> None:
        groups: dict[tuple[str, str | None], list[InputBinding]] = defaultdict(list)
        fractions: set[str] = set()
        for binding in self.bindings:
            if isinstance(binding.transform, Share):
                groups[(binding.transform.parameter, binding.module)].append(binding)
            elif isinstance(binding.transform, Fraction):
                fractions.add(binding.transform.parameter)
        shared = fractions & {parameter for parameter, _ in groups}
        if shared:
            raise ValueError(
                f"{self.engine}: parameters {sorted(shared)} name both a share "
                "pair and a fraction."
            )
        by_parameter: dict[str, set[tuple[tuple[str, ...], str]]] = defaultdict(set)
        for (parameter, module), members in groups.items():
            complements = sorted(member.transform.complement for member in members)
            if complements != [False, True]:
                raise ValueError(
                    f"{self.engine}: share {parameter!r} must be one share and "
                    f"its complement in module {module}."
                )
            for member in members:
                by_parameter[parameter].add((member.concepts, member.engine_entity))
        for parameter, shapes in by_parameter.items():
            if len(shapes) != 1:
                raise ValueError(
                    f"{self.engine}: share {parameter!r} must split one concept "
                    "on one engine entity."
                )

    def __hash__(self) -> int:
        return hash((self.engine, self.engine_version, self.bindings))

    # --- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form that :meth:`from_dict` reads back unchanged."""
        return {
            "format": MAPPING_FORMAT,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "entity_correspondence": dict(self.entity_correspondence),
            "input_declaration": self.input_declaration.value,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "unmapped": dict(self.unmapped),
            "structural_inputs": list(self.structural_inputs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ConceptMapping:
        """The mapping ``data`` describes, validated like any other.

        Raises:
            ValueError: If the format is not :data:`MAPPING_FORMAT`, a field is
                missing, unexpected or malformed, or the mapping breaks a
                construction rule.
        """
        with _parsing("concept mapping"):
            fields = _record_fields(
                data,
                "concept mapping",
                required=(
                    "engine",
                    "engine_version",
                    "entity_correspondence",
                    "input_declaration",
                    "bindings",
                    "unmapped",
                ),
                optional=("format", "structural_inputs"),
            )
            if fields.pop("format", None) != MAPPING_FORMAT:
                raise ValueError(
                    f"A concept mapping must declare format {MAPPING_FORMAT!r}."
                )
            _text_fields(
                fields,
                "concept mapping",
                ("engine", "engine_version", "input_declaration"),
            )
            for name in ("entity_correspondence", "unmapped"):
                value = fields[name]
                if not isinstance(value, Mapping) or not all(
                    isinstance(key, str) and isinstance(text, str)
                    for key, text in value.items()
                ):
                    raise ValueError(
                        f"A concept mapping's {name!r} must map text to text."
                    )
            if not isinstance(fields["bindings"], list | tuple):
                raise ValueError("A concept mapping's 'bindings' must be a list.")
            fields["bindings"] = tuple(
                InputBinding.from_dict(binding) for binding in fields["bindings"]
            )
            structural = fields.get("structural_inputs", ())
            if not isinstance(structural, list | tuple) or not all(
                isinstance(name, str) for name in structural
            ):
                raise ValueError(
                    "A concept mapping's 'structural_inputs' must list input names."
                )
            fields["structural_inputs"] = tuple(structural)
            return cls(**fields)

    # --- Queries -----------------------------------------------------------

    def bindings_for(self, concept_id: str) -> tuple[InputBinding, ...]:
        """The bindings that read ``concept_id``."""
        concept(concept_id)
        return self._by_concept.get(concept_id, ())

    def inputs_for(self, concept_id: str) -> tuple[InputRef, ...]:
        """The engine inputs ``concept_id`` feeds."""
        return tuple(binding.ref for binding in self.bindings_for(concept_id))

    def refs(self) -> tuple[InputRef, ...]:
        """Every engine input the mapping feeds."""
        return tuple(binding.ref for binding in self.bindings)

    def share_parameters(self) -> tuple[str, ...]:
        """The share and fraction parameters encoding needs."""
        return tuple(
            sorted(
                {
                    binding.transform.parameter
                    for binding in self.bindings
                    if isinstance(binding.transform, Share | Fraction)
                }
            )
        )

    def take_up_programs(self) -> tuple[str, ...]:
        """The take-up program keys encoding needs a rate for."""
        return tuple(
            sorted(
                {
                    binding.transform.program
                    for binding in self.bindings
                    if isinstance(binding.transform, TakeUpThreshold)
                }
            )
        )

    def legal_alignments(
        self, *, authority: str, legal_vintage: str
    ) -> tuple[ConceptAlignment, ...]:
        """Every single-concept binding to a legal input, as an alignment.

        For an Axiom mapping these are the fact-to-law alignments the
        mapping already asserts: concept, the engine's canonical legal id,
        relation and evidence. The caller supplies the authority and the
        legal vintage (for example the mapping's pinned RuleSpec commit).
        Bindings of one name in several modules give one alignment per
        canonical id.
        """
        out: dict[tuple[str, str], ConceptAlignment] = {}
        for binding in self.bindings:
            alignment = binding.legal_alignment(
                authority=authority, legal_vintage=legal_vintage
            )
            if alignment is not None:
                key = (alignment.source_concept, alignment.canonical_concept)
                out.setdefault(key, alignment)
        return tuple(out.values())

    def is_executable(self, binding: InputBinding) -> bool:
        """Whether :meth:`encode` computes the binding (it is not deferred)."""
        return binding.group_rule is None

    def invertible_concepts(self) -> tuple[str, ...]:
        """Concepts :meth:`decode` recovers exactly from the engine inputs.

        A concept is invertible when an executable binding holds it unchanged
        (identity), scales it by a fixed factor, recodes it injectively,
        allocates it to the reference person, or splits it into a complete
        share pair; the household reference person is recovered from a
        reference-person role flag. Shares and scaled amounts recover to
        float rounding; everything else recovers exactly.
        """

        out: list[str] = []
        for item in CONCEPTS:
            if self._decoder(item) is not None:
                out.append(item.id)
        return tuple(out)

    def _decoder(self, item: Concept) -> tuple[InputBinding, ...] | None:
        # Only a binding computed from the concept can invert it; a binding
        # that merely reads it to place values (allocation reads the
        # reference person) cannot.
        candidates = [
            binding
            for binding in self.bindings_for(item.id)
            if self.is_executable(binding) and item.id in binding.concepts
        ]
        for binding in candidates:
            transform = binding.transform
            if isinstance(transform, Identity | AllocateToReferencePerson | Scale):
                return (binding,)
            if isinstance(transform, Recode) and transform.injective:
                return (binding,)
            if (
                isinstance(transform, RelationshipRole)
                and transform.role is Role.REFERENCE_PERSON
                and item.id == "fact:household.reference_person_id"
            ):
                return (binding,)
        shares: dict[tuple[str, str | None], list[InputBinding]] = defaultdict(list)
        for binding in candidates:
            if isinstance(binding.transform, Share):
                shares[(binding.transform.parameter, binding.module)].append(binding)
        for members in shares.values():
            if len(members) == 2:
                return tuple(members)
        return None

    # --- Execution ---------------------------------------------------------

    def encode(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        shares: Mapping[str, float] | None = None,
        take_up_rates: Mapping[str, float | None] | None = None,
    ) -> EncodedInputs:
        """Compute the engine inputs a concept frame supplies.

        Every executable binding whose concepts are all present in ``tables``
        is computed. Bindings on engine group entities are returned as
        deferred (:meth:`encode_groups` executes them on built units). Share
        parameters and take-up rates are needed only for the
        bindings that use them. A take-up rate of ``None`` marks a program
        the caller leaves unseeded: its flag is not written, so the engine
        applies its own default.

        Raises:
            ValueError: If a needed share or rate is missing or outside
                [0, 1], or a relationship role or allocation lacks the
                pointers it reads.
        """

        shares = dict(shares or {})
        rates = dict(take_up_rates or {})
        person = tables["person"]
        household = tables["household"]
        out = {
            "person": person.loc[:, [_PERSON_ID, _PERSON_HOUSEHOLD_ID]].copy(),
            "household": household.loc[:, [_HOUSEHOLD_ID]].copy(),
        }
        deferred: list[InputBinding] = []
        context = _Context(person, household)
        for binding in self.bindings:
            if not self.is_executable(binding):
                deferred.append(binding)
                continue
            if not all(context.has(concept_id) for concept_id in binding.concepts):
                continue
            if (
                isinstance(binding.transform, TakeUpThreshold)
                and binding.transform.program in rates
                and rates[binding.transform.program] is None
            ):
                continue
            values = _apply(binding, context, shares, rates)
            out[binding.concept_entity][binding.engine_input] = values
        return EncodedInputs(
            tables=MappingProxyType(out),
            deferred=tuple(deferred),
        )

    def encode_groups(
        self,
        tables: Mapping[str, pd.DataFrame],
        memberships: Mapping[str, GroupMembership],
        *,
        modules: Iterable[str] | None = None,
        knobs: GroupKnobs | None = None,
        state_bindings: Iterable[StateBinding] = (),
        shares: Mapping[str, float] | None = None,
        take_up_rates: Mapping[str, float] | None = None,
    ) -> EncodedGroupInputs:
        """Compute the engine group inputs a concept frame and its units supply.

        Every group binding whose engine entity has a membership, whose
        module passes the filter and whose concepts are all present in
        ``tables`` is executed by its :class:`GroupRule`:

        - ``sum_over_members``: the members' values summed;
        - ``any_member``: true when any member's value is;
        - ``reference_member``: the unit head's value (a
          :class:`UnitComposition` is the unit's own);
        - ``household_value``: the household's value on each of its units;
        - ``allocate_to_reference_unit``: the household amount on the unit
          that contains the household's reference person and zero on its
          other units, or shared per adult when ``knobs`` says so; either
          way each household's units sum back to its amount.

        ``person_role`` bindings, and bindings on an engine entity with no
        membership, are returned as deferred. Each state binding is executed
        the same way from its model-state columns. ``knobs`` then zero or
        scale named inputs.

        Args:
            tables: The concept frame's ``person`` and ``household`` tables,
                carrying any model-state columns the state bindings read.
            memberships: Engine group entity (``Family``) -> its units.
            modules: The RuleSpec modules to encode for; ``None`` encodes all.
            knobs: Declared encoding alternatives; ``None`` is the default.
            state_bindings: Group inputs fed from model state.
            shares: Share and fraction parameters, as for :meth:`encode`.
            take_up_rates: Take-up rates, as for :meth:`encode`.

        Raises:
            ValueError: If a membership does not match the frame (a person
                outside every unit, a unit without exactly one head, members
                in two households), a state binding feeds an input a concept
                binding or another state binding also feeds, a knob names an
                input this call does not produce or cannot change, or a value
                a transform needs is missing.
        """

        knobs = knobs or GroupKnobs()
        selected = None if modules is None else frozenset(modules)
        shares = dict(shares or {})
        rates = dict(take_up_rates or {})
        person = tables["person"]
        household = tables["household"]
        context = _Context(person, household)
        units: dict[str, _Units] = {}
        for entity, membership in memberships.items():
            if not isinstance(membership, GroupMembership):
                raise ValueError(f"The {entity!r} membership is not a GroupMembership.")
            if any(
                group.membership.entity == membership.entity for group in units.values()
            ):
                raise ValueError(
                    f"Two engine entities name the frame entity {membership.entity!r}."
                )
            units[entity] = _Units(membership, context)
        out = {
            group.membership.entity: group.membership.units.loc[
                :, [group.membership.id_column]
            ].reset_index(drop=True)
            for group in units.values()
        }
        # One encoded column per (frame entity, name): an input name shared by
        # several modules is computed once, as _check_shared_names requires.
        produced: dict[str, set[str]] = defaultdict(set)
        allocated: set[str] = set()
        deferred: list[InputBinding] = []
        concept_names = {
            (binding.engine_input, binding.engine_entity)
            for binding in self.bindings
            if binding.group_rule is not None
        }

        def keep(module: str | None) -> bool:
            return selected is None or module in selected

        def write(entity: str, name: str, values: np.ndarray) -> None:
            frame_entity = units[entity].membership.entity
            out[frame_entity][name] = values
            produced[name].add(frame_entity)

        for binding in self.bindings:
            if binding.group_rule is None or not keep(binding.module):
                continue
            group = units.get(binding.engine_entity)
            if group is None or binding.group_rule is GroupRule.PERSON_ROLE:
                deferred.append(binding)
                continue
            if not all(context.has(concept_id) for concept_id in binding.concepts):
                continue
            if (
                isinstance(binding.transform, TakeUpThreshold)
                and binding.transform.program in rates
                and rates[binding.transform.program] is None
            ):
                continue
            if binding.group_rule is GroupRule.ALLOCATE_TO_REFERENCE_UNIT:
                allocated.add(binding.engine_input)
            mode = knobs.allocation.get(binding.engine_input, Allocation.REFERENCE_UNIT)
            write(
                binding.engine_entity,
                binding.engine_input,
                _group_values(binding, context, group, mode, shares, rates),
            )

        seen_state: dict[tuple[str, str], tuple[InputRef, dict[str, object]]] = {}
        for state in state_bindings:
            if not keep(state.module):
                continue
            key = (state.engine_input, state.engine_entity)
            if key in concept_names:
                raise ValueError(
                    f"{state.ref.label()} is bound by a concept and by state "
                    "(one encoded column per input name)."
                )
            signature = {
                name: value
                for name, value in state.to_dict().items()
                if name not in ("module", "canonical_input")
            }
            if key in seen_state:
                first, first_signature = seen_state[key]
                if first == state.ref:
                    raise ValueError(f"{state.ref.label()} is bound twice by state.")
                if first_signature != signature:
                    raise ValueError(
                        f"{state.engine_input!r} on {state.engine_entity!r} is "
                        f"computed differently in {first.module} and {state.module}."
                    )
                continue
            seen_state[key] = (state.ref, signature)
            group = units.get(state.engine_entity)
            if group is None:
                raise ValueError(
                    f"State binding {state.engine_input!r} needs a "
                    f"{state.engine_entity!r} membership."
                )
            write(
                state.engine_entity,
                state.engine_input,
                _state_values(state, context, group, knobs.state_columns),
            )

        unknown = sorted(knobs.targets() - set(produced))
        if unknown:
            raise ValueError(
                f"Knobs name inputs this encoding did not produce: {unknown}."
            )
        ambiguous = sorted(name for name in knobs.targets() if len(produced[name]) > 1)
        if ambiguous:
            raise ValueError(
                f"Knob targets {ambiguous} live on several group entities."
            )
        not_allocated = sorted(set(knobs.allocation) - allocated)
        if not_allocated:
            raise ValueError(
                f"{not_allocated} are not allocated; no allocation knob applies to them."
            )
        read = {
            column
            for state in state_bindings
            if keep(state.module)
            for column in state.columns
        }
        stray = sorted(set(knobs.state_columns) - read)
        if stray:
            raise ValueError(f"Knobs replace state columns no binding reads: {stray}.")
        for name in sorted(knobs.zeroed | set(knobs.factors)):
            (frame_entity,) = produced[name]
            table = out[frame_entity]
            values = table[name].to_numpy()
            if values.dtype.kind not in "iuf":
                raise ValueError(f"Knob target {name!r} is not numeric.")
            if name in knobs.zeroed:
                table[name] = np.zeros(len(values), dtype=values.dtype)
            else:
                table[name] = values.astype(np.float64) * knobs.factors[name]
        return EncodedGroupInputs(
            tables=MappingProxyType(out),
            deferred=tuple(deferred),
        )

    def decode(self, tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Recover every invertible concept present in engine input tables.

        ``tables`` has the shape :meth:`encode` returns. A concept is
        recovered when all the engine inputs of its inverting binding(s) are
        present.
        """

        person = tables["person"]
        household = tables["household"]
        out = {
            "person": person.loc[:, [_PERSON_ID, _PERSON_HOUSEHOLD_ID]].copy(),
            "household": household.loc[:, [_HOUSEHOLD_ID]].copy(),
        }
        for item in CONCEPTS:
            decoder = self._decoder(item)
            if decoder is None:
                continue
            sources = {
                binding.engine_input: tables[binding.concept_entity]
                for binding in decoder
            }
            if any(name not in table.columns for name, table in sources.items()):
                continue
            out[item.entity][item.name] = _invert(
                item, decoder, person, household, tables
            )
        return out


# --- Execution helpers -----------------------------------------------------


class _Context:
    """A concept frame, with pointer lookups done by exact integer ids.

    Pointers are matched through row positions, never through float
    arrays, so person ids beyond 2**53 resolve exactly.
    """

    def __init__(self, person: pd.DataFrame, household: pd.DataFrame) -> None:
        self.person = person
        self.household = household
        self._person_ids = pd.Index(person[_PERSON_ID].to_numpy())
        self._household_rows = pd.Index(
            household[_HOUSEHOLD_ID].to_numpy()
        ).get_indexer(person[_PERSON_HOUSEHOLD_ID].to_numpy())

    def has(self, concept_id: str) -> bool:
        item = CONCEPT_BY_ID[concept_id]
        return item.name in self.table(item.entity).columns

    def table(self, entity: str) -> pd.DataFrame:
        return self.person if entity == "person" else self.household

    def values(self, concept_id: str) -> pd.Series:
        item = CONCEPT_BY_ID[concept_id]
        return self.table(item.entity)[item.name]

    def person_rows(self, pointer: pd.Series) -> np.ndarray:
        """Each pointer's target person row; -1 when null or unknown."""
        present = pointer.notna().to_numpy()
        rows = np.full(len(pointer), -1, dtype=np.int64)
        if present.any():
            rows[present] = self._person_ids.get_indexer(
                pointer[present].to_numpy(dtype=np.int64)
            )
        return rows

    def household_amount_for_persons(self, concept_id: str) -> np.ndarray:
        """A household amount on each person's row."""
        amounts = self.values(concept_id).to_numpy(dtype=np.float64)
        return amounts[self._household_rows]

    def reference_rows(self) -> np.ndarray:
        """For each person, the row of their household's reference person."""
        _require(self, _REFERENCE_PERSON, "Placing values on the reference person")
        by_household = self.person_rows(self.household["reference_person_id"])
        return by_household[self._household_rows]


def _apply(
    binding: InputBinding,
    context: _Context,
    shares: Mapping[str, float],
    rates: Mapping[str, float],
) -> np.ndarray:
    transform = binding.transform
    concepts = binding.concepts
    if isinstance(transform, Identity):
        return context.values(concepts[0]).to_numpy(copy=True)
    if isinstance(transform, Recode):
        forward = transform.forward()
        values = context.values(concepts[0]).tolist()
        outside = sorted({repr(value) for value in values if value not in forward})
        if outside:
            raise ValueError(
                f"{concepts[0]} holds values outside its domain: {outside[:5]}."
            )
        return np.asarray(
            [forward[value] for value in values],
            dtype=object if isinstance(next(iter(forward.values())), str) else None,
        )
    if isinstance(transform, Share):
        share = _unit_interval(shares, transform.parameter, "share")
        factor = 1.0 - share if transform.complement else share
        return context.values(concepts[0]).to_numpy(dtype=np.float64) * factor
    if isinstance(transform, Sum):
        return np.sum(
            [
                context.values(concept_id).to_numpy(dtype=np.float64)
                for concept_id in concepts
            ],
            axis=0,
        )
    if isinstance(transform, ScaledSum):
        total = np.sum(
            [
                context.values(concept_id).to_numpy(dtype=np.float64)
                for concept_id in concepts
            ],
            axis=0,
        )
        return total * transform.factor
    if isinstance(transform, UnitComposition):
        raise ValueError(
            f"{binding.engine_input!r} tests unit composition; it needs unit "
            "membership (encode_groups)."
        )
    if isinstance(transform, Product):
        first, second = (
            context.values(concept_id).to_numpy(dtype=np.float64)
            for concept_id in concepts
        )
        return first * second
    if isinstance(transform, AllocateToReferencePerson):
        is_reference = context.reference_rows() == np.arange(len(context.person))
        amounts = context.household_amount_for_persons(concepts[0])
        return np.where(is_reference, amounts, 0.0)
    if isinstance(transform, Predicate):
        result = np.ones(len(context.table(binding.concept_entity)), dtype=bool)
        for concept_id, allowed in transform.clauses:
            result &= context.values(concept_id).isin(allowed).to_numpy()
        return result
    if isinstance(transform, RelationshipRole):
        return _role_flags(transform, context)
    if isinstance(transform, Positive):
        return context.values(concepts[0]).to_numpy(dtype=np.float64) > 0
    if isinstance(transform, Scale):
        return context.values(concepts[0]).to_numpy(dtype=np.float64) * transform.factor
    if isinstance(transform, Fraction):
        fraction = _unit_interval(shares, transform.parameter, "fraction")
        return context.values(concepts[0]).to_numpy(dtype=np.float64) * fraction
    if isinstance(transform, CoresidentChildCount):
        return _child_counts(context, transform.max_age)
    if isinstance(transform, TakeUpThreshold):
        rate = _unit_interval(rates, transform.program, "take-up rate")
        seeds = context.values(concepts[0]).to_numpy(dtype=np.float64)
        return derive_take_up_draws(seeds, transform.program) < rate
    raise TypeError(f"Unknown transform {transform!r}.")  # pragma: no cover


def _unit_interval(values: Mapping[str, float], key: str, what: str) -> float:
    if key not in values:
        raise ValueError(f"Encoding needs the {what} {key!r}.")
    value = float(values[key])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"The {what} {key!r} must lie on [0, 1], got {value}.")
    return value


def _require(context: _Context, concept_id: str, what: str) -> None:
    if not context.has(concept_id):
        raise ValueError(f"{what} needs the {concept_id} concept column.")


def _role_flags(transform: RelationshipRole, context: _Context) -> np.ndarray:
    role = transform.role
    for concept_id in transform.concepts:
        _require(context, concept_id, f"Role {role.value}")
    person = context.person
    rows = np.arange(len(person))
    if role is Role.PARENT_OF_CORESIDENT_CHILD:
        return _child_counts(context, transform.max_child_age) > 0
    if role in (Role.HAS_PARTNER, Role.NO_PARTNER):
        has = person["partner_person_id"].notna().to_numpy()
        return has if role is Role.HAS_PARTNER else ~has
    reference = context.reference_rows()
    if role is Role.REFERENCE_PERSON:
        return reference == rows
    partner = context.person_rows(person["partner_person_id"])
    reference_partner = np.where(reference >= 0, partner[np.maximum(reference, 0)], -1)
    flags = reference_partner == rows
    if role is Role.UNMARRIED_PARTNER_OF_REFERENCE_PERSON:
        married = person["legal_marital_status"].astype(object).eq("married")
        flags = flags & ~married.to_numpy(dtype=bool)
    return flags


def _child_counts(context: _Context, max_age: int | None) -> np.ndarray:
    person = context.person
    for concept_id in _PARENT_CONCEPTS:
        _require(context, concept_id, "A co-resident child count")
    counted = np.ones(len(person), dtype=bool)
    if max_age is not None:
        _require(context, "fact:person.age", "An age-limited child count")
        counted = person["age"].to_numpy(dtype=np.float64) <= max_age
    counts = np.zeros(len(person), dtype=np.int64)
    for column in ("parent_1_person_id", "parent_2_person_id"):
        parents = context.person_rows(person[column])
        named = parents[counted & (parents >= 0)]
        counts += np.bincount(named, minlength=len(person))
    return counts


def _state_column(column: str) -> tuple[str, str]:
    """``<entity>.<column>`` split, for a person or household column."""
    entity, _, name = column.partition(".") if isinstance(column, str) else ("", "", "")
    if entity not in CONCEPT_ENTITIES or not name or "." in name:
        raise ValueError(
            f"A state column is 'person.<column>' or 'household.<column>', "
            f"not {column!r}."
        )
    return entity, name


class _Units:
    """A membership checked against a concept frame, as row positions."""

    def __init__(self, membership: GroupMembership, context: _Context) -> None:
        self.membership = membership
        units = membership.units
        columns = (
            membership.id_column,
            membership.household_column,
            membership.head_column,
        )
        missing = [name for name in columns if name not in units.columns]
        if missing:
            raise ValueError(f"A {membership.entity} unit table lacks {missing}.")
        person = context.person
        for series in (membership.person_unit, membership.person_role):
            if not isinstance(series, pd.Series) or not series.index.equals(
                person.index
            ):
                raise ValueError(
                    f"{membership.entity} membership must align to the person "
                    "table's index."
                )
        if membership.person_unit.isna().any():
            raise ValueError(f"Every person needs a {membership.entity}.")
        for name, values in (
            ("person unit ids", membership.person_unit),
            *((column, units[column]) for column in columns),
        ):
            if values.dtype.kind not in "iu" and not pd.api.types.is_integer_dtype(
                values.dtype
            ):
                raise ValueError(
                    f"{membership.entity} {name} must be integers, not {values.dtype}."
                )
        ids = units[membership.id_column].to_numpy(dtype=np.int64)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{membership.entity} ids must be unique.")
        rows = pd.Index(ids).get_indexer(
            membership.person_unit.to_numpy(dtype=np.int64)
        )
        if (rows < 0).any():
            raise ValueError(
                f"{int((rows < 0).sum())} person(s) name an unknown "
                f"{membership.entity}."
            )
        roles = membership.person_role.astype(object).to_numpy()
        known = {role.value for role in UnitRole}
        if not all(role in known for role in roles.tolist()):
            raise ValueError(f"Unit roles come from {sorted(known)}.")
        n = len(units)
        if (np.bincount(rows, minlength=n) == 0).any():
            raise ValueError(f"Every {membership.entity} needs a member.")
        is_head = roles == UnitRole.HEAD.value
        if not (np.bincount(rows[is_head], minlength=n) == 1).all():
            raise ValueError(f"Every {membership.entity} needs exactly one head.")
        if (np.bincount(rows[roles == UnitRole.PARTNER.value], minlength=n) > 1).any():
            raise ValueError(f"A {membership.entity} has at most one partner.")
        head_rows = np.empty(n, dtype=np.int64)
        head_rows[rows[is_head]] = np.flatnonzero(is_head)
        person_ids = person[_PERSON_ID].to_numpy(dtype=np.int64)
        if not np.array_equal(
            person_ids[head_rows], units[membership.head_column].to_numpy(np.int64)
        ):
            raise ValueError(
                f"Each {membership.entity}'s head column must name its head member."
            )
        if "partner_person_id" in person.columns:
            partner = context.person_rows(person["partner_person_id"])
            head_of = head_rows[rows]
            is_partner = roles == UnitRole.PARTNER.value
            if (partner[is_partner] != head_of[is_partner]).any():
                raise ValueError(
                    f"A {membership.entity} partner must be its head's partner."
                )
            heads = np.flatnonzero(is_head)
            mate = partner[heads]
            together = mate >= 0
            together[together] = rows[mate[together]] == rows[heads[together]]
            if (roles[mate[together]] != UnitRole.PARTNER.value).any():
                raise ValueError(
                    f"A head's partner in the same {membership.entity} must be "
                    "its partner."
                )
        unit_households = units[membership.household_column].to_numpy(np.int64)
        person_households = person[_PERSON_HOUSEHOLD_ID].to_numpy(dtype=np.int64)
        if not np.array_equal(person_households, unit_households[rows]):
            raise ValueError(
                f"Every {membership.entity} must nest in its declared household."
            )
        household_rows = pd.Index(
            context.household[_HOUSEHOLD_ID].to_numpy()
        ).get_indexer(unit_households)
        if (household_rows < 0).any():
            raise ValueError(f"A {membership.entity} names an unknown household.")
        self.rows = rows
        self.roles = roles
        self.head_rows = head_rows
        self.household_rows = household_rows
        self.is_adult = roles != UnitRole.DEPENDENT_CHILD.value
        self.n = n

    def count(self, role: UnitRole) -> np.ndarray:
        """How many members of each unit hold ``role``."""
        return np.bincount(self.rows[self.roles == role.value], minlength=self.n)

    def collapse(self, rule: GroupRule, values: np.ndarray) -> np.ndarray:
        """Person values collapsed onto units by a person-reading rule."""
        if rule is GroupRule.REFERENCE_MEMBER:
            return values[self.head_rows]
        if rule is GroupRule.ANY_MEMBER:
            flags = np.zeros(self.n, dtype=bool)
            np.logical_or.at(flags, self.rows, np.asarray(values, dtype=bool))
            return flags
        if rule is GroupRule.SUM_OVER_MEMBERS:
            values = np.asarray(values)
            kind = np.int64 if values.dtype.kind in "biu" else np.float64
            totals = np.zeros(self.n, dtype=kind)
            np.add.at(totals, self.rows, values.astype(kind))
            return totals
        raise TypeError(
            f"Rule {rule} does not collapse person values."
        )  # pragma: no cover

    def reference_units(self, context: _Context) -> np.ndarray:
        """Whether each unit contains its household's reference person."""
        _require(context, _REFERENCE_PERSON, "Allocating to the reference unit")
        references = context.person_rows(context.household["reference_person_id"])
        households = self.household_rows
        reference = references[households]
        if (reference < 0).any():
            raise ValueError("A household with units has no reference person.")
        own = context.person[_PERSON_HOUSEHOLD_ID].to_numpy(dtype=np.int64)
        declared = context.household[_HOUSEHOLD_ID].to_numpy(dtype=np.int64)
        if not np.array_equal(own[reference], declared[households]):
            raise ValueError("A household's reference person must be its member.")
        return self.rows[reference] == np.arange(self.n)


def _group_values(
    binding: InputBinding,
    context: _Context,
    group: _Units,
    mode: Allocation,
    shares: Mapping[str, float],
    rates: Mapping[str, float],
) -> np.ndarray:
    rule = binding.group_rule
    transform = binding.transform
    if isinstance(transform, UnitComposition):
        partners = group.count(UnitRole.PARTNER)
        children = group.count(UnitRole.DEPENDENT_CHILD)
        feature = transform.feature
        if feature is UnitFeature.HAS_PARTNER:
            return partners > 0
        if feature is UnitFeature.HAS_DEPENDENT_CHILD:
            return children >= 1
        if feature is UnitFeature.TWO_OR_MORE_DEPENDENT_CHILDREN:
            return children >= 2
        return (partners == 0) & (children >= 1)
    values = _apply(binding, context, shares, rates)
    if rule is GroupRule.HOUSEHOLD_VALUE:
        return np.asarray(values)[group.household_rows]
    if rule is GroupRule.ALLOCATE_TO_REFERENCE_UNIT:
        amounts = np.asarray(values, dtype=np.float64)[group.household_rows]
        if mode is Allocation.REFERENCE_UNIT:
            return np.where(group.reference_units(context), amounts, 0.0)
        adults = np.bincount(
            group.rows, weights=group.is_adult.astype(np.float64), minlength=group.n
        )
        per_household = np.bincount(
            group.household_rows, weights=adults, minlength=len(context.household)
        )
        return amounts * adults / per_household[group.household_rows]
    return group.collapse(rule, np.asarray(values))


def _state_values(
    state: StateBinding,
    context: _Context,
    group: _Units,
    replaced: Mapping[str, str],
) -> np.ndarray:
    table = context.table(state.entity)
    columns = []
    for declared in state.columns:
        _, name = _state_column(replaced.get(declared, declared))
        if name not in table.columns:
            raise ValueError(
                f"State binding {state.engine_input!r} needs the {state.entity} "
                f"column {name!r}."
            )
        values = table[name]
        if values.isna().any():
            raise ValueError(f"State column {name!r} has missing values.")
        columns.append(values)
    if state.test is StateTest.ANY_TRUE:
        flags = np.zeros(len(table), dtype=bool)
        for values in columns:
            if values.dtype.kind != "b":
                raise ValueError(
                    f"State column {values.name!r} must be boolean for any_true."
                )
            flags |= values.to_numpy(dtype=bool)
    else:
        values = columns[0]
        kind = values.dtype.kind
        expected = (
            kind == "b"
            if isinstance(state.value, bool)
            else kind in "iu"
            if isinstance(state.value, int)
            else kind in "OUT" or isinstance(values.dtype, pd.StringDtype)
        )
        if not expected:
            raise ValueError(
                f"State column {values.name!r} ({values.dtype}) cannot equal "
                f"{state.value!r}."
            )
        flags = values.astype(object).eq(state.value).to_numpy(dtype=bool)
    if state.group_rule is GroupRule.HOUSEHOLD_VALUE:
        result = flags[group.household_rows]
    else:
        result = group.collapse(state.group_rule, flags)
    return ~result if state.negate else result


def _invert(
    item: Concept,
    decoder: tuple[InputBinding, ...],
    person: pd.DataFrame,
    household: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
) -> np.ndarray:
    first = decoder[0]
    transform = first.transform
    source = tables[first.concept_entity][first.engine_input]
    if isinstance(transform, Share):
        total = np.zeros(len(source), dtype=np.float64)
        for binding in decoder:
            total += tables[binding.concept_entity][binding.engine_input].to_numpy(
                dtype=np.float64
            )
        return total
    if isinstance(transform, Identity):
        return _as_concept_dtype(item, source.to_numpy())
    if isinstance(transform, Scale):
        return _as_concept_dtype(
            item, source.to_numpy(dtype=np.float64) / transform.factor
        )
    if isinstance(transform, Recode):
        inverse = transform.inverse()
        return _as_concept_dtype(
            item,
            np.asarray([inverse[value] for value in source.tolist()], dtype=object),
        )
    if isinstance(transform, AllocateToReferencePerson):
        sums = (
            pd.Series(source.to_numpy(dtype=np.float64))
            .groupby(person[_PERSON_HOUSEHOLD_ID].to_numpy())
            .sum()
        )
        return sums.reindex(household[_HOUSEHOLD_ID].to_numpy()).fillna(0.0).to_numpy()
    if isinstance(transform, RelationshipRole):
        flagged = person.loc[
            source.to_numpy(dtype=bool), [_PERSON_HOUSEHOLD_ID, _PERSON_ID]
        ]
        if not flagged[_PERSON_HOUSEHOLD_ID].is_unique:
            raise ValueError("A household has more than one reference person.")
        lookup = pd.Series(
            flagged[_PERSON_ID].to_numpy(),
            index=flagged[_PERSON_HOUSEHOLD_ID].to_numpy(),
        )
        households = household[_HOUSEHOLD_ID].to_numpy()
        if not pd.Index(households).isin(lookup.index).all():
            raise ValueError("A household has no reference person.")
        return lookup.reindex(households).astype("Int64").to_numpy()
    raise TypeError(
        f"{item.id} has no inverse through {transform!r}."
    )  # pragma: no cover


def _as_concept_dtype(item: Concept, values: np.ndarray) -> np.ndarray:
    if item.dtype == "int":
        numbers = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(numbers)) or not np.all(numbers == np.round(numbers)):
            raise ValueError(f"{item.id} decodes only from whole numbers.")
        return numbers.astype(np.int64)
    if item.dtype == "float":
        return np.asarray(values, dtype=np.float64)
    if item.dtype == "bool":
        flags = np.asarray(values, dtype=object)
        if not all(value in (True, False) for value in flags.tolist()):
            raise ValueError(f"{item.id} decodes only from true and false.")
        return flags.astype(bool)
    return np.asarray(values, dtype=object)


# --- Coverage --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CoverageReport:
    """How a mapping covers an engine's input surface.

    Attributes:
        engine: The mapping's engine.
        engine_version: The version the mapping was reviewed against.
        schema_sha256: :func:`~microcosm.frame.concepts.concept_schema_sha256`
            at report time.
        input_count: How many engine inputs the surface has.
        concept_inputs: Concept id -> the engine inputs it feeds (or reads to
            place values).
        covered_inputs: Surface inputs some concept feeds.
        unmapped_concepts: Concept id -> why no engine input takes it.
        uncovered_inputs: Engine inputs no concept feeds, less structure.
        structural_inputs: Engine inputs that carry structure.
        unknown_inputs: Mapped inputs missing from the surface (an error).
    """

    engine: str
    engine_version: str
    schema_sha256: str
    input_count: int
    concept_inputs: Mapping[str, tuple[InputRef, ...]]
    covered_inputs: tuple[InputRef, ...]
    unmapped_concepts: Mapping[str, str]
    uncovered_inputs: tuple[InputRef, ...]
    structural_inputs: tuple[InputRef, ...]
    unknown_inputs: tuple[InputRef, ...]

    @property
    def covered_count(self) -> int:
        """How many surface inputs some concept feeds.

        Covered, structural and uncovered inputs partition the surface, so
        the three counts sum to :attr:`input_count`.
        """
        return len(self.covered_inputs)

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form, stable under key sorting."""

        def refs(values: Iterable[InputRef]) -> list[dict[str, str]]:
            return [
                {
                    key: value
                    for key, value in ref._asdict().items()
                    if value is not None
                }
                for ref in values
            ]

        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "schema_sha256": self.schema_sha256,
            "input_count": self.input_count,
            "covered_count": self.covered_count,
            "concept_inputs": {
                key: refs(value) for key, value in self.concept_inputs.items()
            },
            "covered_inputs": refs(self.covered_inputs),
            "unmapped_concepts": dict(self.unmapped_concepts),
            "uncovered_inputs": refs(self.uncovered_inputs),
            "structural_inputs": refs(self.structural_inputs),
            "unknown_inputs": refs(self.unknown_inputs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CoverageReport:
        """Read a report :meth:`to_dict` wrote.

        Raises:
            ValueError: If ``data`` is not an object, a field is missing,
                unexpected or malformed, or the recorded covered count
                disagrees with the covered inputs listed.
        """
        with _parsing("coverage report"):
            return cls._read(data)

    @classmethod
    def _read(cls, data: object) -> CoverageReport:
        fields = _record_fields(
            data,
            "coverage report",
            required=(
                "engine",
                "engine_version",
                "schema_sha256",
                "input_count",
                "covered_count",
                "concept_inputs",
                "covered_inputs",
                "unmapped_concepts",
                "uncovered_inputs",
                "structural_inputs",
                "unknown_inputs",
            ),
        )
        _text_fields(
            fields, "coverage report", ("engine", "engine_version", "schema_sha256")
        )
        for name in ("input_count", "covered_count"):
            value = fields[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"A coverage report's {name!r} must be a non-negative integer."
                )

        def refs(name: str, values: object) -> tuple[InputRef, ...]:
            if not isinstance(values, list | tuple):
                raise ValueError(
                    f"A coverage report's {name!r} must list input references."
                )
            out = []
            for value in values:
                ref = _record_fields(
                    value,
                    "input reference",
                    required=("name", "entity"),
                    optional=("module",),
                )
                _text_fields(ref, "input reference", ("name", "entity", "module"))
                out.append(InputRef(**ref))
            return tuple(out)

        if not isinstance(fields["concept_inputs"], Mapping):
            raise ValueError("A coverage report's 'concept_inputs' must be an object.")
        unmapped = fields["unmapped_concepts"]
        if not isinstance(unmapped, Mapping) or not all(
            isinstance(key, str) and isinstance(reason, str)
            for key, reason in unmapped.items()
        ):
            raise ValueError(
                "A coverage report's 'unmapped_concepts' must map text to text."
            )
        report = cls(
            engine=fields["engine"],
            engine_version=fields["engine_version"],
            schema_sha256=fields["schema_sha256"],
            input_count=fields["input_count"],
            concept_inputs=MappingProxyType(
                {
                    key: refs("concept_inputs", value)
                    for key, value in fields["concept_inputs"].items()
                }
            ),
            covered_inputs=refs("covered_inputs", fields["covered_inputs"]),
            unmapped_concepts=MappingProxyType(dict(fields["unmapped_concepts"])),
            uncovered_inputs=refs("uncovered_inputs", fields["uncovered_inputs"]),
            structural_inputs=refs("structural_inputs", fields["structural_inputs"]),
            unknown_inputs=refs("unknown_inputs", fields["unknown_inputs"]),
        )
        if report.covered_count != fields["covered_count"]:
            raise ValueError(
                "A coverage report's covered count disagrees with its list."
            )
        return report

    def to_markdown(self) -> str:
        """A readable report."""
        lines = [
            f"# Concept coverage: {self.engine}",
            "",
            f"Engine version `{self.engine_version}`; concept schema "
            f"`{self.schema_sha256[:12]}`. Generated by "
            "`tools/refresh_concept_coverage.py`; do not edit by hand.",
            "",
            f"- Engine inputs: {self.input_count}",
            f"- Fed by a concept: {self.covered_count}",
            f"- Structural: {len(self.structural_inputs)}",
            f"- Uncovered: {len(self.uncovered_inputs)}",
            f"- Mapped but missing from the engine: {len(self.unknown_inputs)}",
            "",
            "## Inputs each concept feeds",
            "",
            "| Concept | Engine inputs |",
            "|---|---|",
        ]
        # Rows follow the schema's declaration order, whatever order the
        # report's mappings hold, so a report read back from sorted JSON
        # renders identically.
        order = [item.id for item in CONCEPTS]
        for concept_id in order:
            if concept_id not in self.concept_inputs:
                continue
            refs = self.concept_inputs[concept_id]
            cell = "<br>".join(f"`{ref.name}` ({ref.entity})" for ref in refs)
            lines.append(f"| `{concept_id}` | {cell} |")
        lines += ["", "## Concepts no engine input takes", ""]
        if self.unmapped_concepts:
            lines += ["| Concept | Reason |", "|---|---|"]
            lines += [
                f"| `{concept_id}` | {self.unmapped_concepts[concept_id]} |"
                for concept_id in order
                if concept_id in self.unmapped_concepts
            ]
        else:
            lines.append("None.")
        lines += ["", "## Inputs no concept covers", ""]
        by_entity: dict[str, list[InputRef]] = defaultdict(list)
        for ref in self.uncovered_inputs:
            by_entity[ref.entity if ref.module is None else f"{ref.module}"].append(ref)
        for group in sorted(by_entity):
            names = sorted(
                ref.name if ref.module is None else f"{ref.name} ({ref.entity})"
                for ref in by_entity[group]
            )
            lines += [
                f"### {group} ({len(names)})",
                "",
                ", ".join(f"`{n}`" for n in names),
                "",
            ]
        if self.unknown_inputs:
            lines += ["## Mapped inputs missing from the engine", ""]
            lines += [f"- `{ref.label()}`" for ref in self.unknown_inputs]
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


class _InputSurface(Protocol):
    def variables(self) -> Iterable[str]: ...

    def variable_metadata(self, name: str) -> VariableMetadata: ...


def rules_engine_input_refs(engine: _InputSurface) -> tuple[InputRef, ...]:
    """Every input a typed engine accepts, scoped by its entity.

    Works for any adapter (or import-free index) whose
    ``variable_metadata`` resolves input names; Axiom's does not, because its
    inputs are untyped, so Axiom surfaces come from
    :mod:`microcosm.frame.adapters.axiom_input_surface` instead.
    """

    return tuple(
        sorted(
            InputRef(name, engine.variable_metadata(name).entity)
            for name in engine.variables()
        )
    )


def coverage_report(
    mapping: ConceptMapping,
    engine_inputs: Iterable[InputRef],
) -> CoverageReport:
    """Compare ``mapping`` with the engine's input surface.

    Args:
        mapping: The engine's concept mapping.
        engine_inputs: Every input the engine accepts, each scoped as the
            engine scopes it (entity, and module for Axiom).
    """

    surface = tuple(sorted(set(engine_inputs)))
    surface_set = set(surface)
    structural_names = set(mapping.structural_inputs)
    concept_inputs = {
        item.id: tuple(sorted(mapping.inputs_for(item.id)))
        for item in CONCEPTS
        if mapping.bindings_for(item.id)
    }
    bound = set(mapping.refs())
    covered = tuple(ref for ref in surface if ref in bound)
    structural = tuple(
        ref for ref in surface if ref.name in structural_names and ref not in bound
    )
    uncovered = tuple(
        ref for ref in surface if ref not in bound and ref.name not in structural_names
    )
    unknown = tuple(sorted(bound - surface_set))
    return CoverageReport(
        engine=mapping.engine,
        engine_version=mapping.engine_version,
        schema_sha256=concept_schema_sha256(),
        input_count=len(surface),
        concept_inputs=MappingProxyType(concept_inputs),
        covered_inputs=covered,
        unmapped_concepts=MappingProxyType(dict(mapping.unmapped)),
        uncovered_inputs=uncovered,
        structural_inputs=structural,
        unknown_inputs=unknown,
    )
