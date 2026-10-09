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
does not carry; they declare a :class:`GroupRule` and are reported as
deferred rather than executed. No adapter applies group rules yet: that is
the work of a future unit-construction step that builds engine units from
the relationship pointers.

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
    _require_int64_ids,
    _text_fields,
    concept,
    concept_schema_sha256,
    derive_take_up_draws,
)
from microcosm.frame.schema import VariableMetadata

__all__ = [
    "AllocateToReferencePerson",
    "ConceptMappedEngine",
    "ConceptMapping",
    "CoresidentChildCount",
    "CoverageReport",
    "EncodedInputs",
    "Fraction",
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
    "Share",
    "Sum",
    "TakeUpThreshold",
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

    This module declares the rule; it does not apply it. Applying it needs
    engine unit membership, which no adapter builds from concepts yet.
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


Transform = (
    Identity
    | Recode
    | Share
    | Fraction
    | Positive
    | Scale
    | Sum
    | Product
    | AllocateToReferencePerson
    | Predicate
    | RelationshipRole
    | CoresidentChildCount
    | TakeUpThreshold
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
    elif isinstance(transform, Scale):
        out["factor"] = transform.factor
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
    if kind == "scale":
        factor = required("factor")
        if isinstance(factor, bool) or not isinstance(factor, int | float):
            raise ValueError("A scale factor must be a number.")
        return Scale(factor=factor)
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
        if isinstance(self.transform, RelationshipRole | CoresidentChildCount):
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
#: week or month, a sum adds amounts, and a share or fraction splits one. Each
#: reads annual flows only, so none of them takes a stock (a value at the
#: reference date), a usual rate or a persistent draw by accident; a binding
#: that needs one needs a new transform. The rule checks the concepts a
#: binding computes from (``concepts``), not the household reference person a
#: binding allocated to the reference unit also reads to place its value. A
#: product is outside the rule: it multiplies a usual rate (weekly hours) by
#: weeks.
_FLOW_TRANSFORMS = (Scale, Sum, Share, Fraction)


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
    if isinstance(transform, Share | Fraction | Sum | AllocateToReferencePerson):
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
    if isinstance(transform, Sum) and len(items) < 2:
        raise ValueError(f"Binding for {name!r}: a sum reads two or more amounts.")
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
            membership and were not executed.
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
        deferred. Share parameters and take-up rates are needed only for the
        bindings that use them. A take-up rate of ``None`` marks a program
        the caller leaves unseeded: its flag is not written, so the engine
        applies its own default.

        Raises:
            ValueError: If a needed share or rate is missing or outside
                [0, 1], a relationship role or allocation lacks the
                pointers it reads, or an unsigned id or pointer exceeds
                ``2**63 - 1`` (ids are matched as int64).
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
    arrays, so person ids beyond 2**53 resolve exactly. Ids are matched as
    int64, so an unsigned id above 2**63 - 1, which would wrap (2**64 - 1
    onto -1), is refused.
    """

    def __init__(self, person: pd.DataFrame, household: pd.DataFrame) -> None:
        _require_int64_ids(person, household, "Encoding matches")
        self.person = person
        self.household = household
        self._person_ids = pd.Index(_matched_ids(person[_PERSON_ID]))
        self._household_rows = pd.Index(
            _matched_ids(household[_HOUSEHOLD_ID])
        ).get_indexer(_matched_ids(person[_PERSON_HOUSEHOLD_ID]))

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


def _matched_ids(values: pd.Series) -> np.ndarray:
    """Ids ready for exact matching: null-free integers as int64, else as is.

    pandas matches against a narrower unsigned index by casting the targets
    down to it (261 onto 5 for uint8), so null-free integer ids are widened
    first; :func:`_require_int64_ids` has already refused any int64 cannot
    hold.
    """

    if values.dtype.kind in "iu" and not values.hasnans:
        return values.to_numpy(dtype=np.int64)
    return values.to_numpy()


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
