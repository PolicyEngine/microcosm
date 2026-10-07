"""Engine-neutral concept schema: the primitive facts Microcosm content holds.

Microcosm's content columns are policyengine-us input names today. That ties
every population to one rules engine and blocks *transport*: recalibrating a
shareable donor file (US ACS and CPS public-use households) to another
country's public targets, where that country's own rules engine computes its
taxes and benefits. This module defines the neutral layer that removes the
tie. It declares, once, the record-level facts a household survey observes
and that stay meaningful across countries: age, sex, co-resident
relationships, income by component, liquid financial assets, hours and weeks
worked, tenure and housing costs, disability, education and a persistent
take-up seed.

Each :class:`Concept` states what a value *is*: its entity, dtype, unit,
period semantics, currency and price-level handling, provenance class and
transport rule. Engines never appear here. Each rules-engine adapter maps
concepts to its own inputs with a :class:`~microcosm.frame.concept_mapping.
ConceptMapping`, so the same concept frame can feed policyengine-us,
policyengine-uk or an Axiom RuleSpec country.

Two kinds of canonical concept
------------------------------
The schema's concepts are *primitive facts*. Their ids have the form
``fact:<entity>.<name>`` (``fact:person.employment_income``); the ``fact``
namespace holds jurisdiction-neutral record-level content, including
Microcosm's own persistent generated state (the take-up seed). *Legal
concepts* are what a jurisdiction's law derives from primitives. They carry
Axiom RuleSpec ids of the form ``<jurisdiction>:<citation path>#<name>``
(``us:statutes/26/32/c/2#earned_income``). Chronicle feeds also use
statistical ids of the form ``<authority>.<name>``
(``census_pep.resident_population``). :func:`canonical_concept_kind` tells
the three apart by syntax alone.

Alignment
---------
A :class:`ConceptAlignment` is Chronicle's ``concept_alignment`` record: its
fields are ``chronicle.concepts.ConceptAlignment``'s (plus the feeds'
``concept_alignment_key``) and its relation vocabulary is Chronicle's
(:class:`AlignmentRelation`), and it accepts every record Chronicle writes,
so one record type serves both repositories. A concept carries the
alignments from itself to legal concepts, one or more per jurisdiction,
under stricter rules (evidence, authority and legal vintage required). This
module ships the record type and its validation; the per-jurisdiction legal
alignments are follow-on work and every concept's tuple is empty today.

Concept frames
--------------
A concept frame is a pair of tables, ``person`` and ``household``, using the
kernel's id conventions (``person_id``, ``person_household_id`` and
``household_id``). A concept column is named by the concept's short name on
its entity's table (``age`` on ``person``). Any other column is a
country-specific extension, which the neutral layer does not describe and
transport drops. :func:`validate_concept_tables` checks a concept frame
against every declared contract, including relationship-pointer integrity.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import blake2b, sha256
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame.schema import (  # the kernel's own vocabularies
    _DTYPE_KINDS,
    _PERIOD_SEMANTICS,
    EntitySchema,
)

__all__ = [
    "ATTAINMENT_DOMAIN",
    "CONCEPTS",
    "CONCEPT_BY_ID",
    "CONCEPT_ENTITIES",
    "CONCEPT_FRAME_SCHEMA",
    "CONCEPT_SCHEMA_VERSION",
    "ENROLLMENT_DOMAIN",
    "HOUSEHOLD_ID_COLUMN",
    "MARITAL_STATUS_DOMAIN",
    "PERSON_HOUSEHOLD_ID_COLUMN",
    "PERSON_ID_COLUMN",
    "RENTED_TENURES",
    "SEX_DOMAIN",
    "TENURE_DOMAIN",
    "AlignmentRelation",
    "CanonicalConceptKind",
    "Concept",
    "ConceptAlignment",
    "ConceptFrameDeclaration",
    "ConceptViolation",
    "ContentBasis",
    "IndexFamily",
    "MonetaryHandling",
    "ProvenanceClass",
    "TemporalBasis",
    "TransportRule",
    "Unit",
    "canonical_concept_kind",
    "concept",
    "concept_for_column",
    "concept_schema_sha256",
    "concepts_for_entity",
    "derive_take_up_draws",
    "split_for_transport",
    "validate_concept_tables",
]

#: Version of the concept schema's declared contracts. Bump it whenever a
#: concept is added, removed or changes any declared field, so artifacts that
#: pin :func:`concept_schema_sha256` can say which schema they were written
#: against.
CONCEPT_SCHEMA_VERSION = 2

#: The two entities concept content lives on. Engine group entities (tax
#: units, benefit units, SPM units, families) are engine constructs built
#: from relationships. Today the US operator in microcosm.frame.units builds
#: them from raw CPS roster columns and the UK adapter requires them already
#: present; building them from these concepts' pointers is future work.
CONCEPT_ENTITIES: tuple[str, ...] = ("person", "household")

#: A concept frame's entity structure, in the kernel's id conventions.
CONCEPT_FRAME_SCHEMA = EntitySchema(group_entities=("household",))
#: The id columns every concept frame carries.
PERSON_ID_COLUMN = CONCEPT_FRAME_SCHEMA.person_id_column
PERSON_HOUSEHOLD_ID_COLUMN = CONCEPT_FRAME_SCHEMA.membership_column("household")
HOUSEHOLD_ID_COLUMN = CONCEPT_FRAME_SCHEMA.id_column("household")
_PERSON_ID = PERSON_ID_COLUMN
_HOUSEHOLD_ID = HOUSEHOLD_ID_COLUMN
_PERSON_HOUSEHOLD_ID = PERSON_HOUSEHOLD_ID_COLUMN

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PRIMITIVE_ID = re.compile(r"^fact:(person|household)\.([a-z][a-z0-9_]*)$")
# Axiom RuleSpec ids: a jurisdiction code (``us``, ``uk``, ``nz``, or a
# subnational ``us-ca``), a citation path, and a ``#``-separated name that may
# itself be dotted (``#input.wages``).
_LEGAL_ID = re.compile(
    r"^[a-z]{2}(?:-[a-z0-9]+)*:[A-Za-z0-9_./-]+#[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$"
)
# Chronicle statistical ids: ``<authority>.<name>``, authority lowercase.
_STATISTICAL_ID = re.compile(r"^[a-z][a-z0-9_-]*\.[A-Za-z0-9_.:-]+$")


class Unit(StrEnum):
    """The unit a concept's values are expressed in.

    ``base_currency`` follows ``microcosm.build.monetary_targets``: amounts in
    whole units of the concept frame's declared currency, with no implicit
    scaling (never thousands, never cents).
    """

    YEARS = "years"
    BASE_CURRENCY = "base_currency"
    HOURS_PER_WEEK = "hours_per_week"
    WEEKS = "weeks"
    CATEGORY = "category"
    BOOLEAN = "boolean"
    PERSON_ID = "person_id"
    UNIT_INTERVAL = "unit_interval"


class TemporalBasis(StrEnum):
    """What a value measures over its period.

    ``annual_flow`` matches ``microcosm.build.monetary_targets``.
    """

    #: An amount or count accumulated over the reference year. Its period is
    #: ``year``.
    ANNUAL_FLOW = "annual_flow"
    #: The record's state at the survey reference date.
    REFERENCE_STATE = "reference_state"
    #: A usual intensity during the reference year's weeks of activity.
    USUAL_RATE = "usual_rate"
    #: Constant for the life of the record identity.
    PERSISTENT = "persistent"


class ProvenanceClass(StrEnum):
    """How a concept's values come to exist.

    This classifies the *concept*, not individual cells: a producer's item
    imputation of an observed concept is still ``observed`` class, flagged per
    value elsewhere. The vocabulary is deliberately distinct from Chronicle's
    fact-level ``provenance_class`` (administrative, census, model_output,
    survey_aggregate), which classifies aggregate sources.
    """

    #: The survey instrument asks for it directly.
    OBSERVED = "observed"
    #: A deterministic, jurisdiction-neutral function of observed items, such
    #: as relationship pointers built from a household roster.
    DERIVED = "derived"
    #: Produced by Microcosm's own randomness; never observed.
    GENERATED = "generated"


class TransportRule(StrEnum):
    """What transport does with a concept column.

    Transport recalibrates a donor country's records to a target country's
    public targets (see ``docs/concept-schema-transport-adr.md``).
    """

    #: The value means the same thing in any country and is kept as is.
    CARRY = "carry"
    #: Converted to the target currency, then quantile-mapped to the target
    #: country's distribution for that component before reweighting.
    QUANTILE_MAP = "quantile_map"
    #: A donor-country program receipt. Transport drops it; the target
    #: country's own rules compute the target program.
    DROP = "drop"


class IndexFamily(StrEnum):
    """The price or earnings index family that moves an amount across years.

    A declaration for aging and transport, not an index series: the series
    for a given country and year come from that country's parameters or
    Chronicle. No code consumes this field yet.
    """

    EARNINGS = "earnings"
    MIXED_INCOME = "mixed_income"
    CAPITAL_INCOME = "capital_income"
    PENSIONS = "pensions"
    RENTS = "rents"
    OWNER_HOUSING_COSTS = "owner_housing_costs"
    CONSUMER_PRICES = "consumer_prices"


class AlignmentRelation(StrEnum):
    """Chronicle's concept-relation vocabulary, reused verbatim.

    Chronicle (``chronicle.core.ALLOWED_CONCEPT_RELATIONS``) accepts exactly
    these five. The relation reads from the source concept to the canonical
    concept, SKOS-style:

    - ``exact``: the same quantity.
    - ``broad_match``: the canonical concept is broader than the source.
    - ``narrow_match``: the canonical concept is narrower than the source.
    - ``approximate``: overlapping but neither contains the other; also used
      where one quantity stands in for a different one (the design note's
      "proxy"), with the difference stated in the evidence.
    - ``source_label``: no semantic alignment; the canonical id only restates
      the source's own label.
    """

    EXACT = "exact"
    BROAD_MATCH = "broad_match"
    NARROW_MATCH = "narrow_match"
    APPROXIMATE = "approximate"
    SOURCE_LABEL = "source_label"


class CanonicalConceptKind(StrEnum):
    """The three kinds of canonical concept id, told apart by syntax."""

    #: ``fact:<entity>.<name>``: a primitive fact in this schema.
    PRIMITIVE = "primitive"
    #: ``<jurisdiction>:<citation path>#<name>``: an Axiom RuleSpec concept.
    LEGAL = "legal"
    #: ``<authority>.<name>``: a statistical measure named by its publisher.
    STATISTICAL = "statistical"


class ContentBasis(StrEnum):
    """The quality tier a population's content carries.

    The design note calls this the release's quality tier. The code names it
    content basis to keep it apart from the repository's evidence-tier
    releases, target criticality tiers and UK region tiers.
    """

    #: The country's own microdata (the UK FRS).
    OWN_DATA = "own_data"
    #: A public donor file recalibrated to the country's public targets.
    #: Joint distributions follow the donor except along calibrated margins.
    TRANSPORT = "transport"
    #: No microdata at all: records synthesized from published aggregates.
    AGGREGATES_ONLY = "aggregates_only"


def canonical_concept_kind(concept_id: str) -> CanonicalConceptKind:
    """Classify a canonical concept id by its syntax.

    Raises:
        ValueError: If the id matches none of the three grammars.
    """

    if _PRIMITIVE_ID.match(concept_id):
        return CanonicalConceptKind.PRIMITIVE
    if concept_id.startswith("fact:"):
        raise ValueError(
            f"Malformed primitive concept id {concept_id!r}; expected "
            "'fact:<person|household>.<name>'."
        )
    if _LEGAL_ID.match(concept_id):
        return CanonicalConceptKind.LEGAL
    if _STATISTICAL_ID.match(concept_id):
        return CanonicalConceptKind.STATISTICAL
    raise ValueError(
        f"Unrecognized canonical concept id {concept_id!r}; expected "
        "'fact:<entity>.<name>', '<jurisdiction>:<path>#<name>' or "
        "'<authority>.<name>'."
    )


_ALIGNMENT_FIELDS: tuple[str, ...] = (
    "canonical_concept",
    "source_concept",
    "relation",
    "fact_key",
    "source_record_id",
    "authority",
    "evidence_url",
    "evidence_notes",
    "legal_vintage",
    "concept_alignment_key",
)


def _record_fields(
    data: object,
    what: str,
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> dict[str, object]:
    """``data`` as a dict, checked to be an object holding only allowed fields.

    Raises:
        ValueError: If ``data`` is not an object, or a field is unexpected or
            missing; the message names the fields.
    """

    if not isinstance(data, Mapping):
        raise ValueError(f"A {what} must be an object, not {type(data).__name__}.")
    fields = dict(data)
    required = frozenset(required)
    unexpected = sorted(map(str, set(fields) - required - frozenset(optional)))
    if unexpected:
        raise ValueError(f"A {what} has unexpected fields {unexpected}.")
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"A {what} lacks {missing}.")
    return fields


def _text_fields(fields: Mapping[str, object], what: str, names: Iterable[str]) -> None:
    """Check that each named field is text where present and not null."""

    for name in names:
        value = fields.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"A {what}'s {name!r} must be text, not {type(value).__name__}."
            )


@contextmanager
def _parsing(what: str) -> Iterator[None]:
    """Report any other structural fault in a serialized record as ValueError.

    Readers check the shapes they expect and name the field at fault; this
    catches what those checks leave (an unhashable value, a list where a pair
    belongs), so malformed input never escapes as another exception type.
    """

    try:
        yield
    except (AttributeError, IndexError, KeyError, OverflowError, TypeError) as error:
        raise ValueError(f"Malformed {what}: {error}") from error


@dataclass(frozen=True, kw_only=True)
class ConceptAlignment:
    """One source-to-canonical concept alignment: Chronicle's record.

    The fields are ``chronicle.concepts.ConceptAlignment``'s, plus the
    ``concept_alignment_key`` Chronicle's feeds carry on each record, and
    construction checks only what Chronicle's record type guarantees: the
    relation is one of Chronicle's five and both concept ids are non-empty.
    Any alignment Chronicle writes therefore loads here unchanged, including
    ``source_label`` records whose source and canonical ids coincide.
    Chronicle's own validation additionally requires evidence on an
    ``exact`` alignment; :meth:`issues` reports that. The alignments a
    :class:`Concept` carries must meet stricter rules, which the concept
    enforces.

    Attributes:
        canonical_concept: What the source aligns to: a legal, statistical or
            primitive concept id, or a publisher's bare id.
        source_concept: The aligned concept (for a concept's own alignments,
            its ``fact:`` id).
        relation: How the two relate, read source to canonical.
        fact_key: The Chronicle fact the alignment was asserted on, if any.
        source_record_id: The source record it came from, if any.
        authority: Who asserts the alignment.
        evidence_url: Where the canonical definition is published.
        evidence_notes: Why the relation holds, including any difference.
        legal_vintage: The legal vintage the canonical concept is read at
            (``tax_year_2026``).
        concept_alignment_key: The feed's content key for the record.
    """

    canonical_concept: str
    source_concept: str
    relation: AlignmentRelation
    fact_key: str | None = None
    source_record_id: str | None = None
    authority: str | None = None
    evidence_url: str | None = None
    evidence_notes: str | None = None
    legal_vintage: str | None = None
    concept_alignment_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", AlignmentRelation(self.relation))
        if not self.source_concept or not self.canonical_concept:
            raise ValueError("An alignment names its source and canonical concepts.")

    def issues(self) -> tuple[str, ...]:
        """Chronicle's validation findings for this record (none when clean)."""
        if self.relation is AlignmentRelation.EXACT and not (
            self.evidence_notes or self.evidence_url
        ):
            return ("Exact source-to-canonical concept alignments need evidence.",)
        return ()

    @property
    def canonical_kind(self) -> CanonicalConceptKind | None:
        """The kind of the canonical concept, or None for a bare id."""
        try:
            return canonical_concept_kind(self.canonical_concept)
        except ValueError:
            return None

    @property
    def jurisdiction(self) -> str | None:
        """The canonical concept's jurisdiction code, for legal concepts."""
        if self.canonical_kind is not CanonicalConceptKind.LEGAL:
            return None
        return self.canonical_concept.split(":", 1)[0]

    def to_dict(self) -> dict[str, str]:
        """A JSON-ready record with Chronicle's keys, omitting absent ones."""
        record = {name: getattr(self, name) for name in _ALIGNMENT_FIELDS}
        record["relation"] = self.relation.value
        return {key: value for key, value in record.items() if value is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ConceptAlignment:
        """Read a Chronicle ``concept_alignment`` record.

        Raises:
            ValueError: If ``data`` is not an object, a key is not one of the
                record's fields, a required one is missing, or a value is not
                text.
        """
        with _parsing("concept alignment"):
            if not isinstance(data, Mapping):
                raise ValueError("A concept alignment must be an object.")
            unknown = set(data) - set(_ALIGNMENT_FIELDS)
            if unknown:
                raise ValueError(
                    f"Unknown concept-alignment fields {sorted(map(str, unknown))}."
                )
            missing = {"canonical_concept", "source_concept", "relation"} - set(data)
            if missing:
                raise ValueError(f"Concept alignment lacks {sorted(missing)}.")
            _text_fields(data, "concept alignment", _ALIGNMENT_FIELDS)
            return cls(**data)


@dataclass(frozen=True, kw_only=True)
class MonetaryHandling:
    """Currency and price-level handling for an amount concept.

    Every amount is nominal: whole units of the concept frame's declared
    currency (:attr:`ConceptFrameDeclaration.currency`), at the prices of the
    concept's own period. Currency belongs to the frame, never to a column:
    a frame is meant to carry one declaration. Nothing yet binds a
    declaration to its tables or checks it when encoding, so keeping one
    currency per frame is the producer's contract.

    Attributes:
        index_family: The index family that moves the amount across years.
        signed: Whether the amount may be negative (a net profit or loss).
    """

    index_family: IndexFamily
    signed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_family", IndexFamily(self.index_family))

    @property
    def price_basis(self) -> str:
        """Always ``"nominal"``: prices of the concept's own period."""
        return "nominal"


@dataclass(frozen=True, kw_only=True)
class Concept:
    """One engine-neutral primitive fact.

    Attributes:
        id: ``fact:<entity>.<name>``.
        label: A short human label.
        definition: What the value includes and excludes, precisely enough to
            decide whether a source item or an engine input measures it.
        dtype: Kernel dtype kind: ``float``, ``int``, ``bool`` or ``str``.
        unit: The unit values are expressed in.
        period: Kernel period semantics: ``year``, ``month`` or ``point``.
        temporal_basis: What the value measures over its period. An annual
            flow's period is ``year``.
        provenance: How values come to exist.
        transport: What transport does with the column.
        monetary: Currency and price-level handling; set exactly when the
            unit is ``base_currency``.
        domain: The closed category set; set exactly when the unit is
            ``category``.
        lower: Inclusive lower bound for numeric values, if any.
        upper: Upper bound for numeric values, if any.
        upper_inclusive: Whether :attr:`upper` itself is allowed.
        nullable: Whether a missing value is meaningful. Only relationship
            pointers are nullable: null means the relation is absent.
        standard: The international definition the concept follows, if any.
        alignments: Alignments from this concept to legal concepts.
    """

    id: str
    label: str
    definition: str
    dtype: str
    unit: Unit
    period: str
    temporal_basis: TemporalBasis
    provenance: ProvenanceClass
    transport: TransportRule
    monetary: MonetaryHandling | None = None
    domain: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None
    upper_inclusive: bool = True
    nullable: bool = False
    standard: str | None = None
    alignments: tuple[ConceptAlignment, ...] = field(default=())

    def __post_init__(self) -> None:
        for name, enum in (
            ("unit", Unit),
            ("temporal_basis", TemporalBasis),
            ("provenance", ProvenanceClass),
            ("transport", TransportRule),
        ):
            object.__setattr__(self, name, enum(getattr(self, name)))
        match = _PRIMITIVE_ID.match(self.id)
        if match is None:
            raise ValueError(
                f"Concept id {self.id!r} must have the form "
                "'fact:<person|household>.<name>'."
            )
        if not self.label or not self.definition:
            raise ValueError(f"Concept {self.id!r} needs a label and a definition.")
        if self.dtype not in _DTYPE_KINDS:
            raise ValueError(
                f"Concept {self.id!r} dtype must be one of {_DTYPE_KINDS}."
            )
        if self.period not in _PERIOD_SEMANTICS:
            raise ValueError(
                f"Concept {self.id!r} period must be one of {_PERIOD_SEMANTICS}."
            )
        if self.temporal_basis is TemporalBasis.ANNUAL_FLOW and self.period != "year":
            raise ValueError(
                f"Concept {self.id!r}: an annual flow accumulates over the year, "
                f"so its period is year, not {self.period}."
            )
        self._validate_unit_contract()
        self._validate_bounds()
        self._validate_transport()
        self._validate_alignments()

    def _validate_alignments(self) -> None:
        """A concept's own alignments meet stricter rules than Chronicle's.

        Each relates this concept to a legal or statistical concept (never
        another primitive), asserts a semantic relation (not
        ``source_label``), names its authority, carries evidence, states the
        legal vintage of a legal target, and appears once.
        """
        seen: set[tuple[str, str]] = set()
        for alignment in self.alignments:
            where = f"Concept {self.id!r} alignment to {alignment.canonical_concept!r}"
            if alignment.source_concept != self.id:
                raise ValueError(f"{where} has source {alignment.source_concept!r}.")
            kind = alignment.canonical_kind
            if kind not in (
                CanonicalConceptKind.LEGAL,
                CanonicalConceptKind.STATISTICAL,
            ):
                raise ValueError(
                    f"{where}: a concept aligns only to legal or statistical "
                    "concepts; primitives are the neutral layer."
                )
            if alignment.relation is AlignmentRelation.SOURCE_LABEL:
                raise ValueError(f"{where} asserts no semantic relation.")
            if not alignment.authority:
                raise ValueError(f"{where} names no authority.")
            if alignment.issues() or not (
                alignment.evidence_notes or alignment.evidence_url
            ):
                raise ValueError(f"{where} carries no evidence.")
            if kind is CanonicalConceptKind.LEGAL and not alignment.legal_vintage:
                raise ValueError(f"{where} states no legal vintage.")
            key = (alignment.canonical_concept, alignment.relation.value)
            if key in seen:
                raise ValueError(f"{where} appears twice.")
            seen.add(key)

    def _validate_unit_contract(self) -> None:
        is_money = self.unit is Unit.BASE_CURRENCY
        if is_money != (self.monetary is not None):
            raise ValueError(
                f"Concept {self.id!r}: monetary handling is required exactly "
                "for base_currency amounts."
            )
        if is_money and self.dtype != "float":
            raise ValueError(f"Amount concept {self.id!r} must be float.")
        is_category = self.unit is Unit.CATEGORY
        if is_category != bool(self.domain):
            raise ValueError(
                f"Concept {self.id!r}: a domain is required exactly for "
                "category concepts."
            )
        if is_category:
            if self.dtype != "str":
                raise ValueError(f"Category concept {self.id!r} must be str.")
            if len(set(self.domain)) != len(self.domain):
                raise ValueError(f"Concept {self.id!r} has duplicate categories.")
            bad = [value for value in self.domain if not _NAME_PATTERN.match(value)]
            if bad:
                raise ValueError(
                    f"Concept {self.id!r} categories must be snake_case: {bad}."
                )
        if (self.unit is Unit.BOOLEAN) != (self.dtype == "bool"):
            raise ValueError(
                f"Concept {self.id!r}: the boolean unit and bool dtype go together."
            )
        if self.nullable and self.unit is not Unit.PERSON_ID:
            raise ValueError(
                f"Concept {self.id!r}: only person-id pointers may be nullable "
                "(null means the relation is absent)."
            )
        if self.unit is Unit.PERSON_ID and (
            self.dtype != "int" or not self.name.endswith("_person_id")
        ):
            raise ValueError(
                f"Pointer concept {self.id!r} must be an int named *_person_id."
            )

    def _validate_bounds(self) -> None:
        numeric = self.dtype in ("float", "int")
        if not numeric and (self.lower is not None or self.upper is not None):
            raise ValueError(f"Non-numeric concept {self.id!r} cannot have bounds.")
        if self.lower is not None and self.upper is not None:
            if not self.lower < self.upper:
                raise ValueError(f"Concept {self.id!r} bounds are empty.")
        if self.monetary is not None:
            if self.monetary.signed and self.lower is not None:
                raise ValueError(f"Signed amount {self.id!r} cannot be bounded below.")
            if not self.monetary.signed and self.lower != 0.0:
                raise ValueError(
                    f"Unsigned amount {self.id!r} must have lower bound 0."
                )

    def _validate_transport(self) -> None:
        if self.transport is TransportRule.QUANTILE_MAP and self.monetary is None:
            raise ValueError(
                f"Only amounts can be quantile-mapped; {self.id!r} is not one."
            )
        if self.provenance is ProvenanceClass.GENERATED and (
            self.transport is not TransportRule.CARRY
        ):
            raise ValueError(
                f"Generated concept {self.id!r} must carry: regenerating it "
                "would break persistence."
            )

    @property
    def entity(self) -> str:
        """The concept's entity: ``person`` or ``household``."""
        return self.id.split(":", 1)[1].split(".", 1)[0]

    @property
    def name(self) -> str:
        """The short name, also the concept column's name on its table."""
        return self.id.split(".", 1)[1]

    def contract(self) -> dict[str, object]:
        """The concept's declared contract as canonical JSON-ready data."""
        return {
            "id": self.id,
            "label": self.label,
            "definition": self.definition,
            "dtype": self.dtype,
            "unit": self.unit.value,
            "period": self.period,
            "temporal_basis": self.temporal_basis.value,
            "provenance": self.provenance.value,
            "transport": self.transport.value,
            "monetary": None
            if self.monetary is None
            else {
                "price_basis": self.monetary.price_basis,
                "index_family": self.monetary.index_family.value,
                "signed": self.monetary.signed,
            },
            "domain": list(self.domain),
            "lower": self.lower,
            "upper": self.upper,
            "upper_inclusive": self.upper_inclusive,
            "nullable": self.nullable,
            "standard": self.standard,
            "alignments": [alignment.to_dict() for alignment in self.alignments],
        }


def _amount(
    name: str,
    label: str,
    definition: str,
    *,
    index_family: IndexFamily,
    entity: str = "person",
    signed: bool = False,
    transport: TransportRule = TransportRule.QUANTILE_MAP,
    standard: str | None = None,
) -> Concept:
    return Concept(
        id=f"fact:{entity}.{name}",
        label=label,
        definition=definition,
        dtype="float",
        unit=Unit.BASE_CURRENCY,
        period="year",
        temporal_basis=TemporalBasis.ANNUAL_FLOW,
        provenance=ProvenanceClass.OBSERVED,
        transport=transport,
        monetary=MonetaryHandling(index_family=index_family, signed=signed),
        lower=None if signed else 0.0,
        standard=standard,
    )


def _pointer(
    name: str, label: str, definition: str, *, entity: str, nullable: bool = True
) -> Concept:
    return Concept(
        id=f"fact:{entity}.{name}",
        label=label,
        definition=definition,
        dtype="int",
        unit=Unit.PERSON_ID,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.DERIVED,
        transport=TransportRule.CARRY,
        nullable=nullable,
    )


_CANBERRA = "Canberra Group Handbook on Household Income Statistics (UNECE, 2011)"
_ISCED = "International Standard Classification of Education 2011 (UNESCO)"

#: Tenure categories. The two social-rent categories separate a public
#: landlord (UK council, NZ Kainga Ora, US public housing agency) from a
#: non-profit one (UK housing association, NZ registered community housing
#: provider), a split UK and NZ rules both use.
TENURE_DOMAIN: tuple[str, ...] = (
    "owned_outright",
    "owned_with_mortgage",
    "rented_public_authority",
    "rented_nonprofit_social",
    "rented_private",
    "rent_free",
)

#: Highest completed education, ISCED 2011 levels 0-8 in order.
ATTAINMENT_DOMAIN: tuple[str, ...] = (
    "less_than_primary",
    "primary",
    "lower_secondary",
    "upper_secondary",
    "post_secondary_non_tertiary",
    "short_cycle_tertiary",
    "bachelor_or_equivalent",
    "master_or_equivalent",
    "doctoral_or_equivalent",
)

#: Current enrollment by ISCED level band; ``tertiary`` is ISCED 5-8.
ENROLLMENT_DOMAIN: tuple[str, ...] = (
    "not_enrolled",
    "early_childhood",
    "primary",
    "lower_secondary",
    "upper_secondary",
    "post_secondary_non_tertiary",
    "tertiary",
)

MARITAL_STATUS_DOMAIN: tuple[str, ...] = (
    "never_married",
    "married",
    "separated",
    "divorced",
    "widowed",
)

SEX_DOMAIN: tuple[str, ...] = ("female", "male")

#: The rented tenures: the households that pay rent.
RENTED_TENURES: tuple[str, ...] = tuple(
    value for value in TENURE_DOMAIN if value.startswith("rented_")
)

#: The concept schema, in declaration order.
CONCEPTS: tuple[Concept, ...] = (
    # --- Demography -------------------------------------------------------
    Concept(
        id="fact:person.age",
        label="Age",
        definition=(
            "Age in completed years at the survey reference date. Top-coded "
            "sources keep their top code; the value is never imputed from "
            "legal age thresholds."
        ),
        dtype="int",
        unit=Unit.YEARS,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        lower=0.0,
        upper=130.0,
    ),
    Concept(
        id="fact:person.sex",
        label="Sex",
        definition=(
            "Sex as the survey records it. Both donor surveys and every mapped "
            "engine record two categories; a source with more must map them "
            "before storing this column."
        ),
        dtype="str",
        unit=Unit.CATEGORY,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        domain=SEX_DOMAIN,
    ),
    Concept(
        id="fact:person.legal_marital_status",
        label="Legal marital status",
        definition=(
            "Legal marital status at the reference date. 'married' includes "
            "civil partnerships and civil unions. Cohabitation is not a "
            "marital status: it is a partner pointer without marriage."
        ),
        dtype="str",
        unit=Unit.CATEGORY,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        domain=MARITAL_STATUS_DOMAIN,
    ),
    # --- Relationships ----------------------------------------------------
    _pointer(
        "partner_person_id",
        "Co-resident partner",
        (
            "person_id of this person's spouse, civil partner or cohabiting "
            "partner living in the same household; null if none. The "
            "relation is symmetric. This is the design note's spouse pointer; "
            "legal_marital_status says whether the couple is married."
        ),
        entity="person",
    ),
    _pointer(
        "parent_1_person_id",
        "First co-resident parent",
        (
            "person_id of a parent (birth, adoptive or step) living in the "
            "same household; null if none. When only one parent is "
            "co-resident it is always parent 1."
        ),
        entity="person",
    ),
    _pointer(
        "parent_2_person_id",
        "Second co-resident parent",
        (
            "person_id of a second co-resident parent, distinct from parent "
            "1; null if none."
        ),
        entity="person",
    ),
    _pointer(
        "reference_person_id",
        "Household reference person",
        (
            "person_id of the household's reference person (householder, "
            "HRP): the member the survey's relationship roster is anchored "
            "on. Never null: every household has exactly one."
        ),
        entity="household",
        nullable=False,
    ),
    # --- Labour income ----------------------------------------------------
    _amount(
        "employment_income",
        "Employment income",
        (
            "Gross wages and salaries from all employee jobs in the reference "
            "year, including overtime, tips, bonuses and commissions, before "
            "employee income tax and social contributions. Excludes employer "
            "social and pension contributions and benefits in kind."
        ),
        index_family=IndexFamily.EARNINGS,
        standard=f"{_CANBERRA}: employee income (cash)",
    ),
    _amount(
        "nonfarm_self_employment_income",
        "Non-farm self-employment income",
        (
            "Net profit or loss from unincorporated non-farm businesses and "
            "professional practice in the reference year: receipts less "
            "business expenses, before income tax. Farm profit is the "
            "separate farm_self_employment_income concept, never included "
            "here."
        ),
        index_family=IndexFamily.MIXED_INCOME,
        signed=True,
        standard=f"{_CANBERRA}: income from self-employment",
    ),
    _amount(
        "farm_self_employment_income",
        "Farm self-employment income",
        (
            "Net profit or loss from operating a farm as its owner, tenant or "
            "sharecropper in the reference year, before income tax."
        ),
        index_family=IndexFamily.MIXED_INCOME,
        signed=True,
        standard=f"{_CANBERRA}: income from self-employment",
    ),
    # --- Capital income ---------------------------------------------------
    _amount(
        "interest_income",
        "Interest income",
        (
            "Interest received on deposits, bonds and loans in the reference "
            "year, gross of tax withheld, whether or not the payer's "
            "jurisdiction taxes it."
        ),
        index_family=IndexFamily.CAPITAL_INCOME,
        standard=f"{_CANBERRA}: property income from financial assets",
    ),
    _amount(
        "dividend_income",
        "Dividend income",
        (
            "Dividends and other distributions received from corporations and "
            "investment funds in the reference year, gross of tax withheld."
        ),
        index_family=IndexFamily.CAPITAL_INCOME,
        standard=f"{_CANBERRA}: property income from financial assets",
    ),
    _amount(
        "rental_income",
        "Net rental income",
        (
            "Rent received from letting land and buildings other than the "
            "household's own dwelling, net of expenses, in the reference "
            "year. Income from subletting part of the own dwelling is "
            "excluded."
        ),
        index_family=IndexFamily.CAPITAL_INCOME,
        signed=True,
        standard=f"{_CANBERRA}: property income from non-financial assets",
    ),
    _amount(
        "realized_capital_gains",
        "Realized capital gains",
        (
            "Net gains less losses realized on the disposal of assets in the "
            "reference year. Not household income in the Canberra sense; "
            "carried because tax systems tax it."
        ),
        index_family=IndexFamily.CAPITAL_INCOME,
        signed=True,
    ),
    # --- Financial wealth -------------------------------------------------
    Concept(
        id="fact:person.liquid_financial_assets",
        label="Liquid financial assets",
        definition=(
            "Value at the reference date of the person's deposits (checking, "
            "current, savings and money-market accounts and term deposits with "
            "banks and other deposit-taking institutions), shares and "
            "investment-fund units, and bonds and other debt securities, "
            "before deducting any debt. A jointly held asset is divided among "
            "its owners, so each asset counts once in a household total. "
            "Excludes notes and coins, money lent to others, balances in "
            "pension and retirement-savings schemes (such as US IRAs and "
            "401(k) plans or NZ KiwiSaver), life insurance, equity in a "
            "business the person runs, and real estate."
        ),
        dtype="float",
        unit=Unit.BASE_CURRENCY,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.QUANTILE_MAP,
        monetary=MonetaryHandling(index_family=IndexFamily.CONSUMER_PRICES),
        lower=0.0,
    ),
    # --- Pensions ---------------------------------------------------------
    _amount(
        "private_pension_income",
        "Private pension income",
        (
            "Regular income from employer-sponsored and personal pension "
            "schemes and annuities in the reference year. Excludes "
            "social-insurance and state pensions, lump-sum withdrawals and "
            "withdrawals from tax-favoured savings wrappers that are not "
            "pension schemes."
        ),
        index_family=IndexFamily.PENSIONS,
        standard=f"{_CANBERRA}: pensions from non-social-security schemes",
    ),
    _amount(
        "public_pension_income",
        "Public old-age pension income",
        (
            "Old-age pension received from a social-insurance or state scheme "
            "(US Social Security retirement, UK State Pension, NZ "
            "Superannuation) in the reference year. A donor-country program "
            "receipt: transport drops it and the target country's rules "
            "compute its own."
        ),
        index_family=IndexFamily.PENSIONS,
        transport=TransportRule.DROP,
        standard=f"{_CANBERRA}: social-security pensions",
    ),
    # --- Work intensity ---------------------------------------------------
    Concept(
        id="fact:person.usual_weekly_hours",
        label="Usual weekly hours worked",
        definition=(
            "Usual hours worked per week, across all jobs and "
            "self-employment, in the weeks the person worked during the "
            "reference year. Zero for a person who did not work."
        ),
        dtype="float",
        unit=Unit.HOURS_PER_WEEK,
        period="year",
        temporal_basis=TemporalBasis.USUAL_RATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        lower=0.0,
        upper=168.0,
    ),
    Concept(
        id="fact:person.weeks_worked",
        label="Weeks worked",
        definition=(
            "Weeks in the reference year in which the person did any paid "
            "work, including paid leave."
        ),
        dtype="int",
        unit=Unit.WEEKS,
        period="year",
        temporal_basis=TemporalBasis.ANNUAL_FLOW,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        lower=0.0,
        upper=53.0,
    ),
    # --- Disability and education ----------------------------------------
    Concept(
        id="fact:person.has_disability",
        label="Has a disability",
        definition=(
            "Reports a functional limitation: difficulty seeing, hearing, "
            "walking or climbing stairs, remembering or concentrating, "
            "self-care, communicating or living independently, as the "
            "source survey's functional-difficulty items record it. Sources "
            "differ in items, severity cut-offs and duration tests (the "
            "ACS/CPS six-question set, the Washington Group short set, the "
            "FRS limiting long-standing illness question), so a producer "
            "documents its instrument. A benefit program's legal disability "
            "status is a different, legal concept."
        ),
        dtype="bool",
        unit=Unit.BOOLEAN,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        standard=(
            "Functional-difficulty survey items, such as the Washington Group "
            "short set or the ACS/CPS six-question set"
        ),
    ),
    Concept(
        id="fact:person.educational_attainment",
        label="Educational attainment",
        definition="Highest education level completed, by ISCED 2011 level.",
        dtype="str",
        unit=Unit.CATEGORY,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        domain=ATTAINMENT_DOMAIN,
        standard=_ISCED,
    ),
    Concept(
        id="fact:person.education_enrollment",
        label="Current education enrollment",
        definition=(
            "The ISCED 2011 level band of the education programme the person "
            "is enrolled in at the reference date; not_enrolled otherwise."
        ),
        dtype="str",
        unit=Unit.CATEGORY,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        domain=ENROLLMENT_DOMAIN,
        standard=_ISCED,
    ),
    Concept(
        id="fact:person.enrolled_full_time",
        label="Enrolled full time",
        definition=(
            "Enrolled in education full time as the programme defines it. "
            "False for anyone not enrolled."
        ),
        dtype="bool",
        unit=Unit.BOOLEAN,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
    ),
    # --- Take-up ----------------------------------------------------------
    Concept(
        id="fact:person.take_up_seed",
        label="Persistent take-up seed",
        definition=(
            "A uniform draw on [0, 1) fixed for the life of the person's "
            "record identity. Program take-up draws derive from it by "
            "derive_take_up_draws(seed, program); a program is taken up when "
            "its draw falls below the rate aligned to that country's "
            "caseloads. Stored as float64: the draws hash the seed's exact "
            "bits, so a float32 round trip would change every draw."
        ),
        dtype="float",
        unit=Unit.UNIT_INTERVAL,
        period="point",
        temporal_basis=TemporalBasis.PERSISTENT,
        provenance=ProvenanceClass.GENERATED,
        transport=TransportRule.CARRY,
        lower=0.0,
        upper=1.0,
        upper_inclusive=False,
    ),
    # --- Housing ----------------------------------------------------------
    Concept(
        id="fact:household.tenure",
        label="Tenure",
        definition=(
            "The terms on which the household occupies its dwelling at the "
            "reference date. Social renting is split by landlord type: a "
            "public authority or a non-profit social landlord. Renting from a "
            "private landlord with a housing subsidy is still rented_private."
        ),
        dtype="str",
        unit=Unit.CATEGORY,
        period="point",
        temporal_basis=TemporalBasis.REFERENCE_STATE,
        provenance=ProvenanceClass.OBSERVED,
        transport=TransportRule.CARRY,
        domain=TENURE_DOMAIN,
    ),
    _amount(
        "rent",
        "Rent",
        (
            "Contract rent charged for the dwelling over the reference year, "
            "before deducting any housing benefit or subsidy, excluding "
            "separately billed utilities and service charges. Zero for "
            "owners and rent-free occupiers."
        ),
        index_family=IndexFamily.RENTS,
        entity="household",
    ),
    _amount(
        "mortgage_interest",
        "Mortgage interest",
        (
            "Interest paid over the reference year on loans secured on the "
            "household's own dwelling."
        ),
        index_family=IndexFamily.OWNER_HOUSING_COSTS,
        entity="household",
    ),
    _amount(
        "mortgage_principal",
        "Mortgage principal repayments",
        (
            "Capital repaid over the reference year on loans secured on the "
            "household's own dwelling."
        ),
        index_family=IndexFamily.OWNER_HOUSING_COSTS,
        entity="household",
    ),
    _amount(
        "property_tax",
        "Recurrent tax on the dwelling",
        (
            "Recurrent local taxes on the dwelling that the household itself "
            "is liable for, for the reference year, before rebates or "
            "reductions. Who is liable differs by country: owners for US real "
            "estate tax, NZ rates and the Belgian withholding tax on "
            "immovable property; occupiers, renters included, for UK council "
            "tax. Tax a landlord pays and recovers through rent is excluded."
        ),
        index_family=IndexFamily.OWNER_HOUSING_COSTS,
        entity="household",
    ),
)

#: Concepts by id, read-only.
CONCEPT_BY_ID: Mapping[str, Concept] = MappingProxyType(
    {item.id: item for item in CONCEPTS}
)
if len(CONCEPT_BY_ID) != len(CONCEPTS):  # pragma: no cover - import-time guard
    raise RuntimeError("Concept ids must be unique.")

_CONCEPT_BY_COLUMN: Mapping[tuple[str, str], Concept] = MappingProxyType(
    {(item.entity, item.name): item for item in CONCEPTS}
)


def concept(concept_id: str) -> Concept:
    """Return the concept with id ``concept_id``.

    Raises:
        KeyError: If no concept has that id.
    """

    try:
        return CONCEPT_BY_ID[concept_id]
    except KeyError:
        raise KeyError(f"Unknown concept {concept_id!r}.") from None


def concepts_for_entity(entity: str) -> tuple[Concept, ...]:
    """The concepts that live on ``entity``, in declaration order."""

    if entity not in CONCEPT_ENTITIES:
        raise ValueError(f"Concepts live on {CONCEPT_ENTITIES}, not {entity!r}.")
    return tuple(item for item in CONCEPTS if item.entity == entity)


def concept_for_column(entity: str, column: str) -> Concept | None:
    """The concept a concept-frame column holds, or None for other columns."""

    return _CONCEPT_BY_COLUMN.get((entity, column))


def concept_schema_sha256(concepts: Iterable[Concept] = CONCEPTS) -> str:
    """SHA-256 of the schema's declared contracts, for artifacts that pin it.

    Alignments are left out: they record how a concept relates to each
    jurisdiction's law and grow as jurisdictions are aligned, without
    changing what any stored value means.
    """

    payload = {
        "version": CONCEPT_SCHEMA_VERSION,
        "concepts": [
            {
                key: value
                for key, value in item.contract().items()
                if key != "alignments"
            }
            for item in concepts
        ],
    }
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Concept frames
# ---------------------------------------------------------------------------

_ISO_CURRENCY = re.compile(r"^[A-Z]{3}$")
_COUNTRY_CODE = re.compile(r"^[a-z]{2}$")


@dataclass(frozen=True, kw_only=True)
class ConceptFrameDeclaration:
    """What a concept frame as a whole declares about its content.

    Attributes:
        country: The country the population represents (lowercase ISO
            3166-1 alpha-2, as the repository's country packages use).
        currency: ISO 4217 code every amount is expressed in.
        reference_year: The reference year amounts and states refer to.
        content_basis: The quality tier.
        donor_country: For transport, the country whose records supplied the
            content; required exactly for transport, and different from
            :attr:`country`.
    """

    country: str
    currency: str
    reference_year: int
    content_basis: ContentBasis
    donor_country: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_basis", ContentBasis(self.content_basis))
        if not _COUNTRY_CODE.match(self.country):
            raise ValueError(f"country must be a lowercase ISO code: {self.country!r}.")
        if not _ISO_CURRENCY.match(self.currency):
            raise ValueError(f"currency must be an ISO 4217 code: {self.currency!r}.")
        is_transport = self.content_basis is ContentBasis.TRANSPORT
        if is_transport != (self.donor_country is not None):
            raise ValueError(
                "A donor country is declared exactly for transport content."
            )
        if self.donor_country is not None:
            if not _COUNTRY_CODE.match(self.donor_country):
                raise ValueError(
                    f"donor_country must be a lowercase ISO code: "
                    f"{self.donor_country!r}."
                )
            if self.donor_country == self.country:
                raise ValueError(
                    "Transport needs a donor country other than the target."
                )


@dataclass(frozen=True)
class ConceptViolation:
    """One way a concept frame breaks its declared contract."""

    entity: str
    column: str
    code: str
    message: str
    rows: int = 0


def validate_concept_tables(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[ConceptViolation, ...]:
    """Check a concept frame against every contract the schema declares.

    ``tables`` holds a ``person`` and a ``household`` table with the kernel's
    id columns. Each concept column present is checked for dtype, nullability,
    category domain and bounds; relationship pointers are checked for
    referential integrity (target exists, lives in the same household, is not
    the person), partner symmetry, parent ordering, distinct parents, no
    parent who is also the partner, and an acyclic parent graph; every
    household must have exactly one reference person among its members.
    The consistency rules the definitions state are checked across concepts:
    full-time enrollment needs an enrollment, weekly hours and weeks worked
    are zero together, only renters pay rent, and only owners with a
    mortgage pay mortgage interest or principal. Columns the schema does not
    describe are extensions and pass unchecked.

    Returns:
        Every violation found, empty for a valid frame.

    Raises:
        ValueError: If a table or an id column is missing, ids are not unique,
            or a person points at an unknown household.
    """

    person, household = _require_structure(tables)
    violations: list[ConceptViolation] = []
    for entity, table in (("person", person), ("household", household)):
        for column in table.columns:
            item = concept_for_column(entity, column)
            if item is not None:
                violations.extend(_check_column(item, table[column]))
    unreadable = {
        (item.entity, item.column) for item in violations if item.code == "dtype"
    }
    violations.extend(_check_pointers(person, household))
    violations.extend(_check_consistency(person, household, unreadable))
    return tuple(violations)


def _check_consistency(
    person: pd.DataFrame,
    household: pd.DataFrame,
    unreadable: set[tuple[str, str]],
) -> list[ConceptViolation]:
    """Check the cross-concept rules the definitions state.

    Columns already reported as unreadable (wrong dtype) are skipped; missing
    values never count as a breach here, since the column check reports them.
    """
    out: list[ConceptViolation] = []

    def usable(entity: str, table: pd.DataFrame, *columns: str) -> bool:
        return all(
            column in table.columns and (entity, column) not in unreadable
            for column in columns
        )

    def flags(values: pd.Series) -> np.ndarray:
        return values.astype(object).eq(True).to_numpy(dtype=bool)

    def positive(values: pd.Series) -> np.ndarray:
        numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        return np.nan_to_num(numbers, nan=0.0) > 0

    def nonzero(values: pd.Series) -> np.ndarray:
        numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        return np.nan_to_num(numbers, nan=0.0) != 0

    def add(entity: str, column: str, mask: np.ndarray, message: str) -> None:
        if mask.any():
            out.append(
                ConceptViolation(
                    entity, column, "inconsistent", message, int(mask.sum())
                )
            )

    if usable("person", person, "enrolled_full_time", "education_enrollment"):
        not_enrolled = (
            person["education_enrollment"].astype(object).eq("not_enrolled").to_numpy()
        )
        add(
            "person",
            "enrolled_full_time",
            flags(person["enrolled_full_time"]) & not_enrolled,
            "Full time needs an enrollment.",
        )
    if usable("person", person, "usual_weekly_hours", "weeks_worked"):
        hours = person["usual_weekly_hours"]
        weeks = person["weeks_worked"]
        known = (hours.notna() & weeks.notna()).to_numpy()
        add(
            "person",
            "usual_weekly_hours",
            known & (positive(hours) != positive(weeks)),
            "Weekly hours and weeks worked are zero together.",
        )
    if usable("household", household, "tenure"):
        tenure = household["tenure"].astype(object)
        known = tenure.notna().to_numpy()
        if usable("household", household, "rent"):
            add(
                "household",
                "rent",
                known
                & nonzero(household["rent"])
                & ~tenure.isin(RENTED_TENURES).to_numpy(),
                "Only renters pay rent.",
            )
        with_mortgage = tenure.eq("owned_with_mortgage").to_numpy(dtype=bool)
        for column in ("mortgage_interest", "mortgage_principal"):
            if usable("household", household, column):
                add(
                    "household",
                    column,
                    known & nonzero(household[column]) & ~with_mortgage,
                    "Only owners with a mortgage repay one.",
                )
    return out


def _require_structure(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [entity for entity in CONCEPT_ENTITIES if entity not in tables]
    if missing:
        raise ValueError(f"A concept frame needs {list(CONCEPT_ENTITIES)} tables.")
    person, household = tables["person"], tables["household"]
    for table, columns in (
        (person, (_PERSON_ID, _PERSON_HOUSEHOLD_ID)),
        (household, (_HOUSEHOLD_ID,)),
    ):
        absent = [column for column in columns if column not in table.columns]
        if absent:
            raise ValueError(f"Concept frame table lacks id column(s) {absent}.")
    for table, columns in (
        (person, (_PERSON_ID, _PERSON_HOUSEHOLD_ID)),
        (household, (_HOUSEHOLD_ID,)),
    ):
        for column in columns:
            ids = table[column]
            if ids.dtype.kind not in "iu" or ids.isna().any():
                raise ValueError(
                    f"Concept frame id column {column!r} must hold non-null "
                    f"integers, found dtype {ids.dtype}."
                )
    if not person[_PERSON_ID].is_unique or not household[_HOUSEHOLD_ID].is_unique:
        raise ValueError("Concept frame ids must be unique.")
    unknown = ~person[_PERSON_HOUSEHOLD_ID].isin(household[_HOUSEHOLD_ID])
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} person(s) name an unknown household.")
    return person, household


def _check_column(item: Concept, values: pd.Series) -> list[ConceptViolation]:
    entity, column = item.entity, item.name
    out: list[ConceptViolation] = []

    def add(code: str, message: str, rows: int = 0) -> None:
        out.append(ConceptViolation(entity, column, code, message, rows))

    nulls = values.isna()
    if nulls.any() and not item.nullable:
        add("null", f"{column} is not nullable.", int(nulls.sum()))
    present = values[~nulls]
    kind = values.dtype.kind
    expected_kinds = {
        "float": "fi",
        "int": "iu",
        "bool": "b",
        "str": "OUT",
    }[item.dtype]
    if kind not in expected_kinds:
        add("dtype", f"{column} must hold {item.dtype}, found dtype {values.dtype}.")
        return out
    if item.dtype == "str":
        not_text = ~np.fromiter(
            (isinstance(value, str) for value in present.to_numpy(dtype=object)),
            dtype=bool,
            count=len(present),
        )
        if not_text.any():
            add("dtype", f"{column} must hold strings.", int(not_text.sum()))
            return out
    if item.temporal_basis is TemporalBasis.PERSISTENT and item.dtype == "float":
        if values.dtype != np.float64:
            add(
                "dtype",
                f"{column} is persistent state and must be stored as float64, "
                f"found {values.dtype}.",
            )
            return out
    if item.domain:
        outside = ~present.isin(item.domain)
        if outside.any():
            add(
                "domain",
                f"{column} has values outside {list(item.domain)}: "
                f"{sorted(set(present[outside]))[:5]}.",
                int(outside.sum()),
            )
    if item.dtype in ("float", "int") and len(present):
        numbers = present.astype("float64")
        finite = np.isfinite(numbers.to_numpy())
        if not finite.all():
            add("non_finite", f"{column} must be finite.", int((~finite).sum()))
            numbers = numbers[finite]
        if item.lower is not None:
            below = numbers < item.lower
            if below.any():
                add("bounds", f"{column} is below {item.lower}.", int(below.sum()))
        if item.upper is not None:
            above = (
                numbers > item.upper if item.upper_inclusive else numbers >= item.upper
            )
            if above.any():
                add("bounds", f"{column} exceeds {item.upper}.", int(above.sum()))
    return out


def _check_pointers(
    person: pd.DataFrame, household: pd.DataFrame
) -> list[ConceptViolation]:
    out: list[ConceptViolation] = []
    ids = pd.Index(person[_PERSON_ID].to_numpy())
    own_household = person[_PERSON_HOUSEHOLD_ID].to_numpy()
    rows = np.arange(len(person))

    def add(
        entity: str, column: str, code: str, mask: np.ndarray, message: str
    ) -> None:
        if mask.any():
            out.append(ConceptViolation(entity, column, code, message, int(mask.sum())))

    def positions(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Each pointer's target row (-1 when absent or unknown), and presence."""
        present = values.notna().to_numpy()
        found = np.full(len(values), -1, dtype=np.int64)
        if present.any():
            found[present] = ids.get_indexer(values[present].to_numpy(dtype=np.int64))
        return found, present

    pointer_rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for column in ("partner_person_id", "parent_1_person_id", "parent_2_person_id"):
        if column not in person.columns or not _is_integer_column(person[column]):
            continue
        found, present = positions(person[column])
        pointer_rows[column] = (found, present)
        linked = found >= 0
        add(
            "person",
            column,
            "dangling",
            present & ~linked,
            f"{column} names no person.",
        )
        other = linked & (own_household[np.maximum(found, 0)] != own_household)
        add(
            "person",
            column,
            "cross_household",
            other,
            f"{column} points outside the person's household.",
        )
        add(
            "person",
            column,
            "self",
            linked & (found == rows),
            f"{column} points at the person.",
        )

    if "partner_person_id" in pointer_rows:
        found, _ = pointer_rows["partner_person_id"]
        linked = found >= 0
        back = np.where(
            linked, pointer_rows["partner_person_id"][0][np.maximum(found, 0)], -1
        )
        add(
            "person",
            "partner_person_id",
            "asymmetric",
            linked & (back != rows),
            "Partners must point at each other.",
        )

    first = pointer_rows.get("parent_1_person_id")
    second = pointer_rows.get("parent_2_person_id")
    if second is not None:
        first_present = first[1] if first is not None else np.zeros(len(rows), bool)
        add(
            "person",
            "parent_2_person_id",
            "parent_order",
            second[1] & ~first_present,
            "parent_2 is set while parent_1 is null.",
        )
        if first is not None:
            add(
                "person",
                "parent_2_person_id",
                "same_parent",
                (second[0] >= 0) & (second[0] == first[0]),
                "The two parents must differ.",
            )
    if "partner_person_id" in pointer_rows:
        partner_rows = pointer_rows["partner_person_id"][0]
        clash = np.zeros(len(rows), dtype=bool)
        for parent in (first, second):
            if parent is not None:
                clash |= (partner_rows >= 0) & (parent[0] == partner_rows)
        add(
            "person",
            "partner_person_id",
            "partner_is_parent",
            clash,
            "A person's partner cannot also be their parent.",
        )
    parent_edges = [parent[0] for parent in (first, second) if parent is not None]
    cyclic = _parent_cycle_members(len(rows), parent_edges)
    if cyclic:
        out.append(
            ConceptViolation(
                "person",
                "parent_1_person_id",
                "parent_cycle",
                "Parent pointers form a cycle.",
                cyclic,
            )
        )

    if "reference_person_id" in household.columns and _is_integer_column(
        household["reference_person_id"]
    ):
        found, present = positions(household["reference_person_id"])
        own = household[_HOUSEHOLD_ID].to_numpy()
        member = (found >= 0) & (own_household[np.maximum(found, 0)] == own)
        add(
            "household",
            "reference_person_id",
            "not_member",
            present & ~member,
            "The reference person must belong to the household.",
        )
    return out


def _is_integer_column(values: pd.Series) -> bool:
    # NumPy integer dtypes and pandas' nullable Int64 both report kind "i".
    return values.dtype.kind in "iu"


def _parent_cycle_members(n_persons: int, parent_rows: list[np.ndarray]) -> int:
    """Count persons on a cycle of parent pointers, or on a path between two.

    Kahn's algorithm peels the child-to-parent graph from both ends: first
    persons nobody names as a parent, then persons who name no parent. A
    person left after both peels lies on a cycle or on a path joining
    cycles; an acyclic graph peels completely.
    """

    child_parts: list[np.ndarray] = []
    parent_parts: list[np.ndarray] = []
    for found in parent_rows:
        linked = found >= 0
        child_parts.append(np.flatnonzero(linked))
        parent_parts.append(found[linked])
    if not child_parts:
        return 0
    child = np.concatenate(child_parts)
    parent = np.concatenate(parent_parts)
    from_children = _unpeeled(n_persons, source=child, target=parent)
    from_parents = _unpeeled(n_persons, source=parent, target=child)
    return int((from_children & from_parents).sum())


def _unpeeled(n: int, *, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Nodes Kahn's algorithm cannot remove from edges ``source -> target``."""

    remaining = np.bincount(target, minlength=n)
    order = np.argsort(source, kind="stable")
    ordered_source, ordered_target = source[order], target[order]
    starts = np.searchsorted(ordered_source, np.arange(n + 1))
    ready = list(np.flatnonzero(remaining == 0))
    left = np.ones(n, dtype=bool)
    while ready:
        node = ready.pop()
        left[node] = False
        for nxt in ordered_target[starts[node] : starts[node + 1]]:
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                ready.append(int(nxt))
    return left


def split_for_transport(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[str, ...]]]:
    """Keep only the columns transport may carry across countries.

    Transport drops every donor-country program receipt (concepts whose rule
    is :attr:`TransportRule.DROP`) and every column the schema does not
    describe, since an extension column's meaning is tied to the donor
    country. Id and membership columns stay.

    Returns:
        The kept tables and, per entity, the dropped column names.
    """

    structural = {
        "person": {_PERSON_ID, _PERSON_HOUSEHOLD_ID},
        "household": {_HOUSEHOLD_ID},
    }
    kept: dict[str, pd.DataFrame] = {}
    dropped: dict[str, tuple[str, ...]] = {}
    for entity in CONCEPT_ENTITIES:
        table = tables[entity]
        keep: list[str] = []
        drop: list[str] = []
        for column in table.columns:
            item = concept_for_column(entity, column)
            if column in structural[entity] or (
                item is not None and item.transport is not TransportRule.DROP
            ):
                keep.append(column)
            else:
                drop.append(column)
        kept[entity] = table.loc[:, keep].copy()
        dropped[entity] = tuple(drop)
    return kept, dropped


def derive_take_up_draws(
    seeds: Iterable[float] | np.ndarray, program: str
) -> np.ndarray:
    """Derive one program's take-up draws from persistent take-up seeds.

    Each draw is a BLAKE2b hash of the seed's exact float64 bits and the
    program key, scaled to [0, 1). The derivation is deterministic, so a
    record keeps its draw for a program across rebuilds and across countries;
    different program keys give unrelated draws, so programs are taken up
    independently given the seed. A program is taken up when its draw falls
    below the program's rate.

    Args:
        seeds: Persistent seeds on [0, 1).
        program: A non-empty program key naming the country and program
            (``"us.snap"``), so two countries' programs never share draws.

    Raises:
        ValueError: If ``program`` is empty or a seed lies outside [0, 1).
    """

    if not program:
        raise ValueError("A take-up program key must be non-empty.")
    # Adding zero turns -0.0 into 0.0, so equal seeds always hash alike.
    values = np.asarray(seeds, dtype=np.float64) + 0.0
    if values.ndim != 1:
        raise ValueError("Take-up seeds must be one-dimensional.")
    if len(values) and not (np.all(values >= 0.0) and np.all(values < 1.0)):
        raise ValueError("Take-up seeds must lie on [0, 1).")
    key = program.encode("utf-8")
    denominator = float(2**64)
    draws = np.empty(len(values), dtype=np.float64)
    for index, bits in enumerate(values.view(np.uint64).tolist()):
        digest = _blake2b_u64(bits.to_bytes(8, "big") + b"\x00" + key)
        draws[index] = digest / denominator
    # 2**64 - 1 over 2**64 rounds to exactly 1.0 in float64; keep [0, 1).
    np.minimum(draws, math.nextafter(1.0, 0.0), out=draws)
    return draws


def _blake2b_u64(payload: bytes) -> int:
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")
