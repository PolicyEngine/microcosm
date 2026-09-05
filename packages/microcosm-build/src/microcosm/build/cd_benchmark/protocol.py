"""Compile the approved canonical CD published-benchmark protocol document.

The exact approved bytes ship as a packaged resource beside this module. Nothing
here reads a path named inside the document: ``/identity/legacy_design_parents``,
``/reference_authority`` and the code locators are *locators* recording what was
inspected when the contract was written, never inputs this module opens.

Compilation is total and closed. Every threshold is carried as an exact
:class:`~decimal.Decimal` parsed from the document's decimal string, never as a
float; every vocabulary is compared against the document rather than assumed;
unknown block shapes refuse compilation. :func:`compile_protocol` will compile
any candidate document so that a successor can be inspected, but only bytes
whose SHA-256 equals :data:`APPROVED_PROTOCOL_SHA256` compile to a protocol with
``canonical`` set, and approval is resolved from that digest through an injected
authority (:mod:`microcosm.build.cd_benchmark.approval`) — never from a field
inside the document and never from a caller's label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Any

from microcosm.build.cd_benchmark.canonical import (
    CanonicalJsonError,
    digest,
    is_sha256_hex,
    strict_load,
)
from microcosm.build.cd_benchmark.reasons import ExposureStatus, Reason

__all__ = [
    "APPROVED_PROTOCOL_BYTES",
    "APPROVED_PROTOCOL_ID",
    "APPROVED_PROTOCOL_SHA256",
    "CompiledProtocol",
    "DistributionBlock",
    "PartitionRules",
    "PointBlock",
    "PrecisionKind",
    "ProtocolError",
    "ReferenceBin",
    "RESERVED_FAMILIES",
    "ScopeRules",
    "SupportBounds",
    "assert_vocabulary_closed",
    "assign_bin",
    "canonical_protocol",
    "compile_protocol",
    "load_canonical_bytes",
]

#: Digest and byte count of the approved canonical document, verified against
#: the review directory's plan, its independent response and the peer review.
APPROVED_PROTOCOL_SHA256 = (
    "f7574a2a00734ce64ceb9c101ebc52469a176d37ee32b02c140d3ea2e6c568ea"
)
APPROVED_PROTOCOL_BYTES = 86626
APPROVED_PROTOCOL_ID = "us-cd-published-benchmark-v4"
_RESOURCE = "us-cd-published-benchmark-v4.json"

#: ``/exposure_ledger/reservation``: these table families and every derivative
#: stay out of fitting, tuning and selection.
RESERVED_FAMILIES = frozenset({"B19001", "B25003"})

_FAMILY_ROOT = re.compile(r"([A-Z][0-9]{4,5})(?:_|\Z)")


class ProtocolError(ValueError):
    """The document is not a compilable CD published-benchmark contract."""


class PrecisionKind(StrEnum):
    """Which ``/reference_status/share_precision`` rule a distribution uses."""

    PUBLISHED_PERCENTAGE_MOE = "published_percentage_moe"
    SUBSET_SHARE_APPROXIMATION = "subset_share_approximation"


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object.")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} must be a nonempty string.")
    return value


def _flag(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{label} must be a JSON boolean.")
    return value


def _decimal(value: object, label: str) -> Decimal:
    """Parse an exact decimal-string threshold; floats are refused."""
    if not isinstance(value, str):
        raise ProtocolError(f"{label} must be an exact decimal string.")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise ProtocolError(f"{label} is not a decimal: {value!r}.") from None
    if not parsed.is_finite():
        raise ProtocolError(f"{label} must be finite.")
    return parsed


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{label} must be a JSON integer.")
    return value


def _bound(value: object, label: str) -> Decimal | None:
    """A bin edge: an exact integer, or ``None`` for an open side."""
    if value is None:
        return None
    return Decimal(_integer(value, label))


@dataclass(frozen=True)
class ReferenceBin:
    """One published bin, or the protocol-constructed ``unclassified`` residual.

    ``lower_inclusive``/``upper_exclusive`` follow the document's boundary rule:
    lower inclusive, upper exclusive, ``None`` meaning minus/plus infinity. The
    residual bin carries no publisher variable and a reference share of exactly
    zero; it is a protocol construct, not a publisher cell.
    """

    id: str
    publisher_variable: str | None
    publisher_label: str | None
    lower_inclusive: Decimal | None
    upper_exclusive: Decimal | None
    reference_share: Decimal | None
    unclassified: bool
    authority: str | None


@dataclass(frozen=True)
class PointBlock:
    """A mandatory scalar block compared by relative discrepancy."""

    id: str
    family: str
    dataset: str
    unit: str
    universe: str
    mandatory: bool
    rule: str
    variables: tuple[str, ...]
    point_relative_bound: Decimal
    reference_relative_moe_ceiling: Decimal
    controlled_reference: str | None
    exposure_families: frozenset[str]
    count_once: bool
    duplicate_reference_aliases: tuple[str, ...]

    @property
    def reserved(self) -> bool:
        """Whether any exposure family is a reserved holdout family."""
        return bool(self.exposure_families & RESERVED_FAMILIES)


@dataclass(frozen=True)
class DistributionBlock:
    """A mandatory categorical block compared by per-bin share and total variation.

    ``independent_votes`` is ``False`` where the document says so (tenure's owner
    and renter bins) and ``None`` where it says nothing. Undeclared is not a
    licence to treat bins as independent votes: ``/claim`` withholds that
    licence for overlapping cells, and ``/status_contract/no_rescue`` forbids
    rescuing a block with them.
    """

    id: str
    family: str
    dataset: str
    unit: str
    mandatory: bool
    denominator_variable: str
    bins: tuple[ReferenceBin, ...]
    max_each_bin_discrepancy_pp: Decimal
    max_total_variation: Decimal
    reference_moe_ceiling_pp: Decimal
    precision_kind: PrecisionKind
    precision_variables: tuple[str, ...]
    exposure_families: frozenset[str]
    independent_votes: bool | None
    open_upper: bool
    boundary_rule: str | None
    additional_bounds: MappingProxyType
    complement_only_when: str | None

    @property
    def reserved(self) -> bool:
        """Whether any exposure family is a reserved holdout family."""
        return bool(self.exposure_families & RESERVED_FAMILIES)

    @property
    def unclassified_bin(self) -> ReferenceBin:
        """The protocol-constructed residual bin."""
        return next(item for item in self.bins if item.unclassified)

    @property
    def known_bins(self) -> tuple[ReferenceBin, ...]:
        """Published bins, excluding the residual."""
        return tuple(item for item in self.bins if not item.unclassified)


@dataclass(frozen=True)
class SupportBounds:
    """``/support`` floors and ceilings, as exact decimals and integers."""

    minimum_positive_source_household_origins: int
    minimum_positive_origins_per_positive_reference_bin: int
    minimum_origin_cluster_concentration_ess: Decimal
    maximum_origin_denominator_share: Decimal
    maximum_origin_positive_bin_share: Decimal
    nonnegative_finite_weights_required: bool
    required_for_every_district_and_relevant_denominator: bool


@dataclass(frozen=True)
class PartitionRules:
    """``/categorical_partition`` closure and residual-mass rules."""

    maximum_unclassified_fraction: Decimal
    comparison: str
    renormalize_known_bins: bool
    unknown_bin_id: str
    applies_to: frozenset[str]


@dataclass(frozen=True)
class ScopeRules:
    """``/scope`` — the predeclared geographic and product scope."""

    country: str
    area: str
    congress: int
    district_count: int
    acs_year: int
    acs_product: str
    puerto_rico: str
    allowed_missing_mandatory_block_fraction: Decimal
    allowed_missing_mandatory_district_fraction: Decimal
    dc_canonical_id: str
    dc_published_id: str
    published_and_canonical_geography_ids_retained: bool


@dataclass(frozen=True)
class CompiledProtocol:
    """A compiled contract document plus the digest of the bytes it came from."""

    sha256: str
    size_bytes: int
    protocol_id: str
    schema: str
    status: str
    parent_sha256: str
    hash_scope: str
    point_blocks: tuple[PointBlock, ...]
    distribution_blocks: tuple[DistributionBlock, ...]
    partition: PartitionRules
    support: SupportBounds
    scope: ScopeRules
    reason_codes: frozenset[str]
    exposure_status_values: frozenset[str]
    independent_axes: tuple[str, ...]
    scope_reduction_order: tuple[str, ...]
    scoring_permitted_in_document: bool
    approval_installed_in_document: bool

    @property
    def canonical(self) -> bool:
        """Whether these bytes are the exact approved canonical document."""
        return (
            self.sha256 == APPROVED_PROTOCOL_SHA256
            and self.size_bytes == APPROVED_PROTOCOL_BYTES
        )

    @property
    def blocks(self) -> tuple[PointBlock | DistributionBlock, ...]:
        """Every mandatory block, point blocks first, in document order."""
        return (*self.point_blocks, *self.distribution_blocks)

    def block(self, block_id: str) -> PointBlock | DistributionBlock:
        """Return the block with ``block_id``.

        Raises:
            ProtocolError: If no block carries that id.
        """
        for item in self.blocks:
            if item.id == block_id:
                return item
        raise ProtocolError(f"Unknown block id {block_id!r}.")


def _families(block: dict[str, Any]) -> frozenset[str]:
    """Reference table-family roots whose exposure this block consumes."""
    declared = block.get("exposure_families")
    if declared is not None:
        if not isinstance(declared, list) or not declared:
            raise ProtocolError("exposure_families must be a nonempty list.")
        return frozenset(_text(item, "exposure family") for item in declared)
    match = _FAMILY_ROOT.match(_text(block.get("family"), "block family"))
    if match is None:
        raise ProtocolError(
            f"Block family {block.get('family')!r} names no published table root "
            "and declares no exposure_families."
        )
    return frozenset({match.group(1)})


def _bin(item: object) -> ReferenceBin:
    data = _object(item, "distribution bin")
    identifier = _text(data.get("id"), "bin id")
    unclassified = "authority" in data
    if unclassified:
        if data.get("publisher_variable") is not None:
            raise ProtocolError("The residual bin cannot name a publisher variable.")
        share = _decimal(data.get("reference_share"), "residual reference_share")
        if share != 0:
            raise ProtocolError("The residual bin's reference share must be zero.")
        return ReferenceBin(
            id=identifier,
            publisher_variable=None,
            publisher_label=None,
            lower_inclusive=None,
            upper_exclusive=None,
            reference_share=share,
            unclassified=True,
            authority=_text(data.get("authority"), "residual bin authority"),
        )
    variable = (
        data["publisher_variable"] if "publisher_variable" in data else identifier
    )
    return ReferenceBin(
        id=identifier,
        publisher_variable=_text(variable, "bin publisher variable"),
        publisher_label=data.get("publisher_label"),
        lower_inclusive=_bound(data.get("lower_inclusive"), "bin lower_inclusive"),
        upper_exclusive=_bound(data.get("upper_exclusive"), "bin upper_exclusive"),
        reference_share=None,
        unclassified=False,
        authority=None,
    )


def _point_block(block: dict[str, Any]) -> PointBlock:
    ceilings = [
        key
        for key in (
            "ordinary_reference_relative_moe_ceiling",
            "reference_relative_moe_ceiling",
        )
        if key in block
    ]
    if len(ceilings) != 1:
        raise ProtocolError(
            "A point block declares exactly one relative reference MOE ceiling."
        )
    variables = block.get("variables")
    if not isinstance(variables, list) or not variables:
        raise ProtocolError("A point block declares its reference variables.")
    aliases = block.get("duplicate_reference_aliases", [])
    if not isinstance(aliases, list):
        raise ProtocolError("duplicate_reference_aliases must be a list.")
    return PointBlock(
        id=_text(block.get("id"), "block id"),
        family=_text(block.get("family"), "block family"),
        dataset=_text(block.get("dataset"), "block dataset"),
        unit=_text(block.get("unit"), "block unit"),
        universe=_text(block.get("universe"), "block universe"),
        mandatory=_flag(block.get("mandatory"), "block mandatory"),
        rule=_text(block.get("rule"), "block rule"),
        variables=tuple(_text(item, "block variable") for item in variables),
        point_relative_bound=_decimal(
            block.get("point_relative_bound"), "point_relative_bound"
        ),
        reference_relative_moe_ceiling=_decimal(block[ceilings[0]], ceilings[0]),
        controlled_reference=block.get("controlled_reference"),
        exposure_families=_families(block),
        count_once=bool(block.get("count_once", False)),
        duplicate_reference_aliases=tuple(
            _text(item, "duplicate reference alias") for item in aliases
        ),
    )


_PRECISION_CEILINGS = {
    "reference_published_moe_ceiling_pp": PrecisionKind.PUBLISHED_PERCENTAGE_MOE,
    "reference_approximate_moe_ceiling_pp": PrecisionKind.SUBSET_SHARE_APPROXIMATION,
}


def _distribution_block(block: dict[str, Any]) -> DistributionBlock:
    present = [key for key in _PRECISION_CEILINGS if key in block]
    if len(present) != 1:
        raise ProtocolError(
            "A distribution block declares exactly one reference precision ceiling."
        )
    bins = tuple(_bin(item) for item in block.get("bins", []))
    if len(bins) < 2 or sum(item.unclassified for item in bins) != 1:
        raise ProtocolError("A distribution declares its bins plus one residual bin.")
    if not bins[-1].unclassified:
        raise ProtocolError("The residual bin is declared last.")
    identifiers = [item.id for item in bins]
    if len(set(identifiers)) != len(identifiers):
        raise ProtocolError("Duplicate bin ids.")
    extra = {
        key: _decimal(value, key)
        for key, value in block.items()
        if key.startswith("max_")
        and key.endswith("_pp")
        and key not in {"max_each_bin_discrepancy_pp"}
    }
    return DistributionBlock(
        id=_text(block.get("id"), "block id"),
        family=_text(block.get("family"), "block family"),
        dataset=_text(block.get("dataset"), "block dataset"),
        unit=_text(block.get("unit"), "block unit"),
        mandatory=_flag(block.get("mandatory"), "block mandatory"),
        denominator_variable=_text(block.get("denominator"), "block denominator"),
        bins=bins,
        max_each_bin_discrepancy_pp=_decimal(
            block.get("max_each_bin_discrepancy_pp"), "max_each_bin_discrepancy_pp"
        ),
        max_total_variation=_decimal(
            block.get("max_total_variation"), "max_total_variation"
        ),
        reference_moe_ceiling_pp=_decimal(block[present[0]], present[0]),
        precision_kind=_PRECISION_CEILINGS[present[0]],
        precision_variables=tuple(
            _text(item, "precision variable")
            for item in block.get("precision_variables", [])
        ),
        exposure_families=_families(block),
        independent_votes=block.get("owner_and_renter_independent_votes"),
        open_upper=bool(block.get("open_upper_age", False)),
        boundary_rule=block.get("boundary_rule"),
        additional_bounds=MappingProxyType(extra),
        complement_only_when=block.get("owner_is_renter_complement_only_when"),
    )


def compile_protocol(raw: bytes) -> CompiledProtocol:
    """Compile canonical protocol bytes into typed, exact rules.

    Args:
        raw: The exact document bytes. Their SHA-256 is the protocol identity;
            the document carries no self-hash member (``/identity/hash_scope``).

    Raises:
        ProtocolError: On any missing, mistyped or unknown-shaped section.
    """
    try:
        document = _object(strict_load(raw), "protocol document")
    except CanonicalJsonError as error:
        raise ProtocolError(str(error)) from None
    identity = _object(document.get("identity"), "/identity")
    if identity.get("algorithm") != "sha256":
        raise ProtocolError("Only sha256 protocol identity is supported.")
    parent = _text(identity.get("protocol_parent_sha256"), "protocol_parent_sha256")
    if not is_sha256_hex(parent):
        raise ProtocolError("protocol_parent_sha256 must be a lowercase digest.")
    blocks = document.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ProtocolError("/blocks must be a nonempty list.")
    point, distribution = [], []
    for item in blocks:
        block = _object(item, "block")
        if "bins" in block:
            distribution.append(_distribution_block(block))
        else:
            point.append(_point_block(block))
    identifiers = [item.id for item in (*point, *distribution)]
    if len(set(identifiers)) != len(identifiers):
        raise ProtocolError("Duplicate block ids.")

    partition_doc = _object(
        document.get("categorical_partition"), "/categorical_partition"
    )
    unknown_bin = _object(partition_doc.get("unknown_bin"), "unknown_bin")
    partition = PartitionRules(
        maximum_unclassified_fraction=_decimal(
            partition_doc.get("maximum_unclassified_fraction_of_known_denominator"),
            "maximum_unclassified_fraction_of_known_denominator",
        ),
        comparison=_text(
            partition_doc.get("maximum_unclassified_fraction_comparison"),
            "maximum_unclassified_fraction_comparison",
        ),
        renormalize_known_bins=_flag(
            partition_doc.get("renormalize_known_bins"), "renormalize_known_bins"
        ),
        unknown_bin_id=_text(unknown_bin.get("id"), "unknown bin id"),
        applies_to=frozenset(
            _text(item, "partition applies_to") for item in partition_doc["applies_to"]
        ),
    )
    support_doc = _object(document.get("support"), "/support")
    support = SupportBounds(
        minimum_positive_source_household_origins=_integer(
            support_doc.get("minimum_positive_source_household_origins"),
            "minimum_positive_source_household_origins",
        ),
        minimum_positive_origins_per_positive_reference_bin=_integer(
            support_doc.get("minimum_positive_origins_per_positive_reference_bin"),
            "minimum_positive_origins_per_positive_reference_bin",
        ),
        minimum_origin_cluster_concentration_ess=_decimal(
            support_doc.get("minimum_origin_cluster_concentration_ESS"),
            "minimum_origin_cluster_concentration_ESS",
        ),
        maximum_origin_denominator_share=_decimal(
            support_doc.get("maximum_origin_denominator_share"),
            "maximum_origin_denominator_share",
        ),
        maximum_origin_positive_bin_share=_decimal(
            support_doc.get("maximum_origin_positive_bin_share"),
            "maximum_origin_positive_bin_share",
        ),
        nonnegative_finite_weights_required=_flag(
            support_doc.get("nonnegative_finite_weights_required"),
            "nonnegative_finite_weights_required",
        ),
        required_for_every_district_and_relevant_denominator=_flag(
            support_doc.get("required_for_every_district_and_relevant_denominator"),
            "required_for_every_district_and_relevant_denominator",
        ),
    )
    scope_doc = _object(document.get("scope"), "/scope")
    alias = _object(scope_doc.get("DC_identity_alias"), "DC_identity_alias")
    scope = ScopeRules(
        country=_text(scope_doc.get("country"), "scope country"),
        area=_text(scope_doc.get("area"), "scope area"),
        congress=_integer(scope_doc.get("congress"), "scope congress"),
        district_count=_integer(scope_doc.get("district_count"), "district_count"),
        acs_year=_integer(scope_doc.get("acs_year"), "acs_year"),
        acs_product=_text(scope_doc.get("acs_product"), "acs_product"),
        puerto_rico=_text(scope_doc.get("puerto_rico"), "puerto_rico"),
        allowed_missing_mandatory_block_fraction=_decimal(
            scope_doc.get("allowed_missing_mandatory_block_fraction"),
            "allowed_missing_mandatory_block_fraction",
        ),
        allowed_missing_mandatory_district_fraction=_decimal(
            scope_doc.get("allowed_missing_mandatory_district_fraction"),
            "allowed_missing_mandatory_district_fraction",
        ),
        dc_canonical_id=_text(alias.get("canonical"), "DC canonical id"),
        dc_published_id=_text(alias.get("published"), "DC published id"),
        published_and_canonical_geography_ids_retained=_flag(
            scope_doc.get("published_and_canonical_geography_ids_retained"),
            "published_and_canonical_geography_ids_retained",
        ),
    )
    status_doc = _object(document.get("status_contract"), "/status_contract")
    ledger_doc = _object(document.get("exposure_ledger"), "/exposure_ledger")
    approval_doc = _object(document.get("approval"), "/approval")
    return CompiledProtocol(
        sha256=digest(raw),
        size_bytes=len(raw),
        protocol_id=_text(document.get("protocol_id"), "/protocol_id"),
        schema=_text(document.get("schema"), "/schema"),
        status=_text(document.get("status"), "/status"),
        parent_sha256=parent,
        hash_scope=_text(identity.get("hash_scope"), "/identity/hash_scope"),
        point_blocks=tuple(point),
        distribution_blocks=tuple(distribution),
        partition=partition,
        support=support,
        scope=scope,
        reason_codes=frozenset(_object(status_doc.get("reason_codes"), "reason_codes")),
        exposure_status_values=frozenset(
            _object(ledger_doc.get("exposure_status_values"), "exposure_status_values")
        ),
        independent_axes=tuple(status_doc["independent_axes_required"]),
        scope_reduction_order=tuple(status_doc["scope_reduction_order"]),
        scoring_permitted_in_document=_flag(
            approval_doc.get("scoring_permitted"), "/approval/scoring_permitted"
        ),
        approval_installed_in_document=_flag(
            approval_doc.get("installed"), "/approval/installed"
        ),
    )


def load_canonical_bytes() -> bytes:
    """Read the packaged approved document and verify its digest and size.

    Raises:
        ProtocolError: If the packaged bytes are not the approved document.
    """
    raw = resources.files(__package__).joinpath(_RESOURCE).read_bytes()
    if len(raw) != APPROVED_PROTOCOL_BYTES or digest(raw) != APPROVED_PROTOCOL_SHA256:
        raise ProtocolError("Packaged protocol resource is not the approved document.")
    return raw


@lru_cache(maxsize=1)
def canonical_protocol() -> CompiledProtocol:
    """Return the compiled approved canonical protocol.

    Raises:
        ProtocolError: If the packaged document is not the approved bytes, does
            not identify itself as the approved protocol, or has drifted from
            the closed reason/exposure vocabularies this package implements.
    """
    compiled = compile_protocol(load_canonical_bytes())
    if not compiled.canonical or compiled.protocol_id != APPROVED_PROTOCOL_ID:
        raise ProtocolError("Packaged protocol is not the approved canonical contract.")
    assert_vocabulary_closed(compiled)
    return compiled


def assert_vocabulary_closed(compiled: CompiledProtocol) -> None:
    """Check this package's closed vocabularies equal the document's.

    Raises:
        ProtocolError: On any code this package would emit that the document
            does not define, or any document code this package cannot emit.
    """
    for name, implemented, declared in (
        ("reason", {item.value for item in Reason}, set(compiled.reason_codes)),
        (
            "exposure status",
            {item.value for item in ExposureStatus},
            set(compiled.exposure_status_values),
        ),
    ):
        if implemented != declared:
            raise ProtocolError(
                f"Closed {name} vocabulary differs from the document: "
                f"only in code {sorted(implemented - declared)}, "
                f"only in document {sorted(declared - implemented)}."
            )


def assign_bin(block: DistributionBlock, value: Decimal) -> str | None:
    """Return the bin id for ``value`` under the declared boundary rule.

    Lower bounds are inclusive and upper bounds exclusive; a ``None`` lower or
    upper bound is minus or plus infinity. No rounding is applied — a monetary
    amount of ``9999.5`` stays below a ``10000`` cutpoint. Returns ``None`` when
    no published bin covers the value, which the caller may treat as
    unclassified mass only under known membership and an approved mapping.
    """
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ProtocolError("Bin assignment needs a finite Decimal value.")
    for item in block.known_bins:
        if item.lower_inclusive is None and item.upper_exclusive is None:
            raise ProtocolError(f"Bin {item.id!r} declares no numeric boundary.")
        if item.lower_inclusive is not None and value < item.lower_inclusive:
            continue
        if item.upper_exclusive is not None and value >= item.upper_exclusive:
            continue
        return item.id
    return None
