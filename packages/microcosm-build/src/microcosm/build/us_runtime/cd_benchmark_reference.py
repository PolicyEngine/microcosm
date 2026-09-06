"""Project the frozen CD reference inventory into the protocol's typed values.

``microcosm.build.cd_benchmark`` compiles the approved
``us-cd-published-benchmark-v4`` document and implements every reducer, but by
design it "reads no candidate, population, HDF5, ACS or Census artifact"
(``docs/us-cd-published-benchmark-contract.md``). Its ``ReferenceCell`` and
``PrecisionResult`` inputs are caller-injected with no default. This module is
the missing caller for the *reference* half only: it reads the frozen
``cd_reference`` inventory that the contract pins by digest and turns one
explicitly requested block, for one explicitly requested district, into those
typed values.

It lives in ``us_runtime`` and not inside ``cd_benchmark`` precisely so the pure
package keeps that documented no-file-access property and its measured import
cost. The dependency arrow stays one way: this module imports the pure package;
the pure package never imports this one.

What this is not
----------------

Projection is reference reading. It is not scoring and it does not create a
scoring authorization: no ``ApprovalResolution``, ``BenchmarkVerdict`` or
``ExposureStatus`` is produced here, and nothing in this module can make
``evaluate_scope`` treat implementation approval as scoring approval. It
registers no target, writes no ledger, takes no lock, selects no threshold and
edits no bound — every bound used is compiled out of the approved document.

It also forms no national reference. District point sums are not published
national rows, no national margin of error can be inferred from district
margins, and ``/scope/national_published_reference_scope`` says a national claim
needs separately acquired direct national references and its own protocol.
There is deliberately no function here that sums districts.

Holdout discipline
------------------

``/exposure_ledger/reservation`` reserves the ``B19001`` and ``B25003`` table
families and every derivative. This module never sweeps blocks: every entry
point takes one explicit ``block_id``, every projected cell and precision
reading carries its ``reserved_family`` provenance, and the command-line
interface refuses a reserved block unless the operator names
``--include-reserved``. Reading a published reference is not fitting, tuning or
selection — but it is recorded as reserved so a downstream consumer cannot lose
that fact. No freshness or exposure verdict is computed or exposed.

Reuse, not reimplementation
---------------------------

Sentinel and annotation semantics come from the pinned adapter
:func:`microcosm.build.us_runtime.cd_reference.classify_acs_value`, reached
through :func:`microcosm.build.cd_benchmark.reference.reference_cell`. The three
precision readers, the district-scope check, the strict canonical JSON loader
and the reserved-family set are the shipped ones. Block and metric rosters are
derived from the compiled protocol rather than retyped, and no numeric bound
appears in this file.

What a caller cannot state
--------------------------

Only :func:`load_pinned_inventory` can record an inventory's identity, because
only it verifies bytes: it enforces the stated digest and the pinned byte count
against the file before anything is parsed.
:meth:`ReferenceInventory.from_document` is handed an already-parsed document,
so it records no digest and no byte count at all and refuses a caller who tries
to supply either. A parsed document is deep-copied into read-only mappings and
tuples, so neither the caller's original object nor the indexed inventory can be
changed afterwards: a projection is evidence about the bytes that were verified,
not about whatever the object happens to hold later.

Canonical document, canonical values
------------------------------------

The audited top-level key list, the audited ``candidate_evaluation`` scalar and
the pinned inventory byte count are read from the **packaged canonical protocol
document** and from nowhere else. A ``protocol`` argument selects blocks, scope
and bounds; it does not redirect those three. Rather than quietly attributing
canonical facts to some other contract, the two entry points that mix them —
:func:`load_pinned_inventory` and :meth:`ReferenceInventory.from_document` —
refuse a non-canonical :class:`CompiledProtocol`. Reading them out of an
arbitrary document instead would widen what this module treats as authority, and
that is a protocol-scope change, deliberately not made here.

Reserved tables are indexed, not read
-------------------------------------

``/scope/required_district_set`` requires an identical district set across the
reserved tables, so indexing validates the structure of ``B19001`` and
``B25003`` — reservation status, district records, published aliases, cell keys
and cell shape — even when the caller projects only ``S0101``. That is
structural validation of the bundle: no reserved estimate or margin is
classified, built into a :class:`ReferenceCell` or turned into a precision
reading unless a reserved block is explicitly named. The aggregate payload says
so in as many words, so a reader cannot infer from a non-reserved run that the
reserved bytes were never looked at.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from microcosm.build.cd_benchmark.canonical import (
    CanonicalJsonError,
    canonical_bytes,
    digest,
    is_sha256_hex,
    strict_load,
)
from microcosm.build.cd_benchmark.evaluation import (
    CLAIMS_NOT_ESTABLISHED,
    district_scope_reasons,
)
from microcosm.build.cd_benchmark.protocol import (
    RESERVED_FAMILIES,
    CompiledProtocol,
    DistributionBlock,
    PointBlock,
    PrecisionKind,
    canonical_protocol,
    load_canonical_bytes,
)
from microcosm.build.cd_benchmark.reasons import Reason, dedupe
from microcosm.build.cd_benchmark.reference import (
    PrecisionResult,
    ReferenceCell,
    ReferenceError,
    published_percentage_precision,
    reference_cell,
    reference_validity_reasons,
    relative_moe_precision,
    subset_share_precision,
)
from microcosm.build.cd_benchmark.universe import (
    PINNED_ADAPTER_MODULES,
    verify_pinned_adapters,
)

__all__ = [
    "BlockProjection",
    "CellRequest",
    "DistrictRoster",
    "PROJECTION_CLAIMS_NOT_ESTABLISHED",
    "PUBLISHED_PERCENTAGE_UNITS",
    "ProjectedCell",
    "ProjectedPrecision",
    "ProjectionCompleteness",
    "RESERVED_FAMILY_TABLES",
    "ReferenceInventory",
    "ReferenceProjectionError",
    "block_cell_roster",
    "cell_units",
    "cell_universe",
    "count_units",
    "district_roster",
    "inventory_candidate_evaluation",
    "inventory_top_level_keys",
    "is_reserved_family",
    "load_pinned_inventory",
    "main",
    "pinned_inventory_sha256",
    "pinned_inventory_size_bytes",
    "project_block",
    "project_block_cells",
    "project_block_precision",
    "projection_completeness",
    "require_pinned_adapters",
    "roster_scope_reasons",
    "table_for_variable",
]

#: ``/exposure_ledger/reservation``, as the compiled protocol package states it.
#: Re-exported rather than retyped so there is one reserved-family definition.
RESERVED_FAMILY_TABLES = RESERVED_FAMILIES

#: The published unit of the ACS subject-table percent columns that
#: ``/reference_status/share_precision/age`` names as the age block's precision
#: source. The composition check in
#: :func:`microcosm.build.cd_benchmark.evaluation.evaluate_block` requires this
#: exact unit on a published-percentage precision cell; a test pins that by
#: running the shipped evaluator rather than by restating its table here.
PUBLISHED_PERCENTAGE_UNITS = "percent"

#: The shape constants the producer :mod:`microcosm.build.us_runtime.cd_reference`
#: writes into every bundle. Learned from that producer, not from the inventory.
INVENTORY_KIND = "us_cd_reference_inventory"
INVENTORY_SCHEMA_VERSION = 1
INVENTORY_PURPOSE = "reference_inventory_only"
INVENTORY_LINEAGE = "published"
RESERVED_TABLE_STATUS = "reserved_same_source_holdout"
UNRESERVED_TABLE_STATUS = "inventory_only"

#: What a completed projection still does not establish. The pure package's own
#: list is reused verbatim and extended with the things projection specifically
#: must not be read as.
PROJECTION_CLAIMS_NOT_ESTABLISHED = (
    *CLAIMS_NOT_ESTABLISHED,
    "candidate scoring",
    "protocol scoring authorization",
    "exposure freshness or holdout eligibility",
    "target registration",
    "statistical readiness",
)

_COUNT_UNIT_SUFFIX = "_for_counts"

#: Cell roles. ``precision`` is only used where a block's precision rule reads a
#: separate published column; a subset-share block reuses its bin and
#: denominator cells and mints no extra role.
POINT_ROLE = "point"
DENOMINATOR_ROLE = "denominator"
BIN_ROLE = "bin"
PRECISION_ROLE = "precision"


class ReferenceProjectionError(ValueError):
    """The pinned inventory cannot be projected as the protocol requires."""


@lru_cache(maxsize=1)
def _document() -> Mapping[str, object]:
    """The approved protocol document, parsed from the packaged bytes."""
    return strict_load(load_canonical_bytes())


def _authority(*path: str) -> object:
    """Read a value out of the **packaged canonical** protocol document.

    This deliberately ignores any ``protocol`` argument in the caller's scope:
    the audit list, the audited scalar and the pinned byte count below are facts
    about the approved canonical contract, and re-reading them from some other
    document would make this module an authority on that document instead.
    :func:`_canonical_only` is what keeps the two from being mixed.
    """
    value: object = _document()["reference_authority"]
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ReferenceProjectionError(
                f"The protocol declares no /reference_authority/{'/'.join(path)}."
            )
        value = value[key]
    return value


def _canonical_only(protocol: CompiledProtocol) -> CompiledProtocol:
    """Refuse to mix canonical-document authority with a different contract.

    :func:`inventory_top_level_keys`, :func:`inventory_candidate_evaluation` and
    :func:`pinned_inventory_size_bytes` always read the packaged canonical
    document. Enforcing those against a bundle while a different compiled
    protocol supplies the blocks and scope would attribute canonical facts to a
    contract that never stated them, so it is refused instead of documented away.
    """
    if not isinstance(protocol, CompiledProtocol):
        raise ReferenceProjectionError("A compiled protocol is required.")
    if not protocol.canonical:
        raise ReferenceProjectionError(
            "The audited key list, the audited candidate_evaluation scalar and "
            "the pinned inventory byte count are read from the packaged "
            "canonical protocol document, so a non-canonical protocol "
            f"({protocol.sha256}) cannot be used to index an inventory."
        )
    return protocol


def pinned_inventory_sha256(protocol: CompiledProtocol | None = None) -> str:
    """The inventory digest the approved document pins, never retyped here.

    Unlike the three canonical-only readers below, this one *does* read the
    protocol it is given, because the digest is a compiled field of that
    document rather than a value taken from the packaged bytes.
    """
    return (protocol or canonical_protocol()).reference_inventory_sha256


def pinned_inventory_size_bytes() -> int:
    """The inventory byte count the approved **canonical** document pins.

    Canonical-only: there is no ``protocol`` argument because there is no other
    document this module will read a pinned size out of.
    """
    value = _authority("inventory", "bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceProjectionError("The pinned inventory size is not a count.")
    return value


def inventory_top_level_keys() -> frozenset[str]:
    """The inventory's top-level key set, as the document's own audit records it.

    Canonical-only, for the reason :func:`_authority` gives.
    """
    value = _authority("candidate_evaluation_scalar_audit", "top_level_keys")
    if not isinstance(value, list) or not all(isinstance(key, str) for key in value):
        raise ReferenceProjectionError("The audited top-level key list is not text.")
    if len(set(value)) != len(value):
        raise ReferenceProjectionError("The audited top-level key list repeats a key.")
    return frozenset(value)


def inventory_candidate_evaluation() -> str:
    """The audited ``candidate_evaluation`` scalar the inventory must carry.

    Canonical-only, for the reason :func:`_authority` gives.
    """
    value = _authority("candidate_evaluation_scalar_audit", "candidate_evaluation")
    if not isinstance(value, str) or not value:
        raise ReferenceProjectionError("The audited candidate_evaluation is not text.")
    return value


def is_reserved_family(block_id_or_table: str) -> bool:
    """Whether a table name, variable or block family names a reserved family.

    ``/exposure_ledger/reservation`` reserves the family and every derivative, so
    a variable inside a reserved table is reserved too.
    """
    if not isinstance(block_id_or_table, str) or not block_id_or_table:
        raise ReferenceProjectionError("A reserved-family probe needs a name.")
    if block_id_or_table in RESERVED_FAMILY_TABLES:
        return True
    return block_id_or_table.split("_")[0] in RESERVED_FAMILY_TABLES


def require_pinned_adapters(protocol: CompiledProtocol) -> Mapping[str, bool]:
    """Refuse to project unless the installed pure adapters match the pins.

    A changed adapter is a scope review, not a pin to refresh in passing, and a
    projection made through a drifted classifier would be evidence about nothing.
    """
    verdicts = verify_pinned_adapters(protocol)
    drifted = sorted(name for name, ok in verdicts.items() if not ok)
    if drifted:
        raise ReferenceProjectionError(
            "Installed adapters no longer match the contract's pins: "
            + ", ".join(drifted)
            + ". That is a scope review, not a projection."
        )
    return verdicts


def table_for_variable(variable: str) -> str:
    """The published table a variable belongs to.

    This is the same rule the shipped composition check applies
    (``cell.table == cell.variable.split("_")[0]``), so the projection cannot
    label a cell with a table the evaluator would reject.
    """
    if not isinstance(variable, str) or not variable.strip():
        raise ReferenceProjectionError("A published variable name is required.")
    return variable.split("_")[0]


def count_units(block: PointBlock | DistributionBlock) -> str:
    """The published count unit the document declares for this block.

    A point block declares one ``unit``. A distribution block declares a
    compound unit whose count segment ends in ``_for_counts``; that segment is
    read, never guessed.
    """
    if isinstance(block, PointBlock):
        return block.unit
    for part in block.unit.split(";"):
        candidate = part.strip()
        if candidate.endswith(_COUNT_UNIT_SUFFIX):
            return candidate[: -len(_COUNT_UNIT_SUFFIX)]
    raise ReferenceProjectionError(
        f"Block {block.id!r} declares no {_COUNT_UNIT_SUFFIX} unit segment."
    )


def cell_units(block: PointBlock | DistributionBlock, *, role: str) -> str:
    """The unit label a projected cell carries for this block and role."""
    if (
        role == PRECISION_ROLE
        and isinstance(block, DistributionBlock)
        and block.precision_kind is PrecisionKind.PUBLISHED_PERCENTAGE_MOE
    ):
        return PUBLISHED_PERCENTAGE_UNITS
    return count_units(block)


def cell_universe(
    protocol: CompiledProtocol, block: PointBlock | DistributionBlock
) -> str:
    """The universe label a projected cell carries for this block.

    A point block declares its ``universe``. A distribution block does not, so
    the universe is taken from the point block that publishes the very same
    denominator variable — ``age_distribution`` shares ``population_total``'s
    variable, and ``income_distribution``/``tenure_distribution`` share
    ``occupied_household_total``'s variable or its declared duplicate alias.
    Nothing is invented and no mapping table is retyped.
    """
    if isinstance(block, PointBlock):
        return block.universe
    owners = [
        point
        for point in protocol.point_blocks
        if block.denominator_variable
        in (*point.variables, *point.duplicate_reference_aliases)
    ]
    if len(owners) != 1:
        raise ReferenceProjectionError(
            f"Denominator {block.denominator_variable!r} of block {block.id!r} is "
            f"published by {len(owners)} point blocks; the universe is undetermined."
        )
    return owners[0].universe


@dataclass(frozen=True)
class CellRequest:
    """One published cell a block needs, with the role it plays in that block."""

    variable: str
    role: str
    bin_id: str | None


def block_cell_roster(
    protocol: CompiledProtocol, block: PointBlock | DistributionBlock
) -> tuple[CellRequest, ...]:
    """Every published cell one block reads, derived from the compiled protocol.

    A point block reads its declared variables. A distribution block reads its
    denominator, each published bin variable, and — only where its precision
    rule names a separate published percentage roster — those precision columns
    as well. A subset-share block reuses the bin and denominator cells it already
    reads, so no duplicate request is emitted.
    """
    requests: list[CellRequest] = []
    if isinstance(block, PointBlock):
        requests.extend(
            CellRequest(variable=variable, role=POINT_ROLE, bin_id=None)
            for variable in block.variables
        )
    else:
        requests.append(
            CellRequest(
                variable=block.denominator_variable,
                role=DENOMINATOR_ROLE,
                bin_id=None,
            )
        )
        requests.extend(
            CellRequest(variable=item.publisher_variable, role=BIN_ROLE, bin_id=item.id)
            for item in block.known_bins
        )
        if block.precision_kind is PrecisionKind.PUBLISHED_PERCENTAGE_MOE:
            requests.extend(
                CellRequest(variable=variable, role=PRECISION_ROLE, bin_id=None)
                for variable in block.precision_variables
            )
    seen: dict[str, CellRequest] = {}
    for request in requests:
        if request.variable in seen:
            raise ReferenceProjectionError(
                f"Block {block.id!r} requests {request.variable!r} twice; a cell "
                "cannot be counted under two roles."
            )
        seen[request.variable] = request
    if not requests:
        raise ReferenceProjectionError(f"Block {block.id!r} requests no cell.")
    return tuple(requests)


def required_tables(protocol: CompiledProtocol) -> tuple[str, ...]:
    """The published tables every mandatory block depends on, sorted."""
    families: set[str] = set()
    for block in protocol.blocks:
        if block.mandatory:
            families |= set(block.exposure_families)
    if not families:
        raise ReferenceProjectionError("The protocol declares no mandatory block.")
    return tuple(sorted(families))


@dataclass(frozen=True)
class DistrictRoster:
    """The canonical district set and its published aliases, both retained."""

    geography_ids: tuple[str, ...]
    published_geography_ids: Mapping[str, str]
    tables: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.geography_ids)


@dataclass(frozen=True)
class ReferenceInventory(Mapping):
    """A validated, indexed view of one pinned reference inventory document.

    The document is exposed unchanged (this object *is* a mapping over it), and
    the indexes exist so a per-district projection does not rescan the bundle.
    Construction validates shape and cross-checks the document against the
    compiled protocol; it never repairs anything.
    """

    document: Mapping[str, object] = field(repr=False)
    tables: tuple[str, ...] = field()
    datasets: Mapping[str, str] = field()
    geographies: Mapping[str, Mapping[str, object]] = field(repr=False)
    published_geography_ids: Mapping[str, str] = field(repr=False)
    cells: Mapping[tuple[str, str, str], Mapping[str, object]] = field(
        compare=False, repr=False
    )
    sha256: str | None = None
    size_bytes: int | None = None

    @property
    def identity_verified(self) -> bool:
        """Whether this inventory's identity was checked, not merely asserted.

        True only when :func:`load_pinned_inventory` read the file, matched its
        size against the pinned byte count and its SHA-256 against the digest
        the caller stated, before parsing a byte. There is no other way to make
        it true, so a false value is an honest "these bytes were never checked",
        never a missing label on verified bytes.
        """
        return self.sha256 is not None

    def __getitem__(self, key: str) -> object:
        return self.document[key]

    def __iter__(self):
        return iter(self.document)

    def __len__(self) -> int:
        return len(self.document)

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, object],
        *,
        protocol: CompiledProtocol | None = None,
        **unverifiable: object,
    ) -> ReferenceInventory:
        """Validate and index an already-parsed inventory document.

        The result carries **no** digest and **no** byte count, because this
        constructor never saw a file and so can verify neither:
        :attr:`identity_verified` is false and the aggregate payload reports a
        null ``inventory_sha256``. Passing ``sha256=`` or ``size_bytes=`` is
        refused rather than believed — an identity nothing checked is worse in
        an evidence artifact than no identity at all. Use
        :func:`load_pinned_inventory` when the identity has to be established.

        The document is deep-copied into read-only containers, so a caller that
        keeps its own reference to the object passed in cannot change what this
        inventory was validated as afterwards.

        Raises:
            ReferenceProjectionError: On an attempt to state an unverified
                identity, on a non-canonical protocol, or on any shape,
                identity, duplicate or protocol-disagreement defect. Nothing is
                repaired or defaulted.
        """
        if unverifiable:
            raise ReferenceProjectionError(
                "from_document cannot label an inventory's identity: "
                f"{sorted(unverifiable)} would be recorded without ever being "
                "checked against a file. Only load_pinned_inventory, which "
                "verifies the bytes it reads, may state them."
            )
        protocol = _canonical_only(protocol or canonical_protocol())
        return _index_inventory(document, protocol, None, None)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ReferenceProjectionError(message)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReferenceProjectionError(f"{label} must be a JSON object.")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    """A JSON array, in the immutable form :func:`_frozen_json` leaves it in."""
    if not isinstance(value, tuple):
        raise ReferenceProjectionError(f"{label} must be a JSON array.")
    return value


def _frozen_json(value: object, label: str = "The inventory") -> object:
    """Deep-copy parsed JSON into containers nothing can change afterwards.

    Every object becomes a fresh :class:`~types.MappingProxyType` over a fresh
    dict, and every array becomes a tuple. The copy matters as much as the
    freeze: wrapping the caller's own dicts would leave them writable *behind*
    the proxy, so a fixture that kept a reference could still change an
    estimate's raw text after the digest was verified — the projected value
    would move while the file, its digest and the recorded identity all stood
    still. Copying severs that.

    JSON meaning is preserved exactly. Scalars are left as the parser produced
    them, ``null`` stays ``None``, key order is preserved, objects stay
    mappings, arrays stay ordered sequences, and both container kinds serialize
    back to the JSON they came from. Only mutability is removed.

    Anything that is not JSON data is refused rather than passed through, which
    keeps :meth:`ReferenceInventory.from_document` exactly as strict about shape
    as the byte-verifying loader: a tuple, set or arbitrary object handed in
    where the protocol requires a JSON array is a refusal, not a quiet
    acceptance.
    """
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReferenceProjectionError(
                    f"{label} has a JSON object key that is not text: {key!r}."
                )
            frozen[key] = _frozen_json(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(
            _frozen_json(item, f"{label}[{position}]")
            for position, item in enumerate(value)
        )
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ReferenceProjectionError(f"{label} is not JSON data: {type(value).__name__}.")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceProjectionError(f"{label} must be a nonempty string.")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceProjectionError(f"{label} must be a positive integer.")
    return value


def _datasets(document: Mapping[str, object]) -> Mapping[str, str]:
    """Table -> dataset, read from the bundle's own recorded source captures."""
    sources = _mapping(document.get("sources"), "sources")
    captures = _sequence(sources.get("census"), "sources.census")
    datasets: dict[str, str] = {}
    for item in captures:
        capture = _mapping(item, "a census capture")
        table = _text(capture.get("table"), "a census capture table")
        dataset = _text(capture.get("dataset"), "a census capture dataset")
        if datasets.setdefault(table, dataset) != dataset:
            raise ReferenceProjectionError(
                f"Table {table!r} is captured under two datasets."
            )
    _require(datasets, "The bundle records no census capture.")
    return MappingProxyType(datasets)


def _table_geographies(
    table_document: Mapping[str, object], table: str
) -> tuple[dict[str, Mapping[str, object]], dict[str, str]]:
    entries = _sequence(table_document.get("geographies"), f"{table}.geographies")
    identifiers = _sequence(
        table_document.get("geography_ids"), f"{table}.geography_ids"
    )
    geographies: dict[str, Mapping[str, object]] = {}
    published: dict[str, str] = {}
    for item in entries:
        record = _mapping(item, f"a {table} geography")
        canonical = _text(record.get("geography_id"), f"a {table} geography_id")
        if canonical in geographies:
            raise ReferenceProjectionError(
                f"Table {table!r} repeats district {canonical!r}."
            )
        geographies[canonical] = record
        published[canonical] = _text(
            record.get("published_geo_id"),
            f"the published GEO_ID of {canonical!r} in {table!r}",
        )
    if sorted(geographies) != list(identifiers):
        raise ReferenceProjectionError(
            f"Table {table!r} geography_ids disagree with its geography records."
        )
    _require(geographies, f"Table {table!r} carries no district.")
    return geographies, published


def _table_cells(
    table_document: Mapping[str, object], table: str
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    entries = _sequence(table_document.get("cells"), f"{table}.cells")
    cells: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for item in entries:
        record = _mapping(item, f"a {table} cell")
        canonical = _text(record.get("geography_id"), f"a {table} cell geography_id")
        variable = _text(record.get("variable"), f"a {table} cell variable")
        if table_for_variable(variable) != table:
            raise ReferenceProjectionError(
                f"Variable {variable!r} is filed under table {table!r}."
            )
        for side in ("estimate", "moe"):
            classified = _mapping(record.get(side), f"a {table} cell {side}")
            for key in ("raw", "annotation", "class", "annotation_conflict"):
                if key not in classified:
                    raise ReferenceProjectionError(
                        f"A {table} cell {side} is missing {key!r}."
                    )
        key = (table, canonical, variable)
        if key in cells:
            raise ReferenceProjectionError(
                f"Table {table!r} repeats cell {variable!r} for {canonical!r}."
            )
        cells[key] = record
    return cells


def _check_reservation(table_document: Mapping[str, object], table: str) -> None:
    reservation = _mapping(table_document.get("reservation"), f"{table}.reservation")
    reserved = is_reserved_family(table)
    expected = RESERVED_TABLE_STATUS if reserved else UNRESERVED_TABLE_STATUS
    _require(
        reservation.get("status") == expected,
        f"Table {table!r} declares reservation status "
        f"{reservation.get('status')!r}, not {expected!r}.",
    )
    _require(
        reservation.get("targets_allowed") is False
        and reservation.get("tuning_allowed") is False,
        f"Table {table!r} does not refuse targets and tuning.",
    )
    _require(
        reservation.get("covers_derived_versions") is True,
        f"Table {table!r} does not cover derived versions.",
    )


def _index_inventory(
    document: Mapping[str, object],
    protocol: CompiledProtocol,
    sha256: str | None,
    size_bytes: int | None,
) -> ReferenceInventory:
    """Validate a parsed inventory against the protocol and index it.

    The document is deep-frozen first, so every mapping the returned inventory
    exposes — the document itself, each table, each geography record, each cell
    and each cell's ``estimate``/``moe`` — is read-only and is this module's own
    copy. ``sha256`` and ``size_bytes`` are recorded only when the caller
    verified them against a file's bytes; :func:`load_pinned_inventory` is the
    one caller that can.

    Every required table is validated structurally, reserved families included,
    because ``/scope/required_district_set`` requires an identical district set
    across them. No reserved value is classified or read here; that happens only
    where a caller names a reserved block.
    """
    _canonical_only(protocol)
    document = _mapping(_frozen_json(document), "The inventory")
    expected_keys = inventory_top_level_keys()
    actual_keys = frozenset(document)
    if actual_keys != expected_keys:
        raise ReferenceProjectionError(
            "Inventory top-level keys disagree with the document's audit: "
            f"missing {sorted(expected_keys - actual_keys)}, "
            f"unexpected {sorted(actual_keys - expected_keys)}."
        )
    _require(
        document.get("kind") == INVENTORY_KIND,
        f"The bundle is not a {INVENTORY_KIND}.",
    )
    _require(
        document.get("schema_version") == INVENTORY_SCHEMA_VERSION,
        f"The bundle is not schema version {INVENTORY_SCHEMA_VERSION}.",
    )
    _require(
        document.get("purpose") == INVENTORY_PURPOSE,
        f"The bundle purpose is not {INVENTORY_PURPOSE!r}.",
    )
    _require(
        document.get("candidate_evaluation") == inventory_candidate_evaluation(),
        "The bundle does not record the audited candidate_evaluation scalar.",
    )
    _require(
        document.get("target_activation") is False,
        "The bundle does not record target_activation false.",
    )

    scope = _mapping(document.get("scope"), "scope")
    _require(scope.get("country") == protocol.scope.country, "Scope country differs.")
    _require(
        str(scope.get("states", "")).lower() == protocol.scope.area.lower(),
        "Scope area differs from the protocol's declared area.",
    )
    _require(
        str(scope.get("puerto_rico", "")).startswith("excluded"),
        "Scope does not exclude Puerto Rico.",
    )
    _require(
        scope.get("acs_year") == protocol.scope.acs_year,
        "Scope ACS year differs from the protocol.",
    )
    _require(
        scope.get("target_congress") == protocol.scope.congress,
        "Scope Congress differs from the protocol.",
    )

    datasets = _datasets(document)
    acs = _mapping(document.get("acs"), "acs")
    tables = required_tables(protocol)
    missing_tables = [table for table in tables if table not in acs]
    _require(
        not missing_tables,
        f"The bundle carries no ACS table {missing_tables}.",
    )

    geographies: dict[str, Mapping[str, object]] | None = None
    published: dict[str, str] | None = None
    cells: dict[tuple[str, str, str], Mapping[str, object]] = {}
    base_table = tables[0]
    for table in tables:
        table_document = _mapping(acs.get(table), f"acs.{table}")
        _require(
            table_document.get("table") == table,
            f"Table {table!r} is filed under the wrong key.",
        )
        _require(
            table_document.get("year") == protocol.scope.acs_year,
            f"Table {table!r} is not the protocol's ACS year.",
        )
        _require(
            table_document.get("congress") == protocol.scope.congress,
            f"Table {table!r} is not the protocol's Congress.",
        )
        _require(
            table_document.get("lineage") == INVENTORY_LINEAGE,
            f"Table {table!r} is not published lineage.",
        )
        _require(
            table in datasets,
            f"Table {table!r} has no recorded source capture.",
        )
        _check_reservation(table_document, table)
        table_geographies, table_published = _table_geographies(table_document, table)
        if geographies is None:
            geographies, published = table_geographies, table_published
        elif sorted(table_geographies) != sorted(geographies):
            raise ReferenceProjectionError(
                f"Table {table!r} district set differs from {base_table!r}; "
                "/scope/required_district_set requires an identical set."
            )
        elif table_published != published:
            raise ReferenceProjectionError(
                f"Table {table!r} published GEO_IDs differ from {base_table!r}."
            )
        cells.update(_table_cells(table_document, table))

    assert geographies is not None and published is not None
    return ReferenceInventory(
        document=document,
        tables=tables,
        datasets=datasets,
        geographies=MappingProxyType(geographies),
        published_geography_ids=MappingProxyType(published),
        cells=MappingProxyType(cells),
        sha256=sha256,
        size_bytes=size_bytes,
    )


def load_pinned_inventory(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
    protocol: CompiledProtocol | None = None,
) -> ReferenceInventory:
    """Load the pinned inventory, verifying its digest **before** parsing it.

    There is deliberately no default digest: an unpinned inventory is not a
    reference authority, so omitting ``expected_sha256`` is a ``TypeError`` and
    not a permissive load. When the caller states the digest the protocol pins,
    the protocol's pinned byte count is enforced too, and it is checked from the
    file's size before a single byte is read.

    Args:
        path: The inventory file.
        expected_sha256: Lowercase hex SHA-256 the file must have.
        expected_size_bytes: Byte count the file must have; defaults to the
            protocol's pinned count when ``expected_sha256`` is the pinned digest.
        protocol: Compiled protocol; the canonical one by default. It must be
            the canonical document, because the audited key list and the pinned
            byte count are read from the packaged canonical bytes regardless of
            what is passed here.

    Returns:
        A :class:`ReferenceInventory` whose ``sha256`` and ``size_bytes`` are
        the values actually observed on the file, so its ``identity_verified``
        is true. This is the only constructor that can say that.

    Raises:
        ReferenceProjectionError: On a non-canonical protocol, a bad digest
            argument, a missing file, a size or digest mismatch, malformed
            canonical JSON, or any shape or protocol disagreement. The bytes are
            never parsed on a mismatch.
    """
    protocol = _canonical_only(protocol or canonical_protocol())
    require_pinned_adapters(protocol)
    if not is_sha256_hex(expected_sha256):
        raise ReferenceProjectionError(
            "expected_sha256 must be a lowercase 64-character hex digest."
        )
    if expected_size_bytes is None and expected_sha256 == pinned_inventory_sha256(
        protocol
    ):
        expected_size_bytes = pinned_inventory_size_bytes()
    path = Path(path)
    if not path.is_file():
        raise ReferenceProjectionError(f"No inventory file at {path}.")
    actual_size = path.stat().st_size
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        raise ReferenceProjectionError(
            f"Inventory is {actual_size} bytes, not the pinned "
            f"{expected_size_bytes}; refusing before reading it."
        )
    data = path.read_bytes()
    actual_sha256 = digest(data)
    if actual_sha256 != expected_sha256:
        raise ReferenceProjectionError(
            f"Inventory digest {actual_sha256} does not match the pinned "
            f"{expected_sha256}; refusing before parsing it."
        )
    try:
        document = strict_load(data.rstrip(b"\n"))
    except CanonicalJsonError as error:
        raise ReferenceProjectionError(
            f"Inventory is not canonical JSON: {error}"
        ) from None
    return _index_inventory(document, protocol, actual_sha256, actual_size)


def district_roster(
    inventory: ReferenceInventory, *, protocol: CompiledProtocol | None = None
) -> DistrictRoster:
    """The canonical district identities and their published aliases.

    ``/scope/required_district_set`` selects the distinct canonical district
    identities and requires an identical set across the reserved tables; that
    identity is established when the inventory is indexed, so this only reads it.
    """
    if not isinstance(inventory, ReferenceInventory):
        raise ReferenceProjectionError(
            "district_roster needs a validated ReferenceInventory."
        )
    protocol = protocol or canonical_protocol()
    if tuple(sorted(required_tables(protocol))) != tuple(sorted(inventory.tables)):
        raise ReferenceProjectionError(
            "The inventory was indexed against a different table roster."
        )
    return DistrictRoster(
        geography_ids=tuple(sorted(inventory.geographies)),
        published_geography_ids=inventory.published_geography_ids,
        tables=inventory.tables,
    )


def roster_scope_reasons(
    roster: DistrictRoster,
    *,
    protocol: CompiledProtocol | None = None,
    present_geography_ids: Iterable[str] | None = None,
) -> tuple[tuple[Reason, ...], Mapping[str, object]]:
    """Run the shipped district-scope check over this roster.

    The required set is the inventory's own canonical roster; ``present`` is what
    a caller actually projected, defaulting to the whole roster. The DC canonical
    and published identities are both retained and checked exactly as
    :func:`microcosm.build.cd_benchmark.evaluation.district_scope_reasons` states.
    """
    protocol = protocol or canonical_protocol()
    present = (
        tuple(roster.geography_ids)
        if present_geography_ids is None
        else tuple(present_geography_ids)
    )
    return district_scope_reasons(
        protocol.scope,
        required_geography_ids=roster.geography_ids,
        present_geography_ids=present,
        published_geography_ids=roster.published_geography_ids,
    )


@dataclass(frozen=True)
class ProjectedCell:
    """One typed published cell plus the provenance the protocol requires kept."""

    cell: ReferenceCell
    role: str
    bin_id: str | None
    reserved_family: bool
    exposure_families: frozenset[str]

    @property
    def variable(self) -> str:
        return self.cell.variable


@dataclass(frozen=True)
class ProjectedPrecision:
    """One precision reading, unchanged, plus its provenance.

    ``result`` is the shipped :class:`PrecisionResult` exactly as the reader
    produced it — status, measured value, ceiling, reasons, evidence and source
    cells. This envelope adds provenance beside it and strips nothing.
    """

    result: PrecisionResult
    variables: tuple[str, ...]
    reserved_family: bool
    exposure_families: frozenset[str]

    @property
    def kind(self) -> str:
        return self.result.kind


@dataclass(frozen=True)
class BlockProjection:
    """One explicitly requested block, for one district, or an honest gap.

    A complete projection carries the block's full cell roster and its full
    precision roster. An incomplete one carries whatever cells exist, no
    precision at all (a partial roster is not a reading), and
    ``required_cell_or_district_missing`` with the exact missing variables in
    evidence. Nothing is defaulted, interpolated or borrowed from another
    district.
    """

    block_id: str
    geography_id: str
    published_geo_id: str | None
    cells: tuple[ProjectedCell, ...]
    precision: tuple[ProjectedPrecision, ...]
    reserved_family: bool
    exposure_families: frozenset[str]
    missing_variables: tuple[str, ...]
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether every required cell was found and every reading produced."""
        return not self.missing_variables

    @property
    def reference_cells(self) -> tuple[ReferenceCell, ...]:
        """The pure :class:`ReferenceCell` values, for the shipped evaluator."""
        return tuple(item.cell for item in self.cells)

    @property
    def precision_results(self) -> tuple[PrecisionResult, ...]:
        """The pure :class:`PrecisionResult` values, for the shipped evaluator."""
        return tuple(item.result for item in self.precision)


def _classified(record: Mapping[str, object], side: str) -> Mapping[str, object]:
    return _mapping(record.get(side), f"cell {side}")


def _build_cell(
    inventory: ReferenceInventory,
    protocol: CompiledProtocol,
    block: PointBlock | DistributionBlock,
    request: CellRequest,
    geography_id: str,
) -> ReferenceCell:
    table = table_for_variable(request.variable)
    record = inventory.cells[(table, geography_id, request.variable)]
    table_document = _mapping(inventory["acs"], "acs")[table]
    estimate = _classified(record, "estimate")
    moe = _classified(record, "moe")
    dataset = inventory.datasets[table]
    if dataset != block.dataset:
        raise ReferenceProjectionError(
            f"Table {table!r} was captured from dataset {dataset!r} but block "
            f"{block.id!r} declares {block.dataset!r}."
        )
    fields = {
        "E": estimate["raw"],
        "M": moe["raw"],
        "EA": estimate["annotation"],
        "MA": moe["annotation"],
        "published_GEO_ID": inventory.published_geography_ids[geography_id],
        "canonical_geography_id": geography_id,
        "dataset": dataset,
        "table": table,
        "variable": request.variable,
        "year": _positive_int(table_document["year"], f"{table}.year"),
        "congress": _positive_int(table_document["congress"], f"{table}.congress"),
        "units": cell_units(block, role=request.role),
        "universe": cell_universe(protocol, block),
    }
    try:
        built = reference_cell(fields)
    except ReferenceError as error:
        raise ReferenceProjectionError(
            f"Cell {request.variable!r} for {geography_id!r} is not a usable "
            f"published observation: {error}"
        ) from None
    _require(
        built.estimate.adapter_class == estimate["class"]
        and built.moe.adapter_class == moe["class"],
        f"The pinned adapter classifies {request.variable!r} for {geography_id!r} "
        "differently from the inventory; that is a scope review, not a projection.",
    )
    _require(
        built.estimate.annotation_conflict == bool(estimate["annotation_conflict"])
        and built.moe.annotation_conflict == bool(moe["annotation_conflict"]),
        f"Annotation-conflict status for {request.variable!r} at {geography_id!r} "
        "disagrees with the inventory.",
    )
    return built


def project_block_cells(
    inventory: ReferenceInventory,
    *,
    block_id: str,
    geography_id: str,
    protocol: CompiledProtocol | None = None,
) -> tuple[ReferenceCell, ...]:
    """Project one block's complete cell roster for one district.

    This is the strict form: an absent district or cell is a refusal, because a
    partial roster is not the block. Use :func:`project_block` when you want the
    complete missing result instead of the refusal.

    Raises:
        ReferenceProjectionError: If the district or any required cell is absent,
            or a cell cannot be read as the protocol requires.
    """
    protocol = protocol or canonical_protocol()
    block = protocol.block(block_id)
    found, missing = _gather(inventory, protocol, block, geography_id)
    if missing:
        raise ReferenceProjectionError(
            f"Block {block_id!r} at {geography_id!r} is missing published cells "
            f"{list(missing)}."
        )
    return tuple(item.cell for item in found)


def _gather(
    inventory: ReferenceInventory,
    protocol: CompiledProtocol,
    block: PointBlock | DistributionBlock,
    geography_id: str,
) -> tuple[tuple[ProjectedCell, ...], tuple[str, ...]]:
    if not isinstance(inventory, ReferenceInventory):
        raise ReferenceProjectionError("A validated ReferenceInventory is required.")
    if not isinstance(geography_id, str) or not geography_id.strip():
        raise ReferenceProjectionError("A canonical district identity is required.")
    roster = block_cell_roster(protocol, block)
    reserved = bool(block.exposure_families & RESERVED_FAMILY_TABLES)
    if geography_id not in inventory.geographies:
        return (), tuple(request.variable for request in roster)
    found: list[ProjectedCell] = []
    missing: list[str] = []
    for request in roster:
        table = table_for_variable(request.variable)
        if (table, geography_id, request.variable) not in inventory.cells:
            missing.append(request.variable)
            continue
        found.append(
            ProjectedCell(
                cell=_build_cell(inventory, protocol, block, request, geography_id),
                role=request.role,
                bin_id=request.bin_id,
                reserved_family=reserved or is_reserved_family(request.variable),
                exposure_families=block.exposure_families,
            )
        )
    return tuple(found), tuple(missing)


def project_block_precision(
    protocol: CompiledProtocol,
    cells: Iterable[ReferenceCell],
    *,
    block_id: str,
) -> tuple[PrecisionResult, ...]:
    """Read one block's complete precision roster from its projected cells.

    Which reader applies is the document's choice, compiled: a point block reads
    the relative margin of error against its relative ceiling; a distribution
    block reads either the published percentage column roster or the frozen
    subset-share approximation, each against its own percentage-point ceiling.
    No ceiling is retyped here and no reading is substituted for another.

    Raises:
        ReferenceProjectionError: If a required cell is absent from ``cells`` or
            a variable appears twice.
    """
    block = protocol.block(block_id)
    by_variable: dict[str, ReferenceCell] = {}
    for cell in cells:
        if not isinstance(cell, ReferenceCell):
            raise ReferenceProjectionError("Precision needs ReferenceCell inputs.")
        if cell.variable in by_variable:
            raise ReferenceProjectionError(
                f"Variable {cell.variable!r} was supplied twice."
            )
        by_variable[cell.variable] = cell

    def take(variable: str) -> ReferenceCell:
        if variable not in by_variable:
            raise ReferenceProjectionError(
                f"Block {block_id!r} precision needs cell {variable!r}."
            )
        return by_variable[variable]

    if isinstance(block, PointBlock):
        return tuple(
            relative_moe_precision(
                take(variable), ceiling=block.reference_relative_moe_ceiling
            )
            for variable in block.variables
        )
    if block.precision_kind is PrecisionKind.PUBLISHED_PERCENTAGE_MOE:
        return tuple(
            published_percentage_precision(
                take(variable), ceiling=block.reference_moe_ceiling_pp
            )
            for variable in block.precision_variables
        )
    denominator = take(block.denominator_variable)
    return tuple(
        subset_share_precision(
            take(item.publisher_variable),
            denominator,
            ceiling=block.reference_moe_ceiling_pp,
        )
        for item in block.known_bins
    )


def project_block(
    inventory: ReferenceInventory,
    *,
    block_id: str,
    geography_id: str,
    protocol: CompiledProtocol | None = None,
) -> BlockProjection:
    """Project one explicitly requested block for one district.

    There is no default block and no sweep: a caller that wants a reserved
    family must name it. The result carries the reserved-family provenance, the
    exposure families the block consumes, and — when the roster is incomplete —
    ``required_cell_or_district_missing`` with the exact missing variables,
    rather than a shorter roster that would read as complete.
    """
    protocol = protocol or canonical_protocol()
    block = protocol.block(block_id)
    reserved = bool(block.exposure_families & RESERVED_FAMILY_TABLES)
    found, missing = _gather(inventory, protocol, block, geography_id)
    published = inventory.published_geography_ids.get(geography_id)
    problems: list[Reason] = []
    for item in found:
        problems.extend(reference_validity_reasons(item.cell))
    precision: tuple[ProjectedPrecision, ...] = ()
    if missing:
        problems.append(Reason.REQUIRED_CELL_OR_DISTRICT_MISSING)
    else:
        results = project_block_precision(
            protocol, (item.cell for item in found), block_id=block_id
        )
        precision = tuple(
            ProjectedPrecision(
                result=result,
                variables=tuple(cell.variable for cell in result.cells),
                reserved_family=reserved
                or any(is_reserved_family(cell.variable) for cell in result.cells),
                exposure_families=block.exposure_families,
            )
            for result in results
        )
        for item in precision:
            problems.extend(item.result.reasons)
    expected_published = (
        protocol.scope.dc_published_id
        if geography_id == protocol.scope.dc_canonical_id
        else geography_id
    )
    alias_ok = published is not None and published == expected_published
    if published is not None and not alias_ok:
        problems.append(Reason.IDENTITY_MISMATCH)
    return BlockProjection(
        block_id=block.id,
        geography_id=geography_id,
        published_geo_id=published,
        cells=found,
        precision=precision,
        reserved_family=reserved,
        exposure_families=block.exposure_families,
        missing_variables=missing,
        reasons=dedupe(problems),
        evidence=MappingProxyType(
            {
                "mandatory": block.mandatory,
                "dataset": block.dataset,
                "requested_block_only": True,
                "expected_published_geo_id": expected_published,
                "published_and_canonical_geography_ids_retained": True,
                "reserved_exposure_families": tuple(
                    sorted(block.exposure_families & RESERVED_FAMILY_TABLES)
                ),
                "duplicate_reference_aliases_not_projected": tuple(
                    block.duplicate_reference_aliases
                )
                if isinstance(block, PointBlock)
                else (),
                "count_once": block.count_once
                if isinstance(block, PointBlock)
                else None,
                "precision_kind": None
                if isinstance(block, PointBlock)
                else str(block.precision_kind),
                "missing_variables": missing,
                "candidate_measurement": "not_read",
                "exposure_status": "not_determined_by_projection",
                "scoring_authorization": "not_created_by_projection",
            }
        ),
    )


@dataclass(frozen=True)
class ProjectionCompleteness:
    """Aggregate counts about a projection. Never a value, sum or verdict.

    Nothing here reports a published estimate, a candidate measurement, a
    national total or a benchmark outcome. It answers one question: how much of
    the reference the protocol requires is actually present and readable.
    """

    protocol_sha256: str
    inventory_sha256: str | None
    inventory_size_bytes: int | None
    inventory_identity_verified: bool
    roster_district_count: int
    districts_requested: tuple[str, ...]
    blocks: tuple[Mapping[str, object], ...]
    reserved_families_indexed: tuple[str, ...]
    roster_reasons: tuple[str, ...]
    roster_notes: Mapping[str, object]
    claims_not_established: tuple[str, ...] = PROJECTION_CLAIMS_NOT_ESTABLISHED

    def payload(self) -> dict:
        """A JSON-ready aggregate payload."""
        return {
            "kind": "us_cd_reference_projection_completeness",
            "protocol_sha256": self.protocol_sha256,
            "inventory_sha256": self.inventory_sha256,
            "inventory_size_bytes": self.inventory_size_bytes,
            "inventory_identity_verified": self.inventory_identity_verified,
            "roster_district_count": self.roster_district_count,
            "districts_requested": list(self.districts_requested),
            "blocks": [dict(block) for block in self.blocks],
            "roster_scope_reasons": list(self.roster_reasons),
            "roster_scope_notes": _jsonable(self.roster_notes),
            "claims_not_established": list(self.claims_not_established),
            "candidate_scoring": False,
            "target_registration": False,
            "exposure_ledger_write": False,
            "threshold_selection": False,
            "national_reference_cells": (
                "not_available; district sums are not published national rows and "
                "require separately acquired direct national references"
            ),
            "reserved_families_indexed_for_structure": list(
                self.reserved_families_indexed
            ),
            "reserved_families_projected_into_values": sorted(
                {
                    str(family)
                    for block in self.blocks
                    for family in block.get("exposure_families", ())
                    if str(family) in RESERVED_FAMILY_TABLES
                }
            ),
            "reserved_table_indexing": (
                "indexing validated the structure of every required table, the "
                "reserved families included, because the protocol's required "
                "district set must be identical across them; no reserved "
                "estimate or margin was classified or read into a value unless "
                "its block is listed above"
            ),
            "pinned_adapters": list(PINNED_ADAPTER_MODULES),
        }


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | frozenset | set):
        items = sorted(value, key=str) if isinstance(value, frozenset | set) else value
        return [_jsonable(item) for item in items]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)


def projection_completeness(
    projections: Iterable[BlockProjection],
    *,
    protocol: CompiledProtocol | None = None,
    roster: DistrictRoster | None = None,
    inventory: ReferenceInventory | None = None,
) -> ProjectionCompleteness:
    """Reduce projections to aggregate counts — a completeness statement only.

    This is explicitly not a score, a verdict or a readiness claim: it counts
    districts, cells, validity classes, precision statuses and reasons. It never
    adds an estimate to another estimate.

    The same block at the same district may be supplied only once. Counting one
    district twice would report more districts projected than were ever
    requested, which is the one thing a completeness statement must not do.

    Raises:
        ReferenceProjectionError: If a projection is repeated for the same block
            and district, or is not a :class:`BlockProjection`.
    """
    protocol = protocol or canonical_protocol()
    grouped: dict[str, list[BlockProjection]] = {}
    districts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for projection in projections:
        if not isinstance(projection, BlockProjection):
            raise ReferenceProjectionError(
                "Completeness counts BlockProjection values, not "
                f"{type(projection).__name__}."
            )
        key = (projection.block_id, projection.geography_id)
        if key in seen:
            raise ReferenceProjectionError(
                f"Block {projection.block_id!r} at {projection.geography_id!r} "
                "was supplied twice; a district counted twice is not a "
                "completeness statement."
            )
        seen.add(key)
        grouped.setdefault(projection.block_id, []).append(projection)
        if projection.geography_id not in districts:
            districts.append(projection.geography_id)
    blocks: list[Mapping[str, object]] = []
    for block_id in sorted(grouped):
        items = grouped[block_id]
        block = protocol.block(block_id)
        validity: dict[str, int] = {}
        statuses: dict[str, int] = {}
        reasons: dict[str, int] = {}
        cell_count = 0
        precision_count = 0
        complete = 0
        for projection in items:
            complete += int(projection.complete)
            cell_count += len(projection.cells)
            precision_count += len(projection.precision)
            for item in projection.cells:
                for side in ("estimate", "moe"):
                    label = f"{side}:{getattr(item.cell, side).adapter_class}"
                    validity[label] = validity.get(label, 0) + 1
            for item in projection.precision:
                key = str(item.result.status)
                statuses[key] = statuses.get(key, 0) + 1
            for reason in projection.reasons:
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
        blocks.append(
            MappingProxyType(
                {
                    "block_id": block_id,
                    "mandatory": block.mandatory,
                    "reserved_family": bool(
                        block.exposure_families & RESERVED_FAMILY_TABLES
                    ),
                    "exposure_families": tuple(sorted(block.exposure_families)),
                    "expected_cells_per_district": len(
                        block_cell_roster(protocol, block)
                    ),
                    "districts_projected": len(items),
                    "districts_complete": complete,
                    "districts_incomplete": len(items) - complete,
                    "cells_projected": cell_count,
                    "precision_readings": precision_count,
                    "adapter_class_counts": dict(sorted(validity.items())),
                    "precision_status_counts": dict(sorted(statuses.items())),
                    "reason_counts": dict(sorted(reasons.items())),
                }
            )
        )
    if roster is None and inventory is not None:
        roster = district_roster(inventory, protocol=protocol)
    if roster is None:
        reasons_tuple: tuple[Reason, ...] = ()
        notes: Mapping[str, object] = MappingProxyType({})
        roster_count = 0
    else:
        # ``districts`` unconditionally, never ``districts or None``: an empty
        # projection is an empty projection. Defaulting it to the whole roster
        # would report a run that projected nothing as roster-complete.
        reasons_tuple, notes = roster_scope_reasons(
            roster, protocol=protocol, present_geography_ids=districts
        )
        roster_count = len(roster)
    return ProjectionCompleteness(
        protocol_sha256=protocol.sha256,
        inventory_sha256=None if inventory is None else inventory.sha256,
        inventory_size_bytes=None if inventory is None else inventory.size_bytes,
        inventory_identity_verified=(
            False if inventory is None else inventory.identity_verified
        ),
        roster_district_count=roster_count,
        districts_requested=tuple(districts),
        blocks=tuple(blocks),
        reserved_families_indexed=tuple(
            sorted(set(required_tables(protocol)) & set(RESERVED_FAMILY_TABLES))
        ),
        roster_reasons=tuple(str(reason) for reason in reasons_tuple),
        roster_notes=notes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project pinned CD published references into the protocol's typed "
            "cells and precision readings, and report aggregate completeness "
            "only. No candidate is read, no score is produced, no target is "
            "registered and no ledger is written."
        )
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--expect-sha256",
        required=True,
        help="the inventory digest to enforce before parsing; no default",
    )
    parser.add_argument(
        "--block",
        action="append",
        default=[],
        metavar="BLOCK_ID",
        help="repeatable; there is no all-blocks mode",
    )
    parser.add_argument("--district", action="append", default=[], metavar="GEO_ID")
    parser.add_argument(
        "--all-roster-districts",
        action="store_true",
        help="project every district in the inventory roster",
    )
    parser.add_argument(
        "--include-reserved",
        action="store_true",
        help="required to name a B19001 or B25003 reserved-family block",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        required=True,
        help="affirm aggregate-only output; this tool has no other mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "write the aggregate payload here; the file is created exclusively "
            "and an existing path is refused, never replaced"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded, aggregate-only reference projection.

    Every argument that can be judged without the inventory is judged before it
    is opened: naming both district selectors or neither, repeating a block or a
    district, and naming a reserved block without ``--include-reserved`` all
    fail before a 38.7 MB read rather than after it.

    ``--output`` is created exclusively. An existing path is refused with its
    bytes untouched, because the artifact this writes is a dated record and a
    rerun that silently replaced one would destroy the evidence it claims to be.
    On success the exact bytes written — canonical JSON and its terminating
    newline — are hashed and that digest is printed, so hashing the file
    reproduces the printed line.
    """
    args = _parser().parse_args(argv)
    try:
        protocol = canonical_protocol()
        if not args.block:
            raise ReferenceProjectionError(
                "Name at least one --block; this tool never sweeps blocks."
            )
        if len(set(args.block)) != len(args.block):
            raise ReferenceProjectionError("A block was requested twice.")
        blocks = [protocol.block(block_id) for block_id in args.block]
        reserved = sorted(
            block.id
            for block in blocks
            if block.exposure_families & RESERVED_FAMILY_TABLES
        )
        if reserved and not args.include_reserved:
            raise ReferenceProjectionError(
                f"Blocks {reserved} read reserved families "
                f"{sorted(RESERVED_FAMILY_TABLES)}; pass --include-reserved to "
                "state that deliberately."
            )
        if args.all_roster_districts and args.district:
            raise ReferenceProjectionError(
                "Choose --all-roster-districts or explicit --district, not both."
            )
        if not args.all_roster_districts and not args.district:
            raise ReferenceProjectionError(
                "Name --district or --all-roster-districts; there is no default."
            )
        if len(set(args.district)) != len(args.district):
            raise ReferenceProjectionError("A district was requested twice.")
        inventory = load_pinned_inventory(
            args.inventory, expected_sha256=args.expect_sha256, protocol=protocol
        )
        roster = district_roster(inventory, protocol=protocol)
        districts = (
            list(roster.geography_ids)
            if args.all_roster_districts
            else list(args.district)
        )
        projections = [
            project_block(
                inventory,
                block_id=block.id,
                geography_id=geography_id,
                protocol=protocol,
            )
            for block in blocks
            for geography_id in districts
        ]
        summary = projection_completeness(
            projections, protocol=protocol, roster=roster, inventory=inventory
        )
        written = canonical_bytes(summary.payload()) + b"\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with args.output.open("xb") as stream:
                    stream.write(written)
            except FileExistsError:
                # Handled explicitly, and only around the exclusive create, so
                # this message can never describe some other FileExistsError.
                # Refusing to replace an aggregate record is a deliberate
                # outcome of this tool, not an incidental filesystem error.
                raise ReferenceProjectionError(
                    f"{args.output} already exists and is not replaced; name a "
                    "fresh --output path."
                ) from None
    except (ReferenceProjectionError, CanonicalJsonError, OSError, ValueError) as error:
        print(f"Reference projection failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        print(
            f"Aggregate completeness written to {args.output}; "
            f"sha256 of the {len(written)} bytes written {digest(written)}"
        )
    else:
        sys.stdout.write(written.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
