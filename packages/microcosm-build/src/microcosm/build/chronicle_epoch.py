"""Epoch tolerance for Chronicle (formerly Ledger) identities.

PolicyEngine Ledger is being renamed Chronicle. Per the migration spec on
PolicyEngine/chronicle#143 the rename of *identities* happens **by epoch,
never in place**: Chronicle introduces chronicle-era hash domains
(``chronicle.aggregate_fact.v3``, ``chronicle.semantic_fact.v3``, ...) and
chronicle-named schema ids (``policyengine_chronicle.consumer_artifact.v2``,
``chronicle.consumer_fact.v2``) for **newly emitted** rows at a declared
cutover release. Ledger-era ids (``ledger.aggregate_fact.v2``,
``policyengine_ledger.consumer_artifact.v1``, ...) stay valid *history*
forever: no golden regenerates and no witnessed row is rewritten.

Microcosm is a consumer, so it must accept both eras **before** Chronicle
flips emit. This module is the single place that knows which identities are
which era. The rules it encodes:

1. **Fact keys are opaque.** Microcosm never reconstructs a Chronicle key
   from a payload, so it never needs to know how the domain string feeds the
   hash. Comparison stays exact string equality, which is correct in both
   eras; what changes is that a validator asking "is this a Chronicle fact
   key?" must answer yes for either epoch.
2. **Never hard-code a single epoch in a validator.** Epoch detection here is
   *structural* — the namespace segment of the key domain — not a lookup in a
   frozen list of domain strings. Chronicle has declared the family-``v3``
   spellings for the two fact families named in the spec; the version numbers
   the remaining families will take are not declared yet, and this module
   deliberately does not guess them.
3. **Microcosm-minted keys are epoch-independent.** Keys Microcosm mints live
   in Microcosm-owned namespaces (``microcosm.derived_fact.*``,
   ``populace_us_trade.*``) and are frozen at v1 by microcosm#639. They carry
   no Chronicle epoch, and a chronicle-era source row must mint the
   byte-identical Microcosm key a ledger-era source row does — otherwise the
   cutover would silently re-identify derived facts.
4. **Nothing renames on disk.** Diagnostic field names
   (``ledger_aggregate_fact_key``, ``ledger_commit``), H5 attributes,
   ``populace_*`` ids, goldens, and fixtures are frozen at v1.

Only the identity strings the migration spec names explicitly are pinned as
literals here. Everything else is derived.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS",
    "ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS",
    "CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION",
    "CHRONICLE_EPOCH",
    "EPOCHS",
    "FACT_KEY_FIELDS",
    "FactKeyIdentity",
    "LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "LEDGER_CONSUMER_FACT_SCHEMA_VERSION",
    "LEDGER_EPOCH",
    "LEDGER_FACT_KEY_DOMAINS",
    "consumer_artifact_schema_epoch",
    "consumer_fact_schema_epoch",
    "describe_accepted_consumer_artifact_schema_versions",
    "describe_accepted_consumer_fact_schema_versions",
    "fact_key_epoch",
    "feed_fact_key_epochs",
    "is_accepted_consumer_artifact_schema_version",
    "is_accepted_consumer_fact_schema_version",
    "is_chronicle_fact_key",
    "parse_fact_key",
    "row_fact_key_epochs",
]

#: The ledger era: everything Chronicle emitted under its former name.
LEDGER_EPOCH = "ledger"

#: The chronicle era: everything emitted from the declared cutover forward.
CHRONICLE_EPOCH = "chronicle"

#: Both eras, oldest first. A validator iterates this; it never names one.
EPOCHS: tuple[str, ...] = (LEDGER_EPOCH, CHRONICLE_EPOCH)

#: Consumer-artifact manifest ``schema_version``, per epoch. Both spellings
#: are declared in the chronicle#143 migration spec.
LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION = "policyengine_ledger.consumer_artifact.v1"
CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION = (
    "policyengine_chronicle.consumer_artifact.v2"
)

#: Per-row consumer-fact ``schema_version``, per epoch.
LEDGER_CONSUMER_FACT_SCHEMA_VERSION = "ledger.consumer_fact.v1"
CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION = "chronicle.consumer_fact.v2"

CONSUMER_ARTIFACT_SCHEMA_VERSION_BY_EPOCH: Mapping[str, str] = {
    LEDGER_EPOCH: LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    CHRONICLE_EPOCH: CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION,
}
CONSUMER_FACT_SCHEMA_VERSION_BY_EPOCH: Mapping[str, str] = {
    LEDGER_EPOCH: LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
    CHRONICLE_EPOCH: CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
}

#: Membership sets. Loaders test membership, never equality with one era.
ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS = frozenset(
    CONSUMER_ARTIFACT_SCHEMA_VERSION_BY_EPOCH.values()
)
ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS = frozenset(
    CONSUMER_FACT_SCHEMA_VERSION_BY_EPOCH.values()
)

#: Ledger-era fact-key domains, as *observed* in the feeds and fixtures this
#: repo carries. Recorded for documentation and for the frozen-history tests;
#: epoch detection does not consult it, so a family this list has never seen
#: still resolves to its epoch.
LEDGER_FACT_KEY_DOMAINS: Mapping[str, str] = {
    "aggregate_fact": "ledger.aggregate_fact.v2",
    "semantic_fact": "ledger.semantic_fact.v2",
    "fact": "ledger.fact.v1",
    "source_cell": "ledger.source_cell.v1",
    "dimension_set": "ledger.dimension_set.v2",
    "concept_alignment": "ledger.concept_alignment.v2",
}

#: Consumer-fact row fields that carry a single Chronicle key, in the order
#: :mod:`microcosm.build.ledger_targets` resolves them.
FACT_KEY_FIELDS: tuple[str, ...] = (
    "aggregate_fact_key",
    "semantic_fact_key",
    "fact_key",
    "legacy_fact_key",
    "dimension_set_key",
)

#: Row fields carrying a *list* of Chronicle keys.
_FACT_KEY_LIST_PATHS: tuple[tuple[str, ...], ...] = (("lineage", "source_cell_keys"),)

#: Nested single-key paths.
_FACT_KEY_NESTED_PATHS: tuple[tuple[str, ...], ...] = (
    ("concept_alignment", "concept_alignment_key"),
)

_VERSION_PATTERN = re.compile(r"^v\d+$")


@dataclass(frozen=True)
class FactKeyIdentity:
    """The structure of a Chronicle-shaped key, ``<domain>:<digest>``.

    ``epoch`` is ``"ledger"`` or ``"chronicle"`` for Chronicle-issued keys and
    ``None`` for keys minted in some other namespace — Microcosm's own
    ``microcosm.derived_fact.*`` and ``populace_us_trade.*`` derived keys, for
    instance, which are deliberately outside both eras.
    """

    domain: str
    namespace: str
    family: str
    version: str
    digest: str
    epoch: str | None


def parse_fact_key(key: object) -> FactKeyIdentity | None:
    """Split ``<namespace>.<family>.v<n>:<digest>``; ``None`` if not that shape.

    Parsing is deliberately shallow. Microcosm does not recompute Chronicle
    digests, so it needs only enough structure to answer "which epoch issued
    this key?" — and it must answer that without a frozen list of domain
    strings, so a chronicle-era family this code has never seen still resolves.
    """
    if not isinstance(key, str) or not key:
        return None
    domain, separator, digest = key.partition(":")
    if not separator or not domain or not digest:
        return None
    segments = domain.split(".")
    if len(segments) < 3:
        return None
    version = segments[-1]
    if not _VERSION_PATTERN.match(version):
        return None
    namespace = segments[0]
    family = ".".join(segments[1:-1])
    if not namespace or not family:
        return None
    return FactKeyIdentity(
        domain=domain,
        namespace=namespace,
        family=family,
        version=version,
        digest=digest,
        epoch=namespace if namespace in EPOCHS else None,
    )


def fact_key_epoch(key: object) -> str | None:
    """The epoch that issued ``key``, or ``None`` for a foreign namespace."""
    identity = parse_fact_key(key)
    return None if identity is None else identity.epoch


def is_chronicle_fact_key(key: object) -> bool:
    """True when ``key`` is a Chronicle-issued key of *either* epoch."""
    return fact_key_epoch(key) is not None


def row_fact_key_epochs(row: object) -> frozenset[str]:
    """Every Chronicle epoch appearing in one consumer-fact row's keys.

    Empty for a row whose keys are all Microcosm-minted: those namespaces are
    outside both eras by design.
    """
    if not isinstance(row, Mapping):
        return frozenset()
    epochs: set[str] = set()
    for field in FACT_KEY_FIELDS:
        epoch = fact_key_epoch(row.get(field))
        if epoch is not None:
            epochs.add(epoch)
    for path in _FACT_KEY_NESTED_PATHS:
        epoch = fact_key_epoch(_at(row, path))
        if epoch is not None:
            epochs.add(epoch)
    for path in _FACT_KEY_LIST_PATHS:
        values = _at(row, path)
        if isinstance(values, Iterable) and not isinstance(values, str | bytes):
            for value in values:
                epoch = fact_key_epoch(value)
                if epoch is not None:
                    epochs.add(epoch)
    return frozenset(epochs)


def feed_fact_key_epochs(rows: Iterable[Any]) -> tuple[str, ...]:
    """Chronicle epochs observed across a whole feed, in :data:`EPOCHS` order.

    A mixed-epoch feed — ledger-era history beside chronicle-era rows — is
    expected during the cutover window and is reported, not rejected.
    """
    observed: set[str] = set()
    for row in rows:
        observed |= row_fact_key_epochs(row)
    return tuple(epoch for epoch in EPOCHS if epoch in observed)


def is_accepted_consumer_artifact_schema_version(value: object) -> bool:
    """True for the consumer-artifact manifest schema id of either epoch."""
    return value in ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS


def is_accepted_consumer_fact_schema_version(value: object) -> bool:
    """True for the per-row consumer-fact schema id of either epoch."""
    return value in ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS


def consumer_artifact_schema_epoch(value: object) -> str | None:
    """The epoch of a consumer-artifact schema id, or ``None`` if unknown."""
    return _epoch_of(value, CONSUMER_ARTIFACT_SCHEMA_VERSION_BY_EPOCH)


def consumer_fact_schema_epoch(value: object) -> str | None:
    """The epoch of a per-row consumer-fact schema id, or ``None``."""
    return _epoch_of(value, CONSUMER_FACT_SCHEMA_VERSION_BY_EPOCH)


def describe_accepted_consumer_artifact_schema_versions() -> str:
    """Both accepted artifact schema ids, oldest era first, for messages."""
    return _describe(CONSUMER_ARTIFACT_SCHEMA_VERSION_BY_EPOCH)


def describe_accepted_consumer_fact_schema_versions() -> str:
    """Both accepted per-row fact schema ids, oldest era first, for messages."""
    return _describe(CONSUMER_FACT_SCHEMA_VERSION_BY_EPOCH)


def _describe(by_epoch: Mapping[str, str]) -> str:
    return ", ".join(repr(by_epoch[epoch]) for epoch in EPOCHS)


def _epoch_of(value: object, by_epoch: Mapping[str, str]) -> str | None:
    for epoch in EPOCHS:
        if value == by_epoch[epoch]:
            return epoch
    return None


def _at(row: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = row
    for segment in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current
