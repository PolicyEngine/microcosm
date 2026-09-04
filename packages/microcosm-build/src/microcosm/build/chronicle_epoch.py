"""Epoch tolerance for Chronicle (formerly Ledger) identities.

PolicyEngine Ledger is being renamed Chronicle. Per the migration spec on
PolicyEngine/chronicle#143 the rename of *identities* happens **by epoch,
never in place**: Chronicle introduces chronicle-era hash domains and
chronicle-named schema ids for **newly emitted** rows at a declared cutover
release, and ledger-era ids stay valid *history* forever. No golden
regenerates and no witnessed row is rewritten.

Microcosm is a consumer, so it must accept both eras **before** Chronicle
flips emit. This module is the single place that knows which identities are
which era. The rules it encodes:

1. **Fact keys are opaque.** Microcosm never reconstructs a Chronicle key
   from a payload, so it never needs to know how the domain string feeds the
   hash. Comparison stays exact string equality, which is correct in both
   eras; what changes is that a validator asking "is this a Chronicle fact
   key?" must answer yes for either epoch.
2. **Epoch is a declaration, not a guess.** Every identity this module calls
   Chronicle-issued is listed in :data:`DECLARED_IDENTITIES` below, keyed by
   ``(namespace, family, version)``. Structural parsing of
   ``<namespace>.<family>.v<n>:<digest>`` still happens, but only so an
   *undeclared* key in a Chronicle namespace can be reported as
   :data:`UNDECLARED` in provenance — never silently promoted to issued
   identity. A cutover that ships a spelling this table does not carry is a
   one-row diff here, made deliberately, with the reader able to see which
   spellings the build was actually prepared for.
3. **Microcosm-minted keys are epoch-independent.** Keys Microcosm mints live
   in Microcosm-owned namespaces (``microcosm.derived_fact.*``,
   ``populace_us_trade.*``) and are frozen at v1 by microcosm#639. They carry
   no Chronicle epoch, and a chronicle-era source row must mint the
   byte-identical Microcosm key a ledger-era source row does — otherwise the
   cutover would silently re-identify derived facts.
4. **Nothing renames on disk.** Diagnostic field names
   (``ledger_aggregate_fact_key``, ``ledger_commit``), H5 attributes,
   ``populace_*`` ids, goldens, and fixtures are frozen at v1.

Provenance for the ledger-era half of the table was read out of
``PolicyEngine/chronicle`` at ``origin/main`` rather than restated from a
plan: ``policyengine_chronicle/consumer.py`` for the artifact id,
``chronicle/consumer_contract.py`` for the per-row id and the ``.v2`` fact
domains, ``chronicle/core.py`` and ``chronicle/sources/`` for the ``.v1``
source domains. The chronicle-era half is the successor each of those takes
under the rule chronicle#143 states for the rename — same family, new
namespace, version bumped by one — with the two ids the consumer migration
pins spelled out explicitly. Chronicle has published no per-family
enumeration of its own; when it does, this table is what changes.
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
    "CHRONICLE_FACT_KEY_DOMAINS",
    "CHRONICLE_NAMESPACES",
    "DECLARED_IDENTITIES",
    "DECLARED_IDENTITY_EPOCHS",
    "DECLARED_IDENTITY_REGISTRY",
    "EPOCHS",
    "FACT_KEY_EPOCH_LABELS",
    "FACT_KEY_FIELDS",
    "FACT_KEY_IDENTITY_KIND",
    "CONSUMER_ARTIFACT_SCHEMA_IDENTITY_KIND",
    "CONSUMER_FACT_SCHEMA_IDENTITY_KIND",
    "DeclaredIdentity",
    "FactKeyIdentity",
    "LEDGER_CONSUMER_FACT_SCHEMA_VERSION",
    "LEDGER_EPOCH",
    "LEDGER_FACT_KEY_DOMAINS",
    "MICROCOSM_CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "UNDECLARED",
    "consumer_artifact_schema_epoch",
    "consumer_fact_schema_epoch",
    "declared_identity",
    "describe_accepted_consumer_artifact_schema_versions",
    "describe_accepted_consumer_fact_schema_versions",
    "fact_key_epoch",
    "fact_key_epoch_label",
    "feed_fact_key_epochs",
    "feed_undeclared_fact_key_domains",
    "is_accepted_consumer_artifact_schema_version",
    "is_accepted_consumer_fact_schema_version",
    "is_chronicle_fact_key",
    "parse_fact_key",
    "row_fact_key_epochs",
]

#: The ledger era: everything Chronicle issued under its former name.
LEDGER_EPOCH = "ledger"

#: The chronicle era: everything issued from the declared cutover forward.
CHRONICLE_EPOCH = "chronicle"

#: Both eras, oldest first. A validator iterates this; it never names one.
EPOCHS: tuple[str, ...] = (LEDGER_EPOCH, CHRONICLE_EPOCH)

#: The label for an identity in a Chronicle namespace that this module's
#: declaration table does not carry — an unrecognized family, or a family at
#: a version nobody declared. Such an identity is *reported*, so a release
#: manifest shows that the feed contained something the build was not
#: prepared for, but it is never counted as Chronicle-issued identity.
UNDECLARED = "undeclared"

#: The labels :func:`feed_fact_key_epochs` can emit, in report order.
FACT_KEY_EPOCH_LABELS: tuple[str, ...] = (*EPOCHS, UNDECLARED)

#: Namespace -> epoch, for every namespace Chronicle issues identities in.
#: Fact keys and per-row schema ids use the bare ``ledger`` / ``chronicle``
#: namespace; the consumer-artifact manifest id uses the package-qualified
#: ``policyengine_ledger`` / ``policyengine_chronicle`` one.
CHRONICLE_NAMESPACES: Mapping[str, str] = {
    "ledger": LEDGER_EPOCH,
    "policyengine_ledger": LEDGER_EPOCH,
    "chronicle": CHRONICLE_EPOCH,
    "policyengine_chronicle": CHRONICLE_EPOCH,
}

#: Identity kinds. An identity's kind says where it is allowed to appear:
#: a fact-key domain never gates a manifest and a manifest id is never a key.
CONSUMER_ARTIFACT_SCHEMA_IDENTITY_KIND = "consumer_artifact_schema"
CONSUMER_FACT_SCHEMA_IDENTITY_KIND = "consumer_fact_schema"
FACT_KEY_IDENTITY_KIND = "fact_key"

_VERSION_PATTERN = re.compile(r"^v\d+$")


@dataclass(frozen=True)
class DeclaredIdentity:
    """One identity spelling the migration declares, and the epoch that owns it."""

    namespace: str
    family: str
    version: str
    epoch: str
    kind: str

    @property
    def identity(self) -> str:
        """The identity string as it appears in an artifact, ``a.b.vN``."""
        return f"{self.namespace}.{self.family}.{self.version}"

    @property
    def registry_key(self) -> tuple[str, str, str]:
        """The ``(namespace, family, version)`` triple this is registered under."""
        return (self.namespace, self.family, self.version)


def _declare(identity: str, *, epoch: str, kind: str) -> DeclaredIdentity:
    """Build one registry row from its literal identity string.

    Every argument is a literal at the call site below, so the table reads as
    the list of spellings it is, and a typo fails at import rather than at
    cutover.
    """
    namespace, _, remainder = identity.partition(".")
    family, _, version = remainder.rpartition(".")
    if not namespace or not family or not _VERSION_PATTERN.match(version):
        raise ValueError(f"Not a Chronicle identity spelling: {identity!r}.")
    if CHRONICLE_NAMESPACES.get(namespace) != epoch:
        raise ValueError(
            f"{identity!r} is declared for the {epoch!r} epoch but its "
            f"namespace {namespace!r} belongs to "
            f"{CHRONICLE_NAMESPACES.get(namespace)!r}."
        )
    return DeclaredIdentity(
        namespace=namespace, family=family, version=version, epoch=epoch, kind=kind
    )


#: The consumer-artifact manifest ``schema_version`` Microcosm's **own**
#: minted artifacts declare (``import_entry_facts.write_consumer_artifact``).
#: Frozen at v1 by microcosm#639: these bytes must not move, and they must
#: keep loading forever.
MICROCOSM_CONSUMER_ARTIFACT_SCHEMA_VERSION = "policyengine_ledger.consumer_artifact.v1"

#: The consumer-artifact manifest ``schema_version`` **Chronicle** stamps
#: today (``policyengine_chronicle/consumer.py`` on chronicle ``main``). Still
#: ledger-era: chronicle#143 has not flipped the namespace yet.
PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION = "policyengine_ledger.consumer_artifact.v2"

#: The chronicle-era successor of the id above: same payload, chronicle
#: namespace, version bumped.
CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION = (
    "policyengine_chronicle.consumer_artifact.v3"
)

#: The per-row consumer-fact ``schema_version`` Chronicle stamps today
#: (``chronicle/consumer_contract.py`` on chronicle ``main``), and its
#: chronicle-era successor.
LEDGER_CONSUMER_FACT_SCHEMA_VERSION = "ledger.consumer_fact.v1"
CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION = "chronicle.consumer_fact.v2"

#: Fact-key domains, ledger-era spelling beside its chronicle-era successor.
#: The ledger column is what Chronicle's ``main`` hashes with today; the
#: version numbers differ by family (the source-side domains are still v1
#: where the derived ones are v2), which is why a version can never be
#: assumed and why this table is explicit rather than computed at the call
#: site.
_FACT_KEY_DOMAIN_DECLARATIONS: tuple[tuple[str, str], ...] = (
    # chronicle/core.py
    ("ledger.fact.v1", "chronicle.fact.v2"),
    # chronicle/sources/cells.py, chronicle/sources/rows.py
    ("ledger.source_cell.v1", "chronicle.source_cell.v2"),
    ("ledger.source_row.v1", "chronicle.source_row.v2"),
    ("ledger.source_column.v1", "chronicle.source_column.v2"),
    ("ledger.source_row_value.v1", "chronicle.source_row_value.v2"),
    # chronicle/consumer_contract.py, policyengine_chronicle/consumer.py
    ("ledger.aggregate_fact.v2", "chronicle.aggregate_fact.v3"),
    ("ledger.semantic_fact.v2", "chronicle.semantic_fact.v3"),
    ("ledger.concept_alignment.v2", "chronicle.concept_alignment.v3"),
    ("ledger.dimension_set.v2", "chronicle.dimension_set.v3"),
    ("ledger.observed_measure.v2", "chronicle.observed_measure.v3"),
    ("ledger.source_release.v2", "chronicle.source_release.v3"),
    ("ledger.source_series.v2", "chronicle.source_series.v3"),
    ("ledger.universe_constraint_set.v2", "chronicle.universe_constraint_set.v3"),
)

#: Every identity spelling this consumer is declared to understand. The
#: registry below is keyed off it; nothing outside this tuple is Chronicle
#: identity as far as Microcosm is concerned.
DECLARED_IDENTITIES: tuple[DeclaredIdentity, ...] = (
    _declare(
        MICROCOSM_CONSUMER_ARTIFACT_SCHEMA_VERSION,
        epoch=LEDGER_EPOCH,
        kind=CONSUMER_ARTIFACT_SCHEMA_IDENTITY_KIND,
    ),
    _declare(
        PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
        epoch=LEDGER_EPOCH,
        kind=CONSUMER_ARTIFACT_SCHEMA_IDENTITY_KIND,
    ),
    _declare(
        CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION,
        epoch=CHRONICLE_EPOCH,
        kind=CONSUMER_ARTIFACT_SCHEMA_IDENTITY_KIND,
    ),
    _declare(
        LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
        epoch=LEDGER_EPOCH,
        kind=CONSUMER_FACT_SCHEMA_IDENTITY_KIND,
    ),
    _declare(
        CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
        epoch=CHRONICLE_EPOCH,
        kind=CONSUMER_FACT_SCHEMA_IDENTITY_KIND,
    ),
    *(
        _declare(identity, epoch=epoch, kind=FACT_KEY_IDENTITY_KIND)
        for ledger_domain, chronicle_domain in _FACT_KEY_DOMAIN_DECLARATIONS
        for identity, epoch in (
            (ledger_domain, LEDGER_EPOCH),
            (chronicle_domain, CHRONICLE_EPOCH),
        )
    ),
)

#: The registry: ``(namespace, family, version)`` -> the declared identity.
#: One triple can only belong to one epoch, and the constructor above
#: refuses a row whose namespace disagrees with the epoch it claims.
DECLARED_IDENTITY_REGISTRY: Mapping[tuple[str, str, str], DeclaredIdentity] = {
    declaration.registry_key: declaration for declaration in DECLARED_IDENTITIES
}
if len(DECLARED_IDENTITY_REGISTRY) != len(DECLARED_IDENTITIES):  # pragma: no cover
    raise ValueError("Duplicate (namespace, family, version) in DECLARED_IDENTITIES.")

#: The same registry reduced to what epoch resolution actually asks it:
#: ``(namespace, family, version)`` -> epoch.
DECLARED_IDENTITY_EPOCHS: Mapping[tuple[str, str, str], str] = {
    registry_key: declaration.epoch
    for registry_key, declaration in DECLARED_IDENTITY_REGISTRY.items()
}


def _identities_of(kind: str) -> tuple[str, ...]:
    """Declared identity strings of one kind, oldest era first."""
    return tuple(
        declaration.identity
        for era in EPOCHS
        for declaration in DECLARED_IDENTITIES
        if declaration.kind == kind and declaration.epoch == era
    )


#: Membership sets. Loaders test membership, never equality with one era.
#: The artifact set carries three spellings, not two: Microcosm's own minted
#: v1, Chronicle's published v2, and the chronicle-era v3 successor.
ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS = frozenset(
    _identities_of(CONSUMER_ARTIFACT_SCHEMA_IDENTITY_KIND)
)
ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS = frozenset(
    _identities_of(CONSUMER_FACT_SCHEMA_IDENTITY_KIND)
)

#: Fact-key domains by family, per epoch. Nothing consults these at runtime —
#: resolution goes through the registry — but they document the table and let
#: tests pin it against the feeds this repo carries.
LEDGER_FACT_KEY_DOMAINS: Mapping[str, str] = {
    declaration.family: declaration.identity
    for declaration in DECLARED_IDENTITIES
    if declaration.kind == FACT_KEY_IDENTITY_KIND and declaration.epoch == LEDGER_EPOCH
}
CHRONICLE_FACT_KEY_DOMAINS: Mapping[str, str] = {
    declaration.family: declaration.identity
    for declaration in DECLARED_IDENTITIES
    if declaration.kind == FACT_KEY_IDENTITY_KIND
    and declaration.epoch == CHRONICLE_EPOCH
}

#: Consumer-fact row fields that carry a single Chronicle key. The first four
#: are the identifiers :mod:`microcosm.build.ledger_targets` resolves a fact
#: by; the rest are the remaining single-key fields published rows carry, all
#: of them observed on the captured feed in
#: ``tests/fixtures/uk_target_reference_feed_rows.jsonl``. The list is for
#: *witnessing* epochs, so it is deliberately wider than the resolution set: a
#: row whose aggregate key is still ledger-era but whose source-release key has
#: crossed the cutover must report both epochs, not one.
FACT_KEY_FIELDS: tuple[str, ...] = (
    "aggregate_fact_key",
    "semantic_fact_key",
    "fact_key",
    "legacy_fact_key",
    "dimension_set_key",
    "observed_measure_key",
    "source_release_key",
    "source_series_key",
    "universe_constraint_set_key",
)

#: Row fields carrying a *list* of Chronicle keys.
_FACT_KEY_LIST_PATHS: tuple[tuple[str, ...], ...] = (
    ("lineage", "source_cell_keys"),
    ("lineage", "source_row_keys"),
)

#: Nested single-key paths.
_FACT_KEY_NESTED_PATHS: tuple[tuple[str, ...], ...] = (
    ("concept_alignment", "concept_alignment_key"),
)


@dataclass(frozen=True)
class FactKeyIdentity:
    """The structure of a Chronicle-shaped key, ``<domain>:<digest>``.

    ``epoch`` is the *declared* epoch — ``"ledger"`` or ``"chronicle"`` — and
    is ``None`` for anything :data:`DECLARED_IDENTITY_REGISTRY` does not carry.
    ``namespace_epoch`` is the weaker, structural answer: the epoch the key's
    namespace belongs to, whatever family or version follows it. The two
    differ exactly for a key in a Chronicle namespace whose spelling was never
    declared, which :attr:`epoch_label` reports as :data:`UNDECLARED` rather
    than promoting it to issued identity.

    Both are ``None`` for keys minted in some other namespace — Microcosm's
    own ``microcosm.derived_fact.*`` and ``populace_us_trade.*`` derived keys,
    for instance, which are deliberately outside both eras.
    """

    domain: str
    namespace: str
    family: str
    version: str
    digest: str
    epoch: str | None
    namespace_epoch: str | None

    @property
    def declared(self) -> bool:
        """True when this exact spelling is in the declaration table."""
        return self.epoch is not None

    @property
    def epoch_label(self) -> str | None:
        """The declared epoch, :data:`UNDECLARED`, or ``None`` if foreign."""
        if self.epoch is not None:
            return self.epoch
        return UNDECLARED if self.namespace_epoch is not None else None


def parse_fact_key(key: object) -> FactKeyIdentity | None:
    """Split ``<namespace>.<family>.v<n>:<digest>``; ``None`` if not that shape.

    Parsing is deliberately shallow. Microcosm does not recompute Chronicle
    digests, so it needs only enough structure to look the spelling up in
    :data:`DECLARED_IDENTITY_REGISTRY` — and, when the lookup misses, to say
    whether the miss was in a Chronicle namespace (undeclared, worth
    reporting) or in some other namespace entirely (not Chronicle's at all).
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
    declaration = DECLARED_IDENTITY_REGISTRY.get((namespace, family, version))
    if declaration is not None and declaration.kind != FACT_KEY_IDENTITY_KIND:
        # A manifest or per-row schema id is a declared identity, but it is
        # not a fact-key domain: keys never carry it.
        declaration = None
    return FactKeyIdentity(
        domain=domain,
        namespace=namespace,
        family=family,
        version=version,
        digest=digest,
        epoch=None if declaration is None else declaration.epoch,
        namespace_epoch=CHRONICLE_NAMESPACES.get(namespace),
    )


def declared_identity(identity: object) -> DeclaredIdentity | None:
    """The declaration for an identity string, or ``None`` if undeclared."""
    if not isinstance(identity, str) or not identity:
        return None
    namespace, _, remainder = identity.partition(".")
    family, _, version = remainder.rpartition(".")
    if not namespace or not family:
        return None
    return DECLARED_IDENTITY_REGISTRY.get((namespace, family, version))


def fact_key_epoch(key: object) -> str | None:
    """The epoch that *declared* ``key``'s domain, or ``None``.

    ``None`` covers both a foreign namespace and a Chronicle namespace whose
    exact spelling was never declared; :func:`fact_key_epoch_label`
    distinguishes them.
    """
    identity = parse_fact_key(key)
    return None if identity is None else identity.epoch


def fact_key_epoch_label(key: object) -> str | None:
    """``"ledger"``, ``"chronicle"``, :data:`UNDECLARED`, or ``None``."""
    identity = parse_fact_key(key)
    return None if identity is None else identity.epoch_label


def is_chronicle_fact_key(key: object) -> bool:
    """True when ``key`` is a *declared* Chronicle key of either epoch.

    An undeclared ``chronicle.*`` spelling is deliberately false here: it may
    turn out to be Chronicle's, but this build has not been told so, and a
    validator that answered yes would be treating a guess as identity.
    """
    return fact_key_epoch(key) is not None


def row_fact_key_epochs(row: object) -> frozenset[str]:
    """Every epoch label appearing in one consumer-fact row's Chronicle keys.

    Contains :data:`UNDECLARED` when the row carries a key in a Chronicle
    namespace whose spelling this consumer does not declare. Empty for a row
    whose keys are all Microcosm-minted: those namespaces are outside both
    eras by design.
    """
    return frozenset(label for label, _ in _row_labelled_keys(row))


def feed_undeclared_fact_key_domains(rows: Iterable[Any]) -> tuple[str, ...]:
    """Sorted Chronicle-namespace key domains the declaration table lacks.

    Reported alongside the epochs so a release manifest names *what* it did
    not recognise, rather than only that something went unrecognised.
    """
    undeclared: set[str] = set()
    for row in rows:
        for label, domain in _row_labelled_keys(row):
            if label == UNDECLARED:
                undeclared.add(domain)
    return tuple(sorted(undeclared))


def feed_fact_key_epochs(rows: Iterable[Any]) -> tuple[str, ...]:
    """Epoch labels observed across a whole feed, in report order.

    A mixed-epoch feed — ledger-era history beside chronicle-era rows — is
    expected during the cutover window and is reported, not rejected. So is
    an undeclared spelling: it appears as :data:`UNDECLARED`, never folded
    into either era.
    """
    observed: set[str] = set()
    for row in rows:
        observed |= row_fact_key_epochs(row)
    return tuple(label for label in FACT_KEY_EPOCH_LABELS if label in observed)


def is_accepted_consumer_artifact_schema_version(value: object) -> bool:
    """True for a declared consumer-artifact manifest schema id, either epoch.

    Non-string JSON — a list or object where a schema id belongs — is simply
    not accepted. It must not raise here: the loader turns a rejection into
    the documented unsupported-schema ``ValueError``, and a ``TypeError``
    escaping a membership test would bypass that message entirely.
    """
    return (
        isinstance(value, str) and value in ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS
    )


def is_accepted_consumer_fact_schema_version(value: object) -> bool:
    """True for a declared per-row consumer-fact schema id, either epoch."""
    return isinstance(value, str) and value in ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS


def consumer_artifact_schema_epoch(value: object) -> str | None:
    """The epoch of a consumer-artifact schema id, or ``None`` if undeclared."""
    return _schema_epoch(value, CONSUMER_ARTIFACT_SCHEMA_IDENTITY_KIND)


def consumer_fact_schema_epoch(value: object) -> str | None:
    """The epoch of a per-row consumer-fact schema id, or ``None``."""
    return _schema_epoch(value, CONSUMER_FACT_SCHEMA_IDENTITY_KIND)


def describe_accepted_consumer_artifact_schema_versions() -> str:
    """Every accepted artifact schema id, oldest era first, for messages."""
    return _describe(CONSUMER_ARTIFACT_SCHEMA_IDENTITY_KIND)


def describe_accepted_consumer_fact_schema_versions() -> str:
    """Every accepted per-row fact schema id, oldest era first, for messages."""
    return _describe(CONSUMER_FACT_SCHEMA_IDENTITY_KIND)


def _describe(kind: str) -> str:
    return ", ".join(repr(identity) for identity in _identities_of(kind))


def _schema_epoch(value: object, kind: str) -> str | None:
    declaration = declared_identity(value)
    if declaration is None or declaration.kind != kind:
        return None
    return declaration.epoch


def _row_labelled_keys(row: object) -> tuple[tuple[str, str], ...]:
    """``(epoch label, key domain)`` for every Chronicle key in one row."""
    if not isinstance(row, Mapping):
        return ()
    labelled: list[tuple[str, str]] = []

    def record(value: object) -> None:
        identity = parse_fact_key(value)
        if identity is None or identity.epoch_label is None:
            return
        labelled.append((identity.epoch_label, identity.domain))

    for field in FACT_KEY_FIELDS:
        record(row.get(field))
    for path in _FACT_KEY_NESTED_PATHS:
        record(_at(row, path))
    for path in _FACT_KEY_LIST_PATHS:
        values = _at(row, path)
        if isinstance(values, Iterable) and not isinstance(values, str | bytes):
            for value in values:
                record(value)
    return tuple(labelled)


def _at(row: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = row
    for segment in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current
