"""Immutable types produced by the spec-engine front end.

The compiler deliberately exposes immutable, JSON-shaped records rather than
the mutable objects returned by a YAML parser.  Domain wrappers make it
impossible to accidentally pass (for example) publication configuration to a
source compiler while retaining a compact, lossless wire projection.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .seeds import SeedProtocol

type JsonScalar = None | bool | int | float | str


@dataclass(frozen=True, slots=True)
class FrozenMap(Mapping[str, "FrozenValue"]):
    """A deterministic, recursively immutable string-keyed mapping."""

    entries: tuple[tuple[str, FrozenValue], ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(key for key, _ in self.entries)
        if keys != tuple(sorted(keys)):
            raise ValueError("FrozenMap entries must be sorted by key")
        if len(keys) != len(set(keys)):
            raise ValueError("FrozenMap entries must have unique keys")

    def __getitem__(self, key: str) -> FrozenValue:
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def get_typed(self, key: str, expected: type) -> object:
        value = self[key]
        if not isinstance(value, expected):
            raise TypeError(f"{key!r} is not {expected.__name__}")
        return value


type FrozenValue = JsonScalar | tuple[FrozenValue, ...] | FrozenMap


def freeze_json(value: object) -> FrozenValue:
    """Convert a validated JSON value to its immutable representation."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        entries: list[tuple[str, FrozenValue]] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("spec mappings require string keys")
            entries.append((key, freeze_json(value[key])))
        return FrozenMap(tuple(entries))
    raise TypeError(f"not a JSON value: {type(value).__name__}")


def thaw_json(value: FrozenValue) -> JsonScalar | list | dict[str, object]:
    """Return the canonical mutable wire representation."""

    if isinstance(value, FrozenMap):
        return {key: thaw_json(child) for key, child in value.entries}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value


class ResourceKind(StrEnum):
    BUNDLE = "bundle"
    SOURCES = "sources"
    SPINE = "spine"
    GEOGRAPHY = "geography"
    IMPUTATION = "imputation"
    TAKE_UP = "take_up"
    BATTERY = "battery"
    CALIBRATION = "calibration"
    SELECTION = "selection"
    PUBLICATION = "publication"
    VINTAGES = "vintages"
    CATALOGS = "catalogs"
    SCHEMA = "schema"
    LEGACY_JSON = "legacy_json"


class Surface(StrEnum):
    NORMATIVE = "normative"
    RUN_REQUEST = "run_request"
    EXECUTION_PROFILE = "execution_profile"
    OPERATIONAL = "operational"
    CHAIN_STATE = "chain_state"
    DOCUMENTATION = "documentation"


class SeedSiteOwnerKind(StrEnum):
    """Closed owner namespaces for a resolved stochastic draw site."""

    PRODUCER_NODE = "producer_node"
    SOURCE_STAGE = "source_stage"
    PIPELINE_OPERATION = "pipeline_operation"


@dataclass(frozen=True, slots=True)
class ResourceDescriptor:
    path: PurePosixPath
    kind: ResourceKind
    schema_id: str

    def to_wire(self) -> dict[str, str]:
        return {
            "path": self.path.as_posix(),
            "kind": self.kind.value,
            "schema_id": self.schema_id,
        }


@dataclass(frozen=True, slots=True)
class MigrationReceipt:
    id: str
    sha256: str

    def to_wire(self) -> dict[str, str]:
        return {"id": self.id, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class GrammarReceipt:
    schema_version: int
    canonicalizer_version: int
    migration_chain: tuple[MigrationReceipt, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonicalizer_version": self.canonicalizer_version,
            "migration_chain": [item.to_wire() for item in self.migration_chain],
        }


@dataclass(frozen=True, slots=True)
class FileReceipt:
    sha256: str
    byte_size: int

    def to_wire(self) -> dict[str, object]:
        return {"sha256": self.sha256, "byte_size": self.byte_size}


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """Typed wrapper around one normalized domain resource."""

    descriptor: ResourceDescriptor
    value: FrozenValue

    def to_wire(self) -> object:
        return thaw_json(self.value)


@dataclass(frozen=True, slots=True)
class BundleSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class SourcesSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class SpineSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class GeographySpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class ImputationSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class TakeUpSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class BatterySpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class CalibrationSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class SelectionSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class PublicationSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class VintagesSpec(DomainSpec):
    pass


@dataclass(frozen=True, slots=True)
class CatalogsSpec(DomainSpec):
    pass


DOMAIN_TYPE_BY_KIND: Mapping[ResourceKind, type[DomainSpec]] = {
    ResourceKind.BUNDLE: BundleSpec,
    ResourceKind.SOURCES: SourcesSpec,
    ResourceKind.SPINE: SpineSpec,
    ResourceKind.GEOGRAPHY: GeographySpec,
    ResourceKind.IMPUTATION: ImputationSpec,
    ResourceKind.TAKE_UP: TakeUpSpec,
    ResourceKind.BATTERY: BatterySpec,
    ResourceKind.CALIBRATION: CalibrationSpec,
    ResourceKind.SELECTION: SelectionSpec,
    ResourceKind.PUBLICATION: PublicationSpec,
    ResourceKind.VINTAGES: VintagesSpec,
    ResourceKind.CATALOGS: CatalogsSpec,
    ResourceKind.SCHEMA: DomainSpec,
    ResourceKind.LEGACY_JSON: DomainSpec,
}


@dataclass(frozen=True, slots=True)
class ResolvedResource:
    descriptor: ResourceDescriptor
    domain: DomainSpec
    file_receipt: FileReceipt
    projections: FrozenMap

    def surface(self, surface: Surface) -> FrozenValue:
        return self.projections.get(surface.value, FrozenMap())


@dataclass(frozen=True, slots=True)
class SymbolRef:
    namespace: str
    id: str
    source_path: str


@dataclass(frozen=True, slots=True)
class SeedSiteOwner:
    kind: SeedSiteOwnerKind
    id: str

    def to_wire(self) -> dict[str, str]:
        return {"kind": self.kind.value, "id": self.id}


@dataclass(frozen=True, slots=True)
class SeedSiteBinding:
    site: str
    owners: tuple[SeedSiteOwner, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "site": self.site,
            "owners": [owner.to_wire() for owner in self.owners],
        }


@dataclass(frozen=True, slots=True)
class EntitySpec:
    """A resolved frame entity used by typed columns and scopes."""

    id: str


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """A compiler-resolved node, virtual output, or bound virtual resource."""

    id: str
    kind: str
    producing_stages: tuple[str, ...]
    entity: EntitySpec | None
    key: str | None
    lifetime: str
    validation: str
    binding: FrozenValue


@dataclass(frozen=True, slots=True)
class ScopeSpec:
    """A named row scope in a finite, compiler-decidable predicate space."""

    id: str
    predicate_space: str
    entity: EntitySpec | None
    predicate: FrozenValue
    source_path: str


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    key: str
    entity: EntitySpec
    dtype: str
    unit: str
    period: str
    vintage: str | None
    nullable: bool
    domain: str
    public_stability: str
    unit_waiver: str | None


@dataclass(frozen=True, slots=True)
class SurfaceObjects:
    normative: FrozenMap = field(default_factory=FrozenMap)
    run_request: FrozenMap = field(default_factory=FrozenMap)
    execution_profile: FrozenMap = field(default_factory=FrozenMap)
    operational: FrozenMap = field(default_factory=FrozenMap)
    chain_state: FrozenMap = field(default_factory=FrozenMap)
    documentation: FrozenMap = field(default_factory=FrozenMap)

    def for_surface(self, surface: Surface) -> FrozenMap:
        return getattr(self, surface.value)


@dataclass(frozen=True, slots=True)
class SpecBinding:
    country: str
    schema_id: str
    schema_version: int
    canonicalizer_version: int
    spec_sha256: str
    attestation: str = "mirror-attested"

    def to_wire(self) -> dict[str, object]:
        return {
            "country": self.country,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "canonicalizer_version": self.canonicalizer_version,
            "spec_sha256": self.spec_sha256,
            "attestation": self.attestation,
        }


@dataclass(frozen=True, slots=True)
class ResolvedSpec:
    country: str
    schema_version: int
    resources: tuple[ResolvedResource, ...]
    grammar_receipt: GrammarReceipt
    surfaces: SurfaceObjects
    entities: tuple[EntitySpec, ...]
    artifacts: tuple[ArtifactSpec, ...]
    scopes: tuple[ScopeSpec, ...]
    columns: tuple[ColumnSpec, ...]
    references: tuple[SymbolRef, ...]
    vintage_authorities: FrozenMap
    generated_authorities: FrozenMap
    seed_protocol: SeedProtocol
    seed_site_bindings: tuple[SeedSiteBinding, ...]
    file_receipts: FrozenMap
    package_fingerprint: str
    spec_sha256: str
    documentation_sha256: str

    def resource(self, kind: ResourceKind | str) -> ResolvedResource:
        selected = ResourceKind(kind)
        matches = tuple(
            resource
            for resource in self.resources
            if resource.descriptor.kind is selected
        )
        if len(matches) != 1:
            raise KeyError(
                f"expected exactly one {selected.value!r} resource, got {len(matches)}"
            )
        return matches[0]

    @property
    def spec_binding(self) -> SpecBinding:
        return SpecBinding(
            country=self.country,
            schema_id="country_spec",
            schema_version=self.schema_version,
            canonicalizer_version=self.grammar_receipt.canonicalizer_version,
            spec_sha256=self.spec_sha256,
        )

    def domain(self, kind: ResourceKind | str) -> DomainSpec:
        return self.resource(kind).domain
