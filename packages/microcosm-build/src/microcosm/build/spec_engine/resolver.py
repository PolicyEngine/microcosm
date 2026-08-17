"""Cross-reference closure for normalized country bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .canonical import canonical_json_bytes
from .model import (
    ArtifactSpec,
    ColumnSpec,
    EntitySpec,
    ScopeSpec,
    SymbolRef,
)


class SpecResolutionError(ValueError):
    """A typed reference is duplicate, dangling, or structurally invalid."""


@dataclass(frozen=True, slots=True)
class KernelRegistry:
    ids: frozenset[str]
    implementation_sha256: str

    @classmethod
    def from_ids(cls, ids: Iterable[str]) -> KernelRegistry:
        normalized = frozenset(_strip_prefix(item, "kernel") for item in ids)
        digest = hashlib.sha256(
            canonical_json_bytes(sorted(normalized))
        ).hexdigest()
        return cls(normalized, digest)

    def contains(self, value: str) -> bool:
        return _strip_prefix(value, "kernel") in self.ids


@dataclass(frozen=True, slots=True)
class SeedProtocol:
    id: str
    streams: frozenset[str]
    implementation_id: str
    implementation_sha256: str


# This registry is immutable compiler grammar, not a second authored country
# resource.  D3 fills the US bundle with references to these existing legacy
# call-site protocols; F1 is where a broker begins enforcing them.
LEGACY_V1_STREAMS = frozenset(
    {
        "sampling_asec",
        "sampling_acs",
        "puf_clone_attachment",
        "build_model",
        "puf_archived_disaggregation",
        "puf_live_disaggregation",
        "ssi_weighted_replacement",
        "ssi_model",
        "sipp_training_cap",
        "qrf_fit_draw",
        "stable_entity_draw",
        "calibration",
        "exact_k_selection",
        "geography_legacy",
        "geography_block_draw",
    }
)
LEGACY_V1_PROTOCOL = SeedProtocol(
    id="legacy-v1",
    streams=LEGACY_V1_STREAMS,
    implementation_id="microcosm.seed-protocol.legacy-v1",
    implementation_sha256=hashlib.sha256(
        canonical_json_bytes(
            {
                "id": "legacy-v1",
                "streams": sorted(LEGACY_V1_STREAMS),
                "boundary": "mirror-only-until-f1",
            }
        )
    ).hexdigest(),
)


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    references: tuple[SymbolRef, ...]
    entities: tuple[EntitySpec, ...]
    artifacts: tuple[ArtifactSpec, ...]
    scopes: tuple[ScopeSpec, ...]
    columns: tuple[ColumnSpec, ...]


def _strip_prefix(value: str, namespace: str) -> str:
    prefix = f"{namespace}:"
    return value[len(prefix) :] if value.startswith(prefix) else value


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecResolutionError(f"{location}: expected mapping")
    return value


def _array(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise SpecResolutionError(f"{location}: expected array")
    return value


def _declare_array_ids(
    value: object,
    *,
    location: str,
    field: str = "id",
) -> frozenset[str]:
    declared: set[str] = set()
    for index, row in enumerate(_array(value, location)):
        item = _mapping(row, f"{location}/{index}")
        candidate = item.get(field)
        if not isinstance(candidate, str):
            raise SpecResolutionError(
                f"{location}/{index}/{field}: expected identifier"
            )
        if candidate in declared:
            raise SpecResolutionError(
                f"{location}/{index}/{field}: duplicate id {candidate!r}"
            )
        declared.add(candidate)
    return frozenset(declared)


def _declare_mapping_ids(value: object, *, location: str) -> frozenset[str]:
    mapping = _mapping(value, location)
    return frozenset(mapping)


def _walk_strings(value: object, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_strings(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}/{index}")
    elif isinstance(value, str):
        yield path or "/", value


def _require(
    target: str,
    declared: frozenset[str],
    *,
    namespace: str,
    location: str,
) -> None:
    if target not in declared:
        raise SpecResolutionError(
            f"{location}: dangling {namespace} reference {target!r}"
        )


def resolve_cross_references(
    resources: Mapping[str, object],
    *,
    kernel_registry: KernelRegistry,
) -> ResolutionResult:
    """Resolve every typed namespace used by the selected country bundle."""

    sources = resources.get("sources", {})
    vintages = resources.get("vintages", {})
    spine = resources.get("spine", {})
    imputation = resources.get("imputation", {})
    bundle = _mapping(resources.get("bundle", {}), "bundle")

    source_ids = _declare_array_ids(
        _mapping(sources, "sources").get("sources", []),
        location="sources/sources",
    )
    vintage_ids = _declare_array_ids(
        _mapping(vintages, "vintages").get("records", []),
        location="vintages/records",
    )
    spine_map = _mapping(spine, "spine")
    channel_ids = _declare_array_ids(
        spine_map.get("channels", []), location="spine/channels"
    )
    support_ids = _declare_array_ids(
        spine_map.get("support_roles", []), location="spine/support_roles"
    )
    imputation_map = _mapping(imputation, "imputation")
    model_ids = _declare_mapping_ids(
        imputation_map.get("models", {}), location="imputation/models"
    )
    concept_ids = _declare_mapping_ids(
        imputation_map.get("concepts", {}), location="imputation/concepts"
    )
    family_rows = imputation_map.get("families", [])
    family_ids = _declare_array_ids(family_rows, location="imputation/families")
    graph = _mapping(imputation_map.get("producer_graph", {}), "producer_graph")
    node_ids = _declare_array_ids(
        graph.get("nodes", []), location="imputation/producer_graph/nodes"
    )
    external_stage_ids = frozenset(
        str(item) for item in _array(graph.get("external_stages", []), "external_stages")
    )

    chaining = _mapping(imputation_map.get("chaining", {}), "imputation/chaining")
    for split_index, split_row in enumerate(
        _array(chaining.get("split_after", []), "imputation/chaining/split_after")
    ):
        split = _mapping(split_row, f"imputation/chaining/split_after/{split_index}")
        family = split.get("family")
        if isinstance(family, str):
            _require(
                family,
                family_ids,
                namespace="family",
                location=f"imputation/chaining/split_after/{split_index}/family",
            )

    protocol_id = bundle.get("seed_protocol")
    if protocol_id != LEGACY_V1_PROTOCOL.id:
        # derived-v2 is valid grammar but intentionally has no F0 execution
        # protocol.  It can compile only after its immutable registry lands.
        raise SpecResolutionError(
            f"bundle/seed_protocol: unsupported F0 protocol {protocol_id!r}"
        )

    references: list[SymbolRef] = []
    for kind, resource in sorted(resources.items()):
        for pointer, string in _walk_strings(resource):
            location = f"{kind}{pointer}"
            if string.startswith("kernel:"):
                target = _strip_prefix(string, "kernel")
                if not kernel_registry.contains(string):
                    raise SpecResolutionError(
                        f"{location}: dangling kernel reference {target!r}"
                    )
                references.append(SymbolRef("kernel", target, location))
            elif string.startswith("source:"):
                target = _strip_prefix(string, "source")
                _require(target, source_ids, namespace="source", location=location)
                references.append(SymbolRef("source", target, location))
            elif string.startswith("vintage:"):
                target = _strip_prefix(string, "vintage")
                _require(target, vintage_ids, namespace="vintage", location=location)
                references.append(SymbolRef("vintage", target, location))
            elif string.startswith("stream:"):
                target = _strip_prefix(string, "stream")
                _require(
                    target,
                    LEGACY_V1_PROTOCOL.streams,
                    namespace="stream",
                    location=location,
                )
                references.append(SymbolRef("stream", target, location))

    if channel_ids:
        for channel_index, channel_row in enumerate(
            _array(spine_map.get("channels", []), "spine/channels")
        ):
            channel = _mapping(channel_row, f"spine/channels/{channel_index}")
            source_value = channel.get("source")
            channel_sources = (
                [source_value] if isinstance(source_value, str) else source_value
            )
            for source_index, source_id in enumerate(
                _array(channel_sources, f"spine/channels/{channel_index}/source")
            ):
                if isinstance(source_id, str):
                    _require(
                        source_id,
                        source_ids,
                        namespace="source",
                        location=(
                            f"spine/channels/{channel_index}/source/{source_index}"
                        ),
                    )
        assembly = _mapping(spine_map.get("assembly", {}), "spine/assembly")
        anchor = assembly.get("mass_anchor_channel")
        if isinstance(anchor, str):
            _require(
                anchor,
                channel_ids,
                namespace="channel",
                location="spine/assembly/mass_anchor_channel",
            )
        for index, row in enumerate(_array(family_rows, "imputation/families")):
            family = _mapping(row, f"imputation/families/{index}")
            donor = _mapping(family.get("donor", {}), f"families/{index}/donor")
            if isinstance(donor.get("channel"), str):
                _require(
                    str(donor["channel"]),
                    channel_ids,
                    namespace="channel",
                    location=f"imputation/families/{index}/donor/channel",
                )
            if isinstance(donor.get("support_role"), str):
                _require(
                    str(donor["support_role"]),
                    support_ids,
                    namespace="support_role",
                    location=f"imputation/families/{index}/donor/support_role",
                )
            model = family.get("model")
            if isinstance(model, str):
                _require(
                    model,
                    model_ids,
                    namespace="model",
                    location=f"imputation/families/{index}/model",
                )
            for target_index, target_row in enumerate(
                _array(family.get("targets", []), f"families/{index}/targets")
            ):
                target = _mapping(target_row, "target")
                for concept_index, concept in enumerate(
                    _array(target.get("requires_concepts", []), "requires_concepts")
                ):
                    if isinstance(concept, str):
                        _require(
                            concept,
                            concept_ids,
                            namespace="concept",
                            location=(
                                f"imputation/families/{index}/targets/"
                                f"{target_index}/requires_concepts/{concept_index}"
                            ),
                        )

    for index, node_row in enumerate(_array(graph.get("nodes", []), "nodes")):
        node = _mapping(node_row, f"nodes/{index}")
        for dependency_index, dependency in enumerate(
            _array(node.get("depends_on", []), "depends_on")
        ):
            if isinstance(dependency, str) and dependency not in (
                node_ids | external_stage_ids
            ):
                raise SpecResolutionError(
                    f"imputation/producer_graph/nodes/{index}/depends_on/"
                    f"{dependency_index}: dangling stage reference {dependency!r}"
                )

    # Compiler-facing typed catalogs.  Empty catalogs remain valid for minimal
    # country bundles; the full US bundle populates them in D3.
    entity_by_id = {
        name: EntitySpec(name)
        for name in ("person", "tax_unit", "spm_unit", "household", "family", "benunit")
    }
    columns: list[ColumnSpec] = []
    catalogs = _mapping(resources.get("catalogs", {}), "catalogs")
    for index, row in enumerate(_array(catalogs.get("columns", []), "catalogs/columns")):
        record = _mapping(row, f"catalogs/columns/{index}")
        contract = _mapping(record.get("contract", {}), "contract")
        entity_id = contract.get("entity")
        if not isinstance(entity_id, str) or entity_id not in entity_by_id:
            raise SpecResolutionError(
                f"catalogs/columns/{index}/contract/entity: unknown entity"
            )
        columns.append(
            ColumnSpec(
                key=str(record.get("key")),
                entity=entity_by_id[entity_id],
                dtype=str(contract.get("dtype")),
                period=str(contract.get("period")),
                nullable=bool(contract.get("nullable")),
            )
        )

    artifacts = tuple(
        ArtifactSpec(id=node_id, kind="producer_node")
        for node_id in sorted(node_ids)
    )
    # Row-scope algebra is retained in normalized resources; typed scope
    # extraction grows in D4 when producer compilation assigns stable ids.
    return ResolutionResult(
        references=tuple(sorted(references, key=lambda row: (row.source_path, row.id))),
        entities=tuple(entity_by_id.values()),
        artifacts=artifacts,
        scopes=tuple(),
        columns=tuple(columns),
    )
