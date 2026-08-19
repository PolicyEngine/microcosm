"""Narrow US runtime view of compiler-issued bundle authorities.

The spec engine issues a country-generic :class:`RuntimeAuthorities`
capability.  This module narrows that capability for the US build without
reopening the resolved bundle, invoking the legacy aggregate adapter, or
consulting constants-era loaders.  Values handed to orchestration remain the
compiler's recursively immutable objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from microcosm.build.spec_engine.compiler_ir import CompiledNode
from microcosm.build.spec_engine.model import (
    FrozenMap,
    FrozenValue,
    ResourceKind,
)
from microcosm.build.spec_engine.runtime_authorities import RuntimeAuthorities


class USSpecAuthorityError(ValueError):
    """A compiler capability cannot be narrowed to the US build contract."""


class USAuthorityProjection(StrEnum):
    """Closed compatibility projections consumed by the US pool driver."""

    PUBLICATION = "publication"
    SAMPLING = "sampling"
    IMPUTATION = "imputation"
    TAKE_UP = "take_up"
    BATTERY = "battery"
    BATTERY_COMPONENTS = "battery_components"
    STACKED_AUTHORITY = "stacked_authority"
    STACKED_CHECKPOINT_STATIC_COMPONENTS = "stacked_checkpoint_static_components"


@dataclass(frozen=True, slots=True)
class NodePort:
    """Typed entity/column selector for one compiled node port."""

    entity: str
    column: str
    scope: str | None = None

    def __post_init__(self) -> None:
        for name in ("entity", "column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise USSpecAuthorityError(
                    f"node port {name} must be a non-empty string"
                )
        if self.scope is not None and (
            not isinstance(self.scope, str) or not self.scope
        ):
            raise USSpecAuthorityError(
                "node port scope must be absent or a non-empty string"
            )


@dataclass(frozen=True, slots=True)
class NodeQuery:
    """Closed, typed lookup over fields already present on ``CompiledNode``."""

    node_id: str | None = None
    kernel_ref: str | None = None
    input_port: NodePort | None = None
    output_port: NodePort | None = None
    seed_site_id: str | None = None

    def __post_init__(self) -> None:
        if not any(
            value is not None
            for value in (
                self.node_id,
                self.kernel_ref,
                self.input_port,
                self.output_port,
                self.seed_site_id,
            )
        ):
            raise USSpecAuthorityError("node query requires at least one field")
        for name in ("node_id", "kernel_ref", "seed_site_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise USSpecAuthorityError(
                    f"node query {name} must be absent or a non-empty string"
                )
        if self.kernel_ref is not None and not self.kernel_ref.startswith("kernel:"):
            raise USSpecAuthorityError(
                "node query kernel_ref must use the kernel: namespace"
            )
        for name in ("input_port", "output_port"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, NodePort):
                raise TypeError(f"node query {name} must be NodePort")


def _merge_behavior_value(
    left: FrozenValue,
    right: FrozenValue,
    *,
    location: str,
) -> FrozenValue:
    """Rejoin complementary frozen surface projections without thawing."""

    if isinstance(left, FrozenMap) and isinstance(right, FrozenMap):
        keys = sorted(set(left) | set(right))
        entries: list[tuple[str, FrozenValue]] = []
        for key in keys:
            if key in left and key in right:
                value = _merge_behavior_value(
                    left[key],
                    right[key],
                    location=f"{location}/{key}",
                )
            elif key in left:
                value = left[key]
            else:
                value = right[key]
            entries.append((key, value))
        return FrozenMap(tuple(entries))
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise USSpecAuthorityError(
                f"{location}: behavior surface arrays have different lengths"
            )
        return tuple(
            _merge_behavior_value(
                left_item,
                right_item,
                location=f"{location}/{index}",
            )
            for index, (left_item, right_item) in enumerate(
                zip(left, right, strict=True)
            )
        )
    if type(left) is not type(right) or left != right:
        raise USSpecAuthorityError(f"{location}: behavior surface values conflict")
    return left


def _behavior_authority(authorities: RuntimeAuthorities) -> FrozenMap:
    merged = _merge_behavior_value(
        authorities.normative,
        authorities.execution_profile,
        location="behavior",
    )
    if not isinstance(merged, FrozenMap):  # pragma: no cover - fixed root types
        raise AssertionError("behavior authority root is not an object")
    return merged


def _port_matches(
    rows: tuple[FrozenMap, ...],
    port: NodePort,
    *,
    scope_field: str,
) -> bool:
    return any(
        row.get("entity") == port.entity
        and row.get("column") == port.column
        and (port.scope is None or row.get(scope_field) == port.scope)
        for row in rows
    )


def _node_matches(node: CompiledNode, query: NodeQuery) -> bool:
    if query.node_id is not None and node.id != query.node_id:
        return False
    if query.kernel_ref is not None and node.kernel_ref != query.kernel_ref:
        return False
    if query.input_port is not None and not _port_matches(
        node.inputs,
        query.input_port,
        scope_field="required_scope",
    ):
        return False
    if query.output_port is not None and not _port_matches(
        node.outputs,
        query.output_port,
        scope_field="coverage_scope",
    ):
        return False
    if query.seed_site_id is not None and all(
        site.id != query.seed_site_id for site in node.seed_sites
    ):
        return False
    return True


@dataclass(frozen=True, slots=True)
class USSpecAuthority:
    """Immutable US-only view of a generation-1 runtime capability."""

    authority_sha256: str
    spec_sha256: str
    identity_generation: int
    _behavior: FrozenMap = field(repr=False)
    _projections: FrozenMap = field(repr=False)
    _declared_sources: FrozenMap = field(repr=False)
    _nodes: tuple[CompiledNode, ...] = field(repr=False)

    @property
    def behavior_resources(self) -> FrozenMap:
        """Return all compiler-selected executable resource projections."""

        return self._behavior

    def behavior_resource(self, kind: ResourceKind | str) -> FrozenMap:
        """Return one behavior resource without operational/doc surfaces."""

        try:
            selected = ResourceKind(kind).value
        except ValueError as error:
            raise USSpecAuthorityError(
                f"unknown behavior resource kind {kind!r}"
            ) from error
        value = self._behavior.get(selected)
        if not isinstance(value, FrozenMap):
            raise USSpecAuthorityError(
                f"US behavior authority has no object resource {selected!r}"
            )
        return value

    @property
    def declared_sources(self) -> FrozenMap:
        """Return the compiler-sealed source pins and acquisition receipts."""

        return self._declared_sources

    def declared_source(self, source_id: str) -> FrozenMap:
        """Return exactly one source registry row by its typed id."""

        if not isinstance(source_id, str) or not source_id:
            raise USSpecAuthorityError("declared source id must be non-empty")
        rows = self._declared_sources.get("sources")
        if not isinstance(rows, tuple):
            raise USSpecAuthorityError("declared source registry is absent")
        matches = tuple(
            row
            for row in rows
            if isinstance(row, FrozenMap) and row.get("id") == source_id
        )
        if len(matches) != 1:
            raise USSpecAuthorityError(
                f"declared source {source_id!r} must match exactly once; "
                f"matched {len(matches)}"
            )
        return matches[0]

    def projection(self, projection: USAuthorityProjection | str) -> FrozenMap:
        """Return one closed compatibility projection by typed name."""

        try:
            selected = USAuthorityProjection(projection).value
        except ValueError as error:
            raise USSpecAuthorityError(
                f"unknown US authority projection {projection!r}"
            ) from error
        value = self._projections.get(selected)
        if not isinstance(value, FrozenMap):
            raise USSpecAuthorityError(f"US runtime projection {selected!r} is absent")
        return value

    @property
    def publication(self) -> FrozenMap:
        return self.projection(USAuthorityProjection.PUBLICATION)

    @property
    def sampling(self) -> FrozenMap:
        return self.projection(USAuthorityProjection.SAMPLING)

    @property
    def imputation(self) -> FrozenMap:
        return self.projection(USAuthorityProjection.IMPUTATION)

    @property
    def take_up(self) -> FrozenMap:
        return self.projection(USAuthorityProjection.TAKE_UP)

    @property
    def battery(self) -> FrozenMap:
        return self.projection(USAuthorityProjection.BATTERY)

    @property
    def battery_components(self) -> FrozenMap:
        return self.projection(USAuthorityProjection.BATTERY_COMPONENTS)

    @property
    def stacked_authority(self) -> FrozenMap:
        return self.projection(USAuthorityProjection.STACKED_AUTHORITY)

    @property
    def stacked_checkpoint_static_components(self) -> FrozenMap:
        return self.projection(
            USAuthorityProjection.STACKED_CHECKPOINT_STATIC_COMPONENTS
        )

    @property
    def nodes(self) -> tuple[CompiledNode, ...]:
        """Return compiled nodes in the compiler's deterministic total order."""

        return self._nodes

    def nodes_matching(self, query: NodeQuery) -> tuple[CompiledNode, ...]:
        """Return every ordered node matching the supplied typed fields."""

        if not isinstance(query, NodeQuery):
            raise TypeError("nodes_matching requires NodeQuery")
        return tuple(node for node in self._nodes if _node_matches(node, query))

    def require_node(self, query: NodeQuery) -> CompiledNode:
        """Return exactly one typed node match, refusing absence or ambiguity."""

        matches = self.nodes_matching(query)
        if len(matches) != 1:
            raise USSpecAuthorityError(
                "US compiled-node query must match exactly one node; "
                f"matched {len(matches)}"
            )
        return matches[0]


def compile_us_spec_authority(
    authorities: RuntimeAuthorities,
) -> USSpecAuthority:
    """Narrow a compiler-issued generation-1 capability to the US runtime."""

    if not isinstance(authorities, RuntimeAuthorities):
        raise TypeError("compile_us_spec_authority requires RuntimeAuthorities")
    if authorities.spec_binding.country != "us":
        raise USSpecAuthorityError(
            "US spec authority requires a runtime capability for country 'us'"
        )
    if authorities.identity_generation != 1:
        raise USSpecAuthorityError("US spec authority requires identity_generation 1")
    provenance = authorities.run_provenance_identity(
        run_request={}, execution_receipt={}
    ).to_wire()
    binding = provenance.get("spec_binding")
    if (
        provenance.get("identity_generation") != 1
        or not isinstance(binding, dict)
        or binding.get("attestation") != "bundle-authoritative"
    ):
        raise USSpecAuthorityError(
            "US spec authority requires a bundle-authoritative capability"
        )
    if authorities.execution_abi.get("present") is not True:
        raise USSpecAuthorityError(
            "US spec authority requires a present physical execution ABI"
        )

    behavior = _behavior_authority(authorities)
    bundle = behavior.get(ResourceKind.BUNDLE.value)
    if not isinstance(bundle, FrozenMap) or bundle.get("identity_generation") != 1:
        raise USSpecAuthorityError(
            "US behavior authority does not bind identity_generation 1"
        )
    for projection in USAuthorityProjection:
        authorities.projection(projection.value)

    return USSpecAuthority(
        authority_sha256=authorities.authority_sha256,
        spec_sha256=authorities.spec_binding.spec_sha256,
        identity_generation=authorities.identity_generation,
        _behavior=behavior,
        _projections=authorities.projections,
        _declared_sources=authorities.declared_sources,
        _nodes=authorities.nodes,
    )


__all__ = [
    "NodePort",
    "NodeQuery",
    "USAuthorityProjection",
    "USSpecAuthority",
    "USSpecAuthorityError",
    "compile_us_spec_authority",
]
