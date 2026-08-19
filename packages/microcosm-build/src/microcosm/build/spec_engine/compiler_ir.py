"""Compile a resolved country bundle into the immutable F0 plan IR.

This module is deliberately a compiler only.  It does not import the legacy
runtime, construct execution authority, or run a kernel.  Its outputs are the
lossless producer graph, a deterministic stage DAG, the selected seed
protocol's complete owner map, and static transitive node slices.  F1 can use
those slices when it adds run inputs and artifact hashes to form executable
reuse keys; F0 records only the configuration-side mirror.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .canonical import sha256_json
from .model import (
    FrozenMap,
    FrozenValue,
    ResolvedSpec,
    ResourceKind,
    SeedSiteOwnerKind,
    SpecBinding,
    Surface,
    freeze_json,
    thaw_json,
)
from .resolver import F0_KERNEL_REGISTRY
from .typed_closure import TypedClosureError, compile_producer_outputs

COMPILER_IR_ABI_VERSION = 2
EXECUTOR_CONTRACT_ABI = "compiled-node-direct-contracts-v1"
ROW_CLASSIFIER_IMPLEMENTATION_DOMAIN = (
    "microcosm.spec-engine.row-classifier-implementation.v1"
)
_NODE_SLICE_DOMAIN = "microcosm.spec-engine.node-slice.v1"
_NODE_KEY_DOMAIN = "microcosm.spec-engine.static-node-key.v1"
_NO_WRITE_ACTIONS = frozenset(
    {
        "consume_only_byte_exact_noop",
        "origin_projection_masked_noop",
        "producer_masked_byte_exact_noop",
        "scope_masked_noop",
    }
)


class CompilerIRError(ValueError):
    """A resolved bundle cannot be represented as a closed compiled plan."""


def _frozen_mapping(value: object, *, location: str) -> FrozenMap:
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenMap):
        raise CompilerIRError(f"{location}: object required")
    return frozen


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CompilerIRError(f"{location}: object required")
    return value


def _array(value: object, *, location: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise CompilerIRError(f"{location}: array required")
    return value


def _string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompilerIRError(f"{location}: non-empty string required")
    return value


def _wire(value: FrozenValue) -> object:
    return thaw_json(value)


@dataclass(frozen=True, slots=True)
class CompilerIRABI:
    version: int
    sha256: str
    source_inventory: tuple[tuple[str, str], ...]

    def to_wire(self) -> dict[str, object]:
        return {"version": self.version, "sha256": self.sha256}


def _compiler_ir_abi() -> CompilerIRABI:
    """Attest the compiler implementation without machine-local paths."""

    module_paths = (
        (
            "microcosm.build.spec_engine.compiler_ir",
            Path(__file__).resolve(),
        ),
        (
            "microcosm.build.spec_engine.executor",
            Path(__file__).resolve().with_name("executor.py"),
        ),
        (
            "microcosm.build.spec_engine.scope_algebra",
            Path(__file__).resolve().with_name("scope_algebra.py"),
        ),
        (
            "microcosm.build.spec_engine.typed_closure",
            Path(__file__).resolve().with_name("typed_closure.py"),
        ),
    )
    inventory = tuple(
        (module, hashlib.sha256(path.read_bytes()).hexdigest())
        for module, path in module_paths
    )
    digest = sha256_json(
        {
            "version": COMPILER_IR_ABI_VERSION,
            "source_inventory": [
                {"module": module, "sha256": source_sha256}
                for module, source_sha256 in inventory
            ],
            "contracts": {
                "graph": "raw-dependencies-plus-closed-cell-write-scopes-v1",
                "node_slice": _NODE_SLICE_DOMAIN,
                "row_classifier": ROW_CLASSIFIER_IMPLEMENTATION_DOMAIN,
                "seed_map": "legacy-v1-exhaustive-owner-map-v1",
                "executor": EXECUTOR_CONTRACT_ABI,
            },
        }
    )
    return CompilerIRABI(
        version=COMPILER_IR_ABI_VERSION,
        sha256=digest,
        source_inventory=inventory,
    )


def current_compiler_ir_abi() -> CompilerIRABI:
    """Return the ABI attestation for the currently installed executor stack."""

    return _compiler_ir_abi()


def row_classifier_contract(
    compiler_ir_abi: CompilerIRABI,
    scope_registry: FrozenMap,
) -> tuple[str, str]:
    """Return the closed classifier reference and implementation attestation."""

    registry_wire = _mapping(
        _wire(scope_registry), location="producer_graph/scope_registry"
    )
    predicate_space = _string(
        registry_wire.get("predicate_space"),
        location="producer_graph/scope_registry/predicate_space",
    )
    classifier_ref = f"classifier:{predicate_space}"
    implementation_sha256 = sha256_json(
        {
            "domain": ROW_CLASSIFIER_IMPLEMENTATION_DOMAIN,
            "compiler_ir_abi": compiler_ir_abi.to_wire(),
            "scope_registry": registry_wire,
        }
    )
    return classifier_ref, implementation_sha256


def kernel_implementation_contract(kernel_ref: str) -> str:
    """Return the installed compiler registry's pin for one kernel reference.

    The resolved authored node owns the reference.  The installed compiler
    owns the closed implementation namespace used to derive its pin.  Keeping
    this derivation in one place lets the executor independently re-derive the
    direct ``CompiledNode`` lift instead of trusting self-consistent hashes on
    an object supplied at dispatch time.
    """

    if not isinstance(kernel_ref, str) or not kernel_ref.startswith("kernel:"):
        raise CompilerIRError(f"invalid executable kernel reference {kernel_ref!r}")
    if not F0_KERNEL_REGISTRY.contains(kernel_ref):
        raise CompilerIRError(f"unknown executable kernel reference {kernel_ref!r}")
    if not F0_KERNEL_REGISTRY.has_implementation(kernel_ref):
        raise CompilerIRError(
            f"kernel reference {kernel_ref!r} has no executable implementation"
        )
    return sha256_json(
        {
            "registry_sha256": F0_KERNEL_REGISTRY.implementation_sha256,
            "kernel": kernel_ref,
        }
    )


@dataclass(frozen=True, slots=True)
class StageDagNode:
    id: str
    kind: str
    kernel: str
    depends_on: tuple[str, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "kernel": self.kernel,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True, slots=True)
class StageDag:
    external_stages: tuple[str, ...]
    nodes: tuple[StageDagNode, ...]
    edges: tuple[tuple[str, str], ...]
    waves: tuple[tuple[str, ...], ...]
    order: tuple[str, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "external_stages": list(self.external_stages),
            "nodes": [node.to_wire() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
            "waves": [list(wave) for wave in self.waves],
            "order": list(self.order),
        }


@dataclass(frozen=True, slots=True)
class ProducerNodeIR:
    id: str
    name: str
    kind: str
    kernel: str
    source: FrozenMap
    capabilities: FrozenMap
    mutations: FrozenMap
    depends_on: tuple[str, ...]
    inputs: tuple[FrozenMap, ...]
    outputs: tuple[FrozenMap, ...]
    write_scopes: tuple[FrozenMap, ...]

    def compiled_overlay_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "depends_on": list(self.depends_on),
            "compiled_outputs": [_wire(output) for output in self.outputs],
            "write_scopes": [_wire(scope) for scope in self.write_scopes],
        }

    def schedule_contract_wire(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": [_wire(row) for row in self.inputs],
            "outputs": [
                {
                    "entity": output["entity"],
                    "column": output["column"],
                    "coverage_scope": output["coverage_scope"],
                }
                for output in (
                    _mapping(_wire(row), location="compiled output")
                    for row in self.outputs
                )
            ],
        }


@dataclass(frozen=True, slots=True)
class ProducerGraphIR:
    present: bool
    authored: FrozenMap | None
    scope_registry: FrozenMap | None
    nodes: tuple[ProducerNodeIR, ...]
    external_stages: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    waves: tuple[tuple[str, ...], ...]
    order: tuple[str, ...]
    incomparable_node_policy: FrozenMap
    schedule_sha256: str

    @property
    def ownership_matrix(self) -> tuple[Mapping[str, object], ...]:
        if self.authored is None:
            return ()
        wire = _mapping(_wire(self.authored), location="producer_graph")
        return tuple(
            _mapping(value, location="producer_graph/ownership_matrix")
            for value in _array(
                wire.get("ownership_matrix", []),
                location="producer_graph/ownership_matrix",
            )
        )

    @property
    def compiled_output_count(self) -> int:
        return sum(len(node.outputs) for node in self.nodes)

    def schedule_wire(self) -> dict[str, object]:
        if self.authored is None:
            return {
                "schema_version": 0,
                "external_stages": [],
                "scope_coverage": {},
                "contracts": [],
                "edges": [],
                "waves": [],
                "order": [],
            }
        authored = _mapping(_wire(self.authored), location="producer_graph")
        return {
            "schema_version": authored["graph_schema_version"],
            "external_stages": list(self.external_stages),
            "scope_coverage": authored["scope_coverage"],
            "contracts": [
                node.schedule_contract_wire()
                for node in sorted(self.nodes, key=lambda item: item.name)
            ],
            "edges": [list(edge) for edge in self.edges],
            "waves": [list(wave) for wave in self.waves],
            "order": list(self.order),
        }

    def to_wire(self) -> dict[str, object]:
        if not self.present:
            return {
                "present": False,
                "nodes": [],
                "edges": [],
                "waves": [],
                "order": [],
                "incomparable_node_policy": _wire(self.incomparable_node_policy),
                "schedule_sha256": self.schedule_sha256,
            }
        assert self.authored is not None
        return {
            "present": True,
            "authored": _wire(self.authored),
            "nodes": [node.compiled_overlay_wire() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
            "waves": [list(wave) for wave in self.waves],
            "order": list(self.order),
            "incomparable_node_policy": _wire(self.incomparable_node_policy),
            "schedule_sha256": self.schedule_sha256,
        }


@dataclass(frozen=True, slots=True)
class SeedSiteIR:
    id: str
    stream: str
    contract: FrozenMap
    owners: tuple[tuple[str, str], ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "stream": f"stream:{self.stream}",
            "contract": _wire(self.contract),
            "owners": [
                {"kind": kind, "id": owner_id} for kind, owner_id in self.owners
            ],
        }


@dataclass(frozen=True, slots=True)
class SeedOwnerIR:
    kind: str
    id: str
    sites: tuple[str, ...]
    streams: tuple[str, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "id": self.id,
            "sites": list(self.sites),
            "streams": [f"stream:{stream}" for stream in self.streams],
        }


@dataclass(frozen=True, slots=True)
class SeedStreamMap:
    protocol_id: str
    implementation_id: str
    implementation_sha256: str
    sites: tuple[SeedSiteIR, ...]
    owners: tuple[SeedOwnerIR, ...]

    def owner(self, kind: str, owner_id: str) -> SeedOwnerIR | None:
        return next(
            (row for row in self.owners if row.kind == kind and row.id == owner_id),
            None,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "implementation_id": self.implementation_id,
            "implementation_sha256": self.implementation_sha256,
            "sites": [site.to_wire() for site in self.sites],
            "owners": [owner.to_wire() for owner in self.owners],
        }


@dataclass(frozen=True, slots=True)
class ResolvedParam:
    path: str
    value_sha256: str
    value: FrozenValue

    def to_wire(self) -> dict[str, object]:
        return {
            "path": self.path,
            "value_sha256": self.value_sha256,
            "value": _wire(self.value),
        }


@dataclass(frozen=True, slots=True)
class TransitiveNodeSlice:
    id: str
    local_slice_sha256: str

    def to_wire(self) -> dict[str, str]:
        return {"id": self.id, "local_slice_sha256": self.local_slice_sha256}


@dataclass(frozen=True, slots=True)
class CompiledNode:
    id: str
    execution_rank: int
    node_key: str
    node_slice_sha256: str
    kernel_ref: str
    kernel_implementation_sha256: str
    depends_on: tuple[str, ...]
    inputs: tuple[FrozenMap, ...]
    outputs: tuple[FrozenMap, ...]
    capabilities: FrozenMap
    mutations: FrozenMap
    write_scopes: tuple[FrozenMap, ...]
    scope_registry: FrozenMap
    row_classifier_ref: str
    row_classifier_implementation_sha256: str
    compiler_ir_abi: FrozenMap
    seed_protocol_sha256: str
    seed_sites: tuple[SeedSiteIR, ...]
    seed_streams: tuple[str, ...]
    resolved_params: tuple[ResolvedParam, ...]
    transitive_nodes: tuple[TransitiveNodeSlice, ...]

    def node_slice_wire(self) -> dict[str, object]:
        return {
            "domain": _NODE_SLICE_DOMAIN,
            "resolved_params": [row.to_wire() for row in self.resolved_params],
            "transitive_nodes": [row.to_wire() for row in self.transitive_nodes],
        }

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "execution_rank": self.execution_rank,
            "node_key": self.node_key,
            "node_slice_sha256": self.node_slice_sha256,
            "kernel": {
                "ref": self.kernel_ref,
                "implementation_sha256": self.kernel_implementation_sha256,
            },
            "depends_on": list(self.depends_on),
            "inputs": [_wire(row) for row in self.inputs],
            "outputs": [_wire(row) for row in self.outputs],
            "row_classifier": {
                "ref": self.row_classifier_ref,
                "implementation_sha256": (self.row_classifier_implementation_sha256),
            },
            "seed_streams": [f"stream:{stream}" for stream in self.seed_streams],
            "node_slice": self.node_slice_wire(),
        }


@dataclass(frozen=True, slots=True)
class CompiledSpecIR:
    spec_binding: SpecBinding
    compiler_ir_abi: CompilerIRABI
    normalized_resources: FrozenMap
    surfaces: FrozenMap
    typed_inventory: FrozenMap
    generated_authorities: FrozenMap
    vintage_authorities: FrozenMap
    stage_dag: StageDag
    producer_graph: ProducerGraphIR
    seed_stream_map: SeedStreamMap
    nodes: tuple[CompiledNode, ...]

    def resources_wire(self) -> dict[str, object]:
        value = _wire(self.normalized_resources)
        assert isinstance(value, dict)
        return value

    def resource(self, kind: ResourceKind | str) -> Mapping[str, object]:
        selected = ResourceKind(kind).value
        return _mapping(
            self.resources_wire()[selected], location=f"resources/{selected}"
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "compiler_ir_abi": self.compiler_ir_abi.to_wire(),
            "spec_binding": self.spec_binding.to_wire(),
            "surfaces": _wire(self.surfaces),
            "typed_inventory": _wire(self.typed_inventory),
            "authorities": {
                "generated": _wire(self.generated_authorities),
                "vintages": _wire(self.vintage_authorities),
            },
            "stage_dag": self.stage_dag.to_wire(),
            "producer_graph": self.producer_graph.to_wire(),
            "seed_stream_map": self.seed_stream_map.to_wire(),
            "nodes": [node.to_wire() for node in self.nodes],
        }


def _normalized_resources(spec: ResolvedSpec) -> tuple[dict[str, object], FrozenMap]:
    resources: dict[str, object] = {}
    for resource in spec.resources:
        kind = resource.descriptor.kind
        if kind in {ResourceKind.SCHEMA, ResourceKind.LEGACY_JSON}:
            continue
        if kind.value in resources:
            raise CompilerIRError(
                f"resources: duplicate normalized domain {kind.value!r}"
            )
        resources[kind.value] = resource.domain.to_wire()
    frozen = _frozen_mapping(resources, location="resources")
    return resources, frozen


def _resolved_surfaces(spec: ResolvedSpec) -> FrozenMap:
    return _frozen_mapping(
        {
            surface.value: _wire(spec.surfaces.for_surface(surface))
            for surface in Surface
        },
        location="surfaces",
    )


def _typed_inventory(spec: ResolvedSpec) -> FrozenMap:
    return _frozen_mapping(
        {
            "entities": [{"id": entity.id} for entity in spec.entities],
            "artifacts": [
                {
                    "id": artifact.id,
                    "kind": artifact.kind,
                    "producing_stages": list(artifact.producing_stages),
                    "entity": (None if artifact.entity is None else artifact.entity.id),
                    "key": artifact.key,
                    "lifetime": artifact.lifetime,
                    "validation": artifact.validation,
                    "binding": _wire(artifact.binding),
                }
                for artifact in spec.artifacts
            ],
            "scopes": [
                {
                    "id": scope.id,
                    "predicate_space": scope.predicate_space,
                    "entity": None if scope.entity is None else scope.entity.id,
                    "predicate": _wire(scope.predicate),
                    "source_path": scope.source_path,
                }
                for scope in spec.scopes
            ],
            "columns": [
                {
                    "key": column.key,
                    "entity": column.entity.id,
                    "dtype": column.dtype,
                    "unit": column.unit,
                    "period": column.period,
                    "vintage": column.vintage,
                    "nullable": column.nullable,
                    "domain": column.domain,
                    "public_stability": column.public_stability,
                    "unit_waiver": column.unit_waiver,
                }
                for column in spec.columns
            ],
            "references": [
                {
                    "namespace": reference.namespace,
                    "id": reference.id,
                    "source_path": reference.source_path,
                }
                for reference in spec.references
            ],
        },
        location="typed_inventory",
    )


def _validate_ownership_matrix(
    graph: Mapping[str, object],
    *,
    node_by_id: Mapping[str, Mapping[str, object]],
    compiled_outputs: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[Mapping[str, object], ...]:
    rows = tuple(
        _mapping(value, location=f"producer_graph/ownership_matrix/{index}")
        for index, value in enumerate(
            _array(
                graph.get("ownership_matrix", []),
                location="producer_graph/ownership_matrix",
            )
        )
    )
    seen: set[tuple[str, str, str, int]] = set()
    for index, row in enumerate(rows):
        location = f"producer_graph/ownership_matrix/{index}"
        entity = _string(row.get("entity"), location=f"{location}/entity")
        target = _string(row.get("target"), location=f"{location}/target")
        origin = _string(row.get("origin"), location=f"{location}/origin")
        clone_index = row.get("clone_index")
        if isinstance(clone_index, bool) or not isinstance(clone_index, int):
            raise CompilerIRError(f"{location}/clone_index: integer required")
        key = (entity, target, origin, clone_index)
        if key in seen:
            raise CompilerIRError(f"{location}: duplicate ownership cell {key!r}")
        seen.add(key)

        final_owner = _string(
            row.get("final_owner"), location=f"{location}/final_owner"
        )
        if final_owner not in node_by_id:
            raise CompilerIRError(
                f"{location}/final_owner: dangling producer {final_owner!r}"
            )
        actions = tuple(
            _mapping(value, location=f"{location}/producer_actions/{action_index}")
            for action_index, value in enumerate(
                _array(
                    row.get("producer_actions", []),
                    location=f"{location}/producer_actions",
                )
            )
        )
        producers: set[str] = set()
        final_actions: list[str] = []
        for action_index, action in enumerate(actions):
            action_location = f"{location}/producer_actions/{action_index}"
            producer = _string(
                action.get("producer"), location=f"{action_location}/producer"
            )
            if producer in producers:
                raise CompilerIRError(
                    f"{action_location}/producer: duplicate producer {producer!r}"
                )
            producers.add(producer)
            if producer not in node_by_id:
                raise CompilerIRError(
                    f"{action_location}/producer: dangling producer {producer!r}"
                )
            action_kind = _string(
                action.get("action"), location=f"{action_location}/action"
            )
            if action_kind not in _NO_WRITE_ACTIONS and not any(
                output.get("entity") == entity and output.get("column") == target
                for output in compiled_outputs[producer]
            ):
                raise CompilerIRError(
                    f"{action_location}/producer: producer {producer!r} does not "
                    f"declare output {entity}.{target}"
                )
            owns_final = action.get("owns_final")
            if not isinstance(owns_final, bool):
                raise CompilerIRError(f"{action_location}/owns_final: boolean required")
            if owns_final:
                final_actions.append(producer)
        if final_actions != [final_owner]:
            raise CompilerIRError(
                f"{location}: exactly one owns_final action must equal final_owner; "
                f"expected={final_owner!r}, actual={final_actions!r}"
            )
    return rows


def _write_scope(
    *,
    producer: str,
    output: Mapping[str, object],
    ownership_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    entity = _string(output.get("entity"), location="compiled output/entity")
    column = _string(output.get("column"), location="compiled output/column")
    coverage_scope = _string(
        output.get("coverage_scope"), location="compiled output/coverage_scope"
    )
    matching = [
        row
        for row in ownership_rows
        if row.get("entity") == entity and row.get("target") == column
    ]
    segments: list[dict[str, object]] = []
    for row in matching:
        actions = [
            _mapping(value, location="ownership producer action")
            for value in _array(
                row.get("producer_actions", []),
                location="ownership producer_actions",
            )
            if _mapping(value, location="ownership producer action").get("producer")
            == producer
        ]
        if len(actions) != 1:
            raise CompilerIRError(
                f"ownership row {entity}.{column} does not name {producer!r} once"
            )
        action = _string(
            actions[0].get("action"), location="ownership producer action/action"
        )
        if action not in _NO_WRITE_ACTIONS:
            segments.append(
                {
                    "predicate": "origin_clone",
                    "origin": row["origin"],
                    "clone_index": row["clone_index"],
                    "write_policy": action,
                }
            )
    if not matching:
        segments.append(
            {
                "predicate": "coverage_scope",
                "coverage_scope": coverage_scope,
                "write_policy": "declared_output_write",
            }
        )
    if not segments:
        raise CompilerIRError(
            f"producer {producer!r} declares {entity}.{column} but owns no cells"
        )
    if column == "@resolved_weight":
        mode = "resolved_weight"
    elif column.startswith("@"):
        mode = "virtual_receipt"
    elif column.endswith("_id") or "support_" in column:
        mode = "structural_column"
    else:
        mode = "column_cells"
    return {
        "entity": entity,
        "column": column,
        "row_scope": coverage_scope,
        "mode": mode,
        "cell_segments": segments,
    }


def _segment_atoms(
    segment: Mapping[str, object],
    *,
    scope_coverage: Mapping[str, object],
) -> set[tuple[str, int]]:
    if segment.get("predicate") == "origin_clone":
        return {(str(segment["origin"]), int(segment["clone_index"]))}
    declared = _mapping(
        scope_coverage.get("declared", {}), location="scope_coverage/declared"
    )
    scope = str(segment.get("coverage_scope"))
    values = _array(declared.get(scope, [scope]), location=f"scope_coverage/{scope}")
    scopes = {str(value) for value in values}
    if "whole_pool" in scopes:
        return {
            (origin, clone_index)
            for origin in ("asec", "acs")
            for clone_index in (0, 1, 2)
        }
    result: set[tuple[str, int]] = set()
    if "asec_source" in scopes:
        result.add(("asec", 0))
    if "acs_source" in scopes:
        result.add(("acs", 0))
    if "puf_clone" in scopes:
        result.update(
            (origin, clone_index)
            for origin in ("asec", "acs")
            for clone_index in (1, 2)
        )
    if "receipt" in scopes:
        result.add(("receipt", 0))
    return result


def _incomparable_policy(
    *,
    nodes: Sequence[ProducerNodeIR],
    edges: Sequence[tuple[str, str]],
    scope_coverage: Mapping[str, object],
) -> FrozenMap:
    reachable = {node.id: set() for node in nodes}
    for producer, consumer in edges:
        reachable[producer].add(consumer)
    changed = True
    while changed:
        changed = False
        for node_id in reachable:
            expanded = set(reachable[node_id])
            for child in tuple(reachable[node_id]):
                expanded.update(reachable[child])
            if expanded != reachable[node_id]:
                reachable[node_id] = expanded
                changed = True

    incomparable = 0
    disjoint = 0
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            if right.id in reachable[left.id] or left.id in reachable[right.id]:
                continue
            incomparable += 1
            overlap = False
            for left_value in left.write_scopes:
                left_scope = _mapping(_wire(left_value), location="left write scope")
                for right_value in right.write_scopes:
                    right_scope = _mapping(
                        _wire(right_value), location="right write scope"
                    )
                    if (left_scope["entity"], left_scope["column"]) != (
                        right_scope["entity"],
                        right_scope["column"],
                    ):
                        continue
                    for left_segment_value in _array(
                        left_scope["cell_segments"], location="left cell segments"
                    ):
                        left_segment = _mapping(
                            left_segment_value, location="left cell segment"
                        )
                        for right_segment_value in _array(
                            right_scope["cell_segments"],
                            location="right cell segments",
                        ):
                            right_segment = _mapping(
                                right_segment_value, location="right cell segment"
                            )
                            if _segment_atoms(
                                left_segment, scope_coverage=scope_coverage
                            ) & _segment_atoms(
                                right_segment, scope_coverage=scope_coverage
                            ):
                                overlap = True
                                break
                        if overlap:
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if overlap:
                raise CompilerIRError(
                    "incomparable producer nodes have overlapping exact writes: "
                    f"{left.id!r}, {right.id!r}"
                )
            disjoint += 1
    return _frozen_mapping(
        {
            "requirement": "commute_or_disjoint_writes",
            "proof_method": ("transitive_closure_and_closed_cell_segment_intersection"),
            "overlap_rule": "explicit_commutativity_proof_required",
            "commutativity_proofs": [],
            "incomparable_pair_count": incomparable,
            "disjoint_write_pair_count": disjoint,
        },
        location="incomparable_node_policy",
    )


def _compile_producer_graph(resources: Mapping[str, object]) -> ProducerGraphIR:
    imputation_value = resources.get("imputation")
    if imputation_value is None:
        empty_policy = _frozen_mapping(
            {
                "requirement": "commute_or_disjoint_writes",
                "proof_method": (
                    "transitive_closure_and_closed_cell_segment_intersection"
                ),
                "overlap_rule": "explicit_commutativity_proof_required",
                "commutativity_proofs": [],
                "incomparable_pair_count": 0,
                "disjoint_write_pair_count": 0,
            },
            location="incomparable_node_policy",
        )
        schedule = {
            "schema_version": 0,
            "external_stages": [],
            "scope_coverage": {},
            "contracts": [],
            "edges": [],
            "waves": [],
            "order": [],
        }
        return ProducerGraphIR(
            present=False,
            authored=None,
            scope_registry=None,
            nodes=(),
            external_stages=(),
            edges=(),
            waves=(),
            order=(),
            incomparable_node_policy=empty_policy,
            schedule_sha256=sha256_json(schedule),
        )
    imputation = _mapping(imputation_value, location="imputation")
    graph = _mapping(
        imputation.get("producer_graph", {}), location="imputation/producer_graph"
    )
    if not graph:
        # A defaulted shared-core imputation resource has no executable graph.
        resources_without_imputation = dict(resources)
        resources_without_imputation.pop("imputation", None)
        return _compile_producer_graph(resources_without_imputation)
    node_rows = tuple(
        _mapping(value, location=f"producer_graph/nodes/{index}")
        for index, value in enumerate(
            _array(graph.get("nodes", []), location="producer_graph/nodes")
        )
    )
    node_by_id: dict[str, Mapping[str, object]] = {}
    rank: dict[str, int] = {}
    for index, row in enumerate(node_rows):
        node_id = _string(row.get("id"), location=f"producer_graph/nodes/{index}/id")
        name = _string(row.get("name"), location=f"producer_graph/nodes/{index}/name")
        if node_id != name:
            raise CompilerIRError(f"producer_graph/nodes/{index}: id must equal name")
        if node_id in node_by_id:
            raise CompilerIRError(
                f"producer_graph/nodes/{index}/id: duplicate {node_id!r}"
            )
        node_by_id[node_id] = row
        rank[node_id] = index

    external_stages = tuple(
        sorted(
            {
                _string(value, location="producer_graph/external_stages")
                for value in _array(
                    graph.get("external_stages", []),
                    location="producer_graph/external_stages",
                )
            }
        )
    )
    if set(external_stages) & set(node_by_id):
        raise CompilerIRError(
            "producer_graph/external_stages: overlaps producer node ids"
        )
    try:
        compiled_outputs = compile_producer_outputs(resources)
    except TypedClosureError as error:
        raise CompilerIRError(str(error)) from error
    if set(compiled_outputs) != set(node_by_id):
        raise CompilerIRError(
            "producer_graph: compiled output keys do not equal producer node ids"
        )

    ownership_rows = _validate_ownership_matrix(
        graph,
        node_by_id=node_by_id,
        compiled_outputs=compiled_outputs,
    )
    scope_coverage = _mapping(
        graph.get("scope_coverage", {}), location="producer_graph/scope_coverage"
    )
    scope_registry = _frozen_mapping(
        graph.get("scope_registry"),
        location="producer_graph/scope_registry",
    )
    declared_scopes = _mapping(
        scope_coverage.get("declared", {}),
        location="producer_graph/scope_coverage/declared",
    )
    predecessors: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    for node_id, node in node_by_id.items():
        for input_index, input_value in enumerate(
            _array(node.get("inputs", []), location=f"{node_id}/inputs")
        ):
            input_row = _mapping(
                input_value, location=f"{node_id}/inputs/{input_index}"
            )
            producing_stage = _string(
                input_row.get("producing_stage"),
                location=f"{node_id}/inputs/{input_index}/producing_stage",
            )
            if producing_stage in external_stages:
                continue
            if producing_stage not in node_by_id:
                raise CompilerIRError(
                    f"{node_id}/inputs/{input_index}/producing_stage: dangling "
                    f"producer {producing_stage!r}"
                )
            required_scope = _string(
                input_row.get("required_scope"),
                location=f"{node_id}/inputs/{input_index}/required_scope",
            )
            matching_outputs = [
                output
                for output in compiled_outputs[producing_stage]
                if output.get("entity") == input_row.get("entity")
                and output.get("column") == input_row.get("column")
            ]
            if not any(
                required_scope
                in {
                    str(scope)
                    for scope in _array(
                        declared_scopes.get(
                            str(output.get("coverage_scope")),
                            [output.get("coverage_scope")],
                        ),
                        location="scope_coverage",
                    )
                }
                for output in matching_outputs
            ):
                raise CompilerIRError(
                    f"producer input {node_id}/{input_row.get('entity')}."
                    f"{input_row.get('column')} has no scope-compatible "
                    f"output from {producing_stage!r}"
                )
            predecessors[node_id].add(producing_stage)

    edges = tuple(
        sorted(
            (
                (producer, consumer)
                for consumer, producer_ids in predecessors.items()
                for producer in producer_ids
            ),
            key=lambda edge: (rank[edge[0]], rank[edge[1]]),
        )
    )
    successors: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    indegree = {node_id: len(values) for node_id, values in predecessors.items()}
    for producer, consumer in edges:
        successors[producer].add(consumer)
    remaining = set(node_by_id)
    waves: list[tuple[str, ...]] = []
    while remaining:
        wave = tuple(
            sorted(
                (node_id for node_id in remaining if indegree[node_id] == 0),
                key=rank.__getitem__,
            )
        )
        if not wave:
            raise CompilerIRError(
                f"producer_graph: dependency cycle among {sorted(remaining)!r}"
            )
        waves.append(wave)
        remaining.difference_update(wave)
        for producer in wave:
            for consumer in successors[producer]:
                indegree[consumer] -= 1
    order = tuple(node_id for wave in waves for node_id in wave)

    nodes: list[ProducerNodeIR] = []
    for node_id in order:
        node = node_by_id[node_id]
        kernel = _string(node.get("kernel"), location=f"{node_id}/kernel")
        if not kernel.startswith("kernel:") or not F0_KERNEL_REGISTRY.contains(kernel):
            raise CompilerIRError(
                f"{node_id}/kernel: not pinned by the F0 kernel registry: {kernel!r}"
            )
        if not F0_KERNEL_REGISTRY.has_implementation(kernel):
            raise CompilerIRError(
                f"{node_id}/kernel: contract-only F0 kernel has no producer "
                f"implementation pin: {kernel!r}"
            )
        input_rows = tuple(
            _frozen_mapping(value, location=f"{node_id}/inputs")
            for value in _array(node.get("inputs", []), location=f"{node_id}/inputs")
        )
        output_rows = tuple(
            _frozen_mapping(value, location=f"{node_id}/compiled_outputs")
            for value in compiled_outputs[node_id]
        )
        write_scopes = tuple(
            _frozen_mapping(
                _write_scope(
                    producer=node_id,
                    output=_mapping(_wire(output), location="compiled output"),
                    ownership_rows=ownership_rows,
                ),
                location=f"{node_id}/write_scopes",
            )
            for output in output_rows
        )
        nodes.append(
            ProducerNodeIR(
                id=node_id,
                name=str(node["name"]),
                kind=_string(node.get("kind"), location=f"{node_id}/kind"),
                kernel=kernel,
                source=_frozen_mapping(node, location=f"{node_id}/source"),
                capabilities=_frozen_mapping(
                    node.get("capabilities"),
                    location=f"{node_id}/capabilities",
                ),
                mutations=_frozen_mapping(
                    node.get("mutations"),
                    location=f"{node_id}/mutations",
                ),
                depends_on=tuple(sorted(predecessors[node_id], key=rank.__getitem__)),
                inputs=input_rows,
                outputs=output_rows,
                write_scopes=write_scopes,
            )
        )
    policy = _incomparable_policy(
        nodes=nodes,
        edges=edges,
        scope_coverage=scope_coverage,
    )
    authored = _frozen_mapping(graph, location="producer_graph")
    provisional = ProducerGraphIR(
        present=True,
        authored=authored,
        scope_registry=scope_registry,
        nodes=tuple(nodes),
        external_stages=external_stages,
        edges=edges,
        waves=tuple(waves),
        order=order,
        incomparable_node_policy=policy,
        schedule_sha256="",
    )
    return ProducerGraphIR(
        present=True,
        authored=authored,
        scope_registry=scope_registry,
        nodes=tuple(nodes),
        external_stages=external_stages,
        edges=edges,
        waves=tuple(waves),
        order=order,
        incomparable_node_policy=policy,
        schedule_sha256=sha256_json(provisional.schedule_wire()),
    )


def _compile_seed_stream_map(
    spec: ResolvedSpec,
    *,
    resources: Mapping[str, object],
    producer_graph: ProducerGraphIR,
) -> SeedStreamMap:
    protocol = spec.seed_protocol
    protocol_sites = {site.id: site for site in protocol.sites}
    bindings = {binding.site: binding for binding in spec.seed_site_bindings}
    if len(bindings) != len(spec.seed_site_bindings):
        raise CompilerIRError("seed_site_bindings: duplicate site")
    if bindings and set(bindings) != set(protocol_sites):
        raise CompilerIRError(
            "seed_site_bindings: non-empty binding map must cover the selected "
            "protocol exactly"
        )

    source_stage_ids = {
        str(row.get("stage"))
        for row in (
            _mapping(value, location="sources/stages")
            for value in _array(
                _mapping(resources.get("sources", {}), location="sources").get(
                    "stages", []
                ),
                location="sources/stages",
            )
        )
    }
    pipeline_operation_ids: set[str] = set()
    spine = _mapping(resources.get("spine", {}), location="spine")
    pipeline_value = spine.get("pipeline_contract")
    if pipeline_value is not None:
        pipeline = _mapping(pipeline_value, location="spine/pipeline_contract")
        for field in (
            "stacked_operator_order",
            "pre_clone_source_operator_order",
            "post_clone_source_operator_order",
            "derive_operator_order",
            "auxiliary_operations",
        ):
            pipeline_operation_ids.update(
                str(value)
                for value in _array(
                    pipeline.get(field, []),
                    location=f"spine/pipeline_contract/{field}",
                )
            )
    valid_owners = {
        SeedSiteOwnerKind.PRODUCER_NODE.value: set(producer_graph.order),
        SeedSiteOwnerKind.SOURCE_STAGE.value: source_stage_ids,
        SeedSiteOwnerKind.PIPELINE_OPERATION.value: pipeline_operation_ids,
    }
    owner_sites: dict[tuple[str, str], list[str]] = {}
    site_rows: list[SeedSiteIR] = []
    for site in protocol.sites:
        binding = bindings.get(site.id)
        owners = () if binding is None else binding.owners
        owner_rows: list[tuple[str, str]] = []
        for owner in owners:
            kind = owner.kind.value
            if owner.id not in valid_owners[kind]:
                raise CompilerIRError(
                    f"seed site {site.id!r}: dangling {kind} owner {owner.id!r}"
                )
            key = (kind, owner.id)
            if key in owner_rows:
                raise CompilerIRError(
                    f"seed site {site.id!r}: duplicate owner {kind}:{owner.id}"
                )
            owner_rows.append(key)
            owner_sites.setdefault(key, []).append(site.id)
        contract = site.to_wire()
        contract.pop("id")
        contract.pop("stream")
        site_rows.append(
            SeedSiteIR(
                id=site.id,
                stream=site.stream,
                contract=_frozen_mapping(contract, location=f"seed site {site.id}"),
                owners=tuple(owner_rows),
            )
        )
    owner_rows = tuple(
        SeedOwnerIR(
            kind=kind,
            id=owner_id,
            sites=tuple(site_ids),
            streams=tuple(
                dict.fromkeys(protocol_sites[site_id].stream for site_id in site_ids)
            ),
        )
        for (kind, owner_id), site_ids in sorted(owner_sites.items())
    )
    return SeedStreamMap(
        protocol_id=protocol.id,
        implementation_id=protocol.implementation_id,
        implementation_sha256=protocol.implementation_sha256,
        sites=tuple(site_rows),
        owners=owner_rows,
    )


def _resolved_param(
    path: str,
    value: object,
) -> ResolvedParam:
    frozen = freeze_json(value)
    return ResolvedParam(
        path=path,
        value_sha256=sha256_json(value),
        value=frozen,
    )


def _node_take_up_params(
    node: ProducerNodeIR,
    *,
    resources: Mapping[str, object],
) -> tuple[ResolvedParam, ...]:
    """Resolve take-up rows whose typed output variable is owned by ``node``.

    The relation is an exact column join, not an id/name convention.  It is
    needed for mixed measured/transferred surfaces whose source-stage pointer
    lives in the take-up contract rather than in a producer virtual resource.
    """

    take_up_value = resources.get("take_up")
    if take_up_value is None:
        return ()
    take_up = _mapping(take_up_value, location="take_up")
    output_columns = {
        str(output["column"])
        for output in (
            _mapping(_wire(value), location="compiled output") for value in node.outputs
        )
        if not str(output["column"]).startswith("@")
    }
    result: list[ResolvedParam] = []
    for index, program_value in enumerate(
        _array(take_up.get("programs", []), location="take_up/programs")
    ):
        program = _mapping(program_value, location=f"take_up/programs/{index}")
        variable = program.get("variable")
        if isinstance(variable, str) and variable in output_columns:
            result.append(_resolved_param(f"/take_up/programs/{index}", program))
    return tuple(result)


def _typed_seed_owner_references(
    params: Sequence[ResolvedParam],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Find only schema-typed seed-owner references in resolved node inputs.

    Owner ids are ordinary strings on the wire, so a recursive value/name
    search would be authority by coincidence.  These four field shapes are
    the closed reference grammar used by producer and take-up contracts.
    """

    locations: dict[tuple[str, str], list[str]] = {}

    def add(kind: str, owner_id: object, *, location: str) -> None:
        if not isinstance(owner_id, str) or not owner_id:
            raise CompilerIRError(f"{location}: non-empty {kind} reference required")
        paths = locations.setdefault((kind, owner_id), [])
        if location not in paths:
            paths.append(location)

    def walk(value: object, *, location: str) -> None:
        if isinstance(value, Mapping):
            if "source_stage_ref" in value:
                reference = value["source_stage_ref"]
                if reference is not None:
                    reference_row = _mapping(
                        reference,
                        location=f"{location}/source_stage_ref",
                    )
                    add(
                        SeedSiteOwnerKind.SOURCE_STAGE.value,
                        reference_row.get("stage_id"),
                        location=f"{location}/source_stage_ref/stage_id",
                    )
            if "source_operation_ref" in value:
                reference_row = _mapping(
                    value["source_operation_ref"],
                    location=f"{location}/source_operation_ref",
                )
                add(
                    SeedSiteOwnerKind.SOURCE_STAGE.value,
                    reference_row.get("stage"),
                    location=f"{location}/source_operation_ref/stage",
                )
            if "source_operator" in value:
                add(
                    SeedSiteOwnerKind.PIPELINE_OPERATION.value,
                    value["source_operator"],
                    location=f"{location}/source_operator",
                )
            if "source_operator_registry" in value:
                for index, owner_id in enumerate(
                    _array(
                        value["source_operator_registry"],
                        location=f"{location}/source_operator_registry",
                    )
                ):
                    add(
                        SeedSiteOwnerKind.PIPELINE_OPERATION.value,
                        owner_id,
                        location=f"{location}/source_operator_registry/{index}",
                    )
            for key, child in value.items():
                walk(child, location=f"{location}/{key}")
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                walk(child, location=f"{location}/{index}")

    for param in params:
        walk(_wire(param.value), location=param.path)
    return tuple(
        (kind, owner_id, tuple(paths))
        for (kind, owner_id), paths in sorted(locations.items())
    )


def _seed_reference_params(
    references: Sequence[tuple[str, str, tuple[str, ...]]],
    *,
    resources: Mapping[str, object],
) -> tuple[ResolvedParam, ...]:
    """Bind the normalized records that typed owner references resolve to."""

    source_rows = {
        _string(row.get("stage"), location=f"sources/stages/{index}/stage"): (
            index,
            row,
        )
        for index, row in enumerate(
            _mapping(value, location=f"sources/stages/{index}")
            for index, value in enumerate(
                _array(
                    _mapping(resources.get("sources", {}), location="sources").get(
                        "stages", []
                    ),
                    location="sources/stages",
                )
            )
        )
    }
    spine = _mapping(resources.get("spine", {}), location="spine")
    pipeline_value = spine.get("pipeline_contract")
    pipeline = (
        _mapping(pipeline_value, location="spine/pipeline_contract")
        if pipeline_value is not None
        else {}
    )
    pipeline_fields = (
        "stacked_operator_order",
        "pre_clone_source_operator_order",
        "post_clone_source_operator_order",
        "derive_operator_order",
        "auxiliary_operations",
    )
    pipeline_memberships: dict[str, list[dict[str, object]]] = {}
    for field in pipeline_fields:
        for index, operation in enumerate(
            _array(
                pipeline.get(field, []),
                location=f"spine/pipeline_contract/{field}",
            )
        ):
            operation_id = _string(
                operation,
                location=f"spine/pipeline_contract/{field}/{index}",
            )
            pipeline_memberships.setdefault(operation_id, []).append(
                {"field": field, "index": index}
            )

    result: list[ResolvedParam] = []
    seen: set[tuple[str, str]] = set()
    for kind, owner_id, _ in references:
        key = (kind, owner_id)
        if key in seen:
            continue
        seen.add(key)
        if kind == SeedSiteOwnerKind.SOURCE_STAGE.value:
            if owner_id not in source_rows:
                raise CompilerIRError(
                    f"typed source_stage reference is dangling: {owner_id!r}"
                )
            index, row = source_rows[owner_id]
            result.append(_resolved_param(f"/sources/stages/{index}", row))
        elif kind == SeedSiteOwnerKind.PIPELINE_OPERATION.value:
            memberships = pipeline_memberships.get(owner_id)
            if memberships is None:
                raise CompilerIRError(
                    f"typed pipeline_operation reference is dangling: {owner_id!r}"
                )
            result.append(
                _resolved_param(
                    f"/spine/pipeline_contract@operation={owner_id}",
                    {"id": owner_id, "memberships": memberships},
                )
            )
        else:  # pragma: no cover - closed by the caller's grammar
            raise CompilerIRError(f"unknown seed-owner reference kind {kind!r}")
    return tuple(result)


def _effective_seed_grant(
    node: ProducerNodeIR,
    *,
    params: Sequence[ResolvedParam],
    seed_stream_map: SeedStreamMap,
) -> tuple[ResolvedParam, tuple[SeedSiteIR, ...], tuple[str, ...]]:
    """Compile the node's direct and typed-reference seed authority."""

    references = _typed_seed_owner_references(params)
    contributing: list[SeedOwnerIR] = []
    direct = seed_stream_map.owner(SeedSiteOwnerKind.PRODUCER_NODE.value, node.id)
    if direct is not None:
        contributing.append(direct)
    for kind, owner_id, _ in references:
        owner = seed_stream_map.owner(kind, owner_id)
        if owner is not None and owner not in contributing:
            contributing.append(owner)

    site_ids = {site_id for owner in contributing for site_id in owner.sites}
    sites = tuple(site for site in seed_stream_map.sites if site.id in site_ids)
    streams = tuple(dict.fromkeys(site.stream for site in sites))
    determinism = _string(
        node.capabilities.get("determinism"),
        location=f"{node.id}/capabilities/determinism",
    )
    if determinism == "seeded" and not sites:
        raise CompilerIRError(
            f"producer node {node.id!r} declares seeded determinism but has "
            "no effective seed-site grant"
        )
    if determinism == "deterministic" and sites:
        raise CompilerIRError(
            f"producer node {node.id!r} declares deterministic behavior but "
            "has an effective seed-site grant"
        )

    grant = _resolved_param(
        f"/compiled/effective_seed_grant@producer={node.id}",
        {
            "configuration_references": [
                {"kind": kind, "id": owner_id, "paths": list(paths)}
                for kind, owner_id, paths in references
            ],
            "grant_sources": [owner.to_wire() for owner in contributing],
            "sites": [site.to_wire() for site in sites],
        },
    )
    return grant, sites, streams


def _node_resolved_params(
    node: ProducerNodeIR,
    *,
    resources: Mapping[str, object],
    producer_graph: ProducerGraphIR,
    spec: ResolvedSpec,
) -> tuple[ResolvedParam, ...]:
    params: list[ResolvedParam] = [
        _resolved_param(
            f"/imputation/producer_graph/nodes/{node.id}", _wire(node.source)
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node.id}/depends_on",
            list(node.depends_on),
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node.id}/inputs",
            [_wire(input_row) for input_row in node.inputs],
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node.id}/outputs",
            [_wire(output) for output in node.outputs],
        ),
        _resolved_param(
            f"/compiled/producer_graph/nodes/{node.id}/write_scopes",
            [_wire(scope) for scope in node.write_scopes],
        ),
    ]
    if producer_graph.scope_registry is None:
        raise CompilerIRError(
            f"producer node {node.id!r} has no compiled scope registry"
        )
    params.append(
        _resolved_param(
            "/imputation/producer_graph/scope_registry",
            _wire(producer_graph.scope_registry),
        )
    )
    imputation = _mapping(resources.get("imputation", {}), location="imputation")
    family_matches: list[tuple[int, Mapping[str, object]]] = []
    for index, family_value in enumerate(
        _array(imputation.get("families", []), location="imputation/families")
    ):
        family = _mapping(family_value, location=f"imputation/families/{index}")
        producer = (
            family.get("execution_contract")
            if family.get("stage") == "primary_puf_qrf"
            else family.get("runtime_name")
            if family.get("stage") == "late_producer_dag"
            else None
        )
        if producer == node.id:
            family_matches.append((index, family))
    if len(family_matches) > 1:
        raise CompilerIRError(
            f"producer node {node.id!r} resolves more than one imputation family"
        )
    if family_matches:
        family_index, family = family_matches[0]
        params.append(_resolved_param(f"/imputation/families/{family_index}", family))
        models = _mapping(imputation.get("models", {}), location="imputation/models")
        model = family.get("model")
        if isinstance(model, str):
            params.append(_resolved_param(f"/imputation/models/{model}", models[model]))
        predictor_blocks = _mapping(
            imputation.get("predictor_blocks", {}),
            location="imputation/predictor_blocks",
        )
        for predictor in _array(
            family.get("predictors", []), location="imputation/family/predictors"
        ):
            predictor_id = str(predictor)
            params.append(
                _resolved_param(
                    f"/imputation/predictor_blocks/{predictor_id}",
                    predictor_blocks[predictor_id],
                )
            )
        profiles = _mapping(
            _mapping(
                imputation.get("transfer_execution", {}),
                location="imputation/transfer_execution",
            ).get("profiles", {}),
            location="imputation/transfer_execution/profiles",
        )
        contract_id = family.get("execution_contract")
        if isinstance(contract_id, str) and contract_id in profiles:
            params.append(
                _resolved_param(
                    f"/imputation/transfer_execution/profiles/{contract_id}",
                    profiles[contract_id],
                )
            )

    if producer_graph.authored is not None:
        graph = _mapping(_wire(producer_graph.authored), location="producer_graph")
        for field in (
            "execution_receipt_contract",
            "resource_semantics",
            "scope_coverage",
        ):
            params.append(
                _resolved_param(f"/imputation/producer_graph/{field}", graph[field])
            )
        matching_ownership = [
            row
            for row in producer_graph.ownership_matrix
            if any(
                action.get("producer") == node.id
                for action in (
                    _mapping(value, location="ownership action")
                    for value in _array(
                        row.get("producer_actions", []),
                        location="ownership actions",
                    )
                )
            )
        ]
        if matching_ownership:
            params.append(
                _resolved_param(
                    f"/imputation/producer_graph/ownership_matrix@producer={node.id}",
                    matching_ownership,
                )
            )

    output_keys = {
        f"{output['entity']}.{output['column']}"
        for output in (
            _mapping(_wire(value), location="compiled output") for value in node.outputs
        )
        if not str(output["column"]).startswith("@")
        or output["column"] == "@resolved_weight"
    }
    column_rows = [
        {
            "key": column.key,
            "entity": column.entity.id,
            "dtype": column.dtype,
            "unit": column.unit,
            "period": column.period,
            "vintage": column.vintage,
            "nullable": column.nullable,
            "domain": column.domain,
            "public_stability": column.public_stability,
            "unit_waiver": column.unit_waiver,
        }
        for column in spec.columns
        if column.key in output_keys
    ]
    if column_rows:
        params.append(
            _resolved_param(f"/compiled/columns@producer={node.id}", column_rows)
        )
    params.extend(_node_take_up_params(node, resources=resources))
    params.extend(
        _seed_reference_params(
            _typed_seed_owner_references(params),
            resources=resources,
        )
    )
    return tuple(params)


def _compile_nodes(
    spec: ResolvedSpec,
    *,
    resources: Mapping[str, object],
    compiler_ir_abi: CompilerIRABI,
    producer_graph: ProducerGraphIR,
    seed_stream_map: SeedStreamMap,
) -> tuple[CompiledNode, ...]:
    if not producer_graph.nodes:
        return ()
    node_by_id = {node.id: node for node in producer_graph.nodes}
    if producer_graph.scope_registry is None:
        raise CompilerIRError("compiled producer graph has no scope registry")
    compiler_ir_abi_wire = _frozen_mapping(
        compiler_ir_abi.to_wire(), location="compiler_ir_abi"
    )
    row_classifier_ref, row_classifier_implementation_sha256 = row_classifier_contract(
        compiler_ir_abi, producer_graph.scope_registry
    )
    local_sha256: dict[str, str] = {}
    ancestor_ids: dict[str, set[str]] = {}
    compiled: list[CompiledNode] = []
    for execution_rank, node_id in enumerate(producer_graph.order):
        node = node_by_id[node_id]
        kernel_implementation_sha256 = kernel_implementation_contract(node.kernel)
        params = _node_resolved_params(
            node,
            resources=resources,
            producer_graph=producer_graph,
            spec=spec,
        )
        params = (
            *params,
            _resolved_param(
                f"/compiled/producer_graph/nodes/{node.id}/execution_rank",
                execution_rank,
            ),
            _resolved_param(
                f"/compiled/producer_graph/nodes/{node.id}/kernel",
                {
                    "ref": node.kernel,
                    "implementation_sha256": kernel_implementation_sha256,
                },
            ),
            _resolved_param(
                f"/compiled/producer_graph/nodes/{node.id}/row_classifier",
                {
                    "ref": row_classifier_ref,
                    "implementation_sha256": (row_classifier_implementation_sha256),
                },
            ),
        )
        grant_param, sites, streams = _effective_seed_grant(
            node,
            params=params,
            seed_stream_map=seed_stream_map,
        )
        params = (*params, grant_param)
        local_wire = [param.to_wire() for param in params]
        local_sha256[node_id] = sha256_json(local_wire)
        ancestors = set(node.depends_on)
        for predecessor in node.depends_on:
            ancestors.update(ancestor_ids[predecessor])
        ancestor_ids[node_id] = ancestors
        transitive = tuple(
            TransitiveNodeSlice(
                id=candidate,
                local_slice_sha256=local_sha256[candidate],
            )
            for candidate in producer_graph.order
            if candidate in ancestors
        )
        slice_wire = {
            "domain": _NODE_SLICE_DOMAIN,
            "resolved_params": local_wire,
            "transitive_nodes": [row.to_wire() for row in transitive],
        }
        node_slice_sha256 = sha256_json(slice_wire)
        node_key = sha256_json(
            {
                "domain": _NODE_KEY_DOMAIN,
                "compiler_ir_abi": compiler_ir_abi.to_wire(),
                "node_slice_sha256": node_slice_sha256,
                "kernel": {
                    "ref": node.kernel,
                    "implementation_sha256": kernel_implementation_sha256,
                },
                "seed_protocol_sha256": seed_stream_map.implementation_sha256,
            }
        )
        compiled.append(
            CompiledNode(
                id=node_id,
                execution_rank=execution_rank,
                node_key=node_key,
                node_slice_sha256=node_slice_sha256,
                kernel_ref=node.kernel,
                kernel_implementation_sha256=kernel_implementation_sha256,
                depends_on=node.depends_on,
                inputs=node.inputs,
                outputs=node.outputs,
                capabilities=node.capabilities,
                mutations=node.mutations,
                write_scopes=node.write_scopes,
                scope_registry=producer_graph.scope_registry,
                row_classifier_ref=row_classifier_ref,
                row_classifier_implementation_sha256=(
                    row_classifier_implementation_sha256
                ),
                compiler_ir_abi=compiler_ir_abi_wire,
                seed_protocol_sha256=seed_stream_map.implementation_sha256,
                seed_sites=sites,
                seed_streams=streams,
                resolved_params=params,
                transitive_nodes=transitive,
            )
        )
    return tuple(compiled)


def compile_spec(spec: ResolvedSpec) -> CompiledSpecIR:
    """Compile one immutable ``ResolvedSpec`` without constructing authority.

    The input has already passed parse/schema/cross-reference resolution, but
    the compiler repeats all graph-local closure checks.  That defense matters
    for callers constructing dataclass fixtures and guarantees that emitted
    plan locks cannot contain dangling dependencies or ambiguous ownership.
    """

    if not isinstance(spec, ResolvedSpec):
        raise TypeError("compile_spec requires a ResolvedSpec")
    resources, frozen_resources = _normalized_resources(spec)
    compiler_ir_abi = _compiler_ir_abi()
    producer_graph = _compile_producer_graph(resources)
    stage_dag = StageDag(
        external_stages=producer_graph.external_stages,
        nodes=tuple(
            StageDagNode(
                id=node.id,
                kind=node.kind,
                kernel=node.kernel,
                depends_on=node.depends_on,
            )
            for node in producer_graph.nodes
        ),
        edges=producer_graph.edges,
        waves=producer_graph.waves,
        order=producer_graph.order,
    )
    seed_stream_map = _compile_seed_stream_map(
        spec,
        resources=resources,
        producer_graph=producer_graph,
    )
    nodes = _compile_nodes(
        spec,
        resources=resources,
        compiler_ir_abi=compiler_ir_abi,
        producer_graph=producer_graph,
        seed_stream_map=seed_stream_map,
    )
    return CompiledSpecIR(
        spec_binding=spec.spec_binding,
        compiler_ir_abi=compiler_ir_abi,
        normalized_resources=frozen_resources,
        surfaces=_resolved_surfaces(spec),
        typed_inventory=_typed_inventory(spec),
        generated_authorities=spec.generated_authorities,
        vintage_authorities=spec.vintage_authorities,
        stage_dag=stage_dag,
        producer_graph=producer_graph,
        seed_stream_map=seed_stream_map,
        nodes=nodes,
    )


__all__ = [
    "COMPILER_IR_ABI_VERSION",
    "EXECUTOR_CONTRACT_ABI",
    "ROW_CLASSIFIER_IMPLEMENTATION_DOMAIN",
    "CompiledNode",
    "CompiledSpecIR",
    "CompilerIRABI",
    "CompilerIRError",
    "ProducerGraphIR",
    "ProducerNodeIR",
    "SeedOwnerIR",
    "SeedSiteIR",
    "SeedStreamMap",
    "StageDag",
    "StageDagNode",
    "compile_spec",
    "row_classifier_contract",
]
