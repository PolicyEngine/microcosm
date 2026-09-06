"""Resolve the bundle's compiler-facing columns, artifacts, and row scopes.

The JSON schemas close each individual row.  This module closes the relations
between rows: every physical producer output has one column contract, every
virtual output and resource binding has one typed artifact, and every row-scope
reference names a finite predicate whose intersections the compiler can decide.
It deliberately has no dependency on the executor or constants-era runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from .model import (
    ArtifactSpec,
    ColumnSpec,
    EntitySpec,
    FrozenMap,
    ScopeSpec,
    freeze_json,
)


class TypedClosureError(ValueError):
    """A typed column, artifact, or scope relation is not closed."""


@dataclass(frozen=True, slots=True)
class TypedClosureResult:
    artifacts: tuple[ArtifactSpec, ...]
    scopes: tuple[ScopeSpec, ...]
    columns: tuple[ColumnSpec, ...]


_PRODUCER_SCOPE_SPACE = "producer_origin_clone_or_receipt"
_PRODUCER_SCOPE_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "acs_source": ("origin:acs/clone:0",),
    "asec_source": ("origin:asec/clone:0",),
    "puf_clone": (
        "origin:acs/clone:1",
        "origin:acs/clone:2",
        "origin:asec/clone:1",
        "origin:asec/clone:2",
    ),
    "receipt": ("receipt:virtual",),
    "whole_pool": (
        "origin:acs/clone:0",
        "origin:acs/clone:1",
        "origin:acs/clone:2",
        "origin:asec/clone:0",
        "origin:asec/clone:1",
        "origin:asec/clone:2",
    ),
}
_PRODUCER_SCOPE_UNIVERSE = tuple(
    sorted({atom for atoms in _PRODUCER_SCOPE_DEFINITIONS.values() for atom in atoms})
)

_TAKE_UP_SCOPE_SPACE = "take_up_support_channel"
_TAKE_UP_SCOPE_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "asec_rows": ("support_channel:asec",),
    "puf_support_rows": ("support_channel:puf_tax_detail",),
}
_TAKE_UP_SCOPE_UNIVERSE = tuple(
    sorted(atom for atoms in _TAKE_UP_SCOPE_DEFINITIONS.values() for atom in atoms)
)


def _scope_registry_wire(
    *,
    predicate_space: str,
    universe: tuple[str, ...],
    definitions: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    return {
        "predicate_space": predicate_space,
        "universe": list(universe),
        "scopes": [
            {"id": scope_id, "atoms": list(atoms)}
            for scope_id, atoms in sorted(definitions.items())
        ],
    }


def producer_scope_registry_wire() -> dict[str, object]:
    """Return the immutable compiler-owned producer-scope registry."""

    return _scope_registry_wire(
        predicate_space=_PRODUCER_SCOPE_SPACE,
        universe=_PRODUCER_SCOPE_UNIVERSE,
        definitions=_PRODUCER_SCOPE_DEFINITIONS,
    )


def take_up_scope_registry_wire() -> dict[str, object]:
    """Return the immutable compiler-owned take-up-scope registry."""

    return _scope_registry_wire(
        predicate_space=_TAKE_UP_SCOPE_SPACE,
        universe=_TAKE_UP_SCOPE_UNIVERSE,
        definitions=_TAKE_UP_SCOPE_DEFINITIONS,
    )


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypedClosureError(f"{location}: expected mapping")
    return value


def _array(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise TypedClosureError(f"{location}: expected array")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypedClosureError(f"{location}: expected non-empty string")
    return value


def _column_key(
    entity: object,
    column: object,
    *,
    location: str,
    entity_by_id: Mapping[str, EntitySpec],
) -> str:
    entity_id = _string(entity, f"{location}/entity")
    if entity_id not in entity_by_id:
        raise TypedClosureError(f"{location}/entity: unknown entity {entity_id!r}")
    column_id = _string(column, f"{location}/column")
    return f"{entity_id}.{column_id}"


def _split_column_key(
    key: object,
    *,
    location: str,
    entity_by_id: Mapping[str, EntitySpec],
) -> tuple[str, str]:
    value = _string(key, location)
    entity, separator, column = value.partition(".")
    if not separator or not column or "." in column:
        raise TypedClosureError(f"{location}: expected entity.column key")
    if entity not in entity_by_id:
        raise TypedClosureError(f"{location}: unknown entity prefix {entity!r}")
    return entity, column


def _graph_rows(
    resources: Mapping[str, object],
) -> tuple[Mapping[str, object], Sequence[object]]:
    imputation = _mapping(resources.get("imputation", {}), "imputation")
    graph = _mapping(imputation.get("producer_graph", {}), "imputation/producer_graph")
    return graph, _array(graph.get("nodes", []), "imputation/producer_graph/nodes")


def compile_producer_outputs(
    resources: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    """Join family-owned modeled outputs to authored structural node outputs.

    Primary and late family targets are the sole authority for modeled output
    membership and row coverage.  Producer nodes retain only structural and
    virtual outputs.  This compiler-owned join rejects ambiguous links and
    collisions, then restores the constants-era ``(entity, column)`` order for
    every family-linked node.
    """

    imputation = _mapping(resources.get("imputation", {}), "imputation")
    graph = _mapping(
        imputation.get("producer_graph", {}), "imputation/producer_graph"
    )
    nodes = _array(graph.get("nodes", []), "imputation/producer_graph/nodes")
    nodes_by_id: dict[str, Mapping[str, object]] = {}
    for node_index, node_value in enumerate(nodes):
        location = f"imputation/producer_graph/nodes/{node_index}"
        node = _mapping(node_value, location)
        node_id = _string(node.get("id"), f"{location}/id")
        if node_id in nodes_by_id:
            raise TypedClosureError(f"{location}/id: duplicate producer node {node_id!r}")
        nodes_by_id[node_id] = node

    family_rows: dict[str, list[dict[str, object]]] = {}
    family_kinds: dict[str, str] = {}
    family_locations: dict[str, str] = {}
    for family_index, family_value in enumerate(
        _array(imputation.get("families", []), "imputation/families")
    ):
        family_location = f"imputation/families/{family_index}"
        family = _mapping(family_value, family_location)
        stage = _string(family.get("stage"), f"{family_location}/stage")
        if stage == "primary_puf_qrf":
            producer_field = "execution_contract"
            expected_kind = "primary_puf"
        elif stage == "late_producer_dag":
            producer_field = "runtime_name"
            expected_kind = "late_transfer"
        else:
            for target_index, target_value in enumerate(
                _array(family.get("targets", []), f"{family_location}/targets")
            ):
                target = _mapping(
                    target_value, f"{family_location}/targets/{target_index}"
                )
                if "output_coverage_scope" in target:
                    raise TypedClosureError(
                        f"{family_location}/targets/{target_index}/"
                        "output_coverage_scope: only primary and late family "
                        "targets own producer outputs"
                    )
            continue
        producer = _string(
            family.get(producer_field), f"{family_location}/{producer_field}"
        )
        if producer in family_rows:
            raise TypedClosureError(
                f"{family_location}/{producer_field}: producer node {producer!r} "
                "is linked from more than one family; "
                f"first={family_locations[producer]}"
            )
        outputs: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for target_index, target_value in enumerate(
            _array(family.get("targets", []), f"{family_location}/targets")
        ):
            target_location = f"{family_location}/targets/{target_index}"
            target = _mapping(target_value, target_location)
            entity = _string(target.get("entity"), f"{target_location}/entity")
            column = _string(target.get("name"), f"{target_location}/name")
            coverage_scope = _string(
                target.get("output_coverage_scope"),
                f"{target_location}/output_coverage_scope",
            )
            key = (entity, column)
            if key in seen:
                raise TypedClosureError(
                    f"{target_location}: duplicate expanded producer output {key!r}"
                )
            seen.add(key)
            outputs.append(
                {
                    "entity": entity,
                    "column": column,
                    "coverage_scope": coverage_scope,
                    # Family target names use defs.identifier and therefore
                    # cannot represent temporary/validation-only virtual rows.
                    "temporary": False,
                    "validation_only": False,
                }
            )
        family_rows[producer] = outputs
        family_kinds[producer] = expected_kind
        family_locations[producer] = family_location

    for producer, expected_kind in family_kinds.items():
        node = nodes_by_id.get(producer)
        if node is None:
            raise TypedClosureError(
                f"{family_locations[producer]}: dangling producer node {producer!r}"
            )
        if node.get("kind") != expected_kind:
            raise TypedClosureError(
                f"{family_locations[producer]}: producer node {producer!r} must "
                f"have kind {expected_kind!r}"
            )
    orphan_nodes = sorted(
        node_id
        for node_id, node in nodes_by_id.items()
        if node.get("kind") in {"primary_puf", "late_transfer"}
        and node_id not in family_rows
    )
    if orphan_nodes:
        raise TypedClosureError(
            "imputation/producer_graph/nodes: modeled producers without families "
            f"{orphan_nodes!r}"
        )

    compiled: dict[str, tuple[Mapping[str, object], ...]] = {}
    for node_id, node in nodes_by_id.items():
        authored: list[Mapping[str, object]] = []
        authored_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
        for output_index, output_value in enumerate(
            _array(
                node.get("outputs", []),
                f"imputation/producer_graph/nodes/{node_id}/outputs",
            )
        ):
            location = f"imputation/producer_graph/nodes/{node_id}/outputs/{output_index}"
            output = _mapping(output_value, location)
            key = (
                _string(output.get("entity"), f"{location}/entity"),
                _string(output.get("column"), f"{location}/column"),
            )
            previous = authored_by_key.get(key)
            if previous is not None:
                relation = "duplicate" if previous == output else "conflicting"
                raise TypedClosureError(
                    f"{location}: {relation} authored producer output {key!r}"
                )
            authored_by_key[key] = output
            authored.append(output)
        expanded = family_rows.get(node_id, [])
        for output in expanded:
            key = (str(output["entity"]), str(output["column"]))
            previous = authored_by_key.get(key)
            if previous is not None:
                relation = "duplicates" if previous == output else "conflicts with"
                raise TypedClosureError(
                    f"imputation/producer_graph/nodes/{node_id}/outputs: authored "
                    f"output {key!r} {relation} its family-owned output"
                )
        combined = [*authored, *expanded]
        if expanded:
            combined.sort(key=lambda row: (str(row["entity"]), str(row["column"])))
        compiled[node_id] = tuple(combined)
    return compiled


def _family_target_keys(
    resources: Mapping[str, object],
    *,
    entity_by_id: Mapping[str, EntitySpec],
) -> frozenset[str]:
    imputation = _mapping(resources.get("imputation", {}), "imputation")
    result: set[str] = set()
    for family_index, family_value in enumerate(
        _array(imputation.get("families", []), "imputation/families")
    ):
        family_location = f"imputation/families/{family_index}"
        family = _mapping(family_value, family_location)
        for target_index, target_value in enumerate(
            _array(family.get("targets", []), f"{family_location}/targets")
        ):
            target_location = f"{family_location}/targets/{target_index}"
            target = _mapping(target_value, target_location)
            key = _column_key(
                target.get("entity"),
                target.get("name"),
                location=target_location,
                entity_by_id=entity_by_id,
            )
            result.add(key)
    return frozenset(result)


def _producer_outputs(
    nodes: Sequence[object],
    *,
    compiled_outputs: Mapping[str, Sequence[Mapping[str, object]]],
    entity_by_id: Mapping[str, EntitySpec],
) -> tuple[
    frozenset[str],
    dict[str, list[tuple[str, Mapping[str, object]]]],
    bool,
]:
    physical: set[str] = set()
    virtual: dict[str, list[tuple[str, Mapping[str, object]]]] = {}
    strict_graph = False
    for node_index, node_value in enumerate(nodes):
        node_location = f"imputation/producer_graph/nodes/{node_index}"
        node = _mapping(node_value, node_location)
        node_id = _string(node.get("id"), f"{node_location}/id")
        if "outputs" not in node:
            continue
        strict_graph = True
        for output_index, output_value in enumerate(
            compiled_outputs.get(node_id, ()),
        ):
            output_location = f"{node_location}/outputs/{output_index}"
            output = _mapping(output_value, output_location)
            key = _column_key(
                output.get("entity"),
                output.get("column"),
                location=output_location,
                entity_by_id=entity_by_id,
            )
            column = key.split(".", 1)[1]
            if column.startswith("@") and column != "@resolved_weight":
                virtual.setdefault(key, []).append((node_id, output))
            else:
                physical.add(key)
    if physical.intersection(virtual):
        raise TypedClosureError(
            "imputation/producer_graph/outputs: output classified as both "
            "physical column and virtual artifact"
        )
    return frozenset(physical), virtual, strict_graph


def _metadata_waiver_ids(catalogs: Mapping[str, object]) -> frozenset[str]:
    result: set[str] = set()
    for index, value in enumerate(
        _array(catalogs.get("metadata_waivers", []), "catalogs/metadata_waivers")
    ):
        location = f"catalogs/metadata_waivers/{index}"
        waiver = _mapping(value, location)
        waiver_id = _string(waiver.get("id"), f"{location}/id")
        if waiver_id in result:
            raise TypedClosureError(
                f"{location}/id: duplicate metadata waiver {waiver_id!r}"
            )
        expires_on = _string(waiver.get("expires_on"), f"{location}/expires_on")
        try:
            expiry = date.fromisoformat(expires_on)
        except ValueError as error:
            raise TypedClosureError(
                f"{location}/expires_on: expected an ISO calendar date"
            ) from error
        if expiry < date.today():
            raise TypedClosureError(
                f"{location}/expires_on: metadata waiver expired on {expires_on}"
            )
        result.add(waiver_id)
    return frozenset(result)


def _resolve_columns(
    resources: Mapping[str, object],
    *,
    entity_by_id: Mapping[str, EntitySpec],
    physical_output_keys: frozenset[str],
) -> tuple[ColumnSpec, ...]:
    catalogs = _mapping(resources.get("catalogs", {}), "catalogs")
    rows = _array(catalogs.get("columns", []), "catalogs/columns")
    # Empty catalogs are the compatibility form used by minimal country
    # bundles.  Once a bundle declares any contracts the catalog closes over
    # every physical output, family target, and take-up variable.
    if not rows:
        return ()

    waiver_ids = _metadata_waiver_ids(catalogs)
    used_waivers: set[str] = set()
    keys: set[str] = set()
    columns: list[ColumnSpec] = []
    for index, value in enumerate(rows):
        location = f"catalogs/columns/{index}"
        row = _mapping(value, location)
        key = _string(row.get("key"), f"{location}/key")
        if key in keys:
            raise TypedClosureError(f"{location}/key: duplicate key {key!r}")
        entity_id, _ = _split_column_key(
            key,
            location=f"{location}/key",
            entity_by_id=entity_by_id,
        )
        contract = _mapping(row.get("contract", {}), f"{location}/contract")
        contract_entity = _string(contract.get("entity"), f"{location}/contract/entity")
        if contract_entity != entity_id:
            raise TypedClosureError(
                f"{location}/key: entity prefix {entity_id!r} does not match "
                f"contract entity {contract_entity!r}"
            )
        unit_waiver_value = contract.get("unit_waiver")
        unit_waiver = None
        if unit_waiver_value is not None:
            unit_waiver = _string(unit_waiver_value, f"{location}/contract/unit_waiver")
            if unit_waiver not in waiver_ids:
                raise TypedClosureError(
                    f"{location}/contract/unit_waiver: dangling metadata waiver "
                    f"{unit_waiver!r}"
                )
            used_waivers.add(unit_waiver)
        definition_period = contract.get("definition_period", contract.get("period"))
        vintage_value = contract.get("vintage")
        columns.append(
            ColumnSpec(
                key=key,
                entity=entity_by_id[entity_id],
                dtype=str(contract.get("dtype")),
                unit=str(contract.get("unit")),
                period=str(definition_period),
                vintage=None if vintage_value is None else str(vintage_value),
                nullable=bool(contract.get("nullable")),
                domain=str(contract.get("domain")),
                public_stability=str(contract.get("public_stability")),
                unit_waiver=unit_waiver,
            )
        )
        keys.add(key)

    orphan_waivers = sorted(waiver_ids - used_waivers)
    if orphan_waivers:
        raise TypedClosureError(
            f"catalogs/metadata_waivers: unused waiver ids {orphan_waivers!r}"
        )

    target_keys = _family_target_keys(resources, entity_by_id=entity_by_id)
    take_up = _mapping(resources.get("take_up", {}), "take_up")
    take_up_keys: set[str] = set()
    for index, value in enumerate(
        _array(take_up.get("programs", []), "take_up/programs")
    ):
        location = f"take_up/programs/{index}"
        program = _mapping(value, location)
        variable = _string(program.get("variable"), f"{location}/variable")
        matches = sorted(key for key in keys if key.rsplit(".", 1)[-1] == variable)
        if len(matches) != 1:
            raise TypedClosureError(
                f"catalogs/columns: take-up variable {variable!r} must have "
                f"exactly one contract; found {matches!r}"
            )
        take_up_keys.add(matches[0])

    expected = physical_output_keys | target_keys | frozenset(take_up_keys)
    unknown = sorted(keys - expected)
    if expected and unknown:
        raise TypedClosureError(
            f"catalogs/columns: keys absent from compiled typed outputs {unknown!r}"
        )
    missing_physical = sorted(physical_output_keys - keys)
    if missing_physical:
        raise TypedClosureError(
            "catalogs/columns: missing compiled physical output contracts "
            f"{missing_physical!r}"
        )
    missing_targets = sorted(target_keys - keys)
    if missing_targets:
        raise TypedClosureError(
            f"catalogs/columns: missing imputation target contracts {missing_targets!r}"
        )
    return tuple(sorted(columns, key=lambda column: column.key))


def _artifact_binding(value: object) -> FrozenMap:
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenMap):
        raise AssertionError("artifact bindings are mappings")
    return frozen


def _node_artifacts(nodes: Sequence[object]) -> list[ArtifactSpec]:
    result: list[ArtifactSpec] = []
    seen: set[str] = set()
    for index, value in enumerate(nodes):
        location = f"imputation/producer_graph/nodes/{index}"
        node = _mapping(value, location)
        node_id = _string(node.get("id"), f"{location}/id")
        if node_id in seen:
            raise TypedClosureError(
                f"{location}/id: duplicate producer node {node_id!r}"
            )
        seen.add(node_id)
        metadata = {
            key: node[key] for key in ("kind", "kernel", "capabilities") if key in node
        }
        result.append(
            ArtifactSpec(
                id=node_id,
                kind="producer_node",
                producing_stages=(node_id,),
                entity=None,
                key=None,
                lifetime="plan",
                validation="compiler_resolved_node",
                binding=_artifact_binding(metadata),
            )
        )
    return result


def _virtual_output_artifacts(
    rows_by_key: Mapping[str, Sequence[tuple[str, Mapping[str, object]]]],
    *,
    entity_by_id: Mapping[str, EntitySpec],
) -> list[ArtifactSpec]:
    result: list[ArtifactSpec] = []
    for key in sorted(rows_by_key):
        entity_id, _ = _split_column_key(
            key,
            location=f"imputation/producer_graph/outputs/{key}",
            entity_by_id=entity_by_id,
        )
        rows = rows_by_key[key]
        temporary = {row.get("temporary") for _, row in rows}
        validation_only = {row.get("validation_only") for _, row in rows}
        if not temporary <= {True, False} or len(temporary) != 1:
            raise TypedClosureError(
                f"imputation/producer_graph/outputs/{key}: inconsistent temporary flag"
            )
        if not validation_only <= {True, False} or len(validation_only) != 1:
            raise TypedClosureError(
                f"imputation/producer_graph/outputs/{key}: inconsistent "
                "validation_only flag"
            )
        bindings = [
            {"producer": producer, **dict(row)}
            for producer, row in sorted(rows, key=lambda item: item[0])
        ]
        result.append(
            ArtifactSpec(
                id=key,
                kind="virtual_output",
                producing_stages=tuple(sorted({producer for producer, _ in rows})),
                entity=entity_by_id[entity_id],
                key=key,
                lifetime="temporary" if temporary == {True} else "persistent",
                validation=(
                    "validation_only"
                    if validation_only == {True}
                    else "materialized_output"
                ),
                binding=_artifact_binding({"rows": bindings}),
            )
        )
    return result


def _virtual_resource_artifacts(
    nodes: Sequence[object],
    *,
    entity_by_id: Mapping[str, EntitySpec],
    strict_graph: bool,
) -> list[ArtifactSpec]:
    # Compatibility resolver fixtures predate typed virtual-resource rows.  A
    # graph becomes strict as soon as it declares typed outputs/scope registry.
    if not strict_graph:
        return []
    grouped: dict[str, list[dict[str, object]]] = {}
    for node_index, node_value in enumerate(nodes):
        node_location = f"imputation/producer_graph/nodes/{node_index}"
        node = _mapping(node_value, node_location)
        producer = _string(node.get("id"), f"{node_location}/id")
        for resource_index, resource_value in enumerate(
            _array(
                node.get("virtual_resources", []), f"{node_location}/virtual_resources"
            )
        ):
            location = f"{node_location}/virtual_resources/{resource_index}"
            resource = _mapping(resource_value, location)
            resource_id = _string(resource.get("id"), f"{location}/id")
            entity_id, column = _split_column_key(
                resource_id,
                location=f"{location}/id",
                entity_by_id=entity_by_id,
            )
            if not column.startswith("@"):
                raise TypedClosureError(
                    f"{location}/id: virtual resource key must begin with '@'"
                )
            kind = _string(resource.get("kind"), f"{location}/kind")
            resolution = _string(resource.get("resolution"), f"{location}/resolution")
            binding = _mapping(resource.get("binding"), f"{location}/binding")
            _string(binding.get("resource_kind"), f"{location}/binding/resource_kind")
            row: dict[str, object] = {
                "producer": producer,
                "entity": entity_id,
                "kind": kind,
                "resolution": resolution,
                "binding": dict(binding),
            }
            if "dynamic_field" in resource:
                row["dynamic_field"] = resource["dynamic_field"]
            grouped.setdefault(resource_id, []).append(row)

    result: list[ArtifactSpec] = []
    for resource_id in sorted(grouped):
        rows = sorted(grouped[resource_id], key=lambda row: str(row["producer"]))
        producers = [str(row["producer"]) for row in rows]
        if len(producers) != len(set(producers)):
            raise TypedClosureError(
                f"imputation/producer_graph/virtual_resources/{resource_id}: "
                "producer declares the same binding id more than once"
            )
        kinds = {str(row["kind"]) for row in rows}
        resolutions = {str(row["resolution"]) for row in rows}
        entities = {str(row["entity"]) for row in rows}
        if len(kinds) != 1 or len(resolutions) != 1 or len(entities) != 1:
            raise TypedClosureError(
                f"imputation/producer_graph/virtual_resources/{resource_id}: "
                "shared binding id has inconsistent kind, resolution, or entity"
            )
        resolution = next(iter(resolutions))
        entity_id = next(iter(entities))
        result.append(
            ArtifactSpec(
                id=resource_id,
                kind="virtual_resource_binding",
                producing_stages=tuple(producers),
                entity=entity_by_id[entity_id],
                key=resource_id,
                lifetime="plan" if resolution == "static_exact" else "run",
                validation=resolution,
                binding=_artifact_binding({"rows": rows}),
            )
        )
    return result


def _scope_references(value: object, *, path: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key in {
                "coverage_scope",
                "output_coverage_scope",
                "required_scope",
                "row_scope",
            }:
                if child is None:
                    continue
                if not isinstance(child, str) or not child:
                    raise TypedClosureError(
                        f"{child_path}: named closed row-scope reference required"
                    )
                result.append((child_path, child))
            else:
                result.extend(_scope_references(child, path=child_path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            result.extend(_scope_references(child, path=f"{path}/{index}"))
    return result


def _registry_rows(
    registry_value: object,
    *,
    location: str,
    expected_space: str,
    expected_universe: tuple[str, ...],
    expected_definitions: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    registry = _mapping(registry_value, location)
    predicate_space = _string(
        registry.get("predicate_space"), f"{location}/predicate_space"
    )
    if predicate_space != expected_space:
        raise TypedClosureError(
            f"{location}/predicate_space: expected {expected_space!r}, "
            f"got {predicate_space!r}"
        )
    universe = tuple(
        sorted(
            _string(value, f"{location}/universe/{index}")
            for index, value in enumerate(
                _array(registry.get("universe"), f"{location}/universe")
            )
        )
    )
    if len(universe) != len(set(universe)):
        raise TypedClosureError(f"{location}/universe: duplicate finite atom")
    if universe != expected_universe:
        raise TypedClosureError(
            f"{location}/universe: expected exact finite atoms "
            f"{list(expected_universe)!r}, got {list(universe)!r}"
        )
    rows: dict[str, tuple[str, ...]] = {}
    for index, value in enumerate(_array(registry.get("scopes"), f"{location}/scopes")):
        row_location = f"{location}/scopes/{index}"
        row = _mapping(value, row_location)
        scope_id = _string(row.get("id"), f"{row_location}/id")
        if scope_id in rows:
            raise TypedClosureError(
                f"{row_location}/id: duplicate scope id {scope_id!r}"
            )
        atoms = tuple(
            sorted(
                _string(atom, f"{row_location}/atoms/{atom_index}")
                for atom_index, atom in enumerate(
                    _array(row.get("atoms"), f"{row_location}/atoms")
                )
            )
        )
        if len(atoms) != len(set(atoms)):
            raise TypedClosureError(f"{row_location}/atoms: duplicate finite atom")
        dangling_atoms = sorted(set(atoms) - set(universe))
        if dangling_atoms:
            raise TypedClosureError(
                f"{row_location}/atoms: atoms outside registry universe "
                f"{dangling_atoms!r}"
            )
        rows[scope_id] = atoms
    if rows != dict(expected_definitions):
        raise TypedClosureError(
            f"{location}/scopes: expected exact closed definitions "
            f"{dict(expected_definitions)!r}, got {rows!r}"
        )
    return rows


def _resolved_scopes(
    resources: Mapping[str, object],
    *,
    graph: Mapping[str, object],
    include_family_output_scopes: bool,
) -> tuple[ScopeSpec, ...]:
    result: list[ScopeSpec] = []
    graph_without_registry = {
        key: value for key, value in graph.items() if key != "scope_registry"
    }
    producer_refs = _scope_references(
        graph_without_registry, path="imputation/producer_graph"
    )
    if include_family_output_scopes:
        imputation = _mapping(resources.get("imputation", {}), "imputation")
        producer_refs.extend(
            _scope_references(
                imputation.get("families", []), path="imputation/families"
            )
        )
    if producer_refs or "scope_registry" in graph:
        if "scope_registry" not in graph:
            raise TypedClosureError(
                "imputation/producer_graph/scope_registry: required by row-scope "
                "references"
            )
        definitions = _registry_rows(
            graph["scope_registry"],
            location="imputation/producer_graph/scope_registry",
            expected_space=_PRODUCER_SCOPE_SPACE,
            expected_universe=_PRODUCER_SCOPE_UNIVERSE,
            expected_definitions=_PRODUCER_SCOPE_DEFINITIONS,
        )
        used: set[str] = set()
        for location, scope_id in producer_refs:
            if scope_id not in definitions:
                raise TypedClosureError(
                    f"{location}: dangling producer row scope {scope_id!r}"
                )
            used.add(scope_id)
        orphans = sorted(set(definitions) - used)
        if orphans:
            raise TypedClosureError(
                f"imputation/producer_graph/scope_registry: orphan scopes {orphans!r}"
            )
        result.extend(
            ScopeSpec(
                id=scope_id,
                predicate_space=_PRODUCER_SCOPE_SPACE,
                entity=None,
                predicate=freeze_json({"atoms": list(atoms)}),
                source_path=(
                    f"imputation/producer_graph/scope_registry/scopes/{scope_id}"
                ),
            )
            for scope_id, atoms in sorted(definitions.items())
        )

    take_up = _mapping(resources.get("take_up", {}), "take_up")
    take_up_without_registry = {
        key: value for key, value in take_up.items() if key != "scope_registry"
    }
    take_up_refs = _scope_references(take_up_without_registry, path="take_up")
    if take_up_refs or "scope_registry" in take_up:
        if "scope_registry" not in take_up:
            raise TypedClosureError(
                "take_up/scope_registry: required by mixed-ownership row scopes"
            )
        definitions = _registry_rows(
            take_up["scope_registry"],
            location="take_up/scope_registry",
            expected_space=_TAKE_UP_SCOPE_SPACE,
            expected_universe=_TAKE_UP_SCOPE_UNIVERSE,
            expected_definitions=_TAKE_UP_SCOPE_DEFINITIONS,
        )
        used: set[str] = set()
        for location, scope_id in take_up_refs:
            if scope_id not in definitions:
                raise TypedClosureError(
                    f"{location}: dangling take-up row scope {scope_id!r}"
                )
            used.add(scope_id)

        for program_index, program_value in enumerate(
            _array(take_up.get("programs", []), "take_up/programs")
        ):
            program = _mapping(program_value, f"take_up/programs/{program_index}")
            if program.get("ownership") != "mixed":
                continue
            segment_atoms: list[set[str]] = []
            for segment_index, segment_value in enumerate(
                _array(
                    program.get("segments", []),
                    f"take_up/programs/{program_index}/segments",
                )
            ):
                segment = _mapping(
                    segment_value,
                    f"take_up/programs/{program_index}/segments/{segment_index}",
                )
                scope_id = _string(
                    segment.get("row_scope"),
                    f"take_up/programs/{program_index}/segments/{segment_index}/row_scope",
                )
                atoms = set(definitions[scope_id])
                if any(atoms.intersection(previous) for previous in segment_atoms):
                    raise TypedClosureError(
                        f"take_up/programs/{program_index}/segments: mixed take-up "
                        "scope predicates overlap"
                    )
                segment_atoms.append(atoms)
            covered = set().union(*segment_atoms) if segment_atoms else set()
            if covered != set(_TAKE_UP_SCOPE_UNIVERSE):
                raise TypedClosureError(
                    f"take_up/programs/{program_index}/segments: mixed take-up "
                    "scope predicates are not exhaustive over the two support channels"
                )
        orphans = sorted(set(definitions) - used)
        if orphans:
            raise TypedClosureError(
                f"take_up/scope_registry: orphan scopes {orphans!r}"
            )
        result.extend(
            ScopeSpec(
                id=scope_id,
                predicate_space=_TAKE_UP_SCOPE_SPACE,
                entity=None,
                predicate=freeze_json({"atoms": list(atoms)}),
                source_path=f"take_up/scope_registry/scopes/{scope_id}",
            )
            for scope_id, atoms in sorted(definitions.items())
        )
    return tuple(result)


def resolve_typed_closure(
    resources: Mapping[str, object],
    *,
    entities: Sequence[EntitySpec],
) -> TypedClosureResult:
    """Return the fully cross-referenced typed compiler inventories."""

    entity_by_id = {entity.id: entity for entity in entities}
    graph, nodes = _graph_rows(resources)
    compiled_outputs = compile_producer_outputs(resources)
    physical_outputs, virtual_outputs, output_strict = _producer_outputs(
        nodes,
        compiled_outputs=compiled_outputs,
        entity_by_id=entity_by_id,
    )
    strict_graph = output_strict or "scope_registry" in graph
    columns = _resolve_columns(
        resources,
        entity_by_id=entity_by_id,
        physical_output_keys=physical_outputs,
    )
    artifacts = _node_artifacts(nodes)
    artifacts.extend(
        _virtual_output_artifacts(virtual_outputs, entity_by_id=entity_by_id)
    )
    artifacts.extend(
        _virtual_resource_artifacts(
            nodes,
            entity_by_id=entity_by_id,
            strict_graph=strict_graph,
        )
    )
    scopes = _resolved_scopes(
        resources,
        graph=graph,
        include_family_output_scopes=output_strict,
    )
    return TypedClosureResult(
        artifacts=tuple(artifacts),
        scopes=scopes,
        columns=columns,
    )


__all__ = [
    "compile_producer_outputs",
    "TypedClosureError",
    "TypedClosureResult",
    "producer_scope_registry_wire",
    "resolve_typed_closure",
    "take_up_scope_registry_wire",
]
