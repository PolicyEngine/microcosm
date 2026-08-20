"""Lossless boundary between US :class:`Frame` values and the spec executor.

The generic executor deliberately refuses columns outside a compiled node's
physical contract.  A production ``Frame`` is wider than any one node, so this
module keeps the two concerns separate:

* :func:`frame_to_projection` exposes only compiler-declared columns plus the
  schema's stable keys and membership columns;
* :func:`legacy_result_to_patch` proves that every column left outside that
  projection was preserved (or copied exactly from a native row by an expand);
* :func:`merge_projection_into_frame` checks a validated projection against
  the legacy result before restoring the untouched, wider frame surface.

Row authorization is derived from the compiler's finite atom universe and the
entity-prefixed support channel/clone columns.  No source or program name is
embedded here.  The same derivation is used by the trusted EXPAND row
classifier, including fail-closed source-row recovery for remapped clone ids.
"""

from __future__ import annotations

import copy
import pickle
import re
from collections.abc import Hashable, Mapping, Sequence
from numbers import Integral
from typing import Any

import pandas as pd

from microcosm.build.spec_engine.compiler_ir import CompiledNode
from microcosm.build.spec_engine.executor import (
    ImmutableFrameProjection,
    KernelPatch,
    RowClassification,
    StructuralDelta,
    WeightState,
)
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.spec_engine.scope_algebra import (
    ClosedScopeRegistry,
    ScopeAlgebraError,
)
from microcosm.build.us_runtime.support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

__all__ = [
    "FrameProjectionCodecError",
    "classify_added_support_rows",
    "frame_to_projection",
    "legacy_result_to_patch",
    "merge_projection_into_frame",
    "projection_to_frame",
]


class FrameProjectionCodecError(ValueError):
    """A frame cannot cross the compiled executor boundary without loss."""


VirtualKey = tuple[str, str]

_ORIGIN_CLONE_ATOM = re.compile(r"^origin:(?P<origin>[^/]+)/clone:(?P<clone>-?\d+)$")


def _wire_mapping(value: object, *, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FrameProjectionCodecError(f"{location} must be an object")
    return dict(value)


def _node_mapping(value: object, *, location: str) -> dict[str, object]:
    thawed = thaw_json(value)  # type: ignore[arg-type]
    return _wire_mapping(thawed, location=location)


def _node_scope_registry(node: CompiledNode) -> ClosedScopeRegistry:
    try:
        return ClosedScopeRegistry.from_wire(thaw_json(node.scope_registry))
    except (ScopeAlgebraError, TypeError, ValueError) as error:
        raise FrameProjectionCodecError(
            f"compiled node {node.id!r} has an invalid scope registry: {error}"
        ) from error


def _node_structural_delta(node: CompiledNode) -> StructuralDelta:
    capabilities = _node_mapping(node.capabilities, location="capabilities")
    try:
        return StructuralDelta(capabilities["structural_delta"])
    except (KeyError, ValueError) as error:
        raise FrameProjectionCodecError(
            f"compiled node {node.id!r} has no valid structural_delta"
        ) from error


def _input_rows(node: CompiledNode) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for input_index, value in enumerate(node.inputs):
        row = _node_mapping(value, location=f"inputs/{input_index}")
        rows.append(row)
        alternatives = row.get("alternatives", [])
        if not isinstance(alternatives, list):
            raise FrameProjectionCodecError(
                f"inputs/{input_index}/alternatives must be an array"
            )
        for group_index, group in enumerate(alternatives):
            if not isinstance(group, list):
                raise FrameProjectionCodecError(
                    f"inputs/{input_index}/alternatives/{group_index} "
                    "must be an array"
                )
            for cell_index, cell in enumerate(group):
                rows.append(
                    _wire_mapping(
                        cell,
                        location=(
                            f"inputs/{input_index}/alternatives/{group_index}/"
                            f"{cell_index}"
                        ),
                    )
                )
    return tuple(rows)


def _output_rows(node: CompiledNode) -> tuple[dict[str, object], ...]:
    return tuple(
        _node_mapping(value, location=f"outputs/{index}")
        for index, value in enumerate(node.outputs)
    )


def _write_scope_rows(node: CompiledNode) -> tuple[dict[str, object], ...]:
    return tuple(
        _node_mapping(value, location=f"write_scopes/{index}")
        for index, value in enumerate(node.write_scopes)
    )


def _entity_column(
    row: Mapping[str, object],
    *,
    location: str,
) -> tuple[str, str]:
    entity = row.get("entity")
    column = row.get("column")
    if not isinstance(entity, str) or not entity:
        raise FrameProjectionCodecError(f"{location}/entity must be non-empty")
    if not isinstance(column, str) or not column:
        raise FrameProjectionCodecError(f"{location}/column must be non-empty")
    return entity, column


def _physical_contract_columns(node: CompiledNode) -> dict[str, frozenset[str]]:
    """Mirror the executor's closed physical-column inventory."""

    columns: dict[str, set[str]] = {}
    rows = (*_input_rows(node), *_output_rows(node), *_write_scope_rows(node))
    for index, row in enumerate(rows):
        entity, column = _entity_column(row, location=f"physical_contract/{index}")
        if not column.startswith("@"):
            columns.setdefault(entity, set()).add(column)
    return {entity: frozenset(values) for entity, values in columns.items()}


def _schema_contract(
    schema: EntitySchema,
) -> tuple[
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, str]],
]:
    entity_keys = {
        entity: schema.entity_id_column(entity) for entity in schema.entities
    }
    membership_columns = {
        schema.person_entity: tuple(
            schema.membership_column(group) for group in schema.group_entities
        )
    }
    membership_targets = {
        schema.person_entity: {
            schema.membership_column(group): group
            for group in schema.group_entities
        }
    }
    return entity_keys, membership_columns, membership_targets


def _link_targets(
    frame: Frame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    specs = {link.name: link for link in frame.schema.links}
    tables: dict[str, pd.DataFrame] = {}
    targets: dict[str, dict[str, str]] = {}
    for name in frame.links:
        link = specs[name]
        target_map = {
            frame.schema.entity_id_column(link.left_entity): link.left_entity,
            frame.schema.entity_id_column(link.right_entity): link.right_entity,
        }
        tables[name] = frame.link(name).loc[:, list(target_map)].copy(deep=True)
        targets[name] = target_map
    return tables, targets


def _projection_tables(
    frame: Frame,
    node: CompiledNode,
) -> dict[str, pd.DataFrame]:
    physical = _physical_contract_columns(node)
    entity_keys, memberships, _targets = _schema_contract(frame.schema)
    result: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        implicit = {entity_keys[entity], *memberships.get(entity, ())}
        selected = implicit | set(physical.get(entity, ()))
        columns = [column for column in table.columns if column in selected]
        missing_implicit = sorted(implicit - set(columns))
        if missing_implicit:  # pragma: no cover - Frame validates this first
            raise FrameProjectionCodecError(
                f"frame is missing implicit columns for {entity!r}: "
                f"{missing_implicit!r}"
            )
        projected = table.loc[:, columns].copy(deep=True)
        projected.attrs = copy.deepcopy(table.attrs)
        result[entity] = projected
    return result


def _coerce_clone_index(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise FrameProjectionCodecError(f"{location} must be an integer")
    return int(value)


def _row_atom(
    *,
    entity: str,
    row_id: Hashable,
    channel: object,
    clone_index: object,
    registry: ClosedScopeRegistry,
) -> str:
    if not isinstance(channel, str) or not channel:
        raise FrameProjectionCodecError(
            f"row {entity}.{row_id!r} has an invalid support channel"
        )
    clone = _coerce_clone_index(
        clone_index,
        location=f"row {entity}.{row_id!r} support clone index",
    )
    atom = f"origin:{channel}/clone:{clone}"
    if atom not in registry.universe:
        raise FrameProjectionCodecError(
            f"row {entity}.{row_id!r} resolves to atom {atom!r}, outside "
            f"compiler predicate space {registry.predicate_space!r}"
        )
    return atom


def _frame_row_atoms(
    frame: Frame,
    registry: ClosedScopeRegistry,
) -> dict[str, dict[Hashable, frozenset[str]]]:
    result: dict[str, dict[Hashable, frozenset[str]]] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        key = frame.schema.entity_id_column(entity)
        channel_column = support_channel_column(entity)
        clone_column = support_clone_index_column(entity)
        missing = [
            column for column in (channel_column, clone_column) if column not in table
        ]
        if missing:
            raise FrameProjectionCodecError(
                f"entity {entity!r} lacks compiler row-lineage columns "
                f"{missing!r}"
            )
        rows: dict[Hashable, frozenset[str]] = {}
        for row_id, channel, clone_index in zip(
            table[key],
            table[channel_column],
            table[clone_column],
            strict=True,
        ):
            try:
                hash(row_id)
            except TypeError as error:  # pragma: no cover - Frame validates ids
                raise FrameProjectionCodecError(
                    f"entity {entity!r} has an unhashable stable id"
                ) from error
            rows[row_id] = frozenset(
                {
                    _row_atom(
                        entity=entity,
                        row_id=row_id,
                        channel=channel,
                        clone_index=clone_index,
                        registry=registry,
                    )
                }
            )
        result[entity] = rows
    return result


def _normalize_virtual_key(key: object) -> VirtualKey:
    if isinstance(key, tuple) and len(key) == 2:
        entity, column = key
    elif isinstance(key, str) and "." in key:
        entity, column = key.split(".", 1)
    else:
        raise FrameProjectionCodecError(
            "virtual keys must be (entity, column) pairs or 'entity.column' strings"
        )
    if not isinstance(entity, str) or not entity:
        raise FrameProjectionCodecError("virtual key entity must be non-empty")
    if not isinstance(column, str) or not column:
        raise FrameProjectionCodecError("virtual key column must be non-empty")
    return entity, column


def _normalize_virtual_mapping(
    values: Mapping[object, object] | None,
) -> dict[VirtualKey, object]:
    result: dict[VirtualKey, object] = {}
    for raw_key, value in (values or {}).items():
        key = _normalize_virtual_key(raw_key)
        if key in result:
            raise FrameProjectionCodecError(f"virtual key {key!r} is repeated")
        result[key] = copy.deepcopy(value)
    return result


def _tolerated_absence_receipts(node: CompiledNode) -> frozenset[str]:
    result: set[str] = set()
    for index, row in enumerate(
        _node_mapping(value, location=f"inputs/{index}")
        for index, value in enumerate(node.inputs)
    ):
        receipts = row.get("tolerated_absence_receipts", [])
        if not isinstance(receipts, list) or any(
            not isinstance(receipt, str) or not receipt for receipt in receipts
        ):
            raise FrameProjectionCodecError(
                f"inputs/{index}/tolerated_absence_receipts must be a string array"
            )
        result.update(receipts)
    return frozenset(result)


def _virtual_contract_keys(node: CompiledNode) -> frozenset[VirtualKey]:
    result: set[VirtualKey] = set()
    for index, row in enumerate((*_input_rows(node), *_output_rows(node))):
        entity, column = _entity_column(row, location=f"virtual_contract/{index}")
        if column.startswith("@"):
            result.add((entity, column))
    return frozenset(result)


def _virtual_inputs(
    frame: Frame,
    node: CompiledNode,
    available_inputs: Mapping[object, object] | None,
) -> dict[VirtualKey, object]:
    supplied = _normalize_virtual_mapping(available_inputs)
    contract_keys = _virtual_contract_keys(node)
    tolerated = _tolerated_absence_receipts(node)
    extras = sorted(
        key
        for key in supplied
        if key not in contract_keys and key[1] not in tolerated
    )
    if extras:
        raise FrameProjectionCodecError(
            f"available virtual inputs are outside node {node.id!r}: {extras!r}"
        )

    result = dict(supplied)
    for entity, column in contract_keys:
        key = (entity, column)
        if key in result:
            continue
        if column == "@resolved_weight" and entity in frame.entities:
            resolved = frame.resolve_weights(entity)
            result[key] = resolved.values
            continue
        if entity == "frame":
            metadata_key = column.removeprefix("@")
            if metadata_key in frame.metadata:
                result[key] = copy.deepcopy(frame.metadata[metadata_key])
    return result


def frame_to_projection(
    frame: Frame,
    *,
    node: CompiledNode,
    available_inputs: Mapping[object, object] | None = None,
) -> ImmutableFrameProjection:
    """Project one full frame onto the exact surface declared by ``node``.

    Support channel and clone columns are read from the full frame to derive
    trusted row atoms; they are exposed to the kernel only when the compiled
    physical contract itself names them.
    """

    if not isinstance(frame, Frame):
        raise TypeError("frame_to_projection requires a Frame")
    if not isinstance(node, CompiledNode):
        raise TypeError("frame_to_projection requires a CompiledNode")
    frame.revalidate()
    registry = _node_scope_registry(node)
    entity_keys, membership_columns, membership_targets = _schema_contract(
        frame.schema
    )
    links, link_targets = _link_targets(frame)
    weights = {
        entity: WeightState(
            frame.weights_for(entity).values,
            frame.weights_for(entity).kind.value,
        )
        for entity in frame.weighted_entities
    }
    try:
        return ImmutableFrameProjection(
            _projection_tables(frame, node),
            entity_keys=entity_keys,
            membership_columns=membership_columns,
            membership_targets=membership_targets,
            links=links,
            link_targets=link_targets,
            weights=weights,
            strata=frame.strata,
            strata_entity=frame.schema.person_entity,
            mass_history=frame.mass_log,
            metadata=frame.metadata,
            virtual_receipts=_virtual_inputs(frame, node, available_inputs),
            row_atoms=_frame_row_atoms(frame, registry),
        )
    except FrameProjectionCodecError:
        raise
    except (TypeError, ValueError) as error:
        raise FrameProjectionCodecError(
            f"frame cannot be represented by the executor projection: {error}"
        ) from error


def projection_to_frame(
    projection: ImmutableFrameProjection,
    *,
    schema: EntitySchema,
) -> Frame:
    """Materialize the exact kernel-visible projection as a narrow ``Frame``.

    Legacy physical functions can therefore retain their typed ``Frame`` API
    without receiving any column that the compiled node did not declare.
    Virtual receipts remain separate inputs and are intentionally not copied
    into frame metadata.
    """

    if not isinstance(projection, ImmutableFrameProjection):
        raise TypeError("projection_to_frame requires an immutable projection")
    if not isinstance(schema, EntitySchema):
        raise TypeError("projection_to_frame requires an EntitySchema")
    if projection.entities != schema.entities:
        raise FrameProjectionCodecError(
            "projection entity order differs from the supplied frame schema"
        )
    parts = projection._parts()
    tables = {
        entity: projection.table(entity) for entity in projection.entities
    }
    tables.update(
        {name: table.copy(deep=True) for name, table in parts["links"].items()}
    )
    weights = {
        entity: Weights(
            projection.weights_for(entity).values,
            WeightKind(projection.weights_for(entity).kind),
        )
        for entity in parts["weights"]
    }
    return Frame(
        tables,
        schema,
        weights,
        projection.strata,
        mass_log=projection.mass_history,
        metadata=projection.metadata,
    )


def _native_source_candidates(
    *,
    entity: str,
    row: pd.Series,
    row_id: Hashable,
    clone_index: int,
    native_ids: frozenset[Hashable],
) -> frozenset[Hashable]:
    explicit: set[Hashable] = set()
    for column in (
        support_source_id_column(entity),
        spine_source_id_column(entity),
    ):
        if column not in row.index:
            continue
        candidate = row[column]
        try:
            if candidate in native_ids:
                explicit.add(candidate)
        except TypeError:
            continue
    if len(explicit) > 1:
        raise FrameProjectionCodecError(
            f"added row {entity}.{row_id!r} has ambiguous explicit source ids "
            f"{sorted(map(repr, explicit))!r}"
        )
    if explicit:
        return frozenset(explicit)

    if (
        isinstance(row_id, bool)
        or not isinstance(row_id, Integral)
        or clone_index <= 0
    ):
        return frozenset()
    remapped = int(row_id)
    max_digits = max(1, len(str(abs(remapped))))
    inferred: set[Hashable] = set()
    for power in range(1, max_digits + 2):
        candidate = remapped - clone_index * (10**power)
        if candidate in native_ids:
            inferred.add(candidate)
    return frozenset(inferred)


def classify_added_support_rows(
    entity: str,
    table: pd.DataFrame,
    entity_key: str,
    added_ids: frozenset[Hashable],
    registry: ClosedScopeRegistry,
) -> Mapping[Hashable, RowClassification]:
    """Classify EXPAND rows from compiler atoms and recover native sources.

    Explicit entity-prefixed support/spine source ids win when they name a
    native stable id.  Older clone frames without that lineage are supported
    only when the canonical power-of-ten remap has exactly one native inverse.
    Ambiguous or missing lineage is refused.
    """

    if not isinstance(entity, str) or not entity:
        raise FrameProjectionCodecError("classifier entity must be non-empty")
    if not isinstance(table, pd.DataFrame):
        raise TypeError("classifier table must be a DataFrame")
    if entity_key not in table:
        raise FrameProjectionCodecError(
            f"classifier table for {entity!r} lacks key {entity_key!r}"
        )
    required = (support_channel_column(entity), support_clone_index_column(entity))
    missing = [column for column in required if column not in table]
    if missing:
        raise FrameProjectionCodecError(
            f"classifier table for {entity!r} lacks support lineage {missing!r}"
        )
    if table[entity_key].duplicated().any():
        raise FrameProjectionCodecError(
            f"classifier key {entity}.{entity_key} must be unique"
        )
    rows = table.set_index(entity_key, drop=False)
    observed_ids = frozenset(rows.index.tolist())
    if not added_ids <= observed_ids:
        raise FrameProjectionCodecError(
            f"classifier added ids are absent from {entity!r}: "
            f"{sorted(map(repr, added_ids - observed_ids))!r}"
        )
    native_ids = observed_ids - added_ids
    if not native_ids:
        raise FrameProjectionCodecError(
            f"classifier for {entity!r} has no native source rows"
        )

    result: dict[Hashable, RowClassification] = {}
    for row_id in added_ids:
        row = rows.loc[row_id]
        if isinstance(row, pd.DataFrame):  # pragma: no cover - duplicate check above
            raise FrameProjectionCodecError(
                f"classifier key {entity}.{row_id!r} is not unique"
            )
        clone_index = _coerce_clone_index(
            row[support_clone_index_column(entity)],
            location=f"row {entity}.{row_id!r} support clone index",
        )
        atom = _row_atom(
            entity=entity,
            row_id=row_id,
            channel=row[support_channel_column(entity)],
            clone_index=clone_index,
            registry=registry,
        )
        sources = _native_source_candidates(
            entity=entity,
            row=row,
            row_id=row_id,
            clone_index=clone_index,
            native_ids=native_ids,
        )
        if len(sources) != 1:
            raise FrameProjectionCodecError(
                f"added row {entity}.{row_id!r} must resolve to exactly one "
                f"native source id; candidates={sorted(map(repr, sources))!r}"
            )
        result[row_id] = RowClassification(
            atoms=frozenset({atom}),
            source_row_id=next(iter(sources)),
        )
    return result


def _exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    try:
        return pickle.dumps(left, protocol=5) == pickle.dumps(right, protocol=5)
    except (AttributeError, pickle.PickleError, TypeError, ValueError):
        return False


def _metadata_normal_form(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(
            (key, _metadata_normal_form(child)) for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_metadata_normal_form(child) for child in value)
    if isinstance(value, set | frozenset):
        normalized = [_metadata_normal_form(child) for child in value]
        return tuple(sorted(normalized, key=lambda child: pickle.dumps(child, 5)))
    return value


def _metadata_equal(left: object, right: object) -> bool:
    return _exact_equal(_metadata_normal_form(left), _metadata_normal_form(right))


def _metadata_changes(
    before: Frame,
    result: Frame,
    virtual_writes: Mapping[VirtualKey, object],
) -> None:
    before_metadata = before.metadata
    result_metadata = result.metadata
    removed = sorted(set(before_metadata) - set(result_metadata))
    if removed:
        raise FrameProjectionCodecError(
            f"legacy result removed immutable frame metadata: {removed!r}"
        )
    for key, value in result_metadata.items():
        if key in before_metadata and _metadata_equal(before_metadata[key], value):
            continue
        virtual_key = ("frame", f"@{key}")
        if virtual_key not in virtual_writes or not _metadata_equal(
            virtual_writes[virtual_key], value
        ):
            raise FrameProjectionCodecError(
                f"frame metadata change {key!r} lacks an equal declared "
                f"virtual write {virtual_key!r}"
            )


def _value_at(table: pd.DataFrame, key: str, row_id: Hashable, column: str) -> object:
    selected = table.loc[table[key].eq(row_id), column]
    if len(selected) != 1:  # pragma: no cover - Frame validates stable ids
        raise FrameProjectionCodecError(
            f"stable row {row_id!r} is not unique in {key!r}"
        )
    return selected.iloc[0]


def _validate_external_table_columns(
    before: Frame,
    result: Frame,
    node: CompiledNode,
    registry: ClosedScopeRegistry,
) -> None:
    physical = _physical_contract_columns(node)
    entity_keys, memberships, _membership_targets = _schema_contract(before.schema)
    for entity in before.entities:
        before_table = before.table(entity)
        result_table = result.table(entity)
        implicit = {entity_keys[entity], *memberships.get(entity, ())}
        executor_columns = implicit | set(physical.get(entity, ()))
        removed_columns = set(before_table) - set(result_table)
        added_columns = set(result_table) - set(before_table)
        if removed_columns:
            raise FrameProjectionCodecError(
                f"legacy result removed columns from {entity!r}: "
                f"{sorted(removed_columns)!r}"
            )
        unauthorized_added = added_columns - executor_columns
        if unauthorized_added:
            raise FrameProjectionCodecError(
                f"legacy result added columns outside node {node.id!r} on "
                f"{entity!r}: {sorted(unauthorized_added)!r}"
            )

        key = entity_keys[entity]
        before_ids = frozenset(before_table[key].tolist())
        result_ids = frozenset(result_table[key].tolist())
        added_ids = result_ids - before_ids
        classifications: Mapping[Hashable, RowClassification] = {}
        if added_ids:
            classifications = classify_added_support_rows(
                entity,
                result_table,
                key,
                added_ids,
                registry,
            )
        external_columns = [
            column for column in before_table if column not in executor_columns
        ]
        for column in external_columns:
            if str(before_table[column].dtype) != str(result_table[column].dtype):
                raise FrameProjectionCodecError(
                    f"external column dtype changed for {entity}.{column}"
                )
            for row_id in before_ids & result_ids:
                if not _exact_equal(
                    _value_at(before_table, key, row_id, column),
                    _value_at(result_table, key, row_id, column),
                ):
                    raise FrameProjectionCodecError(
                        f"external column changed outside executor authority at "
                        f"{entity}.{column}[{row_id!r}]"
                    )
            for row_id, classification in classifications.items():
                if not _exact_equal(
                    _value_at(
                        before_table,
                        key,
                        classification.source_row_id,
                        column,
                    ),
                    _value_at(result_table, key, row_id, column),
                ):
                    raise FrameProjectionCodecError(
                        f"expanded row did not copy external column exactly at "
                        f"{entity}.{column}[{row_id!r}]"
                    )


def _validate_external_link_columns(before: Frame, result: Frame) -> None:
    if before.links != result.links:
        raise FrameProjectionCodecError(
            "legacy result changed the executor link-table inventory"
        )
    specs = {link.name: link for link in before.schema.links}
    for name in before.links:
        link = specs[name]
        targets = (
            before.schema.entity_id_column(link.left_entity),
            before.schema.entity_id_column(link.right_entity),
        )
        before_table = before.link(name)
        result_table = result.link(name)
        external = [column for column in before_table if column not in targets]
        if set(before_table) != set(result_table):
            raise FrameProjectionCodecError(
                f"legacy result changed link columns for {name!r}"
            )
        before_keys = {
            tuple(row) for row in before_table.loc[:, list(targets)].itertuples(index=False)
        }
        result_keys = {
            tuple(row) for row in result_table.loc[:, list(targets)].itertuples(index=False)
        }
        if external and before_keys != result_keys:
            raise FrameProjectionCodecError(
                f"link {name!r} changed row keys while carrying columns outside "
                "executor authority"
            )
        if not external:
            continue
        before_indexed = before_table.set_index(list(targets), drop=False)
        result_indexed = result_table.set_index(list(targets), drop=False)
        for row_key in before_keys:
            for column in external:
                if not _exact_equal(
                    before_indexed.loc[row_key, column],
                    result_indexed.loc[row_key, column],
                ):
                    raise FrameProjectionCodecError(
                        f"external link column changed at {name}.{column}{row_key!r}"
                    )


def _validate_external_surfaces(
    before: Frame,
    result: Frame,
    node: CompiledNode,
) -> None:
    before.revalidate()
    result.revalidate()
    if before.schema != result.schema:
        raise FrameProjectionCodecError("legacy result changed the Frame schema")
    registry = _node_scope_registry(node)
    _validate_external_table_columns(before, result, node, registry)
    _validate_external_link_columns(before, result)


def legacy_result_to_patch(
    before_frame: Frame,
    result_frame: Frame,
    *,
    node: CompiledNode,
    virtual_writes: Mapping[object, object] | None = None,
) -> KernelPatch:
    """Encode a legacy callback result as a node-bounded executor patch.

    Frame metadata is intentionally not placed in ``KernelPatch.metadata``:
    executor metadata is immutable.  Each metadata addition/change must be an
    equal ``frame.@...`` virtual write and is restored only after the executor
    validates that write.
    """

    if not isinstance(before_frame, Frame) or not isinstance(result_frame, Frame):
        raise TypeError("legacy_result_to_patch requires Frame inputs")
    if not isinstance(node, CompiledNode):
        raise TypeError("legacy_result_to_patch requires a CompiledNode")
    normalized_writes = _normalize_virtual_mapping(virtual_writes)
    declared_virtual_outputs = {
        _entity_column(row, location=f"outputs/{index}")
        for index, row in enumerate(_output_rows(node))
        if isinstance(row.get("column"), str)
        and str(row["column"]).startswith("@")
    }
    extras = sorted(set(normalized_writes) - declared_virtual_outputs)
    if extras:
        raise FrameProjectionCodecError(
            f"virtual writes are outside node {node.id!r} outputs: {extras!r}"
        )
    _validate_external_surfaces(before_frame, result_frame, node)
    _metadata_changes(before_frame, result_frame, normalized_writes)

    before_links = set(before_frame.links)
    result_links = set(result_frame.links)
    projected_links, _targets = _link_targets(result_frame)
    result_weights = {
        entity: WeightState(
            result_frame.weights_for(entity).values,
            result_frame.weights_for(entity).kind.value,
        )
        for entity in result_frame.weighted_entities
    }
    return KernelPatch(
        structural_delta=_node_structural_delta(node),
        tables=_projection_tables(result_frame, node),
        links=projected_links,
        drop_links=frozenset(before_links - result_links),
        weights=result_weights,
        drop_weights=frozenset(
            set(before_frame.weighted_entities) - set(result_frame.weighted_entities)
        ),
        strata=result_frame.strata,
        replace_strata=True,
        mass_history=result_frame.mass_log,
        metadata=None,
        virtual_writes=normalized_writes,
    )


def _assert_projection_matches_result(
    projection: ImmutableFrameProjection,
    result: Frame,
    node: CompiledNode,
) -> None:
    expected = frame_to_projection(result, node=node)
    actual_parts = projection._parts()
    expected_parts = expected._parts()
    for field in (
        "tables",
        "entity_keys",
        "membership_columns",
        "membership_targets",
        "links",
        "link_targets",
        "weights",
        "strata",
        "strata_entity",
        "mass_history",
        "row_atoms",
    ):
        if not _exact_equal(actual_parts[field], expected_parts[field]):
            raise FrameProjectionCodecError(
                f"validated projection differs from the legacy result on {field}"
            )


def merge_projection_into_frame(
    projection: ImmutableFrameProjection,
    *,
    before_frame: Frame,
    legacy_result_frame: Frame,
    node: CompiledNode,
    validated_metadata: Mapping[str, Any],
) -> Frame:
    """Restore a validated narrow projection to its checked full-frame result.

    ``validated_metadata`` is an explicit trust-boundary argument.  It must be
    exactly the metadata carried by ``legacy_result_frame``; the caller should
    provide it only after the corresponding virtual writes have passed the
    executor.
    """

    if not isinstance(projection, ImmutableFrameProjection):
        raise TypeError("merge_projection_into_frame requires a projection")
    if not isinstance(before_frame, Frame) or not isinstance(
        legacy_result_frame, Frame
    ):
        raise TypeError("merge_projection_into_frame requires Frame inputs")
    if not isinstance(validated_metadata, Mapping):
        raise TypeError("validated_metadata must be a mapping")
    if not _metadata_equal(validated_metadata, legacy_result_frame.metadata):
        raise FrameProjectionCodecError(
            "validated metadata differs from the legacy result metadata"
        )
    _validate_external_surfaces(before_frame, legacy_result_frame, node)
    _assert_projection_matches_result(projection, legacy_result_frame, node)

    tables = {
        entity: legacy_result_frame.table(entity).copy(deep=True)
        for entity in legacy_result_frame.entities
    }
    tables.update(
        {
            name: legacy_result_frame.link(name).copy(deep=True)
            for name in legacy_result_frame.links
        }
    )
    weights = {
        entity: Weights(
            legacy_result_frame.weights_for(entity).values,
            WeightKind(legacy_result_frame.weights_for(entity).kind.value),
        )
        for entity in legacy_result_frame.weighted_entities
    }
    return Frame(
        tables,
        legacy_result_frame.schema,
        weights,
        legacy_result_frame.strata,
        mass_log=legacy_result_frame.mass_log,
        metadata=validated_metadata,
    )
