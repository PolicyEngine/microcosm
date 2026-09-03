"""Graph execution: immutable projection, validation, patching, and reuse."""

from __future__ import annotations

import hashlib
import json
import socket
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame import Frame, WeightKind, Weights

from .canonical import canonical_json, sha256_domain
from .codecs import SOURCE_CODECS, SourceCodecRegistry
from .decl import (
    GATE_OUTCOMES,
    ROWS_ALL,
    CompiledGraph,
    Node,
    Owned,
    Ownership,
    StructuralDelta,
)
from .errors import NodeRejectedError
from .kernel import (
    Capabilities,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Tolerance,
)
from .keys import (
    _capabilities_projection,
    artifact_key,
    frame_key,
    node_key,
    seed,
    source_content_key,
    weights_key,
)
from .manifest import Decision, NodeReceipt, RunManifest
from .population import (
    Population,
    _expand_cells,
    entrant_strata_receipt,
    expand_lineage_receipt,
    expand_writes_receipt,
    mass_record_receipt,
    patch,
    restore_cached_expand,
    weight_cap_receipt,
)
from .store import (
    ContentStore,
    ResumePolicy,
    StoreCorrupt,
    StoreMiss,
    StoreUnavailable,
)

__all__ = ["NodeRejected", "NodeRejectedError", "run_graph"]

# Compatibility spelling from the initial interface.  Every rejection is an
# instance of the amended shared runtime exception.
NodeRejected = NodeRejectedError

_CERTIFYING_GATE_OUTCOMES = frozenset({"pass", "not_applicable"})
_EXPAND_WRITE_CLASSES = ("entrant", "copied-rewrite", "new-column")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _cache_record_key(key: str) -> str:
    return sha256_domain("node-receipt", canonical_json((key,)))


def _opaque_artifact_key(key: str, name: str) -> str:
    return sha256_domain("node-artifact", canonical_json((key, name)))


def _normal_json_mapping(value: Mapping[str, object], label: str) -> dict[str, object]:
    """Validate and detach a descriptive mapping through canonical JSON."""

    try:
        restored = json.loads(canonical_json(value))
    except (TypeError, ValueError) as error:
        raise NodeRejected(
            f"{label} must contain canonical JSON values: {error}"
        ) from error
    if not isinstance(restored, dict):  # pragma: no cover - Mapping encodes as object
        raise NodeRejected(f"{label} must encode as an object.")
    return restored


def _failed_gate_result(
    node: Node,
    population: Population | None,
    error: Exception,
) -> KernelResult:
    """Turn a gate evaluation exception into an owned, evidenced failure."""

    if node.structural is not StructuralDelta.NONE or population is None:
        raise NodeRejected(
            f"Gate node {node.id!r} cannot recover an exception without an "
            "ordinary population-bound verdict column."
        ) from error
    columns: dict[tuple[str, str], pd.Series] = {}
    for owned in node.outputs:
        if owned.dtype != "string":
            raise NodeRejected(
                f"Gate node {node.id!r} output {owned.entity}.{owned.column} "
                "must declare dtype 'string' to record a failed verdict."
            ) from error
        ids = _owned_ids(population.frame, owned, node_id=node.id)
        columns[(owned.entity, owned.column)] = pd.Series(
            "fail",
            index=ids,
            name=owned.column,
            dtype="string",
        )
    return KernelResult(
        columns=MappingProxyType(columns),
        receipt={
            "outcome": "fail",
            "evidence": {
                "exception_type": type(error).__name__,
                "message": str(error),
            },
        },
    )


def _transitive_ancestors(compiled: CompiledGraph, node_id: str) -> tuple[str, ...]:
    """Return every predecessor of ``node_id`` in canonical node order."""

    pending = list(compiled.predecessors[node_id])
    ancestors: set[str] = set()
    while pending:
        predecessor = pending.pop()
        if predecessor in ancestors:
            continue
        ancestors.add(predecessor)
        pending.extend(compiled.predecessors[predecessor])
    return tuple(candidate for candidate in compiled.order if candidate in ancestors)


def _release_tier(
    compiled: CompiledGraph,
    node_id: str,
    receipts: Mapping[str, NodeReceipt],
) -> tuple[str, tuple[str, ...]]:
    """Derive a release tier solely from gate receipts in its ancestry."""

    gate_ids = tuple(
        ancestor
        for ancestor in _transitive_ancestors(compiled, node_id)
        if receipts[ancestor].capabilities.role is KernelRole.GATE
    )
    certified = all(
        receipts[gate_id].receipt.get("outcome") in _CERTIFYING_GATE_OUTCOMES
        for gate_id in gate_ids
    )
    return ("certified" if certified else "evidence"), gate_ids


def _validate_release_tier(node: Node, result: KernelResult, derived: str) -> None:
    """Require a release kernel's owned tier answer to equal the derivation."""

    tier_outputs = [owned for owned in node.outputs if owned.column == "tier"]
    if len(tier_outputs) != 1 or tier_outputs[0].dtype != "string":
        raise NodeRejected(
            f"Release node {node.id!r} must own exactly one string tier column."
        )
    owned = tier_outputs[0]
    series = result.columns[(owned.entity, owned.column)]
    answers = set(series.dropna().astype(str))
    if series.isna().any() or answers != {derived}:
        raise NodeRejected(
            f"Release node {node.id!r} returned tier {sorted(answers)!r}, "
            f"but gate ancestry derives {derived!r}."
        )


def _decision_names(decisions: tuple[Decision, ...]) -> frozenset[str]:
    names: set[str] = set()
    for decision in decisions:
        payload = dict(decision)
        names.add(payload.get("name", decision.kind))
    return frozenset(names)


def _required_decision_names(node: Node) -> tuple[str, ...]:
    """Validate and return a release node's normative decision requirements."""

    required = node.params.get("requires_decisions", ())
    if not isinstance(required, tuple) or any(
        not isinstance(name, str) or not name for name in required
    ):
        raise NodeRejected(
            f"Release node {node.id!r} params['requires_decisions'] must be a "
            "tuple of non-empty decision names."
        )
    if len(set(required)) != len(required):
        raise NodeRejected(
            f"Release node {node.id!r} repeats a required decision name."
        )
    return required


def _release_outcome(node: Node, tier: str, decisions: tuple[Decision, ...]) -> str:
    required = _required_decision_names(node)
    if not set(required) <= _decision_names(decisions):
        return "unreached"
    return "pass" if tier == "certified" else "fail"


def _dtype_matches(series: pd.Series, token: str) -> bool:
    dtype = series.dtype
    if token == "boolean":
        return isinstance(dtype, pd.BooleanDtype)
    if token == "Int64":
        return isinstance(dtype, pd.Int64Dtype)
    if token == "string":
        return isinstance(dtype, pd.StringDtype)
    expected = {
        "bool": np.dtype(np.bool_),
        "int32": np.dtype(np.int32),
        "int64": np.dtype(np.int64),
        "float32": np.dtype(np.float32),
        "float64": np.dtype(np.float64),
    }[token]
    return dtype == expected


def _dtype_token(series: pd.Series) -> str:
    dtype = series.dtype
    if isinstance(dtype, pd.BooleanDtype):
        return "boolean"
    if isinstance(dtype, pd.Int64Dtype):
        return "Int64"
    if isinstance(dtype, pd.StringDtype):
        return "string"
    numpy_dtype = np.dtype(dtype)
    by_dtype = {
        np.dtype(np.bool_): "bool",
        np.dtype(np.int32): "int32",
        np.dtype(np.int64): "int64",
        np.dtype(np.float32): "float32",
        np.dtype(np.float64): "float64",
    }
    try:
        return by_dtype[numpy_dtype]
    except KeyError as error:
        raise NodeRejected(
            f"Frame column has unsupported dtype {dtype!s}; graph columns use "
            "the frozen DTYPES tokens."
        ) from error


def _mask_values(table: pd.DataFrame, column: str, *, node_id: str) -> np.ndarray:
    series = table[column]
    if not (
        pd.api.types.is_bool_dtype(series.dtype)
        or isinstance(series.dtype, pd.BooleanDtype)
    ):
        raise NodeRejected(
            f"Node {node_id!r} row mask {column!r} has dtype {series.dtype!s}, "
            "not bool/boolean."
        )
    if series.isna().any():
        raise NodeRejected(f"Node {node_id!r} row mask {column!r} contains nulls.")
    return series.to_numpy(dtype=np.bool_, copy=True)


def _owned_ids(frame: Frame, owned: Owned, *, node_id: str) -> pd.Index:
    table = frame.table(owned.entity)
    id_column = frame.schema.entity_id_column(owned.entity)
    if owned.rows == ROWS_ALL:
        selected = table
    else:
        selected = table.loc[_mask_values(table, owned.rows, node_id=node_id)]
    return pd.Index(selected[id_column].to_numpy(copy=True), name=id_column)


def _set_read_only(array: object) -> None:
    if isinstance(array, np.ndarray):
        try:
            array.setflags(write=False)
        except ValueError:
            pass


def _freeze_series(series: pd.Series) -> None:
    """Make every discoverable backing buffer read-only in place."""

    extension = series.array
    for attribute in ("_data", "_mask", "_ndarray"):
        _set_read_only(getattr(extension, attribute, None))
    try:
        _set_read_only(series.to_numpy(copy=False))
    except (TypeError, ValueError):
        pass


def _freeze_frame(table: pd.DataFrame) -> pd.DataFrame:
    frozen = table.copy(deep=True)
    for column in frozen.columns:
        _freeze_series(frozen[column])
    _set_read_only(frozen.index.to_numpy(copy=False))
    return frozen


def _update_scalar(digest: hashlib._Hash, value: object) -> None:
    if value is pd.NA:
        payload = b"pd.NA"
    elif value is pd.NaT:
        payload = b"pd.NaT"
    elif value is None:
        payload = b"None"
    elif isinstance(value, (float, np.floating)):
        payload = b"f" + np.asarray([value], dtype=np.float64).tobytes()
    elif isinstance(value, (bool, np.bool_)):
        payload = b"b1" if bool(value) else b"b0"
    elif isinstance(value, (int, np.integer)):
        payload = b"i" + str(int(value)).encode("ascii")
    elif isinstance(value, str):
        payload = b"s" + value.encode("utf-8")
    elif isinstance(value, (bytes, np.bytes_)):
        payload = b"y" + bytes(value)
    else:
        payload = b"r" + repr(value).encode("utf-8")
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _update_array(digest: hashlib._Hash, values: object) -> None:
    array = np.asarray(values)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    if array.dtype.hasobject:
        for value in array.ravel(order="C"):
            _update_scalar(digest, value)
    else:
        digest.update(np.ascontiguousarray(array).tobytes())


def _update_series(digest: hashlib._Hash, series: pd.Series) -> None:
    digest.update(str(series.dtype).encode("utf-8"))
    digest.update(b"\0")
    extension = series.array
    data = getattr(extension, "_data", None)
    mask = getattr(extension, "_mask", None)
    if isinstance(data, np.ndarray) and isinstance(mask, np.ndarray):
        _update_array(digest, data)
        _update_array(digest, mask)
    else:
        _update_array(digest, series.to_numpy(dtype=object, copy=False))
    _update_array(digest, series.index.to_numpy(copy=False))


def _context_digest(context: KernelContext) -> bytes:
    digest = hashlib.sha256(b"microcosm-graph/kernel-context/1\0")
    for entity in sorted(context.tables):
        table = context.tables[entity]
        digest.update(entity.encode("utf-8") + b"\0")
        for column in table.columns:
            digest.update(str(column).encode("utf-8") + b"\0")
            _update_series(digest, table[column])
    for entity in sorted(context.weights):
        weights = context.weights[entity]
        digest.update(entity.encode("utf-8") + b"\0")
        digest.update(weights.kind.value.encode("ascii") + b"\0")
        _update_array(digest, weights.values)
    _update_series(digest, context.strata)
    return digest.digest()


def _structural_columns(frame: Frame, entity: str) -> list[str]:
    columns = [frame.schema.entity_id_column(entity)]
    if entity == frame.schema.person_entity:
        columns.extend(
            frame.schema.membership_column(group)
            for group in frame.schema.group_entities
        )
    return columns


def _materialized_expand_coordinates(node: Node) -> frozenset[tuple[str, str]]:
    """Return and validate the carried EXPAND cells an ordinary node claims."""

    raw_materialized = node.params.get("materialized_expand_outputs", ())
    if not isinstance(raw_materialized, tuple) or any(
        not isinstance(value, str) or "." not in value for value in raw_materialized
    ):
        raise NodeRejected(
            f"Node {node.id!r} params['materialized_expand_outputs'] must be a "
            "tuple of 'entity.column' strings."
        )
    materialized: set[tuple[str, str]] = set()
    owned_by_coordinate = {
        (output.entity, output.column): output for output in node.outputs
    }
    for value in raw_materialized:
        entity, column = value.split(".", 1)
        coordinate = (entity, column)
        output = owned_by_coordinate.get(coordinate)
        if output is None or output.rewrite:
            raise NodeRejected(
                f"Node {node.id!r} materialized EXPAND output {value!r} must be "
                "one of its non-rewrite owned cells."
            )
        materialized.add(coordinate)
    if len(materialized) != len(raw_materialized):
        raise NodeRejected(f"Node {node.id!r} repeats a materialized EXPAND output.")
    return frozenset(materialized)


def _expand_rewrite_coordinates(
    compiled: CompiledGraph, node: Node
) -> frozenset[tuple[str, str]]:
    """Return overlays an EXPAND's full-cell same-version claimant rewrites."""

    if node.structural is not StructuralDelta.EXPAND:
        return frozenset()
    overlay_coordinates = _expand_writer_coordinates(node)
    rewrites: set[tuple[str, str]] = set()
    for (version, entity, column), owner_id in compiled.owners.items():
        coordinate = (entity, column)
        if version != node.id or coordinate not in overlay_coordinates:
            continue
        owner = compiled.graph.node(owner_id)
        output = next(
            candidate
            for candidate in owner.outputs
            if (candidate.entity, candidate.column) == coordinate
        )
        if output.rewrite and output.rows == ROWS_ALL:
            rewrites.add(coordinate)
    return frozenset(rewrites)


def _project_context(
    node: Node,
    population: Population | None,
    *,
    key: str,
    sources: Mapping[str, Path],
    tolerances: Mapping[tuple[str, str], Tolerance | None],
) -> KernelContext:
    if population is None:
        return KernelContext(
            node=node,
            tables=MappingProxyType({}),
            weights=MappingProxyType({}),
            strata=pd.Series([], dtype=object, name="stratum"),
            params=node.params,
            rng=np.random.default_rng(seed(key)),
            sources=MappingProxyType({name: sources[name] for name in node.sources}),
            tolerances=tolerances,
        )

    frame = population.frame
    slices: dict[str, list[object]] = {}
    for slice_ in node.inputs:
        slices.setdefault(slice_.entity, []).append(slice_)

    materialized = _materialized_expand_coordinates(node)
    for entity, column in materialized:
        coordinate = (entity, column)
        if population.owners.get(coordinate) != population.version:
            raise NodeRejected(
                f"Node {node.id!r} materialized EXPAND output "
                f"{entity}.{column!s} was not "
                f"installed by population version {population.version!r}."
            )

    tables: dict[str, pd.DataFrame] = {}
    entity_masks: dict[str, np.ndarray] = {}
    projected_entities = set(slices)
    projected_entities.update(owned.entity for owned in node.outputs)
    for entity in sorted(projected_entities):
        table = frame.table(entity)
        entity_slices = slices.get(entity, [])
        row_specs = {slice_.rows for slice_ in entity_slices}
        if len(row_specs) > 1:
            raise NodeRejected(
                f"Node {node.id!r} declares incompatible row masks for entity "
                f"{entity!r}; KernelContext has one table per entity."
            )
        columns = _structural_columns(frame, entity)
        for slice_ in entity_slices:
            columns.extend(slice_.columns)
        for owned in node.outputs:
            if owned.entity != entity or not owned.rewrite:
                continue
            if owned.column not in table:
                raise NodeRejected(
                    f"Node {node.id!r} rewrite incumbent "
                    f"{owned.entity}.{owned.column} is absent."
                )
            columns.append(owned.column)
        for materialized_entity, materialized_column in sorted(materialized):
            if materialized_entity != entity:
                continue
            if materialized_column not in table:
                raise NodeRejected(
                    f"Node {node.id!r} materialized EXPAND output "
                    f"{materialized_entity}.{materialized_column} is absent."
                )
            columns.append(materialized_column)
        columns = list(dict.fromkeys(columns))
        if row_specs and next(iter(row_specs)) != ROWS_ALL:
            row_column = str(next(iter(row_specs)))
            mask = _mask_values(table, row_column, node_id=node.id)
        else:
            mask = np.ones(len(table), dtype=np.bool_)
        entity_masks[entity] = mask
        tables[entity] = _freeze_frame(table.loc[mask, columns])

    weights: dict[str, Weights] = {}
    for entity in sorted(projected_entities):
        try:
            effective = frame.resolve_weights(entity)
        except ValueError:
            if entity in frame.weighted_entities:
                raise
            # Some coarser groups contain members whose inherited person
            # weights differ, so Frame deliberately refuses to invent one
            # group weight.  The frozen context has no "needs weights" bit;
            # omit that ambiguous inherited entry while retaining its table.
            continue
        values = effective.values[entity_masks[entity]]
        weights[entity] = Weights(values=values, kind=effective.kind)

    person_mask = entity_masks.get(
        frame.schema.person_entity,
        np.ones(frame.n(frame.schema.person_entity), dtype=np.bool_),
    )
    strata = frame.strata.loc[person_mask].copy()
    _freeze_series(strata)
    return KernelContext(
        node=node,
        tables=MappingProxyType(tables),
        weights=MappingProxyType(weights),
        strata=strata,
        params=node.params,
        rng=np.random.default_rng(seed(key)),
        sources=MappingProxyType({name: sources[name] for name in node.sources}),
        tolerances=tolerances,
    )


def _input_tolerances(
    compiled: CompiledGraph,
    node_id: str,
    kernels: KernelRegistry,
    *,
    writers: Mapping[tuple[str, str], tuple[str, ...]] | None = None,
) -> Mapping[tuple[str, str], Tolerance | None]:
    """Resolve each read coordinate to the loosest tolerance of its writers."""

    writer_map = _input_writers(compiled, node_id) if writers is None else writers
    resolved: dict[tuple[str, str], Tolerance | None] = {}
    for coordinate, writer_ids in writer_map.items():
        bounds = [
            tolerance
            for writer_id in writer_ids
            if (
                tolerance := kernels.get(
                    compiled.graph.node(writer_id).kernel
                ).capabilities.tolerance
            )
            is not None
        ]
        resolved[coordinate] = (
            None
            if not bounds
            else Tolerance(
                rtol=max(bound.rtol for bound in bounds),
                atol=max(bound.atol for bound in bounds),
                ulps=max(bound.ulps for bound in bounds),
            )
        )
    return MappingProxyType(resolved)


def _input_writers(
    compiled: CompiledGraph,
    node_id: str,
    *,
    receipts: Mapping[str, NodeReceipt] | None = None,
) -> Mapping[tuple[str, str], tuple[str, ...]]:
    """Return causal writer lists for explicit, rewrite, and claim reads."""

    node = compiled.graph.node(node_id)
    if node.structural is StructuralDelta.CREATE:
        return MappingProxyType({})
    input_version = (
        compiled.versions[node_id]
        if node.structural is StructuralDelta.NONE
        else node.base
    )
    assert input_version is not None
    coordinates = {
        (owned.entity, owned.column) for owned in node.outputs if owned.rewrite
    }
    coordinates.update(_materialized_expand_coordinates(node))
    coordinates.update(
        (slice_.entity, column) for slice_ in node.inputs for column in slice_.columns
    )
    writers: dict[tuple[str, str], tuple[str, ...]] = {}
    for coordinate in sorted(coordinates):
        entity, column = coordinate
        writers[coordinate] = _writers_of(
            compiled,
            input_version,
            entity,
            column,
            exclude_node=node.id,
            receipts=receipts,
        )
    return MappingProxyType(writers)


def _expand_writer_coordinates(node: Node) -> frozenset[tuple[str, str]]:
    """Coordinates an EXPAND declares it may materialize."""

    if node.structural is not StructuralDelta.EXPAND:
        return frozenset()
    return frozenset((entity, column) for entity, column, _dtype in _expand_cells(node))


def _expand_declared_payload(node: Node) -> list[str]:
    """Canonical receipt spellings of every declared EXPAND coordinate."""

    return [
        f"{entity}.{column}"
        for entity, column in sorted(_expand_writer_coordinates(node))
    ]


def _parse_expand_declared(node: Node, raw: object) -> frozenset[tuple[str, str]]:
    """Validate the executor-authored EXPAND declaration attestation."""

    expected = tuple(_expand_declared_payload(node))
    if not isinstance(raw, list | tuple) or tuple(raw) != expected:
        raise ValueError(
            f"EXPAND node {node.id!r} expand_declared must exactly equal its "
            f"canonical declaration {expected!r}."
        )
    return _expand_writer_coordinates(node)


def _parse_expand_writes(
    node: Node, raw: object
) -> Mapping[tuple[str, str], tuple[str, ...]]:
    """Validate an executor-authored EXPAND coordinate/row-class record."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"EXPAND node {node.id!r} expand_writes must be a mapping.")
    declared = _expand_writer_coordinates(node)
    parsed: dict[tuple[str, str], tuple[str, ...]] = {}
    for spelling, raw_classes in raw.items():
        if not isinstance(spelling, str) or spelling.count(".") != 1:
            raise ValueError(
                f"EXPAND node {node.id!r} expand_writes coordinate {spelling!r} "
                "must be an 'entity.column' string."
            )
        entity, column = spelling.split(".")
        coordinate = (entity, column)
        if coordinate not in declared:
            raise ValueError(
                f"EXPAND node {node.id!r} expand_writes names undeclared "
                f"coordinate {spelling!r}."
            )
        if not isinstance(raw_classes, list | tuple) or not raw_classes:
            raise ValueError(
                f"EXPAND node {node.id!r} expand_writes {spelling!r} must name "
                "at least one row class."
            )
        classes = tuple(raw_classes)
        if any(not isinstance(value, str) for value in classes):
            raise ValueError(
                f"EXPAND node {node.id!r} expand_writes {spelling!r} row classes "
                "must be strings."
            )
        canonical = tuple(value for value in _EXPAND_WRITE_CLASSES if value in classes)
        if classes != canonical:
            raise ValueError(
                f"EXPAND node {node.id!r} expand_writes {spelling!r} row classes "
                f"must be unique and ordered as {_EXPAND_WRITE_CLASSES!r}."
            )
        parsed[coordinate] = classes
    return MappingProxyType(parsed)


def _validate_materialized_expand_outputs(
    compiled: CompiledGraph,
    node: Node,
    population: Population | None,
    receipts: Mapping[str, NodeReceipt],
) -> None:
    """Bind the no-Slice materialization bridge to its immediate EXPAND."""

    if "materialized_expand_outputs" not in node.params:
        return
    materialized = _materialized_expand_coordinates(node)
    if population is None:
        raise NodeRejected(
            f"Node {node.id!r} uses materialized_expand_outputs without an "
            "incumbent population; an immediate EXPAND population is required."
        )
    holder = compiled.graph.node(population.version)
    if holder.structural is not StructuralDelta.EXPAND:
        raise NodeRejected(
            f"Node {node.id!r} uses materialized_expand_outputs on population "
            f"version {population.version!r}, whose holder is {holder.structural.name}; "
            "an immediate EXPAND population is required."
        )
    holder_receipt = receipts.get(holder.id)
    if holder_receipt is None:  # compiled population ancestry should prevent this
        raise NodeRejected(
            f"Node {node.id!r} cannot validate materialized_expand_outputs: "
            f"EXPAND population version {holder.id!r} has no runtime receipt."
        )
    try:
        expand_declared = _parse_expand_declared(
            holder, holder_receipt.receipt.get("expand_declared")
        )
    except ValueError as error:  # executor-authored receipts cannot be malformed
        raise NodeRejected(
            f"Node {node.id!r} cannot validate materialized_expand_outputs for "
            f"EXPAND population version {holder.id!r}: {error}"
        ) from error
    for entity, column in sorted(materialized):
        if (entity, column) not in expand_declared:
            raise NodeRejected(
                f"Node {node.id!r} names materialized EXPAND output "
                f"{entity}.{column}, but EXPAND population version {holder.id!r} "
                "did not declare that coordinate."
            )


def _writers_of(
    compiled: CompiledGraph,
    version: str,
    entity: str,
    column: str,
    *,
    exclude_node: str | None = None,
    receipts: Mapping[str, NodeReceipt] | None = None,
) -> tuple[str, ...]:
    """All nodes that wrote rows of ``entity.column`` as seen from ``version``.

    The result is in causal order: the originating producer, EXPAND
    materializers, rewrites, and materialization claimants. Structural nodes
    that only carry the coordinate do not appear.
    """

    coordinate = (entity, column)
    newest_first: list[str] = []

    def add(writer_id: str) -> None:
        if writer_id != exclude_node and writer_id not in newest_first:
            newest_first.append(writer_id)

    while True:
        holder = compiled.graph.node(version)
        owner_id = compiled.owners.get((version, entity, column))
        if owner_id is not None:
            owner = compiled.graph.node(owner_id)
            output = next(
                owned
                for owned in owner.outputs
                if (owned.entity, owned.column) == coordinate
            )
            inherited = (
                output.rewrite
                or output.rows != ROWS_ALL
                or coordinate in _materialized_expand_coordinates(owner)
            )
            add(owner_id)
            if not inherited:
                break

        if _expand_wrote_rows(holder, coordinate, receipts):
            add(holder.id)
        if holder.structural is StructuralDelta.CREATE or holder.base is None:
            break
        version = holder.base
    return tuple(reversed(newest_first))


def _expand_wrote_rows(
    node: Node,
    coordinate: tuple[str, str],
    receipts: Mapping[str, NodeReceipt] | None,
) -> bool:
    """Whether this EXPAND actually wrote any row of a coordinate."""

    if coordinate not in _expand_writer_coordinates(node):
        return False
    if receipts is None:
        # Static preflight has no runtime receipt with which to refine the
        # declaration. Exact writer ids are checked during execution.
        return True
    node_receipt = receipts.get(node.id)
    if node_receipt is None:
        return False
    try:
        writes = _parse_expand_writes(node, node_receipt.receipt.get("expand_writes"))
    except ValueError as error:  # executor-authored receipts cannot be malformed
        raise NodeRejected(str(error)) from error
    return coordinate in writes


def _validate_series(
    node: Node,
    owned: Owned,
    series: pd.Series,
    population: Population,
) -> None:
    if not isinstance(series, pd.Series):
        raise NodeRejected(
            f"Node {node.id!r} output {owned.entity}.{owned.column} is "
            f"{type(series).__name__}, not a pandas Series."
        )
    if series.index.has_duplicates:
        raise NodeRejected(
            f"Node {node.id!r} output {owned.entity}.{owned.column} repeats ids."
        )
    expected = _owned_ids(population.frame, owned, node_id=node.id)
    actual = pd.Index(series.index)
    if len(actual) != len(expected) or set(actual.tolist()) != set(expected.tolist()):
        missing = expected.difference(actual).tolist()[:5]
        extra = actual.difference(expected).tolist()[:5]
        raise NodeRejected(
            f"Node {node.id!r} output {owned.entity}.{owned.column} ids do not "
            f"equal its owned ids; missing={missing}, extra={extra}."
        )
    if not _dtype_matches(series, owned.dtype):
        raise NodeRejected(
            f"Node {node.id!r} output {owned.entity}.{owned.column} has dtype "
            f"{series.dtype!s}, not declared {owned.dtype!r}."
        )
    if owned.ownership is Ownership.ABSENT and not series.isna().all():
        raise NodeRejected(
            f"Node {node.id!r} wrote a value into ABSENT-owned "
            f"{owned.entity}.{owned.column}."
        )


def _validate_filter_mask(node: Node, series: object, population: Population) -> None:
    if not isinstance(series, pd.Series):
        raise NodeRejected(f"FILTER node {node.id!r} mask is not a pandas Series.")
    if series.index.has_duplicates:
        raise NodeRejected(f"FILTER node {node.id!r} mask repeats person ids.")
    frame = population.frame
    person_entity = frame.schema.person_entity
    id_column = frame.schema.entity_id_column(person_entity)
    expected = pd.Index(
        frame.table(person_entity)[id_column].to_numpy(copy=True), name=id_column
    )
    actual = pd.Index(series.index)
    if len(actual) != len(expected) or set(actual.tolist()) != set(expected.tolist()):
        raise NodeRejected(
            f"FILTER node {node.id!r} mask ids do not equal the base person ids."
        )
    if not (
        series.dtype == np.dtype(np.bool_) or isinstance(series.dtype, pd.BooleanDtype)
    ):
        raise NodeRejected(
            f"FILTER node {node.id!r} mask has dtype {series.dtype!s}, not bool."
        )
    if series.isna().any():
        raise NodeRejected(f"FILTER node {node.id!r} mask contains nulls.")


def _validate_create(node: Node, frame: Frame) -> None:
    try:
        frame.revalidate()
    except (TypeError, ValueError) as error:
        raise NodeRejected(
            f"Node {node.id!r} returned an invalid Frame: {error}"
        ) from error
    # Amendment 15: every name the graph will spell as entity.column is dot-free.
    dotted_names = sorted(
        name
        for entity in frame.entities
        for name in (entity, *map(str, frame.table(entity).columns))
        if "." in name
    )
    if dotted_names:
        raise NodeRejected(
            f"CREATE node {node.id!r} returned a Frame with dotted names "
            f"{dotted_names[:5]}; entity and column names may not contain '.'."
        )
    expected_columns = {(owned.entity, owned.column) for owned in node.outputs}
    actual_columns = {
        (entity, str(column))
        for entity in frame.entities
        for column in frame.table(entity).columns
        if column not in _structural_columns(frame, entity)
    }
    if actual_columns != expected_columns:
        missing = sorted(expected_columns - actual_columns)
        extra = sorted(actual_columns - expected_columns)
        raise NodeRejected(
            f"CREATE node {node.id!r} data columns do not exactly equal its "
            f"declaration; missing={missing}, extra={extra}."
        )
    for owned in node.outputs:
        table = frame.table(owned.entity)
        if owned.column not in table:
            raise NodeRejected(
                f"CREATE node {node.id!r} did not load declared column "
                f"{owned.entity}.{owned.column}."
            )
        series = table[owned.column]
        if not _dtype_matches(series, owned.dtype):
            raise NodeRejected(
                f"CREATE node {node.id!r} loaded {owned.entity}.{owned.column} "
                f"as {series.dtype!s}, not {owned.dtype!r}."
            )
        if owned.ownership is Ownership.ABSENT and not series.isna().all():
            raise NodeRejected(
                f"CREATE node {node.id!r} loaded a value into ABSENT-owned "
                f"{owned.entity}.{owned.column}."
            )


def _validate_result(
    node: Node,
    kernel_capabilities: Capabilities,
    result: KernelResult,
    population: Population | None,
    *,
    cache_hit: bool = False,
) -> tuple[dict[str, object], dict[str, bytes]]:
    if not isinstance(result, KernelResult):
        raise NodeRejected(
            f"Node {node.id!r} kernel returned {type(result).__name__}, not KernelResult."
        )
    if not isinstance(result.columns, Mapping):
        raise NodeRejected(f"Node {node.id!r} result.columns is not a mapping.")
    if result.expand is not None and not isinstance(result.expand, Mapping):
        raise NodeRejected(f"Node {node.id!r} result.expand is not a mapping.")
    if not isinstance(result.artifacts, Mapping):
        raise NodeRejected(f"Node {node.id!r} result.artifacts is not a mapping.")
    if not isinstance(result.receipt, Mapping):
        raise NodeRejected(f"Node {node.id!r} result.receipt is not a mapping.")
    if result.strata is not None and not isinstance(result.strata, pd.Series):
        raise NodeRejected(f"Node {node.id!r} result.strata is not a Series.")
    if result.strata is not None and (
        cache_hit or node.structural is not StructuralDelta.EXPAND or not node.entrants
    ):
        raise NodeRejected(
            f"Node {node.id!r} returned entrant strata outside a fresh "
            "entrants=True EXPAND."
        )
    if kernel_capabilities.structural is not node.structural:
        raise NodeRejected(
            f"Node {node.id!r} declares structural={node.structural.value!r}, but "
            f"kernel capabilities declare {kernel_capabilities.structural.value!r}."
        )

    expected = {(owned.entity, owned.column): owned for owned in node.outputs}
    try:
        got = set(result.columns)
    except (TypeError, ValueError) as error:
        raise NodeRejected(
            f"Node {node.id!r} result.columns has malformed coordinates."
        ) from error
    if any(
        not isinstance(coordinate, tuple)
        or len(coordinate) != 2
        or any(not isinstance(part, str) for part in coordinate)
        for coordinate in got
    ):
        raise NodeRejected(
            f"Node {node.id!r} result.columns keys must be (entity, column) strings."
        )
    if node.structural is StructuralDelta.NONE:
        if got != set(expected):
            raise NodeRejected(
                f"Node {node.id!r} returned output keys {sorted(got)!r}, not exactly "
                f"its owned keys {sorted(expected)!r}."
            )
    elif node.structural is not StructuralDelta.EXPAND and got:
        raise NodeRejected(
            f"Structural node {node.id!r} returned column outputs; structural "
            "results use frame, keep, or weights."
        )

    if node.structural is StructuralDelta.FILTER:
        if population is None:  # pragma: no cover - compiler gives FILTER a base
            raise NodeRejected(f"FILTER node {node.id!r} has no base population.")
        _validate_filter_mask(node, result.keep, population)
    elif result.keep is not None:
        raise NodeRejected(f"Non-FILTER node {node.id!r} returned a keep mask.")

    if node.structural not in {StructuralDelta.CREATE, StructuralDelta.EXPAND} and (
        result.frame is not None
    ):
        raise NodeRejected(f"Node {node.id!r} returned a Frame outside CREATE/EXPAND.")
    if node.structural is StructuralDelta.CREATE and result.frame is None:
        raise NodeRejected(f"CREATE node {node.id!r} did not return a Frame.")
    if node.structural is StructuralDelta.EXPAND:
        if cache_hit and result.frame is None:
            raise NodeRejected(
                f"Cached EXPAND node {node.id!r} has no executor frame artifact."
            )
        if not cache_hit and result.frame is not None:
            raise NodeRejected(
                f"EXPAND node {node.id!r} returned a Frame; kernels return "
                "source lineage, cells, and weights, and the executor expands."
            )
        if not cache_hit and result.expand is None:
            raise NodeRejected(
                f"EXPAND node {node.id!r} returned no per-entity lineage."
            )
        if cache_hit and result.expand is not None:
            raise NodeRejected(
                f"Cached EXPAND node {node.id!r} returned kernel lineage instead "
                "of its executor frame artifact."
            )
    elif result.expand is not None:
        raise NodeRejected(f"Non-EXPAND node {node.id!r} returned expansion lineage.")
    if result.frame is not None and not isinstance(result.frame, Frame):
        raise NodeRejected(f"Node {node.id!r} result.frame is not a Frame.")
    if node.structural is StructuralDelta.CREATE:
        assert result.frame is not None
        _validate_create(node, result.frame)

    lineage_expand = node.structural is StructuralDelta.EXPAND
    if lineage_expand:
        weight_entity = node.params.get("expand_weight_entity")
        weight_kind = node.params.get("expand_weight_kind")
        if not isinstance(weight_entity, str) or not weight_entity:
            raise NodeRejected(
                f"EXPAND node {node.id!r} has no normative weight entity."
            )
        if not isinstance(weight_kind, str) or not weight_kind:
            raise NodeRejected(f"EXPAND node {node.id!r} has no normative weight kind.")
        if result.weights is None:
            raise NodeRejected(f"EXPAND node {node.id!r} returned no weights.")
    elif (node.weights is None) != (result.weights is None):
        state = "returned" if result.weights is not None else "did not return"
        raise NodeRejected(
            f"Node {node.id!r} {state} weights inconsistently with its declaration."
        )
    if result.weights is not None and not isinstance(result.weights, Weights):
        raise NodeRejected(f"Node {node.id!r} result.weights is not Weights.")

    if population is not None:
        for coordinate, owned in expected.items():
            _validate_series(node, owned, result.columns[coordinate], population)

    artifacts: dict[str, bytes] = {}
    for name, payload in result.artifacts.items():
        if not isinstance(name, str) or not name:
            raise NodeRejected(
                f"Node {node.id!r} artifact names must be non-empty strings."
            )
        if not isinstance(payload, bytes):
            raise NodeRejected(f"Node {node.id!r} artifact {name!r} is not bytes.")
        artifacts[name] = payload
    receipt = _normal_json_mapping(result.receipt, f"Node {node.id!r} receipt")
    if node.structural is StructuralDelta.EXPAND:
        if cache_hit:
            if not isinstance(receipt.get("expand"), dict):
                raise NodeRejected(
                    f"Cached EXPAND node {node.id!r} has no lineage receipt."
                )
        else:
            assert result.expand is not None
            try:
                receipt["expand"] = expand_lineage_receipt(result.expand)
                assert population is not None
                strata_receipt = entrant_strata_receipt(
                    population.frame, node, result.expand, result.strata
                )
            except (TypeError, ValueError) as error:
                raise NodeRejected(
                    f"EXPAND node {node.id!r} returned malformed lineage or "
                    f"entrant strata: {error}"
                ) from error
            receipt.pop("entrant_strata", None)
            if strata_receipt is not None:
                receipt["entrant_strata"] = strata_receipt
    if kernel_capabilities.role is KernelRole.GATE:
        outcome = receipt.get("outcome")
        if outcome not in GATE_OUTCOMES:
            raise NodeRejected(
                f"Gate node {node.id!r} returned outcome {outcome!r}; expected "
                f"one of {GATE_OUTCOMES!r}."
            )
    return receipt, artifacts


def _validate_entrant_materialization_contract(
    compiled: CompiledGraph,
    node: Node,
    population: Population | None,
    receipt: Mapping[str, object],
) -> None:
    """Require every entrant's carried data cells to have downstream claims."""

    if not node.entrants or population is None:
        return
    raw_expand = receipt.get("expand")
    if not isinstance(raw_expand, Mapping):
        return  # the ordinary EXPAND validation reports the malformed receipt
    entrant_entities: set[str] = set()
    for entity, entries in raw_expand.items():
        if not isinstance(entity, str) or not isinstance(entries, list):
            continue
        if any(
            isinstance(entry, list) and len(entry) == 2 and entry[1] is None
            for entry in entries
        ):
            entrant_entities.add(entity)

    frame = population.frame
    for entity in sorted(entrant_entities):
        if entity not in frame.entities:
            continue  # lineage validation supplies the node-naming rejection
        structural = set(_structural_columns(frame, entity))
        if (
            compiled.graph.mass_partition is not None
            and compiled.graph.mass_partition[0] == entity
        ):
            structural.add(compiled.graph.mass_partition[1])
        for column in frame.table(entity).columns:
            column = str(column)
            if column in structural:
                continue
            coordinate = (entity, column)
            claimant_id = compiled.owners.get((node.id, entity, column))
            if claimant_id is None:
                raise NodeRejected(
                    f"EXPAND node {node.id!r} entrant cell {entity}.{column} "
                    "has no materialized_expand_outputs ownership claim."
                )
            claimant = compiled.graph.node(claimant_id)
            claimed = claimant.params.get("materialized_expand_outputs", ())
            spelling = f"{entity}.{column}"
            output = next(
                (
                    owned
                    for owned in claimant.outputs
                    if (owned.entity, owned.column) == coordinate
                ),
                None,
            )
            if (
                not isinstance(claimed, tuple)
                or spelling not in claimed
                or output is None
                or output.rewrite
            ):
                raise NodeRejected(
                    f"EXPAND node {node.id!r} entrant cell {spelling} is not "
                    f"declared through node {claimant_id!r}'s "
                    "materialized_expand_outputs."
                )
            if output.rows != ROWS_ALL:
                raise NodeRejected(
                    f"EXPAND node {node.id!r} entrant cell {spelling} is claimed "
                    f"through masked rows {output.rows!r}; materialization bridge "
                    "claims must use rows='all'."
                )
            carried_dtype = _dtype_token(frame.table(entity)[column])
            if output.dtype != carried_dtype:
                raise NodeRejected(
                    f"EXPAND node {node.id!r} entrant cell {spelling} is claimed "
                    f"as {output.dtype!r}; its carried dtype is {carried_dtype!r}."
                )


def _create_population(node: Node, frame: Frame) -> Population:
    # Entity ids and membership columns are structural Frame columns rather
    # than declaration-owned data cells, but Population ownership is total
    # over the physical carrier.  The CREATE version supplies all of them.
    return Population.from_frame(frame, node.id)


def _validate_population_declaration(node: Node, population: Population | None) -> None:
    """Reject declarations that cross implicit population boundaries."""

    if node.weights is not None and node.structural is StructuralDelta.NONE:
        raise NodeRejected(
            f"Node {node.id!r} declares a weight transition without creating a "
            "structural population version."
        )
    if population is None or node.structural is not StructuralDelta.NONE:
        return
    schema = population.frame.schema
    structural = {
        (entity, schema.entity_id_column(entity)) for entity in schema.entities
    }
    structural.update(
        (schema.person_entity, schema.membership_column(group))
        for group in schema.group_entities
    )
    for owned in node.outputs:
        if (owned.entity, owned.column) in structural:
            raise NodeRejected(
                f"Node {node.id!r} cannot own structural column "
                f"{owned.entity}.{owned.column}; use a structural node."
            )


def _apply_result(
    node: Node,
    result: KernelResult,
    population: Population | None,
    *,
    cache_hit: bool = False,
    mass_partition: tuple[str, str] | None = None,
    rewrite_coordinates: frozenset[tuple[str, str]] = frozenset(),
) -> Population:
    if (
        mass_partition is not None
        and node.structural is StructuralDelta.NONE
        and any(
            (owned.entity, owned.column) == mass_partition for owned in node.outputs
        )
    ):
        entity, column = mass_partition
        raise NodeRejected(
            f"Node {node.id!r} cannot own mass partition {entity}.{column}; "
            "partition values are fixed by the structural population."
        )
    if node.structural is StructuralDelta.CREATE:
        assert result.frame is not None
        return _create_population(node, result.frame)
    assert population is not None
    if (
        cache_hit
        and node.structural is StructuralDelta.EXPAND
        and result.frame is not None
    ):
        try:
            return restore_cached_expand(
                population,
                node,
                result,
                mass_partition=mass_partition,
                rewrite_coordinates=rewrite_coordinates,
            )
        except (TypeError, ValueError) as error:
            raise NodeRejected(
                f"Node {node.id!r} cached EXPAND rejected: {error}"
            ) from error
    if node.structural is StructuralDelta.FILTER:
        person_entity = population.frame.schema.person_entity
        id_column = population.frame.schema.entity_id_column(person_entity)
        ids = pd.Index(
            population.frame.table(person_entity)[id_column].to_numpy(copy=True),
            name=id_column,
        )
        mask = (
            result.keep.reindex(ids).to_numpy(dtype=np.bool_, copy=True)  # type: ignore[union-attr]
        )
        result = KernelResult(
            frame=population.frame.select(mask),
            weights=result.weights,
            artifacts=result.artifacts,
            receipt=result.receipt,
        )
    # The frozen context has no base-Frame field.  A pure REWEIGHT kernel can
    # therefore return only its typed replacement weights; the executor binds
    # those weights to the incumbent structural frame here.
    if node.structural is StructuralDelta.REWEIGHT and result.frame is None:
        result = KernelResult(
            columns=result.columns,
            frame=population.frame,
            weights=result.weights,
            artifacts=result.artifacts,
            receipt=result.receipt,
        )
    try:
        return patch(
            population,
            node,
            result,
            mass_partition=mass_partition,
            rewrite_coordinates=rewrite_coordinates,
        )
    except NodeRejected:
        raise
    except (TypeError, ValueError) as error:
        raise NodeRejected(f"Node {node.id!r} patch rejected: {error}") from error


def _series_for_column(frame: Frame, entity: str, column: str) -> pd.Series:
    table = frame.table(entity)
    id_column = frame.schema.entity_id_column(entity)
    return pd.Series(
        table[column].array.copy(),
        index=pd.Index(table[id_column].to_numpy(copy=True), name=id_column),
        name=column,
        dtype=table[column].dtype,
    )


def _write_node(
    store: ContentStore,
    node: Node,
    *,
    key: str,
    kernel_impl_hash: str,
    capabilities: Capabilities,
    result: KernelResult,
    population: Population,
    receipt: Mapping[str, object],
    opaque_artifacts: Mapping[str, bytes],
    verify_existing: bool,
) -> tuple[dict[tuple[str, str], str], dict[str, object]]:
    columns: dict[tuple[str, str], tuple[pd.Series, str]] = {}
    if node.structural is StructuralDelta.NONE:
        declared = {(owned.entity, owned.column): owned for owned in node.outputs}
        for coordinate, series in result.columns.items():
            columns[coordinate] = (series, declared[coordinate].dtype)
    else:
        for entity in population.frame.entities:
            for column in population.frame.table(entity).columns:
                series = _series_for_column(population.frame, entity, column)
                columns[(entity, column)] = (series, _dtype_token(series))

    column_entries: list[dict[str, str]] = []
    manifest_artifacts: dict[tuple[str, str], str] = {}
    for (entity, column), (series, token) in sorted(columns.items()):
        output_key = artifact_key(key, entity, column)
        store.put_column(
            output_key,
            series,
            declared_dtype=token,
            entity_ids=series.index,
            node_key=key,
            verify_existing=verify_existing,
        )
        column_entries.append({"entity": entity, "column": column, "key": output_key})
        manifest_artifacts[(entity, column)] = output_key

    stored_frame_key: str | None = None
    if node.structural is not StructuralDelta.NONE:
        stored_frame_key = frame_key(key)
        store.put_frame(
            stored_frame_key,
            population.frame,
            node_key=key,
            verify_existing=verify_existing,
        )

    weight_entry: dict[str, str] | None = None
    if result.weights is not None:
        if node.weights is not None:
            entity = node.weights.entity
        elif node.structural is StructuralDelta.EXPAND and isinstance(
            node.params.get("expand_weight_entity"), str
        ):
            entity = str(node.params["expand_weight_entity"])
        else:  # defended by result validation
            raise NodeRejected(
                f"Node {node.id!r} returned weights without an entity contract."
            )
        id_column = population.frame.schema.entity_id_column(entity)
        ids = population.frame.table(entity)[id_column]
        weights_series = pd.Series(
            result.weights.values,
            index=pd.Index(ids.to_numpy(copy=True), name=id_column),
            dtype="float64",
            name="weights",
        )
        stored_weights_key = weights_key(key, entity)
        store.put_column(
            stored_weights_key,
            weights_series,
            declared_dtype="float64",
            entity_ids=weights_series.index,
            node_key=key,
            verify_existing=verify_existing,
        )
        weight_entry = {
            "entity": entity,
            "kind": result.weights.kind.value,
            "key": stored_weights_key,
        }

    opaque_entries: list[dict[str, str]] = []
    for name, payload in sorted(opaque_artifacts.items()):
        output_key = _opaque_artifact_key(key, name)
        store.put_bytes(
            output_key,
            payload,
            node_key=key,
            verify_existing=verify_existing,
        )
        opaque_entries.append({"name": name, "key": output_key})

    record: dict[str, object] = {
        "schema_version": 1,
        "node_id": node.id,
        "node_key": key,
        "kernel_ref": node.kernel,
        "kernel_impl_hash": kernel_impl_hash,
        "capabilities": _capabilities_projection(capabilities),
        "receipt": dict(receipt),
        "columns": column_entries,
        "frame_key": stored_frame_key,
        "weight": weight_entry,
        "opaque": opaque_entries,
    }
    store.put_json(
        _cache_record_key(key),
        record,
        node_key=key,
        verify_existing=verify_existing,
    )
    return manifest_artifacts, record


def _require_record_shape(
    raw: object,
    node: Node,
    *,
    key: str,
    kernel_impl_hash: str,
    capabilities: Capabilities,
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise StoreCorrupt(f"Cached receipt for node {node.id!r} is not an object.")
    required = {
        "schema_version",
        "node_id",
        "node_key",
        "kernel_ref",
        "kernel_impl_hash",
        "capabilities",
        "receipt",
        "columns",
        "frame_key",
        "weight",
        "opaque",
    }
    if set(raw) != required:
        raise StoreCorrupt(
            f"Cached receipt for node {node.id!r} has fields {sorted(raw)}, "
            f"not {sorted(required)}."
        )
    if raw["schema_version"] != 1:
        raise StoreUnavailable(
            f"Cached receipt for node {node.id!r} uses unsupported schema "
            f"{raw['schema_version']!r}."
        )
    expected = (node.id, key, node.kernel, kernel_impl_hash)
    actual = (
        raw["node_id"],
        raw["node_key"],
        raw["kernel_ref"],
        raw["kernel_impl_hash"],
    )
    if actual != expected:
        raise StoreCorrupt(
            f"Cached receipt identity for node {node.id!r} is {actual!r}, "
            f"not {expected!r}."
        )
    expected_capabilities = _capabilities_projection(capabilities)
    stored_capabilities = raw["capabilities"]
    if (
        isinstance(stored_capabilities, Mapping)
        and "tolerance" not in stored_capabilities
    ):
        raise StoreMiss(
            f"Cached receipt for node {node.id!r} has legacy_capabilities: "
            "the schema-v1 contract omits tolerance."
        )
    if raw["capabilities"] != expected_capabilities:
        raise StoreMiss(
            f"Cached receipt capabilities for node {node.id!r} disagree with "
            "the registered kernel contract."
        )
    if node.structural is StructuralDelta.EXPAND:
        raw_receipt = raw["receipt"]
        if not isinstance(raw_receipt, Mapping):
            raise StoreCorrupt(f"Cached node {node.id!r} receipt is malformed.")
        if "expand_writes" not in raw_receipt:
            raise StoreMiss(
                f"Cached EXPAND node {node.id!r} predates expand_writes provenance."
            )
        if "expand_declared" not in raw_receipt:
            raise StoreMiss(
                f"Cached EXPAND node {node.id!r} predates expand_declared provenance."
            )
        try:
            _parse_expand_declared(node, raw_receipt["expand_declared"])
            _parse_expand_writes(node, raw_receipt["expand_writes"])
        except ValueError as error:
            raise StoreCorrupt(
                f"Cached EXPAND node {node.id!r} has malformed EXPAND provenance."
            ) from error
    return raw


def _record_entries(
    record: Mapping[str, object], field: str
) -> list[dict[str, object]]:
    value = record[field]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise StoreCorrupt(f"Cached node receipt field {field!r} is malformed.")
    return value  # type: ignore[return-value]


def _load_record(
    store: ContentStore,
    node: Node,
    *,
    key: str,
    kernel_impl_hash: str,
    capabilities: Capabilities,
) -> dict[str, object]:
    raw = store.load_json(_cache_record_key(key))
    return _require_record_shape(
        raw,
        node,
        key=key,
        kernel_impl_hash=kernel_impl_hash,
        capabilities=capabilities,
    )


def _tolerance_writer_payload(
    writers: Mapping[tuple[str, str], tuple[str, ...]],
) -> dict[str, list[str]]:
    return {
        f"{entity}.{column}": list(writer_ids)
        for (entity, column), writer_ids in writers.items()
    }


def _require_tolerance_writer_receipt(
    node: Node,
    record: Mapping[str, object],
    writers: Mapping[tuple[str, str], tuple[str, ...]],
    *,
    exact: bool,
) -> None:
    """Reject cache receipts predating or disagreeing with writer provenance."""

    expected = _tolerance_writer_payload(writers)
    if not expected:
        return
    raw_receipt = record.get("receipt")
    if not isinstance(raw_receipt, Mapping):
        raise StoreCorrupt(f"Cached node {node.id!r} receipt is malformed.")
    raw_capabilities = raw_receipt.get("capabilities")
    actual = (
        raw_capabilities.get("tolerance_writers")
        if isinstance(raw_capabilities, Mapping)
        else None
    )
    expected_coordinates = set(expected)
    matches = (
        actual == expected
        if exact
        else (isinstance(actual, Mapping) and set(actual) == expected_coordinates)
    )
    if not matches:
        raise StoreMiss(
            f"Cached node {node.id!r} has stale tolerance_writers provenance."
        )

    evidence = raw_receipt.get("evidence")
    if isinstance(evidence, Mapping) and "tolerance" in evidence:
        evidence_writers = evidence.get("tolerance_writers")
        evidence_matches = (
            evidence_writers == expected
            if exact
            else (
                isinstance(evidence_writers, Mapping)
                and set(evidence_writers) == expected_coordinates
            )
        )
        if not evidence_matches:
            raise StoreMiss(
                f"Cached node {node.id!r} has stale evidence "
                "tolerance_writers provenance."
            )


def _preflight_record(store: ContentStore, record: Mapping[str, object]) -> None:
    for entry in _record_entries(record, "columns"):
        store.load_column(str(entry.get("key")))
    frame_artifact = record["frame_key"]
    if frame_artifact is not None:
        store.load_frame(str(frame_artifact))
    weight = record["weight"]
    if weight is not None:
        if not isinstance(weight, dict) or "key" not in weight:
            raise StoreCorrupt("Cached node weight entry is malformed.")
        store.load_column(str(weight["key"]))
    for entry in _record_entries(record, "opaque"):
        store.load_bytes(str(entry.get("key")))


def _load_cached_result(
    store: ContentStore,
    node: Node,
    population: Population | None,
    record: Mapping[str, object],
) -> tuple[KernelResult, dict[tuple[str, str], str]]:
    stored_columns: dict[tuple[str, str], pd.Series] = {}
    manifest_artifacts: dict[tuple[str, str], str] = {}
    for entry in _record_entries(record, "columns"):
        try:
            entity = str(entry["entity"])
            column = str(entry["column"])
            output_key = str(entry["key"])
        except KeyError as error:
            raise StoreCorrupt("Cached node column entry is malformed.") from error
        coordinate = (entity, column)
        if coordinate in stored_columns:
            raise StoreCorrupt(f"Cached node repeats column {entity}.{column}.")
        stored_columns[coordinate] = store.load_column(output_key)
        manifest_artifacts[coordinate] = output_key

    result_columns: dict[tuple[str, str], pd.Series] = {}
    if node.structural is StructuralDelta.NONE:
        for owned in node.outputs:
            coordinate = (owned.entity, owned.column)
            try:
                result_columns[coordinate] = stored_columns[coordinate]
            except KeyError as error:
                raise StoreMiss(
                    f"Cached node {node.id!r} is missing {owned.entity}.{owned.column}."
                ) from error

    loaded_frame: Frame | None = None
    frame_artifact = record["frame_key"]
    if frame_artifact is not None:
        loaded_frame = store.load_frame(str(frame_artifact))
    if node.structural is not StructuralDelta.NONE and loaded_frame is None:
        raise StoreMiss(f"Cached structural node {node.id!r} has no frame artifact.")

    loaded_weights: Weights | None = None
    weight = record["weight"]
    if weight is not None:
        if not isinstance(weight, dict):
            raise StoreCorrupt(f"Cached node {node.id!r} weight entry is malformed.")
        try:
            entity = str(weight["entity"])
            kind = WeightKind(str(weight["kind"]))
            weight_series = store.load_column(str(weight["key"]))
        except (KeyError, ValueError) as error:
            raise StoreCorrupt(
                f"Cached node {node.id!r} weight entry is malformed."
            ) from error
        expected_weight_entity = (
            node.weights.entity
            if node.weights is not None
            else node.params.get("expand_weight_entity")
            if node.structural is StructuralDelta.EXPAND
            else None
        )
        if entity != expected_weight_entity:
            raise StoreCorrupt(
                f"Cached node {node.id!r} carries undeclared weights for {entity!r}."
            )
        loaded_weights = Weights(
            values=weight_series.to_numpy(dtype=np.float64, copy=True), kind=kind
        )
    elif node.weights is not None or node.structural is StructuralDelta.EXPAND:
        raise StoreMiss(f"Cached node {node.id!r} is missing its weights artifact.")

    opaque: dict[str, bytes] = {}
    for entry in _record_entries(record, "opaque"):
        try:
            name = str(entry["name"])
            output_key = str(entry["key"])
        except KeyError as error:
            raise StoreCorrupt("Cached opaque artifact entry is malformed.") from error
        opaque[name] = store.load_bytes(output_key)

    raw_receipt = record["receipt"]
    if not isinstance(raw_receipt, dict):
        raise StoreCorrupt(f"Cached node {node.id!r} receipt is malformed.")

    # Reapply FILTER/REWEIGHT to the current base so graph mass checks and
    # ledgers are reconstructed on a hit.  Their stored final frame was loaded
    # above solely for content validation.
    result_frame = loaded_frame
    loaded_keep: pd.Series | None = None
    if (
        node.structural is StructuralDelta.FILTER
        and loaded_frame is not None
        and population is not None
    ):
        person_entity = population.frame.schema.person_entity
        id_column = population.frame.schema.entity_id_column(person_entity)
        base_ids = population.frame.table(person_entity)[id_column]
        kept_ids = set(loaded_frame.table(person_entity)[id_column].tolist())
        loaded_keep = pd.Series(
            base_ids.isin(kept_ids).to_numpy(dtype=np.bool_),
            index=pd.Index(base_ids.to_numpy(copy=True), name=id_column),
            dtype="bool",
        )
        result_frame = None
    if (
        node.structural is StructuralDelta.REWEIGHT
        and loaded_weights is not None
        and population is not None
    ):
        result_frame = None
    return (
        KernelResult(
            columns=MappingProxyType(result_columns),
            frame=result_frame,
            keep=loaded_keep,
            weights=loaded_weights,
            artifacts=MappingProxyType(opaque),
            receipt=MappingProxyType(raw_receipt),
        ),
        manifest_artifacts,
    )


def _source_paths_and_keys(
    compiled: CompiledGraph,
    sources: Mapping[str, Path],
    store: ContentStore,
) -> tuple[dict[str, Path], dict[str, str]]:
    declared = {source.name: source for source in compiled.graph.sources}
    used = {name for node in compiled.graph.nodes for name in node.sources}
    missing = sorted(used - sources.keys())
    if missing:
        raise FileNotFoundError(f"No source path supplied for {missing!r}.")
    unknown = sorted(sources.keys() - declared.keys())
    if unknown:
        raise ValueError(f"Source paths supplied for undeclared names {unknown!r}.")
    resolved: dict[str, Path] = {}
    identities: dict[str, str] = {}
    for name in sorted(used):
        path = Path(sources[name]).resolve(strict=True)
        # Codec availability is verified before any kernel can execute.  The
        # CREATE kernel remains the declared computation that invokes it.
        codec = declared[name].codec
        configured = store.codecs
        if configured is None:
            SOURCE_CODECS.get(codec)
        elif isinstance(configured, SourceCodecRegistry):
            configured.get(codec)
        elif isinstance(configured, Mapping):
            loader = configured.get(codec)
            if not callable(loader):
                raise StoreUnavailable(f"Source codec {codec!r} is not installed.")
        else:  # defended by ContentStore.__init__
            raise StoreUnavailable("ContentStore has an invalid codec registry.")
        resolved[name] = path
        identities[name] = source_content_key(name, path)
    return resolved, identities


def _all_node_keys(
    compiled: CompiledGraph,
    kernels: KernelRegistry,
    source_keys: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    keys: dict[str, str] = {}
    implementations: dict[str, str] = {}
    for node_id in compiled.order:
        node = compiled.graph.node(node_id)
        kernel = kernels.get(node.kernel)
        implementation = kernel.implementation_hash()
        if not isinstance(implementation, str) or not implementation:
            raise ValueError(
                f"Kernel {node.kernel!r} returned an invalid implementation hash."
            )
        implementations[node_id] = implementation
        keys[node_id] = node_key(
            compiled,
            node_id,
            keys,
            implementation,
            source_keys,
            kernel_capabilities=kernel.capabilities,
        )
    return keys, implementations


def _preflight_require(
    compiled: CompiledGraph,
    store: ContentStore,
    keys: Mapping[str, str],
    implementations: Mapping[str, str],
    kernels: KernelRegistry,
) -> None:
    missing: list[str] = []
    for node_id in compiled.order:
        node = compiled.graph.node(node_id)
        try:
            record = _load_record(
                store,
                node,
                key=keys[node_id],
                kernel_impl_hash=implementations[node_id],
                capabilities=kernels.get(node.kernel).capabilities,
            )
            _require_tolerance_writer_receipt(
                node,
                record,
                _input_writers(compiled, node_id),
                exact=False,
            )
            _preflight_record(store, record)
        except StoreMiss:
            missing.append(node_id)
    if missing:
        raise StoreMiss(
            "resume='require' found cache misses before execution: "
            + ", ".join(repr(node_id) for node_id in missing)
        )


def _preflight_expand_declarations(compiled: CompiledGraph) -> None:
    """Reject malformed runtime EXPAND conventions before keys or cache I/O."""

    for node_id in compiled.order:
        node = compiled.graph.node(node_id)
        if node.structural is not StructuralDelta.EXPAND:
            continue
        try:
            _expand_writer_coordinates(node)
        except (TypeError, ValueError) as error:
            raise NodeRejected(
                f"Node {node.id!r} expand_cells declaration rejected: {error}"
            ) from error


def run_graph(
    compiled: CompiledGraph,
    *,
    sources: Mapping[str, Path],
    store: ContentStore,
    kernels: KernelRegistry,
    resume: ResumePolicy = "auto",
    decisions: tuple[Decision, ...] = (),
) -> RunManifest:
    """Execute a compiled graph with content-addressed reuse and receipts."""

    if resume not in ("auto", "require", "forbid"):
        raise ValueError("resume must be 'auto', 'require', or 'forbid'.")
    normalized_decisions: list[Decision] = []
    for decision in decisions:
        if isinstance(decision, Decision):
            normalized_decisions.append(decision)
        elif isinstance(decision, Mapping):
            normalized_decisions.append(Decision.from_mapping(decision))
        else:
            raise TypeError("decisions must contain Decision records or mappings.")
    decisions = tuple(normalized_decisions)

    _preflight_expand_declarations(compiled)
    started_at = _now()
    source_paths, source_keys = _source_paths_and_keys(compiled, sources, store)
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    if resume == "require":
        _preflight_require(compiled, store, keys, implementations, kernels)

    populations: dict[str, Population] = {}
    receipts: dict[str, NodeReceipt] = {}
    for node_id in compiled.order:
        node_started = time.perf_counter()
        node = compiled.graph.node(node_id)
        key = keys[node_id]
        implementation = implementations[node_id]
        kernel = kernels.get(node.kernel)
        if kernel.capabilities.structural is not node.structural:
            raise NodeRejected(
                f"Node {node.id!r} structural declaration does not match kernel "
                "capabilities."
            )

        if node.structural is StructuralDelta.CREATE:
            incumbent: Population | None = None
        elif node.structural is StructuralDelta.NONE:
            incumbent = populations[compiled.versions[node_id]]
        else:
            assert node.base is not None
            incumbent = populations[node.base]
        _validate_population_declaration(node, incumbent)
        _validate_materialized_expand_outputs(compiled, node, incumbent, receipts)
        input_writers = _input_writers(compiled, node_id, receipts=receipts)
        input_tolerances = _input_tolerances(
            compiled, node_id, kernels, writers=input_writers
        )
        tolerance_writers = _tolerance_writer_payload(input_writers)

        hit = False
        replace_stale_record = False
        result: KernelResult | None = None
        record: dict[str, object] | None = None
        manifest_artifacts: dict[tuple[str, str], str] = {}
        if resume != "forbid":
            try:
                record = _load_record(
                    store,
                    node,
                    key=key,
                    kernel_impl_hash=implementation,
                    capabilities=kernel.capabilities,
                )
                try:
                    _require_tolerance_writer_receipt(
                        node, record, input_writers, exact=True
                    )
                except StoreMiss:
                    # This key predates the writer-provenance contract or was
                    # produced for different runtime entrant lineage.
                    replace_stale_record = True
                    raise
                result, manifest_artifacts = _load_cached_result(
                    store, node, incumbent, record
                )
                hit = True
            except StoreMiss:
                replace_stale_record = store.has(_cache_record_key(key))
                if resume == "require":  # defended by preflight; handles races
                    raise

        if result is None:
            context = _project_context(
                node,
                incumbent,
                key=key,
                sources=source_paths,
                tolerances=input_tolerances,
            )
            before = _context_digest(context)
            try:
                result = kernel.run(context)
            except Exception as error:
                if kernel.capabilities.role is KernelRole.GATE:
                    result = _failed_gate_result(node, incumbent, error)
                elif isinstance(error, StoreUnavailable):
                    raise
                else:
                    raise NodeRejected(
                        f"Node {node.id!r} kernel {node.kernel!r} failed: {error}"
                    ) from error
            after = _context_digest(context)
            if before != after:
                raise NodeRejected(f"Node {node.id!r} mutated its input context.")
            for name in node.sources:
                current = source_content_key(name, source_paths[name])
                if current != source_keys[name]:
                    raise NodeRejected(
                        f"Node {node.id!r} changed source {name!r} while running."
                    )

        normalized_receipt, opaque = _validate_result(
            node,
            kernel.capabilities,
            result,
            incumbent,
            cache_hit=hit,
        )
        _validate_entrant_materialization_contract(
            compiled, node, incumbent, normalized_receipt
        )
        if kernel.capabilities.role is KernelRole.RELEASE:
            derived_tier, gate_ids = _release_tier(compiled, node_id, receipts)
            _validate_release_tier(node, result, derived_tier)
            required_decisions = _required_decision_names(node)
            normalized_receipt["tier"] = derived_tier
            normalized_receipt["outcome"] = (
                "pass" if derived_tier == "certified" else "fail"
            )
            normalized_receipt["gate_ancestry"] = list(gate_ids)
            # Required names are derived from normative node params and live in
            # authenticated release provenance. The signed records themselves
            # remain top-level run provenance and never enter a node key.
            normalized_receipt["requires_decisions"] = list(required_decisions)
        receipt_capabilities = _capabilities_projection(kernel.capabilities)
        if tolerance_writers:
            receipt_capabilities["tolerance_writers"] = tolerance_writers
        normalized_receipt["capabilities"] = receipt_capabilities
        evidence = normalized_receipt.get("evidence")
        if (
            tolerance_writers
            and isinstance(evidence, Mapping)
            and "tolerance" in evidence
        ):
            normalized_receipt["evidence"] = {
                **evidence,
                "tolerance_writers": tolerance_writers,
            }
        expand_rewrites = _expand_rewrite_coordinates(compiled, node)
        updated = _apply_result(
            node,
            result,
            incumbent,
            cache_hit=hit,
            mass_partition=compiled.graph.mass_partition,
            rewrite_coordinates=expand_rewrites,
        )
        if node.structural is StructuralDelta.EXPAND:
            assert incumbent is not None
            normalized_receipt["expand_declared"] = _expand_declared_payload(node)
            try:
                authored_expand_writes = expand_writes_receipt(
                    incumbent.frame,
                    updated.frame,
                    node,
                    normalized_receipt,
                    rewrite_coordinates=expand_rewrites,
                )
            except (TypeError, ValueError) as error:
                raise NodeRejected(
                    f"Node {node.id!r} expand_writes receipt rejected: {error}"
                ) from error
            if hit:
                try:
                    stored_expand_writes = _parse_expand_writes(
                        node, normalized_receipt.get("expand_writes")
                    )
                except ValueError as error:  # defended by cached-record validation
                    raise StoreCorrupt(
                        f"Cached EXPAND node {node.id!r} has malformed "
                        "expand_writes provenance."
                    ) from error
                stored_payload = {
                    f"{entity}.{column}": list(classes)
                    for (entity, column), classes in stored_expand_writes.items()
                }
                if stored_payload != authored_expand_writes:
                    raise StoreCorrupt(
                        f"Cached EXPAND node {node.id!r} expand_writes provenance "
                        "disagrees with its materialized frame."
                    )
            normalized_receipt["expand_writes"] = authored_expand_writes
        if node.structural not in {
            StructuralDelta.NONE,
            StructuralDelta.CREATE,
        }:
            existing_mass = normalized_receipt.get("mass", {})
            if not isinstance(existing_mass, Mapping):  # defended by mass validation
                raise NodeRejected(
                    f"Node {node.id!r} receipt['mass'] is not a mapping."
                )
            try:
                authored_mass = mass_record_receipt(updated.mass_ledger[-1])
            except (TypeError, ValueError) as error:
                raise NodeRejected(
                    f"Node {node.id!r} mass receipt rejected: {error}"
                ) from error
            normalized_receipt["mass"] = {**existing_mass, **authored_mass}
        normalized_receipt.update(weight_cap_receipt(updated, node))
        cache_receipt = normalized_receipt
        run_receipt = dict(cache_receipt)
        if kernel.capabilities.role is KernelRole.RELEASE:
            run_receipt["outcome"] = _release_outcome(
                node, str(cache_receipt["tier"]), decisions
            )
        if node.structural is StructuralDelta.NONE:
            populations[compiled.versions[node_id]] = updated
        else:
            populations[node.id] = updated

        if not hit:
            manifest_artifacts, record = _write_node(
                store,
                node,
                key=key,
                kernel_impl_hash=implementation,
                capabilities=kernel.capabilities,
                result=result,
                population=updated,
                receipt=cache_receipt,
                opaque_artifacts=opaque,
                verify_existing=(resume != "forbid" and not replace_stale_record),
            )

        assert record is not None
        raw_frame_key = record["frame_key"]
        receipt_frame_key = None if raw_frame_key is None else str(raw_frame_key)
        raw_weight = record["weight"]
        if raw_weight is None:
            receipt_weight_key = None
        elif isinstance(raw_weight, dict) and isinstance(raw_weight.get("key"), str):
            receipt_weight_key = raw_weight["key"]
        else:  # generated records cannot reach this branch
            raise StoreCorrupt(f"Node {node.id!r} weight identity is malformed.")
        receipt_opaque: dict[str, str] = {}
        for entry in _record_entries(record, "opaque"):
            name = entry.get("name")
            artifact_identity = entry.get("key")
            if not isinstance(name, str) or not isinstance(artifact_identity, str):
                raise StoreCorrupt(
                    f"Node {node.id!r} opaque artifact identity is malformed."
                )
            receipt_opaque[name] = artifact_identity

        receipts[node_id] = NodeReceipt(
            key=key,
            hit=hit,
            seed=seed(key),
            kernel_ref=node.kernel,
            kernel_impl_hash=implementation,
            capabilities=kernel.capabilities,
            receipt=run_receipt,
            artifacts=MappingProxyType(dict(manifest_artifacts)),
            wall_time=time.perf_counter() - node_started,
            frame_key=receipt_frame_key,
            weight_key=receipt_weight_key,
            opaque_artifacts=MappingProxyType(receipt_opaque),
        )

    return RunManifest(
        country=compiled.graph.country,
        nodes=MappingProxyType(receipts),
        decisions=decisions,
        started_at=started_at,
        finished_at=_now(),
        host=socket.gethostname(),
        populations=MappingProxyType(
            {version: population.frame for version, population in populations.items()}
        ),
        mass_ledgers=MappingProxyType(
            {
                version: population.mass_ledger
                for version, population in populations.items()
            }
        ),
    )
