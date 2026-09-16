"""Graph execution: immutable projection, validation, patching, and reuse."""

from __future__ import annotations

import hashlib
import json
import pickle
import socket
import stat
import struct
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame import Frame, WeightKind, Weights

from . import keys as graph_keys
from .artifact_edges import scope_payload, typed_contracts, value_from_descriptor
from .availability import (
    EXECUTION_SCHEMA,
    execution_state,
    has_execution,
    unavailable_artifacts,
    validate_execution,
)
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
    ArtifactValue,
    Capabilities,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Numeric,
    NumericScope,
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
    return graph_keys.opaque_artifact_key(key, name)


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
            # A gate that declares typed outputs and raises produced none of
            # them. The executor, not the kernel, records that state so the
            # outputs' consumers are left unreached rather than the run
            # aborted (amendment 7 stays true for every legal node shape).
            **(
                {
                    "execution": {
                        "schema": EXECUTION_SCHEMA,
                        "state": "gate_exception",
                        "unavailable_artifacts": sorted(
                            output.name for output in node.artifact_outputs
                        ),
                    }
                }
                if node.artifact_outputs
                else {}
            ),
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


def _observer_snapshot(population: Population) -> Population:
    """Detach every observation from the executable population and its cache.

    Pandas deep copies retain object-cell referents and some immutable-by-API
    axis/category buffers. An in-memory, in-band round trip copies those too;
    only pandas objects from this admitted population are serialized here.
    No external pickle bytes are accepted, retained, or persisted. Frame and
    graph records are reconstructed explicitly to avoid reflective copy/pickle
    writes to their dataclass namespaces (which source identities may seal).

    When enabled, this costs one full detached population and a temporary
    serialized table buffer per callback. Observers may retain that snapshot;
    changing it immediately or later cannot change a kernel input or store write.
    """
    frame = population.frame
    tables = {name: frame.table(name) for name in frame.entities}
    tables.update({name: frame.link(name) for name in frame.links})
    tables, strata = pickle.loads(pickle.dumps((tables, frame.strata), protocol=5))

    def copied_record(record):
        return replace(
            record,
            **{
                field.name: deepcopy(getattr(record, field.name))
                for field in fields(record)
            },
        )

    snapshot = Frame(
        tables,
        replace(
            frame.schema,
            group_entities=deepcopy(frame.schema.group_entities),
            links=tuple(copied_record(link) for link in frame.schema.links),
        ),
        {
            entity: Weights(
                frame.weights_for(entity).values, frame.weights_for(entity).kind
            )
            for entity in frame.weighted_entities
        },
        strata,
        mass_log=tuple(copied_record(record) for record in frame.mass_log),
        metadata=frame.metadata,
    )
    return Population(
        snapshot,
        population.version,
        dict(population.owners),
        dict(population.weight_kind),
        mass_ledger=tuple(copied_record(record) for record in population.mass_ledger),
        design_weights=population.design_weights,
    )


def _update_scalar(digest: hashlib._Hash, value: object) -> None:
    if value is pd.NA:
        payload = b"pd.NA"
    elif value is pd.NaT:
        payload = b"pd.NaT"
    elif value is None:
        payload = b"None"
    elif type(value) is float:
        # Exact Python floats pack straight to the same eight IEEE-754 bytes
        # the one-element float64 array below produces, sign, payload and all,
        # without building that list and array. Subclasses and numpy floating
        # scalars keep the array path: only an exact float is short-circuited.
        payload = b"f" + struct.pack("=d", value)
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


def _framed(payloads: np.ndarray) -> bytes:
    """Frame equal-width payload rows exactly as ``_update_scalar`` frames one."""

    rows, width = payloads.shape
    prefix = np.frombuffer(
        np.full(rows, width, dtype="<u8").tobytes(), dtype=np.uint8
    ).reshape(rows, 8)
    return np.concatenate([prefix, payloads], axis=1).tobytes()


def _object_stream(series: pd.Series) -> bytes | None:
    """The bytes ``_update_array`` writes for this column's object projection.

    ``to_numpy(dtype=object)`` boxes a plain numpy column into one Python object
    per value, and ``_update_scalar`` then frames each box.  For the three kinds
    that dominate a survey frame the boxed value is always the same exact type,
    so the whole framed stream is a pure function of the column's bytes and can
    be built at C speed.  Byte-for-byte identical output is the contract; any
    column this cannot reproduce exactly returns None and takes the loop.
    """

    dtype = series.dtype
    if not isinstance(dtype, np.dtype) or dtype.kind not in "fiub":
        return None
    values = np.ascontiguousarray(series.to_numpy(copy=False))
    if values.ndim != 1 or values.dtype != dtype:
        return None
    rows = values.shape[0]
    if rows == 0:
        return b""
    if dtype.kind == "b":
        # bool -> b"b1" / b"b0"; np.bool_ boxes to bool, checked before int.
        payloads = np.empty((rows, 2), dtype=np.uint8)
        payloads[:, 0] = ord("b")
        payloads[:, 1] = np.where(values, ord("1"), ord("0"))
        return _framed(payloads)
    if dtype.kind == "f":
        # Every float width boxes to an exact Python float, which packs to the
        # same eight native-order IEEE-754 bytes as the float64 cast, NaN
        # payload, signed zero and infinities included.
        wide = np.ascontiguousarray(values, dtype="<f8")
        payloads = np.empty((rows, 9), dtype=np.uint8)
        payloads[:, 0] = ord("f")
        payloads[:, 1:] = np.frombuffer(wide.tobytes(), dtype=np.uint8).reshape(rows, 8)
        return _framed(payloads)
    # Signed and unsigned integers box to Python ints, whose decimal rendering
    # numpy reproduces exactly; payload width therefore varies per value.
    text = values.astype("S")
    width = text.dtype.itemsize
    raw = np.frombuffer(text.tobytes(), dtype=np.uint8).reshape(rows, width)
    filled = raw != 0  # 'S' pads the short renderings on the right with NUL
    lengths = filled.sum(axis=1).astype("<u8")
    block = np.empty((rows, 9 + width), dtype=np.uint8)
    block[:, :8] = np.frombuffer((lengths + 1).tobytes(), dtype=np.uint8).reshape(
        rows, 8
    )
    block[:, 8] = ord("i")
    block[:, 9:] = raw
    keep = np.ones((rows, 9 + width), dtype=bool)
    keep[:, 9:] = filled
    return block[keep].tobytes()


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
        stream = _object_stream(series)
        if stream is None:
            _update_array(digest, series.to_numpy(dtype=object, copy=False))
        else:
            # The header the object projection would have written: its dtype is
            # "object" and its shape is the column's own one-dimensional shape.
            digest.update(b"object\0")
            digest.update(f"({len(series)},)".encode("ascii"))
            digest.update(b"\0")
            digest.update(stream)
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
    for name, value in sorted(context.artifacts.items()):
        digest.update(
            canonical_json(
                (
                    name,
                    value.key,
                    value.producer_key,
                    value.type.name,
                    value.type.schema_version,
                    scope_payload(value.numerics),
                )
            )
        )
        digest.update(len(value.payload).to_bytes(8, "little"))
        digest.update(value.payload)
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
    numerics: Mapping[tuple[str, str], NumericScope],
    artifacts: Mapping[str, ArtifactValue] | None = None,
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
            artifacts={} if artifacts is None else artifacts,
            tolerances=tolerances,
            numerics=numerics,
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
        artifacts={} if artifacts is None else artifacts,
        tolerances=tolerances,
        numerics=numerics,
    )


_NUMERIC_RANK = {
    Numeric.BITWISE: 0,
    Numeric.PLATFORM_BITWISE: 1,
    Numeric.TOLERANCE_BOUND: 2,
}


def _input_numerics(
    compiled: CompiledGraph,
    node_id: str,
    kernels: KernelRegistry,
    *,
    writers: Mapping[tuple[str, str], tuple[str, ...]] | None = None,
) -> Mapping[tuple[str, str], NumericScope]:
    """Resolve each read coordinate to its loosest writer numeric scope."""

    writer_map = _input_writers(compiled, node_id) if writers is None else writers
    resolved: dict[tuple[str, str], NumericScope] = {}
    for coordinate, writer_ids in writer_map.items():
        capabilities = tuple(
            kernels.get(compiled.graph.node(writer_id).kernel).capabilities
            for writer_id in writer_ids
        )
        numeric = max(
            (capability.numeric for capability in capabilities),
            key=_NUMERIC_RANK.__getitem__,
            default=Numeric.BITWISE,
        )
        bounds = tuple(
            capability.tolerance
            for capability in capabilities
            if capability.numeric is Numeric.TOLERANCE_BOUND
        )
        tolerance = (
            None
            if not bounds
            else Tolerance(
                rtol=max(bound.rtol for bound in bounds if bound is not None),
                atol=max(bound.atol for bound in bounds if bound is not None),
                ulps=max(bound.ulps for bound in bounds if bound is not None),
            )
        )
        platform = (
            graph_keys.platform_fingerprint()
            if any(
                capability.numeric is Numeric.PLATFORM_BITWISE
                for capability in capabilities
            )
            else None
        )
        resolved[coordinate] = NumericScope(
            numeric=numeric,
            tolerance=tolerance,
            platform=platform,
        )
    return MappingProxyType(resolved)


def _input_tolerances(
    compiled: CompiledGraph,
    node_id: str,
    kernels: KernelRegistry,
    *,
    writers: Mapping[tuple[str, str], tuple[str, ...]] | None = None,
    numerics: Mapping[tuple[str, str], NumericScope] | None = None,
) -> Mapping[tuple[str, str], Tolerance | None]:
    """Project each input numeric scope to its legacy tolerance value."""

    scopes = (
        _input_numerics(compiled, node_id, kernels, writers=writers)
        if numerics is None
        else numerics
    )
    return MappingProxyType(
        {coordinate: scope.tolerance for coordinate, scope in scopes.items()}
    )


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
    unavailable = execution_state(result.receipt) == "gate_exception"
    for output in node.artifact_outputs:
        if output.name not in artifacts and not unavailable:
            # A cached record that lacks a declared artifact is a miss, but
            # that decision belongs to `_require_record_shape`, which runs
            # inside the miss-to-recompute fallback. By the time a restored
            # result reaches here the record has already been accepted, so a
            # still-absent artifact is corruption, not a miss.
            error = StoreCorrupt if cache_hit else NodeRejected
            raise error(
                f"Node {node.id!r} is missing declared artifact {output.name!r}."
            )
    receipt = _normal_json_mapping(result.receipt, f"Node {node.id!r} receipt")
    try:
        validate_execution(
            receipt,
            kernel_capabilities,
            {output.name: output for output in node.artifact_outputs},
            artifacts,
            has_products=bool(
                result.columns or result.frame is not None or result.weights is not None
            ),
        )
    except ValueError as error:
        raise NodeRejected(
            f"Node {node.id!r} execution evidence rejected: {error}"
        ) from error
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
    typed_artifacts: Mapping[str, object] | None = None,
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
        # Schema 2 only when the node declares typed artifacts, so a graph
        # that predates amendment 19 keeps its schema-1 records and its
        # store hits; schema 3 only for a record carrying an executor
        # execution state (gate exception or unreached).
        "schema_version": 3
        if execution_state(receipt)
        else 2
        if typed_artifacts
        else 1,
        **({"typed_artifacts": dict(typed_artifacts)} if typed_artifacts else {}),
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
    typed_artifacts: Mapping[str, object] | None = None,
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
    if typed_artifacts:
        required.add("typed_artifacts")
    if set(raw) != required:
        raise StoreCorrupt(
            f"Cached receipt for node {node.id!r} has fields {sorted(raw)}, "
            f"not {sorted(required)}."
        )
    raw_receipt = raw["receipt"]
    if not isinstance(raw_receipt, Mapping):
        raise StoreCorrupt(f"Cached node {node.id!r} receipt is malformed.")
    exceptional = execution_state(raw_receipt)
    if raw["schema_version"] != (3 if exceptional else 2 if typed_artifacts else 1):
        raise StoreUnavailable(
            f"Cached receipt for node {node.id!r} uses unsupported schema "
            f"{raw['schema_version']!r}."
        )
    try:
        validate_execution(
            raw_receipt,
            capabilities,
            (typed_artifacts or {}).get("outputs", {}),
            {
                entry.get("name"): entry.get("key")
                for entry in _record_entries(raw, "opaque")
            },
            has_products=bool(
                raw["columns"]
                or raw["frame_key"] is not None
                or raw["weight"] is not None
            ),
        )
    except ValueError as error:
        raise StoreCorrupt(
            f"Cached node {node.id!r} execution evidence rejected: {error}"
        ) from error
    if typed_artifacts:
        if raw.get("typed_artifacts") != dict(typed_artifacts):
            raise StoreCorrupt(
                f"Cached node {node.id!r} typed artifact contracts disagree with "
                "the graph."
            )
        opaque = _record_entries(raw, "opaque")
        names = [entry.get("name") for entry in opaque]
        if len(set(names)) != len(names):
            raise StoreCorrupt(f"Cached node {node.id!r} repeats an opaque artifact.")
        actual_outputs = {entry.get("name"): entry.get("key") for entry in opaque}
        for output in node.artifact_outputs:
            if exceptional:
                # The record proves the output was never produced; its
                # absence is that evidence, not a miss.
                continue
            if output.name not in actual_outputs:
                raise StoreMiss(
                    f"Cached node {node.id!r} is missing declared artifact "
                    f"{output.name!r}."
                )
            if actual_outputs[output.name] != _opaque_artifact_key(key, output.name):
                raise StoreCorrupt(
                    f"Cached node {node.id!r} artifact identity mismatch."
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
    if node.structural is StructuralDelta.EXPAND and exceptional != "unreached":
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
    typed_artifacts: Mapping[str, object] | None = None,
) -> dict[str, object]:
    raw = store.load_json(_cache_record_key(key))
    return _require_record_shape(
        raw,
        node,
        key=key,
        kernel_impl_hash=kernel_impl_hash,
        capabilities=capabilities,
        typed_artifacts=typed_artifacts,
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
    *,
    identities: _SourceIdentities | None = None,
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
    derived: dict[str, str] = {}
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
        derived[name] = (
            source_content_key(name, path)
            if identities is None
            else identities.key(name, path)
        )
    return resolved, derived


def _stat_identity(info: object) -> tuple[int, int, int, int, int]:
    """The five fields every source owner in this repository binds a file by."""

    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _followed_identity(path: Path) -> tuple[object, ...]:
    """The identity a member link resolves to, or why it resolves to nothing."""

    try:
        info = path.stat()
    except OSError as error:
        return ("absent", error.errno)
    return (stat.S_IFMT(info.st_mode), _stat_identity(info))


def _source_stat_signature(path: Path) -> tuple[object, ...]:
    """A read-free signature of everything ``source_content_key`` would read.

    A regular file contributes its own stat identity. A directory contributes
    its own, plus the relative name, file type and stat identity of every entry
    ``_directory_identity`` would walk -- the roster as well as the members,
    because the directory identity hashes relative names and a file added or
    removed moves it with no member's stat changing. Anything else contributes
    its stat identity alone, so an unexpected node type is never cached past a
    change.

    A member that is a symlink contributes its target's identity as well.
    ``_directory_identity`` selects members with ``is_file()`` and reads them
    with ``read_bytes()``, and both follow the link, so the target's bytes are
    inside the content key while the link's own five stat fields never move
    when those bytes change. Following it here keeps the same member selection
    on both sides, so such a change is a miss at the next node that declares
    the source rather than a refusal deferred to run end. A link that resolves
    to nothing contributes that fact and raises nothing, exactly as
    ``_directory_identity`` skips it.
    """

    info = path.lstat()
    if stat.S_ISREG(info.st_mode):
        return ("file", _stat_identity(info))
    if not stat.S_ISDIR(info.st_mode):
        return ("other", stat.S_IFMT(info.st_mode), _stat_identity(info))
    entries = []
    for candidate in sorted(path.rglob("*")):
        entry = candidate.lstat()
        member = (
            candidate.relative_to(path).as_posix(),
            stat.S_IFMT(entry.st_mode),
            _stat_identity(entry),
        )
        if stat.S_ISLNK(entry.st_mode):
            member = (*member, _followed_identity(candidate))
        entries.append(member)
    return ("dir", _stat_identity(info), tuple(entries))


class _SourceIdentities:
    """One run's source content keys, re-derived only when a source may have moved.

    ``source_content_key`` itself stays pure and uncached: it is the declared
    path-independent identity of a source's *content*, and three key and codec
    tests call it twice on one path with the bytes changed in between. This
    cache is the caller's, lives in one ``run_graph`` call, and never outlives
    it.

    A cached key is reused only when the path's stat signature is identical to
    the signature taken both immediately before and immediately after the read
    that produced the key. A derivation whose two signatures disagree -- the
    source moved while it was being read -- is not cached at all, so every
    consumer re-derives it exactly as it does today.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, tuple[tuple[object, ...], str]] = {}

    def key(self, name: str, path: Path) -> str:
        signature = _source_stat_signature(path)
        entry = self._entries.get(name)
        if entry is not None and entry[0] == signature:
            return entry[1]
        identity = source_content_key(name, path)
        if _source_stat_signature(path) == signature:
            self._entries[name] = (signature, identity)
        else:
            self._entries.pop(name, None)
        return identity

    def derive(self, name: str, path: Path) -> str:
        """Re-derive one key with the cache bypassed, and refresh the cache."""

        self._entries.pop(name, None)
        return self.key(name, path)


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


def _blocked_by(
    compiled: CompiledGraph,
    node: Node,
    keys: Mapping[str, str],
    receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """The predecessors whose outputs this node cannot have, by node key.

    An unreached causal parent propagates (its version, base, cells or bytes
    were never produced); a typed byte input whose producer recorded it as
    unavailable blocks its consumer. Every other predecessor is available,
    including a failed gate that produced its verdict column.
    """

    blocked = {
        parent: keys[parent]
        for parent in compiled.predecessors[node.id]
        if execution_state(receipts.get(parent, {})) == "unreached"
    }
    for binding in node.artifact_inputs:
        producer = compiled.graph.node(binding.producer)
        if binding.artifact in unavailable_artifacts(
            receipts.get(binding.producer, {}),
            {output.name: output for output in producer.artifact_outputs},
        ):
            blocked[binding.producer] = keys[binding.producer]
    return dict(sorted(blocked.items()))


def _unreached_node(
    compiled: CompiledGraph,
    node: Node,
    *,
    blockers: Mapping[str, str],
    receipts: Mapping[str, NodeReceipt],
    store: ContentStore,
    key: str,
    implementation: str,
    capabilities: Capabilities,
    typed: Mapping[str, object],
    resume: ResumePolicy,
) -> NodeReceipt:
    """Record a proven lack of inputs without running or inventing products.

    The receipt names each blocker by node key, so a cached unreached record
    is a hit only while the same inputs are unavailable for the same reason;
    the record holds no columns, frame, weights or bytes. A release that is
    unreached has a failed or unreached gate in its ancestry by construction
    and stays evidence-tier.
    """

    receipt: dict[str, object] = {
        "outcome": "unreached",
        "execution": {
            "schema": EXECUTION_SCHEMA,
            "state": "unreached",
            "blocked_by": dict(blockers),
        },
        "evidence": {"reason": "Required graph inputs are unavailable."},
        "capabilities": _capabilities_projection(capabilities),
    }
    if capabilities.role is KernelRole.RELEASE:
        tier, gate_ids = _release_tier(compiled, node.id, receipts)
        if tier != "evidence":
            raise NodeRejected(
                f"Release node {node.id!r} is unreached but no ancestral gate "
                "failed or was unreached."
            )
        receipt.update(
            tier=tier,
            gate_ancestry=list(gate_ids),
            requires_decisions=list(_required_decision_names(node)),
        )
    hit = False
    replace_stale_record = False
    if resume != "forbid":
        try:
            record = _load_record(
                store,
                node,
                key=key,
                kernel_impl_hash=implementation,
                capabilities=capabilities,
                typed_artifacts=typed,
            )
            if record["receipt"] != receipt:
                raise StoreCorrupt(
                    f"Cached node {node.id!r} blocked provenance disagrees with "
                    "its inputs."
                )
            hit = True
        except StoreMiss:
            replace_stale_record = store.has(_cache_record_key(key))
            if resume == "require":  # defended by preflight; handles races
                raise
    if not hit:
        record = {
            "schema_version": 3,
            **({"typed_artifacts": dict(typed)} if typed else {}),
            "node_id": node.id,
            "node_key": key,
            "kernel_ref": node.kernel,
            "kernel_impl_hash": implementation,
            "capabilities": _capabilities_projection(capabilities),
            "receipt": receipt,
            "columns": [],
            "frame_key": None,
            "weight": None,
            "opaque": [],
        }
        store.put_json(
            _cache_record_key(key),
            record,
            node_key=key,
            verify_existing=resume != "forbid" and not replace_stale_record,
        )
    return NodeReceipt(
        key=key,
        hit=hit,
        seed=seed(key),
        kernel_ref=node.kernel,
        kernel_impl_hash=implementation,
        capabilities=capabilities,
        receipt=receipt,
        typed_artifacts=typed,
    )


def _preflight_require(
    compiled: CompiledGraph,
    store: ContentStore,
    keys: Mapping[str, str],
    implementations: Mapping[str, str],
    kernels: KernelRegistry,
) -> None:
    missing: list[str] = []
    receipts: dict[str, Mapping[str, object]] = {}
    for node_id in compiled.order:
        node = compiled.graph.node(node_id)
        try:
            record = _load_record(
                store,
                node,
                key=keys[node_id],
                kernel_impl_hash=implementations[node_id],
                capabilities=kernels.get(node.kernel).capabilities,
                typed_artifacts=typed_contracts(compiled, node, keys, kernels),
            )
            if any(parent not in receipts for parent in compiled.predecessors[node_id]):
                # A missing parent already made this run a miss; without its
                # receipt the blockers below could not be derived honestly.
                missing.append(node_id)
                continue
            blockers = _blocked_by(compiled, node, keys, receipts)
            if blockers:
                if record["receipt"].get("execution") != {
                    "schema": EXECUTION_SCHEMA,
                    "state": "unreached",
                    "blocked_by": blockers,
                }:
                    raise StoreCorrupt(
                        f"Cached node {node_id!r} blocked provenance disagrees "
                        "with its inputs."
                    )
            elif execution_state(record["receipt"]) == "unreached":
                raise StoreCorrupt(
                    f"Cached node {node_id!r} has no unavailable input blocker."
                )
            else:
                _require_tolerance_writer_receipt(
                    node,
                    record,
                    _input_writers(compiled, node_id),
                    exact=False,
                )
            _preflight_record(store, record)
            receipts[node_id] = record["receipt"]
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
    _population_observer: Callable[[str, Population], None] | None = None,
    _verification_epoch: Mapping[str, object] | None = None,
) -> RunManifest:
    """Execute a compiled graph with content-addressed reuse and receipts.

    The private population observer exposes a detached snapshot of each node's
    admitted population, design anchors included, to an integrating verifier.
    It runs for cold execution and for restored cache hits alike, before the
    node is persisted; changes to the snapshot cannot alter execution or
    persistence, and an exception it raises refuses the run. It is never a
    kernel capability, enters no key or receipt, and an unreached node has no
    population to observe.

    The private verification-epoch record is a caller's own counts mapping --
    a country runtime that scopes source verification around the whole run
    hands in the record that scope yields. It is attached to the manifest
    unchanged, as a live view rather than a copy, so counts the caller
    finalises when its scope closes are present by the time the caller holds
    the manifest. It enters no key, no receipt and no cache record.
    """

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
    # One run's source identities. The run-start pass populates the cache as it
    # reads, so the per-node check below is a stat walk rather than a second
    # full read; the run-end pass bypasses the cache entirely.
    source_identities = _SourceIdentities()
    source_paths, source_keys = _source_paths_and_keys(
        compiled, sources, store, identities=source_identities
    )
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    contracts = {
        node_id: typed_contracts(compiled, compiled.graph.node(node_id), keys, kernels)
        for node_id in compiled.order
    }
    if resume == "require":
        _preflight_require(compiled, store, keys, implementations, kernels)

    populations: dict[str, Population] = {}
    receipts: dict[str, NodeReceipt] = {}
    receipt_payloads: dict[str, Mapping[str, object]] = {}
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

        blockers = _blocked_by(compiled, node, keys, receipt_payloads)
        if blockers:
            receipts[node_id] = _unreached_node(
                compiled,
                node,
                blockers=blockers,
                receipts=receipts,
                store=store,
                key=key,
                implementation=implementation,
                capabilities=kernel.capabilities,
                typed=contracts[node_id],
                resume=resume,
            )
            receipt_payloads[node_id] = receipts[node_id].receipt
            continue

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
        input_numerics = _input_numerics(
            compiled, node_id, kernels, writers=input_writers
        )
        input_tolerances = _input_tolerances(
            compiled,
            node_id,
            kernels,
            writers=input_writers,
            numerics=input_numerics,
        )
        tolerance_writers = _tolerance_writer_payload(input_writers)

        typed = contracts[node_id]
        # Authenticate every declared byte edge against the producer's receipt
        # and its descriptor now, without reading a payload: identity is all
        # this check needs, and a node that hits its cached record never runs a
        # kernel, so its inputs' bytes would be read for nothing (a fitted
        # model is not small).
        for binding in node.artifact_inputs:
            entry = typed["inputs"][binding.name]
            producer_receipt = receipts[binding.producer]
            if producer_receipt.opaque_artifacts.get(binding.artifact) != entry["key"]:
                # Generated receipts cannot reach this branch: a producer that
                # hit its record had this identity checked by
                # `_require_record_shape`, and one that ran got it from
                # `_write_node` under the same derivation. It stands so the
                # consumer's read is guarded by its own check rather than by
                # the producer's, should those two paths ever diverge.
                raise StoreCorrupt(
                    f"Node {node.id!r} artifact producer receipt disagrees with its "
                    "declaration."
                )
            value_from_descriptor(b"", entry)

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
                    typed_artifacts=typed,
                )
                if execution_state(record["receipt"]) == "unreached":
                    raise StoreCorrupt(
                        f"Cached node {node_id!r} has no unavailable input blocker."
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
            artifact_values: dict[str, ArtifactValue] = {
                binding.name: value_from_descriptor(
                    store.load_bytes(typed["inputs"][binding.name]["key"]),
                    typed["inputs"][binding.name],
                )
                for binding in node.artifact_inputs
            }
            context = _project_context(
                node,
                incumbent,
                key=key,
                sources=source_paths,
                tolerances=input_tolerances,
                numerics=input_numerics,
                artifacts=artifact_values,
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
            else:
                # The execution state is executor evidence about what a
                # kernel could not produce; a kernel that returns one is a
                # contract rejection, not an exception raised while a gate
                # computed its verdict.
                if (
                    isinstance(result, KernelResult)
                    and isinstance(result.receipt, Mapping)
                    and has_execution(result.receipt)
                ):
                    raise NodeRejected(
                        f"Node {node.id!r} kernel receipt may not author executor "
                        "execution metadata."
                    )
            after = _context_digest(context)
            if before != after:
                raise NodeRejected(f"Node {node.id!r} mutated its input context.")
            for name in node.sources:
                # A stat signature that still matches the one taken across this
                # source's own read reuses that read's key; anything that moved
                # is re-derived in full here, so the refusal still precedes
                # every _write_node this run performs.
                current = source_identities.key(name, source_paths[name])
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

        if _population_observer is not None:
            _population_observer(node_id, _observer_snapshot(updated))

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
                typed_artifacts=typed,
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
            typed_artifacts=typed,
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
        receipt_payloads[node_id] = receipts[node_id].receipt

    # Every source is re-derived in full, cache bypassed, before any caller
    # receives a manifest. This closes the two cases a stat signature cannot
    # decide -- a rewrite that leaves all five stat fields identical, and a
    # change made during a node that declares no source at all, which the
    # per-node check has never seen -- and it is written inline rather than
    # through _source_paths_and_keys because a build test counts calls to that
    # exact code object.
    final_identities = {}
    moved = []
    for name in sorted(source_paths):
        identity = source_identities.derive(name, source_paths[name])
        final_identities[name] = identity
        if identity != source_keys[name]:
            moved.append(name)
    if moved:
        raise NodeRejected(
            "Run changed source "
            + ", ".join(repr(name) for name in moved)
            + " while executing."
        )

    return RunManifest(
        country=compiled.graph.country,
        nodes=MappingProxyType(receipts),
        source_identities=MappingProxyType(final_identities),
        verification_epoch=({} if _verification_epoch is None else _verification_epoch),
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
