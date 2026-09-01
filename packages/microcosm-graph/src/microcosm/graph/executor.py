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
from .codecs import SOURCE_CODECS
from .decl import (
    ROWS_ALL,
    CompiledGraph,
    Node,
    Owned,
    Ownership,
    StructuralDelta,
)
from .kernel import Capabilities, KernelContext, KernelRegistry, KernelResult
from .keys import artifact_key, frame_key, node_key, seed, source_content_key
from .manifest import Decision, NodeReceipt, RunManifest
from .population import Population, patch
from .store import (
    ContentStore,
    ResumePolicy,
    StoreCorrupt,
    StoreMiss,
    StoreUnavailable,
)

__all__ = ["NodeRejected", "run_graph"]


class NodeRejected(RuntimeError):  # noqa: N818 - acceptance contract spelling
    """A kernel result violated its node declaration or execution boundary."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _cache_record_key(key: str) -> str:
    return sha256_domain("node-receipt", canonical_json((key,)))


def _opaque_artifact_key(key: str, name: str) -> str:
    return sha256_domain("node-artifact", canonical_json((key, name)))


def _capabilities_payload(capabilities: Capabilities) -> dict[str, object]:
    return {
        "determinism": capabilities.determinism.value,
        "numeric": capabilities.numeric.value,
        "seed_source": capabilities.seed_source.value,
        "structural": capabilities.structural.value,
        "consumes_se": capabilities.consumes_se,
        "dependencies": list(capabilities.dependencies),
    }


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


def _project_context(
    node: Node,
    population: Population | None,
    *,
    key: str,
    sources: Mapping[str, Path],
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
        )

    frame = population.frame
    slices: dict[str, list[object]] = {}
    for slice_ in node.inputs:
        slices.setdefault(slice_.entity, []).append(slice_)

    tables: dict[str, pd.DataFrame] = {}
    entity_masks: dict[str, np.ndarray] = {}
    for entity in sorted(slices):
        table = frame.table(entity)
        entity_slices = slices[entity]
        row_specs = {slice_.rows for slice_ in entity_slices}
        if len(row_specs) > 1:
            raise NodeRejected(
                f"Node {node.id!r} declares incompatible row masks for entity "
                f"{entity!r}; KernelContext has one table per entity."
            )
        columns = _structural_columns(frame, entity)
        for slice_ in entity_slices:
            columns.extend(slice_.columns)
        columns = list(dict.fromkeys(columns))
        if row_specs and next(iter(row_specs)) != ROWS_ALL:
            row_column = str(next(iter(row_specs)))
            mask = _mask_values(table, row_column, node_id=node.id)
        else:
            mask = np.ones(len(table), dtype=np.bool_)
        entity_masks[entity] = mask
        tables[entity] = _freeze_frame(table.loc[mask, columns])

    input_entities = {slice_.entity for slice_ in node.inputs}
    weights: dict[str, Weights] = {}
    for entity in sorted(input_entities):
        effective = frame.resolve_weights(entity)
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
    )


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


def _validate_create(node: Node, frame: Frame) -> None:
    try:
        frame.revalidate()
    except (TypeError, ValueError) as error:
        raise NodeRejected(
            f"Node {node.id!r} returned an invalid Frame: {error}"
        ) from error
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
) -> tuple[dict[str, object], dict[str, bytes]]:
    if not isinstance(result, KernelResult):
        raise NodeRejected(
            f"Node {node.id!r} kernel returned {type(result).__name__}, not KernelResult."
        )
    if not isinstance(result.columns, Mapping):
        raise NodeRejected(f"Node {node.id!r} result.columns is not a mapping.")
    if not isinstance(result.artifacts, Mapping):
        raise NodeRejected(f"Node {node.id!r} result.artifacts is not a mapping.")
    if not isinstance(result.receipt, Mapping):
        raise NodeRejected(f"Node {node.id!r} result.receipt is not a mapping.")
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
    elif got:
        # Structural output lives in the returned Frame.  Accepting a second
        # column channel would make the authoritative value ambiguous.
        raise NodeRejected(
            f"Structural node {node.id!r} returned column outputs as well as a Frame."
        )

    if node.structural is StructuralDelta.NONE and result.frame is not None:
        raise NodeRejected(f"Non-structural node {node.id!r} returned a Frame.")
    if (
        node.structural
        in {
            StructuralDelta.CREATE,
            StructuralDelta.FILTER,
            StructuralDelta.EXPAND,
        }
        and result.frame is None
    ):
        raise NodeRejected(f"Structural node {node.id!r} did not return a Frame.")
    if result.frame is not None and not isinstance(result.frame, Frame):
        raise NodeRejected(f"Node {node.id!r} result.frame is not a Frame.")
    if node.structural is StructuralDelta.CREATE:
        assert result.frame is not None
        _validate_create(node, result.frame)

    if (node.weights is None) != (result.weights is None):
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
    return _normal_json_mapping(result.receipt, f"Node {node.id!r} receipt"), artifacts


def _create_population(node: Node, frame: Frame) -> Population:
    # Entity ids and membership columns are structural Frame columns rather
    # than declaration-owned data cells, but Population ownership is total
    # over the physical carrier.  The CREATE version supplies all of them.
    return Population.from_frame(frame, node.id)


def _apply_result(
    node: Node,
    result: KernelResult,
    population: Population | None,
) -> Population:
    if node.structural is StructuralDelta.CREATE:
        assert result.frame is not None
        return _create_population(node, result.frame)
    assert population is not None
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
        return patch(population, node, result)
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
        assert node.weights is not None
        entity = node.weights.entity
        id_column = population.frame.schema.entity_id_column(entity)
        ids = population.frame.table(entity)[id_column]
        weights_series = pd.Series(
            result.weights.values,
            index=pd.Index(ids.to_numpy(copy=True), name=id_column),
            dtype="float64",
            name="weights",
        )
        weights_key = artifact_key(key, entity, "__weights__")
        store.put_column(
            weights_key,
            weights_series,
            declared_dtype="float64",
            entity_ids=weights_series.index,
            node_key=key,
            verify_existing=verify_existing,
        )
        weight_entry = {
            "entity": entity,
            "kind": result.weights.kind.value,
            "key": weights_key,
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
        "capabilities": _capabilities_payload(capabilities),
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
    raw: object, node: Node, *, key: str, kernel_impl_hash: str
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
) -> dict[str, object]:
    raw = store.load_json(_cache_record_key(key))
    return _require_record_shape(raw, node, key=key, kernel_impl_hash=kernel_impl_hash)


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
        if node.weights is None or entity != node.weights.entity:
            raise StoreCorrupt(
                f"Cached node {node.id!r} carries undeclared weights for {entity!r}."
            )
        loaded_weights = Weights(
            values=weight_series.to_numpy(dtype=np.float64, copy=True), kind=kind
        )
    elif node.weights is not None:
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

    # Reapply REWEIGHT to the current base so graph mass checks and ledgers are
    # reconstructed on a hit.  The stored final frame was loaded above solely
    # for content validation.
    result_frame = loaded_frame
    if (
        node.structural is StructuralDelta.REWEIGHT
        and loaded_weights is not None
        and population is not None
    ):
        result_frame = population.frame
    return (
        KernelResult(
            columns=MappingProxyType(result_columns),
            frame=result_frame,
            weights=loaded_weights,
            artifacts=MappingProxyType(opaque),
            receipt=MappingProxyType(raw_receipt),
        ),
        manifest_artifacts,
    )


def _source_paths_and_keys(
    compiled: CompiledGraph, sources: Mapping[str, Path]
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
        SOURCE_CODECS.get(declared[name].codec)
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
        )
    return keys, implementations


def _preflight_require(
    compiled: CompiledGraph,
    store: ContentStore,
    keys: Mapping[str, str],
    implementations: Mapping[str, str],
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
            )
            _preflight_record(store, record)
        except StoreMiss:
            missing.append(node_id)
    if missing:
        raise StoreMiss(
            "resume='require' found cache misses before execution: "
            + ", ".join(repr(node_id) for node_id in missing)
        )


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
    if any(not isinstance(decision, Decision) for decision in decisions):
        raise TypeError("decisions must contain Decision records.")

    started_at = _now()
    source_paths, source_keys = _source_paths_and_keys(compiled, sources)
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    if resume == "require":
        _preflight_require(compiled, store, keys, implementations)

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

        hit = False
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
                )
                result, manifest_artifacts = _load_cached_result(
                    store, node, incumbent, record
                )
                hit = True
            except StoreMiss:
                if resume == "require":  # defended by preflight; handles races
                    raise

        if result is None:
            context = _project_context(node, incumbent, key=key, sources=source_paths)
            before = _context_digest(context)
            try:
                result = kernel.run(context)
            except StoreUnavailable:
                raise
            except Exception as error:
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
            node, kernel.capabilities, result, incumbent
        )
        updated = _apply_result(node, result, incumbent)
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
                receipt=normalized_receipt,
                opaque_artifacts=opaque,
                verify_existing=resume != "forbid",
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
            receipt=normalized_receipt,
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
