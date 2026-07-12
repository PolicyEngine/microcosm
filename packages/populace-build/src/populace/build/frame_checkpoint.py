"""Low-copy, deterministic HDF checkpoints for :class:`populace.frame.Frame`.

The normal PolicyEngine dataset writers copy whole entity tables and flatten
typed weights into table columns.  Build checkpoints have a different job:
they must preserve the kernel object exactly while keeping write-time memory
bounded.  This module therefore writes one pandas Series at a time in HDF
fixed format.  Schema, column order and dtypes, typed weights, strata, and the
mass-change log live in one canonical metadata record.

Fixed-format object columns use Python pickles internally.  Checkpoints are
therefore trusted local build artifacts and must not be loaded from untrusted
sources.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.frame import (
    EntitySchema,
    Frame,
    LinkSpec,
    MassChangeRecord,
    WeightKind,
    Weights,
)

__all__ = [
    "FRAME_CHECKPOINT_SCHEMA_VERSION",
    "LoadedFrameCheckpoint",
    "load_frame_checkpoint",
    "write_frame_checkpoint",
]

FRAME_CHECKPOINT_SCHEMA_VERSION = 1

_ARTIFACT_KIND = "populace_frame_checkpoint"
_ROOT = "_populace_frame_checkpoint"
_METADATA_KEY = f"{_ROOT}/metadata"

_SUPPORTED_OBJECT_INFERRED_DTYPES = frozenset(
    {
        "boolean",
        "bytes",
        "empty",
        "floating",
        "integer",
        "mixed-integer-float",
        "string",
    }
)


@dataclass(frozen=True)
class LoadedFrameCheckpoint:
    """A restored frame and the caller-owned JSON metadata stored with it."""

    frame: Frame
    metadata: dict[str, object]


def write_frame_checkpoint(
    path: str | Path,
    frame: Frame,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write ``frame`` as a deterministic, low-copy checkpoint.

    Entity and link tables are serialized one column at a time.  At most one
    normalized column is copied by this function; it never passes a complete
    DataFrame to pandas' HDF serializer.

    Args:
        path: Destination HDF5 path.
        frame: Kernel frame to checkpoint.
        metadata: Optional JSON-compatible build state to return on load.

    Returns:
        The destination path.

    Raises:
        TypeError: If ``frame`` or ``metadata`` has the wrong type, or a table
            contains a dtype/object value family that fixed HDF cannot preserve
            under the deterministic checkpoint contract.
        ValueError: If metadata is not finite JSON, or table columns are not
            unique strings.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    external_metadata = _normalize_external_metadata(metadata)
    checkpoint_metadata = _checkpoint_metadata(frame, external_metadata)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with pd.HDFStore(temporary_path, mode="w") as store:
            _put_fixed_series(
                store,
                _METADATA_KEY,
                pd.Series(
                    [_canonical_json(checkpoint_metadata)],
                    name="metadata_json",
                    dtype="str",
                ),
            )
            for table_index, (_name, table) in enumerate(_frame_tables(frame)):
                for column_index, column in enumerate(table.columns):
                    key = _table_column_key(table_index, column_index)
                    _put_fixed_series(store, key, _normalized_series(table[column]))

            for weight_index, entity in enumerate(frame.weighted_entities):
                table_index = frame.table(entity).index
                values = frame.weights_for(entity).values
                series = pd.Series(
                    values,
                    index=table_index,
                    name=f"{entity}_weight",
                    dtype=np.float64,
                    copy=False,
                )
                _put_fixed_series(store, _weight_key(weight_index), series)

            _put_fixed_series(store, _strata_key(), _normalized_series(frame.strata))
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def load_frame_checkpoint(path: str | Path) -> LoadedFrameCheckpoint:
    """Load a checkpoint written by :func:`write_frame_checkpoint`.

    The loader reconstructs the declared entity schema and link tables rather
    than assuming a country schema, then restores every weight kind and
    :class:`MassChangeRecord` before constructing the validated ``Frame``.
    """

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Frame checkpoint not found: {input_path}.")

    with pd.HDFStore(input_path, mode="r") as store:
        stored_metadata = _read_metadata(store, input_path)
        schema = _schema_from_metadata(stored_metadata)
        table_specs = _require_list(stored_metadata, "tables")
        tables: dict[str, pd.DataFrame] = {}
        for table_index, raw_spec in enumerate(table_specs):
            spec = _require_dict(raw_spec, f"tables[{table_index}]")
            name = _require_string(spec, "name", label=f"tables[{table_index}]")
            column_specs = _require_list(spec, "columns")
            columns: list[pd.Series] = []
            expected_index: pd.Index | None = None
            for column_index, raw_column in enumerate(column_specs):
                column_spec = _require_dict(
                    raw_column,
                    f"tables[{table_index}].columns[{column_index}]",
                )
                column = _require_string(
                    column_spec,
                    "name",
                    label=f"tables[{table_index}].columns[{column_index}]",
                )
                dtype = _require_string(
                    column_spec,
                    "dtype",
                    label=f"tables[{table_index}].columns[{column_index}]",
                )
                key = _table_column_key(table_index, column_index)
                series = _read_fixed_series(store, key, input_path)
                series = _restore_declared_dtype(series, dtype, label=f"{name}.{column}")
                series.name = column
                if expected_index is None:
                    expected_index = series.index
                elif not series.index.equals(expected_index):
                    raise ValueError(
                        f"Frame checkpoint {input_path} has misaligned columns in "
                        f"table {name!r}; {column!r} does not share the first "
                        "column's index."
                    )
                columns.append(series)
            if not columns:
                raise ValueError(
                    f"Frame checkpoint {input_path} table {name!r} has no columns."
                )
            tables[name] = pd.concat(columns, axis=1, copy=False)

        weight_specs = _require_list(stored_metadata, "weights")
        weights: dict[str, Weights] = {}
        for weight_index, raw_spec in enumerate(weight_specs):
            spec = _require_dict(raw_spec, f"weights[{weight_index}]")
            entity = _require_string(
                spec,
                "entity",
                label=f"weights[{weight_index}]",
            )
            raw_kind = _require_string(
                spec,
                "kind",
                label=f"weights[{weight_index}]",
            )
            try:
                kind = WeightKind(raw_kind)
            except ValueError as exc:
                raise ValueError(
                    f"Frame checkpoint {input_path} has unknown weight kind "
                    f"{raw_kind!r} for {entity!r}."
                ) from exc
            series = _read_fixed_series(store, _weight_key(weight_index), input_path)
            if entity not in tables:
                raise ValueError(
                    f"Frame checkpoint {input_path} stores weights for unknown "
                    f"table {entity!r}."
                )
            if not series.index.equals(tables[entity].index):
                raise ValueError(
                    f"Frame checkpoint {input_path} weights for {entity!r} are "
                    "not aligned to its table index."
                )
            weights[entity] = Weights(series.to_numpy(dtype=np.float64), kind)

        strata_spec = _require_dict(stored_metadata.get("strata"), "strata")
        strata_dtype = _require_string(strata_spec, "dtype", label="strata")
        strata = _read_fixed_series(store, _strata_key(), input_path)
        strata = _restore_declared_dtype(strata, strata_dtype, label="strata")
        strata.name = "stratum"

    mass_log = _mass_log_from_metadata(stored_metadata, input_path)
    try:
        frame = Frame(
            tables,
            schema,
            weights,
            strata,
            mass_log=mass_log,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frame checkpoint {input_path} violates Frame invariants: {exc}"
        ) from exc

    external = stored_metadata.get("external_metadata", {})
    if not isinstance(external, dict):
        raise ValueError(
            f"Frame checkpoint {input_path} external_metadata must be an object."
        )
    return LoadedFrameCheckpoint(frame=frame, metadata=dict(external))


def _checkpoint_metadata(
    frame: Frame,
    external_metadata: dict[str, object],
) -> dict[str, object]:
    table_specs: list[dict[str, object]] = []
    for name, table in _frame_tables(frame):
        if any(not isinstance(column, str) for column in table.columns):
            raise TypeError(
                f"Frame checkpoint table {name!r} must use string column names."
            )
        if table.columns.has_duplicates:
            raise ValueError(
                f"Frame checkpoint table {name!r} has duplicate column names."
            )
        columns: list[dict[str, str]] = []
        for column in table.columns:
            _validate_supported_series(table[column], label=f"{name}.{column}")
            columns.append({"name": column, "dtype": str(table[column].dtype)})
        table_specs.append(
            {
                "name": name,
                "role": "entity" if name in frame.entities else "link",
                "columns": columns,
            }
        )

    _validate_supported_series(frame.strata, label="strata")
    schema = frame.schema
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": FRAME_CHECKPOINT_SCHEMA_VERSION,
        "schema": {
            "person_entity": schema.person_entity,
            "group_entities": list(schema.group_entities),
            "links": [
                {
                    "name": link.name,
                    "left_entity": link.left_entity,
                    "right_entity": link.right_entity,
                }
                for link in schema.links
            ],
        },
        "tables": table_specs,
        "weights": [
            {
                "entity": entity,
                "kind": frame.weights_for(entity).kind.value,
            }
            for entity in frame.weighted_entities
        ],
        "strata": {"dtype": str(frame.strata.dtype)},
        "mass_log": [_mass_change_payload(record) for record in frame.mass_log],
        "external_metadata": external_metadata,
    }


def _frame_tables(frame: Frame) -> tuple[tuple[str, pd.DataFrame], ...]:
    entities = tuple((entity, frame.table(entity)) for entity in frame.entities)
    links = tuple((link, frame.link(link)) for link in frame.links)
    return (*entities, *links)


def _validate_supported_series(series: pd.Series, *, label: str) -> None:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        raise TypeError(
            f"Frame checkpoint does not support categorical dtype for {label!r}; "
            "convert it to a stable string or numeric dtype first."
        )
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise TypeError(
            f"Frame checkpoint does not support timezone-aware dtype for {label!r}; "
            "convert it to timezone-naive datetime64 first."
        )
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) and not isinstance(
        dtype, pd.StringDtype
    ):
        raise TypeError(
            f"Frame checkpoint does not support extension dtype {dtype!s} for "
            f"{label!r}; use a NumPy or pandas string dtype."
        )
    if pd.api.types.is_object_dtype(dtype):
        inferred = pd.api.types.infer_dtype(series, skipna=True)
        if inferred not in _SUPPORTED_OBJECT_INFERRED_DTYPES:
            raise TypeError(
                f"Frame checkpoint does not support object value family "
                f"{inferred!r} for {label!r}; use a homogeneous scalar bool, "
                "string, bytes, integer, or floating column."
            )
        return
    if any(
        predicate(dtype)
        for predicate in (
            pd.api.types.is_bool_dtype,
            pd.api.types.is_integer_dtype,
            pd.api.types.is_float_dtype,
            pd.api.types.is_string_dtype,
            pd.api.types.is_datetime64_dtype,
            pd.api.types.is_timedelta64_dtype,
        )
    ):
        return
    raise TypeError(
        f"Frame checkpoint does not support dtype {dtype!s} for {label!r}."
    )


def _normalized_series(series: pd.Series) -> pd.Series:
    """Return a deterministic Series, copying at most this one column."""

    _validate_supported_series(series, label=str(series.name))
    if pd.api.types.is_object_dtype(series.dtype) and series.isna().any():
        values = series.to_numpy(dtype=object, copy=True)
        values[pd.isna(values)] = np.nan
        return pd.Series(
            values,
            index=series.index,
            name=series.name,
            dtype=object,
            copy=False,
        )
    if pd.api.types.is_float_dtype(series.dtype) and series.isna().any():
        values = series.to_numpy(copy=True)
        values[pd.isna(values)] = np.nan
        return pd.Series(
            values,
            index=series.index,
            name=series.name,
            dtype=series.dtype,
            copy=False,
        )
    return series


def _put_fixed_series(store: pd.HDFStore, key: str, series: pd.Series) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        store.put(key, series, format="fixed", track_times=False)


def _read_fixed_series(
    store: pd.HDFStore,
    key: str,
    path: Path,
) -> pd.Series:
    if f"/{key}" not in store.keys():
        raise ValueError(f"Frame checkpoint {path} is missing HDF key {key!r}.")
    value = store[key]
    if not isinstance(value, pd.Series):
        raise ValueError(
            f"Frame checkpoint {path} HDF key {key!r} must contain a Series."
        )
    return value


def _read_metadata(store: pd.HDFStore, path: Path) -> dict[str, Any]:
    metadata_series = _read_fixed_series(store, _METADATA_KEY, path)
    if len(metadata_series) != 1:
        raise ValueError(
            f"Frame checkpoint {path} metadata must contain exactly one value."
        )
    try:
        metadata = json.loads(str(metadata_series.iloc[0]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Frame checkpoint {path} metadata is not valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Frame checkpoint {path} metadata must be a JSON object.")
    if metadata.get("artifact_kind") != _ARTIFACT_KIND:
        raise ValueError(
            f"Frame checkpoint {path} has unknown artifact kind "
            f"{metadata.get('artifact_kind')!r}."
        )
    version = metadata.get("schema_version")
    if version != FRAME_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Frame checkpoint {path} schema version is {version!r}; expected "
            f"{FRAME_CHECKPOINT_SCHEMA_VERSION}."
        )
    return metadata


def _schema_from_metadata(metadata: Mapping[str, Any]) -> EntitySchema:
    raw_schema = _require_dict(metadata.get("schema"), "schema")
    person_entity = _require_string(raw_schema, "person_entity", label="schema")
    raw_groups = _require_list(raw_schema, "group_entities")
    if any(not isinstance(group, str) or not group for group in raw_groups):
        raise ValueError("Frame checkpoint schema.group_entities must be strings.")
    raw_links = _require_list(raw_schema, "links")
    links: list[LinkSpec] = []
    for index, raw_link in enumerate(raw_links):
        link = _require_dict(raw_link, f"schema.links[{index}]")
        links.append(
            LinkSpec(
                name=_require_string(link, "name", label=f"schema.links[{index}]"),
                left_entity=_require_string(
                    link,
                    "left_entity",
                    label=f"schema.links[{index}]",
                ),
                right_entity=_require_string(
                    link,
                    "right_entity",
                    label=f"schema.links[{index}]",
                ),
            )
        )
    try:
        return EntitySchema(
            person_entity=person_entity,
            group_entities=tuple(raw_groups),
            links=tuple(links),
        )
    except ValueError as exc:
        raise ValueError(f"Frame checkpoint carries an invalid schema: {exc}") from exc


def _restore_declared_dtype(
    series: pd.Series,
    dtype: str,
    *,
    label: str,
) -> pd.Series:
    if str(series.dtype) == dtype:
        return series
    try:
        restored = series.astype(object if dtype == "object" else dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frame checkpoint cannot restore declared dtype {dtype!r} for "
            f"{label!r}; stored dtype is {series.dtype!s}."
        ) from exc
    if str(restored.dtype) != dtype:
        raise ValueError(
            f"Frame checkpoint restored {label!r} as {restored.dtype!s}, not "
            f"the declared dtype {dtype!r}."
        )
    return restored


def _mass_log_from_metadata(
    metadata: Mapping[str, Any],
    path: Path,
) -> tuple[MassChangeRecord, ...]:
    entries = _require_list(metadata, "mass_log")
    records: list[MassChangeRecord] = []
    for index, raw_entry in enumerate(entries):
        entry = _require_dict(raw_entry, f"mass_log[{index}]")
        try:
            declared_factor = entry.get("declared_factor")
            records.append(
                MassChangeRecord(
                    entity=str(entry["entity"]),
                    old_total=float(entry["old_total"]),
                    new_total=float(entry["new_total"]),
                    declared_factor=(
                        None if declared_factor is None else float(declared_factor)
                    ),
                    reason=str(entry["reason"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Frame checkpoint {path} mass_log[{index}] is malformed."
            ) from exc
    return tuple(records)


def _mass_change_payload(record: MassChangeRecord) -> dict[str, object]:
    return {
        "entity": record.entity,
        "old_total": record.old_total,
        "new_total": record.new_total,
        "declared_factor": record.declared_factor,
        "reason": record.reason,
    }


def _normalize_external_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"metadata must be a mapping, got {type(metadata).__name__}."
        )
    try:
        normalized = json.loads(_canonical_json(dict(metadata)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Frame checkpoint metadata must be finite and JSON-compatible."
        ) from exc
    if not isinstance(normalized, dict):  # pragma: no cover - dict input guarantees this
        raise ValueError("Frame checkpoint metadata must normalize to an object.")
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _table_column_key(table_index: int, column_index: int) -> str:
    return f"{_ROOT}/tables/t{table_index:05d}/columns/c{column_index:05d}"


def _weight_key(weight_index: int) -> str:
    return f"{_ROOT}/weights/w{weight_index:05d}"


def _strata_key() -> str:
    return f"{_ROOT}/strata"


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Frame checkpoint {label} must be an object.")
    return value


def _require_list(container: Mapping[str, Any], key: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Frame checkpoint {key} must be a list.")
    return value


def _require_string(
    container: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Frame checkpoint {label}.{key} must be a string.")
    return value
