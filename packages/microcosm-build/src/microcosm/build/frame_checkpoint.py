"""Low-copy, deterministic HDF5 checkpoints for :class:`microcosm.frame.Frame`.

Build checkpoints must preserve the in-memory kernel object without making a
full-frame serialization copy.  Tables are therefore written one column at a
time, and each table index is stored only once.  Every HDF5 object disables
creation timestamps so equivalent frames produce byte-identical files even
when separate stage processes write them at different wall-clock times.

Object columns use a small deterministic scalar codec.  Checkpoints are local
build artifacts and must not be loaded from untrusted sources.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import (
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

FRAME_CHECKPOINT_SCHEMA_VERSION = 3
_LEGACY_FRAME_CHECKPOINT_SCHEMA_VERSION = 2
_SUPPORTED_FRAME_CHECKPOINT_SCHEMA_VERSIONS = frozenset(
    {_LEGACY_FRAME_CHECKPOINT_SCHEMA_VERSION, FRAME_CHECKPOINT_SCHEMA_VERSION}
)

_ARTIFACT_KIND = "populace_frame_checkpoint"
_ROOT = "_populace_frame_checkpoint"
_METADATA_DATASET = "metadata_json"

_ENCODING_NUMPY = "numpy"
_ENCODING_DATETIME = "datetime64"
_ENCODING_TIMEDELTA = "timedelta64"
_ENCODING_OBJECT = "object_scalars_v1"
_ENCODING_NULLABLE_BOOLEAN = "nullable_boolean_v1"

_TAG_NONE = 0
_TAG_PD_NA = 1
_TAG_PD_NAT = 2
_TAG_FALSE = 3
_TAG_TRUE = 4
_TAG_INTEGER = 5
_TAG_FLOAT64 = 6
_TAG_STRING = 7
_TAG_BYTES = 8

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
    """A restored frame and caller-owned JSON metadata stored alongside it."""

    frame: Frame
    metadata: dict[str, object]


def write_frame_checkpoint(
    path: str | Path,
    frame: Frame,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write a deterministic checkpoint, one column at a time."""

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    external_metadata = _normalize_external_metadata(metadata)
    checkpoint_metadata = _checkpoint_metadata(frame, external_metadata)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    h5py = _h5py()
    try:
        with h5py.File(temporary_path, mode="w") as h5:
            root = h5.create_group(_ROOT, track_times=False)
            _write_bytes_dataset(
                root,
                _METADATA_DATASET,
                _canonical_json(checkpoint_metadata).encode("utf-8"),
            )
            tables_group = root.create_group("tables", track_times=False)
            for table_index, (_name, table) in enumerate(_frame_tables(frame)):
                table_group = tables_group.create_group(
                    f"t{table_index:05d}", track_times=False
                )
                table_spec = checkpoint_metadata["tables"][table_index]
                _write_index(table_group, table.index, table_spec["index"])
                columns_group = table_group.create_group("columns", track_times=False)
                for column_index, column in enumerate(table.columns):
                    column_group = columns_group.create_group(
                        f"c{column_index:05d}", track_times=False
                    )
                    _write_series(
                        column_group,
                        table[column],
                        table_spec["columns"][column_index],
                    )

            weights_group = root.create_group("weights", track_times=False)
            for weight_index, entity in enumerate(frame.weighted_entities):
                _write_numpy_dataset(
                    weights_group,
                    f"w{weight_index:05d}",
                    frame.weights_for(entity).values,
                )

            strata_group = root.create_group("strata", track_times=False)
            _write_series(
                strata_group,
                frame.strata,
                checkpoint_metadata["strata"],
            )
            h5.flush()
        os.replace(temporary_path, output_path)
        _fsync_parent_directory(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _fsync_parent_directory(path: Path) -> None:
    """Persist a completed atomic rename in its containing directory."""

    # O_DIRECTORY is not universal. A read-only directory descriptor is the
    # supported POSIX fallback; failures to open or fsync still propagate.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_frame_checkpoint(
    path: str | Path,
    *,
    frame_metadata: Mapping[str, object] | None = None,
) -> LoadedFrameCheckpoint:
    """Load a checkpoint and reconstruct its schema, data, and audit metadata.

    ``frame_metadata`` lets a caller restore separately bound
    :class:`~microcosm.frame.Frame` metadata during the loader's one Frame
    construction. This avoids a second full-table copy for large checkpoints.
    The caller remains responsible for validating that metadata against its
    own artifact identity or embedded external metadata.
    """

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Frame checkpoint not found: {input_path}.")
    if frame_metadata is not None and not isinstance(frame_metadata, Mapping):
        raise TypeError("frame_metadata must be a mapping when provided.")

    h5py = _h5py()
    with h5py.File(input_path, mode="r") as h5:
        root = _require_h5_group(h5, _ROOT, input_path)
        stored_metadata = _read_metadata(root, input_path)
        schema = _schema_from_metadata(stored_metadata)
        table_specs = _require_list(stored_metadata, "tables")
        tables_group = _require_h5_group(root, "tables", input_path)
        tables: dict[str, pd.DataFrame] = {}
        for table_index, raw_spec in enumerate(table_specs):
            spec = _require_dict(raw_spec, f"tables[{table_index}]")
            name = _require_string(spec, "name", label=f"tables[{table_index}]")
            table_group = _require_h5_group(
                tables_group, f"t{table_index:05d}", input_path
            )
            index_spec = _require_dict(
                spec.get("index"), f"tables[{table_index}].index"
            )
            index = _read_index(table_group, index_spec, input_path)
            column_specs = _require_list(spec, "columns")
            columns_group = _require_h5_group(table_group, "columns", input_path)
            data: dict[str, pd.Series] = {}
            for column_index, raw_column in enumerate(column_specs):
                column_spec = _require_dict(
                    raw_column, f"tables[{table_index}].columns[{column_index}]"
                )
                column = _require_string(
                    column_spec,
                    "name",
                    label=f"tables[{table_index}].columns[{column_index}]",
                )
                column_group = _require_h5_group(
                    columns_group, f"c{column_index:05d}", input_path
                )
                series = _read_series(
                    column_group,
                    column_spec,
                    input_path,
                    label=f"{name}.{column}",
                )
                if len(series) != len(index):
                    raise ValueError(
                        f"Frame checkpoint {input_path} column {name}.{column} has "
                        f"{len(series)} rows; its table index has {len(index)}."
                    )
                # Keep the declared dtype attached to the Series.  Passing an
                # object ndarray lets pandas' constructor infer ``str`` under
                # its future string-inference mode, which would rewrite the
                # checkpoint schema on the next write.
                series.index = index
                data[column] = series
            table = pd.DataFrame(data, index=index, copy=False)
            table.columns.name = _axis_name_from_payload(
                spec.get("columns_name"),
                label=f"tables[{table_index}].columns_name",
            )
            tables[name] = table

        weight_specs = _require_list(stored_metadata, "weights")
        weights_group = _require_h5_group(root, "weights", input_path)
        weights: dict[str, Weights] = {}
        for weight_index, raw_spec in enumerate(weight_specs):
            spec = _require_dict(raw_spec, f"weights[{weight_index}]")
            entity = _require_string(spec, "entity", label=f"weights[{weight_index}]")
            raw_kind = _require_string(spec, "kind", label=f"weights[{weight_index}]")
            try:
                kind = WeightKind(raw_kind)
            except ValueError as exc:
                raise ValueError(
                    f"Frame checkpoint {input_path} has unknown weight kind "
                    f"{raw_kind!r} for {entity!r}."
                ) from exc
            if entity not in tables:
                raise ValueError(
                    f"Frame checkpoint {input_path} stores weights for unknown "
                    f"table {entity!r}."
                )
            values = _read_numpy_dataset(
                weights_group, f"w{weight_index:05d}", input_path
            )
            if values.ndim != 1 or len(values) != len(tables[entity]):
                raise ValueError(
                    f"Frame checkpoint {input_path} weights for {entity!r} do not "
                    "align to its table."
                )
            weights[entity] = Weights(values.astype(np.float64, copy=False), kind)

        expected_household_kind = stored_metadata.get("household_weight_kind")
        actual_household_kind = (
            weights["household"].kind.value if "household" in weights else None
        )
        if expected_household_kind != actual_household_kind:
            raise ValueError(
                f"Frame checkpoint {input_path} household_weight_kind is "
                f"{expected_household_kind!r}, but typed weights carry "
                f"{actual_household_kind!r}."
            )

        strata_spec = _require_dict(stored_metadata.get("strata"), "strata")
        strata_group = _require_h5_group(root, "strata", input_path)
        strata = _read_series(
            strata_group, strata_spec, input_path, label="strata"
        ).set_axis(tables[schema.person_entity].index)
        strata.name = "stratum"

    external = stored_metadata.get("external_metadata", {})
    if not isinstance(external, dict):
        raise ValueError(
            f"Frame checkpoint {input_path} external_metadata must be an object."
        )
    mass_log = _mass_log_from_metadata(stored_metadata, input_path)
    try:
        frame = Frame(
            tables,
            schema,
            weights,
            strata,
            mass_log=mass_log,
            metadata=frame_metadata,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frame checkpoint {input_path} violates Frame invariants: {exc}"
        ) from exc

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
        columns = [
            {"name": column, **_series_spec(table[column], label=f"{name}.{column}")}
            for column in table.columns
        ]
        table_specs.append(
            {
                "name": name,
                "role": "entity" if name in frame.entities else "link",
                "index": _index_spec(table.index, label=f"{name}.index"),
                "columns_name": _axis_name_payload(
                    table.columns.name, label=f"{name}.columns.name"
                ),
                "columns": columns,
            }
        )

    strata_spec = _series_spec(frame.strata, label="strata")
    uses_nullable_boolean = any(
        column.get("encoding") == _ENCODING_NULLABLE_BOOLEAN
        for table in table_specs
        for column in table["columns"]
    ) or any(
        table["index"].get("encoding") == _ENCODING_NULLABLE_BOOLEAN
        for table in table_specs
    )
    uses_nullable_boolean = uses_nullable_boolean or (
        strata_spec.get("encoding") == _ENCODING_NULLABLE_BOOLEAN
    )
    schema = frame.schema
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": (
            FRAME_CHECKPOINT_SCHEMA_VERSION
            if uses_nullable_boolean
            else _LEGACY_FRAME_CHECKPOINT_SCHEMA_VERSION
        ),
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
        "household_weight_kind": (
            frame.weights_for("household").kind.value
            if "household" in frame.weighted_entities
            else None
        ),
        "strata": strata_spec,
        "mass_log": [_mass_change_payload(record) for record in frame.mass_log],
        "external_metadata": external_metadata,
    }


def _frame_tables(frame: Frame) -> tuple[tuple[str, pd.DataFrame], ...]:
    entities = tuple((entity, frame.table(entity)) for entity in frame.entities)
    links = tuple((link, frame.link(link)) for link in frame.links)
    return (*entities, *links)


def _series_spec(series: pd.Series, *, label: str) -> dict[str, object]:
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
    if isinstance(dtype, pd.BooleanDtype):
        return {
            "dtype": "boolean",
            "encoding": _ENCODING_NULLABLE_BOOLEAN,
            "has_null_mask": bool(series.isna().any()),
        }
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) and not isinstance(
        dtype, pd.StringDtype
    ):
        raise TypeError(
            f"Frame checkpoint does not support extension dtype {dtype!s} for "
            f"{label!r}; use a NumPy or pandas string dtype."
        )
    if pd.api.types.is_object_dtype(dtype):
        inferred = pd.api.types.infer_dtype(series, skipna=True)
        if (
            inferred not in _SUPPORTED_OBJECT_INFERRED_DTYPES
            and not _contains_only_object_null_sentinels(series)
        ):
            raise TypeError(
                f"Frame checkpoint does not support object value family "
                f"{inferred!r} for {label!r}; use homogeneous scalar bool, "
                "string, bytes, integer, or floating values."
            )
        encoding = _ENCODING_OBJECT
    elif isinstance(dtype, pd.StringDtype):
        # ``str(dtype)`` collapses to "str"/"string", which pandas resolves
        # to the *environment-default* storage on restore (pyarrow when
        # installed). Record storage and NA marker explicitly so a
        # checkpoint restores one physical dtype everywhere.
        return {
            "dtype": str(dtype),
            "encoding": _ENCODING_OBJECT,
            "string_storage": dtype.storage,
            "string_na_marker": "pd_na" if dtype.na_value is pd.NA else "nan",
        }
    elif pd.api.types.is_datetime64_dtype(dtype):
        encoding = _ENCODING_DATETIME
    elif pd.api.types.is_timedelta64_dtype(dtype):
        encoding = _ENCODING_TIMEDELTA
    elif any(
        predicate(dtype)
        for predicate in (
            pd.api.types.is_bool_dtype,
            pd.api.types.is_integer_dtype,
            pd.api.types.is_float_dtype,
        )
    ):
        encoding = _ENCODING_NUMPY
    else:
        raise TypeError(
            f"Frame checkpoint does not support dtype {dtype!s} for {label!r}."
        )
    return {"dtype": str(dtype), "encoding": encoding}


def _contains_only_object_null_sentinels(series: pd.Series) -> bool:
    return all(
        value is None
        or value is pd.NA
        or value is pd.NaT
        or isinstance(value, (float, np.floating))
        and math.isnan(float(value))
        for value in series.to_numpy(dtype=object, copy=False)
    )


def _index_spec(index: pd.Index, *, label: str) -> dict[str, object]:
    if isinstance(index, pd.MultiIndex):
        raise TypeError(f"Frame checkpoint does not support MultiIndex for {label!r}.")
    base: dict[str, object] = {
        "name": _axis_name_payload(index.name, label=f"{label}.name"),
        "dtype": str(index.dtype),
    }
    if isinstance(index, pd.RangeIndex):
        return {
            **base,
            "kind": "range",
            "start": index.start,
            "stop": index.stop,
            "step": index.step,
        }
    series = pd.Series(index.array, copy=False)
    return {**base, "kind": "values", **_series_spec(series, label=label)}


def _write_index(group: Any, index: pd.Index, spec: Mapping[str, object]) -> None:
    if spec["kind"] == "range":
        return
    index_group = group.create_group("index", track_times=False)
    _write_series(index_group, pd.Series(index.array, copy=False), spec)


def _read_index(group: Any, spec: Mapping[str, Any], path: Path) -> pd.Index:
    kind = spec.get("kind")
    name = _axis_name_from_payload(spec.get("name"), label="index.name")
    if kind == "range":
        try:
            return pd.RangeIndex(
                start=int(spec["start"]),
                stop=int(spec["stop"]),
                step=int(spec["step"]),
                name=name,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Frame checkpoint {path} has malformed RangeIndex."
            ) from exc
    if kind != "values":
        raise ValueError(f"Frame checkpoint {path} has unknown index kind {kind!r}.")
    index_group = _require_h5_group(group, "index", path)
    series = _read_series(index_group, spec, path, label="index")
    dtype = _declared_dtype(spec, path=path, label="index")
    try:
        return pd.Index(series.array, dtype=dtype, name=name)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frame checkpoint {path} cannot restore index dtype {dtype!r}."
        ) from exc


def _write_series(group: Any, series: pd.Series, spec: Mapping[str, object]) -> None:
    encoding = spec["encoding"]
    if encoding == _ENCODING_NUMPY:
        _write_numpy_dataset(group, "values", series.to_numpy(copy=False))
        return
    if encoding in {_ENCODING_DATETIME, _ENCODING_TIMEDELTA}:
        values = series.to_numpy(copy=False).view(np.int64)
        _write_numpy_dataset(group, "values", values)
        return
    if encoding == _ENCODING_OBJECT:
        offsets, payload = _encode_object_values(
            series.to_numpy(dtype=object, copy=False)
        )
        _write_numpy_dataset(group, "offsets", offsets)
        _write_bytes_dataset(group, "payload", payload)
        return
    if encoding == _ENCODING_NULLABLE_BOOLEAN:
        values = series.to_numpy(
            dtype=np.bool_,
            na_value=False,
            copy=False,
        )
        _write_numpy_dataset(group, "values", values)
        if spec.get("has_null_mask") is True:
            null_mask = series.isna().to_numpy(dtype=np.uint8, copy=False)
            _write_numpy_dataset(group, "null_mask", null_mask)
        return
    raise RuntimeError(f"Unknown checkpoint series encoding {encoding!r}.")


def _read_series(
    group: Any,
    spec: Mapping[str, Any],
    path: Path,
    *,
    label: str,
) -> pd.Series:
    dtype = _require_string(spec, "dtype", label=label)
    restore_dtype = _declared_dtype(spec, path=path, label=label)
    encoding = _require_string(spec, "encoding", label=label)
    if encoding == _ENCODING_NUMPY:
        values = _read_numpy_dataset(group, "values", path)
        series = pd.Series(values, copy=False)
    elif encoding in {_ENCODING_DATETIME, _ENCODING_TIMEDELTA}:
        values = _read_numpy_dataset(group, "values", path)
        if values.dtype != np.dtype(np.int64):
            raise ValueError(
                f"Frame checkpoint {path} {label!r} stores {encoding} as "
                f"{values.dtype!s}, not int64."
            )
        series = pd.Series(values.view(np.dtype(dtype)), copy=False)
    elif encoding == _ENCODING_OBJECT:
        offsets = _read_numpy_dataset(group, "offsets", path)
        payload = _read_bytes_dataset(group, "payload", path)
        values = _decode_object_values(offsets, payload, path=path, label=label)
        series = pd.Series(values, dtype=restore_dtype, copy=False)
    elif encoding == _ENCODING_NULLABLE_BOOLEAN:
        if dtype != "boolean":
            raise ValueError(
                f"Frame checkpoint {path} {label!r} nullable boolean encoding "
                f"has declared dtype {dtype!r}, not 'boolean'."
            )
        has_null_mask = spec.get("has_null_mask")
        if type(has_null_mask) is not bool:
            raise ValueError(
                f"Frame checkpoint {path} {label!r} nullable boolean "
                "has_null_mask must be a boolean."
            )
        values = _read_numpy_dataset(group, "values", path)
        if values.ndim != 1 or values.dtype != np.dtype(np.bool_):
            raise ValueError(
                f"Frame checkpoint {path} {label!r} nullable boolean values "
                "must be a one-dimensional bool array."
            )
        if has_null_mask:
            if "null_mask" not in group:
                raise ValueError(
                    f"Frame checkpoint {path} {label!r} is missing its null mask."
                )
            null_mask = _read_numpy_dataset(group, "null_mask", path)
            if (
                null_mask.ndim != 1
                or null_mask.dtype != np.dtype(np.uint8)
                or len(null_mask) != len(values)
                or ((null_mask != 0) & (null_mask != 1)).any()
                or not null_mask.any()
            ):
                raise ValueError(
                    f"Frame checkpoint {path} {label!r} null mask must be a "
                    "one-dimensional uint8 0/1 array aligned to its values."
                )
            mask = null_mask.astype(np.bool_, copy=False)
            if values[mask].any():
                raise ValueError(
                    f"Frame checkpoint {path} {label!r} null mask covers "
                    "noncanonical true storage bits."
                )
        else:
            if "null_mask" in group:
                raise ValueError(
                    f"Frame checkpoint {path} {label!r} has an unexpected null mask."
                )
            mask = np.zeros(len(values), dtype=np.bool_)
        series = pd.Series(
            pd.arrays.BooleanArray(values, mask, copy=False),
            copy=False,
        )
    else:
        raise ValueError(
            f"Frame checkpoint {path} has unknown encoding {encoding!r} for {label!r}."
        )
    if not _dtype_matches(series.dtype, restore_dtype):
        try:
            series = series.astype(restore_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Frame checkpoint {path} cannot restore dtype {dtype!r} for "
                f"{label!r}; stored dtype is {series.dtype!s}."
            ) from exc
    if not _dtype_matches(series.dtype, restore_dtype):
        raise ValueError(
            f"Frame checkpoint {path} restored {label!r} as {series.dtype!s}, "
            f"not declared dtype {dtype!r}."
        )
    return series


def _declared_dtype(
    spec: Mapping[str, Any], *, path: Path, label: str
) -> str | pd.BooleanDtype | pd.StringDtype:
    """The concrete dtype a spec restores to, environment-independently.

    String dtypes resolve through the recorded storage and NA marker — the
    pandas string forms ("str"/"string") pick the *environment-default*
    storage, which flips to pyarrow wherever pyarrow is installed and would
    make a checkpoint restore to different physical dtypes on different
    machines. Legacy specs (written before storage was recorded) restore to
    python storage: that is what every environment those checkpoints were
    verified in produced, and it is the build's canonical string policy.
    """
    dtype = _require_string(spec, "dtype", label=label)
    if dtype == "boolean":
        return pd.BooleanDtype()
    if dtype not in ("str", "string"):
        return dtype
    storage = spec.get("string_storage", "python")
    if storage not in ("python", "pyarrow"):
        raise ValueError(
            f"Frame checkpoint {path} {label!r} has unknown string storage {storage!r}."
        )
    na_marker = spec.get("string_na_marker", "nan" if dtype == "str" else "pd_na")
    if na_marker not in ("nan", "pd_na"):
        raise ValueError(
            f"Frame checkpoint {path} {label!r} has unknown string NA marker "
            f"{na_marker!r}."
        )
    return pd.StringDtype(
        storage=storage,
        na_value=np.nan if na_marker == "nan" else pd.NA,
    )


def _dtype_matches(
    actual: Any,
    expected: str | pd.BooleanDtype | pd.StringDtype,
) -> bool:
    if isinstance(expected, pd.BooleanDtype):
        return actual == expected
    if isinstance(expected, pd.StringDtype):
        return actual == expected and getattr(actual, "storage", None) == (
            expected.storage
        )
    return str(actual) == expected


def _encode_object_values(values: np.ndarray) -> tuple[np.ndarray, bytes]:
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    payload = bytearray()
    for index, value in enumerate(values):
        payload.extend(_encode_object_scalar(value))
        offsets[index + 1] = len(payload)
    return offsets, bytes(payload)


def _encode_object_scalar(value: object) -> bytes:
    if value is None:
        return bytes([_TAG_NONE])
    if value is pd.NA:
        return bytes([_TAG_PD_NA])
    if value is pd.NaT:
        return bytes([_TAG_PD_NAT])
    if isinstance(value, (bool, np.bool_)):
        return bytes([_TAG_TRUE if bool(value) else _TAG_FALSE])
    if isinstance(value, (int, np.integer)):
        return bytes([_TAG_INTEGER]) + str(int(value)).encode("ascii")
    if isinstance(value, (float, np.floating)):
        return bytes([_TAG_FLOAT64]) + struct.pack("<d", float(value))
    if isinstance(value, str):
        return bytes([_TAG_STRING]) + value.encode("utf-8")
    if isinstance(value, (bytes, np.bytes_)):
        return bytes([_TAG_BYTES]) + bytes(value)
    raise TypeError(
        "Frame checkpoint object codec received unsupported scalar "
        f"{type(value).__name__}."
    )


def _decode_object_values(
    offsets: np.ndarray,
    payload: bytes,
    *,
    path: Path,
    label: str,
) -> np.ndarray:
    if offsets.ndim != 1 or offsets.dtype != np.dtype(np.int64) or len(offsets) < 1:
        raise ValueError(
            f"Frame checkpoint {path} {label!r} has malformed object offsets."
        )
    if offsets[0] != 0 or offsets[-1] != len(payload) or (np.diff(offsets) < 1).any():
        raise ValueError(
            f"Frame checkpoint {path} {label!r} has invalid object boundaries."
        )
    values = np.empty(len(offsets) - 1, dtype=object)
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        chunk = payload[int(start) : int(stop)]
        tag, body = chunk[0], chunk[1:]
        try:
            if tag == _TAG_NONE and not body:
                value = None
            elif tag == _TAG_PD_NA and not body:
                value = pd.NA
            elif tag == _TAG_PD_NAT and not body:
                value = pd.NaT
            elif tag == _TAG_FALSE and not body:
                value = False
            elif tag == _TAG_TRUE and not body:
                value = True
            elif tag == _TAG_INTEGER:
                value = int(body.decode("ascii"))
            elif tag == _TAG_FLOAT64 and len(body) == 8:
                value = struct.unpack("<d", body)[0]
            elif tag == _TAG_STRING:
                value = body.decode("utf-8")
            elif tag == _TAG_BYTES:
                value = body
            else:
                raise ValueError
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"Frame checkpoint {path} {label!r} has invalid object value "
                f"at row {index}."
            ) from exc
        values[index] = value
    return values


def _write_numpy_dataset(group: Any, name: str, values: np.ndarray) -> None:
    group.create_dataset(name, data=np.asarray(values), track_times=False)


def _write_bytes_dataset(group: Any, name: str, value: bytes) -> None:
    group.create_dataset(
        name,
        data=np.frombuffer(value, dtype=np.uint8),
        track_times=False,
    )


def _read_numpy_dataset(group: Any, name: str, path: Path) -> np.ndarray:
    dataset = _require_h5_dataset(group, name, path)
    return np.asarray(dataset)


def _read_bytes_dataset(group: Any, name: str, path: Path) -> bytes:
    values = _read_numpy_dataset(group, name, path)
    if values.ndim != 1 or values.dtype != np.dtype(np.uint8):
        raise ValueError(
            f"Frame checkpoint {path} HDF dataset {name!r} must be uint8 bytes."
        )
    return values.tobytes()


def _read_metadata(root: Any, path: Path) -> dict[str, Any]:
    raw = _read_bytes_dataset(root, _METADATA_DATASET, path)
    try:
        metadata = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Frame checkpoint {path} metadata is not valid JSON."
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Frame checkpoint {path} metadata must be a JSON object.")
    if metadata.get("artifact_kind") != _ARTIFACT_KIND:
        raise ValueError(
            f"Frame checkpoint {path} has unknown artifact kind "
            f"{metadata.get('artifact_kind')!r}."
        )
    version = metadata.get("schema_version")
    if version not in _SUPPORTED_FRAME_CHECKPOINT_SCHEMA_VERSIONS:
        raise ValueError(
            f"Frame checkpoint {path} schema version is {version!r}; expected "
            f"one of {sorted(_SUPPORTED_FRAME_CHECKPOINT_SCHEMA_VERSIONS)}."
        )
    nullable_spec_count = 0
    for label, spec in _checkpoint_series_specs(metadata):
        dtype = spec.get("dtype")
        encoding = spec.get("encoding")
        declares_nullable = dtype == "boolean"
        uses_nullable_encoding = encoding == _ENCODING_NULLABLE_BOOLEAN
        if version == _LEGACY_FRAME_CHECKPOINT_SCHEMA_VERSION and (
            declares_nullable or uses_nullable_encoding
        ):
            raise ValueError(
                f"Frame checkpoint {path} schema version 2 cannot carry nullable "
                f"boolean spec {label!r}."
            )
        if version == FRAME_CHECKPOINT_SCHEMA_VERSION and (
            declares_nullable != uses_nullable_encoding
        ):
            raise ValueError(
                f"Frame checkpoint {path} schema version 3 nullable boolean spec "
                f"{label!r} must pair declared dtype 'boolean' with encoding "
                f"{_ENCODING_NULLABLE_BOOLEAN!r}."
            )
        if uses_nullable_encoding:
            nullable_spec_count += 1
            if type(spec.get("has_null_mask")) is not bool:
                raise ValueError(
                    f"Frame checkpoint {path} {label!r} nullable boolean "
                    "has_null_mask must be a boolean."
                )
    if version == FRAME_CHECKPOINT_SCHEMA_VERSION and nullable_spec_count == 0:
        raise ValueError(
            f"Frame checkpoint {path} schema version 3 requires at least one "
            "nullable boolean spec."
        )
    return metadata


def _checkpoint_series_specs(
    metadata: Mapping[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return every serialized series spec for version-policy validation."""

    specs: list[tuple[str, dict[str, Any]]] = []
    for table_index, raw_table in enumerate(_require_list(metadata, "tables")):
        table = _require_dict(raw_table, f"tables[{table_index}]")
        name = table.get("name", f"tables[{table_index}]")
        index = _require_dict(table.get("index"), f"tables[{table_index}].index")
        if index.get("kind") == "values":
            specs.append((f"{name}.index", index))
        for column_index, raw_column in enumerate(_require_list(table, "columns")):
            column = _require_dict(
                raw_column,
                f"tables[{table_index}].columns[{column_index}]",
            )
            specs.append((f"{name}.{column.get('name', column_index)}", column))
    specs.append(("strata", _require_dict(metadata.get("strata"), "strata")))
    return tuple(specs)


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
                    link, "left_entity", label=f"schema.links[{index}]"
                ),
                right_entity=_require_string(
                    link, "right_entity", label=f"schema.links[{index}]"
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


def _mass_log_from_metadata(
    metadata: Mapping[str, Any], path: Path
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


def _axis_name_payload(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Frame checkpoint {label} must be finite.")
        return value
    if isinstance(value, tuple):
        return {"tuple": [_axis_name_payload(item, label=label) for item in value]}
    raise TypeError(
        f"Frame checkpoint {label} has unsupported type {type(value).__name__}."
    )


def _axis_name_from_payload(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict) and set(value) == {"tuple"}:
        items = value["tuple"]
        if isinstance(items, list):
            return tuple(_axis_name_from_payload(item, label=label) for item in items)
    raise ValueError(f"Frame checkpoint {label} is malformed.")


def _normalize_external_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}.")
    try:
        normalized = json.loads(_canonical_json(dict(metadata)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Frame checkpoint metadata must be finite and JSON-compatible."
        ) from exc
    if not isinstance(normalized, dict):  # pragma: no cover
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


def _require_h5_group(container: Any, name: str, path: Path) -> Any:
    h5py = _h5py()
    if name not in container or not isinstance(container[name], h5py.Group):
        raise ValueError(f"Frame checkpoint {path} is missing HDF group {name!r}.")
    return container[name]


def _require_h5_dataset(container: Any, name: str, path: Path) -> Any:
    h5py = _h5py()
    if name not in container or not isinstance(container[name], h5py.Dataset):
        raise ValueError(f"Frame checkpoint {path} is missing HDF dataset {name!r}.")
    return container[name]


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Frame checkpoint {label} must be an object.")
    return value


def _require_list(container: Mapping[str, Any], key: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Frame checkpoint {key} must be a list.")
    return value


def _require_string(container: Mapping[str, Any], key: str, *, label: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Frame checkpoint {label}.{key} must be a string.")
    return value


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - US extra supplies h5py
        raise RuntimeError(
            "Frame checkpoints require h5py; install microcosm-build[us]."
        ) from exc
    return h5py
