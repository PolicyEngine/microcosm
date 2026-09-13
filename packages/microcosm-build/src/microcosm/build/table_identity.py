"""Native table identity without constructing a population Frame.

The canonical_scalar_v1 values codec is factored unchanged from the US late
producer runtime. Its domain labels and optional string normalization remain
stable for existing receipts. native_table_identity additionally binds a closed
native dtype/schema descriptor, including pandas string storage and NA kind.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import canonicalize_table_string_dtypes

TABLE_VALUES_DIGEST_CODEC = "canonical_scalar_v1"
_TABLE_DIGEST_CHUNK_ROWS = 65_536


def _digest_part(
    digest,
    *,
    domain: str,
    payload: bytes | bytearray | memoryview | np.ndarray,
) -> None:
    """Append one length-framed, domain-separated byte field to a digest."""

    domain_bytes = domain.encode("utf-8")
    payload_view = memoryview(payload)
    if payload_view.format != "B" or payload_view.ndim != 1:
        payload_view = payload_view.cast("B")
    digest.update(struct.pack("<I", len(domain_bytes)))
    digest.update(domain_bytes)
    digest.update(struct.pack("<Q", payload_view.nbytes))
    digest.update(payload_view)


def _little_endian_bytes(values: np.ndarray) -> np.ndarray:
    """Return a contiguous, explicitly little-endian numeric byte source."""

    array = np.asarray(values)
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def _scalar_bytes(value: object) -> bytes:
    """Encode one supported object scalar without lossy intermediary hashes."""

    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return b"null"
    if isinstance(value, (bool, np.bool_)):
        return b"bool\x01" if bool(value) else b"bool\x00"
    if isinstance(value, (int, np.integer)):
        return b"integer\x00" + str(int(value)).encode("ascii")
    if isinstance(value, (float, np.floating)):
        if isinstance(value, np.floating) and value.dtype.itemsize > 8:
            raise TypeError(
                "US late-producer content digest does not support object "
                f"floating scalar {value.dtype!s}."
            )
        return b"float64\x00" + struct.pack("<d", float(value))
    if isinstance(value, (complex, np.complexfloating)):
        if isinstance(value, np.complexfloating) and value.dtype.itemsize > 16:
            raise TypeError(
                "US late-producer content digest does not support object "
                f"complex scalar {value.dtype!s}."
            )
        numeric = complex(value)
        return b"complex128\x00" + struct.pack("<dd", numeric.real, numeric.imag)
    if isinstance(value, str):
        return b"string\x00" + value.encode("utf-8")
    if isinstance(value, (bytes, np.bytes_)):
        return b"bytes\x00" + bytes(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        return b"datetime64ns\x00" + struct.pack("<q", timestamp.value)
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        delta = pd.Timedelta(value)
        return b"timedelta64ns\x00" + struct.pack("<q", delta.value)
    raise TypeError(
        "US late-producer content digest received unsupported object scalar "
        f"{type(value).__name__}."
    )


def _digest_variable_width_values(
    digest,
    values: Sequence[object],
    missing: np.ndarray,
    *,
    domain: str,
    strings_only: bool,
) -> None:
    """Stream framed string or object scalars in bounded-memory chunks."""

    for chunk_index, start in enumerate(
        range(0, len(values), _TABLE_DIGEST_CHUNK_ROWS)
    ):
        stop = min(start + _TABLE_DIGEST_CHUNK_ROWS, len(values))
        lengths = np.zeros(stop - start, dtype="<u8")
        payload = bytearray()
        for local_index, value in enumerate(values[start:stop]):
            if missing[start + local_index]:
                encoded = b""
            elif strings_only:
                if not isinstance(value, str):
                    raise TypeError(
                        "US late-producer string digest received non-string "
                        f"scalar {type(value).__name__}."
                    )
                encoded = value.encode("utf-8")
            else:
                encoded = _scalar_bytes(value)
            lengths[local_index] = len(encoded)
            payload.extend(encoded)
        chunk_domain = f"{domain}/chunk/{chunk_index}"
        _digest_part(
            digest,
            domain=f"{chunk_domain}/lengths",
            payload=lengths,
        )
        _digest_part(
            digest,
            domain=f"{chunk_domain}/payload",
            payload=payload,
        )


def _digest_series_values(
    digest,
    series: pd.Series,
    *,
    domain: str,
) -> None:
    """Hash one ordered logical Series with explicit dtype and null domains."""

    dtype = series.dtype
    missing = series.isna().to_numpy(dtype=bool)
    _digest_part(
        digest,
        domain=f"{domain}/dtype",
        payload=str(dtype).encode("utf-8"),
    )
    _digest_part(
        digest,
        domain=f"{domain}/row_count",
        payload=struct.pack("<Q", len(series)),
    )
    _digest_part(
        digest,
        domain=f"{domain}/null_bitmap",
        payload=np.ascontiguousarray(missing, dtype=np.uint8),
    )

    if pd.api.types.is_bool_dtype(dtype):
        encoded = series.to_numpy(dtype=np.uint8, na_value=0)
    elif pd.api.types.is_integer_dtype(dtype):
        numpy_dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
        encoded = series.to_numpy(dtype=numpy_dtype, na_value=0)
    elif pd.api.types.is_float_dtype(dtype):
        numpy_dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
        encoded = series.to_numpy(dtype=numpy_dtype, na_value=0.0)
        if missing.any():
            encoded = encoded.copy()
            encoded[missing] = 0.0
    elif pd.api.types.is_complex_dtype(dtype):
        numpy_dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
        encoded = series.to_numpy(dtype=numpy_dtype, na_value=0.0j)
        if missing.any():
            encoded = encoded.copy()
            encoded[missing] = 0.0j
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        encoded = series.array.asi8.copy()
        encoded[missing] = 0
    elif pd.api.types.is_timedelta64_dtype(dtype):
        encoded = series.array.asi8.copy()
        encoded[missing] = 0
    elif isinstance(dtype, pd.StringDtype):
        _digest_variable_width_values(
            digest,
            series.array,
            missing,
            domain=f"{domain}/strings",
            strings_only=True,
        )
        return
    else:
        _digest_variable_width_values(
            digest,
            series.array,
            missing,
            domain=f"{domain}/objects",
            strings_only=False,
        )
        return

    _digest_part(
        digest,
        domain=f"{domain}/fixed_width_values",
        payload=_little_endian_bytes(encoded),
    )


def table_values_sha256(
    table: pd.DataFrame,
    *,
    normalize_strings: bool = False,
) -> str:
    """Hash ordered table scalars directly with typed, null-aware framing."""

    values = (
        canonicalize_table_string_dtypes(
            table,
            boundary="late-producer content digest",
            table_name="declared_surface",
        )
        if normalize_strings
        else table
    )
    if isinstance(values.index, pd.MultiIndex):
        index_levels = [
            pd.Series(values.index.get_level_values(level), copy=False)
            for level in range(values.index.nlevels)
        ]
    else:
        index_levels = [pd.Series(values.index, copy=False)]
    header = {
        "codec": TABLE_VALUES_DIGEST_CODEC,
        "columns": [str(column) for column in values.columns],
        "dtypes": [
            str(values.iloc[:, index].dtype) for index in range(values.shape[1])
        ],
        "index_type": type(values.index).__name__,
        "index_dtype": str(values.index.dtype),
        "index_level_dtypes": [str(level.dtype) for level in index_levels],
        "index_names": [
            None if name is None else str(name) for name in values.index.names
        ],
    }
    digest = hashlib.sha256()
    _digest_part(
        digest,
        domain="late_table_digest_codec",
        payload=TABLE_VALUES_DIGEST_CODEC.encode("ascii"),
    )
    _digest_part(
        digest,
        domain="late_table_header_json",
        payload=json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )
    for level_index, level in enumerate(index_levels):
        _digest_series_values(
            digest,
            level,
            domain=f"index_level/{level_index}",
        )
    for column_index in range(values.shape[1]):
        _digest_series_values(
            digest,
            values.iloc[:, column_index],
            domain=f"column/{column_index}",
        )
    return digest.hexdigest()


def _native_dtype(dtype: object) -> dict[str, str]:
    if isinstance(dtype, pd.StringDtype):
        return {
            "kind": "pandas_string",
            "storage": dtype.storage,
            "na_value": "pd.NA" if dtype.na_value is pd.NA else "nan",
        }
    if isinstance(dtype, np.dtype) and dtype.kind in "biufO":
        return {"kind": "numpy", "dtype": dtype.str}
    if isinstance(
        dtype,
        (
            pd.BooleanDtype,
            pd.Int8Dtype,
            pd.Int16Dtype,
            pd.Int32Dtype,
            pd.Int64Dtype,
            pd.UInt8Dtype,
            pd.UInt16Dtype,
            pd.UInt32Dtype,
            pd.UInt64Dtype,
            pd.Float32Dtype,
            pd.Float64Dtype,
        ),
    ):
        return {"kind": "pandas_nullable", "dtype": str(dtype)}
    raise TypeError(f"Unsupported native table identity dtype: {dtype!r}.")


def native_table_identity(table: pd.DataFrame) -> dict[str, Any]:
    """Bind native values, order, index and a closed exact dtype schema.

    This narrow identity supports numeric, boolean and string donor tables,
    including nullable numeric columns. Categorical/temporal/custom extension
    dtypes and MultiIndex axes require a separately declared contract. Nulls
    retain their positions; individual floating NaN payload bits are canonical.
    No normalization or coercion is performed on the caller's table.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Native table identity requires a pandas DataFrame.")
    if isinstance(table.index, pd.MultiIndex) or isinstance(
        table.columns, pd.MultiIndex
    ):
        raise TypeError("Native table identity does not support MultiIndex axes.")
    if table.columns.has_duplicates or not all(
        isinstance(x, str) and x for x in table.columns
    ):
        raise TypeError(
            "Native table identity requires unique nonempty string columns."
        )
    for axis in (table.index, table.columns):
        if axis.name is not None and not isinstance(axis.name, str):
            raise TypeError(
                "Native table identity requires string or absent axis names."
            )
    series = [
        pd.Series(table.index),
        *(table.iloc[:, i] for i in range(table.shape[1])),
    ]
    for values in series:
        if values.dtype == object and not all(
            isinstance(x, str) for x in values.dropna()
        ):
            raise TypeError("Native object dtype may contain only strings or nulls.")
    identity = {
        "schema": "microcosm.native_table_identity.v1",
        "values_sha256": table_values_sha256(table, normalize_strings=False),
        "index": {
            "type": type(table.index).__name__,
            "name": table.index.name,
            "dtype": _native_dtype(table.index.dtype),
        },
        "column_axis": {
            "type": type(table.columns).__name__,
            "name": table.columns.name,
            "dtype": _native_dtype(table.columns.dtype),
        },
        "columns": [
            {"name": name, "dtype": _native_dtype(table[name].dtype)}
            for name in table.columns
        ],
    }
    identity["sha256"] = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return identity
