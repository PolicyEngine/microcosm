"""Atomic, content-validated storage for graph artifacts.

Objects live at ``objects/<key[:2]>/<key>``.  Each object contains a
canonical ``meta.json`` whose payload table records the SHA-256 and length of
every other file.  Loaders validate the complete table before decoding any
payload, so a damaged object is never mistaken for a cache miss.

Persisted NumPy arrays never use pickle.  Native numeric columns use their
exact NumPy buffer; nullable booleans and integers use a canonical value
buffer plus a null mask; strings use one UTF-8 byte buffer plus int64 offsets
and a null mask.  The latter representation round-trips embedded NULs and
distinguishes an empty string from a missing value without an object array.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import struct
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import pandas as pd

from microcosm.frame import (
    EntitySchema,
    Frame,
    LinkSpec,
    MassChangeRecord,
    WeightKind,
    Weights,
    nullable_boolean_values_and_mask,
)

from .errors import (
    GraphRuntimeError,
    StoreCorruptError,
    StoreMissError,
    StoreUnavailableError,
)

__all__ = [
    "ContentStore",
    "ResumePolicy",
    "StoreCorrupt",
    "StoreCorruptError",
    "StoreError",
    "StoreMiss",
    "StoreMissError",
    "StoreUnavailable",
    "StoreUnavailableError",
]

type ResumePolicy = Literal["auto", "require", "forbid"]

_STORE_FORMAT = "microcosm-graph-content-store-v1"
_FRAME_FORMAT = "microcosm-graph-frame-v1"
_KEY = re.compile(r"[0-9a-f]{64}\Z")

_ENCODING_NUMPY = "numpy-v1"
_ENCODING_NULLABLE_BOOLEAN = "nullable-boolean-v1"
_ENCODING_NULLABLE_INTEGER = "nullable-integer-v1"
_ENCODING_UTF8 = "utf8-offsets-v1"
_ENCODING_OBJECT = "object-scalars-v1"

_TAG_NONE = 0
_TAG_PD_NA = 1
_TAG_PD_NAT = 2
_TAG_FALSE = 3
_TAG_TRUE = 4
_TAG_INTEGER = 5
_TAG_FLOAT64 = 6
_TAG_STRING = 7
_TAG_BYTES = 8

_DECLARED_DTYPES = frozenset(
    {"bool", "boolean", "int32", "int64", "Int64", "float32", "float64", "string"}
)


# Pre-amendment aliases remain importable for callers that adopted the shard
# before the shared Error-suffixed exception contract landed.
StoreError = GraphRuntimeError
StoreMiss = StoreMissError
StoreCorrupt = StoreCorruptError
StoreUnavailable = StoreUnavailableError


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Store metadata must be finite, JSON-compatible data."
        ) from error


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(value)


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, _canonical_json(value))


def _write_array(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("ContentStore never persists object NumPy arrays.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_table(root: Path) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StoreCorrupt(f"Staged store payload cannot be a symlink: {path}.")
        if not path.is_file() or path.name == "meta.json":
            continue
        relative = path.relative_to(root).as_posix()
        payloads[relative] = {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    return payloads


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for directory, _children, files in os.walk(root):
        path = Path(directory)
        directories.append(path)
        for filename in files:
            descriptor = os.open(path / filename, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _safe_payload_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise StoreCorrupt("Store payload names must be non-empty strings.")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or value != candidate.as_posix()
    ):
        raise StoreCorrupt(f"Unsafe store payload name {value!r}.")
    return value


def _require_key(key: str) -> str:
    if not isinstance(key, str) or _KEY.fullmatch(key) is None:
        raise ValueError("Content-store keys must be 64 lowercase hexadecimal digits.")
    return key


def _load_json_file(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StoreCorrupt(f"Stored {label} is not readable canonical JSON.") from error


def _verified_meta(path: Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    if not path.exists():
        raise StoreMiss(f"No stored object at {path}.")
    if path.is_symlink() or not path.is_dir():
        raise StoreCorrupt(f"Stored object path is not a real directory: {path}.")
    metadata_path = path / "meta.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise StoreCorrupt(f"Stored object {path} has no regular meta.json.")
    metadata = _load_json_file(metadata_path, label=f"metadata for {path.name}")
    if not isinstance(metadata, dict):
        raise StoreCorrupt(f"Stored object {path} metadata must be an object.")
    if metadata.get("format") != _STORE_FORMAT:
        raise StoreUnavailable(
            f"Stored object {path} needs unsupported format {metadata.get('format')!r}."
        )
    if metadata.get("key") != path.name or _KEY.fullmatch(path.name) is None:
        raise StoreCorrupt(f"Stored object {path} does not bind its directory key.")
    if expected_kind is not None and metadata.get("kind") != expected_kind:
        raise StoreCorrupt(
            f"Stored object {path.name} is kind {metadata.get('kind')!r}, "
            f"not {expected_kind!r}."
        )
    raw_payloads = metadata.get("payloads")
    if not isinstance(raw_payloads, dict):
        raise StoreCorrupt(f"Stored object {path.name} has no payload table.")

    expected_names: set[str] = set()
    for raw_name, raw_record in raw_payloads.items():
        name = _safe_payload_name(raw_name)
        if not isinstance(raw_record, dict):
            raise StoreCorrupt(f"Payload record {name!r} is not an object.")
        expected_hash = raw_record.get("sha256")
        expected_size = raw_record.get("size")
        if not isinstance(expected_hash, str) or _KEY.fullmatch(expected_hash) is None:
            raise StoreCorrupt(f"Payload {name!r} has an invalid SHA-256.")
        if type(expected_size) is not int or expected_size < 0:
            raise StoreCorrupt(f"Payload {name!r} has an invalid size.")
        payload_path = path / name
        if payload_path.is_symlink() or not payload_path.is_file():
            raise StoreCorrupt(
                f"Stored object {path.name} is missing payload {name!r}."
            )
        try:
            actual_size = payload_path.stat().st_size
            actual_hash = _sha256_file(payload_path)
        except OSError as error:
            raise StoreCorrupt(f"Stored payload {name!r} cannot be read.") from error
        if actual_size != expected_size or actual_hash != expected_hash:
            raise StoreCorrupt(
                f"Stored payload {name!r} failed its size/SHA-256 check."
            )
        expected_names.add(name)

    actual_names: set[str] = set()
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise StoreCorrupt(f"Stored object contains a symlink: {candidate}.")
        if candidate.is_file() and candidate != metadata_path:
            actual_names.add(candidate.relative_to(path).as_posix())
    if actual_names != expected_names:
        raise StoreCorrupt(
            f"Stored object {path.name} payload table differs from files on disk."
        )
    return metadata


def _load_array(path: Path, *, label: str) -> np.ndarray:
    try:
        with path.open("rb") as stream:
            values = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise StoreCorrupt(f"Stored array {label!r} cannot be decoded.") from error
    if not isinstance(values, np.ndarray) or values.dtype.hasobject:
        raise StoreCorrupt(f"Stored array {label!r} is not a pickle-free ndarray.")
    return values


def _dtype_matches_declared(dtype: object, token: str) -> bool:
    if token == "string":
        return isinstance(dtype, pd.StringDtype) and str(dtype) == "string"
    return str(dtype) == token


def _string_dtype(spec: Mapping[str, Any]) -> pd.StringDtype:
    storage = spec.get("string_storage")
    marker = spec.get("string_na_marker")
    if storage not in {"python", "pyarrow"} or marker not in {"pd_na", "nan"}:
        raise StoreCorrupt("Stored string dtype metadata is malformed.")
    na_value = pd.NA if marker == "pd_na" else np.nan
    try:
        return pd.StringDtype(storage=storage, na_value=na_value)
    except ImportError as error:
        raise StoreUnavailable(
            f"Stored string data requires pandas string storage {storage!r}."
        ) from error
    except TypeError as error:  # pandas without explicit NA-marker support
        if marker == "nan":
            raise StoreUnavailable(
                "This pandas version cannot restore the stored string NA policy."
            ) from error
        try:
            return pd.StringDtype(storage=storage)
        except ImportError as nested:
            raise StoreUnavailable(
                f"Stored string data requires pandas string storage {storage!r}."
            ) from nested


def _encode_offsets(chunks: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty(len(chunks) + 1, dtype=np.int64)
    offsets[0] = 0
    payload = bytearray()
    for index, chunk in enumerate(chunks):
        payload.extend(chunk)
        offsets[index + 1] = len(payload)
    return offsets, np.frombuffer(bytes(payload), dtype=np.uint8).copy()


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
        "Object columns may contain only null sentinels, bool, int, float, "
        f"str, or bytes; found {type(value).__name__}."
    )


def _decode_object_chunks(
    offsets: np.ndarray, payload: np.ndarray, *, label: str
) -> np.ndarray:
    raw = payload.tobytes()
    _validate_offsets(offsets, len(raw), label=label, require_nonempty=True)
    values = np.empty(len(offsets) - 1, dtype=object)
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        chunk = raw[int(start) : int(stop)]
        tag, body = chunk[0], chunk[1:]
        try:
            if tag == _TAG_NONE and not body:
                value: object = None
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
        except (UnicodeDecodeError, ValueError) as error:
            raise StoreCorrupt(
                f"Stored object scalar {label}[{index}] is malformed."
            ) from error
        values[index] = value
    return values


def _validate_offsets(
    offsets: np.ndarray,
    payload_length: int,
    *,
    label: str,
    require_nonempty: bool = False,
) -> None:
    invalid = (
        offsets.ndim != 1
        or offsets.dtype != np.dtype(np.int64)
        or len(offsets) < 1
        or offsets[0] != 0
        or offsets[-1] != payload_length
        or (np.diff(offsets) < (1 if require_nonempty else 0)).any()
    )
    if invalid:
        raise StoreCorrupt(f"Stored offsets for {label!r} are malformed.")


def _write_series(root: Path, series: pd.Series) -> dict[str, object]:
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pandas Series, got {type(series).__name__}.")
    dtype = series.dtype
    spec: dict[str, object] = {
        "length": len(series),
        "pandas_dtype": str(dtype),
    }
    if isinstance(dtype, pd.BooleanDtype):
        values, mask = nullable_boolean_values_and_mask(series)
        _write_array(root / "values.npy", values)
        _write_array(root / "mask.npy", mask)
        return {**spec, "encoding": _ENCODING_NULLABLE_BOOLEAN}
    if (
        isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and pd.api.types.is_integer_dtype(dtype)
    ):
        numpy_dtype = getattr(dtype, "numpy_dtype", None)
        if numpy_dtype is None:
            raise TypeError(f"Unsupported nullable integer dtype {dtype!s}.")
        mask = series.isna().to_numpy(dtype=np.bool_, copy=False)
        values = series.to_numpy(dtype=numpy_dtype, na_value=0, copy=True)
        values[mask] = 0
        _write_array(root / "values.npy", values)
        _write_array(root / "mask.npy", mask)
        return {**spec, "encoding": _ENCODING_NULLABLE_INTEGER}
    if isinstance(dtype, pd.StringDtype):
        mask = series.isna().to_numpy(dtype=np.bool_, copy=False)
        chunks: list[bytes] = []
        for missing, value in zip(mask, series.astype(object), strict=True):
            if missing:
                chunks.append(b"")
            elif isinstance(value, str):
                chunks.append(value.encode("utf-8"))
            else:  # pragma: no cover - pandas StringDtype guarantees strings
                raise TypeError("StringDtype yielded a non-string value.")
        offsets, payload = _encode_offsets(chunks)
        _write_array(root / "values.npy", payload)
        _write_array(root / "offsets.npy", offsets)
        _write_array(root / "mask.npy", mask)
        return {
            **spec,
            "encoding": _ENCODING_UTF8,
            "string_storage": dtype.storage,
            "string_na_marker": "pd_na" if dtype.na_value is pd.NA else "nan",
        }
    if pd.api.types.is_object_dtype(dtype):
        chunks = [_encode_object_scalar(value) for value in series.to_numpy(object)]
        offsets, payload = _encode_offsets(chunks)
        _write_array(root / "values.npy", payload)
        _write_array(root / "offsets.npy", offsets)
        return {**spec, "encoding": _ENCODING_OBJECT}
    if isinstance(dtype, pd.CategoricalDtype) or isinstance(dtype, pd.DatetimeTZDtype):
        raise TypeError(f"ContentStore does not support dtype {dtype!s}.")
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        raise TypeError(f"ContentStore does not support extension dtype {dtype!s}.")
    values = series.to_numpy(copy=False)
    if values.dtype.hasobject:
        raise TypeError(f"ContentStore cannot persist dtype {dtype!s} without pickle.")
    _write_array(root / "values.npy", values)
    return {**spec, "encoding": _ENCODING_NUMPY}


def _read_series(root: Path, spec: Mapping[str, Any], *, label: str) -> pd.Series:
    length = spec.get("length")
    pandas_dtype = spec.get("pandas_dtype")
    encoding = spec.get("encoding")
    if type(length) is not int or length < 0 or not isinstance(pandas_dtype, str):
        raise StoreCorrupt(f"Stored series specification for {label!r} is malformed.")
    try:
        if encoding == _ENCODING_NUMPY:
            values = _load_array(root / "values.npy", label=f"{label}.values")
            if values.ndim != 1:
                raise StoreCorrupt(f"Stored series {label!r} is not one-dimensional.")
            series = pd.Series(values, copy=False)
        elif encoding == _ENCODING_NULLABLE_BOOLEAN:
            values = _load_array(root / "values.npy", label=f"{label}.values")
            mask = _load_mask(root / "mask.npy", length=length, label=label)
            if values.ndim != 1 or values.dtype != np.dtype(np.bool_):
                raise StoreCorrupt(f"Stored nullable boolean {label!r} has bad values.")
            if len(values) != length or values[mask].any():
                raise StoreCorrupt(
                    f"Stored nullable boolean {label!r} has noncanonical null bits."
                )
            series = pd.Series(pd.arrays.BooleanArray(values, mask, copy=False))
        elif encoding == _ENCODING_NULLABLE_INTEGER:
            values = _load_array(root / "values.npy", label=f"{label}.values")
            mask = _load_mask(root / "mask.npy", length=length, label=label)
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise StoreCorrupt(f"Stored nullable integer {label!r} has bad values.")
            if len(values) != length or (values[mask] != 0).any():
                raise StoreCorrupt(
                    f"Stored nullable integer {label!r} has noncanonical null bits."
                )
            array = pd.array(values, dtype=pandas_dtype)
            array[mask] = pd.NA
            series = pd.Series(array)
        elif encoding == _ENCODING_UTF8:
            payload = _load_array(root / "values.npy", label=f"{label}.values")
            offsets = _load_array(root / "offsets.npy", label=f"{label}.offsets")
            mask = _load_mask(root / "mask.npy", length=length, label=label)
            if payload.ndim != 1 or payload.dtype != np.dtype(np.uint8):
                raise StoreCorrupt(f"Stored UTF-8 payload {label!r} is not bytes.")
            _validate_offsets(offsets, len(payload), label=label)
            if len(offsets) != length + 1:
                raise StoreCorrupt(f"Stored UTF-8 offsets {label!r} have bad length.")
            raw = payload.tobytes()
            objects = np.empty(length, dtype=object)
            for index, (start, stop) in enumerate(
                zip(offsets[:-1], offsets[1:], strict=True)
            ):
                if mask[index]:
                    if start != stop:
                        raise StoreCorrupt(
                            f"Stored UTF-8 null {label}[{index}] carries hidden bytes."
                        )
                    objects[index] = pd.NA
                else:
                    try:
                        objects[index] = raw[int(start) : int(stop)].decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise StoreCorrupt(
                            f"Stored UTF-8 value {label}[{index}] is invalid."
                        ) from error
            series = pd.Series(objects, dtype=_string_dtype(spec))
        elif encoding == _ENCODING_OBJECT:
            payload = _load_array(root / "values.npy", label=f"{label}.values")
            offsets = _load_array(root / "offsets.npy", label=f"{label}.offsets")
            if payload.ndim != 1 or payload.dtype != np.dtype(np.uint8):
                raise StoreCorrupt(f"Stored object payload {label!r} is not bytes.")
            objects = _decode_object_chunks(offsets, payload, label=label)
            series = pd.Series(objects, dtype=object)
        else:
            raise StoreUnavailable(
                f"Stored series {label!r} needs unavailable encoding {encoding!r}."
            )
    except StoreError:
        raise
    except ImportError as error:
        raise StoreUnavailable(
            f"A dependency needed to restore {label!r} is unavailable."
        ) from error
    except (TypeError, ValueError, OverflowError) as error:
        raise StoreCorrupt(
            f"Stored series {label!r} cannot be reconstructed."
        ) from error
    if len(series) != length or str(series.dtype) != pandas_dtype:
        raise StoreCorrupt(
            f"Stored series {label!r} restored as {series.dtype!s}/{len(series)}, "
            f"not {pandas_dtype}/{length}."
        )
    return series


def _load_mask(path: Path, *, length: int, label: str) -> np.ndarray:
    mask = _load_array(path, label=f"{label}.mask")
    if mask.ndim != 1 or mask.dtype != np.dtype(np.bool_) or len(mask) != length:
        raise StoreCorrupt(f"Stored null mask for {label!r} is malformed.")
    return mask


def _axis_name_payload(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Axis names must be finite.")
        return value
    if isinstance(value, tuple):
        return {"tuple": [_axis_name_payload(item) for item in value]}
    raise TypeError(f"Unsupported axis-name type {type(value).__name__}.")


def _axis_name_from_payload(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict) and set(value) == {"tuple"}:
        items = value["tuple"]
        if isinstance(items, list):
            return tuple(_axis_name_from_payload(item) for item in items)
    raise StoreCorrupt("Stored axis name is malformed.")


def _write_index(root: Path, index: pd.Index) -> dict[str, object]:
    if isinstance(index, pd.MultiIndex):
        raise TypeError("ContentStore does not support MultiIndex.")
    base: dict[str, object] = {
        "dtype": str(index.dtype),
        "name": _axis_name_payload(index.name),
    }
    if isinstance(index, pd.RangeIndex):
        return {
            **base,
            "kind": "range",
            "start": index.start,
            "stop": index.stop,
            "step": index.step,
        }
    return {
        **base,
        "kind": "values",
        "series": _write_series(root, pd.Series(index.array, copy=False)),
    }


def _read_index(root: Path, spec: Mapping[str, Any], *, label: str) -> pd.Index:
    name = _axis_name_from_payload(spec.get("name"))
    dtype = spec.get("dtype")
    if not isinstance(dtype, str):
        raise StoreCorrupt(f"Stored index {label!r} has no dtype.")
    if spec.get("kind") == "range":
        try:
            result: pd.Index = pd.RangeIndex(
                start=spec["start"], stop=spec["stop"], step=spec["step"], name=name
            )
        except (KeyError, TypeError, ValueError) as error:
            raise StoreCorrupt(f"Stored RangeIndex {label!r} is malformed.") from error
    elif spec.get("kind") == "values":
        series_spec = spec.get("series")
        if not isinstance(series_spec, dict):
            raise StoreCorrupt(f"Stored index {label!r} has no series spec.")
        series = _read_series(root, series_spec, label=label)
        try:
            result = pd.Index(series.array, dtype=series.dtype, name=name)
        except (TypeError, ValueError) as error:
            raise StoreCorrupt(f"Stored index {label!r} cannot be restored.") from error
    else:
        raise StoreCorrupt(f"Stored index {label!r} has an unknown kind.")
    if str(result.dtype) != dtype:
        raise StoreCorrupt(
            f"Stored index {label!r} restored as {result.dtype!s}, not {dtype}."
        )
    return result


def _serialized_series_hash(root: Path, spec: Mapping[str, object]) -> str:
    digest = hashlib.sha256(b"microcosm-graph/serialized-series/1\0")
    encoded_spec = _canonical_json(dict(spec))
    digest.update(len(encoded_spec).to_bytes(8, "little"))
    digest.update(encoded_spec)
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


class ContentStore:
    """A filesystem content store rooted at ``root``."""

    def __init__(self, root: Path, *, codecs: object | None = None) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.tmp = self.root / "tmp"
        if codecs is not None and not (
            isinstance(codecs, Mapping) or callable(getattr(codecs, "get", None))
        ):
            raise TypeError("codecs must be a mapping, codec registry, or None.")
        self.codecs = codecs
        self.objects.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)

    def object_path(self, key: str) -> Path:
        """Return the canonical directory for ``key`` without reading it."""

        key = _require_key(key)
        return self.objects / key[:2] / key

    def has(self, key: str) -> bool:
        """Whether an object path is visible.

        A malformed visible directory is still a hit; its loader raises
        :class:`StoreCorrupt` rather than silently treating corruption as a miss.
        """

        return self.object_path(key).exists()

    contains = has

    def metadata(self, key: str, *, kind: str | None = None) -> Mapping[str, Any]:
        """Return validated object metadata."""

        return _verified_meta(self.object_path(key), expected_kind=kind)

    def _put(
        self,
        key: str,
        kind: str,
        build: Callable[[Path], Mapping[str, object]],
        *,
        verify_existing: bool = True,
    ) -> Path:
        key = _require_key(key)
        destination = self.object_path(key)
        if verify_existing and destination.exists():
            _verified_meta(destination, expected_kind=kind)
            return destination
        staging = self.tmp / uuid.uuid4().hex
        staging.mkdir(parents=False, exist_ok=False)
        try:
            details = dict(build(staging))
            reserved = {"format", "key", "kind", "payloads"}.intersection(details)
            if reserved:
                raise ValueError(f"Store object details use reserved keys: {reserved}.")
            metadata = {
                "format": _STORE_FORMAT,
                "key": key,
                "kind": kind,
                **details,
                "payloads": _payload_table(staging),
            }
            _write_json(staging / "meta.json", metadata)
            _fsync_tree(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _fsync_directory(destination.parent)
            try:
                os.replace(staging, destination)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                if verify_existing:
                    if not destination.exists():
                        raise
                    _verified_meta(destination, expected_kind=kind)
            _fsync_directory(destination.parent)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def put_column(
        self,
        key: str,
        series: pd.Series,
        *,
        declared_dtype: str,
        entity_ids: pd.Index | pd.Series | np.ndarray | None = None,
        node_key: str | None = None,
        verify_existing: bool = True,
    ) -> Path:
        """Atomically store one declared graph column.

        ``entity_ids`` defaults to the Series index and is stored separately so
        cache hits recover the exact id-indexed result.  Its serialized hash is
        recorded in ``meta.json``.
        """

        if declared_dtype not in _DECLARED_DTYPES:
            raise ValueError(f"Unknown declared dtype token {declared_dtype!r}.")
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"series must be a pandas Series, got {type(series).__name__}."
            )
        if not _dtype_matches_declared(series.dtype, declared_dtype):
            raise TypeError(
                f"Column dtype {series.dtype!s} is not declared dtype {declared_dtype!r}."
            )
        if entity_ids is None:
            ids = series.index
        elif isinstance(entity_ids, pd.Series):
            ids = pd.Index(entity_ids.array, name=entity_ids.name)
        else:
            ids = pd.Index(entity_ids)
        if len(ids) != len(series):
            raise ValueError("entity_ids must have one value per stored column row.")
        bound_node_key = key if node_key is None else node_key

        def build(root: Path) -> Mapping[str, object]:
            value_spec = _write_series(root, series.reset_index(drop=True))
            ids_spec = _write_index(root / "ids", ids)
            entity_hash = _serialized_series_hash(
                root / "ids",
                ids_spec["series"] if ids_spec["kind"] == "values" else ids_spec,
            )
            return {
                "declared_dtype": declared_dtype,
                "pandas_dtype": str(series.dtype),
                "length": len(series),
                "entity_id_hash": entity_hash,
                "node_key": bound_node_key,
                "values": value_spec,
                "ids": ids_spec,
            }

        return self._put(key, "column", build, verify_existing=verify_existing)

    write_column = put_column

    def load_column(
        self,
        key: str,
        *,
        declared_dtype: str | None = None,
        entity_ids: pd.Index | pd.Series | np.ndarray | None = None,
        node_key: str | None = None,
    ) -> pd.Series:
        """Load and validate an id-indexed stored column."""

        path = self.object_path(key)
        metadata = _verified_meta(path, expected_kind="column")
        stored_dtype = metadata.get("declared_dtype")
        if stored_dtype not in _DECLARED_DTYPES:
            raise StoreCorrupt(f"Stored column {key} has an invalid dtype token.")
        if declared_dtype is not None and stored_dtype != declared_dtype:
            raise StoreCorrupt(
                f"Stored column {key} declares {stored_dtype!r}, not {declared_dtype!r}."
            )
        if node_key is not None and metadata.get("node_key") != node_key:
            raise StoreCorrupt(f"Stored column {key} belongs to a different node.")
        value_spec = metadata.get("values")
        ids_spec = metadata.get("ids")
        if not isinstance(value_spec, dict) or not isinstance(ids_spec, dict):
            raise StoreCorrupt(f"Stored column {key} lacks series metadata.")
        values = _read_series(path, value_spec, label=f"column {key}")
        ids = _read_index(path / "ids", ids_spec, label=f"column {key} ids")
        expected_hash = metadata.get("entity_id_hash")
        actual_hash = _serialized_series_hash(
            path / "ids",
            ids_spec["series"] if ids_spec.get("kind") == "values" else ids_spec,
        )
        if expected_hash != actual_hash:
            raise StoreCorrupt(f"Stored column {key} entity-id hash is invalid.")
        if len(values) != len(ids) or metadata.get("length") != len(values):
            raise StoreCorrupt(f"Stored column {key} values and ids do not align.")
        if str(values.dtype) != metadata.get(
            "pandas_dtype"
        ) or not _dtype_matches_declared(values.dtype, stored_dtype):
            raise StoreCorrupt(f"Stored column {key} did not preserve its dtype.")
        if entity_ids is not None:
            if isinstance(entity_ids, pd.Series):
                expected_ids = pd.Index(entity_ids.array, name=entity_ids.name)
            else:
                expected_ids = pd.Index(entity_ids)
            if not ids.identical(expected_ids):
                raise StoreCorrupt(f"Stored column {key} has different entity ids.")
        values.index = ids
        return values

    read_column = load_column

    def put_frame(
        self,
        key: str,
        frame: Frame,
        *,
        node_key: str | None = None,
        verify_existing: bool = True,
    ) -> Path:
        """Atomically persist a complete Frame population version."""

        if not isinstance(frame, Frame):
            raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
        frame.revalidate()
        bound_node_key = key if node_key is None else node_key

        def build(root: Path) -> Mapping[str, object]:
            _write_frame(root, frame)
            return {"frame_format": _FRAME_FORMAT, "node_key": bound_node_key}

        return self._put(key, "frame", build, verify_existing=verify_existing)

    write_frame = put_frame

    def load_frame(self, key: str, *, node_key: str | None = None) -> Frame:
        """Load a complete, content-verified Frame population version."""

        path = self.object_path(key)
        metadata = _verified_meta(path, expected_kind="frame")
        if node_key is not None and metadata.get("node_key") != node_key:
            raise StoreCorrupt(f"Stored frame {key} belongs to a different node.")
        return _read_frame(path, metadata)

    read_frame = load_frame

    @staticmethod
    def load_frame_path(path: Path) -> Frame:
        """Load a frame from its object directory (the ``frame-store`` codec)."""

        object_path = Path(path)
        metadata = _verified_meta(object_path, expected_kind="frame")
        return _read_frame(object_path, metadata)

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        node_key: str | None = None,
        verify_existing: bool = True,
    ) -> Path:
        """Store one opaque, content-validated byte artifact."""

        if not isinstance(payload, bytes):
            raise TypeError("Opaque store payloads must be bytes.")

        def build(root: Path) -> Mapping[str, object]:
            _write_bytes(root / "payload.bin", payload)
            return {"node_key": key if node_key is None else node_key}

        return self._put(key, "bytes", build, verify_existing=verify_existing)

    def load_bytes(self, key: str) -> bytes:
        """Load one opaque byte artifact."""

        path = self.object_path(key)
        _verified_meta(path, expected_kind="bytes")
        try:
            return (path / "payload.bin").read_bytes()
        except OSError as error:  # verified above; protects a concurrent removal
            raise StoreCorrupt(f"Stored byte payload {key} disappeared.") from error

    def put_json(
        self,
        key: str,
        payload: Mapping[str, object],
        *,
        kind: str = "json",
        node_key: str | None = None,
        verify_existing: bool = True,
    ) -> Path:
        """Store a canonical JSON object, useful for node receipts/indexes."""

        if not isinstance(payload, Mapping):
            raise TypeError("JSON store payloads must be mappings.")
        normalized = json.loads(_canonical_json(dict(payload)))

        def build(root: Path) -> Mapping[str, object]:
            _write_json(root / "payload.json", normalized)
            return {"node_key": key if node_key is None else node_key}

        return self._put(key, kind, build, verify_existing=verify_existing)

    def load_json(self, key: str, *, kind: str = "json") -> dict[str, object]:
        """Load a canonical JSON object."""

        path = self.object_path(key)
        _verified_meta(path, expected_kind=kind)
        payload = _load_json_file(path / "payload.json", label=f"JSON payload {key}")
        if not isinstance(payload, dict):
            raise StoreCorrupt(f"Stored JSON payload {key} is not an object.")
        return payload


def _schema_payload(schema: EntitySchema) -> dict[str, object]:
    return {
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
    }


def _write_frame(root: Path, frame: Frame) -> None:
    _write_json(root / "schema.json", _schema_payload(frame.schema))
    table_specs: list[dict[str, object]] = []
    table_rows = [
        *(("entity", entity, frame.table(entity)) for entity in frame.entities),
        *(("link", name, frame.link(name)) for name in frame.links),
    ]
    for table_index, (role, name, table) in enumerate(table_rows):
        if any(not isinstance(column, str) for column in table.columns):
            raise TypeError(f"Frame table {name!r} must use string column names.")
        table_root = root / "tables" / f"t{table_index:05d}"
        columns: list[dict[str, object]] = []
        for column_index, column in enumerate(table.columns):
            series_spec = _write_series(
                table_root / "columns" / f"c{column_index:05d}", table[column]
            )
            columns.append({"name": column, "series": series_spec})
        table_specs.append(
            {
                "name": name,
                "role": role,
                "index": _write_index(table_root / "index", table.index),
                "columns_name": _axis_name_payload(table.columns.name),
                "columns": columns,
            }
        )

    weight_specs: list[dict[str, object]] = []
    for weight_index, entity in enumerate(frame.weighted_entities):
        weights = frame.weights_for(entity)
        filename = f"weights/w{weight_index:05d}.npy"
        _write_array(root / filename, weights.values)
        weight_specs.append(
            {"entity": entity, "kind": weights.kind.value, "file": filename}
        )
    strata_spec = _write_series(root / "strata", frame.strata)
    mass_log = [
        {
            "entity": record.entity,
            "old_total": record.old_total,
            "new_total": record.new_total,
            "declared_factor": record.declared_factor,
            "reason": record.reason,
        }
        for record in frame.mass_log
    ]
    _write_json(
        root / "frame.json",
        {
            "format": _FRAME_FORMAT,
            "tables": table_specs,
            "weights": weight_specs,
            "strata": strata_spec,
            "mass_log": mass_log,
        },
    )


def _read_schema(path: Path) -> EntitySchema:
    payload = _load_json_file(path / "schema.json", label="frame schema")
    if not isinstance(payload, dict):
        raise StoreCorrupt("Stored frame schema is not an object.")
    groups = payload.get("group_entities")
    links_payload = payload.get("links")
    if not isinstance(groups, list) or not isinstance(links_payload, list):
        raise StoreCorrupt("Stored frame schema groups/links are malformed.")
    try:
        links = tuple(
            LinkSpec(
                name=item["name"],
                left_entity=item["left_entity"],
                right_entity=item["right_entity"],
            )
            for item in links_payload
            if isinstance(item, dict)
        )
        if len(links) != len(links_payload):
            raise ValueError("non-object link")
        return EntitySchema(
            person_entity=payload["person_entity"],
            group_entities=tuple(groups),
            links=links,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StoreCorrupt("Stored frame schema is invalid.") from error


def _read_frame(path: Path, metadata: Mapping[str, Any]) -> Frame:
    if metadata.get("frame_format") != _FRAME_FORMAT:
        raise StoreUnavailable(
            f"Stored frame requires unavailable codec {metadata.get('frame_format')!r}."
        )
    schema = _read_schema(path)
    manifest = _load_json_file(path / "frame.json", label="frame manifest")
    if not isinstance(manifest, dict) or manifest.get("format") != _FRAME_FORMAT:
        raise StoreCorrupt("Stored frame manifest is invalid.")
    raw_tables = manifest.get("tables")
    raw_weights = manifest.get("weights")
    strata_spec = manifest.get("strata")
    raw_mass_log = manifest.get("mass_log")
    if not all(
        (
            isinstance(raw_tables, list),
            isinstance(raw_weights, list),
            isinstance(strata_spec, dict),
            isinstance(raw_mass_log, list),
        )
    ):
        raise StoreCorrupt("Stored frame manifest fields are malformed.")

    tables: dict[str, pd.DataFrame] = {}
    roles: dict[str, str] = {}
    for table_index, raw_table in enumerate(raw_tables):
        if not isinstance(raw_table, dict):
            raise StoreCorrupt("Stored frame table spec is not an object.")
        name = raw_table.get("name")
        role = raw_table.get("role")
        index_spec = raw_table.get("index")
        raw_columns = raw_table.get("columns")
        if (
            not isinstance(name, str)
            or role not in {"entity", "link"}
            or not isinstance(index_spec, dict)
            or not isinstance(raw_columns, list)
            or name in tables
        ):
            raise StoreCorrupt("Stored frame table metadata is malformed.")
        table_root = path / "tables" / f"t{table_index:05d}"
        index = _read_index(table_root / "index", index_spec, label=f"{name}.index")
        columns: dict[str, pd.Series] = {}
        for column_index, raw_column in enumerate(raw_columns):
            if not isinstance(raw_column, dict):
                raise StoreCorrupt("Stored frame column spec is not an object.")
            column = raw_column.get("name")
            series_spec = raw_column.get("series")
            if (
                not isinstance(column, str)
                or not isinstance(series_spec, dict)
                or column in columns
            ):
                raise StoreCorrupt("Stored frame column metadata is malformed.")
            series = _read_series(
                table_root / "columns" / f"c{column_index:05d}",
                series_spec,
                label=f"{name}.{column}",
            )
            if len(series) != len(index):
                raise StoreCorrupt(
                    f"Stored frame column {name}.{column} is misaligned."
                )
            series.index = index
            columns[column] = series
        table = pd.DataFrame(columns, index=index, copy=False)
        table.columns.name = _axis_name_from_payload(raw_table.get("columns_name"))
        tables[name] = table
        roles[name] = role

    expected_entities = set(schema.entities)
    expected_links = {link.name for link in schema.links}
    if {name for name, role in roles.items() if role == "entity"} != expected_entities:
        raise StoreCorrupt("Stored frame entity tables do not match its schema.")
    if {name for name, role in roles.items() if role == "link"} != expected_links:
        raise StoreCorrupt("Stored frame link tables do not match its schema.")

    weights: dict[str, Weights] = {}
    for raw_weight in raw_weights:
        if not isinstance(raw_weight, dict):
            raise StoreCorrupt("Stored frame weight spec is not an object.")
        entity = raw_weight.get("entity")
        kind_value = raw_weight.get("kind")
        filename = raw_weight.get("file")
        if (
            not isinstance(entity, str)
            or not isinstance(kind_value, str)
            or not isinstance(filename, str)
            or entity in weights
            or entity not in tables
        ):
            raise StoreCorrupt("Stored frame weight metadata is malformed.")
        try:
            kind = WeightKind(kind_value)
        except ValueError as error:
            raise StoreCorrupt(
                f"Stored frame has unknown weight kind {kind_value!r}."
            ) from error
        values = _load_array(
            path / _safe_payload_name(filename), label=f"{entity} weights"
        )
        if values.ndim != 1 or values.dtype != np.dtype(np.float64):
            raise StoreCorrupt(
                f"Stored weights for {entity!r} are not float64 vectors."
            )
        try:
            weights[entity] = Weights(values, kind)
        except (TypeError, ValueError) as error:
            raise StoreCorrupt(f"Stored weights for {entity!r} are invalid.") from error

    strata = _read_series(path / "strata", strata_spec, label="strata")
    person_index = tables[schema.person_entity].index
    if len(strata) != len(person_index):
        raise StoreCorrupt("Stored frame strata do not align to persons.")
    strata.index = person_index
    strata.name = "stratum"

    mass_log: list[MassChangeRecord] = []
    for raw_record in raw_mass_log:
        if not isinstance(raw_record, dict):
            raise StoreCorrupt("Stored frame mass record is not an object.")
        try:
            entity = raw_record["entity"]
            reason = raw_record["reason"]
            old_total = float(raw_record["old_total"])
            new_total = float(raw_record["new_total"])
            declared = raw_record.get("declared_factor")
            declared_factor = None if declared is None else float(declared)
            if not isinstance(entity, str) or not isinstance(reason, str):
                raise TypeError
            if not math.isfinite(old_total) or not math.isfinite(new_total):
                raise ValueError
            if declared_factor is not None and not math.isfinite(declared_factor):
                raise ValueError
            mass_log.append(
                MassChangeRecord(
                    entity=entity,
                    old_total=old_total,
                    new_total=new_total,
                    declared_factor=declared_factor,
                    reason=reason,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise StoreCorrupt("Stored frame mass record is malformed.") from error
    try:
        return Frame(
            tables,
            schema,
            weights,
            strata,
            mass_log=tuple(mass_log),
        )
    except ImportError as error:
        raise StoreUnavailable(
            "A dependency needed to restore the Frame is missing."
        ) from error
    except (TypeError, ValueError) as error:
        raise StoreCorrupt(
            f"Stored Frame violates Frame invariants: {error}"
        ) from error
