"""Check that a pooled computation gives byte-identical results at any worker count.

A caller wraps its pooled computation in ``run(workers) -> result``, where the
result is whatever the caller joins from the pool (arrays, totals, payloads),
and calls :func:`assert_identical_across_worker_counts`. The helper runs it at
each worker count and compares canonical bytes, not values:

* arrays by type, dtype (with its byte order and any fields), shape and raw
  bytes, so ``0.0`` and ``-0.0``, NaNs with different bits, and ``int32`` and
  ``int64`` all count as different; masked arrays by their mask too;
* every other value with its type, and floats by their IEEE 754 bits;
* mappings with their keys in order, since a payload written to JSON keeps
  that order;
* tuples, lists, dataclasses and pandas objects by type and contents, pandas
  objects with their dtypes (a categorical's categories) and index names.

A difference is reported by its place in the result, such as
``result['totals'][3]``.
"""

from __future__ import annotations

import dataclasses
import enum
import struct
from collections.abc import Callable, Mapping, Sequence

import numpy as np

__all__ = ["assert_identical_across_worker_counts", "canonical_result_bytes"]


def _framed(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">Q", len(payload)) + payload


def _type_name(value: object) -> bytes:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}".encode()


def _dtype_bytes(dtype: np.dtype) -> bytes:
    """A dtype with its byte order and, for structured dtypes, its fields.

    ``dtype.str`` alone is ``|V8`` for every 8-byte structured dtype, so it
    would lose field names, field types and their byte order.
    """
    fields = dtype.descr if dtype.fields is not None or dtype.subdtype else None
    # The character code tells apart types with the same layout, such as
    # ``longlong`` and ``int64`` where they differ.
    return repr((dtype.str, dtype.char, fields)).encode()


def _pandas_values(value) -> object:
    """The array behind a Series or Index: NumPy, or a pandas extension array."""
    if isinstance(value.dtype, np.dtype):
        return value.to_numpy()
    return value.array


def _extension_array_bytes(values) -> bytes:
    """A pandas extension array, with its missing entries kept apart from NaN."""
    try:
        return _extension_array_bytes_or_raise(values)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "canonical_result_bytes has no encoding for pandas values of dtype "
            f"{values.dtype!r}: {error}"
        ) from error


def _subclass_state_bytes(value: object, base: type) -> bytes:
    """What a subclass of ``base`` keeps beyond the base's own data.

    A pandas subclass lists it in ``_metadata`` (policyengine-core's
    ``MicroSeries`` keeps its weights there); an ``ndarray`` subclass keeps it
    in its instance dictionary (``EnumArray`` keeps its enum class there).
    """
    if type(value) is base:
        return b""
    if base is np.ndarray:
        state = dict(sorted(getattr(value, "__dict__", {}).items()))
    else:
        own = set(getattr(base, "_metadata", ()))
        state = {
            name: getattr(value, name, None)
            for name in sorted(getattr(type(value), "_metadata", ()))
            if name not in own
        }
    return _framed(b"z", canonical_result_bytes(state))


def _extension_array_bytes_or_raise(values) -> bytes:
    import pandas as pd

    kind = _framed(b"t", _type_name(values)) + _framed(
        b"d", repr(values.dtype).encode()
    )
    if isinstance(values, pd.Categorical):
        # Explicit categories: a dtype's repr elides them beyond 100.
        return _framed(
            b"c",
            kind
            + canonical_result_bytes(values.categories)
            + canonical_result_bytes(bool(values.ordered))
            + canonical_result_bytes(np.asarray(values.codes)),
        )
    missing = canonical_result_bytes(np.asarray(values.isna(), dtype=bool))
    if hasattr(values, "asi8"):  # Datetime, timedelta and period arrays.
        return _framed(
            b"x", kind + missing + canonical_result_bytes(np.asarray(values.asi8))
        )
    numpy_dtype = getattr(values.dtype, "numpy_dtype", None)
    if numpy_dtype is not None:  # Nullable integers, floats and booleans.
        present = values.to_numpy(dtype=numpy_dtype, na_value=numpy_dtype.type(0))
    else:
        present = values.to_numpy(dtype=object, na_value=None)
    return _framed(b"x", kind + missing + canonical_result_bytes(np.asarray(present)))


def _values_bytes(values) -> bytes:
    if isinstance(values, np.ndarray):
        return canonical_result_bytes(values)
    return _extension_array_bytes(values)


def _pandas_bytes(value: object) -> bytes | None:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a workspace dependency
        return None
    kind = _framed(b"t", _type_name(value))
    if isinstance(value, pd.DataFrame):
        parts = [
            canonical_result_bytes(value.columns),
            canonical_result_bytes(value.index),
            canonical_result_bytes(dict(value.attrs)),
            _subclass_state_bytes(value, pd.DataFrame),
        ]
        for position in range(value.shape[1]):
            parts.append(canonical_result_bytes(value.iloc[:, position]))
        return _framed(b"P", kind + b"".join(parts))
    if isinstance(value, pd.Series):
        return _framed(
            b"P",
            kind
            + canonical_result_bytes(value.name)
            + canonical_result_bytes(repr(value.dtype))
            + canonical_result_bytes(value.index)
            + canonical_result_bytes(dict(value.attrs))
            + _subclass_state_bytes(value, pd.Series)
            + _values_bytes(_pandas_values(value)),
        )
    if isinstance(value, pd.MultiIndex):
        # Each level's values in index order: its levels and codes would also
        # carry values a slice no longer uses.
        levels = b"".join(
            canonical_result_bytes(value.get_level_values(position))
            for position in range(value.nlevels)
        )
        return _framed(b"P", kind + canonical_result_bytes(list(value.names)) + levels)
    if isinstance(value, pd.Index):
        frequency = getattr(value, "freq", None)
        return _framed(
            b"P",
            kind
            + canonical_result_bytes(list(value.names))
            + canonical_result_bytes(repr(value.dtype))
            + canonical_result_bytes(None if frequency is None else str(frequency))
            + _values_bytes(_pandas_values(value)),
        )
    return None


def canonical_result_bytes(value: object) -> bytes:
    """Bytes that are equal exactly when two results are bit-for-bit the same.

    Every value carries its type, so ``True`` and ``np.True_``, an ``IntEnum``
    member and its ``int``, and ``bytes`` and ``bytearray`` differ. Structured
    arrays and records are encoded field by field, never by raw bytes, which
    would include padding and the addresses of object fields.

    Raises:
        TypeError: For a value of a type it has no encoding for.
    """
    if value is None:
        return b"N"
    if type(value) is bool:
        return b"B1" if value else b"B0"
    if isinstance(value, type):
        return _framed(b"T", f"{value.__module__}.{value.__qualname__}".encode())
    if isinstance(value, enum.Enum):
        return _framed(
            b"G",
            _framed(b"t", _type_name(value))
            + canonical_result_bytes(value.name)
            + canonical_result_bytes(value.value),
        )
    if isinstance(value, np.ndarray):
        if isinstance(value, np.ma.MaskedArray):
            # Its instance dictionary is internal state; the data, mask and
            # fill value are what it holds.
            return _framed(
                b"K",
                _framed(b"t", _type_name(value))
                + canonical_result_bytes(np.asarray(value.data))
                + canonical_result_bytes(np.ma.getmaskarray(value))
                + canonical_result_bytes(np.asarray(value.fill_value)),
            )
        kind = _framed(b"t", _type_name(value)) + _subclass_state_bytes(
            value, np.ndarray
        )
        header = kind + _framed(b"d", _dtype_bytes(value.dtype))
        header += _framed(b"s", repr(value.shape).encode())
        if value.dtype.names is not None:
            fields = b"".join(
                canonical_result_bytes(name)
                + canonical_result_bytes(np.ascontiguousarray(value[name]))
                for name in value.dtype.names
            )
            return _framed(b"R", header + fields)
        if value.dtype.hasobject:
            items = b"".join(canonical_result_bytes(item) for item in value.ravel())
            return _framed(b"O", header + items)
        return _framed(b"A", header + np.ascontiguousarray(value).tobytes())
    if isinstance(value, np.generic):
        scalar = _framed(b"t", _type_name(value)) + _dtype_bytes(value.dtype)
        if value.dtype.names is not None:  # A record of a structured array.
            fields = b"".join(
                canonical_result_bytes(name) + canonical_result_bytes(value[name])
                for name in value.dtype.names
            )
            return _framed(b"V", scalar + fields)
        return _framed(b"S", scalar + b":" + np.asarray(value).tobytes())
    kind = _framed(b"t", _type_name(value))
    if isinstance(value, int):
        return _framed(b"I", kind + str(int(value)).encode())
    if isinstance(value, float):
        return _framed(b"F", kind + struct.pack(">d", value))
    if isinstance(value, complex):
        return _framed(b"C", kind + struct.pack(">dd", value.real, value.imag))
    if isinstance(value, str):
        return _framed(b"U", kind + str.encode(value, "utf-8", "surrogatepass"))
    if isinstance(value, memoryview):
        layout = repr((value.format, value.shape)).encode()
        return _framed(b"Y", kind + _framed(b"m", layout) + value.tobytes())
    if isinstance(value, bytes | bytearray):
        return _framed(b"Y", kind + bytes(value))
    pandas_bytes = _pandas_bytes(value)
    if pandas_bytes is not None:
        return pandas_bytes
    if isinstance(value, Mapping):
        items = b"".join(
            canonical_result_bytes(key) + canonical_result_bytes(item)
            for key, item in value.items()
        )
        return _framed(b"M", kind + items)
    if isinstance(value, tuple | list):
        items = b"".join(canonical_result_bytes(item) for item in value)
        return _framed(b"L", kind + items)
    if isinstance(value, set | frozenset):
        items = b"".join(sorted(canonical_result_bytes(item) for item in value))
        return _framed(b"E", kind + items)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        items = b"".join(
            canonical_result_bytes(field.name)
            + canonical_result_bytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
        return _framed(b"D", kind + items)
    raise TypeError(
        "canonical_result_bytes has no encoding for values of type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _differ(left: object, right: object) -> bool:
    return canonical_result_bytes(left) != canonical_result_bytes(right)


def _first_array_difference(left: np.ndarray, right: np.ndarray, place: str) -> str:
    if left.dtype != right.dtype or left.shape != right.shape:
        return f"{place} (dtype or shape)"
    if left.shape == ():
        return place
    if left.dtype.names is None and not left.dtype.hasobject:
        left_items = np.ascontiguousarray(left).reshape(-1).view(np.uint8)
        right_items = np.ascontiguousarray(right).reshape(-1).view(np.uint8)
        differing = np.flatnonzero(left_items != right_items)
        flat = (
            int(differing[0]) // max(left.dtype.itemsize, 1) if differing.size else None
        )
    else:
        flat = next(
            (
                position
                for position, (left_item, right_item) in enumerate(
                    zip(left.ravel(), right.ravel(), strict=True)
                )
                if _differ(left_item, right_item)
            ),
            None,
        )
    if flat is None:
        return place
    position = np.unravel_index(flat, left.shape)
    return f"{place}[{', '.join(str(int(axis)) for axis in position)}]"


def _first_pandas_difference(left, right, place: str) -> str | None:
    """The place two pandas objects first differ, or None if they aren't pandas."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a workspace dependency
        return None
    if isinstance(left, pd.DataFrame):
        for name, part in (("columns", "columns"), ("index", "index")):
            if _differ(getattr(left, part), getattr(right, part)):
                return f"{place}.{name}"
        if _differ(dict(left.attrs), dict(right.attrs)):
            return f"{place}.attrs"
        for position in range(left.shape[1]):
            left_column, right_column = left.iloc[:, position], right.iloc[:, position]
            if _differ(left_column, right_column):
                label = f"{place}[{left.columns[position]!r}]"
                return _first_difference(left_column, right_column, label)
        return place
    if isinstance(left, pd.Series):
        for name in ("index", "name", "attrs"):
            left_part, right_part = getattr(left, name), getattr(right, name)
            if name == "attrs":
                left_part, right_part = dict(left_part), dict(right_part)
            if _differ(left_part, right_part):
                return f"{place}.{name}"
        if repr(left.dtype) != repr(right.dtype):
            return f"{place}.dtype"
        if isinstance(left.dtype, np.dtype):
            located = _first_array_difference(
                left.to_numpy(), right.to_numpy(), f"{place}.iloc"
            )
            return place if located == f"{place}.iloc" else located
        return place
    if isinstance(left, pd.Index):
        return place
    return None


def _first_difference(left: object, right: object, place: str) -> str:
    """The place in two results where their canonical bytes first differ."""
    if type(left) is not type(right):
        return f"{place} (types {type(left).__name__} and {type(right).__name__})"
    pandas_place = _first_pandas_difference(left, right, place)
    if pandas_place is not None:
        return pandas_place
    if isinstance(left, Mapping):
        # Keys compare by their bytes: a NaN key is never == itself.
        if [canonical_result_bytes(key) for key in left] != [
            canonical_result_bytes(key) for key in right
        ]:
            return f"{place} (its keys or their order)"
        for (key, left_item), right_item in zip(
            left.items(), right.values(), strict=True
        ):
            if _differ(left_item, right_item):
                return _first_difference(left_item, right_item, f"{place}[{key!r}]")
        return place
    if isinstance(left, tuple | list):
        if len(left) != len(right):
            return f"{place} (lengths {len(left)} and {len(right)})"
        for position, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            if _differ(left_item, right_item):
                return _first_difference(left_item, right_item, f"{place}[{position}]")
        return place
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        for field in dataclasses.fields(left):
            left_item, right_item = (
                getattr(left, field.name),
                getattr(right, field.name),
            )
            if _differ(left_item, right_item):
                return _first_difference(left_item, right_item, f"{place}.{field.name}")
        return place
    if isinstance(left, np.ndarray) and not isinstance(left, np.ma.MaskedArray):
        return _first_array_difference(left, right, place)
    return place


def assert_identical_across_worker_counts(
    run: Callable[[int], object], *, worker_counts: Sequence[int] = (1, 3)
) -> object:
    """Run ``run(workers)`` at each worker count; require byte-identical results.

    Returns the result of the first worker count, so a test can check it
    further.

    Raises:
        AssertionError: Naming the worker count and the first place where its
            result differs from the first worker count's.
    """
    counts = tuple(worker_counts)
    if len(counts) < 2 or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in counts
    ):
        raise ValueError(
            f"worker_counts needs at least two positive worker counts; got {counts}."
        )
    reference = run(counts[0])
    reference_bytes = canonical_result_bytes(reference)
    for workers in counts[1:]:
        result = run(workers)
        if canonical_result_bytes(result) != reference_bytes:
            place = _first_difference(reference, result, "result")
            raise AssertionError(
                f"Results with workers={workers} differ from workers={counts[0]} "
                f"at {place}."
            )
    return reference
