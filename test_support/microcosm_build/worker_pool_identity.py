"""Check that a pooled computation gives byte-identical results at any worker count.

A caller wraps its pooled computation in ``run(workers) -> result``, where the
result is whatever the caller joins from the pool (arrays, totals, payloads),
and calls :func:`assert_identical_across_worker_counts`. The helper runs it at
each worker count and compares canonical bytes, not values:

* arrays by dtype, shape and raw bytes, so ``0.0`` and ``-0.0``, NaNs with
  different bits, and ``int32`` and ``int64`` all count as different;
* floats by their IEEE 754 bits;
* mappings with their keys in order, since a payload written to JSON keeps
  that order;
* tuples, lists, dataclasses and pandas objects by type and contents.

A difference is reported by its place in the result, such as
``result['totals'][3]``.
"""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Callable, Mapping, Sequence

import numpy as np

__all__ = ["assert_identical_across_worker_counts", "canonical_result_bytes"]


def _framed(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">Q", len(payload)) + payload


def _type_name(value: object) -> bytes:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}".encode()


def _pandas_bytes(value: object) -> bytes | None:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a workspace dependency
        return None
    if isinstance(value, pd.DataFrame):
        parts = [
            canonical_result_bytes(list(value.columns)),
            canonical_result_bytes(value.index),
        ]
        for column in value.columns:
            parts.append(canonical_result_bytes(value[column]))
        return _framed(b"P", _type_name(value) + b"".join(parts))
    if isinstance(value, pd.Series):
        return _framed(
            b"P",
            _type_name(value)
            + canonical_result_bytes(value.name)
            + canonical_result_bytes(str(value.dtype))
            + canonical_result_bytes(value.index)
            + canonical_result_bytes(np.asarray(value.to_numpy())),
        )
    if isinstance(value, pd.Index):
        return _framed(
            b"P",
            _type_name(value)
            + canonical_result_bytes(str(value.dtype))
            + canonical_result_bytes(np.asarray(value.to_numpy())),
        )
    return None


def canonical_result_bytes(value: object) -> bytes:
    """Bytes that are equal exactly when two results are bit-for-bit the same.

    Raises:
        TypeError: For a value of a type it has no encoding for.
    """
    if value is None:
        return b"N"
    if isinstance(value, bool | np.bool_):
        return b"B1" if value else b"B0"
    if isinstance(value, np.ndarray):
        header = f"{value.dtype.str}{value.shape}".encode()
        if value.dtype.hasobject:
            items = b"".join(canonical_result_bytes(item) for item in value.ravel())
            return _framed(b"O", _framed(b"h", header) + items)
        return _framed(
            b"A", _framed(b"h", header) + np.ascontiguousarray(value).tobytes()
        )
    if isinstance(value, np.generic):
        return _framed(b"S", value.dtype.str.encode() + b":" + value.tobytes())
    if isinstance(value, int):
        return _framed(b"I", str(value).encode())
    if isinstance(value, float):
        return _framed(b"F", struct.pack(">d", value))
    if isinstance(value, complex):
        return _framed(b"C", struct.pack(">dd", value.real, value.imag))
    if isinstance(value, str):
        return _framed(b"U", value.encode("utf-8", "surrogatepass"))
    if isinstance(value, bytes | bytearray | memoryview):
        return _framed(b"Y", bytes(value))
    pandas_bytes = _pandas_bytes(value)
    if pandas_bytes is not None:
        return pandas_bytes
    if isinstance(value, Mapping):
        items = b"".join(
            canonical_result_bytes(key) + canonical_result_bytes(item)
            for key, item in value.items()
        )
        return _framed(b"M", _framed(b"t", _type_name(value)) + items)
    if isinstance(value, tuple | list):
        items = b"".join(canonical_result_bytes(item) for item in value)
        return _framed(b"L", _framed(b"t", _type_name(value)) + items)
    if isinstance(value, set | frozenset):
        items = b"".join(sorted(canonical_result_bytes(item) for item in value))
        return _framed(b"E", _framed(b"t", _type_name(value)) + items)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        items = b"".join(
            canonical_result_bytes(field.name)
            + canonical_result_bytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
        return _framed(b"D", _framed(b"t", _type_name(value)) + items)
    raise TypeError(
        "canonical_result_bytes has no encoding for values of type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _first_difference(left: object, right: object, place: str) -> str:
    """The place in two results where their canonical bytes first differ."""
    if type(left) is not type(right):
        return f"{place} (types {type(left).__name__} and {type(right).__name__})"
    if isinstance(left, Mapping):
        if list(left) != list(right):
            return f"{place} (its keys or their order)"
        for key in left:
            if canonical_result_bytes(left[key]) != canonical_result_bytes(right[key]):
                return _first_difference(left[key], right[key], f"{place}[{key!r}]")
        return place
    if isinstance(left, tuple | list):
        if len(left) != len(right):
            return f"{place} (lengths {len(left)} and {len(right)})"
        for position, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            if canonical_result_bytes(left_item) != canonical_result_bytes(right_item):
                return _first_difference(left_item, right_item, f"{place}[{position}]")
        return place
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        for field in dataclasses.fields(left):
            left_item, right_item = (
                getattr(left, field.name),
                getattr(right, field.name),
            )
            if canonical_result_bytes(left_item) != canonical_result_bytes(right_item):
                return _first_difference(left_item, right_item, f"{place}.{field.name}")
        return place
    if (
        isinstance(left, np.ndarray)
        and left.dtype == right.dtype
        and left.shape == right.shape
        and not left.dtype.hasobject
    ):
        left_bytes = np.ascontiguousarray(left).reshape(-1).view(np.uint8)
        right_bytes = np.ascontiguousarray(right).reshape(-1).view(np.uint8)
        differing = np.flatnonzero(left_bytes != right_bytes)
        if differing.size:
            flat = int(differing[0]) // max(left.dtype.itemsize, 1)
            position = np.unravel_index(flat, left.shape) if left.shape else ()
            return f"{place}[{', '.join(str(int(axis)) for axis in position)}]"
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
