"""Whole-series selection parity for the physical storage parts.

Invented series only: no data file, no store, no archive, no engine.  Every
case compares the live ``_storage_parts`` against ``_reference_storage_parts``
below — a verbatim copy of the pre-change body — in both selector forms, and
every mutation case requires visible *and* hidden bytes to move the parts.
``slice(None)`` and an all-True mask must be byte-identical; a subset mask must
keep behaving exactly as it did.
"""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from microcosm.graph import population as population_ops
from microcosm.graph.population import (
    PopulationError,
    _object_storage_values,
    _storage_parts,
    storage_equal,
)


def _reference_storage_parts(series, selected):
    """The exact pre-change body, kept verbatim as the parity reference."""

    nulls = series.isna().to_numpy(dtype=np.bool_, copy=False)[selected]
    array = series.array
    data = getattr(array, "_data", None)
    mask = getattr(array, "_mask", None)
    if isinstance(data, np.ndarray) and isinstance(mask, np.ndarray):
        values = np.ascontiguousarray(data[selected]).tobytes()
        bitmap = np.ascontiguousarray(mask[selected]).tobytes()
        return values, bitmap
    if isinstance(series.dtype, pd.StringDtype):
        payload = bytearray()
        for value, is_null in zip(
            series.to_numpy(dtype=object, copy=False)[selected], nulls, strict=True
        ):
            if is_null:
                payload.extend((0).to_bytes(8, "little"))
            else:
                encoded = str(value).encode("utf-8")
                payload.extend((len(encoded) + 1).to_bytes(8, "little"))
                payload.extend(encoded)
        return bytes(payload), np.ascontiguousarray(nulls).tobytes()
    values = series.to_numpy(copy=False)[selected]
    if values.dtype == object:
        return (
            _object_storage_values(values),
            np.ascontiguousarray(nulls).tobytes(),
        )
    return (
        np.ascontiguousarray(values).tobytes(),
        np.ascontiguousarray(nulls).tobytes(),
    )


def _nan(payload):
    """A float64 NaN with an exact, distinguishable payload."""

    return struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000 | payload))[0]


_NAN_ONE = _nan(0x1)
_NAN_TWO = _nan(0x2)
_SUBNORMAL = 5e-324


class _Money(float):
    """A float subclass: an object leaf must still encode by value."""


def _object_series(values):
    array = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        array[position] = value
    return pd.Series(array)


def _string_series(values):
    return pd.Series(
        pd.array(list(values), dtype=pd.StringDtype(storage="python", na_value=pd.NA))
    )


_CASES = {
    "empty-float64": lambda: pd.Series(np.array([], dtype=np.float64)),
    "empty-object": lambda: _object_series(()),
    "empty-string": lambda: _string_series(()),
    "single-float64": lambda: pd.Series(np.array([-0.0], dtype=np.float64)),
    "float64-specials": lambda: pd.Series(
        np.array(
            [
                -0.0,
                0.0,
                float("inf"),
                float("-inf"),
                _SUBNORMAL,
                -_SUBNORMAL,
                _NAN_ONE,
                _NAN_TWO,
            ],
            dtype=np.float64,
        )
    ),
    "float32": lambda: pd.Series(np.array([-0.0, 1.5, np.inf], dtype=np.float32)),
    "int64": lambda: pd.Series(np.array([-1, 0, 2**40], dtype=np.int64)),
    "int32": lambda: pd.Series(np.array([-1, 0, 7], dtype=np.int32)),
    "bool": lambda: pd.Series(np.array([True, False, True], dtype=np.bool_)),
    "datetime64": lambda: pd.Series(
        pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    ),
    "Int64-masked": lambda: pd.Series(
        pd.arrays.IntegerArray(
            np.array([1, 999, 3], dtype=np.int64),
            np.array([False, True, False], dtype=np.bool_),
        )
    ),
    "boolean-masked": lambda: pd.Series(
        pd.arrays.BooleanArray(
            np.array([True, True, False], dtype=np.bool_),
            np.array([False, True, False], dtype=np.bool_),
        )
    ),
    "Float64-masked": lambda: pd.Series(
        pd.arrays.FloatingArray(
            np.array([1.0, _NAN_ONE, -0.0], dtype=np.float64),
            np.array([False, True, False], dtype=np.bool_),
        )
    ),
    "string-python": lambda: _string_series(("a", None, "ccc")),
    "string-unicode": lambda: _string_series(("ä", "🙂", None)),
    "default-string": lambda: pd.Series(["a", None, "ccc"]),
    "object-mixed": lambda: _object_series(
        ("a", 5, 2.5, None, pd.NA, True, b"raw", "")
    ),
    "object-float-signs": lambda: _object_series((-0.0, 0.0, _NAN_ONE, _NAN_TWO)),
    "object-big-int": lambda: _object_series((2**70, -(2**70))),
    "object-numpy-leaves": lambda: _object_series(
        (
            np.float64(-0.0),
            np.float32(1.5),
            np.int64(7),
            np.bool_(True),
            np.bytes_(b"x"),
        )
    ),
    "object-float-subclass": lambda: _object_series((_Money(1.5), _Money(-0.0))),
    "strided": lambda: pd.Series(np.arange(12, dtype=np.int64))[::3],
    "reversed": lambda: pd.Series(np.arange(6, dtype=np.float64))[::-1],
    "frame-column-slice": lambda: pd.DataFrame(
        {
            "a": np.arange(6, dtype=np.float64),
            "b": np.arange(6, dtype=np.int64),
        }
    )["a"].iloc[1:5],
}


def _selectors(length):
    yield "all-true", np.ones(length, dtype=np.bool_)
    yield "all-false", np.zeros(length, dtype=np.bool_)
    if length:
        alternating = np.zeros(length, dtype=np.bool_)
        alternating[::2] = True
        yield "alternating", alternating
        last = np.zeros(length, dtype=np.bool_)
        last[-1] = True
        yield "last-only", last


@pytest.mark.parametrize("name", sorted(_CASES))
def test_whole_series_slice_matches_the_all_true_reference(name):
    series = _CASES[name]()
    expected = _reference_storage_parts(series, np.ones(len(series), dtype=np.bool_))
    assert _storage_parts(series, slice(None)) == expected


@pytest.mark.parametrize("name", sorted(_CASES))
def test_whole_series_slice_matches_the_live_all_true_mask(name):
    series = _CASES[name]()
    mask = np.ones(len(series), dtype=np.bool_)
    assert _storage_parts(series, slice(None)) == _storage_parts(series, mask)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_subset_selection_is_unchanged(name):
    series = _CASES[name]()
    for label, selector in _selectors(len(series)):
        assert _storage_parts(series, selector) == _reference_storage_parts(
            series, selector
        ), label


def test_non_native_endian_bytes_are_preserved_by_both_selectors():
    try:
        integers = pd.Series(np.array([1, 2, 3], dtype=">i8"))
        floats = pd.Series(np.array([-0.0, 1.5], dtype=">f8"))
    except (TypeError, ValueError) as error:  # pragma: no cover - pandas policy
        pytest.skip(f"pandas refuses a non-native byte order here: {error}")
    for series in (integers, floats):
        mask = np.ones(len(series), dtype=np.bool_)
        assert _storage_parts(series, slice(None)) == _reference_storage_parts(
            series, mask
        )
        assert _storage_parts(series, slice(None)) == _storage_parts(series, mask)


def _mutation_pairs():
    yield (
        "hidden-masked-data",
        pd.Series(
            pd.arrays.IntegerArray(
                np.array([1, 999, 3], dtype=np.int64),
                np.array([False, True, False], dtype=np.bool_),
            )
        ),
        pd.Series(
            pd.arrays.IntegerArray(
                np.array([1, 1000, 3], dtype=np.int64),
                np.array([False, True, False], dtype=np.bool_),
            )
        ),
    )
    yield (
        "null-mask",
        pd.Series(
            pd.arrays.IntegerArray(
                np.array([1, 2, 3], dtype=np.int64),
                np.array([False, True, False], dtype=np.bool_),
            )
        ),
        pd.Series(
            pd.arrays.IntegerArray(
                np.array([1, 2, 3], dtype=np.int64),
                np.array([False, False, False], dtype=np.bool_),
            )
        ),
    )
    yield (
        "hidden-masked-float",
        pd.Series(
            pd.arrays.FloatingArray(
                np.array([1.0, _NAN_ONE], dtype=np.float64),
                np.array([False, True], dtype=np.bool_),
            )
        ),
        pd.Series(
            pd.arrays.FloatingArray(
                np.array([1.0, _NAN_TWO], dtype=np.float64),
                np.array([False, True], dtype=np.bool_),
            )
        ),
    )
    yield (
        "signed-zero",
        pd.Series(np.array([-0.0, 1.0], dtype=np.float64)),
        pd.Series(np.array([0.0, 1.0], dtype=np.float64)),
    )
    yield (
        "nan-payload",
        pd.Series(np.array([_NAN_ONE], dtype=np.float64)),
        pd.Series(np.array([_NAN_TWO], dtype=np.float64)),
    )
    yield (
        "infinity-sign",
        pd.Series(np.array([float("inf")], dtype=np.float64)),
        pd.Series(np.array([float("-inf")], dtype=np.float64)),
    )
    yield (
        "subnormal",
        pd.Series(np.array([_SUBNORMAL], dtype=np.float64)),
        pd.Series(np.array([2 * _SUBNORMAL], dtype=np.float64)),
    )
    yield (
        "string-content",
        _string_series(("a", "b")),
        _string_series(("a", "c")),
    )
    yield (
        "string-null",
        _string_series(("a", None)),
        _string_series(("a", "")),
    )
    yield (
        "object-leaf-type",
        _object_series((5,)),
        _object_series((5.0,)),
    )
    yield (
        "object-signed-zero",
        _object_series((-0.0,)),
        _object_series((0.0,)),
    )
    yield (
        "object-null-sentinel",
        _object_series((None,)),
        _object_series((pd.NA,)),
    )


@pytest.mark.parametrize("label,left,right", list(_mutation_pairs()), ids=lambda v: v)
def test_every_mutation_moves_both_selector_forms(label, left, right):
    mask = np.ones(len(left), dtype=np.bool_)
    assert _storage_parts(left, slice(None)) != _storage_parts(right, slice(None))
    assert _storage_parts(left, mask) != _storage_parts(right, mask)
    assert _reference_storage_parts(left, mask) != _reference_storage_parts(right, mask)


def test_storage_equal_passes_a_whole_series_slice(monkeypatch):
    """The optimization is at the call site, not only in the callee."""

    captured = []
    original = population_ops._storage_parts

    def recording(series, selected):
        captured.append(selected)
        return original(series, selected)

    monkeypatch.setattr(population_ops, "_storage_parts", recording)
    left = pd.Series(np.array([1.0, 2.0], dtype=np.float64))
    assert storage_equal(left, left.copy())
    monkeypatch.undo()

    assert captured == [slice(None), slice(None)]


def test_storage_equal_still_separates_visible_and_hidden_differences():
    left = pd.Series(
        pd.arrays.IntegerArray(
            np.array([1, 999, 3], dtype=np.int64),
            np.array([False, True, False], dtype=np.bool_),
        )
    )
    right = pd.Series(
        pd.arrays.IntegerArray(
            np.array([1, 1000, 3], dtype=np.int64),
            np.array([False, True, False], dtype=np.bool_),
        )
    )
    assert storage_equal(left, left.copy())
    assert not storage_equal(left, right)
    # A subset that excludes the masked row compares only the visible bytes.
    visible = np.array([True, False, True], dtype=np.bool_)
    assert storage_equal(left, right, visible)


def test_storage_equal_keeps_its_dtype_length_and_position_contract():
    left = pd.Series(np.array([1.0, 2.0], dtype=np.float64))
    assert not storage_equal(left, pd.Series(np.array([1, 2], dtype=np.int64)))
    assert not storage_equal(left, pd.Series(np.array([1.0], dtype=np.float64)))
    with pytest.raises(ValueError, match="row-aligned bool mask"):
        storage_equal(left, left.copy(), np.array([0, 1]))
    with pytest.raises(ValueError, match="row-aligned bool mask"):
        storage_equal(left, left.copy(), np.array([True], dtype=np.bool_))


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(np.array([1.0, 2.0], dtype=np.float64)),
        pd.Series(
            pd.arrays.IntegerArray(
                np.array([1, 2], dtype=np.int64),
                np.array([False, True], dtype=np.bool_),
            )
        ),
        _string_series(("a", "b")),
        _object_series(("a", 2)),
    ],
)
def test_a_wrong_length_mask_still_fails_closed(series):
    with pytest.raises(IndexError):
        _storage_parts(series, np.ones(len(series) + 1, dtype=np.bool_))


def test_an_unencodable_object_leaf_still_fails_closed():
    series = _object_series((object(),))
    with pytest.raises(PopulationError, match="storage-object-leaf"):
        _storage_parts(series, slice(None))
    with pytest.raises(PopulationError, match="storage-object-leaf"):
        _storage_parts(series, np.ones(1, dtype=np.bool_))


def test_object_float_leaves_encode_by_value_across_float_types():
    """numpy floats and float subclasses widen exactly like a plain float."""

    # NaN is left out on purpose: whether numpy round-trips a payload is a
    # separate question from this helper, and is pinned in the executor test.
    for value in (1.5, -0.0, float("inf"), 5e-324):
        plain = _object_series((value,))
        for equivalent in (np.float64(value), _Money(value)):
            assert _storage_parts(plain, slice(None)) == _storage_parts(
                _object_series((equivalent,)), slice(None)
            )
    # np.float32 is a different value, so it must not collide with float64.
    assert _storage_parts(_object_series((np.float32(0.1),)), slice(None)) != (
        _storage_parts(_object_series((0.1,)), slice(None))
    )
