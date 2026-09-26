"""Whole-series selection parity for the full-PUF placement's seal parts.

Invented series and tables only: no genuine microdata, no source admission, no
store, archive or native read, and no tax engine.  ``_series_parts`` now asks
the storage helper for the whole series with ``slice(None)`` instead of
building an all-True selector; the emitted parts, their order and their bytes
must be exactly what the pre-change all-True form emitted, which is what
``_reference_series_parts`` below preserves verbatim.
"""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_full_puf_enrichment as placement
from microcosm.graph import population as population_ops
from microcosm.graph import store as store_ops
from microcosm.graph.canonical import canonical_json


def _reference_series_parts(series):
    """The exact pre-change body, with its all-True whole-series selector."""

    yield repr(series.dtype).encode()
    if pd.api.types.is_object_dtype(series.dtype):
        # Portable typed scalar bytes, never Python object pointers.
        yield from (store_ops._encode_object_scalar(v) for v in series)
    else:
        yield from population_ops._storage_parts(
            series, np.ones(len(series), dtype=np.bool_)
        )


def _reference_axis_parts(axis):
    placement.require(not isinstance(axis, pd.MultiIndex), "FULL_PUF_AXIS")
    yield type(axis).__name__.encode()
    yield canonical_json(store_ops._axis_name_payload(axis.name))
    yield from _reference_series_parts(pd.Series(axis.array, copy=False))


def _reference_table_stamp(table):
    parts = [
        *_reference_axis_parts(table.index),
        *_reference_axis_parts(table.columns),
        str(table.flags.allows_duplicate_labels).encode(),
        *(part for column in table for part in _reference_series_parts(table[column])),
    ]
    return placement._digest(parts)


def _nan(payload):
    return struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000 | payload))[0]


_NAN_ONE = _nan(0x1)
_NAN_TWO = _nan(0x2)


def _object_series(values):
    array = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        array[position] = value
    return pd.Series(array)


def _string_series(values):
    return pd.Series(
        pd.array(list(values), dtype=pd.StringDtype(storage="python", na_value=pd.NA))
    )


_SERIES = {
    "empty-float64": lambda: pd.Series(np.array([], dtype=np.float64)),
    "empty-object": lambda: _object_series(()),
    "float64-specials": lambda: pd.Series(
        np.array(
            [-0.0, 0.0, float("inf"), float("-inf"), 5e-324, _NAN_ONE, _NAN_TWO],
            dtype=np.float64,
        )
    ),
    "int64": lambda: pd.Series(np.array([-1, 0, 2**40], dtype=np.int64)),
    "bool": lambda: pd.Series(np.array([True, False], dtype=np.bool_)),
    "Int64-masked": lambda: pd.Series(
        pd.arrays.IntegerArray(
            np.array([1, 999, 3], dtype=np.int64),
            np.array([False, True, False], dtype=np.bool_),
        )
    ),
    "Float64-masked": lambda: pd.Series(
        pd.arrays.FloatingArray(
            np.array([1.0, _NAN_ONE], dtype=np.float64),
            np.array([False, True], dtype=np.bool_),
        )
    ),
    "string-python": lambda: _string_series(("a", None, "ccc")),
    "default-string": lambda: pd.Series(["a", None, "ccc"]),
    "object-mixed": lambda: _object_series(("a", 5, 2.5, None, pd.NA, True, b"raw")),
    "object-float-signs": lambda: _object_series((-0.0, 0.0, _NAN_ONE)),
    "strided": lambda: pd.Series(np.arange(12, dtype=np.int64))[::3],
    "reversed": lambda: pd.Series(np.arange(6, dtype=np.float64))[::-1],
}


@pytest.mark.parametrize("name", sorted(_SERIES))
def test_series_parts_match_the_all_true_reference(name):
    series = _SERIES[name]()
    assert tuple(placement._series_parts(series)) == tuple(
        _reference_series_parts(series)
    )


def test_series_parts_ask_for_the_whole_series_by_slice(monkeypatch):
    captured = []
    original = population_ops._storage_parts

    def recording(series, selected):
        captured.append(selected)
        return original(series, selected)

    monkeypatch.setattr(population_ops, "_storage_parts", recording)
    dense = tuple(placement._series_parts(_SERIES["int64"]()))
    objects = tuple(placement._series_parts(_SERIES["object-mixed"]()))
    monkeypatch.undo()

    assert captured == [slice(None)], "object columns never reach the helper"
    assert dense and objects


def _tables():
    yield (
        "mixed",
        pd.DataFrame(
            {
                "amount": np.array([-0.0, 1.5, float("inf")], dtype=np.float64),
                "count": np.array([1, 2, 3], dtype=np.int64),
                "nullable": pd.arrays.IntegerArray(
                    np.array([1, 999, 3], dtype=np.int64),
                    np.array([False, True, False], dtype=np.bool_),
                ),
                "text": pd.array(
                    ["a", None, "c"],
                    dtype=pd.StringDtype(storage="python", na_value=pd.NA),
                ),
            }
        ),
    )
    yield (
        "empty",
        pd.DataFrame(
            {
                "amount": np.array([], dtype=np.float64),
                "count": np.array([], dtype=np.int64),
            }
        ),
    )
    yield (
        "labelled-axes",
        pd.DataFrame(
            {"a": np.array([1.0, 2.0], dtype=np.float64)},
            index=pd.Index([7, 8], name="row"),
        ),
    )


@pytest.mark.parametrize("label,table", list(_tables()), ids=lambda value: value)
def test_table_stamp_matches_the_all_true_reference(label, table):
    assert placement._table_stamp(table) == _reference_table_stamp(table)


def _mutated_tables(base):
    table = base.copy()
    table.loc[0, "amount"] = 0.0
    yield "signed-zero", table

    table = base.copy()
    table["nullable"] = pd.arrays.IntegerArray(
        np.array([1, 1000, 3], dtype=np.int64),
        np.array([False, True, False], dtype=np.bool_),
    )
    yield "hidden-masked-data", table

    table = base.copy()
    table["nullable"] = pd.arrays.IntegerArray(
        np.array([1, 999, 3], dtype=np.int64),
        np.array([False, False, False], dtype=np.bool_),
    )
    yield "null-mask", table

    table = base.copy()
    table["text"] = pd.array(
        ["a", None, "z"], dtype=pd.StringDtype(storage="python", na_value=pd.NA)
    )
    yield "string-value", table

    yield "column-order", base[["count", "amount", "nullable", "text"]]
    yield "row-order", base.iloc[::-1]
    yield "index-labels", base.set_axis(pd.Index([5, 6, 7], name="row"))


def test_every_table_mutation_moves_the_stamp():
    base = dict(_tables())["mixed"]
    baseline = placement._table_stamp(base)
    assert baseline == _reference_table_stamp(base)
    for label, mutated in _mutated_tables(base):
        moved = placement._table_stamp(mutated)
        assert moved == _reference_table_stamp(mutated), label
        assert moved != baseline, label


def test_axis_parts_match_the_reference_and_still_refuse_a_multiindex():
    for axis in (
        pd.RangeIndex(3),
        pd.Index([7, 8], name="row"),
        pd.Index(["a", "b"], name="label"),
        pd.Index([], dtype=np.int64),
    ):
        assert tuple(placement._axis_parts(axis)) == tuple(_reference_axis_parts(axis))
    multi = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
    with pytest.raises(ValueError, match="FULL_PUF_AXIS"):
        tuple(placement._axis_parts(multi))
