"""``copy_table`` and ``copy_series``: pandas' deep copy, with each column copied once.

The graph executor hands kernels tables that hold one pandas block per column,
and ``DataFrame.copy(deep=True)`` of such a table copies every column, stacks
the copies of each NumPy dtype into a new block and reorders it, so it briefly
holds about three copies. ``copy_table`` makes the same table, block for block,
from one copy of each column. These tests compare it with pandas' copy on
invented tables of every column type a population or a kernel's input can
hold, check that the copy shares no memory with its source, that a write into
the copy leaves the source's bytes unchanged, and bound the copy's peak memory.
"""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import table_copies
from microcosm.frame.table_copies import copy_series, copy_table
from test_support.microcosm_frame.table_layouts import (
    block_layout,
    column_arrays,
    memory_digest,
    shares_memory,
)

# Row 1 of every column differs from row 0, so a write of row 1 into row 0
# changes bytes wherever it lands.
COLUMN_VALUES: dict[str, Callable[[], object]] = {
    "float64": lambda: np.array([0.5, -1.25, 2.0, -0.0, 3.5, 7.0]),
    "int64": lambda: np.array([5, 1, 4, 2, 8, 6], dtype=np.int64),
    "float64_second": lambda: np.array([1.5, 2.25, -3.0, 0.0, 9.5, 4.0]),
    "float32": lambda: np.array([0.5, -1.25, 2.0, -0.0, 3.5, 7.0], dtype=np.float32),
    "int32": lambda: np.array([5, 1, 4, 2, 8, 6], dtype=np.int32),
    "bool": lambda: np.array([True, False, False, True, False, True]),
    "int64_second": lambda: np.array([7, 3, 9, 1, 0, 2], dtype=np.int64),
    "object": lambda: np.array(["a", 1, 2.5, "d", "e", "f"], dtype=object),
    "Int64": lambda: pd.array([5, 1, None, 2, 8, 6], dtype="Int64"),
    "Int64_no_missing": lambda: pd.array([5, 1, 4, 2, 8, 6], dtype="Int64"),
    "Float64": lambda: pd.array([0.5, -1.25, None, 4.0, 3.5, 7.0], dtype="Float64"),
    "boolean": lambda: pd.array(
        [True, False, None, True, False, True], dtype="boolean"
    ),
    "boolean_no_missing": lambda: pd.array(
        [True, False, False, True, False, True], dtype="boolean"
    ),
    "category": lambda: pd.Categorical(["x", "y", "x", "z", "y", "x"]),
    "category_ordered": lambda: pd.Categorical(
        ["x", "y", "x", "z", "y", "x"], categories=["z", "y", "x"], ordered=True
    ),
    "category_object": lambda: pd.Categorical(
        ["x", "y", "x", "z", "y", "x"],
        categories=pd.Index(["x", "y", "z"], dtype=object),
    ),
    "string_python": lambda: pd.array(
        ["a", "b", None, "d", "e", "f"], dtype=pd.StringDtype("python")
    ),
    "string_pyarrow": lambda: pd.array(
        ["a", "b", None, "d", "e", "f"], dtype=pd.StringDtype("pyarrow")
    ),
    "str": lambda: pd.array(["a", "b", "c", "d", "e", "f"], dtype="str"),
}
COLUMNS = tuple(COLUMN_VALUES)
#: Columns pandas stores in two-dimensional blocks built from extension arrays;
#: a table holding one is copied by pandas itself.
PANDAS_COPIED_VALUES: dict[str, Callable[[], object]] = {
    "datetime": lambda: pd.date_range("2020-01-01", periods=6, freq="7D").array,
    "timedelta": lambda: pd.timedelta_range("1D", periods=6).array,
    "datetime_tz": lambda: (
        pd.date_range("2020-01-01", periods=6, freq="7D", tz="Europe/London").array
    ),
    "period": lambda: pd.period_range("2020-01", periods=6, freq="M").array,
}
INDEX_KINDS: dict[str, Callable[[], pd.Index | None]] = {
    "default range": lambda: None,
    "named range from 1": lambda: pd.RangeIndex(1, 7, name="row"),
    "int64": lambda: pd.Index(np.array([5, 3, 9, 1, 7, 2]), name="row"),
    "object": lambda: pd.Index(list("uvwxyz"), dtype=object, name="row"),
    "multi": lambda: pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 3), ("b", 4), ("c", 5), ("c", 6)],
        names=["part", "row"],
    ),
}


def _columns(names=COLUMNS, values=COLUMN_VALUES) -> dict[str, object]:
    return {name: values[name]() for name in names}


def _one_block_per_column(index=None, names=COLUMNS, values=COLUMN_VALUES):
    """A table holding each column in its own block, as kernel inputs do."""

    table = pd.DataFrame(_columns(names, values), index=index, copy=False)
    assert len(table._mgr.blocks) == len(names)
    return table


def _read_only(table: pd.DataFrame) -> pd.DataFrame:
    """Mark every array behind ``table`` read-only, as kernel inputs are."""

    for block in table._mgr.blocks:
        arrays = [block.values] if isinstance(block.values, np.ndarray) else []
        arrays.extend(
            getattr(block.values, name)
            for name in ("_ndarray", "_data", "_mask", "_codes")
            if isinstance(getattr(block.values, name, None), np.ndarray)
        )
        for array in arrays:
            array.setflags(write=False)
    return table


#: Source layouts. Each builds a new table.
SOURCES: dict[str, Callable[[], pd.DataFrame]] = {
    "one block per column": _one_block_per_column,
    "one read-only block per column": lambda: _read_only(_one_block_per_column()),
    "one block per dtype": lambda: pd.DataFrame(_columns()),
    "columns reordered by loc": lambda: _one_block_per_column().loc[
        :, list(reversed(COLUMNS))
    ],
    "rows selected by loc": lambda: _one_block_per_column().loc[
        np.array([True, False, True, True, False, True])
    ],
}


def _layout_set(table: pd.DataFrame) -> list[str]:
    return sorted(map(repr, block_layout(table)))


def _assert_same_as_pandas_copy(copied: pd.DataFrame, source: pd.DataFrame) -> None:
    expected = source.copy(deep=True)
    pd.testing.assert_frame_equal(
        copied,
        expected,
        check_exact=True,
        check_index_type=True,
        check_column_type=True,
        check_flags=True,
    )
    assert copied.attrs == expected.attrs
    assert _layout_set(copied) == _layout_set(expected)


@pytest.mark.parametrize("source_layout", SOURCES)
def test_the_copy_has_the_values_and_blocks_of_pandas_deep_copy(source_layout):
    source = SOURCES[source_layout]()
    source.attrs["receipt"] = {"stage": "invented"}

    copied = copy_table(source)

    _assert_same_as_pandas_copy(copied, source)
    assert type(copied) is pd.DataFrame


def test_a_table_of_one_block_per_column_keeps_the_order_of_pandas_blocks():
    source = _read_only(_one_block_per_column())

    assert block_layout(copy_table(source)) == block_layout(source.copy(deep=True))


@pytest.mark.parametrize("index_kind", INDEX_KINDS)
def test_every_index_kind_is_kept(index_kind):
    source = _one_block_per_column(index=INDEX_KINDS[index_kind]())

    _assert_same_as_pandas_copy(copy_table(source), source)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(lambda: _one_block_per_column().iloc[:0], id="no rows"),
        pytest.param(lambda: pd.DataFrame(index=pd.RangeIndex(3)), id="no columns"),
        pytest.param(
            lambda: pd.DataFrame(
                {"a": np.arange(3.0), "b": np.arange(3), "c": np.ones(3)}, copy=False
            ).set_axis(["a", "b", "a"], axis=1),
            id="repeated column label",
        ),
        pytest.param(
            lambda: pd.DataFrame(
                {("x", "a"): np.arange(3.0), ("y", "b"): np.arange(3.0)}, copy=False
            ),
            id="two-level column labels",
        ),
        pytest.param(
            lambda: pd.DataFrame(
                {0: np.arange(3.0), 1: pd.array([1, None, 3], dtype="Int64")},
                copy=False,
            ),
            id="integer column labels",
        ),
    ],
)
def test_unusual_shapes_and_labels_are_kept(source):
    table = source()

    _assert_same_as_pandas_copy(copy_table(table), table)


def test_flags_are_kept():
    source = _one_block_per_column().set_flags(allows_duplicate_labels=False)

    copied = copy_table(source)

    _assert_same_as_pandas_copy(copied, source)
    assert copied.flags.allows_duplicate_labels is False


@pytest.mark.parametrize("source_layout", SOURCES)
def test_the_copy_shares_no_memory_with_its_source(source_layout):
    source = SOURCES[source_layout]()

    copied = copy_table(source)

    assert shares_memory(copied, source) == []
    for name in ("category", "category_ordered", "category_object"):
        assert copied[name].cat.categories.array is not (
            source[name].cat.categories.array
        )


def _write_with_loc(table, name):
    table.loc[table.index[0], name] = table[name].iloc[1]


def _write_with_iat(table, name):
    table.iat[0, table.columns.get_loc(name)] = table[name].iloc[1]


def _write_into_array(table, name):
    values = table[name].array
    values[0] = values[1]


def _reverse_array_in_place(table, name):
    values = table[name].array
    values[:] = values[::-1]


def _write_into_values(table, name):
    values = table[name].values
    values[0] = values[1]


def _write_into_to_numpy(table, name):
    values = table[name].to_numpy()
    values[0] = values[1]


def _add_at_into_to_numpy(table, name):
    np.add.at(table[name].to_numpy(), [0], 1)


def _write_into_categories(table, name):
    categories = table[name].cat.categories.values
    categories[0] = categories[1] + "_written"


#: Writes a kernel might make into its own copy. Some raise for some column
#: types (a read-only array, a ufunc a type lacks). ``to_numpy()`` returns a
#: separate converted array for some extension types, so a write into it may
#: land nowhere; every other write that does not raise lands in the copy.
WRITES: dict[str, Callable[[pd.DataFrame, str], None]] = {
    "loc": _write_with_loc,
    "iat": _write_with_iat,
    "array item": _write_into_array,
    "array reversed in place": _reverse_array_in_place,
    "values item": _write_into_values,
    "to_numpy item": _write_into_to_numpy,
    "np.add.at into to_numpy": _add_at_into_to_numpy,
    "categories item": _write_into_categories,
}


def _applies(write: str, name: str) -> bool:
    if write == "categories item":
        return name.startswith("category")
    if write == "np.add.at into to_numpy":
        return name in {"float64", "int64", "float32", "int32"}
    return True


@pytest.mark.parametrize(
    ("source_layout", "write", "name"),
    [
        (source_layout, write, name)
        for source_layout in ("one block per column", "one read-only block per column")
        for write in WRITES
        for name in COLUMNS
        if _applies(write, name)
    ],
)
def test_a_write_into_the_copy_leaves_the_source_unchanged(source_layout, write, name):
    source = SOURCES[source_layout]()
    before = memory_digest(source)
    copied = copy_table(source)
    copy_before = memory_digest(copied)

    try:
        WRITES[write](copied, name)
    except (TypeError, ValueError):
        pass
    else:
        if write != "to_numpy item":
            assert memory_digest(copied) != copy_before, "the write did not land"

    assert memory_digest(source) == before


def test_a_pandas_write_into_the_copy_is_made_in_place():
    # No other object refers to the copy's blocks, so pandas writes into them
    # rather than copying a block first, as into pandas' own deep copy.
    copied = copy_table(_read_only(_one_block_per_column()))
    block_memory = [
        array.__array_interface__["data"][0]
        for _, column in copied.items()
        for array in column_arrays(column)
    ]

    for name in ("float64", "int64", "bool", "Int64", "boolean", "string_python"):
        _write_with_loc(copied, name)

    assert block_memory == [
        array.__array_interface__["data"][0]
        for _, column in copied.items()
        for array in column_arrays(column)
    ]


def test_a_range_index_is_rebuilt_and_other_indexes_are_viewed():
    # A view of a RangeIndex shares the cache holding its values once
    # materialized, and np.asarray returns that cached array writable.
    ranged = _one_block_per_column(index=pd.RangeIndex(1, 7, name="row"))
    copied = copy_table(ranged)
    np.asarray(copied.index)[0] = 99
    assert list(np.asarray(ranged.index)) == [1, 2, 3, 4, 5, 6]

    indexed = _one_block_per_column(index=INDEX_KINDS["int64"]())
    assert np.shares_memory(
        np.asarray(copy_table(indexed).index), np.asarray(indexed.index)
    )


@pytest.mark.parametrize("column", PANDAS_COPIED_VALUES)
def test_two_dimensional_extension_columns_are_copied_by_pandas(column):
    values = {**COLUMN_VALUES, **PANDAS_COPIED_VALUES}
    source = _one_block_per_column(
        names=("float64", column, "float64_second", "Int64", "category"),
        values=values,
    )

    copied = copy_table(source)

    _assert_same_as_pandas_copy(copied, source)
    assert block_layout(copied) == block_layout(source.copy(deep=True))
    _assert_categories_and_range_index_are_new(copied, source)


def _assert_categories_and_range_index_are_new(copied, source):
    assert shares_memory(copied, source) == []
    assert copied["category"].cat.categories.array is not (
        source["category"].cat.categories.array
    )
    np.asarray(copied.index)[0] = 99
    assert np.asarray(source.index)[0] != 99


def test_numpy_dtypes_sharing_a_name_are_copied_by_pandas():
    source = pd.DataFrame(
        {
            "native": np.arange(3, dtype="<f8"),
            "swapped": np.arange(3, dtype=">f8"),
        },
        copy=False,
    )

    copied = copy_table(source)

    _assert_same_as_pandas_copy(copied, source)
    assert block_layout(copied) == block_layout(source.copy(deep=True))


def test_a_pandas_subclass_is_copied_by_pandas():
    class Subclassed(pd.DataFrame):
        _metadata = ["note"]

        @property
        def _constructor(self):
            return Subclassed

    source = Subclassed(
        _columns(("float64", "float64_second", "Int64", "category")), copy=False
    )
    source.note = "kept"

    copied = copy_table(source)

    assert type(copied) is Subclassed
    assert copied.note == "kept"
    pd.testing.assert_frame_equal(copied, source.copy(deep=True), check_exact=True)
    _assert_categories_and_range_index_are_new(copied, source)


def test_without_the_pandas_block_constructor_tables_are_copied_by_pandas(
    monkeypatch,
):
    monkeypatch.setattr(table_copies, "create_dataframe_from_blocks", None)
    source = _one_block_per_column()

    copied = copy_table(source)

    _assert_same_as_pandas_copy(copied, source)
    assert block_layout(copied) == block_layout(source.copy(deep=True))
    _assert_categories_and_range_index_are_new(copied, source)


def test_copying_a_table_of_one_block_per_column_holds_one_copy_at_its_peak():
    rows = 200_000
    generator = np.random.default_rng(7)
    columns: dict[str, object] = {
        f"float64_{number}": generator.normal(size=rows) for number in range(12)
    }
    columns.update(
        {f"int64_{number}": generator.integers(0, 9, rows) for number in range(3)}
    )
    columns["bool"] = generator.random(rows) < 0.5
    columns["Int64"] = pd.array(generator.integers(0, 9, rows), dtype="Int64")
    columns["boolean"] = pd.array(generator.random(rows) < 0.5, dtype="boolean")
    source = _read_only(pd.DataFrame(columns, copy=False))
    column_bytes = sum(
        array.nbytes for _, column in source.items() for array in column_arrays(column)
    )
    del columns

    gc.collect()
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        copied = copy_table(source)
        held, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert copied.shape == source.shape
    # pandas' own deep copy of this table peaks at about three copies.
    assert (held - baseline) / column_bytes < 1.02
    assert (peak - baseline) / column_bytes < 1.05


SERIES_VALUES: dict[str, Callable[[], object]] = {
    "object": lambda: np.array(["a", "a", "b", "b", "c", "c"], dtype=object),
    "str": lambda: pd.array(["a", "a", "b", "b", "c", "c"], dtype="str"),
    "category": lambda: pd.Categorical(["a", "a", "b", "b", "c", "c"]),
    "int64": lambda: np.array([1, 1, 2, 2, 3, 3], dtype=np.int64),
    "bool": lambda: np.array([True, True, False, False, True, True]),
    "Int64": lambda: pd.array([1, None, 2, 2, 3, 3], dtype="Int64"),
    "string_python": lambda: pd.array(
        ["a", None, "b", "b", "c", "c"], dtype=pd.StringDtype("python")
    ),
    **PANDAS_COPIED_VALUES,
}


@pytest.mark.parametrize("kind", SERIES_VALUES)
@pytest.mark.parametrize("index_kind", ["default range", "named range from 1", "int64"])
def test_a_series_copy_equals_pandas_deep_copy_and_shares_no_memory(kind, index_kind):
    source = pd.Series(
        SERIES_VALUES[kind](), index=INDEX_KINDS[index_kind](), name="stratum"
    )
    source.attrs["receipt"] = "invented"

    copied = copy_series(source)

    pd.testing.assert_series_equal(
        copied,
        source.copy(deep=True),
        check_exact=True,
        check_index_type=True,
        check_flags=True,
    )
    assert copied.attrs == {"receipt": "invented"}
    assert not any(
        np.shares_memory(array, other)
        for array in column_arrays(copied)
        for other in column_arrays(source)
    )
    if kind == "category":
        assert copied.cat.categories.array is not source.cat.categories.array
    if isinstance(source.index, pd.RangeIndex):
        np.asarray(copied.index)[0] = 99
        assert np.asarray(source.index)[0] != 99


def test_a_series_subclass_is_copied_by_pandas():
    class Subclassed(pd.Series):
        @property
        def _constructor(self):
            return Subclassed

    source = Subclassed(np.arange(3.0))

    copied = copy_series(source)

    assert type(copied) is Subclassed
    pd.testing.assert_series_equal(copied, source.copy(deep=True), check_exact=True)
    assert not np.shares_memory(copied.to_numpy(), source.to_numpy())
