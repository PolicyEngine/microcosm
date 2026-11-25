"""Kernel contexts read the population's own column memory, read-only (#1679).

The executor hands each kernel new pandas objects whose buffers are read-only
views of the population's buffers, so a node that reads every row holds no
second copy of its input tables and a node with a row mask holds only the
selected rows. A write through pandas, an item write into an array a column
hands out or into its ``.base``, and ``setflags(write=True)`` raise or change
only the kernel's own objects; the population's bytes stay the same. pandas
copies a column before writing into it, so a pandas write that changes no
value still runs, as it did on the deep copy kernels got before.
``np.<ufunc>.at`` writes even into a read-only array: such a write reaches the
population's memory, and the executor's before/after comparison of the context
rejects the node before anything derived from it is applied or stored. Every
frame here is invented.
"""

from __future__ import annotations

import gc
import hashlib
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.graph.executor as graph_executor
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.decl import (
    ROWS_ALL,
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.executor import NodeRejected, run_graph
from microcosm.graph.kernel import (
    Capabilities,
    Determinism,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
)
from microcosm.graph.population import Population
from microcosm.graph.store import ContentStore

_NODE_KEY = hashlib.sha256(b"test_graph_executor_input_views").hexdigest()
_MASKED_ARRAY_TYPES = (
    pd.arrays.BooleanArray,
    pd.arrays.FloatingArray,
    pd.arrays.IntegerArray,
)

# Row 1 of every column holds a value that differs from row 0, and no column
# reads the same reversed, so a write that copies row 1 into row 0 or reverses
# a column changes bytes if it lands.
COLUMN_VALUES: dict[str, Callable[[], object]] = {
    "float64": lambda: np.array([0.5, -1.25, 2.0, -0.0, 3.5, 7.0]),
    "float32": lambda: np.array([0.5, -1.25, 2.0, -0.0, 3.5, 7.0], dtype=np.float32),
    "int64": lambda: np.array([5, 1, 4, 2, 8, 6], dtype=np.int64),
    "int32": lambda: np.array([5, 1, 4, 2, 8, 6], dtype=np.int32),
    "bool": lambda: np.array([True, False, False, True, False, True]),
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
    "string_python": lambda: pd.array(
        ["a", "b", None, "d", "e", "f"], dtype=pd.StringDtype("python")
    ),
    "string_pyarrow": lambda: pd.array(
        ["a", "b", None, "d", "e", "f"], dtype=pd.StringDtype("pyarrow")
    ),
    "str": lambda: pd.array(["a", "b", "c", "d", "e", "f"], dtype="str"),
    "object": lambda: np.array(["a", 1, 2.5, "d", "e", "f"], dtype=object),
}
#: Storage the executor has no read-only view for, so it copies and freezes it.
COPIED_COLUMN_VALUES: dict[str, Callable[[], object]] = {
    "datetime": lambda: pd.date_range("2020-01-01", periods=6, freq="7D").array,
    "datetime_tz": lambda: (
        pd.date_range("2020-01-01", periods=6, freq="7D", tz="Europe/London").array
    ),
}
SHARED_COLUMNS = tuple(COLUMN_VALUES)
COPIED_COLUMNS = tuple(COPIED_COLUMN_VALUES)
COLUMNS = SHARED_COLUMNS + COPIED_COLUMNS
STRATA_VALUES: dict[str, Callable[[], object]] = {
    "object": lambda: np.array(["a", "a", "b", "b", "c", "c"], dtype=object),
    "category": lambda: pd.Categorical(["a", "a", "b", "b", "c", "c"]),
    "int64": lambda: np.array([1, 1, 2, 2, 3, 3], dtype=np.int64),
    "bool": lambda: np.array([True, True, False, False, True, True]),
}


def _ufunc_at_ignores_read_only() -> bool:
    """Whether this NumPy's ``np.add.at`` writes into a read-only array."""

    values = np.zeros(1)
    values.setflags(write=False)
    try:
        np.add.at(values, [0], 1)
    except ValueError:
        return False
    return bool(values[0] == 1)


#: NumPy 2.4.6 ignores the flag. Should a release enforce it, a ufunc.at write
#: is refused like any item write, and the tests that expect it to land skip.
UFUNC_AT_IGNORES_READ_ONLY = _ufunc_at_ignores_read_only()
needs_ufunc_at_through_read_only = pytest.mark.skipif(
    not UFUNC_AT_IGNORES_READ_ONLY,
    reason="this NumPy refuses ufunc.at into a read-only array",
)
SELECTED = np.array([True, False, True, True, False, True])
#: Declared out of the table's own order, so a projection that followed the
#: table's layout instead of the declaration would be seen.
DECLARED = tuple(reversed(COLUMNS))
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
#: The index kinds the write tests run under: one with no buffer, one with.
WRITE_INDEX_KINDS = ("default range", "int64")


def _kinds_frame(
    *,
    strata_dtype: str = "object",
    index: str = "default range",
    object_columns: bool = False,
) -> Frame:
    row_index = INDEX_KINDS[index]()
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype=np.int64),
            "person_household_id": np.array([10, 10, 20, 20, 30, 30], dtype=np.int64),
            "selected": SELECTED.copy(),
            **{column: make() for column, make in COLUMN_VALUES.items()},
            **{column: make() for column, make in COPIED_COLUMN_VALUES.items()},
        },
        index=row_index,
    )
    if object_columns:
        person.columns = pd.Index(list(person.columns), dtype=object, name="variable")
    person.attrs["provenance"] = {"source": "invented"}
    household = pd.DataFrame({"household_id": np.array([10, 20, 30], dtype=np.int64)})
    # pandas would read an object array of strings as its ``str`` dtype.
    strata = pd.Series(
        STRATA_VALUES[strata_dtype](),
        index=person.index,
        name="stratum",
        dtype=object if strata_dtype == "object" else None,
    )
    strata.attrs["kind"] = {"source": "invented"}
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.DESIGN)},
        strata,
    )


def _kinds_population(**options: object) -> Population:
    return Population.from_frame(_kinds_frame(**options), "survey")


def _reader(columns: tuple[str, ...], rows: str = ROWS_ALL) -> Node:
    if rows != ROWS_ALL and rows not in columns:
        columns = (*columns, rows)
    return Node(
        "reader",
        "reader@1",
        inputs=(Slice("person", columns, rows=rows),),
        outputs=(Owned("person", "out", "float64"),),
        population="survey",
    )


def _project(
    population: Population, columns: tuple[str, ...] = COLUMNS, rows: str = ROWS_ALL
) -> KernelContext:
    return graph_executor._project_context(
        _reader(columns, rows),
        population,
        key=_NODE_KEY,
        sources={},
        tolerances={},
        numerics={},
    )


def _is_arrow_backed(values: object) -> bool:
    return hasattr(values, "_pa_array")


def _storage(values: pd.Series | pd.Index) -> tuple[object, ...]:
    """Every stored byte of a column or index, in a comparable form.

    Object arrays are compared by which object each position refers to.
    Positions a nullable column's mask hides are included: a byte-identical
    population keeps them too.
    """

    array = values.array
    parts: list[object] = [str(values.dtype)]
    if isinstance(values, pd.MultiIndex):
        parts.extend(_storage(level) for level in values.levels)
        parts.extend(codes.tobytes() for codes in values.codes)
    elif isinstance(array, _MASKED_ARRAY_TYPES):
        parts.extend((array._data.tobytes(), array._mask.tobytes()))
    elif isinstance(array, pd.Categorical):
        parts.extend((array.codes.tobytes(), _storage(array.categories)))
    elif isinstance(array, pd.arrays.DatetimeArray):
        parts.append(array.asi8.tobytes())
    elif _is_arrow_backed(array):
        parts.extend(
            None if buffer is None else buffer.to_pybytes()
            for chunk in array._pa_array.chunks
            for buffer in chunk.buffers()
        )
    else:
        stored = np.asarray(array)
        if stored.dtype.hasobject:
            parts.append(tuple((id(item), repr(item)) for item in stored))
        else:
            parts.append(stored.tobytes())
    return tuple(parts)


def _population_storage(population: Population) -> tuple[object, ...]:
    frame = population.frame
    parts: list[object] = []
    for entity in frame.entities:
        table = frame.table(entity)
        parts.append((entity, tuple(table.columns), _storage(table.index)))
        parts.extend((column, _storage(table[column])) for column in table.columns)
    parts.append(_storage(frame.strata))
    for entity in frame.weighted_entities:
        parts.append((entity, frame.weights_for(entity).values.tobytes()))
    return tuple(parts)


def _memory_of(series: pd.Series) -> list[object]:
    """The NumPy arrays, or pyarrow buffer addresses, a column is stored in."""

    array = series.array
    if isinstance(array, _MASKED_ARRAY_TYPES):
        return [array._data, array._mask]
    if isinstance(array, pd.Categorical):
        return [array.codes]
    if isinstance(array, pd.arrays.DatetimeArray):
        return [array.asi8]
    if _is_arrow_backed(array):
        return [
            buffer.address
            for chunk in array._pa_array.chunks
            for buffer in chunk.buffers()
            if buffer is not None
        ]
    return [np.asarray(array)]


def _shares(context_part: object, population_part: object) -> bool:
    if isinstance(context_part, np.ndarray):
        return np.shares_memory(context_part, population_part)
    return context_part == population_part


def _writable_views(values: pd.Series | pd.Index) -> list[str]:
    """Arrays a column or index hands out that share its memory and can be written.

    Each array's chain of ``.base`` arrays is followed, and each is also asked
    to become writable with ``setflags(write=True)``.
    """

    if isinstance(values, (pd.RangeIndex, pd.MultiIndex)):
        # A RangeIndex stores no values: the array it materializes is its
        # own. A MultiIndex is copied for each context.
        return []
    array = values.array
    if isinstance(values, pd.Index):
        stored = [np.asarray(values)] if isinstance(values.dtype, np.dtype) else []
    else:
        stored = [part for part in _memory_of(values) if isinstance(part, np.ndarray)]
    candidates: dict[str, object] = {
        "to_numpy()": values.to_numpy(),
        "values": values.values,
        "np.asarray(array)": np.asarray(array),
    }
    if isinstance(array, _MASKED_ARRAY_TYPES):
        candidates.update({"_data": array._data, "_mask": array._mask})
    if isinstance(array, pd.Categorical):
        candidates.update({"_ndarray": array._ndarray, "codes": array.codes})
    found = []
    for name, candidate in candidates.items():
        depth = 0
        while isinstance(candidate, np.ndarray):
            label = name + ".base" * depth
            if any(np.shares_memory(candidate, part) for part in stored):
                if candidate.flags.writeable:
                    found.append(label)
                else:
                    try:
                        candidate.view().setflags(write=True)
                    except ValueError:
                        pass
                    else:
                        found.append(f"{label}.setflags(write=True)")
            candidate = candidate.base
            depth += 1
    return found


# ----------------------------------------------------------------------
# What a context holds
# ----------------------------------------------------------------------


@pytest.mark.parametrize("column", COPIED_COLUMNS)
def test_a_column_type_with_no_view_is_copied_and_read_only(column: str) -> None:
    population = _kinds_population()
    table = _project(population).tables["person"]
    (ours,) = _memory_of(table[column])
    (theirs,) = _memory_of(population.frame.table("person")[column])
    assert not np.shares_memory(ours, theirs)
    assert _writable_views(table[column]) == []
    pd.testing.assert_series_equal(
        table[column], population.frame.table("person")[column], check_exact=True
    )


@pytest.mark.parametrize("index", WRITE_INDEX_KINDS)
@pytest.mark.parametrize("column", SHARED_COLUMNS)
def test_an_every_row_context_shares_the_population_column_memory(
    column: str, index: str
) -> None:
    population = _kinds_population(index=index)
    context = _project(population)
    shared = zip(
        _memory_of(context.tables["person"][column]),
        _memory_of(population.frame.table("person")[column]),
        strict=True,
    )
    assert all(_shares(ours, theirs) for ours, theirs in shared)


@pytest.mark.parametrize("index", ["int64", "object"])
def test_an_every_row_context_shares_the_population_index_memory(index: str) -> None:
    population = _kinds_population(index=index)
    context = _project(population)
    assert np.shares_memory(
        np.asarray(context.tables["person"].index),
        np.asarray(population.frame.table("person").index),
    )
    assert np.shares_memory(
        np.asarray(context.strata.index), np.asarray(population.frame.strata.index)
    )


@pytest.mark.parametrize("strata_dtype", STRATA_VALUES)
def test_an_every_row_context_shares_the_population_strata(strata_dtype: str) -> None:
    population = _kinds_population(strata_dtype=strata_dtype)
    context = _project(population)
    shared = zip(
        _memory_of(context.strata), _memory_of(population.frame.strata), strict=True
    )
    assert all(_shares(ours, theirs) for ours, theirs in shared)


@pytest.mark.parametrize("index", INDEX_KINDS)
@pytest.mark.parametrize("rows", [ROWS_ALL, "selected"])
def test_no_context_buffer_can_be_written_or_made_writable(
    rows: str, index: str
) -> None:
    for strata_dtype in STRATA_VALUES:
        context = _project(
            _kinds_population(strata_dtype=strata_dtype, index=index), rows=rows
        )
        table = context.tables["person"]
        for column in table.columns:
            assert _writable_views(table[column]) == [], column
        assert _writable_views(context.strata) == []
        assert _writable_views(table.index) == []
        assert _writable_views(context.strata.index) == []


def test_a_masked_context_holds_only_its_own_copy_of_the_selected_rows() -> None:
    population = _kinds_population()
    context = _project(population, rows="selected")
    table = context.tables["person"]
    assert len(table) == int(SELECTED.sum())
    for column in COLUMNS:
        population_arrays = [
            part
            for part in _memory_of(population.frame.table("person")[column])
            if isinstance(part, np.ndarray)
        ]
        for part in _memory_of(table[column]):
            if isinstance(part, np.ndarray):
                assert not any(
                    np.shares_memory(part, theirs) for theirs in population_arrays
                ), column


@pytest.mark.parametrize("object_columns", [False, True])
@pytest.mark.parametrize("index", INDEX_KINDS)
@pytest.mark.parametrize("rows", [ROWS_ALL, "selected"])
@pytest.mark.parametrize("strata_dtype", STRATA_VALUES)
def test_a_context_holds_what_a_deep_copy_of_the_slice_holds(
    rows: str, strata_dtype: str, index: str, object_columns: bool
) -> None:
    """Kernels see the same columns, order, dtypes, index, values and attrs.

    The slice is what the executor handed out before #1679:
    ``table.loc[mask, columns].copy(deep=True)``, columns in declaration
    order after the structural ones.
    """

    population = _kinds_population(
        strata_dtype=strata_dtype, index=index, object_columns=object_columns
    )
    frame = population.frame
    context = _project(population, DECLARED, rows=rows)
    mask = (
        np.ones(frame.n("person"), dtype=bool)
        if rows == ROWS_ALL
        else frame.table("person")[rows].to_numpy()
    )
    columns = ["person_id", "person_household_id", *DECLARED]
    if rows != ROWS_ALL:
        columns.append(rows)
    expected = frame.table("person").loc[mask, columns].copy(deep=True)
    table = context.tables["person"]
    assert list(table.columns) == columns
    pd.testing.assert_frame_equal(
        table, expected, check_exact=True, check_index_type=True
    )
    assert type(table.index) is type(expected.index)
    assert table.index.names == expected.index.names
    assert table.columns.equals(expected.columns)
    assert table.columns.dtype == expected.columns.dtype
    assert table.columns.name == expected.columns.name
    assert table.attrs == expected.attrs == {"provenance": {"source": "invented"}}
    assert table.attrs is not frame.table("person").attrs
    for column in columns:
        ours = hashlib.sha256()
        theirs = hashlib.sha256()
        graph_executor._update_series(ours, table[column])
        graph_executor._update_series(theirs, expected[column])
        assert ours.digest() == theirs.digest(), column
    expected_strata = frame.strata.loc[mask].copy()
    pd.testing.assert_series_equal(
        context.strata, expected_strata, check_exact=True, check_index_type=True
    )
    assert type(context.strata.index) is type(expected_strata.index)
    assert (
        context.strata.attrs
        == expected_strata.attrs
        == {"kind": {"source": "invented"}}
    )
    assert context.strata.attrs["kind"] is not frame.strata.attrs["kind"]


def test_a_write_into_the_context_attrs_leaves_the_population_attrs() -> None:
    population = _kinds_population()
    context = _project(population)
    context.tables["person"].attrs["provenance"]["source"] = "changed"
    context.tables["person"].attrs["added"] = True
    assert population.frame.table("person").attrs == {
        "provenance": {"source": "invented"}
    }


# ----------------------------------------------------------------------
# Writes a kernel might make
# ----------------------------------------------------------------------


def _second_value(table: pd.DataFrame, column: str) -> object:
    return table[column].iloc[1]


def _write_loc(table: pd.DataFrame, column: str) -> None:
    table.loc[table.index[0], column] = _second_value(table, column)


def _write_iloc(table: pd.DataFrame, column: str) -> None:
    table.iloc[0, table.columns.get_loc(column)] = _second_value(table, column)


def _write_iat(table: pd.DataFrame, column: str) -> None:
    table.iat[0, table.columns.get_loc(column)] = _second_value(table, column)


def _write_at(table: pd.DataFrame, column: str) -> None:
    table.at[table.index[0], column] = _second_value(table, column)


def _assign_column(table: pd.DataFrame, column: str) -> None:
    table[column] = table[column].iloc[::-1].to_numpy()


def _add_column(table: pd.DataFrame, column: str) -> None:
    table[f"{column}_added"] = table[column]


def _write_to_numpy(table: pd.DataFrame, column: str) -> None:
    values = table[column].to_numpy()
    values[0] = values[1]


def _write_values(table: pd.DataFrame, column: str) -> None:
    values = table[column].values
    values[0] = values[1]


def _write_array(table: pd.DataFrame, column: str) -> None:
    values = table[column].array
    values[0] = values[1]


def _write_asarray_of_array(table: pd.DataFrame, column: str) -> None:
    values = np.asarray(table[column].array)
    values[0] = values[1]


def _add_in_place(table: pd.DataFrame, column: str) -> None:
    values = table[column].to_numpy()
    values += values


def _copyto(table: pd.DataFrame, column: str) -> None:
    values = table[column].to_numpy()
    np.copyto(values, values[::-1])


def _put(table: pd.DataFrame, column: str) -> None:
    values = table[column].to_numpy()
    np.put(values, 0, values[1])


def _fillna_in_place(table: pd.DataFrame, column: str) -> None:
    table.fillna({column: _second_value(table, column)}, inplace=True)


def _update(table: pd.DataFrame, column: str) -> None:
    table.update(pd.DataFrame({column: table[column].iloc[::-1].to_numpy()}))


def _sort_in_place(table: pd.DataFrame, column: str) -> None:
    table.sort_values(column, inplace=True, ascending=False)


def _write_index(table: pd.DataFrame, column: str) -> None:
    values = np.asarray(table.index)
    values[0] = values[1]


def _write_base_of_to_numpy(table: pd.DataFrame, column: str) -> None:
    owner = table[column].to_numpy().base
    owner.flat[0] = owner.flat[1]


def _write_asarray_of_base(table: pd.DataFrame, column: str) -> None:
    owner = np.asarray(table[column].to_numpy().base)
    owner.flat[0] = owner.flat[1]


def _write_after_setflags(table: pd.DataFrame, column: str) -> None:
    values = table[column].to_numpy()
    values.setflags(write=True)
    values[0] = values[1]


def _write_asarray_after_setflags(table: pd.DataFrame, column: str) -> None:
    values = np.asarray(table[column].array)
    values.setflags(write=True)
    values[0] = values[1]


def _write_base_of_codes(table: pd.DataFrame, column: str) -> None:
    owner = table[column].array.codes.base
    owner[0] = owner[1]


WRITES: dict[str, Callable[[pd.DataFrame, str], None]] = {
    "loc cell": _write_loc,
    "iloc cell": _write_iloc,
    "iat cell": _write_iat,
    "at cell": _write_at,
    "column assignment": _assign_column,
    "new column": _add_column,
    "item write through to_numpy()": _write_to_numpy,
    "item write through values": _write_values,
    "item write through array": _write_array,
    "item write through np.asarray(array)": _write_asarray_of_array,
    "+= on to_numpy()": _add_in_place,
    "np.copyto into to_numpy()": _copyto,
    "np.put into to_numpy()": _put,
    "fillna(inplace=True)": _fillna_in_place,
    "update": _update,
    "sort_values(inplace=True)": _sort_in_place,
    "item write into the index": _write_index,
    "item write through to_numpy().base": _write_base_of_to_numpy,
    "item write through np.asarray(to_numpy().base)": _write_asarray_of_base,
    "setflags(write=True) on to_numpy(), then an item write": _write_after_setflags,
    "setflags(write=True) on np.asarray(array), then an item write": (
        _write_asarray_after_setflags
    ),
    "item write through codes.base": _write_base_of_codes,
}


def _table_storage(table: pd.DataFrame) -> tuple[object, ...]:
    return (
        tuple(table.columns),
        _storage(table.index),
        tuple(_storage(table[column]) for column in table.columns),
    )


def _lands_on_a_writable_table(write: str, column: str) -> bool:
    """Whether ``write`` changes a table that shares no memory with anything."""

    table = _kinds_frame().table("person").copy(deep=True)
    before = _table_storage(table)
    try:
        WRITES[write](table, column)
    except Exception:  # noqa: BLE001 - pandas or NumPy refused it on its own
        return False
    return _table_storage(table) != before


#: The (write, column) pairs that change bytes when nothing is read-only. The
#: others pandas or NumPy refuses or redirects on any table, so they would pass
#: whatever the executor did.
LANDING_WRITES = tuple(
    (write, column)
    for write in WRITES
    for column in COLUMNS
    if _lands_on_a_writable_table(write, column)
)


def _refused(error: Exception) -> bool:
    """Whether ``error`` is NumPy or pandas refusing a write into read-only memory."""

    message = str(error)
    return isinstance(error, ValueError) and (
        "read-only" in message or "WRITEABLE" in message
    )


def test_every_write_kind_and_column_kind_has_a_case_that_lands() -> None:
    """The write tests are not vacuous for any kind of write or column."""

    writes = {write for write, _ in LANDING_WRITES}
    assert writes == set(WRITES)
    array_writes = {
        "item write through to_numpy()",
        "item write through values",
        "item write through array",
        "item write through np.asarray(array)",
        "item write through to_numpy().base",
        "setflags(write=True) on to_numpy(), then an item write",
        "item write through codes.base",
    }
    for column in COLUMNS:
        assert any((write, column) in LANDING_WRITES for write in array_writes), column


#: Writes run under a row mask and under a buffer-backed index too. Under a row
#: mask the context views ``.loc``'s own copy, and an int64 index matters only
#: to writes that reach arrays or the index, so the rest run once.
_ARRAY_AND_INDEX_WRITES = frozenset(
    {
        "item write through to_numpy()",
        "item write through values",
        "item write through array",
        "item write through np.asarray(array)",
        "item write through to_numpy().base",
        "setflags(write=True) on to_numpy(), then an item write",
        "item write through codes.base",
        "item write into the index",
    }
)
#: pandas writes land in the kernel's own table, so they must not raise, under
#: a row mask too, where the context views ``.loc``'s copy.
PANDAS_WRITES = frozenset(
    {
        "loc cell",
        "iloc cell",
        "iat cell",
        "at cell",
        "column assignment",
        "new column",
        "fillna(inplace=True)",
        "update",
        "sort_values(inplace=True)",
    }
)
_WRITE_CASES = tuple(
    (write, column, rows, index)
    for write, column in LANDING_WRITES
    # The index write does not depend on the column: once is enough.
    if write != "item write into the index" or column == "float64"
    for rows, index in (
        (ROWS_ALL, "default range"),
        *((("selected", "default range"),) if write in PANDAS_WRITES else ()),
        *(((ROWS_ALL, "int64"),) if write in _ARRAY_AND_INDEX_WRITES else ()),
    )
)


@pytest.mark.parametrize(("write", "column", "rows", "index"), _WRITE_CASES)
def test_a_write_into_a_context_leaves_the_population_byte_identical(
    write: str, column: str, rows: str, index: str
) -> None:
    """A pandas write runs and stays local; any other is refused as read-only."""

    population = _kinds_population(index=index)
    before = _population_storage(population)
    context = _project(population, rows=rows)
    if write in PANDAS_WRITES:
        WRITES[write](context.tables["person"], column)
    else:
        try:
            WRITES[write](context.tables["person"], column)
        except Exception as error:  # noqa: BLE001 - checked just below
            assert _refused(error), repr(error)
    assert _population_storage(population) == before


def _strata_other(strata: pd.Series) -> object:
    return strata.iloc[2]


def _flat_write(values: np.ndarray, value: object) -> None:
    values.flat[0] = value


STRATA_WRITES: dict[str, Callable[[pd.Series], None]] = {
    "iloc": lambda strata: strata.iloc.__setitem__(0, _strata_other(strata)),
    "item write through np.asarray(array)": lambda strata: np.asarray(
        strata.array
    ).__setitem__(0, _strata_other(strata)),
    "item write through values": lambda strata: strata.values.__setitem__(
        0, _strata_other(strata)
    ),
    "item write through to_numpy()": lambda strata: strata.to_numpy().__setitem__(
        0, _strata_other(strata)
    ),
    "item write through to_numpy().base": lambda strata: _flat_write(
        strata.to_numpy().base, _strata_other(strata)
    ),
    "setflags(write=True), then an item write": lambda strata: (
        np.asarray(strata.array).setflags(write=True),
        np.asarray(strata.array).__setitem__(0, _strata_other(strata)),
    ),
}


def _strata_write_lands(write: str, strata_dtype: str) -> bool:
    strata = _kinds_frame(strata_dtype=strata_dtype).strata.copy(deep=True)
    before = _storage(strata)
    try:
        STRATA_WRITES[write](strata)
    except Exception:  # noqa: BLE001 - pandas or NumPy refused it on its own
        return False
    return _storage(strata) != before


LANDING_STRATA_WRITES = tuple(
    (write, strata_dtype)
    for write in STRATA_WRITES
    for strata_dtype in STRATA_VALUES
    if _strata_write_lands(write, strata_dtype)
)


@pytest.mark.parametrize(("write", "strata_dtype"), LANDING_STRATA_WRITES)
def test_a_write_into_the_context_strata_stays_local_or_is_refused(
    write: str, strata_dtype: str
) -> None:
    """The rules for tables hold for the strata: pandas writes stay local.

    On the old frozen copy a pandas write into the strata raised; now it lands
    in the kernel's own strata, which the before/after comparison sees, as a
    write into a table does. Item writes into the arrays they hand out raise.
    """

    population = _kinds_population(strata_dtype=strata_dtype)
    before = _population_storage(population)
    context = _project(population)
    digest = graph_executor._context_digest(context)
    if write == "iloc":
        STRATA_WRITES[write](context.strata)
        assert graph_executor._context_digest(context) != digest
    else:
        with pytest.raises(ValueError) as refusal:
            STRATA_WRITES[write](context.strata)
        assert _refused(refusal.value), repr(refusal.value)
    assert _population_storage(population) == before


@pytest.mark.parametrize("rows", [ROWS_ALL, "selected"])
def test_categorical_strata_hand_out_their_own_categories(rows: str) -> None:
    population = _kinds_population(strata_dtype="category")
    before = _population_storage(population)
    context = _project(population, rows=rows)
    context.strata.cat.categories.name = "renamed"
    assert population.frame.strata.cat.categories.name is None
    assert list(population.frame.strata.cat.categories) == ["a", "b", "c"]
    assert _population_storage(population) == before


@pytest.mark.parametrize("column", ["category", "category_ordered"])
def test_a_categorical_column_hands_out_its_own_categories(column: str) -> None:
    population = _kinds_population()
    stored = population.frame.table("person")[column]
    table = _project(population, (column,)).tables["person"]
    assert table[column].dtype == stored.dtype
    assert table[column].cat.ordered is stored.cat.ordered
    table[column].cat.categories.name = "renamed"
    assert stored.cat.categories.name is None


def test_every_strata_kind_has_a_write_that_lands() -> None:
    for strata_dtype in STRATA_VALUES:
        assert any(kind == strata_dtype for _, kind in LANDING_STRATA_WRITES)
    assert {write for write, _ in LANDING_STRATA_WRITES} >= {
        "iloc",
        "item write through to_numpy().base",
    }


@needs_ufunc_at_through_read_only
@pytest.mark.parametrize(
    "column", ["float64", "float32", "int64", "int32", "Int64_no_missing"]
)
def test_a_ufunc_at_write_into_a_context_changes_the_context_digest(
    column: str,
) -> None:
    """``np.add.at`` writes into a read-only array; the digest comparison sees it."""

    context = _project(_kinds_population())
    before = graph_executor._context_digest(context)
    np.add.at(context.tables["person"][column].to_numpy(), [0], 1)
    assert graph_executor._context_digest(context) != before


_STRATA_UFUNC_WRITES: dict[str, Callable[[pd.Series], None]] = {
    "object": lambda strata: np.add.at(strata.to_numpy(), [0], "z"),
    "int64": lambda strata: np.add.at(strata.to_numpy(), [0], 1),
    "bool": lambda strata: np.add.at(strata.to_numpy().view(np.uint8), [0], 1),
}


@needs_ufunc_at_through_read_only
@pytest.mark.parametrize("strata_dtype", _STRATA_UFUNC_WRITES)
def test_a_ufunc_at_write_into_the_strata_changes_the_context_digest(
    strata_dtype: str,
) -> None:
    population = _kinds_population(strata_dtype=strata_dtype)
    before_bytes = _storage(population.frame.strata)
    context = _project(population)
    before = graph_executor._context_digest(context)
    _STRATA_UFUNC_WRITES[strata_dtype](context.strata)
    assert _storage(population.frame.strata) != before_bytes, "the write landed"
    assert graph_executor._context_digest(context) != before


@needs_ufunc_at_through_read_only
def test_a_ufunc_at_write_into_the_index_changes_the_context_digest() -> None:
    population = _kinds_population(index="int64")
    context = _project(population, ("float64",))
    before = graph_executor._context_digest(context)
    np.add.at(np.asarray(context.tables["person"].index), [0], 1)
    assert population.frame.table("person").index[0] == 6, "the write landed"
    assert graph_executor._context_digest(context) != before


# ----------------------------------------------------------------------
# Through run_graph
# ----------------------------------------------------------------------

SOURCE = SourceRef("survey", "csv-tables", description="invented bytes")
GRAPH_COLUMNS = {
    "float64": "float64",
    "float32": "float32",
    "int64": "int64",
    "int32": "int32",
    "bool": "bool",
    "Int64": "Int64",
    "Int64_no_missing": "Int64",
    "boolean": "boolean",
    "boolean_no_missing": "boolean",
    "string_python": "string",
    "string_pyarrow": "string",
}
#: int64 shares its pandas block with the id columns, so a write through
#: ``.base`` would reach columns the node did not declare.
GRAPH_COLUMNS_FOR_WRITES = frozenset(GRAPH_COLUMNS) - {"float32"}
CREATE = Node(
    "survey",
    "source@1",
    sources=("survey",),
    structural=StructuralDelta.CREATE,
    outputs=tuple(
        Owned("person", column, token) for column, token in GRAPH_COLUMNS.items()
    ),
)


def _graph_frame(_context: KernelContext) -> KernelResult:
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype=np.int64),
            "person_household_id": np.array([10, 10, 20, 20, 30, 30], dtype=np.int64),
            **{column: COLUMN_VALUES[column]() for column in GRAPH_COLUMNS},
        }
    )
    household = pd.DataFrame({"household_id": np.array([10, 20, 30], dtype=np.int64)})
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b", "b", "c", "c"], name="stratum", dtype=object),
    )
    return KernelResult(frame=frame)


class _Kernel:
    def __init__(
        self,
        ref: str,
        capabilities: Capabilities,
        compute: Callable[[KernelContext], KernelResult],
    ) -> None:
        self.ref = ref
        self.capabilities = capabilities
        self.compute = compute

    def implementation_hash(self) -> str:
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        return self.compute(context)


def _zeros(table: pd.DataFrame) -> KernelResult:
    ids = pd.Index(table["person_id"].to_numpy(copy=True), name="person_id")
    return KernelResult(
        columns={
            ("person", "out"): pd.Series(np.zeros(len(ids)), index=ids, dtype="float64")
        }
    )


def _writer_graph(column: str) -> Graph:
    writer = Node(
        "writer",
        "writer@1",
        inputs=(Slice("person", (column,)),),
        outputs=(Owned("person", "out", "float64"),),
        population="survey",
    )
    return Graph("toy", (SOURCE,), (CREATE, writer))


def _registry(
    write: Callable[[KernelContext], None],
    *,
    compute: Callable[[KernelContext], KernelResult] | None = None,
    capabilities: Capabilities | None = None,
) -> KernelRegistry:
    def written_then_zeros(context: KernelContext) -> KernelResult:
        write(context)
        return _zeros(context.tables["person"])

    registry = KernelRegistry()
    registry.register(
        _Kernel(
            "source@1",
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
            _graph_frame,
        )
    )
    registry.register(
        _Kernel(
            "writer@1",
            capabilities or Capabilities(Determinism.DETERMINISTIC),
            compute or written_then_zeros,
        )
    )
    return registry


def _source_directory(root: Path) -> Path:
    root.mkdir()
    (root / "value.txt").write_text("0", encoding="utf-8")
    return root


def _watch_create() -> tuple[
    Callable[[str, Population], None], list[Population], list[tuple[object, ...]]
]:
    """An observer keeping the live CREATE population and its bytes at admission."""

    held: list[Population] = []
    admitted: list[tuple[object, ...]] = []

    def observe(node_id: str, population: Population) -> None:
        if node_id == "survey":
            held.append(population)
            admitted.append(_population_storage(population))

    return observe, held, admitted


def _run(
    graph: Graph,
    registry: KernelRegistry,
    store: ContentStore,
    source: Path,
    observe: Callable[[str, Population], None] | None = None,
):
    return run_graph(
        compile_graph(graph),
        sources={"survey": source},
        store=store,
        kernels=registry,
        _population_observer=observe,
        _population_observer_detach=False,
    )


_RUN_GRAPH_WRITES = (
    "loc cell",
    "column assignment",
    "item write through to_numpy()",
    "item write through values",
    "item write through array",
    "item write through np.asarray(array)",
    "item write through to_numpy().base",
    "setflags(write=True) on to_numpy(), then an item write",
)


@pytest.mark.parametrize(
    ("write", "column"),
    [
        (write, column)
        for write, column in LANDING_WRITES
        if write in _RUN_GRAPH_WRITES and column in GRAPH_COLUMNS_FOR_WRITES
    ],
)
def test_a_kernel_that_writes_into_its_inputs_changes_nothing_it_reads(
    tmp_path: Path, write: str, column: str
) -> None:
    """The write raises, which fails the node, or the comparison rejects it."""

    def vandal(context: KernelContext) -> None:
        WRITES[write](context.tables["person"], column)

    observe, held, admitted = _watch_create()
    with pytest.raises(NodeRejected, match="'writer'"):
        _run(
            _writer_graph(column),
            _registry(vandal),
            ContentStore(tmp_path / "store"),
            _source_directory(tmp_path / "source"),
            observe,
        )
    (population,) = held
    assert _population_storage(population) == admitted[0]


@pytest.mark.parametrize(
    ("ufunc", "column"),
    [
        ("add", "float64"),
        ("add", "int64"),
        ("add", "int32"),
        ("add", "float32"),
        ("add", "Int64_no_missing"),
        ("maximum", "float64"),
        ("negative", "int64"),
        ("add", "strata"),
    ],
)
def test_a_ufunc_at_write_is_rejected_and_the_store_keeps_the_source_bytes(
    tmp_path: Path, ufunc: str, column: str
) -> None:
    """``ufunc.at`` writes into a read-only array: rejected, and nothing stored."""

    source = _source_directory(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    create_only = Graph("toy", (SOURCE,), (CREATE,))
    first = _run(create_only, _registry(lambda context: None), store, source)
    stored_before = _stored_objects(store)
    landed: list[bool] = []

    def write_at(context: KernelContext) -> None:
        if column == "strata":
            values = context.strata.to_numpy()
            before = values.copy()
            np.add.at(values, [0], "z")
        else:
            values = context.tables["person"][column].to_numpy()
            before = values.copy()
            if ufunc == "negative":
                np.negative.at(values, [0])
            else:
                getattr(np, ufunc).at(values, [0], 100)
        landed.append(not np.array_equal(values, before))

    read = "float64" if column == "strata" else column
    # A NumPy that enforced the read-only flag would refuse the write, which
    # fails the node instead.
    expected = "mutated its input context" if UFUNC_AT_IGNORES_READ_ONLY else "failed"
    with pytest.raises(NodeRejected, match=expected):
        _run(_writer_graph(read), _registry(write_at), store, source)
    assert landed == ([True] if UFUNC_AT_IGNORES_READ_ONLY else [])
    assert _stored_objects(store) == stored_before

    again = _run(create_only, _registry(lambda context: None), store, source)
    assert again.nodes["survey"].hit
    reloaded = again.population("survey").person
    original = first.population("survey").person
    pd.testing.assert_series_equal(
        again.population("survey").strata, first.population("survey").strata
    )
    for name in GRAPH_COLUMNS:
        # The store keeps values, not the bytes a nullable column's mask hides.
        pd.testing.assert_series_equal(
            reloaded[name], original[name], check_dtype=False, check_exact=True
        )
        if isinstance(original[name].dtype, np.dtype):
            assert _storage(reloaded[name]) == _storage(original[name]), name


def _stored_objects(store: ContentStore) -> dict[str, bytes]:
    return {
        path.relative_to(store.objects).as_posix(): path.read_bytes()
        for path in sorted(store.objects.rglob("*"))
        if path.is_file()
    }


def _write_a_copy(table: pd.DataFrame, column: str) -> pd.Series:
    work = table.copy()
    work.loc[work.index[0], column] = work[column].iloc[1]
    return work[column]


def _write_a_selection(table: pd.DataFrame, column: str) -> pd.Series:
    work = table[["person_id", column]]
    work.loc[work.index[0], column] = work[column].iloc[1]
    return work[column]


def _write_a_column_series(table: pd.DataFrame, column: str) -> pd.Series:
    work = table[column]
    work.iloc[0] = work.iloc[1]
    return work


def _assign_into_a_new_table(table: pd.DataFrame, column: str) -> pd.Series:
    return table.assign(**{column: table[column].iloc[::-1].to_numpy()})[column]


def _write_a_copied_array(table: pd.DataFrame, column: str) -> pd.Series:
    values = table[column].to_numpy(copy=True)
    values[0] = values[1]
    if values.dtype.kind in "fiu":
        np.add.at(values, [1], 0)
    return pd.Series(values, index=table.index)


OWN_COPY_WRITES: dict[str, Callable[[pd.DataFrame, str], pd.Series]] = {
    "loc into table.copy()": _write_a_copy,
    "loc into table[columns]": _write_a_selection,
    "iloc into table[column]": _write_a_column_series,
    "table.assign": _assign_into_a_new_table,
    "writes into to_numpy(copy=True)": _write_a_copied_array,
}


@pytest.mark.parametrize("column", ["float64", "Int64", "boolean", "string_python"])
@pytest.mark.parametrize("write", OWN_COPY_WRITES)
def test_a_kernel_that_writes_only_its_own_copy_runs_and_changes_nothing(
    tmp_path: Path, write: str, column: str
) -> None:
    changed: list[bool] = []

    def own_copy(context: KernelContext) -> None:
        table = context.tables["person"]
        written = OWN_COPY_WRITES[write](table, column)
        changed.append(
            not written.reset_index(drop=True).equals(
                table[column].reset_index(drop=True)
            )
        )

    observe, held, admitted = _watch_create()
    _run(
        _writer_graph(column),
        _registry(own_copy),
        ContentStore(tmp_path / "store"),
        _source_directory(tmp_path / "source"),
        observe,
    )
    assert changed == [True]
    assert _population_storage(held[0]) == admitted[0]


VALUE_PRESERVING_WRITES: dict[str, Callable[[pd.DataFrame, str], None]] = {
    "loc into an empty selection": lambda table, column: table.loc.__setitem__(
        (table[column] < -1e9, column), 0.0
    ),
    "iloc cell with its own value": lambda table, column: table.iloc.__setitem__(
        (0, table.columns.get_loc(column)), table[column].iloc[0]
    ),
    "update with the same values": lambda table, column: table.update(
        pd.DataFrame({column: table[column].to_numpy().copy()}, index=table.index)
    ),
    "at cell with its own value": lambda table, column: table.at.__setitem__(
        (table.index[0], column), table[column].iloc[0]
    ),
    "loc[index, column] = the column": lambda table, column: table.loc.__setitem__(
        (table.index, column), table[column].to_numpy().copy()
    ),
    "loc[:, column] *= 1": lambda table, column: table.loc.__setitem__(
        (slice(None), column), table.loc[:, column] * 1
    ),
}


@pytest.mark.parametrize("write", VALUE_PRESERVING_WRITES)
def test_a_pandas_write_that_changes_no_value_still_runs(
    tmp_path: Path, write: str
) -> None:
    """As on the deep copy: pandas copies the column first, and nothing changed."""

    def same_values(context: KernelContext) -> None:
        VALUE_PRESERVING_WRITES[write](context.tables["person"], "float64")

    observe, held, admitted = _watch_create()
    manifest = _run(
        _writer_graph("float64"),
        _registry(same_values),
        ContentStore(tmp_path / "store"),
        _source_directory(tmp_path / "source"),
        observe,
    )
    assert manifest.population("survey").person["out"].tolist() == [0.0] * 6
    assert _population_storage(held[0]) == admitted[0]


_VALIDATION_WRITES: dict[str, Callable[[pd.DataFrame], None]] = {
    "loc cell": lambda table: table.loc.__setitem__((table.index[0], "float64"), 99.0),
    "clip(inplace=True)": lambda table: table.clip(lower=2.0, inplace=True),
}


@pytest.mark.parametrize("write", _VALIDATION_WRITES)
def test_a_validation_kernel_that_writes_into_its_input_is_rejected(
    tmp_path: Path, write: str
) -> None:
    """Rejected as before, not stored as a failed verdict from a read-only error."""

    def check(context: KernelContext) -> KernelResult:
        table = context.tables["person"]
        _VALIDATION_WRITES[write](table)
        ids = pd.Index(table["person_id"].to_numpy(copy=True), name="person_id")
        return KernelResult(
            columns={("person", "out"): pd.Series(0.0, index=ids, dtype="float64")},
            receipt={"outcome": "pass", "evidence": {}},
        )

    observe, held, admitted = _watch_create()
    with pytest.raises(NodeRejected, match="mutated its input context"):
        _run(
            _writer_graph("float64"),
            _registry(
                lambda context: None,
                compute=check,
                capabilities=Capabilities(
                    Determinism.DETERMINISTIC, role=KernelRole.GATE
                ),
            ),
            ContentStore(tmp_path / "store"),
            _source_directory(tmp_path / "source"),
            observe,
        )
    assert _population_storage(held[0]) == admitted[0]


_STRIDED_ROWS = 1_003


def _strided_population(*, strided: bool = True) -> Population:
    """Float columns stored as strided views of one C-order matrix, or not."""

    matrix = np.random.default_rng(5).normal(size=(_STRIDED_ROWS, 3))
    # Row labels and strata from one C-order integer matrix are strided too.
    labels = np.arange(2 * _STRIDED_ROWS, dtype=np.int64).reshape(_STRIDED_ROWS, 2)
    labels[:, 1] %= 3
    if not strided:
        matrix = np.asfortranarray(matrix)
        labels = np.asfortranarray(labels)
    index = pd.Index(labels[:, 0], name="row", copy=False)
    person = pd.DataFrame(
        matrix, columns=["x0", "x1", "x2"], index=index, copy=False
    ).assign(
        person_id=np.arange(1, _STRIDED_ROWS + 1, dtype=np.int64),
        person_household_id=np.ones(_STRIDED_ROWS, dtype=np.int64),
        count=pd.array(np.arange(_STRIDED_ROWS), dtype="Int64"),
    )
    frame = Frame.adopt(
        {"person": person, "household": pd.DataFrame({"household_id": [1]})},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0]), WeightKind.DESIGN)},
        pd.Series(labels[:, 1], index=index, name="stratum", copy=False),
    )
    for stored in (
        frame.table("person")["x0"].to_numpy(),
        np.asarray(frame.table("person").index),
        frame.strata.to_numpy(),
    ):
        assert stored.flags.c_contiguous is not strided
    return Population.from_frame(frame, "survey")


def test_a_table_with_a_strided_column_is_viewed_through_the_old_deep_copy() -> None:
    """The columns sit side by side in one contiguous block, as in the old copy.

    ``np.dot`` can round differently when its operands' layout or alignment
    changes, so separate contiguous copies would not reproduce the old bytes.
    """

    population = _strided_population()
    context = _project(population, ("x0", "x1", "x2"))
    table = context.tables["person"]
    expected = (
        population.frame.table("person")
        .loc[:, ["person_id", "person_household_id", "x0", "x1", "x2"]]
        .copy(deep=True)
    )
    pd.testing.assert_frame_equal(table, expected, check_exact=True)
    addresses = [table[name].to_numpy().ctypes.data for name in ("x0", "x1", "x2")]
    row_bytes = _STRIDED_ROWS * 8
    assert addresses == [
        addresses[0],
        addresses[0] + row_bytes,
        addresses[0] + 2 * row_bytes,
    ]
    assert not np.shares_memory(
        table["x0"].to_numpy(), population.frame.table("person")["x0"].to_numpy()
    )
    assert _writable_views(table["x0"]) == []
    assert _writable_views(table.index) == []
    assert np.asarray(table.index).flags.c_contiguous


def test_strided_strata_follow_the_rules_for_tables() -> None:
    population = _strided_population()
    before = _population_storage(population)
    context = _project(population, ("x0",))
    pd.testing.assert_series_equal(
        context.strata, population.frame.strata.copy(deep=True), check_exact=True
    )
    assert context.strata.to_numpy().flags.c_contiguous
    assert _writable_views(context.strata) == []
    assert _writable_views(context.strata.index) == []
    with pytest.raises(ValueError, match="read-only"):
        context.strata.to_numpy()[0] = 2
    digest = graph_executor._context_digest(context)
    context.strata.iloc[0] = context.strata.iloc[0] + 7
    assert graph_executor._context_digest(context) != digest
    assert _population_storage(population) == before


def _set_own_value(table: pd.DataFrame, column: str, *, through_array: bool) -> None:
    if through_array:
        table[column].array[0] = table[column].iloc[0]
    else:
        table.loc[table.index[0], column] = table[column].iloc[0]


@pytest.mark.parametrize("through_array", [False, True])
@pytest.mark.parametrize("column", ["x0", "count"])
def test_a_write_has_the_same_outcome_whatever_the_population_layout(
    column: str, through_array: bool
) -> None:
    """A node key's verdict must not depend on how the population is stored."""

    outcomes = []
    for strided in (True, False):
        table = _project(_strided_population(strided=strided), ("x0", "count")).tables[
            "person"
        ]
        try:
            _set_own_value(table, column, through_array=through_array)
        except ValueError as error:
            assert _refused(error), repr(error)
            outcomes.append("refused")
        else:
            outcomes.append("ran")
    assert outcomes[0] == outcomes[1]
    assert outcomes[0] == ("refused" if through_array else "ran")


def test_a_table_kept_after_its_context_is_freed_stays_read_only() -> None:
    """Documented limit: a table sharing a context's memory must be copied to write.

    While the kernel runs, pandas copies before writing. Once the context is
    freed, a shallow table derived from it is the only owner of read-only
    memory, so an in-place pandas write into it raises.
    """

    population = _kinds_population()
    before = _population_storage(population)
    context = _project(population, ("float64",))
    derived = context.tables["person"].reset_index()
    del context
    gc.collect()
    with pytest.raises(ValueError, match="read-only"):
        derived.iloc[0, derived.columns.get_loc("float64")] = 99.0
    copied = derived.copy()
    copied.iloc[0, copied.columns.get_loc("float64")] = 99.0
    assert _population_storage(population) == before


@needs_ufunc_at_through_read_only
def test_a_bool_byte_written_through_a_uint8_view_changes_the_digest() -> None:
    """``np.add.at`` can store a 2 that still reads as True; the digest sees it."""

    population = _kinds_population()
    context = _project(population, ("bool",))
    before = graph_executor._context_digest(context)
    np.add.at(context.tables["person"]["bool"].to_numpy().view(np.uint8), [0], 1)
    assert population.frame.table("person")["bool"].to_numpy().view(np.uint8)[0] == 2
    assert graph_executor._context_digest(context) != before


class _TaggedFrame(pd.DataFrame):
    _metadata = ["tag"]

    @property
    def _constructor(self) -> type[_TaggedFrame]:
        return _TaggedFrame


def test_a_pandas_subclass_table_keeps_its_type_and_metadata() -> None:
    frame = _kinds_frame()
    person = _TaggedFrame(frame.table("person"))
    person.tag = "invented"
    tagged = Frame(
        {"person": person, "household": frame.table("household")},
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    for rows in (ROWS_ALL, "selected"):
        population = Population.from_frame(tagged, "survey")
        before = _population_storage(population)
        table = _project(population, rows=rows).tables["person"]
        assert type(table) is _TaggedFrame
        assert table.tag == "invented"
        for column in ("float64", "Int64_no_missing", "string_python"):
            assert not any(
                isinstance(ours, np.ndarray) and np.shares_memory(ours, theirs)
                for ours, theirs in zip(
                    _memory_of(table[column]),
                    _memory_of(population.frame.table("person")[column]),
                    strict=True,
                )
            ), column
            try:
                _write_array(table, column)
            except ValueError as error:
                assert _refused(error), repr(error)
        assert _population_storage(population) == before


class _TaggedSeries(pd.Series):
    _metadata = ["tag"]

    @property
    def _constructor(self) -> type[_TaggedSeries]:
        return _TaggedSeries


def test_pandas_subclass_strata_keep_their_type_and_metadata() -> None:
    frame = _kinds_frame()
    strata = _TaggedSeries(frame.strata)
    strata.tag = "invented"
    tagged = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {"household": frame.weights_for("household")},
        strata,
    )
    population = Population.from_frame(tagged, "survey")
    before = _population_storage(population)
    context = _project(population)
    assert type(context.strata) is _TaggedSeries
    assert context.strata.tag == "invented"
    assert not np.shares_memory(
        np.asarray(context.strata.array), np.asarray(population.frame.strata.array)
    )
    with pytest.raises(ValueError, match="read-only"):
        context.strata.iloc[0] = "c"
    assert _population_storage(population) == before


# ----------------------------------------------------------------------
# Peak memory
# ----------------------------------------------------------------------

_PEAK_ROWS = 200_000


def _wide_population() -> Population:
    generator = np.random.default_rng(1679)
    columns: dict[str, object] = {
        "person_id": np.arange(_PEAK_ROWS, dtype=np.int64),
        "person_household_id": np.arange(_PEAK_ROWS, dtype=np.int64) // 2,
        "half": np.arange(_PEAK_ROWS) % 2 == 0,
    }
    for number in range(24):
        columns[f"float64_{number}"] = generator.normal(size=_PEAK_ROWS)
    for number in range(2):
        columns[f"Int64_{number}"] = pd.array(
            generator.integers(0, 100, size=_PEAK_ROWS), dtype="Int64"
        )
        columns[f"boolean_{number}"] = pd.array(
            generator.random(_PEAK_ROWS) < 0.5, dtype="boolean"
        )
    frame = Frame(
        {
            "person": pd.DataFrame(columns),
            "household": pd.DataFrame(
                {"household_id": np.arange(_PEAK_ROWS // 2, dtype=np.int64)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(_PEAK_ROWS // 2), WeightKind.DESIGN)},
        pd.Series(np.zeros(_PEAK_ROWS, dtype=np.int64), name="stratum"),
    )
    return Population.from_frame(frame, "survey")


def _input_bytes(population: Population) -> int:
    table = population.frame.table("person")
    total = 0
    for column in table.columns:
        values = table[column].array
        if isinstance(values, _MASKED_ARRAY_TYPES):
            total += values._data.nbytes + values._mask.nbytes
        else:
            total += table[column].to_numpy().nbytes
    return total


def _traced_projection(
    population: Population, rows: str, columns: tuple[str, ...] | None = None
) -> tuple[int, int]:
    """Bytes the context holds after projection, and the peak during it."""

    if columns is None:
        columns = tuple(
            column
            for column in population.frame.table("person").columns
            if column not in {"person_id", "person_household_id"}
        )
    gc.collect()
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        context = _project(population, columns, rows)
        held, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(context.tables["person"]) == (
        _PEAK_ROWS if rows == ROWS_ALL else _PEAK_ROWS // 2
    )
    return held - baseline, peak - baseline


def test_an_every_row_context_allocates_no_copy_of_its_tables() -> None:
    """No copy of the tables: the context's own allocations stay small.

    With the deep copy (before #1679) the context held 1.11 times the bytes of
    the input columns and the projection peaked at 1.18 times; with views it
    holds 0.04 times and peaks at 0.11 times. What is left is the effective
    weights, which are still copied, and pandas' objects; the row mask and
    the weights' temporaries add to the peak.
    """

    population = _wide_population()
    held, peak = _traced_projection(population, ROWS_ALL)
    input_bytes = _input_bytes(population)
    assert held < 0.1 * input_bytes
    assert peak < 0.25 * input_bytes


def test_a_masked_context_allocates_one_copy_of_the_selected_rows() -> None:
    """Half the rows: ``.loc`` copies them once, and nothing copies them again.

    One copy of half the rows is 0.5 times the input columns. With the deep
    copy (before #1679) the projection peaked at 1.02 times; with views it
    peaks at 0.61 times.
    """

    population = _wide_population()
    held, peak = _traced_projection(population, "half")
    input_bytes = _input_bytes(population)
    assert held < 0.65 * input_bytes
    assert peak < 0.8 * input_bytes


def test_a_masked_context_copies_only_the_columns_it_reads() -> None:
    """Two of 31 columns under a row mask: no copy of the other columns' rows.

    Copying every column's selected rows first would cost 0.5 times the input
    columns on its own.
    """

    population = _wide_population()
    held, peak = _traced_projection(population, "half", ("float64_0", "half"))
    input_bytes = _input_bytes(population)
    assert held < 0.2 * input_bytes
    assert peak < 0.3 * input_bytes
