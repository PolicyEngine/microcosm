"""Kernel contexts read the population's own column memory, read-only (#1679).

The executor hands each kernel new pandas objects whose buffers are read-only
views of the population's buffers, so a node that reads every row holds no
second copy of its input tables and a node with a row mask holds only the
selected rows. A write through pandas, or an item write into an array a column
hands out, raises or changes only the kernel's own objects; the population's
bytes stay the same. ``np.<ufunc>.at`` ignores the read-only flag: such a write
reaches the population's memory, and the executor's before/after comparison of
the context rejects the node before anything derived from it is applied or
stored. Every frame here is invented.
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
    "string_python": lambda: pd.array(
        ["a", "b", None, "d", "e", "f"], dtype=pd.StringDtype("python")
    ),
    "string_pyarrow": lambda: pd.array(
        ["a", "b", None, "d", "e", "f"], dtype=pd.StringDtype("pyarrow")
    ),
    "str": lambda: pd.array(["a", "b", "c", "d", "e", "f"], dtype="str"),
    "object": lambda: np.array(["a", 1, 2.5, "d", "e", "f"], dtype=object),
}
COLUMNS = tuple(COLUMN_VALUES)
NUMPY_COLUMNS = ("float64", "float32", "int64", "int32", "bool", "object")
SELECTED = np.array([True, False, True, True, False, True])


def _kinds_frame(*, strata_dtype: str = "object") -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype=np.int64),
            "person_household_id": np.array([10, 10, 20, 20, 30, 30], dtype=np.int64),
            "selected": SELECTED.copy(),
            **{column: make() for column, make in COLUMN_VALUES.items()},
        }
    )
    person.attrs["provenance"] = {"source": "invented"}
    household = pd.DataFrame({"household_id": np.array([10, 20, 30], dtype=np.int64)})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b", "b", "c", "c"], name="stratum", dtype=strata_dtype),
    )


def _kinds_population(*, strata_dtype: str = "object") -> Population:
    return Population.from_frame(_kinds_frame(strata_dtype=strata_dtype), "survey")


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


def _writable_views(series: pd.Series) -> list[str]:
    """Names of the arrays a column hands out that share its memory and are writable."""

    array = series.array
    stored = [part for part in _memory_of(series) if isinstance(part, np.ndarray)]
    candidates: dict[str, object] = {
        "to_numpy()": series.to_numpy(),
        "values": series.values,
        "np.asarray(array)": np.asarray(array),
    }
    if isinstance(array, _MASKED_ARRAY_TYPES):
        candidates.update({"_data": array._data, "_mask": array._mask})
    if isinstance(array, pd.Categorical):
        candidates.update({"_ndarray": array._ndarray, "codes": array.codes})
    return [
        name
        for name, candidate in candidates.items()
        if isinstance(candidate, np.ndarray)
        and candidate.flags.writeable
        and any(np.shares_memory(candidate, part) for part in stored)
    ]


# ----------------------------------------------------------------------
# What a context holds
# ----------------------------------------------------------------------


@pytest.mark.parametrize("column", COLUMNS)
def test_an_every_row_context_shares_the_population_column_memory(
    column: str,
) -> None:
    population = _kinds_population()
    context = _project(population)
    shared = zip(
        _memory_of(context.tables["person"][column]),
        _memory_of(population.frame.table("person")[column]),
        strict=True,
    )
    assert all(_shares(ours, theirs) for ours, theirs in shared)


@pytest.mark.parametrize("strata_dtype", ["object", "category"])
def test_an_every_row_context_shares_the_population_strata(strata_dtype: str) -> None:
    population = _kinds_population(strata_dtype=strata_dtype)
    context = _project(population)
    shared = zip(
        _memory_of(context.strata), _memory_of(population.frame.strata), strict=True
    )
    assert all(_shares(ours, theirs) for ours, theirs in shared)


@pytest.mark.parametrize("rows", [ROWS_ALL, "selected"])
def test_no_context_buffer_is_writable(rows: str) -> None:
    for strata_dtype in ("object", "category"):
        context = _project(_kinds_population(strata_dtype=strata_dtype), rows=rows)
        table = context.tables["person"]
        for column in table.columns:
            assert _writable_views(table[column]) == [], column
        assert _writable_views(context.strata) == []
        if not isinstance(table.index, pd.RangeIndex):
            assert not np.asarray(table.index).flags.writeable


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


@pytest.mark.parametrize("rows", [ROWS_ALL, "selected"])
@pytest.mark.parametrize("strata_dtype", ["object", "category"])
def test_a_context_holds_what_a_deep_copy_of_the_slice_holds(
    rows: str, strata_dtype: str
) -> None:
    """Kernels see the same columns, order, dtypes, index, values and attrs."""

    population = _kinds_population(strata_dtype=strata_dtype)
    frame = population.frame
    context = _project(population, rows=rows)
    mask = (
        np.ones(frame.n("person"), dtype=bool)
        if rows == ROWS_ALL
        else frame.table("person")[rows].to_numpy()
    )
    columns = ["person_id", "person_household_id", *COLUMNS]
    if rows != ROWS_ALL:
        columns.append(rows)
    expected = frame.table("person").loc[mask, columns].copy(deep=True)
    table = context.tables["person"]
    pd.testing.assert_frame_equal(
        table, expected, check_exact=True, check_index_type=True
    )
    assert type(table.index) is type(expected.index)
    assert table.columns.equals(expected.columns)
    assert table.columns.dtype == expected.columns.dtype
    assert table.attrs == expected.attrs
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
}


@pytest.mark.parametrize("rows", [ROWS_ALL, "selected"])
@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("write", WRITES)
def test_a_write_into_a_context_leaves_the_population_byte_identical(
    write: str, column: str, rows: str
) -> None:
    population = _kinds_population()
    before = _population_storage(population)
    context = _project(population, rows=rows)
    try:
        WRITES[write](context.tables["person"], column)
    except (TypeError, ValueError, AssertionError):
        pass  # refusing the write is one of the two accepted outcomes
    assert _population_storage(population) == before


@pytest.mark.parametrize("strata_dtype", ["object", "category"])
def test_a_write_into_the_context_strata_leaves_the_population_strata(
    strata_dtype: str,
) -> None:
    population = _kinds_population(strata_dtype=strata_dtype)
    before = _population_storage(population)
    for write in (
        lambda strata: strata.__setitem__(0, strata.iloc[2]),
        lambda strata: np.asarray(strata.array).__setitem__(0, strata.iloc[2]),
        lambda strata: strata.values.__setitem__(0, strata.iloc[2]),
        lambda strata: strata.to_numpy().__setitem__(0, strata.iloc[2]),
    ):
        context = _project(population)
        try:
            write(context.strata)
        except (TypeError, ValueError):
            pass
        assert _population_storage(population) == before


@pytest.mark.parametrize(
    "column", ["float64", "float32", "int64", "int32", "Int64_no_missing"]
)
def test_a_ufunc_at_write_into_a_context_changes_the_context_digest(
    column: str,
) -> None:
    """``np.add.at`` ignores the read-only flag; the digest comparison sees it."""

    context = _project(_kinds_population())
    before = graph_executor._context_digest(context)
    np.add.at(context.tables["person"][column].to_numpy(), [0], 1)
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
        pd.Series(["a", "a", "b", "b", "c", "c"], name="stratum"),
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


def _registry(write: Callable[[KernelContext], None]) -> KernelRegistry:
    def compute(context: KernelContext) -> KernelResult:
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
        _Kernel("writer@1", Capabilities(Determinism.DETERMINISTIC), compute)
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


@pytest.mark.parametrize(
    "column", ["float64", "int32", "bool", "Int64", "boolean", "string_python"]
)
@pytest.mark.parametrize(
    "write",
    [
        "loc cell",
        "column assignment",
        "item write through to_numpy()",
        "item write through values",
        "item write through array",
        "item write through np.asarray(array)",
    ],
)
def test_a_kernel_that_writes_into_its_inputs_changes_nothing_it_reads(
    tmp_path: Path, write: str, column: str
) -> None:
    """The write raises, lands in the kernel's own copy, or rejects the node.

    ``to_numpy()`` and ``np.asarray(array)`` of a nullable column with a
    missing value return a converted copy, so a write into one is the
    kernel's own and the run completes.
    """

    def vandal(context: KernelContext) -> None:
        WRITES[write](context.tables["person"], column)

    observe, held, admitted = _watch_create()
    try:
        _run(
            _writer_graph(column),
            _registry(vandal),
            ContentStore(tmp_path / "store"),
            _source_directory(tmp_path / "source"),
            observe,
        )
    except NodeRejected as error:
        assert "'writer'" in str(error)
    (population,) = held
    assert _population_storage(population) == admitted[0]


@pytest.mark.parametrize(
    "column", ["float64", "int64", "int32", "float32", "Int64_no_missing"]
)
def test_a_ufunc_at_write_is_rejected_and_the_store_keeps_the_source_bytes(
    tmp_path: Path, column: str
) -> None:
    """The one write that passes a read-only view: rejected, and nothing stored."""

    source = _source_directory(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    create_only = Graph("toy", (SOURCE,), (CREATE,))
    first = _run(create_only, _registry(lambda context: None), store, source)
    stored_before = _stored_objects(store)
    landed: list[bool] = []

    def add_at(context: KernelContext) -> None:
        values = context.tables["person"][column].to_numpy()
        before = values.copy()
        np.add.at(values, [0], 1)
        landed.append(not np.array_equal(values, before))

    with pytest.raises(NodeRejected, match="mutated its input context"):
        _run(_writer_graph(column), _registry(add_at), store, source)
    assert landed == [True]
    stored_after = _stored_objects(store)
    assert {name: stored_after.get(name) for name in stored_before} == stored_before

    again = _run(create_only, _registry(lambda context: None), store, source)
    assert again.nodes["survey"].hit
    reloaded = again.population("survey").person
    original = first.population("survey").person
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
    for number in range(12):
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


def _traced_projection(population: Population, rows: str) -> tuple[int, int]:
    """Bytes the context holds after projection, and the peak during it."""

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

    With the deep copy (before #1679) the context held 1.18 times the bytes of
    the input columns and the projection peaked at 1.31 times. What is left
    is the row mask and the effective weights, which are still copied, and
    pandas' objects.
    """

    population = _wide_population()
    held, peak = _traced_projection(population, ROWS_ALL)
    input_bytes = _input_bytes(population)
    assert held < 0.15 * input_bytes
    assert peak < 0.2 * input_bytes


def test_a_masked_context_allocates_one_copy_of_the_selected_rows() -> None:
    """Half the rows: ``.loc`` copies them once, and nothing copies them again.

    One copy of half the rows is 0.5 times the input columns. With the deep
    copy (before #1679) the projection peaked at 1.04 times.
    """

    population = _wide_population()
    held, peak = _traced_projection(population, "half")
    input_bytes = _input_bytes(population)
    assert held < 0.65 * input_bytes
    assert peak < 0.75 * input_bytes
