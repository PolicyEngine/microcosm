"""UK kernels build their Frame from one copy of their input tables.

``_minimal_frame`` (the UK stage kernels) and ``context_frame`` (the UK
population, target, calibration, national and terminal kernels) copy each of
the executor's context tables once, with ``copy_table``, in the block layout
``copy(deep=True)`` gave them, and build the Frame with ``Frame.adopt``, which
keeps the copies without copying them again. Before, pandas' deep copy of a
context table, which holds one block per column, briefly held about three
copies of it, and the ``Frame`` constructor copied the result once more. Every
population here is invented.
"""

from __future__ import annotations

import gc
import hashlib
import tracemalloc
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.graph_kernels import _minimal_frame
from microcosm.build.uk_runtime.graph_population import context_frame
from microcosm.build.uk_runtime.national_frame import UK_NATIONAL_SCHEMA
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import executor as graph_executor
from microcosm.graph.decl import Node, Owned, Slice
from microcosm.graph.population import Population
from test_support.microcosm_frame.table_layouts import (
    block_layout,
    column_arrays,
    memory_digest,
)

_NODE_KEY = hashlib.sha256(b"test_uk_kernel_frame_copies").hexdigest()
_PERSONS = 12
# Row 1 of every column differs from row 0, so a write of row 1 into row 0
# changes bytes wherever it lands.
PERSON_VALUES: dict[str, Callable[[], object]] = {
    "float64": lambda: np.linspace(-2.5, 3.0, _PERSONS),
    "int64": lambda: np.array([5, 1, 4, 2, 8, 6, 3, 9, 0, 7, 11, 10]),
    "float64_second": lambda: np.linspace(9.0, -1.5, _PERSONS),
    "float32": lambda: np.linspace(0.5, 6.0, _PERSONS, dtype=np.float32),
    "int32": lambda: np.arange(_PERSONS, 0, -1, dtype=np.int32),
    "bool": lambda: np.arange(_PERSONS) % 3 == 1,
    "Int64": lambda: pd.array([5, 1, None, 2, 8, 6, 3, 9, 0, 7, 1, 2], dtype="Int64"),
    "Int64_no_missing": lambda: pd.array(np.arange(_PERSONS)[::-1], dtype="Int64"),
    "boolean": lambda: pd.array(
        [True, False, None, True, False, True] * 2, dtype="boolean"
    ),
    "boolean_no_missing": lambda: pd.array(
        [True, False, False, True, False, True] * 2, dtype="boolean"
    ),
    "string": lambda: pd.array(
        ["a", "b", None, "d", "e", "f"] * 2, dtype=pd.StringDtype("python", pd.NA)
    ),
    "str": lambda: pd.array(list("abcdefghijkl"), dtype="str"),
    "category": lambda: pd.Categorical(list("xyxzyxxyxzyx")),
}
PERSON_COLUMNS = tuple(PERSON_VALUES)
STRATA_VALUES: dict[str, Callable[[], object]] = {
    "object": lambda: np.array(list("aabbccaabbcc"), dtype=object),
    "category": lambda: pd.Categorical(list("aabbccaabbcc")),
}


def _population(strata_kind: str = "object") -> Population:
    """Twelve persons in six benefit units and three households."""

    person = pd.DataFrame(
        {
            "person_id": np.arange(_PERSONS, dtype=np.int64),
            "person_benunit_id": np.arange(_PERSONS, dtype=np.int64) // 2,
            "person_household_id": np.arange(_PERSONS, dtype=np.int64) // 4,
            **{name: values() for name, values in PERSON_VALUES.items()},
        }
    )
    frame = Frame(
        {
            "person": person,
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(6, dtype=np.int64),
                    "benunit_rent": np.linspace(100.0, 600.0, 6),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.arange(3, dtype=np.int64),
                    "household_income": np.array([3.5, -1.0, 2.25]),
                    "household_region": pd.array(
                        ["north", "south", "east"], dtype=pd.StringDtype("python")
                    ),
                }
            ),
        },
        UK_NATIONAL_SCHEMA,
        {"household": Weights(np.array([1.5, 2.0, 0.5]), WeightKind.DESIGN)},
        pd.Series(STRATA_VALUES[strata_kind](), name="stratum"),
        metadata={"time_period": "2024"},
    )
    return Population.from_frame(frame, "survey")


def _context(population: Population, person_columns=PERSON_COLUMNS):
    """The executor's context for a node reading every row of every entity.

    The person columns are declared out of the table's own order, so a frame
    that followed the table's order instead of the declared one would be seen.
    """

    node = Node(
        "reader",
        "uk.test.reader@1",
        inputs=(
            Slice("person", tuple(reversed(person_columns)), rows="all"),
            Slice("benunit", ("benunit_rent",), rows="all"),
            Slice("household", ("household_region", "household_income"), rows="all"),
        ),
        outputs=(Owned("person", "written", "float64"),),
        params={"time_period": "2024"},
        population="survey",
    )
    return graph_executor._project_context(
        node,
        population,
        key=_NODE_KEY,
        sources={},
        tolerances={},
        numerics={},
    )


def _frame_before_one_copy(context) -> Frame:
    """The Frame ``_minimal_frame`` built before it copied each column once."""

    return Frame(
        {
            entity: context.tables[entity].copy(deep=True)
            for entity in UK_NATIONAL_SCHEMA.entities
        },
        UK_NATIONAL_SCHEMA,
        {"household": context.weights["household"]},
        context.strata.copy(deep=True),
        metadata={"time_period": str(context.params.get("time_period", "2024"))},
    )


def _context_frame_before_one_copy(context) -> Frame:
    """The Frame ``context_frame`` built before it copied each column once."""

    return Frame(
        {
            entity: context.tables[entity]
            .loc[
                :,
                list(
                    context.frame_column_order.get(
                        entity, tuple(context.tables[entity].columns)
                    )
                ),
            ]
            .copy(deep=True)
            for entity in UK_NATIONAL_SCHEMA.entities
        },
        UK_NATIONAL_SCHEMA,
        {"household": context.weights["household"]},
        context.strata.copy(deep=True),
        mass_log=getattr(context, "frame_mass_log", ()),
        metadata=getattr(context, "frame_metadata", {})
        or {"time_period": str(context.params["time_period"])},
    )


BUILDERS: dict[str, tuple[Callable, Callable]] = {
    "_minimal_frame": (_minimal_frame, _frame_before_one_copy),
    "context_frame": (context_frame, _context_frame_before_one_copy),
}


@pytest.mark.parametrize("strata_kind", STRATA_VALUES)
@pytest.mark.parametrize("builder", BUILDERS)
def test_the_kernel_frame_equals_the_frame_built_before(builder, strata_kind):
    build, build_before = BUILDERS[builder]
    context = _context(_population(strata_kind))

    frame = build(context)
    before = build_before(context)

    for entity in UK_NATIONAL_SCHEMA.entities:
        table, table_before = frame.table(entity), before.table(entity)
        pd.testing.assert_frame_equal(
            table,
            table_before,
            check_exact=True,
            check_index_type=True,
            check_column_type=True,
            check_flags=True,
        )
        assert table.attrs == table_before.attrs
        # The same blocks, each column at the same offset in an equally shaped
        # block, so BLAS products see the layout they saw before.
        assert sorted(map(repr, block_layout(table))) == sorted(
            map(repr, block_layout(table_before))
        )
    assert block_layout(frame.table("person")) == block_layout(before.table("person"))
    pd.testing.assert_series_equal(
        frame.strata, before.strata, check_exact=True, check_index_type=True
    )
    assert frame.weights_for("household") is before.weights_for("household")
    assert frame.metadata == before.metadata
    assert frame.mass_log == before.mass_log


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_kernel_frame_shares_no_memory_with_its_context(builder):
    build, _ = BUILDERS[builder]
    context = _context(_population("category"))

    frame = build(context)

    for entity in UK_NATIONAL_SCHEMA.entities:
        context_arrays = [
            array
            for _, column in context.tables[entity].items()
            for array in column_arrays(column)
        ]
        for name, column in frame.table(entity).items():
            assert not any(
                np.shares_memory(array, context_array)
                for array in column_arrays(column)
                for context_array in context_arrays
            ), (entity, name)
    assert not any(
        np.shares_memory(array, context_array)
        for array in column_arrays(frame.strata)
        for context_array in column_arrays(context.strata)
    )


def _write_with_loc(values: pd.DataFrame | pd.Series, name: str | None) -> None:
    if name is None:
        values.loc[values.index[0]] = values.iloc[1]
    else:
        values.loc[values.index[0], name] = values[name].iloc[1]


def _column(values: pd.DataFrame | pd.Series, name: str | None) -> pd.Series:
    return values if name is None else values[name]


def _write_into_array(values, name):
    array = _column(values, name).array
    array[0] = array[1]


def _write_into_values(values, name):
    array = _column(values, name).values
    array[0] = array[1]


def _write_into_to_numpy(values, name):
    array = _column(values, name).to_numpy()
    array[0] = array[1]


def _add_at_into_to_numpy(values, name):
    np.add.at(_column(values, name).to_numpy(), [0], 1)


def _write_into_index(values, name):
    np.asarray(values.index)[0] = 99


def _write_into_categories(values, name):
    categories = _column(values, name).cat.categories.values
    categories[0] = categories[1] + "_written"


#: Writes a kernel might make into a column of its own Frame. Some raise for
#: some column types (a read-only array, a ufunc a type lacks); the rest must
#: change only the Frame.
WRITES: dict[str, Callable] = {
    "loc": _write_with_loc,
    "array item": _write_into_array,
    "values item": _write_into_values,
    "to_numpy item": _write_into_to_numpy,
    "np.add.at into to_numpy": _add_at_into_to_numpy,
    "categories item": _write_into_categories,
}
#: The strata take the column writes and a write into their index.
STRATA_WRITES: dict[str, Callable] = {
    **{write: WRITES[write] for write in WRITES if write != "np.add.at into to_numpy"},
    "index item": _write_into_index,
}


def _applies(write: str, name: str) -> bool:
    if write == "categories item":
        return name == "category"
    if write == "np.add.at into to_numpy":
        return name in {"float64", "int64", "float32", "int32"}
    return True


def _population_digests(population: Population) -> dict[str, str]:
    frame = population.frame
    digests = {entity: memory_digest(frame.table(entity)) for entity in frame.entities}
    digests["strata"] = memory_digest(frame.strata)
    return digests


def _context_digests(context) -> tuple[bytes, dict[str, str]]:
    return graph_executor._context_digest(context), {
        **{entity: memory_digest(table) for entity, table in context.tables.items()},
        "strata": memory_digest(context.strata),
    }


@pytest.mark.parametrize(
    ("builder", "write", "name"),
    [
        (builder, write, name)
        for builder in BUILDERS
        for write in WRITES
        for name in PERSON_COLUMNS
        if _applies(write, name)
    ],
)
def test_a_write_into_a_person_column_changes_neither_context_nor_population(
    builder, write, name
):
    build, _ = BUILDERS[builder]
    population = _population()
    context = _context(population)
    population_before = _population_digests(population)
    context_before = _context_digests(context)
    frame = build(context)

    try:
        WRITES[write](frame.table("person"), name)
    except (TypeError, ValueError):
        pass

    assert _context_digests(context) == context_before
    assert _population_digests(population) == population_before


@pytest.mark.parametrize("entity", UK_NATIONAL_SCHEMA.entities)
@pytest.mark.parametrize("builder", BUILDERS)
def test_a_write_into_a_table_index_changes_neither_context_nor_population(
    builder, entity
):
    # The tables here have a RangeIndex. A view of one shares the cache holding
    # its values once materialized, which np.asarray returns writable.
    build, _ = BUILDERS[builder]
    population = _population()
    context = _context(population)
    population_before = _population_digests(population)
    context_before = _context_digests(context)
    frame = build(context)

    try:
        _write_into_index(frame.table(entity), None)
    except ValueError:
        pass

    assert _context_digests(context) == context_before
    assert _population_digests(population) == population_before


@pytest.mark.parametrize(
    ("builder", "write", "strata_kind"),
    [
        (builder, write, strata_kind)
        for builder in BUILDERS
        for write in STRATA_WRITES
        for strata_kind in STRATA_VALUES
        if write != "categories item" or strata_kind == "category"
    ],
)
def test_a_write_into_the_strata_changes_neither_context_nor_population(
    builder, write, strata_kind
):
    build, _ = BUILDERS[builder]
    population = _population(strata_kind)
    context = _context(population)
    population_before = _population_digests(population)
    context_before = _context_digests(context)
    frame = build(context)

    try:
        STRATA_WRITES[write](frame.strata, None)
    except (TypeError, ValueError):
        pass

    assert _context_digests(context) == context_before
    assert _population_digests(population) == population_before


def _measured_population(rows: int) -> tuple[Population, tuple[str, ...]]:
    """Persons with 20 float64 and 2 int64 columns, read by every node."""

    generator = np.random.default_rng(11)
    person = {
        "person_id": np.arange(rows, dtype=np.int64),
        "person_benunit_id": np.arange(rows, dtype=np.int64) // 2,
        "person_household_id": np.arange(rows, dtype=np.int64) // 4,
        "int64_0": generator.integers(0, 100, rows),
        "int64_1": generator.integers(0, 100, rows),
        **{f"float64_{number}": generator.normal(size=rows) for number in range(20)},
    }
    households = rows // 4
    frame = Frame(
        {
            "person": pd.DataFrame(person),
            "benunit": pd.DataFrame(
                {"benunit_id": np.arange(rows // 2, dtype=np.int64)}
            ),
            "household": pd.DataFrame(
                {"household_id": np.arange(households, dtype=np.int64)}
            ),
        },
        UK_NATIONAL_SCHEMA,
        {"household": Weights(np.ones(households), WeightKind.DESIGN)},
        pd.Series(
            np.where(np.arange(rows) % 3 == 0, "a", "b"), dtype=object, name="stratum"
        ),
    )
    read = ("int64_0", "int64_1", *(f"float64_{number}" for number in range(20)))
    return Population.from_frame(frame, "survey"), read


def _measured_node(read: tuple[str, ...]) -> Node:
    return Node(
        "reader",
        "uk.test.reader@1",
        inputs=(
            Slice("person", read, rows="all"),
            Slice("benunit", ("benunit_id",), rows="all"),
            Slice("household", ("household_id",), rows="all"),
        ),
        outputs=(Owned("person", "written", "float64"),),
        params={"time_period": "2024"},
        population="survey",
    )


@pytest.mark.parametrize("builder", BUILDERS)
def test_projection_to_kernel_frame_holds_one_copy_at_its_peak(builder):
    """A regression bound on the memory from projection to the kernel's Frame.

    The bytes copied are every column of every context table and the strata.
    Before this change the step peaked at about 2.5 times them: pandas' deep
    copy of a table of one block per column holds the column copies, their
    stacked block and its reordered copy, and the Frame constructor copied the
    result again. Now each column is copied once, and what remains above that
    is the context's own small allocations (its weights copy and row masks)
    and the Frame's checks of ids and memberships.
    """

    build, _ = BUILDERS[builder]
    population, read = _measured_population(200_000)
    node = _measured_node(read)

    def project():
        return graph_executor._project_context(
            node,
            population,
            key=_NODE_KEY,
            sources={},
            tolerances={},
            numerics={},
        )

    build(project())  # loads what the first call loads, outside the measurement
    gc.collect()
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        context = project()
        frame = build(context)
        held, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    copied_bytes = sum(
        array.nbytes
        for table in context.tables.values()
        for _, column in table.items()
        for array in column_arrays(column)
    ) + sum(array.nbytes for array in column_arrays(context.strata))
    assert frame.table("person").shape == (200_000, 25)
    assert (held - baseline) / copied_bytes < 1.15
    assert (peak - baseline) / copied_bytes < 1.35
