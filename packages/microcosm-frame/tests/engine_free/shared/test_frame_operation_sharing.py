"""``with_weights``, ``place`` and ``select`` build their result through ``_adopt``.

These operations used to rebuild their result with the :class:`Frame`
constructor, which deep-copies every table. They now hand the result's tables
to ``Frame._adopt``. Under pandas copy-on-write the result therefore shares the
memory of every column it does not change with its source frame, and a write
made through pandas to either frame never reaches the other, because each
frame holds its own DataFrame objects. When copy-on-write is off (the pandas
2.x default), ``_adopt`` deep-copies instead; these tests replace the
copy-on-write check to take that path, because pandas 3 always has it on.
"""

import warnings
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import (
    EntitySchema,
    Frame,
    LinkSpec,
    MassChange,
    WeightKind,
    Weights,
)
from microcosm.frame import bundle as bundle_module

_LINKED_SCHEMA = EntitySchema(
    group_entities=("household", "firm"),
    links=(LinkSpec(name="jobs", left_entity="person", right_entity="firm"),),
)
_UNLINKED_SCHEMA = EntitySchema(group_entities=("household", "firm"))


def _entity_tables() -> dict[str, pd.DataFrame]:
    """Four persons in two households and two firms."""

    return {
        "person": pd.DataFrame(
            {
                "person_id": np.array([0, 1, 2, 3], dtype="int64"),
                "person_household_id": np.array([1, 1, 2, 2], dtype="int64"),
                "person_firm_id": np.array([10, 10, 20, 20], dtype="int64"),
                "age": np.array([40, 8, 30, 28], dtype="int64"),
                "income": np.array([50_000.0, 0.0, 30_000.0, 20_000.0]),
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": np.array([1, 2], dtype="int64"),
                "rent": np.array([900.0, 1_200.0]),
                "rooms": np.array([3, 5], dtype="int64"),
                "state": ["CA", "NY"],
            }
        ),
        "firm": pd.DataFrame(
            {
                "firm_id": np.array([10, 20], dtype="int64"),
                "payroll": np.array([1.0e6, 2.0e6]),
            }
        ),
    }


def _jobs_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.array([0, 2, 3], dtype="int64"),
            "firm_id": np.array([10, 20, 20], dtype="int64"),
            "hours": np.array([40.0, 20.0, 35.0]),
        }
    )


def _weights() -> dict[str, Weights]:
    return {
        "household": Weights(values=np.array([100.0, 200.0]), kind=WeightKind.DESIGN)
    }


def _strata() -> pd.Series:
    return pd.Series(["cps", "cps", "puf", "puf"], index=pd.RangeIndex(4))


@pytest.fixture
def linked_frame() -> Frame:
    """A frame with a link table, strata, metadata and one mass-log entry."""

    frame = Frame(
        {**_entity_tables(), "jobs": _jobs_table()},
        _LINKED_SCHEMA,
        _weights(),
        _strata(),
        metadata={"stage": "source"},
    )
    return frame.with_weights(
        "household",
        Weights(values=np.array([150.0, 300.0]), kind=WeightKind.IMPORTANCE),
        mass=MassChange(factor=1.5, reason="test resampling"),
    )


@pytest.fixture
def unlinked_frame() -> Frame:
    """The same frame without its link table, which ``select`` refuses."""

    return Frame(
        _entity_tables(),
        _UNLINKED_SCHEMA,
        _weights(),
        _strata(),
        metadata={"stage": "source"},
    )


@pytest.fixture(autouse=True)
def copy_on_write_on() -> Iterator[None]:
    """Turn copy-on-write on under pandas 2.x; pandas 3 always has it on."""

    if int(pd.__version__.split(".", 1)[0]) >= 3:
        yield
        return
    with pd.option_context("mode.copy_on_write", True):
        yield


@pytest.fixture
def copy_on_write_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``Frame._adopt`` see pandas copy-on-write as off."""

    monkeypatch.setattr(bundle_module, "_pandas_copy_on_write", lambda: False)


def _reweighted(frame: Frame) -> Frame:
    current = frame.weights_for("household")
    return frame.with_weights(
        "household",
        Weights(values=current.values * 2.0, kind=WeightKind.CALIBRATED),
        mass=MassChange(factor=2.0, reason="test calibration"),
    )


def _income_to_household(frame: Frame) -> Frame:
    return frame.place("income", "household", how="sum")


def _rent_to_person(frame: Frame) -> Frame:
    return frame.place("rent", "person", how="broadcast")


def _every_person(frame: Frame) -> Frame:
    return frame.select(np.ones(frame.n("person"), dtype=bool))


def _first_household(frame: Frame) -> Frame:
    return frame.select(frame.person["person_household_id"].to_numpy() == 1)


#: Each operation, with the fixture whose frame it runs on.
_OPERATIONS: dict[str, tuple[str, Callable[[Frame], Frame]]] = {
    "with_weights": ("linked_frame", _reweighted),
    "place person to household": ("linked_frame", _income_to_household),
    "place household to person": ("linked_frame", _rent_to_person),
    "select every person": ("unlinked_frame", _every_person),
    "select one household": ("unlinked_frame", _first_household),
}


def _source_and_result(request: pytest.FixtureRequest, name: str):
    fixture, operation = _OPERATIONS[name]
    source = request.getfixturevalue(fixture)
    return source, operation(source)


def _tables_of(frame: Frame) -> dict[str, pd.DataFrame]:
    """Every entity table and link table the frame stores, by name."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables.update({name: frame.link(name) for name in frame.links})
    return tables


def _numeric_columns_sharing_memory(
    left: dict[str, pd.DataFrame], right: dict[str, pd.DataFrame]
) -> dict[tuple[str, str], bool]:
    """For each numeric column both table sets carry, whether its data is shared."""

    sharing: dict[tuple[str, str], bool] = {}
    for table_name, left_table in left.items():
        right_table = right.get(table_name)
        if right_table is None:
            continue
        for column in left_table.select_dtypes("number").columns:
            if column not in right_table.columns:
                continue
            sharing[(table_name, column)] = np.shares_memory(
                left_table[column].to_numpy(), right_table[column].to_numpy()
            )
    return sharing


def _snapshot(frame: Frame) -> dict[str, object]:
    """Deep copies of everything a write could change, for a later comparison."""

    return {
        "tables": {
            name: table.copy(deep=True) for name, table in _tables_of(frame).items()
        },
        "weights": {
            entity: frame.weights_for(entity).values.copy()
            for entity in frame.weighted_entities
        },
        "strata": frame.strata.copy(deep=True),
    }


def _assert_matches_snapshot(frame: Frame, snapshot: dict[str, object]) -> None:
    tables = _tables_of(frame)
    assert tables.keys() == snapshot["tables"].keys()
    for name, expected in snapshot["tables"].items():
        pd.testing.assert_frame_equal(tables[name], expected, check_exact=True)
    for entity, expected in snapshot["weights"].items():
        np.testing.assert_array_equal(frame.weights_for(entity).values, expected)
    pd.testing.assert_series_equal(frame.strata, snapshot["strata"], check_exact=True)
    frame.revalidate()


def _write_every_numeric_column(frame: Frame) -> None:
    """Write through pandas into the first row of every numeric column."""

    for table in _tables_of(frame).values():
        for column in table.select_dtypes("number").columns:
            table.loc[table.index[0], column] = -7


def test_with_weights_shares_every_column_with_the_source(linked_frame) -> None:
    reweighted = _reweighted(linked_frame)

    sharing = _numeric_columns_sharing_memory(
        _tables_of(reweighted), _tables_of(linked_frame)
    )
    assert ("jobs", "hours") in sharing
    assert all(sharing.values()), sharing


@pytest.mark.parametrize(
    ("operation", "owner", "destination"),
    [
        (_income_to_household, "person", "household"),
        (_rent_to_person, "household", "person"),
    ],
    ids=["person to household", "household to person"],
)
def test_place_shares_every_table_it_does_not_change(
    linked_frame, operation, owner: str, destination: str
) -> None:
    placed = operation(linked_frame)

    untouched = {
        name: table
        for name, table in _tables_of(placed).items()
        if name not in (owner, destination)
    }
    assert set(untouched) == {"firm", "jobs"}
    sharing = _numeric_columns_sharing_memory(untouched, _tables_of(linked_frame))
    assert sharing and all(sharing.values()), sharing
    destination_sharing = _numeric_columns_sharing_memory(
        {destination: placed.table(destination)},
        {destination: linked_frame.table(destination)},
    )
    assert destination_sharing and all(destination_sharing.values()), (
        destination_sharing
    )
    # The table the column moves out of keeps sharing its other columns.
    owner_sharing = _numeric_columns_sharing_memory(
        {owner: placed.table(owner)}, {owner: linked_frame.table(owner)}
    )
    assert owner_sharing and all(owner_sharing.values()), owner_sharing


def test_select_keeping_every_person_shares_every_column(unlinked_frame) -> None:
    selected = _every_person(unlinked_frame)

    sharing = _numeric_columns_sharing_memory(
        _tables_of(selected), _tables_of(unlinked_frame)
    )
    assert sharing and all(sharing.values()), sharing


@pytest.mark.parametrize("operation", list(_OPERATIONS))
def test_every_result_table_is_a_distinct_object(request, operation: str) -> None:
    source, result = _source_and_result(request, operation)

    source_tables = _tables_of(source)
    for name, table in _tables_of(result).items():
        assert table is not source_tables[name], name


@pytest.mark.parametrize("operation", list(_OPERATIONS))
def test_results_carry_metadata_and_the_mass_log(request, operation: str) -> None:
    source, result = _source_and_result(request, operation)

    assert result.metadata == source.metadata
    if operation == "with_weights":
        assert result.mass_log[:-1] == source.mass_log
        assert len(result.mass_log) == len(source.mass_log) + 1
    else:
        assert result.mass_log == source.mass_log
    result.revalidate()


@pytest.mark.parametrize("operation", list(_OPERATIONS))
def test_a_pandas_write_to_the_result_leaves_the_source_unchanged(
    request, operation: str
) -> None:
    source, result = _source_and_result(request, operation)
    before = _snapshot(source)

    _write_every_numeric_column(result)

    _assert_matches_snapshot(source, before)


@pytest.mark.parametrize("operation", list(_OPERATIONS))
def test_a_pandas_write_to_the_source_leaves_the_result_unchanged(
    request, operation: str
) -> None:
    source, result = _source_and_result(request, operation)
    before = _snapshot(result)

    _write_every_numeric_column(source)

    _assert_matches_snapshot(result, before)


@pytest.mark.parametrize("operation", list(_OPERATIONS))
def test_operations_deep_copy_when_copy_on_write_is_off(
    request, copy_on_write_off: None, operation: str
) -> None:
    source, result = _source_and_result(request, operation)

    sharing = _numeric_columns_sharing_memory(_tables_of(result), _tables_of(source))
    assert sharing, "the comparison found no numeric columns"
    assert not any(sharing.values()), sharing


def _fragmented_frame() -> Frame:
    """A frame whose person and household tables hold one block per column.

    A table built from a mapping of Series with ``copy=False`` keeps every
    column in its own internal block, as the checkpoint loader's tables do.
    pandas emits ``PerformanceWarning`` when a column is inserted into a table
    with more than 100 blocks.
    """

    tables = _entity_tables()
    fragmented = {}
    for name in ("person", "household"):
        table = tables[name]
        columns = {column: table[column] for column in table.columns}
        for position in range(120):
            columns[f"{name}_filler_{position}"] = pd.Series(
                np.full(len(table), float(position)), index=table.index
            )
        fragmented[name] = pd.DataFrame(columns, index=table.index, copy=False)
    tables.update(fragmented)
    return Frame._adopt(tables, _UNLINKED_SCHEMA, _weights(), _strata())


@pytest.mark.parametrize(
    "operation",
    [_income_to_household, _rent_to_person],
    ids=["person to household", "household to person"],
)
def test_place_into_a_fragmented_table_emits_no_performance_warning(
    operation,
) -> None:
    frame = _fragmented_frame()

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        placed = operation(frame)

    placed.revalidate()


@pytest.mark.parametrize(
    ("operation", "column", "destination"),
    [
        (_income_to_household, "income", "household"),
        (_rent_to_person, "rent", "person"),
    ],
    ids=["person to household", "household to person"],
)
def test_place_into_a_fragmented_table_keeps_columns_order_and_dtypes(
    operation, column: str, destination: str
) -> None:
    fragmented = _fragmented_frame()
    consolidated = Frame(
        _tables_of(fragmented),
        fragmented.schema,
        {"household": fragmented.weights_for("household")},
        fragmented.strata,
    )

    from_fragmented = operation(fragmented)
    from_consolidated = operation(consolidated)

    for name in from_consolidated.entities:
        pd.testing.assert_frame_equal(
            from_fragmented.table(name),
            from_consolidated.table(name),
            check_exact=True,
        )
    assert list(from_fragmented.table(destination).columns)[-1] == column
