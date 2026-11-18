"""The check that finds raw writes into column memory frames share.

Frames built by ``with_weights``, ``place``, ``select``, ``replace_tables`` and
``with_columns`` share unchanged columns under pandas copy-on-write. Copy-on-write
keeps pandas writes apart, but a raw write into the shared memory changes every
frame holding it. These tests make those raw writes on purpose and check that
the comparison run before output is written names the changed column, and that
writes pandas keeps apart never trip it.
"""

import gc
import warnings
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import (
    EntitySchema,
    Frame,
    MassChange,
    SharedColumnChangedError,
    WeightKind,
    Weights,
    check_shared_columns,
    engine_tables,
    put_frame_table,
)
from microcosm.frame import bundle as bundle_module
from microcosm.frame import shared_columns as shared_columns_module

_SCHEMA = EntitySchema(group_entities=("household",))
_PERSON_COUNT = 4


def _source_frame() -> Frame:
    """Two households of two people, with one column of each stored form."""

    person = pd.DataFrame(
        {
            "person_id": np.arange(_PERSON_COUNT, dtype="int64"),
            "person_household_id": np.array([1, 1, 2, 2], dtype="int64"),
            "income": np.array([50_000.0, 0.0, 30_000.0, 20_000.0]),
            "age": np.array([40, 8, 30, 28], dtype="int64"),
            "count_complete": pd.array([1, 2, 3, 4], dtype="Int64"),
            "count_with_gap": pd.array([1, None, 3, 4], dtype="Int64"),
            "amount_complete": pd.array([1.5, 2.5, 3.5, 4.5], dtype="Float64"),
            "flag_complete": pd.array([True, False, True, False], dtype="boolean"),
            "region": pd.Categorical(["north", "south", "north", "south"]),
            "label_python": pd.array(
                ["a", "b", "c", "d"], dtype=pd.StringDtype("python")
            ),
            "label_object": pd.Series(["a", "b", "c", "d"], dtype=object),
            "tags": pd.Series([["a"], ["b"], ["c"], ["d"]], dtype=object),
            "born": pd.date_range("2000-01-01", periods=_PERSON_COUNT),
            "born_utc": pd.date_range("2000-01-01", periods=_PERSON_COUNT, tz="UTC"),
            "born_month": pd.period_range("2000-01", periods=_PERSON_COUNT, freq="M"),
            "rare_count": pd.arrays.SparseArray([0, 1, 0, 2]),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array([1, 2], dtype="int64"),
            "rent": np.array([900.0, 1_200.0]),
        }
    )
    return Frame(
        {"person": person, "household": household},
        _SCHEMA,
        {"household": Weights(np.array([100.0, 200.0]), WeightKind.DESIGN)},
        pd.Series(["cps", "cps", "puf", "puf"], dtype=object),
    )


def _reweighted(frame: Frame) -> Frame:
    current = frame.weights_for("household")
    return frame.with_weights(
        "household",
        Weights(current.values * 2.0, WeightKind.CALIBRATED),
        mass=MassChange(factor=2.0, reason="test calibration"),
    )


def _with_rent_on_people(frame: Frame) -> Frame:
    return frame.place("rent", "person", how="broadcast")


def _every_person(frame: Frame) -> Frame:
    return frame.select(np.ones(frame.n("person"), dtype=bool))


def _every_table_replaced_by_itself(frame: Frame) -> Frame:
    return frame.replace_tables()


def _with_new_column(frame: Frame) -> Frame:
    return frame.with_columns("person", new_measure=np.ones(frame.n("person")))


_SHARING_OPERATIONS: dict[str, Callable[[Frame], Frame]] = {
    "with_weights": _reweighted,
    "place": _with_rent_on_people,
    "select every person": _every_person,
    "replace_tables": _every_table_replaced_by_itself,
    "with_columns": _with_new_column,
}


@pytest.fixture(autouse=True)
def copy_on_write_on() -> Iterator[None]:
    """Turn copy-on-write on under pandas 2.x; pandas 3 always has it on."""

    if int(pd.__version__.split(".", 1)[0]) >= 3:
        yield
        return
    with pd.option_context("mode.copy_on_write", True):
        yield


def _write_through_to_numpy(column: pd.Series) -> None:
    column.to_numpy()[1] = column.to_numpy()[0]


def _write_category_through_values(column: pd.Series) -> None:
    column.values[1] = "north"


def _add_at_into_to_numpy(column: pd.Series) -> None:
    np.add.at(column.to_numpy(), [1], 1.0)


def _write_through_asarray_of_array(column: pd.Series) -> None:
    np.asarray(column.array)[1] = np.asarray(column.array)[0]


def _write_in_place_through_torch(column: pd.Series) -> None:
    torch = pytest.importorskip("torch")
    with warnings.catch_warnings():
        # torch warns that the array it wraps is read-only, which is the point.
        warnings.simplefilter("ignore", UserWarning)
        torch.from_numpy(column.to_numpy()).add_(1.0)


def _write_through_extension_array(column: pd.Series) -> None:
    column.array[1] = 99


def _write_string_through_to_numpy(column: pd.Series) -> None:
    column.to_numpy()[1] = "changed"


def _replace_list_through_asarray(column: pd.Series) -> None:
    np.asarray(column.array)[1] = ["changed"]


def _add_at_into_datetime_integers(column: pd.Series) -> None:
    np.add.at(column.array.asi8, [1], 1)


def _add_at_into_sparse_values(column: pd.Series) -> None:
    np.add.at(column.array.sp_values, [0], 1)


_RAW_WRITES: dict[str, tuple[str, Callable[[pd.Series], None]]] = {
    "to_numpy on Int64": ("count_complete", _write_through_to_numpy),
    "to_numpy on Float64": ("amount_complete", _write_through_to_numpy),
    "to_numpy on boolean": ("flag_complete", _write_through_to_numpy),
    "values on category": ("region", _write_category_through_values),
    "add.at on float64": ("income", _add_at_into_to_numpy),
    "asarray of array on float64": ("income", _write_through_asarray_of_array),
    "torch in place on float64": ("income", _write_in_place_through_torch),
    "array on Int64 with a gap": ("count_with_gap", _write_through_extension_array),
    "to_numpy on python strings": ("label_python", _write_string_through_to_numpy),
    "asarray of array on object": ("label_object", _write_through_asarray_of_array),
    "asarray of array on lists": ("tags", _replace_list_through_asarray),
    "add.at on time-zone dates": ("born_utc", _add_at_into_datetime_integers),
    "add.at on periods": ("born_month", _add_at_into_datetime_integers),
    "add.at on sparse values": ("rare_count", _add_at_into_sparse_values),
}


@pytest.mark.parametrize("raw_write", sorted(_RAW_WRITES))
def test_the_check_names_a_column_changed_by_a_raw_write(raw_write: str) -> None:
    column_name, write = _RAW_WRITES[raw_write]
    source = _source_frame()
    result = _reweighted(source)

    write(source.table("person")[column_name])

    for frame in (source, result):
        with pytest.raises(SharedColumnChangedError) as raised:
            frame.check_shared_columns()
        assert raised.value.changed_columns == (("person", column_name),)
        assert "person" in str(raised.value)
        assert column_name in str(raised.value)


@pytest.mark.parametrize("operation", sorted(_SHARING_OPERATIONS))
def test_every_sharing_operation_makes_its_columns_checked(operation: str) -> None:
    source = _source_frame()
    result = _SHARING_OPERATIONS[operation](source)
    result.check_shared_columns()

    _add_at_into_to_numpy(source.table("person")["income"])

    with pytest.raises(SharedColumnChangedError) as raised:
        result.check_shared_columns()
    assert raised.value.changed_columns == (("person", "income"),)


def test_pyarrow_strings_are_not_hashed() -> None:
    """Their buffers are immutable, and hashing text is the slowest case."""

    pytest.importorskip("pyarrow")
    column = pd.Series(["a", "b"], dtype=pd.StringDtype("pyarrow"))

    assert shared_columns_module._column_buffers(column) == ()


def test_a_group_table_kept_whole_by_a_partial_select_is_checked() -> None:
    source = _source_frame()
    # Dropping one of a household's two people keeps every household row.
    result = source.select(np.array([True, False, True, True]))

    _add_at_into_to_numpy(source.table("household")["rent"])

    with pytest.raises(SharedColumnChangedError) as raised:
        result.check_shared_columns()
    assert raised.value.changed_columns == (("household", "rent"),)


def test_a_raw_write_into_shared_strata_is_named() -> None:
    source = _source_frame()
    result = _reweighted(source)

    np.asarray(source.strata.array)[1] = "changed"

    with pytest.raises(SharedColumnChangedError) as raised:
        result.check_shared_columns()
    assert raised.value.changed_columns == (("strata", "strata"),)


def test_every_changed_column_is_named_at_once() -> None:
    source = _source_frame()
    result = _reweighted(source)

    _add_at_into_to_numpy(source.table("person")["income"])
    _add_at_into_to_numpy(source.table("household")["rent"])

    with pytest.raises(SharedColumnChangedError) as raised:
        result.check_shared_columns()
    assert raised.value.changed_columns == (
        ("person", "income"),
        ("household", "rent"),
    )


@pytest.mark.parametrize("operation", sorted(_SHARING_OPERATIONS))
def test_unchanged_frames_pass_the_check(operation: str) -> None:
    source = _source_frame()
    result = _SHARING_OPERATIONS[operation](source)

    source.check_shared_columns()
    result.check_shared_columns()


def _replace_whole_column(table: pd.DataFrame) -> None:
    table["income"] = table["income"].to_numpy() + 1.0


def _add_column(table: pd.DataFrame) -> None:
    table["added"] = 1.0


def _drop_column_in_place(table: pd.DataFrame) -> None:
    table.drop(columns=["income"], inplace=True)


def _write_one_cell(table: pd.DataFrame) -> None:
    table.loc[table.index[0], "income"] = -1.0


def _replace_nullable_column(table: pd.DataFrame) -> None:
    table["count_with_gap"] = pd.array([9, 9, None, 9], dtype="Int64")


def _replace_category_column(table: pd.DataFrame) -> None:
    table["region"] = pd.Categorical(["south"] * _PERSON_COUNT)


_PANDAS_WRITES: dict[str, Callable[[pd.DataFrame], None]] = {
    "replace a whole column": _replace_whole_column,
    "add a column": _add_column,
    "drop a column in place": _drop_column_in_place,
    "write one cell": _write_one_cell,
    "replace a nullable column": _replace_nullable_column,
    "replace a category column": _replace_category_column,
}


@pytest.mark.parametrize("pandas_write", sorted(_PANDAS_WRITES))
def test_writes_made_through_pandas_do_not_trip_the_check(pandas_write: str) -> None:
    source = _source_frame()
    result = _reweighted(source)

    _PANDAS_WRITES[pandas_write](result.table("person"))

    result.check_shared_columns()
    source.check_shared_columns()


def test_replacing_a_column_after_its_sharer_is_gone_passes_the_check() -> None:
    source = _source_frame()
    result = _reweighted(source)
    del source
    gc.collect()

    person = result.table("person")
    person["income"] = person["income"].to_numpy() + 1.0
    person["income"] = person["income"].to_numpy() + 1.0

    result.check_shared_columns()


def test_freed_memory_is_forgotten() -> None:
    before = shared_columns_module._tracked_owner_count()
    table = pd.DataFrame({"income": np.arange(1_000, dtype=float)})
    shared_columns_module.record_shared_columns({"person": table})
    assert shared_columns_module._tracked_owner_count() == before + 1

    del table
    gc.collect()

    assert shared_columns_module._tracked_owner_count() == before


def test_a_new_array_at_a_freed_address_is_not_compared() -> None:
    """An array allocated where a freed one lived is different memory."""

    size = 4_096
    table = pd.DataFrame({"income": np.zeros(size)})
    shared_columns_module.record_shared_columns({"person": table})
    address = table["income"].to_numpy().__array_interface__["data"][0]
    del table
    gc.collect()

    # The allocator usually hands the same block back; try until it does, so
    # the test exercises the reused address whenever the platform allows it.
    arrays = [np.ones(size) for _ in range(32)]
    reused = [
        array for array in arrays if array.__array_interface__["data"][0] == address
    ]
    replacement = reused[0] if reused else arrays[0]

    check_shared_columns({"person": pd.DataFrame({"income": replacement}, copy=False)})


def _counted_digests(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the length of every buffer the check hashes."""

    hashed_lengths: list[int] = []
    original = shared_columns_module._buffer_digest

    def counting_digest(buffer: np.ndarray) -> bytes:
        hashed_lengths.append(len(buffer))
        return original(buffer)

    monkeypatch.setattr(shared_columns_module, "_buffer_digest", counting_digest)
    return hashed_lengths


def test_a_column_is_hashed_once_however_many_frames_share_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_frame()
    hashed_lengths = _counted_digests(monkeypatch)

    first = _reweighted(source)
    hashed_for_first_frame = len(hashed_lengths)
    assert hashed_for_first_frame > 0

    second = _reweighted(first)
    third = second.replace_tables()

    assert len(hashed_lengths) == hashed_for_first_frame
    assert third.n("person") == _PERSON_COUNT


def test_frames_that_share_nothing_are_not_hashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashed_lengths = _counted_digests(monkeypatch)

    source = _source_frame()
    # A selection dropping a household copies every row it keeps.
    selected = source.select(np.array([True, True, False, False]))
    source.check_shared_columns()
    selected.check_shared_columns()

    assert hashed_lengths == []


def test_nothing_is_recorded_when_copy_on_write_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bundle_module, "_pandas_copy_on_write", lambda: False)
    hashed_lengths = _counted_digests(monkeypatch)

    result = _reweighted(_source_frame())
    result.check_shared_columns()

    assert hashed_lengths == []


class _RecordingStore:
    """Stands in for an HDF store and records what reaches it."""

    def __init__(self) -> None:
        self.written_keys: list[str] = []

    def put(self, key: str, *_arguments: object, **_options: object) -> None:
        self.written_keys.append(key)


def test_the_table_writer_refuses_a_changed_shared_column() -> None:
    source = _source_frame()
    result = _reweighted(source)
    _add_at_into_to_numpy(source.table("person")["income"])
    store = _RecordingStore()

    with pytest.raises(SharedColumnChangedError) as raised:
        put_frame_table(
            store, "person", result.table("person"), preferred_format="fixed"
        )

    assert raised.value.changed_columns == (("person", "income"),)
    assert store.written_keys == []


def test_the_table_writer_accepts_unchanged_shared_columns() -> None:
    source = _source_frame()
    result = _reweighted(source)
    store = _RecordingStore()

    put_frame_table(
        store, "household", result.table("household"), preferred_format="fixed"
    )

    assert store.written_keys == ["household"]


def test_engine_tables_refuses_a_frame_with_a_changed_shared_column() -> None:
    """``engine_tables`` deep-copies, so a later check could not see the change."""

    source = _source_frame()
    result = _reweighted(source)
    _add_at_into_to_numpy(source.table("person")["income"])

    with pytest.raises(SharedColumnChangedError):
        engine_tables(result)
