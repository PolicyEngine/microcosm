"""Invented, source-independent tests of the directional cache replay boundary."""

from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.survey_population_replay import (
    SEAL_PROTOCOL,
    _axis,
    _axis_seal,
    _same_axis_seal,
    replayed_frame_seal,
    replayed_population_seal,
    same_replayed_frame,
    same_replayed_frame_seals,
    same_replayed_population,
    same_replayed_population_seals,
    seal_identity,
)
from microcosm.frame import EntitySchema, Frame, MassChangeRecord, WeightKind, Weights
from microcosm.graph.population import MassRecord, Population
from microcosm.graph.store import ContentStore

_ERROR = "^SURVEY_POPULATION_REPLAY_"


def _fields(record):
    return {field.name: getattr(record, field.name) for field in fields(record)}


def _verdict(call, *arguments):
    """The refusal code, or ``None`` for acceptance."""
    try:
        call(*arguments)
    except ValueError as error:
        return str(error)
    return None


def _sealed_verdict(seal, compare, expected, actual):
    """The seal path's verdict: a refusal from either constructor counts.

    Every predicate ``same_replayed_*`` makes about one operand alone is
    asserted when that operand's seal is built, so the battery below must
    treat a construction refusal and a comparison refusal as the same event.
    """
    try:
        expected_seal, actual_seal = seal(expected), seal(actual)
    except ValueError as error:
        return str(error)
    return _verdict(compare, expected_seal, actual_seal)


# Optional receipt: with MICROCOSM_BATTERY_RECEIPT set to a path, every
# agreement below appends one row naming the test, the object path's verdict
# and the seal path's. It is off by default, writes nothing when unset, and
# changes no assertion -- see experiments/native-retention-seal/battery-receipt.json.
_RECEIPT_PATH = os.environ.get("MICROCOSM_BATTERY_RECEIPT")


def _record(kind, direct, sealed):
    if not _RECEIPT_PATH:
        return
    with open(_RECEIPT_PATH, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "test": os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0],
                    "operand": kind,
                    "object_path": direct,
                    "seal_path": sealed,
                }
            )
            + "\n"
        )


def _agree_frames(expected, actual):
    """Both paths must reach the same verdict, with the same refusal code.

    This is the discrimination battery of ``docs/us-native-retention-seal.md``
    section 5: every mutation below is driven through the object comparison
    that exists today and through the content seal that replaces it, and a
    mutation the seal misses fails here rather than being quietly dropped.
    """
    direct = _verdict(same_replayed_frame, expected, actual)
    sealed = _sealed_verdict(
        replayed_frame_seal, same_replayed_frame_seals, expected, actual
    )
    _record("frame", direct, sealed)
    assert direct == sealed, f"object path {direct!r}, seal path {sealed!r}"
    return direct


def _agree_populations(expected, actual):
    direct = _verdict(same_replayed_population, expected, actual)
    sealed = _sealed_verdict(
        replayed_population_seal, same_replayed_population_seals, expected, actual
    )
    _record("population", direct, sealed)
    assert direct == sealed, f"object path {direct!r}, seal path {sealed!r}"
    return direct


def _refuses_frames(expected, actual):
    code = _agree_frames(expected, actual)
    assert code is not None and code.startswith("SURVEY_POPULATION_REPLAY_"), code
    return code


def _refuses_populations(expected, actual):
    code = _agree_populations(expected, actual)
    assert code is not None and code.startswith("SURVEY_POPULATION_REPLAY_"), code
    return code


def _accepts_frames(expected, actual):
    assert _agree_frames(expected, actual) is None


def _accepts_populations(expected, actual):
    assert _agree_populations(expected, actual) is None


def _frame():
    index = pd.Index([101, 103, 107], dtype="int64", name="source_row")
    person = pd.DataFrame(
        {
            "person_id": np.array([1, 2, 3], dtype=np.int64),
            "person_household_id": np.array([10, 10, 20], dtype=np.int64),
            "person_tax_unit_id": np.array([30, 30, 40], dtype=np.int64),
            "nullable_integer": pd.arrays.IntegerArray(
                np.array([11, 71, 73], dtype=np.int64),
                np.array([False, True, True]),
            ),
            "nullable_boolean": pd.arrays.BooleanArray(
                np.array([True, True, True]), np.array([False, True, True])
            ),
            "native_float": np.array(
                [0x8000000000000000, 0x7FF8000000000011, 0x3FF4000000000000],
                dtype=np.uint64,
            ).view(np.float64),
            "text": pd.array(["", "invented\x00label", pd.NA], dtype="string"),
            "object_value": pd.Series(
                ["invented object label", b"invented bytes", None],
                index=index,
                dtype=object,
            ),
        },
        index=index,
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": [10, 20]}),
            "tax_unit": pd.DataFrame({"tax_unit_id": [30, 40]}),
        },
        EntitySchema(group_entities=("household", "tax_unit")),
        {"household": Weights(np.array([1.5, 2.5]), WeightKind.IMPORTANCE)},
        pd.Series(["urban", "urban", "rural"], index=index, dtype=object),
        mass_log=(MassChangeRecord("household", 2.0, 4.0, 2.0, "invented"),),
        metadata={"source": {"arm": "invented", "members": (1, 2)}, "zero": -0.0},
    )


def _rebuild(frame, **changes):
    arguments = {
        "tables": {entity: frame.table(entity) for entity in frame.entities},
        "schema": frame.schema,
        "weights": {
            entity: frame.weights_for(entity) for entity in frame.weighted_entities
        },
        "strata": frame.strata,
        "mass_log": frame.mass_log,
        "metadata": frame.metadata,
    }
    arguments.update(changes)
    return Frame(**arguments)


def _with_column(frame, name, values):
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][name] = values
    return _rebuild(frame, tables=tables)


def _population(frame):
    return Population.from_frame(
        frame,
        "allocated",
        mass_ledger=(
            MassRecord(
                "allocation",
                "reweight",
                "declared",
                12.0,
                4.0,
                (("invented", 12.0),),
                (("invented", 4.0),),
                entity="household",
            ),
        ),
        design_weights={"household": np.array([5.0, 7.0])},
    )


def test_actual_store_roundtrip_accepts_canonical_nulls_and_typed_objects(tmp_path):
    expected = _frame()
    before_integer = expected.person["nullable_integer"].array._data.tobytes()
    before_boolean = expected.person["nullable_boolean"].array._data.tobytes()
    store = ContentStore(tmp_path / "invented-store")
    store.put_frame("a" * 64, expected)
    actual = store.load_frame("a" * 64)

    assert expected.person["nullable_integer"].array._data[1:].tolist() == [71, 73]
    assert actual.person["nullable_integer"].array._data[1:].tolist() == [0, 0]
    assert actual.person["nullable_boolean"].array._data[1:].tolist() == [False, False]
    assert actual.person["native_float"].to_numpy().tobytes() == (
        expected.person["native_float"].to_numpy().tobytes()
    )
    assert (
        actual.person["object_value"].iloc[0] == expected.person["object_value"].iloc[0]
    )
    assert (
        actual.person["object_value"].iloc[0]
        is not expected.person["object_value"].iloc[0]
    )
    _accepts_frames(expected, actual)
    original_population = _population(expected)
    _accepts_populations(
        original_population, replace(original_population, frame=actual)
    )
    # Acceptance is read-only and must not canonicalize the original in place.
    assert expected.person["nullable_integer"].array._data.tobytes() == before_integer
    assert expected.person["nullable_boolean"].array._data.tobytes() == before_boolean


def test_unchanged_noncanonical_hidden_storage_is_accepted():
    _accepts_frames(_frame(), _frame())


@pytest.mark.parametrize("column", ["nullable_integer", "nullable_boolean"])
@pytest.mark.parametrize("change", ["partial_canonical", "noncanonical", "reverse"])
def test_hidden_storage_exception_is_directional_and_whole_column(column, change):
    expected, actual = _frame(), _frame()
    if change == "reverse":
        expected.person[column].array._data[1:] = 0
    elif change == "partial_canonical":
        actual.person[column].array._data[1] = 0
    elif column == "nullable_integer":
        actual.person[column].array._data[1:] = [81, 83]
    else:
        expected.person[column].array._data[1:] = [False, True]
        actual.person[column].array._data[1:] = [True, False]
    _refuses_frames(expected, actual)


@pytest.mark.parametrize("change", ["value", "mask", "dtype", "membership", "text"])
def test_observed_values_masks_and_dtypes_are_exact(change):
    expected = _frame()
    actual = _frame()
    if change == "value":
        actual.person["nullable_integer"].array._data[0] += 1
    elif change == "mask":
        actual.person["nullable_integer"].array._mask[1] = False
    elif change == "dtype":
        actual = _with_column(
            actual,
            "nullable_integer",
            actual.person["nullable_integer"].astype("Int32"),
        )
    elif change == "membership":
        actual = _with_column(
            actual, "person_household_id", np.array([10, 20, 20], dtype=np.int64)
        )
    else:
        actual = _with_column(
            actual,
            "text",
            pd.array(["changed", "invented\x00label", pd.NA], dtype="string"),
        )
    _refuses_frames(expected, actual)


@pytest.mark.parametrize(
    "change",
    [
        "rows",
        "columns",
        "index_name",
        "index_dtype",
        "column_axis_name",
        "strata",
        "schema_order",
    ],
)
def test_axes_schema_and_strata_are_exact(change):
    expected = _frame()
    tables = {entity: expected.table(entity).copy() for entity in expected.entities}
    arguments = {"tables": tables}
    if change == "rows":
        tables["person"] = tables["person"].iloc[[2, 1, 0]]
        arguments["strata"] = expected.strata.iloc[[2, 1, 0]]
    elif change == "columns":
        tables["person"] = tables["person"].loc[
            :, list(reversed(tables["person"].columns))
        ]
    elif change == "index_name":
        tables["household"].index = tables["household"].index.rename("other_axis")
    elif change == "index_dtype":
        tables["household"].index = pd.Index([0, 1], dtype="int32")
    elif change == "column_axis_name":
        tables["household"].columns = tables["household"].columns.rename(
            "other_columns"
        )
    elif change == "strata":
        arguments["strata"] = pd.Series(
            ["rural", "urban", "rural"], index=expected.person.index, dtype=object
        )
    else:
        arguments["schema"] = EntitySchema(group_entities=("tax_unit", "household"))
    actual = _rebuild(expected, **arguments)
    _refuses_frames(expected, actual)


@pytest.mark.parametrize(
    "change",
    [
        "nested_metadata",
        "metadata_scalar_type",
        "metadata_signed_zero",
        "mass_log",
        "weight_bits",
        "weight_signed_zero",
        "weight_kind",
    ],
)
def test_frame_context_and_exact_weight_contract_are_preserved(change):
    expected = _frame()
    if change == "nested_metadata":
        actual = _rebuild(
            expected,
            metadata={"source": {"arm": "changed", "members": (1, 2)}, "zero": -0.0},
        )
    elif change == "metadata_scalar_type":
        expected = _rebuild(expected, metadata={"flag": True})
        actual = _rebuild(expected, metadata={"flag": 1})
    elif change == "metadata_signed_zero":
        actual = _rebuild(
            expected,
            metadata={"source": {"arm": "invented", "members": (1, 2)}, "zero": 0.0},
        )
    elif change == "mass_log":
        actual = _rebuild(
            expected, mass_log=(replace(expected.mass_log[0], reason="changed"),)
        )
    elif change == "weight_signed_zero":
        expected = _rebuild(
            expected,
            weights={
                "household": Weights(np.array([-0.0, 2.5]), WeightKind.IMPORTANCE)
            },
        )
        actual = _rebuild(
            expected,
            weights={"household": Weights(np.array([0.0, 2.5]), WeightKind.IMPORTANCE)},
        )
    else:
        values = expected.weights_for("household").values.copy()
        kind = WeightKind.IMPORTANCE
        if change == "weight_bits":
            values[0] = np.nextafter(values[0], np.inf)
        else:
            kind = WeightKind.CALIBRATED
        actual = _rebuild(expected, weights={"household": Weights(values, kind)})
    _refuses_frames(expected, actual)


@pytest.mark.parametrize(
    "dtype,bits",
    [
        ("float64", (0x8000000000000000, 0x7FF8000000000011, 0x3FF4000000000000)),
        ("float32", (0x80000000, 0x7FC00011, 0x3FA00000)),
    ],
)
@pytest.mark.parametrize("slot", [0, 1])
def test_native_signed_zero_and_nan_payload_bits_cannot_change(dtype, bits, slot):
    unsigned = "uint64" if dtype == "float64" else "uint32"
    values = np.array(bits, dtype=unsigned)
    expected = _with_column(_frame(), "native_float", values.view(dtype))
    changed = values.copy()
    changed[slot] = 0 if slot == 0 else int(changed[slot]) + 1
    actual = _with_column(expected, "native_float", changed.view(dtype))
    _refuses_frames(expected, actual)


@pytest.mark.parametrize(
    "before,after",
    [
        (np.int64(7), 7),
        (np.bool_(True), True),
        (np.float32(1.25), 1.25),
        (np.bytes_(b"x"), b"x"),
    ],
)
def test_object_scalar_equivalence_uses_actual_store_types(before, after):
    frame = _frame()
    expected = _with_column(
        frame,
        "object_value",
        pd.Series([before, None, None], index=frame.person.index, dtype=object),
    )
    actual = _with_column(
        expected,
        "object_value",
        pd.Series([after, None, None], index=expected.person.index, dtype=object),
    )
    _accepts_frames(expected, actual)


@pytest.mark.parametrize(
    "before,after",
    [(True, 1), (1, 1.0), (None, pd.NA), (pd.NaT, None), ("x", b"x"), (-0.0, 0.0)],
)
def test_object_scalar_types_null_sentinels_and_float_bits_cannot_change(before, after):
    frame = _frame()
    expected = _with_column(
        frame,
        "object_value",
        pd.Series([before, None, None], index=frame.person.index, dtype=object),
    )
    actual = _with_column(
        frame,
        "object_value",
        pd.Series([after, None, None], index=frame.person.index, dtype=object),
    )
    _refuses_frames(expected, actual)


def test_object_nan_payload_bits_cannot_change():
    frame = _frame()
    payloads = np.array([0x7FF8000000000011, 0x7FF8000000000012], dtype=np.uint64).view(
        np.float64
    )
    expected = _with_column(
        frame,
        "object_value",
        pd.Series([payloads[0], None, None], index=frame.person.index, dtype=object),
    )
    actual = _with_column(
        frame,
        "object_value",
        pd.Series([payloads[1], None, None], index=frame.person.index, dtype=object),
    )
    _refuses_frames(expected, actual)


@pytest.mark.parametrize("kind", ["object", "category", "nullable_float"])
def test_equal_unsupported_store_values_still_refuse(kind, tmp_path):
    frame = _frame()
    if kind == "object":
        values = pd.Series(
            [{"unsupported": 1}, None, None], index=frame.person.index, dtype=object
        )
    elif kind == "category":
        values = pd.Categorical(["a", "b", "a"])
    else:
        values = pd.array([1.0, pd.NA, 2.0], dtype="Float64")
    unsupported = _with_column(frame, "object_value", values)
    with pytest.raises(TypeError):
        ContentStore(tmp_path / "unsupported-store").put_frame("b" * 64, unsupported)
    _refuses_frames(unsupported, unsupported)


@pytest.mark.parametrize("axis", ["index", "columns"])
@pytest.mark.parametrize("before,after", [(True, 1), (-0.0, 0.0), ((True,), (1,))])
def test_axis_names_preserve_store_scalar_types(axis, before, after):
    expected, actual = _frame(), _frame()
    getattr(expected.table("household"), axis).name = before
    getattr(actual.table("household"), axis).name = after
    # Pandas' axis equality alone treats each pair as equal.
    assert getattr(expected.table("household"), axis).identical(
        getattr(actual.table("household"), axis)
    )
    _refuses_frames(expected, actual)


def test_multiindex_outside_reviewed_store_profile_refuses():
    frame = _frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"].index = pd.MultiIndex.from_tuples([("a", 0), ("b", 1)])
    unsupported = _rebuild(frame, tables=tables)
    _refuses_frames(unsupported, unsupported)


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "owner",
        "design_value",
        "design_signed_zero",
        "missing_design",
        "ledger",
        "weight_kind",
    ],
)
def test_population_provenance_and_original_design_bytes_are_exact(change):
    expected = _population(_frame())
    if change == "version":
        actual = replace(expected, version="another_version")
    elif change == "owner":
        owners = dict(expected.owners)
        owners[("person", "nullable_integer")] = "another_owner"
        actual = replace(expected, owners=owners)
    elif change == "design_value":
        actual = replace(
            expected,
            design_weights={"household": np.array([5.0, np.nextafter(7.0, np.inf)])},
        )
    elif change == "design_signed_zero":
        expected = replace(
            expected, design_weights={"household": np.array([-0.0, 7.0])}
        )
        actual = replace(expected, design_weights={"household": np.array([0.0, 7.0])})
    elif change == "missing_design":
        actual = replace(expected, design_weights={})
    elif change == "ledger":
        actual = replace(
            expected,
            mass_ledger=(
                replace(expected.mass_ledger[0], node_id="another_allocation"),
            ),
        )
    else:
        frame = _rebuild(
            expected.frame,
            weights={"household": Weights(np.array([1.5, 2.5]), WeightKind.CALIBRATED)},
        )
        actual = replace(
            expected, frame=frame, weight_kind={"household": WeightKind.CALIBRATED}
        )
    _refuses_populations(expected, actual)


# --- Discriminations the battery above did not reach ---
#
# Every test below drives one property through ``_refuses_*``/``_accepts_*``,
# so each one proves the object comparison and the content seal agree on it.
# The three marked "seal gap" are discriminations that
# ``survey_population_preparation._frame_identity`` does NOT make, which is why
# the seal is purpose-built rather than reusing it
# (``docs/us-native-retention-seal.md`` section 0).


def test_table_flags_are_part_of_the_comparison():  # seal gap
    expected, actual = _frame(), _frame()
    actual.person.flags.allows_duplicate_labels = False
    assert (
        _refuses_frames(expected, actual)
        == "SURVEY_POPULATION_REPLAY_TABLE_TYPE_OR_FLAGS"
    )


def test_every_documented_flag_is_folded_not_only_the_named_one():
    """The seal reads ``type(flags)._keys`` so a new pandas flag is covered."""
    keys = type(_frame().person.flags)._keys
    assert type(keys) is set and keys and all(type(key) is str for key in keys)
    sealed = replayed_frame_seal(_frame())
    person = next(entity for entity in sealed[6] if entity[0] == "person")
    assert tuple(key for key, _ in person[1]) == tuple(sorted(keys))


def test_the_exact_index_class_is_part_of_the_comparison():
    """A ``RangeIndex`` and an ``Index`` of the same values are not identical."""
    expected, actual = _frame(), _frame()
    table = actual.table("household")
    assert type(table.index) is pd.RangeIndex
    table.index = pd.Index(list(table.index), dtype=table.index.dtype)
    assert type(table.index) is pd.Index
    assert table.index.equals(expected.table("household").index)
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_AXIS"


def test_the_columns_axis_dtype_is_part_of_the_comparison():
    expected, actual = _frame(), _frame()
    table = actual.table("household")
    table.columns = pd.Index(list(table.columns), dtype=object)
    assert table.columns.dtype != expected.table("household").columns.dtype
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_AXIS"


def test_an_axis_byte_difference_equals_does_not_see_keeps_the_series_code():
    """``identical`` accepts ``-0.0`` for ``0.0``; the byte walk still refuses."""
    expected, actual = _frame(), _frame()
    for frame, first in ((expected, -0.0), (actual, 0.0)):
        table = frame.table("household")
        table.index = pd.Index([first, 1.0], dtype="float64")
    assert expected.table("household").index.identical(actual.table("household").index)
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_NATIVE_BITS"


def test_an_axis_nan_payload_difference_keeps_the_series_code():
    payloads = np.array([0x7FF8000000000011, 0x7FF8000000000012], dtype=np.uint64).view(
        np.float64
    )
    expected, actual = _frame(), _frame()
    for frame, value in ((expected, payloads[0]), (actual, payloads[1])):
        frame.table("household").index = pd.Index([value, 1.0], dtype="float64")
    assert expected.table("household").index.identical(actual.table("household").index)
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_NATIVE_BITS"


def test_index_comparables_beyond_the_name_are_part_of_the_comparison():
    """``DatetimeIndex._comparables`` carries ``freq``; the bytes do not."""
    stamps = ["2020-01-01", "2020-01-02"]
    with_freq = pd.DatetimeIndex(stamps, freq="D")
    without = pd.DatetimeIndex(stamps)
    assert type(with_freq)._comparables == ["name", "freq"]
    assert with_freq.to_numpy().tobytes() == without.to_numpy().tobytes()
    assert not with_freq.identical(without)
    expected, actual = _frame(), _frame()
    expected.table("household").index = with_freq
    actual.table("household").index = without
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_AXIS"


def test_native_nan_payload_bits_are_part_of_the_comparison():  # seal gap
    payloads = (0x7FF8000000000011, 0x7FF8000000000012)
    frames = [
        _with_column(
            _frame(),
            "native_float",
            np.array(
                [bits, 0x3FF4000000000000, 0x8000000000000000], dtype=np.uint64
            ).view(np.float64),
        )
        for bits in payloads
    ]
    assert _refuses_frames(*frames) == "SURVEY_POPULATION_REPLAY_NATIVE_BITS"


def test_quiet_and_signalling_nan_are_distinguished():  # seal gap
    frames = [
        _with_column(
            _frame(),
            "native_float",
            np.array(
                [bits, 0x3FF4000000000000, 0x8000000000000000], dtype=np.uint64
            ).view(np.float64),
        )
        for bits in (0x7FF8000000000000, 0x7FF0000000000001)
    ]
    assert _refuses_frames(*frames) == "SURVEY_POPULATION_REPLAY_NATIVE_BITS"


def test_a_native_dtype_change_at_equal_values_still_refuses():
    expected = _with_column(_frame(), "native_int", np.array([1, 2, 3], dtype=np.int64))
    actual = _with_column(expected, "native_int", np.array([1, 2, 3], dtype=np.int32))
    assert (
        _refuses_frames(expected, actual)
        == "SURVEY_POPULATION_REPLAY_SERIES_DTYPE_OR_LENGTH"
    )


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_string_storage_and_null_policy_are_part_of_the_comparison(storage):
    other = "pyarrow" if storage == "python" else "python"
    values = ["a", "b", None]
    expected = _with_column(
        _frame(), "text", pd.array(values, dtype=pd.StringDtype(storage))
    )
    actual = _with_column(
        _frame(), "text", pd.array(values, dtype=pd.StringDtype(other))
    )
    assert (
        _refuses_frames(expected, actual)
        == "SURVEY_POPULATION_REPLAY_SERIES_DTYPE_OR_LENGTH"
    )


def test_a_declared_link_table_refuses_on_both_paths():
    from microcosm.frame import LinkSpec

    frame = _frame()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["relations"] = pd.DataFrame(
        {"person_id": [1, 2, 3], "household_id": [10, 10, 20]}
    )
    linked = _rebuild(
        frame,
        tables=tables,
        schema=EntitySchema(
            group_entities=("household", "tax_unit"),
            links=(LinkSpec("relations", "person", "household"),),
        ),
    )
    assert linked.links == ("relations",)
    assert _refuses_frames(linked, linked) == "SURVEY_POPULATION_REPLAY_FRAME_CONTEXT"


def test_strata_index_name_is_part_of_the_comparison():
    expected, actual = _frame(), _frame()
    strata = actual.strata.copy()
    strata.index = strata.index.rename("other_strata_axis")
    assert _refuses_frames(expected, _rebuild(actual, strata=strata)) == (
        "SURVEY_POPULATION_REPLAY_AXIS"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("owners", {("person", "person_id"): 7}),
        ("weight_kind", {"household": "importance"}),
        ("mass_ledger", [MassRecord("a", "reweight", "declared", 1.0, 1.0, (), ())]),
        ("mass_ledger", ({"node_id": "allocation"},)),
        ("version", 7),
    ],
)
def test_one_sided_population_type_assertions_refuse_on_both_paths(field, value):
    """These fire when the seal is built rather than when it is compared.

    ``Population`` validates its own construction, so each defect is written
    past that validation the way ``test_graph_executor`` writes its own
    (``object.__setattr__`` on the frozen record), which is the only way a
    caller could actually present one.
    """
    expected = _population(_frame())
    actual = _population(_frame())
    object.__setattr__(actual, field, value)
    assert (
        _refuses_populations(expected, actual)
        == "SURVEY_POPULATION_REPLAY_POPULATION_CONTEXT"
    )


@pytest.mark.parametrize("field", ["weight_kind", "design_weights"])
def test_mapping_order_is_part_of_the_comparison(field):
    frame = _rebuild(
        _frame(),
        weights={
            "household": Weights(np.array([1.5, 2.5]), WeightKind.IMPORTANCE),
            "tax_unit": Weights(np.array([3.5, 4.5]), WeightKind.IMPORTANCE),
        },
    )
    expected = Population.from_frame(
        frame,
        "allocated",
        design_weights={
            "household": np.array([5.0, 7.0]),
            "tax_unit": np.array([9.0, 11.0]),
        },
    )
    reversed_mapping = dict(reversed(list(getattr(expected, field).items())))
    actual = replace(expected, **{field: reversed_mapping})
    assert tuple(getattr(actual, field)) != tuple(getattr(expected, field))
    assert (
        _refuses_populations(expected, actual)
        == "SURVEY_POPULATION_REPLAY_POPULATION_CONTEXT"
    )


def test_owner_mapping_order_alone_is_accepted_by_both_paths():
    """``owners`` is compared sorted, so its insertion order must not matter."""
    expected = _population(_frame())
    actual = replace(expected, owners=dict(reversed(list(expected.owners.items()))))
    assert tuple(actual.owners) != tuple(expected.owners)
    _accepts_populations(expected, actual)


_DTYPE_CENSUS = (
    "int8",
    "int64",
    "uint64",
    "float32",
    "float64",
    "bool",
    "object",
    "<U5",
    "datetime64[ns]",
    "Int8",
    "Int64",
    "UInt64",
    "boolean",
)


def test_the_dtype_token_is_faithful_to_the_predicate_it_replaces():
    """Token equality must be exactly ``type(a) is type(b) and a == b``.

    ``str(dtype)`` alone is not enough: ``StringDtype("python")`` and
    ``StringDtype("pyarrow")`` both spell ``string``.
    """
    dtypes = [pd.Series([], dtype=name).dtype for name in _DTYPE_CENSUS]
    dtypes += [pd.StringDtype("python"), pd.StringDtype("pyarrow")]
    dtypes += [pd.StringDtype("python", na_value=np.nan)]
    for left in dtypes:
        for right in dtypes:
            predicate = type(left) is type(right) and left == right
            token = (type(left), left) == (type(right), right)
            assert predicate == token, (left, right, predicate, token)


def test_the_seal_is_proportional_to_columns_and_not_to_rows():
    """The whole point: sealing nineteen nodes must not cost nineteen frames."""
    import pickle

    def sized(rows):
        index = pd.RangeIndex(rows)
        person = pd.DataFrame(
            {
                "person_id": np.arange(rows, dtype=np.int64),
                "person_household_id": np.zeros(rows, dtype=np.int64),
                "person_tax_unit_id": np.zeros(rows, dtype=np.int64),
                "native_float": np.arange(rows, dtype=np.float64),
            },
            index=index,
        )
        return Frame(
            {
                "person": person,
                "household": pd.DataFrame({"household_id": [0]}),
                "tax_unit": pd.DataFrame({"tax_unit_id": [0]}),
            },
            EntitySchema(group_entities=("household", "tax_unit")),
            {"household": Weights(np.array([1.0]), WeightKind.IMPORTANCE)},
            pd.Series(["urban"] * rows, index=index, dtype=object),
        )

    sizes = [
        len(pickle.dumps(replayed_frame_seal(sized(n)))) for n in (8, 8_000, 800_000)
    ]
    # A 100,000-fold growth in rows may move the record only by the width of
    # the row counts and shapes it records, which is tens of bytes.
    assert max(sizes) - min(sizes) < 128, sizes


def test_seal_identity_moves_with_the_seal_and_is_stable_without_it():
    expected = _population(_frame())
    assert seal_identity(replayed_population_seal(expected)) == seal_identity(
        replayed_population_seal(_population(_frame()))
    )
    actual = replace(expected, version="another_version")
    assert seal_identity(replayed_population_seal(expected)) != seal_identity(
        replayed_population_seal(actual)
    )


# --- Refusals where BOTH operands are equal ---
#
# These are the hardest half of the seal's obligation and the easiest to miss:
# the comparison refuses a pair that is byte-identical, because it is asserting
# something about ONE operand. A seal cannot carry them as content, so it must
# assert them when it is built -- which is what ``_sealed_verdict`` above lets
# this battery check. Each case below was found by probing the comparison
# rather than by reading it.


def test_a_mass_change_record_subclass_refuses_though_the_values_match():
    class _Subclass(MassChangeRecord):
        pass

    frame = _frame()
    record = frame.mass_log[0]
    substituted = _rebuild(frame, mass_log=(_Subclass(**_fields(record)),))
    assert (
        _refuses_frames(substituted, substituted)
        == "SURVEY_POPULATION_REPLAY_FRAME_CONTEXT"
    )


def test_a_mass_record_subclass_refuses_though_the_values_match():
    class _Subclass(MassRecord):
        pass

    substituted = _population(_frame())
    record = substituted.mass_ledger[0]
    object.__setattr__(substituted, "mass_ledger", (_Subclass(**_fields(record)),))
    assert (
        _refuses_populations(substituted, substituted)
        == "SURVEY_POPULATION_REPLAY_POPULATION_CONTEXT"
    )


def test_a_population_subclass_refuses_though_the_content_matches():
    class _Subclass(Population):
        pass

    expected = _population(_frame())
    actual = _Subclass(
        expected.frame,
        expected.version,
        dict(expected.owners),
        dict(expected.weight_kind),
        mass_ledger=expected.mass_ledger,
        design_weights=dict(expected.design_weights),
    )
    assert (
        _refuses_populations(expected, actual)
        == "SURVEY_POPULATION_REPLAY_POPULATION_TYPE"
    )


@pytest.mark.parametrize("carrier", ["pyarrow", "sparse"])
def test_a_non_pandas_masked_carrier_refuses_though_the_bytes_match(carrier):
    """Only pandas' own ``_data``/``_mask`` storage is a reviewed profile."""
    if carrier == "pyarrow":
        pytest.importorskip("pyarrow")
        values = pd.array([1, 2, 3], dtype="int64[pyarrow]")
    else:
        values = pd.arrays.SparseArray(np.array([1, 2, 3], dtype=np.int64))
    frame = _frame()
    unsupported = _with_column(frame, "object_value", values)
    code = _refuses_frames(unsupported, unsupported)
    assert code in (
        "SURVEY_POPULATION_REPLAY_MASKED_STORAGE",
        "SURVEY_POPULATION_REPLAY_UNSUPPORTED_EXTENSION_DTYPE",
    ), code


def test_a_degenerate_mask_dtype_refuses_though_both_sides_carry_it():
    frame = _frame()
    array = frame.person["nullable_integer"].array
    object.__setattr__(array, "_mask", array._mask.astype(np.uint8))
    assert _refuses_frames(frame, frame) == "SURVEY_POPULATION_REPLAY_MASKED_STORAGE"


def test_an_ndarray_subclass_backing_refuses_though_both_sides_carry_it():
    class _Subclass(np.ndarray):
        pass

    frame = _frame()
    array = frame.person["nullable_integer"].array
    object.__setattr__(array, "_data", array._data.view(_Subclass))
    assert _refuses_frames(frame, frame) == "SURVEY_POPULATION_REPLAY_MASKED_STORAGE"


def test_a_str_subclass_cell_refuses_though_the_characters_match():
    """``np.str_`` arrives for free from a numpy array and is not ``str``."""
    frame = _frame()
    values = pd.array(["alpha", "beta", pd.NA], dtype=pd.StringDtype("python"))
    substituted = _with_column(frame, "text", values)
    backing = substituted.person["text"].array._ndarray
    backing[0] = np.array(["alpha"])[0]
    assert type(backing[0]) is not str and backing[0] == "alpha"
    assert (
        _refuses_frames(substituted, substituted)
        == "SURVEY_POPULATION_REPLAY_STRING_VALUE"
    )


def test_metadata_key_order_is_part_of_the_comparison():
    expected = _rebuild(_frame(), metadata={"first": 1, "second": 2})
    actual = _rebuild(_frame(), metadata={"second": 2, "first": 1})
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_FRAME_CONTEXT"


def test_object_nan_sign_is_part_of_the_comparison():
    bits = np.array([0x7FF8000000000000, 0xFFF8000000000000], dtype=np.uint64).view(
        np.float64
    )
    frame = _frame()
    frames = [
        _with_column(
            frame,
            "object_value",
            pd.Series([value, None, None], index=frame.person.index, dtype=object),
        )
        for value in bits
    ]
    assert _refuses_frames(*frames) == "SURVEY_POPULATION_REPLAY_OBJECT_VALUE"


def test_non_finite_metadata_keeps_its_sign_on_both_paths():
    """``_encode_frame_metadata`` spells a float64 in hex, infinities included.

    The seal folds the store codec's bytes, not ``graph_context``'s normative
    JSON, which is the reason it can carry a value canonical JSON refuses.
    """
    expected = _rebuild(_frame(), metadata={"value": float("inf")})
    _accepts_frames(expected, _rebuild(_frame(), metadata={"value": float("inf")}))
    actual = _rebuild(_frame(), metadata={"value": float("-inf")})
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_FRAME_CONTEXT"


# --- What an adversarial pass over the seal found ---
#
# Every case below broke the seal's first draft and is pinned here. The first
# is the dangerous direction -- the seal ACCEPTED a pair the comparison
# refuses -- and the rest each refused under a code the comparison does not
# use for that defect.


def _with_person_axis(frame, index):
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].set_axis(index, axis=0)
    strata = frame.strata.copy()
    strata.index = index
    return _rebuild(frame, tables=tables, strata=strata)


def test_an_object_axis_keeps_its_null_sentinel_identity():
    """pandas 3 re-infers ``pd.Series(index.array)`` and loses this.

    ``pd.Index(["r", "s", None], dtype=object)`` and the same index holding
    ``pd.NA`` both materialise as a ``str`` Series of ``["r", "s", nan]``, so a
    fold taken through a Series accepts a pair ``Index.identical`` refuses. The
    seal folds ``np.asarray(index.array)`` instead.
    """
    frame = _frame()
    expected = _with_person_axis(frame, pd.Index(["r", "s", None], dtype=object))
    actual = _with_person_axis(frame, pd.Index(["r", "s", pd.NA], dtype=object))
    assert list(expected.person.index) != list(actual.person.index)
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_AXIS"


@pytest.mark.parametrize("kind", ["bool", "complex", "float"])
def test_an_axis_byte_difference_equals_tolerates_keeps_the_series_code(kind):
    """``array_equivalent`` is tolerant for kinds ``f``, ``c`` and ``b``.

    ``Index.identical`` therefore passes and the byte walk refuses, so the
    seal's value fold has to collapse exactly what ``array_equivalent``
    collapses or the refusal lands on ``AXIS`` instead.
    """
    if kind == "bool":
        left = np.array([1, 0, 1], dtype=np.uint8).view(np.bool_)
        right = np.array([2, 0, 1], dtype=np.uint8).view(np.bool_)
    elif kind == "complex":
        left = np.array([1 + 2j, complex(float("nan"), 0.0), 3 + 0j])
        right = left.copy()
        right.view(np.uint64)[2] = 0x7FF8000000000011
    else:
        left = np.array([-0.0, 1.0, 2.0])
        right = np.array([0.0, 1.0, 2.0])
    frame = _frame()
    expected = _with_person_axis(frame, pd.Index(left))
    actual = _with_person_axis(frame, pd.Index(right))
    assert expected.person.index.identical(actual.person.index)
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_NATIVE_BITS"


def test_an_axis_dtype_class_difference_keeps_the_series_code():
    """``Index.identical`` compares dtypes with ``==``; the class is ``_series``'."""
    frame = _frame()
    expected = _with_person_axis(
        frame, pd.Index(np.array([101, 103, 107], dtype=np.longlong), name="source_row")
    )
    actual = _with_person_axis(
        frame, pd.Index(np.array([101, 103, 107], dtype=np.int64), name="source_row")
    )
    left, right = expected.person.index, actual.person.index
    if type(left.dtype) is type(right.dtype):
        pytest.skip("this platform does not carry a distinct longlong dtype class")
    assert left.dtype == right.dtype and left.identical(right)
    assert (
        _refuses_frames(expected, actual)
        == "SURVEY_POPULATION_REPLAY_SERIES_DTYPE_OR_LENGTH"
    )


def test_a_structured_dtype_carrying_an_object_field_refuses_as_native_bits():
    """``_array_bytes_equal`` returns False here; it does not raise its own code."""
    frame = _with_column(_frame(), "structured", np.zeros(3, dtype=[("a", "O")]))
    assert _refuses_frames(frame, frame) == "SURVEY_POPULATION_REPLAY_NATIVE_BITS"


@pytest.mark.parametrize(
    "left,right",
    [
        ((0, 1, 1), (0, 1, 7)),
        ((0, 0, 1), (5, 5, 2)),
    ],
)
def test_range_index_parameters_that_no_label_shows_are_accepted(left, right):
    """``equals`` compares materialised labels, so the descriptor is invisible.

    A seal derived from the store's index encoding would fold start/stop/step
    (``store.py`` writes them for ``kind='range'``) and be stricter than the
    predicate it replaces. The axis seal materialises the labels instead.
    """
    one, other = pd.RangeIndex(*left), pd.RangeIndex(*right)
    assert list(one) == list(other) and one.identical(other)
    assert _verdict(_same_axis_seal, _axis_seal(one), _axis_seal(other)) is None
    assert _verdict(_axis, one, other) is None


def test_a_comparable_whose_equality_refuses_itself_still_refuses():
    """``identical`` applies ``==`` per comparable; a tuple compare would not."""

    class _Refusing(float):
        def __eq__(self, other):
            return False

        __hash__ = float.__hash__

    name = _Refusing(1.0)
    frame = _frame()
    expected, actual = _frame(), _frame()
    for target in (expected, actual):
        target.table("household").index = pd.Index([0, 1], name=name)
    del frame
    assert (
        expected.table("household").index.name is actual.table("household").index.name
    )
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_AXIS"


def test_a_masked_backing_that_is_falsy_but_not_zero_is_not_canonical():
    """``np.all(rd[rm] == 0)`` is the predicate; truthiness is not it."""
    expected, actual = _frame(), _frame()
    for frame, fill in ((expected, "a"), (actual, "")):
        array = frame.person["nullable_integer"].array
        object.__setattr__(array, "_data", np.array(["x", fill, fill], dtype="<U1"))
    assert (
        _refuses_frames(expected, actual)
        == "SURVEY_POPULATION_REPLAY_NONCANONICAL_NULL_BACKING"
    )


def test_a_series_subclass_column_refuses_though_the_values_match():
    """pandas propagates a subclass that overrides ``_constructor``."""

    class _SubSeries(pd.Series):
        @property
        def _constructor(self):
            return _SubSeries

    frame = _frame()
    strata = _SubSeries(frame.strata)
    assert type(strata.copy()) is _SubSeries
    substituted = _rebuild(frame, strata=strata)
    if type(substituted.strata) is pd.Series:
        pytest.skip("the frame normalised the subclass away on this pandas")
    assert (
        _refuses_frames(substituted, substituted)
        == "SURVEY_POPULATION_REPLAY_SERIES_DTYPE_OR_LENGTH"
    )


def test_an_observer_snapshot_preserves_the_replay_seal():
    """The base financial runner seals what it keeps, and drops what it does not.

    A declared consumer is sealed after ``_observer_snapshot`` detaches it and
    every other node is sealed live, so the two must produce the same record --
    otherwise a dropped node's seal would describe a population the comparison
    would never have seen.
    """
    from microcosm.graph.executor import _observer_snapshot

    live = _population(_frame())
    detached = _observer_snapshot(live)
    assert detached is not live
    assert detached.frame.table("person") is not live.frame.table("person")
    assert replayed_population_seal(detached) == replayed_population_seal(live)
    _accepts_populations(live, detached)
    _accepts_populations(detached, live)


def test_the_strata_name_is_part_of_the_comparison():
    """``Frame`` normalises the strata name, so this is only reachable after."""
    expected, actual = _frame(), _frame()
    assert actual.strata.name == expected.strata.name == "stratum"
    actual.strata.name = "renamed_strata"
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_STRATA_NAME"


def test_a_string_columns_missing_mask_alone_is_part_of_the_comparison():
    expected = _with_column(
        _frame(), "text", pd.array(["a", "b", pd.NA], dtype="string")
    )
    actual = _with_column(_frame(), "text", pd.array(["a", pd.NA, "b"], dtype="string"))
    assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_STRING_MASK"


@pytest.mark.parametrize("name", [np.int64(5), Decimal("1"), b"bytes"])
def test_an_axis_name_outside_the_store_grammar_refuses_on_both_paths(name):
    """``_name_bytes`` converts the codec's own refusal into this code."""
    frame = _frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"].index = tables["household"].index.rename(name)
    unsupported = _rebuild(frame, tables=tables)
    assert (
        _refuses_frames(unsupported, unsupported)
        == "SURVEY_POPULATION_REPLAY_UNSUPPORTED_AXIS_NAME"
    )


@pytest.mark.parametrize(
    "left,right,accepted",
    [
        ("Int64", "Int64", True),
        ("Int64", "Int64-value", False),
        ("Int64", "Int64-mask", False),
        ("string", "string", True),
        ("string", "string-value", False),
        ("datetime", "datetime", True),
        ("datetime", "datetime-nat", False),
    ],
)
def test_exotic_index_dtypes_agree_on_both_paths(left, right, accepted):
    """A masked, string or datetime axis materialises through ``np.asarray``.

    These are the acceptance halves as much as the refusals: a seal that is
    stricter than the predicate it replaces turns a green run red, and an axis
    is the easiest place to become stricter by accident.
    """
    built = {
        "Int64": pd.Index(pd.array([1, 2, pd.NA], dtype="Int64")),
        "Int64-value": pd.Index(pd.array([1, 3, pd.NA], dtype="Int64")),
        "Int64-mask": pd.Index(pd.array([1, pd.NA, 3], dtype="Int64")),
        "string": pd.Index(pd.array(["a", "b", pd.NA], dtype="string")),
        "string-value": pd.Index(pd.array(["a", "c", pd.NA], dtype="string")),
        "datetime": pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "datetime-nat": pd.DatetimeIndex(["2020-01-01", "NaT", "2020-01-03"]),
    }
    frame = _frame()
    expected = _with_person_axis(frame, built[left])
    actual = _with_person_axis(frame, built[right])
    if accepted:
        _accepts_frames(expected, actual)
    else:
        assert _refuses_frames(expected, actual) == "SURVEY_POPULATION_REPLAY_AXIS"


def test_a_foreign_frame_seal_record_refuses_frame_seal_protocol():
    """The three codes this change adds are fail-closed guards, not comparisons.

    They cannot go through the agreement driver above, because the object path
    never holds a seal, so a malformed or foreign record has no counterpart
    there. They are what stands between a swapped, truncated or foreign record
    and a silently vacuous ``FINANCIAL_NODE_POPULATION_CHANGED``, and until
    this test they were the only refusals in the module with no test at all.
    """
    good = replayed_frame_seal(_frame())
    assert len(good) == 11 and good[0] == SEAL_PROTOCOL
    for spoiled in (
        list(good),  # not a tuple at all
        good[:-1],  # truncated
        good + (None,),  # lengthened
        ("microcosm.us.survey-population-replay-seal.v0", *good[1:]),  # older tag
    ):
        with pytest.raises(ValueError, match=_ERROR + "FRAME_SEAL_PROTOCOL$"):
            same_replayed_frame_seals(good, spoiled)
        with pytest.raises(ValueError, match=_ERROR + "FRAME_SEAL_PROTOCOL$"):
            same_replayed_frame_seals(spoiled, good)

    # Agreeing on the wrong protocol is still a refusal: the guard pins the
    # version, it does not merely require the two sides to match.
    stale = ("microcosm.us.survey-population-replay-seal.v0", *good[1:])
    with pytest.raises(ValueError, match=_ERROR + "FRAME_SEAL_PROTOCOL$"):
        same_replayed_frame_seals(stale, stale)


def test_a_foreign_population_seal_record_refuses_population_seal_protocol():
    """The population guard, and that it fires before any content comparison."""
    good = replayed_population_seal(_population(_frame()))
    assert len(good) == 8 and good[0] == SEAL_PROTOCOL
    for spoiled in (
        list(good),
        good[:-1],
        good + (None,),
        ("microcosm.us.survey-population-replay-seal.v0", *good[1:]),
    ):
        with pytest.raises(ValueError, match=_ERROR + "POPULATION_SEAL_PROTOCOL$"):
            same_replayed_population_seals(good, spoiled)
        with pytest.raises(ValueError, match=_ERROR + "POPULATION_SEAL_PROTOCOL$"):
            same_replayed_population_seals(spoiled, good)

    stale = ("microcosm.us.survey-population-replay-seal.v0", *good[1:])
    with pytest.raises(ValueError, match=_ERROR + "POPULATION_SEAL_PROTOCOL$"):
        same_replayed_population_seals(stale, stale)

    # A record that is both truncated and different in content refuses under
    # the protocol code, not under the content code the difference would earn.
    other = replayed_population_seal(
        _population(_rebuild(_frame(), metadata={"source": {"arm": "other"}}))
    )
    assert same_replayed_population_seals(good, good) is None
    with pytest.raises(ValueError, match=_ERROR + "POPULATION_SEAL_PROTOCOL$"):
        same_replayed_population_seals(good, other[:-1])


def test_seal_identity_refuses_anything_that_is_not_a_seal_record():
    """``seal_identity`` is what the run retains per node, so it fails closed.

    A list digests as readily as a tuple under ``repr``, so without this guard
    a caller that handed it the wrong object would get a plausible digest back
    and the node's fence would compare that digest to itself for the rest of
    the run.
    """
    good = replayed_population_seal(_population(_frame()))
    assert len(seal_identity(good)) == 64
    for foreign in (list(good), {"seal": good}, repr(good), None, 7, iter(good)):
        with pytest.raises(ValueError, match=_ERROR + "SEAL_TYPE$"):
            seal_identity(foreign)

    # The frame seal nested inside a population seal *is* a tuple, so it
    # digests -- the guard is a type fence, not a protocol fence, and the two
    # protocol guards above are what catch a record of the wrong shape.
    assert len(seal_identity(good[1])) == 64
