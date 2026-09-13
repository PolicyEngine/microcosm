"""Invented, source-independent tests of the directional cache replay boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.survey_population_replay import (
    same_replayed_frame,
    same_replayed_population,
)
from microcosm.frame import EntitySchema, Frame, MassChangeRecord, WeightKind, Weights
from microcosm.graph.population import MassRecord, Population
from microcosm.graph.store import ContentStore

_ERROR = "^SURVEY_POPULATION_REPLAY_"


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
    assert same_replayed_frame(expected, actual) is None
    original_population = _population(expected)
    assert (
        same_replayed_population(
            original_population, replace(original_population, frame=actual)
        )
        is None
    )
    # Acceptance is read-only and must not canonicalize the original in place.
    assert expected.person["nullable_integer"].array._data.tobytes() == before_integer
    assert expected.person["nullable_boolean"].array._data.tobytes() == before_boolean


def test_unchanged_noncanonical_hidden_storage_is_accepted():
    assert same_replayed_frame(_frame(), _frame()) is None


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(expected, actual)


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(expected, actual)


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(expected, actual)


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(expected, actual)


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(expected, actual)


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
    assert same_replayed_frame(expected, actual) is None


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(expected, actual)


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(expected, actual)


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(unsupported, unsupported)


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(expected, actual)


def test_multiindex_outside_reviewed_store_profile_refuses():
    frame = _frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"].index = pd.MultiIndex.from_tuples([("a", 0), ("b", 1)])
    unsupported = _rebuild(frame, tables=tables)
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_frame(unsupported, unsupported)


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
    with pytest.raises(ValueError, match=_ERROR):
        same_replayed_population(expected, actual)
