"""Invented differential and refusal tests; no private source population."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_survey_population_replay import _frame, _population, _rebuild, _with_column

from microcosm.build.us_runtime import _survey_population_witness as witness
from microcosm.build.us_runtime import survey_population_replay as oracle
from microcosm.build.us_runtime.survey_population_replay import same_replayed_population
from microcosm.graph.store import ContentStore


def _outcome(function, *args):
    try:
        function(*args)
    except ValueError as error:
        return str(error)
    return None


def _compare(expected, actual):
    original = _outcome(same_replayed_population, expected, actual)
    observed = _outcome(
        witness.same_witness,
        witness.population_witness(expected),
        witness.population_witness(actual),
    )
    assert observed == original
    return observed


def test_complete_store_materialization_and_direction(tmp_path):
    before = _population(_frame())
    store = ContentStore(tmp_path / "invented")
    store.put_frame("c" * 64, before.frame)
    after = replace(before, frame=store.load_frame("c" * 64))
    assert _compare(before, after) is None
    assert (
        _compare(after, before) == "SURVEY_POPULATION_REPLAY_NONCANONICAL_NULL_BACKING"
    )


@pytest.mark.parametrize("dtype", [*witness._MASKED])
def test_directional_null_backing_pairs(dtype):
    def population(hidden):
        array = pd.array([1, None, 0], dtype=dtype)
        array._data[1] = hidden
        return _population(_with_column(_frame(), "masked_test", array))

    variants = [population(0), population(1)]
    if type(dtype) is not pd.BooleanDtype:
        variants.append(population(2))
    for expected in variants:
        for actual in variants:
            _compare(expected, actual)
    # Each ordered pair is checked directly, without treating a -> zero <- b
    # as a proof that a and b compare equal.


@pytest.mark.parametrize(
    "dtype", ["bool", "int8", "uint32", "int64", "float32", "float64", ">i8", ">f8"]
)
def test_native_dtype_and_changed_present_bits(dtype):
    expected = _population(
        _with_column(_frame(), "numeric", np.array([1, 0, 1], dtype=dtype))
    )
    actual = _population(
        _with_column(_frame(), "numeric", np.array([1, 0, 1], dtype=dtype))
    )
    assert _compare(expected, actual) is None
    actual.frame.person.loc[actual.frame.person.index[0], "numeric"] = (
        False if dtype == "bool" else 0
    )
    assert _compare(expected, actual) is not None


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("na_value", [pd.NA, float("nan")])
def test_string_storage_and_na_policy(storage, na_value):
    dtype = pd.StringDtype(storage=storage, na_value=na_value)
    expected = _population(
        _with_column(
            _frame(), "string_test", pd.array(["", "x\x00y", None], dtype=dtype)
        )
    )
    actual = _population(
        _with_column(
            _frame(), "string_test", pd.array(["", "x\x00y", None], dtype=dtype)
        )
    )
    assert _compare(expected, actual) is None


def test_string_storage_and_na_cross_policy_pairs_match_oracle():
    cases = [
        pd.Series(
            ["", "value", None], dtype=pd.StringDtype(storage=storage, na_value=na)
        )
        for storage in ("python", "pyarrow")
        for na in (pd.NA, float("nan"))
    ]
    for left in cases:
        for right in cases:
            assert _outcome(oracle._series, left, right) == _outcome(
                witness._same_series, witness._series(left), witness._series(right)
            )


@pytest.mark.parametrize(
    "change",
    ["value", "mask", "dtype", "flags", "version", "owners", "design", "strata"],
)
def test_supported_population_refusals_match_oracle(change):
    expected, actual = _population(_frame()), _population(_frame())
    if change == "value":
        actual.frame.person["nullable_integer"].array._data[0] += 1
    elif change == "mask":
        actual.frame.person["nullable_integer"].array._mask[1] = False
    elif change == "dtype":
        actual = replace(
            actual,
            frame=_with_column(
                actual.frame,
                "nullable_integer",
                actual.frame.person.nullable_integer.astype("Int32"),
            ),
        )
    elif change == "flags":
        actual.frame.person.flags.allows_duplicate_labels = False
    elif change == "version":
        actual = replace(actual, version="changed")
    elif change == "owners":
        actual = replace(
            actual, owners={**actual.owners, ("person", "native_float"): "changed"}
        )
    elif change == "design":
        actual = replace(actual, design_weights={"household": np.array([6.0, 7.0])})
    else:
        actual.frame.strata.iloc[0] = "changed"
    assert _compare(expected, actual) is not None


@pytest.mark.parametrize(
    "axis",
    [
        pd.date_range("2020-01-01", periods=3, freq="D"),
        pd.DatetimeIndex(pd.date_range("2020-01-01", periods=3, freq="D"), freq=None),
        pd.timedelta_range("0 days", periods=3, freq="D"),
        pd.period_range("2020-01", periods=3, freq="M"),
        pd.CategoricalIndex(["a", "b", "a"], ordered=True),
        pd.MultiIndex.from_tuples([("a", 1)]),
        pd.Index([0.0, -0.0]),
        pd.Index(["a"], name=float("nan")),
        pd.Index(["a", None], dtype=object),
    ],
)
def test_unmodeled_axes_fail_closed(axis):
    with pytest.raises(ValueError, match="^SURVEY_POPULATION_WITNESS_"):
        witness._axis(axis)


def test_exact_range_index_models_pandas_sequence_equality():
    axes = [
        pd.RangeIndex(1, 4),
        pd.RangeIndex(4, -2, -2),
        pd.RangeIndex(1, 2),
        pd.RangeIndex(1, 3, 2),
        pd.RangeIndex(2, 2),
        pd.RangeIndex(4, 4, 3),
        pd.RangeIndex(1, 4, name="different"),
    ]
    for left in axes:
        for right in axes:
            assert _outcome(oracle._axis, left, right) == _outcome(
                witness._same_axis, witness._axis(left), witness._axis(right)
            )


def test_new_axis_comparator_semantics_refuse(monkeypatch):
    monkeypatch.setattr(pd.RangeIndex, "_comparables", ["name", "unreviewed"])
    with pytest.raises(ValueError, match="WITNESS_AXIS_COMPARATORS$"):
        witness._axis(pd.RangeIndex(3))


def test_range_index_stale_array_cannot_hide_changed_range():
    expected, actual = pd.RangeIndex(3), pd.RangeIndex(3)
    _ = actual.array
    actual._range = range(1, 4)
    assert not expected.identical(actual)
    assert (
        _outcome(oracle._axis, expected, actual)
        == _outcome(witness._same_axis, witness._axis(expected), witness._axis(actual))
        == "SURVEY_POPULATION_REPLAY_AXIS"
    )


def test_range_index_unreviewed_storage_refuses():
    axis = pd.RangeIndex(3)
    axis._range = [0, 1, 2]
    with pytest.raises(ValueError, match="WITNESS_RANGE_STORAGE$"):
        witness._axis(axis)


def test_dtype_name_does_not_grant_type_admission():
    class StringDtype(pd.StringDtype):
        pass

    class Int64Dtype(pd.Int64Dtype):
        pass

    for dtype in (StringDtype(), Int64Dtype(), pd.CategoricalDtype(["a", "b"])):
        with pytest.raises(ValueError, match="^SURVEY_POPULATION_WITNESS_DTYPE$"):
            witness._dtype(dtype)


def test_immutable_bytes_do_not_alias_observation():
    population = _population(_frame())
    captured = witness.population_witness(population)
    assert type(captured) is bytes
    population.frame.person["nullable_integer"].array._data[0] += 1
    assert captured != witness.population_witness(population)
    witness.same_witness(captured, captured)


@pytest.mark.parametrize("left,right", [(True, 1), (False, 0), (1, 1.0), (-0.0, 0.0)])
def test_context_scalar_type_is_canonical_not_python_equality(left, right):
    expected = _population(_rebuild(_frame(), metadata={"flag": left}))
    actual = _population(_rebuild(_frame(), metadata={"flag": right}))
    assert _compare(expected, actual) == "SURVEY_POPULATION_REPLAY_FRAME_CONTEXT"


@pytest.mark.parametrize("left,right", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_mass_ledger_scalar_types_remain_canonical(left, right):
    expected = _population(_frame())
    row = expected.mass_ledger[0]
    expected = replace(expected, mass_ledger=(replace(row, before_total=left),))
    actual = replace(expected, mass_ledger=(replace(row, before_total=right),))
    # Independently changing a typed numeric receipt is never Python's 1 == True.
    assert _compare(expected, actual) == "SURVEY_POPULATION_REPLAY_POPULATION_CONTEXT"


@pytest.mark.parametrize("left,right", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_frame_mass_log_scalar_types_remain_canonical(left, right):
    frame = _frame()
    row = frame.mass_log[0]
    expected = _population(_rebuild(frame, mass_log=(replace(row, old_total=left),)))
    actual = _population(_rebuild(frame, mass_log=(replace(row, old_total=right),)))
    assert _compare(expected, actual) == "SURVEY_POPULATION_REPLAY_FRAME_CONTEXT"


def test_strided_array_digest_matches_contiguous_copy():
    values = np.arange(36, dtype=np.int64).reshape(6, 6)[::2, ::2]
    assert witness._array(values) == witness._array(values.copy(order="C"))


def test_native_float_bits_and_typed_object_pairs_match_oracle():
    values = np.array(
        [
            0x0000000000000000,
            0x8000000000000000,
            0x7FF8000000000001,
            0x7FF8000000000002,
        ],
        dtype="uint64",
    ).view("float64")
    for left in values:
        for right in values:
            a, b = pd.Series([left]), pd.Series([right])
            assert _outcome(oracle._series, a, b) == _outcome(
                witness._same_series, witness._series(a), witness._series(b)
            )
    objects = [None, pd.NA, pd.NaT, True, False, 0, 1, np.int64(1), *values, "text"]
    for left in objects:
        for right in objects:
            a = pd.Series([left], dtype=object)
            b = pd.Series([right], dtype=object)
            assert _outcome(oracle._series, a, b) == _outcome(
                witness._same_series, witness._series(a), witness._series(b)
            )


def test_eager_admission_refusal_has_documented_earlier_precedence():
    index = _frame().person.index
    expected = _population(
        _with_column(
            _frame(),
            "object_value",
            pd.Series(["first", "second", None], index=index, dtype=object),
        )
    )
    actual = _population(
        _with_column(
            _frame(),
            "object_value",
            pd.Series(["changed", object(), None], index=index, dtype=object),
        )
    )
    assert (
        _outcome(same_replayed_population, expected, actual)
        == "SURVEY_POPULATION_REPLAY_OBJECT_VALUE"
    )
    with pytest.raises(ValueError, match="^SURVEY_POPULATION_WITNESS_SCALAR$"):
        witness.population_witness(actual)
