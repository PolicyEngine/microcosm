"""Independent, invented ACS observations; no microdata or model execution."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Load only this pure module, without importing the aggregate US runtime or
# any ASEC production module. This also selects this worktree's exact file.
_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/microcosm/build/us_runtime/acs_housing_universe.py"
)
_SPEC = importlib.util.spec_from_file_location("acs_housing_universe", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
housing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(housing)

_COLUMNS = [
    "interview_scope",
    "physical_unit",
    "household_kind",
    "tenure_subtype",
    "occupied_hu",
    "hu_tenure_class",
    "unresolved_reasons",
    "TEN_valid",
]


def observations(rows=((1, 1, 1),), *, index=None):
    return pd.DataFrame(
        rows,
        columns=pd.Index(["TYPEHUGQ", "NP", "TEN"], dtype=object),
        index=index,
        dtype=np.int64,
    )


def classify(frame, valid=None):
    if valid is None:
        valid = np.ones(len(frame), dtype=bool)
    return housing.classify_acs_housing_universe(frame, tenure_valid=valid)


def assert_rows(output, rows):
    assert type(output) is pd.DataFrame
    assert output.columns.tolist() == _COLUMNS
    assert all(dtype == np.dtype(np.uint8) for dtype in output.dtypes)
    assert output.to_numpy().tolist() == rows


def writable(values):
    values.setflags(write=True)
    return values


def test_mortgage_outright_paying_renter_and_no_cash_are_distinct():
    output, _ = classify(observations([(1, 2, 1), (1, 1, 2), (1, 3, 3), (1, 1, 4)]))
    assert_rows(
        output,
        [
            [0, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 2, 1, 1, 0, 1],
            [0, 1, 0, 3, 1, 2, 0, 1],
            [0, 1, 0, 4, 1, 2, 0, 1],
        ],
    )
    assert (output["occupied_hu"] == 1).sum() == 4
    assert (output["hu_tenure_class"] == 1).sum() == 2
    assert (output["hu_tenure_class"] == 2).sum() == 2


def test_unknown_tenure_stays_in_total_as_unclassified():
    output, _ = classify(
        observations([(1, 1, 1), (1, 1, 3), (1, 1, 4), (1, 1, 0)]),
        np.array([True, True, True, False]),
    )
    assert_rows(
        output,
        [
            [0, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 3, 1, 2, 0, 1],
            [0, 1, 0, 4, 1, 2, 0, 1],
            [0, 1, 0, 0, 1, 3, 2, 0],
        ],
    )
    assert (output["occupied_hu"] == 1).sum() == 4
    assert (output["hu_tenure_class"] == 2).sum() == 2


def test_invalid_tenure_payload_is_never_interpreted():
    # Even payloads that resemble valid tenure or overflow uint8 mean nothing.
    for payload in [
        np.iinfo(np.int64).min,
        -1,
        0,
        1,
        2,
        3,
        4,
        256,
        np.iinfo(np.int64).max,
    ]:
        output, _ = classify(observations([(1, 1, payload)]), np.array([False]))
        assert_rows(output, [[0, 1, 0, 0, 1, 3, 2, 0]])


def test_zero_person_housing_unit_does_not_assert_vacancy():
    output, _ = classify(
        observations([(1, 0, 1), (1, 0, 4), (1, 0, -800)]),
        np.array([True, True, False]),
    )
    assert_rows(
        output,
        [
            [0, 1, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 4, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 0],
        ],
    )


def test_both_gq_types_with_blank_tenure_stay_outside_without_conflict():
    output, _ = classify(
        observations([(2, 1, 0), (3, 2, -987), (2, 0, 400)]),
        np.zeros(3, dtype=bool),
    )
    assert_rows(output, [[0, 2, 0, 0, 2, 0, 0, 0]] * 3)


def test_gq_valid_tenure_is_preserved_without_entering_hu_universe():
    output, _ = classify(observations([(2, 1, 1), (3, 1, 4)]))
    assert_rows(output, [[0, 2, 0, 1, 2, 0, 0, 1], [0, 2, 0, 4, 2, 0, 0, 1]])


def test_every_row_keeps_position_and_duplicate_nonmonotonic_index():
    frame = observations(
        [(3, 1, 0), (1, 1, 4), (2, 1, 0), (1, 0, 2)],
        index=pd.Index([51, 9, 51, 3], name="invented_household"),
    )
    output, _ = classify(frame, np.array([False, True, False, True]))
    pd.testing.assert_index_equal(output.index, frame.index)
    assert output.index is not frame.index
    assert_rows(
        output,
        [
            [0, 2, 0, 0, 2, 0, 0, 0],
            [0, 1, 0, 4, 1, 2, 0, 1],
            [0, 2, 0, 0, 2, 0, 0, 0],
            [0, 1, 0, 2, 0, 0, 1, 1],
        ],
    )


def test_extra_columns_and_model_tenure_do_not_change_results():
    frame = observations([(1, 1, 4), (1, 1, 1)])
    expected, expected_header = classify(frame)
    frame["tenure_type"] = pd.Series(["OWNER", None], dtype=object)
    frame["unrelated"] = [object(), object()]
    frame.attrs["source_path"] = "/invented/private/source"
    output, header = classify(frame)
    pd.testing.assert_frame_equal(output, expected)
    assert header == expected_header
    assert output.attrs == {}


def test_np_has_no_invented_upper_bound():
    output, _ = classify(observations([(1, np.iinfo(np.int64).max, 4)]))
    assert_rows(output, [[0, 1, 0, 4, 1, 2, 0, 1]])


def test_empty_input_has_owned_axes_and_full_uint8_schema():
    frame = observations([], index=pd.RangeIndex(7, 7, name="empty"))
    output, header = classify(frame)
    assert_rows(output, [])
    pd.testing.assert_index_equal(output.index, frame.index)
    assert output.index is not frame.index
    assert header == classify(observations())[1]


def test_only_exact_dataframe_is_accepted():
    class DerivedFrame(pd.DataFrame):
        pass

    for invalid in [
        None,
        {},
        [[1, 1, 1]],
        observations().iloc[0],
        DerivedFrame(observations()),
    ]:
        with pytest.raises(
            housing.ACSHousingUniverseRefusalError,
            match="^households_must_be_exact_dataframe$",
        ):
            housing.classify_acs_housing_universe(
                invalid, tenure_valid=np.array([True])
            )


def test_all_three_exact_required_column_labels_are_mandatory():
    for column in ["TYPEHUGQ", "NP", "TEN"]:
        frame = observations().drop(columns=column)
        with pytest.raises(
            housing.ACSHousingUniverseRefusalError,
            match="^required_ACS_columns_missing$",
        ):
            classify(frame)
    frame = observations()
    frame.columns = pd.MultiIndex.from_tuples(
        [(name, "value") for name in frame.columns]
    )
    with pytest.raises(
        housing.ACSHousingUniverseRefusalError, match="^required_ACS_columns_missing$"
    ):
        classify(frame)


def test_duplicate_required_or_ignored_columns_are_refused():
    for labels in [
        ["TYPEHUGQ", "NP", "TEN", "TEN"],
        ["TYPEHUGQ", "NP", "TEN", "extra", "extra"],
    ]:
        frame = pd.DataFrame([np.ones(len(labels), dtype=np.int64)], columns=labels)
        with pytest.raises(
            housing.ACSHousingUniverseRefusalError,
            match="^household_columns_must_be_unique$",
        ):
            classify(frame)


@pytest.mark.parametrize("column", ["TYPEHUGQ", "NP", "TEN"])
def test_raw_columns_refuse_coercion_and_nullable_convenience(column):
    invalid_values = [
        pd.Series([1], dtype=np.int8),
        pd.Series([1], dtype=np.int32),
        pd.Series([1], dtype=np.uint64),
        pd.Series([1.0], dtype=np.float64),
        pd.Series([np.nan], dtype=np.float64),
        pd.Series([1], dtype=object),
        pd.Series(["1"], dtype=object),
        pd.Series([True], dtype=bool),
        pd.Series([1], dtype="Int64"),
        pd.Series([pd.NA], dtype="Int64"),
        pd.Series([True], dtype="boolean"),
    ]
    for values in invalid_values:
        frame = observations()
        frame[column] = values
        with pytest.raises(
            housing.ACSHousingUniverseRefusalError,
            match="^ACS_columns_must_be_numpy_int64$",
        ):
            # An unavailable TEN still requires the exact physical int64 dtype.
            classify(frame, np.array([False]))


def test_validity_requires_exact_numpy_bool_array():
    class DerivedArray(np.ndarray):
        pass

    for invalid in [
        [True],
        (True,),
        pd.Series([True]),
        pd.array([True], dtype="boolean"),
        np.array([1], dtype=np.uint8),
        np.array([1.0]),
        np.array([True], dtype=object),
        np.array([True]).view(DerivedArray),
        np.ma.array([True], mask=[False]),
        None,
    ]:
        with pytest.raises(
            housing.ACSHousingUniverseRefusalError,
            match="^TEN_validity_must_be_matching_bool_vector$",
        ):
            housing.classify_acs_housing_universe(observations(), tenure_valid=invalid)


def test_validity_requires_one_dimension_and_matching_length():
    for invalid in [
        np.array(True),
        np.array([[True]]),
        np.array([], dtype=bool),
        np.array([True, False]),
    ]:
        with pytest.raises(
            housing.ACSHousingUniverseRefusalError,
            match="^TEN_validity_must_be_matching_bool_vector$",
        ):
            classify(observations(), invalid)


def test_domains_are_strict_and_errors_do_not_expose_values_or_keys():
    for column, values, reason in [
        ("TYPEHUGQ", [-1, 0, 4, 928374], "TYPEHUGQ_outside_domain"),
        ("NP", [-1, np.iinfo(np.int64).min], "NP_must_be_nonnegative"),
        ("TEN", [-1, 0, 5, 928374], "valid_TEN_outside_domain"),
    ]:
        for value in values:
            frame = observations(index=pd.Index(["/invented/private/row-key"]))
            frame[column] = np.array([value], dtype=np.int64)
            with pytest.raises(housing.ACSHousingUniverseRefusalError) as refusal:
                classify(frame)
            assert str(refusal.value) == reason
            assert refusal.value.__cause__ is None
    # Validity does not excuse an out-of-domain GQ tenure observation.
    with pytest.raises(
        housing.ACSHousingUniverseRefusalError, match="^valid_TEN_outside_domain$"
    ):
        classify(observations([(2, 1, 0)]))


def test_validity_is_mandatory_keyword_only_and_readonly_strided_is_allowed():
    with pytest.raises(TypeError):
        housing.classify_acs_housing_universe(observations())
    with pytest.raises(TypeError):
        housing.classify_acs_housing_universe(observations(), np.array([True]))
    frame = observations([(1, 1, 4), (1, 1, -99)])
    mask = np.array([True, False, False, True])[::2]
    mask.setflags(write=False)
    output, _ = classify(frame, mask)
    assert_rows(output, [[0, 1, 0, 4, 1, 2, 0, 1], [0, 1, 0, 0, 1, 3, 2, 0]])


def test_header_is_unbound_complete_data_independent_and_fresh():
    frame = observations(index=pd.Index(["/invented/private/household-key"]))
    _, header = classify(frame)
    _, fresh = classify(
        observations([(3, 0, -777), (1, 1, 4)]), np.array([False, True])
    )
    assert header == fresh
    assert header["schema_version"] == 1
    assert header["artifact_kind"] == "microcosm.acs_housing_universe_unbound.v1"
    assert header["source_authentication"] == "unbound_semantic_classifier"
    assert header["release_eligible"] is False
    assert set(header["codebooks"]) == set(_COLUMNS)
    assert header["codebooks"]["tenure_subtype"] == {
        "0": "unavailable",
        "1": "owned_with_mortgage_or_loan",
        "2": "owned_outright",
        "3": "paying_rent",
        "4": "no_rent_paid",
    }
    assert "not_observed" in header["codebooks"]["interview_scope"]["0"]
    assert "not_observed" in header["codebooks"]["household_kind"]["0"]
    assert (
        "never an observed zero"
        in header["validity_placeholder_semantics"]["invalid_TEN"]
    )
    assert {
        "no_authenticated_source_or_source_pins",
        "no_CPS_interview_or_household_kind_observations",
        "no_B19001_income_concept_bridge",
        "no_scientific_certification",
    }.issubset(header["limits"])
    encoded = json.dumps(header, allow_nan=False)
    assert "/invented/private" not in encoded
    assert "row_count" not in encoded
    assert "sha256" not in encoded
    header["codebooks"]["tenure_subtype"]["4"] = "changed"
    header["source_mapping"]["TYPEHUGQ"]["1"] = "changed"
    header["validity_placeholder_semantics"]["invalid_TEN"] = "changed"
    header["limits"].clear()
    assert classify(observations())[1] == fresh


def test_values_masks_and_both_axes_are_owned_in_both_directions():
    frame = observations(
        [(1, 1, 4), (1, 1, 1)], index=pd.Index([31, 19], name="household")
    )
    mask = np.array([True, True])
    output, _ = classify(frame, mask)
    sibling, _ = classify(frame, mask)
    # DataFrame.copy(deep=True) can share axis buffers, so construct the
    # expected values and axes independently of either classifier result.
    expected_output = pd.DataFrame(
        [[0, 1, 0, 4, 1, 2, 0, 1], [0, 1, 0, 1, 1, 1, 0, 1]],
        index=pd.Index([31, 19], name="household"),
        columns=_COLUMNS,
        dtype=np.uint8,
    )
    assert not np.shares_memory(output.index.to_numpy(), frame.index.to_numpy())
    assert not np.shares_memory(output.columns.to_numpy(), frame.columns.to_numpy())
    assert not np.shares_memory(output.to_numpy(), frame.to_numpy())
    assert not np.shares_memory(output.to_numpy(), sibling.to_numpy())
    assert not np.shares_memory(output["TEN_valid"].to_numpy(), mask)
    writable(frame.to_numpy())[0, 2] = 2
    writable(frame.index.to_numpy())[0] = 100
    writable(frame.columns.to_numpy())[0] = "changed_input"
    mask[:] = False
    pd.testing.assert_frame_equal(output, expected_output)
    expected_input = pd.DataFrame(
        [[1, 1, 2], [1, 1, 1]],
        index=pd.Index([100, 19], name="household"),
        columns=pd.Index(["changed_input", "NP", "TEN"], dtype=object),
        dtype=np.int64,
    )
    writable(output.to_numpy())[0, 0] = 8
    writable(output.index.to_numpy())[0] = 200
    # String-backed indexes can allocate in to_numpy(); mutate the actual
    # backing array so this checks column ownership on both pandas variants.
    output.columns.values[0] = "changed_output"
    output.index.name = "changed_name"
    assert output.iloc[0, 0] == 8
    assert output.index[0] == 200
    assert output.columns[0] == "changed_output"
    pd.testing.assert_frame_equal(frame, expected_input)
    pd.testing.assert_frame_equal(sibling, expected_output)


def test_readonly_source_columns_are_not_mutated():
    arrays = {
        "TYPEHUGQ": np.array([1, 3], dtype=np.int64),
        "NP": np.array([1, 1], dtype=np.int64),
        "TEN": np.array([4, -99], dtype=np.int64),
    }
    for values in arrays.values():
        values.setflags(write=False)
    frame = pd.DataFrame(arrays, copy=False)
    output, _ = classify(frame, np.array([True, False]))
    assert_rows(output, [[0, 1, 0, 4, 1, 2, 0, 1], [0, 2, 0, 0, 2, 0, 0, 0]])
    assert frame.to_numpy().tolist() == [[1, 1, 4], [3, 1, -99]]
    assert all(not values.flags.writeable for values in arrays.values())


def test_rangeindex_materialized_cache_is_independent_both_directions():
    for mutate_input in [True, False]:
        frame = observations(
            [(1, 1, 1), (1, 1, 4)], index=pd.RangeIndex(8, 4, -2, name="household")
        )
        original_cache = frame.index.to_numpy(copy=False)
        output, _ = classify(frame)
        copied_cache = output.index.to_numpy(copy=False)
        pd.testing.assert_index_equal(output.index, frame.index)
        assert type(output.index) is pd.RangeIndex
        assert not np.shares_memory(original_cache, copied_cache)
        target, other = (
            (original_cache, copied_cache)
            if mutate_input
            else (copied_cache, original_cache)
        )
        writable(target)[0] = 555
        assert other.tolist() == [8, 6]


def test_multiindex_and_categorical_index_keep_detached_storage():
    indexes = [
        pd.MultiIndex.from_arrays([[2, 1], [8, 9]], names=["group", "household"]),
        pd.CategoricalIndex(
            [2, 1], categories=[1, 2, 3], ordered=True, name="household"
        ),
    ]
    for index in indexes:
        frame = observations([(1, 1, 1), (1, 1, 4)], index=index)
        original_values = index.to_numpy(copy=False)
        output, _ = classify(frame)
        copied_values = output.index.to_numpy(copy=False)
        pd.testing.assert_index_equal(output.index, index)
        assert not np.shares_memory(original_values, copied_values)
        writable(copied_values)[0] = (5, 5) if isinstance(index, pd.MultiIndex) else 5
        assert original_values.tolist() == (
            [(2, 8), (1, 9)] if isinstance(index, pd.MultiIndex) else [2, 1]
        )
        if isinstance(index, pd.MultiIndex):
            for level, copied_level, codes, copied_codes in zip(
                index.levels,
                output.index.levels,
                index.codes,
                output.index.codes,
                strict=True,
            ):
                assert not np.shares_memory(level.to_numpy(), copied_level.to_numpy())
                assert not np.shares_memory(codes, copied_codes)
        else:
            assert not np.shares_memory(index.codes, output.index.codes)
            assert not np.shares_memory(
                index.categories.to_numpy(), output.index.categories.to_numpy()
            )
