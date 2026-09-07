"""Three-table extraction against outputs captured from the unedited wrapper."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import housing_inputs as housing
from microcosm.frame import US_SCHEMA, Frame, MassChangeRecord, WeightKind, Weights

GOLDEN = Path(__file__).with_name("golden") / "us_housing_tables_pre_extraction.json"


def _frame(*, incumbent=False):
    households = np.array([11, 20, 30], dtype=np.int64)
    membership = np.array([30, 11, 20, 20, 30], dtype=np.int64)
    spm_ids = np.array([201, 205, 209], dtype=np.int64)
    spm_membership = np.array([209, 201, 205, 205, 209], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": np.arange(5, dtype=np.int64),
            "person_household_id": membership,
            "person_spm_unit_id": spm_membership,
            "SPM_CAPHOUSESUB": np.array([450.0, -0.0, 0.0, 0.0, 450.0]),
            "SPM_TENMORTSTATUS": np.array([2.0, 3.0, 1.0, 1.0, 2.0]),
            "person_marker": np.arange(5, dtype=np.float64),
        },
        index=pd.Index([100, 120, 125, 131, 180], name="original_person_row"),
    )
    person["nullable_marker"] = pd.array([1, None, 3, None, 5], dtype="Int64")
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": households,
                "H_TENURE": [1.0, 2.0, 3.0],
                "household_marker": [3, 4, 5],
            },
            index=pd.Index([51, 52, 58], name="household_row"),
        ),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": spm_ids, "spm_marker": [8, 9, 10]},
            index=pd.Index(["c", "a", "b"], name="spm_row"),
        ),
    }
    for i, group in enumerate(US_SCHEMA.group_entities):
        if group in {"household", "spm_unit"}:
            continue
        person[f"person_{group}_id"] = membership + 1000 * (i + 1)
        tables[group] = pd.DataFrame(
            {
                f"{group}_id": households + 1000 * (i + 1),
                f"{group}_marker": [1.0, 2.0, 3.0],
            }
        )
    if incumbent:
        tables["household"].insert(1, "tenure_type", "old")
        tables["spm_unit"].insert(1, "receives_housing_assistance", -1.0)
        tables["spm_unit"].insert(2, "takes_up_housing_assistance_if_eligible", "old")
        tables["spm_unit"].insert(3, "spm_unit_tenure_type", -1.0)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(np.array([4.0, 6.0, 8.0]), WeightKind.IMPORTANCE),
            "spm_unit": Weights(np.array([8.0, 9.0, 10.0]), WeightKind.DESIGN),
        },
        strata=pd.Series(["fixture"] * 5, index=person.index),
        metadata={"fixture": {"version": 1}},
        mass_log=(
            MassChangeRecord("household", 9.0, 18.0, 2.0, "fixture mass history"),
        ),
    )


def _snapshot(table):
    return {
        "columns": list(table.columns),
        "dtypes": [str(dtype) for dtype in table.dtypes],
        "index": {
            "class": type(table.index).__name__,
            "dtype": str(table.index.dtype),
            "name": table.index.name,
            "values": table.index.tolist(),
        },
        "values": table.to_numpy().tolist(),
    }


INVALID = {
    "household_unknown": ("household", "H_TENURE", 0, 9.0),
    "household_fraction": ("household", "H_TENURE", 0, 1.5),
    "household_missing": ("household", "H_TENURE", 0, np.nan),
    "household_infinite": ("household", "H_TENURE", 0, np.inf),
    "spm_unknown": ("person", "SPM_TENMORTSTATUS", 1, 0.0),
    "spm_fraction": ("person", "SPM_TENMORTSTATUS", 1, 1.5),
    "spm_missing": ("person", "SPM_TENMORTSTATUS", 1, np.nan),
    "subsidy_negative": ("person", "SPM_CAPHOUSESUB", 1, -1.0),
    "subsidy_missing": ("person", "SPM_CAPHOUSESUB", 1, np.nan),
    "subsidy_disagrees": ("person", "SPM_CAPHOUSESUB", 3, 5.0),
    "spm_disagrees": ("person", "SPM_TENMORTSTATUS", 3, 2.0),
}


def _invalid_frame(case):
    original = _frame()
    tables = {e: original.table(e).copy() for e in original.entities}
    entity, column, row, value = INVALID[case]
    tables[entity].iloc[row, tables[entity].columns.get_loc(column)] = value
    return Frame(
        tables,
        US_SCHEMA,
        {e: original.weights_for(e) for e in original.weighted_entities},
        original.strata,
        metadata=original.metadata,
    )


@pytest.mark.parametrize("incumbent", [False, True])
def test_pure_tables_and_frame_wrapper_equal_pre_edit_outputs(incumbent):
    frame = _frame(incumbent=incumbent)
    before = {e: frame.table(e).copy(deep=True) for e in frame.entities}
    household, spm = housing.derive_us_housing_tables(
        frame.person, frame.table("household"), frame.table("spm_unit")
    )
    expected = json.loads(GOLDEN.read_text())["outputs"][str(incumbent).lower()]
    assert {"household": _snapshot(household), "spm_unit": _snapshot(spm)} == expected
    wrapped = housing.derive_us_housing_inputs(frame)
    pd.testing.assert_frame_equal(household, wrapped.table("household"))
    pd.testing.assert_frame_equal(spm, wrapped.table("spm_unit"))
    for entity in frame.entities:
        pd.testing.assert_frame_equal(frame.table(entity), before[entity])
        if entity not in {"household", "spm_unit"}:
            pd.testing.assert_frame_equal(wrapped.table(entity), before[entity])
    assert wrapped.weighted_entities == frame.weighted_entities
    for entity in frame.weighted_entities:
        np.testing.assert_array_equal(
            wrapped.weights_for(entity).values, frame.weights_for(entity).values
        )
        assert wrapped.weights_for(entity).kind == frame.weights_for(entity).kind
    pd.testing.assert_series_equal(wrapped.strata, frame.strata)
    assert wrapped.metadata == frame.metadata and wrapped.mass_log == frame.mass_log
    assert household is not frame.table("household") and spm is not frame.table(
        "spm_unit"
    )
    household.iloc[0, household.columns.get_loc("household_marker")] = -42
    assert frame.table("household").household_marker.iloc[0] == 3


@pytest.mark.parametrize("case", list(INVALID))
def test_exact_pre_edit_invalid_raw_and_unit_constancy_refusals(case):
    frame = _invalid_frame(case)
    expected = json.loads(GOLDEN.read_text())["refusals"][case]
    for call in (
        lambda: housing.derive_us_housing_tables(
            frame.person, frame.table("household"), frame.table("spm_unit")
        ),
        lambda: housing.derive_us_housing_inputs(frame),
    ):
        with pytest.raises(ValueError) as caught:
            call()
        assert str(caught.value) == expected["message"]
        assert type(caught.value).__name__ == expected["type"]


def test_pure_table_ids_align_independently_of_person_and_group_row_order():
    frame = _frame()
    baseline = housing.derive_us_housing_inputs(frame)
    person = frame.person.iloc[[3, 1, 4, 0, 2]].copy()
    household = frame.table("household").iloc[[1, 2, 0]].copy()
    spm = frame.table("spm_unit").iloc[[2, 0, 1]].copy()
    # These are real table views. No invalid unsorted group Frame is constructed.
    result_household, result_spm = housing.derive_us_housing_tables(
        person, household, spm
    )
    pd.testing.assert_frame_equal(
        result_household, baseline.table("household").iloc[[1, 2, 0]]
    )
    pd.testing.assert_frame_equal(
        result_spm, baseline.table("spm_unit").iloc[[2, 0, 1]]
    )
    assert result_spm.receives_housing_assistance.tolist() == [True, False, False]
    assert result_spm.takes_up_housing_assistance_if_eligible.tolist() == [
        True,
        False,
        False,
    ]


def test_missing_raw_columns_retain_explicit_refusals():
    frame = _frame()
    for entity, column in [
        ("person", "SPM_CAPHOUSESUB"),
        ("person", "SPM_TENMORTSTATUS"),
        ("person", "person_spm_unit_id"),
        ("household", "H_TENURE"),
    ]:
        tables = {e: frame.table(e) for e in ("person", "household", "spm_unit")}
        tables[entity] = tables[entity].drop(columns=column)
        with pytest.raises(ValueError, match=column):
            housing.derive_us_housing_tables(
                tables["person"], tables["household"], tables["spm_unit"]
            )


def test_existing_zero_household_code_keeps_its_original_none_mapping():
    frame = _frame()
    household = frame.table("household").copy()
    household.iloc[2, household.columns.get_loc("H_TENURE")] = 0.0
    result, _ = housing.derive_us_housing_tables(
        frame.person, household, frame.table("spm_unit")
    )
    expected = housing.derive_us_housing_inputs(frame).table("household").copy()
    expected["H_TENURE"] = household.H_TENURE
    pd.testing.assert_frame_equal(result, expected)
