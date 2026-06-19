from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk import (
    LONG_GEOGRAPHY_COLUMNS,
    area_support_summary,
    build_stacked_local_matrix,
    sort_households_by_id,
    stacked_design_weights,
    stacked_weights_to_long,
)


def test_sort_households_by_id_keeps_attributes_with_ids() -> None:
    raw = pd.DataFrame(
        {
            "household_id": [2, 1, 3],
            "household_weight": [20.0, 10.0, 30.0],
            "region": ["Wales", "England", "Scotland"],
        }
    )

    sorted_households = sort_households_by_id(raw)

    assert sorted_households["household_id"].tolist() == [1, 2, 3]
    assert sorted_households["household_weight"].tolist() == [10.0, 20.0, 30.0]
    assert sorted_households["region"].tolist() == ["England", "Wales", "Scotland"]


def test_sort_households_by_id_rejects_duplicates() -> None:
    raw = pd.DataFrame({"household_id": [1, 1], "household_weight": [1.0, 2.0]})

    with pytest.raises(ValueError, match="must be unique"):
        sort_households_by_id(raw)


def test_build_stacked_local_matrix_uses_area_blocks_and_group_metrics() -> None:
    metrics = {
        "England": pd.DataFrame(
            {
                "hmrc/employment_income/amount": [1.0, 0.0, 2.0],
                "uc_households": [0.0, 1.0, 1.0],
            }
        ),
        "Scotland": pd.DataFrame(
            {
                "hmrc/employment_income/amount": [10.0, 0.0, 20.0],
                "uc_households": [0.0, 2.0, 2.0],
            }
        ),
    }
    targets = pd.DataFrame(
        {
            "code": ["S001", "E001"],
            "hmrc/employment_income/amount": [30.0, 3.0],
            "uc_households": [4.0, 2.0],
        }
    )

    stacked = build_stacked_local_matrix(
        metrics,
        targets,
        area_codes=["E001", "S001"],
        area_groups={"E001": "England", "S001": "Scotland"},
    )

    assert stacked.matrix.shape == (4, 6)
    assert stacked.targets.tolist() == [3.0, 2.0, 30.0, 4.0]
    assert stacked.target_frame["area_code"].tolist() == [
        "E001",
        "E001",
        "S001",
        "S001",
    ]
    dense = stacked.matrix.toarray()
    np.testing.assert_allclose(dense[0], [1.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(dense[1], [0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(dense[2], [0.0, 0.0, 0.0, 10.0, 0.0, 20.0])
    np.testing.assert_allclose(dense[3], [0.0, 0.0, 0.0, 0.0, 2.0, 2.0])


def test_build_stacked_local_matrix_rejects_drifted_household_index() -> None:
    metrics = {
        "England": pd.DataFrame({"population": [1.0, 2.0]}, index=[101, 102]),
        "Scotland": pd.DataFrame({"population": [20.0, 10.0]}, index=[102, 101]),
    }
    targets = pd.DataFrame({"code": ["E001"], "population": [3.0]})

    with pytest.raises(ValueError, match="household index"):
        build_stacked_local_matrix(
            metrics,
            targets,
            area_codes=["E001"],
            area_groups={"E001": "England"},
        )


def test_build_stacked_local_matrix_validates_explicit_household_ids() -> None:
    metrics = pd.DataFrame({"population": [1.0, 2.0]}, index=[102, 101])
    targets = pd.DataFrame({"code": ["E001"], "population": [3.0]})

    with pytest.raises(ValueError, match="household_ids"):
        build_stacked_local_matrix(
            metrics,
            targets,
            area_codes=["E001"],
            household_ids=[101, 102],
        )


def test_stacked_design_weights_split_base_mass_across_areas() -> None:
    weights = stacked_design_weights([100.0, 200.0], 4)

    np.testing.assert_allclose(weights, [25.0, 50.0] * 4)


def test_stacked_design_weights_preserve_zero_mass_by_default() -> None:
    weights = stacked_design_weights([0.0, 1e-9], 2)

    np.testing.assert_allclose(weights, [0.0, 5e-10, 0.0, 5e-10])


def test_stacked_design_weights_can_apply_explicit_optimizer_floor() -> None:
    weights = stacked_design_weights([0.0, 1e-9], 2, min_weight=1e-4)

    np.testing.assert_allclose(weights, [1e-4, 1e-4, 1e-4, 1e-4])


def test_stacked_weights_to_long_preserves_source_metadata() -> None:
    household_frame = pd.DataFrame(
        {
            "household_id": [102, 101],
            "source_year": [2022, 2023],
            "source_household_id": ["b", "a"],
            "source_household_key": ["2022:b", "2023:a"],
            "clone_index": [3, 0],
        }
    )

    long = stacked_weights_to_long(
        [0.5, 0.0, 1.5, 2.0],
        ["E001", "S001"],
        [101, 102],
        area_type="constituency",
        household_frame=household_frame,
    )

    assert tuple(long.columns) == LONG_GEOGRAPHY_COLUMNS
    assert long["weight"].tolist() == [0.5, 1.5, 2.0]
    assert long["area_code"].tolist() == ["E001", "S001", "S001"]
    assert long["household_id"].tolist() == [101, 101, 102]
    assert long["source_household_key"].tolist() == ["2023:a", "2023:a", "2022:b"]
    assert long["clone_index"].tolist() == [0, 0, 3]


def test_stacked_weights_to_long_rejects_missing_household_metadata() -> None:
    household_frame = pd.DataFrame({"household_id": [101], "source_year": [2023]})

    with pytest.raises(ValueError, match="missing household_id"):
        stacked_weights_to_long(
            [1.0, 2.0],
            ["E001"],
            [101, 102],
            area_type="constituency",
            household_frame=household_frame,
        )


def test_area_support_summary_counts_nonzero_households() -> None:
    long = stacked_weights_to_long(
        [0.5, 0.0, 1.5, 2.0],
        ["E001", "S001"],
        [101, 102],
        area_type="constituency",
        drop_zero=False,
    )

    summary = area_support_summary(long)

    assert summary["area_code"].tolist() == ["E001", "S001"]
    assert summary["nonzero_households"].tolist() == [1, 2]
    assert summary["weight_sum"].tolist() == [0.5, 3.5]
