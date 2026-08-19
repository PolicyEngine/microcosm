from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.raking import MarginSpec, iterative_proportional_fit


def test_two_margin_sweep_matches_hand_computed_ratios() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "kind": ["x", "y", "x", "y"],
            "value": [1.0, 3.0, 2.0, 4.0],
        }
    )

    raked = iterative_proportional_fit(
        frame,
        columns=("value",),
        margins=(
            MarginSpec("group", {"a": {"value": 8.0}, "b": {"value": 12.0}}),
            MarginSpec("kind", {"x": {"value": 9.0}, "y": {"value": 11.0}}),
        ),
        iterations=1,
    )

    np.testing.assert_allclose(
        raked["value"],
        [6.0, 66.0 / 7.0, 12.0, 88.0 / 7.0],
    )


def test_single_pass_income_margin_degenerates_to_incumbent_training_calibration():
    frame = pd.DataFrame(
        {
            "income_band": ["low", "low", "high", "high"],
            "gas": [10.0, 30.0, 2.0, 0.0],
            "electricity": [4.0, 6.0, 0.0, 0.0],
        }
    )

    raked = iterative_proportional_fit(
        frame,
        columns=("gas", "electricity"),
        margins=(
            MarginSpec(
                "income_band",
                {
                    "low": {"gas": 80.0, "electricity": 20.0},
                    "high": {"gas": 10.0, "electricity": 5.0},
                },
            ),
        ),
        iterations=1,
    )

    assert raked["gas"].tolist() == [40.0, 120.0, 20.0, 0.0]
    assert raked["electricity"].tolist() == [16.0, 24.0, 0.0, 0.0]


def test_weighted_and_unweighted_means_use_distinct_denominators() -> None:
    frame = pd.DataFrame(
        {
            "band": ["a", "a"],
            "value": [10.0, 30.0],
            "weight": [1.0, 3.0],
        }
    )

    unweighted = iterative_proportional_fit(
        frame,
        columns=("value",),
        margins=(MarginSpec("band", {"a": {"value": 40.0}}),),
        iterations=1,
    )
    weighted = iterative_proportional_fit(
        frame,
        columns=("value",),
        margins=(MarginSpec("band", {"a": {"value": 40.0}}),),
        iterations=1,
        weight_column="weight",
    )

    assert unweighted["value"].tolist() == [20.0, 60.0]
    np.testing.assert_allclose(weighted["value"], [16.0, 48.0])


def test_zero_empty_and_unmapped_cells_are_left_untouched() -> None:
    frame = pd.DataFrame(
        {
            "band": ["zero", "empty", "unmapped"],
            "value": [0.0, 5.0, 7.0],
        }
    )

    raked = iterative_proportional_fit(
        frame,
        columns=("value",),
        margins=(
            MarginSpec(
                "band",
                {
                    "zero": {"value": 10.0},
                    "absent": {"value": 20.0},
                },
            ),
        ),
        iterations=1,
    )

    assert raked["value"].tolist() == [0.0, 5.0, 7.0]


def test_margin_sweep_order_is_observable_and_pinned() -> None:
    frame = pd.DataFrame(
        {
            "first": ["a", "a"],
            "second": ["x", "y"],
            "value": [1.0, 3.0],
        }
    )

    raked = iterative_proportional_fit(
        frame,
        columns=("value",),
        margins=(
            MarginSpec("first", {"a": {"value": 8.0}}),
            MarginSpec("second", {"x": {"value": 2.0}, "y": {"value": 6.0}}),
        ),
        iterations=1,
    )

    assert raked["value"].tolist() == [2.0, 6.0]
