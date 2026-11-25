"""The weight bound's contract, declared references and measurement (#1729).

The contract tests run the solver itself, so the statements in
``microcosm.calibrate.weight_bound`` are held to what it does. Every number
below is made up for the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import Target, TargetSet, calibrate
from microcosm.calibrate.weight_bound import (
    WEIGHT_BOUND_REFERENCES,
    weight_bound_reference,
    weight_bound_report,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

#: A pooled frame: two heavy rows (27) and four light rows (1).
STARTING = np.asarray([27.0, 27.0, 1.0, 1.0, 1.0, 1.0])
RATIO = 5.0


def _frame(weights: np.ndarray) -> Frame:
    n = len(weights)
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n),
                    "person_household_id": np.arange(n),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.arange(n),
                    "heavy": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    "light": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, WeightKind.DESIGN)},
    )


def _targets(light: float) -> TargetSet:
    # Far more mass on the light rows than their bound admits.
    return TargetSet(
        [
            Target(
                name="light",
                entity="household",
                measure="light",
                value=light,
                period=2024,
            )
        ]
    )


def _solve(frame: Frame, **options):
    return calibrate(
        frame,
        _targets(40.0),
        weight_entity="household",
        epochs=200,
        learning_rate=0.05,
        mass="conserve",
        max_weight_ratio=RATIO,
        **options,
    )


def test_the_bound_is_against_the_frame_not_the_warm_start() -> None:
    frame = _frame(STARTING)
    first = _solve(frame)
    # The light rows are pinned at five times their own starting weight.
    np.testing.assert_allclose(first.weights[2:], RATIO * STARTING[2:], rtol=1e-6)
    assert first.realized_max_weight_ratio <= RATIO * (1 + 1e-12)

    # A second batch that continues from the first batch's weights keeps the
    # bound against the frame: batches do not compound it.
    second = _solve(frame, warm_start_weights=first.weights)
    assert (second.weights <= RATIO * STARTING * (1 + 1e-12)).all()
    assert second.realized_max_weight_ratio <= RATIO * (1 + 1e-12)


def test_a_new_frame_is_a_new_reference() -> None:
    first = _solve(_frame(STARTING))
    # A stage whose frame carries the first stage's weights bounds against
    # those: against the original weights the two stages admit RATIO ** 2.
    chained = _solve(_frame(first.weights))
    against_original = weight_bound_report(
        chained.weights, STARTING, max_weight_ratio=RATIO
    )
    assert chained.realized_max_weight_ratio <= RATIO * (1 + 1e-12)
    assert RATIO < against_original["realized_max_ratio"] <= RATIO**2 * (1 + 1e-12)
    assert against_original["records_above_bound"] == 4


def test_a_heavy_record_gets_a_larger_absolute_bound() -> None:
    result = _solve(_frame(STARTING))
    own = weight_bound_report(result.weights, STARTING, max_weight_ratio=RATIO)
    assert own["reference"] == "own_starting_weight"
    assert own["records_above_bound"] == 0
    assert own["admits_the_starting_mass"] is True
    # Under the same ratio a heavy row may reach 27 times a light row's bound.
    assert RATIO * STARTING[0] == 27 * RATIO * STARTING[2]

    # Against one shared weight (the mean, 58 / 6), the heavy rows start
    # under the bound here, and the report says what the bound would admit.
    shared = weight_bound_reference(STARTING, "mean")
    declared = weight_bound_report(
        result.weights, STARTING, max_weight_ratio=RATIO, reference=shared
    )
    assert declared["reference"] == "declared"
    assert declared["records_starting_above_bound"] == 0
    assert declared["realized_max_ratio"] == pytest.approx(
        (result.weights / (58 / 6)).max()
    )
    # At a tight ratio the heavy rows start above a shared bound.
    tight = weight_bound_report(
        result.weights, STARTING, max_weight_ratio=2.0, reference=shared
    )
    assert tight["records_starting_above_bound"] == 2
    assert tight["admits_the_starting_mass"] is True


def test_the_declared_references() -> None:
    assert WEIGHT_BOUND_REFERENCES == ("own", "mean", "group_mean")
    np.testing.assert_array_equal(weight_bound_reference(STARTING), STARTING)
    np.testing.assert_allclose(weight_bound_reference(STARTING, "mean"), 58 / 6)
    areas = ["A", "B", "A", "A", "B", "B"]
    np.testing.assert_allclose(
        weight_bound_reference(STARTING, "group_mean", groups=areas),
        [29 / 3] * 6,
    )
    np.testing.assert_allclose(
        weight_bound_reference([4.0, 2.0, 6.0], "group_mean", groups=[1, 2, 1]),
        [5.0, 2.0, 5.0],
    )
    with pytest.raises(ValueError, match="must be one of"):
        weight_bound_reference(STARTING, "median")
    with pytest.raises(ValueError, match="needs each record's group"):
        weight_bound_reference(STARTING, "group_mean")
    with pytest.raises(ValueError, match="do not align"):
        weight_bound_reference(STARTING, "group_mean", groups=["A"])
    with pytest.raises(ValueError, match="strictly positive"):
        weight_bound_reference([1.0, 0.0])


def test_the_report_measures_and_refuses_bad_inputs() -> None:
    weights = np.asarray([10.0, 1.0, 0.0, 9.0])
    starting = np.asarray([1.0, 1.0, 1.0, 1.0])
    report = weight_bound_report(weights, starting, max_weight_ratio=5.0)
    assert report == {
        "reference": "own_starting_weight",
        "max_weight_ratio": 5.0,
        "records": 4,
        "realized_max_ratio": 10.0,
        "records_above_bound": 2,
        "weight_share_above_bound": 0.95,
        "records_starting_above_bound": 0,
        "admits_the_starting_mass": True,
    }
    # Bounds that sum below the starting total cannot conserve the mass.
    assert (
        weight_bound_report(
            weights, starting, max_weight_ratio=0.5, reference=starting
        )["admits_the_starting_mass"]
        is False
    )
    for arguments, match in (
        (dict(max_weight_ratio=0.0), "positive and finite"),
        (dict(max_weight_ratio=5.0, reference=[1.0, 1.0]), "do not align"),
        (dict(max_weight_ratio=5.0, reference=[1.0, 1.0, 0.0, 1.0]), "positive"),
    ):
        with pytest.raises(ValueError, match=match):
            weight_bound_report(weights, starting, **arguments)
    with pytest.raises(ValueError, match="do not align"):
        weight_bound_report(weights[:3], starting, max_weight_ratio=5.0)
