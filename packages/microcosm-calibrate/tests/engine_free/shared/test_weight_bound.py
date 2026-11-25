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
                    "first": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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


# ---------------------------------------------------------------------------
# A declared reference, enforced by the solver
# ---------------------------------------------------------------------------


def _targets_first(value: float) -> TargetSet:
    """A target on the first heavy row alone."""

    return TargetSet(
        [
            Target(
                name="first",
                entity="household",
                measure="first",
                value=value,
                period=2024,
            )
        ]
    )


def test_the_starting_weights_as_the_reference_are_the_default_solve() -> None:
    frame = _frame(STARTING)
    default = _solve(frame)
    declared = _solve(frame, max_weight_reference=STARTING.copy())
    # The same bytes: the reference only replaces what the ratio multiplies.
    assert declared.weights.tobytes() == default.weights.tobytes()
    assert declared.loss_trajectory.tobytes() == default.loss_trajectory.tobytes()
    assert "weight_bound" not in default.options
    assert declared.options["weight_bound"] == {
        "reference": "declared",
        "records_started_above_bound": 0,
        "realized_max_ratio_to_reference": pytest.approx(RATIO),
    }


def test_a_shared_reference_gives_every_record_one_absolute_bound() -> None:
    frame = _frame(STARTING)
    shared = weight_bound_reference(STARTING, "mean")  # 58 / 6 for every record
    # The target asks 50 of the first heavy row, which starts at 27.
    solve = dict(
        weight_entity="household",
        epochs=400,
        learning_rate=0.05,
        mass="conserve",
        max_weight_ratio=3.0,
    )
    own = calibrate(frame, _targets_first(50.0), **solve)
    result = calibrate(
        frame, _targets_first(50.0), **solve, max_weight_reference=shared
    )
    bound = 3.0 * shared
    # Against its own starting weight the row may reach 81, so it reaches 50.
    assert own.weights[0] == pytest.approx(50.0, rel=0.02)
    # Against the shared weight every record has the bound 29, and the row
    # stops there.
    assert (result.weights <= bound * (1 + 1e-12)).all()
    assert result.weights[0] == pytest.approx(29.0, rel=1e-6)
    assert result.weights.sum() == pytest.approx(STARTING.sum())
    assert result.options["weight_bound"] == {
        "reference": "declared",
        "records_started_above_bound": 0,
        "realized_max_ratio_to_reference": pytest.approx(3.0),
    }
    # The frame's weights are still the solve's starting point and record.
    np.testing.assert_array_equal(result.initial_weights, STARTING)
    report = weight_bound_report(
        result.weights, STARTING, max_weight_ratio=3.0, reference=shared
    )
    assert report["records_above_bound"] == 0


def test_a_record_that_starts_above_its_bound_starts_from_the_bound() -> None:
    frame = _frame(STARTING)
    shared = weight_bound_reference(STARTING, "mean")
    # Two times the mean (19.3) is below the heavy rows' starting weight (27).
    result = calibrate(
        frame,
        _targets(4.0),  # the light rows already hit it: nothing to fit
        weight_entity="household",
        epochs=5,
        mass="conserve",
        max_weight_ratio=2.0,
        max_weight_reference=shared,
    )
    bound = 2.0 * shared
    assert result.options["weight_bound"]["records_started_above_bound"] == 2
    assert (result.weights <= bound * (1 + 1e-12)).all()
    assert result.weights.sum() == pytest.approx(STARTING.sum())
    # The mass the heavy rows gave up went to the light rows.
    assert (result.weights[2:] > STARTING[2:]).all()
    # Free mass: the heavy rows are held to the bound and nothing is refilled.
    free = calibrate(
        frame,
        _targets(4.0),
        weight_entity="household",
        epochs=5,
        mass="free",
        max_weight_ratio=2.0,
        max_weight_reference=shared,
    )
    assert (free.weights <= bound * (1 + 1e-12)).all()
    assert free.weights.sum() < STARTING.sum()


def test_a_warm_start_keeps_the_declared_bound() -> None:
    frame = _frame(STARTING)
    shared = weight_bound_reference(STARTING, "mean")
    options = dict(max_weight_reference=shared)
    first = _solve(frame, **options)
    second = _solve(frame, warm_start_weights=first.weights, **options)
    assert (second.weights <= RATIO * shared * (1 + 1e-12)).all()
    assert second.weights.sum() == pytest.approx(STARTING.sum())


def test_the_softmax_parametrization_holds_the_declared_bound() -> None:
    shared = weight_bound_reference(STARTING, "mean")
    result = calibrate(
        _frame(STARTING),
        _targets_first(50.0),
        weight_entity="household",
        epochs=400,
        learning_rate=0.05,
        mass="conserve",
        mass_parametrization="softmax",
        max_weight_ratio=3.0,
        max_weight_reference=shared,
    )
    assert (result.weights <= 3.0 * shared * (1 + 1e-12)).all()
    assert result.weights[0] == pytest.approx(29.0, rel=1e-5)
    assert result.weights.sum() == pytest.approx(STARTING.sum())


def test_bounds_that_cannot_hold_the_mass_are_refused() -> None:
    frame = _frame(STARTING)
    small = np.full(6, 1.0)  # 5 x 6 = 30 admitted, 58 to conserve
    with pytest.raises(ValueError, match="mass cannot be conserved"):
        _solve(frame, max_weight_reference=small)
    # Free mass has no total to hold.
    free = calibrate(
        frame,
        _targets(4.0),
        weight_entity="household",
        epochs=3,
        mass="free",
        max_weight_ratio=RATIO,
        max_weight_reference=small,
    )
    assert (free.weights <= RATIO * small * (1 + 1e-12)).all()
    # A ratio below one is admitted when the reference is large enough.
    large = np.full(6, 100.0)
    below_one = calibrate(
        frame,
        _targets(4.0),
        weight_entity="household",
        epochs=3,
        mass="conserve",
        max_weight_ratio=0.5,
        max_weight_reference=large,
    )
    assert below_one.weights.sum() == pytest.approx(STARTING.sum())
    # Without a reference a ratio below one cannot conserve the mass.
    with pytest.raises(ValueError, match="is infeasible"):
        calibrate(
            frame,
            _targets(4.0),
            weight_entity="household",
            epochs=3,
            mass="conserve",
            max_weight_ratio=0.5,
        )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (dict(max_weight_ratio=None), "pass max_weight_ratio too"),
        (dict(method="prox"), "Adam solve without L0 gates"),
        (dict(l0_lambda=0.01), "Adam solve without L0 gates"),
        (dict(target_records=3), "Adam solve without L0 gates"),
        (dict(max_weight_reference=np.ones(5)), "must match"),
        (dict(max_weight_reference=np.zeros(6)), "strictly positive"),
        (dict(max_weight_reference=np.full(6, np.nan)), "strictly positive"),
    ],
)
def test_an_invalid_declared_reference_is_refused(options, match) -> None:
    call = dict(
        weight_entity="household",
        epochs=3,
        mass="conserve",
        max_weight_ratio=RATIO,
        max_weight_reference=np.full(6, 20.0),
    )
    call.update(options)
    with pytest.raises(ValueError, match=match):
        calibrate(_frame(STARTING), _targets(4.0), **call)


def test_the_prebuilt_system_solve_takes_the_reference_too() -> None:
    import torch

    from microcosm.calibrate.solve import solve_constraint_system

    operator = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    shared = weight_bound_reference(STARTING, "mean")
    arguments = dict(
        mass="conserve", max_weight_ratio=3.0, epochs=400, learning_rate=0.05
    )
    weights, _trajectory = solve_constraint_system(
        operator, np.asarray([50.0]), STARTING, max_weight_reference=shared, **arguments
    )
    assert (weights <= 3.0 * shared * (1 + 1e-12)).all()
    assert weights[0] == pytest.approx(29.0, rel=1e-6)
    own, _ = solve_constraint_system(
        operator, np.asarray([50.0]), STARTING, **arguments
    )
    assert own[0] == pytest.approx(50.0, rel=0.02)
