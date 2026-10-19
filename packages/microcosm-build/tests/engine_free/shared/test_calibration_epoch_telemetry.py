"""The calibration epochs every build forwards to its telemetry (microcosm#1292)."""

import pytest

from microcosm.build.calibration_epoch_telemetry import (
    CALIBRATION_EPOCH_EVERY,
    CALIBRATION_MAX_EPOCH_ROWS,
    calibration_epoch_stride,
    forwards_calibration_epoch,
)


def _forwarded_epochs(epochs: int, *, every: int) -> list[int]:
    return [
        epoch
        for epoch in range(1, epochs + 1)
        if forwards_calibration_epoch(
            {
                "kind": "calibration_epoch",
                "epoch": epoch,
                "epochs": epochs,
                "loss": 1.0,
            },
            every=every,
        )
    ]


def test_calibration_epoch_rule_forwards_each_solve_first_last_and_every_tenth_epoch():
    assert CALIBRATION_EPOCH_EVERY == 10
    assert _forwarded_epochs(25, every=CALIBRATION_EPOCH_EVERY) == [1, 10, 20, 25]
    # A last epoch on the stride is forwarded once.
    assert _forwarded_epochs(30, every=CALIBRATION_EPOCH_EVERY) == [1, 10, 20, 30]
    # A solve shorter than the stride still reports its first and final loss.
    assert _forwarded_epochs(5, every=CALIBRATION_EPOCH_EVERY) == [1, 5]
    # A one-epoch solve's first epoch is its last, forwarded once.
    assert _forwarded_epochs(1, every=CALIBRATION_EPOCH_EVERY) == [1]
    assert _forwarded_epochs(40, every=13) == [1, 13, 26, 39, 40]


def test_forwarded_calibration_epochs_fall_at_every_solve_boundary():
    """A reader can tell consecutive solves apart from the forwarded rows alone.

    A selection and its refit run the same number of epochs, and each solve's
    epoch counter restarts at 1. The calibration-diagnostics build monitor
    starts a new solver pass when a row's epoch is lower than the row before
    it (or its epoch total differs), so the forwarded epochs must fall at each
    boundary, as the full per-epoch stream does, even when both solves are
    shorter than the stride.
    """

    boundaries_without_a_fall = []
    for every in (CALIBRATION_EPOCH_EVERY, 13):
        for epochs in range(2, 41):
            # Both solves forward the same epochs: the refit's first forwarded
            # row directly follows the selection's last.
            forwarded = _forwarded_epochs(epochs, every=every)
            if forwarded[0] >= forwarded[-1]:
                boundaries_without_a_fall.append((epochs, every))

    assert boundaries_without_a_fall == []


def test_calibration_epoch_rule_ignores_other_progress_events():
    # Budget-search reports share the progress callback but are not epochs;
    # the closing report carries no epoch at all.
    assert not forwards_calibration_epoch(
        {"kind": "budget_probe", "epoch": 10, "epochs": 10}, every=10
    )
    assert not forwards_calibration_epoch({"kind": "budget_search_done"}, every=10)


@pytest.mark.parametrize(
    ("total_epochs", "expected_stride"),
    [
        (1, 10),
        (3_000, 10),
        (24_000, 10),
        (24_001, 11),
        (30_000, 13),
        (120_000, 50),
    ],
)
def test_calibration_epoch_stride_grows_to_keep_the_strided_epochs_under_the_cap(
    total_epochs, expected_stride
):
    assert CALIBRATION_MAX_EPOCH_ROWS == 2_400
    stride = calibration_epoch_stride(total_epochs)
    assert stride == expected_stride
    assert total_epochs // stride <= CALIBRATION_MAX_EPOCH_ROWS


@pytest.mark.parametrize(
    ("epochs_per_solve", "solves", "expected_rows"),
    [
        # The default US run: an L0 selection and its refit.
        (1_500, 2, 302),
        # The default UK size run: the dense solve, ten budget-search probes
        # and the refit, 2,400 epochs on the stride and each solve's first.
        (2_000, 12, 2_412),
        (15_000, 2, 2_310),
        (30_000, 1, 2_309),
        # Last epochs off the stride took even the rule without first epochs
        # past a flat 2,400 rows (to 2,401 here).
        (3_428, 7, 2_408),
    ],
)
def test_forwarded_rows_stay_within_the_cap_plus_two_per_solve(
    epochs_per_solve, solves, expected_rows
):
    """The stride caps the strided epochs; each solve adds its first and last."""

    every = calibration_epoch_stride(epochs_per_solve * solves)
    forwarded_rows = solves * len(_forwarded_epochs(epochs_per_solve, every=every))

    assert forwarded_rows == expected_rows
    assert forwarded_rows <= CALIBRATION_MAX_EPOCH_ROWS + 2 * solves
