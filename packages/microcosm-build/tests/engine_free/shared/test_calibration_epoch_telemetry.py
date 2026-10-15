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


def test_calibration_epoch_rule_forwards_every_tenth_and_each_solve_last_epoch():
    assert CALIBRATION_EPOCH_EVERY == 10
    assert _forwarded_epochs(25, every=CALIBRATION_EPOCH_EVERY) == [10, 20, 25]
    # A last epoch on the stride is forwarded once.
    assert _forwarded_epochs(30, every=CALIBRATION_EPOCH_EVERY) == [10, 20, 30]
    # A solve shorter than the stride still reports its final loss.
    assert _forwarded_epochs(5, every=CALIBRATION_EPOCH_EVERY) == [5]
    assert _forwarded_epochs(40, every=13) == [13, 26, 39, 40]


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
def test_calibration_epoch_stride_grows_to_keep_forwarded_rows_under_the_cap(
    total_epochs, expected_stride
):
    assert CALIBRATION_MAX_EPOCH_ROWS == 2_400
    stride = calibration_epoch_stride(total_epochs)
    assert stride == expected_stride
    assert total_epochs // stride <= CALIBRATION_MAX_EPOCH_ROWS
