"""Which calibration epochs a build forwards to its telemetry.

The calibrator reports one ``calibration_epoch`` event per epoch. Every build,
UK and US, forwards the same subset to its staging bundle and its telemetry
emitter (microcosm#1292): every ``CALIBRATION_EPOCH_EVERY``-th epoch and the
last epoch of each solve, so the loss curve and every solve's final loss
survive at a tenth of the rows.

A UK size run at 2,000 epochs solves the dense pool, up to ten full-length L0
probes and the refit: about 24,000 epochs. One forwarded epoch is one row in
the staging bundle's ``calibration_progress.json`` and one event in its
``events.ndjson``, both under the staging contract's 5 MiB remote cap, so the
stride also grows with the run: no run forwards more than
``CALIBRATION_MAX_EPOCH_ROWS`` strided epochs, whatever ``--epochs`` says.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "CALIBRATION_EPOCH_EVERY",
    "CALIBRATION_MAX_EPOCH_ROWS",
    "calibration_epoch_stride",
    "forwards_calibration_epoch",
]

CALIBRATION_EPOCH_EVERY = 10
CALIBRATION_MAX_EPOCH_ROWS = 2400


def calibration_epoch_stride(total_epochs: int) -> int:
    """The forwarding stride for a run whose solves total ``total_epochs``.

    At least ``CALIBRATION_EPOCH_EVERY``, and large enough that the strided
    epochs stay within ``CALIBRATION_MAX_EPOCH_ROWS``. A caller that cannot
    know its exact solve count (a budget search may stop early) passes the
    most epochs the run can solve.
    """

    return max(
        CALIBRATION_EPOCH_EVERY,
        -(-int(total_epochs) // CALIBRATION_MAX_EPOCH_ROWS),
    )


def forwards_calibration_epoch(event: Mapping[str, object], *, every: int) -> bool:
    """Whether a build forwards this calibrator report at stride ``every``.

    Only ``calibration_epoch`` reports are forwarded: every ``every``-th epoch
    and each solve's last. A solve's epoch counter restarts at 1, so a
    selection and its refit each keep their final epoch.
    """

    if event.get("kind") != "calibration_epoch":
        return False
    epoch = int(event["epoch"])
    return epoch % every == 0 or epoch == int(event["epochs"])
