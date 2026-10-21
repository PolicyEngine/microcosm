"""Which calibration epochs a build forwards to its telemetry.

The calibrator reports one ``calibration_epoch`` event per epoch. Every build,
UK and US, forwards the same subset to its staging bundle and its telemetry
emitter (microcosm#1292): the first and last epoch of each solve and every
``CALIBRATION_EPOCH_EVERY``-th epoch, so the loss curve and every solve's
starting and final loss survive at a tenth of the rows.

Each solve's epoch counter restarts at 1, and forwarding that first epoch
keeps the restart visible: the forwarded epochs fall at every boundary
between solves, as the full per-epoch stream does. A reader that starts a
new solve when the epoch falls (the calibration-diagnostics build monitor
does) therefore sees every solve, even when ``--epochs`` is no longer than
the stride.

A UK size run at 2,000 epochs solves the dense pool, up to ten full-length L0
probes and the refit: about 24,000 epochs. One forwarded epoch is one row in
the staging bundle's ``calibration_progress.json`` and one event in its
``events.ndjson``, both under the staging contract's 5 MiB remote cap, so the
stride also grows with the run: no run forwards more than
``CALIBRATION_MAX_EPOCH_ROWS`` epochs on the stride, whatever ``--epochs``
says. Each solve's first and last epoch can add one row each, so a run of
``solves`` solves forwards at most ``CALIBRATION_MAX_EPOCH_ROWS + 2 * solves``
rows.
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

    At least ``CALIBRATION_EPOCH_EVERY``, and large enough that the epochs on
    the stride stay within ``CALIBRATION_MAX_EPOCH_ROWS``; each solve's first
    and last epoch are forwarded beside them. A caller that cannot know its
    exact solve count (a budget search may stop early) passes the most epochs
    the run can solve.
    """

    return max(
        CALIBRATION_EPOCH_EVERY,
        -(-int(total_epochs) // CALIBRATION_MAX_EPOCH_ROWS),
    )


def forwards_calibration_epoch(event: Mapping[str, object], *, every: int) -> bool:
    """Whether a build forwards this calibrator report at stride ``every``.

    Only ``calibration_epoch`` reports are forwarded: each solve's first and
    last epoch and every ``every``-th epoch. A solve's epoch counter restarts
    at 1, so a selection and its refit each keep their first and final epoch,
    and the forwarded epochs fall where the refit begins.
    """

    if event.get("kind") != "calibration_epoch":
        return False
    epoch = int(event["epoch"])
    return epoch == 1 or epoch % every == 0 or epoch == int(event["epochs"])
