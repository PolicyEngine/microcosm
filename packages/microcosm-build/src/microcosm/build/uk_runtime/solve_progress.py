"""Readable progress lines for a UK doctrine solve and its size selection.

The calibrator reports one ``calibration_epoch`` event per epoch and, under a
budget search, one ``budget_probe`` event per finished probe and a closing
``budget_search_done`` event. A five-hour size run is otherwise silent between
"solving..." and its manifest (microcosm#355), so this turns those events into
timestamped lines a log reader can follow: a loss line every ``every`` epochs
and at the last epoch, one line per probe with its penalty, open mass and
drawability verdict, and one line when the search stops.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime

ProgressLine = Callable[[str], None]
ProgressEvent = Mapping[str, object]

DEFAULT_PROGRESS_EVERY = 100


def _stage(event: ProgressEvent) -> str:
    if event.get("budget_search"):
        return (
            f"probe {event.get('budget_iteration')}/{event.get('budget_iters')} "
            f"(lambda {float(event.get('l0_lambda', float('nan'))):.3g})"
        )
    phase = event.get("phase")
    if phase == "size_refit":
        return "refit"
    if phase == "size_search":
        return "search"
    return "dense solve"


def _stamp() -> str:
    return datetime.now(UTC).strftime("%H:%M:%SZ")


def uk_solve_progress_callback(
    sink: ProgressLine, *, every: int = DEFAULT_PROGRESS_EVERY
) -> Callable[[dict[str, object]], None]:
    """Build a calibrator progress callback that writes lines to ``sink``."""
    if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
        raise ValueError(f"every must be a positive integer, got {every!r}.")

    def callback(event: dict[str, object]) -> None:
        kind = event.get("kind")
        if kind == "calibration_epoch":
            epoch = int(event["epoch"])
            epochs = int(event["epochs"])
            if epoch % every == 0 or epoch == epochs:
                sink(
                    f"{_stamp()} {_stage(event)}: epoch {epoch}/{epochs} "
                    f"loss {float(event['loss']):.5g}"
                )
        elif kind == "budget_probe":
            measure = event.get("measure")
            detail = ""
            if event.get("certainty_count") is not None:
                detail = (
                    f", certainties {event.get('certainty_count')}, boundary draw "
                    f"{event.get('boundary_draw')} from mass "
                    f"{float(event.get('boundary_mass', 0.0)):.1f} "
                    f"(max {float(event.get('boundary_max', 0.0)):.3f})"
                )
            sink(
                f"{_stamp()} probe {event.get('budget_iteration')}/"
                f"{event.get('budget_iters')} done: lambda "
                f"{float(event.get('l0_lambda', float('nan'))):.4g}, "
                f"{event.get('budget_basis')} {measure} for "
                f"{event.get('target_records')} requested, verdict "
                f"{event.get('verdict')}{detail}"
            )
        elif kind == "budget_search_done":
            sink(
                f"{_stamp()} search stopped: {event.get('stopped_on')} after "
                f"{event.get('evaluations')}/{event.get('budget_iters')} probes; "
                f"selected lambda {event.get('selected_l0_lambda')}, measure "
                f"{event.get('selected_measure')} for {event.get('target_records')}, "
                f"drawable {event.get('selected_feasible')}"
            )

    return callback
