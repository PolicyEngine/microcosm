"""Country-neutral observations for build-stage execution."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from microcosm.frame import Frame

__all__ = [
    "ObservedTransform",
    "StageEventObserver",
    "StageEventRun",
    "StageEventStatus",
    "StageObservation",
    "StageObservationRun",
    "StageObserver",
]


type StageEventStatus = Literal["started", "completed", "failed"]
type StageEventObserver = Callable[[str, StageEventStatus, Mapping[str, object]], None]


class StageEventRun:
    """Emit a timed lifecycle for one named operation."""

    def __init__(
        self,
        *,
        stage_id: str,
        observer: StageEventObserver | None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not stage_id:
            raise ValueError("stage_id must be non-empty.")
        self.stage_id = stage_id
        self.observer = observer
        self.clock = clock
        self._started_at: float | None = None
        self._finished = False

    def __enter__(self) -> StageEventRun:
        if self._started_at is not None:
            raise RuntimeError("A StageEventRun cannot be reused.")
        self._started_at = self.clock()
        self._emit("started", {"elapsed_seconds": 0.0})
        return self

    def complete(self, **details: object) -> float:
        """Emit completion details and return elapsed seconds."""

        if self._started_at is None:
            raise RuntimeError("StageEventRun has not started.")
        if self._finished:
            raise RuntimeError("StageEventRun has already finished.")
        elapsed_seconds = self.clock() - self._started_at
        completed_details = dict(details)
        completed_details["elapsed_seconds"] = elapsed_seconds
        self._finished = True
        self._emit("completed", completed_details)
        return elapsed_seconds

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> bool:
        if exception_type is not None and not self._finished:
            assert self._started_at is not None
            self._finished = True
            self._emit(
                "failed",
                {"elapsed_seconds": self.clock() - self._started_at},
            )
        elif not self._finished:
            raise RuntimeError("StageEventRun exited without completing the operation.")
        return False

    def _emit(self, status: StageEventStatus, details: Mapping[str, object]) -> None:
        if self.observer is not None:
            self.observer(self.stage_id, status, details)


@dataclass(frozen=True)
class StageObservation:
    """Aggregate-only notification emitted around one stage execution."""

    stage_id: str
    status: StageEventStatus
    elapsed_seconds: float
    produced_column_count: int
    entity_row_counts: Mapping[str, int]


type StageObserver = Callable[[StageObservation], None]


def _entity_row_counts(frame: Frame) -> dict[str, int]:
    return {entity: int(len(frame.table(entity))) for entity in frame.entities}


class StageObservationRun:
    """Construct consistent observations around one complete stage operation."""

    def __init__(
        self,
        *,
        stage_id: str,
        input_frame: Frame,
        produced_column_count: int,
        observer: StageObserver | None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not stage_id:
            raise ValueError("stage_id must be non-empty.")
        if produced_column_count < 0:
            raise ValueError("produced_column_count must be non-negative.")
        self.stage_id = stage_id
        self.input_frame = input_frame
        self.produced_column_count = produced_column_count
        self.observer = observer
        self._event_run = StageEventRun(
            stage_id=stage_id,
            observer=self._emit_event,
            clock=clock,
        )

    def __enter__(self) -> StageObservationRun:
        self._event_run.__enter__()
        return self

    def complete(self, result: Frame) -> float:
        """Emit the completed observation and return elapsed seconds."""

        return self._event_run.complete(
            produced_column_count=self.produced_column_count,
            entity_row_counts=_entity_row_counts(result),
        )

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> bool:
        return self._event_run.__exit__(exception_type, exception, traceback)

    def _emit_event(
        self,
        stage_id: str,
        status: StageEventStatus,
        details: Mapping[str, object],
    ) -> None:
        completed = status == "completed"
        row_counts = (
            details["entity_row_counts"]
            if completed
            else _entity_row_counts(self.input_frame)
        )
        assert isinstance(row_counts, Mapping)
        self._emit(
            StageObservation(
                stage_id=stage_id,
                status=status,
                elapsed_seconds=float(details["elapsed_seconds"]),
                produced_column_count=(
                    int(details["produced_column_count"]) if completed else 0
                ),
                entity_row_counts={
                    str(entity): int(count) for entity, count in row_counts.items()
                },
            )
        )

    def _emit(self, observation: StageObservation) -> None:
        if self.observer is not None:
            self.observer(observation)


class ObservedTransform:
    """Add shared stage observations to a callable frame transform."""

    def __init__(
        self,
        transform: object,
        *,
        stage_id: str,
        produced_column_count: int,
        observer: StageObserver | None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        source_runner = getattr(transform, "run_with_sources", None)
        if not callable(transform) and not callable(source_runner):
            raise TypeError("transform must be callable or implement run_with_sources.")
        self.transform = transform
        self.stage_id = stage_id
        self.produced_column_count = produced_column_count
        self.observer = observer
        self.clock = clock

    def _run(self, runner: Callable[..., Frame], frame: Frame, *args: object) -> Frame:
        with StageObservationRun(
            stage_id=self.stage_id,
            input_frame=frame,
            produced_column_count=self.produced_column_count,
            observer=self.observer,
            clock=self.clock,
        ) as observation:
            result = runner(frame, *args)
            observation.complete(result)
            return result

    def __call__(self, frame: Frame) -> Frame:
        if not callable(self.transform):
            raise TypeError("transform is not directly callable; use run_with_sources.")
        return self._run(self.transform, frame)

    def run_with_sources(self, frame: Frame, sources: object) -> Frame:
        runner = getattr(self.transform, "run_with_sources", None)
        if callable(runner):
            return self._run(runner, frame, sources)
        return self._run(self.transform, frame)

    def __getattr__(self, name: str) -> object:
        return getattr(self.transform, name)
