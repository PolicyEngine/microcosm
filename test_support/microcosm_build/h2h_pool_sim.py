"""A simulated US head-to-head scorer log, for the d844 early-stop estimate.

It follows ``tools/score_us_release_head_to_head.py`` and CPython's
``ProcessPoolExecutor`` (3.9 and later, spawn context) where they set the
timing:

* tasks are chunk-major; the parent submits ``2 × workers`` at once, then
  one more per result it consumes, consuming results in submission order
  and printing ``after receiving`` after that submit;
* every result the pool receives releases a token; a submit takes a free
  token, or else starts a worker if fewer than ``workers`` have started;
* a start blocks the parent for ``start_cost`` seconds (the child imports
  the scorer and reads the frame); results still arrive meanwhile, but the
  parent consumes and submits nothing;
* workers take queued tasks in submission order, the longest-idle first;
* a worker's first slice carries a one-time cost.

Durations are whole seconds per artifact and chunk. The log carries the
scorer's own line formats. Nothing here is the scorer: a change to its
submit loop or to CPython's spawning needs a matching change here.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

#: Compiling the yardstick and loading the incumbent, before its pool opens.
LOAD_SECONDS = 600
#: Loading the candidate after the incumbent's pool is released.
CANDIDATE_LOAD_SECONDS = 60
#: The simulated comparison and scorecards after the candidate's last slice.
SCORECARD_SECONDS = 300


#: The runner's line before it starts the tool (tools/modal_us_stage.py). Its
#: budget is beyond any test's, so a test's ``max_wall_seconds`` sets the
#: deadline.
RUNNER_BUDGET = "BUDGET: the tool may run 1000000000s (limited by max_wall_seconds)"


def stamp(epoch: float) -> str:
    return datetime.fromtimestamp(int(epoch), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def line(epoch: float, text: str, rss: float) -> str:
    return f"[h2h {stamp(epoch)}] {text}: peak_rss={rss:.2f} GiB"


@dataclass
class Pool:
    lines: list[tuple[float, str]] = field(default_factory=list)
    opened: float = 0.0
    inits: list[float] = field(default_factory=list)
    #: When each worker, in start order, finished its first slice (None for
    #: a worker started too late to get one).
    first_ends: list[float | None] = field(default_factory=list)
    #: When each task (numbered from 1, chunk-major) started.
    starts: dict[int, float] = field(default_factory=dict)
    #: Each worker's first task, in start order (None for a worker that got
    #: none).
    first_tasks: list[int | None] = field(default_factory=list)
    #: The last slice's completion, and the parent's release after it.
    end: float = 0.0
    released: float = 0.0


def pool(
    artifact: str,
    *,
    start: float,
    workers: int,
    start_cost: int,
    chunk_seconds: list[int],
    slices_per_chunk: int,
    first_slice_extra: int = 0,
    rss: float = 4.0,
) -> Pool:
    out = Pool(opened=start)
    out.lines.append(
        (start, line(start, f"before starting {artifact} slice workers", 9.0))
    )
    chunks = len(chunk_seconds)
    tasks = [(c, s) for c in range(chunks) for s in range(slices_per_chunk)]
    now = float(start)
    tokens = 0
    submitted = 0
    first_end: dict[int, float] = {}
    first_task: dict[int, int] = {}
    idle: deque[int] = deque()
    queue: deque[int] = deque()
    pending: deque[int] = deque()
    finished: dict[int, float] = {}
    running: list[tuple[float, int, int]] = []

    def dispatch() -> None:
        while idle and queue:
            worker, task = idle.popleft(), queue.popleft()
            seconds = chunk_seconds[tasks[task][0]]
            if worker not in first_end:
                seconds += first_slice_extra
                first_end[worker] = now + seconds
                first_task[worker] = task + 1
            out.starts[task + 1] = now
            heapq.heappush(running, (now + seconds, task, worker))

    def advance(to: float) -> None:
        nonlocal now, tokens
        while running and running[0][0] <= to:
            at, task, worker = heapq.heappop(running)
            now = at
            finished[task] = at
            tokens += 1
            chunk, index = tasks[task]
            text = (
                f"after scoring {artifact} chunk {chunk + 1}/{chunks} "
                f"slice {index + 1}/{slices_per_chunk}"
            )
            out.lines.append((at, line(at, text, rss)))
            idle.append(worker)
            dispatch()
        now = max(now, to)

    def submit() -> None:
        nonlocal submitted, tokens
        if submitted >= len(tasks):
            return
        queue.append(submitted)
        pending.append(submitted)
        submitted += 1
        dispatch()
        if tokens:
            tokens -= 1
        elif len(out.inits) < workers:
            advance(now + start_cost)
            out.inits.append(now)
            text = f"after initializing {artifact} slice worker"
            out.lines.append((now, line(now, text, rss)))
            idle.append(len(out.inits) - 1)
            dispatch()

    for _ in range(2 * workers):
        submit()
    while pending:
        head = pending.popleft()
        while head not in finished:
            advance(running[0][0])
        # The scorer submits the next task before it prints the line, so a
        # submit that starts a worker delays the line by the start.
        submit()
        out.lines.append(
            (now, line(now, f"after receiving {artifact} slice result", 9.0))
        )
        chunk, index = tasks[head]
        if index == slices_per_chunk - 1:
            text = f"after scoring {artifact} chunk {chunk + 1}/{chunks}"
            out.lines.append((now, line(now, text, 9.0)))
    out.end = max(finished.values())
    out.released = now
    out.lines.append((now, line(now, f"after releasing {artifact} scoring state", 9.0)))
    out.lines.sort(key=lambda row: row[0])
    out.first_ends = [first_end.get(w) for w in range(len(out.inits))]
    out.first_tasks = [first_task.get(w) for w in range(len(out.inits))]
    return out


@dataclass
class Run:
    started_at: float
    incumbent: Pool
    candidate: Pool
    incumbent_seconds: list[int]
    candidate_seconds: list[int]
    candidate_slices: int
    scorecard_seconds: int = SCORECARD_SECONDS

    @property
    def tool_end(self) -> float:
        return self.candidate.end + self.scorecard_seconds

    def log(self, until: float | None = None) -> list[str]:
        """The run's log to ``until``: the runner's budget line, printed before
        it starts the tool, then the scorer's records."""

        rows = sorted(
            [*self.incumbent.lines, *self.candidate.lines], key=lambda row: row[0]
        )
        return [RUNNER_BUDGET] + [
            text for at, text in rows if until is None or at <= until
        ]

    def times(self) -> list[float]:
        """Every second at which a record was written, in order."""

        return sorted({at for at, _ in [*self.incumbent.lines, *self.candidate.lines]})


def run(
    *,
    started_at: float,
    workers: int,
    incumbent_seconds: list[int],
    candidate_seconds: list[int],
    incumbent_slices: int,
    candidate_slices: int,
    start_cost: int,
    first_slice_extra: int = 0,
    load_seconds: int = LOAD_SECONDS,
    candidate_load_seconds: int = CANDIDATE_LOAD_SECONDS,
    scorecard_seconds: int = SCORECARD_SECONDS,
) -> Run:
    incumbent = pool(
        "incumbent",
        start=started_at + load_seconds,
        workers=workers,
        start_cost=start_cost,
        chunk_seconds=incumbent_seconds,
        slices_per_chunk=incumbent_slices,
        first_slice_extra=first_slice_extra,
        rss=4.0,
    )
    candidate = pool(
        "candidate",
        start=incumbent.released + candidate_load_seconds,
        workers=workers,
        start_cost=start_cost,
        chunk_seconds=candidate_seconds,
        slices_per_chunk=candidate_slices,
        first_slice_extra=first_slice_extra,
        rss=11.0,
    )
    return Run(
        started_at,
        incumbent,
        candidate,
        list(incumbent_seconds),
        list(candidate_seconds),
        candidate_slices,
        scorecard_seconds,
    )
