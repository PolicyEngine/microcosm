"""Estimate a running US head-to-head's end from its own log: the d844 early stop.

The scorer (``tools/score_us_release_head_to_head.py``) prints one stderr
line at each memory boundary, and the Modal runner streams the tool's output
to the container log, so ``modal app logs <app-id>`` carries them while the
run is live. With ``--workers`` and no ``--worker-max-slices``:

* each spawned worker prints ``after initializing <artifact> slice worker``
  once and ``after scoring <artifact> chunk c/C slice s/S`` per slice;
* the parent prints ``before starting <artifact> slice workers``,
  ``after receiving <artifact> slice result`` per slice, ``after scoring
  <artifact> chunk c/C`` per chunk and ``after releasing <artifact> scoring
  state``.

Every line carries a UTC timestamp to the second and the process's
``ru_maxrss``. The scorer scores the incumbent in one pool, then the
candidate in a fresh pool.

The pool, as the estimate models it. Tasks are slices, numbered chunk-major.
The parent submits ``2 × workers`` tasks at once, then consumes results in
submission order, submitting one more task as it takes each result and
printing ``after receiving`` after that submit. ``ProcessPoolExecutor``
(CPython 3.9 and later, ``concurrent/futures/process.py``) starts workers on
demand: each result it receives releases a token; a submit takes a free
token, or else starts a worker if fewer than ``workers`` have started. A
start blocks the parent while the spawned child imports the scorer and reads
its arguments down a pipe; results still arrive meanwhile. The model takes
the block to last until the worker's first line, which comes after its
initializer, once the parent is free again: an approximation that can date
the parent's next submit late. The parent prints that submit's ``after
receiving`` line just before the worker's first line, in the same second in
every fixture; a poll between the two would read the submit as starting no
worker the log shows, a log gap for that poll (inferred, not seen). Workers
take queued tasks in submission order. A worker's first slice carries
one-time costs.

Rebuilding the past. Under that model the log fixes every submit's time:
the opening submits come at the pool's opening and after each worker start;
later ones when the parent takes a result, once it has arrived and the
previous ``after receiving`` line is out. A submit that starts no worker
(it takes a token, or the pool is full) is dated by its own ``after
receiving`` line, which the parent prints straight after it: a result can
reach the parent well after its slice line. Which
submits start a worker follows from the tokens, and the log's worker starts
check it. Stamps are whole seconds, so a result finishing in the same second
as a start can look like a token the submit did not have; a start logged
before a later submit is then put back on the submit that made it, and the
report notes it. A tie can also misdate slices with no note: a completion
stamped in a spawning submit's second but landing after it gives that
submit a token, the spawn moves to the next submit, and an idle wait reads
as slice time. It errs slow on the first chunk measured: on the toy pool in
the tests chunk 1 reads 6.3% slow and chunk 2 5.5% slow, never fast, and on
simulated d844-shaped pools chunk 1 read more than 5% slow in 29 of 200
(review r4, MODEL-6). A run that fits with less spare than that can be
stopped. A task starts as soon as it has been submitted and a
started worker is free, so at any moment the tasks started are the first
``min(submitted, finished + workers started)``. That dates every slice's
start, so each slice's own time is its completion less its start. A chunk's
seconds per slice is the mean over its slices other than workers' first
ones (``min_slices`` at least). A result the parent took whose slice line is
missing means the log streams lost lines: the rebuild stops there and the
report carries ``log_gap``. So does a pool whose worker, slice or receive
lines are there without its opening line, which no rebuild can place, and
a rebuild that does not fit the log: one that starts slices after they
end, meets a submit that took no token and started no worker it can see,
or counts starts it cannot explain. Those are the marks lost worker-start
lines leave, which the parent's later lines cannot reveal; no intact
simulated log gives one (review r5). A stretch that no stream covers between
the first record and the last is reported as ``stream_holes``: the shape a
restarted stream leaves when it does not replay what the app printed. A
slice line lost in it reads as a straggler before any log gap can show, so
after a hole the verdict waits while a slice that began before the hole
ended has no line (``hole_suspects``). A start line lost in it leaves the
rebuild a worker short until the model needs the lost worker and finds no
line for it (a log gap), and meanwhile the estimate reads late: its replay
starts one worker more than remain, and the mean start cost, without the
lost start, can be far off (review r6: a lost 78 s start among 119 to 328 s
ones read 460 s late). No later line tells a restart that lost a start line
from one that lost nothing, but each worker prints its start line once, so a
pool that has logged a start for every worker has lost none. Until then,
after a hole that could hold its start lines (one that ends once the pool
has opened and begins before the pool's release line), a time NO-GO waits
(``starts_unconfirmed``); a stall still stops the run. A released pool's
starts count only while they still set the candidate's start cost, before
its first start line. A pool that never starts every worker stays held, and
in the simulated d844 shape the last start comes 15 start costs after the
first (4,500 s at 300 s each), so a restart there can delay a stop by about
that long. A newest stream holding nothing yet beside an earlier one that
has records may be a hole that has not ended: the estimate and the stall
check wait for its first record (``newest_stream_empty``). A stream that
begins in the very second the streams before it end and repeats records
they showed there is reported in ``restart_overlaps`` (see ``parse_logs``).
These rules come from searching simulated pools with stretches of their
logs removed, for a damaged log that gives NO-GO where the intact one gives
GO (``docs/evidence/us-h2h-speed``): what the search did not find is not
ruled out. Streams are read in the watcher's order, ``app-2.log`` before
``app-10.log``.
Records from before ``--started-at`` mean the streams hold an earlier
attempt's log; the verdict waits.

The end. The estimate replays the model forward from the rebuilt present:
the running slices, the queue, the free tokens, the workers still to start
and the parent's place. Slices take their chunk's measured seconds; a chunk
not measured yet takes a measured chunk's, scaled by prior per-chunk times
(``--chunk-weights``: slice times measured locally for both artifacts) from
whichever measured chunk gives the slowest result. Before any candidate
chunk is measured, the candidate's seconds are the incumbent's, scaled
chunk by chunk by the prior ratio of the two; after, its unmeasured chunks
scale the candidate's prior by how far the incumbent's measured chunks
departed from theirs, through one ratio pooled over the candidate's
measured chunks. A candidate that does not share a departure still reads
late: in review r4's scenario (the incumbent's chunk 5 at 1.147 times its
prior, the candidate at its prior) by about 800 s. A start costs the mean of
the starts logged so far (the incumbent's until the candidate's pool logs
one, slice or not), so a candidate whose workers start slower reads early,
toward GO, until its first start line. A first slice costs the mean extra
measured so far. A running slice already past its chunk's pace is taken to
run as far past it again. That reads early until the poll passes the
midpoint of the pace and the slice's real time, then late, by up to the
slice's real time less its pace as the poll nears its end: a 3,000 s
straggler among 40 s slices reads 1,530 s late polled 2,285 s in, and up to
2,960 s late just before it ends (review r5). So one long straggler can give
a NO-GO to a run that fits. While the
incumbent is still scoring, its replay comes first and the candidate's
whole pool follows at once (loading the candidate is left out), with the
prior's slices per chunk. A fixed allowance for the comparison and
scorecards ends it: 600 s, unmeasured for the candidate (the measured 12 s
tail is the incumbent's), so a run whose last slice ends less than 600 s
before the budget is stopped though it would very likely complete.

This is an estimate, made again at every poll, not a bound. What bounds an
attempt is elsewhere: the class timeout (derived from the cost cap), the
runner's ``max_wall_seconds`` stop, the class's hard memory limit and the
scorer's own 20 GiB per-process guard. The estimate only lets the watcher
stop early a run that will not finish inside its budget, so less of the cap
is spent for nothing.

The budget is the plan's ``max_wall_seconds`` from ``--started-at``, or less
when the runner's ``ATTEMPTS:`` or ``BUDGET:`` line in the log says so. The
verdict is ``WAIT`` until a pace and (while workers are still to start) a
start cost are known, and while the log has a gap; then ``GO``, or
``NO-GO`` when the estimated end is past the budget less a margin (none by
default). Once the candidate has scored every slice the verdict stays
``GO``: what is left is cheap, and only the runner's budget stops it. A
stall is ``NO-GO`` at any point: no scorer record for an hour, or for four
times the slowest estimated slice if that is longer, counted from
``--started-at`` before the first record. The first estimate scales the
one chunk it has measured onto all the work ahead, so a prior off on that
chunk moves it by that departure's share of the rest: a run that fits with
less spare than that can be stopped. The same happens later when the
incumbent's measured chunk 5 departs from the prior and the
incumbent-times-prior estimate carries it onto the candidate: 300-600 s
late on review r4's grid. Where the log does not follow the model
(a worker start it cannot explain, a slice that seems to end before it
began), the report says so.

Memory is reported, never a verdict. On Linux a process keeps its
pre-``exec`` high-water mark in ``ru_maxrss``, and a spawned child's
pre-``exec`` memory is the parent's, so a worker's line reads the larger of
the parent's peak at the spawn and the worker's own peak (inferred from the
Linux and gVisor sources; not measured here). The lines cannot show a
worker's own growth until it passes that floor.

Nothing here talks to Modal: it reads a saved log.
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import math
import re
import sys
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ARTIFACTS = ("incumbent", "candidate")
#: Exit status for NO-GO; 0 is GO or WAIT. Anything else is a failure of the
#: projection itself, which the watcher must not read as either.
NO_GO_EXIT = 3

_STAMP = r"\[h2h (?P<at>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\] "
_RSS = r": peak_rss=(?P<rss>\d+(?:\.\d+)?) GiB"
_WORKER_INIT = re.compile(
    _STAMP + r"after initializing (?P<artifact>\w+) slice worker" + _RSS
)
_WORKER_SLICE = re.compile(
    _STAMP + r"after scoring (?P<artifact>\w+) chunk (?P<chunk>\d+)/(?P<chunks>\d+) "
    r"slice (?P<slice>\d+)/(?P<slices>\d+)" + _RSS
)
_PARENT = re.compile(
    _STAMP + r"(?P<what>before starting|after receiving|after scoring|after releasing) "
    r"(?P<artifact>\w+)(?P<rest>[^:\[]*)" + _RSS
)
_ATTEMPTS = re.compile(
    r"ATTEMPTS: \d+ earlier unfinished attempt\(s\); "
    r"tool budget for this attempt (?P<seconds>\d+)s"
)
_BUDGET = re.compile(r"BUDGET: the tool may run (?P<seconds>\d+)s")
#: Any scorer boundary line, the pool's or not: a sign of life for the stall.
_ANY_RECORD = re.compile(_STAMP + r"[^:\[]+" + _RSS)


@dataclass(frozen=True)
class Event:
    """One scorer memory line."""

    at: float
    kind: str  # "init", "slice" or "parent"
    artifact: str
    rss_gib: float
    chunk: int | None = None
    chunks: int | None = None
    slice: int | None = None
    slices: int | None = None
    what: str | None = None


@dataclass
class Parsed:
    events: list[Event] = field(default_factory=list)
    #: The tool's budget for this attempt, where the runner printed one.
    budget_seconds: list[int] = field(default_factory=list)
    #: How many of those came from a ``BUDGET:`` line, which the runner
    #: prints just before it starts the tool. An ``ATTEMPTS:`` line comes
    #: earlier, before staging, and its budget can be longer: ``BUDGET:``
    #: cuts it to the time left before the class timeout.
    budget_lines: int = 0
    notes: list[str] = field(default_factory=list)
    #: The time of every scorer boundary line, the pool's or not.
    record_times: list[float] = field(default_factory=list)
    #: Each stream's scorer record times, in its order: a restarted stream
    #: that does not replay earlier lines leaves a hole between them.
    stream_times: list[list[float]] = field(default_factory=list)
    #: Seconds in which a stream began just as the streams before it ended
    #: there, repeating records they showed in that second: a replay from
    #: inside it or the parent printing the same line again, which the count
    #: cannot tell apart, so it may be short there (see ``parse_logs``).
    overlaps: list[float] = field(default_factory=list)


def _epoch(stamp: str) -> float:
    return (
        datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
    )


def _records(
    lines: Iterable[str],
) -> tuple[list[tuple[str, Event]], list[int], int, list[float]]:
    """The scorer records, the runner's budget seconds, how many of those
    were ``BUDGET:`` lines, and the time of every scorer boundary line in one
    stream.

    A line may carry any prefix, and two processes writing at once can put
    two records on one line, so every record on a line counts.
    """

    records: list[tuple[str, Event]] = []
    budgets: list[int] = []
    budget_lines = 0
    times: list[float] = []
    for line in lines:
        budgets.extend(int(m["seconds"]) for m in _ATTEMPTS.finditer(line))
        found_budgets = [int(m["seconds"]) for m in _BUDGET.finditer(line)]
        budgets.extend(found_budgets)
        budget_lines += len(found_budgets)
        times.extend(_epoch(m["at"]) for m in _ANY_RECORD.finditer(line))
        # A worker's "after scoring ... slice s/S" line also fits the parent's
        # "after scoring ..." shape; a span a worker record holds is the worker's.
        found = [
            (kind, match)
            for kind, pattern in (("slice", _WORKER_SLICE), ("init", _WORKER_INIT))
            for match in pattern.finditer(line)
        ]
        taken = [match.span() for _, match in found]
        found += [
            ("parent", match)
            for match in _PARENT.finditer(line)
            if not any(
                match.start() < end and start < match.end() for start, end in taken
            )
        ]
        for kind, match in sorted(found, key=lambda item: item[1].start()):
            common = {
                "at": _epoch(match["at"]),
                "kind": kind,
                "artifact": match["artifact"],
                "rss_gib": float(match["rss"]),
            }
            if kind == "slice":
                event = Event(
                    **common,
                    chunk=int(match["chunk"]),
                    chunks=int(match["chunks"]),
                    slice=int(match["slice"]),
                    slices=int(match["slices"]),
                )
            elif kind == "parent":
                event = Event(**common, what=match["what"])
            else:
                event = Event(**common)
            records.append((match.group(0), event))
    return records, budgets, budget_lines, times


def parse_logs(streams: Sequence[Iterable[str]]) -> Parsed:
    """The scorer's records and the runner's budget lines across ``streams``.

    Each stream is one ``modal app logs`` attempt on the same log; the
    watcher restarts a stream that ends, and a restart may replay lines an
    earlier attempt already saw. Within a stream every record counts (the
    parent prints identical lines within a second), and across streams each
    record counts as many times as the stream showing it most.

    That rule never counts a record twice, whether a restart replays from the
    log's start, from inside the last second the earlier streams reached, or
    not at all. It can count too few: a restart that does not replay, begun
    in the very second the earlier streams end, may print again a line they
    showed in that second (the parent's identical ``after receiving`` lines),
    and the two count once. A replay that began in that second shows the same
    thing, and whether ``modal app logs`` replays, and from where, is untried,
    so the two cannot be told apart; counting both would turn a replay into
    extra worker starts or results and can stop a run that fits (review r6).
    Such a second is reported in ``overlaps``, so a log gap it causes is not
    taken for lines proven lost.
    """

    parsed = Parsed()
    best: dict[str, int] = {}
    events: dict[str, Event] = {}
    seen = 0
    first: float | None = None
    reach: float | None = None
    for stream in streams:
        records, budgets, budget_lines, times = _records(stream)
        parsed.budget_seconds.extend(budgets)
        parsed.budget_lines += budget_lines
        parsed.record_times.extend(times)
        parsed.stream_times.append(sorted(times))
        counts: dict[str, int] = {}
        for text, event in records:
            counts[text] = counts.get(text, 0) + 1
            events[text] = event
        seen += len(records)
        begins = min(times, default=None)
        if (
            begins is not None
            and begins == reach
            and first is not None
            and first < reach
            and any(events[text].at == reach and text in best for text in counts)
        ):
            parsed.overlaps.append(begins)
        for text, count in counts.items():
            best[text] = max(best.get(text, 0), count)
        if times:
            first = begins if first is None else min(first, begins)
            reach = max(times) if reach is None else max(reach, max(times))
    for text, count in best.items():
        parsed.events.extend([events[text]] * count)
    parsed.events.sort(key=lambda event: event.at)
    kept = sum(best.values())
    if seen > kept:
        parsed.notes.append(
            f"{seen - kept} scorer records repeated across {len(streams)} streams "
            "were counted once"
        )
    return parsed


def parse_log(lines: Iterable[str]) -> Parsed:
    """The scorer's records and the runner's budget lines in one stream."""

    return parse_logs([lines])


@dataclass(frozen=True)
class Prior:
    """Seconds per slice by chunk (chunk 1 first), measured off Modal, and the
    candidate's slices per chunk (its rows over the batch size), which its
    log lines give only once its pool has scored a slice."""

    candidate: tuple[float, ...] = ()
    incumbent: tuple[float, ...] = ()
    candidate_slices: int | None = None

    def __post_init__(self) -> None:
        if any(not value > 0 for value in (*self.candidate, *self.incumbent)):
            raise ValueError("prior slice seconds must be positive")
        if (
            self.candidate
            and self.incumbent
            and (len(self.candidate) != len(self.incumbent))
        ):
            raise ValueError("prior slice seconds must cover the same chunks")
        if self.candidate_slices is not None and self.candidate_slices < 1:
            raise ValueError("the candidate needs at least 1 slice per chunk")


@dataclass(frozen=True)
class Limits:
    """What one attempt may use, from the plan."""

    workers: int
    started_at: float
    max_wall_seconds: float
    #: Comparison, scorecards and the tool's exit after the last slice.
    post_scoring_seconds: float = 600.0
    #: NO-GO when the estimated end is past the budget less this fraction.
    #: ``--started-at`` precedes the tool's start and the allowance above is
    #: generous, so a margin inside the budget would stop runs that finish.
    time_margin: float = 0.0
    #: Slices (other than workers' first) a chunk needs before its pace counts.
    min_slices: int = 3
    #: A stall: no scorer record for this many times the slowest estimated
    #: slice, or ``stall_floor_seconds``, whichever is longer. The floor
    #: covers loading the candidate and starting its first worker, which
    #: print nothing after the incumbent's last record.
    stall_factor: float = 4.0
    stall_floor_seconds: float = 3600.0
    prior: Prior = Prior()
    #: The runner prints a BUDGET line before it starts the tool, so streams
    #: with scorer records but no budget line began late (see ``project``).
    budget_line_required: bool = True

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("the estimate needs at least 1 worker")
        if not self.max_wall_seconds > 0:
            raise ValueError("max_wall_seconds must be positive")
        if not 0 <= self.time_margin < 1:
            raise ValueError("time_margin must be in [0, 1)")
        if self.min_slices < 1:
            raise ValueError("min_slices must be at least 1")


@dataclass(frozen=True)
class _Pool:
    #: The parent's ``before starting`` line, if logged.
    opened: float | None
    inits: tuple[float, ...]
    #: The parent's ``after receiving`` lines; the m-th is task m's result.
    receives: tuple[float, ...]
    #: Completion time by task: (chunk - 1) * slices_per_chunk + slice, the
    #: order the scorer submits tasks and its workers take them.
    by_task: Mapping[int, float]
    chunks: int | None
    slices_per_chunk: int | None
    worker_peak: float
    init_peak: float

    @property
    def done(self) -> int:
        return len(self.by_task)

    @property
    def total(self) -> int | None:
        if self.chunks and self.slices_per_chunk:
            return self.chunks * self.slices_per_chunk
        return None


def _pool(events: Sequence[Event], artifact: str) -> _Pool:
    mine = [e for e in events if e.artifact == artifact]
    slices = [e for e in mine if e.kind == "slice"]
    shapes = {(e.chunks, e.slices) for e in slices}
    if len(shapes) > 1:
        raise ValueError(
            f"{artifact}: slice lines disagree on the chunk shape {shapes}"
        )
    chunks, per_chunk = next(iter(shapes)) if shapes else (None, None)
    by_task: dict[int, float] = {}
    for event in slices:
        assert event.chunk is not None and event.slice is not None
        assert per_chunk is not None
        by_task[(event.chunk - 1) * per_chunk + event.slice] = event.at
    parent = [e for e in mine if e.kind == "parent"]
    opened = [e.at for e in parent if e.what == "before starting"]
    return _Pool(
        opened=min(opened) if opened else None,
        inits=tuple(sorted(e.at for e in mine if e.kind == "init")),
        receives=tuple(sorted(e.at for e in parent if e.what == "after receiving")),
        by_task=by_task,
        chunks=chunks,
        slices_per_chunk=per_chunk,
        worker_peak=max(
            (e.rss_gib for e in mine if e.kind in ("init", "slice")), default=0.0
        ),
        init_peak=max((e.rss_gib for e in mine if e.kind == "init"), default=0.0),
    )


#: Tasks in a pool whose slice count is not logged yet.
_UNKNOWN_TOTAL = 10**9


@dataclass
class History:
    """A pool's past under the model, rebuilt from its log up to ``now``."""

    submitted: dict[int, float]
    started: dict[int, float]
    #: Each started worker's first task, in start order.
    firsts: list[int]
    #: Seconds from each logged start's triggering submit to its line.
    start_costs: list[float]
    tokens: int
    consumed: int
    #: The opening submits are not all made yet.
    in_burst: bool
    #: A start is under way, triggered by the submit at this time.
    blocked_since: float | None
    #: Free workers in the order they take tasks, and whether each has had one.
    idle_has_run: list[bool]
    notes: list[str]
    #: Tasks whose result the parent took but whose slice line is missing:
    #: lines the log streams lost.
    gaps: list[int] = field(default_factory=list)
    #: The notes no intact log gives (a start or token the log cannot
    #: supply). The property test that rebuilds intact simulated logs
    #: (test_the_rebuilt_schedule_is_the_simulated_one) finds no note at all;
    #: only the tie note, a start put back on its submit, comes from intact
    #: logs, and it is not among these.
    unexplained: list[str] = field(default_factory=list)


def reconstruct(
    pool: _Pool, workers: int, now: float, total: int | None = None
) -> History | None:
    """The pool's history to ``now`` under the model (see the module).

    ``total`` is the pool's task count when its lines do not show it yet (no
    slice scored); without either, the pool is taken to be unbounded."""

    if pool.opened is None or pool.opened > now:
        return None
    total = pool.total or total or _UNKNOWN_TOTAL
    finished = sorted(t for t in pool.by_task.values() if t <= now)
    inits = [t for t in pool.inits if t <= now]
    receives = [t for t in pool.receives if t <= now]
    notes: list[str] = []
    unexplained: list[str] = []

    def finished_by(at: float) -> int:
        return bisect.bisect_right(finished, at)

    submitted: dict[int, float] = {}
    start_costs: list[float] = []
    taken = spawns = count = 0
    blocked: float | None = None
    opening = min(2 * workers, total)
    clock = pool.opened
    while count < opening:
        count += 1
        submitted[count] = clock
        if finished_by(clock) - taken > 0:
            taken += 1
        elif spawns < workers:
            if spawns < len(inits):
                start_costs.append(inits[spawns] - clock)
                clock = max(clock, inits[spawns])
                spawns += 1
            elif receives:
                unexplained.append(
                    "an opening submit the model says starts a worker did not"
                )
            else:
                blocked = clock
                break
    in_burst = count < opening
    consumed = 0
    previous = clock
    gaps: list[int] = []
    while not in_burst and blocked is None and consumed < total:
        task = consumed + 1
        arrived = pool.by_task.get(task)
        receipt = receives[task - 1] if len(receives) >= task else None
        if arrived is None:
            if receipt is not None:
                # The parent took this result, so its slice ended: its line
                # is missing from the streams.
                gaps.append(task)
            break
        # The parent takes the result once it has arrived and the previous
        # ``after receiving`` line is out.
        ready = max(arrived, previous)
        if ready > now:
            break
        consumed = task
        parent_free = ready
        if 2 * workers + task <= total:
            # A worker start the model has not explained, logged before this
            # submit, came from an earlier submit the model gave a token: that
            # submit took none, so the token is still free.
            resynced = 0
            while (
                spawns < min(workers, len(inits)) and inits[spawns] <= ready and taken
            ):
                spawns += 1
                taken -= 1
                resynced += 1
            if resynced:
                notes.append(
                    f"{resynced} worker start(s) logged before the submit after "
                    f"result {task} came from earlier submits the model gave a "
                    "token; their start costs are not measured"
                )
            # A submit that starts no worker returns at once and the parent
            # prints ``after receiving`` straight after it, so that line dates
            # it (a result can reach the parent well after its slice line).
            received = max(ready, receipt) if receipt is not None else ready
            if finished_by(ready) - taken > 0:
                taken += 1
                submitted[2 * workers + task] = received
            elif spawns < workers:
                submitted[2 * workers + task] = ready
                if spawns < len(inits) and inits[spawns] >= ready:
                    start_costs.append(inits[spawns] - ready)
                    parent_free = inits[spawns]
                    spawns += 1
                elif receipt is not None:
                    unexplained.append(
                        f"the submit after result {task} took no token and started "
                        "no worker the log shows"
                    )
                else:
                    blocked = ready
                    break
            else:
                # The pool is full, so this submit starts no worker either.
                submitted[2 * workers + task] = received
        previous = receipt if receipt is not None else parent_free
    if spawns < len(inits):
        unexplained.append(
            f"the log shows {len(inits)} worker starts and the model explains {spawns}"
        )

    submit_times = sorted(submitted.values())
    explained = inits[:spawns]
    started: dict[int, float] = {}
    begun = 0
    for at in sorted({*submit_times, *finished, *explained}):
        reach = min(
            bisect.bisect_right(submit_times, at),
            finished_by(at) + bisect.bisect_right(explained, at),
        )
        while begun < reach:
            begun += 1
            started[begun] = at
    firsts = [finished_by(at) + k for k, at in enumerate(explained, 1)]
    running = begun - finished_by(now)
    idle = max(0, len(explained) - running)
    arrivals = sorted(
        [(t, 0, True) for t in finished] + [(t, 1, False) for t in explained]
    )
    idle_line = [ran for _, _, ran in arrivals[len(arrivals) - idle :]] if idle else []
    # A worker that arrived by its start has had a task only once its first
    # task has started.
    return History(
        submitted=submitted,
        started=started,
        firsts=firsts,
        start_costs=start_costs,
        tokens=max(0, finished_by(now) - taken),
        consumed=consumed,
        in_burst=in_burst,
        blocked_since=blocked,
        idle_has_run=idle_line,
        notes=notes + unexplained,
        gaps=gaps,
        unexplained=unexplained,
    )


@dataclass(frozen=True)
class Timing:
    """Slice times measured from a rebuilt history."""

    #: Seconds per slice, by chunk, with the number of slices behind each.
    chunk_seconds: dict[int, float]
    chunk_slices: dict[int, int]
    #: Workers' first slices: (chunk, seconds).
    firsts: list[tuple[int, float]]
    start_cost: float | None
    notes: list[str]

    def first_extra(self, seconds: Mapping[int, float]) -> float | None:
        """Mean extra seconds of a worker's first slice over ``seconds``."""

        extras = [max(0.0, d - seconds[c]) for c, d in self.firsts if c in seconds]
        return sum(extras) / len(extras) if extras else None


def _unplaced(pool: _Pool, history: History | None) -> set[int]:
    """Slices that end before the rebuild starts them, or that it never starts.

    On an intact log there are none: the property test that rebuilds intact
    simulated logs (test_the_rebuilt_schedule_is_the_simulated_one) recovers
    every slice's start exactly, and the real three-worker log's slices all
    end after their starts. A lost worker-start line leaves them: the
    rebuild then has one worker too few and dates later slices late.
    """

    if history is None:
        return set()
    return {
        task
        for task, at in pool.by_task.items()
        if (began := history.started.get(task)) is None or at < began
    }


def _start_cost(history: History) -> float | None:
    """The mean of the logged start costs: after a hole, used only once the
    pool has logged a start for every worker (see ``project``)."""

    costs = history.start_costs
    return sum(costs) / len(costs) if costs else None


def timing(
    pool: _Pool,
    history: History | None,
    need: int,
) -> Timing:
    if history is None:
        return Timing({}, {}, [], None, [])
    if not pool.slices_per_chunk:
        # No slice yet: no pace, but the starts logged so far are measured.
        return Timing({}, {}, [], _start_cost(history), list(history.notes))
    per_chunk = pool.slices_per_chunk
    firsts = set(history.firsts)
    regular: dict[int, list[float]] = {}
    first: list[tuple[int, float]] = []
    unplaced = _unplaced(pool, history)
    for task, at in pool.by_task.items():
        if task in unplaced:
            continue
        began = history.started[task]
        chunk = (task - 1) // per_chunk + 1
        if task in firsts:
            first.append((chunk, at - began))
        else:
            regular.setdefault(chunk, []).append(at - began)
    notes = (
        [f"{len(unplaced)} slices end before the model starts them"] if unplaced else []
    )
    seconds = {c: sum(v) / len(v) for c, v in regular.items() if len(v) >= need}
    return Timing(
        chunk_seconds=seconds,
        chunk_slices={c: len(v) for c, v in regular.items() if c in seconds},
        firsts=first,
        start_cost=_start_cost(history),
        notes=notes + history.notes,
    )


def replay(
    *,
    now: float,
    workers: int,
    total: int,
    per_chunk: int,
    seconds: Mapping[int, float],
    start_cost: float,
    first_extra: float,
    history: History | None = None,
    finished: Mapping[int, float] | None = None,
    notes: list[str] | None = None,
) -> tuple[float, float]:
    """The model run forward from ``history`` at ``now`` (a fresh pool opening
    at ``now`` without one): when its last slice ends and when the parent is
    done with it.

    A running slice past its expected end is taken to run as far past it
    again: assuming it ends now would hide the work it still holds."""

    finished_at = dict(finished or {})
    clock, last = now, max(finished_at.values(), default=now)
    heap: list[tuple[float, int, int]] = []
    idle: deque[int] = deque()
    queue: deque[int] = deque()
    has_run: set[int] = set()
    worker = 0

    def chunk_of(task: int) -> int:
        return (task - 1) // per_chunk + 1

    if history is None:
        submitted = consumed = tokens = spawns = 0
        in_burst, blocked = True, None
    else:
        submitted, consumed = len(history.submitted), history.consumed
        tokens, spawns = history.tokens, len(history.firsts)
        in_burst, blocked = history.in_burst, history.blocked_since
        firsts = set(history.firsts)
        overdue = 0
        for task in sorted(history.started):
            if task in finished_at:
                continue
            worker += 1
            has_run.add(worker)
            end = history.started[task] + seconds[chunk_of(task)]
            end += first_extra if task in firsts else 0.0
            if end < now:
                overdue += 1
                end = 2 * now - end
            heapq.heappush(heap, (end, task, worker))
        if overdue and notes is not None:
            notes.append(
                f"{overdue} running slice(s) are past their chunk's pace; each is "
                "taken to run as far past it again"
            )
        queue.extend(t for t in sorted(history.submitted) if t not in history.started)
        for ran in history.idle_has_run:
            worker += 1
            idle.append(worker)
            if ran:
                has_run.add(worker)

    def dispatch() -> None:
        while idle and queue:
            who, task = idle.popleft(), queue.popleft()
            took = seconds[chunk_of(task)] + (0.0 if who in has_run else first_extra)
            has_run.add(who)
            heapq.heappush(heap, (clock + took, task, who))

    def advance(to: float) -> None:
        nonlocal clock, tokens, last
        while heap and heap[0][0] <= to:
            at, task, who = heapq.heappop(heap)
            clock = at
            finished_at[task] = at
            tokens += 1
            last = max(last, at)
            idle.append(who)
            dispatch()
        clock = max(clock, to)

    def add_worker() -> None:
        nonlocal spawns, worker
        spawns += 1
        worker += 1
        idle.append(worker)
        dispatch()

    def submit() -> None:
        nonlocal submitted, tokens
        if submitted >= total:
            return
        submitted += 1
        queue.append(submitted)
        dispatch()
        if tokens:
            tokens -= 1
        elif spawns < workers:
            advance(clock + start_cost)
            add_worker()

    dispatch()
    if blocked is not None:
        advance(max(clock, blocked + start_cost))
        add_worker()
    if in_burst:
        while submitted < min(2 * workers, total):
            submit()
    while consumed < total:
        head = consumed + 1
        while head not in finished_at:
            if not heap:
                raise ValueError("the replay ran out of work before the last slice")
            advance(heap[0][0])
        consumed = head
        submit()
    return last, clock


def _filled(
    measured: Mapping[int, float],
    chunks: int,
    prior: Sequence[float],
    pooled: bool = False,
) -> tuple[dict[int, float], list[str]]:
    """Every chunk's seconds per slice: measured, or scaled from a measured one.

    By default an unmeasured chunk takes the slowest result of scaling each
    measured chunk by the prior. ``pooled`` scales by one ratio over all the
    measured chunks instead: for a prior that already carries the other
    artifact's departures, which this pool may not share, the slowest result
    would compound them."""

    notes: list[str] = []
    seconds = dict(measured)
    missing = [c for c in range(1, chunks + 1) if c not in measured]
    if not missing:
        return seconds, notes
    if prior and len(prior) == chunks and pooled:
        ratio = sum(measured.values()) / sum(prior[m - 1] for m in measured)
        for chunk in missing:
            seconds[chunk] = prior[chunk - 1] * ratio
        notes.append(
            f"chunks {missing} not measured yet: scaled from the prior by one "
            "ratio pooled over the measured chunks"
        )
    elif prior and len(prior) == chunks:
        for chunk in missing:
            seconds[chunk] = max(
                measured[m] * prior[chunk - 1] / prior[m - 1] for m in measured
            )
        notes.append(
            f"chunks {missing} not measured yet: scaled from measured chunks by "
            "the prior per-chunk times (the slowest result)"
        )
    else:
        slowest = max(measured.values())
        for chunk in missing:
            seconds[chunk] = slowest
        notes.append(
            f"chunks {missing} not measured yet and no prior for {chunks} chunks: "
            "each takes the slowest measured chunk's time, which a slower later "
            "chunk makes optimistic"
        )
    return seconds, notes


def _stream_holes(
    streams: Sequence[Sequence[float]], now: float
) -> list[tuple[float, float]]:
    """Stretches no stream covers between the first record and the last.

    A restarted stream that replays what the app printed covers the earlier
    stream's span; one that does not starts after it, and any line printed in
    between is lost unless another stream holds it. A hole is not proof of a
    lost line (the scorer can be silent for minutes)."""

    spans = sorted(
        (kept[0], kept[-1])
        for times in streams
        if (kept := [t for t in times if t <= now])
    )
    holes: list[tuple[float, float]] = []
    reach = None
    for first, last in spans:
        if reach is not None and first > reach:
            holes.append((reach, first))
        reach = last if reach is None else max(reach, last)
    return holes


def project(parsed: Parsed, limits: Limits, now: float) -> dict[str, object]:
    """The verdict at ``now`` and the measurements behind it."""

    events = [e for e in parsed.events if e.at <= now]
    incumbent, candidate = (_pool(events, name) for name in ARTIFACTS)
    workers = limits.workers
    budget = min([limits.max_wall_seconds, *parsed.budget_seconds])
    deadline = limits.started_at + budget
    parent_peak = max((e.rss_gib for e in events if e.kind == "parent"), default=0.0)
    report: dict[str, object] = {
        "workers": workers,
        "now_utc": _utc(now),
        "budget_seconds": budget,
        "deadline_utc": _utc(deadline),
        "measured": {
            name: {
                "worker_starts": len(pool.inits),
                "slices_scored": pool.done,
                "slices_total": pool.total,
            }
            for name, pool in (("incumbent", incumbent), ("candidate", candidate))
        },
        "memory_reported_not_judged": {
            "parent_peak_gib": parent_peak,
            "incumbent_worker_peak_gib": incumbent.worker_peak,
            "candidate_worker_peak_gib": candidate.worker_peak,
            "candidate_worker_init_peak_gib": candidate.init_peak,
        },
    }
    early = sum(1 for t in parsed.record_times if t < limits.started_at)
    if not early and candidate.total is not None and candidate.done >= candidate.total:
        # Every slice is scored; stopping now would only throw the scores away.
        return {
            **report,
            "verdict": "GO",
            "reasons": [
                "the candidate has scored every slice; the comparison and "
                "scorecards remain, and only the runner's own budget stops them"
            ],
        }
    waiting: list[str] = []
    if early:
        # A relaunch into the same run folder overwrites started-at, so the
        # earlier attempt's streams would be read as this attempt's.
        report["records_before_start"] = early
        waiting.append(
            f"{early} scorer records predate --started-at: the streams hold an "
            "earlier attempt's log; give each attempt its own run folder"
        )
    if len(candidate.inits) > workers or len(incumbent.inits) > workers:
        waiting.append(
            f"{len(incumbent.inits)} and {len(candidate.inits)} worker starts for "
            f"{workers} workers: a restarted container or recycling, which the "
            "estimate does not model"
        )
    pools = (("incumbent", incumbent), ("candidate", candidate))
    histories = {name: reconstruct(pool, workers, now) for name, pool in pools}
    gaps = {
        name: history.gaps
        for name, history in histories.items()
        if history is not None and history.gaps
    }
    log_gap: dict[str, object] = {name: tasks[:5] for name, tasks in gaps.items()}
    for name, pool in pools:
        if pool.opened is None and (pool.inits or pool.receives or pool.by_task):
            # Without the opening line the pool cannot be rebuilt, and the
            # estimate would replay all of it from now.
            log_gap[name] = "no opening line"
            waiting.append(
                f"the streams lost the {name}'s opening line (its worker, slice "
                "or receive lines are there), so its pool cannot be rebuilt; "
                "the early stop is off and the runner's budget bounds the run"
            )
    if gaps:
        waiting.append(
            "the streams lost lines: the parent took results whose slice lines "
            f"are missing ({', '.join(f'{n} task {t[0]}' for n, t in gaps.items())}), "
            "so the rebuilt pool is wrong in an unknown direction; the early stop "
            "is off and the runner's budget bounds the run"
        )
    for name, pool in pools:
        # A lost line the parent's later lines cannot reveal, such as a worker
        # start in a stretch a restarted stream did not replay, still shows:
        # the rebuild has a worker too few, so it starts later slices after
        # they end (review r5) or meets a submit it cannot explain.
        history = histories[name]
        # More starts than workers already waits, as a restart or recycling.
        if history is None or name in log_gap or len(pool.inits) > workers:
            continue
        unplaced = len(_unplaced(pool, history))
        misfits = [
            *(
                [f"{unplaced} slices end before the rebuild starts them"]
                * bool(unplaced)
            ),
            *history.unexplained,
        ]
        if misfits:
            # A lost start leaves one unexplained submit per later result.
            shown = "; ".join(misfits[:3]) + (
                f"; and {len(misfits) - 3} more" if len(misfits) > 3 else ""
            )
            log_gap[name] = shown
            waiting.append(
                f"the {name}'s rebuilt pool does not fit its log ({shown}), "
                "which no intact simulated log gives: a line it dates starts by (a "
                "worker start, say) is likely missing, so the rebuilt pool is wrong "
                "in an unknown direction; the early stop is off and the runner's "
                "budget bounds the run"
            )
    # The runner's BUDGET line comes just before the tool's first output, so a
    # log with scorer records but no BUDGET line began late or lost it, and
    # may have missed the line that shortens the deadline most. An ATTEMPTS:
    # line comes before staging and its budget can be longer, so it does not
    # stand in. The BUDGET line only shortens the deadline, so NO-GO against
    # the shortest budget seen still holds; GO does not.
    budget_unseen = (
        limits.budget_line_required
        and not parsed.budget_lines
        and (bool(events) or any(t <= now for t in parsed.record_times))
    )
    budget_unseen_reason = (
        "the streams hold scorer records but not the runner's BUDGET line, "
        "printed just before the tool starts: they began late or lost it, and "
        "it can be shorter than the plan's max_wall_seconds or an ATTEMPTS: "
        "line, so the verdict is WAIT rather than GO; NO-GO against the "
        "shortest budget seen still holds, since the BUDGET line only shortens it"
    )
    if budget_unseen:
        log_gap["runner"] = "no BUDGET line"
    if log_gap:
        report["log_gap"] = log_gap
    overlaps = [at for at in parsed.overlaps if at <= now]
    if overlaps:
        report["restart_overlaps"] = [_utc(at) for at in overlaps]
        if log_gap.keys() & {"incumbent", "candidate"}:
            waiting.append(
                "a log stream began in the second the one before it ended and "
                f"repeated records from it ({', '.join(map(_utc, overlaps))}): a "
                "replay, or the parent printing the same line again, which the "
                "count cannot tell apart, so it counts them once and that second "
                "may be short; the gap may come from that rather than from lines "
                "the streams lost"
            )
    holes = _stream_holes(parsed.stream_times, now)
    if holes:
        report["stream_holes"] = [
            {"from_utc": _utc(start), "to_utc": _utc(end)} for start, end in holes
        ]
    # A start line lost in a hole leaves the rebuild a worker short and its
    # start cost off, which no later line reveals until the model needs the
    # lost worker, and it reads late by more than a start when the lost start
    # was a cheap one (review r6). Each worker prints its start line once, so
    # a pool that has logged a start for every worker has lost none: until
    # then, after a hole that could hold the pool's start lines (it ends once
    # the pool has opened and begins before its release line), a time NO-GO
    # waits. A released pool's starts still feed the estimate only as the
    # incumbent's start cost, until the candidate logs a start of its own.
    released = {
        name: min(
            (
                e.at
                for e in events
                if e.artifact == name
                and e.kind == "parent"
                and e.what == "after releasing"
            ),
            default=None,
        )
        for name, _ in pools
    }
    unconfirmed = {
        name: len(pool.inits)
        for name, pool in pools
        if pool.opened is not None
        and len(pool.inits) < workers
        and (released[name] is None or (name == "incumbent" and not candidate.inits))
        and any(
            end >= pool.opened and (released[name] is None or start < released[name])
            for start, end in holes
        )
    }
    if unconfirmed:
        report["starts_unconfirmed"] = {
            name: f"{logged} of {workers}" for name, logged in unconfirmed.items()
        }
    # The newest stream (streams come in the watcher's order) holding nothing
    # yet while an earlier one does: a restart that has delivered nothing, so
    # whatever the app printed since the last record may be lost. It is not a
    # hole until the stream's first record shows where it ends.
    kept = [[t for t in times if t <= now] for times in parsed.stream_times]
    reopened = len(kept) > 1 and not kept[-1] and any(kept[:-1])
    if reopened:
        # Whatever the app printed since the last record may be lost, and the
        # rebuild cannot tell a quiet stretch (a worker start, the candidate
        # loading, the pool opening) from lost lines, so the estimate waits
        # for the stream's first record whether or not a slice is running
        # (review r6). The stall check waits too: silence here is the
        # stream's, not the app's.
        report["newest_stream_empty"] = True
        waiting.append(
            "the newest log stream, restarted after the last one ended, has "
            "delivered nothing yet: what the app printed since the last record "
            "may be lost, so the estimate and the stall check wait for its "
            "first record; the runner's budget bounds the run meanwhile"
        )
    hole_end = holes[-1][1] if holes else None
    if hole_end is not None:
        # A slice the rebuild has running that began before a hole ended may
        # have ended inside it, its line lost. Until the parent takes that
        # result (a log gap) it reads as a straggler, which the replay takes
        # to run as far past its pace again: a false NO-GO. A slice that is
        # really running clears when its line arrives.
        suspects: dict[str, list[int]] = {}
        for name, pool in pools:
            history = histories[name]
            if history is None:
                continue
            tasks = sorted(
                task
                for task, began in history.started.items()
                if began < hole_end and task not in pool.by_task
            )
            if tasks:
                suspects[name] = tasks
        if suspects:
            report["hole_suspects"] = {name: t[:5] for name, t in suspects.items()}
            waiting.append(
                "a log stream restarted without replaying (a hole), and "
                + ", ".join(
                    f"{len(t)} of the {name}'s slices" for name, t in suspects.items()
                )
                + " began before it ended and have no line yet: one that ended in "
                "the hole has lost its line and would read as a straggler, so the "
                "estimate waits until each line arrives or the parent takes that "
                "result (a log gap); the runner's budget bounds the run meanwhile"
            )
    if waiting:
        # No estimate, but what the rebuild noticed still helps read the gap.
        estimate, why = None, None
        notes = [
            f"{name}: {note}"
            for name, history in histories.items()
            if history is not None
            for note in history.notes
        ]
    else:
        estimate, notes, why = _estimate(incumbent, candidate, limits, now)
    if budget_unseen:
        waiting.append(budget_unseen_reason)
    if estimate is None and why:
        waiting.append(why)
    slowest = max((estimate or {}).get("chunk_seconds_used", {}).values(), default=0.0)
    seen = [t for t in parsed.record_times if t <= now] + [e.at for e in events]
    stall = None if reopened else _stall(seen, candidate, slowest, limits, now)
    if estimate is None:
        if stall:
            return {
                **report,
                "verdict": "NO-GO",
                "reasons": [stall],
                "waiting": waiting,
                "estimator_notes": notes,
            }
        return {
            **report,
            "verdict": "WAIT",
            "reasons": waiting,
            "estimator_notes": notes,
        }
    end = estimate.pop("end")
    time_line = deadline - limits.time_margin * budget
    reasons: list[str] = []
    held: list[str] = []
    if end > time_line and unconfirmed:
        held.append(
            f"time: estimated end {_utc(end)} is past {_utc(time_line)} by "
            f"{end - time_line:.0f} s, but a log stream restarted without "
            "replaying after "
            + " and ".join(
                f"the {name}'s pool opened, which has logged {logged} of its "
                f"{workers} worker starts"
                for name, logged in unconfirmed.items()
            )
            + ": a start line lost in the hole would make the estimate read "
            "late, and only a start for every worker shows none was lost, so it "
            "waits until then; a stall and the runner's budget still stop the run"
        )
    elif end > time_line:
        reasons.append(
            f"time: estimated end {_utc(end)} is past {_utc(time_line)} "
            f"({limits.time_margin:.0%} inside the {budget:.0f} s budget ending "
            f"{_utc(deadline)})"
        )
    if stall:
        reasons.append(stall)
    verdict = "NO-GO" if reasons else "WAIT" if held or budget_unseen else "GO"
    return {
        **report,
        "verdict": verdict,
        "reasons": reasons
        or [*held, *([budget_unseen_reason] if budget_unseen else [])],
        "estimator_notes": notes,
        "estimate": {
            **estimate,
            "chunk_seconds_used": {
                str(c): round(s, 1) for c, s in estimate["chunk_seconds_used"].items()
            },
            "estimated_end_utc": _utc(end),
            "estimated_tool_seconds_from_start": round(end - limits.started_at, 1),
        },
    }


def _estimate(
    incumbent: _Pool,
    candidate: _Pool,
    limits: Limits,
    now: float,
) -> tuple[dict | None, list[str], str | None]:
    """The estimated end and its inputs, or None and why not yet."""

    workers, prior, need = limits.workers, limits.prior, limits.min_slices
    histories = {
        "incumbent": reconstruct(incumbent, workers, now),
        "candidate": reconstruct(candidate, workers, now),
    }
    timings = {
        "incumbent": timing(incumbent, histories["incumbent"], need),
        "candidate": timing(candidate, histories["candidate"], need),
    }
    notes = [
        f"{name}: {note}" for name, found in timings.items() for note in found.notes
    ]
    measured = {name: found.chunk_seconds for name, found in timings.items()}
    inc_cost = timings["incumbent"].start_cost
    cand_cost = timings["candidate"].start_cost or inc_cost
    if measured["candidate"]:
        assert candidate.chunks is not None
        fill_prior: Sequence[float] = prior.candidate
        carried = [
            c
            for c in measured["incumbent"]
            if len(prior.candidate) == len(prior.incumbent) == candidate.chunks
        ]
        if carried:
            # The incumbent's measured chunks say how this pool departs from
            # the prior chunk by chunk; carry that into the candidate's fill.
            fill_prior = tuple(
                prior.candidate[c - 1]
                * (
                    measured["incumbent"][c] / prior.incumbent[c - 1]
                    if c in measured["incumbent"]
                    else 1.0
                )
                for c in range(1, candidate.chunks + 1)
            )
        seconds, more = _filled(
            measured["candidate"], candidate.chunks, fill_prior, pooled=bool(carried)
        )
        if carried and len(seconds) > len(measured["candidate"]):
            more.append(
                f"the candidate's fill carries the incumbent's measured chunks "
                f"{sorted(carried)} against the prior"
            )
        pace_from = "candidate"
    elif measured["incumbent"] and prior.candidate and prior.incumbent:
        assert incumbent.chunks is not None
        if len(prior.candidate) != incumbent.chunks:
            return None, notes, "the prior does not cover the registry's chunks"
        base, more = _filled(measured["incumbent"], incumbent.chunks, prior.incumbent)
        seconds = {
            c: base[c] * prior.candidate[c - 1] / prior.incumbent[c - 1] for c in base
        }
        more.append("candidate pace: the incumbent's, times the prior ratio")
        pace_from = "incumbent x prior"
    else:
        return (
            None,
            notes,
            f"no pace yet: a chunk is measured once {need} of its slices other "
            "than workers' first have ended (the incumbent's counts for the "
            "candidate only with a prior for both)",
        )
    notes.extend(more)
    chunks = len(seconds)
    if measured["incumbent"]:
        inc_seconds, inc_notes = _filled(measured["incumbent"], chunks, prior.incumbent)
    else:
        inc_seconds, inc_notes = {}, []
    inc_extra = timings["incumbent"].first_extra(inc_seconds)
    cand_extra = timings["candidate"].first_extra(seconds)
    cand_extra = inc_extra if cand_extra is None else cand_extra
    # Both artifacts share the registry's chunks; the candidate's slices per
    # chunk come from its own lines once one prints, else from the prior.
    cand_slices = candidate.slices_per_chunk or prior.candidate_slices
    if cand_slices is None:
        return (
            None,
            notes,
            "the candidate's slices per chunk are not known before its pool "
            "scores a slice, and the prior gives none",
        )
    cand_history = histories["candidate"]
    if cand_history is not None and candidate.total is None:
        # Rebuilt as unbounded; a pool of fewer than 2 x workers tasks makes
        # fewer opening submits, so rebuild it at its size.
        cand_history = reconstruct(candidate, workers, now, chunks * cand_slices)
    if cand_history is not None:
        if len(cand_history.firsts) < workers and cand_cost is None:
            return None, notes, "no start cost yet for the workers still to start"
        last, _ = replay(
            now=now,
            workers=workers,
            total=chunks * cand_slices,
            per_chunk=cand_slices,
            seconds=seconds,
            start_cost=cand_cost or 0.0,
            first_extra=cand_extra or 0.0,
            history=cand_history,
            finished={t: at for t, at in candidate.by_task.items() if at <= now},
            notes=notes,
        )
    else:
        # The incumbent is still scoring (or its pool is done and the
        # candidate is loading): its replay, then the candidate's whole pool.
        inc_history = histories["incumbent"]
        if inc_history is None or not incumbent.slices_per_chunk or inc_cost is None:
            return None, notes, "no start cost or incumbent shape yet"
        notes.extend(f"incumbent: {note}" for note in inc_notes)
        _, released = replay(
            now=now,
            workers=workers,
            total=chunks * incumbent.slices_per_chunk,
            per_chunk=incumbent.slices_per_chunk,
            seconds=inc_seconds,
            start_cost=inc_cost,
            first_extra=inc_extra or 0.0,
            history=inc_history,
            finished={t: at for t, at in incumbent.by_task.items() if at <= now},
            notes=notes,
        )
        last, _ = replay(
            now=released,
            workers=workers,
            total=chunks * cand_slices,
            per_chunk=cand_slices,
            seconds=seconds,
            start_cost=cand_cost or inc_cost,
            first_extra=cand_extra or 0.0,
        )
    return (
        {
            "end": last + limits.post_scoring_seconds,
            "pace_from": pace_from,
            "measured_chunk_seconds": {
                name: {str(c): round(s, 1) for c, s in values.items()}
                for name, values in measured.items()
            },
            "measured_chunk_slices": {
                name: {str(c): n for c, n in found.chunk_slices.items()}
                for name, found in timings.items()
            },
            "chunk_seconds_used": seconds,
            "start_seconds": {"incumbent": inc_cost, "candidate": cand_cost},
            "first_slice_extra_seconds": {
                "incumbent": inc_extra,
                "candidate": cand_extra,
            },
            "candidate_slices_per_chunk": cand_slices,
        },
        notes,
        None,
    )


def _stall(
    seen: Sequence[float],
    candidate: _Pool,
    slowest: float,
    limits: Limits,
    now: float,
) -> str | None:
    """Why the run looks stuck, or None.

    Until the candidate has scored every slice, some scorer record should
    appear well within the allowance: the floor (which covers staging,
    loading a frame and starting a pool's first worker) or a multiple of the
    slowest estimated slice, whichever is longer. Before any record the
    allowance runs from ``--started-at``, so a run whose log never arrives
    (a hang before the first slice, or streams that fail) is not watched as
    WAIT forever.
    """

    if candidate.total is not None and candidate.done >= candidate.total:
        return None
    last = max(seen, default=limits.started_at)
    allowed = max(limits.stall_floor_seconds, limits.stall_factor * slowest)
    if now - last <= allowed:
        return None
    since = (
        f"since {_utc(last)}"
        if seen
        else f"at all since the launch at {_utc(last)} (a log stream that "
        "delivers nothing reads the same; go-no-go.log notes each restart)"
    )
    return f"stall: no scorer record {since}, {now - last:.0f} s against {allowed:.0f} s allowed"


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(math.floor(epoch), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def limits_from_plan(
    plan_path: Path, started_at: float, prior: Prior | None = None
) -> Limits:
    """The plan's workers and wall budget."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import modal_us_stage_plan as plan_lib

    _, plan = plan_lib.load_plan(plan_path)
    if plan.tool.name != "us-release-head-to-head":
        raise SystemExit(f"{plan_path}: not a us-release-head-to-head plan")
    if "worker_max_slices" in plan.options:
        raise SystemExit(
            f"{plan_path}: sets worker_max_slices; the estimate assumes each "
            "worker lives for its whole pool"
        )
    workers = plan.options.get("workers")
    if not isinstance(workers, int) or workers < 1:
        raise SystemExit(f"{plan_path}: the estimate needs workers >= 1")
    if plan.max_wall_seconds is None:
        raise SystemExit(f"{plan_path}: sets no max_wall_seconds")
    return Limits(
        workers=workers,
        started_at=started_at,
        max_wall_seconds=plan.max_wall_seconds,
        prior=prior or Prior(),
    )


def _prior(path: Path | None) -> Prior:
    if path is None:
        return Prior()
    data = json.loads(path.read_text())
    slices = data.get("slices_per_chunk", {}).get("candidate")
    return Prior(
        candidate=tuple(float(v) for v in data.get("candidate_mean_slice_seconds", ())),
        incumbent=tuple(float(v) for v in data.get("incumbent_mean_slice_seconds", ())),
        candidate_slices=None if slices is None else int(slices),
    )


def _stream_order(paths: Sequence[Path]) -> list[Path]:
    """The watcher's ``app-N.log`` streams by N (a shell glob puts app-10
    before app-2); other names keep the order given."""

    if all(re.fullmatch(r"app-\d+", path.stem) for path in paths):
        return sorted(paths, key=lambda path: int(path.stem.removeprefix("app-")))
    return list(paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "logs",
        type=Path,
        nargs="+",
        help="saved `modal app logs` streams of one run (one per attempt)",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--started-at",
        required=True,
        help="UTC launch time, YYYY-MM-DDTHH:MM:SSZ; it precedes the tool's "
        "start, so the deadline comes early: no overspend, but a long wait "
        "before the tool starts can stop a run that would fit",
    )
    parser.add_argument(
        "--chunk-weights",
        type=Path,
        help="JSON with candidate_mean_slice_seconds and "
        "incumbent_mean_slice_seconds, one per chunk",
    )
    parser.add_argument(
        "--now", help="UTC time to judge at (default: the clock); for replays"
    )
    args = parser.parse_args(argv)
    limits = limits_from_plan(
        args.plan, _epoch(args.started_at), _prior(args.chunk_weights)
    )
    parsed = parse_logs(
        [
            path.read_text(errors="replace").splitlines()
            for path in _stream_order(args.logs)
        ]
    )
    now = _epoch(args.now) if args.now else time.time()
    report = project(parsed, limits, now)
    print(json.dumps({**report, "notes": parsed.notes}, indent=2))
    return NO_GO_EXIT if report["verdict"] == "NO-GO" else 0


if __name__ == "__main__":
    raise SystemExit(main())
