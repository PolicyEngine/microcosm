"""Project a running US head-to-head from its own log: the d844 go/no-go.

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

Every line carries a UTC timestamp and the process's own peak RSS
(``ru_maxrss``). The scorer scores the incumbent in one pool of workers,
then the candidate in a fresh pool, and both artifacts have the same number
of slices in every registry chunk. That lets the projection scale the
incumbent's whole run by the candidate's measured speed:

* ``d1`` for each artifact is the mean time from a worker's initialization
  to its first chunk-1 slice. Workers start one at a time, so the log pairs
  only the workers that initialized before the first slice finished: their
  first slices are the earliest completions, in initialization order, while
  a later worker's first slice can finish after an early worker's second;
* the incumbent's total slice work is the sum of its last ``workers``
  completions less the sum of its initializations: once the task queue
  empties, each worker finishes exactly one more slice, so those are each
  worker's last completion, and a worker given tasks back to back is busy
  from its initialization to its last completion (any idle time a real
  pool adds counts as work, which only makes the projection later);
* dividing by the slices per chunk gives the per-slice durations summed
  over chunks; the candidate's sum is that times ``d1(candidate) /
  d1(incumbent)``;
* the candidate ends no later than its mean initialization plus its total
  work over the workers plus its longest slice (Graham's bound for list
  scheduling); the longest slice is bounded by the candidate's per-chunk sum,
  and a fixed allowance covers the scorecards.

Memory projects each candidate worker's peak as the highest peak seen plus
the steeper of the two artifacts' per-slice growth for the slices a worker
still has to score, and adds the parent's peak. Both projections are
conservative where the log cannot attribute a line to a worker.

The verdict is ``WAIT`` until the incumbent has finished and the
candidate's pool has initialized and scored a chunk-1 slice per worker
(both artifacts need at least as many slices per chunk as workers), then
``GO`` or ``NO-GO``. A ``NO-GO`` names each
limit the run would cross; the runbook's watcher stops the app on it.
Nothing here talks to Modal: it reads a saved log.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ARTIFACTS = ("incumbent", "candidate")
#: The scorer's per-process limit (MAX_RSS_BYTES in the scorer): 20 GiB.
SCORER_PROCESS_LIMIT_GIB = 20.0
#: The scorer prints peaks to two decimals of a GiB.
RSS_RESOLUTION_GIB = 0.01

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
    r"(?P<artifact>\w+)(?P<rest>[^:]*)" + _RSS
)


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


def _epoch(stamp: str) -> float:
    return (
        datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
    )


def parse_events(lines: Iterable[str]) -> tuple[list[Event], list[str]]:
    """The scorer's memory lines in ``lines``, in order, and notes on oddities.

    A line may carry any prefix (a log timestamp, a container id). A worker
    slice line seen again is a duplicated log and is dropped with a note.
    """

    events: list[Event] = []
    notes: list[str] = []
    seen_slices: set[tuple[str, int, int]] = set()
    duplicates = 0
    for line in lines:
        match = _WORKER_SLICE.search(line)
        if match:
            key = (match["artifact"], int(match["chunk"]), int(match["slice"]))
            if key in seen_slices:
                duplicates += 1
                continue
            seen_slices.add(key)
            events.append(
                Event(
                    at=_epoch(match["at"]),
                    kind="slice",
                    artifact=match["artifact"],
                    rss_gib=float(match["rss"]),
                    chunk=int(match["chunk"]),
                    chunks=int(match["chunks"]),
                    slice=int(match["slice"]),
                    slices=int(match["slices"]),
                )
            )
            continue
        match = _WORKER_INIT.search(line)
        if match:
            events.append(
                Event(
                    at=_epoch(match["at"]),
                    kind="init",
                    artifact=match["artifact"],
                    rss_gib=float(match["rss"]),
                )
            )
            continue
        match = _PARENT.search(line)
        if match:
            events.append(
                Event(
                    at=_epoch(match["at"]),
                    kind="parent",
                    artifact=match["artifact"],
                    rss_gib=float(match["rss"]),
                    what=match["what"],
                )
            )
    if duplicates:
        notes.append(f"dropped {duplicates} repeated worker slice lines")
    return events, notes


@dataclass(frozen=True)
class Limits:
    """What one attempt may use, from the plan and its class."""

    workers: int
    started_at: float
    max_wall_seconds: float
    memory_limit_gib: float
    process_limit_gib: float = SCORER_PROCESS_LIMIT_GIB
    #: Comparison, scorecards and the tool's exit after the last slice.
    post_scoring_seconds: float = 600.0
    #: GO needs the projected end this fraction inside the deadline.
    time_margin: float = 0.05
    #: GO needs projected memory this fraction inside each limit.
    memory_margin: float = 0.10

    def __post_init__(self) -> None:
        if self.workers < 2:
            raise ValueError("the projection needs a pool of at least 2 workers")
        for name in ("max_wall_seconds", "memory_limit_gib", "process_limit_gib"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        for name in ("time_margin", "memory_margin"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")


@dataclass(frozen=True)
class _Pool:
    inits: tuple[float, ...]
    init_peak: float
    slice_times: tuple[float, ...]
    chunk1_times: tuple[float, ...]
    slice_peak: float
    slices_seen: int
    chunks: int | None
    slices_per_chunk: int | None
    parent_peak: float


def _pool(events: Sequence[Event], artifact: str) -> _Pool:
    inits = sorted(e.at for e in events if e.kind == "init" and e.artifact == artifact)
    init_peaks = [
        e.rss_gib for e in events if e.kind == "init" and e.artifact == artifact
    ]
    slices = [e for e in events if e.kind == "slice" and e.artifact == artifact]
    shapes = {(e.chunks, e.slices) for e in slices}
    if len(shapes) > 1:
        raise ValueError(
            f"{artifact}: slice lines disagree on the chunk shape {shapes}"
        )
    chunks, per_chunk = next(iter(shapes)) if shapes else (None, None)
    parent = [e.rss_gib for e in events if e.kind == "parent"]
    return _Pool(
        inits=tuple(inits),
        init_peak=max(init_peaks, default=0.0),
        slice_times=tuple(sorted(e.at for e in slices)),
        chunk1_times=tuple(sorted(e.at for e in slices if e.chunk == 1)),
        slice_peak=max((e.rss_gib for e in slices), default=0.0),
        slices_seen=len(slices),
        chunks=chunks,
        slices_per_chunk=per_chunk,
        parent_peak=max(parent, default=0.0),
    )


def _first_slice_seconds(pool: _Pool, workers: int) -> tuple[float, int] | None:
    """Mean seconds from initialization to a worker's first chunk-1 slice.

    Pairs the workers that initialized by the first chunk-1 completion with
    the earliest completions, in order; returns the mean and the pair count,
    or None until every worker has initialized and those slices are in.
    """

    if len(pool.inits) < workers or not pool.chunk1_times:
        return None
    pairs = sum(1 for start in pool.inits if start < pool.chunk1_times[0])
    if not pairs or len(pool.chunk1_times) < pairs:
        return None
    total = sum(
        done - start
        for start, done in zip(
            pool.inits[:pairs], pool.chunk1_times[:pairs], strict=True
        )
    )
    return total / pairs, pairs


def _growth_per_slice(pool: _Pool, workers: int) -> float:
    """Highest peak above the initialization peak, per slice a worker scored.

    The log cannot say which worker printed a line, so the growth divides by
    the mean slices per worker: an upper bound on any one worker's slope when
    peaks grow evenly with slices. The difference widens by one print
    resolution, since each printed peak may sit up to half of it from the
    real one, so growth too small to show yet still counts.
    """

    if not pool.slices_seen:
        return 0.0
    rise = max(0.0, pool.slice_peak - pool.init_peak) + RSS_RESOLUTION_GIB
    return rise / max(1.0, pool.slices_seen / workers)


def project(events: Sequence[Event], limits: Limits) -> dict[str, object]:
    """The go/no-go verdict and the measurements behind it."""

    incumbent, candidate = (_pool(events, name) for name in ARTIFACTS)
    workers = limits.workers
    deadline = limits.started_at + limits.max_wall_seconds
    report: dict[str, object] = {
        "workers": workers,
        "deadline_utc": _utc(deadline),
        "measured": {
            name: {
                "worker_inits": len(pool.inits),
                "slices_scored": pool.slices_seen,
                "slices_total": (
                    pool.chunks * pool.slices_per_chunk
                    if pool.chunks and pool.slices_per_chunk
                    else None
                ),
                "worker_init_peak_gib": pool.init_peak,
                "worker_slice_peak_gib": pool.slice_peak,
            }
            for name, pool in (("incumbent", incumbent), ("candidate", candidate))
        },
        "parent_peak_gib": max(incumbent.parent_peak, candidate.parent_peak),
    }
    waiting: list[str] = []
    for name, pool in (("incumbent", incumbent), ("candidate", candidate)):
        if len(pool.inits) > workers:
            waiting.append(
                f"{name}: {len(pool.inits)} worker initializations for {workers} "
                "workers (a duplicated log, or recycling the projection does "
                "not model); restart the watcher on a fresh log"
            )
    first_incumbent = _first_slice_seconds(incumbent, workers)
    first_candidate = _first_slice_seconds(candidate, workers)
    if incumbent.chunks is None or incumbent.slices_per_chunk is None:
        waiting.append("the incumbent has scored no slice yet")
    elif incumbent.slices_seen < incumbent.chunks * incumbent.slices_per_chunk:
        waiting.append(
            f"the incumbent has scored {incumbent.slices_seen} of "
            f"{incumbent.chunks * incumbent.slices_per_chunk} slices"
        )
    for name, pool, first in (
        ("incumbent", incumbent, first_incumbent),
        ("candidate", candidate, first_candidate),
    ):
        if first is None:
            waiting.append(
                f"the {name} has {len(pool.inits)} of {workers} worker "
                f"initializations and {len(pool.chunk1_times)} chunk-1 slices: "
                "not yet every worker's start and each early worker's first slice"
            )
    if candidate.chunks is not None and candidate.chunks != incumbent.chunks:
        waiting.append(
            f"the artifacts have {incumbent.chunks} and {candidate.chunks} registry "
            "chunks; the projection assumes one registry"
        )
    if waiting:
        return {**report, "verdict": "WAIT", "reasons": waiting}

    assert first_incumbent is not None and first_candidate is not None
    d_incumbent, incumbent_pairs = first_incumbent
    d_candidate, candidate_pairs = first_candidate
    assert incumbent.slices_per_chunk is not None
    assert candidate.slices_per_chunk is not None and candidate.chunks is not None
    # The incumbent's total slice work: each worker's last completion (the
    # last `workers` completions) less its initialization.
    incumbent_total = incumbent.chunks * incumbent.slices_per_chunk
    notes: list[str] = []
    if (
        sum(1 for done in incumbent.slice_times if done <= incumbent.inits[-1])
        <= incumbent_total - workers
    ):
        work = sum(incumbent.slice_times[-workers:]) - sum(incumbent.inits)
    else:
        # The last worker started so late that it may have found the queue
        # empty, so the last completions need not be one per worker. Each
        # worker is busy at most from its start to the pool's last slice.
        work = sum(
            max(0.0, incumbent.slice_times[-1] - start) for start in incumbent.inits
        )
        notes.append(
            "the incumbent's last worker started late; its work uses the "
            "busy-span upper bound"
        )
    incumbent_chunk_sum = work / incumbent.slices_per_chunk
    speed = d_candidate / d_incumbent
    candidate_chunk_sum = speed * incumbent_chunk_sum
    scoring = candidate.slices_per_chunk * candidate_chunk_sum / workers
    # Graham's bound adds the longest slice; no chunk's slice is longer than
    # the per-chunk durations summed.
    tail = candidate_chunk_sum
    candidate_start = sum(candidate.inits) / len(candidate.inits)
    projected_end = candidate_start + scoring + tail + limits.post_scoring_seconds

    candidate_total = candidate.chunks * candidate.slices_per_chunk
    # The mean slices left per worker, plus one per chunk and one more for a
    # worker that scores more than its share.
    remaining_per_worker = (
        max(0, candidate_total - candidate.slices_seen) / workers + candidate.chunks + 1
    )
    growth = max(
        _growth_per_slice(incumbent, workers), _growth_per_slice(candidate, workers)
    )
    worker_peak = (
        max(candidate.init_peak, candidate.slice_peak)
        + RSS_RESOLUTION_GIB / 2
        + growth * remaining_per_worker
    )
    parent_peak = max(incumbent.parent_peak, candidate.parent_peak)
    container_peak = workers * worker_peak + parent_peak

    reasons: list[str] = []
    time_line = deadline - limits.time_margin * limits.max_wall_seconds
    if projected_end > time_line:
        reasons.append(
            f"time: projected end {_utc(projected_end)} is past "
            f"{_utc(time_line)} ({limits.time_margin:.0%} inside the "
            f"{limits.max_wall_seconds:.0f} s budget ending {_utc(deadline)})"
        )
    memory_line = (1 - limits.memory_margin) * limits.memory_limit_gib
    if container_peak > memory_line:
        reasons.append(
            f"memory: projected container peak {container_peak:.1f} GiB is over "
            f"{memory_line:.1f} GiB ({limits.memory_margin:.0%} inside the "
            f"{limits.memory_limit_gib:.0f} GiB limit)"
        )
    process_line = (1 - limits.memory_margin) * limits.process_limit_gib
    if worker_peak > process_line:
        reasons.append(
            f"memory: projected worker peak {worker_peak:.2f} GiB is over "
            f"{process_line:.2f} GiB ({limits.memory_margin:.0%} inside the "
            f"scorer's {limits.process_limit_gib:.0f} GiB per-process limit)"
        )
    return {
        **report,
        "verdict": "NO-GO" if reasons else "GO",
        "reasons": reasons,
        "estimator_notes": notes,
        "projection": {
            "incumbent_first_slice_seconds": round(d_incumbent, 1),
            "incumbent_first_slice_pairs": incumbent_pairs,
            "candidate_first_slice_seconds": round(d_candidate, 1),
            "candidate_first_slice_pairs": candidate_pairs,
            "candidate_speed_ratio": round(speed, 4),
            "candidate_scoring_seconds": round(scoring + tail, 1),
            "projected_end_utc": _utc(projected_end),
            "projected_tool_seconds_from_start": round(
                projected_end - limits.started_at, 1
            ),
            "growth_gib_per_slice": round(growth, 4),
            "worker_peak_gib": round(worker_peak, 2),
            "container_peak_gib": round(container_peak, 1),
        },
    }


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(math.floor(epoch), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def limits_from_plan(plan_path: Path, started_at: float) -> Limits:
    """The plan's workers and wall budget and its class's memory limit."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import modal_us_stage_plan as plan_lib

    _, plan = plan_lib.load_plan(plan_path)
    if plan.tool.name != "us-release-head-to-head":
        raise SystemExit(f"{plan_path}: not a us-release-head-to-head plan")
    if "worker_max_slices" in plan.options:
        raise SystemExit(
            f"{plan_path}: sets worker_max_slices; the projection assumes each "
            "worker lives for its whole pool"
        )
    workers = plan.options.get("workers")
    if not isinstance(workers, int) or workers < 2:
        raise SystemExit(f"{plan_path}: the projection needs workers >= 2")
    if plan.max_wall_seconds is None:
        raise SystemExit(f"{plan_path}: sets no max_wall_seconds")
    memory_mib = plan.resources.memory_limit_mib or plan.resources.memory_mib
    return Limits(
        workers=workers,
        started_at=started_at,
        max_wall_seconds=plan.max_wall_seconds,
        memory_limit_gib=memory_mib / 1024,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", type=Path, help="a saved `modal app logs` stream")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--started-at",
        required=True,
        help="UTC start of the attempt, YYYY-MM-DDTHH:MM:SSZ; the launch time "
        "(earlier) is conservative",
    )
    args = parser.parse_args(argv)
    limits = limits_from_plan(args.plan, _epoch(args.started_at))
    events, notes = parse_events(args.log.read_text(errors="replace").splitlines())
    report = project(events, limits)
    print(json.dumps({**report, "notes": notes}, indent=2))
    return 1 if report["verdict"] == "NO-GO" else 0


if __name__ == "__main__":
    raise SystemExit(main())
