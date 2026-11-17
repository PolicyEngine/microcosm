"""Lost log lines and the watcher's verdict, in simulated runs.

Not a measurement. A log stream that ends is restarted by the watcher as a
new file; if the new stream does not replay what the app printed, the lines
in between are lost. This script cuts such a stretch out of simulated logs
(test_support's pool simulator, the model the estimate assumes), reads the
two pieces as the watcher's two streams, and compares the verdict of
tools/us_head_to_head_projection.py with the intact log's at the same poll.
A flip is GO turned to NO-GO (a run that fits stopped) or NO-GO turned to
GO (an early stop missed); WAIT in place of either is the hold the lost-line
rules give, which leaves the run to the runner's budget.

Three searches:

- d844: the 16-worker d844 shape from this folder's slice means, times a
  speed factor f, with worker start cost T, against the plan's 14,505 s
  budget and chunk-weights.json as the prior. Every stretch of W seconds,
  starting at every STEP seconds before the poll, at each poll.
- random: pools drawn as the property tests draw them (2 to 16 workers,
  1 to 5 chunks of 20 to 300 s slices, start costs 1 to 400 s), with their
  own slice times as the prior, cut at 15 random record times by 5, 20 or
  60 s, and judged twice: against a budget 600 to 900 s past the simulated
  end (a run that fits) and against one 600 s short of it (a run that
  overruns).
- lossless: the d844 grid's start costs and speed factors 1 to 3, each log
  split into two streams with nothing lost (a restart that does not replay
  but misses no line) every 900 s, polled every 600 s as the watcher does.
  It counts the polls whose verdict differs from the intact log's, up to the
  intact log's first NO-GO (when the watcher would stop the app), and how
  much later the split log's first NO-GO comes.

usage: python -I scripts/lost_lines.py CHECKOUT [--pools N] [--seeds 3,7,11]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("checkout", type=Path)
parser.add_argument("--pools", type=int, default=300)
parser.add_argument("--seeds", default="3,7,11")
args = parser.parse_args()
sys.path.insert(0, str(args.checkout))
from test_support.microcosm_build import h2h_pool_sim as sim  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "us_head_to_head_projection", args.checkout / "tools/us_head_to_head_projection.py"
)
projection = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = projection
spec.loader.exec_module(projection)

START = 1_794_816_000
weights_path = args.checkout / "docs/evidence/us-h2h-speed/chunk-weights.json"
weights = json.loads(weights_path.read_text())


def offset(line: str) -> float | None:
    times = projection.parse_log([line]).record_times
    return times[0] if times else None


def cut(log, offsets, start, end):
    first = [line for line, t in zip(log, offsets) if t is None or t < start]
    second = [line for line, t in zip(log, offsets) if t is not None and t >= end]
    return first, second


def flipped(intact: str, damaged: str) -> bool:
    return {intact, damaged} == {"GO", "NO-GO"}


def d844_run(f: float, start_cost: int):
    return sim.run(
        started_at=START,
        workers=16,
        incumbent_seconds=[
            round(x * f) for x in weights["incumbent_mean_slice_seconds"]
        ],
        candidate_seconds=[
            round(x * f) for x in weights["candidate_mean_slice_seconds"]
        ],
        incumbent_slices=29,
        candidate_slices=177,
        start_cost=start_cost,
        first_slice_extra=round(5 * f),
        load_seconds=round(64 * f),
        candidate_load_seconds=round(11 * f),
        scorecard_seconds=600,
    )


print("d844 shape (16 workers, 14,505 s budget):")
d844_limits = projection.Limits(
    workers=16,
    started_at=START,
    max_wall_seconds=14_505,
    prior=projection._prior(weights_path),
)
for f, start_cost, width, step, polls in (
    (1, 300, 60, 30, (1200, 3600, 6000, 8400, 10_800, 13_200)),
    (1, 300, 10, 20, (1200, 4800, 9600)),
    (1.5, 300, 60, 30, (1200, 4800, 9600)),
    (3, 100, 60, 30, (1200, 2400, 3600)),
):
    run = d844_run(f, start_cost)
    full = run.log()
    offsets = [offset(line) for line in full]
    pairs: Counter[tuple[str, str]] = Counter()
    for poll in polls:
        at = START + poll
        log = [line for line, t in zip(full, offsets) if t is None or t <= at]
        kept = [t for t in offsets if t is None or t <= at]
        intact = projection.project(projection.parse_log(log), d844_limits, at)[
            "verdict"
        ]
        for start in range(0, poll - width, step):
            first, second = cut(log, kept, START + start, START + start + width)
            if len(first) + len(second) == len(log):
                continue
            damaged = projection.project(
                projection.parse_logs([first, second]), d844_limits, at
            )["verdict"]
            pairs[intact, damaged] += 1
    flips = sum(n for (i, d), n in pairs.items() if flipped(i, d))
    print(
        f"  f={f} T={start_cost} W={width} step={step} polls={','.join(map(str, polls))}"
        f" (tool ends {run.tool_end - START:.0f} s): "
        + ", ".join(f"{i}->{d} {n}" for (i, d), n in sorted(pairs.items()))
        + f"; flips {flips}"
    )

print("random pools:")
for seed in (int(s) for s in args.seeds.split(",")):
    rng = random.Random(seed)
    pairs = Counter()
    short_pairs: Counter[tuple[str, str]] = Counter()
    for _ in range(args.pools):
        workers = rng.randint(2, 16)
        chunks = rng.randint(1, 5)
        params = dict(
            workers=workers,
            incumbent_seconds=[rng.randint(20, 300) for _ in range(chunks)],
            candidate_seconds=[rng.randint(20, 300) for _ in range(chunks)],
            incumbent_slices=rng.randint(1, 3 * workers),
            candidate_slices=rng.randint(1, 10 * workers),
            start_cost=rng.randint(1, 400),
            first_slice_extra=rng.randint(0, 200),
        )
        run = sim.run(started_at=START, **params)
        prior = projection.Prior(
            candidate=tuple(map(float, run.candidate_seconds)),
            incumbent=tuple(map(float, run.incumbent_seconds)),
            candidate_slices=run.candidate_slices,
        )
        limits = projection.Limits(
            workers=workers,
            started_at=START,
            max_wall_seconds=run.tool_end - START + 600 + rng.randint(0, 300),
            prior=prior,
        )
        short = projection.Limits(
            workers=workers,
            started_at=START,
            max_wall_seconds=run.tool_end - START - 600,
            prior=prior,
        )
        full = run.log()
        times = run.times()
        offsets = [offset(line) for line in full]
        for _ in range(15):
            start = rng.choice(times)
            width = rng.choice([5, 20, 60])
            at = start + width + rng.choice([10, 40, 120, 300])
            if at >= run.candidate.end:
                continue
            log = [line for line, t in zip(full, offsets) if t is None or t <= at]
            kept = [t for t in offsets if t is None or t <= at]
            first, second = cut(log, kept, start, start + width)
            if len(first) + len(second) == len(log):
                continue
            for counts, these in ((pairs, limits), (short_pairs, short)):
                intact = projection.project(projection.parse_log(log), these, at)[
                    "verdict"
                ]
                damaged = projection.project(
                    projection.parse_logs([first, second]), these, at
                )["verdict"]
                counts[intact, damaged] += 1
    for label, counts in (("fits", pairs), ("overruns", short_pairs)):
        flips = sum(n for (i, d), n in counts.items() if flipped(i, d))
        print(
            f"  seed {seed}, {args.pools} pools, {sum(counts.values())} lost "
            f"stretches, budget {label}: "
            + ", ".join(f"{i}->{d} {n}" for (i, d), n in sorted(counts.items()))
            + f"; flips {flips}"
        )

print("restarts that lose nothing (d844 grid):")
RATE = 0.00254688  # USD per second, the class's non-preemptible list price
for f in (1, 1.5, 2, 3):
    for start_cost in (40, 100, 175, 300, 450, 600):
        run = d844_run(f, start_cost)
        full = run.log()
        offsets = [offset(line) for line in full]
        stop = min(run.tool_end - START, 14_505)
        polls = range(600, int(stop) + 1, 600)

        def judge(at, streams):
            return projection.project(projection.parse_logs(streams), d844_limits, at)[
                "verdict"
            ]

        intact = {}
        for poll in polls:
            at = START + poll
            intact[poll] = judge(
                at, [[line for line, t in zip(full, offsets) if t is None or t <= at]]
            )
        first_no_go = next((p for p in polls if intact[p] == "NO-GO"), None)
        held = flips = splits = 0
        delay = 0.0
        for split in range(300, int(first_no_go or stop), 900):
            splits += 1
            differ = 0
            for poll in polls:
                if poll < split:
                    continue
                at = START + poll
                first = [
                    line
                    for line, t in zip(full, offsets)
                    if t is None or t < START + split
                ]
                second = [
                    line
                    for line, t in zip(full, offsets)
                    if t is not None and START + split <= t <= at
                ]
                verdict = judge(at, [first, second])
                differ += verdict != intact[poll]
                flips += flipped(intact[poll], verdict)
                if first_no_go is not None and verdict == "NO-GO":
                    delay = max(delay, poll - first_no_go)
                    break
            else:
                if first_no_go is not None:
                    delay = max(delay, stop - first_no_go)
            held = max(held, differ)
        fits = run.tool_end - START <= 14_505
        line = (
            f"  f={f} T={start_cost} ({'fits' if fits else 'overruns'}, "
            f"intact NO-GO at {first_no_go or '-'} s): {splits} splits, "
            f"verdict differs at up to {held} poll(s)"
        )
        if first_no_go is not None:
            line += f", NO-GO up to {delay:.0f} s later (${delay * RATE:.2f})"
        print(line + f"; flips {flips}", flush=True)
