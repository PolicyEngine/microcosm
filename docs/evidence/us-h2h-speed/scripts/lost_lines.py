"""Lost log lines and the watcher's verdict, in simulated runs.

Not a measurement. A log stream that ends is restarted by the watcher as a
new file; if the new stream does not replay what the app printed, the lines
in between are lost, and whether ``modal app logs`` replays, and from where,
is untried. This script damages simulated logs (test_support's pool
simulator, the model the estimate assumes) the ways a restart can, reads the
pieces as the watcher's streams, and compares the verdict of
tools/us_head_to_head_projection.py with the intact log's at the same poll.

The estimate does not depend on the budget (the time margin is 0), so each
poll is judged once, with no budget, and the verdict at any budget follows:
NO-GO when the log stalls, or when the estimated end passes the budget and no
pool's starts are unconfirmed (``starts_unconfirmed``, which holds a time
NO-GO as WAIT); WAIT when there is no estimate or the hold applies; GO
otherwise. Every 50th poll is also judged directly at both budgets below,
and the count of mismatches is printed. A run with S seconds of spare is
stopped falsely where the damaged log gives NO-GO and the intact one does
not, so each search reports the largest **excess** at the polls the hold
leaves free: the damaged estimate less the intact estimate at the same poll.
A run whose intact estimate has more spare than that is never stopped by the
damage searched here. The largest excess at held polls is reported too: it
is what the hold guards against. Each search also gives every verdict
transition (intact -> damaged) at two budgets: a tight fit, the true end
plus 265 s (the spare of the d844 shape at f = 1, T = 300 s), and an
overrun, the true end less 600 s. The true end counts the estimate's 600 s
tail allowance. The intact log's own NO-GO at the tight fit is counted
apart: that is the estimate's error, not the damage's.

Sections:

- cut-d844, cut-random: a restart that does not replay loses [s, s + w):
  w = 1, 10 and 60 s ending or starting at each worker start, pool opening
  and release line, and w = 0 (nothing lost), 10 and 60 s every 120 s (d844)
  or 300 s (random), plus pairs of 10 s cuts at two start lines 300 to
  2,400 s apart (two restarts): every pair when there are fewer than 120,
  else an even sample of 60 to 90. Polls come every 5 s to 300 s after the
  last cut and every 30 s to 1,200 s, so the first ones see the new stream
  still empty.
- boundary: a restart inside one second, at every second with records: the
  log split there with nothing lost (a continuation), a replay from that
  second's first record, and a replay of only the last record seen, each
  polled at the next three 600 s polls.
- lossless: the d844 log split with nothing lost every 60 s, polled every
  600 s as the watcher does: every verdict transition, and how much later
  than the intact log's the split log's first NO-GO comes (a missed stop).

Shapes: d844 (16 workers, this folder's slice means times f, worker start
cost T or one cost per start with the last repeating, chunk-weights.json as
the prior), and random pools drawn as the property tests draw them (1 to 16
workers, 1 to 5 chunks of 20 to 300 s slices) with one to four start costs of
1 to 400 s, the last repeating, and their own slice times as the prior.
``--constant-starts`` keeps each random pool's first start cost for every
start and draws the same pools otherwise, which separates what varying start
costs do to the estimate from what the damage does.

usage: python -I scripts/lost_lines.py CHECKOUT --section SECTION
           [--pools N] [--seeds 3,7,11] [--rows ROWS.jsonl] [--constant-starts]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("checkout", type=Path)
parser.add_argument(
    "--section",
    choices=("cut-d844", "cut-random", "boundary", "lossless"),
    required=True,
)
parser.add_argument("--pools", type=int, default=100)
parser.add_argument("--seeds", default="3,7,11")
parser.add_argument("--rows", type=Path, help="JSON lines for polls with excess > 0")
parser.add_argument(
    "--constant-starts",
    action="store_true",
    help="random pools keep only their first start cost (the same pools otherwise)",
)
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
BIG = 10**9
SPARE = 265
SHORT = 600
RATE = 0.00254688  # USD per second, the class's non-preemptible list price
weights_path = args.checkout / "docs/evidence/us-h2h-speed/chunk-weights.json"
weights = json.loads(weights_path.read_text())
D844_PRIOR = projection._prior(weights_path)
rows = args.rows.open("w") if args.rows else None
direct = Counter()


def offset(line: str) -> float | None:
    times = projection.parse_log([line]).record_times
    return times[0] if times else None


def d844_run(f: float, start_cost) -> sim.Run:
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


def random_pools(seed: int, count: int):
    rng = random.Random(seed)
    for k in range(count):
        workers = rng.randint(1, 16)
        chunks = rng.randint(1, 5)
        params = dict(
            workers=workers,
            incumbent_seconds=[rng.randint(20, 300) for _ in range(chunks)],
            candidate_seconds=[rng.randint(20, 300) for _ in range(chunks)],
            incumbent_slices=rng.randint(1, 3 * workers),
            candidate_slices=rng.randint(1, 10 * workers),
            start_cost=[rng.randint(1, 400) for _ in range(rng.randint(1, 4))],
            first_slice_extra=rng.randint(0, 200),
        )
        if args.constant_starts:
            params["start_cost"] = params["start_cost"][0]
        yield f"seed {seed} pool {k}", params, sim.run(started_at=START, **params)


def own_prior(run: sim.Run):
    return projection.Prior(
        candidate=tuple(map(float, run.candidate_seconds)),
        incumbent=tuple(map(float, run.incumbent_seconds)),
        candidate_slices=run.candidate_slices,
    )


class Shape:
    def __init__(self, name, run, workers, prior, scorecard) -> None:
        self.name, self.run, self.workers, self.prior = name, run, workers, prior
        # The estimate's end is the last slice plus a 600 s tail allowance.
        self.truth = run.tool_end - START + 600 - scorecard
        self.full = run.log()
        self.offsets = [offset(line) for line in self.full]

    def limits(self, budget: float = BIG):
        return projection.Limits(
            workers=self.workers,
            started_at=START,
            max_wall_seconds=budget,
            prior=self.prior,
        )

    def upto(self, at: float) -> list[str]:
        return [
            line for line, t in zip(self.full, self.offsets) if t is None or t <= at
        ]


def d844_shape(f, start_cost) -> Shape:
    label = start_cost if isinstance(start_cost, int) else f"{start_cost[:3]}..."
    return Shape(f"d844 f={f} T={label}", d844_run(f, start_cost), 16, D844_PRIOR, 600)


def end_of(report: dict) -> float | None:
    estimate = report.get("estimate")
    return estimate["estimated_tool_seconds_from_start"] if estimate else None


def verdict_at(report: dict, budget: float) -> str:
    """The verdict ``report``, judged with no budget, gives at ``budget``."""

    if report["verdict"] == "NO-GO":
        return "NO-GO"  # a stall, at any budget
    end = end_of(report)
    if end is None:
        return "WAIT"
    if end > budget:
        return "WAIT" if report.get("starts_unconfirmed") else "NO-GO"
    if report["verdict"] == "WAIT":
        return "WAIT"
    return "GO"


class Tally:
    def __init__(self, truth: float = 0.0) -> None:
        self.budgets = {"fit": truth + SPARE, "short": truth - SHORT}
        self.pairs = {label: Counter() for label in self.budgets}
        self.polls = self.estimates = self.held = self.stalls = 0
        self.excess = self.held_excess = float("-inf")

    def add(self, intact, damaged, ctx, judge=None) -> None:
        self.polls += 1
        for label, budget in self.budgets.items():
            pair = (verdict_at(intact, budget), verdict_at(damaged, budget))
            self.pairs[label][pair] += 1
            if judge is not None and self.polls % 50 == 0:
                direct["checked"] += 1
                if judge(budget) != pair[1]:
                    direct["mismatched"] += 1
                    print(f"    MISMATCH {ctx} budget {budget}", flush=True)
        if damaged["verdict"] == "NO-GO" and intact["verdict"] != "NO-GO":
            self.stalls += 1
        ends = end_of(intact), end_of(damaged)
        if None not in ends:
            self.estimates += 1
            excess = ends[1] - ends[0]
            if damaged.get("starts_unconfirmed"):
                self.held += 1
                self.held_excess = max(self.held_excess, excess)
            else:
                self.excess = max(self.excess, excess)
                if excess > 0 and rows:
                    rows.write(json.dumps({**ctx, "excess": round(excess, 1)}) + "\n")

    def merge(self, other: Tally) -> None:
        for label in self.pairs:
            self.pairs[label].update(other.pairs[label])
        self.polls += other.polls
        self.estimates += other.estimates
        self.held += other.held
        self.stalls += other.stalls
        self.excess = max(self.excess, other.excess)
        self.held_excess = max(self.held_excess, other.held_excess)

    def line(self) -> str:
        def table(counter: Counter) -> str:
            return ", ".join(f"{i}->{d} {n}" for (i, d), n in sorted(counter.items()))

        def seconds(value: float) -> str:
            return "-" if value == float("-inf") else f"{value:.1f} s"

        fit = self.pairs["fit"]
        false = fit[("GO", "NO-GO")] + fit[("WAIT", "NO-GO")]
        own = sum(n for (i, _), n in fit.items() if i == "NO-GO")
        return (
            f"{self.polls} polls, {self.estimates} with both estimates, max excess "
            f"{seconds(self.excess)} where the hold is off ({self.held} held, max "
            f"excess there {seconds(self.held_excess)}); at the true end +{SPARE} "
            f"s: {table(fit)} (false NO-GO {false}; the intact log's own NO-GO "
            f"{own}); at the true end -{SHORT} s: {table(self.pairs['short'])}; "
            f"NO-GO from a stall the intact log does not show {self.stalls}"
        )


POLLS = list(range(5, 301, 5)) + list(range(330, 1201, 30))


def streams_for(shape: Shape, cuts, at) -> list[list[str]]:
    """The streams restarts at each cut ``(s, w)`` leave, losing [s, s + w)."""

    pairs = list(zip(shape.full, shape.offsets))
    out = [[line for line, t in pairs if t is None or t < cuts[0][0]]]
    for (a, wa), (b, _) in zip(cuts, cuts[1:]):
        out.append([line for line, t in pairs if t is not None and a + wa <= t < b])
    s, w = cuts[-1]
    out.append([line for line, t in pairs if t is not None and s + w <= t <= at])
    return out


def cut_search(shape: Shape, grid: int) -> Tally:
    offsets = shape.offsets
    seen = sorted({t for t in offsets if t is not None})
    marks = [
        t
        for line, t in zip(shape.full, offsets)
        if t is not None
        and any(k in line for k in ("initializing", "before starting", "releasing"))
    ]
    cuts: set = set()
    for t in marks:
        for w in (1, 10, 60):
            cuts.update({((t, w),), ((t - w + 1, w),)})
    for t in range(int(seen[0]), int(seen[-1]), grid):
        for w in (0, 10, 60):
            cuts.add(((t, w),))
    doubles = [
        ((a, 10), (b, 10))
        for i, a in enumerate(marks)
        for b in marks[i + 1 :]
        if 300 <= b - a <= 2400
    ]
    cases = sorted(cuts) + doubles[:: max(1, len(doubles) // 60)]
    limits = shape.limits()
    tally = Tally(shape.truth)
    intact_at: dict = {}
    for case in cases:
        if any(
            w and not any(t is not None and s <= t < s + w for t in offsets)
            for s, w in case
        ):
            continue  # loses nothing, yet is not a w = 0 split
        last = case[-1][0] + case[-1][1]
        for d in POLLS:
            at = last + d
            if at >= shape.run.candidate.end:
                break
            if at not in intact_at:
                intact_at[at] = projection.project(
                    projection.parse_log(shape.upto(at)), limits, at
                )
            streams = streams_for(shape, case, at)
            damaged = projection.project(projection.parse_logs(streams), limits, at)

            def judge(budget, streams=streams, at=at):
                parsed = projection.parse_logs(streams)
                return projection.project(parsed, shape.limits(budget), at)["verdict"]

            ctx = {
                "shape": shape.name,
                "cuts": [(s - START, w) for s, w in case],
                "poll": at - START,
            }
            tally.add(intact_at[at], damaged, ctx, judge)
    return tally


def boundary_search(shape: Shape) -> dict[str, Tally]:
    offsets = shape.offsets
    limits = shape.limits()
    arms = {
        arm: Tally(shape.truth) for arm in ("split", "replay-second", "replay-last")
    }
    end = shape.run.candidate.end - START
    for s in sorted({t for t in offsets if t is not None}):
        idx = [i for i, t in enumerate(offsets) if t == s]
        i0, i1 = idx[0], idx[-1] + 1
        polls = [p for p in range(600, int(end) + 1, 600) if START + p >= s][:3]
        for p in polls:
            at = START + p
            log = shape.upto(at)
            intact = projection.project(projection.parse_log(log), limits, at)
            cases = {
                "replay-second": [log[:i1], log[i0:]],
                "replay-last": [log[:i1], log[i1 - 1 :]],
            }
            if i1 - i0 >= 2:
                k = i0 + (i1 - i0) // 2
                cases["split"] = [log[:k], log[k:]]
            for arm, streams in cases.items():
                damaged = projection.project(projection.parse_logs(streams), limits, at)
                ctx = {"shape": shape.name, "arm": arm, "second": s - START, "poll": p}
                arms[arm].add(intact, damaged, ctx)
    return arms


def lossless_line(f, start_cost) -> str:
    shape = d844_shape(f, start_cost)
    limits = shape.limits(14_505)
    stop = min(shape.run.tool_end - START, 14_505)
    polls = range(600, int(stop) + 1, 600)

    def judge(at, streams):
        return projection.project(projection.parse_logs(streams), limits, at)["verdict"]

    intact = {p: judge(START + p, [shape.upto(START + p)]) for p in polls}
    first_no_go = next((p for p in polls if intact[p] == "NO-GO"), None)
    transitions: Counter = Counter()
    delay = 0.0
    splits = 0
    pairs = list(zip(shape.full, shape.offsets))
    for split in range(60, int(first_no_go or stop), 60):
        splits += 1
        first = [line for line, t in pairs if t is None or t < START + split]
        for poll in polls:
            if poll < split:
                continue
            at = START + poll
            second = [
                line for line, t in pairs if t is not None and START + split <= t <= at
            ]
            verdict = judge(at, [first, second])
            transitions[intact[poll], verdict] += 1
            if first_no_go is not None and verdict == "NO-GO":
                delay = max(delay, poll - first_no_go)
                break
        else:
            if first_no_go is not None:
                delay = max(delay, stop - first_no_go)
    fits = shape.run.tool_end - START <= 14_505
    line = (
        f"  {shape.name} ({'fits' if fits else 'overruns'}, intact NO-GO at "
        f"{first_no_go or '-'} s): {splits} splits; "
        + ", ".join(f"{i}->{d} {n}" for (i, d), n in sorted(transitions.items()))
    )
    if first_no_go is not None:
        line += f"; first NO-GO up to {delay:.0f} s later (${delay * RATE:.2f})"
    return line


D844_SHAPES = [
    (1, 300),
    (1.01, 300),
    (1.02, 300),
    (1, 150),
    (2, 150),
    (1.5, 300),
    (3, 100),
    (1, 40),
    (1, 600),
    (1, [100] + [320] * 15),
    (1, [320] + [200] * 15),
    (1, [200, 400] * 8),
]

began = time.monotonic()
if args.section == "cut-d844":
    print("cut: a restart that does not replay loses a stretch (d844)", flush=True)
    for f, start_cost in D844_SHAPES:
        shape = d844_shape(f, start_cost)
        tally = cut_search(shape, 120)
        print(
            f"  {shape.name} (true end {shape.truth:.0f} s): {tally.line()} "
            f"[{time.monotonic() - began:.0f} s]",
            flush=True,
        )
elif args.section == "cut-random":
    print("cut: a restart that does not replay loses a stretch (random)", flush=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        total = Tally()
        worst = float("-inf")
        for name, params, run in random_pools(seed, args.pools):
            shape = Shape(
                name, run, params["workers"], own_prior(run), sim.SCORECARD_SECONDS
            )
            tally = cut_search(shape, 300)
            total.merge(tally)
            if tally.excess > worst:
                worst = tally.excess
                print(
                    f"    {name} {params}: max excess {tally.excess:.1f} s", flush=True
                )
        print(
            f"  seed {seed}, {args.pools} pools: {total.line()} "
            f"[{time.monotonic() - began:.0f} s]",
            flush=True,
        )
elif args.section == "boundary":
    print("boundary: a restart inside one second", flush=True)
    shapes = [d844_shape(1, 300), d844_shape(1, [100] + [320] * 15)]
    shapes += [
        Shape(name, run, params["workers"], own_prior(run), sim.SCORECARD_SECONDS)
        for name, params, run in random_pools(3, 30)
    ]
    totals = {arm: Tally() for arm in ("split", "replay-second", "replay-last")}
    for shape in shapes:
        for arm, tally in boundary_search(shape).items():
            totals[arm].merge(tally)
            if shape.name.startswith("d844"):
                print(f"  {shape.name}, {arm}: {tally.line()}", flush=True)
    for arm, total in totals.items():
        print(f"  all {len(shapes)} shapes, {arm}: {total.line()}", flush=True)
else:
    print(
        "lossless: a split every 60 s that loses nothing, polled every 600 s",
        flush=True,
    )
    for f in (1, 1.5, 2, 3):
        for start_cost in (40, 100, 175, 300, 450, 600, [100] + [320] * 15):
            print(lossless_line(f, start_cost), flush=True)
print(
    f"direct judgements: {direct['checked']} checked, "
    f"{direct['mismatched']} differ from the derived verdict; "
    f"{time.monotonic() - began:.0f} s",
    flush=True,
)
