"""Simulated d844 runs and the watcher's verdicts on them, from the Mac times.

Not a measurement. For each speed factor f (Modal seconds per Mac second,
the same for every chunk unless --chunk5 says otherwise; --chunk5-artifact
candidate slows only the candidate's chunk 5) and worker start
cost T, it simulates one 16-worker run with test_support's pool simulator
(on-demand worker starts as CPython makes them, chunk-major tasks, twice the
workers outstanding, in-order consumption) from this folder's per-chunk
slice means, then polls tools/us_head_to_head_projection.py every 600 s, as
the watcher does, against the plan's 14,505 s budget, with this folder's
chunk-weights.json as the prior. Start-up is the Mac's import, compile and
load times times f; the comparison and scorecards are taken as 600 s; each
worker's first slice carries 5 s more, times f. Slices within a chunk take
the same time, so workers run in step.

Columns: the simulated tool's seconds and cost to its end (the runner stops
it at 14,505 s); the poll and cost of the first verdict other than WAIT,
with its estimate; the first NO-GO's poll and cost; FALSE when a NO-GO came
for a run that fits.

A run fits when its last slice ends at least 600 s before the budget
ends: the estimate's allowance for the comparison and scorecards. The
runner's budget counts from the tool's own start; the watcher's counts from
started-at, which launch.sh writes before `modal run`, so container start
and staging come out of the watcher's budget only. --pre-tool sets that gap
(0 by default: both budgets start together).

usage: python -I scripts/projection_grid.py CHECKOUT [--chunk5 FACTOR]
       [--chunk5-artifact both|candidate] [--pre-tool SECONDS]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("checkout", type=Path)
parser.add_argument(
    "--chunk5",
    type=float,
    default=1.0,
    help="chunk 5 runs this many times slower on Modal than the prior says",
)
parser.add_argument(
    "--chunk5-artifact",
    choices=("both", "candidate"),
    default="both",
    help="which artifacts' chunk 5 --chunk5 slows",
)
parser.add_argument(
    "--pre-tool",
    type=int,
    default=0,
    help="seconds from started-at to the tool's own start",
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

weights_path = args.checkout / "docs/evidence/us-h2h-speed/chunk-weights.json"
weights = json.loads(weights_path.read_text())
prior = projection._prior(weights_path)
cand = weights["candidate_mean_slice_seconds"]
inc = weights["incumbent_mean_slice_seconds"]
RATE = 0.00254688  # USD per second, 16 CPU, 288 GiB, non-preemptible
BUDGET = 14505
STARTUP = 38 + 23 + 3  # import, compile, load the incumbent (Mac seconds)
BETWEEN = 11  # load the candidate (Mac seconds)
POST = 600
POLL = 600
START = 1_794_816_000


def scaled(means: list[float], f: float, slow: bool = True) -> list[int]:
    out = [round(x * f) for x in means]
    out[-1] = round(means[-1] * f * (args.chunk5 if slow else 1.0))
    return out


SLOW_INCUMBENT = args.chunk5_artifact == "both"
print(
    f"chunk 5 slowed {args.chunk5}x beyond the prior "
    f"({'both artifacts' if SLOW_INCUMBENT else 'candidate only'}); "
    f"budget {BUDGET} s; started-at {args.pre_tool} s before the tool"
)
print(
    f"{'f':>4} {'T':>4} | {'tool s':>7} {'$':>6} {'fits':>4} | "
    f"{'first s':>7} {'$':>6} {'verdict':>7} {'est s':>7} | {'NO-GO s':>7} {'$':>6}"
)
for f in (1, 1.5, 2, 3, 4, 5, 7.6):
    for start_cost in (40, 100, 175, 300, 450, 600):
        run = sim.run(
            started_at=START,
            workers=16,
            incumbent_seconds=scaled(inc, f, SLOW_INCUMBENT),
            candidate_seconds=scaled(cand, f),
            incumbent_slices=29,
            candidate_slices=177,
            start_cost=start_cost,
            first_slice_extra=round(5 * f),
            load_seconds=round(STARTUP * f),
            candidate_load_seconds=round(BETWEEN * f),
            scorecard_seconds=POST,
        )
        end = run.tool_end - START
        fits = end <= BUDGET
        limits = projection.Limits(
            workers=16,
            started_at=START - args.pre_tool,
            max_wall_seconds=BUDGET,
            prior=prior,
        )
        first = no_go = None
        at = START + POLL
        while at <= START + min(end, BUDGET):
            report = projection.project(projection.parse_log(run.log(at)), limits, at)
            if first is None and report["verdict"] != "WAIT":
                first = (at - START, report)
            if report["verdict"] == "NO-GO":
                no_go = at - START
                break
            at += POLL
        row = (
            f"{f:>4} {start_cost:>4} | {end:>7.0f} {min(end, BUDGET) * RATE:>6.2f} "
            f"{'yes' if fits else 'no':>4} | "
        )
        if first is None:
            row += f"{'-':>7} {'-':>6} {'WAIT':>7} {'-':>7} | "
        else:
            seconds, report = first
            estimate = report.get("estimate", {}).get(
                "estimated_tool_seconds_from_start", "-"
            )
            row += (
                f"{seconds:>7.0f} {seconds * RATE:>6.2f} {report['verdict']:>7} "
                f"{estimate:>7} | "
            )
        if no_go is None:
            row += f"{'-':>7} {'-':>6}"
        else:
            row += f"{no_go:>7.0f} {no_go * RATE:>6.2f}" + (" FALSE" if fits else "")
        print(row, flush=True)
