"""Seed faults in the d844 early-stop estimate and check its tests catch each.

Each fault is one edit to ``tools/us_head_to_head_projection.py``. The script
applies it in place, runs the estimate's test file, and restores the module
whatever happens; it refuses to start if the module differs from the
checkout's HEAD, so a run that is killed can be undone with ``git checkout``.
Every run gets a fresh Hypothesis example database, so an example saved
under one fault cannot replay under the next and fake a catch. A fault that
no test catches is reported as SURVIVED and the script exits nonzero.

Run from the repository root:
    uv run python docs/evidence/us-h2h-speed/scripts/mutation_check.py

``--runner`` replaces ``python -m pytest`` with another command that takes
pytest's arguments (a wrapper that sets up an engine-free environment, say).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "tools" / "us_head_to_head_projection.py"
TESTS = (
    ROOT
    / "packages"
    / "microcosm-build"
    / "tests"
    / "engine_free"
    / "us"
    / "test_us_head_to_head_projection.py"
)
MUTATIONS = {
    "the opening burst is workers, not twice that": (
        "    opening = min(2 * workers, total)\n",
        "    opening = min(workers, total)\n",
    ),
    "receive lines are ignored": (
        "        receipt = receives[task - 1] if len(receives) >= task else None\n",
        "        receipt = None\n",
    ),
    "a token-taking submit is dated by its result, not its receive line": (
        "                taken += 1\n                submitted[2 * workers + task] = received\n",
        "                taken += 1\n                submitted[2 * workers + task] = ready\n",
    ),
    "a full-pool submit is dated by its result, not its receive line": (
        "                # The pool is full, so this submit starts no worker either.\n"
        "                submitted[2 * workers + task] = received\n",
        "                submitted[2 * workers + task] = ready\n",
    ),
    "the replay starts every remaining worker at once": (
        "        elif spawns < workers:\n"
        "            advance(clock + start_cost)\n"
        "            add_worker()\n",
        "        elif spawns < workers:\n"
        "            while spawns < workers:\n"
        "                add_worker()\n",
    ),
    "replayed worker starts cost nothing": (
        "            advance(clock + start_cost)\n",
        "            advance(clock)\n",
    ),
    "replayed submits never take a token": (
        "        if tokens:\n            tokens -= 1\n",
        "        if False:\n            tokens -= 1\n",
    ),
    "a worker's first slice costs nothing extra": (
        "            took = seconds[chunk_of(task)] + (0.0 if who in has_run else first_extra)\n",
        "            took = seconds[chunk_of(task)]\n",
    ),
    "workers' first slices count toward the chunk's pace": (
        "        if task in firsts:\n            first.append",
        "        if False:\n            first.append",
    ),
    "unmeasured chunks scale from the fastest result": (
        "            seconds[chunk] = max(\n",
        "            seconds[chunk] = min(\n",
    ),
    "no allowance for the comparison and scorecards": (
        '            "end": last + limits.post_scoring_seconds,\n',
        '            "end": last,\n',
    ),
    "NO-GO comes 600 s late": (
        "    if end > time_line:\n",
        "    if end > time_line + 600:\n",
    ),
    "an unexplained start is never put back on its submit": (
        "            while spawns < min(workers, len(inits)) and inits[spawns] < ready and taken:\n",
        "            while False:\n",
    ),
    "candidate starts count only after its first slice": (
        "        costs = history.start_costs\n"
        "        return Timing(\n"
        "            {}, {}, [], sum(costs) / len(costs) if costs else None, list(history.notes)\n"
        "        )\n",
        "        return Timing({}, {}, [], None, list(history.notes))\n",
    ),
    "an overdue running slice ends now": (
        "                end = 2 * now - end\n",
        "                end = now\n",
    ),
    "the candidate's fill ignores the incumbent's measured chunks": (
        "        if carried:\n",
        "        if False:\n",
    ),
    "a lost slice line is not a gap": (
        "                gaps.append(task)\n",
        "                pass\n",
    ),
    "streams that began late can still say GO": (
        '    verdict = "NO-GO" if reasons else "WAIT" if budget_unseen else "GO"\n',
        '    verdict = "NO-GO" if reasons else "GO"\n',
    ),
    "before any record the stall allowance never starts": (
        "    last = max(seen, default=limits.started_at)\n",
        "    last = max(seen, default=now)\n",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        default=f"{shlex.quote(sys.executable)} -m pytest",
        help="command that runs pytest with the arguments after it",
    )
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{MODULE.relative_to(ROOT)}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    original = MODULE.read_text()
    if original != head:
        print(f"{MODULE} differs from HEAD; commit or restore it first")
        return 2
    for name, (old, _) in MUTATIONS.items():
        if original.count(old) != 1:
            print(f"fault {name!r} no longer matches the module exactly once")
            return 2
    survived = []
    try:
        for name, (old, new) in MUTATIONS.items():
            MODULE.write_text(original.replace(old, new))
            with tempfile.TemporaryDirectory() as store:
                env = {**os.environ, "HYPOTHESIS_STORAGE_DIRECTORY": store}
                result = subprocess.run(
                    [
                        *shlex.split(args.runner),
                        "-q",
                        "-x",
                        "-p",
                        "no:randomly",
                        "-p",
                        "no:cacheprovider",
                        str(TESTS),
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                )
            caught = result.returncode == 1
            failed = [
                line.split("::")[-1].split(" ")[0]
                for line in result.stdout.splitlines()
                if line.startswith("FAILED")
            ]
            if caught:
                print(f"killed    {name}: {', '.join(failed) or 'a test failed'}")
            elif result.returncode == 0:
                print(f"SURVIVED  {name}")
                survived.append(name)
            else:
                print(f"ERROR     {name}: pytest exited {result.returncode}")
                print(result.stdout[-2000:], result.stderr[-2000:])
                survived.append(name)
    finally:
        MODULE.write_text(original)
    print(f"{len(MUTATIONS) - len(survived)} of {len(MUTATIONS)} faults caught")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
