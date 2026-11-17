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
``--only TEXT`` (repeatable) seeds only the faults whose name contains TEXT.
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
        "    elif end > time_line:\n",
        "    elif end > time_line + 600:\n",
    ),
    "an unexplained start is never put back on its submit": (
        "            while (\n"
        "                spawns < min(workers, len(inits)) and inits[spawns] <= ready and taken\n"
        "            ):\n",
        "            while False:\n",
    ),
    "candidate starts count only after its first slice": (
        "        return Timing({}, {}, [], _start_cost(history), list(history.notes))\n",
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
        '    verdict = "NO-GO" if reasons else "WAIT" if held or budget_unseen else "GO"\n',
        '    verdict = "NO-GO" if reasons else "WAIT" if held else "GO"\n',
    ),
    "a start logged in its submit's second is taken as an earlier submit's": (
        "inits[spawns] <= ready and taken\n",
        "inits[spawns] < ready and taken\n",
    ),
    "a pool without its opening line is not a gap": (
        "        if pool.opened is None and (pool.inits or pool.receives or pool.by_task):\n",
        "        if False:\n",
    ),
    "the carried fill takes the slowest result, not one pooled ratio": (
        'measured["candidate"], candidate.chunks, fill_prior, pooled=bool(carried)\n',
        'measured["candidate"], candidate.chunks, fill_prior, pooled=False\n',
    ),
    "records from before started-at count as this attempt's": (
        "    early = sum(1 for t in parsed.record_times if t < limits.started_at)\n",
        "    early = 0\n",
    ),
    "streams with a hole report none": (
        "            holes.append((reach, first))\n",
        "            pass\n",
    ),
    "before any record the stall allowance never starts": (
        "    last = max(seen, default=limits.started_at)\n",
        "    last = max(seen, default=now)\n",
    ),
    "a rebuild that starts slices after they end is not a gap": (
        "        unplaced = len(_unplaced(pool, history))\n",
        "        unplaced = 0\n",
    ),
    "a submit or start the rebuild cannot explain is not a gap": (
        "            *history.unexplained,\n",
        "",
    ),
    "a slice from before a hole with no line yet is no suspect": (
        "            if tasks:\n",
        "            if False:\n",
    ),
    "a newest stream with nothing in it yet is no hole": (
        "    reopened = len(kept) > 1 and not kept[-1] and any(kept[:-1])\n",
        "    reopened = False\n",
    ),
    "slices from before a hole with no line yet hold nothing": (
        "            waiting.append(\n"
        '                "a log stream restarted without replaying (a hole), and "\n'
        '                + ", ".join(\n'
        '                    f"{len(t)} of the {name}\'s slices"',
        "            [].append(\n"
        '                "a log stream restarted without replaying (a hole), and "\n'
        '                + ", ".join(\n'
        '                    f"{len(t)} of the {name}\'s slices"',
    ),
    "suspects count from the start of the hole, not its end": (
        "    hole_end = holes[-1][1] if holes else None\n",
        "    hole_end = holes[-1][0] if holes else None\n",
    ),
    "an empty newest stream holds nothing": (
        '        report["newest_stream_empty"] = True\n        waiting.append(\n',
        '        report["newest_stream_empty"] = True\n        [].append(\n',
    ),
    "the stall check runs while the newest stream is empty": (
        "    stall = None if reopened else _stall(seen, candidate, slowest, limits, now)\n",
        "    stall = _stall(seen, candidate, slowest, limits, now)\n",
    ),
    "a hole never leaves a pool's starts unconfirmed": (
        "        and len(pool.inits) < workers\n",
        "        and False\n",
    ),
    "the hold lifts a start before every worker has logged one": (
        "        and len(pool.inits) < workers\n",
        "        and len(pool.inits) < workers - 1\n",
    ),
    "a hole that ended before a pool opened holds it": (
        "            end >= pool.opened and (released[name] is None or start < released[name])\n",
        "            (released[name] is None or start < released[name])\n",
    ),
    "a hole that began after a pool released holds it": (
        "            end >= pool.opened and (released[name] is None or start < released[name])\n",
        "            end >= pool.opened\n",
    ),
    "a released incumbent stays held after the candidate's first start": (
        '        and (released[name] is None or (name == "incumbent" and not candidate.inits))\n',
        '        and (released[name] is None or name == "incumbent")\n',
    ),
    "unconfirmed starts hold no NO-GO": (
        "    if end > time_line and unconfirmed:\n",
        "    if end > time_line and False:\n",
    ),
    "the hold also holds a stall": (
        "    if stall:\n        reasons.append(stall)\n",
        "    if stall and not held:\n        reasons.append(stall)\n",
    ),
    "the start cost is the slowest logged start, not the mean": (
        "    return sum(costs) / len(costs) if costs else None\n",
        "    return max(costs) if costs else None\n",
    ),
    "a stream beginning in the second the log reached adds that second up": (
        "            best[text] = max(best.get(text, 0), count)\n",
        "            best[text] = (\n"
        "                best.get(text, 0) + count\n"
        "                if begins == reach\n"
        "                else max(best.get(text, 0), count)\n"
        "            )\n",
    ),
    "a restart repeating the last second's records is not reported": (
        "            parsed.overlaps.append(begins)\n",
        "            pass\n",
    ),
    "an opening submit that started no logged worker is explained": (
        "                unexplained.append(\n"
        '                    "an opening submit the model says starts a worker did not"\n',
        "                [].append(\n"
        '                    "an opening submit the model says starts a worker did not"\n',
    ),
    "streams are read in glob order, app-10 before app-2": (
        '        return sorted(paths, key=lambda path: int(path.stem.removeprefix("app-")))\n',
        "        return list(paths)\n",
    ),
    "the command line reads the streams in the order given": (
        "            for path in _stream_order(args.logs)\n",
        "            for path in args.logs\n",
    ),
    "an ATTEMPTS: line stands in for the BUDGET line": (
        "        and not parsed.budget_lines\n",
        "        and not parsed.budget_seconds\n",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        default=f"{shlex.quote(sys.executable)} -m pytest",
        help="command that runs pytest with the arguments after it",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="seed only the faults whose name contains this text",
    )
    args = parser.parse_args()
    chosen = {
        name: edit
        for name, edit in MUTATIONS.items()
        if not args.only or any(text in name for text in args.only)
    }
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
        for name, (old, new) in chosen.items():
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
    print(f"{len(chosen) - len(survived)} of {len(chosen)} faults caught")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
