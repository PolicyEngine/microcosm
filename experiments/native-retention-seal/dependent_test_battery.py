"""Run every test file that imports a module this lane touched, one per process.

CI runs the build shard's groups in separate pytest processes precisely so that
one file's ``monkeypatch.setattr`` on a module-level function cannot trip
``survey_population_preparation._producer``'s live-code seal for every later
file -- the native-scale lane's report section 9a is the write-up of what
happens when they share one. This runs them the way CI does, with a bounded
number of processes at a time, and records each file's own summary line.

    python experiments/native-retention-seal/dependent_test_battery.py OUT.json [JOBS]
"""

import concurrent.futures
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]  # the worktree root
PYTHON = ROOT / ".venv/bin/python"
FILES = sorted(
    str(path.relative_to(ROOT))
    for path in (ROOT / "packages/microcosm-build/tests").glob("test_*.py")
    if any(
        marker in path.read_text(encoding="utf-8")
        for marker in (
            "survey_population_replay",
            "graph_atomic_survey_financial",
            "graph_implementation",
        )
    )
)


def run(relative):
    started = time.monotonic()
    completed = subprocess.run(
        [str(PYTHON), "-m", "pytest", relative, "-p", "no:randomly"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = [line for line in completed.stdout.splitlines() if line.strip()]
    summary = next(
        (
            line
            for line in reversed(tail)
            if " passed" in line or " failed" in line or " error" in line
        ),
        "<no summary line>",
    )
    return {
        "file": relative,
        "returncode": completed.returncode,
        "summary": summary.strip(),
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def main():
    out = pathlib.Path(sys.argv[1])
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for result in pool.map(run, FILES):
            results.append(result)
            print(
                f"{'ok ' if result['returncode'] == 0 else 'FAIL'} "
                f"{result['file']}  {result['summary']}",
                flush=True,
            )
    out.write_text(
        json.dumps(
            {
                "scope": "diagnostic; not a build, certification or release artifact",
                "release_eligible": False,
                "invocation": "one pytest process per file, as CI runs them",
                "files": len(results),
                "failing_files": [r["file"] for r in results if r["returncode"] != 0],
                "results": sorted(results, key=lambda r: r["file"]),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"\nfiles={len(results)} failing={len([r for r in results if r['returncode']])}"
    )


if __name__ == "__main__":
    main()
