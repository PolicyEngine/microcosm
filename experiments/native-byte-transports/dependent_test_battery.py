"""Run every test file that names a module this lane touched, one per process.

CI runs the build shard's groups in separate pytest processes so that one
file's ``monkeypatch.setattr`` on a module-level function cannot trip
``survey_population_preparation._producer``'s live-code seal for every later
file; the native-scale lane's report §9a is the write-up of what happens when
they share one. This runs them the way CI does, with a bounded number of
processes at a time, and records each file's own summary line. The pattern is
the retention lane's ``dependent_test_battery.py``; the markers are this lane's.

    python experiments/native-byte-transports/dependent_test_battery.py OUT.json [JOBS]
"""

import concurrent.futures
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]  # the worktree root
PYTHON = ROOT / ".venv/bin/python"
MARKERS = (
    "acs_person_coverage_authentication",
    "acs_native_coverage_binding",
    "acs_population_catalogue",
    "survey_population_preparation",
    "graph_survey_population",
    "survey_origin_budget",
    "graph_survey_budget",
    "graph_survey_calibration",
    "graph_survey_age_artifact",
    "current_survey_predictors",
    "graph_child_property_income",
    "current_child_property_income_source",
    "graph_joint_empirical",
)
FILES = sorted(
    str(path.relative_to(ROOT))
    for package in ("microcosm-build", "microcosm-fit")
    for path in (ROOT / "packages" / package / "tests").glob("test_*.py")
    if any(marker in path.read_text(encoding="utf-8") for marker in MARKERS)
)


def run(relative):
    started = time.monotonic()
    completed = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            relative,
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
        ],
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
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "packages"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    results = []
    print(f"head={head} dirty_packages={'yes' if dirty else 'no'} files={len(FILES)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for result in pool.map(run, FILES):
            results.append(result)
            print(
                f"{'ok ' if result['returncode'] == 0 else 'FAIL'} "
                f"{result['file']}  {result['summary']}  ({result['wall_seconds']}s)",
                flush=True,
            )
    out.write_text(
        json.dumps(
            {
                "scope": "diagnostic; not a build, certification or release artifact",
                "release_eligible": False,
                "invocation": "one pytest process per file, as CI runs them",
                "head": head,
                "packages_dirty_at_start": bool(dirty),
                "markers": list(MARKERS),
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
