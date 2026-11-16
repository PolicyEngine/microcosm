"""Time the real scorer's slices for both d844 artifacts, interleaved, untraced.

Purpose: give review r2 (PROJ-1, N1) a measured per-chunk candidate/incumbent
slice-time ratio, and a candidate-based Mac estimate, without Modal. Loads the
yardstick once and both artifacts once, then for each registry chunk scores
incumbent and candidate slices in alternation (so host-load drift hits both
alike). One process, no tracing. Writes the report after every slice.

usage: python -I time_slices.py ROOT INCUMBENT_H5 CANDIDATE_H5 LEDGER OUT_JSON
ROOT is a microcosm checkout at the plan's scorer pin (2cfe3043d).
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

ROOT, INCUMBENT, CANDIDATE, LEDGER, OUT = map(Path, sys.argv[1:6])
sys.path[:0] = [
    str(ROOT / "tools"),
    *map(str, sorted((ROOT / "packages").glob("*/src"))),
]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np  # noqa: E402, F401 - loaded after the thread limits above
import psutil  # noqa: E402

BATCH = 2000
ABORT_RSS = 28 * 1024**3  # shared host: stop well before trouble
process = psutil.Process()
peak = {"rss": 0}


def watch() -> None:
    while True:
        time.sleep(0.5)
        rss = process.memory_info().rss
        peak["rss"] = max(peak["rss"], rss)
        if rss >= ABORT_RSS:
            os._exit(75)


threading.Thread(target=watch, daemon=True).start()
report: dict = {
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "root": str(ROOT),
    "batch": BATCH,
    "python": sys.version,
    "phases": [],
    "slices": [],
}


def save() -> None:
    report["max_sampled_rss_gib"] = round(peak["rss"] / 1024**3, 3)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


def phase(name: str, began: float) -> None:
    report["phases"].append(
        {
            "phase": name,
            "seconds": round(time.monotonic() - began, 1),
            "rss_gib": round(process.memory_info().rss / 1024**3, 3),
        }
    )
    save()
    print(json.dumps(report["phases"][-1]), flush=True)


t = time.monotonic()
import score_us_release_head_to_head as scorer  # noqa: E402

phase("import", t)
t = time.monotonic()
yardstick = scorer.compile_yardstick(
    ledger_facts=LEDGER,
    age_targets=False,
    allow_unaged_dollar_targets=True,
    congressional_district_vintage_crosswalk=(
        scorer.release.default_congressional_district_vintage_crosswalk_path()
    ),
)
phase("compile_yardstick", t)
frames = {}
for name, path in (("incumbent", INCUMBENT), ("candidate", CANDIDATE)):
    t = time.monotonic()
    artifact = scorer.load_artifact(path)
    frame, _ = scorer.release._with_base_population_mass_repair(artifact.frame)
    del artifact
    batches = tuple(
        scorer.release._household_position_batches(frame.n("household"), BATCH)
    )
    frames[name] = (frame, batches)
    phase(f"load_{name}", t)

specs_all = yardstick.registry.specs
chunk_size = scorer.MATERIALIZE_SCORE_CHUNK_SPECS
chunks = max(1, -(-len(specs_all) // chunk_size))
report["spec_count"] = len(specs_all)
report["chunks"] = chunks
report["slices_per_artifact"] = {k: len(v[1]) for k, v in frames.items()}
save()


def score(name: str, chunk: int, index: int, role: str) -> None:
    frame, batches = frames[name]
    start = (chunk - 1) * chunk_size
    specs = specs_all[start : start + chunk_size]
    weights = yardstick.loss_weights[start : start + len(specs)]
    wall, cpu = time.monotonic(), time.process_time()
    result = scorer._score_household_slice(
        frame,
        specs,
        batches[index],
        chunk_loss_weights=weights,
        artifact_name=name,
        chunk_label=f"chunk {chunk}/{chunks}",
        slice_index=index,
        slice_count=len(batches),
        maximum_microsim_batch_size=BATCH,
    )
    row = {
        "artifact": name,
        "chunk": chunk,
        "slice": index + 1,
        "role": role,
        "spec_count": len(specs),
        "wall_s": round(time.monotonic() - wall, 2),
        "cpu_s": round(time.process_time() - cpu, 2),
        "rss_gib": round(process.memory_info().rss / 1024**3, 3),
        "ru_maxrss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3
        ),
        "load1": round(os.getloadavg()[0], 1),
        "score_sha256": hashlib.sha256(
            result.estimates.tobytes() + result.targets.tobytes()
        ).hexdigest()[:16],
    }
    del result
    report["slices"].append(row)
    save()
    print(json.dumps(row), flush=True)


# One warm-up slice each in chunk 1 (engine caches fill on first use).
score("incumbent", 1, 0, "warmup")
score("candidate", 1, 0, "warmup")
picks = {"incumbent": (7, 21), "candidate": (44, 132)}
for chunk in range(1, chunks + 1):
    for rep in range(2):
        score("incumbent", chunk, picks["incumbent"][rep], "timed")
        score("candidate", chunk, picks["candidate"][rep], "timed")
report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
