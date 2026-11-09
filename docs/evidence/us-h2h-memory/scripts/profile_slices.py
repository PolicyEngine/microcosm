"""Measure the real scorer on consecutive incumbent slices in one process.

Run with the pinned routea Python; all Microcosm imports use this checkout.
Only aggregate scores, allocation locations, and object counts are written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import time
import tracemalloc
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "tools"), *map(str, (ROOT / "packages").glob("*/src"))]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--reform-measures", type=int)
    parser.add_argument("--without-cleanup", action="store_true")
    args = parser.parse_args()
    process = psutil.Process()
    maximum_rss = 0
    stop = threading.Event()

    def watch() -> None:
        nonlocal maximum_rss
        while not stop.wait(0.2):
            rss = process.memory_info().rss
            maximum_rss = max(maximum_rss, rss)
            # Leave room below the operator's 16 GiB cap for an allocation
            # between samples; abort without continuing a large calculation.
            if rss >= 14 * 1024**3:
                os._exit(75)

    threading.Thread(target=watch, daemon=True).start()
    print(json.dumps({"phase": "engine_import", "pid": os.getpid()}), flush=True)
    import score_us_release_head_to_head as scorer

    yardstick = scorer.compile_yardstick(
        ledger_facts=args.ledger,
        age_targets=False,
        allow_unaged_dollar_targets=True,
        congressional_district_vintage_crosswalk=(
            scorer.release.default_congressional_district_vintage_crosswalk_path()
        ),
    )
    artifact = scorer.load_artifact(args.incumbent)
    frame, _ = scorer.release._with_base_population_mass_repair(artifact.frame)
    start = (args.chunk - 1) * scorer.MATERIALIZE_SCORE_CHUNK_SPECS
    specs = yardstick.registry.specs[
        start : start + scorer.MATERIALIZE_SCORE_CHUNK_SPECS
    ]
    loss_weights = yardstick.loss_weights[start : start + len(specs)]
    if args.reform_measures is not None:
        requested = [
            reform.measure
            for reform in scorer.release.US_JCT_TAX_EXPENDITURE_REFORMS
            if reform.measure in {spec.measure for spec in specs}
        ]
        excluded = set(requested[args.reform_measures :])
        selected = [i for i, spec in enumerate(specs) if spec.measure not in excluded]
        specs = tuple(specs[i] for i in selected)
        loss_weights = loss_weights[selected]
    score_slice = scorer._score_household_slice
    if args.without_cleanup:
        score_slice = score_slice.__wrapped__
    tracemalloc.start(1)

    def census() -> dict:
        gc.collect()
        counts = Counter(
            f"{type(obj).__module__}.{type(obj).__qualname__}"
            for obj in gc.get_objects()
        )
        return {
            "rss_bytes": process.memory_info().rss,
            "traced_bytes": tracemalloc.get_traced_memory()[0],
            "sys_modules": len(sys.modules),
            "variable_modules": sum(
                "/variables/" in str(getattr(module, "__file__", ""))
                and "policyengine_us" in str(getattr(module, "__file__", ""))
                for module in sys.modules.values()
                if module is not None
            ),
            "type_counts": dict(counts.most_common(30)),
        }

    report = {
        "engine_versions": {
            name: importlib.metadata.version(name)
            for name in ("policyengine-us", "policyengine-core")
        },
        "batch_size": args.batch_size,
        "chunk": args.chunk,
        "spec_count": len(specs),
        "cleanup": not args.without_cleanup,
        "requested_reform_measures": [
            reform.measure
            for reform in scorer.release.US_JCT_TAX_EXPENDITURE_REFORMS
            if reform.measure in {spec.measure for spec in specs}
        ],
        "slices": [],
    }
    print(
        json.dumps({key: value for key, value in report.items() if key != "slices"}),
        flush=True,
    )
    for index in range(args.slices):
        before = census()
        snapshot = tracemalloc.take_snapshot()
        began = time.monotonic()
        result = score_slice(
            frame,
            specs,
            np.arange(index * args.batch_size, (index + 1) * args.batch_size),
            chunk_loss_weights=loss_weights,
            artifact_name="incumbent",
            chunk_label=f"chunk {args.chunk}/5",
            slice_index=index,
            slice_count=(frame.n("household") + args.batch_size - 1) // args.batch_size,
            maximum_microsim_batch_size=args.batch_size,
        )
        digest = hashlib.sha256(
            result.estimates.tobytes()
            + result.targets.tobytes()
            + result.scales.tobytes()
        ).hexdigest()
        del result
        after = census()
        differences = tracemalloc.take_snapshot().compare_to(snapshot, "lineno")[:12]
        del snapshot
        row = {
            "slice": index + 1,
            "seconds": time.monotonic() - began,
            "before": before,
            "after": after,
            "score_sha256": digest,
            "allocation_deltas": [str(stat) for stat in differences],
        }
        report["slices"].append(row)
        report["maximum_sampled_rss_bytes"] = maximum_rss
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {key: value for key, value in row.items() if key != "allocation_deltas"}
            ),
            flush=True,
        )
    stop.set()


if __name__ == "__main__":
    main()
