#!/usr/bin/env python3
"""SYNTHETIC benchmark for the #908 per-target snapshot path.

Every number this script produces comes from **invented** target matrices and
an **invented** frame built from a seeded RNG. Nothing here reads native
microdata, runs a country engine, or touches a network. The dimension points
below were chosen to bracket an order of magnitude; they are NOT measured from
the real US or UK calibration target registries, and the timings are NOT US or
UK calibration times or upload times. Read them as "what does the codec and the
local store cost per target row and per epoch", nothing more.

Run (from the repo root, inside the isolated lane interpreter)::

    python experiments/908_target_snapshot_bench.py

Bounded on purpose: one numeric thread, a 2 GiB address-space cap and a 120 s
CPU-time cap are set before torch loads, so the script cannot quietly grow into
a machine-sized job.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

_MEMORY_BYTES = 2 * 1024**3


def _bound_resource(which: int, value: int) -> int | None:
    """Apply a soft rlimit where the platform allows it; report what stuck."""
    soft, hard = resource.getrlimit(which)
    if hard != resource.RLIM_INFINITY and value > hard:
        return None
    try:
        resource.setrlimit(which, (value, hard))
    except (ValueError, OSError):
        return None
    return value


_CPU_LIMIT = _bound_resource(resource.RLIMIT_CPU, 120)
_MEMORY_LIMIT = _bound_resource(resource.RLIMIT_AS, _MEMORY_BYTES)
if _MEMORY_LIMIT is None:
    # macOS refuses RLIMIT_AS here; RLIMIT_DATA is the portable fallback.
    _MEMORY_LIMIT = _bound_resource(resource.RLIMIT_DATA, _MEMORY_BYTES)
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402 - after the resource and thread bounds
import pandas as pd  # noqa: E402 - after the resource and thread bounds
import torch  # noqa: E402 - after the resource and thread bounds

from microcosm.calibrate import (  # noqa: E402 - after the resource bounds
    EVERY_EPOCH,
    Target,
    TargetSet,
    TargetSnapshotCadence,
    TargetSnapshotObserver,
    TargetSnapshotWriter,
    calibrate,
)
from microcosm.frame import (  # noqa: E402 - after the resource bounds
    EntitySchema,
    Frame,
    WeightKind,
    Weights,
)

torch.set_num_threads(1)

#: Invented target-count dimension points for the codec measurement.
TARGET_COUNTS = (500, 2_000, 5_000, 10_000)

#: Invented solver dimension point: households x targets x epochs.
SOLVER_RECORDS = 10_000
SOLVER_TARGETS = 100
SOLVER_EPOCHS = 300

#: Timed repeats per cadence; the table reports the best (least noisy) run.
REPEATS = 3

_SCHEMA = EntitySchema(group_entities=("household",))


def _bound(n_targets: int):
    """A bound observer over ``n_targets`` invented targets, sinking to nowhere."""
    rng = np.random.default_rng(0)
    names = tuple(f"synthetic_target_{index:06d}@2024" for index in range(n_targets))
    targets = rng.lognormal(12.0, 1.0, n_targets)
    observer = TargetSnapshotObserver(sink=lambda payload: None, run_id="synthetic")
    return observer.bind(names=names, targets=targets), rng.lognormal(
        12.0, 1.0, n_targets
    )


def codec_costs() -> list[dict[str, object]]:
    """Build/validate/serialize/write cost per snapshot, by target count."""
    rows: list[dict[str, object]] = []
    for n_targets in TARGET_COUNTS:
        bound, estimates = _bound(n_targets)
        repeats = max(3, 30_000 // n_targets)
        start = time.perf_counter()
        for _ in range(repeats):
            # snapshot() validates the payload it builds, so this is the full
            # build + validate cost a real emission pays.
            payload = bound.snapshot(
                estimates, epoch=1, epochs=1, iterate="current", loss=0.1
            )
        build_seconds = (time.perf_counter() - start) / repeats

        start = time.perf_counter()
        for _ in range(repeats):
            serialized = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False)
        serialize_seconds = (time.perf_counter() - start) / repeats

        with tempfile.TemporaryDirectory() as directory:
            writer = TargetSnapshotWriter(Path(directory), history_limit=8)
            start = time.perf_counter()
            for index in range(repeats):
                copy = dict(payload)
                copy["sequence"] = index + 1
                writer(copy)
            write_seconds = (time.perf_counter() - start) / repeats

        rows.append(
            {
                "n_targets": n_targets,
                "build_and_validate_ms": round(build_seconds * 1e3, 3),
                "serialize_ms": round(serialize_seconds * 1e3, 3),
                "store_write_ms": round(write_seconds * 1e3, 3),
                "serialized_bytes": len(serialized.encode("utf-8")),
                "bytes_per_target": round(
                    len(serialized.encode("utf-8")) / n_targets, 1
                ),
            }
        )
    return rows


def _synthetic_frame_and_targets() -> tuple[Frame, TargetSet]:
    rng = np.random.default_rng(1)
    household_ids = np.arange(SOLVER_RECORDS, dtype="int64")
    columns = {
        f"measure_{index:03d}": rng.lognormal(3.0, 0.5, SOLVER_RECORDS)
        for index in range(SOLVER_TARGETS)
    }
    weights = np.full(SOLVER_RECORDS, 100.0)
    household = pd.DataFrame({"household_id": household_ids, **columns})
    person = pd.DataFrame(
        {"person_id": household_ids, "person_household_id": household_ids}
    )
    frame = Frame(
        {"person": person, "household": household},
        _SCHEMA,
        {"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        [
            Target(
                name=name,
                period=2024,
                entity="household",
                measure=name,
                # An invented 8% miss, so the optimizer has somewhere to go.
                value=float((values * weights).sum()) * 1.08,
            )
            for name, values in columns.items()
        ]
    )
    return frame, targets


def solver_overhead() -> list[dict[str, object]]:
    """Added wall-clock from snapshot emission, at three cadences.

    One discarded warm-up run first (the first calibrate of a process pays
    matrix-build and torch warm-up that would otherwise be charged to whichever
    cadence happened to run first), then the best of ``REPEATS`` timed runs per
    cadence. Each run's returned weights are compared bitwise against the
    observer-off run, so the table also evidences that emission changed nothing.
    """
    frame, targets = _synthetic_frame_and_targets()
    cadences = (
        ("observer_off", None),
        ("bounded_every_25", TargetSnapshotCadence(every=25)),
        ("every_epoch", TargetSnapshotCadence(every=EVERY_EPOCH)),
    )
    calibrate(frame, targets, epochs=SOLVER_EPOCHS, seed=0)  # warm-up, discarded

    rows: list[dict[str, object]] = []
    reference: np.ndarray | None = None
    for label, cadence in cadences:
        best_seconds = float("inf")
        emitted = retained_chunks = retained_bytes = 0
        for _ in range(REPEATS):
            with tempfile.TemporaryDirectory() as directory:
                observer = (
                    None
                    if cadence is None
                    else TargetSnapshotObserver(
                        sink=TargetSnapshotWriter(Path(directory)),
                        run_id="synthetic",
                        cadence=cadence,
                    )
                )
                start = time.perf_counter()
                result = calibrate(
                    frame,
                    targets,
                    epochs=SOLVER_EPOCHS,
                    seed=0,
                    target_snapshots=observer,
                )
                best_seconds = min(best_seconds, time.perf_counter() - start)
                history = Path(directory) / "history"
                if observer is not None:
                    retained_chunks = len(list(history.glob("*.json")))
                    retained_bytes = sum(
                        path.stat().st_size for path in history.glob("*.json")
                    )
                    emitted = int(
                        json.loads(
                            (Path(directory) / "history_index.json").read_text(
                                encoding="utf-8"
                            )
                        )["last_sequence"]
                    )
        if reference is None:
            reference = result.weights.copy()
            identical = True
        else:
            identical = bool(np.array_equal(reference, result.weights))
        rows.append(
            {
                "cadence": label,
                f"best_of_{REPEATS}_seconds": round(best_seconds, 4),
                "closing_loss": float(result.closing_loss),
                "weights_bitwise_equal_to_observer_off": identical,
                "snapshots_emitted": emitted or None,
                "chunks_retained_on_disk": retained_chunks,
                "retained_bytes_on_disk": retained_bytes,
            }
        )
    return rows


def main() -> int:
    report = {
        "label": "SYNTHETIC — invented matrices and frames, not US or UK calibration",
        "dimension_points": {
            "codec_target_counts": list(TARGET_COUNTS),
            "solver_records": SOLVER_RECORDS,
            "solver_targets": SOLVER_TARGETS,
            "solver_epochs": SOLVER_EPOCHS,
        },
        "bounds": {
            "numeric_threads": torch.get_num_threads(),
            "cpu_seconds_limit_applied": _CPU_LIMIT,
            "memory_bytes_limit_applied": _MEMORY_LIMIT,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "native_inputs_read": 0,
        },
        "codec_costs": codec_costs(),
        "solver_overhead": solver_overhead(),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
