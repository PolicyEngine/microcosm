"""Bounded invented numerical checker timing; no source or target inputs."""

from __future__ import annotations

import json
import time

import numpy as np

from microcosm.calibrate.group_bounds import GroupedUpperBounds

n = 20_000
weights = np.full(n, 40.0, dtype=np.float64)
started = time.perf_counter()
groups = GroupedUpperBounds(
    tuple(range(n)), np.repeat(np.arange(n // 2), 2), np.full(n // 2, 200.0)
)
construction = time.perf_counter() - started
timings = []
for _ in range(3):
    started = time.perf_counter()
    totals = groups.check(weights)
    timings.append(time.perf_counter() - started)
assert np.all(totals == 80.0)
print(
    json.dumps(
        {
            "invented_households": n,
            "origins": n // 2,
            "repetitions": 3,
            "construction_seconds": construction,
            "check_seconds": timings,
            "median_check_seconds": float(np.median(timings)),
            "constraint_digest": groups.constraint_digest,
        },
        sort_keys=True,
    )
)
