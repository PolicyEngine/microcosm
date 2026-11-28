"""The estimate's replay on the one real pool log: predicted against actual end.

A measurement of the model on CPython's pool, not of Modal. The log is the
pinned scorer's incumbent pool with three workers on the Mac
(packages/microcosm-build/tests/fixtures/us_h2h/); the candidate's pool never
opened (see README). Every 600 s from the pool's opening, as the watcher
polls, it rebuilds the pool from the lines up to then, replays it forward
with the paces measured so far (unmeasured chunks scaled by
chunk-weights.json) and prints the predicted last slice against the actual.

usage: python -I scripts/replay_pool_log.py CHECKOUT
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

checkout = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "us_head_to_head_projection", checkout / "tools" / "us_head_to_head_projection.py"
)
projection = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = projection
spec.loader.exec_module(projection)

log = (
    checkout
    / "packages/microcosm-build/tests/fixtures/us_h2h/pool-run-3-workers-2cfe3043d.log"
)
events = projection.parse_log(log.read_text().splitlines()).events
prior = projection._prior(checkout / "docs/evidence/us-h2h-speed/chunk-weights.json")
WORKERS = 3
full = projection._pool(events, "incumbent")
assert full.opened is not None and full.slices_per_chunk and full.chunks
actual = max(full.by_task.values())
span = actual - full.opened
print(
    f"pool opened {projection._utc(full.opened)}; last slice {projection._utc(actual)}"
)
print(f"({span:.0f} s for {full.done} slices on {WORKERS} workers)")
print(" poll s  done  chunks measured  predicted last slice   error s  error %")
at = full.opened + 600
while at < actual:
    pool = projection._pool([e for e in events if e.at <= at], "incumbent")
    history = projection.reconstruct(pool, WORKERS, at)
    timing = projection.timing(pool, history, 3)
    if not timing.chunk_seconds or history is None:
        print(f"{at - full.opened:7.0f} {pool.done:5d}  WAIT")
    else:
        seconds, _ = projection._filled(
            timing.chunk_seconds, full.chunks, prior.incumbent
        )
        last, _ = projection.replay(
            now=at,
            workers=WORKERS,
            total=full.chunks * full.slices_per_chunk,
            per_chunk=full.slices_per_chunk,
            seconds=seconds,
            start_cost=timing.start_cost or 0.0,
            first_extra=timing.first_extra(seconds) or 0.0,
            history=history,
            finished={t: when for t, when in pool.by_task.items() if when <= at},
        )
        measured = ",".join(str(c) for c in sorted(timing.chunk_seconds))
        print(
            f"{at - full.opened:7.0f} {pool.done:5d}  {measured:15s}  "
            f"{projection._utc(last)}  {last - actual:+8.0f}  {100 * (last - actual) / span:+6.1f}"
        )
    at += 600
