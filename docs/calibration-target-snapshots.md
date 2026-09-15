# Calibration target snapshots

An opt-in observer exposes target estimates already computed during an Adam
solve. It also records the estimates on the weights actually returned. Use it
to inspect fit over time without adding another model evaluation or changing
the optimization steps.

```python
from pathlib import Path

from microcosm.calibrate import (
    TargetSnapshotCadence,
    TargetSnapshotObserver,
    TargetSnapshotWriter,
    calibrate,
)

observer = TargetSnapshotObserver(
    sink=TargetSnapshotWriter(Path("runs/example/targets"), history_limit=256),
    run_id="example",
    cadence=TargetSnapshotCadence(every=25),
)
result = calibrate(frame, targets, method="adam", target_snapshots=observer)
```

The caller supplies its existing frame and targets. No observer is enabled by
default. The example cadence is a caller choice, not a measured country-build
default. A sink exception propagates, matching the existing progress callback.

Each snapshot has an ordered target identity, target values, achieved estimates,
relative errors, sequence, phase and iterate labels. Duplicate target names
remain distinguishable by ordered position. Honest nonfinite intermediate
estimates are represented by nulls and counts. Intermediate float32 estimates
and final float64 estimates are labeled with their actual iterate semantics.
Budget probes and subsequent selection/refit phases share one emission sequence.
`epoch` counts completed optimizer updates: zero is the starting state. An
in-loop estimate precedes that iteration's update; its selected counterpart uses
the same convention. Cadence counts loss evaluations starting at one, so an
every-25 cadence can emit epoch 24. The selected budget-search snapshot retains
the winning probe's identity, even when another probe ran afterward. If computing
a relative error overflows, the error is null and `non_finite_rows` counts it;
finite operands remain visible.

Metadata accepts bounded flat scalar mappings and checked string identifiers;
it does not accept nested payloads or record vectors. `best_retained` must be a
complete `{available, epoch, loss}` mapping in a serialized snapshot. The emitter
can fill that mapping when no best iterate exists; the public validator rejects
a serialized null in its place. Each delivered payload is detached from the
caller's metadata and subsequent snapshots.

The local writer atomically replaces `latest.json`. History chunks become
visible only after their complete bytes are written and synced, and an existing
chunk cannot be overwritten. Retention and dropped-history counts are explicit.
Use a fresh run directory unless intentionally adopting existing history.
Atomic history publication requires a filesystem that supports hard links and
directory synchronization. Unsupported filesystem errors propagate to the caller;
the writer does not expose partially written history as a fallback.

This implements the solver and local storage portion of
[issue 908](https://github.com/PolicyEngine/microcosm/issues/908). Country-host
wiring, staging upload, a dashboard consumer and native cadence/performance
acceptance remain separate work. The grouped US solver currently lives on the
launch integration branch and needs its own integration of this observer.

The reviewed repair branch passed 307 calibration tests before the final strict
null correction. All 52 snapshot tests then passed after that one-line correction,
including passive solver parity tests. Independent review closed the metadata,
detachment, partial-publication and codec findings. See the
[invented benchmark evidence](../experiments/908-target-snapshot-bench-receipts.md)
for overhead measurements; those are not native US or UK benchmarks.

Fable's later PR review identified a mixed epoch convention, a missing winning
budget-probe label and a relative-error overflow case. Four new counterexamples
failed first; all 56 snapshot tests and all 311 calibration tests then passed
after those corrections. The 26 affected build identity and graph parity checks
also passed after recalculating the calibration source pins; fit and simulation
pins were preserved. These changes affect diagnostics only. The public writer still validates each incoming
payload because callers can supply serialized snapshots independently of an
observer.
