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

## The grouped US solver

The grouped/fixed-zero Adam path (`grouped_upper_bounds`, optionally
`grouped_preserve_zeros`) is instrumented at the same two seams and with the
same guarantees: the in-loop snapshot reads the exact float32 estimate tensor
the epoch's loss was computed from, and the closing `selected` snapshot reads
the same float64 `problem.estimates` the final diagnostics use. Enabling the
observer on a grouped run adds no matrix evaluation, no model evaluation and no
RNG advance, and returns bit-identical weights and trajectory.

Two grouped specifics a consumer must read correctly:

- Grouped Adam is a **closing-state** algorithm. It never runs the retain-best
  rule, so every grouped snapshot carries
  `best_retained = {"available": false, "epoch": null, "loss": null}` and the
  closing snapshot is stamped `epoch == epochs`. That is not the same as an
  ordinary run whose retain-best rule was switched off, so grouped snapshots
  also carry bounded `selection` labels — `rule: "closing_state"`,
  `constraint_mode: "grouped_upper_bounds"` and `grouped_preserve_zeros` —
  which mirror the run's `options["iterate_selection"]`.
- In-loop snapshots are **pre-update loss evaluations**, the same convention the
  ungrouped Adam loop uses. They are deliberately not the post-update projected
  accepted vector that the private `_post_projection_observer` proof seam
  reports after the next completed update: those are different vectors, and
  producing target totals for the accepted one would require an extra matrix evaluation.
  That private seam carries record-length weights, household IDs, the group map
  and the absolute-bound vector; none of it reaches a snapshot, and the public
  codec rejects record-level key names outright.

Grouped runs keep every existing guard: fixed zeros stay in full-population
coordinates, the accepted-weight byte equality and ordered-household-ID checks
still run *after* the closing snapshot sink, and the unsupported grouped modes
(scalar cap, conserved mass, prox, L0/exact-k, gate initialization) are still
refused before the optimizer is constructed. Snapshot support does not turn any
of them into a permitted numeric path.

## US fiscal host

The actual dense fiscal kernel accepts the shared observer through an optional,
keyword-only constructor argument:

```python
from microcosm.build.us_runtime.graph_fiscal_dense_calibration import (
    FiscalDenseCalibrationKernel,
)
from microcosm.calibrate import TargetSnapshotCadence, TargetSnapshotObserver

observer = TargetSnapshotObserver(
    sink=host_owned_sink,
    run_id="fiscal-run",
    candidate_id="candidate-under-review",
    cadence=TargetSnapshotCadence(every=25),
)
kernels.register(FiscalDenseCalibrationKernel(target_snapshots=observer))
```

The host owns the sink and any local writer directory. The kernel retains the
observer only on its instance and forwards it to the existing `calibrate` call.
Omitting the argument or passing `None` disables observation. Observer values,
callbacks and paths are absent from node parameters, implementation identity,
cache keys, graph artifacts and receipts. Changing this source implementation
changes its fingerprint normally; changing observer configuration does not.

The solver binds snapshot identity to the actual compiled target order and
values from the fiscal measurement. The existing final matrix/target, household
ID, weight and measurement checks run after the sink completes. Sink exceptions
propagate, as with the ordinary progress callback. Each delivered dictionary is
detached: mutating its aggregate metadata or target rows cannot change the next
payload or optimizer result. This does not make an arbitrary callback safe to
mutate unrelated live application state.

A required graph cache hit does not execute the optimizer and emits no target
snapshots, even when a new observer is supplied. Hosts should retain the previous
run's diagnostics and history with their identities; a replay must not be shown
as a new optimizer run. Complete-population and source admission remain the
country host's responsibility.

This implements the solver, local storage and dense fiscal-host portions of
[issue 908](https://github.com/PolicyEngine/microcosm/issues/908). Staging upload,
a dashboard consumer and native cadence/performance acceptance remain separate
work. See [the fiscal-host integration evidence](../experiments/fiscal-target-snapshot-host-20260912.md).

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
pins were preserved. These changes affect diagnostics only. The public writer
still validates each incoming payload because callers can supply serialized
snapshots independently of an observer.
