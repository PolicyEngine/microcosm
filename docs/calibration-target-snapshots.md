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

## The grouped US solver

The grouped/fixed-zero Adam path (`grouped_upper_bounds`, optionally
`grouped_preserve_zeros`) is instrumented at the same two seams and with the
same guarantees: the in-loop snapshot reads the exact float32 estimate tensor
the epoch's loss was computed from, and the closing `selected` snapshot reads
the same float64 `problem.estimates` the final diagnostics use. Enabling the
observer on a grouped run adds no matrix evaluation, no model evaluation and no
RNG advance, and returns bit-identical weights and trajectory.

Four grouped specifics a consumer must read correctly:

- Grouped Adam is a **closing-state** algorithm. It never runs the retain-best
  rule, so every grouped snapshot carries
  `best_retained = {"available": false, "epoch": null, "loss": null}` and the
  closing snapshot is stamped `epoch == epochs`. That is not the same as an
  ordinary run whose retain-best rule was switched off, so grouped snapshots
  also carry bounded `selection` labels: `rule`, `constraint_mode`,
  `grouped_preserve_zeros` and `epoch_convention`. Only `rule` mirrors a value
  in the result — it is exactly `options["iterate_selection"]`, i.e.
  `"closing_state"`. `constraint_mode` has no `options` counterpart of that
  name, and the snapshot's `grouped_preserve_zeros` is a bare boolean while
  `options["grouped_preserve_zeros"]` is a dict that is present only when the
  option is on.
- **`selection` is populated in two shapes and absent in a third**, so read it
  defensively. `rule: "best_feasible_loss"` is the ordinary retain-best receipt
  (with `selected_epoch`, `epochs_executed`, `epoch_convention`,
  `selected_loss_float32`, `closing_iterate_loss_float32`), attached to the
  closing row only. `rule: "closing_state"` is the grouped mode label set,
  attached to every grouped row including the in-loop ones. But an **ungrouped**
  closing-state run — gated, mass-conserving, or `l2_lambda > 0`, where
  `retain_best` is off — leaves `selection` `null` on every row even though its
  `options["iterate_selection"]` is also `"closing_state"`. So `selection` tells
  you the selection rule on a grouped or retain-best run and nothing at all on
  an ungrouped closing-state run, and `selection is not None` does not identify
  the returned fit — `iterate == "selected"` does. The schema version stays `1`
  because no envelope field was added, removed or retyped: `selection` has
  always been an open bounded scalar mapping that may be `null`, and the codec
  accepts all three cases by construction.
- In-loop snapshots are **pre-update loss evaluations**, the same convention the
  ungrouped Adam loop uses, and the relationship to the private
  `_post_projection_observer` proof seam is a **systematic one-epoch lag**, not
  merely a different vector. At the top of loop iteration `e` the log
  parameters are the ones that iteration `e-1`'s single overwrite wrote from
  the accepted state the proof seam reported at `epoch e`; the row built from
  them is stamped `epoch = e + 1`. So grouped snapshot `epoch = k` is the fit
  of accepted state `k - 1`, up to an `exp(log(x))` round trip and the float32
  cast — **not** of the accepted state the proof seam stamps `epoch = k`. Two
  consequences at full cadence: accepted state 0 gets no snapshot of its own
  (synthesizing one from the weight-only callback would need an extra matrix
  evaluation, so it is deliberately not done), and `epoch == epochs` is stamped
  twice for two states one update apart — the `current` row (state
  `epochs - 1`) and the `selected` row (state `epochs`). `iterate` and
  `epoch_convention` are what separate them.
- The private seam carries record-length weights, household IDs, the group map
  and the absolute-bound vector; none of it reaches a snapshot, and the public
  codec rejects record-level key names outright.

Grouped runs keep every existing guard: fixed zeros stay in full-population
coordinates, the accepted-weight byte equality and ordered-household-ID checks
still run *after* the closing snapshot sink, and the unsupported grouped modes
(scalar cap, conserved mass, prox, L0/exact-k, gate initialization) are still
refused before the optimizer is constructed. Snapshot support does not turn any
of them into a permitted numeric path.

**A `selected` snapshot is not proof that the run finished.** That guard order
is deliberate — emitting after the checks would open a late window in which a
sink could mutate the returned weights without anything revalidating them — but
it means the sink is handed the closing payload *before* the admission checks
run. A grouped run that then trips one of them raises and returns no
`CalibrationResult`, while a `TargetSnapshotWriter` sink has already written
that `selected` payload into `latest.json`. A consumer must take completion from
the run's own exit, not from the presence of a `selected` snapshot.

This implements the solver and local storage portion of
[issue 908](https://github.com/PolicyEngine/microcosm/issues/908). Country-host
wiring, staging upload, a dashboard consumer and native cadence/performance
acceptance remain separate work. In particular
`us_runtime/graph_fiscal_dense_calibration.py` still calls `calibrate` without
an observer: a solver-side integration alone produces no snapshot files for
that path, and wiring it needs an explicit host-owned instrumentation seam
rather than a callback or filesystem path serialized into graph node params.

The reviewed repair branch passed 307 calibration tests before the final strict
null correction. All 52 snapshot tests then passed after that one-line correction,
including passive solver parity tests. Independent review closed the metadata,
detachment, partial-publication and codec findings. See the
[invented benchmark evidence](../experiments/908-target-snapshot-bench-receipts.md)
for overhead measurements; those are not native US or UK benchmarks.
