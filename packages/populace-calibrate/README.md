# populace-calibrate

The **representation operator** of the [populace](../../DESIGN.md) stack
(`import populace.calibrate`). The only place
[`CALIBRATED`](../populace-frame/src/populace/frame/weights.py) weights are
produced: it compiles declared facts into a sparse linear constraint system over
a [`Frame`](../populace-frame) and solves for the weight vector that best
reproduces them.

## What it does

1. **Declare facts as targets.** A `Target` is a known population aggregate — a
   control total (`sum`), a count, or an average (`mean`) — on some entity, with
   an optional tolerance and provenance. A `TargetSet` groups them.
2. **Compile to a sparse system.** `build_constraint_matrix(frame, targets,
   weight_entity)` turns the targets into a `CalibrationProblem`: a CSR matrix
   `A` (one row per `(target, period)`, one column per record of the calibrated
   entity), the target vector `b`, and the initial weights. `A @ w` estimates
   every target's aggregate. Multi-period targets stack as extra rows over the
   **same** weight vector — the charter's "one weight per trajectory".
   Uncompilable targets (missing column, zero `mean` denominator) are **skipped
   and reported**, never dropped silently.
3. **Solve for calibrated weights.** `calibrate(frame, targets, ...)` optimizes
   the log-weights with torch Adam to minimize **capped weighted MAPE**:
   `weighted_mean(min(abs((A @ w - b) / scale), cap))`. By default
   `scale = max(abs(target), abs(initial_estimate), 1)` and `cap = 10`
   (1000%). Weights stay strictly positive by construction (`w = exp(log_w)`).
   The result carries a new `Frame` with `CALIBRATED` weights, per-target
   diagnostics, and the loss trajectory.

## Load-bearing options

- **`mass="free"`** (default) lets the total weight move to fit the targets;
  **`mass="conserve"`** projects every step back to the input total, so the
  calibrated population conserves the starting mass exactly. `"free"` records a
  `MassChange` on the frame's mass log; `"conserve"` uses the kernel's
  conservation check.
- **`max_weight_ratio`** is a **hard** per-record bound — no calibrated weight
  exceeds `max_weight_ratio * initial_weight`, clamped after every step. This is
  the documented guard against the tail **landmine**: a rare high-value,
  near-zero-weight donor whose weight detonates on reweight and blows up an
  aggregate (the $201T-scale failure the charter exists to prevent).
- **`target_records`** turns on hard-concrete L0 gates with **budget control**:
  the solver searches `l0_lambda` so the
  achieved non-zero count tracks the record budget, and reports the penalty it
  settled on (`result.l0_lambda`) — the generate-big-then-prune path
  (300k → 3M → 30M pools). A supplied `l0_lambda` warm-starts the search.
- **`l0_lambda`** alone (no `target_records`) prunes at a fixed penalty: `> 0`
  gates the pool, `0.0` keeps every record.

## Example

```python
from populace.calibrate import Target, TargetSet, calibrate

targets = TargetSet([
    Target(name="population", entity="household", aggregation="count",
           value=330_000_000, source="Census 2024 estimate"),
    Target(name="total_income", entity="household", measure="income",
           aggregation="sum", value=23_000_000_000_000, tolerance=1e11),
])

result = calibrate(frame, targets, weight_entity="household",
                   max_weight_ratio=10.0)
calibrated_frame = result.frame          # CALIBRATED weights
result.fraction_within_10pct             # representation quality
```

## Why a shard

`populace-calibrate` pulls **torch** and sparse/L0 solvers; an analyst doing
only imputation should never install them (and vice versa for
[`populace-fit`](../populace-fit)'s scikit-learn / quantile-forest). The shard
split earns its keep on disjoint heavy dependencies, per the charter.

Importing the shard asserts kernel compatibility (`populace-frame` 0.1.x) — the
constellation gate from `DESIGN.md`.
