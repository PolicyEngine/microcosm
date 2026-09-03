# microcosm-calibrate

The **representation operator** of the [microcosm](../../DESIGN.md) stack
(`import microcosm.calibrate`). The only place
[`CALIBRATED`](../microcosm-frame/src/microcosm/frame/weights.py) weights are
produced: it compiles declared facts into a sparse linear constraint system over
a [`Frame`](../microcosm-frame) and solves for the weight vector that best
reproduces them.

## What it does

1. **Declare facts as targets.** A `Target` is a known population aggregate — a
   sum of a prepared measure column on some entity, with an optional tolerance
   and provenance. Count-like facts use prepared indicator/count columns. A
   `TargetSet` groups them.
2. **Compile to a sparse system.** `build_constraint_matrix(frame, targets,
   weight_entity)` turns the targets into a `CalibrationProblem`: a CSR matrix
   `A` (one row per `(target, period)`, one column per record of the calibrated
   entity), the target vector `b`, and the initial weights. `A @ w` estimates
   every target's aggregate. Multi-period targets stack as extra rows over the
   **same** weight vector — the charter's "one weight per trajectory".
   Uncompilable targets (missing columns or invalid lengths) are **skipped and
   reported**, never dropped silently.
3. **Solve for calibrated weights.** `calibrate(frame, targets, ...)` minimizes
   **capped weighted MAPE**:
   `weighted_mean(min(abs((A @ w - b) / scale), cap))`. By default
   `scale = max(abs(target), 1)` and `cap = 10`
   (1000%). The default `method="adam"` optimizes log-weights with torch Adam,
   keeping weights strictly positive by construction (`w = exp(log_w)`).
   `method="prox"` optimizes non-negative weight ratios with a proximal
   soft-threshold step for L1 selection, so unneeded records can become exact
   zeros.
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
- **`l1_lambda`** uses `method="prox"` and adds a proximal L1 penalty on
  `mean(weight / initial_weight)`. The soft-threshold step can prune records to
  exact zero while keeping the recorded objective coefficient explicit.
- **`l2_lambda`** is an experimental soft concentration knob: positive values
  add `l2_lambda * mean((pre_gate_weight / initial_weight) ** 2)` to the loss.
  With no L0 gates this is the calibrated weight ratio; with L0 gates it is
  intentionally latent/pre-gate, so a nearly closed gate cannot hide an exploding
  underlying weight. It is useful for ESS/design-effect sweeps when
  `max_weight_ratio` is too blunt, especially with `mass="conserve"`; under
  `mass="free"` it also penalizes total weight scale. It is not a safety
  guarantee and does not replace the hard ratio cap. In the two-stage
  `calibrate_l0_refit` path, `l2_lambda` applies to both the L0 selection and
  the refit; `refit_l2_lambda` overrides the refit stage alone — the stage
  whose weights ship — so selection-only and refit-only penalties are both
  expressible.

Every result reports its weight-concentration coordinates alongside fit:
`result.effective_sample_size` (Kish ESS, `(Σw)² / Σw²`),
`result.realized_max_weight_ratio`, and `result.top_1pct_weight_share`, all
serialized into `calibration_diagnostics.json` — so an accuracy-vs-spread
frontier can be read off any run's artifact. The standalone
`effective_sample_size(weights)` scores any weight vector, e.g. a published
artifact's.

Registry-backed diagnostics use schema 7. Each target publishes structured
`source`, `variable`, and `dimensions` objects, and the artifact publishes a
top-level dimension dictionary. The `source.id` remains the stable provider
identifier, while `source.label` comes from separate country-owned provider
label mappings in `microcosm.calibrate.provider_labels`; labels are not copied
into Chronicle facts or repeated in target-reference metadata. Ledger geography
metadata becomes one typed geography dimension per level (for example,
`geography_country` or `geography_state`), with stable geography identifiers,
producer-owned labels, and deterministic value order. Ledger filter and layout
dimensions remain separate non-geographic dimensions. This applies to every
country release that passes its `TargetRegistry`, including the UK and US
release builders. Calls without a registry retain legacy target identity fields
because they do not provide enough declared information to construct structured
identities.

Schema 7 also separates the statistic category from its measurement. For
legacy Ledger concepts whose declared unit agrees with a trailing `_count` or
`_amount`, the suffix is represented as `variable.measure` (`count` or `total`)
instead of remaining in `variable.id`. For example,
`hmrc.spi_employment_income_count` and
`hmrc.spi_employment_income_amount` both use the variable identifier
`spi_employment_income`; their measure values remain distinct. An explicit
`diagnostic_variable_id` or `variable` metadata value always takes precedence
and is not rewritten.

## Example

```python
from microcosm.calibrate import Target, TargetSet, calibrate

targets = TargetSet([
    Target(name="population", entity="household", measure="household_count",
           value=330_000_000, source="Census 2024 estimate"),
    Target(name="total_income", entity="household", measure="income",
           value=23_000_000_000_000, tolerance=1e11),
])

result = calibrate(frame, targets, weight_entity="household",
                   max_weight_ratio=10.0)
calibrated_frame = result.frame          # CALIBRATED weights
result.fraction_within_10pct             # representation quality
```

## Why a shard

`microcosm-calibrate` pulls **torch** and sparse/L0 solvers; an analyst doing
only imputation should never install them (and vice versa for
[`microcosm-fit`](../microcosm-fit)'s scikit-learn / quantile-forest). The shard
split earns its keep on disjoint heavy dependencies, per the charter.

Importing the shard asserts kernel compatibility (`microcosm-frame` 0.1.x) — the
constellation gate from `DESIGN.md`.
