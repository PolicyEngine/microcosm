# Penalized calibration toward the design weights

`microcosm.calibrate.calibrate` has two opt-in options for spreading calibrated
weights: `l2_basis`, the functional form of the `l2_lambda` penalty, and
`mass_parametrization`, how the Adam solve holds the total under
`mass="conserve"`. Their defaults (`"record"`, `"projection"`) are the historical
solve, byte for byte. This page gives the algebra, the evidence for each claim,
and what the options can and cannot do for the ACS local-area release. The
full-scale sweep and the recommendation are in
[`experiments/us-acs-local-l2-basis-20260928/README.md`](../experiments/us-acs-local-l2-basis-20260928/README.md).

## The two bases

Write `d` for the anchor (the initial, design weights unless `l2_anchor` says
otherwise), `w` for the calibrated weights, `r = w / d`, and `D = sum(d)`.

| | `l2_basis="record"` (default) | `l2_basis="chi_square"` |
|---|---|---|
| penalty | `mean(r ** 2)` | `sum(d * (r - 1) ** 2) / D` = `sum((w - d) ** 2 / d) / D` |
| value at `w = d` | 1 | 0 |
| target-free optimum, `mass="conserve"` | `w ∝ d ** 2` | `w = d` (an explicit anchor whose total differs is rescaled to the input total) |
| target-free optimum, `mass="free"` | `w → 0` | `w = d` |
| cost of a record collapsing to 0 | none (0 is its cheapest ratio) | its anchor share `d_i / D` |
| `options["l2_penalty"]` (initial anchor) | `mean_initial_pre_gate_weight_ratio_squared` | `chi_square_initial_pre_gate_weight_distance` |

The chi-square form is GREG's distance function. Minimizing target loss plus
`l2_lambda` times it is penalized ("ridge") calibration, the soft form of GREG:
it works where exact GREG cannot, with thousands of partly inconsistent soft
targets, nonnegativity, a ratio cap and 1.6 million records.

Identities (tested in
`packages/microcosm-calibrate/tests/engine_free/shared/test_l2_basis.py`):

- When `sum(w) = D`, the distance equals `sum(w ** 2 / d) / D - 1`: the
  weighting effect of `w` relative to `d`, minus one.
- With uniform `d` it equals `n / ESS(w) - 1`, so it is a direct Kish ESS
  control.
- The float32 penalty the solver differentiates agrees with the float64
  reference `microcosm.calibrate.chi_square_distance`.
- Under a mass constraint the reduced gradient of the chi-square penalty
  vanishes at `w = d`, and that of the record penalty at `w ∝ d ** 2`.

`CalibrationResult.chi_square_distance` reports the realized distance from the
input weights for every solve, whatever its penalty.

Kish ESS is not in general monotone in `l2_lambda`. What is monotone, for exact
minimizers, is the penalty itself: adding the optimality inequalities of two
solves at `l2_lambda` values `a < b` gives
`(b - a) * (P(w_b) - P(w_a)) <= 0`. With unequal design weights, calibration can
raise Kish ESS above the design's, and pulling back toward `d` then lowers it.
In the ACS local release, calibration lowers ESS from the design's 36,288 to
13,631 in the published weights (13,646 in the sweep's reproduction), and as `l2_lambda → ∞` the solve returns to the design's figure; the
sweep measures the path in between.

The path tests check each solve against the exact optimum of the same convex
program, computed by CLARABEL on the tests' own problems (fixture
`tests/fixtures/l2_basis_path_reference.json`, generator
`experiments/us-acs-local-l2-basis-20260928/path_reference.py`). The
largest excess measured there is 0.0026. The excess bound `eps` = 0.006 is
empirical (twice that maximum, rounded up to 1e-3); the monotonicity slack
`2·eps/Δλ` is derived from it, since two `eps`-optimal solves of the same
feasible set satisfy `(λb − λa)(Pb − Pa) ≤ 2·eps`. The exact optima also show the textbook structure: between
`l2_lambda = 0.01` and `0.3` the optimal distance is flat, because the solve
meets every feasible target exactly and picks the closest such point to `d`,
which is GREG's solution.

## The mass parametrization

Under `mass="conserve"` the historical solve takes an Adam step on the
log-weights, then shifts every log-weight by the same amount so the total
returns to the input total (`mass_parametrization="projection"`). Adam divides
each coordinate's gradient by its own running RMS plus `eps` (1e-8) before
stepping. When a gradient is well above `eps`, the coordinate moves about `lr`
in the direction of the gradient's sign, whatever its size. If every record's
gradient then has the same sign, all records step by about `lr` and the shift
takes them straight back. Every target missing on the same side produces
this; a net demand for mass with targets on both sides does not. Such a point
is a fixed point of the projected iteration whether or not it is the
constrained optimum. At the constrained optimum the log-weight gradient is
proportional to `w` (the constraint's normal), not zero, so the iteration has
no systematic pull toward it; it gets there only through records whose
gradients change sign. Records whose gradient is zero or near `eps` take
smaller steps, which the shift does not cancel, and that weakens the
mechanism. At the ACS release's scale it is weak in exactly this way
(`experiments/us-acs-local-l2-basis-20260928/gradient_scale.py`). At the
design weights the median record's log-weight gradient is 3.8e-8 and 23% are
at or below `eps`; at the release's weights the median is 1.2e-8 and 47% are
at or below `eps`. The signs are mixed, 33% positive at the design and 52% at
the release. So the sharp same-sign stall does not occur there, and whether
the projection solve still falls short is an empirical question the
full-scale sweep answers by solving the release's own surface both ways.

`mass_parametrization="softmax"` optimizes `w = D * softmax(log_w)` instead.
The total holds by construction, and autograd hands Adam the gradient with its
component along the constraint already removed. For a smooth objective that
reduced gradient vanishes at the constrained optimum; the capped-MAPE loss has
kinks, so Adam still oscillates there on the scale of `lr`, as it does under
`mass="free"`. The ratio cap remains a per-step clamp, alternated with the
softmax-invariant renormalization for up to 32 rounds, plus the closing
float64 projection that both parametrizations share. It requires
`mass="conserve"` and no L0 gates.

**Known limitation: the cap rounds run out at production scale.** Each
renormalization lifts the clamped records back over the cap by a shrinking
amount, so the rounds converge geometrically but need not reach float32
exactness in 32. A step that runs out ends on a clamp, and its next forward
pass optimizes `total·softmax(log_w)` past the cap, by an amount that is not
recorded.
`options["iterate_selection_receipt"]["softmax_cap_rounds_exhausted_epochs"]`
counts those steps. On small problems it is rare. On the ACS local release it
is the normal state: 165-400 of the last 400 epochs in every run that records
it, including 400 at the share-0.5, `l2_lambda = 0.03` configuration on the
full surface and on one fold, and 400 in every share-0.9 run. Only the
closing projection makes the returned vector exact. The returned loss stays
within 0.3% of the last trajectory loss in every run that records the count,
but projection runs, which have no cap loop, show the same gap (0.01-0.33%), so
that says nothing about the overshoot; its size is not measured. An
exact capped-softmax step (water-filling the excess onto the uncapped records)
would remove the limitation. Until then, prefer `"projection"` at that scale,
where the stall does not bite (below).

Evidence:

- `test_projection_parametrization_stalls_under_uniform_mass_pressure` pins
  the stall in its sharpest form. When every target sits above its design
  total, the projection
  solve returns the design weights unchanged after 200 epochs, while the
  softmax solve halves the loss. The same mechanism means the record penalty
  alone never moves a projection solve: its log-space gradient is positive for
  every record.
- `experiments/us-acs-local-l2-basis-20260928/optimizer_reference.py` runs the
  kernel under each setting and compares it with CLARABEL's exact optimum of
  the same convex program:

| Setting | Problems | Mean excess objective at λ = 0 / 0.001 / 0.01 / 0.1 / 1 | Max excess | Problems with non-monotone distance |
|---|---|---|---:|---:|
| `mass="conserve"`, `"projection"` (default) | 16 small (120 records, 6 targets; half press on the total) | 0.0285 / 0.0268 / 0.0304 / 0.1608 / 0.0697 | 0.3384 | 94% |
| `mass="conserve"`, `"softmax"` | 16 small (120 records, 6 targets; half press on the total) | 0.0013 / 0.0016 / 0.0013 / 0.0017 / 0.0008 | 0.0034 | 0% |
| `mass="free"` | 16 small (120 records, 6 targets; half press on the total) | 0.0004 / 0.0015 / 0.0020 / 0.0019 / 0.0014 | 0.0042 | 0% |
| `mass="conserve"`, `"projection"` (default) | 3 large (3,000 records, 150 targets in both directions) | 0.0023 / 0.0037 / 0.0062 / 0.0081 / 0.0061 | 0.0139 | 0% |
| `mass="conserve"`, `"softmax"` | 3 large (3,000 records, 150 targets in both directions) | 0.0012 / 0.0016 / 0.0018 / 0.0012 / 0.0010 | 0.0023 | 0% |
| `mass="free"` | 3 large (3,000 records, 150 targets in both directions) | 0.0009 / 0.0016 / 0.0018 / 0.0013 / 0.0010 | 0.0023 | 0% |

  Excess objective is the kernel solve's loss plus `l2_lambda` times its
  distance, minus CLARABEL's optimum of the same program; the loss cap never
  binds on these surfaces. Receipts: `results/optimizer_reference.csv` and
  `results/optimizer_reference_summary.json` in that experiment directory.
  The small problems where every target presses on the total show the stall.
  On the larger problems with targets in both directions, projection is never
  non-monotone but still lands 1.9 to 6.7 times further from the optimum than
  softmax.

## At the ACS release's scale

On the published ACS local release (1,588,854 households, 4,459 targets),
recalibrated from its own checkpoint (`experiments/us-acs-local-l2-basis-20260928/`):

- **Parametrization.** The per-record gradients there are near Adam's `eps`
  with mixed signs, so the projection stall does not occur. The two
  parametrizations trace the same training frontier for `l2_lambda ≤ 0.03`
  (at `0` softmax reaches 18% more ESS, 16,132 against 13,646, at a 3% higher
  loss, about the size of the unpenalized solve's 2.7% run-to-run variation). At `0.1` and above, projection falls inside softmax's
  frontier: at `1` it reaches ESS 30,582 at loss 0.0785, where softmax's path,
  interpolated between its `0.3` and `1` solves, gives about 0.055. Equal
  `l2_lambda` is not an equal-ESS comparison. At `0.03` the held-out
  comparison is a toss-up: softmax has slightly lower capped error, projection
  more held-out targets within 10% on both folds. Projection's training loss is
  6% lower there, against 0.3% run-to-run variation. With that and the
  cap-loop limitation above, projection is the recommended parametrization.
- **Record basis.** The record basis at `0.1` lowers national ESS from 13,646
  to 8,689, as its `w ∝ d ** 2` optimum predicts.
- **Chi-square basis.** The chi-square basis raises ESS smoothly with
  `l2_lambda`. At `0.03` under projection it costs 0.3 points of training fit
  and no measurable held-out fit: it is ahead of the release on both measures
  (capped error and share within 10%) on both rotated folds. Held-out
  run-to-run variation was not measured and those folds also chose
  `l2_lambda`, so the gain is optimistic.
- **What limits ESS is the starting weights.** The experiment's README has the
  frontier, the holdout, candidate ESS floors and the recommendation.
- **On the weighted loss.** With the national release's target weighting
  (microcosm#1104), `l2_lambda` is in units of a different loss, so the
  README's "On the weighted loss" section re-picks it. Held-out weighted error
  is lowest at projection `l2_lambda = 0.1` on the default weights. But the
  weighting gives each state's district populations one shared budget, and at
  0.1 it leaves 135 of 436 trained district populations more than 10% off.
  `l2_lambda = 0.03` with `--target-family-loss-multiplier
  census_population=8` misfits 3 and has slightly lower held-out error (by
  0.6%). The multiplier was chosen after the first pass of runs.

## Using it

```python
from microcosm.calibrate import calibrate

result = calibrate(
    frame,
    targets,
    mass="conserve",
    max_weight_ratio=5.0,
    l2_lambda=...,
    l2_basis="chi_square",
    # mass_parametrization="softmax" for problems where every target presses
    # the same way; see the cap-loop limitation above.
)
result.options["l2_basis"], result.options["mass_parametrization"]
result.chi_square_distance, result.effective_sample_size
```

The ACS local-area tool takes the same settings as
`--l2-lambda`, `--l2-basis {record,chi_square}` and
`--mass-parametrization {projection,softmax}`. It records them, with the
realized chi-square distance, in `calibration_summary.json` and in the build
manifest's `calibration` block. Both join the tool's solver-settings stamp,
so `--resume` and the already-complete shortcut refuse weights solved under
any other solver setting (cap, loss cap, `l2_lambda`, `l2_basis`,
`mass_parametrization`, seed, epoch batch); a stamp written before these two
options existed reads as the historical solve. The manifest's refresh recipe
carries the three flags whenever they differ from the historical solve.
`calibrate_l0_refit` takes `l2_basis` / `refit_l2_basis`
and `refit_mass_parametrization`; `refit_l0_selection` and `static_aging` take
`l2_basis`.

## What the penalty cannot fix in the ACS local release

The chi-square penalty pulls toward the design weights, so the most it can
recover is the design weights' own concentration. In the ACS local staging,
that concentration is already high by construction. The 57,240 donor-spine
records (the Build O sparse release) carry half the design mass with a within-
spine ESS of 9,174, while the 1,531,614 ACS records carry the other half with a
within-spine ESS of 812,434. National design ESS is therefore 36,288 (2.3% of
records). Massachusetts shows the same split
(`massachusetts_by_spine_at_release_design` in that experiment's
`results/seeding_options.json`). Its 35,056 ACS records have a
design ESS of 20,238, while its 1,522 donor records carry 49.6% of the state's
design mass with an ESS of 255, the national release's own Massachusetts figure,
since the donor spine is that release's records at half weight. Together the
state's design ESS is 1,020, and each of its nine districts is 108-134 (the ACS
records alone give 2,020-2,444 per district). As `l2_lambda → ∞` the solve
returns to those combined figures. The penalty bounds the distance from them,
not ESS itself, but no tested setting lifted national ESS above them. Reaching the ACS-only figures needs a larger ACS
share of the mass in the staging (`--acs-share`), which is a construction
decision outside calibration; `experiments/us-acs-local-l2-basis-20260928/
seeding_options.py` measures the starting ESS under other shares and under
donor location clones.
(Measured on the calibration checkpoint of the 2026-09-23 release,
`populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z`.)
