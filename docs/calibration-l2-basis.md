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
`mass="free"`. After every step the log-weights are projected exactly onto
the capped simplex `{w : sum(w) = D, w <= cap * d}`, and the closing float64
projection that both parametrizations share makes the returned vector exact.
It requires `mass="conserve"` and no L0 gates.

**The per-step cap projection.** It is the Kullback-Leibler projection: one
common shift `s` for every log-weight, then a clamp to each log cap,
`log_w <- min(log_w + s, log(cap * d))`. Uncapped records keep their
relative weights, the direction the softmax cannot see, and only records
whose share would pass their caps are held at them. `s` solves
`sum(min(exp(log_w + s), cap * d)) = D`, whose left side is continuous and
nondecreasing in `s`. Active-set rounds find it: with `C` the records already
at their caps, the next shift solves
`sum(cap * d[C]) + exp(s) * sum(exp(log_w)[not C]) = D`. That shift is a lower
bound on the answer for every `C`, so `C` only grows, and a round that adds
no record is exact. If eight rounds pass without that, a sort over the
remaining records finishes it, which bounds the worst case at `O(n log n)`.
The shift is computed in float64. The solve records
`options["iterate_selection_receipt"]["softmax_in_loop_max_cap_ratio"]`, the
largest ratio of a realized in-loop weight to its float64 cap over every
forward pass and the closing iterate, before the closing projection.
`test_softmax_cap_projection.py` pins the projection for every drawn input,
including zero design weights and a cap of exactly 1. It conserves the total,
respects every cap, is idempotent and invariant to a constant added to the
log-weights, and moves an input already within its caps by the shift alone.
It also meets the Kullback-Leibler projection's KKT conditions, equals a
float64 water-fill written from the definition, and gives the same answer by
its active-set and sorted paths, including on an input built to need one
round per record.

**What the projection replaced.** Before, each step renormalized and clamped
in a loop of at most 32 rounds. On the ACS local release the rounds ran out on
most epochs, and how far that left the in-loop weights past their caps was
not recorded. It has since been measured
(`experiments/us-acs-local-l2-basis-20260928/results/exact_cap.md` and
`results/exact_cap_float32.json`):

- **Why the loop ran out.** Each round's shift was `log(D)` in float64 minus a
  float32 `logsumexp`. On a vector already within its caps that difference
  does not reach zero. It bottoms out at the float32 rounding residual of
  `log(D)` itself, 7.8e-8 here. That shift lifts the one to three capped
  records with log caps below 2 in magnitude one float32 step over their caps,
  the clamp puts them back, and the next round repeats it. On full-scale vectors
  the projection had already made exact, the loop still ran out in 10 of 12
  trials.
- **How far past the cap it was.** Not materially. Two full-scale
  configurations whose earlier runs ran out were rerun under both steps: share
  0.5 at `l2_lambda = 0.03`, and share 0.9 unpenalized. The largest in-loop
  weight over its cap was 1 + 1.2e-5 and 1 + 8.7e-6 under the loop, and
  1 + 9.0e-6 under the projection at both. The loop ran out on 713 and 705 of
  800 steps. The projection settled in one to three rounds on every step and
  took 12.7-22.5 ms a step, against the loop's 48.7-53.8 ms.
- **What sets that residual.** The float32 softmax. The projected
  log-weights sit within their float32 log caps, but the solve realizes
  `D * softmax(log_w)` with a float32 normalizer over 1.59M records, whose
  total was off by up to 8.2e-6. The cap ratio exceeded that total error by
  at most 6.1e-7. With a float64 normalizer on the same log-weights the largest
  ratio was 1 + 4.9e-7, the float32 rounding of the log-weights themselves.
- **The solves.** National ESS agreed within 0.1% between the two steps
  (21,834 against 21,835, and 93,609 against 93,598). At share 0.5 and
  `l2_lambda = 0.03` the projection's final loss was 0.01899. Three loop runs
  of that configuration gave 0.01910-0.01917, so the projection's loss is
  0.6-0.9% lower, from one run. At share 0.9 the loss was 0.2% lower (0.03129
  against 0.03137).

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
  6% lower there, against 0.3% run-to-run variation, so projection is the
  recommended parametrization. Rerun with the exact cap projection, softmax's
  training loss at `0.03` is 0.01899 (one run), still 5% above projection's
  0.01806.
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
    # the same way (above).
)
result.options["l2_basis"], result.options["mass_parametrization"]
result.chi_square_distance, result.effective_sample_size
```

The ACS local-area tool takes the same settings as
`--l2-lambda`, `--l2-basis {record,chi_square}` and
`--mass-parametrization {projection,softmax}`. It records them, with the
realized chi-square distance and, under softmax, the last epoch batch's
largest in-loop weight over its cap, in `calibration_summary.json` and in the
build manifest's `calibration` block. Both join the tool's solver-settings stamp,
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
