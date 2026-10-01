# ACS local release: effective sample size versus fit under a design-weight penalty

Full-scale evidence for the chi-square L2 basis (`l2_basis="chi_square"`) and
the softmax mass parametrization added in this branch; the algebra and the
small-problem evidence are in [docs/calibration-l2-basis.md](../../docs/calibration-l2-basis.md).
Everything here recalibrates the published ACS local release,
`populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z` (1,588,854
households, 4,459 targets), from its own calibration checkpoint, on Modal. It is
research evidence: nothing was published, and no default changed.

## The question

On 2026-09-28 David Trimmer measured the release's weights in #microcosm:
national Kish ESS of about 14k of 1.59M, Massachusetts 477 with half its weight
on 152 records, and district ESS of 41-64. Threshold statistics (a 1.9 pp drop
in Massachusetts child poverty) rested on a few dozen households. He asked
whether to anchor calibration to the design weights. This experiment measures
what that anchor buys, what it costs, and what limits it.

## Findings

1. **Most of the concentration is in the starting weights, not the solve.**
   - The staging's `--acs-share 0.5` gives the 57,240 donor (ASEC-by-PUF) rows half the mass.
   - Those rows are the national release's records at half weight, with a within-spine ESS of 9,174.
   - So the design weights start at a national ESS of 36,288, a Massachusetts ESS of 1,020 and a median district ESS of 80, before any calibration.
   - The 1,531,614 ACS rows alone have an ESS of 812,434.
   - Calibration then moves the donor share of mass from 50% to 68% and cuts ESS to 13,646.
   - Receipts: `results/seeding_options.{json,md}`, and the national and per-spine blocks of every run.
2. **The anchor works, and at the current seeding it improves out-of-sample fit.**
   - The chi-square penalty pulls toward the design weights themselves. The old record-weighted L2 penalty pulls toward their square: at λ = 0.1 it *lowers* ESS to 8,689.
   - With the chi-square basis, ESS rises smoothly with λ.
   - On held-out targets, λ = 0.03 fits better than the release (see the table), consistent with a small penalty trading a little training fit for less overfitting.
3. **The ceiling is the seeding.** As λ grows the solve returns to the starting weights, so no λ lifts ESS above the prior's.
   - A larger ACS share raises the ceiling several-fold. At share 0.9 without a penalty: ESS 93,524, Massachusetts 2,704, no district below 50, and 95.8% of targets within 10%.
   - It also worsens held-out fit, mainly on SOI targets (held-out SOI within 10% falls from 64% to 54% at share 0.9), while held-out district populations improve.
   - That is consistent with how the staging builds the rows: the ACS rows' tax variables are transferred from the donor rows by QRF (the tool's `donor_sparse_selection_training_set` limitation), so the donor rows carry tax detail the ACS rows only approximate.
4. **The softmax parametrization is not needed at this scale.**
   - The per-record gradients here are near Adam's `eps`, with mixed signs (`results/gradient_scale.json`). The same-sign stall that the small-problem evidence shows does not occur.
   - Projection and softmax land on the same frontier. At λ = 0 softmax finds a solution with ESS 16,132 and loss 0.0160, against projection's 13,646 and 0.0155; neither dominates.

## Key configurations

"Held-out error" is the mean capped scaled error on a rotated 20% of targets
(`microcosm.build.holdout.rotated_folds(4459, n_folds=5, seed=20260529)`)
never seen by that solve; lower is better. MA is Massachusetts. Each
configuration's training fit and concentration come from its full-surface run.

<!-- key-configs:start (written by analyze.py) -->
| Configuration | National ESS | Top-1% share | State ESS min | CD ESS median / min | CDs < 50 | MA ESS | Train within 10% | Held-out error, fold 0 / 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Release: share 0.5, projection, λ 0 | 13,646 | 69.3% | 195 | 33 / 12 | 352 | 448 | 97.6% | 0.1296 / 0.1236 |
| Share 0.5, chi-square λ 0.03 | 21,835 | 61.6% | 213 | 49 / 23 | 239 | 620 | 97.1% | 0.1242 / 0.1166 |
| Share 0.5, chi-square λ 0.1 | 24,895 | 59.2% | 222 | 55 / 29 | 132 | 664 | 95.5% | 0.1263 / 0.1210 |
| Share 0.5, chi-square λ 1 | 33,418 | 53.3% | 269 | 73 / 47 | 4 | 907 | 75.7% | 0.1505 / – |
| Share 0.7, chi-square λ 0.01 | 34,280 | 46.9% | 319 | 81 / 24 | 30 | 992 | 97.1% | 0.1466 / 0.1416 |
| Share 0.9, no penalty | 93,524 | 27.8% | 599 | 227 / 64 | 0 | 2,704 | 95.8% | 0.1765 / 0.1767 |
| Share 0.9, chi-square λ 0.01 | 130,608 | 24.2% | 612 | 317 / 85 | 0 | 3,285 | 95.1% | 0.1785 / – |
| Share 0.964, no penalty | 244,979 | 17.9% | 594 | 578 / 265 | 0 | 6,591 | 94.9% | 0.1952 / 0.1984 |
<!-- key-configs:end -->

The full grid, the per-family fit, the holdout detail and candidate floors are
in `results/frontier.md`, `results/floors.md` and `results/frontier.csv`:

![ESS vs training fit by prior and λ](results/frontier.png)

## Recommendation

Two decisions, kept separate because they trade different things.

**1. Calibration: adopt the chi-square penalty at the current seeding.**
On today's loss, use `--l2-basis chi_square --l2-lambda 0.03` with
`--mass-parametrization softmax`.
- It is a Pareto improvement on everything measured here:
  - national ESS from 13,646 to 21,835;
  - Massachusetts from 448 to 620;
  - districts below ESS 50 from 352 to 239;
  - the smallest district from 12 to 23.
- Training targets within 10% fall from 97.6% to 97.1%.
- Held-out error is lower than the release's on both folds (0.1242 vs 0.1296, and 0.1166 vs 0.1236).
- λ = 0.1 buys more ESS (24,895; 132 districts below 50) and still beats the release out of sample, at 2 pp of training fit.
- Projection at λ = 0.03 lands close to softmax (fold 1: held-out 0.1191 vs 0.1166, ESS 21,220 vs 22,415), so the parametrization is a minor choice.
- **λ is in units of the loss.** The ACS local build weights every target equally today (`calibrate` gets no `target_loss_weights`), and IRS SOI cells are 3,819 of the 4,459 targets. A change to weight them as the national release does is planned. Re-pick λ on that loss by rerunning this harness with its target weights before shipping a default; the shape of the trade is unlikely to change, but its scale may.

**2. Construction: the ACS share of the mass is the bigger lever, and it costs out-of-sample fit.**

| ACS share | National ESS | MA ESS | Districts below 50 | Held-out error, fold 0 / 1 |
|---|---:|---:|---:|---:|
| 0.5, no penalty (release) | 13,646 | 448 | 352 | 0.1296 / 0.1236 |
| 0.7, λ 0.01 | 34,280 | 992 | 30 | 0.1466 / 0.1416 |
| 0.9, no penalty | 93,524 | 2,704 | 0 | 0.1765 / 0.1767 |

That is a methodology call: more stable local estimates against worse
aggregate SOI fit. It also overlaps the support-mix bake-off (#1067, decision
d701), which recommends ACS rows plus CPS, seeded in proportion to their
counts, for the local file.

**ESS floor as a release gate.** At the current seeding, a district floor of
50 is out of reach without wrecking fit (it needs λ ≈ 1, at 76% within 10%).
So the gate should catch regressions now and rise with the seeding:
- **now:** every state at least 200, every district at least 20, and no
  district below a quarter of its starting ESS. The release fails all three
  (195, 12 and 15 districts); λ = 0.03 passes them.
- **with an ACS share of 0.9 or more:** every state at least 500 and every
  district at least 50.

Main's tool already records ESS by state and district in the calibration
summary, so the gate is a finalize-stage check on data it already has.

## How it was run

- **Inputs.** The release's calibration checkpoint was a 28.7 GB dense
  float32 frame. The sparse copy used here (`target_matrix.npz`, 4,459 ×
  1,588,854, 24,773,532 nonzeros) was checked value for value against 3,000
  sampled dense rows (`results/csr_verification.json`). Design weights and
  target values equal the release's.
- **Harness.** `sweep.py` calls `microcosm.calibrate.calibrate` with the
  tool's settings: Adam, lr 0.02, `mass="conserve"`, `max_weight_ratio=5`,
  `target_loss_cap=1`, seed 0, 800 epochs as two warm-started batches of 400.
  Each run adds its own `l2_lambda`, `l2_basis`, `mass_parametrization` and
  prior (`acs_share`).
  - The prior rescales each spine to its share of the unchanged total, keeping
    proportions within a spine. That matches `--acs-share` in the staging.
  - The prior is the frame's weights, so it is also the chi-square anchor and
    the base of the 5x cap.
- **Compute.** `modal_sweep.py` ran each configuration in its own 8-CPU
  container from the committed kernel, gated on a reproduction of the release.
  - The reproduction matched the release: loss 0.015485 against 0.015491,
    within-10% 97.58% against 97.58%, ESS 13,646 against 13,631, and
    chi-square distance 0.7269 against 0.7270.
  - Every run's metrics are in `results/runs/`. Weights stay in
    `_build_artifacts/acs-local-l2-basis-20260928/runs/` and the Modal volume
    `microcosm-acs-l2-basis-sweep`.
- **Analysis.** `analyze.py` builds the frontier, the floors, the key table
  and the chart from those metrics. `seeding_options.py` measures starting
  ESS under other shares and under donor location clones.
  `gradient_scale.py` measures the gradient scale. `optimizer_reference.py`
  and `test_path_reference.py` compare the kernel with CLARABEL's exact
  optimum on small problems.

## Limits

- **Holdout is interpolation.** The holdout draws random targets, most of
  them SOI state cells whose neighbours stay in training. It measures
  aggregate generalization, not the variance of threshold statistics. ESS is
  the variance side of that trade, measured here for every state and district.
- **Folds.** Two folds (0 and 1) cover the compared configurations. Fold-to-fold
  noise in held-out error is visible in the table; the ordering is what the
  recommendation relies on.
- **Clones.** Donor location clones were measured only for their starting
  ESS. A solve on cloned rows needs a rebuilt target matrix. The support-mix
  bake-off (#1067) compares clones and ACS rows in the solve.
- **Epochs.** 800 epochs, like the release; longer solves were not run.
- **Not measured.** Poverty and program estimates were not recomputed, since
  the engine was not run. ESS is the proxy for their precision.
