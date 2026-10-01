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
on 152 records (over 36,491 Massachusetts records in the files he read), and
district ESS of 41-64. On this checkpoint's 36,578 Massachusetts households the
published weights give 446 with 149 records holding half, and the reproduction
below gives 448; the published national figure is 13,631 and the reproduction's
13,646. Threshold statistics (a 1.9 pp drop
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
   - On held-out targets, projection at λ = 0.03 fits better than the release on both measures and both folds (see the table), consistent with a small penalty trading a little training fit for less overfitting. The λ was picked on the same two folds it is reported on.
3. **The seeding sets the limit.** As λ grows the solve returns to the starting weights. On every tested path the national ESS stays below the prior's (at share 0.5, λ = 3 reaches 35,386 against the prior's 36,288), though the penalty is a distance, not an ESS bound, and single districts can exceed their prior's ESS (at λ = 1 some do).
   - A larger ACS share raises that limit several-fold. At share 0.9 without a penalty: ESS 93,524, Massachusetts 2,704, no district below 50, and 95.8% of targets within 10%.
   - It also worsens held-out fit, mainly on SOI targets (held-out SOI within 10% falls from 64% to 54% at share 0.9), while held-out district populations improve.
   - That is consistent with how the staging builds the rows: the ACS rows' tax variables are transferred from the donor rows by QRF (the tool's `donor_sparse_selection_training_set` limitation), so the donor rows carry tax detail the ACS rows only approximate.
4. **The softmax parametrization is not needed at this scale, and its cap loop runs out of rounds here.**
   - The per-record gradients here are near Adam's `eps`, with mixed signs (`results/gradient_scale.json`). The same-sign stall that the small-problem evidence shows does not occur.
   - The two parametrizations agree for λ ≤ 0.03. At λ = 0 neither dominates (softmax ESS 16,132 at loss 0.0160, projection 13,646 at 0.0155). At λ ≥ 0.1 projection falls inside softmax's training frontier. At λ = 0.1 projection reaches ESS 22,739 at loss 0.0228, where softmax's path (interpolated between its λ = 0.03 and 0.1 solves) gives about 0.021. At λ = 1 projection reaches 30,582 at 0.0785, against about 0.055 on softmax's path (between λ = 0.3 and 1). Equal λ is not an equal-ESS comparison.
   - Softmax's per-step cap loop runs out of its 32 rounds on most epochs at this scale: in the runs that record it (heads 9ef71ed6 and bc763cb4), 165-400 of the last batch's 400 epochs, including 400 at share 0.5, λ = 0.03 on fold 0 and every share-0.9 run. Those epochs optimize total·softmax(log_w) slightly past the 5x cap; only the closing projection makes the returned weights exact. In every run that records the count, the returned loss is within 0.3% of the last trajectory loss (0.017571 vs 0.017530 at share 0.5, λ = 0.03, fold 0). That bounds the last step and the closing projection together, not the in-loop overshoot, which is not recorded. So the effect looks small, but its size is not measured. Projection, which has no cap loop, is the recommended parametrization.

## Key configurations

"Held-out error" is the mean capped scaled error, and "held-out within 10%"
the share of targets within 10%, on a rotated 20% of targets
(`microcosm.build.holdout.rotated_folds(4459, n_folds=5, seed=20260529)`)
never seen by that solve. MA is Massachusetts. Each configuration's training fit
and concentration come from its full-surface run; "–" marks a fold not run.

<!-- key-configs:start (written by analyze.py) -->
| Configuration | National ESS | Top-1% share | State ESS min | CD ESS median / min | CDs < 50 | MA ESS | Train within 10% | Held-out error, fold 0 / 1 | Held-out within 10%, fold 0 / 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Release (reproduced): share 0.5, projection, λ 0 | 13,646 | 69.3% | 195 | 33 / 12 | 352 | 448 | 97.6% | 0.1296 / 0.1236 | 63.3% / 65.8% |
| Share 0.5, projection, chi-square λ 0.03 | 20,690 | 63.7% | 204 | 46 / 22 | 278 | 589 | 97.3% | 0.1247 / 0.1191 | 64.2% / 67.0% |
| Share 0.5, softmax, chi-square λ 0.03 | 21,835 | 61.6% | 213 | 49 / 23 | 239 | 620 | 97.1% | 0.1242 / 0.1166 | 63.1% / 66.3% |
| Share 0.5, projection, chi-square λ 0.1 | 22,739 | 62.0% | 207 | 51 / 27 | 208 | 616 | 95.9% | 0.1285 / – | 61.7% / – |
| Share 0.5, softmax, chi-square λ 0.1 | 24,895 | 59.2% | 222 | 55 / 29 | 132 | 664 | 95.5% | 0.1263 / 0.1210 | 61.5% / 63.8% |
| Share 0.5, softmax, chi-square λ 1 | 33,418 | 53.3% | 269 | 73 / 47 | 4 | 907 | 75.7% | 0.1505 / – | 50.1% / – |
| Share 0.7, softmax, chi-square λ 0.01 | 34,280 | 46.9% | 319 | 81 / 24 | 30 | 992 | 97.1% | 0.1466 / 0.1416 | 60.4% / 60.5% |
| Share 0.9, softmax, no penalty | 93,524 | 27.8% | 599 | 227 / 64 | 0 | 2,704 | 95.8% | 0.1765 / 0.1767 | 56.1% / 56.7% |
| Share 0.9, softmax, chi-square λ 0.01 | 130,608 | 24.2% | 612 | 317 / 85 | 0 | 3,285 | 95.1% | 0.1785 / – | 54.9% / – |
| Share 0.964, softmax, no penalty | 244,979 | 17.9% | 594 | 578 / 265 | 0 | 6,591 | 94.9% | 0.1952 / 0.1984 | 51.6% / 52.4% |
<!-- key-configs:end -->

The full grid, the per-family fit, the holdout detail and candidate floors are
in `results/frontier.md`, `results/floors.md` and `results/frontier.csv`:

![ESS vs training fit by prior and λ](results/frontier.png)

## Recommendation

Two decisions, kept separate because they trade different things.

**1. Calibration: adopt the chi-square penalty at the current seeding.**
On today's loss, use `--l2-basis chi_square --l2-lambda 0.03` with the
default `--mass-parametrization projection`.
- Against the release it improves every concentration measure:
  - national ESS from 13,646 to 20,690;
  - Massachusetts from 448 to 589;
  - districts below ESS 50 from 352 to 278;
  - the smallest district from 12 to 22.
- Out of sample it is better on both measures and both folds: held-out error 0.1247 and 0.1191 against 0.1296 and 0.1236, and held-out within 10% 64.2% and 67.0% against 63.3% and 65.8%.
- In sample it costs a little: training loss rises from 0.0155 to 0.0181, and training targets within 10% fall from 97.6% to 97.3%.
- Softmax at λ = 0.03 gives a little more ESS (21,835) and lower held-out error, but its held-out within 10% is slightly *below* the release's on fold 0 (63.1% vs 63.3%), its training fit is worse, and its cap loop runs out at this scale (finding 4). Projection is the default and the better-supported choice.
- λ = 0.1 (softmax) buys more ESS (24,895; 132 districts below 50) and lower held-out error than the release, but lower held-out within 10% on both folds (61.5% and 63.8%), and 2 pp of training fit.
- **λ is in units of the loss.** The ACS local build weights every target equally today (`calibrate` gets no `target_loss_weights`), and IRS SOI cells are 3,819 of the 4,459 targets. A change to weight them as the national release does is planned. Re-pick λ on that loss by rerunning this harness with its target weights before shipping a default.

**2. Construction: the ACS share of the mass is the bigger lever, and it costs out-of-sample fit.**

| ACS share | National ESS | MA ESS | Districts below 50 | Held-out error, fold 0 / 1 |
|---|---:|---:|---:|---:|
| 0.5, no penalty (release reproduced, projection) | 13,646 | 448 | 352 | 0.1296 / 0.1236 |
| 0.7, λ 0.01 (softmax) | 34,280 | 992 | 30 | 0.1466 / 0.1416 |
| 0.9, no penalty (softmax) | 93,524 | 2,704 | 0 | 0.1765 / 0.1767 |

The share-0.7 and share-0.9 rows are softmax solves (at share 0.5, softmax
alone gave 18% more ESS than projection at λ = 0); a projection solve at those
shares was not run.

That is a methodology call: more stable local estimates against worse
aggregate SOI fit. It also overlaps the support-mix bake-off (#1067, decision
d701), which recommends ACS rows plus CPS, seeded in proportion to their
counts, for the local file.

**ESS floor as a release gate.** At the current seeding, no tested
configuration gets every district to ESS 50: even λ = 1 (76% within 10%)
leaves 4 districts below it, and λ = 3 leaves 1. So the gate should catch
calibration-induced collapse now and rise with the seeding. Across each
configuration's three solves (full surface, fold 0, fold 1):

| Configuration | Districts below 25% of their starting ESS | Smallest district ESS | Smallest state ESS |
|---|---|---|---|
| Release reproduced (projection, λ 0) | 15 / 22 / 24 | 11.7 / 10.7 / 7.8 | 195 / 203 / 177 |
| Softmax, λ 0 | 8 / 5 / – | 13.1 / 14.8 / – | 201 / 210 / – |
| Projection, λ 0.03 | 0 / 0 / 0 | 22.4 / 22.9 / 24.2 | 204 / 205 / 193 |
| Softmax, λ 0.03 | 0 / 0 / 0 | 23.0 / 23.0 / 24.1 | 213 / 216 / 203 |

- **The gate to adopt now:** no district below a quarter of its starting ESS.
  It separates cleanly at share 0.5: every penalized solve there has 0, and
  the release has 15-24. It measures collapse relative to the seeding, so it
  binds harder as the seeding improves: at share 0.9 the unpenalized solves
  leave 261-328 districts below a quarter of their (much higher) starting ESS,
  λ = 0.01 leaves 72-121, λ = 0.03 leaves 33, and λ = 0.1 leaves 0. A
  share-0.9 release would pass it only with a penalty around 0.1.
- **An absolute district floor of 15:** it also separates (release at most 11.7, λ = 0.03 at least 22.4), with some margin.
- **No state floor:** a state floor near 200 sits inside the solves' spread (177-216), so it would flap.
- **With an ACS share of 0.9 or more:** raise the floors to every district at least 50 and every state at least 500. At share 0.9 the smallest district is 64-85 and the smallest state 599-612.

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
  and `path_reference.py` compare the kernel with CLARABEL's exact
  optimum on small problems.
- **Kernel heads.** The grid ran on three heads: 35ad6657 (the full-surface
  grid, `release_repro`, the first holdout runs), 9ef71ed6 (the second-pass
  holdout runs) and bc763cb4 (projection at λ = 0.03 and every fold-1 run).
  Each run records its head in `provenance` (and `kernel_head` in
  `results/frontier.csv`). Between 35ad6657 and bc763cb4 (and this branch's
  head), `git diff -- packages/microcosm-calibrate/src` changes `solve.py`
  only in docstrings and comments, the softmax cap-exhaustion counter (a
  receipt field), an `assert` on an unreachable branch, a guard that only
  raises, and an `L0RefitResult` property; the other changed calibrate
  modules came from main and are not on `calibrate`'s path. HEAD_EQUIVALENCE

## Limits

- **Holdout is interpolation.** The holdout draws random targets, most of
  them SOI state cells whose neighbours stay in training. It measures
  aggregate generalization, not the variance of threshold statistics. ESS is
  the variance side of that trade, measured here for every state and district.
- **Folds.** Folds 0 and 1 cover the configurations the recommendation
  compares; λ = 1 and share 0.9 with λ = 0.01 have fold 0 only. λ was chosen
  on the same folds it is reported on, so the held-out gains are optimistic by
  that selection. Fold-to-fold noise is visible in the table.
- **Clones.** Donor location clones were measured only for their starting
  ESS. A solve on cloned rows needs a rebuilt target matrix. The support-mix
  bake-off (#1067) compares clones and ACS rows in the solve.
- **Epochs.** 800 epochs, like the release; longer solves were not run.
- **Not measured.** Poverty and program estimates were not recomputed, since
  the engine was not run. ESS is the proxy for their precision.
