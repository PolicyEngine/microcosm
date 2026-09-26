# Support oracle for microcosm#940 (state x AGI-band SOI targets)

Analysis only, 2026-09-26. Every number here is **pre-calibration** and uses
**proxy AGI** (defined below), computed from the offline #982 predictions and
SOI Historic Table 2. Nothing was calibrated, aged at the record level, or run
through the PolicyEngine engine.

Files:

- `support_oracle.py` — the deterministic script (two runs produce
  byte-identical outputs).
- `support_by_state_band.csv` — 408 rows = 2 variants x 51 states x 4 bands,
  every metric below plus the raw SOI values.
- `support_summary.json` — per-band counts, worst-10 lists, Colorado rows,
  sensitivity, national sums, the Colorado reproduction check.

## What was computed

**Inputs.** `imputed_current.parquet` (old design: the QRF copied each unit's
own survey income) and `imputed_demographic_midrank6_earn.parquet` (the chosen
#1033 design), 231,007 PUF-clone recipient tax units each, 8-tree fits, from
the tail lane's archive
`/Users/maxghenis/PolicyEngine/_recovered/lane958-958-964-982-archive-20260926/982/`
(`MICROCOSM_940_PREDICTIONS_DIR`; the lane's `.lane958/982/` copy, which the
first run read, moved there on 2026-09-26). The SOI files are read from
Chronicle's `db/data/irs_soi/historic_table_2/` (`MICROCOSM_940_SOI_DIR`);
`23in55cmcsv.csv` sha256 `d1f7c890...668f` is the file the Chronicle
TY2023 state AGI package registers. Rerunning from this directory reproduces
both output files byte for byte.
SOI Historic Table 2 TY2023 (`23in55cmcsv.csv`) and TY2022 (`22in55cmcsv.csv`),
50 states + DC (US, OA, PR excluded), `N1` = returns, `A00100` = AGI in $
thousands. State postal codes mapped to FIPS with the standard table (CO=08,
CA=06, DC=11, WY=56 asserted; the five largest states by PUF-half weight come
out as CA, TX, FL, NY, PA).

**Proxy AGI** on the PUF half = the sum of the 15 imputed income items
(wages, self-employment, taxable interest, qualified and non-qualified
dividends, short- and long-term gains, pensions, IRA distributions, rental,
estate, farm, miscellaneous, partnership, S-corp). No adjustments, no Social
Security, no capital-gains distributions.

**Survey half** = the same units at the same weight `w` with income =
the parquet's `survey_total`. Note: `survey_total` is **not** the six-item
predictor sum. `compare_predictors.py` builds it from its `SURVEY` list (up to
14 survey items: the six predictor items plus pensions, IRA distributions,
rental, farm operations, unemployment compensation, alimony, miscellaneous);
43,343 of 231,007 rows differ from the six-item sum. I used `survey_total`
because it is closer in scope to the 15-item PUF proxy; the six-item version is
carried alongside as `svy6_*` / `ratio6_*` columns and moves the ratios by a
few percent (Colorado $1M+: ratio_A 1.62 vs 1.61). The survey half's maximum
income is $3.16M, so the survey channel has almost no support above that.

**Frame view** = PUF half at weight `w` + survey half at weight `w`; each
channel carries half the population (PUF-half weights sum to 91.66M; x2 =
183.3M). The **x2 view** (PUF half doubled, "as if this half were the whole
population") is the #982 README convention and is reported for comparison.

**Targets at 2024** (the rule the compiler will use), per state `s`, measure
`m` in {returns, AGI}, band `b` in {100k_200k, 200k_500k, 500k_1m, 1m_plus}
(HT2 `AGI_STUB` 7, 8, 9, 10):

    share(s,b,m)  = TY2023 HT2 value(s,b,m) / sum over TY2023 stubs 1..10
    control(s,m)  = TY2022 HT2 stub-0 value(s,m)
    target(s,b,m) = share x control;  AGI x 1.1203460 (CBO AGI 2022->2024); counts not aged

TY2023 stub 0 equals the sum of stubs 1..10 to 3.5e-5 (returns) and exactly
(AGI), so the denominator choice is immaterial. Summed over the 51 states the
2024 targets are 787,789 returns / $2,725.6B for $1M+, against SOI Table 1.1
TY2023 $1M+ of 799,094 returns / $2,543.5B (five sub-bands summed from
`score_variants.py`).

**Solver contract** (read from the worktree, not assumed):
`packages/microcosm-build/src/microcosm/build/us/spec/calibration.yaml` sets
`hard_constraints.max_weight_ratio: 5.0`, `mass: conserve`, and the loss
`weighted_mean(min(abs((A@w-target)/target_scale), target_loss_cap))` with
`target_scale = max(abs(target), 1)` and `target_loss_cap: 1.0`.
`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py` optimises
log-weights with Adam and clamps **upward only**
(`log_w.clamp_(max=torch.log(upper))`, `upper = max_weight_ratio * w0`); there
is no lower bound, so a weight may fall toward zero. The yaml's
`infeasibility_contract` says `structural_cap_mass_or_empty_support: refuse`
and its attainment block says `positive_target_zero_support: fail`.

**Feasibility flags** (per cell, frame view, records `(a_j, w0_j)`):

- `c1_count_reachable`: `N_target <= 5 * sum(w0)`
- `c2_amount_reachable`: `A_target <= 5 * sum(w0 * a)` (all `a > 0` in these bands)
- `c3_mean_in_range`: `min(a) <= A_target / N_target <= max(a)`
- `feasible_necessary` = all three. These are **necessary, not sufficient**:
  each record's weight is shared by every target it enters, mass is conserved
  frame-wide, and the cap prevents putting all of a cell's weight on its top
  record, so the true reachable set is smaller.
- `over2x_A` / `over2x_N`: frame start > 2 x target, i.e. the starting relative
  error exceeds the 1.0 loss cap and the target contributes **zero gradient at
  the start** (it can re-enter the active region only if other targets move
  the same records).
- `zero_support`: no frame record in the cell.

## Colorado reproduction check (chosen variant, $1M+, x2 view)

| Quantity | #982 README | Computed here | |
|---|---|---|---|
| PUF records at $1M+ proxy AGI | 15 | 15 | reproduces |
| Weighted returns, x2 | ~14,847 | 14,847.2 | reproduces |
| Largest record's share of income above $1M | 84.0% | 83.96% | reproduces |
| Largest record's proxy AGI | $125.4M | $125.40M | reproduces |
| Largest record's weight, x2 | 867 | 867.2 | reproduces |

The 84% is the share of income **above $1M** (`w * max(agi - 1e6, 0)`), as in
`score_variants.py`. The same record's share of the cell's total proxy AGI is
75.9%. In the old design the largest Colorado $1M+ record is $12.8M with a
20.3% share (31.6% of income above $1M).

## Colorado, all four bands (pre-calibration, proxy AGI)

Targets at 2024 (both variants): returns 559,959 / 247,962 / 36,620 / 15,031;
AGI $86.1B / $78.8B / $26.9B / $50.2B for 100k_200k / 200k_500k / 500k_1m /
1m_plus. Raw TY2023 HT2: 581,520 / 257,510 / 38,030 / 15,610 returns and
$81.1B / $74.2B / $25.4B / $47.3B.

| Band | Variant | PUF n | PUF N | PUF A $bn | PUF mean | PUF max | Largest share of PUF A | Largest w (half / x2) | Survey n | Survey N | Survey A $bn | Frame N | Frame A $bn | ratio_N | ratio_A | 5x count cap | 5x amount cap $bn | Target mean | Feasible (necessary) | >2x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k_200k | current | 574 | 316,972 | 44.7 | $141k | $200k | 0.4% | 1,138 / 2,276 | 565 | 309,786 | 43.3 | 626,758 | 88.0 | 1.12 | 1.02 | 3,133,789 | 440 | $154k | yes | no |
| 100k_200k | chosen | 569 | 312,760 | 43.0 | $138k | $199k | 0.5% | 1,206 / 2,412 | 565 | 309,786 | 43.3 | 622,547 | 86.3 | 1.11 | 1.00 | 3,112,733 | 432 | $154k | yes | no |
| 200k_500k | current | 309 | 164,652 | 47.3 | $287k | $493k | 1.1% | 1,394 / 2,789 | 283 | 148,922 | 41.9 | 313,574 | 89.3 | 1.27 | 1.13 | 1,567,871 | 446 | $318k | yes | no |
| 200k_500k | chosen | 220 | 114,549 | 32.0 | $279k | $500k | 1.1% | 935 / 1,870 | 283 | 148,922 | 41.9 | 263,471 | 74.0 | 1.06 | 0.94 | 1,317,356 | 370 | $318k | yes | no |
| 500k_1m | current | 36 | 19,368 | 13.3 | $685k | $992k | 6.4% | 934 / 1,868 | 17 | 8,404 | 6.0 | 27,773 | 19.3 | 0.76 | 0.71 | 138,863 | 96 | $736k | yes | no |
| 500k_1m | chosen | 21 | 12,867 | 9.1 | $707k | $954k | 11.7% | 1,394 / 2,789 | 17 | 8,404 | 6.0 | 21,271 | 15.1 | 0.58 | 0.56 | 106,357 | 75 | $736k | yes | no |
| 1m_plus | current | 26 | 13,775 | 31.9 | $2.32M | $12.8M | 20.3% | 749 / 1,499 | 13 | 7,617 | 9.8 | 21,392 | 41.7 | 1.42 | 0.83 | 106,962 | 209 | $3.34M | yes | no |
| 1m_plus | chosen | 15 | 7,424 | 71.7 | $9.65M | $125.4M | 75.9% | 434 / 867 | 13 | 7,617 | 9.8 | 15,041 | 81.5 | 1.00 | 1.62 | 75,206 | 407 | $3.34M | yes | no |

Colorado $1M+ in the chosen variant passes all three necessary conditions
(count 1.00x, amount 1.62x, target mean $3.34M inside [$1.0M, $125.4M]) and is
under 2x, so it is not "infeasible" or "zero-gradient" in this oracle; it is
one record carrying 76% of the cell's amount. Dropping that record leaves 14
PUF + 13 survey records, ratio_A 0.54, still feasible (necessary).

## Per-band counts, 51 states (pre-calibration, proxy AGI)

Infeasible = fails at least one necessary condition, with the three failures
broken out (a state can fail more than one).

### Old design (`current`)

| Band | PUF records (min state) | States PUF n<10 | Zero support | Infeasible (count cap / amount cap / mean out of range) | A_frame>2x | N_frame>2x | Largest record >50% of PUF A | ratio_A median [min, max] | ratio_N median [min, max] |
|---|---|---|---|---|---|---|---|---|---|
| 100k_200k | 37,200 (274) | 0 | 0 | 0 (0 / 0 / 0) | 0 | 0 | 0 | 0.99 [0.83, 1.21] | 1.06 [0.91, 1.22] |
| 200k_500k | 18,154 (133) | 0 | 0 | 0 (0 / 0 / 0) | 0 | 0 | 0 | 1.17 [0.91, 1.53] | 1.26 [1.01, 1.57] |
| 500k_1m | 3,098 (20) | 0 | 0 | 0 (0 / 0 / 0) | 1 (RI) | 1 (RI) | 0 | 1.09 [0.67, 2.01] | 1.18 [0.76, 2.10] |
| 1m_plus | 1,570 (9) | 2 | 0 | 12 (0 / 1 / 12) | 0 | 2 (NE, WV) | 1 (OH) | 0.63 [0.12, 1.63] | 1.31 [0.57, 2.41] |

### Chosen design (`demographic_midrank6_earn`)

| Band | PUF records (min state) | States PUF n<10 | Zero support | Infeasible (count cap / amount cap / mean out of range) | A_frame>2x | N_frame>2x | Largest record >50% of PUF A | ratio_A median [min, max] | ratio_N median [min, max] |
|---|---|---|---|---|---|---|---|---|---|
| 100k_200k | 37,680 (282) | 0 | 0 | 0 (0 / 0 / 0) | 0 | 0 | 0 | 0.97 [0.84, 1.20] | 1.05 [0.91, 1.24] |
| 200k_500k | 13,302 (78) | 0 | 0 | 0 (0 / 0 / 0) | 0 | 0 | 0 | 0.99 [0.78, 1.30] | 1.07 [0.85, 1.36] |
| 500k_1m | 1,990 (13) | 0 | 0 | 0 (0 / 0 / 0) | 0 | 0 | 0 | 0.85 [0.52, 1.43] | 0.91 [0.58, 1.53] |
| 1m_plus | 1,078 (4) | 11 | 0 | 9 (0 / 2 / 9) | 2 (SC, WV) | 1 (WV) | 4 (CO, NV, SC, WV) | 0.58 [0.11, 2.70] | 0.96 [0.44, 2.31] |

Reading the 1m_plus row for the chosen design:

- No cell is empty and the count cap never binds: even the thinnest cell
  (DE, 4 PUF + 3 survey records) has 5x headroom on returns.
- The binding failure is the **mean** condition: in 9 states the SOI target mean
  ($2.75M-$5.93M) exceeds every record in the cell, so no reweighting of the
  existing records can hit both the count and the amount. Two of them (WY, DE)
  also fail the 5x amount cap outright.
- ratio_A distribution: 4 states below 0.25, 18 in [0.25, 0.5), 19 in
  [0.5, 1), 8 in [1, 2), 2 at or above 2. The frame starts **below** the SOI
  amount in 41 of 51 states; the solver would have to push top-record weights
  up (toward the 5x cap) almost everywhere.
- The two over-2x cells (WV 2.70x, SC 2.03x) are each one record: the largest
  carries 64% / 56% of the state's PUF amount, and removing it brings both under
  2x (1.24x, 1.05x).
- 11 states have fewer than 10 PUF records at $1M+ (DE 4, IA 7, IN 7, MS 7,
  KY 8, RI 8, WI 8, MN 9, MT 9, SD 9, WY 9). The old design had 2 (KY 9, ME 9)
  because it spread survey-topcoded incomes across more clones; its
  $1M+ amounts were correspondingly lower (national x2 $2.22T vs $2.70T).

### States failing the mean condition at $1M+ (chosen design)

| State | PUF n | Survey n | Target N | Target A $bn | Target mean | Frame max record | 5x amount cap $bn | ratio_A |
|---|---|---|---|---|---|---|---|---|
| DE | 4 | 3 | 1,646 | 4.55 | $2.76M | $1.35M | 4.45 | 0.20 |
| IA | 7 | 6 | 4,321 | 13.87 | $3.21M | $1.47M | 19.97 | 0.29 |
| ID | 10 | 6 | 3,175 | 10.67 | $3.36M | $2.42M | 10.90 | 0.20 |
| ME | 11 | 6 | 2,056 | 5.66 | $2.75M | $2.75M | 27.07 | 0.96 |
| MI | 17 | 13 | 15,917 | 51.12 | $3.21M | $2.83M | 101.95 | 0.40 |
| MT | 9 | 5 | 2,236 | 7.27 | $3.25M | $3.08M | 7.83 | 0.22 |
| RI | 8 | 5 | 1,892 | 6.08 | $3.22M | $3.08M | 11.76 | 0.39 |
| WI | 8 | 5 | 9,758 | 31.29 | $3.21M | $2.51M | 44.62 | 0.29 |
| WY | 9 | 9 | 1,720 | 10.20 | $5.93M | $2.23M | 5.44 | 0.11 |

(ME fails by a hair: target mean $2.7545M vs max record $2.7543M.) The old
design's 12 failures are AR, FL, ID, IN, KY, LA, MO, MT, ND, OK, UT, WY, with
Florida the notable one: 84 PUF records but a max of $2.9M against a $4.59M
target mean, because the old design's clones inherited survey topcodes.

## Worst 10 states at $1M+ by |log(ratio_A)| (chosen design)

| State | PUF n | Survey n | Frame N | Target N | ratio_N | Frame A $bn | Target A $bn | ratio_A | Largest share of PUF A | Max record | Failing | ratio_A without largest record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WY | 9 | 9 | 844 | 1,720 | 0.49 | 1.09 | 10.20 | 0.11 | 18% | $2.2M | amount cap, mean | 0.10 |
| DE | 4 | 3 | 724 | 1,646 | 0.44 | 0.89 | 4.55 | 0.20 | 31% | $1.3M | amount cap, mean | 0.16 |
| ID | 10 | 6 | 1,633 | 3,175 | 0.51 | 2.18 | 10.67 | 0.20 | 15% | $2.4M | mean | 0.19 |
| MT | 9 | 5 | 1,009 | 2,236 | 0.45 | 1.57 | 7.27 | 0.22 | 23% | $3.1M | mean | 0.18 |
| WI | 8 | 5 | 7,111 | 9,758 | 0.73 | 8.92 | 31.29 | 0.29 | 27% | $2.5M | mean | 0.24 |
| IA | 7 | 6 | 3,342 | 4,321 | 0.77 | 3.99 | 13.87 | 0.29 | 23% | $1.5M | mean | 0.26 |
| IN | 7 | 4 | 5,465 | 9,534 | 0.57 | 8.80 | 29.19 | 0.30 | 28% | $3.2M | - | 0.24 |
| IL | 25 | 16 | 21,272 | 30,493 | 0.70 | 37.45 | 108.07 | 0.35 | 23% | $18.7M | - | 0.29 |
| NH | 14 | 6 | 2,886 | 3,688 | 0.78 | 4.73 | 13.25 | 0.36 | 12% | $4.3M | - | 0.32 |
| WV | 16 | 10 | 2,863 | 1,238 | 2.31 | 8.97 | 3.32 | 2.70 | 64% | $77.0M | - (over 2x) | 1.24 |

The highest ratios are WV 2.70, SC 2.03, CO 1.62, VA 1.28, AK 1.15, AL 1.15;
the four states where one record exceeds half the PUF amount are CO (76%), NV
(78%), WV (64%), SC (56%). The old design's worst list is WY 0.12, MT 0.27,
FL 0.30, ID 0.34, ND 0.36, NV 0.37, NY 0.38, AR 0.42, IN 0.43, IL 0.43 (all
under-target; its only over-target $1M+ amounts are WV 1.63 and OH 1.56, the
latter one $122M record at 67%).

## Sensitivity: remove each state's single largest $1M+ record

| | Old design | Chosen design |
|---|---|---|
| States passing all necessary conditions | 39 -> 21 | 42 -> 30 |
| States flipping feasible -> infeasible | 18 | 12 (AR, IL, IN, KS, KY, MD, MN, MO, MS, ND, OR, VT) |
| States over 2x on amount | 0 -> 0 | 2 -> 0 (SC, WV) |
| Median ratio_A | 0.63 -> 0.58 | 0.58 -> 0.50 |

In the chosen design, 12 of the 42 passing states pass the mean condition only
because of their single largest record; without it, the target mean exceeds
every remaining record. That is the one-record dependence in the direction
opposite to Colorado's: most states are short of top-tail income, not long.

## National sums at $1M+ (sum over 51 states; returns / AGI $bn)

| | Target 2024 | PUF half | Survey half | Frame (PUF + survey) | ratio N / A | PUF x2 | SOI Table 1.1 TY2023 |
|---|---|---|---|---|---|---|---|
| Old design | 787,789 / 2,725.6 | 633,202 / 1,111.5 | 326,243 / 407.3 | 959,445 / 1,518.8 | 1.22 / 0.56 | 1,266,404 / 2,223.0 | 799,094 / 2,543.5 |
| Chosen design | 787,789 / 2,725.6 | 420,000 / 1,351.6 | 326,243 / 407.3 | 746,243 / 1,758.9 | 0.95 / 0.65 | 840,000 / 2,703.2 | 799,094 / 2,543.5 |

The chosen design's x2 view (840,000 returns / $2,703B) reproduces the #982
README's national $1M+ story (sum of its five sub-band rows = 839,999). The
frame view is lower because the survey half, with a $3.16M ceiling, contributes
only $407B of $1M+ income at 326k returns; the assembled frame therefore starts
at 0.65 of the national $1M+ amount target even though the PUF channel alone
is on target.

## What these numbers can and cannot tell you

They can:

- Show that no state x band cell is empty and none fails the 5x **count** cap,
  in either variant, so the `structural_cap_mass_or_empty_support: refuse` and
  `positive_target_zero_support: fail` contracts would not trip on support
  alone for these four bands.
- Identify the cells where the pair (returns, AGI) is unreachable by any
  per-cell reweighting of the current records under the 5x cap: 9 states at
  $1M+ in the chosen design (12 in the old), all because the SOI target mean
  exceeds the cell's largest record, plus WY and DE on the amount cap.
- Identify cells that start beyond the 1.0 loss cap (zero initial gradient):
  SC and WV amounts at $1M+ in the chosen design (RI at 500k_1m and NE/WV
  counts in the old), and show each is a single-record effect.
- Quantify how thin and how single-record-dependent the $1M+ cells are (11
  states under 10 PUF records; 12 states whose feasibility rests on one
  record; four states with one record above half the amount).

They cannot:

- Predict the calibrated outcome. Weights are shared across every target a
  record enters, mass is conserved frame-wide, L0 selection changes the
  support, and the cap forbids concentrating a cell on its top record; the
  flags are necessary conditions on one cell at a time.
- Stand in for engine AGI. Proxy AGI omits adjustments, Social Security and
  other AGI components; the survey half's income is the saved broad survey
  total (up to 14 items), not the engine's AGI for those households; band
  membership will shift when the engine computes AGI for both halves.
- Reflect production fits. These are 8-tree offline fits on the 2026-09-12
  donor frame with no tail stratum (#958) and no clone selection; the record
  counts, maxima and shares will differ in a production build.
- Reflect record-level aging. Incomes were not aged from the donor year;
  only the AGI targets carry the x1.1203460 factor.
- Speak to bands below $100k or to the sub-$1M national comparison, which
  were not computed here.

## Limitations restated

PUF half only for imputed income; survey half approximated with the saved
`survey_total` (broad survey list, up to 14 items, not six; six-item version
in `svy6_*`); 8-tree fits; no tail stratum; no engine AGI; no aging of
incomes; no clone selection; feasibility conditions are per-cell and necessary
only; the AGI aging factor 1.1203460 was supplied by the caller and matches the
"x1.1203 over 2022->2024" CBO AGI default documented in
`us_runtime/fiscal_targets.py`.
