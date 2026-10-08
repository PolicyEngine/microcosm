# UK sparse selection: target-weight rules (A) and L2 (E) before size (F)

microcosm#1124. Pre-registration, committed before any run. Results are added below
the markers by `analyze.py`; nothing here was measured yet.

## What is varied, and what is held

The baseline is one finished K=25, 60,000-household graph build (the v6 warm restart
of the #1115 re-measure, `runs/uk-local-k25-675d2a3fd`): its ordered problem, its dense
solve D, its informed L0 search, its exact-count draw and its refit S0. The dense
reference is never re-solved, so every delta is the selection's.

- **A, the target-weight rule** (`uk_runtime.target_weights`): `grain_equal` (the
  run's rule, the control), `grain_family_equal`, `grain_family_equal_sqrt_count`,
  `nation_grain_family_equal`, `nation_grain_family_equal_sqrt_count`.
- **E, the L2 penalty** (`uk_runtime.dataset_size`, chi-square basis only):
  refit anchors `initial` (the refit's Horvitz–Thompson baseline at the run's
  floor) and `uniform` (its mean, a direct Kish-ESS control).
- **The refit baseline floor** (`baseline_pi_floor`), paired with E by María's
  ruling of 2026-10-07: at floor 0 the 194 boundary draws hold 86 % of the
  baseline mass, so a design anchor there pulls the certainties down.

## Ladder

1. **Step 0** (`tools/run_uk_size_experiment.py build-cache`, `control`, then
   `census`): the stored refit must come back bit for bit before any variant is
   read. The census then records what the stored support and its caps allow: mass
   capacity against D by nation and household type at floors 0, 0.1, 0.5 and 1,
   the areas no refit on this support can lift to criterion 6, the search's
   capacity bound k_min, the penalty scales that place the λ grid, each rule's
   loss shares and the early size triggers.
2. **Step 1a**, `experiments.json` (26 refits on the stored support):
   - controls C1–C5: iterate switch, floor only, floor plateau, optimiser noise;
   - E under `grain_equal`: floor 0 `initial` at 1e-3, 1e-2 and 1e-1 (predicted harmful), floor 0 `uniform` at 1e-2, and floor 0.5 at 1e-3, 3e-3, 1e-2, 3e-2 and 1e-1 for each anchor;
   - A at λ 0: the four rules at floor 0.5, and `grain_family_equal` and `nation_grain_family_equal` at floor 0.
3. **Step 1b**, written by `plan-step1b` straight after 1a by the rules below
   (`--stage ae` from 1a's scorecard, then `--stage holdout` once the A×E points
   are scored):
   - A×E: the best A rule × each anchor × 3 λ around that anchor's E knee, with λ scaled by the A rule's loss level at S0;
   - holdout: a refit-level rotated holdout (`mode: refit_holdout`, 5 local folds) for C0, C2, the best E per anchor, the best A and the best A×E.
4. **Step 1c, exploratory** (chosen after step 1 by María, 2026-10-08; not
   pre-registered, reported apart): `step1c_exploratory.json` runs
   `grain_family_equal_sqrt_count` at floor 0.5 with the uniform anchor at λ 1e-3,
   3e-3 and 1e-2 times its loss ratio at S0 (2.1439), the low-λ region step 1b's knee
   rule skipped, plus the 5-fold holdout of `E_unif_f0.5_1e-2`. Its sets carry the
   prefix `X_`.
5. Report to María. Step 2 (search re-runs) and step 3 (size) run only on her pick.

**Selection rules** (fixed now):
- the best E per anchor is the largest λ that still meets criteria 1–5 of the acceptance below;
- the best A is the rule that meets the most criteria, ties broken by the `grain_equal` yardstick loss;
- the knee is the smallest λ whose criterion-6 count is within 10 % of that anchor's best.

**Readings fixed before any step-1a result** (2026-10-08, with María's go for
steps 0–1; `plan-step1b` applies them):
- Best E and the knee are read on each anchor's floor-0.5 ladder (λ 1e-3 to
  1e-1). The floor-0 E points stay the harm check and are reported beside it.
- Until this point the knee rule said "largest". That makes the knee the top of
  the ladder whenever collapse falls with λ, which is not a knee, so it is
  corrected to "smallest" before any result exists.
- The A×E points run at floor 0.5, at the knee and its two half-decade
  neighbours (past the grid's ends when the knee sits on one).
- The best A×E for the holdout follows the plan's winner rule: criteria 1–5
  first, then the criterion-6 count, then the `grain_equal` yardstick. No
  configuration can pass criterion 6 on v6: the census finds 18 constituencies
  and 11 local authorities whose kept rows cannot reach 25 % of their dense ESS
  even at equal weights.
- The λ grid stays as pre-registered. The census puts its penalty pressure
  λ·P/L at S0 half a decade low for `uniform` and half a decade high for
  `initial`; the neighbours rule above lets 1b reach half a decade past either
  end.

## Acceptance (each configuration against D and S0 of the same baseline)

1. Household total within 1 % of D.
2. Each nation's weight share within 5 % relative of D's.
3. Lone-person share within one point of D's.
4. No local (grain, family) loses more than one point of within-10 share against S0.
5. National rows past 25 % at most D's count plus 2.
6. No constituency or local authority below 25 % of its dense ESS (María's
   ruling of 2026-10-07); the absolute < 50 counts and the minimum ESS are
   reported beside it.

## Predictions (recorded before the runs)

- C1 equals C0 within the noise of C5.
- The floor alone (C2) restores households, the NI share and the lone-person
  share. It also collapses per-area ESS: #355's Q50f at floor 0.001 put 643 of
  650 constituencies under 50.
- `initial` at floor 0 worsens monotonically with λ.
- At floor 0.5, E trades fit for per-area ESS.
- Refit-only A at floor 0 moves little: half the kept rows sit on the stretch cap.
- The selection stage, not the refit, is where A acts (step 2).

## Running it

`run.sh` holds the commands. Every compute step needs `--confirm-exclusive`
(no other solve on the machine). Weights stay outside the repository. Only
`publish` output, disclosure-controlled, is copied into `results/`.

No release gate runs in the harness: the acceptance criteria above are
measurements of every configuration. A configuration the solver chain refuses
(for example a refit that loses positive support) is recorded as a failed
receipt with its reason, and the run moves on to the next one.

## Results

<!-- results:start -->
| set | households vs D | NI share | lone share | constituencies < 50 | LAs < 50 | below 25 % of dense ESS | national past 25 % | grain_equal loss | criteria passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AE_gfes_init_f0.5_0.068 | -1.79 % | 2.06 % | 24.77 % | 168 | 19 | 140 | 12 | 0.0275 | – |
| AE_gfes_init_f0.5_0.21 | -3.28 % | 2.00 % | 23.28 % | 153 | 20 | 134 | 26 | 0.0427 | – |
| AE_gfes_init_f0.5_0.68 | -5.35 % | 1.97 % | 22.24 % | 169 | 27 | 174 | 60 | 0.0813 | – |
| AE_gfes_unif_f0.5_0.068 | -1.87 % | 2.40 % | 23.87 % | 11 | 7 | 44 | 17 | 0.0302 | – |
| AE_gfes_unif_f0.5_0.21 | -4.21 % | 2.52 % | 21.75 % | 2 | 6 | 29 | 46 | 0.0496 | – |
| AE_gfes_unif_f0.5_0.68 | -7.28 % | 2.99 % | 20.36 % | 0 | 5 | 29 | 180 | 0.1049 | – |
| A_gfe_f0 | -3.29 % | 2.16 % | 23.53 % | 221 | 31 | 195 | 13 | 0.0284 | – |
| A_gfe_f0.5 | -0.10 % | 2.59 % | 29.02 % | 646 | 266 | 996 | 9 | 0.0177 | household_total, lone_person_share, nation_shares |
| A_gfes_f0.5 | -0.06 % | 2.63 % | 28.97 % | 649 | 268 | 997 | 15 | 0.0193 | household_total, local_family_within_10, lone_person_share, nation_shares |
| A_ngfe_f0 | -3.46 % | 2.17 % | 23.46 % | 226 | 34 | 211 | 13 | 0.0300 | – |
| A_ngfe_f0.5 | -0.11 % | 2.51 % | 28.95 % | 648 | 268 | 999 | 6 | 0.0182 | household_total, lone_person_share |
| A_ngfes_f0.5 | -0.20 % | 2.55 % | 28.97 % | 649 | 273 | 999 | 11 | 0.0192 | household_total, local_family_within_10, lone_person_share |
| C1_f0_l2_1e-9 | -3.52 % | 2.01 % | 23.04 % | 171 | 20 | 154 | 14 | 0.0222 | local_family_within_10 |
| C2_f0.5 | -0.36 % | 2.50 % | 27.08 % | 591 | 154 | 898 | 4 | 0.0130 | household_total, local_family_within_10, national_past_25 |
| C3_f0.5_l2_1e-9 | -0.32 % | 2.49 % | 27.12 % | 594 | 160 | 899 | 4 | 0.0133 | household_total, local_family_within_10, national_past_25 |
| C4a_f0.1 | -0.27 % | 2.54 % | 27.14 % | 592 | 157 | 900 | 4 | 0.0126 | household_total, local_family_within_10, national_past_25 |
| C4b_f1.0 | -0.33 % | 2.49 % | 27.08 % | 590 | 152 | 894 | 5 | 0.0131 | household_total, local_family_within_10, national_past_25 |
| C5_lr0.1485 | -3.49 % | 2.01 % | 23.04 % | 167 | 20 | 156 | 15 | 0.0221 | local_family_within_10 |
| D | 0.00 % | 2.69 % | 28.63 % | 0 | 3 | – | 3 | 0.0113 | – |
| E_init_f0.5_1e-1 | -3.22 % | 2.02 % | 23.22 % | 129 | 17 | 112 | 13 | 0.0253 | – |
| E_init_f0.5_1e-2 | -0.99 % | 2.16 % | 24.86 % | 182 | 23 | 153 | 6 | 0.0164 | household_total, local_family_within_10 |
| E_init_f0.5_1e-3 | -0.49 % | 2.45 % | 26.07 % | 421 | 56 | 509 | 4 | 0.0141 | household_total, local_family_within_10, national_past_25 |
| E_init_f0.5_3e-2 | -1.73 % | 2.07 % | 24.28 % | 144 | 18 | 130 | 9 | 0.0198 | local_family_within_10 |
| E_init_f0.5_3e-3 | -0.66 % | 2.32 % | 25.41 % | 277 | 29 | 277 | 6 | 0.0150 | household_total, local_family_within_10 |
| E_init_f0_1e-1 | -26.97 % | 2.87 % | 25.59 % | 315 | 163 | 431 | 393 | 0.2446 | – |
| E_init_f0_1e-2 | -5.79 % | 2.09 % | 21.92 % | 167 | 19 | 162 | 17 | 0.0282 | – |
| E_init_f0_1e-3 | -3.61 % | 2.01 % | 23.01 % | 162 | 20 | 161 | 15 | 0.0225 | local_family_within_10 |
| E_unif_f0.5_1e-1 | -4.41 % | 2.27 % | 21.76 % | 3 | 7 | 32 | 12 | 0.0260 | – |
| E_unif_f0.5_1e-2 | -0.76 % | 2.38 % | 24.86 % | 65 | 9 | 69 | 7 | 0.0163 | household_total, local_family_within_10 |
| E_unif_f0.5_1e-3 | -0.56 % | 2.51 % | 26.20 % | 422 | 40 | 434 | 4 | 0.0142 | household_total, local_family_within_10, national_past_25 |
| E_unif_f0.5_3e-2 | -2.21 % | 2.29 % | 23.45 % | 19 | 7 | 49 | 9 | 0.0195 | local_family_within_10 |
| E_unif_f0.5_3e-3 | -0.32 % | 2.46 % | 25.57 % | 200 | 19 | 189 | 6 | 0.0149 | household_total, local_family_within_10 |
| E_unif_f0_1e-2 | -3.94 % | 2.01 % | 22.57 % | 117 | 13 | 101 | 15 | 0.0230 | – |
| S0 | -3.47 % | 2.01 % | 23.04 % | 170 | 20 | 158 | 15 | 0.0221 | local_family_within_10 |

Step 1b picks:

- best E (initial): None; knee λ 0.1
- best E (uniform): None; knee λ 0.1
- best A: A_gfes_f0.5
- A×E rule grain_family_equal_sqrt_count, λ × 2.144
- best A×E: AE_gfes_unif_f0.5_0.21

Refit-level holdouts (means over the local folds):

- H_AE_gfes_unif_f0.5_0.21: held loss 0.0934 under its rule, 0.1291 under grain_equal; within 10 % 65.4 %, within 25 % 90.3 %
- H_A_gfes_f0.5: held loss 0.1536 under its rule, 0.2122 under grain_equal; within 10 % 50.7 %, within 25 % 74.6 %
- H_C0: held loss 0.1146 under its rule, 0.1146 under grain_equal; within 10 % 70.5 %, within 25 % 89.7 %
- H_C2_f0.5: held loss 0.2024 under its rule, 0.2024 under grain_equal; within 10 % 53.4 %, within 25 % 76.9 %

Step 0 census:

- self-checks passed: True
- k_min at the search cap: 24007 (nations summed: 28860) for 60000 households
- areas no refit on this support can lift to criterion 6: constituency 18, la 11
- early size (F) trigger: True
- capacity / D at floor 0: England 10.38, Northern Ireland 12.36, Scotland 6.35, Wales 8.71
- capacity / D at floor 0.1: England 10.32, Northern Ireland 5.54, Scotland 8.75, Wales 9.14
- capacity / D at floor 0.5: England 10.32, Northern Ireland 5.49, Scotland 8.81, Wales 9.13
- capacity / D at floor 1: England 10.32, Northern Ireland 5.49, Scotland 8.82, Wales 9.12
- uniform anchor: suggested grid shift 1 half-decades
- initial anchor at floor 0: suggested grid shift -2 half-decades
- initial anchor at floor 0.1: suggested grid shift 0 half-decades
- initial anchor at floor 0.5: suggested grid shift -1 half-decades
- initial anchor at floor 1: suggested grid shift -2 half-decades
<!-- results:end -->
