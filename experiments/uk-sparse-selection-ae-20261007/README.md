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
4. Report to María. Step 2 (search re-runs) and step 3 (size) run only on her pick.

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
Not measured yet.
<!-- results:end -->
