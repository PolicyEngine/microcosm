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

1. **Step 0** (`tools/run_uk_size_experiment.py build-cache`, then `control`): the
   stored refit must come back bit for bit before any variant is read.
2. **Step 1a**, `experiments.json` (26 refits on the stored support):
   - controls C1–C5: iterate switch, floor only, floor plateau, optimiser noise;
   - E under `grain_equal`: floor 0 `initial` at 1e-3, 1e-2 and 1e-1 (predicted harmful), floor 0 `uniform` at 1e-2, and floor 0.5 at 1e-3, 3e-3, 1e-2, 3e-2 and 1e-1 for each anchor;
   - A at λ 0: the four rules at floor 0.5, and `grain_family_equal` and `nation_grain_family_equal` at floor 0.
3. **Step 1b**, written after 1a by the rules below:
   - A×E: the best A rule × each anchor × 3 λ around that anchor's E knee, with λ scaled by the A rule's loss level at S0;
   - holdout: a refit-level rotated holdout (`mode: refit_holdout`, 5 local folds) for C0, C2, the best E per anchor, the best A and the best A×E.
4. Report to María. Step 2 (search re-runs) and step 3 (size) run only on her pick.

**Selection rules** (fixed now):
- the best E per anchor is the largest λ that still meets criteria 1–5 of the acceptance below;
- the best A is the rule that meets the most criteria, ties broken by the `grain_equal` yardstick loss;
- the knee is the largest λ whose criterion-6 count is within 10 % of that anchor's best.

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

## Results

<!-- results:start -->
Not measured yet.
<!-- results:end -->
