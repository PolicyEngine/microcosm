# SIPP nominal-equity review fixes — journal

New journal for the review-fix child. The parent journal
`PROGRESS-us-sipp-nominal-equity-first-fit.md` is left in place unchanged;
it is the build record, this is the fix record.

Branch `us-sipp-equity-review-fixes-20260906`, a fresh dependent child of
`963cf8ca74e696f774b003d9a6e832f79a18e573` (itself a descendant of the accepted
integration commit `9ece4cadddd1933524231a46a959086b0ff74429`) at
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-sipp-equity-review-fixes-20260906`.
Nothing is pushed. No PR, no merge, no deploy, no genuine fit.

Root verdict being answered: `R/us-sipp-equity-root-review.md`, REQUEST_CHANGES,
with the executed counterexamples in
`R/us-sipp-equity-root-counterexamples-v2.py/.json/.log`.

## State

Started. Read the three controlling plans, the first-fit REPORT, the root
review and its counterexample script/output, and the full four production
files plus the graph core's `keys.py` / `kernel.py` identity path.

## Scope (unchanged from the parent)

Nominal December-2022 signed `THEQ_VEH`, household size as the only predictor,
two trees, source flag 9 unresolved. No 2024 bridge, no PolicyEngine alias,
no scientific fit, no weights calibration. Invented fixtures only.

## Done

- (nothing yet)

## Next

1. Bind the closed contract identity into donor/recipient/support node params
   and verify param-vs-kernel-contract on execution.
2. Compare supplied recipient household sizes exactly before any int64 cast.
3. Concrete close-issue fixes: declared numpy/pandas dependencies, closed-roster
   shard placement and duplicate refusal, `scratch == source_root`, bounded
   object copy, signed-but-finite target conversion.
4. Regression tests, full suite, lint, CI registration check.
5. Report + capture into R.
