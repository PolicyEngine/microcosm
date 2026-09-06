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

1. Contract identity is declared in the donor, recipient and support node
   params and verified against the loaded contract on execution.
2. Supplied recipient household sizes are compared exactly before any cast.
3. Close issues: declared numpy/pandas dependencies; closed-roster symlink,
   shard-placement and duplicate refusals; `scratch == store.root`; bounded
   object copy; a signed-but-finite target conversion.
4. 62 tests pass (38 existing, 24 new). `ruff check .` is green across the
   repository; it reported 12 errors at the parent. `ci_test_groups --verify`
   is ok and the new file lands in `us-qs` / `rest`, not `[defaulted]`.
5. The root's own counterexample script, repointed at this worktree, now
   fails where it previously observed five warm hits. The adapted probe
   `R/us-sipp-equity-review-fixes-counterexamples.py` passes.

## State

Complete and reported. Nothing pushed; no PR, merge or deploy. The code
awaits the root's peer review before any genuine fit.

## Not done, deliberately

No genuine fit, no 2024 bridge, no PolicyEngine alias, no weights
calibration, no scientific validation. Source flag 9 stays unresolved. The
global runtime-classification lane and the parent journal are untouched.
