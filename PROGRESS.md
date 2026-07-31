# Progress

## State

PR #583 final guard remediation is implemented on
`multispine-pool-build-578`. The contract now certifies only enumerated static
surfaces and names adversarial construction and runtime-materialized names as
explicit boundaries. The full guard lane is green, including every runtime
module and the pinned 54-module build graph.

## Done

- Read `CLAUDE.md` and the PR-review remediation workflow.
- Confirmed the requested branch, commit, clean worktree, and round-5 HOLD
  receipt at `/Users/maxghenis/PolicyEngine/_buildo-runtime/reviews/sol_583_r5.log`.
- Clustered the requested work into contract narrowing, four natural-class
  catches, explicit pinned out-of-scope misses, runtime constant-plumbing
  verification, and handoff accuracy.
- Replaced the unsupportable completeness claim with the owner-directed
  tripwire contract and its two named out-of-scope classes.
- Added static folding for `lower`, `upper`, `casefold`, `title`, and
  `capitalize`.
- Propagated all static loop/comprehension choices through `str.format` and
  f-string interpolation; the exact `for-entity-format` repro now reports both
  guarded columns by name.
- Routed direct `df.__getitem__(column)` through the same selector analysis as
  `df[column]`.
- Added named-expression factory alias discovery; strict-method walrus
  aliasing is pinned for both guarded and opaque expressions.
- Added the exact round-5 reverse-slice, `format_map`, `__doc__`,
  `__annotations__`, and container-indexed-method repros to
  `test_documented_out_of_scope_evasions_are_not_caught`, asserting their
  intentional current misses.
- Passed the complete guard file: 113 tests.
- Reverified the value-preserving runtime constant plumbing with the full
  adult-care and SSI files: 85 passed, 2 skipped. `support_provenance.py` owns
  the unchanged `"person_support_channel"` value; guarded consumers
  `adult_care.py` and `ssi_take_up.py` import it without behavior changes.
- Passed repository-wide Ruff and `git diff --check`.
- Passed the full `populace-build` suite: 3,335 passed, 85 skipped, with five
  pre-existing warnings, in 107.47 seconds.
- Prepared an accurate replacement PR body covering the enumerated contract,
  the sole `operator_boundary.py` owner addition, and all three runtime
  constant-plumbing files. The live connector update was canceled, so the
  replacement will be preserved verbatim in the final handoff rather than
  applied through another channel.

## Next

- Write `/private/tmp/583_fix4_handoff.md` with exact receipts and the
  unapplied PR-body replacement.
- Restore `PROGRESS.md` byte-for-byte to `origin/main` before final handoff.
