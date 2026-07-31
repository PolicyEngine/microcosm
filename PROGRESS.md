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

## Next

- Commit the completed guard contract, implementation, regressions, and this
  receipt.
- Reverify adult-care and SSI constant-plumbing value preservation.
- Run the full `populace-build` suite and Ruff; update the PR body and write
  `/private/tmp/583_fix4_handoff.md`.
- Restore `PROGRESS.md` byte-for-byte to `origin/main` before final handoff.
