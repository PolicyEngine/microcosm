# SIPP equity first genuine fit — preparation journal

Session journal for branch `us-sipp-equity-first-genuine-preparation-20260906`.
The worktree was created from `origin/main` (`c1b83241`) and fast-forwarded to
the reviewed dependent head `9372fb6b` before any work here.

Local only. Nothing pushed, no PR, no merge, no deploy, **no genuine fit and no
genuine source open**. This branch *prepares* a reviewed genuine rung; it does
not execute one. Root reviews before anything genuine runs.

This journal is history, not state; check git for current truth.

## Why this branch exists

Two separate jobs, in one bounded preparation:

1. Close the three test/documentation items Fable left non-blocking in
   `us-sipp-equity-persistence-final-fable51-review.md` (APPROVE at `9372fb6b`).
   **Production bytes stay identical to `9372fb6b`.**
2. Author, but do not run, the first genuine equity fit: a runner, a roster,
   invented preflight tests and a report. Execution requires a separately
   authored root/peer approval bound to the runner, the roster and the head.

## State — repository work complete

- [x] Fetch `origin`; confirm `origin/main` = `c1b8324183819cf91511c002bf724a2676524e87`
      is an ancestor of `9372fb6b`, so the fast-forward is real, not a rebase.
- [x] Confirm the sibling worktree exists at the required path, created from
      `c1b83241` and fast-forwarded to `9372fb6b` (reflog), clean, no work on it.
- [x] Read the Fable review, the root plan and its v2/v3 addenda, the packaged
      contract, the prior graph roster, and the accepted source and
      weighted-coverage root reviews.
- [x] Fable item 1 — the fresh-process warm subprocess reports each loaded
      microcosm module's resolved path and whether it is inside the worktree
      root; the parent asserts every module is inside this checkout and that
      the bytes there still hash to the reported digest.
- [x] Fable item 2 — the contradictory recipient docstring in
      `test_a_second_valid_source_store_moves_the_donor_and_descendant_keys`.
- [x] Fable item 3 — admission exhaustiveness derived from the contract's own
      fields, not from three literals that cannot disagree.
- [x] Focused tests: **99 passed, 0 failed, 0 skipped**, unchanged in count and
      per-file distribution (18 / 15 / 7 / 24 / 35).
- [x] `ruff check` and `ruff format --check` on both changed files: pass.
- [x] `git diff 9372fb6b -- 'packages/*/src' docs changelog.d` is empty apart
      from the one new changelog fragment: **no production byte moved.**

## What is deliberately not in this repository

The genuine runner, its roster, the invented preflight tests, the per-process
module evidence and the report live in the review directory
`~/architecture-reviews/microcosm-20260904`, as every prior genuine rung's
harness does. They are external evidence harnesses, not shipped code, and they
pin **this commit** as the frozen head. Adding them here would put an
unreviewed execution path inside the package.

Root reviews the runner, the roster and the head before any genuine fit. The
runner refuses to execute without an approval JSON naming all three.

## Not done here, ever, on this branch

No genuine fit. No open of the genuine projection, its binary, the source CSV
or the accepted store. No push, PR, merge or deploy. No new dependency, no new
environment, no PolicyEngine import, no 2024 bridge, no PolicyEngine alias, no
calibration and no population-quality claim.
