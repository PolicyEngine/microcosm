# income-v3 graph export — Fable 5.1 review fixes

Child worktree of the root exporter head. Scope: close the actionable items in
`us-income-v3-graph-export-fable51-review.md` (REQUEST_CHANGES) narrowly, without
exporting or installing anything.

## State

- Worktree: `_worktrees/microcosm-us-income-v3-export-review-fixes-20260906`
- Branch: `us-income-v3-export-review-fixes-20260906`
- Base (root exporter head): `1056bd56ac4c483d7afb03d9c1e255a244eb1f95`
- Upstream base: `origin/main` = `c1b8324183819cf91511c002bf724a2676524e87`
- Started: 2026-09-06

## Done

- (nothing yet)

## Next

1. Read the reviewed four-node + income-v3 exporter/test/doc bytes and record
   their pre-change hashes.
2. Fix finding 1: publish the source contract's own price basis name.
3. Fix finding 2: bind engine `evaluation_sha256` and accounting `accounting_sha256`.
4. Fix finding 3: validate each graph `reported_income` summary against direct.
5. Fix finding 4: port the four-node resealing helper; assert specific refusals.
6. Fix finding 5: accounting-column, carried-cells and lineage negatives.
7. Re-run old + new suites, ruff, and write the report.

## Constraints

- Local only. No source opens, no model runs, no push, no deploy, no export.
- The provisional export at `R/us-income-v3-genuine-graph-candidate` is a
  REJECTED audit artifact; never overwrite it.
