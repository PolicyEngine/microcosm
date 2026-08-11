# Round 8 progress: cross-origin structural absence

## State

Investigating the real 1% smoke-r5 `TYPEHUGQ` failure on
`tail-stratum-support-652` at `27b07c73`. The worktree started clean. Work is
local-only: no build, push, GitHub operation, network access, state shelving,
or unrelated root-journal edit.

## Done

- Read `CLAUDE.md` and the applicable debugging workflow.
- Confirmed the requested branch and exact starting commit.
- Compared the checkout with the locally cached `origin/main`. The branch is
  75 commits ahead and 15 behind that cache; the requested checkout is being
  preserved because this round explicitly targets PR #660 at `27b07c73` and
  forbids network access.
- Confirmed the GitNexus graph tools are unavailable in this session; the
  equivalent call-site and execution-flow trace will be performed directly
  from source, checkpoint receipts, and tests.

## Next

- Inspect smoke-r5 checkpoint and launcher receipts to identify the exact
  `TYPEHUGQ` rows and consuming lineage.
- Enumerate every origin-specific raw input required over `whole_pool`, then
  choose and implement the narrowest honest absence declaration for each.
- Add the exact 1,688-row regression, ACS-side fail-closed regression, and an
  exhaustive cross-origin raw-input audit.
- Run the requested focused, #583, full-workspace, formatting, lint, and diff
  checks; obtain an independent read-only review; report the smoke-r6
  prediction to stdout.
