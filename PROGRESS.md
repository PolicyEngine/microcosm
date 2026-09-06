# ACS housing universe — Fable follow-up (bounded)

Branch `us-acs-hu-fable-followup-20260905`, dependent worktree from frozen
`375efae2a2f861f8283e771c0b92a38f4fe46a25` (parser fixes). Parent worktree
`_worktrees/microcosm-us-acs-housing-parser-fixes-20260905` and the original
`6b9f91ce` worktree `_worktrees/microcosm-us-acs-housing-universe-source-20260905`
are preserved untouched.

Source review: `architecture-reviews/microcosm-20260904/us-acs-hu-source-fable51-review.md`
(head 6b9f91ce, base a276a3c9, verdict REQUEST_CHANGES).

## Scope

Four required Fable items plus one optional cheap check and a docs note.
No genuine ACS archives, no network, no PE data fit/calibration/alias/scoring,
no cap raises, no big-data architecture. Registry/inventory untouched unless
runtime code forces a scoped update.

## State

- [x] Status/fetch/compare before new child (parent clean at 375efae, 96 ahead / 0 behind `origin/main` c1b83241)
- [x] Dependent worktree created, PROGRESS.md committed
- [ ] Read actual source of the two runtime modules + two ACS tests
- [ ] RED tests for items 1-4 (+ optional 5) against 375 production
- [ ] Fixes for items 3, 4 (+ optional 5)
- [ ] Docs note: 64 MiB envelope may need seed selection (no size forecast as fact)
- [ ] GREEN: full affected files, own shards, unique receipts
- [ ] Freeze commit + durable report/receipt

## Next

Read `acs_housing_universe_source.py` and `graph_acs_housing_universe.py` at the
exact line ranges the review cites, before writing a single test.

## Done

Nothing beyond scaffolding yet.
