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
no cap raises, no big-data architecture. Registry, inventory and the pinned
archive manifest are untouched: the implementation cases stayed green without
any scoped update, so none was made.

## State

- [x] Status/fetch/compare before new child (parent clean at 375efae, 96 ahead / 0 behind `origin/main` c1b83241)
- [x] Dependent worktree created; progress file committed from the start
- [x] Read the two runtime modules and the two ACS test files at the cited lines
- [x] RED: 4 new cases failed against 375 production (`us-acs-hu-followup-tests-red`)
- [x] Fixes for review items 3 and 4, plus the optional receipt-pin binding
- [x] Docs: seeded CREATE, 64 MiB envelope, vacancy refusals, new reason codes
- [x] GREEN: 310/310, zero failures/errors/skips (`us-acs-hu-followup-tests-green-first`)
- [x] Restored the pre-existing repo `PROGRESS.md` this file had wrongly replaced
- [ ] Freeze commit + durable report/receipt

## Done

- **Item 1 (test only).** `test_create_seed_subset_then_select_subset_cold_and_warm`
  seeds CREATE with `serialnos`, prunes further at SELECT, and runs cold then
  warm through a real `ContentStore` with both kernels forbidden on replay. It
  asserts the retained person index equals the parent's labels and is *not*
  `0..n-1`. This passed against 375 production: the review's concern was a
  coverage gap, not a defect, and the gapped index now has a proof.
  `test_vacancy_only_seed_and_selection_refuse` covers both empty routes.
- **Item 2 (test only).** `test_registered_codec_loader_prepares_through_a_real_store`
  resolves the loader from `us_source_codecs()` and calls it through a real
  store, checking the `acs-hu-source-captures-v1` child, its single capture and
  that the caller's source directory is untouched. Also passed as written.
- **Item 3 (fix).** Candidate readback size failures were raising the pinned
  archives' own `SOURCE_SIZE` / `FILE_TOO_LARGE`. Only that one `_copy` call's
  size refusals are renamed to `RECONSTRUCTION_SIZE`; every other code from that
  call, and every parent-capture refusal, is re-raised unchanged. Two tests:
  one for the candidate code, one asserting the parents keep theirs.
- **Item 4 (fix).** `_capture`'s failure record is now written under
  `suppress(Exception)`, so an unwritable `failure.json` no longer replaces the
  refusal it records. Previously the OSError surfaced as
  `SOURCE_PREPARATION_REFUSED`, losing `SOURCE_SHA256`.
- **Optional.** `bind_acs_housing_evidence` now checks the source receipt's
  `format`, `release_eligible`, `vintage` and `archives` against `_pins()`
  (`SOURCE_PINS`), so the evidence is self-verifying rather than covered only
  transitively through the implementation hash.

## Not done, deliberately

No genuine source run, no separate-process fresh-worker probe (the 6b9
cold/warm fresh-worker results stay attributed to 6b9), no cap change, no
size forecast stated as fact, no push, no merge.
