# Candidate quality producer transport — integration composition

Session journal (history, not state). Dated 2026-09-05.

## Scope

Freeze the already-written candidate evidence *transport* into the integration
lineage so a reviewable producer exists before the next genuine candidate.
This is a bounded composition, not a redesign and not new science.

- Base: integration `1ae0d347986d8d69d9461a09ab367237934f00a9`
  (60 ahead of `origin/main` `c1b83241`, 0 behind).
- Branch/worktree: `candidate-quality-producer-integration-20260905` in
  `_worktrees/microcosm-candidate-quality-producer-integration-20260905`.
- Source of the copied bytes: staged (uncommitted) work in
  `_worktrees/microcosm-candidate-quality-20260905` at base `96faa5d5`
  (an ancestor of the integration head). That checkout is READ ONLY here.

## In scope (copy only)

- `packages/microcosm-build/src/microcosm/build/candidate_quality.py`
- `packages/microcosm-build/tests/test_candidate_quality.py`
- `tools/build_candidate_quality.py`
- `docs/candidate-quality.md` (transport sections; geographic-release-scope
  section trimmed — see below)

## Explicitly out of scope (separate scientific/policy work)

`us_release_scope.py` and its tests, the CD acceptance protocol draft, the
fiscal/exact-k CLI changes, the old `CLAUDE.md` edit, and the old
`tools/ci_test_groups.py` overwrite (which would have reverted two entries
that landed in the integration lineage since `96faa5d5`).

## State

- [x] status/fetch/base comparison; fresh worktree from the explicit integration commit
- [x] PROGRESS.md scaffold
- [ ] verbatim copy of the four transport files + blob-hash proof
- [ ] doc geographic-release-scope trim
- [ ] additive CI inventory registration + changelog fragment
- [ ] tests / lint / CI inventory verify
- [ ] live writer+reader exercise on a temporary output root
- [ ] transport correctness assessment
- [ ] report to R/candidate-quality-producer-integration-author.md

## Next

Copy the four files and prove staged blob == copied blob.
