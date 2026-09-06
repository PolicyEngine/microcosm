# PROGRESS — composed prepared-ASEC + native-ACS population graph

Lane branch `microcosm-us-composed-population-20260906`, based on
`us-launch-integration-20260906` at `ab10f77854b7b9e8057a07dbec40d15c10b96297`.
`origin/main` (`c1b83241`) is an ancestor of that base; the base is 124 commits
ahead and 0 behind. Review root for receipts:
`~/architecture-reviews/microcosm-20260904`, files prefixed
`us-composed-population-`.

## State

Orientation only. Worktree created, base relation recorded, repo AGENTS.md /
CLAUDE.md read. No source or kernel changes yet.

## Done

- Created the dependent worktree and branch from the frozen base commit.
- Recorded the base/origin-main relation.
- Read `graph_sources.py` (the existing `asec_raw_stage` + `acs_native`
  assembly seam that must not change) and the head of `graph_asec_prepared.py`.

## Next

- Read the prepared-ASEC and ACS housing-universe source contracts in full and
  test whether the prepared Frame satisfies `prepare_stacked_spine`.
- Build the separately versioned composed seam over the prepared source.
