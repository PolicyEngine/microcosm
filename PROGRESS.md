# Candidate quality transport — integration composition

Session journal for the bounded composition that lands the existing candidate
evidence transport in the integration lineage. History, not state: check git
and the delivered report for current truth.

## Scope

Compose only the reviewed evidence *transport* onto integration
`1ae0d347986d8d69d9461a09ab367237934f00a9`:

- `packages/microcosm-build/src/microcosm/build/candidate_quality.py`
- `packages/microcosm-build/tests/test_candidate_quality.py`
- `tools/build_candidate_quality.py`
- `docs/candidate-quality.md` (geographic-release-scope section trimmed)

Deliberately excluded — separate scientific/policy work, not this lane's:
`us_release_scope.py` and its tests, the CD acceptance protocol draft, the
fiscal/exact-k CLI changes, the old `CLAUDE.md` edits, and the old
`tools/ci_test_groups.py` (whose `EXPLICIT_SHARED_SPEC` predates two entries
that exist on integration — registration here is additive instead).

Source is the staged-but-uncommitted work in
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-quality-20260905`
at `96faa5d5`. That checkout is read-only for this lane: blobs are read by OID
through `git cat-file`, never through its working copy.

## State

Bootstrapping.

## Done

- Confirmed integration head `1ae0d347` is 60 commits ahead of, and a clean
  descendant of, `origin/main` `c1b83241`.
- Confirmed old base `96faa5d5` is an ancestor of `origin/main`.
- Extracted the four staged blobs by OID; all four sha256 match the recorded
  `candidate-quality-producer-staged-blobs.json` manifest.
- Created this worktree/branch from the explicit integration head.

## Next

- Copy the four files byte-exact and prove blob equality.
- Trim the doc's geographic-release-scope section.
- Assess transport correctness; land any concrete fix as its own explained commit.
- Tests, ruff, `tools/ci_test_groups.py --verify`; freeze; write the report.
