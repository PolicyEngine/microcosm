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

Transport copied, doc scoped, registrations added. Composition exercised
end-to-end against both the local reader and the frozen dashboard reader.
Awaiting the independent correctness review before freezing.

## Done

- Confirmed integration head `1ae0d347` is 60 commits ahead of, and a clean
  descendant of, `origin/main` `c1b83241`.
- Confirmed old base `96faa5d5` is an ancestor of `origin/main`.
- Extracted the four staged blobs by OID; all four sha256 match the recorded
  `candidate-quality-producer-staged-blobs.json` manifest.
- Created this worktree/branch from the explicit integration head.
- Copied all four files byte-exact; each path's blob OID in the copy commit
  equals its staged OID.
- Trimmed the doc's geographic-release-scope section, which described an
  exact-k schema 2, `--analysis-scope`, `--acceptance-policy`,
  `--gate-congressional-district-targets` and a CD acceptance draft link that
  do not exist in this composition. Replaced with what is true here:
  eligibility is recorded rather than computed, arrives from a separate
  authoritative acceptance producer, and no geographic eligibility is created.
- Registered the test additively in `EXPLICIT_SHARED_SPEC` (one added line;
  the staged version of that file would have dropped two entries integration
  has since gained). Added a changelog fragment scoped to the transport only.
- 25/25 transport tests pass; ruff check and ruff format clean repo-wide;
  `tools/ci_test_groups.py --verify` exits 0 with the file no longer
  `[defaulted]` and present in both the `rest` and `shared-spec` groups.
- Ran a 16-assertion end-to-end probe of the real writer and reader on a
  dedicated temp root: exact byte preservation, content-addressed manifest,
  only `artifacts/` and `manifests/` written, and clean `CandidateQualityError`
  refusals for tampering, crossed identity, run-scope-claiming-population,
  embedded alias mismatch, non-JSON source, eligible-without-completion and
  escaping paths.
- Ran the frozen `334b826` dashboard reader (against a copy whose reader files
  hash-match the live worktree, so that checkout stayed untouched) over
  evidence written by this composition: ordinary evidence accepted with its
  diagnostics and estimate intact; tampering gives `hash_mismatch`; a crossed
  candidate id and a wrong manifest digest are refused; a run-scoped
  attachment is refused as population evidence.

## Next

- Fold in the independent review's confirmed findings, each as its own
  explained commit.
- Re-run tests, lint and the CI inventory check; freeze; write the report to
  `R/candidate-quality-producer-integration-author.md`.
