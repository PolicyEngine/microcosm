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
- [x] verbatim copy of the four transport files + blob-hash proof (`c5a7feec`)
- [x] doc geographic-release-scope trim (`3a970f95`)
- [x] additive CI inventory registration + changelog fragment (`e9fe5604`)
- [x] targeted tests, neighbour tests, ruff check, ruff format --check, `--verify`
- [x] live writer+reader+CLI exercise on a temporary dedicated output root
- [ ] fast-lane `rest` group (283 files) -- running
- [ ] transport correctness assessment -- multi-agent review running
- [ ] report to R/candidate-quality-producer-integration-author.md

## Verification so far

Environment: `uv sync --all-packages --locked` (lockfile unchanged, no upgrades),
Python 3.14.4. `microcosm.build.candidate_quality` resolves to this worktree's
`packages/microcosm-build/src/...`, sha256 `3c202ff4...`, equal to the committed blob.

- `pytest packages/microcosm-build/tests/test_candidate_quality.py` -- 25 passed
- `pytest test_quality_aggregates.py test_cross_grain.py` -- 98 passed
- `ruff check .` -- clean; `ruff format --check` on the four changed files -- clean
- `tools/ci_test_groups.py --verify` -- exit 0; the new test is in fast `rest`
  and engine `shared-spec`, and no longer under `[defaulted]`
- Live exercise, 29/29 checks: genuine `quality_aggregates` support/distribution
  output over an INVENTED 4-household toy Frame, sealed and read back. Exact byte
  preservation across non-canonical indentation; crossed candidate/population
  identities refused with nothing written; tampered manifest and tampered
  attachment both fail their digests; superseded manifests stay readable; a
  deleted index does not block a direct digest read; the CLI resolves a relative
  source and `--no-index` writes none.
- Separate path-guard probe: relative, nested, absolute, backslash and colon
  partition paths are all refused; only the control is accepted.

## Observed, worth recording

`provenance` and `geography` are omitted from the manifest entirely when the
caller does not supply them, rather than emitted as `{}` / `null`. The frozen
dashboard reader tolerates this (`strings(undefined)` yields `{}`;
`raw.geography != null` leaves it null), so it is compatible, not a defect --
but the TS type declares both as required fields, so it reads as looser than
the type suggests.

## Next

Finish the fast-lane run and the correctness assessment, then write the report.
