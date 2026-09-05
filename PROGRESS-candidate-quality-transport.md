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

## Independently verified probe results (2026-09-05)

Confirmed sound:
- Manifest digests are reproducible across roots for identical inputs.
- Re-sealing different bytes at an existing content-address is refused
  ("Existing immutable artifact differs").
- An index whose country disagrees with the candidate is refused.
- A model-scoped identity that contradicts a declared `provenance.model_version`
  is refused. With no provenance declared there is nothing to contradict, so the
  model identity is self-declared only -- provenance is what binds it.
- Relative, nested, absolute, backslash and colon partition paths are all refused.
- No absolute local filesystem path reaches the written manifest or index.
- Literal `NaN` / `Infinity` / `-Infinity` tokens are refused.

Confirmed gaps (reported, deliberately NOT patched -- see below):
- G1: `1e400` / `-1e400` bypass the `parse_constant` hook, because that hook only
  fires on the literal tokens. The value is sealed, survives verbatim, and a
  consumer re-parsing it gets `Infinity` (JS `JSON.parse` agrees). Metadata is
  protected by `_bytes(allow_nan=False)`; attachment payloads are not.
- G2: `RecursionError` from deeply nested JSON escapes uncaught -- the handler
  catches `(ValueError, UnicodeError)` and `RecursionError` is neither. Same for
  `FileNotFoundError` / `IsADirectoryError` on a missing or non-regular source,
  and on a read for an absent manifest digest. Callers cannot uniformly catch
  `CandidateQualityError` to separate bad evidence from an environment problem.
- G3: an attachment payload that omits its identity fields entirely is accepted.
  The embedded cross-check only compares keys the payload actually declares, so
  such a payload's identity binding rests wholly on the producer's sidecar claim.
  This is the attested-versus-verified line and belongs in the report, not in a
  silent patch.

Decision: the transport bytes stay verbatim. The point of this change is to
freeze the already-reviewed writer for review; adding unreviewed fixes to it
would undercut that. G1--G3 go to root and the independent review to decide.
