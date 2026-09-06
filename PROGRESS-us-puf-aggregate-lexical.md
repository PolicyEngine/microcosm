# PUF monetary aggregate lexical separation — lane journal

This is the lane journal for `us-puf-aggregate-lexical-20260906`. The root
`PROGRESS.md` is cumulative history for other lanes and is not touched here.

## State

Completed locally on 2026-09-06: the bounded source-contract correction adjudicated by root in
`us-puf-monetary-aggregate-separation-root-adjudication.md`, over Fable's
row-preserving representation. Code only: no genuine run, no release, no
runner change, no source or status re-pin.

## Provenance

- Reviews read: `us-puf-monetary-aggregate-separation-plan.md`,
  `puf-aggregate-separation-fable51-capture.md`,
  `us-puf-monetary-aggregate-separation-root-adjudication.md` (root wins).
- Accepted source worktree `microcosm-us-puf-monetary-root-20260906` at
  `ee530cb95276e39793bbd99da47cc2680448885e`, inspected clean, left untouched.
- This worktree branched from `origin/main` (`c1b832418`) and fast-forwarded
  onto the accepted dependency `ee530cb95`.
- Aggregate-only input evidence: `us-puf-monetary-delivered-lexical-audit.json`
  (4 fractional / 207,692 integer tokens in each of 12 fields) and
  `us-puf-monetary-aggregate-token-alignment.json` (the same 4 tokens are the
  disclosure aggregates in every field; 175 module files verified against
  `ee530cb9`). No PUF, status artifact or capture byte is read by this lane.

## Done

- [x] Read the plan, the Fable capture and the root adjudication.
- [x] Inspected the accepted worktree status and head; created this child.
- [x] Implemented the aggregate lexical grammar, sentinel, immutable
  `amount_known`, and status alignment before amount parsing.
- [x] Added version-3 identities, per-return numeric facts and aggregate token
  classes, and metadata for the universe, lexical authority and operational
  digit/width limits. Kept the raw source module and both resource files exact.
- [x] Added the aggregate-only summary and exact local lexical accessor.
- [x] Reviewed the partial implementation after the external session limit;
  fixed reversible NumPy write protection and missing encoded grammar bounds.
- [x] Ran the focused monetary/raw invented-data suites: 302 passed; ruff
  check and format check passed on both changed Python files.
- [x] Finished the source-contract documentation and changelog fragment.
- [x] Preserved the accepted version-2 dependency and genuine failed-run evidence.

## Next

The external handoff is
`architecture-reviews/microcosm-20260904/us-puf-aggregate-lexical-REPORT.md`
and its JSON companion, under `/Users/maxghenis`. They record the final head,
loaded module origins and completed checks without hashing this journal into
itself. A separate reviewed runner amendment must update version-3 identities,
metadata, row-class reconciliation and mask/sentinel checks before any genuine
rerun. This lane does not change or execute that runner. The standing build
order remains the combined ASEC+ACS pool first, then PUF-enriched clones of that
pool; no prior-wage inputs are introduced.
