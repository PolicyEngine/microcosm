# PUF monetary aggregate lexical separation — lane journal

This is the lane journal for `us-puf-aggregate-lexical-20260906`. The root
`PROGRESS.md` is cumulative history for other lanes and is not touched here.

## State

Implementing the bounded source-contract correction adjudicated by root in
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

## Next

- [ ] Aggregate lexical grammar, sentinel, `amount_known`, parse ordering.
- [ ] Facts, column metadata and envelope key-set changes; version 3 identity.
- [ ] Aggregate-only summary and the local lexical accessor.
- [ ] Tests, docs, changelog fragment, report.
