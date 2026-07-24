# Progress

## State

Populace #516 whole-row donor outlier screen is complete on
`mortgage-donor-outlier-screen` (rebased onto `origin/main` after the #515
interim carve merged as #525). The `puf_tax_detail` donor now drops tax units
whose grouped raw mortgage interest reaches $10M before the #515 carve
(pinned-artifact effect: 3,066 rows, weight 3,684 of ~161M, removing $2.947T
of phantom mortgage-interest mass), with the checkpoint schema bumped to v3
so post-carve pre-screen checkpoints rebuild.

## Done

- Confirmed a clean starting worktree at `aef1c56`.
- Read the repository guidance and established the #515 donor carve as the
  screen's required downstream boundary.
- Started source-level audits of every donor-frame consumer, checkpoint
  validation, row-count pins, and existing donor-fact summaries.
- Attempted the requested GitNexus impact workflow; the managed filesystem
  denied its global registry write. Its local index also exposed a broad
  `build/` ignore mismatch, so the completed impact audit uses direct source
  call sites and tests.
- Added `US_PUF_DONOR_MORTGAGE_OUTLIER_CEILING = 10_000_000.0` with the
  structural rationale and pinned-artifact receipts.
- Added a whole-row screen on grouped raw person `home_mortgage_interest`
  after tax-unit assembly, before the #515 carve, with retained-index reset.
- Confirmed no downstream consumer pairs donor rows to the original HDF arrays
  or carries a stale donor-length vector; values and weights always originate
  from the same screened frame.
- Bumped the primary QRF checkpoint schema from v2 to v3 and made the stale
  checkpoint regression track the live constant while retaining literal-v1
  corruptions.
- Added regression coverage for the exact grouped boundary, whole-row removal,
  retained/carved $5M row, raw-$10.5M pre-carve ordering, and constant.
- Requested suites pass: PUF support/QRF 53; plan/gates 195; fiscal targets
  139; populace-data 138 with 1 skip. The directly affected tail-bound suite
  adds 12 passes. Ruff format/check and `git diff --check` are clean.
- Wrote `SOL_516_REPORT.md` with the exact seam, consumer-by-consumer file:line
  audit, expected 208,611-row real-artifact effect, verification results, count
  sweep, and deliberately untouched surfaces.

## Next

- Orchestrating session: review the dirty worktree and report, then commit and
  open/update the PR as appropriate.
- The next authorized restricted-data rebuild should confirm the expected
  208,611 donor rows in the existing base summary; no artifact build or
  certification was performed in this PR-CI task.
