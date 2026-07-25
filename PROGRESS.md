# Progress

## Current task

The active ledger for the QBI v3 wiring work is
`PROGRESS_QBI_V3_WIRING.md`. Branch `qbi-v3-wiring` combines the completed
`qbi-v2-content` engine/content lane with the completed `qbi-v3-evidence`
resource lane and remains offline.

## Current state

- The prescribed sibling merge is committed in
  `.claude/worktrees/populace-wt-530`.
- The version-3 assumptions builder, persisted calibration, runtime/host
  wiring, diagnostics, and final report are complete. Full-workspace pytest,
  Ruff, repository contracts, reproducibility, and offline wheel checks pass.

## Current done

- Created `qbi-v3-wiring` from `qbi-v2-content`.
- Merged the evidence branch; the only conflict was the expected adjacency in
  this shared journal, resolved by retaining both dedicated sibling ledgers.
- The completed sibling histories remain in `PROGRESS_QBI_V2_CONTENT.md` and
  `PROGRESS_QBI_V3_EVIDENCE.md`.
- Completed the read-only architecture and evidence inventory, including the
  byte-identity boundary for all existing random streams.
- Emitted the reproducible full-artifact assumptions resource, wired the five
  new independent v3 streams, and added unit, host-parity, synthetic, and
  restricted replay coverage.
- Completed the full-workspace run with 3,293 passes, 132 skips, and zero
  failures; completed Ruff, formatting, spec-only, manifest, incumbent,
  entrypoint, reproducibility, and wheel checks.
- Recorded the complete handoff in `FINAL_REPORT_QBI_V3_WIRING.md`.

## Current next

- Supervisor pushes `qbi-v3-wiring` and opens the pull request.

## Historical progress

The remaining sections are prior-lane records retained for branch history.

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

- PR #527 review cycle, then merge. After both #525 and #527: rebuild the
  base/release; the mortgage critical-fit ratchet (0.20 -> 0.15) waits on a
  run that holds per `us_critical_targets.py`.
- Root record-level ETL carve stays open on populace#515.
