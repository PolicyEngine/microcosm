# QBI ownership lane progress

## State

The `battery-qbi-ownership` lane is in evidence collection at `2c7a7218`.
No production source, gate, band, ceiling, fold, seed, exclusion, build artifact,
or logbook chain has been changed. The source adjudication has been mapped to
the current tree, and the implementation surfaces for QBI terminal-role scope,
ACS-transfer realized-regime receipts, and margin-origin attribution are under
review.

The mandated `uv sync --all-packages --extra us` was attempted first. The
default cache is outside the managed filesystem, and a retry with a writable
cache reached dependency download but the sandbox has no network. The exact
current `uv.lock` is present in the already-synced `microcosm-f1` worktree, so
verification will use that read-only dependency environment while forcing all
Microcosm imports to this worktree's source.

## Done

- Read `CLAUDE.md` and the binding lane orders.
- Read the adjudication's comparator mechanism, reviewed-exclusion QBI finding,
  workstream 5, all ten QBI check rows, and reproducibility checkpoint record.
- Confirmed the eight amount checks cover four positive QBI legs and remain
  BLOCKERs until producer/transfer/reconciliation-stage attribution is recorded.
- Confirmed the two dead SSTB boolean checks are a separate role-ownership fix:
  clone 0 is intentionally false and the SHA-bound failed checkpoint shows the
  live clone-1 signal in band with all nine coupled invariants at zero.
- Traced the ACS-transfer gap: realized QRF regimes are returned by the fitter
  but omitted from pattern provenance, bank metadata, and public receipts.
- Traced the terminal evaluator: it already evaluates per-target clone roles;
  the canonical registry and compatibility projections hard-code role 0.
- Verified that GitNexus is not configured in this session and switched to
  direct code/call-site/test tracing as the documented fallback.
- Created `_LANE-NOTES.md` as the durable evidence and command journal.
- Gated this journal-only step with all five package test shards and Ruff:
  build 5,959 passed/39 skipped; frame 294 passed/36 skipped; fit 93 passed;
  calibrate 201 passed; data 275 passed/1 skipped; Ruff clean.

## Next

1. Implement and commit fail-closed realized-regime persistence for ACS-transfer
   target-bank patterns and receipts.
2. Implement and commit the two-target clone-1 terminal-role authority fix with
   role-aware compatibility projections and regressions.
3. Commit the cited SHA-bound checkpoint evidence and reproducible per-stage
   ownership evidence for all eight amount checks, including the reconciliation
   delta for BDC and REIT/PTP.
4. Implement the smallest safe ownership-attribution receipt/regression and the
   1% whole-pool reconciliation/invariant rerun that host/build constraints allow;
   document the exact 25% follow-up plan.
5. Run the full per-shard suite and Ruff for every coherent commit, keep this
   file and `_LANE-NOTES.md` current, and write the final report to
   `FINAL_REPORT.md`.

The former root entry described the merged mortgage-donor lane and is retained
in Git history through commit `2c7a7218`; it is not current state for this lane.
