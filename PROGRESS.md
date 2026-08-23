# Progress: 25% replacement-candidate runbook, round 4

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

## State

Round 4 is in progress. The owner-supplied full Federal Reserve SCF 2022 Stata
extract has cleared round 3's first mandatory-input stop: its release-118
header, exact size, owner SHA-256, and current loader resolution all verify. No
pool or release builder has run.

The requested launcher must remain serial, off-chain, non-publishing, and
guarded before every stage by process, reclaimable-memory, AC-power, and
go-marker checks. Dense and sparse are separate stage-2 invocations. Sparse may run only
if current-main tooling can derive a pool-bound sparse-57k selection authority
without inventing a path, pin, or owner ruling.

## Done

- Read `CLAUDE.md`, `FINAL_REPORT.md`,
  `experiments/candidate_25pct/input_audit_r3.md`, and
  `experiments/candidate_25pct/dry_run_r3.md` in full before round-4 work.
- Confirmed the worktree was clean at `27798ddb` on
  `candidate-25pct-runbook`, 11 commits ahead of `origin/main`.
- Read the GitNexus exploration skill. Its graph tools and MCP server are not
  exposed in this session, so source tracing will use direct current-tree
  inspection, parser help, tests, and host evidence.
- Established the round-3 constraints that continue to apply: identical
  unified target surface, 75--86 GiB measured f025 pool peaks, and separate
  dense/sparse release invocations.
- Verified and recorded every round-4 input/parser decision in
  `experiments/candidate_25pct/input_audit_r4.md`, including the full-SCF hash,
  incumbent SSI basis, zero-waiver omission, and legacy pool-manifest wrapper
  pin.
- Determined that current main cannot select an exact new 57,240-record support:
  its manifest tool only serializes an already-selected H5, while legacy L0 is
  fixed-penalty rather than exact-count. Stage 2b will stop dense-only with the
  exact owner question instead of fabricating a selection.
- Selected evidence-backed reclaimable-memory gates: 90 GiB for pool and 110
  GiB for dense. The latter clears the July dense pressure peak of 96.83 GiB.
- Added the committed-source launcher at
  `experiments/candidate_25pct/run-candidate.sh`: it pins the commit and pool
  manifest across retries, authenticates completed outputs before skips,
  resumes a gate-failed pool from checkpoints, samples process-tree RSS, keeps
  append-only logs, remains off-chain, and stops sparse at the owner ruling.
- Passed Bash 3.2 syntax, ShellCheck, and whitespace checks for the launcher.

## Next

1. Install the byte-identical launcher at the requested host output path and
   run a real `--dry-run` without starting either builder.
2. Commit the round-4 dry-run receipt, lane notes, final progress state, and
   `FINAL_REPORT.md`. Do not push, publish, promote, tune, or touch pending
   logbook-chain state.
