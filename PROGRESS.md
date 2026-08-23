# Progress: 25% replacement-candidate runbook, round 4

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

## State

Round 4 is in progress. The owner-supplied full Federal Reserve SCF 2022 Stata
extract clears round 3's first mandatory-input stop, subject to local header,
SHA-256, and loader-path verification. No pool or release builder has run.

The requested launcher must remain serial, off-chain, non-publishing, and
guarded before every stage by process, reclaimable-space, AC-power, and go-marker
checks. Dense and sparse are separate stage-2 invocations. Sparse may run only
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

## Next

1. Verify the new full-SCF bytes, Stata header, and current loader resolution;
   refresh every explicit input/hash and current parser flag.
2. Trace whether current-main tools can derive a pool-bound sparse-57k
   selection source and whether `keogh_distributions` remains authorized.
3. Inspect July pressure logs and the f025 probe evidence for stage thresholds,
   wall times, and peak-RSS annotations.
4. Build and commit the guarded launcher in coherent steps, then run `bash -n`
   and a real `--dry-run` without starting either builder.
5. Commit the round-4 dry-run receipt, lane notes, final progress state, and
   `FINAL_REPORT.md`. Do not push, publish, promote, tune, or touch pending
   logbook-chain state.
