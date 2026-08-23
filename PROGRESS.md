# Progress: 25% replacement-candidate runbook, round 4

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

## State

Round 4 is complete within the writable worktree. The full-SCF blocker is
cleared; the guarded dense-first launcher, round-4 audit, real exit-0 dry-run,
runtime handoff, and final report are ready. No pool or release builder ran.

Dense is GO after the owner installs the committed launcher at the required
host path. Sparse is STOP pending a selection-authority ruling. The managed
filesystem denied that one external copy, so the external launcher does not
yet exist and no claim to the contrary is made.

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
- Ran the scorer's live `--help`; its required/optional argument surface
  matches the printed dense command.
- Ran the committed launcher in real `--dry-run` mode at `6327ec02`; all input
  hashes, the SCF Stata header, parser flags, and rendered commands passed. The
  full exit-0 transcript is committed in
  `experiments/candidate_25pct/dry_run_r4.md`.
- Attempted the required host-path installation. The managed sandbox denied
  creation of `_buildo-runtime/out/candidate-25` as outside its writable roots,
  so the receipt uses the byte-identical committed canonical launcher and does
  not claim that the external copy exists.
- Updated `_LANE-NOTES.md` with per-stage wall/RSS evidence, 90/110 GiB launch
  gates, the sparse STOP, the required exact-byte host installation, and the
  exact `launchctl submit` owner action.
- Wrote the complete round-4 outcome and remaining host action to
  `FINAL_REPORT.md`.

## Next

1. Owner: copy the committed `run-candidate.sh` to the required external path,
   verify the worktree is clean, and use the exact `_LANE-NOTES.md` launch line.
2. Owner: rule on legacy non-exact L0 versus a newly ratified exact-57,240
   support authority before any sparse stage is added.
