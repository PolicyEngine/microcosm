# Progress: 25% replacement-candidate runbook, round 2

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

## State

**Stopped at ordered recovery item 1: the Ledger v9.4 consumer artifact cannot
be reproduced byte-for-byte by Chronicle.** The exact provenance commit rejects
the archived feed under its consumer schema; the only historical artifact
writer that can read the mixed legacy feed rewrites the facts from `b3c08356…`
to `f455145f…`. No substitute artifact, manifest path, or manifest pin was
invented. No pool build, release build, publication, promotion, push, SCF
download, or logbook-chain write occurred.

The candidate mode is owner-ruled as exact-k. The owner-stated seed `0` is
recorded pending Max's ratification. Source/history review found no defensible
builder default or July-incumbent value for `pi_hi`; current authority declares
it required with `default: null`. That is a second independent owner-directed
stop.

## Done

- Read `CLAUDE.md`, the prior `FINAL_REPORT.md`,
  `experiments/candidate_25pct/input_audit.md`, and
  `experiments/candidate_25pct/dry_run.md` before taking round-2 action.
- Confirmed the worktree starts clean at `4f616cb2b0f9`, four commits ahead of
  `origin/main` (`d69131a3534a`).
- Read the GitNexus exploration workflow. No GitNexus repository resource or
  query tool is exposed in this session, so code-contract tracing will use the
  checked-out sources and local repository history directly.
- Established the serial stop rule: later inputs may be researched read-only in
  parallel, but they will not be produced until every prior ordered input is
  reproduced and verified.
- Located the host v9.4 provenance receipt: 37,405 rows from Ledger commit
  `0575510d93ec`, pinned to
  `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.
- Ran `build-consumer-artifact` from an archived snapshot of that exact commit.
  It rejected row 1 for a missing required `assertion` before producing an
  artifact.
- Ran the earliest historical artifact writer that can read the legacy rows as
  a corroborating `/private/tmp` check. Its 37,405 output rows measured
  `f455145f07a3047a325effc957f0d5dc8d4e317e96fec594a5625ef30e20cff6`
  and differed starting at byte 23, so the output was rejected and not moved.
- Recorded the complete evidence in
  `experiments/candidate_25pct/input_audit_r2.md` and the non-run disposition in
  `experiments/candidate_25pct/dry_run_r2.md`.
- Stopped later-input recovery, as required, before any SCF download,
  incumbent-diagnostics generation, frozen-surface compilation, host-script
  creation, or dry-run.

## Next

The owner must choose a new reviewed contract; this lane cannot make that
choice without fabricating provenance. Viable directions are either:

1. ratify a new Chronicle-native consumer feed/artifact and update the reviewed
   facts pin away from `b3c08356…`, or
2. add and review a Chronicle compatibility exporter that preserves the legacy
   mixed-contract bytes and emits a producer-owned manifest.

The owner must also ratify an exact-k `pi_hi`. After both issues are resolved,
restart ordered recovery at item 1; do not resume at SCF or create a partial
host command from the currently missing pins.
