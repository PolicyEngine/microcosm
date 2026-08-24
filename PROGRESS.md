# Progress: candidate 25% stage 2b, owner ruling A

Date: 2026-08-24

Branch: `candidate-25pct-runbook`

## State

Implementation has started from clean commit `b8176985`, which already merges
the PolicyEngine-US 1.819.0 lock from `origin/main` (`7b90bb18`). The task is to
replace the deliberate sparse STOP with the owner-authorized legacy
fixed-penalty L0 release, extend the real dry-run receipt, and leave all build
and publication paths unexecuted.

No pool, release, scorer, publication, promotion, or push has run in this
round.

## Done

- Read `CLAUDE.md` and the GitNexus exploration skill before task actions.
- Confirmed the worktree was clean on `candidate-25pct-runbook` at
  `b8176985`.
- Confirmed the branch contains the `origin/main` PolicyEngine-US 1.819.0 lock
  merge and identified `origin/main` commit `7b90bb18` as the current-main code
  authority available locally.
- Tried the skill's GitNexus resource discovery. Its graph server/tools are not
  exposed in this session, so contract tracing will use direct source, tests,
  parser help, and committed/host receipts.
- Started independent read-only reviews of the legacy sparse builder contract,
  current launcher conventions, and required validation surface.

## Next

1. Read every required prior report and the complete launcher, then trace the
   legacy sparse/L0 path on current main with exact source citations.
2. Record the complete stage-2b input/contract decision, including the 0.8
   default, 6,000 epochs, zero-waiver handling, and whether Keogh protection
   belongs to legacy cold L0.
3. Implement and commit the guarded, off-chain sparse release stage and keep
   this journal current after each coherent step.
4. Run the real `--dry-run`, commit `dry_run_r5.md`, and pass `bash -n`,
   ShellCheck, and final repository checks.
5. Write and commit the final outcome to `FINAL_REPORT.md`.
