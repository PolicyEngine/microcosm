# Progress: 25% replacement-candidate runbook, round 3

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

## State

**In progress: enumerating every owner-ruled legacy release-arm input before
authorizing the guarded runbook.** The target-surface gate passed: legacy dense
and exact-k reach the same unconditional unified compiler and materialization
path. The first replacement is explicitly
`one-surface + pkg3, legacy release arm, not exact-k certified`, using the bare
v9.4 Ledger facts pin, `--dense-default-dataset`, and seed `0`. The exact-k
artifact/feed re-pin and `pi_hi` decision are deferred to the next candidate.

No pool or release build, publication, promotion, push, tuning, or logbook-chain
write is authorized in this round. Any differing target surface or missing
legacy-arm input is a hard stop.

## Done

- Read `CLAUDE.md`, `FINAL_REPORT.md`,
  `experiments/candidate_25pct/input_audit.md`,
  `experiments/candidate_25pct/input_audit_r2.md`, and
  `experiments/candidate_25pct/dry_run_r2.md` before round-3 investigation.
- Confirmed the worktree starts clean at `08ceb451` on
  `candidate-25pct-runbook`, seven commits ahead of the local `origin/main`
  reference.
- Read the GitNexus exploration workflow for the required compiler-path trace.
- GitNexus graph resources were not exposed in this session, so completed the
  trace directly from current-main source and PR #741 history.
- Recorded the owner ruling that supersedes round 2's exact-k stop for this
  first replacement only.
- Proved that post-#741 there is one unconditional
  `compile_us_fiscal_target_registry` call and no per-run target-membership
  flags; both exact-k and dense consume the same materialized registry.
- Recorded the qualification that dense mode also changes the post-calibration
  SSI delivery fence, not target membership.
- Began the code-cited evidence in
  `experiments/candidate_25pct/input_audit_r3.md`.

## Next

1. Enumerate and hash every legacy-arm stage-2 input from the July 28 incumbent
   invocation through parser and loader code; stop without substitution if any
   input is absent.
2. If the input gate passes, write the guarded off-chain two-stage host script,
   validate it with `bash -n`, execute its real `--dry-run`, and commit the
   receipt plus updated audit/journals.
3. Write the final outcome to `FINAL_REPORT.md`. Do not run either builder and
   do not publish, promote, push, or touch pending logbook-chain state.
