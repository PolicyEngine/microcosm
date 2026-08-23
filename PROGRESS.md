# Progress: 25% replacement-candidate runbook, round 3

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

## State

**Stopped at a missing mandatory full-SCF stage-2 input before launcher
construction.** The target-surface gate passed: legacy dense and exact-k reach
the same unconditional unified compiler and materialization path. The first
replacement remains explicitly
`one-surface + pkg3, legacy release arm, not exact-k certified`, using the bare
v9.4 Ledger facts pin, `--dense-default-dataset`, and seed `0`. The exact-k
artifact/feed re-pin and `pi_hi` decision are deferred to the next candidate.

Current main unconditionally needs the full SCF extract at
`/Users/maxghenis/.cache/microcosm/scf/p22i6.dta`. It is absent, its adjacent
archive is absent, and the loader has no archive/member SHA pin. A second
independent mismatch remains: one dense-default invocation emits one dense H5,
not both the requested dense and sparse artifacts.

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
- Enumerated and measured all present July-explicit and current-main implicit
  stage-2 file inputs in
  `experiments/candidate_25pct/input_audit_r3.md`.
- Confirmed the zero-waiver route is to omit
  `--qrf-tail-concentration-exclusions`; the optional loader returns `{}`.
- Found and hash-checked all six pinned stage-1 sources without running the
  pool builder.
- Stopped on the absent full 2022 SCF extract without download or substitution.
- Recorded the non-run receipt in
  `experiments/candidate_25pct/dry_run_r3.md`; no external launcher, partial
  command, `bash -n` claim, or script dry-run was fabricated.
- Recorded process-tree evidence: same-week f025 attempts took about 2.8 hours
  but peaked at 75.46 and 85.85 GiB, while the July dense release attempt
  reached 96.83 GiB. The proposed 22 GiB estimate is not defensible as peak
  RSS.
- Wrote the stopped round-3 outcome and required owner handoff to
  `FINAL_REPORT.md`.

## Next

1. Owner supplies and pins the exact full-SCF bytes accepted for this arm.
2. Owner clarifies whether the deliverable is the single dense legacy
   candidate requested by the ruling, or separately authorizes a second sparse
   build. Only then construct, syntax-check, and dry-run the guarded launcher.
3. Owner clarifies whether zero waivers applies only to per-run/operator
   registers (implemented here) or also to current-main's checked-in exclusion
   semantics.
4. Do not run either builder, publish, promote, push, or touch pending
   logbook-chain state in this round.
