# Progress: candidate 25% stage 2b, owner ruling A

Date: 2026-08-24

Branch: `candidate-25pct-runbook`

## State

Contract tracing is complete against local current-main authority
`origin/main` at `7b90bb18`. Implementation will replace the deliberate sparse
STOP with a second, serial legacy release invocation that lets the current
builder derive support by cold fixed-penalty L0 at its literal default 0.8.
The release will accept the realized non-exact count and will not supply a
frozen selection, an exact-count rule, `pi_hi`, Keogh mass protection, or an
operator exclusion register.

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
- Read the required `FINAL_REPORT.md`,
  `experiments/candidate_25pct/input_audit_r3.md`,
  `experiments/candidate_25pct/dry_run_r4.md`, and the complete 892-line
  launcher.
- Verified the release builder, its parser tests, and the informed-selection
  design are byte-identical to `origin/main` at `7b90bb18`.
- Traced the cold legacy branch: `DEFAULT_L0_REFIT_LAMBDA_SHARE = 0.8`
  (`tools/build_us_fiscal_refresh_release.py:437`), the parser installs that
  default and describes the fixed penalty
  (`tools/build_us_fiscal_refresh_release.py:1132-1140`), the effective penalty
  is `share / candidate_households`
  (`tools/build_us_fiscal_refresh_release.py:10436-10450`), and omitting
  `--dense-default-dataset` reaches `calibrate_l0_refit`, which records the
  realized `n_nonzero` support rather than enforcing a count
  (`tools/build_us_fiscal_refresh_release.py:10636-10677`).
- Enumerated the sparse delta from stage 2a: no additional immutable input;
  reuse the stage-1 H5 and every stage-2a source, give sparse its own output,
  checkpoint, release-ID, log, RSS, and completion state, omit the dense flag,
  and use the incumbent's 6,000 epochs. The historical incumbent launcher is
  the epoch authority at
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:123-143`.
- Adjudicated Keogh-carrier protection as out of scope for ruling A. Current
  parser help says the flag preserves carriers that a protect-swap placed in a
  *frozen selection* (`tools/build_us_fiscal_refresh_release.py:927-938`), and
  the implementation doctrine repeats that provenance at lines 2001-2016.
  The incumbent's 6,000-epoch run was frozen-support dense polish, so its use
  of the flag is not evidence that cold legacy L0 owns it.
- Chose omission for zero operator waivers. Current main rejects the retired
  `--zero-support-exclusions` flag
  (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:52-93`),
  while omitting the optional QRF-tail register loads `{}` and records a null
  path/hash (`tools/build_us_fiscal_refresh_release.py:7922-7949,11258-11290`).
- Found a pre-existing readiness mismatch: the launcher prints and finally
  rechecks 85 GiB but still initially waits for 90 GiB (pool) and 110 GiB
  (dense). Stage 2b implementation will make every wait/recheck use the stated
  85 GiB gate.

## Next

1. Implement and commit the guarded, off-chain sparse release stage and keep
   this journal current after each coherent step.
2. Run the real `--dry-run`, commit `dry_run_r5.md`, and pass `bash -n`,
   ShellCheck, and final repository checks.
3. Write and commit the final outcome to `FINAL_REPORT.md`.
