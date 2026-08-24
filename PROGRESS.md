# Progress: candidate 25% stage 2b, owner ruling A

Date: 2026-08-24

Branch: `candidate-25pct-runbook`

## State

Contract tracing and the stage-2b implementation are complete against local
current-main authority `origin/main` at `7b90bb18`. The deliberate sparse STOP
is replaced by a second, serial legacy release invocation that lets the current
builder derive support by cold fixed-penalty L0 at its literal default 0.8.
The release accepts the realized non-exact count and supplies no frozen
selection, exact-count rule, `pi_hi`, Keogh mass protection, operator exclusion
register, or tuning override. Static shell validation passes; the implementation
is ready for its clean-commit dry run.

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
- Corrected the initial input enumeration after tracing the SSI retry contract:
  stage 2b needs exactly one sparse-specific immutable input beyond stage 2a,
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/attempt6_basis_schema3_seed.json`
  at SHA-256
  `25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a`.
  The incumbent launcher supplies this schema-3 seed at
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:133-134`;
  current main accepts schema 2/3/4 bases
  (`packages/microcosm-build/src/microcosm/build/us_runtime/ssi_take_up.py:902-918`)
  and rejects retry-of-retry current-schema release artifacts at lines 943-959.
  The launcher authenticates the seed and validates its output provenance.
- Reused the stage-1 H5 and every common stage-2a source, gave sparse its own
  output, checkpoint, release-ID, log, RSS, and completion state, omitted the
  dense flag, and used the incumbent's 6,000 epochs. The historical incumbent
  launcher is the epoch authority at
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
  (dense). The implementation makes every wait/recheck use the stated 85 GiB
  gate.
- Replaced `stage_sparse_stop` with owner-ruling-A stage 2b. It runs after the
  dense release, off-chain and with `--no-staging`, persists the required
  `populace-us-2024-onesurface-pkg3-legacy-sparse-<sha8>-<ts>` ID, monitors RSS,
  supports authenticated idempotent skips, and emits the artifact identity and
  sparse scorer command.
- Added a lightweight AST/argv preflight which proves the checked-in builder's
  parser wires the L0 flag to the sole 0.8 constant while proving the sparse
  command omits the L0 flag itself and every frozen, exact-k, waiver, warm-start,
  and tuning flag.
- Added post-build sparse authentication for the release method, non-exact
  realized support, 6,000 selection/refit epochs, effective L0 penalty,
  absence of frozen/exact/Keogh paths, code/Ledger/pool pins, schema-3 SSI basis
  receipt, no staging, and green release/QRF gates.
- Extended dry-run output through stage 2b and added both candidate scorer
  commands. `bash -n`, ShellCheck, and `git diff --check` pass on the pending
  implementation.
- Found the pre-existing external `candidate-25/code.commit` pin at
  `8fa966d9` (with no stage-1 artifact). Execute mode still refuses any commit
  mismatch. Dry-run now reports that conflict without consuming or changing
  external state, so the new committed command can receive a read-only receipt;
  owner cleanup or a fresh output-root ruling remains required before execution.

## Next

1. Commit the guarded, off-chain sparse implementation and this journal.
2. From that clean commit, run `--dry-run` and commit the verbatim
   `dry_run_r5.md` receipt.
3. Write and commit the final outcome to `FINAL_REPORT.md`, then verify a clean
   branch and final checks.
