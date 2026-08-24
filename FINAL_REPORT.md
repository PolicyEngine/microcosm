# Final report: candidate 25% stage 2b under owner ruling A

Date: 2026-08-24

Branch: `candidate-25pct-runbook`

Required label: **one-surface + pkg3, legacy release arm, not exact-k
certified**.

## Outcome

Stage 2b is implemented in
`experiments/candidate_25pct/run-candidate.sh`. The launcher now runs a second,
serial release after stage 2a and lets the new 25% pool derive its own support
through current main's cold legacy fixed-penalty L0 path at the literal default
share `0.8`. It accepts the realized record count: there is no frozen
selection-source manifest, exact-count assertion, `pi_hi`, or target count.

The sparse release is off-chain, uses `--no-staging`, persists a commit-bound
release ID, has its own checkpoints/log/RSS CSV/completion marker, validates
authenticated outputs before any idempotent skip, and prints its artifact path,
SHA-256, and scorer command at completion.

The final committed launcher passed `bash -n`, ShellCheck, diff hygiene, and a
real exit-0 dry-run. No pool, release, scorer, publication, promotion, push, or
staging operation was run by this task.

## Current-main legacy L0 contract

The local current-main authority is `origin/main` at `7b90bb18`, including the
PolicyEngine-US 1.819.0 lock. The inspected builder, tests, and runtime files on
this branch are byte-identical to that authority.

- `DEFAULT_L0_REFIT_LAMBDA_SHARE = 0.8` lives at
  `tools/build_us_fiscal_refresh_release.py:437`. The parser wires
  `--l0-refit-lambda-share` to that constant and documents division by the
  candidate household count at lines 1132-1140. The sparse command deliberately
  omits the flag so it consumes the literal default.
- Current main's epoch default is 1,500 at lines 438 and 1105-1112. Stage 2b
  therefore explicitly supplies `--epochs 6000`.
- With `--dense-default-dataset`, the builder follows the full-pool
  `dense_no_l0` branch at lines 10610-10635. Stage 2a uses that flag and 3,000
  epochs.
- Without `--dense-default-dataset`, a selection source, or exact-k arguments,
  the builder computes `0.8 / n_candidate_households` at lines 10436-10450 and
  calls `calibrate_l0_refit` at lines 10636-10655. It records
  `result.selection.n_nonzero` and the exported support at lines 10656-10677;
  it does not assert an exact or strict-subset count.

The material command distinction is therefore:

```text
stage 2a dense:  --base-h5 <stage-1 pool> --dense-default-dataset --epochs 3000
stage 2b sparse: --base-h5 <stage-1 pool>                         --epochs 6000
```

Both invocations carry the same common, pinned stage-2 sources and their own
output/checkpoint/release-ID state. The sparse invocation adds no selection
artifact and no L0 tuning flag.

The incumbent epoch authority is
`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:123-143`,
which passes `--epochs 6000` at line 140. That incumbent was a frozen-support,
`--dense-default-dataset` polish run, so it establishes the requested epoch
count but is not evidence for cold-L0 selection or Keogh-path semantics.

## Sparse-specific input and zero-waiver ruling

Beyond stage 2a's immutable inputs, stage 2b needs exactly one additional
pinned input:

```text
path:   /Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/attempt6_basis_schema3_seed.json
size:   4,782 bytes
sha256: 25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a
```

The incumbent sparse launcher supplies that path and pin at
`buildp_sparse9.sh:133-134`. Current main requires the path/hash pair and
authenticates it before use
(`tools/build_us_fiscal_refresh_release.py:6041-6109`). The SSI runtime accepts
schema-2/3 legacy capacity/floor seeds at
`packages/microcosm-build/src/microcosm/build/us_runtime/ssi_take_up.py:902-918`.
Its current-schema retry-of-retry guard at lines 943-959 is why the launcher
does not substitute the incumbent's final schema-4 output. Post-build
authentication requires the sparse release's `us_ssi_take_up.json` to record
this exact schema-3 source hash and to be a schema-4 `release_final` artifact.
Sparse keeps hard SSI delivery enforcement; only the dense diagnostic arm gets
fences (`tools/build_us_fiscal_refresh_release.py:10758-10771`).

The standing zero-waiver rule is implemented by omission:

- Current main rejects the retired `--zero-support-exclusions` option, including
  an attempted empty file
  (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:52-93`).
  The launcher does not pass it.
- `--qrf-tail-concentration-exclusions` is optional. Omission returns `{}` at
  `tools/build_us_fiscal_refresh_release.py:7922-7935` and records a null file,
  null hash, and empty reviewed-exclusions map at lines 11258-11290. The
  launcher omits it and validates those receipts.
- The sparse command preflight also forbids all other gate bypasses, evidence
  switches, loss/ratio/L2 tuning, warm start, target-family multipliers,
  selection/exact-k arguments, and target-aging/tolerance overrides.

Mutable sparse checkpoints, the persisted release ID, logs, RSS CSV, output
root, artifact, and completion marker are run state, not additional immutable
inputs.

## Keogh-carrier decision

`--selection-mass-protection keogh_distributions` is omitted.

Current main's parser help says the option injects a synthetic mass target so a
refit cannot crush carriers that a protect-swap placed in a **frozen
selection** (`tools/build_us_fiscal_refresh_release.py:927-938`). The
implementation doctrine repeats that protect-swap/frozen-selection contract at
lines 2001-2016. The incumbent's use of the flag occurred alongside a frozen
selection source and dense polish; it does not make the flag part of cold
legacy L0.

Although the parser can mechanically accept the option without a selection
manifest, doing so here would add a synthetic target beyond owner ruling A and
would be new tuning. The launcher omits it, statically rejects its appearance,
and post-build validation rejects any
`selection_mass_protection.*` diagnostic target.

## Implemented stage 2b

The sparse stage writes beneath
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/release-sparse/`
and uses release IDs of the required form:

```text
populace-us-2024-onesurface-pkg3-legacy-sparse-<sha8>-<UTC timestamp>
```

Its invocation carries the authenticated stage-1 `pool.h5`, all common stage-2
sources, the schema-3 SSI basis and pin, seed 0, 6,000 epochs, sparse checkpoint
and output roots, `--skip-reform-validation`, and `--no-staging`. It omits the
dense flag, selection H5/manifest, selection mass protection, exact-k, `pi_hi`,
exact-k pool manifest, QRF exclusion register, warm start, and tuning flags.

The implementation also provides:

- serial stage order: pool, dense release, then sparse release;
- the same 85-GiB reclaimable-memory readiness gate for all stages, plus no
  running pool/release builder, AC power, and the owner `.max-go` marker;
- reauthentication after every wait and immediately before launch;
- `/usr/bin/time -l` logs and 30-second process-tree RSS CSV sampling;
- unconditional off-chain execution by unsetting
  `POPULACE_LOGBOOK_PREV_ROW_DIGEST` globally and with `env -u` at each command;
- persisted commit-bound release IDs and authenticated marker reconstruction;
- output validation for the artifact hash, build ID, `l0_refit` identity,
  default share/effective penalty, 6,000 selection/refit epochs, positive
  realized count no greater than the candidate pool, exported-count equality,
  absence of selection/warm-start/exact-k/Keogh paths, code/Ledger/pool pins,
  SSI basis provenance, disabled staging, and green calibration/QRF gates;
- final dense and sparse artifact identities followed by exact dense and sparse
  head-to-head scorer commands. The launcher prints but never runs the scorer.

The owner ruling text is recorded in the script header, runtime journal line,
and `PROGRESS.md`. The earlier actual wait mismatch (90 GiB for pool and 110 GiB
for dense despite an 85-GiB plan/recheck) was corrected so every initial wait
and final recheck now uses 85 GiB.

## Dry-run and verification

The final implementation bytes were tested from clean commit
`52a2bcfbd98d55444ec55abaffce41cbd773a184`:

```text
./experiments/candidate_25pct/run-candidate.sh --dry-run
exit 0
```

Launcher SHA-256:

```text
484874a22d63e8a0faf16f3eb504eed152a0a2d5993d9b6db5d5e18aeee69838
```

The dry-run authenticated every immutable input, verified current parser
surfaces and the SCF header, emitted the owner-ruling contract proof, printed
the pool/dense/sparse commands with 85-GiB plans, and printed both artifact and
scorer lines. Full stdout and side-effect checks are committed verbatim in
`experiments/candidate_25pct/dry_run_r5.md`.

Validation status:

```text
bash -n experiments/candidate_25pct/run-candidate.sh       PASS
shellcheck experiments/candidate_25pct/run-candidate.sh    PASS
owner-ruling AST/argv contract preflight                   PASS
./experiments/candidate_25pct/run-candidate.sh --dry-run   PASS (exit 0)
git diff --check                                           PASS
```

No pool/release build, scorer, publication, promotion, staging, push, or
pending Logbook-chain action was performed. The candidate pool directory
remained empty and `release-sparse/` remained absent. A pre-existing external
smoke log changed independently during the review window; this launcher never
reads or writes the `smoke/` subtree.

## Execute-mode prerequisite

The pre-existing external candidate root is pinned to commit
`8fa966d9398efc3a445845051501082295a244c9`, and its dense release-ID file is
also pre-existing. It contains no stage-1 pool artifact. The new dry-run reports
the mismatch without mutation, while execute mode correctly fails closed before
any stage action.

No external state was deleted, overwritten, or repinned. Before an actual
launch, the owner must explicitly authorize reconciliation of that stale root
or choose a fresh output-root policy. This prerequisite does not affect the
committed runbook or its read-only dry-run receipt.

## Commits

- `78197b9a` — start the owner-ruling-A progress journal.
- `6f9c4ad7` — record the current-main sparse contract.
- `24d9ff18` — implement the guarded owner-ruling-A sparse stage.
- `52a2bcfb` — align validation with the non-exact realized-count contract.
- `175a607a` — commit the exit-0 round-5 dry-run receipt.

This final report and the completed progress journal are committed as the last
coherent step. No push was made.
