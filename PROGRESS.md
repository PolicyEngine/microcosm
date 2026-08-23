# QBI ownership lane progress

## State

The ownership evidence and the supported two-check terminal-role correction
are implemented. The SHA-verified failed-attempt checkpoints, thirteen bank
targets, publication manifest/gates, late-transfer receipt, exact comparator
metrics, 52 target-pattern regime cells, and nine coupled invariants validate
as one closed evidence artifact. The canonical extractor is fail-before-write,
byte-reproducible, and rejects debug SHA bypass for its canonical output.

The decomposition distinguishes terminal value provenance from the first
failing criterion. All eight terminal clone-0 values have `qrf_transfer`
provenance. All four incidence checks and UBIA QED first fail in the late
transfer; BDC, REIT/PTP, and W2 QED first fail at the clone-1 PUF producer and
worsen in transfer. The regression now fails if any canonical amount receipt
copy omits its origin channel. The coupled refit plan records the structural
1% diagnostics and whole-pool acceptance harness needed before a safe amount
surface change; no unsupported amount-model change is included here.

The battery now compares exactly
`sstb_self_employment_income_would_be_qualified` and `business_is_sstb` on the
PUF-detail clone-1 role. Every other physical target remains on clone 0. The
role declaration is immutable and fail-closed, the live/spec contracts preserve
the registered role, and duplicate registration of one physical target across
roles is rejected.

No gate, band, ceiling, fold, seed, exclusion, build artifact, or logbook chain
has been changed. This lane has started no pool build; the headless order assigns
all builds to the host queue.

## Done

- Inspected salvage commits `03e23e42`, `21f95b71`, `8942ef97`, and newest
  `321b3185`. Recovered the regime/origin receipt implementation and useful
  tests; superseded stale wrappers, journal placeholders, unsafe SHA-skipped
  evidence, and the conflation of terminal provenance with first-failing-stage
  ownership.
- Ran the mandated `uv sync --all-packages --extra us` successfully with a
  sandbox-writable uv cache.
- Audited the adjudication's workstream-5, remediation-order, reviewed-
  exclusion, and failed-attempt identity citations before forming an ownership
  view.
- Repaired two incomplete recovered fixtures exposed by the full diagnostic
  gate: the frozen legacy checkpoint materializer binding and the mandatory
  `unmodeled_rows` count in canonical stacked HDF5 receipts.
- Reran the affected tests successfully: exact-k ladder 3/3, stacked HDF5
  loader 38/38, and deterministic trade-entry build 1/1.
- Completed the prerequisite gate: frame 294 passed/36 skipped; fit 93 passed;
  calibrate 201 passed; data 275 passed/1 skipped; build 5,977 passed/39
  skipped; Ruff and `git diff --check` clean. Native/thread pools were bounded
  to one for the final build-shard run after an unbounded run starved four
  unchanged subprocess tests; those four also pass unchanged in isolation.
- Authenticated the failed 25% publication manifest/gates, all three stage
  checkpoint hashes and identities, thirteen target-bank checkpoint files,
  and the post-PUF transfer receipt against their frozen release/run bindings.
- Recomputed every QBI batch availability-pattern regime from the frozen
  108,073-row donor support. All 52 target-pattern cells agree with the fitter;
  each red amount is `zero_inflated_positive` for every pattern.
- Reran the nine coupled invariants on transferred and terminal checkpoints.
  All nine terminal counts are zero; the pre-reconciliation transfer has 2,996
  BDC and 22,350 REIT/PTP exposure mismatches plus the seven expected coupled
  identity deltas recorded in `evidence.json`.
- Added a four-target regression proving that group, aggregate, and signed
  execution receipts share the exact origin digest and that removing
  `origin.channel` from any copy is rejected.
- Reproduced canonical `evidence.json` twice at SHA-256
  `38e60c1ec5e39b86df957148c877b3062ca97028f33ea0d1411013c2911c4b55`;
  the extractor reports zero adjudication mismatches and validation `passed`.
- Completed the evidence gate: frame 294 passed/36 skipped; fit 93 passed;
  calibrate 201 passed; data 275 passed/1 skipped; build 5,981 passed/39
  skipped; Ruff and `git diff --check` clean.
- Implemented the adjudicated terminal-role fix as an exact-two authority:
  only the two SSTB boolean checks use PUF-detail clone 1. Added fail-closed
  role scope, role-aware surface/plan/metric/transfer/comparator construction,
  live/spec contract validation, authority repins, and direct regressions.
- The focused terminal-role suite is green: 424 passed. The four non-build
  package shards are also green: frame 294 passed/36 skipped; fit 93 passed;
  calibrate 201 passed; data 275 passed/1 skipped.
- The exhaustive build-package gate is green over all 255 sorted test files:
  5,997 passed, 39 skipped. The monolithic command twice exceeded the
  execution service's one-hour foreground ceiling without a reported failure,
  so coverage was completed as ten deterministic, disjoint file slices with
  unchanged tests and thresholds. Ruff and `git diff --check` are clean.

## Next

1. Commit the fully gated exact-two terminal-role correction.
2. Semantically integrate current `origin/main`, preserving both its selected
   post-transfer-calibration evidence and this lane's all-target origin/regime
   contract under a fresh authority version; repin and rerun the full gate.
3. Recheck every documentation `module:line` citation after integration,
   replace `FINAL_REPORT.md`, and leave both journals current.
4. Hand the amount-surface diagnostic/refit to the host queue: instrument and
   compare the frozen 1% baseline first; run the 25% certification only after
   the 1% structural change clears all eight checks and nine invariants.

The former root entry described the merged mortgage-donor lane and is retained
in Git history through commit `2c7a7218`; it is not current state for this lane.
