# Round 5 progress

## State

Round 5 is in progress on `smoke-gapfill-ordering-fix`, starting from exact
HEAD `b4ecd763b21507f659d1122bfbca6fdb7d238fee`. C1 through C4 and A1 are
implemented; the remaining work is the complete proof battery and final
report. Work is local and offline: no push, GitHub,
network, 1% smoke, state shelving, or root build journal is in scope.

The Round 4 verifier report has been read in full. Its five required changes
are C1 cross-arm earnings-universe semantics, C2 authority/schema v6 bumps,
C3 live QBI reconciliation receipt authentication, C4 named and correctly
ordered rejection contracts, and A1 precise changelog coverage.

## Done

- Confirmed the requested branch and exact starting HEAD.
- Confirmed the starting worktree was clean.
- Read `AGENTS.md`, `CLAUDE.md`, `README.md`, and `DESIGN.md`.
- Read the PolicyEngine data and development-standards skill instructions.
- Read all 164 lines of the Round 4 verification report.
- Recorded the verifier's exact twelve probes and full-workspace proof
  requirements.
- Inspected local base refs without fetching, as network use is prohibited.
  The branch is 15 commits ahead of and 3 commits behind the locally available
  `origin/main`; the user explicitly required continuing in this worktree.
- Traced the certified ASEC producer and inspected Build J's local release
  artifact. All 15,509 under-15 ASEC people have numeric-zero WSAL_VAL,
  SEMP_VAL, mapped employment, and mapped self-employment; all seven all-child
  ASEC tax units are retained with zero earnings sums. The PUF-detail ASEC
  clone likewise has 21,737 under-15 zeros and 22 retained all-child units.
- Matched that established produced-frame contract in ACS with an explicit,
  named universe-zero application. Raw WAGP/SEMP blanks remain unchanged;
  only mapped under-15 leaves become zero, with exact rule/count/digest
  receipts. All-child units remain QRF recipients with receipted zero
  predictors.
- Removed the aggregation-time structural-null mask/coercion path. Eligible
  mapped or raw nulls now fail at the original “missing values before
  coercion” preflight, with the responsible WAGP/SEMP rule ID in the error.
- Restricted all employment/self-employment person allocation, including the
  SSTB self-employment derivative, to age-15-plus rows. Under-15 first-person
  fallbacks and all-child units stay numeric zero even for positive unit
  predictions.
- Applied the same all-clone ACS universe to QBI base-source preservation and
  rewrote the ordering document's former empty-sum/null blessing.
- Added regressions for a mixed unit whose first person is under 15, an
  all-child unit, eligible nulls at preflight, and 14+/16+ rule mutations.
  The 11 focused earnings/universe/stacked-pass tests pass.
- Closed the independent-review provenance gap: the universe application now
  rejects any pre-existing mapped child value, including zero, so a generic
  earlier fill cannot acquire a false named receipt. Added a raw-blank
  end-to-end stacked-pass regression that verifies exact application counts,
  raw preservation, mapped zeros, one mixed unit, and one all-child unit.
- Bumped the primary-QRF checkpoint schema, stacked checkpoint materializer,
  and stacked canonical authority to version 6. The outer base identity now
  binds primary-QRF schema 6 plus canonical ACS earnings-universe and QBI
  reconciliation contract identities.
- Added parameterized rejection coverage for every stale v1--v5 primary-QRF
  root/target, outer materializer, and self-consistent stacked authority,
  including the former strict two-control v5 payload. New C2 tests pass 16/16;
  the full primary-QRF suite passes.
- Added the canonical QBI receipt-v2 schema, deterministic transition
  recomputation, exact key/type/count/SHA validation, nested universe
  authentication, live-output digest binding, and fixed-point validation.
  QBI unit tests cover forged SHA/count/extra-field receipts and both
  regenerated and post-receipt output mutations.
- Enforced exact QBI receipt routing and live-frame authentication at
  generation, simulated checkpoint emission, durable checkpoint write/load,
  simulated resume, legacy/stacked manifest construction, and legacy/stacked
  publication entry. Missing, wrong-route, ambiguous, forged, and rebound
  receipts fail closed.
- Added real frame-bound QBI receipts to checkpoint/publication fixtures and
  boundary-specific regressions that corrupt receipts only after earlier
  validation. The QBI, multispine runtime, and pool-tool focused suites pass
  together.
- Completed the ordering authority and changelog coverage: certified ASEC
  evidence and artifact counts, both named ACS rules, exact preflight and
  nonfinite rejection semantics, v6 identities, full QBI authentication
  lifecycle, and explicit fit-weight/checkpoint/manifest/publication audit
  rows are now documented.
- Independent review found and reproduced the remaining QBI exactness gaps
  before the workspace battery: subset-only live universe comparison, no live
  binding for QBI drivers, Boolean count acceptance, a `None`-valued ambiguous
  route, arbitrary added-column laundering, forged preservation counts, and a
  self-consistent alternate fixed point with a freshly generated receipt. The
  receipt now requires exact live universe equality, exact driver binding,
  exact preservation equations, and a canonical seed-only person inventory;
  Boolean counts and wrong-route key presence fail. Generation separately
  anchors the validated transition in immutable frame metadata and a typed
  receipt SHA carried outside stage receipts through results and checkpoints.
  Durable checkpoint H5 metadata and sidecars content-bind that authority, and
  every resume/manifest/publication boundary requires it. Coherently rehashed
  universes, driver/output fixed points, laundered columns, forged counts, and
  freshly reissued alternate-fixed-point receipts now fail closed.
- The QBI (31), multispine runtime (49), and pool-tool (135) suites pass after
  the independent transition-authority propagation; focused regressions also
  prove authority mismatch rejection at resume, durable write, manifest
  construction, and publication.
- All twelve verifier probes pass as 48 maintained cases: 8 source/universe
  cases, 8 QRF/QBI cases, 15 stale-version cases, and 17 canonical-boundary
  and starvation-ordering cases.
- The #583 source-spine-blindness guard passes exactly 495 tests. Its first run
  correctly caught an unrecognized metadata-rebuild expression and a direct
  support-channel driver name; the authority binder now uses the guard-approved
  metadata expansion, and the driver digest binds the derived support role
  without reading source-channel columns. The full 215-test QBI/runtime/tool
  focus remains green after that correction.
- The foreground workspace battery has completed the frame/fit/calibration/data,
  generic-build, UK-build, and first US-build chunks: 831, 662, 777 collected
  tests, and 988 outcomes respectively. The next US chunk exposed seven legacy
  support-test call sites whose produced-frame fixtures predated the required
  `person.age` allocation input. The imputation fixtures now declare adult
  ages while raw-source fixtures remain operator-free; all affected/adjacent
  cases pass without weakening the production fail-closed rule.
- The corrected second US-build shard passes in one uninterrupted foreground
  run: 1,270 passed and 2 skipped (1,272 collected outcomes). It includes both
  the produced-frame age-input paths and the raw-source operator boundary.
- The first final-US-shard run found two synthetic stacked sources that defined
  only the WAGP/employment half of the now-atomic twin earnings surface. Their
  ACS fixtures now include SEMP and mapped self-employment zeros; the end-to-end
  fixture also makes unemployment incidence deterministic from the required
  `is_female` predictor. Both formerly failing cases pass together.
- The corrected final US shard passes 1,092/1,092. Across all six foreground
  workspace chunks, pytest reports 5,565 passed and 58 skipped outcomes. One UK
  skip is a collection-level module skip, so the battery covers 5,622 collected
  test functions plus that one module-level skip.
- Repository-wide Ruff lint and the full commit-range whitespace check pass.
  Ruff reformatted one newly changed test expression; both affected stacked
  tests still pass, and that file now passes its format check.
- Final quality replay passes repository-wide Ruff lint, Ruff format-check on
  all 14 changed Python files, and `git diff --check b4ecd763..HEAD`.
- The final committed-code replay of all twelve verifier probes passes as 48
  maintained cases: probes 1--6 = 8, probes 7--9 = 8, probe 10 = 15 stale v1--v5
  cases, and probes 11--12 = 17 canonical-seam/starvation cases.

## Next

1. Complete the final line-level audit and independent review.
2. Update this file and `FINAL_REPORT.md`, commit the final receipts, and print
   the final report to stdout.
