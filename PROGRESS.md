# Round 5 progress

## State

Round 5 is in progress on `smoke-gapfill-ordering-fix`, starting from exact
HEAD `b4ecd763b21507f659d1122bfbca6fdb7d238fee`. C1 and C4 are implemented and
focused tests pass. Work is local and offline: no push, GitHub, network, 1%
smoke, state shelving, or root build journal is in scope.

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

## Next

1. Bump and bind both changed authority surfaces to v6 with stale-version
   rejection coverage.
2. Authenticate QBI transition receipts at every generation, checkpoint,
   resume, manifest, and publication boundary.
3. Complete the changelog and the remaining authority/audit documentation.
4. Run focused tests, the exact 495-test #583 guard, all twelve probes, the
   chunked foreground full workspace battery, and changed-file quality gates.
5. Update this file and `FINAL_REPORT.md`, commit the final receipts, and print
   the final report to stdout.
