# Progress

## State

Populace #462 split-PR remediation is in progress on
`loss-contract-alignment`. The target outcome is register alignment only: the
critical-row loss multiplier is removed per populace#492, the builder and
publisher share one congressional-district classifier, and all four Sol
round-1 findings are closed with behavioral containment tests.

## Done

- Confirmed the worktree is clean at `068854d` and tracks the pushed branch;
  this remediation will not be pushed.
- Read `/Users/maxghenis/PolicyEngine/_reviews/sol-491-out.md` and accepted its
  three HIGH reproductions plus the MEDIUM anti-drift gap as the oracle.
- Confirmed the existing shared register still contains the #490 medical 0.25
  adjudication block; that tolerance and comment will remain untouched.
- Located every current Python reference to the critical-target loss
  multiplier across tools, tests, and historical experiment entry points.
- Removed the critical-row loss multiplier constant, CLI/validation, final
  overlay, telemetry, diagnostics, scorer provenance, experiment pins, and
  multiplier-specific tests per populace#492.
- Restored `_fiscal_target_loss_weights` source-identically to `origin/main`;
  both scorer files and both historical experiment files also match main.
- Confirmed the required multiplier grep has zero Python hits and the focused
  loss/diagnostics test selection passes (8 passed).
- Fixed Sol finding 1 by removing the builder adapter's `irs_soi.` prefix
  narrowing; the supplied `other.table_1_4.all.bad_amount@2024` reproduction
  now fails the builder gate at relative error 1 versus the 0.25 cap.
- Added and passed focused selector-parity/reproduction coverage (2 passed).
- Fixed Sol finding 2 with one dependency-light exported CD classifier that
  unions all six owner-approved evidence signals; publisher and builder are
  thin adapters over it.
- Gave the builder's exact/semantic, Table 1.4, and zero-support paths the same
  registry-backed metadata surface, closing both directions of CD drift.
- Compiled a real CD Ledger reference, isolated each evidence key into its own
  registry row, and proved equal six-row excluded name sets/counts plus full
  gate exclusion on both consumers (3 focused tests passed).
- Fixed Sol finding 3 by making every matched fit row require a recorded
  numeric `relative_error`; missing/`None` now emits the owner-specified
  publish-contract failure instead of silently relying on recomputation.
- Added the supplied adversarial Table 1.4 regression and updated the generic
  gate contract test; the full gate test file and focused builder tests pass.

## Next

- Add the finding-4 behavioral containment battery and rerun the three
  round-1 reproductions.
- Run the full requested suites and static checks, then update this file and
  `FINAL_REPORT.md` with final receipts.
