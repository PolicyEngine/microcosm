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

## Next

- Remove multiplier machinery and restore the loss-weight call shapes and
  behavior to `origin/main`.
- Fix Sol findings 1-3 in separate commits: selector parity, shared CD
  classification, and missing recorded-relative-error rejection.
- Add the finding-4 behavioral containment battery and rerun the three
  round-1 reproductions.
- Run the full requested suites and static checks, then update this file and
  `FINAL_REPORT.md` with final receipts.
