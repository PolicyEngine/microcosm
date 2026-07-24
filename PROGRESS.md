# Progress

## State

Populace #462 split-PR remediation is complete on
`loss-contract-alignment`, based on `origin/main` at `7b6e10b`. The outcome is
register alignment only: the multiplier is removed per populace#492, both
consumers share CD classification, and all four Sol round-1 findings are fixed
with behavioral containment coverage. Nothing was pushed at the time of
this handoff; the branch was subsequently pushed and merged as #491
(2026-07-22), so this remediation is on `main`.

## Done

- Read the Sol round-1 report and used its three HIGH reproductions plus the
  MEDIUM anti-drift gap as the acceptance oracle.
- Removed every critical-row multiplier surface and restored the loss helper,
  scorer calls, and historical experiment calls to their `origin/main` shape.
- Fixed the Table 1.4 prefix narrowing and captured the exact builder failure
  for `other.table_1_4.all.bad_amount@2024`.
- Added the exported six-signal shared CD classifier, converted both consumers
  to thin wrappers, and passed registry metadata through all builder gate
  paths that classify target rows.
- Compiled a real CD Ledger reference and proved equal six-row excluded name
  sets/counts and full-gate exclusion builder-side and publisher-side.
- Made missing/`None` recorded relative error a builder failure with the
  required publish-contract message; stale and non-numeric checks remain.
- Added behavioral containment for exact, semantic, Table-pattern,
  missing/non-finite, incumbent-escape, and every CD evidence class, including
  an explicit guard against any accepted-name-prefix narrowing.
- Preserved the #490 medical 0.25 adjudication block and comment byte-for-byte.
- Reran the requested combined suite: 264 passed and 3 skipped (267 collected).
- Reran the three Sol scenarios: the two malformed rows are builder-rejected
  with exact failure strings, while the CD row is symmetrically excluded by
  both consumers as required.
- Confirmed zero multiplier grep hits, Ruff check/format-check cleanliness on
  all applicable touched files, and a clean `git diff --check`.
- Replaced `FINAL_REPORT.md` with the complete split-PR outcome and receipts.
- Attempted `/Users/maxghenis/PolicyEngine/_reviews/sol-491-fix-out.md`; the
  sandbox rejected the write with `Operation not permitted`, so the full
  report will be printed to stdout per the requested fallback.

## Next

- None — resolved. This handoff (and the branch it describes) was pushed and
  merged as #491; nothing remains pending from it.
