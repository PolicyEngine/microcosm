# Attempt 9 register maintenance

## State

Complete. Both runtime registers are written and validated, and the final
report is in `FINAL_REPORT.md`.

## Done

- Created `attempt9-registers` from local `origin/main` at
  `35371553b6c22bc0499d3b23110088e7cf582379`.
- Confirmed the requested worktree was clean before switching branches.
- Bound the Build-P target-frame checkpoint to base-P SHA
  `ad13e5ead6366060f704d4a53dc1f4974d610dd4927f54a89c69c72fb55e63cb`
  and independently verified its 57,240-identity frozen-selection digest
  `1812b819b0a2e6b7f7f72b68125881dd2a171301c179befef6ff3df47589657e`.
- Measured all four KY/LA under-$1 taxable-interest cells at exactly zero:
  zero nonzero records, zero absolute signal, and zero design-weighted
  aggregate. Attempt-8c diagnostics independently report initial and final
  estimates of `0.0` for all four.
- Reconstructed attempt 8c's 3,000-epoch final calibration weights from the
  checkpoint. The replay exactly matches final loss `0.01852712099663438`,
  all 5,675 final estimates (maximum absolute delta `0.0`), and every target
  surface hash.
- Re-ran the gate's exact top-100 weighted-absolute-mass calculation. Eleven
  columns are strictly above 75%. The old register's three unused entries
  reconcile with the log: `farm_rent_income` is thin at 370 carriers,
  `qualified_reit_and_ptp_income` is dense, and
  `w2_wages_from_qualified_business` is thin at 466 carriers.
- Confirmed the 8c diagnostic cross-checks: mortgage Table 2.1 is
  `$224,295,761,736.252` against `$186,310,104,604.25217`
  (`+20.388404%`), and medical HT2 is `+17.325927%`.
- Wrote the Build-P zero-support register with 31 entries. Mapping validation
  proves all 27 carried entries are unchanged and only the requested four
  measured-zero cells were added.
- Wrote the Build-P QRF register with the exact 11-column concentrated set and
  each column's measured share/carrier evidence.
- Parsed both JSON files with `json.tool`, checked their object/string schemas
  with `jq`, and loaded them through the release's actual loader functions.
- Re-ran the exact QRF gate with the new register: passed with no failures,
  stale entries, dormant entries, or unused register keys.
- Wrote the complete measurement and validation report to `FINAL_REPORT.md`.

## Next

- The next Build-P release may consume the two validated runtime registers.
- Mortgage Table 2.1 and medical HT2 remain report-only findings for separate
  follow-up.
