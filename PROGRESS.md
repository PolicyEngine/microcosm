# Attempt 9 register maintenance

## State

Measurements complete and internally reconciled. Runtime register edits are
authorized by the measured frame but have not yet been made.

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

## Next

- Copy the Build-M zero-support register unchanged and append only the four
  measured-zero Build-P cells with evidence in the reason strings.
- Write the Build-P QRF register with exactly the 11 measured concentrated
  columns and current shares/carrier counts.
- Parse both files through the release's actual loaders and re-run the QRF
  gate with the new register to prove exact key-set use.
- Write and commit `FINAL_REPORT.md`.
