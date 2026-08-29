# Armenia v1 DIAGNOSTIC calibration run — progress

## State

Inputs and method are validated; implementation is next. The checked-out PR
branch remains untouched. Diagnostic artifacts are confined to the git-excluded
`_diagnostic/` directory. This journal is mirrored as `PROGRESS.md` only on the
local side ref `refs/heads/diagnostic-armenia-814-journal`; the mirror is
created with Git plumbing and never with `git add`.

## Done

- Read `CLAUDE.md` and the complete Armenia `HARVEST.md` and `NOTES.md` doctrine.
- Confirmed `_diagnostic/` is excluded by the shared Git exclude file.
- Confirmed the checked-out branch starts clean at `305a13ed`.
- Chose the registry-pinned published `policyengine/populace-us` 2024 HDF5 as
  the diagnostic donor; its 462,915,783-byte blob matches the required SHA-256.
- Inspected the HDF5 with `h5py`: 57,240 household rows and 166,321 person rows
  link exactly; cloning each household once across 11 marzes yields 629,640
  household-weight records while preserving 1,829,531 cloned person links.
- Selected 616 leaf targets with their declared units: 418 population
  age-by-sex-by-marz cells (2024-01-01 estimate), 66 ILCS household-size-by-marz
  cells at their own total, and 132 LFS employed-age-by-sex-by-marz cells.
- Excluded the 462 industry cells: the donor has occupation and class-of-worker
  fields but no industry/NAICS/SIC field, so no deterministic NACE bridge exists.
- Selected `PEIO1COW > 0` as the direct current-job/class-of-worker employment
  indicator; its difference from the Armenian ILO `empj` concept will remain a
  named diagnostic limitation.
- Traced `TargetSpec` -> `TargetSet` -> CSR matrix -> `calibrate`. In this branch
  `method="apg"` is a deprecated compatibility alias that runs and records Adam
  over log-weights; there is no public direct-CSR solver seam.
- Chose sparse pre-aggregated household member-count columns so the public
  matrix compiler and serializable `TargetSpec` path remain in use without a
  dense 629,640 x 616 DataFrame.

## Next

- Implement the reproducible compiler/runner and input validation.
- Compile sparse targets, clone households once per marz, then run the single
  main solve through the APG compatibility entry point with no pruning,
  sampling, value-space calibration, L1, or L2.
- Emit fit, weight-health, provenance, decision, and weights artifacts.
- Verify all outputs and confirm `git status --short` contains no diagnostic
  files.
