# Armenia v1 DIAGNOSTIC calibration run — progress

## State

The first main solve is complete and confirms that the default loss cap plus
256 epochs are under-converged; one permitted same-method sanity rerun is next. The checked-out PR
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
- Implemented `_diagnostic/run_calibration.py` with strict schema, unit, total,
  donor-hash, linkage, support, and sparse-matrix validation plus all required
  DIAGNOSTIC-only artifact writers. The runner passes Python compilation and
  Ruff checks under the already-synced local environment.
- Completed the plain `apg`-entry-point main solve at 256 epochs. The installed
  package normalized that deprecated alias to Adam/log-weights as expected; the
  616 x 629,640 matrix stayed sparse CSR with 3,232,526 nonzeros. Final loss was
  0.12182061, but the trajectory was still falling and the worst small LFS cell
  remained 1,131.33% high, so this is not treated as a converged final result.
- Diagnosed the first pass's loss-cap pathology: 74 of 616 cells started at or
  past cap 10 (and therefore had no direct gradient), including 51 of 56 Vayots
  Dzor cells; one cell remained past cap at the end. Preserved all 15 first-pass
  artifacts under `_diagnostic/attempts/main_256_cap10/`.
- Added a validated `--target-loss-cap` runner option, cap-census and attempt
  history reporting, dynamic solver provenance, validation of the excluded
  industry and alternative-census units/dimensions, and explicit reporting of
  the start's 8,691,465.96 implied person mass. Compilation and Ruff pass.

## Next

- Perform one same-method 1,024-epoch hyperparameter sanity rerun at cap 100
  (above the maximum initial scaled miss of about 90.47), with no pruning,
  sampling, value-space calibration, L1, or L2.
- Emit fit, weight-health, provenance, decision, and weights artifacts.
- Verify all outputs and confirm `git status --short` contains no diagnostic
  files.
