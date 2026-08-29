# Armenia v1 DIAGNOSTIC calibration run — progress

## State

In progress. The checked-out PR branch remains untouched. Diagnostic artifacts
are confined to the git-excluded `_diagnostic/` directory. This journal is
mirrored as `PROGRESS.md` only on the local side ref
`refs/heads/diagnostic-armenia-814-journal`; the mirror is created with Git
plumbing and never with `git add`.

## Done

- Read `CLAUDE.md` and the complete Armenia `HARVEST.md` and `NOTES.md` doctrine.
- Confirmed `_diagnostic/` is excluded by the shared Git exclude file.
- Confirmed the checked-out branch starts clean at `305a13ed`.
- Chose the registry-pinned published `policyengine/populace-us` 2024 HDF5 as
  the diagnostic donor, subject to local hash and schema verification.

## Next

- Inspect the GitNexus repository context and calibration/frame implementation.
- Inventory and validate all local donor and harvested inputs, including units.
- Compile sparse targets, clone households once per marz, solve with APG, and
  emit fit, weight-health, provenance, and decision artifacts.
- Verify all outputs and confirm `git status --short` contains no diagnostic
  files.
