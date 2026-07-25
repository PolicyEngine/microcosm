# QBI v3 wiring progress

## State

The offline `qbi-v3-wiring` branch is being assembled in
`.claude/worktrees/populace-wt-530`. The completed v2 simulation/content lane
and v3 evidence-resource lane are merged; evidence-consuming v3 simulation
paths and replay calibration remain to be implemented.

## Done

- Created `qbi-v3-wiring` from local `qbi-v2-content`.
- Merged local `qbi-v3-evidence` with no network access.
- Resolved the sole merge conflict in the shared `PROGRESS.md` journal by
  retaining both dedicated sibling ledgers.
- Confirmed the country-package manifest merged without conflict.
- Read the GitNexus exploration skill. No GitNexus query/context tools are
  exposed in this environment, so source flows will be traced directly.

## Next

- Inventory the versioned engine, assumptions builder, evidence schemas,
  package contracts, and replay diagnostics.
- Build and persist the full-artifact employer-gate calibration.
- Wire version 3 with independent random streams while preserving v1/v2 bytes.
- Add synthetic and restricted replay diagnostics, then run focused and full
  validation.
- Write the final report to the requested output file and leave every result
  committed locally.
