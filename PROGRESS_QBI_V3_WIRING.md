# QBI v3 wiring progress

## State

The offline `qbi-v3-wiring` branch is being assembled in
`.claude/worktrees/populace-wt-530`. The completed v2 simulation/content lane
and v3 evidence-resource lane are merged. The engine, host transform, evidence
schemas, package contracts, and restricted replay path are now mapped;
evidence-consuming v3 assumptions and simulation paths remain to be
implemented.

## Done

- Created `qbi-v3-wiring` from local `qbi-v2-content`.
- Merged local `qbi-v3-evidence` with no network access.
- Resolved the sole merge conflict in the shared `PROGRESS.md` journal by
  retaining both dedicated sibling ledgers.
- Confirmed the country-package manifest merged without conflict.
- Read the GitNexus exploration skill. No GitNexus query/context tools are
  exposed in this environment, so source flows will be traced directly.
- Traced the v1/v2 donor simulation, QRF target selection, post-QRF host SSTB
  transform, monolithic/checkpoint builder dispatch, public exports, package
  manifest, and restricted replay tests.
- Confirmed v1 must remain the unchanged production default and that adding
  version 3 to the supported-version tuple automatically exposes it only as an
  explicit CLI choice.
- Recorded the byte-identity boundary: v1's combined W-2/UBIA stream and v2's
  qualification, SSTB, W-2, UBIA, and investment families must not gain or
  lose draws.
- Confirmed the evidence forms, finest-industry partition, usable
  receipts/wage/UBIA rows, SCF income bands, and full-artifact person-weight
  mapping.
- Resolved the record-level legal-form seam for implementation: a positive
  qualified combined partnership/S-corporation source selects the required
  17/53 latent split; remaining positive-QBI records use the sole-proprietor
  proxy. The assumptions resource will state this rule explicitly.
- Chosen the evidence-preserving UBIA branch allowed by the binding design:
  deterministic intensity conditional on the latent industry. The industry
  draw already supplies observed cross-industry heterogeneity, while the SOI
  aggregate tables do not identify an additional within-industry dispersion
  scale.
- Confirmed the output-file convention is a committed
  `FINAL_REPORT_QBI_V3_WIRING.md`; no output-path environment variable is
  present.

## Next

- Build and persist the full-artifact employer-gate calibration.
- Wire version 3 with independent random streams while preserving v1/v2 bytes.
- Add synthetic and restricted replay diagnostics, then run focused and full
  validation.
- Write the final report to the requested output file and leave every result
  committed locally.
