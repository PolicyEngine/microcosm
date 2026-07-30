# Progress

## State

Populace #395 increment 1 is in implementation on
`multispine-operator-ordering-395`, based on `origin/main` at `0d99d8a`.
The opt-in pre-operator spine assembly seam and pre-calibration
spine-agreement contract are implemented and focused tests pass. PUF
operator-clone roles are now separate from immutable source-spine provenance.
The population-operator migration and spine-blindness structural guard are
implemented and their focused regression suites pass. The call-graph/target
ordering design note and changelog fragment are complete. Current sparse and
dense release behavior remains unwired to the new seam.

## Done

- Read epic #578, including the governing Scope hardening: one suite per
  country, exact-k as the only US release variation, full geography in every
  file, and retirement of the separate local artifact.
- Confirmed the locally cached `origin/main` and requested base commit both
  resolve to `0d99d8a`.
- Created this isolated worktree and branch.
- Attempted `git fetch origin`; the managed environment cannot resolve
  `github.com`, so the lane uses the already exact local remote-tracking ref.
- Read the repository agent guidance and selected the GitNexus exploration,
  impact-analysis, and refactoring workflows for the cross-module seam.
- Mapped the two current build call graphs: ASEC enrichment and PUF cloning,
  transfer, derivation, seeding, and geography all precede the late ACS
  append; the ACS builder transfers from that already-operated donor.
- Added the opt-in `assemble_spines(...)` stage. It deterministically combines
  two or more raw US peer frames, aligns nullable source columns, remaps ID
  collisions, conserves the anchor household mass, and records immutable
  source channel/source ID/clone-index-zero provenance on every entity.
- Added focused assembly tests covering mass allocation, provenance, raw input
  immutability, collision remapping, future channels, dtype contracts, and
  rejection of PUF as a peer spine.
- Verified the assembly test module (8 tests), its ruff lint, and diff
  whitespace checks using the existing Populace environment.
- Added the canonical spine-agreement registry with exact coverage of the
  declared ACS transferred-input surface, including deterministic derived
  transfer outputs and normalized batch families.
- Declared one fixed statistic and tolerance contract for every registry
  column: weighted nonzero-incidence ratio in `[0.8, 1.25]` and a weighted
  conditional q10/q25/q50/q75/q90 symmetric-relative envelope no greater than
  `0.25`. The gate compares every source-spine pair and batches all failures.
- Added focused agreement tests covering registry exactness, weighted
  measurement, batched failures, nullable evidence, malformed registries, and
  rejection of the legacy PUF clone role as source-spine provenance.
- Split immutable source provenance from operator clone roles: assembly emits
  an assembly-unique `*_source_id`, the original local
  `*_spine_source_id`, the unchanged source channel, and clone index zero.
- Extended the existing PUF clone entrypoint to accept an assembled frame.
  Every source spine receives a native and primary PUF-detail copy; source
  channels and both source-ID fields remain unchanged while structural IDs
  and clone indices carry operator identity.
- Routed PUF QRF recipients and the capital-gains tail through clone
  provenance instead of source-spine provenance while retaining the legacy
  unassembled call path and private keyword compatibility.
- Added combined-frame clone/QRF tests and passed the 69-test PUF support,
  QRF-chain, capital-gains-tail, and multispine-clone regression set. Assembly
  tests and focused ruff/diff checks also pass.
- Migrated the population operators that formerly treated
  `*_support_channel` as an ASEC/PUF switch to the centralized clone-role
  resolver. On assembled frames, clone indices determine the native or
  PUF-detail role while arbitrary declared source channels remain inert.
- Preserved the current unassembled lineage's fail-closed checks: without raw
  spine IDs, the historical channel field must still contain complete,
  clone-consistent ASEC/PUF role labels.
- Added an AST guard over the full US runtime and an exact registry of the 27
  migrated population-operator modules. It rejects direct source-channel or
  source-spine reads and separately pins clone-index routing in the PUF clone,
  QRF, and tail stages.
- Passed the 28-module focused operator regression suite after the compatibility
  validation change; the four-test structural guard, ruff checks, and diff
  whitespace checks also pass.
- Added the design note mapping the current two-build call graph, consumed
  state, provenance axes, target ordering, raw-only boundary, fixed agreement
  statistics, calibration boundary, and increment-1 compatibility boundary.
- Added the #395 changelog fragment.

## Next

- Run the current-lineage builder suites and final combined verification.
