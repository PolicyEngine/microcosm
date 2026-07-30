# Progress

## State

Populace #395 increment 1 is in implementation on
`multispine-operator-ordering-395`, based on `origin/main` at `0d99d8a`.
The opt-in pre-operator spine assembly seam is implemented and focused tests
pass. Operator clone-role separation, the spine-blindness structural guard,
and the pre-calibration spine-agreement gate contract are in progress. Current
sparse and dense release behavior remains unwired to the new seam.

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

## Next

- Commit the assembly seam and its tests.
- Separate PUF clone role from immutable source-spine provenance, then add
  structural spine-blindness enforcement.
- Add the spine-agreement gate specification and registry.
- Add the design note, changelog fragment, and focused/full verification.
