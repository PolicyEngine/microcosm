# US combined-survey PUF-support clone graph stage

Session journal for the bounded whole-household clone EXPAND stage. Accurate
when written; check git for current truth.

## Scope (root-adopted, bounded)

Build the missing graph stage that turns an already combined ASEC+ACS native
(clone-index 0) US Frame, carried at its incoming IMPORTANCE household weights,
into a two-role population: the native role plus one full-attachment PUF-detail
clone, with every incoming household weight split in half across the two roles.

In scope:

- Full one-copy detail attachment only (fraction 1.0, no seeded partial arm).
- Preserve both ASEC and ACS source channels, all six entities and every
  membership, the stratum column and the source columns.
- `clone_us_frame_for_puf_support` / `validate_puf_clone_attachment` reused as
  the direct behavioral oracle.
- Real `KernelResult.expand` lineage with explicit household weights; clone
  metadata written through declared graph outputs.

Out of scope, deliberately: PUF donors, QRF fits, draws, placement, any
tax-benefit value, prior wages, partial attachment, calibration. Donor `S006`
never enters host weights. Development profile; not release eligible.

## State

- Fresh worktree `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-combined-clone-graph-20260906`
  on branch `us-combined-clone-graph-20260906`.
- Base: `84e7a011f7bc4d8cb5a7cbd2599b4d893558b1e0` (accepted local composition
  branch), which is a clean fast-forward descendant of `origin/main`
  `c1b8324183819cf91511c002bf724a2676524e87`. No conflicts; nothing merged by
  hand.
- Reading the EXPAND executor contract, the direct clone operator and the
  existing composed-population graph stage.

## Done

- Inspected `/Users/maxghenis/PolicyEngine/microcosm` status/remotes, fetched
  `origin/main`, proved `84e7a011f` descends from it, created the fresh
  worktree preserving all others.
- Read the inventory report
  `us-puf-combined-clone-integration-next-REPORT.json` (an inventory, not a
  scientific approval).

## Next

- Pin the runtime venv and confirm the worktree source actually loads.
- Design and implement the EXPAND kernel plus its clone-metadata outputs.
- Invented mixed-source fixture tests, cold/warm replay, graph-vs-direct
  bijection comparison.
- Write the review report to the review directory.
