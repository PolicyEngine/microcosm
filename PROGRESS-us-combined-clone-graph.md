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
- Runtime: no venv in this worktree and the shared
  `/Users/maxghenis/PolicyEngine/microcosm/.venv` has neither `microcosm-graph`
  nor `jsonschema`. Reads run under the existing
  `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-preclone-composition-20260905/.venv`
  interpreter with `PYTHONPATH` pinned to this worktree's six `packages/*/src`
  trees and `PYTHONDONTWRITEBYTECODE=1 -B`. Verified: every imported
  `microcosm.*` module resolves inside this worktree. No installs.
- Executor contract, direct clone operator, provenance owner, spine assembly
  and the one existing production EXPAND family (UK) all read.

## Design (settled, from the read sources)

`_patch_expand` (microcosm-graph/population.py) takes per-entity lineage
(new id -> copied base id), the declared `expand_cells` overlays, and explicit
weights for `expand_weight_entity`; it carries every other column itself and
remaps copied persons' memberships onto the copied groups by clone ordinal.
The stage therefore:

1. Rebuilds a minimal US Frame from its declared slices, household weights and
   strata, plus a derived spine-assembly manifest keyed to a declared
   `source_channels` param.
2. Calls `clone_us_frame_for_puf_support(..., clone_attachment_fraction=1.0,
   clone_attachment_seed=0)`, which internally runs
   `validate_puf_clone_attachment` full-clone identity. Oracle and validator
   are the same call the direct path makes.
3. Derives the lineage from the operator's own output by the
   `(entity_source_id, clone_index)` bijection and cross-checks it positionally;
   it never assumes an ID offset rule.
4. Returns the six `*_support_clone_index` overlays and the halved IMPORTANCE
   household weights. A follow-on claim node owns those six cells as full-cell
   rewrites, which is also what makes the executor accept a copied row whose
   clone index differs from its source.

Mass: `conserve`. Halving household weights while doubling rows leaves the
executor's weighted person mass per stratum unchanged, so the ledger is a real
check rather than a declared one.

## Done

- Inspected `/Users/maxghenis/PolicyEngine/microcosm` status/remotes, fetched
  `origin/main`, proved `84e7a011f` descends from it, created the fresh
  worktree preserving all others.
- Read the inventory report
  `us-puf-combined-clone-integration-next-REPORT.json` (an inventory, not a
  scientific approval).
- Pinned the runtime and proved the worktree source is what loads.
- Probed an invented mixed ASEC/ACS pool through `assemble_spines` and through
  the direct clone operator: IMPORTANCE weights halve, ids remap by decimal
  multiplier on the detail role only, native ids and both source channels
  survive.

## Next

- Implement `graph_combined_clone.py` (EXPAND kernel + claim kernel + nodes).
- Invented mixed-source fixture tests: exact-field preservation, per-pair
  weight sums, refusals, cold/warm replay, graph-vs-direct bijection over every
  column of every entity.
- Write the review report to the review directory.
