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
  `origin/main`, proved `84e7a011f` descends from it, worked in the fresh
  worktree, preserving all others.
- Read the inventory report
  `us-puf-combined-clone-integration-next-REPORT.json` (an inventory, not a
  scientific approval).
- Pinned the runtime and proved the worktree source is what loads: 266 loaded
  `microcosm.*` modules, all inside this worktree, every working-tree blob
  equal to its committed blob.
- Implemented `graph_combined_clone.py`: the `EXPAND` kernel that calls
  `clone_us_frame_for_puf_support` / `validate_puf_clone_attachment` and
  returns lineage + six clone-index overlays + halved IMPORTANCE household
  weights, the claim kernel that owns those cells as full-cell rewrites, and
  the two-node declaration factory.
- Wrote `test_us_graph_combined_clone.py`: 40 tests over an invented mixed
  ASEC/ACS host with colliding raw ids, a zero-weight household, unequal
  weights, two tax units in one household, non-contiguous ids and a person
  order that is not the id order. Graph equals the direct operator column for
  column through the declared `(channel, source id, role)` key; native rows are
  untouched; both channels survive in both roles; every pair sums to its parent
  exactly; required warm replay recomputes nothing; ten refusal paths.
- Documented the stage in `docs/us-combined-survey-clone-graph.md` and pointed
  `CLAUDE.md` at it and at its limits.
- Recorded evidence in the review directory: test log/XML, engine-free rerun,
  peer-suite regression run (808 tests), lint log, source proof, identity JSON.

## Verified numbers (this worktree, head recorded in the report)

- New suite: 40 passed, 0 failed, 0 skipped.
- Peer suites (`microcosm-graph`, `microcosm-frame`, `test_us_puf_support.py`,
  `test_us_graph.py`): 808 tests, 0 failures, 0 errors, 68 skipped.
- `ruff check .` clean; `tools/ci_test_groups.py --verify` = ok, file lands in
  `rest` (fast) and `us-am` (engine-us), never `[defaulted]`.
- Executor mass ledger for the stage: `conserve`, weighted person mass
  3279.456 before and after, identical in every stratum.
- `puf_support.py` sha256 `ac74ae55…e266ed` — byte-identical to the accepted
  composition head; the operator was not edited.

## Next (not this lane)

- Root/Fable review of this bounded stage.
- Separately reviewed scopes: recipient feature construction, return-native
  donor admission, per-target fit, draw, placement. None of them belong here.
