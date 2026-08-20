# F1 continuation r4 progress

## State

- Work is local on `spec-engine-f1`; nothing has been pushed, no sample build
  has run in this continuation, and the logbook pending chain is untouched.
- Cached `origin/main` and the final #698 branch are merged. The 72-site seed
  ledger supersedes the merged 53-site ledger, and the corrected compiler,
  seed, bundle, loader, and graph identities are pinned.
- Deliverable 7 is complete at `5228cf5c`. Its exact-cell code correction was
  swept into concurrent shared-index commit `5875be22`; history was preserved
  and the boundary is documented in the D7 handoff.
- Deliverable 4 is not complete: bundle mode selects typed plan/provenance
  authorities but physical stages still execute through constants, and the
  production artifact collector/comparator is not wired. Deliverables 5 and 6
  therefore cannot be certified honestly.
- The first post-merge full-suite run reached 100% and exposed only three
  fail-closed audit pins for the new authority modules. The reviewed
  classification/runtime-graph correction passes all 495 tests in the
  spine-blindness module; a clean full-suite rerun is next.
- A 1% run is prohibited on this host: four recorded cold primary-QRF peaks
  are 78.91--96.95 GiB RSS, above the lane's 20 GiB ceiling.

## Done

- Attempted the required fetch first; DNS was unavailable, so merged cached
  `origin/main` at `35fb3ed0` and cached final #698 at `da45dfcd` while
  preserving both main's schema/guard/archive work and the 72-site correction.
- Corrected the finalizer structural delta, late-transfer effects, three
  target-balanced cap protocols, two source seed materials, and five
  deterministic hash classifications; regenerated the US bundle and F0
  coverage evidence in `5875be22`.
- Re-pinned the 72-site/57-owner/131-binding seed protocol and all affected
  country, loader, graph, and compiled-map identities. Coverage now reports
  41,911/41,911 fields and 40/40 inventory checks.
- Verified the correction suite (101 passed), merged/D7 suite (229 passed), D7
  exact-cell suite (70 passed), generated-bundle check, coverage check, focused
  spine audit (3 passed), complete spine-blindness module (495 passed), Ruff,
  and whitespace checks at their respective checkpoints.
- Confirmed read-only that no valid owner 1%/25% command exists at this state:
  the physical executor and sealed two-tier production comparison remain
  preconditions, and measured 1% memory requires a suitably provisioned host.

## Next

1. Commit the reviewed spine-blindness audit correction and rerun the full
   suite from the clean committed tree.
2. Append the exact identity/test receipts and honest D4--D6 stop to the lane
   journal, rollout status, and `FINAL_REPORT.md`; do not create certification
   evidence for builds that did not run.
3. A future authorized continuation must wire the 38 compiled producer nodes
   through the executor/brokers, seal exact artifact member inventories, add
   calibration ownership, and pass the cold D4 fixture gate before issuing
   any 1% or 25% command.

## F1 D7 split-out progress — 2026-08-20 current-head audit

This section is the split-out's `state`/`done`/`next` journal and supersedes
only earlier D7 status text. It does not update the main lane's D4--D6/D8
state.

### State

- Deliverable 7 is complete at `5228cf5c`: every active production-shaped
  derived-closure, segment, and lineage-dashboard consumer is compiler-fed,
  and exact-cell authority preserves disjoint graph/family atoms.
- Concurrent main-lane commit `12df8c45` advanced the shared branch after the
  D7 handoff. It changed no D7 source or test file.
- The D7 working tree is clean apart from three pre-existing untracked charter
  drafts. Nothing was pushed and no pool build or sample rung ran.

### Done

- Re-audited tracked tests/tools and found no active reference to the held
  lineage YAML, 392-column inventory, old closure helper, or authored-class
  dashboard fields. Two independent read-only reviews approved the compiler
  boundary and exact-cell closure.
- Re-ran all four D7 modules: 70 passed in 420.42 seconds. Two separate hash
  seeds emitted byte-identical dashboard JSON with SHA-256
  `03c08eb0e59b4ce2fa0d2ffe2bcf62e503005569b1171520a46068dec99ef7df`.
- Ran every test shard serially. Calibrate, data, fit, and frame completed with
  865 passed and 37 skipped. The build shard completed with 6,322 passed, 37
  skipped, and three spine-blindness failures collected before concurrent
  commit `12df8c45` classified the new main-lane runtime modules. The exact
  three tests pass at current HEAD (3 passed in 4.42 seconds); the main lane
  separately records all 495 tests in that module passing.
- Scoped Ruff check/format and the D7 diff whitespace check pass. Repository-
  wide Ruff still reports one main-lane import-order finding at
  `tools/build_us_multispine_pool.py:1157`; D7 does not alter that D5/D6/D8
  driver.
- Validation was serialized with no build or sample workload. The sandbox
  blocks both `ps` and `/usr/bin/time -l` resource reporting, so an exact RSS
  peak is unavailable and is not asserted.

### Next

- No D7 implementation work remains. Preserve the compiler-derived seam and
  exact-cell regression when the main lane changes compiler outputs.
- The main lane owns its clean one-shot full-suite rerun, the unrelated Ruff
  import-order finding, physical executor work, and certification gates.
