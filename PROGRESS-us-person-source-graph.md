# US relationship/eligibility pre-clone source graph adapters

Branch `us-person-source-graph-20260905`, base `ebb47671`.

## State

Implemented, tested and committed. Remaining: final regression sweep record,
author receipt under the review directory, and the build report.

## Done

- Read AGENTS/CLAUDE/DESIGN, `docs/us-production-graph-development.md`, and the
  reviewed patterns (`graph_prior_year.py`, `graph_early_preparation.py`,
  `graph_cps_carried.py`, the registered `multispine_pool` observers).
- Captured the pre-edit direct-chain oracle on a tiny synthetic multispine
  frame: full six-operator pre-clone payload/receipt/context SHA-256 digests,
  pinned in `test_full_legacy_pre_clone_composition_retains_pre_edit_byte_fingerprints`.
- Added registered pool observers
  (`observe_multispine_relationship_person`,
  `observe_multispine_eligibility_person`) plus their boundary-column,
  boundary-validation and implementation-contract helpers. Role stripping,
  CPS selection, canonical `Weights` selection, all-null output omission and
  the full-pool ID merge stay inside the pool owner. The public
  `*_person_carries_signal` predicates run before the always-recomputing
  `prepare_*_person` helpers, preserving the wrappers' order.
- Added `graph_person_preparation.py`: boundary + derive nodes per family,
  typed context, declared person cells, no gate node (neither wrapper gates
  its own outputs). Factories take `base`/`context_producer`.
- Tests: 149 in `test_us_graph_person_preparation.py`. Composed graph/direct
  parity on cold/auto/require; eligibility on the actual direct housing state;
  pass-through, all-null, native-ACS, shuffled-order, refusal and malformed
  cases; implementation-identity binding including `_person_signal_summary`,
  dependencies, the source-stage resource, live output rosters, executed
  seed/period, and end-to-end cache invalidation.
- Mutation-checked the suite: 9 of 10 seeded defects were caught; the tenth
  (dropping the all-null output omission) was behaviourally inert for these
  two families and is now bound by an exact projection-equality assertion.
- Classified the adapter in `test_us_spine_blindness.py` (no fence exemption);
  `tools/ci_test_groups.py --verify` is ok and the new file lands in
  `rest`/`us-am`, never `[defaulted]`.
- Docs: `docs/us-person-preparation-graph.md`; pointer added to
  `docs/us-person-preparation-helpers.md`. Changelog fragment added.

## Next

1. Broader regression sweep across pool/input/graph suites.
2. Author receipt under `~/architecture-reviews/microcosm-20260904/`.
3. `us-person-source-graph-build.md`.

## Limits (standing)

- No push/deploy/calibrate; no held-out reads; no acceptance-threshold change.
- Housing is authored in a separate worktree: not edited, not awaited. The
  composed graph uses a declared stand-in in housing's slot, and a separate
  fixture runs the real direct housing operator to reach the state eligibility
  runs on.
- Nominal engineering parity only; no source-year price alignment here.
- Tiny synthetic fixtures only. Nothing here certifies a data artifact.
