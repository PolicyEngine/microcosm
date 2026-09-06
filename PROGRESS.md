# US national age activation — development lane

Branch `us-national-age-activation-20260906`, worktree
`_worktrees/microcosm-us-national-age-activation-20260906`, started from the
accepted integration lineage `84e7a011`.

## Scope

One concrete calibration connection, development only:

1. An explicit **immutable activation declaration** that binds the accepted
   direct-national ACS 2024 reference bundle
   (`national-derived-inventory.json`, sha256
   `750b624e…0baf7`) and creates a **new** `TargetRegistry` artifact for the
   18 disjoint S0101 `C01_002…C01_019` age counts.
2. A **source-blind population operator** that produces those 18 disjoint
   household age-count columns from declared person ages and household
   membership.
3. An ergonomic composition joining that count node to the **existing**
   `demographic_calibration_node`, unchanged.

Explicitly **not** in scope: genuine population reads, full-source parses,
fits, scoring, national or CD validation, releases, pushes.

## State

Implementation, tests, docs, changelog and CI classification are complete and
committed. The lane is ready for root/Fable review; nothing is pushed.

- [x] Verified worktree head, clean tree, correct branch, `84e7a011` ancestry.
- [x] Verified reference-bundle digest `750b624e…0baf7` matches root acceptance.
- [x] Child-only pytest runner; foreign editable shards purged; every loaded
      `microcosm.*` module re-verified as resolving inside this worktree.
- [x] `national_age_activation.py` — declaration + activation (committed
      `02cee568`, hardened `ed85aeef`, `8222c9a8`).
- [x] `graph_national_age_counts.py` — operator + composition (`3d862703`,
      `a950843c`).
- [x] Tests: `test_us_national_age_activation.py` (26 cases),
      `test_us_national_age_counts.py` (45 cases), including the real graph
      run and the fresh-process warm replay.
- [x] Spine-blindness classification for both new modules, with no
      provenance-owner exemption (`319ede54`).
- [x] `docs/us-national-age-activation.md`, `CLAUDE.md` pointer and the
      towncrier fragment (`d8c8dd3e`).
- [x] Focused suite green; `tools/ci_test_groups.py --verify` ok; ruff
      check/format clean.
- [x] Report written to the review directory.

## Done

- Scouted `demographic_calibration_graph.py`, `TargetRegistry`/`TargetSpec`,
  `cd_reference.classify_acs_value`, the graph `Node`/`Slice`/`Owned` contract,
  and the `test_us_spine_blindness.py` classification lists.
- `national_age_activation.py`: the immutable `NATIONAL_AGE_ACTIVATION`
  declaration and `activate_national_age_targets`, which authenticates the
  pinned inventory, its request descriptors and its raw responses, re-derives
  every selected cell from those exact bytes through the existing
  `classify_acs_value`, verifies publisher labels/group/concept/int predicate
  and the single national row, and mints a **new** `TargetRegistry`.
- `graph_national_age_counts.py`: the source-blind `us.national_age_counts@1`
  operator owning the 18 int64 columns, plus
  `national_age_calibration_nodes` composing it with the unchanged
  `demographic_calibration_node`.
- The graph contract splits the declared read surface: `age` is an owned cell
  and travels in a `Slice`; entity ids and memberships are structural and no
  node owns them, so they are declared in the node's normative parameters and
  the kernel refuses a household view beyond its declared id column.
- The published total `S0101_C01_001` is re-derived and required to be
  partitioned by the 18 activated bands, then discarded — it is deliberately
  not a nineteenth loss term.

## Next

Root/Fable review of the exact implemented activation and its evidence.
Genuine calibration is a separate, later, explicitly authorized step.

## Limits carried from the brief

- Age-only development calibration. Not national/CD validation, not a release.
- Method rung: observed interview-age features from pooled source cohorts,
  calibrated to 2024 published counts. No birth dates, no invented aging, no
  claim that individuals were observed in 2024.
- B19001 and B25003 stay reserved holdouts and never enter this lane.
- Published standard errors travel with the rows; the current solver does not
  consume them. Controlled-estimate uncertainty is null, never zero.
- No genuine population was read, parsed, fit, scored or calibrated here. The
  only real data touched is the public national ACS aggregate bundle.
