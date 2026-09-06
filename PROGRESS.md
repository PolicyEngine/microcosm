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

- [x] Verified worktree head `84e7a011`, clean, correct branch.
- [x] Verified reference-bundle digest `750b624e…0baf7` matches root acceptance.
- [x] Child-only pytest runner written; foreign editable shards purged;
      baseline `test_us_demographic_calibration_graph.py` 20 passed.
- [x] Activation declaration module + tests (78 passed).
- [x] Household age-count operator, composition, graph and warm-replay tests
      (43 passed).
- [ ] Docs, changelog, CI classification guards.
- [ ] Full focused suite + report.

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

## Next

Classify the two new runtime modules in the spine-blindness guard, verify
`tools/ci_test_groups.py --verify`, write the methods doc and changelog
fragment, then run the focused suite and write the report.

## Limits carried from the brief

- Age-only development calibration. Not national/CD validation, not a release.
- Method rung: observed interview-age features from pooled source cohorts,
  calibrated to 2024 published counts. No birth dates, no invented aging, no
  claim that individuals were observed in 2024.
- B19001 and B25003 stay reserved holdouts and never enter this lane.
- Published standard errors travel with the rows; the current solver does not
  consume them. Controlled-estimate uncertainty is null, never zero.
