# US SIPP December housing source projection — lane journal

Branch `us-sipp-housing-projection-20260906`. Base `origin/main`
`c1b8324183819cf91511c002bf724a2676524e87`, fast-forwarded to the frozen parent
`ab10f77854b7b9e8057a07dbec40d15c10b96297` (`origin/main` is an ancestor of the
frozen head; the frozen head is 124 commits ahead, so the fast-forward is exact).

Scope: housing metadata, a pure housing measurement kernel, and an executable,
versioned source projection **recipe**. No genuine source extraction, no fit, no
transfer, no publication. Root runs the extraction after reviewing this lane.

## State

Complete and root-reviewable. Nothing was pushed; no PR exists.

## Done

- Confirmed the base/frozen-head relation and created this worktree.
- Hash-verified the cached official 2023 SIPP data dictionary
  (`0f1c5fbd…894f`, 3,991,831 bytes, 3,371 pages) and transcribed every field,
  status flag, code domain and published universe from its own pages.
- Confirmed all 29 recipe columns exist in the pinned parent's header. Only the
  header line was read; no data row was opened.
- Added `microcosm.build.sipp_housing_measurements` (pure kernel) and
  `microcosm.build.sipp_housing_projection` (versioned recipe, diagnostics,
  receipt). Both reuse the frozen modules' internals.
- Left `sipp_december_source.py` and `sipp_measurements.py` byte-identical, so
  the already-published producer binding and receipts stay valid.
- 70 invented-fixture tests in
  `packages/microcosm-build/tests/test_us_sipp_housing_projection.py`; 240 tests
  pass across the four affected SIPP files, plus the 498 `test_us_spine_blindness`
  guards. `tools/ci_test_groups.py --verify` reports `verification=ok` with the
  new file in `rest` / `us-qs`, never `[defaulted]`.
- Documented the contract in `docs/us-sipp-december-housing-projection.md` and
  added the changelog fragment and a CLAUDE.md pointer.
- Wrote the bounded first-32-SSUID extraction proposal and its invented-data
  self-check under the review root. The proposal was run **plan-only**; it has
  never opened a SIPP byte.

## Next (for root, not this lane)

- Review the recipe and run the proposal with `--execute` for
  `first_32_ssuids`.
- Only after reading that receipt and diagnostics, consider the gated
  `all_ssuids` rung.
- Common support, geographic coverage and the fit/evaluation split remain
  unbuilt: they need a recipient universe and predictors that do not exist yet.
