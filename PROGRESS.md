# Progress: replacement scorecard

## State

The `replacement-scorecard` lane is active. The task: one head-to-head
scoring path for the live US incumbent and the not-yet-built 25% bundle-mode
candidate, score the incumbent now, and leave the owner the exact candidate
command. The scorer (`tools/score_us_release_head_to_head.py`) and its test
file are committed with green affected suites; the incumbent scoring run is
the current step.

No pool build, push, gate change, threshold change, or band change has
occurred. `ps ax | grep build_us_multispine_pool` is checked before every
scoring step.

## Done

- `uv sync --all-packages --extra us` complete; environment normal.
- Verified every symbol the salvaged draft imports and every cited mechanism
  against the code: compile surface (five inputs, no membership flags),
  dropped/skipped detection, sqrt–concept-budget–50/50 loss weights, capped
  weighted-MAPE aggregate, relative-error rule, attribution row keys, battery
  registries (131 + 1 joint) and receipt keys, pool-manifest authentication
  chain, `terminal_gates` manifest shape.
- **Corrected the incumbent identity** (see `_LANE-NOTES.md`): live
  policyengine.py is 5.0.3 (2026-08-21), whose bundled manifest resolves the
  US default dataset to `policyengine/populace-us` revision
  `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`,
  filename `populace_us_2024.h5`, SHA-256 `48b9d479...` — microcosm's own
  buildp sparse artifact, not enhanced_cps_2024 (4.15.0-era). Cached bytes
  hash-verified.
- Probed the real incumbent bytes through the scorer's own probe path:
  entity-tables layout, no artifact metadata (caught), CD legacy-waiver path
  (attrs null, 436 positive unique CD geoids), calibrated weights, provenance
  columns present on all three battery entities (person/tax_unit/spm_unit),
  **zero `acs` rows anywhere** → battery inapplicable with the observed
  empty-ACS-side reason.
- `tools/score_us_release_head_to_head.py` + 7 contract/determinism/fixture
  tests committed; head-to-head, state-files scorer, refresh-builder,
  pool-h5-io, fiscal-targets, release-target-parity files green (428 tests);
  ruff clean. `test_us_stacked_spine.py` runs long in the background as part
  of full-suite validation (no package code was touched by this lane).

## Next

- Check the build queue, then run the incumbent scoring (RSS < 20 GiB) with a
  persistent `--target-materialization-cache-dir` so repeated bounded
  invocations resume the 11 JCT reform vectors; commit
  `experiments/replacement_scorecard/incumbent_48b9d479.{json,md}`.
- Record the owner's candidate command and the comparison doctrine in
  `_LANE-NOTES.md`; finish full workspace suite; update `FINAL_REPORT.md`.
