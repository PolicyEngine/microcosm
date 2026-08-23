# Progress: replacement scorecard

## State

The `replacement-scorecard` lane is active. The scorer now has one role-neutral
path for the live US incumbent and either a finished-H5 or authenticated-pool
candidate. Fiscal scoring streams fixed registry chunks across fixed household
slices, and a gate-failed current stacked publication remains authenticated
scorecard evidence without being promoted to simulation-ready. The incumbent
side is now scored and committed evidence is being prepared; the owner handoff,
final suite, and final report remain.

No pool build, push, gate change, threshold change, or band change has
occurred. `ps ax | grep build_us_multispine_pool` is checked before every
scoring step.

## Done

- `UV_CACHE_DIR=/tmp/microcosm-scorecard-uv-cache uv sync --all-packages
  --extra us` complete (the sandbox forbids the default user cache).
- Inspected all four salvage snapshots. The newest snapshot `5577ee4c` exactly
  matched the inherited uncommitted scorer; its household-slicing direction
  was retained deliberately, then replaced with slice-local scoring because
  assembling 8,192 full-pool measure arrays would exceed 20 GiB on a 25%
  dense pool.
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
- Added an authenticated scoring-only pool loader. It accepts only the exact
  current stacked `status=gate_failed` / `simulation_ready=false` pair while
  retaining all manifest/diagnostics/H5 digest, schema, run-ID, materializer,
  gate-alias, weight-kind, transition-authority, and row-count checks. The
  production simulation-ready loader remains strict.
- Added a finished-artifact battery evaluator that retains the canonical
  authority, 132-comparison surface, 369 scalar-leg contract, formulas,
  support rules, and tolerances, but explicitly does not claim an assembly or
  tail receipt. The one structural ACS group-quarters scope is reconstructed
  from retained artifact columns and marked unauthenticated.
- Added loader-contract, missing-column, deterministic top-level fixture,
  streaming-vs-one-shot, dense memory-plan, scalar-leg completeness,
  origin-scope, pool-status, and metadata-free battery tests. The combined
  focused run reached 18/19; its sole test-message mismatch was corrected and
  the failed test reran green. Ruff, formatting, byte compilation, and
  `git diff --check` pass.
- Merged the six newer `origin/main` commits cleanly at merge commit
  `34d93846`. Post-merge, all 13 scorer tests and all 8 mainline US
  fiscal-memory tests pass (21/21).
- Completed the incumbent-only score against the exact Build P artifact and
  all 32,842 registry rows. Results: weighted loss
  `0.11462448275649702`, fraction within 10% `0.2669143170330674`, 57,240
  nonzero household weights, and all 132 battery comparisons / 369 scalar
  legs explicitly inapplicable because observed clone-0 ACS support is empty.
  The run peaked at 18.666 GiB and wrote
  `experiments/replacement_scorecard/incumbent_48b9d479.{json,md}`.

## Next

- Commit the incumbent result and its run receipt. Record the exact candidate
  command and comparison doctrine in
  `_LANE-NOTES.md`, run the full workspace suite, and replace the stale
  `FINAL_REPORT.md` with this lane's final evidence.
