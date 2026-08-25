# Final report: stacked-pool to release CD-vintage provenance

## Outcome

The producer/consumer defect is fixed. The production stacked route now
produces the same authenticated congressional-district contract that release
preflight requires: the canonical crosswalk SHA, target vintage
`119th_congress`, and positive household
`congressional_district_geoid` support. Geography remains outside source
assembly, and the release guard still fails closed on every mismatch
(`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-406`;
`tools/build_us_fiscal_refresh_release.py:2565-2613,8608-8612`).

## Five-part repair

1. The stacked CLI requires path/SHA pairs for the national PUMA ladder and
   canonical 117th-to-119th CD crosswalk. Both declared hashes must equal the
   repository pins before the source bytes are authenticated
   (`tools/build_us_multispine_pool.py:563-594,897-1004`). The real geography
   operator runs immediately after operator-free source assembly and before
   gap fill or cloning
   (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-406`;
   `tools/build_us_multispine_pool.py:5290-5334`).

2. The configured namespace, checkpoint identity, persisted stage receipts,
   and terminal manifest bind both byte authorities, source and target
   vintages, assignment declaration, overlap algorithm, operator order, seed
   site/stream/value, and ordered native-household output receipt
   (`tools/build_us_multispine_pool.py:814-928,1300-1469,1711-1930,3530-3635,3989-4027,4133-4215`).

3. Nullable H5 publication accepts validated root attributes, writes them in
   the same temporary HDFStore as the fixed entity tables, verifies their
   exact round trip, and atomically replaces the destination only after all
   checks pass. The stacked publisher supplies the crosswalk-SHA and target-
   vintage attrs
   (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:1463-1699`;
   `tools/build_us_multispine_pool.py:4423-4519`). Schema-9 loading also binds
   those physical attrs and the live household geography/clone lineage back to
   the manifest
   (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:671-923,1032-1193,1321-1364`).

4. Release preflight now reads root attrs and the household frame from one
   HDFStore snapshot through the shared fixed/table-aware reader. Its existing
   equality checks for crosswalk SHA, current target vintage, and positive
   household district support are unchanged
   (`tools/build_us_fiscal_refresh_release.py:2565-2677`).

5. A tiny fixture invokes the real stacked entry point, post-assembly
   geography operator, fixed-H5 publisher, and real release assertion. It
   proves both root attrs and positive household district support; consumer
   tests reject changed attrs, changed native geography, missing lineage, and
   divergent clone geography
   (`packages/microcosm-build/tests/test_us_multispine_pool_tool.py:2279-2335`;
   `packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py:1357-1440`;
   `packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:1061-1184`).

## Assignment design and anti-rot

A one-value deterministic PUMA-to-CD lookup is not defensible because 2020
PUMAs and congressional districts do not nest. The existing national ladder
preserves observed ACS PUMA, assigns missing ASEC PUMA proportional to 2020
PUMA population, then draws CD and county within PUMA proportional to block-
population overlap. Stable row/state/PUMA order and the ledgered
`legacy_puma_ladder` / `geography_legacy` / seed `0` contract make the draw
reproducible for fixed inputs
(`packages/microcosm-build/src/microcosm/build/us_runtime/puma_ladder.py:1-20,293-383,638-698`;
`packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:870-887`;
`packages/microcosm-build/src/microcosm/build/us/spec/spine.yaml:421-432`).

The spec and schemas now declare the crosswalk source/vintage authority and
geography assignment, while field-usage and inventory owners cover the new
surface exactly
(`packages/microcosm-build/src/microcosm/build/us/spec/geography.yaml:3-37`;
`packages/microcosm-build/src/microcosm/build/us/spec/sources.yaml:69-86`;
`packages/microcosm-build/src/microcosm/build/us/spec/vintages.yaml:45-53`;
`packages/microcosm-build/src/microcosm/build/spec_engine/field_usage.py:387-393,671-692,803-822`;
`packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py:348-380,1649-1712`).

The generated US spec SHA is
`5378bb9189aec96f50da22aac71e5bd2c3d919e9795f6ef2147e0bc9c739dd8e`.
Coverage is exact at 42,120/42,120 fields, 49 claims, and 41/41 inventory
checks. Principal legitimate movements are pointer inventory
`6d7353c6...` to `bc4a948a...`, full checkpoint `b6a47fac...` to
`a128a85f...`, and the new geography identity `f49425ca...`; executable
versions move checkpoint 11 to 12, terminal manifest 8 to 9, and nullable H5
materializer 2 to 3
(`tools/spec_engine_coverage.py:42-45`;
`packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py:348-380`;
`docs/evidence/spec-engine/us-f0-coverage.json:10,775-788,1890,2027-2057,2602`;
`tools/build_us_multispine_pool.py:332-360`;
`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:75-105`).

## Verification

- `microcosm-calibrate`: 203 passed.
- `microcosm-data`: 318 passed / 2 skipped.
- `microcosm-fit`: 93 passed.
- `microcosm-frame`: 295 passed / 36 skipped.
- `microcosm-build` partition A: 4,155 passed / 36 skipped.
- `microcosm-build` partition B: 2,177 passed / 3 skipped.
- Accepted workspace total: **7,241 passed / 77 skipped / 0 failed**.
- The high-risk real integration and H5 consumer cases are included in those
  build partitions
  (`packages/microcosm-build/tests/test_us_multispine_pool_tool.py:2279-2335`;
  `packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py:1357-1440`).
- Bundle freshness reports the spec SHA above; coverage freshness reports
  42,120/42,120 fields and 41/41 inventory checks
  (`tools/generate_us_bundle_from_constants.py:355-437`;
  `tools/spec_engine_coverage.py:378-398`).
- Repository-wide Ruff, smoke-script shell syntax, and `git diff --check` pass.
  The source-blind import-graph anti-rot pin now covers the exact 69-module
  graph reached by the three new shared validators
  (`packages/microcosm-build/tests/test_us_spine_blindness.py:3270-3315`).

No production pool or release build ran, no push occurred, and
`logbook-pending-chain.txt` was not touched.

## Candidate handoff

Stage 1 must add the canonical PUMA-ladder and CD-crosswalk path/SHA pairs
listed exactly at the end of `_LANE-NOTES.md`, run off-chain, and rebuild under
the new checkpoint identity. The old smoke checkpoints and pool publication
are not reusable: configured identity now includes both new input pins and the
geography contract, checkpoint materializer is 12, terminal manifest schema is
9, and nullable H5 materializer is 3
(`tools/build_us_multispine_pool.py:332-360,1300-1555`;
`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:75-105`).
The six pre-existing immutable source artifacts can be reused; the Stage-1
checkpoint namespace and pool H5 must be regenerated with the two additional
authenticated authorities (`tools/build_us_multispine_pool.py:931-1004,1292-1319`).
