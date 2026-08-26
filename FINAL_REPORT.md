# Final report: local ACS hours transfer and signal gate (#765)

## Outcome

Issue #765 is fixed in commits `8917b5bc` and `d37df3b3`. ACS-spine persons in
the local-area lane now receive a QRF-modeled
`weekly_hours_worked_before_lsr` from the same national hours family used by
the stacked-spine path, with the existing family/pattern/fit-draw seed evidence
retained. Local finalize also applies `us_hours_worked_signal_gate` to the two
persisted hours leaves on ACS persons alone and treats failure as a hard
simulation-readiness blocker
(`tools/_legacy/build_us_acs_multispine_base.py:80-129,267-317`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1507-1627`;
`tools/build_us_acs_local_release.py:1196-1241,1507-1663`).

No build or release driver ran, no push occurred, and no plausibility band or
acceptance threshold changed. Repository verification is green at **7,247
passed / 77 skipped / 0 failed**. The required fresh local-lane data build is
deferred to the serial host queue; the exact fail-closed command is the literal
end of `_LANE-NOTES.md`.

## Verified defect trace

1. The national source operator requires raw ASEC `HRSWK`, `A_HRS1`, and
   `WKSWORK`; it maps `HRSWK` directly to
   `weekly_hours_worked_before_lsr`. The national pool declares this operator
   pre-clone and dynamically exposes the persisted weekly leaf as
   `person/source_operator_hours_worked` in its pre-clone gap-fill plan
   (`packages/microcosm-build/src/microcosm/build/us_runtime/hours_worked.py:104-185`;
   `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:491-501,692-785,2139-2174`).
   That family reaches the shared QRF transfer before cloning in the stacked
   route
   (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:9251-9305`).

2. The generic ACS transfer declaration deliberately excludes runtime-owned
   hours-before-LSR, although it includes `hours_worked_last_week`; native ACS
   mapping does not supply the three ASEC source columns, so rerunning the
   source operator on ACS persons would fail rather than model the leaf
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:307-357`;
   `packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:61-174`;
   `packages/microcosm-build/src/microcosm/build/us_runtime/hours_worked.py:104-175`).
   Before this repair, the local builder used only that generic declaration
   (`5abda614:tools/_legacy/build_us_acs_multispine_base.py:219-263,344-415,489-523`).

3. A missing local leaf entered the reviewed-null inventory. Consumer export
   then filled each registered null with the installed PolicyEngine-US
   variable default and wrote those filled bytes to the published H5. That is
   the silent defaulting mechanism reproduced by #765, not a fitted hours
   value
   (`tools/_legacy/build_us_acs_multispine_base.py:1119-1155`;
   `tools/build_us_acs_local_release.py:259-356,1096-1155`).

4. The shared hours gate already rejected missing/constant columns and
   implausible weighted worked share or worker mean, and national pool/fiscal
   paths called it. The local finalize path did not call it at base commit
   `5abda614`, leaving no hard check on the defaulted ACS leaf
   (`packages/microcosm-build/src/microcosm/build/us_runtime/hours_worked.py:359-421`;
   `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:1792-1821`;
   `tools/build_us_fiscal_refresh_release.py:9333-9360`;
   `5abda614:tools/build_us_acs_local_release.py:1432-1578`).

## Transfer repair

`local_acs_transfer_target_families()` copies the generic declaration, reads
the exact national pre-clone family, requires its target to remain the
singleton `weekly_hours_worked_before_lsr`, and fails closed on missing,
drifted, or duplicate ownership. The staging driver resolves this plan once
and passes it unchanged through donor-consumption coverage, QRF transfer, and
post-transfer recipient coverage
(`tools/_legacy/build_us_acs_multispine_base.py:80-129,267-317,397-418`).

The repair adds no RNG site and authors no parallel imputation mechanism. The
shared transfer derives the family seed and each availability-pattern seed,
records QRF fit/draw evidence, and all three draw sites remain in the seed
ledger; the national family remains the authored spec owner
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1507-1627,3190-3204`;
`packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:615-684`;
`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:1021-1053`).

The fixture begins with raw ASEC hours columns, runs the real hours seeder,
clones the donor, proves the weekly leaf is identical on base-ASEC and
PUF-detail support roles, and then runs the real local QRF. It verifies the
actual `auto` selection resolves to the PUF-detail channel and that ACS weekly
hours are non-null, include positive values, are nonconstant, are not all 40,
and carry family/pattern seeds
(`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:540-676`;
`packages/microcosm-build/tests/test_us_acs_multispine.py:297-378`). Main-wiring
anti-rot proves donor coverage and transfer consume the same local plan
(`packages/microcosm-build/tests/test_us_acs_multispine_legacy_builder.py:166-206,209-416`).

## Signal-gate repair

The shared gate now accepts a validated requested-column tuple and an aligned
optional person mask while retaining its original all-output default. It
requires the weekly leaf in every requested set, scopes weights, uniqueness,
worked share, and worker mean consistently, and leaves the existing bands
unchanged at `(0.35, 0.62)` and `(30.0, 45.0)`
(`packages/microcosm-build/src/microcosm/build/us_runtime/hours_worked.py:81-122,267-421`).

The local wrapper requests the two pool-persisted leaves because
`weeks_worked` is formula-owned and projected away. It selects only
`acs_2024_1yr` persons so a plausible donor spine cannot mask a constant ACS
majority, and it fails closed when the person spine tag or ACS population is
missing. Finalize evaluates the wrapper on the consumer artifact at
`args.out_h5`, serializes its receipt, includes it in `hard_failures`, and
marks the summary not simulation-ready on failure
(`tools/build_us_acs_local_release.py:1196-1241,1507-1515,1537-1547,1628-1663`).

The regression reproduces the exact #765 signature: donor rows make the whole
combined frame pass while every ACS weekly-hours value is the historical 40
default. The ACS-scoped gate detects the constant leaf, local finalize exits
on `hours_worked_signal`, and the finalized summary records that gate as its
sole blocker. Because the implementation checks signal rather than a
particular default value, any future constant default fails the same way
(`packages/microcosm-build/tests/test_us_hours_worked.py:305-351`;
`packages/microcosm-build/tests/test_us_acs_local_release_tool.py:49-88,278-384`;
`packages/microcosm-build/src/microcosm/build/us_runtime/hours_worked.py:396-420`).

## Verification

- Required dependency preparation completed with
  `UV_CACHE_DIR=.uv-cache uv sync --all-packages --extra us`: 123 packages
  resolved and 100 installed distributions checked.
- `microcosm-calibrate`: 203 passed.
- `microcosm-data`: 318 passed / 2 skipped.
- `microcosm-fit`: 93 passed.
- `microcosm-frame`: 295 passed / 36 skipped.
- `microcosm-build` partition A: 4,161 passed / 36 skipped.
- `microcosm-build` partition B: 2,177 passed / 3 skipped.
- Accepted workspace total: **7,247 passed / 77 skipped / 0 failed**.
- The final affected four-module slice passed 60 tests after formatting.
- Measured peak RSS was 13.193 GiB for the heavier build partition and 6.753
  GiB for the remaining package shard, both below 15 GiB.
- Bundle freshness retained spec SHA
  `5378bb9189aec96f50da22aac71e5bd2c3d919e9795f6ef2147e0bc9c739dd8e`;
  coverage freshness retained 42,120/42,120 fields and 41/41 inventory checks.
- Ruff lint/format, the queue-command shell syntax check, and
  `git diff --check` pass.

No production build was used as verification: the binding host-ownership rule
reserved builds for the serial queue. The mandated `pgrep` was attempted, but
this managed sandbox denies process enumeration, so no build driver was
invoked at all.

## Serial-host handoff

The local verification must start only after candidate 25 publishes its sparse
completion marker and neither
`tools/build_us_multispine_pool` nor
`tools/build_us_fiscal_refresh_release` is running. It must use a fresh output
root and run both ACS staging and `--stage all`; old staging, checkpoints, and
consumer H5 cannot be reused because materialization records the staging SHA
and later stages re-verify it
(`tools/build_us_acs_local_release.py:771-898`).

The literal command at the end of `_LANE-NOTES.md` pins the candidate sparse
release identity, PUMA-ladder SHA, and consumer-facts-feed SHA; refuses dirty
or pre-existing state; sets both staging estimates to 15,000,000,000 bytes
(13.97 GiB); and creates a fresh local package. The serial queue must attach
its external process-tree monitor to enforce the binding <15-GiB RSS ceiling,
because the local release runner only records RSS samples
(`packages/microcosm-build/src/microcosm/build/us_runtime/base_pool.py:56-91`;
`tools/_legacy/build_us_acs_multispine_base.py:73-75,1158-1185`;
`tools/build_us_acs_local_release.py:2057-2080`).

## David-ready reply

The local lane was feeding the generic ACS transfer plan, which deliberately
omitted the runtime-owned `weekly_hours_worked_before_lsr`; native ACS rows do
not carry the ASEC source columns needed by the seeder, so consumer projection
silently materialized the then-current 40-hour engine default, while donor
rows let a combined-surface check look healthy and local finalize never called
the hours gate
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:307-357`;
`packages/microcosm-build/src/microcosm/build/us_runtime/hours_worked.py:104-175`;
`tools/build_us_acs_local_release.py:259-356,1096-1155`;
`packages/microcosm-build/tests/test_us_acs_local_release_tool.py:49-64`). The
fix grafts the exact national seed-ledgered `source_operator_hours_worked`
singleton into the local QRF plan, proves it through the actual PUF-detail
donor channel, and hard-gates the two persisted hours leaves on ACS persons
alone
(`tools/_legacy/build_us_acs_multispine_base.py:80-129,267-317`;
`tools/build_us_acs_local_release.py:1196-1241,1507-1663`). Rebuild fresh ACS
staging and the entire local release; do not reuse old staging, checkpoints,
or H5 because materialization pins the staging SHA
(`tools/build_us_acs_local_release.py:771-898`).

# Historical final report: stacked-pool to release CD-vintage provenance

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
