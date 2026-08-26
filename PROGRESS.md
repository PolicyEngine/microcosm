# Progress: local ACS hours transfer and signal gate (#765)

## State

Implementation and repository verification are complete on
`local-hours-transfer-765`. The local staging builder now grafts the exact
national `person/source_operator_hours_worked` singleton into its ACS QRF plan
and routes one plan through donor coverage, transfer, and recipient coverage
(`tools/_legacy/build_us_acs_multispine_base.py:80-129,267-317,397-418`).
Local finalize now applies the unchanged shared hours plausibility bands to
the two persisted hours leaves on ACS-spine persons only and makes failure a
hard simulation-readiness blocker
(`packages/microcosm-build/src/microcosm/build/us_runtime/hours_worked.py:81-122,267-421`;
`tools/build_us_acs_local_release.py:1196-1241,1507-1515,1537-1547,1628-1663`).

The accepted workspace result is 7,247 passed / 77 skipped / 0 failed. Bundle
and coverage freshness, Ruff, and `git diff --check` pass. No data build,
release, push, gate/threshold tuning, or production-artifact mutation has
occurred. The required fresh local verification remains queued for the serial
host and its process-tree RSS monitor; its exact command is the literal end of
`_LANE-NOTES.md`.

## Done

- Read the repository and GitNexus debugging instructions.
- Attempted `uv sync --all-packages --extra us` first. The default user cache
  is sandbox-inaccessible; the exact retry with `UV_CACHE_DIR=.uv-cache`
  resolved 123 packages and checked 100 installed distributions. Verified
  PolicyEngine-US 1.819.0, PolicyEngine-Core 3.31.0, and this worktree's
  editable `microcosm-build` install.
- Attempted `gh issue view 765 --repo PolicyEngine/microcosm --comments`; shell
  network access is unavailable. The indexed read-only GitHub surface returned
  the issue body (no comments), which matches the supplied David Trimmer trace.
- Attempted the required `pgrep` check; process enumeration is denied by the
  sandbox. No build command has been or will be started in this lane.
- Read the GitNexus debugging skill; this session exposes no GitNexus graph
  tools, so investigation will use direct source tracing and tests.
- Reconciled the worktree journals byte-for-byte with salvage commit
  `1b9ec45c`; the salvage contains no source implementation beyond these two
  journal files.
- Verified the national #578 mechanism, the local transfer-plan omission, the
  raw-ASEC seeding incompatibility, the reviewed-null default materialization,
  all three existing transfer RNG ledger sites, and the missing local finalize
  gate directly in source. Exact citations are recorded in `_LANE-NOTES.md`.
- Discarded one invalid calibration receipt after its traceback exposed a
  cloned `pytest` script whose shebang pointed at the sibling
  `microcosm-one-surface` environment. Current-worktree verification uses
  `.venv/bin/python -m pytest`; all editable module paths and `.pth` files were
  independently checked to point at this worktree.
- Current-worktree baseline receipts: calibrate 203 passed; data 318 passed /
  2 skipped; fit 93 passed; frame 295 passed / 36 skipped. The first build
  partition passed 4,156 tests / skipped 36. The complementary partition
  passed 2,172 / skipped 3 while five child-process tests timed out under the
  occupied-host load; all five passed together in a fresh isolated rerun in
  403.92 seconds. Thus every collected build test passed before source edits.
- Extended the local ACS staging plan with the national pre-clone
  `person/source_operator_hours_worked` family. It owns exactly
  `weekly_hours_worked_before_lsr`, copies rather than mutates the generic ACS
  declaration, rejects duplicate ownership, and rejects national target drift
  (`tools/_legacy/build_us_acs_multispine_base.py:80-129`).
- Routed the extended plan through the staging driver's donor-consumption
  gate, transfer call, and post-transfer coverage audit. The existing ACS QRF
  implementation therefore supplies the already-ledgered family, pattern, and
  fit/draw seed evidence without adding an RNG site or changing the authored
  imputation bundle
  (`tools/_legacy/build_us_acs_multispine_base.py:267-317,397-418`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1507-1627,3190-3204`;
  `packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:615-684`).
- Added local-plan anti-rot coverage and a real tiny-frame transfer fixture
  proving ASEC raw-hour seeding survives the PUF support clone selected by the
  local `auto` channel and ACS rows receive non-null, positive, nonconstant,
  non-default weekly hours plus family/pattern seed provenance
  (`packages/microcosm-build/tests/test_us_acs_multispine.py:297-378`).
- Extended the shared hours summary/gate API with a validated requested-column
  set and an aligned person mask while retaining its original default behavior
  and bands. The local wrapper selects ACS persons and the two pool-persisted
  leaves because `weeks_worked` is formula-owned and projected away
  (`packages/microcosm-build/src/microcosm/build/us_runtime/hours_worked.py:81-122,267-421`;
  `tools/build_us_acs_local_release.py:1196-1241`).
- Wired the ACS-scoped result into the final gate report and hard-failure list
  after loading the consumer H5, so donor rows cannot conceal a defaulted ACS
  column and a failure marks the release not simulation-ready
  (`tools/build_us_acs_local_release.py:1507-1515,1537-1547,1628-1663`).
- Added the exact silent-failure regression: the combined donor-heavy surface
  passes, the ACS-only constant-40 surface fails, and local finalize writes
  `hours_worked_signal` as its hard blocker
  (`packages/microcosm-build/tests/test_us_hours_worked.py:305-351`;
  `packages/microcosm-build/tests/test_us_acs_local_release_tool.py:49-88,278-384`).
- Post-edit verification passed: calibrate 203; data 318 / 2 skipped; fit 93;
  frame 295 / 36 skipped; build partitions 4,161 / 36 skipped and 2,177 / 3
  skipped. The accepted total is **7,247 passed / 77 skipped / 0 failed**. The
  affected four-module slice passed 60 tests after final formatting. The
  heavier partition's measured child-process peak was 13.193 GiB and the
  remaining package shard peak was 6.753 GiB, both below 15 GiB.
- Regenerated and checked the US bundle at spec SHA
  `5378bb9189aec96f50da22aac71e5bd2c3d919e9795f6ef2147e0bc9c739dd8e`;
  coverage remains exact at 42,120/42,120 fields and 41/41 inventory checks.
  The transfer reuses the authored national family and existing ledgered QRF
  draw sites, so no spec or generated surface changed
  (`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:1021-1053`;
  `packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:615-684`).

## Next

1. Run the exact fresh staging plus local-release command at the end of
   `_LANE-NOTES.md` only after the current candidate relinquishes the serial
   host, with the queue's external process-tree RSS monitor enforcing <15 GiB.
2. Reply to David with the one-paragraph handoff immediately above that
   command; do not reuse old local staging, checkpoints, or consumer H5.

# Historical progress: stacked-pool to release CD-vintage provenance

## State

Implementation and validation are complete. The stacked
producer now authenticates and applies household geography after source
assembly, carries its authority through checkpoint and publication identities,
publishes verified CD-vintage H5 attributes, and reaches the unchanged release
guard through the shared fixed/table-aware reader
(`tools/build_us_multispine_pool.py:814-928,1300-1469,1857-1930,5290-5334`;
`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:1032-1193,1463-1699`;
`tools/build_us_fiscal_refresh_release.py:2565-2677`).

No pool/release build, push, guard weakening, operator-boundary weakening, or
`logbook-pending-chain.txt` access has occurred.

## Done

- Reconciled the worktree with salvage commit `ca26ea21`: its tracked tree was
  already exactly present, so the best salvage was retained and audited rather
  than reimplemented.
- Completed the required all-package US environment sync from the exact lock
  using the writable offline uv cache after the managed sandbox refused the
  default cache.
- Added the two required pinned geography authority pairs, the ledgered seeded
  PUMA-overlap assignment, schema-12 checkpoint identity, assignment receipts,
  schema-9 terminal manifest validation, and schema-3 H5 materializer binding.
- Added atomic nullable-H5 root-attribute write/verification and authenticated
  manifest-to-H5 geography/clone-lineage validation.
- Made release preflight read root attributes and fixed household frames in one
  `HDFStore` handle while retaining all existing SHA, target-vintage, and
  positive-support checks.
- Added a real tiny stacked-pool publication to release-preflight integration
  test plus negative attr, digest, missing-lineage, and divergent-clone tests
  (`packages/microcosm-build/tests/test_us_multispine_pool_tool.py:2279-2335`;
  `packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py:1357-1440`).
- Regenerated and checked the US anti-rot chain: spec SHA
  `5378bb9189aec96f50da22aac71e5bd2c3d919e9795f6ef2147e0bc9c739dd8e`,
  42,120/42,120 configuration fields, 49 claims, and 41/41 inventory checks.
- Passed the complete accepted workspace suite: 7,241 passed, 77 skipped, and
  0 failed. The two memory-bounded build partitions were 4,155 passed / 36
  skipped and 2,177 passed / 3 skipped; calibrate, data, fit, and frame account
  for the remaining 909 passed / 38 skipped.
- Passed repository-wide Ruff, bundle freshness, coverage freshness, smoke
  script syntax, and `git diff --check`.
- Recorded the exact candidate Stage-1 path/SHA additions and the checkpoint
  invalidation verdict in `_LANE-NOTES.md` and `FINAL_REPORT.md`.

## Next

1. Candidate Stage 1 adds both authenticated geography authority pairs and
   rebuilds under checkpoint materializer 12.
2. Candidate Stage 2 consumes the new schema-9 manifest/materializer-3 H5 and
   passes the unchanged release preflight.
3. Do not reuse pre-fix Stage-1 checkpoints or pool publications; immutable
   source artifacts remain reusable.

## Historical prior lane

The PolicyEngine-US 1.819.0 lock-bump journal previously in this file is
historical: that lane merged into `origin/main` at `7b90bb18` on 2026-08-24.
Its final state remains available at commit `05d254aa` and its detailed
receipts remain in the historical section of `_LANE-NOTES.md`.
