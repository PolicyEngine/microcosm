# Progress: local ACS hours transfer and signal gate (#765)

## State

The issue trace and all three claimed mechanism links have been verified in
source on branch `local-hours-transfer-765` at the post-#763 `origin/main`
base. The local staging transfer now grafts the exact national
`source_operator_hours_worked` singleton into the generic ACS plan, uses that
one extended plan for donor coverage, QRF transfer, and recipient coverage,
and fails closed if the national ownership surface drifts. The transfer step
is focused-, ownership-, spec-, and lint-green. No data build, release, push,
gate/threshold tuning, or production-artifact mutation has occurred.

The required US-extra dependency sync is complete from the exact lock. Direct
shell access to GitHub and host process enumeration are blocked by the managed
sandbox; both mandated commands were attempted, and the lane will invoke no
build driver under any circumstance. The task-scoped journal draft found in
the worktree has been reconciled to this session's observed receipts.

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
  declaration, rejects duplicate ownership, and rejects national target drift.
- Routed the extended plan through the staging driver's donor-consumption
  gate, transfer call, and post-transfer coverage audit. The existing ACS QRF
  implementation therefore supplies the already-ledgered family, pattern, and
  fit/draw seed evidence without adding an RNG site or changing the authored
  imputation bundle.
- Added local-plan anti-rot coverage and a real tiny-frame transfer fixture
  proving ACS rows receive non-default weekly hours plus family/pattern seed
  provenance. Receipts: 32 focused tests passed in 100.70 seconds; 104
  ownership/spec tests passed in 1,655.35 seconds; repository-wide Ruff and
  `git diff --check` passed.

## Next

1. Commit the green local-transfer implementation and test receipts.
2. Wire the existing hours gate into local finalize, add the exact defaulted-
   column regression, and commit with a green suite.
3. Finish `_LANE-NOTES.md` with the serial-queue build command and David-ready
   summary; write the final report to `FINAL_REPORT.md`.

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
