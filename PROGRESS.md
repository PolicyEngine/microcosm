# Progress: stacked-pool to release CD-vintage provenance

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
