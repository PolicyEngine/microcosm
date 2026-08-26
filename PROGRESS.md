# Progress: historical scorer columns

## State

The scorer-legacy implementation and complete workspace verification are
green. Loaded entity H5s, legacy flat H5s, and authenticated pool H5s now share
one current-engine ownership normalization seam before scoring. The real
incumbent-versus-pool scorer is the remaining acceptance step.

The goal is to make head-to-head scoring deterministically remove and receipt
formula-owned columns from loaded historical incumbent or candidate H5
artifacts, while keeping the fresh-release leaf-only export gate strict.

## Done

- Read the repository operating instructions and confirmed the worktree starts
  clean on `scorer-legacy-incumbent-columns`.
- Confirmed the requested defect boundary: historical scorer inputs only; no
  pool build, release build, publishing, push, or validation bypass is in scope.
- Started this committed lane journal before implementation work.
- Reused the release gate's cached, period-sensitive
  `PolicyEngineUSVariableMetadataIndex` classification at the scorer loading
  boundary without changing the builder gate.
- Added fail-closed dependency-closure checks on each dropped output, including
  entity-correct leaf presence, before removing any artifact column.
- Added schema-3 JSON and Markdown receipts with deterministic total count and
  sorted column names per entity; clean artifacts seal an explicit empty
  receipt.
- Preserved tables, weights, strata, mass log, and frame metadata across the
  scorer-only normalization.
- Added the changelog fragment and three requested H5 tests: drop-and-score,
  missing-leaf refusal, and clean empty receipt. The full targeted scorer test
  file passes (16 passed). Its first explicit receipt run exposed and fixed an
  empty-entity pandas drop edge before full-shard verification.
- Passed repository-wide Ruff through the required `uv run --no-sync` command
  using a writable offline cache, and passed CI test-inventory verification for
  all 309 tracked test files.
- Passed all five full pytest shards in separate processes: build 6,516 passed /
  45 skipped; frame 295 / 36; calibrate 203 / 0; data 318 / 2; fit 93 / 0.
  Aggregate: 7,425 passed, 83 skipped, 0 failed.

## Next

1. Run the real incumbent-versus-pool acceptance scorer from this worktree.
2. Record exact evidence in `out.md`, finalize this journal, and leave a clean
   committed worktree.

## Historical prior lane

The stacked-pool CD-vintage provenance journal previously in this file is
historical. Its final state remains available at commit `19854a9f`.
