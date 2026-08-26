# Progress: historical scorer columns

## State

Final lane state as of 2026-08-26: implementation and complete workspace
verification are green. The real acceptance invocation cleared the original
incumbent formula-column defect and completed all incumbent scoring, then
stopped at a separate authenticated candidate-manifest failure:
late primary-QRF worker binding changed. No scorecard files were written.

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
- Ran the exact acceptance scorer from this worktree. It loaded the live
  incumbent, dropped and receipted one person column
  (has_marketplace_health_coverage), scored all five incumbent chunks and 12
  household slices per chunk, and released that state at 19.20 GiB peak RSS.
- Captured the complete subsequent traceback in out.md. Candidate
  authentication refused the pool manifest before its H5 loaded, so the run
  correctly stopped without bypassing the late-producer binding or emitting a
  partial JSON/Markdown scorecard.

## Next

1. Resolve or rebuild candidate-25 so its sealed primary-PUF-QRF worker binding
   matches the current authenticated late-producer contract.
2. Rerun the unchanged acceptance scorer command; do not weaken or bypass pool
   authentication.

## Historical prior lane

The stacked-pool CD-vintage provenance journal previously in this file is
historical. Its final state remains available at commit `19854a9f`.
