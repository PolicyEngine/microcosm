# Progress

## State

PR #583 round-1 HOLD remediation is in progress on
`multispine-pool-build-578`. The four blocking seams are the raw-stage producer
boundary, crash-safe pool publication, pandas string-read guard coverage, and
the deprecated-but-supported local ACS release producer.

## Done

- Confirmed the worktree starts clean at review head `a20e847`.
- Read `CLAUDE.md` and the full adversarial review at
  `_buildo-runtime/reviews/sol_583.log`.
- Selected the GitNexus debugging and impact-analysis workflows for lineage
  tracing and blast-radius checks.
- Attempted to build the GitNexus index; the managed lane denied its global
  registry write, so direct source/call-site/test tracing is the active
  fallback. Removed the generated untracked local cache.
- Extended the spine-blindness visitor over pandas string APIs and committed
  the exact reviewer `.get()` and `.query()` mutations plus `.eval()`,
  `.filter(items=...)`, and `.loc[...]` regressions in `884c255`. The focused
  guard suite passes (11 tests) and Ruff is clean.
- Reworked pool publication to invalidate first, stage H5 and diagnostics
  under one run ID, publish them atomically, and write the readiness manifest
  last in `b75f352`. The three reviewer interruption points now retain a
  non-ready tombstone; focused H5/pool coverage passes (17 tests).
- Restored the deprecated ACS staging CLI by delegating to the preserved
  pre-shim implementation in `tools/_legacy` in `f8bcb3b`. Its local-release
  recipe, summary/reviewed-null contract, helper compatibility, and legacy
  suite pass (37 tests); focused Ruff is clean.
- Split the pooled ASEC producer at a dedicated, operator-untouched raw-stage
  artifact; its exact `LKWEEKS`/`ED_VAL` mappings, artifact binding, resume
  repair, complete operator-family exclusion, and unchanged legacy checkpoint
  sequence are covered by regressions.
- Rewired the multispine pool to consume only that raw artifact, assemble and
  clone first, run the full 20-operator source-input chain on raw-evidenced
  rows, preserve ACS-native cells, and transfer the remaining nullable peer
  inputs with explicit ownership. Consolidated blocker coverage passes,
  including the legacy local-release path and structural guard.
- Closed the independent lineage-review findings: ACS native exceptions bind
  exact mapping contracts, source-kernel projections carry no false full-pool
  receipt, production wiring and the immutable 20-operator order are tested,
  and the pool-specific agreement registry covers every expanded transfer,
  take-up, SSI, and joint immigration surface.
- Bound the raw artifact's declared source-construction identity to the live
  structural identity, with a substituted-identity regression.
- Added a readiness loader that accepts only a green manifest whose publication
  run ID and digests match the H5 metadata and agreement diagnostics; the three
  interruption regressions now exercise that reader as well as the tombstone.
- Committed the reviewed assembly-first lineage and readiness integration in
  `5b930ab`.
- Ran the complete `populace-build` suite: 3,233 passed, 85 skipped, with only
  five pre-existing runtime/deprecation warnings. Repository-wide Ruff passes.
- Independent final adversarial review of committed `b421bfa` found no
  remaining blocker across the four round-1 HOLD items.
- Established the required final constraint: this live checkpoint will be
  restored byte-for-byte to `origin/main` before handoff so no root journal
  ships in the PR diff.

## Next

- Write the external handoff.
- Restore `PROGRESS.md` to `origin/main` and commit the cleanup.
