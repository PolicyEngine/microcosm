# Progress

## State

Populace #462 fix 3b is in progress on `qrf-tail-bound-462`, starting from the
clean `origin/main` commit that includes #477, #478, #479, and #480. The settled
design is a per-target weighted-donor-quantile clip at the PUF tax-detail
finalizer seam; no manifest, `populace-fit`, or other pipeline stage changes are
in scope.

## Done

- Confirmed the worktree is on `qrf-tail-bound-462`, clean, and exactly at
  `origin/main` (`3b17aaf`).
- Read the required GitNexus exploration and impact-analysis workflows.
- Confirmed the GitNexus MCP index/tools are unavailable in this workspace;
  repository-native symbol and call-site inspection will provide the fallback
  impact analysis.
- Preserved the prior fix-3a history in Git; this file now tracks the new fix-3b
  work from its starting commit.

## Next

- Inspect `puf_support.py`, its finalizer telemetry pattern, all neighboring
  snapping/pruning/calibration sets, and existing finalizer fixtures.
- Add the weighted inverse-CDF helper, settled configuration, fail-loud entry
  validation, tax-unit clipping, and per-target diagnostics.
- Add the requested focused, regression, failure, and real-donor pin tests.
- Run Ruff, focused tests, and the full `packages/populace-build/tests` suite;
  commit every coherent step and keep the branch unpushed.
- Write the completed verification report to the designated output file.
