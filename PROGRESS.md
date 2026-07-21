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
- Located the existing public inverse-CDF implementation,
  `populace.frame.wquantile`, including its hand-computed uneven-weight tests at
  and above an exact cumulative boundary; the finalizer will reuse it.
- Mapped the finalizer ordering: raw tax-unit draws are normalized/snapped,
  reconciled, placed on tax units or people, sparsified, then signed-mass
  calibrated. The new bound belongs before this loop and is constrained to
  passthrough outputs, so it cannot interact with those later transforms.
- Mapped the publishable telemetry seam: optional finalizer diagnostics can be
  carried through both the monolithic imputation helper and checkpointed QRF
  finalization into the `qrf_finalization` stage metadata and final build
  summary, alongside the existing weights-audit record.
- Resolved targeted-test compatibility without weakening production drift
  checks: configuration keys are validated against the canonical production
  output universe, while a custom subset finalization applies only configured
  keys present in that subset. Requiring every global key in each custom subset
  would break the unchanged snapping, pruning, signed-calibration, educator,
  and checkpoint-equivalence fixtures.
- Verified the exact pinned local PUF exists at the requested path and SHA. Its
  actual positive tax-unit donor support has weighted inverse-CDF p99.9
  `211500.84797884867`, finite and positive and below the required `594483.0`
  ceiling; the design-block condition is not triggered.

## Next

- Add the weighted inverse-CDF helper, settled configuration, fail-loud entry
  validation, tax-unit clipping, and per-target diagnostics.
- Add the requested focused, regression, failure, and real-donor pin tests.
- Run Ruff, focused tests, and the full `packages/populace-build/tests` suite;
  commit every coherent step and keep the branch unpushed.
- Write the completed verification report to the designated output file.
