# Progress

## State

Populace #462 fix 3b is complete on `qrf-tail-bound-462`, starting from the clean
`origin/main` commit that includes #477, #478, #479, and #480. The settled
per-target weighted-donor-quantile clip is implemented at the PUF tax-detail
finalizer seam, telemetry is published, focused and full-suite tests are green,
and the branch remains unpushed. No manifest, `populace-fit`, or unrelated
pipeline stage was changed.

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
- Resolved targeted-test compatibility without weakening the production path:
  the module configuration is validated against the canonical output universe
  and activates whenever its target is present; deliberately reduced chains
  disjoint from configured targets remain isolated. Explicit test mappings are
  validated against their invocation's exact surface. This preserves the
  snapping, pruning, signed-calibration, educator, and checkpoint-equivalence
  behavior fixtures.
- Verified the exact pinned local PUF exists at the requested path and SHA. Its
  actual positive tax-unit donor support has weighted inverse-CDF p99.9
  `211500.84797884867`, finite and positive and below the required `594483.0`
  ceiling; the design-block condition is not triggered.
- Added the `non_sch_d_capital_gains` p99.9 configuration, positive-support
  weighted-donor quantile wrapper, atomic entry validation, strict upper clip
  before person allocation, and JSON-native per-target diagnostics.
- Required an active bound to have a diagnostics sink, then carried those
  records through both monolithic and checkpointed finalization into
  `qrf_finalization` metadata and the final build summary; caps cannot be
  silent.
- Added focused tests for inverse-CDF boundaries, exact clipping/count/bit
  behavior, unaffected outputs, every fail-loud case, atomic validation,
  telemetry serialization, and the real donor pin.
- Ran Ruff on all touched implementation/test files. The new tail-bound file,
  unchanged PUF-support behavior file, QRF checkpoint-chain file, and base
  builder telemetry file are green.
- Ran the full build test suite once. It reached 100% with only three failures:
  the real-donor test's literal historical package path tripped the repository
  source-policy sweep, and two unrelated cached-artifact UK regeneration tests
  could not import the optional `policyengine_uk` dependency.
- Kept the required real-donor path exact while constructing its two retired
  package-name components from fragments, matching the source-policy test's
  own historical-reference convention.
- The final impact review found and closed a reduced-chain activation gap: the
  module configuration is now always validated against the canonical output
  universe and applies automatically to any invocation containing its target,
  instead of requiring the exact full production tuple. Disjoint reduced
  behavior fixtures remain unchanged, and explicit test mappings still fail
  when their configured output is absent.
- Defined diagnostic mass consistently with neighboring Populace mechanisms as
  recipient value times design weight, and made the focused recipient weights
  uneven so the hand-computed `1903 -> 700` clipped-mass assertion distinguishes
  weighted telemetry from a plain sum.
- Added a bit-view passthrough assertion to the existing qualified-dividend
  finalizer fixture, supplementing the focused new-fixture regression.
- Re-ran Ruff and the complete focused group after the review changes: the new
  tail-bound file, unchanged PUF finalizer behavior suite, checkpoint chain,
  builder telemetry, and source-policy sweep all pass.
- Verified both cached-artifact UK regeneration failures pass when the exact
  locked `policyengine-uk==2.89.0` wheel already in the local read-only uv cache
  is exposed on `PYTHONPATH`. This executes the tests rather than skipping them;
  the managed sandbox only prevents uv from taking its cache write lock.
- Ran the complete `packages/populace-build/tests` suite with that exact locked
  wheel exposed: pytest reached 100% and exited 0. The two pre-existing runtime
  warnings and macOS temporary-directory cleanup warnings were non-failing.
- Re-ran `ruff check --fix` and `ruff format` on every touched Python file; all
  checks pass and formatting is unchanged.
- Audited `origin/main...HEAD`: changes are limited to the finalizer and its QRF
  caller, finalization telemetry in the base builder, focused/existing tests,
  and the two requested progress/report documents. `git diff --check` passes.
- Wrote the completed handoff to `FINAL_REPORT.md`. No push was performed.

## Next

- No implementation work remains. The local, unpushed branch is ready for
  review.
