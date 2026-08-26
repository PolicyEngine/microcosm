# Health-coverage label lane progress

## Round 2 state — 2026-08-25

- Resumed the clean `battery-health-coverage-labels` worktree at `7a66c5a9`;
  the round-1 mechanism repair and provenance commits are present and no push
  is authorized.
- The cold 1% host verification regressed the failing-gate count from 127 to
  128: three legs greened and four health-label legs were introduced or moved.
- The host-owned verification and baseline gate artifacts are read-only. No
  local data build, calibration, publication, threshold change, band change,
  seed change, or comparator exclusion is authorized.
- The GitNexus debugging skill is active, but its graph-query tools are not
  exposed in this session; round 1 also recorded a sandbox-blocked registry.
  This pass therefore uses direct source/call-site traces, checkpointed-frame
  receipts, and focused regression tests.
- Read-only checkpoint evidence identifies two distinct mechanisms: the IHS
  ACS mapper is exact but the pooled ASEC source silently treated missing
  2022/2023 health recodes as false; Marketplace moved when complete ESI/IHS
  latent model-draw steps were deleted from its chained QRF family, while the
  later direct-purchase and TRICARE draws also inherit the changed narrow
  Medicaid donor outcome.
- The generating repairs, regression pins, and deterministic evidence are
  implemented and the full permitted suite is green. No build has run; the
  corrected ASEC input receipts and existing host checkpoints are read-only
  evidence only.

## Round 2 done

- Reloaded `CLAUDE.md`, the GitNexus debugging workflow, the existing lane
  journal, and the round-1 progress/report history.
- Confirmed the worktree started clean on the requested branch with exactly
  the two round-1 commits ahead of its base.
- Identified the four required diagnoses: IHS overshoot, TRICARE movement away
  from band, newly red Marketplace, and moved non-Marketplace direct purchase.
- Matched the two 38,604-row clone-0 frames exactly by person and weight;
  measured every before/after category mass and every ACS boolean transition.
- Proved from target-bank metadata and the QRF completed-prefix code that
  stable target indexes did not preserve the family chain after native holes.
- Wrote `experiments/health_labels/ROUND2_DIAGNOSIS.md` and extended the IHS
  owner-adjudication equation without changing any comparator rule.
- Centralized the exact nine ASEC label/source mappings and made each source
  vintage fail closed unless the complete nonblank `{1, 2}` surface is present.
- Restored every declared target's latent model-draw step in each partial
  bounded ACS family, preserving complete native output cells and durable bank
  order.
- Bumped and regenerated the ACS transfer execution contract and invalidated
  target banks written under the old partial-family semantics.
- Added exact mapping/overlap, missing-vintage, native-preservation,
  peer-invariance, and cold/warm bank regressions; all focused checks pass.
- Verified the complete suite: build 5,979 passed/37 skipped (the 60
  subprocess-sensitive logbook and coverage-tool tests ran as an isolated
  green group); calibrate 201 passed; data 275 passed/1 skipped; fit 93 passed;
  frame 294 passed/36 skipped.
- Verified Ruff, `git diff --check`, the generated US bundle at
  `0f9ddf0bb16469459a5c87ec1667d74b9a1425d5b3d1753f80e26dba8f000e10`,
  and spec-engine coverage at 41,380/41,380 fields and 40/40 inventory checks.

## Round 2 next

- Commit the coherent generating-repair step with its updated journal.
- Write the completion report to `FINAL_REPORT.md`, commit it, and leave the
  clean branch ready for the next host-owned cold 1% run using the corrected
  ASEC source receipts.

## State

- Started 2026-08-21 on `battery-health-coverage-labels`.
- Repository instructions loaded.
- Required `uv sync --all-packages --extra us` was attempted twice. The global
  cache is sandbox-inaccessible; with a worktree-local `UV_CACHE_DIR`, uv built
  every local shard but could not download locked `huggingface-hub==1.18.0`
  because this environment has no DNS/network access. The pre-existing virtual
  environment is therefore being borrowed from the exact-lock
  `microcosm-f1` worktree with `uv run --no-sync` and this worktree's source on
  `PYTHONPATH`.
- The required pre-commit Ruff gate and all five package test shards are green:
  build 5,996; calibrate 201; data 275; fit 93; frame 294 (the data and frame
  shards include 1 and 36 skips, respectively).
- Three generating defects are repaired in the worktree: ACS `HINS1` now
  produces native ESI, ACS `HINS7` now produces native IHS, and the ASEC
  Medicaid leaf now uses narrow `NOW_CAID` instead of composite `NOW_MCAID`.
- The provenance matrix classifies seven detailed labels as concept mismatches:
  Marketplace, non-Marketplace direct purchase, Medicaid, other means-tested,
  TRICARE, CHAMPVA, and VA. No exclusion or comparator rule has been added.
- Source-of-truth adjudication identifies three health-label blocker checks:
  CHAMPVA, Indian Health Service, and other means-tested coverage. CHAMPVA is
  one leg of the label audit, not its goal.

## Done

- Read `CLAUDE.md`, the GitNexus debugging workflow, the arm-split
  adjudication, and the four-way synthesis.
- Enumerated all nine registered health-coverage labels and began their
  ASEC/ACS/clone-0 provenance audit.
- Established a green pre-change baseline for Ruff and every package shard.
- Added label-specific loader, mapping, strict-code, receipt, transfer-preservation,
  target-bank-hole, and Medicaid no-cross-talk regressions; focused tests pass.
- Drafted `experiments/health_labels/PROVENANCE.md` with all nine source paths
  and the exact owner-adjudication equations.
- The combined pre-commit gate is green: Ruff; build 5,973 passed/39 skipped;
  calibrate 201 passed; data 275 passed/1 skipped; fit 93 passed; frame 294
  passed/36 skipped. Peak observed build-shard RSS was about 11.6 GiB.
- Confirmed that process enumeration is denied by the managed sandbox. No data
  build will start until the required host-queue check can be satisfied.

## Next

- Commit the mechanisms, regressions, provenance table, and updated journal.
- Run an off-chain 1% build only when the host build queue and memory rules
  permit.
- Record evidence, failure-line diffs, and the final report.

## Historical handoff carried from `origin/main`

The following mortgage-lane handoff was current when written and is retained
as history only; it is not the state or next action of this branch.

### Historical state

Microcosm #516 whole-row donor outlier screen was complete on
`mortgage-donor-outlier-screen` (rebased onto `origin/main` after the #515
interim carve merged as #525). The `puf_tax_detail` donor dropped tax units
whose grouped raw mortgage interest reached $10M before the #515 carve
(pinned-artifact effect: 3,066 rows, weight 3,684 of ~161M, removing $2.947T
of phantom mortgage-interest mass), with the checkpoint schema bumped to v3
so post-carve pre-screen checkpoints rebuilt.

### Historical done

- Confirmed a clean starting worktree at `aef1c56`.
- Read the repository guidance and established the #515 donor carve as the
  screen's required downstream boundary.
- Started source-level audits of every donor-frame consumer, checkpoint
  validation, row-count pins, and existing donor-fact summaries.
- Attempted the requested GitNexus impact workflow; the managed filesystem
  denied its global registry write. Its local index also exposed a broad
  `build/` ignore mismatch, so the completed impact audit used direct source
  call sites and tests.
- Added `US_PUF_DONOR_MORTGAGE_OUTLIER_CEILING = 10_000_000.0` with the
  structural rationale and pinned-artifact receipts.
- Added a whole-row screen on grouped raw person `home_mortgage_interest`
  after tax-unit assembly, before the #515 carve, with retained-index reset.
- Confirmed no downstream consumer paired donor rows to the original HDF
  arrays or carried a stale donor-length vector; values and weights always
  originated from the same screened frame.
- Bumped the primary QRF checkpoint schema from v2 to v3 and made the stale
  checkpoint regression track the live constant while retaining literal-v1
  corruptions.
- Added regression coverage for the exact grouped boundary, whole-row removal,
  retained/carved $5M row, raw-$10.5M pre-carve ordering, and constant.
- Requested suites passed: PUF support/QRF 53; plan/gates 195; fiscal targets
  139; microcosm-data 138 with 1 skip. The directly affected tail-bound suite
  added 12 passes. Ruff format/check and `git diff --check` were clean.
- Wrote `SOL_516_REPORT.md` with the exact seam, consumer-by-consumer file:line
  audit, expected 208,611-row real-artifact effect, verification results, count
  sweep, and deliberately untouched surfaces.

### Historical next

- PR #527 review cycle, then merge. After both #525 and #527: rebuild the
  base/release; the mortgage critical-fit ratchet (0.20 -> 0.15) waited on a
  run that held per `us_critical_targets.py`.
- Root record-level ETL carve remained open on microcosm#515.
