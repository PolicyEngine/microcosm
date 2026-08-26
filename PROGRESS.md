# Progress: opt-in release from a gate-failed US pool

## State

Containment, carriage, and both preflight modes are now corrected and
focused-tested. Ordinary preflight authenticates identified base pools through
the same strict/explicit-opt-in seam; manifest-only publication review reads
the canonical `build.base_pool` receipt and keeps the red evidence outside its
automated exit calculation. No pool or release has been built, published, or
pushed.

## Done

- Read the repository operating instructions and the GitNexus impact-analysis
  workflow.
- Confirmed the worktree is initially clean and the pre-built environment is
  available.
- Established this lane-specific committed journal before implementation.
- Audited all readiness/status consumers under `tools/` and `packages/`; the
  config-driven exact-k launcher remains intentionally green-only and the
  pool producer still declares that calibration requires readiness.
- The initial builder pass and tests exposed the flag under exact-k; this is
  committed history but is not the final contract and is being corrected.
- Added manifest-only publication review, a prominent red-battery/human-review
  banner, complete machine-readable carriage, and mixed-mode coverage proving
  the carried verdict does not alter an existing preflight exit code.
- Added the changelog fragment. All 25 preflight tests and focused Ruff pass.
- Added and focused-tested the shared pool classifier, sidecar/H5 path binding,
  and self-contained red/green receipt builder in `h5_io.py`.
- Moved the opt-in to the legacy base-H5 arm, restored exact-k's strict loader
  and receipt, carried `base_pool` into both manifests, and added regression
  coverage for missing manifests, red/no-flag refusal, flag-on routing, and
  release-manifest visibility.
- Applied identical base-pool authentication to ordinary preflight, added the
  explicit flag there, surfaced authenticated and carried red receipts in both
  human and JSON output, and passed all 30 focused preflight tests.
- Corrected the changelog fragment to describe the closed legacy bypass rather
  than the unchanged exact-k arm.
- Documented the authenticated base-pool and human publication boundary in the
  repository release-flow guide.

## Next

1. Run the complete affected test files and adjacent release-contract tests.
2. Run repository-wide Ruff and full pytest one process per shard.
3. Record the audit and verification evidence, write
   `out.md`, and leave the branch clean with coherent local commits.

## Historical context

The stacked-pool CD-vintage provenance journal previously occupying this file
was complete on `origin/main` before this lane began. Its detailed final state
remains available in the parent history at `2263df3673f5`. The earlier
PolicyEngine-US 1.819.0 lock-bump journal remains available at `05d254aa`.
