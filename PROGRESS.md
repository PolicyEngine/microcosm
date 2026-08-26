# Progress: opt-in release from a gate-failed US pool

## State

Implementation and focused verification are complete. Default pool loading
remains on the strict simulation-ready loader; only
`--allow-gate-failed-base-pool` selects the existing authenticated scoring
loader. Publication preflight now renders the carried red battery separately
from its unchanged automated checks/exit code. No pool or release has been
built, published, or pushed.

## Done

- Read the repository operating instructions and the GitNexus impact-analysis
  workflow.
- Confirmed the worktree is initially clean and the pre-built environment is
  available.
- Established this lane-specific committed journal before implementation.
- Audited all readiness/status consumers under `tools/` and `packages/`; the
  config-driven exact-k launcher remains intentionally green-only and the
  pool producer still declares that calibration requires readiness.
- Added builder parser/loader routing, red-verdict validation, and conditional
  carriage under the existing exact-k receipt without changing either H5
  loader contract or the default green receipt shape.
- Added focused tests for explicit flag parsing, strict default routing and
  refusal, alternate-loader routing, and release-manifest carriage; focused
  Ruff and the first three tests pass.
- Added manifest-only publication review, a prominent red-battery/human-review
  banner, complete machine-readable carriage, and mixed-mode coverage proving
  the carried verdict does not alter an existing preflight exit code.
- Added the changelog fragment. All 25 preflight tests and focused Ruff pass.

## Next

1. Run the complete builder test file and adjacent manifest/loader/contract
   tests; fix and commit any findings.
2. Run repository-wide Ruff and full pytest one process per shard.
3. Record the audit and verification evidence, write
   `out.md`, and leave the branch clean with coherent local commits.

## Historical context

The stacked-pool CD-vintage provenance journal previously occupying this file
was complete on `origin/main` before this lane began. Its detailed final state
remains available in the parent history at `2263df3673f5`. The earlier
PolicyEngine-US 1.819.0 lock-bump journal remains available at `05d254aa`.
