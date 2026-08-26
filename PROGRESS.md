# Progress: opt-in release from a gate-failed US pool

## State

Implementation is in correction after the source audit showed that the first
builder pass attached the opt-in to the already-fail-closed exact-k arm rather
than the legacy `--base-h5` arm. The shared H5 seam now identifies pool H5s,
binds sidecars to the requested bytes, and builds complete verdict carriage.
No pool or release has been built, published, or pushed.

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

## Next

1. Move the opt-in and carriage from exact-k to authenticated legacy
   `--base-h5`, add the closed-bypass regression, and keep exact-k strict.
2. Authenticate base pools in ordinary preflight and teach manifest-only review
   the generic `build.base_pool` carriage shape.
3. Run repository-wide Ruff and full pytest one process per shard.
4. Record the audit and verification evidence, write
   `out.md`, and leave the branch clean with coherent local commits.

## Historical context

The stacked-pool CD-vintage provenance journal previously occupying this file
was complete on `origin/main` before this lane began. Its detailed final state
remains available in the parent history at `2263df3673f5`. The earlier
PolicyEngine-US 1.819.0 lock-bump journal remains available at `05d254aa`.
