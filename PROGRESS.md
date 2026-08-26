# Progress: opt-in release from a gate-failed US pool

## State

Work began on 2026-08-26 from branch `release-from-gate-failed-pool` at
`2263df3673f5`. The lane is mapping the existing release-builder and
publication-preflight contracts before adding the explicit fail-closed opt-in.
No pool or release has been built, published, or pushed.

## Done

- Read the repository operating instructions and the GitNexus impact-analysis
  workflow.
- Confirmed the worktree is initially clean and the pre-built environment is
  available.
- Established this lane-specific committed journal before implementation.

## Next

1. Audit the loader seam, manifest schemas, preflight, tests, and every
   `simulation_ready`/`gate_failed`/`load_simulation_ready` consumer.
2. Add the explicit builder opt-in, carried agreement-gate verdict, preflight
   visibility, focused tests, and a changelog fragment without changing the
   battery or loader contracts.
3. Run Ruff and full pytest one process per shard, record evidence, write
   `out.md`, and leave the branch clean with coherent local commits.

## Historical context

The stacked-pool CD-vintage provenance journal previously occupying this file
was complete on `origin/main` before this lane began. Its detailed final state
remains available in the parent history at `2263df3673f5`. The earlier
PolicyEngine-US 1.819.0 lock-bump journal remains available at `05d254aa`.
