# Progress: authenticated gate-failed base-pool release

## State

Implementation is starting from `origin/main` at `2263df36`. The legacy
`--base-h5` release path and publication preflight are being traced so that a
multispine pool cannot bypass its terminal manifest, while an explicit
human-visible opt-in can carry a terminal `gate_failed` verdict into a release.

No pool/release build, publication, push, network access, battery change, or
gate-logic change has occurred.

## Done

- Read the repository operating instructions and the GitNexus debugging,
  impact-analysis, and CLI workflows.
- Confirmed branch `release-from-gate-failed-pool` is clean at `2263df36`.
- Checked the local GitNexus CLI; this worktree has no index, so source tracing
  will use offline repository search and direct code inspection.

## Next

1. Trace pool identity stamping, authenticated loader contracts, both legacy
   H5 consumers, release-manifest construction, and relevant tests.
2. Audit every `simulation_ready`, `gate_failed`, and
   `load_simulation_ready` consumer under `tools/` and `packages/`.
3. Implement and commit fail-closed containment, explicit opt-in and verdict
   carriage, then preflight authentication/surfacing and regression tests.
4. Add the changelog fragment, run Ruff and the full suite one process per
   shard, finalize `out.md` and this journal, and leave a clean worktree.

## Historical prior lane

The stacked-pool CD-vintage provenance journal previously in this file is
historical: that work is present in the `origin/main` base used by this lane.
Its final journal remains available at commit `2263df36^`.
