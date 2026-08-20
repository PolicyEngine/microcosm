# Battery package 3 progress

## State

Source-operator two-part calibration (13 checks), adult-care
post-reconciliation (2 checks), and unemployment-compensation targeted
calibration (1 check) have an uncontaminated 1% before artifact on
`battery-pkg3-two-part` at `164027e2`. No artifact-producing code or frozen
battery configuration has changed yet.

The required `uv sync --all-packages --extra us` was attempted first. The
default cache is sandbox-read-only; a retry with a writable cache reached PyPI
but DNS is unavailable. Tests therefore use the already-synced
`microcosm-707` virtual environment, whose `uv.lock` SHA-1 exactly matches this
worktree, with `PYTHONPATH` pinned to this worktree's five package source
directories.

## Done

- Read `CLAUDE.md`, the battery adjudication, all 16 assigned machine-readable
  rows, citation registry entries C04-C08, and every cited source range.
- Confirmed the frozen comparator separately tests weighted sign incidence and
  conditional carrier quantiles.
- Confirmed direct transfer draws are produced per availability-pattern QRF,
  and adult-care reconciliation occurs after transfer.
- Started parallel source-operator, adult-care, and unemployment-compensation
  implementation analyses.
- Passed the four deterministic contract tests covering sign-separated battery
  behavior, adult-care structural reconciliation, and the declared transfer
  surface for this journal checkpoint.
- Preserved the mandated off-chain build contract: 1% sample, sample seed 578,
  no previous-row digest, and no pending-chain file changes.
- Completed the before build with the canonical 100-tree fits under a hard
  13.5 GiB process guard; peak observed per-process RSS was 10.733 GiB. The
  expected frozen terminal battery failed and wrote its non-ready diagnostics.
- Extracted all 16 assigned before measurements. At 1%, adult-care and
  workers'-compensation conditional quantiles are below the battery's five-row
  support floor; their manual five-quantile diagnostics are recorded alongside
  the honest unsupported gate status in `_LANE-NOTES.md`.
- Completed the implementation design: deterministic nearest-weight carrier
  mass, five-grid donor inverse-ECDF amounts, exact ACS clone-0 mutable scope,
  frozen unemployment/disability carriers, byte-preserved negative
  self-employment values, adult-care structural qualification, and
  unemployment-compensation-limited weeks capacity.

## Next

1. Add artifact-side calibrations and focused behavioral tests without changing
   any band, threshold, seed, fold, or comparator.
2. Commit each coherent green step and update this journal plus
   `_LANE-NOTES.md`.
3. Rebuild at 1%, report per-check before/after movement, run the complete test
   and lint suites, and write `FINAL_REPORT.md`.
