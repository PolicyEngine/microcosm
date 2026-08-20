# Battery package 3 progress

## State

Source-operator two-part calibration (13 checks), adult-care
post-reconciliation (2 checks), and unemployment-compensation targeted
calibration (1 check) are in diagnosis on `battery-pkg3-two-part` at
`164027e2`. No artifact-producing code or frozen battery configuration has
changed yet.

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

## Next

1. Record a 1% before artifact and extract all 16 assigned check values.
2. Add artifact-side calibrations and focused behavioral tests without changing
   any band, threshold, seed, fold, or comparator.
3. Commit each coherent green step and update this journal plus
   `_LANE-NOTES.md`.
4. Rebuild at 1%, report per-check before/after movement, run the complete test
   and lint suites, and write `FINAL_REPORT.md`.
