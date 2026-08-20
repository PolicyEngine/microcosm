# US reviewed-exclusion machinery lane notes

## Scope and invariants

- Branch/worktree: `battery-excl-machinery` in the assigned worktree only.
- Port reviewed-exclusion machinery, not an exclusion decision.
- The committed US register must remain empty; only the owner may approve an
  actual exclusion.
- Frozen bands, thresholds, seeds, folds, and comparators are immutable.
- Builds use at most the 1% sample with seed 578, remain off-chain, and never
  touch `logbook-pending-chain.txt`.
- Every mechanism statement added below will carry a source file and line
  citation.

## Evidence read

- `experiments/battery_burndown/ADJUDICATION.md` from the read-only
  `microcosm-arm-split` worktree.
- Its `adjudication.json`, filtered to the two
  `REVIEWED-EXCLUSION-CANDIDATE` rows and the reviewed-exclusion support
  record.

## Work log

- 2026-08-19: initialized the lane journals before implementation.
- 2026-08-19: the required uv sync is currently environment-blocked: the
  managed sandbox denies writes to the default cache, and the writable-cache
  retry cannot resolve PyPI DNS. No dependency or lockfile change was made.
