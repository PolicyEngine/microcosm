# Battery package 3 progress

## State

The artifact-side correction for all 16 assigned checks is implemented and the
PR test surface is green in memory-isolated shards. The carrier correction is
terminal full-sample reference-margin calibration, however, not the named
cross-fitted/held-out remedy; that adjudication-level gap is explicitly
unresolved because this branch lacks the required fold/comparator authority.
No frozen battery band, threshold, comparator, seed, or fold changed. The
uncontaminated 1% before artifact remains recorded at commit `5f5e5e91`; the
calibrated 1% after build and per-check measurement are the next step.

The required `uv sync --all-packages --extra us` was attempted first. The
default cache is sandbox-read-only; a retry with a writable cache reached PyPI
but DNS is unavailable. Verification therefore uses the already-synced
`microcosm-707` environment, whose `uv.lock` SHA-1 matches this worktree, with
`PYTHONPATH` pinned to this worktree's five package source directories.

## Done

- Read `CLAUDE.md`, the adjudication and all assigned machine-readable rows,
  citation registry entries C04-C08, and every cited source range.
- Built and measured the canonical off-chain 1% before artifact with sample
  seed 578, clone fraction 1, clone seed 578, no predecessor digest, and no
  pending-chain file mutation.
- Added a declared nine-target positive-leg policy. It preserves unemployment
  and disability carriers, matches the other assigned positive carrier
  margins, maps mutable positive amounts to exact reference support at the
  frozen five quantiles, and byte-preserves negative, negative-zero,
  zero-weight, and nonmutable values.
- Bound production application to exact ASEC clone-0 reference rows, ACS
  clone-0 recipients, and transferred-null mutable cells. Adult-care additions
  are limited to qualifying people and one candidate per empty tax unit;
  positive weeks additions require positive unemployment compensation.
- Reused the adult-care qualifying predicate for both calibration and the
  final reconciliation, which must be a verified no-op after calibration.
- Recorded and validated exact donor-support QRF regimes for every ordinary
  and banked ACS availability-pattern fit without changing fit seeds, folds,
  estimator counts, or draw behavior.
- Added schema-v2 calibration receipts with explicit terminal-versus-
  generation verification boundaries. Terminal validation independently
  replays live masks, row identities, output bytes, weights, carrier metrics,
  conditional quantiles, and coupled constraints; adversarial fully rehashed
  scope and diagnostic forgeries are rejected.
- Changed the two pinned SIPP chunk readers to streaming type inference. This
  leaves downstream explicit numeric coercion and locked output facts intact
  while reducing the 3.73 GB donor-test peaks from above the safety ceiling to
  0.49 GiB and 0.53 GiB.
- Verified all 225 `microcosm-build` test files green, split into fresh pytest
  processes where needed. Also verified `microcosm-fit` (93 passed),
  `microcosm-calibrate` (201 passed), `microcosm-frame` (294 passed, 36
  skipped), and `microcosm-data` (275 passed, one skipped).
- Ran repository-wide `ruff check .`, touched-file `ruff format --check`, and
  `git diff --check` successfully. Repository-wide `ruff format --check .`
  still identifies 49 pre-existing, mostly unrelated files; none was
  reformatted as part of this lane.
- Kept final verification processes below 10 GiB with a 10 GiB/20 ms guard.
  During diagnosis, the earlier 13.5 GiB/250 ms guard observed one rapid SIPP
  parser spike at 15.424 GiB before terminating it; the reader fix and all
  successful reruns remained far below the cap. This exception is retained in
  the journal rather than concealed.

## Next

1. Commit the coherent green implementation and journal state.
2. Rebuild off-chain at exactly 1% with sample/clone seed 578 under the
   tightened memory guard.
3. Extract all 16 after measurements and source-preservation invariants, then
   write and commit `_LANE-NOTES.md`, `PROGRESS.md`, and `FINAL_REPORT.md`.
