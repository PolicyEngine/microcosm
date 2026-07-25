# Progress

## State

Floor-aware SSI take-up prior implementation is committed on
`ssi-floor-aware-prior`, branched from `origin/main` at `0ac3422`. Focused SSI,
plan, and builder tests pass. The first full package run returned 1 because one
ordinary-miss test double omitted the new retry-applicability detail; that
fixture is corrected and its three parameterized cases now pass. A clean full
rerun and the final report remain.

## Done

- Confirmed the designated worktree was clean and did not touch the active
  release-build worktree.
- Read the repository guidance.
- Created `ssi-floor-aware-prior` directly from `origin/main`.
- Audited all `_band_prior` consumers, SSI artifact pins, the
  prior-weight-basis flag path, and seeded stable-draw semantics. GitNexus was
  unavailable without creating an index, so the audit used direct call sites.
- Replaced the floor-blind `target / capacity` law with
  `(target - floor) / (capacity - floor)` for the feasible unsaturated regime.
- Preserved the one-shot source-keyed Bernoulli draw and saturation fallback.
- Added explicit assignment-basis/current-weight prior statuses, drawable
  capacities, and empty-band diagnostics; bumped the SSI artifact schema to 4
  while retaining schema 2/3 basis compatibility.
- Made enforced reporter-floor excess and no-drawable-support conditions fail
  the delivery gate with support-specific guidance.
- Updated the runtime/source/take-up contracts, CLI schema help, builder
  fixtures, and the existing tests that encoded the old expectation.
- Added exact attempt-5 arithmetic and seeded overshoot regressions, every
  requested edge-case test, and fixed-seed monotone-selection coverage.
- Resolved independent diff-review findings: tightened exact numeric
  assertions, chained artifact loading into floor-aware assignment, made the
  no-drawable prior reporter-only rather than Bernoulli(1), made all saturated
  enforced capacity shortages explicit support failures, distinguished
  retryable weight drift from structural failures, and hardened malformed
  empty-band count validation.
- Focused SSI + plan tests pass; focused builder SSI tests pass. Changed-file
  Ruff check and format check are clean.
- Committed the coherent implementation as `8c14bed`.
- Ran the required full package suite without piping and captured
  `PYTEST_RC=1`; its only failures were the three modes of
  `test_main_writes_diagnostics_before_post_calibration_gate_failure`.
- Classified that fixture's deliberately ordinary 18–64 delivery miss as
  prior-weight-basis retryable. All three focused modes pass (`FOCUSED_RC=0`);
  production behavior is unchanged.

## Next

- Commit the corrected ordinary-miss fixture and this journal update.
- Rerun `uv run pytest packages/populace-build/tests/ -q` and capture its
  return code first, then repeat changed-file Ruff gates.
- Write `FINAL_REPORT.md`, finalize this journal, and commit without pushing.
