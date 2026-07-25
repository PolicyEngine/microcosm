# Progress

## State

Floor-aware SSI take-up prior implementation is committed on
`ssi-floor-aware-prior`, branched from `origin/main` at `0ac3422`. Focused SSI,
plan, and builder tests pass. The first full package run returned 1 because one
ordinary-miss test double omitted the new retry-applicability detail; that
fixture is corrected and its three parameterized cases now pass. A clean full
rerun returned `PYTEST_RC=0`. Final audit then found two mixed/boundary gate
classifications; their fixes and focused regressions pass, and one final full
rerun plus the report remain.

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
- Committed the corrected ordinary-miss fixture as `c6b5b55`.
- Reran the complete required package suite unpiped and captured
  `PYTEST_RC=0`.
- Ran both required Ruff gates on all six changed Python files:
  `RUFF_CHECK_RC=0` and `RUFF_FORMAT_RC=0`.
- The final independent audit caught two uncovered interactions: an exact
  reporter-only `target == floor == capacity` band was incorrectly classified
  as insufficient support, and a mixed structural/ordinary failure could
  advertise a retry incapable of clearing the run.
- Made triple equality an exact pass; made a run retryable only when it has
  ordinary delivery misses and no structural or invalid-diagnostics blocker;
  and made the builder's non-retry artifact wording truthful for structural,
  invalid, and mixed failures. The complete SSI test module plus focused
  builder regressions pass (`FOCUSED_EDGE_RC=0`).

## Next

- Commit the final-audit edge fixes and this journal update.
- Rerun `uv run pytest packages/populace-build/tests/ -q` after the audit
  fixes and capture its return code first, then repeat changed-file Ruff
  gates.
- Write `FINAL_REPORT.md`, finalize this journal, and commit without pushing.
