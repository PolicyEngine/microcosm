# Repeal-revenue validation progress

## State

- Active on branch `repeal-validation-298` in the dedicated
  `.claude/worktrees/populace-wt-530` worktree.
- Based on local `qbi-v2-engine` HEAD `807141e`; all work is offline.
- Implementation and required verification are complete. The supervisor owns
  source re-verification, dashboard coordination, push, and PR creation.

## Done

- Verified the requested worktree is clean and at the required local branch
  point.
- Created `repeal-validation-298` without fetching or touching another
  worktree.
- Read the GitNexus exploration workflow. Its MCP tools are unavailable in
  this sandbox, so source tracing uses repository-native searches.
- Traced the release path from
  `populace.build.us_runtime.reform_validation` through
  `tools/build_us_fiscal_refresh_release.py::_write_reform_validation` to the
  release's diagnostics-only `reform_validation.json` artifact.
- Confirmed ordinary neutralization arithmetic is
  `reform income_tax - baseline income_tax`, with one cached baseline and
  released transient simulations.
- Confirmed the new family must be separate from in-sample estimates: using
  calibration estimates would recreate the repeal-revenue blind spot.
- Identified the established June-surface neutralizations:
  `salt_deduction`, `interest_deduction`, `charitable_deduction`,
  `medical_expense_deduction`, `casualty_loss_deduction`,
  `qualified_business_income_deduction`, `tip_income_deduction`, and
  `overtime_income_deduction`.
- Verified all eight names exist in the installed pinned rules-engine registry
  at version 1.764.6.
- Added and declared `us/repeal_revenue_benchmarks.json` with eight component
  rows, the six provisional populace#298 transcriptions, and explicit null
  tips/overtime placeholders.
- Added strict resource loading through the existing exact-one reform shape,
  including schema version, required field, unique ID, scalar benchmark,
  provenance, provisional-label, period, and neutralization validation.
- Added shipped-resource and malformed-shape tests plus a live registry
  contract for all eight neutralized variables.
- Focused resource/loader/spec-only tests pass; the registry contract also
  passes separately against the pinned engine-enabled environment.
- Bumped the release validation payload to schema v2 and added the separate
  `repeal_revenue_benchmarks` table with `diagnostic_only`, independent
  `simulated` status, and ordered rows.
- Each row carries the reform definition, period/provisional label, benchmark
  and provenance, income-tax baseline/reform totals, modeled repeal delta, and
  signed relative gap `(modeled - benchmark) / benchmark`; null benchmarks
  retain modeled magnitudes and emit null gaps.
- Reused the existing lazy baseline cache and transient-simulation release
  lifecycle; all repeal rows bypass in-sample calibration estimates.
- Wired `repeal_revenue_benchmark_specs()` into the fiscal-refresh release's
  `_write_reform_validation()` H5-backed path.
- Added a synthetic-frame arithmetic/shape test, null-benchmark test, and a
  regression proving skipped repeal diagnostics never change the existing
  publish-guard flag.
- Focused reform-validation, publish-guard, spec-only, country-spec, and
  release-tool import tests pass; touched-file Ruff and diff checks are clean.
- Added `changelog.d/repeal-validation-298.added.md`.
- Tightened `budget_measure` and `description` schema handling so explicit
  null/non-string metadata is rejected instead of coerced; focused regression
  tests pass.
- Full touched-file Ruff format/check, JSON parsing, and `git diff --check`
  pass.
- Full workspace pytest passed against the supplied restricted PUF and pinned
  engine environment: 3,434 passed, 58 skipped, 6 warnings in 16m41s.
- Wrote `FINAL_REPORT_REPEAL_VALIDATION.md` with the release hook, exact
  schema-v2 payload shape, arithmetic, verification receipts, and all
  certification/scope caveats.

## Next

- Supervisor: verify the six provisional JCX-45-25 values and their component
  scope before certified use.
- Supervisor: confirm the external dashboard reads schema v2 and the additive
  repeal table.
- Supervisor: push `repeal-validation-298` and open the PR.
