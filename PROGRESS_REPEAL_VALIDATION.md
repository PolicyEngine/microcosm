# Repeal-revenue validation progress

## State

- Active on branch `repeal-validation-298` in the dedicated
  `.claude/worktrees/populace-wt-530` worktree.
- Based on local `qbi-v2-engine` HEAD `807141e`; all work is offline.
- The existing validation machinery and release seam are mapped. The next
  coherent step is the declared benchmark resource and loader contracts.

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

## Next

- Add the provisional repeal benchmark resource and its schema/registry
  contracts.
- Add diagnostics-only runner output and synthetic arithmetic tests.
- Run Ruff, focused tests, and the full workspace `uv run pytest`.
- Write the final handoff report and record all verification results here.
