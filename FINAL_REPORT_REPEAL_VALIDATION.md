# Repeal-revenue validation final report

## Status

Complete on local branch `repeal-validation-298`, based on local
`qbi-v2-engine` HEAD `807141e`. No network operation, push, or PR action was
performed.

## What landed

- Added the declared, pure-JSON country resource
  `packages/populace-build/src/populace/build/us/repeal_revenue_benchmarks.json`
  and listed it in `us/country_package.json`.
- Added eight FY2026 component rows:
  - SALT: $59.5 billion.
  - Home mortgage interest: $53.0 billion.
  - Charitable contributions: $81.5 billion.
  - Medical expenses: $13.8 billion.
  - Casualty losses: $0.2 billion.
  - Section 199A QBI: $76.4 billion.
  - Tips exemption: null benchmark placeholder.
  - Overtime exemption: null benchmark placeholder.
- Marked every row `provisional: true`. The six numeric rows carry the exact
  required source string:
  `JCT JCX-45-25, FY2026 individuals line — provisional, transcribed from
  populace#298; re-verify against the publication before first certified use`.
- Added strict resource loading through `repeal_revenue_benchmark_specs()`.
  It checks schema version, required fields, unique IDs, period and metadata
  types, finite positive or null benchmarks, source/provisional fields, and
  exactly one existing reform shape (`neutralized_variable` or
  `parameter_changes`).
- Added schema, malformed-contract, synthetic arithmetic, null-benchmark,
  diagnostics-only, and pinned variable-registry tests.
- Added `changelog.d/repeal-validation-298.added.md`.

## Release runner hook

The existing US fiscal-refresh release path now loads the standing family in
`tools/build_us_fiscal_refresh_release.py::_write_reform_validation()`:

1. `load_default_reform_specs(period=PERIOD)` loads the existing families.
2. `repeal_revenue_benchmark_specs(period=PERIOD)` loads the new family.
3. `default_simulate_factory(dataset_path)` runs against the freshly exported,
   calibrated release H5.
4. `reform_validation_payload(..., repeal_benchmarks=...)` independently
   simulates every repeal and writes the table into `reform_validation.json`.

The new rows share the existing lazy current-law baseline by period and budget
measure. Every repeal gets its own transient simulation, which passes through
the existing explicit engine-release and garbage-collection lifecycle. Repeal
rows never consult `in_sample_estimates`, even if the same component has a
calibration amount target.

This family is diagnostics-only. It defines no tolerance, pass/fail value, or
gate. Its `simulated` status is separate from the pre-existing
`out_of_sample_simulated` publication-guard field, so missing repeal diagnostics
cannot block publication.

## Exact payload shape

`REFORM_VALIDATION_SCHEMA_VERSION` is now 2. Existing `reforms` rows remain
unchanged; the payload adds this top-level table:

```json
{
  "schema_version": 2,
  "baseline_period": 2024,
  "scoring_window": "see per-row reform or benchmark window",
  "out_of_sample_simulated": true,
  "reforms": [],
  "repeal_revenue_benchmarks": {
    "diagnostic_only": true,
    "simulated": true,
    "rows": [
      {
        "id": "repeal_salt",
        "name": "State and local tax deduction (SALT)",
        "period": 2026,
        "provisional": true,
        "description": "...",
        "neutralized_variable": "salt_deduction",
        "benchmark": 59500000000.0,
        "benchmark_source": "JCT JCX-45-25, FY2026 individuals line — provisional, transcribed from populace#298; re-verify against the publication before first certified use",
        "benchmark_window": "FY2026",
        "budget_measure": "income_tax",
        "baseline_total": 0.0,
        "reform_total": 0.0,
        "modeled_revenue_delta": 0.0,
        "relative_gap": -1.0
      }
    ]
  }
}
```

The numeric zeros above illustrate types, not release results. For each actual
row:

- `modeled_revenue_delta = reform_total - baseline_total`.
- `relative_gap = (modeled_revenue_delta - benchmark) / benchmark`.
- Positive relative gaps mean modeled repeal revenue exceeds the benchmark;
  negative gaps mean a shortfall.
- A null benchmark still gets simulated and publishes its modeled magnitude,
  but `relative_gap` is null.
- If repeal simulation is explicitly skipped, the table has
  `simulated: false` and its modeled fields are null.
- A parameter-defined future row would carry `parameter_changes` instead of
  `neutralized_variable`; the two shapes are mutually exclusive.

## Verification

- Touched Python files: Ruff format check and Ruff check pass.
- New and modified JSON resources parse successfully.
- Focused reform-validation, country-package, country-spec, publish-guard, and
  release-tool import tests pass.
- The live registry contract passes against pinned PolicyEngine-US 1.764.6 for
  all eight neutralized variables.
- Full workspace command passed with the restricted PUF path:
  `3434 passed, 58 skipped, 6 warnings in 1001.43s (16:41)`.
- `git diff --check` passes.

## Supervisor re-verification

Before first certified use:

1. Recheck all six provisional numeric values directly against JCX-45-25 and
   confirm that each FY2026 individuals line has the same component scope.
2. Confirm that tips and overtime have no applicable JCX-45-25 repeal-revenue
   lines. The repository's JCX-35-25 enactment scores are intentionally not
   substituted; the tips line also includes a separate employer-credit scope.
3. Mortgage currently neutralizes `interest_deduction` to preserve exact
   comparability with the June/#298 surface. That output also contains
   non-mortgage interest. If the publication is strictly mortgage-only,
   evaluate switching to the valid narrower
   `deductible_mortgage_interest` output.
4. Charitable currently neutralizes the established itemized
   `charitable_deduction` output. It does not neutralize the separate 2026
   `charitable_deduction_for_non_itemizers`; add a list neutralization if the
   JCX-45-25 line includes both scopes.
5. Casualty uses `casualty_loss_deduction`. Its 2026 current-law activation
   parameter is false in the pinned engine, so a modeled zero is expected and
   should remain visible as the diagnostic gap rather than be patched away.
6. Verify the external validation dashboard accepts schema version 2 and the
   additive `repeal_revenue_benchmarks` table before deploying a release that
   contains it.

The supervisor can now push the local branch and coordinate the dashboard and
source-certification checks.
