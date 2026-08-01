# US multispine hours transfer-leaf adjudication

Date: 2026-08-01

Base: `8828dee`

Engine: `policyengine-us==1.764.6`, `policyengine-core==3.26.11`

This receipt adjudicates the pool-run-3 failure before changing the transfer
plan. Engine facts were evaluated with the synced sibling virtual environment
and this worktree on `PYTHONPATH`, so the installed engine was pinned while the
Populace classifier and plan came from this branch.

## Hours variables

| Transfer name | ASEC source | Engine metadata | Legacy enhanced-CPS treatment | Pool verdict |
| --- | --- | --- | --- | --- |
| `weekly_hours_worked_before_lsr` | `HRSWK` | person, float, year, default `40.0`, no formulas, input leaf | Built as `weekly_hours_worked`, QRF-imputed onto the PUF clone half, then renamed to the before-LSR leaf for export | Keep as a transferred input leaf |
| `hours_worked_last_week` | `A_HRS1` | person, float, year, default `0`, no formulas, input leaf | Carried directly, QRF-imputed onto the PUF clone half, and exported as an input leaf; also consumed by the legacy FLSA derivation | Keep as a transferred input leaf |
| `weeks_worked` | `WKSWORK`, clipped to 0--52 | person, int, year, default `0`, formula from `2025-01-01`; the formula returns the prior year's `weeks_worked`; formula-owned under Populace's period-agnostic classifier | Carried directly, QRF-imputed onto the PUF clone half, and deliberately exported for 2024 because the legacy guard classified formulas by dataset period; the same export failed for 2025 | Drop from the transfer plan and final pool input surface |

The current multispine pool does not run the ORG/FLSA operator before its
simulation boundary. `org_wages.py` is a legitimate pre-simulation consumer in
the separate legacy fiscal-refresh path, but it is not wired into
`POOL_SOURCE_OPERATOR_ORDER`, `derive_multispine_pool_inputs`, or
`tools/build_us_multispine_pool.py`. Therefore it does not justify persisting a
formula-owned canonical `weeks_worked` column in this input-only pool. Raw
`WKSWORK` remains source evidence; any future pool-local ORG integration must
consume a private/source representation and remove it before simulation.

The behavioral deviation is explicit: the archived build treated
`weeks_worked` as a valid 2024 input because its only engine formula begins in
2025. The required Populace transfer guard treats a variable with a formula in
any period as formula-owned, so the corrected pool cannot preserve that 2024
override.

## Legacy receipt

The local archive checkout at commit
`42ed5d45c56df80d754fbe24cce21cfeb8d05cbe` provides the frozen evidence:

- `policyengine_us_data/datasets/cps/census_cps.py` loads `HRSWK`, `WKSWORK`,
  and `A_HRS1`.
- `policyengine_us_data/datasets/cps/cps.py` maps the three sources as described
  above and clips `WKSWORK` to 0--52.
- `policyengine_us_data/datasets/cps/extended_cps.py` lists all three in the
  CPS-only imputation surface and renames `weekly_hours_worked` to
  `weekly_hours_worked_before_lsr` before export.
- `tests/unit/test_extended_cps.py` preserves `weeks_worked` for a 2024 export
  and rejects it for 2025, confirming the period-specific legacy exception.

## Initial full-plan sweep

The uncorrected pool plan contained 117 distinct targets. The exact production
classifier reported:

```text
formula_owned_targets ['weeks_worked']
unknown_targets ['medicare_part_b_premiums']
```

The second result is an additional leaf-classification finding. In
PolicyEngine-US 1.764.6, `medicare_part_b_premiums` is unknown, while
`medicare_part_b_premiums_reported` is a person/float/year input leaf with
default `0` and no formulas. The corrected producer and transfer inventory must
use the reported leaf name.

## Post-fix full-plan sweep

After removing canonical `weeks_worked` and moving the Part B source carry to
`medicare_part_b_premiums_reported`, the live engine sweep returned:

```text
policyengine-us 1.764.6
targets 116
sorted_names_sha256 b62b3038c7ddb83fd6b59bdf4a4549ce40ebb11a1b0d0ea040b656186e5efa2a
formula_owned []
unknown []
non_leaves []
```

`assert_acs_transfer_targets_are_input_leaves` now owns the production
formula-owned rejection used by `transfer_acs_inputs` and by the complete
producer/dtype sweep. The strict sweep also requires all 116 names to appear in
`PolicyEngineUSEngine.variables()`. PR CI installs the workspace US extra so
the live classifier cannot silently degrade to its static fallback; the clean
wheel suite may still skip the live-engine-only test because country engines
are optional wheel dependencies.
