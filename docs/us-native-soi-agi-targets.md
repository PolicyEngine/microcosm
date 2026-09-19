# Native integration of SOI Table 1.1 targets

This compiler-only port adopts the target representation from
[Microcosm #960](https://github.com/PolicyEngine/microcosm/pull/960), reviewed at
`b563d11be314bdd608f736c553f435b9236820df` and retrieved September 19, 2026.
[Issue #958](https://github.com/PolicyEngine/microcosm/issues/958) distinguishes
these target changes from the separate full-vector tail support work.

The integration base is the clean native continuation
`59daeda3bf4a745eca2119e59520c6f691afe252`, descended from the active native API
source `5258a4c7f6d05cb1e68806d51b38996ea116bfb5`. At review, the compiler, its
existing tests and fact-to-target guide on that base were byte-identical to
fetched main `16c8e78d2f60d629da5d70294f643bce5c4597e2`. A sibling worktree carries
this port; the measurement and owner-PR worktrees are unchanged. Only compiler
source, compiler tests and documentation are adopted. The PR's population,
prototype reweighting and reform-score scripts are excluded.

## Accepted facts and year alignment

The rescue applies to stale `irs_soi.table_1_1` facts with all filing statuses,
national geography, measures `return_count` or `adjusted_gross_income`, and a
lower AGI bound of at least $100,000. Other tables, state/status slices and lower
cross-period bands remain excluded. The
[IRS table index](https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income)
identifies Table 1.1 as all returns by size and accumulated size of AGI, and
Table 1.2 as a separate marital-status table. The $100,000 edge is the model's
existing tax-unit-count approximation threshold, not a statutory filing limit.

Each stale class is represented as its share of the same-vintage Table 1.1
national total, multiplied by the latest eligible national total of that
measure. Controls cannot come after the build year or before the class year.
An absent or zero same-vintage denominator leaves the class unbound. The
existing dynamic target selection chooses the latest eligible class vintage.

For a 2022 class aligned to a 2023 control and aged to a 2024 build:

| Field | Amount target after aging |
|---|---|
| `name` / source-record identity | Original 2022 source record |
| `ledger_fact_period` | `2022` |
| `uprating_from_period` | `2022` |
| `uprating_to_period` | `2023` |
| `source_period` | `2023`, the existing ager's alignment origin |
| `target_period` / `aged_to` | `2024` |

The amount receives the realized 2022-to-2023 control ratio once, then the
remaining 2023-to-2024 CBO AGI ratio once. Counts receive the realized national
return-count ratio; the amount ager does not apply a CBO dollar factor to counts.
The nominal AGI edges remain the declared source edges. Same-period facts bypass
this rescue and preserve their existing behavior, including the existing
below-$100,000 behavior. No target IDs, API arguments or default compiler options
are changed.

## Integration boundary

The existing `compile_us_fiscal_target_registry` entry point consumes this code
when the separately admitted native release source includes the commit. No
Chronicle feed, the shared 32,867-target record, parity manifest, calibration
registry snapshot or frozen comparison surface is regenerated here. A future
build must record its actual newly compiled surface and requalify the relevant
frozen identities. The PR's separately reported target counts are not counts
verified on the shared native Chronicle artifact.

These targets do not create high-income support or repair its income
composition. They cannot establish support adequacy, calibration feasibility or
scientific improvement. The separate full-vector tail and ownership work remains
a launch prerequisite under #958. No dataset, calibration, model or reform run
was performed for this port. The held-out top-rate benchmark remains held out;
this change adds no target family for it.

Validation uses invented facts to exercise compiler acceptance, exclusions,
future/control-year selection, missing denominators, original identity and the
complete two-stage amount-aging chain. The official TY2023 workbook was acquired
and hashed during review, but no installed XLS reader was available, so the
upstream test's published numeric fixture was not independently cell-verified.
That source-review limitation does not change the values compiled from supplied
facts; production code introduces no hardcoded target amounts.
