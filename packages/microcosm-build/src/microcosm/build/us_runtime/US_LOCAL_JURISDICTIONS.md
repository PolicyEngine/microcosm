# US local-jurisdiction flags

policyengine-us 2.2.1 (and 2.60.4) declares four local-tax jurisdictions as
plain inputs with no formula. A flag stays `False` unless the dataset writes
it:

| Flag | Entity | Rule | 2020 geography | Population | Engine consumer |
| --- | --- | --- | --- | ---: | --- |
| `in_denver` | household | county | county 08031 (same blocks as Denver city, 08/20000) | 715,522 | Denver property-tax relief eligibility |
| `in_yonkers` | household | place | NY place 84000 (Yonkers city) | 211,569 | `ny_yonkers_income_tax` |
| `in_wilmington` | household | place | DE place 77580 (Wilmington city) | 70,898 | `de_wilmington_earned_income_tax` |
| `pa_philadelphia_wage_tax_resident` | person | county | county 42101 (same blocks as Philadelphia city, 42/60000) | 1,603,797 | `pa_philadelphia_wage_tax` |

Local flags that have a formula on `county_fips` (`in_nyc`, `in_la` and the
rest) need nothing: the engine computes them from the county the geography
line writes.

## Registry

`us_local_jurisdictions.json` holds the table above, and
`local_jurisdictions.py` pins its SHA-256. A county rule holds when
`county_fips` equals the registry county. A place rule holds when the first two
digits of `county_fips` equal the registry state and `place_fips` equals the
registry place code; place codes repeat across states, so the state is part of
the match. A person flag takes its household's value.

`tools/verify_us_local_jurisdiction_registry.py` checks the registry against
three Census sources:

- the 2020 Gazetteer place and county files (GEOID, ANSI code, name, land and
  water area; a county-rule jurisdiction's county and city have identical
  areas);
- the 2020 P.L. 94-171 `P1_001N` count for each place and county; and
- any block ladder: the blocks whose household would get a flag sum to the
  registry population, and a county-rule flag covers the same blocks as its
  city.

`block_ladder_checks` records the result for ladders `7ba39b95` and `6840b990`.
With `--record` the tool writes a ladder's counts into the registry and prints
the new SHA-256 to pin.

## Geography lines

- **Block ladder** (`with_household_us_geography_ladder`): one block fixes
  county and place, so the line writes all four flags.
- **Block-location draw** (microcosm#1047): call
  `with_us_local_jurisdiction_flags(frame, rungs=BLOCK_GEOGRAPHY_RUNGS)` once
  `county_fips` and `place_fips` are final.
- **PUMA ladder** (`with_household_us_puma_ladder`, the stacked pool): the
  ladder draws a county but no place. The line writes `in_denver` and the
  Philadelphia flag and never writes `in_yonkers` or `in_wilmington`; it also
  removes a stale column. The assignment summary records those two as
  `unavailable`, and the engine ABI lock lists them as
  `unavailable_place_jurisdiction_input`.
- **Pooled ACS line** (`with_optional_acs_spine`): donor rows arrive with
  block-line flags and ACS rows without. The pool re-derives the county-rule
  flags for every row from the final county and removes the place-rule flags.
  Without a PUMA ladder it removes every flag, because ACS rows then have no
  county. `build_optional_acs_multispine` records the pool's receipt as
  `provenance["local_jurisdiction_flags"]` on both paths.

Missing, null or malformed geography raises; no line writes `False` for a
household whose county or place it does not know.

## Existing artifacts

A base H5 built before this registry carries no flags. Two steps start from
an existing artifact rather than a geography line, and both call
`rederive_us_local_jurisdiction_flags`:

- the L0/refit export (`export_us_l0_refit_h5`) calls it on the selected
  support and records the receipt as `local_jurisdiction_flags` in its
  summary; and
- the fiscal refresh release builder calls it on the loaded base or prepared
  pool, again on the frozen support when a selection source reduces it, and
  again on the export support after the L0 or full-pool weights attach. It
  records each receipt, keyed `base`, `frozen_support` and `export`, under
  `support_value_repairs.local_jurisdiction_flags` in the calibration
  diagnostics.

A third step, the annual static-aging lane (`us_annual_static_aging`), keeps its
parent's columns and base-year bytes, so it cannot write the flags. It calls
`us_local_jurisdiction_artifact_receipt` instead, which refuses a parent that
lacks a flag its geography decides or carries one that disagrees with it, and
records the receipt as `base.local_jurisdiction_flags` in the candidate
manifest. A parent built before the flags existed therefore cannot seed annual
files until a rebuild adds them.

The helper reads the rungs from the artifact's own geography
(`us_local_jurisdiction_rungs_present`): the county rung when `county_fips` is
on every row, and the place rung only when every row also has `place_fips` and
a 15-digit `block_geoid`. A pooled artifact whose ACS rows have no block
therefore gets the county-rule flags only. A flag the artifact already carries
must agree with its geography (household and person flags both), or the step
fails.

The rungs belong to the whole table, so removing rows can add one: a frozen
support or L0 support that keeps only block-anchored donors decides
`in_yonkers` and `in_wilmington` even though the pooled base did not. That is
why the builder derives the flags again after each step that removes rows.
Deriving a support that already carries exactly the flags it decides returns
the same frame, so these repeat calls copy no tables.

## Gates

Both ladder gates fail:

- a flag column that disagrees with its rule;
- a null or non-boolean flag;
- a place-rule flag on a table with no complete `place_fips`; and
- a missing flag that the line's own rung decides. The block-ladder gate
  requires `in_denver`, `in_yonkers` and `in_wilmington`; the PUMA-ladder gate
  requires `in_denver`.

The gates see only the household table, so they check the household flags.
`us_local_jurisdiction_flag_failures(household, person=...)` also checks that
each person carries its household's value.

The Philadelphia flag records residence only. `pa_philadelphia_wage_tax` also
needs `pa_philadelphia_wage_tax_taxable_wages`, which no line supplies yet, so
the tax stays zero in built datasets.
