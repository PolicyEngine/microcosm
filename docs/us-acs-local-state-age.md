# US ACS local-area state age groups (microcosm#1528)

Without age targets, and with state population targets from the 2020 Census
PUMA ladder, the ACS local-area release misses Census's age distribution in
nearly every state. Against the Census Vintage 2025 estimates for July 1,
2024, Build P's state slices count 10.2% more children than Census and 6.0%
fewer adults, and in 49 of the 50 states and DC the under-18 count exceeds
Census's by more than 5% (microcosm#1528 has the table). The `age` family, in
`--families` by default, binds each state's population by age group and moves
the state population targets to the same estimates.

## What the family binds

PolicyEngine/chronicle#417 adds the Chronicle package
`census-pep-2025-state-age-bands`. It reads Census's `sc-est2025-alldata6.csv`
and emits the resident population of each state and DC on July 1, 2024: one
record set per state (`census_pep.v2025.cy2024.state_resident_population_age_bands.<ff>`,
record set spec `census_pep.v2025.state_resident_population_age_bands.v1`),
each with sixteen values, the fifteen age groups below and the all-ages total.

| Value id | Ages |
|---|---|
| `age_0` to `age_5` | each single year from 0 to 5 |
| `6_to_11` | 6 to 11 |
| `12_to_17` | 12 to 17 |
| `18_to_24` | 18 to 24 |
| `25_to_34`, `35_to_44`, `45_to_54`, `55_to_64` | ten-year groups |
| `65_to_74` | 65 to 74 |
| `75_plus` | 75 and over |
| `all` | every age |

The single years let a reform that starts or stops at any age under 6 read
Census counts directly, and the 12-to-17 and 18-to-24 groups split at 18.

`compile_us_fiscal_target_registry(..., state_age_band_targets=True)` compiles
one `census_population` target per age-group fact and none for the all-ages
rows. Without the flag, the default, the compile drops every Vintage 2025
age-group fact before it builds targets, so every other build keeps its
targets: on the e86a83bb feed, the release's compile (`age_targets=True` with
the congressional district crosswalk, then
`apply_us_medicaid_enrollment_substitutions`) gives the same 33,250 targets as
the previous feed, and opening the fence adds the 765 age groups (51 x 15) and
changes no other target. Only the ACS local release opens it.
The target-parity manifest declares the `census_pep.v2025` family an
`off_by_default` reviewed exclusion.

Each age-group target counts, in each household of the target's state, the
persons at least as old as the group's lower bound and younger than its upper
bound (`_population_age_household_values` in
`tools/build_us_fiscal_refresh_release.py`).

## State totals and district shares

With the family, `state_age_surface` reads each state's all-ages row as its
total (`us_state_age_band_population_totals` in `us_runtime/fiscal_targets.py`),
and `ladder_population` passes those totals to `with_state_population_totals`,
which:

- replaces the ladder's 51 state populations with the Vintage 2025 totals;
- multiplies each state's 2020-ladder congressional-district populations by
  the state's ratio of new to old total, so each district keeps its 2020 share
  of its state and the state's districts sum to its new total (the largest
  district takes the floating-point residual);
- leaves its inputs unchanged and passes every other geography through.

On the county rung, `with_county_populations` then reconciles each state's
counties to the new state total, as it reconciles them to the ladder's.

Measured on the 2020 ladder (`us_puma_ladder_2020.npz`, CAS `39a2ab2a…`) and
the e86a83bb feed:

| | 2020 ladder | Vintage 2025, July 1, 2024 |
|---|---:|---:|
| 50 states and DC | 331,449,281 | 340,003,797 (+2.58%) |
| Indiana | 6,785,528 | 6,934,754 (+2.20%) |

The state ratios run from 0.9853 (West Virginia) to 1.0880 (Idaho). After
scaling, each state's districts (436 in all) sum exactly to its total.

## The surface

In `--soi-mode state`, the default, on the e86a83bb feed:

| | Without `age` | With `age` |
|---|---:|---:|
| `usda_snap` | 102 | 102 |
| `cms_medicaid` | 51 | 51 |
| `irs_soi` | 4,227 | 4,227 |
| `census_population` age groups | 0 | 765 |
| **Admin specs** | **4,380** | **5,145** |
| Population marginals (51 states, 436 districts) | 487 | 487 |
| **Targets** | **4,867** | **5,632** |

The age family changes the population marginals' levels, not their number.
Nationally the 765 groups sum to 340,003,797, the 51 state totals, and the
under-18 groups sum to 72,548,879.

## Refusals

The materialize stage stops with "The ledger feed's Census Vintage 2025 state
totals are unusable (microcosm#1528)" when a state's all-ages row names no
state, holds no positive number, or repeats. It stops with "The ledger feed
cannot supply the Census Vintage 2025 state age surface (microcosm#1528)" and
lists each problem when the feed:

- has other than 51 state totals;
- names a state's age groups without its total;
- gives a state other than fifteen age groups;
- dates a state's age groups to a period other than 2024;
- gives age groups that do not tile ages 0 and over;
- gives age groups that do not sum exactly to the state's total.

`with_state_population_totals` refuses totals that do not name exactly the
ladder's states, and any zero, negative or non-finite total.

## Receipts

`run_identity.json` and `materialize_rss.json` record `state_population`:
`source` and `district_source` name where the state and district levels come
from, and with the family the age surface's record set spec, period, 51
states, 15 groups per state, 765 targets and their population, 340,003,797.
The population targets' `source` fields carry the same names:

| Target | Without `age` | With `age` |
|---|---|---|
| State | `US Census Bureau 2020 PUMA population ladder` | `US Census Bureau Vintage 2025 state population estimates (July 1, 2024)` |
| Congressional district | `US Census Bureau 2020 PUMA population ladder` | `US Census Bureau 2020 PUMA population ladder district shares, scaled to Vintage 2025 state population estimates (July 1, 2024)` |

The reviewed limitation `cd_population_marginal_vintage_2020` keeps its id
and, with the family, says that each district keeps its 2020 Census share
scaled to its state's Vintage 2025 estimate.

## Group quarters

Census resident population counts people in group quarters. The staging frame
keeps each group-quarters person as a one-person household weighted by PWGTP,
except children under 15: the staging builder drops 1,946 of them (51,806
weighted, 0.037% of the raw ACS household weight before removal) and records
the reviewed limitation `acs_group_quarters_children_outside_spm_universe` (see
[the ACS SPM role note](us-acs-spm-role.md)). Census's 0-to-5, 6-to-11 and
12-to-17 counts include those children, so calibration spreads their weight
over household children of the same state and age group. Nationally they make
up 0.07% of the 72,548,879 children under 18 that Census counts.

## Reproduce the previous surface

`--families snap,medicaid,soi` builds the surface from before microcosm#1528:
no age targets, and state and district populations from the 2020 ladder.
`materialize_rss.json` records each checkpoint's families, and the package
stage's refresh recipe names them with `--families`; a record that names no
families reads as `snap,medicaid,soi`, the default until microcosm#1528.
Command lines stored before microcosm#1528, such as the release recipes in
`experiments/us-acs-local-hours-rebuild-20260922/`, name no `--families`, so
they take the new default: they add age targets on this feed and refuse a feed
without the age groups. To replay one, add `--families snap,medicaid,soi` and,
when the command names no `--soi-mode`, the build's recorded mode. That
experiment's recipes name none and its manifest records `totals`
([the SOI surface note](us-acs-local-soi-target-surface.md) gives the command).

## What the family leaves out

The family adds no sex, race or Hispanic-origin dimension and no age targets
for counties or districts, and the release still has no household-count
target (microcosm#880).

As of 2026-11-06 no release has fit this surface. PolicyEngine/microcosm#1531
reports the current slices' age fit and a projection of this family onto them,
from the state audit in PolicyEngine/child-poverty-impact-dashboard#129. A
rebuild needs a national donor that passes the staging build's donor checks,
which the default donor `populace-us-2024-spm-20260915` and its child
`populace-us-2024-spm-receipts-20260923` both fail.
