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
  county.

Missing, null or malformed geography raises; no line writes `False` for a
household whose county or place it does not know. Both ladder gates fail a
flag column that disagrees with its rule, or a place-rule flag on a table with
no complete `place_fips`.

The Philadelphia flag records residence only. `pa_philadelphia_wage_tax` also
needs `pa_philadelphia_wage_tax_taxable_wages`, which no line supplies yet, so
the tax stays zero in built datasets.
