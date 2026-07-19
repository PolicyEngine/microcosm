# Congressional-district vintage crosswalk (117th → 119th)

`congressional_district_vintage_crosswalk.csv` is a versioned, population-weighted
crosswalk from **117th-Congress** congressional districts (the geography of the
IRS SOI congressional-district table Populace calibrates against) to the
**119th-Congress** districts (the current Populace CD surface that
`policyengine.py` consumes). It is consumed by
`populace.build.us_runtime.congressional_district_vintage`
(`translate_congressional_district_facts_to_current_vintage`) to translate
old-vintage SOI CD facts onto the current district set, per
[PolicyEngine/populace#205](https://github.com/PolicyEngine/populace/issues/205).

Columns:

- `source_geography_id` — 117th-Congress district (`5001700US` + state FIPS +
  district, at-large/delegate `00`).
- `target_geography_id` — 119th-Congress district (`5001900US` prefix, same
  geoid convention).
- `pair_population` — the 2020 P.L. 94-171 population of the blocks shared by
  the pair; per-state sums conserve the 2020 state totals exactly.
- `weight` — `pair_population` divided by the source district's total assigned
  population: the normalized share, **summing to 1.0 per source district in
  the raw file** (loader-enforced to 1e-9 for the packaged artifact).

The translated CD targets are **derived build artifacts, never Ledger facts** —
the fact-vs-computed boundary of
[PolicyEngine/ledger#71](https://github.com/PolicyEngine/ledger/issues/71). The
crosswalk itself is regenerable from primary Census sources and carries its own
lineage; the same declared-consumer-side-transform pattern applies to the
Belgian NIS-code vintage work in
[PolicyEngine/ledger#69](https://github.com/PolicyEngine/ledger/issues/69).

## How it is built

Regenerate with:

```
uv run --python 3.13 --package populace-build --group dev python \
    tools/build_us_congressional_district_vintage_crosswalk.py \
    --out packages/populace-build/src/populace/build/us_runtime/data/congressional_district_vintage_crosswalk.csv \
    --cache-dir ~/.cache/populace-us-geography
```

The method is a **single-vintage block overlay** — both district assignments are
read on the same 2020 census blocks, so no 2010↔2020 block bridge is needed:

| Role | Source | Format |
|---|---|---|
| Old (117th) district of each 2020 block | 2020 Block Assignment Files, `CD` layer (`BlockAssign_ST{fips}_{usps}_CD.txt`) | `BLOCKID\|DISTRICT` |
| Current (119th) district of each 2020 block | 119th Congressional District BEF (`NationalCD119.txt`) | `GEOID,CDFP` |
| Weight | 2020 P.L. 94-171 `POP100` per block (`{usps}geo2020.pl`, summary level 750) | pipe-delimited |

The 2020 Block Assignment Files were published with the 2020 P.L. 94-171 release
and carry the **116th-Congress plan** — whose district geography is identical to
the 117th — expressed on **2020** tabulation blocks. Joining that old assignment
and the current 119th BEF on the same 2020 blocks, weighted by 2020 block
population, redistributes each old district's population across the current
districts it overlaps.

**Population is the correct default basis**: congressional apportionment and
one-person-one-vote redistricting are population operations, and equal-population
districts make an at-large state split ≈ evenly (e.g. Montana's old at-large
district splits 50/50 into MT-01/MT-02). ACS income/tax proxy weights for
*fiscal* targets are a documented future refinement (#205).

Source URLs and their SHA-256s, plus the crosswalk SHA-256 and full per-state
population conservation, are recorded in
`congressional_district_vintage_crosswalk.csv.provenance.json` (written by the
builder). Base source pages:

- 119th CD BEF: `https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/2025/119-congressional-district-befs/cd119.zip`
- 2020 Block Assignment Files: `https://www2.census.gov/geo/docs/maps-data/data/baf2020/BlockAssign_ST{fips}_{usps}.zip`
- 2020 P.L. 94-171 Redistricting Data: `https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/{state}/{usps}2020.pl.zip`

The exact per-file URLs and their SHA-256s are in the provenance sidecar; these
are the templates the builder resolves.

## Geoid conventions

Old geoids use the `5001700US` prefix and current geoids `5001900US`; the last
four characters are `state_fips + district` (`SSDD`). At-large states and the DC
non-voting delegate normalize to district `00` (the repo-wide convention shared
with `block_ladder_sources`), so DC is `…US1100` on both sides.

## Conservation and coverage (current build)

- 1,444 rows; **436 source districts → 436 current districts** (the full 119th
  Congress set, including one-district states and DC).
- Every populated 2020 block in all 50 states + DC is covered:
  **331,449,281 people**, with **zero** unmatched or cross-state population.
- The issue's shrinking-state districts (CA-53, IL-18, MI-14, NY-27, OH-16,
  PA-18, WV-03) appear only as **sources** and are redistributed into current
  districts; the growing-state districts (CO-08, FL-28, MT-02, NC-14, OR-06,
  TX-37, TX-38) appear as populated **targets**.

Because each `weight` is its pair's share of the old district's population
(`pair_population` over the source total, so weights sum to 1 per source), the
translation redistributes old-vintage totals across current districts and
**conserves state and national totals exactly** before any period uprating — the
#205 acceptance property.
