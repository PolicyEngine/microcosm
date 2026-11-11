# Congressional-district vintage crosswalk (117th → 119th)

`congressional_district_vintage_crosswalk.csv` is a versioned, population-weighted
crosswalk from **117th-Congress** congressional districts (the geography of the
IRS SOI congressional-district table Microcosm calibrates against) to the
**119th-Congress** districts (the current Microcosm CD surface that
`policyengine.py` consumes). It is consumed by
`microcosm.build.us_runtime.congressional_district_vintage`
(`translate_congressional_district_facts_to_current_vintage`) to translate
old-vintage SOI CD facts onto the current district set, per
[PolicyEngine/microcosm#205](https://github.com/PolicyEngine/microcosm/issues/205).

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
uv run --python 3.13 --package microcosm-build --group dev python \
    tools/build_us_congressional_district_vintage_crosswalk.py \
    --cd-plan-registry build/us/us_cd_plan_registry_2020.npz \
    --out packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv
```

The builder reads the pinned US block -> congressional-district plan registry
(`../US_CD_PLAN_REGISTRY.md`) and refuses any other file. The method is a
**block overlay** of two of its plans on the same 2020 census blocks:

| Role | Registry plan | Underlying source |
|---|---|---|
| Old (117th) district of each 2020 block | `117th_congress` | 2020 Block Assignment Files, `CD` layer; North Carolina carried from its 2019 plan |
| Current (119th) district of each 2020 block | `119th_congress` | 119th Congressional District BEF (`NationalCD119.txt`) |
| Weight | the registry's `population` | 2020 P.L. 94-171 `POP100` per block |

The 2020 Block Assignment Files carry the **116th-Congress plans** on 2020
tabulation blocks. Census reports North Carolina as the only state whose
districts changed between the 116th and 117th Congress. Its 117th districts
are the 2019 remedial plan, which the registry carries onto 2020 blocks from
the General Assembly's block file. On 2020 blocks the 2016 and 2019 North
Carolina plans put only 55% of the state's population in the same district.
An earlier build read the BAF layer directly and so mapped North Carolina's SOI
districts through the 2016 plan; only North Carolina's rows changed when the
builder moved to the registry. Joining the 117th and 119th assignments on the
same 2020 blocks, weighted by 2020 block population, redistributes each old
district's population across the current districts it overlaps.

**Population is the correct default basis**: congressional apportionment and
one-person-one-vote redistricting are population operations, and equal-population
districts make an at-large state split ≈ evenly (e.g. Montana's old at-large
district splits 50/50 into MT-01/MT-02). ACS income/tax proxy weights for
*fiscal* targets are a documented future refinement (#205).

The registry's SHA-256 and its 117th-plan known deviations, plus the crosswalk
SHA-256 and full per-state population conservation, are recorded in
`congressional_district_vintage_crosswalk.csv.provenance.json` (written by the
builder). Every underlying Census and General Assembly source file, with its
URL and SHA-256, is in the registry's receipt,
`../us_cd_plan_registry.provenance.json`.

## Geoid conventions

Old geoids use the `5001700US` prefix and current geoids `5001900US`; the last
four characters are `state_fips + district` (`SSDD`). At-large states and the DC
non-voting delegate normalize to district `00` (the repo-wide convention shared
with `block_ladder_sources`), so DC is `…US1100` on both sides.

## Conservation and coverage (current build)

- 1,449 rows; **436 source districts → 436 current districts** (the full 119th
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
