# US PUMA-anchored geography ladder

The US ACS artifact's geography spine is anchored at the **2020 Public Use
Microdata Area** — the finest geography the ACS PUMS publishes. This is the US
analogue of the UK output-area ladder (populace #354) and a sibling of the US
block ladder (`geography_ladder.py`): one national dataset, filterable at any
grain, never per-area files (populace #275).

## What it does

- An **ACS-spine** record already carries its `puma`; the ladder keeps it.
- An **ASEC-spine** record knows only `state_fips`; the ladder draws a PUMA
  within that state proportional to 2020 PUMA population.
- Every coarser layer then derives from the PUMA by a population-weighted draw
  over the PUMA's block-population overlap with that layer:
  - **congressional district (119th)** — `(PUMA, CD)` overlap
  - **county** — `(PUMA, county)` overlap
  - **tract** (behind `assign_tract`) — `(PUMA, tract)` overlap; when assigned,
    the county derives structurally from the tract so the two never disagree.

Congressional district, county and state are the launch requirement; tract is
the flagged-off stretch rung. Draws come from one seeded generator consumed in
a fixed sorted order (states, then PUMAs), so an assignment is reproducible
from its `seed`.

The dense ACS-spine artifact is filtered to state / congressional district /
county for local analysis because every record carries all three.

## Why one block pass

2020 tabulation blocks nest in 2020 census tracts, and 2020 tracts nest in 2020
PUMAs. So `tract = block // 10**4`, `county = block // 10**10`, and
`tract → PUMA` is a lookup. Summing 2020 block populations by `(PUMA, CD)`,
`(PUMA, county)`, `(PUMA, tract)` and by PUMA yields the three overlap tables
and the anchor population from a single block pass. Every populated block
contributes to exactly one PUMA, one CD, one county and one tract, so **each
overlap table conserves its PUMA's population exactly** — the loader refuses an
artifact that violates this.

## Sources (pinned in `us_puma_ladder.provenance.json`)

| Role | Census source | Vintage |
|---|---|---|
| tract → PUMA (and county via the tract prefix) | 2020 Census Tract to 2020 PUMA relationship file (`2020_Census_Tract_to_2020_PUMA.txt`) | 2020 |
| block → 119th CD | 119th Congressional District BEF (`NationalCD119.txt` in `cd119.zip`) | 119th Congress |
| block → population (the weight) | 2020 P.L. 94-171 geographic headers, per state (`{usps}geo2020.pl`) | 2020 |

Every source's URL + SHA-256, the reproducible artifact SHA-256, and the
national validation totals are recorded in `us_puma_ladder.provenance.json`.
The `block → CD` and `block → population` parsers are reused unchanged from the
block ladder (`block_ladder_sources.py`); the only source this ladder adds is
the small tract-to-PUMA relationship file.

## Build

```bash
uv run python tools/build_us_puma_ladder_artifact.py \
    --out build/us/us_puma_ladder_2020.npz \
    --cache-dir ~/.cache/populace-us-geography
```

The tool downloads (and caches) the sources, runs the block pass, aggregates,
writes one national NPZ, and self-checks it by loading it back through
`load_us_puma_ladder` before writing the provenance summary. The artifact is
byte-reproducible from the pinned sources.

National build (all 51 states): **2,462 PUMAs**, **331,449,281 people** (the
2020 apportionment population, conserved exactly), **436** congressional
districts, **3,143** counties, 83,848 tract overlaps.

## Wire-up (documented, not rewired)

The runtime — `load_us_puma_ladder` → `assign_us_puma_ladder` (or
`with_household_us_puma_ladder`) → `us_puma_ladder_gate` — is the drop-in for
the ACS/ASEC spine's geography stage. Wiring it behind the country-spec
geography-spine schema (a PUMA-anchored spine method, and reading the built
artifact from the release bundle) is the follow-up, kept out of this change to
stay reviewable — the same posture #354 took for the UK ladder.
