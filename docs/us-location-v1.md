# US household location, version 1

Every US line locates a household the same way. It draws one 2020 census block at random, in proportion to block population, within the finest geography the household's source provides. Every other geography is then looked up from that block. Max ruled this on 27 September 2026, and it implements the block-first draw of microcosm#696. Fidelity comes from calibrating weights afterwards, not from the draw.

The code is `packages/microcosm-build/src/microcosm/build/us_runtime/block_location.py`. The CPS source geography is in `cps_source_geography.py`, and Census's identified-county lists are in `cps_identified_county_sources.py` and `data/cps_asec_identified_counties.csv`.

## The rule

A household's candidate blocks depend on what its source publishes:

| Source record | Candidate blocks | Kind recorded |
|---|---|---|
| ACS PUMS (publishes a 2020 PUMA) | the blocks of that PUMA | `puma` |
| CPS ASEC with `GTCO` ≠ 0 | the blocks of that county | `county` |
| CPS ASEC with `GTCO` = 0 | the blocks of its state outside the counties excluded for its ASEC year | `state_unidentified_counties` |
| Anything else | the blocks of its state | `state` |

The GTCO = 0 kind can also be narrowed to the record's CBSA, recorded as `state_unidentified_counties_cbsa`. That happens only when the source's CBSA codes agree with the ladder's delineation.

Within the candidate set, a block's probability is its 2020 P.L. 94-171 population divided by the set's population. The inverse CDF is exact: blocks in geoid order, cumulative population, one uniform per draw.

Every other geography comes from the block. Tract and county are prefixes of the block geoid. Place, SLDU, SLDL, CBSA, PUMA and the congressional district of every attached plan are block-level lookups in the ladder. A household's geographies therefore cannot contradict one another, and a new district map is a lookup, not a new draw.

## CPS county identification

Max, 16 August 2026: county identification in the public file is countywide. A record without a county code does not live in an identified county, so the draw must not leak mass into counties that are positively ruled out.

Census publishes the list: "List 4: FIPS County Codes" in each year's ASEC technical documentation, which notes that "Counties are only included on this list if the entire county is identified." `tools/build_us_cps_identified_counties.py` parses it for ASEC 2023, 2024 and 2025 (277 counties each) and ASEC 2026 (361). The PDF SHA-256s are recorded in the CSV's provenance file.

The list and the files disagree in the same way in all three years (`census_cps_2022/2023/2024.h5`, identical in the raw `hhpub25.csv`):

- **Coded in the file but not on the list (9):** 06017, 06061, 06077, 06113, 25009, 25021, 48439 (Tarrant), 53033 (King) and 53061 (Snohomish). Coverage is weighted CPS persons coded to the county divided by its 2020 population, normalized so the median coded county is 1. Tarrant's coverage is 0.12 to 0.19, King's 0.39 to 0.43 and Snohomish's 0.09 to 0.21: those files code only part of the county. The five CA/MA counties sit at 0.68 to 1.25.
- **On the list but never coded (6):** 09011 (New London), 18157, 24037, 36045, 48183 (Gregg) and 48303 (Lubbock). Households there carry `GTCO = 0`.

The 2026 list and file disagree too (361 listed, 307 coded). The file still codes Connecticut's old counties, which the list omits.

So a county is excluded for a `GTCO = 0` record only with two confirmations: it is on that ASEC year's list, and that year's file codes it. That leaves 271 excluded counties in each of 2023 to 2025. If a year has no official list, the data-derived set is the documented fallback. Each source year keeps its own set, and `census_cps_{Y}.h5` pairs with ASEC `Y + 1`.

**Years without the whole-county guarantee.** The 2026 documentation drops the sentence "Counties are only included on this list if the entire county is identified". The CSV's `entire_county_guarantee` column records, per year, whether a year's preamble carries it.

The 2026 file also breaks the premise itself. Every Delaware county is listed and coded for about its whole population (coverage 0.90 to 1.04), yet 49 Delaware households carry `GTCO = 0`. So for a year without the guarantee nothing is excluded, and a `GTCO = 0` record draws from its whole state; that year's `basis` is `no_whole_county_guarantee_state_draw`. In 2023 to 2025, Delaware and DC have no `GTCO = 0` households, and every state keeps candidate blocks for its `GTCO = 0` records. An empty candidate set would fail the draw rather than fall back.

**Opt-in, off by default: `partial_county_remainder`.** A partially coded county is still a full candidate in the GTCO = 0 pool, so in expectation it collects about (1 + coverage) of its population. The option scales such a county's blocks by max(0, 1 − coverage). That makes its expected mass equal its census population; a test checks this against a simulated draw. The ruled default is plain block population, and switching the default is Max's call.

**CBSA narrowing is automatic but not active today.** It applies only when every coded (county, `GTCBSA`) pair in the files matches the ladder's county-to-CBSA lookup. The CPS codes an older delineation than the ladder's OMB 2023 file (20 pairs disagree), so the check fails and the narrowing is skipped. The result is recorded in the manifest.

## The ladder

`tools/build_us_block_ladder_artifact.py` now adds a `puma` array: each block's 2020 PUMA, taken from its tract via Census's 2020 tract-to-PUMA relationship file. The array is additive to block-ladder schema 1. `load_us_block_ladder` ignores arrays it does not require, so legacy builds read the new file unchanged, and `load_us_location_ladder` validates the PUMA.

The national rebuild is `us_block_ladder_2020_puma.npz` (sha256 `9f2dc986…`):

- 5,769,942 blocks, 2,462 PUMAs, 436 districts, 3,143 counties, population 331,449,281;
- its seven core arrays are byte-identical to Route A's ladder (`7ba39b95…`);
- its per-PUMA population and its (PUMA, CD) and (PUMA, county) overlaps equal the PUMA ladder (`39a2ab2a…`) cell for cell (2,462, 4,043 and 4,620 cells).

So within a PUMA, the new draw's distribution of districts and counties is exactly the legacy PUMA ladder's, except that the two are now consistent with each other.

Congressional-district plans other than the ladder's primary (119th) are attached from a block-to-plan registry with `attach_congressional_district_plans`. Each attached plan becomes a household column `congressional_district_geoid__<plan>`. A plan the ladder already carries must agree block for block.

The registry is PolicyEngine/microcosm#1041, and a follow-up adds a `--cd-plan-registry` flag to each tool. Its plans are the 117th through 120th Congresses. Each plan's recorded sources, including known deviations, are copied into the manifest. For example, the 2020 Block Assignment File's CD layer is the 116th plan, which differs from the 117th in North Carolina, so #1041 builds NC's 117th from the 2019 plan instead.

## Keyed draws and clones

Each draw uses `stable_identity_uniforms("{household_id}:{clone}", seed, salt="us_block_location.population_draw.v1")`. A household's block therefore depends only on:

- the seed;
- its own id;
- its clone index;
- its candidate set.

It does not depend on row order, on which other households are present, or on how many clones were drawn.

With `clones = K`, each household is drawn K times. Clone k of household h becomes household h·K + k with weight w/K. `clone_frame_for_location` repeats every entity, remaps ids and membership, and remaps person-reference columns while keeping the 0 sentinel. Weighted mass is unchanged.

Every line currently refuses K > 1, because Route A's and the stacked pool's duplicate-copy checks do not yet recognize a location clone. That follow-up is tracked separately.

## Invariants

Each of these is a property test in `tests/engine_free/us/test_us_block_location.py` (Hypothesis over nested synthetic ladders) or `test_us_cps_source_geography.py`:

1. **Consistency.** Every derived geography equals the ladder's lookup of the drawn block, the household's state never changes, and `us_block_location_gate` passes. The gate fails, naming the column, when any single derived column is tampered with.
2. **Containment.** A household's block lies in its source geography:
   - a PUMA record's block is in its PUMA;
   - a coded CPS record's block is in its county;
   - a GTCO = 0 record's block is in its state and outside the excluded counties;
   - a CBSA-narrowed record's block is in its CBSA.

   An empty candidate set fails the draw; it never falls back to a coarser geography.
3. **Exactness and convergence.**
   - On an evenly spaced grid of M uniforms, every block receives its population share of M to within 1.
   - With a fixed seed, 200,000 draws pass a chi-square test against population shares.
   - County-within-PUMA and district-within-PUMA frequencies match the population overlaps.
   - The keyed uniforms pass a Kolmogorov–Smirnov test, and clones are uncorrelated.
4. **Determinism.** The same inputs and seed give identical output. Permuting the input rows changes nothing per household. Clone k's block is the same for any K > k.
5. **Conservation.**
   - Each household appears K times, with clone indices 0 to K−1 and clone ids key·K + k.
   - Its clone weights sum to its weight (relative tolerance 1e-12).
   - Frame cloning conserves every weighted entity's mass, keeps linkage valid, and places every parent in the same clone.
6. **Both confirmations.** The excluded set is exactly the intersection of listed and coded counties, per year.

The differential tests are:

- the block ladder's PUMA against the PUMA ladder built from the same synthetic sources;
- the national artifacts against each other (marked slow; set `MICROCOSM_US_BLOCK_LADDER_PUMA`, `MICROCOSM_US_BLOCK_LADDER_ROUTE_A` and `MICROCOSM_US_PUMA_LADDER` to run);
- an attached plan against the ladder's own copy of the same plan.
