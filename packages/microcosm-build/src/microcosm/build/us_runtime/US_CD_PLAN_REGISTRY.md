# US congressional-district plan registry

A household's location is one 2020 census tabulation block, and every other
geography is a lookup from that block. Congressional districts are the one
layer with several live versions on the same blocks, so this registry records,
for every populated 2020 block, its district under each registered plan. A
block draw can then derive every plan's district from the chosen block, and
SOI's 117th-Congress district tables become exact block sums with no
crosswalk.

Code: `cd_plan_registry.py`. Build: `tools/build_us_cd_plan_registry_artifact.py`.
Receipt (artifact SHA-256, sources, per-plan totals, every check's result):
`us_cd_plan_registry.provenance.json`.

## Plans

| Plan id | Districts | Source | Apportionment |
|---|---|---|---|
| `117th_congress` | 2021–2023; the geography of the IRS SOI congressional-district tables for tax years 2020–2022 | Census 2020 Block Assignment Files, `CD` layer; North Carolina carried from its 2019 plan (below) | 2010 |
| `118th_congress` | 2023–2025 | Census 118th Congressional District block equivalency file | 2020 |
| `119th_congress` | 2025–2027; the block ladder's primary plan | Census 119th Congressional District block equivalency file | 2020 |
| `120th_congress` | the November 3, 2026 election, including the 2025–26 mid-decade redraws | Census 120th Congressional District block equivalency file (published 2026-08-31); Missouri from its 2022 map (below) | 2020 |

District geoids are `state_fips * 100 + district`. At-large states and the
DC delegate are district `00`, the repo-wide convention. Each plan's metadata
records its source files (URL and SHA-256), its apportionment, the blocks
Census lists as split by a district line, and any known deviations.

### North Carolina in the 117th plan

The 2020 BAF `CD` layer carries the 116th-Congress plans on 2020 blocks.
Census reports North Carolina as the only state whose districts changed
between the 116th and 117th Congress: the 2020 election used the 2019
remedial plan (HB 1029, S.L. 2019-249), which Census never tabulated on 2020
blocks. On 2020 blocks the two North Carolina plans put only 55% of the
state's population in the same district, so the 116th rows would be wrong for
SOI's 117th-plan targets there.

The build therefore carries the General Assembly's block file for the 2019
plan (on 2010 blocks) onto 2020 blocks through the Census 2010–2020
tabulation block relationship file. Each 2020 block takes the district whose
2010 blocks cover the most of its land area, then water area, then the lower
district code. The 2020 blocks with no North Carolina 2010 counterpart take
their block group's majority district by population, falling back to the
tract and then the county. Two checks run on every build and are recorded in
the receipt:

- **The method reproduces Census.** Run on the 2016 plan, the same method
  matches Census's own 2020-block version of that plan (the BAF `CD` layer)
  for 99.992% of North Carolina's population. The build refuses below 99.9%.
- **The result matches the official figures.** The carried plan's district
  populations are within 704 people (0.09%) of the General Assembly's official
  2020-census populations for the 2019 plan. The build refuses above 0.5%.

289 of the state's 2020 blocks (16,671 people) straddle a 2019-plan district
line. Four have no North Carolina 2010 counterpart; one of them is populated
(13 people).

### Missouri in the 120th plan

Census's 120th rows for Missouri encode the 2025 HB 1 map, and Census's 120th
landing page warns they may not be the districts in place for November 2026.
The Missouri Supreme Court held on 2026-09-03 (von Glahn v. Hoskins) that HB 1
cannot take effect before a referendum on the November 2026 ballot. The U.S.
Supreme Court held on 2026-09-25 (26A388, per curiam) that "the 2022 map—not
the 2025 map—must be used in the 2026 congressional election." Missouri's
`120th_congress` rows are therefore its 2022 map, taken from the 119th file.
The build checks that those rows are unchanged from the 118th. This follows
the court record as of 2026-09-27; a later ruling would mean a rebuild.

## Invariants

Checked when the artifact is built (`assemble_us_cd_plan_registry`) and again
on every load (`load_us_cd_plan_registry`):

- **Universe.** The blocks are exactly the 2020 P.L. 94-171 blocks with
  positive population in the 50 states and DC (5,769,942 blocks, 331,449,281
  people), sorted and unique: the block ladder's universe.
- **Partition.** Every block has exactly one district under every plan, and
  that district lies in the block's state. A populated block that a plan's
  source leaves unassigned is a build error, never a gap.
- **Apportionment.** Each state's districts under a plan are exactly its House
  apportionment for that plan's census: `00` for an at-large state or DC,
  otherwise `01` through `n`. So every apportioned district contains a
  populated block.
- **Conservation.** Therefore, for every plan, district populations sum
  exactly to state populations, and those sum to the national total.
  `summarize_cd_plan_registry` re-checks this.

The build also cross-checks its sources against each other and refuses on any
disagreement:

- the packaged seat counts against the Census apportionment table;
- each Census block equivalency file's state files against that state's rows
  in its national file;
- the states whose blocks change from the 118th to the 119th, and from the
  119th to the 120th, against the states Census ships a state file for (its
  list of redrawn states);
- the one block Census lists as split by a district line (Colorado
  080010096072000) against the district Census tabulates it with;
- with `--block-ladder`, the registry against the ladder: the same blocks,
  the same populations, and the same 119th district on every block.

The property tests (`test_us_cd_plan_registry_properties.py`) check the
partition, conservation, apportionment and determinism invariants on random
geographies. They also check that dropping, crossing or merging any assignment
is refused, and that the crosswalk picks the largest overlap regardless of row
order.

## Build

```bash
uv run python tools/build_us_cd_plan_registry_artifact.py \
    --out build/us/us_cd_plan_registry_2020.npz \
    --block-ladder build/us/us_block_ladder_2020.npz \
    --provenance-json packages/microcosm-build/src/microcosm/build/us_runtime/us_cd_plan_registry.provenance.json
```

The tool downloads and caches its sources, builds the NPZ (about 18 MB), loads
it back through the validating loader, and writes the receipt. The output is
byte-reproducible from the pinned sources. A national build takes about two
minutes and 2.4 GB of memory.

## Use

```python
from microcosm.build.us_runtime.cd_plan_registry import load_pinned_us_cd_plan_registry

registry = load_pinned_us_cd_plan_registry(path)  # refuses any other file
registry.plans["117th_congress"]  # district per block, aligned to registry.block_geoid
registry.plan_sources["117th_congress"]["known_deviations"]
```

`load_pinned_us_cd_plan_registry` refuses any file whose SHA-256 differs from
the packaged receipt. `load_us_cd_plan_registry(path, expected_sha256=...)`
pins to another digest. The registry assigns no households. A block draw
looks up each plan's district from the household's block.
