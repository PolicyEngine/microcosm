# US household county geography, 2023

Investigation for microcosm#922 item 3, against base `dec45a439`.

## Engine and downstream contracts

`uv.lock:1692–1707` pins **policyengine-us 2.2.1**, with wheel SHA-256
`0993a6c73fcdfbe171a796aca9d302741bd8e00ab83388ad81c15a08740318c6`.
The [locked wheel](https://files.pythonhosted.org/packages/5e/09/86ac3968d2a18e28fce3ba1186fc21728e9f299b96a806d6f204f1e0aaed/policyengine_us-2.2.1-py3-none-any.whl)
was independently downloaded and hash-verified. Paths below refer to files in
that wheel; there is no matching GitHub version tag. Inspection without
importing the engine finds:

- `policyengine_us/variables/household/demographic/geographic/county/county_enum.py:315–322`
  names Connecticut's eight legacy counties, rather than planning regions.
- `policyengine_us/variables/household/demographic/geographic/county/county.py:36–69`
  joins `county_fips` to the engine county dataset, converts the result to
  that enum, and falls back to the first county in the state on an unrecognized
  code. Replacing Connecticut's legacy codes would therefore change rules
  evaluation rather than merely relabel geography.
- `policyengine_us/tools/geography/county_helpers.py:12–23`
  explicitly loads `county_fips_2020.csv.gz`. Its Connecticut rows are
  `09001,09003,09005,09007,09009,09011,09013,09015`.
- `policyengine_us/variables/input/geography.py:130–135`
  declares `county_fips` as a string household input. The engine's
  `policyengine_us/variables/gov/states/ct/oec/c4k/ct_c4k_region.py:22–30`
  uses county enum names to approximate Care 4 Kids regions. The unrecognized
  planning-region codes would fall back to Fairfield, changing the program's
  regional computation.
- `packages/microcosm-frame/src/microcosm/frame/adapters/policyengine_us.py:670–682`
  documents the engine's default county-based SPM measurement and its use of
  stored `county_fips`.
- `packages/microcosm-build/src/microcosm/build/us_runtime/geography_ladder.py:443–447,500–525`
  uses `county_fips` for NYC shares and requires the legacy county to match the
  block prefix. `puma_ladder.py:464–468,517–536` also checks county/state and NYC
  geography. These contracts must keep their current interpretation.
- `packages/microcosm-build/src/microcosm/build/us_runtime/base_pool.py:529–547`
  checks donor `county_fips` against the legacy tract prefix during pooling.

**Decision:** add the five-digit text household column `county_fips_2023`.
Preserve `county_fips` for the locked engine, and preserve 2020 block/tract IDs.
Register the new column as deliberately stored geography metadata, since it
is not a PE-US input (`packages/microcosm-data/src/microcosm/data/stored_inputs.py:320`).

## Reusing the existing mapping

PR #1072 already supplies the correct route:

- `block_ladder_sources.py:225–275` parses the 2020 BAF MCD block-to-town layer.
- `block_ladder_sources.py:393–515` parses the official town-to-planning-region
  crosswalk and joins blocks through their towns, refusing missing towns.
- `tools/build_us_block_ladder_artifact.py:67–82,285–310` identifies the Census
  source URLs and applies that join for CBSA assignment.

These paths are under `packages/microcosm-build/src/microcosm/build/us_runtime/`
unless prefixed with `tools/`. Read-only inspection of the existing cached
sources finds 169 defined towns, nine planning regions, and 49,845 defined-town
blocks. All 880 corresponding 2020 tracts resolve to exactly one planning
region. Undefined coastal-water geography is outside that support. A compact
tract table can therefore reuse the existing block/town mapping without
deriving boundaries or guessing from legacy counties.

The PUMA ladder's default county-only path needs a population-weighted tract
draw conditional on the already drawn PUMA and legacy county. This draw runs
after the existing draws and supplies only the new county column. With
`assign_tract=True`, the new county comes directly from the assigned tract.

## Implementation

The compact crosswalk, original Census town crosswalk, and provenance JSON
are packaged under `us_runtime/data/`. The loader checks SHA-256 digests.
`tools/build_us_county_vintage_crosswalk.py` reproduces them from the existing
cached CT BAF, town crosswalk, and P.L. 94-171 archives using the maintained
parsers. Its normal CLI was run successfully without building a ladder.

Both household ladder assignments now write `county_fips_2023`. PUMA's
conditional draw is isolated in `puma_county_vintage.py`. The new stored-input
register entry is bound to its producer by tests. Both geography gates require
the additional column and check completeness, text format, valid CT regions,
non-CT identity, and agreement with assigned block/tract geography when present.
Consequently, older artifacts without the new column fail these updated gates
and the default L0 export's required-geography check.

## Validation

Engine-free checks cover national support of 3,144 county-equivalents in the
50 states plus DC, one region per CT town, non-CT identity, one non-null value
per household, and CT population conservation across county vintages. Hypothesis
checks generated mapping invariants and both PUMA tract modes are exercised.
The source-backed crosswalk conserves Connecticut's 3,605,944 people across
all 42,008 populated blocks, eight legacy counties and nine planning regions.

The primary engine-free run passed **275 tests**, with **one skip**, serially
on Python 3.14.7. It covered the four new mapping/assignment/gate modules and
the existing block/PUMA ladder, stored-input register, L0 export, base-pool,
and block/PUMA source suites. `uv run ruff check .`,
`python3 tools/ci_test_plan.py verify`, and `git diff --check` passed.
Two additional seeded regression tests passed against outputs captured from
the original `dec45a439` PUMA implementation: PUMA, CD, legacy county and optional
tract assignments remain unchanged. Across both invocations: **277 passed,
one skipped**.

No full ladder NPZ is present in this workspace or the existing geography
cache. Use source-derived fixtures for CI and provide an optional read-only
check against an existing national ladder.

The skipped test is
`test_us_county_vintage.py::test_existing_full_ladder_has_3144_counties_without_rebuilding`.
To run it, set `MICROCOSM_US_BLOCK_LADDER_ARTIFACT` to an existing national
block-ladder NPZ and invoke that pytest node. The passing fixture check covers
the complete county support universe using 4,015 actual Census tracts; it does
not certify a full NPZ artifact. No ladder or release was built or published.

## Release-tool follow-up

The task excludes edits to `tools/build_us_acs_local_release.py` and
`tools/us_acs_local_cd_surface.py`. In particular,
`tools/build_us_acs_local_release.py:1203–1211` has a lean household checkpoint
allowlist containing only `county_fips`; that checkpoint will need to retain
`county_fips_2023` before the local county-target lane can use the new geography.
County calibration/target joins must explicitly select the 2023 column while
engine computation continues to use the legacy column.

axiom: n/a: data geography infrastructure
