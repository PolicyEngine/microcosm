# US local-area annual projections

The County Files ([microcosm#922](https://github.com/PolicyEngine/microcosm/issues/922))
need the county-calibrated 2024 local-area build projected to 2026 and 2027.
`microcosm.build.us_annual_static_aging` does this from a build of the local
release, `microcosm-us-2024-local`, as it does for national builds
([the national guide](us-annual-static-aging.md)). The candidate it writes is
assembled as a build of the release `microcosm-us-2024-local-annual`.

## What the builder takes

The builder reads the base build from its folder (`--base-build-dir`): its
`release_manifest.json`, `build_manifest.json` and, when present,
`sha256sums.txt`. Both manifests must name release `microcosm-us-2024-local`
and the same build. The dataset is the build's one microdata artifact,
`microcosm_us_2024_acs_local` in `microcosm_us_2024_acs_local.h5`; a local
build claims no default dataset. The release tool's package stage writes the
H5 beside `releases/`, not in the build folder, so `--base-artifact-root` names
the directory that holds it. A build downloaded from the Hub has its H5 in its
folder and needs no root.

The builder refuses the base before writing anything when the H5's SHA-256
differs from the one the release manifest records, or from the one the
ledger lists. It reads the base build's ID (the candidate's
`parent_release`) and its dataset key from the manifests, so a local base
takes no `--base-h5`, `--base-sha256`, `--parent-release` or `--base-dataset`.

The H5 must be the local release's own layout: the six entity tables in fixed
format, written by `microcosm.build.us_runtime.h5_io.write_nullable_us_h5`,
a `_time_period` of 2024, the artifact-metadata row
`{"artifact_kind": "calibrated_local_area_artifact", "entity_hdf_format":
"fixed_nullable", "household_weight_kind": "calibrated"}`, and
`household_weight` as the only stored weight. Every base and projected
household weight must be finite and strictly positive.

## What it writes

`--years 2026 2027` writes exactly those two years. Without `--years`, the
builder writes every year after the base year through `--end-year`. Each
projected year's H5 is `microcosm_us_<year>_acs_local.h5`, under the key
`microcosm_us_<year>_acs_local`. The base H5 is not copied into the
candidate. The candidate directory holds only:

- `annual_manifest.json`, the candidate manifest, written last;
- `build_status.json`;
- `microcosm_us_<year>_acs_local.h5` and `projection_<year>.json` for each
  projected year.

Each projected year is written in the base's layout: the same six fixed-format
tables, with the same columns in the same order and the same rows, the same
metadata row and root attributes (the `block_v1` location attributes when the
base has them), and `_time_period` set to the year. The builder checks the
written file round trip, its stored column order, its metadata row and its
root attributes. It also checks that policyengine-us's `USSingleYearDataset`
opens the file at its year.

The builder's `_verify_native_local` guard refuses a requested year that
differs from the file's stored year. The upstream `USSingleYearDataset`
constructor itself reads the stored period without enforcing a requested
period, and an unbounded `Microsimulation` can extend a single-year input.
Engine callers must select the correct file and bound its projection window
with `us_dataset_through`. The fixture workflow test checks own-year input
calculations with that bounded dataset; independent acceptance must also
check any earlier-year inputs policy formulas read.

The candidate manifest has `layout: "fixed_nullable"`, and these entries:

- `base`: the dataset key, the year (2024), `parent_release` (the base
  build's ID), `release` (`microcosm-us-2024-local`), the path and the
  SHA-256 of the base H5.
- `settings`: the national settings plus `years`, the projected years.
- `metadata.dataset_years`: maps the base key to the year map, for example
  `{"2024": "microcosm_us_2024_acs_local", "2026": "microcosm_us_2026_acs_local",
  "2027": "microcosm_us_2027_acs_local"}`.
- `artifacts`: each year's file name, SHA-256, year, row counts, columns,
  round-trip receipt and per-year receipt. The base's entry has
  `source: "base_build"`. The assembler takes that file from the base build,
  not from the candidate.

Any downstream household-measures sidecar must identify the projected file by
its name and SHA-256. This builder and the assembler do not create or attach
those sidecars. Adding materialized columns to an H5 changes its hash and
column inventory; a separate delivery export must define and validate that
schema, rather than mutating a hash-bound annual candidate in place.

## What the projection changes

Static aging changes two things, as on the national line
([static aging](static-aging.md)):

- **Household weights.** The weights are recalibrated toward a target for
  each SSA age-by-sex cell: its 2024 weighted count times SSA's projected
  national growth for that cell. The target is anchored on the frame, so the
  2024 calibration is kept and only the growth is applied. It is a
  calibration target, not an identity: a cell's projected count can miss it,
  within the calibration's fit and the weight cap below.
- **Monetary amounts.** Each column that follows an engine uprating series
  gets one national factor, the same for every county. The factor depends on
  the kind of series:
  - **A national total** (a CBO or IRS SOI aggregate). The factor is
    `growth × base weighted total / reweighted total`: the column's 2024
    values summed nationally under the 2024 weights, divided by the same
    values summed under the new weights. The column's national weighted
    total then grows by exactly the series' growth, and the factor absorbs
    only what the reweighting did not. For example, a $100 amount whose
    household weight moves from 1 to 1.1, under growth of 1.2, becomes
    $100 × 1.2 / 1.1 = $109.09, not $120. A column with both positive
    amounts and losses gets two such factors, one solved over its positive
    amounts and one over its losses, so every value keeps its sign.
  - **A per-person rate or a price index.** The factor is the index's ratio,
    projected year over 2024.

  A county's projected total for a column therefore moves with both its own
  reweighting and the national factor.

It imposes no state or county target. State and county totals and shares
therefore move only through each household's age-sex composition. A county
with more people in fast-growing cells gains share. No household's weight
moves above 5 times its 2024 weight (`--max-weight-ratio`). The projection
does not re-target state or county totals, and it adds no county-level
re-targeting. County rows still sum to their state by construction, because
each household keeps its one county and state. The 2024 county calibration's
fit to public county targets is not re-imposed in later years.

Each year's `projection_<year>.json` records the drift in
`geography_shares`. For `state_fips`, and for `county_fips_2023` (or
`county_fips` when the household table lacks it), it lists each value's 2024
and projected weighted household share, and the largest absolute change in
any value's share.

Every projected year is a forecast from the 2024 base. That includes 2026,
and 2027 is three years after the base year.

Static aging changes nothing else. Every column the year's uprating did not
factor is checked equal to the base's, in value and dtype, before the year is
written. That covers ids, memberships, every geography column (`state_fips`,
`county_fips`, `county_fips_2023`, `congressional_district_geoid`, `puma`,
`block_geoid`, `tract_geoid`, `place_fips`, `sldu`, `sldl`, `cbsa_code`) and
the person attributes: age, relationships, work indicators, industry and
occupation. A factor on a geography column, on `age` or `is_female`, on an
`is_*` flag, on `worked_last_year`, or on a relationship, industry or
occupation column is refused. A factored
integer column is stored as float64 in projected years. Record IDs do not
track individual lives across years.

**Age cells.** The local frame mixes single-year ACS ages with CPS-coded
donor ages, where 80 stands for 80–84 and 85 for 85 and over. Aging pools
every age the CPS way, 80–84 at 80 and 85 and over at 85, matching SSA's
cells (`age_top=85`, `age_bands={80: 84}`). The pooling happens in a
projection-only column of a slim aging frame. The stored `age` column is
unchanged.

## Running it on the 2024 county build

These are the steps for the real build. Publication is a separate, reviewed
step.

1. **Gather the inputs.**
   - `$PKG`: the release tool's `--out` directory from the 2024 county build,
     holding `releases/microcosm-us-2024-local/` and
     `microcosm_us_2024_acs_local.h5`.
   - `$SSA_CSV`: the SSA population projection, with its reviewed SHA-256.
   - A clean policyengine-us checkout at the version the base build lists
     as `built_with_model_package`, installed in the environment. A local
     build lists no compatible-package ranges, so the projection must run on
     the base's own policyengine-us and policyengine-core versions.
2. **Build the candidate:**

   ```sh
   uv run --no-sync python -m microcosm.build.us_annual_static_aging \
     --base-build-dir "$PKG/releases/microcosm-us-2024-local" \
     --base-artifact-root "$PKG" \
     --ssa-csv "$SSA_CSV" --ssa-sha256 "$SSA_SHA256" \
     --model-source "$MODEL_SOURCE" --model-commit "$MODEL_COMMIT" \
     --model-version "$MODEL_VERSION" \
     --output-dir "$CANDIDATE" \
     --base-year 2024 --end-year 2027 --years 2026 2027
   ```

3. **Compute each year's household measures** on
   `$CANDIDATE/microcosm_us_2026_acs_local.h5` and
   `$CANDIDATE/microcosm_us_2027_acs_local.h5`. This projection builder does
   not compute or attach those measures.
4. **Write the acceptance report.** Run the independent acceptance checks on
   2024, 2026 and 2027. Write a passing schema-1
   `us_annual_projection_acceptance` report bound to the SHA-256 of
   `$CANDIDATE/annual_manifest.json`, as in the national guide's step 3.
   This branch validates that report; it provides no acceptance-runner CLI.
5. **Assemble the build:**

   ```sh
   uv run --no-sync python tools/assemble_us_annual_release_dir.py \
     --base-build-dir "$PKG/releases/microcosm-us-2024-local" \
     --base-artifact-root "$PKG" \
     --candidate-dir "$CANDIDATE" --acceptance "$ACCEPTANCE_REPORT" \
     --out "$ASSEMBLY_DIR"
   ```

   This writes `$ASSEMBLY_DIR/releases/microcosm-us-2024-local-annual/`.
   It prints the build and the publish and promote commands.

   The assembler copies the annual build's release and build manifest
   claims (reviewed limitations, package versions, gates and the rest) and
   every artifact entry from the base build's local folder. Before the
   hand-over, the publish command downloads both manifests of the base
   build from the Hub into a new, empty cache, never from the Hub client's
   own cache, which returns a cached file whose ETag is unchanged without
   hashing it. It refuses the annual build unless every claim it inherited
   equals the published one and every artifact entry it carries, kind and
   role included, equals the published entry. It also refuses any artifact
   that neither the published base build nor the annual projection declares
   (the projection manifest, the acceptance report, and the H5 and receipt
   of each projected year `metadata.dataset_years` declares), such as a
   file added to the base folder after publication. Last, it refuses a base
   build that the repository's `releases.json` has withdrawn or does not
   list, as for a national base, reading `releases.json` into a new, empty
   cache as well, so a copy cached before a withdrawal cannot stand in for
   it when the Hub cannot be asked. Assemble from an unedited copy of the
   published base build.

**Memory.** Let S be the base frame's size in memory. The H5 is about 10 GB
on disk, and fixed-format numeric blocks load at about their stored size, so
S is about 10 GB, or more if its text columns expand in memory. The peak
comes inside the per-year export. It holds about five full copies of the
tables:

- the frame;
- the export copy;
- the base-year dataset;
- the year's tables;
- the year's dataset.

On top of those sit the slim aging frame, a fraction of S, and about 2 GB of
cell counts and engine objects. That comes to roughly 5.3 × S + 2 GB, or 55
to 80 GB. Run it alone on the 128 GiB host. The calibration result's frame
shares the calibrated frame's tables instead of copying them, and the result is
released before the export. The write and the loader check each hold
about 2.5 × S.

**Wall time.** Roughly 30 to 60 minutes for two years on an otherwise idle
host. This is an estimate; the builder has not been run at this scale. The
setup takes about 5 to 10 minutes: loading and hashing the base, building the
tax-benefit system, and the uprating map. Each year takes about 12 to 25
minutes:

- calibration, 300 epochs over 1.6 million household weights and 164 cells,
  a few minutes;
- the export copies and the carried-column check;
- the fixed-format write and its round-trip read;
- the native reload.
