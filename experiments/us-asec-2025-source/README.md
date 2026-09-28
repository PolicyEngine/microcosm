# Income year 2025 ASEC source (`census_cps_2025.h5`)

Built and pinned on 27 September 2026. It is the processed ASEC 2026 file that
`asec_sources.ASEC_SOURCE_ARTIFACTS[2025]` names, and one third of the default
pool (2023, 2024, 2025). See `docs/us-asec-source-pins.md` for how builds use
it.

| | |
|---|---|
| Census archive | `https://www2.census.gov/programs-surveys/cps/datasets/2026/march/asecpub26csv.zip`, 139,103,894 bytes, SHA-256 `fe819d0d2fc4470c282e76c247b8aa8b6054811619730307e9f2fe6186707d93`, Last-Modified 15 September 2026 |
| Output | `census_cps_2025.h5`, 304,753,967 bytes, SHA-256 `4c5a32188b6acfbcfbeb9f5d719d3847873887b72ebdbb19bc402c16e0a64e58` |
| Hosted at | `policyengine/microcosm-us-sources` revision `efe5e2107b252506ea9b868071967cf73cc410bc` |
| Loader | `PolicyEngine/policyengine-us-data@40314a75` plus [`census_cps_2025.patch`](census_cps_2025.patch) (`MaxGhenis/policyengine-us-data@9f177ddd`, branch `census-cps-2025`) |

## Why this loader

The 2022-2024 files were made by the archived US data package's `CensusCPS`
loader, but nobody recorded which commit. The package is archived, so it takes
no PRs. To build 2025 "the same way as 2024", the loader revision that produced
the pinned 2024 file had to be identified first.

Three revisions rebuilt `census_cps_2024.h5` from a live download of
`asecpub25csv.zip`. The download's SHA-256 was `318845a2…`, equal to the
archive pinned in `education_assistance_source`. Each rebuild was compared with
the pinned file, table by table, with `assert_frame_equal(check_exact=True)`
([`receipts/loader_controls_2024.json`](receipts/loader_controls_2024.json)):

| Loader | Tables equal to pinned 2024 | Note |
|---|---|---|
| `42ed5d45` (archived head) | family, household, spm_unit, tax_unit | person has 7 extra columns: `PERRP`, `ED_VAL`, `FIN_VAL`, `NOW_GRPFTYP`, `NOW_HIPAID`, `NOW_OWNGRP`, `SRVS_VAL`; the 162 shared columns are equal |
| `40314a75` (#824, "Construct CPS tax units from household records") | all five | chosen |
| `cabe62d2` (the same change before merge) | all five | |

The archived head was not used, for two reasons. A 2025 file with its extra
columns would differ in shape from 2024. And its `ED_VAL` would make
`fill_asec_education_assistance_source` refuse ("found a preexisting ED_VAL
column with values").

Rebuilds reproduce tables, not bytes. All three rebuilds have the pinned file's
exact byte length but different SHA-256s, because HDF5 object metadata is not
byte-deterministic. That is why files are pinned by the uploaded bytes. A second
2025 build from the same archive, with the current
[`scripts/run_loader.py`](scripts/run_loader.py), again gave five identical
tables, the same length and a different digest
([`receipts/reproduction_2025.json`](receipts/reproduction_2025.json)).

## The loader patch

`pppub26.csv` no longer carries `SPM_BBSUBVAL`, the SPM unit's broadband
subsidy. The loader requested it for every income year after 2020, so an
unpatched run refuses with `KeyError: Missing required CPS person columns:
SPM_BBSUBVAL`. The patch makes three changes:

1. Add `CPS_URL_BY_YEAR[2025]` (`…/2026/march/asecpub26csv.zip`) and
   `CensusCPS_2025`.
2. Request `SPM_BBSUBVAL` only for income years 2021-2024
   (`SPM_BBSUBVAL_INCOME_YEARS`), the files that publish it. Before, the three
   places that built the SPM column list each had their own `<= 2020` test; one
   helper, `spm_unit_columns_for_year`, replaces them.
3. Add `tests/unit/datasets/test_census_cps_2025.py` (125 tests, all passing
   under the archived package's environment). Two of them are exhaustive over
   income years 1990-2100: every URL is the following survey year, and
   `SPM_BBSUBVAL` is requested in exactly 2021-2024. Two more run `generate()`
   end to end on a synthetic ASEC 2026 archive with the network mocked: 2025
   writes all five tables without `SPM_BBSUBVAL`, and 2024 still refuses
   without it. Run against the unpatched loader, the file fails to import.

The column is omitted rather than zero-filled. Census did not publish it for
2025, and nothing in Microcosm reads it. Microcosm's pooled reader loads only
the `person` and `household` tables, and `SPM_BBSUBVAL` does not match
microunit's `_VAL` pass-through rule.

## Reproduce

```bash
git -C policyengine-us-data fetch https://github.com/MaxGhenis/policyengine-us-data census-cps-2025
git -C policyengine-us-data checkout 9f177ddd6197dd0e296ad05d080865a3e5c2cdbf
cd policyengine-us-data && uv sync --locked --no-dev
PYTHONPATH=$PWD .venv/bin/python <microcosm>/experiments/us-asec-2025-source/scripts/run_loader.py 2025 /tmp/census_cps_2025.h5
uv run python <microcosm>/experiments/us-asec-2025-source/scripts/compare_stores.py ~/.cache/microcosm/asec/census_cps_2025.h5 /tmp/census_cps_2025.h5
```

Equivalently, apply `census_cps_2025.patch` to `40314a75`. Write to an
explicit path, never the package's `storage/` folder. Other builds hash the
files there.

## Schema diff against `census_cps_2024.h5`

Full output: [`receipts/schema_diff_2024_2025.json`](receipts/schema_diff_2024_2025.json)
(from [`scripts/schema_diff.py`](scripts/schema_diff.py)). No shared column
changed dtype.

| Table | Rows 2024 → 2025 | Columns | Dropped | Added |
|---|---|---|---|---|
| person | 142,125 → 134,729 | 162 → 161 | `SPM_BBSUBVAL` | none |
| household | 55,762 → 53,344 | 157 → 153 | `HBBSUB_MNTH`, `HBBSUB_YN`, `I_HBBSUBMNTH`, `I_HBBSUBYN`, `HECC_PROBTIME`, `HXCC_PROBTIME` | `HTCC_PROBLOSS`, `HXCC_PROBLOSS` |
| family | 62,479 → 59,262 | 85 → 85 | none | none |
| spm_unit | 58,147 → 55,401 | 39 → 38 | `SPM_BBSUBVAL` | none |
| tax_unit | 74,697 → 71,085 | 1 → 1 | none | none |

- **Renames.** None. Census's ASEC 2026 change list shows `HECC_PROBTIME`
  ("time lost from work due to child care problems", 0-999) as removed. It
  shows `HTCC_PROBLOSS` ("days of work lost due to child care issues",
  0-260) as added. That is a re-specified question, not a renamed column. The
  broadband-subsidy columns go with the end of the subsidy fields.
- **Raw-file changes the processed file never carried.** `pppub26.csv` also
  drops the twelve 5-year-migration fields (`M5G*`, `I_M5G1-3`). The loader
  never requested them, and no pinned H5 has them.
- **Column order.** Person, family, SPM unit and tax unit keep 2024's order.
  The household table's shared columns follow `hhpub26.csv`'s own order, which
  differs. Microcosm reads columns by name.
- **Value ranges.** Top codes and maxima move as they do every year:
  - 39 person, 24 household, 26 family and 15 SPM-unit numeric columns reach
    outside their 2024 range, among them income top codes (`WSAL_VAL`,
    `PTOTVAL`, `DIV_VAL`), `SPM_WEIGHT`/`HSUP_WGT` and SPM thresholds.
  - `H_YEAR`, `FILEDATE` and `YYYYMM` move by one survey year.
- **Code sets** (columns with at most 60 values).
  - `PEINUSYR` gains code 29, Census's yearly re-binning (ASEC 2025: 28 =
    2022-2025; ASEC 2026: 28 = 2022-2023, 29 = 2024-2026).
  - `GTCSA` gains 4 combined statistical areas, and `GTCBSASZ` gains a code.
  - The rest are household-size and line-number codes present in one year and
    not the other.
- **`NOW_*` coverage recodes (#720).** All 18 that the 2024 file carries are
  present, with codes {1, 2}. Weighted under-65 "yes" counts, 2024 → 2025, in
  millions:
  - `NOW_MCAID` 53.8 → 50.9
  - `NOW_CAID` 51.5 → 48.5
  - `NOW_GRP` 166.3 → 165.5
  - `NOW_MRK` 14.6 → 14.3
  - `NOW_PRIV` 194.3 → 193.1
  - `NOW_PUB` 60.7 → 58.0
  - `NOW_COV` 248.3 → 244.4

  The 2022 and 2023 files carry 2 of them.
- **Geography.**
  - `GESTFIPS`: all 51 state codes in both years.
  - `GTCO` nonzero: 41.7% of households, 45.4% weighted (2024: 41.3%, 45.3%).
  - `GTCBSA` nonzero: 74.2%, 82.0% weighted (2024: 75.4%, 82.5%); 280
    distinct codes (2024: 261).
  - `GTCSA` nonzero: 41.5% (2024: 42.3%).
- **`H_IDNUM`.** A 20-character string, unique per household, equal to the
  first 20 characters of every member's `PERIDNUM`.

## Measurements

[`receipts/measurements_2025.json`](receipts/measurements_2025.json), from
[`scripts/measure_pool.py`](scripts/measure_pool.py).

- **Households.** 53,344 (134,729 persons; 137.1 million weighted), against
  55,762 (142,125; 135.0 million) for income 2024. The default pool
  (2023/2024/2025) holds 165,357 households and 421,119 persons; the old pool
  (2022/2023/2024) held 168,852 and 432,523.
- **Rotation overlap with income year 2024.**
  - 16,601 households share an `H_IDNUM` with the income-2024 file: 31.1% of
    2025 households, 31.5% weighted by `HSUP_WGT`. 29.8% of 2024 households
    reappear.
  - The links are the same households. The reference person's sex agrees on
    98.3% of linked pairs, and their age advances by 0-2 years on 96.4%. For
    random pairs the figures are 49.0% and 4.6%.
  - Most linked pairs carry the same `H_MIS` value in both files, so `H_MIS`
    does not predict linkability here.
  - 39,927 persons (29.6% of 2025 persons) share a `PERIDNUM`.
- **County identification.**
  - 22,267 households have a nonzero `GTCO`: 41.7%, 45.4% weighted.
  - They code 307 distinct counties (income 2024: 280).
  - Against the ASEC 2026 identified-county list (`cpsmar26.pdf` List 4, 361
    counties), packaged on the location-v1 branch: 287 are coded and listed,
    20 coded but not listed, and 74 listed but not coded. The 20 include the
    four old Connecticut county codes, which List 4 no longer names.
  - The list is keyed by survey year, and income year 2025 selects survey year
    2026. The file carries every household column the location code reads:
    `H_SEQ`, `GESTFIPS`, `GTCO`, `GTCBSA`, `HSUP_WGT`, `H_NUMPER`.

## Pipeline smoke on the default pool

[`receipts/pipeline_smoke_2023_2025.json`](receipts/pipeline_smoke_2023_2025.json).
The run used `tools/build_us_puf_support_base.py` with `--stage
source_construction`, then `--stage pre_clone_enrichment`. It passed
`--asec-h5` and `--asec-h5-sha256` for 2023, 2024 and 2025, the three local
Census archives, and `PYTHONHASHSEED=0`. Both stages succeeded: 64 s and 11.0
GB peak, then 38 s and 13.6 GB.

- **Pins.** All three `--asec-h5-sha256` pins were verified before either stage
  ran; the 2025 pin now has a canonical value to match.
- **#720 restore.** It joined every person one-to-one: 144,265, 142,125 and
  134,729. For 2023 it added the 11 reviewed columns. For 2024 and 2025 every
  reviewed column the H5 carries equals the Census member.
- **Pool shares.** Equal thirds, anchored to the 2024 weighted population.
- **Signals.** Every pre-clone signal passed:
  - SPM independence role share 0.628; minor role share 0.018; composition
    PASS.
  - Relationship, eligibility, housing, Medicare take-up, pregnancy and WIC
    claim.
