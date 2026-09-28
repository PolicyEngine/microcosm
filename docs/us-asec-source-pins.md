# US base build: pinned CPS ASEC inputs

The base stage of `tools/build_us_puf_support_base.py` reads processed CPS
ASEC files, one per income year, through `--asec-h5 YEAR=PATH`. Four income
years are pinned: 2022, 2023, 2024 and 2025. A new build pools the newest three
(the default pool). Until 18 September 2026 the 2022-2024 files existed in one
untracked directory on one build machine, and the build recorded their digests
without comparing them to anything.

Files are named by income year. The survey year is the income year plus one:
`census_cps_2025.h5` is the ASEC 2026 (`asecpub26csv.zip`).

## Where the files live

The files are in the public Hugging Face dataset repository
[`policyengine/microcosm-us-sources`](https://huggingface.co/datasets/policyengine/microcosm-us-sources)
(CC BY 4.0; they derive from public Census Bureau files). Microcosm addresses
each file at the revision of the upload that added it, never by branch. A later
upload therefore cannot move what a build reads, and adding a year does not
change the URL an earlier year resolves to.

`microcosm.build.us_runtime.asec_sources` owns the coordinates:

| Income year | File | Revision | SHA-256 | Bytes |
|---|---|---|---|---|
| 2022 | `census_cps_2022.h5` | `78acc83e…` | `7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e` | 301,129,278 |
| 2023 | `census_cps_2023.h5` | `78acc83e…` | `cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88` | 299,036,610 |
| 2024 | `census_cps_2024.h5` | `78acc83e…` | `ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d` | 323,994,739 |
| 2025 | `census_cps_2025.h5` | `efe5e210…` | `4c5a32188b6acfbcfbeb9f5d719d3847873887b72ebdbb19bc402c16e0a64e58` | 304,753,967 |

The revisions are `78acc83ea8b099a97cb0d658bbed91ea75aae8b0` (2026-09-18: the
2022-2024 files, mirrored unchanged) and
`efe5e2107b252506ea9b868071967cf73cc410bc` (2026-09-27: the 2025 file; the
three earlier files' bytes are unchanged at this revision).

The 2022-2024 digests are also recorded as hermetic-input evidence in
`ecps_parity_known_gaps.json`, which describes the historical Build J inputs.
`test_us_asec_sources.py` asserts that the evidence and the pins agree for
every year the evidence names.

## The default pool

`asec_sources.ASEC_DEFAULT_POOL_INCOME_YEARS` is `(2023, 2024, 2025)`. It is
derived as the newest three pinned income years, not written out, so pinning a
2026 file moves it to `(2024, 2025, 2026)`.

Income year 2022 left the default on 27 September 2026, for two reasons:

- It is the oldest vintage.
- Its processed file carries only 2 of the 18 `NOW_*` at-interview coverage
  recodes (`NOW_GRP`, `NOW_MRK`; microcosm #720). Its reported-coverage inputs
  depend on `asec_census_person_columns` restoring reviewed recodes from the
  Census archive. The 2025 file carries all 18. Income year 2023 has the same
  gap and stays in the default until a newer year displaces it.

2022 stays pinned, fetchable and accepted by every registry, so a historical
income-2022..2024 build remains byte-reproducible. Pass its years explicitly.

A default-pool base build passes the three processed files with their pins
and, for offline runs, the three Census archives:

```bash
uv run python tools/fetch_us_asec_sources.py   # prints the --asec-h5 / --asec-h5-sha256 lines
PYTHONHASHSEED=0 uv run python tools/build_us_puf_support_base.py \
  --asec-h5 2023=… --asec-h5-sha256 2023=cb578173… \
  --asec-h5 2024=… --asec-h5-sha256 2024=ec36604c… \
  --asec-h5 2025=… --asec-h5-sha256 2025=4c5a3218… \
  --asec-education-source 2023=asecpub24csv.zip \
  --asec-education-source 2024=asecpub25csv.zip \
  --asec-education-source 2025=asecpub26csv.zip \
  --asec-2023-weeks-unemployed-source asecpub23csv.zip \
  …
```

`--asec-education-source` years without a mapping are fetched from Census and
verified against the same pins. The last flag is explained under "What a
2022-free pool still reads".

### Every pooled year needs four registries

A pooled income year needs its processed H5 and its Census survey archive. The
archive is used for four things: the `ED_VAL` and `PAW_TYP` sidecars, the SPM
independence role, and the #720 person-column restore. Each of these is pinned
per income year:

| Registry | Module | 2025 value (measured from `asecpub26csv.zip`) |
|---|---|---|
| `ASEC_SOURCE_ARTIFACTS` | `asec_sources` | the H5 above |
| `ASEC_EDUCATION_ASSISTANCE_ARCHIVES` | `education_assistance_source` | archive `fe819d0d…` (139,103,894 bytes); `pppub26.csv` `f7892069…` (259,887,437 bytes, crc32 `477a9e19`); 134,729 rows; 2,574 `ED_VAL` > 0 |
| `ASEC_SPM_ROLE_SOURCES` | `spm_role_source` | 55,401 SPM units |
| `ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS` | `public_assistance_type_source` | `PAW_TYP` counts (134,163, 317, 223, 26); 566 PAW-positive; 343 TANF-typed |

`test_every_pinned_year_is_pinned_in_every_year_keyed_registry` requires the
four key sets to be equal. A year missing from any of them therefore fails in
CI, not partway through a build. Each loader re-derives its values from the
archive and refuses a mismatch. Each one accepted the 2025 archive when run
against the local copy on 27 September 2026.

The loaders' default `income_years` is the default pool. Every caller in the
build passes the pooled years explicitly, so the default applies only to a
bare call.

## Income year 2025

`census_cps_2025.h5` was built on 27 September 2026 from
`asecpub26csv.zip` (139,103,894 bytes, SHA-256 `fe819d0d…`, Last-Modified 15
September 2026). It was built by the loader that produced the pinned 2024 file:
the archived US data package's `CensusCPS` at commit `40314a75`. That package
is archived, so it cannot take a PR. Instead, one commit on Max's fork of it
(`9f177ddd`, branch `census-cps-2025`) adds the income-year 2025 URL and stops
requesting `SPM_BBSUBVAL`. The experiment README links both commits.

The recipe, receipts and schema diff are in
[`experiments/us-asec-2025-source/`](../experiments/us-asec-2025-source/README.md).
In brief:

- **Loader choice.** Rebuilt from `asecpub25csv.zip`, `40314a75` reproduces
  every table of the pinned `census_cps_2024.h5` exactly. The later archived
  head `42ed5d45` adds seven person columns, among them `ED_VAL`, which
  `fill_asec_education_assistance_source` refuses to overwrite.
- **Schema.** Relative to 2024, only the Census-side changes appear:
  - `person` (161 columns) and `spm_unit` (38) lack `SPM_BBSUBVAL`, which the
    ASEC 2026 file no longer publishes. It is omitted, not zero-filled.
  - `household` (153) drops six broadband and child-care-problem columns and
    adds `HTCC_PROBLOSS` and `HXCC_PROBLOSS`.
  - No Microcosm reader uses any of these columns.
  - All 18 `NOW_*` recodes are present, with codes {1, 2}.
- **Size.** 53,344 households and 134,729 persons, against 55,762 and 142,125
  for 2024. The default pool holds 165,357 households and 421,119 persons; the
  old pool held 168,852 and 432,523.
- **Rotation.** 16,601 of the 53,344 households (31.1%; 31.5% weighted by
  `HSUP_WGT`) share an `H_IDNUM` with the income-2024 file. The links are real:
  the reference person's sex agrees on 98.3% of linked pairs, and their age
  advances by 0-2 years on 96.4%, against 49% and 5% for random pairs. 39,927
  persons (29.6%) share a `PERIDNUM`.
- **County identification.** 41.7% of households (45.4% weighted) have a
  nonzero `GTCO`, against 41.3% (45.3%) for 2024. The file codes 307 distinct
  counties; the income-2024 file codes 280.

## Fetching

`fetch_asec_source(year)` returns a verified local path. It first checks the
archived US data repository checkout's storage directory and
`~/.cache/microcosm/asec/`, to spare a 300 MB transfer on machines that already
hold the file. Otherwise it downloads the file at its own pinned revision
through `huggingface_hub`. Every candidate, the download included, must match
the pinned byte length and SHA-256 before it is returned; a download that does
not raises `ValueError`. Passing `cache_dir` skips the convenience copies and
downloads into that `huggingface_hub` cache.

`tools/fetch_us_asec_sources.py` resolves files and prints the builder's
arguments, one year per line:

- With no arguments it resolves the default pool.
- `--all-pinned` resolves every pinned year.
- Explicit years resolve exactly those years.

```bash
uv run python tools/fetch_us_asec_sources.py                 # 2023 2024 2025
uv run python tools/fetch_us_asec_sources.py 2022 2023 2024  # historical pool
```

## Verifying in the build

`--asec-h5-sha256 YEAR=SHA256`, passed once per year, makes the builder refuse
a `--asec-h5` input whose bytes differ from the declared digest. The check runs
in `main()` before any stage, so a wrong input refuses in seconds rather than
surfacing as drift hours later.

Every pin is parsed and checked before any file is read. Each of the following
refuses with a message naming the value:

- a pin without `--asec-h5`;
- a value that is not `YEAR=SHA256`;
- a year that is not an integer;
- a digest that is not 64 hexadecimal characters;
- a year named twice;
- a year that no `--asec-h5` mapping provides;
- for a year with a canonical pin, a declared digest that differs from it. The
  flag can narrow nothing and re-pin nothing, and 2025 now has a canonical pin.

Once any year is pinned, every `--asec-h5` year must be pinned, so a build is
either fully pinned or not pinned at all. Then each pinned file is hashed in
year order; a missing file and a digest mismatch refuse the same way.

The pins are locked into the checkpoint `run_config` (`asec_h5_sha256`, a
`{year: digest}` mapping with the digest lower-cased). A resume with different
pins refuses, and an equivalently spelled pin resumes. The raw values are
forwarded to every staged child process.

The flag is opt-in. Fixture-driven tests build from synthetic
`census_cps_*.h5` files and are not held to the production digests. A certified
from-scratch build passes it for every pooled year; nothing enforces that yet,
and the US release build rule is where it belongs.

## What a 2022-free pool still reads

The `source_construction` and `pre_clone_enrichment` stages ran on the real
2023/2024/2025 files, fully pinned, on 27 September 2026 (see the experiment
README). Things that still name income year 2022 or the old pool:

- **The LKWEEKS sidecar.** The builder loads the income-2022 sidecar
  `asecpub23csv.zip` in every mode. It fetches the file when
  `--asec-2023-weeks-unemployed-source` is omitted, and records it as the
  `LKWEEKS` source pin. For a pool without 2022 the repair fills no rows,
  because every other file carries `LKWEEKS`. The raw-stage checkpoint
  validator requires a non-empty `LKWEEKS` pin list, so the sidecar cannot
  simply be dropped. Pass the local archive when building offline.
- **The generated spec.** `us/spec/sources.yaml` `asec_raw_stage` still pins
  the income-2022..2024 raw-stage checkpoint (`51e9fafc…`) and its
  `asec_2022/2023/2024` authorities. That checkpoint is the certified lineage
  and is unchanged. The first default-pool base build produces the checkpoint
  to pin next.
- **The support-spine spec.** `us/support_spine.json` (spec mode, not the
  production path) declares two sources, the target year and the year before.
- **Money.** The pooled person tables carry nominal dollars of each income
  year, as they did for 2022-2024; no base-build stage restates them by source
  year. `target_aging` ages calibration targets, not survey records. The mean
  income year of the default pool is 2024, the target year; for the old pool
  it was 2023.
- **Prior-year income.** The cohort with no prior-year link inside the pool
  moves from 2022 to 2023.
- **Arrival year.** `immigration._ARRIVAL_YEAR_MIDPOINTS` is shared across
  vintages. Census re-bins `PEINUSYR`'s top codes every year:
  - ASEC 2025: code 28 = 2022-2025.
  - ASEC 2026: code 28 = 2022-2023, and a new code 29 = 2024-2026.

  28 → 2023 stays inside both intervals. 29 → 2024 clamps the 1,121 most
  recent arrivals of 2025 to the 2024 target year (zero years in the US).
