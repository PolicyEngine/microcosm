# US base build: pinned CPS ASEC inputs

The base stage of `tools/build_us_puf_support_base.py` reads three processed
CPS ASEC files, one per income year 2022, 2023 and 2024, through
`--asec-h5 YEAR=PATH`. Until 18 September 2026 those files existed in one
untracked directory on one build machine, and the build recorded their
digests without comparing them to anything.

## Where the files live

They are mirrored, unchanged, to the public Hugging Face dataset repository
[`policyengine/microcosm-us-sources`](https://huggingface.co/datasets/policyengine/microcosm-us-sources)
(CC BY 4.0; they derive from public Census Bureau files). Microcosm addresses
them by upload revision, never by branch, so a later upload cannot move what
a build reads.

`microcosm.build.us_runtime.asec_sources` owns the coordinates:

| Income year | File | SHA-256 | Bytes |
|---|---|---|---|
| 2022 | `census_cps_2022.h5` | `7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e` | 301,129,278 |
| 2023 | `census_cps_2023.h5` | `cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88` | 299,036,610 |
| 2024 | `census_cps_2024.h5` | `ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d` | 323,994,739 |

Revision `78acc83ea8b099a97cb0d658bbed91ea75aae8b0`. The same digests are
recorded independently as hermetic-input evidence in
`ecps_parity_known_gaps.json`, and `test_us_asec_sources.py` asserts the two
rosters agree, so there is one source of truth.

## Fetching

`fetch_asec_source(year)` returns a verified local path. It checks the archived
US data repository checkout's storage directory and
`~/.cache/microcosm/asec/` first, to spare a 300 MB transfer on machines that
already hold the file, then downloads the pinned revision through
`huggingface_hub`. Every candidate, the download included, must match the
pinned byte length and SHA-256 before it is returned; a download that does not
raises `ValueError`. Passing `cache_dir` skips the convenience copies and
downloads into that `huggingface_hub` cache.

`tools/fetch_us_asec_sources.py` resolves all three and prints the builder's
arguments, one year per line:

```bash
uv run python tools/fetch_us_asec_sources.py
```

## Verifying in the build

`--asec-h5-sha256 YEAR=SHA256`, passed once per year, makes the builder refuse
a `--asec-h5` input whose bytes differ from the declared digest. The check runs
in `main()` before any stage, so a wrong input refuses in seconds rather than
surfacing as drift hours later. Every pin is parsed and checked before any
file is read: a pin without `--asec-h5`, a value that is not `YEAR=SHA256`, a
year that is not an integer, a digest that is not 64 hexadecimal characters, a
year named twice, a year no `--asec-h5` mapping provides, and, for a year with
a canonical pin, a declared digest that differs from it (the flag can narrow
nothing and re-pin nothing) all refuse with a message naming the value. Once
any year is pinned, every `--asec-h5` year must be pinned, so a build is either
fully pinned or not pinned at all. Then each pinned file is hashed in year
order; a missing file and a digest mismatch refuse the same way. The pins are
locked into the checkpoint `run_config` (`asec_h5_sha256`, a `{year: digest}`
mapping with the digest lower-cased) so a resume with different pins refuses
and an equivalently spelled pin resumes, and the raw values are forwarded to
every staged child process.

The flag is opt-in. Fixture-driven tests build from synthetic
`census_cps_*.h5` files and are not held to the production digests. A
certified from-scratch build passes it for all three years; nothing enforces
that yet, and the US release build rule is where it belongs.

## Building a processed file

The pinned files were written by the archived US data package's `CensusCPS`
loader, which takes no changes. Microcosm now builds them itself:
`microcosm.build.us_runtime.asec_processed_source` is the library, and
`tools/build_us_asec_processed_source.py` the command line.

```bash
uv run python tools/build_us_asec_processed_source.py \
    --income-year 2024 --output /tmp/census_cps_2024.h5 \
    --archive ~/data/asecpub25csv.zip --compare-to-pinned \
    --receipt /tmp/build_2024.json
```

### Input

The builder reads the Census public-use archive for survey year
`income_year + 1` (`asecpubYYcsv.zip`). Before it reads any member, it checks
the archive's byte length and SHA-256, and its person member's name, byte
length and CRC-32, against the income year's pin in
`education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES`. The person
file's row count must also equal the pinned count. A year without a pin is
refused. Without `--archive`, the archive comes from the education-assistance
cache (`~/.cache/microcosm/cps/asec_education`), which downloads and verifies
it if it is absent.

### Output

It writes the loader's five tables, as pandas `HDFStore` fixed-format frames,
in the loader's order:

| Table | Rows | Columns |
|---|---|---|
| `person` | every person record | the requested person, SPM and tax columns the file carries, in file order; then `CENSUS_TAX_ID` |
| `family` | families with a person record | every family column |
| `household` | households with a person record | every household column |
| `tax_unit` | one per constructed tax unit | `TAX_ID`, ascending from 1 |
| `spm_unit` | one per SPM unit, indexed by `SPM_ID` | the SPM columns, from the unit's first person row |

- **Missing values.** Empty fields are 0.
- **Optional columns.** An optional person column the file lacks (a `NOW_*`
  coverage recode or `PTOTVAL`) is added as zeros. Any other missing column
  refuses the build and names the column.
- **`SPM_BBSUBVAL`.** It is requested only for income years 2021-2024, the
  files that publish it. For income year 2025 it is omitted, not zero-filled.
- **`TAX_ID`.** `CENSUS_TAX_ID` keeps Census's tax unit, and `TAX_ID` is the
  constructed one.

### Tax units

Tax units come from `microunit.construct_tax_units` in its `policyengine`
mode. It is the constructor `microcosm.frame.units` already uses, and a
refactor of the construction the loader called. Within each household, it
forms married spouse-present pairs and a unit for each other person 18 or
older or ever married, then assigns dependents to those units. The qualifying-relative gross income limit is the IRS exemption
amount for the income year. The builder refuses a year for which microunit
publishes no amount: $5,050 for 2024, $5,200 for 2025 and $5,300 for 2026 are
published.

The construction has three preconditions it does not check. Breaking any of
them leaves a unit without a head, so it drops a person or raises
`StopIteration`:

1. Line numbers are unique within a household.
2. No person is in two married pairs. A married spouse-present person is
   paired with the member their `A_SPOUSE` names if that member is married
   spouse-present too, whether or not the pointer is returned. A chain (A
   names B, B names C) therefore puts B in two pairs.
3. No person names their own line as a parent.

The builder checks all three first (`audit_tax_unit_inputs`) and refuses
records that fail, naming the households. Every pinned Census year satisfies
them, with no one-sided spouse pointer and no pointer to an absent line. The
audit is recorded in each build receipt.

### pandas 3

The loader ran pandas 2.3, and Microcosm runs pandas 3. The builder keeps the
loader's column types in two places where pandas 3 would differ:

- **Identifiers.** `PERIDNUM` (22 digits) and `H_IDNUM` (20 digits) are read
  as text. pandas 3 parses digit strings too wide for 64 bits into Python
  integers and drops leading zeros; about 30% of `PERIDNUM` values start with
  `0`. Any other column that wide keeps its text too.
- **Text columns.** They are stored as `object` columns. A column in pandas
  3's string dtype would be stored as a one-dimensional block.

### Comparing with a pinned file

HDF5 files are not byte-deterministic, so a rebuild matches tables, not
SHA-256. `compare_asec_processed_stores` (and `--compare-to` or
`--compare-to-pinned`) compares two stores table by table, in two ways:

1. **As frames.** `assert_frame_equal(check_exact=True)`, plus per-column
   dtype and value checks.
2. **As HDF5 storage, column by column.** Each column must have the same
   atom, dataset attributes, block dimensions and bit-identical values. The
   column labels, the index and the frame attributes must also be identical.

How pandas groups same-dtype columns into blocks is reported, not required:
readers place columns by label, and pandas 3 groups household and SPM-unit
columns in a different order from pandas 2.3. The tool exits with status 1
when the stores differ. It refuses to overwrite a file that holds a pinned
input's bytes.

### Evidence

Receipts are in `experiments/us-asec-processed-source-builder/`.

| Income year | Against the pinned file |
|---|---|
| 2024 | All five tables equal. |
| 2025 | All five tables equal. |
| 2022, 2023 | `family`, `household` and `spm_unit` equal. `person` and `tax_unit` differ by design, because these files predate loader commit `40314a75`. |

For 2022 and 2023:

- Every pinned person column other than `TAX_ID` is equal, and in the same
  order.
- The rebuild's `CENSUS_TAX_ID` equals the pinned `TAX_ID`.
- The rebuild adds the columns the older list lacked.
- The pinned `tax_unit` is the older loader's per-Census-tax-unit sum of the
  ten tax columns.

The pins stay the build inputs. A rebuild of either year is a newer-schema
file, not a replacement.

On 20,000 seeded synthetic households, the builder's tax units equal the
archived construction's, run in its own environment, person by person for
income years 2022-2025. The synthetic households cover one-sided spouse
pointers, pointers to absent lines, invalid relationship codes and incomes
on both sides of each year's limit. CI checks the first 400 of them as golden
cases.

### Adding an ASEC year

For ASEC 2027 (income year 2026):

1. **Pin the archive.** Add `asecpub27csv.zip` to
   `ASEC_EDUCATION_ASSISTANCE_ARCHIVES`, with its size, SHA-256, person
   member, CRC-32 and row count. The other year-keyed registries need the year
   too.
2. **Check the limit.** microunit must publish the year's dependent gross
   income limit. $5,300 for 2026 is published.
3. **Control.** Rebuild the previous year with `--compare-to-pinned`; it must
   report every table equal.
4. **Build.** Build the new year. If Census dropped a requested column, the
   build refuses and names it. Decide from Census's change list whether to
   omit it for the new year, as `SPM_BBSUBVAL` is for 2025, or to stop.
5. **Pin the output.** Upload the file to `policyengine/microcosm-us-sources`
   in its own commit, and pin its bytes and revision in
   `ASEC_SOURCE_ARTIFACTS`.
