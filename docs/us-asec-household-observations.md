# ASEC household observations

`microcosm.build.us_runtime.asec_household_observations` attaches seven observed
household fields to an actual v4 ASEC source Frame: `HTOTVAL`, `H_LIVQRT`,
`HRHTYPE`, `H_HHTYPE`, `H_SEQ`, `H_TENURE`, and `GESTFIPS`. Output names have the
`asec_` prefix. All observations retain their exact signed int64 values, including
zero. The helper does not classify housing units, group quarters, or unknown
codes; convert prices; define an engine income concept; or evaluate a candidate.

## Input authority

The caller must authenticate the complete checkpoint file against its trusted
SHA256 and load that exact file with `load_asec_raw_stage_checkpoint_v4`. Pass the
resulting full Frame, metadata, and authenticated checkpoint SHA to
`with_asec_household_observations`. Its in-memory structural/v4 checks cannot
authenticate arbitrary caller-supplied observation values. A checksum string
alone is not proof that the caller performed the input-file check.

For every income year, pass an `AsecHouseholdObservationSource(year, path, sha256)`
whose year and hash match the loaded checkpoint's `source_receipt.sources`.
Identical files may be relocated: paths are locators, while years and content
hashes determine receipt identity. Missing or repeated years and changed pins
are refused. The original checkpoint and all raw H5s remain unchanged.

## Numeric-only reading and exact join

The raw household group must use the captured pandas fixed HDF format. The
h5py reader validates column metadata and numeric block shapes, then reads only
the selected ordinary int64 observations. It never reads raw person tables,
household pandas-index values, or unrelated object/VLARRAY blocks. External and
soft links, virtual datasets, and external raw storage on required datasets are
refused. Each source is hashed through the same open file before and after the
read; path and file identities must remain stable. This is a trusted local-file
contract, not a general HDF sandbox.

The exact join uses `(income year, H_SEQ)`. Actual person memberships must identify
one `(source_year, source_household_id)` per normalized household, and each raw
key may represent only one normalized household. Raw keys must be unique within
year and every referenced key must match. Joined `H_TENURE` and `GESTFIPS` must
equal the already carried household tenure and state. The original ordered,
sorted household IDs and all person/index associations are preserved. Extra raw
households are counted as unreferenced; they do not create population rows.

## Result and integration

`AsecHouseholdObservationResult.frame` is the augmented full source. Every
existing table column, index, typed weight, stratum, metadata field and mass
record is retained. Existing attachment columns cause refusal. The result is
not relabeled as the original v4 checkpoint content. `result.receipt` returns an
independent JSON-safe copy binding the declared input checkpoint hash, raw file
pins, reader schema, join counts, exact ordered household-ID digest and output
column digests. It contains no household observation values.

The initial ASEC-only columns are int64. A later union with another source must
represent absence explicitly, using the existing proved `Int64` promotion at the
graph storage boundary; absence is neither zero nor a housing-unit classification.
A separately reviewed price-alignment stage may produce a separately named
adjusted amount while retaining the raw observation. Nominal multiyear fiscal
leaves are not implicitly aligned by adding these columns.

This helper registers no source codec or graph nodes and writes no checkpoint.
The caller owns the augmented source's implementation identity, serialization,
source/price graph composition and later scientific comparability decisions.
Fixture checks establish engineering contracts only. They do not establish an
actual join's completeness, income-distribution accuracy, or release eligibility.
