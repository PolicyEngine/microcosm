# US congressional-district reference inventory

`microcosm.build.us_runtime.cd_reference` builds a local inventory of public
reference observations and their uncertainty. It accepts no candidate population,
computes no candidate score, and activates no calibration targets or tolerances.
This is a reference preparation step, not evidence of district accuracy.

## Sources and scope

The producer captures the complete 2024 ACS one-year S0101 subject table and
B01001, B19001, and B25003 detailed tables for congressional districts. It retains
each table's metadata and every published estimate, margin of error, and estimate
and MOE annotation. The scope includes the 50 states and DC; Puerto Rico is
explicitly excluded. The [2024 ACS geography contract](https://www.census.gov/programs-surveys/acs/geography-acs/geography-boundaries-by-year/2024.html)
uses the 119th Congress. DC's published district code 98 maps to the existing
Microcosm canonical code 00 while the original code remains in the inventory.

S0101's subject-table groups remain distinct from B01001's detailed sex-by-age
groups. B19001 and B25003, including derived versions, are reserved as holdout
blocks and unavailable for calibration targets or tuning before candidate
scoring. These are **same-source** holdouts: ACS tabulations and ACS PUMS share
a survey source. All four tables remain inventory-only in this producer.

The IRS inputs are the official tax-year 2022 [CSV](https://www.irs.gov/pub/irs-soi/22incd.csv),
[workbook](https://www.irs.gov/pub/irs-soi/22incdall.xlsx), and
[guide](https://www.irs.gov/pub/irs-soi/22incddocguide.docx). A supplied
`source-index.json` must bind the exact CSV and XLSX bytes to their official URLs,
SHA-256 digests, and sizes. `22incddocguide.sha256` must bind the guide. The
producer extracts text directly from the pinned DOCX XML and records its parent
digest. It does not rely on third-party summaries.

## Uncertainty and disclosure

ACS margins of error have [90% confidence](https://www.census.gov/programs-surveys/acs/methodology/sample-size-and-data-quality/sample-size-definitions.html).
The inventory preserves their original values; it performs no variance conversion
or covariance approximation. [Annotations take precedence](https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-api-variable-types.html)
over estimate/MOE values. Known numeric sentinels, controlled-estimate markers,
open intervals, missing cells, and unknown annotations have separate classes.
Conflicting annotations and sentinels are flagged. Controlled MOE sentinels are
retained rather than silently replaced by zero.

The IRS guide describes a population in section C, while footnote 1 identifies an
SOI sample input. Published cell variance is not supplied, so the inventory records
`not_published_needs_method_resolution`, without assigning zero variance. A numeric
CSV zero does not prove a cell is unsuppressed. Workbook disclosure markers remain
located by sheet XML part and cell, separately from CSV AGI rows; no unsupported
join between those representations is attempted. Monetary units are thousands of
dollars, but that label does not establish a rounding increment.

The guide lists AGI codes 0–10 while the supplied official CSV contains codes 0–9.
Raw codes and their row counts remain visible. Income-bin labels are withheld
pending reconciliation of this discrepancy.

## Geography and lineage

IRS observations use the 117th Congress. National `STATEFIPS=00, CD=00` rows,
state totals with `CD=00`, and published district rows have distinct scopes.
The eight 117th-Congress at-large states/DC use explicitly labeled state-total
proxies, not fabricated published district rows.

The packaged 117th-to-119th crosswalk is verified against its provenance digest
and vintages. Outgoing source weights must agree with pair populations. To describe
each target district's composition, the inventory divides each incoming
`pair_population` by that **target's** total pair population. It records the largest
incoming share and its complement. These describe alignment exposure using 2020
block population; they are not estimates or upper bounds of fiscal allocation
error. Outgoing source weights cannot answer this target-mixture question.

The inventory reports source and target coverage gaps. Published observations,
crosswalked values, and state-rescaled values have separate lineage statuses.
Only published observations are included; this producer does not create either
transformed value product. Published table totals are not asserted to be additive
across suppressed, combined, or differently assigned source cells.

## Local acquisition and replay

From an installed Microcosm workspace (Python 3.13 or later):

```sh
uv run --no-sync python -m microcosm.build.us_runtime.cd_reference \
  --output-dir /absolute/path/reference-bundle \
  --irs-authority-dir /absolute/path/irs-authority \
  --crosswalk packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv \
  --crosswalk-provenance packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv.provenance.json \
  --census-key-service CENSUS_API_KEY \
  --census-key-account YOUR_ACCOUNT
```

The optional credential is read through a captured `agent-secret` subprocess and
kept in memory. Only fixed Census requests are permitted. Redirects are refused;
transport errors omit URLs and remote error text. Credentials and keyed URLs are
never written to the bundle. Unauthenticated public acquisition is available by
omitting both credential flags.

Each request is captured once into `raw/<content-sha256>.json`, with a key-free
request descriptor in `requests/`. Descriptors record the retrieval time separately
from the raw content digest and the fixed source year/dataset. Existing captures
are verified rather than fetched again. Add `--offline` to replay without reading
a credential or using the network. Keep the pinned IRS and crosswalk inputs
available at the supplied paths. A changed input or corrupted cache fails closed;
use a new output directory for a deliberate source refresh.

`reference-inventory.json` contains the complete normalized inventory and pins.
It is a local evidence artifact, not a frontend payload. A downstream consumer can
project bounded aggregate views while retaining this bundle's digest and source
identities. Reference readiness is separate from candidate execution status,
release eligibility, and heldout evaluation results.

## Verification

```sh
uv run --no-sync pytest -q \
  packages/microcosm-build/tests/test_us_cd_reference.py \
  packages/microcosm-build/tests/test_us_cd_reference_sources.py
```

Tests use tiny public-shaped fixtures. They cover annotation precedence,
suppression classes, table separation and holdout reservations, DC/PR scope,
national/state/proxy distinctions, incoming-target shares, source pin corruption,
credential-safe transport, and byte-identical offline replay. They do not execute
microsimulation or evaluate a candidate population.
