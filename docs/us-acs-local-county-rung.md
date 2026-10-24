# ACS local county calibration rung

This implements microcosm#922 item 1: an optional county rung on the US
ACS local release line. County logic lives in
`tools/us_acs_local_county_surface.py`; the release driver and district helper
only call its hooks. The initial surface uses ladder population and public IRS
county shares. Future ACS county families can use the same county row,
holdout, and reliability interfaces.

## Plan and invariants

- Population measures are household size times county membership. County
  ladder populations are normalized within each state to its state target.
- County SOI targets use within-state county fact shares of an already
  materialized state concept. County measures split that state's stored sparse
  row, so they require no additional rules-engine columns.
- Every household belongs to exactly one county within its state for each
  concept; county rows reconstruct the state row. County population targets
  and each SOI concept's targets sum to their state target.
- Missing counties, facts, or household support are reported and refused.
  Only an explicitly capped smoke may continue with missing support.
- Holdout roles are deterministic given their salt. Scoring compares held-out
  county targets with state target times county population share.
- Reliability reports show supporting rows, distinct source households, Kish
  ESS, training and holdout error. At fixed error, increasing ESS cannot worsen
  the grade. Missing holdout evidence is explicit.

Engine-free fixture and Hypothesis tests cover these invariants. No full
materialization, release build, publication, or annual projection is performed
as part of this implementation. The county rung applies to the base-year
ACS local calibration; 2026/2027 outputs still need the separate annual
projection and acceptance path.

## Surface and evidence

`--geographies state,cd,county` adds county population constraints; `cd` is
optional. `--soi-mode state_county` keeps the existing `state` administrative
surface and adds county SOI children. It requires both `state` and `county`
geographies and `soi` in `--families`. It does not add district SOI constraints.
The county column is configurable with `--county-column`; its default is
`county_fips`. Household codes must be integer five-digit county FIPS and must
agree with each household's `state_fips`.

The population measure is `household_size * 1[county]`. County targets aggregate
the ladder's `county_overlap_population` by `county_overlap_county`, then scale
each state's counties to that state's PUMA population total. Floating-point
rounding residuals go to the largest county. Census PEP county facts are useful
future evidence, but do not replace these default ladder targets.

County SOI shares use the same measure and source period as an unsliced,
all-return state parent:

```text
county share = county source value / sum(county source values in its state)
county target = existing state target * county share
county measure row = existing state CSR measure row * 1[county]
```

The source package currently carries two eligible concepts. Both already have
state parents, so the additional engine-column count is zero. `state_county`
requires both concepts, complete county facts and matching state parents for
every state, even when an entire concept or parent block is absent:

| Source measure ID | Concept | Unit |
|---|---|---|
| `return_count` | `irs_soi.individual_income_tax_returns` | count |
| `adjusted_gross_income` | `us:statutes/26/62#adjusted_gross_income`, source concept `irs_soi.adjusted_gross_income` | USD |

The feed has already converted the AGI publisher values from thousands of USD
to USD; the county rung does not apply another scale factor. It refuses
duplicate facts, negative or nonfinite shares, zero share denominators,
incomplete county inventories, incompatible boundaries, and missing or
ambiguous state parents. No unsupported concept receives a new engine measure.
Full runs also require the ladder to declare all 3,144 county-equivalents;
matching omissions from both ladder and SOI cannot pass as a complete release.
An ACS county family can later provide child targets with the same state-parent
metadata and reuse the row splitting, roles, and diagnostics interfaces. That
extension must declare its own measure semantics and complete county inventory.

Holdout assignment hashes whole `state × concept family` blocks with the salt,
so sibling county targets in a state never reveal a held-out county share
through their sum. Population rows remain training constraints. The defaults
are `--county-holdout-fraction 0.1` and
`--county-holdout-salt acs-local-county-v1`. The report compares calibrated and
design-weight estimates against held-out county targets, alongside the
pro-rata baseline `state target * county population / state population`.

`county_support.json` names unsupported counties before materialization.
`county_surface.json` records the SOI reconciliation or its refusal reason.
A county with no household support is refused unless both
`--allow-partial-geography` and a development `--sample-fraction` below 1 are
explicitly supplied. Positive SOI targets with no nonzero household measure
are also reported and refused outside that smoke exception. The exception
does not excuse missing source facts or mismatched county boundaries, and
development samples cannot be packaged as releases.

Calibration writes schema-2 `county_reliability.json` next to its calibration
outputs and includes it in the packaged evidence. Each county reports its
supporting household rows, distinct `(household_spine, household_source_id)` pairs when
available, and training/holdout target counts and mean/maximum absolute relative
errors. `kish_ess` aggregates final weights by those source pairs within each
county before calculating ESS; grades use this source ESS. `kish_ess_rows`
separately labels physical-row ESS, including clones. For example, 200 equal
donors each split into two clones have source ESS 200 (grade B at adequate fit)
and row ESS 400; cloning cannot earn grade A. Missing or null source provenance
makes source ESS unavailable and the grade U. The report binds the run identity,
calibrated weights,
and configured county column. Unsupported smoke counties remain explicit with
zero support and grade `U`.

For each target, relative error is
`abs(estimate - target) / max(abs(target), 1)`. The grade uses the maximum error
across all of the county's training and holdout targets:

The A/B/C/D thresholds are **provisional** and recorded as such in the report.

| Grade | Minimum source Kish ESS | Maximum error |
|---|---:|---:|
| A | 400 | 0.10 |
| B | 100 | 0.20 |
| C | 30 | 0.50 |
| D | otherwise, with holdout evidence | otherwise |
| U | any | no county holdout or source-identity evidence |

Rules are evaluated from A through C, then D; a county without holdout evidence
receives U even if its training fit is good, as does a county without source
identity. Thus a 10% block holdout can leave
many counties ungraded when only two concepts are available. At fixed error,
raising ESS cannot worsen a grade. These are report-only evidence grades;
Kish ESS is not a county standard error or a guarantee of unbiased estimates.

## Audited feed and remaining input work

The release's packaged pin is
`packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json`, pointing
to Chronicle commit `51aa40fd28e2ca81937752515e773d123fecfb71`. The corresponding
local feed is
`/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_51aa40f.jsonl`.
The audit streamed its actual bytes: 40,178 facts, 169,038,757 bytes, SHA256
`965a29ac9458edb58a05184917ee4a19772fd3dc7b4588db6eb6da5803641aac`.
It contains **zero county facts**: 23,968 congressional-district, 12,251 state,
and 3,959 country facts. The historical release recipe uses a `--feed`
placeholder; that recipe does not prove county facts were exported.

Both county source packages already exist at the pinned Chronicle commit.
They are omitted by the explicit `chronicle_feed_scope.json` selection. A
reviewed feed extension needs the following additional pairs:

```json
[
  {
    "record_set_id": "irs_soi.ty2022.county_totals",
    "period_type": "tax_year",
    "period_value": "2022",
    "package_id": "soi-county-2022",
    "package": "packages/irs_soi/county_2022",
    "build_year": 2022
  },
  {
    "record_set_id": "census_pep.vintage2024.county_population",
    "period_type": "calendar_year",
    "period_value": "2024",
    "package_id": "census-pep-county-population-2024",
    "package": "packages/census/pep_county_2024",
    "build_year": 2024
  }
]
```

Add the packages to their build-year `runs` entries, update the scope pair
count, rebuild with `tools/build_us_chronicle_feed.py`, and review/update the
feed and scope hashes and fact-to-target parity/fencing evidence. The current
fiscal compiler excludes county SOI geography; this rung consumes the raw
county facts and splits existing state measures. Population defaults still
use the ladder. Future ACS county packages require an additional scoped feed
extension. This implementation performs **no feed or scope mutation** and
does not claim an available county feed hash.

The inspected Chronicle publisher inputs have two concrete blockers:

- **Kalawao:** the SOI CSV has 3,143 counties and omits `15005` Kalawao County;
  the PEP CSV has all 3,144. A complete county SOI block currently fails with
  that county named. A source correction or an explicitly reviewed county
  treatment is needed before a full SOI run; this implementation introduces
  no fallback and does not silently borrow Maui's facts.
- **Connecticut:** both packages use the nine planning-region codes `09110`
  through `09190`. The household county column and ladder must use those same
  boundaries. Select `--county-column county_fips_2023` when the parallel
  planning-region work supplies that column, and use its matching ladder.
  Changing the column name alone cannot translate legacy county targets.

SOI facts carry `geography.level = "county"` and Census-style IDs such as
`0500000US01001`; the row geography is authoritative even though the record
set itself groups all counties. SOI geography metadata says vintage `2022`,
PEP says `2024`; actual county-code inventories must agree across feed,
ladder, and households. The audited raw publisher artifact hashes are
`2c04ff235bcbacb990b316bf79c7c628089a9a4c1dd9593ac51f74a109b13343`
for `22incyallnoagi.csv` and
`abcc8720d669e793bbfdcd440eeec37a78db3b452adbe4ccd1eadf7c72b522b9`
for `co-est2024-alldata.csv`.

## Full-run requirements and command

A full run needs a reviewed county-inclusive Ledger feed and its actual SHA256,
a staging H5 and summary with household support in every county, a matching
county-boundary PUMA ladder, and resolution of the Kalawao SOI gap. The
historical staging file was approximately 10.7 GB. The historical ladder was
446,791 bytes with SHA256
`39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8`;
that old digest is reference evidence, not an approved planning-region ladder.

At 3,144 counties and the current two SOI concepts, the county increment is
`3,144 × (population + returns + AGI) = 9,432` rows. With the pinned 4,380 state
administrative rows, 51 state population rows and 436 district population
rows, this is approximately **14,299 rows** including holdouts. The exact
count depends on the reviewed feed and ladder; an incomplete SOI county block
is refused rather than reducing that count silently.

For the historical 1,588,854-household staging population, each household can
contribute to at most three added county rows. That gives at most **4,766,562
added nnz**, approximately **38.2 MB** of raw float32/int32 CSR data and indices,
plus about 38 KB of row pointers. Zero measures reduce nnz. Temporary matrix
copies and Python overhead add memory, and existing state rows and the
engine/frame dominate the total. Historical full materialization peaked at
75.6–93.9 GB; calibration was about 67.6 GB. Allocate **128 GiB** and start with
`--hh-chunk 20000` pending a measured county run. These are planning estimates,
not a county-build benchmark. The implementation does not allocate a dense
households-by-county-target matrix.

Once the input blockers are resolved, set these paths and the new reviewed
feed hash in the execution environment. This example uses the planning-region
column; use `county_fips` only with a feed and ladder matching that column.
The following is documentation only and has not been executed:

County rows activate a separate solver profile: `--county-target-loss-cap 10`
and `--county-max-weight-ratio 5` by default. The calibrator accepts a scalar
loss cap, so the override applies to the joint run, including tax targets.
Without county geographies or `--soi-mode state_county`, the historical loss
cap of 1 and per-record weight cap of 5 (or their explicit overrides) remain
unchanged. Later calibration stages resolve the profile from the checkpoint's
recorded surface. Effective values and the run-level scope are stamped into
the calibration summary, resume settings, build manifest and refresh recipe.

Start with loss cap **10** and weight cap **5**, then compare a weight cap of
**20**. The [county pilot (microcosm#1377)](https://github.com/PolicyEngine/microcosm/pull/1377),
[`experiments/us-county-pilot-20261024/README.md` on `lwi-county-pilot`](https://github.com/PolicyEngine/microcosm/blob/lwi-county-pilot/experiments/us-county-pilot-20261024/README.md),
found that 297 counties (9.4%) started more than 100% off and received no
corrective gradient under cap 1. Cap 10 left only one such county. The 5×
weight cap bound; with loss cap 10 and weight cap 20, 98–99% of counties were
within 10% on the trained families. The pilot omitted the release's tax
families, and 57,240 donor households carried 68% of the starting weight.
The **tax-target cost is unmeasured until the joint solve runs**; these are
starting settings, not evidence that the production tradeoff is acceptable.

```bash
export COUNTY_FEED=/path/to/consumer_facts_county.jsonl
export COUNTY_FEED_SHA256='<new-reviewed-64-hex-feed-sha256>'
export COUNTY_STAGING_H5=/path/to/acs_multispine_staging.h5
export COUNTY_STAGING_SUMMARY=/path/to/acs_multispine_staging.summary.json
export COUNTY_LADDER=/path/to/matching_us_puma_ladder.npz
export COUNTY_RUN=/path/to/new/county-run
export COUNTY_COLUMN=county_fips_2023

uv run python tools/build_us_acs_local_release.py \
  --stage all \
  --staging-h5 "$COUNTY_STAGING_H5" \
  --staging-summary "$COUNTY_STAGING_SUMMARY" \
  --feed "$COUNTY_FEED" --feed-sha256 "$COUNTY_FEED_SHA256" \
  --ladder "$COUNTY_LADDER" \
  --families snap,medicaid,soi --geographies state,cd,county \
  --soi-mode state_county --county-column "$COUNTY_COLUMN" \
  --county-holdout-fraction 0.1 --county-holdout-salt acs-local-county-v1 \
  --county-target-loss-cap 10 --county-max-weight-ratio 5 \
  --sample-fraction 1.0 --hh-chunk 20000 \
  --checkpoint-dir "$COUNTY_RUN/checkpoints" \
  --out-h5 "$COUNTY_RUN/populace_us_2024_acs_local.h5" \
  --out "$COUNTY_RUN/release"
```

`--stage all` runs materialize, calibrate, QA, finalize, and local packaging.
It does not publish. The command deliberately omits the partial-geography
escape hatch. Run setup first with `uv sync --all-packages --extra us` in the
intended numerical environment. The recorded refresh recipe preserves the
geographies, county column, holdout fraction and salt.

The driver fixes `PERIOD = 2024` and has no `--year` switch. The command builds
a **2024 base candidate**, not the contracted 2026/2027 files. Those years need
the separate [annual projection and acceptance path](us-annual-static-aging.md)
with year-specific target evidence and renewed county consistency,
support/ESS, training/holdout, and reliability checks. Annual projection changes
weights, so the base county fit does not establish future-year county fit.

Engine-free fixture and Hypothesis coverage belongs in the existing
`packages/microcosm-build/tests/engine_free/us/` layout. Run the focused county
tests with at most two workers, the related release/district tests, the test
layout verifier, `uv run ruff check .`, and `uv run ruff format --check` on
touched Python files; report actual results in the PR.
No release build, full materialization, annual projection, publication, or
county-inclusive feed regeneration is part of this implementation's validation.

## Validation record

Validated in a separate `.venv-engine-free` environment with no country engine
installed, after the requested US setup completed:

- County fixture suite: **36 passed**, including five Hypothesis properties
  with up to 60 generated examples each. The materialization smoke uses three
  fixture households, a fake engine pass, and real sparse checkpoint/H5 I/O.
- Existing release, sparse district, district property, geography-label, and
  loss-weight suites: **125 passed, 2 skipped**, using `-n 2 --dist loadfile`.
  The skips are optional checks requiring an explicitly configured pinned
  Chronicle feed; the actual feed inventory was audited separately above.
- `uv run ruff check .`: clean.
- `uv run --no-sync python tools/ci_test_plan.py verify`: clean, 772 test files
  in registered directories.

The tests cover county/state conservation, exact household partitioning and
parent-row reuse, complete source blocks and support refusal, deterministic
holdout without training leakage, monotone grades, custom county-column
persistence, state/concept loss budgets, and current-versus-stale report
packaging. No production materialization, release build, annual projection,
feed rebuild, or publication was run.
