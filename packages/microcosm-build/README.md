# microcosm-build

The build end of the microcosm stack: typed stage plans over
`microcosm.frame.Frame` with **declarative donor graphs** (every imputation
names its donor survey and fails loudly — no silent fallbacks), and the
**dataset acceptance gates** every release must pass before publishing:

- **parity** — every variable layer the incumbent populates, the candidate
  populates (no all-zero gaps on engine-known inputs);
- **support** — every imputed value lies inside its donor's realized range;
- **aggregate-vs-admin** — weighted aggregates land within declared tolerance
  of administrative anchors from the
  [target registry](../microcosm-calibrate/), with signs checked (this gate
  catches the class of failure where calibration silently drives net
  short-term capital gains to −$3.9T);
- **export surface** — every replacement artifact can prove that its
  exported variables match a reference surface, with only documented
  structural extras or reviewed exclusions (for UK, this is the eFRS
  compatibility check);
- **target surface** — the calibration target set covers the reference
  target surface and may only be wider, not narrower (for UK, Microcosm must
  calibrate to at least the eFRS target surface);
- **per-family fit** — the calibration's within-10% share is reported per
  source family, while only broad family-level misses block publication so
  one family cannot hide inside the global average;
- **rotated holdout** — deterministic target folds so *every* target is held
  out exactly once across rotations, instead of one lucky split.

All gate losses use the calibrator's capped weighted-MAPE helper
`weighted_mean(min(abs((estimate − target) / scale), cap))` — scorers consume
the same functions, so there is no calibrator-vs-scorer objective mismatch.

## Target source contract

Microcosm calibration targets are Ledger-owned. Production Microcosm builds must
materialize target values from Ledger target profiles before calibration; raw
ONS, HMRC, IRS, Census, DWP, or other administrative tables are source inputs
for Ledger, not direct calibration targets inside Microcosm. Temporary migration
harnesses may accept processed tables only when they are explicitly labelled
experimental and are not used to publish a production Microcosm artifact.

The `us` extra adds the rules engine for formula/export checks. Country source
loaders are not Python dependencies: source stages are declared in packaged
JSON manifests and executed by shared Microcosm runtimes.

Country namespaces under `microcosm.build.us` and `microcosm.build.uk` are
resource packages only. They may contain specs and data artifacts, but no Python
modules; guard tests enforce this so country content stays declarative.

## UK local-geography path

`microcosm.build.uk_runtime.local_rowwise` is the UK local-solve surface: one
weight per cloned household, each household assigned to exactly one area by
the OA geography ladder, so an area's target rows draw support only from the
households assigned there. The matrix builder fails closed when an assigned
area is missing from the target surface, and every solve runs under the
reviewed `UK_LOCAL_SOLVE_DOCTRINE` (declared loss cap and weight-ratio
stretch bound, no per-target knobs) with the microcosm#492 past-cap census on
every result. The pre-ladder stacked `areas x households` research harness
(`local_runner` and the stacked matrix/solve helpers) was removed with
microcosm#612 increment 2 — the rowwise clone path is the single local-solve
story, and positional-assignment safety now rides on the Frame kernel's
sorted-group-id invariants rather than a helper.

`microcosm.build.uk_runtime.local_geography` keeps the area-target alignment
contract (`align_area_targets`): target providers pass explicit area tables,
and the module aligns metric columns to a canonical area-code order without
importing the incumbent UK data package.

`microcosm.build.uk_runtime.local_targets` declares the constituency and local-authority
metric surface used by the local build: HMRC employment/self-employment amount
and count rows, ONS age bands, Universal Credit household rows, constituency
UC-by-children rows, and the LA income/tenure/rent rows. It accepts a
PolicyEngine-UK-like simulation object and returns household-indexed metric
tables; it still takes target values as explicit input tables. `local_solver`
wraps the Microcosm calibrator's log-weight optimizer for the rowwise solve and
records per-area/per-metric diagnostics on every result.

## UK firm generation

`microcosm.build.uk_runtime.firm_generation` is the experimental migration of
the UK firm generator from `PolicyEngine/firm-microsim-paper`. Its preferred
input is Ledger consumer facts reshaped with
`uk_firm_source_data_from_ledger_facts`; it draws a synthetic firm population,
calibrates firm weights through Microcosm's shared calibration optimizer, and
returns rows with Microcosm-style identifiers and `firm_weight`.
HMRC band tables select the configured `data_vintage` from a `Year` or
`Financial_Year` column; sector tables select a matching vintage column, or a
single available value when the caller supplies already-single-vintage source
data.

This is not yet the production firm microsimulation path: Ledger currently
covers the current paper target surface used by the adapter, including ONS
SIC-by-turnover, ONS SIC-by-employment, HMRC VAT-registered firms by SIC, and
HMRC net VAT liability by SIC. VAT liability is now an explicit rule-evaluator
input to generation; production runs should provide an Axiom RuleSpec artifact
through `AxiomVATRuleEvaluator`. The processed-table reader remains only for
paper-repository migration comparisons.

Build the row-wise local-geography H5 from a compact Microcosm UK H5 with:

```bash
uv run --project packages/microcosm-build --extra uk python \
  tools/build_uk_rowwise_dataset.py \
  --input-h5 /path/to/populace_uk_2023.h5 \
  --out /tmp/populace-uk-rowwise \
  --n-clones 2 \
  --constituency-codes /path/to/constituencies_2024.csv \
  --la-codes /path/to/local_authorities_2021.csv
```

If `--crosswalk` is omitted, the driver builds
`uk_official_geography_crosswalk.csv.gz` from public ONS, NRS, NISRA, and
postcode sources. It writes the cloned row-wise H5, a geography coverage CSV,
and `rowwise_build_manifest.json` with input/output hashes, row counts, target
coverage, weight preservation, and weakest local-support diagnostics.

## US plan status

`microcosm.build.us_runtime` declares the US build: stage order, donor graph with
citations (`US_DONORS`), the manifest-ready `BuildConfig`, and the packaged
source-stage manifest (`US_SOURCE_MANIFEST`). The stage *implementations* are
injected (`us_plan(implementations)`) and the plan refuses to assemble with any
stage missing — no stubs, no fallbacks. Release-specific benchmark comparison
harnesses live outside this repo.
