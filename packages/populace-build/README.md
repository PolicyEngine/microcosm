# populace-build

The build end of the populace stack: typed stage plans over
`populace.frame.Frame` with **declarative donor graphs** (every imputation
names its donor survey and fails loudly — no silent fallbacks), and the
**dataset acceptance gates** every release must pass before publishing:

- **parity** — every variable layer the incumbent populates, the candidate
  populates (no all-zero gaps on engine-known inputs);
- **support** — every imputed value lies inside its donor's realized range;
- **aggregate-vs-admin** — weighted aggregates land within declared tolerance
  of administrative anchors from the
  [target registry](../populace-calibrate/), with signs checked (this gate
  catches the class of failure where calibration silently drives net
  short-term capital gains to −$3.9T);
- **export surface** — every replacement artifact can prove that its
  exported variables match a reference surface, with only documented
  structural extras or reviewed exclusions;
- **target surface** — the calibration target set covers the reference
  target surface and may only be wider, not narrower;
- **per-family fit** — the calibration's within-10% share is reported per
  source family, while only broad family-level misses block publication so
  one family cannot hide inside the global average;
- **rotated holdout** — deterministic target folds so *every* target is held
  out exactly once across rotations, instead of one lucky split.

All gate losses use the calibrator's capped weighted-MAPE helper
`weighted_mean(min(abs((estimate − target) / scale), cap))` — scorers consume
the same functions, so there is no calibrator-vs-scorer objective mismatch.

The `us` extra adds the rules engine for formula/export checks. Country source
loaders are not Python dependencies: source stages are declared in packaged
JSON manifests and executed by shared Populace runtimes.

Country namespaces under `populace.build.us` and `populace.build.uk` are
resource packages only. They may contain specs and data artifacts, but no Python
modules; guard tests enforce this so country content stays declarative.

## UK local-geography path

`populace.build.uk_runtime.local_geography` holds the Populace-owned replacement shape
for UK constituency and local-authority geography. It supports both the stacked
local-area layout used by the US local ECPS flow and an assigned row-wise path
for UK builds that have a finest-available area assignment:

```text
stacked:  column = area_index * n_households + household_index
assigned: column = household_index, and target rows only see households assigned to that area code
```

The solved weights export to a long sidecar with `(area_type, area_code,
household_id, weight)` rows plus source-year/source-household lineage. This is
the format PolicyEngine can group by directly for constituency and local
authority outputs, and it avoids preserving the legacy dense
`areas x households` matrix artifact.

The module does not import an incumbent UK data package. Engine runners and
target providers pass household metric tables and aligned target tables into
`build_stacked_local_matrix`, `build_assigned_local_matrix`, or
`build_local_candidate`; this keeps Populace as the owner of the build surface
while historical incumbent comparisons remain external migration benchmarks.
The helper `sort_households_by_id` also
codifies the 2024-25 FRS fix: household attributes and weights must be sorted
by the same stable household ID before any positional assignment.

`populace.build.uk_runtime.local_targets` declares the constituency and local-authority
metric surface used by the local build: HMRC employment/self-employment amount
and count rows, ONS age bands, Universal Credit household rows, constituency
UC-by-children rows, and the LA income/tenure/rent rows. It accepts a
PolicyEngine-UK-like simulation object and returns household-indexed metric
tables. It can derive the metric subset from Ledger target profiles while
target values remain explicit build inputs. `local_solver` wraps the Populace
calibrator's log-weight optimizer for stacked and assigned local weights and
records per-area/per-metric diagnostics before the solved weights are exported
with `stacked_weights_to_long` or `assigned_weights_to_long`.

`populace.build.uk_runtime.local_runner` is the Populace-owned candidate build path. It
loads explicit area and target tables, aligns a sorted household frame with
source-year/source-household/clone lineage, optionally computes household
metrics once per UK country by setting the PolicyEngine-UK `region` input, then
solves and writes `local_geography_weights.csv.gz`,
`solve_diagnostics.csv`, `area_support_summary.csv`, and `solve_summary.json`.
It accepts already-pooled or already-cloned household pools, so the compact UK
artifact can remain the fast national default while a separate `local` variant
scales up with pooled FRS years, cloned records, and L0 budget control.

Build the row-wise local-geography H5 from a compact Populace UK H5 with:

```bash
uv run --project packages/populace-build --extra uk python \
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

Like the US plan, UK migration comparisons against earlier production datasets
belong in release/benchmark harnesses outside this package. The build code here
must not import or depend on the incumbent UK data package; `source_manifest.py`
rejects incumbent country data-package references in declarative source specs.

The packaged `uk/source_stages.json` is the Populace-owned raw-input parity
contract for the UK build: FRS base tables, WAS wealth/debt/vehicles with the
cash ISA and stocks-and-shares ISA split, LCFS consumption and bus fare spend,
ETB VAT and public services, DfT bus/rail amount anchors, NHS usage, SPI
high-income income/reliefs, FRS-only pension/savings/reported-benefit fill,
Advani-Summers capital gains, salary sacrifice, SLC student-loan plan
assignment, and row-wise OA/LA/constituency geography. It is a declarative
resource, not a country Python runtime; shared Populace runtimes load and
execute specs.

## US plan status

`populace.build.us_runtime` declares the US build: stage order, donor graph with
citations (`US_DONORS`), the manifest-ready `BuildConfig`, and the packaged
source-stage manifest (`US_SOURCE_MANIFEST`). The stage *implementations* are
injected (`us_plan(implementations)`) and the plan refuses to assemble with any
stage missing — no stubs, no fallbacks. Release-specific benchmark comparison
harnesses live outside this repo.
