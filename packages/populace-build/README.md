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
  structural extras or reviewed exclusions (for UK, this is the eFRS
  compatibility check);
- **target surface** — the calibration target set covers the reference
  target surface and may only be wider, not narrower (for UK, Populace must
  calibrate to at least the eFRS target surface);
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

## UK local-geography path

`populace.build.uk.local_geography` holds the Populace-owned replacement shape
for UK constituency and local-authority geography. It uses the same stacked
local-area layout as the US local ECPS flow:

```text
column = area_index * n_households + household_index
```

The solved weights export to a long sidecar with `(area_type, area_code,
household_id, weight)` rows plus source-year/source-household lineage. This is
the format PolicyEngine can group by directly for constituency and local
authority outputs, and it avoids preserving the legacy dense
`areas x households` matrix artifact.

The module does not import the incumbent UK data package. Engine runners and
target providers pass household metric tables and aligned target tables into
`build_stacked_local_matrix`; this keeps Populace clean while the target source
files move over. The helper `sort_households_by_id` also codifies the 2024-25
FRS fix: household attributes and weights must be sorted by the same stable
household ID before any positional assignment.

`populace.build.uk.local_targets` declares the constituency and local-authority
metric surface used by the local build: HMRC employment/self-employment amount
and count rows, ONS age bands, Universal Credit household rows, constituency
UC-by-children rows, and the LA income/tenure/rent rows. It accepts a
PolicyEngine-UK-like simulation object and returns household-indexed metric
tables; it still takes target values as explicit input tables. `local_solver`
wraps the Populace calibrator's log-weight optimizer for stacked local weights
and records per-area/per-metric diagnostics before the solved weights are
exported with `stacked_weights_to_long`.

## US plan status

`populace.build.us` declares the US build: stage order, donor graph with
citations (`US_DONORS`), the manifest-ready `BuildConfig`, and the packaged
source-stage manifest (`US_SOURCE_MANIFEST`). The stage *implementations* are
injected (`us_plan(implementations)`) and the plan refuses to assemble with any
stage missing — no stubs, no fallbacks. Release-specific benchmark comparison
harnesses live outside this repo.
