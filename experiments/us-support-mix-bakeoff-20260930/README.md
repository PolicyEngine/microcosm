# Support-mix bake-off receipts (2026-09-30)

Outputs of `tools/bakeoff_us_support_mix.py report` over the 60 arm receipts
(30 arms × national and local products), plus the compile and differential
receipts. [`docs/us-support-mix-bakeoff.md`](../../docs/us-support-mix-bakeoff.md)
gives the design, verdict and caveats.

| File | What |
|---|---|
| `dimensions_by_arm.csv` | One row per arm and product: rows, distinct-household ESS, held-out error by dimension and level, ACS-native error by level |
| `headline.md` | The same, as markdown tables by product |
| `arms.csv` | Per-arm counts, ESS variants, training loss, solve time, peak RSS, report-only SPM rates |
| `holdout_by_dimension_level.csv` | Held-out error by product, arm, dimension group, dimension and level (unsupported targets score as misses) |
| `holdout_by_dimension_level_common_support.csv` | The same over targets every arm supports |
| `holdout_unsupported.csv` | Held-out targets no row in an arm can move |
| `replicate_spread.csv` | SD across three disjoint ACS draws (300k arms) |
| `replicate_mean.csv` | Mean across the same three draws |
| `acs_native_by_level_measure.csv` | ACS-native error by product, arm, level and measure |
| `compile.json` | Registry compile: 32,842 specs, 419 concepts, split roles by level |
| `diffcheck.json` | Production per-target columns vs concept × mask on 4,000 households per source: max scaled difference 0.0 |
| `acs_truth.json` | ACS 2024 1-year summary-file cells and their SHA-256 |
| `partcheck_cps.json`, `partcheck_acs.json` | Stored concept parts against a fresh materialization, under the final concept specs, of 200 sampled households per part: max scaled difference 5.8e-8 |
| `spm_flags.json` | ACS group-quarters share and the SPM zero-adult records the support leaves out (`spm-flags`) |
| `materialize/` | Per-part engine-pass receipts (households, wall time, peak RSS) and the measure list |
| `asec_quality.py`, `asec_quality.json` | CPS year response-quality check (script and its output) |
| `memo_tables.py` | Builds the memo's results tables from these files |

The folder holds no per-arm receipts (10–15 MB each, with every held-out
target's estimate) and no saved weights. Both stayed in the run's work
directory, which no longer exists. The `materialize/` receipts predate the
source-h5, concept-spec and null-fill-summary digests the memo describes; on
2026-09-30 their numerical fields matched the work copies'.
