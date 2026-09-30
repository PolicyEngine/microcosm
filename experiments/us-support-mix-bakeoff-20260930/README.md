# Support-mix bake-off receipts (2026-09-30)

Outputs of `tools/bakeoff_us_support_mix.py report` over the 60 arm receipts
(30 arms × national and local products), plus the compile and differential
receipts. The design, verdict and caveats are in
[`docs/us-support-mix-bakeoff.md`](../../docs/us-support-mix-bakeoff.md).

| File | What |
|---|---|
| `dimensions_by_arm.csv` | One row per arm and product: rows, distinct-household ESS, held-out error by dimension and level, ACS-native error by level |
| `headline.md` | The same, as markdown tables by product |
| `arms.csv` | Per-arm counts, ESS variants, training loss, solve time, peak RSS, report-only SPM rates |
| `holdout_by_dimension_level.csv` | Held-out error by product, arm, dimension group, dimension and level (unsupported targets score as misses) |
| `holdout_by_dimension_level_common_support.csv` | The same over targets every arm supports |
| `holdout_unsupported.csv` | Held-out targets no row in an arm can move |
| `replicate_spread.csv` | SD across three disjoint ACS draws (300k arms) |
| `acs_native_by_level_measure.csv` | ACS-native error by product, arm, level and measure |
| `compile.json` | Registry compile: 32,842 specs, 419 concepts, split roles by level |
| `diffcheck.json` | Production per-target columns vs concept × mask on 4,000 households per source: max scaled difference 0.0 |
| `acs_truth.json` | ACS 2024 1-year summary-file cells and their SHA-256 |

Per-arm receipts (with every held-out target's estimate) and saved weights
stay in the run's work directory; they are 10–15 MB each.
