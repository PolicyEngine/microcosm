# 958: US top-income tail receipts (2026-09-19)

Receipts for PolicyEngine/microcosm#958. Scripts and aggregate results are in
[`958-us-top-income-tail/`](958-us-top-income-tail/). No microdata is committed.
Cells with 1 to 9 unweighted records are suppressed.

## What was measured

| Artifact | Identity |
|---|---|
| Certified dataset | `populace_us_2024`, release `populace-us-2024-spm-20260915`, parent `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z` |
| Stage checkpoints | From-scratch base build of main's chain at `76325eb7`, 2026-09-12: ASEC 2022 to 2024 pooled, processed `puf_2024.h5` donor, raw `puf_2015.csv` for source-year alignment |
| Engine | policyengine-us 2.2.1 |
| Chronicle feed | `consumer_facts_us_c5e5bf8.jsonl`, 39,158 facts, 2026-09-18 |
| Code read | `origin/main` `16c8e78d2` |

"Proxy AGI" is the tax-unit sum of AGI-entering income inputs. It locates the
tail. It is not engine AGI.

## Stage table

`results/stage_proxy_agi_bands.json`, from `checkpoint_proxy_agi_bands.py` and
`pool_proxy_agi_bands.py`.

| Stage | tax units ≥ $5M | $5M–10M units / $B | $10M+ units / $B |
|---|---:|---:|---:|
| Processed 2024 PUF donor | 19,034 | 40.0k / 275 | 25.4k / 797 |
| `001_pre_clone_enrichment` (raw pooled ASEC) | 0 | 0 | 0 |
| `002_clone_feature_extraction` | 0 | 0 | 0 |
| `004_qrf_finalization` | 18 | 6.4k / 42 | 1.7k / 25 |
| `005_capital_gains_tail_transfer` | 9,891 | 18.8k / 128 | 10.8k / 355 |
| `023_final_export` | 9,890 | 18.8k / 128 | 10.8k / 355 |
| Certified file, 2024 base year | 11 | 12.0k / 72 | suppressed |
| SOI TY2023 Table 1.1 | n/a | 49.3k / 336 | 30.4k / 908 |

In the raw pooled ASEC no person has total income above $3.15M. The certified
file has no clone-index-2 rows: its parent commit `cae8640` predates
`puf_capital_gains_tail.py` (first commit `66609b086`, 2026-07-28).

## Target compile

`compile_table_1_1_targets.py` against the feed above with the canonical
district crosswalk, `target_period=2024`, `age_targets=True`.

| | compiled specs | Table 1.1 rows |
|---|---:|---:|
| `origin/main` | 32,866 | 2 |
| PR #960 | 32,882 | 18 |

The 16 new rows are in `results/compiled_table_1_1_specs.json`.

## Prototype validation

**This is not a Microcosm build.** `build_candidate.py` clones high-AGI
single-tax-unit households of the certified file and overwrites their tax
detail from processed-PUF donors, which carry their own weights.
`rake_candidate.py` then tilts household weights onto the compiled targets at
2024 by minimum entropy distance, holds every household with no tax unit at or
above $100k fixed apart from one uniform factor that conserves total mass, and
enforces the solver's 5x weight cap. It stands in for the solve and enforces
none of the other 32,866 targets. `score_candidate.py` scores a 37% to 39.6%
top rate for 2026 with `policyengine_us.Microsimulation`. The harness
reproduces the policyengine 6.0.0 result on the certified file exactly:
+$23.508B and a $905.1B base.

The capital-gains-only rule reproduces main's stage: a weighted q99.5 boundary
of $1,685,506.66 and 15,228 donors.

| Scenario, scored for 2026 | Reform, $B | Ordinary income above the 37% threshold, $B | Returns in bracket |
|---|---:|---:|---:|
| Certified file | +23.5 | 905 | 1.87M |
| Capital-gains-only tail at donor weights, no reweighting | +25.6 | 986 | 1.94M |
| Full-vector tail (proxy AGI ≥ $5M) at donor weights, no reweighting | +33.8 | 1,302 | 1.92M |
| No tail + Table 1.1 size-of-AGI targets (`10m_plus` rows stay at −100%) | +20.8 | 800 | 1.19M |
| Capital-gains-only tail + Table 1.1 size-of-AGI targets | **+18.8** | 725 | 1.20M |
| Full-vector tail + Table 1.1 size-of-AGI targets | +28.4 | 1,091 | 1.20M |
| Full-vector tail + Table 1.1 + Table 1.4 wages and net capital gains by size of AGI | +30.7 | 1,179 | 1.32M |
| Same, tail thinned to 3,170 donors with weights preserved by decile | +30.6 | 1,177 | 1.32M |

External comparisons: SOI TY2023 Table 3.4 has $1,160B taxed at 37% on 1.10M
returns. CRS R49052 reports $41.1B for 2026 (PSL Tax-Calculator) and $29.8B
for FY2026 (Yale TBL). Scaling the SOI base to 2026 by income-tax growth gives
about $1.44T and about $37B. That scaling is an extrapolation.

Findings:

1. **Reweighting the certified file alone cannot work.** With no tail, the
   `10m_plus` rows stay at −100% under the 5x cap
   (`results/reweight_no_tail.json`), the score falls to +$20.8B, and total
   2026 AGI falls from $19.16T to $17.25T because the top classes' AGI has no
   record to land on.
2. **The size-of-AGI targets must not ship with a capital-gains-only tail.**
   They remove the excess $1M–2M returns, which carry ordinary income, and
   the tail replaces that AGI with preferential income. The score falls to
   +$18.8B. The capital-gains-only tail holds $1,528B of AGI above $10M in
   2026 and $99B of 37% base.
3. **A full-vector tail restores most of the base.** At donor weights it
   already sits within 2% to 18% of SOI in the two top classes.
4. **Composition explains part of the remainder.** After matching SOI's AGI
   shape, the candidate is 52.5% net capital gains above $10M against SOI's
   39.5%, and 28.2% against 18.1% from $2M to $5M
   (`results/composition_full_vector_tail_vs_soi_table_1_4.json`). Table 1.4
   wages and net capital gains by size of AGI are already in the feed, their
   measure ids already map to engine variables, and they age on their own CBO
   series (wages 1.056, net capital gains 1.315, AGI 1.087 for 2023 to 2024).
5. **A budgeted stratum is enough.** 3,170 donors reproduce the 19,034-donor
   result to within $0.1B, about 5% of a 57,240 record budget.
6. **About $6B of the gap to the extrapolated $37B is unexplained here.**
   Candidates not tested: interest, dividend and pass-through shares by size
   of AGI, which the feed does not carry, deductions at the top, and the
   extrapolation itself.

Side effect to watch in a real solve: meeting the size classes moves about 7%
more weight onto households below $100k in this prototype, because the
certified file holds about 6M too many tax units above $100k.

## Reproduce

```bash
PY=python  # an environment with policyengine-us 2.2.1
$PY certified_agi.py populace_us_2024.h5 2024 certified_agi_2024.parquet
$PY build_candidate.py --certified populace_us_2024.h5 --donor puf_2024.h5 \
    --certified-agi certified_agi_2024.parquet --variant full_vector \
    --max-donors 3000 --out cand_prerake.h5 --receipt build.json
$PY rake_candidate.py cand_prerake.h5 results/compiled_table_1_1_specs.json \
    cand.h5 reweight.json results/table_1_4_component_targets.json
$PY score_candidate.py cand.h5 2026 score.json
```
