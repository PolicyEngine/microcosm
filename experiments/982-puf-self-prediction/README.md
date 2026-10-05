# PUF imputation self-prediction (#982): receipts

Issue: PolicyEngine/microcosm#982. Fix merged in #1033 (branch
`us-982-puf-demographic-rank-predictors`). Scripts in this folder:
`compare_predictors.py` (fit and predict per predictor variant),
`score_variants.py` (score the PUF-clone half ×2 against SOI) and
`actual_pool_bands.py` (score the real pool, both halves at frame weight).
Scored output: `results/scores_8_trees.json` and
`results/actual_pool_bands.json`; input counts and per-run metadata (trees,
rows, peak memory): `results/inputs_and_runs.json`.

## The defect

The US PUF tax-detail QRF conditioned every income target on the recipient's
own survey value of that item. On the donor side each of those six predictor
columns was an alias of the donor's own imputed output column, so the forest
learned output = input. On the 2026-09-12 genuine-build inputs (the `current`
variant below), PUF-clone wages have rank correlation 1.000 with the same
unit's survey wages; among units with positive survey wages, 99.98% land within
10% of their survey value; and no clone exceeds the survey maximum.
Survey topcodes and underreporting therefore carried into the PUF half, which
is why that half lacked the $5M+ tail (#958). The survey half, which carries
the other half of the weight, has no units there at all; see
[Real pool at frame weight](#real-pool-at-frame-weight).

## The fix

Predictors (`PUF_TAX_DETAIL_DEFAULT_PREDICTORS`):

| Predictor | Survey side | PUF side |
|---|---|---|
| Filing status, unit size | unchanged | unchanged |
| Head age, spouse age (0 if none), head is female, dependent count | person age, sex and tax-unit role | processed PUF person arrays |
| Income rank share | weighted mid-rank of the six-item income total within the PUF-detail recipient rows | the same, within the donor table, on the donor's own outputs |
| Has earnings | nonzero survey wages or self-employment income | nonzero donor wage or self-employment output |

The six income items (wages, self-employment, taxable interest, dividends,
short- and long-term gains) enter only through the bounded rank and the
participation flag, never as levels. The mid-rank puts each unit at the middle
of the population slice its weight stands for; ties share their slice's
middle.

## Evidence

All numbers below are pre-calibration, from one fit on the saved 2026-09-12
donor frame (211,677 PUF tax units) predicted onto the 231,007 saved PUF-clone
recipients, weights = PUF-clone half ×2 ("as if this half were the whole
population"). AGI is a proxy: the sum of the 15 imputed income items. Run with
8 trees per variant (memory was constrained); every variant uses the same
configuration. SOI references: Table 1.1 TY2023 counts and AGI; Table 1.4
shares.

The ×2 weighting compares the variants' distributions with one another and
with SOI's shape. It is not a level comparison with SOI: in the real pool the
survey half carries the other 50% of the weight and has no units at $5M+ proxy
AGI. For levels, see
[Real pool at frame weight](#real-pool-at-frame-weight).

### Returns by size of proxy AGI: PUF-clone half ×2 (distribution check, not a level comparison)

| Band | Old design | Rank only | Rank + flag (chosen) | Within-group rank + flag | SOI 1.1 |
|---|---|---|---|---|---|
| $1M-1.5M | 834,034 | 374,022 | 397,045 | 329,429 | 368,931 |
| $1.5M-2M | 199,843 | 132,121 | 147,280 | 126,008 | 147,290 |
| $2M-5M | 200,741 | 224,117 | 213,125 | 206,878 | 203,229 |
| $5M-10M | 23,600 | 47,582 | 48,629 | 49,569 | 49,262 |
| $10M+ | 8,186 | 31,404 | 33,920 | 29,041 | 30,382 |

AGI in $10M+ (×2): old $185B, chosen $1,037B, SOI $908B. Capital gains share
of $10M+ AGI: old 0.1%, chosen 42.6%, SOI Table 1.4 39.5%.

### Wage self-prediction and participation: PUF-clone half ×2 (distribution check, not a level comparison)

| Measure | Old | Rank only | Rank + flag | Within-group |
|---|---|---|---|---|
| Rank correlation with survey wages | 1.000 | 0.849 | 0.906 | 0.902 |
| Within 10% of survey value (units with positive survey wages) | 99.98% | 23.6% | 24.0% | 2.8% |
| Records above the survey maximum | 0 | 47 | 50 | 31 |
| Survey non-earner units given any earnings (of 67,908) | 127 | 36,774 | 7 | 6 |
| Same, weighted share | 0.17% | 55.1% | 0.009% | 0.010% |
| Wage-positive share (survey 67.7%) | 67.7% | 80.5% | 67.5% | 67.0% |
| Imputed wage total (survey $11.28T) | $11.28T | $9.76T | $9.68T | $7.93T |

Why the flag: the PUF covers filers, so the survey's large zero-earnings group
ranks level with low but positive PUF incomes; rank alone handed positive wages
to 47.9% (weighted) of survey units with no wages or self-employment income. Why pooled rather than within-group ranks: ranking
earners only among earners matched survey earners to a PUF earner distribution
that includes many small separate returns (for example dependents filing their
own), cutting imputed wages to $7.93T (×2); the pooled rank keeps $9.68T (×2)
with the same participation fidelity.

## Real pool at frame weight

The PUF-detail expansion (`microcosm.build.us_runtime.puf_support`) splits
each unit's weight evenly between two clone channels. Clone index 0, the
native survey channel, keeps the survey values; clone index 1
(`PUF_TAX_DETAIL_CLONE_INDEX`) receives the imputed ones.
`actual_pool_bands.py` combines both halves at frame weight and writes
`results/actual_pool_bands.json`. The figures are from the same fit as above
(8 trees, proxy AGI), before calibration and before main's capital-gains tail
stage (#567). "Old design" is the `current` variant; "#1033" is
`demographic_midrank6_earn`, the rank + flag design that #1033 merged.

Clone 0 carries 50.0% of the tax-unit weight (`clone0_weight_share`) and has
no units at $5M+ proxy AGI in either variant: in both $5M+ bands,
`clone1_returns` equals `returns`. Its proxy AGI sums the 11 of the 15 items
that its person table carries; estate, farm, partnership and S-corporation
income are absent (`clone0_items_used`).

| Band | Old design | #1033 | SOI Table 1.1 |
|---|---|---|---|
| $1M–1.5M returns | 701k (1.90×) | 483k (1.31×) | 369k |
| $5M–10M returns | 11.8k (0.24×) | 24.3k (0.49×) | 49.3k |
| $10M+ returns | 4.1k (0.13×) | 17.0k (0.56×) | 30.4k |
| $10M+ AGI | $92B (0.10×) | $519B (0.57×) | $908B |
| Colorado income above $1M | $20.2B (0.64×; largest record 28%) | $66.3B (2.1×; largest record 81%) | $31.7B (Historic Table 2, via #940) |

"Largest record" is the share of the state's income above $1M carried by its
single largest contributor.

#1033 nearly quadruples (3.9×) the pool's proxy AGI at $5M+ ($174B to $675B)
and lifts its $5M+ returns from 15.9k to 41.3k. On its own, though, it
reaches about half of SOI there (52% of the 79.6k returns and 54% of the $1,244B), not
SOI. At $5M+ the real-pool figures are exactly half of the ×2 ones (for
example 48,629 to 24,315 returns at $5M–10M), because only the PUF-clone half
has units there. The bands in between ($1.5M–5M) sit at 0.59–0.84× SOI in
returns and 0.52–0.83× in AGI, in both designs. At $1M–1.5M the survey half alone holds 284k returns, 77% of
SOI's count, in both designs. All five bands, with AGI and record counts, are
in `results/actual_pool_bands.json`.

These figures correct the ×2 level comparisons first posted on #982, #940 and
#1033; a correction is posted on each.

## Known limits

- Colorado (#940) is not fixed by this change. In the PUF-clone half the
  chosen variant has 15 Colorado records at $1M+ proxy AGI. The largest
  (survey six-item total $1.1M) draws $125.4M of proxy AGI at frame weight
  434, shown as 867 in the ×2 scores. In the real pool at frame weight,
  Colorado's income above $1M is $66.3B, 2.1× SOI's $31.7B, and that one
  record carries 81% of it. The ×2 scores put the total at $128.5B, about 4×
  SOI, with that record at 84% (`colorado_1m_plus`); those are not levels.
  State top-tail totals therefore stay noisy until state × AGI-band amount
  targets (#940) enter calibration.
- All variants ran at 8 trees because memory was constrained; production fits
  more trees. The comparison is like for like across variants.
- The within-10% and wage-total rows are pre-calibration; calibration targets
  wage totals.
