# PUF imputation self-prediction (#982): receipts

Issue: PolicyEngine/microcosm#982. Branch `us-982-puf-demographic-rank-predictors`.
Scripts in this folder: `compare_predictors.py` (fit and predict per predictor
variant) and `score_variants.py` (score against SOI). Scored output:
`results/scores_8_trees.json`.

## The defect

The US PUF tax-detail QRF conditioned every income target on the recipient's
own survey value of that item. On the donor side each of those six predictor
columns was an alias of the donor's own imputed output column, so the forest
learned output = input. Measured on the 2026-09-12 genuine-build checkpoints,
PUF-clone wages had rank correlation 1.000 with the same unit's survey wages
and 99.98% landed within 10% of them; no clone exceeded the survey maximum.
Survey topcodes and underreporting therefore carried into the PUF half, which
is where the missing $5M+ tail came from (#958).

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
recipients, weights = PUF-clone half x2. AGI is a proxy: the sum of the 15
imputed income items. Run with 8 trees per variant (memory was constrained);
every variant uses the same configuration. SOI references: Table 1.1 TY2023
counts and AGI; Table 1.4 shares.

### Returns by size of proxy AGI

| Band | Old design | Rank only | Rank + flag (chosen) | Within-group rank + flag | SOI 1.1 |
|---|---|---|---|---|---|
| $1M-1.5M | 834,034 | 374,022 | 397,045 | 329,429 | 368,931 |
| $1.5M-2M | 199,843 | 132,121 | 147,280 | 126,008 | 147,290 |
| $2M-5M | 200,741 | 224,117 | 213,125 | 206,878 | 203,229 |
| $5M-10M | 23,600 | 47,582 | 48,629 | 49,569 | 49,262 |
| $10M+ | 8,186 | 31,404 | 33,920 | 29,041 | 30,382 |

AGI in $10M+: old $185B, chosen $1,037B, SOI $908B. Capital gains share of
$10M+ AGI: old 0.1%, chosen 42.6%, SOI Table 1.4 39.5%.

### Wages: self-prediction and participation

| Measure | Old | Rank only | Rank + flag | Within-group |
|---|---|---|---|---|
| Rank correlation with survey wages | 1.000 | 0.849 | 0.906 | 0.902 |
| Share within 10% of survey value | 99.98% | 23.6% | 24.0% | 2.8% |
| Records above the survey maximum | 0 | 47 | 50 | 31 |
| Survey non-earners given any earnings | 0 | 36,774 units | 7 of 67,908 | 6 of 67,908 |
| Wage-positive share (survey 67.7%) | 67.7% | 80.5% | 67.5% | 67.0% |
| Imputed wage total (survey $11.28T) | $11.28T | $9.76T | $9.68T | $7.93T |

Why the flag: the PUF covers filers, so the survey's large zero-earnings group
ranks level with low but positive PUF incomes; rank alone handed wages to 48%
of survey non-earners. Why pooled rather than within-group ranks: ranking
earners only among earners matched survey earners to a PUF earner distribution
that includes many small separate returns (for example dependents filing their
own), cutting imputed wages to $7.93T; the pooled rank keeps $9.68T with the
same participation fidelity.

## Known limits

- Colorado (#940) is not fixed by this change. The chosen variant has 15
  Colorado records at $1M+ proxy AGI; one survey-weighted record draws $100M+
  and carries 84% of the state's income above $1M, so state top-tail totals are
  noisy until state x AGI-band amount targets (#940) enter calibration.
- The old design's 32-tree run (2026-09-22, same inputs) gave the same
  self-prediction numbers as the 8-tree run above; the chosen variant was not
  rerun at 32 trees.
- The within-10% and wage-total rows are pre-calibration; calibration targets
  wage totals.
