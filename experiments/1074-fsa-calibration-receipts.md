# microcosm#1074 receipts: Federal Student Aid portfolio targets

Generated 2026-11-06 by `experiments/fit_1074_fsa_calibration.py`, commit `8a9d335a7f767e8406325ec7611338d6a5603610`, Modal run `20261106T142002Z-8a9d335a7` (`experiments/modal_1074.py`). Machine-readable copy: `experiments/receipts/1074-fsa-calibration-fit.json`.

Host: the published default `populace-us-2024-spm-20260915` (sha256 `6496cc4393d4…`, 57,240 households), with the SCF 2025 education-debt stage run at seed 0. Feed: the US feed the branch pins, rebuilt at Chronicle `41b2fb2b` (`1d45a89c2817…`, 40,642 rows, as pinned), which carries the FSA rows. Arms: **release** = the published weights; **A** = a dense refit to the `national_state` surface without the FSA family; **B** = the same refit with it; **+ record L2** and **+ chi-square L2** = A and B with a concentration penalty at the stated `l2_lambda`, on `calibrate`'s default `record` basis (the mean squared weight ratio, which spreads weight without anchoring to the design; the only basis the release's `--l2-lambda` passes) or its `chi_square` basis (the anchor-weighted distance from the published weights); **counts x0** = B with the balance targets at full loss weight and the borrower counts at none. Same epochs (1500), learning rate, 5x weight cap and conserved mass as a release, on the published support rather than the whole base a Route A build solves. Not a release: no gate ran and nothing was published.

## FSA targets: the reconciled levels

| | Balance | Borrowers |
|---|---:|---:|
| 50 states and DC, as published | $1,627.4B | 42.38M |
| Puerto Rico and Other (leave) | $16.7B | 433,400 |
| Not Reported (allocated) | $52.2B | 1.95M |
| Location table total | $1,696.3B | 44.76M |
| Allocation factor | 1.0317 | 1.0456 |
| In-scope control (every family sums to it) | $1,679.1B | 44.31M |

Reference date: Data as of Mar. 31, 2026 (`2026-03`), the latest quarter all four tables publish. The targets bind that stock unaged (`period_alignment = stock_at_source_reference_date`), and the build writes the calibrated balances at its period, 2024, since the H5 records one period for every column. policyengine-us#10415 (merged 2026-10-30) holds the balances at their stored value in every year, so a 2024 or a 2026 read returns each person's March 2026 stock. policyengine-us also raises household weights by population growth, x1.0165 from 2024 to 2026 at the student debt pilot's pin, so a 2026 read of these weights shows every level here 1.65% higher. The age bands group the stored, unaged `age`.

FSA's Parent PLUS row ($117.9B, 6.8% of the loan-type table's dollar rows) is diagnostic: 34 CFR 685.220(f) has the Secretary pay off each loan a Direct Consolidation Loan consolidates, so FSA likely files a consolidated Parent PLUS loan under Consolidation and the share is likely a lower bound on the stage's column. FSA's notes don't say.


## Every arm at a glance

The fit, the levels and the weight concentration the release's L2 setting depends on. Reference: FSA, except the household share, which is the SCF's; FSA's Parent PLUS share is diagnostic (no target binds it). The x0 arms keep the 65 borrower-count targets at none of their loss weight, so those counts only validate.

| | Reference | Release | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All FSA targets within 5% (130) |  | 0% | 1% | 95% | 0% | 76% | 0% | 59% | 57% | 45% |
| FSA balance targets within 5% (65) |  | 0% | 2% | 98% | 0% | 92% | 0% | 80% | 95% | 80% |
| FSA borrower-count targets within 5% (65) |  | 0% | 0% | 91% | 0% | 60% | 0% | 38% | 18% | 9% |
| FSA balance targets: mean error |  | 47.6% | 47.1% | 0.3% | 47.6% | 1.5% | 48.2% | 4.2% | 1.4% | 4.8% |
| FSA borrower-count targets: mean error |  | 49.8% | 50.2% | 1.3% | 49.7% | 6.6% | 50.2% | 13.2% | 17.8% | 25.2% |
| Non-FSA capped error |  | 5.30% | 3.30% | 3.66% | 3.85% | 3.86% | 3.93% | 4.18% | 3.42% | 3.84% |
| Non-FSA targets within 5% (6,102) |  | 85.2% | 92.0% | 90.5% | 90.4% | 90.1% | 89.8% | 87.7% | 91.5% | 90.4% |
| Households with federal education debt | 18.2% | 16.4% | 16.6% | 33.9% | 16.6% | 32.4% | 16.3% | 28.0% | 30.4% | 25.7% |
| Borrowers | 44.31M | 21.10M | 21.54M | 44.27M | 21.34M | 42.42M | 21.00M | 36.61M | 39.68M | 33.72M |
| Mean balance per borrower | $37,894 | $40,688 | $41,125 | $37,886 | $40,101 | $39,359 | $40,319 | $43,996 | $42,244 | $49,164 |
| Parent PLUS share of the balance (FSA: diagnostic) | 6.8% | 10.7% | 10.8% | 10.2% | 10.8% | 10.7% | 10.5% | 11.3% | 10.3% | 10.8% |
| Effective sample size |  | 9,174 | 4,149 | 2,932 | 6,435 | 4,484 | 8,487 | 6,222 | 3,346 | 5,302 |
| Median household's share of its published weight |  | 100.00% | 2.14% | 0.05% | 13.32% | 1.23% | 81.16% | 0.00% | 0.23% | 5.90% |
| Borrowers age 24 and Younger | 13.8% | 12.8% | 12.8% | 13.8% | 12.9% | 12.8% | 12.9% | 11.9% | 12.0% | 11.0% |
| Borrowers age 25 to 34 | 31.2% | 31.1% | 31.1% | 31.2% | 31.1% | 31.8% | 30.8% | 30.4% | 31.5% | 31.0% |
| Borrowers age 35 to 49 | 33.5% | 37.0% | 37.5% | 33.5% | 37.3% | 35.0% | 37.3% | 36.6% | 36.4% | 36.3% |
| Borrowers age 50 to 61 | 14.3% | 13.6% | 13.0% | 14.3% | 13.1% | 13.5% | 13.4% | 14.3% | 13.2% | 14.2% |
| Borrowers age 62 and Older | 7.1% | 5.6% | 5.5% | 7.1% | 5.6% | 6.9% | 5.7% | 6.9% | 6.9% | 7.5% |
| Borrower age profile: distance from FSA (TVD) | | 0.035 | 0.040 | 0.001 | 0.038 | 0.020 | 0.038 | 0.031 | 0.032 | 0.032 |

## Fit by FSA family

Mean absolute relative error (share of targets within 5%).

| FSA family | Targets | Release | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fsa_age_band_balance | 5 | 46.4% (0%) | 44.5% (0%) | 0.1% (100%) | 46.8% (0%) | 0.3% (100%) | 47.4% (0%) | 3.4% (60%) | 0.1% (100%) | 0.6% (100%) |
| fsa_age_band_borrowers | 5 | 54.8% (0%) | 54.0% (0%) | 0.1% (100%) | 54.4% (0%) | 6.1% (40%) | 54.9% (0%) | 19.3% (0%) | 13.1% (20%) | 25.2% (0%) |
| fsa_debt_size_band_balance | 9 | 51.1% (0%) | 50.6% (0%) | 0.6% (100%) | 51.2% (0%) | 4.5% (78%) | 51.6% (0%) | 12.8% (67%) | 8.7% (78%) | 18.8% (56%) |
| fsa_debt_size_band_borrowers | 9 | 49.4% (0%) | 49.7% (0%) | 2.4% (89%) | 49.9% (0%) | 5.5% (67%) | 50.1% (0%) | 13.7% (44%) | 11.5% (44%) | 20.6% (44%) |
| fsa_state_balance | 51 | 47.0% (0%) | 46.7% (2%) | 0.2% (98%) | 47.0% (0%) | 1.1% (94%) | 47.6% (0%) | 2.8% (84%) | 0.2% (98%) | 2.7% (82%) |
| fsa_state_borrowers | 51 | 49.4% (0%) | 49.9% (0%) | 1.2% (90%) | 49.3% (0%) | 6.8% (61%) | 49.7% (0%) | 12.6% (41%) | 19.4% (14%) | 26.1% (4%) |

## Headline levels

Reference: FSA's in-scope control; the Parent PLUS rows are FSA's diagnostic loan-type row, and the household share has no FSA figure.

| | FSA | Release | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Federal balance, own plus Parent PLUS | $1,679.1B | $858.4B | $886.0B | $1,677.3B | $855.8B | $1,669.7B | $846.7B | $1,610.5B | $1,676.2B | $1,657.9B |
| Borrowers (people with a balance) | 44.31M | 21.10M | 21.54M | 44.27M | 21.34M | 42.42M | 21.00M | 36.61M | 39.68M | 33.72M |
| Parent PLUS balance (FSA: diagnostic) | $114.8B | $92.1B | $95.8B | $171.4B | $92.2B | $178.0B | $88.6B | $182.3B | $173.1B | $178.6B |
| Parent PLUS share of the balance | 6.8% | 10.7% | 10.8% | 10.2% | 10.8% | 10.7% | 10.5% | 11.3% | 10.3% | 10.8% |
| Mean balance per borrower | $37,894 | $40,688 | $41,125 | $37,886 | $40,101 | $39,359 | $40,319 | $43,996 | $42,244 | $49,164 |
| Households with federal education debt |  | 16.4% | 16.6% | 33.9% | 16.6% | 32.4% | 16.3% | 28.0% | 30.4% | 25.7% |

## Households with federal education debt by household income quintile

The income is the household income the imputation conditions on, over the SCF primary economic unit. The student debt pilot counts borrowers as people and cuts tax-unit AGI over all tax units, so these rows are not its income tables. Where Q4 tops a row, Q5 - Q1 is an endpoint contrast rather than a slope.

Each arm's own weighted quintiles (the SCF's are its own families'):

| | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 - Q1, points | Bottom two quintiles' share of borrower households |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCF 2025 | 11.4% | 17.1% | 17.8% | 26.6% | 17.9% | +6.5 | 31.4% |
| Release | 14.3% | 15.8% | 14.7% | 23.1% | 14.2% | +0.0 | 36.6% |
| A | 15.7% | 15.7% | 15.8% | 22.0% | 13.9% | -1.8 | 37.8% |
| B | 37.1% | 36.2% | 27.7% | 41.2% | 27.4% | -9.6 | 43.2% |
| A + record L2 | 14.9% | 15.8% | 15.1% | 22.4% | 14.7% | -0.2 | 37.1% |
| B + record L2 | 32.4% | 33.6% | 27.0% | 41.6% | 27.3% | -5.1 | 40.7% |
| A + chi-square L2 | 13.9% | 16.2% | 14.5% | 22.6% | 14.5% | +0.6 | 36.9% |
| B + chi-square L2 | 25.0% | 27.2% | 23.2% | 38.7% | 26.0% | +0.9 | 37.3% |
| B, counts x0 | 30.9% | 32.3% | 26.1% | 37.9% | 24.6% | -6.2 | 41.6% |
| B + record L2, counts x0 | 24.1% | 26.0% | 21.5% | 35.0% | 22.1% | -2.0 | 38.9% |

At the release arm's cut points: every household keeps the quintile it has at the published weights, so each column holds the same households in every arm. The release row reproduces its own-cut row. Income range of each release quintile: Q1 -$945,807 to $34,347; Q2 $34,347 to $62,074; Q3 $62,074 to $100,000; Q4 $100,000 to $170,995; Q5 $170,995 to $13,904,328.

| | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 - Q1, points | Bottom two quintiles' share of borrower households |
|---|---:|---:|---:|---:|---:|---:|---:|
| Release | 14.3% | 15.8% | 14.7% | 23.1% | 14.2% | +0.0 | 36.6% |
| A | 14.9% | 16.8% | 15.6% | 22.2% | 13.8% | -1.1 | 35.4% |
| B | 36.6% | 37.8% | 28.2% | 41.4% | 27.1% | -9.4 | 40.5% |
| A + record L2 | 14.5% | 16.1% | 15.4% | 22.7% | 14.4% | -0.1 | 35.2% |
| B + record L2 | 32.1% | 34.1% | 27.7% | 41.8% | 27.0% | -5.1 | 38.8% |
| A + chi-square L2 | 14.0% | 16.0% | 14.9% | 22.8% | 14.2% | +0.3 | 36.2% |
| B + chi-square L2 | 24.7% | 27.2% | 24.0% | 38.7% | 25.5% | +0.8 | 36.2% |
| B, counts x0 | 30.3% | 33.4% | 26.6% | 37.9% | 24.3% | -6.0 | 38.5% |
| B + record L2, counts x0 | 23.7% | 26.2% | 22.5% | 35.0% | 21.7% | -2.0 | 36.8% |

Each fixed quintile's share of household weight:

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| Release | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| A | 18.1% | 18.9% | 24.6% | 19.0% | 19.3% |
| B | 18.8% | 18.1% | 24.5% | 19.7% | 18.9% |
| A + record L2 | 18.7% | 19.4% | 23.5% | 19.0% | 19.4% |
| B + record L2 | 18.7% | 19.3% | 23.4% | 19.5% | 19.2% |
| A + chi-square L2 | 19.2% | 20.2% | 21.8% | 19.4% | 19.4% |
| B + chi-square L2 | 19.0% | 20.0% | 22.0% | 20.1% | 19.0% |
| B, counts x0 | 18.1% | 18.6% | 24.5% | 19.9% | 18.9% |
| B + record L2, counts x0 | 18.5% | 19.4% | 23.4% | 19.5% | 19.2% |

## Every other family: does adding FSA cost the rest of the surface?

Mean absolute relative error over the `national_state` surface's non-FSA targets, by provider/category (the dashboard's family), each miss capped at 100% of its target as the loss caps it, sorted by how much adding the FSA family moves it:

| Family | Targets | Release | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 | B - A (pp) | B + record L2 - A + record L2 (pp) | B + chi-square L2 - A + chi-square L2 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hhs_acf_tanf/hhs_acf_tanf.hhs_acf_tanf | 30 | 29.8% | 18.3% | 20.6% | 22.5% | 22.4% | 23.9% | 25.1% | 20.1% | 24.0% | +2.30 | -0.04 | +1.21 |
| ssa/ssa.ssi_federal_payment_recipients | 3 | 28.5% | 15.6% | 17.2% | 24.3% | 23.5% | 29.4% | 29.6% | 15.2% | 24.0% | +1.61 | -0.80 | +0.13 |
| jct/jct.individual_tax_expenditure_revenue_loss | 11 | 42.3% | 35.3% | 36.7% | 39.9% | 39.7% | 41.4% | 41.2% | 36.8% | 39.7% | +1.37 | -0.23 | -0.18 |
| bea/bea.nipa_proprietors_income | 1 | 11.6% | 5.9% | 7.2% | 11.2% | 10.0% | 13.5% | 12.6% | 7.6% | 11.0% | +1.33 | -1.22 | -0.88 |
| irs_soi/irs_soi.capital_gains_gross | 104 | 30.1% | 17.0% | 18.0% | 26.0% | 24.9% | 26.2% | 24.0% | 17.3% | 25.3% | +1.04 | -1.06 | -2.21 |
| irs_soi/irs_soi.assigned_aca_ptc | 104 | 0.9% | 0.6% | 1.6% | 0.7% | 1.1% | 0.7% | 1.8% | 0.9% | 0.8% | +0.99 | +0.39 | +1.10 |
| irs_soi/irs_soi.refundable_ctc | 104 | 0.8% | 0.3% | 1.3% | 0.7% | 1.2% | 0.7% | 1.6% | 0.5% | 0.9% | +0.92 | +0.44 | +0.93 |
| irs_soi/irs_soi.real_estate_taxes | 106 | 6.5% | 5.3% | 6.2% | 6.3% | 6.6% | 6.3% | 7.4% | 5.8% | 6.4% | +0.91 | +0.30 | +1.04 |
| irs_soi/irs_soi.eitc | 781 | 3.1% | 2.5% | 3.3% | 2.7% | 3.1% | 2.7% | 3.4% | 2.9% | 2.9% | +0.82 | +0.39 | +0.77 |
| irs_soi/irs_soi.salt_deduction | 106 | 1.9% | 1.8% | 2.5% | 2.1% | 2.2% | 2.1% | 2.5% | 2.1% | 2.1% | +0.71 | +0.09 | +0.40 |
| irs_soi/irs_soi.ctc | 104 | 1.5% | 0.6% | 1.2% | 1.2% | 1.4% | 1.2% | 1.6% | 1.0% | 1.3% | +0.65 | +0.24 | +0.45 |
| irs_soi/irs_soi.taxable_interest_income | 1038 | 6.0% | 5.7% | 6.3% | 6.0% | 6.0% | 6.0% | 6.9% | 5.8% | 5.9% | +0.59 | -0.03 | +0.89 |
| irs_soi/irs_soi.charitable_deduction | 2 | 7.5% | 6.8% | 7.4% | 6.2% | 6.6% | 6.6% | 6.8% | 7.3% | 6.5% | +0.52 | +0.46 | +0.22 |
| irs_soi/irs_soi.interest_deduction | 2 | 2.0% | 1.7% | 1.3% | 2.4% | 2.1% | 2.5% | 1.2% | 1.3% | 2.1% | -0.44 | -0.31 | -1.22 |
| irs_soi/irs_soi.unemployment_compensation | 106 | 0.1% | 0.3% | 0.6% | 0.2% | 0.6% | 0.2% | 2.4% | 0.3% | 0.4% | +0.37 | +0.41 | +2.16 |
| irs_soi/irs_soi.count | 260 | 31.7% | 20.5% | 20.9% | 22.9% | 22.3% | 23.5% | 21.7% | 20.7% | 22.5% | +0.36 | -0.50 | -1.75 |
| irs_soi/irs_soi.qualified_dividends | 104 | 0.3% | 0.5% | 0.8% | 0.5% | 0.5% | 0.4% | 1.1% | 0.6% | 0.5% | +0.27 | +0.05 | +0.70 |
| irs_soi/irs_soi.tax_exempt_interest_income | 104 | 0.1% | 0.7% | 0.9% | 1.0% | 1.0% | 0.9% | 1.2% | 0.8% | 0.9% | +0.23 | -0.03 | +0.28 |
| census_stc/census_stc.individual_income_tax_collections | 44 | 13.9% | 13.7% | 13.9% | 13.9% | 14.0% | 14.1% | 14.1% | 13.8% | 13.9% | +0.19 | +0.05 | -0.04 |
| irs_soi/irs_soi.estate_losses | 2 | 50.0% | 50.2% | 50.0% | 50.2% | 50.0% | 50.0% | 50.0% | 50.1% | 50.1% | -0.18 | -0.18 | -0.00 |
| irs_soi/irs_soi.income_tax | 104 | 0.6% | 0.7% | 0.8% | 0.8% | 0.9% | 0.8% | 0.8% | 0.7% | 0.8% | +0.17 | +0.09 | +0.03 |
| irs_soi/irs_soi.ordinary_dividend_income | 108 | 0.8% | 0.8% | 0.9% | 0.8% | 0.7% | 0.9% | 1.1% | 0.8% | 0.8% | +0.13 | -0.10 | +0.13 |
| usda_snap/usda_snap.usda_snap | 104 | 7.9% | 0.1% | 0.2% | 0.6% | 0.7% | 0.9% | 1.2% | 0.1% | 0.7% | +0.13 | +0.01 | +0.36 |
| bea/bea.nipa_wages_and_salaries | 1 | 13.3% | 13.6% | 13.5% | 13.5% | 13.5% | 13.6% | 13.6% | 13.5% | 13.4% | -0.12 | +0.04 | +0.01 |
| irs_soi/irs_soi.tip_income | 1 | 21.3% | 0.1% | 0.0% | 0.1% | 0.1% | 0.0% | 0.1% | 0.1% | 0.1% | -0.11 | -0.01 | +0.04 |
| irs_soi/irs_soi.medical_expense_deduction | 104 | 6.0% | 6.3% | 6.4% | 6.2% | 6.4% | 6.2% | 7.0% | 6.2% | 6.4% | +0.10 | +0.22 | +0.82 |
| irs_soi/irs_soi.income_tax_before_credits | 104 | 1.6% | 1.6% | 1.7% | 1.9% | 2.0% | 1.9% | 1.9% | 1.7% | 1.9% | +0.10 | +0.08 | +0.01 |
| cms_medicare/cms_medicare.cms_medicare | 1 | 13.8% | 13.5% | 13.6% | 13.5% | 13.4% | 13.7% | 13.7% | 13.2% | 13.6% | +0.10 | -0.08 | +0.06 |
| irs_soi/irs_soi.taxable_income | 107 | 0.3% | 0.4% | 0.5% | 0.6% | 0.6% | 0.7% | 0.8% | 0.4% | 0.6% | +0.09 | +0.02 | +0.08 |
| irs_soi/irs_soi.taxable_social_security | 108 | 0.3% | 0.2% | 0.3% | 0.3% | 0.3% | 0.3% | 0.3% | 0.3% | 0.3% | +0.08 | +0.03 | +0.04 |
| irs_soi/irs_soi.employment_income | 106 | 0.1% | 0.1% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.1% | 0.2% | +0.07 | +0.03 | +0.05 |
| irs_soi/irs_soi.adjusted_gross_income | 261 | 19.7% | 0.6% | 0.6% | 2.8% | 2.2% | 3.4% | 1.7% | 0.6% | 2.3% | +0.07 | -0.60 | -1.67 |
| irs_soi/irs_soi.partnership_and_s_corp_income | 104 | 0.6% | 0.4% | 0.4% | 0.6% | 0.6% | 0.6% | 0.7% | 0.3% | 0.6% | +0.06 | -0.06 | +0.08 |
| irs_soi/irs_soi.estate_income | 2 | 0.0% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.2% | 0.1% | -0.05 | -0.03 | -0.02 |
| cbo/cbo.cbo | 5 | 4.5% | 4.6% | 4.6% | 4.6% | 4.6% | 4.6% | 4.7% | 4.6% | 4.5% | -0.05 | +0.06 | +0.06 |
| irs_soi/irs_soi.ira_distributions | 108 | 0.1% | 0.1% | 0.2% | 0.1% | 0.2% | 0.1% | 0.2% | 0.1% | 0.2% | +0.04 | +0.03 | +0.09 |
| irs_soi/irs_soi.rent_and_royalty_net_income | 104 | 4.0% | 3.1% | 3.1% | 3.1% | 3.1% | 3.1% | 3.4% | 3.1% | 3.1% | +0.04 | -0.01 | +0.27 |
| irs_soi/irs_soi.non_sch_d_capital_gains | 2 | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% | 0.0% | 0.1% | 0.1% | 0.1% | -0.03 | -0.02 | +0.09 |
| ssa/ssa.ssi_payments | 51 | 3.7% | 0.4% | 0.4% | 0.5% | 0.9% | 1.1% | 1.5% | 0.6% | 0.5% | -0.03 | +0.34 | +0.42 |
| hhs_acf_liheap/hhs_acf_liheap.hhs_acf_liheap | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% | 0.1% | -0.01 | -0.00 | +0.09 |
| ssa/ssa.ssa | 6 | 3.7% | 0.1% | 0.1% | 1.4% | 1.3% | 2.2% | 2.0% | 0.1% | 1.4% | -0.01 | -0.14 | -0.18 |
| federal_reserve/federal_reserve.federal_reserve | 1 | 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | +0.01 | -0.03 | +0.01 |
| census_pep/census_pep.resident_population | 936 | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.2% | 0.1% | 0.1% | +0.01 | +0.01 | +0.08 |
| cms_medicaid/cms_medicaid.cms_medicaid | 135 | 3.1% | 2.3% | 2.3% | 2.3% | 2.3% | 2.3% | 2.3% | 2.3% | 2.3% | +0.01 | +0.01 | -0.01 |
| ssa/ssa.ssi_recipients | 51 | 4.8% | 0.7% | 0.7% | 2.0% | 1.5% | 2.1% | 2.4% | 0.7% | 1.6% | +0.00 | -0.49 | +0.32 |
| irs_soi/irs_soi.itemized_taxable_income_deductions | 55 | 2.0% | 2.3% | 2.3% | 2.5% | 2.4% | 2.3% | 2.8% | 2.3% | 2.5% | -0.00 | -0.14 | +0.47 |
| cms_aca/cms_aca.cms_aca | 102 | 1.6% | 0.2% | 0.2% | 0.7% | 0.6% | 1.1% | 0.9% | 0.2% | 0.6% | +0.00 | -0.12 | -0.14 |
| irs_soi/irs_soi.business_net_profits | 104 | 1.3% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% | -0.00 | +0.02 | +0.01 |
| irs_soi/irs_soi.deductible_mortgage_interest | 2 | 0.4% | 0.0% | 0.0% | 0.1% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | +0.00 | -0.08 | -0.06 |
| irs_soi/irs_soi.taxable_pension_income | 108 | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.2% | 0.1% | 0.1% | -0.00 | +0.01 | +0.10 |
| **all non-FSA targets** | 6102 | 5.3% | 3.3% | 3.7% | 3.8% | 3.9% | 3.9% | 4.2% | 3.4% | 3.8% | +0.36 | +0.01 | +0.25 |

## Racial composition of borrowers, before and after calibration

Borrower households by the reference person's race (the SCF's `racecl4`; the SCF column pools the five implicates):

| | SCF 2025 | Release | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 | B - release (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| white_non_hispanic | 57.6% | 60.8% | 60.0% | 57.8% | 62.1% | 60.2% | 60.6% | 59.9% | 56.9% | 60.7% | -2.9 |
| black_non_hispanic | 18.6% | 17.4% | 16.4% | 18.6% | 16.8% | 18.9% | 17.3% | 18.0% | 17.7% | 18.4% | +1.3 |
| hispanic | 9.2% | 12.9% | 13.8% | 14.5% | 12.4% | 12.4% | 13.1% | 13.2% | 15.8% | 12.3% | +1.6 |
| other_or_multiple | 14.7% | 9.0% | 9.8% | 9.1% | 8.7% | 8.5% | 9.0% | 8.9% | 9.6% | 8.6% | +0.1 |

Borrowers (people) by their own race:

| | Release | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| white_non_hispanic | 60.4% | 59.3% | 57.4% | 61.5% | 59.5% | 60.2% | 59.6% | 56.1% | 59.9% |
| black_non_hispanic | 16.7% | 16.2% | 18.2% | 16.5% | 18.3% | 16.8% | 17.2% | 17.3% | 17.7% |
| hispanic | 13.3% | 14.1% | 14.5% | 12.6% | 12.8% | 13.4% | 13.5% | 16.2% | 12.6% |
| other_or_multiple | 9.5% | 10.4% | 9.8% | 9.5% | 9.4% | 9.6% | 9.7% | 10.4% | 9.8% |

Adults with a balance, by race:

| | Release | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| white_non_hispanic | 7.7% | 7.9% | 15.8% | 7.7% | 14.8% | 7.6% | 13.1% | 13.7% | 11.8% |
| black_non_hispanic | 12.9% | 12.4% | 27.1% | 12.8% | 26.3% | 12.8% | 22.1% | 23.3% | 20.9% |
| hispanic | 6.2% | 6.6% | 14.5% | 6.7% | 14.4% | 6.3% | 11.6% | 14.3% | 11.1% |
| other_or_multiple | 7.9% | 8.3% | 15.3% | 8.0% | 15.4% | 7.9% | 13.6% | 15.3% | 13.0% |

## Weight movement against the published weights

| | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Borrower households' weight ratio | 1.01 | 2.07 | 1.01 | 1.97 | 1.00 | 1.71 | 1.85 | 1.57 |
| Other households' weight ratio | 0.998 | 0.791 | 0.998 | 0.809 | 1.001 | 0.861 | 0.833 | 0.889 |
| Borrower households' share of weight | 16.6% | 33.9% | 16.6% | 32.4% | 16.3% | 28.0% | 30.4% | 25.7% |
| Median household weight ratio | 0.02 | 4.5e-04 | 0.13 | 0.01 | 0.81 | 2.4e-06 | 2.3e-03 | 0.06 |
| 99th percentile weight ratio | 4.76 | 4.97 | 1.97 | 2.74 | 1.98 | 2.96 | 4.98 | 2.37 |
| Effective sample size (release 9,174) | 4,149 | 2,932 | 6,435 | 4,484 | 8,487 | 6,222 | 3,346 | 5,302 |

## FSA facts that do not bind

| Disposition | Facts | Reason |
|---|---:|---|
| allocated | 2 | 'Not Reported' has no state (FSA says it includes FFEL and Perkins loans whose commercial lenders need not report an address); it is allocated to every reported location in proportion to that location's reported value. |
| allocated | 2 | Age not reported: in no partition, so the reported bands, which bind as shares of the in-scope control, absorb it pro rata. |
| diagnostic | 12 | ED-held FFEL IBR by partial financial hardship: no plan or hardship column in the build; diagnostic only. |
| diagnostic | 45 | ED-held FFEL by repayment plan: no plan column in the build; diagnostic only. |
| diagnostic | 6 | FSA's notes don't say whether its Parent PLUS row leaves out loans the Secretary paid off through consolidation (34 CFR 685.220(f)), the stage's column holds every loan for a child's education, and every other family pins the total. |
| diagnostic | 24 | Federally managed IBR/PAYE by partial financial hardship: no plan or hardship column in the build; diagnostic only. |
| diagnostic | 65 | Federally managed portfolio by repayment plan: no plan column in the build; diagnostic only. |
| diagnostic | 24 | Income-driven plans by partial financial hardship: no plan or hardship column in the build; diagnostic only. |
| diagnostic | 24 | Recipient-based totals by loan program (Direct Loan, FFEL, Perkins). The stage's balances do not record the program, and FSA counts the student, not the parent, as the recipient of a Parent PLUS loan; the borrower-based location table sets the level instead. |
| diagnostic | 57 | Repayment plan: diagnostic only. The SCF records only an income-driven yes/no, the build carries no plan column, and FSA's recipient counts are loan-level (microcosm#1074). |
| excluded | 6 | Loan-type recipient counts are loan-level (a borrower with two loan types counts twice), so they are not counts of people. |
| excluded | 2 | Stafford (Combined) is a subtotal of rows the table already lists. |
| other_quarter | 58 | Not the reference quarter: every family binds at the latest quarter all four tables publish (2026-03). |
| out_of_scope | 2 | 'Other' is a borrower address outside the states and territories the table lists, outside the host universe; it leaves the in-scope control with its pro-rata share of Not Reported. |
| out_of_scope | 2 | Puerto Rico is outside the host universe (the CPS samples the 50 states and DC). Its rows, with their pro-rata share of Not Reported, leave the in-scope control. |

Materialization dropped 0 FSA targets.

## Every FSA target

| Target | FSA (reconciled) | Release | A | B | A + record L2 | B + record L2 | A + chi-square L2 | B + chi-square L2 | B, counts x0 | B + record L2, counts x0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Federal student loan balance, Alabama | $26.8B | $8.3B (-69.0%) | $7.3B (-72.7%) | $26.8B (-0.0%) | $7.9B (-70.7%) | $26.1B (-2.6%) | $8.3B (-69.2%) | $24.2B (-9.8%) | $26.8B (-0.0%) | $24.7B (-7.8%) |
| Federal student loan balance, Alaska | $2.5B | $1.1B (-53.6%) | $1.4B (-42.4%) | $2.5B (-0.1%) | $1.2B (-52.7%) | $2.2B (-11.9%) | $1.4B (-44.9%) | $2.4B (-4.1%) | $2.5B (-0.1%) | $2.1B (-17.1%) |
| Federal student loan balance, Arizona | $35.1B | $24.3B (-30.7%) | $14.4B (-59.0%) | $35.1B (+0.2%) | $18.5B (-47.1%) | $35.0B (-0.1%) | $18.2B (-48.1%) | $34.2B (-2.6%) | $35.0B (-0.1%) | $35.1B (+0.1%) |
| Federal student loan balance, Arkansas | $15.0B | $6.0B (-59.9%) | $6.1B (-59.3%) | $15.0B (-0.0%) | $6.3B (-57.6%) | $15.0B (+0.0%) | $6.1B (-59.3%) | $15.0B (+0.1%) | $14.9B (-0.1%) | $14.9B (-0.2%) |
| Federal student loan balance, California | $161.7B | $91.3B (-43.5%) | $99.2B (-38.6%) | $161.8B (+0.1%) | $96.8B (-40.1%) | $161.7B (-0.0%) | $90.9B (-43.8%) | $161.4B (-0.2%) | $161.6B (-0.0%) | $161.9B (+0.1%) |
| Federal student loan balance, Colorado | $30.6B | $12.2B (-60.0%) | $6.4B (-79.1%) | $30.7B (+0.0%) | $9.0B (-70.5%) | $30.6B (-0.2%) | $10.4B (-66.1%) | $28.6B (-6.8%) | $30.6B (-0.1%) | $30.6B (-0.3%) |
| Federal student loan balance, Connecticut | $20.4B | $8.2B (-59.8%) | $8.1B (-60.2%) | $20.4B (-0.0%) | $7.9B (-61.1%) | $20.4B (-0.1%) | $7.9B (-61.1%) | $20.3B (-0.4%) | $20.4B (-0.0%) | $20.3B (-0.5%) |
| Federal student loan balance, Delaware | $5.8B | $2.6B (-54.7%) | $3.3B (-42.6%) | $5.8B (-0.1%) | $2.9B (-49.4%) | $5.3B (-7.5%) | $2.9B (-49.1%) | $5.7B (-1.3%) | $5.8B (-0.1%) | $4.6B (-20.1%) |
| Federal student loan balance, District of Columbia | $6.6B | $2.6B (-60.6%) | $1.5B (-77.5%) | $6.6B (+0.1%) | $2.5B (-62.1%) | $6.6B (-0.3%) | $2.3B (-64.6%) | $6.6B (-0.2%) | $6.6B (-0.3%) | $6.3B (-5.0%) |
| Federal student loan balance, Florida | $116.7B | $42.9B (-63.2%) | $43.9B (-62.4%) | $116.6B (-0.1%) | $42.8B (-63.4%) | $116.4B (-0.2%) | $41.6B (-64.4%) | $93.9B (-19.6%) | $116.6B (-0.1%) | $116.7B (+0.0%) |
| Federal student loan balance, Georgia | $77.0B | $38.4B (-50.1%) | $51.9B (-32.5%) | $76.9B (-0.1%) | $33.3B (-56.8%) | $77.0B (+0.0%) | $38.0B (-50.6%) | $76.8B (-0.2%) | $77.0B (+0.0%) | $76.9B (-0.1%) |
| Federal student loan balance, Hawaii | $5.1B | $3.3B (-35.6%) | $3.2B (-36.0%) | $5.1B (+0.0%) | $3.2B (-36.7%) | $5.1B (-0.0%) | $3.1B (-39.1%) | $5.1B (+0.0%) | $5.1B (+0.1%) | $5.0B (-0.3%) |
| Federal student loan balance, Idaho | $7.7B | $6.2B (-20.5%) | $6.4B (-17.2%) | $7.8B (+0.2%) | $6.5B (-16.0%) | $7.8B (+0.2%) | $6.3B (-18.2%) | $7.7B (-0.1%) | $7.7B (+0.0%) | $7.7B (-0.1%) |
| Federal student loan balance, Illinois | $67.4B | $34.1B (-49.3%) | $50.4B (-25.2%) | $67.4B (+0.1%) | $47.7B (-29.2%) | $67.4B (-0.0%) | $39.5B (-41.3%) | $67.3B (-0.2%) | $67.4B (-0.0%) | $67.2B (-0.2%) |
| Federal student loan balance, Indiana | $32.0B | $25.3B (-20.8%) | $26.4B (-17.3%) | $32.0B (-0.0%) | $28.2B (-11.8%) | $32.0B (-0.1%) | $25.5B (-20.1%) | $32.0B (+0.1%) | $32.0B (+0.0%) | $32.0B (+0.1%) |
| Federal student loan balance, Iowa | $13.9B | $5.5B (-60.6%) | $6.5B (-53.1%) | $13.9B (+0.1%) | $6.1B (-56.5%) | $13.9B (-0.1%) | $6.1B (-56.5%) | $13.6B (-2.2%) | $13.9B (+0.0%) | $13.3B (-4.4%) |
| Federal student loan balance, Kansas | $13.5B | $6.5B (-52.1%) | $6.5B (-52.1%) | $13.5B (-0.0%) | $6.9B (-49.2%) | $13.5B (-0.4%) | $6.5B (-52.0%) | $13.3B (-1.7%) | $13.5B (-0.1%) | $12.6B (-6.4%) |
| Federal student loan balance, Kentucky | $22.3B | $12.8B (-42.6%) | $20.2B (-9.5%) | $22.3B (-0.1%) | $13.2B (-40.6%) | $22.3B (-0.1%) | $13.6B (-38.8%) | $22.3B (+0.1%) | $22.3B (-0.1%) | $22.3B (+0.1%) |
| Federal student loan balance, Louisiana | $25.9B | $6.1B (-76.6%) | $9.0B (-65.4%) | $24.2B (-6.7%) | $6.6B (-74.4%) | $19.7B (-23.8%) | $7.0B (-72.8%) | $17.9B (-31.0%) | $23.7B (-8.5%) | $15.3B (-41.1%) |
| Federal student loan balance, Maine | $6.9B | $4.0B (-42.0%) | $2.7B (-60.9%) | $6.9B (+0.1%) | $3.6B (-48.5%) | $6.9B (+0.0%) | $3.5B (-49.9%) | $6.9B (-0.3%) | $6.9B (+0.0%) | $6.9B (-0.1%) |
| Federal student loan balance, Maryland | $39.6B | $14.4B (-63.8%) | $15.2B (-61.5%) | $39.7B (+0.2%) | $14.4B (-63.6%) | $39.6B (-0.0%) | $13.9B (-64.8%) | $32.8B (-17.3%) | $39.6B (-0.1%) | $36.7B (-7.3%) |
| Federal student loan balance, Massachusetts | $34.5B | $19.3B (-44.0%) | $16.5B (-52.3%) | $34.4B (-0.1%) | $18.7B (-45.8%) | $34.4B (-0.1%) | $18.7B (-45.8%) | $34.4B (-0.2%) | $34.5B (+0.0%) | $34.5B (+0.2%) |
| Federal student loan balance, Michigan | $54.9B | $34.0B (-38.1%) | $39.5B (-28.0%) | $54.9B (+0.0%) | $34.5B (-37.1%) | $54.9B (+0.0%) | $34.2B (-37.6%) | $54.9B (+0.0%) | $55.0B (+0.1%) | $55.0B (+0.1%) |
| Federal student loan balance, Minnesota | $28.4B | $20.9B (-26.4%) | $21.5B (-24.1%) | $28.4B (-0.0%) | $22.3B (-21.3%) | $28.4B (-0.1%) | $21.5B (-24.3%) | $28.3B (-0.1%) | $28.4B (-0.0%) | $28.4B (+0.0%) |
| Federal student loan balance, Mississippi | $18.4B | $7.9B (-57.1%) | $6.3B (-65.8%) | $18.4B (-0.0%) | $7.9B (-56.9%) | $18.4B (-0.0%) | $8.0B (-56.3%) | $18.3B (-0.2%) | $18.4B (-0.0%) | $18.3B (-0.1%) |
| Federal student loan balance, Missouri | $31.6B | $15.7B (-50.2%) | $24.7B (-21.7%) | $31.6B (+0.1%) | $18.4B (-41.7%) | $31.6B (+0.2%) | $17.6B (-44.1%) | $31.5B (-0.2%) | $31.6B (+0.1%) | $31.7B (+0.3%) |
| Federal student loan balance, Montana | $4.6B | $3.6B (-23.5%) | $3.5B (-25.1%) | $4.6B (-0.1%) | $3.6B (-22.8%) | $4.6B (+0.1%) | $3.6B (-22.9%) | $4.6B (-0.1%) | $4.6B (+0.1%) | $4.6B (-0.1%) |
| Federal student loan balance, Nebraska | $8.5B | $3.9B (-54.3%) | $4.3B (-49.1%) | $8.5B (-0.0%) | $3.7B (-56.7%) | $8.5B (-0.0%) | $3.9B (-54.4%) | $8.5B (-0.0%) | $8.5B (-0.0%) | $8.5B (-0.0%) |
| Federal student loan balance, Nevada | $13.6B | $5.4B (-60.4%) | $6.8B (-50.2%) | $13.6B (-0.2%) | $6.0B (-55.9%) | $13.1B (-3.6%) | $5.6B (-58.5%) | $12.5B (-8.2%) | $13.6B (-0.3%) | $12.1B (-10.8%) |
| Federal student loan balance, New Hampshire | $7.0B | $3.6B (-48.4%) | $2.9B (-58.9%) | $7.0B (-0.1%) | $3.1B (-55.9%) | $7.0B (-0.3%) | $3.2B (-53.9%) | $6.9B (-1.7%) | $7.0B (-0.1%) | $7.0B (+0.1%) |
| Federal student loan balance, New Jersey | $49.9B | $22.9B (-54.1%) | $27.5B (-44.9%) | $50.0B (+0.1%) | $24.2B (-51.5%) | $50.0B (+0.1%) | $23.1B (-53.7%) | $49.8B (-0.2%) | $49.9B (-0.1%) | $49.9B (-0.2%) |
| Federal student loan balance, New Mexico | $8.3B | $9.6B (+15.8%) | $4.6B (-44.1%) | $8.3B (+0.0%) | $7.4B (-10.5%) | $8.2B (-0.2%) | $7.2B (-12.9%) | $8.3B (-0.0%) | $8.2B (-0.2%) | $8.3B (+0.0%) |
| Federal student loan balance, New York | $103.1B | $59.0B (-42.7%) | $52.1B (-49.4%) | $103.0B (-0.1%) | $54.5B (-47.1%) | $103.1B (-0.0%) | $54.4B (-47.3%) | $102.7B (-0.3%) | $103.1B (+0.0%) | $102.7B (-0.3%) |
| Federal student loan balance, North Carolina | $57.6B | $30.7B (-46.7%) | $48.5B (-15.8%) | $57.6B (-0.0%) | $34.4B (-40.2%) | $57.5B (-0.1%) | $33.0B (-42.6%) | $57.6B (+0.0%) | $57.6B (+0.1%) | $57.7B (+0.2%) |
| Federal student loan balance, North Dakota | $2.8B | $1.5B (-47.1%) | $1.6B (-42.6%) | $2.8B (+0.0%) | $1.6B (-44.3%) | $2.8B (-0.1%) | $1.5B (-47.7%) | $2.8B (-0.2%) | $2.8B (+0.0%) | $2.8B (-0.0%) |
| Federal student loan balance, Ohio | $66.3B | $24.9B (-62.4%) | $17.5B (-73.7%) | $66.4B (+0.0%) | $22.7B (-65.7%) | $66.2B (-0.2%) | $24.0B (-63.8%) | $53.9B (-18.7%) | $66.4B (+0.0%) | $66.3B (-0.1%) |
| Federal student loan balance, Oklahoma | $17.6B | $12.7B (-28.1%) | $10.4B (-41.1%) | $17.6B (+0.0%) | $11.9B (-32.6%) | $17.7B (+0.1%) | $12.1B (-31.3%) | $17.6B (-0.1%) | $17.6B (-0.1%) | $17.6B (-0.0%) |
| Federal student loan balance, Oregon | $21.5B | $9.8B (-54.5%) | $10.2B (-52.6%) | $21.4B (-0.1%) | $9.8B (-54.5%) | $21.5B (-0.0%) | $9.3B (-56.8%) | $21.4B (-0.1%) | $21.5B (+0.1%) | $21.4B (-0.3%) |
| Federal student loan balance, Pennsylvania | $71.8B | $27.9B (-61.1%) | $19.5B (-72.9%) | $71.7B (-0.1%) | $24.6B (-65.7%) | $71.7B (-0.2%) | $24.8B (-65.5%) | $67.3B (-6.2%) | $71.8B (-0.1%) | $71.9B (+0.1%) |
| Federal student loan balance, Rhode Island | $5.3B | $2.8B (-46.9%) | $2.2B (-58.6%) | $5.3B (-0.2%) | $2.7B (-48.8%) | $5.3B (-0.1%) | $2.6B (-50.1%) | $5.2B (-0.9%) | $5.3B (-0.0%) | $4.9B (-5.9%) |
| Federal student loan balance, South Carolina | $32.4B | $17.5B (-46.1%) | $14.4B (-55.5%) | $32.4B (+0.0%) | $13.3B (-58.9%) | $32.4B (-0.0%) | $15.0B (-53.7%) | $32.4B (-0.1%) | $32.4B (-0.1%) | $32.3B (-0.4%) |
| Federal student loan balance, South Dakota | $3.8B | $2.5B (-33.3%) | $1.2B (-67.7%) | $3.8B (-0.2%) | $1.8B (-52.5%) | $3.8B (+0.0%) | $1.8B (-53.7%) | $3.8B (-0.1%) | $3.8B (-0.2%) | $3.8B (-0.4%) |
| Federal student loan balance, Tennessee | $35.7B | $30.7B (-14.0%) | $34.8B (-2.6%) | $35.7B (+0.1%) | $30.9B (-13.5%) | $35.6B (-0.2%) | $30.1B (-15.7%) | $35.7B (-0.0%) | $35.7B (-0.0%) | $35.7B (+0.1%) |
| Federal student loan balance, Texas | $142.7B | $69.0B (-51.6%) | $44.8B (-68.6%) | $142.6B (-0.0%) | $58.3B (-59.2%) | $142.4B (-0.2%) | $68.1B (-52.3%) | $137.8B (-3.4%) | $142.7B (+0.0%) | $142.8B (+0.0%) |
| Federal student loan balance, Utah | $11.5B | $7.1B (-38.2%) | $9.2B (-19.8%) | $11.4B (-0.2%) | $8.6B (-24.6%) | $11.5B (+0.0%) | $8.6B (-25.2%) | $11.5B (+0.0%) | $11.4B (-0.0%) | $11.5B (+0.1%) |
| Federal student loan balance, Vermont | $3.0B | $1.5B (-49.1%) | $1.4B (-51.9%) | $3.0B (-0.1%) | $1.6B (-48.0%) | $3.0B (-0.2%) | $1.5B (-49.3%) | $3.0B (-0.3%) | $3.0B (-0.0%) | $3.0B (-0.2%) |
| Federal student loan balance, Virginia | $46.9B | $23.9B (-49.1%) | $37.9B (-19.2%) | $46.9B (-0.1%) | $31.0B (-34.0%) | $46.9B (-0.0%) | $26.7B (-43.1%) | $46.9B (-0.1%) | $46.7B (-0.6%) | $47.0B (+0.1%) |
| Federal student loan balance, Washington | $30.5B | $16.0B (-47.4%) | $13.4B (-56.1%) | $30.6B (+0.2%) | $15.4B (-49.5%) | $30.5B (+0.0%) | $15.5B (-49.2%) | $30.5B (-0.1%) | $30.6B (+0.0%) | $30.6B (+0.2%) |
| Federal student loan balance, West Virginia | $7.9B | $5.6B (-29.1%) | $6.4B (-19.2%) | $7.9B (+0.1%) | $5.6B (-29.9%) | $7.9B (+0.1%) | $5.7B (-28.5%) | $7.9B (-0.3%) | $7.9B (+0.1%) | $8.0B (+0.3%) |
| Federal student loan balance, Wisconsin | $24.8B | $11.6B (-53.1%) | $11.8B (-52.5%) | $24.7B (-0.1%) | $11.0B (-55.5%) | $24.7B (-0.2%) | $11.6B (-53.2%) | $24.8B (-0.0%) | $24.7B (-0.1%) | $24.8B (+0.0%) |
| Federal student loan balance, Wyoming | $1.9B | $799,653,011 (-56.9%) | $494,870,182 (-73.4%) | $1.9B (+0.1%) | $732,630,963 (-60.6%) | $1.8B (-0.4%) | $732,076,352 (-60.6%) | $1.8B (-1.0%) | $1.9B (-0.1%) | $1.8B (-4.3%) |
| Federal student loan borrowers, Alabama | 694,297 | 289,060 (-58.4%) | 243,215 (-65.0%) | 693,978 (-0.0%) | 291,274 (-58.0%) | 693,894 (-0.1%) | 294,882 (-57.5%) | 616,843 (-11.2%) | 570,816 (-17.8%) | 504,085 (-27.4%) |
| Federal student loan borrowers, Alaska | 67,443 | 42,678 (-36.7%) | 43,211 (-35.9%) | 67,443 (-0.0%) | 41,674 (-38.2%) | 67,372 (-0.1%) | 45,318 (-32.8%) | 67,428 (-0.0%) | 69,403 (+2.9%) | 62,942 (-6.7%) |
| Federal student loan borrowers, Arizona | 957,900 | 601,141 (-37.2%) | 497,847 (-48.0%) | 957,635 (-0.0%) | 571,502 (-40.3%) | 957,850 (-0.0%) | 563,579 (-41.2%) | 957,804 (-0.0%) | 1.01M (+5.2%) | 850,115 (-11.3%) |
| Federal student loan borrowers, Arkansas | 427,034 | 177,054 (-58.5%) | 163,361 (-61.7%) | 426,916 (-0.0%) | 178,336 (-58.2%) | 381,008 (-10.8%) | 179,112 (-58.1%) | 372,325 (-12.8%) | 316,039 (-26.0%) | 294,494 (-31.0%) |
| Federal student loan borrowers, California | 4.10M | 2.27M (-44.6%) | 2.46M (-40.0%) | 4.27M (+4.1%) | 2.27M (-44.6%) | 4.10M (+0.1%) | 2.30M (-44.0%) | 3.74M (-8.9%) | 5.56M (+35.7%) | 3.83M (-6.4%) |
| Federal student loan borrowers, Colorado | 799,487 | 315,383 (-60.6%) | 238,130 (-70.2%) | 735,765 (-8.0%) | 256,792 (-67.9%) | 683,236 (-14.5%) | 283,706 (-64.5%) | 619,234 (-22.5%) | 558,524 (-30.1%) | 570,322 (-28.7%) |
| Federal student loan borrowers, Connecticut | 538,917 | 209,988 (-61.0%) | 227,777 (-57.7%) | 501,053 (-7.0%) | 221,602 (-58.9%) | 393,990 (-26.9%) | 215,277 (-60.1%) | 389,453 (-27.7%) | 343,781 (-36.2%) | 334,669 (-37.9%) |
| Federal student loan borrowers, Delaware | 144,088 | 57,988 (-59.8%) | 63,045 (-56.2%) | 116,981 (-18.8%) | 58,815 (-59.2%) | 103,308 (-28.3%) | 59,610 (-58.6%) | 107,502 (-25.4%) | 93,766 (-34.9%) | 81,524 (-43.4%) |
| Federal student loan borrowers, District of Columbia | 119,829 | 54,421 (-54.6%) | 42,781 (-64.3%) | 119,587 (-0.2%) | 52,682 (-56.0%) | 99,710 (-16.8%) | 52,745 (-56.0%) | 100,556 (-16.1%) | 101,470 (-15.3%) | 85,385 (-28.7%) |
| Federal student loan borrowers, Florida | 2.87M | 1.01M (-64.9%) | 1.23M (-57.1%) | 2.87M (-0.0%) | 1.10M (-61.8%) | 2.87M (-0.1%) | 1.02M (-64.5%) | 1.88M (-34.5%) | 2.41M (-16.1%) | 2.18M (-24.1%) |
| Federal student loan borrowers, Georgia | 1.78M | 919,709 (-48.3%) | 950,383 (-46.6%) | 1.78M (-0.1%) | 812,952 (-54.3%) | 1.78M (+0.0%) | 883,048 (-50.4%) | 1.60M (-10.1%) | 1.51M (-15.4%) | 1.38M (-22.3%) |
| Federal student loan borrowers, Hawaii | 126,521 | 67,915 (-46.3%) | 69,764 (-44.9%) | 126,445 (-0.1%) | 64,639 (-48.9%) | 108,879 (-13.9%) | 64,102 (-49.3%) | 121,087 (-4.3%) | 90,422 (-28.5%) | 86,535 (-31.6%) |
| Federal student loan borrowers, Idaho | 226,378 | 169,349 (-25.2%) | 162,410 (-28.3%) | 226,691 (+0.1%) | 171,437 (-24.3%) | 226,360 (-0.0%) | 170,119 (-24.9%) | 226,281 (-0.0%) | 201,885 (-10.8%) | 191,389 (-15.5%) |
| Federal student loan borrowers, Illinois | 1.67M | 829,826 (-50.4%) | 1.02M (-38.9%) | 1.68M (+0.1%) | 967,554 (-42.2%) | 1.68M (+0.1%) | 888,286 (-47.0%) | 1.53M (-8.7%) | 1.75M (+4.8%) | 1.48M (-11.8%) |
| Federal student loan borrowers, Indiana | 939,497 | 495,461 (-47.3%) | 484,992 (-48.4%) | 940,490 (+0.1%) | 524,705 (-44.2%) | 894,635 (-4.8%) | 494,213 (-47.4%) | 788,172 (-16.1%) | 667,012 (-29.0%) | 612,441 (-34.8%) |
| Federal student loan borrowers, Iowa | 442,719 | 207,088 (-53.2%) | 244,868 (-44.7%) | 442,812 (+0.0%) | 246,415 (-44.3%) | 442,727 (+0.0%) | 232,243 (-47.5%) | 439,255 (-0.8%) | 391,584 (-11.6%) | 372,207 (-15.9%) |
| Federal student loan borrowers, Kansas | 396,607 | 216,541 (-45.4%) | 217,204 (-45.2%) | 396,283 (-0.1%) | 210,204 (-47.0%) | 395,795 (-0.2%) | 205,392 (-48.2%) | 395,745 (-0.2%) | 352,779 (-11.1%) | 327,238 (-17.5%) |
| Federal student loan borrowers, Kentucky | 643,689 | 292,355 (-54.6%) | 324,504 (-49.6%) | 642,982 (-0.1%) | 301,880 (-53.1%) | 502,096 (-22.0%) | 301,844 (-53.1%) | 460,475 (-28.5%) | 415,228 (-35.5%) | 388,865 (-39.6%) |
| Federal student loan borrowers, Louisiana | 725,666 | 233,743 (-67.8%) | 276,139 (-61.9%) | 725,608 (-0.0%) | 254,508 (-64.9%) | 695,302 (-4.2%) | 254,307 (-65.0%) | 569,783 (-21.5%) | 713,289 (-1.7%) | 480,569 (-33.8%) |
| Federal student loan borrowers, Maine | 195,114 | 118,655 (-39.2%) | 99,175 (-49.2%) | 174,720 (-10.5%) | 111,983 (-42.6%) | 155,052 (-20.5%) | 112,727 (-42.2%) | 151,904 (-22.1%) | 129,347 (-33.7%) | 138,730 (-28.9%) |
| Federal student loan borrowers, Maryland | 880,732 | 383,331 (-56.5%) | 358,717 (-59.3%) | 882,681 (+0.2%) | 348,840 (-60.4%) | 797,626 (-9.4%) | 356,349 (-59.5%) | 698,021 (-20.7%) | 668,101 (-24.1%) | 586,287 (-33.4%) |
| Federal student loan borrowers, Massachusetts | 941,693 | 461,478 (-51.0%) | 386,101 (-59.0%) | 941,377 (-0.0%) | 424,090 (-55.0%) | 939,561 (-0.2%) | 436,936 (-53.6%) | 838,923 (-10.9%) | 749,982 (-20.4%) | 664,562 (-29.4%) |
| Federal student loan borrowers, Michigan | 1.44M | 818,012 (-43.2%) | 837,603 (-41.8%) | 1.44M (+0.2%) | 845,311 (-41.3%) | 1.44M (+0.1%) | 846,726 (-41.2%) | 1.37M (-4.7%) | 1.35M (-6.1%) | 1.18M (-17.8%) |
| Federal student loan borrowers, Minnesota | 807,852 | 479,087 (-40.7%) | 492,375 (-39.1%) | 808,939 (+0.1%) | 498,244 (-38.3%) | 807,322 (-0.1%) | 474,126 (-41.3%) | 772,525 (-4.4%) | 628,115 (-22.2%) | 579,858 (-28.2%) |
| Federal student loan borrowers, Mississippi | 477,120 | 218,191 (-54.3%) | 224,237 (-53.0%) | 476,598 (-0.1%) | 238,167 (-50.1%) | 476,924 (-0.0%) | 237,876 (-50.1%) | 476,462 (-0.1%) | 439,802 (-7.8%) | 377,321 (-20.9%) |
| Federal student loan borrowers, Missouri | 864,211 | 467,783 (-45.9%) | 551,313 (-36.2%) | 866,484 (+0.3%) | 473,091 (-45.3%) | 827,194 (-4.3%) | 453,619 (-47.5%) | 704,641 (-18.5%) | 699,977 (-19.0%) | 638,546 (-26.1%) |
| Federal student loan borrowers, Montana | 132,167 | 91,338 (-30.9%) | 85,572 (-35.3%) | 132,185 (+0.0%) | 92,508 (-30.0%) | 132,066 (-0.1%) | 88,988 (-32.7%) | 132,010 (-0.1%) | 103,721 (-21.5%) | 104,696 (-20.8%) |
| Federal student loan borrowers, Nebraska | 254,610 | 121,030 (-52.5%) | 138,899 (-45.4%) | 254,518 (-0.0%) | 116,982 (-54.1%) | 243,594 (-4.3%) | 125,654 (-50.6%) | 254,434 (-0.1%) | 221,026 (-13.2%) | 190,403 (-25.2%) |
| Federal student loan borrowers, Nevada | 384,687 | 186,934 (-51.4%) | 222,037 (-42.3%) | 384,819 (+0.0%) | 209,749 (-45.5%) | 384,335 (-0.1%) | 196,542 (-48.9%) | 383,398 (-0.3%) | 370,051 (-3.8%) | 328,423 (-14.6%) |
| Federal student loan borrowers, New Hampshire | 196,264 | 99,529 (-49.3%) | 92,624 (-52.8%) | 196,037 (-0.1%) | 100,443 (-48.8%) | 177,348 (-9.6%) | 97,255 (-50.4%) | 195,856 (-0.2%) | 155,781 (-20.6%) | 141,874 (-27.7%) |
| Federal student loan borrowers, New Jersey | 1.29M | 434,985 (-66.3%) | 602,349 (-53.4%) | 1.33M (+2.7%) | 493,077 (-61.8%) | 1.14M (-11.4%) | 444,460 (-65.6%) | 825,646 (-36.1%) | 889,330 (-31.2%) | 794,059 (-38.5%) |
| Federal student loan borrowers, New Mexico | 236,312 | 153,854 (-34.9%) | 88,501 (-62.5%) | 236,372 (+0.0%) | 130,674 (-44.7%) | 234,934 (-0.6%) | 131,832 (-44.2%) | 231,400 (-2.1%) | 168,427 (-28.7%) | 145,589 (-38.4%) |
| Federal student loan borrowers, New York | 2.57M | 1.22M (-52.5%) | 1.19M (-53.7%) | 2.57M (-0.0%) | 1.25M (-51.3%) | 2.57M (+0.0%) | 1.17M (-54.5%) | 2.06M (-19.7%) | 2.39M (-7.0%) | 2.12M (-17.6%) |
| Federal student loan borrowers, North Carolina | 1.44M | 724,328 (-49.8%) | 944,014 (-34.5%) | 1.44M (-0.0%) | 781,496 (-45.8%) | 1.44M (-0.1%) | 736,742 (-48.9%) | 1.23M (-15.1%) | 1.42M (-1.6%) | 1.13M (-21.4%) |
| Federal student loan borrowers, North Dakota | 92,434 | 52,351 (-43.4%) | 64,873 (-29.8%) | 92,459 (+0.0%) | 56,986 (-38.3%) | 86,875 (-6.0%) | 57,164 (-38.2%) | 92,296 (-0.1%) | 77,813 (-15.8%) | 72,218 (-21.9%) |
| Federal student loan borrowers, Ohio | 1.85M | 688,756 (-62.8%) | 502,379 (-72.9%) | 1.75M (-5.5%) | 598,848 (-67.7%) | 1.56M (-15.7%) | 636,326 (-65.6%) | 1.12M (-39.4%) | 1.14M (-38.3%) | 1.07M (-42.1%) |
| Federal student loan borrowers, Oklahoma | 534,002 | 315,480 (-40.9%) | 256,421 (-52.0%) | 533,046 (-0.2%) | 278,947 (-47.8%) | 488,539 (-8.5%) | 287,292 (-46.2%) | 457,897 (-14.3%) | 364,342 (-31.8%) | 331,626 (-37.9%) |
| Federal student loan borrowers, Oregon | 550,628 | 301,707 (-45.2%) | 332,615 (-39.6%) | 551,119 (+0.1%) | 314,051 (-43.0%) | 551,927 (+0.2%) | 295,612 (-46.3%) | 551,104 (+0.1%) | 555,256 (+0.8%) | 541,097 (-1.7%) |
| Federal student loan borrowers, Pennsylvania | 1.94M | 633,413 (-67.3%) | 491,520 (-74.6%) | 1.95M (+0.4%) | 596,122 (-69.2%) | 1.79M (-7.7%) | 593,886 (-69.4%) | 1.21M (-37.5%) | 1.31M (-32.3%) | 1.19M (-38.6%) |
| Federal student loan borrowers, Rhode Island | 156,531 | 92,122 (-41.1%) | 70,723 (-54.8%) | 156,453 (-0.0%) | 85,639 (-45.3%) | 147,036 (-6.1%) | 85,603 (-45.3%) | 152,189 (-2.8%) | 125,733 (-19.7%) | 117,488 (-24.9%) |
| Federal student loan borrowers, South Carolina | 817,995 | 298,872 (-63.5%) | 271,050 (-66.9%) | 817,867 (-0.0%) | 263,051 (-67.8%) | 641,626 (-21.6%) | 278,393 (-66.0%) | 520,314 (-36.4%) | 568,292 (-30.5%) | 443,053 (-45.8%) |
| Federal student loan borrowers, South Dakota | 122,025 | 48,085 (-60.6%) | 35,747 (-70.7%) | 122,019 (-0.0%) | 44,802 (-63.3%) | 83,281 (-31.8%) | 44,975 (-63.1%) | 109,890 (-9.9%) | 72,259 (-40.8%) | 60,410 (-50.5%) |
| Federal student loan borrowers, Tennessee | 935,732 | 651,826 (-30.3%) | 756,414 (-19.2%) | 935,996 (+0.0%) | 700,270 (-25.2%) | 934,186 (-0.2%) | 659,940 (-29.5%) | 934,739 (-0.1%) | 936,414 (+0.1%) | 857,702 (-8.3%) |
| Federal student loan borrowers, Texas | 4.13M | 1.85M (-55.3%) | 1.83M (-55.8%) | 4.13M (+0.0%) | 1.97M (-52.4%) | 4.13M (-0.1%) | 1.91M (-53.9%) | 3.30M (-20.2%) | 4.35M (+5.4%) | 3.47M (-16.0%) |
| Federal student loan borrowers, Utah | 327,595 | 240,375 (-26.6%) | 274,153 (-16.3%) | 327,645 (+0.0%) | 279,576 (-14.7%) | 327,980 (+0.1%) | 273,713 (-16.4%) | 328,051 (+0.1%) | 372,679 (+13.8%) | 352,658 (+7.7%) |
| Federal student loan borrowers, Vermont | 78,213 | 49,621 (-36.6%) | 51,982 (-33.5%) | 78,147 (-0.1%) | 51,051 (-34.7%) | 78,218 (+0.0%) | 50,175 (-35.8%) | 78,240 (+0.0%) | 84,754 (+8.4%) | 78,036 (-0.2%) |
| Federal student loan borrowers, Virginia | 1.14M | 514,211 (-54.7%) | 511,957 (-54.9%) | 1.14M (+0.1%) | 505,720 (-55.4%) | 1.01M (-11.4%) | 513,258 (-54.8%) | 876,028 (-22.8%) | 751,021 (-33.8%) | 652,339 (-42.5%) |
| Federal student loan borrowers, Washington | 811,512 | 485,772 (-40.1%) | 383,931 (-52.7%) | 811,969 (+0.1%) | 445,269 (-45.1%) | 810,707 (-0.1%) | 461,752 (-43.1%) | 763,096 (-6.0%) | 716,112 (-11.8%) | 609,402 (-24.9%) |
| Federal student loan borrowers, West Virginia | 238,717 | 97,733 (-59.1%) | 110,576 (-53.7%) | 235,181 (-1.5%) | 104,567 (-56.2%) | 163,104 (-31.7%) | 102,201 (-57.2%) | 183,155 (-23.3%) | 154,690 (-35.2%) | 125,020 (-47.6%) |
| Federal student loan borrowers, Wisconsin | 741,455 | 308,645 (-58.4%) | 296,859 (-60.0%) | 741,873 (+0.1%) | 305,209 (-58.8%) | 718,159 (-3.1%) | 309,176 (-58.3%) | 571,448 (-22.9%) | 529,918 (-28.5%) | 460,505 (-37.9%) |
| Federal student loan borrowers, Wyoming | 57,300 | 34,488 (-39.8%) | 26,021 (-54.6%) | 57,366 (+0.1%) | 30,914 (-46.1%) | 57,191 (-0.2%) | 31,350 (-45.3%) | 56,938 (-0.6%) | 49,425 (-13.7%) | 43,979 (-23.2%) |
| Federal student loan balance, borrower age 24 and Younger | $84.9B | $62.6B (-26.2%) | $62.6B (-26.3%) | $84.8B (-0.1%) | $59.9B (-29.4%) | $84.8B (-0.1%) | $59.9B (-29.5%) | $84.8B (-0.2%) | $84.9B (+0.0%) | $84.9B (+0.0%) |
| Federal student loan balance, borrower age 25 to 34 | $462.0B | $224.0B (-51.5%) | $229.6B (-50.3%) | $461.9B (-0.0%) | $227.0B (-50.9%) | $461.8B (-0.0%) | $218.1B (-52.8%) | $424.1B (-8.2%) | $461.8B (-0.0%) | $461.6B (-0.1%) |
| Federal student loan balance, borrower age 35 to 49 | $679.5B | $350.0B (-48.5%) | $341.9B (-49.7%) | $677.8B (-0.2%) | $337.4B (-50.3%) | $670.5B (-1.3%) | $345.1B (-49.2%) | $656.3B (-3.4%) | $677.0B (-0.4%) | $658.9B (-3.0%) |
| Federal student loan balance, borrower age 50 to 61 | $310.2B | $161.2B (-48.0%) | $192.1B (-38.1%) | $310.2B (-0.0%) | $173.9B (-44.0%) | $310.1B (-0.0%) | $165.1B (-46.8%) | $310.1B (-0.0%) | $310.0B (-0.1%) | $310.1B (-0.0%) |
| Federal student loan balance, borrower age 62 and Older | $142.5B | $60.6B (-57.5%) | $59.8B (-58.0%) | $142.6B (+0.1%) | $57.6B (-59.6%) | $142.5B (+0.0%) | $58.6B (-58.9%) | $135.2B (-5.1%) | $142.6B (+0.0%) | $142.5B (-0.0%) |
| Federal student loan borrowers, borrower age 24 and Younger | 6.13M | 2.69M (-56.1%) | 2.76M (-55.0%) | 6.13M (-0.0%) | 2.75M (-55.2%) | 5.43M (-11.5%) | 2.71M (-55.7%) | 4.36M (-29.0%) | 4.75M (-22.5%) | 3.70M (-39.6%) |
| Federal student loan borrowers, borrower age 25 to 34 | 13.85M | 6.55M (-52.7%) | 6.71M (-51.5%) | 13.81M (-0.3%) | 6.65M (-52.0%) | 13.50M (-2.5%) | 6.46M (-53.3%) | 11.11M (-19.7%) | 12.49M (-9.8%) | 10.45M (-24.5%) |
| Federal student loan borrowers, borrower age 35 to 49 | 14.84M | 7.81M (-47.4%) | 8.08M (-45.5%) | 14.84M (+0.0%) | 7.96M (-46.3%) | 14.83M (-0.0%) | 7.83M (-47.3%) | 13.38M (-9.8%) | 14.46M (-2.5%) | 12.25M (-17.4%) |
| Federal student loan borrowers, borrower age 50 to 61 | 6.33M | 2.87M (-54.6%) | 2.81M (-55.7%) | 6.33M (-0.0%) | 2.79M (-55.9%) | 5.73M (-9.5%) | 2.81M (-55.7%) | 5.24M (-17.2%) | 5.23M (-17.4%) | 4.79M (-24.3%) |
| Federal student loan borrowers, borrower age 62 and Older | 3.16M | 1.17M (-63.0%) | 1.19M (-62.5%) | 3.16M (-0.0%) | 1.19M (-62.4%) | 2.94M (-7.0%) | 1.19M (-62.4%) | 2.51M (-20.6%) | 2.75M (-13.1%) | 2.52M (-20.2%) |
| Federal student loan balance, borrower debt size Less than 5K | $18.7B | $6.9B (-62.9%) | $7.0B (-62.7%) | $18.1B (-3.3%) | $7.1B (-62.3%) | $15.0B (-19.9%) | $6.8B (-63.4%) | $10.7B (-43.0%) | $11.0B (-41.0%) | $7.7B (-58.6%) |
| Federal student loan balance, borrower debt size 5K to 10K | $52.0B | $19.7B (-62.1%) | $17.8B (-65.8%) | $50.9B (-2.0%) | $18.8B (-63.9%) | $42.5B (-18.3%) | $19.3B (-62.8%) | $31.1B (-40.2%) | $35.8B (-31.2%) | $23.1B (-55.6%) |
| Federal student loan balance, borrower debt size 10K to 20K | $131.2B | $56.0B (-57.3%) | $56.6B (-56.8%) | $131.0B (-0.1%) | $57.7B (-56.0%) | $131.1B (-0.0%) | $56.3B (-57.1%) | $93.5B (-28.7%) | $131.2B (-0.0%) | $83.6B (-36.2%) |
| Federal student loan balance, borrower debt size 20K to 40K | $268.0B | $140.3B (-47.7%) | $158.0B (-41.1%) | $268.1B (+0.0%) | $146.2B (-45.5%) | $268.0B (-0.0%) | $140.6B (-47.5%) | $263.6B (-1.7%) | $268.1B (+0.0%) | $268.0B (-0.0%) |
| Federal student loan balance, borrower debt size 40K to 60K | $204.3B | $110.4B (-46.0%) | $115.1B (-43.6%) | $204.2B (-0.0%) | $114.2B (-44.1%) | $204.4B (+0.0%) | $111.6B (-45.4%) | $204.2B (-0.0%) | $204.5B (+0.1%) | $204.4B (+0.0%) |
| Federal student loan balance, borrower debt size 60K to 80K | $175.6B | $124.1B (-29.3%) | $130.2B (-25.9%) | $175.8B (+0.1%) | $128.5B (-26.8%) | $177.9B (+1.3%) | $125.0B (-28.8%) | $177.1B (+0.8%) | $175.7B (+0.0%) | $175.6B (+0.0%) |
| Federal student loan balance, borrower debt size 80K to 100K | $126.4B | $61.0B (-51.7%) | $51.4B (-59.3%) | $126.3B (-0.1%) | $56.2B (-55.5%) | $126.3B (-0.1%) | $59.5B (-52.9%) | $126.2B (-0.1%) | $126.4B (+0.0%) | $126.5B (+0.1%) |
| Federal student loan balance, borrower debt size 100K to 200K | $350.7B | $150.1B (-57.2%) | $154.6B (-55.9%) | $350.8B (+0.0%) | $144.8B (-58.7%) | $350.8B (+0.0%) | $145.1B (-58.6%) | $350.9B (+0.0%) | $358.5B (+2.2%) | $351.1B (+0.1%) |
| Federal student loan balance, borrower debt size 200K+ | $352.2B | $189.8B (-46.1%) | $195.3B (-44.5%) | $352.1B (-0.0%) | $182.3B (-48.2%) | $353.8B (+0.4%) | $182.4B (-48.2%) | $353.3B (+0.3%) | $365.1B (+3.7%) | $418.0B (+18.7%) |
| Federal student loan borrowers, borrower debt size Less than 5K | 6.82M | 2.70M (-60.5%) | 2.80M (-58.9%) | 6.67M (-2.3%) | 2.83M (-58.6%) | 5.85M (-14.3%) | 2.69M (-60.5%) | 4.21M (-38.3%) | 4.24M (-37.9%) | 3.13M (-54.2%) |
| Federal student loan borrowers, borrower debt size 5K to 10K | 7.12M | 2.81M (-60.6%) | 2.53M (-64.5%) | 7.12M (-0.0%) | 2.67M (-62.5%) | 5.97M (-16.2%) | 2.75M (-61.4%) | 4.38M (-38.5%) | 4.89M (-31.3%) | 3.21M (-54.9%) |
| Federal student loan borrowers, borrower debt size 10K to 20K | 9.10M | 3.88M (-57.3%) | 3.90M (-57.1%) | 9.02M (-0.9%) | 3.97M (-56.3%) | 9.00M (-1.0%) | 3.90M (-57.1%) | 6.46M (-29.0%) | 8.89M (-2.3%) | 5.67M (-37.7%) |
| Federal student loan borrowers, borrower debt size 20K to 40K | 9.40M | 5.08M (-46.0%) | 5.66M (-39.8%) | 9.37M (-0.2%) | 5.27M (-43.9%) | 9.45M (+0.6%) | 5.09M (-45.8%) | 9.39M (-0.0%) | 9.33M (-0.7%) | 9.37M (-0.2%) |
| Federal student loan borrowers, borrower debt size 40K to 60K | 4.15M | 2.27M (-45.4%) | 2.36M (-43.1%) | 4.18M (+0.6%) | 2.35M (-43.5%) | 4.20M (+1.0%) | 2.29M (-44.9%) | 4.18M (+0.6%) | 4.22M (+1.6%) | 4.18M (+0.7%) |
| Federal student loan borrowers, borrower debt size 60K to 80K | 2.57M | 1.80M (-29.8%) | 1.90M (-26.2%) | 2.55M (-0.7%) | 1.87M (-27.5%) | 2.57M (-0.0%) | 1.82M (-29.3%) | 2.57M (+0.1%) | 2.54M (-1.2%) | 2.53M (-1.5%) |
| Federal student loan borrowers, borrower debt size 80K to 100K | 1.38M | 707,878 (-48.9%) | 590,979 (-57.3%) | 1.45M (+4.6%) | 650,403 (-53.0%) | 1.45M (+5.0%) | 688,693 (-50.3%) | 1.46M (+5.2%) | 1.47M (+6.1%) | 1.46M (+5.6%) |
| Federal student loan borrowers, borrower debt size 100K to 200K | 2.57M | 1.16M (-54.9%) | 1.15M (-55.1%) | 2.59M (+0.8%) | 1.11M (-57.0%) | 2.64M (+2.7%) | 1.11M (-56.8%) | 2.67M (+3.8%) | 2.71M (+5.4%) | 2.65M (+3.0%) |
| Federal student loan borrowers, borrower debt size 200K+ | 1.19M | 692,617 (-41.6%) | 647,735 (-45.4%) | 1.32M (+11.1%) | 634,724 (-46.5%) | 1.29M (+8.9%) | 656,320 (-44.7%) | 1.28M (+7.5%) | 1.38M (+16.7%) | 1.51M (+27.2%) |
