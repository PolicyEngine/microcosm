# microcosm#1073 receipts: SCF 2025 federal student loan balances

Generated 2026-11-04 by `experiments/build_1073_scf_education_debt_receipt.py` on branch `scf-2025-education-debt`, commit `63776f2ce39e68bd88b47e7a8a3bb3137c30a8ba`, Modal run `20261104T190152Z-63776f2ce` (`experiments/modal_1073.py`). Machine-readable copy: `experiments/receipts/1073-scf-education-debt-fit.json`.

Host columns impute the published default `populace-us-2024-spm-20260915` at its calibrated weights under three settings: the first receipt's construction (oldest adult, `A_MARITL` marriage, own children, whole-household income, every candidate predictor); the SCF-matching codings with every candidate predictor; and the build, which also leaves out interest and dividends and Social Security and pensions. The cause and the evidence are in the stage notes and `experiments/receipts/1073-predictor-shift-diagnosis.json`.

Every figure is weighted. SCF rows pool the five implicates with `wgt` (X42001/5). Household figures group by the reference person's age and race and by weighted quintiles of income less capital gains of the setting's income unit, the income concept that setting conditions on. Borrower figures count people with a federal balance (own plus Parent PLUS), in Federal Student Aid's balance and age bands; SCF borrowers are the reference person, the spouse or partner and an adult child holding a loan for their own education, whose public-file age band (18-27 reads 25) is spread evenly over its years. Nothing here is a gate or a calibration target.

## Source checks

| Check | Loader | Fed Alternate Table 13 |
|---|---:|---:|
| Families with any education debt | 19.92% | 19.92% |
| Median education debt among them | $25,500 | $25,500 |

| Share of education-loan balances | Value |
|---|---:|
| loans 1 6 of all education debt | 99.4% |
| x7179 remaining loans of all education debt | 0.6% |
| federal of loans 1 6 | 87.4% |
| non federal of loans 1 6 | 12.0% |
| federal status blank of loans 1 6 | 0.6% |

| Federal balances on loans 1-6, by whose education (codes as published) | Share |
|---|---:|
| 1 respondent (reference person) | 54.8% |
| 2 spouse or partner | 31.4% |
| 3 child (incl. grandchild, other relative) | 13.8% |
| 6 multiple people | 0.0% |
| 7 unrelated person | 0.0% |
| -7 other | 0.0% |

### X8000: are the whose-education codes reversed?

Completed degree loans coded 1 or 2 in families with a spouse or partner. A loan discriminates when it is consistent with exactly one partner's highest degree (X5931, X6111: at least the program's level); the last columns count those consistent with the coded person (code 1 the reference person, code 2 the spouse or partner), as published, before the stage swaps codes 1 and 2 of loans 4 and 6 in reversed families. Counts pool the five implicates (implicate 1 in brackets).

| Loans | X8000 | Code | Discriminating | Consistent with the coded person |
|---|---|---:|---:|---:|
| 1, 2, 3, 5 | not reversed | 1 | 180 (36) | 175 (35) |
| 1, 2, 3, 5 | not reversed | 2 | 208 (40) | 203 (40) |
| 1, 2, 3, 5 | reversed | 1 | 73 (15) | 70 (14) |
| 1, 2, 3, 5 | reversed | 2 | 470 (92) | 470 (92) |
| 4, 6 | not reversed | 1 | 20 (4) | 20 (4) |
| 4, 6 | not reversed | 2 | 5 (1) | 5 (1) |
| 4, 6 | reversed | 1 | 49 (9) | 0 (0) |
| 4, 6 | reversed | 2 | 5 (1) | 0 (0) |
| 4 | not reversed | 1 | 15 (3) | 15 (3) |
| 4 | not reversed | 2 | 5 (1) | 5 (1) |
| 4 | reversed | 1 | 34 (6) | 0 (0) |
| 4 | reversed | 2 | 5 (1) | 0 (0) |
| 6 | not reversed | 1 | 5 (1) | 5 (1) |
| 6 | not reversed | 2 | 0 (0) | 0 (0) |
| 6 | reversed | 1 | 15 (3) | 0 (0) |
| 6 | reversed | 2 | 0 (0) | 0 (0) |

The swap changes the holder of 0.3% of federal balances ($2,781,387,400; 80 loans pooled, 16 in implicate 1): loan 4 0.2% (60 loans), loan 6 0.0% (20 loans).

### Loans for a child's education (SCF 2025, weighted)

FSA's Direct Parent PLUS balance is 6.7% of its portfolio at FY2026 Q3, a floor, since consolidated Parent PLUS loans sit in the consolidation total.

| Reading | Parent PLUS share | Child's own share | Parent PLUS holders | Child borrowers | Borrowers |
|---|---:|---:|---:|---:|---:|
| Before: every code-3 loan Parent PLUS, codes as published | 13.8% | 0.0% | 4.4M | 0.0M | 27.2M |
| X8000 correction only | 13.8% | 0.0% | 4.4M | 0.0M | 26.9M |
| Build: co-resident adult child unless only a parent could borrow | 9.4% | 4.3% | 2.2M | 2.2M | 27.2M |
| Each loan against the dependent limit alone, for an only fitting child too | 9.0% | 4.7% | 2.1M | 2.3M | 27.2M |
| Alternative: within-limit undergraduate loans for a dependent-age child also Parent PLUS | 11.7% | 2.0% | 4.1M | 0.4M | 27.1M |
| Previous limit: $31,000 for every loan year | 9.4% | 4.3% | 2.2M | 2.2M | 27.2M |
| Alternative: over-the-limit loans the child's own up to $46,000 before July 2008 and $57,500 from then | 8.6% | 5.1% | 2.0M | 2.5M | 27.2M |

Federal loans for a child's education (code 3) that each rule moves. Loans pool the five implicates (implicate 1 in brackets); weighted loans and dollars use `wgt`, after the X7179 rule. The earliest such loan was borrowed in 1989 (any federal loan: 1983).

| Rule | Loans | Weighted loans | Dollars | Share of federal |
|---|---:|---:|---:|---:|
| Graduate loans no listed child fits: head's own, not Parent PLUS | 15 (3) | 30,525 | $0.82B | 0.1% |
| Loan-year limit: child's own under a flat $31,000, now Parent PLUS | 0 (0) | 0 | $0.00B | 0.0% |
| Loan-year limit: Parent PLUS under a flat $31,000, now the child's | 0 (0) | 0 | $0.00B | 0.0% |
| Parent PLUS although a listed child fits (over the dependent limit) | 121 (24) | 716,437 | $29.98B | 3.0% |
| Of those, the child's own under a $46,000 (before July 2008) or $57,500 limit | 73 (14) | 407,767 | $7.77B | 0.8% |
| Each within its year's limit, but over what the child's total through that year leaves | 49 (9) | 408,474 | $5.31B | 0.5% |
| Of those, Parent PLUS in the build (one listed child fits) | 27 (5) | 270,979 | $3.79B | 0.4% |
| Of those, the child's in the build (two or more listed children fit) | 22 (4) | 137,494 | $1.52B | 0.2% |
| Of those, Parent PLUS with each loan against the limit alone | 0 (0) | 0 | $0.00B | 0.0% |
| Of those, the child's with each loan against the limit alone | 49 (9) | 408,474 | $5.31B | 0.5% |
| Any loan the child's with each loan alone, Parent PLUS in the build | 27 (5) | 270,979 | $3.79B | 0.4% |
| Undergraduate, before 2008, fitting a child who could be dependent | 0 (0) | 0 | $0.00B | 0.0% |
| Of those, within their year's limit | 0 (0) | 0 | $0.00B | 0.0% |
| Of those, over their year's limit but within $31,000 | 0 (0) | 0 | $0.00B | 0.0% |
| Of those, over $31,000 | 0 (0) | 0 | $0.00B | 0.0% |
| All loans for a child borrowed before 1993 | 5 (1) | 62,470 | $0.16B | 0.0% |
| All loans for a child borrowed 1993-2007 | 52 (10) | 366,000 | $12.60B | 1.3% |
| All loans for a child borrowed 2008 or later | 938 (188) | 5,994,996 | $125.55B | 12.5% |

IBR new-borrower status is an upper bound: the SCF sees only loans still owed, so paid-off pre-2014 loans are missed, and a consolidated loan is dated by its most recent underlying loan. 29.3% of federal balances dated 2014 or later are on a loan that has been consolidated or refinanced (X9320-X9325).

## Parent PLUS and borrowers against Federal Student Aid

Before is the previous rule (every loan for a child's education the head's Parent PLUS loan, the residual of the composition draw); the host's before column is the build in the previous receipt (commit `2a6e65a73`, Modal run `20261022T153020Z-2a6e65a73`). FSA's Parent PLUS share is a floor; its dollars are the whole federal portfolio, of which household surveys capture part, so the shares compare and the levels do not. FSA borrowers come from its by-age table (FY2026 Q2), which counts the parent on a Parent PLUS loan; its 42.3M unduplicated recipients (FY2026 Q3) count the student instead.

| | FSA | SCF 2025, before | SCF 2025 | Host: Build, before | Host: Build |
|---|---:|---:|---:|---:|---:|
| Parent PLUS share of federal balances | 6.7% (floor) | 13.8% | 9.4% | 16.8% | 10.7% |
| Parent PLUS balances | $115.0B | $138.3B | $94.2B | $143.9B | $92.1B |
| Federal balances | $1,724.5B | $1,005.9B | $1,005.9B | $858.4B | $858.4B |
| Borrowers (people) | 44.9M | 27.2M | 27.2M | 21.2M | 21.1M |
| Borrowers aged 24 and younger | 13.8% | 5.7% | 10.7% | 5.0% | 12.8% |
| Borrowers aged 25 to 34 | 31.2% | 33.1% | 35.2% | 29.2% | 31.1% |
| Borrowers aged 35 to 49 | 33.5% | 38.1% | 37.4% | 37.8% | 37.0% |
| Borrowers aged 50 to 61 | 14.3% | 16.2% | 11.4% | 20.8% | 13.6% |
| Borrowers aged 62 and older | 7.1% | 6.9% | 5.4% | 7.2% | 5.6% |

## Fit

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| Households with federal education debt | 18.2% | 17.9% | 14.4% | 14.5% | 16.4% |
| Mean among them | $41,052 | $38,156 | $44,483 | $42,174 | $42,680 |
| Median among them | $25,000 | $23,000 | $25,000 | $25,000 | $23,000 |
| Weighted federal education debt | $1.006T | $0.919T | $0.786T | $0.747T | $0.858T |
| Parent PLUS share of the total | 9.4% | 6.8% | 11.1% | 11.3% | 10.7% |
| Borrowers (people) | 27.2M | 25.8M | 18.6M | 18.4M | 21.1M |
| Mean balance per borrower | $36,931 | $35,590 | $42,148 | $40,513 | $40,688 |
| Median balance per borrower | $21,000 | $21,000 | $25,000 | $25,000 | $22,224 |
| IBR new borrowers among borrowers (upper bound) | 59.5% | 58.0% | 58.8% | 60.7% | 61.8% |
| Graduate share of own original principal | 47.4% | 44.6% | 50.9% | 47.8% | 48.2% |

### Borrowers by FSA balance band

FSA: borrowers by debt size, FY2026 Q2.

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build | FSA |
|---|---:|---:|---:|---:|---:|---:|
| under_5k | 11.9% | 10.7% | 10.6% | 11.6% | 12.8% | 15.4% |
| 5k_to_10k | 15.5% | 16.1% | 16.1% | 15.5% | 13.3% | 16.1% |
| 10k_to_20k | 18.2% | 18.2% | 15.2% | 15.7% | 18.4% | 20.5% |
| 20k_to_40k | 22.8% | 24.1% | 23.8% | 23.9% | 24.1% | 21.2% |
| 40k_to_60k | 13.0% | 12.9% | 10.7% | 11.9% | 10.7% | 9.4% |
| 60k_to_80k | 7.8% | 8.3% | 9.3% | 8.7% | 8.6% | 5.8% |
| 80k_to_100k | 3.7% | 3.7% | 5.0% | 3.7% | 3.4% | 3.1% |
| 100k_to_200k | 4.9% | 4.3% | 6.4% | 6.0% | 5.5% | 5.8% |
| 200k_plus | 2.2% | 1.8% | 2.9% | 2.9% | 3.3% | 2.7% |

### Borrowers by FSA age band

FSA: borrowers by age, FY2026 Q2, of those with an age reported.

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build | FSA |
|---|---:|---:|---:|---:|---:|---:|
| 24_and_younger | 10.7% | 10.5% | 12.2% | 12.1% | 12.8% | 13.8% |
| 25_to_34 | 35.2% | 36.2% | 30.3% | 31.9% | 31.1% | 31.2% |
| 35_to_49 | 37.4% | 36.6% | 37.5% | 37.4% | 37.0% | 33.5% |
| 50_to_61 | 11.4% | 10.9% | 13.6% | 13.2% | 13.6% | 14.3% |
| 62_and_older | 5.4% | 5.8% | 6.4% | 5.4% | 5.6% | 7.1% |

### Households with federal education debt, by age of reference person

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| 24_and_younger | 22.2% | 25.0% | 22.7% | 21.1% | 21.8% |
| 25_to_34 | 36.2% | 36.9% | 26.7% | 27.1% | 28.1% |
| 35_to_49 | 27.2% | 27.4% | 22.3% | 20.9% | 24.4% |
| 50_to_61 | 16.7% | 14.8% | 14.2% | 13.3% | 16.4% |
| 62_and_older | 4.2% | 3.5% | 3.7% | 3.1% | 3.7% |

Median among borrowers:

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| 24_and_younger | $10,000 | $14,000 | $6,000 | $14,000 | $12,662 |
| 25_to_34 | $24,900 | $23,000 | $20,200 | $21,600 | $20,100 |
| 35_to_49 | $25,000 | $22,000 | $25,670 | $25,000 | $23,000 |
| 50_to_61 | $28,000 | $27,000 | $29,000 | $30,000 | $28,742 |
| 62_and_older | $26,000 | $27,000 | $38,000 | $26,000 | $34,000 |

### Households with federal education debt, by race (`racecl4`)

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| white_non_hispanic | 17.2% | 16.7% | 13.3% | 13.3% | 15.3% |
| black_non_hispanic | 24.3% | 23.2% | 22.8% | 22.8% | 24.9% |
| hispanic | 14.1% | 16.2% | 12.3% | 11.8% | 13.8% |
| other_or_multiple | 19.5% | 18.9% | 15.3% | 17.0% | 18.4% |

Median among borrowers:

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| white_non_hispanic | $23,000 | $23,000 | $25,000 | $25,000 | $23,000 |
| black_non_hispanic | $30,000 | $25,000 | $28,000 | $28,000 | $28,000 |
| hispanic | $23,000 | $20,000 | $24,764 | $23,000 | $19,200 |
| other_or_multiple | $20,000 | $21,000 | $24,000 | $22,000 | $18,000 |

### Households with federal education debt, by income quintile

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| 1 | 11.4% | 10.4% | 12.7% | 13.6% | 14.3% |
| 2 | 17.1% | 17.8% | 13.6% | 14.6% | 15.8% |
| 3 | 17.8% | 18.0% | 13.4% | 13.0% | 14.7% |
| 4 | 26.6% | 26.3% | 20.0% | 19.2% | 23.1% |
| 5 | 17.9% | 16.8% | 12.4% | 11.8% | 14.2% |

Median among borrowers:

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| 1 | $19,200 | $19,200 | $20,000 | $20,000 | $20,000 |
| 2 | $22,000 | $21,000 | $26,000 | $23,000 | $20,000 |
| 3 | $21,000 | $21,000 | $23,000 | $23,000 | $23,000 |
| 4 | $28,000 | $25,000 | $25,000 | $26,000 | $26,000 |
| 5 | $28,000 | $28,000 | $28,000 | $26,814 | $23,000 |

## Predictor ablation (five-fold hold-out on SCF families)

Each variant refits both QRFs. Errors are drawn minus observed; TVD is total variation distance between band distributions; MAE averages the absolute error of the share with debt across the groups. `educ` is the SCF reference person's education, which the CPS side cannot supply until microcosm#297, so that row shows what the predictor would add. The last column is the weighted share of held-out families observed with federal debt but no Parent PLUS that are drawn some Parent PLUS anyway.

| Variant | share_with_debt_error_pp | mean_among_borrowers_error_pct | median_among_borrowers_error_pct | household_balance_band_tvd | head_age_band_share_mae_pp | race_share_mae_pp | income_quintile_share_mae_pp | borrower_balance_band_tvd | borrower_age_band_tvd | household_rank_correlation | parent_plus_drawn_without_observed_parent_plus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_8_without_age | -0.250 | -1.707 | -4.000 | 0.023 | 4.812 | 0.648 | 0.672 | 0.033 | 0.125 | 0.154 | 0.048 |
| shared_8 | -0.532 | -4.099 | -8.000 | 0.041 | 1.193 | 1.078 | 0.876 | 0.038 | 0.021 | 0.173 | 0.024 |
| shared_8_plus_household_income | -0.873 | -4.378 | -8.000 | 0.036 | 1.351 | 1.359 | 0.923 | 0.031 | 0.012 | 0.175 | 0.025 |
| plus_household_income_and_education | -1.474 | -5.582 | -4.816 | 0.040 | 1.321 | 2.067 | 1.474 | 0.056 | 0.033 | 0.208 | 0.018 |
| build_without_interest_and_ss_pension (build) | -0.304 | -7.054 | -8.000 | 0.030 | 1.260 | 1.069 | 0.675 | 0.023 | 0.014 | 0.168 | 0.027 |
| build_plus_education | -0.920 | -7.077 | -8.000 | 0.028 | 1.749 | 1.789 | 0.920 | 0.037 | 0.035 | 0.207 | 0.020 |

## Child shares drawn for households without an adult child

Such a share is a loan for a student who lives elsewhere: its graduate part (by the drawn child graduate share) joins the head's own balance as graduate debt, and the rest is the head's Parent PLUS loan.

| Draw | Households | Share of borrower households | Graduate part to the head's own balance | Undergraduate part to Parent PLUS |
|---|---:|---:|---:|---:|
| SCF hold-out (build) | 0.00M | 0.0% | $0.00B (0.0%) | $0.00B (0.0%) |
| Host: Build | 0.00M | 0.0% | $0.00B (0.0%) | $0.00B (0.0%) |

The host row repeats the build's draw. It matches the build in its weighted total (yes) and, person by person, in the four balance columns placed on the build's people: own balances yes, Parent PLUS balances yes.

## Host fit, before and after

Imputed with build seed 0 onto `populace-us-2024-spm-20260915` (57,240 households, sha256 `6496cc4393d4...`) at its calibrated household weights; the published weights were calibrated without these columns. Build signal gate: passed. Errors are imputed minus SCF 2025.

| Setting | Households with debt | Parent PLUS share | share_with_debt_error_pp | mean_among_borrowers_error_pct | median_among_borrowers_error_pct | household_balance_band_tvd | head_age_band_share_mae_pp | race_share_mae_pp | income_quintile_share_mae_pp | borrower_balance_band_tvd | borrower_age_band_tvd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First receipt | 14.4% | 11.1% | -3.742 | 8.357 | 0.000 | 0.062 | 3.554 | 2.834 | 4.252 | 0.065 | 0.049 |
| SCF codings, all predictors | 14.5% | 11.3% | -3.701 | 2.732 | 0.000 | 0.040 | 4.170 | 2.570 | 4.568 | 0.039 | 0.034 |
| Build | 16.4% | 10.7% | -1.741 | 3.966 | -8.000 | 0.046 | 2.391 | 0.942 | 2.866 | 0.048 | 0.045 |
| SCF 2025 | 18.2% | 9.4% |  |  |  |  |  |  |  |  |  |

SCF listing code 4 also counts children-in-law, so the SCF's adult-children predictor can count a son- or daughter-in-law that the host's CPS parent pointers never do. On the host, 1.0% of households have an adult child's spouse or cohabiting partner present (1.27M people: 1.12M named by `A_SPOUSE` and 0.14M by `PECOHAB`; against 33.63M adult children in 20.6% of households).
