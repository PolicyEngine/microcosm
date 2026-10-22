# microcosm#1073 receipts: SCF 2025 federal student loan balances

Generated 2026-10-22 by `experiments/build_1073_scf_education_debt_receipt.py` on branch `scf-2025-education-debt`, commit `2a6e65a739b79a8232e1177610d11928fe92e26c`, Modal run `20261022T153020Z-2a6e65a73` (`experiments/modal_1073.py`). Machine-readable copy: `experiments/receipts/1073-scf-education-debt-fit.json`.

Host columns impute the published default `populace-us-2024-spm-20260915` at its calibrated weights under three settings: the first receipt's construction (oldest adult, `A_MARITL` marriage, own children, whole-household income, every candidate predictor); the SCF-matching codings with every candidate predictor; and the build, which also leaves out interest and dividends and Social Security and pensions. The cause and the evidence are in the stage notes and `experiments/receipts/1073-predictor-shift-diagnosis.json`.

Every figure is weighted. SCF rows pool the five implicates with `wgt` (X42001/5). Household figures group by the reference person's age and race and by weighted quintiles of income less capital gains of the setting's income unit, the income concept that setting conditions on. Borrower figures count people with a federal balance (own plus Parent PLUS), in Federal Student Aid's balance and age bands; SCF borrowers are the reference person and spouse. Nothing here is a gate or a calibration target.

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

| Federal balances on loans 1-6, by whose education | Share |
|---|---:|
| 1 respondent (reference person) | 54.8% |
| 2 spouse or partner | 31.4% |
| 3 child (incl. grandchild, other relative) | 13.8% |
| 6 multiple people | 0.0% |
| 7 unrelated person | 0.0% |
| -7 other | 0.0% |

## Fit

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| Households with federal education debt | 18.2% | 17.9% | 14.4% | 14.5% | 16.4% |
| Mean among them | $41,052 | $38,130 | $44,483 | $42,174 | $42,680 |
| Median among them | $25,000 | $23,000 | $25,000 | $25,000 | $23,000 |
| Weighted federal education debt | $1.006T | $0.919T | $0.786T | $0.747T | $0.858T |
| Parent PLUS share of the total | 13.8% | 12.2% | 21.5% | 15.5% | 16.8% |
| Borrowers (people) | 27.2M | 25.7M | 18.5M | 18.5M | 21.2M |
| Mean balance per borrower | $36,934 | $35,733 | $42,554 | $40,307 | $40,447 |
| Median balance per borrower | $21,000 | $21,000 | $24,000 | $24,000 | $22,000 |
| IBR new borrowers among borrowers | 59.2% | 59.3% | 60.6% | 61.7% | 61.5% |
| Graduate share of own original principal | 47.7% | 45.0% | 53.6% | 49.8% | 48.3% |

### Borrowers by FSA balance band

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| under_5k | 11.9% | 10.2% | 10.4% | 11.8% | 12.5% |
| 5k_to_10k | 15.5% | 16.6% | 16.7% | 15.4% | 13.6% |
| 10k_to_20k | 18.6% | 18.7% | 15.5% | 16.1% | 19.0% |
| 20k_to_40k | 22.3% | 23.6% | 23.0% | 23.8% | 23.5% |
| 40k_to_60k | 12.8% | 12.0% | 10.4% | 11.5% | 10.8% |
| 60k_to_80k | 8.1% | 8.6% | 9.6% | 9.1% | 8.7% |
| 80k_to_100k | 3.7% | 3.9% | 5.1% | 3.9% | 3.2% |
| 100k_to_200k | 4.9% | 4.7% | 6.4% | 5.5% | 5.6% |
| 200k_plus | 2.2% | 1.6% | 2.9% | 2.9% | 3.1% |

### Borrowers by FSA age band

| | SCF 2025 | SCF hold-out (build) | Host: First receipt | Host: SCF codings, all predictors | Host: Build |
|---|---:|---:|---:|---:|---:|
| 24_and_younger | 5.7% | 6.5% | 3.8% | 5.5% | 5.0% |
| 25_to_34 | 33.1% | 33.9% | 26.6% | 30.9% | 29.2% |
| 35_to_49 | 38.1% | 38.0% | 39.8% | 37.3% | 37.8% |
| 50_to_61 | 16.2% | 15.1% | 21.1% | 19.4% | 20.8% |
| 62_and_older | 6.9% | 6.6% | 8.7% | 6.9% | 7.2% |

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
| 35_to_49 | $25,000 | $22,900 | $25,670 | $25,000 | $23,000 |
| 50_to_61 | $28,000 | $27,000 | $29,000 | $30,000 | $28,742 |
| 62_and_older | $26,000 | $26,997 | $38,000 | $26,000 | $34,000 |

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

Each variant refits both QRFs. Errors are drawn minus observed; TVD is total variation distance between band distributions; MAE averages the absolute error of the share with debt across the groups. `educ` is the SCF reference person's education, which the CPS side cannot supply until microcosm#297, so that row shows what the predictor would add.

| Variant | share_with_debt_error_pp | mean_among_borrowers_error_pct | median_among_borrowers_error_pct | household_balance_band_tvd | head_age_band_share_mae_pp | race_share_mae_pp | income_quintile_share_mae_pp | borrower_balance_band_tvd | borrower_age_band_tvd | household_rank_correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_8_without_age | -0.250 | -1.737 | -4.000 | 0.022 | 4.812 | 0.648 | 0.672 | 0.028 | 0.107 | 0.154 |
| shared_8 | -0.532 | -4.092 | -8.000 | 0.041 | 1.193 | 1.078 | 0.876 | 0.037 | 0.016 | 0.173 |
| shared_8_plus_household_income | -0.873 | -4.350 | -8.000 | 0.037 | 1.351 | 1.359 | 0.923 | 0.030 | 0.009 | 0.175 |
| plus_household_income_and_education | -1.474 | -5.567 | -4.816 | 0.040 | 1.321 | 2.067 | 1.474 | 0.050 | 0.023 | 0.208 |
| build_without_interest_and_ss_pension (build) | -0.304 | -7.119 | -8.000 | 0.032 | 1.260 | 1.069 | 0.675 | 0.033 | 0.015 | 0.168 |
| build_plus_education | -0.920 | -7.087 | -8.000 | 0.028 | 1.749 | 1.789 | 0.920 | 0.029 | 0.031 | 0.207 |

## Host fit, before and after

Imputed with build seed 0 onto `populace-us-2024-spm-20260915` (57,240 households, sha256 `6496cc4393d4...`) at its calibrated household weights; the published weights were calibrated without these columns. Build signal gate: passed. Errors are imputed minus SCF 2025.

| Setting | Households with debt | Parent PLUS share | share_with_debt_error_pp | mean_among_borrowers_error_pct | median_among_borrowers_error_pct | household_balance_band_tvd | head_age_band_share_mae_pp | race_share_mae_pp | income_quintile_share_mae_pp | borrower_balance_band_tvd | borrower_age_band_tvd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First receipt | 14.4% | 21.5% | -3.742 | 8.357 | 0.000 | 0.062 | 3.554 | 2.834 | 4.252 | 0.070 | 0.085 |
| SCF codings, all predictors | 14.5% | 15.5% | -3.701 | 2.732 | 0.000 | 0.040 | 4.170 | 2.570 | 4.568 | 0.040 | 0.032 |
| Build | 16.4% | 16.8% | -1.741 | 3.966 | -8.000 | 0.046 | 2.391 | 0.942 | 2.866 | 0.045 | 0.050 |
| SCF 2025 | 18.2% | 13.8% |  |  |  |  |  |  |  |  |  |
