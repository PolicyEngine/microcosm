# microcosm#1073 receipts: SCF 2025 federal student loan balances

Generated 2026-10-12 by `experiments/build_1073_scf_education_debt_receipt.py` on branch `scf-2025-education-debt`. Machine-readable copy: `experiments/receipts/1073-scf-education-debt-fit.json`.

Every figure is weighted. SCF rows pool the five implicates with `wgt` (X42001/5). Household figures group by the reference person's age and race and by weighted quintiles of household income less capital gains, the income concept the imputation conditions on. Borrower figures count people with a federal balance (own plus Parent PLUS), in Federal Student Aid's balance and age bands; SCF borrowers are the reference person and spouse. Nothing here is a gate or a calibration target.

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

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| Households with federal education debt | 18.2% | 17.3% | 14.4% |
| Mean among them | $41,052 | $39,267 | $44,483 |
| Median among them | $25,000 | $23,000 | $25,000 |
| Weighted federal education debt | $1.006T | $0.916T | $0.786T |
| Parent PLUS share of the total | 13.8% | 11.6% | 21.5% |
| Borrowers (people) | 27.2M | 24.7M | 18.5M |
| Mean balance per borrower | $36,934 | $37,030 | $42,554 |
| Median balance per borrower | $21,000 | $22,000 | $24,000 |
| IBR new borrowers among borrowers | 59.2% | 59.2% | 60.6% |
| Graduate share of own original principal | 47.7% | 46.2% | 53.6% |

### Borrowers by FSA balance band

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| under_5k | 11.9% | 10.8% | 10.4% |
| 5k_to_10k | 15.5% | 15.2% | 16.7% |
| 10k_to_20k | 18.6% | 18.3% | 15.5% |
| 20k_to_40k | 22.3% | 24.0% | 23.0% |
| 40k_to_60k | 12.8% | 12.2% | 10.4% |
| 60k_to_80k | 8.1% | 9.4% | 9.6% |
| 80k_to_100k | 3.7% | 3.5% | 5.1% |
| 100k_to_200k | 4.9% | 4.8% | 6.4% |
| 200k_plus | 2.2% | 1.9% | 2.9% |

### Borrowers by FSA age band

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| 24_and_younger | 5.7% | 6.1% | 3.8% |
| 25_to_34 | 33.1% | 33.3% | 26.6% |
| 35_to_49 | 38.1% | 38.4% | 39.8% |
| 50_to_61 | 16.2% | 15.8% | 21.1% |
| 62_and_older | 6.9% | 6.4% | 8.7% |

### Households with federal education debt, by age of reference person

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| 24_and_younger | 22.2% | 24.4% | 22.7% |
| 25_to_34 | 36.2% | 34.7% | 26.7% |
| 35_to_49 | 27.2% | 27.2% | 22.3% |
| 50_to_61 | 16.7% | 14.5% | 14.2% |
| 62_and_older | 4.2% | 3.3% | 3.7% |

Median among borrowers:

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| 24_and_younger | $10,000 | $11,000 | $6,000 |
| 25_to_34 | $24,900 | $24,600 | $20,200 |
| 35_to_49 | $25,000 | $22,000 | $25,670 |
| 50_to_61 | $28,000 | $28,000 | $29,000 |
| 62_and_older | $26,000 | $28,000 | $38,000 |

### Households with federal education debt, by race (`racecl4`)

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| white_non_hispanic | 17.2% | 15.9% | 13.3% |
| black_non_hispanic | 24.3% | 22.5% | 22.8% |
| hispanic | 14.1% | 16.1% | 12.3% |
| other_or_multiple | 19.5% | 19.2% | 15.3% |

Median among borrowers:

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| white_non_hispanic | $23,000 | $24,000 | $25,000 |
| black_non_hispanic | $30,000 | $24,600 | $28,000 |
| hispanic | $23,000 | $20,000 | $24,764 |
| other_or_multiple | $20,000 | $22,000 | $24,000 |

### Households with federal education debt, by income quintile

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| 1 | 11.4% | 10.3% | 12.7% |
| 2 | 17.1% | 17.1% | 13.6% |
| 3 | 17.8% | 17.9% | 13.4% |
| 4 | 26.6% | 25.5% | 20.0% |
| 5 | 17.9% | 15.7% | 12.4% |

Median among borrowers:

| | SCF 2025 | SCF hold-out draw | Imputed onto populace-us-2024-spm-20260915 |
|---|---:|---:|---:|
| 1 | $19,200 | $19,200 | $20,000 |
| 2 | $22,000 | $22,000 | $26,000 |
| 3 | $21,000 | $23,000 | $23,000 |
| 4 | $28,000 | $25,000 | $25,000 |
| 5 | $28,000 | $26,000 | $28,000 |

## Predictor ablation (five-fold hold-out on SCF families)

Each variant refits both QRFs. Errors are drawn minus observed; TVD is total variation distance between band distributions; MAE averages the absolute error of the share with debt across the groups. `educ` is the SCF reference person's education, which the CPS side cannot supply until microcosm#297, so that row shows what the predictor would add.

| Variant | share_with_debt_error_pp | mean_among_borrowers_error_pct | median_among_borrowers_error_pct | household_balance_band_tvd | head_age_band_share_mae_pp | race_share_mae_pp | income_quintile_share_mae_pp | borrower_balance_band_tvd | borrower_age_band_tvd | household_rank_correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_8_without_age | -0.250 | -1.737 | -4.000 | 0.022 | 4.812 | 0.648 | 0.672 | 0.028 | 0.107 | 0.154 |
| shared_8 | -0.532 | -4.092 | -8.000 | 0.041 | 1.193 | 1.078 | 0.876 | 0.037 | 0.016 | 0.173 |
| shared_8_plus_household_income (build) | -0.873 | -4.350 | -8.000 | 0.037 | 1.351 | 1.359 | 0.923 | 0.030 | 0.009 | 0.175 |
| plus_household_income_and_education | -1.474 | -5.567 | -4.816 | 0.040 | 1.321 | 2.067 | 1.474 | 0.050 | 0.023 | 0.208 |

## Host fit

Imputed with build seed 0 onto `populace-us-2024-spm-20260915` (57,240 households, sha256 `6496cc4393d4...`) at its calibrated household weights; the published weights were calibrated without these columns. Signal gate: passed.

| Metric (imputed minus SCF 2025) | Value |
|---|---:|
| share_with_debt_error_pp | -3.742 |
| mean_among_borrowers_error_pct | 8.357 |
| median_among_borrowers_error_pct | 0.000 |
| household_balance_band_tvd | 0.062 |
| head_age_band_share_mae_pp | 3.554 |
| race_share_mae_pp | 2.834 |
| income_quintile_share_mae_pp | 4.252 |
| borrower_balance_band_tvd | 0.070 |
| borrower_age_band_tvd | 0.085 |
