## Donor totals against SOI actuals

| SOI line | Donor weighted total, $B | SOI TY2021, $B | SOI TY2023, $B | Donor / TY2021 | Donor / TY2023 |
|---|---:|---:|---:|---:|---:|
| Wages | 9,078.3 | 9,022.4 | 10,204.1 | 1.006 | 0.890 |
| Taxable interest | 105.7 | 103.5 | 313.8 | 1.021 | 0.337 |
| Tax-exempt interest | 55.3 | 55.5 | 66.1 | 0.996 | 0.837 |
| Qualified dividends | 297.1 | 295.9 | 336.1 | 1.004 | 0.884 |
| Capital gain distributions | 24.3 | 23.9 | 9.3 | 1.018 | 2.603 |
| Taxable IRA distributions | 406.1 | 408.4 | 438.1 | 0.994 | 0.927 |
| Taxable pensions | 861.8 | 858.0 | 932.1 | 1.004 | 0.925 |
| Social Security benefits | 790.7 | 791.2 | 973.6 | 0.999 | 0.812 |
| Returns (donor weight total), M | 161.18 | 160.82 | 160.60 | 1.002 | 1.004 |

## Donor columns by factor

| Factor carried, TY2015 to TY2021 | Microcosm donor column | Source fields | Records with a nonzero source value | Share matching the archived 2015-to-2021 value | Share matching 2015, 2022 or 2023 | Modal ratio to the source value |
|---|---|---|---:|---:|---:|---:|
| SOI per-return growth of its own line | `employment_income_before_lsr` | E00200 | 165,548 | 1.0000 | 0.0000 | 1.1871 |
| SOI per-return growth of its own line | `taxable_interest_income` | E00300 | 119,548 | 1.0000 | 0.0000 | 1.0105 |
| SOI per-return growth of its own line | `tax_exempt_interest_income` | E00400 | 46,071 | 1.0000 | 0.0000 | 0.8397 |
| SOI per-return growth of its own line | `qualified_dividend_income` | E00650 | 93,111 | 1.0000 | 0.0000 | 1.3628 |
| SOI per-return growth of its own line | `non_qualified_dividend_income` | E00600 less E00650, each by its own line | 72,822 | 1.0000 | 0.0000 |  |
| SOI per-return growth of its own line | `non_sch_d_capital_gains` | E01100 | 6,219 | 1.0000 | 0.0000 | 1.9333 |
| SOI per-return growth of its own line | `taxable_ira_distributions` | E01400 | 26,825 | 1.0000 | 0.0000 | 1.5092 |
| SOI per-return growth of its own line | `taxable_private_pension_income` | E01700 | 42,428 | 1.0000 | 0.0000 | 1.1637 |
| SOI per-return growth of its own line | `social_security_retirement` | E02400 | 42,664 | 1.0000 | 0.0000 | 1.2234 |
| Flat 2% a year for itemized deductions | `charitable_cash_donations` | E19800 | 102,134 | 1.0000 | 0.0000 | 1.1262 |
| Flat 2% a year for itemized deductions | `real_estate_taxes` | E18500 | 106,533 | 1.0000 | 0.0000 | 1.1262 |
| Flat 2% a year for itemized deductions | `home_mortgage_interest` + `investment_interest_expense` | E19200, split by Microcosm | 91,148 | 0.7612 | 0.0000 | 1.1262 |
| AGI per-return growth (the archived default) | `alimony_expense` | E03500 | 607 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `alimony_income` | E00800 | 314 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `casualty_loss` | E20500 | 105 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `charitable_non_cash_donations` | E20100 | 57,586 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `domestic_production_ald` | E03240 | 13,424 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `educator_expense` | E03220 | 3,979 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `estate_income` | E26390 less E26400 | 7,434 | 1.0000 | 0.0000 |  |
| AGI per-return growth (the archived default) | `farm_income` | T27800 | 1,158 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `farm_rent_income` | E27200 | 1,439 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `health_savings_account_ald` | E03290 | 7,109 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `investment_income_elected_form_4952` | E58990 | 3,553 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `long_term_capital_gains_before_response` | P23250 | 83,332 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `long_term_capital_gains_on_collectibles` | E24518 | 1,297 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `miscellaneous_income` | E01200 | 30,921 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `qualified_tuition_expenses` | E03230 | 1,396 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `rental_income` | E25850 less E25860 | 49,409 | 1.0000 | 0.0000 |  |
| AGI per-return growth (the archived default) | `salt_refund_income` | E00700 | 53,710 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `self_employed_pension_contributions_desired` | E03300 | 12,277 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `short_term_capital_gains` | P22250 | 65,942 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `student_loan_interest` | E03210 | 9,341 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `traditional_ira_contributions_desired` | E03150 | 5,217 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `unrecaptured_section_1250_gain` | E24515 | 10,607 | 1.0000 | 0.0000 | 1.3560 |
| AGI per-return growth (the archived default) | `unreimbursed_business_employee_expenses` | E20400 | 80,099 | 1.0000 | 0.0000 | 1.3560 |
| Sign-split business growth, then AGI growth again | `self_employment_income_before_lsr` + `sstb_self_employment_income_before_lsr` | E00900 | 50,824 | 1.0000 | 0.0000 | 1.6739 |
| None: TY2015 level | `partnership_income` | E26190 less E26180 plus E25980 less E25960 | 60,763 | 1.0000 | 1.0000 |  |
| None: TY2015 level | `farm_operations_income` | E02100 | 8,318 | 1.0000 | 1.0000 | 1.0000 |

For the two TY2015 columns the hypotheses coincide: no archived list names their fields, so every hypothesis equals the raw value.

## The leg the 2024 branch never applied

| Archived series | 2015 to 2021, applied | 2021 to 2022 | 2021 to 2023, not applied |
|---|---:|---:|---:|
| E00200 (employment_income) | 1.1871 | 1.0760 | 1.1325 |
| E00300 (taxable_interest_income) | 1.0105 | 1.2863 | 3.0352 |
| E00400 (exempt_interest) | 0.8397 | 0.9977 | 1.1921 |
| E00600 (ordinary_dividends) | 1.3914 | 1.0622 | 1.3048 |
| E00650 (qualified_dividends) | 1.3628 | 1.0552 | 1.1373 |
| E01100 (capital_gains_distributions) | 1.9333 | 0.5367 | 0.3915 |
| E01000 pos_only (capital_gains_gross) | 2.6866 | 0.6178 | 0.4722 |
| E01400 (ira_distributions) | 1.5092 | 1.0686 | 1.0744 |
| E01700 (taxable_pension_income) | 1.1637 | 1.0592 | 1.0879 |
| E02400 (total_social_security) | 1.2234 | 1.0978 | 1.2324 |
| E02300 (unemployment_compensation) | 7.1791 | 0.1444 | 0.1483 |
| E00900 pos_only (business_net_profits) | 1.2344 | 1.0479 | 1.0271 |
| E00900 neg_only (business_net_losses) | 1.6422 | 1.2573 | 1.4553 |
| E26270 pos_only (partnership_and_s_corp_income) | 1.5313 | 1.0479 | 1.0119 |
| REMAINING_VARIABLES (adjusted_gross_income) | 1.3560 | 0.9994 | 1.0346 |
| S006 weights (return count) | 1.0686 | 1.0032 | 0.9986 |

## From donor to PUF arm

| Donor column | Donor at its own weights, $B | PUF arm ×2 at design weights, $B | PUF arm / donor | ASEC arm ×2 at design weights, $B | Own series at TY2023 / level the donor carries |
|---|---:|---:|---:|---:|---:|
| Wages | 9,078.3 | 9,860.4 | 1.09 | 11,562.0 | 1.133 |
| Taxable interest | 105.7 | 158.7 | 1.50 | 436.2 | 3.035 |
| Tax-exempt interest | 55.3 | 108.9 | 1.97 | 0.0 | 1.192 |
| Qualified dividends | 297.1 | 363.9 | 1.22 | 97.0 | 1.137 |
| Non-qualified dividends (h) | 92.3 | 137.6 | 1.49 | 119.6 |  |
| Short-term gains (d) | 55.0 | 106.9 | 1.94 | 28.9 | 0.936 |
| Long-term gains (d) | 1,112.6 | 2,225.8 | 2.00 | 212.2 | 0.936 |
| Capital gain distributions (e) | 24.3 | 20.3 | 0.84 | 0.0 | 0.392 |
| Taxable pensions | 861.8 | 1,574.1 | 1.83 | 335.6 | 1.088 |
| Taxable IRA distributions | 406.1 | 730.7 | 1.80 | 110.8 | 1.074 |
| Social Security, retirement component (a) | 790.7 | 1,024.8 | 1.30 | 1,198.6 | 1.232 |
| Schedule C profit, with its SSTB share (b) | 699.1 | 1,024.3 | 1.47 | 550.0 | 0.757 |
| Partnership income, positive (c) | 766.2 | 821.5 | 1.07 | 0.0 | 1.550 |
| Rental income, positive | 142.5 | 277.4 | 1.95 | 245.4 | 0.929 |
| Estate income, positive | 48.0 | 150.7 | 3.14 | 0.0 | 1.020 |
| Cash charitable gifts (f) | 196.8 | 269.7 | 1.37 | 0.0 | 1.040 |
| Real estate taxes (f) | 227.3 | 312.3 | 1.37 | 0.0 | 1.040 |
| Mortgage interest (f) | 378.4 | 476.2 | 1.26 | 0.0 | 1.040 |
| Farm operations income, positive (i) | 15.1 | 17.2 | 1.14 | 41.9 |  |
| Partnership self-employment earnings, positive (i) | 106.9 | 122.0 | 1.14 | 0.0 |  |
| Tax units, M | 161.2 | 184.1 | 1.14 | 184.1 | |

## Release, national rows

| National row, aged to 2024 | Source | Target, $B | PUF arm ×2 / target, design weights | ASEC arm ×2 / target, design weights | Estimate / target, design weights | Estimate / target, final weights | PUF-arm part, final / design | ASEC-arm part, final / design |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Wages | TY2023 Table 1.4 | 10,773.4 | 0.915 | 1.073 | 0.994 | 0.996 | 1.255 | 0.786 |
| Taxable interest | TY2023 Table 4.3, excluding dependents | 340.4 | 0.466 | 1.281 | 0.874 | 0.844 | 1.650 | 0.717 |
| Tax-exempt interest | TY2022 Historic Table 2 | 61.7 | 1.765 | 0.000 | 0.882 | 1.004 | 1.138 | n/a |
| Ordinary dividends | TY2023 Table 4.3, excluding dependents | 544.8 | 0.921 | 0.398 | 0.659 | 0.839 | 1.523 | 0.694 |
| Ordinary dividends | TY2022 Historic Table 2 | 457.4 | 1.096 | 0.474 | 0.785 | 0.999 | 1.523 | 0.694 |
| Qualified dividends | TY2022 Historic Table 2 | 346.6 | 1.050 | 0.280 | 0.665 | 0.974 | 1.670 | 0.694 |
| Taxable pensions | TY2023 Table 1.4 | 1,013.4 | 1.553 | 0.331 | 0.942 | 0.989 | 1.105 | 0.791 |
| Taxable IRA distributions | TY2023 Table 1.4 | 476.4 | 1.534 | 0.233 | 0.883 | 1.019 | 1.180 | 0.985 |
| Taxable Social Security (a) | TY2023 Table 1.4 | 573.0 | 1.591 | 0.694 | 1.143 | 0.892 | 0.870 | 0.576 |
| Taxable Social Security (a) | TY2022 Historic Table 2 | 510.8 | 1.785 | 0.779 | 1.282 | 1.000 | 0.870 | 0.576 |
| Schedule C net profit | TY2022 Historic Table 2 | 472.3 | 1.490 | 1.164 | 1.327 | 1.035 | 0.878 | 0.655 |
| Partnership and S-corp net income | TY2022 Historic Table 2 | 1,157.6 | 0.710 | 0.000 | 0.355 | 0.975 | 2.747 | n/a |
| Net capital gains | TY2023 Table 1.4 | 1,270.9 | 1.756 | 0.190 | 0.973 | 0.980 | 1.017 | 0.923 |
| Capital gain distributions (e) | TY2023 Table 1.4 | 10.2 | 2.000 | 0.000 | 1.000 | 1.000 | 1.000 | n/a |
| Estate and trust net income | TY2023 Table 1.4 | 52.1 | 2.895 | 0.000 | 1.447 | 1.001 | 0.691 | n/a |
| Rent and royalty net income (g) | TY2022 Historic Table 2 | 95.9 | 2.049 | 2.521 | 2.285 | 2.640 | 1.174 | 1.140 |
| Real estate taxes | TY2023 Table 2.1, itemizers | 118.1 | 1.109 | 0.000 | 0.555 | 1.007 | 1.817 | n/a |
| Mortgage interest | TY2023 Table 2.1, itemizers | 186.3 | 1.204 | 0.000 | 0.602 | 1.034 | 1.719 | n/a |
| Charitable gifts | TY2023 Table 2.1, itemizers | 230.5 | 0.966 | 0.000 | 0.483 | 1.137 | 2.354 | n/a |
| AGI | TY2023 Table 1.1 | 16,619.2 | 1.118 | 0.832 | 0.975 | 1.004 | 1.219 | 0.777 |

## A rebase alone, at design weights

| National row | Estimate / target today | PUF arm ×2 / target × own-series factor | Estimate / target after |
|---|---:|---:|---:|
| Wages | 0.994 | 0.915 × 1.133 | 1.055 |
| Taxable interest | 0.874 | 0.466 × 3.035 | 1.348 |
| Tax-exempt interest | 0.882 | 1.765 × 1.192 | 1.052 |
| Qualified dividends | 0.665 | 1.050 × 1.137 | 0.737 |
| Partnership and S-corp net income | 0.355 | 0.710 × 1.550 | 0.550 |

## Release, weights by arm

| Arm | Households | Weight, design (M) | Weight, final (M) | Final / design | Household ratio: 5th, 50th, 95th percentile | Share at the 5x cap |
|---|---:|---:|---:|---:|---|---:|
| ASEC | 168,852 | 69.05 | 63.37 | 0.918 | 0.09, 0.34, 5.00 | 6.5% |
| PUF clone | 168,852 | 68.99 | 74.69 | 1.083 | 0.09, 0.39, 5.00 | 8.8% |
| Capital-gains tail clone | 15,228 | 0.06 | 0.04 | 0.693 | 0.25, 0.55, 1.97 | 1.0% |
