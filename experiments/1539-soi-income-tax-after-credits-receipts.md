# SOI income tax after credits on the published release (microcosm#1539)

This run scores the published Microcosm US release whose H5 SHA-256 begins `6496cc4393d4` (calibration diagnostics `870449b44e86`) with policyengine-us 2.2.1 and policyengine-core 3.32.5, the versions that built the release, over 57,240 households and 79,729 tax units for 2024. **Before** is the old row definition, `max(income_tax, 0)` per tax unit (`income_tax > 0` for return rows); **after** is Form 1040 line 22, `max(income_tax_before_credits - income_tax_capped_non_refundable_credits, 0)`. **Initial** and **final** are the release's calibration weights before and after fitting. All figures are weighted sums; no record leaves the run.

## Reconstruction check

On the published H5, the old definition comes within 0.55% of every initial estimate and 2.30% of every final estimate the release's diagnostics record (national amount +0.21%, national returns +0.06% at final weights). Both definitions below come from the same H5 run, so their difference compares like with like.

## National rows

| Row | SOI target | Weights | Before | Error | After | Error | After / before |
|---|---:|---|---:|---:|---:|---:|---:|
| A06500 amount ($B) | 2,358.72 | initial | 3,373.63 | +43.03% | 3,318.16 | +40.68% | -1.64% |
| A06500 amount ($B) | 2,358.72 | final | 2,366.41 | +0.33% | 2,313.85 | -1.90% | -2.22% |
| N06500 returns (M) | 113.56 | initial | 121.47 | +6.96% | 123.58 | +8.82% | +1.74% |
| N06500 returns (M) | 113.56 | final | 113.27 | -0.26% | 116.02 | +2.17% | +2.43% |

## Engine totals at final weights ($B)

| Variable | Total |
|---|---:|
| `income_tax` | 2,262.53 |
| `income_tax_before_credits` | 2,410.28 |
| `income_tax_before_refundable_credits` | 2,367.40 |
| `income_tax_capped_non_refundable_credits` | 96.43 |
| `income_tax_non_refundable_credits` | 117.18 |
| `income_tax_refundable_credits` | 104.87 |
| `net_investment_income_tax` | 53.55 |

Line 22 is `income_tax_before_credits` less `income_tax_capped_non_refundable_credits`; `income_tax_before_refundable_credits` adds the net investment income tax and three other Schedule 2 Part II taxes (the recapture of investment credit, unreported payroll tax and the qualified retirement penalty) to it, and `income_tax` subtracts the refundable credits.

## States at final weights

| State | Amount target ($B) | Before | After | Returns target (M) | Before | After |
|---|---:|---:|---:|---:|---:|---:|
| AK | 4.31 | +1.09% | -0.40% | 0.277 | +0.08% | +1.01% |
| AL | 20.94 | +0.21% | -1.77% | 1.429 | +0.46% | +4.16% |
| AR | 12.54 | +1.64% | -0.62% | 0.850 | +0.05% | +3.45% |
| AZ | 42.17 | -0.09% | -2.29% | 2.398 | -0.03% | +3.10% |
| CA | 350.43 | +0.19% | -2.38% | 13.528 | -0.02% | +2.06% |
| CO | 47.70 | +0.18% | -2.18% | 2.261 | +0.06% | +1.75% |
| CT | 40.92 | +0.32% | -2.43% | 1.370 | +0.08% | +2.67% |
| DC | 9.45 | +0.10% | -2.16% | 0.273 | +0.09% | +2.64% |
| DE | 6.06 | +1.58% | +0.23% | 0.368 | -0.13% | +2.71% |
| FL | 197.13 | +1.90% | -2.02% | 7.710 | -0.00% | +3.12% |
| GA | 61.13 | +0.75% | -1.38% | 3.286 | -0.07% | +2.62% |
| HI | 8.05 | +0.18% | -2.31% | 0.502 | +0.40% | +2.27% |
| IA | 16.19 | +0.11% | -1.50% | 1.103 | +0.04% | +2.55% |
| ID | 9.83 | +2.85% | +1.67% | 0.620 | -0.02% | +3.74% |
| IL | 94.66 | +0.11% | -1.41% | 4.459 | +0.02% | +1.78% |
| IN | 33.13 | +3.29% | +2.09% | 2.259 | -0.09% | +2.65% |
| KS | 15.76 | +2.03% | +0.24% | 0.975 | +0.03% | +2.05% |
| KY | 17.73 | +1.85% | +1.27% | 1.346 | -1.01% | +1.51% |
| LA | 19.57 | +0.11% | -2.02% | 1.249 | +0.08% | +3.43% |
| MA | 77.04 | +0.21% | -1.97% | 2.735 | +0.01% | +1.77% |
| MD | 46.36 | +2.79% | +1.37% | 2.268 | +0.02% | +4.17% |
| ME | 7.24 | +0.07% | -1.71% | 0.521 | +0.02% | +1.97% |
| MI | 53.07 | +0.93% | -0.42% | 3.365 | +0.75% | +3.55% |
| MN | 39.37 | +0.17% | -0.94% | 2.169 | -0.12% | +2.73% |
| MO | 31.39 | +2.65% | +0.70% | 2.023 | +0.02% | +0.50% |
| MS | 9.62 | +0.29% | -1.65% | 0.770 | +0.11% | +3.32% |
| MT | 6.85 | +0.30% | -2.37% | 0.395 | +0.59% | +1.08% |
| NC | 60.21 | +1.36% | -0.30% | 3.475 | -0.07% | +2.46% |
| ND | 4.91 | +2.02% | +1.00% | 0.283 | +0.11% | +0.87% |
| NE | 11.17 | +3.34% | +1.70% | 0.690 | +0.20% | +1.09% |
| NH | 12.36 | +0.10% | -1.57% | 0.580 | -0.21% | +1.14% |
| NJ | 88.58 | -0.04% | -1.73% | 3.434 | -0.03% | +2.16% |
| NM | 8.19 | +1.30% | -0.19% | 0.645 | -0.01% | +1.87% |
| NV | 23.61 | +0.09% | -3.08% | 1.107 | +0.10% | +2.15% |
| NY | 188.90 | +0.16% | -2.16% | 6.942 | +0.01% | +2.28% |
| OH | 60.74 | +0.85% | -0.43% | 4.130 | +0.04% | +2.31% |
| OK | 16.54 | +0.00% | -2.53% | 1.140 | +0.06% | +4.99% |
| OR | 24.79 | +0.48% | -1.34% | 1.514 | -0.02% | +2.50% |
| PA | 82.93 | +0.24% | -1.22% | 4.552 | +0.35% | +2.00% |
| RI | 7.12 | -0.03% | -1.61% | 0.418 | -0.06% | +1.94% |
| SC | 26.70 | +0.94% | -0.35% | 1.706 | -0.10% | +3.51% |
| SD | 5.55 | +2.16% | +0.52% | 0.323 | -0.01% | +4.06% |
| TN | 39.33 | +0.30% | -2.27% | 2.274 | -0.03% | +1.41% |
| TX | 203.38 | +1.77% | -1.06% | 9.180 | +0.24% | +3.49% |
| UT | 20.19 | +3.06% | +0.09% | 1.108 | +0.21% | +1.23% |
| VA | 64.87 | +0.67% | -1.18% | 3.078 | -0.06% | +2.85% |
| VT | 3.75 | +4.98% | +3.38% | 0.247 | +0.04% | +1.99% |
| WA | 68.88 | -0.03% | -2.25% | 2.937 | -0.09% | +1.42% |
| WI | 34.67 | +2.95% | +1.72% | 2.200 | +0.47% | +2.12% |
| WV | 6.08 | -0.01% | -1.25% | 0.530 | +0.02% | +2.40% |
| WY | 4.86 | +1.03% | -3.28% | 0.207 | +0.07% | +1.15% |
