# #1228 re-solve summary

| Arm | Own loss | Own rows within 10% | Own 312 rows within 10% | Common rows within 10% | Common mean capped error | Kish ESS | Corr. with published |
|---|---:|---:|---:|---:|---:|---:|---:|
| published |  |  |  | 0.9392 | 0.03067 | 77,758 | 1.0000 |
| release | 0.028420 | 0.9341 | 0.840 | 0.9396 | 0.03064 | 77,794 | 0.9999 |
| concept_exact | 0.026406 | 0.9389 | 0.926 | 0.9396 | 0.03074 | 77,306 | 0.9174 |
| concept_exact_no_business_count | 0.025833 | 0.9404 | 0.931 | 0.9409 | 0.03042 | 82,769 | 0.8495 |
| drop | 0.024141 | 0.9433 | 0.846 | 0.9443 | 0.02848 | 84,834 | 0.8531 |
| rebase_positive_part | 0.025521 | 0.9406 | 0.929 | 0.9413 | 0.03011 | 81,036 | 0.9057 |

## Model totals (2024 weights; $B and millions)

| Arm | Concept | Positive $B | Negative $B | Net $B | Positive M | Negative M | Nonzero M |
|---|---|---:|---:|---:|---:|---:|---:|
| published | schedule_c_total | 836.9 | -15.9 | 821.0 | 30.64 | 0.65 | 31.30 |
| published | schedule_c_nonsstb | 488.9 | -4.3 | 484.6 | 26.30 | 0.31 | 26.61 |
| published | sstb_only | 348.0 | -11.6 | 336.4 | 4.42 | 0.37 | 4.79 |
| published | partnership_s_corp | 1,128.5 | -108.0 | 1,020.5 | 9.48 | 2.01 | 11.49 |
| published | rental_only | 335.2 | -86.4 | 248.8 | 9.07 | 5.02 | 14.09 |
| published | rental_plus_farm_rent | 345.5 | -92.1 | 253.3 | 9.32 | 5.18 | 14.50 |
| published | farm_rent_only | 13.2 | -8.7 | 4.5 | 0.44 | 0.29 | 0.73 |
| release | schedule_c_total | 836.9 | -15.9 | 821.0 | 30.64 | 0.65 | 31.29 |
| release | schedule_c_nonsstb | 489.0 | -4.3 | 484.7 | 26.30 | 0.31 | 26.61 |
| release | sstb_only | 347.9 | -11.6 | 336.3 | 4.42 | 0.37 | 4.79 |
| release | partnership_s_corp | 1,129.0 | -108.0 | 1,021.0 | 9.48 | 2.01 | 11.49 |
| release | rental_only | 335.0 | -86.4 | 248.6 | 9.07 | 5.02 | 14.09 |
| release | rental_plus_farm_rent | 345.3 | -92.2 | 253.1 | 9.32 | 5.18 | 14.50 |
| release | farm_rent_only | 13.2 | -8.7 | 4.5 | 0.44 | 0.29 | 0.73 |
| concept_exact | schedule_c_total | 532.2 | -25.3 | 506.9 | 28.08 | 1.29 | 29.37 |
| concept_exact | schedule_c_nonsstb | 436.5 | -9.4 | 427.2 | 21.10 | 0.86 | 21.95 |
| concept_exact | sstb_only | 95.6 | -15.9 | 79.8 | 7.04 | 0.47 | 7.51 |
| concept_exact | partnership_s_corp | 1,248.7 | -128.9 | 1,119.8 | 7.26 | 2.26 | 9.52 |
| concept_exact | rental_only | 182.7 | -84.3 | 98.4 | 5.75 | 3.57 | 9.32 |
| concept_exact | rental_plus_farm_rent | 199.5 | -88.2 | 111.3 | 6.15 | 3.70 | 9.85 |
| concept_exact | farm_rent_only | 20.1 | -7.2 | 12.9 | 0.55 | 0.25 | 0.80 |
| concept_exact_no_business_count | schedule_c_total | 513.0 | -19.5 | 493.5 | 13.65 | 0.75 | 14.40 |
| concept_exact_no_business_count | schedule_c_nonsstb | 422.5 | -5.8 | 416.8 | 10.45 | 0.39 | 10.84 |
| concept_exact_no_business_count | sstb_only | 90.5 | -13.8 | 76.7 | 3.24 | 0.39 | 3.63 |
| concept_exact_no_business_count | partnership_s_corp | 1,252.5 | -132.3 | 1,120.2 | 7.25 | 2.28 | 9.52 |
| concept_exact_no_business_count | rental_only | 185.4 | -86.1 | 99.3 | 5.75 | 3.57 | 9.32 |
| concept_exact_no_business_count | rental_plus_farm_rent | 202.6 | -89.5 | 113.1 | 6.14 | 3.70 | 9.84 |
| concept_exact_no_business_count | farm_rent_only | 20.5 | -6.7 | 13.8 | 0.54 | 0.26 | 0.80 |
| drop | schedule_c_total | 1,332.9 | -12.1 | 1,320.7 | 19.03 | 0.52 | 19.55 |
| drop | schedule_c_nonsstb | 1,013.0 | -3.8 | 1,009.2 | 14.71 | 0.31 | 15.02 |
| drop | sstb_only | 319.8 | -8.3 | 311.5 | 4.36 | 0.22 | 4.58 |
| drop | partnership_s_corp | 986.9 | -175.8 | 811.1 | 5.13 | 2.32 | 7.45 |
| drop | rental_only | 208.8 | -113.8 | 95.1 | 8.71 | 5.54 | 14.25 |
| drop | rental_plus_farm_rent | 216.1 | -119.8 | 96.2 | 8.95 | 5.71 | 14.67 |
| drop | farm_rent_only | 10.1 | -8.9 | 1.2 | 0.42 | 0.30 | 0.72 |
| rebase_positive_part | schedule_c_total | 627.7 | -14.4 | 613.3 | 21.89 | 0.60 | 22.49 |
| rebase_positive_part | schedule_c_nonsstb | 524.4 | -3.5 | 521.0 | 16.77 | 0.31 | 17.07 |
| rebase_positive_part | sstb_only | 103.3 | -11.0 | 92.3 | 5.17 | 0.31 | 5.49 |
| rebase_positive_part | partnership_s_corp | 1,247.9 | -128.5 | 1,119.4 | 7.27 | 2.25 | 9.52 |
| rebase_positive_part | rental_only | 186.5 | -68.3 | 118.2 | 6.19 | 4.48 | 10.67 |
| rebase_positive_part | rental_plus_farm_rent | 192.9 | -73.1 | 119.8 | 6.35 | 4.66 | 11.01 |
| rebase_positive_part | farm_rent_only | 8.8 | -7.3 | 1.6 | 0.31 | 0.27 | 0.57 |

## The 312 rows: US value and state fit on each concept

| Arm | Measure | US target | US, release concept | US, #1228 concept | States within 10%, release concept | States within 10%, #1228 concept |
|---|---|---:|---:|---:|---:|---:|
| published | schedule_c_income_returns | 30,354,680 | 26,300,787 | 31,295,990 | 0.76 | 0.20 |
| published | schedule_c_income_amount | 472,289,642,162 | 488,894,776,129 | 820,989,668,708 | 0.94 | 0.00 |
| published | partnership_scorp_income_returns | 9,551,770 | 9,479,193 | 11,489,274 | 0.98 | 0.08 |
| published | partnership_scorp_income_amount | 1,157,602,275,777 | 1,128,517,678,096 | 1,020,542,634,029 | 0.90 | 0.57 |
| published | rental_royalty_income_returns | 9,398,010 | 9,321,529 | 14,090,443 | 1.00 | 0.00 |
| published | rental_royalty_income_amount | 95,949,567,370 | 253,318,942,421 | 248,787,751,423 | 0.47 | 0.33 |
| release | schedule_c_income_returns | 30,354,680 | 26,295,799 | 31,289,969 | 0.76 | 0.20 |
| release | schedule_c_income_amount | 472,289,642,162 | 488,979,977,389 | 820,954,710,358 | 0.94 | 0.00 |
| release | partnership_scorp_income_returns | 9,551,770 | 9,476,976 | 11,485,126 | 0.98 | 0.08 |
| release | partnership_scorp_income_amount | 1,157,602,275,777 | 1,128,981,889,247 | 1,021,022,818,601 | 0.90 | 0.57 |
| release | rental_royalty_income_returns | 9,398,010 | 9,318,330 | 14,087,963 | 1.00 | 0.00 |
| release | rental_royalty_income_amount | 95,949,567,370 | 253,129,223,409 | 248,630,372,193 | 0.47 | 0.33 |
| concept_exact | schedule_c_income_returns | 30,354,680 | 21,097,308 | 29,368,664 | 0.00 | 0.94 |
| concept_exact | schedule_c_income_amount | 472,289,642,162 | 436,541,144,439 | 506,922,149,127 | 0.22 | 0.92 |
| concept_exact | partnership_scorp_income_returns | 9,551,770 | 7,262,592 | 9,522,293 | 0.10 | 1.00 |
| concept_exact | partnership_scorp_income_amount | 1,157,602,275,777 | 1,248,728,293,658 | 1,119,829,818,708 | 0.51 | 0.86 |
| concept_exact | rental_royalty_income_returns | 9,398,010 | 6,146,225 | 9,320,362 | 0.00 | 1.00 |
| concept_exact | rental_royalty_income_amount | 95,949,567,370 | 111,266,691,574 | 98,375,781,509 | 0.33 | 0.82 |
| concept_exact_no_business_count | schedule_c_income_returns | 30,354,680 | 10,446,932 | 14,401,264 | 0.00 | 0.00 |
| concept_exact_no_business_count | schedule_c_income_amount | 472,289,642,162 | 422,542,007,445 | 493,450,521,143 | 0.27 | 0.94 |
| concept_exact_no_business_count | partnership_scorp_income_returns | 9,551,770 | 7,247,623 | 9,523,958 | 0.12 | 1.00 |
| concept_exact_no_business_count | partnership_scorp_income_amount | 1,157,602,275,777 | 1,252,526,622,019 | 1,120,216,282,646 | 0.47 | 0.86 |
| concept_exact_no_business_count | rental_royalty_income_returns | 9,398,010 | 6,140,639 | 9,323,595 | 0.00 | 1.00 |
| concept_exact_no_business_count | rental_royalty_income_amount | 95,949,567,370 | 113,142,552,295 | 99,319,784,559 | 0.31 | 0.84 |
| drop | schedule_c_income_returns | 30,354,680 | 14,705,483 | 19,545,854 | 0.00 | 0.04 |
| drop | schedule_c_income_amount | 472,289,642,162 | 1,013,049,234,570 | 1,320,738,618,653 | 0.00 | 0.00 |
| drop | partnership_scorp_income_returns | 9,551,770 | 5,125,505 | 7,448,438 | 0.00 | 0.18 |
| drop | partnership_scorp_income_amount | 1,157,602,275,777 | 986,909,097,088 | 811,085,770,783 | 0.06 | 0.10 |
| drop | rental_royalty_income_returns | 9,398,010 | 8,952,775 | 14,248,242 | 0.51 | 0.08 |
| drop | rental_royalty_income_amount | 95,949,567,370 | 96,244,798,434 | 95,051,311,739 | 0.84 | 0.49 |
| rebase_positive_part | schedule_c_income_returns | 30,354,680 | 16,765,414 | 22,493,042 | 0.00 | 0.00 |
| rebase_positive_part | schedule_c_income_amount | 472,289,642,162 | 524,427,962,673 | 613,289,949,111 | 0.49 | 0.00 |
| rebase_positive_part | partnership_scorp_income_returns | 9,551,770 | 7,268,319 | 9,523,299 | 0.12 | 1.00 |
| rebase_positive_part | partnership_scorp_income_amount | 1,157,602,275,777 | 1,247,921,767,077 | 1,119,446,019,406 | 0.51 | 0.86 |
| rebase_positive_part | rental_royalty_income_returns | 9,398,010 | 6,349,815 | 10,669,611 | 0.00 | 0.43 |
| rebase_positive_part | rental_royalty_income_amount | 95,949,567,370 | 119,793,997,240 | 118,216,927,758 | 0.14 | 0.16 |
