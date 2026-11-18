---
kind: proposal
client: "Living Wage Institute"
status: "draft v0.1 (2026-11-18): for María and Max to review, not for sending"
pe-doc:
  psl_foundation: false
  split_tables: true
  pdf_name: "PolicyEngine - County Files proof of concept report"
---

# County Files proof of concept: the 2024 county file

**What we built, how well it matches official county statistics, and what comes next**

A PolicyEngine report · for the Living Wage Institute · November 2026

## Summary

- **We are delivering a 2024 county file.** It is one microdata file of synthetic households that covers all 3,144 US counties and county-equivalents. It is weighted to represent about 122.5 million households and 331 million people, and it runs in PolicyEngine's US tax-benefit model (policyengine-us 2.2.1).
- **State results are strong.** After calibration, 96% of the 4,227 IRS state tax targets are within 10% of their official value, and every SNAP, Medicaid and state population target is within 10%.
- **County totals are useful, but county distributions are not yet reliable.** Two-thirds of the county targets used in calibration are within 10%. Against county statistics the calibration never saw, the file estimates total earnings and total household income better than a simple population-share split. It is worse than that split for the number of households with earnings and for the income distribution.
- **The main limit is the effective sample in each county.** In the median county, the weights behave like a sample of about 6 households. National donor households make up 3.6% of the records but carry 77% of the weight, so a few of them dominate each county.
- **Use the file for county totals and for ranking counties,** with care. Do not yet use it for distributions inside a county, such as income brackets, poverty rates or the share of households with earnings.
- **Next, we will raise the effective sample in each county,** balance how much cloned records contribute to the calibration, and improve the calibration fit. We will also produce the 2026 and 2027 files, the household measures and the schema manifest.

## What we are delivering

| Item | In the agreement | Status for Monday |
|---|---|---|
| The 2024 county file | Yes | Built on 18 November 2026. All automated checks passed. Shared in a private Hugging Face repository, `policyengine/lwi-county-files`, with a read-only access token. |
| A written delivery notice | Yes | Max Ghenis sends it. |
| This report: the fit and the limits | For the proof of concept | This document. |
| Methods documentation | Yes | A draft is ready: [[link to the methods draft]] |
| Schema documentation | Yes | [[Schema documentation for Monday, or moved to the stretch goals]] |
| The 2026 and 2027 county files | Stretch goal | Not yet produced. |
| Household measures (earnings, gross income and net income for each household) | Stretch goal | Not yet produced. |
| The schema manifest (every table and column, with types and counts) | Stretch goal | Not yet produced. |

## The 2024 county file

| Feature | Value |
|---|---|
| Year | 2024 |
| Geography | 3,144 counties and county-equivalents on the 2023 county codes, including Connecticut's nine planning regions; 436 congressional districts. Puerto Rico is not included. |
| Household records | 1,586,908: 1,529,668 from the 2024 American Community Survey (ACS) and 57,240 national donor households |
| Person records | 3,587,263: 3,420,942 ACS and 166,321 donor |
| Weighted totals | About 122.5 million households and 331 million people |
| Model | policyengine-us 2.2.1 |
| File | `populace_us_2024_acs_local.h5`, 10.6 GB (HDF5) |

Every county has at least one household record. The median county has 144 records, and the largest has 35,707. Ten very small counties have only one or two records. Three counties had no records of their own, so we moved three households into each of them from other counties in the same PUMA (Public Use Microdata Area, the smallest area the ACS microdata identifies).

## How we built it

The method has four steps. The methods documentation gives the detail ([[link to the methods draft]]), and our working papers explain the main choices.

1. **Households and their counties.** We start from the 2024 ACS 1-year public-use microdata. The ACS gives each household's PUMA, not its county. So we assign each ACS household a county within its PUMA, in proportion to the population each county shares with that PUMA.
2. **National donor households.** The ACS does not collect detailed tax items. We add 57,240 national donor households from the Current Population Survey (CPS ASEC) with tax detail from the IRS public-use file, from PolicyEngine's published national data. These households keep the county that the national data gives them.
3. **Imputation.** We give ACS households the characteristics the ACS lacks, such as detailed tax items, from the donor households, with quantile regression forests (working paper: [microcosm.institute/support/paper](https://microcosm.institute/support/paper)). Imputation gives records their characteristics. It does not change how many households each record represents.
4. **Calibration.** We then adjust the household weights so that the file matches 16,440 official statistics at once: 4,482 for states, 436 for congressional districts and 11,522 for counties. A cap keeps each household's final weight at most five times its starting weight. A paper on this calibration method is in preparation.

The targets come from IRS Statistics of Income (county tax returns and adjusted gross income; state detail on income items, deductions and credits, by income band), USDA SNAP (state households and benefits, fiscal year 2024), CMS Medicaid enrollment (state, 2024), Census population (state, district and county) and ACS households (state and county).

To test the file, we held out 1,052 county targets, about one in ten, from the calibration. We also kept five ACS county tables out of the calibration completely: total earnings, households with earnings, total household income, the household income distribution, and median household income.

## How well the file fits

We measure each target's error as the difference between the file's estimate and the official value, divided by the official value. "Within 10%" means an error of 10% or less in either direction.

### States and the nation

| Targets | Number | Within 10% | Median error |
|---|---|---|---|
| IRS tax statistics, by state | 4,227 | 96.0% | 0.12% |
| SNAP households and benefits, by state | 102 | 100% | 0.11% |
| Medicaid enrollment, by state | 51 | 100% | 0.08% |
| Population, by state | 51 | 100% | 0.07% |
| ACS households, by state | 51 | 94.1% | 4.2% |
| Population, by congressional district | 436 | 92.2% | 0.58% |

National totals are the sum of the state estimates. Most national tax totals are within 4.5% of the official value. A few groups are further off:

- tax returns with adjusted gross income of $1 million or more: 129% too high;
- returns with capital gains: 32% too low;
- taxable interest on returns with adjusted gross income below $1: 29% too low;
- returns with taxable interest and adjusted gross income of $500,000 or more: 13% too high;
- the total number of households: 5.2% too low.

### Counties: targets used in calibration

| County targets | Number | Within 10% | Median error |
|---|---|---|---|
| Population | 3,144 | 67.5% | 3.7% |
| IRS tax returns | 2,702 | 70.8% | 3.6% |
| IRS adjusted gross income | 2,846 | 82.3% | 0.4% |
| ACS households | 2,830 | 47.6% | 10.8% |
| **All county targets** | **11,522** | **67.1%** | **3.4%** |

In 856 counties (27%), every county target used in calibration is within 10%. Before calibration, only 15% of county targets were within 10%.

### Counties: targets held out of calibration

For the 1,052 held-out county targets, we compare the file with a simple baseline: each state's total split across its counties by population.

| Held-out county targets | Number | Median error: file | Median error: baseline | Within 10%: file | Within 10%: baseline |
|---|---|---|---|---|---|
| IRS tax returns | 441 | 14.8% | 5.7% | 36.7% | 73.0% |
| IRS adjusted gross income | 297 | 32.8% | 39.2% | 13.5% | 11.8% |
| ACS households | 314 | 19.6% | 5.4% | 26.1% | 76.4% |
| **All** | **1,052** | **20.5%** | **8.1%** | **27.0%** | **56.8%** |

The file beats the baseline for county adjusted gross income, but not for counts of tax returns or households.

### Counties: ACS statistics never used in calibration

| ACS county statistic | Median error: file | Median error: baseline | Counties where the file is closer |
|---|---|---|---|
| Total earnings | 19.3% | 28.8% | 61% |
| Total household income (approximate) | 13.8% | 21.3% | 63% |
| Households with earnings | 15.8% | 7.2% | 30% |
| Household income distribution (16 brackets) | 101%¹ | 25%¹ | 0.4% |

The file also ranks counties better than the baseline by earnings per person and by household income per person. The rank correlation with the official values is 0.57 for the file against 0.37 for the baseline for earnings, and 0.68 against 0.46 for household income. For households with earnings, the two are about equal (0.28 and 0.27). Within a state, the population-share split cannot rank counties at all; there, the file's median rank correlation is 0.49 for earnings and 0.64 for household income.

The file's median household income is too high in most counties: in the median county it is 17% above the published ACS value. Its rank correlation with the published county medians is 0.30.

### Compared with the earlier state-and-district file

PolicyEngine's published 2024 state-and-district file did not have county targets. On the 4,459 targets that both files used, the share within 10% is almost the same: 97.0% before and 96.9% now. Adding the county targets did lower two things: congressional district population fell from 97.0% to 92.2% within 10%, and the national effective sample fell from about 12,100 to about 7,900.

## How reliable each county is

Each county gets a provisional reliability grade from two measures: its effective sample and its largest error. The largest error is taken over the county's calibration targets, its held-out targets and its ACS checks.

| Grade | Rule | Counties |
|---|---|---|
| A | Effective sample at least 400, and largest error at most 10% | 0 |
| B | Effective sample at least 100, and largest error at most 20% | 3 |
| C | Effective sample at least 30, and largest error at most 50% | 103 |
| D | Lower than C, with test evidence | 3,038 |
| U | No test evidence | 0 |

The effective sample is the number of equally weighted households that would give the same precision as the county's actual weights. It is small in almost every county:

- The median county's effective sample is about 6. The largest is about 146.
- Only 167 counties reach an effective sample of 30. Together they hold 22.6% of the US population.
- Of the 279 counties with 250,000 people or more, 246 have an effective sample below 30.

Most D grades have both causes: 2,905 counties have an effective sample below 30 and a largest error above 50%. The largest error comes from the household income distribution in almost every county.

**Why the effective samples are small.** There are three reasons:

1. **Donor households carry most of the weight.** The 57,240 national donor households are 3.6% of the records, but they carry 77% of the household weight. Before calibration they carried half. In each county, a few of them hold most of the weight.
2. **Calibration concentrates the weight further.** After calibration, 1% of the household records hold 81% of the weight. Before calibration, they held 51%.
3. **The calibration gives most of its attention to state targets.** State and district targets make up 94% of the calibration's weighting, and county targets 6%. So the calibration fits state totals closely and county targets more loosely.

## Limits: how to use the file

**Group results by county** with the household table's `county_fips_2023` column. The reliability files use the same county codes. The model does not read this column, so join it to model results by `household_id`. The repository's README shows how.

**Use with care:**

- county totals of population, tax returns, adjusted gross income, earnings and household income;
- rankings of counties by earnings or income per person;
- state results, and national results other than the tax groups listed above.

**Do not use yet:**

- distributions within a county, such as income brackets, poverty rates, or the share of households with earnings;
- county median household income, which the file overstates;
- results for single small counties. Ten counties have only one or two household records.

**Other limits of this proof of concept:**

- **County assignment within a PUMA.** We assign ACS households to counties by population share within their PUMA. Many PUMAs, especially in rural areas, cover several counties. Inside such a PUMA, differences between counties come only from the calibration to county targets, not from the survey itself.
- **Donor households are national records.** They keep the county that PolicyEngine's national data gives them, and their characteristics come from national surveys, not from the county.
- **The grades are provisional.** The grade rule will change as the method improves.
- **Temporary data shortcuts.** For this proof of concept, strike benefits are zero for every household, because no household in the source surveys reports them. S-corporation income is not separated from partnership income: the national data holds both as one combined partnership amount, so the separate S-corporation items are zero.
- **Model use.** The file loads with policyengine-us 2.2.1's standard dataset loader. On a computer with 32 GB of memory, the national simulation built in 90 seconds and used at most 18 GB, and it computed 2024 totals of population, employment income and adjusted gross income. For one state, the model also computed income tax, SNAP and household net income, with no missing values. We have not run full-model calculations, such as household net income, for the whole country at once. On a 32 GB computer, run them state by state or in chunks, or use a larger machine. The model ignores 254 extra columns: raw survey fields and build identifiers.
- **One year only.** The file is for 2024. The 2026 and 2027 files are a stretch goal.

## Next steps

1. **Raise the effective sample in each county.** Spread each county's weight over more households, mainly by lowering the share of the weight that donor households carry. We will measure progress with the effective sample per county and the share of the population in counties at grade C or better.
2. **Balance the contribution of cloned records to the calibration.** Keep any single source household, together with its copies, from dominating the weights that the calibration chooses.
3. **Improve the calibration fit.** Give the county targets more weight, so that held-out county targets beat the population-share baseline, and fix the national tax outliers listed above. We will also bring district population back to its earlier fit.
4. **Complete the stretch goals.** Produce the 2026 and 2027 county files, the household measures and the schema manifest.
5. **Build the final County Files.** Rebuild from PolicyEngine's certified national data and remove the shortcuts this proof of concept uses.

## Appendix: file provenance

| Item | Value |
|---|---|
| Build | `populace-us-2024-buildo-acs-local-f8b1039f00-20261118T011728Z` |
| File | `populace_us_2024_acs_local.h5`, 10,637,811,669 bytes |
| SHA-256 | `82314a3b5a6a95762faf6eb4f5b499ce56608ad6a39a27b75e11c75b4945bd3c` |
| Built | 17–18 November 2026 |
| Code | Microcosm, commits `72b7072bf` (ACS staging) and `f8b1039f0` (calibration and packaging) |
| Model | policyengine-us 2.2.1 |
| Checks | All 23 automated release checks passed. |
| Loader test | On 18 November 2026, `Microsimulation(dataset=<file>)` in policyengine-us 2.2.1 built the national simulation in 90 seconds, with a peak of 18 GB of memory |
| Reliability files | `county_reliability.csv` and `county_reliability.json`, one record per county, with its grade, effective sample and errors |

---

¹ The income distribution error is the sum, over the 16 ACS income brackets, of the differences between the estimated and published numbers of households, divided by the county's number of households. When the estimated and published totals agree, an error of 100% means that half of the households fall in the wrong bracket.

Sources: US Census Bureau, American Community Survey 2024 1-year public-use microdata and county tables B11001, B19001, B19013, B19025, B19051 and B19061; IRS Statistics of Income, state and county data; USDA Food and Nutrition Service, SNAP state participation and benefits, fiscal year 2024; CMS Medicaid enrollment, 2024; US Census Bureau population data; US Census Bureau and Bureau of Labor Statistics, Current Population Survey Annual Social and Economic Supplement, and the IRS Statistics of Income public-use file, both through PolicyEngine's national data.
