# microcosm#297 receipt: CPS educational attainment against ACS S1501

Generated 2026-10-22 by `experiments/build_297_attainment_receipt.py` at commit `e2621e231c6cbc62939711e7c967be314624f65c`, Modal run `20261022T171517Z-e2621e231` (`experiments/modal_297.py`). Machine-readable copy: `experiments/receipts/297-cps-educational-attainment-s1501.json`.

The build stage ran on the published release `populace-us-2024-spm-20260915` (166,321 persons, 57,240 households) at its calibrated weights. Every person joined its own Census member row; the signal gate passed. The S1501 facts are a **temporary draft input** (PolicyEngine/chronicle#347, unmerged): 676 facts, 357 compiled targets (51 states and DC x 7 classes, population 25 and over), 0 mismatches against the raw Census response, largest gap between a state's categories and its total 0.

## Classes

| Class | CPS ASEC `A_HGA` | ACS PUMS `SCHL` | Six-class view |
|---|---|---|---|
| `less_than_9th_grade` | 31-34 | 01-11 | less than high school |
| `9th_to_12th_grade_no_diploma` | 35-38 | 12-15 | less than high school |
| `high_school_graduate_includes_equivalency` | 39 | 16, 17 | high school or GED |
| `some_college_no_degree` | 40 | 18, 19 | some college, no degree |
| `associates_degree` | 41, 42 | 20 | associate's |
| `bachelors_degree` | 43 | 21 | bachelor's |
| `graduate_or_professional_degree` | 44, 45, 46 | 22, 23, 24 | graduate or professional |
| `not_in_universe` | 0 (Children) | blank, or under 15 | |

## National, population 25 and over (sum of the 51 state cells)

| Class | S1501 | Release | S1501 share | Release share |
|---|---:|---:|---:|---:|
| < 9th grade | 10.91M | 7.82M | 4.6% | 3.3% |
| 9th-12th, no diploma | 12.87M | 11.70M | 5.5% | 5.0% |
| High school or GED | 60.70M | 65.69M | 25.7% | 27.9% |
| Some college, no degree | 43.65M | 33.29M | 18.5% | 14.1% |
| Associate's | 20.85M | 26.24M | 8.8% | 11.1% |
| Bachelor's | 52.23M | 57.04M | 22.1% | 24.2% |
| Graduate or professional | 34.70M | 33.79M | 14.7% | 14.3% |

## State cells (357)

| Measure | Value |
|---|---:|
| Mean abs share error | 2.29 pp |
| 90th percentile abs share error | 5.21 pp |
| Max abs share error | 10.84 pp |
| Median abs level error | 15.3% |
| 90th percentile abs level error | 46.3% |
| Max abs level error | 83.3% |
| Fewest records in a cell | 4 |
| Fewest households in a cell | 3 |
| S1501 25+ total vs release 25+ population, median abs | 0.14% |
| S1501 25+ total vs release 25+ population, max abs | 3.94% |

Share error is the release's share of a state's population 25 and over in a class minus S1501's; level error compares counts.

### Classes the degree-ROI analysis uses

Two-year degree = `associates_degree`, four-year degree = `bachelors_degree`, each against the `high_school_graduate_includes_equivalency` baseline.

| Role | Class | Mean abs share error | Max abs share error | Median abs level error | Fewest records |
|---|---|---:|---:|---:|---:|
| baseline | `high_school_graduate_includes_equivalency` | 2.87 pp | 10.84 pp | 8.5% | 95 |
| two_year_degree | `associates_degree` | 2.45 pp | 7.29 pp | 19.2% | 21 |
| four_year_degree | `bachelors_degree` | 2.32 pp | 8.11 pp | 8.7% | 221 |

### Largest share errors

| State | Class | S1501 share | Release share | Error | Records |
|---|---|---:|---:|---:|---:|
| MS | High school or GED | 29.4% | 40.2% | +10.84 pp | 527 |
| MN | Some college, no degree | 19.2% | 9.3% | -9.88 pp | 230 |
| HI | Some college, no degree | 18.6% | 8.8% | -9.82 pp | 240 |
| MT | Some college, no degree | 22.7% | 13.3% | -9.41 pp | 218 |
| AK | High school or GED | 28.5% | 36.8% | +8.32 pp | 464 |
| ND | Some college, no degree | 21.7% | 13.4% | -8.31 pp | 178 |
| VA | Bachelor's | 24.1% | 32.2% | +8.11 pp | 718 |
| MD | Bachelor's | 23.2% | 31.0% | +7.86 pp | 545 |
| GA | Some college, no degree | 18.8% | 11.0% | -7.86 pp | 276 |
| ME | High school or GED | 29.5% | 37.0% | +7.50 pp | 471 |

### Reading the comparison

- **Some college against associate's is an instrument difference.** Nationally the release has 14.1% some college, no degree against S1501's 18.5%, and 11.1% associate's against 8.8%. The release is below S1501 on some college in 49 of 51 states and above it on associate's in 44. The gap has one sign almost everywhere, so it reads as a difference between the instruments rather than sampling: the CPS interviews by telephone or visit and offers occupational and academic associate's answers, the ACS is mostly self-response and includes group quarters (Census, "Educational Attainment Fact Sheet", https://www.census.gov/topics/education/educational-attainment/guidance/factsheet-acs-cps.html). Binding the two cells separately would move weight to settle a definition. Their sum is -2.1 pp off nationally (25.3% against 27.3%), less than the some-college cell alone; the merged rows below give the state spread.
- **Thin cells.** 13 cells have fewer than 30 records; 10 of them are `less_than_9th_grade`. The standalone calibration's largest weight ratio (131) is in LA. Merging the two classes below a diploma (the six-class view) thickens those cells.

| Merged row | Mean abs share error | Max abs share error |
|---|---:|---:|
| less_than_high_school | 1.73 pp | 6.44 pp |
| some_college_or_associates | 2.93 pp | 7.45 pp |

## Held out: population 18 to 24

Not compiled: the ACS counts college students in dormitories (group quarters) in the state of their college, the CPS in their parents' household.
Mean abs share error 6.81 pp, max 34.03 pp; mean signed error by category: less_than_high_school_graduate +2.43 pp, high_school_graduate_includes_equivalency -3.96 pp, some_college_or_associates_degree +2.82 pp, bachelors_degree_or_higher -1.28 pp.

## Standalone calibration to the cells

Per-state minimum-KL reweighting of the release's household weights to the seven 25+ cells, holding the state's population under 25 fixed. Not the release solve, which fits every target jointly; it shows what binding these cells alone asks of the weights.

| Measure | Value |
|---|---:|
| Max abs level error after | 0.0000% |
| Weight ratio, lowest state 1st percentile | 0.380 |
| Weight ratio, highest state 99th percentile | 5.944 |
| Weight ratio, min / max | 0.176 / 130.971 |
| Kish effective sample size kept, lowest state | 82.6% |

## SCF education-debt imputation (microcosm#1073, PR #1246)

The #1073 hold-out already measured SCF `educ` as a predictor: `build_plus_education` raised the household rank correlation (0.168 to 0.207) but moved the share with debt from -0.30 pp to -0.92 pp. With this column the host side of that predictor exists: map SCF `educ` (reference person) and the host reference person's `cps_educational_attainment` to the same seven classes, and condition the QRFs on it. The share error suggests adding it with the share calibrated to FSA borrower counts (microcosm#1074) rather than as a bare predictor. Nothing here changes #1246.
