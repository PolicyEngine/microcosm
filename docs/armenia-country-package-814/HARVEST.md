# Armenia v1 fact-harvest contract (#814)

This is the Chronicle handoff for the engine-free Armenia package. The country
spec contains selectors only: observed values must be published by a future
`ledger-am` fact package and must never be embedded in
`target_references.json`. This inventory was prepared offline from the evidence
in issue #814; `HARVEST` below means that the exact table, value, or primary URL
was not supplied and must not be guessed.

For every fact, Chronicle must retain the publisher, exact landing and machine
URL, table or file identifier, retrieval date, reference period, units,
dimensions, revisions, uncertainty where published, and the transformation from
source cells to the declared Ledger key. It must also prove that each activated
calibration row maps to a direct donor column or a deterministic pre-built
indicator. No Armenian rules engine may be used to manufacture a v1 measure.

## One-to-one target-reference worklist

There is exactly one row below for each current entry in
`am/target_references.json`.

| Target reference | Future `ledger-am` key | Value and cells to publish | Source/table | URL | Required vintage and harvest status |
|---|---|---|---|---|---|
| `armstat_population_by_age_sex_marz` | `armstat.population.age_sex_marz` | Population counts by age × sex × marz, including a reconciled national total | ArmStat Statbank population table, 2022-census-vintage series | <https://statbank.armstat.am> | 2024. **HARVEST:** PxWeb table code, API query, category codes, the projection/recalibration metadata, and all cells for the 10 marzes plus Yerevan. |
| `armstat_ilcs_households_by_size_marz` | `armstat.ilcs.households_by_size_marz` | Official-weighted household counts by reviewed household-size band × marz | 2024 ILCS household file, joined to members only as needed to establish household size | [household files](https://armstat.am/en/?nid=205); [member files](https://armstat.am/en/?nid=206); [NADA catalogue](https://microdata.armstat.am/index.php/catalog) | 2024, with weights recalibrated to the 2022 census. **HARVEST:** exact files, variables, missing-value rules, official weight, size bands, current sample size, and weighted cells. |
| `armstat_ilcs_households_by_consumption_band_marz` | `armstat.ilcs.households_by_consumption_band_marz` | Official-weighted household counts by reviewed consumption band × marz; the bands describe raw consumption, not poverty status | 2024 ILCS household and consumption/product files | [household files](https://armstat.am/en/?nid=205); [product files](https://armstat.am/en/?nid=207); [NADA catalogue](https://microdata.armstat.am/index.php/catalog) | 2024, 2022-census-recalibrated weights. **HARVEST:** exact welfare aggregate, reference period, price treatment, band boundaries, indicator recipe, weight, and cells. |
| `armstat_ilcs_households_by_income_band_marz` | `armstat.ilcs.households_by_income_band_marz` | Official-weighted household counts by reviewed raw-income band × marz, retained as a diagnostic because income is underreported and seasonal | 2024 ILCS household/member files | [household files](https://armstat.am/en/?nid=205); [member files](https://armstat.am/en/?nid=206); [World Bank documentation](https://microdata.worldbank.org/index.php/catalog/3591) | 2024, 2022-census-recalibrated weights. **HARVEST:** income concept, recall period, annualisation, missing/negative-value treatment, band boundaries, indicator recipe, and cells. |
| `armstat_lfs_employed_by_age_sex_marz` | `armstat.lfs.employed_by_age_sex_marz` | Official-weighted employed-person counts by age band × sex × marz | 2024 ArmStat Labour Force Survey member microdata | [ArmStat microdata index](https://armstat.am/en/?nid=15) | 2024. **HARVEST:** exact LFS file URL, employment-status codes, age bands, sex and marz codes, official weight, universe, and cells. |
| `armstat_lfs_employees_by_industry_sex_marz` | `armstat.lfs.employees_by_industry_sex_marz` | Official-weighted employee counts by industry × sex × marz | 2024 ArmStat Labour Force Survey member microdata | [ArmStat microdata index](https://armstat.am/en/?nid=15) | 2024. **HARVEST:** exact file URL, employee-status and industry codes, classification vintage, sex/marz codes, official weight, universe, and cells. |
| `armstat_src_payroll_employees_by_industry_sex_marz` | `armstat.src.payroll_employees_by_industry_sex_marz` | Administrative payroll-employee counts by industry × sex × marz | One of the nine ArmStat-published State Revenue Committee register series available since 2018 | <https://statbank.armstat.am> | 2024. **HARVEST:** exact dataset/table and URL, count concept, coverage, industry classification, dimensions, suppression/revision policy, and cells. |
| `armstat_src_payroll_wages_by_industry_sex_marz` | `armstat.src.payroll_wages_by_industry_sex_marz` | Administrative wage amount by industry × sex × marz, in a documented period basis and currency | Matching ArmStat/State Revenue Committee payroll-register series | <https://statbank.armstat.am> | 2024. **HARVEST:** exact dataset/table and URL, whether the published value is a total or mean, AMD units, monthly/annual basis, coverage, dimensions, and cells. Do not activate until the donor candidate has a reviewed destination-unit bridge. |
| `armstat_average_monthly_nominal_wage` | `armstat.wages.average_monthly_nominal_wage` | Mean monthly nominal wage; issue evidence gives **AMD 287,172 for 2024**. Chronicle must pair it with a compatible count before applying `count_x_mean` | ArmStat average-monthly-nominal-wage time series | **HARVEST:** exact ArmStat table and URL | 2024. Validation-only. Harvest the exact cell, units, population coverage, revision status, and compatible employee count; never fit it alongside the equivalent payroll components plus total. |
| `armstat_pensioner_caseload` | `armstat.pensions.pensioner_count` | Current pensioner count. The supplied **about 446,249 pensioners in January 2022** is stale evidence and is not the target value | ArmStat pensioner-count time series | **HARVEST:** exact ArmStat table and URL | 2024. **HARVEST:** same-period count, exact reference date/average-period concept, coverage, and revision status. Do not populate this key with the 2022 evidence. |
| `armstat_pension_payment_total` | `armstat.pensions.payment_total` | Same-vintage pensioner count × average pension. Issue evidence gives **AMD 49,605 in December 2023** for the mean, but no compatible 2024 pair | ArmStat Table `AM.G017` (average pension, cited via CEIC) plus the matching ArmStat caseload series | **HARVEST:** direct ArmStat table URLs and source files | 2024. **HARVEST:** same-period count and mean, payment frequency, units, coverage, and aggregation recipe. Never multiply the January 2022 count by the December 2023 mean. |
| `armstat_family_social_benefit_families` | `armstat.social_protection.family_social_benefit_families` | Current count of families receiving family and social benefits | ArmStat time series “Number of families receiving family and social benefits” | **HARVEST:** exact ArmStat table and URL | 2024. **HARVEST:** exact cell, reference-period concept, family unit, program coverage, and revision status. Rule-derived eligibility is not a substitute for the administrative count. |
| `armstat_sna_household_consumption` | `armstat.sna.household_final_consumption_expenditure` | SNA 2008 household final-consumption-expenditure amount for macro validation | ArmStat institutional-sector accounts, household-sector chapter / GDP by income-generation materials | [ArmStat national accounts](https://armstat.am/en/?nid=202) | 2024. **HARVEST:** exact table/file URL, cell, AMD units and scale, sector/perimeter, revisions, and analyst-approved bridge to the direct candidate aggregate. Validation band only, never a solver objective. |
| `armstat_sna_household_disposable_income` | `armstat.sna.household_disposable_income` | SNA 2008 household disposable-income amount for macro validation | ArmStat institutional-sector accounts, household-sector chapter | [ArmStat national accounts](https://armstat.am/en/?nid=202) | 2024. **HARVEST:** exact table/file URL, cell, gross/net concept, AMD units and scale, sector/perimeter, revisions, and analyst-approved bridge to the direct candidate aggregate. Validation band only, never a solver objective. |

Before activation, reconcile leaf components to published totals and choose one
representation as the calibration objective. Component rows and their total must
not both contribute independent loss. For every amount row, the Ledger fact and
candidate column must also share currency, unit scale, reference period, and
population concept. The current US support pool is not, by itself, evidence that
its dollar-valued columns are suitable AMD candidates.

## Support-artifact authentication and column proof

The v1 base is a pre-built, public `policyengine/populace-us` support artifact,
not raw Armenian microdata and not a newly pooled ASEC build. Before any runtime
binding or release, harvest and pin:

- the exact Hugging Face repository type, immutable revision, filename, format,
  SHA-256, byte size, build/data identifier, and publication date; no `latest`
  or branch fallback is permitted;
- the artifact's exact licence text and redistribution terms;
- table grains, identity/link columns, weight kind, row counts, dtypes, units,
  missingness, value ranges, and support-stratum definitions;
- proof for every output declared by the Armenia source contract, including the
  direct age, sex, household-structure, consumption, income, employment,
  pension, and family-benefit candidates and every deterministic indicator
  recipe; and
- the permanent per-record provenance fields and release banner identifying
  every record as a US donor support record.

If the certified artifact lacks a declared direct column or deterministic
pre-built indicator, compilation/execution must refuse the target rather than
derive it from Armenian tax-benefit rules or silently substitute another field.

## Geography harvest

Harvest the 2022-census-vintage PxWeb distribution used to assign marz and the
complete post-amalgamation community parent/child roster. The evidence supplied
for #814 establishes **10 marzes plus Yerevan (11 top-level units)** and **71
consolidated communities** after the 2021–22 reform. Those counts are checks, not
a licence to invent codes.

The geography fact package must provide the exact table/API query, marz and
community codes and labels, the 71-to-11 parent mapping, population counts,
vintage/effective dates, and any suppressed or zero cells. It must document how
Yerevan is encoded and whether the published community distribution has a
different reference date from the population calibration table. The current
clone count is inherited configuration from the Belgium walking skeleton, not
an Armenian statistic; runtime activation requires a support review.

## ILCS access, sample, and licence verification

The supplied evidence says that ILCS is annual since 2001; public household and
member-level microdata are downloadable for 2004–2024 in SPSS/XLS; the
post-2012 design has approximately 5,184 households per year; and collection
uses monthly rotation with representativeness only to marz level. The exact 2024
sample size, file set, variable dictionaries, join keys, official weights, and
rotation treatment remain harvest items. The 2024 weights were recalibrated to
the 2022 census, creating a series break against 2022–23; retain that break in
fact provenance.

ArmStat open dissemination is treated as public in the release contract, but
Chronicle must capture the exact licence text, attribution, modification and
redistribution terms, and any per-file conditions. This is especially important
for a future native-ILCS v2; the donor-pool v1 republishes no ILCS unit record.

## Holdouts and non-target validation evidence

These supplied facts must not be copied into a calibration selector merely
because they are known:

- The ArmStat 2024 poverty snapshot reports **21.7%** poverty at the average
  line, with a **19.5–23.9%** confidence interval, and **0.6%** extreme poverty.
  The source is [the 2020–2024 poverty snapshot](https://armstat.am/file/article/poverty_2025_en_2.pdf).
  Poverty is a permanent holdout because it is downstream of survey-measured
  tax-benefit quantities. Consumption, not income, is ArmStat's official welfare
  measure; income is underreported and seasonal.
- The #814 charter reports that a Central Bank of Armenia UNECE GENA 2024
  presentation put survey consumption at **26.9% of macro household
  consumption**. Its primary presentation URL, exact numerator/denominator
  concepts, period, and cell must be harvested before the observation can inform
  a broad macro-realism band; the reported ratio is not itself an acceptance
  threshold.
- OECD Revenue Statistics in Asia and the Pacific 2025 reports 2023 PIT of
  **AMD 554,554 million (5.87% of GDP)**, CIT **AMD 321,521 million**, SSC
  **AMD 90,292 million**, VAT **AMD 767,174 million**, excises **AMD 149,446
  million**, and total tax **AMD 2,221,925 million (23.5% of GDP)**. Harvest the
  exact country-note URL, table cells, units, revisions, and component hierarchy.
  These administrative facts are fiscal cross-checks, not current v1 target
  references; do not introduce component-plus-total double counting.
- The Ministry of Finance 2025 outturn reports revenues of **AMD 2,886 billion**,
  including tax revenues and duties of **AMD 2,725 billion**. Harvest the exact
  budget-execution report URL and cells. This is a 2025 cross-check only.
- ArmStat reports average monthly nominal wages of **AMD 287,172 in 2024** and
  **AMD 303,140 in 2025**. The 2024 value maps to the validation-only reference
  above; the 2025 value is a cross-check and must not leak into a 2024 target.
- Supplied family-benefit context gives a basic amount of **AMD 18,000 per
  month**, plus **AMD 5,500–8,000 per child**, and spending of approximately
  **0.5% of GDP in 2021**. These rule/context facts do not fill the missing 2024
  administrative caseload and are not engine outputs for v1.

## External-oracle validation-band registry

The following studies remain documentation-only external oracles. None has a
Ledger fact or an approved band endpoint, so none is a gate. For each, harvest
the full citation, primary URL, downloadable tables, policy/data vintage,
population and income concept, estimate and uncertainty, and the proposed low
and high band endpoints:

| Oracle | Vintage supplied in #814 | Harvest before use |
|---|---:|---|
| CEQ Working Paper 43 fiscal-incidence results | 2017 | Exact paper/version and URL; Armenian estimates, units, survey/policy vintage, uncertainty, and concept mapping |
| World Bank PIT-reform microsimulation | 2019 | Exact report and URL; baseline/reform scenario definitions, estimates, units, uncertainty, and concept mapping |
| World Bank Armenia fiscal-incidence work | 2025 | Exact report and URL; estimates, source-data vintage, units, uncertainty, and concept mapping |

`validation_bands.json` may carry these only after review, with explicit source,
period, measure, units, and low/high values. Band endpoints must come from the
evidence and an ex-ante steward decision, never from the candidate being tested.
