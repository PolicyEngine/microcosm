---
kind: proposal
client: "Living Wage Institute"
status: "draft v0.1 (2026-11-19): for María and Max to review, not for sending"
pe-doc:
  psl_foundation: false
  split_tables: true
  pdf_name: "PolicyEngine - County Files proof of concept methods"
---

# County Files proof of concept: methods

**How we built and tested the 2024 county file**

A PolicyEngine methods note · for the Living Wage Institute · November 2026

This document describes how we built the 2024 county file for the County Files proof of concept (PoC), as the build ran on 17–18 November 2026. We ran the build in two parts with PolicyEngine's Microcosm code; the appendix gives the code versions, the settings and the digests of the inputs:

- **Staging** builds the household records from the American Community Survey (ACS) and a national donor file, gives each household a county, and imputes the inputs the ACS lacks.
- **The release build** runs seven stages in order. `fills` and `agi` check the staged file first: its blank inputs, and each county's households and income. `materialize` sets program take-up, moves nine ACS households into three counties that the county draws left empty, fills reviewed defaults, and builds the targets and the holdout. Take-up means whether an eligible person or household receives a benefit. `calibrate` sets the weights. `qa` runs PolicyEngine's tax-benefit model on the calibrated file. `finalize` and `package` write the release files and run the last checks.

The PoC report gives the results, and the schema documentation lists the file's tables and columns; both come with this document. We have not yet produced the 2026 and 2027 files or the household measures the agreement lists (earnings, gross income and net income for each household), so this document covers neither.

## Data sources and years

We built the 2024 county file from two sets of household records and from published statistics, with PolicyEngine's US tax-benefit model. We list each source at the vintage the build read on 17–18 November 2026. The build records the SHA-256 digest of every input file; the appendix lists the main ones.

| Source | Vintage | What the build uses it for |
|---|---|---|
| ACS public-use microdata, household and person files | 2024 1-year | 1,529,668 ACS household records with 3,420,942 people. Of these, 181,260 are group-quarters records: one person each, living in a place such as a nursing home or college dormitory rather than a housing unit, and counting as a household of one. These counts come after staging drops group-quarters records of children under 15 (see "What we impute onto ACS households, and from which donors") |
| PolicyEngine's national donor file, release `populace-us-2024-spm-receipts-20260923`, in a copy we patched | See "Donor households" below the table | 57,240 donor household records with 166,321 person records; we train the imputation on their person records, tax units (the people on one tax return) and SPM units (the family unit of the Supplemental Poverty Measure) |
| Current Population Survey Annual Social and Economic Supplement (CPS ASEC), person files `pppub23`–`pppub25` | Survey years 2023–2025 (income years 2022–2024) | The donor households' survey records; three work columns our patch adds |
| Internal Revenue Service (IRS) Statistics of Income (SOI) public-use file (PUF), in PolicyEngine's processed version 1.8.0 | Tax year 2015, scaled forward to 2024 | The tax items on the donor file's tax-detail records, through PolicyEngine's national build (see "Donor households" below the table) |
| Census 2020 tract-to-PUMA (Public Use Microdata Area) relationship file; 2020 Census redistricting (P.L. 94-171) block populations; the Census file of each block's 119th Congress district | 2020 Census; 119th Congress districts | The PUMA ladder (see "Population" below): the county and district draws within each PUMA, and every population target |
| SOI Historic Table 2, state data | Tax year 2022; tax year 2023 for 408 rows | 4,227 state tax targets, and the state totals we split into county tax targets |
| SOI county file (`22incyallnoagi.csv`) | Tax year 2022 | Each county's share of the sum of its state's county rows, for returns and for adjusted gross income (AGI). Each of the two has 3,143 county targets (Kalawao and Maui share one); we hold out 441 return targets and 297 AGI targets and train on the rest |
| SOI national totals: Publication 1304 Tables 1.1, 1.4 and 4.3; the SOI congressional district file `22incd.csv` | Tax years 2022 and 2023; see "Tax targets" for the district file | Growth from 2022 to 2023, for aging SOI dollar amounts; national totals for rescaling some state rows |
| Congressional Budget Office (CBO) revenue projections by income source | February 2026 | Growth from 2023 to 2024, for aging SOI dollar amounts |
| US Department of Agriculture (USDA), Supplemental Nutrition Assistance Program (SNAP) state participation and benefits | Fiscal year 2024 | 102 state targets |
| Centers for Medicare & Medicaid Services (CMS), state Medicaid enrollment from its Medicaid and Children's Health Insurance Program (CHIP) enrollment data, April 2026 release | December 2024 (Rhode Island: November 2024) | 51 state targets; Medicaid take-up on ACS households |
| ACS county table B11001 (households) | 2020–2024 5-year | 51 state household targets, and 3,144 county household targets, of which we train on 2,830 and hold out 314 |
| ACS county tables B19061, B19051, B19025 and B19001; B19013 (median household income) | 2020–2024 5-year | Checks that never enter calibration; B19013 serves only as a diagnostic |
| Social Security Administration (SSA), Supplemental Security Income (SSI) Monthly Statistics, Table 1 | December 2024 | SSI take-up on ACS households |
| Census Survey of Income and Program Participation (SIPP) public-use file | 2023 file | SSI disability status on ACS households |
| Pew Research Center; Higher Ed Immigration Portal | Pages published July 2024 and March 2021 | National counts that set immigration status on ACS households |
| USDA SNAP participation rate; the Department of Health and Human Services (HHS) Office of the Assistant Secretary for Planning and Evaluation (ASPE) take-up rate for Temporary Assistance for Needy Families (TANF), from its 24th Welfare Indicators report | The run records no year | SNAP and TANF take-up on ACS households |
| policyengine-us; policyengine-core | 2.2.1; 3.32.5 | The model the build ran; the release manifest records both versions |

**Donor households.** Donor households are survey households from PolicyEngine's national donor file. We add them to the ACS households and train the imputation on their person records, tax units and SPM units. The file comes from three releases in the public Hugging Face dataset `policyengine/populace-us`: a national build and two later releases that add columns to it.

- The national build, `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z` of 28 July 2026, chose 57,240 household records from 337,704 candidates. Its records name its inputs as the CPS ASEC for income years 2022–2024 and PolicyEngine's processed PUF, version 1.8.0. In that build's code, version 1.8.0 is the 2015 public-use file, scaled forward to 2024.
- In PolicyEngine's national build code at that build's commit (`cae8640`), each CPS ASEC household appears as two records that share its weight equally. One keeps the household as the CPS ASEC recorded it. The other, a tax-detail record, takes its tax items from the PUF: quantile regression forests trained on the PUF's tax units draw them, given the tax unit's filing status, its number of people and its incomes (wages, self-employment income, taxable interest, dividends and capital gains). The code then spreads each tax unit's amounts over its members in their CPS shares. Because the national build then chose among these records, a donor household can appear in the donor file as one of its two records or as both (see "What donor records carry").
- Release `populace-us-2024-spm-20260915` added an SPM role flag. Release `populace-us-2024-spm-receipts-20260923` added reported receipt of SNAP, TANF and WIC (the Special Supplemental Nutrition Program for Women, Infants, and Children). Both drew on the same ASEC person files, and neither changed an existing column.

For this PoC we patched a copy of the last release. We added two industry columns and a worked-last-year column from the ASEC person files. We also added two columns that hold only zeros: strike benefits, and passive partnership and S-corporation income. We kept every household and person record.

**Population.** A PUMA is the smallest area the ACS microdata identifies. The PUMA ladder is a table of each 2020 PUMA's population, summed from 2020 Census block populations and split across census tracts, counties and congressional districts. For the county targets, we group the tracts by their 2023 county code; under those codes, Connecticut's tracts fall into its nine planning regions. We then scale each state's counties to the state's PUMA total. Every population target comes from this ladder, so states, districts and counties each sum to 331,449,281, the ladder's total of 2020 Census block populations. The county facts file also holds the Census Bureau's 2024 county population estimates and ACS table B01003 (total population); no target or check uses either.

**Tax targets.** Most state tax targets come from tax year 2022:

- 2,397 totals across all income levels, for income items, deductions, credits, tax, returns and AGI;
- 510 earned income tax credit (EITC) rows, in total and by number of children;
- 912 rows of taxable interest, amount and returns, in nine AGI bands.

The other 408 come from tax year 2023: returns and AGI in the four AGI bands from $100,000 up.

Before aging, we rescale some rows across tax years. Each 2023 AGI-band row keeps its share of its state's 2023 total, applied to the state's 2022 total of returns or AGI. We scale other 2022 rows to an SOI national total for tax year 2023: taxable interest (1,014 rows, in total and by AGI band) and net capital gains (102 rows, amount and returns). For the 51 net capital gains return rows, that total comes from the SOI congressional district file `22incd.csv`. In the target feed, that file carries the label tax year 2023 but the title "Congressional District Data 2022", so its tax year is uncertain.

We then age every SOI dollar amount to 2024 with national growth. A dollar row that we rescaled to a 2023 national total ages from 2023; every other dollar row ages from 2022. SOI's national totals for 2022 and 2023 give the growth from 2022 to 2023, and CBO's February 2026 projections give the growth from 2023 to 2024. Most dollar amounts grow with total AGI; wages and net capital gains grow with their own series. We do not age counts.

We split two state totals into county tax targets: returns, and AGI after aging. Each county's target equals its state's total times the county's share of the sum of its state's county rows in the 2022 SOI county file. The 2022 SOI county file and the ACS 2020–2024 county tables both already list Connecticut by its nine planning regions (09110 to 09190), so the build uses their rows as published, with no crosswalk. The SOI county file has no row for Kalawao County, Hawaii. The build's code cites the IRS guide to the county file, which says the IRS leaves out items with fewer than 20 returns for a county. Kalawao shares one cell with Maui County for these two targets. Apart from that cell, the build stops unless each state's county rows in the SOI county file and in ACS table B11001 match the 2023 counties of the population targets one for one.

**Program targets.** The SNAP targets are each state's average monthly households and total benefits in fiscal year 2024. The Medicaid targets are each state's total Medicaid enrollment in December 2024. CMS reports Rhode Island's December 2024 enrollment as zero, with a footnote that the state could not provide the data, so we use Rhode Island's November 2024 figure (273,400).

**ACS county tables.** These 5-year estimates cover survey responses from 2020 through 2024, with dollar amounts in 2024 inflation-adjusted dollars. Each county's household target equals its published B11001 estimate, and each state's household target equals the sum of its counties' estimates.

## How households get their counties

The 2024 county file holds 1,529,668 household records from the 2024 ACS 1-year public-use microdata and 57,240 national donor household records. Of the ACS records, 181,260 are group-quarters records, each for one person. These counts come after staging drops 1,946 group-quarters records of children under 15 (see "What we impute onto ACS households, and from which donors"). The two kinds of household get their geography in different ways.

### ACS households

Each ACS household record carries its state and its PUMA. We keep both. A PUMA can span several counties and congressional districts, so we assign those from the PUMA ladder (see "Population" under "Data sources and years").

We built the ladder from 2020 Census block populations (the P.L. 94-171 redistricting file) and the Census file of each block's congressional district. Blocks nest in tracts, and each 2020 tract lies in one 2020 PUMA. Summing block populations therefore gives the number of people each PUMA shares with each 2020 county and with each district. We call that number the population overlap. Of the ladder's 2,462 PUMAs, 610 overlap more than one county and 1,276 overlap more than one district.

Within its PUMA, each ACS household gets a district from one random draw, with odds proportional to the PUMA's population overlap with each district. It gets a county from a separate draw, with odds proportional to the PUMA's population overlap with each county. One random-number generator with a fixed seed (0) makes every draw. With the same inputs, row order and software versions, the code makes the same draws; we have not rebuilt the file to confirm this. Each household gets one district and one county and keeps its whole weight: the draw does not split a household across counties.

The draws use only the PUMA's population overlaps, not the household's survey answers, and the district draw does not depend on the county draw. The code checks only that the PUMA, district and county lie in the household's state. We found no check that a household's district and county overlap.

Staging gives ACS households no tract or block. In the file, the tract and block columns hold empty values for every ACS household.

The file carries two county codes. `county_fips` holds 2020 counties, which divide Connecticut into its eight former counties. `county_fips_2023` holds 2023 county-equivalents, which divide Connecticut into its nine planning regions (09110 to 09190). The county targets and the minimum-support step below use `county_fips_2023`. Outside Connecticut the two codes agree. For a Connecticut ACS household, staging also draws a 2020 tract within the household's PUMA and the county it drew, again by population, and records that tract's planning region. It does not keep the tract.

### Donor households

Donor households keep the district, county, tract and block stored in the donor file. Staging checks that each stored district and county lies in the household's state and that the county matches the tract's county. It sets the household's PUMA from the tract and takes `county_fips_2023` from the tract.

The donor file's household table comes unchanged from PolicyEngine's 28 July 2026 national build. Our patch added only person columns. By their published records, the two later releases added no household column and changed no existing column. The stored geography is therefore the national build's.

In PolicyEngine's national build code at that build's commit (`cae8640`), each household gets its geography from two draws. The first draws a congressional district within the household's state, with odds in proportion to each district's number of tax returns in the IRS congressional district data. The second draws one 2020 census block within that district, in proportion to block population. The tract and the county follow from the block. A donor household's county therefore comes from these draws, not from the county its CPS ASEC record names.

### Counties without households

Staging draws each ACS household's county at random and sets no minimum number of households per county. After the draws, three counties had no household of any kind: Kalawao County, Hawaii (15005), Blaine County, Nebraska (31009) and Loving County, Texas (48301). The 2020 Census counted 82, 431 and 64 residents in them. Before calibration, the `materialize` stage runs a minimum-support step. In this run, the step moved three households into each of the three counties. Its rules:

- It applies only to a county with residents in the ladder and no supporting household. Every household supports its county except ACS group-quarters records. A county with one or two supporting households does not qualify.
- It looks for movers first in the PUMA that shares the most people with the county, then in the next.
- A mover must be a supporting household with no stored tract or block, so donor households never move. Its two county codes must agree, and its own county must keep at least three supporting households.
- It ranks candidates by a hash of a fixed key (`acs-local-county-support-v1`), the destination county and the household ID, so row order does not matter. It moves the top three, or none if fewer than three qualify.
- A move rewrites the two county codes and nothing else. The household keeps its state, PUMA, weight and the district it drew.

The `materialize` stage records each move. When the `calibrate` stage writes the calibrated file, it reloads the staged file and applies the same recorded moves again.

Each county took its three households from its one overlapping PUMA:

| County | PUMA | Moved from (county: households) |
|---|---|---|
| 15005 Kalawao, HI | 1500100 | 15007: 1; 15009: 2 |
| 31009 Blaine, NE | 3100300 | 31079: 2; 31081: 1 |
| 48301 Loving, TX | 4807300 | 48165: 1; 48377: 1; 48389: 1 |

All nine are ACS households. After the step, all 3,144 counties have at least one supporting household: 1,405,648 in all, with a median of 144 per county and 35,707 in the largest county. Ten counties have only one or two, and the step does not apply to them. Counting every household record, group quarters included (1,586,908), the median county has 165 records, the largest has 39,788, and six counties have one or two.

## What we impute onto ACS households, and from which donors

Staging loads 1,531,614 household records with 3,422,888 people from the 2024 ACS 1-year public-use file; it skips the file's 100,355 vacant housing units. Before it imputes anything, it drops the 1,946 group-quarters records of children under 15. These records carry 0.04% of the ACS household weight, and the build records the drop as an interim exclusion. The build's recorded reason: each such child forms a one-person SPM unit with no adult. The PolicyEngine model version the build ran measures every SPM unit, so a single such unit makes it refuse every SPM measurement on the file.

That leaves 1,529,668 household records with 3,420,942 people. Of these, 181,260 are group-quarters records of one person each. The ACS records each person's age, sex, wages, self-employment income, reported SSI, disabilities, work hours, industry and occupation, and each household's tenure and property tax. PolicyEngine's model needs many more inputs, such as income by tax type, deductions, health coverage and program receipt.

We fill most of the missing inputs from PolicyEngine's national donor file: 57,240 household records with 166,321 person records. Staging runs the imputation. The release build's `materialize` stage then sets program take-up and gives the remaining blanks the model's default values.

### Method

We use quantile regression forests in Microcosm's own form, with sign classifiers, chained draws and donor weights. A quantile regression forest predicts the whole distribution of a value, not only its average. Staging runs three imputation passes in order. The first imputes usual weekly hours. The second imputes child support, retirement-account distributions, and work and disability income. The third imputes everything else. In each pass:

- **Training records.** Donor person records, tax units or SPM units. The first two passes train on the donors' ASEC person records and the third on their tax-detail copies (both defined below). Each forest resamples the donors in proportion to their weights in the donor file. The sign classifier below uses the same weights directly.
- **Predictors.** Age, sex and state, plus, where the ACS record has them, wages, self-employment income, Social Security, retirement income, interest, dividend and rental income, household-head status and housing tenure. Tax-unit and SPM-unit fits use the unit's number of people, of women and of household heads, its summed ages, state, housing tenure and summed incomes. Each pattern of available predictors gets its own fit, on donors that have those predictors and all of the family's columns, so the build never reads a blank as a zero.
- **Second-pass predictors.** For child support, the second pass adds the ACS's "all other income" amount. For retirement-account distributions and for work and disability income, it swaps the donors' retirement-income measure for one closer to the ACS's definition.
- **Joint within a family.** In this run, each family holds at most eight columns. The build splits a wider group of columns into batches, and each batch counts as its own family. The forests draw a family's columns in sequence, each conditioned on the predictors and on the columns already drawn. One family's draws do not enter another family's fit. For example, the forests draw the 54 person-level tax columns jointly only within each of their seven batches.
- **Random draws.** For each amount, a classifier picks the sign (negative, zero or positive), and the forest returns a value at a random quantile. The draws therefore follow the donors' distribution rather than an average. The run seed, 0, combined with each family's name and pattern of predictors, fixes every draw. Each forest in these passes has 32 trees.
- **Blanks only.** The forests fill only blank cells. Where the ACS reports a model input, its value stays as it is. Where the ACS reports only a combined amount, the build uses that amount as a predictor and draws the model's inputs from donors (next paragraph).
- **Weights unchanged.** Imputation passes the ACS household weights through unchanged. After imputation, staging pools the ACS and donor records and gives each group half of the donor file's household weight, 61.27 million each, before calibration.

The ACS reports Social Security, retirement income, and interest, dividend and rental income as one amount each per person. The build uses these amounts only as predictors. The model's inputs for these items, such as Social Security by type, pensions, individual retirement account (IRA) distributions, interest and dividends by tax treatment, and rent received, come from donor draws. No build step makes the draws add up to the ACS amount.

### What the run imputed

In the table below, "ASEC" and "PUF" name the two roles a donor person record can hold. An "ASEC" record holds a person as the CPS ASEC recorded them. A "PUF" record holds a tax-detail copy of such a person; the build code names this role `puf_tax_detail`. "Donor households", under "Data sources and years", describes how PolicyEngine's national build makes these copies.

Staging imputed 101 columns in 25 families:

| What staging imputes | Columns | Unit | ACS records filled | Donor records trained on |
|---|---:|---|---:|---:|
| Usual weekly hours | 1 | person | 39,800 | 50,490 ASEC |
| Child support received and paid | 2 | person | 3,420,942 | 66,001 ASEC |
| Retirement-account distributions: 401(k), 403(b), simplified employee pension (SEP), Keogh and Roth IRA | 5 | person | 3,420,942 | 66,001 ASEC |
| Workers' compensation and disability benefits | 2 | person | 3,420,942 | 66,001 ASEC |
| Interest, dividends, capital gains, pensions, Social Security by type, and business and deduction items | 54 | person | 3,420,942 | 100,320 PUF |
| Mortgage, health savings account deduction and two other tax items | 6 | tax unit | 2,095,799 | 48,755 PUF |
| Health coverage, full-time student, pregnancy and WIC | 12 | person | 3,420,942 | 100,320 PUF |
| Assets, medical costs, tax-exempt pensions, unemployment and veterans' benefits, and hours last week | 10 | person | 3,420,942 | 100,320 PUF |
| Own children in household | 1 | person | 3,420,942 | 100,320 PUF |
| Care need and adult-care spending | 2 | person | 3,420,942 | 100,320 PUF |
| Pre-subsidy rent | 1 | person | 3,239,682 (all but the 181,260 in group quarters) | 100,320 PUF |
| SNAP and TANF receipt, housing assistance take-up and childcare costs | 4 | SPM unit | 1,529,668 | 36,614 PUF |
| Capital gain distributions, a fixed share of imputed long-term gains, with no forest | 1 | person | 3,420,942 | none |

Three rules limit or replace forest draws:

- **Pregnancy.** Only women aged 15 to 44 get a forest draw (603,628 people). The build sets everyone else to not pregnant (2,817,314 people).
- **Adult care.** The build keeps adult-care spending only on people who qualify, at most one per tax unit. It removed the spending of 12,926 people who did not qualify and of 40 extra people in a tax unit; 6,348 people keep it.
- **SNAP receipt.** The ACS household's own answer replaces the forest's draw on every ACS SPM unit (see "Take-up and other assigned values").

### Self-employment, partnership and S-corporation income

The ACS reports one self-employment amount per person (SEMP). The 2024 ACS subject definitions, as the build code cites them, count farm income and income from a person's own partnership in SEMP, but not profit from an incorporated business. So:

- We keep each person's ACS self-employment amount. The forests still draw its three parts (farm operations, partnership self-employment earnings and specified-service business income). The build then sets all three to zero for all 3,420,942 ACS people. Of those draws, 44,478, 73,164 and 114,327 had a nonzero value. A gate, one of the build's automated checks (see "Automated gates"), found no ACS person whose self-employment differs by more than $1 from SEMP in 2024 dollars.
- The build sets partnership-income draws to zero for three groups. The first is people who report nonzero SEMP (171,732). The second is a head or spouse whose tax-unit partner reports nonzero SEMP (84,177). The 2024 ACS form asks people to report jointly received income under one person (item 43, as the code cites it). The third is people with blank SEMP, all aged under 15 (508,152). That removed 16,571 nonzero draws. At the ACS's own household weights, before pooling rescales them, they came to $159.4 billion of income and $13.9 billion of losses. Partnership income stays nonzero on 139,930 ACS records.
- S-corporation income and passive partnership and S-corporation income stay zero on every ACS and donor record. Both columns hold only zeros in the donor file, so the forests have nothing to learn from and draw zero. S-corporation income comes unchanged from the published donor file; our patch added the passive column (see "What donor records carry"). The recorded reason for both: the donor file holds partnership and S-corporation income as one combined partnership amount, with no record of the split.
- A staging check refuses any donor column that the third pass reads and that holds only default values, unless the staging command lists it. The command listed these two columns. Strike benefits, which also hold only zeros in the donor file, needed no listing, because the third pass does not read them.

### Take-up and other assigned values

Rules, seeded draws and other models set these values on ACS records. One of them, housing assistance receipt, copies a forest draw.

**Staging:**

- **SNAP receipt.** The ACS household's answer to the food stamp question replaces the forest's draw on every ACS SPM unit. It sets 137,941 units to true and the other 1,210,467 housing units to false. The 181,260 people in group quarters get false.
- **SSI disability criteria.** An archived quantile regression forest, trained on the 2023 SIPP with 100 trees and 9,346 training records, runs on ACS people. The code describes it as the donor file's own SSI model. ACS people under 65 who report SSI meet the criteria. In all, 303,302 ACS people meet them.
- **Immigration status.** The build reads ACS citizenship and exact year of entry. It maps other ACS fields, such as place of birth, health coverage at interview, class of worker, and Social Security and SSI receipt, onto the ASEC's legal-status indicators. Seeded draws then aim at national counts of undocumented workers (8.3 million, Pew Research Center) and undocumented students (408,000, Higher Ed Immigration Portal), minus what the donor records already carry.
- **Usual weekly hours.** 547,952 ACS people lacked them. The build gives the 508,152 children under 15 zero hours, and the forest imputes the other 39,800.

**The release build's `materialize` stage, before calibration:**

- **SNAP take-up.** The flag means a unit would take up SNAP if eligible; PolicyEngine's model decides eligibility. Units that report SNAP get the flag. Other units draw it at 79.8%, so that 82% of ACS SPM units, by weight, carry it: USDA's national SNAP participation rate.
- **TANF take-up.** Units draw the flag at 21.9%, the rate in HHS's 24th Welfare Indicators report. The draw ignores imputed TANF receipt: 10,559 of the 13,536 ACS units with imputed TANF receipt drew no take-up.
- **SSI and Medicaid take-up.** PolicyEngine's model first runs on the donor and ACS households to find who could get SSI or Medicaid. ACS people who report SSI then take up SSI, and people with ACS Medicaid or other means-tested coverage take up Medicaid. The rest draw toward SSA counts by age band and CMS enrollment counts by state, minus what the donor records already carry. In all, 1,835,876 ACS people take up SSI if eligible, and 3,359,039 take up Medicaid if eligible. The model applies eligibility.
- **Other flags.** Medicare take-up follows ACS Medicare coverage. Children aged 3 to 5 draw Head Start take-up at the donor records' weighted take-up share, 4.9%. Housing assistance receipt copies the imputed housing-assistance take-up, a forest draw, and stays false in group quarters. Adults aged 18 to 64 draw an exemption from SNAP's work time limit at 8%, the statutory cap.

When the `calibrate` stage writes the calibrated file, it applies these assignments again and stops if they differ from the ones the `materialize` stage recorded.

### Reviewed null fills

After these steps, 58 model inputs still have blank cells: 143,641,752 on ACS records, and 166,321 on donor records (every donor person record's reported SSI). The build writes the model's own default value into each of these 143,808,073 blanks. A reviewed list of 66 model inputs governs this step. Its other 8 entries cover the release build's assignments above: six take-up flags, housing assistance receipt and the SNAP work time limit exemption. The calibrated file stores these defaults. For example, every ACS person gets zero strike benefits, tips and hourly wage, no Hispanic flag, race code 0 and no self-employed flag. Every ACS household gets zero net worth, and every ACS tax unit takes up the EITC. A blank model input missing from the list stops the build.

### What donor records carry

The donor file's 166,321 person records come from the CPS ASEC for income years 2022, 2023 and 2024 (54,464, 54,654 and 57,203 records). Of these, 66,001 are ASEC person records and 100,320 are tax-detail copies. They hold 135,833 distinct people, each counted once per income year: 30,488 people appear twice, once in each role. In the county file, the 57,240 donor household records hold 47,277 distinct source households (see "Copies" under "The reliability grade").

For this PoC we used a patched copy of PolicyEngine's published donor file. The patch added three ASEC work columns (detailed industry, major industry and whether the person worked last year). It matched each of the 166,321 donor person records to its ASEC record exactly once, with no identity mismatches. It re-derived eight existing columns and found them equal on every record. It also added strike benefits and passive partnership and S-corporation income as zeros, and changed no column the published file already held. The patch tool's code gives a fixed reason for strike benefits: no donor record, and no record in the ASEC person files for survey years 2023 to 2026, reports them. The patch run itself read only the files for survey years 2023 to 2025, for the work columns.

## Calibration: the targets, how we weight them, and the weight cap

Calibration sets each household record's weight: the number of US households (for a group-quarters record, people) that the record stands for. Each record starts from its design weight (see "Design weights and the solver"). The `calibrate` stage then moves the 1,586,908 weights so that the file's totals come close to 16,440 targets: 4,482 for states, 436 for congressional districts and 11,522 for counties. No national target takes part.

Only some targets equal published figures. The state SNAP and Medicaid targets equal published state figures. The state SOI targets start from published state tables; we age SOI dollar amounts to 2024 and rescale some rows across tax years (see "Tax targets" under "Data sources and years"). Each county ACS household target equals the county's published B11001 estimate, and each state household target equals the sum of its counties' estimates. Each county SOI target splits a state target by the county's share in the SOI county file. It matches the county's own figure in that file only where the state target matches the sum of the state's county rows. In this run that holds for the county return counts of Utah (29 trained targets) and Vermont (14 held-out targets). Every population target, at every level, comes from the PUMA ladder.

### The targets

| Targets | Level | Trained | Held out | Source |
|---|---|---:|---:|---|
| SOI, all returns: the return count, AGI, and 45 return counts and dollar amounts for 23 income, deduction, credit and tax items | State | 2,397 | 0 | SOI Historic Table 2, tax year 2022 |
| SOI taxable interest: returns and dollars in 9 AGI bands | State | 912 | 0 | SOI Historic Table 2, tax year 2022 (the band under $1 has no row for 3 states) |
| SOI EITC: claims and dollars, in total and by number of children | State | 510 | 0 | SOI Historic Table 2, tax year 2022 |
| SOI return count and AGI in four AGI bands from $100,000 up | State | 408 | 0 | SOI Historic Table 2, tax year 2023 |
| SNAP: average monthly households and total benefits | State | 102 | 0 | USDA, fiscal year 2024 |
| Medicaid enrollment | State | 51 | 0 | CMS, December 2024 (Rhode Island: November 2024) |
| ACS households | State | 51 | 0 | Sum of the state's county estimates from ACS 2020–2024 5-year table B11001 |
| Population | State; district; county | 51; 436; 3,144 | 0 | The PUMA ladder |
| SOI return count; SOI AGI | County | 2,702; 2,846 | 441; 297 | SOI Historic Table 2 state totals, tax year 2022, split by shares from the SOI county file (`22incyallnoagi.csv`), tax year 2022 |
| ACS households | County | 2,830 | 314 | ACS 2020–2024 5-year table B11001 county estimates |
| **All** | | **16,440** | **1,052** | |

**County targets.** Each county target splits a state target in proportion to a county figure:

```text
county target = state target × county figure ÷ sum of the county figures in the state
```

- **SOI:** the state target is the state's all-returns count for tax year 2022, or its tax year 2022 AGI aged to 2024. The county figures come from the SOI county file (`22incyallnoagi.csv`), also tax year 2022.
- **ACS households:** the state target is the sum of the state's county estimates, so each county target equals the county's own published estimate. When the build counts the file's households for these targets, it leaves out ACS group-quarters records.
- **Population:** for each county, we add up the population it shares with each PUMA in the PUMA ladder. We then scale the state's counties so that they add up to the state's ladder population.

So each concept's county targets add up to their state target; the largest county takes any rounding remainder.

The SOI county file has no row for Kalawao County, Hawaii (15005), and neither does the county facts file the run read. The build's code cites the IRS guide to the county file, which says the IRS leaves out items with fewer than 20 returns for a county. For the two SOI concepts, one target covers Kalawao and Maui (15009) together, with Maui's published share. Kalawao keeps its own population and household targets.

The run dropped no target. It ran with a setting that drops a county SOI target when every household in that county has a zero value for it, and no target met that rule.

### How we weight the targets

For each target, the solver computes a scaled error: (estimate − target) ÷ max(|target|, 1). Every trained target's value is 27 or more, so the scaled error equals the relative error. The loss is the weighted average of the absolute scaled errors, with each one capped at 10 (an error of 1,000%). One cap covers every target at every level. At the design weights, 20 targets had an error over 10; at the final weights, 9 did.

A target's loss weight sets how much its error counts in that average. It comes from five steps:

1. Each target is a count (people, households, returns, enrollees) or a dollar amount.
2. Its base weight is the square root of its value, divided by the average square root over targets of the same kind. A larger target weighs more, but less than in proportion to its size.
3. Some targets share a budget. The county targets of one concept in one state together weigh as much as the largest of them alone. The district population targets of one state share a budget the same way. Each state target has its own budget.
4. Counts and dollar amounts each get half of the total weight.
5. The code rescales the weights to an average of 1. No group of targets got an extra multiplier in this run.

Two shares describe each group of targets. Its loss weight share is its share of the summed loss weights, which the build fixes before the solve. Its realized loss share is its share of the loss at the final weights, where the loss comes to 0.0159:

| Targets | Loss weight share | Realized loss share |
|---|---:|---:|
| State SOI | 83.50% | 64.08% |
| All state and district targets, state SOI included | 94.01% | 72.91% |
| All county targets | 5.99% | 27.09% |

### Design weights and the solver

The solve starts from the design weights. Staging builds them when it pools the ACS and donor records. It takes each ACS household's housing-unit weight (WGTP; a group-quarters record takes its person weight) and each donor household's weight in the donor file. It then scales each group to half of the donor file's household weight total: 61.27 million each, 122.5 million in all. These scaled weights are the design weights. The solve keeps their total fixed. The weight cap, every weight ratio below and the loss at the design weights all use them.

The solver, Adam (a standard gradient method), works on the logarithm of each weight, so every weight stays positive. It takes 800 steps at a learning rate of 0.02, in two runs of 400 steps, with seed 0. Each step uses every trained target and every household record. The second run starts from the first run's weights, with a fresh optimizer state. No penalty pulls weights toward their design values. The file keeps the weights from the last step.

The loss comes to 0.173 at the design weights, 0.0208 after the first 400 steps and 0.0159 after all 800.

### The weight cap

No household record's final weight may exceed 5 times its design weight. A donor household's design weight equals exactly half its weight in the donor file, so its limit equals 2.5 times that weight. The cap acts on each record as a hard limit, not a penalty. After every step, the solver lowers any weight over its limit to the limit. It then rescales all weights by one factor to the fixed total and applies the limit again. Both runs measure the limit against the design weights.

A rescale factor below 1 leaves the records at the limit under it, at 5 times the factor. The largest final ratio is 4.9297, not 5, and 44,083 records end within 0.01% of it. So the last step's factor came to about 0.986 (4.9297 ÷ 5). The cap has no lower counterpart: 1,246,252 records (78.5% of 1,586,908) end with less than 1% of their design weight.

### The holdout

We hold out county targets in whole holdout groups. A holdout group covers one state's county targets for one concept: SOI returns, SOI AGI or ACS households. The 153 holdout groups cover 9,430 county targets. For each group, the build hashes the group's name with a fixed key, `acs-local-county-v1`, using SHA-256, and turns the result into a number from 0 to 1. It leaves the group out of the solve when that number falls under 0.1. The draw depends only on the group's name and the key, never on the other groups. County population targets belong to no holdout group and always train.

The code gives its reason for holding out whole groups: "to prevent sibling leakage". A group's county targets add up to its state target, which trains. If the group's other counties trained, the state target minus their sum would give away the held-out county.

The run held out 18 of the 153 holdout groups: 1,052 of the 9,430 eligible county targets (11.2%), or 8.4% of all 12,574 county targets. The 1,052 break down into 441 SOI return-count targets, 297 SOI AGI targets and 314 ACS household targets. The run held out no state or district target.

The solver never sees a held-out target; after the solve, the build scores them (see "How we test the file"). The ACS county tables for earnings and income (B19061, B19051, B19025 and B19001) and for median household income (B19013) never enter the loss either. After calibration, the build scores the first four as checks and reports B19013 as a diagnostic only.

## How we test the file

We test the 2024 county file in four ways: county targets held out of the calibration, ACS county tables the calibration never uses, a reliability grade for each county, and 23 gates. For the results, see the PoC report.

### Held-out county targets

We hold county targets out of the calibration in whole holdout groups. We give the rule under "The holdout" in the calibration section. This run held out 18 of the 153 holdout groups: 1,052 of the 9,430 eligible county targets (441 SOI return counts, 297 SOI AGI and 314 ACS households). County population targets belong to no holdout group, and the run held out no state or district target, so all of these trained.

For each held-out target, the error is |estimate − target| / max(|target|, 1), where the estimate is the file's weighted total for that county and concept. A target counts as within 10% when its error is 0.10 or less. We compare the file with a pro-rata baseline: the state target times the county's share of the state's population in the PUMA ladder. The build also scores the design weights the same way.

### ACS county tables the calibration never uses

Four ACS 2020–2024 5-year county tables never enter the calibration's targets, its loss or its weights:

- B19061: aggregate household earnings;
- B19051: households with earnings;
- B19025: aggregate household income;
- B19001: households in 16 income brackets.

After calibration, we score all four tables in all 3,144 counties. We also compare each county's weighted median household income in the file with ACS table B19013. That comparison serves as a diagnostic only and never enters a grade.

These checks test how the file splits each state's total among its counties. Each county's target splits the file's own state total, under the final weights, in the ACS county proportions:

```
target(county) = file's state total × ACS county value ÷ sum of ACS values over the state's counties
```

So if the file's earnings or income sum differs from the ACS concept by the same proportion in every county of a state, that difference cancels. The pro-rata baseline splits the same state total by population: the file's state total × county population ÷ state population. It uses the same PUMA ladder county populations as the held-out targets' baseline.

For B19061, B19051 and B19025, the error is |estimate − target| / max(|target|, 1). For B19001, the error adds |estimate − target| over the 16 brackets and divides the sum by max(sum of the county's bracket targets, 1). For each table, the build also records the share of counties where the file's error falls below the baseline's.

The build computes household earnings and income sums from stored inputs, not from model outputs, and uses them only for these checks and the B19013 comparison. For each household, it adds up its people's values, and a missing value counts as zero. The code describes the ACS household tables as leaving out group quarters, both institutional and noninstitutional, so the checks leave out ACS group-quarters records too; every donor household counts.

- **Earnings for the checks:** wages, self-employment income and farm income, plus partnership self-employment earnings on donor records.
- **Household income for the checks:** wages, self-employment and farm income, plus the stored income inputs. Those inputs cover interest, dividends, rent, estate income, Social Security, SSI, pensions, IRA distributions, unemployment compensation, veterans' benefits, workers' compensation, alimony, child support and miscellaneous income.
- On donor records, the build also adds partnership and S-corporation income to the income sum. On ACS records it leaves both out; the code's stated reason is that the ACS self-employment answer "already counts partnership income". The income sum also leaves out donor partnership self-employment earnings, which partnership income mostly holds already.
- In this run, S-corporation income is zero, so it adds nothing. TANF receipt (`receives_tanf`) appears in the file only as a yes/no flag, and the stored inputs have no column for TANF or public assistance dollars. So the income sum includes neither.
- The build marks the two income tables, B19025 and B19001, `approximate: true`. The code describes the income sum as an approximation of the ACS household income concept.

### The reliability grade

The build writes each county's provisional grade to `county_reliability.json` and `county_reliability.csv`, and records the rule in the JSON:

```
"provisional": true,
"A": "source ESS >=400 and max error <=0.10",
"B": "source ESS >=100 and max error <=0.20",
"C": "source ESS >=30 and max error <=0.50",
"D": "otherwise with holdout or validation evidence",
"U": "no county holdout or validation evidence (no household support) or source household identity unavailable",
"error": "maximum absolute relative error over the county's training and holdout targets and its validation families (the income distribution as its bracket L1 error); denominator max(abs(target),1)"
```

- **Effective sample size (ESS).** The number of equally weighted households that would carry the same information as the actual weights: (sum of the weights)² ÷ (sum of the squared weights). For the grade's "source ESS" (`kish_ess`), the build first adds up the final weights of rows that share one source household: "final weights summed within county by (household_spine, household_source_id), excluding ACS group quarters". `household_spine` marks a row as ACS (`acs_2024_1yr`) or donor (`asec_puf`); `household_source_id` holds the ID of the household the row came from.
- **Support.** The build counts support as in "Counties without households": ACS group-quarters records count toward neither support nor ESS.
- **Copies.** Staging sets each ACS household's source ID to its ACS household ID from before pooling, so no two ACS rows share one. A donor household can appear twice under one source ID, as its ASEC record and as its tax-detail copy. The file's 57,240 donor household records hold 47,277 distinct source households. The source ESS counts a household and its copy as one only when both sit in the same county. Across all counties, the 1,405,648 supporting rows count as 1,404,416 source households, and 240 counties hold at least one such pair.
- **Row ESS.** The build also reports `kish_ess_rows`, which counts each row on its own, copies included, and likewise leaves out ACS group quarters. The grade does not use it.
- **Error.** The largest error over the county's training targets, its held-out targets and the four ACS tables, with B19001 as its bracket error. In the CSV, `grade_driver` holds the target or table that set it.
- **Evidence.** A county gets a grade from A to D only if it has a supporting household and either a held-out target or a scored ACS table.

The code returns U when a county lacks that evidence or its source household identity. Otherwise it tries A, then B, then C, and returns the first grade whose two limits the county meets; failing all three, it returns D. At a fixed error, a higher ESS therefore cannot lower a grade. In the CSV, `grade_provisional` reads true for every county.

### Automated gates

The build ran 23 automated checks, which the build calls gates. Seventeen stop the build when they fail, and all 17 passed:

- **Geography (1):** each household's PUMA, congressional district and county must be present and must agree with its state. New York City must keep a plausible share of the weight.
- **Household inputs (12):** the build assigns some inputs to ACS records: hours worked, program take-up, immigration status, disability, transferred income, SSI disability criteria, SSI and Medicaid take-up, and SNAP receipt. Each must be complete and must vary. Where a gate checks them against native ACS answers or plausibility ranges, they must agree. Every input left at a model default must appear on a reviewed list. ACS self-employment income must count once. Occupation, industry and worked-last-year inputs must be complete.
- **Calibration (1):** the final loss must fall under 10, the same setting that caps each target's scaled error in the loss. It must also fall under the loss at the start of the second 400-step run (0.0208). The final weights must sum to within 0.1% of the design weights' sum.
- **Model readiness (3):** the file stores no input that policyengine-us 2.2.1 lacks, outside a reviewed list. Every SPM unit the model measures has an adult, so the SPM calculation cannot stop on a unit without one. The build wrote its consumer export, and the `qa` stage loaded the calibrated file as stored, with no added fills.

The other six gates record evidence and do not stop the build. Five of them record what they measure and have no pass condition:

- held-out district targets (none in this run);
- the weight share and ESS of ACS and donor households;
- a note that staging checks the donor inputs;
- the households and weight of ACS and donor records;
- the `qa` stage's load and its SSI and SNAP figures.

The sixth flags congressional districts whose ESS under the final weights fell under a quarter of their ESS under the design weights, or under 15. It can be set to stop the build, and this run did not set it. In this run, 271 of the 436 districts fell under the quarter floor and 155 under 15, 275 districts in all, so the gate recorded that its criteria were not met.

The `qa` stage loaded the calibrated file with Microcosm's own loader, not policyengine-us's dataset loader. It then ran policyengine-us 2.2.1 for 2024, with no reform, in 80 chunks of 20,000 households, for eight variables: SSI, six SNAP-related variables and the state code. For a separate test on 18 November that loaded the file with policyengine-us's standard loader, see the PoC report.

## Appendix: build identifiers and settings

| Item | Value |
|---|---|
| Build | `populace-us-2024-buildo-acs-local-f8b1039f00-20261118T011728Z`, 17–18 November 2026 |
| File | `populace_us_2024_acs_local.h5`, 10,637,811,669 bytes, SHA-256 `82314a3b5a6a95762faf6eb4f5b499ce56608ad6a39a27b75e11c75b4945bd3c` |
| Code | Microcosm, commit `72b7072bf` (staging) and commit `f8b1039f0` (the release build) |
| Model | policyengine-us 2.2.1, policyengine-core 3.32.5 |
| Donor file | Release `populace-us-2024-spm-receipts-20260923` (SHA-256 `00946972…`), in the copy we patched (SHA-256 `438f47f4…`) |
| National build behind the donor file | `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`, Microcosm commit `cae8640` |
| Target feed | `consumer_facts_us_41b2fb2.jsonl` (SHA-256 `1d45a89c…`) |
| County facts | `consumer_facts_us_county_3e9d3a0.jsonl` (SHA-256 `67b0998a…`) |
| Household location | The legacy rule: county and district draws within each PUMA, as in "How households get their counties" |
| Imputation | Quantile regression forests with 32 trees; seed 0 |
| Minimum support | Three households for each county without one; ranking key `acs-local-county-support-v1` |
| Holdout | Whole holdout groups, held out when their hash falls under 0.1 (`--county-holdout-fraction 0.1`); key `acs-local-county-v1` |
| Empty county SOI targets | Dropped when every household in the county has a zero value (`--county-drop-empty-soi-rows`); none was dropped |
| Loss | Mean of the absolute scaled errors, weighted by loss weight; each error capped at 10 |
| Weight cap | 5 times the design weight (`--county-max-weight-ratio 5`) |
| Solver | Adam on log weights; 800 steps in two runs of 400; learning rate 0.02; seed 0 |
| District effective-sample gate | Records evidence only; the build was not run with `--district-ess-gate-blocking` |

---

Sources: the table in "Data sources and years" lists every source with its vintage and its use in the build.
