# NZ public calibration-target inventory (v0, 2026-07-06)

Public New Zealand data margins for **microcosm-nz** ([epic #343](https://github.com/PolicyEngine/microcosm/issues/343)) — calibrating donor microdata to NZ population/administrative margins, running tax-benefit microsimulation (rules from [rulespec-nz](https://github.com/TheAxiomFoundation/rulespec-nz)), and estimating **dollar** benefit take-up gaps.

**Scope.** Public calibration targets only. For each source: dataset/table, URL (landing + direct download where confirmable), variables, geographic granularity, reference period, cadence, format, licence — and which targets enable **dollar-level** (not just count) calibration.

**Verification standard.** Every entry was checked against the live source on 2026-07-06:

- **[FETCH-OK]** — file/repo/PDF retrieved and contents read, or direct-download URL returned HTTP 200 with correct content-type.
- **[SNIPPET]** — content confirmed via search extraction of the source's own pages; landing page is a JS/bot-gated SPA but the URL is real and resolves in a browser.
- **[UNCONFIRMED]** — referenced but not verified; a lead, not a fact.

**Portal / bot-wall caveat.** Three NZ government hosts return JS challenges or WAF blocks to scripted clients: `stats.govt.nz` and its DataFinder/ADE SPAs (Imperva/Incapsula), `treasury.govt.nz` HTML landing pages (Cloudflare), and `catalogue.data.govt.nz` CKAN (Imperva). Their **direct file CDNs are not gated** — `ird.govt.nz`, `msd.govt.nz`, `tenancy.govt.nz`, Treasury's `/sites/default/files/…`, and Stats NZ `/assets/Uploads/…` all returned HTTP 200 to plain requests. Prefer file CDNs for pipelines.

---

## Summary — the ~15 highest-value targets for v1

Ranked by calibration value for a dollar-level take-up estimate. **$** = enables dollar-level calibration; **#** = count/structure margin.

| Rank | Target | Source | Type | Why it matters |
|---|---|---|---|---|
| 1 | **Individual taxable income distribution by band** (+ age × income band) | IRD *Taxable income distribution of individuals 2025* | $ / # | The income backbone. Admin-based, full-population, bands to $200k+, 2001–2024, plus age×band. [FETCH-OK] |
| 2 | **Benefit expenditure by program, in dollars** | MSD *Annual Report 2025* — *Benefits or Related Expenses* schedule | $ | The dollar control totals: NZS $23.19B, Jobseeker+Emergency $4.64B, Accommodation Assistance $2.23B, SLP $2.67B, Sole Parent $2.26B, Hardship $0.75B, WEP $0.56B (2024/25). Explicit CC BY 4.0. [FETCH-OK] |
| 3 | **Benefit recipient counts by benefit × age × ethnicity × duration × region** | MSD *Quarterly Benefit Fact Sheets* (Mar 2026) | # | Recipient marginals for every main benefit + supplementary (AS, DA, TAS). National→region→TA→service-centre. Count-only. [FETCH-OK] |
| 4 | **Working for Families: expenditure, families, average claim by credit type** | IRD *Working for Families statistics Sept 2025* | $ / # | WfF dollar + count margins by FTC/IWTC/MFTC, family size (2020+), recipient income distribution (2020+). [FETCH-OK] |
| 5 | **Modelled dollar amounts per program by income position** | Treasury *DistributionExplorer* data CSVs (HES24 + BEFU25, TY25–29) | $ / # | TAWA-modelled $ per program (AS, Core Benefits, FTC, IWTC, MFTC, BestStart, WFF, WEP, FamilyBoost, NZ Super, Income Tax) × Household/Family/Individual × income quantile/band, with weighted population per cell. The **full-entitlement denominator** for the take-up gap. MIT. [FETCH-OK] |
| 6 | **Age × sex × region / territorial authority population** | Stats NZ *Subnational population estimates at 30 June 2025* (2023-base) | # | Primary demographic reweighting frame: 16 regions / 67 TAs / 21 Auckland boards by age×sex. [SNIPPET; prior-quarter direct xlsx FETCH-OK] |
| 7 | **Accommodation Supplement recipients + average payment** | MSD Fact Sheets (counts, by W&I region/TA) + Annual Report (Accommodation Assistance $2.23B) | $ / # | AS is the take-up wedge (~44% receipt among potentially eligible). Count × implied average payment = AS dollar base. [FETCH-OK] |
| 8 | **Household composition / family type** | Stats NZ 2023 Census — families, households and housing | # | Household/family-type marginals (1-family 67.7%; 1,780,527 total households). [SNIPPET] + ArcGIS/DataFinder [FETCH-OK for hub] |
| 9 | **Market rent by TA / SA2 × bedrooms** | MBIE *Tenancy Bond Data* (rental bond CSVs + Market Rent API) | $ | Geometric-mean & quartile rents by TA/SA2/bedroom, monthly, from 1993 — the dollar input AS entitlement is computed against. CC BY 3.0 NZ. [FETCH-OK] |
| 10 | **Program-level fiscal control totals (admin dollars)** | Treasury *AN 24/01* `fiscal_totals.csv` | $ | Documented national $ totals per program (AS $1.531B, FTC $2.008B, Best Start $48.6M for TY2018/19), source-noted. CC0. [FETCH-OK] |
| 11 | **Tenure (own/rent/family trust) by area** | Stats NZ 2023 Census — tenure of household | # | Owner/renter split (66% owned; 604,884 renting) by geography — the renter population eligible for AS. [SNIPPET via DataInfo+] |
| 12 | **Total personal income bands (census)** | Stats NZ 2023 Census — total personal income | # | Census income bands ($1–10k … $200k+) cross-tabbable with age/sex/region/ethnicity at SA2. [SNIPPET; ArcGIS/DataFinder FETCH-OK] |
| 13 | **Māori / Pacific / Asian population × age × region** | Stats NZ 2023 Census — ethnic group counts | # | Equity-cut marginals (Māori descent 978,246). Enables take-up gap by ethnicity. [SNIPPET] + Te Whata (iwi) |
| 14 | **NZ Superannuation demographics** | MSD Fact Sheets (NZS/VP) + Annual Report (941,511 recipients; $23.19B) | $ / # | Largest transfer by dollars; near-universal but needs the age×region recipient frame. [FETCH-OK] |
| 15 | **Minimum wage series + average weekly earnings** | MBIE minimum wage ($23.95 from Apr 2026) + Stats NZ QES/LEED | $ | Anchors the bottom of the earnings distribution and the national/TA earnings margin. [FETCH-OK min wage; SNIPPET QES/LEED] |

**Dollar-level calibration design (numerator/denominator):**
- **Denominator (full-entitlement modelled $):** Treasury DistributionExplorer CSVs (#5) — dollars per program by income position, assuming full take-up.
- **Numerator (actual $ paid):** MSD Annual Report expenditure schedule (#2) and AN24/01 fiscal totals (#10); IRD WfF expenditure (#4) for in-work credits.
- **Average payment** = program expenditure (#2) ÷ recipient count (#3) distributes dollars across the calibrated population.
- **Take-up gap** = actual paid vs full-entitlement modelled, anchored empirically by the WfF ~87% and AS ~44% take-up studies (M5/M6 below).

---

## 1. Stats NZ

**Portal state (2026-07-06).** NZ.Stat retired 13 Sep 2024, replaced by **Aotearoa Data Explorer (ADE)**. **Infoshare still live** but being replaced by ADE Time Series Expansion (alpha Jul 2026, beta Sep 2026, go-live **April 2027**). Today: census + subnational estimates → ADE; time series (QES, national estimates) → Infoshare; spatial census → ArcGIS hub + DataFinder. Licence throughout: **CC BY 4.0** (NZGOAL default).

| # | Dataset | Landing / access | Variables | Granularity | Period | Cadence | Status |
|---|---|---|---|---|---|---|---|
| S1 | 2023 Census — totals by topic, individuals | `https://2023census-statsnz.hub.arcgis.com/maps/29a82d5a0ea24a3880219bcb3df126dc` (CSV/GeoJSON/Shapefile + feature-service API) | Age, sex, personal income band, ethnicity, occupation, work status; 2013/2018/2023 | **SA2** (aggregable to TA/region) | 7 Mar 2023 | Decennial | [FETCH-OK] |
| S2 | 2023 Census — totals by topic, households | `https://2023census-statsnz.hub.arcgis.com/datasets/StatsNZ::2023-census-totals-by-topic-for-households-by-sa2-1/about?layer=1` | Household composition, usual residents, bedrooms, tenure, income sources | SA2 | 2023 | Decennial | [FETCH-OK] |
| S3 | 2023 Census — families and extended families | `https://maps-by-statsnz.hub.arcgis.com/datasets/2023-census-totals-by-topic-for-families-and-extended-families-by-sa2-1` | Family type (couple / couple+children / one-parent), family income sources | SA2 | 2023 | Decennial | [FETCH-OK] |
| S4 | Census population/dwelling by age × region/TA (DataFinder) | `https://datafinder.stats.govt.nz/data/` (layers 117618 regional, 117593 TA/board; Koordinates export + WFS, API key) | Population by age group & sex, 2018 & 2023 | Region; TA/board | 2023 | Decennial | [SNIPPET] |
| S6 | Census concept metadata (DataInfo+) | Income: `https://datainfoplus.stats.govt.nz/item/nz.govt.stats/69a79e8e-0338-48d5-a74b-ce08eeef2250` | **2023 income bands: $1–$10,000 … $150,001–$200,000, $200,001+**; equivalised income; tenure | — | 2023 | — | [FETCH-OK via search] |
| S7 | Subnational population estimates at 30 Jun 2025 (2023-base) | `https://www.stats.govt.nz/information-releases/subnational-population-estimates-at-30-june-2025/` (+ ADE; direct-xlsx pattern confirmed for Jun-2024 vintage) | ERP by **age × sex** | **16 regions / 67 TAs / 21 boards** | 30 Jun 2025 | Annual | [SNIPPET/FETCH-OK] |
| S8 | National population estimates at 30 Jun 2025 | `https://www.stats.govt.nz/information-releases/national-population-estimates-at-30-june-2025/` + Infoshare (DPE) | ERP by **single year of age**, sex | National | 30 Jun 2025 | Quarterly/annual | [SNIPPET] |
| S9 | Subnational population projections 2023-base–2053 | `https://www.stats.govt.nz/information-releases/subnational-population-projections-2023base-2053/` | Projected population age × sex | Region, TA | to 2053 | ~5-yearly | [SNIPPET] |
| S10 | HES income & housing costs | `https://www.stats.govt.nz/information-releases/household-income-and-housing-cost-statistics-year-ended-june-2023/` | Household income by source, housing costs, deciles; expenditure every 3 yrs | National + limited regional | YE Jun 2023 (2023/24 collected, ~19,100 hh) | Annual | [SNIPPET] |
| S11 | Microdata access (CURF / IDI Data Lab / SURF) | `https://www.stats.govt.nz/integrated-data/apply-to-use-microdata-for-research/` | Unit records (HES, HLFS…). CURFs free since 2014, usable outside Data Lab; IDI = secure Data Lab | Unit record | various | — | [SNIPPET] |
| S12 | QES average weekly/hourly earnings | Infoshare (Work income and spending) | Avg weekly earnings ($1,716 Mar 2026 q; FTE), by industry/sector | **National** | Mar 2026 q | Quarterly | [SNIPPET] |
| S13 | LEED sub-national earnings | ADE — LEED quarterly | Filled jobs, **mean & median earnings** by **region and TA**, ethnicity | Region + TA | Mar 2025 q | Quarterly | [SNIPPET] |

Notes: **ArcGIS hub (S1–S3) is the best machine-readable census route** (not behind the SPA wall). **Family/household *projections* are stale (2013-base)** — use census counts for structure. Census income is banded and individual — no joint income.

## 2. IRD

All statistical spreadsheets on the un-gated `ird.govt.nz` CDN (HTTP 200 confirmed); mirrored on data.govt.nz under **CC BY 4.0**.

| # | Dataset | Direct download (verified) | Variables | Period | Cadence | Status |
|---|---|---|---|---|---|---|
| I1 | **Taxable income distribution of individuals 2025** | `https://www.ird.govt.nz/-/media/project/ir/home/documents/about-us/tax-statistics---current/revenue-and-refunds/tax-on-taxable-income/tax-on-taxable-income/taxable-income-distribution-of-individuals-2025.xlsx` | Individuals & total taxable income by band; tax by band; **Tab 4 = age × income band (2024)** | 2001–2024 TY (upd. Mar 2026) | Annual | [FETCH-OK] |
| I2 | Wage and salary distributions | landing: `/about-us/tax-statistics/revenue-refunds/wage-salary-distributions/wage-and-salary-statistics-datasets` | Wage/salary income by band | annual series | Annual | [SNIPPET] |
| I3 | **Working for Families statistics — Sept 2025** | `https://www.ird.govt.nz/-/media/project/ir/home/documents/about-us/tax-statistics---current/social-policy/wff-stats/working-for-families-statistics---sept-2025.xlsx` | **Expenditure by credit type; families; average claim** (FTC/IWTC/MFTC); families by size (2020+); **recipient income distribution (2020+)** | 2001–2024 TY | Annual | [FETCH-OK] |
| I4 | Student loan scheme statistics | `https://www.ird.govt.nz/-/media/project/ir/home/documents/about-us/tax-statistics---current/student-loan/student-loan-statistics/student-loan-statistics.xlsx` | Borrowers ($16.8B Mar 2026; 492,838 NZ-based Dec 2025), balances, repayments | to 31 Mar 2026 | Quarterly | [FETCH-OK] |
| I6 | KiwiSaver statistics | landing: `/about-us/tax-statistics/kiwisaver/datasets` | Members, contributions, withdrawals | monthly + annual | Monthly | [SNIPPET] |

Notes: **I1, I3, I4 are the load-bearing IRD targets.** IRD income tables are **individual-level** — joint/couple income unpublished. **FamilyBoost: no public statistics dataset confirmed** — only a modelled `Value_Type` in Treasury DistributionExplorer [UNCONFIRMED — re-check].

## 3. MSD

**Critical structural finding:** the **quarterly Benefit Fact Sheets are COUNT-ONLY** (every sheet opened; only hardship reports dollars). **Dollar-level data comes from the Annual Report** expenditure schedule (explicit CC BY 4.0).

| # | Dataset | Direct download (verified) | Variables | Period | Cadence | $/# | Status |
|---|---|---|---|---|---|---|---|
| M1 | Quarterly Benefit Fact Sheets (10 Excel + PDF) | e.g. `https://www.msd.govt.nz/documents/about-msd-and-our-work/publications-resources/statistics/benefit/2026/quarterly-benefit-fact-sheets-national-benefit-tables-march-2026.xlsx` | Recipients by benefit, gender, age, ethnicity, duration, child age | Mar 2026 q (5-yr series) | Quarterly | # | [FETCH-OK] |
| M1d | Fact Sheets — W&I supplementary (AS, DA, TAS) | `.../quarterly-benefit-fact-sheets-w-i-supp-tables-march-2026.xlsx` | AS, Disability Allowance, TAS recipient counts | Mar 2021→Mar 2026 | Quarterly | # | [FETCH-OK] |
| M2 | **Annual Report 2025** (G.60) | `https://www.msd.govt.nz/documents/about-msd-and-our-work/publications-resources/corporate/annual-report/2025/msd-annual-report-2025.pdf` | **Benefits or Related Expenses ($000 by benefit, 2023/24 + 2024/25)**; NZS 941,511 recipients | FY 2024/25 | Annual | **$** | [FETCH-OK] |
| M3 | Benefit System Report 2025 | `https://www.msd.govt.nz/documents/about-msd-and-our-work/publications-resources/research/benefit-system/benefit-system-report-2025.pdf` | Main-benefit + supplementary receipt (717,900 supplementary; **AS ~381,600 Jun 2025**) | to Jun 2025 | Annual | mostly # | [FETCH-OK] |
| M4 | Hardship assistance | in Fact Sheet supplementary XLSX | Grants **and dollars** by type & reason | Mar 2026 q | Quarterly | **both** | [FETCH-OK] |
| M5 | **WfF take-up** — McLeod & Wilson 2022 | `https://thehub.sia.govt.nz/sitemap/estimates-of-working-for-families-eligibility-and-take-up-rates-2007-2020` | FTC/IWTC take-up 2007–2020 (IDI): **~87% overall 2020; 75–85% non-benefit families** | 2007–2020 | one-off | rates | [SNIPPET] |
| M6 | **AS take-up** — Income Support Survey pack 7 | `https://www.msd.govt.nz/documents/about-msd-and-our-work/work-programmes/income-support-survey/findings-pack-7-accommodation-supplement-self-reported-take-up-and-other-findings.pdf` | Self-reported AS take-up among potentially eligible: **43.9% receiving**; by benefit status, ethnicity, age; non-receipt reasons | 2022 survey | one-off | **take-up anchor** | [FETCH-OK] |
| M7 | data.govt.nz Fact Sheets mirror (CSV) | `https://catalogue.data.govt.nz/organization/ministry-of-social-development` | same counts, CSV | back-catalogue | Quarterly | # | [SNIPPET] |
| M8 | BEFU 2025 benefit forecasts | landing: `/statistics/befu/budget-economic-and-fiscal-update-2025.html` | recipient forecasts (JS, SPS, SLP) | Mar 2025→Jun 2029 | twice-yearly | # | [SNIPPET] |

**Verbatim dollar control totals (M2, 2024/25 actual, $000):** Accommodation Assistance **2,232,026**; Disability Assistance 491,784; Hardship Assistance 754,599; Jobseeker Support & Emergency **4,640,562**; NZ Superannuation **23,191,199**; Orphan's/UCB 402,311; Sole Parent Support **2,255,092**; Student Allowances 573,646; Supported Living Payment **2,668,450**; Other 1,057,867; **Total 38,267,536**. Appropriations additionally split: Childcare Assistance 167,204; Emergency Housing 74,407; **Winter Energy Payment 561,678**; Veterans' Pension 131,128; Youth Payment 78,115; TIA 15,699.

**No standing AS-by-Area-1–4 table exists** — AS is published by W&I region/TA (count-only); Area-level splits surface only via OIA/third-party analysis. Plan a TA→Area crosswalk (H4) or an OIA.

## 4. Treasury

Public analysis ships as **code + data on GitHub** and interactive tools. **Treasury outputs assume full take-up** — they are the **denominator** of the take-up gap, not actuals.

| # | Dataset | Access | Contents | Period | Licence | Status |
|---|---|---|---|---|---|---|
| T1 | **DistributionExplorer data CSVs** (TAWA outputs) | `https://github.com/Treasury-Analytics-and-Insights/DistributionExplorer` → `app/data/DE_HES24_BEFU25_TY{25..29}_SQ.csv` (~3.9MB each) | `Value_Type` ∈ {AS, Core Benefits, FTC, IWTC, MFTC, BestStart, WFF, WEP, **FamilyBoost**, NZ Super, Income Tax, ACC Levy, earnings, housing costs, disposable/AHC income}; `Value` ($) + `Population` per cell; Household/Family/Individual × income quantile/band | TY2025–29 (HES24+BEFU25) | **MIT** | [FETCH-OK] |
| T3 | **AN 25/01** — EMTRs ("The Cost of Working More") | `https://github.com/Treasury-Analytics-and-Insights/AN25-01` (code + `TAR381_emtr_B24_no_raw_info.xlsx`); PDF `https://treasury.govt.nz/sites/default/files/2025-01/an25-01.pdf` | EMTR distribution by family type; **94% of individuals below 50% EMTR; ~13% couple-parent and ~30% sole-parent families above**; student-loan repayments excluded | TY25 | **no LICENSE file** (note PDF Crown CC BY) | [FETCH-OK] |
| T4 | **AN 24/01** fiscal incidence | `https://github.com/Treasury-Analytics-and-Insights/analytical-note-24-01-effects-of-taxes-and-benefits` → `data/fiscal_totals.csv` | **Program-level admin $ totals** (AS 1,531,000,000; FTC 2,007,807,000; Best Start 48,610,000; TY2018/19), source-noted; incidence by decile | TY2018/19 | **CC0-1.0** | [FETCH-OK] |
| T5 | **IncomeExplorer** (EMTR calculator) | `https://github.com/Treasury-Analytics-and-Insights/IncomeExplorer` (live: `treasury-analytics-and-insights.github.io/IncomeExplorer/`) | Household EMTR/RR/PTR calculator; parameter scripts | param. by TY; upd. 2025-08 | **MIT** | [FETCH-OK] |
| T6 | **emtr** (Python port) | `https://github.com/Treasury-Analytics-and-Insights/emtr` | `emtr.py`, `taxabate.py`; **`parameters/TY14…TY27_BEFU23.yaml`** (full abatement scales); `test/ref/emtr_output_{1..6}.csv` (28-col worked outputs) | params BEFU23, TY14–27; dormant 2024-02 | **no LICENSE file** | [FETCH-OK] |
| T7 | TAWA methodology report | `https://treasury.govt.nz/sites/default/files/2024-08/tawa-model-methodology.pdf` | Static non-behavioural microsim; HES linked to IDI; reweighted to Stats NZ demographics + MSD forecasts; benchmarked to fiscal totals | Aug 2024 | CC BY 4.0 | [FETCH-OK] |
| T8 | Tax outturn data | landing gated; xlsx pattern `treasury.govt.nz/system/files/YYYY-MM/tax-history-<mon><yy>.xlsx` | Monthly receipts + accrual by tax type, to 1990 | latest Feb 2026 | Crown | [UNCONFIRMED filename] |

Also: `github.com/Treasury-Analytics-and-Insights/Calibration` (MIT, 2018) — "calibrate household surveys to known administrative totals"; directly relevant methodology.

## 5. Housing (HUD dashboard, MBIE bond data, AS areas)

| # | Dataset | Access | Variables | Period | Licence | $/# | Status |
|---|---|---|---|---|---|---|---|
| H1 | **Government Housing Dashboard data download** | `https://www.hud.govt.nz/assets/Uploads/Documents/Housing-Dashboard/Housing-dashboard-data-download-May-2026-v2.xlsx` (~14MB; monthly filename pattern) | Social-housing register (Priority A/B, placements), stock, emergency/transitional, **AS tab**, Key Stats by TLA | May 2026 (series to Jun 2017) | Crown/CC BY 4.0 | # | [FETCH-OK] |
| H3 | **MBIE tenancy bond data** | `https://www.tenancy.govt.nz/assets/Uploads/Tenancy/Rental-bond-data/detailed-monthly-tla-tenancy-v2.csv` (2.2MB) · `detailed-monthly-region-tenancy-v2.csv` · `detailed-quarterly-tenancy-2020-to-2026.csv` (11MB, **SA2**) · `detailed-quarterly-tenancy-93-to-19.zip` | Bonds lodged, active bonds, **mean/geometric-mean/quartile rents** by bedrooms & dwelling type | Feb 1993–Apr 2026 | **CC BY 3.0 NZ** | **$** | [FETCH-OK all 4] |
| H3b | Market Rent API | `https://portal.api.business.govt.nz/api/market-rent` (`/statistics`; API key) | JSON/CSV rents by region/TA/**SA2**, bedrooms, 6-mo rolling | monthly | CC BY 3.0 NZ | $ | [FETCH-OK] |
| H4 | **AS Area 1–4 definitions** | `https://www.workandincome.govt.nz/map/deskfile/extra-help-information/accommodation-supplement-tables/definitions-of-areas.html` (also Social Security Regulations 2018) | locality→Area mapping | eff. 1 Apr 2018 | Crown | ref | [FETCH-OK] |
| H4b | AS maximum weekly rates by area | `https://www.workandincome.govt.nz/products/a-z-benefits/accommodation-supplement.html` | max AS by area × household type | Apr 2026 | Crown | **$** | [FETCH-OK] |
| H4c | **2027 AS area redraw** (Budget 2025) | `https://www.msd.govt.nz/about-msd-and-our-work/newsroom/budget/2025/factsheets/changes-to-the-accommodation-supplement-through-budget-2025.html` | homeowner entry 30%→40%; **boundaries redrawn, implemented April 2027**; ~45 TAs affected | Budget 2025 | Crown | — | [SNIPPET] |
| H5 | IRRS (income-related rent subsidy) $ | Kāinga Ora AR / Vote Housing | public-housing rent subsidy $ + recipients | annual | Crown | $ | [UNCONFIRMED] |

## 6. Other

- **Minimum wage** (Employment NZ, [FETCH-OK]): **from 1 Apr 2026 adult $23.95/hr**, starting-out/training $19.16 (2025: $23.50/$18.80); history to 1997. `https://www.employment.govt.nz/pay-and-hours/pay-and-wages/minimum-wage/minimum-wage-rates-and-types`
- **QES earnings**: avg weekly $1,716 (Mar 2026 q), national by industry — Infoshare. **LEED**: mean/median earnings by TA, quarterly — ADE.
- **NZS demographics**: MSD NZS/VP fact sheet xlsx [FETCH-OK] + Annual Report.
- **Ethnicity margins**: Census ethnic-group counts by age × region (Māori descent 978,246); Te Whata for iwi.
- **FamilyBoost**: no public statistics table found [UNCONFIRMED] — only the modelled Treasury margin (T1).

---

## Gaps / risks

1. **No public joint/couple/family income distribution.** IRD is individual-band; census income is individual and banded; WfF family-income distribution covers recipients only. The joint-income structure driving WfF/AS abatement must be imputed (HES/IDI) or targeted via recipient-family margins. **The most important structural gap.**
2. **Take-up-relevant cross-tabs are thin** (e.g., renters by income × region × household type). Reconstruct via calibration + rules engine, not direct targets.
3. **AS by Area 1–4 is not a standing series** — need TA→Area crosswalk (H4) or OIA; known break at 2027-04-01 (H4c redraw).
4. **IRRS dollars unconfirmed in open dashboard** — source from Kāinga Ora/Vote Housing.
5. **Benefit expenditure is annual + national only** — regional dollars must assume uniform average payment within benefit (flag as modelling assumption).
6. **Treasury distributional outputs assume full take-up** — denominator, not actuals; never calibrate to them as actuals.
7. **FamilyBoost unvalidatable** against admin data today.
8. **Household/family projections stale (2013-base)** — use census counts.
9. **Portal migration risk**: Infoshare → ADE Time Series (April 2027) will change URLs/schemas; pin to file CDNs.
10. **Bot walls** on stats.govt.nz SPA, treasury.govt.nz HTML, catalogue.data.govt.nz — use the open file CDNs and GitHub for pipelines.

## Licence summary

| Source group | Licence | Confidence |
|---|---|---|
| Stats NZ (census, estimates, HES, QES, LEED) | CC BY 4.0 (NZGOAL default) | High |
| IRD statistics | CC BY 4.0 (data.govt.nz facet) | High |
| MSD Annual Report 2025 | CC BY 4.0 | Confirmed (in PDF) |
| MSD Fact Sheets (data.govt.nz mirror) | CC BY 4.0 | High |
| MSD Benefit System Report; take-up PDFs | unstated | Unconfirmed |
| Treasury DistributionExplorer, IncomeExplorer | MIT | Confirmed |
| Treasury analytical-note-24-01 | CC0-1.0 | Confirmed |
| **Treasury AN25-01, emtr** | **no LICENSE file** | Confirmed absent — do not assume |
| TAWA methodology PDF; Budget docs | CC BY 4.0 / Crown | High |
| **MBIE tenancy bond data** | **CC BY 3.0 NZ** (not 4.0) | Confirmed |
| MBIE minimum wage; Work and Income pages | CC BY 4.0 / Crown | High |

---

*Prepared for microcosm-nz (#343), 2026-07-06; all entries verified against live sources that day per the status column. Re-verify Infoshare URLs after the April 2027 ADE migration, IRD/MSD filenames each release cycle, and the HUD dashboard xlsx monthly. Companion: rulespec-nz provides the rules; the ~44% AS take-up (M6) and ~87% WfF take-up (M5) anchor the dollar take-up gap.*
