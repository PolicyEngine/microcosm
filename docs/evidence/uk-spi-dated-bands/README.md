# Binding SPI band rows at their own tax year (option 1 pilot)

Evidence for PolicyEngine/microcosm#1413. The pilot binds the four SPI 2023-24
Table 3.6 pension band rows at 2023-24 through dated engine measures
(`spi_2023_*`) instead of projecting them to calendar 2025. This folder keeps
aggregates only. The spine, the calibrated H5 files and the licensed inputs
stay outside the tree under `~/microcosm-uk-work/runs/`.

## A/B result: the pilot does not win

The pilot fits the rows it binds. On every test outside them it is worse than
the baseline, and further than the baseline from the re-index that
PolicyEngine/microcosm#1414 merged for #1069. It should not replace that
re-index.

### Builds

All three are national builds on the same FRS 2024-25 spine, with weights
solved at 2025 over 1,273 targets.

- **Baseline:** main at 7d45c0c33, with the projected rows (H5 `de754f0b…`).
- **Pilot:** this PR's rows at 2023-24 (H5 `285fd3b6…`), built from local
  commit 71856c6fa. That commit is this branch at 058690bdb plus a fill of
  the hold's approval fields marked "PILOT BUILD ONLY". It was never pushed.
- **#1414 candidate:** the re-index, from #1069's run (H5 `2834c9e8…`). Only
  State Pension recipients and the allowance were measured on it.

The measurements ran the pilot build's tools. The ITL in-sample flags were
recomputed from the receipts under this PR's later rule: a sum is in sample
only when its parts are and nobody it counts is outside them. The receipts
(aggregates only) are under `~/microcosm-uk-work/runs/spi-dated-bands/measures/`,
summarised by `summarize_ab.py` there. The baseline's recipient and allowance
receipts equal the #1069 lane's receipts for the same file.

### State Pension recipients by taxpayer status, 2025-26

HMRC's implied ranges are 8.65m–8.81m taxpaying and 3.58m–3.74m not
taxpaying. They come from #1414's
`docs/evidence/uk-pensioner-taxpayers-1069/hmrc_implied_recipients.json`:
SPI Table 3.6's 7.98m State Pension taxpayers over ITL Table 2.1's 8.16m
taxpayers of State Pension age (2023-24), applied to calendar 2025's twelve
areas and United Kingdom.

| Build | Taxpaying | Not taxpaying | All recipients |
|---|---|---|---|
| Baseline | 8.84m (0.4% above the range) | 3.53m (1.3% below) | 12.37m |
| Pilot | 9.26m (5.1% above) | 3.13m (12.6% below) | 12.38m |
| #1414 candidate | 8.74m (inside) | 3.63m (inside) | 12.37m |

Against the baseline, the pilot moves about 0.4m recipients from not taxpaying
to taxpaying.

### SPI pension band rows on both definitions

The table gives the total relative error by family on two definitions:

- **2023**, the pilot's definition: published 2023-24 values against the dated
  2023-24 measures.
- **2025**, the baseline's definition: those values projected to 2025 by the
  declared indices, against 2025 income.

| Family | Definition | Baseline | Pilot |
|---|---|---|---|
| State Pension count | 2023 | -4.15% | +0.12% |
| State Pension count | 2025 | +5.68% | +10.66% |
| State Pension amount | 2023 | -2.02% | +1.72% |
| State Pension amount | 2025 | +6.73% | +11.28% |
| Private pension count | 2023 | -4.94% | -0.02% |
| Private pension count | 2025 | +1.41% | +5.31% |
| Private pension amount | 2023 | -5.03% | +0.02% |
| Private pension amount | 2025 | -0.86% | +3.44% |

Each build is closer on the definition it binds, so this table cannot rank
them.

### ITL July 2026 projections, calendar 2025

The tolerance is 5% for national cells and the residents' total, and 10% for
each area. Neither build binds Tables 2.1 or 2.2. Both bind Table 2.5's
bands, so those cells are in sample.

| Table: measure | Cells | Baseline misses | Pilot misses | Baseline largest | Pilot largest |
|---|---|---|---|---|---|
| 2.1: State Pension age, UK | 1 | 0 | 0 | -0.81% | +3.92% |
| 2.1: 65 and over, UK | 1 | 0 | 0 | -0.53% | +3.79% |
| 2.1: below State Pension age, UK | 1 | 0 | 0 | -0.69% | -1.68% |
| 2.2: State Pension age, 12 areas and residents | 13 | 0 | 1 | +6.44% (North East) | +9.87% (London) |
| 2.2: 65 and over, 12 areas and residents | 13 | 0 | 2 | -6.42% (East Midlands) | +11.56% (London) |
| 2.5: all taxpayers | 1 | 0 | 0 | -0.72% | -0.40% |
| 2.5: taxpayers by band (in sample) | 11 | 0 | 0 | -3.93% | -3.13% |

The pilot misses three cells:

- residents at State Pension age, +5.84% (baseline +1.03%);
- residents aged 65 and over, +5.68% (baseline +1.28%);
- London aged 65 and over, +11.56%.

**Dated check.** This compares each file's own 2023-24 taxpayers by band, on
the dated measures, with Table 2.5's 2023-24 outturn.

- Both builds miss the same 6 of 11 bands: every band from £50,000 up except
  £200,000–£500,000.
- The largest miss is at £1m–£2m on both: +26% for the baseline, +28% for the
  pilot.
- All taxpayers are within 0.6% on both.

So this check does not separate the builds. It does show that the
carried-back file misses HMRC's 2023-24 distribution above £50,000. The check
cannot say whether the weights, the population factor or the engine's indices
cause this.

### Sole-income State Pension allowance

This runs uk-autumn-budget-dashboard-2026#182 at fa9c42b1 through
policyengine.py 6.11.1 and policyengine-uk 2.151.0. Each cell is people
directly affected, then the cost to the Exchequer.

| Design | Year | Baseline | Pilot | #1414 candidate |
|---|---|---|---|---|
| Sole income | 2027-28 | 1.235m, £84m | 1.117m, £75m | 1.326m, £93m |
| Sole income | 2031-32 | 1.731m, £330m | 1.542m, £289m | 1.842m, £361m |
| Total income | 2027-28 | 0.574m, £28m | 0.491m, £24m | 0.593m, £30m |
| Total income | 2031-32 | 1.214m, £148m | 1.051m, £123m | 1.252m, £154m |

The pilot gives the lowest count and cost. The candidate, whose taxpaying
split is inside HMRC's ranges, gives the highest.

### Calibration fit (in sample)

| | Baseline | Pilot |
|---|---|---|
| Final loss | 0.00553 | 0.00535 |
| Targets within 10% | 98.98% | 99.21% |
| Effective sample size | 7,527 | 7,561 |

The two builds solve different definitions of the four families, so these
figures are not a ranking either.

### Why the pilot loses (not isolated)

The pilot fits the 2023-24 rows almost exactly. Yet the same weights put
taxpaying State Pension recipients 5.1% above HMRC's 2025-26 range, and
taxpayers of State Pension age 3.9% above Table 2.1. HMRC's ITL tables are
projections from the same SPI year. So the file's path from 2023-24 to
2025-26 disagrees with HMRC's projection from that year, and no measurement
here says which link breaks. The candidates are:

- the weights, solved at 2025;
- the population factor, which is national and by sex and age band;
- the engine's carry-back indices for pensions and other income;
- the State Pension's own carry-back (paid at 2023-24 rates);
- HMRC's projection, which is not outturn.

The dated check misses mainly above £50,000, where pensioners are few, so it
does not locate the error either.

One build was not run: these rows combined with #1414's bound Table 2.2
targets. That build would also put Table 2.2 in sample, which removes the
out-of-sample test.

## Extending option 1 to every SPI band family

### What binds SPI band rows today

Read from `uk/uk_population_targets.json` at this branch's head.

| Family | Rows | Value moved to calendar 2025 by |
|---|---|---|
| `hmrc.spi.{employment,self_employment,property}_income.amount_by_total_income_band` | 3 targets × 13 bands | Table 2.5 total-income growth of the band (`hmrc.itl_2026.total_income_growth_by_total_income_band`) |
| `hmrc.spi.dividend_income.amount_by_total_income_band` | 13 bands | per-capita GDP index |
| `hmrc.spi.savings_interest_income.amount_by_total_income_band` | 13 bands | household interest to the dataset year, then per-capita GDP |
| `hmrc.spi.{employment,self_employment,dividend,property,savings_interest}_income.count_by_total_income_band` | 5 targets × 13 bands | Table 2.5 taxpayer-count growth of the band |
| `hmrc.spi.{state_pension,private_pension_income}.{count,amount}_by_total_income_band` | 4 targets × 13 bands | **this pilot:** none; bound at 2023-24 |
| `hmrc.spi_region.{taxpayers,total_income,income_tax}_by_region_<band>` | 30 targets × 12 areas | Table 2.5 growth of the band (count, total income, total tax) |
| `hmrc.{employment,self_employment}_income.{count,amount}` (constituency, local authority) | unbanded | Table 2.5 all-band growth |

### Which have the band-projection problem

Every banded family has nominal 2023-24 edges and model incomes measured at
2025. Each moves its value by the growth of the whole nominal band among all
taxpayers.

- **Pension rows.** The rows' own group (pensioners) grows differently from
  the band's taxpayers, which is where the projection fails. The pensioner
  rows grew 4.8% while HMRC projects taxpayers of State Pension age 10.4%
  higher (BAND-PROJECTION.md).
- **Employment rows.** Employees are most of each band's taxpayers, so the
  band's growth is close to their own. These are the least exposed.
- **Self-employment, property, dividend and savings-interest rows.** Their
  recipients' incomes move with other indices; dividends and savings already
  use per-capita GDP or the interest path, not the band's. These are exposed
  as the pension rows were, to a degree no build has measured.
- **Regional bands.** These apply the United Kingdom band growth to every
  area. HMRC's own Table 2.2 shows areas growing at different rates.
- **Unbanded area tables.** These cross no edge, so they do not have this
  problem. Their all-band growth is a separate approximation.

### What extending needs

The pilot's 2023-24 simulation already computes every income above. Adding a
family adds names, not simulations:

1. Extend `UK_SPI_DATED_MEASURE` and `UK_SPI_ADDITIVE_MEASURES`
   (`uk_runtime/spi_dated_measures.py`) with `employment_income`,
   `self_employment_income`, `dividend_income`, `property_income` and
   `savings_interest_income`. These read the engine variables the 2025 rows
   bind, at 2023.
2. For the regional amount rows, add additive twins of the two classifiers:
   total income and Income Tax, carrying the population factor. Today
   `spi_2023_total_income` and `spi_2023_income_tax` are classifiers, read
   unscaled.
3. Rebind each family as the pilot rebinds the pension rows:
   - groupby `spi_2023_total_income`;
   - taxpayer filter `spi_2023_income_tax > 0`;
   - `measurement_period` 2023, `require_matching_fact_period`;
   - no `uprating_index`.
4. Regional population. The population factor is national, by sex and age
   band. A regional factor needs ONS mid-year estimates by region, sex and
   age for 2023 and 2025, two more vendored resources. Without them, the
   regional rows carry the national age-sex change, and regional migration
   between 2023 and 2025 is not represented.

### What the dated rows assert, and their limits

- **The rows describe a reweighted 2025 file, not a 2023 one.** Weights stay
  solved at 2025 and ages are held, as policyengine-uk holds them when it
  projects. A dated row therefore describes the 2025 file's people, each
  scaled by how many people of their sex and age band lived in the United
  Kingdom at 1 October 2023 against mid-2025, with incomes carried back by
  the engine's own indices.
- **The 2023-24 distribution is pinned in the 2025 weights' terms.** This is
  sound to the extent that the engine's indices carry incomes as the economy
  did. The dated Table 2.5 check (`tools/check_uk_itl_projections.py
  --dated-year 2023`) puts the file's own 2023-24 taxpayers by band beside
  the year HMRC observed. It is a joint test of the 2025 weights, the
  population factor and the engine's indices: a pass does not isolate the
  indices, and a miss does not say which of the three moved it.
- **Composition change within an age and sex band is not captured.** This
  includes 2023-25 migration's income profile.

## Do the US SOI band rows have the same problem?

Partly, in a milder form. The code was read at this branch; no US build was
run, and the pinned US feed is not on this machine.

- **Raw cross-period AGI bands are refused.** The US build compiles at
  `target_period` 2024 (`tools/build_us_fiscal_refresh_release.py`,
  `PERIOD = 2024`). `_is_untransformed_cross_period_agi_slice`
  (`us_runtime/fiscal_targets.py`) marks every bounded-AGI SOI fact whose tax
  year differs from the build period, and the compile drops it. That refusal
  is the US form of the UK fix, made earlier (microcosm#489, #940). The
  exceptions are:
  - EITC decompositions;
  - Historic Table 2 interest and capital-gains slices;
  - state Historic Table 2 bands from $100k.
- **The excepted rows bind as source-year shares of a total that binds:**
  - state AGI bands via `_rebase_soi_state_agi_bands`, onto the state total;
  - EITC via `_uprate_cross_period_eitc_decompositions`, onto the active
    EITC or AGI-bucket total;
  - interest and capital gains via one national realized factor.

  The totals then age by CBO series ratios. Counts never age
  (`us_runtime/target_aging.py`).
- **Edges stay nominal and AGI is measured at 2024.** So the shares carry
  the same drift the UK band rows had:
  - (Inferred) More returns cross each edge as AGI grows from TY2022 or
    TY2023 to 2024, so source-year shares understate the build-year shares
    above an edge, most at $500k–$1M and $1M+.
  - (Inferred) Unaged state counts above $100k are understated by the
    returns that crossed $100k.
- **Levels are not stale.** The US has no stale nominal level, which was the
  UK pension rows' larger failure. No US measure is dated to its fact's year.
  Option 1 there would run policyengine-us at the SOI tax year on
  carried-back inputs, with population and filer factors, and bind shares and
  levels at that year.
- **Open questions:**
  - which SOI tax years the pinned US feed carries;
  - how large the share drift is, which a one-year differential would
    measure when TY2024 Historic Table 2 lands.
