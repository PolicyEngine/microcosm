# Binding SPI band rows at their own tax year (option 1 pilot)

Evidence for PolicyEngine/microcosm#1413. The pilot binds the four SPI 2023-24
Table 3.6 pension band rows at 2023-24 through dated engine measures
(`spi_2023_*`) instead of projecting them to calendar 2025. This folder keeps
aggregates only. The spine, the calibrated H5 files and the licensed inputs
stay outside the tree under `~/microcosm-uk-work/runs/`.

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
  --dated-year 2023`) tests exactly that: the file's own 2023-24 taxpayers by
  band, against the year HMRC observed.
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
