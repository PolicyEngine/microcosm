# UK income targets at the calibration year (microcosm#280 lane)

How the UK national calibration binds the HMRC income facts at its period, why
the 2023-24 SPI rows move before they bind, and what the calendar-year window
means. Rulings by María on 2026-09-22; evidence in the 2026-09-21 assessment
behind PolicyEngine/chronicle#280.

## The finding

The v20 national candidate (September 2026) fitted every SPI component band
and still measured income tax 13 percent under the OBR line, the state-pension
bands 8 to 18 percent over and the self-employment £20-40k bands over. All 129
`hmrc_spi` references were the published 2023-24 values bound at period 2025
as identity holds: a value moves only when its reference declares an
`uprating_index` with a UK applier, and the family declared none. The engine
fixes every pensioner's amount at the period's rate, so the state-pension
family showed the seam exactly: summed over the taxpayer bands the calibrated
spine sat at 1.005 of the 2023-24 target times the new State Pension rate ratio.
Employment income sat at 0.906 of its uprated target, private pensions at 0.918,
dividends at 0.924: roughly £120bn of 2025 taxable income the weights could not
supply.

## The period convention

The calibration binds at calendar 2025. In the engine the OBR indices are
calendar-year series keyed at 1 January, so period 2025 is calendar 2025 for
incomes; tax-year policy amounts (rates, allowances, the State Pension) carry
the 2025-26 value for the whole period (verified on the v20 file: state pension
at 2025 over the 2024 report is 1.041 for every recipient). Facts published by
April-to-March years therefore bind as the months to the end of the calendar
year: three twelfths of the year opening in Y-1 and nine twelfths of the year
opening in Y. That is the `calendar_year_window` value operation
(`ledger_targets.CALENDAR_YEAR_WINDOW_WEIGHTS`): it resolves under
`latest_not_after`, takes both years from one series identity, refuses a window
with either year absent, and writes each member's year, weight, value and
assertion on the spec.

Facts already at a calendar-year period (the DWP monthly cubes) keep their
calendar-2025 windows; the ESA rows join them below.

## Uprating the SPI 2023-24 rows

Every `hmrc_spi` reference declares an `uprating_index`; the appliers live in
`uk_runtime/hmrc_uprating.py` and are registered in `UK_UPRATING_APPLIERS`, so
the generator and the runtime apply the same factor and the committed
membership carries the uprated value with its receipt.

- Amount rows: `policyengine_uk_parameter:<path>`, the ratio of the pinned
  engine's parameter between 1 January of the fact's opening year and 1
  January of the calibration year, the instants the SPI donor rebasing already
  uses. Employment income follows `obr.average_earnings`, self-employment
  `obr.per_capita.mixed_income`, dividends and property `obr.per_capita.gdp`,
  private pensions `obr.private_pension_index`, savings interest
  `ons.household_interest_income`; the state pension follows
  `gov.dwp.state_pension.new_state_pension.amount`, the rate the engine pays
  (203.85 to 230.25 a week, 1.1295). On policyengine-uk 2.98.0 the 2023 to 2025
  factors are 1.1067, 1.0344, 1.0826, 1.0877 and 1.1819.
- Count rows: `hmrc.itl_2026.taxpayer_count_growth_by_total_income_band`, HMRC's
  own projected growth in Income Tax payers for the Table 2.5 band that contains
  the SPI band (Income Tax liabilities statistics, July 2026), as the
  calendar-2025 window of the 2024-25 and 2025-26 projections over the 2023-24
  outturn, read from the vendored `hmrc_itl_taxpayer_counts.json` so a resource
  that lags the feed pin refuses. Two SPI bands share each of the 30-50k,
  50-100k and 200-500k Table 2.5 bands; 1m+ takes 1m-2m and 2m+.

The spec metadata records the index, the parameter or resource, the instants
or years, both values, the factor and the pre-alignment value; the membership
lists the same under `uprating_holds`.

## The calibration-year anchors

HMRC's Table 2.5 binds three targets (the `hmrc_itl` family) at the
calendar-2025 window: Income Tax payers (`income_tax > 0`), total income and
Income Tax liabilities by eleven total-income bands, all sliced on the engine's
`total_income`. They
are the anchors the component bands lack: every component can fit while tax
is short, and a level shift of the whole family is invisible to the family
itself. The window of the two projections sums to £322.7bn of liabilities and
39.5m taxpayers; OBR's FY2025-26 receipts line (£331.4bn) stays bound beside
it, 2.7 percent apart, the difference being the fiscal-year basis of the OBR
row.

The SPI savings-interest rows (Table 3.7, taxpayers' interest from banks and
building societies, ISA interest excluded) bind uprated by the household
interest index. The ONS UKEA HAXV row is households' D.41 interest resources in
the national accounts, a concept no cash-interest variable reaches; it stays on
the measure-exclusion register as a measured diagnostic with that reason
(microcosm#866).

## ESA on the payment-type cube

The three ESA rows bind DWP's Stat-Xplore ESA caseload by payment type
(chronicle#282) as the mean of the four quarterly points inside calendar 2025
(February, May, August, November), summed over payment types:

- `dwp.esa_claimants`: contributions based, both, income based; the published
  total's credits-only claimants (78k to 84k through 2025) have their National
  Insurance record credited without payment and are left out rather than
  described as unpublished;
- `dwp.esa_contrib_claimants`: contributions based plus both, matching the
  model's `esa_contrib > 0`, which includes dual claimants;
- `dwp.esa_income_claimants`: income based plus both, matching `esa_income > 0`.

2025 is the year of the ESA income-related managed migration to Universal
Credit (income based 499k in February to 30k in November), so a single point
would misstate the year; the calendar mean is what the FRS 2024-25 base can be
asked to represent.

## The OBR rows at the window

The 27 OBR receipts and expenditure rows that resolved by latest-not-after to
FY2025-26 now bind at the calendar-2025 window too (three twelfths of the
FY2024-25 outturn, nine twelfths of the FY2025-26 forecast, from the March
2026 EFO series), the same rule as the HMRC liabilities anchors. Income tax
moves from £331.4bn to £325.1bn, the window of the £305.9bn outturn and the
£331.4bn forecast, and sits 0.7 percent above the HMRC liabilities window
(£322.7bn): the two anchors now share a basis. Two OBR rows keep their
declaration: the Universal
Credit total is the sum of its two capped-and-uncapped rows (each of which
takes the window on its own), and the cars share of fuel duty is pinned to the
FY2024-25 outturn where the vehicle split is published.

## The region tier

The SPI Table 3.11 rows (income and tax by region and country, all taxpayers)
bind on the twelve-area region tier (microcosm#905) as the `hmrc_spi_region`
family: Income Tax payers, total income and Income Tax liabilities by the ten
regional total-income bands, one reference per area scoped by the household-
region predicate the generator stamps, sliced on the engine's `total_income`
by the row's own band edges. The publisher's regional bands stop at 200,000
and over. Each row moves to 2025 by HMRC's projected growth for the same
measure in the Table 2.5 band(s) the regional band spans (Income Tax
liabilities statistics, July 2026): payers by `taxpayer_count_growth`, total
income by `total_income_growth`, liabilities by `total_tax_growth`, all read
from the vendored `hmrc_itl_taxpayer_counts.json`, which now carries the
three Table 2.5 measures. Two SPI bands share the 30-50k and 50-100k Table
2.5 bands; the 200k-and-over row takes the window over 200k-500k, 500k-1m,
1m-2m and 2m-and-over together. Ruled by María on 2026-09-22: the regions bind
uprated, not as diagnostics. Table 2.2 publishes taxpayer counts by region only
and is not bound.

## Reserved income rows on the spine

The spine stage `spi_income_band_donors` (between `spi_support_channel` and
`hmrc_spi_income_spine`) reserves 120 SPI-channel households for each HMRC
Table 2.5 total-income band from £200,000: 200k-500k, 500k-1m, 1m-2m and 2m
and over. Each is a whole FRS household copied at clone index 2, flagged
`household_is_spi_income_band_donor` with its band in
`spi_income_band_donor_lower_bound`, with one carrier adult
(`person_is_spi_income_band_carrier`) drawn without replacement by the SPI
2022-23 tape's FACT-weighted propensity for the band given region, sex and
age band. The copy starts at the build tax year's published band taxpayers
over 120 (359k/120 = 2,992 for 200k-500k on the 2024-25 projection, 61k/120
= 508, 20k/120 = 167 and 10k/120 = 83), mass added and receipted as the CGT
band donors' is. The income stage's `resample_band_donor_leaves` operation
then gives every carrier a band-conditional draw: one tape record whose
published total income (TEI + TII) lies in the band, FACT-weighted with
replacement from the carrier's region where that regional pool holds at
least 20 records, otherwise nationally, all eighteen stage-1 leaves copied
together and uprated as the forest draws are. Composite records stay in the
pools as published. Non-carrier adults in donor households keep the ordinary
forest draw; stage 2 refills their FRS-only inputs as for every synthetic
row. The stage-health gate `uk_stage_spi_income_band_donors_support` checks
that every band carries its donors at positive band-exact weight with one
carrier each. The methodology and the evidence for it are in
`repos/uk-spi-income-band-donors-plan-2026-09-23.md` and
`repos/uk-upper-tail-investigation-2026-09-23.md`: before this stage the
spine, like the enhanced FRS, carried no record above £1.51m of total income.

## Not done here

- The property-income amount rows stay signed out: the spine's
  `property_income` is the FRS rent received (sub-lets, lodgers, royalties)
  while the SPI concept is landlords' net income after expenses.
- The ESA rows' migration residual and the SPI support channel's benefit fill
  (microcosm#840, #867, #869) are spine items.
