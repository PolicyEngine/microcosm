# Company cars (dashboard #316)

Budget 2026 raises every company car appropriate percentage by one point in 2030-31
(Table 6.1 row 86). policyengine-uk 2.165.0 (PolicyEngine/policyengine-uk#2750) taxes
the company car and car fuel benefits, and the Class 1A NICs on them, from nine person
inputs that default to no car, so a file that sets none of them scores the rise at £0.
The `company_cars` stage sets six of them. It runs right after the SPI income stage
(`hmrc_spi_income_spine`), so every later stage reads the tape rows' pay without the
car benefits, as the engine's `employment_income` excludes them: `salary_sacrifice` and
its bounds, the take-up draws' earnings rules and the CGT and ISA stages' income proxy
(María's ruling of 31 October 2026). The proxy sums `employment_income` without
`benefits_in_kind`, so it leaves the car benefits out on every row, as it always has on
FRS rows. Two things the SPI income stage set from the pay before the carve are dealt
with here: the tape rows' employee pension contributions, which the carve-out leaves
room for, and their UC gainful self-employment, which the stage derives again on the
carved pay (below).

| Input | Where it comes from |
|---|---|
| `company_car_list_price` | The FRS CARVAL band, drawn within it from WAS round-8 company cars' list prices |
| `company_car_fuel_type` | The FRS FUELTYP code; HMRC Table 2C's shares among the unknown fuels, without electricity where the holder reports private fuel |
| `company_car_co2_emissions` | HMRC Table 2A's shares among cars that emit CO2; zero for an electric car |
| `company_car_electric_range` | 50 miles for a car at 1 to 50 g/km |
| `company_car_capital_contributions` | The FRS CARAMT, where CARCON says the employee contributed |
| `company_car_private_fuel_provided` | The FRS EXPBEN03, on a car that is not electric |

Availability, payments for private use and the 2020 registration flag keep the
engine's defaults: no survey records them.

## Who has a car

The FRS asks each current employee job whether the employer provided a company car
(EXPBEN01) in the last 12 months. The root stage, `frs_spine`, carries each person's
car from their car job with the lowest JOBTYPE, the main job first, so a person has at
most one, as in the engine. The survey asks CARVAL and CARCON of a company van too;
they are taken as the car's.

SPI-channel rows carry their FRS twin's carriers, as they carry every root column,
and keep their twin's car. A row drawn from the SPI tape (`person_is_spi_income_recipient`)
has the tape's PAY + EPB + TAXTERM as pay, where EPB is every taxable expense payment
and benefit in kind. EPB is drawn on uniforms the car never conditions, so every tape
row keeps its twin's car, whatever EPB it drew, and the part of the car's benefits above
its EPB moves from its PAY into its EPB (below; María's rulings of 3 November 2026). The
channel's other rows,
the undrawn copies and the dependants aged 16 to 19, keep their twin's pay, and keep
its car as an FRS row does.

## The draws

Every draw is keyed on the person's identity, so reordering rows moves none.

- **Fuel.** Petrol is `PETROL`; a hybrid or another fuel is `ALTERNATIVE_FUEL`; an
  electric car is `ELECTRIC`; a diesel is `DIESEL_RDE2`, since every new diesel has
  met RDE2 since January 2021 and company cars turn over in three to four years. The
  diesel supplement sets a diesel's appropriate percentage, so it moves the baseline,
  and through the scaled cars the base the 2030-31 rise acts on (below). The unknown
  fuels draw at HMRC Table 2C's shares together. A holder who reports private fuel,
  which an electric car cannot carry, draws among petrol and diesel alone, and the
  others draw electricity at the share that keeps Table 2C's among all of them at the
  stage's weights (all of them, where the private fuel holders leave too few). Matching
  Table 2C at the stage's weights makes each drawn fuel depend on the other drawn cars'
  private fuel and weights, as the CO2 split does: reordering rows moves none, but a
  car's fuel can move when other rows or weights do. It never draws a hybrid:
  Table 2C counts hybrids inside petrol and diesel, so a drawn petrol or diesel car
  enters the CO2 draw below as a non-hybrid. The effect is small.
- **List price.** WAS round 8 (April 2020 to March 2022) records company cars' list
  prices to the nearest £1,000. They are grown to the build year by the engine's CPI
  index from 2021, and each holder draws within their CARVAL band from the weighted
  quantile function of the WAS cars inside it. The open band (£40,001 and over) and
  an unknown band draw from the WAS cars of the same fuel class (electric or not). A
  cell with fewer than 3 WAS households falls back to all fuel classes, then to a
  uniform draw within the band (up to £80,000 for the open band). A uniform draw
  starts no lower than £5,000, the least price the carve-out keeps a car at, so band
  1's runs from £5,000 to £10,000.
- **CO2.** HMRC publishes recipients by CO2 band and by fuel, but no table crosses
  them. An electric car emits none. The others are split between 1 to 50 g/km and the
  bands above it so that the weighted share at 1 to 50 g/km matches Table 2A's among
  cars that emit CO2, hybrids first; the bands above are drawn at Table 2A's shares,
  uniformly within each. Every figure in the open band (170 g/km and over) is above
  the maximum percentages' thresholds. The draw matches Table 2A among the cars it
  draws for. The carve-out below then drops or scales the high-CO2 cars first, so the
  receipt states the shares of the cars the output holds beside the draw's (`co2.output`
  and `co2.drawn`).

## The tape rows' pay

On a tape row the engine's car and fuel benefits come out of `employment_income`,
capped at EPB + PAY within the pay the row's employee pension contribution leaves
(`cap: epb_then_pay`). EPB holds the benefits first, and the part above EPB moves from
the row's PAY (`hmrc_spi_pay`) into its EPB (`hmrc_spi_employment_benefits`): its PAY +
EPB + TAXTERM stays the tape's, and its split agrees with what the engine taxes (María's
rulings of 3 November 2026). Until then the cap was EPB alone (María's ruling of 30
October 2026), with PAY only where EPB was zero. That dropped or shrank the cars of the
rows that drew a little EPB, which says no more about the car than a zero does (Vahid's
review of microcosm#1491). A row whose EPB and PAY can't hold the car loses it as below.
The SPI income stage capped that contribution at the pay
it drew, and `salary_sacrifice`, which runs later on the carved pay, refuses a
contribution above the pay that remains; it keeps the salary and bonus sacrifice within
that pay itself. A row whose contribution takes all of its pay has no room, so it loses
its car and keeps its pay. The receipt measures what the room costs beside a cap of
EPB + PAY within pay alone:
the cars it scales further or drops, and the list price they give up
(`spi_carve_out.cars_the_employee_contribution_scales_or_drops` and
`weighted_list_price_given_up_for_the_employee_contribution_gbp`). Re-capping the
contribution at the carved pay instead would keep those cars. Employer contributions stay as the SPI income stage set
them on the drawn PAY + EPB + TAXTERM (the FRS twin's rate times that pay, where the
twin has one), as on every tape row that draws EPB, car or not: the carve-out leaves
them alone.

1. Where the car and fuel benefits do not fit together, the private fuel goes first,
   so a car that fits without its fuel is always kept (María's ruling of 31 October
   2026).
2. A car whose benefit still exceeds the cap has its list price scaled down until it
   fits, with its contributions dropped. The benefit is the price above the capped
   contributions times the appropriate percentage, so the scaling is exact.
3. A car that would need a list price below £5,000 to fit is dropped, and its EPB
   and PAY stay in pay.

A scaled car's price is the room over its appropriate percentage, so what sets the
percentage sets the price, and with it the base the 2030-31 rise acts on: the CO2
draw, the 50-mile electric range and the RDE2 diesel. On £2,000 of room a car at 8%
(1 to 50 g/km with the 50-mile range) scales to £25,000, one at 14% (under 30 miles)
to £14,286 and an RDE2 diesel at 25% to £8,000, while an electric car at 2% fits
unscaled up to £100,000. Those assumptions move the 2030-31 base through the scaled
cars, not only the baseline. The receipt records the list price the scaling takes off
and that of the cars it drops (`spi_carve_out.weighted_list_price_scaled_away_gbp`
and `weighted_list_price_dropped_gbp`).

The stage then has the engine price its output again. On every tape row, pay plus the
engine's benefits must equal the pay the row brought into the stage, the tape's PAY +
EPB + TAXTERM. The engine computes in single precision, so the receipt records the
largest miss rounded up to the penny, and the gate allows 5p. The SPI employment
income targets read employment income plus `benefits_in_kind`, so they read the same
total as before the stage.

The SPI income stage set a tape row's UC gainful self-employment from the pay it drew,
by the root's rule: a self-employed main job, or a side trade whose profit is above pay
(uk-data#525). A carved row takes the rule again on its carved pay, so a side trade
whose profit lies between the carved and the drawn pay becomes gainful, as the engine
reads the carved pay. The receipt counts those rows
(`spi_carve_out.uc_gainful_self_employment_on_the_carved_pay`).

## What to expect of the 2030-31 score

Budget 2026 raises every appropriate percentage by one point, the zero-emission rate
and the maximum included. So the rise adds about 1% of each car's list price above
its capped contributions to its taxable value, plus 1% of the fuel benefit
multiplier for each fuel benefit holder. On an unscaled car an electric car counts as
much as any other: its frozen percentage moves the baseline, not the rise. On a scaled
car the percentage sets the price, so it moves the base too (above). The base the rise
acts on is the build year's fleet carried forward. Fuel type and CO2 are not uprated,
list prices and contributions grow by CPI, and the weights are not projected.

Expect the static 2030-31 cost to come in well under Table 6.1's £310m, from the
size of that base. The OBR's NICs line (£92.4m, at Class 1A's 15%) implies about
£616m of extra taxable value, a list price base of about £60bn. HMRC's 2024-25
taxable value implies roughly half of that, so a file that matches Table 1's
920,000 holders and £3,070m, as the national now binds (below), should still fall
about 30 to 45% short. With Table 2C's fuel counts bound as well the electric mix is
pinned, but the prices are not: a different price mix reaches the same total. So the
stage's receipt records the weighted total list price of electric cars and of the
others, for the results to read the gap as base size.

Two parts of a shortfall were measured first (María's ruling of 31 October 2026:
measure first), and the first is corrected now.

- **The tape rows' EPB draw.** Until 3 November 2026 a tape row kept its twin's car
  only where its drawn EPB was positive, and the national `f6e33b93` lost about
  272,000 weighted holders and £10.6bn of list price that way, at the stage's
  weights. They are
  kept now, their benefits moved from PAY into EPB. The receipt records the weighted
  holders kept without EPB and the pay moved (`holders.spi_channel_tape_without_epb`
  and `spi_carve_out.weighted_pay_moved_into_epb_gbp`), and how many of those rows the
  car leaves with no PAY, their cash pay being their termination pay alone
  (`spi_carve_out.kept_leaving_no_pay`). It also counts the rows with EPB whose car
  takes more than it (`spi_carve_out.kept_with_epb_topped_up_from_pay`).
- **The carve-out.** Cars that do not fit their EPB and PAY are
  scaled or dropped, above.

These figures, like the list price totals, are at the stage's weights, before
calibration moves the holders. The results measure the base on the calibrated file.

## Bound and measured

The national calibration binds five of HMRC's 2024-25 facts from the Chronicle feed
(María's ruling of 3 November 2026), in their own family, `hmrc_benefits_in_kind`:

| Target | HMRC | Frame |
|---|---|---|
| `hmrc.benefits_in_kind.company_car_recipients` | Table 1, car recipients, 920,000 | People with a list price |
| `hmrc.benefits_in_kind.company_car_taxable_value` | Table 1, car taxable value, £3,070m | The engine's company car benefit in 2024-25 |
| `hmrc.benefits_in_kind.company_car_fuel.fully_electric` | Table 2C, 465,000 | `ELECTRIC` |
| `hmrc.benefits_in_kind.company_car_fuel.petrol_including_hybrids` | Table 2C, 390,000 | `PETROL` and `ALTERNATIVE_FUEL` |
| `hmrc.benefits_in_kind.company_car_fuel.diesel_including_hybrids` | Table 2C, 60,000 | `DIESEL_RDE2` and `DIESEL_NON_RDE2` |

- **The taxable value** is measured in 2024-25 (`company_car_benefit_2024`,
  `measurement_period` 2024), as the CGT rows are: the calibration year's appropriate
  percentages are a point higher, so measuring 2025 would set 2024-25's total against
  2025-26's rates and pull the base down.
- **The counts** count stored inputs, so the year they are measured in moves nothing
  but the fact. They are held at 2024-25 (`uprating_holds.json`): no index projects
  company car holders, and HMRC's 2025-26 figures are due in July 2027.
- **The fuel counts** bring the fuel mix to HMRC's rather than the FRS respondents':
  about a third of the FRS's company cars are electric, and half of HMRC's. Table 2C
  counts hybrids with their fuel, and the stage maps every FRS hybrid to
  `ALTERNATIVE_FUEL`, which counts with petrol here, since diesel hybrids are few. If
  the reweighting distorts other targets, redrawing the fuel types in the stage is
  the next step.
- The three fuel rows sum to 915,000 against Table 1's 920,000, from HMRC's rounding.

The stage still draws fuel and CO2 from its vendored copy of Tables 2A and 2C, which
the weight solve never reads. Comparing the calibrated file with Table 2A (by CO2),
Table 3A (by income range) and Table 4 (by region) is a manual step.
`uk_stage_company_cars_invariants` blocks a release on
any record that breaks an invariant, a tape row whose employee pension contribution
exceeds its carved pay included, and on a tape row whose pay plus benefits misses the
tape's total by more than 5p.

## Disclosure

The FRS, WAS and SPI are UK Data Service safeguarded data. The stage's receipt
carries rounded counts, support bands and weighted aggregates only, and no record's
value. A weighted figure reads the floor's marker when any cell it rests on has
fewer distinct source households than the floor, so support copies count once.
Shares that sum to one are withheld together, since one share is the others'
complement. A count of records that break an invariant is zero exact and banded
otherwise.
