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

SPI-channel rows carry their FRS twin's carriers, as they carry every root column.
A row drawn from the SPI tape (`person_is_spi_income_recipient`) has the tape's PAY +
EPB + TAXTERM as pay, where EPB is every taxable expense payment and benefit in kind,
so it keeps its twin's car only where its drawn EPB is positive. The channel's other
rows, the undrawn copies and the dependants aged 16 to 19, keep their twin's pay, and
keep its car as an FRS row does.

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
  stage's weights (all of them, where the private fuel holders leave too few). The tape
  rows that lose their car for want of EPB draw at the holders' electric share. Matching
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
capped at EPB (María's ruling of 30 October 2026) within the pay the row's employee
pension contribution leaves. The SPI income stage capped that contribution at the pay
it drew, and `salary_sacrifice`, which runs later on the carved pay, refuses a
contribution above the pay that remains; it keeps the salary and bonus sacrifice within
that pay itself. A row whose contribution takes all of its pay has no room, so it loses
its car and keeps its pay. The room bites only where the SPI stage's drawn contribution
exceeds the row's cash pay, PAY + TAXTERM, since that stage capped it at a pay that
includes EPB. The receipt measures what it costs beside a cap of EPB within pay alone:
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
   stays in pay.

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
taxable value implies roughly half of that, so even a file that matches Table 1's
920,000 holders and £3,070m should fall about 30 to 45% short. Matching Table 1's
taxable value does not pin the base, since a different electric mix and different
prices reach the same total. So the stage's receipt records the weighted total list
price of electric cars and of the others, for the results to read the gap as base
size.

Two parts of a shortfall are measured here rather than corrected (María's ruling of 31
October 2026: measure first).

- **The tape rows' EPB draw.** A tape row keeps its twin's car only where its drawn
  EPB is positive, and EPB is drawn on uniforms the car never conditions. The SPI
  channel holds half the working-age prior mass, so before calibration holders and
  the base come to about half the FRS level plus half of it times the share of tape
  rows that draw EPB. The receipt records the weighted holders and list price lost
  this way (`holders.spi_dropped_without_epb`).
- **The carve-out.** Cars that do not fit their EPB are scaled or dropped, above.

These figures, like the list price totals, are at the stage's weights, before
calibration moves the holders. The results measure the base on the calibrated file.

## Measured, not bound

No HMRC company car fact is a calibration target: the stage draws fuel and CO2 to
Tables 2A and 2C, so binding them would be circular. No build step reads the other
tables after calibration yet: comparing the calibrated file with HMRC's Table 1
(recipients, taxable value, income tax and NICs), Table 3A (by income range) and Table
4 (by region) is a manual step. `uk_stage_company_cars_invariants` blocks a release on
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
