# Company cars (dashboard #316)

Budget 2026 raises every company car appropriate percentage by one point in 2030-31
(Table 6.1 row 86). policyengine-uk 2.165.0 (PolicyEngine/policyengine-uk#2750) taxes
the company car and car fuel benefits, and the Class 1A NICs on them, from nine person
inputs that default to no car, so a file that sets none of them scores the rise at £0.
The `company_cars` stage, the last in the UK roster, sets six of them.

| Input | Where it comes from |
|---|---|
| `company_car_list_price` | The FRS CARVAL band, drawn within it from WAS round-8 company cars' list prices |
| `company_car_fuel_type` | The FRS FUELTYP code; HMRC Table 2C's shares where it is unknown |
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
Their pay is the SPI tape's PAY + EPB + TAXTERM, where EPB is every taxable expense
payment and benefit in kind, so a twin's car is kept only where the row's drawn EPB
is positive.

## The draws

Every draw is keyed on the person's identity, so reordering rows moves none.

- **Fuel.** Petrol is `PETROL`; a hybrid or another fuel is `ALTERNATIVE_FUEL`; an
  electric car is `ELECTRIC`; a diesel is `DIESEL_RDE2`, since every new diesel has
  met RDE2 since January 2021 and company cars turn over in three to four years (the
  diesel supplement moves the baseline only, not the 2030-31 rise). An unknown fuel
  draws at HMRC Table 2C's shares. It never draws a hybrid: Table 2C counts hybrids
  inside petrol and diesel, so a drawn petrol or diesel car enters the CO2 draw
  below as a non-hybrid. The effect is small.
- **List price.** WAS round 8 (April 2020 to March 2022) records company cars' list
  prices to the nearest £1,000. They are grown to the build year by the engine's CPI
  index from 2021, and each holder draws within their CARVAL band from the weighted
  quantile function of the WAS cars inside it. The open band (£40,001 and over) and
  an unknown band draw from the WAS cars of the same fuel class (electric or not). A
  cell with fewer than 3 WAS households falls back to all fuel classes, then to a
  uniform draw within the band (up to £80,000 for the open band).
- **CO2.** HMRC publishes recipients by CO2 band and by fuel, but no table crosses
  them. An electric car emits none. The others are split between 1 to 50 g/km and the
  bands above it so that the weighted share at 1 to 50 g/km matches Table 2A's among
  cars that emit CO2, hybrids first; the bands above are drawn at Table 2A's shares,
  uniformly within each. Every figure in the open band (170 g/km and over) is above
  the maximum percentages' thresholds.

## SPI-channel pay

On an SPI-channel row the engine's car and fuel benefits come out of
`employment_income` (María's ruling of 30 October 2026). They are capped at EPB,
and at the pay the row's pension deductions leave: `salary_sacrifice` keeps employee
contributions and salary sacrifice within pay and bonus sacrifice within half of it,
and the carve-out keeps them there.

1. Private fuel goes first where its benefit alone exceeds the cap.
2. A car whose benefit still exceeds what is left has its list price scaled down
   until it fits, with its contributions dropped. The benefit is the price above the
   capped contributions times the appropriate percentage, so the scaling is exact.
3. A car that would need a list price below £5,000 to fit is dropped, and its EPB
   stays in pay.

The stage then has the engine price its output again. On every SPI-channel row, pay
plus the engine's benefits must equal the pay the row brought into the stage (the
tape's PAY + EPB + TAXTERM, less any salary sacrifice). The engine computes in single
precision, so the receipt records the largest miss rounded up to the penny, and the
gate allows 5p. The SPI employment income targets read employment income plus
`benefits_in_kind`, so they read the same total as before the stage.

## What to expect of the 2030-31 score

Budget 2026 raises every appropriate percentage by one point, the zero-emission rate
and the maximum included. So the rise adds about 1% of each car's list price above
its capped contributions to its taxable value, plus 1% of the fuel benefit
multiplier for each fuel benefit holder. Electric cars count as much as any other:
the frozen electric share moves the baseline, not the rise. The base the rise acts
on is the build year's fleet carried forward. Fuel type and CO2 are not uprated,
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

## Measured, not bound

No HMRC company car fact is a calibration target: the stage draws fuel and CO2 to
Tables 2A and 2C, so binding them would be circular. The national build reports the
calibrated file against HMRC's Table 1 (recipients, taxable value, income tax and
NICs), Table 3A (by income range) and Table 4 (by region).
`uk_stage_company_cars_invariants` blocks a release on any record that breaks an
invariant, and on SPI-channel pay off the tape's total.

## Disclosure

The FRS, WAS and SPI are UK Data Service safeguarded data. The stage's receipt
carries rounded counts, support bands and weighted aggregates only, and no record's
value. A weighted figure reads the floor's marker when any cell it rests on has
fewer distinct source households than the floor, so support copies count once.
Shares that sum to one are withheld together, since one share is the others'
complement. A count of records that break an invariant is zero exact and banded
otherwise.
