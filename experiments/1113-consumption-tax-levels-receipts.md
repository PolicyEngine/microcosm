# microcosm#1113 receipts: consumption-tax levels (VAT, road fuel, domestic energy)

Plan: `repos/microcosm-vat-fuel-energy-implementation-plan.md` (María's rulings of 2026-10-08: D1
DESNZ volume at QEP prices stays the energy level, D2 the stage sets the road-fuel level and
incidence, D3 VAT is held out until the engine factor is split). Covers microcosm#1113 and
microcosm#1123 item 3. Branch `uk-consumption-tax-levels`, written on main `75167a688` (#1121
merged) and rebased on main `7f235941c` on 2026-10-09; both lock policyengine-uk 2.122.2 on
policyengine-core 3.32.19. The measurements below ran before the rebase.

Group A needs nothing from PolicyEngine/chronicle#322. Group B re-pins the feed to the Chronicle
commit that carries #322's facts (merged as #323), then adds the division capture and the per-fuel
cars benchmark.

## Commits

- A1 `26e8e3bb7`: `obr.vat`, `ons.household_electricity_expenditure` and
  `ons.household_gas_expenditure` move to the measure-exclusion register, measured on every
  evaluation with `tools/diagnose_uk_consumption_taxes.py`.
- A2 `f7a02a321`: the `level_road_fuel` step sets petrol plus diesel to ONS Consumer Trends 07.2.2
  for calendar 2024, less the donor's own other-fuels share (gate `road_fuel_level`).
- A3 `b609d0767`: the `redraw_zero_road_fuel` step gives every flagged fuel-car household a
  positive spend (gate `road_fuel_incidence`).
- A4 `149fb61b5`: the contract binds `ons.household_road_fuel_expenditure` (07.2.2, calendar 2025,
  GBP 35.099bn), and `obr.fuel_duties_cars` moves to the register. María confirmed this binding on
  2026-10-09.
- A5 `8db8d6868`: the `recompose_from_remainder` step writes housing and transport around their
  levelled parts (gate `recomposed_totals`).
- A8 `e3f2a3a58`: the litres audit labels its per-fuel ratios as a uniform cars share. The plan's
  quarterly 04.5.x and 07.2.2 vendoring is dropped, because no code would read it. B3 replaces the
  label.
- A7 `21a4d3a77`: `uk_diagnostics.consumption_drift` reports the stage-levelled totals and shares
  at design and final weights.
- A6 `ea433d1c5`: every LCFS spend column is at calendar-2024 prices, the H5's year. Energy is
  priced and levelled for calendar 2024, and bus yields are re-priced to 2024 by fares index. The
  gate `consumption_basis` holds every level-setting step to that year, so policyengine-uk can
  project each column from the H5's year with no basis of its own. This replaces the plan's
  per-column declaration (María's ruling of 2026-10-09).
- B1 `2232d9117`: the feed re-pins to Chronicle `1ee7dfe` (#323, 352,549 rows). The re-pin moves
  no compiled target value on either surface and leaves the three compile-parity receipts as they
  were. (A4's new row had already added one ledger-only row to each national receipt.) The seven
  Consumer Trends series already in the feed come back on ONS's domestic concept under new keys
  with the same values.
- B2 `543533a5d`: the stage receipt reports each division's capture of ONS household spending
  (`division_capture`, report-only).
- B3 `59339e4ab`: the litres audit benchmarks each fuel against HMRC clearances times DESNZ's cars
  share of that fuel's road use.
- `3ed090d03`: the charter-H2 spine fixture, regenerated at Group B's head. In the graph only the
  stage contract hashes move, for `lcfs_consumption` and the ten stages that declare a re-vendored
  resource. Each review commit that moves the fixture regenerates it.
- Review round 1 (`352310965`) and review round 2: see the two sections at the end.

## microcosm#1121's head calibration

Arm `head-r3` (tree `c7fe76e8a`, ledger feed `825406f` rebuilt; loss 0.00712, 98.3% of 1,167 rows
within 10%), before this branch:

- `obr.vat` reads +24.7% at design weights and +24.1% calibrated, against the 25% bound.
- `obr.fuel_duties_cars` reads -34.5% at design weights and -17.7% calibrated.
- The ONS electricity and gas rows start at +9.4% and +23.4% and close through the weights.
- From design to final weights, with every row bound, petrol moves +21% and diesel +34%, the
  gas-connected share goes from 84.5% to 78.9%, and fuel-car households with no fuel from 34.3% to
  30.0%. The 2026-09-30 build reads the same 84.5% and 78.9%: the design share is the DESNZ meter
  share the stage imposes, and the same gas row pulls it.

## Why the levels belong to the stage

Measured on the 2026-09-30 national build (`c5a1cba87`). The scripts and their aggregate outputs
are in `docs/evidence/uk-consumption-taxes-1113/`.

- **Energy volumes (M1).** At design weights the frame holds 93.0 TWh of electricity and
  260.1 TWh of gas, equal to Energy Trends, with 84.5% of households connected to gas. At final
  weights it holds 82.9 and 205.2 TWh with 78.9% connected.
- **Energy sources (M2).** For calendar 2025, ONS 04.5.1 electricity is GBP 26.52bn, 0.90 of
  Energy Trends volume times QEP unit and standing charges (GBP 29.60bn) and 1.11 of unit cost
  alone. ONS 04.5.2 gas is GBP 15.34bn, 0.81 of GBP 18.94bn and below unit cost alone (0.955). Meters
  against households explains about 2%. Coverage, period and VAT cannot close a gap where the
  household-sector total sits below the unit cost of the measured volume, so the ONS rows are a
  published-source residual, not a weight question.
- **Road fuel (M3).** ONS 07.2.2 is GBP 37.025bn for calendar 2024 and GBP 35.099bn for 2025. The
  LCFS diary records about a quarter less; the frame matched the donor per household (GBP 966 against
  GBP 972). Once fuel is levelled, the engine's 2025 `fuel_duty` reads -9.9% against OBR cars on its
  RAC price divisors and -8.1% on DESNZ's. ONS 07.2.2 at DESNZ 2025 pump prices implies about
  GBP 13.5bn of duty; the OBR cars line, which includes business-paid car fuel, is about 7% above it. About a
  third of fuel-car households carry no annual fuel (32.4% of the donor's car households record
  none in the two-week diary).
- **VAT.** `vat` divides household VAT by `gov.simulation.microdata_vat_coverage` (0.38, a constant
  since policyengine-uk PR #690, January 2023, with no recorded derivation; policyengine-uk#1996).
  The frame's raw household VAT is about 46% of OBR receipts, so the factor sets the level.

## Masked re-solve of the 2026-09-30 problem

M4 re-solved the stored national problem (1,090 rows, not the 1,167 of #1121's head) with the
doctrine options (1,500 epochs, lr 0.02, seed 0, `max_weight_ratio` 10, loss cap 10, mass free):
once with every row bound, and once with the four rows masked together. The bound arm reproduced
the stored weights exactly. No arm masks one row alone, and the problem predates this branch's
stage levels.

- With the four rows masked together, the other 1,086 rows fit no worse: 98.5% within 10% against
  98.4%, none past 25% against one, and ESS +3.7%.
- What binding the four costs the frame, as their joint effect: total consumption 5% (GBP 55bn)
  lower, petrol 18% and diesel 36% higher, gas 19% lower (GBP 15.35bn against GBP 18.97bn) and the
  gas-connected share at 78.9% against 84.6%.
- Unbound, VAT reads +35% and OBR cars -27%, so both are signed register entries, not rows left
  under the target-fit gate.

## Stage-only development run

`run_stage_dev.sh` re-runs the committed `lcfs_consumption` stage on the finished head-r3 spine
(63,956 households, about 50 seconds and 4.6 GB) and runs every lcfs stage-health gate on its
receipts. The spine already holds the stages after `lcfs_consumption`, so the before and after
columns differ by the re-run as well as by this branch. All seven gates pass: support,
energy_rake, bus_pricing, road_fuel_incidence, road_fuel_level, recomposed_totals and
consumption_basis.

- **Incidence.** The chain left 16,362 of the 48,529 flagged fuel-car households at zero fuel
  (33.7% of rows, 33.2% of their weight); after the redraw none are. The redraw's mean is GBP 1,794 against GBP 1,880 for households
  that drew fuel, with a 73.8% petrol share, from a model fitted on the 2,427 donors with positive
  road fuel.
- **Level.** The redraw raises prior-weighted petrol plus diesel from GBP 26.89bn to GBP 39.63bn,
  and the level step scales it by 0.931 to GBP 36.887bn: the published GBP 37.025bn less the donor's
  0.372% other-fuels share. Petrol is 68.4% of it. Annualised two-week spend runs high for the
  households that did buy fuel, and the level step corrects that as well.
- **Recomposed totals.** Housing is its draw (GBP 180.1bn), less the drawn electricity and gas
  (GBP 56.5bn), plus DESNZ times QEP energy (GBP 48.8bn): GBP 173.5bn. Transport is its draw
  (GBP 173.0bn), less the drawn petrol and diesel (GBP 28.7bn), plus the levelled fuel
  (GBP 36.9bn): GBP 181.5bn. Remainders floor at zero for 2,368 and 1,240 households
  (GBP 1.1bn and 0.3bn). No household's total is below its parts, against 2.7% (energy) and
  1.8% (fuel) before. The chain's domestic-energy target is the electricity and gas the stage
  derives from the diary: GBP 54.9bn on the donor, against GBP 60.1bn for all of its fuel, light
  and power. No column carries the liquid and solid fuels of COICOP 04.5, and the housing
  remainder holds only part of them (review round 2 corrects an earlier claim that it held them
  all). The receipt records their published spend (ONS 04.5.3 and 04.5.4, GBP 1.547bn for 2024).
- **Totals.** Consumption across the twelve divisions reads GBP 971.4bn. The spine's stored
  columns read GBP 979.0bn, but that -0.8% is mostly re-run drift, not this branch: against the
  base stage re-run on the same spine it is +0.2% (review round 2). Electricity is GBP 29.5bn and gas GBP 19.2bn at calendar-2024 prices, with 84.6% of
  households connected to gas. The litres audit's total ratio, frame over the cars benchmark, is
  0.951.

### Calendar 2024 for every spend column

policyengine-uk projects every column from the H5's year, 2024, with calendar-year growth, and the
calibration uprates the same values to the calendar-2025 window. Two levels were on FY2024-25:
energy and the bus yields. A6 moves them to calendar 2024:
- Energy uses QEP's calendar-2024 averages paid and the DESNZ volume over the four quarters of
  2024. Against the FY2024-25 basis, electricity moves +1.75% (GBP 29.02bn to 29.52bn) and gas
  +1.58% (GBP 18.93bn to 19.23bn).
- Each area's bus yield is re-priced by its fares index. London's fares were flat through
  2024-25, so its factor is 1.000. England outside London is 0.965, because the fare cap rose in
  January 2025. Scotland is 0.988 on its calendar-year index, and Northern Ireland 0.982 on
  England's. Bus fares move -2.05% (GBP 4.19bn to 4.10bn).
- Road fuel was already levelled to calendar 2024. The diary columns were not: a full year's CPI
  step left the April-to-March diary a quarter past it, which review round 2 corrects.

Housing moves -0.5%: the drawn electricity and gas it nets rise with the donor's re-priced energy,
by more than the levelled energy it adds back.

### Why the remainders are not chain targets

The plan's first version drew `housing_water_excluding_domestic_energy` and
`transport_excluding_road_fuel` as chain targets. On the same spine that took transport to
GBP 218.6bn and consumption to GBP 1,012.7bn (+3.4%). Drawn in-sample on the donor, the transport
remainder came out 13.0% and 10.0% above the donor total at seeds 0 and 1: the positive mean rose
from GBP 4,299 to 4,830 and 4,607. The parent came out 1.0% and 5.2% above. The QRF over-draws the
vehicle-purchase tail, which dominates the remainder once fuel is gone, and the spine's richer
households amplify it. The step therefore keeps each total's own draw and subtracts the chain's
draws of its parts, which the chain draws after, and conditional on, the total.

## Group B development run

The same stage-only run at B3 (`stage_dev_run.group-b.json`, 66 seconds): all seven gates pass, and
every Group A figure above is unchanged apart from the litres audit's total, which B3 re-bases.

### Division capture (B2)

Each division's prior-weighted frame total for calendar 2024, against ONS Consumer Trends for 2024
less the classes a household diary does not record: 02.3 narcotics, 04.2 owner-occupiers' imputed
rent and 12.6.1 FISIM (GBP 350.6bn together).

- The twelve divisions total GBP 971.4bn against GBP 1,334.6bn in survey scope: 72.8% of ONS's
  domestic concept and 72.0% of the national one. The donor's own diary totals GBP 780.7bn at
  2023-24 prices, 58.5% of the same scope. Both sides hold about the same households (29.0m and
  28.5m of weight), so the gap is in the draws per household, which condition on the frame's own
  incomes.
- Housing reads 1.059. Of its GBP 9.7bn excess, GBP 8.0bn is D1's energy level: DESNZ volume at
  QEP prices (GBP 48.8bn) against ONS 04.5.1 plus 04.5.2 for 2024 (GBP 40.7bn).
- Furnishings (05) read 1.251: the frame draws GBP 95.1bn against ONS's GBP 76.0bn and the
  donor's GBP 57.5bn, 63% more per household than the donor. Health (06) is 70% above the donor
  per household, though still 0.586 of ONS. The top 1% of households carry 15% of the frame's furnishings and 18%
  of its health spend. That is the shape of the tail over-draw that ruled out drawing the transport
  remainder as a chain target, but this run does not test it.
- Education reads 0.085, half the donor's own 0.177: ONS 10 includes tuition financed by student
  loans and international students' fees.
- Alcohol and tobacco (0.469), clothing (0.409) and restaurants and hotels (0.457) are the other
  low readings. The last also carries non-residents' spending on the domestic concept.

### Litres audit (B3)

For 2024 cars burn 97.8% of road petrol and 39.2% of road diesel (DESNZ ktoe). Against HMRC's
fiscal 2024-25 clearances times those shares, the frame's petrol litres read 1.011 and its diesel
litres 0.709. The total reads 0.894, against 0.951 on the OBR cars share of duty receipts (58.3%,
uniform across fuels). The level step keeps the donor's mix (petrol 68.4% of spend), so the diesel
shortfall is the diary's petrol-diesel split against DESNZ's cars.

## Review round 1 (2026-10-10)

Vahid's round-1 pass on PolicyEngine/microcosm#1167 found nothing blocking. The changes it led to,
with María's rulings of 2026-10-10:

- **Scope offset on the bound row (ruled: recorded, not scaled).** ONS 07.2.2 carries other motor
  fuels and lubricants, 0.37% of the class on the diary. The stage levels petrol plus diesel to
  the class less that share, and the target is the full class, so the row reads about 0.4% low
  from scope alone. The binding notes record it.
- **Concept of the bound row.** ONS publishes 07.2.2 on the domestic concept. It includes
  non-residents' fuel bought in the UK, which the residents-only frame does not hold, and leaves
  out residents' fuel bought abroad. Consumer Trends publishes those flows for total spending
  only: for 2024, GBP 64.7bn by non-residents in the UK (3.8% of the domestic total) and
  GBP 78.9bn by residents abroad. It splits them by no class, so their size for road fuel is not
  known and no bridge is applied.
- **Energy exclusions restated (approved).** The two register reasons named the FY2024-25 basis
  the stage left on 2026-10-09. They now name calendar 2024 and quote that year beside 2025: ONS
  04.5.1 is GBP 26.10bn against the stage's GBP 29.52bn (0.88), and ONS 04.5.2 is GBP 14.64bn
  against GBP 19.23bn (0.76). The 2025 ratios (0.90 and 0.81) were already calendar-year.
- **Model year.** policyengine-uk applies the fiscal 2024-25 rules to model year 2024, sampling
  each parameter at 30 April. The spend columns are at calendar-2024 prices, about a quarter
  before the midpoint of that policy year. The calibration binds calendar-year facts as published
  and fiscal-year facts at the calendar window (María's rule of 2026-09-22). The stage notes say
  so.
- **Vehicle electricity.** The diary's 07.2.2 has three lines (c72211, c72212, c72213) and none for
  vehicle electricity, so nothing joins the other-fuels share. The stage notes say so.

## Review round 2 (2026-10-11)

María's independent adversarial review of PolicyEngine/microcosm#1167 at `352310965`. The item
numbers are the review's.

### Stage changes

- **Diary columns on calendar 2024 (item 1).** A full year's CPI step (2.5%) carried the
  April-to-March 2023-24 diary to about October 2024, a quarter past the basis A6 claims, and
  `consumption_basis` checked only the target year. The twelve divisions and bus fares now
  declare the diary-year window. The engine's CPI index is calendar-year, so it places that year
  at three quarters of 2023 and one of 2024, and the step to calendar 2024 is 1.0186. The gate
  reads each column's declared window. Income predictors keep the full year's step, to the FRS
  2024-25 incomes they are matched against.
- **Transport holds the priced bus fares (item 6).** The recomposition nets the chain's drawn bus
  fares with the drawn petrol and diesel, and adds the priced bus fares back with the levelled
  fuel.
- **Frame-level bounds on the two totals (item 4).** Housing and transport are back in the
  committed support bounds (GBP -20,000 to 300,000 and GBP 0 to 200,000), which the terminal
  support gate applies to the emitted frame. On this run they reach GBP 119,000 and GBP 176,000.
  The `recomposed_totals` sums are identities on any receipt the step wrote; its gate note now
  says it guards the declaration and the receipt, not the size of the totals.
- **Report-only blocks cannot stop a build (item 12).** The division capture records why it has
  no figures when a Consumer Trends row is missing or ambiguous for the year. The litres audit
  falls back to the OBR cars share when DESNZ has no vehicle split for it. A vendored resource
  that lags the feed pin still fails.
- **Selectors (item 12).** The Consumer Trends targets pin the price basis, concept and seasonal
  adjustment PolicyEngine/chronicle#323 added, and the capture's row lookups pin the same variant.
  The references still resolve to the same facts.

### Measurements at this round's head

`stage_dev_run.review-r2.json`, on the same head-r3 spine. All seven gates pass.

- **Against the base stage.** `stage_dev_run.base-7f235941c.json` re-runs main's stage
  (`7f235941c`) on the same spine, which separates this branch from re-run drift. The drift is
  large: the base re-run reads GBP 969.6bn against the spine's stored GBP 979.0bn, GBP 8bn of it
  in transport.
  - Consumption moves from GBP 969.6bn to GBP 966.8bn (-0.3%).
  - The ten diary divisions move -0.7% (GBP 616.5bn to 612.3bn). The bridge is -0.6% of that;
    the rest is the redraw moving with the donor.
  - Housing moves from GBP 180.1bn to 172.1bn: its draw (GBP 178.7bn), less the drawn electricity
    and gas (GBP 56.5bn), plus the levelled energy (GBP 48.8bn). The floor adds GBP 1.1bn across
    2,438 households.
  - Transport moves from GBP 173.0bn to 182.4bn: its draw (GBP 172.6bn), less the drawn petrol,
    diesel and bus fares (GBP 31.9bn), plus the levelled fuel (GBP 36.9bn) and the priced bus
    fares (GBP 4.1bn). The floor adds GBP 0.6bn across 2,312 households.
  - Before this round the branch read +0.2% against the base (GBP 971.4bn): housing -6.6bn,
    transport +8.5bn, and the other ten divisions identical. The -0.8% quoted earlier compared
    with the stored spine (item 7).
- **Incidence and level.** 16,376 of the 48,529 flagged fuel-car households drew zero (33.7% of
  rows, 33.1% of their weight); none do after the redraw. The level factor is 0.931, and petrol is
  68.2% of the levelled GBP 36.887bn.
- **Division capture.** The twelve divisions are 72.4% of ONS's domestic concept in survey scope
  and 71.7% of the national one. Housing reads 1.051. Furnishings read 1.242, 61% more per
  household than the donor's diary, and health 0.585, 70% more.
- **Litres audit.** Petrol reads 1.008 of its cars benchmark and diesel 0.713. The total reads
  0.894, against 0.951 on the OBR cars share.
- **Energy tail.** The 39 households with a housing total above GBP 42,000 get there through
  electricity and gas of up to GBP 87,000 and GBP 47,000 a year. Energy values of that size
  predate this branch (the stored spine holds electricity up to GBP 76,000). The recomposition
  makes housing contain them; nothing bounds the energy columns themselves.
- **Totals contain their parts.** No household's housing total is below its electricity and gas,
  and none's transport total is below its petrol, diesel and bus fares.

### The bound row at design weights (item 9)

ONS's own move from 2024 to 2025 is -5.2% (GBP 37.025bn to 35.099bn). policyengine-uk 2.122.2
carries petrol and diesel spending -2.7%. With the 0.37% scope offset, the row reads about +2.3%
at design weights on that version (GBP 35.89bn against GBP 35.10bn). The frame sits above the
target, so the calibration closes the gap by moving weight away from fuel buyers. This is a desk
figure from the engine's indices; R1 records the measured one.

### Decisions recorded

- **Nothing release-blocking holds calibrated domestic energy, the VAT base or fuel duty
  (item 3).** That follows from D1 to D3. The stage gates hold the energy and fuel levels at prior
  weights, the bound ONS row holds household road-fuel spend, and the report-only
  `consumption_drift` shows the rest. The gate notes and the two older signed exclusions
  (`obr.fuel_duties`, COICOP 04.5) now say so; they still described the rows as bound. A
  release-blocking limit on the drift is a possible follow-up once R1 has measured it.
- **The fuel flag (item 5).** The flag is a draw at the non-zero-emission share of the car stock,
  keyed on the household identity. It is the stage's own designation of who buys fuel: an
  unflagged vehicle owner is the zero-emission household and has no fuel, before and after this
  branch, so the redraw gives fuel to no household the stage holds as electric. What stands is
  that the designation is random with respect to income and is all-or-nothing per household. The
  stage notes record that as a known residual.
- **Other fuels (item 12).** The diary's fuel, light and power (`p537`) is GBP 60.1bn on the
  donor; the electricity and gas the stage derives from it are GBP 54.9bn. For 2,978 of the 4,202
  diaries the derivation splits all of `p537`, so their liquid and solid fuels leave with the
  drawn energy. For the 1,224 that report a combined payment, the rest of `p537` (GBP 5.2bn)
  stays in the housing remainder. The earlier claim that the remainder holds all the other fuels
  was wrong.
- **Northern Ireland's bus index (item 12).** Its factor (0.982) is on England's index, a declared
  proxy that carries the January 2025 rise in the English fare cap. It is left in place and the
  stage notes say so.
- **Exclusion reasons (item 8).** The four reasons give design-to-final moves and the masked-arm
  comparison separately, and say the four rows were masked together on the 2026-09-30 problem.
- **The repeated 84.5% to 78.9% (item 12).** Not a copy error. Both builds were measured from
  their files: #1121's head reads 84.54% and 78.87%, the 2026-09-30 build 84.52% and 78.93%.

### Follow-up issue (item 2)

Three of the new exclusions tracked #1113, which the PR closes, and the two energy rows retire on
a reconciliation this branch did not do. On María's call the open items are one follow-up issue,
PolicyEngine/microcosm#1207, under the two deferral registers: the calibration-target items under
#736 and the spine items under #796. The three exclusions now track #1207, and the PR still closes
#1113.

## Not run

- The national remeasure (R1) runs on whatever policyengine-uk Microcosm locks then. If the VAT,
  fuel-duty and energy engine fixes are released by then, the lock moves first in its own PR.
  R1 also reads `ons.household_road_fuel_expenditure` at design weights and records its total gap.
  PolicyEngine/microcosm#1207 carries R1 and its acceptance.
