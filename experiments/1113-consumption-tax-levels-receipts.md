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
- B1 `2232d9117`: the feed re-pins to Chronicle `1ee7dfe` (#323, 352,549 rows). No compiled target
  value moves on either surface, and the three compile-parity receipts are unchanged. The seven
  Consumer Trends series already in the feed come back on ONS's domestic concept under new keys
  with the same values.
- B2 `543533a5d`: the stage receipt reports each division's capture of ONS household spending
  (`division_capture`, report-only).
- B3 `59339e4ab`: the litres audit benchmarks each fuel against HMRC clearances times DESNZ's cars
  share of that fuel's road use.
- `3ed090d03`: the charter-H2 spine fixture, regenerated once at the head. In the graph only the
  stage contract hashes move, for `lcfs_consumption` and the ten stages that declare a re-vendored
  resource.

## microcosm#1121's head calibration

Arm `head-r3` (tree `c7fe76e8a`, ledger feed `825406f` rebuilt; loss 0.00712, 98.3% of 1,167 rows
within 10%), before this branch:

- `obr.vat` reads +24.7% at design weights and +24.1% calibrated, against the 25% bound.
- `obr.fuel_duties_cars` reads -34.5% at design weights and -17.7% calibrated.
- The ONS electricity and gas rows start at +9.4% and +23.4% and close through the weights. That
  moves petrol +21% and diesel +34%, the gas-connected share from 84.5% to 78.9%, and fuel-car
  households with no fuel from 34.3% to 30.0%.

## Why the levels belong to the stage

Measured on the 2026-09-30 national build (`c5a1cba87`); outputs in
`data/ukds/acceptance/vat-fuel-energy-measure-2026-10-08/`.

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

M4 re-solved the stored national problem with the doctrine options (1,500 epochs, lr 0.02, seed 0,
`max_weight_ratio` 10, loss cap 10, mass free), with and without the four rows. The baseline arm
reproduced the stored weights exactly.

- With the four rows masked, the other 1,086 rows lose nothing: 98.5% within 10% against 98.4%,
  none past 25% against one, and ESS +3.7%.
- What the rows cost the frame: VAT held total consumption 5% (GBP 55bn) low; the cars fuel row
  pushed petrol up 18% and diesel 36%; the gas row cut gas 19% and the gas-connected share to 78.9%.
- Unbound, VAT reads +35% and OBR cars -27%, so both are signed register entries, not rows left
  under the target-fit gate.

## Stage-only development run

`run_stage_dev.sh` re-runs the committed `lcfs_consumption` stage on the finished head-r3 spine
(63,956 households, about 50 seconds and 4.6 GB) and runs every lcfs stage-health gate on its
receipts. The spine already holds the stages after `lcfs_consumption`, so the before and after
columns differ by the re-run as well as by this branch. All seven gates pass: support,
energy_rake, bus_pricing, road_fuel_incidence, road_fuel_level, recomposed_totals and
consumption_basis.

- **Incidence.** The chain left 33.2% of the 48,529 flagged fuel-car households (16,362) at zero
  fuel; after the redraw none are. The redraw's mean is GBP 1,794 against GBP 1,880 for households
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
  1.8% (fuel) before. The chain's domestic-energy target is the diary's electricity plus gas, so
  the liquid and solid fuels of COICOP 04.5 have no column of their own and stay inside the
  housing remainder at the diary's level (about GBP 5bn on the donor, with the split's residue);
  the receipt records their published spend beside it (ONS 04.5.3 and 04.5.4, GBP 1.547bn for
  2024).
- **Totals.** Consumption across the twelve divisions moves from GBP 979.0bn to GBP 971.4bn
  (-0.8%). Electricity is GBP 29.5bn and gas GBP 19.2bn at calendar-2024 prices, with 84.6% of
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
- The diary columns and road fuel were already on calendar 2024 and don't move.

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
  donor's GBP 57.5bn, 66% more per household than the donor. Health (06) is 73% above the donor,
  though still 0.586 of ONS. The top 1% of households carry 15% of the frame's furnishings and 18%
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

## Not run

- The national remeasure (R1) runs on whatever policyengine-uk Microcosm locks then. If the VAT,
  fuel-duty and energy engine fixes are released by then, the lock moves first in its own PR.
