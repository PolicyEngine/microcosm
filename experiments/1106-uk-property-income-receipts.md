# microcosm#1106 receipts: UK property income and property wealth

Aggregates and counts of 10 or more only. Scripts and licensed outputs live in `data/ukds/acceptance/1106-property-income/`.

## A. SPI net property-income amounts bound

Regenerated from the pinned Chronicle feed 825406f on #1121's head `c098fdcc4` (policyengine-uk 2.122.2).

- The 13 SPI Table 3.7 amount rows by total-income band compile to £31.78bn for 2025: the 2023-24 net amounts (£29.36bn) times the pinned per-capita GDP ratio 1.0826.
- The incumbent fixture holds £55.78bn for the same rows, 1.9 times the SPI amounts. Example bands: £50,000 to £70,000 is £5.85bn against £10.26bn; £100,000 to £150,000 is £2.76bn against £4.85bn.
- Active national references: 1,231 → 1,244 (13 added, none changed). Signed-out references: 16 → 15. The local surface is unchanged.
- #1121's re-derived uprating pins at 2.122.2 leave the per-capita GDP ratio, and so these values, exactly as on main.
- The 13 new specs move the compiled register, so uk-candidate-eval's frozen scoring register needs a re-freeze before the next national solve.

## B. The WAS property total kept off the engine

Measured with `scripts/measure_property_wealth_drop.py` on the A9 national candidate (main after #1100, calibrated weights, 63,941 households, 29.22m weighted), running policyengine-uk 2.100.0 at 2025 twice: on the frame as the release wrote it before this PR, and with the release-export drops applied.

| Total, 2025 | Persisted WAS total | Engine-derived | Change | Target |
|---|---|---|---|---|
| `property_wealth` | £7,469.5bn | £7,633.4bn | +2.2% | none |
| `household_land_value` | £4,595.0bn (+0.8%) | £4,697.3bn (+3.0%) | +2.2% | £4,559.8bn |
| `land_value` | £6,653.6bn (−6.5%) | £6,755.9bn (−5.1%) | +1.5% | £7,117.8bn |
| `total_wealth` | £15,030.7bn | £15,194.6bn | +1.1% | none |

- With the column dropped, the engine's `property_wealth` equals the sum of the three persisted components on every household.
- With it persisted, the saved 2024 total is below the 2025 component sum for 45,232 of 63,941 households, and below the 2025 main residence value for 34,731 of 47,046 owners: the saved total is not uprated while its parts are.
- By region, `household_land_value` moves between −2.6% (Wales) and +3.7% (West Midlands); Northern Ireland −0.2%, London +2.8%. `corporate_land_value` does not move.
- These are measurements at fixed A9 weights. A calibration of this head reweights against the land targets, so the head-to-head against the incumbent (rule 1, `direct_h5` on a pre-#1106 H5) shows the land rows moving for the fix, not the weights.

## C3 feasibility: landlords' receipts from profit (before implementation)

Measured with `scripts/receipts_schedule_feasibility.py` on the A9 candidate's landlords (9,759 records with `property_income` > 0, 2.820m weighted, £29.99bn profit at the 2024 base), against PRIS 2024-25: individuals' receipts £49.81bn, allowable expenses other than residential finance costs about £18.9bn, and Table 13's landlord counts by receipts band scaled to individuals (2.85m of 2.88m).

- **A constant expense share of 0.40** gives receipts of £49.98bn and expenses of £19.99bn, but leaves the spine's lumps in the band counts: 0.073m landlords at £60,000 to £70,000 against PRIS's 0.030m, and 0.029m at £80,000 to £90,000 against 0.010m.
- **A schedule matched on quantiles** maps each landlord's profit rank to the same rank of PRIS's receipts distribution and keeps receipts at least equal to profit. It reproduces all eleven band counts within 0.016m (the largest gap is £10,000 to £20,000: 0.835m against 0.851m). Its implied expense share has median 0.39, with 80% of landlords between 0.33 and 0.45. Its totals run above PRIS (receipts £51.46bn, expenses £21.47bn) because the top band's spread is assumed; the stage fits that spread to the totals.
- So decision 7's band schedule is feasible with plausible shares. These figures use calibrated A9 weights; the stage runs on design weights, and the PRIS targets in C4 then hold the fit through calibration.
