# LCFS 2024-25 and the consumption-stage fixes: measurement scripts and outputs

Evidence for microcosm#1164: the `lcfs_consumption` stage moved from the Living Costs and Food Survey 2023-24 (UKDS SN 9468) to 2024-25 (SN 9603, 2nd edition), with the fixes that ship with it. `receipts.json` carries the digests of the files here and of the spine the runs measured.

## What was measured

- **Stage-only runs** (`scripts/stage_dev_run.py`, `scripts/run_stage_dev_at.sh`). Each tree's own `lcfs_consumption` stage, on the LCFS tabs its own spec pins, re-run on one finished K25 spine (about 64,000 households, built on 2026-10-10 with policyengine-uk 2.122.2, before #1145's `property_components` stage), with every lcfs stage-health gate run on its receipts:
  - `stage_dev_run.control-2023-24-final.json`: the base tree, #1145's head (63943a76b), on LCFS 2023-24;
  - `stage_dev_run.treatment-2024-25-final.json`: this branch, on LCFS 2024-25.
- **The NTS-then-LCFS chain** (`scripts/nts_lcfs_chain_run.py`). `nts_bus_travel` then `lcfs_consumption` on the same spine, because the LCFS stage prices bus fares from the NTS trips: `nts_lcfs_chain_run.control.json` and `nts_lcfs_chain_run.treatment.json`.

Both arms run the same spine and policyengine-uk 2.123.0, so a run's comparison with the other arm isolates this branch's stage. The spine already holds the stages after the LCFS stage, so compare the arms, not a run with the spine's stored columns (`before_spine`), which mixes in redraw drift. The R1 remeasure on fresh spines follows the merge of #1198 and #1200.

## Results

Every LCFS and NTS stage-health gate passes on both arms, and no household is left outside a committed support bound. Prior-weighted annual totals, £bn:

| Column | Control (LCFS 2023-24) | This branch (LCFS 2024-25) |
|---|---|---|
| `food_and_non_alcoholic_beverages_consumption` | 109.6 | 113.8 |
| `alcohol_and_tobacco_consumption` | 21.8 | 19.7 |
| `clothing_and_footwear_consumption` | 32.2 | 30.1 |
| `housing_water_and_electricity_consumption` | 171.4 | 179.2 |
| `household_furnishings_consumption` | 94.9 | 80.2 |
| `health_consumption` | 19.5 | 23.5 |
| `transport_consumption` | 185.5 | 187.2 |
| `communication_consumption` | 24.6 | 25.5 |
| `recreation_consumption` | 139.5 | 183.0 |
| `education_consumption` | 4.6 | 20.3 |
| `restaurants_and_hotels_consumption` | 83.1 | 85.4 |
| `miscellaneous_consumption` | 84.3 | 97.4 |
| Twelve divisions | 970.8 | 1,045.2 |
| `electricity_consumption` | 29.5 | 29.6 |
| `gas_consumption` | 19.2 | 19.1 |
| `petrol_spending` | 25.5 | 26.1 |
| `diesel_spending` | 11.4 | 10.7 |
| `bus_fare_spending` | 4.1 | 4.1 |

- **The diary year.** The 2024-25 diary records more spending than 2023-24: its own weighted twelve-division total rises about 7 % at base-year prices, with health (+58 %), education (+74 %) and recreation (+14 %) leading in current prices. The division capture against ONS Consumer Trends 2024 (domestic concept less the classes a diary does not record) rises from 0.73 to 0.78 overall. Furnishings moves from 1.25 to 1.05; recreation from 0.85 to 1.12, above ONS, and education from 0.11 to 0.48. The chain's draws run above the donor's own weighted totals on several divisions (recreation and education about 1.5 times), the over-draw follow-up.
- **The bridge.** The diary columns now step a quarter back to calendar 2024 (factor below one) where they stepped three quarters forward, and income predictors take a factor of one. The recomposition's uprating residue falls from £1.05bn (housing) and £1.92bn (transport) to −£0.46bn and −£0.24bn.
- **Energy.** The labelled energy split reads the donor's electricity at £30.4bn and gas at £20.9bn a year, with £3.2bn of other fuels (oil, bottled gas, solid fuels) left in the housing remainder; its parts sum to 1.4 % above the diary's fuel, light and power (P537). The old split gave gas to 98 % of donors and electricity to 92-94 %; the labelled one gives gas to about 80 % and electricity to 98 %. The DESNZ level holds electricity and gas totals on both arms.
- **Gas connection.** The frame's gas-connected share falls from 84.6 % to 79.2 %. The old split's near-universal gas let the connection walk disconnect down to every region's published DESNZ share; the labelled split's donors pay for gas less often than DESNZ counts meters, so nine of the eleven Great Britain regions now fall short of their share (receipted, as the gate allows), most in the North East, East Midlands, North West and Yorkshire. A follow-up connects the shortfall.
- **The NEED rake.** Recipients band on HBAI gross income. The rake converges at 2.40 % electricity and 2.82 % gas in the income margin, 2.88 % gas after the household ceilings, against 0.6 % on the control's HBAI-net band; the `energy_rake` gate's rule is 3.5 % (María's ruling of 2026-10-12). Under the household ceilings the gas cap binds on about 0.35 % of gas kWh and electricity's on none.
- **The NTS band.** On HBAI gross income the recipients' person-weighted NTS income bands (under £25k, £25-50k, £50k and over) move from 6 / 21 / 73 % to 13 / 30 / 57 %; London bus trips rise 1.6 % and other local bus trips fall 1.0 %, and bus fares are within 1 %.
- **Prepayment meters.** The donor's weighted prepayment share is 11.0 %; the frame's is 11.5 % (10.5 % in Great Britain, about 3.0 million households).
- **Road fuel.** The incidence redraw leaves no flagged household at zero and the level holds ONS 07.2.2 less the donor's other-fuels share.

## Decisions behind the energy changes

- **Which income bands the NEED rake.** The engine's `household_gross_income` adds capital gains and the imputed value of NHS, education and transport spending, about £15,000 a household on average, and missed NEED by 7.8 %. HBAI gross income (the HBAI net income with policyengine-uk's HBAI deductions added back) has a distribution close to the LCFS donor's gross income (weighted medians about £46k and £49k). HBAI net income leaves the two top NEED bands thin (about 6 % of households against about 14 % on HBAI gross), so the rake drives about 2 % of gas kWh above the domestic meter limit and the ceilings then leave those cells 3.2 % short.
- **How the ceilings hand out what they take.** Handed to every household in proportion, the kWh capped households give up moved the cells holding most of them off their NEED means (3.2 % gas on the HBAI-gross band); re-raked within the capped households' own cells, it moves the fit 0.06 points.

## What is not here

- **Unit records.** The household dumps the runs write (`*.households.parquet`) stay outside the repository, with the diagnostic runs behind the decisions above.
- **Licensed inputs.** The LCFS and NTS tabs and the spine.
- **Small cells and extremes.** `scripts/sanitize_outputs.py` drops exact donor extremes, remainder minima and the donor-floor negative mass, writes a count of one to nine households "<10" and drops the weighted quantity those households carry, and keeps each NEED margin's largest deviation rather than its cells.

## Re-running

The runs need the licensed LCFS 2023-24 and 2024-25 tabs, the NTS 2002-2024 tabs and the spine:

```bash
scripts/run_stage_dev_at.sh <tree> <label> [stage_dev_run.py | nts_lcfs_chain_run.py]
```

It runs `<tree>`'s own stage code with this worktree's environment, waits up to 15 minutes for a quieter machine, and writes `<script>.<label>.json` and the household dump to `$PE/data/ukds/acceptance/1164-lcfs-2024-25/`. A stage-only run takes about 90 seconds and 7 GB; the chain about 150 seconds.

## Provenance

Paths inside the scripts and outputs are machine-local. The scripts are copies of the ones run, formatted by this repository's formatter.
