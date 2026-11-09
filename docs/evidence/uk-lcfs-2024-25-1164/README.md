# LCFS 2024-25 and the consumption-stage fixes: measurement scripts and outputs

Evidence for microcosm#1164: the `lcfs_consumption` stage moved from the Living Costs and Food Survey 2023-24 (UKDS SN 9468) to 2024-25 (SN 9603, 2nd edition), with the fixes that ship with it. `receipts.json` carries the digests of the files here and of the spine the runs measured.

## What was measured

- **Stage-only runs** (`scripts/stage_dev_run.py`, `scripts/run_stage_dev_at.sh`). Each tree's own `lcfs_consumption` stage, on the LCFS tabs its own spec pins, re-run on one finished K25 spine (about 64,000 households, built on 2026-10-10 with policyengine-uk 2.122.2, before #1145's `property_components` stage), with every lcfs stage-health gate run on its receipts:
  - `stage_dev_run.control-2023-24-final.json`: the base tree, #1200's head (a50ffba5e, which carries #1145 and #1198), on LCFS 2023-24;
  - `stage_dev_run.treatment-2024-25-final.json`: this branch, on LCFS 2024-25.
- **The NTS-then-LCFS chain** (`scripts/nts_lcfs_chain_run.py`). `nts_bus_travel` then `lcfs_consumption` on the same spine, because the LCFS stage prices bus fares from the NTS trips: `nts_lcfs_chain_run.control.json` and `nts_lcfs_chain_run.treatment.json`.
- **The ETB VAT stage** (`scripts/etb_vat_dev_run.py`). `etb_vat` on the same spine, each tree's own stage on the ETB tab its spec pins, with the stage's support gate: `etb_vat_dev_run.control.json` and `etb_vat_dev_run.treatment.json`.

Both arms run the same spine and policyengine-uk 2.123.0, so the comparison isolates this branch's stages. The spine already holds the stages after the LCFS stage, so compare the arms, not a run with the spine's stored columns (`before_spine`), which mixes in redraw drift. The R1 remeasure on fresh spines follows.

## Results

Every LCFS and NTS stage-health gate passes on both arms, and no household is left outside a committed support bound. Prior-weighted annual totals, £bn:

| Column | Base (LCFS 2023-24) | This branch (LCFS 2024-25) |
|---|---|---|
| `food_and_non_alcoholic_beverages_consumption` | 109.6 | 113.8 |
| `alcohol_and_tobacco_consumption` | 21.8 | 19.7 |
| `clothing_and_footwear_consumption` | 32.2 | 30.1 |
| `housing_water_and_electricity_consumption` | 171.4 | 179.5 |
| `household_furnishings_consumption` | 94.9 | 80.2 |
| `health_consumption` | 19.5 | 23.5 |
| `transport_consumption` | 185.3 | 187.2 |
| `communication_consumption` | 24.6 | 25.5 |
| `recreation_consumption` | 139.5 | 183.0 |
| `education_consumption` | 4.6 | 20.3 |
| `restaurants_and_hotels_consumption` | 83.1 | 85.4 |
| `miscellaneous_consumption` | 84.3 | 97.4 |
| Twelve divisions | 970.6 | 1,045.6 |
| `electricity_consumption` | 29.5 | 29.6 |
| `gas_consumption` | 19.2 | 19.2 |
| `petrol_spending` | 25.6 | 26.1 |
| `diesel_spending` | 11.3 | 10.7 |
| `bus_fare_spending` | 4.1 | 4.1 |

- **The diary year.** The 2024-25 diary records more spending than 2023-24: its own weighted twelve-division total rises about 7 % at base-year prices, with health (+58 %), education (+74 %) and recreation (+14 %) leading in current prices. The division capture against ONS Consumer Trends 2024 (domestic concept less the classes a diary does not record) rises from 0.73 to 0.78 overall. Furnishings moves from 1.25 to 1.05; recreation from 0.85 to 1.12, above ONS, and education from 0.11 to 0.48. The chain's draws run above the donor's own weighted totals on several divisions (recreation and education about 1.5 times), the over-draw follow-up.
- **The bridge.** The diary columns now step a quarter back to calendar 2024 (factor below one) where they stepped three quarters forward, and income predictors and the litre proxies take a factor of one. The recomposition's uprating residue falls from £1.05bn (housing) and £1.91bn (transport) to −£0.46bn and −£0.24bn.
- **Energy.** The labelled energy split reads the donor's electricity at £30.1bn and gas at £20.7bn a year, with £3.2bn of other fuels (oil, bottled gas, solid fuels) left in the housing remainder. Its parts sum to 0.6 % above the diary's fuel, light and power (P537): 0.2 points are refunds read as zero, and 0.4 points are the unclipped parts differing from P537, for 8 % of donors (weighted) by more than a penny a week and for 1 % by more than 50p. The account amounts (b489, b490) equal the payment less the rebate for at least 98 % of each fuel's account payers. The old split gave gas to 98 % of donors and electricity to 92-94 %; the labelled one gives gas to about 80 % and electricity to 98 %. The DESNZ level holds the electricity and gas totals on both arms.
- **Gas connection.** With donors paying for gas less often than DESNZ counts gas meters, the declared connect leg fills every Great Britain region to its published share (the frame's gas-connected share is 83.7 % against the base's 84.6 %; Northern Ireland, with no published share, keeps its draw).
- **The NEED rake.** Recipients band on HBAI gross income. The rake converges at 2.40 % electricity and 2.63 % gas in the income margin (gas: accommodation 1.40 %, tenure 0.71 %, region 0.11 %) against the `energy_rake` rule of 3.5 % (María's ruling of 2026-10-12); the base, banded on HBAI net income, converges at 0.6 %. The household ceilings bind on 2.6 % of gas kWh and on no electricity. That share moves with the forest's draw of the gas tail, not with the donor: the previous head's split gives 0.5 % with this head's code, and this head with the chain seed moved by one gives 1.4 %, while the donor's raked gas moves by under 2 % for all but about 200 donors (those whose combined bill nets a rebate or whose account amount differs from the payment less the rebate). Every gate passes on all three draws, and the gas income-margin residual stays between 2.6 % and 2.8 %.
- **The NTS band.** On HBAI gross income the recipients' person-weighted NTS income bands (under £25k, £25-50k, £50k and over) move from 6 / 21 / 73 % to 13 / 30 / 57 %; London bus trips rise 1.6 % and other local bus trips fall 1.0 %, and bus fares are within 1 %.
- **Prepayment meters** (microcosm#1110). The donor's weighted prepayment share is 11.0 %; the frame's is 11.5 %, and 10.5 % in Great Britain, about 3.0 million households. The government counted around three million prepayment households in Great Britain when the Energy Price Guarantee prepayment discount began in July 2023: a different reference period, households in both (Ofgem and DESNZ count meters or customers).
- **Road fuel.** The incidence redraw leaves no flagged household at zero, and the level holds ONS 07.2.2 less the donor's other-fuels share.
- **The ETB VAT rate.** Drawn on the ETB's own disposable-income concept, at identity-keyed uniforms and within [0, 1] (microcosm#1225), the weighted mean full-rate VAT expenditure rate moves from 0.401 to 0.398, and its profile by quintile of HBAI net income from 0.358 / 0.412 / 0.413 / 0.416 / 0.405 to 0.354 / 0.398 / 0.404 / 0.415 / 0.419: it now rises with income, where the engine's net income, inflated by in-kind spending, dipped in the top quintile. No recipient keeps a negative or zero rate. The support gate passes on both trees.

## Decisions behind the energy changes

- **Which income bands the NEED rake.** The engine's `household_gross_income` adds capital gains and the imputed value of NHS, education and transport spending, about £15,000 a household on average, and missed NEED by 7.8 %. HBAI gross income (the HBAI net income with policyengine-uk's HBAI deductions added back) has a distribution close to the LCFS donor's gross income (weighted medians about £46k and £49k). HBAI net income leaves the two top NEED bands thin (about 6 % of households against about 14 % on HBAI gross), so the rake drives about 2 % of gas kWh above the domestic meter limit and the ceilings then leave those cells 3.2 % short.
- **How the ceilings hand out what they take.** Handed to every household in proportion, the kWh capped households give up moved the cells holding most of them off their NEED means (3.2 % gas on the HBAI-gross band); re-raked within the capped households' own cells, it moves the fit 0.06 points.
- **Which income the ETB VAT rate is drawn on.** On the K25 spine, against the ETB 2023 donor's `disinc` carried to 2024 on average earnings, the engine's `household_net_income` puts 12.8 % of recipients' mass in the wrong donor income decile (median about £47k against the donor's £39k), HBAI net income 4.1 % and the ETB's own concept, gross cash income less income tax, National Insurance and council tax, 3.0 %.
- **How the shortfall is connected.** A gradient-boosted classifier of the donor's gas payers orders the zero-gas households a region connects (a weighted identity-keyed order, so likelier payers go first), and a regime-gated QRF of the payers' gas spend gives each its gas. Without the connect leg nine of the eleven Great Britain regions fell short of their share and the frame's gas-connected share was 79.2 %.

## What is not here

- **Unit records.** The household dumps the runs write (`*.households.parquet`) stay outside the repository, with the diagnostic runs behind the decisions above.
- **Licensed inputs.** The LCFS and NTS tabs and the spine.
- **Small cells, exact counts and extremes.** `scripts/sanitize_outputs.py` drops exact donor extremes, remainder minima and the donor-floor negative mass. It writes a count of one to nine households "<10" and drops the weighted quantity those households carry, rounds every other count (to the nearest 10 under 1,000, the nearest 100 above), bands a count of rows picked out by an extreme value ("10-49", "50-99", then the nearest 100), and keeps each NEED margin's largest deviation rather than its cells. The stage receipts band their own donor counts too (`uk_runtime/receipt_counts.py`).

## Re-running

The runs need the licensed LCFS 2023-24 and 2024-25 tabs, the NTS 2002-2024 tabs, the ETB 1977-2024 household tab and the spine:

```bash
scripts/run_stage_dev_at.sh <tree> <label> [stage_dev_run.py | nts_lcfs_chain_run.py | etb_vat_dev_run.py]
```

It runs `<tree>`'s own stage code with this worktree's environment, waits up to `WAIT` seconds (15 minutes by default) for a quieter machine, and writes `<script>.<label>.json` and the household dump to `$PE/data/ukds/acceptance/1164-lcfs-2024-25/`. A stage-only run takes about 90 seconds and 7 GB on a quiet machine; the chain about 150 seconds.

## Provenance

Paths inside the scripts and outputs are machine-local. The scripts are copies of the ones run, formatted by this repository's formatter.
