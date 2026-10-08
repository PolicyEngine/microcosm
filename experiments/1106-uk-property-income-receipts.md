# microcosm#1106 receipts: UK property income and property wealth

Aggregates and counts of 10 or more only. Scripts and licensed outputs live in `data/ukds/acceptance/1106-property-income/`.

## A. SPI net property-income amounts bound

Regenerated from the pinned Chronicle feed 825406f on #1121's head (`c098fdcc4` when first measured, rebased to `692e8b7b3` with the values unchanged; policyengine-uk 2.122.2).

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

## C3. Landlords' receipts and finance costs on the licensed spine

Measured on the C3 arm's spine (`48a277264`, built 2026-10-08 with `scripts/arm.sh` from the measurement tree `repos/populace-1106-arm`, licensed inputs, local staging only) with `scripts/property_receipt.py`. The ledger is the 825406f consumer artifact rebuilt the same day in `repos/arch-data-825406f`, whose two digests equal `uk/chronicle_feed.json`'s pin; the original artifact directory lost its `consumer_facts.jsonl` at 00:02 that day. Figures are at stage-time design weights (2.21m weighted landlords with profit, 2,635 records before the CGT stages clone households) unless marked calibrated.

- **The tape.** Rebased to 2024 prices as the income stage rebases it (profit x1.0924, finance costs x1.9380), the SPI 2022-23 tape's 2.88m landlords carry £33.9bn of profit and £11.18bn of restricted finance costs, 38% of them with costs. PRIS 2024-25 puts individuals' residential finance costs at £11.08bn.
- **FRS finance costs.** Of the 1.03m weighted FRS-concept landlords, 32% draw finance costs, matching the tape's share in each of the ten deciles of profit after finance costs (for example 45% in the lowest cell against 48% realised, 27% in the highest against 27%). The draws total £2.37bn, added back to FRS profit (FRS channel: £9.5bn reported, £11.9bn after the add-back). The SPI channel's 1.17m landlords carry £16.8bn of profit and £7.78bn of finance costs from the tape draw. In all, finance costs are £10.15bn on 0.82m landlords.
- **Receipts.** £47.01bn in all (£0.28bn of it sub-letting rent on 38 records), so deductible expenses come to £18.30bn against PRIS individuals' £18.95bn. The median expense share is 0.52 (10th to 90th percentile 0.37 to 0.56). No receipts fall below profit; no landlord's profit exceeds the band the walk put them in.
- **Band fit.** Every band from £10,000 holds its individuals-basis count to within 1,100 landlords, against a largest landlord weight of 9,562 (for example £10,000 to £20,000: 850,918 against 851,042). The lowest band takes the remainder: 644,691 against PRIS's 1,286,458, because the spine carries 2.21m landlords at design weights against PRIS's 2.85m individuals.
- **The top band's multiplier pins at 1.0.** The 39,583 landlords of the open band would need mean receipts of £145,858 for the receipts total, after the bounded bands' realised means (their midpoints) at PRIS counts; the spine's top landlords by profit already average £150,583 at `max(100,000, profit)` (mean profit £139,550, median £113,607, 87% on the SPI channel), so their receipts equal their profit and they carry no deductible expenses. Two causes are visible: midpoint means overstate the bounded bands if density falls within each band, and PRIS counts UK property only ("UK property income declared") while the SPI's `INCPROP` and the engine's receipts concept include overseas property. The stage gate passes (the cap is 5).
- **Coherence.** 87.8% of landlords' weight sits in households holding other residential or non-residential property (WAS).
- **Spine gates.** All 33 pass, the new `uk_stage_property_components` included.

## C6. Calibration arms

Each arm builds the spine (licensed inputs, local staging only) and runs the national calibration with `scripts/arm.sh`; `scripts/arm_summary.py` summarises them. C3 and C4 were calibrated twice: once as committed, and once with a measurement-only edit recorded in the arm's `tree.diff` (never committed).

- **Control, #1121's head `692e8b7b3` (policyengine-uk 2.122.2).** Loss 0.27534 to 0.00712; 98.3% of 1,167 targets within 10%; 7 of 7 calibrated-seam gates pass.
- **C3, `48a277264` (A, B, C0, C1 and the stage; no PRIS targets).** Loss 0.27890 to 0.00714; 98.1% of 1,180 within 10%. The 13 SPI property amounts, unbound on the control, fit to 0.1% (48.0% off at design weights). The only gate failure is `uk_target_fit` on #1121's West Midlands deferral, now stale: that cell sits at +23.1%, inside the 25% bound. With the deferral dropped (arm `c3x`), 7 of 7 gates pass.
- **C4, `82c744ccc` (+ the 13 PRIS targets).** Loss 0.27259 to 0.00708; 98.0% of 1,193 within 10%; every PRIS row within 0.3% (receipts £51.93bn against £51.99bn, residential finance costs £12.12bn against £12.13bn). Three regressions against C3: the SPI property amounts slip (£20,000 to £30,000 of total income −17.8%, £15,000 to £20,000 −6.9%, £40,000 to £50,000 −7.0%); `uk_weight_ratio` fails (family-folded maximum-to-median weight 1,213 against 1,151); and the West Midlands cell is back out at +26.8%.
- **C4 without PRIS's lowest band (arm `c4b`, measurement only).** Loss 0.00704; the SPI amounts still slip (worst −17.7%) and `uk_weight_ratio` still fails. The lowest band is not the cause.

**Why the PRIS targets pull against the SPI amounts.** At the C3 calibration's weights, which fit the SPI profit and bind no receipts, receipts come to £55.39bn against PRIS's £49.81bn (2024 values) and the expense share to 0.42 against 0.38. Landlords sit too high in receipts: 1.16m in the £10,000 to £20,000 band against PRIS's 0.85m, 0.47m against 0.33m in the next, and 0.71m in the lowest against 1.29m. The walk fills each band to PRIS's absolute count, but at stage time the spine carries 2.21m weighted landlords against PRIS's 2.85m, so each landlord's profit rank lands in a higher receipts band than PRIS's shares imply, and the calibration scales that up. Binding the PRIS targets then has to move weight across total-income bands.

**A walk filled by PRIS's shares (offline, `scripts/walk_variants.py`).** Re-running the walk on the C3 spine with each band's mass set to its PRIS share of the spine's own landlord total, and evaluating at the `c3x` weights (an approximation for an arm calibrated with that walk: the calibration reads receipts only through income tax, where the engine chooses between the property allowance and actual expenses): receipts £46.11bn, expense share 0.30, and band counts of 1.19m, 0.90m, 0.35m, 0.16m and 0.06m from the lowest band up, against PRIS's 1.29m, 0.85m, 0.33m, 0.15m and 0.08m. The total band misallocation falls from about 1.12m landlords to about 0.25m. The bands from £70,000 run below PRIS (the open band 24,100 against 39,600) and the top band's multiplier still pins at 1.

**Engine quantities at 2025 (`scripts/engine_effects.py`, policyengine-uk 2.123.0 on both calibrated candidates; control against `c3x`).**
- Income tax: £327.05bn to £326.40bn. The two candidates are calibrated separately, so this is not a policy effect.
- `property_income` £31.68bn to £34.01bn (2.80m landlords); `property_rental_income` £58.22bn; `property_finance_costs` £11.71bn on 1.07m landlords (PRIS 2025: £12.13bn).
- `property_finance_cost_relief` £1.90bn for 1.03m landlords (pe-uk#2172's illustration on Microcosm: £1.14bn to £2.52bn); finance costs carried forward £2.21bn for 0.31m.
- The property allowance deduction falls from £0.117bn (0.22m landlords) to £0.062bn (0.12m).
- Means tests that now read property income after finance costs: Housing Benefit −£21m, Council Tax Reduction −£26m, Pension Credit +£10m, Universal Credit +£3m.
- Computing `income_tax` took 0.33 s against 0.23 s, the cost of the allowance-or-expenses branching.

## C7. The walk filled by PRIS shares (`c57765505`, arm `c5`)

Each band now takes its share of the individuals-basis landlords times the spine's landlord weight (scale 0.775 at stage time: 2.21m against 2.85m).

- **On the licensed spine.** Every band, the lowest included, holds its target to within rounding (the lowest: 996,007 against 996,725). Receipts are £39.12bn at design weights; deductible expenses £10.68bn on £28.44bn of profit, an expense share of 0.27 (median 0.32, 10th to 90th percentile 0.26 to 0.38), against PRIS's 0.38: the spine's profit per landlord at design weights is about 19% above PRIS's implied profit per landlord. The open top band still pins at a multiplier of 1.
- **Calibrated, with the 13 PRIS targets.** Loss 0.27690 to 0.00722; 97.9% of 1,193 targets within 10%. The PRIS rows fit (receipts −3.3%, finance costs 0.0%, every band within 0.3%), but the SPI property amounts now overshoot at the top of the middle: £50,000 to £70,000 of total income +15.9% and £70,000 to £100,000 +48.2% (the count-filled walk's C4 arm undershot £15,000 to £50,000 instead, worst −17.8%). `uk_target_fit` fails on the £70,000 to £100,000 row and `uk_weight_ratio` fails again.
- **Reading.** Receipts are a fixed function of profit rank, so the eleven band counts constrain the profit distribution and compete with the SPI's profit-by-income targets whichever rule fills the bands. Without PRIS targets the calibration reads receipts only through the allowance-or-expenses choice in income tax, so "PRIS as diagnostics" comes close to the C3 arm's fit with the receipts the walk gives at its weights (the offline evaluation in C6: £46.11bn, lowest three bands 1.19m, 0.90m, 0.35m).
- **Calibrated with only the two PRIS totals (arm `c5t`, the eleven band counts dropped, measurement only).** Loss 0.27652 to 0.00704; 98.1% of 1,182 within 10%; receipts and finance costs fit exactly. The SPI property amounts still overshoot (£50,000 to £70,000 and £70,000 to £100,000 of total income fail `uk_target_fit`, worst +30.9%) and `uk_weight_ratio` fails. At the C3 weights the share-filled walk gives £46.11bn of receipts at 2024 values (about £48.1bn at 2025), so the receipts total asks for about 8% more receipts than the spine's landlords carry and moves weight toward them.
- **Why the totals conflict too.** PRIS's 2.85m landlords and £49.81bn of receipts include landlords whose expenses meet or exceed their receipts; the FRS records them with a loss, which counts as zero, and the spine gives them no receipts. The spine's landlords are the profit-making subset, so any PRIS count or receipts target lands on them. The FRS identifies few of them, though: in 2024-25, 67 adults (0.11m weighted) report a loss from other property, against 1,022 (1.75m) reporting a profit. Other parts of the gap are in the walk itself: at calibrated weights the bands from £70,000 run below PRIS, and the open band carries no expenses.

## C8. PRIS as a diagnostic

María's decision (2026-10-08): PRIS binds no calibration target. The three PRIS targets are reverted (active national references return to 1,244), and the share-filled walk stays. The stage vendors PRIS 2024-25: its receipt reports receipts, deductible expenses, finance costs and the band counts against PRIS at design weights, and the stage gate holds every band above the lowest to its share of the spine's landlords.

The arms above ran on the commits named there, before the branch was rebased onto main after #1121 merged (the rebased commits keep their subjects). The rebase brought in a changelog sentence, #1121's receipts note and an equivalent rewrite of one weighted share in `frs_uc_start_up_period`.
