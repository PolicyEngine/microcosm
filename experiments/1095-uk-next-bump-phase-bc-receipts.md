# microcosm#1095 and #1278: the next policyengine-uk bump, Phases B and C receipts

This receipt gives no count of source households and no row's model figure that rests on fewer than ten of them. Nor does it give a model figure of a row of the SPI channel, or of a row that rests in large part or in part on the SPI channel's households: those rest on the SPI Public Use Tape (SN 9422). Publisher figures appear as published. The fit and weight statistics are over the whole national surface, as in `experiments/1095-uk-next-bump-phase-a-receipts.md`.

## The question

Phase B locks policyengine-uk 2.151.0, the first release past pe-uk#2573 and pe-uk#2580. Phase C writes the inputs those releases read and the fixes they unlock:

- the energy spending's price period, `energy_price_basis` (pe-uk#2571);
- the legacy closure draw reads the claimant's and partner's reports (pe-uk#2549);
- the UC take-up redraw keeps a reported income-related ESA award off a UC claim (pe-uk#2563);
- Scotland's carers enter the Carer's Allowance take-up (#1319);
- Widowed Parent's Allowance leaves FRS code 6 by rate (#1222);
- spousal maintenance gets its own column (#1320);
- bonus sacrifice is imputed to OBR's 2027-28 base through the engine (#1340);
- liquid and solid heating fuels are drawn (#1278);
- the LCFS divisions' excess over the total ratio is levelled (#1278);
- the WAS share split and the debt secured on other property (#1095);
- England's single adult council tax discount binds at the national grain (#953);
- an engine test pins the engine's 2025 relabel of student loan plans.

Arm 2 then re-measures the rows still held, and the ORR rail-support row is bound on table 7271, the plan's default on the open 7270-versus-7271 ruling, unless an arm measures a distortion.

## Method

- **Control:** b2572-c0main, main at 6bdb8001c on policyengine-uk 2.147.2: one spine and one calibration, review date 2026-10-23. The PR's base, f9f28e4a0, adds #1347 to it, which changes only local-line declarations.
- **Arm 2:** one spine and one calibration at the bump head, review date 2026-10-23. The calibration first stopped because the new single adult discount row carried no uprating declaration; the hold that declares it changes no stage, so the spine stands.
- **Arm 2b:** a calibrate-only run on arm 2's spine with the four UC payment bands that arm 2 fits unbound now bound, review date 2026-10-24, the date the Housing Benefit and JSA entries are renewed.
- **Arm 2c:** a calibrate-only run on arm 2's spine with the ORR row bound on table 7271 beside the four bands, at the same review date.
- Arm 2 measures Phases A, B and C together against main. Arms 1 and 1b measured Phase A's bindings alone; the weekly annualisation (uk-data#531) changes the spine, so arm 2 is its first measurement.
- Each run's fit is recomputed from its own compiled matrix and final weights, so a row whose published figures are withheld is still measured. The script output stays with the licensed spines, outside the repository.

## Results

Both runs pass all seven terminal gates. Arm 2 binds 1,265 rows against the control's 1,259: the three band-H region rows, Plan 5's liable stock and `obr.vat` from Phase A, and the single adult discount row from C9.

Over the 1,258 rows both runs bind that have a nonzero target:

- within 10 percent of the target: 99.21 percent in the control, 99.05 percent in arm 2;
- mean absolute relative error: 0.56 percent, 0.55 percent;
- effective sample size: about 7,400, about 7,500;
- share of design weight stretched 9.5 times or more: 0.67 percent, 0.59 percent.

Two shared rows move past the 10 percent line and none moves the other way. One is a row of the SPI channel, which this receipt does not name. The other is a UC payment-distribution band whose figures are withheld, so this receipt gives no figure of it.

The new rows reach their targets: England's single adult discount row fits within 10 percent of MHCLG's 8,496,178 dwellings (Council Taxbase line 8, as at September 2025, in the revision of 21 January 2026), and `obr.vat` and Plan 5's liable stock fit more closely still. The band-H region rows bind with their figures withheld where fewer than ten source households carry them.

## The held rows

Measured unbound at arm 2's calibrated weights:

- four of the twelve UC payment-distribution bands come within 10 percent, so they bind (arm 2b below); the other eight stay outside it, seven short and one over;
- the four Housing Benefit rows stay outside the 25 percent bound at design weights and short at the calibrated weights;
- DWP's JSA claimants: 73 percent short at design weights and 29 percent short at the calibrated weights; every claimant the row counts is in an FRS household.

The UC bands rest in large part, and the Housing Benefit rows in part, on the SPI channel's households, so this receipt gives no figure of them. The Housing Benefit and JSA entries are renewed, signed on 2026-10-24 to 2027-01-24, pending the Housing Benefit and JSA PR (#867, #869).

**Arm 2b.** With the four bands bound, all seven terminal gates pass and the run binds 1,269 rows. Over the 1,264 rows arm 2 and arm 2b both bind that have a nonzero target:

- within 10 percent of the target: 99.05 percent in arm 2, 99.13 percent in arm 2b;
- mean absolute relative error: 0.56 percent, 0.57 percent;
- effective sample size: about 7,500, about 7,400;
- share of design weight stretched 9.5 times or more: 0.59 percent, 0.61 percent.

No shared row leaves the 10 percent band, and one, a UC household count, enters it. The four bands reach their targets, and binding them lowers the share of their carriers' design weight stretched 9.5 times or more. Arm 2b's commit differs from the head's in reason text only, so arm 2b is the head's calibration.

**Arm 2c, the ORR row on table 7271.** Bound on table 7271, the row reaches its target, but the weights concentrate past the release bound: `uk_weight_ratio` fails, so the run stops at the terminal gates. The effective sample size falls by about a tenth against arm 2b, and the UC household count that arm 2b brought inside 10 percent leaves it again. That is a measured distortion, so the row stays held out on table 7270's basis, its entry's reason records the measurement, and the basis ruling stays open.

## By item

**The ESA rule.** DWP's income-related ESA claimant row fits within half a percent in both runs; no household of the SPI channel carries it. With the rule, the units that report income-related ESA keep it at the calibration year, so the engine's new closure on a UC claim (pe-uk#2563) does not take the award off them. Housing Benefit keeps the stored flag, pending the HB and JSA PR (#867, #869).

**C13, the LCFS divisions.** On arm 2's spine, seven of the twelve COICOP divisions stood above the twelve's shared frame-over-donor ratio and came down to it: recreation, education, household furnishings, health, miscellaneous goods and services, transport (by its remainder alone) and communication. Since Vahid's review (item 9; María's ruling of 2026-10-24), a division's ratio is the spend the chain drew for it against the donor's same spend, so housing's and transport's levelled parts no longer count as over-draw, and the cut is made once. On arm 3's spine six divisions come down: recreation, education, household furnishings, health, miscellaneous goods and services and transport. Communication now sits below the shared ratio and keeps its draw, as do food, housing, alcohol and tobacco, clothing and restaurants and hotels. The ratios rest in part on the SPI channel's households and stay with the licensed data.

**C7, the heating fuels.** ONS Consumer Trends puts household spending on liquid fuels (04.5.3) at £1.195bn and on solid fuels (04.5.4) at £0.352bn in 2024, as published. The LCFS donor records about 2.1 and 1.8 times those at diary prices, and the stage draws both fuels at the donor's level; the receipt sets each beside ONS's figure, report-only, as #1278 ruled. The donor's liquid fuels hold bottled gas, which COICOP puts under 04.5.2 (gas) and ONS's 04.5.3 leaves out; without it they are about 1.8 times ONS's figure, so part of the gap is the concept and part the diary's capture (Vahid's review, item 8). A costing on either column would come out near twice ONS's spend. Under calibration the solid fuels move by more than the 2 percent the consumption drift block flags; the flag is warn-only. Electricity and gas spend, and the gas-connected share, stay inside their flags.

**The 52-week rule (uk-data#531).** Annualising every weekly FRS amount at 52 weeks lowers annual pay, self-employment income, the HMRC replay leaves and the reported benefits by 0.34 percent before calibration. The bound rows that absorb it are the HMRC SPI income-band counts and amounts, the OBR income tax and National Insurance receipts, and the OBR and DWP benefit spending and caseload rows (Vahid's review, item 15). Arm 2 measures it together with every other change.

**The benefit cap (#1349).** The total counts the same benefit units as its fourteen amount bands, which partition it, so a band's figures could be given back by the total less the other bands; the complementary check withholds a second row wherever one band rests on fewer than ten source households. Housing Benefit-capped households have no row: DWP's Table 4 counts Universal Credit households, so the rows bind the UC reduction alone (item 17).

**C10, the WAS property draw.** On every household, the directly held shares, the unit and investment trusts and the stocks-and-shares ISA sum to `corporate_wealth`, and every debt secured on other property sits on property of its class within the mortgages on other property. The wealth coherence gate holds both as identities.

**C3, bonus sacrifice.** The conversion meets OBR's 2027-28 base of £13.0bn above the £2,000 threshold at the stage's weights. Since Vahid's review (items 5 and 6; María's rulings of 2026-10-24), the sacrifice is added beside pay, which stays as it is, and an SPI-channel adult draws its bonus share of pay from a base-FRS adult in its pay cell. The anchor grows the amounts with OBR average earnings and the weights with the ONS population, where the OBR grows its base with employment and earnings (item 7). The model's projected base, at the stage's and at the final weights, stays with the licensed data.

**C6, student loans.** At 2025 the engine relabels every holder by a course started at 18, which moves a share of the stage's England Plan 2 borrowers to Plan 1. The engine test pins that behaviour; the fix belongs in policyengine-uk. The borrowers rest in large part on the SPI channel's households, so the share stays with the licensed data.

**The HBAI scorecard (#731).** It ran on arm 2b's calibrated file, the head's calibration, at engine year 2024 against HBAI FYE 2025. Its four results rest in part on the SPI channel's households, so they stay with the licensed data.

## Not in this PR

- C8: main's #1338 already fits the childcare rates on DfE's January basis and binds the 9 to 11 month and age-1 registrations.
- C14: a diagnostic reference carries one fact, and the Welsh fare receipts row is a difference of two; it needs its own design.
- The regional tenure, rent and UC-by-children rows through the cross-grain bridge: no national contract row carries them yet; they wait on #1305 and the local surface.
