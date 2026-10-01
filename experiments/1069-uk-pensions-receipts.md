# microcosm#1069 receipts: UK pensions

Plan: `repos/microcosm-uk-pensions-implementation-plan.md`, approved 2026-09-30 (rulings R1–R15). The plan
was executed in this order on branch `uk-pensions-1069`, rebased onto main 3598c38de:

- c5, c1–c4 and the £40–60 band sign-out came earlier (see their own commits).
- This note covers the c5 fix (dae9cc916), c6 (7cac27b27), c7 (f795e9121), c8 (9b66ab395), the D6
  retirement (b19f14d98), c9 (8b3e9f993), the c6 follow-up (be5c13245), c10 (13089945a), c11
  (fcb5c46b5) and the c8/c9 follow-up (d9fca7bb7).
- c0 waits for PolicyEngine/policyengine-uk#1899.

Builds. Every arm is a licensed local build: the spine (the build_930 recipe with the five NTS tabs and
`--was-person-tab`, `--no-staging`), then the national role (`--staging-local-only`). They use the
Chronicle artifact 825406f (PolicyEngine/chronicle#302 via #305) and the 2026-09-30 national build as the
incumbent. The scripts live in `data/ukds/acceptance/pensions-1069/scripts/`:

- `arm.sh` and `arm2.sh` (`arm2.sh` refines the quiet gate);
- `arm_readout.py`, which reads out from the ordered dense solution, so it also works on arms whose
  battery blocked the H5;
- `arm_h5.py`, which writes measurement H5s from a spine and its calibrated weights;
- `wp7.sh`, `wp7_analyse.py` and `relief_decomposition.py`.

Every H5 is measurement only and never released. Engine: policyengine-uk 2.100.0 (the repo's pin).

The arms, in order:

- c4: the starting point.
- c5fix.
- c6s50 and c6s20, the σ_p variants; 0.2 was never committed.
- c7.
- c9: c8 and c9 together, which blocked on Table 3.8.
- final: d9fca7bb7, reusing the c9 spine. Its battery passed and it wrote an H5.
- finals20: the final state at σ_p = 0.2, a variant.

## Part A. The SPI channel's 66-year-olds (c5 fix)

The c4 arm left 402 66-year-old records without a State Pension, so only 81% of 66-year-olds received
it. That share is what squeezed age 65 inside the ONS 65–69 band (R3).

- **Where the zeros sat (c4 spine).** 330 of the 402 records, carrying 88% of their weight, were SPI-channel
  synthetic records. The SPI channel's zero-State-Pension share was 100%, 51% and 6% at 65, 66 and 67; the
  FRS respondents' was 7% at 66 and 3–6% at 67–71.
- **Mechanism.** c5's donors carry whole-year ages plus a uniform fraction, while the hosts carry whole
  years. A host at exactly 66.0 sits on the forest's 66.0 split and reads a 65.x donor's leaf, which never
  holds a State Pension. In a synthetic forest the whole-year query gives 0% receipt at 66 and the
  mid-year query 100%.
- **Fix (dae9cc916).** Stage-1 queries the forest at the whole-year age plus half a year
  (`_stage1_query_predictors`, also used for the base redraw). Band carriers now pool only donors on
  their own side of State Pension age (`_resample_band_donor_leaves(state_pension_age=…)`, with a receipt
  of matched carriers per band).
- **Arm c5fix (spine 39c1254a).**
  - Loss 0.00826 on 1,148 targets (c4: 0.00908 on 1,149); 98.0% within 10%; ESS 5,003.
  - No row above 25%. The London £12,570–15,000 income-tax row moved from +26.2% to +1.9%, and the
    50–70k State Pension band (D6) from +28.6% to +23.0%.
  - State Pension receipt at 66: 95%.
  - Single ages 64–68 against ONS mid-2024: +1/−15/0/+2/0% (c4: −5/−27/−7/+20/+4%).

## Part B. The SPI channel's pension-age mass (c6)

- **What the prior mix showed (c4 spine).**
  - 66+ people at prior weights were 13.88m: 6.07m FRS and 7.81m SPI channel. Pension-age households were
    5.67m in the SPI channel against 4.44m from the FRS.
  - The SPI channel's uniform within-region weights inflate pension-age mass: the FRS sample
    over-represents pensioner households, and uniform weights keep that composition. Design-weight 66+
    is about 12.1m (ONS about 12.4m).
  - The taxpayer share of the 66+ is 67.7% in the FRS channel (HMRC 67%) and 83% in the SPI channel.
  - The non-taxpayer share of State Pension is 25.9% in the FRS channel and 10.6% in the SPI channel;
    HMRC and DWP imply 32.4% (£46.1bn of £142.3bn).
- **Change (7cac27b27).** The allocation is stratified by region × pension-age household (any member at
  or over State Pension age), with a pension-age share σ_p. The committed value is 0.5 (provisional).
- **Arms.**
  - c6s50 (spine 6721f725): loss 0.00824 on 1,148; within 10% 97.8%; ESS 5,033.
    - Prior 66+ people 12.33m.
    - SPI State Pension bands: 20–30k +15.8%, 50–70k +23.6%.
    - Non-taxpayer State Pension £36.8bn; 66+ calibrated FRS 6.87m / SPI 5.66m.
  - c6s20 (spine 0bbf0895; a measurement variant, never committed): loss 0.00857; ESS 4,967.
    - SPI bands: 20–30k +10.1%, 50–70k +24.0%.
    - Non-taxpayer State Pension £38.6bn.
    - Single ages worse: 62 +15%, 63 +10%, 72 −17%.
  - Pre-calibration Pension Credit spend: c5fix −17.5%, c6s50 −23.5%, c6s20 −5.8%. Attendance Allowance
    −12%, −21% and −17%.
- **HBAI (measurement H5s, Part H).** Pensioner quintile shares before housing costs against HBAI's
  22/25/21/18/14: c6s50 20.9/26.5/20.5/18.7/13.4; c6s20 21.7/27.8/20.3/17.9/12.2.

## Part C. D6 retired (b19f14d98)

The 50–70k State Pension band fits at +23.0%, +23.6% and +24.0% on the c5fix, c6s50 and c6s20 arms.
The terminal `uk_target_fit` gate fails its reviewed deferral (ruling D6, 2026-09-23, expiring
2026-10-21) as stale, so every arm from c5fix to c7 blocked its H5 on that one line. The entry is
retired, as the SE deferrals were, and the register is empty.

## Part D. Pension Credit (c7)

- **Rows.** 14 caseload targets (`dwp_pension_credit`; the CY2025 mean of four quarterly points): Great
  Britain, six type-by-partner cells, six age bands and Northern Ireland. `obr.pension_credit` stays bound.
- **Take-up.** A new stage, `pension_credit_take_up`, after the SPI chain. Within each DWP component a
  reporter claims, and the other entitled units claim at r = (t·E − R)/(E − R), with FYE2024 take-up of 69%
  (Guarantee Credit) and 37% (Savings Credit only).
  - Stage gate on the c7 spine: realised take-up 69.4% (1,463 entitled units in the sample) and 39.1%
    (392).
- **Arm c7 (spine e5788819).** Loss 0.00818 on 1,162; within 10% 97.8%; ESS 5,004 (66+ households 2,071).
  All 14 caseload rows are within 0.3%.
- **The prior misses.** Before calibration the caseload is 0.889m against 1.391m (−36%) and spend −35.3%
  (c6s50 −23.5%).
  - The take-up lands, so the miss is on the entitled side: too few entitled pensioner units at the
    σ_p = 0.5 prior.
  - Calibration closes it. R9's fallback, solving take-up against the caseload, would not help: the
    rates already reproduce DWP's.

## Part E. Contributions (c8, c9) and the rows that left the fit

- **c8 (9b66ab395).** The FRS spine no longer clips personal and stakeholder contributions at the 95th
  percentile of every PENPROV amount; the clip removed 28% of the reported amount on 1.8% of rows. The
  commit also authored four SPI Table 3.8 targets (net pay and relief at source, amount and count by
  total-income band; amounts uprated by the earnings index under R5, counts by taxpayer growth).
- **c9 (8b3e9f993).**
  - Employer contributions are an ASHE 2024 rate draw on pay, by scheme type and sector, replacing three
    times the employee contribution.
    - On the c9 spine: 10,102 members with earnings and a mean member rate of 8.4%.
    - Members by ASHE type: 2,204 defined benefit, 2,824 defined contribution, 4,201 group personal and
      568 group stakeholder, plus 305 not specified.
  - Converted salary-sacrifice records lose the sacrificed pay: 4,625 converted rows take the headcount
    from 2.70m to 5.44m, against the 5.4m staging target.
  - Contract:
    - DWP employer contributions (£104.2bn) against the engine's employer contributions plus salary
      sacrifice;
    - DWP gross employee contributions (£45.5bn employees' own plus £20.7bn relief = £66.2bn);
    - the 2024-25 salary-sacrifice amount (£24.6bn);
    - the 7.7m users total leaves the fit.
- **Arm c9 (be5c13245, spine c2232d87).** The battery blocked on 12 Table 3.8 rows.
  - Loss 0.01513 on 1,216 targets (c7: 0.00818); within 10% 95.6%; ESS 4,060 (c7: 5,004). Single ages
    distorted (69 +21%).
  - Totals against SPI 2023-24, uprated:
    - net pay £23.0bn and 9.66m contributors: +67% and +68% at prior, +136% and +78% calibrated;
    - relief at source £21.1bn and 10.87m: −28% and −82% at prior, −2% and −74% calibrated;
    - DWP employee gross: −43% prior, −18% calibrated;
    - DWP employer: +12% prior, +0.1% calibrated;
    - salary-sacrifice amount: +57% prior, on target calibrated.
- **Diagnosis.**
  - Table 3.8 splits contributions by relief mechanism; the engine's columns split by provider.
    `employee_pension_contributions` holds every workplace deduction, and `personal_pension_contributions`
    holds the FRS personal and stakeholder pensions. Workplace group personal pensions and many master
    trusts take relief at source, so the engine's "net pay" is too large and its relief-at-source
    contributors far too few (about 2.0m against 10.9m). No reweighting reaches that.
  - DWP's gross employee figure also pulls against SPI's net pay: SPI net pay plus relief at source is
    £44.1bn, close to DWP's £45.5bn employees' own contributions, not the £66.2bn gross.
- **Follow-up (c11b).** The four Table 3.8 targets and the DWP employee row leave the fit
  (`scripts/c11b_unbind.py`; active references 1,283 → 1,230). The clip removal, the employer draw, the
  conversion fix and the DWP employer row stay. Re-specification options are in Part I.

## Part F. Pension-age benefits (c10, 13089945a)

- Attendance Allowance cases with entitlement in England (1,597,003) and Wales (124,118): the mean of the
  four calendar-2025 quarterly points, in the new family `dwp_attendance_allowance`.
  - Scotland is not bound. Its caseload moves to Pension Age Disability Payment through 2025 (171k in
    February, 25k in November), which the engine does not model.
- Pension-age Housing Benefit benefit units in Great Britain, as the twelve-month mean for 2025:
  1,107,464 across all tenures, 890,664 in the social and 216,779 in the private rented sector.
  - The tenure cells are Stat-Xplore detail rows (measure `claimants`), and the engine reads the eldest
    adult's age against 66.
- Winter Fuel Payment recipients in England and Wales, winter 2025/26 (11,003,237; fact key 31256472…),
  sit beside `obr.winter_fuel_allowance` as a required diagnostic and are never fitted
  (PolicyEngine/policyengine-uk#1760).
- **Final arm.**
  - Every row fits within 0.2%: Attendance Allowance England −0.2% and Wales 0.0%; pension-age Housing
    Benefit −0.2% (social −0.2%, private 0.0%).
  - The prior was far short: Attendance Allowance −35% (England) and −23% (Wales); pension-age Housing
    Benefit −41% (social −40%, private −49%). The frame lacks low-income pensioners at prior weights, the
    same signal as Pension Credit (Part D).
  - `obr.attendance_allowance` (amount) moves from 0.0% to +2.6%, and `obr.winter_fuel_allowance` from
    +0.1% to +3.2%.

## Part G. The salary-sacrifice relief counterfactual (c11, fcb5c46b5)

- **Route.** The resolution loop resolves `input_substitution_counterfactual` bindings through the
  provider. `UKMeasureResolver` builds one extra simulation per substitution: the sacrifice is zeroed and
  added to pay at the calibration year, and the result is the output's delta against the baseline. The
  delta lands in a slash-free column.
- **Rows (HMRC Table 6.2, 2024-25).**
  - Employee NICs relief: £1.0bn.
  - Employer NICs relief: restated from 13.8% to the engine's 2025 rate of 15%, so £3.4bn becomes
    £3.70bn (new engine index parameter).
  - Income-tax relief total: £8.8bn.
  - c9's interim amount row retires; the three rate bands stay excluded until the band-test fix.
- **Engine test.** £3,000 returned to pay is taxed at 20% income tax, 8% employee NICs and 15% employer
  NICs.
- **Decomposition at c7 (before the c9 pay fix; scripts/relief_decomposition.py).**
  - Sacrificed amount £36.9bn across 7.67m sacrificers.
  - Income-tax delta £8.31bn: the Child Benefit charge contributes +£0.09bn and the annual-allowance charge
    −£0.27bn.
  - Employee NICs delta £1.27bn, employer £4.13bn.
  - At c7 the deltas double-count, because converted records' pay still carried the sacrifice; c9's fix
    removes that.
- **Final arm (fits).**
  - Income-tax relief total £8.8bn: −0.0% (prior −2.5%).
  - Employee NICs relief £1.0bn: −0.1% (prior +20.6%).
  - Employer NICs relief £3.70bn: −0.1% (prior +10.3%).
- **Final-arm decomposition on the H5.**
  - Income-tax delta £8.86bn: the Child Benefit charge contributes +£0.09bn and the annual-allowance charge
    −£0.39bn, so £9.16bn excluding the charges.
  - Employee NICs £1.01bn, employer £3.72bn.
  - The sacrificed amount is £36.3bn across 7.66m sacrificers, against the £24.6bn HMRC's employer relief
    implies (c9's amount row held it exactly).
    - In the final H5, 1.24m sacrificers (16%) have no pay but carry £18.8bn (52% of the amount), and
      0.35m are under 16 (£5.3bn). At prior, on the c9 spine, no-pay records already hold 64% of the
      amount (£24.9bn of £38.8bn); in the c9 arm, with the amount row bound, they held £11.3bn of £24.7bn.
    - The salary-sacrifice stage's forest predicts a sacrifice for everyone not asked (the #1063
      non-earner finding). Returned pay on those records carries less relief per pound (the personal
      allowance and the NICs thresholds absorb it), so the solver inflates amounts to reach the relief
      totals.
    - So the R7 premise, that the employer row anchors the amount, does not hold until the non-earner
      defect is fixed (a #1063 fold-in). Keeping the amount row bound beside the relief rows is the
      interim option.

## Part H. Validation (WP7) and the final arm

- **Final arm (d9fca7bb7, `runs/uk-pensions-1069-final`).** The battery passed and the H5 was written.
  - Loss 0.00730 on 1,170 targets; within 10% 97.5%; ESS 4,900 (66+ households 2,268); top-1% weight
    share 0.272. For comparison: c7 0.00818 on 1,162, and the 2026-09-30 main build 0.00836 on 1,090.
  - Measure resolution, including the one counterfactual simulation, takes about 10 minutes.
  - No row above 25%. The worst rows:
    - the 50–70k State Pension band +24.5%;
    - East Midlands total income £12,570–15,000 +24.4%;
    - VAT +23.9% (PolicyEngine/policyengine-uk#1996);
    - Child Benefit +22.7% (#1063);
    - employment income £15–20k −21.8% and £20–30k −21.1%.
  - SPI State Pension bands: 20–30k +19.7%, 30–40k +20.0%, 40–50k +12.3%, 50–70k +24.5%, 70–100k +16.6%.
- **Engine, FY2025-26, on the final H5 (pension_types.py).**
  - State Pension £142.5bn and 12.42m recipients, against the £142.3bn resident level (GB £138.6bn plus NI
    £3.7bn).
  - New State Pension 4.85m (£54.2bn); basic 7.57m; additional 9.18m.
  - Pension Credit 1.46m benefit units and £6.15bn (targets 1.45m and £6.11bn): Guarantee Credit only
    0.859m, both 0.430m, Savings Credit only 0.171m.
  - Ages 64–68: 0.874, 0.633, 0.721, 0.757 and 0.698m, against ONS 0.80, 0.77, 0.75, 0.71 and 0.68m.
- **HBAI (pensioner_position.py).**
  - Pensioner quintile shares before housing costs: 20.6/26.2/21.4/18.8/13.0, against HBAI's
    22/25/21/18/14. The pre-#1069 control (ch3) had 18.3/24.0/22.0/18.8/17.0.
  - Pensioner relative poverty: 14.2% BHC and 14.7% AHC, against HBAI FYE 2025's 16% and 14% (ch3: 13.1%
    and 11.3%). The arms run from 14.2% to 16.0% BHC.
- **CPI-only State Pension probe (reform_probe.py, wp7_analyse.py; 300 household bootstrap draws).**
  - Budget effect +£1.20bn, +£1.79bn, +£2.43bn and +£3.10bn for 2027 to 2030 (SE ≤ £0.08bn).
  - ch3 is 10–13% lower and the #979 build 7–8% lower: mostly the State Pension level (the engine's £131bn
    on ch3 against £142.5bn now).
  - Decile loss shares in 2030: 7.4/8.9/12.6/13.0/11.4/11.4/12.7/8.8/7.7/6.0, SE ≤ 1.0 point. ch3's
    heavy D4 record (SE 3.4 points) is gone.
- **Stability.** A second calibration seed would change nothing: the national solve is deterministic
  (L0 λ = 0; the seed only feeds L0 gate sampling). Stability is read from the household bootstrap above
  and from the σ_p variant below.
- **σ_p variant at the final state (finals20, spine 70f4cb44; uncommitted).**
  - The battery blocks on one row: the 50–70k State Pension band at +25.6% (final +24.5%).
  - Loss 0.00758 against 0.00730; within 10% 97.7% against 97.5%; ESS 4,994 against 4,900; top-1% weight
    share 0.267 against 0.272.
  - SPI bands: 20–30k +13.9% and 30–40k +15.1% (better), 70–100k +22.6% (worse).
  - Non-taxpayer State Pension £37.5bn against £35.5bn.
  - Prior gaps are closer: Pension Credit −24% against −36%; Attendance Allowance (England) −32% against
    −35%; pension-age Housing Benefit −28% against −41%.
  - Single ages: 62 +15%, 63 +11%, 64 −6%, 65 −17%, 69 +12% (final: +8/+2/+9/−19/+20%).
  - HBAI quintiles: 21.5/27.9/21.1/16.7/12.7 (Q2 +2.9 points against HBAI; the final's largest gap is 1.4).
    Pensioner relative poverty: 15.1% BHC and 15.1% AHC.
  - CPI-only probe against the final: the budget effect agrees within 0.5% in every year, inside WP7's 3%
    criterion. Decile loss shares differ by up to 1.2–1.8 points, just outside its 1-point criterion: 0.2
    moves loss towards D1–D4. The bootstrap SE is 0.6–1.0 points.

## Part I. Remaining gaps and rulings owed

- **c0 (State Pension age by date of birth).** Blocked on PolicyEngine/policyengine-uk#1899, still open.
  The 70–74 type-boundary zig-zag remains until it lands, with single ages 69 to 73 off by 10% to 20%.
- **Engine follow-up (repos/policyengine-uk-1069, 4 local commits).** These cover reform propagation, the
  2025-26 Pension Credit rates and guarantee uprating, the April 2027 rate and the state_pension column
  guard. Items 2 (the basic-to-new double count) and 3 (Additional State Pension by CPI) wait for #1899.
  Seven engine rulings are owed.
- **σ_p (SPI channel pension-age share).** 0.5 is committed provisionally; the evidence is in Part B and
  the final-state variant. Her ruling.
- **Single ages (R3).** At the final state 65 is −19% and 64 +9% (c5fix: −15% and +1%). The fallback,
  splitting 65–69 into 65, 66 and 67–69, is available on her ruling.
- **SPI State Pension bands 20–100k: +12% to +24.5% at the final state.** The 50–70k band (D6, now retired)
  sits at +24.5%, just inside the 25% bound. At σ_p = 0.2 it crosses (+25.6%) and blocks the H5.
- **Non-taxpayer State Pension.** £35–39bn against about £46bn implied (the HMRC/DWP split).
- **Pension Credit entitlement at prior: −36%** (σ_p interaction); calibration closes it.
- **Contribution rows out of the fit (Part E).** Options:
  - (1) Bind combined Table 3.8 amounts (net pay plus relief at source) against employee plus personal
    contributions, and drop the counts.
  - (2) Model the relief mechanism from PENPROV.STEMPPEN: group personal and stakeholder workplace schemes
    to relief at source, occupational to net pay. NEST-type master trusts stay ambiguous.
  - (3) Keep both families as diagnostics.
  - The DWP employee figure: bind employees' own £45.5bn, the gross £66.2bn, or neither.
- **Salary-sacrifice relief.**
  - The relief rows fit, but the sacrificed amount rises to £36.3bn (£24.6bn implied), because no-pay
    records hold half of it (Part G). Options: keep the c9 amount row bound beside the relief rows, or
    land #1063's earners-only fix first. R7 retired the amount row, so this is her call.
  - The three income-tax rate bands are deferred until the band test reads after-allowance thresholds and
    the Scottish bands; the exclusions expire 2026-11-25.
  - The FY2024-25 relief and amount rows are not earnings-uprated (consistent with R1 and R10), a gap of
    about 4–5%.
  - The employer row is restated to 15% (my call, to confirm).
  - The income-tax delta includes the Child Benefit and annual-allowance charges.
- **Winter Fuel.** A diagnostic only, until PolicyEngine/policyengine-uk#1760.
- **Attendance Allowance in Scotland (Pension Age Disability Payment).** Not bound.
- **Salary-sacrifice anchor.** Not yet vendored from Chronicle (a plan item; the values already match the
  feed).
- **WP7 second seed.** The national solve is deterministic, so stability is measured by bootstrap and
  across the σ_p variant.
- **Time-critical, outside this PR.** The `slc.borrowers.plan_2_*` measure exclusions expire 2026-10-03;
  an expired entry raises in every build and in the certifier.
- **Outward steps, each on her go.** Push the branch and open the draft PR; comment on
  PolicyEngine/chronicle#302; flag the deleted parameters on PolicyEngine/policyengine-uk#1899; open the
  engine follow-up issue and PR.
