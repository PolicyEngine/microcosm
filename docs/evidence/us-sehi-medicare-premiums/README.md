# Medicare premiums in the self-employed health premium attribution

Options and measurements for one methodology decision: how, if at all, the
self-employed health insurance deduction should count the premiums of
self-employed people who are on Medicare. The `other_health_insurance_premiums`
stage withholds every premium from them today.

Every figure below is a real PolicyEngine-US microsimulation on the certified
default (`populace_us_2024`, build `populace-us-2024-spm-20260915`, sha256
`6496cc43…aee84`, period 2024) unless a row says it is a stored-data
tabulation. Scripts and aggregate outputs are in this folder.

*Work in progress: the pinned-engine (2.2.1) runs and two #10120 rows are still running; the summary and decision follow when they land.*

## The question

`attribute_self_employed_health_premiums`
(`packages/microcosm-build/src/microcosm/build/us_runtime/other_health_insurance.py`)
copies the reported non-Part-B premium (`health_insurance_premiums_without_medicare_part_b`,
CPS `PHIP_VAL`) onto the person input `health_insurance_premiums` for
self-employed people. It withholds the copy from the "Medicare proxy" (age 65 or
older, or any `social_security_disability`) and from people with measured ESI.
The stated reason is to keep PolicyEngine-US's medical-expense premium concept
unchanged. It also sets `is_self_employed` for everyone with a profitable
business, Medicare proxy included, so the engine's gate is already open for
them.

## What the law counts

- **26 U.S.C. 162(l)(1)** allows "the amount paid during the taxable year for
  insurance which constitutes medical care" for the taxpayer, spouse, dependents
  and children under 27, limited by (l)(2)(A) to earned income from the business
  under which the plan is established, and barred by (l)(2)(B) in months of
  eligibility for a subsidized employer plan.
- **Form 7206 instructions**, p. 1, the same sentence in
  [2023](https://www.irs.gov/pub/irs-prior/i7206--2023.pdf#page=1),
  [2024](https://www.irs.gov/pub/irs-prior/i7206--2024.pdf) and
  [2025](https://www.irs.gov/pub/irs-prior/i7206--2025.pdf): "Medicare premiums
  you voluntarily pay to obtain insurance in your name that is similar to
  qualifying private health insurance can be used to figure the deduction."
- **IRS Chief Counsel Advice [201228037](https://www.irs.gov/pub/irs-wd/1228037.pdf)**
  (May 1, 2012; it "may not be used or cited as precedent"):
  - "All Medicare Parts are insurance that constitutes medical care under
    section 162(l)."
  - A sole proprietor "must pay the Medicare premiums directly". A partner's or
    2% S corporation shareholder's premiums must be paid or reimbursed by the
    partnership or S corporation and reported as guaranteed payments or W-2
    wages.
  - Medicare premiums for a spouse, dependent or child under 27 count.
- The Form 7206 instructions let a Schedule C or F filer hold the policy in
  their own name; a partner's policy in the partner's name counts only if the
  partnership reimburses and reports the premiums as guaranteed payments.

So Medicare premiums (Part B, and by the CCA every Part) count. Medigap and
other private premiums count as for anyone else. The readings that remain open
are about the data and the model, not the statute:

1. which Medicare premiums the data can see (the reported non-Part-B premium,
   the engine's modeled Part B, the CPS-reported Medicare premium);
2. whether a modeled amount (Part B net of Medicare Savings Program coverage)
   should stand in for an amount paid;
3. partners: no input records partnership reimbursement, for Medicare or any
   other premium.

## What the data holds

From the [CPS ASEC 2025 technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf):

- `PHIP_VAL` (question HIPREM) is what the person paid out of pocket for all
  health insurance premiums covering themselves or others in the household,
  comprehensive and supplemental. The question tells Social Security and SSDI
  recipients: "DO NOT include the $(amount reported) per month from Medicare
  deductions from (Social Security/…) payments mentioned earlier."
- `PEMCPREM` ("Edited Medicare premium amount", universe `MCARE=1`) is that
  monthly Medicare deduction from Social Security, or the monthly Medicare
  payment, with the instruction "Include Medicare Advantage, Part B, and Part D
  premiums." The certified file stores it as `medicare_part_b_premiums`, a
  retired engine input that `microcosm.data.stored_inputs` says the engine
  ignores.

So for an enrollee on Social Security, `PHIP_VAL` holds Medigap, private and
supplemental premiums, and Medicare premiums are in `PEMCPREM`.

## What the engine does with each input

Read in the installed policyengine-us 2.2.1 (the pinned engine):

- `medical_expense_health_insurance_premiums` returns the direct
  `health_insurance_premiums` input when it is nonzero, and otherwise
  `health_insurance_premiums_without_medicare_part_b + medicare_part_b_premium × medicare_enrolled`.
  Its readers: federal Schedule A (`itemized_medical_expenses`), SNAP's
  medical deduction, HUD, Medicaid medically needy, Montana, Ohio's insured
  unreimbursed expenses and Pennsylvania child care.
- `self_employed_health_insurance_premiums` adds `health_insurance_premiums`
  for `is_self_employed` people; its only reader is
  `self_employed_health_insurance_ald_person`.
- `medicare_part_b_premium` is the gross premium (standard plus IRMAA) less
  Medicare Savings Program coverage, for `medicare_enrolled` people.
  `medicare_enrolled` is `takes_up_medicare_if_eligible` for Medicare-eligible
  people, and eligibility is age 65+ or 24 months of SSDI. The certified file
  stores no SSDI-months input, so only the age route is live. A build can
  reproduce the engine's enrollment from stored age and take-up exactly (0 rows
  differ on the full file).
- Five state and local formulas read the bare `health_insurance_premiums`
  input (Missouri's premium subtraction and child-care income, Michigan
  household resources, Ohio's uninsured expenses, Orange County general relief).

So copying a Medicare enrollee's reported non-Part-B premium into the direct
input replaces their decomposed premium (reported plus modeled Part B) with the
reported premium alone, and every medical-expense reader loses their Part B.
That is why the stage withholds it.

## Who the proxy withholds

Stored-data tabulation (`scripts/data_only.py`, household weights) and the
engine's own enrollment (`prepare`), on the certified default:

| | Schedule C rule (shipped) | Widened rule (microcosm#1208) |
|---|---:|---:|
| Self-employed (`is_self_employed`) | 36.75M | 39.27M |
| … in the Medicare proxy | 7.03M | 8.01M |
| … of whom outside measured ESI (withheld today) | 5.59M | 6.30M |
| … … Medicare enrollees in the engine (age 65+ with Medicare take-up) | 4.79M | 5.43M |
| … … not enrollees (mostly under-65 SSDI recipients) | 0.80M | 0.87M |
| … … reporting a non-Part-B premium | 1.91M ($7.50B) | 2.24M ($8.64B) |
| … … enrollees reporting no premium of their own (Part B only) | 3.04M | 3.37M |
| Modeled Part B net of MSP for the withheld group | $7.27B | $8.42B |
| … gross, before MSP coverage | $10.04B | $11.39B |
| CPS-reported Medicare premium (`PEMCPREM`) for the withheld group | $8.69B | $10.38B |

In 2024 on this single-year file, modeled IRMAA, the Part A premium and the
Part D surcharge are zero for this group at both engine versions (IRMAA reads
income from two years earlier, which the file does not hold).

## Options

- **Status quo.** Withhold every premium from the Medicare proxy.
- **B. Non-enrollees only (data).** Narrow the proxy to the engine's own
  enrollment predicate (age 65+ and `takes_up_medicare_if_eligible`), so the
  reported premium is attributed to proxy people the engine does not treat as
  enrollees. Their modeled Part B is zero, so the medical-expense concept stays
  unchanged by construction. A build can compute the predicate from stored
  columns exactly (0 rows differ from the engine on the full file); it must
  track the engine if SSDI months are ever stored (microcosm#876).
- **A. Reported premium for enrollees too (data).** Copy the reported
  non-Part-B premium for enrollees. This replaces their decomposed premium with
  the reported premium alone, so every medical-expense reader loses their Part B.
- **C. Reported premium plus modeled Part B, frozen into the data.** Copy the
  reported premium plus the engine's `medicare_part_b_premium` (net of MSP) into
  the direct input. The resolver is unchanged in 2024, but the frozen amount
  ignores later reforms to Part B, IRMAA or the MSPs, and it uprates with the
  direct input rather than with Part B.
- **E. Engine-side (policyengine-us).** Compute the premium base in the engine
  from the direct input when it is nonzero, and otherwise from the reported
  premium plus the standard Part B premium net of MSP for enrollees, behind an
  employer-plan bar. No Microcosm change is needed: for proxy people who are not
  enrollees the same formula gives the reported premium, so E covers B's people
  too. It stays live under reforms and in later years. Using the full resolver
  (gross Part B including IRMAA) raises `CycleError` on the #10120 head
  (`self_employed_health_insurance_ald_person → … → gross_medicare_part_b_premium
  → filing_status → … → dependent_taxable_ss_magi → self_employed_health_insurance_ald_person`);
  the standard-premium form ran without a cycle.

Sub-choices inside C and E, each measured below: whether to count enrollees
whose only premium is Part B; whether partner-only enrollees count (a partner's
premiums qualify only if the partnership pays or reimburses them); and whether
the CPS-reported Medicare premium should stand in for modeled Part B.

## Measurements

### PolicyEngine-US #10120 head, widened plan-business rule

policyengine-us#10120 at `47bba5e893` (2.38.24; Form 7206 per-business limit),
with microcosm#1208's widened `is_self_employed` (Schedule C, farm or
partnership profit). Period 2024. "Returns" are tax units with a positive
deduction, tax-unit weighted. "Revenue loss" is JCT's tax-expenditure row as
the calibration computes it: income tax with `self_employed_health_insurance_ald`
neutralized minus income tax with it, in the same scenario.

| Scenario | Returns with the deduction | Deduction | per return | of which Medicare proxy | Revenue loss (JCT row) | Federal income tax vs status quo | Medical-expense premiums vs status quo |
|---|---:|---:|---:|---:|---:|---:|---:|
| Status quo (proxy withheld) | 3.73M | $15.98B | $4,284 | $0.00B | $2.56B |  | 0 |
| B. Reported premium, non-enrollees only | 3.91M | $16.70B | $4,276 | $0.72B | $2.75B | −$0.20B | +$0.00B |
| A. Reported premium, all | 5.92M | $22.32B | $3,770 | $6.34B | $3.62B | −$0.98B | −$3.53B |
| C. Reported + modeled Part B (frozen in data) | AssertionError: AssertionError()… | | | | | | |
| E. Engine-side: reported + standard Part B net of MSP | 8.98M | $29.21B | $3,254 | $13.28B |  | −$1.88B | +$0.00B |
| C'. Reported + CPS Medicare premium (PEMCPREM) | 8.45M | $30.72B | $3,635 | $14.82B |  |  | +$2.40B |
| C, only enrollees reporting a premium | 5.92M | $24.90B | $4,204 | $8.92B |  |  | +$0.00B |
| C, partner-only enrollees withheld | 8.53M | $27.92B | $3,272 | $12.00B |  |  | +$0.00B |
| C + modeled Part A and Part D IRMAA | 8.98M | $29.21B | $3,254 | $13.28B |  |  | +$0.00B |
| Engine-side, full resolver (gross Part B incl. IRMAA) | CycleError: CycleError('Circular definition detected on formula self_emp… | | | | | | |
| *SOI Table 1.4, TY2023* | 3.60M | $31.23B | $8,686 | | | | |
| *JCT tax expenditure, CY2024 (calibration target)* | | | | | $8.40B | | |

*Pending (re-run queued): row C failed a float32 input check in the first run and is being re-run, with the neutralized E run that gives E's revenue loss. C is expected to equal E person by person in 2024 (it does on a subsample), but this table will carry the measured row.*

What the runs show:

- **B** (non-enrollees only) adds 0.18M returns and $0.72B of deduction and
  raises the revenue loss by $0.20B. No medical-expense premium changes.
- **A** (reported premium for enrollees) takes $3.53B of Part B out of 1.68M
  enrollees' medical-expense premiums. That cuts the Schedule A medical
  deduction by $1.87B and SNAP by $0.05B, so federal income tax falls only
  $0.98B for $6.34B more deduction. A is dominated by C/E, which add the same
  reported premium without that loss.
- **E** (and C, expected to give the same 2024 values; re-run pending) raises the
  deduction to $29.21B on 8.98M returns. Federal income tax falls $1.88B and
  state income tax $0.36B; medical-expense premiums do not change for anyone.
  The Schedule A medical deduction rises $0.69B because the lower AGI lowers
  its floor.
- **Against SOI**, the status quo already matches the return count (3.73M vs
  3.60M) but holds half the dollars ($15.98B vs $31.23B; $4,284 per return vs
  $8,686). E reaches 94% of SOI's dollars but 2.5 times its returns, and SOI's
  returns already include the Medicare enrollees who claim. Enrollees whose only
  premium is modeled Part B account for 3.05M of E's returns and $4.31B of its
  deduction (E against "C, only enrollees reporting a premium").
- **Partners.** Withholding enrollees whose only profitable business is a
  partnership removes 0.44M returns and $1.29B.
- **CPS-reported Medicare premium** in place of modeled Part B gives $30.72B on
  8.45M returns, but it raises medical-expense premiums by $2.40B for 4.78M
  people, because `PEMCPREM` holds Part D and Advantage premiums the engine's
  Part B does not.
- **Support channel.** Under E, $20.51B of the $29.26B person-level deduction
  is on PUF tax-detail support rows, whose reported premium is the QRF's
  imputation, and $8.76B on ASEC respondents. (The person-level total exceeds
  the $29.21B tax-unit total by $0.06B: on the #10120 head a dependent's own
  deduction is not part of the filer's.)
- **Possible overlap.** 0.98M enrollees outside ESI with no Social Security
  report a premium of their own. The CPS question only excludes Medicare
  premiums deducted from Social Security, so a directly billed Part B premium may
  already be in their reported premium. C and E would then count up to $1.44B
  of their modeled Part B twice; 86% of that is on PUF-support rows. (The
  engine's own medical-expense premium already adds the two for them today.)
- **MSP.** Modeled Part B is net of Medicare Savings Program coverage applied on
  eligibility, with no take-up; MSP covers $2.97B of the $11.39B gross Part B in
  the withheld group.

### PolicyEngine-US 2.2.1 (pinned), Schedule C rule

*Pending: queued under the shared heavy lock.*


## Files

- `scripts/measure.py`, `scripts/run_all.py`: the staged microsimulation runs
  (one simulation per process).
- `scripts/data_only.py`: the stored-data tabulations.
- `outputs/*.json`: aggregate outputs. Unit-level arrays stay out of the
  repository.

Re-run one engine with

```bash
python -I scripts/run_all.py <label> <python> <out_prefix> <populace_us_2024.h5> [--widen]
```

The 2.2.1 runs used a Python 3.13 environment with policyengine-us 2.2.1 and
policyengine-core 3.32.5; the widened runs used the policyengine-us#10120 head
(`47bba5e893`, 2.38.24, core 3.33.2) with `--widen`. Paths inside the outputs
are machine-local.
