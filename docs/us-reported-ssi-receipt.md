# US reported SSI receipt (`receives_ssi`)

PolicyEngine-US has a `receives_ssi` input: a monthly person boolean with
default `False` and no formula. Its rules that key on actual SSI receipt read
`(ssi > 0) | receives_ssi`. Some grant eligibility to recipients: state
supplements (for example Georgia, Indiana, Michigan, Pennsylvania, Washington
and Wyoming on main, and Colorado's AND-CS under 9 CCR 2503-5 §3.546 once
PolicyEngine/policyengine-us#10145 merges), the Medicaid SSI categories, SNAP
categorical eligibility and Head Start. Others exclude recipients, such as
several California county general-assistance programs. Reported receipt
cannot switch modeled receipt off; that stays the job of
`takes_up_ssi_if_eligible`.

The published default `populace-us-2024-spm-20260915` carries the raw ASEC
answers (`SSI_VAL`, `SSI_YN`, `RESNSSI1`, `RESNSSI2`) but no `receives_ssi`.
The engine therefore treats every person as not reporting receipt, and each
receipt-keyed rule depends only on modeled SSI.

## What the base build now carries

`cps_carried.reported_ssi_receipt` sets `receives_ssi` from values the
respondent gave. A person is marked when:

- their annual `SSI_VAL` is positive; and
- at least one reported reason is their own (disabled, blind or other), or no
  reason is recorded.

Two groups are deliberately not marked:

- **Payees for a child.** A respondent whose only reasons are "on behalf of a
  disabled child" or "on behalf of a blind child" (`RESNSSI` codes 3 and 4;
  [2024 ASEC data dictionary](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar24.pdf))
  is the child's payee, and the amount on their record is the child's
  benefit. On the pinned default these are 133 rows (446,000 weighted people
  in 2024).
- **Children under 15.** ASEC asks SSI only of people aged 15 and older
  (`SSI_YN` is not in universe below 15).

Neither case names the child who receives the payment, and the producer does
not guess one. On the pinned default, `SSI_YN == 1` exactly when
`SSI_VAL > 0` (1,510 of 1,510 rows).

The annual report is broadcast to all twelve modeled months, like
`receives_snap`, `receives_tanf` and `receives_wic`, because the annual
source cannot show when receipt started or ended. Only the receipt fact is
carried. ASEC defines SSI as payments from "state or local welfare offices,
the Federal government, or both", so a reported amount may include a state
supplement, and PolicyEngine-US has no amount input that a microsimulation
should fill. Sending the amount as the `ssi` input would replace the federal
SSI formula for every reporter, so SSI reforms would no longer move them. The
`ssi_reported` input is deprecated and feeds nothing.

The producer runs before cloning, so PUF tax-detail clones carry the same
value as their ASEC source person. This is the same as `SSI_VAL` itself, the
take-up anchor and the other receipt inputs. The stacked ACS chain transfers
it with the other CPS-carried person inputs (family
`source_operator_cps_carried`).

## Release contract

`receives_ssi` is a hard requirement of the US release input-coverage gate,
with no reviewed exclusion
(`release_input_coverage.US_ASEC_REPORTED_SSI_RECEIPT_REQUIRED_INPUTS`). The
same gate checks two invariants on the export's person table:

- every marked person reports a positive `SSI_VAL` (where `SSI_VAL` is
  observed); and
- every marked person carries `takes_up_ssi_if_eligible`. The take-up stage
  anchors every `SSI_VAL` reporter as a claimant, and the receipt is a subset
  of those reporters.

The published default fails the gate on `receives_ssi` until a national build
persists it.

## Measured effect

[`evidence/us-reported-ssi-receipt/`](evidence/us-reported-ssi-receipt/README.md)
runs PolicyEngine-US on the pinned default with and without the column. It
also breaks down why the model and the survey disagree.
