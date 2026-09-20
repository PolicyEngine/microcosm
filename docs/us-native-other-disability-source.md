# Native ASEC other-disability source

`qualify_current_asec_other_disability(preparation)` reduces the two published
CPS ASEC disability income slots to one non-workers-compensation leaf. It reads
no raw member: it borrows
[`current_asec_retirement_detail_source`](../packages/microcosm-build/src/microcosm/build/us_runtime/current_asec_retirement_detail_source.py),
which already qualifies `DIS_VAL1/2`, `DIS_SC1/2`, `DIS_YN` and their allocation
and topcode literals against the retained money owner. It issues no source
capability, fits nothing, and attaches no canonical consumer input. Callers
retain the genuine owners and requalify after their relevant I/O.

## What the leaf is

The retired eCPS pipeline summed the two annual slots wherever the slot's
source code was not 1, workers' compensation. That arithmetic lives in
`disability_benefits.derive_us_disability_benefits_from_asec`, and this adapter
reuses that function and its archived parameter dictionary rather than
restating them; a drift in either refuses with `ARCHIVED_ROSTER` or
`IMPLEMENTATION_CHANGED`. The exclusion is bound to the printed meaning of code
1, not to the bare integer, so renaming the published label refuses too.

Social Security disability and workers' compensation are separate leaves and
are not folded in here. The `DIS_CS` and `DIS_HP` work-limitation answers the
detail owner keeps beside these slots describe leaving or limiting work; they
are not this leaf's dollars, and their allocation flags are deliberately
outside this family's provenance.

## What stays unknown

The archived arithmetic silently produced a zero for every row it could read.
This adapter does not. Each slot is classified into one closed vocabulary
(`SLOT_KINDS`); only the first four resolve, and the rest leave the person
unknown:

| slot kind | contributes | when |
| --- | --- | --- |
| `reported_source_slot` | the published amount | yes receipt, readable non-workers-compensation code, nonzero amount |
| `excluded_workers_compensation` | 0 | yes receipt and a readable code 1, whatever the slot paid |
| `unused_source_slot` | 0 | yes receipt, code 0, zero literal: no source in this slot |
| `nonreceipt_slot` | 0 | an observed "no" to `DIS_YN` |
| `niu_not_observed_zero` | unknown | `DIS_YN = 0` |
| `outside_age_universe` | unknown | under the printed 15+ reporting universe |
| `unresolved_slot_reporting` | unknown | missing, malformed, out-of-range or contradictory literals |

A person's leaf is known only when both slots resolve. A "yes" answer with no
populated source in either slot is `affirmed_receipt_without_reported_source`
and stays unknown rather than becoming a zero. A recipient zero under a
reported non-workers-compensation source stays ambiguous, because the printed
entry says `0 = none or niu`. Under-15 rows and NIU rows stay unknown even
though the archived arithmetic reads them as zero; the divergence is recorded
in `other_disability_archived_arithmetic_amount` and never adopted.

## Provenance is not knownness

Every published allocation flag for this family (`I_DISYN`, `I_DISSC1`,
`I_DISSC2`, `I_DISVL1`, `I_DISVL2`) is evaluated against its own printed
conditional universe, read from the detail owner's entries rather than
restated. A flag outside its universe, unpopulated, or unreadable is labeled as
exactly that. A publisher allocation never qualifies or disqualifies a receipt:
`other_disability_allocation_qualifies_receipt` is `False` on every row.
Topcode flags describe only the dollars this leaf admits, and
`topcode_corrected` stays `False`.

## Clone transport

`attach_other_disability_columns(values, receiving)` is the thin attachable
interface. It joins on the pre-clone source id, checks that each receiving row's
spine source identity matches the qualified row's `native_person_id`, requires a
uniform clone index set per original, and copies one qualified row to every
clone of that person. No draw is taken and no value is redrawn; the returned
receipt records `redraw_issued: False` and `draws_consumed: 0`. Rows on an arm
this source never observes — ACS today — carry a null amount, `known` false and
the reason `acs_source_unobserved`. They are never zero.

The attachment returns the canonical leaf under `disability_benefits` and every
other column under the `survey_other_disability_` report prefix. It returns
columns to the host and mutates no receiving cell.

## Boundaries

The pure `project_other_disability` function accepts a supplied detail basis so
its semantics can be inspected and tested; neither it nor
`other_disability_values_seal` confers authority. `compose_other_disability`
seals the borrowed detail owner before and after projection and rechecks its own
active implementation dependencies, and
`qualify_current_asec_other_disability` additionally rechecks the retained
preparation and native issuance.

ACS completion, SSI eligibility, tax treatment, topcode correction, host
composition, calibration and release remain separate work owned elsewhere. The
tests use invented literals; passing them establishes these contracts, not
actual-data quality, coverage or release readiness.
