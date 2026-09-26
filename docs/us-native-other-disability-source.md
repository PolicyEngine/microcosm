# Native ASEC other-disability source

`qualify_current_asec_other_disability(preparation)` reduces the two published
CPS ASEC disability income slots to one non-workers-compensation leaf. It adds
no raw reader of its own: it borrows
[`current_asec_retirement_detail_source`](../packages/microcosm-build/src/microcosm/build/us_runtime/current_asec_retirement_detail_source.py),
which already qualifies `DIS_VAL1/2`, `DIS_SC1/2`, `DIS_YN` and their allocation
and topcode literals, and cross-checks the two retained `DIS_VAL` amounts — and
only those — against the authenticated money owner's own bits. The public entry point
calls that owner's qualifier, so the single member capture and its
requalification happen there and are not repeated or replaced here; the pure
`compose_other_disability` and `project_other_disability` entry points read no
member at all. This module issues no source capability, fits nothing, and
attaches no canonical consumer input. Callers retain the genuine owners and
requalify after their relevant I/O.

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

The archived arithmetic returned a number for every row whose four literals
it could read, and never an unknown; a row that was never asked the question
read whatever its literals happened to hold, which on a consistent
never-asked row is zero and on a contradictory one need not be. This adapter does not. Each slot is classified into one closed vocabulary
(`SLOT_KINDS`); only the first four resolve, and the rest leave the person
unknown:

| slot kind | contributes | when |
| --- | --- | --- |
| `reported_source_slot` | the published amount | yes receipt, readable non-workers-compensation code, nonzero amount |
| `excluded_workers_compensation` | 0 | yes receipt, a readable code 1, and a readable amount, whatever that amount was |
| `unused_source_slot` | 0 | yes receipt, code 0, zero literal: no source in this slot |
| `nonreceipt_slot` | 0 | a "no" to `DIS_YN` with code 0 and a zero literal |
| `niu_not_observed_zero` | unknown | `DIS_YN = 0` with code 0 and a zero literal |
| `outside_age_universe` | unknown | under the printed 15+ reporting universe |
| `unresolved_slot_reporting` | unknown | missing, malformed, out-of-range or contradictory literals |

A receipt answer alone does not settle a slot. A "no" or an NIU answer that
arrives beside a nonzero amount or a populated source code is a contradiction,
not a nonreceipt or an NIU: the owner labels it as such and the slot stays
unresolved.

A workers' compensation slot whose amount cell is missing or unreadable is
`unresolved_slot_reporting`, not a known zero: a record that cannot be read
here is not evidence that its source code was read correctly.

A person's leaf is known only when both slots resolve. A "yes" answer with no
populated source in either slot is `affirmed_receipt_without_reported_source`
and stays unknown rather than becoming a zero. A recipient zero under a
reported non-workers-compensation source stays ambiguous, because the printed
entry says `0 = none or niu`. Under-15 rows and NIU rows stay unknown
wherever the archived arithmetic would have read them; whatever it read — zero
on a consistent row, a positive number on a contradictory one, or nothing at
all when a literal will not parse — is recorded in
`other_disability_archived_arithmetic_amount` and
`other_disability_archived_arithmetic_evaluable`, and never adopted.

That replay runs over the four fields **as the owner parsed them**, so it is
evaluable only where each read inside its own printed domain. The archived
function itself accepted any finite number, so a source code outside the
published table — `11`, say — is a row the retired pipeline would have summed
and this replay reports as unevaluable. It is a replay of the admitted
literals, not of every byte the old pipeline consumed.

## Provenance is not knownness

Every published allocation flag for this family (`I_DISYN`, `I_DISSC1`,
`I_DISSC2`, `I_DISVL1`, `I_DISVL2`) is evaluated against its own printed
conditional universe, read from the detail owner's entries rather than
restated. A flag outside its universe, unpopulated, or unreadable is labeled as
exactly that, in that order of precedence: a flag whose universe cannot be
resolved, or that sits outside it, is answered before its own literal is
judged, so an unpopulated or malformed literal is only reported where the flag
was in its universe to begin with.

Two family-level answers are reported, under names that say which is which.
`other_disability_published_flag_origin` is the routing owner's reading, which
is universe-blind by that owner's charter: a nonzero flag read outside its own
printed universe can still count as a publisher allocation there — unless some
flag's literal is unresolved, which that owner answers first.
`other_disability_allocation_status` is this module's own reading, built from
the per-flag labels above, so a flag outside its universe does not allocate
this family. Its precedence is: an in-universe publisher allocation settles the
family; otherwise an unresolved universe or literal, then an unpopulated flag,
then a clear in-universe zero; and `no_flag_in_its_universe` only where every
printed universe — all of which are `X > 0` here — is false. Neither qualifies or disqualifies a receipt:
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
seals the borrowed detail owner before and after projection and rechecks its
own active implementation dependencies. That fence binds every module-level
constant of this file, detached rather than aliased; every function defined in
it; the file's bytes as read at the end of its own import, so an edit after
that point refuses instead of being reported — the window between the loader's
read and that one is not attested — and an **enumerated** set of borrowed callables — the detail
owner's qualifier, literal projection, slot status, amount pair, member capture
and amount comparison and seal; the routing owner's receipt, literal,
code-frame, capture, digest and allocation helpers; and the archived arithmetic
with its own input guard and parameter dictionary. That list is deliberate, not
a transitive closure: it covers the callables this module's own path depends on
for the literals it reads, and it does not attest anything those callables in
turn call. `qualify_current_asec_other_disability` additionally rechecks the
retained preparation and native issuance.

The detail owner cross-checks the retained `DIS_VAL1/2` amounts against the
authenticated money owner's bits, but `DIS_YN`, `DIS_SC1/2` and the allocation
and topcode literals reach this leaf from the member capture alone. That is why
the capture path is inside the fence, and it is a boundary a consumer should
know about rather than a property this module can prove on its own.

ACS completion, SSI eligibility, tax treatment, topcode correction, host
composition, calibration and release remain separate work owned elsewhere. The
tests use invented literals; passing them establishes these contracts, not
actual-data quality, coverage or release readiness.
