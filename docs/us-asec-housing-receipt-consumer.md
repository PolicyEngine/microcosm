# ASEC housing receipt consumer and pinned-engine charge probes

`microcosm.build.us_runtime.asec_housing_receipt_consumer` reads the interview-time
housing evidence that [`asec_housing_status`](us-asec-housing-status.md) already
carries and states, explicitly, which SPM units that evidence can and cannot
place. `microcosm.build.us_runtime.housing_charge_probes` measures what the
installed engine actually does with the one input the consumer could ever supply.
Neither module opens a source file, fits a model, derives a rent amount or
produces release-eligible evidence.

## Five quantities, never fused

| Quantity | Grain | Engine binding | Status here |
| --- | --- | --- | --- |
| Observed receipt status | ASEC household interview | `receives_housing_assistance` | The only thing this slice observes |
| Counterfactual full contract charge | SPM unit | `pre_subsidy_rent` | **Unavailable**; needs a year-correct source |
| Reported contract rent | Household | none | Separate measured quantity, retained |
| Modeled HAP | SPM unit | `hud_hap` | Formula output; never an input |
| Capped SPM valuation | SPM unit | `spm_unit_capped_housing_subsidy` | Corroboration only |

`compose_full_contract_charge` refuses every combination. The two shortcuts the
[reported-rent bridge assessment](../../architecture-reviews/microcosm-20260904/us-reported-rent-consumer-bridge.md)
rejected — reported rent plus the capped valuation, and reported rent plus a
modeled HAP computed off the charge being solved for — refuse under their own
names, and the residual case refuses as `FULL_CONTRACT_CHARGE_SOURCE_UNAVAILABLE`.

`corroborate_with_capped_valuation` cross-tabulates the capped amount against the
status and changes nothing. A receipt household with a zero cap and a nonreceipt
household with a positive cap both stay as observed: the capped valuation is a
censored SPM resource, so disagreement is a diagnostic, not a status correction.

## Household status to SPM assignment

`interpret_household_receipt` renames the carried status into an explicit view and
rechecks the carried invariants — code domains, `receipt_valid` agreeing with the
status, conflict exclusivity, and no unknown reason on a valid answer — rather
than trusting them. Unknown, conflict and not-in-universe households keep their
own views. No zero-origin, allocation-quality or routing evidence is dropped, and
nothing invalid becomes observed nonreceipt.

`resolve_spm_receipt_scope` carries the household answer to SPM units without
inventing a recipient:

| Household view | Sole SPM unit | Two or more SPM units |
| --- | --- | --- |
| Receipt | `assigned_receipt_sole_spm_unit` | `withheld_scope_ambiguous_receipt` |
| Nonreceipt | `assigned_nonreceipt_sole_spm_unit` | `withheld_scope_ambiguous_nonreceipt` |
| Unknown | `withheld_unknown` | `withheld_unknown` |
| Conflict | `withheld_conflict` | `withheld_conflict` |
| Not in universe | `withheld_not_in_universe` | `withheld_not_in_universe` |

The positive answer names the housing unit, not one of several SPM units sharing
it. The probes below measure why replication is not conservative: in the pinned
engine each SPM unit draws the household's whole payment standard, so copying one
household's receipt into two units funds two whole voucher budgets inside one
housing unit. Census's proration of its own valuation across SPM units is not
evidence that each unit independently holds a voucher.

The negative answer describes the housing unit itself, so it is separated rather
than fused with the positive case. `project_engine_receipt_inputs` requires the
caller to declare `multi_unit_negative="refuse"` or `"distribute"`; there is no
default. Every other withheld class stays withheld under both policies.

## Refusing the blanket engine Boolean

The projection is a three-state `uint8` code — `false`, `true`, `withheld` — not a
Boolean column, so a withheld unit cannot be mistaken for an observed `False`.
`engine_receives_housing_assistance` returns the Boolean input only when nothing
is withheld and otherwise refuses with the withheld count. No household is dropped
to reach a complete column.

`engine_takes_up_housing_assistance_if_eligible` and
`annual_income_year_participation` always refuse. Take-up is not observed at
interview and the engine's input defaults to `True`; an interview-time answer is
not income-year participation. Both refusals are recorded in the projection
receipt alongside the source authentication, so a synthetically derived projection
stays visibly unauthenticated.

`housing_receipt_diagnostics` reports raw view counts, unknown-reason and conflict
bit counts, per-field allocation quality, zero-origin counts, assignment counts,
the number of multi-SPM-unit households and whether the engine Boolean is
complete. All of it is computed on invented frames here.

## What the pinned engine actually does

`housing_charge_probes` writes five invented households through the real
`PolicyEngineUSEngine.write_dataset` gate and simulates them with
`reform_validation.default_simulate_factory`, once as baseline and once under a
`gov.hud.abolition` reform. At policyengine-us 1.819.0 for 2024:

* **The full charge is fixed and independent.** `pre_subsidy_rent` is identical in
  both runs. It is an engine input with no formula.
* **Assistance policy moves HAP and tenant rent.** Abolition zeroes
  `housing_assistance` and lifts `rent` to the full charge for every unit. It does
  **not** zero `hud_hap`, which keeps computing — an observed HAP is not the
  amount the engine pays.
* **Utility reimbursement cannot create negative tenant rent.** For the household
  paying its own utilities, `hud_gross_rent` is the charge plus a $3,816 utility
  allowance and HAP exceeds the charge, yet `rent` floors at zero rather than
  going negative.
* **The household bill conserves across SPM units.** Tenant rent is allocated in
  proportion to each person's charge within an SPM unit, and the household's three
  people pay exactly the household charge less both units' assistance. Each unit's
  `hud_max_subsidy` is nonetheless the household's whole payment standard less
  that unit's own tenant payment.
* **The receipt flag has real dollar consequences.** Two units differing only in
  `receives_housing_assistance` differ by $8,400 of modeled assistance, because
  current receipt bypasses the modeled renter/income test. That gap is what an
  unjustified `False` would assert.
* **Formula-owned outputs stay refused.** `rent`, `housing_assistance`, `hud_hap`
  and `spm_unit_capped_housing_subsidy` each block the export with nothing
  written. The consumer's declared seam names the same four.

## Blocking defects, not launch permission

The probes also record that the bundled fair-market-rent table carries only fiscal
years 2025 and 2026, so a 2024 request falls forward to FY2025, and that the
utility allowance is zero outside the encoded counties. Bedrooms are not observed
for the assisted universe and `pha_payment_standard` carries an undated Los
Angeles branch. `counterfactual_charge_availability` reports the charge as
unavailable and names these defects. They are unresolved consumer defects; an FMR
shortcut without a year-correct source and an explicit tenant-paid utility
contract is refused, not deferred.
