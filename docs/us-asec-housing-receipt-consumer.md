# ASEC housing receipt consumer and pinned-engine charge probes

`microcosm.build.us_runtime.asec_housing_receipt_consumer` reads the interview-time
housing evidence that [`asec_housing_status`](us-asec-housing-status.md) already
carries, together with the occupied-housing-unit evidence
[`asec_housing_universe`](us-asec-housing-universe.md) already carries, and
states explicitly which SPM units that evidence can and cannot place.
`microcosm.build.us_runtime.housing_charge_probes` measures what the installed
engine actually does with the one input the consumer could ever supply. Neither
module opens a source file, fits a model, derives a rent amount or produces
release-eligible evidence.

## Five quantities, never fused

| Quantity | Grain | Engine binding | Status here |
| --- | --- | --- | --- |
| Observed receipt status | ASEC household interview | `receives_housing_assistance` | The only thing this slice observes |
| Counterfactual full contract charge | SPM unit | `pre_subsidy_rent` | **Unavailable**; needs a year-correct source |
| Reported contract rent | Household | none | Separate measured quantity, retained |
| Modeled HAP | SPM unit | `hud_hap` | Formula output; never an input |
| Capped SPM valuation | SPM unit | `spm_unit_capped_housing_subsidy` | Corroboration only |

`compose_full_contract_charge` refuses every combination. The two shortcuts the
earlier reported-rent bridge assessment rejected — reported rent plus the capped
valuation, and reported rent plus a modeled HAP computed off the charge being
solved for — refuse under their own names, and the residual case refuses as
`FULL_CONTRACT_CHARGE_SOURCE_UNAVAILABLE`. What a real charge source would have
to establish is written down in
[`us-housing-counterfactual-charge-requirements.md`](us-housing-counterfactual-charge-requirements.md),
which `counterfactual_charge_availability()` names as a repository-relative
path. It is a required future measurement; this consumer implements no part of
it and produces no charge.

`corroborate_with_capped_valuation` cross-tabulates the capped amount against the
status and changes nothing. A receipt household with a zero cap and a nonreceipt
household with a positive cap both stay as observed: the capped valuation is a
censored SPM resource, so disagreement is a diagnostic, not a status correction.

## Reading the evidence, and refusing to guess at its axis

`receipt_columns_from_status` projects a validated artifact: its derived columns,
its `household_id` and `income_year` axes, its codebook, its
`non_housing_unit_scope` declaration and its own `source_authentication`.

`receipt_columns_from_attached_household` takes a `HousingStatusAttachedAsec` —
the result of `attach_housing_status` — validates it, and derives the
authentication label, period kind, status content digest and household axis from
that attachment. There are no free-string provenance parameters, so a caller
cannot label synthetically classified columns byte-verified. A plain DataFrame is
refused. The attached `uint8` columns are read at their own dtype and never cast,
because a silent `uint8` cast of a wider column would wrap an out-of-domain value
before any validation saw it.

`interpret_household_receipt` renames the carried status into an explicit view and
rechecks the carried invariants — code domains, `receipt_valid` agreeing with the
status, conflict exclusivity, household-axis uniqueness, and the carried
codebook's agreement with the view names this consumer renames into — rather than
trusting them. Unknown, conflict and not-in-universe households keep their own
views. Every one of the fourteen derived columns is kept, including `route`,
`group_quarters`, `HPUBLIC_valid`, `HLORENT_valid` and the four zero-origin
columns, and nothing invalid becomes observed nonreceipt. The stored arrays are
non-writeable.

`resolve_spm_receipt_scope` requires the caller's household axis to be the
artifact's own axis in the artifact's own order, and refuses anything else with
`CALLER_HOUSEHOLD_IDENTITY`. Person order and SPM-unit order stay free: they are
resolved by identity, so permuting them changes nothing. Before this binding
existed, a reordered household axis silently placed one household's answer on
another's housing unit.

## Occupied-housing-unit evidence is required, not assumed

An engine assignment asserts something about an occupied housing unit. The ASEC
status alone does not establish one: its frozen classifier declares
`non_housing_unit_scope` as `not_derived`, and it classifies a group-quarters
household (HRHTYPE 9 or 10) with a public-housing yes as ordinary receipt.

`resolve_spm_receipt_scope` therefore takes a required `housing_unit` argument
built by `housing_unit_evidence_from_universe` or
`housing_unit_evidence_from_attached_universe`. There is no default and no
opt-out: missing evidence can never become a `True` or a `False`. The evidence
must cover the same households, in the same order, for the same income-year
cohort, at the same authentication as the status. Authenticated artifacts must
also carry the same source identity: household identifiers are source-local
ordinals and can collide across unrelated checkpoints. The consumer compares
digests of the authenticated source identities, without publishing their contents.
When both inputs are attachments, the universe attachment must name the status
attachment's receipt as its parent. Matching rows alone cannot establish that
attachment chain. Synthetic evidence keeps its unauthenticated status and carries
no authenticated source identity.

| Occupied-HU evidence | Assignment |
| --- | --- |
| Occupied housing unit, primary household | The status table below applies |
| Outside a housing unit (group quarters) | `withheld_outside_housing_unit` |
| Not established | `withheld_housing_unit_not_established` |
| Household kind disagrees with the status's own `group_quarters` | `withheld_housing_unit_conflict` |

The raw observation survives all four: a group-quarters household still counts as
`observed_receipt` in `household_view_counts` and in the diagnostics. Only the
assignment is withheld.

The projection receipt labels the count grains explicitly: household views and
occupied-HU counts describe households, while assignment counts describe SPM units.

## Household status to SPM assignment

With bound occupied-HU evidence, `resolve_spm_receipt_scope` carries the
household answer to SPM units without inventing a recipient:

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
default. `distribute` moves only `withheld_scope_ambiguous_nonreceipt` rows with
bound occupied-HU evidence. Every other withheld class — unknown, conflict,
not-in-universe, ambiguous receipt and each occupied-HU class — stays withheld
under both policies.

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
receipt alongside the status authentication, the occupied-HU authentication and
both content digests, so a synthetically derived projection stays visibly
unauthenticated.

`housing_receipt_diagnostics` reports raw view counts, route counts,
group-quarters counts, per-answer validity counts, unknown-reason and conflict
bit counts, per-field allocation quality, zero-origin counts, the carried
`non_housing_unit_scope` declaration, assignment counts, occupied-HU and
household-kind counts, the number of multi-SPM-unit households and whether the
engine Boolean is complete. Every code name and bit name is read out of the
carried source codebook rather than a copy kept in the consumer, so a codebook
change in a frozen artifact moves these labels instead of silently
desynchronizing them. All of it is computed on invented frames here.

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
  going negative. The more useful way to say it: the engine discards the excess
  rather than paying it out. The HCV utility reimbursement is not modeled. That
  is a model gap listed with the blocking defects below, not a consumer defect.
* **The household bill conserves across SPM units.** Tenant rent is allocated in
  proportion to each person's charge within an SPM unit, and the household's three
  people pay exactly the household charge less both units' assistance. Each unit's
  `hud_max_subsidy` is nonetheless the household's whole payment standard less
  that unit's own tenant payment.
* **The receipt flag has real dollar consequences.** Two units differing only in
  `receives_housing_assistance` differ by $8,400 of modeled assistance. Two
  engine behaviours produce that gap and both are load-bearing: current receipt
  bypasses the modeled renter and income test in
  `is_eligible_for_housing_assistance`, and `takes_up_housing_assistance_if_eligible`
  defaults to `True`, so a unit made eligible by the flag also takes up by
  default. That gap is what an unjustified `False` would assert — and what an
  unjustified `True` would create.
* **Formula-owned outputs stay refused.** `rent`, `housing_assistance`, `hud_hap`
  and `spm_unit_capped_housing_subsidy` each block the export with nothing
  written. The consumer's declared seam names the same four.

`compose_full_contract_charge` refuses by name and never inspects values, and
there is no producer of `pre_subsidy_rent` anywhere in this consumer — that
absence, not the declaration, is what keeps a reported rent out of the charge
input. The export gate accepts `pre_subsidy_rent` as an ordinary input, so the
declaration is a statement of intent rather than an enforcing gate.

## Blocking defects, not launch permission

The probes also record that the bundled fair-market-rent table carries only fiscal
years 2025 and 2026, so a 2024 request falls forward to FY2025, and that the
utility allowance is zero outside the encoded counties. Bedrooms are not observed
for the assisted universe, and `pha_payment_standard` selects a Los Angeles
County schedule from hard-coded dollar amounts with no year parameter — the probe
suite does not pin that branch's value, so it is named as an undated branch read
in the installed formula, not as a measured number. The unmodeled HCV utility
reimbursement belongs on the same list. `counterfactual_charge_availability`
reports the charge as unavailable and names these defects. They are unresolved
consumer defects; an FMR shortcut without a year-correct source and an explicit
tenant-paid utility contract is refused, not deferred.
