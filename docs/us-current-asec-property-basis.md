# ASEC property donor basis

`build_asec_property_basis` is a pure operation over the selected, qualified ASEC
person descriptions. It does not qualify a source, issue a handle, read files,
fit a model or assign tax leaves. The owning country host retains the actual
preparation and source owners, checks them before consumption and requalifies
and checks their complete seals after its last relevant I/O before return or
export. Reconstructing the descriptive input or output dataclasses grants no
source authority.

## Inputs and outputs

The keyword-only inputs are the `person` tables from the current interest,
income-routing and dividend/survivor qualifiers, named `interest`,
`income_routing` and `dividend`. Every table must have the same exact ordered
`person_id` index, matching unique `native_person_id` values and matching source
ages. Both ID axes are int64. Source ages may use different numeric storage
but must be finite, integer-valued ages 0–99 and equal without tolerance. No
sorting or inner join repairs a mismatch. A common row permutation is valid;
permuting one source alone refuses.

`original_household_membership` is an int64 Series on that exact person index.
`original_household_design_weights` is a finite, nonnegative Series indexed by
unique int64 `household_id`, with exactly the membership's household set.
Extra, missing or duplicated household keys refuse. These inputs must be the
original household design mapping, before importance allocation or cloning;
this pure operation cannot authenticate that origin. Zero-weight persons are
retained. No engine or microsimulation aggregation occurs.

The descriptive `AsecPropertyBasis` contains:

- `person`: the component values, reported total, discrepancies, named retirement
  derivations, original household mapping and the two eligibility masks.
- `provenance`: detached copies of all three supplied qualified tables, with
  `interest.`, `income_routing.` and `dividend.` prefixes. Published amounts,
  unknownness, allocation and disclosure indicators are preserved independently.
- `exclusions`: independent reasons for exclusion on the complete person axis.
  A person can have multiple reasons; reason totals must not be summed as a
  disjoint partition.
- `summary`: counts and original design-weight mass for the whole cohort, each
  eligibility mask, the joint-fit complement and each exclusion reason.

The shared names live in `property_income_constants.py`. `PROPERTY_COMPONENTS`
is, in order, `property_ordinary_interest`, `property_retirement_interest`,
`property_dividends`, `property_broad_receipts`. `PROPERTY_REPORTED_TOTAL` names
`property_reported_total`. They are donor observations or named routing
contributions, never already-reconciled model draws.

## Measurement bridge

Ordinary interest uses qualified `TRDINT_VAL`; dividends use qualified
`DIV_VAL`; broad signed property receipts use qualified `RNT_VAL`. The latter
is not a pure rental-income tax leaf. Dividends need a qualified yes/positive
or no/zero receipt. A receipt-yes zero for dividends or signed property stays
unknown under the published none-or-NIU coding.

Retirement-account interest uses the two qualified account slots. Known no
with both slots NIU and zero supplies the already-qualified zero contributions.
Known yes requires at least one declared account with a known positive amount.
Every other slot must be another known positive account or an explicitly unused
slot: readable account code 0, published amount 0, in-range amount literal and
`unreported_account_slot` source status. The latter contributes a **derived**
zero and gets its own boolean, slot count and derivation label. Its original
canonical amount stays unknown in provenance. An amount zero by itself,
a declared account with unknown/zero amount, yes without any active account,
a missing code, or contradictory receipt/account evidence does not resolve the
retirement total. The derivation applies only when the complete route resolves.

The reported donor total is separately retained as `INT_VAL + DIV_VAL + RNT_VAL`.
The component sum uses ordinary plus retirement-account interest, dividends and
broad property receipts. Both `interest_component_discrepancy` and
`reported_minus_component_total` remain visible, with signed magnitudes. No
balancing rewrites an ASEC observation. Allocation never filters the donor fit.
Disclosure flags do not prove the cause of a discrepancy.

Other-income clearance requires known nonreceipt with a NIU category, or a
readable reported category outside property codes 5–8 and unspecified code 19.
A reported outside category resolves its route even if its own dollar amount
is ambiguous; no other-income amount is included in this basis. Missing,
unreadable or contradictory routing stays unresolved. The dividend qualifier's
nullable survivor clearance is checked against its readable codes and retained
as `survivor_visible_routes_clear`. It covers only `SUR_SC1` and `SUR_SC2`.
The 2025 dictionary describes `SRVS_VAL` as including edited sources 1/2 plus
unedited sources 3/4 (PDF page 49, printed page 6C-28); those additional source
types are not qualified here. A positive survivor receipt therefore cannot
establish complete absence of estate/trust income even when both visible
slots are outside codes 8 and 10.

The first bridge accepts only known survivor nonreceipt with readable NIU
slots, exposed as `survivor_full_scope_clear`. Positive receipts have a separate
`survivor_additional_sources_unresolved` exclusion and weighted coverage row.
Possible visible property overlaps and unknown visible routes retain their own
exclusion reasons. Overlapping exclusion rows must not be added together.
Neither other income nor survivor amounts enter the component or reported sum.

`reported_total_eligible` requires age 15+, a known finite reported total, and
clear other-income routes and complete survivor scope. It is a diagnostic sample, not a second
aggregate model. `joint_component_fit_eligible` additionally requires all four
components and an exactly zero interest discrepancy. Missing or under-15 values
are never turned into analytic zeros. Negative and zero net totals remain valid:
`(100, 20, 0, -120)` has total zero and keeps both positive interest components.

The summary reports `design_weighted_person_mass` (one mapped original household
design weight per selected person) separately from `union_household_design_mass`
(each household with at least one selected person counted once). Overlapping
selection groups can share household mass. Stable finite summation avoids
row-order-dependent loss of small weights. These are coverage diagnostics, not
calibrated population claims. Conditional donor support, bias from exclusions
and held-out model quality still need the model owner's diagnostics.

## Source and acceptance boundaries

The source contracts use the pinned
[2025 ASEC dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
(SHA256 `5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`).
The bridge targets the broad property concept in the
[2024 ACS questionnaire](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2024/quest24.pdf#page=18).
The live ACS guide follows the 2025 form; its retirement-interest wording and
the historical 2016 Census methods discussion corroborate the concept without
establishing vintage or respondent identity. ASEC calendar-year and ACS
rolling-year measurement remain different. Broad property routing and
retirement-account earnings remain distinct from taxability.

The bounded tests use invented tables and a composition test through the real
pure source projectors using invented literals. They cover both identity axes,
permutations, missing/contradictory routes, unknown account slots, signed and
zero-net examples, separate masks, allocation preservation, weight mapping,
empty/zero-weight records and refusal of malformed/nonfinite descriptions.
They do not run source qualification, native data, the country model, fitting,
cloning or replay. The later graph owns joint draws and signed reconciliation;
the later host must preserve original ASEC observations and share survey draws
across original/PUF clone pairs before the PUF alternative takes ownership.
