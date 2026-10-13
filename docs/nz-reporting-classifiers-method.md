# New Zealand reporting classifier method

The reporting population contains one row per benefit unit. The declared cells
are income decile × family type and the three receipt groups. The classifier
parameters live in
[reporting_classifiers.json](../packages/microcosm-build/src/microcosm/build/nz/reporting_classifiers.json).
This document records the method basis and the checks needed before activation.

## A1 Income decile (provisional)

The provisional stratifier is annual gross benefit-unit income for TY2026–27,
before income tax, Accommodation Supplement, and Working for Families. It
includes market income and paid main-benefit, New Zealand Superannuation, and
Veterans Pension amounts. Including benefit income avoids ranking all units
with no market income as one zero-income tie; excluding Accommodation Supplement
keeps the reported outcome out of its own stratifier. Equivalised disposable
income is a defensible alternative that would change the reporting cells, so
the gross-income basis still needs confirmation before publication.

Market components are employment, nonfarm and farm self-employment, interest,
dividend, rental, and private-pension income. Annual market amounts use an
annualisation factor of one. Benefit income is the assigned receipt flag
multiplied by the engine's before-tax weekly paid amount after its income test,
then multiplied by 52. Entitlement does not substitute for assigned receipt.
All components must use the same before-tax basis. A net paid amount or a gross
maximum rate does not establish the required before-tax paid amount.

The current declaration leaves unavailable before-tax main-benefit and Veterans
Pension paid amount columns null and refuses activation. Veterans Pension
receipt must also be present: a missing flag cannot mean no receipt. Rental
income is after business expenses, and the before-tax interpretation of rental
and private-pension inputs must be verified before activation.

Each benefit unit uses its calibrated weight. The transport declaration selects
the household calibrated-weight carrier and associates its weight with each
unit. Units are sorted by gross income, then by a unique stable unit identifier.
The identifier must survive rebuilds of the same pinned donor and unit rule;
shared source-household lineage does not merge units. Clones remain distinct
reporting units and can land in different cells, including when they have equal
income and weight.

The declared rank convention is the cumulative-weight midpoint. With preceding
weight `P`, unit weight `w`, total weight `W`, and `n` bins, the label is
`min(floor(n * (2 * P + w) / (2 * W)) + 1, n)`. Bins are left-closed: exact
equality with boundary `k/n` enters bin `k + 1`. No unit is split between bins.
Zero-weight units use the cumulative mass at their sorted position; rank one
is clipped into the final bin. An all-zero population is refused. The midpoint
convention makes A1's tie and boundary rule precise and remains provisional.

Each supplied float64 weight is an exact dyadic rational. Expressing weights as
integer numerators over a common power-of-two denominator makes prefix sums and
the midpoint division exact. This determines labels without cumulative floating
point error or a boundary tolerance. A common power-of-two scaling with exactly
representable float64 results preserves every label. Weighted shares are reported
separately using the scenario's weights and need not be exactly one tenth per cell.

The central scenario S0 supplies the baseline cells. Entitlement scenarios
branch after calibration, so their units, weights, and stratifying incomes are
the same. Calibration variants may change weights, and the receipt variant may
change income; each variant carries S0's cell by stable unit identifier. The
weighted share of every fixed cell must be reported for every scenario and
variant. These fixed memberships preserve comparisons even when variant cells
are no longer weighted tenths.

The disclosure on New Zealand Superannuation remains material: the current
shared-residence input defaults false, overstating payments to single people
who share a home and moving those units upward in the income ranking.

## A2 Receipt group

The legal basis is the Social Security Regulations 2018 (LI 2018/202), read for
this method as at 5 April 2026. [Regulation 17, PCO
LMS96264](https://www.legislation.govt.nz/regulation/public/2018/0202/latest/LMS96264.html)
defines a beneficiary by payment of a main benefit, New Zealand Superannuation,
or a veteran's pension. Everyone else is a non-beneficiary for that regulation.
The base-rate rules in regulation 17(2) depend on the applicant's benefit and,
for a couple, include a spouse or partner component.

[Regulation 18, PCO
LMS96265](https://www.legislation.govt.nz/regulation/public/2018/0202/latest/LMS96265.html)
applies its income reduction only to a non-beneficiary as defined in regulation
17. It reduces the assessed amount by 25 cents for each dollar of relevant
weekly income above the applicable threshold. A recipient applicant receiving
New Zealand Superannuation or a veteran's pension therefore does not fall under
that non-beneficiary reduction.

The three-way reporting split is a reporting convention. The legal definition
includes both main-benefit and pension recipients as beneficiaries, and the
base-rate provisions depend on the applicant; they do not establish a priority
between partners in a mixed receipt unit. The reporting convention assigns:

1. `superannuitant` if any member has assigned New Zealand Superannuation or
   Veterans Pension receipt;
2. otherwise `beneficiary` if any member has assigned main-benefit receipt;
3. otherwise `other`.

The current main-benefit predicates name Jobseeker Support, Sole Parent Support,
and Supported Living Payment. The priority order, default group, membership
columns, and receipt predicates are spec parameters. Every required assigned
receipt flag must be present and boolean; missing Veterans Pension support must
be reported and refused rather than assumed false. Each scenario reports the
count of units satisfying multiple receipt predicates.

The S0–S4 Accommodation Supplement exclusion selects the same superannuitant
predicate as the reporting classifier. Their excluded units must equal the
`superannuitant` reporting group. That group's Accommodation Supplement cells
are structural zeros in those scenarios and belong in the gate's declared
structural-zero list; actual zero diagnostics still depend on reported amounts.

The Accommodation Supplement non-beneficiary input also needs a receipt audit.
For an applicant receiving New Zealand Superannuation or Veterans Pension it
must be false under regulation 17(1). The current any-member receipt binding
includes New Zealand Superannuation and the supported main benefits but requires
extension when Veterans Pension receipt becomes available. This audit checks the
legal payment distinction separately from the reporting group's priority rule.
