# Development Social Security beneficiary convention

`microcosm.build.us_runtime.survey_social_security_beneficiaries` provides a
pure allocation contract for an explicitly restricted development subset. It
does not qualify actual survey sources, attach graph or Frame columns, run an
engine, or make a release ready. Its public dataclasses contain descriptive
claims, not source-owner capabilities. Validation establishes internal
consistency only.

## Report evidence and beneficiary income

The existing Social Security report classifier remains unchanged. Its category
probabilities and conditional mean report shares do not identify beneficiaries
or their benefit incidence. In particular, converting four positive scores
into four positive canonical benefit amounts would manufacture component
incidence. The new core has no such conversion.

Reporting identity need not equal beneficiary identity. ASEC permits combined
family reports; its reason 7 does not distinguish disabled, dependent, and
surviving children. ACS combines Social Security and Railroad Retirement and
uses a rolling twelve-month period. See the [2025 ASEC dictionary, pages
48–49](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
and [2024 ACS questionnaire, page
18](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2024/quest24.pdf).

Tax treatment can depend on the entitled beneficiary even when another person
receives the payment. [IRS Publication 915, “Who is
taxed”](https://www.irs.gov/publications/p915). Children can receive family
benefits; exclusion from a survey's reporting universe is not benefit
ineligibility. [SSA family eligibility](https://www.ssa.gov/family/eligibility).

## Implemented convention

`resolve_singleton_asec_retirement` requires both keyword assumptions to be
explicitly true:

- `closed_benefit_unit`: this report includes no nonresident's benefit, and
  this resident has no benefits reported elsewhere.
- `stable_roster_over_income_year`: the interview roster is an adequate
  benefit allocation boundary throughout the income year.

The helper accepts only a complete, one-person ASEC housing-unit roster whose
sole person is the reporter; a positive finite annual OASDI report with receipt
code 1; age within the ASEC 15+ reporting universe; known reasons containing
only retirement and optional NIU; and no publisher allocation. This is a
declared source convention, not statutory eligibility logic. Interview age
does not override the reported retirement category.

The full report amount is assigned to that person's retirement component in
one exact edge. The other three canonical components are zero **by the
convention**, not observed beneficiary zeros. The output records
`resolved_by_convention`, the two assumptions, definition resolution and
calendar-year resolution. It makes no observed-beneficiary, source-admission,
engine-compatibility or release claim.

ACS, reason 7/8, other/missing reasons, unknown/incomplete rosters, group
quarters, publisher allocation, unknown money, report zero, unresolved
definition and noncalendar periods cannot use this shortcut. These cases need
separate source evidence or an accepted future allocation model.

## Artifacts and consumer boundary

- `ReportLot` retains the original survey/vintage/reporter coordinates,
  amount or `None`, raw reasons, receipt, publisher allocation, definition and
  period. Amount and period changes do not create a new lot identity.
- `BeneficiaryCandidate` is a possible person/component edge with a basis;
  it has no money. It can describe a child or unknown/nonresident possibility
  without assigning incidence or an equal share. Candidates are descriptive;
  this slice does not authenticate relationships or construct a family graph.
- `LotAllocation` keeps candidates, assignments and explicit resolution.
  `unresolved_report_lot` leaves the complete reported amount unresolved.
  Unknown (`None`), zero, and positive unresolved amounts are distinct.
  Zero unresolved dollars do not mean resolved beneficiary assignment.
- `validate_development_consumer_subset` revalidates every supplied lot and
  returns four-vectors only for an exact, nonempty requested person subset and
  calendar year. Any unresolved lot fails. Repeated original lot coordinates
  fail even if their payloads differ. Multiple reports for one beneficiary
  fail because overlap has not been resolved. The vectors use the existing
  `survey_social_security.COMPONENTS` order and retain convention origin.

The consumer check cannot discover omitted lots or prove that a caller's
closed-benefit-unit assumption is true. The returned object is a structural
development contract, not an admitted engine input. Do not drop uncertainty
flags or pass unknown zeros to the engine. A parent combined report plus a
child's independent report must not be automatically added or subtracted.

## Integration still required

A later graph adapter must bind the live source owner and physical digest,
original source-person identities, the full qualified roster and readset, and
source period and definition. It must preserve report evidence separately from
the existing classifier, expose each candidate/allocation edge, validate the
consumer subset, and attach only through the exact supported original/clone
mapping. Renamed report keys cannot replace that source ownership check.

No graph adapter, actual-source qualification, clone fan-out, cold/required
replay, consumer engine execution, or national/CD completeness proof is
implemented by this core. It neither changes weights nor synthesizes missing
persons or reports. Family allocation, report overlap, ACS Railroad Retirement
separation and calendar conversion remain separate prerequisites for broad
coverage.

The invented test module covers the accepted convention, refusal boundaries,
report-zero/unknown distinction, candidate-only children, exact single-lot
conservation, duplicate/overlapping reports, constructible-artifact
revalidation, and the positive-class-score incidence regression. Those tests
do not establish scientific validity or source authority. Poverty remains
comparison-only: no target, tuning, candidate selection or release gate; no
prior wages are introduced.
