# NSECE attendance population qualification — 2026-09-12

Related: [#915](https://github.com/PolicyEngine/microcosm/issues/915) and
[PR #916](https://github.com/PolicyEngine/microcosm/pull/916).

The real survey adapter, ASEC harmonization, fiscal-builder attendance stage,
and native population export are implemented. The final aggregate reports below
record the source/model diagnostics and the attendance-only state comparison.
This is a local population candidate, not a calibrated or published replacement
population, and the reports retain `production_ready: false`.

- [Source and production-stage qualification](qualified-preparation.json)
- [All-state population comparison](qualified-population-comparison.json)
- [Reproduction and field mapping](../../docs/us-childcare-attendance.md)

No individual survey records, identifiers, raw archives, or H5 populations are
committed. Reports contain aggregate diagnostics and exact artifact/code hashes.
The earlier [source-only report](nsece-2024-v1-validation.json) is historical
(f065a3ae), predating the corrected gap-code classification and population model.
It does not describe the final candidate.

## Source coverage and model

| Source status | Children |
| --- | ---: |
| Complete classified calendars | 7,460 |
| Reconstructed from observed regular weekly hours | 3,046 |
| Still excluded under age 13 | 1,105 |
| Outside source age domain | 134 |
| Total | 11,745 |

The 10,506 usable donors include measured nonparticipants and modeled schedules
for May/fall questionnaire respondents. The latter preserve regular weekly hours
but borrow days and irregular care; they are not observed full schedules. The
remaining excluded records comprise 882 ambiguous calendars, 174 partial
calendars, and 49 unusable noncalendar summaries. Conditional matching cannot
identify their missing schedules without additional assumptions.

Matching uses age, region, parents' last-week work, and household income in 2023
dollars. The explicit fallback hierarchy always retains age. Tied nearest-hour
donors are all retained. A household shared-rank mixture models sibling
participation; it does not assert common provider identity.

The source evaluation uses five household-separated folds. Overall observed ECE
participation is 46.51%, compared with 46.55% predicted, and weekly hours are
14.24 observed versus 14.08 predicted. For youngest sibling pairs, joint
attendance is 32.86% observed versus 32.28% predicted; independent draws predict
25.99%. The masked-calendar check holds out 1,581 children by whole household: weekly
hours are 13.10 observed versus 13.16 reconstructed, and days are 1.72 versus
1.79. Subgroup discrepancies remain visible in the final report. These diagnostics informed development and
must not be described as an untouched external acceptance sample.

## Population boundary

The exact BuildP parent contains 166,321 people, 57,240 households, and 31,889
children ages 0–12. The stage restores temporary income predictors for all three
ASEC cohorts from pinned Census sources while preserving the parent's original
columns, raw missingness, entity links, weights, and period. Its native export
adds only the three attendance inputs and a provenance receipt.

Outside ages 0–12, the explicit export policy inherits existing engine baseline
values where observations are absent. The 134,432 out-of-domain people include
557 disabled teenagers ages 13–17. Their attendance has not been estimated by
this source. Preserving baseline behavior does not establish nonattendance.

The state experiment applies 2026 policies to fixed source ages and incomes,
without aging or uprating, and calls each direct state child-care subsidy
variable. Only attendance changes. Provider, activity, expense, enrollment, and
take-up inputs remain as in the parent. Benefit amounts are potential modeled
benefits, not national CCDF spending or caseload estimates. Source-selection,
true summer schedules, provider-specific pricing, and older-child coverage
remain limitations for population publication.


## Final state results

All 51 jurisdictions were evaluated. All-zero state results fell from **31 to 3**:
California, Maryland, and Nevada. Positive modeled subsidies became available
in 28 additional jurisdictions. The aggregate annual potential benefit changes
from $2.253 billion to $5.286 billion under this fixed-population experiment;
these amounts are not calibrated spending estimates.

All 31,889 under-13 children have resolved inputs. Weighted attendance is 48.47%,
with 1.959 days and 14.081 hours per week averaged across all children, including
nonparticipants. Exact four-field support covers 31,152 children; 576 use
age/parent-work/income and 161 use age/parent-work. None needs the age-only level.
The report gives target distributions by age, region, work, and income.

The [remaining-input diagnostic](qualified-remaining-state-inputs.json) identifies
separate blockers in January 2026:

| State | Evidence on this candidate |
| --- | --- |
| CA | 341 SPM units meet CAPP eligibility, but state-specific days/month and weeks/month remain zero, making the time coefficient zero. |
| MD | 115 SPM units meet CCS eligibility, but every provider type is `NONE`, giving a zero reimbursement rate. |
| NV | 149 SPM units meet the income test, but every activity test is false. |

These require additional source/engine mapping work. The PR does not infer
licensed provider status or approved CCDF activity from attendance alone, and
does not claim to close every part of #915.

### Local checks

916 distinct targeted tests passed across the attendance/source, architecture,
fiscal-builder, release-coverage, and US bundle suites. Lint, tracked CI test
inventory, generated bundle validation, and diff whitespace checks passed.
The real production stage completed on the pinned parent, and native export
reloaded successfully with original data, weights, and period preserved.
Full GitHub CI runs separately on the submitted commit.
