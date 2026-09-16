# NSECE attendance pooled-model investigation — 2026-09-16

**PR #916 remains draft.** A partially pooled donor model improves conditional
child predictions and larger-family means. An exploratory population-moment
dependence fit passes 13 of the original 15 household screens, but still fails
two hours checks and all three checks for observed children with unresolved
siblings. Neither experimental model has replaced the production-stage recipe.

**Runtime update after merging main:** the branch now uses PolicyEngine-US
2.2.1 and Core 3.32.5. Population artifacts, native-loader receipts and benefit
estimates below were produced with PolicyEngine-US 1.819.0 and Core 3.31.0.
They remain historical evidence, not validation of a population under the new
runtime. Runtime-bound attendance receipts require rebuilding from the original
parent before a new candidate can be used; the merge does not bypass that check.
The complete three-model survey comparison was rerun with 2.2.1: every model
result and diagnostic code hash matches the pre-upgrade run exactly. The linked
pooled report records the new runtime; this survey replay is separate from
population and benefit validation.

- [Initial pooling and evaluation plan](pooled-matching-plan.txt)
- [Subsequent exploratory dependence-objective plan](pooled-moments-plan.txt)
- [All three models, full diagnostics and structural check](pooled-matching-validation.json)

## What was tested

The new donor model blends each sparse cell with broader empirical distributions
at the same exact age. It adds household size, work, income, and region in that
order. A cell's contribution is `effective_households / (effective_households + 10)`;
the strength ten was fixed before inspecting results. Survey-weight concentration
is measured at the donor-household level. Joint day/hour schedules, including
measured nonattendance, remain intact.

The sibling fit uses every available observed pair, including families with
unresolved other siblings. Household weight is divided among the household's
observed pairs. Every fitting pair's entire household is excluded from its donor
distributions. The initial fit minimizes individual pair cross-product errors
for participation, days and hours. It improved mean errors but worsened several
correlations. A separately recorded exploratory alternative fits the three
weighted population cross-product moments instead. It retains one shared-rank
coefficient and the same fixed donor distributions; no per-outcome coefficients
or revised thresholds are selected from the evaluation results.

Every component is refitted inside five household-separated folds. All 7,460
measured child calendars are scored, alongside 4,450 observed pairs from 2,106
households and the original complete-household diagnostics. Previously inspected
households, including the earlier reserved partition, are explicitly treated as
development data. This is not independent validation or evidence of release readiness.

| Quantity | Existing model | Pooled, individual-pair fit | Pooled, population-moment fit |
| --- | ---: | ---: | ---: |
| Failed original household screens / 15 | 9 | 4 | 2 |
| Failed observed-child screens / 18 | 3 | 3 | 3 |
| 3+ household mean-days error | +25.35% | +12.01% | +12.01% |
| 3+ household mean-hours error | +23.36% | +15.91% | +15.91% |
| Conditional mean days prediction MSE | 5.761 | 4.634 | 4.634 |
| Conditional mean weekly-hours prediction MSE | 575.310 | 482.798 | 482.798 |

Conditional-mean prediction errors improve approximately 20% for days and 16%
for hours. This is not a claim that random donor draws are more accurate: the
expected squared error of a single hours draw rises from 891.140 to 905.994.
The report contains both quantities. Marginals are identical across the pooled
fit objectives because the dependence coefficient changes only joint behavior.

## What remains unresolved

For the youngest pair in 3+ complete households, the population-moment model's
weekly-hours correlation is 0.456 versus 0.562 observed (gap 0.106; limit 0.10).
Its hours cross-product is 437.072 versus 337.347 (+29.6%; limit 20%). A change
to sibling dependence alone cannot satisfy both with these predicted marginals:

| Existing screen | Required hours cross-product interval |
| --- | ---: |
| Correlation within 0.10 of observed | 440.651–553.157 |
| Cross-product within 20% of observed | 269.878–404.816 |

The intervals do not overlap. Correlation equals `(E[XY] - E[X]E[Y]) / (SD[X]SD[Y])`.
Independent and coupled arms have the same marginal means and variances, so the
report can recover this relationship and test compatibility without choosing
another coefficient. This rules out a dependence-only fix **for these evaluated
marginals**; it does not rule out better marginal models or establish that the
selected complete families represent the whole population.

The 761 observed children whose siblings have unresolved calendars remain a
material selection concern. Observed versus pooled predictions are 66.0% versus
46.7% participation, 2.819 versus 1.915 days/week, and 23.830 versus 14.135 hours/week.
Hours are underpredicted by 40.7%. Both pooled objectives share these failures;
better complete-family results do not waive them. Missing children are never
scored as zero, and these observed-subgroup differences do not identify their
unobserved schedules or a causal missingness effect.

**Decision:** keep pooling experimental. The next substantive change must improve
the conditional marginal model and address calendar nonresponse using measured
information or explicit, tested assumptions. Further rho tuning, resplitting this
same survey, or lowering predictions to complete-household means would not close
the evidence gap. A population trial would also require consistent source bridge
and target integration, followed by new benefit/transport sensitivity evidence.

Reproduce the complete comparison with new local output paths:

```bash
uv run python tools/validate_us_childcare_pooling.py \
  --household-tsv /local/39466-0005-Data.tsv \
  --calendar-tsv /local/39466-0004-Data.tsv \
  --compare-moment-fit --report /local/pooled-comparison.json
```

Synthetic tests cover exact-age support, household-cluster pooling, survey-weight
and row-order invariance, whole-household exclusions at both fit and evaluation
boundaries, inclusion of partially observed families, joint schedule preservation,
structural screen compatibility, and JSON-safe undefined metrics. Tests reside
directly in the build shard's tracked CI inventory; CI does not access survey data.
Attendance recipe code is unchanged by this experiment. Before the main merge,
its full recipe identity matched the last verified population artifact. The
engine upgrade changes that runtime-bound identity; no population rebuild or
new state benefit estimate is claimed here.

The 616-test attendance/source/architecture regression run passed. After making
undefined relative-error flags JSON-safe, all 96 focused source/pooling tests
passed again. Lint, formatting, tracked CI inventory and build-wheel source-byte
checks passed. The final real-source run reproduces both initial model arms
exactly and records the current diagnostic code hashes; only aggregate evidence
is committed. GitHub CI is separate and has not been monitored.

After the main merge, 1,142 attendance/source, architecture, fiscal-builder,
coverage, serializer and native-adapter tests passed under the updated lock.
A separate 249-test pool-tool/specification run also passed. The merged coverage
report and spec digest were regenerated; lint, formatting, tracked CI inventory
and current-wheel source/engine-lock byte checks passed. The complete survey
replay above reproduces every model result exactly under PolicyEngine-US 2.2.1.

## Previous hard household-size investigation — 2026-09-16

**PR #916 remains draft.** Adding household size improves larger-family means,
but the proposed matcher still fails joint-schedule validation. It is retained
as an experiment and **has not replaced the production-stage matching recipe**.

- [Comparison plan recorded before the new results](household-size-plan.txt)
- [Development comparison and calendar-selection diagnostics](household-size-development.json)
- [Reserved-household comparison](household-size-reserved-validation.json)
- [Pre-upgrade artifact verification](household-review-artifact-verification.json)

The challenger adds the number of rostered children ages 0–12, capped at three,
to the existing age/region/work/income matching. Missing calendars still count
toward household size. Its fallback retains size until the final age-only level.
Both donor selection and the sibling-dependence fit use the revised predictors.
The shared-rank mixture cannot change each child's conditional mean: adjusting
its dependence coefficient alone cannot correct a mean attendance discrepancy.

Before fitting the challenger, a deterministic 20% household partition was
reserved. Every development donor pool and dependence fit excludes it. Five-fold
development used the remaining households; the fixed challenger was then scored
once on the reserved households. All metrics integrate weighted donor CDFs
exactly. Earlier diagnostics had already used this survey, so this is an
internal comparison, **not untouched external validation**. No model was retuned
after viewing the reserved results.

| Reserved comparison, 128 complete households with 3+ children | Observed | Existing matcher | Size-conditioned challenger |
| --- | ---: | ---: | ---: |
| Mean total days/week | 4.332 | 5.892 (+36.0%) | 4.964 (+14.6%) |
| Mean total hours/week | 32.575 | 40.793 (+25.2%) | 33.430 (+2.6%) |
| SD of total hours/week | 54.828 | 49.677 | 43.246 (−21.1%) |

The reserved comparison includes 376 complete sibling households overall.
The existing matcher fails 10 of 15 provisional screens; the challenger fails
8 of 15. Better larger-family means do not establish a realistic joint
distribution. For example, the youngest-pair weekly-hours cross-product error
in 3+ households increases from 18.8% to 36.6%. The challenger's dependence
coefficient reaches its upper bound of one in every development fit and the
reserved fit, yet several joint-participation/intensity checks still fail.
On the development folds, the larger-family hours mean is still 21.2% high.

Two limitations prevent treating the mean improvement as a resolved model:

1. **Sparse conditioning:** in the development pool, the median exact cell
   falls from 12 donor households to 5. The share of observed children in cells
   with fewer than ten donor households rises from 37.2% to 87.9%. These counts
   precede fold exclusions, which can further reduce support; ten is a
   descriptive cutoff, not a tuned matching rule.
2. **Selected validation households:** totals can be scored only when every
   under-13 calendar is complete. In development households with 3+ children,
   observed children in fully complete households average 9.86 hours/week;
   observed children with unresolved siblings average 26.22. All observed
   children in that size group average 12.62. These child-weighted means show
   selection differences, not the missing children's outcomes or a causal
   missingness effect. Forcing predictions down to the complete-household
   average would not establish unbiased population attendance.

**Decision:** do not adopt hard household-size strata on these results. The
next model design needs to control sparse-cell instability and assess calendar
selection explicitly, scoring all observed child marginals by household size
alongside complete-household joint moments. A partial-pooling or household-level
model requires a new evaluation plan; another split of this same inspected
survey would still not be external acceptance evidence. Noncalendar transport
sensitivity and older-child/provider gaps remain unresolved.

Reproduce either partition, using a new report path for each run:

```bash
uv run python tools/validate_us_childcare_household_size.py \
  --household-tsv /local/39466-0005-Data.tsv \
  --calendar-tsv /local/39466-0004-Data.tsv \
  --partition development --report /local/household-development.json
# Repeat with --partition validation and a separate report path only after
# freezing the challenger; do not use those outcomes for further tuning.
```

The tool verifies the original source hashes and reports the plan, diagnostic
code, and recipe hashes. It writes aggregates only. Regression tests cover
reserved-household exclusion from every fit and donor pool, poisoning reserved
outcomes without changing development results, actual donor conditioning,
counting unresolved siblings, and rejecting reconstructed calendars as truth.

The 593-test source/attendance/architecture regression run passed, as did lint,
formatting, the tracked CI test inventory, and exact source-byte checks for both
updated modules in the built wheel. The current recipe was rebuilt from the
original parent because the fitter's code hash changed. Both native loaders
validate its receipt; all 166,321 people's attendance values and IDs and all
57,240 household weights equal the previous candidate exactly. The default
dependence fit and sensitivity evaluator are unchanged, so the historical
benefit and sensitivity estimates still apply. These checks do not certify
the statistical model or authorize population publication.

## Review revision — 2026-09-15

**PR #916 remains draft.** The engineering safeguards have been strengthened,
but the expanded household diagnostics fail provisional statistical screens.
The earlier aggregate means did not establish population validity. Nothing in
these reports authorizes publication or changes PolicyEngine-US defaults.

- [Criteria declared before the expanded runs](review-validation-criteria.txt)
- [Revised source/stage and household diagnostics](review-source-stage-validation.json)
- [All-state noncalendar sensitivity results](review-transport-sensitivity.json)
- [Final artifact verification](review-artifact-verification.json)
- [Reproduction commands and receipt contract](../../docs/us-childcare-attendance.md)

The criteria are developmental screens informed by earlier diagnostics, not an
untouched external evaluation or a maintainer-approved release standard.

## Response to review

- **C1:** bind attendance to source/recipe/runtime/settings and person-level
  content; restore and verify the binding in both native loaders. Reject stale,
  missing or changed receipts. An identical rerun validates existing values;
  changed source/seed/policy requires rebuilding from the original parent.
- **A1:** recheck every row's completeness and valid schedule at final fiscal
  export, independently of optional source flags or generic coverage overrides.
  Persist and verify the receipt after native serialization.
- **A2:** reject missing/blank household identities and unresolved household
  links before converting IDs to strings or assigning shared ranks.
- **A3:** evaluate the actual shared-rank schedule mixture, including days/hours
  cross-moments and correlations and all-child totals for larger households.
  The expanded evidence exposes a remaining model limitation; it does not close
  this statistical concern.
- **S1:** declare screens and measure benefit sensitivity while preserving every
  measured regular-hour value. The state-level sensitivity remains unresolved;
  missing days and irregular care are not identified by this experiment.
- **S2:** record the actual calendar, sibling fit, bridge, predictor
  harmonization, transfer and outside-domain operations in execution order.

## Expanded household results

Five household-separated folds contain 1,941 complete sibling households,
including 661 with three or more children. Predictions integrate the empirical
weighted donor CDFs exactly; no favorable simulation seed is selected.

| Quantity | Observed | Shared-rank model |
| --- | ---: | ---: |
| Youngest-pair days correlation | 0.580 | 0.442 |
| Youngest-pair weekly-hours correlation | 0.523 | 0.356 |
| Mean total days/week, households with 3+ children | 4.510 | 5.653 |
| Mean total hours/week, households with 3+ children | 33.085 | 40.814 |

Nine of fifteen provisional screens fail. Larger-household mean total days are
25.35% too high and hours 23.36% too high. A fitted binary-participation mixture
is not enough to establish realistic household schedules. Household-size
conditioning was investigated using the separately reserved internal comparison above.
The tested hard household-size conditioning was not adopted; retuning on these
folds would not create independent validation.

## Noncalendar assumption sensitivity

All 51 jurisdictions use the same parent, source, matching fields, survey
weights, seed and policy year. Each alternative modifies only modeled bridge
components and retransfers the resulting joint schedules; measured regular
hours are unchanged. Source-selection, missing partial calendars and true summer
attendance are separate uncertainties this stress test does not resolve.

| Scenario | National annual potential benefits | Change from candidate |
| --- | ---: | ---: |
| Current candidate | $5.286 billion | — |
| No modeled irregular hours | $5.209 billion | −1.46% |
| One fewer modeled day | $5.222 billion | −1.22% |
| One more modeled day | $5.298 billion | +0.22% |

State flags above 20% include TN (−36.0%, no irregular hours), IA (+60.0%),
KS (−49.4%) and MS (−69.6%) with one fewer day, and AR (+43.9%) with one
more day. OK also flags at −40.0%, but that is only a $2.06 change from an
approximately $5.15 baseline and must not be read as a material spending result.
Day changes also change daily hours and can cross state policy thresholds;
these are joint-schedule assumption tests, not monotonic attendance effects.
These estimates are potential modeled benefits, not calibrated CCDF expenditure,
caseload estimates or confidence intervals.

## Revision code checks

The 750-test regression run passed, covering the complete source-spine
architecture guard, pool-tool regressions, NSECE source/receipt behavior, and
the unconditional fiscal export guard. The preceding focused attendance and
release-coverage run passed 165 tests; native H5, serializer and fiscal-builder
checks also passed before the architecture fixes were verified in the final run.
After replacing the large receipt dictionary with a sequence, all 67 focused
source/receipt and architecture tests passed again. This avoids quadratic
traversal in immutable Frame metadata without changing attendance values.
Repository lint, format checks, the tracked CI inventory and the exact
42,159-field/41-inventory coverage audit passed. The build wheel was rebuilt;
its three new modules match the source bytes and import from the unpacked wheel.
These code checks do not certify restricted data or resolve the statistical gaps.

## Historical evidence

The September 12–13 reports below describe the previous execution and remain
available for audit. Their artifact receipts predate the content-binding
contract; rebuild them from the original parent before using the revised release
path. The attendance point estimates are unchanged by the engineering revision.

## Original population candidate — 2026-09-12

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


## CI integration correction — 2026-09-13

The initial commit's full CI exposed three omissions outside the local targeted
selection: the new native writer was absent from the serializer registry, the
new source descriptor's three manifest fields were missing from field-ledger
pins and the generated coverage report, and a pool-tool test retained the old
US spec hash. The same failures repeated across Python versions and test lanes.

The writer now uses the shared nullable-boolean table boundary and has registry
round-trip coverage for mixed and all-missing Boolean columns. The configuration
ledger retains exact counts and pointer hashes with explicit validation claims
for the new descriptor; its semantic/missing-sink checks remain enforced.

The [full-population export verification](ci-export-verification.json) reran the
corrected writer on the qualified checkpoint. All 166,321 people's entity-table
values, dtypes, weights, and the time period match the original qualified native
candidate exactly. Attendance estimates and the 31-to-3 state comparison are
unchanged; artifact bytes have their own new hash.

The correction passed 579 tests across the serializer/source, field-ledger,
coverage-report, pool-spec identity, and architecture regression runs. The
coverage generator reports 42,159/42,159 fields and 41/41 inventory checks.
Repository lint, CI test inventory, and the build wheel also passed locally.
