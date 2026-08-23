# Universal Credit caseload diagnosis — UK data issue #452

Date: 2026-08-22

## Verdict

The issue's roughly 40% national undercount does not reproduce in the licensed
exact-lock aggregate diagnostic. On the Great Britain benefit-unit definition
used by the DWP facts, the pinned enhanced-FRS v1.56.14 artifact produces 6.180
million benefit units with positive Universal Credit in static 2025, against
the 6.759 million administrative target compiled by microcosm#735. The signed
gap is -0.579 million, or **-8.6%**. The exact-lock v1.56.16 artifact sensitivity
is -8.3%; a current-upstream 2.91.0 cross-check reproduces both UC tables to the
displayed precision.

The smaller aggregate gap does not resolve the defect. The element diagnostic
gaps run in both directions: housing is +12.8%, LCWRA -13.6%, carer -50.5%, and
childcare +146.8%. Family-type gaps range from -3.5% to -22.5%. Element counts
overlap and must not be added.

The record-path audit identifies four overlapping mechanisms:

1. the frozen take-up assignment gates final awards;
2. the enhanced-artifact reported signal is below the administrative caseload
   and is used as a take-up anchor rather than an award;
3. the eligibility proxy appears to count private pension rights as UC capital;
4. a small, highly weighted positive-award support carries much of the total.

These are not additive causal shares. The evidence supports a weighted
record-path accounting and specific hypotheses. Causal attribution still needs
a predeclared licensed counterfactual.

## Evidence status and comparison contract

Terms are deliberate:

- **Measured**: an aggregate computed from a locally cached licensed artifact.
  No row-level licensed data were written to this repository.
- **Code-confirmed mechanism**: a gate, formula, or data transformation read in
  the pinned source.
- **Diagnostic association**: a measured overlap, not a causal estimate.
- **Counterfactual required**: the named mechanism must be changed while other
  inputs are held fixed before a causal increment can be assigned.

Primary diagnostic baseline:

| Item | Value |
|---|---|
| Artifact | `enhanced_frs_2024_25.h5`, upstream UK data v1.56.14 |
| SHA-256 | `97a07f9ccb54019e4550e70980c561c985523e6bbc43d21938d01536e37d6c3e` |
| Records | 61,223 benefit units; 52,846 households; 113,617 people |
| Simulation | Exact lock: PolicyEngine UK 2.89.0, Core 3.26.11, static 2025 |
| Geography | Great Britain; Northern Ireland and `UNKNOWN` excluded (`UNKNOWN` positive-UC mass was zero) |
| Caseload predicate | final `universal_credit > 0`, benefit-unit grain |
| Element predicate | final UC positive and named element positive |
| Weights | direct `benunit_weight`: maximum member `person_weight`, itself the household weight |

The committed parity reference pins the v1.56.14 artifact identity and record
counts and records `would_claim_uc` as a persisted benefit-unit input
(`packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:67-85,219-231,370-392`).
Microcosm locks PolicyEngine UK 2.89.0 and Core 3.26.11
(`uv.lock:1366-1367,1393-1398`). The aggregate-only rerun used those exact
packages. Benefit-unit weights follow household → person → maximum member
projection
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/demographic/person_weight.py:5-12`;
`policyengine-uk@2.89.0:policyengine_uk/variables/household/demographic/benunit_weight.py:4-11`).
This remains diagnosis rather than certification: no standard acceptance or
post-#735 calibration receipt was produced.

The current upstream model has a potentially relevant formula difference. The
locked 2.89.0 final formula subtracts the benefit-cap reduction only, whereas
2.91.0 also subtracts `uc_deductions`; either version floors the result at zero
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-36`).
That drift can change the positive-award predicate in principle. On these two
artifacts, however, separate 2.89.0 and 2.91.0 aggregate runs produced identical
UC positive-award benefit-unit, element, family, and child counts to the
displayed precision.

The sensitivity artifact is v1.56.16, SHA-256
`e433e532b17bd8ce76030156285816e33d44e93edabd2204adbef71d19a68712`.
That release contains a benefit-unit sort fix: from FRS 2024-25, raw
benefit-unit order differed from the engine's sorted entity order, so persisted
benefit-unit variables could attach to the wrong unit
(`policyengine-uk-data@12a1e028:policyengine_uk_data/datasets/frs.py:549-554`).
It is a sensitivity, not a replacement for Microcosm's accepted parity pin.

All administrative comparisons are Great Britain, monthly, and benefit-unit
grain. Chronicle distinguishes the DWP use of “household” from a physical
household
(`chronicle@7754c8c:packages/dwp/uc_households_family_type_april_december_2025/source_package.yaml:1-7,27-35`;
`chronicle@7754c8c:packages/dwp/uc_childcare_element_march_2021_august_2025/source_package.yaml:1-4,27-35`).
Housing, LCWRA, carer, family type, and child count use April–December 2025
monthly averages. Childcare uses January–August 2025, the available committed
2025 history. Chronicle's tests enforce benefit-unit grain, GB geography,
monthly observations, and full April–December coverage for the family and child
histories (`chronicle@7754c8c:tests/test_chronicle_source_package.py:339-438`).
The eight childcare observations are 188,000, 191,000, 191,000, 190,000,
193,000, 188,000, 171,000, and 160,000, whose mean is 184,000; the committed
package defines the series and its test pins its entity, geography, coverage,
and endpoint values
(`chronicle@7754c8c:packages/dwp/uc_childcare_element_march_2021_august_2025/source_package.yaml:39-65`;
`chronicle@7754c8c:tests/test_chronicle_source_package.py:339-360`).

The overall 6,758,888.9 target is the mean of the nine available 2025 monthly
facts, April through December, not a 12-month mean. The committed values and
coverage are pinned at
`chronicle@7754c8c:tests/test_chronicle_source_package.py:285-320`. The source
derives monthly counts from publisher-rounded deductions counts and shares
(`chronicle@7754c8c:packages/dwp/uc_deductions_march_2025_february_2026/source_package.yaml:147-170`).
Microcosm averages the matching months that exist in the year and does not
require 12 (`packages/microcosm-build/src/microcosm/build/ledger_targets.py:530-537,951-970`).

## Measured model-to-administration gaps

### Overall and elements

| Series | Admin average, m | v1.56.14 exact-lock model, m | Signed gap, m | Signed gap | v1.56.16 exact-lock model, m | v1.56.16 gap |
|---|---:|---:|---:|---:|---:|---:|
| UC benefit units | 6.758889 | 6.1795 | -0.579389 | -8.6% | 6.1990 | -8.3% |
| Housing entitlement | 4.303623 | 4.8555 | +0.551877 | +12.8% | 4.8684 | +13.1% |
| LCWRA entitlement | 2.416590 | 2.0886 | -0.327990 | -13.6% | 2.1017 | -13.0% |
| Carer entitlement | 1.134377 | 0.5616 | -0.572777 | -50.5% | 0.6414 | -43.5% |
| Childcare element | 0.184000 | 0.4542 | +0.270200 | +146.8% | 0.4712 | +156.1% |

The administrative histories are committed publisher observations: housing at
`chronicle@7754c8c:db/data/dwp/uc_households_housing_entitlement_april_december_2025/uc_households_housing_entitlement_april_december_2025.json:80-90`,
LCWRA at
`chronicle@7754c8c:db/data/dwp/uc_households_lcwra_entitlement_april_december_2025/uc_households_lcwra_entitlement_april_december_2025.json:80-91`,
carer at
`chronicle@7754c8c:db/data/dwp/uc_households_carer_entitlement_april_december_2025/uc_households_carer_entitlement_april_december_2025.json:80-90`,
and childcare coverage at
`chronicle@7754c8c:tests/test_chronicle_source_package.py:339-360`.

The v1.56.14 UK-wide modeled count is 6.3275 million. Northern Ireland
contributes 0.1480 million and `UNKNOWN` contributes zero, leaving the aligned
GB value of 6.1795 million. `benunit_region` takes the first member's household
region and its enum includes both `UNKNOWN` and Northern Ireland
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/demographic/benunit_region.py:5-16`;
`policyengine-uk@2.89.0:policyengine_uk/variables/household/demographic/geography.py:6-19`).
Comparing UK directly with the GB fact would give -6.4%, but that is the wrong
geography. The FRS spine includes Northern Ireland as region 13
(`packages/microcosm-build/src/microcosm/build/uk_runtime/frs_spine.py:60-81`).

### Family type

| Family type | Admin average, m | v1.56.14 model, m | Signed gap | v1.56.16 model, m | v1.56.16 gap |
|---|---:|---:|---:|---:|---:|
| Single, no children | 3.446962 | 2.9939 | -13.1% | 2.9688 | -13.9% |
| Single, with children | 2.226220 | 2.1473 | -3.5% | 2.1347 | -4.1% |
| Couple, no children | 0.284346 | 0.2205 | -22.5% | 0.2250 | -20.9% |
| Couple, with children | 0.899535 | 0.8178 | -9.1% | 0.8705 | -3.2% |

The four administrative categories average 6.8571 million, 1.5% above the
deductions-derived overall target. This is an independently queried source-basis
difference, not a residual to force away
(`chronicle@7754c8c:packages/dwp/uc_households_family_type_april_december_2025/source_package.yaml:1-4`;
`chronicle@7754c8c:db/data/dwp/uc_households_family_type_april_december_2025/uc_households_by_family_type_april_december_2025.json:80-106`).

This diagnosis uses the model's benefit-unit `family_type`: exactly two adults
defines “couple,” any model `is_child` defines “with children,” and the four enum
values map directly to the four displayed rows
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/demographic/family_type.py:4-29`).
That is an operational mapping, not proof that every model `is_child` matches
DWP claim-record semantics. The current legacy calibration contract instead
maps the facts to physical-household indicators
using “any UC benefit unit” plus household predicates
(`packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:3098-3142,3224-3317`).
The local-target census records the grain mismatch: two UC units in one physical
household can contribute differently to the UC count and household child band
(`packages/microcosm-build/src/microcosm/build/uk_runtime/local_target_census.py:474-492`).

### Number of children

| Children | Admin average, m | v1.56.14 model, m | Signed gap | v1.56.16 model, m | v1.56.16 gap |
|---|---:|---:|---:|---:|---:|
| 0 | 3.731308 | 3.2144 | -13.9% | 3.1938 | -14.4% |
| 1 | 1.278537 | 1.2196 | -4.6% | 1.2337 | -3.5% |
| 2 | 1.100618 | 1.0241 | -7.0% | 1.0371 | -5.8% |
| 3 | 0.494553 | 0.5099 | +3.1% | 0.4995 | +1.0% |
| 4 | 0.174669 | 0.1330 | -23.9% | 0.1560 | -10.7% |
| 5+ | 0.077377 | 0.0785 | +1.5% | 0.0790 | +2.1% |

The monthly child-count facts are at
`chronicle@7754c8c:db/data/dwp/uc_households_children_april_december_2025/uc_households_by_number_of_children_april_december_2025.json:80-118`.
The model rows sum `is_child` inside each benefit unit and cap the displayed
band at five, so the last row is 5+
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/demographic/benunit/benunit_count_children.py:4-11`).
As with family type, this is an aligned grain but an operational concept mapping.
The current contract uses physical-household child count
(`packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:2798-2840`),
so these facts cannot simply replace the legacy references without a grain
adjudication.

## Weighted record-path accounting

This arithmetic is aligned to GB for the exact-lock v1.56.14 diagnostic. A
“reporter” means the sum of person-level `universal_credit_reported` within the
benefit unit is positive. It describes model paths; it is not an additive causal
decomposition of the -0.579 million residual.

| Bridge step | Weighted benefit units, m |
|---|---:|
| Positive reported signal in enhanced artifact, GB | 5.5360 |
| Plus non-reporters with positive modeled UC | +2.3516 |
| Minus reporters with zero modeled UC | -1.7080 |
| Modeled UC, GB | 6.1795 |
| Administrative UC, GB | 6.7589 |
| Model minus administration | -0.5794 |

The 1.7080 million weighted reporter drop partitions arithmetically as follows:

| Reporter path to zero | Records | Weighted, m | Interpretation |
|---|---:|---:|---|
| `would_claim_uc` false | 2,550 | 0.8151 | Diagnostic association; v1.56.14 ordering contamination |
| Claim flag true, eligibility false | 1,497 | 0.7786 | Diagnostic association; 1,491 fail capital |
| Claim flag true, eligible, final award zero (reduction source not decomposed) | 286 | 0.1143 | Diagnostic association |
| Total reporter drop | 4,333 | 1.7080 | Arithmetic identity |

Displayed bridge components are independently rounded; the identity uses the
unrounded aggregate values.

The enhanced-artifact reported-input mass is 1.223 million below the GB
administrative count. It includes both non-SPI-synthetic artifact rows and
SPI-synthetic support, so it does not identify an FRS-underreporting magnitude.
Enhancement, weighting, universe differences, take-up alignment, and eligibility
all overlap. This is a measured artifact classification; the build contract
flags the cloned SPI support channel and explicitly refills
`universal_credit_reported` on those rows
(`packages/microcosm-build/src/microcosm/build/uk_runtime/spi_support.py:134-155,210-217,340-348`).

## Attribution by mechanism

### 1. Take-up modeling

**Code-confirmed mechanism.** The take-up contract fixes UC at 0.55 and marks it
frozen for incumbent parity under adjudication U8
(`packages/microcosm-build/src/microcosm/build/uk/take_up_contract.json:63-78`).
The assignment targets `int(rate * n_units)` over the unweighted benefit-unit
population, forces aligned positive reporters into the anchor set, and fills
the remainder from non-anchors
(`packages/microcosm-build/src/microcosm/build/stochastic_assignment.py:57-84`;
`packages/microcosm-build/src/microcosm/build/uk_runtime/frs_take_up.py:105-151,178-221`).
Therefore 55% is not an estimated take-up rate among eligible units.

The model treats `would_claim_uc` as an input and gates both pre-cap and final UC
through it
(`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/would_claim_uc.py:4-16`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit_pre_benefit_cap.py:4-15`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`).
Reported UC is a separate input, not an award formula
(`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit_reported.py:4-11`).

**Measured association.** v1.56.14 has 33,709 of 61,223 UK units flagged,
55.06%. In the aligned GB reporter bridge, 2,550 records / 0.815 million
weighted units have the flag false. The parity reference independently records
a 0.550594 UK-wide nonzero share
(`packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:370-378`).

**Caveat.** Only 55.3% of GB reporter records are flagged in v1.56.14. Because
the upstream sort fix explicitly concerns persisted benefit-unit alignment
(`policyengine-uk-data@12a1e028:policyengine_uk_data/datasets/frs.py:549-554`),
0.815 million is not a defensible causal “take-up contribution.”

**Counterfactual required.** On a sorted, accepted artifact, hold other inputs
fixed and set `would_claim_uc = True`. Report total, element, family, and child
caseload changes. Re-adjudicate both the 0.55 value and its denominator; do not
tune the rate to hit the administrative total.

### 2. Enhanced-artifact reported-signal shortfall

**Code-confirmed mechanism.** Positive reported amounts are aggregated to form
take-up anchors (`packages/microcosm-build/src/microcosm/build/uk_runtime/frs_take_up.py:105-122`).
They are not substituted for a modeled award: `universal_credit_reported` is an
input while final UC is formula-owned
(`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit_reported.py:4-11`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`).

**Measured association.** In GB, the artifact has 5,708 reporter records
representing 5.536 million units. Of those, 1,375 / 3.828 million retain a
positive modeled award and 4,333 / 1.708 million do not. The model adds 1,736
non-reporter records / 2.352 million units. These reporter records include
enhancement support; they are not all raw FRS observations. The artifact
classification follows the flagged SPI support contract
(`packages/microcosm-build/src/microcosm/build/uk_runtime/spi_support.py:134-155,210-217,340-348`).

**Conclusion boundary.** The enhanced-artifact reported signal is lower than
the administrative caseload, and modeled imputation is material. No bridge row
is a pure survey-underreporting share. A causal measurement-error estimate
would require linked administrative validation or an independently adjudicated
reporting model.

### 3. Eligibility mechanics

**Code-confirmed mechanism.** A unit is UC-eligible only if it has a working-age
adult and assessed capital does not exceed the parameterized limit
(`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/is_uc_eligible.py:4-15`).
When claimant capital is absent, `uc_assessable_capital` allocates residual
household capital across adults; claimant-level input can override the proxy
(`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/uc_assessable_capital.py:16-43`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/uc_reported_capital.py:4-16`).
The capital-source declaration calls the allocation an approximation and
includes `corporate_wealth`
(`policyengine-uk@c93e1a05:policyengine_uk/parameters/gov/dwp/universal_credit/means_test/capital/sources.yaml:1-20,34-39`).

The material candidate is that the `corporate_wealth` input includes private
pensions
(`policyengine-uk@c93e1a05:policyengine_uk/variables/input/corporate_wealth.py:4-12`).
Microcosm constructs it from non-DB pension wealth plus shares, options, trusts,
and ISAs
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:255-264`),
and the upstream data implementation likewise includes non-DB pensions
(`policyengine-uk-data@12a1e028:policyengine_uk_data/datasets/imputations/wealth.py:145-155`).
Universal Credit Regulations 2013 Schedule 10 paragraph 10 disregards pension
scheme rights ([official legislation](https://www.legislation.gov.uk/uksi/2013/376/pdfs/uksi_20130376_en.pdf)).

**Measured association.** `uc_reported_capital` is populated for zero artifact
records, so the proxy always supplies assessed capital. In the aligned GB
reporter-drop bridge, 1,491 of 1,497 claim-flagged ineligible records fail the
capital test; the whole path represents about 0.779 million weighted units.
This measures assessable-capital overlap directly. It does not assign that
overlap to the pension component of `corporate_wealth`.

**Conclusion boundary.** This is the strongest eligibility candidate, not its
causal effect. Split pension rights from countable investments; do not relax the
statutory capital limit to fit the aggregate.

**Counterfactual required.** Remove pension rights from countable corporate
capital while holding shares, ISAs, trusts, property, the legal limit, weights,
and random draws fixed. Run alone and crossed with the take-up counterfactual.

### 4. Weighting

**Code-confirmed mechanism.** National calibration materializes activated
targets and changes household weights, with a default maximum weight ratio of
10 (`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:21-46,50-94`).
The solver minimizes a joint capped relative-error objective and returns the
frame with calibrated weights
(`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:473-518,1331-1363,1738-1787,1840-1865`).
It does not change take-up flags, capital, eligibility, family composition, or
element formulas.

**Measured association.** Only 3,615 of 61,223 benefit-unit records have
positive modeled UC in v1.56.14. Their mean weight is 1,750 versus 580 overall.
Reported-signal benefit units with positive awards average 2,478 and positive-
award units without that signal average 1,187. In the UK-wide diagnostic, the
361 highest-weight UC-positive records—approximately the top 10% of 3,615—carry
65.7% of modeled caseload.

**Conclusion boundary.** Weighting has large leverage but no separately
identified additive share: every weighted bridge row already includes it.

**Counterfactual required.** Re-aggregate identical recipient support under
source/design, current enhanced-artifact, and post-#735 weights. Report all
elements, family type, child count, effective sample size, maximum weight ratio,
and top-decile mass without changing the weight ceiling or objective.

## Element-specific mechanism findings

The aggregate residual is not a uniform scale error.

| Element | Code-defined support | Diagnosis |
|---|---|---|
| Housing | Social rent enters at rent; private rent is LHA-capped; non-dependent deductions are removed (`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/housing_costs_element/uc_housing_costs_element.py:11-29`). | +12.8%. Decompose tenure, rent source/imputation, LHA binding, and weights before changing mechanics. |
| Childcare | Pays the lower of cap and coverage-rate × reported expense when the childcare work condition holds (`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/childcare_element/uc_childcare_element.py:4-17`; `policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/childcare_element/uc_childcare_work_condition.py:11-18`). | +146.8%. Audit expense prevalence, work-condition support, final-award overlap, and weights; do not scale the output. |
| LCWRA | The element uses `uc_limited_capability_for_WRA`, which returns `is_disabled_for_benefits` (`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/disability_element/limited_work_ability/uc_LCWRA_element.py:4-15`; `policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/disability_element/limited_work_ability/uc_limited_capability_for_WRA.py:4-12`). Dataset mode derives the disability flag from reported disability-benefit amounts (`policyengine-uk-data@12a1e028:policyengine_uk_data/datasets/disability_benefits.py:36-48,168-187`). | -13.6%. Receipt of listed disability benefits is not a direct LCWRA decision; validate the proxy. |
| Carer | The element requires `benunit_has_carer`, driven by a positive count of people receiving Carer's Allowance or Scottish Carer Support Payment (`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/carer_element/uc_carer_element.py:4-14`; `policyengine-uk@c93e1a05:policyengine_uk/variables/household/demographic/benunit_has_carer.py:4-11`; `policyengine-uk@c93e1a05:policyengine_uk/variables/household/demographic/num_carers.py:4-11`; `policyengine-uk@c93e1a05:policyengine_uk/variables/household/demographic/is_carer_for_benefits.py:4-11`; `policyengine-uk@c93e1a05:policyengine_uk/variables/input/care.py:28-41`). | -50.5%. This receipt proxy is a strong candidate because UC carer entitlement is not identical to receipt of those benefits. |

The elements feed the maximum-award formula
(`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/uc_maximum_amount.py:11-20`),
but income reduction and the benefit cap can reduce an exact-lock case to zero;
current upstream also subtracts deductions
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit_pre_benefit_cap.py:4-15`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:12-36`).
The table identifies support paths and hypotheses, not isolated contributions.

## Exactly what microcosm#735 fixes

#735 makes one precise correction: it activates the administrative overall GB
count as a 2025 national calibration objective.

1. The legacy identifier `dwp.uc.households` now selects GB
   `dwp.uc_benefit_units` facts, uses benefit-unit grain, and applies
   `calendar_year_average`
   (`packages/microcosm-build/src/microcosm/build/uk/target_references.json:9361-9378`).
2. The model measurement is a benefit-unit count filtered to final
   `universal_credit > 0`; the contract notes that DWP “household” means the UC
   unit of assessment
   (`packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:9980-10019`).
3. The membership receipt resolves 11 matching facts overall, nine eligible in
   2025, to 6,758,888.8889
   (`packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:7595-7609`).
4. It replaces the incumbent 6,700,000 with a target 58,888.9 higher, a signed
   change of +0.879%
   (`packages/microcosm-build/src/microcosm/build/uk/ledger_compile_parity_incumbent_2025_signed_differences.json:2293-2299`).
5. When national calibration runs, the row enters the joint household-weight
   solve
   (`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:50-94`).

## Exactly what microcosm#735 cannot fix

- **It does not itself run calibration or build an artifact.** The committed
  staging record has no post-#735 national-calibration evidence; the build adds
  that artifact entry only when calibration evidence exists
  (`packages/microcosm-build/src/microcosm/build/uk/national_staging_build_record.json:1-43,98-116`;
  `packages/microcosm-build/src/microcosm/build/uk_runtime/national_build.py:549-558`).
- **It cannot guarantee exact equality.** The UC row competes with all active
  rows in a joint capped relative-error objective
  (`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:473-518,1331-1363`),
  and the UC calibration test requires movement toward the fact, not equality
  (`packages/microcosm-build/tests/test_uk_national_calibration.py:185-200`).
- **It cannot change recipient support.** Calibration changes weights only
  (`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:1738-1787,1840-1865`),
  so it cannot flip take-up, repair capital treatment, create childcare
  expenses, or change carer/LCWRA support.
- **It does not activate #188's four element histories.** The committed target
  surface contains no selectors for the four new element concepts, while the
  Chronicle packages are present and tested
  (`chronicle@7754c8c:tests/test_chronicle_bundle.py:163-202`).
- **It does not move legacy family/child bindings to the #188 histories.** The
  representative references remain old-concept, physical-household, May-2025
  rows
  (`packages/microcosm-build/src/microcosm/build/uk/target_references.json:4050-4070,4165-4185`).
- **It does not reconcile universes.** Family facts average 1.5% above the
  deductions total, the target is GB while the FRS includes NI, and four
  source-only “No payment” facts remain outside active fanout
  (`packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:7619-7623`).
- **It cannot validate reform caseload.** Baseline weight alignment does not
  repair the take-up and eligibility formulas that decide which units respond
  to reform; those code paths remain as cited above.

## Smallest defensible fix path

1. **Use microcosm#736 to sign the measurement contract before adding targets.**
   Keep GB geography unless NI facts are supplied; use benefit-unit grain;
   adjudicate deductions total versus independently queried Stat-Xplore cubes;
   decide nil-payment treatment; and replace or sign out physical-household
   family/child bindings. Where semantics cannot be aligned, use an owner-signed
   exclusion or reconciliation entry.
2. **Repair support mechanics upstream.** Split pension rights from countable
   corporate investments without changing the legal capital ceiling;
   re-adjudicate UC take-up value and denominator; validate or replace carer and
   LCWRA receipt proxies. Do not add element multipliers.
3. **Resolve artifact alignment.** The parity pin predates the benefit-unit sort
   fix. Re-pin only through the licensed acceptance path and repeat the
   diagnosis; v1.56.16 is sensitivity evidence, not a receipt substitute.
4. **Run a predeclared licensed matrix.** `B`: sorted baseline; `T`: force
   take-up true; `P`: remove pension rights from countable capital; `TP`: both.
   Aggregate each under design, enhanced, and post-#735 weights. Report total,
   four elements, family type, child count, reporter bridge, effective sample
   size, maximum weight ratio, and concentration. Report main effects and
   interaction rather than forcing overlap into additive shares.
5. **Only then assess calibration.** Require per-target diagnostics and test
   whether element or family fit worsens while the total improves. Keep all seeds,
   ceilings, folds, bands, and gates unchanged.

## Claims requiring licensed confirmation

- causal increments from take-up, pension-capital treatment, and interaction;
- design-weight versus enhanced-weight totals;
- post-#735 calibrated total, element, family, and child-count outcomes;
- reform-caseload sensitivity after support mechanics are repaired.

The exact-lock aggregate and aligned GB reporter accounting are complete, but
the remaining counterfactuals are not. The defensible issue conclusion is
narrow: the stale 40% aggregate gap is not reproduced; the pinned exact-lock
diagnostic measures -8.6%, while UC composition and recipient support remain
materially wrong. #735 supplies a calibration fact, not a mechanism fix.
