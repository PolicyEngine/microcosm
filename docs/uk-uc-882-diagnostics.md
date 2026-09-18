# UC target repair diagnostics, September 2026

The fresh controlled comparison shows that paid-claim target repairs improve
the broad UC fit but **do not close the lone-parent gap**. Against the same
paid target, the fresh lone-parent shortfall changes from 14.37% to 14.01%.
The [population comparison receipt](evidence/uk-uc-882/calibration-comparison.json)
records these completed development calibrations and their input hashes. These
fit results do not come from the [source-only compilation receipt](evidence/uk-uc-882/source-target-diff.json),
which checks source targets without a population solve. No certified release is claimed.
The [target contract](uk-uc-paid-target-contract.md) describes the statistical
repair and its remaining approximations. [#882](https://github.com/PolicyEngine/microcosm/issues/882)
remains open.

## Retained-spine experiment

The baseline uses merged Microcosm `a19582bd`, released PolicyEngine-UK 2.97.0
and the authenticated retained FRS/SPI spine. All arms hold the 52,846
household identities/order, priors, source ancestry, simulation outputs, active
366-row roster and family coefficients fixed. The historical 371-row/UK2.94
replay is not the causal control: five additional non-UC exclusions and the
engine upgrade were already merged before this experiment.

Each solve uses 1,500 updates, learning rate .02, seed zero, free mass,
maximum weight/prior ratio 10, no sparsity penalty, target loss cap 10 and
`family_equal`. No extra epochs or new joint targets are introduced. The loss
cap is not a 10% fit tolerance. The tool validates input, code and dependency
identities at both ends of each run and authenticates the baseline's saved
inputs and weights before comparing them.

| Fit against each arm's own source target | Old contract | Paid CY2025 | Paid FY2025/26 |
|---|---:|---:|---:|
| UC headline | −4.49% | −4.29% | −5.12% |
| Single with children | −18.12% | −12.78% | −15.31% |
| Couple with children | −0.03% | +0.09% | +0.05% |
| Five-plus reported children | +0.29% | −0.07% | −0.05% |

Changing the denominator explains part of the apparent lone-parent improvement.
Against the **same CY paid target of 2,138,780**, the old-contract weights are
14.77% below target; CY recalibration is 12.78% below. The estimate rises from
1,822,793 to 1,865,550. The repair therefore improves the weighted estimate
without establishing closure.

The CY structural-family and allowance-family arms produce identical matrices
and weights. Both use a matrix identical to the baseline on this retained
sample. The changed values are ten right-hand-side targets; the family
measurement introduces no new support here. It is an allowance-based model
proxy: under the tested UK2.97 claimant inputs it coincides with structural
family typing. It does not recover administrative ineligible-partner cases or
change claimant roles. `UNKNOWN` handles zero model allowance defensively; it
does not identify observed administrative missing-family records.

Evaluated against one common CY matrix, target vector and coefficient vector,
the ten broad count rows' weighted mean absolute percentage error falls from
9.15% to 4.35%; seven of ten are within 5%, compared with five before. The FY
weights give 5.97% on that same CY comparison. This does not select CY because
it fits better: the two windows represent different observation conventions.

CY household ESS changes from 13,846 to 13,722 and original-source household
ESS from 5,026 to 4,956. The maximum ratio remains 10. Compared with the
same-engine old-contract fit, CY income tax changes +0.12%, National Insurance
+0.15%, state pension +0.57%, Child Benefit −0.29%, employment income +0.16%
and Housing Benefit +3.09%. Housing Benefit remains a reviewed excluded
calibration row; this is an outcome change, not a validation pass.

After these retained runs, a compiler guard was tightened to reject missing
publication identities. Recompiling the real feed before and after that change
gives identical values and metadata for all 415 targets. The fresh run below
uses the tightened guard; the completed retained comparisons remain bound to
their original code receipts.

A later metadata correction expresses the five paid child-count averages as
explicit twelve-month source windows. It removes their misleading
December-to-year hold records and records the months and publication identity.
It leaves all 415 compiled values/statuses and all 366 active target identities
and their order unchanged. The other 410 references, including 101 other
monthly UC references, and all unrelated holds are unchanged. This correction
does not require another population solve or alter the comparisons above.

## Fresh combined source build and calibration

The full unsampled source build completes in 391.51 seconds with 40 graph
nodes and all 15 source-scope gates passing. It uses stock UK 2.97.0, the
pinned FRS2024/25 inputs and seed 578. The first attempt exposed two missing
read declarations on the capital stage: `person.is_benunit_head` and
`person.is_parent` existed in the full checkpoint but were removed by graph
scoping. Adding those dependencies fixes execution. The engine-free
[test](../packages/microcosm-build/tests/test_uk_uc_capital_coherence.py) runs the
production split node through executor projection and the actual capital
transform, with positive SPI reporters and claimant roles that disagree with
legal marriage. It checks the resulting donor values and redraw count.

Existing [graph-stage integration](../packages/microcosm-build/tests/test_uk_graph.py)
and [H2 parity](../packages/microcosm-graph/tests/test_acceptance_h_parity.py)
exercise all 28 real transforms using committed synthetic source tables and
the UK engine; the legacy-versus-graph parity check passes. Running every
stage does not guarantee every conditional read is reached: capital donor
selection reads claimant roles only when a reporter redraw occurs. The new
regression forces that branch. A broader inventory of conditional branches
and reusable projected-stage fixtures remains follow-up work.

The final spine contains 52,846 households, 61,213 benefit units and 113,590
persons, with complete original ancestry for all descendants. The 16,288
original households and 18,850 original benefit units survive. The capital
stage processes 893 SPI reporter redraws and refreshes 265 would-claim flags.
Source child counts and claimant couple status remain unchanged through the
inspected same-ID imputation, reporter and capital stages. Source-stage
snapshots do not store calculated UC awards/components, so they cannot
establish the first stage at which an upper payment cell became empty.

The production national solve completes at the same 1,500-update budget in
180.90 seconds including compilation and checks. Its paid lone-parent estimate
is 1,839,214 against 2,138,780, a **14.01% shortfall**. The headline is 4.71%
below target; couples with children and five-plus reported children are within
0.13%. This confirms that the repaired targets do not close the gap on the
fresh combined pipeline either.

An authenticated private replay reproduces every initial and final estimate
for all 366 canonical rows exactly, along with the selected iterate, loss,
ESS and maximum weight ratio. It saves diagnostic matrices/weights only and
preserves the original terminal refusal. A separate old-contract solve then
holds that fresh population, source identities, priors, awards, matrix, row
order, coefficients and solver budget fixed.

On the same paid target, the fresh old-contract weights estimate 1,831,438
lone-parent claims (−14.37%), compared with 1,839,214 after repair (−14.01%).
The apparent own-contract change from −17.73% to −14.01% is mostly a changed
comparison denominator. The ten broad UC rows' error on common CY targets
falls from 8.97% to 4.72%, with seven of ten within 5%, compared with five.
The retained-to-fresh differences combine prior pipeline changes and are not
an isolated causal estimate of the capital graph repair.

Its lone-parent row has 1,917 household rows from 677 original households,
source ESS 237.37 and a largest-source share of 2.42%. Its individual-row
capacity under the existing caps is 12.35 million against a 2.14 million
target. The same supported-108-UC and named-38-protected-row feasibility
checks described below pass on the fresh matrix. The lone-parent miss is
therefore not explained by that row's own lack of support or capacity.

A bounded directional diagnostic raises existing lone-parent-support weights
enough to improve the row by one percentage point. That direction increases
the full objective, chiefly through non-UC rows including the benefit-cap
caseload, Scottish Child Payment spending, ONS lone-parent households and
funded childcare. This identifies pressure along that particular direction;
it does not prove those rows prevent every compensating reallocation.

Six follow-up linear programs confirm that distinction. The broad ten UC
rows remain jointly feasible at 5% as the five named opposing rows are added
one by one. All 108 supported UC/TCL rows plus those five are also feasible.
The bounds remain zero to ten times each prior, with no total-mass or ESS
constraint. The largest check leaves the other 253 active rows unconstrained.
Thus these named rows alone do not establish an unavoidable conflict; the
remaining objective/constraint audit must examine the wider system and the
concentration of any feasible redistribution.

Across the wider system, all 364 supported rows are infeasible at 5%, and
the remaining 363 are still infeasible after omitting OBR CGT. Lone-parent
fit within 5% becomes feasible when the other 362 rows may deviate by 25%.
A further witness preserves total household weight and limits each weight
to ±25% of its current value, intersected with the existing prior caps.
It reaches a 0.75% lone-parent shortfall; household ESS changes from 14,050
to 13,791 and original-source ESS from 5,062 to 4,993. However, its OBR
income-tax estimate falls 14.05% and total NI falls 9.87% from current
estimates; income tax reaches its permitted −25% target boundary. The two
zero UC rows and OBR CGT are omitted only from these declared diagnostic
subsets, with no production target changes. No ESS constraint or
calibration-loss optimization is imposed, and no witness weights are
exported or proposed for release.

The terminal gate refuses export: both unsupported payment bands remain at
−100%, and five old fit exemptions are now within the permitted bound and
flagged as stale. No calibrated H5 or passing calibration build record is
produced. The signed failed-gate report and its bound diagnostic file are
retained; they are not converted into a release pass. The source HMRC replay
has reviewed-exclusion coverage only, with zero numerical comparison coverage,
so the source build's gate pass is not an HMRC fit claim.

The five retired exemptions originated in [#796](https://github.com/PolicyEngine/microcosm/issues/796).
That adjudication deferred four UC composition residuals and a private-pension
count residual attributed at the time to related solver pressure. Its earlier
UC support/capital explanation was subsequently marked historical as claimant
and child definitions changed; removing these entries does not validate that
old causal attribution. The [pre-removal register](https://github.com/PolicyEngine/microcosm/blob/cc9c953c72b003406f4aeec8ae5e82bd211099bc/packages/microcosm-build/src/microcosm/build/uk/target_fit_reviewed_exclusions.json)
preserves the original reasons, approval and expiry dates, and annotations.

The fresh candidate's observed errors are inside the unchanged absolute 25%
release bound:

| Retired target ID | Relative error |
|---|---:|
| `dwp.uc.households_children_1@2025` | −19.448861% |
| `dwp.uc.households_children_2@2025` | −6.420325% |
| `dwp.uc.households_children_5_or_more@2025` | +0.124633% |
| `dwp.uc.households_single_with_children@2025` | −14.006391% |
| `hmrc/private_pension_income_count_income_band_100_000_to_150_000@2025` | +0.044352% |

The [`retired_fit_deferral_evidence` block](evidence/uk-uc-882/calibration-comparison.json)
records these exact target values, estimates and errors from the completed
1,500-update, `family_equal` candidate, not a default-budget run. The canonical
diagnostic SHA-256 is
`971e6f8cac94b69ebf82364e83845f27d19fce38490a032524db237eb4884463`;
its original failed-gate report and exact recovery are bound in the
[population receipt](evidence/uk-uc-882/calibration-comparison.json).

The gate deliberately rejects an exemption once its target is back inside
the bound. Retiring these five entries removes those stale-exemption failures
and restores ordinary enforcement: any renewed absolute error above 25%
blocks release. All five targets remain in the solve. Replaying all 366 saved
errors through the updated register leaves exactly the two empty-tail
failures. `obr.capital_gains_tax@2025` remains deferred under
[#875](https://github.com/PolicyEngine/microcosm/issues/875), with a +44.771038%
error in this candidate; its exemption is unchanged. Neither the 25% gate nor
the calibration exclusions or optimizer defaults change. This does not
rewrite the original failed report or establish readiness under the default
256 updates and uniform target allocation.

## Empty cells and bounded feasibility

Two positive-target childless-couple payment rows have zero model records in
every retained arm:

| Monthly payment band | Source target | Model support |
|---|---:|---:|
| £2,300.01–£2,400 | 746.33 | 0 |
| £2,400.01–£2,500 | 605.78 | 0 |

Their annual matrix bounds are £27,600.12≤UC<£28,800.12 and
£28,800.12≤UC≤£30,000, retaining the source labels’ penny offset. Reweighting
leaves both at −100% under every feasible weight vector. These rows stay in
the evaluation; they are not silently excluded to claim a successful fit.
The same limitation was acknowledged in the
[diagnostic comment on #883](https://github.com/PolicyEngine/microcosm/pull/883#issuecomment-5587342177).

Diagnostic linear programs use bounds of zero to ten times each prior and
explicit 5% row tolerances. The complete 110-row UC/TCL set is infeasible
because of the two zero rows. Two separate, explicitly scoped checks are
feasible: the other 108 UC/TCL rows, and the ten broad UC counts together with
four named OBR targets and 24 HMRC employment amount/count bands. Neither
proves feasibility of all 366 rows, or of all 108 UC/TCL rows jointly with
every protected target. Feasible witness weights are not release candidates
or proof that Adam has converged. They justify investigating competing rows
and objective allocation before increasing the epoch budget.

The amount comparison remains unresolved: DWP monthly cash can contain
advances and payments on the claimant's behalf, while the model provides an
annual recurring award. No recurring entitlement is invented to fill these
cells. Component, deduction and timing evidence must precede support generation.

The fresh component audit reproduces all 24 active childless-couple matrix
rows exactly from annual model masks. Twenty copies from seven original
families have gross maximum UC above £2,300 per month-equivalent, but all
have income reductions. Even the largest recorded maximum-minus-reduction
balance is only £2,131.58 per month-equivalent; no pre-cap award reaches
the unsupported bands. Later benefit caps or deductions therefore do not
explain the upper-band zeros in this captured state. Eight of these copies
would claim and six have a positive payment. This identifies sparse
high-element/low-income combinations and take-up inputs for further audit;
it does not establish an incorrect income formula.

The £1,800–£2,200 near-tail has only 14 copies from seven original families,
source ESS 4.09 and a largest-source share of 38.62%. Two adjacent bands each
rely on one source despite fitting closely. The recorded UC maximum,
pre-cap and final-payment components reconcile within one penny annually.
Exact monthly cash timing and the first source stage of tail absence remain
unobserved.

## Five-plus children and independent support

The CY paid five-plus target itself has 80 household rows from 29 original
source households, source ESS 9.96 and a largest-source share of 27.6%.
Its near-zero residual does not demonstrate robust joint support.

The matrix contrast **reported five-plus minus TCL five-child households
minus TCL six-plus households** has four nonzero clone rows from just **one
original family**. Its source ESS is one and its largest-source share is 100%.
For CY the fitted contrast is 21,079 against target arithmetic of 21,040.
The three component rows' 5% tolerances imply a contrast interval of
[14,460, 27,620]. Removing all descendants of that one source leaves attainable
contrast [0, 0], so the three rows cannot jointly retain that fit. The FY arm
has the same one-source dependency.

The fresh paid five-plus row has 78 household rows from 28 original households,
source ESS 9.87 and a largest-source share of 27.5%. The contrast still has
four rows from one source, with fitted contribution 21,051. The fresh build
therefore does not remove this dependence.

The lost paid source is still in the data: all four descendants remain, with
the same roles, five-child count, eligibility, elements and income reductions.
Two SPI copies lose reporter/would-claim status. Fresh saved checkpoints never
promote those copies to reporters; capital coherence does not demote them.
The retained replay had a promotion this complete rebuild does not reproduce.
This whole-build comparison cannot isolate the earlier donor/assignment change
responsible, and does not justify restoring a promotion solely to recover fit.

This subtraction explains optimizer algebra across different concepts and
windows. It is not an administrative statistical identity or a newly inferred
joint. Extra clones of the same record do not provide independent evidence.
Diagnostics group descendants before calculating source concentration and
leave-one-source-out capacity.

## Date, claim and relationship limits

The source audit found birth-date fields entirely blank in the delivered
adult and child files. Official 2024/25 documentation confirms completed
age, adult age-80 top-coding and household `INTDATE` as interview-start date.
It does not establish the numeric TAB date encoding or exactness of every age
at that date. [UKDS variable listing](https://doc.ukdataservice.ac.uk/doc/9563/mrdoc/excel/9563_frs2425_variable_listing_eul.xlsx),
[derived-variable summary](https://doc.ukdataservice.ac.uk/doc/9563/mrdoc/excel/9563_dv_summary_2425.xlsx),
[questionnaire](https://doc.ukdataservice.ac.uk/doc/9563/mrdoc/pdf/9563_frs_2024-25_question_instructions_final.pdf#page=14)

A conditional interval sensitivity, assuming days since 1960 and accurate age
at the household start date, finds three original large families that could
have no post-cutoff child. None is confirmed unaffected. Two survive as paid
five-plus families in the retained spine; neither supplies the one-source
contrast above. The previous model-year-minus-age proxies do not identify
these ambiguous cases. No exact birthdays are imputed and no blanket age
increment is applied.

The next date repair must retain raw age/date validity and ancestry, verify
the encoding, then test before/after/ambiguous intervals around 6 April 2017
independently of model-year labels. April open/nil claim history, TCL exceptions,
education transitions and exact administrative child attachment remain
separate limitations. Annual zero awards or take-up flags cannot establish
open nil claims. The new Chronicle crosses do not supply these histories.

## Evidence and reproduction

The [aggregate comparison receipt](evidence/uk-uc-882/calibration-comparison.json)
contains the selected target values/estimates, source concentration, protected
outcomes, runtime and input/output hashes for each completed arm. It omits
individual source IDs and records. Original private archives and failed-gate
receipts remain unchanged.

`tools/diagnose_uk_uc_matrix.py` authenticates retained replay inputs, measures
the complete active roster and writes private diagnostic matrices/weights.
`tools/diagnose_uk_uc_support.py` reads a completed matrix receipt to reproduce
source concentration, capacity, named-subset LP probes and removal sensitivity:

```sh
uv run --no-sync python tools/diagnose_uk_uc_support.py \
  --run-dir /path/to/completed-run --output-dir /path/to/new-support-report
```

The latter performs no population simulation or calibration fit. Its LP
witnesses answer only their explicitly named feasibility questions. Keep
`source_group_influence_private.json` local; aggregate reports omit IDs.

The committed [LP evidence producer](../tools/diagnose_uk_uc_lp_evidence.py)
reproduces the later named-opposition, full-supported and bounded-redistribution
profiles, including heterogeneous tolerances and the fixed-mass/current-weight
bounds. With the authenticated fresh archives and the recorded Python 3.13.14,
NumPy 2.4.6 and SciPy 1.17.1 environment, run:

```sh
uv run --no-sync python tools/diagnose_uk_uc_lp_evidence.py \
  --run-dir /path/to/fresh-current-recovered \
  --output-dir /path/to/new-lp-reproduction \
  --profile all \
  --expected-evidence docs/evidence/uk-uc-882/calibration-comparison.json
```

The formatted producer reproduces all three historical families' deterministic
constraint, solver-status, fit, concentration and protected-outcome fields
exactly. Runtime and historical narrative/provenance are excluded from this
comparison. The tool refuses changed archives, missing or unexpected zero
rows, and mismatched replay versions; it verifies successful witnesses against
the declared constraints before reporting feasibility. It exports aggregate
results and verification receipts, without witness weights or source IDs.

The public comparison receipt retains the historical receipt/script hashes
and adds a separate reproduction attestation for the committed producer,
imported support helper, saved inputs, versions and result. Its
`expected_evidence_sha256` binds the comparison file read before appending
that attestation. A later replay against the augmented file checks the same
historical fields and creates its own receipt/hash. No population build,
model calculation or Adam calibration was rerun for this reproduction.
