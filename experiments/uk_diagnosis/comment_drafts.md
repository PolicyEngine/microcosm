# Draft issue comments — owner review only

These comments are **DRAFTS and were not posted**. The owner should replace the
relative evidence links with commit permalinks before posting to
`policyengine-uk-data`.

## DRAFT — policyengine-uk-data issue #452

**DRAFT for owner review; not posted**

The post-#735 administrative baseline materially changes the aggregate
diagnosis. On an aligned Great Britain benefit-unit basis, the pinned
enhanced-FRS v1.56.14 diagnostic gives 6.180 million benefit units with positive
modeled UC
against 6.759 million administrative UC benefit units: **-0.579 million
(-8.6%)**.
The older roughly 40% headline is not reproduced in the exact PolicyEngine UK
2.89.0/Core 3.26.11 lock; the exact-lock v1.56.16 artifact sensitivity, which
contains the benefit-unit sort fix, is 6.199 million (-8.3%). A current-upstream
2.91.0 cross-check reproduces positive-award benefit-unit,
element, family, and child counts to displayed precision despite a potentially
relevant formula difference:
2.89.0 subtracts only the benefit-cap reduction from final UC, while 2.91.0
also subtracts `uc_deductions`, which can change whether the award is positive
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-36`).
These are licensed aggregate diagnostics, not a standard acceptance or
calibration receipt. Full definitions and evidence:
[`uc_caseload_452.md`](uc_caseload_452.md).

The element diagnostic gaps are much larger than the total:

| UC series | Admin | Model v1.56.14 | Signed gap |
|---|---:|---:|---:|
| Housing element | 4.304m | 4.856m | +12.8% |
| LCWRA element | 2.417m | 2.089m | -13.6% |
| Carer element | 1.134m | 0.562m | -50.5% |
| Childcare element | 0.184m | 0.454m | +146.8% |

Family-type gaps are single/no children -13.1%, single/children -3.5%,
couple/no children -22.5%, and couple/children -9.1%. Element counts overlap
and do not sum. The four named family-history categories total 6.857 million,
1.5% above the deductions-series total, so they are a composition diagnostic
rather than an alternative total; the small unknown/missing category is not in
the table.

The mechanism audit does **not** support treating the residual as one additive
“take-up gap”:

- The frozen contract assigns a 55% take-up flag, anchors reported claimants,
  and gates final UC through `would_claim_uc`
  (`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk/take_up_contract.json:63-78`;
  `microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/stochastic_assignment.py:57-84`;
  `microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk_runtime/frs_take_up.py:105-151,178-221`;
  `policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/would_claim_uc.py:4-16`;
  `policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`).
  In the aligned GB accounting, 3.828 million positive-award benefit units had a
  positive enhanced-artifact reported signal and 2.352 million did not.
- The enhanced-artifact signal cannot independently identify FRS underreporting:
  it includes enhancement support and anchors take-up rather than defining the
  award
  (`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk_runtime/frs_take_up.py:105-151`;
  `policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit_reported.py:4-11`).
- Of 1.708 million weighted GB reporters receiving zero modeled UC, 0.815
  million failed the claim flag, about 0.779 million were flagged but
  ineligible—almost all through capital—and 0.114 million were eligible with
  final award zero;
  that last reduction source was not decomposed.
  Eligibility and capital flow through `is_uc_eligible` and
  `uc_assessable_capital`
  (`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/is_uc_eligible.py:4-15`;
  `policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/uc_assessable_capital.py:16-43`).
  A candidate for adjudication is that `corporate_wealth` includes private
  pension rights before UC capital assessment
  (`policyengine-uk@c93e1a05:policyengine_uk/variables/input/corporate_wealth.py:4-12`;
  `microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:255-264`).
  The v1.56.14 benefit-unit ordering defect contaminates the 0.815 million claim
  flag row, so it is a path count—not a take-up contribution; v1.56.16 is only
  a sensitivity
  (`policyengine-uk-data@12a1e028:policyengine_uk_data/datasets/frs.py:549-554`).
- Weighting is consequential: in the UK-wide v1.56.14 diagnostic, the 361
  highest-weight modeled-UC records, approximately the top 10% of 3,615, carry
  65.7% of modeled caseload. Take-up, reporting, eligibility, and weighting are
  overlapping diagnostics, not causal contributions.

microcosm#735 fixes one precise thing: it replaces 6.700 million with the
Ledger-derived average of **6,758,888.9 GB benefit units** and activates the
`universal_credit > 0` predicate as one national calibration objective
(`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk/target_references.json:9361-9378`;
`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:7595-7609`;
`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:21-94`).
Calibration can only change weights inside the joint solver
(`microcosm@2aa96795:packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:473-518,1331-1363,1738-1787,1840-1865`).

It therefore does **not** change take-up flags, eligibility, element formulas,
or recipient support; reconcile the remaining GB/UK and physical-household/
benefit-unit mismatches; or guarantee exact equality. It adds no element target,
and no committed post-#735 calibrated run demonstrates the resulting caseload.

Proposed next steps:

1. In microcosm#736, reconcile GB versus UK, benefit unit versus physical
   household, the family-history source basis, and nil-payment treatment before
   adding the chronicle#188 element/history facts.
2. Run licensed, predeclared counterfactuals separating current rules, forced
   take-up, exclusion of pension rights from assessable capital, and
   design/current/calibrated weights.
3. Re-pin the sorted artifact through licensed acceptance, then repeat the
   exact-lock diagnosis after a real post-#735 calibration. Do not tune the 55%
   rate, solver limits, gates, folds, bands, ceilings, or seeds toward the target.

The exact-lock aggregate run exists, but causal counterfactuals and a post-#735
calibration do not. Treat -8.6% and the component gaps as diagnosis, not a
certification result.

---

## DRAFT — policyengine-uk-data issue #448

**DRAFT for owner review; not posted**

This diagnosis separates two defects that should not share a corrective scalar:
**council-tax level/reconciliation** and **`owned_land` imputation stability**.
They are analytically distinct, but not fully causally independent for net
council tax.
Full evidence: [`council_tax_448.md`](council_tax_448.md).

For the licensed v1.56.14 exact-lock diagnostic, modeled static 2025 net council
tax is £41.440 billion in England, £3.555 billion in Scotland, and £2.609
billion in Wales. Against the old fixture comparators, that is -6.1%,
+16.5%, and +8.7%. Against current official publications, Scotland is +4.9%
versus £3.389 billion net billed after CTR and excluding water/sewerage; Wales
is +3.4% versus £2.524 billion collectable, while the same publication reports
£2.404 billion collected. Wales is not strictly like-for-like. The “roughly
20%” headline is comparator-sensitive and is not reproduced as a stable current
country result. Sources: [Scotland](https://www.gov.scot/publications/council-tax-collection-statistics-2025-26/pages/3/),
[Scottish methodology](https://www.gov.scot/publications/council-tax-collection-statistics-2025-26/pages/7/),
[Wales](https://www.gov.wales/council-tax-collection-rates-april-2025-march-2026-html),
and [England](https://www.gov.uk/government/statistics/council-tax-levels-set-by-local-authorities-in-england-2025-to-2026/council-tax-levels-set-by-local-authorities-in-england-2025-to-2026).

There is an unresolved target-definition/reconciliation problem before any
weighting fix. Country fixture components sum to £49.569 billion, while the
active national target is £50.925
billion, leaving £1.356 billion to reconcile; Northern Ireland domestic rates
are represented separately at £0.494 billion. The contract binds the broad
national value to household `council_tax_less_benefit`, while country references
have no active facts
(`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:1068-1093,1216-1240`;
`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:2762-2823`).
This belongs in microcosm#736 as a period/geography/definition reconciliation,
or an owner-signed exclusion if no like-for-like devolved fact exists—not as a
calibration scalar.

The model floor exposes a separate English mismatch: nominal CTR exceeds gross
liability on 462 artifact records / 0.559 million weighted households. The
£0.512 billion excess is why raw subtraction gives £40.927 billion but actual
floored net is £41.440 billion
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/council_tax_benefit.py:11-15`).
Reconcile unsupported-scheme reported CTR with gross liability before changing
either scalar.

A separate CTR concern may raise net liabilities. `claims_all_entitled_benefits`
uses a simulation-wide `.sum() < 1`, and
`would_claim_council_tax_reduction` consumes that result
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/income/claims_all_entitled_benefits.py:4-28`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/would_claim_council_tax_reduction.py:4-16`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/simulated_council_tax_reduction_benunit.py:24-94`).
The switch was false for all 61,223 units in this diagnostic. That is a
candidate semantics bug, not a quantified causal attribution; all shared
benefit consumers need tests before a change
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/housing_benefit/would_claim_housing_benefit.py:4-14`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/would_claim_CTC.py:4-14`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/would_claim_WTC.py:4-14`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/would_claim_IS.py:4-14`).

The exact-lock fixed-row/fixed-wealth/fixed-weight sensitivity confirms why this
needs adjudication: a literal per-benefit-unit input raises nominal UK CTR from
£2.277 billion to £3.941 billion and lowers net Scotland from £3.555 billion to
£3.104 billion, Wales from £2.609 billion to £2.474 billion, and England from
£41.440 billion to £40.361 billion. It moves the devolved values past the closest
current comparators and widens England's diagnostic gap; vectorizing toward a
target is not a defensible fix.

For gross liability, `owned_land` is downstream. Council tax is built from FRS
`ctannual`, including the Scottish water correction and region × band ×
single-adult cell imputation
(`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk_runtime/frs_council_tax.py:71-103`).
It then enters the WAS predictor frame before `owned_land` is drawn
(`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:333-339,350-427`).
Gross council tax remains a dataset input
(`policyengine-uk@2.89.0:policyengine_uk/variables/input/consumption/property/council_tax.py:4-15`),
so the land draw cannot change per-record gross before calibration. Net
liability has an indirect path: `owned_land` predicts later wealth draws
including `savings`, savings enters the CTR capital test, and CTR is subtracted
from gross
(`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:398-423`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/simulated_council_tax_reduction_benunit.py:85-92`;
`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`).
Its sign and size have not been isolated. A future calibration adds another
aggregate path because the active land target can change household weights and
therefore weighted gross and net totals
(`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:50-94`).

The #733 acceptance evidence supplied for this lane reports `owned_land`
adjacent-seed swing worsening at the 2024–25 artifact to **71.4% nationally and
112.9% in the worst region**, versus 37.7% nationally at 2023–24. The existing
owner-reviewed exclusion still records the older 37.7% evidence
(`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk/input_mass_reviewed_exclusions.json:13-18`).
The fitter already separates zero/positive regimes and uses stochastic weighted
draws
(`microcosm@2aa96795:packages/microcosm-fit/src/microcosm/fit/qrf.py:101-217,950-1003,1333-1380`),
so selecting a favorable seed or changing a gate or clipping threshold toward
passing would paper over the instability. A mechanism change is defensible only
if it is specified from evidence before its results are inspected.

Smallest defensible path:

1. Put council-tax component reconciliation and any signed exclusion in #736;
   preserve the Scottish water correction.
2. Adjudicate the population-wide CTR switch upstream with tests for all shared
   consumers, then rerun gross/CTR/net decomposition.
3. Have the owner attach the #733 receipt and separately decide (a) whether to
   supersede or re-sign the input-mass exclusion for the new vintage and (b)
   whether to hold, exclude, or reconcile the active household-land target
   (`microcosm@2aa96795:packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:9198-9222`).
4. Test an aggregate-preserving positive-tail / chain-isolation change under a
   fixed, predeclared adjacent-seed matrix. Report national, regional,
   nonzero-count, quantile, concentration, downstream wealth, CTR, and net-tax
   effects. Use common random numbers or predictor ablation to distinguish the
   `owned_land` increment from the whole wealth-chain seed effect, without
   tuning thresholds or seeds.

No pool build was run. Exact-lock baseline levels and the fixed-weight CTR
sensitivity are complete; both proposed fixes and country results after an
actual post-#735 calibration still require a licensed run before certification.
The named #733 receipt metrics were supplied to this lane, but the receipt file
itself was absent from the checkout.
