# Final report: UK UC caseload and council-tax diagnosis

Date: 2026-08-22

## Outcome

The two upstream defects remain open, but the new fact baseline changes their
shape.

- **UC:** on the aligned Great Britain benefit-unit diagnostic, modeled 2025
  caseload is 6.180 million against 6.759 million in administration: **-0.579
  million (-8.6%)**. The stale roughly 40% headline is not reproduced under the
  exact PolicyEngine UK 2.89.0/Core 3.26.11 lock; a current-upstream 2.91.0
  cross-check reproduces the displayed UC counts.
  The aggregate masks large and opposing element diagnostic gaps: housing
  +12.8%, LCWRA -13.6%, carer -50.5%, and childcare +146.8%. The complete evidence and
  family-type results are in
  [the UC diagnosis](experiments/uk_diagnosis/uc_caseload_452.md).
- **Council tax:** the reported roughly 20% Scottish/Welsh overstatement is not
  stable across definitions. Against the old fixtures the net gaps are +16.5%
  and +8.7%; against the closest current official comparators they are +4.9%
  and +3.4%, with period and quantity mismatches still unresolved. The full
  comparison is in [the council-tax diagnosis](experiments/uk_diagnosis/council_tax_448.md).
  The corrected exact-lock England net is £41.440 billion: nominal CTR exceeds
  gross on 462 records, so the model's household zero floor raises net by
  £0.512 billion relative to raw subtraction
  (`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`).
- **`owned_land`:** this is a separate sparse-tail problem. The #733 acceptance
  values supplied to this lane are 71.4% national and 112.9% maximum-region
  adjacent-seed movement for 2024–25, versus 37.7% nationally for 2023–24. The
  named new receipt was absent locally, so those values are reported as supplied
  baseline evidence rather than independently re-derived.

The owner-ready, unposted text for both upstream issues is in
[the draft-comments file](experiments/uk_diagnosis/comment_drafts.md).

## Diagnosis boundary

The UC and council-tax baseline results are aggregate-only licensed diagnostics
on the parity-pinned v1.56.14 artifact under Microcosm's exact 2.89.0/Core
3.26.11 lock (`uv.lock:1366-1367,1393-1398`), plus a v1.56.16 artifact
sensitivity containing the benefit-unit sort fix. They are not standard
acceptance or calibration receipts. No
row-level licensed output was written or committed, and temporary writable HDF
copies were deleted after measurement. The current 2.91.0 formula difference
is material in principle: 2.89.0 subtracts
only the benefit-cap reduction from final UC, while 2.91.0 also subtracts
`uc_deductions`, which can turn a positive award into zero
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-36`).
Separate runs produced the same UC positive-award benefit-unit, element, family,
and child counts to the displayed precision.

The UC residual cannot be assigned to one additive mechanism. The take-up
contract applies a frozen 0.55 assignment before the model gates awards through
`would_claim_uc`
(`packages/microcosm-build/src/microcosm/build/uk/take_up_contract.json:63-78`;
`packages/microcosm-build/src/microcosm/build/uk_runtime/frs_take_up.py:105-151`;
`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`),
capital eligibility consumes a proxy that includes `corporate_wealth`
(`policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/uc_assessable_capital.py:16-43`;
`policyengine-uk@c93e1a05:policyengine_uk/parameters/gov/dwp/universal_credit/means_test/capital/sources.yaml:1-20,34-39`;
`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:255-264`),
and calibration changes weights inside a joint constrained solve
(`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:21-94`;
`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:473-518,1331-1363`).
The weighted bridge and exact upstream model citations are preserved in the UC
report; causal shares require the predeclared licensed counterfactual matrix
specified there.

microcosm#735 fixes exactly the overall target contract: it selects GB
benefit-unit facts, averages the available 2025 months, resolves 6,758,888.9,
and activates the final-award predicate as a calibration objective
(`packages/microcosm-build/src/microcosm/build/uk/target_references.json:9361-9378`;
`packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:7595-7609`;
`packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:9980-10019`).
It does not itself run calibration: the committed staging record has no such
result and the build records one only when evidence is supplied
(`packages/microcosm-build/src/microcosm/build/uk/national_staging_build_record.json:98-116`;
`packages/microcosm-build/src/microcosm/build/uk_runtime/national_build.py:549-558`).
The solver replaces weights rather than recipient inputs or formulas
(`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:1738-1787,1840-1865`),
so #735 cannot repair recipient support or validate reform caseloads. It also
does not add the four new element targets or reconcile administrative
universes; those inventory boundaries are detailed in the UC report.

Council tax and `owned_land` must be separated at the **per-record gross**
boundary before calibration:
council tax is copied into the WAS predictor frame before land is drawn
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:333-339,350-427`),
and gross council tax remains a dataset input
(`policyengine-uk@2.89.0:policyengine_uk/variables/input/consumption/property/council_tax.py:4-15`).
They are not fully independent for **net** council tax: `owned_land` enters
later wealth predictors including `savings`, savings gates CTR through its
capital test, and CTR is subtracted from gross
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:398-423`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/simulated_council_tax_reduction_benunit.py:85-92`;
`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`).
The sign and magnitude of that indirect path remain unmeasured. The
council-tax path also needs definition reconciliation and adjudication of the
simulation-wide Council Tax Reduction take-up switch
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/income/claims_all_entitled_benefits.py:4-28`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/would_claim_council_tax_reduction.py:4-16`).
An exact-lock fixed-weight sensitivity already shows that the literal
per-benefit-unit interpretation raises nominal UK CTR from £2.277 billion to
£3.941 billion, moves the devolved values past the closest comparators, and
widens England's diagnostic gap. It must not be vectorized toward passing. A
future land-target calibration adds another
path: changed land draws can change solved weights and therefore weighted gross
and net totals
(`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:50-94`).
The land path needs a fixed, predeclared stability experiment on its two-part
stochastic fit
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:101-217,950-1003,1333-1380`).

## Owner handoff

microcosm#736 should own the signed measurement decisions:

1. reconcile GB versus UK, benefit-unit versus physical-household grain,
   nil-payment treatment, and independently queried UC histories before adding
   element/family targets;
2. reconcile the broad £50.925 billion council-tax reference with country
   quantities and Northern Ireland domestic rates by period and definition;
3. attach the #733 `owned_land` receipt, then decide separately whether to
   supersede the input-mass exclusion and whether to hold, exclude, or reconcile
   the active land reference.

Upstream model/data work should then test, without tuning, the UC pension-capital
and take-up candidates, finish decomposing the shared CTR switch, and test an
isolated aggregate-preserving `owned_land` process. The land experiment must
also measure the possible savings/CTR/net propagation separately from the whole
wealth-chain seed effect. The exact counterfactuals and required outputs are
enumerated in the two diagnosis reports.

## Controls and validation

- No pool was built; no calibration was run; no gate, band, ceiling, fold, seed,
  clipping rule, target, or owner-only exclusion was changed.
- No issue comment was posted, no branch was pushed, and no licensed row-level
  data was committed.
- The initial journal commit passed every `packages/*/tests` shard and Ruff.
  The closing deliverable commit is created only after the same exact all-shard
  and Ruff command exits successfully with this complete tree; no file is
  changed between that gate and the commit.
