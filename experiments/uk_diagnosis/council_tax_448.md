# Council tax and `owned_land` diagnosis — UK data issue #448

Date: 2026-08-22

Status: diagnosis only. No pool build, calibration, target tuning, seed
selection, clipping change, or owner-only exclusion change was performed.

## Verdict

The two symptoms require separate diagnoses and should not share a corrective
scalar. They are not fully causally independent for **net** council tax.

- The “about 20% high” council-tax headline is not stable against the new
  baseline. On the pinned 2024–25 artifact, modeled net council tax is 16.5%
  above the old Scottish fixture and 8.7% above the old Welsh fixture. Against
  the closest current official comparators, the diagnostics are +4.9% and
  +3.4%. The quantities and periods are not identical, so none is yet a
  calibration-ready truth.
- `owned_land` remains a severe sparse-tail instability. The microcosm#733
  acceptance result supplied to this lane reports a 71.4% national and 112.9%
  maximum-region adjacent-seed swing at 2024–25, versus 37.7% nationally at
  2023–24. The named receipt was not present in this checkout, so the new values
  are quoted as task-supplied baseline evidence, not re-derived here.
- Gross council tax is upstream of `owned_land` and remains a dataset input, so
  land cannot affect **per-record gross** liability before calibration
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:333-339,350-427`;
  `policyengine-uk@2.89.0:policyengine_uk/variables/input/consumption/property/council_tax.py:4-15`).
  But `owned_land` enters the sequential predictors for later wealth draws,
  including `savings`; savings gates CTR through its capital test; and CTR is
  subtracted from gross council tax to form net
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:398-423`;
  `policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/simulated_council_tax_reduction_benunit.py:85-92`;
  `policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`).
  The size of that indirect `owned_land` → savings → CTR → net path has not
  been isolated. A future calibration creates a second aggregate path because
  the land target can change weights, and weighted gross council tax changes
  with them
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:50-94`).

The smallest honest path remains two-track: reconcile council-tax definitions
and adjudicate a shared Council Tax Reduction take-up switch; separately repair
the `owned_land` instability and its active household-land target. The licensed
counterfactual must also quantify their narrow interaction through savings and
CTR rather than assuming independence.

## Evidence basis and limits

The aggregate-only diagnostic used:

- `enhanced_frs_2024_25.h5` at v1.56.14, SHA-256
  `97a07f9ccb54019e4550e70980c561c985523e6bbc43d21938d01536e37d6c3e`,
  the #733 pin recorded at
  `packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:381-392`;
- the exact Microcosm lock, PolicyEngine UK 2.89.0 / Core 3.26.11, static 2025;
- v1.56.16 under the same exact lock as an artifact-sensitivity check.

This is not certification. The diagnostic matches the locked versions
(`uv.lock:1366-1367,1393-1398`), but no standard acceptance or post-#735
national-calibration receipt was produced and no pool was built. Licensed unit
records and row-level outputs are not reproduced here; writable temporary
artifact copies required by the HDF loader were removed after the aggregate
runs.

The aggregates are weighted household sums:

- gross = `council_tax`, a household dataset input
  (`policyengine-uk@2.89.0:policyengine_uk/variables/input/consumption/property/council_tax.py:4-15`);
- nominal CTR = `council_tax_reduction`, which aggregates benefit-unit CTR to household
  (`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/council_tax_reduction.py:4-16`);
- net = `council_tax_less_benefit`, gross less CTR floored at zero
  (`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:4-21`).

## 1. Council tax

### 1.1 Country aggregates

| Nation | Households, m | Gross, £bn | Nominal CTR, £bn | Effective CTR, £bn | Actual net, £bn |
|---|---:|---:|---:|---:|---:|
| England | 26.178 | 42.937 | 2.009 | 1.497 | 41.440 |
| Scotland | 2.708 | 3.804 | 0.249 | 0.249 | 3.555 |
| Wales | 1.580 | 2.619 | 0.010 | 0.010 | 2.609 |

“Effective CTR” is gross minus the actual household-floored net, so it can be
smaller than nominal CTR. The v1.56.16 exact-lock sensitivity gave actual net
£41.564 billion, £3.561 billion, and £2.620 billion respectively. The national
pattern is not an artifact-version discontinuity, though Welsh CTR is small
enough to be relatively sensitive.

### 1.2 Comparator choice changes the result

| Nation | Old 2025 fixture | Model net gap | Closest current 2025–26 publication | Model net gap |
|---|---:|---:|---:|---:|
| England | £44.117bn | -6.1% | £44.118bn council-tax requirement | -6.1% |
| Scotland | £3.051bn | +16.5% | £3.389bn net billed after CTR, excluding water/sewerage | +4.9% |
| Wales | £2.401bn | +8.7% | £2.524bn collectable, with CTRS treatment | +3.4% |

Official sources are the [England 2025–26 council-tax requirement](https://www.gov.uk/government/statistics/council-tax-levels-set-by-local-authorities-in-england-2025-to-2026/council-tax-levels-set-by-local-authorities-in-england-2025-to-2026),
[Scottish amount billed](https://www.gov.scot/publications/council-tax-collection-statistics-2025-26/pages/3/),
[Scottish methodology](https://www.gov.scot/publications/council-tax-collection-statistics-2025-26/pages/7/),
and [Welsh collection statistics](https://www.gov.wales/council-tax-collection-rates-april-2025-march-2026-html).

These are diagnostic comparators, not interchangeable facts:

- England publishes a requirement, not collected receipts.
- Scotland publishes billed liability net of CTR and excluding water and
  sewerage.
- Wales publishes collected and collectable quantities with CTRS-specific
  treatment; £2.524 billion is only the closest available quantity.
- The model is static calendar 2025; the publications are fiscal 2025–26.

Using Scottish gross council tax against the old net fixture gives +24.7%, near
the issue headline, but mixes definitions. The target contract binds council
tax to net `council_tax_less_benefit`, not gross `council_tax`
(`packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:1068-1093`).
The blanket “~20%” overstatement is therefore not reproduced as a stable current
country result.

### 1.3 Gross construction is not yet the demonstrated defect

For observed FRS values, the build uses `ctannual`. In Scotland it subtracts
nonnegative water and sewerage amounts. Missing values are imputed from region ×
council-tax-band × single-adult cell means
(`packages/microcosm-build/src/microcosm/build/uk_runtime/frs_council_tax.py:71-103`).
The implemented Scottish direction is already the required subtraction.

This lane did not isolate reported versus cell-mean-imputed gross liability in
Microcosm's exact lock. There is no evidence yet for changing council-tax bands,
cell definitions, or gross-liability scalars.

### 1.4 The net floor exposes a separate English reconciliation

The model floors gross less household CTR at zero
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`).
On 462 English artifact records representing 0.559 million weighted households,
nominal CTR exceeds gross liability. The excess is £0.512 billion, which is why
raw gross-minus-CTR is £40.927 billion but actual net is £41.440 billion.
Those displayed aggregate components are independently rounded.
Unsupported schemes select reported benefit rather than simulated CTR
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/council_tax_benefit.py:11-15`).
The aggregate diagnostic locates all of this floor excess on unsupported-scheme
records. That reported-CTR/gross mismatch needs reconciliation before either a
gross or CTR scalar is considered.

### 1.5 Council Tax Reduction has a concrete code-level candidate

`claims_all_entitled_benefits` is declared as a benefit-unit boolean, but the
formula sums seven reported-benefit arrays before comparing the population-wide
sum with one
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/income/claims_all_entitled_benefits.py:4-28`).
The resulting `.sum() < 1` is a simulation-wide scalar, not one result per
benefit unit. In this artifact it is false for all 61,223 benefit units, although
51,855 have a within-benefit-unit sum below £1 across the seven reported fields
(51,849 are exactly zero).

`would_claim_council_tax_reduction` ORs that scalar with reported CTR
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/would_claim_council_tax_reduction.py:4-16`).
In this diagnostic the take-up support therefore reduces to reported CTR only.
For supported schemes, the formula restricts awards to the household-head unit,
multiplies by `would_claim`, and applies eligible liability, maximum support,
excess-income withdrawal, non-dependent deductions, and the savings limit
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/simulated_council_tax_reduction_benunit.py:24-94`).
Supported schemes use that simulated result instead of the reported amount
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/council_tax_benefit.py:4-15`).

Current-upstream support diagnostic (2.91.0):

| Nation | Real FRS reported-CTR records routed to supported schemes | Positive modeled award | Zero: savings | Other formula zero |
|---|---:|---:|---:|---:|
| Wales | 341 | 80 | 102 | 159 |
| Scotland | 451 | 192 | 98 | 161 |

These counts exclude SPI-synthetic records. The corresponding supported-scheme
reporter counts across all artifact rows are 578 in Wales and 669 in Scotland.
For Wales, the other 159 real rows split into 119 income-taper zeros and 40 with
no eligible liability. The 80 positive real records have mean weight 148 versus
614 among zeroed real records. The exact-lock aggregate calculation across all
artifact rows gives nominal CTR of £249 million in Scotland and £10 million in
Wales.

Suppressing CTR raises `council_tax_less_benefit` by construction
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`).
That establishes direction, not the causal country contribution. The source
comment may intend a simulation-wide policy switch, so semantics must be
adjudicated before changing it. The variable is also shared by housing benefit
and legacy-credit take-up consumers; a one-line vectorization would be wider
than issue #448
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/housing_benefit/would_claim_housing_benefit.py:4-14`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/would_claim_CTC.py:4-14`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/would_claim_WTC.py:4-14`;
`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/would_claim_IS.py:4-14`).

An exact-lock, fixed-row, fixed-wealth, fixed-weight counterfactual tested the
literal per-benefit-unit reading without changing code or tuning a target:

| Measure | Current scalar semantics | Per-benefit-unit input |
|---|---:|---:|
| Nominal UK CTR | £2.277bn | £3.941bn |
| Actual UK net council tax | £47.603bn | £45.940bn |
| England net | £41.440bn | £40.361bn |
| Scotland net | £3.555bn | £3.104bn |
| Wales net | £2.609bn | £2.474bn |

Against the closest current comparators, that moves Scotland from +4.9% to
-8.4%, Wales from +3.4% to -2.0%, and England from -6.1% to -8.5%. The
comparators remain non-equivalent, but the direction is decisive for diagnosis:
a one-line vectorization is not a target-fitting fix. The shared contract must
be adjudicated and all consumers tested.

### 1.6 The new target surface needs reconciliation first

The active 2025 reference is the broad national value:

- `obr.council_tax` = £50.925163826 billion
  (`packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:2762-2774`);
- it binds household `council_tax_less_benefit`
  (`packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:1068-1093`).

England, Scotland, Wales, and domestic-rates references have no active facts at
or before the period
(`packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:2777-2823`).
Their old fixture values are £44.117 billion, £3.051 billion, £2.401 billion,
and £0.494 billion
(`packages/microcosm-build/src/microcosm/build/uk/ledger_compile_parity_incumbent_2025_signed_differences.json:325-351`).

The three council-tax components sum to £49.569 billion, £1.356 billion below
the active national target. Adding the separate £0.494 billion domestic-rates
fixture still leaves £0.862 billion unreconciled. No accrual, coverage, or
geography adjustment should be invented to close that identity.

The exact-lock artifact's actual UK net is £47.603 billion, £3.322 billion or
6.5% below the active £50.925 billion target. That uncalibrated model-to-target
gap is distinct from the £1.356 billion identity gap among reference
definitions.

Chronicle carries OBR rows labelled Scotland and Wales, but the 2025 records use
UK geography `K02000001` and entity `person`
(`chronicle@7754c8c:packages/obr/efo_expenditure_march_2026/source_package.yaml:326-410,627-712`).
That explains why country-filtered household references find no facts: this is a
source-geography and measurement reconciliation, not evidence that country
values are zero.

The build records `artifacts.national_calibration` only when calibration evidence
exists (`packages/microcosm-build/src/microcosm/build/uk_runtime/national_build.py:549-558`),
and the committed staging record contains no post-#735 entry
(`packages/microcosm-build/src/microcosm/build/uk/national_staging_build_record.json:98-116`).
No committed run demonstrates how the broad total changes country composition.

## 2. `owned_land`

### 2.1 Sparse-tail concentration

Aggregate-only v1.56.14 diagnostics:

| Nation | Total, £bn | Records > 0 | Top 10 share | Share of `household_land_value` |
|---|---:|---:|---:|---:|
| England | 326.3 | 348 | 79.7% | 7.0% |
| Scotland | 62.0 | 16 | 100.0% | 21.6% |
| Wales | 17.8 | 56 | 97.4% | 9.5% |
| Northern Ireland | 15.0 | 336 | 67.5% | 27.2% |
| UK | 421.1 | — | — | 8.1% |

These are stored-output diagnostics under incumbent artifact weights, not a QRF
rerun. Totals are weighted sums; positive-record counts are unweighted and
include synthetic rows; each top-10 share uses the ten largest weighted
`weight × owned_land` contributions; and the final column is a ratio of weighted
sums. The parity 1.4306% below is unweighted, whereas the exclusion's 0.7% is
weighted.

The parity reference records only 1.4306% of rows as nonzero
(`packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:320-332`).
The reviewed exclusion records a 0.7% weighted nonzero share and the earlier
2023–24 result: 37.7% national adjacent-seed movement, 2.41× London, and 4.6×
Wales
(`packages/microcosm-build/src/microcosm/build/uk/input_mass_reviewed_exclusions.json:13-18`).

The #733 acceptance values supplied for this task are 71.4% national and 112.9%
maximum-region movement at 2024–25. The referenced
`data/ukds/acceptance/723-frs-2024-25/owned_land_stability_receipt.json` was not
present in this checkout or another readable committed tree; those numbers were
not independently re-derived in this lane.

### 2.2 Why the draw is unstable

The WAS mapping sends `DVLUKValR8_sum` to `owned_land`
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:120-129`).
With zeros and positives in donor support, the fitter classifies the target as
zero-inflated positive, fits a weighted sign gate and positive-magnitude forest,
and draws both sign and conditional quantile from the seeded stream
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:101-150,950-1003,1333-1380`).
Donor weights enter through weighted bootstrap
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:188-217`).

Microcosm declares seed 0 and passes it to the wealth fit
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:78,193-201`).
The adjacent-seed receipt therefore measures realization sensitivity, not an
undeclared seed.

`owned_land` is drawn first in the wealth chain; later property, corporate, and
financial targets consume prior draws
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:368-423`).
Post-draw protection clips each output to donor-realized minimum and maximum but
does not preserve national or regional aggregates
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:481-496`).
Sixteen positive Scottish records can therefore control the Scottish aggregate
while remaining inside donor support.

### 2.3 The existing exclusion does not neutralize calibration exposure

The current exclusion is scoped to the `efrs-post-calibration` input-mass check
and expires on 2026-09-20
(`packages/microcosm-build/src/microcosm/build/uk/input_mass_reviewed_exclusions.json:3-18`).
It neither changes the imputed column nor disables a calibration reference.
Exclusion changes are owner-only.

PolicyEngine computes household land value as `property_wealth` multiplied by a
regional intensity, plus `owned_land`
(`policyengine-uk@2.89.0:policyengine_uk/variables/household/wealth/household_land_value.py:14-56`).
Microcosm has an active £4.559831 trillion target bound to that variable
(`packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:9198-9222`;
`packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:7358-7369`).
An unstable draw would enter a future calibration loss unless #736 explicitly
holds, excludes, or reconciles that target.

### 2.4 Dependency boundary: no gross path, possible net path

Council tax is already present on the household frame and copied into the WAS
predictor frame before wealth fitting
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:333-339`).
The wealth stage then writes `owned_land` and later wealth variables
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:350-427`).
The policy model's gross council tax remains a dataset input with no wealth
formula
(`policyengine-uk@2.89.0:policyengine_uk/variables/input/consumption/property/council_tax.py:4-15`).
The dependency is one-way for **per-record gross** council tax before
calibration: council tax can help predict land, but land cannot feed back into
the gross input formula.

That does not prove independence from **net** council tax. Within the sequential
wealth chain, `owned_land` and `property_wealth` become predictors for later
draws, and `savings` is one of the later outputs
(`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:398-423`).
The Scottish/Welsh CTR formula requires household savings to be below its
capital limit, and net council tax subtracts CTR from gross
(`policyengine-uk@2.89.0:policyengine_uk/variables/gov/local_authorities/council_tax_reduction/simulated_council_tax_reduction_benunit.py:68-92`;
`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`).
An unstable land realization can therefore propagate to net through later
savings. This lane has not measured that path's sign or magnitude, and the
simulation-wide take-up switch can mask it by suppressing CTR first.

There is a second prospective aggregate path. `household_land_value` is an
active calibration target, so a changed land draw can change solved household
weights; those weights then change aggregate gross and net council tax even
though the per-record gross input is fixed
(`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:50-94`;
`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:1738-1787,1840-1865`).

## 3. Smallest defensible fix paths

### Council tax

1. **Reconcile before calibrating.** In microcosm#736, record period, geography,
   and exact quantity for every intended fact: gross liability, net billed, net
   receipts, requirement, collectable, or collected. Reconcile the £50.925
   billion national total with country components and NI domestic rates. If no
   like-for-like devolved fact exists, use an owner-signed exclusion or a
   reconciliation-layer entry rather than a convenient scalar.
2. **Adjudicate the shared take-up contract.** Decide whether
   `claims_all_entitled_benefits` is per benefit unit or deliberately
   simulation-wide. If per unit, make the predicate explicitly vector-valued
   and add consumer tests for CTR, housing benefit, CTC, WTC, and Income Support.
3. **Extend the predeclared counterfactual.** The exact-lock current-versus-per-
   benefit-unit comparison is now complete under fixed rows, wealth, and weights.
   Add reported/imputed gross and reporter/non-reporter CTR decompositions, then
   repeat after a real post-#735 calibration. Freeze accepted wealth inputs
   first, then compare the predeclared adjacent-seed wealth realizations to bound
   the aggregate wealth-imputation effect on CTR and net without attributing all
   of it to `owned_land`.
4. **Do not change gross imputation yet.** Revisit region × band × single-adult
   cells only if the exact-lock decomposition leaves a gross residual after
   definitions and CTR are aligned. Preserve the Scottish water correction. Do
   not tune scalars, bands, weights, gates, or target ceilings.

### `owned_land`

1. **Use exclusion only as an owner-controlled bridge.** The owner should attach
   or restore the #733 receipt and decide in #736 whether the active
   `household_land_value` reference is held, excluded, or reconciled. The
   existing input-mass exclusion alone is insufficient.
2. **Isolate the sparse target.** Test removing `owned_land` from the shared
   sequential wealth chain and fitting an independent two-part zero/positive
   process. Compare only predeclared alternatives, such as an
   aggregate-preserving positive-tail model. Any donor-tail rule must be fixed
   from survey methodology before results are inspected.
3. **Predeclare acceptance.** Use a fixed adjacent-seed matrix and report
   national/country totals, nonzero incidence, positive quantiles, top-record
   concentration, and downstream effects on `property_wealth`,
   `household_land_value`, later chained wealth outputs, CTR, and net council
   tax. Add a common-random-number or predictor-ablation experiment to isolate
   the incremental `owned_land` → savings path from the whole wealth-chain seed
   effect. Require fidelity as well as stability; selecting a favorable seed or
   widening a gate is not a fix.

## 4. Claims requiring licensed confirmation

- reported-versus-imputed gross council tax by nation;
- country outcomes after a real post-#735 national calibration;
- the `owned_land` → later savings → CTR → net effect, separated from the full
  wealth-chain seed effect;
- direct inspection of the #733 `owned_land` receipt at its named path;
- candidate `owned_land` imputation results and downstream wealth effects.

Until those runs exist, the defensible conclusion is narrower than the issue's
original wording: council tax has an unresolved target-definition/reconciliation
problem, an English reported-CTR floor mismatch, and a shared take-up-contract
candidate; `owned_land` has a separately demonstrated sparse-tail stability
defect. Land cannot alter the per-record gross input before calibration, but it
may affect net through later savings/CTR and may affect weighted gross and net
through future calibration weights.
