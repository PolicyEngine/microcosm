# 100-record exact-k versus PolicyBench's sampled 100

This is a local evidence record comparing the frozen PolicyBench sample with a
support chosen by Microcosm's maintained L0/exact-k calibration path. It does
not evaluate models or certify either design for a benchmark or release.
Every reported result is rendered from the bundled JSONs by
`compare_designs.py render-readme`.

The household comparison is the centerpiece. Household rows are confined to
[households.csv](households.csv) and [household_ids.json](household_ids.json),
following the task's latest rule; this README reports their aggregate contrast.
The CSV groups the sampled and national-target exact-k sets and sorts each by
employment income. Both exports include only public-dataset household rows.

## Household comparison

<!-- BEGIN generated:contrast -->
| unweighted household count / weight diagnostic | sampled | exact_k_national_all | exact_k_headline |
|---|---|---|---|
| households | 100 | 100 | 100 |
| one-person households | 58 | 26 | 33 |
| households of 6+ | 1 | 2 | 1 |
| head 65+ | 33 | 23 | 19 |
| head under 30 | 14 | 8 | 9 |
| with children | 18 | 38 | 25 |
| zero employment income | 36 | 24 | 29 |
| employment income above population p90 | 6 | 16 | 22 |
| employment income above population p99 | 0 | 5 | 9 |
| SNAP | 16 | 29 | 23 |
| Medicaid | 17 | 36 | 25 |
| SSI | 5 | 1 | 2 |
| EITC | 11 | 35 | 16 |
| Social Security | 43 | 26 | 28 |
| any of SNAP / Medicaid / SSI | 21 | 44 | 29 |
| two or more of SNAP / Medicaid / SSI / EITC | 16 | 34 | 24 |
| distinct states | 34 | 40 | 39 |
| in the top 1% of design weight | 58 | 6 | 11 |
| PolicyBench-eligible | 100 | 76 | 83 |
| median design-weight percentile | 99% | 73% | 67% |
| max/mean weight | 1.0 | 21.0 | 8.4 |
| top-10 share of weight | 10.0% | 73.1% | 55.4% |
| Kish ESS | 100.0 | 12.4 | 24.5 |
| weight / design weight, median | 15 | 3,829 | 4,954 |
| weight / design weight, max | 34,033 | 8,038 | 11,465 |

Sampled versus national-target exact-k: 58 versus 26 one-person households; 33 versus 23 retirement-age heads; 6 versus 16 households above the population's upper-decile employment-income threshold; 21 versus 44 receiving SNAP, Medicaid or SSI; 34 versus 40 states. The top-weight households carry 10.0% versus 73.1% of their sets' total weight.
<!-- END generated:contrast -->

<!-- BEGIN generated:households_sampled -->
Sampled household rows are in [households.csv](households.csv), under `design=sampled`, sorted by employment income.
<!-- END generated:households_sampled -->

<!-- BEGIN generated:households_exact_k -->
National-target exact-k household rows are in [households.csv](households.csv), under `design=exact_k`, sorted by employment income. Identifiers and mirrored rows for the sampled and national-target exact-k sets are in [household_ids.json](household_ids.json); the headline-target sensitivity is reported only in aggregates.
<!-- END generated:households_exact_k -->

<!-- BEGIN generated:leverage -->
Population design-weight threshold for the top percentile: 68,014.53; population median design weight: 36.22. Weighted employment-income upper-decile threshold: $188,210; upper-percentile threshold: $500,000.

exact_k_national_all: 6 selected households are in the top design-weight percentile, 26 are one-person households, 5 exceed the upper-percentile employment-income threshold, and 76 satisfy PolicyBench eligibility. The largest weights carry 73.1% of mass with Kish ESS 12.4.

exact_k_headline: 11 selected households are in the top design-weight percentile, 33 are one-person households, 9 exceed the upper-percentile employment-income threshold, and 83 satisfy PolicyBench eligibility. The largest weights carry 55.4% of mass with Kish ESS 24.5.

Deduction: these diagnostics indicate how far a selected support behaves as integration points for the fitted aggregates (quadrature nodes). Concentration or unusual income support alone does not establish why any individual household was selected, and does not measure model-evaluation coverage.
<!-- END generated:leverage -->

## What the designs measure

- **PPS sample, self-weighted:** the frozen public sample, with each household
  assigned eligible population weight divided by sample size. This is an
  approximation to an inclusion-corrected estimator.
- **PPS sample, post-stratified:** the same households, refitted to each target
  set with ordinary calibration and no L0 selection.
- **Exact-k:** L0 selection from the full source population, the maintained
  exact-cardinality draw, and an ordinary refit on the selected support.
  `select_exact_k` returns support-aligned inclusion probabilities
  (`packages/microcosm-calibrate/src/microcosm/calibrate/exact_k.py:527`).
  `refit_l0_selection` starts from the selected original weights divided by
  those probabilities and normalized to full-file mass
  (`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:2382`).
- **Sampling distribution:** repeated PPS draws with self-weighting and
  post-stratification, using the frozen draw's eligibility and public split.

The primary fit measure is uncapped MAPE with equal weight per target.
The denominator is the target's absolute value with a unit floor, matching
`default_target_loss_scales`
(`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:594`).
Capped MAPE is secondary. The fit tables evaluate each design against the same
compiled targets, including targets unsupported by a particular sampled set.
Kish effective sample size measures weight concentration, not model accuracy.

The national set contains **26 feed-declared zero-valued targets**. For these
rows the denominator is one unit, so the uncapped fixed-scale “MAPE” includes
absolute dollar or count misses; it is not a conventional percentage error
relative to a nonzero target. A zero EITC-dollar target dominates that mean.
The feed metadata does not establish whether this zero is substantive or
reflects an extraction/source-data issue, and the underlying workbook cells
were not verified. The [zero-target audit](zero_target_audit.json) records the
observations and provenance. All targets were held fixed across designs.
Capped error and within-tolerance shares provide additional context; improving
the capped objective can coexist with increasing an error already above its
cap (a deduction from the capped loss, documented in the audit).

## Reading the results

<!-- BEGIN generated:reading -->
On national_all, uncapped MAPE is 1863675480.0% for the frozen self-weighted sample and 2116681946.4% after post-stratification, versus 58.7% for exact-k.

On headline, uncapped MAPE is 38.6% for the frozen self-weighted sample and 4.1% after post-stratification, versus 17.9% for exact-k.

Deduction: self-weighting preserves an interpretable approximately equal-weight draw from PolicyBench's eligible population. Post-stratification can improve aggregate fit on those same households, but changes their influence. Exact-k spends the household budget on fitting the chosen aggregates across the full source population; its weight concentration, eligibility and receipt coverage determine the cost of that fit. The national and headline runs answer different target-set questions and should be compared separately.
<!-- END generated:reading -->

## Source and reproduction

<!-- BEGIN generated:source -->
Source release: `populace-us-2024-5da5a95-20260611`; artifact SHA-256 `f32c2e5e9098bc6540724fdd5debf963af495da4c29b3a7a63fb53c2a4bb5a34`; maintained loader: `microcosm.build.us_runtime.h5_io.load_legacy_calibrated_us_h5`. Engine versions: `{"numpy": "2.4.6", "policyengine_core": "3.32.5", "policyengine_us": "2.2.1", "torch": "2.12.0"}`.

The full file has 75,112 households with total design weight 160,159,870.9. PolicyBench's eligible pool has 63,128 households and 76.9% of that weight. This scope difference matters: the primary self-weighted sample expands to the eligible pool, while exact-k starts from the full file. The fit tables include the sample rescaled to full-file mass as a sensitivity check.

Frozen snapshot: `paper/snapshot/20260501/runs/us_full_run_20260612_policyengine_4_16_1_populace`; scenarios CSV SHA-256 `71b16212f0c0b3e5d13d8694ce57e362c23248665806c4d6dea7b23ef472858a`; metadata SHA-256 `03a66e90b86e9bd0cc77f27520784bd581777762f749675dc716e24c1b8eaebb`.

Producer source receipt: `{"git_head": "2b85b7b220c33811336913ab29d97acbf1274455", "modules_sha256": {"microcosm.calibrate.exact_k": "8a400c46e4c94773dc0d83122f173c8c3bd405694d411989b10f792597b94196", "microcosm.calibrate.gates": "14608a5ee02132f06eaad4a59de3b051c8b8e1d07bd1768f4eb57f237173cef7", "microcosm.calibrate.matrix": "ca609f47ba7f5bf55ab8a0c85c07d2dac9b7a52fffcf0816f8289f143764c55a", "microcosm.calibrate.solve": "de380f489c84d6773e8c63fc41333ce038bc6f81447fb9388b939620b4c5b4c4"}, "script_sha256": "12ee1e6fca64eee09ad2521851be5015cf54826471c8a210db55a1011a9fe0ad"}`.

Smoke/pipeline-only output: False.
<!-- END generated:source -->

From the repository root, use the existing environment:

```bash
OMP_NUM_THREADS=4 .venv/bin/python experiments/policybench-exact-k-100-vs-sampled-20260922/compare_designs.py run
```

Run output belongs in a local ignored log. Each completed design is also
written to its own JSON before subsequent designs run. Re-render with the same
command using `render-readme` in place of `run`. The recorded settings and seed
count below are authoritative for this run. Use `validate` in place of `run`
to check the bundled exports, summaries and generated README blocks.
Aggregate receipts survive interruption; household arrays stay in memory and
must be recomputed after a restart.

## Settings

<!-- BEGIN generated:settings -->
- Design 2 and the exact-k refit: `{"method": "adam", "mass": "free", "max_weight_ratio": 10.0, "epochs": 1024, "learning_rate": 0.02, "seed": 0}`.
- Exact-k L0 selection: `{"method": "adam", "mass": "conserve", "max_weight_ratio": null, "target_records": 100, "budget_basis": "nonzero_count", "epochs": 1024, "learning_rate": 0.02, "init_mean": 0.5, "temperature": 0.25, "budget_iters": 10, "seed": 0}`; Sampford boundary draw seed 0.
- Design 4: `{"n_seeds": 200, "seed_generator_seed": 20260922}`.
- The primary MAPE is uncapped and uses equal (calibrate default) target weights and scale `s = max(|target|, 1) (microcosm.calibrate.default_target_loss_scales)`. The secondary capped MAPE caps each target's absolute relative error at 10. The unit floor also defines the error for zero-valued targets; this is a scaled relative error, not an undefined division by zero.
- Certainty threshold for the exact-k draw: 0.95. L0 selection's weight-ratio bound is disabled when its setting is null. The post-stratification bound is relative to the sample's starting equal weights; the exact-k refit bound is relative to its normalized inclusion-corrected starting weights.
<!-- END generated:settings -->

## Targets and the no-poverty/SPM check

<!-- BEGIN generated:targets -->
- Registry compiled from the labelled feed: 32,867 specs at all geographies; 505 national (metadata.ledger_geography_level == 'country' or metadata.geography_scope == 'national').
- Labelled filter keys that the recorded checkout's guard refuses: 302 specs; 302 accepted as verified restatements, 0 excluded.
- Materialized on the file: 505 national targets (0 not materialized); used after the poverty/SPM exclusion: 504. Families used: bea 2, cbo 5, census_population 18, cms_medicaid 3, cms_medicare 1, federal_reserve 1, hhs_acf_tanf 1, irs_soi 451, jct 11, ssa 9, usda_snap 2.
- Headline subset: 32 targets (population by age band (Census PEP): 18; wages and salaries (BEA NIPA): 1; wages and salaries (CBO): 1; SNAP benefits (USDA): 1; SNAP average monthly households (USDA): 1; Medicaid enrollment (CMS, Dec 2024): 1; SSI payments (SSA): 1; SSI recipients by age (SSA): 3; EITC amount and returns (IRS SOI TY2024 filing season): 2; AGI and return count (IRS SOI Table 1.1): 2; AGI (CBO): 1). National AGI-by-band rows in the compiled set: 0.
- Excluded under the no-poverty/SPM rule: 1 (`hhs_acf_liheap.fy2024.national_profile.state_programs.households_served` (base_variable=spm_unit_energy_subsidy)). After exclusion, targets naming poverty or SPM: 0; measuring a poverty or SPM variable: 0.
- Batch invariance: 400 households re-materialized across a batch boundary; 0 of their values differ (max absolute difference 0).
<!-- END generated:targets -->

Poverty and SPM measures are excluded from fitting and selection; the audit
above covers names and source/filter variables. Structural SPM-unit counts
appear only in PolicyBench's pre-existing eligibility rule. They are not a
poverty measure. The household descriptors are post-selection diagnostics.

## Frozen sample verification

The reference sampling and split implementations were read at
`policybench/scenarios.py:1020` (eligibility),
`policybench/scenarios.py:1044` (PPS draw), and
`policybench/scenarios.py:1498` (public/private split) in the read-only
PolicyBench checkout.

<!-- BEGIN generated:frozen_draw -->
IDs recovered through `scenarios.csv scenario_json.metadata.household_id (PolicyBench writes the source household id there)`. The frozen public set came from 125 requested draws with seed 42, private fraction 0.2, split seed 1042, leaving 100 public scenarios. Reproduction: policybench/scenarios.py draw re-run here: eligibility filter, numpy.random.default_rng(seed).choice(ids, requested, replace=False, p=w/sum(w)), then split_scenarios' sha256 id-hash public split.

| filing-status basis for eligibility | eligible households | draw reproduced exactly (ids, order) | overlap with frozen 100 |
|---|---|---|---|
| filing_status_as_shipped | 63,128 | True | 100 |
| filing_status_measured | 63,128 | True | 100 |

Row checks of the file against `scenarios.csv` for the 100 frozen households: state 100, filing status 100, adults 100, children 100, employment income within $1 100, total income within $1 100. Exact equality counts: employment income 81, total income 55. Largest requested-draw-count × p over eligible households: 0.3851.

Weight correction: sum of eligible design weights / 100 (self-weighting approximation). microcosm.calibrate.exact_k supplies inclusion probabilities only for its own Sampford draw (select_exact_k); it has no function for numpy Generator.choice(replace=False, p=...), so the self-weighting approximation is used and checked empirically against the seed draws (sampling_distribution.json self_weighting_check)
<!-- END generated:frozen_draw -->

## Exact-k selection

<!-- BEGIN generated:exact_k -->
| target set | budget measure | settled l0_lambda | measure at settle | open-probability mass | gates open at eval (pi > threshold) | gates with pi >= 0.95 | certainties drawn | boundary draws | boundary draws with pi <= threshold | probes (search stopped on) | draw failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| national_all | nonzero_count | 0.000487 | 99 | 529.7 | 99 | 6 | 6 | 94 | 90 | 7 (acceptable_within_tolerance) | no |
| headline | nonzero_count | 0.0003652 | 102 | 524.4 | 102 | 0 | 0 | 100 | 97 | 7 (acceptable_within_tolerance) | no |

Budget-search probes (penalty, budget measure):

| target set | l0_lambda | measure |
|---|---|---|
| national_all | 0.001 | 34 |
| national_all | 1e-05 | 1722 |
| national_all | 0.0001 | 409 |
| national_all | 0.0003162 | 155 |
| national_all | 0.0005623 | 89 |
| national_all | 0.0004217 | 121 |
| national_all | 0.000487 | 99 |
| headline | 0.001 | 23 |
| headline | 1e-05 | 2578 |
| headline | 0.0001 | 566 |
| headline | 0.0003162 | 134 |
| headline | 0.0005623 | 60 |
| headline | 0.0004217 | 86 |
| headline | 0.0003652 | 102 |

national_all: final draw count 100; positive refit weights 100; boundary draw needed: True. The boundary draw selected 90 households whose L0 gates were closed at deterministic evaluation. The final support is therefore a maintained probability-based boundary draw, rather than simply the households with open L0 gates.

headline: final draw count 100; positive refit weights 100; boundary draw needed: True. The boundary draw selected 97 households whose L0 gates were closed at deterministic evaluation. The final support is therefore a maintained probability-based boundary draw, rather than simply the households with open L0 gates.
<!-- END generated:exact_k -->

Deterministically open L0 gates and the final exact-k support are distinct:
the maintained boundary draw can select positive-probability records whose
L0 gates are closed at evaluation. No households are hand-selected to force
the requested count. A failed maintained selection or draw remains a failed
result, with its available search receipts retained.

## Fit against shared targets

<!-- BEGIN generated:fit -->
national_all (504 targets):

| design | MAPE (uncapped) | capped MAPE (secondary) | within 10% | within 25% | Kish ESS | max/mean weight | top-10 mass |
|---|---|---|---|---|---|---|---|
| Full file (75,112 design weights) | 397010718.4% | 55.0% | 32.9% | 53.8% | 1,760.2 | 178.0 | 1.9% |
| 1. PPS sample, self-weighted | 1863675480.0% | 82.2% | 8.9% | 18.7% | 100.0 | 1.0 | 10.0% |
| 1b. Same sample, full-file mass sensitivity | 2422100169.0% | 84.3% | 12.9% | 22.6% | 100.0 | 1.0 | 10.0% |
| 2. PPS sample, post-stratified | 2116681946.4% | 65.0% | 28.8% | 35.5% | 32.0 | 6.5 | 47.2% |
| 3. Exact-k 100, refit | 58.7% | 58.7% | 23.4% | 33.1% | 12.4 | 21.0 | 73.1% |
| 3a. Exact-k 100, HT weights before refit | 105.0% | 101.0% | 10.5% | 17.1% | 11.9 | 16.5 | 78.5% |

headline (32 targets):

| design | MAPE (uncapped) | capped MAPE (secondary) | within 10% | within 25% | Kish ESS | max/mean weight | top-10 mass |
|---|---|---|---|---|---|---|---|
| Full file (75,112 design weights) | 6.7% | 6.7% | 84.4% | 93.8% | 1,760.2 | 178.0 | 1.9% |
| 1. PPS sample, self-weighted | 38.6% | 38.6% | 6.2% | 28.1% | 100.0 | 1.0 | 10.0% |
| 1b. Same sample, full-file mass sensitivity | 30.9% | 30.9% | 18.8% | 43.8% | 100.0 | 1.0 | 10.0% |
| 2. PPS sample, post-stratified | 4.1% | 4.1% | 90.6% | 96.9% | 49.3 | 4.8 | 35.6% |
| 3. Exact-k 100, refit | 17.9% | 17.9% | 71.9% | 78.1% | 24.5 | 8.4 | 55.4% |
| 3a. Exact-k 100, HT weights before refit | 47.3% | 47.3% | 9.4% | 28.1% | 16.5 | 11.8 | 68.6% |
<!-- END generated:fit -->

<!-- BEGIN generated:worst -->
| target set | design | worst five targets (signed scaled relative error) |
|---|---|---|
| national_all | 1. sampled, self-weighted | `irs_soi.ty2023.table_2_5.eitc_by_agi_children.three_or_more_qualifying_children.30k_to_35k.eitc_total` 939169118978%; `irs_soi.ty2023.table_2_5.eitc_by_agi_children.three_or_more_qualifying_children.30k_to_35k.eitc_returns` 123234384%; `irs_soi.ty2023.table_2_5.eitc_by_agi_children.two_qualifying_children.1k_to_2k.eitc_total` 40882%; `irs_soi.ty2023.table_2_5.eitc_by_agi_children.two_qualifying_children.1k_to_2k.eitc_returns` 10246%; `irs_soi.ty2023.table_2_5.eitc_by_agi_children.two_qualifying_children.15k_to_16k.eitc_total` 935% |
| national_all | 2. sampled, post-stratified | `irs_soi.ty2023.table_2_5.eitc_by_agi_children.three_or_more_qualifying_children.30k_to_35k.eitc_total` 1066667698354%; `irs_soi.ty2023.table_2_5.eitc_by_agi_children.three_or_more_qualifying_children.30k_to_35k.eitc_returns` 139964288%; `irs_soi.ty2023.table_2_5.eitc_by_agi_children.two_qualifying_children.1k_to_2k.eitc_total` 7721%; `irs_soi.ty2023.table_2_5.eitc_by_agi_children.two_qualifying_children.1k_to_2k.eitc_returns` 1874%; `irs_soi.ty2023.table_2_5.eitc_by_agi_children.two_qualifying_children.50k_plus.eitc_returns` 130% |
| national_all | 3. exact-k, refit | `irs_soi.ty2023.table_1_4.all.capital_gain_distributions_returns` 200%; `irs_soi.ty2022.historic_table_2.us.500k_to_1m.taxable_interest_returns` 165%; `jct.tax_expenditures.cy2024.health_savings_account_deduction.revenue_loss` 155%; `irs_soi.ty2023.table_1_4.all.net_capital_gains_returns` 115%; `irs_soi.ty2023.congressional_district_2022.all_returns.us.tax_exempt_interest_returns` 102% |
| headline | 1. sampled, self-weighted | `ssa_ssi_monthly.month2024_12.ssi_federal_payment_recipients.by_age.under_18.recipient_count` -100%; `census_pep.cy2024.national_resident_population_age.30_to_34.population` -69%; `census_pep.cy2024.national_resident_population_age.0_to_4.population` -60%; `ssa_ssi_monthly.month2024_12.ssi_federal_payment_recipients.by_age.age_18_to_64.recipient_count` 58%; `census_pep.cy2024.national_resident_population_age.15_to_19.population` -56% |
| headline | 2. sampled, post-stratified | `ssa_ssi_monthly.month2024_12.ssi_federal_payment_recipients.by_age.under_18.recipient_count` -100%; `cbo.revenue_projection.ty2024.income_by_source.wages_and_salaries.projected_amount` 14%; `irs_soi.ty2023.table_1_1.all.return_count` 12%; `cms_medicaid.month2024_12.state_enrollment.us.total_medicaid_enrollment` -0%; `census_pep.cy2024.national_resident_population_age.50_to_54.population` 0% |
| headline | 3. exact-k, refit | `ssa_ssi_monthly.month2024_12.ssi_federal_payment_recipients.by_age.age_18_to_64.recipient_count` -100%; `ssa_ssi_monthly.month2024_12.ssi_federal_payment_recipients.by_age.age_65_or_older.recipient_count` -86%; `ssa_supplement.cy2024.oasdi_ssi_payments.ssi_payments.payment_amount` -81%; `census_pep.cy2024.national_resident_population_age.75_to_79.population` -80%; `census_pep.cy2024.national_resident_population_age.25_to_29.population` -71% |
<!-- END generated:worst -->

## Sampling distribution

<!-- BEGIN generated:sampling -->
200 of 200 requested fresh seeds completed (time budget 60 minutes; stopped on budget: False); seeds from numpy.random.default_rng(20260922).integers(0, 2**31 - 1), skipping duplicates and PolicyBench's seed 42; eligible pool 63,128 households.

Seed-count / stopping explanation: all requested seeds completed. Eligibility: policybench/scenarios.py::_eligible_households: one tax unit, one SPM unit, one family, >=1 adult (age >= 18), first-person filing status SINGLE/JOINT/HEAD_OF_HOUSEHOLD. Procedure: PolicyBench's own procedure per seed: PPS without replacement of 125 eligible households, then the id-hash public split (split_seed 1042), keeping the public 100; design 1 = self-weighted; design 2 = post-stratified with the sample-refit settings

| target set | design | uncapped MAPE p5 / p50 / p95 | capped MAPE p5 / p50 / p95 | within 10% p5 / p50 / p95 |
|---|---|---|---|---|
| national_all | 1. self-weighted | 77.0% / 100.2% / 1893769072.5% | 74.4% / 83.8% / 94.7% | 8.3% / 10.9% / 13.9% |
| national_all | 2. post-stratified | 57.0% / 78.3% / 2288701470.3% | 57.0% / 62.7% / 69.2% | 25.8% / 28.6% / 31.9% |
| headline | 1. self-weighted | 31.3% / 37.9% / 45.7% | 31.3% / 37.9% / 45.7% | 3.1% / 12.5% / 21.9% |
| headline | 2. post-stratified | 1.1% / 4.9% / 10.8% | 1.1% / 4.9% / 10.8% | 81.2% / 90.6% / 96.9% |

| target set | set | uncapped MAPE | MAPE: share of seeds at or below | within 10% | within-10%: share of seeds at or below |
|---|---|---|---|---|---|
| national_all | frozen draw, design 1 | 1863675480.0% | 94% | 8.9% | 12% |
| national_all | frozen draw, design 2 | 2116681946.4% | 94% | 28.8% | 58% |
| national_all | exact-k refit vs design 2 seeds | 58.7% | 10% | 23.4% | 0% |
| headline | frozen draw, design 1 | 38.6% | 57% | 6.2% | 26% |
| headline | frozen draw, design 2 | 4.1% | 30% | 90.6% | 70% |
| headline | exact-k refit vs design 2 seeds | 17.9% | 100% | 71.9% | 0% |

Self-weighting check over the 200 seed draws (public inclusions by design-weight decile of the eligible pool):

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| observed | 0 | 1 | 6 | 18 | 28 | 40 | 68 | 94 | 214 | 19531 |
| expected (100 w / sum w per draw) | 0.0 | 1.6 | 9.9 | 17.7 | 27.4 | 40.8 | 61.3 | 100.3 | 198.3 | 19,542.7 |
<!-- END generated:sampling -->

Lower MAPE is better; a lower MAPE percentile means fewer random samples
achieved equally low error. Higher within-tolerance share is better, so its
percentile has the opposite interpretation. This distribution varies the PPS
sample seed; it is not an uncertainty interval for exact-k optimization.

## Weight concentration

<!-- BEGIN generated:concentration -->
| target set | design | max/mean | top-10 share | Kish ESS | >5x own median | >10x own median | >5x equal share | >10x equal share |
|---|---|---|---|---|---|---|---|---|
| national_all | full file | 178.0 | 1.9% | 1,760.2 | 12263 | 7410 | n/a | n/a |
| national_all | 1. sampled, self-weighted | 1.0 | 10.0% | 100.0 | 0 | 0 | 0 | 0 |
| national_all | 2. sampled, post-stratified | 6.5 | 47.2% | 32.0 | 15 | 5 | 8 | 0 |
| national_all | 3a. exact-k HT before refit | 16.5 | 78.5% | 11.9 | 27 | 23 | 7 | 3 |
| national_all | 3. exact-k, refit | 21.0 | 73.1% | 12.4 | 16 | 12 | 4 | 1 |
| headline | 2. sampled, post-stratified | 4.8 | 35.6% | 49.3 | 9 | 0 | 6 | 0 |
| headline | 3a. exact-k HT before refit | 11.8 | 68.6% | 16.5 | 29 | 23 | 7 | 2 |
| headline | 3. exact-k, refit | 8.4 | 55.4% | 24.5 | 24 | 14 | 5 | 0 |
<!-- END generated:concentration -->

Own-median thresholds use each design's positive output weights. Equal-share
thresholds use eligible-pool mass for sampled designs and full-file mass for
exact-k, divided by support size.

## Weighted composition

<!-- BEGIN generated:composition -->
| weighted share | population | sampled (1) | sampled (2, national) | sampled (2, headline) | exact-k (national) | exact-k (headline) |
|---|---|---|---|---|---|---|
| household_size 1 | 39.2% | 58.0% | 49.1% | 49.5% | 36.4% | 37.7% |
| household_size 2 | 35.2% | 24.0% | 35.3% | 31.6% | 32.4% | 31.2% |
| household_size 3 | 12.7% | 9.0% | 8.8% | 9.5% | 9.1% | 17.2% |
| household_size 4+ | 13.0% | 9.0% | 6.8% | 9.4% | 22.0% | 13.9% |
| children 0 | 74.6% | 82.0% | 85.0% | 79.6% | 75.5% | 71.7% |
| children 1 | 13.0% | 6.0% | 4.8% | 10.6% | 2.5% | 14.9% |
| children 2+ | 12.4% | 12.0% | 10.2% | 9.8% | 22.0% | 13.4% |
| head_age <30 | 16.6% | 14.0% | 18.0% | 18.5% | 16.8% | 6.7% |
| head_age 30-49 | 33.9% | 31.0% | 37.0% | 29.0% | 39.0% | 48.8% |
| head_age 50-64 | 23.3% | 22.0% | 22.1% | 24.7% | 8.9% | 20.9% |
| head_age 65+ | 26.1% | 33.0% | 22.9% | 27.8% | 35.3% | 23.6% |
| employment_income_quintile Q1 | 27.7% | 36.0% | 27.3% | 31.2% | 20.9% | 30.6% |
| employment_income_quintile Q2 | 12.3% | 15.0% | 11.2% | 15.3% | 2.3% | 14.9% |
| employment_income_quintile Q3 | 20.0% | 18.0% | 14.1% | 16.5% | 26.0% | 13.4% |
| employment_income_quintile Q4 | 20.0% | 19.0% | 28.2% | 20.7% | 24.9% | 21.3% |
| employment_income_quintile Q5 | 20.0% | 12.0% | 19.1% | 16.3% | 25.9% | 19.8% |
| receipt receives_snap | 14.4% | 16.0% | 11.7% | 12.4% | 13.7% | 15.5% |
| receipt receives_medicaid | 23.3% | 17.0% | 12.1% | 21.4% | 22.4% | 25.7% |
| receipt receives_ssi | 3.5% | 5.0% | 3.0% | 3.5% | 0.3% | 0.9% |
| receipt eitc_positive | 14.6% | 11.0% | 4.1% | 13.3% | 3.7% | 16.9% |
| receipt social_security_positive | 33.9% | 43.0% | 29.2% | 41.2% | 15.5% | 44.1% |
| distinct states | 51 | 34 | 34 | 34 | 40 | 39 |

Employment-income quintile cut points (population, weighted): 0, 19,993, 53,000, 103,219.

Ties stay in the same income band, so the full population's quintile shares need not be equal.
<!-- END generated:composition -->

## Household column definitions

<!-- BEGIN generated:descriptors -->
| output column | base variable | materialized target column | household value |
|---|---|---|---|
| employment_income | employment_income | cbo.revenue_projection.ty2024.income_by_source.wages_and_salaries.projected_amount | household value of the target's measure |
| agi_or_total_income | adjusted_gross_income | cbo.revenue_projection.ty2024.income_by_source.adjusted_gross_income.projected_amount | household value of the target's measure |
| receives_snap | snap | usda_snap.fy2024.national_benefits.national_total.total_benefits | measure > 0 |
| receives_medicaid | medicaid_enrolled | cms_medicaid.month2024_12.state_enrollment.us.total_medicaid_enrollment | measure > 0 |
| receives_ssi | ssi | ssa_supplement.cy2024.oasdi_ssi_payments.ssi_payments.payment_amount | measure > 0 |
| eitc_positive | eitc | irs_soi.ty2024.filing_season_week47.eitc_all_returns.earned_income_credit.total_earned_income_credit_amount | measure > 0 |
| social_security_positive | social_security | ssa_supplement.cy2024.oasdi_ssi_payments.social_security_benefits.payment_amount | measure > 0 |

Structural descriptors: household_size: count of person rows in the household; children: members under 18 (household_size - adults); head_age: age of the is_household_head member (oldest member if none flagged); employment_income: cbo.revenue_projection.ty2024.income_by_source.wages_and_salaries.projected_amount; weights: shares are weighted by each design's own weights. The `agi_or_total_income` column uses adjusted gross income, aggregated to the household. It is distinct from the frozen scenario's sum of monetary income fields used in the reproduction check.

Receipt flags describe positive modeled amounts or enrollment in this measurement frame. The head-age retirement proxy does not establish employment or retirement status.
<!-- END generated:descriptors -->

## Overlap

<!-- BEGIN generated:overlap -->
| overlap | households |
|---|---|
| distinct households across seed draws | 3326 |
| seed draws | 200 |
| frozen 100 in any seed draw | 93 |
| exact k national all in frozen 100 | 0 |
| exact k national all in any seed draw | 18 |
| exact k national all policybench eligible | 76 |
| exact k headline in frozen 100 | 0 |
| exact k headline in any seed draw | 19 |
| exact k headline policybench eligible | 83 |
| exact k national all vs headline shared | 0 |
<!-- END generated:overlap -->

## Runtime and memory

<!-- BEGIN generated:runtime -->
| step | seconds | peak RSS GB |
|---|---|---|
| engine facts child | 40.9 | 9.5 |
| target compile | 12.5 | 4.0 |
| materialize all batches | 2,114.1 | 16.3 |
| batch invariance child | 178.9 | 6.1 |
| national all design2 post stratified | 2.0 | 3.5 |
| national all design3 l0 selection | 1,657.2 | 3.9 |
| national all design3 refit | 0.1 | 3.9 |
| headline design2 post stratified | 0.1 | 3.9 |
| headline design3 l0 selection | 229.6 | 4.5 |
| headline design3 refit | 0.2 | 4.5 |
| sampling distribution (400 calibrations) | 63 | 4.5 |
| whole run | 4,332 | 4.5 (parent high-water) |

Measurement definitions: peak_rss_gb: sampled every 0.25 s: this process plus any live child (psutil RSS sum); child_peak_rss_gb: the child's own getrusage high-water mark.

Main calibration timings are retained in [runtime.json](runtime.json); each sampling-refit timing and memory reading is in [sampling_distribution.json](sampling_distribution.json). Parent high-water RSS is distinct from simultaneous parent-plus-child RSS.
<!-- END generated:runtime -->

Final reporting corrections are documented in [source_revision.json](source_revision.json)
and [agi_scope_audit.json](agi_scope_audit.json). They correct the export description
and distinguish unbounded AGI totals from actual AGI bands. The original producer
hash and successful run validation are retained; no target membership, household
selection, calibration weight, or fit result changed. Completed checks are recorded
in [verification.json](verification.json) and [validation.json](validation.json).

## What was not measured

No PolicyBench harness or language model was run. Aggregate target fit and
household coverage do not establish model-ranking stability, policy-reform
accuracy, out-of-sample performance, or release readiness. The self-weighting
approximation does not recover exact first-order inclusion probabilities for
the PPS-without-replacement draw followed by its fixed public split. Current
engine-materialized descriptors are not a claim to reproduce every frozen
engine output. Earlier unlogged failed attempts are not evidence for this
record's calibration outcomes.
