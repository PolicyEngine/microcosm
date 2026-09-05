# #834 / #789 receipts — childcare and Tax-Free Childcare port, Chronicle re-pin, bus targets

Plan: `repos/uk-834-childcare-tfc-port-plan.md` (canonical, María's rulings A0–A20). Licensed
evidence lives under `data/ukds/acceptance/834-childcare-tfc/` (SDC-safe aggregates, minimum
cell count 3); this file carries the committed, digest-pinned receipts the PR's signed entries
anchor to.

## Part A — Chronicle feed re-pin (increment I0)

Feed rebuilt from Chronicle `PolicyEngine/chronicle` main `6fb700e` (2026-09-04; carries #220
childcare, #231 the recovered #202, #239 the UC crosses / Child Benefit / census composition)
with `chronicle build-bundle --suite uk` → `build-consumer-artifact`:

| | value |
|---|---|
| `consumer_facts.jsonl` sha256 | `6ae49d7d7ab297df25a0b9bfe2d6776827c672d284fbb360957fe8337089549f` |
| `manifest.json` sha256 | `dcda51d6496aea67f768a284e7955c7520e7c8b91e2bed3569f247567b7153f0` |
| rows / schema | 128 717 / `policyengine_ledger.consumer_artifact.v2` |
| previous pin | `33ca98a` / `4395a4e7…`, 108 112 rows (local census); the national surface had been generated on `226358e7…`, 107 550 rows |

### Why the re-pin had to repair the contract (plan §5c)

On the previously pinned feed the committed national surface did not compile: 19 references
unsupported (`compile_uk_target_registry`), which `tools/calibrate_uk_national_dataset.py`
refuses. On the rebuilt feed, 12 were unsupported before the I0 corrections. After I0:

| surface | before (committed) | after I0 on `6ae49d7d…` |
|---|---|---|
| national membership | active 408, multi_fact 1, no_fact 7, signed_excluded 1 | **active 408**, no_fact 7, signed_excluded 2 |
| national runtime compile | 389 compiled / 19 unsupported on the pinned feed | **408 compiled / 0 unsupported** |
| local membership | active 19 618, no_fact_at_or_before_period 314, no_fact_for_area 1 586, compile_error 1 | **active 19 932**, no_fact_for_area 1 586, compile_error 1 (the signed `E14001416` SPI mean) |

Corrections (each a declaration, verified by the regeneration test and the runtime compile):

- UC composition rows (`dwp.uc.households_{single,couple}_{no,with}_children`,
  `_children_{1..5_or_more}`): exact dimension-key pins (`dimensions: [family_type]` /
  `[number_of_children]`), so the #239 child-entitlement and payment-indicator crosses no
  longer match; the generator's sum inference no longer treats a `dimensions` list as a sum.
- Caseload `dwp.uc.households`: `source_measure_id: total_units` +
  `groupby_dimension: dwp.uc_deductions_month` → binds DWP UC deductions statistics Table 1
  only; calendar-year average over April–December 2025 = **6 758 889** (the #735 value),
  1.45% below the Stat-Xplore family-type sum 6 858 287 (chronicle#247 asks for the
  cube's Total row).
- CGT totals: `groupby_dimension: hmrc.cgt_table1_line` (Table 5's UK rows carried the same
  concepts and values, £127.316bn gains at TY2024).
- One temporal basis for `dwp_universal_credit`: every row is a calendar-year average; the 100
  payment-band rows move from the May-2025 snapshot to the 2025 mean; the Scotland
  youngest-child row is re-bound at benefit-unit grain (`youngest_child_age < 1`).
- `ons.population.scotland_households_3plus_children` signed-excluded (its selector reaches
  person-level mid-year population rows; #736).
- Local `ons.rent.private_rent`: the stale deferral `private_rent_pipr_after_target_period`
  retired; 314 rows bind the calendar-year average of the 2025 PIPR months (12 members each).
- Feed guard: the calibration runner refuses a feed whose facts digest differs from the
  committed pin (`--allow-unpinned-feed` is a recorded diagnostic override); a hermetic test
  regenerates the surfaces from the pinned identity and skips only when the feed is absent.

## Part B — Licensed before-receipts (L0, L1)

All at design weights on full-scale spines (52 846 households, 113 626 persons, 61 234 benefit
units), engine policyengine-uk 2.94.0 for the receipts.

### L0 — spine-n (main `47c74225` + #842 + #850, built at 2.92.1), read at 2.94.0

`l0_spine_n_childcare_receipt.json`. The routed share reads its engine default (1.0): the column
is absent from the spine.

| row (model / fact) | 2024 basis (FY2024-25 / Jan-2024; extended Jan-2025 ages 2–4) | 2025 basis (FY2025-26 / Jan-2025) |
|---|---|---|
| TFC spending (£0.6322bn / £0.5998bn) | 1.073× | 1.174× |
| TFC children (1 085 020 / 1 151 515) | 1.020× | 0.951× |
| Working-parent children ages 2–4, England (621 482) | 1.283× | 1.261× |
| Early learning 2-year-olds, England (115 852 / 95 031) | 0.596× | 0.738× |
| Universal-only children, England (416 537 / 396 965) | 1.104× | 1.177× |

### L1 — engine-only twin (branch `eb8abcdd`: floor ≥2.93, lock 2.94.0; no new column)

`l1-engine-twin/l1-engine.h5`, 28 stages, spine battery 15/15. Payload against spine-n
(`l1_vs_spine_n_payload.json`): benunit and time_period tables byte-equal; person differs only
in `student_loan_balance` (26 rows); household differs in 37 columns — the WAS, LCFS, ETB,
property and consumption outputs of the QRF stages whose predictors include
`household_net_income` (which carries `tax_free_childcare`, moved by 2.92.2's gross-rate
arithmetic). No income, benefit or childcare input column moves. The childcare receipt on L1
(`l1_engine_childcare_receipt.json`) is identical to L0.

## Part C — column twin (L2), release round trip (L3), re-fit (L5), composition (L4)

### L4-pre — the I0 surface calibrated, before any childcare row exists

Purpose: attribute the contract repair separately from the childcare targets. Input: the L1
engine-only twin (2.94.0, no routed share); surface: the I0 commit's 408 references on feed
`6ae49d7d…`; solve: 1 500 epochs, `family_equal`, the packaged measure exclusions (364 targets in
the matrix); driver tree: a scratch worktree at the I0 commit (`l4pre-i0/`).

| | value |
|---|---|
| initial → final loss | 0.374 → 0.0259 |
| within 10% / within 25% | 330 / 357 of 364 |
| ESS / max weight ratio / top-1% share | 5 733 / 10.0 / 0.189 |
| caseload `dwp.uc.households` (6 758 889) | −25.4% → **−12.7%** |
| 84 payment-band rows (2025 averages) | median 0.1%, max 3.6% |
| `hmrc.cgt.gains_total` (£127.3bn, TY2024) / `taxpayers_total` (584 000) | −50.1% → −0.6% / −38.9% → −0.2% |
| **`obr.capital_gains_tax` (£21.8bn, 2025-26)** | −27.6% → **+43.7%** |
| Scotland UC youngest child < 1 (14 333) | +8.5% → −0.2% |

The terminal battery blocks on seven `uk_target_fit` cells: the four UC with-children cells
(single_with_children −34.0%, children_1 −25.8%, children_2 −39.7%, children_5_or_more −28.4%),
two state-pension SPI bands (+33.2%, +26.7%) — all signed on the publication stack's
`target_fit_reviewed_exclusions.json`, which main does not carry — and one new cell:
**OBR CGT receipts +43.7%.** The re-pin moved the two CGT totals from tax year 2023-24 (the
latest fact on the previous feed) to 2024-25 (HMRC's 2026 release, £127.3bn of gains against
£65.9bn a year earlier: the realisations brought forward ahead of the October 2024 rate rise). The
solve fits the 2024-25 gains to 0.6% and, at 2025-26 policy, the frame's CGT liability then
overshoots OBR's 2025-26 receipts line by 44% (it was −25% under with the 2023-24 gains). Neither
vintage of gains reproduces the OBR line; the choice is the CGT lane's (#467 / #725 / #552, #736
"CGT 2025 level") and is put to María as A21.

**Defect found and fixed here:** I0 had re-bound `dwp.uc.scotland_households_child_under_1` at
benefit-unit grain with a `region` filter; the calibration's measure provider has no categorical
household-to-benefit-unit broadcast (`compute_uk_measure_input` broadcasts categoricals to
persons only), so the run refused with `provider does not know benunit.region`. The row keeps
its household-grain binding (`household_conditions`, the pre-I0 form) with the tightened
selector and the family's calendar-year average; validated by this run.

### L2 — column twin (I1 + I2 tree on `14c71008`, built at 2.94.0)

Tree: the I0 commit plus the uncommitted Lane A/B increments (`TREE_HEAD`: `14c71008`, 46 files
changed); the same licensed inputs and the same `build_twin.sh` invocation as L1. Build: exit 0,
28 stages, rows 113 626 / 61 234 / 52 846, all 15 spine gates `passed`
(`l2-column-twin/l2-column.spine_gates.json`), engine 2.94.0. The new column is present from
`frs_person_draws` onward with nonzero share 1.0 (every person carries the build-year value; the
degenerate-column register carries its signed entry).

**Payload against L1 (`compare_uk_h5_payload.py`, `l2_vs_l1_payload.json`)** — the only
between-twin difference is the new column and its consequences:

| table | columns only in L2 | columns moved | rows moved |
|---|---|---|---|
| benunit | — | 0 / 11 | 0 |
| time_period | — | 0 | 0 |
| person | `tax_free_childcare_spend_routed_share` | 1 (`student_loan_balance`) | 64 (the `top_up_to_stock` selection; no cash column moves) |
| household | — | 38 / 66 | see below |

The 38 household columns are exactly the imputed surfaces (WAS wealth, LCFS consumption, ETB
services) and their deterministic derivatives; no FRS-carried column moves. Root cause, proven
at engine level (`l2_vs_l1_net_income_decomposition.log`): the routed share lowers
`tax_free_childcare` for 1 414 households (unweighted £1.228m → £0.797m; the £2 000 cap binds for
some, hence 0.65 rather than 0.593), `hbai_household_net_income` moves for exactly those 1 414
and for no others, and the three imputation stages read `household_net_income` /
`hbai_household_net_income` as predictors. Market income, income tax, National Insurance,
council tax, UC, child benefit, housing benefit, pension credit, student-loan repayments and CGT
are byte-equal between the twins.

| stage | households re-drawn | of which TFC households | mechanism |
|---|---|---|---|
| `was_wealth` | 654 | 536 | predictor move → correlated-rank re-draw (the 686 E5 class); the remainder are threshold neighbours (owner share 0.89 vs 0.63 in the frame) |
| `lcfs_consumption` | 1 196 | 946 | as above, plus WAS-drawn predictors |
| `etb_services` | 1 027 | 768 | as above |
| `regional_property_uprating` | 31 084 (every owner) | — | deterministic: the regional owner mean re-solves once any owner draw moves; median relative move 3.2e-4 |
| LCFS energy IPF (`electricity`, `gas`, folded `domestic_energy_consumption`) | 52 586 | — | deterministic: the weighted IPF on income/tenure/accommodation/region margins re-solves for the whole frame; median relative move 6e-4 – 1.1e-3 |

Aggregate mass per column (`l2_vs_l1_household_magnitudes.json`, unweighted column totals):
largest moves `bus_fare_spending` +0.89%, `education_consumption` +0.72%, `diesel_spending`
+0.54%, `rail_usage`/`rail_subsidy_spending` −0.37%; every other column within ±0.2%, the energy
trio and the property pair within ±1.2e-4. On the release payload these feed back into
`household_tax` (VAT for 1 342 households, fuel duty for 453, and a sub-£1 energy-tax ripple on
40 491) and `household_benefits` (the 1 414 TFC households plus the in-kind ETB columns); net
household income moves by −£0.27m unweighted across the frame.

Classification for the swap register: **net-new person column + engine-predictor re-draw of the
imputed household surfaces**, the same class the L1 twin showed against spine-n under the engine
bump, with a smaller footprint (1 414 root households against a frame-wide net-income shift).
No cash column, weight column or benefit-unit column moves.

### L3 — release round trip (L2 read by policyengine-uk 2.94.0)

`l0_childcare_receipt.py` on `l2-column.h5` (`l3_l2_column_childcare_receipt.json`): the engine
reads `tax_free_childcare_spend_routed_share` at 0.593 (float32) for every person at both 2024
and 2025 — the 2024-04-01 contract value, as the build-year lock requires. Against the same five
facts as L0/L1 (model ÷ target, weights as built, before any calibration):

| target (basis) | L0 = L1 | L3 |
|---|---|---|
| TFC top-up £ (FY2024-25 / FY2025-26) | 1.073 / 1.174 | **0.698 / 0.772** |
| TFC children with used accounts | 1.020 / 0.951 | 1.020 / 0.951 |
| extended entitlement children 2–4, England (Jan 2024 / Jan 2025) | 1.283 / 1.261 | 1.283 / 1.261 |
| targeted 2-year-olds, England | 0.596 / 0.738 | 0.596 / 0.738 |
| universal-only 3–4s, England | 1.104 / 1.177 | 1.104 / 1.177 |

Only the spend row moves, by the routed share (×0.65 on the top-up, the cap binding for part of
the population) — the four count rows are identical to L0 to three decimals, which is the
"routed share touches spend, not eligibility or take-up" property the issue describes. The L0
spend ratio of 1.07 was the un-routed accident that #472 corrected; the re-fit at L5 works from
the routed 0.70.

### L5 — re-fit of the four take-up rates (A6: 2024 base, hours frozen)

`tools/fit_uk_childcare_takeup.py` on `l2-column.h5` (sha `2c69544b…`), feed `6ae49d7d…`,
`--target-period 2024`, `--vintage-override dfe.funded_childcare.working_parent_children_2_to_4=2025`
(the Jan-2024 2-year-old cell is suppressed; Part E), seed 42, engine 2.94.0, routed share 0.593
injected from the contract, hours distribution frozen (15.019 / 4.972 / [0, 30]). Targets as
compiled: £632.2m, 1 085 020, 621 482, 115 852, 416 537.

**First run (`fit_2024.json`, defaults: L-BFGS-B, maxiter 5, eps 1e-2): did not converge**
(`success: false`, loss 0.302). TFC 0.88 → 0.959, extended 0.812 → 0.559, universal 0.563 →
0.447, targeted **0.597 → 0.597 (never moved)**; achieved ÷ target: TFC spend 0.785, TFC
children 1.104, extended 0.943, targeted 0.508, universal 0.984.

Diagnosis (`l5-refit/runner_probe.log`, the fitter's own runner on the same H5):

| probe | result |
|---|---|
| stored build draws, no injection | targeted 69 050 (0.596×) from **94 flagged sample children**; 1 290 two-year-olds in the frame, 94 with a positive entitlement |
| runner at the frozen rates, seed 42 / seed 0 | targeted 54 889 / 62 521 (0.47× / 0.54×) — the realisation noise of a ~157-child eligible base carrying £-weights of hundreds each |
| targeted rate 0.597 → 0.607 (the optimizer's step) | **identical count** (54 889): a 0.01 step flips no unit at this base, so the forward difference reads zero and L-BFGS-B never moves the parameter |
| targeted rate 0.697 / 0.90 | 66 101 / 88 036 — the row responds, but even at 0.90 it reaches 0.76× |

So the targeted row is (i) below the draw resolution of a 1e-2 finite-difference step and (ii)
near its reachable ceiling at design weights: the eligible base — England, a qualifying benefit
or UC/TC criterion, **and not extended-eligible** (the engine's mutual exclusion; extended now
covers working-parent 2-year-olds) — is ~157 sample children, and a rate of 1.0 lands at roughly
0.85–1.0× on the build's realisation. That is a frame/rules property the calibration weights
absorb (the publication-stack solve, L4), not something a take-up scalar reaches; uk-data's own
release sits at 0.78× on the same row. The TFC pair is the routed-share tension the issue
predicted: with spend routed at 0.593, the least-squares compromise pushes the TFC rate to ~0.96
(children +10%, spend −21%) because the frame's spend per child is below HMRC's.

The fitter gains an `--eps` flag (default unchanged at 1e-2, recorded in the receipt) so the
step can exceed the smallest eligible base's resolution; the fitter's draws are its own
identity-keyed realisation (seed 42 by default; seed 0 does not reproduce the build's stage
draws either), so a fitted rate is a population parameter and the build's flip set differs by
sampling noise — for targeted, ±10–20% at this base. The exact fix — an expected-count objective
(rate × weight over the rules-eligible base, no draw) with an analytic gradient — is a fitter
redesign outside this PR and is filed as a follow-up.

**Second run (`fit_2024_eps05.json`: maxiter 10, eps 0.05):** loss 0.302 → 0.077, still
`success: false` at the iteration cap, and two of the four rates on the upper bound:

| rate | incumbent | fitted | achieved ÷ target at the fitted vector |
|---|---|---|---|
| `tax_free_childcare` | 0.88 | **1.000 (bound)** | spend 0.806, children 1.142 |
| `extended_childcare` | 0.812 | 0.562 | extended 2–4 0.948 |
| `targeted_childcare` | 0.597 | **1.000 (bound)** | targeted 0.878 |
| `universal_childcare` | 0.563 | 0.436 | universal-only 0.955 |

The TFC objective is flat between 0.9 and 1.0 (spend 0.79–0.81 against children 1.10–1.14: the
two HMRC rows cannot both be met once spend is routed at 0.593, because the frame's spend per
child with a used account sits ~30% below HMRC's; uk-data's comparator ratios are 1.01 / 0.99 on
its own frame), so the bound hit is the optimizer sliding along a flat valley, not a measured
take-up of 100%. The targeted bound is the reachable-ceiling problem above. A rate of 1.0 also
makes the persisted `would_claim_*` column constant, which the degenerate-surface gate refuses
without a signed entry — a fit cannot sign that.

**Third run — the fitter re-designed (`fit_2024_expected.json`, tool version 3):** the second
run's rates were landed and the spine rebuilt, and the rebuilt twin exposed a seam the receipts
must carry: the build keys every take-up flag as `seed 0 : output : int64 benunit_id` at the
`frs_take_up` stage, and later stages (SPI support channel, CGT incidence clone, age tail)
clone and re-key rows, so no re-drawn realisation can sit on the persisted one — re-hashing the
final ids agrees with the persisted universal flag on 66% of rows, chance being 51% — and the
fitter's own draws (seed 42 on the engine's int32-cast ids, 2³¹ < the spine's ids) were a third
realisation. At identical rates the universal-only count read 397 592 / 434 097 / 463 758 across
the three (fitter s42 / s0 / build). The fitter therefore no longer draws: two engine runs with
the flags forced on (extended on, then off) give each row's weighted per-row contributions, and
each target's expectation is linear in its own rate and bilinear with the extended rate through
the engine's mutual exclusions (targeted and universal need a family that is not
extended-eligible). The objective is smooth, exact for the build (the persisted flags are one
draw from it), and reports each row's ceiling at a take-up of one. Checks: at the incumbent
rates the expectations read spend 0.71, children 1.005, extended 1.32, targeted 0.49, universal
1.06 against the L3 measurements 0.70 / 1.02 / 1.28 / 0.60 / 1.10 (draw noise, amplified for
small bases by cloning); the unit test pins the bilinear form.

| rate | incumbent | fitted (converged, loss 0.080) | expectation ÷ target | ceiling at 1.0 |
|---|---|---|---|---|
| `tax_free_childcare` | 0.88 | 0.997 | spend 0.804 · children 1.138 | 0.806 · 1.142 |
| `extended_childcare` | 0.812 | **0.6054** | 0.986 | 1.629 |
| `targeted_childcare` | 0.597 | 1.000 (bound) | 0.852 | 0.792 (0.852 at the fitted extended rate) |
| `universal_childcare` | 0.563 | **0.4539** | 1.000 | 1.581 |

**Landed (plan A6 as exercised):** `extended_childcare` 0.6054 and `universal_childcare`
0.4539 as `2024-04-06` entries with `status: fitted_offline` and a `fitting_receipt` (receipt
sha, seed, engine 2.94.0, input sha, feed sha, contract sha at fit, optimizer block, achieved ÷
target). `tax_free_childcare` (0.88) and `targeted_childcare` (0.597) are **held at their
incumbent values**: TFC sits at its ceiling because with spend routed at 0.593 the frame's
spend per child with a used account is ~30% below HMRC's, so the two HMRC rows cannot both be
met (spend 0.81 against children 1.14 anywhere above 0.9); targeted's eligible base — England, a
qualifying benefit or UC/TC criterion, and not extended-eligible — reaches 0.85 of the Jan-2024
count at a take-up of one. Both are put to María (A22/A23); a rate of 1.0 would also make the
persisted flag column constant, which the degenerate-surface gate refuses. The two pairs are
separable (targeted's rate does not move the extended count; TFC touches no DfE row), so the
landed pair is the joint fit's own solution for those rows. The eps-0.05 landing above
(0.5622 / 0.4356) is superseded.

**Rebuilt on this contract (`l5-refit-twin-b/`, sha `37280e7c…`):** 28 stages, rows
113 626 / 61 234 / 52 846, 15 / 15 spine gates; persisted shares extended 0.6053, universal
0.4512 (rates 0.6054 / 0.4539). Payload against the L2 column twin
(`l5b_vs_l2_payload.json`): benunit — the two re-fitted flags move (12 374 and 6 694 rows), the
seven other draw columns byte-equal; person — 28 `student_loan_balance` rows (the stock
top-up's selection); household — 33 imputed columns through the predictors, as in L2 vs L1
(energy IPF 52 586 rows, property rescale 24 948 owners, wealth/consumption draws ≤ 445 rows);
time_period byte-equal. Signed in the swap register as `childcare-take-up-refit-834`
(`mechanism_change`, max |Δshare| 0.2068 / 0.1103 against the packaged reference). Release
round trip (`l5b_refit_childcare_receipt.json`, engine 2.94.0, design weights):

| target (basis 2024 / 2025) | L3 (incumbent rates) | twin-b (landed) | expectation at the landed vector |
|---|---|---|---|
| TFC top-up £ | 0.698 / 0.772 | 0.698 / 0.772 | 0.710 |
| TFC children with used accounts | 1.020 / 0.951 | 1.020 / 0.951 | 1.005 |
| extended 2–4, England | 1.283 / 1.261 | **0.999 / 0.982** | 0.986 |
| targeted 2-year-olds, England | 0.596 / 0.738 | 0.628 / 0.772 | 0.509 |
| universal-only 3–4s, England | 1.104 / 1.177 | **1.087 / 1.157** | 1.000 |

The extended row lands on its expectation; universal realises 9% above and targeted 23% above
theirs — the persisted flags are one identity-keyed draw and the SPI/CGT clone stages copy a
source family's flag onto every clone, so the realised count's effective sample is far smaller
than its row count on the small bases (targeted: ~157 candidate children). The expectation is
what the rate controls; the calibration weights absorb the realisation.

### L4 — composition on `uk-publication-stack-834`

The composition branch (`repos/populace-834-l4`, `uk-publication-stack-834` = the #834 tree plus
the publication stack's target-fit deferral register and its gate wiring, ported by hand from
stack `6d3f984a`; main's local-candidate branch of the evaluator preserved) runs
`tools/calibrate_uk_national_dataset.py` on the rebuilt twin against the pinned feed (facts
`6ae49d7d…`, manifest `dcda51d6…`), 1 500 epochs, Adam 0.02, max weight ratio 10, the packaged
measure exclusions, release id `uk-spine-assessment-834-l4`.

**Run 1 (first landing 0.5622 / 0.4356, per-target uniform weights — superseded).** The runner
was invoked without `--target-weight-rule`, so every one of the 371 targets weighed 1.0 where
L4-pre had weighed families equally (`weight_kind: provided`, each UC cell ≈ 0.01). Loss 0.374 →
0.0135, 95.4% within 10%, ESS 8 154; the seven new rows all within 0.2%, and three of the four
deferred UC with-children cells inside 1.2% (children_5_or_more −28.3%) — but at the price the
family weighting exists to prevent: `hmrc.cgt.gains_total` −30.3% (OBR CGT exact),
`ons.savings_interest_income` −74.8%, `voa.council_tax_stock.band_a` +34.2%,
`slc.borrowers.plan_2_liable` −26.6%. The 100-row UC family dominated the solve. Kept as the
weighting-sensitivity receipt (`l4-834/run1/`), not as the composition measurement. The
terminal battery also failed closed on the ported gate (`KeyError: exclusions_evaluated_on`):
the binding's required-artifact set lacked the exclusion clock, fixed on the branch
(`7234d305`).

**Run 2 (converged landing 0.6054 / 0.4539, `family_equal` as L4-pre; `l4-834/run2/`).** Loss
0.352 → 0.0235, 90.0% of 371 targets within 10%, ESS 5 868 — L4-pre read 0.374 → 0.0259, 90.7%
of 364, 5 733. The seven new rows all bind and fit (initial → final):

| row | initial | final |
|---|---|---|
| `hmrc.tfc.government_top_up` (£599.8m, FY2025-26) | 0.767 | 0.999 |
| `hmrc.tfc.children_with_used_accounts` (1 151 515) | 0.945 | 0.999 |
| `dfe.funded_childcare.working_parent_children_2_to_4` (621 482) | 0.975 | 1.000 |
| `dfe.funded_childcare.early_learning_2_year_olds` (95 031) | 0.766 | 0.997 |
| `dfe.funded_childcare.universal_only_children` (396 965) | 1.149 | 0.995 |
| `dft.bus_fare_receipts.england` (£3.417bn) | 0.569 | 0.999 |
| `dft.bus_net_support.england` (£3.025bn) | 0.712 | 1.002 |

The pre-existing surface is where L4-pre left it, to the second decimal: the seven cells
outside `uk_target_fit`'s 25% are the same seven — OBR CGT +44.2% (A21), the four UC
with-children cells (single_with_children −35.3%, children_1 −28.9%, children_2 −38.9%,
children_5_or_more −28.6%; signed on the stack's register to 2026-09-30) and the two
state-pension SPI bands (+34.6% / +25.3%; the 50–70k deferral the stack retired as stale
twice, back outside the bound here); UC caseload −13.4% (L4-pre −12.7%); the two HMRC CGT
totals within 0.3%; the Scotland youngest-child row +0.3%; the private-pension 100–150k band
+24.7% (L4-pre +24.9%). So the childcare and bus rows are reachable by the weights on this
frame (targeted's 0.77× at design weights closes to 0.997 — the A22 row needs no deferral),
and the port neither helps nor harms the #145 residuals. Against uk-data 1.57.2's release ratios
at 2024 (universal 0.93, extended 1.16, targeted 0.78, TFC children 1.01, spend 0.99) the
composed surface binds the 2025 counterparts to within 0.5% each.

The terminal battery on run 2 reports `uk_target_fit` as `evidence_absent` (missing
`exclusions_evaluated_on`): the ported gate reads the run clock from the battery's artifacts,
and main's calibration seam (`calibration_run.py`) supplies the clock only to the spine build's
battery, where the stack's own runner supplied it to the seam. Fixed on the composition
branch (the seam now passes the evaluation date, as the rowwise candidate build does).

**Run 3 → run 4.** The re-run refused at the input sidecar: the runner's `--build-record-json`
is the seam's own output record, not the input's sidecar (the seam reads that from the H5's
`.build.json` path), and runs 1–2 had been pointed at the twin's spine record, so run 2 had
overwritten `l5-refit-b.build.json` with the seam's record (kept as
`l4-834/run2/calibration_build_record.json`). The H5, gate report and shares were untouched.
The twin is rebuilt into `l5-refit-twin-b2/` (sha `c63d824f…`; `l5b2_vs_l5b_payload.json`:
`payload_identical: true` — benunit, household, person and time_period byte-equal to twin-b,
the twin-determinism property of the 686 receipts holding on this tree) and
run 4 measures that build on the rebased composition branch (`91d740d6`, `385efc4d`,
`955c5dd2` on `db489ac0`); the solve is seeded, so run 4 reproduces run 2's numbers and adds
the gate verdict.

**Run 4 (`l4-834/run4/`, twin-b2, the composition measurement).** Every final ratio equals run
2's (max |difference| 0.0). Terminal battery with the ported register, evaluated 2026-09-05:
`uk_target_fit` **fails** and blocks at the terminal phase —

| disposition | cells |
|---|---|
| deferred (in force to 2026-09-30, microcosm#796) | the four UC with-children cells: single_with_children −35.3%, children_1 −28.9%, children_2 −38.9%, children_5_or_more −28.6% |
| failing, unsigned | `obr.capital_gains_tax` +44.2% (A21); `hmrc/state_pension_income_band_50_000_to_70_000` +34.6% (the deferral the stack retired as "measured stale twice", outside the bound again here as at L4-pre's +33.2%); `hmrc/state_pension_income_band_20_000_to_30_000` +25.3% (L4-pre +26.7%) |
| stale (back inside the bound; the rot rule says remove) | `hmrc/private_pension_income_count_income_band_100_000_to_150_000` +24.7% |

None of the four is a #834 row: the seven new rows pass, and the A22 question (would the
targeted row need a deferral?) is answered no. `release_candidate: false` throughout: the
seam never signs shippability.

**Run 5 (`l4-834/run5/`, the signed CGT deferral).** María ruled A21 as option 1 on 2026-09-05:
the facts stay bound as published and the OBR miss is signed on the stack register —
`obr.capital_gains_tax@2025`, adjudication microcosm#875 (the declared 2024-25 → 2025
translation of the gains total: the index ruling, applying `uprating_index` in the compile
step, the taxpayer count flat at 584k under the £3 000 exempt amount, the cash-lag mapping for
self-assessed OBR lines), window 2026-09-05 → 2026-10-05, on `uk-publication-stack-834`
(`7aab6959`); #736's "CGT 2025 level" item points at it. Every final ratio equals run 4's
(max |difference| 0.0); the battery now reads five in-force deferrals (the four UC
with-children cells and the CGT cell), two unsigned failures — the state-pension SPI bands
50–70k +34.6% and 20–30k +25.3% — and the private-pension 100–150k entry stale (+24.7%). Those
three are the stack's, not #834's, and stay open for its next re-cut. While the CGT entry is
in force the calibrated 2025-26 base carries ~40% more CGT liability than OBR expects; the
publication's known issues must say so.

## Part D — bus (#789)

E9's Part D (`experiments/685-net-new-stages-receipts.md`) is the sizing evidence: at design
weights England fares 0.56× BUS05ai and net support 0.71× BUS05bi; London 0.28×; England outside
London 0.74× / 0.97×; the LCFS fare gradient runs against NTS0705a (#790's disposition).

Re-measured on the L2 column twin against the FY2025 facts the two active rows bind
(`l2_bus_design_weight_receipt.json`, design weights, `region` ≠ Scotland/Wales/NI as England):

| row | fact (FY2025, £bn) | L2 twin at design weights | ratio |
|---|---|---|---|
| `dft.bus_fare_receipts.england` (active) | 3.417 | 1.945 | 0.569 |
| `dft.bus_net_support.england` (active) | 3.025 | 2.153 | 0.712 |
| London fares (held, `region == LONDON`) | 1.347 | 0.372 | 0.276 |
| outside-London fares (held) | 2.070 | 1.573 | 0.760 |
| outside-London net support (held) | 1.895 | 1.836 | 0.969 |

The frame reproduces E9's shape to the second decimal, so the routed share and the re-pin
leave the bus surface where E9 measured it: the England rows go to the solver at 0.57× / 0.71×,
which is the #790 gradient question (London's 0.28× against outside-London's 0.76× is the LCFS
fare gradient, not a weighting residual) and the reason the London / outside-London rows are
declared with their model-side filters but signed-excluded rather than bound. The feed carries
no UK-level fare or support fact (BUS05i is England-only), so the `.uk` rows stay signed-excluded
on that ground.

## Part E — the childcare rows at both bases (declared once, resolved by period)

Compiled from the pinned feed with the committed declarations (`compile_ledger_target_references`),
restamped at each target period — the rule A1 ruled:

| row | at 2024 (fitter basis) | at 2025 (calibration basis) |
|---|---|---|
| `hmrc.tfc.government_top_up` | £632.2m (FY2024-25) | £599.8m (FY2025-26) |
| `hmrc.tfc.children_with_used_accounts` | 1 085 020 | 1 151 515 |
| `dfe.funded_childcare.working_parent_children_2_to_4` | refuses: DfE suppresses the Jan-2024 2-year-old cell, the sum's cardinality guard reports one member of two (the fitter uses its Jan-2025 vintage override) | 621 482 (Jan-2025, 2 members) |
| `dfe.funded_childcare.early_learning_2_year_olds` | 115 852 (Jan-2024) | 95 031 (Jan-2025) |
| `dfe.funded_childcare.universal_only_children` | 416 537 (Jan-2024, `difference` of 2) | 396 965 (Jan-2025) |

The 2024 column is the #834 contract as written; the 2025 column is what the calibration binds.
