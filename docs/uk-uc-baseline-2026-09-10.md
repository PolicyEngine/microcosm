# UK Universal Credit baseline, 10 September 2026 rebuild

This section records the Universal Credit (UC) readout of the 10 September 2026 rebuild, the baseline the element bindings and the later UC repairs are measured against. It is a measurement at pinned inputs, not a certification: neither file is shippable, and no target or exclusion changes on the strength of these numbers. Issue record: [microcosm#882](https://github.com/PolicyEngine/microcosm/issues/882#issuecomment-5633606144). Evaluation page: https://uk-dataset-evaluation.vercel.app/55k-rebuild-2026-09-10/.

## Pins

| item | value |
|---|---|
| spine | spine-q, `data/ukds/acceptance/spine-q-355/spine-q.h5`, sha `cf1f9dda198a06819241ed3358f68dd99c675eb7ca45fa766a3ed4b07eb0337f` (28 stages, 52,846 / 61,213 / 113,590 households / benefit units / persons) |
| code | main `0afb1235` plus `--baseline-pi-floor` (this branch's floor-knob commits) |
| Chronicle feed (baseline build) | `ec7169b5db40b9f54117c80f70f14efc1dd0fedd`, facts `4a50ee9568a01bbb57f73d927084ed6b4b9e52249b51a2338455874ae6e382b5`, 131,450 rows |
| Chronicle feed (element targets, section below) | `474a0ae100e9dbfa43c167e9e200ec07dcfc6643`, facts `bb12d77a661ef1649c2907211bfd58bc031dcf7d41d2e5d39a09d63eb7266d3d`, 141,400 rows (chronicle#260 packages with chronicle#263 labels) |
| engine | policyengine-uk 2.97.0, core 3.31.0 |
| surface | 20,794 rows: 364 national, 20,430 local (constituency and local authority) |
| doctrine | `grain_equal`, 2,000 epochs, learning rate 0.15, maximum weight ratio 10, target loss cap 10, seed 42, no sparsity penalty in the solve |
| Q50f | 55,000-row L0 candidate, `--selection-pi-hi 0.5 --baseline-pi-floor 0.001`, run `spine-q/f100-k15-h55000-e2000-p50-f001-s42` |
| D2 | dense twin, 792,690 rows, run `spine-q/f100-k15-dense-e2000-s42` |

Receipts for the build itself: `experiments/355-uk-dataset-size-receipts.md`.

## Bound UC rows

The ten broad paid-claim rows (calendar-2025 targets from #891), relative error from each run's `calibration_diagnostics.json`:

| row | target | Q50f | D2 |
|---|---:|---:|---:|
| `dwp.uc.households` | 6,197,311 | +2.80 % | +2.99 % |
| `dwp.uc.households_single_no_children` | 2,990,070 | +1.90 % | +1.85 % |
| `dwp.uc.households_single_with_children` | 2,138,780 | +2.08 % | +3.84 % |
| `dwp.uc.households_couple_no_children` | 241,131 | +0.76 % | +2.25 % |
| `dwp.uc.households_couple_with_children` | 826,161 | +8.71 % | +5.26 % |
| `dwp.uc.households_children_1` | 1,190,124 | +0.03 % | +0.03 % |
| `dwp.uc.households_children_2` | 1,049,896 | +8.38 % | +9.23 % |
| `dwp.uc.households_children_3` | 477,888 | +6.13 % | +5.57 % |
| `dwp.uc.households_children_4` | 170,714 | −0.38 % | +0.97 % |
| `dwp.uc.households_children_5_or_more` | 76,320 | −0.67 % | −0.11 % |

For scale, the #892 fresh build measured the headline at −4.71 %, lone parents at −14.01 % and one child at −19.45 %.

The other bound rows: all 15 two-child-limit rows within 2.03 % on both files; Scotland child under one −0.97 % / −0.41 %; benefit-cap capped households −0.03 % / +0.08 %; the 82 bound payment bands within 4.34 % (Q50f) / 5.08 % (D2). Of the 108 bound national UC rows, 105 are within 5 % on Q50f (104 on D2) and none is past 10 %.

Local UC rows (3,508 across constituencies and local authorities): Q50f 99.29 % within 10 %, 100 % within 25 %, worst 15.6 %; D2 99.40 % / 100 % / 12.6 %. The childless bucket is the weakest inside the family (60.6 % within 5 % against 93–95 % for the child buckets).

## Excluded UC rows, as measured by the incumbent-surface evaluator

The three childless-couple payment bands from £26,400 upward stay at −100 % on both files (no model record). Of the 16 bands excluded under uk-data#452, seven are within 25 % on both files: lone-parent £12,000–13,200 (+14.7 % / +10.9 %), £13,200–14,400 (−15.7 % / −3.2 %), £14,400–15,600 (+10.2 % / +3.9 %), £15,600–16,800 (−10.7 % / −12.6 %), £16,800–18,000 (+14.1 % / +11.7 %), single £9,600–10,800 (+13.9 % / +2.1 %), and the HMRC self-employment count at £500k–1m. D2 alone adds lone-parent £18,000–19,200 (−6.9 %) and single £14,400–15,600 (−15.9 %). The rest stay 26–143 % off. Retirement is a signed register decision.

The OBR in-cap / outside-cap rows measure −96.7 % / +503 % on both files. Their sum (£79.29bn, FY2025/26) against the model's total UC: Q50f £80.38bn (+1.4 %), D2 £80.82bn (+1.9 %), enhanced FRS 1.57.3 £75.85bn (−4.3 %).

## Element support, conditioned on paid UC

Measured on Q50f at its calibrated weights with `tools/diagnose_uk_uc_elements.py` (Great Britain, benefit units with `universal_credit > 0`, policy year 2025) against the April–December 2025 means of the DWP all-claims series then in the feed:

| element | model, paid UC | published mean | gap |
|---|---:|---:|---:|
| LCWRA entitlement (`uc_LCWRA_element > 0`) | 2.62m | 2.42m | +8 % |
| deductions (`uc_deductions > 0`) | 2.99m (46.6 % of paid) | 3.0–3.1m (46–47 %) | ≈0 % |
| housing element, any tenure | 4.01m | 4.30m | −7 % |
| housing element, social rented only | 2.35m | not then in the feed | — |
| carer element | 0.60m | 1.13m | −47 % |
| childcare element | 456k (£2.06bn) | 184k Jan–Aug 2025 (≈£0.8bn) | +148 % |

Three readings behind the table. The feed's housing series was labelled `Social Rented Sector` but was the all-tenure count (its Chronicle package grouped tenure codes 1+2+3; corrected in chronicle#260). The evaluation page's T4 admin legs measured elements without a UC award condition, which is why they showed LCWRA at +216 % (7.29m unconditioned) and carer at +5 % (1.14m unconditioned). Nil-payment open claims cannot be identified in the model: the would-claim ∧ eligible ∧ positive-maximum proxy gives 14.3m units, so element rows bind on paid claims through the element × Payment Indicator crosses chronicle#260 added.

The two gaps are model concepts, not weights. The engine's carer condition is Carer's Allowance receipt (`is_carer_for_benefits` = `receives_carer_benefit`) while the UC carer element needs 35 hours a week of care for a qualifying-benefit recipient without Carer's Allowance; the engine's `care_hours` input is never filled by the spine. The childcare element is paid to every benefit unit where all adults work and childcare costs are reported, with no take-up or incidence lever.

## Downstream legs against enhanced FRS 1.57.3

T5 "UC taper to 20 %": Q50f −35.4bn, D2 −34.9bn against an expected −17.2 ± 15 (eFRS −22.6bn). T6 Resolution Foundation UC: 13 ok / 14 flagged, closer on 8 of 10 benchmarks; UC expenditure on the 2029-30 basis £91.1bn against RF's £86bn; mean monthly UC childcare element £636 against RF's £420. T6 childcare: UC childcare element £2.15bn against a £0.81bn benchmark.

## Support, concentration and the non-UC rows

Q50f: Kish ESS 16,785 (0.305 of 55,000), max/median weight 57, household mass 29.07m (−0.6 %), per-area support gate failed (643 of 650 constituencies under ESS 50). D2: Kish ESS 130,504, max/median 296 (fails the reviewed 100 ceiling), mass 29.15m (−0.3 %). Protected non-UC rows on the joint solve: income tax −19.4 % / −19.6 %, NI −3.9 % / −3.9 %, state pension −20.2 % / −13.9 %, child benefit +23.0 % / +21.1 %; the ONS lone-parent-with-dependent-children household row +21.9 % / +19.0 %. The #882 national-only diagnostic arm (family_equal, 1,500 updates) had income tax −12.7 %, state pension −6.4 %, child benefit +17.7 %; whether the UC rows or the local surface account for the difference cannot be read from these receipts.

## Element targets on the chronicle#260 feed (paid-claim basis)

Compile-time target values are calendar-2025 means of the Payment Indicator = Yes cells (childcare: the all-claims ODS count; deductions: March to December 2025), from the chronicle `474a0ae` artifact (chronicle#260 packages with chronicle#263 labels). Model values are Great Britain benefit units with `universal_credit > 0` and the element `> 0` at each file's calibrated weights, from `tools/diagnose_uk_uc_elements.py` (receipts `elements_q50f.json`, `elements_d2.json`).

| target | published (CY2025 paid mean) | Q50f | D2 | Q50f gap | D2 gap |
|---|---:|---:|---:|---:|---:|
| `dwp.uc.households_lcwra_element` (LCWRA entitlement) | 2,272,425 | 2,619,296 | 2,731,661 | +15.3 % | +20.2 % |
| `dwp.uc.households_carer_element` (carer entitlement) | 1,078,514 | 595,807 | 602,749 | -44.8 % | -44.1 % |
| `dwp.uc.households_housing_element` (housing entitlement, any tenure) | 4,037,650 | 4,006,395 | 4,063,719 | -0.8 % | +0.6 % |
| `dwp.uc.households_housing_element_social_rented` (housing entitlement, social rented) | 2,354,588 | 2,348,245 | 2,474,679 | -0.3 % | +5.1 % |
| `dwp.uc.households_housing_element_private_rented` (housing entitlement, private rented) | 1,570,545 | 1,658,150 | 1,589,040 | +5.6 % | +1.2 % |
| `dwp.uc.households_childcare_element` (childcare element (all claims, ODS)) | 178,667 | 455,634 | 451,401 | +155.0 % | +152.6 % |
| `dwp.uc.households_with_deduction` (households with a deduction) | 3,130,000 | 2,990,545 | 2,991,319 | -4.5 % | -4.4 % |
| `obr.universal_credit` (total UC expenditure, UK, FY2025/26) | £79.29bn | £80.38bn | £80.82bn | +1.4 % | +1.9 % |

Deductions and the two tenure rows are within 6 % on both files and LCWRA within 15 % (Q50f) / 20 % (D2); those four rows are active. LCWRA stays in the objective by the rule its binding note states: a concept mismatch goes to the register only when it puts the row outside the 25 % release bound at the baseline. Carer and childcare are measured but held out of the objective on the reviewed exclusion register until their model gaps are repaired (see the element-support section above). The any-tenure housing row is also measured but held out: DWP's 'Yes' is social plus private plus 'Other or unknown' tenure, the model has no other/unknown tenure, so its any-tenure count is identically the two tenure rows and the published total exceeds what the model can reach by the other/unknown cell (112,518 in calendar 2025, 2.8 %); fitting all three would over-fill the tenure rows. The register entries are signed by María in review of the element PR (2026-09-15) with a one-month window. The paid-claim targets sit below the April–December all-claims means quoted in the earlier section because the calendar window includes the lower January–March 2025 months and excludes nil-payment claims.

Paid GB caseload for scale: Q50f 6,416,983, D2 6,428,401; share of paid units with earnings 51.6 % / 49.6 %.

## Repairs measured against this baseline (16 September 2026, the #882 repairs PR)

Receipts: `experiments/882-uc-repairs-receipts.md`. Twins built with the licensed recipe at design weights: a control at the branch base (51c31438, engine 2.97.0) and the repaired branch (engine 2.98.0, PolicyEngine/policyengine-uk#1857). Projections onto Q50f use the design-weight ratio repaired / control.

- Carer element: the engine's carer condition now holds on Carer's Allowance receipt or 35 weekly care hours, and the spine fills `care_hours` from the FRS adult HOURTOT band with `would_claim_carers_allowance` following reported receipt. Paid-UC carer-element units 425,757 → 768,273 (+80 %); projection 1,075,100 against 1,078,514 (−0.3 %). Carer's Allowance spend stays on reported claimants once the take-up flag is re-derived after the SPI fill. The register row is retired.
- Childcare element: `would_claim_uc_childcare` is drawn per family type (single 0.441, couple 0.2682) so the expected paid-UC count equals the published calendar-2025 means at the Q50f weights; on the twin the realized count is 116,324 against 353,470 at rate one, below the expectation because SPI-synthetic units copy their donor's flag on a base of 378 and 201 records; projection 149,950 against 178,667 (−16 %). The register row is retired with that caveat recorded.
- Legacy rows: the A16 exclusions `obr.housing_benefit` and `dwp.jsa_claimants` are re-adjudicated to 2026-12-08 on the mechanism receipts (the SPI support channel carries reported Housing Benefit and JSA into synthetic households; the Universal Credit draw had zeroed Housing Benefit for pension-age units). Gating the draw to units with an adult under State Pension age lifts engine Housing Benefit from 401,489 to 644,183 units at design weights (£2.61bn → £4.11bn) and moves paid UC −1.8 % (the rate now realizes 0.559 among the eligible units where the old all-units draw realized 0.582); three Housing Benefit caseload rows bind as measured diagnostics.
- Benefit-cap amount bands: fourteen DWP Table 4 bands bind at benefit-unit grain; the up-to-£100 band is in the objective and the thirteen others are measured on the register. On Q50f the tail above £1,300 a month is 14.2 % against 0.6 % published; on the spine at design weights it is 6.6 % from eight source households, so the Q50f tail is a selection-and-calibration effect on a thin base.
