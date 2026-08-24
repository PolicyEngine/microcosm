# US release replacement scorecard

This is evidence for the owner's flip decision, not an automatic publication verdict. No gate, threshold, tolerance, or band is applied by this scorecard. The complete per-target table is in the JSON twin of this file.

## Frozen yardstick

- Fiscal registry: `c4ac617743f2` with 32,842 targets from Ledger facts `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.
- Weighted aggregate: `sqrt_value_concept_budget_weighted_mape_50_50_amount_count_target_scale_cap_100pct` with cap `1.0` and no family multipliers.
- Weighting rule: raw weight = sqrt(max(abs(target value), 1)); mean-normalized within amount/count value basis; each semantic concept group's total weight is scaled to its largest row weight; the amount and count bases are then scaled to equal total budget; the vector is mean-normalized; the aggregate is the weighted mean of per-target |actual - target| / max(|target|, 1), each capped at 100%.
- Terminal battery: 131 single-column plus 1 joint comparison(s); all are by-origin-only (ASEC vs ACS within one artifact).

## Artifact summary

| role | artifact SHA-256 | loader | households | nonzero weights | weighted loss | within 10% | terminal battery |
|---|---|---|---:|---:|---:|---:|---|
| incumbent | `48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e` | microcosm_entity_h5 | 57,240 | 57,240 | 0.11462448 | 0.26691432 | inapplicable |

The incumbent SHA-256 matches the artifact policyengine.py `5.0.3` resolves as the US default dataset: `policyengine/populace-us` revision `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`, file `populace_us_2024.h5` (build `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`).

## incumbent: loss by family and value basis

| family | basis | targets | weight share | loss contribution | weighted mean capped error | worst target |
|---|---|---:|---:|---:|---:|---|
| irs_soi | amount | 15,935 | 0.45322305 | 0.07624405 | 0.16822633 | irs_soi.ty2023.congressional_district_2022.all_returns.us.adjusted_gross_income@2024 |
| irs_soi | count | 15,425 | 0.38466844 | 0.0315149 | 0.08192744 | irs_soi.ty2023.congressional_district_2022.all_returns.us.net_capital_gains_returns@2024 |
| census_population | count | 936 | 0.08032061 | 0.00140236 | 0.01745958 | census_pep.cy2024.national_resident_population_age.30_to_34.population@2024 |
| federal_reserve | amount | 1 | 0.01417628 | 0.00097313 | 0.06864463 | federal_reserve_z1.cy2023.households_nonprofits_balance_sheet.net_worth.fl152090005.amount_outstanding@2024 |
| cms_medicaid | count | 134 | 0.01952493 | 0.00076891 | 0.03938119 | cms_medicaid.month2024_12.state_enrollment.us.total_chip_enrollment@2024 |
| bea | amount | 2 | 0.00557508 | 0.0006263 | 0.11233864 | bea_nipa.cy2024.total_wages_salaries.a034rc.wages_salaries_amount@2024 |
| state_income_tax | amount | 44 | 0.00436743 | 0.00061952 | 0.14184899 | census_stc.fy2023.individual_income_tax_collections.md.t40.collections@2024 |
| jct | amount | 11 | 0.00169048 | 0.0006009 | 0.35546219 | jct.tax_expenditures.cy2024.self_employed_pension_contribution_deduction.revenue_loss@2024 |
| cbo | amount | 5 | 0.01190517 | 0.00054369 | 0.04566852 | cbo.revenue_projection.ty2024.income_by_source.net_business_income.projected_amount@2024 |
| cms_aca | count | 102 | 0.00738269 | 0.00034612 | 0.04688278 | cms_aca.oep2024.state_marketplace.tx.aptc_recipients@2024 |
| ssa | count | 54 | 0.00298108 | 0.00032081 | 0.1076164 | ssa_ssi_monthly.month2024_12.ssi_federal_payment_recipients.by_age.under_18.recipient_count@2024 |
| ssa | amount | 57 | 0.00571689 | 0.00027115 | 0.04742979 | ssa_supplement.cy2024.ssi_payments.by_area_category.california_total.payment_amount@2024 |
| hhs_acf_tanf | amount | 30 | 0.00043213 | 0.00013478 | 0.31189822 | hhs_acf_tanf.fy2024.cash_assistance.ca.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds@2024 |
| usda_snap | count | 52 | 0.00477804 | 0.00012844 | 0.02688085 | usda_snap.fy2024.state_average_monthly_households.wro.ca.average_monthly_households@2024 |
| usda_snap | amount | 52 | 0.00248916 | 8.15440354e-05 | 0.03275967 | usda_snap.fy2024.national_benefits.national_total.total_benefits@2024 |
| cms_medicare | amount | 1 | 0.00042433 | 4.19377533e-05 | 0.098834 | cms_medicare.cy2024.part_b_premium_income.premiums_from_enrollees.actual_amount@2024 |
| hhs_acf_liheap | count | 1 | 0.00034422 | 5.93640860e-06 | 0.01724608 | hhs_acf_liheap.fy2024.national_profile.state_programs.households_served@2024 |

## incumbent: worst 50 targets by loss contribution

| target | period | family | target value | actual | relative error | loss contribution |
|---|---:|---|---:|---:|---:|---:|
| federal_reserve_z1.cy2023.households_nonprofits_balance_sheet.net_worth.fl152090005.amount_outstanding | 2024 | federal_reserve | 1.56080700e+14 | 1.66794801e+14 | 0.06864463 | 0.00097313 |
| bea_nipa.cy2024.total_wages_salaries.a034rc.wages_salaries_amount | 2024 | bea | 1.23879290e+13 | 1.05507363e+13 | -0.14830507 | 0.0005923 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.adjusted_gross_income | 2024 | irs_soi | 1.44248104e+13 | 1.63737258e+13 | 0.13510856 | 0.00058227 |
| irs_soi.ty2022.historic_table_2.us.all.adjusted_gross_income | 2024 | irs_soi | 1.47824922e+13 | 1.63737258e+13 | 0.10764312 | 0.00046962 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.net_capital_gains_returns | 2024 | irs_soi | 2.98457100e+07 | 1.21772927e+07 | -0.59199186 | 0.00045922 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.qualified_business_income_deduction_returns | 2024 | irs_soi | 2.50150400e+07 | 4.04497744e+07 | 0.61701818 | 0.00043819 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.taxable_interest_amount | 2024 | irs_soi | 1.23790578e+11 | 3.15594365e+11 | 1.54942153 | 0.00039924 |
| irs_soi.ty2022.historic_table_2.us.all.taxable_income_amount | 2024 | irs_soi | 1.16794025e+13 | 1.27125689e+13 | 0.08846055 | 0.00034304 |
| irs_soi.ty2023.table_4_3.all_returns_excluding_dependents.all.adjusted_gross_income | 2024 | irs_soi | 1.51968693e+13 | 1.63737258e+13 | 0.07744072 | 0.00034256 |
| irs_soi.ty2023.table_1_1.all.adjusted_gross_income | 2024 | irs_soi | 1.52860174e+13 | 1.63737258e+13 | 0.07115709 | 0.00031568 |
| irs_soi.ty2023.table_1_2.all_returns.all.adjusted_gross_income | 2024 | irs_soi | 1.52860174e+13 | 1.63737258e+13 | 0.07115709 | 0.00031568 |
| irs_soi.ty2022.historic_table_2.us.all.wages_salaries_amount | 2024 | irs_soi | 9.70885301e+12 | 1.05507363e+13 | 0.08671295 | 0.00030659 |
| irs_soi.ty2023.table_1_4.all.net_capital_gains_amount | 2024 | irs_soi | 9.66168014e+11 | 1.21405571e+12 | 0.2565679 | 0.00028616 |
| irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child.total.eitc_returns | 2024 | irs_soi | 4.23487958e+06 | 8.22712539e+06 | 0.94270587 | 0.00027546 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.limited_state_local_taxes_amount | 2024 | irs_soi | 2.50437565e+11 | 1.30064329e+11 | -0.48065168 | 0.00027294 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.interest_paid_deduction_amount | 2024 | irs_soi | 1.40135155e+11 | 2.27342799e+11 | 0.62231097 | 0.00026434 |
| irs_soi.ty2023.table_4_3.all_returns_excluding_dependents.all.taxable_income_amount | 2024 | irs_soi | 1.19173635e+13 | 1.27125689e+13 | 0.06672662 | 0.00026138 |
| irs_soi.ty2023.table_2_5.eitc_by_agi_children.no_qualifying_children.total.eitc_returns | 2024 | irs_soi | 3.30510983e+06 | 6.60019571e+06 | 0.99696714 | 0.00025736 |
| cbo.revenue_projection.ty2024.income_by_source.net_business_income.projected_amount | 2024 | cbo | 1.91600000e+12 | 1.60512572e+12 | -0.16225171 | 0.00025484 |
| irs_soi.ty2023.table_1_2.all_returns.all.taxable_income_amount | 2024 | irs_soi | 1.19444470e+13 | 1.27125689e+13 | 0.06430787 | 0.00025219 |
| irs_soi.ty2023.congressional_district_2022.all_returns.ca_total.adjusted_gross_income | 2024 | irs_soi | 1.94793409e+12 | 2.24143342e+12 | 0.15067211 | 0.00023862 |
| irs_soi.ty2023.table_2_5.eitc_by_agi_children.two_qualifying_children.total.eitc_returns | 2024 | irs_soi | 2.82144267e+06 | 5.41652682e+06 | 0.91977207 | 0.00021937 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.income_tax_liability_amount | 2024 | irs_soi | 2.05068370e+12 | 2.32071326e+12 | 0.13167782 | 0.00021397 |
| irs_soi.ty2022.historic_table_2.state_broad.ca.all.adjusted_gross_income | 2024 | irs_soi | 1.98700070e+12 | 2.24143342e+12 | 0.12804863 | 0.00020481 |
| irs_soi.ty2023.congressional_district_2022.all_returns.tx_total.adjusted_gross_income | 2024 | irs_soi | 1.21852666e+12 | 1.41751435e+12 | 0.16330187 | 0.00020455 |
| irs_soi.ty2023.congressional_district_2022.all_returns.ca_total.limited_state_local_taxes_amount | 2024 | irs_soi | 7.49343950e+10 | 2.59009009e+10 | -0.65435231 | 0.00020325 |
| irs_soi.ty2022.historic_table_2.us.500k_to_1m.taxable_interest_amount | 2024 | irs_soi | 2.77364912e+10 | 7.66587312e+10 | 1.76382223 | 0.00018898 |
| irs_soi.ty2023.congressional_district_2022.all_returns.fl_total.adjusted_gross_income | 2024 | irs_soi | 1.09239177e+12 | 1.26612524e+12 | 0.15903953 | 0.00018862 |
| irs_soi.ty2020.form_w2_social_security_tips.box_7_social_security_tips.return_count | 2024 | irs_soi | 6.03861300e+06 | 2.89974185e+06 | -0.51980002 | 0.00018137 |
| irs_soi.ty2023.congressional_district_2022.all_returns.ny_total.limited_state_local_taxes_amount | 2024 | irs_soi | 4.12540850e+10 | 9.43502839e+09 | -0.77129469 | 0.00017776 |
| irs_soi.ty2022.historic_table_2.state_broad.ca.all.wages_salaries_amount | 2024 | irs_soi | 1.35171286e+12 | 1.53192271e+12 | 0.13331962 | 0.00017588 |
| irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount | 2024 | irs_soi | 2.10534565e+12 | 2.32071326e+12 | 0.10229561 | 0.00016842 |
| irs_soi.ty2023.table_2_5.eitc_by_agi_children.three_or_more_qualifying_children.total.eitc_returns | 2024 | irs_soi | 1.55714243e+06 | 3.03241140e+06 | 0.9474207 | 0.00016787 |
| irs_soi.ty2022.historic_table_2.state_broad.tx.all.adjusted_gross_income | 2024 | irs_soi | 1.25211384e+12 | 1.41751435e+12 | 0.13209702 | 0.00016773 |
| irs_soi.ty2023.congressional_district_2022.all_returns.tx_total.qualified_business_income_deduction_returns | 2024 | irs_soi | 1.98114000e+06 | 3.64235160e+06 | 0.83851298 | 0.00016759 |
| irs_soi.ty2023.table_2_1.itemized_all_returns.all.adjusted_gross_income | 2024 | irs_soi | 4.66933240e+12 | 4.98817777e+12 | 0.068285 | 0.00016743 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.limited_state_local_taxes_returns | 2024 | irs_soi | 1.08425800e+07 | 1.47093435e+07 | 0.35662762 | 0.00016674 |
| irs_soi.ty2023.congressional_district_2022.all_returns.ca_total.net_capital_gains_returns | 2024 | irs_soi | 3.80893000e+06 | 1.54004608e+06 | -0.59567488 | 0.00016507 |
| cms_medicaid.month2024_12.state_enrollment.us.total_chip_enrollment | 2024 | cms_medicaid | 7.30587400e+06 | 4.24737432e+06 | -0.4186357 | 0.00016067 |
| irs_soi.ty2023.congressional_district_2022.all_returns.ca_total.qualified_business_income_deduction_returns | 2024 | irs_soi | 3.08014000e+06 | 5.06500313e+06 | 0.64440679 | 0.00016059 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.partnership_scorp_income_amount | 2024 | irs_soi | 9.96633005e+11 | 1.13442471e+12 | 0.13825721 | 0.00015662 |
| irs_soi.ty2022.historic_table_2.state_broad.fl.all.adjusted_gross_income | 2024 | irs_soi | 1.12178219e+12 | 1.26612524e+12 | 0.12867297 | 0.00015464 |
| irs_soi.ty2022.historic_table_2.us.1m_plus.taxable_interest_amount | 2024 | irs_soi | 1.47438602e+11 | 9.58877540e+10 | -0.34964282 | 0.00015234 |
| irs_soi.ty2023.congressional_district_2022.all_returns.ca_total.charitable_amount | 2024 | irs_soi | 3.23532190e+10 | 5.64925978e+10 | 0.74611985 | 0.00015228 |
| irs_soi.ty2023.congressional_district_2022.all_returns.ca_total.taxable_interest_amount | 2024 | irs_soi | 1.66550920e+10 | 4.36606944e+10 | 1.62146221 | 0.00014644 |
| irs_soi.ty2022.historic_table_2.state_broad.ca.all.taxable_income_amount | 2024 | irs_soi | 1.61442096e+12 | 1.77672164e+12 | 0.10053182 | 0.00014494 |
| jct.tax_expenditures.cy2024.self_employed_pension_contribution_deduction.revenue_loss | 2024 | jct | 1.66000000e+10 | 2.03311316e+08 | -0.98775233 | 0.00014441 |
| irs_soi.ty2023.congressional_district_2022.all_returns.fl_total.taxable_interest_amount | 2024 | irs_soi | 1.57682910e+10 | 3.74716372e+10 | 1.37639179 | 0.00014249 |
| irs_soi.ty2023.congressional_district_2022.all_returns.us.income_tax_before_credits_amount | 2024 | irs_soi | 2.18325166e+12 | 2.36379885e+12 | 0.08269646 | 0.00013865 |
| irs_soi.ty2023.congressional_district_2022.all_returns.ny_total.taxable_interest_amount | 2024 | irs_soi | 1.45116820e+10 | 3.03204487e+10 | 1.08938211 | 0.00013669 |

## Terminal by-origin battery

### incumbent

All 132 comparisons inapplicable — artifact carries no positive-weight clone-0 ACS-stacked origin rows (observed channels are listed in observed_origins); the ASEC-vs-ACS battery has an empty ACS side and is definitionally inapplicable.
- `person` origin rows: asec=66,001
- `spm_unit` origin rows: asec=23,286
- `tax_unit` origin rows: asec=30,974

## Mechanism citations

- `battery_channel_constants`: `packages/microcosm-build/src/microcosm/build/us_runtime/support_provenance.py:31,333-344; packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:263`
- `battery_finished_h5_materialization`: `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:3137-3215 (ephemeral SSI gate view); packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7808-7897,11474-11492,11530-11898 (finished-artifact evidence seam and canonical evaluator)`
- `battery_origin_masks`: `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11644-11709,11824-11832`
- `battery_receipt_keys`: `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11948-12154`
- `battery_registry`: `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3011-3025`
- `canonical_scorer_seam`: `tools/score_us_fiscal_targets.py:383-528`
- `cd_provenance_check`: `tools/build_us_fiscal_refresh_release.py:2519-2567,2570-2597`
- `chunked_materialize_score`: `tools/score_us_release_head_to_head.py:654-852,1399-1477; tools/build_us_fiscal_refresh_release.py:3730-3750; packages/microcosm-calibrate/src/microcosm/calibrate/matrix.py:286-355; score.py:79-142`
- `entity_h5_loader`: `tools/build_us_fiscal_refresh_release.py:2454-2471`
- `fraction_within_10pct`: `packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:249-260`
- `incumbent_package_resolution`: `policyengine.py@5.0.3 src/policyengine/data/bundle/manifest.json:113-140,156-160,181-189; src/policyengine/provenance/manifest.py:180-187,270-299,301-318,540-560; src/policyengine/provenance/dataset_sources.py:57-74,77-117; src/policyengine/tax_benefit_models/us/model.py:423-462`
- `legacy_flat_loader`: `tools/score_us_fiscal_targets.py:240-333`
- `loss_aggregate`: `packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:471-537,576-600 (relative_error_loss, the single canonical loss definition)`
- `loss_attribution`: `packages/microcosm-calibrate/src/microcosm/calibrate/_target_loss_attribution.py:143-176 (per-row capped error, weight share, and contribution formulas, applied with whole-registry weight normalization)`
- `loss_weighting`: `tools/build_us_fiscal_refresh_release.py:344-348,481-516,5781-5814,6214-6290`
- `materialization_drop_detection`: `tools/build_us_fiscal_refresh_release.py:4323-4350`
- `matrix_skip_detection`: `packages/microcosm-calibrate/src/microcosm/calibrate/matrix.py:286-355`
- `pool_battery_persistence`: `tools/build_us_multispine_pool.py:3263-3334,3533-3550,3766-3786; packages/microcosm-build/src/microcosm/build/gates.py:690-700`
- `pool_manifest_authentication`: `packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:371-588,719-869`
- `relative_error`: `packages/microcosm-calibrate/src/microcosm/calibrate/score.py:25-51`
- `single_registry_surface`: `packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1017`
