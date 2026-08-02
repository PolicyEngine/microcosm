# ACS-transfer predictor dtype adjudication

Date: 2026-08-01

Base: `f53032f`

Implementation: `783005b`; full primary-chain guard: `0e23a5e`

Runtime used for the executable sweep: pandas 3.0.3, NumPy 2.4.6,
quantile-forest 1.4.2, scikit-learn 1.8.0, PolicyEngine-US 1.764.6, and
PolicyEngine-Core 3.26.11.

## Root cause

Cross-spine assembly must represent a source-specific non-float column on peer
rows. It therefore creates an `object` column containing the source's Python or
NumPy booleans and `None` on peer rows. The source-output merge assigns later
boolean values into that existing column without changing its physical dtype.
The result is a valid *semantic boolean* even though pandas reports `object`.

The target path fixed in #589 recognized that shape through the shared
`_is_semantic_boolean` predicate. The predictor path still called
`_is_numeric_or_bool`, whose baseline implementation accepted only physical
boolean or numeric dtypes. Production therefore stopped on `is_female` before
QRF construction:

```text
TypeError: ACS transfer donor person predictors must be numeric/boolean: ['is_female'].
```

The sweep found one sibling: raw `is_household_head` is also an object-backed
semantic boolean after assembly. It flowed through `_person_column`, which used
the same rejecting helper. No other sibling violation was found. Tenure is the
only other object-backed predictor source and already has an explicit enum/code
codec.

The correction makes `_is_numeric_or_bool` delegate to the same
`_is_semantic_boolean` predicate used by targets. Every predictor boundary then
uses the columnwise float encoder: booleans become `1.0`/`0.0`, and missing
cells become `NaN`. This avoids pandas' unsafe whole-DataFrame object conversion
and keeps missingness visible.

## Transfer-family sweep

The producer-observing guard executes the real assembly, pre-clone source
operators, clone, primary PUF chain, and post-clone source operators. It then
constructs every transfer family's actual donor and recipient feature surface.
The fixture exposes every optional predictor on both sides, while the runtime
still fits the recipient's observed optional-feature subsets.

The surface labels below mean:

- `P10`: person required `age`, `is_female`, and state plus all seven optional
  person predictors. Non-housing fits use 3 required + a subset of 7 optional.
- `P10-H`: the same ten columns, with household-head and tenure promoted to
  required for housing and the five income analogs left pattern-optional.
- `G11`: four required group aggregates plus the seven optional group
  aggregates. Each fit uses the required four + an observed optional subset.

| Entity | Transfer family | Targets | Swept predictor surface |
| --- | --- | ---: | --- |
| person | `puf_tax_itemization` | 52 | `P10` |
| person | `adult_care` | 2 | `P10` |
| person | `housing` | 1 | `P10-H` |
| person | `model_required_numeric` | 7 | `P10` |
| person | `model_required_boolean` | 14 | `P10` |
| person | `model_required_discrete` | 1 | `P10` |
| person | `source_operator_hours_worked` | 1 | `P10` |
| person | `source_operator_prior_year_income` | 2 | `P10` |
| person | `source_operator_relationship_inputs` | 2 | `P10` |
| person | `source_operator_medicare_take_up` | 1 | `P10` |
| person | `source_operator_wic_claim` | 1 | `P10` |
| person | `source_operator_child_support` | 2 | `P10` |
| person | `source_operator_disability_benefits` | 1 | `P10` |
| person | `source_operator_workers_compensation` | 1 | `P10` |
| person | `source_operator_weeks_unemployed` | 1 | `P10` |
| person | `source_operator_retirement_contributions` | 3 | `P10` |
| person | `source_operator_retirement_distributions` | 5 | `P10` |
| person | `source_operator_immigration` | 2 | `P10` |
| person | `source_operator_education_inputs` | 6 | `P10` |
| tax unit | `puf_tax_itemization` | 6 | `G11` |
| SPM unit | `benefit_participation` | 1 | `G11` |
| SPM unit | `model_required_numeric` | 1 | `G11` |
| SPM unit | `source_operator_housing_inputs` | 1 | `G11` |
| SPM unit | `source_operator_energy_subsidy` | 1 | `G11` |

Totals: 24 families and 115 distinct transfer targets: 105 person, 6 tax-unit,
and 4 SPM-unit targets. Target-family chunking does not change its feature
surface.

## ACS predictor dtype sweep

The table is exhaustive over the 32 `(entity, predictor)` pairs: ten person
features, eleven tax-unit features, and eleven SPM-unit features. “Pool/source”
records the physical producer shape before or during feature construction;
“QRF” records the dtype after the model-boundary encoder. Nullable values are
allowed only before row selection; every fit and predict matrix is finite.

| Entity | Predictor | Pool/source physical dtype | QRF dtype | Verdict |
| --- | --- | --- | --- | --- |
| person | `age` | `float64` | `float64` | Pass |
| person | `is_female` | semantic-bool `object`, nullable across spines | `float64` | Fixed observed failure |
| person | `__acs_transfer_state_fips` | observed `int64` after broadcast | `float64` | Pass |
| person | `__acs_transfer_employment_income` | nullable `float64` | `float64` | Pass by pattern |
| person | `__acs_transfer_self_employment_income` | nullable `float64` | `float64` | Pass by pattern |
| person | `__acs_transfer_social_security_income` | nullable `float64` analog | `float64` | Pass by pattern |
| person | `__acs_transfer_retirement_income` | nullable `float64` analog | `float64` | Pass by pattern |
| person | `__acs_transfer_interest_dividend_rental_income` | nullable `float64` analog | `float64` | Pass by pattern |
| person | `__acs_transfer_is_household_head` | semantic-bool `object` source, normalized to nullable `float64` | `float64` | Fixed latent sibling |
| person | `__acs_transfer_tenure_code` | object enum or numeric source, explicitly mapped to nullable `float64` | `float64` | Pass |
| tax unit | `__acs_transfer_person_count` | derived `float64` | `float64` | Pass |
| tax unit | `__acs_transfer_age_sum` | derived nullable `float64` | `float64` | Pass complete-case |
| tax unit | `__acs_transfer_female_count` | derived nullable `float64` | `float64` | Pass after boolean fix |
| tax unit | `__acs_transfer_state_fips` | derived nullable `float64` | `float64` | Pass complete-case |
| tax unit | `__acs_transfer_group_employment_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| tax unit | `__acs_transfer_group_self_employment_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| tax unit | `__acs_transfer_group_social_security_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| tax unit | `__acs_transfer_group_retirement_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| tax unit | `__acs_transfer_group_investment_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| tax unit | `__acs_transfer_group_head_count` | derived nullable `float64` | `float64` | Pass after boolean fix |
| tax unit | `__acs_transfer_group_tenure_code` | derived nullable `float64` | `float64` | Pass by pattern |
| SPM unit | `__acs_transfer_person_count` | derived `float64` | `float64` | Pass |
| SPM unit | `__acs_transfer_age_sum` | derived nullable `float64` | `float64` | Pass complete-case |
| SPM unit | `__acs_transfer_female_count` | derived nullable `float64` | `float64` | Pass after boolean fix |
| SPM unit | `__acs_transfer_state_fips` | derived nullable `float64` | `float64` | Pass complete-case |
| SPM unit | `__acs_transfer_group_employment_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| SPM unit | `__acs_transfer_group_self_employment_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| SPM unit | `__acs_transfer_group_social_security_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| SPM unit | `__acs_transfer_group_retirement_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| SPM unit | `__acs_transfer_group_investment_income_sum` | derived nullable `float64` | `float64` | Pass by pattern |
| SPM unit | `__acs_transfer_group_head_count` | derived nullable `float64` | `float64` | Pass after boolean fix |
| SPM unit | `__acs_transfer_group_tenure_code` | derived nullable `float64` | `float64` | Pass by pattern |

Incomplete group states are now handled without a false cross-state error:
finite member states are checked for genuine disagreement, while a partially
missing group receives a `NaN` state and is removed by the group complete-case
mask.

## Primary PUF QRF predictor sweep

The primary 65-target chain's eight base predictors are a separate feature
pipeline. Its established preparation converts both donor and recipient
tax-unit features to numeric values and uses concept-specific numeric
`fillna(0.0)`. This is not a boolean null-to-false policy. The QRF boundary
casts the matrix to `float64`. The guard calls the real
`prepare_us_puf_tax_detail_chain_inputs` seam used by the production targetwise
chain, then inspects donor and recipient preparation for all 65 prefixes.

| Base predictor | Prepared donor | Prepared recipient | QRF matrix | Verdict |
| --- | --- | --- | --- | --- |
| `puf_predictor_filing_status_code` | `float64` | `float64` | `float64` | Pass |
| `puf_predictor_tax_unit_person_count` | `int64` | `float64` | `float64` | Pass by numeric cast |
| `puf_predictor_employment_income` | `float64` | `float64` | `float64` | Pass |
| `puf_predictor_self_employment_income` | `float64` | `float64` | `float64` | Pass |
| `puf_predictor_taxable_interest_income` | `float64` | `float64` | `float64` | Pass |
| `puf_predictor_dividend_income` | `float64` | `float64` | `float64` | Pass |
| `puf_predictor_short_term_capital_gains` | `float64` | `float64` | `float64` | Pass |
| `puf_predictor_long_term_capital_gains` | `float64` | `float64` | `float64` | Pass |

At chain step `k`, the preceding zero through 64 targets join these eight
predictors. Donor numeric coercion and recipient raw-prior-draw validation keep
those additional features finite `float64`; no semantic boolean enters this
base set.

The full expanding-chain inventory follows. For each row, the effective set is
the base eight plus every target in the preceding rows; “count” is therefore
8 through 72. All 65 donor predecessors are producer `float64`, all recipient
raw-prior draws are required to be finite `float64`, and both sides reach QRF
as finite `float64` matrices. Raw boolean-like and year draws are intentionally
not snapped before they condition the next target.

| Step | Target being fit | Effective predictor count |
| ---: | --- | ---: |
| 1 | `employment_income_before_lsr` | 8 |
| 2 | `self_employment_income_before_lsr` | 9 |
| 3 | `taxable_interest_income` | 10 |
| 4 | `qualified_dividend_income` | 11 |
| 5 | `non_qualified_dividend_income` | 12 |
| 6 | `tax_exempt_interest_income` | 13 |
| 7 | `short_term_capital_gains` | 14 |
| 8 | `long_term_capital_gains_before_response` | 15 |
| 9 | `long_term_capital_gains_on_collectibles` | 16 |
| 10 | `non_sch_d_capital_gains` | 17 |
| 11 | `taxable_private_pension_income` | 18 |
| 12 | `taxable_ira_distributions` | 19 |
| 13 | `social_security_retirement` | 20 |
| 14 | `social_security_disability` | 21 |
| 15 | `social_security_dependents` | 22 |
| 16 | `social_security_survivors` | 23 |
| 17 | `alimony_income` | 24 |
| 18 | `alimony_expense` | 25 |
| 19 | `salt_refund_income` | 26 |
| 20 | `charitable_cash_donations` | 27 |
| 21 | `charitable_non_cash_donations` | 28 |
| 22 | `real_estate_taxes` | 29 |
| 23 | `home_mortgage_interest` | 30 |
| 24 | `investment_interest_expense` | 31 |
| 25 | `investment_income_elected_form_4952` | 32 |
| 26 | `student_loan_interest` | 33 |
| 27 | `educator_expense` | 34 |
| 28 | `qualified_tuition_expenses` | 35 |
| 29 | `casualty_loss` | 36 |
| 30 | `unreimbursed_business_employee_expenses` | 37 |
| 31 | `traditional_ira_contributions_desired` | 38 |
| 32 | `self_employed_pension_contributions_desired` | 39 |
| 33 | `rental_income` | 40 |
| 34 | `estate_income` | 41 |
| 35 | `farm_income` | 42 |
| 36 | `farm_operations_income` | 43 |
| 37 | `farm_rent_income` | 44 |
| 38 | `miscellaneous_income` | 45 |
| 39 | `partnership_income` | 46 |
| 40 | `s_corp_income` | 47 |
| 41 | `partnership_self_employment_net_earnings` | 48 |
| 42 | `estate_income_would_be_qualified` | 49 |
| 43 | `farm_operations_income_would_be_qualified` | 50 |
| 44 | `farm_rent_income_would_be_qualified` | 51 |
| 45 | `partnership_s_corp_income_would_be_qualified` | 52 |
| 46 | `rental_income_would_be_qualified` | 53 |
| 47 | `self_employment_income_would_be_qualified` | 54 |
| 48 | `sstb_self_employment_income_would_be_qualified` | 55 |
| 49 | `business_is_sstb` | 56 |
| 50 | `qualified_bdc_income` | 57 |
| 51 | `qualified_reit_and_ptp_income` | 58 |
| 52 | `sstb_self_employment_income_before_lsr` | 59 |
| 53 | `sstb_unadjusted_basis_qualified_property` | 60 |
| 54 | `sstb_w2_wages_from_qualified_business` | 61 |
| 55 | `unadjusted_basis_qualified_property` | 62 |
| 56 | `w2_wages_from_qualified_business` | 63 |
| 57 | `domestic_production_ald` | 64 |
| 58 | `unrecaptured_section_1250_gain` | 65 |
| 59 | `first_home_mortgage_balance` | 66 |
| 60 | `second_home_mortgage_balance` | 67 |
| 61 | `first_home_mortgage_interest` | 68 |
| 62 | `second_home_mortgage_interest` | 69 |
| 63 | `first_home_mortgage_origination_year` | 70 |
| 64 | `second_home_mortgage_origination_year` | 71 |
| 65 | `health_savings_account_ald` | 72 |

## Null-handling verdict

Required means that a predictor *column* must exist, not that every row must be
observed. The final ACS-transfer contract is:

1. Required and optional columns must have a physical numeric dtype or satisfy
   the one shared semantic-boolean predicate. Positive and negative infinity
   are rejected.
2. Columnwise encoding maps `True` to `1.0`, `False` to `0.0`, and
   `None`/`pd.NA`/`NaN` to `NaN`. It never maps a null boolean to `False`.
3. A donor row is fit-eligible only if all predictors selected for its
   recipient availability pattern are finite and every chained target is
   complete.
4. A recipient row is modeled only if every required predictor is finite.
   Missing optional predictors remove that feature from the row's pattern;
   housing's required head/tenure predictors instead make the row ineligible.
5. An ineligible missing-target recipient remains null and is counted in
   `unmodeled_recipient_rows`. If a family has no eligible recipient at all,
   the existing explicit error remains.

This masking is necessary at the ACS boundary. The generic QRF target validator
does not define predictor missingness semantics, and a non-degenerate fit with
NaN predictors is rejected by the pinned quantile-forest/scikit-learn stack.
The QRF documentation now says callers that require complete cases must filter
before fit. Tests assert that no non-finite predictor reaches either the ACS
fit or predict boundary.

## Production-path and guard receipts

The regression traverses assembly, real source merge, clone, primary PUF QRF,
post-clone completion, automatic donor selection, ACS predictor validation,
and ACS QRF fit/predict:

```text
assembled is_female: object, 4 null peer rows, observed values are bool
prepared/final is_female: object, 0 null rows, all observed values are bool
auto donor: puf_tax_detail, 6 person rows, importance weights
ACS QRF: 1 fit + 1 predict, all 10 features finite float64
result: 4 imputed recipient rows, 0 unmodeled rows
```

The expanded positive guard audits 115 targets, 32 ACS `(entity, predictor)`
pairs, all eight primary base predictors, and all 65 exact primary-chain
predictor sets (8 through 72 columns). Its predictor mutation test runs the real
producer chain but changes one produced `is_female` value from bool to integer,
creating a mixed object column. The guard fails closed and names `is_female`,
demonstrating that CI observes producer drift rather than merely testing a
hand-constructed validator input.

## Verification

All commands used the synced, locked sibling environment with this worktree's
five package `src` directories on `PYTHONPATH`, because the task prohibited
network access and this worktree had no reusable environment.

| Check | Result |
| --- | --- |
| Focused ACS transfer + multispine pool + spine assembly + PUF support + QRF suites | 192 passed, 0 failed, 2 warnings in 319.00s |
| #583 `test_us_spine_blindness.py` guard | 495 passed, 0 failed in 5.72s |
| Full workspace pytest suite | 4,744 passed, 59 skipped, 0 failed, 7 warnings in 5,725.20s (1:35:25) |
| `ruff check .` | Clean |
| Changed-file `ruff format --check` | 4 files already formatted |
| `git diff --check` | Clean |

The full-suite warnings were one expected overflow refusal fixture, two
PolicyEngine-US invalid-divide warnings, one legacy-matrix invalid-subtract
warning, two PyTorch sparse warnings, and one sandbox-specific joblib physical
core-count fallback. There were no test failures or warning-driven deviations.

No network, push, restricted-data certification, artifact promotion, or release
operation was attempted. Wheel packaging was not run because no packaging file
or import surface changed. The GitNexus offline index built successfully but
could not register under the sandbox's read-only home registry, so graph queries
were replaced with source tracing and executable producer-path observations.
