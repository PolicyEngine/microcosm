"""Regenerate the US release input-column coverage manifest (microcosm #368).

The manifest is the declared column-coverage contract the release gate enforces:
every input column the reference eCPS exports must be persisted by a microcosm
release as a real key with non-default signal, or carry a reviewed exclusion.

Derivation (fully from checked-in, sha-pinned facts — no transient artifact):

- Required surface = every input-variable column the pinned reference eCPS
  populates, i.e. the ``nonzero_shares`` keys of ``ecps_parity_reference.json``,
  plus explicit later inputs needed by shipped reform probes
  (computed once from the sha-verified ``enhanced_cps_2024.h5``; an input the
  incumbent exports but leaves all-zero is not a coverage requirement, the same
  rule the parity gate uses).
- Status per column:
    * ``reviewed_exclusion`` — the column is a documented incumbent-parity gap
      (an entry in ``ecps_parity_known_gaps.json``, carrying that register's
      reason and tracking issue), so the current candidate does not populate it
      yet. EXCEPT the SSI countable-resource asset inputs (below).
    * ``required`` — every other populated layer, PLUS the SSI countable-resource
      asset inputs. Per #368 ("it must be an actual gate"), the asset inputs get
      NO exclusion even though they are currently absent, so the gate ships red
      for today's artifacts and Deliverable 2 (asset restoration) turns it green.

The SSI asset inputs are ``bank_account_assets``, ``stock_assets``, and
``bond_assets`` — the exact ``adds`` list of ``ssi_countable_resources``. With
them absent, countable resources are 0 for every record, so the SSI asset-limit
reform probe scores $0; that is the failure the gate exists to surface.

Run:  uv run python tools/build_us_release_input_coverage_manifest.py
It rewrites packages/microcosm-build/src/microcosm/build/us/
release_input_coverage_manifest.json. A test asserts the committed file matches
this regeneration, so the manifest cannot silently drift from the pinned eCPS
surface or the parity register.
"""

from __future__ import annotations

import json
from pathlib import Path

US_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "us"
)
MANIFEST_PATH = US_PACKAGE_DIR / "release_input_coverage_manifest.json"

#: The ``adds`` list of PolicyEngine-US ``ssi_countable_resources``: the asset
#: leaves SSI eligibility keys on. Absent from the release, they zero countable
#: resources and the SSI asset-limit reform scores $0. Per #368 these ship as
#: hard requirements with NO reviewed exclusion.
SSI_COUNTABLE_RESOURCE_ASSETS = (
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
)

# The pinned reference H5 predates the retired pipeline's FLSA-premium export,
# OBBBA's distinct qualifying passenger-vehicle interest leaf, its five final
# desired retirement-contribution inputs, its final SIPP-imputed SSI
# disability criterion, and the #282 capital-gain-distributions route split's
# Schedule-D leg. These later inputs are hard requirements because the
# shipped validation provisions must bind.
POST_REFERENCE_ECPS_REQUIRED_INPUTS = (
    "fsla_overtime_premium",
    "qualified_passenger_vehicle_loan_interest",
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
    "meets_ssi_disability_criteria",
    "schedule_d_capital_gain_distributions",
    "health_insurance_premiums",
    "is_self_employed",
    "pre_subsidy_care_expenses",
    "is_incapable_of_self_care",
)

# Per-column annotations for post-reference hard requirements whose absence
# from today's artifacts is the intended red gate (the #368 SSI-asset
# pattern, extended by #462 to the capital-gain-distributions route).
POST_REFERENCE_COLUMN_NOTES = {
    "schedule_d_capital_gain_distributions": (
        "Schedule D line 13 route leg of the #282 capital-gain-distributions "
        "split (memo component of long_term_capital_gains, written by the "
        "capital_gain_distributions source stage); required with NO reviewed "
        "exclusion per PolicyEngine/microcosm#462 so a release whose export "
        "drops the route (the Build M live default shipped it at $0 while "
        "non_sch_d_capital_gains carried 7.3x its SOI target) fails the "
        "coverage gate. Currently absent — this is the intended red gate "
        "until the Build N rebuild carries the split through."
    ),
    "health_insurance_premiums": (
        "Self-employed premium attribution leaf (PolicyEngine/microcosm#451 "
        "item 2): the deterministic attribution operation of the "
        "other_health_insurance_premiums release stage copies the reported "
        "non-Part-B premium onto this person input for strictly-positive "
        "Schedule C people outside the Medicare proxy, which is the only "
        "channel populating the section 162(l) self-employed health ALD "
        "(SOI Pub 1304 Table 1.4 TY2023: 3,595,764 returns / $31.23B, "
        "ledger#105). Absent on pre-attribution artifacts by construction; "
        "any new release produces it."
    ),
    "is_self_employed": (
        "Gate flag for the self-employed premium attribution "
        "(PolicyEngine/microcosm#451 item 2): opens the defined_for gate on "
        "the engine's self_employed_health_insurance_premiums "
        "adds-aggregation for every strictly-positive Schedule C person. "
        "Written by the same attribution operation as "
        "health_insurance_premiums."
    ),
    "pre_subsidy_care_expenses": (
        "Adult/disabled-dependent care expense leaf of the section 21 CDCC "
        "(PolicyEngine/microcosm#451 item 1), written by the adult_care_inputs "
        "base-builder stage: without it every CDCC reform binding through "
        "adult care scores exactly $0 (the #368 absent-input class). "
        "Currently absent — the intended red gate until the next base "
        "rebuild carries the stage through."
    ),
    "is_incapable_of_self_care": (
        "Section 21 qualifying-individual flag for the CDCC adult-care leg "
        "(PolicyEngine/microcosm#451 item 1), derived from the measured ASEC "
        "self-care difficulty item PEDISDRS by the adult_care_inputs "
        "base-builder stage. Also read by SNAP/Medicaid work-requirement "
        "exemptions and state CDCC analogs in PolicyEngine-US 1.764.6. "
        "Currently absent — the intended red gate until the next base "
        "rebuild carries the stage through."
    ),
}

QBI_INPUTS = (
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
    "sstb_self_employment_income_would_be_qualified",
    "business_is_sstb",
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "sstb_self_employment_income_before_lsr",
    "sstb_unadjusted_basis_qualified_property",
    "sstb_w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)

CHILD_SUPPORT_INPUTS = (
    "child_support_received",
    "child_support_expense",
)

DISABILITY_BENEFITS_INPUTS = ("disability_benefits",)

WORKERS_COMPENSATION_INPUTS = ("workers_compensation",)

WEEKS_UNEMPLOYED_INPUTS = ("weeks_unemployed",)

WIC_CLAIM_INPUTS = ("would_claim_wic",)

EDUCATOR_EXPENSE_INPUTS = ("educator_expense",)

OTHER_HEALTH_INSURANCE_INPUTS = ("other_health_insurance_premiums",)

PRIOR_YEAR_INCOME_INPUTS = (
    "self_employment_income_last_year",
    "previous_year_income_available",
)

FARM_BUSINESS_INCOME_INPUTS = (
    "farm_operations_income",
    "farm_rent_income",
)

SIPP_VEHICLE_INPUTS = (
    "household_vehicles_owned",
    "household_vehicles_value",
)

VOLUNTARY_FILING_INPUTS = ("would_file_taxes_voluntarily",)

SSI_TAKE_UP_INPUTS = ("takes_up_ssi_if_eligible",)

HEAD_START_INPUTS = ("takes_up_head_start_if_eligible",)

SCF_NET_WORTH_INPUTS = ("net_worth",)

FORM_4952_INPUTS = ("investment_income_elected_form_4952",)

CAPITAL_GAIN_DETAIL_INPUTS = (
    "long_term_capital_gains_on_collectibles",
    "unrecaptured_section_1250_gain",
)

SALT_REFUND_INPUTS = ("salt_refund_income",)

ENERGY_SUBSIDY_INPUTS = ("spm_unit_energy_subsidy",)

RELATIONSHIP_INPUTS = (
    "is_household_head",
    "is_separated",
    "is_surviving_spouse",
)

HOUSING_INPUTS = (
    "pre_subsidy_rent",
    "receives_housing_assistance",
    "takes_up_housing_assistance_if_eligible",
    "spm_unit_tenure_type",
    "tenure_type",
)

RETIREMENT_DISTRIBUTION_INPUTS = (
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "tax_exempt_ira_distributions",
    "taxable_ira_distributions",
    "keogh_distributions",
    "taxable_sep_distributions",
)

# Reference-populated inputs whose primary-source restoration has shipped.
# They remain hard requirements even if a stale parity-gap entry is
# accidentally reintroduced later.
RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS = (
    "alimony_expense",
    "alimony_income",
    "casualty_loss",
    *CHILD_SUPPORT_INPUTS,
    *DISABILITY_BENEFITS_INPUTS,
    *WORKERS_COMPENSATION_INPUTS,
    *WEEKS_UNEMPLOYED_INPUTS,
    *WIC_CLAIM_INPUTS,
    *EDUCATOR_EXPENSE_INPUTS,
    *OTHER_HEALTH_INSURANCE_INPUTS,
    *FARM_BUSINESS_INCOME_INPUTS,
    *SCF_NET_WORTH_INPUTS,
    *SIPP_VEHICLE_INPUTS,
    *VOLUNTARY_FILING_INPUTS,
    *HEAD_START_INPUTS,
    *SSI_TAKE_UP_INPUTS,
    *FORM_4952_INPUTS,
    *CAPITAL_GAIN_DETAIL_INPUTS,
    *SALT_REFUND_INPUTS,
    *ENERGY_SUBSIDY_INPUTS,
    *RELATIONSHIP_INPUTS,
    *HOUSING_INPUTS,
    *PRIOR_YEAR_INCOME_INPUTS,
    *RETIREMENT_DISTRIBUTION_INPUTS,
    "domestic_production_ald",
    "household_weight",
    "investment_interest_expense",
    "spm_unit_pre_subsidy_childcare_expenses",
    "unreimbursed_business_employee_expenses",
    *QBI_INPUTS,
)

RETIREMENT_CONTRIBUTION_INPUTS = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)

AOTC_EDUCATION_INPUTS = (
    "qualified_tuition_expenses",
    "is_pursuing_credential_for_american_opportunity_credit",
    "attends_eligible_educational_institution_for_american_opportunity_credit",
    "is_enrolled_at_least_half_time_for_american_opportunity_credit",
    "has_american_opportunity_credit_1098_t_or_exception",
    "has_american_opportunity_credit_institution_ein",
)

#: Pinned reform-coverage probes. Raising the SSI resource limit from
#: the 2024 statutory $2,000 individual / $3,000 couple to $10,000 / $20,000 is
#: a pure relaxation that binds only through ``ssi_countable_resources``. Nonzero
#: iff the asset inputs are restored. Dense-native reference magnitudes: +$1.6B
#: at $10k/$20k and ~+$16.1B with no limit (microcosm #356). ``min_abs_effect`` is
#: a floor far below the plausible effect but far above simulation noise, so a
#: structural $0 fails while a real (even conservative) score passes.
REFORM_COVERAGE_PROBES = [
    {
        "id": "prior_year_self_employment_neutralization",
        "name": "Prior-year self-employment support neutralization",
        "parameter_changes": {},
        "neutralized_variable": "self_employment_income_last_year",
        "budget_measure": "tax_unit_earned_income_last_year",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["self_employment_income_last_year"],
        "min_abs_effect": 1_000_000_000.0,
        "reason": (
            "PolicyEngine-US adds self_employment_income_last_year to "
            "earned_income_last_year and then aggregates it over tax-unit "
            "nondependents. The Wyden-Smith ACTC lookback is the downstream "
            "policy consumer, but its parameter also reads formula-owned "
            "prior-year wages, so neutralizing this one leaf is the unique "
            "coverage probe. The SHA-locked strict equal-share 2022-2024 ASEC "
            "pool carries $357.24 billion of weighted net source amount; "
            "without the adjacent-year carry the neutralization is a "
            "structural zero. The distinct "
            "previous_year_income_available flag has no formula consumer in "
            "PolicyEngine-US 1.764.6 and remains protected by the hard "
            "non-default column gate."
        ),
        "issue": "PolicyEngine/microcosm#38",
    },
    {
        "id": "keogh_distribution_neutralization",
        "name": "Keogh distribution neutralization",
        "parameter_changes": {},
        "neutralized_variable": "keogh_distributions",
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["keogh_distributions"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US includes keogh_distributions directly in taxable "
            "retirement distributions and federal gross income. Neutralizing "
            "only the measured code-5 ASEC leaf lowers federal income tax. On "
            "the 865,046-person staged Build-J support, the locked ASEC source "
            "carries $148.97 million of weighted Keogh distributions and the "
            "baseline-minus-neutralized income-tax effect is +$24.47 million. "
            "Without the restored DST_SC*/DST_VAL* mapping the effect is a "
            "structural zero."
        ),
        "issue": "PolicyEngine/microcosm#38",
    },
    {
        "id": "tip_income_neutralization",
        "name": "Tip income neutralization",
        "parameter_changes": {},
        "neutralized_variable": "tip_income",
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["tip_income"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "PolicyEngine-US treats W-2 wages as already containing tips; "
            "tip_income is the attribution column whose only federal tax "
            "channel is the OBBBA qualified-tips deduction (2025-2028), so "
            "the probe runs at 2026 law and the sign is negative: "
            "neutralizing the column removes the deduction and raises "
            "reform-side tax. Measured on certified Build N (549 carriers, "
            "$34.28 billion weighted): baseline-minus-reform -$1.63 "
            "billion. External class: IRS SOI W-2 Table 4.B Box 7 $26.79 "
            "billion (TY2020); JCT JCX-35-25 no-tax-on-tips FY2026 "
            "-$10.121 billion (ledger fact jct.obbba_title_vii.fy2026."
            "no_tax_on_tips.revenue_effect); Treasury filing-season claims "
            "over 7.5 million filers averaging over $7,000 (sb0517, June "
            "2026). A structural zero means the tip attribution column was "
            "dropped or the deduction channel broke."
        ),
        "issue": "PolicyEngine/microcosm#451",
    },
    {
        "id": "fsla_overtime_premium_neutralization",
        "name": "FLSA overtime premium neutralization",
        "parameter_changes": {},
        "neutralized_variable": "fsla_overtime_premium",
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["fsla_overtime_premium"],
        "min_abs_effect": 4_000_000_000.0,
        "reason": (
            "PolicyEngine-US treats W-2 wages as already containing "
            "overtime; fsla_overtime_premium is the attribution column "
            "whose only federal tax channel is the OBBBA qualified-overtime "
            "deduction (2025-2028), so the probe runs at 2026 law with a "
            "negative expected sign. Measured on certified Build N (8,748 "
            "carriers, $114.79 billion weighted): baseline-minus-reform "
            "-$16.86 billion. External anchors: JCT JCX-35-25 "
            "no-tax-on-overtime FY2026 -$32.806 billion (ledger fact "
            "jct.obbba_title_vii.fy2026.no_tax_on_overtime.revenue_effect "
            "— the only exact-valued official overtime anchor: no BLS or "
            "Census FLSA-premium aggregate exists); Treasury filing-season "
            "claims over 29 million filers averaging over $3,100, an "
            "approximately $90 billion claimed floor (sb0517, June 2026). "
            "A structural zero means the attribution column was dropped or "
            "the deduction channel broke."
        ),
        "issue": "PolicyEngine/microcosm#451",
    },
    {
        "id": "household_head_childcare_cap_neutralization",
        "name": "Household-head childcare earned-cap neutralization",
        "parameter_changes": {},
        "neutralized_variable": "is_household_head",
        "budget_measure": "spm_unit_capped_work_childcare_expenses",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["is_household_head"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US uses is_household_head to identify whose earnings "
            "cap SPM work-childcare expenses when an SPM unit contains multiple "
            "tax units. Neutralizing only the measured head flag falls back to "
            "tax-unit roles and changes the cap. On the 865,046-person staged "
            "Build-J artifact, baseline-minus-neutralized capped expenses are "
            "-$265.50 million. Without the restored P_SEQ input the "
            "neutralization is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#38",
    },
    {
        "id": "self_employed_health_premium_neutralization",
        "name": "Self-employed health premium neutralization",
        "parameter_changes": {},
        "neutralized_variable": "health_insurance_premiums",
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["health_insurance_premiums", "is_self_employed"],
        "min_abs_effect": 300_000_000.0,
        "reason": (
            "PolicyEngine-US computes the section 162(l) self-employed health "
            "ALD as min(total_self_employment_income, "
            "self_employed_health_insurance_premiums), where the premium "
            "aggregation adds the person health_insurance_premiums input under "
            "the is_self_employed gate — the two leaves the deterministic "
            "attribution operation of the other_health_insurance_premiums "
            "release stage populates for strictly-positive Schedule C people "
            "outside the Medicare proxy and outside measured employer-"
            "sponsored coverage (the 162(l)(2)(B) subsidized-plan exclusion "
            "proxy). The Medicare-proxy guard keeps the statutory "
            "medical-expense premium concept numerically invariant, so this "
            "neutralization isolates exactly the ALD channel on federal "
            "income tax. Measured on the certified Build N frame "
            "(c3e378a-20260722T010408Z, seed 0): $16.37 billion attributed to "
            "1,402 carrier rows (3.57 million weighted people, against the "
            "SOI Pub 1304 Table 1.4 TY2023 fact of 3,595,764 returns / "
            "$31.23 billion, ledger#105, buildn v9.2 feed), baseline ALD "
            "$11.97 billion, baseline-minus-reform -$1.45 billion; the "
            "level gap to the banked SOI fact is the calibration solve's to "
            "close over this support. A structural zero means the attribution "
            "operation was dropped or the defined_for gate never opened."
        ),
        "issue": "PolicyEngine/microcosm#451",
    },
    {
        "id": "cdcc_adult_care_expense_neutralization",
        "name": "CDCC adult-care expense neutralization",
        "parameter_changes": {},
        "neutralized_variable": "pre_subsidy_care_expenses",
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["is_incapable_of_self_care", "pre_subsidy_care_expenses"],
        "min_abs_effect": 30_000_000.0,
        "reason": (
            "PolicyEngine-US sums pre_subsidy_care_expenses (via care_expenses) "
            "into cdcc_relevant_expenses as the section 21 adult/disabled-"
            "dependent care leg; the is_incapable_of_self_care flag supplies "
            "the 21(b)(1) qualifying individuals and the 21(d)(2) spouse "
            "deeming, and stays active in both arms so the neutralization "
            "isolates exactly the dollar leg. Measured on the certified "
            "Build N frame (c3e378a-20260722T010408Z, seed 0) after the "
            "adult_care_inputs stage: 2,067 measured PEDISDRS carriers "
            "(5.998 million weighted, a 1.76% person share consistent with the "
            "published self-care difficulty prevalence), $3.41 billion of "
            "donor-matched expenses on 137 carrier rows, baseline-minus-reform "
            "-$153.9 million on federal income tax. A structural zero means "
            "the base rebuild dropped the stage or the CDCC adult-care channel "
            "broke."
        ),
        "issue": "PolicyEngine/microcosm#451",
    },
    {
        "id": "spm_unit_energy_subsidy_neutralization",
        "name": "SPM energy-subsidy neutralization",
        "parameter_changes": {},
        "neutralized_variable": "spm_unit_energy_subsidy",
        "budget_measure": "spm_unit_benefits",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["spm_unit_energy_subsidy"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "PolicyEngine-US adds this measured LIHEAP resource dollar-for-dollar "
            "to spm_unit_benefits. Neutralizing the leaf must therefore lower "
            "benefits by its weighted source mass; without the restored "
            "SPM_ENGVAL carry, the effect is a structural zero. No OBBBA "
            "provision consumes this SPM resource, so the direct neutralization "
            "is the uniquely isolating policy-engine probe."
        ),
        "issue": "PolicyEngine/microcosm#32",
    },
    {
        "id": "medicare_take_up_neutralization",
        "name": "Measured Medicare enrollment neutralization",
        "parameter_changes": {},
        "neutralized_variable": "takes_up_medicare_if_eligible",
        "budget_measure": "medicare_cost",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["takes_up_medicare_if_eligible"],
        "min_abs_effect": 1_000_000_000.0,
        "reason": (
            "PolicyEngine-US computes medicare_enrolled from the measured "
            "takes_up_medicare_if_eligible leaf and modeled eligibility, then "
            "gates Medicare costs on enrollment. Neutralizing only the "
            "restored MCARE == 1 leaf must reduce aggregate Medicare cost; "
            "without the measured carry the probe is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#312",
    },
    {
        "id": "ssi_asset_limit_10k_20k",
        "name": "SSI asset limits raised to $10k individual / $20k couple",
        "parameter_changes": {
            "gov.ssa.ssi.eligibility.resources.limit.individual": {
                "2024-01-01.2100-12-31": 10_000
            },
            "gov.ssa.ssi.eligibility.resources.limit.couple": {
                "2024-01-01.2100-12-31": 20_000
            },
        },
        "budget_measure": "ssi",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": list(SSI_COUNTABLE_RESOURCE_ASSETS),
        "min_abs_effect": 1_000_000_000.0,
        "reason": (
            "Raising the SSI resource limit is a pure relaxation that only "
            "changes who passes meets_ssi_resource_test, which compares "
            "ssi_countable_resources = bank_account_assets + stock_assets + "
            "bond_assets against the limit. With those asset inputs absent, "
            "countable resources are 0 for every record, everyone already "
            "passes the resource test, and raising the limit scores exactly "
            "$0. Dense-native reference: +$1.6B at $10k/$20k, ~+$16.1B with no "
            "limit (PolicyEngine/microcosm#356)."
        ),
        "issue": "PolicyEngine/microcosm#356",
    },
    {
        "id": "ssi_disability_criteria_neutralization",
        "name": "SSI disability-criteria neutralization",
        "parameter_changes": {},
        "neutralized_variable": "meets_ssi_disability_criteria",
        "budget_measure": "ssi",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["meets_ssi_disability_criteria"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "PolicyEngine-US requires the person-level disability criterion "
            "for non-aged SSI eligibility. Neutralizing only the restored "
            "SIPP-imputed criterion must therefore remove SSI from otherwise "
            "eligible disabled or blind people, so baseline-minus-neutralized "
            "SSI is positive. If the post-reference exported input is absent "
            "or degenerate, this isolated eligibility channel scores exactly "
            "$0. A deterministic 6,000-source-household Build-J smoke with "
            "the pinned SIPP and SCF donors scored +$586.393 million "
            "baseline-minus-neutralized SSI; the $100 million floor retains "
            "ample sampling margin while rejecting a materially weakened "
            "criterion channel."
        ),
        "issue": "PolicyEngine/microcosm#312",
    },
    {
        "id": "ssi_take_up_neutralization",
        "name": "SSI take-up neutralization",
        "parameter_changes": {},
        "neutralized_variable": "takes_up_ssi_if_eligible",
        "budget_measure": "ssi",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["takes_up_ssi_if_eligible"],
        "min_abs_effect": 10_000_000_000.0,
        "reason": (
            "PolicyEngine-US gates SSI benefits on the restored person-level "
            "take-up leaf after eligibility. Neutralizing only that leaf must "
            "therefore remove SSI from source-reported and SSA-count-calibrated "
            "recipients, so baseline-minus-neutralized SSI is positive. If the "
            "restored exported input is absent, all false, or not persisted, "
            "this isolated channel scores exactly $0. A production-ingredient "
            "sparse smoke (staged artifact sha256 c5939dad81153da51b2cc57081"
            "ddb3e729700366144868df742b3ad86eafcd7c; restored artifact sha256 "
            "9269360d3409fdc15c90c43dda394ada6c91eff5bb64c12ccd9def7d670dd077) "
            "measured +$57,114,569,526.38 of baseline-minus-neutralized 2024 "
            "SSI. The $10 billion floor retains over 5.7x observed margin while "
            "rejecting a materially degenerate persisted flag."
        ),
        "issue": "PolicyEngine/microcosm#312",
    },
    {
        "id": "head_start_take_up_neutralization",
        "name": "Measured Head Start take-up neutralization",
        "parameter_changes": {},
        "neutralized_variable": "takes_up_head_start_if_eligible",
        "budget_measure": "head_start",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": list(HEAD_START_INPUTS),
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "PolicyEngine-US gates Head Start benefits on the person-level "
            "take-up leaf after modeled age, income, and categorical "
            "eligibility. Neutralizing only the restored SIPP response model "
            "must therefore remove Head Start from measured-proxy recipients, "
            "so baseline-minus-neutralized Head Start is positive. If the "
            "restored export is absent, all false, or not persisted, this "
            "isolated channel scores exactly $0. A production-ingredient sparse "
            "smoke (staged artifact sha256 "
            "67ad74b9ad9222ed342a0279dfc8175e872966fa59f86aeecb7fad52021ba500) "
            "measured +$4,575,181,976.69 of baseline-minus-neutralized 2024 "
            "Head Start. The $100 million floor retains over 45x observed "
            "margin while remaining far above numerical noise."
        ),
        "issue": "PolicyEngine/microcosm#312",
    },
    {
        "id": "aotc_abolition",
        "name": "American Opportunity Tax Credit abolition",
        "parameter_changes": {
            "gov.irs.credits.education.american_opportunity_credit.abolition": {
                "2024-01-01.2100-12-31": True
            }
        },
        "budget_measure": "american_opportunity_credit",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": list(AOTC_EDUCATION_INPUTS),
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Abolishing the American Opportunity Tax Credit sets the credit "
            "to zero, so baseline-minus-reform AOTC must be positive. With "
            "qualified tuition or any of the five affirmative AOTC factual "
            "inputs absent or degenerate, the baseline credit is a structural "
            "zero and the abolition scores exactly $0."
        ),
        "issue": "PolicyEngine/microcosm#253",
    },
    {
        "id": "savers_credit_abolition",
        "name": "Retirement Saver's Credit abolition",
        "parameter_changes": {
            "gov.irs.credits.retirement_saving.contributions_cap": {
                "2024-01-01.2100-12-31": 0
            }
        },
        "budget_measure": "savers_credit",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": list(RETIREMENT_CONTRIBUTION_INPUTS),
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting the Saver's Credit contribution cap to zero abolishes "
            "the credit, so baseline-minus-reform Saver's Credit must be "
            "positive. PolicyEngine-US builds qualified contributions from "
            "the realized forms of all five desired retirement-contribution "
            "inputs. If the desired-input family is absent or degenerate, "
            "the baseline credit is a structural zero and abolition scores $0."
        ),
        "issue": "PolicyEngine/microcosm#278",
    },
    {
        "id": "qbi_reit_ptp_rate_abolition",
        "name": "Section 199A qualified REIT/PTP component abolition",
        "parameter_changes": {
            "gov.irs.deductions.qbi.max.reit_ptp_rate": {"2024-01-01.2100-12-31": 0}
        },
        "budget_measure": "qualified_business_income_deduction",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["qualified_reit_and_ptp_income"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Setting only the qualified REIT/PTP component rate to zero "
            "removes that component from the Section 199A deduction, so "
            "baseline-minus-reform QBID must be positive. Without populated "
            "qualified_reit_and_ptp_income the change is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#298",
    },
    {
        "id": "qbi_wage_property_guardrails_zeroed",
        "name": "Section 199A W-2 wage and UBIA guardrails zeroed",
        "parameter_changes": {
            "gov.irs.deductions.qbi.max.w2_wages.rate": {"2024-01-01.2100-12-31": 0},
            "gov.irs.deductions.qbi.max.w2_wages.alt_rate": {
                "2024-01-01.2100-12-31": 0
            },
            "gov.irs.deductions.qbi.max.business_property.rate": {
                "2024-01-01.2100-12-31": 0
            },
        },
        "budget_measure": "qualified_business_income_deduction",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": [
            "w2_wages_from_qualified_business",
            "unadjusted_basis_qualified_property",
        ],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Zeroing all W-2 wage and UBIA cap rates tightens the Section "
            "199A deduction for high-income qualified businesses, so "
            "baseline-minus-reform QBID must be positive. If the total W-2 "
            "and UBIA inputs are absent, both baseline and reform guardrails "
            "are zero and the change is a structural zero. The archived "
            "all-or-nothing SSTB routing leaves its SSTB-allocable copies to "
            "the hard signal gate rather than overclaiming reform coverage."
        ),
        "issue": "PolicyEngine/microcosm#298",
    },
    {
        "id": "qbi_farm_operations_income_exclusion",
        "name": "Exclude farm-operations income from Section 199A QBI",
        "parameter_changes": {
            "gov.irs.deductions.qbi.income_definition": {
                "2026-01-01.2026-12-31": [
                    "self_employment_income",
                    "partnership_s_corp_income",
                    "farm_rent_income",
                    "rental_income",
                    "estate_income",
                ]
            }
        },
        "budget_measure": "qualified_business_income_deduction",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "either",
        "binding_inputs": ["farm_operations_income"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only farm_operations_income from the 2026 Section 199A "
            "income definition isolates the restored Schedule F leaf. The "
            "leaf is signed and two-channel: the ASEC channel carries the "
            "measured FRSE farm self-employment values (net-positive in the "
            "pooled frame) and the PUF channel carries the donor-pinned "
            "signed Schedule F values (net-negative, loss-heavy, "
            "microcosm#435), so the aggregate QBID effect direction is a "
            "property of the frame mix and the solve, not of coverage. The "
            "probe therefore requires a binding effect of at least the floor "
            "in either direction; a structural zero still fails."
        ),
        "issue": "PolicyEngine/microcosm#298",
    },
    {
        "id": "qbi_farm_rent_income_exclusion",
        "name": "Exclude farm-rent income from Section 199A QBI",
        "parameter_changes": {
            "gov.irs.deductions.qbi.income_definition": {
                "2026-01-01.2026-12-31": [
                    "self_employment_income",
                    "partnership_s_corp_income",
                    "farm_operations_income",
                    "rental_income",
                    "estate_income",
                ]
            }
        },
        "budget_measure": "qualified_business_income_deduction",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["farm_rent_income"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only farm_rent_income from the 2026 Section 199A income "
            "definition isolates the restored signed E27200 leaf. The real "
            "staged candidate produces +$9.14M baseline-minus-reform QBID. "
            "Without farm_rent_income the reform is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#298",
    },
    {
        "id": "domestic_production_ald_reactivation",
        "name": "Former Section 199 domestic-production deduction reactivation",
        "parameter_changes": {
            "gov.irs.ald.deductions": {
                "2024-01-01.2024-12-31": [
                    "loss_ald",
                    "self_employment_tax_ald",
                    "student_loan_interest_ald",
                    "early_withdrawal_penalty",
                    "alimony_expense_ald",
                    "educator_expense",
                    "health_savings_account_ald",
                    "self_employed_health_insurance_ald",
                    "self_employed_pension_contribution_ald",
                    "traditional_ira_contributions",
                    "qualified_adoption_assistance_expense",
                    "us_bonds_for_higher_ed",
                    "specified_possession_income",
                    "puerto_rico_income",
                    "domestic_production_ald",
                ]
            }
        },
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["domestic_production_ald"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US 1.764.6 excludes the former Section 199 deduction "
            "from current-law above-the-line deductions. This probe preserves "
            "the exact 2024 list and adds only domestic_production_ald, so "
            "baseline-minus-reform income tax must be positive. Without the "
            "restored E03240 input, reactivation is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#298",
    },
    {
        "id": "form_4952_election_neutralization",
        "name": "Form 4952 elected investment income neutralization",
        "parameter_changes": {},
        "neutralized_variable": "investment_income_elected_form_4952",
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["investment_income_elected_form_4952"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US subtracts the tax-unit sum of "
            "investment_income_elected_form_4952 from net capital gain. "
            "Neutralizing only that leaf increases preferential net capital "
            "gain and lowers income tax, so baseline-minus-reform income tax "
            "must be positive. Without the restored E58990 input the "
            "neutralization is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#274",
    },
    {
        "id": "salt_refund_income_neutralization",
        "name": "State and local tax refund income neutralization",
        "parameter_changes": {},
        "neutralized_variable": "salt_refund_income",
        "budget_measure": "state_income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["salt_refund_income"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US 1.764.6 includes salt_refund_income in the "
            "South Carolina, Idaho, and West Virginia subtraction lists. "
            "Neutralizing only that leaf removes the state subtraction and "
            "raises state income tax, so baseline-minus-reform state income "
            "tax must be negative. Without the restored E00700 input the "
            "neutralization is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#38",
    },
    {
        "id": "collectibles_gain_neutralization",
        "name": "Long-term collectibles gain neutralization",
        "parameter_changes": {},
        "neutralized_variable": "long_term_capital_gains_on_collectibles",
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["long_term_capital_gains_on_collectibles"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US includes collectibles in capital_gains_28_percent_"
            "rate_gain. Neutralizing only the E24518 memo leaf reclassifies "
            "those gains from the special 28-percent bucket to the ordinary "
            "preferential capital-gain schedule, so baseline-minus-reform "
            "income tax must be positive. Without the restored leaf the "
            "neutralization is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#274",
    },
    {
        "id": "unrecaptured_section_1250_gain_neutralization",
        "name": "Unrecaptured section 1250 gain neutralization",
        "parameter_changes": {},
        "neutralized_variable": "unrecaptured_section_1250_gain",
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["unrecaptured_section_1250_gain"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US taxes the E24515 memo leaf at the special "
            "unrecaptured-section-1250 rate. Neutralizing only that leaf "
            "reclassifies the same net gain onto the ordinary preferential "
            "capital-gain schedule, so baseline-minus-reform income tax must "
            "be positive. Without the restored leaf the neutralization is a "
            "structural zero."
        ),
        "issue": "PolicyEngine/microcosm#274",
    },
    {
        "id": "child_support_received_snap_exclusion",
        "name": "Exclude child-support receipts from SNAP unearned income",
        "parameter_changes": {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "disability_benefits",
                    "workers_compensation",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        },
        "budget_measure": "snap",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": ["child_support_received"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only child_support_received from SNAP unearned-income "
            "sources lowers countable income and must increase SNAP for some "
            "recipients. Without the measured/QRF child-support receipt leaf, "
            "the source-list reform is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#32",
    },
    {
        "id": "child_support_expense_snap_deduction_abolition",
        "name": "Abolish the SNAP child-support expense deduction",
        "parameter_changes": {
            "gov.usda.snap.income.deductions.allowed": {
                "2024-01-01.2024-12-31": [
                    "snap_standard_deduction",
                    "snap_earned_income_deduction",
                    "snap_dependent_care_deduction",
                    "snap_excess_medical_expense_deduction",
                    "snap_excess_shelter_expense_deduction",
                ]
            }
        },
        "budget_measure": "snap",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["child_support_expense"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only snap_child_support_deduction raises countable net "
            "income and must reduce SNAP in states that take the expense as a "
            "net-income deduction. Without the measured/QRF positive expense "
            "leaf, abolition is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#32",
    },
    {
        "id": "disability_benefits_snap_exclusion",
        "name": "Exclude disability benefits from SNAP unearned income",
        "parameter_changes": {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "workers_compensation",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "child_support_received",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        },
        "budget_measure": "snap",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": ["disability_benefits"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only disability_benefits from SNAP unearned-income "
            "sources lowers countable income and must increase SNAP for some "
            "recipients. Without the measured/QRF non-workers-compensation "
            "benefit leaf, the source-list reform is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#38",
    },
    {
        "id": "workers_compensation_snap_exclusion",
        "name": "Exclude workers' compensation from SNAP unearned income",
        "parameter_changes": {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "disability_benefits",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "child_support_received",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        },
        "budget_measure": "snap",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": ["workers_compensation"],
        "min_abs_effect": 10_000_000.0,
        "reason": (
            "Removing only workers_compensation from SNAP unearned-income "
            "sources lowers countable income and must increase SNAP for some "
            "recipients. A production-ingredient 30,000-household smoke scored "
            "+$28.26M reform-minus-baseline; without the measured WC_VAL carry "
            "and PUF-half QRF, the source-list reform is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#32",
    },
    {
        "id": "wic_claim_neutralization",
        "name": "WIC claim neutralization",
        "parameter_changes": {},
        "neutralized_variable": "would_claim_wic",
        "budget_measure": "wic",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["would_claim_wic"],
        "min_abs_effect": 25_000_000.0,
        "reason": (
            "PolicyEngine-US multiplies each eligible person's monthly WIC "
            "food package by would_claim_wic. A 6,000-household production-"
            "ingredient smoke with the FNS category-rate stage scored "
            "+$57.19M baseline-minus-neutralized; without the restored claim "
            "surface the probe is a structural zero."
        ),
        "issue": "PolicyEngine/microcosm#312",
    },
    {
        "id": "educator_expense_ald_abolition",
        "name": "Abolish the educator-expense above-the-line deduction",
        "parameter_changes": {
            "gov.irs.ald.deductions": {
                "2024-01-01.2024-12-31": [
                    "loss_ald",
                    "self_employment_tax_ald",
                    "student_loan_interest_ald",
                    "early_withdrawal_penalty",
                    "alimony_expense_ald",
                    "health_savings_account_ald",
                    "self_employed_health_insurance_ald",
                    "self_employed_pension_contribution_ald",
                    "traditional_ira_contributions",
                    "qualified_adoption_assistance_expense",
                    "us_bonds_for_higher_ed",
                    "specified_possession_income",
                    "puerto_rico_income",
                ]
            }
        },
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": ["educator_expense"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only educator_expense from the above-the-line deduction "
            "source list raises taxable income and must increase income tax for "
            "some filers. Without the restored PUF E03220 leaf, abolition is a "
            "structural zero."
        ),
        "issue": "PolicyEngine/microcosm#32",
    },
    {
        "id": "alimony_expense_ald_abolition",
        "name": "Alimony expense above-the-line deduction abolition",
        "parameter_changes": {
            "gov.irs.ald.alimony_expense.divorce_year_threshold[0].amount": {
                "2024-01-01.2100-12-31": False
            }
        },
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["alimony_expense"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "The retired export has no nondefault divorce_year input, so "
            "PolicyEngine-US applies its default year 0 through the first "
            "eligibility bracket. Setting that bracket's amount to false "
            "abolishes the alimony-expense above-the-line deduction on the "
            "release, so baseline-minus-reform income tax must be negative. "
            "With alimony_expense absent or degenerate, the abolition scores "
            "exactly $0."
        ),
        "issue": "PolicyEngine/microcosm#38",
    },
    {
        "id": "obbba_casualty_loss_limit",
        "name": "OBBBA casualty-loss deduction reactivation",
        "parameter_changes": {
            "gov.irs.deductions.itemized.casualty.active": {
                "2026-01-01.2026-12-31": True
            }
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["casualty_loss"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Reactivating the casualty-loss deduction lowers income tax only "
            "for tax units with casualty_loss above the statutory AGI floor, "
            "so baseline-minus-reform income tax must be positive. With the "
            "casualty-loss input absent or degenerate, the reactivation scores "
            "exactly $0."
        ),
        "issue": "PolicyEngine/microcosm#32",
    },
    {
        "id": "obbba_misc_itemized_deductions",
        "name": "OBBBA miscellaneous-itemized deduction reactivation",
        "parameter_changes": {
            "gov.irs.deductions.itemized.misc.applies": {"2026-01-01.2026-12-31": True}
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["unreimbursed_business_employee_expenses"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Reactivating the miscellaneous itemized deduction lowers income "
            "tax only for tax units with qualifying expenses above its AGI "
            "floor, so baseline-minus-reform income tax must be positive. "
            "Without unreimbursed_business_employee_expenses, the retired "
            "pipeline's only populated miscellaneous-expense input, the "
            "reactivation is a structural zero on the export."
        ),
        "issue": "PolicyEngine/microcosm#32",
    },
    {
        "id": "obbba_cdcc",
        "name": "OBBBA Child and Dependent Care Credit reversion",
        "parameter_changes": {
            "gov.irs.credits.cdcc.phase_out.max": {"2026-01-01.2026-12-31": 0.35},
            "gov.irs.credits.cdcc.phase_out.min": {"2026-01-01.2026-12-31": 0.2},
            "gov.irs.credits.cdcc.phase_out.amended_structure.applies": {
                "2026-01-01.2026-12-31": False
            },
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["spm_unit_pre_subsidy_childcare_expenses"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Reverting the OBBBA CDCC enhancement raises income tax, so "
            "baseline-minus-reform income tax must be negative. With measured "
            "pre-subsidy childcare expenses absent or degenerate, no filer has "
            "qualifying care expenses and the reversion scores exactly $0."
        ),
        "issue": "PolicyEngine/microcosm#278",
    },
    {
        "id": "obbba_no_tax_on_tips",
        "name": "OBBBA no-tax-on-tips deduction",
        "parameter_changes": {
            "gov.irs.deductions.tip_income.cap": {"2026-01-01.2026-12-31": 0}
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": [
            "tip_income",
            "treasury_tipped_occupation_code",
        ],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting the OBBBA tip-deduction cap to zero removes the deduction, "
            "so baseline-minus-reform income tax must be negative in 2026. "
            "With tip_income or the Treasury tipped-occupation code absent, "
            "qualified tip income is zero and the repeal scores exactly $0. "
            "External anchor: JCT JCX-35-25 no-tax-on-tips FY2026 -$10.121 "
            "billion (ledger fact jct.obbba_title_vii.fy2026.no_tax_on_tips."
            "revenue_effect); certified Build N measures -$1.63 billion."
        ),
        "issue": "PolicyEngine/microcosm#38",
    },
    {
        "id": "obbba_no_tax_on_overtime",
        "name": "OBBBA no-tax-on-overtime deduction",
        "parameter_changes": {
            "gov.irs.deductions.overtime_income.cap.JOINT": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SINGLE": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.HEAD_OF_HOUSEHOLD": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SURVIVING_SPOUSE": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SEPARATE": {
                "2026-01-01.2026-12-31": 0
            },
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["fsla_overtime_premium"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting every OBBBA overtime-deduction cap to zero removes the "
            "deduction, so reform income tax rises and baseline-minus-reform "
            "must be negative in 2026. With fsla_overtime_premium absent or "
            "degenerate, qualified overtime is zero and the repeal scores $0. "
            "External anchor: JCT JCX-35-25 no-tax-on-overtime FY2026 "
            "-$32.806 billion (ledger fact jct.obbba_title_vii.fy2026."
            "no_tax_on_overtime.revenue_effect); certified Build N measures "
            "-$16.86 billion."
        ),
        "issue": "PolicyEngine/microcosm#242",
    },
    {
        "id": "obbba_auto_loan_interest",
        "name": "OBBBA no-tax-on-auto-loan-interest deduction",
        "parameter_changes": {
            "gov.irs.deductions.auto_loan_interest.cap": {"2026-01-01.2026-12-31": 0}
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["qualified_passenger_vehicle_loan_interest"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting the OBBBA auto-loan-interest deduction cap to zero "
            "removes the deduction, so reform income tax rises and "
            "baseline-minus-reform must be negative in 2026. With qualified "
            "passenger-vehicle loan interest absent or degenerate, the repeal "
            "scores exactly $0."
        ),
        "issue": "PolicyEngine/microcosm#252",
    },
    {
        "id": "tx_snap_additional_vehicle_exemption_abolition",
        "name": "Texas SNAP additional-vehicle exemption abolition",
        "parameter_changes": {
            "gov.hhs.tanf.non_cash.tx_additional_vehicle_exemption": {
                "2026-01-01.2100-12-31": 0
            }
        },
        "budget_measure": "snap",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": list(SIPP_VEHICLE_INPUTS),
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Setting the Texas TANF non-cash additional-vehicle exemption "
            "to zero tightens the asset test used by Texas SNAP categorical "
            "eligibility, so baseline-minus-reform SNAP must be positive. "
            "PolicyEngine-US computes the exemption from both household "
            "vehicle count and value; if either restored SIPP vehicle input "
            "is absent or degenerate, this vehicle-specific reform loses its "
            "intended binding channel. A persisted 30,000-household Microcosm "
            "smoke scored +$3.58 million in 2026 SNAP."
        ),
        "issue": "PolicyEngine/microcosm#49",
    },
    {
        "id": "voluntary_filing_aca_ptc_neutralization",
        "name": "Voluntary tax filing ACA PTC neutralization",
        "parameter_changes": {},
        "neutralized_variable": "would_file_taxes_voluntarily",
        "budget_measure": "aca_ptc",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": list(VOLUNTARY_FILING_INPUTS),
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "PolicyEngine-US includes would_file_taxes_voluntarily in "
            "tax_unit_is_filer alongside required and credit filers. "
            "Neutralizing only the restored SIPP filing-response leaf therefore "
            "removes ACA premium tax credits from otherwise eligible voluntary "
            "filers, so baseline-minus-neutralized aca_ptc must be positive. "
            "With the filing leaf absent or degenerate, this isolated response "
            "channel scores exactly $0."
        ),
        "issue": "PolicyEngine/microcosm#312",
    },
    {
        "id": "pre_subsidy_rent_neutralization",
        "name": "Pre-subsidy rent neutralization",
        "parameter_changes": {},
        "neutralized_variable": "pre_subsidy_rent",
        "budget_measure": "snap",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["pre_subsidy_rent"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US includes pre_subsidy_rent in the SNAP shelter "
            "deduction. Neutralizing only the restored ACS rent leaf therefore "
            "reduces SNAP; on the pinned retired small eCPS artifact under "
            "PolicyEngine-US 1.764.6, baseline-minus-neutralized SNAP is "
            "+$11.731 billion. Without nondefault ACS rent, that effect is a "
            "structural zero. The "
            "other source-mapped housing leaves are enforced by their exact "
            "ASEC mappings and signal gate; household tenure_type has no "
            "standalone PolicyEngine-US 1.764.6 formula consumer."
        ),
        "issue": "PolicyEngine/microcosm#32",
    },
    {
        "id": "housing_assistance_take_up_neutralization",
        "name": "Measured housing-assistance take-up neutralization",
        "parameter_changes": {},
        "neutralized_variable": "takes_up_housing_assistance_if_eligible",
        "budget_measure": "housing_assistance",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["takes_up_housing_assistance_if_eligible"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "PolicyEngine-US multiplies HUD HAP by the restored SPM-unit "
            "take-up leaf after eligibility. Microcosm keeps that leaf exactly "
            "equal to source-backed housing-assistance receipt, so neutralizing "
            "it must remove the assistance paid to measured/imputed recipients. "
            "A 6,000-household production-ingredient smoke scored $202.795 "
            "million baseline-minus-neutralized; the $100 million floor is "
            "below that observed subset effect but far above numerical noise. "
            "A default-only or absent carry makes the source reconciliation "
            "or this uniquely isolating probe fail."
        ),
        "issue": "PolicyEngine/microcosm#312",
    },
]


def _load(name: str) -> dict:
    return json.loads((US_PACKAGE_DIR / name).read_text(encoding="utf-8"))


def build_manifest() -> dict:
    parity = _load("ecps_parity_reference.json")
    known_gaps = _load("ecps_parity_known_gaps.json")["known_gaps"]

    populated_layers = {
        name for name, share in parity["nonzero_shares"].items() if float(share) > 0.0
    } | set(POST_REFERENCE_ECPS_REQUIRED_INPUTS)
    ssi_assets = set(SSI_COUNTABLE_RESOURCE_ASSETS)

    missing_assets = sorted(ssi_assets - populated_layers)
    if missing_assets:
        raise ValueError(
            "SSI countable-resource asset inputs are not in the reference eCPS "
            f"populated surface, cannot pin them as required: {missing_assets}."
        )
    restored_inputs = set(RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS)
    missing_restored = sorted(restored_inputs - populated_layers)
    if missing_restored:
        raise ValueError(
            "Restored reference inputs are absent from the reference populated "
            f"surface: {missing_restored}."
        )
    stale_restored_gaps = sorted(restored_inputs & set(known_gaps))
    if stale_restored_gaps:
        raise ValueError(
            "Restored reference inputs cannot remain in the parity-gap register: "
            f"{stale_restored_gaps}."
        )

    columns: dict[str, dict] = {}
    for name in sorted(populated_layers):
        if name in known_gaps and name not in ssi_assets:
            entry = known_gaps[name]
            columns[name] = {
                "status": "reviewed_exclusion",
                "reason": str(entry["reason"]),
                "issue": str(entry["issue"]),
            }
        else:
            column = {"status": "required"}
            if name in ssi_assets:
                # These are currently absent; #368 forbids an exclusion so the
                # gate fails until Build J restores them.
                column["note"] = (
                    "SSI countable-resource asset input; required with NO "
                    "reviewed exclusion per PolicyEngine/microcosm#368 so the "
                    "gate fails until the asset stage is restored (Deliverable "
                    "2). Currently absent — this is the intended red gate."
                )
            elif name in POST_REFERENCE_COLUMN_NOTES:
                column["note"] = POST_REFERENCE_COLUMN_NOTES[name]
            columns[name] = column

    required = sorted(n for n, c in columns.items() if c["status"] == "required")
    reviewed = sorted(
        n for n, c in columns.items() if c["status"] == "reviewed_exclusion"
    )

    # Provenance without naming the retired data package. Only the eCPS parity
    # reference, its loader, and its test are allow-listed to name the incumbent
    # (test_us_plan.test_no_incumbent_data_package_references_in_live_tree); this
    # manifest is not, so it records the sha-locked coordinates that DON'T name a
    # package (filename + content sha + revision + vintage) and points at that
    # allow-listed parity reference for the full record. Nothing reads these
    # fields — the gate and the anti-rot check derive the surface from
    # ecps_parity_reference.json directly — so this stays pure documentation.
    parity_source = dict(parity["source"])
    reference = {
        "derived_from": "ecps_parity_reference.json",
        "filename": str(parity_source.get("filename", "")),
        "revision": str(parity_source.get("revision", "")),
        "sha256": str(parity_source.get("sha256", "")),
        "vintage": str(parity_source.get("vintage", "")),
        "period": str(parity_source.get("period", "")),
        "note": (
            "Column surface derived from the populated input layers of the "
            "pinned, sha-verified reference eCPS recorded in "
            "ecps_parity_reference.json (the launch parity contract, "
            "PolicyEngine/microcosm#313) — the single allow-listed record of the "
            "incumbent coordinates. This manifest names no data package."
        ),
    }

    return {
        "schema_version": 1,
        "issue": "PolicyEngine/microcosm#368",
        "description": (
            "Declared full-coverage contract for a US release: every input "
            "column the reference eCPS exports must be persisted as a key with "
            "non-default signal, or carry a reviewed exclusion. Enforced as a "
            "hard release gate (microcosm.build.us_runtime.release_input_"
            "coverage) that generalizes assert_required_us_release_source_"
            "columns from 5 columns to the full eCPS input surface."
        ),
        "reference": reference,
        "derivation": (
            "Required surface = input columns in the pinned, sha-verified "
            "ecps_parity_reference.json populated layers, plus the documented "
            "post-reference fsla_overtime_premium, "
            "qualified_passenger_vehicle_loan_interest, five desired "
            "retirement-contribution inputs, "
            "meets_ssi_disability_criteria required by shipped validation "
            "probes, and the #282 Schedule-D capital-gain-distributions "
            "route leg schedule_d_capital_gain_distributions "
            "(PolicyEngine/microcosm#462). "
            "status='reviewed_exclusion' for ecps_parity_known_gaps.json entries "
            "(reason+issue from that register); EXCEPT every primary-source "
            "restoration pinned by RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS "
            "(including the Section 199A QBI family), and the SSI countable-"
            "resource asset inputs (bank_account_assets, "
            "stock_assets, bond_assets), which are status='required' with NO "
            "exclusion per PolicyEngine/microcosm#368 "
            "so the gate fails on today's artifacts and asset restoration "
            "(Deliverable 2) turns it green. All other populated layers are "
            "'required'. Regenerate with "
            "tools/build_us_release_input_coverage_manifest.py."
        ),
        "counts": {
            "required": len(required),
            "reviewed_exclusion": len(reviewed),
            "total": len(columns),
        },
        "ssi_countable_resource_assets": list(SSI_COUNTABLE_RESOURCE_ASSETS),
        "columns": columns,
        "reform_coverage_probes": REFORM_COVERAGE_PROBES,
    }


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {MANIFEST_PATH} — {manifest['counts']['required']} required, "
        f"{manifest['counts']['reviewed_exclusion']} reviewed exclusions, "
        f"{len(manifest['reform_coverage_probes'])} reform probe(s)."
    )


if __name__ == "__main__":
    main()
