"""US fiscal target-profile requirements and declared target facts.

JCT tax-expenditure rows must be computed from simple neutralization reforms:
run baseline income tax, neutralize one provision, run income tax again, and
use ``reform_income_tax - baseline_income_tax`` as the per-household
calibration row. This avoids treating tax expenditures as ordinary aggregate
columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Any, Literal

from populace.build.gates import TargetCoverageRequirement
from populace.build.ledger_targets import (
    LedgerTargetParityReport,
    LedgerTargetReference,
    compile_ledger_target_references,
)
from populace.calibrate import TargetRegistry, TargetSpec

__all__ = [
    "US_FISCAL_MACRO_REALISM_BANDS",
    "US_FISCAL_TARGET_REGISTRY",
    "US_FISCAL_TARGET_SPECS",
    "US_FISCAL_TARGET_COVERAGE_REQUIREMENTS",
    "US_FISCAL_TARGET_LEDGER_REFERENCES",
    "US_FISCAL_TARGET_REFERENCES",
    "US_FISCAL_TARGET_SUPPORT_EXCLUSIONS",
    "US_FISCAL_LEDGER_PARITY_REGISTRY",
    "US_FISCAL_LEDGER_PARITY_REPORT",
    "US_JCT_TAX_EXPENDITURE_REFORMS",
    "US_JCT_TAX_EXPENDITURE_TARGET_SPECS",
    "US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES",
    "SOI_VARIABLE_MAP",
    "US_SOI_FISCAL_TARGET_SPECS",
    "US_SOI_FISCAL_TARGET_REFERENCES",
    "US_STATE_INCOME_TAX_TARGET_SPECS",
    "US_STATE_INCOME_TAX_TARGET_REFERENCES",
    "SimpleTaxExpenditureReform",
    "compile_us_fiscal_target_registry",
]

TaxExpenditureReformKind = Literal["neutralize_variable"]
TaxExpenditureMatrixRow = Literal["reform_minus_baseline_income_tax"]


STATE_FIPS_TO_POSTAL: dict[str, str] = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}


SOI_AMOUNT_MEASURE_VARIABLES: dict[str, str] = {
    "adjusted_gross_income": "adjusted_gross_income",
    "actc_amount": "refundable_ctc",
    "ctc_amount": "ctc",
    "eitc_amount": "eitc",
    "eitc_no_children_amount": "eitc",
    "eitc_one_child_amount": "eitc",
    "eitc_three_or_more_children_amount": "eitc",
    "eitc_total": "eitc",
    "eitc_two_children_amount": "eitc",
    "income_tax_before_credits_amount": "income_tax_before_credits",
    "income_tax_liability_amount": "income_tax",
    "interest_paid_deduction_amount": "interest_deduction",
    "itemized_deductions_amount": "itemized_taxable_income_deductions",
    "limited_state_local_taxes_amount": "salt_deduction",
    "medical_dental_expense_amount": "medical_expense_deduction",
    "net_capital_gains_amount": "capital_gains_gross",
    "ordinary_dividends_amount": "ordinary_dividends",
    "partnership_scorp_income_amount": "partnership_and_s_corp_income",
    "premium_tax_credit_amount": "assigned_aca_ptc",
    "qualified_dividends_amount": "qualified_dividends",
    "real_estate_taxes_amount": "real_estate_taxes",
    "rental_royalty_income_amount": "rent_and_royalty_net_income",
    "schedule_c_income_amount": "business_net_profits",
    "tax_exempt_interest_amount": "tax_exempt_interest_income",
    "taxable_income_amount": "taxable_income",
    "taxable_interest_amount": "taxable_interest_income",
    "taxable_ira_distributions_amount": "ira_distributions",
    "taxable_pension_income_amount": "taxable_pension_income",
    "taxable_social_security_amount": "taxable_social_security",
    "total_itemized_deductions_amount": "itemized_taxable_income_deductions",
    "total_earned_income_credit_amount": "eitc",
    "unemployment_compensation_amount": "unemployment_compensation",
    "wages_salaries_amount": "employment_income",
    "charitable_amount": "charitable_deduction",
}


SOI_RETURN_MEASURE_VARIABLES: dict[str, str] = {
    "actc_claims": "refundable_ctc",
    "ctc_claims": "ctc",
    "eitc_claims": "eitc",
    "eitc_no_children_claims": "eitc",
    "eitc_one_child_claims": "eitc",
    "eitc_returns": "eitc",
    "eitc_three_or_more_children_claims": "eitc",
    "eitc_two_children_claims": "eitc",
    "income_tax_before_credits_returns": "income_tax_before_credits",
    "income_tax_liability_returns": "income_tax",
    "limited_state_local_taxes_returns": "salt_deduction",
    "medical_dental_expense_returns": "medical_expense_deduction",
    "net_capital_gains_returns": "capital_gains_gross",
    "ordinary_dividends_returns": "ordinary_dividends",
    "partnership_scorp_income_returns": "partnership_and_s_corp_income",
    "premium_tax_credit_returns": "assigned_aca_ptc",
    "qualified_dividends_returns": "qualified_dividends",
    "real_estate_taxes_claims": "real_estate_taxes",
    "rental_royalty_income_returns": "rent_and_royalty_net_income",
    "return_count": "count",
    "schedule_c_income_returns": "business_net_profits",
    "tax_exempt_interest_returns": "tax_exempt_interest_income",
    "taxable_income_returns": "taxable_income",
    "taxable_interest_returns": "taxable_interest_income",
    "taxable_ira_distributions_returns": "ira_distributions",
    "taxable_pension_income_returns": "taxable_pension_income",
    "taxable_social_security_returns": "taxable_social_security",
    "total_earned_income_credit_returns": "eitc",
    "unemployment_compensation_returns": "unemployment_compensation",
    "wages_salaries_returns": "employment_income",
}


SOI_VARIABLE_MAP: dict[str, str] = {
    "adjusted_gross_income": "adjusted_gross_income",
    "business_net_profits": "self_employment_income",
    "business_net_losses": "self_employment_income",
    "capital_gains_distributions": "capital_gains",
    "capital_gains_gross": "capital_gains",
    "capital_gains_losses": "capital_losses",
    "charitable_deduction": "charitable_deduction",
    "ctc": "ctc",
    "employment_income": "employment_income",
    "eitc": "eitc",
    "estate_income": "estate_income",
    "estate_losses": "estate_income",
    "exempt_interest": "tax_exempt_interest_income",
    "income_tax": "income_tax",
    "income_tax_before_credits": "income_tax_before_credits",
    "interest_deduction": "interest_deduction",
    "ira_distributions": "taxable_ira_distributions",
    "itemized_taxable_income_deductions": "itemized_taxable_income_deductions",
    "medical_expense_deduction": "medical_expense_deduction",
    "ordinary_dividends": "ordinary_dividend_income",
    "partnership_and_s_corp_income": "tax_unit_partnership_s_corp_income",
    "partnership_and_s_corp_losses": "tax_unit_partnership_s_corp_income",
    "assigned_aca_ptc": "assigned_aca_ptc",
    "qualified_business_income_deduction": "qualified_business_income_deduction",
    "qualified_dividends": "qualified_dividend_income",
    "real_estate_taxes": "real_estate_taxes",
    "rent_and_royalty_net_income": "rent_and_royalty_net_income",
    "rent_and_royalty_net_losses": "rent_and_royalty_net_income",
    "refundable_ctc": "refundable_ctc",
    "salt_deduction": "salt_deduction",
    "tax_exempt_interest_income": "tax_exempt_interest_income",
    "taxable_income": "taxable_income",
    "taxable_interest_income": "taxable_interest_income",
    "taxable_pension_income": "taxable_pension_income",
    "taxable_social_security": "tax_unit_taxable_social_security",
    "tip_income": "tip_income",
    "total_pension_income": "pension_income",
    "total_social_security": "tax_unit_social_security",
    "unemployment_compensation": "unemployment_compensation",
}

_SOI_BASE_VARIABLE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "ctc": ("ctc", "ctc_limiting_tax_liability"),
    "ordinary_dividends": (
        "qualified_dividend_income",
        "non_qualified_dividend_income",
    ),
    "rent_and_royalty_net_income": ("rental_income", "farm_rent_income"),
    "rent_and_royalty_net_losses": ("rental_income", "farm_rent_income"),
}

_SOI_EITC_CHILD_COUNT_LAYOUT_DIMENSIONS = frozenset(
    {
        "us.tax.earned_income_credit_qualifying_children",
        "us.tax.eitc_qualifying_children",
    }
)
_SOI_EITC_DECOMPOSITION_AMOUNT_MEASURES = frozenset({"eitc_total"})
_SOI_EITC_DECOMPOSITION_RETURN_MEASURES = frozenset({"eitc_returns"})
_SOI_EITC_TOTAL_AMOUNT_MEASURES = frozenset(
    {"eitc_amount", "total_earned_income_credit_amount"}
)
_SOI_EITC_TOTAL_RETURN_MEASURES = frozenset(
    {"eitc_claims", "total_earned_income_credit_returns"}
)
_SOI_TOTAL_UPRATED_AMOUNT_MEASURES = frozenset({"taxable_interest_amount"})
_SOI_TOTAL_UPRATED_RETURN_MEASURES = frozenset({"taxable_interest_returns"})
_SOI_TOTAL_UPRATED_DECOMPOSITION_MEASURES = (
    _SOI_TOTAL_UPRATED_AMOUNT_MEASURES | _SOI_TOTAL_UPRATED_RETURN_MEASURES
)
_SOI_FORM_W2_ITEM_LAYOUT_DIMENSION = "irs_soi.form_w2_item"
_SOI_FORM_W2_SOCIAL_SECURITY_TIP_ITEMS = frozenset(
    {
        "box_7_social_security_tips",
    }
)
_SOI_ITEMIZED_ONLY_VARIABLES = frozenset(
    {
        "charitable_deduction",
        "interest_deduction",
        "itemized_taxable_income_deductions",
        "medical_expense_deduction",
        "real_estate_taxes",
        "salt_deduction",
    }
)


DIRECT_LEDGER_TARGETS: dict[
    tuple[str, str, str | None], tuple[str, str, dict[str, str]]
] = {
    ("cbo", "projected_amount", "adjusted_gross_income"): (
        "adjusted_gross_income",
        "cbo",
        {"target_role": "cbo_adjusted_gross_income"},
    ),
    ("cbo", "projected_amount", "wages_and_salaries"): (
        "employment_income",
        "cbo",
        {"target_role": "cbo_wages_and_salaries"},
    ),
    ("cbo", "projected_amount", "qualified_dividend_income"): (
        "qualified_dividend_income",
        "cbo",
        {"target_role": "cbo_qualified_dividend_income"},
    ),
    ("cbo", "projected_amount", "net_capital_gain"): (
        "capital_gains",
        "cbo",
        {"target_role": "cbo_net_capital_gain"},
    ),
    ("cbo", "projected_amount", "net_business_income"): (
        "cbo_net_business_income",
        "cbo",
        {"target_role": "cbo_net_business_income"},
    ),
    ("ssa", "payment_amount", "social_security_benefits"): (
        "social_security",
        "ssa",
        {"target_role": "social_security_total"},
    ),
    ("ssa", "payment_amount", "social_security_retirement_benefits"): (
        "social_security_retirement",
        "ssa",
        {"target_role": "ssa_retirement_total"},
    ),
    ("ssa", "payment_amount", "social_security_disability_benefits"): (
        "social_security_disability",
        "ssa",
        {"target_role": "ssa_disability_total"},
    ),
    ("ssa", "payment_amount", "social_security_survivors_benefits"): (
        "social_security_survivors",
        "ssa",
        {"target_role": "ssa_survivors_total"},
    ),
    ("ssa", "payment_amount", "social_security_dependents_benefits"): (
        "social_security_dependents",
        "ssa",
        {"target_role": "ssa_dependents_total"},
    ),
    ("ssa", "payment_amount", "ssi_payments"): (
        "ssi",
        "ssa",
        {"target_role": "ssi_total"},
    ),
    ("usda_snap", "total_benefits", None): (
        "snap",
        "usda_snap",
        {"target_role": "snap_total"},
    ),
    ("hhs_acf_tanf", "all_funds", None): (
        "tanf",
        "hhs_acf_tanf",
        {"target_role": "tanf_total"},
    ),
    ("cms_nhe", "expenditure_amount", "medicaid_title_xix"): (
        "medicaid",
        "cms_medicaid",
        {
            "target_role": "medicaid_spending",
            "calibration_role": "validation_only",
            "exclusion_reason": (
                "PolicyEngine-US allocates medicaid spending from state totals "
                "through person_weight-dependent denominators. Reweighting then "
                "recomputes per-person medicaid costs, so this is not a linear "
                "calibration row."
            ),
        },
    ),
    ("cms_medicare", "actual_amount", "premiums_from_enrollees"): (
        "gross_medicare_part_b_premium",
        "cms_medicare",
        {"target_role": "medicare_part_b_premium_total"},
    ),
}


IndicatorBaseVariables = str | tuple[str, ...]
IndicatorLedgerTarget = tuple[IndicatorBaseVariables, str, dict[str, str]]


INDICATOR_LEDGER_TARGETS: dict[tuple[str, str], IndicatorLedgerTarget] = {
    ("cms_aca", "marketplace_enrollment"): (
        "has_marketplace_health_coverage_at_interview",
        "cms_aca",
        {"target_role": "aca_enrollment"},
    ),
    # CMS reports APTC recipients as people. We proxy this as an indicator-sum
    # over eligible people in tax units with positive assigned PTC: the tax-unit
    # assigned_aca_ptc is projected to person grain (indicator_map_to) and
    # filtered to is_aca_ptc_eligible. Validated at 20.24M vs the 19.74M CMS
    # target (+2.5%); see PR #68.
    ("cms_aca", "aptc_recipients"): (
        "assigned_aca_ptc",
        "cms_aca",
        {
            "target_role": "aca_ptc_recipients",
            "indicator_map_to": "person",
            "indicator_filter_variable": "is_aca_ptc_eligible",
        },
    ),
    ("cms_aca", "bronze_aptc_consumers"): (
        "selected_marketplace_plan_benchmark_ratio",
        "cms_aca",
        {
            "target_role": "aca_bronze_aptc_consumers",
            "measure_mode": "less_than_indicator_sum",
            "indicator_less_than": "1.0",
            "indicator_filter_variable": "assigned_aca_ptc",
        },
    ),
    ("cms_medicaid", "total_medicaid_enrollment"): (
        "medicaid_enrolled",
        "cms_medicaid",
        {"target_role": "medicaid_enrollment"},
    ),
    ("cms_medicaid", "total_medicaid_chip_enrollment"): (
        ("medicaid_enrolled", "chip_enrolled"),
        "cms_medicaid",
        {"target_role": "medicaid_chip_enrollment"},
    ),
}


US_FISCAL_TARGET_SUPPORT_EXCLUSIONS: dict[str, str] = {
    "census_stc.fy2024.individual_income_tax_collections.tn.t40.collections": (
        "Tennessee has no modeled 2024 state individual income tax support in "
        "PolicyEngine-US; this STC residual collection row cannot be estimated "
        "from the current state_income_tax variable."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.ar.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Arkansas under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.az.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Arizona under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.co.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Colorado under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.ct.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Connecticut under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.de.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Delaware under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.ga.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Georgia under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.id.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Idaho under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.in.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Indiana under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.ks.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Kansas under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.ky.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Kentucky under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.me.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Maine under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.mi.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Michigan under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.mt.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Montana under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.nm.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in New Mexico under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.nv.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Nevada under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.ok.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Oklahoma under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.or.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Oregon under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.pa.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Pennsylvania under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.sd.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in South Dakota under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.tx.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Texas under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.vt.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in Vermont under PolicyEngine-US state TANF formulas."
    ),
    "hhs_acf_tanf.fy2024.cash_assistance.wv.basic_assistance_excluding_relative_foster_care_and_adoption_guardianship.all_funds": (
        "Current 2024 base microdata have zero positive TANF benefit support "
        "in West Virginia under PolicyEngine-US state TANF formulas."
    ),
    "irs_soi.ty2022.historic_table_2.state_agi.ak.under_1.return_count": (
        "Current 2024 base microdata have zero Alaska return-count support in "
        "the SOI under-$1 AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.under_1.income_tax_before_credits_returns": (
        "Current 2024 base microdata have zero positive income-tax-before-"
        "credits support in the SOI under-$1 AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.under_1.income_tax_before_credits_amount": (
        "Current 2024 base microdata have zero income-tax-before-credits "
        "amount support in the SOI under-$1 AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.under_1.income_tax_liability_returns": (
        "Current 2024 base microdata have zero positive income-tax-liability "
        "support in the SOI under-$1 AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.under_1.income_tax_liability_amount": (
        "Current 2024 base microdata have zero income-tax-liability amount "
        "support in the SOI under-$1 AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.1_to_10k.taxable_income_returns": (
        "Current 2024 base microdata have zero positive taxable-income support "
        "in this SOI AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.1_to_10k.taxable_income_amount": (
        "Current 2024 base microdata have zero taxable-income amount support "
        "in this SOI AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.500k_to_1m.actc_claims": (
        "Current-law PolicyEngine-US refundable CTC support is zero in this "
        "high-income SOI AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.500k_to_1m.actc_amount": (
        "Current-law PolicyEngine-US refundable CTC amount support is zero in "
        "this high-income SOI AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.500k_to_1m.medical_dental_expense_returns": (
        "Current 2024 base microdata have zero positive medical-expense "
        "deduction support in this SOI AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.500k_to_1m.medical_dental_expense_amount": (
        "Current 2024 base microdata have zero medical-expense deduction "
        "amount support in this SOI AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.1m_plus.medical_dental_expense_returns": (
        "Current 2024 base microdata have zero positive medical-expense "
        "deduction support in this SOI AGI bin."
    ),
    "irs_soi.ty2022.historic_table_2.us.1m_plus.medical_dental_expense_amount": (
        "Current 2024 base microdata have zero medical-expense deduction "
        "amount support in this SOI AGI bin."
    ),
}


@dataclass(frozen=True)
class SimpleTaxExpenditureReform:
    """A JCT target row backed by one simple neutralization reform.

    The builder implementation should construct a PolicyEngine reform that
    calls ``neutralize_variable(neutralized_variable)``, calculate
    ``income_tax`` under both baseline and reform, and calibrate to the
    household-level delta ``reform - baseline``.
    """

    target_name: str
    neutralized_variable: str
    measure: str
    period: int | str
    source: str = ""
    kind: TaxExpenditureReformKind = "neutralize_variable"
    output_variable: str = "income_tax"
    matrix_row: TaxExpenditureMatrixRow = "reform_minus_baseline_income_tax"

    @classmethod
    def from_target_reference(
        cls, reference: LedgerTargetReference
    ) -> SimpleTaxExpenditureReform:
        """Build the reform contract from a declared JCT target row."""
        return cls(
            target_name=reference.name,
            neutralized_variable=reference.metadata.get("neutralized_variable", ""),
            source=reference.source or "",
            measure=reference.measure or "",
            period=reference.period or 0,
            kind=reference.metadata.get("kind", ""),
            output_variable=reference.metadata.get("output_variable", ""),
            matrix_row=reference.metadata.get("matrix_row", ""),
        )

    def __post_init__(self) -> None:
        if not self.target_name:
            raise ValueError("target_name is required.")
        if not self.neutralized_variable:
            raise ValueError(f"{self.target_name}: neutralized_variable is required.")
        if self.measure != self.target_name:
            raise ValueError(
                f"{self.target_name}: JCT targets must use a precomputed "
                "measure column named after the target row, not an ordinary "
                "income_tax column."
            )
        if self.kind != "neutralize_variable":
            raise ValueError(
                f"{self.target_name}: JCT targets must use a simple "
                "neutralize_variable reform."
            )
        if self.output_variable != "income_tax":
            raise ValueError(
                f"{self.target_name}: JCT targets must fit the income_tax delta."
            )
        if self.matrix_row != "reform_minus_baseline_income_tax":
            raise ValueError(
                f"{self.target_name}: JCT matrix row must be reform income_tax "
                "minus baseline income_tax."
            )

    def coverage_requirement(self) -> TargetCoverageRequirement:
        """The target-profile requirement satisfied by this JCT row."""
        return TargetCoverageRequirement(
            requirement_id=f"jct_tax_expenditure:{self.neutralized_variable}",
            label=f"JCT tax expenditure for {self.neutralized_variable}",
            accepted_names=(self.target_name,),
            required_measures=(self.target_name,),
            required_metadata=(
                ("kind", self.kind),
                ("output_variable", self.output_variable),
                ("matrix_row", self.matrix_row),
                ("neutralized_variable", self.neutralized_variable),
            ),
            notes=(
                "Must be computed as a simple neutralize_variable reform and "
                "calibrated to income_tax(reform) - income_tax(baseline)."
            ),
        )


def _load_us_fiscal_target_references() -> tuple[LedgerTargetReference, ...]:
    payload = json.loads(
        files("populace.build.us").joinpath("fiscal_target_references.json").read_text()
    )
    if payload.get("country") != "us":
        raise ValueError("US fiscal target manifest must declare country='us'.")
    allowed_operations = set(payload.get("allowed_value_operations") or ())
    if allowed_operations != {"identity"}:
        raise ValueError(
            "US fiscal target references currently permit only identity value "
            f"resolution from Ledger facts; got {sorted(allowed_operations)!r}."
        )
    return tuple(LedgerTargetReference(**raw) for raw in payload["target_references"])


def compile_us_fiscal_target_registry(
    facts: object,
    *,
    target_period: int | str = 2024,
) -> TargetRegistry:
    """Resolve US fiscal targets from an external Ledger fact feed."""
    materialized_facts = tuple(facts)
    references = (
        *_dynamic_us_fiscal_target_references(
            materialized_facts,
            target_period=target_period,
        ),
        *_references_for_target_period(
            US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES,
            target_period=target_period,
        ),
    )
    registry = compile_ledger_target_references(
        materialized_facts,
        references,
        country="us",
    )
    registry = _uprate_cross_period_eitc_decompositions(registry)
    return _uprate_cross_period_soi_decompositions(
        registry,
        materialized_facts,
        target_period=target_period,
    )


def _uprate_cross_period_eitc_decompositions(
    registry: TargetRegistry,
) -> TargetRegistry:
    """Scale stale EITC AGI/child decompositions to active EITC controls.

    SOI Table 2.5 releases AGI-by-qualifying-child EITC distributions later
    than all-return totals. When Populace uses a prior-year decomposition for a
    target-year build, it must not treat old nominal bins as hard target-year
    levels. Prefer active AGI-bucket EITC facts when Ledger has them, preserving
    the latest child split inside each AGI bucket. Fall back to all-return EITC
    totals when only totals are available.
    """

    target_totals = _eitc_active_totals(registry)
    target_agi_totals = _eitc_active_agi_totals(registry)
    source_totals = _eitc_source_decomposition_totals(registry)
    source_totals = {
        **_eitc_source_total_fallbacks(registry),
        **source_totals,
    }
    specs: list[TargetSpec] = []
    for spec in registry.specs:
        if spec.metadata.get("requires_total_eitc_uprating") != "true":
            specs.append(spec)
            continue
        kind = _eitc_total_kind(spec)
        source_period = spec.metadata.get("source_period", "")
        agi_uprating_group = _eitc_agi_uprating_group(
            spec,
            target_agi_totals.get(kind or "", ()),
        )
        if agi_uprating_group is not None:
            source_total = _eitc_source_decomposition_total_for_bounds(
                registry,
                kind=kind,
                source_period=source_period,
                lower=agi_uprating_group.lower,
                upper=agi_uprating_group.upper,
            )
            if source_total not in (None, 0):
                factor = agi_uprating_group.value / source_total
                metadata = {
                    **dict(spec.metadata),
                    "uprating_index": _eitc_agi_uprating_index(kind),
                    "uprating_from_period": source_period,
                    "uprating_to_period": str(spec.period),
                    "uprating_index_source_period": ",".join(
                        agi_uprating_group.source_periods
                    ),
                    "uprating_agi_lower_bound": _format_bound(agi_uprating_group.lower),
                    "uprating_agi_upper_bound": _format_bound(agi_uprating_group.upper),
                    "uprating_factor": _format_float(factor),
                }
                if len(agi_uprating_group.source_record_ids) == 1:
                    metadata["uprating_index_source_record_id"] = (
                        agi_uprating_group.source_record_ids[0]
                    )
                else:
                    metadata["uprating_index_source_record_ids"] = ",".join(
                        agi_uprating_group.source_record_ids
                    )
                specs.append(
                    replace(
                        spec,
                        value=spec.value * factor,
                        metadata=metadata,
                    )
                )
                continue

        target_total = target_totals.get(kind or "")
        source_total = source_totals.get((kind, source_period))
        if target_total is None or source_total in (None, 0):
            continue
        factor = target_total.value / source_total
        specs.append(
            replace(
                spec,
                value=spec.value * factor,
                metadata={
                    **dict(spec.metadata),
                    "uprating_index": _eitc_uprating_index(kind),
                    "uprating_from_period": source_period,
                    "uprating_to_period": str(spec.period),
                    "uprating_index_source_period": target_total.source_period,
                    "uprating_index_source_record_id": target_total.source_record_id,
                    "uprating_factor": _format_float(factor),
                },
            )
        )
    return TargetRegistry(specs, country=registry.country)


@dataclass(frozen=True)
class _SoiTotalControl:
    value: float
    source_period: str
    source_record_id: str
    period_key: tuple[int, int, str]


def _uprate_cross_period_soi_decompositions(
    registry: TargetRegistry,
    facts: tuple[object, ...],
    *,
    target_period: int | str,
) -> TargetRegistry:
    """Scale stale SOI AGI slices to same-scope active SOI totals."""

    source_totals = _soi_total_controls_by_source_period(facts)
    active_totals = _soi_active_total_controls(facts, target_period=target_period)
    specs: list[TargetSpec] = []
    for spec in registry.specs:
        if spec.metadata.get("requires_total_soi_uprating") != "true":
            specs.append(spec)
            continue

        measure_id = spec.metadata.get("source_measure_id", "")
        source_period = spec.metadata.get("source_period", "")
        key = _soi_total_control_key_from_spec(spec)
        source_total = source_totals.get((*key, source_period))
        active_total = active_totals.get(key)
        if source_total in (None, 0) or active_total is None:
            continue

        factor = active_total.value / source_total
        specs.append(
            replace(
                spec,
                value=spec.value * factor,
                metadata={
                    **dict(spec.metadata),
                    "uprating_index": _soi_total_uprating_index(measure_id),
                    "uprating_from_period": source_period,
                    "uprating_to_period": str(spec.period),
                    "uprating_index_source_period": active_total.source_period,
                    "uprating_index_source_record_id": active_total.source_record_id,
                    "uprating_factor": _format_float(factor),
                },
            )
        )
    return TargetRegistry(specs, country=registry.country)


def _soi_total_controls_by_source_period(
    facts: tuple[object, ...],
) -> dict[tuple[str, str, str, str], float]:
    totals: dict[tuple[str, str, str, str], float] = {}
    for fact in facts:
        if not _is_soi_total_uprating_control_fact(fact):
            continue
        key = (
            *_soi_total_control_key_from_fact(fact),
            str(_period_value(fact)),
        )
        totals[key] = _numeric_value(fact)
    return totals


def _soi_active_total_controls(
    facts: tuple[object, ...],
    *,
    target_period: int | str,
) -> dict[tuple[str, str, str], _SoiTotalControl]:
    totals: dict[tuple[str, str, str], _SoiTotalControl] = {}
    target_period_key = _period_key_from_value(target_period)
    for fact in facts:
        if not _is_soi_total_uprating_control_fact(fact):
            continue
        period_key = _period_key(fact)
        if not _not_after_target_period(period_key, target_period_key):
            continue
        source_record_id = _source_record_id(fact)
        if not source_record_id:
            continue
        key = _soi_total_control_key_from_fact(fact)
        candidate = _SoiTotalControl(
            value=_numeric_value(fact),
            source_period=str(_period_value(fact)),
            source_record_id=source_record_id,
            period_key=period_key,
        )
        current = totals.get(key)
        if current is None or _prefer_candidate(
            candidate.period_key,
            current.period_key,
            target_period_key=target_period_key,
        ):
            totals[key] = candidate
    return totals


def _is_soi_total_uprating_control_fact(fact: object) -> bool:
    if _source_name(fact) != "irs_soi":
        return False
    if _measure_id(fact) not in _SOI_TOTAL_UPRATED_DECOMPOSITION_MEASURES:
        return False
    if _geography_level(fact) not in {"country", "state"}:
        return False
    if not _source_record_id(fact):
        return False
    if not _is_all_income_range(fact):
        return False
    lower, upper = _agi_bounds(fact)
    return lower == "-inf" and upper == "inf"


def _soi_total_control_key_from_fact(fact: object) -> tuple[str, str, str]:
    state_fips = _state_fips(fact)
    geography_key = f"state:{state_fips}" if state_fips else "country"
    return (
        _measure_id(fact),
        geography_key,
        _filing_status_label(_dimensions(fact).get("filing_status")) or "",
    )


def _soi_total_control_key_from_spec(spec: TargetSpec) -> tuple[str, str, str]:
    state_fips = spec.metadata.get("state_fips", "")
    geography_key = f"state:{state_fips}" if state_fips else "country"
    return (
        spec.metadata.get("source_measure_id", ""),
        geography_key,
        spec.metadata.get("filing_status", ""),
    )


def _soi_total_uprating_index(measure_id: str) -> str:
    if measure_id in _SOI_TOTAL_UPRATED_RETURN_MEASURES:
        return "total_taxable_interest_returns"
    return "total_taxable_interest_amount"


@dataclass(frozen=True)
class _EitcActiveTotal:
    value: float
    source_period: str
    source_record_id: str
    period_key: tuple[int, int, str]


@dataclass(frozen=True)
class _EitcAgiTotal:
    value: float
    source_periods: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    period_key: tuple[int, int, str]
    lower: float
    upper: float


def _eitc_active_totals(registry: TargetRegistry) -> dict[str, _EitcActiveTotal]:
    totals: dict[str, _EitcActiveTotal] = {}
    for spec in registry.specs:
        kind = _eitc_total_kind(spec)
        if kind is None:
            continue
        if spec.metadata.get("target_role") not in _eitc_total_roles(kind):
            continue
        source_period = spec.metadata.get("source_period", "")
        candidate = _EitcActiveTotal(
            value=spec.value,
            source_period=source_period,
            source_record_id=spec.metadata.get("ledger_source_record_id", spec.name),
            period_key=_period_key_from_value(source_period),
        )
        current = totals.get(kind)
        if current is None or _prefer_candidate(
            candidate.period_key,
            current.period_key,
            target_period_key=_period_key_from_value(spec.period),
        ):
            totals[kind] = candidate
    return totals


def _eitc_active_agi_totals(
    registry: TargetRegistry,
) -> dict[str, tuple[_EitcAgiTotal, ...]]:
    latest: dict[tuple[str, float, float], _EitcAgiTotal] = {}
    for spec in registry.specs:
        kind = _eitc_total_kind(spec)
        if kind is None or not _is_eitc_agi_total_spec(spec):
            continue
        lower, upper = _bounds_from_metadata(spec)
        candidate = _EitcAgiTotal(
            value=spec.value,
            source_periods=(spec.metadata.get("source_period", ""),),
            source_record_ids=(
                spec.metadata.get("ledger_source_record_id", spec.name),
            ),
            period_key=_period_key_from_value(spec.metadata.get("source_period", "")),
            lower=lower,
            upper=upper,
        )
        key = (kind, lower, upper)
        current = latest.get(key)
        if current is None or _prefer_candidate(
            candidate.period_key,
            current.period_key,
            target_period_key=_period_key_from_value(spec.period),
        ):
            latest[key] = candidate

    by_kind: dict[str, list[_EitcAgiTotal]] = {}
    for (kind, _, _), total in latest.items():
        by_kind.setdefault(kind, []).append(total)
    return {
        kind: tuple(sorted(totals, key=lambda item: (item.lower, item.upper)))
        for kind, totals in by_kind.items()
    }


def _eitc_source_decomposition_totals(
    registry: TargetRegistry,
) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = {}
    for spec in registry.specs:
        if not _is_eitc_decomposition_spec(spec):
            continue
        if spec.metadata.get("agi_lower_bound") != "-inf":
            continue
        if spec.metadata.get("agi_upper_bound") != "inf":
            continue
        kind = _eitc_total_kind(spec)
        if kind is None:
            continue
        source_period = spec.metadata.get("source_period", "")
        key = (kind, source_period)
        totals[key] = totals.get(key, 0.0) + spec.value
    return totals


def _eitc_source_total_fallbacks(
    registry: TargetRegistry,
) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = {}
    for spec in registry.specs:
        kind = _eitc_total_kind(spec)
        if kind is None:
            continue
        if spec.metadata.get("target_role") not in _eitc_total_roles(kind):
            continue
        source_period = spec.metadata.get("source_period", "")
        totals[(kind, source_period)] = spec.value
    return totals


def _eitc_source_decomposition_total_for_bounds(
    registry: TargetRegistry,
    *,
    kind: str | None,
    source_period: str,
    lower: float,
    upper: float,
) -> float | None:
    total = 0.0
    matched = False
    for spec in registry.specs:
        if not _is_eitc_decomposition_spec(spec):
            continue
        if _eitc_total_kind(spec) != kind:
            continue
        if spec.metadata.get("source_period", "") != source_period:
            continue
        spec_lower, spec_upper = _bounds_from_metadata(spec)
        if _contains_bounds(lower, upper, spec_lower, spec_upper):
            total += spec.value
            matched = True
    return total if matched else None


def _eitc_agi_uprating_group(
    spec: TargetSpec,
    active_agi_totals: tuple[_EitcAgiTotal, ...],
) -> _EitcAgiTotal | None:
    if not active_agi_totals:
        return None
    source_period_key = _period_key_from_value(spec.metadata.get("source_period", ""))
    spec_lower, spec_upper = _bounds_from_metadata(spec)
    containing = [
        total
        for total in active_agi_totals
        if _period_not_before(total.period_key, source_period_key)
        and _contains_bounds(total.lower, total.upper, spec_lower, spec_upper)
    ]
    if containing:
        return min(containing, key=lambda total: total.upper - total.lower)

    contained = [
        total
        for total in active_agi_totals
        if _period_not_before(total.period_key, source_period_key)
        and _contains_bounds(spec_lower, spec_upper, total.lower, total.upper)
    ]
    if not _covers_interval(contained, lower=spec_lower, upper=spec_upper):
        return None
    return _merge_eitc_agi_totals(contained, lower=spec_lower, upper=spec_upper)


def _merge_eitc_agi_totals(
    totals: list[_EitcAgiTotal],
    *,
    lower: float,
    upper: float,
) -> _EitcAgiTotal:
    source_periods = tuple(
        dict.fromkeys(period for total in totals for period in total.source_periods)
    )
    source_record_ids = tuple(
        dict.fromkeys(
            source_record_id
            for total in totals
            for source_record_id in total.source_record_ids
        )
    )
    return _EitcAgiTotal(
        value=sum(total.value for total in totals),
        source_periods=source_periods,
        source_record_ids=source_record_ids,
        period_key=max(total.period_key for total in totals),
        lower=lower,
        upper=upper,
    )


def _is_eitc_agi_total_spec(spec: TargetSpec) -> bool:
    if _is_eitc_decomposition_spec(spec):
        return False
    if spec.metadata.get("ledger_filter_eitc_child_count"):
        return False
    if spec.metadata.get("filing_status") != "All":
        return False
    if spec.metadata.get("state_fips"):
        return False
    if spec.metadata.get("source_measure_id") not in (
        _SOI_EITC_TOTAL_AMOUNT_MEASURES | _SOI_EITC_TOTAL_RETURN_MEASURES
    ):
        return False
    lower, upper = _bounds_from_metadata(spec)
    return not (lower == -float("inf") and upper == float("inf"))


def _bounds_from_metadata(spec: TargetSpec) -> tuple[float, float]:
    return (
        _bound_from_metadata_value(spec.metadata.get("agi_lower_bound", "-inf")),
        _bound_from_metadata_value(spec.metadata.get("agi_upper_bound", "inf")),
    )


def _bound_from_metadata_value(value: str) -> float:
    if value == "-inf":
        return -float("inf")
    if value == "inf":
        return float("inf")
    return float(value)


def _contains_bounds(
    outer_lower: float,
    outer_upper: float,
    inner_lower: float,
    inner_upper: float,
) -> bool:
    return outer_lower <= inner_lower and inner_upper <= outer_upper


def _covers_interval(
    totals: list[_EitcAgiTotal],
    *,
    lower: float,
    upper: float,
) -> bool:
    if not totals:
        return False
    cursor = lower
    for total in sorted(totals, key=lambda item: (item.lower, item.upper)):
        if total.lower != cursor:
            return False
        cursor = total.upper
        if cursor == upper:
            return True
        if cursor > upper:
            return False
    return cursor == upper


def _period_not_before(
    candidate: tuple[int, int, str],
    minimum: tuple[int, int, str],
) -> bool:
    if not candidate[0] or not minimum[0]:
        return True
    return candidate[1] >= minimum[1]


def _format_bound(value: float) -> str:
    if value == -float("inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    return _format_float(value)


def _is_eitc_decomposition_spec(spec: TargetSpec) -> bool:
    metadata = spec.metadata
    if metadata.get("source_measure_id") not in (
        _SOI_EITC_DECOMPOSITION_AMOUNT_MEASURES
        | _SOI_EITC_DECOMPOSITION_RETURN_MEASURES
    ):
        return False
    return ".table_2_5.eitc_by_agi_children." in metadata.get(
        "ledger_layout_record_set_id", ""
    )


def _eitc_total_kind(spec: TargetSpec) -> str | None:
    measure_id = spec.metadata.get("source_measure_id", "")
    if measure_id in (
        _SOI_EITC_DECOMPOSITION_AMOUNT_MEASURES | _SOI_EITC_TOTAL_AMOUNT_MEASURES
    ):
        return "amount"
    if measure_id in (
        _SOI_EITC_DECOMPOSITION_RETURN_MEASURES | _SOI_EITC_TOTAL_RETURN_MEASURES
    ):
        return "returns"
    return None


def _eitc_total_roles(kind: str) -> frozenset[str]:
    if kind == "amount":
        return frozenset({"eitc_total"})
    if kind == "returns":
        return frozenset({"eitc_returns_total"})
    return frozenset()


def _eitc_uprating_index(kind: str | None) -> str:
    if kind == "returns":
        return "total_eitc_returns"
    return "total_eitc_amount"


def _eitc_agi_uprating_index(kind: str | None) -> str:
    if kind == "returns":
        return "agi_eitc_returns"
    return "agi_eitc_amount"


def _format_float(value: float) -> str:
    return f"{value:.15g}"


def _dynamic_us_fiscal_target_references(
    facts: tuple[object, ...],
    *,
    target_period: int | str,
) -> tuple[LedgerTargetReference, ...]:
    candidates: list[
        tuple[tuple[str, ...], tuple[int, int, str], LedgerTargetReference]
    ] = []
    for fact in facts:
        reference = _reference_from_ledger_fact(fact, target_period=target_period)
        if reference is not None:
            candidates.append((_dynamic_target_key(fact), _period_key(fact), reference))
    latest: dict[
        tuple[str, ...], tuple[tuple[int, int, str], LedgerTargetReference]
    ] = {}
    target_period_key = _period_key_from_value(target_period)
    for key, period_key, reference in candidates:
        if not _not_after_target_period(period_key, target_period_key):
            continue
        current = latest.get(key)
        if current is None:
            latest[key] = (period_key, reference)
            continue
        if _prefer_candidate(
            period_key,
            current[0],
            target_period_key=target_period_key,
        ):
            latest[key] = (period_key, reference)
    return tuple(reference for _, reference in latest.values())


def _dynamic_target_key(fact: object) -> tuple[str, ...]:
    """Semantic identity for generated fiscal targets, excluding source period.

    Ledger should retain old source years. A Populace build should activate only
    one fact for each model target shape, choosing the latest source period
    separately. The key therefore includes source, measure, geography, layout,
    dimensions, and universe constraints, but strips period-like tokens from
    record-set identifiers such as ``irs_soi.ty2023.table_1_1``.
    """

    model_target_key = _model_target_key(fact)
    if model_target_key is not None:
        return model_target_key
    return (
        _source_name(fact),
        _measure_id(fact),
        _geography_level(fact),
        _geography_id(fact),
        _str_at(fact, "entity", "name"),
        _str_at(fact, "aggregation", "method"),
        _normalized_record_set_id(_str_at(fact, "layout", "record_set_id")),
        _str_at(fact, "layout", "groupby_dimension"),
        _str_at(fact, "layout", "groupby_value_id"),
        json.dumps(_dimensions(fact), sort_keys=True, separators=(",", ":")),
        json.dumps(_constraint_rows(fact), sort_keys=True, separators=(",", ":")),
    )


def _model_target_key(fact: object) -> tuple[str, ...] | None:
    source_name = _source_name(fact)
    mapping = _direct_target_mapping(fact)
    if mapping is None:
        indicator_mapping = INDICATOR_LEDGER_TARGETS.get(
            (source_name, _measure_id(fact))
        )
        if indicator_mapping is None:
            return None
        base_variable, family, metadata = indicator_mapping
        target_role = metadata.get("target_role", "")
        measure_mode = metadata.get("measure_mode", "indicator_sum")
    else:
        base_variable, family, metadata = mapping
        target_role = metadata.get("target_role", "")
        measure_mode = metadata.get("measure_mode", "sum")
    base_variables = (
        ",".join(base_variable) if isinstance(base_variable, tuple) else base_variable
    )
    return (
        family,
        base_variables,
        target_role,
        measure_mode,
        metadata.get("indicator_map_to", ""),
        metadata.get("indicator_filter_variable", ""),
        _geography_level(fact),
        _geography_id(fact),
        _state_fips(fact) or "",
        _str_at(fact, "layout", "groupby_value_id"),
        json.dumps(_dimensions(fact), sort_keys=True, separators=(",", ":")),
        json.dumps(_constraint_rows(fact), sort_keys=True, separators=(",", ":")),
    )


def _normalized_record_set_id(record_set_id: str) -> str:
    if not record_set_id:
        return ""
    return ".".join(
        part for part in record_set_id.split(".") if not _is_period_token(part)
    )


def _is_period_token(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    if normalized.startswith("month"):
        normalized = normalized[len("month") :]
    parts = normalized.split("_", maxsplit=1)
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return len(parts[0]) == 4 and len(parts[1]) in {1, 2}
    if normalized[:2] in {"ty", "cy", "fy"}:
        normalized = normalized[2:]
    return normalized.isdigit() and len(normalized) == 4


def _period_key(fact: object) -> tuple[int, int, str]:
    return _period_key_from_value(_period_value(fact))


def _period_key_from_value(value: object) -> tuple[int, int, str]:
    label = "" if value is None else str(value)
    normalized = label.lower().replace("-", "_")
    for prefix in ("month", "tax_year_", "calendar_year_", "fiscal_year_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    parts = normalized.split("_", maxsplit=1)
    if len(parts) == 2:
        year, month = parts
        if year.isdigit() and month.isdigit():
            return (1, int(year) * 100 + int(month), label)
    try:
        return (1, int(normalized) * 100 + 99, label)
    except ValueError:
        return (0, 0, label)


def _prefer_candidate(
    candidate: tuple[int, int, str],
    current: tuple[int, int, str],
    *,
    target_period_key: tuple[int, int, str],
) -> bool:
    candidate_is_eligible = _not_after_target_period(candidate, target_period_key)
    current_is_eligible = _not_after_target_period(current, target_period_key)
    if candidate_is_eligible != current_is_eligible:
        return candidate_is_eligible
    return candidate > current


def _not_after_target_period(
    source_period_key: tuple[int, int, str],
    target_period_key: tuple[int, int, str],
) -> bool:
    if not source_period_key[0] or not target_period_key[0]:
        return True
    return source_period_key[1] <= target_period_key[1]


def _reference_from_ledger_fact(
    fact: object,
    *,
    target_period: int | str,
) -> LedgerTargetReference | None:
    source_record_id = _source_record_id(fact)
    if source_record_id in US_FISCAL_TARGET_SUPPORT_EXCLUSIONS:
        return None
    source_name = _source_name(fact)
    if source_name == "irs_soi":
        return _soi_reference_from_fact(fact, target_period=target_period)
    if source_name == "census_stc":
        return _state_income_tax_reference_from_fact(fact, target_period=target_period)
    if source_name == "census_pep":
        return _population_age_reference_from_fact(fact, target_period=target_period)
    if source_name == "jct":
        return None
    return _direct_reference_from_fact(fact, target_period=target_period)


def _soi_reference_from_fact(
    fact: object,
    *,
    target_period: int | str,
) -> LedgerTargetReference | None:
    if _geography_level(fact) not in {"country", "state"}:
        return None
    measure_id = _measure_id(fact)
    variable = SOI_AMOUNT_MEASURE_VARIABLES.get(measure_id)
    is_count = False
    if variable is None:
        variable = SOI_RETURN_MEASURE_VARIABLES.get(measure_id)
        is_count = variable is not None
    if variable is None:
        return None
    override = _soi_layout_variable_override(fact, measure_id=measure_id)
    if override is not None:
        variable, is_count = override

    lower, upper = _agi_bounds(fact)
    cross_period_agi_slice = _is_untransformed_cross_period_agi_slice(
        fact,
        agi_lower_bound=lower,
        agi_upper_bound=upper,
        target_period=target_period,
    )
    requires_total_eitc_uprating = _is_cross_period_fact(
        fact, target_period=target_period
    ) and _is_soi_eitc_decomposition_fact(fact, measure_id)
    requires_total_soi_uprating = (
        cross_period_agi_slice
        and _is_cross_period_fact(fact, target_period=target_period)
        and _is_soi_total_uprated_decomposition_fact(fact, measure_id)
    )
    if (
        cross_period_agi_slice
        and not requires_total_eitc_uprating
        and not requires_total_soi_uprating
    ):
        return None
    status = _filing_status_label(_dimensions(fact).get("filing_status"))
    if status is None:
        return None

    source_record_id = _source_record_id(fact)
    if not source_record_id:
        return None
    display_variable = _soi_display_variable(variable)
    metadata = {
        "source_measure_id": measure_id,
        "source_variable": variable,
        "source_period": str(_period_value(fact)),
        "target_period": str(target_period),
        "target_role": _soi_target_role(fact, measure_id),
        "variable": display_variable,
        "materializer": "irs_soi_slice",
        "measure_mode": _soi_measure_mode(variable, is_count=is_count),
        "agi_lower_bound": lower,
        "agi_upper_bound": upper,
        "filing_status": status,
        **_soi_layout_filter_metadata(fact),
    }
    base_variables = _soi_base_variables(variable)
    if len(base_variables) == 1:
        metadata["base_variable"] = base_variables[0]
    elif len(base_variables) > 1:
        metadata["base_variables"] = ",".join(base_variables)
    if variable in _SOI_ITEMIZED_ONLY_VARIABLES:
        metadata["itemized_only"] = "true"
    state_fips = _state_fips(fact)
    if state_fips:
        metadata["state_fips"] = state_fips
    if requires_total_eitc_uprating:
        metadata["requires_total_eitc_uprating"] = "true"
    if requires_total_soi_uprating:
        metadata["requires_total_soi_uprating"] = "true"
    return LedgerTargetReference(
        name=source_record_id,
        ledger_source_record_id=source_record_id,
        entity="household",
        measure=source_record_id,
        period=target_period,
        family="irs_soi",
        signed=_numeric_value(fact) < 0,
        metadata=metadata,
    )


def _soi_measure_mode(variable: str, *, is_count: bool) -> str:
    if is_count:
        return "indicator_sum"
    return "sum"


def _soi_display_variable(variable: str) -> str:
    if variable == "ordinary_dividends":
        return "ordinary_dividend_income"
    return variable


def _soi_base_variables(variable: str) -> tuple[str, ...]:
    if variable == "count":
        return ()
    return _SOI_BASE_VARIABLE_OVERRIDES.get(variable, (SOI_VARIABLE_MAP[variable],))


def _soi_layout_variable_override(
    fact: object, *, measure_id: str
) -> tuple[str, bool] | None:
    groupby_dimension = _str_at(fact, "layout", "groupby_dimension")
    groupby_value = _str_at(fact, "layout", "groupby_value_id")
    if (
        measure_id == "return_count"
        and groupby_dimension in _SOI_EITC_CHILD_COUNT_LAYOUT_DIMENSIONS
        and groupby_value
        and groupby_value != "all"
    ):
        return "eitc", True
    if (
        measure_id == "return_count"
        and groupby_dimension == _SOI_FORM_W2_ITEM_LAYOUT_DIMENSION
        and groupby_value in _SOI_FORM_W2_SOCIAL_SECURITY_TIP_ITEMS
    ):
        return "tip_income", True
    return None


def _soi_layout_filter_metadata(fact: object) -> dict[str, str]:
    groupby_dimension = _str_at(fact, "layout", "groupby_dimension")
    groupby_value = _str_at(fact, "layout", "groupby_value_id")
    if (
        groupby_dimension in _SOI_EITC_CHILD_COUNT_LAYOUT_DIMENSIONS
        and groupby_value
        and groupby_value != "all"
    ):
        return {"ledger_filter_eitc_child_count": groupby_value}
    return {}


def _is_soi_eitc_decomposition_fact(fact: object, measure_id: str) -> bool:
    if measure_id not in (
        _SOI_EITC_DECOMPOSITION_AMOUNT_MEASURES
        | _SOI_EITC_DECOMPOSITION_RETURN_MEASURES
    ):
        return False
    record_set_id = _str_at(fact, "layout", "record_set_id")
    return ".table_2_5.eitc_by_agi_children." in record_set_id


def _is_soi_total_uprated_decomposition_fact(
    fact: object,
    measure_id: str,
) -> bool:
    if measure_id not in _SOI_TOTAL_UPRATED_DECOMPOSITION_MEASURES:
        return False
    return not _is_all_income_range(fact)


def _state_income_tax_reference_from_fact(
    fact: object,
    *,
    target_period: int | str,
) -> LedgerTargetReference | None:
    if _measure_id(fact) != "collections" or _geography_level(fact) != "state":
        return None
    state_fips = _state_fips(fact)
    source_record_id = _source_record_id(fact)
    if not state_fips or not source_record_id:
        return None
    return LedgerTargetReference(
        name=source_record_id,
        ledger_source_record_id=source_record_id,
        entity="household",
        measure=source_record_id,
        period=target_period,
        family="state_income_tax",
        metadata={
            "source_measure_id": "collections",
            "source_period": str(_period_value(fact)),
            "target_period": str(target_period),
            "state_fips": state_fips,
            "target_role": "state_income_tax",
        },
    )


def _population_age_reference_from_fact(
    fact: object,
    *,
    target_period: int | str,
) -> LedgerTargetReference | None:
    if _measure_id(fact) != "population":
        return None
    geography_level = _geography_level(fact)
    if geography_level == "country":
        state_fips = None
        geography_scope = "national"
    elif geography_level == "state":
        state_fips = _state_fips(fact)
        if state_fips is None:
            return None
        geography_scope = "state"
    else:
        return None
    source_record_id = _source_record_id(fact)
    if not source_record_id:
        return None
    lower, upper = _age_bounds(fact)
    if lower == "-inf" and upper == "inf":
        return None
    metadata = {
        "materializer": "population_age",
        "measure_mode": "indicator_sum",
        "source_measure_id": "population",
        "source_period": str(_period_value(fact)),
        "target_period": str(target_period),
        "target_role": "population_age",
        "geography_scope": geography_scope,
        "age_lower_bound": lower,
        "age_upper_bound": upper,
    }
    groupby_value_id = _str_at(fact, "layout", "groupby_value_id")
    if groupby_value_id:
        metadata["age_group"] = groupby_value_id
    if state_fips:
        metadata["state_fips"] = state_fips
    return LedgerTargetReference(
        name=source_record_id,
        ledger_source_record_id=source_record_id,
        entity="household",
        measure=source_record_id,
        period=target_period,
        family="census_population",
        metadata=metadata,
    )


def _direct_reference_from_fact(
    fact: object,
    *,
    target_period: int | str,
) -> LedgerTargetReference | None:
    source_name = _source_name(fact)
    measure_id = _measure_id(fact)
    mapping = _direct_target_mapping(fact)
    measure_mode = "sum"
    if mapping is None:
        indicator_mapping = INDICATOR_LEDGER_TARGETS.get((source_name, measure_id))
        if indicator_mapping is None:
            return None
        base_variable, family, metadata = indicator_mapping
        metadata = {"measure_mode": "indicator_sum", **metadata}
    else:
        base_variable, family, metadata = mapping
        metadata = dict(metadata)
    if metadata.get("calibration_role") == "validation_only":
        return None

    source_record_id = _source_record_id(fact)
    if not source_record_id:
        return None
    if _geography_level(fact) == "state":
        state_fips = _state_fips(fact)
        if state_fips is None:
            return None
    elif _geography_level(fact) != "country":
        return None
    else:
        state_fips = None

    metadata = {
        **metadata,
        "materializer": "policyengine_variable",
        "measure_mode": metadata.get("measure_mode", measure_mode),
        "source_measure_id": measure_id,
        "source_period": str(_period_value(fact)),
        "target_period": str(target_period),
    }
    if isinstance(base_variable, tuple):
        metadata["base_variables"] = ",".join(base_variable)
    else:
        metadata["base_variable"] = base_variable
    if state_fips:
        metadata["state_fips"] = state_fips
    return LedgerTargetReference(
        name=source_record_id,
        ledger_source_record_id=source_record_id,
        entity="household",
        measure=source_record_id,
        period=target_period,
        family=family,
        signed=_numeric_value(fact) < 0,
        metadata=metadata,
    )


def _direct_target_mapping(
    fact: object,
) -> tuple[str | tuple[str, ...], str, dict[str, str]] | None:
    source_name = _source_name(fact)
    measure_id = _measure_id(fact)
    group_value = _str_at(fact, "layout", "groupby_value_id") or None
    return DIRECT_LEDGER_TARGETS.get(
        (source_name, measure_id, group_value)
    ) or DIRECT_LEDGER_TARGETS.get((source_name, measure_id, None))


def _references_for_target_period(
    references: tuple[LedgerTargetReference, ...],
    *,
    target_period: int | str,
) -> tuple[LedgerTargetReference, ...]:
    return tuple(
        replace(
            reference,
            period=target_period,
            metadata={
                **dict(reference.metadata),
                "target_period": str(target_period),
            },
        )
        for reference in references
    )


def _soi_target_role(fact: object, measure_id: str) -> str:
    if _is_all_income_range(fact):
        if measure_id == "premium_tax_credit_amount":
            return "aca_spending"
        if measure_id == "premium_tax_credit_returns":
            return "aca_ptc_returns"
    if _geography_level(fact) == "country" and _is_all_income_range(fact):
        roles = {
            "income_tax_before_credits_amount": "income_tax_before_credits_total",
            "income_tax_liability_amount": "federal_income_tax_total",
            "eitc_amount": "eitc_total",
            "total_earned_income_credit_amount": "eitc_total",
            "total_earned_income_credit_returns": "eitc_returns_total",
            "actc_amount": "refundable_ctc_total",
            "ctc_amount": "ctc_total",
            "charitable_amount": "charitable_deduction_total",
            "interest_paid_deduction_amount": "interest_deduction_total",
            "itemized_deductions_amount": "itemized_deduction_total",
            "limited_state_local_taxes_amount": "salt_deduction_total",
            "medical_dental_expense_amount": "medical_expense_deduction_total",
            "total_itemized_deductions_amount": "itemized_deduction_total",
            "unemployment_compensation_amount": "unemployment_compensation_total",
        }
        if measure_id in roles:
            return roles[measure_id]
    return "soi_fiscal_distribution"


def _is_all_income_range(fact: object) -> bool:
    dimensions = _dimensions(fact)
    lower, upper = _agi_bounds(fact)
    return (
        str(dimensions.get("income_range", "all")) == "all"
        and str(dimensions.get("filing_status", "all")) == "all"
        and lower == "-inf"
        and upper == "inf"
    )


def _is_untransformed_cross_period_agi_slice(
    fact: object,
    *,
    agi_lower_bound: str,
    agi_upper_bound: str,
    target_period: int | str,
) -> bool:
    """Refuse stale nominal SOI AGI bins as target-period hard targets."""
    if agi_lower_bound == "-inf" and agi_upper_bound == "inf":
        return False
    source_period_key = _period_key(fact)
    target_period_key = _period_key_from_value(target_period)
    if not source_period_key[0] or not target_period_key[0]:
        return False
    return source_period_key[1] != target_period_key[1]


def _is_cross_period_fact(fact: object, *, target_period: int | str) -> bool:
    source_period_key = _period_key(fact)
    target_period_key = _period_key_from_value(target_period)
    if not source_period_key[0] or not target_period_key[0]:
        return False
    return source_period_key[1] != target_period_key[1]


def _agi_bounds(fact: object) -> tuple[str, str]:
    lower = "-inf"
    upper = "inf"
    for constraint in _constraint_rows(fact):
        if not isinstance(constraint, dict):
            continue
        variable = str(constraint.get("variable") or "")
        if "adjusted_gross_income" not in variable:
            continue
        operator = str(constraint.get("operator") or "")
        value = constraint.get("value")
        if value is None:
            continue
        if operator in {">", ">="}:
            lower = str(float(value))
        elif operator in {"<", "<="}:
            upper = str(float(value))
    return lower, upper


def _age_bounds(fact: object) -> tuple[str, str]:
    lower = "-inf"
    upper = "inf"
    for constraint in _constraint_rows(fact):
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("variable") or "") != "age":
            continue
        value = constraint.get("value")
        if value is None:
            continue
        operator = str(constraint.get("operator") or "")
        if operator in {">", ">="}:
            lower = _format_number(value)
        elif operator in {"<", "<="}:
            upper = _format_number(value)
    return lower, upper


def _format_number(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _filing_status_label(value: object) -> str | None:
    raw = str(value or "all").lower()
    labels = {
        "all": "All",
        "head_of_household": "Head of Household",
        "married_filing_jointly_surviving_spouse": (
            "Married Filing Jointly/Surviving Spouse"
        ),
        "married_filing_separately": "Married Filing Separately",
        "single": "Single",
    }
    return labels.get(raw)


def _state_fips(fact: object) -> str | None:
    geoid = _geography_id(fact)
    if not geoid.startswith("0400000US"):
        return None
    fips = geoid.removeprefix("0400000US")
    if fips not in STATE_FIPS_TO_POSTAL:
        return None
    return fips


def _numeric_value(fact: object) -> float:
    value = _at(fact, "value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _source_name(fact: object) -> str:
    return _str_at(fact, "observed_measure", "source_name") or _str_at(
        fact, "source", "source_name"
    )


def _measure_id(fact: object) -> str:
    return _str_at(fact, "observed_measure", "source_measure_id") or _str_at(
        fact, "layout", "measure_id"
    )


def _source_record_id(fact: object) -> str:
    return _str_at(fact, "lineage", "source_record_id")


def _period_value(fact: object) -> object:
    return _at(fact, "period", "value")


def _geography_level(fact: object) -> str:
    return _str_at(fact, "geography", "level")


def _geography_id(fact: object) -> str:
    return _str_at(fact, "geography", "id")


def _dimensions(fact: object) -> dict[str, object]:
    dimensions = _at(fact, "dimensions")
    return dict(dimensions) if isinstance(dimensions, dict) else {}


def _constraint_rows(fact: object) -> tuple[object, ...]:
    constraints = _at(fact, "constraints")
    if isinstance(constraints, list | tuple):
        return tuple(constraints)
    universe_constraints = _at(fact, "universe_constraints", "constraints")
    if isinstance(universe_constraints, list | tuple):
        return tuple(universe_constraints)
    return ()


def _at(obj: object, *path: str) -> Any:
    current = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _str_at(obj: object, *path: str) -> str:
    value = _at(obj, *path)
    return "" if value is None else str(value)


US_FISCAL_TARGET_REFERENCES: tuple[LedgerTargetReference, ...] = (
    _load_us_fiscal_target_references()
)
US_FISCAL_TARGET_LEDGER_REFERENCES: tuple[LedgerTargetReference, ...] = (
    US_FISCAL_TARGET_REFERENCES
)
# Compatibility constants only: target values are no longer packaged in
# Populace. Release builds must call compile_us_fiscal_target_registry() with an
# external Ledger consumer-facts artifact.
US_FISCAL_TARGET_SPECS: tuple[TargetSpec, ...] = ()
US_FISCAL_TARGET_REGISTRY = TargetRegistry(US_FISCAL_TARGET_SPECS, country="us")
US_FISCAL_LEDGER_PARITY_REGISTRY = TargetRegistry((), country="us")
US_FISCAL_LEDGER_PARITY_REPORT: LedgerTargetParityReport = LedgerTargetParityReport(
    passed=True,
    failures=(),
    details={"mode": "value-free references require external Ledger facts"},
)
US_JCT_TAX_EXPENDITURE_TARGET_SPECS: tuple[TargetSpec, ...] = ()
US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES: tuple[LedgerTargetReference, ...] = tuple(
    reference for reference in US_FISCAL_TARGET_REFERENCES if reference.family == "jct"
)
US_JCT_TAX_EXPENDITURE_REFORMS: tuple[SimpleTaxExpenditureReform, ...] = tuple(
    SimpleTaxExpenditureReform.from_target_reference(reference)
    for reference in US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES
)
US_STATE_INCOME_TAX_TARGET_SPECS: tuple[TargetSpec, ...] = ()
US_STATE_INCOME_TAX_TARGET_REFERENCES: tuple[LedgerTargetReference, ...] = tuple(
    reference
    for reference in US_FISCAL_TARGET_REFERENCES
    if reference.family == "state_income_tax"
)
US_SOI_FISCAL_TARGET_SPECS: tuple[TargetSpec, ...] = ()
US_SOI_FISCAL_TARGET_REFERENCES: tuple[LedgerTargetReference, ...] = tuple(
    reference
    for reference in US_FISCAL_TARGET_REFERENCES
    if reference.family == "irs_soi"
)

US_FISCAL_TARGET_COVERAGE_REQUIREMENTS: tuple[TargetCoverageRequirement, ...] = (
    TargetCoverageRequirement(
        requirement_id="federal_income_tax_total",
        label="Federal individual income tax return liability total",
        accepted_families=("irs_soi",),
        required_metadata=(("target_role", "federal_income_tax_total"),),
        notes=(
            "Must be the IRS SOI return-level income tax liability total. "
            "CBO individual income tax receipts are macro cash receipts and "
            "remain reference diagnostics, not hard return-level calibration "
            "targets."
        ),
    ),
    TargetCoverageRequirement(
        requirement_id="irs_agi_distribution",
        label="SOI AGI distribution and top-tail controls",
        accepted_name_substrings=(".adjusted_gross_income",),
        min_matches=20,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_wages_distribution",
        label="SOI wages by AGI bracket",
        accepted_name_substrings=(".wages_salaries_amount",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_business_income_distribution",
        label="SOI business income by AGI bracket",
        accepted_name_substrings=(".schedule_c_income_amount",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_partnership_s_corp_distribution",
        label="SOI partnership and S-corp income by AGI bracket",
        accepted_name_substrings=(".partnership_scorp_income_amount",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_capital_gains_distribution",
        label="SOI capital gains by AGI bracket",
        accepted_name_substrings=(".net_capital_gains_amount",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_dividends_distribution",
        label="SOI dividends by AGI bracket",
        accepted_name_substrings=(".ordinary_dividends_amount",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_interest_distribution",
        label="SOI taxable interest by AGI bracket",
        accepted_name_substrings=(".taxable_interest_amount",),
        min_matches=100,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_pension_distribution",
        label="SOI pension income by AGI bracket",
        accepted_name_substrings=(".taxable_pension_income_amount",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_social_security_distribution",
        label="SOI Social Security by AGI bracket",
        accepted_name_substrings=(".taxable_social_security_amount",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="itemized_deduction_total",
        label="SOI itemized deduction amount total",
        accepted_families=("irs_soi",),
        required_metadata=(("target_role", "itemized_deduction_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="salt_deduction_total",
        label="SOI state and local tax deduction amount total",
        accepted_families=("irs_soi",),
        required_metadata=(("target_role", "salt_deduction_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="medical_expense_deduction_total",
        label="SOI medical expense deduction amount total",
        accepted_families=("irs_soi",),
        required_metadata=(("target_role", "medical_expense_deduction_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="social_security_total",
        label="SSA Social Security total",
        accepted_families=("ssa",),
        required_metadata=(("target_role", "social_security_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="ssi_total",
        label="SSA SSI total",
        accepted_families=("ssa",),
        required_metadata=(("target_role", "ssi_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="snap_total",
        label="USDA SNAP total",
        accepted_families=("usda_snap",),
        required_metadata=(("target_role", "snap_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="unemployment_compensation_total",
        label="SOI unemployment compensation total",
        accepted_families=("irs_soi",),
        required_metadata=(("target_role", "unemployment_compensation_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="ssa_social_security_components",
        label="SSA Social Security component totals",
        accepted_families=("ssa",),
        min_matches=4,
    ),
    TargetCoverageRequirement(
        requirement_id="eitc_total",
        label="SOI EITC total",
        accepted_families=("irs_soi",),
        required_metadata=(("target_role", "eitc_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="refundable_ctc_total",
        label="Refundable CTC total",
        accepted_families=("irs_soi",),
        required_metadata=(("target_role", "refundable_ctc_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="ctc_total",
        label="Child Tax Credit total",
        accepted_families=("irs_soi",),
        required_metadata=(("target_role", "ctc_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="aca_marketplace",
        label="ACA marketplace spending and enrollment",
        accepted_families=("cms_aca",),
        min_matches=2,
    ),
    TargetCoverageRequirement(
        requirement_id="medicaid_enrollment",
        label="Medicaid enrollment",
        accepted_families=("cms_medicaid",),
        required_metadata=(("target_role", "medicaid_enrollment"),),
    ),
    TargetCoverageRequirement(
        requirement_id="medicaid_chip_enrollment",
        label="Medicaid and CHIP combined enrollment",
        accepted_families=("cms_medicaid",),
        required_metadata=(("target_role", "medicaid_chip_enrollment"),),
    ),
    TargetCoverageRequirement(
        requirement_id="medicare_part_b_premium_total",
        label="Medicare Part B premium income from enrollees",
        accepted_families=("cms_medicare",),
        required_metadata=(("target_role", "medicare_part_b_premium_total"),),
    ),
    TargetCoverageRequirement(
        requirement_id="state_income_tax",
        label="State individual income tax collections",
        accepted_families=("state_income_tax",),
        required_metadata=(("target_role", "state_income_tax"),),
        min_matches=44,
    ),
    TargetCoverageRequirement(
        requirement_id="population_age_national",
        label="Census PEP national population by age",
        accepted_families=("census_population",),
        required_metadata=(
            ("target_role", "population_age"),
            ("geography_scope", "national"),
        ),
        min_matches=18,
    ),
    TargetCoverageRequirement(
        requirement_id="population_age_state",
        label="Census PEP state population by age",
        accepted_families=("census_population",),
        required_metadata=(
            ("target_role", "population_age"),
            ("geography_scope", "state"),
        ),
        min_matches=918,
    ),
    *(spec.coverage_requirement() for spec in US_JCT_TAX_EXPENDITURE_REFORMS),
)

US_FISCAL_MACRO_REALISM_BANDS: dict[str, tuple[float, float]] = {
    "federal_income_tax_to_gdp": (0.07, 0.11),
    "agi_to_gdp": (0.50, 0.70),
    "spm_below_threshold_rate": (0.06, 0.18),
}
