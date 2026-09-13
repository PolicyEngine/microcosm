"""Country-owned labels for programmatically declared target categories."""

from __future__ import annotations

import pytest

from microcosm.calibrate import (
    CALIBRATION_VARIABLE_LABELS_BY_COUNTRY,
    US_CALIBRATION_VARIABLE_LABELS,
    calibration_variable_label,
)


def _pairs(value: str) -> set[tuple[str, str]]:
    return {tuple(line.split("|", 1)) for line in value.splitlines() if line.strip()}


_CURRENT_US_VARIABLES = _pairs(
    """bea|nipa_proprietors_income
bea|nipa_wages_and_salaries
bea_nipa|proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments
bea_nipa|wages_and_salaries
cbo|cbo
cbo|adjusted_gross_income_projection
cbo|net_business_income_projection
cbo|net_capital_gain_projection
cbo|qualified_dividend_income_projection
cbo|wages_and_salaries_projection
census_acs|resident_population
census_acs|population_by_age
census_acs|population_by_sex_and_age
census_pep|resident_population
census_stc|individual_income_tax_collections
cms_aca|cms_aca
cms_aca|aptc_consumers
cms_aca|marketplace_plan_selections
cms_medicaid|cms_medicaid
cms_medicaid|total_chip_enrollment
cms_medicaid|total_medicaid_chip_enrollment
cms_medicaid|total_medicaid_enrollment
cms_medicare|cms_medicare
cms_medicare|part_b_premium_income
federal_reserve|federal_reserve
federal_reserve_z1|federal_reserve.z1.households_nonprofits_net_worth
hhs_acf_liheap|hhs_acf_liheap
hhs_acf_liheap|households_served_by_state_programs
hhs_acf_tanf|hhs_acf_tanf
hhs_acf_tanf|cash_assistance_expenditures
irs_soi|adjusted_gross_income
irs_soi|assigned_aca_ptc
irs_soi|business_net_profits
irs_soi|capital_gains_gross
irs_soi|charitable_deduction
irs_soi|count
irs_soi|ctc
irs_soi|deductible_mortgage_interest
irs_soi|eitc
irs_soi|employment_income
irs_soi|estate_income
irs_soi|estate_losses
irs_soi|income_tax
irs_soi|income_tax_before_credits
irs_soi|interest_deduction
irs_soi|ira_distributions
irs_soi|itemized_taxable_income_deductions
irs_soi|medical_expense_deduction
irs_soi|miscellaneous_income
irs_soi|miscellaneous_losses
irs_soi|non_sch_d_capital_gains
irs_soi|ordinary_dividend_income
irs_soi|partnership_and_s_corp_income
irs_soi|qualified_dividends
irs_soi|qualified_business_income_deduction
irs_soi|real_estate_taxes
irs_soi|refundable_ctc
irs_soi|rent_and_royalty_net_income
irs_soi|salt_deduction
irs_soi|tax_exempt_interest_income
irs_soi|taxable_income
irs_soi|taxable_interest_income
irs_soi|taxable_pension_income
irs_soi|taxable_social_security
irs_soi|tax_filer_individual_count
irs_soi|tip_income
irs_soi|unemployment_compensation
jct|individual_tax_expenditure_revenue_loss
ssa|ssa
ssa|ssi_payments
ssa|ssi_recipients
ssa_ssi_monthly|ssa.ssi_federal_payment_recipient
ssa_supplement|ssa.annual_oasdi_or_ssi_payment
ssa_supplement|ssa.ssi_payment
ssa_supplement|ssa.ssi_recipient
unspecified|selection_mass_protection.keogh_distributions
usda_snap|average_monthly_households
usda_snap|usda_snap
usda_snap|total_benefits"""
)


def test_current_programmatic_categories_have_labels() -> None:
    missing = {
        (source_id, variable_id)
        for source_id, variable_id in _CURRENT_US_VARIABLES
        if calibration_variable_label("us", source_id, variable_id) is None
    }

    assert missing == set()


def test_variable_label_resolution_is_scoped_by_country_and_provider() -> None:
    assert calibration_variable_label("us", "irs_soi", "eitc") == "EITC"
    assert calibration_variable_label("UK", "obr", "efo_receipts") is None
    assert calibration_variable_label("uk", "hmrc", "efo_receipts") is None
    assert calibration_variable_label("be", "statbel", "population") is None


def test_variable_label_registries_are_immutable() -> None:
    with pytest.raises(TypeError):
        CALIBRATION_VARIABLE_LABELS_BY_COUNTRY["be"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        US_CALIBRATION_VARIABLE_LABELS["irs_soi"] = {}  # type: ignore[index]
