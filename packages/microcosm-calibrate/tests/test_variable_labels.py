"""Country-owned Level 2 labels used by schema-7 diagnostics."""

from __future__ import annotations

import pytest

from microcosm.calibrate import (
    CALIBRATION_VARIABLE_LABELS_BY_COUNTRY,
    UK_CALIBRATION_VARIABLE_LABELS,
    US_CALIBRATION_VARIABLE_LABELS,
    calibration_variable_label,
)


def _pairs(value: str) -> set[tuple[str, str]]:
    return {tuple(line.split("|", 1)) for line in value.splitlines() if line.strip()}


_CURRENT_US_VARIABLES = _pairs(
    """bea_nipa|proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments
bea_nipa|wages_and_salaries
cbo|adjusted_gross_income_projection
cbo|net_business_income_projection
cbo|net_capital_gain_projection
cbo|qualified_dividend_income_projection
cbo|wages_and_salaries_projection
census_pep|resident_population
census_stc|individual_income_tax_collections
cms_aca|aptc_consumers
cms_aca|marketplace_plan_selections
cms_medicaid|total_chip_enrollment
cms_medicaid|total_medicaid_chip_enrollment
cms_medicaid|total_medicaid_enrollment
cms_medicare|part_b_premium_income
federal_reserve_z1|federal_reserve.z1.households_nonprofits_net_worth
hhs_acf_liheap|households_served_by_state_programs
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
irs_soi|non_sch_d_capital_gains
irs_soi|ordinary_dividend_income
irs_soi|partnership_and_s_corp_income
irs_soi|qualified_dividends
irs_soi|real_estate_taxes
irs_soi|refundable_ctc
irs_soi|rent_and_royalty_net_income
irs_soi|salt_deduction
irs_soi|tax_exempt_interest_income
irs_soi|taxable_income
irs_soi|taxable_interest_income
irs_soi|taxable_pension_income
irs_soi|taxable_social_security
irs_soi|tip_income
irs_soi|unemployment_compensation
jct|individual_tax_expenditure_revenue_loss
ssa_ssi_monthly|ssa.ssi_federal_payment_recipient
ssa_supplement|ssa.annual_oasdi_or_ssi_payment
ssa_supplement|ssa.ssi_payment
ssa_supplement|ssa.ssi_recipient
unspecified|selection_mass_protection.keogh_distributions
usda_snap|average_monthly_households
usda_snap|total_benefits"""
)


_CURRENT_UK_VARIABLES = _pairs(
    """dwp|esa_claimants
dwp|jsa_claimants
dwp|uc_benefit_units
dwp|uc_capped_households_total
dwp|uc_households
dwp|uc_tcl_children_by_children_3
dwp|uc_tcl_children_by_children_4
dwp|uc_tcl_children_by_children_5
dwp|uc_tcl_children_by_children_6_plus
dwp|uc_tcl_children_by_disability_claimant_pip
dwp|uc_tcl_children_by_disability_disabled_child_element
dwp|uc_tcl_headline_children_affected_by_policy
dwp|uc_tcl_headline_children_within_affected_households
dwp|uc_tcl_headline_households_affected
dwp|uc_tcl_households_by_children_3
dwp|uc_tcl_households_by_children_4
dwp|uc_tcl_households_by_children_5
dwp|uc_tcl_households_by_children_6_plus
dwp|uc_tcl_households_by_disability_claimant_pip
dwp|uc_tcl_households_by_disability_disabled_child_element
hmrc|cgt_gains_total
hmrc|cgt_taxpayers_total
hmrc|salary_sacrifice_pension_users
hmrc|spi_dividend_income
hmrc|spi_employment_income
hmrc|spi_private_pension_income
hmrc|spi_property_income
hmrc|spi_self_employment_income
hmrc|spi_state_pension
isc|pupils_at_member_schools
obr|efo_expenditure
obr|efo_receipts
ons|household_interest_resources
ons|households_by_type
ons|mid_year_population_estimate
ons|nbs_land_value_households
ons|nbs_land_value_nfc
ons|nbs_land_value_total
ons|pse_headcount_total_public_sector
scotgov|chargeable_dwellings_band_a
scotgov|chargeable_dwellings_band_b
scotgov|chargeable_dwellings_band_c
scotgov|chargeable_dwellings_band_d
scotgov|chargeable_dwellings_band_e
scotgov|chargeable_dwellings_band_f
scotgov|chargeable_dwellings_band_g
scotgov|chargeable_dwellings_band_h
scotgov|chargeable_dwellings_total
scotgov|social_security_assistance_spending
slc|maintenance_loan_amount_paid
slc|maintenance_loan_recipients
slc|student_loan_borrowers
slc|student_loan_net_repayments_plan_1
slc|student_loan_net_repayments_plan_2_full_time
slc|student_loan_net_repayments_total_higher_education
slc|targeted_support_amount_awarded
slc|targeted_support_recipients
voa|ct_stock_all_properties
voa|ct_stock_band_a
voa|ct_stock_band_b
voa|ct_stock_band_c
voa|ct_stock_band_d
voa|ct_stock_band_e
voa|ct_stock_band_f
voa|ct_stock_band_g
voa|ct_stock_band_h"""
)


@pytest.mark.parametrize(
    ("country", "pairs"),
    [("us", _CURRENT_US_VARIABLES), ("uk", _CURRENT_UK_VARIABLES)],
)
def test_current_schema_7_variables_have_labels(
    country: str,
    pairs: set[tuple[str, str]],
) -> None:
    missing = {
        (source_id, variable_id)
        for source_id, variable_id in pairs
        if calibration_variable_label(country, source_id, variable_id) is None
    }

    assert missing == set()


def test_variable_label_resolution_is_scoped_by_country_and_provider() -> None:
    assert calibration_variable_label("us", "irs_soi", "eitc") == "EITC"
    assert calibration_variable_label("UK", "obr", "efo_receipts") == ("EFO receipts")
    assert calibration_variable_label("uk", "hmrc", "efo_receipts") is None
    assert calibration_variable_label("be", "statbel", "population") is None


def test_variable_label_registries_are_immutable() -> None:
    with pytest.raises(TypeError):
        CALIBRATION_VARIABLE_LABELS_BY_COUNTRY["be"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        UK_CALIBRATION_VARIABLE_LABELS["obr"]["new"] = "New"  # type: ignore[index]
    with pytest.raises(TypeError):
        US_CALIBRATION_VARIABLE_LABELS["irs_soi"] = {}  # type: ignore[index]
