"""Country-owned display labels for calibration statistic categories.

Schema 7 calls these categories ``variable`` objects.  A category may combine
several Chronicle facts or measures, so its display label belongs with
Microcosm's calibration grouping rather than with any one source fact.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "CALIBRATION_VARIABLE_LABELS_BY_COUNTRY",
    "UK_CALIBRATION_VARIABLE_LABELS",
    "US_CALIBRATION_VARIABLE_LABELS",
    "calibration_variable_label",
]


def _immutable_registry(
    labels: dict[str, dict[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    return MappingProxyType(
        {
            source_id: MappingProxyType(dict(variable_labels))
            for source_id, variable_labels in labels.items()
        }
    )


US_CALIBRATION_VARIABLE_LABELS: Mapping[str, Mapping[str, str]] = _immutable_registry(
    {
        "bea_nipa": {
            "proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments": (
                "Proprietors' income"
            ),
            "wages_and_salaries": "Wages and salaries",
        },
        "cbo": {
            "adjusted_gross_income_projection": "Adjusted gross income projection",
            "net_business_income_projection": "Net business income projection",
            "net_capital_gain_projection": "Net capital gain projection",
            "qualified_dividend_income_projection": (
                "Qualified dividend income projection"
            ),
            "wages_and_salaries_projection": "Wages and salaries projection",
        },
        "census_pep": {
            "resident_population": "Resident population",
        },
        "census_stc": {
            "individual_income_tax_collections": ("Individual income tax collections"),
        },
        "cms_aca": {
            "aptc_consumers": "APTC consumers",
            "marketplace_plan_selections": "Marketplace plan selections",
        },
        "cms_medicaid": {
            "total_chip_enrollment": "Total CHIP enrollment",
            "total_medicaid_chip_enrollment": "Total Medicaid and CHIP enrollment",
            "total_medicaid_enrollment": "Total Medicaid enrollment",
        },
        "cms_medicare": {
            "part_b_premium_income": "Part B premium income",
        },
        "federal_reserve_z1": {
            "federal_reserve.z1.households_nonprofits_net_worth": (
                "Households and nonprofit organizations net worth"
            ),
        },
        "hhs_acf_liheap": {
            "households_served_by_state_programs": (
                "Households served by state programs"
            ),
        },
        "hhs_acf_tanf": {
            "cash_assistance_expenditures": "Cash assistance expenditures",
        },
        "irs_soi": {
            "adjusted_gross_income": "Adjusted gross income",
            "assigned_aca_ptc": "Assigned ACA PTC",
            "business_net_profits": "Business net profits",
            "capital_gains_gross": "Gross capital gains",
            "charitable_deduction": "Charitable deduction",
            "count": "Individual income tax returns",
            "ctc": "CTC",
            "deductible_mortgage_interest": "Deductible mortgage interest",
            "eitc": "EITC",
            "employment_income": "Employment income",
            "estate_income": "Estate income",
            "estate_losses": "Estate losses",
            "income": "Income",
            "income_tax": "Income tax",
            "income_tax_before_credits": "Income tax before credits",
            "interest_deduction": "Interest deduction",
            "ira_distributions": "IRA distributions",
            "itemized_taxable_income_deductions": (
                "Itemized taxable income deductions"
            ),
            "medical_expense_deduction": "Medical expense deduction",
            "non_sch_d_capital_gains": "Non-Schedule D capital gains",
            "ordinary_dividend_income": "Ordinary dividend income",
            "partnership_and_s_corp_income": "Partnership and S corporation income",
            "qualified_dividends": "Qualified dividends",
            "real_estate_taxes": "Real estate taxes",
            "refundable_ctc": "Refundable CTC",
            "rent_and_royalty_net_income": "Rent and royalty net income",
            "salt_deduction": "SALT deduction",
            "tax_exempt_interest_income": "Tax-exempt interest income",
            "taxable_income": "Taxable income",
            "taxable_interest_income": "Taxable interest income",
            "taxable_pension_income": "Taxable pension income",
            "taxable_social_security": "Taxable Social Security",
            "tip_income": "Tip income",
            "unemployment_compensation": "Unemployment compensation",
        },
        "jct": {
            "individual_tax_expenditure_revenue_loss": (
                "Individual tax expenditure revenue loss"
            ),
        },
        "ssa_ssi_monthly": {
            "ssa.ssi_federal_payment_recipient": "Federal SSI payment recipients",
        },
        "ssa_supplement": {
            "ssa.annual_oasdi_or_ssi_payment": "Annual OASDI or SSI payments",
            "ssa.ssi_payment": "SSI payments",
            "ssa.ssi_recipient": "SSI recipients",
        },
        "unspecified": {
            "selection_mass_protection.keogh_distributions": (
                "Keogh distributions selection-mass constraint"
            ),
        },
        "usda_snap": {
            "average_monthly_households": "Average monthly households",
            "total_benefits": "Total benefits",
        },
    }
)


UK_CALIBRATION_VARIABLE_LABELS: Mapping[str, Mapping[str, str]] = _immutable_registry(
    {
        "dwp": {
            "esa_claimants": "ESA claimants",
            "jsa_claimants": "JSA claimants",
            "uc_benefit_units": "Universal Credit benefit units",
            "uc_capped_households_total": "Universal Credit capped households",
            "uc_households": "Universal Credit households",
            "uc_tcl_children_by_children_3": (
                "Universal Credit two-child limit: children in households with 3 children"
            ),
            "uc_tcl_children_by_children_4": (
                "Universal Credit two-child limit: children in households with 4 children"
            ),
            "uc_tcl_children_by_children_5": (
                "Universal Credit two-child limit: children in households with 5 children"
            ),
            "uc_tcl_children_by_children_6_plus": (
                "Universal Credit two-child limit: children in households with 6+ children"
            ),
            "uc_tcl_children_by_disability_claimant_pip": (
                "Universal Credit two-child limit: children by claimant PIP status"
            ),
            "uc_tcl_children_by_disability_disabled_child_element": (
                "Universal Credit two-child limit: children by disabled-child element"
            ),
            "uc_tcl_headline_children_affected_by_policy": (
                "Universal Credit two-child limit: children affected"
            ),
            "uc_tcl_headline_children_within_affected_households": (
                "Universal Credit two-child limit: children in affected households"
            ),
            "uc_tcl_headline_households_affected": (
                "Universal Credit two-child limit: households affected"
            ),
            "uc_tcl_households_by_children_3": (
                "Universal Credit two-child limit: households with 3 children"
            ),
            "uc_tcl_households_by_children_4": (
                "Universal Credit two-child limit: households with 4 children"
            ),
            "uc_tcl_households_by_children_5": (
                "Universal Credit two-child limit: households with 5 children"
            ),
            "uc_tcl_households_by_children_6_plus": (
                "Universal Credit two-child limit: households with 6+ children"
            ),
            "uc_tcl_households_by_disability_claimant_pip": (
                "Universal Credit two-child limit: households by claimant PIP status"
            ),
            "uc_tcl_households_by_disability_disabled_child_element": (
                "Universal Credit two-child limit: households by disabled-child element"
            ),
        },
        "hmrc": {
            "cgt_gains_total": "Total CGT gains",
            "cgt_taxpayers_total": "Total CGT taxpayers",
            "salary_sacrifice_pension_users": "Salary-sacrifice pension users",
            "spi_dividend_income": "SPI dividend income",
            "spi_employment_income": "SPI employment income",
            "spi_private_pension_income": "SPI private pension income",
            "spi_property_income": "SPI property income",
            "spi_self_employment_income": "SPI self-employment income",
            "spi_state_pension": "SPI state pension",
        },
        "isc": {
            "pupils_at_member_schools": "Pupils at member schools",
        },
        "obr": {
            "efo_expenditure": "EFO expenditure",
            "efo_receipts": "EFO receipts",
        },
        "ons": {
            "household_interest_resources": "Household interest resources",
            "households_by_type": "Households by type",
            "mid_year_population_estimate": "Mid-year population estimate",
            "nbs_land_value_households": "NBS land value: households",
            "nbs_land_value_nfc": "NBS land value: non-financial corporations",
            "nbs_land_value_total": "NBS land value: total",
            "population": "Population",
            "pse_headcount_total_public_sector": "Public-sector employment headcount",
        },
        "scotgov": {
            "chargeable_dwellings_band_a": "Chargeable dwellings: Council Tax band A",
            "chargeable_dwellings_band_b": "Chargeable dwellings: Council Tax band B",
            "chargeable_dwellings_band_c": "Chargeable dwellings: Council Tax band C",
            "chargeable_dwellings_band_d": "Chargeable dwellings: Council Tax band D",
            "chargeable_dwellings_band_e": "Chargeable dwellings: Council Tax band E",
            "chargeable_dwellings_band_f": "Chargeable dwellings: Council Tax band F",
            "chargeable_dwellings_band_g": "Chargeable dwellings: Council Tax band G",
            "chargeable_dwellings_band_h": "Chargeable dwellings: Council Tax band H",
            "chargeable_dwellings_total": "Total chargeable dwellings",
            "social_security_assistance_spending": (
                "Social security assistance spending"
            ),
        },
        "slc": {
            "maintenance_loan_amount_paid": "Maintenance loan amount paid",
            "maintenance_loan_recipients": "Maintenance loan recipients",
            "student_loan_borrowers": "Student loan borrowers",
            "student_loan_net_repayments_plan_1": (
                "Student loan net repayments: Plan 1"
            ),
            "student_loan_net_repayments_plan_2_full_time": (
                "Student loan net repayments: full-time Plan 2"
            ),
            "student_loan_net_repayments_total_higher_education": (
                "Student loan net repayments: higher education total"
            ),
            "targeted_support_amount_awarded": "Targeted support amount awarded",
            "targeted_support_recipients": "Targeted support recipients",
        },
        "voa": {
            "ct_stock_all_properties": "Council Tax stock: all properties",
            "ct_stock_band_a": "Council Tax stock: band A",
            "ct_stock_band_b": "Council Tax stock: band B",
            "ct_stock_band_c": "Council Tax stock: band C",
            "ct_stock_band_d": "Council Tax stock: band D",
            "ct_stock_band_e": "Council Tax stock: band E",
            "ct_stock_band_f": "Council Tax stock: band F",
            "ct_stock_band_g": "Council Tax stock: band G",
            "ct_stock_band_h": "Council Tax stock: band H",
        },
    }
)


CALIBRATION_VARIABLE_LABELS_BY_COUNTRY: Mapping[
    str, Mapping[str, Mapping[str, str]]
] = MappingProxyType(
    {
        "uk": UK_CALIBRATION_VARIABLE_LABELS,
        "us": US_CALIBRATION_VARIABLE_LABELS,
    }
)


def calibration_variable_label(
    country: str,
    source_id: str,
    variable_id: str,
) -> str | None:
    """Return a country-owned label for a provider's statistic category."""

    country_labels = CALIBRATION_VARIABLE_LABELS_BY_COUNTRY.get(country.strip().lower())
    if country_labels is None:
        return None
    source_labels = country_labels.get(source_id.strip())
    if source_labels is None:
        return None
    return source_labels.get(variable_id.strip())
