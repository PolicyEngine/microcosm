"""Country-owned display labels for programmatically declared categories.

A category may combine several Chronicle facts or measures, so its display
label belongs with Microcosm's calibration grouping rather than with any one
source fact.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "CALIBRATION_VARIABLE_LABELS_BY_COUNTRY",
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
        "bea": {
            "nipa_proprietors_income": "Proprietors' income",
            "nipa_wages_and_salaries": "Wages and salaries",
        },
        "bea_nipa": {
            "proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments": (
                "Proprietors' income"
            ),
            "wages_and_salaries": "Wages and salaries",
        },
        "cbo": {
            "cbo": "Economic projections",
            "adjusted_gross_income_projection": "Adjusted gross income projection",
            "net_business_income_projection": "Net business income projection",
            "net_capital_gain_projection": "Net capital gain projection",
            "qualified_dividend_income_projection": (
                "Qualified dividend income projection"
            ),
            "wages_and_salaries_projection": "Wages and salaries projection",
        },
        "census_acs": {
            "resident_population": "Resident population",
            "population_by_age": "Population by age",
            "population_by_sex_and_age": "Population by sex and age",
        },
        "census_pep": {
            "resident_population": "Resident population",
        },
        "census_stc": {
            "individual_income_tax_collections": ("Individual income tax collections"),
        },
        "cms_aca": {
            "cms_aca": "ACA marketplace",
            "aptc_consumers": "APTC consumers",
            "marketplace_plan_selections": "Marketplace plan selections",
        },
        "cms_medicaid": {
            "cms_medicaid": "Medicaid and CHIP",
            "total_chip_enrollment": "Total CHIP enrollment",
            "total_medicaid_chip_enrollment": "Total Medicaid and CHIP enrollment",
            "total_medicaid_enrollment": "Total Medicaid enrollment",
        },
        "cms_medicare": {
            "cms_medicare": "Medicare",
            "part_b_premium_income": "Part B premium income",
        },
        "federal_reserve_z1": {
            "federal_reserve.z1.households_nonprofits_net_worth": (
                "Households and nonprofit organizations net worth"
            ),
        },
        "federal_reserve": {
            "federal_reserve": "Household finances",
        },
        "hhs_acf_liheap": {
            "hhs_acf_liheap": "LIHEAP",
            "households_served_by_state_programs": (
                "Households served by state programs"
            ),
        },
        "hhs_acf_tanf": {
            "hhs_acf_tanf": "TANF",
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
            "miscellaneous_income": "Miscellaneous income",
            "miscellaneous_losses": "Miscellaneous losses",
            "non_sch_d_capital_gains": "Non-Schedule D capital gains",
            "ordinary_dividend_income": "Ordinary dividend income",
            "partnership_and_s_corp_income": "Partnership and S corporation income",
            "qualified_dividends": "Qualified dividends",
            "qualified_business_income_deduction": (
                "Qualified business income deduction"
            ),
            "real_estate_taxes": "Real estate taxes",
            "refundable_ctc": "Refundable CTC",
            "rent_and_royalty_net_income": "Rent and royalty net income",
            "salt_deduction": "SALT deduction",
            "tax_exempt_interest_income": "Tax-exempt interest income",
            "taxable_income": "Taxable income",
            "taxable_interest_income": "Taxable interest income",
            "taxable_pension_income": "Taxable pension income",
            "taxable_social_security": "Taxable Social Security",
            "tax_filer_individual_count": "Individuals represented by tax returns",
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
        "ssa": {
            "ssa": "Social Security",
            "ssi_payments": "SSI payments",
            "ssi_recipients": "SSI recipients",
        },
        "unspecified": {
            "selection_mass_protection.keogh_distributions": (
                "Keogh distributions selection-mass constraint"
            ),
        },
        "usda_snap": {
            "average_monthly_households": "Average monthly households",
            "usda_snap": "SNAP",
            "total_benefits": "Total benefits",
        },
    }
)


CALIBRATION_VARIABLE_LABELS_BY_COUNTRY: Mapping[
    str, Mapping[str, Mapping[str, str]]
] = MappingProxyType({"us": US_CALIBRATION_VARIABLE_LABELS})


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
