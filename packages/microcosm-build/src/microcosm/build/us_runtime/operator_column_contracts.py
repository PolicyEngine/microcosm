"""Source-visible operator column declarations, without operation imports.

The operation modules re-export these same objects. Source boundary checks can
inspect ownership without loading transfer code or its reference tables.
"""

from __future__ import annotations

# Declared by qbi_inputs.py; operations retain their historical imports.

_GENERAL_QUALIFICATION_FLAGS: tuple[str, ...] = (
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
)

_SSTB_QUALIFICATION_FLAG = "sstb_self_employment_income_would_be_qualified"

US_QBI_BOOLEAN_OUTPUT_COLUMNS: tuple[str, ...] = (
    *_GENERAL_QUALIFICATION_FLAGS,
    _SSTB_QUALIFICATION_FLAG,
    # Keep the classifier last so the chained QRF can condition the SSTB draw
    # on the qualification flags it must agree with.
    "business_is_sstb",
)

US_QBI_NONNEGATIVE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "sstb_unadjusted_basis_qualified_property",
    "sstb_w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)

US_QBI_OUTPUT_COLUMNS: tuple[str, ...] = (
    *US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "sstb_self_employment_income_before_lsr",
    "sstb_unadjusted_basis_qualified_property",
    "sstb_w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)


# Declared by acs_transfer.py; operations retain their historical imports.

#: Person columns the default transfer DERIVES deterministically after the
#: QRF fits (never fitted themselves). Coverage checks require them on the
#: recipient exactly like declared plan targets.
ACS_DERIVED_TRANSFER_INPUTS: tuple[str, ...] = (
    "schedule_d_capital_gain_distributions",
)


# Declared by puf_support.py; operations retain their historical imports.

# Shared structural limit; importing it must not load the PUF donor operations.
PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID = 10**15 - 1

US_PUF_SUPPORT_STAGE_NAME = "puf_support_channel"

PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "tax_exempt_interest_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "long_term_capital_gains_on_collectibles",
    "non_sch_d_capital_gains",
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
    "alimony_income",
    "alimony_expense",
    "salt_refund_income",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "real_estate_taxes",
    "home_mortgage_interest",
    "investment_interest_expense",
    "investment_income_elected_form_4952",
    "student_loan_interest",
    "educator_expense",
    "qualified_tuition_expenses",
    "casualty_loss",
    "unreimbursed_business_employee_expenses",
    # The engine owns the realized contribution amounts through the
    # IRA-limit scale and self-employment caps; the persistable leaves are
    # the desired contributions, equal to the PUF's observed deductions at
    # baseline (issue #278).
    "traditional_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
    "rental_income",
    "estate_income",
    "farm_income",
    "farm_operations_income",
    "farm_rent_income",
    "miscellaneous_income",
    "partnership_income",
    "s_corp_income",
    "partnership_self_employment_net_earnings",
    *US_QBI_OUTPUT_COLUMNS,
)

PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS: tuple[str, ...] = (
    "domestic_production_ald",
    "unrecaptured_section_1250_gain",
    "first_home_mortgage_balance",
    "second_home_mortgage_balance",
    "first_home_mortgage_interest",
    "second_home_mortgage_interest",
    "first_home_mortgage_origination_year",
    "second_home_mortgage_origination_year",
    "health_savings_account_ald",
)


# Declared by puf_capital_gains_tail.py; operations retain their historical imports.

PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS = (
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "long_term_capital_gains_on_collectibles",
    "non_sch_d_capital_gains",
)

PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS = ("unrecaptured_section_1250_gain",)

PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN = "puf_capital_gains_tail_transfer_applied"

PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN = "puf_capital_gains_tail_donor_source_id"

PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN = (
    "puf_capital_gains_tail_donor_is_synthetic"
)

PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN = (
    "puf_capital_gains_tail_donor_filing_status_code"
)

PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN = (
    "puf_capital_gains_tail_donor_agi_band_index"
)

PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN = "puf_capital_gains_tail_transfer_weight"
