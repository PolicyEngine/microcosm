"""Tests split from packages/microcosm-build/tests/test_us_misc_itemized.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_misc_itemized import *


def test_policyengine_us_contract_is_a_person_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variable = CountryTaxBenefitSystem().variables[
        "unreimbursed_business_employee_expenses"
    ]

    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0
