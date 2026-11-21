"""Tests split from packages/microcosm-build/tests/test_us_alimony.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_alimony import *


def test_policyengine_us_contract_includes_strike_person_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for name in ("alimony_income", "alimony_expense", "strike_benefits"):
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert str(variable.definition_period).lower() == "year"
        assert variable.default_value == 0

    # policyengine-us 2.145.3 routes the deduction through a person-level
    # leaf that reads alimony_expense (and divorce_year for the TCJA cut-off).
    assert "alimony_expense_ald_person" in (
        variables["alimony_expense_ald"].formula.__code__.co_consts
    )
    assert "alimony_expense" in {
        constant
        for formula in variables["alimony_expense_ald_person"].formulas.values()
        for constant in formula.__code__.co_consts
    }
