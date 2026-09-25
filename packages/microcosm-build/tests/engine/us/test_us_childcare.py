"""Tests split from packages/microcosm-build/tests/test_us_childcare.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_childcare import *


def test_policyengine_us_contract_is_spm_unit_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    variable = variables[_OUTPUT]

    assert variable.is_input_variable()
    assert variable.entity.key == "spm_unit"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0
    assert not variables["cdcc_relevant_expenses"].is_input_variable()
