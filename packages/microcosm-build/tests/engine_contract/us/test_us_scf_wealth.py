"""Tests split from packages/microcosm-build/tests/test_us_scf_wealth.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_scf_wealth import *


def test_policyengine_2_2_1_net_worth_input_contract() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables["net_worth"]
    assert variable.is_input_variable()
    assert variable.entity.key == "household"
    assert variable.value_type is float
    assert variable.default_value == 0
    assert str(variable.definition_period).lower() == "year"
