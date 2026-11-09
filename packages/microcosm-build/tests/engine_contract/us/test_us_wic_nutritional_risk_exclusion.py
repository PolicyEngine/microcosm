"""Tests split from packages/microcosm-build/tests/test_us_wic_nutritional_risk_exclusion.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_wic_nutritional_risk_exclusion import *


def test_policyengine_2_2_1_defaults_risk_true_and_uses_it_for_eligibility() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "2.2.1"
    system = CountryTaxBenefitSystem()
    variable = system.variables["is_wic_at_nutritional_risk"]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "month"
    assert variable.value_type is bool
    assert variable.default_value is True

    eligibility_formula = system.variables["is_wic_eligible"].get_formula("2024-01")
    assert eligibility_formula is not None
    assert 'person("is_wic_at_nutritional_risk", period)' in inspect.getsource(
        eligibility_formula
    )
