"""Tests split from packages/microcosm-build/tests/test_us_retirement_distributions.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_retirement_distributions import *


def test_policyengine_2_2_1_contract_and_live_binding() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variables = CountryTaxBenefitSystem().variables
    for name in _OUTPUTS:
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert variable.value_type is float
        assert variable.default_value == 0
        assert str(variable.definition_period).lower() == "year"

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 65},
                "employment_income": {"2024": 60_000},
                "keogh_distributions": {"2024": 10_000},
            }
        },
        "tax_units": {"tax_unit": {"members": ["adult"]}},
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }

    class NeutralizeKeogh(Reform):
        def apply(self) -> None:
            self.neutralize_variable("keogh_distributions")

    baseline = Simulation(situation=situation)
    neutralized = Simulation(situation=situation, reform=NeutralizeKeogh)
    assert baseline.calculate("taxable_retirement_distributions", 2024)[0] == 10_000
    assert neutralized.calculate("taxable_retirement_distributions", 2024)[0] == 0
    for measure in ("irs_gross_income", "adjusted_gross_income"):
        assert baseline.calculate(measure, 2024)[0] == pytest.approx(
            neutralized.calculate(measure, 2024)[0] + 10_000
        )
    assert (
        baseline.calculate("income_tax", 2024)[0]
        > neutralized.calculate("income_tax", 2024)[0]
    )
