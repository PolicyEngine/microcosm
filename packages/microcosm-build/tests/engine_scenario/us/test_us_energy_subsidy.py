"""Tests split from packages/microcosm-build/tests/test_us_energy_subsidy.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_energy_subsidy import *


def test_policyengine_2_2_1_contract_and_live_neutralization() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "spm_unit"
    assert variable.value_type is float
    assert variable.default_value == 0
    assert str(variable.definition_period).lower() == "year"

    situation = {
        "people": {"adult": {"age": {"2024": 40}}},
        "tax_units": {"tax_unit": {"members": ["adult"]}},
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {
            "spm_unit": {
                "members": ["adult"],
                _OUTPUT: {"2024": 2_400.0},
            }
        },
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
        "marital_units": {"marital_unit": {"members": ["adult"]}},
    }

    class NeutralizeEnergySubsidy(Reform):
        def apply(self) -> None:
            self.neutralize_variable(_OUTPUT)

    baseline = Simulation(situation=situation)
    neutralized = Simulation(
        situation=situation,
        reform=NeutralizeEnergySubsidy,
    )
    assert baseline.calculate(_OUTPUT, 2024)[0] == pytest.approx(2_400.0)
    assert neutralized.calculate(_OUTPUT, 2024)[0] == 0.0
    for downstream in ("spm_unit_benefits", "spm_unit_net_income"):
        effect = (
            baseline.calculate(downstream, 2024)[0]
            - neutralized.calculate(downstream, 2024)[0]
        )
        assert effect == pytest.approx(2_400.0)
