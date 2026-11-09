"""Tests split from packages/microcosm-build/tests/test_us_ssi_take_up.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_ssi_take_up import *


def test_policyengine_us_2_2_1_take_up_flag_controls_positive_ssi() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert variable.value_type is bool
    assert variable.default_value is True

    def situation(takes_up: bool) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2024": 40},
                    "meets_ssi_disability_criteria": {"2024": True},
                    _OUTPUT: {"2024": takes_up},
                }
            },
            "tax_units": {
                "unit": {
                    "members": ["adult"],
                    "filing_status": {"2024": "SINGLE"},
                }
            },
            "families": {"family": {"members": ["adult"]}},
            "spm_units": {"spm": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    "state_code": {"2024": "CA"},
                }
            },
            "marital_units": {"marital": {"members": ["adult"]}},
        }

    active = Simulation(situation=situation(True))
    neutralized = Simulation(situation=situation(False))
    period = "2024-12"
    assert active.calculate("uncapped_ssi", period)[0] == pytest.approx(943.0)
    assert neutralized.calculate("uncapped_ssi", period)[0] == pytest.approx(943.0)
    assert active.calculate("ssi", period)[0] == pytest.approx(943.0)
    assert neutralized.calculate("ssi", period)[0] == 0.0
