"""Tests split from packages/microcosm-build/tests/test_us_ssi_disability_criteria.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_ssi_disability_criteria import *


def test_policyengine_us_2_2_1_ssi_is_positive_then_zero_when_neutralized() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert variable.value_type is bool
    assert variable.default_value is False

    def situation(criterion: bool) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2024": 40},
                    _OUTPUT: {"2024": criterion},
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

    assert active.calculate("ssi", "2024-01")[0] == pytest.approx(943.0)
    assert neutralized.calculate("ssi", "2024-01")[0] == 0.0
