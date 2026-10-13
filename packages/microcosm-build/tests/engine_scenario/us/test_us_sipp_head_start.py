"""Tests split from packages/microcosm-build/tests/test_us_sipp_head_start.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_sipp_head_start import *


def test_policyengine_us_2_2_1_head_start_is_zero_when_neutralized() -> None:
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
                "child": {
                    "age": {"2024": 4},
                    _OUTPUT: {"2024": takes_up},
                }
            },
            "tax_units": {
                "unit": {
                    "members": ["child"],
                    "filing_status": {"2024": "SINGLE"},
                }
            },
            "families": {"family": {"members": ["child"]}},
            "spm_units": {"spm": {"members": ["child"]}},
            "households": {
                "household": {
                    "members": ["child"],
                    "state_code": {"2024": "CA"},
                }
            },
            "marital_units": {"marital": {"members": ["child"]}},
        }

    active = Simulation(situation=situation(True))
    neutralized = Simulation(situation=situation(False))
    assert active.calculate("head_start", "2024")[0] > 0.0
    assert neutralized.calculate("head_start", "2024")[0] == 0.0
