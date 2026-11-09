"""Tests split from packages/microcosm-build/tests/test_us_wic_claim.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_wic_claim import *


def test_policyengine_contract_and_live_wic_neutralization() -> None:
    import inspect

    from policyengine_us import CountryTaxBenefitSystem, Simulation

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    assert version("policyengine-us") == "2.2.1"
    system = CountryTaxBenefitSystem()
    variable = system.variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert variable.value_type is bool
    assert bool(variable.default_value) is True
    assert str(variable.definition_period).lower() == "month"
    mother_formula = inspect.getsource(system.variables["is_mother"].formula)
    assert 'person("is_parent", period)' in mother_formula
    assert "female & has_children" in mother_formula

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "wic_claim_neutralization"
    )
    situation = {
        "people": {
            "mother": {
                "age": {"2024": 27},
                "is_pregnant": {"2024": True},
                "is_wic_at_nutritional_risk": {"2024": True},
                _OUTPUT: {"2024": True},
            }
        },
        "tax_units": {"tax_unit": {"members": ["mother"]}},
        "families": {"family": {"members": ["mother"]}},
        "spm_units": {"spm_unit": {"members": ["mother"]}},
        "households": {
            "household": {
                "members": ["mother"],
                "state_code": {"2024": "CA"},
            }
        },
        "marital_units": {"marital_unit": {"members": ["mother"]}},
    }
    baseline = Simulation(situation=situation)
    neutralized = Simulation(situation=situation, reform=_build_reform(probe))
    assert baseline.calculate("wic", 2024)[0] > 0
    assert neutralized.calculate("wic", 2024)[0] == 0
