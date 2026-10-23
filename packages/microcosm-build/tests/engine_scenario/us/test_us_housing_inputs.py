"""Tests split from packages/microcosm-build/tests/test_us_housing_inputs.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_housing_inputs import *


def test_policyengine_us_input_contracts_and_rent_consumer() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    expected_entities = {
        "pre_subsidy_rent": "person",
        "receives_housing_assistance": "spm_unit",
        "takes_up_housing_assistance_if_eligible": "spm_unit",
        "spm_unit_tenure_type": "spm_unit",
        "tenure_type": "household",
    }
    for name, entity in expected_entities.items():
        variable = system.variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == entity
    assert "pre_subsidy_rent" in system.variables["hud_gross_rent"].adds


def test_policyengine_us_live_housing_take_up_neutralization() -> None:
    from importlib.metadata import version

    from policyengine_us import Simulation

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    assert version("policyengine-us") == "2.60.4"
    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 0},
                "pre_subsidy_rent": {"2024": 12_000},
            }
        },
        "tax_units": {"tu": {"members": ["adult"]}},
        "families": {"fam": {"members": ["adult"]}},
        "spm_units": {
            "spm": {
                "members": ["adult"],
                "receives_housing_assistance": {"2024": True},
                "takes_up_housing_assistance_if_eligible": {"2024": True},
                "spm_unit_tenure_type": {"2024": "RENTER"},
            }
        },
        "households": {
            "hh": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
                "county_fips": {"2024": "06037"},
                "bedrooms": {"2024": 1},
            }
        },
        "marital_units": {"mu": {"members": ["adult"]}},
    }
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "housing_assistance_take_up_neutralization"
    )

    baseline = Simulation(situation=situation)
    neutralized = Simulation(situation=situation, reform=_build_reform(probe))

    assert baseline.calculate("is_eligible_for_housing_assistance", 2024)[0]
    # policyengine-us 2.60.4 adds LACDA's utility allowance to gross rent
    # ($271 a month for a one-bedroom in county 06037; 2.2.1 had none):
    # $12,000 rent + $3,252 allowance - $25 minimum rent.
    assert baseline.calculate("hud_utility_allowance", 2024)[0] == pytest.approx(3_252)
    assert baseline.calculate("housing_assistance", 2024)[0] == pytest.approx(15_227)
    assert neutralized.calculate("housing_assistance", 2024)[0] == 0
