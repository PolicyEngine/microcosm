"""Tests split from packages/microcosm-build/tests/test_us_domestic_production.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_domestic_production import *


def test_policyengine_us_contract_is_a_tax_unit_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variable = CountryTaxBenefitSystem().variables["domestic_production_ald"]

    assert variable.is_input_variable()
    assert variable.entity.key == "tax_unit"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0


def test_2024_reactivation_probe_binds_only_when_the_input_is_populated() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "domestic_production_ald_reactivation"
    )
    reform = Reform.from_dict(dict(probe.parameter_changes), country_id="us")
    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 100_000},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
                "domestic_production_ald": {"2024": 10_000},
            }
        },
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    baseline = Simulation(situation=situation)
    reformed = Simulation(
        tax_benefit_system=CountryTaxBenefitSystem(reform=(reform,)),
        situation=situation,
    )

    assert baseline.calculate("above_the_line_deductions", 2024)[0] == 0.0
    assert reformed.calculate("above_the_line_deductions", 2024)[0] == 10_000.0
    effect = (
        baseline.calculate("income_tax", 2024)[0]
        - reformed.calculate("income_tax", 2024)[0]
    )
    assert effect > 1_000.0
