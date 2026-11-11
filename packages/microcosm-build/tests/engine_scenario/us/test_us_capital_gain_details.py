"""Tests split from packages/microcosm-build/tests/test_us_capital_gain_details.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_capital_gain_details import *


def test_policyengine_us_contracts_and_household_bindings() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    system = CountryTaxBenefitSystem()
    collectibles = system.variables["long_term_capital_gains_on_collectibles"]
    unrecaptured = system.variables["unrecaptured_section_1250_gain"]
    assert collectibles.is_input_variable()
    assert collectibles.entity.key == "person"
    assert unrecaptured.is_input_variable()
    assert unrecaptured.entity.key == "tax_unit"

    adult = {
        "age": {2024: 45},
        "employment_income": {2024: 120_000},
        "long_term_capital_gains_before_response": {2024: 100_000},
        "long_term_capital_gains_on_collectibles": {2024: 25_000},
    }
    entities = {
        "tax_units": {
            "unit": {
                "members": ["adult"],
                "filing_status": {2024: "SINGLE"},
                "unrecaptured_section_1250_gain": {2024: 20_000},
            }
        },
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code_str": {2024: "MA"},
            }
        },
        "marital_units": {"marital": {"members": ["adult"]}},
    }
    baseline = Simulation(situation={"people": {"adult": adult}, **entities})

    probes = {probe.id: probe for probe in us_release_reform_coverage_probes()}
    no_collectibles = Simulation(
        situation={"people": {"adult": adult}, **entities},
        reform=_build_reform(probes["collectibles_gain_neutralization"]),
    )
    no_unrecaptured = Simulation(
        situation={"people": {"adult": adult}, **entities},
        reform=_build_reform(probes["unrecaptured_section_1250_gain_neutralization"]),
    )

    assert (
        baseline.calculate("income_tax", 2024)[0]
        > no_collectibles.calculate("income_tax", 2024)[0]
    )
    assert (
        baseline.calculate("income_tax", 2024)[0]
        > no_unrecaptured.calculate("income_tax", 2024)[0]
    )
