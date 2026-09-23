"""Tests split from packages/microcosm-build/tests/test_us_form_4952.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_form_4952 import *


def test_policyengine_us_contract_and_net_capital_gain_binding() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    variable = CountryTaxBenefitSystem().variables[
        "investment_income_elected_form_4952"
    ]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    adult = {
        "age": {2024: 45},
        "long_term_capital_gains_before_response": {2024: 100_000},
        "qualified_dividend_income": {2024: 0},
        "short_term_capital_gains": {2024: 0},
    }
    entities = {
        "tax_units": {
            "unit": {"members": ["adult"], "filing_status": {2024: "SINGLE"}}
        },
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code_str": {2024: "CA"},
            }
        },
        "marital_units": {"marital": {"members": ["adult"]}},
    }
    baseline = Simulation(situation={"people": {"adult": adult}, **entities})
    election = Simulation(
        situation={
            "people": {
                "adult": {
                    **adult,
                    "investment_income_elected_form_4952": {2024: 25_000},
                }
            },
            **entities,
        }
    )

    assert baseline.calculate("net_capital_gain", 2024)[0] == pytest.approx(100_000)
    assert election.calculate("net_capital_gain", 2024)[0] == pytest.approx(75_000)
    assert (
        election.calculate("income_tax", 2024)[0]
        > baseline.calculate("income_tax", 2024)[0]
    )

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "form_4952_election_neutralization"
    )
    neutralized = Simulation(
        situation={
            "people": {
                "adult": {
                    **adult,
                    "investment_income_elected_form_4952": {2024: 25_000},
                }
            },
            **entities,
        },
        reform=_build_reform(probe),
    )
    assert (
        election.calculate("income_tax", 2024)[0]
        > neutralized.calculate("income_tax", 2024)[0]
    )
