"""Tests split from packages/microcosm-build/tests/test_us_salt_refund_income.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_salt_refund_income import *


def test_policyengine_us_contract_and_state_tax_binding() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    variable = CountryTaxBenefitSystem().variables["salt_refund_income"]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    year = 2024
    adult = {
        "age": {year: 40},
        "employment_income": {year: 100_000},
        "salt_refund_income": {year: 10_000},
    }
    entities = {
        "tax_units": {"unit": {"members": ["adult"]}},
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {year: "SC"},
            }
        },
        "marital_units": {"marital": {"members": ["adult"]}},
    }
    baseline = Simulation(situation={"people": {"adult": adult}, **entities})
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "salt_refund_income_neutralization"
    )
    neutralized = Simulation(
        situation={"people": {"adult": adult}, **entities},
        reform=_build_reform(probe),
    )

    assert baseline.calculate("sc_subtractions", year)[0] == pytest.approx(10_000)
    assert neutralized.calculate("sc_subtractions", year)[0] == 0
    assert (
        baseline.calculate("state_income_tax", year)[0]
        < neutralized.calculate("state_income_tax", year)[0]
    )
