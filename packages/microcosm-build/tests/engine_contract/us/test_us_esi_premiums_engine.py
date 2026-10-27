"""Engine contract of the ESI employer premium input (microcosm #454).

``docs/us-esi-employer-premiums.md`` and the ``meps_esi_premiums`` stage notes
say how PolicyEngine-US consumes the input. These tests compute it in the
locked engine, so an engine release that reroutes it fails here rather than
silently changing what the stage's column means.
"""

from __future__ import annotations

import pytest

from microcosm.build.us_runtime import US_ESI_EMPLOYER_PREMIUM_COLUMN

PERIOD = 2024
WAGES = 60_000.0
MEASURES = (
    "irs_employment_income",
    "payroll_tax_gross_wages",
    "adjusted_gross_income",
    "income_tax",
    "employee_social_security_tax",
    "cbo_household_market_income",
)


def _totals(**person_inputs: float) -> dict[str, float]:
    from policyengine_us import Simulation

    person = {"age": {PERIOD: 40}, "employment_income": {PERIOD: WAGES}}
    person |= {name: {PERIOD: value} for name, value in person_inputs.items()}
    simulation = Simulation(
        situation={
            "people": {"worker": person},
            "tax_units": {"tax_unit": {"members": ["worker"]}},
            "spm_units": {"spm_unit": {"members": ["worker"]}},
            "families": {"family": {"members": ["worker"]}},
            "marital_units": {"marital_unit": {"members": ["worker"]}},
            "households": {
                "household": {"members": ["worker"], "state_code": {PERIOD: "TX"}}
            },
        }
    )
    return {
        measure: float(simulation.calculate(measure, PERIOD).sum())
        for measure in MEASURES
    }


@pytest.fixture(scope="module")
def baseline() -> dict[str, float]:
    return _totals()


def test_the_input_is_a_person_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variable = CountryTaxBenefitSystem().variables[US_ESI_EMPLOYER_PREMIUM_COLUMN]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"


def test_the_engine_parameter_list_names_the_input() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    parameters = CountryTaxBenefitSystem().parameters
    assert US_ESI_EMPLOYER_PREMIUM_COLUMN in (
        parameters.gov.household.cbo_market_income_additions(f"{PERIOD}-01-01")
    )


def test_an_employer_premium_moves_only_cbo_household_market_income(baseline) -> None:
    premium = 10_000.0
    with_premium = _totals(**{US_ESI_EMPLOYER_PREMIUM_COLUMN: premium})

    assert with_premium["cbo_household_market_income"] - baseline[
        "cbo_household_market_income"
    ] == pytest.approx(premium)
    for measure in MEASURES:
        if measure != "cbo_household_market_income":
            assert with_premium[measure] == pytest.approx(baseline[measure]), measure
