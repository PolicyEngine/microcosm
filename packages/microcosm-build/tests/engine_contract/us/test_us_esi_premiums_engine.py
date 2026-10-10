"""Engine contract of the two ESI premium inputs (microcosm #454).

``docs/us-esi-employer-premiums.md`` and the ``meps_esi_premiums`` stage notes
say how PolicyEngine-US consumes each input. These tests compute it in the
locked engine, so an engine release that reroutes either input fails here
rather than silently changing what the stage's columns mean.
"""

from __future__ import annotations

import pytest

from microcosm.build.us_runtime import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PRE_TAX_PREMIUM_COLUMN,
)

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


def test_both_inputs_are_person_year_input_leaves() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for name in (US_ESI_EMPLOYER_PREMIUM_COLUMN, US_ESI_PRE_TAX_PREMIUM_COLUMN):
        variable = variables[name]
        assert variable.is_input_variable(), name
        assert variable.entity.key == "person", name
        assert str(variable.definition_period).lower() == "year", name


def test_the_engine_parameter_lists_name_the_inputs() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    parameters = CountryTaxBenefitSystem().parameters
    instant = f"{PERIOD}-01-01"
    assert US_ESI_EMPLOYER_PREMIUM_COLUMN in (
        parameters.gov.household.cbo_market_income_additions(instant)
    )
    assert US_ESI_PRE_TAX_PREMIUM_COLUMN in (
        parameters.gov.irs.gross_income.pre_tax_contributions(instant)
    )
    assert US_ESI_PRE_TAX_PREMIUM_COLUMN in (
        parameters.gov.irs.gross_income.fica_pre_tax_contributions(instant)
    )


def test_a_pre_tax_premium_lowers_income_tax_and_fica_wages_dollar_for_dollar(
    baseline,
) -> None:
    premium = 2_000.0
    reduced = _totals(**{US_ESI_PRE_TAX_PREMIUM_COLUMN: premium})

    for measure in (
        "irs_employment_income",
        "payroll_tax_gross_wages",
        "adjusted_gross_income",
    ):
        assert reduced[measure] - baseline[measure] == pytest.approx(-premium), measure
    assert baseline["irs_employment_income"] == pytest.approx(WAGES)
    assert reduced["income_tax"] < baseline["income_tax"]
    # 6.2% employee OASDI on the premium no longer in FICA wages.
    assert reduced["employee_social_security_tax"] - baseline[
        "employee_social_security_tax"
    ] == pytest.approx(-0.062 * premium)


def test_an_employer_premium_moves_only_cbo_household_market_income(baseline) -> None:
    premium = 10_000.0
    with_premium = _totals(**{US_ESI_EMPLOYER_PREMIUM_COLUMN: premium})

    assert with_premium["cbo_household_market_income"] - baseline[
        "cbo_household_market_income"
    ] == pytest.approx(premium)
    for measure in MEASURES:
        if measure != "cbo_household_market_income":
            assert with_premium[measure] == pytest.approx(baseline[measure]), measure
