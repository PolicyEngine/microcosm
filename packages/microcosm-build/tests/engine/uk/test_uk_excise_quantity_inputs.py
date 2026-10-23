"""policyengine-uk charges excise on the inputs the LCFS stage writes (microcosm#1359).

The ``level_excise_quantities`` step writes policyengine-uk's household
excise inputs: litres a year of each alcohol product and the beer and cider
draught shares, cigarettes a week and grams a week of the other tobacco
products, which alcohol and tobacco duty read (PolicyEngine/policyengine-uk#1876),
and the annual gambling duty bases (PolicyEngine/policyengine-uk#2590, from
policyengine-uk 2.151.0). The export-surface gate checks a column only
against the enhanced-FRS reference, so these checks hold the step to the
engine it is locked with: every column it can write is a household input
there, the engine's week is the step's, the strengths and rates a
declaration divides by are the ones the engine multiplies back, and the
duties the engine charges on the inputs are linear in them and reach
household tax.
"""

from __future__ import annotations

import importlib

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.lcfs_consumption import (
    UK_LCFS_EXCISE_OUTPUT_COLUMNS,
    UK_PE_UK_WEEKS_IN_YEAR,
    excise_quantities_operation,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_RELEASE_EXPORT_DROPPED_COLUMNS,
)
from test_support.microcosm_build.uk_excise_facts import EXCISE_DECLARATION

#: A household's quantities for each kind of alcohol and tobacco input.
QUANTITIES = {
    "beer_litres": 100.0,
    "wine_litres": 50.0,
    "spirits_litres": 10.0,
    "cigarettes_per_week": 20.0,
    "hand_rolling_tobacco_grams_per_week": 10.0,
    "other_tobacco_grams_per_week": 5.0,
}
#: A household's gambling bases, pounds a year.
GAMBLING_BASES = {
    "remote_gaming_net_losses": 200.0,
    "land_based_fixed_odds_betting_net_losses": 150.0,
    "remote_other_betting_net_losses": 100.0,
    "machine_games_standard_rate_net_takings": 80.0,
    "national_lottery_ticket_spending": 120.0,
    "land_based_casino_net_losses": 50.0,
}


def test_every_excise_output_is_an_engine_household_input() -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for column in UK_LCFS_EXCISE_OUTPUT_COLUMNS:
        variable = variables[column]
        assert variable.entity.key == "household", column
        assert variable.definition_period == "year", column
        assert variable.value_type is float, column
        assert not variable.formulas, column
        assert column not in UK_RELEASE_EXPORT_DROPPED_COLUMNS["household"], column


def test_the_engines_week_is_the_steps() -> None:
    from policyengine_core.commons.formulas import WEEKS_IN_YEAR

    assert WEEKS_IN_YEAR == UK_PE_UK_WEEKS_IN_YEAR


def _declared_divisors(declaration) -> dict[str, dict]:
    """Every strength and gambling rate a declaration divides by."""

    divisors = {
        column: output["strength"]
        for column, output in declaration["outputs"].items()
        if output.get("strength") is not None
    }
    divisors.update(
        {
            f"gambling.{name}": rate
            for name, rate in declaration.get("gambling", {}).get("rates", {}).items()
        }
    )
    return divisors


def test_declared_strengths_and_rates_are_the_engines() -> None:
    """A strength or rate the step divides by is the one the engine applies."""

    parameters = importlib.import_module("policyengine_uk.system").system.parameters
    committed = excise_quantities_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    declarations = [EXCISE_DECLARATION, *([] if committed is None else [committed])]
    checked = 0
    for declaration in declarations:
        instant = f"{declaration['period_value']}-04-30"
        for name, divisor in _declared_divisors(declaration).items():
            node = parameters
            for part in divisor["parameter_path"].split("."):
                node = getattr(node, part)
            assert float(node(instant)) == pytest.approx(divisor["value"]), name
            checked += 1
    assert checked >= 12


def _simulate(year: int, inputs: dict[str, float], variables: tuple[str, ...]):
    from policyengine_uk import Simulation

    household = {column: {year: value} for column, value in inputs.items()}
    simulation = Simulation(
        situation={
            "people": {"adult": {"age": {year: 40}}},
            "benunits": {"benunit": {"members": ["adult"]}},
            "households": {"household": {"members": ["adult"], **household}},
        }
    )
    return tuple(float(simulation.calculate(name, year)[0]) for name in variables)


@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_the_engine_charges_duty_in_proportion_to_the_quantities(year: int) -> None:
    names = ("alcohol_duty", "tobacco_duty", "household_tax")
    none, one, two = (
        _simulate(year, {k: v * scale for k, v in QUANTITIES.items()}, names)
        for scale in (0.0, 1.0, 2.0)
    )

    assert none[:2] == (0.0, 0.0)
    assert one[0] > 0 and one[1] > 0
    assert two[0] == pytest.approx(2 * one[0])
    assert two[1] == pytest.approx(2 * one[1])
    # Household tax carries both duties.
    assert one[2] - none[2] == pytest.approx(one[0] + one[1])


@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_the_engine_charges_gambling_duty_in_proportion_to_the_bases(
    year: int,
) -> None:
    names = ("gambling_duty", "household_tax")
    none, one, two = (
        _simulate(year, {k: v * scale for k, v in GAMBLING_BASES.items()}, names)
        for scale in (0.0, 1.0, 2.0)
    )

    assert none[0] == 0.0
    assert one[0] > 0
    assert two[0] == pytest.approx(2 * one[0])
    assert one[1] - none[1] == pytest.approx(one[0])
