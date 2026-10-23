"""policyengine-uk charges excise on the inputs the LCFS stage writes (microcosm#1359).

The ``level_excise_quantities`` step writes policyengine-uk's household
quantity inputs: litres a year of each alcohol product, cigarettes a week and
grams a week of the other tobacco products, which alcohol and tobacco duty
read (PolicyEngine/policyengine-uk#1876). The export-surface gate checks a
column only against the enhanced-FRS reference, so these checks hold the step
to the engine it is locked with: every column it can write is a household
input there, the engine's week is the step's, the strengths a declaration
divides by are the ones the engine multiplies back, and the duties the
engine charges on the inputs are linear in them and reach household tax.
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

#: A household's quantities for each kind of input the step writes.
QUANTITIES = {
    "beer_litres": 100.0,
    "wine_litres": 50.0,
    "spirits_litres": 10.0,
    "cigarettes_per_week": 20.0,
    "hand_rolling_tobacco_grams_per_week": 10.0,
    "other_tobacco_grams_per_week": 5.0,
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


def test_declared_strengths_are_the_engines() -> None:
    """A strength the step divides by is the one the engine multiplies back."""

    parameters = importlib.import_module("policyengine_uk.system").system.parameters
    committed = excise_quantities_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    declarations = [EXCISE_DECLARATION, *([] if committed is None else [committed])]
    checked = 0
    for declaration in declarations:
        instant = f"{declaration['period_value']}-04-30"
        for column, output in declaration["outputs"].items():
            strength = output.get("strength")
            if strength is None:
                continue
            node = parameters
            for part in strength["parameter_path"].split("."):
                node = getattr(node, part)
            assert float(node(instant)) == strength["value"], column
            checked += 1
    assert checked >= 3


def _duties(year: int, scale: float) -> tuple[float, float, float]:
    from policyengine_uk import Simulation

    household = {column: {year: value * scale} for column, value in QUANTITIES.items()}
    simulation = Simulation(
        situation={
            "people": {"adult": {"age": {year: 40}}},
            "benunits": {"benunit": {"members": ["adult"]}},
            "households": {"household": {"members": ["adult"], **household}},
        }
    )
    return tuple(
        float(simulation.calculate(variable, year)[0])
        for variable in ("alcohol_duty", "tobacco_duty", "household_tax")
    )


@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_the_engine_charges_duty_in_proportion_to_the_quantities(year: int) -> None:
    none = _duties(year, 0.0)
    one = _duties(year, 1.0)
    two = _duties(year, 2.0)

    assert none[:2] == (0.0, 0.0)
    assert one[0] > 0 and one[1] > 0
    assert two[0] == pytest.approx(2 * one[0])
    assert two[1] == pytest.approx(2 * one[1])
    # Household tax carries both duties.
    assert one[2] - none[2] == pytest.approx(one[0] + one[1])
