"""policyengine-uk charges vehicle excise duty on the LCFS car step's inputs (microcosm#1095 B3).

The ``assign_car_attributes_from_nts`` step writes five person inputs no
earlier stage wrote. The export-surface gate checks only that they are
allow-listed, so these checks hold the step to the engine it is locked with:
each is a person input there whose default is what a person without a car
keeps, the fuel types are the engine's names, the duty regimes start where
the engine's do, each CO2 band's midpoint is charged its own band's rate, and
on the step's ISO dates the engine charges each regime's duty.
"""

from __future__ import annotations

from datetime import date

import pytest

from microcosm.build.uk_runtime.nts_car_attributes import (
    UK_CAR_INPUT_COLUMNS,
    UK_CAR_INPUT_DEFAULTS,
    UK_FIRST_YEAR_CO2_MIDPOINTS,
    UK_GRADUATED_CO2_MIDPOINTS,
    UK_VED_GRADUATED_START,
    UK_VED_MODERN_START,
)

#: The step's reference date, 1 October 2025, sits in rate year 2025-26.
YEAR = 2025


@pytest.fixture(scope="module")
def system():
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


@pytest.fixture(scope="module")
def duty(system):
    return system.parameters(f"{YEAR}-06-01").gov.dft.vehicle_excise_duty


def test_every_car_input_is_an_engine_person_input_with_the_steps_default(
    system,
) -> None:
    for column in UK_CAR_INPUT_COLUMNS:
        variable = system.variables[column]
        assert variable.entity.key == "person", column
        assert variable.definition_period == "year", column
        assert not variable.formulas, column
        default = variable.default_value
        if column == "car_fuel_type":
            default = default.name
        elif column == "car_first_registration_date":
            default = default.isoformat()
        assert default == UK_CAR_INPUT_DEFAULTS[column], column
    assert {
        value.name for value in system.variables["car_fuel_type"].possible_values
    } == {
        "PETROL",
        "DIESEL_RDE2",
        "DIESEL_NON_RDE2",
        "ALTERNATIVE_FUEL",
        "ELECTRIC",
    }
    # The list price the step leaves alone charges no supplement.
    assert system.variables["car_list_price"].default_value == 0


def test_the_duty_regimes_start_where_the_engines_do(duty) -> None:
    assert date(*duty.modern_registration_start) == UK_VED_MODERN_START
    assert date(*duty.graduated_registration_start) == UK_VED_GRADUATED_START


def test_each_band_midpoint_is_charged_its_own_bands_rate(duty) -> None:
    """The NTS bands are the engine's brackets in order, one to one."""

    for scale, midpoints in (
        (duty.first_year, UK_FIRST_YEAR_CO2_MIDPOINTS),
        (duty.graduated, UK_GRADUATED_CO2_MIDPOINTS),
    ):
        assert len(scale.thresholds) == len(midpoints)
        for band, midpoint in midpoints.items():
            assert scale.calc(midpoint, right=True) == scale.amounts[band - 1], band


def test_the_engine_charges_each_regimes_duty_on_the_steps_inputs(duty) -> None:
    from policyengine_uk import Simulation

    cars = {
        # Registered in the rate year: the first-year rate.
        "new": ("PETROL", 121.0, 1400.0, "2025-06-01", False),
        "new_diesel": ("DIESEL_NON_RDE2", 121.0, 2000.0, "2025-06-01", False),
        # The standard rate after the first licence, electric cars too.
        "modern": ("PETROL", 121.0, 1400.0, "2019-05-01", False),
        "electric": ("ELECTRIC", 0.0, 0.0, "2020-05-01", False),
        # Table A to M by CO2, then engine size before March 2001.
        "graduated": ("DIESEL_NON_RDE2", 158.0, 2000.0, "2010-05-01", False),
        "engine": ("PETROL", 0.0, 2000.0, "1999-01-01", False),
        "exempt": ("PETROL", 121.0, 1400.0, "2019-05-01", True),
        "none": tuple(UK_CAR_INPUT_DEFAULTS[c] for c in UK_CAR_INPUT_COLUMNS),
    }
    people = {
        name: {"age": {YEAR: 40}}
        | {
            column: {YEAR: value}
            for column, value in zip(UK_CAR_INPUT_COLUMNS, inputs, strict=True)
        }
        for name, inputs in cars.items()
    }
    simulation = Simulation(
        situation={
            "people": people,
            "benunits": {f"{name}_unit": {"members": [name]} for name in people},
            "households": {"household": {"members": list(people)}},
        }
    )
    charged = dict(
        zip(people, simulation.calculate("car_vehicle_excise_duty", YEAR), strict=True)
    )
    assert charged == pytest.approx(
        {
            "new": duty.first_year.calc(121.0, right=True),
            "new_diesel": duty.first_year_diesel_non_rde2.calc(121.0, right=True),
            "modern": duty.standard_rate,
            "electric": duty.standard_rate,
            "graduated": duty.graduated.calc(158.0, right=True),
            "engine": duty.large_engine_rate,
            "exempt": 0.0,
            "none": 0.0,
        }
    )
    assert simulation.calculate("vehicle_excise_duty", YEAR).sum() == pytest.approx(
        sum(charged.values())
    )
