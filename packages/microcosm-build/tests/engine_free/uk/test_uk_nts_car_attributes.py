"""policyengine-uk's vehicle excise duty inputs from the NTS cars (microcosm#1095 B3, VED)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from microcosm.build.uk_runtime.nts_car_attributes import (
    UK_CAR_INPUT_COLUMNS,
    UK_CAR_INPUT_DEFAULTS,
    UK_FIRST_YEAR_CO2_MIDPOINTS,
    UK_GRADUATED_CO2_MIDPOINTS,
    UK_VED_GRADUATED_START,
    UK_VED_MODERN_START,
    _donor_pools,
    assign_car_attributes,
)

PARAMETERS = {
    "seed": 0,
    "salt": "test_car_attributes",
    "reference_date": "2025-10-01",
    "oldest_age_years": 25,
    "default_engine_size_cc": 1600,
}


def _nts_cars() -> pd.DataFrame:
    """Kept cars in every fifth: a petrol, a diesel, an electric car, each at three ages."""

    rows = []
    for fifth in range(1, 6):
        for fuel, band, graduated in ((1, 7, 4), (2, 8, 6), (3, 1, 1)):
            for age in (2, 9, 13):
                rows.append(
                    {
                        "householdid": fifth,
                        "carown_b01id": 1,
                        "vehproptypets_b01id": fuel,
                        "vehco2bands_b01id": band,
                        "vehco2tax2008_b01id": graduated,
                        "enginecap": 0 if fuel == 3 else 1400,
                        "vehage_b01id": age,
                        "hhincqisengts_b01id": fifth,
                        "w3": 1.0,
                    }
                )
    # Neither a company car nor a car without a linked band is a donor.
    rows.append({**rows[0], "carown_b01id": 2})
    rows.append({**rows[0], "vehco2bands_b01id": -8})
    return pd.DataFrame(rows)


def _frame():
    person = pd.DataFrame(
        {
            "person_id": [11, 12, 13, 21, 31, 32, 41],
            "person_household_id": [1, 1, 1, 2, 3, 3, 4],
            "person_benunit_id": [1, 1, 1, 2, 3, 3, 4],
            "age": [45, 50, 6, 30, 40, 38, 70],
            "company_car_list_price": [0, 0, 0, 0, 30000.0, 0, 0],
            # The child's DLA higher-rate mobility exempts a car in the unit.
            "dla_m_category": [
                "NONE",
                "NONE",
                "HIGHER",
                "NONE",
                "NONE",
                "NONE",
                "NONE",
            ],
            "pip_m_category": [
                "NONE",
                "NONE",
                "NONE",
                "NONE",
                "NONE",
                "NONE",
                "ENHANCED",
            ],
        }
    )
    return person, {
        "household_ids": [1, 2, 3, 4],
        "vehicles": [2, 1, 2, 3],
        "weights": [1.0, 2.0, 1.0, 1.0],
        "has_fuel": [True, False, True, True],
        "fifths": [5, 1, 3, 2],
        "zero_emission_share": [0.5, 0.02, 0.02, 0.02],
    }


def test_cars_go_to_adults_oldest_first_and_skip_company_cars() -> None:
    person, households = _frame()
    result = assign_car_attributes(
        person=person, cars=_nts_cars(), parameters=PARAMETERS, **households
    )
    cars = result.person
    holds = cars["car_first_registration_date"] != "9999-01-01"
    # Household 1: both adults hold a car, the child none. Household 3's
    # company-car holder keeps the defaults; its other adult holds its own
    # car. Household 4 has one adult for three cars.
    assert holds.tolist() == [True, True, False, True, False, True, True]
    for column, value in UK_CAR_INPUT_DEFAULTS.items():
        assert cars.loc[2, column] == value
        assert cars.loc[4, column] == value
    receipt = result.receipt
    # Own cars at weight: 2 + 2 x 1 + 1 + 3; assigned 2 + 2 + 1 + 1.
    assert receipt["weighted_own_cars"] == pytest.approx(8.0)
    assert receipt["assigned_share_of_own_cars"] == pytest.approx(6 / 8)
    assert receipt["dropped_share_of_own_cars"] == pytest.approx(2 / 8)
    assert receipt["weighted_company_cars_left_out"] == pytest.approx(1.0)


def test_fuel_follows_the_households_fuel_flag() -> None:
    person, households = _frame()
    cars = assign_car_attributes(
        person=person, cars=_nts_cars(), parameters=PARAMETERS, **households
    ).person
    # Household 2 buys no fuel: its car is electric with no CO2 or engine.
    assert cars.loc[3, "car_fuel_type"] == "ELECTRIC"
    assert cars.loc[3, "car_co2_emissions"] == 0.0
    assert cars.loc[3, "car_engine_size"] == 0.0
    # Household 1's first car, the oldest adult's, burns fuel.
    assert cars.loc[1, "car_fuel_type"] in {
        "PETROL",
        "DIESEL_RDE2",
        "DIESEL_NON_RDE2",
        "ALTERNATIVE_FUEL",
    }
    assert set(cars["car_fuel_type"]) <= {
        "PETROL",
        "DIESEL_RDE2",
        "DIESEL_NON_RDE2",
        "ALTERNATIVE_FUEL",
        "ELECTRIC",
    }


def test_registration_ages_the_car_from_the_reference_date_and_sets_its_co2() -> None:
    person, households = _frame()
    cars = assign_car_attributes(
        person=person, cars=_nts_cars(), parameters=PARAMETERS, **households
    ).person
    reference = date(2025, 10, 1)
    for row in cars.index[cars["car_first_registration_date"] != "9999-01-01"]:
        registered = date.fromisoformat(cars.loc[row, "car_first_registration_date"])
        years = (reference - registered).days / 365.25
        # The donors' bands: half a year to a year, 6-7 and 13-18 years.
        assert any(
            low - 0.01 <= years <= high + 0.01
            for low, high in ((0.5, 1.0), (6.0, 7.0), (13.0, 18.0))
        )
        co2 = cars.loc[row, "car_co2_emissions"]
        fuel = cars.loc[row, "car_fuel_type"]
        if fuel == "ELECTRIC":
            assert co2 == 0.0
        elif registered >= UK_VED_MODERN_START:
            assert co2 in UK_FIRST_YEAR_CO2_MIDPOINTS.values()
        else:
            assert registered >= UK_VED_GRADUATED_START
            assert co2 in UK_GRADUATED_CO2_MIDPOINTS.values()
        if fuel.startswith("DIESEL"):
            rde2 = registered >= date(2021, 1, 1)
            assert fuel == ("DIESEL_RDE2" if rde2 else "DIESEL_NON_RDE2")


def test_a_mobility_component_exempts_one_car() -> None:
    person, households = _frame()
    cars = assign_car_attributes(
        person=person, cars=_nts_cars(), parameters=PARAMETERS, **households
    ).person
    # The child holds no car: the unit's oldest holder's car is exempt. The
    # 70-year-old with PIP's enhanced mobility exempts their own.
    assert cars["car_is_ved_exempt"].tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]


def test_draws_are_keyed_and_reproducible() -> None:
    person, households = _frame()
    first = assign_car_attributes(
        person=person, cars=_nts_cars(), parameters=PARAMETERS, **households
    ).person
    # The person table's order does not move a draw.
    second = assign_car_attributes(
        person=person.sample(frac=1.0, random_state=3),
        cars=_nts_cars(),
        parameters=PARAMETERS,
        **households,
    ).person.sort_index()
    assert first[list(UK_CAR_INPUT_COLUMNS)].equals(second[list(UK_CAR_INPUT_COLUMNS)])
    other = assign_car_attributes(
        person=person,
        cars=_nts_cars(),
        parameters={**PARAMETERS, "salt": "other"},
        **households,
    ).person
    assert not first[list(UK_CAR_INPUT_COLUMNS)].equals(
        other[list(UK_CAR_INPUT_COLUMNS)]
    )


def test_each_linked_car_stands_for_its_fifth_and_age_bands_kept_cars() -> None:
    cars = _nts_cars()
    # Half fifth 2's newest diesels lack the registration record.
    newest = (cars["hhincqisengts_b01id"] == 2) & (cars["vehage_b01id"] == 2)
    cars = pd.concat(
        [
            cars,
            cars[newest & (cars["vehproptypets_b01id"] == 2)].assign(
                vehco2bands_b01id=-8, vehco2tax2008_b01id=-8
            ),
        ],
        ignore_index=True,
    )
    pools, linked_share = _donor_pools(cars)
    # 3 cars a band in fifth 2, one unlinked beside the newest: 4 / 3.
    fuel = pools[(2, "fuel_burning")]
    assert fuel.loc[fuel["age"] == 2, "weight"].tolist() == pytest.approx([4 / 3] * 2)
    assert fuel.loc[fuel["age"] != 2, "weight"].tolist() == pytest.approx([1.0] * 4)
    zero = pools[(2, "zero_emission")]
    assert zero.loc[zero["age"] == 2, "weight"].tolist() == pytest.approx([4 / 3])
    # Fifth 1's unlinked car: 45 linked of 47 kept.
    assert linked_share == pytest.approx(45 / 47)
    # The newest band's kept cars hold their place in the draw.
    assert fuel["weight"].sum() + zero["weight"].sum() == pytest.approx(10.0)


def test_refuses_a_fifth_without_donors() -> None:
    person, households = _frame()
    cars = _nts_cars()
    cars = cars[
        ~((cars["hhincqisengts_b01id"] == 3) & (cars["vehco2bands_b01id"] == 1))
    ]
    with pytest.raises(ValueError, match="no zero_emission car in fifth 3"):
        assign_car_attributes(
            person=person, cars=cars, parameters=PARAMETERS, **households
        )
    # Kept cars of an age band no linked car shares cannot be stood for.
    unlinked = _nts_cars().assign(
        vehco2bands_b01id=lambda c: c["vehco2bands_b01id"].where(
            ~((c["hhincqisengts_b01id"] == 4) & (c["vehage_b01id"] == 9)), -8
        )
    )
    with pytest.raises(ValueError, match="fifth 4 and age band 9 hold no car"):
        assign_car_attributes(
            person=person, cars=unlinked, parameters=PARAMETERS, **households
        )
    with pytest.raises(ValueError, match="belong to a frame household"):
        assign_car_attributes(
            person=person.assign(person_household_id=[1, 1, 1, 2, 3, 3, 9]),
            cars=_nts_cars(),
            parameters=PARAMETERS,
            **households,
        )
