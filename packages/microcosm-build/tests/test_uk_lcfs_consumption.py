from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.lcfs_consumption import (
    BUS_FARE_LCFS_CODES,
    LCFS_ACCOMM_MAP,
    LCFS_TENURE_MAP,
    UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
    assign_recipient_has_fuel,
    clean_lcfs_consumption_table,
    derive_energy_from_lcfs,
    support_clip_to_donor,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame


def _household() -> pd.DataFrame:
    base = {
        "case": [1, 2, 3, 4],
        "g018": [2, 1, 3, 1],
        "g019": [0, 1, 2, 0],
        "gorx": [7, 12, 10, 1],
        "p389p": [100.0, 200.0, 300.0, 400.0],
        "p344p": [150.0, 250.0, 350.0, 450.0],
        "weighta": [1.5, 2.0, 2.5, 3.0],
        "a122": [4, 8, 5, 7],
        "a121": [4, 5, 6, 7],
        "b226": [6.0, 0.0, 0.0, 0.0],
        "b489": [0.0, 9.0, 8.0, 0.0],
        "b490": [0.0, 4.0, 0.0, 0.0],
        "p537": [10.0, 20.0, 30.0, -1.0],
    }
    for source in (
        "p601",
        "p602",
        "p603",
        "p604",
        "p605",
        "p606",
        "p607",
        "p608",
        "p609",
        "p610",
        "p611",
        "p612",
        "c72211",
        "c72212",
        *BUS_FARE_LCFS_CODES,
    ):
        base[source] = [1.0, 2.0, 3.0, 4.0]
    return pd.DataFrame(base)


def _person() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [1, 1, 3],
            "b303p": [10.0, 5.0, 7.0],
            "b3262p": [1.0, 2.0, 3.0],
            "p049p": [4.0, 5.0, 6.0],
        }
    )


def test_lcfs_donor_cleaning_arithmetic_and_lossy_maps() -> None:
    donor = clean_lcfs_consumption_table(_person(), _household())

    assert donor["region"].tolist() == [
        "LONDON",
        "NORTHERN_IRELAND",
        "WALES",
        "NORTH_EAST",
    ]
    assert LCFS_TENURE_MAP[4] == "RENT_PRIVATELY"
    assert LCFS_TENURE_MAP[8] == "RENT_PRIVATELY"
    assert LCFS_ACCOMM_MAP[4] == "FLAT"
    assert LCFS_ACCOMM_MAP[5] == "FLAT"
    assert donor["household_weight"].tolist() == [1500.0, 2000.0, 2500.0, 3000.0]
    assert np.isclose(
        donor.loc[0, "employment_income"],
        (10.0 + 5.0) * (365.25 / 7),
    )
    assert donor.loc[1, "employment_income"] == 0.0
    assert np.isclose(donor.loc[0, "bus_fare_spending"], 3.0 * (365.25 / 7))


def test_energy_split_exercises_four_cases_fallback_and_clamp() -> None:
    household = _household()
    split = derive_energy_from_lcfs(household)

    assert split["electricity_consumption"].tolist() == [6.0, 5.0, 4.8, 0.0]
    assert split["gas_consumption"].tolist() == [4.0, 4.0, 3.2, 0.0]

    fallback = household.copy()
    fallback["b226"] = 0.0
    fallback["b489"] = 0.0
    fallback["p537"] = [10.0, 20.0, 30.0, 40.0]

    split = derive_energy_from_lcfs(fallback)

    np.testing.assert_allclose(
        split["electricity_consumption"], [5.2, 10.4, 15.6, 20.8]
    )
    np.testing.assert_allclose(split["gas_consumption"], [4.8, 9.6, 14.4, 19.2])


def test_recipient_has_fuel_is_conditioned_on_vehicle_count_and_deterministic() -> None:
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [10, 20],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {
                "household_id": [10, 20],
                "household_weight": [1.0, 1.0],
                "num_vehicles": [0, 2],
            }
        ),
        time_period="2023",
    )

    first = assign_recipient_has_fuel(frame, rate=1.0, seed=0)
    second = assign_recipient_has_fuel(frame, rate=1.0, seed=0)

    assert first.tolist() == [False, True]
    assert second.tolist() == first.tolist()


def test_support_clip_exempts_raked_energy_columns() -> None:
    donor = pd.DataFrame(
        {column: [1.0, 5.0] for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS}
    )
    draws = pd.DataFrame(
        {column: [0.0, 10.0] for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS}
    )

    clipped = support_clip_to_donor(
        draws,
        donor,
        exempt={
            "electricity_consumption",
            "gas_consumption",
            "domestic_energy_consumption",
        },
    )

    assert clipped["food_and_non_alcoholic_beverages_consumption"].tolist() == [
        1.0,
        5.0,
    ]
    assert clipped["electricity_consumption"].tolist() == [0.0, 10.0]


def test_has_fuel_bridge_accepts_lcfs_native_predictor_names() -> None:
    # Regression for the licensed-build failure: the LCFS donor frame carries
    # hbai_household_net_income / is_adult / is_child, not the WAS names the
    # bridge model is fit on. The bridge must rename before predicting.
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_HAS_FUEL_PREDICTORS,
        bridge_has_fuel_to_lcfs,
    )

    rng = np.random.default_rng(7)
    n = 120
    was = pd.DataFrame(
        {
            "household_net_income": rng.uniform(1e4, 6e4, n),
            "num_adults": rng.integers(1, 4, n).astype(float),
            "num_children": rng.integers(0, 3, n).astype(float),
            "private_pension_income": rng.uniform(0, 1e4, n),
            "employment_income": rng.uniform(0, 5e4, n),
            "self_employment_income": rng.uniform(0, 1e4, n),
            "region": rng.choice(["LONDON", "WALES"], n),
            "num_vehicles": rng.integers(0, 3, n).astype(float),
            "weight": rng.uniform(0.5, 2.0, n),
        }
    )
    lcfs = pd.DataFrame(
        {
            "hbai_household_net_income": [2e4, 3e4, 4e4],
            "is_adult": [1.0, 2.0, 3.0],
            "is_child": [0.0, 1.0, 2.0],
            "private_pension_income": [0.0, 1e3, 2e3],
            "employment_income": [1e4, 2e4, 3e4],
            "self_employment_income": [0.0, 0.0, 5e3],
            "region": ["LONDON", "WALES", "LONDON"],
        }
    )
    assert not set(UK_LCFS_HAS_FUEL_PREDICTORS) <= set(lcfs.columns)

    first, record = bridge_has_fuel_to_lcfs(lcfs, was, seed=0, n_estimators=10)
    second, _ = bridge_has_fuel_to_lcfs(lcfs, was, seed=0, n_estimators=10)

    assert record.fit_name.endswith("has_fuel")
    values = first["has_fuel_consumption"].to_numpy(dtype=float)
    assert ((values >= 0.0) & (values <= 1.0)).all()
    assert first["has_fuel_consumption"].tolist() == (
        second["has_fuel_consumption"].tolist()
    )
