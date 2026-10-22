"""The local energy metrics read the stored spend where it is stored (#1207).

policyengine-uk uprates ``electricity_consumption`` and ``gas_consumption``
by CPI away from the period they were supplied for, so the local metrics
read them at the dataset period: the stage's calendar-2024 spend, turned
back into the kWh it raked at the stage's own prices.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.domestic_energy_targets import (
    household_energy_kwh,
    household_energy_prices,
)
from microcosm.build.uk_runtime.local_targets import compute_household_metrics

STORED = 2024
PERIOD = 2025


def _simulation():
    from policyengine_uk import Simulation

    return Simulation(
        situation={
            "people": {"adult": {"age": {str(STORED): 40}}},
            "benunits": {"benunit": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    "region": {str(STORED): "NORTH_WEST"},
                    "electricity_consumption": {str(STORED): 900.0},
                    "gas_consumption": {str(STORED): 700.0},
                }
            },
        }
    )


def test_la_energy_metrics_read_the_stored_spend_at_the_dataset_period():
    sim = _simulation()

    metrics = compute_household_metrics(sim, "la", period=PERIOD, input_period=STORED)

    kwh = household_energy_kwh(
        [900.0], [700.0], region=["NORTH_WEST"], prices=household_energy_prices()
    )
    assert metrics["energy/electricity_kwh"].tolist() == pytest.approx(
        kwh["electricity_kwh"].tolist()
    )
    assert metrics["energy/gas_kwh"].tolist() == pytest.approx(kwh["gas_kwh"].tolist())
    assert metrics["energy/gas_connected_households"].tolist() == [1.0]
    # At the calibration period the engine does not return the stored spend.
    at_period = float(np.asarray(sim.calculate("electricity_consumption", PERIOD))[0])
    assert at_period != pytest.approx(900.0)
