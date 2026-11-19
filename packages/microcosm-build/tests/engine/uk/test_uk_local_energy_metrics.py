"""The local energy metrics read the stored spend where it is stored (#1207).

policyengine-uk uprates ``electricity_consumption`` and ``gas_consumption``
by CPI away from the period they were supplied for, so the local metrics
read them at the dataset period: the stage's calendar-2024 spend, turned
back into the kWh it raked at the stage's own prices.
"""

from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.uk_runtime.domestic_energy_targets import (
    household_energy_kwh,
    household_energy_prices,
)
from microcosm.build.uk_runtime.frs_annual_income import (
    ANNUAL_SHARE_COLUMNS,
    OTHER_SPELLS_COLUMNS,
)
from microcosm.build.uk_runtime.local_targets import compute_household_metrics
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind

STORED = 2024
PERIOD = 2025


def _resolver(tmp_path) -> UKMeasureResolver:
    """One adult's household, its spend stored for the dataset period."""

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_benunit_id": [1],
                "person_household_id": [1],
                "age": [40.0],
                # A full year at today's income: the annual measures the
                # local hmrc rows read are the current ones.
                **{column: [1.0] for column in ANNUAL_SHARE_COLUMNS},
                **{column: [0.0] for column in OTHER_SPELLS_COLUMNS},
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [1.0],
                "region": ["NORTH_WEST"],
                "tenure_type": ["OWNED_OUTRIGHT"],
                "main_residence_value": [250_000.0],
                "council_tax": [0.0],
                "rent": [0.0],
                "electricity_consumption": [900.0],
                "gas_consumption": [700.0],
            }
        ),
        time_period=str(STORED),
        weight_kind=WeightKind.DESIGN,
    )
    return UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=PERIOD
    )


class _RecordingEngine:
    """The engine, recording the period of each variable the resolver reads."""

    def __init__(self, engine):
        self._engine = engine
        self.periods: dict[str, set] = {}

    def calculate(self, variable, period=None, *args, **kwargs):
        self.periods.setdefault(variable, set()).add(period)
        return self._engine.calculate(variable, period, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)


def test_la_energy_metrics_read_the_stored_spend_at_the_dataset_period(
    tmp_path, monkeypatch
):
    resolver = _resolver(tmp_path)
    engine = _RecordingEngine(resolver.simulation)
    monkeypatch.setattr(resolver, "simulation", engine)

    metrics = compute_household_metrics(
        resolver, "la", period=PERIOD, input_period=STORED
    )

    kwh = household_energy_kwh(
        [900.0], [700.0], region=["NORTH_WEST"], prices=household_energy_prices()
    )
    assert metrics["energy/electricity_kwh"].tolist() == pytest.approx(
        kwh["electricity_kwh"].tolist()
    )
    assert metrics["energy/gas_kwh"].tolist() == pytest.approx(kwh["gas_kwh"].tolist())
    assert metrics["energy/gas_connected_households"].tolist() == [1.0]
    # Read where they are stored, never where the engine would uprate them.
    for variable in ("electricity_consumption", "gas_consumption", "region"):
        assert engine.periods[variable] == {STORED}, variable
