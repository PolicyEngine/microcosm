"""The locked engine's bus pricing parameters beside the stage's (microcosm#1207)."""

from __future__ import annotations

import policyengine_uk
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.bus_fare_pricing import (
    UK_ENGINE_BUS_AREA_BY_STAGE_AREA,
    bus_engine_parity,
    bus_fare_prices,
)
from microcosm.build.uk_runtime.lcfs_consumption import (
    UK_LCFS_VENDORED_RESOURCES,
    bus_pricing_operation,
)
from microcosm.build.uk_runtime.measure_simulation import (
    engine_bus_pricing_parameters,
)


def test_engine_bus_parameters_stand_beside_the_stages() -> None:
    system = policyengine_uk.CountryTaxBenefitSystem()
    engine = engine_bus_pricing_parameters(system, 2025)
    bus = system.parameters("2025-01-01").gov.dft.bus
    for name in ("boardings_per_trip", "yield_per_boarding"):
        assert set(engine[name]) == set(UK_ENGINE_BUS_AREA_BY_STAGE_AREA.values())
        for area, value in engine[name].items():
            assert value == float(getattr(getattr(bus, name), area))
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    prices = bus_fare_prices(
        bus_pricing_operation(stage), allowed_resources=UK_LCFS_VENDORED_RESOURCES
    )
    parity = bus_engine_parity(prices, engine)
    # The engine's values come from the same published derivation (#954): a
    # re-pin or a restatement can open a small gap, which the receipt marks; a
    # gap of a whole per cent would mean the two derive different things.
    for entry in parity["areas"].values():
        for name in ("boardings_per_trip", "yield_per_boarding"):
            assert abs(entry[name]["relative_gap"]) < 0.01, (name, entry)
    assert isinstance(parity["within_tolerance"], bool)
    assert parity["tolerance"] == pytest.approx(0.001)
