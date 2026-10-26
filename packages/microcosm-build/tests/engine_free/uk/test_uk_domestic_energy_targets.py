"""DESNZ sub-national domestic energy as local-authority targets (#1207)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.desnz_constituency_energy import (
    UKDesnzConstituencyEnergyFacts,
)
from microcosm.build.uk_runtime.domestic_energy_targets import (
    GREAT_BRITAIN_GEOGRAPHY_ID,
    NORTHERN_IRELAND_GEOGRAPHY_ID,
    NORTHERN_IRELAND_PERIOD_TYPES,
    SUBNATIONAL_CONSUMPTION_CONCEPTS,
    UK_ENERGY_LEVEL_HOLD,
    UK_ENERGY_LEVEL_INDEX_CONCEPT,
    UK_GAS_CONNECTED_HOLD,
    UK_GAS_METER_SHARE_INDEX_CONCEPT,
    UK_LOCAL_ENERGY_RESOURCE,
    _meter_rows_by_roster_area,
    align_census_households_by_gas_meter_share,
    align_energy_to_stage_level,
    declared_energy_pricing,
    gb_energy_level_factors,
    household_energy_kwh,
    household_energy_prices,
    northern_ireland_consumption,
)
from microcosm.build.uk_runtime.energy_pricing import (
    ELECTRICITY_KWH,
    GAS_KWH,
    SUBNATIONAL_METER_CONCEPTS,
    SUBNATIONAL_PERIOD_TYPES,
    UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
    published_energy_level,
)
from microcosm.build.uk_runtime.lcfs_consumption import (
    energy_kwh_to_spend,
    energy_spend_to_kwh,
    lcfs_energy_pricing,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.ledger_targets import UK_UPRATING_APPLIERS
from microcosm.build.uk_runtime.uprating_holds import is_uprating_hold
from microcosm.calibrate import TargetRegistry, TargetSpec

ELECTRICITY_TARGET = "desnz.subnational.electricity_kwh"
GAS_CONNECTED_TARGET = "desnz.subnational.gas_connected_households"


def _northern_ireland_row(column: str, value: float, **changes) -> dict:
    fuel = "electricity" if column == ELECTRICITY_KWH else "gas"
    row = {
        "concept": SUBNATIONAL_CONSUMPTION_CONCEPTS[column],
        "geography": {"id": NORTHERN_IRELAND_GEOGRAPHY_ID, "level": "country"},
        "period": {"type": NORTHERN_IRELAND_PERIOD_TYPES[fuel], "value": 2024},
        "dimensions": {
            "metric": "consumption",
            "fuel": fuel,
            "domestic_scope": "all_domestic",
        },
        "unit": "kwh",
        "value": value,
    }
    return {**row, **changes}


# Northern Ireland's all-domestic consumption beside Great Britain's
# vendored totals (synthetic values; the feed's rows are vendored).
_NORTHERN_IRELAND = [
    _northern_ireland_row(ELECTRICITY_KWH, 3.0e9),
    _northern_ireland_row(GAS_KWH, 1.0e9),
]


#: The period types the feed stamps (microcosm#1300), spelled out rather than
#: read from the constant under test.
_FEED_PERIOD_TYPES = {"electricity": "calendar_year", "gas": "gas_year"}


def _meter_row(
    fuel: str, area: str, value: float, *, period_type: str | None = None
) -> dict:
    # Gas rows carry DESNZ's gas year and electricity rows the calendar year,
    # as the feed stamps them (microcosm#1300).
    return {
        "concept": SUBNATIONAL_METER_CONCEPTS[fuel],
        "geography": {"id": area, "level": "local_authority"},
        "period": {"type": period_type or _FEED_PERIOD_TYPES[fuel], "value": 2024},
        "dimensions": {"fuel": fuel, "metric": "meter_count"},
        "unit": "count",
        "value": value,
    }


def _cell(
    target_id: str, area: str, value: float, *, fact_period: str = "2024"
) -> TargetSpec:
    return TargetSpec(
        name=f"{target_id}@{area}@2025",
        entity="household",
        value=value,
        measure=target_id,
        period=2025,
        family="domestic_energy",
        source="test",
        metadata={
            "contract_target_id": target_id,
            "geography_id": area,
            "ledger_fact_period": fact_period,
        },
    )


def _reference(target_id: str, index: str, **selector: object) -> LedgerTargetReference:
    return LedgerTargetReference(
        name=target_id,
        ledger_selector={"source_name": "desnz", **selector},
        entity="household",
        measure=target_id,
        period=2025,
        uprating_index=index,
    )


def test_household_kwh_is_the_stage_s_own_conversion():
    """The metric reads back the kWh the stage priced: one inverse, both ways."""

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    energy = lcfs_energy_pricing(stage)
    assert household_energy_prices() == energy.prices
    rng = np.random.default_rng(1207)
    region = np.resize(np.array(sorted(energy.prices.region_of_frs_region)), 36)
    kwh = pd.DataFrame(
        {
            ELECTRICITY_KWH: rng.uniform(500.0, 6000.0, len(region)),
            GAS_KWH: np.where(
                np.arange(len(region)) % 3 == 0,
                0.0,
                rng.uniform(2000.0, 20000.0, len(region)),
            ),
        }
    )
    spend = energy_kwh_to_spend(kwh, energy=energy, region=region)

    back = household_energy_kwh(
        spend["electricity_consumption"],
        spend["gas_consumption"],
        region=region,
        prices=energy.prices,
    )

    np.testing.assert_allclose(back[ELECTRICITY_KWH], kwh[ELECTRICITY_KWH], rtol=1e-12)
    np.testing.assert_allclose(back[GAS_KWH], kwh[GAS_KWH], rtol=1e-12)
    stage_kwh = energy_spend_to_kwh(spend, energy=energy, region=region)
    np.testing.assert_array_equal(back[ELECTRICITY_KWH], stage_kwh[ELECTRICITY_KWH])
    np.testing.assert_array_equal(back[GAS_KWH], stage_kwh[GAS_KWH])


def test_household_kwh_refuses_misaligned_columns():
    with pytest.raises(ValueError, match="align"):
        household_energy_kwh(
            [900.0, 800.0],
            [0.0],
            region=["LONDON", "WALES"],
            prices=household_energy_prices(),
        )


def test_the_level_factor_puts_great_britain_on_the_stage_level():
    # microcosm#1278: Great Britain's part of the UK level is its share of the
    # sub-national total with Northern Ireland's row added, so each factor is
    # the UK level over the UK sub-national total.
    factors, receipt = gb_energy_level_factors(local_rows=_NORTHERN_IRELAND)

    level, _ = published_energy_level(declared_energy_pricing())
    northern_ireland = {ELECTRICITY_KWH: 3.0e9, GAS_KWH: 1.0e9}
    for column, concept in SUBNATIONAL_CONSUMPTION_CONCEPTS.items():
        (great_britain,) = vendored_rows(
            UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
            concept=concept,
            geography_id=GREAT_BRITAIN_GEOGRAPHY_ID,
            period_value=2024,
        )
        united_kingdom = great_britain["value"] + northern_ireland[column]
        assert factors[column] == pytest.approx(level[column] / united_kingdom)
        fuel = receipt["by_fuel"][column]
        assert fuel["great_britain_share"] == pytest.approx(
            great_britain["value"] / united_kingdom
        )
        assert fuel["subnational_northern_ireland_kwh"] == northern_ireland[column]
    assert receipt["period_value"] == 2024
    assert receipt["northern_ireland"]["periods"] == {
        ELECTRICITY_KWH: {"period_type": "fiscal_year", "period_value": 2024},
        GAS_KWH: {"period_type": "gas_year", "period_value": 2024},
    }


def test_the_vendored_resource_carries_northern_ireland_s_two_rows():
    # The abf0921 feed's Northern Ireland row per fuel, all domestic meters:
    # electricity on the 2024 financial year, gas on the 2024 gas year.
    consumption, receipt = northern_ireland_consumption(2024)
    assert set(consumption) == {ELECTRICITY_KWH, GAS_KWH}
    assert all(value > 0 for value in consumption.values())
    assert receipt["periods"][ELECTRICITY_KWH]["period_type"] == "fiscal_year"
    assert receipt["periods"][GAS_KWH]["period_type"] == "gas_year"


def test_northern_ireland_consumption_reads_one_all_domestic_row_per_fuel():
    consumption, _ = northern_ireland_consumption(2024, rows=_NORTHERN_IRELAND)
    assert consumption == {ELECTRICITY_KWH: 3.0e9, GAS_KWH: 1.0e9}
    # Northern Ireland files electricity under three domestic scopes: only the
    # all-domestic row counts, so a standard-meter row beside it changes nothing.
    standard = _northern_ireland_row(
        ELECTRICITY_KWH,
        2.0e9,
        dimensions={
            "metric": "consumption",
            "fuel": "electricity",
            "domestic_scope": "domestic_standard",
        },
    )
    assert (
        northern_ireland_consumption(2024, rows=[*_NORTHERN_IRELAND, standard])[0]
        == consumption
    )
    with pytest.raises(ValueError, match="expected one vendored row"):
        northern_ireland_consumption(2024, rows=_NORTHERN_IRELAND[1:])
    with pytest.raises(ValueError, match="expected one vendored row"):
        northern_ireland_consumption(
            2024, rows=[*_NORTHERN_IRELAND, _NORTHERN_IRELAND[0]]
        )
    # Electricity stamped as a calendar year is not Northern Ireland's row.
    calendar = _northern_ireland_row(
        ELECTRICITY_KWH, 3.0e9, period={"type": "calendar_year", "value": 2024}
    )
    with pytest.raises(ValueError, match="expected one vendored row"):
        northern_ireland_consumption(2024, rows=[calendar, _NORTHERN_IRELAND[1]])


def test_each_authority_keeps_desnz_s_split_at_the_stage_level():
    registry = TargetRegistry(
        [
            _cell(ELECTRICITY_TARGET, "E06000001", 400.0),
            _cell(ELECTRICITY_TARGET, "W06000001", 100.0),
        ],
        country="uk",
    )
    reference = _reference(
        ELECTRICITY_TARGET,
        UK_ENERGY_LEVEL_INDEX_CONCEPT,
        source_concept=SUBNATIONAL_CONSUMPTION_CONCEPTS[ELECTRICITY_KWH],
    )

    aligned = align_energy_to_stage_level(
        reference, registry, local_rows=_NORTHERN_IRELAND
    )

    factor = gb_energy_level_factors(local_rows=_NORTHERN_IRELAND)[0][ELECTRICITY_KWH]
    assert [spec.value for spec in aligned.specs] == pytest.approx(
        [400.0 * factor, 100.0 * factor]
    )
    spec = aligned.specs[0]
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(factor)
    assert float(spec.metadata["ledger_value_before_alignment"]) == 400.0
    assert spec.metadata["uprating_hold"] == UK_ENERGY_LEVEL_HOLD
    # The factor moves the level, not the period: the 2024 row stays a hold
    # the register declares.
    assert is_uprating_hold(
        replace(spec, metadata={**spec.metadata, "ledger_fact_period": "2024"}), 2025
    )


def test_the_level_applier_refuses_a_reference_without_one_fuel():
    registry = TargetRegistry(
        [_cell(ELECTRICITY_TARGET, "E06000001", 1.0)], country="uk"
    )
    both = _reference(
        ELECTRICITY_TARGET,
        UK_ENERGY_LEVEL_INDEX_CONCEPT,
        source_concept=sorted(SUBNATIONAL_CONSUMPTION_CONCEPTS.values()),
    )
    with pytest.raises(ValueError, match="one sub-national consumption concept"):
        align_energy_to_stage_level(both, registry, local_rows=_NORTHERN_IRELAND)
    other = _reference(ELECTRICITY_TARGET, UK_GAS_METER_SHARE_INDEX_CONCEPT)
    assert align_energy_to_stage_level(other, registry) is registry


def test_the_level_applier_refuses_a_row_from_another_year():
    """A later DESNZ edition in the feed is not levelled on the 2024 total."""

    registry = TargetRegistry(
        [_cell(ELECTRICITY_TARGET, "E06000001", 1.0, fact_period="2025")],
        country="uk",
    )
    reference = _reference(
        ELECTRICITY_TARGET,
        UK_ENERGY_LEVEL_INDEX_CONCEPT,
        source_concept=SUBNATIONAL_CONSUMPTION_CONCEPTS[ELECTRICITY_KWH],
    )
    with pytest.raises(ValueError, match="stage's level is calendar 2024"):
        align_energy_to_stage_level(reference, registry, local_rows=_NORTHERN_IRELAND)


_METERS = [
    _meter_row("gas", "E06000001", 300.0),
    _meter_row("electricity", "E06000001", 400.0),
    # Barnsley under DESNZ's April 2025 code; the census cell keeps the roster's.
    _meter_row("gas", "E08000038", 90.0),
    _meter_row("electricity", "E08000038", 100.0),
    # Orkney has electricity meters and no gas network.
    _meter_row("electricity", "S12000023", 50.0),
]


def test_census_households_take_the_authority_s_gas_meter_share():
    registry = TargetRegistry(
        [
            _cell(GAS_CONNECTED_TARGET, "E06000001", 1000.0),
            _cell(GAS_CONNECTED_TARGET, "E08000016", 500.0),
        ],
        country="uk",
    )
    reference = _reference(GAS_CONNECTED_TARGET, UK_GAS_METER_SHARE_INDEX_CONCEPT)

    aligned = align_census_households_by_gas_meter_share(
        reference, registry, rows=_METERS
    )

    assert [spec.value for spec in aligned.specs] == pytest.approx([750.0, 450.0])
    spec = aligned.specs[1]
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(0.9)
    assert float(spec.metadata["ledger_value_before_alignment"]) == 500.0
    # The census base is held and the household controls set its level.
    assert spec.metadata["uprating_hold"] == UK_GAS_CONNECTED_HOLD
    assert is_uprating_hold(
        replace(spec, metadata={**spec.metadata, "ledger_fact_period": "2021"}), 2025
    )


def _seat_cell(target_id: str, area: str, value: float) -> TargetSpec:
    spec = _cell(target_id, area, value)
    return replace(
        spec, metadata={**spec.metadata, "ledger_geography_level": "constituency"}
    )


def _seat_facts(meters: dict[str, dict[str, float]]) -> UKDesnzConstituencyEnergyFacts:
    rows = tuple(
        {
            "observed_measure": {"source_concept": SUBNATIONAL_METER_CONCEPTS[fuel]},
            "period": {"type": _FEED_PERIOD_TYPES[fuel], "value": 2024},
            "geography": {"id": seat, "level": "constituency"},
            "value": value,
        }
        for seat, by_fuel in meters.items()
        for fuel, value in by_fuel.items()
    )
    return UKDesnzConstituencyEnergyFacts(
        rows=rows,
        inputs_sha256="",
        rows_sha256="",
        parameters={},
        inputs={},
        generator={},
    )


def test_a_seat_and_its_authorities_take_one_level_factor():
    # microcosm#1095: the seat cells split the authorities' totals, so one
    # factor per fuel keeps the two grains equal within every region.
    registry = TargetRegistry(
        [
            _cell(ELECTRICITY_TARGET, "E06000001", 400.0),
            _seat_cell(ELECTRICITY_TARGET, "E14000001", 250.0),
        ],
        country="uk",
    )
    reference = _reference(
        ELECTRICITY_TARGET,
        UK_ENERGY_LEVEL_INDEX_CONCEPT,
        source_concept=SUBNATIONAL_CONSUMPTION_CONCEPTS[ELECTRICITY_KWH],
    )
    aligned = align_energy_to_stage_level(
        reference, registry, local_rows=_NORTHERN_IRELAND
    )
    factor = gb_energy_level_factors(local_rows=_NORTHERN_IRELAND)[0][ELECTRICITY_KWH]
    assert [spec.value for spec in aligned.specs] == pytest.approx(
        [400.0 * factor, 250.0 * factor]
    )


def test_a_seat_takes_the_gas_meter_share_of_its_split_meters():
    registry = TargetRegistry(
        [
            _cell(GAS_CONNECTED_TARGET, "E06000001", 1000.0),
            _seat_cell(GAS_CONNECTED_TARGET, "E14000001", 800.0),
        ],
        country="uk",
    )
    reference = _reference(GAS_CONNECTED_TARGET, UK_GAS_METER_SHARE_INDEX_CONCEPT)
    aligned = align_census_households_by_gas_meter_share(
        reference,
        registry,
        rows=_METERS,
        seat_facts=_seat_facts({"E14000001": {"electricity": 100.0, "gas": 80.0}}),
    )
    authority, seat = aligned.specs
    assert authority.value == pytest.approx(750.0)
    assert seat.value == pytest.approx(640.0)
    assert authority.metadata["uprating_index_resource"] == UK_LOCAL_ENERGY_RESOURCE
    assert seat.metadata["uprating_index_resource"] == (
        "desnz_constituency_energy_facts.json"
    )
    assert seat.metadata["uprating_hold"] == UK_GAS_CONNECTED_HOLD


def test_a_seat_without_gas_meters_is_refused_not_zeroed():
    registry = TargetRegistry(
        [_seat_cell(GAS_CONNECTED_TARGET, "S14000051", 10.0)], country="uk"
    )
    reference = _reference(GAS_CONNECTED_TARGET, UK_GAS_METER_SHARE_INDEX_CONCEPT)
    with pytest.raises(ValueError, match="signed deferral"):
        align_census_households_by_gas_meter_share(
            reference,
            registry,
            rows=_METERS,
            seat_facts=_seat_facts({"S14000051": {"electricity": 100.0}}),
        )


def test_an_authority_without_gas_meters_is_refused_not_zeroed():
    registry = TargetRegistry(
        [_cell(GAS_CONNECTED_TARGET, "S12000023", 10.0)], country="uk"
    )
    reference = _reference(GAS_CONNECTED_TARGET, UK_GAS_METER_SHARE_INDEX_CONCEPT)
    with pytest.raises(ValueError, match="signed deferral"):
        align_census_households_by_gas_meter_share(reference, registry, rows=_METERS)


def test_a_meter_row_on_another_period_type_is_named_not_read_as_missing():
    # A gas row restamped by calendar year would otherwise leave the authority
    # with no gas count and read as one without a gas network.
    rows = [
        _meter_row("gas", "E06000001", 300.0, period_type="calendar_year"),
        _meter_row("electricity", "E06000001", 400.0),
    ]
    registry = TargetRegistry(
        [_cell(GAS_CONNECTED_TARGET, "E06000001", 10.0)], country="uk"
    )
    reference = _reference(GAS_CONNECTED_TARGET, UK_GAS_METER_SHARE_INDEX_CONCEPT)
    with pytest.raises(
        ValueError, match=r"gas meter counts are stamped \['calendar_year'\]"
    ):
        align_census_households_by_gas_meter_share(reference, registry, rows=rows)


def test_the_committed_meter_counts_cover_the_roster_on_their_period_types():
    assert SUBNATIONAL_PERIOD_TYPES == _FEED_PERIOD_TYPES
    meters = _meter_rows_by_roster_area(2024, None)
    assert len(meters) == 350
    assert sum(set(fuels) == {"electricity", "gas"} for fuels in meters.values()) == 347


def test_a_gas_meter_share_above_one_is_refused():
    rows = [
        _meter_row("gas", "E06000001", 401.0),
        _meter_row("electricity", "E06000001", 400.0),
    ]
    registry = TargetRegistry(
        [_cell(GAS_CONNECTED_TARGET, "E06000001", 10.0)], country="uk"
    )
    reference = _reference(GAS_CONNECTED_TARGET, UK_GAS_METER_SHARE_INDEX_CONCEPT)
    with pytest.raises(ValueError, match=r"not in \(0, 1\]"):
        align_census_households_by_gas_meter_share(reference, registry, rows=rows)


def test_both_appliers_are_registered():
    assert (
        UK_UPRATING_APPLIERS[UK_ENERGY_LEVEL_INDEX_CONCEPT]
        is align_energy_to_stage_level
    )
    assert (
        UK_UPRATING_APPLIERS[UK_GAS_METER_SHARE_INDEX_CONCEPT]
        is align_census_households_by_gas_meter_share
    )


def test_a_household_paying_nothing_needs_no_price_region():
    """No spend, no kWh, whatever the region (microcosm#1207 review round 1).

    policyengine-uk's region enum carries UNKNOWN, which no QEP price region
    covers: a household there that pays nothing still reads zero, and one
    that pays is refused rather than priced somewhere arbitrary.
    """

    prices = household_energy_prices()
    kwh = household_energy_kwh(
        [0.0, 900.0], [0.0, 0.0], region=["UNKNOWN", "LONDON"], prices=prices
    )
    assert kwh[ELECTRICITY_KWH][0] == 0.0
    assert kwh[ELECTRICITY_KWH][1] > 0.0
    assert kwh[GAS_KWH].tolist() == [0.0, 0.0]
    with pytest.raises(KeyError, match="UNKNOWN"):
        household_energy_kwh([900.0], [0.0], region=["UNKNOWN"], prices=prices)
