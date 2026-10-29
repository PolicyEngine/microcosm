"""DESNZ sub-national domestic energy as local-authority targets (microcosm#1207).

DESNZ's sub-national electricity and gas statistics publish each Great
Britain authority's domestic consumption (kWh) and meter counts. Three
local-authority rows bind from them, each through a declared
``uprating_index`` the reference generators and the runtime compiles share:

* consumption (``desnz.energy_trends.gb_domestic_level``): the sub-national
  totals stand on another basis than Energy Trends, the level the LCFS stage
  rakes every household's kWh to (meter readings allocated to the year, and
  weather-corrected for gas). The rows bind as a distribution: one factor per
  fuel puts Great Britain's authorities on the stage's own level, so they
  carry only DESNZ's split between areas. Energy Trends is a United Kingdom
  total and the sub-national statistics cover Great Britain: Great Britain's
  part of the level is its share of DESNZ's sub-national domestic consumption
  with Northern Ireland's row added, so each factor is the UK level over the
  UK sub-national total and both nations stay on DESNZ's sub-national basis
  (microcosm#1278). DESNZ calls Northern Ireland's figures not directly
  comparable with Great Britain's: its electricity runs on financial years
  and its gas on Northern Ireland gas years;
* gas-connected households (``desnz.subnational.gas_meter_share``): each
  authority's census households times its gas meters over its electricity
  meters, the share the stage already imposes by region, here by authority.

The same three rows bind by constituency (microcosm#1095). DESNZ publishes no
constituency cut, so a seat's consumption and meters are Microcosm's split of
its authorities' totals by DESNZ's small-area figures
(``uk/desnz_constituency_energy_facts.json``, ``desnz_constituency_energy``):
the seat cells take the same factor per fuel, and their meter share reads the
split's seat meters.

The rows keep the fact's own period: consumption is DESNZ's 2024 kWh,
calendar 2024 for electricity and gas year 2024 for gas, which the
sub-national lookups pair with calendar 2024
(``energy_pricing.SUBNATIONAL_PERIOD_TYPES``), against the frame's stored
calendar-2024 spend, and the gas-connected base is the census, whose level
the household controls set. Each cell carries ``uprating_hold`` beside its
factor, so the holds register declares both.

The frame side: :func:`household_energy_kwh` turns each household's stored
spend back into kWh at the stage's declared QEP prices (the inverse of the
stage's own kWh-to-spend step), so the local metrics and the drift receipt
read one function.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import LedgerTargetReference, TargetRegistry
from microcosm.build.uk_runtime.desnz_constituency_energy import (
    FACTS_RESOURCE as UK_CONSTITUENCY_ENERGY_RESOURCE,
)
from microcosm.build.uk_runtime.desnz_constituency_energy import (
    UKDesnzConstituencyEnergyFacts,
    load_uk_desnz_constituency_energy_facts,
    seat_meter_counts,
)
from microcosm.build.uk_runtime.energy_pricing import (
    ELECTRICITY_FUEL,
    ELECTRICITY_KWH,
    FUEL_OF_COLUMN,
    GAS_FUEL,
    GAS_KWH,
    SUBNATIONAL_METER_CONCEPTS,
    SUBNATIONAL_PERIOD_TYPES,
    UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
    EnergyPrices,
    pricing_operation,
    published_energy_level,
    qep_prices,
    spend_to_kwh,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import (
    rows_matching,
    vendored_rows,
)

UK_ENERGY_LEVEL_INDEX_CONCEPT = "desnz.energy_trends.gb_domestic_level"
UK_GAS_METER_SHARE_INDEX_CONCEPT = "desnz.subnational.gas_meter_share"
UK_LOCAL_ENERGY_RESOURCE = "desnz_local_energy_facts.json"
UK_ENERGY_STAGE = "lcfs_consumption"
GREAT_BRITAIN_GEOGRAPHY_ID = "K03000001"
UNITED_KINGDOM_GEOGRAPHY_ID = "K02000001"
GREAT_BRITAIN_NATION_IDS = ("E92000001", "W92000004", "S92000003")
SUBNATIONAL_CONSUMPTION_CONCEPTS: Mapping[str, str] = {
    ELECTRICITY_KWH: "desnz.subnational.domestic_electricity_consumption",
    GAS_KWH: "desnz.subnational.domestic_gas_consumption",
}
#: Northern Ireland's row of DESNZ's sub-national domestic consumption, all
#: domestic meters: electricity on financial years from April (the publisher's
#: opening-year label, not weather corrected) and gas on DESNZ's Northern
#: Ireland gas years (15 May to 15 May from 2018, weather corrected). The year
#: labelled Y pairs with Great Britain's year Y, as Great Britain's gas year
#: pairs with its calendar-year electricity (microcosm#1278).
NORTHERN_IRELAND_GEOGRAPHY_ID = "N92000002"
NORTHERN_IRELAND_PERIOD_TYPES: Mapping[str, str] = {
    ELECTRICITY_FUEL: "fiscal_year",
    GAS_FUEL: "gas_year",
}
#: The meter year the gas-connected share is read for: the stage's own.
GAS_METER_SHARE_PERIOD = 2024
UK_ENERGY_LEVEL_BASIS = (
    "DESNZ Energy Trends domestic {fuel} consumption at actual temperature, "
    "United Kingdom, calendar {year} (the level the LCFS stage rakes to), times "
    "Great Britain's share of DESNZ's sub-national domestic {fuel} consumption "
    "with Northern Ireland's row added ({year}, all domestic meters; DESNZ calls "
    "Northern Ireland's figures not directly comparable with Great Britain's), "
    "over DESNZ's sub-national Great Britain total for {year}: the authorities "
    "keep DESNZ's split between areas at the stage's level"
)
UK_GAS_METER_SHARE_BASIS = (
    "DESNZ sub-national domestic gas meters over electricity meters in the "
    "authority, {year}, applied to the authority's census households"
)
UK_SEAT_GAS_METER_SHARE_BASIS = (
    "DESNZ domestic gas meters over electricity meters in the constituency, "
    "{year}, Microcosm's split of its authorities' meters by DESNZ's small-area "
    "figures, applied to the constituency's census households"
)
UK_ENERGY_LEVEL_HOLD = "matched_period"
UK_GAS_CONNECTED_HOLD = "control_rescaled"


def declared_energy_pricing() -> Mapping[str, Any]:
    """The LCFS stage's declared ``price_domestic_energy`` parameters."""

    stage = load_country_spec("uk").sources.stage_map()[UK_ENERGY_STAGE]
    parameters = pricing_operation(stage)
    if parameters is None:
        raise ValueError(f"{UK_ENERGY_STAGE} declares no domestic energy pricing.")
    return parameters


def household_energy_prices(
    parameters: Mapping[str, Any] | None = None,
) -> EnergyPrices:
    """The QEP prices the stage turns kWh into spend with."""

    prices, _ = qep_prices(
        declared_energy_pricing() if parameters is None else parameters
    )
    return prices


def household_energy_kwh(
    electricity_spend: Sequence[float],
    gas_spend: Sequence[float],
    *,
    region: Sequence[str],
    prices: EnergyPrices,
) -> dict[str, np.ndarray]:
    """Each household's electricity and gas kWh from its stored spend.

    The inverse of the stage's kWh-to-spend step: every household pays for
    electricity, and a household is gas-connected when it pays for gas. A
    household that pays nothing for a fuel uses none of it, whatever its
    region, so only paying households are priced.
    """

    electricity = np.asarray(electricity_spend, dtype=float)
    gas = np.asarray(gas_spend, dtype=float)
    regions = np.asarray(region).astype(str)
    if not (len(electricity) == len(gas) == len(regions)):
        raise ValueError(
            "household energy spend and region must align: got "
            f"{len(electricity)}, {len(gas)} and {len(regions)} rows."
        )
    kwh: dict[str, np.ndarray] = {}
    for column, fuel, spend in (
        (ELECTRICITY_KWH, ELECTRICITY_FUEL, electricity),
        (GAS_KWH, GAS_FUEL, gas),
    ):
        paying = spend > 0
        values = np.zeros(len(spend), dtype=float)
        if paying.any():
            values[paying] = spend_to_kwh(
                spend[paying], frs_region=regions[paying], fuel=fuel, prices=prices
            )
        kwh[column] = values
    return kwh


def _one_value(rows: list[dict[str, Any]], *, what: str, unit: str) -> float:
    if len(rows) != 1:
        raise ValueError(f"{what}: expected one vendored row, found {len(rows)}.")
    row = rows[0]
    if str(row.get("unit")) != unit:
        raise ValueError(f"{what} is not in {unit}.")
    value = float(row["value"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be positive, got {row['value']!r}.")
    return value


def _local_energy_rows(
    rows: Sequence[Mapping[str, Any]] | None, **criteria: object
) -> list[dict[str, Any]]:
    """Rows of the local energy resource (or the ones a test passes) matching."""

    if rows is None:
        return vendored_rows(UK_LOCAL_ENERGY_RESOURCE, **criteria)
    return rows_matching({"rows": list(rows)}, **criteria)


def northern_ireland_consumption(
    year: int, *, rows: Sequence[Mapping[str, Any]] | None = None
) -> tuple[dict[str, float], dict[str, Any]]:
    """Northern Ireland's DESNZ sub-national domestic consumption, per fuel.

    The Northern Ireland row (all domestic meters) for the year labelled
    ``year``, in each fuel's own period type. Great Britain's part of the
    Energy Trends UK level is its share of the sub-national total with this
    row added (microcosm#1278; Great Britain's share of UK households stood in
    until Chronicle carried these rows).
    """

    consumption: dict[str, float] = {}
    periods: dict[str, dict[str, Any]] = {}
    for column, concept in SUBNATIONAL_CONSUMPTION_CONCEPTS.items():
        fuel = FUEL_OF_COLUMN[column]
        period_type = NORTHERN_IRELAND_PERIOD_TYPES[fuel]
        consumption[column] = _one_value(
            _local_energy_rows(
                rows,
                concept=concept,
                geography_id=NORTHERN_IRELAND_GEOGRAPHY_ID,
                period_type=period_type,
                period_value=year,
                dimensions={
                    "metric": "consumption",
                    "fuel": fuel,
                    "domestic_scope": "all_domestic",
                },
            ),
            what=(
                f"{UK_LOCAL_ENERGY_RESOURCE}: Northern Ireland {fuel} consumption "
                f"{year}"
            ),
            unit="kwh",
        )
        periods[column] = {"period_type": period_type, "period_value": year}
    return consumption, {
        "resource": UK_LOCAL_ENERGY_RESOURCE,
        "geography_id": NORTHERN_IRELAND_GEOGRAPHY_ID,
        "consumption_kwh": consumption,
        "periods": periods,
        "comparability": (
            "DESNZ calls Northern Ireland's sub-national statistics not directly "
            "comparable with Great Britain's (methodology sections 4.3 and 5.3): "
            "its electricity years run April to March and are not weather "
            "corrected, and its gas meters are classed domestic by profile "
            "marker. The share reads them as published."
        ),
    }


def gb_energy_level_factors(
    parameters: Mapping[str, Any] | None = None,
    *,
    local_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per fuel: the stage's level in Great Britain over DESNZ's GB total.

    Great Britain's part of the UK level is its share of DESNZ's sub-national
    consumption with Northern Ireland's row added, so the factor is the UK
    level over the UK sub-national total. ``local_rows`` stands in for the
    local energy resource's rows (tests).
    """

    parameters = declared_energy_pricing() if parameters is None else parameters
    if str(parameters.get("level_period_type")) != "calendar_year":
        raise ValueError(
            "The local energy rows bind calendar-year sub-national totals: the "
            "stage's level must be declared for a calendar year."
        )
    year = int(parameters["level_period_value"])
    level, level_receipt = published_energy_level(parameters)
    northern_ireland, northern_ireland_receipt = northern_ireland_consumption(
        year, rows=local_rows
    )
    factors: dict[str, float] = {}
    by_fuel: dict[str, dict[str, Any]] = {}
    for column, concept in SUBNATIONAL_CONSUMPTION_CONCEPTS.items():
        fuel = FUEL_OF_COLUMN[column]
        total = _one_value(
            vendored_rows(
                UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
                concept=concept,
                geography_id=GREAT_BRITAIN_GEOGRAPHY_ID,
                period_type=SUBNATIONAL_PERIOD_TYPES[fuel],
                period_value=year,
                dimensions={"metric": "consumption", "fuel": fuel},
            ),
            what=f"{UK_DESNZ_DOMESTIC_ENERGY_RESOURCE}: GB {fuel} consumption {year}",
            unit="kwh",
        )
        share = total / (total + northern_ireland[column])
        great_britain_level = float(level[column]) * share
        factors[column] = great_britain_level / total
        by_fuel[column] = {
            "united_kingdom_level_kwh": float(level[column]),
            "great_britain_share": share,
            "great_britain_level_kwh": great_britain_level,
            "subnational_great_britain_kwh": total,
            "subnational_northern_ireland_kwh": northern_ireland[column],
            "factor": factors[column],
        }
    return factors, {
        "period_value": year,
        "level": level_receipt,
        "northern_ireland": northern_ireland_receipt,
        "by_fuel": by_fuel,
    }


def _cell_area(spec: Any) -> str:
    return str(
        spec.metadata.get("geography_id") or spec.metadata.get("ledger_geography_id")
    )


def _fuel_column(reference: LedgerTargetReference) -> str:
    concepts = reference.ledger_selector.get("source_concept")
    concepts = [concepts] if isinstance(concepts, str) else list(concepts or ())
    columns = {
        column
        for column, concept in SUBNATIONAL_CONSUMPTION_CONCEPTS.items()
        if concept in concepts
    }
    if len(columns) != 1 or len(concepts) != 1:
        raise ValueError(
            f"UK reference {reference.name!r}: {UK_ENERGY_LEVEL_INDEX_CONCEPT} "
            "levels one sub-national consumption concept, not "
            f"{sorted(map(str, concepts))}."
        )
    return columns.pop()


def align_energy_to_stage_level(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    parameters: Mapping[str, Any] | None = None,
    local_rows: Sequence[Mapping[str, Any]] | None = None,
) -> TargetRegistry:
    """Put each authority's DESNZ consumption on the stage's Energy Trends level."""

    if reference.uprating_index != UK_ENERGY_LEVEL_INDEX_CONCEPT:
        return registry
    column = _fuel_column(reference)
    factors, receipt = gb_energy_level_factors(parameters, local_rows=local_rows)
    factor = factors[column]
    basis = UK_ENERGY_LEVEL_BASIS.format(
        fuel=FUEL_OF_COLUMN[column], year=receipt["period_value"]
    )
    aligned = []
    for spec in registry.specs:
        # The factor is the stage's level year over DESNZ's same year: a row
        # resolved to another year (a later edition in the feed) would be
        # levelled on the wrong total.
        fact_period = str(spec.metadata.get("ledger_fact_period") or "")
        if fact_period != str(receipt["period_value"]):
            raise ValueError(
                f"UK target {spec.name!r} resolved DESNZ's {fact_period or '?'} "
                f"consumption, but the stage's level is calendar "
                f"{receipt['period_value']}: pin the rows to the level's year "
                "or move the stage's level."
            )
        metadata = {
            **spec.metadata,
            "uprating_index": UK_ENERGY_LEVEL_INDEX_CONCEPT,
            "uprating_index_basis": basis,
            "uprating_index_resource": UK_LOCAL_ENERGY_RESOURCE,
            "uprating_factor": f"{factor:.15g}",
            "ledger_value_before_alignment": f"{spec.value:.15g}",
            "uprating_hold": UK_ENERGY_LEVEL_HOLD,
        }
        aligned.append(replace(spec, value=spec.value * factor, metadata=metadata))
    return TargetRegistry(aligned, country="uk")


def _meter_rows_by_roster_area(
    year: int, rows: Sequence[Mapping[str, Any]] | None
) -> dict[str, dict[str, float]]:
    """Meter counts per fuel by roster code (DESNZ's recoded areas aliased)."""

    from microcosm.build.uk_runtime.ledger_targets import (
        _local_crosswalk_rosters,
        load_uk_local_area_crosswalk,
    )

    aliases = _local_crosswalk_rosters(load_uk_local_area_crosswalk())[
        "local_authority"
    ]["aliases"]
    meters: dict[str, dict[str, float]] = {}
    for fuel, concept in SUBNATIONAL_METER_CONCEPTS.items():
        expected = SUBNATIONAL_PERIOD_TYPES[fuel]
        # A meter row on another period type would otherwise read as a missing
        # fuel, and the gas-share applier would blame an authority without a
        # gas network (microcosm#1302 review).
        stamped = {
            str((row.get("period") or {}).get("type"))
            for row in _local_energy_rows(
                rows,
                concept=concept,
                geography_level="local_authority",
                period_value=year,
                dimensions={"metric": "meter_count", "fuel": fuel},
            )
        }
        if stamped - {expected}:
            raise ValueError(
                f"{UK_LOCAL_ENERGY_RESOURCE}: {year} {fuel} meter counts are "
                f"stamped {sorted(stamped - {expected})}, where the {fuel} lookups "
                f"read {expected!r} (energy_pricing.SUBNATIONAL_PERIOD_TYPES)."
            )
        for row in _local_energy_rows(
            rows,
            concept=concept,
            geography_level="local_authority",
            period_type=expected,
            period_value=year,
            dimensions={"metric": "meter_count", "fuel": fuel},
        ):
            published = str((row.get("geography") or {}).get("id") or "")
            area = aliases.get(published, (published, ""))[0]
            by_fuel = meters.setdefault(area, {})
            if fuel in by_fuel:
                raise ValueError(
                    f"{UK_LOCAL_ENERGY_RESOURCE}: two {fuel} meter rows for {area!r}."
                )
            if str(row.get("unit")) != "count":
                raise ValueError(
                    f"{UK_LOCAL_ENERGY_RESOURCE}: {area} {fuel} meters are not a count."
                )
            by_fuel[fuel] = float(row["value"])
    return meters


def align_census_households_by_gas_meter_share(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    year: int = GAS_METER_SHARE_PERIOD,
    rows: Sequence[Mapping[str, Any]] | None = None,
    seat_facts: UKDesnzConstituencyEnergyFacts | None = None,
) -> TargetRegistry:
    """Each area's census households times its gas-connected meter share.

    An authority reads its own DESNZ meters (``rows``, by default the vendored
    resource); a constituency reads the split of its authorities' meters
    (``seat_facts``, by default the committed derived resource).
    """

    if reference.uprating_index != UK_GAS_METER_SHARE_INDEX_CONCEPT:
        return registry
    meters: dict[str, dict[str, float]] | None = None
    seat_meters: dict[str, dict[str, float]] | None = None
    aligned = []
    for spec in registry.specs:
        area = _cell_area(spec)
        if str(spec.metadata.get("ledger_geography_level")) == "constituency":
            if seat_meters is None:
                seat_meters = seat_meter_counts(
                    seat_facts or load_uk_desnz_constituency_energy_facts(),
                    year=year,
                    period_types=SUBNATIONAL_PERIOD_TYPES,
                )
            counts = seat_meters.get(area, {})
            resource = UK_CONSTITUENCY_ENERGY_RESOURCE
            basis = UK_SEAT_GAS_METER_SHARE_BASIS.format(year=year)
        else:
            if meters is None:
                meters = _meter_rows_by_roster_area(year, rows)
            counts = meters.get(area, {})
            resource = UK_LOCAL_ENERGY_RESOURCE
            basis = UK_GAS_METER_SHARE_BASIS.format(year=year)
        if set(counts) != {ELECTRICITY_FUEL, GAS_FUEL}:
            raise ValueError(
                f"UK target {spec.name!r}: {resource} has no "
                f"{year} {sorted({ELECTRICITY_FUEL, GAS_FUEL} - set(counts))} "
                f"meter count for {area!r} (an area without a gas network "
                "is a signed deferral, never a zero)."
            )
        if counts[ELECTRICITY_FUEL] <= 0:
            raise ValueError(
                f"UK target {spec.name!r}: {area!r} has no electricity meters."
            )
        share = counts[GAS_FUEL] / counts[ELECTRICITY_FUEL]
        if not 0.0 < share <= 1.0:
            raise ValueError(
                f"UK target {spec.name!r}: the gas-connected share of {area!r} "
                f"is {share!r}, not in (0, 1]."
            )
        metadata = {
            **spec.metadata,
            "uprating_index": UK_GAS_METER_SHARE_INDEX_CONCEPT,
            "uprating_index_basis": basis,
            "uprating_index_resource": resource,
            "uprating_factor": f"{share:.15g}",
            "ledger_value_before_alignment": f"{spec.value:.15g}",
            "uprating_hold": UK_GAS_CONNECTED_HOLD,
        }
        aligned.append(replace(spec, value=spec.value * share, metadata=metadata))
    return TargetRegistry(aligned, country="uk")


def domestic_energy_appliers() -> dict[str, Any]:
    """The appliers this module contributes to ``UK_UPRATING_APPLIERS``."""

    return {
        UK_ENERGY_LEVEL_INDEX_CONCEPT: align_energy_to_stage_level,
        UK_GAS_METER_SHARE_INDEX_CONCEPT: align_census_households_by_gas_meter_share,
    }
