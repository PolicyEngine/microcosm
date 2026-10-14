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
  total and the sub-national statistics cover Great Britain: until Chronicle
  carries Northern Ireland's consumption (PolicyEngine/chronicle#356), Great
  Britain's share of UK households (ONS, Labour Force Survey, in the level's
  year) stands in for its share of the level, a declared interim;
* gas-connected households (``desnz.subnational.gas_meter_share``): each
  authority's census households times its gas meters over its electricity
  meters, the share the stage already imposes by region, here by authority.

The rows keep the fact's own period: consumption is calendar 2024 kWh against
the frame's stored calendar-2024 spend, and the gas-connected base is the
census, whose level the household controls set. Each cell carries
``uprating_hold`` beside its factor, so the holds register declares both.

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
from microcosm.build.uk_runtime.energy_pricing import (
    ELECTRICITY_FUEL,
    ELECTRICITY_KWH,
    FUEL_OF_COLUMN,
    GAS_FUEL,
    GAS_KWH,
    SUBNATIONAL_METER_CONCEPTS,
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
#: ONS's household estimates from the Labour Force Survey: the nations'
#: rows and the UK row of "Families and households in the UK" (Table 5).
LFS_NATION_HOUSEHOLDS_CONCEPT = "ons.households_total"
LFS_UK_HOUSEHOLDS_CONCEPT = "ons.families_households_table5.all_households"
#: The meter year the gas-connected share is read for: the stage's own.
GAS_METER_SHARE_PERIOD = 2024
UK_ENERGY_LEVEL_BASIS = (
    "DESNZ Energy Trends domestic {fuel} consumption at actual temperature, "
    "United Kingdom, calendar {year} (the level the LCFS stage rakes to), times "
    "Great Britain's share of UK households (ONS LFS {year}, the interim until "
    "Chronicle carries Northern Ireland's consumption), over DESNZ's "
    "sub-national Great Britain total for {year}: the authorities keep DESNZ's "
    "split between areas at the stage's level"
)
UK_GAS_METER_SHARE_BASIS = (
    "DESNZ sub-national domestic gas meters over electricity meters in the "
    "authority, {year}, applied to the authority's census households"
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
    electricity, and a household is gas-connected when it pays for gas.
    """

    electricity = np.asarray(electricity_spend, dtype=float)
    gas = np.asarray(gas_spend, dtype=float)
    regions = np.asarray(region).astype(str)
    if not (len(electricity) == len(gas) == len(regions)):
        raise ValueError(
            "household energy spend and region must align: got "
            f"{len(electricity)}, {len(gas)} and {len(regions)} rows."
        )
    return {
        ELECTRICITY_KWH: spend_to_kwh(
            electricity, frs_region=regions, fuel=ELECTRICITY_FUEL, prices=prices
        ),
        GAS_KWH: spend_to_kwh(
            gas,
            frs_region=regions,
            fuel=GAS_FUEL,
            prices=prices,
            connected=gas > 0,
        ),
    }


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


def great_britain_household_share(
    year: int, *, rows: Sequence[Mapping[str, Any]] | None = None
) -> tuple[float, dict[str, Any]]:
    """Great Britain's share of UK households, ONS LFS, calendar ``year``."""

    nations = {
        geography_id: _one_value(
            _local_energy_rows(
                rows,
                concept=LFS_NATION_HOUSEHOLDS_CONCEPT,
                geography_id=geography_id,
                period_value=year,
            ),
            what=f"{UK_LOCAL_ENERGY_RESOURCE}: {geography_id} households {year}",
            unit="count",
        )
        for geography_id in GREAT_BRITAIN_NATION_IDS
    }
    united_kingdom = _one_value(
        _local_energy_rows(
            rows,
            concept=LFS_UK_HOUSEHOLDS_CONCEPT,
            geography_id=UNITED_KINGDOM_GEOGRAPHY_ID,
            period_value=year,
        ),
        what=f"{UK_LOCAL_ENERGY_RESOURCE}: UK households {year}",
        unit="count",
    )
    great_britain = math.fsum(nations.values())
    share = great_britain / united_kingdom
    if not 0.0 < share < 1.0:
        raise ValueError(
            f"Great Britain's share of UK households in {year} is {share!r}, "
            "not in (0, 1): the nations' rows exceed the UK row."
        )
    return share, {
        "resource": UK_LOCAL_ENERGY_RESOURCE,
        "period_value": year,
        "nation_households": dict(sorted(nations.items())),
        "united_kingdom_households": united_kingdom,
        "great_britain_share": share,
        "interim": (
            "Northern Ireland's consumption is not in the feed "
            "(PolicyEngine/chronicle#356): Great Britain's share of UK "
            "households stands in for its share of the Energy Trends level"
        ),
    }


def gb_energy_level_factors(
    parameters: Mapping[str, Any] | None = None,
    *,
    household_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per fuel: the stage's level in Great Britain over DESNZ's GB total."""

    parameters = declared_energy_pricing() if parameters is None else parameters
    if str(parameters.get("level_period_type")) != "calendar_year":
        raise ValueError(
            "The local energy rows bind calendar-year sub-national totals: the "
            "stage's level must be declared for a calendar year."
        )
    year = int(parameters["level_period_value"])
    level, level_receipt = published_energy_level(parameters)
    share, share_receipt = great_britain_household_share(year, rows=household_rows)
    factors: dict[str, float] = {}
    by_fuel: dict[str, dict[str, Any]] = {}
    for column, concept in SUBNATIONAL_CONSUMPTION_CONCEPTS.items():
        fuel = FUEL_OF_COLUMN[column]
        total = _one_value(
            vendored_rows(
                UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
                concept=concept,
                geography_id=GREAT_BRITAIN_GEOGRAPHY_ID,
                period_value=year,
                dimensions={"metric": "consumption", "fuel": fuel},
            ),
            what=f"{UK_DESNZ_DOMESTIC_ENERGY_RESOURCE}: GB {fuel} consumption {year}",
            unit="kwh",
        )
        great_britain_level = float(level[column]) * share
        factors[column] = great_britain_level / total
        by_fuel[column] = {
            "united_kingdom_level_kwh": float(level[column]),
            "great_britain_level_kwh": great_britain_level,
            "subnational_great_britain_kwh": total,
            "factor": factors[column],
        }
    return factors, {
        "period_value": year,
        "level": level_receipt,
        "great_britain_household_share": share_receipt,
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
    household_rows: Sequence[Mapping[str, Any]] | None = None,
) -> TargetRegistry:
    """Put each authority's DESNZ consumption on the stage's Energy Trends level."""

    if reference.uprating_index != UK_ENERGY_LEVEL_INDEX_CONCEPT:
        return registry
    column = _fuel_column(reference)
    factors, receipt = gb_energy_level_factors(
        parameters, household_rows=household_rows
    )
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
        for row in _local_energy_rows(
            rows,
            concept=concept,
            geography_level="local_authority",
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
) -> TargetRegistry:
    """Each authority's census households times its gas-connected meter share."""

    if reference.uprating_index != UK_GAS_METER_SHARE_INDEX_CONCEPT:
        return registry
    meters = _meter_rows_by_roster_area(year, rows)
    basis = UK_GAS_METER_SHARE_BASIS.format(year=year)
    aligned = []
    for spec in registry.specs:
        area = _cell_area(spec)
        counts = meters.get(area, {})
        if set(counts) != {ELECTRICITY_FUEL, GAS_FUEL}:
            raise ValueError(
                f"UK target {spec.name!r}: {UK_LOCAL_ENERGY_RESOURCE} has no "
                f"{year} {sorted({ELECTRICITY_FUEL, GAS_FUEL} - set(counts))} "
                f"meter count for {area!r} (an authority without a gas network "
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
            "uprating_index_resource": UK_LOCAL_ENERGY_RESOURCE,
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
