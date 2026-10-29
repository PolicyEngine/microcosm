"""Price imputed domestic energy at published average prices and rake it in kWh
to the NEED shape at a published level (microcosm#890 F, chronicle#270).

The LCFS diary and the consumption QRF carry energy as pounds. Four published
facts turn that into a base-year (calendar 2024) frame under one declared
operation, ``price_domestic_energy``:

* price: the DESNZ Quarterly Energy Prices average variable unit cost
  (GBP/kWh) and fixed cost (GBP/yr) actually paid, by QEP price region and
  payment method, including VAT (tables 2.2.4 and 2.3.4); the FRS region maps
  to a price region through the committed crosswalk. spend -> kWh =
  max(spend - fixed cost, 0) / unit cost for connected households
  (electricity: every household; gas: connected households), and back;
* connection: the published gas-connected share by region, gas meters over
  electricity meters from the DESNZ subnational statistics; gas-positive
  households are disconnected in an identity-keyed uniform order until the
  region's prior-weighted share is the published one, so every group of
  households loses the same expected share of its connected mass; in a
  region whose draw falls short, zero-gas households are connected in the
  declared ``connect_rule`` order (microcosm#1164);
* shape: the NEED mean kWh by household income band, tenure, property type
  and region (England and Wales; Scotland by income, tenure and property),
  raked in kWh with gas over connected households;
* level: DESNZ Energy Trends domestic consumption at actual temperature,
  summed over the four quarters of the declared calendar year (2024, the
  base year of every LCFS spend column; microcosm#1113), one factor per fuel
  on the raked kWh so the frame's prior-weighted total is the published one.
  A fuel the stage declares in ``spend_level_fuels`` takes the kWh ONS
  household spending buys instead (microcosm#1418): the Consumer Trends
  class for the year, less QEP's United Kingdom average fixed cost on every
  domestic meter DESNZ counts, over QEP's United Kingdom average unit cost;
* ceiling: each household's kWh is capped at the declared
  ``household_kwh_ceiling`` (the NEED domestic-meter limits), the capped
  households' own NEED cells are re-raked below it and the level is
  restored (microcosm#1164).

Every price, share, margin, factor and fit is returned in receipts the stage
records; the ``energy_rake`` stage-health gate recomputes the published
values from the vendored rows and refuses a receipt raked to anything else.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.raking import MarginSpec, iterative_proportional_fit
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

PRICE_DOMESTIC_ENERGY_KIND = "price_domestic_energy"
UK_QEP_ENERGY_PRICES_RESOURCE = "qep_energy_prices.json"
UK_DESNZ_DOMESTIC_ENERGY_RESOURCE = "desnz_domestic_energy_facts.json"
UK_NEED_ENERGY_FACTS_RESOURCE = "need_energy_facts.json"
UK_OFGEM_REGION_CROSSWALK_RESOURCE = "ofgem_region_crosswalk.json"
NEED_ELECTRICITY_CONCEPT = "desnz.need.household_electricity_consumption"
NEED_GAS_CONCEPT = "desnz.need.household_gas_consumption"
ENGLAND_AND_WALES_GEOGRAPHY_ID = "K04000001"
SCOTLAND_GEOGRAPHY_ID = "S92000003"
ELECTRICITY_FUEL = "electricity"
GAS_FUEL = "gas"
ELECTRICITY_KWH = "electricity_kwh"
GAS_KWH = "gas_kwh"
FUEL_OF_COLUMN = {ELECTRICITY_KWH: ELECTRICITY_FUEL, GAS_KWH: GAS_FUEL}
GAS_CONNECTED_POSITIVE_SPEND = "positive_gas_spend"
GAS_CONNECTED_PUBLISHED_METER_SHARE = "published_meter_share"
CONNECTION_RULE_METER_RATIO = "gas_meters_over_electricity_meters"
DISCONNECT_LOWEST_DRAWN_GAS_FIRST = "lowest_drawn_gas_first"
DISCONNECT_IDENTITY_UNIFORM_ORDER = "identity_uniform_order"
DISCONNECT_RULES = (
    DISCONNECT_LOWEST_DRAWN_GAS_FIRST,
    DISCONNECT_IDENTITY_UNIFORM_ORDER,
)
GAS_DISCONNECTION_SALT = "lcfs_consumption:gas_disconnection"
#: The declared rule for connecting households in a region the draw leaves
#: below its published share (microcosm#1164): zero-gas households walked in an
#: identity-keyed order weighted by each one's probability of paying for gas.
CONNECT_GAS_PAYER_PROBABILITY_ORDER = "gas_payer_probability_order"
CONNECT_RULES = (CONNECT_GAS_PAYER_PROBABILITY_ORDER,)
QEP_UNIT_COST_CONCEPTS: Mapping[str, str] = {
    ELECTRICITY_FUEL: "desnz.qep.domestic_electricity_variable_unit_cost",
    GAS_FUEL: "desnz.qep.domestic_gas_variable_unit_cost",
}
QEP_FIXED_COST_CONCEPTS: Mapping[str, str] = {
    ELECTRICITY_FUEL: "desnz.qep.domestic_electricity_fixed_cost",
    GAS_FUEL: "desnz.qep.domestic_gas_fixed_cost",
}
ENERGY_TRENDS_CONCEPTS: Mapping[str, str] = {
    ELECTRICITY_KWH: "desnz.energy_trends.domestic_electricity_consumption",
    GAS_KWH: "desnz.energy_trends.domestic_gas_consumption",
}
SUBNATIONAL_METER_CONCEPTS: Mapping[str, str] = {
    ELECTRICITY_FUEL: "desnz.subnational.domestic_electricity_meter_count",
    GAS_FUEL: "desnz.subnational.domestic_gas_meter_count",
}
#: The period type of each fuel's sub-national statistics. DESNZ's gas runs on
#: gas years (gas year Y, mid-May Y to mid-May Y+1, weather-corrected) and its
#: electricity on calendar years. Microcosm pairs gas year Y with calendar year
#: Y, the year each opens in (microcosm#1300; PolicyEngine/chronicle#356 left
#: the alignment to Microcosm), so the meter ratio and the Great Britain level
#: divide like-labelled years, as they did before Chronicle stamped the gas
#: rows' own period type; every read names its type, so a re-stamped period
#: fails by name.
SUBNATIONAL_PERIOD_TYPES: Mapping[str, str] = {
    ELECTRICITY_FUEL: "calendar_year",
    GAS_FUEL: "gas_year",
}
UNITED_KINGDOM_GEOGRAPHY_ID = "K02000001"
GREAT_BRITAIN_GEOGRAPHY_ID = "K03000001"
#: Northern Ireland's row of DESNZ's sub-national domestic statistics, all
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
#: The stage keys declaring which fuels take their level from ONS household
#: spending instead of the Energy Trends volume (microcosm#1418), and the
#: resource that spending is read from.
SPEND_LEVEL_FUELS_KEY = "spend_level_fuels"
SPEND_LEVEL_RESOURCE_KEY = "spend_level_resource"
UK_ONS_HOUSEHOLD_EXPENDITURE_RESOURCE = "ons_household_expenditure_facts.json"
#: Each fuel's ONS Consumer Trends class: household final consumption
#: expenditure, current prices, calendar year, domestic concept.
ONS_HOUSEHOLD_SPEND_CONCEPTS: Mapping[str, str] = {
    ELECTRICITY_KWH: "ons.household_expenditure.electricity",
    GAS_KWH: "ons.household_expenditure.gas",
}
ONS_HOUSEHOLD_SPEND_COICOP: Mapping[str, str] = {
    ELECTRICITY_KWH: "04.5.1",
    GAS_KWH: "04.5.2",
}
ONS_HOUSEHOLD_SPEND_DIMENSIONS: Mapping[str, str] = {
    "consumption_concept": "domestic",
    "frequency": "annual",
    "price_basis": "current_prices",
    "seasonal_adjustment": "not_seasonally_adjusted",
}
#: The basis each fuel's level block names.
LEVEL_BASIS_ENERGY_TRENDS = "energy_trends_volume"
LEVEL_BASIS_ONS_HOUSEHOLD_SPEND = "ons_household_spend"
_RELATIVE_FACT_TOLERANCE = 1e-9

#: FRS categorical values -> NEED dimension ids. Categories absent here are
#: outside the margin (the incumbent's deliberate leave-alone for converted
#: houses, unknown accommodation and Northern Ireland).
NEED_TENURE_IDS: Mapping[str, str] = {
    "OWNED_OUTRIGHT": "owner_occupied",
    "OWNED_WITH_MORTGAGE": "owner_occupied",
    "RENT_PRIVATELY": "privately_rented",
    "RENT_FROM_COUNCIL": "council_or_housing_association",
    "RENT_FROM_HA": "council_or_housing_association",
}
#: England and Wales publishes terraced houses as end- and mid-terrace rows
#: with no combined row; mid-terrace (the larger group) carries the FRS
#: terraced category there, a translation the receipt names. Scotland
#: publishes one terraced row. Mobile homes have no NEED row; the incumbent
#: held them at the all-dwellings mean, kept here as an explicit translation.
NEED_PROPERTY_TYPE_IDS_BY_GEOGRAPHY: Mapping[str, Mapping[str, str]] = {
    ENGLAND_AND_WALES_GEOGRAPHY_ID: {
        "HOUSE_DETACHED": "detached",
        "HOUSE_SEMI_DETACHED": "semi_detached",
        "HOUSE_TERRACED": "mid_terrace",
        "FLAT": "all_flats",
        "MOBILE": "all_dwellings",
    },
    SCOTLAND_GEOGRAPHY_ID: {
        "HOUSE_DETACHED": "detached",
        "HOUSE_SEMI_DETACHED": "semi_detached",
        "HOUSE_TERRACED": "terraced",
        "FLAT": "all_flats",
        "MOBILE": "all_dwellings",
    },
}
NEED_ENGLAND_WALES_REGION_IDS: Mapping[str, str] = {
    "NORTH_EAST": "north_east",
    "NORTH_WEST": "north_west",
    "YORKSHIRE": "yorkshire_and_the_humber",
    "EAST_MIDLANDS": "east_midlands",
    "WEST_MIDLANDS": "west_midlands",
    "EAST_OF_ENGLAND": "east_of_england",
    "LONDON": "london",
    "SOUTH_EAST": "south_east",
    "SOUTH_WEST": "south_west",
    "WALES": "wales",
}
#: Scotland publishes no regional table; its region margin is the Scotland
#: all-dwellings mean (the income table's all_dwellings row).
NEED_SCOTLAND_REGION_ID = "all_dwellings"
NEED_MARGIN_DIMENSIONS = {
    "income": "household_income_band",
    "tenure": "tenure",
    "accommodation": "property_type",
    "region": "region",
}


@dataclass(frozen=True)
class EnergyPrices:
    """Average unit costs (GBP/kWh) and fixed costs (GBP/yr) per price region."""

    period_type: str
    period_value: int
    payment_method: str
    unit_rate: Mapping[tuple[str, str], float]
    fixed_cost: Mapping[tuple[str, str], float]
    region_of_frs_region: Mapping[str, str]
    uk_average_id: str

    def region_for(self, frs_region: str, fuel: str) -> str:
        try:
            region = self.region_of_frs_region[str(frs_region)]
        except KeyError as error:
            raise KeyError(
                f"FRS region {frs_region!r} has no QEP price region in the declared "
                "crosswalk."
            ) from error
        if (region, fuel) not in self.unit_rate:
            # QEP publishes no gas row for Northern Ireland: the UK average
            # prices the fuel there, as the crosswalk declares.
            return self.uk_average_id
        return region


@dataclass(frozen=True)
class NeedMargins:
    """Mean kWh targets per margin, keyed by (geography id, NEED category id)."""

    period_value: int
    targets: Mapping[str, Mapping[tuple[str, str], Mapping[str, float]]]
    income_bands: tuple[tuple[str, float, float], ...]
    receipt: dict[str, Any]


def pricing_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    for operation in stage.operations:
        if operation.kind == PRICE_DOMESTIC_ENERGY_KIND:
            return dict(operation.parameters)
    return None


def load_ofgem_region_crosswalk() -> dict[str, Any]:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath(UK_OFGEM_REGION_CROSSWALK_RESOURCE)
        .read_text(encoding="utf-8")
    )


def fiscal_year_quarters(fiscal_start: str) -> tuple[str, ...]:
    """The four quarterly coverage start dates of the fiscal year starting then."""

    start = date.fromisoformat(fiscal_start)
    if (start.month, start.day) != (4, 1):
        raise ValueError("fiscal_start must be an April 1st.")
    return (
        f"{start.year}-04-01",
        f"{start.year}-07-01",
        f"{start.year}-10-01",
        f"{start.year + 1}-01-01",
    )


def period_quarters(period_type: str, period_value: int) -> tuple[str, ...]:
    """The four quarterly coverage start dates of a calendar or fiscal year."""

    if period_type == "calendar_year":
        return tuple(f"{int(period_value)}-{month:02d}-01" for month in (1, 4, 7, 10))
    if period_type == "fiscal_year":
        return fiscal_year_quarters(f"{int(period_value)}-04-01")
    raise ValueError("period_type must be calendar_year or fiscal_year.")


def _one_row(rows: Sequence[Mapping[str, Any]], *, what: str) -> Mapping[str, Any]:
    if len(rows) != 1:
        raise ValueError(f"{what}: expected one vendored row, found {len(rows)}.")
    return rows[0]


def qep_prices(parameters: Mapping[str, Any]) -> tuple[EnergyPrices, dict[str, Any]]:
    """Resolve the declared QEP average prices per price region and fuel."""

    resource = str(parameters.get("resource") or "")
    if resource != UK_QEP_ENERGY_PRICES_RESOURCE:
        raise ValueError(
            f"{PRICE_DOMESTIC_ENERGY_KIND} must read {UK_QEP_ENERGY_PRICES_RESOURCE!r}, "
            f"not {resource!r}."
        )
    if str(parameters.get("crosswalk_resource")) != UK_OFGEM_REGION_CROSSWALK_RESOURCE:
        raise ValueError(
            f"{PRICE_DOMESTIC_ENERGY_KIND} must use the committed region crosswalk."
        )
    crosswalk = load_ofgem_region_crosswalk()
    mapping = {str(k): str(v) for k, v in crosswalk["qep_price_region_mapping"].items()}
    uk_average = str(crosswalk["qep_uk_average_id"])
    period_type = str(parameters.get("period_type") or "")
    if period_type not in {"fiscal_year", "calendar_year"}:
        raise ValueError("period_type must be fiscal_year or calendar_year.")
    period_value = int(parameters["period_value"])
    payment_method = str(parameters["payment_method"])
    vat_treatment = str(parameters.get("vat_treatment") or "")
    if vat_treatment != "including_vat":
        raise ValueError(
            "vat_treatment must be including_vat: the frame's spend columns and "
            "the bound ONS rows include VAT."
        )
    metering = str(parameters.get("metering_arrangement") or "standard")
    regions = sorted({*mapping.values(), uk_average})
    unit_rate: dict[tuple[str, str], float] = {}
    fixed_cost: dict[tuple[str, str], float] = {}
    by_region: dict[str, dict[str, Any]] = {}
    for region in regions:
        by_region[region] = {}
        for fuel in (ELECTRICITY_FUEL, GAS_FUEL):
            dims = {
                "payment_method": payment_method,
                "vat_treatment": vat_treatment,
                "fuel": fuel,
            }
            if fuel == ELECTRICITY_FUEL:
                dims["metering_arrangement"] = metering
            unit_rows = vendored_rows(
                resource,
                concept=QEP_UNIT_COST_CONCEPTS[fuel],
                period_type=period_type,
                period_value=period_value,
                groupby_value_id=region,
                dimensions={**dims, "price_component": "variable_unit_cost"},
            )
            fixed_rows = vendored_rows(
                resource,
                concept=QEP_FIXED_COST_CONCEPTS[fuel],
                period_type=period_type,
                period_value=period_value,
                groupby_value_id=region,
                dimensions={**dims, "price_component": "fixed_cost"},
            )
            if not unit_rows and not fixed_rows and fuel == GAS_FUEL:
                # No published gas row for this price region (Northern Ireland).
                continue
            unit = _one_row(unit_rows, what=f"{resource}: {region} {fuel} unit cost")
            if str(unit.get("unit")) != "gbp_per_kwh":
                raise ValueError(
                    f"{resource}: {region} {fuel} unit cost is not gbp_per_kwh."
                )
            unit_value = float(unit["value"])
            if fixed_rows:
                fixed = _one_row(
                    fixed_rows, what=f"{resource}: {region} {fuel} fixed cost"
                )
                if str(fixed.get("unit")) != "gbp_per_year":
                    raise ValueError(
                        f"{resource}: {region} {fuel} fixed cost is not gbp_per_year."
                    )
                fixed_value = float(fixed["value"])
                fixed_record = str(fixed.get("source_record_id", ""))
            elif fuel == ELECTRICITY_FUEL and region != uk_average:
                # QEP publishes no fixed cost for Northern Ireland electricity:
                # its standard tariffs carry the whole charge in the unit rate,
                # so the fixed cost is zero there, a translation the receipt names.
                fixed_value = 0.0
                fixed_record = ""
            else:
                raise ValueError(f"{resource}: {region} {fuel} lacks a fixed cost row.")
            if unit_value <= 0 or fixed_value < 0:
                raise ValueError(
                    f"{resource}: {region} {fuel} prices must be positive."
                )
            unit_rate[(region, fuel)] = unit_value
            fixed_cost[(region, fuel)] = fixed_value
            by_region[region][fuel] = {
                "unit_rate_gbp_per_kwh": unit_value,
                "fixed_cost_gbp_per_year": fixed_value,
                "fixed_cost_published": bool(fixed_rows),
                "source_record_ids": [
                    record
                    for record in (str(unit.get("source_record_id", "")), fixed_record)
                    if record
                ],
            }
    for fuel in (ELECTRICITY_FUEL, GAS_FUEL):
        if (uk_average, fuel) not in unit_rate:
            raise ValueError(f"{resource}: no {fuel} prices for {uk_average!r}.")
    prices = EnergyPrices(
        period_type=period_type,
        period_value=period_value,
        payment_method=payment_method,
        unit_rate=unit_rate,
        fixed_cost=fixed_cost,
        region_of_frs_region=mapping,
        uk_average_id=uk_average,
    )
    gas_at_uk_average = sorted(
        frs
        for frs, region in mapping.items()
        if prices.region_for(frs, GAS_FUEL) == uk_average and region != uk_average
    )
    receipt = {
        "operation": PRICE_DOMESTIC_ENERGY_KIND,
        "resource": resource,
        "crosswalk_resource": UK_OFGEM_REGION_CROSSWALK_RESOURCE,
        "period_type": period_type,
        "period_value": period_value,
        "payment_method": payment_method,
        "vat_treatment": vat_treatment,
        "metering_arrangement": metering,
        "region_of_frs_region": dict(mapping),
        "uk_average_id": uk_average,
        "gas_priced_at_uk_average": gas_at_uk_average,
        "prices_by_region": by_region,
    }
    return prices, receipt


def spend_to_kwh(
    spend: np.ndarray,
    *,
    frs_region: Sequence[str],
    fuel: str,
    prices: EnergyPrices,
    connected: np.ndarray | None = None,
) -> np.ndarray:
    """kWh = max(spend - fixed cost, 0) / unit cost for connected rows, else 0."""

    values = np.asarray(spend, dtype=float)
    regions = np.asarray(frs_region).astype(str)
    unit = np.asarray(
        [
            prices.unit_rate[(prices.region_for(region, fuel), fuel)]
            for region in regions
        ]
    )
    fixed = np.asarray(
        [
            prices.fixed_cost[(prices.region_for(region, fuel), fuel)]
            for region in regions
        ]
    )
    mask = (
        np.ones(len(values), dtype=bool)
        if connected is None
        else np.asarray(connected, dtype=bool)
    )
    return np.where(mask, np.maximum(values - fixed, 0.0) / unit, 0.0)


def kwh_to_spend(
    kwh: np.ndarray,
    *,
    frs_region: Sequence[str],
    fuel: str,
    prices: EnergyPrices,
    connected: np.ndarray | None = None,
) -> np.ndarray:
    """spend = kWh x unit cost + fixed cost for connected rows, else 0."""

    values = np.asarray(kwh, dtype=float)
    regions = np.asarray(frs_region).astype(str)
    unit = np.asarray(
        [
            prices.unit_rate[(prices.region_for(region, fuel), fuel)]
            for region in regions
        ]
    )
    fixed = np.asarray(
        [
            prices.fixed_cost[(prices.region_for(region, fuel), fuel)]
            for region in regions
        ]
    )
    mask = (
        np.ones(len(values), dtype=bool)
        if connected is None
        else np.asarray(connected, dtype=bool)
    )
    return np.where(mask, np.maximum(values, 0.0) * unit + fixed, 0.0)


def published_gas_connected_shares(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Gas meters over electricity meters per FRS region from the DESNZ subnational rows.

    Regions without a subnational area in the crosswalk (Northern Ireland)
    carry ``None`` and keep the stage's declared fallback rule.
    """

    resource = str(parameters.get("connection_resource") or "")
    if resource != UK_DESNZ_DOMESTIC_ENERGY_RESOURCE:
        raise ValueError(
            f"connection_resource must be {UK_DESNZ_DOMESTIC_ENERGY_RESOURCE!r}, "
            f"not {resource!r}."
        )
    rule = str(parameters.get("connection_rule") or "")
    if rule != CONNECTION_RULE_METER_RATIO:
        raise ValueError(f"connection_rule must be {CONNECTION_RULE_METER_RATIO!r}.")
    period_value = int(parameters["connection_period_value"])
    fallback = str(parameters.get("connection_fallback") or "")
    if fallback != GAS_CONNECTED_POSITIVE_SPEND:
        raise ValueError(
            f"connection_fallback must be {GAS_CONNECTED_POSITIVE_SPEND!r} (the rule "
            "for regions without a published meter count)."
        )
    declared_fallback = parameters.get("connection_fallback_regions")
    if (
        not isinstance(declared_fallback, list)
        or not declared_fallback
        or not all(isinstance(region, str) and region for region in declared_fallback)
    ):
        raise ValueError(
            "connection_fallback_regions must name every FRS region that has no "
            "published meter count (a non-empty list of region names)."
        )
    fallback_regions = {str(region) for region in declared_fallback}
    crosswalk = load_ofgem_region_crosswalk()
    areas = crosswalk["subnational_area_mapping"]
    undeclared = sorted(str(region) for region, area in areas.items() if area is None)
    if set(undeclared) != fallback_regions:
        raise ValueError(
            "connection_fallback_regions must be exactly the regions the crosswalk "
            f"maps to no subnational area: declared {sorted(fallback_regions)}, "
            f"crosswalk {undeclared}."
        )
    shares: dict[str, float | None] = {}
    by_region: dict[str, dict[str, Any]] = {}
    for frs_region, area in areas.items():
        if area is None:
            shares[str(frs_region)] = None
            by_region[str(frs_region)] = {"area": None, "rule": fallback}
            continue
        counts: dict[str, float] = {}
        record_ids: list[str] = []
        for fuel in (GAS_FUEL, ELECTRICITY_FUEL):
            row = _one_row(
                vendored_rows(
                    resource,
                    concept=SUBNATIONAL_METER_CONCEPTS[fuel],
                    geography_id=str(area),
                    period_type=SUBNATIONAL_PERIOD_TYPES[fuel],
                    period_value=period_value,
                    dimensions={"metric": "meter_count", "fuel": fuel},
                ),
                what=f"{resource}: {area} {fuel} meter count {period_value}",
            )
            if str(row.get("unit")) != "count":
                raise ValueError(f"{resource}: {area} {fuel} meters are not a count.")
            counts[fuel] = float(row["value"])
            record_ids.append(str(row.get("source_record_id", "")))
        if counts[ELECTRICITY_FUEL] <= 0:
            raise ValueError(f"{resource}: {area} has no electricity meters.")
        share = counts[GAS_FUEL] / counts[ELECTRICITY_FUEL]
        if not 0.0 < share <= 1.0:
            raise ValueError(
                f"{resource}: {area} gas-connected share {share} is not in (0, 1]."
            )
        shares[str(frs_region)] = share
        by_region[str(frs_region)] = {
            "area": str(area),
            "gas_meters": counts[GAS_FUEL],
            "electricity_meters": counts[ELECTRICITY_FUEL],
            "share": share,
            "source_record_ids": record_ids,
        }
    receipt = {
        "rule": GAS_CONNECTED_PUBLISHED_METER_SHARE,
        "connection_rule": rule,
        "resource": resource,
        "period_value": period_value,
        "period_types": dict(SUBNATIONAL_PERIOD_TYPES),
        "fallback": fallback,
        "fallback_regions": sorted(fallback_regions),
        "by_region": by_region,
    }
    return shares, receipt


def impose_gas_connection(
    gas_kwh: Sequence[float],
    *,
    frs_region: Sequence[str],
    weights: Sequence[float],
    shares: Mapping[str, float | None],
    disconnect_rule: str = DISCONNECT_LOWEST_DRAWN_GAS_FIRST,
    identity: Sequence[object] | None = None,
    seed: int = 0,
    connect_probability: Sequence[float] | None = None,
    connect_salt: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The gas-connected mask after imposing each region's published share.

    A greedy weight-fitting walk, not a fitted threshold. Within a region
    whose prior-weighted share of gas-positive households exceeds the
    published share, gas-positive households are walked in the declared
    order. ``lowest_drawn_gas_first`` walks ascending drawn gas kWh, which
    concentrates the disconnections on whichever households draw the least
    gas. ``identity_uniform_order`` walks ascending identity-keyed uniforms
    (``stable_identity_uniforms(identity, seed, GAS_DISCONNECTION_SALT)``),
    so each gas-positive household is equally likely to go and every group
    of households (dwelling type, income band, support channel) loses the
    same expected share of its connected mass; the drawn connection pattern
    keeps its composition and only its level moves to the published share.
    Along the walk a household is disconnected when its weight fits inside
    the remaining excess, skipped (it stays connected) when it would
    overshoot, and after the walk the skipped household nearest the remainder
    is disconnected if that brings the share nearer the published one. The
    achieved share therefore sits within a fraction of one household weight
    of the published share; ``rows_skipped_for_weight`` counts the households
    the walk passed over. Regions with no published share (``None``, the
    stage's declared ``connection_fallback_regions``) keep the positive-gas
    rule.

    A region below its published share keeps every gas-positive household.
    Without ``connect_probability`` the draw cannot create connections and the
    shortfall is receipted. With it (each row's probability of paying for gas,
    the declared ``connect_rule``; microcosm#1164), the region's zero-gas
    households are walked in a weighted identity-keyed order, ascending
    ``-log(u) / p`` for ``u`` from ``stable_identity_uniforms(identity, seed,
    connect_salt)``, which draws them without replacement in proportion to
    ``p``, and connected by the same weight-fitting walk until the share is
    met; the caller gives each connected household its gas. A shortfall then
    remains only if the region runs out of candidates (``candidates_exhausted``).
    """

    if disconnect_rule not in DISCONNECT_RULES:
        raise ValueError(f"disconnect_rule must be one of {DISCONNECT_RULES!r}.")
    gas = np.asarray(gas_kwh, dtype=float)
    regions = np.asarray(frs_region).astype(str)
    weight = np.asarray(weights, dtype=float)
    if not (len(gas) == len(regions) == len(weight)):
        raise ValueError("gas_kwh, frs_region and weights must align.")
    if disconnect_rule == DISCONNECT_IDENTITY_UNIFORM_ORDER:
        if identity is None or len(identity) != len(gas):
            raise ValueError(
                f"{DISCONNECT_IDENTITY_UNIFORM_ORDER!r} needs one identity per row."
            )
        walk_key = stable_identity_uniforms(
            np.asarray(identity), seed=seed, salt=GAS_DISCONNECTION_SALT
        )
    else:
        walk_key = gas
    connect = connect_probability is not None
    if connect:
        probability = np.asarray(connect_probability, dtype=float)
        if (
            len(probability) != len(gas)
            or identity is None
            or len(identity) != len(gas)
        ):
            raise ValueError(
                "connect_probability needs one probability and one identity per row."
            )
        if not connect_salt:
            raise ValueError("connecting households needs a declared connect_salt.")
        connect_key = np.full(len(gas), np.inf)
        weighted = probability > 0
        uniforms = stable_identity_uniforms(
            np.asarray(identity), seed=seed, salt=connect_salt
        )
        connect_key[weighted] = (
            -np.log(np.maximum(uniforms[weighted], 1e-300)) / (probability[weighted])
        )
    connected = gas > 0
    by_region: dict[str, dict[str, Any]] = {}
    for region in sorted(set(regions)):
        rows = np.flatnonzero(regions == region)
        total = float(weight[rows].sum())
        positive = rows[connected[rows]]
        before = float(weight[positive].sum()) / total if total > 0 else 0.0
        target = shares.get(region)
        entry: dict[str, Any] = {
            "rows": int(len(rows)),
            "share_before": before,
            "target_share": target,
        }
        if target is None:
            entry.update({"rule": GAS_CONNECTED_POSITIVE_SPEND, "share_after": before})
            by_region[region] = entry
            continue
        if before <= target and connect:
            entry.update(
                _connect_walk(
                    connected,
                    rows=rows,
                    weight=weight,
                    key=connect_key,
                    total=total,
                    before=before,
                    target=target,
                )
            )
            by_region[region] = entry
            continue
        if before <= target or len(positive) == 0:
            entry.update(
                {
                    "rule": GAS_CONNECTED_PUBLISHED_METER_SHARE,
                    "share_after": before,
                    "rows_disconnected": 0,
                    "weight_disconnected": 0.0,
                    "shortfall": max(target - before, 0.0),
                }
            )
            by_region[region] = entry
            continue
        order = positive[np.argsort(walk_key[positive], kind="stable")]
        excess = (before - target) * total
        remaining = excess
        drop: list[int] = []
        skipped: list[int] = []
        for index in order:
            if remaining <= 0.0:
                break
            if weight[index] <= remaining:
                drop.append(int(index))
                remaining -= weight[index]
            else:
                skipped.append(int(index))
        if remaining > 0.0 and skipped:
            nearest = min(skipped, key=lambda i: abs(weight[i] - remaining))
            if abs(weight[nearest] - remaining) < remaining:
                drop.append(nearest)
                remaining -= weight[nearest]
        drop_index = np.asarray(drop, dtype=int)
        dropped = set(drop)
        connected[drop_index] = False
        after = float(weight[rows[connected[rows]]].sum()) / total
        entry.update(
            {
                "rule": GAS_CONNECTED_PUBLISHED_METER_SHARE,
                "share_after": after,
                "rows_disconnected": int(len(drop_index)),
                "rows_skipped_for_weight": int(
                    sum(1 for i in skipped if i not in dropped)
                ),
                "weight_disconnected": float(weight[drop_index].sum()),
                "shortfall": 0.0,
            }
        )
        by_region[region] = entry
    total_weight = float(weight.sum())
    receipt = {
        "disconnect_rule": disconnect_rule,
        **(
            {"seed": int(seed), "salt": GAS_DISCONNECTION_SALT}
            if disconnect_rule == DISCONNECT_IDENTITY_UNIFORM_ORDER
            else {}
        ),
        "share_before": float(weight[gas > 0].sum()) / total_weight
        if total_weight > 0
        else 0.0,
        "share_after": float(weight[connected].sum()) / total_weight
        if total_weight > 0
        else 0.0,
        "rows_disconnected": int(((gas > 0) & ~connected).sum()),
        **(
            {
                "connect_rule": CONNECT_GAS_PAYER_PROBABILITY_ORDER,
                "connect_salt": connect_salt,
                "rows_connected": int((connected & ~(gas > 0)).sum()),
            }
            if connect
            else {}
        ),
        "by_region": by_region,
    }
    return connected, receipt


def _connect_walk(
    connected: np.ndarray,
    *,
    rows: np.ndarray,
    weight: np.ndarray,
    key: np.ndarray,
    total: float,
    before: float,
    target: float,
) -> dict[str, Any]:
    """Connect a region's zero-gas rows in ``key`` order until its share is met.

    The disconnection walk's weight-fitting rule, mirrored: a household is
    connected when its weight fits inside the remaining shortfall, skipped
    when it would overshoot, and afterwards the skipped household nearest the
    remainder is connected if that brings the share nearer the published one.
    Rows of probability zero (an infinite key) are never candidates.
    """

    candidates = rows[~connected[rows] & np.isfinite(key[rows])]
    order = candidates[np.argsort(key[candidates], kind="stable")]
    remaining = (target - before) * total
    add: list[int] = []
    skipped: list[int] = []
    for index in order:
        if remaining <= 0.0:
            break
        if weight[index] <= remaining:
            add.append(int(index))
            remaining -= weight[index]
        else:
            skipped.append(int(index))
    if remaining > 0.0 and skipped:
        nearest = min(skipped, key=lambda i: abs(weight[i] - remaining))
        if abs(weight[nearest] - remaining) < remaining:
            add.append(nearest)
            remaining -= weight[nearest]
    added = np.asarray(add, dtype=int)
    connected[added] = True
    after = float(weight[rows[connected[rows]]].sum()) / total
    exhausted = remaining > 0.0 and not skipped
    return {
        "rule": GAS_CONNECTED_PUBLISHED_METER_SHARE,
        "share_after": after,
        "rows_disconnected": 0,
        "weight_disconnected": 0.0,
        "rows_connected": int(len(added)),
        "weight_connected": float(weight[added].sum()),
        "rows_skipped_for_weight": int(sum(1 for i in skipped if i not in set(add))),
        "candidates_exhausted": bool(exhausted),
        "shortfall": max(target - after, 0.0) if exhausted else 0.0,
    }


def published_energy_level(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Domestic consumption per fuel (kWh) over the declared year.

    The four Energy Trends quarters of the declared calendar or fiscal year
    are summed; the stage declares calendar 2024, the base year of every LCFS
    spend column (microcosm#1113). A fuel in ``spend_level_fuels`` takes the
    kWh ONS household spending buys instead (:func:`household_spend_level`,
    microcosm#1418); its block keeps the Energy Trends total beside it.
    """

    resource = str(parameters.get("level_resource") or "")
    if resource != UK_DESNZ_DOMESTIC_ENERGY_RESOURCE:
        raise ValueError(
            f"level_resource must be {UK_DESNZ_DOMESTIC_ENERGY_RESOURCE!r}, "
            f"not {resource!r}."
        )
    period_type = str(parameters.get("level_period_type") or "")
    period_value = int(parameters["level_period_value"])
    quarters = period_quarters(period_type, period_value)
    temperature = str(parameters.get("level_temperature_adjustment") or "")
    if temperature != "actual_temperature":
        raise ValueError(
            "level_temperature_adjustment must be actual_temperature (the volume "
            "households consumed, not a weather-corrected one)."
        )
    level: dict[str, float] = {}
    by_fuel: dict[str, dict[str, Any]] = {}
    geographies: set[str] = set()
    for column, concept in ENERGY_TRENDS_CONCEPTS.items():
        rows = vendored_rows(
            resource,
            concept=concept,
            period_type="quarter",
            dimensions={
                "temperature_adjustment": temperature,
                "frequency": "quarterly",
            },
        )
        by_quarter: dict[str, float] = {}
        record_ids: list[str] = []
        for row in rows:
            start = str((row.get("period_coverage") or {}).get("start_date"))
            if start not in quarters:
                continue
            if start in by_quarter:
                raise ValueError(f"{resource}: duplicate {concept} row for {start}.")
            if str(row.get("unit")) != "kwh":
                raise ValueError(f"{resource}: {concept} is not in kwh.")
            by_quarter[start] = float(row["value"])
            record_ids.append(str(row.get("source_record_id", "")))
            geographies.add(str((row.get("geography") or {}).get("id")))
        missing = [q for q in quarters if q not in by_quarter]
        if missing:
            raise ValueError(
                f"{resource}: {concept} lacks quarters starting {missing}."
            )
        if any(value <= 0 for value in by_quarter.values()):
            raise ValueError(f"{resource}: {concept} quarters must be positive.")
        level[column] = float(sum(by_quarter.values()))
        by_fuel[column] = {
            "basis": LEVEL_BASIS_ENERGY_TRENDS,
            "concept": concept,
            "quarters": by_quarter,
            "total_kwh": level[column],
            "source_record_ids": record_ids,
        }
    spend_fuels = declared_spend_level_fuels(parameters)
    if spend_fuels:
        spend_level, spend_blocks = household_spend_level(parameters, fuels=spend_fuels)
        for column, value in spend_level.items():
            by_fuel[column] = {
                "basis": LEVEL_BASIS_ONS_HOUSEHOLD_SPEND,
                **spend_blocks[column],
                "total_kwh": value,
                "energy_trends_kwh": level[column],
                "energy_trends": by_fuel[column],
            }
            level[column] = value
    receipt = {
        "resource": resource,
        "period_type": period_type,
        "period_value": period_value,
        "quarters": list(quarters),
        "temperature_adjustment": temperature,
        "geography_ids": sorted(geographies),
        "spend_level_fuels": list(spend_fuels),
        "by_fuel": by_fuel,
    }
    return level, receipt


def declared_spend_level_fuels(parameters: Mapping[str, Any]) -> tuple[str, ...]:
    """The fuels the stage levels to ONS household spending (none if undeclared)."""

    declared = parameters.get(SPEND_LEVEL_FUELS_KEY)
    if declared is None:
        return ()
    if not isinstance(declared, list) or not declared:
        raise ValueError(
            f"{SPEND_LEVEL_FUELS_KEY} must be a non-empty list of fuels when declared."
        )
    fuels = tuple(str(fuel) for fuel in declared)
    unknown = sorted(set(fuels) - {ELECTRICITY_FUEL, GAS_FUEL})
    if unknown or len(set(fuels)) != len(fuels):
        raise ValueError(
            f"{SPEND_LEVEL_FUELS_KEY} must name {ELECTRICITY_FUEL!r} and {GAS_FUEL!r} "
            f"at most once each, not {list(fuels)}."
        )
    resource = str(parameters.get(SPEND_LEVEL_RESOURCE_KEY) or "")
    if resource != UK_ONS_HOUSEHOLD_EXPENDITURE_RESOURCE:
        raise ValueError(
            f"{SPEND_LEVEL_RESOURCE_KEY} must be "
            f"{UK_ONS_HOUSEHOLD_EXPENDITURE_RESOURCE!r}, not {resource!r}."
        )
    return fuels


def uk_domestic_meters(fuel: str, year: int) -> tuple[float, dict[str, Any]]:
    """Every UK domestic meter of a fuel: DESNZ's Great Britain and Northern Ireland rows.

    Great Britain's sub-national count for the year in the fuel's own period
    type (``SUBNATIONAL_PERIOD_TYPES``), with Northern Ireland's row of the
    year labelled the same (``NORTHERN_IRELAND_PERIOD_TYPES``), as the local
    energy rows pair them.
    """

    resource = UK_DESNZ_DOMESTIC_ENERGY_RESOURCE
    counts: dict[str, float] = {}
    record_ids: list[str] = []
    for nation, geography, period_type in (
        ("great_britain", GREAT_BRITAIN_GEOGRAPHY_ID, SUBNATIONAL_PERIOD_TYPES[fuel]),
        (
            "northern_ireland",
            NORTHERN_IRELAND_GEOGRAPHY_ID,
            NORTHERN_IRELAND_PERIOD_TYPES[fuel],
        ),
    ):
        row = _one_row(
            vendored_rows(
                resource,
                concept=SUBNATIONAL_METER_CONCEPTS[fuel],
                geography_id=geography,
                period_type=period_type,
                period_value=year,
                dimensions={
                    "metric": "meter_count",
                    "fuel": fuel,
                    "domestic_scope": "all_domestic",
                },
            ),
            what=f"{resource}: {geography} {fuel} domestic meters {year}",
        )
        if str(row.get("unit")) != "count":
            raise ValueError(f"{resource}: {geography} {fuel} meters are not a count.")
        value = float(row["value"])
        if value <= 0:
            raise ValueError(f"{resource}: {geography} has no {fuel} meters.")
        counts[nation] = value
        record_ids.append(str(row.get("source_record_id", "")))
    total = float(sum(counts.values()))
    return total, {
        "resource": resource,
        "meters": total,
        "by_nation": counts,
        "source_record_ids": record_ids,
    }


def household_spend_level(
    parameters: Mapping[str, Any], *, fuels: Sequence[str]
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """The kWh each fuel's ONS household spending buys at QEP prices (microcosm#1418).

    ONS Consumer Trends household final consumption expenditure on the fuel's
    class (04.5.1 electricity, 04.5.2 gas), current prices, for the stage's
    declared calendar year, less QEP's United Kingdom average fixed cost on
    every UK domestic meter (:func:`uk_domestic_meters`), over QEP's United
    Kingdom average unit cost, both at the prices the stage declares. The
    meters make the kWh cover every domestic meter, as Energy Trends' domestic
    consumption does; the frame's households pay fewer standing charges than
    there are meters, so its prior-weighted spend sits that much below ONS.

    Northern Ireland's electricity meters stay in the subtraction although
    QEP gives them no fixed cost (Anthony's review of microcosm#1420). Its
    tariffs carry the standing charge in the unit rate: 31.16p a kWh in 2024,
    against the United Kingdom's 25.83p and £205 a year. At Great Britain's
    mean of about 3,300 kWh a meter (DESNZ sub-national, 2024), that premium
    recovers about £177 a meter, so Northern Ireland's spending holds about
    the same fixed charge as the rest of the UK's. Leaving its meters out
    would count about £150m of that charge as 0.6 TWh of electricity.
    """

    period_type = str(parameters.get("level_period_type") or "")
    if period_type != "calendar_year":
        raise ValueError(
            "ONS household spending is published for calendar years: a level "
            f"from it needs level_period_type calendar_year, not {period_type!r}."
        )
    year = int(parameters["level_period_value"])
    prices, _ = qep_prices(parameters)
    if (prices.period_type, prices.period_value) != (period_type, year):
        raise ValueError(
            "The QEP prices that turn ONS spending into kWh must be for the "
            f"level's year ({period_type} {year}), not "
            f"{prices.period_type} {prices.period_value}."
        )
    resource = str(parameters[SPEND_LEVEL_RESOURCE_KEY])
    uk = prices.uk_average_id
    level: dict[str, float] = {}
    blocks: dict[str, dict[str, Any]] = {}
    for column, fuel in FUEL_OF_COLUMN.items():
        if fuel not in fuels:
            continue
        row = _one_row(
            vendored_rows(
                resource,
                concept=ONS_HOUSEHOLD_SPEND_CONCEPTS[column],
                geography_id=UNITED_KINGDOM_GEOGRAPHY_ID,
                period_type="calendar_year",
                period_value=year,
                dimensions={
                    **ONS_HOUSEHOLD_SPEND_DIMENSIONS,
                    "coicop": ONS_HOUSEHOLD_SPEND_COICOP[column],
                },
            ),
            what=f"{resource}: {fuel} household spending {year}",
        )
        if str(row.get("unit")) != "gbp":
            raise ValueError(f"{resource}: {fuel} household spending is not in gbp.")
        spend = float(row["value"])
        meters, meter_block = uk_domestic_meters(fuel, year)
        unit_rate = float(prices.unit_rate[(uk, fuel)])
        fixed_cost = float(prices.fixed_cost[(uk, fuel)])
        standing = meters * fixed_cost
        variable = spend - standing
        if variable <= 0:
            raise ValueError(
                f"{resource}: {fuel} household spending {spend} does not cover the "
                f"standing charges of {meters} meters at {fixed_cost} a year."
            )
        level[column] = variable / unit_rate
        blocks[column] = {
            "spend_resource": resource,
            "spend_concept": ONS_HOUSEHOLD_SPEND_CONCEPTS[column],
            "coicop": ONS_HOUSEHOLD_SPEND_COICOP[column],
            "spend_gbp": spend,
            "spend_source_record_id": str(row.get("source_record_id", "")),
            "price_region": uk,
            "unit_rate_gbp_per_kwh": unit_rate,
            "fixed_cost_gbp_per_year": fixed_cost,
            "domestic_meters": meter_block,
            "standing_charges_gbp": standing,
            "variable_spend_gbp": variable,
        }
        if fuel == ELECTRICITY_FUEL:
            blocks[column]["northern_ireland_meters"] = (
                "in the subtraction: Northern Ireland's tariffs carry the standing "
                "charge in the unit rate"
            )
    return level, blocks


def need_geography(frs_region: Sequence[str]) -> np.ndarray:
    """The NEED geography each household's margins come from ('' for none)."""

    regions = np.asarray(frs_region).astype(str)
    return np.where(
        regions == "SCOTLAND",
        SCOTLAND_GEOGRAPHY_ID,
        np.where(regions == "NORTHERN_IRELAND", "", ENGLAND_AND_WALES_GEOGRAPHY_ID),
    ).astype(object)


def _income_band_edges(band_ids: Sequence[str]) -> tuple[tuple[str, float, float], ...]:
    """Parse NEED income band ids into [lower, upper) GBP edges."""

    parsed: list[tuple[str, float, float]] = []
    for band in band_ids:
        if band == "all_dwellings":
            continue
        if match := re.fullmatch(r"less_than_gbp(\d[\d_]*)", band):
            parsed.append((band, 0.0, float(match.group(1).replace("_", ""))))
        elif match := re.fullmatch(r"gbp(\d[\d_]*)_gbp(\d[\d_]*)", band):
            lower = float(match.group(1).replace("_", ""))
            upper = float(match.group(2).replace("_", "")) + 1.0
            parsed.append((band, lower, upper))
        elif match := re.fullmatch(r"gbp(\d[\d_]*)_or_more", band):
            parsed.append((band, float(match.group(1).replace("_", "")), np.inf))
        else:
            raise ValueError(f"unrecognised NEED income band id {band!r}.")
    parsed.sort(key=lambda item: item[1])
    for (_, _, upper), (_, lower, _) in zip(parsed, parsed[1:], strict=False):
        if upper != lower:
            raise ValueError("NEED income bands are not contiguous.")
    if not parsed or parsed[0][1] != 0.0 or parsed[-1][2] != np.inf:
        raise ValueError("NEED income bands must cover [0, inf).")
    return tuple(parsed)


def need_margins_from_facts(
    resource: str = UK_NEED_ENERGY_FACTS_RESOURCE,
    *,
    period_value: int,
    statistic: str = "mean",
) -> NeedMargins:
    """Mean kWh per margin category from the vendored NEED rows of one consumption year."""

    if resource != UK_NEED_ENERGY_FACTS_RESOURCE:
        raise ValueError(
            f"NEED margins must come from {UK_NEED_ENERGY_FACTS_RESOURCE!r}."
        )
    period_value = int(period_value)
    targets: dict[str, dict[tuple[str, str], dict[str, float]]] = {
        margin: {} for margin in NEED_MARGIN_DIMENSIONS
    }
    record_ids: list[str] = []
    for geography in (ENGLAND_AND_WALES_GEOGRAPHY_ID, SCOTLAND_GEOGRAPHY_ID):
        for margin, dimension in NEED_MARGIN_DIMENSIONS.items():
            for column, concept in (
                (ELECTRICITY_KWH, NEED_ELECTRICITY_CONCEPT),
                (GAS_KWH, NEED_GAS_CONCEPT),
            ):
                rows = [
                    row
                    for row in vendored_rows(
                        resource,
                        concept=concept,
                        geography_id=geography,
                        period_type="calendar_year",
                        period_value=period_value,
                        dimensions={"statistic": statistic},
                    )
                    if dimension in (row.get("dimensions") or {})
                ]
                for row in rows:
                    if str(row.get("unit")) != "kwh":
                        raise ValueError(f"{resource}: expected kwh rows.")
                    category = str(row["dimensions"][dimension])
                    cell = targets[margin].setdefault((geography, category), {})
                    if column in cell:
                        raise ValueError(
                            f"{resource}: duplicate {margin} row {geography}/{category}."
                        )
                    cell[column] = float(row["value"])
                    record_ids.append(str(row.get("source_record_id", "")))
    scotland_region = (SCOTLAND_GEOGRAPHY_ID, NEED_SCOTLAND_REGION_ID)
    if scotland_region not in targets["region"]:
        # Scotland publishes no regional table: its region margin is the
        # Scotland all-dwellings mean carried by the income table.
        all_dwellings = targets["income"].get((SCOTLAND_GEOGRAPHY_ID, "all_dwellings"))
        if all_dwellings is None:
            raise ValueError(
                f"{resource}: Scotland lacks an all_dwellings row for {period_value}."
            )
        targets["region"][scotland_region] = dict(all_dwellings)
    for margin, cells in targets.items():
        for key, cell in cells.items():
            if set(cell) != {ELECTRICITY_KWH, GAS_KWH}:
                raise ValueError(f"{resource}: {margin} cell {key} lacks a fuel.")
    if not targets["region"] or all(
        g != SCOTLAND_GEOGRAPHY_ID for g, _ in targets["income"]
    ):
        raise ValueError(f"{resource}: missing a NEED geography for {period_value}.")
    bands = _income_band_edges(sorted({band for geography, band in targets["income"]}))
    receipt = {
        "resource": resource,
        "period_value": period_value,
        "statistic": statistic,
        "geographies": [ENGLAND_AND_WALES_GEOGRAPHY_ID, SCOTLAND_GEOGRAPHY_ID],
        "translations": {
            "tenure": dict(NEED_TENURE_IDS),
            "property_type": {
                geo: dict(mapping)
                for geo, mapping in NEED_PROPERTY_TYPE_IDS_BY_GEOGRAPHY.items()
            },
            "region": {
                ENGLAND_AND_WALES_GEOGRAPHY_ID: dict(NEED_ENGLAND_WALES_REGION_IDS),
                SCOTLAND_GEOGRAPHY_ID: {"SCOTLAND": NEED_SCOTLAND_REGION_ID},
            },
        },
        "income_bands": [
            {"id": band, "lower": lower, "upper": None if upper == np.inf else upper}
            for band, lower, upper in bands
        ],
        "cells": {
            margin: {
                f"{geography}:{category}": dict(cell)
                for (geography, category), cell in sorted(cells.items())
            }
            for margin, cells in targets.items()
        },
        "source_record_ids": record_ids,
    }
    return NeedMargins(
        period_value=period_value, targets=targets, income_bands=bands, receipt=receipt
    )


def income_band_ids(income: Sequence[float], margins: NeedMargins) -> np.ndarray:
    values = pd.to_numeric(pd.Series(income), errors="coerce").fillna(0.0).to_numpy()
    result = np.full(len(values), "", dtype=object)
    for band, lower, upper in margins.income_bands:
        result[(values >= lower) & (values < upper)] = band
    return result


#: The declared household ceilings (microcosm#1164, #1207): kWh a year above
#: which DESNZ and NEED do not count a meter as domestic (NEED Annex B, Quality
#: Assurance: gas meters under 73,200 kWh a year, electricity meters under
#: 50,000 kWh on profile classes 1 and 2). The DESNZ total the stage levels to
#: counts domestic meters, so no household of the frame is drawn above them.
HOUSEHOLD_KWH_CEILING_KEY = "household_kwh_ceiling"


def declared_household_kwh_ceilings(
    parameters: Mapping[str, Any],
) -> dict[str, float] | None:
    """The declared ``household_kwh_ceiling`` per fuel column, if any."""

    declared = parameters.get(HOUSEHOLD_KWH_CEILING_KEY)
    if declared is None:
        return None
    if not isinstance(declared, Mapping) or set(declared) != {"electricity", "gas"}:
        raise ValueError(
            f"{HOUSEHOLD_KWH_CEILING_KEY} must give electricity and gas kWh."
        )
    ceilings = {
        ELECTRICITY_KWH: float(declared["electricity"]),
        GAS_KWH: float(declared["gas"]),
    }
    if not all(np.isfinite(v) and v > 0 for v in ceilings.values()):
        raise ValueError(f"{HOUSEHOLD_KWH_CEILING_KEY} values must be positive.")
    return ceilings


#: The declared NEED cells whose households start from the cell's NEED mean
#: when every one of them drew no kWh (microcosm#1240). Mobile homes have no
#: NEED property-type row and the incumbent held them at the all-dwellings
#: mean; where every mobile home of a NEED geography drew a spend at or below
#: the standing charge, the multiplying rake has nothing to scale and leaves
#: the cell at zero. Any other cell with a zero current mean still fails the
#: ``energy_rake`` gate.
ZERO_KWH_CELLS_AT_NEED_MEAN_KEY = "zero_kwh_cells_at_need_mean"


def declared_zero_kwh_cells_at_need_mean(
    parameters: Mapping[str, Any],
) -> dict[str, tuple[str, ...]] | None:
    """The declared ``zero_kwh_cells_at_need_mean`` categories per margin, if any."""

    declared = parameters.get(ZERO_KWH_CELLS_AT_NEED_MEAN_KEY)
    if declared is None:
        return None
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError(
            f"{ZERO_KWH_CELLS_AT_NEED_MEAN_KEY} must map NEED margins to categories."
        )
    cells: dict[str, tuple[str, ...]] = {}
    for margin, categories in declared.items():
        if margin not in NEED_MARGIN_DIMENSIONS:
            raise ValueError(
                f"{ZERO_KWH_CELLS_AT_NEED_MEAN_KEY} names {margin!r}, not a NEED "
                f"margin ({sorted(NEED_MARGIN_DIMENSIONS)})."
            )
        if (
            isinstance(categories, str)
            or not isinstance(categories, Sequence)
            or not categories
            or not all(isinstance(c, str) and c for c in categories)
        ):
            raise ValueError(
                f"{ZERO_KWH_CELLS_AT_NEED_MEAN_KEY}.{margin} must list NEED categories."
            )
        cells[str(margin)] = tuple(str(c) for c in categories)
    return cells


def _hold_zero_kwh_cells_at_need_mean(
    frame: pd.DataFrame,
    cells: Mapping[str, Sequence[str]],
    *,
    margins: NeedMargins,
    connected: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, str]]:
    """Start each declared cell whose every household drew no kWh at its NEED mean.

    Works on the rake's own copy before the first sweep. A cell is its
    ``_need_<margin>`` key in each NEED geography, over every row for
    electricity and over the gas-connected rows for gas, as the rake sweeps
    them. A cell with a positive row is left to the rake, and so is one with
    no row or no weight. Records the cells it set, never a household.
    """

    held: list[dict[str, str]] = []
    for margin, categories in cells.items():
        column = f"_need_{margin}"
        if column not in frame:
            raise ValueError(
                f"{ZERO_KWH_CELLS_AT_NEED_MEAN_KEY} names the {margin!r} margin, "
                "which this rake does not sweep."
            )
        keys = frame[column].to_numpy().astype(str)
        for category in categories:
            published = {
                geography: cell
                for (geography, name), cell in margins.targets[margin].items()
                if name == category
            }
            if not published:
                raise ValueError(
                    f"{ZERO_KWH_CELLS_AT_NEED_MEAN_KEY}: NEED publishes no {margin} "
                    f"cell {category!r} for {margins.period_value}."
                )
            for geography, cell in sorted(published.items()):
                key = f"{geography}:{category}"
                for fuel, population in (
                    (ELECTRICITY_KWH, np.ones(len(frame), dtype=bool)),
                    (GAS_KWH, connected),
                ):
                    rows = (keys == key) & population
                    if not rows.any() or float(weights[rows].sum()) <= 0.0:
                        continue
                    values = (
                        pd.to_numeric(frame.loc[rows, fuel], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                    if not (values == 0.0).all():
                        continue
                    frame.loc[rows, fuel] = float(cell[fuel])
                    held.append({"margin": margin, "cell": key, "column": fuel})
    return held


def cap_household_kwh(
    values: np.ndarray,
    *,
    weights: np.ndarray,
    population: np.ndarray,
    ceiling: float,
    max_rounds: int = 100,
    cells: Sequence[tuple[np.ndarray, Mapping[str, float]]] = (),
    sweeps: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cap one fuel's household kWh at ``ceiling`` and keep its weighted total.

    Households above the ceiling are set to it, and the kWh they lose goes to
    the other households of the population in proportion to their own use, so
    the levelled total holds; a household lifted over the ceiling by that is
    capped in the next round. The receipt counts the households drawn above
    the ceiling and the share of the total they gave up, and records no
    household's value.

    With ``cells`` (each margin's row labels and levelled cell-mean targets,
    as the rake fitted them) and ``sweeps``, the households below the ceiling
    are first re-raked to those targets with the capped ones held at it, so
    what a capped household gives up goes to its own NEED cells; the
    proportional step then only restores the total (microcosm#1164: handed to
    every household at once, the kWh moved the cells that hold most capped
    households, detached and high-income ones, off their NEED means).
    """

    capped_values = np.asarray(values, dtype=float).copy()
    weight = np.asarray(weights, dtype=float)
    members = np.asarray(population, dtype=bool)
    total = float(np.dot(weight[members], capped_values[members]))
    at_ceiling = np.zeros(len(capped_values), dtype=bool)
    rounds = 0
    feasible = True
    while True:
        over = members & (capped_values > ceiling)
        if not over.any():
            break
        if rounds == max_rounds:
            raise ValueError(f"capping at {ceiling} kWh did not settle.")
        rounds += 1
        at_ceiling |= over
        capped_values[over] = ceiling
        if cells and sweeps:
            _rake_below_ceiling(
                capped_values,
                weights=weight,
                population=members,
                held=at_ceiling,
                cells=cells,
                sweeps=sweeps,
            )
        free = members & ~at_ceiling & (capped_values > 0)
        free_total = float(np.dot(weight[free], capped_values[free]))
        shortfall = total - float(np.dot(weight[members], capped_values[members]))
        if free_total <= 0:
            # No household is left below the ceiling to take the rest: a frame
            # whose weights fall far short of the households it is levelled to
            # (a synthetic fixture). The total then falls short, the receipt
            # says so, and the energy_rake gate refuses it.
            feasible = False
            break
        capped_values[free] *= 1.0 + shortfall / free_total
    before = np.asarray(values, dtype=float)
    # The outliers are the households drawn above the ceiling; one the
    # rescaling lifted to it gave nothing up.
    outliers = members & (before > ceiling)
    removed = float(np.dot(weight[outliers], before[outliers] - ceiling))
    after = float(np.dot(weight[members], capped_values[members]))
    return capped_values, {
        "ceiling_kwh": float(ceiling),
        "households_capped": int(outliers.sum()),
        "households_at_ceiling": int(at_ceiling.sum()),
        "capped_share_of_total": removed / total if total > 0 else 0.0,
        "rounds": rounds,
        "households_above_after": int(
            (members & (capped_values > ceiling * (1 + 1e-12))).sum()
        ),
        "feasible": feasible,
        "frame_kwh_after": after,
    }


def _rake_below_ceiling(
    values: np.ndarray,
    *,
    weights: np.ndarray,
    population: np.ndarray,
    held: np.ndarray,
    cells: Sequence[tuple[np.ndarray, Mapping[str, float]]],
    sweeps: int,
) -> None:
    """Re-rake, in place, the rows not ``held`` to the cell-mean targets.

    Each sweep visits the margins in order; within a cell the held rows keep
    their value and the others are scaled so the cell's weighted mean over its
    population rows is the target, where the held mass leaves room for it.
    """

    margins = [
        [
            (np.flatnonzero(population & (labels == key)), float(target))
            for key, target in targets.items()
        ]
        for labels, targets in cells
    ]
    for _ in range(sweeps):
        for margin in margins:
            for rows, target in margin:
                if rows.size == 0:
                    continue
                fixed = held[rows]
                wanted = target * float(weights[rows].sum())
                fixed_mass = float(np.dot(weights[rows][fixed], values[rows][fixed]))
                free = rows[~fixed]
                current = float(np.dot(weights[free], values[free]))
                if current > 0 and wanted > fixed_mass:
                    values[free] *= (wanted - fixed_mass) / current


def rake_energy_kwh(
    table: pd.DataFrame,
    *,
    margins: NeedMargins,
    frs_region: Sequence[str],
    income: Sequence[float],
    weights: Sequence[float] | None,
    iterations: int,
    tenure: Sequence[str] | None = None,
    accommodation: Sequence[str] | None = None,
    use_region_margin: bool = False,
    gas_connected: Sequence[bool] | None = None,
    level: Mapping[str, float] | None = None,
    ceilings: Mapping[str, float] | None = None,
    zero_kwh_cells_at_need_mean: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rake ``electricity_kwh`` and ``gas_kwh`` to the NEED shape, then to the level.

    Margins are swept income -> tenure -> accommodation -> region per
    iteration (the incumbent's order); each household's categories are
    prefixed by its NEED geography so England-and-Wales and Scotland rows fit
    their own published means. Households outside every category (Northern
    Ireland, unmapped tenures or dwelling types) are untouched by that margin.

    NEED publishes gas means per gas-metered household and electricity means
    per household, so electricity is raked over every row of a cell and gas
    over its ``gas_connected`` rows only (the cell-mean IPF would otherwise
    spread a per-connected mean over unconnected zeros). Without a mask every
    row counts as connected.

    ``level`` (published kWh per column) then scales every row of a fuel by
    one factor so the weighted frame total equals the published volume: the
    NEED means become the shape (every cell mean moves by the same factor,
    which the fit block records as ``level_factor`` beside the ``shape_target``
    and the levelled ``target``).

    ``ceilings`` (kWh per fuel column, after the level) caps each household
    and hands what it loses to the households below the ceiling in its own
    NEED cells, re-raking them to the levelled targets with the capped ones
    held, then restores the published total (``cap_household_kwh``;
    microcosm#1164). The fit is read after the cap.

    ``zero_kwh_cells_at_need_mean`` (NEED categories per margin) starts each
    named cell whose every household drew no kWh at the cell's NEED mean
    before the first sweep, since a cell mean of zero cannot be scaled; the
    receipt lists the cells it set (microcosm#1240).
    """

    frame = table.copy()
    geography = need_geography(frs_region)
    specs: list[MarginSpec] = []
    scratch: list[str] = []
    categories: dict[str, np.ndarray] = {
        "income": income_band_ids(income, margins),
    }
    if tenure is not None:
        categories["tenure"] = np.asarray(
            [NEED_TENURE_IDS.get(str(value), "") for value in tenure], dtype=object
        )
    if accommodation is not None:
        categories["accommodation"] = np.asarray(
            [
                NEED_PROPERTY_TYPE_IDS_BY_GEOGRAPHY.get(str(geo), {}).get(
                    str(value), ""
                )
                for geo, value in zip(geography, accommodation, strict=True)
            ],
            dtype=object,
        )
    if use_region_margin:
        regions = np.asarray(frs_region).astype(str)
        categories["region"] = np.asarray(
            [
                NEED_SCOTLAND_REGION_ID
                if region == "SCOTLAND"
                else NEED_ENGLAND_WALES_REGION_IDS.get(region, "")
                for region in regions
            ],
            dtype=object,
        )
    fitted: dict[str, list[str]] = {}
    for margin in ("income", "tenure", "accommodation", "region"):
        if margin not in categories:
            continue
        column = f"_need_{margin}"
        keys = np.where(
            (geography != "") & (categories[margin] != ""),
            geography.astype(str) + ":" + categories[margin].astype(str),
            "",
        ).astype(object)
        frame[column] = keys
        scratch.append(column)
        targets = {
            f"{geo}:{category}": {
                ELECTRICITY_KWH: cell[ELECTRICITY_KWH],
                GAS_KWH: cell[GAS_KWH],
            }
            for (geo, category), cell in margins.targets[margin].items()
        }
        specs.append(MarginSpec(column, targets))
        fitted[margin] = sorted(set(keys[keys != ""]) & set(targets))
    weight_column = None
    if weights is not None:
        frame["_need_weight"] = np.asarray(weights, dtype=float)
        weight_column = "_need_weight"
        scratch.append("_need_weight")
    connected = (
        np.ones(len(frame), dtype=bool)
        if gas_connected is None
        else np.asarray(gas_connected, dtype=bool)
    )
    if len(connected) != len(frame):
        raise ValueError("gas_connected must align with the table.")
    weight_values = (
        np.ones(len(frame), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    held_at_need_mean = (
        []
        if zero_kwh_cells_at_need_mean is None
        else _hold_zero_kwh_cells_at_need_mean(
            frame,
            zero_kwh_cells_at_need_mean,
            margins=margins,
            connected=connected,
            weights=weight_values,
        )
    )
    electricity_specs = tuple(
        MarginSpec(
            spec.column,
            {k: {ELECTRICITY_KWH: v[ELECTRICITY_KWH]} for k, v in spec.targets.items()},
        )
        for spec in specs
    )
    gas_specs = tuple(
        MarginSpec(
            spec.column, {k: {GAS_KWH: v[GAS_KWH]} for k, v in spec.targets.items()}
        )
        for spec in specs
    )
    raked = iterative_proportional_fit(
        frame,
        columns=(ELECTRICITY_KWH,),
        margins=electricity_specs,
        iterations=iterations,
        weight_column=weight_column,
    )
    zero_cells = list(raked.attrs.get("raking_zero_current_cells", ()))
    sweep_residuals = {
        ELECTRICITY_KWH: [
            float(v) for v in raked.attrs.get("raking_sweep_residuals", ())
        ]
    }
    gas_raked = iterative_proportional_fit(
        frame.loc[connected],
        columns=(GAS_KWH,),
        margins=gas_specs,
        iterations=iterations,
        weight_column=weight_column,
    )
    zero_cells.extend(gas_raked.attrs.get("raking_zero_current_cells", ()))
    sweep_residuals[GAS_KWH] = [
        float(v) for v in gas_raked.attrs.get("raking_sweep_residuals", ())
    ]
    raked.loc[connected, GAS_KWH] = gas_raked[GAS_KWH].to_numpy(dtype=float)
    raked.loc[~connected, GAS_KWH] = 0.0
    level_factor = {ELECTRICITY_KWH: 1.0, GAS_KWH: 1.0}
    level_receipt: dict[str, Any] | None = None
    if level is not None:
        level_receipt = {}
        for column in (ELECTRICITY_KWH, GAS_KWH):
            published = float(level[column])
            values = raked[column].to_numpy(dtype=float)
            population = (
                np.ones(len(frame), dtype=bool)
                if column == ELECTRICITY_KWH
                else connected
            )
            before = float(np.dot(values[population], weight_values[population]))
            if before <= 0.0 or published <= 0.0:
                raise ValueError(
                    f"{column}: cannot level a frame total of {before} to {published}."
                )
            factor = published / before
            raked[column] = values * factor
            after = float(np.dot(raked[column].to_numpy(dtype=float), weight_values))
            level_factor[column] = factor
            level_receipt[column] = {
                "published_kwh": published,
                "frame_kwh_before": before,
                "factor": factor,
                "frame_kwh_after": after,
            }
    ceiling_receipt: dict[str, Any] | None = None
    if ceilings is not None:
        ceiling_receipt = {}
        for column in (ELECTRICITY_KWH, GAS_KWH):
            population = (
                np.ones(len(frame), dtype=bool)
                if column == ELECTRICITY_KWH
                else connected
            )
            capped, ceiling_receipt[column] = cap_household_kwh(
                raked[column].to_numpy(dtype=float),
                weights=weight_values,
                population=population,
                ceiling=float(ceilings[column]),
                cells=[
                    (
                        frame[spec.column].to_numpy().astype(str),
                        {
                            key: float(cell[column]) * level_factor[column]
                            for key, cell in spec.targets.items()
                            if key in fitted[margin]
                        },
                    )
                    for margin, spec in zip(fitted, specs, strict=True)
                ],
                sweeps=int(iterations),
            )
            raked[column] = capped
    fit = _rake_fit(raked, margins, fitted, connected, weight_values, level_factor)
    receipt = {
        "unit": "kwh",
        "iterations": int(iterations),
        "weighted": weights is not None,
        "margins": list(fitted),
        "margins_period_value": margins.period_value,
        "populated_cells": {margin: len(keys) for margin, keys in fitted.items()},
        "gas_rake_population": "gas_connected_rows",
        "gas_connected_rows": int(connected.sum()),
        "rows": int(len(frame)),
        "zero_current_cells": zero_cells,
        "cells_held_at_need_mean": held_at_need_mean,
        "sweep_residuals": sweep_residuals,
        "level_factor": dict(level_factor),
        "level": level_receipt,
        "ceiling": ceiling_receipt,
        "fit": fit,
    }
    return raked.drop(columns=[c for c in scratch if c in raked]), receipt


def _rake_fit(
    raked: pd.DataFrame,
    margins: NeedMargins,
    fitted: Mapping[str, Sequence[str]],
    connected: np.ndarray,
    weights: np.ndarray,
    level_factor: Mapping[str, float],
) -> dict[str, Any]:
    """Per margin: the prior-weighted cell means after the rake against NEED x level.

    Electricity is averaged over every row of a cell, gas over its connected
    rows, the populations the rake itself used. Each cell records the NEED
    mean (``shape_target``), the fuel's ``level_factor`` and their product,
    the ``target`` the levelled frame is held to. The stage-time
    ``energy_rake`` health check reads the maximum absolute relative
    deviation per margin and fuel from this block (microcosm#890: NEED is
    checked where the rake acts, at prior weights, not on the calibrated
    frame).
    """

    electricity = raked[ELECTRICITY_KWH].to_numpy(dtype=float)
    gas = raked[GAS_KWH].to_numpy(dtype=float)
    fit: dict[str, Any] = {}
    for margin, keys in fitted.items():
        column = f"_need_{margin}"
        labels = raked[column].to_numpy().astype(str)
        cells: dict[str, dict[str, Any]] = {}
        worst = {ELECTRICITY_KWH: 0.0, GAS_KWH: 0.0}
        for key in keys:
            geo, category = key.split(":", 1)
            shape = margins.targets[margin][(geo, category)]
            rows = labels == key
            entry: dict[str, Any] = {}
            for fuel, values, population in (
                (ELECTRICITY_KWH, electricity, rows),
                (GAS_KWH, gas, rows & connected),
            ):
                total_weight = float(weights[population].sum())
                achieved = (
                    float(
                        np.dot(values[population], weights[population]) / total_weight
                    )
                    if total_weight > 0
                    else None
                )
                target = shape[fuel] * float(level_factor[fuel])
                deviation = (
                    None if achieved is None or target <= 0 else achieved / target - 1.0
                )
                entry[fuel] = {
                    "shape_target": shape[fuel],
                    "level_factor": float(level_factor[fuel]),
                    "target": target,
                    "achieved": achieved,
                    "relative_deviation": deviation,
                    "weighted_rows": total_weight,
                }
                if deviation is not None:
                    worst[fuel] = max(worst[fuel], abs(deviation))
            cells[key] = entry
        fit[margin] = {
            "cells": cells,
            "max_abs_relative_deviation": dict(worst),
        }
    return fit
