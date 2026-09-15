"""Price imputed domestic energy at the FY2024-25 Ofgem cap and rake in kWh (microcosm#890 F).

The LCFS diary and the consumption QRF carry energy as pounds. NEED publishes
mean annual kWh by household income band, tenure, property type and region
(England and Wales; Scotland by income, tenure and property). Ofgem publishes
the price cap as a maximum annual charge at two consumption levels per
charge-restriction region, payment method and fuel, not as per-unit rates.
This module ties the three together under one declared operation,
``price_energy_at_cap``:

* unit rate = (benchmark-consumption level - nil-consumption level) / the
  period's benchmark consumption, standing charge = the nil-consumption
  level, both the fiscal-year mean of the four quarterly cap periods, in the
  region the FRS region maps to (a declared crosswalk), including VAT at the
  declared reduced rate, which the GB including-VAT rows must reproduce;
* spend -> kWh = max(spend - standing charge, 0) / unit rate for connected
  households (electricity: every household; gas: positive gas spend), so the
  NEED margins can be raked in the unit NEED publishes;
* kWh -> spend = kWh x unit rate + standing charge for connected households.

Every rate, margin and fit is returned in receipts the stage records.
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
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

PRICE_ENERGY_AT_CAP_KIND = "price_energy_at_cap"
UK_OFGEM_PRICE_CAP_RESOURCE = "ofgem_price_cap_facts.json"
UK_NEED_ENERGY_FACTS_RESOURCE = "need_energy_facts.json"
UK_OFGEM_REGION_CROSSWALK_RESOURCE = "ofgem_region_crosswalk.json"
CAP_LEVEL_CONCEPT = "ofgem.price_cap.cap_level"
BENCHMARK_CONSUMPTION_CONCEPT = "ofgem.price_cap.benchmark_consumption"
NEED_ELECTRICITY_CONCEPT = "desnz.need.household_electricity_consumption"
NEED_GAS_CONCEPT = "desnz.need.household_gas_consumption"
ENGLAND_AND_WALES_GEOGRAPHY_ID = "K04000001"
SCOTLAND_GEOGRAPHY_ID = "S92000003"
GAS_FUEL = "gas"
ELECTRICITY_KWH = "electricity_kwh"
GAS_KWH = "gas_kwh"
GAS_CONNECTED_POSITIVE_SPEND = "positive_gas_spend"
_VAT_CHECK_TOLERANCE = 1e-4

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
class EnergyCapRates:
    """Fiscal-year mean unit rates (GBP/kWh) and standing charges (GBP/yr)."""

    fiscal_start: str
    vat_rate: float
    electricity_fuel: str
    unit_rate: Mapping[tuple[str, str], float]
    standing_charge: Mapping[tuple[str, str], float]
    region_of_frs_region: Mapping[str, str]
    gb_geography_id: str

    def region_for(self, frs_region: str) -> str:
        try:
            return self.region_of_frs_region[str(frs_region)]
        except KeyError as error:
            raise KeyError(
                f"FRS region {frs_region!r} has no Ofgem charge-restriction region "
                "in the declared crosswalk."
            ) from error

    def fuel_id(self, fuel: str) -> str:
        return self.electricity_fuel if fuel == "electricity" else GAS_FUEL


@dataclass(frozen=True)
class NeedMargins:
    """Mean kWh targets per margin, keyed by (geography id, NEED category id)."""

    targets: Mapping[str, Mapping[tuple[str, str], Mapping[str, float]]]
    income_bands: tuple[tuple[str, float, float], ...]
    receipt: dict[str, Any]


def pricing_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    for operation in stage.operations:
        if operation.kind == PRICE_ENERGY_AT_CAP_KIND:
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


def cap_rates(parameters: Mapping[str, Any]) -> tuple[EnergyCapRates, dict[str, Any]]:
    """Resolve the declared FY cap rates per charge-restriction region and fuel."""

    resource = str(parameters.get("resource") or "")
    if resource != UK_OFGEM_PRICE_CAP_RESOURCE:
        raise ValueError(
            f"price_energy_at_cap must read {UK_OFGEM_PRICE_CAP_RESOURCE!r}, "
            f"not {resource!r}."
        )
    if str(parameters.get("crosswalk_resource")) != UK_OFGEM_REGION_CROSSWALK_RESOURCE:
        raise ValueError("price_energy_at_cap must use the committed region crosswalk.")
    crosswalk = load_ofgem_region_crosswalk()
    mapping = {str(k): str(v) for k, v in crosswalk["mapping"].items()}
    gb = str(crosswalk["gb_geography_id"])
    fiscal_start = str(parameters["fiscal_start"])
    quarters = fiscal_year_quarters(fiscal_start)
    payment_method = str(parameters["payment_method"])
    electricity_fuel = str(parameters["electricity_fuel"])
    vat_rate = parameters.get("vat_rate")
    if vat_rate is None:
        raise ValueError(
            "price_energy_at_cap declares no vat_rate (the reduced VAT rate the GB "
            "including-VAT rows must reproduce)."
        )
    vat_rate = float(vat_rate)
    if not 0.0 <= vat_rate < 1.0:
        raise ValueError("vat_rate must lie in [0, 1).")

    def levels(
        region: str, fuel: str, consumption_level: str, vat: str
    ) -> dict[str, float]:
        rows = vendored_rows(
            resource,
            concept=CAP_LEVEL_CONCEPT,
            geography_id=region,
            dimensions={
                "fuel": fuel,
                "consumption_level": consumption_level,
                "payment_method": payment_method,
                "vat_treatment": vat,
            },
        )
        by_quarter = {
            str(row["period_coverage"]["start_date"]): float(row["value"])
            for row in rows
            if str(row["period_coverage"]["start_date"]) in quarters
        }
        missing = [q for q in quarters if q not in by_quarter]
        if missing:
            raise ValueError(
                f"{resource}: {region} {fuel} {consumption_level} ({vat}) lacks "
                f"cap levels for quarters starting {missing}."
            )
        return by_quarter

    def benchmark_kwh(fuel: str) -> dict[str, float]:
        rows = vendored_rows(
            resource,
            concept=BENCHMARK_CONSUMPTION_CONCEPT,
            geography_id=gb,
            dimensions={"fuel": fuel},
        )
        by_quarter = {
            str(row["period_coverage"]["start_date"]): float(row["value"])
            for row in rows
            if str(row["period_coverage"]["start_date"]) in quarters
        }
        missing = [q for q in quarters if q not in by_quarter]
        if missing:
            raise ValueError(f"{resource}: benchmark kWh for {fuel} lacks {missing}.")
        if any(value <= 0 for value in by_quarter.values()):
            raise ValueError(f"{resource}: benchmark kWh for {fuel} must be positive.")
        return by_quarter

    # The GB rows carry both VAT treatments: the declared rate must reproduce
    # the publisher's own including-VAT levels before it is applied anywhere.
    vat_checks: list[dict[str, Any]] = []
    for fuel in (electricity_fuel, GAS_FUEL):
        for level in ("benchmark_consumption", "nil_consumption"):
            excl = levels(gb, fuel, level, "excluding_vat")
            incl = levels(gb, fuel, level, "including_vat")
            for quarter in quarters:
                ratio = incl[quarter] / excl[quarter]
                vat_checks.append(
                    {"fuel": fuel, "level": level, "quarter": quarter, "ratio": ratio}
                )
                if abs(ratio - (1.0 + vat_rate)) > _VAT_CHECK_TOLERANCE:
                    raise ValueError(
                        f"{resource}: GB {fuel} {level} {quarter} including/excluding "
                        f"VAT ratio {ratio:.5f} does not match the declared "
                        f"vat_rate {vat_rate}."
                    )
    regions = sorted({*mapping.values(), gb})
    unit_rate: dict[tuple[str, str], float] = {}
    standing_charge: dict[tuple[str, str], float] = {}
    by_region: dict[str, dict[str, Any]] = {}
    kwh = {fuel: benchmark_kwh(fuel) for fuel in (electricity_fuel, GAS_FUEL)}
    for region in regions:
        by_region[region] = {}
        for fuel in (electricity_fuel, GAS_FUEL):
            benchmark = levels(region, fuel, "benchmark_consumption", "excluding_vat")
            nil = levels(region, fuel, "nil_consumption", "excluding_vat")
            unit_quarters = {
                q: (benchmark[q] - nil[q]) / kwh[fuel][q] * (1.0 + vat_rate)
                for q in quarters
            }
            charge_quarters = {q: nil[q] * (1.0 + vat_rate) for q in quarters}
            if any(value <= 0 for value in unit_quarters.values()):
                raise ValueError(
                    f"{resource}: {region} {fuel} unit rate must be positive."
                )
            unit_rate[(region, fuel)] = float(np.mean(list(unit_quarters.values())))
            standing_charge[(region, fuel)] = float(
                np.mean(list(charge_quarters.values()))
            )
            by_region[region][fuel] = {
                "unit_rate_gbp_per_kwh": unit_rate[(region, fuel)],
                "standing_charge_gbp_per_year": standing_charge[(region, fuel)],
                "quarterly_unit_rates": unit_quarters,
                "quarterly_standing_charges": charge_quarters,
                "benchmark_kwh": kwh[fuel],
            }
    rates = EnergyCapRates(
        fiscal_start=fiscal_start,
        vat_rate=vat_rate,
        electricity_fuel=electricity_fuel,
        unit_rate=unit_rate,
        standing_charge=standing_charge,
        region_of_frs_region=mapping,
        gb_geography_id=gb,
    )
    receipt = {
        "operation": PRICE_ENERGY_AT_CAP_KIND,
        "resource": resource,
        "crosswalk_resource": UK_OFGEM_REGION_CROSSWALK_RESOURCE,
        "fiscal_start": fiscal_start,
        "quarters": list(quarters),
        "payment_method": payment_method,
        "electricity_fuel": electricity_fuel,
        "vat_rate": vat_rate,
        "vat_parameter_path": parameters.get("vat_parameter_path"),
        "vat_check": {
            "rows": len(vat_checks),
            "min_ratio": min(check["ratio"] for check in vat_checks),
            "max_ratio": max(check["ratio"] for check in vat_checks),
        },
        "gas_connected": str(
            parameters.get("gas_connected", GAS_CONNECTED_POSITIVE_SPEND)
        ),
        "region_of_frs_region": dict(mapping),
        "rates_by_region": by_region,
    }
    return rates, receipt


def spend_to_kwh(
    spend: np.ndarray,
    *,
    frs_region: Sequence[str],
    fuel: str,
    rates: EnergyCapRates,
    connected: np.ndarray | None = None,
) -> np.ndarray:
    """kWh = max(spend - standing charge, 0) / unit rate for connected rows, else 0."""

    values = np.asarray(spend, dtype=float)
    regions = np.asarray(frs_region).astype(str)
    fuel_id = rates.fuel_id(fuel)
    unit = np.asarray(
        [rates.unit_rate[(rates.region_for(region), fuel_id)] for region in regions]
    )
    charge = np.asarray(
        [
            rates.standing_charge[(rates.region_for(region), fuel_id)]
            for region in regions
        ]
    )
    mask = (
        np.ones(len(values), dtype=bool)
        if connected is None
        else np.asarray(connected, dtype=bool)
    )
    kwh = np.where(mask, np.maximum(values - charge, 0.0) / unit, 0.0)
    return kwh


def kwh_to_spend(
    kwh: np.ndarray,
    *,
    frs_region: Sequence[str],
    fuel: str,
    rates: EnergyCapRates,
    connected: np.ndarray | None = None,
) -> np.ndarray:
    """spend = kWh x unit rate + standing charge for connected rows, else 0."""

    values = np.asarray(kwh, dtype=float)
    regions = np.asarray(frs_region).astype(str)
    fuel_id = rates.fuel_id(fuel)
    unit = np.asarray(
        [rates.unit_rate[(rates.region_for(region), fuel_id)] for region in regions]
    )
    charge = np.asarray(
        [
            rates.standing_charge[(rates.region_for(region), fuel_id)]
            for region in regions
        ]
    )
    mask = (
        np.ones(len(values), dtype=bool)
        if connected is None
        else np.asarray(connected, dtype=bool)
    )
    return np.where(mask, np.maximum(values, 0.0) * unit + charge, 0.0)


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
    resource: str = UK_NEED_ENERGY_FACTS_RESOURCE, *, statistic: str = "mean"
) -> NeedMargins:
    """Mean kWh per margin category from the vendored NEED rows, both geographies."""

    if resource != UK_NEED_ENERGY_FACTS_RESOURCE:
        raise ValueError(
            f"NEED margins must come from {UK_NEED_ENERGY_FACTS_RESOURCE!r}."
        )
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
            raise ValueError(f"{resource}: Scotland lacks an all_dwellings row.")
        targets["region"][scotland_region] = dict(all_dwellings)
    for margin, cells in targets.items():
        for key, cell in cells.items():
            if set(cell) != {ELECTRICITY_KWH, GAS_KWH}:
                raise ValueError(f"{resource}: {margin} cell {key} lacks a fuel.")
    if not targets["region"] or all(
        g != SCOTLAND_GEOGRAPHY_ID for g, _ in targets["income"]
    ):
        raise ValueError(f"{resource}: missing a NEED geography.")
    bands = _income_band_edges(sorted({band for geography, band in targets["income"]}))
    receipt = {
        "resource": resource,
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
    return NeedMargins(targets=targets, income_bands=bands, receipt=receipt)


def income_band_ids(income: Sequence[float], margins: NeedMargins) -> np.ndarray:
    values = pd.to_numeric(pd.Series(income), errors="coerce").fillna(0.0).to_numpy()
    result = np.full(len(values), "", dtype=object)
    for band, lower, upper in margins.income_bands:
        result[(values >= lower) & (values < upper)] = band
    return result


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rake ``electricity_kwh`` and ``gas_kwh`` to the NEED margins.

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
    gas_raked = iterative_proportional_fit(
        frame.loc[connected],
        columns=(GAS_KWH,),
        margins=gas_specs,
        iterations=iterations,
        weight_column=weight_column,
    )
    zero_cells.extend(gas_raked.attrs.get("raking_zero_current_cells", ()))
    raked.loc[connected, GAS_KWH] = gas_raked[GAS_KWH].to_numpy(dtype=float)
    receipt = {
        "iterations": int(iterations),
        "weighted": weights is not None,
        "margins": list(fitted),
        "populated_cells": {margin: len(keys) for margin, keys in fitted.items()},
        "gas_rake_population": "gas_connected_rows",
        "gas_connected_rows": int(connected.sum()),
        "rows": int(len(frame)),
        "zero_current_cells": zero_cells,
    }
    return raked.drop(columns=[c for c in scratch if c in raked]), receipt
