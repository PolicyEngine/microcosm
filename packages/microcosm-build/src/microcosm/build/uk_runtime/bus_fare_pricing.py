"""Bus fares priced from imputed journeys at a published yield (microcosm#930).

The lcfs stage used to set ``bus_fare_spending`` by raking the diary draw to
the DfT receipts it is calibrated against (microcosm#890, rake-then-bind).
Here the level comes from a quantity the ``nts_bus_travel`` stage imputed and
two published ratios, so the calibration targets constrain something the
stage did not set:

* the yield per fare-paying boarding, ``y = R / (B - C)``: BUS05ai passenger
  fare receipts over BUS01 passenger journeys less the concessionary journeys,
  per area (London and England outside London; Scotland and Northern Ireland
  from their own receipts and journeys);
* boardings per resident trip, ``k = B / (T x P)``: the published boardings over
  the published NTS trips per resident times the resident population, per
  series, which translates a survey trip (one-way, main mode, England
  residents) into operator boardings (interchanges, non-residents, business
  payers; London contactless is not separated).

A person's fares are ``sum over series of trips x k x y`` unless the person is
concessionary-eligible under the statutory rule of their area, whose trips are
priced at zero. ``bus_in_london`` trips of every resident take the London area's
price; ``other_local_bus`` trips take the price of the person's residence area.
Regions with no published receipts (Wales) are left unpriced and keep the
chain's raw draw, clipped to donor support (María's ruling, 2026-09-17).

The receipts and journeys are fiscal-year figures. When the declaration names a
``price_year``, each area's yield is re-priced to that calendar year with a
published fares index (the DfT local bus fares index by quarter, Scotland's
calendar-year index), so bus fares sit on calendar 2024 with every other LCFS
spend column (microcosm#1113). Northern Ireland publishes no fares index; its
yield is re-priced from its own published yields of the two fiscal years the
calendar year overlaps (microcosm#1207), not from England's index, which
carries the January 2025 rise in the English fare cap.

Every input is a vendored Chronicle row; the receipt records every factor, the
frame-implied boardings beside the published ones, and the frame's eligible
trip share beside the publisher's concessionary boarding share, so the
design-weight fit of the bound receipts targets can be read as what it is: a
survey-versus-survey composition check times a survey-versus-admin concession
check, never a level the stage set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.ledger_targets import (
    CALENDAR_YEAR_WINDOW_WEIGHTS,
    LedgerTargetReference,
    TargetRegistry,
)
from microcosm.build.uk_runtime.energy_pricing import period_quarters
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

PRICE_BUS_JOURNEYS_KIND = "price_bus_journeys"
#: ``price_index.basis`` of an area re-priced from its own published yields
#: rather than a fares index (microcosm#1207).
OWN_FISCAL_YIELDS_BASIS = "own_fiscal_yields"
#: policyengine-uk's bus pricing area (``gov.dft.bus.*`` keys) for each of the
#: stage's priced areas (microcosm#1207).
UK_ENGINE_BUS_AREA_BY_STAGE_AREA: Mapping[str, str] = {
    "london": "LONDON",
    "england_outside_london": "ENGLAND_OUTSIDE_LONDON",
    "scotland": "SCOTLAND",
    "northern_ireland": "NORTHERN_IRELAND",
}
#: Relative gap above which the engine's bus pricing has drifted from the
#: stage's: both derive k and y from the same published facts (microcosm#954),
#: so a gap is a re-pin or a publisher restatement the engine has not taken.
UK_BUS_ENGINE_PARITY_TOLERANCE = 0.001
UK_DFT_BUS_JOURNEYS_RESOURCE = "dft_bus_journeys.json"
BUS01_JOURNEYS_CONCEPT = "dft.local_bus_passenger_journeys"
BUS_IN_LONDON = "bus_in_london"
OTHER_LOCAL_BUS = "other_local_bus"
SERIES: tuple[str, str] = (BUS_IN_LONDON, OTHER_LOCAL_BUS)
ONS_POPULATION_CONCEPT = "ons.mid_year_population_estimate"


class BusFarePricingError(ValueError):
    """Raised when the declaration, the vendored rows or the frame refuse pricing."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BusFarePricingError(f"{label} must be a mapping.")
    return value


def published_fact(
    selector: Mapping[str, Any],
    *,
    unit: str,
    allowed_resources: Sequence[str],
    fiscal_start: str | None = None,
    period_value: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """One positive published value from vendored rows under a declared selector.

    ``selector`` names ``resource`` and ``concept`` and may pin
    ``geography_id``, ``groupby_value_id``, ``dimension_values`` (all must
    match) or ``sum_over`` (one dimension, every named value summed). Rows
    with dimensions are excluded unless the selector asks for them, so a
    concept that carries both a total and a split resolves to its total.
    """

    resource = str(selector.get("resource") or "")
    if resource not in allowed_resources:
        raise BusFarePricingError(
            f"selector names {resource!r}, not a resource the stage declares "
            f"({sorted(allowed_resources)})."
        )
    criteria: dict[str, Any] = {"concept": str(selector["concept"])}
    for key in ("geography_id", "groupby_value_id"):
        if selector.get(key):
            criteria[key] = str(selector[key])
    if period_value is not None:
        criteria["period_value"] = int(period_value)
    dimension_values = {
        str(k): str(v) for k, v in dict(selector.get("dimension_values") or {}).items()
    }
    if dimension_values:
        criteria["dimensions"] = dimension_values
    rows = vendored_rows(resource, fiscal_start=fiscal_start, **criteria)
    sum_over = dict(selector.get("sum_over") or {})
    if sum_over:
        if len(sum_over) != 1:
            raise BusFarePricingError("sum_over names exactly one dimension.")
        ((dimension, values),) = sum_over.items()
        wanted = sorted(str(v) for v in values)
        rows = [
            row
            for row in rows
            if str((row.get("dimensions") or {}).get(dimension)) in wanted
        ]
        found = sorted(
            str((row.get("dimensions") or {}).get(dimension)) for row in rows
        )
        if found != wanted:
            raise BusFarePricingError(
                f"{resource}: sum_over {dimension!r} expected {wanted}, found {found}."
            )
    else:
        if not dimension_values:
            rows = [row for row in rows if not (row.get("dimensions") or {})]
        if len(rows) != 1:
            raise BusFarePricingError(
                f"{resource}: expected exactly one row for {dict(selector)}, "
                f"found {len(rows)}."
            )
    units = {str(row.get("unit")) for row in rows}
    if units != {unit}:
        raise BusFarePricingError(
            f"{resource}: expected {unit!r} rows for {dict(selector)}, got {sorted(units)}."
        )
    value = float(sum(float(row["value"]) for row in rows))
    if not np.isfinite(value) or value <= 0:
        raise BusFarePricingError(
            f"{resource}: published value for {dict(selector)} must be positive."
        )
    return value, {
        "selector": dict(selector),
        "value": value,
        "unit": unit,
        "source_record_ids": [str(row.get("source_record_id", "")) for row in rows],
    }


@dataclass(frozen=True)
class AreaPrice:
    """The published inputs and derived factors of one priced area."""

    label: str
    series: str
    regions: tuple[str, ...]
    receipts: float
    boardings: float
    concessionary_boardings: float
    trips_per_person: float
    population: float
    boardings_per_trip: float
    yield_per_fare_paying_boarding: float
    fare_per_trip: float
    concessionary_boarding_share: float
    calendar_price_factor: float
    trips_basis: str
    receipt: dict[str, Any]


@dataclass(frozen=True)
class BusFarePrices:
    london: AreaPrice
    other_by_region: Mapping[str, AreaPrice]
    unpriced_regions: frozenset[str]
    receipt: dict[str, Any]

    def area_for_region(self, region: str) -> AreaPrice | None:
        return self.other_by_region.get(str(region))


def calendar_price_factor(
    spec: Mapping[str, Any],
    *,
    price_year: int,
    fiscal_start: str,
    allowed_resources: Sequence[str],
) -> tuple[float, dict[str, Any]]:
    """The factor that re-prices a fiscal-year yield to a calendar year.

    ``quarterly``: the fares index's mean over the calendar year's four
    quarters over its mean over the fiscal year's (rows keyed by quarter
    coverage). ``annual``: a calendar-year index, the fiscal year starting in
    April of Y being three quarters of Y and one of Y + 1.
    """

    resource = str(spec.get("resource") or "")
    if resource not in allowed_resources:
        raise BusFarePricingError(f"price index resource {resource!r} is not declared.")
    concept = str(spec.get("concept") or "")
    criteria: dict[str, Any] = {"concept": concept}
    if spec.get("geography_id") is not None:
        criteria["geography_id"] = str(spec["geography_id"])
    if not fiscal_start.endswith("-04-01"):
        raise BusFarePricingError("fiscal_start must be an April 1st.")
    fiscal_year = int(fiscal_start[:4])
    frequency = str(spec.get("frequency") or "")
    receipt: dict[str, Any] = {
        "resource": resource,
        "concept": concept,
        "geography_id": criteria.get("geography_id"),
        "frequency": frequency,
        "basis": str(spec.get("basis", "own_index")),
        "price_year": int(price_year),
        "fiscal_start": fiscal_start,
    }
    if frequency == "quarterly":
        by_start: dict[str, float] = {}
        for row in vendored_rows(resource, **criteria):
            start = str((row.get("period_coverage") or {}).get("start_date"))
            if start in by_start:
                raise BusFarePricingError(
                    f"{resource}: duplicate {concept} quarter {start}."
                )
            by_start[start] = float(row["value"])
        calendar = period_quarters("calendar_year", price_year)
        fiscal = period_quarters("fiscal_year", fiscal_year)
        missing = [q for q in (*calendar, *fiscal) if q not in by_start]
        if missing:
            raise BusFarePricingError(
                f"{resource}: {concept} lacks quarters {missing}."
            )
        calendar_value = sum(by_start[q] for q in calendar) / 4.0
        fiscal_value = sum(by_start[q] for q in fiscal) / 4.0
        receipt["calendar_quarters"] = {q: by_start[q] for q in calendar}
        receipt["fiscal_quarters"] = {q: by_start[q] for q in fiscal}
    elif frequency == "annual":
        values = {}
        for year in sorted({int(price_year), fiscal_year, fiscal_year + 1}):
            rows = vendored_rows(
                resource, period_type="calendar_year", period_value=year, **criteria
            )
            if len(rows) != 1:
                raise BusFarePricingError(
                    f"{resource}: expected one {concept} row for {year}, found {len(rows)}."
                )
            values[year] = float(rows[0]["value"])
        calendar_value = values[int(price_year)]
        fiscal_value = 0.75 * values[fiscal_year] + 0.25 * values[fiscal_year + 1]
        receipt["calendar_years"] = {str(year): value for year, value in values.items()}
    else:
        raise BusFarePricingError(f"unsupported price index frequency {frequency!r}.")
    if calendar_value <= 0 or fiscal_value <= 0:
        raise BusFarePricingError(f"{resource}: {concept} must be positive.")
    factor = calendar_value / fiscal_value
    receipt.update(
        {
            "calendar_value": calendar_value,
            "fiscal_value": fiscal_value,
            "factor": factor,
        }
    )
    return factor, receipt


def own_fiscal_yield_price_factor(
    area: Mapping[str, Any],
    *,
    price_year: int,
    fiscal_start: str,
    allowed_resources: Sequence[str],
) -> tuple[float, dict[str, Any]]:
    """Re-price an area's fiscal-year yield to a calendar year from its own yields.

    For an area with no published fares index (Northern Ireland,
    microcosm#1207): the yield ``y = R / (B - C)`` of the fiscal year opening
    in the calendar year and of the one opening the year before, each from the
    area's own declared receipts, boardings and concessionary selectors, are
    weighted by the months each contributes to the calendar year (the
    ``calendar_year_window`` weights: three months of the earlier year, nine
    of the later one), over the later year's yield. A yield moves with fares
    and with the mix of tickets, so the factor is the area's own published
    price movement, not an index.
    """

    label = str(area.get("label", ""))
    spec = _mapping(area.get("price_index"), f"{label}.price_index")
    if set(spec) != {"basis"} or spec["basis"] != OWN_FISCAL_YIELDS_BASIS:
        raise BusFarePricingError(
            f"area {label!r}: an {OWN_FISCAL_YIELDS_BASIS} price index declares "
            "only its basis; the yields come from the area's own selectors."
        )
    if not fiscal_start.endswith("-04-01"):
        raise BusFarePricingError("fiscal_start must be an April 1st.")
    fiscal_year = int(fiscal_start[:4])
    if fiscal_year != int(price_year):
        raise BusFarePricingError(
            f"area {label!r}: {OWN_FISCAL_YIELDS_BASIS} re-prices the fiscal year "
            f"opening in the price year; fiscal_start {fiscal_start} does not open "
            f"in {price_year}."
        )
    years: dict[str, dict[str, Any]] = {}
    calendar_value = 0.0
    for offset, weight in sorted(CALENDAR_YEAR_WINDOW_WEIGHTS.items()):
        start = f"{fiscal_year + offset}-04-01"
        facts = {}
        records: list[str] = []
        for key, unit in (
            ("receipts", "gbp"),
            ("boardings", "count"),
            ("concessionary", "count"),
        ):
            value, fact_receipt = published_fact(
                _mapping(area[key], f"{label}.{key}"),
                unit=unit,
                allowed_resources=allowed_resources,
                fiscal_start=start,
            )
            facts[key] = value
            records.extend(fact_receipt["source_record_ids"])
        if facts["concessionary"] >= facts["boardings"]:
            raise BusFarePricingError(
                f"area {label!r}: concessionary journeys of the fiscal year opening "
                f"{start} are not below its total journeys."
            )
        fiscal_yield = facts["receipts"] / (facts["boardings"] - facts["concessionary"])
        calendar_value += weight * fiscal_yield
        years[start] = {
            **facts,
            "yield_per_fare_paying_boarding": fiscal_yield,
            "weight": weight,
            "source_record_ids": records,
        }
    fiscal_value = years[fiscal_start]["yield_per_fare_paying_boarding"]
    factor = calendar_value / fiscal_value
    return factor, {
        "basis": OWN_FISCAL_YIELDS_BASIS,
        "price_year": int(price_year),
        "fiscal_start": fiscal_start,
        "fiscal_years": years,
        "calendar_value": calendar_value,
        "fiscal_value": fiscal_value,
        "factor": factor,
    }


def bus_engine_parity(
    prices: BusFarePrices,
    engine: Mapping[str, Mapping[str, float]],
    *,
    tolerance: float = UK_BUS_ENGINE_PARITY_TOLERANCE,
) -> dict[str, Any]:
    """The engine's bus pricing parameters against the stage's, area by area.

    policyengine-uk prices bus fares itself from the trips a dataset carries
    (``person_bus_fare_spending``) at its own ``gov.dft.bus.boardings_per_trip``
    and ``yield_per_boarding``, while the national fare rows calibrate the
    stage's priced column; this is the check microcosm#1207 asks for. Both
    sides are fiscal-year values, before the stage's calendar re-pricing.
    ``engine`` maps each parameter name to its value by engine area. Report
    only: a gap beyond ``tolerance`` marks the area, it refuses nothing.
    """

    stage_areas = {
        prices.london.label: prices.london,
        **{area.label: area for area in prices.other_by_region.values()},
    }
    areas: dict[str, Any] = {}
    for label, area in sorted(stage_areas.items()):
        engine_area = UK_ENGINE_BUS_AREA_BY_STAGE_AREA.get(label)
        if engine_area is None:
            areas[label] = {"engine_area": None}
            continue
        entry: dict[str, Any] = {"engine_area": engine_area}
        for name, stage_value in (
            ("boardings_per_trip", area.boardings_per_trip),
            ("yield_per_boarding", area.yield_per_fare_paying_boarding),
        ):
            engine_value = float(engine[name][engine_area])
            relative = stage_value / engine_value - 1.0 if engine_value else None
            entry[name] = {
                "stage": float(stage_value),
                "engine": engine_value,
                "relative_gap": relative,
                "within_tolerance": relative is not None and abs(relative) <= tolerance,
            }
        areas[label] = entry
    priced = [entry for entry in areas.values() if entry["engine_area"] is not None]
    return {
        "report_only": True,
        "tolerance": float(tolerance),
        "areas": areas,
        "within_tolerance": all(
            entry[name]["within_tolerance"]
            for entry in priced
            for name in ("boardings_per_trip", "yield_per_boarding")
        ),
    }


def bus_fare_prices(
    parameters: Mapping[str, Any], *, allowed_resources: Sequence[str]
) -> BusFarePrices:
    """Resolve every area's price from the declaration and the vendored rows."""

    if parameters.get("kind", PRICE_BUS_JOURNEYS_KIND) != PRICE_BUS_JOURNEYS_KIND:
        raise BusFarePricingError("not a price_bus_journeys declaration.")
    fiscal_start = str(parameters["fiscal_start"])
    trip_rates_resource = str(parameters["trip_rates_resource"])
    trip_rates_period = int(parameters["trip_rates_period_value"])
    population_resource = str(parameters["population_resource"])
    population_period = int(parameters["population_period_value"])
    price_year = parameters.get("price_year")
    areas = parameters.get("areas")
    if not isinstance(areas, list) or not areas:
        raise BusFarePricingError("price_bus_journeys declares no areas.")
    unpriced = frozenset(str(r) for r in parameters.get("unpriced_regions", ()))
    london: AreaPrice | None = None
    other_by_region: dict[str, AreaPrice] = {}
    area_receipts: list[dict[str, Any]] = []
    for area in areas:
        area = _mapping(area, "areas[]")
        label = str(area["label"])
        series = str(area["series"])
        if series not in SERIES:
            raise BusFarePricingError(
                f"area {label!r} names an unknown series {series!r}."
            )
        regions = tuple(str(r) for r in area.get("regions", ()))
        receipts, receipts_receipt = published_fact(
            _mapping(area["receipts"], f"{label}.receipts"),
            unit="gbp",
            allowed_resources=allowed_resources,
            fiscal_start=fiscal_start,
        )
        boardings, boardings_receipt = published_fact(
            _mapping(area["boardings"], f"{label}.boardings"),
            unit="count",
            allowed_resources=allowed_resources,
            fiscal_start=fiscal_start,
        )
        concessionary, concessionary_receipt = published_fact(
            _mapping(area["concessionary"], f"{label}.concessionary"),
            unit="count",
            allowed_resources=allowed_resources,
            fiscal_start=fiscal_start,
        )
        if concessionary >= boardings:
            raise BusFarePricingError(
                f"area {label!r}: concessionary journeys {concessionary} are not "
                f"below total journeys {boardings}."
            )
        trips_concept = str(area["trips_concept"])
        trips_per_person, trips_receipt = published_fact(
            {
                "resource": trip_rates_resource,
                "concept": trips_concept,
                "groupby_value_id": "all",
            },
            unit="trips_per_person_per_year",
            allowed_resources=allowed_resources,
            period_value=trip_rates_period,
        )
        population_ids = [str(g) for g in area.get("population_geography_ids", ())]
        if not population_ids:
            raise BusFarePricingError(f"area {label!r} names no population geography.")
        population = 0.0
        population_records: list[str] = []
        for geography_id in population_ids:
            value, receipt = published_fact(
                {
                    "resource": population_resource,
                    "concept": ONS_POPULATION_CONCEPT,
                    "geography_id": geography_id,
                },
                unit="count",
                allowed_resources=allowed_resources,
                period_value=population_period,
            )
            population += value
            population_records.extend(receipt["source_record_ids"])
        resident_trips = trips_per_person * population
        k = boardings / resident_trips
        y = receipts / (boardings - concessionary)
        factor, index_receipt = 1.0, None
        if price_year is not None:
            if not isinstance(area.get("price_index"), Mapping):
                raise BusFarePricingError(
                    f"area {label!r} declares no price_index for price_year {price_year}."
                )
            if area["price_index"].get("basis") == OWN_FISCAL_YIELDS_BASIS:
                factor, index_receipt = own_fiscal_yield_price_factor(
                    area,
                    price_year=int(price_year),
                    fiscal_start=fiscal_start,
                    allowed_resources=allowed_resources,
                )
            else:
                factor, index_receipt = calendar_price_factor(
                    area["price_index"],
                    price_year=int(price_year),
                    fiscal_start=fiscal_start,
                    allowed_resources=allowed_resources,
                )
        price = AreaPrice(
            label=label,
            series=series,
            regions=regions,
            receipts=receipts,
            boardings=boardings,
            concessionary_boardings=concessionary,
            trips_per_person=trips_per_person,
            population=population,
            boardings_per_trip=k,
            yield_per_fare_paying_boarding=y,
            fare_per_trip=k * y * factor,
            concessionary_boarding_share=concessionary / boardings,
            calendar_price_factor=factor,
            trips_basis=str(area.get("trips_basis", "published_series_rate")),
            receipt={
                "label": label,
                "series": series,
                "regions": list(regions),
                "receipts": receipts_receipt,
                "boardings": boardings_receipt,
                "concessionary": concessionary_receipt,
                "trips_per_person": trips_receipt,
                "population": {
                    "geography_ids": population_ids,
                    "period_value": population_period,
                    "value": population,
                    "source_record_ids": population_records,
                },
                "published_resident_trips": resident_trips,
                "boardings_per_trip": k,
                "yield_per_fare_paying_boarding": y,
                "calendar_price_factor": factor,
                "price_index": index_receipt,
                "fare_per_trip": k * y * factor,
                "concessionary_boarding_share": concessionary / boardings,
                "trips_basis": str(area.get("trips_basis", "published_series_rate")),
            },
        )
        area_receipts.append(price.receipt)
        if series == BUS_IN_LONDON:
            if london is not None or regions:
                raise BusFarePricingError(
                    "exactly one area prices the bus_in_london series, for every "
                    "resident, and declares no regions."
                )
            london = price
        else:
            if not regions:
                raise BusFarePricingError(f"area {label!r} declares no regions.")
            for region in regions:
                if region in other_by_region or region in unpriced:
                    raise BusFarePricingError(f"region {region!r} is priced twice.")
                other_by_region[region] = price
    if london is None:
        raise BusFarePricingError("no area prices the bus_in_london series.")
    receipt = {
        "fiscal_start": fiscal_start,
        "price_year": None if price_year is None else int(price_year),
        "trip_rates_period_value": trip_rates_period,
        "population_period_value": population_period,
        "areas": area_receipts,
        "unpriced_regions": sorted(unpriced),
        "scope_translation": str(parameters.get("scope_translation", "")),
        "algebra": (
            "fare = sum over series of trips x k x y x f with k = B / (T x P), "
            "y = R / (B - C) and f the fares index's calendar over fiscal-year "
            "level; frame / R = f x [frame trips / (T x P)] x "
            "[frame fare-paying trip share / (1 - C / B)]: a survey-versus-survey "
            "composition check times a survey-versus-admin concession check, "
            "never a level the stage set."
        ),
    }
    return BusFarePrices(
        london=london,
        other_by_region=other_by_region,
        unpriced_regions=unpriced,
        receipt=receipt,
    )


def price_bus_journeys(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    prices: BusFarePrices,
    trips_columns: Mapping[str, str],
    eligibility_column: str,
    household_weights: np.ndarray,
    raw_household_fares: np.ndarray,
    region_boardings: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Household fares from persons' journeys; unpriced regions keep the raw draw.

    Returns the household fare vector, the boolean mask of priced households
    and the receipt (factors, persons priced and zeroed, the frame-implied
    boardings and eligible trip share per area, totals before and after).
    ``region_boardings`` (BUS01a journeys by English region outside London,
    keyed by FRS region) adds the per-region receipt of
    :func:`england_outside_london_region_shares`.
    """

    for series in SERIES:
        if series not in trips_columns:
            raise BusFarePricingError(f"trips_columns lacks the {series} series.")
    needed = ("person_household_id", eligibility_column, *trips_columns.values())
    missing = [c for c in needed if c not in person]
    if missing:
        raise BusFarePricingError(
            f"the person table lacks {missing}; the nts_bus_travel stage must run first."
        )
    for column in ("household_id", "region"):
        if column not in household:
            raise BusFarePricingError(f"the household table lacks {column!r}.")
    weights = np.asarray(household_weights, dtype=float)
    if len(weights) != len(household):
        raise BusFarePricingError(
            "household_weights must align with the household table."
        )
    region_boardings = dict(region_boardings or {})
    region_by_household = household.set_index("household_id")["region"].map(
        lambda v: str(getattr(v, "name", v))
    )
    hh_ids = person["person_household_id"].to_numpy()
    person_region = region_by_household.reindex(hh_ids).to_numpy().astype(str)
    weight_by_household = pd.Series(weights, index=household["household_id"].to_numpy())
    person_weight = (
        weight_by_household.reindex(hh_ids).fillna(0.0).to_numpy(dtype=float)
    )
    eligible = person[eligibility_column].to_numpy(dtype=bool)
    london_trips = person[trips_columns[BUS_IN_LONDON]].to_numpy(dtype=float)
    other_trips = person[trips_columns[OTHER_LOCAL_BUS]].to_numpy(dtype=float)
    known = set(prices.other_by_region) | set(prices.unpriced_regions)
    unknown = sorted(set(person_region) - known)
    if unknown:
        raise BusFarePricingError(
            f"regions {unknown} are neither priced nor declared unpriced."
        )
    unpriced_person = np.isin(person_region, sorted(prices.unpriced_regions))
    fare = np.zeros(len(person), dtype=float)
    per_area: dict[str, dict[str, float]] = {}

    def _accumulate(
        label: str, trips: np.ndarray, mask: np.ndarray, area: AreaPrice
    ) -> None:
        entry = per_area.setdefault(
            label,
            {
                "weighted_trips": 0.0,
                "weighted_eligible_trips": 0.0,
                "weighted_fare_paying_trips": 0.0,
                "frame_implied_boardings": 0.0,
                "published_boardings": area.boardings,
                "fares": 0.0,
                "persons": 0.0,
            },
        )
        w = person_weight[mask]
        t = trips[mask]
        e = eligible[mask]
        entry["weighted_trips"] += float(np.dot(t, w))
        entry["weighted_eligible_trips"] += float(np.dot(t[e], w[e]))
        entry["weighted_fare_paying_trips"] += float(np.dot(t[~e], w[~e]))
        entry["frame_implied_boardings"] += (
            float(np.dot(t, w)) * area.boardings_per_trip
        )
        entry["persons"] += float(mask.sum())
        priced = np.where(e, 0.0, t * area.fare_per_trip)
        entry["fares"] += float(np.dot(priced, w))
        fare[mask] += priced

    _accumulate("london_series", london_trips, ~unpriced_person, prices.london)
    for region, area in prices.other_by_region.items():
        mask = person_region == region
        if mask.any():
            _accumulate(area.label, other_trips, mask, area)
    household_fare = (
        pd.Series(fare, index=hh_ids)
        .groupby(level=0)
        .sum()
        .reindex(household["household_id"].to_numpy())
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    household_region = (
        household["region"].map(lambda v: str(getattr(v, "name", v))).to_numpy()
    )
    priced_household = ~np.isin(household_region, sorted(prices.unpriced_regions))
    raw = np.asarray(raw_household_fares, dtype=float)
    result = np.where(priced_household, household_fare, raw)
    for entry in per_area.values():
        entry["frame_implied_over_published_boardings"] = (
            entry["frame_implied_boardings"] / entry["published_boardings"]
            if entry["published_boardings"] > 0
            else None
        )
        entry["frame_eligible_trip_share"] = (
            entry["weighted_eligible_trips"] / entry["weighted_trips"]
            if entry["weighted_trips"] > 0
            else None
        )
    receipt = {
        "persons": int(len(person)),
        "persons_priced": int((~unpriced_person).sum()),
        "persons_eligible_zero_priced": int((eligible & ~unpriced_person).sum()),
        "households_priced": int(priced_household.sum()),
        "households_unpriced_keep_raw_draw": int((~priced_household).sum()),
        "unpriced_regions": sorted(prices.unpriced_regions),
        "by_area": per_area,
        "weighted_fares_before": float(
            np.dot(raw[priced_household], weights[priced_household])
        ),
        "weighted_fares_after": float(
            np.dot(result[priced_household], weights[priced_household])
        ),
        "prices": {
            "london_series": prices.london.receipt,
            **{a.label: a.receipt for a in prices.other_by_region.values()},
        },
    }
    if region_boardings:
        receipt["england_outside_london_regions"] = (
            england_outside_london_region_shares(
                region_boardings,
                person_region=person_region,
                person_weight=person_weight,
                other_trips=other_trips,
            )
        )
    return result, priced_household, receipt


def english_region_boardings(
    parameters: Mapping[str, Any], *, allowed_resources: Sequence[str]
) -> dict[str, float]:
    """BUS01a journeys of each English region outside London, keyed by FRS region.

    Read at the declaration's ``fiscal_start`` from the same vendored rows the
    London and England-outside-London boardings come from (microcosm#1207).
    """

    from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE

    fiscal_start = str(parameters.get("fiscal_start") or "")
    boardings: dict[str, float] = {}
    for region, code in FRS_REGION_TO_REGION_CODE.items():
        if not code.startswith("E12") or region == "LONDON":
            continue
        boardings[region], _ = published_fact(
            {
                "resource": UK_DFT_BUS_JOURNEYS_RESOURCE,
                "concept": BUS01_JOURNEYS_CONCEPT,
                "geography_id": code,
            },
            unit="count",
            allowed_resources=allowed_resources,
            fiscal_start=fiscal_start,
        )
    return boardings


def england_outside_london_region_shares(
    region_boardings: Mapping[str, float],
    *,
    person_region: np.ndarray,
    person_weight: np.ndarray,
    other_trips: np.ndarray,
) -> dict[str, Any]:
    """Each region's share of the frame's other local bus trips beside BUS01a's.

    Report only (microcosm#1207): one boardings-per-trip factor prices every
    English region's other local bus trips, so the frame's share of the eight
    regions' trips is its share of their boardings. Set beside BUS01a's share
    of the eight regions' journeys, a region far from one is a gap the
    calibration's regional journey rows will close through the weights:
    interchange, cross-boundary travel and NTS coverage as much as who
    travels. London residents' other local bus trips count toward England
    outside London and fall in no region row; their share is recorded.
    """

    regions = sorted(region_boardings)
    frame = {
        region: float(
            np.dot(
                other_trips[person_region == region],
                person_weight[person_region == region],
            )
        )
        for region in regions
    }
    london = float(
        np.dot(
            other_trips[person_region == "LONDON"],
            person_weight[person_region == "LONDON"],
        )
    )
    frame_total = float(sum(frame.values()))
    published_total = float(sum(region_boardings.values()))
    by_region: dict[str, dict[str, float | None]] = {}
    for region in regions:
        frame_share = frame[region] / frame_total if frame_total > 0 else None
        published_share = float(region_boardings[region]) / published_total
        by_region[region] = {
            "frame_share": frame_share,
            "published_share": published_share,
            "frame_over_published_share": (
                frame_share / published_share if frame_share is not None else None
            ),
        }
    return {
        "report_only": True,
        "basis": (
            "prior weights; each region's residents' other local bus trips at "
            "England outside London's one boardings-per-trip factor, as a share "
            "of the eight regions', beside BUS01a's journeys as a share of theirs"
        ),
        "regions": by_region,
        "london_residents_share_of_other_local_trips": (
            london / (frame_total + london) if frame_total + london > 0 else None
        ),
    }


#: The BUS01a journey rows' period guard (microcosm#1207, review round 1). The
#: rows bind journeys of the fiscal year the lcfs stage prices boardings per
#: trip from and are held there (matched period). A row resolved to another
#: year, a later edition in the feed, would bind journeys the stage's factors
#: do not describe, so it is refused until the stage's ``fiscal_start`` moves
#: with it.
UK_BUS_JOURNEYS_YEAR_INDEX = "dft.bus01.stage_pricing_year"
UK_BUS_JOURNEYS_HOLD = "matched_period"
UK_BUS_JOURNEYS_YEAR_BASIS = (
    "DfT BUS01a passenger journeys of the fiscal year opening {fiscal_start}, "
    "the year the lcfs stage prices boardings per trip from; held there "
    "(factor one, matched period)"
)


def declared_bus_pricing() -> Mapping[str, Any]:
    """The lcfs stage's declared ``price_bus_journeys`` parameters."""

    from microcosm.build.country_spec import load_country_spec

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    for operation in stage.operations:
        if operation.kind == PRICE_BUS_JOURNEYS_KIND:
            return dict(operation.parameters)
    raise BusFarePricingError("the lcfs stage declares no price_bus_journeys step.")


def hold_bus_journeys_at_the_stage_year(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> TargetRegistry:
    """Hold each BUS01a journey row at the stage's pricing year, or refuse it."""

    if reference.uprating_index != UK_BUS_JOURNEYS_YEAR_INDEX:
        return registry
    parameters = declared_bus_pricing() if parameters is None else parameters
    fiscal_start = str(parameters.get("fiscal_start") or "")
    year = fiscal_start[:4]
    if not year.isdigit() or fiscal_start[4:] != "-04-01":
        raise BusFarePricingError(
            f"price_bus_journeys declares fiscal_start {fiscal_start!r}, not an "
            "April 1st."
        )
    basis = UK_BUS_JOURNEYS_YEAR_BASIS.format(fiscal_start=fiscal_start)
    held = []
    for spec in registry.specs:
        fact_period = str(spec.metadata.get("ledger_fact_period") or "")
        if fact_period != year:
            raise ValueError(
                f"UK target {spec.name!r} resolved BUS01a journeys of the fiscal "
                f"year opening in {fact_period or '?'}, but the lcfs stage prices "
                f"boardings per trip from the year opening {fiscal_start}: move "
                "the stage's fiscal_start and the hold together."
            )
        metadata = {
            **spec.metadata,
            "uprating_index": UK_BUS_JOURNEYS_YEAR_INDEX,
            "uprating_index_basis": basis,
            "uprating_index_resource": UK_DFT_BUS_JOURNEYS_RESOURCE,
            "uprating_factor": "1",
            "ledger_value_before_alignment": f"{spec.value:.15g}",
            "uprating_hold": UK_BUS_JOURNEYS_HOLD,
        }
        held.append(replace(spec, metadata=metadata))
    return TargetRegistry(held, country="uk")


def bus_journey_appliers() -> dict[str, Any]:
    """The appliers this module contributes to ``UK_UPRATING_APPLIERS``."""

    return {UK_BUS_JOURNEYS_YEAR_INDEX: hold_bus_journeys_at_the_stage_year}
