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

Every input is a vendored Chronicle row; the receipt records every factor, the
frame-implied boardings beside the published ones, and the frame's eligible
trip share beside the publisher's concessionary boarding share, so the
design-weight fit of the bound receipts targets can be read as what it is: a
survey-versus-survey composition check times a survey-versus-admin concession
check, never a level the stage set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

PRICE_BUS_JOURNEYS_KIND = "price_bus_journeys"
UK_DFT_BUS_JOURNEYS_RESOURCE = "dft_bus_journeys.json"
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
            fare_per_trip=k * y,
            concessionary_boarding_share=concessionary / boardings,
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
                "fare_per_trip": k * y,
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
        "trip_rates_period_value": trip_rates_period,
        "population_period_value": population_period,
        "areas": area_receipts,
        "unpriced_regions": sorted(unpriced),
        "scope_translation": str(parameters.get("scope_translation", "")),
        "algebra": (
            "fare = sum over series of trips x k x y with k = B / (T x P) and "
            "y = R / (B - C); frame / R = [frame trips / (T x P)] x "
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Household fares from persons' journeys; unpriced regions keep the raw draw.

    Returns the household fare vector, the boolean mask of priced households
    and the receipt (factors, persons priced and zeroed, the frame-implied
    boardings and eligible trip share per area, totals before and after).
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
    return result, priced_household, receipt
