"""Bus support priced from the frame's journeys at the published support per boarding.

The ETB stage used to rake ``bus_subsidy_spending`` to the published net
support per area while calibration bound the same facts, the rake-then-bind
shape microcosm#930 retired for fares; María's ruling of 2026-09-23 retires it
for support too. Each priced area's household support is the published
concessionary travel reimbursement per concessionary boarding times the
household's eligible persons' boardings, plus the rest of the published net
support per boarding times every boarding of the household; boardings are the
``nts_bus_travel`` trips times the fare pricing's boardings-per-resident-trip
translation (BUS01 over NTS0705a times population), so fares and support
share one quantity. Areas the publisher gives no components or boardings for
(Wales, Northern Ireland: the NITHC accounts carry bus and rail jointly and
no Welsh boardings are published) keep the ETB chain's raw draw, clipped to
donor support, as Wales fares do. Every input is a vendored Chronicle row (the
BUS01 journeys, concessionary journeys and BUS05 components in
``dft_bus_journeys.json``, the net support in ``dft_bus_value_anchors.json``,
the Scottish components and journeys in ``devolved_bus_finance.json``); the
published totals stay bound as calibration targets, which now measure the
priced quantity instead of a level the stage set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.bus_fare_pricing import (
    BUS_IN_LONDON,
    OTHER_LOCAL_BUS,
    BusFarePricingError,
    bus_fare_prices,
    published_fact,
)

PRICE_BUS_SUPPORT_KIND = "price_bus_support"


def bus_support_pricing_operation(stage: Any) -> Mapping[str, Any] | None:
    """The stage's declared ``price_bus_support`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == PRICE_BUS_SUPPORT_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def _pricing_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "price_bus_journeys",
        "column": "bus_fare_spending",
        "trips_columns": dict(parameters["trips_columns"]),
        "eligibility_column": str(parameters["eligibility_column"]),
        "fiscal_start": str(parameters["fiscal_start"]),
        "trip_rates_resource": str(parameters["trip_rates_resource"]),
        "trip_rates_period_value": int(parameters["trip_rates_period_value"]),
        "population_resource": str(parameters["population_resource"]),
        "population_period_value": int(parameters["population_period_value"]),
        "areas": list(parameters["pricing_areas"]),
        "unpriced_regions": list(parameters.get("unpriced_regions", ())),
    }


def bus_support_factors(
    parameters: Mapping[str, Any], *, allowed_resources: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Every declared support area's published components and per-boarding rates.

    Recomputed from the vendored rows by the stage and by its gate alike, so
    the receipt can never carry a factor the rows do not give.
    """

    fiscal_start = str(parameters["fiscal_start"])
    factors: dict[str, dict[str, Any]] = {}
    for area in parameters["support_areas"]:
        area = dict(area)
        label = str(area["label"])
        values: dict[str, float] = {}
        receipts: dict[str, Any] = {}
        for key, unit in (
            ("reimbursement", "gbp"),
            ("net_support", "gbp"),
            ("boardings", "count"),
            ("concessionary", "count"),
        ):
            values[key], receipts[key] = published_fact(
                dict(area[key]),
                unit=unit,
                allowed_resources=allowed_resources,
                fiscal_start=fiscal_start,
            )
        if values["net_support"] < values["reimbursement"]:
            raise BusFarePricingError(
                f"area {label!r}: net support {values['net_support']} is below the "
                f"concessionary reimbursement {values['reimbursement']} it contains."
            )
        if values["concessionary"] <= 0 or values["boardings"] <= 0:
            raise BusFarePricingError(f"area {label!r}: boardings must be positive.")
        factors[label] = {
            "regions": [str(r) for r in area["regions"]],
            **receipts,
            "reimbursement_per_concessionary_boarding": (
                values["reimbursement"] / values["concessionary"]
            ),
            "other_support_per_boarding": (
                (values["net_support"] - values["reimbursement"]) / values["boardings"]
            ),
            "published_concessionary_boarding_share": (
                values["concessionary"] / values["boardings"]
            ),
        }
    return factors


def price_bus_support(
    parameters: Mapping[str, Any],
    *,
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    raw_support: np.ndarray,
    allowed_resources: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Household bus support: priced per declared area, the raw draw elsewhere.

    Returns the household values, the mask of priced households and the
    receipt. A region that is neither in a support area nor declared a
    ``raw_draw_regions`` member is refused.
    """

    if parameters.get("kind", PRICE_BUS_SUPPORT_KIND) != PRICE_BUS_SUPPORT_KIND:
        raise BusFarePricingError("not a price_bus_support declaration.")
    prices = bus_fare_prices(
        _pricing_parameters(parameters), allowed_resources=allowed_resources
    )
    factors = bus_support_factors(parameters, allowed_resources=allowed_resources)
    raw_draw_regions = frozenset(str(r) for r in parameters.get("raw_draw_regions", ()))
    trips_columns = {
        str(k): str(v) for k, v in dict(parameters["trips_columns"]).items()
    }
    eligibility = str(parameters["eligibility_column"])
    support_column = str(parameters["support_column"])
    for column in ("person_household_id", eligibility, *trips_columns.values()):
        if column not in person:
            raise BusFarePricingError(
                f"the person table lacks {column!r}; the nts_bus_travel stage must run first."
            )
    weights = np.asarray(household_weights, dtype=float)
    region_by_household = household.set_index("household_id")["region"].map(
        lambda v: str(getattr(v, "name", v))
    )
    hh_ids = person["person_household_id"].to_numpy()
    person_region = region_by_household.reindex(hh_ids).to_numpy().astype(str)
    person_weight = (
        pd.Series(weights, index=household["household_id"].to_numpy())
        .reindex(hh_ids)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    eligible = person[eligibility].to_numpy(dtype=bool)
    london_trips = person[trips_columns[BUS_IN_LONDON]].to_numpy(dtype=float)
    other_trips = person[trips_columns[OTHER_LOCAL_BUS]].to_numpy(dtype=float)
    k_london = prices.london.boardings_per_trip
    household_region = (
        household["region"].map(lambda v: str(getattr(v, "name", v))).to_numpy()
    )
    raw = np.asarray(raw_support, dtype=float)
    values = raw.copy()
    priced_household = np.zeros(len(household), dtype=bool)
    covered = set(raw_draw_regions)
    for block in factors.values():
        covered |= set(block["regions"])
    unknown = sorted(set(household_region) - covered)
    if unknown:
        raise BusFarePricingError(
            f"household regions {unknown} are neither in a support area nor declared "
            "raw_draw_regions."
        )
    overlap = raw_draw_regions & {r for b in factors.values() for r in b["regions"]}
    if overlap:
        raise BusFarePricingError(
            f"regions {sorted(overlap)} are both priced and declared raw_draw_regions."
        )
    hh_index = pd.Series(np.arange(len(household)), index=household["household_id"])
    person_position = hh_index.reindex(hh_ids).to_numpy()
    by_area: dict[str, dict[str, Any]] = {}
    for label, block in factors.items():
        regions = tuple(block["regions"])
        mask = np.isin(person_region, regions)
        k_other = np.zeros(len(person), dtype=float)
        for region in regions:
            price = prices.area_for_region(region)
            if price is None:
                raise BusFarePricingError(
                    f"area {label!r} names {region!r}, which the pricing areas do not price."
                )
            k_other[person_region == region] = price.boardings_per_trip
        person_boardings = london_trips * k_london + other_trips * k_other
        person_support = block["other_support_per_boarding"] * person_boardings
        person_support = person_support + np.where(
            eligible,
            block["reimbursement_per_concessionary_boarding"] * person_boardings,
            0.0,
        )
        household_mask = np.isin(household_region, regions)
        summed = np.zeros(len(household), dtype=float)
        np.add.at(summed, person_position[mask], person_support[mask])
        values[household_mask] = summed[household_mask]
        priced_household |= household_mask
        all_boardings = float(np.dot(person_boardings[mask], person_weight[mask]))
        eligible_boardings = float(
            np.dot(person_boardings[mask & eligible], person_weight[mask & eligible])
        )
        priced_total = float(np.dot(values[household_mask], weights[household_mask]))
        raw_total = float(np.dot(raw[household_mask], weights[household_mask]))
        by_area[label] = {
            **block,
            "frame_boardings": all_boardings,
            "frame_eligible_boardings": eligible_boardings,
            "frame_eligible_boarding_share": (
                eligible_boardings / all_boardings if all_boardings > 0 else None
            ),
            "weighted_support_before": raw_total,
            "weighted_support_after": priced_total,
            "priced_over_published": priced_total / block["net_support"]["value"],
            "households": int(household_mask.sum()),
            "persons": int(mask.sum()),
        }
    raw_mask = np.isin(household_region, sorted(raw_draw_regions))
    receipt = {
        "operation": PRICE_BUS_SUPPORT_KIND,
        "support_column": support_column,
        "applied": True,
        "chain_conditioned_on": "raw_draw",
        "basis": (
            "reimbursement per concessionary boarding times the household's eligible "
            "persons' boardings plus the rest of the net support per boarding times "
            "every boarding; boardings are nts_bus_travel trips times the fare "
            "pricing's boardings per resident trip; regions without published "
            "components or boardings keep the clipped raw draw"
        ),
        "boardings_per_trip": {
            "bus_in_london": k_london,
            **{
                a.label: a.boardings_per_trip
                for a in {a.label: a for a in prices.other_by_region.values()}.values()
            },
        },
        "raw_draw_regions": sorted(raw_draw_regions),
        "households_priced": int(priced_household.sum()),
        "households_raw_draw": int(raw_mask.sum()),
        "weighted_support_before": float(np.dot(raw, weights)),
        "weighted_support_after": float(np.dot(values, weights)),
        "raw_draw_weighted_support": {
            region: float(
                np.dot(
                    raw[household_region == region], weights[household_region == region]
                )
            )
            for region in sorted(raw_draw_regions)
        },
        "by_area": by_area,
    }
    return values, priced_household, receipt
