"""What a value-side treatment of bus support would give, recorded not applied.

The ETB stage rakes ``bus_subsidy_spending`` to the published net support per
area and calibration binds the same facts (the rake-then-bind shape microcosm#930
retires for fares). This diagnostic records, per residence area, what the
support would be if it were priced from the frame's journeys the way fares now
are: the published concessionary travel reimbursement per concessionary
boarding times the frame's eligible boardings, plus the rest of the net
support per boarding times every boarding, beside the raked column's
design-weighted total. Boardings are the ``nts_bus_travel`` trips times the
boardings-per-resident-trip translation of the fare pricing (the same
published BUS01 over NTS0705a times population), so the two columns share
one quantity. Every input is a vendored Chronicle row (the BUS01 journeys,
concessionary journeys and BUS05 reimbursement in ``dft_bus_journeys.json``,
the net support in ``dft_bus_value_anchors.json``); nothing here changes a
value — the switch is María's ruling with these numbers in hand.
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

RECORD_SUPPORT_PER_JOURNEY_KIND = "record_support_per_journey"


def support_per_journey_diagnostic(
    parameters: Mapping[str, Any],
    *,
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    raked_support: np.ndarray,
    allowed_resources: Sequence[str],
) -> dict[str, Any]:
    """The receipt of the value-side alternative, per declared residence area."""

    if parameters.get("kind", RECORD_SUPPORT_PER_JOURNEY_KIND) != (
        RECORD_SUPPORT_PER_JOURNEY_KIND
    ):
        raise BusFarePricingError("not a record_support_per_journey declaration.")
    pricing = {
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
    prices = bus_fare_prices(pricing, allowed_resources=allowed_resources)
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
    raked = np.asarray(raked_support, dtype=float)
    by_area: dict[str, dict[str, Any]] = {}
    for area in parameters["support_areas"]:
        area = dict(area)
        label = str(area["label"])
        regions = tuple(str(r) for r in area["regions"])
        fiscal_start = str(parameters["fiscal_start"])
        reimbursement, reimbursement_receipt = published_fact(
            dict(area["reimbursement"]),
            unit="gbp",
            allowed_resources=allowed_resources,
            fiscal_start=fiscal_start,
        )
        net_support, net_support_receipt = published_fact(
            dict(area["net_support"]),
            unit="gbp",
            allowed_resources=allowed_resources,
            fiscal_start=fiscal_start,
        )
        boardings, boardings_receipt = published_fact(
            dict(area["boardings"]),
            unit="count",
            allowed_resources=allowed_resources,
            fiscal_start=fiscal_start,
        )
        concessionary, concessionary_receipt = published_fact(
            dict(area["concessionary"]),
            unit="count",
            allowed_resources=allowed_resources,
            fiscal_start=fiscal_start,
        )
        if net_support < reimbursement:
            raise BusFarePricingError(
                f"area {label!r}: net support {net_support} is below the concessionary "
                f"reimbursement {reimbursement} it contains."
            )
        reimbursement_per_concessionary_boarding = reimbursement / concessionary
        other_support_per_boarding = (net_support - reimbursement) / boardings
        mask = np.isin(person_region, regions)
        # Boardings of the area's residents: each series at its own translation.
        k_other = np.zeros(len(person), dtype=float)
        for region in regions:
            price = prices.area_for_region(region)
            if price is None:
                raise BusFarePricingError(
                    f"area {label!r} names {region!r}, which the pricing areas do not price."
                )
            k_other[person_region == region] = price.boardings_per_trip
        person_boardings = london_trips * k_london + other_trips * k_other
        all_boardings = float(np.dot(person_boardings[mask], person_weight[mask]))
        eligible_boardings = float(
            np.dot(person_boardings[mask & eligible], person_weight[mask & eligible])
        )
        value_side = (
            reimbursement_per_concessionary_boarding * eligible_boardings
            + other_support_per_boarding * all_boardings
        )
        household_mask = np.isin(household_region, regions)
        raked_total = float(np.dot(raked[household_mask], weights[household_mask]))
        by_area[label] = {
            "regions": list(regions),
            "reimbursement": reimbursement_receipt,
            "net_support": net_support_receipt,
            "boardings": boardings_receipt,
            "concessionary": concessionary_receipt,
            "reimbursement_per_concessionary_boarding": reimbursement_per_concessionary_boarding,
            "other_support_per_boarding": other_support_per_boarding,
            "frame_boardings": all_boardings,
            "frame_eligible_boardings": eligible_boardings,
            "frame_eligible_boarding_share": (
                eligible_boardings / all_boardings if all_boardings > 0 else None
            ),
            "published_concessionary_boarding_share": concessionary / boardings,
            "value_side_support": value_side,
            "raked_support": raked_total,
            "value_side_over_raked": (
                value_side / raked_total if raked_total > 0 else None
            ),
            "households": int(household_mask.sum()),
            "persons": int(mask.sum()),
        }
    return {
        "operation": RECORD_SUPPORT_PER_JOURNEY_KIND,
        "support_column": support_column,
        "applied": False,
        "basis": (
            "reimbursement per concessionary boarding times the frame's eligible "
            "boardings plus the rest of the net support per boarding times every "
            "boarding; boardings are nts_bus_travel trips times the fare pricing's "
            "boardings per resident trip"
        ),
        "boardings_per_trip": {
            "bus_in_london": k_london,
            **{
                a.label: a.boardings_per_trip
                for a in {a.label: a for a in prices.other_by_region.values()}.values()
            },
        },
        "by_area": by_area,
    }
