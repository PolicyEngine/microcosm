"""Bus fares priced from journeys at the published yield (microcosm#930 C6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.bus_fare_pricing import (
    BUS_IN_LONDON,
    OTHER_LOCAL_BUS,
    PRICE_BUS_JOURNEYS_KIND,
    BusFarePricingError,
    bus_fare_prices,
    price_bus_journeys,
    published_fact,
)
from microcosm.build.uk_runtime.lcfs_consumption import (
    UK_LCFS_VENDORED_RESOURCES,
    bus_pricing_operation,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows


def _declared() -> dict:
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    parameters = bus_pricing_operation(stage)
    assert parameters is not None and parameters["kind"] == PRICE_BUS_JOURNEYS_KIND
    return parameters


def test_prices_recompute_from_the_vendored_rows() -> None:
    prices = bus_fare_prices(_declared(), allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    london = prices.london
    assert london.series == BUS_IN_LONDON and london.regions == ()
    receipts = float(
        vendored_rows(
            "dft_bus_value_anchors.json",
            concept="dft.local_bus_passenger_fare_receipts",
            fiscal_start="2024-04-01",
            geography_id="E12000007",
        )[0]["value"]
    )
    journeys = [
        r
        for r in vendored_rows(
            "dft_bus_journeys.json",
            concept="dft.local_bus_passenger_journeys",
            fiscal_start="2024-04-01",
            geography_id="E12000007",
        )
    ]
    total = float(next(r["value"] for r in journeys if not r["dimensions"]))
    concessionary = float(
        next(
            r["value"]
            for r in journeys
            if r["dimensions"].get("journey_category") == "total_concessionary"
        )
    )
    assert london.receipts == pytest.approx(receipts)
    assert london.boardings == pytest.approx(total)
    assert london.concessionary_boardings == pytest.approx(concessionary)
    assert london.yield_per_fare_paying_boarding == pytest.approx(
        receipts / (total - concessionary)
    )
    trips = float(
        vendored_rows(
            "dft_bus_value_anchors.json",
            concept="dft.bus_in_london_trips_per_person",
            period_value=2024,
            groupby_value_id="all",
        )[0]["value"]
    )
    population = sum(
        float(r["value"])
        for r in vendored_rows(
            "dft_bus_journeys.json",
            concept="ons.mid_year_population_estimate",
            period_value=2024,
        )
        if str(r["geography"]["id"]).startswith("E12")
    )
    assert london.trips_per_person == pytest.approx(trips)
    assert london.population == pytest.approx(population)
    assert london.boardings_per_trip == pytest.approx(total / (trips * population))
    assert london.fare_per_trip == pytest.approx(
        london.boardings_per_trip * london.yield_per_fare_paying_boarding
    )
    # Order of magnitude: about a pound per fare-paying boarding, two boardings
    # per resident trip, a fare per resident trip between one and five pounds.
    assert 0.5 < london.yield_per_fare_paying_boarding < 2.0
    assert 1.0 < london.boardings_per_trip < 4.0
    assert 1.0 < london.fare_per_trip < 5.0
    assert set(prices.other_by_region) == {
        "NORTH_EAST",
        "NORTH_WEST",
        "YORKSHIRE",
        "EAST_MIDLANDS",
        "WEST_MIDLANDS",
        "EAST_OF_ENGLAND",
        "LONDON",
        "SOUTH_EAST",
        "SOUTH_WEST",
        "SCOTLAND",
        "NORTHERN_IRELAND",
    }
    assert prices.unpriced_regions == frozenset({"WALES"})
    scotland = prices.other_by_region["SCOTLAND"]
    assert scotland.trips_basis == "england_other_local_bus_rate_as_proxy"
    assert scotland.receipts == pytest.approx(391_000_000.0)
    ni = prices.other_by_region["NORTHERN_IRELAND"]
    assert ni.receipts == pytest.approx(49_584_434.28 + 100_498_383.21, rel=1e-6)
    assert ni.concessionary_boardings == pytest.approx(8_960_000.0)
    assert "composition check" in prices.receipt["algebra"]


def test_published_fact_refuses_the_wrong_resource_unit_or_split() -> None:
    selector = {
        "resource": "dft_bus_journeys.json",
        "concept": "dft.local_bus_passenger_journeys",
        "geography_id": "E12000007",
    }
    value, receipt = published_fact(
        selector,
        unit="count",
        allowed_resources=UK_LCFS_VENDORED_RESOURCES,
        fiscal_start="2024-04-01",
    )
    assert value > 1e9 and receipt["unit"] == "count"
    with pytest.raises(BusFarePricingError, match="not a resource the stage declares"):
        published_fact(
            {**selector, "resource": "orr_rail_facts.json"},
            unit="count",
            allowed_resources=UK_LCFS_VENDORED_RESOURCES,
            fiscal_start="2024-04-01",
        )
    with pytest.raises(BusFarePricingError, match="expected 'gbp' rows"):
        published_fact(
            selector,
            unit="gbp",
            allowed_resources=UK_LCFS_VENDORED_RESOURCES,
            fiscal_start="2024-04-01",
        )
    with pytest.raises(BusFarePricingError, match="sum_over"):
        published_fact(
            {
                "resource": "devolved_bus_finance.json",
                "concept": "dfi_ni.translink_bus.passenger_receipts",
                "sum_over": {"service": ["metro_glider", "ulsterbus", "rail"]},
            },
            unit="gbp",
            allowed_resources=UK_LCFS_VENDORED_RESOURCES,
            fiscal_start="2024-04-01",
        )


def test_declaration_refusals() -> None:
    declared = _declared()
    duplicated = {**declared, "areas": [*declared["areas"], declared["areas"][2]]}
    with pytest.raises(BusFarePricingError, match="priced twice"):
        bus_fare_prices(duplicated, allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    no_london = {**declared, "areas": declared["areas"][1:]}
    with pytest.raises(BusFarePricingError, match="bus_in_london"):
        bus_fare_prices(no_london, allowed_resources=UK_LCFS_VENDORED_RESOURCES)


def test_household_fares_sum_non_eligible_persons_and_keep_wales_raw() -> None:
    prices = bus_fare_prices(_declared(), allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "region": ["LONDON", "SOUTH_EAST", "WALES", "SCOTLAND"],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 12, 21, 31, 41, 42],
            "person_household_id": [1, 1, 2, 3, 4, 4],
            "bus_in_london_trips": [100.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            "other_local_bus_trips": [0.0, 10.0, 40.0, 30.0, 20.0, 20.0],
            "bus_pass_eligible": [False, True, False, False, False, True],
        }
    )
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    raw = np.array([500.0, 600.0, 700.0, 800.0])
    fares, priced, receipt = price_bus_journeys(
        person,
        household,
        prices=prices,
        trips_columns={
            BUS_IN_LONDON: "bus_in_london_trips",
            OTHER_LOCAL_BUS: "other_local_bus_trips",
        },
        eligibility_column="bus_pass_eligible",
        household_weights=weights,
        raw_household_fares=raw,
    )
    london = prices.london.fare_per_trip
    england = prices.other_by_region["SOUTH_EAST"].fare_per_trip
    scotland = prices.other_by_region["SCOTLAND"].fare_per_trip
    # Household 1: person 11 pays 100 London trips; person 12 is eligible.
    assert fares[0] == pytest.approx(100.0 * london)
    assert fares[1] == pytest.approx(40.0 * england)
    assert fares[2] == 700.0 and not priced[2]
    assert fares[3] == pytest.approx(20.0 * scotland)
    assert priced.tolist() == [True, True, False, True]
    assert receipt["households_unpriced_keep_raw_draw"] == 1
    assert receipt["persons_eligible_zero_priced"] == 2
    assert receipt["unpriced_regions"] == ["WALES"]
    area = receipt["by_area"]["london_series"]
    assert area["weighted_trips"] == pytest.approx(150.0)
    assert area["weighted_eligible_trips"] == pytest.approx(50.0)
    assert area["frame_eligible_trip_share"] == pytest.approx(50.0 / 150.0)
    assert area["frame_implied_boardings"] == pytest.approx(
        150.0 * prices.london.boardings_per_trip
    )
    assert receipt["weighted_fares_before"] == pytest.approx(500 + 2 * 600 + 4 * 800)
    assert receipt["weighted_fares_after"] == pytest.approx(
        fares[0] + 2 * fares[1] + 4 * fares[3]
    )
    with pytest.raises(
        BusFarePricingError, match="neither priced nor declared unpriced"
    ):
        price_bus_journeys(
            person,
            household.assign(region=["LONDON", "ATLANTIS", "WALES", "SCOTLAND"]),
            prices=prices,
            trips_columns={
                BUS_IN_LONDON: "bus_in_london_trips",
                OTHER_LOCAL_BUS: "other_local_bus_trips",
            },
            eligibility_column="bus_pass_eligible",
            household_weights=weights,
            raw_household_fares=raw,
        )
    with pytest.raises(
        BusFarePricingError, match="nts_bus_travel stage must run first"
    ):
        price_bus_journeys(
            person.drop(columns=["bus_pass_eligible"]),
            household,
            prices=prices,
            trips_columns={
                BUS_IN_LONDON: "bus_in_london_trips",
                OTHER_LOCAL_BUS: "other_local_bus_trips",
            },
            eligibility_column="bus_pass_eligible",
            household_weights=weights,
            raw_household_fares=raw,
        )
