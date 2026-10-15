"""Bus fares priced from journeys at the published yield (microcosm#930 C6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.bus_fare_pricing import (
    BUS_IN_LONDON,
    OTHER_LOCAL_BUS,
    PRICE_BUS_JOURNEYS_KIND,
    UK_BUS_JOURNEYS_HOLD,
    UK_BUS_JOURNEYS_YEAR_INDEX,
    BusFarePricingError,
    bus_fare_prices,
    england_outside_london_region_shares,
    english_region_boardings,
    hold_bus_journeys_at_the_stage_year,
    price_bus_journeys,
    published_fact,
)
from microcosm.build.uk_runtime.lcfs_consumption import (
    UK_LCFS_VENDORED_RESOURCES,
    bus_pricing_operation,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.ledger_targets import UK_UPRATING_APPLIERS
from microcosm.build.uk_runtime.uprating_holds import is_uprating_hold
from microcosm.calibrate import TargetRegistry, TargetSpec


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
    # London fares held flat through 2024-25, so the calendar re-pricing is 1.
    assert london.calendar_price_factor == pytest.approx(1.0)
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
    # DfI's 2025-26 edition restates 2024-25 to the £0.1m (microcosm#1123 re-pin).
    assert ni.receipts == pytest.approx(49_600_000.0 + 100_500_000.0, rel=1e-6)
    assert ni.concessionary_boardings == pytest.approx(8_960_000.0)
    assert "composition check" in prices.receipt["algebra"]
    # Every area's fiscal-year yield is re-priced to calendar 2024 (#1113).
    assert prices.receipt["price_year"] == 2024
    for area in prices.other_by_region.values():
        assert area.fare_per_trip == pytest.approx(
            area.boardings_per_trip
            * area.yield_per_fare_paying_boarding
            * area.calendar_price_factor
        )
    assert ni.receipt["price_index"]["basis"] == "own_fiscal_yields"


def _ni_fiscal_yield(fiscal_start: str) -> float:
    def total(concept: str) -> float:
        return sum(
            float(row["value"])
            for row in vendored_rows(
                "devolved_bus_finance.json", concept=concept, fiscal_start=fiscal_start
            )
        )

    return total("dfi_ni.translink_bus.passenger_receipts") / (
        total("dfi_ni.translink_bus.passenger_journeys")
        - total("dfi_ni.translink_bus.full_fare_concession_passenger_journeys")
    )


def test_northern_ireland_reprices_from_its_own_yields() -> None:
    """NI has no fares index: three months of 2023-24's yield, nine of 2024-25's (#1207)."""

    prices = bus_fare_prices(_declared(), allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    ni = prices.other_by_region["NORTHERN_IRELAND"]
    earlier = _ni_fiscal_yield("2023-04-01")
    later = _ni_fiscal_yield("2024-04-01")
    assert ni.yield_per_fare_paying_boarding == pytest.approx(later)
    assert ni.calendar_price_factor == pytest.approx(
        (0.25 * earlier + 0.75 * later) / later
    )
    years = ni.receipt["price_index"]["fiscal_years"]
    assert {start: year["weight"] for start, year in years.items()} == {
        "2023-04-01": 0.25,
        "2024-04-01": 0.75,
    }
    assert all(year["source_record_ids"] for year in years.values())


def test_own_fiscal_yields_refuses_extra_keys_and_a_mismatched_year() -> None:
    from microcosm.build.uk_runtime.bus_fare_pricing import (
        own_fiscal_yield_price_factor,
    )

    declared = _declared()
    ni = next(area for area in declared["areas"] if area["label"] == "northern_ireland")
    with pytest.raises(BusFarePricingError, match="declares only its basis"):
        own_fiscal_yield_price_factor(
            {
                **ni,
                "price_index": {"basis": "own_fiscal_yields", "frequency": "annual"},
            },
            price_year=2024,
            fiscal_start="2024-04-01",
            allowed_resources=UK_LCFS_VENDORED_RESOURCES,
        )
    with pytest.raises(BusFarePricingError, match="does not open in 2025"):
        own_fiscal_yield_price_factor(
            ni,
            price_year=2025,
            fiscal_start="2024-04-01",
            allowed_resources=UK_LCFS_VENDORED_RESOURCES,
        )


def test_calendar_price_factor_reprices_a_fiscal_year_yield() -> None:
    """Quarterly: calendar over fiscal mean; annual: three quarters of Y (#1113)."""

    from microcosm.build.uk_runtime.bus_fare_pricing import calendar_price_factor

    quarterly = {
        "resource": "dft_bus_value_anchors.json",
        "concept": "dft.local_bus_fares_index",
        "geography_id": "dft:england_outside_london",
        "frequency": "quarterly",
    }
    factor, receipt = calendar_price_factor(
        quarterly,
        price_year=2024,
        fiscal_start="2024-04-01",
        allowed_resources=UK_LCFS_VENDORED_RESOURCES,
    )
    by_start = {
        row["period_coverage"]["start_date"]: float(row["value"])
        for row in vendored_rows(
            "dft_bus_value_anchors.json",
            concept="dft.local_bus_fares_index",
            geography_id="dft:england_outside_london",
        )
    }
    calendar = [by_start[f"2024-{m}-01"] for m in ("01", "04", "07", "10")]
    fiscal = [
        by_start[q] for q in ("2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01")
    ]
    assert factor == pytest.approx(np.mean(calendar) / np.mean(fiscal))
    assert factor < 1.0  # fares rose in January 2025
    assert receipt["calendar_quarters"]["2024-01-01"] == calendar[0]

    annual = {
        "resource": "devolved_bus_finance.json",
        "concept": "scotgov.bus.local_fare_index",
        "geography_id": "S92000003",
        "frequency": "annual",
    }
    factor, receipt = calendar_price_factor(
        annual,
        price_year=2024,
        fiscal_start="2024-04-01",
        allowed_resources=UK_LCFS_VENDORED_RESOURCES,
    )
    years = {int(k): v for k, v in receipt["calendar_years"].items()}
    assert factor == pytest.approx(
        years[2024] / (0.75 * years[2024] + 0.25 * years[2025])
    )
    for change, match in (
        ({"frequency": "monthly"}, "unsupported price index frequency"),
        ({"resource": "family_resources.json"}, "not declared"),
        ({"geography_id": "W92000004"}, "lacks quarters"),
    ):
        with pytest.raises(BusFarePricingError, match=match):
            calendar_price_factor(
                {**quarterly, **change},
                price_year=2024,
                fiscal_start="2024-04-01",
                allowed_resources=UK_LCFS_VENDORED_RESOURCES,
            )


def test_a_price_year_needs_every_area_to_name_its_index() -> None:
    declared = _declared()
    areas = [dict(area) for area in declared["areas"]]
    areas[1].pop("price_index")
    with pytest.raises(BusFarePricingError, match="declares no price_index"):
        bus_fare_prices(
            {**declared, "areas": areas}, allowed_resources=UK_LCFS_VENDORED_RESOURCES
        )
    # Without a price year the yields stay on the publisher's fiscal year.
    unrepriced = bus_fare_prices(
        {key: value for key, value in declared.items() if key != "price_year"},
        allowed_resources=UK_LCFS_VENDORED_RESOURCES,
    )
    assert all(
        area.calendar_price_factor == 1.0
        for area in unrepriced.other_by_region.values()
    )


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


def test_bus_engine_parity_marks_each_area_against_the_engine() -> None:
    """The engine's k and y against the stage's, report only (#1207)."""

    from microcosm.build.uk_runtime.bus_fare_pricing import (
        UK_ENGINE_BUS_AREA_BY_STAGE_AREA,
        bus_engine_parity,
    )

    prices = bus_fare_prices(_declared(), allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    areas = {
        prices.london.label: prices.london,
        **{area.label: area for area in prices.other_by_region.values()},
    }
    engine = {
        "boardings_per_trip": {
            UK_ENGINE_BUS_AREA_BY_STAGE_AREA[label]: area.boardings_per_trip
            for label, area in areas.items()
        },
        "yield_per_boarding": {
            UK_ENGINE_BUS_AREA_BY_STAGE_AREA[label]: area.yield_per_fare_paying_boarding
            for label, area in areas.items()
        },
    }
    parity = bus_engine_parity(prices, engine)
    assert parity["report_only"] is True and parity["within_tolerance"] is True
    assert set(parity["areas"]) == set(UK_ENGINE_BUS_AREA_BY_STAGE_AREA)

    engine["yield_per_boarding"]["NORTHERN_IRELAND"] *= 1.01
    drifted = bus_engine_parity(prices, engine)
    ni = drifted["areas"]["northern_ireland"]["yield_per_boarding"]
    assert ni["relative_gap"] == pytest.approx(1 / 1.01 - 1)
    assert ni["within_tolerance"] is False and drifted["within_tolerance"] is False


def _journey_cell(area: str, fact_period: str) -> TargetSpec:
    return TargetSpec(
        name=f"dft.bus_journeys.{area}",
        entity="household",
        value=1.0e8,
        measure="other_local_bus_boardings",
        period=2025,
        family="dft_bus_journeys",
        source="test",
        metadata={
            "contract_target_id": f"dft.bus_journeys.{area}",
            "ledger_fact_period": fact_period,
        },
    )


def _journey_reference(index: str = UK_BUS_JOURNEYS_YEAR_INDEX):
    return LedgerTargetReference(
        name="dft.bus_journeys.north_east",
        ledger_selector={"source_name": "dft"},
        entity="household",
        measure="other_local_bus_boardings",
        period=2025,
        uprating_index=index,
    )


def test_journey_rows_hold_at_the_year_the_stage_prices_from() -> None:
    """BUS01a FY2024-25 against the stage's fiscal_start (microcosm#1207 r1)."""

    assert _declared()["fiscal_start"] == "2024-04-01"
    registry = TargetRegistry([_journey_cell("north_east", "2024")], country="uk")

    (spec,) = hold_bus_journeys_at_the_stage_year(_journey_reference(), registry).specs

    assert spec.value == 1.0e8
    assert spec.metadata["uprating_factor"] == "1"
    assert spec.metadata["uprating_hold"] == UK_BUS_JOURNEYS_HOLD
    assert "2024-04-01" in spec.metadata["uprating_index_basis"]
    assert is_uprating_hold(spec, 2025)
    assert UK_UPRATING_APPLIERS[UK_BUS_JOURNEYS_YEAR_INDEX] is (
        hold_bus_journeys_at_the_stage_year
    )


def test_a_journey_row_from_another_year_is_refused() -> None:
    """A later BUS01a edition in the feed cannot bind on the stage's FY2024-25 factors."""

    later = TargetRegistry([_journey_cell("north_east", "2025")], country="uk")
    with pytest.raises(ValueError, match="move the stage's fiscal_start"):
        hold_bus_journeys_at_the_stage_year(_journey_reference(), later)
    other = _journey_reference("dft.local_bus_fares_index")
    assert hold_bus_journeys_at_the_stage_year(other, later) is later
    with pytest.raises(BusFarePricingError, match="not an April 1st"):
        hold_bus_journeys_at_the_stage_year(
            _journey_reference(), later, parameters={"fiscal_start": "2024-01-01"}
        )


def test_the_eight_regions_sum_to_england_outside_london() -> None:
    declared = _declared()
    regions = english_region_boardings(
        declared, allowed_resources=UK_LCFS_VENDORED_RESOURCES
    )
    prices = bus_fare_prices(declared, allowed_resources=UK_LCFS_VENDORED_RESOURCES)

    assert sorted(regions) == sorted(
        region
        for region in prices.other_by_region
        if prices.other_by_region[region].label == "england_outside_london"
        and region != "LONDON"
    )
    assert sum(regions.values()) == pytest.approx(
        prices.other_by_region["NORTH_EAST"].boardings, rel=1e-6
    )


def test_region_shares_set_the_frame_beside_bus01a() -> None:
    """Shares of the eight regions' trips and journeys, London residents apart."""

    receipt = england_outside_london_region_shares(
        {"NORTH_EAST": 300.0, "SOUTH_WEST": 100.0},
        person_region=np.array(["NORTH_EAST", "SOUTH_WEST", "LONDON", "WALES"]),
        person_weight=np.array([1.0, 2.0, 1.0, 1.0]),
        other_trips=np.array([20.0, 10.0, 10.0, 99.0]),
    )

    north_east = receipt["regions"]["NORTH_EAST"]
    assert north_east["frame_share"] == pytest.approx(0.5)
    assert north_east["published_share"] == pytest.approx(0.75)
    assert north_east["frame_over_published_share"] == pytest.approx(0.5 / 0.75)
    assert receipt["regions"]["SOUTH_WEST"]["frame_share"] == pytest.approx(0.5)
    # London residents' 10 trips beside the regions' 40.
    assert receipt["london_residents_share_of_other_local_trips"] == pytest.approx(0.2)
    assert receipt["report_only"] is True


def test_the_stage_receipt_carries_the_region_shares() -> None:
    prices = bus_fare_prices(_declared(), allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    household = pd.DataFrame(
        {"household_id": [1, 2], "region": ["NORTH_EAST", "SOUTH_EAST"]}
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 21],
            "person_household_id": [1, 2],
            "bus_in_london_trips": [0.0, 0.0],
            "other_local_bus_trips": [30.0, 10.0],
            "bus_pass_eligible": [False, False],
        }
    )
    _, _, receipt = price_bus_journeys(
        person,
        household,
        prices=prices,
        trips_columns={
            BUS_IN_LONDON: "bus_in_london_trips",
            OTHER_LOCAL_BUS: "other_local_bus_trips",
        },
        eligibility_column="bus_pass_eligible",
        household_weights=np.array([1.0, 1.0]),
        raw_household_fares=np.array([0.0, 0.0]),
        region_boardings={"NORTH_EAST": 1.0, "SOUTH_EAST": 3.0},
    )
    regions = receipt["england_outside_london_regions"]["regions"]
    assert regions["NORTH_EAST"]["frame_share"] == pytest.approx(0.75)
    assert regions["NORTH_EAST"]["published_share"] == pytest.approx(0.25)
