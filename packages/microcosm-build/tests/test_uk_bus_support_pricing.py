"""Bus support priced from journeys at the published support per boarding (microcosm#930)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.bus_fare_pricing import BusFarePricingError
from microcosm.build.uk_runtime.bus_support_pricing import (
    PRICE_BUS_SUPPORT_KIND,
    bus_support_factors,
    bus_support_pricing_operation,
    price_bus_support,
)
from microcosm.build.uk_runtime.etb_services import (
    UK_ETB_SERVICES_VENDORED_RESOURCES,
)

ENGLAND_OUTSIDE_LONDON = (
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "SOUTH_EAST",
    "SOUTH_WEST",
)


def _declared() -> dict:
    stage = load_country_spec("uk").sources.stage_map()["etb_services"]
    parameters = bus_support_pricing_operation(stage)
    assert parameters is not None and parameters["kind"] == PRICE_BUS_SUPPORT_KIND
    assert parameters["support_column"] == "bus_subsidy_spending"
    assert [a["label"] for a in parameters["support_areas"]] == [
        "london",
        "england_outside_london",
        "scotland",
    ]
    assert parameters["raw_draw_regions"] == ["WALES", "NORTHERN_IRELAND"]
    assert parameters["support_areas"][1]["regions"] == list(ENGLAND_OUTSIDE_LONDON)
    return parameters


def _frame() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5, 6],
            "region": [
                "LONDON",
                "SOUTH_EAST",
                "NORTH_WEST",
                "WALES",
                "SCOTLAND",
                "NORTHERN_IRELAND",
            ],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 12, 21, 31, 41, 51, 52, 61],
            "person_household_id": [1, 1, 2, 3, 4, 5, 5, 6],
            "bus_in_london_trips": [100.0, 40.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            "other_local_bus_trips": [0.0, 10.0, 60.0, 30.0, 50.0, 20.0, 80.0, 30.0],
            "bus_pass_eligible": [False, True, True, False, True, True, False, True],
        }
    )
    weights = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    raw = np.array([120.0, 80.0, 90.0, 70.0, 60.0, 50.0])
    return person, household, weights, raw


def test_factors_recompute_from_the_vendored_rows() -> None:
    factors = bus_support_factors(
        _declared(), allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES
    )
    assert set(factors) == {"london", "england_outside_london", "scotland"}
    for label, block in factors.items():
        reimbursement = block["reimbursement"]["value"]
        net = block["net_support"]["value"]
        boardings = block["boardings"]["value"]
        concessionary = block["concessionary"]["value"]
        assert 0 < reimbursement < net, label
        assert 0 < concessionary < boardings, label
        assert block["reimbursement_per_concessionary_boarding"] == pytest.approx(
            reimbursement / concessionary
        )
        assert block["other_support_per_boarding"] == pytest.approx(
            (net - reimbursement) / boardings
        )
        assert block["published_concessionary_boarding_share"] == pytest.approx(
            concessionary / boardings
        )
    # Scotland's components come from the Scottish Transport Statistics rows:
    # concessionary fares inside all government support.
    scotland = factors["scotland"]
    assert scotland["net_support"]["value"] == pytest.approx(499_000_000.0)
    assert scotland["reimbursement"]["value"] == pytest.approx(392_000_000.0)
    assert scotland["regions"] == ["SCOTLAND"]


def test_support_is_priced_per_area_and_raw_elsewhere() -> None:
    parameters = _declared()
    person, household, weights, raw = _frame()
    values, priced, receipt = price_bus_support(
        parameters,
        person=person,
        household=household,
        household_weights=weights,
        raw_support=raw,
        allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
    )
    factors = bus_support_factors(
        parameters, allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES
    )
    k_london = receipt["boardings_per_trip"]["bus_in_london"]
    k_england = receipt["boardings_per_trip"]["england_outside_london"]
    k_scotland = receipt["boardings_per_trip"]["scotland"]
    eol = factors["england_outside_london"]
    lon = factors["london"]
    sco = factors["scotland"]
    # Household 1 (London): person 11 not eligible, 100 London trips; person 12
    # eligible, 40 London + 10 other trips (other trips price at the
    # England-outside-London translation for London residents).
    b11 = 100.0 * k_london
    b12 = 40.0 * k_london + 10.0 * k_england
    expected_1 = (
        lon["other_support_per_boarding"] * (b11 + b12)
        + lon["reimbursement_per_concessionary_boarding"] * b12
    )
    assert values[0] == pytest.approx(expected_1)
    # Household 2 (South East): person 21 eligible, 60 other trips.
    b21 = 60.0 * k_england
    assert values[1] == pytest.approx(
        (
            eol["other_support_per_boarding"]
            + eol["reimbursement_per_concessionary_boarding"]
        )
        * b21
    )
    # Household 3 (North West): person 31 not eligible, 10 London + 30 other.
    b31 = 10.0 * k_london + 30.0 * k_england
    assert values[2] == pytest.approx(eol["other_support_per_boarding"] * b31)
    # Household 5 (Scotland): person 51 eligible 20 trips, person 52 not, 80.
    b51, b52 = 20.0 * k_scotland, 80.0 * k_scotland
    assert values[4] == pytest.approx(
        sco["other_support_per_boarding"] * (b51 + b52)
        + sco["reimbursement_per_concessionary_boarding"] * b51
    )
    # Wales and Northern Ireland keep the raw draw.
    assert values[3] == 70.0 and values[5] == 50.0
    assert priced.tolist() == [True, True, True, False, True, False]
    assert receipt["applied"] is True
    assert receipt["chain_conditioned_on"] == "raw_draw"
    assert receipt["raw_draw_regions"] == ["NORTHERN_IRELAND", "WALES"]
    assert receipt["households_priced"] == 4 and receipt["households_raw_draw"] == 2
    assert receipt["raw_draw_weighted_support"] == {
        "NORTHERN_IRELAND": pytest.approx(7.0 * 50.0),
        "WALES": pytest.approx(5.0 * 70.0),
    }
    england = receipt["by_area"]["england_outside_london"]
    assert england["frame_boardings"] == pytest.approx(3.0 * b21 + 4.0 * b31)
    assert england["frame_eligible_boardings"] == pytest.approx(3.0 * b21)
    assert england["weighted_support_before"] == pytest.approx(3.0 * 80.0 + 4.0 * 90.0)
    assert england["weighted_support_after"] == pytest.approx(
        3.0 * values[1] + 4.0 * values[2]
    )
    assert england["priced_over_published"] == pytest.approx(
        england["weighted_support_after"] / eol["net_support"]["value"]
    )
    assert receipt["weighted_support_after"] == pytest.approx(
        float(np.dot(values, weights))
    )


def test_pricing_refusals() -> None:
    parameters = _declared()
    person, household, weights, raw = _frame()
    with pytest.raises(
        BusFarePricingError, match="nts_bus_travel stage must run first"
    ):
        price_bus_support(
            parameters,
            person=person.drop(columns=["bus_pass_eligible"]),
            household=household,
            household_weights=weights,
            raw_support=raw,
            allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        )
    with pytest.raises(BusFarePricingError, match="neither in a support area"):
        price_bus_support(
            {**parameters, "raw_draw_regions": ["WALES"]},
            person=person,
            household=household,
            household_weights=weights,
            raw_support=raw,
            allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        )
    with pytest.raises(BusFarePricingError, match="both priced and declared"):
        price_bus_support(
            {**parameters, "raw_draw_regions": ["WALES", "NORTHERN_IRELAND", "LONDON"]},
            person=person,
            household=household,
            household_weights=weights,
            raw_support=raw,
            allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        )
    with pytest.raises(BusFarePricingError, match="not a price_bus_support"):
        price_bus_support(
            {**parameters, "kind": "price_bus_journeys"},
            person=person,
            household=household,
            household_weights=weights,
            raw_support=raw,
            allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        )
