"""The value-side bus-support diagnostic beside the ETB rake (microcosm#930 C7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.bus_fare_pricing import (
    BusFarePricingError,
    published_fact,
)
from microcosm.build.uk_runtime.bus_support_per_journey import (
    RECORD_SUPPORT_PER_JOURNEY_KIND,
    support_per_journey_diagnostic,
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
    parameters = next(
        dict(op.parameters)
        for op in stage.operations
        if op.kind == RECORD_SUPPORT_PER_JOURNEY_KIND
    )
    assert parameters["support_column"] == "bus_subsidy_spending"
    assert [a["label"] for a in parameters["support_areas"]] == [
        "london",
        "england_outside_london",
    ]
    assert parameters["support_areas"][1]["regions"] == list(ENGLAND_OUTSIDE_LONDON)
    return parameters


def _frame() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "region": ["LONDON", "SOUTH_EAST", "NORTH_WEST", "WALES"],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 12, 21, 31, 41],
            "person_household_id": [1, 1, 2, 3, 4],
            "bus_in_london_trips": [100.0, 40.0, 0.0, 10.0, 0.0],
            "other_local_bus_trips": [0.0, 10.0, 60.0, 30.0, 50.0],
            "bus_pass_eligible": [False, True, True, False, True],
        }
    )
    weights = np.array([2.0, 3.0, 4.0, 5.0])
    raked = np.array([120.0, 80.0, 90.0, 70.0])
    return person, household, weights, raked


def test_value_side_support_recomputes_from_the_vendored_rows() -> None:
    parameters = _declared()
    person, household, weights, raked = _frame()
    receipt = support_per_journey_diagnostic(
        parameters,
        person=person,
        household=household,
        household_weights=weights,
        raked_support=raked,
        allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
    )
    assert receipt["operation"] == RECORD_SUPPORT_PER_JOURNEY_KIND
    assert receipt["applied"] is False
    assert receipt["support_column"] == "bus_subsidy_spending"
    k_london = receipt["boardings_per_trip"]["bus_in_london"]
    k_england = receipt["boardings_per_trip"]["england_outside_london"]
    assert k_london > 0 and k_england > 0

    area = dict(parameters["support_areas"][1])
    fiscal_start = parameters["fiscal_start"]
    reimbursement, _ = published_fact(
        area["reimbursement"],
        unit="gbp",
        allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        fiscal_start=fiscal_start,
    )
    net_support, _ = published_fact(
        area["net_support"],
        unit="gbp",
        allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        fiscal_start=fiscal_start,
    )
    boardings, _ = published_fact(
        area["boardings"],
        unit="count",
        allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        fiscal_start=fiscal_start,
    )
    concessionary, _ = published_fact(
        area["concessionary"],
        unit="count",
        allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        fiscal_start=fiscal_start,
    )
    assert 0 < reimbursement < net_support
    assert 0 < concessionary < boardings
    england = receipt["by_area"]["england_outside_london"]
    # Households 2 (SOUTH_EAST, weight 3) and 3 (NORTH_WEST, weight 4):
    # person 21 (eligible) 60 other trips; person 31 10 London + 30 other trips.
    all_boardings = 3.0 * 60.0 * k_england + 4.0 * (10.0 * k_london + 30.0 * k_england)
    eligible_boardings = 3.0 * 60.0 * k_england
    assert england["frame_boardings"] == pytest.approx(all_boardings)
    assert england["frame_eligible_boardings"] == pytest.approx(eligible_boardings)
    assert england["frame_eligible_boarding_share"] == pytest.approx(
        eligible_boardings / all_boardings
    )
    assert england["published_concessionary_boarding_share"] == pytest.approx(
        concessionary / boardings
    )
    assert england["reimbursement_per_concessionary_boarding"] == pytest.approx(
        reimbursement / concessionary
    )
    assert england["other_support_per_boarding"] == pytest.approx(
        (net_support - reimbursement) / boardings
    )
    assert england["value_side_support"] == pytest.approx(
        reimbursement / concessionary * eligible_boardings
        + (net_support - reimbursement) / boardings * all_boardings
    )
    assert england["raked_support"] == pytest.approx(3.0 * 80.0 + 4.0 * 90.0)
    assert england["value_side_over_raked"] == pytest.approx(
        england["value_side_support"] / england["raked_support"]
    )
    assert england["households"] == 2 and england["persons"] == 2
    london = receipt["by_area"]["london"]
    assert london["frame_boardings"] == pytest.approx(
        2.0 * (100.0 * k_london + 40.0 * k_london + 10.0 * k_england)
    )
    # Person 12 is eligible: 40 London-series and 10 other-local-bus trips.
    assert london["frame_eligible_boardings"] == pytest.approx(
        2.0 * (40.0 * k_london + 10.0 * k_england)
    )
    assert london["raked_support"] == pytest.approx(2.0 * 120.0)
    # Wales is unpriced and outside every support area: recorded nowhere.
    assert set(receipt["by_area"]) == {"london", "england_outside_london"}


def test_diagnostic_refusals() -> None:
    parameters = _declared()
    person, household, weights, raked = _frame()
    with pytest.raises(
        BusFarePricingError, match="nts_bus_travel stage must run first"
    ):
        support_per_journey_diagnostic(
            parameters,
            person=person.drop(columns=["bus_pass_eligible"]),
            household=household,
            household_weights=weights,
            raked_support=raked,
            allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        )
    foreign = dict(parameters)
    foreign["support_areas"] = [
        {**dict(parameters["support_areas"][0]), "regions": ["LONDON", "WALES"]}
    ]
    with pytest.raises(BusFarePricingError, match="pricing areas do not price"):
        support_per_journey_diagnostic(
            foreign,
            person=person,
            household=household,
            household_weights=weights,
            raked_support=raked,
            allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        )
    with pytest.raises(BusFarePricingError, match="not a record_support_per_journey"):
        support_per_journey_diagnostic(
            {**parameters, "kind": "price_bus_journeys"},
            person=person,
            household=household,
            household_weights=weights,
            raked_support=raked,
            allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
        )
