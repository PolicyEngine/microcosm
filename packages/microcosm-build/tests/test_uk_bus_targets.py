from __future__ import annotations

import json
from importlib import resources as importlib_resources
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "uk_bus05i_fy2025.json"
BUS_PREFIX = "dft.bus_"


def _resource(name: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
    )


def test_bus05i_fy2025_subareas_reconcile_to_england() -> None:
    fixture = json.loads(FIXTURE.read_text())
    rows = fixture["rows"]

    assert len(rows) == 6
    assert {row["period"]["type"] for row in rows} == {"fiscal_year"}
    assert {row["period"]["value"] for row in rows} == {2025}
    assert {row["unit"] for row in rows} == {"gbp"}

    for concept in {
        "dft.local_bus_passenger_fare_receipts",
        "dft.local_bus_total_estimated_net_support",
    }:
        values = {
            row["groupby_value_id"]: row["value"]
            for row in rows
            if row["source_concept"] == concept
        }
        assert values["london"] + values["england_outside_london"] == pytest.approx(
            values["england"], abs=1_000
        )


def test_bus_targets_are_active_or_signed_excluded_as_declared() -> None:
    contract = _resource("uk_population_targets.json")
    references = {
        row["name"] for row in _resource("target_references.json")["target_references"]
    }
    membership = _resource("target_reference_membership.json")
    exclusions = {
        row["target_id"]: row
        for row in _resource("target_reference_signed_exclusions.json")["exclusions"]
    }
    bus = {
        row["target_id"]: row
        for row in contract["targets"]
        if row["target_id"].startswith(BUS_PREFIX)
    }

    active = {"dft.bus_fare_receipts.england", "dft.bus_net_support.england"}
    excluded = set(bus) - active
    assert len(bus) == 8
    assert active <= references
    assert excluded.isdisjoint(references)
    assert set(exclusions) >= excluded
    assert {exclusions[target_id]["reason_id"] for target_id in excluded} == {
        "region_pin_unsupported",
        "statistical_scope_unsupported",
        "no_publisher_uk_total",
    }
    assert all(
        membership["targets"][target_id]["status"] == "signed_excluded"
        for target_id in excluded
    )


def test_bus_selectors_do_not_consume_nts_or_bus0415() -> None:
    targets = [
        row
        for row in _resource("uk_population_targets.json")["targets"]
        if row["target_id"].startswith(BUS_PREFIX)
    ]
    selected_concepts = {
        row["ledger_selector"].get("source_concept", "") for row in targets
    }

    assert selected_concepts == {
        "dft.local_bus_passenger_fare_receipts",
        "dft.local_bus_total_estimated_net_support",
    }
    england_notes = " ".join(
        row["bindings"]["policyengine"]["notes"]
        for row in targets
        if row["target_id"].endswith("england")
    )
    assert "NTS0705a" in england_notes
    assert "BUS0415" in england_notes
    assert "#790" in england_notes
