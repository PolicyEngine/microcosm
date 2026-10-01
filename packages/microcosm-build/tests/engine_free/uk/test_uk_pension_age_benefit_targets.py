"""The pension-age benefit caseloads and the Winter Fuel diagnostic (microcosm#1069 c10)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.local_targets import load_uk_population_contract
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"
QUARTERS = ["2025-02", "2025-05", "2025-08", "2025-11"]
MONTHS = [f"2025-{month:02d}" for month in range(1, 13)]


def _targets() -> dict[str, dict]:
    contract = load_uk_population_contract()
    return {target["target_id"]: target for target in contract["targets"]}


def test_attendance_allowance_binds_england_and_wales_only() -> None:
    targets = _targets()
    attendance = {
        name: target
        for name, target in targets.items()
        if name.startswith("dwp.attendance_allowance.")
    }
    # Scotland's caseload moves to Pension Age Disability Payment through
    # 2025, which the engine does not model, so it is not bound.
    assert set(attendance) == {
        "dwp.attendance_allowance.recipients_england",
        "dwp.attendance_allowance.recipients_wales",
    }
    for country, target in (
        ("ENGLAND", attendance["dwp.attendance_allowance.recipients_england"]),
        ("WALES", attendance["dwp.attendance_allowance.recipients_wales"]),
    ):
        assert target["family"] == "dwp_attendance_allowance"
        assert target["value_operation"] == "monthly_window_average"
        assert target["measurement"]["source_months"] == QUARTERS
        selector = target["ledger_selector"]
        assert selector["source_measure_id"] == "total_cases_with_entitlement"
        assert selector["period_value"] == QUARTERS
        assert set(selector["dimension_values"].values()) == {"all"}
        binding = target["bindings"]["policyengine"]
        assert binding["filters"] == [
            {"variable": "attendance_allowance", "operator": ">", "value": 0}
        ]
        (condition,) = binding["household_conditions"]
        assert (condition["variable"], condition["value"]) == ("country", country)


def test_pension_age_housing_benefit_reads_the_detail_tenure_cells() -> None:
    targets = _targets()
    expected = {
        "dwp.hb.households_pension_age": ("all", "total_claimants"),
        "dwp.hb.households_pension_age_social_rented": (
            "Social Rented Sector",
            "claimants",
        ),
        "dwp.hb.households_pension_age_private_rented": (
            "Private Rented Sector",
            "claimants",
        ),
    }
    for name, (tenure, measure) in expected.items():
        target = targets[name]
        selector = target["ledger_selector"]
        assert selector["dimension_values"] == {
            "client_type": "Pension age",
            "private_or_social_rented": tenure,
        }
        # Stat-Xplore publishes the pension-age tenure cells as detail rows;
        # only the all-tenure row is a total.
        assert selector["source_measure_id"] == measure
        assert selector["period_value"] == MONTHS
        assert target["family"] == "dwp_housing_benefit"
        filters = target["bindings"]["policyengine"]["filters"]
        assert {"variable": "eldest_adult_age", "operator": ">=", "value": 66} in (
            filters
        )


def test_winter_fuel_recipients_are_a_diagnostic_never_a_target() -> None:
    contract = load_uk_population_contract()
    assert "dwp.winter_fuel_payment.recipients" not in {
        target["target_id"] for target in contract["targets"]
    }
    declaration = contract["diagnostic_references"][
        "dwp.winter_fuel_payment.recipients"
    ]
    assert declaration["attach_to_target"] == "obr.winter_fuel_allowance"
    selector = declaration["reference"]["ledger_selector"]
    # DWP publishes England and Wales; Scotland pays its own winter heating
    # payment from winter 2025/26.
    assert selector["geography_id"] == "K04000001"
    assert declaration["reference"]["ledger_fact_key"]


def test_winter_fuel_diagnostic_carries_the_published_count() -> None:
    """Feed-gated: the diagnostic resolves the winter 2025/26 figure."""

    from microcosm.build.uk_runtime.ledger_targets import (
        _attached_diagnostic_metadata,
    )

    root = _TEST_PATHS.repository
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = Path(configured) if configured else root / STABLE_UK_FACT_FEED_NAME
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    with feed.open(encoding="utf-8") as handle:
        facts = tuple(
            json.loads(line)
            for line in handle
            if "winter_fuel_payment_beneficiaries" in line
        )

    metadata = _attached_diagnostic_metadata(facts, "obr.winter_fuel_allowance")

    prefix = "dwp_winter_fuel_recipients_diagnostic"
    assert metadata[f"{prefix}_role"] == "diagnostic_only_not_in_fit"
    assert metadata[f"{prefix}_status"] == "available"
    assert float(metadata[f"{prefix}_value_count"]) == 11_003_237
    assert metadata[f"{prefix}_ledger_geography_id"] == "K04000001"
