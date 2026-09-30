"""The State Pension target rows (microcosm#1069 c3)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed
from microcosm.build.uk_runtime.local_targets import load_uk_population_contract
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
AGE_CELL = re.compile(r"^dwp\.state_pension\.recipients_(male|female)_\d+_(\d+|plus)_")
STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"
PUBLISHER_TYPE_TO_ENGINE = {
    "New State Pension": "NEW",
    "Pre-2016 State Pension": "BASIC",
}
GB_REGIONS = {
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
}


def _pension_targets() -> list[dict]:
    return [
        target
        for target in load_uk_population_contract()["targets"]
        if target["family"] == "dwp_state_pension"
    ]


def _pension_references() -> list:
    return [
        reference
        for reference in load_country_spec("uk").target_references
        if reference.family == "dwp_state_pension"
    ]


def _filter(binding: dict, variable: str, operator: str = "==") -> list:
    return [
        predicate.get("value", predicate.get("equals"))
        for predicate in binding.get("filters", ())
        if predicate["variable"] == variable
        and predicate.get("operator", "==") == operator
    ]


def test_the_state_pension_family_is_26_targets_and_60_references() -> None:
    targets = _pension_targets()
    references = _pension_references()

    assert len(targets) == 26
    assert len(references) == 60
    assert {target["category_id"] for target in targets} == {
        "dwp.state_pension",
        "dfc_ni.state_pension",
    }
    assert all(target["period_match_policy"] == "source_window" for target in targets)


def test_every_row_binds_the_population_its_selector_names() -> None:
    for target in _pension_targets():
        binding = target["bindings"]["policyengine"]
        selector = target["ledger_selector"]
        cells = selector.get("dimension_values", {})
        conditions = binding["household_conditions"]
        assert len(conditions) == 1, target["target_id"]
        (condition,) = conditions
        assert condition["map_to"] == "person"
        if target["target_id"].startswith("dfc_ni."):
            assert condition["value"] == "NORTHERN_IRELAND"
        elif target["target_id"].endswith("_by_english_region"):
            assert (condition["variable"], condition["value"]) == (
                "country",
                "ENGLAND",
            )
        elif target["target_id"].endswith(("_scotland", "_wales")):
            assert condition["value"] == target["target_id"].rsplit("_", 1)[1].upper()
        else:
            assert set(condition["value"]) == GB_REGIONS
        if binding.get("value_variable") == "person_count":
            assert _filter(binding, "state_pension", ">") == [0]
        publisher_type = cells.get("category_of_pension")
        if publisher_type not in (None, "all"):
            assert _filter(binding, "state_pension_type") == [
                PUBLISHER_TYPE_TO_ENGINE[publisher_type]
            ], target["target_id"]
        ages = cells.get("age_bands_and_single_year")
        if ages not in (None, "all"):
            low = _filter(binding, "age", ">=")
            high = _filter(binding, "age", "<=")
            if ages == "90 and over":
                assert (low, high) == ([90], [])
            else:
                assert (low, high) == ([min(ages)], [max(ages)])
                assert [
                    operand["dimension_values"]["age_bands_and_single_year"]
                    for operand in target["value_operands"]
                ] == list(ages)


def test_the_amounts_multiply_recipients_by_the_weekly_mean_at_2025_rates() -> None:
    amounts = [
        target
        for target in _pension_targets()
        if target["target_id"].endswith(".amount")
    ]

    assert [target["target_id"] for target in amounts] == [
        "dwp.state_pension.amount",
        "dfc_ni.state_pension.amount",
    ]
    for target in amounts:
        assert target["value_operation"] == "monthly_window_count_x_mean"
        assert target["value_operands"][1]["period_factor"] == 52
        assert target["uprating_index"] == (
            "policyengine_uk_parameter:gov.dwp.state_pension.new_state_pension.amount"
        )
        assert target["bindings"]["policyengine"]["value_variable"] == "state_pension"


def test_the_amount_bands_keep_the_points_paid_at_the_2025_26_rate() -> None:
    bands = [
        target
        for target in _pension_targets()
        if target["target_id"].endswith("_by_weekly_amount")
    ]

    assert len(bands) == 2
    for target in bands:
        binding = target["bindings"]["policyengine"]
        assert target["measurement"]["source_months"] == [
            "2025-05",
            "2025-08",
            "2025-11",
        ]
        assert binding["groupby_variable"] == "state_pension"
        assert binding["band_period_factor"] == 52
        # The detail measure leaves out the publisher's 'all' margin row.
        assert target["ledger_selector"]["source_measure_id"] == "recipients"
    rows = [
        reference.name
        for reference in _pension_references()
        if "_by_weekly_amount." in reference.name
    ]
    assert len(rows) == 20
    assert not any("grouped_amount_of_benefit_all" in name for name in rows)


def _state_pension_facts() -> list[dict]:
    """The pinned feed's State Pension facts, or skip (the artifact is untracked)."""

    root = _TEST_PATHS.repository
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = Path(configured) if configured else root / STABLE_UK_FACT_FEED_NAME
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    assert hashlib.sha256(feed.read_bytes()).hexdigest() == (
        load_uk_chronicle_feed().facts_sha256
    )
    with feed.open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if "state-pension" in line or "state_pension" in line
        ]


def test_the_shape_rows_reconcile_with_the_great_britain_level() -> None:
    """Feed-gated: every shape partitions the resident caseload it describes."""

    references = [
        reference
        for reference in _pension_references()
        if not reference.name.endswith(".amount")
        and "_by_weekly_amount." not in reference.name
    ]
    compiled = compile_ledger_target_references(
        _state_pension_facts(), references, country="uk"
    )
    values = {spec.name: spec.value for spec in compiled.specs}
    total = values["dwp.state_pension.recipients"]
    ages = sum(value for name, value in values.items() if AGE_CELL.match(name))
    areas = sum(
        value
        for name, value in values.items()
        if "_by_english_region@" in name or name.endswith(("_scotland", "_wales"))
    )

    # DWP's unknown ages (740) and the near-empty type cells left unbound are
    # the only difference.
    assert ages == pytest.approx(total, abs=1_000)
    assert areas == pytest.approx(total, abs=1_000)
    assert values["dfc_ni.state_pension.recipients"] == pytest.approx(330_147.5)
