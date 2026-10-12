"""The Pension Credit caseload target rows (microcosm#1069 c7)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed
from microcosm.build.uk_runtime.local_targets import load_uk_population_contract
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"
PUBLISHER_TYPES = {
    "Guarantee Credit only": ("guarantee_credit", ">", "savings_credit", "<="),
    "Savings Credit only": ("guarantee_credit", "<=", "savings_credit", ">"),
    "Both Guarantee and Savings Credit": (
        "guarantee_credit",
        ">",
        "savings_credit",
        ">",
    ),
}
PARTNERS = {"Has a partner": "COUPLE", "Has no partner": "SINGLE"}
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


def _targets() -> list[dict]:
    return [
        target
        for target in load_uk_population_contract()["targets"]
        if target["family"] == "dwp_pension_credit"
    ]


def _references() -> list:
    return [
        reference
        for reference in load_country_spec("uk").target_references
        if reference.family == "dwp_pension_credit"
    ]


def _predicates(binding: dict, variable: str) -> list[tuple[str, object]]:
    return [
        (
            predicate.get("operator", "=="),
            predicate.get("value", predicate.get("equals")),
        )
        for predicate in binding["filters"]
        if predicate["variable"] == variable
    ]


def test_the_pension_credit_family_is_14_caseload_rows() -> None:
    targets = _targets()

    assert len(targets) == 14
    assert len(_references()) == 14
    assert {target["category_id"] for target in targets} == {
        "dwp.pension_credit",
        "dfc_ni.pension_credit",
    }
    for target in targets:
        assert target["value_operation"] == "monthly_window_average"
        assert target["period_match_policy"] == "source_window"
        assert target["measurement"]["source_months"] == [
            "2025-02",
            "2025-05",
            "2025-08",
            "2025-11",
        ]
        binding = target["bindings"]["policyengine"]
        assert (binding["from_entity"], binding["value_variable"]) == (
            "benunit",
            "benunit_count",
        )
        assert ("<", 0) not in _predicates(binding, "pension_credit")
        assert _predicates(binding, "pension_credit") == [(">", 0)]


def test_every_row_binds_the_cell_its_selector_names() -> None:
    for target in _targets():
        binding = target["bindings"]["policyengine"]
        (condition,) = binding["household_conditions"]
        assert condition["map_to"] == "benunit"
        if target["target_id"].startswith("dfc_ni."):
            assert condition["value"] == "NORTHERN_IRELAND"
            continue
        assert set(condition["value"]) == GB_REGIONS
        cells = target["ledger_selector"]["dimension_values"]
        publisher_type = cells["type_of_pension_credit"]
        if publisher_type != "all":
            gc, gc_operator, sc, sc_operator = PUBLISHER_TYPES[publisher_type]
            assert _predicates(binding, gc) == [(gc_operator, 0)]
            assert _predicates(binding, sc) == [(sc_operator, 0)]
            assert _predicates(binding, "relation_type") == [
                ("==", PARTNERS[cells["partner_indicator"]])
            ]
        age = cells["age_bands_and_single_year"]
        if age != "all":
            low = [
                value
                for op, value in _predicates(binding, "eldest_adult_age")
                if op == ">="
            ]
            high = [
                value
                for op, value in _predicates(binding, "eldest_adult_age")
                if op == "<="
            ]
            if age == "90 and over":
                assert (low, high) == ([90], [])
            else:
                first, last = (int(part) for part in age.split("-"))
                assert (low, high) == ([first], [last])


def _pension_credit_facts() -> list[dict]:
    """The pinned feed's Pension Credit facts, or skip (the artifact is untracked)."""

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
        return [json.loads(line) for line in handle if "pension_credit" in line]


def test_the_cells_partition_the_great_britain_caseload() -> None:
    """Feed-gated: type-by-partner and age cells each sum to the GB caseload."""

    compiled = compile_ledger_target_references(
        _pension_credit_facts(), _references(), country="uk"
    )
    values = {spec.name.split("@")[0]: spec.value for spec in compiled.specs}
    total = values["dwp.pension_credit.benefit_units"]
    cells = sum(
        value for name, value in values.items() if name.endswith(("_couple", "_single"))
    )
    ages = sum(
        value
        for name, value in values.items()
        if name.rsplit("_", 2)[-2:][0].isdigit() and not name.startswith("dfc_ni.")
    )

    # DWP publishes no claims of unknown age in 2025.
    assert total == pytest.approx(1_390_586.25)
    assert cells == pytest.approx(total, abs=10)
    assert ages == pytest.approx(total, abs=10)
    assert 10_000 < values["dfc_ni.pension_credit.benefit_units"] < 100_000
