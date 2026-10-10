"""HMRC higher and additional rate taxpayers by region, nation and area (#1123)."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from importlib import resources as importlib_resources
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed
from microcosm.build.uk_runtime.cross_grain_declarations import (
    load_uk_cross_grain_declarations,
)
from microcosm.build.uk_runtime.ledger_targets import uk_cross_grain_leg_of_area
from microcosm.build.uk_runtime.uprating_holds import load_uk_uprating_holds
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"
NATIONAL = ("hmrc.itl.higher_rate_taxpayers", "hmrc.itl.additional_rate_taxpayers")
AREA = "hmrc.higher_rate_taxpayers.count"
AREA_SPECS = (
    "uk.local_geography.spi_income.by_constituency.v1",
    "uk.local_geography.spi_income.by_local_authority.v1",
)


def _contract() -> dict[str, dict]:
    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )
    return {target["target_id"]: target for target in payload["targets"]}


def test_the_national_rows_fan_out_as_the_calendar_2025_window():
    contract = _contract()
    for target_id, band in zip(NATIONAL, ("HIGHER", "ADDITIONAL"), strict=True):
        target = contract[target_id]
        assert sorted(target["geography_levels"]) == ["country", "region"]
        assert target["value_operation"] == "calendar_year_window"
        rate = target_id.split(".")[2].removesuffix("_rate_taxpayers")
        assert target["ledger_selector"]["dimension_values"] == {"marginal_rate": rate}
        assert {
            "variable": "tax_band",
            "operator": "==",
            "value": band,
        } in target["bindings"]["policyengine"]["filters"]


def test_the_area_cells_bind_england_wales_and_northern_ireland_only():
    target = _contract()[AREA]
    assert target["area_scope"] == {
        level: {"gss_prefixes": ["E", "W", "N"]}
        for level in ("constituency", "local_authority")
    }
    assert target["ledger_selector"]["record_set_spec_id"] == list(AREA_SPECS)
    assert "uprating_index" not in target


def test_each_region_and_nation_row_controls_its_area_cells():
    (bridge,) = [
        entry
        for entry in load_uk_cross_grain_declarations()["bridges"]
        if entry["lower_side"] == f"contract:{AREA}"
    ]
    assert bridge["higher_target_ids"] == ["hmrc.itl.higher_rate_taxpayers"]
    assert bridge["per_geography"] is True
    (hold,) = [
        entry
        for entry in load_uk_uprating_holds()["holds"]
        if AREA in entry["target_ids"]
    ]
    assert (hold["scope"], hold["kind"]) == ("local", "control_rescaled")


def _pinned_feed_facts(predicate) -> list[dict]:
    root = _TEST_PATHS.repository
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = Path(configured) if configured else root / STABLE_UK_FACT_FEED_NAME
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    assert (
        hashlib.sha256(feed.read_bytes()).hexdigest()
        == load_uk_chronicle_feed().facts_sha256
    )
    with feed.open(encoding="utf-8") as handle:
        return [
            fact
            for line in handle
            if line.strip() and predicate(fact := json.loads(line))
        ]


def _measure(fact: dict) -> str:
    return str((fact.get("observed_measure") or {}).get("source_measure_id") or "")


def test_the_area_cells_sum_to_their_2023_24_region_and_nation_rows():
    """At source the area cells and Table 2.2 measure one quantity.

    The bridge rescales each region's or nation's 2023-24 area cells to its
    calendar-2025 row, so the cells must sum to that row's own 2023-24 value
    at both grains: within 1.2% on the pinned feed, Northern Ireland's one
    rounded thousand the widest (cells are rounded to the thousand).
    """

    facts = _pinned_feed_facts(
        lambda fact: _measure(fact) in {"higher_rate", "higher_rate_count"}
    )
    leg_of_area = uk_cross_grain_leg_of_area()
    controls = {
        fact["geography"]["id"]: float(fact["value"])
        for fact in facts
        if _measure(fact) == "higher_rate"
        and fact["period"]["value"] == 2023
        and fact["geography"]["level"] in {"region", "country"}
    }
    sums: dict[tuple[str, str], float] = defaultdict(float)
    for fact in facts:
        level = fact["geography"]["level"]
        area = fact["geography"]["id"]
        if (
            _measure(fact) == "higher_rate_count"
            and level in {"constituency", "local_authority"}
            and fact["layout"]["record_set_spec_id"] in AREA_SPECS
            and area[0] in "EWN"
        ):
            sums[(level, leg_of_area(area))] += float(fact["value"])
    legs = {leg for _, leg in sums}
    assert len(legs) == 11
    for (level, leg), total in sums.items():
        assert total == pytest.approx(controls[leg], rel=0.015), (level, leg)
