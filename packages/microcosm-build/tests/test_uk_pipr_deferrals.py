from __future__ import annotations

import json
from importlib import resources as importlib_resources


def _membership() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_target_reference_membership.json")
        .read_text()
    )


def test_pipr_2025_surface_remains_fully_deferred_after_measurement() -> None:
    target = _membership()["targets"]["ons.rent.private_rent"]
    candidates = target["geography_levels"]["local_authority"]["candidates"]
    assert len(candidates) == 361
    assert sum(row["status"] == "active" for row in candidates) == 0
    assert (
        sum(row["status"] == "no_fact_at_or_before_period" for row in candidates) == 314
    )
    assert sum(row["status"] == "no_fact_for_area" for row in candidates) == 47


def test_pipr_four_signed_reasons_pin_feed_and_crosswalk_measurements() -> None:
    rows = {
        row["reason_id"]: row
        for row in _membership()["signed_deferrals"]
        if row["reason_id"].startswith("private_rent_pipr_")
    }
    assert set(rows) == {
        "private_rent_pipr_after_target_period",
        "private_rent_pipr_english_lad_absent",
        "private_rent_pipr_scotland_brma_grain",
        "private_rent_pipr_ni_absent",
    }
    assert len(rows["private_rent_pipr_after_target_period"]["area_ids"]) == 314
    assert (
        "348 facts total" in rows["private_rent_pipr_after_target_period"]["rationale"]
    )
    assert (
        "E08000038 and E08000039"
        in rows["private_rent_pipr_after_target_period"]["rationale"]
    )
    assert rows["private_rent_pipr_english_lad_absent"]["area_ids"] == [
        "E06000053",
        "E08000016",
        "E08000019",
        "E09000001",
    ]
    assert len(rows["private_rent_pipr_scotland_brma_grain"]["area_ids"]) == 32
    assert (
        "18 Scottish BRMA rows"
        in rows["private_rent_pipr_scotland_brma_grain"]["rationale"]
    )
    assert (
        "cannot translate" in rows["private_rent_pipr_scotland_brma_grain"]["rationale"]
    )
    assert len(rows["private_rent_pipr_ni_absent"]["area_ids"]) == 11
    assert (
        "zero Northern Ireland rows" in rows["private_rent_pipr_ni_absent"]["rationale"]
    )
