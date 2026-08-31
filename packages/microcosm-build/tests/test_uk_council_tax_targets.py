from __future__ import annotations

import json
from importlib import resources as importlib_resources


def _membership() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_target_reference_membership.json")
        .read_text()
    )


def test_council_tax_band_cells_activate_and_defer_as_measured() -> None:
    membership = _membership()
    expected_active = {
        "a": 317,
        "b": 318,
        "c": 318,
        "d": 318,
        "e": 318,
        "f": 318,
        "g": 318,
        "h": 316,
    }
    for band, active_count in expected_active.items():
        target_id = f"voa.council_tax_stock.by_area.band_{band}"
        candidates = membership["targets"][target_id]["geography_levels"][
            "local_authority"
        ]["candidates"]
        assert len(candidates) == 361
        assert sum(row["status"] == "active" for row in candidates) == active_count


def test_council_tax_signed_deferrals_pin_exact_gaps() -> None:
    deferrals = _membership()["signed_deferrals"]
    by_reason: dict[str, list[dict]] = {}
    for row in deferrals:
        if row["reason_id"].startswith("council_tax_"):
            by_reason.setdefault(row["reason_id"], []).append(row)

    assert len(by_reason["council_tax_voa_scotland_absent"]) == 8
    assert {
        len(row["area_ids"]) for row in by_reason["council_tax_voa_scotland_absent"]
    } == {32}
    assert len(by_reason["council_tax_ni_domestic_rates"]) == 8
    assert {
        len(row["area_ids"]) for row in by_reason["council_tax_ni_domestic_rates"]
    } == {11}
    assert by_reason["council_tax_city_of_london_band_a_suppressed"][0]["area_ids"] == [
        "E09000001"
    ]
    assert by_reason["council_tax_wales_band_h_absent"][0]["area_ids"] == [
        "W06000019",
        "W06000024",
    ]


def test_council_tax_activation_adds_2541_references() -> None:
    membership = _membership()
    active = 0
    for band in "abcdefgh":
        target_id = f"voa.council_tax_stock.by_area.band_{band}"
        candidates = membership["targets"][target_id]["geography_levels"][
            "local_authority"
        ]["candidates"]
        active += sum(row["status"] == "active" for row in candidates)
    assert active == 2_541
