from __future__ import annotations

import copy
import json
from importlib import resources as importlib_resources

import pytest

from microcosm.build.uk_runtime.local_targets import load_uk_population_contract
from tools.generate_uk_local_target_references import _area_signed_deferrals


def _crosswalk() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text()
    )


def _membership() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_target_reference_membership.json")
        .read_text()
    )


def test_council_tax_band_cells_activate_and_defer_as_measured() -> None:
    membership = _membership()
    expected_active = {
        "a": 294,
        "b": 294,
        "c": 294,
        "d": 294,
        "e": 294,
        "f": 294,
        "g": 294,
        "h": 0,
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
    wales = by_reason["council_tax_wales_country_control_absent"]
    expected_wales = tuple(
        area_id
        for area_id in _crosswalk()["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("W")
    )
    assert len(wales) == 8
    assert {row["target_id"] for row in wales} == {
        f"voa.council_tax_stock.by_area.band_{band}" for band in "abcdefgh"
    }
    assert {tuple(row["area_ids"]) for row in wales} == {expected_wales}
    assert len(expected_wales) == 22
    assert {row["defer_if_compiles"] for row in wales} == {True}
    assert {
        "no Wales country-level council-tax stock-by-band fact" in row["rationale"]
        for row in wales
    } == {True}
    band_h = by_reason["council_tax_band_h_spine_support_absent"]
    expected_english = tuple(
        area_id
        for area_id in _crosswalk()["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("E")
    )
    assert len(band_h) == 1
    assert tuple(band_h[0]["area_ids"]) == expected_english
    assert len(expected_english) == 296
    assert band_h[0]["defer_if_compiles"] is True
    assert "170 band-H households from 49 raw FRS households" in band_h[0]["rationale"]
    assert "84 of 296 authorities" in band_h[0]["rationale"]
    assert "council_tax_wales_band_h_absent" not in by_reason


def test_council_tax_activation_adds_2058_references() -> None:
    membership = _membership()
    active = 0
    for band in "abcdefgh":
        target_id = f"voa.council_tax_stock.by_area.band_{band}"
        candidates = membership["targets"][target_id]["geography_levels"][
            "local_authority"
        ]["candidates"]
        active += sum(row["status"] == "active" for row in candidates)
    assert active == 2_058


def test_support_floor_deferrals_cover_the_two_authorities_remaining_cells() -> None:
    rows = [
        row
        for row in _membership()["signed_deferrals"]
        if row["reason_id"] == "local_authority_support_floor_excluded"
    ]
    assert len(rows) == 24
    assert sum(len(row["area_ids"]) for row in rows) == 43
    assert {
        area_id: sum(area_id in row["area_ids"] for row in rows)
        for area_id in ("E06000053", "E09000001")
    } == {"E06000053": 20, "E09000001": 23}
    assert all(row["defer_if_compiles"] is True for row in rows)
    assert all(not row["target_id"].endswith("band_h") for row in rows)
    assert {
        "their rows stay in the solve through the constituency families and "
        "the national rows" in row["rationale"]
        for row in rows
    } == {True}


def test_a14_deferral_declarations_cover_only_currently_active_cells() -> None:
    declarations = _area_signed_deferrals(load_uk_population_contract(), _crosswalk())
    band_h = [
        row
        for row in declarations
        if row.reason_id == "council_tax_band_h_spine_support_absent"
    ]
    expected_english = tuple(
        area_id
        for area_id in _crosswalk()["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("E")
    )
    assert len(band_h) == 1
    assert band_h[0].target_id == "voa.council_tax_stock.by_area.band_h"
    assert band_h[0].area_ids == expected_english
    assert len(expected_english) == 296
    assert band_h[0].defer_if_compiles is True

    support = [
        row
        for row in declarations
        if row.reason_id == "local_authority_support_floor_excluded"
    ]
    assert len(support) == 24
    assert sum(len(row.area_ids) for row in support) == 43
    by_area = {
        area_id: sum(area_id in row.area_ids for row in support)
        for area_id in ("E06000053", "E09000001")
    }
    assert by_area == {"E06000053": 20, "E09000001": 23}
    assert all(row.defer_if_compiles for row in support)
    assert all(not row.target_id.endswith("band_h") for row in support)


def test_declared_deferral_roster_matching_no_crosswalk_area_refuses() -> None:
    crosswalk = copy.deepcopy(_crosswalk())
    area_ids = crosswalk["levels"]["local_authority"]["area_ids"]
    area_ids.remove("E09000001")
    area_ids.append("E09999999")

    with pytest.raises(ValueError, match="unmatched area id.*E09000001"):
        _area_signed_deferrals(load_uk_population_contract(), crosswalk)


@pytest.mark.parametrize(
    ("prefix", "expected", "name"),
    [
        ("S", 32, "Scottish"),
        ("N", 11, "Northern Ireland"),
        ("W", 22, "Welsh"),
        ("E", 296, "English"),
    ],
)
def test_council_tax_country_masks_refuse_roster_count_drift(
    prefix: str, expected: int, name: str
) -> None:
    crosswalk = copy.deepcopy(_crosswalk())
    area_ids = crosswalk["levels"]["local_authority"]["area_ids"]
    area_ids.remove(next(area_id for area_id in area_ids if area_id.startswith(prefix)))

    with pytest.raises(
        ValueError,
        match=rf"{name}.*expected {expected}.*measured {expected - 1}",
    ):
        _area_signed_deferrals(load_uk_population_contract(), crosswalk)
