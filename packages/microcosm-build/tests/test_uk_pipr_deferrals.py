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


def _references() -> list[dict]:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_target_references.json")
        .read_text()
    )["target_references"]


def test_pipr_2025_surface_activates_available_calendar_year_averages() -> None:
    target = _membership()["targets"]["ons.rent.private_rent"]
    candidates = target["geography_levels"]["local_authority"]["candidates"]
    assert len(candidates) == 361
    assert sum(row["status"] == "active" for row in candidates) == 314
    assert (
        sum(row["status"] == "no_fact_at_or_before_period" for row in candidates) == 0
    )
    assert sum(row["status"] == "no_fact_for_area" for row in candidates) == 47
    references = [
        row
        for row in _references()
        if row["metadata"]["contract_target_id"] == "ons.rent.private_rent"
    ]
    assert len(references) == 314
    assert {row["value_operation"] for row in references} == {"calendar_year_average"}
    assert {row["metadata"]["fact_aggregation"] for row in references} == {"time_mean"}
    assert all(
        "Calendar-year average" in row["metadata"]["period_basis_note"]
        for row in references
    )


def test_pipr_three_signed_absence_reasons_pin_crosswalk_measurements() -> None:
    rows = {
        row["reason_id"]: row
        for row in _membership()["signed_deferrals"]
        if row["reason_id"].startswith("private_rent_pipr_")
    }
    assert set(rows) == {
        "private_rent_pipr_english_lad_absent",
        "private_rent_pipr_scotland_brma_grain",
        "private_rent_pipr_ni_absent",
    }
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


def test_deferral_for_unknown_contract_target_refuses() -> None:
    contract = copy.deepcopy(load_uk_population_contract())
    contract["targets"] = [
        target
        for target in contract["targets"]
        if target["target_id"] != "ons.rent.private_rent"
    ]

    with pytest.raises(ValueError, match="ons.rent.private_rent.*absent"):
        _area_signed_deferrals(contract, _crosswalk())
