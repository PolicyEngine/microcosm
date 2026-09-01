from __future__ import annotations

import copy

import pytest

from microcosm.build.uk_runtime import data_target_parity
from microcosm.build.uk_runtime.data_target_parity import (
    PORTED_PARITY_STATUSES,
    VALID_PARITY_STATUSES,
    assert_uk_data_target_parity_current,
    build_uk_data_target_parity,
)


def test_uk_data_target_parity_is_current() -> None:
    assert_uk_data_target_parity_current()


def test_every_parity_row_is_complete() -> None:
    rows = build_uk_data_target_parity()["concerns"]
    assert len(rows) == len({row["concern_id"] for row in rows})
    for row in rows:
        assert row["status"] in VALID_PARITY_STATUSES
        assert row["classification"]
        assert row["evidence"]
        assert row["covers"]
        if row["status"] not in PORTED_PARITY_STATUSES:
            assert row["reason"]
            assert set(row["fence"]) == {"origin", "purpose", "verdict_basis"}
            assert all(row["fence"].values())


def test_red_line_rows_are_present() -> None:
    rows = {row["concern_id"]: row for row in build_uk_data_target_parity()["concerns"]}
    for concern_id in (
        "national_obr_national_insurance",
        "national_obr_council_tax",
        "national_ons_land_values",
        "national_council_tax_stock",
    ):
        assert rows[concern_id]["status"] == "ported_national"
    local_rows = [
        row
        for row in rows.values()
        if row["classification"] == "red_line_local_contract_target"
    ]
    assert len(local_rows) >= 25
    assert all(row["status"].startswith("ported_local_") for row in local_rows)


def test_post_ebf733c_additions_are_named() -> None:
    rows = build_uk_data_target_parity()["concerns"]
    dwp = next(
        row
        for row in rows
        if row["concern_id"] == "national_dwp_benefits_and_uc_caseloads"
    )
    education = next(
        row
        for row in rows
        if row["concern_id"] == "dataset_anchor_dfe_education_spending"
    )
    assert "uk-data#458" in dwp["evidence"]
    assert "uk-data#474" in education["reason"]


def test_inventory_and_concern_coverage_form_a_bijection() -> None:
    rows = build_uk_data_target_parity()["concerns"]
    inventory = data_target_parity._load_uk_data_target_inventory()

    data_target_parity._assert_inventory_bijection(rows, inventory)


def test_concern_covering_unknown_inventory_id_fails() -> None:
    rows = copy.deepcopy(build_uk_data_target_parity()["concerns"])
    rows[0]["covers"] = (*rows[0]["covers"], "unknown.inventory.entry")
    inventory = data_target_parity._load_uk_data_target_inventory()

    with pytest.raises(ValueError, match="unknown.inventory.entry"):
        data_target_parity._assert_inventory_bijection(rows, inventory)


def test_inventory_entry_with_no_concern_fails() -> None:
    rows = copy.deepcopy(build_uk_data_target_parity()["concerns"])
    inventory = data_target_parity._load_uk_data_target_inventory()
    covered = {inventory_id for row in rows for inventory_id in row["covers"]}
    victim = next(
        entry["inventory_id"]
        for entry in inventory["entries"]
        if entry["inventory_id"] in covered
        and entry["inventory_id"]
        not in data_target_parity.UK_DATA_TARGET_INVENTORY_HELPER_EXEMPTIONS
    )
    for row in rows:
        row["covers"] = tuple(
            inventory_id for inventory_id in row["covers"] if inventory_id != victim
        )

    with pytest.raises(ValueError, match=victim):
        data_target_parity._assert_inventory_bijection(rows, inventory)


def test_evidence_never_cites_untracked_codex_work() -> None:
    rows = build_uk_data_target_parity()["concerns"]

    assert all(".codex-work" not in row["evidence"] for row in rows)
