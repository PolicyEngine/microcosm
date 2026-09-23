from __future__ import annotations

import copy
import os
from pathlib import Path

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


def test_inventory_entries_carry_well_formed_hashes_and_date() -> None:
    inventory = data_target_parity._load_uk_data_target_inventory()
    assert inventory["extracted_on"]
    for entry in inventory["entries"]:
        assert len(entry["sha256"]) == 64
        assert set(entry["sha256"]) <= set("0123456789abcdef")


def test_inventory_loader_refuses_a_malformed_hash(tmp_path: Path) -> None:
    import json

    inventory = data_target_parity._load_uk_data_target_inventory()
    inventory["entries"][0]["sha256"] = "not-a-digest"
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="malformed sha256"):
        data_target_parity._load_uk_data_target_inventory(path)


def test_inventory_hashes_match_a_local_incumbent_tree() -> None:
    """The recorded hashes are read here, against a real checkout.

    Opt-in: the incumbent tree is not a repository input, so this test skips
    unless MICROCOSM_UK_DATA_TREE names a checkout at the pinned commit. When
    it runs it is the only continuous re-verification of the snapshot; the
    committed artifact otherwise proves that the concerns cover the snapshot,
    not that the snapshot still matches uk-data.
    """

    tree = os.environ.get("MICROCOSM_UK_DATA_TREE")
    if not tree:
        pytest.skip("set MICROCOSM_UK_DATA_TREE to a uk-data checkout to re-hash")
    data_target_parity.verify_uk_data_target_inventory_against_tree(tree)


def test_tree_verifier_refuses_a_drifted_file(tmp_path: Path) -> None:
    import json

    inventory = data_target_parity._load_uk_data_target_inventory()
    entry = inventory["entries"][0]
    (tmp_path / entry["path"]).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / entry["path"]).write_text("# drifted\n")
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        data_target_parity.verify_uk_data_target_inventory_against_tree(tmp_path, path)


def test_la_target_producers_are_inventoried_and_covered() -> None:
    """The LA twins of the constituency producer scripts cannot go silent."""

    inventory = data_target_parity._load_uk_data_target_inventory()
    ids = {entry["inventory_id"] for entry in inventory["entries"]}
    for grain in ("constituencies", "local_authorities"):
        for script in (
            "create_employment_incomes",
            "create_total_incomes",
            "fill_missing_age_demographics",
        ):
            assert (
                data_target_parity._inventory_id(
                    f"datasets/local_areas/{grain}/targets/{script}.py"
                )
                in ids
            )
