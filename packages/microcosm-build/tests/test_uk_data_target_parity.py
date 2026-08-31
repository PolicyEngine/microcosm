from __future__ import annotations

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
