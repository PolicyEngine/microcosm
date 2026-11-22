"""The ONS UK Business 2025 attribute fixtures for the experimental firm table.

``tests/fixtures/uk/firm_attributes_2025`` holds long-format enterprise counts
from Tables 8, 13 and 15 of the ONS workbook tracked in firm-microsim-paper,
written by ``tools/extract_uk_firm_ons_attribute_tables.py``. These checks read
only the committed CSVs and ``.provenance.json``; they never open the workbook.

ONS control-rounds every cell to base 5, so the tables are checked against the
paper's UK totals within a tolerance, never for exact equality with each other.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from typing import Any

import pandas as pd
import pytest

from microcosm.build.uk_runtime.incumbent_surface_evaluation import GSS_REGION_CODES
from test_support.paths import paths_for

FIXTURES = paths_for("microcosm-build").tests / "fixtures" / "uk"
FIXTURE_DIR = FIXTURES / "firm_attributes_2025"
ONS_TURNOVER = FIXTURES / "firm_2024_25" / "ons_firm_turnover.csv"
RELATIVE_TOLERANCE = 0.01

REGION_CODES = (
    "E12000001",
    "E12000002",
    "E12000003",
    "E12000004",
    "E12000005",
    "E12000006",
    "E12000007",
    "E12000008",
    "E12000009",
    "W92000004",
    "S92000003",
    "N92000002",
)
TABLE_8_BANDS = ("0-49", "50-99", "100-249", "250-499", "500-999", "1000-4999", "5000+")
TABLE_13_BANDS = (
    "0-49",
    "50-99",
    "100-249",
    "250-499",
    "500-999",
    "1000-1999",
    "2000-4999",
    "5000-9999",
    "10000-49999",
    "50000+",
)
LEGAL_STATUSES = tuple(range(1, 8))
INDUSTRY_GROUPS = (
    "01-03",
    "05-39",
    "41-43",
    "45",
    "46",
    "47",
    "49-53",
    "55-56",
    "58-63",
    "64-66",
    "68",
    "69-75",
    "77-82",
    "84",
    "85",
    "86-88",
    "90-99",
)

FILES: dict[str, dict[str, Any]] = {
    "ons_region_turnover.csv": {
        "columns": ["sic_division", "region_code", "turnover_band", "enterprises"],
        "keys": {
            "sic_division": None,
            "region_code": REGION_CODES,
            "turnover_band": TABLE_8_BANDS,
        },
        "rows": 88 * len(REGION_CODES) * len(TABLE_8_BANDS),
    },
    "ons_legal_status_turnover.csv": {
        "columns": ["industry_group", "legal_status", "turnover_band", "enterprises"],
        "keys": {
            "industry_group": INDUSTRY_GROUPS,
            "legal_status": LEGAL_STATUSES,
            "turnover_band": TABLE_13_BANDS,
        },
        "rows": len(INDUSTRY_GROUPS) * len(LEGAL_STATUSES) * len(TABLE_13_BANDS),
    },
    "ons_legal_status_region_turnover.csv": {
        "columns": ["region_code", "legal_status", "turnover_band", "enterprises"],
        "keys": {
            "region_code": REGION_CODES,
            "legal_status": LEGAL_STATUSES,
            "turnover_band": TABLE_13_BANDS,
        },
        "rows": len(REGION_CODES) * len(LEGAL_STATUSES) * len(TABLE_13_BANDS),
    },
}


@cache
def _provenance() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / ".provenance.json").read_text(encoding="utf-8"))


@cache
def _table(name: str) -> pd.DataFrame:
    frame = pd.read_csv(
        FIXTURE_DIR / name,
        dtype={
            "sic_division": str,
            "region_code": str,
            "industry_group": str,
            "turnover_band": str,
        },
    )
    return frame


@cache
def _uk_totals() -> pd.Series:
    """The ``Total`` row of the paper's ONS Table 8 extract, by sizeband."""

    turnover = pd.read_csv(ONS_TURNOVER, dtype={"SIC Code": str})
    total = turnover.loc[turnover["Description"] == "Total"]
    assert len(total) == 1
    return total.iloc[0][[*TABLE_8_BANDS, "Total"]].astype(int)


def _assert_close(actual: int, expected: int, label: str) -> None:
    assert expected > 0, label
    assert abs(actual - expected) <= RELATIVE_TOLERANCE * expected, (
        label,
        actual,
        expected,
    )


@pytest.mark.parametrize("name", sorted(FILES))
def test_fixture_bytes_match_provenance(name: str) -> None:
    payload = (FIXTURE_DIR / name).read_bytes()
    record = _provenance()["files"][name]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    assert len(payload) == record["bytes"]
    assert len(_table(name)) == record["rows"]


def test_provenance_records_the_source() -> None:
    provenance = _provenance()
    assert set(provenance["files"]) == set(FILES)
    assert provenance["workbook_tracked_at_paper_commit"] is True
    assert len(provenance["paper_commit"]) == 40
    assert len(provenance["workbook"]["sha256"]) == 64
    assert provenance["workbook"]["source_url"].startswith("https://www.ons.gov.uk/")
    assert list(provenance["region_codes"]) == list(REGION_CODES)
    assert list(provenance["legal_status"]) == [str(code) for code in LEGAL_STATUSES]
    assert list(provenance["industry_groups"]) == list(INDUSTRY_GROUPS)
    bands = provenance["turnover_bands"]
    assert bands["ons_region_turnover.csv"] == list(TABLE_8_BANDS)
    assert bands["ons_legal_status_turnover.csv"] == list(TABLE_13_BANDS)
    assert set(bands["table_13_to_table_8"]) == set(TABLE_13_BANDS)
    assert set(bands["table_13_to_table_8"].values()) == set(TABLE_8_BANDS)


@pytest.mark.parametrize("name", sorted(FILES))
def test_fixture_shape_and_keys(name: str) -> None:
    spec = FILES[name]
    frame = _table(name)
    assert list(frame.columns) == spec["columns"]
    assert len(frame) == spec["rows"]
    keys = list(spec["keys"])
    assert not frame.duplicated(keys).any()
    for column, values in spec["keys"].items():
        if values is not None:
            assert tuple(frame[column].drop_duplicates()) == values
    enterprises = frame["enterprises"]
    assert pd.api.types.is_integer_dtype(enterprises)
    assert (enterprises >= 0).all()
    assert not frame.isna().any().any()


def test_sic_divisions_match_the_paper_turnover_table() -> None:
    divisions = _table("ons_region_turnover.csv")["sic_division"].drop_duplicates()
    turnover = pd.read_csv(ONS_TURNOVER, dtype={"SIC Code": str})
    paper = turnover["SIC Code"].dropna()
    assert divisions.str.fullmatch(r"\d{2}").all()
    assert list(divisions) == list(paper)


def test_region_codes_are_canonical_gss_codes() -> None:
    assert set(REGION_CODES) <= set(GSS_REGION_CODES)
    assert not any(code.endswith("99999999") for code in REGION_CODES)


def test_regional_table_sums_track_uk_totals() -> None:
    frame = _table("ons_region_turnover.csv")
    totals = _uk_totals()
    by_band = frame.groupby("turnover_band")["enterprises"].sum()
    for band in TABLE_8_BANDS:
        _assert_close(int(by_band[band]), int(totals[band]), f"Table 8 {band}")
    _assert_close(
        int(frame["enterprises"].sum()), int(totals["Total"]), "Table 8 total"
    )


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("ons_legal_status_turnover.csv", "Table 13"),
        ("ons_legal_status_region_turnover.csv", "Table 15"),
    ],
)
def test_legal_status_table_sums_track_uk_totals(name: str, label: str) -> None:
    collapse = _provenance()["turnover_bands"]["table_13_to_table_8"]
    frame = _table(name)
    totals = _uk_totals()
    by_band = frame.groupby(frame["turnover_band"].map(collapse))["enterprises"].sum()
    for band in TABLE_8_BANDS:
        _assert_close(int(by_band[band]), int(totals[band]), f"{label} {band}")
    _assert_close(
        int(frame["enterprises"].sum()), int(totals["Total"]), f"{label} total"
    )
