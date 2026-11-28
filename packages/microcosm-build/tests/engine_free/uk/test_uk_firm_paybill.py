"""UK firm paybill: headcount x ONS ASHE section mean annual gross pay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_paybill import (
    UK_FIRM_PAYBILL_ALL_EMPLOYEES,
    UK_FIRM_PAYBILL_FILE,
    UK_FIRM_PAYBILL_LEVELS,
    UK_FIRM_PAYBILL_SOURCE_COLUMNS,
    UK_FIRM_PAYBILL_TAX_YEAR,
    assign_uk_firm_paybill,
    read_uk_firm_paybill_source,
    uk_firm_paybill_fallback_counts,
    uk_firm_paybill_mean_pay,
)
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_EXTENDED_TABLE_COLUMNS,
    UK_FIRM_PAYBILL_TABLE_COLUMNS,
    to_policyengine_uk_firm_table,
    validate_uk_firm_table,
)
from microcosm.build.uk_runtime.occupation_constants import (
    UK_SIC2007_SECTION_DIVISIONS,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
PAYBILL = PATHS.tests / "fixtures" / "uk" / "firm_paybill_2024"
TOOL = PATHS.repository / "tools" / "extract_uk_firm_paybill_table.py"

#: ASHE 2025 Table 4.7a, sheet All, column Mean, read off the workbook.
_SPOT_MEANS = {
    "C": 43879.2,
    "G": 31679.1,
    "I": 21528.1,
    "K": 84079.9,
    "Q": 34429.7,
    "T": 16061.1,
}
_ALL_EMPLOYEES_MEAN = 40444.9


@cache
def _source():
    return read_uk_firm_paybill_source(PAYBILL)


@cache
def _provenance() -> dict:
    return json.loads((PAYBILL / ".provenance.json").read_text(encoding="utf-8"))


@cache
def _load_tool():
    if "extract_uk_firm_paybill_table" in sys.modules:
        return sys.modules["extract_uk_firm_paybill_table"]
    spec = importlib.util.spec_from_file_location("extract_uk_firm_paybill_table", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixture_reads_as_the_published_section_means() -> None:
    source = _source()

    assert set(source.section_mean) == set(UK_SIC2007_SECTION_DIVISIONS) - {"U"}
    assert dict(source.unpublished) == {"U": "x"}
    for section, mean in _SPOT_MEANS.items():
        assert source.section_mean[section] == mean
    assert source.all_employees_mean == _ALL_EMPLOYEES_MEAN
    assert all(mean > 0 for mean in source.section_mean.values())


def test_provenance_pins_the_source_and_the_csv() -> None:
    provenance = _provenance()
    payload = (PAYBILL / UK_FIRM_PAYBILL_FILE).read_bytes()
    record = provenance["files"][UK_FIRM_PAYBILL_FILE]

    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["sha256"] == _source().sha256
    assert record["bytes"] == len(payload)
    assert tuple(record["columns"]) == UK_FIRM_PAYBILL_SOURCE_COLUMNS
    source = provenance["source"]
    assert source["download_url"].startswith("https://www.ons.gov.uk/file?uri=")
    assert source["download_url"].endswith("/2025revised/ashetable42025revised.zip")
    assert source["edition"] == "2025 revised"
    assert source["sheet"] == "All"
    assert source["access_date"] == "2026-11-22"
    assert len(source["zip_sha256"]) == 64
    assert len(source["member_sha256"]) == 64
    assert "Table 4.7a" in source["member"]
    assert UK_FIRM_PAYBILL_TAX_YEAR == "2024-25"
    assert UK_FIRM_PAYBILL_TAX_YEAR in provenance["reference_period"]
    assert provenance["unpublished"]["markers"] == {"U": "x"}


def test_tool_access_date_defaults_to_the_recorded_one() -> None:
    """``--access-date`` defaults to the fixture's date, so a re-extraction
    without it reproduces the committed provenance; a bad date is refused."""

    tool = _load_tool()
    assert tool.ACCESS_DATE == _provenance()["source"]["access_date"]
    assert tool._iso_date("2027-01-05") == "2027-01-05"
    with pytest.raises(argparse.ArgumentTypeError, match="YYYY-MM-DD"):
        tool._iso_date("5 January 2027")


def test_tool_rebuilds_the_csv_from_sheet_rows() -> None:
    """The extractor on rows shaped like the 2025 sheet writes the fixture's
    layout, keeps an ``x`` section blank and refuses a missing section."""

    tool = _load_tool()
    rows: list[tuple] = [
        (
            "SIC2007 Table 4 .7a   Annual pay - Gross (£) - For all employee "
            "jobsa: United Kingdom 2025",
        )
        + (None,) * 7,
        (None,) * 8,
        (None, None, "Number", None, "Annual", None, "Annual", None),
        (None, None, "of jobsb", None, "percentage", None, "percentage", "P"),
        ("Description", "Code", "(thousand)", "Median", "change", "Mean", "change", 10),
        ("ALL EMPLOYEES", None, 24889, 32856.6, 4, 40444.9, 6, 1.0),
    ]
    for section in UK_SIC2007_SECTION_DIVISIONS:
        if section == "U":
            rows.append(("EXTRATERRITORIAL ", "'U ", "x", "x", None, "x", None, "x"))
        else:
            rows.append((f"{section} SECTION ", f"'{section} ", 10, 1.0, 0, 2.5, 0, 0))
        rows.append(("A division ", "'01", 5, 1.0, 0, 9.0, 0, 0))
    table = tool.extract_rows(rows)
    frame = pd.read_csv(
        io.BytesIO(table.to_csv_bytes()),
        dtype=str,
        keep_default_na=False,
    )

    assert tuple(frame.columns) == UK_FIRM_PAYBILL_SOURCE_COLUMNS
    assert frame["sic_section"].tolist() == [
        *UK_SIC2007_SECTION_DIVISIONS,
        UK_FIRM_PAYBILL_ALL_EMPLOYEES,
    ]
    by_section = frame.set_index("sic_section")
    assert by_section.loc["U", "mean_annual_gross_pay_gbp"] == ""
    assert by_section.loc["U", "marker"] == "x"
    assert by_section.loc["A", "mean_annual_gross_pay_gbp"] == "2.5"
    assert by_section.loc["ALL", "mean_annual_gross_pay_gbp"] == "40444.9"
    with pytest.raises(ValueError, match="sections"):
        tool.extract_rows([row for row in rows if row[1] != "'Q "])


def _write_source(tmp_path: Path, edit) -> Path:
    frame = pd.read_csv(
        PAYBILL / UK_FIRM_PAYBILL_FILE, dtype=str, keep_default_na=False
    )
    frame = edit(frame)
    path = tmp_path / UK_FIRM_PAYBILL_FILE
    frame.to_csv(path, index=False)
    return path


def _set(section: str, column: str, value: str):
    def edit(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame["sic_section"] == section, column] = value
        return frame

    return edit


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda f: f[f["sic_section"] != "Q"], "sections A-U"),
        (lambda f: f[f["sic_section"] != "ALL"], "sections A-U"),
        (lambda f: pd.concat([f, f.iloc[[0]]]), "sections A-U"),
        (_set("U", "mean_annual_gross_pay_gbp", "1000"), "has a mean and marker"),
        (_set("C", "mean_annual_gross_pay_gbp", "0"), "must be positive"),
        (_set("C", "mean_annual_gross_pay_gbp", "n/a"), "not a number"),
        (_set("ALL", "marker", "x"), "has a mean and marker"),
        (lambda f: f.drop(columns="marker"), "columns"),
    ],
)
def test_source_refuses_a_broken_table(tmp_path: Path, edit, match) -> None:
    path = _write_source(tmp_path, edit)
    with pytest.raises(ValueError, match=match):
        read_uk_firm_paybill_source(path)


def test_source_refuses_a_missing_all_employees_mean(tmp_path: Path) -> None:
    def edit(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame["sic_section"] == "ALL", "mean_annual_gross_pay_gbp"] = ""
        frame.loc[frame["sic_section"] == "ALL", "marker"] = ":"
        return frame

    with pytest.raises(ValueError, match="all-employees"):
        read_uk_firm_paybill_source(_write_source(tmp_path, edit))


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firm_employees": np.array([3, 0, 2, 5, 1], dtype=np.int64),
            "firm_sic_section": ["C", "K", "U", "UNKNOWN", "I"],
            "firm_weight": [2.0, 1.0, 1.0, 0.5, 4.0],
        }
    )


def test_paybill_is_headcount_times_section_mean() -> None:
    source = _source()
    paybill = assign_uk_firm_paybill(_table(), source)

    assert paybill.dtype == np.float64
    np.testing.assert_allclose(
        paybill,
        [
            3 * 43879.2,
            0.0,
            # U is unpublished and UNKNOWN has no row: the all-employees mean.
            2 * _ALL_EMPLOYEES_MEAN,
            5 * _ALL_EMPLOYEES_MEAN,
            1 * 21528.1,
        ],
    )
    np.testing.assert_allclose(
        uk_firm_paybill_mean_pay(_table(), source),
        [43879.2, 84079.9, _ALL_EMPLOYEES_MEAN, _ALL_EMPLOYEES_MEAN, 21528.1],
    )


def test_generator_rows_map_divisions_to_sections() -> None:
    firms = pd.DataFrame(
        {
            "employment": np.array([4, 1, 2]),
            # Divisions 64 (K), 99 (U) and 0 (no section).
            "sic_code": ["00064", "00099", "00000"],
        }
    )
    np.testing.assert_allclose(
        assign_uk_firm_paybill(firms, _source()),
        [4 * 84079.9, _ALL_EMPLOYEES_MEAN, 2 * _ALL_EMPLOYEES_MEAN],
    )


def test_fallback_counts_report_each_level() -> None:
    counts = uk_firm_paybill_fallback_counts(_table(), _source())

    assert tuple(counts["level"]) == UK_FIRM_PAYBILL_LEVELS
    by_level = counts.set_index("level")
    assert by_level.loc["section", "firms"] == 3
    assert by_level.loc["section", "firms_with_employees"] == 2
    assert by_level.loc["all_employees", "firms"] == 2
    assert by_level.loc["all_employees", "weight"] == pytest.approx(1.5)
    assert by_level.loc["all_employees", "weighted_paybill_gbp"] == pytest.approx(
        2 * _ALL_EMPLOYEES_MEAN + 0.5 * 5 * _ALL_EMPLOYEES_MEAN
    )
    assert counts["weighted_paybill_gbp"].sum() == pytest.approx(
        float(
            (
                _table()["firm_weight"] * assign_uk_firm_paybill(_table(), _source())
            ).sum()
        )
    )


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("firm_employees", -1, "non-negative whole"),
        ("firm_employees", 1.5, "non-negative whole"),
        ("firm_sic_section", "Z", "SICSection"),
    ],
)
def test_paybill_refuses_bad_rows(column, value, match) -> None:
    table = _table().astype({column: object})
    table.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        assign_uk_firm_paybill(table, _source())


def test_paybill_needs_headcount_and_section() -> None:
    with pytest.raises(ValueError, match="needs firm_employees"):
        assign_uk_firm_paybill(_table().drop(columns="firm_sic_section"), _source())


def _attributed_firms() -> pd.DataFrame:
    """Generator rows out of firm_id order, with ONS attribute draws."""

    return pd.DataFrame(
        {
            "firm_id": np.array([4, 2, 1, 3], dtype=np.int64),
            "source_firm_id": np.array([4, 2, 1, 3], dtype=np.int64),
            "source_firm_key": [
                f"2024-25:experimental_uk_firm:{i}" for i in (4, 2, 1, 3)
            ],
            "data_vintage": "2024-25",
            "sic_code": ["00011", "00047", "00004", "00086"],
            "annual_turnover_k": np.array([120.0, 90.0, 0.0, 40.0]),
            "annual_input_k": np.array([80.0, 30.0, 0.0, 10.0]),
            "employment": np.array([12, 3, 0, 1]),
            "firm_weight": [2.5, 10.0, 0.0, 4.0],
            "vat_scope": [True, True, False, False],
            "vat_registered": [True, True, False, True],
            "in_frame": [True, True, False, True],
            "base_weight": [1.0, 2.0, 1.0, 1.0],
            "ons_legal_status": np.array([1, 2, 5, 7], dtype=np.int64),
            "ons_region_code": ["E12000007", "W92000004", "S92000003", "N92000002"],
        }
    )


def test_export_carries_paybill_aligned_by_firm_id() -> None:
    firms = _attributed_firms()
    firms["firm_paybill"] = assign_uk_firm_paybill(firms, _source())
    table = to_policyengine_uk_firm_table(firms, threshold_k=90.0)

    assert tuple(table.columns) == UK_FIRM_PAYBILL_TABLE_COLUMNS
    validate_uk_firm_table(table)
    # Ordered by firm_id: 1 (division 4, B, no employees), 2 (47, G),
    # 3 (86, Q), 4 (11, C).
    np.testing.assert_allclose(
        table["firm_paybill"],
        [0.0, 3 * 31679.1, 1 * 34429.7, 12 * 43879.2],
    )
    np.testing.assert_allclose(
        table["firm_paybill"], assign_uk_firm_paybill(table, _source())
    )
    pd.testing.assert_frame_equal(
        table[list(UK_FIRM_EXTENDED_TABLE_COLUMNS)],
        to_policyengine_uk_firm_table(_attributed_firms(), threshold_k=90.0),
    )


def test_export_refuses_paybill_without_attributes() -> None:
    firms = _attributed_firms().drop(columns=["ons_legal_status", "ons_region_code"])
    firms["firm_paybill"] = 0.0
    with pytest.raises(ValueError, match="firm_paybill without"):
        to_policyengine_uk_firm_table(firms, threshold_k=90.0)


@pytest.mark.parametrize(
    ("row", "value", "match"),
    [
        (1, -1.0, "finite and non-negative"),
        (1, np.inf, "finite and non-negative"),
        (0, 5.0, "zero without employees"),
    ],
)
def test_validator_checks_the_paybill(row, value, match) -> None:
    firms = _attributed_firms()
    firms["firm_paybill"] = assign_uk_firm_paybill(firms, _source())
    table = to_policyengine_uk_firm_table(firms, threshold_k=90.0)
    table.loc[row, "firm_paybill"] = value
    with pytest.raises(ValueError, match=match):
        validate_uk_firm_table(table)
