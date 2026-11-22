#!/usr/bin/env python3
"""Extract ONS UK Business 2025 attribute tables for the experimental UK firm table.

Reads three sheets of the ONS *UK Business: Activity, Size and Location, 2025*
workbook (``ukbusinessworkbook2025new.xlsx``, tracked in
PolicyEngine/firm-microsim-paper under ``data/raw/ons/``) and writes long-format
CSVs of enterprise counts:

* ``ons_region_turnover.csv`` from **Table 8**: SIC division x region x the
  seven ONS turnover sizebands (``sic_division,region_code,turnover_band,
  enterprises``).
* ``ons_legal_status_turnover.csv`` from **Table 13**: broad industry group x
  legal status x the ten Table 13 turnover sizebands (``industry_group,
  legal_status,turnover_band,enterprises``).
* ``ons_legal_status_region_turnover.csv`` from **Table 15**: region x legal
  status x the ten sizebands (``region_code,legal_status,turnover_band,
  enterprises``). Held out: not a draw input.

Each CSV is the full rectangular grid in workbook order, zero cells included.
The aggregate geographies (United Kingdom, Great Britain, England and Wales,
England), every ``Total`` column and row and Table 13/15's ``Total`` legal
status block are dropped. Counts are control-rounded to base 5 by ONS (note 7:
values may be rounded down to zero, so a zero is not necessarily a true zero).
A blank or non-numeric cell inside a table's data block is written as 0 and
counted in ``.provenance.json``; the 2025 workbook has none.

Region codes are the GSS codes from the workbook's own headers (E12000001 to
E12000009, W92000004, S92000003, N92000002). ``rowwise_geography`` writes
placeholder codes for the devolved nations (``W99999999`` and so on), so its
map is not reused here; the engine-free fixture test checks the codes against
``incumbent_surface_evaluation.GSS_REGION_CODES``.

openpyxl is imported only when a workbook is read, so engine-free CI, which
runs the fixture test against the committed CSVs, never needs it.

Usage::

    uv run --no-sync python tools/extract_uk_firm_ons_attribute_tables.py \\
        --workbook ../firm-microsim-paper/data/raw/ons/ukbusinessworkbook2025new.xlsx

    # Exit 1 if re-extraction would change any committed CSV byte.
    uv run --no-sync python tools/extract_uk_firm_ons_attribute_tables.py \\
        --workbook ../firm-microsim-paper/data/raw/ons/ukbusinessworkbook2025new.xlsx \\
        --check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-build"
    / "tests"
    / "fixtures"
    / "uk"
    / "firm_attributes_2025"
)
PROVENANCE_NAME = ".provenance.json"

SOURCE_URL = (
    "https://www.ons.gov.uk/businessindustryandtrade/business/"
    "activitysizeandlocation/datasets/ukbusinessactivitysizeandlocation"
)
SOURCE_URL_NOTE = (
    "ONS dataset landing page, as recorded for the 2025 edition in "
    "firm-microsim-paper data/README.md; not a direct file link."
)
PAPER_REPOSITORY = "PolicyEngine/firm-microsim-paper"
WORKBOOK_SOURCE_PATH = "data/raw/ons/ukbusinessworkbook2025new.xlsx"

#: The 12 UK regions in workbook order, with their GSS codes.
REGIONS: tuple[tuple[str, str], ...] = (
    ("E12000001", "North East"),
    ("E12000002", "North West"),
    ("E12000003", "Yorkshire and The Humber"),
    ("E12000004", "East Midlands"),
    ("E12000005", "West Midlands"),
    ("E12000006", "East"),
    ("E12000007", "London"),
    ("E12000008", "South East"),
    ("E12000009", "South West"),
    ("W92000004", "Wales"),
    ("S92000003", "Scotland"),
    ("N92000002", "Northern Ireland"),
)
REGION_CODES = tuple(code for code, _ in REGIONS)
AGGREGATE_GEOGRAPHY_CODES = ("K02000001", "K03000001", "K04000001", "E92000001")

#: Table 8's seven sizebands (GBP thousands), as in the paper's
#: ``ons_firm_turnover.csv`` columns.
TABLE_8_BANDS = ("0-49", "50-99", "100-249", "250-499", "500-999", "1000-4999", "5000+")
#: Tables 13 and 15 split Table 8's top two bands.
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
TABLE_13_TO_TABLE_8_BAND = {
    "0-49": "0-49",
    "50-99": "50-99",
    "100-249": "100-249",
    "250-499": "250-499",
    "500-999": "500-999",
    "1000-1999": "1000-4999",
    "2000-4999": "1000-4999",
    "5000-9999": "5000+",
    "10000-49999": "5000+",
    "50000+": "5000+",
}

#: ONS IDBR legal status codes, labels verbatim from Tables 13 and 15 row 4.
LEGAL_STATUSES: tuple[tuple[int, str], ...] = (
    (1, "Company (incl. Building Society)"),
    (2, "Sole Proprietor"),
    (3, "Partnership"),
    (4, "Public Corporation/Nationalised Body"),
    (5, "Central Government"),
    (6, "Local Authority"),
    (7, "Non-profit Body or Mutual Association"),
)

#: Table 13's broad industry groups (SIC 2007 division ranges), in order.
INDUSTRY_GROUPS: tuple[tuple[str, str], ...] = (
    ("01-03", "Agriculture, forestry & fishing"),
    ("05-39", "Production"),
    ("41-43", "Construction"),
    ("45", "Motor trades"),
    ("46", "Wholesale"),
    ("47", "Retail"),
    ("49-53", "Transport & Storage (inc. postal)"),
    ("55-56", "Accommodation & food services"),
    ("58-63", "Information & communication"),
    ("64-66", "Finance & insurance"),
    ("68", "Property"),
    ("69-75", "Professional, scientific & technical"),
    ("77-82", "Business administration & support services"),
    ("84", "Public administration & defence"),
    ("85", "Education"),
    ("86-88", "Health"),
    ("90-99", "Arts, entertainment, recreation & other services"),
)

SIC_DIVISION_COUNT = 88
GEOGRAPHY_ROW = 4
BAND_ROW = 5
FIRST_DATA_ROW = 6

REGION_TURNOVER = "ons_region_turnover.csv"
LEGAL_STATUS_TURNOVER = "ons_legal_status_turnover.csv"
LEGAL_STATUS_REGION_TURNOVER = "ons_legal_status_region_turnover.csv"


@dataclass
class Extracted:
    """One CSV's rows plus the source cells it read."""

    sheet: str
    title: str
    header: tuple[str, ...]
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    blank_cells: int = 0
    non_numeric_cells: dict[str, int] = field(default_factory=dict)
    zero_cells: int = 0

    def to_csv_bytes(self) -> bytes:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.header)
        writer.writerows(self.rows)
        return buffer.getvalue().encode("utf-8")


def _split_label(value: Any) -> tuple[str, str]:
    code, separator, label = str(value).strip().partition(" : ")
    if not separator:
        raise ValueError(f"expected 'CODE : label', got {value!r}")
    return code.strip(), label.strip()


def _forward_fill(row: Sequence[Any]) -> list[Any]:
    filled: list[Any] = []
    current = None
    for value in row:
        if value is not None:
            current = value
        filled.append(current)
    return filled


def _count(value: Any, table: Extracted) -> int:
    """Return a cell's count; blank or non-numeric cells are 0 and counted."""

    if value is None or (isinstance(value, str) and not value.strip()):
        table.blank_cells += 1
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        marker = str(value).strip()
        table.non_numeric_cells[marker] = table.non_numeric_cells.get(marker, 0) + 1
        return 0
    if value != int(value) or value < 0:
        raise ValueError(
            f"{table.sheet}: count {value!r} is not a non-negative integer"
        )
    count = int(value)
    if count == 0:
        table.zero_cells += 1
    return count


def _band_columns(
    rows: list[tuple[Any, ...]], sheet: str, bands: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Map each row-4 block header to ``{band: column index}``; checks bands."""

    blocks = _forward_fill(rows[GEOGRAPHY_ROW - 1])
    labels = rows[BAND_ROW - 1]
    columns: dict[str, dict[str, int]] = {}
    for index, (block, band) in enumerate(zip(blocks, labels, strict=False)):
        if index == 0 or block is None or band is None:
            continue
        columns.setdefault(str(block).strip(), {})[str(band).strip()] = index
    expected = [*bands, "Total"]
    for block, by_band in columns.items():
        if list(by_band) != expected:
            raise ValueError(f"{sheet} {block!r}: bands {list(by_band)} != {expected}")
    return columns


def _data_rows(rows: list[tuple[Any, ...]], sheet: str) -> list[tuple[Any, ...]]:
    """Rows from the first data row up to, not including, ``Total``/the end."""

    out = []
    for row in rows[FIRST_DATA_ROW - 1 :]:
        label = row[0]
        if label is None:
            break
        if str(label).strip() == "Total":
            break
        out.append(row)
    if not out:
        raise ValueError(f"{sheet}: no data rows")
    return out


def _title(rows: list[tuple[Any, ...]]) -> str:
    return str(rows[0][0]).strip()


def extract_region_turnover(rows: list[tuple[Any, ...]]) -> Extracted:
    sheet = "Table 8"
    table = Extracted(
        sheet=sheet,
        title=_title(rows),
        header=("sic_division", "region_code", "turnover_band", "enterprises"),
    )
    columns = _band_columns(rows, sheet, TABLE_8_BANDS)
    by_code = {_split_label(block)[0]: block for block in columns}
    expected_codes = [*AGGREGATE_GEOGRAPHY_CODES, *REGION_CODES]
    if list(by_code) != expected_codes:
        raise ValueError(
            f"{sheet}: geography headers {list(by_code)} != {expected_codes}"
        )
    for code, name in REGIONS:
        if _split_label(by_code[code])[1] != name:
            raise ValueError(
                f"{sheet}: {code} is labelled {by_code[code]!r}, not {name!r}"
            )
    data = _data_rows(rows, sheet)
    if len(data) != SIC_DIVISION_COUNT:
        raise ValueError(
            f"{sheet}: expected {SIC_DIVISION_COUNT} SIC divisions, got {len(data)}"
        )
    for row in data:
        division, _ = _split_label(row[0])
        if len(division) != 2 or not division.isdigit():
            raise ValueError(f"{sheet}: unexpected SIC division label {row[0]!r}")
        for code in REGION_CODES:
            for band in TABLE_8_BANDS:
                value = _count(row[columns[by_code[code]][band]], table)
                table.rows.append((division, code, band, value))
    return table


def _status_columns(
    rows: list[tuple[Any, ...]], sheet: str
) -> dict[int, dict[str, int]]:
    columns = _band_columns(rows, sheet, TABLE_13_BANDS)
    by_status: dict[int, dict[str, int]] = {}
    expected = [*(f"{code} - {label}" for code, label in LEGAL_STATUSES), "Total"]
    if list(columns) != expected:
        raise ValueError(f"{sheet}: legal status headers {list(columns)} != {expected}")
    for code, label in LEGAL_STATUSES:
        by_status[code] = columns[f"{code} - {label}"]
    return by_status


def extract_legal_status_turnover(rows: list[tuple[Any, ...]]) -> Extracted:
    sheet = "Table 13"
    table = Extracted(
        sheet=sheet,
        title=_title(rows),
        header=("industry_group", "legal_status", "turnover_band", "enterprises"),
    )
    columns = _status_columns(rows, sheet)
    data = _data_rows(rows, sheet)
    groups = [_split_label(row[0]) for row in data]
    if groups != list(INDUSTRY_GROUPS):
        raise ValueError(
            f"{sheet}: industry groups {groups} != {list(INDUSTRY_GROUPS)}"
        )
    for row, (group, _) in zip(data, groups, strict=True):
        for status, _label in LEGAL_STATUSES:
            for band in TABLE_13_BANDS:
                value = _count(row[columns[status][band]], table)
                table.rows.append((group, status, band, value))
    return table


def extract_legal_status_region_turnover(rows: list[tuple[Any, ...]]) -> Extracted:
    sheet = "Table 15"
    table = Extracted(
        sheet=sheet,
        title=_title(rows),
        header=("region_code", "legal_status", "turnover_band", "enterprises"),
    )
    columns = _status_columns(rows, sheet)
    data = _data_rows(rows, sheet)
    geographies = [_split_label(row[0])[0] for row in data]
    expected = [*AGGREGATE_GEOGRAPHY_CODES, *REGION_CODES]
    if geographies != expected:
        raise ValueError(f"{sheet}: geography rows {geographies} != {expected}")
    for row, code in zip(data, geographies, strict=True):
        if code in AGGREGATE_GEOGRAPHY_CODES:
            continue
        for status, _label in LEGAL_STATUSES:
            for band in TABLE_13_BANDS:
                value = _count(row[columns[status][band]], table)
                table.rows.append((code, status, band, value))
    return table


EXTRACTORS = {
    REGION_TURNOVER: ("Table 8", extract_region_turnover),
    LEGAL_STATUS_TURNOVER: ("Table 13", extract_legal_status_turnover),
    LEGAL_STATUS_REGION_TURNOVER: ("Table 15", extract_legal_status_region_turnover),
}


def extract(workbook: Path) -> dict[str, Extracted]:
    import openpyxl  # noqa: PLC0415 - local-only dependency, kept out of CI imports

    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        return {
            name: extractor(list(book[sheet].iter_rows(values_only=True)))
            for name, (sheet, extractor) in EXTRACTORS.items()
        }
    finally:
        book.close()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(workbook: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workbook.parent), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _paper_state(workbook: Path) -> dict[str, Any]:
    """The paper checkout's HEAD and whether the workbook is tracked, unmodified."""

    head = _git(workbook, "rev-parse", "HEAD")
    if head.returncode != 0:
        return {"paper_commit": None, "workbook_tracked_at_paper_commit": False}
    commit = head.stdout.strip()
    tracked = (
        _git(workbook, "ls-files", "--error-unmatch", workbook.name).returncode == 0
    )
    unmodified = (
        _git(workbook, "diff", "--quiet", "HEAD", "--", workbook.name).returncode == 0
    )
    return {
        "paper_commit": commit,
        "paper_commit_short": commit[:7],
        "workbook_tracked_at_paper_commit": tracked and unmodified,
    }


def provenance(
    workbook: Path, tables: dict[str, Extracted], payloads: dict[str, bytes]
) -> dict[str, Any]:
    data = workbook.read_bytes()
    files = {}
    for name, table in tables.items():
        files[name] = {
            "sha256": _sha256(payloads[name]),
            "bytes": len(payloads[name]),
            "rows": len(table.rows),
            "columns": list(table.header),
            "source_sheet": table.sheet,
            "source_title": table.title,
            "blank_cells_written_as_zero": table.blank_cells,
            "non_numeric_cells_written_as_zero": dict(
                sorted(table.non_numeric_cells.items())
            ),
            "zero_cells": table.zero_cells,
        }
    files[LEGAL_STATUS_REGION_TURNOVER]["role"] = (
        "held out: a check on drawn legal status by region, not a draw input"
    )
    return {
        "source_repository": PAPER_REPOSITORY,
        **_paper_state(workbook),
        "workbook": {
            "source_path": WORKBOOK_SOURCE_PATH,
            "sha256": _sha256(data),
            "bytes": len(data),
            "publication": "ONS, UK Business: Activity, Size and Location, 2025",
            "published": "2025-09-24",
            "source_url": SOURCE_URL,
            "source_url_note": SOURCE_URL_NOTE,
            "reference_period_note": (
                "Workbook note 4: turnover for most traders is from VAT returns "
                "covering the financial year 2023/2024."
            ),
        },
        "extraction_date": dt.date.today().isoformat(),
        "tool": "tools/extract_uk_firm_ons_attribute_tables.py",
        "command": (
            "uv run --no-sync python tools/extract_uk_firm_ons_attribute_tables.py "
            "--workbook <paper>/data/raw/ons/ukbusinessworkbook2025new.xlsx"
        ),
        "unit": "VAT and/or PAYE based enterprises",
        "disclosure": {
            "rounding": "Counts are control rounded to base 5 by ONS (workbook row 3).",
            "zeros": (
                "Workbook note 7: values may be rounded down to zero, so zeros "
                "are not necessarily true zeros; totals across tables may differ "
                "by minor amounts. Zeros are kept as 0."
            ),
            "suppressed_or_blank_cells": (
                "A blank or non-numeric cell inside a data block is written as 0 "
                "and counted per file (blank_cells_written_as_zero, "
                "non_numeric_cells_written_as_zero). The 2025 workbook has none: "
                "disclosure control is by rounding, not suppression markers."
            ),
        },
        "excluded": [
            "Table 8 aggregate geographies K02000001 (UK), K03000001 (GB), "
            "K04000001 (England and Wales), E92000001 (England)",
            "Table 15 aggregate geography rows (the same four codes)",
            "every Total column and the Total row of each table",
            "the Total legal status block of Tables 13 and 15",
        ],
        "region_codes": dict(REGIONS),
        "legal_status": {
            str(code): {
                "label": label,
                "source": "Tables 13 and 15 row 4, verbatim",
            }
            for code, label in LEGAL_STATUSES
        },
        "industry_groups": {
            group: {"label": label, "sic_divisions": group}
            for group, label in INDUSTRY_GROUPS
        },
        "turnover_bands": {
            "unit": "GBP thousands",
            "ons_region_turnover.csv": list(TABLE_8_BANDS),
            "ons_legal_status_turnover.csv": list(TABLE_13_BANDS),
            "ons_legal_status_region_turnover.csv": list(TABLE_13_BANDS),
            "table_8_labels_note": (
                "The same seven labels as the columns of "
                "tests/fixtures/uk/firm_2024_25/ons_firm_turnover.csv."
            ),
            "table_13_to_table_8": dict(TABLE_13_TO_TABLE_8_BAND),
        },
        "sic_division_key": (
            "Two-digit SIC 2007 division as text ('01'); read with dtype=str. "
            "All 88 divisions of Table 8, all-zero divisions included."
        ),
        "files": files,
    }


def _write(out: Path, workbook: Path, tables: dict[str, Extracted]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payloads = {name: table.to_csv_bytes() for name, table in tables.items()}
    for name, payload in payloads.items():
        (out / name).write_bytes(payload)
    record = provenance(workbook, tables, payloads)
    (out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    for name, table in tables.items():
        print(f"wrote {out / name} ({len(table.rows)} rows)")
    print(f"wrote {out / PROVENANCE_NAME}")


def _check(out: Path, tables: dict[str, Extracted]) -> int:
    stale = []
    for name, table in tables.items():
        path = out / name
        if not path.is_file() or path.read_bytes() != table.to_csv_bytes():
            stale.append(name)
    if stale:
        print(f"stale or missing: {', '.join(stale)}", file=sys.stderr)
        return 1
    print(f"{len(tables)} CSVs match the workbook")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-extract and compare bytes with the committed CSVs; write nothing",
    )
    args = parser.parse_args(argv)
    workbook = args.workbook.resolve()
    tables = extract(workbook)
    if args.check:
        return _check(args.out, tables)
    _write(args.out, workbook, tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
