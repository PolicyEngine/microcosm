#!/usr/bin/env python3
"""Extract ONS ASHE section annual gross pay percentiles for UK firms.

Reads the ONS *Earnings and hours worked, industry by two-digit SIC: ASHE
Table 4*, 2025 revised edition (``ashetable42025revised.zip``), member
``Table 4.7a Annual pay - Gross 2025``, sheet ``All`` (all employee jobs), and
writes ``ashe_section_annual_pay_percentiles.csv`` with one row per SIC 2007
section A-U plus one ``ALL`` row (the sheet's ``ALL EMPLOYEES`` row):

``sic_section,description,jobs_thousand,mean,p10,p20,p25,p30,p40,p50,p60,p70,
p75,p80,p90``

``p50`` is the sheet's ``Median`` column; the others are its ``Percentiles``
columns and ``Mean``, pounds a year. A cell ONS does not publish keeps the ONS
marker (``x`` unreliable, ``:`` not available) as its value; it is never
written as 0 or blank. ASHE 2025 annual estimates are tax year 2024-25.

Usage::

    uv run --no-sync python tools/extract_uk_firm_pay_distribution_table.py \\
        --zip <download>/ashetable42025revised.zip [--check]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Sequence
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
    / "firm_pay_distribution_2024"
)
CSV_NAME = "ashe_section_annual_pay_percentiles.csv"
PROVENANCE_NAME = CSV_NAME + ".provenance.json"
LANDING_PAGE = (
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/"
    "earningsandworkinghours/datasets/industry2digitsicashetable4"
)
ZIP_URL = (
    "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/"
    "earningsandworkinghours/datasets/industry2digitsicashetable4/2025revised/"
    "ashetable42025revised.zip"
)
EDITION = "2025 revised"
ACCESS_DATE = "2026-11-22"
MEMBER_MARKER = "Table 4.7a"
MEMBER_SUFFIX = "Annual pay - Gross 2025.xlsx"
SHEET = "All"
TITLE_MARKERS = (
    "SIC2007 Table 4",
    "Annual pay - Gross",
    "For all employee jobs",
    "United Kingdom 2025",
)
HEADER_ROW = 5
ALL_LABEL = "ALL EMPLOYEES"
ALL_KEY = "ALL"
SECTIONS = tuple("ABCDEFGHIJKLMNOPQRSTU")
UNPUBLISHED_MARKERS = ("x", ":")
#: Percentile columns the sheet labels with a number, in sheet order.
PERCENTILES = (10, 20, 25, 30, 40, 60, 70, 75, 80, 90)
HEADER = (
    "sic_section",
    "description",
    "jobs_thousand",
    "mean",
    "p10",
    "p20",
    "p25",
    "p30",
    "p40",
    "p50",
    "p60",
    "p70",
    "p75",
    "p80",
    "p90",
)


def _cell(value: Any) -> str:
    """A number as ``repr(float)``, or the ONS marker the cell holds."""

    if isinstance(value, str) and value.strip() in UNPUBLISHED_MARKERS:
        return value.strip()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"expected a non-negative number or ONS marker: {value!r}")
    return repr(float(value))


def _column(header: Sequence[Any], label: Any) -> int:
    matches = [
        index
        for index, value in enumerate(header)
        if value is not None and (value == label or str(value).strip() == str(label))
    ]
    if len(matches) != 1:
        raise ValueError(f"header row {HEADER_ROW}: {label!r} found {len(matches)}x")
    return matches[0]


def extract_rows(rows: list[tuple[Any, ...]]) -> tuple[str, list[tuple[str, ...]]]:
    title = " ".join(str(rows[0][0]).split())
    for marker in TITLE_MARKERS:
        if marker not in title:
            raise ValueError(f"title {title!r} lacks {marker!r}")
    header = rows[HEADER_ROW - 1]
    source = {
        "description": _column(header, "Description"),
        "code": _column(header, "Code"),
        "jobs_thousand": _column(header, "(thousand)"),
        "mean": _column(header, "Mean"),
        "p50": _column(header, "Median"),
        **{f"p{p}": _column(header, p) for p in PERCENTILES},
    }
    picked: dict[str, tuple[Any, ...]] = {}
    for row in rows[HEADER_ROW:]:
        code = "" if row[source["code"]] is None else str(row[source["code"]])
        code = code.strip().lstrip("'").strip()
        label = row[source["description"]]
        if len(code) == 1 and code.isalpha():
            key = code
        elif label is not None and str(label).strip() == ALL_LABEL:
            key = ALL_KEY
        else:
            continue
        if key in picked:
            raise ValueError(f"row {key} appears twice")
        picked[key] = row
    if tuple(picked) != (ALL_KEY, *SECTIONS):
        raise ValueError(f"rows {list(picked)} are not ALL then sections A-U")
    out = []
    for key in (*SECTIONS, ALL_KEY):
        row = picked[key]
        description = " ".join(str(row[source["description"]]).split())
        out.append(
            (key, description, *(_cell(row[source[name]]) for name in HEADER[2:]))
        )
    return title, out


def to_csv_bytes(rows: list[tuple[str, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def extract(zip_path: Path) -> tuple[str, list[tuple[str, ...]], str, bytes]:
    import openpyxl  # noqa: PLC0415 - local-only dependency, kept out of CI imports

    with zipfile.ZipFile(zip_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if MEMBER_MARKER in name and name.endswith(MEMBER_SUFFIX)
        ]
        if len(names) != 1:
            raise ValueError(f"zip members matching {MEMBER_MARKER!r}: {names}")
        data = archive.read(names[0])
    book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        sheet_rows = list(book[SHEET].iter_rows(values_only=True))
    finally:
        book.close()
    title, rows = extract_rows(sheet_rows)
    return title, rows, names[0], data


def provenance(
    zip_path: Path, member: str, data: bytes, title: str, payload: bytes
) -> dict[str, Any]:
    archive = zip_path.read_bytes()
    return {
        "kind": "uk_firm_pay_distribution_fixture",
        "schema_version": 1,
        "period": "2024-25",
        "accessed": ACCESS_DATE,
        "method": (
            "Cells were copied from the downloaded workbook by "
            "tools/extract_uk_firm_pay_distribution_table.py; none was filled "
            "from memory. p50 is the sheet's Median column. A cell ONS does "
            "not publish keeps its marker ('x' unreliable, ':' not available)."
        ),
        "sources": {
            "ons_ashe_table_4": {
                "publisher": "Office for National Statistics",
                "landing_page": LANDING_PAGE,
                "url": ZIP_URL,
                "edition": EDITION,
                "sha256": hashlib.sha256(archive).hexdigest(),
                "size_bytes": len(archive),
                "member": member,
                "member_sha256": hashlib.sha256(data).hexdigest(),
                "sheet": SHEET,
                "sheet_title": title,
                "reference_period": (
                    "ASHE annual estimates are for the tax year that ended on "
                    "5 April of the reference year: the 2025 edition is "
                    "2024-25."
                ),
                "coverage": (
                    "Employees on adult rates in the same job for more than a "
                    "year; all employee jobs, full-time and part-time."
                ),
                "licence": "Open Government Licence v3.0",
            }
        },
        "files": {
            CSV_NAME: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "columns": list(HEADER),
            }
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-extract and compare bytes with the committed CSV; write nothing",
    )
    args = parser.parse_args(argv)
    zip_path = args.zip.resolve()
    title, rows, member, data = extract(zip_path)
    payload = to_csv_bytes(rows)
    target = args.out / CSV_NAME
    if args.check:
        if not target.is_file() or target.read_bytes() != payload:
            print(f"stale or missing: {CSV_NAME}", file=sys.stderr)
            return 1
        print(f"{CSV_NAME} matches the workbook")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    record = provenance(zip_path, member, data, title, payload)
    (args.out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {target} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
