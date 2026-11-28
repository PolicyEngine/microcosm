#!/usr/bin/env python3
"""Extract DBT Business Population Estimates 2025 size tables for UK firms.

Reads the Department for Business and Trade *Business population estimates
for the UK and regions 2025* detailed tables workbook
(``BPE_2025_detailed_tables.xlsx``, published 2 October 2025, start-2025
estimates) and writes two CSVs:

- ``bpe_legal_status_bands.csv`` from sheet ``Table 3`` (UK private sector by
  legal status and employee size band): the private sector, companies,
  ordinary partnerships and sole proprietorships sections, with business
  number and employment (thousands). A cell published as ``[c]`` (disclosive)
  is left empty and flagged ``[c]`` in the row's flag column;
- ``bpe_size_bands.csv`` from sheet ``Table 1`` (UK private sector: businesses,
  employment, employees, working proprietors, turnover) and sheet ``Table 2``
  (whole economy, private sector, central and local government, non-profit
  organisations: businesses, employment, employees, turnover) by employee
  size band. A column a table does not publish is left empty.

Values are the cells as published; the percentage columns are not kept.
Band labels lose their ``[Note n]`` markers and nothing else.

openpyxl is imported only when a workbook is read, so engine-free CI, which
reads the committed CSVs, never needs it.

Usage::

    uv run --no-sync python tools/extract_uk_firm_bpe_tables.py \\
        --xlsx <download>/BPE_2025_detailed_tables.xlsx

    # Exit 1 if re-extraction would change the committed CSVs' bytes.
    uv run --no-sync python tools/extract_uk_firm_bpe_tables.py \\
        --xlsx <download>/BPE_2025_detailed_tables.xlsx --check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
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
    / "firm_bpe_2025"
)
LEGAL_STATUS_CSV = "bpe_legal_status_bands.csv"
SIZE_BANDS_CSV = "bpe_size_bands.csv"
PROVENANCE_NAME = "firm_bpe_2025.provenance.json"

LANDING_PAGE = (
    "https://www.gov.uk/government/statistics/business-population-estimates-2025"
)
XLSX_URL = (
    "https://assets.publishing.service.gov.uk/media/68dbccc9c487360cc70c9f4e/"
    "BPE_2025_detailed_tables.xlsx"
)
#: The committed fixture's recorded access date: the default for
#: ``--access-date``, which the person who downloads the workbook sets.
ACCESS_DATE = "2026-11-22"

CONTENTS_SHEET = "Contents"
EDITION_TITLE = "BUSINESS POPULATION ESTIMATES FOR THE UK AND REGIONS 2025"
PUBLISHED = "Published 2nd October 2025"

#: Employee size bands every section lists, in published order.
SIZE_BANDS_TABLE_1 = (
    "All businesses",
    "All employers",
    "With no employees (unregistered)",
    "With no employees (registered)",
    "1",
    "2 to 4",
    "5 to 9",
    "10 to 19",
    "20 to 49",
    "50 to 99",
    "100 to 199",
    "200 to 249",
    "250 to 499",
    "500 to 999",
    "1000 or more",
)
SIZE_BANDS_TABLE_2 = (
    "All",
    "Zero or 1 employee",
    *SIZE_BANDS_TABLE_1[5:],
)

TABLE_1 = "Table 1"
TABLE_1_TITLE = "Table 1 (UK private sector)"
TABLE_2 = "Table 2"
TABLE_2_TITLE = "Table 2 (UK whole economy)"
TABLE_3 = "Table 3"
TABLE_3_TITLE = "Tables 3 (UK legal status)"

#: Table 2 section heading -> sector key.
TABLE_2_SECTIONS = (
    ("Whole economy", "whole_economy"),
    ("Private sector", "private_sector"),
    ("Central and Local Government", "central_and_local_government"),
    ("Non-profit Organisations", "non_profit"),
)
#: Table 3 section heading prefix -> legal status key.
TABLE_3_SECTIONS = (
    ("Private sector", "private_sector"),
    ("Companies", "companies"),
    ("Ordinary Partnerships", "ordinary_partnerships"),
    ("Sole proprietorships", "sole_proprietorships"),
)

SUPPRESSED = "[c]"

LEGAL_STATUS_HEADER = (
    "legal_status",
    "size_band",
    "business_number",
    "business_number_flag",
    "employment_thousand",
    "employment_flag",
)
SIZE_BANDS_HEADER = (
    "table",
    "sector",
    "size_band",
    "business_number",
    "employment_thousand",
    "employees_thousand",
    "working_proprietors_thousand",
    "turnover_million",
)


def _text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def _band(value: Any) -> str:
    """A size band label without ``[Note n]`` markers."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(int(value))
    return re.sub(r"\s*\[Note \d+\]", "", _text(value)).strip()


def _cell(value: Any, where: str) -> tuple[str, str]:
    """``(value, flag)``: a number as published, or empty and ``[c]``."""

    if _text(value) == SUPPRESSED:
        return "", SUPPRESSED
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number or [c], got {value!r}")
    if value < 0:
        raise ValueError(f"{where}: negative value {value!r}")
    return (str(int(value)) if value == int(value) else repr(float(value))), ""


def _number(value: Any, where: str) -> str:
    text, flag = _cell(value, where)
    if flag:
        raise ValueError(f"{where}: unexpected suppressed cell")
    return text


def _header_columns(row: tuple[Any, ...], wanted: dict[str, str]) -> dict[str, int]:
    """Column index of each wanted header (matched on its first words)."""

    labels = [_text(cell).lower() for cell in row]
    out = {}
    for key, prefix in wanted.items():
        hits = [i for i, label in enumerate(labels) if label.startswith(prefix)]
        if len(hits) != 1:
            raise ValueError(f"header {prefix!r} found {len(hits)}x in {labels}")
        out[key] = hits[0]
    return out


def _sections(
    rows: list[tuple[Any, ...]],
    start: int,
    headings: tuple[tuple[str, str], ...],
    bands: tuple[str, ...],
) -> dict[str, list[tuple[Any, ...]]]:
    """Each heading's band rows, checked against ``bands`` in order."""

    out: dict[str, list[tuple[Any, ...]]] = {}
    labels = [_text(row[0]) if row else "" for row in rows]
    for heading, key in headings:
        hits = [i for i in range(start, len(rows)) if labels[i].startswith(heading)]
        if len(hits) != 1:
            raise ValueError(f"section {heading!r} found {len(hits)}x")
        block = rows[hits[0] + 1 : hits[0] + 1 + len(bands)]
        got = tuple(_band(row[0]) for row in block)
        if got != bands:
            raise ValueError(f"section {heading!r} bands {got} are not {bands}")
        out[key] = block
    return out


def extract_legal_status(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    """``bpe_legal_status_bands.csv`` rows from sheet ``Table 3``'s cells."""

    if not _text(rows[0][0]).startswith(TABLE_3_TITLE):
        raise ValueError(f"unexpected Table 3 title {_text(rows[0][0])!r}")
    header = next(
        i
        for i, row in enumerate(rows)
        if _text(row[0]).startswith("Employee size band")
    )
    columns = _header_columns(
        rows[header],
        {"business": "business number", "employment": "employment number"},
    )
    out = []
    sections = _sections(rows, header + 1, TABLE_3_SECTIONS, SIZE_BANDS_TABLE_1)
    for _, key in TABLE_3_SECTIONS:
        for row in sections[key]:
            band = _band(row[0])
            where = f"Table 3 {key} {band}"
            business, business_flag = _cell(row[columns["business"]], where)
            employment, employment_flag = _cell(row[columns["employment"]], where)
            out.append(
                (key, band, business, business_flag, employment, employment_flag)
            )
    return out


def extract_size_bands(
    table_1: list[tuple[Any, ...]], table_2: list[tuple[Any, ...]]
) -> list[tuple[str, ...]]:
    """``bpe_size_bands.csv`` rows from sheets ``Table 1`` and ``Table 2``."""

    if not _text(table_1[0][0]).startswith(TABLE_1_TITLE):
        raise ValueError(f"unexpected Table 1 title {_text(table_1[0][0])!r}")
    if not _text(table_2[0][0]).startswith(TABLE_2_TITLE):
        raise ValueError(f"unexpected Table 2 title {_text(table_2[0][0])!r}")
    out = []
    header = next(
        i
        for i, row in enumerate(table_1)
        if _text(row[0]).startswith("Employee size band")
    )
    columns = _header_columns(
        table_1[header],
        {
            "business": "business number",
            "employment": "employment number",
            "employees": "employees number",
            "proprietors": "working proprietors number",
            "turnover": "turnover (£ millions)",
        },
    )
    block = table_1[header + 1 : header + 1 + len(SIZE_BANDS_TABLE_1)]
    if tuple(_band(row[0]) for row in block) != SIZE_BANDS_TABLE_1:
        raise ValueError("Table 1 bands are not the expected bands")
    for row in block:
        band = _band(row[0])
        where = f"Table 1 {band}"
        out.append(
            (
                TABLE_1,
                "private_sector",
                band,
                *(_number(row[columns[key]], where) for key in columns),
            )
        )
    header = next(
        i
        for i, row in enumerate(table_2)
        if _text(row[0]).startswith("Employee size band")
    )
    columns = _header_columns(
        table_2[header],
        {
            "business": "number of businesses",
            "employment": "employment number",
            "employees": "employees number",
            "turnover": "turnover (£ millions)",
        },
    )
    sections = _sections(table_2, header + 1, TABLE_2_SECTIONS, SIZE_BANDS_TABLE_2)
    for _, key in TABLE_2_SECTIONS:
        for row in sections[key]:
            band = _band(row[0])
            where = f"Table 2 {key} {band}"
            business, employment, employees, turnover = (
                _number(row[columns[name]], where)
                for name in ("business", "employment", "employees", "turnover")
            )
            out.append(
                (TABLE_2, key, band, business, employment, employees, "", turnover)
            )
    return out


def to_csv_bytes(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read(xlsx: Path) -> tuple[dict[str, list[tuple[Any, ...]]], list[str]]:
    import openpyxl  # noqa: PLC0415 - local-only dependency, kept out of CI imports

    book = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        sheets = {
            name: list(book[name].iter_rows(values_only=True))
            for name in (TABLE_1, TABLE_2, TABLE_3)
        }
        contents = [
            _text(row[0])
            for row in book[CONTENTS_SHEET].iter_rows(max_row=2, values_only=True)
        ]
    finally:
        book.close()
    return sheets, contents


def extract(
    xlsx: Path,
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], list[str]]:
    sheets, contents = _read(xlsx)
    if contents[:2] != [EDITION_TITLE, PUBLISHED]:
        raise ValueError(
            f"workbook is {contents[:2]!r}, not {[EDITION_TITLE, PUBLISHED]!r}"
        )
    legal = extract_legal_status(sheets[TABLE_3])
    size = extract_size_bands(sheets[TABLE_1], sheets[TABLE_2])
    return legal, size, contents


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iso_date(value: str) -> str:
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"access date {value!r} is not YYYY-MM-DD"
        ) from None


def provenance(
    xlsx: Path,
    payloads: dict[str, tuple[bytes, int, tuple[str, ...]]],
    contents: list[str],
    access_date: str = ACCESS_DATE,
) -> dict[str, Any]:
    workbook = xlsx.read_bytes()
    return {
        "kind": "uk_firm_bpe_size_band_fixture",
        "schema_version": 1,
        "period": "start of 2025 (BPE reference date), not a tax year",
        "accessed": access_date,
        "method": (
            "Figures were read from the downloaded workbook on the access date "
            "by tools/extract_uk_firm_bpe_tables.py; none was filled from "
            "memory. Numbers are the cells as published: business counts in "
            "businesses, employment, employees and working proprietors in "
            "thousands of people, turnover in millions of pounds. A cell "
            "published as [c] (disclosive) is left empty and flagged [c]. "
            "Band labels lose their [Note n] markers; percentage columns are "
            "not kept."
        ),
        "source": {
            "publisher": "Department for Business and Trade",
            "dataset_title": "Business population estimates 2025",
            "landing_page": LANDING_PAGE,
            "url": XLSX_URL,
            "edition": f"{contents[0]}; {contents[1]}",
            "file": xlsx.name,
            "size_bytes": len(workbook),
            "sha256": _sha256(workbook),
            "sheets": {
                LEGAL_STATUS_CSV: [TABLE_3],
                SIZE_BANDS_CSV: [TABLE_1, TABLE_2],
            },
            "tables": {
                TABLE_1: (
                    "Number and percentage of businesses in the private sector "
                    "and their associated employment and turnover, by number "
                    "of employees, UK, start 2025"
                ),
                TABLE_2: (
                    "Number and percentage of businesses and other "
                    "organisations in the whole economy, private sector, "
                    "central and local government, and non-profit "
                    "organisations with associated employment and turnover, "
                    "start 2025"
                ),
                TABLE_3: (
                    "Number and percentage of businesses in the private "
                    "sector and associated employment and turnover, by legal "
                    "status and number of employees, UK, start 2025"
                ),
            },
            "designation": "Official statistics in development",
            "licence": "Open Government Licence v3.0",
        },
        "notes": (
            "Employment is employees plus working proprietors (Table 1 note "
            "3). Companies with one PAYE employee on the IDBR are counted "
            "with no employees, the employee being treated as a working "
            "proprietor (Table 3 notes 3 and 6). Businesses with no employees "
            "are registered (VAT and/or PAYE) or unregistered (Table 3 note "
            "2). Companies include public corporations and nationalised "
            "bodies (Table 3 heading). Turnover excludes SIC 2007 Section K. "
            "Business counts are rounded, so totals may not equal the sum of "
            "their parts."
        ),
        "extraction_date": dt.date.today().isoformat(),
        "tool": "tools/extract_uk_firm_bpe_tables.py",
        "command": (
            "uv run --no-sync python tools/extract_uk_firm_bpe_tables.py "
            "--xlsx <download>/BPE_2025_detailed_tables.xlsx"
        ),
        "files": {
            name: {
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "rows": rows,
                "columns": list(header),
            }
            for name, (payload, rows, header) in payloads.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-extract and compare bytes with the committed CSVs; write nothing",
    )
    parser.add_argument(
        "--access-date",
        type=_iso_date,
        default=ACCESS_DATE,
        help=(
            "date (YYYY-MM-DD) the workbook was downloaded, written to the "
            f"provenance (default {ACCESS_DATE}, the committed fixture's)"
        ),
    )
    args = parser.parse_args(argv)
    xlsx = args.xlsx.resolve()
    legal, size, contents = extract(xlsx)
    payloads = {
        LEGAL_STATUS_CSV: (
            to_csv_bytes(LEGAL_STATUS_HEADER, legal),
            len(legal),
            LEGAL_STATUS_HEADER,
        ),
        SIZE_BANDS_CSV: (
            to_csv_bytes(SIZE_BANDS_HEADER, size),
            len(size),
            SIZE_BANDS_HEADER,
        ),
    }
    if args.check:
        stale = [
            name
            for name, (payload, _, _) in payloads.items()
            if not (args.out / name).is_file()
            or (args.out / name).read_bytes() != payload
        ]
        for name in stale:
            print(f"stale or missing: {name}", file=sys.stderr)
        if stale:
            return 1
        print("CSVs match the workbook")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (payload, rows, _) in payloads.items():
        (args.out / name).write_bytes(payload)
        print(f"wrote {args.out / name} ({rows} rows)")
    record = provenance(xlsx, payloads, contents, args.access_date)
    (args.out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out / PROVENANCE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
