#!/usr/bin/env python3
"""Extract ONS public sector headcounts for the UK firm public sector rescale.

Reads the ONS *Public sector employment* reference table workbook, March 2025
edition (``datasets8.xlsx``, "Public sector employment, UK: March 2025"),
sheet ``Table 1`` (*Public sector employment by sector classification;
Headcount*, United Kingdom, thousands, seasonally adjusted), row ``Mar 2025``,
and writes ``ons_public_sector_employment.csv`` with one row for central
government (series G6NQ) and one for local government (series G6NT):

``sector,ons_legal_status,series_id,period,headcount_thousand``

``ons_legal_status`` is the ONS business-count legal status the firm build
spreads each headcount over (5 central government, 6 local authority).

openpyxl is imported only when a workbook is read, so engine-free CI, which
reads the committed CSV, never needs it.

Usage::

    uv run --no-sync python tools/extract_uk_firm_public_sector_table.py \\
        --xlsx <download>/datasets8.xlsx

    # Also record what a later edition gives for the same quarter.
    uv run --no-sync python tools/extract_uk_firm_public_sector_table.py \\
        --xlsx <download>/datasets8.xlsx --later-xlsx <download>/datasets3.xlsx

    # Exit 1 if re-extraction would change the committed CSV's bytes.
    uv run --no-sync python tools/extract_uk_firm_public_sector_table.py \\
        --xlsx <download>/datasets8.xlsx --check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
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
    / "firm_public_sector_2025"
)
CSV_NAME = "ons_public_sector_employment.csv"
PROVENANCE_NAME = CSV_NAME + ".provenance.json"

LANDING_PAGE = (
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/"
    "publicsectorpersonnel/datasets/publicsectoremploymentreferencetable"
)
_FILE_BASE = (
    "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/"
    "publicsectorpersonnel/datasets/publicsectoremploymentreferencetable/"
)
XLSX_URL = _FILE_BASE + "march2025/datasets8.xlsx"
LATER_XLSX_URL = _FILE_BASE + "june2026/datasets3.xlsx"
#: The committed fixture's recorded access date: the default for
#: ``--access-date``, which the person who downloads the workbook sets.
ACCESS_DATE = "2026-11-22"

INDEX_SHEET = "Index"
EDITION_TITLE = "Public sector employment, UK: March 2025"
SHEET = "Table 1"
TITLE_PREFIX = "Table 1: Public sector employment by sector classification; Headcount"
UNITS = "United Kingdom (thousands), seasonally adjusted"
PERIOD = "Mar 2025"
#: sector, ONS legal status, series identifier, header words.
SECTORS = (
    ("central_government", 5, "G6NQ", ("Central", "government")),
    ("local_government", 6, "G6NT", ("Local", "government")),
)
HEADER = ("sector", "ons_legal_status", "series_id", "period", "headcount_thousand")


def _text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def _period(value: Any) -> str:
    """A period label without a revision marker such as ``[r]``."""

    return _text(value).split("[")[0].strip()


def extract_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """The CSV rows from sheet ``Table 1``'s cell rows."""

    if not _text(rows[0][0]).startswith(TITLE_PREFIX):
        raise ValueError(f"unexpected title {_text(rows[0][0])!r}")
    if not any(_text(cell) == UNITS for row in rows[:8] for cell in row):
        raise ValueError(f"sheet lacks the units line {UNITS!r}")
    series_rows = [
        row for row in rows[:12] if SECTORS[0][2] in [_text(cell) for cell in row]
    ]
    if len(series_rows) != 1:
        raise ValueError(f"series row with {SECTORS[0][2]} found {len(series_rows)}x")
    series = [_text(cell) for cell in series_rows[0]]
    labels = rows[rows.index(series_rows[0]) - 2]
    period_rows = [row for row in rows if _period(row[0]) == PERIOD]
    if len(period_rows) != 1:
        raise ValueError(f"period {PERIOD!r} found {len(period_rows)}x")
    out = []
    for sector, status, series_id, words in SECTORS:
        if series.count(series_id) != 1:
            raise ValueError(f"series {series_id} found {series.count(series_id)}x")
        column = series.index(series_id)
        label = _text(labels[column])
        if not all(word in label for word in words) or "Total" in label:
            raise ValueError(f"series {series_id} sits under {label!r}")
        value = period_rows[0][column]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{sector}: expected a number, got {value!r}")
        if value <= 0 or value != int(value):
            raise ValueError(f"{sector}: expected whole thousands, got {value!r}")
        out.append((sector, status, series_id, PERIOD, int(value)))
    return out


def to_csv_bytes(rows: list[tuple[Any, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read(xlsx: Path) -> tuple[list[tuple[Any, ...]], list[str]]:
    import openpyxl  # noqa: PLC0415 - local-only dependency, kept out of CI imports

    book = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        rows = list(book[SHEET].iter_rows(values_only=True))
        index = [
            _text(row[0])
            for row in book[INDEX_SHEET].iter_rows(max_row=2, values_only=True)
        ]
    finally:
        book.close()
    return rows, index


def extract(xlsx: Path) -> tuple[list[tuple[Any, ...]], list[str]]:
    rows, index = _read(xlsx)
    if index[0] != EDITION_TITLE:
        raise ValueError(f"workbook is {index[0]!r}, not {EDITION_TITLE!r}")
    return extract_rows(rows), index


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
    rows: list[tuple[Any, ...]],
    index: list[str],
    payload: bytes,
    access_date: str = ACCESS_DATE,
    later_xlsx: Path | None = None,
) -> dict[str, Any]:
    workbook = xlsx.read_bytes()
    record: dict[str, Any] = {
        "kind": "uk_firm_public_sector_employment_fixture",
        "schema_version": 1,
        "period": "March 2025 quarter (the last quarter of tax year 2024-25)",
        "accessed": access_date,
        "method": (
            "Figures were read from the downloaded workbook on the access date "
            "by tools/extract_uk_firm_public_sector_table.py; none was filled "
            "from memory. headcount_thousand is the Table 1 cell as published "
            "(thousands of people, seasonally adjusted). ons_legal_status is "
            "the firm build's mapping of each sector to an ONS business-count "
            "legal status (5 central government, 6 local authority), an "
            "assumption beyond the source."
        ),
        "source": {
            "publisher": "Office for National Statistics",
            "dataset_title": "Public sector employment",
            "landing_page": LANDING_PAGE,
            "url": XLSX_URL,
            "edition": f"March 2025 edition: {index[0]}; {index[1]}",
            "file": xlsx.name,
            "size_bytes": len(workbook),
            "sha256": _sha256(workbook),
            "sheet": SHEET,
            "table": (
                "Table 1: Public sector employment by sector classification; "
                "Headcount; United Kingdom (thousands), seasonally adjusted; "
                f"row '{PERIOD}'"
            ),
            "series": {sector: series_id for sector, _, series_id, _ in SECTORS},
            "survey": "Quarterly Public Sector Employment Survey",
            "licence": "Open Government Licence v3.0",
        },
        "edition_note": (
            "The March 2025 edition is not the dataset's latest: the landing "
            "page listed a June 2026 edition (release date 15 September 2026) "
            "on the access date. The March 2025 edition is used so the "
            "headcount is the first published estimate for the quarter that "
            "closes the 2024-25 frame; later editions revise it."
        ),
        "coverage": (
            "Headcount, full-time and part-time. Table 1 does not list the "
            "bodies in each sector; the workbook's Table 2 splits the public "
            "sector total by industry (HM Forces, police, public "
            "administration, education, National Health Service, other), and "
            "its Reclassification notes place English further education "
            "colleges and Network Rail in central government. Police "
            "(England and Wales) are based on projections (footnote 4). "
            "Public corporations (series G7AR) are not extracted."
        ),
        "extraction_date": dt.date.today().isoformat(),
        "tool": "tools/extract_uk_firm_public_sector_table.py",
        "command": (
            "uv run --no-sync python tools/extract_uk_firm_public_sector_table.py "
            "--xlsx <download>/datasets8.xlsx --later-xlsx <download>/datasets3.xlsx"
        ),
        "files": {
            CSV_NAME: {
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "rows": len(rows),
                "columns": list(HEADER),
            }
        },
    }
    if later_xlsx is not None:
        later_rows, later_index = _read(later_xlsx)
        later = later_xlsx.read_bytes()
        record["later_edition"] = {
            "note": (
                "The same row and series in a later edition, recorded for "
                "comparison only; the CSV does not use it."
            ),
            "url": LATER_XLSX_URL,
            "edition": f"{later_index[0]}; {later_index[1]}",
            "file": later_xlsx.name,
            "size_bytes": len(later),
            "sha256": _sha256(later),
            "headcount_thousand": {row[0]: row[4] for row in extract_rows(later_rows)},
        }
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--later-xlsx", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-extract and compare bytes with the committed CSV; write nothing",
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
    rows, index = extract(xlsx)
    payload = to_csv_bytes(rows)
    if args.check:
        path = args.out / CSV_NAME
        if not path.is_file() or path.read_bytes() != payload:
            print(f"stale or missing: {CSV_NAME}", file=sys.stderr)
            return 1
        print(f"{CSV_NAME} matches the workbook")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / CSV_NAME).write_bytes(payload)
    later = args.later_xlsx.resolve() if args.later_xlsx else None
    record = provenance(xlsx, rows, index, payload, args.access_date, later)
    (args.out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out / CSV_NAME} ({len(rows)} rows)")
    print(f"wrote {args.out / PROVENANCE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
