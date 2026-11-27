#!/usr/bin/env python3
"""Extract ONS ASHE section mean annual gross pay for the UK firm paybill.

Reads the ONS *Earnings and hours worked, industry by two-digit SIC: ASHE
Table 4*, 2025 revised edition (``ashetable42025revised.zip``), member
``Table 4.7a Annual pay - Gross 2025``, sheet ``All`` (all employee jobs:
full-time and part-time, both sexes), and writes ``ashe_section_annual_pay.csv``
with one row per SIC 2007 section A-U plus one ``ALL`` row (the sheet's
``ALL EMPLOYEES`` row):

``sic_section,description,jobs_thousand,mean_annual_gross_pay_gbp,marker``

``mean_annual_gross_pay_gbp`` is the sheet's ``Mean`` column (pounds a year)
and ``jobs_thousand`` its indicative ``Number of jobs (thousand)``. A section
ONS does not publish (``x`` unreliable or disclosive, ``:`` not available)
keeps both numbers blank and the ONS marker in ``marker``; it is never
written as 0. The 2025 edition suppresses section U.

ASHE annual estimates cover the tax year that ended on 5 April of the
reference year, so the 2025 edition is tax year 2024-25 (the firm build's
2024-25 vintage); the 2024 edition is 2023-24.

openpyxl is imported only when a workbook is read, so engine-free CI, which
reads the committed CSV, never needs it.

Usage::

    uv run --no-sync python tools/extract_uk_firm_paybill_table.py \\
        --zip <download>/ashetable42025revised.zip

    # Exit 1 if re-extraction would change the committed CSV's bytes.
    uv run --no-sync python tools/extract_uk_firm_paybill_table.py \\
        --zip <download>/ashetable42025revised.zip --check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import zipfile
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
    / "firm_paybill_2024"
)
PROVENANCE_NAME = ".provenance.json"
CSV_NAME = "ashe_section_annual_pay.csv"

DATASET_URL = (
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/"
    "earningsandworkinghours/datasets/industry2digitsicashetable4"
)
ZIP_URL = (
    "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/"
    "earningsandworkinghours/datasets/industry2digitsicashetable4/2025revised/"
    "ashetable42025revised.zip"
)
DATASET_TITLE = "Earnings and hours worked, industry by two-digit SIC: ASHE Table 4"
EDITION = "2025 revised"
#: The committed fixture's recorded access date: the default for
#: ``--access-date``, which the person who downloads the zip sets.
ACCESS_DATE = "2026-11-22"

MEMBER_MARKER = "Table 4.7a"
MEMBER_SUFFIX = "Annual pay - Gross 2025.xlsx"
SHEET = "All"
TITLE_PREFIX = "SIC2007 Table 4"
TITLE_MARKERS = ("Annual pay - Gross", "For all employee jobs", "United Kingdom 2025")
HEADER_ROW = 5
ALL_LABEL = "ALL EMPLOYEES"
ALL_KEY = "ALL"
SECTIONS = tuple("ABCDEFGHIJKLMNOPQRSTU")
#: ONS markers for an unpublished cell.
UNPUBLISHED_MARKERS = ("x", ":")

HEADER = (
    "sic_section",
    "description",
    "jobs_thousand",
    "mean_annual_gross_pay_gbp",
    "marker",
)


@dataclass
class Extracted:
    title: str
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    markers: dict[str, str] = field(default_factory=dict)

    def to_csv_bytes(self) -> bytes:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(self.rows)
        return buffer.getvalue().encode("utf-8")


def _code(value: Any) -> str:
    """A ``Code`` cell without the leading apostrophe and spaces ONS leaves."""

    return "" if value is None else str(value).strip().lstrip("'").strip()


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    if value < 0:
        raise ValueError(f"expected a non-negative number, got {value!r}")
    return repr(float(value))


def _column(header: Sequence[Any], label: str) -> int:
    matches = [
        index
        for index, value in enumerate(header)
        if value is not None and str(value).strip() == label
    ]
    if len(matches) != 1:
        raise ValueError(f"header row {HEADER_ROW}: {label!r} found {len(matches)}x")
    return matches[0]


def _row(row: Sequence[Any], key: str, columns: dict[str, int]) -> tuple:
    description = " ".join(str(row[columns["Description"]]).split())
    jobs = row[columns["(thousand)"]]
    mean = row[columns["Mean"]]
    if isinstance(mean, str) and mean.strip() in UNPUBLISHED_MARKERS:
        return (key, description, "", "", mean.strip())
    if mean is None:
        raise ValueError(f"section {key}: blank mean")
    return (key, description, _number(jobs), _number(mean), "")


def extract_rows(rows: list[tuple[Any, ...]]) -> Extracted:
    title = " ".join(str(rows[0][0]).split())
    if not title.startswith(TITLE_PREFIX):
        raise ValueError(f"unexpected title {title!r}")
    for marker in TITLE_MARKERS:
        if marker not in title:
            raise ValueError(f"title {title!r} lacks {marker!r}")
    header = rows[HEADER_ROW - 1]
    columns = {
        label: _column(header, label) for label in ("Description", "Code", "Mean")
    }
    # The jobs label is split over rows 3-5; its third line sits in row 5.
    columns["(thousand)"] = _column(header, "(thousand)")
    table = Extracted(title=title)
    all_rows = [
        row
        for row in rows[HEADER_ROW:]
        if row[columns["Description"]] is not None
        and str(row[columns["Description"]]).strip() == ALL_LABEL
    ]
    if len(all_rows) != 1:
        raise ValueError(f"{ALL_LABEL!r} found {len(all_rows)}x")
    by_section: dict[str, tuple[Any, ...]] = {}
    order = []
    for row in rows[HEADER_ROW:]:
        code = _code(row[columns["Code"]])
        if len(code) == 1 and code.isalpha():
            if code in by_section:
                raise ValueError(f"section {code} appears twice")
            by_section[code] = row
            order.append(code)
    if tuple(order) != SECTIONS:
        raise ValueError(f"sections {order} != {list(SECTIONS)}")
    for section in SECTIONS:
        record = _row(by_section[section], section, columns)
        table.rows.append(record)
        if record[4]:
            table.markers[section] = record[4]
    record = _row(all_rows[0], ALL_KEY, columns)
    if record[4]:
        raise ValueError(f"{ALL_LABEL!r} mean is not published")
    table.rows.append(record)
    return table


def _member(archive: zipfile.ZipFile) -> str:
    names = [
        name
        for name in archive.namelist()
        if MEMBER_MARKER in name and name.endswith(MEMBER_SUFFIX)
    ]
    if len(names) != 1:
        raise ValueError(f"zip members matching {MEMBER_MARKER!r}: {names}")
    return names[0]


def extract(zip_path: Path) -> tuple[Extracted, str, bytes]:
    import openpyxl  # noqa: PLC0415 - local-only dependency, kept out of CI imports

    with zipfile.ZipFile(zip_path) as archive:
        member = _member(archive)
        data = archive.read(member)
    book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        rows = list(book[SHEET].iter_rows(values_only=True))
    finally:
        book.close()
    return extract_rows(rows), member, data


def _iso_date(value: str) -> str:
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"access date {value!r} is not YYYY-MM-DD"
        ) from None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def provenance(
    zip_path: Path,
    member: str,
    member_bytes: bytes,
    table: Extracted,
    payload: bytes,
    access_date: str = ACCESS_DATE,
) -> dict[str, Any]:
    archive = zip_path.read_bytes()
    return {
        "source": {
            "publisher": "Office for National Statistics",
            "dataset_title": DATASET_TITLE,
            "dataset_id": "ASHE: Table 4",
            "dataset_url": DATASET_URL,
            "edition": EDITION,
            "download_url": ZIP_URL,
            "zip_file": zip_path.name,
            "zip_sha256": _sha256(archive),
            "zip_bytes": len(archive),
            "member": member,
            "member_sha256": _sha256(member_bytes),
            "member_bytes": len(member_bytes),
            "sheet": SHEET,
            "sheet_title": table.title,
            "column": "Mean (Annual pay - Gross, GBP)",
            "jobs_column": "Number of jobs (thousand), indicative only",
            "release_date_note": (
                "The dataset landing page read 'Release date: 22 October 2026' "
                "on the access date; that is the page's latest release (the "
                "2026 provisional edition), not necessarily this edition's. "
                "The zip's members are dated 13 October 2026."
            ),
            "licence": "Open Government Licence v3.0",
            "access_date": access_date,
        },
        "reference_period": (
            "Workbook Notes: annual estimates are for the tax year that ended on "
            "5 April in the reference year, so the 2025 edition is tax year "
            "2024-25. The directory is named for the build's rule period, 2024 "
            "(tax year 2024-25), not for the ASHE edition."
        ),
        "coverage": (
            "Workbook footnote a: employees on adult rates who have been in the "
            "same job for more than a year. All employee jobs, full-time and "
            "part-time; a person with two jobs counts twice."
        ),
        "extraction_date": dt.date.today().isoformat(),
        "tool": "tools/extract_uk_firm_paybill_table.py",
        "command": (
            "uv run --no-sync python tools/extract_uk_firm_paybill_table.py "
            "--zip <download>/ashetable42025revised.zip"
        ),
        "unpublished": {
            "markers": dict(table.markers),
            "rule": (
                "A section whose mean is 'x' or ':' keeps blank numbers and the "
                "marker; it is never written as 0."
            ),
        },
        "all_row": (
            f"sic_section {ALL_KEY!r} is the sheet's {ALL_LABEL!r} row, the "
            "fallback for sections without a published mean."
        ),
        "files": {
            CSV_NAME: {
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "rows": len(table.rows),
                "columns": list(HEADER),
            }
        },
    }


def _write(
    out: Path,
    zip_path: Path,
    member: str,
    data: bytes,
    table: Extracted,
    access_date: str = ACCESS_DATE,
):
    out.mkdir(parents=True, exist_ok=True)
    payload = table.to_csv_bytes()
    (out / CSV_NAME).write_bytes(payload)
    record = provenance(zip_path, member, data, table, payload, access_date)
    (out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out / CSV_NAME} ({len(table.rows)} rows)")
    print(f"wrote {out / PROVENANCE_NAME}")


def _check(out: Path, table: Extracted) -> int:
    path = out / CSV_NAME
    if not path.is_file() or path.read_bytes() != table.to_csv_bytes():
        print(f"stale or missing: {CSV_NAME}", file=sys.stderr)
        return 1
    print(f"{CSV_NAME} matches the workbook")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, required=True)
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
            "date (YYYY-MM-DD) the zip was downloaded, written to the "
            f"provenance (default {ACCESS_DATE}, the committed fixture's)"
        ),
    )
    args = parser.parse_args(argv)
    zip_path = args.zip.resolve()
    table, member, data = extract(zip_path)
    if args.check:
        return _check(args.out, table)
    _write(args.out, zip_path, member, data, table, args.access_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
