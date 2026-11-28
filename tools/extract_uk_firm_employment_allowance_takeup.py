#!/usr/bin/env python3
"""Extract HMRC Employment Allowance take-up tables for the UK firm claim draw.

Reads HMRC *Employment Allowance take-up statistics: 2024 to 2025 tax year
estimates* (``Employment_Allowance_2024-2025.ods``, published 9:30am 25 June
2025), column *2024 to 2025 tax year: Employers benefitting from EA* (and its
*% of EA beneficiaries* neighbour), and writes three CSVs:

``hmrc_ea_takeup_by_size.csv`` from sheet ``5_Employer_size_breakdown``::

    business_size,employers_benefitting,share_percent

one row per published band (``1-9``, ``10-49``, ``50-249``, ``250+``) and a
``Total`` row. Size is the PAYE scheme's employee count at the end of the
tax year. A suppressed cell keeps HMRC's marker (``<500``, ``<1``) as text:
it is never turned into a number.

``hmrc_ea_takeup_by_sector.csv`` from sheet ``2_Sector_breakdown``::

    sic_section,sic_description,employers_benefitting,share_percent

one row per HMRC sector (single SIC 2007 section letters, and ``B,D,E,O,U``
for HMRC's pooled *Other*) and a ``Total`` row.

``hmrc_ea_takeup_by_region.csv`` from sheet ``3_Regional_breakdown``::

    region_code,region_description,employers_benefitting,share_percent

one row per region as published (GSS codes for the English regions, HMRC's
``W99999999``, ``S99999999`` and ``N99999999`` pseudo codes for the nations)
and a ``Total`` row.

Counts are as published (rounded by HMRC to the nearest 1,000, thousands
separators removed); shares are whole percent as published. Cases without
size, sector or region information were apportioned by HMRC over the known
distribution.

The workbook is read with ``uk_runtime/ods_tables.py`` (standard library
only), loaded from its file when a workbook is read, so engine-free CI, which
reads the committed CSVs, never runs this reader.

Usage::

    uv run --no-sync python tools/extract_uk_firm_employment_allowance_takeup.py \\
        --ods <download>/Employment_Allowance_2024-2025.ods

    # Exit 1 if re-extraction would change a committed CSV's bytes.
    uv run --no-sync python tools/extract_uk_firm_employment_allowance_takeup.py \\
        --ods <download>/Employment_Allowance_2024-2025.ods --check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
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
    / "firm_employment_allowance_takeup_2024_25"
)
ODS_TABLES = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "uk_runtime"
    / "ods_tables.py"
)

ODS_URL = (
    "https://assets.publishing.service.gov.uk/media/684bf34372e2ed81c079700c/"
    "Employment_Allowance_2024-2025.ods"
)
LANDING_PAGE = (
    "https://www.gov.uk/government/statistics/"
    "employment-allowance-take-up-2024-to-2025-tax-year-estimate"
)
EDITION = "Employment Allowance take-up statistics: 2024 to 2025 tax year estimates"
PUBLISHED = "2025-06-25"
#: The committed fixture's recorded access date: the default for
#: ``--access-date``, which the person who downloads the workbook sets.
ACCESS_DATE = "2026-11-22"

COVER_SHEET = "Cover_Sheet"
PUBLICATION_LINE = (
    "The data tables in this spreadsheet were originally published at 9:30am "
    "25 June 2025."
)
TAX_YEAR = "2024 to 2025"
COUNT_HEADER = f"{TAX_YEAR} tax year: Employers benefitting from EA"
SHARE_HEADER = f"{TAX_YEAR} tax year: % of EA beneficiaries"

SIZE_SHEET = "5_Employer_size_breakdown"
SECTOR_SHEET = "2_Sector_breakdown"
REGION_SHEET = "3_Regional_breakdown"

SIZE_CSV = "hmrc_ea_takeup_by_size.csv"
SECTOR_CSV = "hmrc_ea_takeup_by_sector.csv"
REGION_CSV = "hmrc_ea_takeup_by_region.csv"

SIZE_HEADER = ("business_size", "employers_benefitting", "share_percent")
SECTOR_HEADER = (
    "sic_section",
    "sic_description",
    "employers_benefitting",
    "share_percent",
)
REGION_HEADER = (
    "region_code",
    "region_description",
    "employers_benefitting",
    "share_percent",
)

#: The band labels sheet 5 must carry, in order.
SIZE_BANDS = ("1-9", "10-49", "50-249", "250+")
#: A whole count with optional thousands separators, or a suppressed marker.
_COUNT = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)$")
_SUPPRESSED = re.compile(r"^<\d+$")


def _text(value: Any) -> str:
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return "" if value is None else " ".join(str(value).split())


def _cell(value: Any, what: str) -> str:
    """A published count or share: digits without separators, or the marker."""

    text = _text(value)
    if _SUPPRESSED.match(text):
        return text
    if not _COUNT.match(text):
        raise ValueError(f"{what}: expected a count or a '<n' marker, got {text!r}")
    return text.replace(",", "")


def _columns(rows: Sequence[Sequence[Any]], first: str) -> tuple[int, int, int]:
    """The header row's index and its 2024-25 count and share columns."""

    headers = [i for i, row in enumerate(rows) if row and _text(row[0]) == first]
    if len(headers) != 1:
        raise ValueError(f"header row starting {first!r} found {len(headers)}x")
    header = [_text(cell) for cell in rows[headers[0]]]
    for label in (COUNT_HEADER, SHARE_HEADER):
        if header.count(label) != 1:
            raise ValueError(f"column {label!r} found {header.count(label)}x")
    return headers[0], header.index(COUNT_HEADER), header.index(SHARE_HEADER)


def _body(rows: Sequence[Sequence[Any]], start: int) -> list[Sequence[Any]]:
    """Rows after the header up to and including the ``Total`` row."""

    out = []
    for row in rows[start + 1 :]:
        if not row or not _text(row[0]):
            continue
        out.append(row)
        if _text(row[0]) == "Total":
            return out
    raise ValueError("table has no Total row")


def extract_size(rows: Sequence[Sequence[Any]]) -> list[tuple[str, ...]]:
    start, count, share = _columns(rows, "Business size")
    out = [
        (
            _text(row[0]),
            _cell(row[count], f"size {_text(row[0])}"),
            _cell(row[share], f"size {_text(row[0])} share"),
        )
        for row in _body(rows, start)
    ]
    labels = tuple(row[0] for row in out)
    if labels != (*SIZE_BANDS, "Total"):
        raise ValueError(f"size bands are {labels}, not {(*SIZE_BANDS, 'Total')}")
    return out


def _coded(
    rows: Sequence[Sequence[Any]], first: str, what: str
) -> list[tuple[str, ...]]:
    start, count, share = _columns(rows, first)
    out = []
    for row in _body(rows, start):
        code = _text(row[0])
        out.append(
            (
                code,
                _text(row[1]),
                _cell(row[count], f"{what} {code}"),
                _cell(row[share], f"{what} {code} share"),
            )
        )
    return out


def extract_sector(rows: Sequence[Sequence[Any]]) -> list[tuple[str, ...]]:
    out = _coded(rows, "ONS SIC Code", "sector")
    for code, *_ in out[:-1]:
        if not re.fullmatch(r"[A-U](?:,[A-U])*", code):
            raise ValueError(f"unexpected sector code {code!r}")
    return out


def extract_region(rows: Sequence[Sequence[Any]]) -> list[tuple[str, ...]]:
    out = _coded(rows, "ONS Region Code", "region")
    for code, *_ in out[:-1]:
        if not re.fullmatch(r"[ENSW]\d{8}", code):
            raise ValueError(f"unexpected region code {code!r}")
    return out


def to_csv_bytes(header: Sequence[str], rows: list[tuple[str, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read_ods_tables(ods: Path) -> dict[str, Any]:
    # Loaded from its file, not through the uk_runtime package, so reading a
    # workbook needs neither the package's other imports nor numpy.
    spec = importlib.util.spec_from_file_location("_ods_tables", ODS_TABLES)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ODS_TABLES}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.read_ods_tables(ods, label="HMRC Employment Allowance take-up")


def extract(ods: Path) -> dict[str, bytes]:
    """The three CSVs' bytes, keyed by file name."""

    tables = _read_ods_tables(ods)
    for sheet in (COVER_SHEET, SIZE_SHEET, SECTOR_SHEET, REGION_SHEET):
        if sheet not in tables:
            raise ValueError(f"the workbook has no sheet {sheet!r}")
    cover = [_text(row[0]) for row in tables[COVER_SHEET].rows if row]
    if not cover or cover[0] != EDITION:
        raise ValueError(f"workbook is {cover[:1]!r}, not {EDITION!r}")
    if PUBLICATION_LINE not in cover:
        raise ValueError(f"cover sheet lacks {PUBLICATION_LINE!r}")
    return {
        SIZE_CSV: to_csv_bytes(SIZE_HEADER, extract_size(tables[SIZE_SHEET].rows)),
        SECTOR_CSV: to_csv_bytes(
            SECTOR_HEADER, extract_sector(tables[SECTOR_SHEET].rows)
        ),
        REGION_CSV: to_csv_bytes(
            REGION_HEADER, extract_region(tables[REGION_SHEET].rows)
        ),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_SHEETS = {SIZE_CSV: SIZE_SHEET, SECTOR_CSV: SECTOR_SHEET, REGION_CSV: REGION_SHEET}
_HEADERS = {SIZE_CSV: SIZE_HEADER, SECTOR_CSV: SECTOR_HEADER, REGION_CSV: REGION_HEADER}
_NOTES = {
    SIZE_CSV: (
        "Size is the number of employees on the claiming PAYE scheme at the "
        "end of the tax year (sheet note); employers may run several PAYE "
        "schemes. The 250+ count and share are published suppressed ('<500', "
        "'<1') and are kept as those markers, never as numbers. The bands "
        "sum to 1,222,000 against a published total of 1,223,000 (HMRC: "
        "'Sum of employer sizes may not sum to the total due to rounding')."
    ),
    SECTOR_CSV: (
        "Sections are SIC 2007 letters; HMRC pools B, D, E, O and U as "
        "'Other', kept here under the code 'B,D,E,O,U' as published. T's "
        "description is HMRC's shortened 'Activities of households as "
        "employers...'. Sector came from the ONS Business Lookup Table with a "
        "94% match rate; unmatched cases were apportioned by HMRC."
    ),
    REGION_CSV: (
        "Codes as published: GSS E12 codes for the English regions and "
        "HMRC's W99999999, S99999999 and N99999999 for Wales, Scotland and "
        "Northern Ireland. Region is the claiming enterprise's postcode in "
        "the ONS Business Lookup Table (94% match rate); unmatched cases were "
        "apportioned by HMRC."
    ),
}


def provenance(name: str, payload: bytes, ods: Path, access_date: str) -> dict:
    source = ods.read_bytes()
    return {
        "kind": "uk_firm_employment_allowance_takeup_fixture",
        "schema_version": 1,
        "period": "tax year 2024-25",
        "accessed": access_date,
        "url": ODS_URL,
        "landing_page": LANDING_PAGE,
        "edition": EDITION,
        "published": PUBLISHED,
        "published_note": (
            f"Cover_Sheet: '{PUBLICATION_LINE}' The next publication was due June 2026."
        ),
        "publisher": "HM Revenue and Customs",
        "licence": "Open Government Licence v3.0",
        "sha256": _sha256(source),
        "size_bytes": len(source),
        "source_file": ods.name,
        "sheet": _SHEETS[name],
        "sheets_read": [COVER_SHEET, _SHEETS[name]],
        "columns": [COUNT_HEADER, SHARE_HEADER],
        "method": (
            "Every figure was read from the downloaded workbook on the access "
            "date by tools/extract_uk_firm_employment_allowance_takeup.py; "
            "none was filled from memory. Counts are as published, rounded by "
            "HMRC to the nearest 1,000, with thousands separators removed; "
            "shares are whole percent as published. Labels are whitespace-"
            "normalised (the region sheet's 'Total ' code loses its trailing "
            "space). An employer counts as benefitting when some allowance "
            "was offset against its employer NICs paid to HMRC (Cover_Sheet). "
            + _NOTES[name]
        ),
        "tool": "tools/extract_uk_firm_employment_allowance_takeup.py",
        "command": (
            "uv run --no-sync python "
            "tools/extract_uk_firm_employment_allowance_takeup.py "
            "--ods <download>/Employment_Allowance_2024-2025.ods"
        ),
        "extraction_date": dt.date.today().isoformat(),
        "file": {
            "name": name,
            "sha256": _sha256(payload),
            "bytes": len(payload),
            "rows": payload.count(b"\n") - 1,
            "columns": list(_HEADERS[name]),
        },
    }


def _iso_date(value: str) -> str:
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"access date {value!r} is not YYYY-MM-DD"
        ) from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ods", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--access-date", type=_iso_date, default=ACCESS_DATE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if a committed CSV differs from a fresh extraction",
    )
    args = parser.parse_args(argv)
    payloads = extract(args.ods)
    if args.check:
        stale = [
            name
            for name, payload in payloads.items()
            if not (args.out / name).is_file()
            or (args.out / name).read_bytes() != payload
        ]
        for name in stale:
            print(f"stale or missing: {name}", file=sys.stderr)
        if not stale:
            print(f"up to date: {', '.join(payloads)}")
        return 1 if stale else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (args.out / name).write_bytes(payload)
        record = provenance(name, payload, args.ods, args.access_date)
        (args.out / f"{name}.provenance.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
