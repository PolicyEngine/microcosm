#!/usr/bin/env python3
"""Extract VOA rateable-value stock tables for the UK firm rateable value.

Reads the Valuation Office Agency's *Non-domestic rating: stock of
properties, March 2026* CSV bundle (``ndr_stock_of_properties_2026.zip``):
the stock of rateable properties (hereditaments) on the **2023 rating lists**
as at 31 March 2026, England and Wales. Writes three CSVs:

``voa_rv_band_sector.csv`` (Tables SOP2.1, SOP2.2, SOP2.3)
    ``geography,rv_band,band_lower_gbp,band_upper_gbp,sector,hereditaments,
    total_rv_gbp_thousand``: hereditaments and total rateable value by
    rateable-value band and RIO sector (``all``, ``retail``, ``offices``,
    ``industry``, ``other``) for ``england_and_wales``, ``england`` and
    ``wales``, with the tables' own ``Total`` rows (blank band bounds). The
    open top band has a blank upper bound.

``voa_property_type.csv`` (Table SOP1.2)
    ``geography,prop_type_code,parent_code,level,description,hereditaments,
    total_rv_gbp_thousand,mean_rv_gbp_thousand,median_rv_gbp_thousand,marker``:
    the same by property type, at levels ``all``, ``sector``, ``sub_sector``
    and ``shop_type``. A cell the VOA suppresses (``[c]``) keeps blank numbers
    and the marker; it is never written as 0.

``voa_stock_by_year.csv`` (Tables SOP3.1 and SOP3.2)
    ``geography,as_at,hereditaments,total_rv_gbp_thousand``: all-property
    stock at 31 March of each year the 2023 rating lists were in force (2024,
    2025, 2026), for the same three geographies.

Rateable values are in GBP thousand, as the bundle's variable-names metadata
says (``Total rateable value (GBP thousands)``); the tool copies the source
digits and does no arithmetic on them.

Usage::

    uv run --no-sync python tools/extract_uk_firm_rateable_value_tables.py \\
        --zip <download>/ndr_stock_of_properties_2026.zip

    # Exit 1 if re-extraction would change a committed CSV's bytes.
    uv run --no-sync python tools/extract_uk_firm_rateable_value_tables.py \\
        --zip <download>/ndr_stock_of_properties_2026.zip --check
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
    / "firm_rateable_value_2024"
)
PROVENANCE_NAME = ".provenance.json"

LANDING_PAGE = (
    "https://www.gov.uk/government/statistics/"
    "non-domestic-rating-stock-of-properties-march-2026"
)
COLLECTION_PAGE = (
    "https://www.gov.uk/government/collections/"
    "non-domestic-rating-stock-of-properties-collection"
)
ZIP_URL = (
    "https://assets.publishing.service.gov.uk/media/69f9bdf9a96f4d06cda76fbf/"
    "ndr_stock_of_properties_2026.zip"
)
EDITION = (
    "Non-domestic rating: stock of properties, March 2026 (published 14 May "
    "2026): stock on the 2023 central and local rating lists as at 31 March 2026"
)
RATING_LIST = "2023 rating list (England and Wales), as at 31 March 2026"
#: The committed fixture's recorded access date.
ACCESS_DATE = "2026-11-22"

BAND_FILE = "voa_rv_band_sector.csv"
PROPERTY_TYPE_FILE = "voa_property_type.csv"
YEAR_FILE = "voa_stock_by_year.csv"

BAND_HEADER = (
    "geography",
    "rv_band",
    "band_lower_gbp",
    "band_upper_gbp",
    "sector",
    "hereditaments",
    "total_rv_gbp_thousand",
)
PROPERTY_TYPE_HEADER = (
    "geography",
    "prop_type_code",
    "parent_code",
    "level",
    "description",
    "hereditaments",
    "total_rv_gbp_thousand",
    "mean_rv_gbp_thousand",
    "median_rv_gbp_thousand",
    "marker",
)
YEAR_HEADER = ("geography", "as_at", "hereditaments", "total_rv_gbp_thousand")

#: Band-table member per geography.
BAND_MEMBERS = {
    "england_and_wales": "table_SOP2_1.csv",
    "england": "table_SOP2_2.csv",
    "wales": "table_SOP2_3.csv",
}
SECTORS = ("all", "retail", "offices", "industry", "other")
TOTAL_BAND = "Total"
PROPERTY_TYPE_MEMBER = "table_SOP1_2.csv"
#: Geography suffix in Table SOP1.2's column names.
PROPERTY_TYPE_SUFFIXES = {
    "england_and_wales": "engwal",
    "england": "eng",
    "wales": "wal",
}
ALL_CODE_SOURCE = "[z]"
ALL_CODE = "ALL"
SUPPRESSED = "[c]"
COUNT_MEMBER = "table_SOP3_1.csv"
RV_MEMBER = "table_SOP3_2.csv"
#: Area code per geography in Tables SOP3.1 and SOP3.2.
AREA_CODES = {
    "england_and_wales": "K04000001",
    "england": "E92000001",
    "wales": "W92000004",
}
#: 31 March years with the 2023 rating lists in force (from 1 April 2023).
YEARS = ("2024", "2025", "2026")

_WHOLE = re.compile(r"^\d+$")
_DECIMAL = re.compile(r"^\d+(\.\d+)?$")
_BAND_RANGE = re.compile(r"^([\d,]+) - ([\d,]+)$")
_BAND_OPEN = re.compile(r"^([\d,]+) and over$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rows(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _whole(value: str, where: str) -> str:
    value = value.strip()
    if not _WHOLE.match(value):
        raise ValueError(f"{where}: {value!r} is not a whole number")
    return value


def _decimal(value: str, where: str) -> str:
    value = value.strip()
    if not _DECIMAL.match(value):
        raise ValueError(f"{where}: {value!r} is not a number")
    return value


def _band_bounds(band: str) -> tuple[str, str]:
    if band == TOTAL_BAND:
        return "", ""
    closed = _BAND_RANGE.match(band)
    if closed:
        return closed.group(1).replace(",", ""), closed.group(2).replace(",", "")
    opened = _BAND_OPEN.match(band)
    if opened:
        return opened.group(1).replace(",", ""), ""
    raise ValueError(f"unrecognised rateable-value band {band!r}")


def band_rows(members: dict[str, bytes]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for geography, member in BAND_MEMBERS.items():
        rows = _rows(members[member])
        if not rows or rows[-1]["rv_band"] != TOTAL_BAND:
            raise ValueError(f"{member}: last row is not {TOTAL_BAND!r}")
        for row in rows:
            band = row["rv_band"].strip()
            lower, upper = _band_bounds(band)
            for sector in SECTORS:
                where = f"{member} {band} {sector}"
                out.append(
                    (
                        geography,
                        band,
                        lower,
                        upper,
                        sector,
                        _whole(row[f"count_{sector}"], where),
                        _whole(row[f"total_rv_{sector}"], where),
                    )
                )
    return out


def property_type_rows(members: dict[str, bytes]) -> list[tuple[str, ...]]:
    rows = _rows(members[PROPERTY_TYPE_MEMBER])
    out: list[tuple[str, ...]] = []
    for geography, suffix in PROPERTY_TYPE_SUFFIXES.items():
        sector = ""
        for row in rows:
            code = row["prop_type_code"].strip()
            if code == ALL_CODE_SOURCE:
                code, parent, level = ALL_CODE, "", "all"
            elif len(code) == 1:
                sector, parent, level = code, ALL_CODE, "sector"
            elif len(code) == 2:
                parent, level = sector, "sub_sector"
            elif len(code) == 3:
                parent, level = code[:2], "shop_type"
            else:
                raise ValueError(f"unrecognised property type code {code!r}")
            where = f"{PROPERTY_TYPE_MEMBER} {code} {geography}"
            cells = [
                row[f"{name}_{suffix}"].strip()
                for name in ("count", "total_rv", "mean_rv", "median_rv")
            ]
            if SUPPRESSED in cells:
                if set(cells) != {SUPPRESSED}:
                    raise ValueError(f"{where}: partly suppressed row")
                numbers, marker = ["", "", "", ""], SUPPRESSED
            else:
                numbers = [
                    _whole(cells[0], where),
                    _whole(cells[1], where),
                    _decimal(cells[2], where),
                    _decimal(cells[3], where),
                ]
                marker = ""
            out.append(
                (
                    geography,
                    code,
                    parent,
                    level,
                    row["description"].strip(),
                    *numbers,
                    marker,
                )
            )
    return out


def year_rows(members: dict[str, bytes]) -> list[tuple[str, ...]]:
    counts = {row["area_code"]: row for row in _rows(members[COUNT_MEMBER])}
    values = {row["area_code"]: row for row in _rows(members[RV_MEMBER])}
    out: list[tuple[str, ...]] = []
    for geography, code in AREA_CODES.items():
        for year in YEARS:
            where = f"{geography} {year}"
            out.append(
                (
                    geography,
                    f"{year}-03-31",
                    _whole(counts[code][year], f"{COUNT_MEMBER} {where}"),
                    _whole(values[code][year], f"{RV_MEMBER} {where}"),
                )
            )
    return out


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _source_members() -> tuple[str, ...]:
    return (
        *BAND_MEMBERS.values(),
        PROPERTY_TYPE_MEMBER,
        COUNT_MEMBER,
        RV_MEMBER,
    )


def extract(zip_path: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """The fixture CSVs' bytes by file name, and the members read."""

    with zipfile.ZipFile(zip_path) as archive:
        members = {name: archive.read(name) for name in _source_members()}
    tables = {
        BAND_FILE: _csv_bytes(BAND_HEADER, band_rows(members)),
        PROPERTY_TYPE_FILE: _csv_bytes(
            PROPERTY_TYPE_HEADER, property_type_rows(members)
        ),
        YEAR_FILE: _csv_bytes(YEAR_HEADER, year_rows(members)),
    }
    return tables, members


def _iso_date(value: str) -> str:
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"access date {value!r} is not YYYY-MM-DD"
        ) from None


def provenance(
    zip_path: Path,
    tables: dict[str, bytes],
    members: dict[str, bytes],
    access_date: str = ACCESS_DATE,
) -> dict[str, Any]:
    archive = zip_path.read_bytes()
    headers = {
        BAND_FILE: BAND_HEADER,
        PROPERTY_TYPE_FILE: PROPERTY_TYPE_HEADER,
        YEAR_FILE: YEAR_HEADER,
    }
    member_tables = {
        BAND_FILE: list(BAND_MEMBERS.values()),
        PROPERTY_TYPE_FILE: [PROPERTY_TYPE_MEMBER],
        YEAR_FILE: [COUNT_MEMBER, RV_MEMBER],
    }
    return {
        "kind": "uk_firm_rateable_value_fixture",
        "schema_version": 1,
        "accessed": access_date,
        "rating_list": RATING_LIST,
        "method": (
            "Every figure was copied digit for digit from the downloaded VOA "
            "CSV bundle by the tool on the access date; none was filled from "
            "memory and none is computed. Rateable values stay in the "
            "source's unit, GBP thousand (the bundle's metadata file "
            "ndr_sop_variable_names.csv: 'Total rateable value (GBP "
            "thousands)'). Band bounds are parsed from the band labels."
        ),
        "source": {
            "publisher": "Valuation Office Agency",
            "title": "Non-domestic rating: stock of properties, March 2026",
            "landing_page": LANDING_PAGE,
            "collection_page": COLLECTION_PAGE,
            "url": ZIP_URL,
            "edition": EDITION,
            "edition_note": (
                "The collection page listed this as the latest edition on the "
                "access date (landing page public-updated-at "
                "2026-05-14T09:30:00+01:00)."
            ),
            "zip_file": zip_path.name,
            "sha256": _sha256(archive),
            "zip_bytes": len(archive),
            "members": {
                name: {"sha256": _sha256(data), "bytes": len(data)}
                for name, data in sorted(members.items())
            },
            "licence": "Open Government Licence v3.0",
        },
        "coverage": (
            "Hereditaments (rateable properties), not firms, in England and "
            "Wales. Scotland and Northern Ireland have their own valuation "
            "rolls and are not in the release. Table SOP1.1 of the same "
            "bundle lists central rating list rateable value separately; it "
            "is not extracted."
        ),
        "reference_period": (
            "Stock at 31 March 2026 on the 2023 rating lists (in force 1 "
            "April 2023 to 31 March 2026). The directory is named for the "
            "firm build's rule period, 2024 (tax year 2024-25), not for the "
            "VOA edition; voa_stock_by_year.csv carries the all-property "
            "stock at 31 March 2024, 2025 and 2026 from the same bundle."
        ),
        "suppression": (
            "A Table SOP1.2 row the VOA marks '[c]' keeps blank numbers and "
            "the marker; it is never written as 0."
        ),
        "extraction_date": dt.date.today().isoformat(),
        "tool": "tools/extract_uk_firm_rateable_value_tables.py",
        "command": (
            "uv run --no-sync python "
            "tools/extract_uk_firm_rateable_value_tables.py "
            "--zip <download>/ndr_stock_of_properties_2026.zip"
        ),
        "files": {
            name: {
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "rows": payload.count(b"\n") - 1,
                "columns": list(headers[name]),
                "source_members": member_tables[name],
            }
            for name, payload in tables.items()
        },
    }


def _write(
    out: Path,
    zip_path: Path,
    tables: dict[str, bytes],
    members: dict[str, bytes],
    access_date: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in tables.items():
        (out / name).write_bytes(payload)
        print(f"wrote {out / name}")
    record = provenance(zip_path, tables, members, access_date)
    (out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out / PROVENANCE_NAME}")


def _check(out: Path, tables: dict[str, bytes]) -> int:
    stale = [
        name
        for name, payload in tables.items()
        if not (out / name).is_file() or (out / name).read_bytes() != payload
    ]
    for name in stale:
        print(f"stale or missing: {name}", file=sys.stderr)
    if stale:
        return 1
    print(f"{', '.join(tables)} match the VOA bundle")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, required=True)
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
            "date (YYYY-MM-DD) the zip was downloaded, written to the "
            f"provenance (default {ACCESS_DATE}, the committed fixture's)"
        ),
    )
    args = parser.parse_args(argv)
    zip_path = args.zip.resolve()
    tables, members = extract(zip_path)
    if args.check:
        return _check(args.out, tables)
    _write(args.out, zip_path, tables, members, args.access_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
