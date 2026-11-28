#!/usr/bin/env python3
"""Extract HMRC R&D tax credit tables for the UK firm R&D assignment.

Reads the HMRC *Research and Development Tax Credits Statistics: September
2026* main tables (``rd_tax_credits_main_tables_2026.ods``) and writes three
CSVs with the published cells kept as published text (thousands separators,
HMRC's suppression marker, ``N/A`` and blanks included), so the fixture holds
raw published values and nothing else:

- ``hmrc_rd_rd1_claims.csv``: sheet ``RD1``, number of claims by scheme and
  financial year, 2016-17 to 2024-25;
- ``hmrc_rd_rd4_expenditure.csv``: sheet ``RD4``, R&D expenditure used to
  claim (£ million) by scheme and financial year;
- ``hmrc_rd_rd6_sector_2024_25.csv``: sheet ``RD6_2425``, number and cost of
  claims by SIC 2007 section and scheme, with total claims, total cost and
  total expenditure (£ million), 2024-25.

The ODS reader (``microcosm.build.uk_runtime.ods_tables``) is imported only
when a workbook is read, so engine-free CI, which reads the committed CSVs,
never opens one.

Usage::

    uv run --no-sync python tools/extract_uk_firm_rd_tax_credits_tables.py \\
        --ods <download>/rd_tax_credits_main_tables_2026.ods

    # Exit 1 if re-extraction would change a committed CSV's bytes.
    uv run --no-sync python tools/extract_uk_firm_rd_tax_credits_tables.py \\
        --ods <download>/rd_tax_credits_main_tables_2026.ods --check
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
    / "firm_rd_tax_credits_2024_25"
)
PROVENANCE_NAME = "provenance.json"

LANDING_PAGE = (
    "https://www.gov.uk/government/statistics/"
    "corporate-tax-research-and-development-tax-credit"
)
ODS_URL = (
    "https://assets.publishing.service.gov.uk/media/6ab52b118cb3c0580e501188/"
    "rd_tax_credits_main_tables_2026.ods"
)
#: The committed fixture's recorded access date: the default for
#: ``--access-date``, which the person who downloads the workbook sets.
ACCESS_DATE = "2026-11-23"
COVER_TITLE = (
    "HM Revenue and Customs: Research and Development Tax Credits Statistics 2026"
)

#: (sheet, CSV name, title prefix, header cells as published, CSV columns).
RD1 = (
    "RD1",
    "hmrc_rd_rd1_claims.csv",
    "RD1: Number of claims for the R&D tax credit for all schemes by financial year",
    (
        "Financial Year",
        "Status",
        "SME Deductions",
        "SME Payable Credits",
        "SME Combination",
        "SME Scheme Intensives & ERIS",
        "SME Total",
        "RDEC by LCs",
        "RDEC claims by SMEs",
        "Merged RDEC by LCs",
        "Merged RDEC by SMEs",
        "RDEC & Merged RDEC Total",
        "Total Claims",
        "Total Returns",
        "Total Companies",
    ),
    (
        "financial_year",
        "status",
        "sme_deductions",
        "sme_payable_credits",
        "sme_combination",
        "sme_intensive_eris",
        "sme_total",
        "rdec_large",
        "rdec_sme",
        "merged_rdec_large",
        "merged_rdec_sme",
        "rdec_merged_total",
        "total_claims",
        "total_returns",
        "total_companies",
    ),
)
RD4 = (
    "RD4",
    "hmrc_rd_rd4_expenditure.csv",
    "RD4: R&D expenditure used to claim R&D tax credits by scheme and financial year",
    (
        "Financial Year",
        "Status",
        "SME Scheme",
        "SME Scheme Intensives & ERIS",
        "RDEC by LCs",
        "RDEC claims by SMEs",
        "Merged RDEC by LCs",
        "Merged RDEC by SMEs",
        "All R&D schemes",
    ),
    (
        "financial_year",
        "status",
        "sme_scheme",
        "sme_intensive_eris",
        "rdec_large",
        "rdec_sme",
        "merged_rdec_large",
        "merged_rdec_sme",
        "all_schemes",
    ),
)
RD6 = (
    "RD6_2425",
    "hmrc_rd_rd6_sector_2024_25.csv",
    "RD6: Industry sector analysis of number and cost of R&D tax credit claims, "
    "2024-25",
    (
        "Sector",
        "Number of SME Scheme claims",
        "Cost of SME Scheme claims",
        "Number of SME Scheme Intensive & ERIS claims",
        "Cost of SME Scheme Intensive & ERIS claims",
        "Number of RDEC claims by LCs",
        "Cost of RDEC claims by LCs",
        "Number of RDEC claims by SMEs",
        "Cost of RDEC claims by SMEs",
        "Number of Merged RDEC claims by LCs",
        "Cost of Merged RDEC claims by LCs",
        "Number of Merged RDEC claims by SMEs",
        "Cost of Merged RDEC claims by SMEs",
        "Total Claims",
        "Total Cost",
        "Total Expenditure",
    ),
    (
        "sector",
        "sme_scheme_claims",
        "sme_scheme_cost_gbp_m",
        "sme_intensive_eris_claims",
        "sme_intensive_eris_cost_gbp_m",
        "rdec_large_claims",
        "rdec_large_cost_gbp_m",
        "rdec_sme_claims",
        "rdec_sme_cost_gbp_m",
        "merged_rdec_large_claims",
        "merged_rdec_large_cost_gbp_m",
        "merged_rdec_sme_claims",
        "merged_rdec_sme_cost_gbp_m",
        "total_claims",
        "total_cost_gbp_m",
        "total_expenditure_gbp_m",
    ),
)
TABLES = (RD1, RD4, RD6)
#: RD6 row labels start with the SIC 2007 section letter, then ". ".
RD6_SECTIONS = tuple("ABCDEFGHIJKLMNOPQRS")
RD6_UNKNOWN = "Unknown or other"
FIRST_YEAR, LAST_YEAR = "2016-17", "2024-25"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return " ".join(str(value).split())


def extract_table(rows: Sequence[Sequence[Any]], spec: tuple) -> list[list[str]]:
    """One sheet's data rows as published text, checked against ``spec``."""

    sheet, _, title, header, columns = spec
    if not rows or not _text(rows[0][0] if rows[0] else "").startswith(title):
        raise ValueError(f"{sheet}: title is not {title!r}")
    found = [
        index
        for index, row in enumerate(rows)
        if tuple(_text(cell) for cell in row[: len(header)]) == header
    ]
    if len(found) != 1:
        raise ValueError(f"{sheet}: header row found {len(found)}x")
    out: list[list[str]] = []
    for row in rows[found[0] + 1 :]:
        first = _text(row[0]) if row else ""
        if first == "End of worksheet":
            break
        if not first:
            raise ValueError(f"{sheet}: blank row label before 'End of worksheet'")
        cells = [_text(row[i]) if i < len(row) else "" for i in range(len(header))]
        out.append(cells)
    else:
        raise ValueError(f"{sheet}: no 'End of worksheet' row")
    if sheet == "RD6_2425":
        labels = [cells[0] for cells in out]
        expected = [label[0] for label in labels[:-1]]
        if expected != list(RD6_SECTIONS) or labels[-1] != RD6_UNKNOWN:
            raise ValueError(f"{sheet}: unexpected sector rows {labels}")
        if not all(label[1:3] == ". " for label in labels[:-1]):
            raise ValueError(f"{sheet}: sector labels are not 'X. name'")
    else:
        years = [cells[0] for cells in out]
        if years[0] != FIRST_YEAR or years[-1] != LAST_YEAR:
            raise ValueError(f"{sheet}: years run {years[0]} to {years[-1]}")
    del columns
    return out


def to_csv_bytes(columns: Sequence[str], rows: list[list[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def extract(ods: Path) -> dict[str, bytes]:
    """The three CSVs' bytes, keyed by file name."""

    from microcosm.build.uk_runtime.ods_tables import (  # noqa: PLC0415
        read_ods_tables,
    )

    tables = read_ods_tables(ods, label="HMRC R&D tax credit statistics tables")
    cover = _text(tables["Cover"].cell(0, 0))
    if cover != COVER_TITLE:
        raise ValueError(f"workbook is {cover!r}, not {COVER_TITLE!r}")
    payloads = {}
    for spec in TABLES:
        sheet, name, _, _, columns = spec
        rows = extract_table(tables[sheet].rows, spec)
        payloads[name] = to_csv_bytes(columns, rows)
    return payloads


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
    ods: Path, payloads: dict[str, bytes], access_date: str = ACCESS_DATE
) -> dict[str, Any]:
    workbook = ods.read_bytes()
    return {
        "kind": "uk_firm_rd_tax_credit_fixture",
        "schema_version": 1,
        "period": (
            "Financial year 2024-25 (accounting periods; provisional, uplifted "
            "by HMRC for claims not yet received)"
        ),
        "accessed": access_date,
        "method": (
            "Cells were read from the downloaded workbook on the access date "
            "by tools/extract_uk_firm_rd_tax_credits_tables.py and written as "
            "published text: thousands separators, HMRC's suppression marker "
            "(a '<' followed by the rounding floor), 'N/A' and blank cells "
            "are kept. None was filled from memory."
        ),
        "source": {
            "publisher": "HM Revenue and Customs",
            "dataset_title": "Corporate tax: Research and Development Tax Credits",
            "landing_page": LANDING_PAGE,
            "url": ODS_URL,
            "edition": (
                "Research and Development Tax Credits Statistics: September "
                "2026 (published 29 September 2026); Main tables 2026; claims "
                "in Company Tax returns received on or before 31 May 2026"
            ),
            "file": ods.name,
            "size_bytes": len(workbook),
            "sha256": _sha256(workbook),
            "sheets": {spec[1]: spec[0] for spec in TABLES},
            "rounding": (
                "All figures independently rounded to the nearest 5 or £5 "
                "million; components may not sum to totals (Cover sheet)."
            ),
            "licence": "Open Government Licence v3.0",
        },
        "edition_note": (
            "The September 2026 edition is the latest on the access date and "
            "replaces the September 2025 edition (2023-24 claims). Its 2024-25 "
            "year is the first with the merged RDEC scheme (accounting "
            "periods beginning on or after 1 April 2024): HMRC notes SME-level "
            "figures for 2024-25 are not comparable with earlier years."
        ),
        "coverage": (
            "RD6 sector is the primary SIC 2007 code of the registered "
            "company, which may not be the sector of the R&D. RD6 gives the "
            "number and cost of claims by scheme but qualifying expenditure "
            "only in total per sector; RD4 gives expenditure by scheme only "
            "nationally. RD1 counts claims (40,325 in 2024-25) and companies "
            "(37,545) separately."
        ),
        "extraction_date": dt.date.today().isoformat(),
        "tool": "tools/extract_uk_firm_rd_tax_credits_tables.py",
        "command": (
            "uv run --no-sync python tools/extract_uk_firm_rd_tax_credits_tables.py "
            "--ods <download>/rd_tax_credits_main_tables_2026.ods"
        ),
        "files": {
            name: {
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "rows": payload.count(b"\n") - 1,
            }
            for name, payload in payloads.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ods", type=Path, required=True)
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
    ods = args.ods.resolve()
    payloads = extract(ods)
    if args.check:
        stale = [
            name
            for name, payload in payloads.items()
            if not (args.out / name).is_file()
            or (args.out / name).read_bytes() != payload
        ]
        for name in stale:
            print(f"stale or missing: {name}", file=sys.stderr)
        if stale:
            return 1
        print("the R&D tax credit CSVs match the workbook")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (args.out / name).write_bytes(payload)
        print(f"wrote {args.out / name}")
    record = provenance(ods, payloads, args.access_date)
    (args.out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out / PROVENANCE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
