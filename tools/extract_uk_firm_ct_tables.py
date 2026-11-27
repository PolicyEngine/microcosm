#!/usr/bin/env python3
"""Extract HMRC Corporation Tax liability tables for the UK firm profit draw.

Reads HMRC *Corporation Tax Statistics 2026* (``corporation_tax_data_tables_
2026.ods``, published 24 September 2026), financial year 2024-25 (1 April 2024
to 31 March 2025, the last column of each table), and writes two CSVs:

``ct_liability_bands.csv`` from sheet ``Tables_9A,B`` (*Tax liability by
band*; this was Table 10 in earlier editions, and ``Tables_10A,B`` is now
trading turnover by band)::

    band,lower_gbp,upper_gbp,companies,liability_gbp_million

one row per liability band as published (``0``, ``>0-100`` ... ``>50m``), then
``POSITIVE`` (Table 9B's *Companies with > £0 tax payable* and Table 9A's
*Total*) and ``ALL_RETURNS`` (*All companies who submitted a tax return*).
``lower_gbp`` is exclusive and ``upper_gbp`` inclusive; the open band has a
blank ``upper_gbp``. Table 9A has no row for the ``0`` band and no liability
for all returns, so those cells are blank, never 0.

``ct_sic_sections.csv`` from sheet ``Tables_6A,B`` (*Tax liability by SIC
section*)::

    hmrc_section,description,companies_with_liability,liability_gbp_million

one row per HMRC section A-U and one ``ALL`` row (the tables' *Total*). HMRC's
``U`` is *Overseas; Unclassified, Dormant and Non-trading*, not SIC section U
alone. This table is held out: the profit draw never reads it.

Counts are as published, rounded by HMRC to the nearest 5; amounts are GBP
million, rounded by HMRC to the nearest million. The "tax liability" of these
tables is *Corporation Tax payable* (its total equals that row of Table 5A).

Usage::

    uv run --no-sync python tools/extract_uk_firm_ct_tables.py \\
        --ods <download>/corporation_tax_data_tables_2026.ods

    # Exit 1 if re-extraction would change a committed CSV's bytes.
    uv run --no-sync python tools/extract_uk_firm_ct_tables.py \\
        --ods <download>/corporation_tax_data_tables_2026.ods --check
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
    / "firm_ct_2024"
)

LANDING_PAGE = (
    "https://www.gov.uk/government/statistics/corporation-tax-statistics-2026"
)
COLLECTION_PAGE = (
    "https://www.gov.uk/government/collections/"
    "analyses-of-corporation-tax-receipts-and-liabilities"
)
ODS_URL = (
    "https://assets.publishing.service.gov.uk/media/6aabc7c52f29b38884818639/"
    "corporation_tax_data_tables_2026.ods"
)
EDITION = (
    "Corporation Tax Statistics 2026, published 24 September 2026 (cover "
    "sheet: all tables updated in September 2026; next release autumn 2027)"
)
FINANCIAL_YEAR = "2024-25"
#: The committed fixtures' recorded access date: the default for
#: ``--access-date``, which the person who downloads the file sets.
ACCESS_DATE = "2026-11-22"

BANDS_SHEET = "Tables_9A,B"
BANDS_TITLE = "Tax liability by band, 2019-20 to 2024-25"
BANDS_AMOUNT_TABLE = "Table 9A: Total tax liability per band"
BANDS_COUNT_TABLE = "Table 9B: Number of companies per band"
BANDS_CSV = "ct_liability_bands.csv"
BANDS_HEADER = (
    "band",
    "lower_gbp",
    "upper_gbp",
    "companies",
    "liability_gbp_million",
)
ZERO_BAND = "0"
POSITIVE_LABEL = "Companies with > £0 tax payable"
ALL_RETURNS_LABEL = "All companies who submitted a tax return"
POSITIVE_KEY = "POSITIVE"
ALL_RETURNS_KEY = "ALL_RETURNS"
TOTAL_LABEL = "Total"

SECTIONS_SHEET = "Tables_6A,B"
SECTIONS_TITLE = (
    "Tax liability by Standard Industrial Classification (SIC) section "
    "2019-20 to 2024-25"
)
SECTIONS_AMOUNT_TABLE = "Table 6A: Tax liability by SIC section"
SECTIONS_COUNT_TABLE = (
    "Table 6B: Number of companies by SIC section with > £0 CT payable"
)
SECTIONS_CSV = "ct_sic_sections.csv"
SECTIONS_HEADER = (
    "hmrc_section",
    "description",
    "companies_with_liability",
    "liability_gbp_million",
)
SECTIONS = tuple("ABCDEFGHIJKLMNOPQRSTU")
ALL_KEY = "ALL"

END_OF_SHEET = "End of worksheet"
_MULTIPLIERS = {"m": 1_000_000, "b": 1_000_000_000}


def _text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def _label(value: Any) -> str:
    """A row label; the ``0`` band arrives as a number in Table 9B."""

    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return _text(value)


def _whole(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    if value != int(value) or value < 0:
        raise ValueError(f"{where}: expected a non-negative whole number, got {value}")
    return int(value)


def _edge(text: str) -> int:
    text = text.replace(",", "").strip()
    multiplier = _MULTIPLIERS.get(text[-1:], 1)
    digits = text[:-1] if multiplier != 1 else text
    if not digits.isdigit():
        raise ValueError(f"band edge {text!r} is not a number")
    return int(digits) * multiplier


def band_edges(label: str) -> tuple[int, int | None]:
    """``(lower, upper)`` pounds of a band label: lower exclusive, upper
    inclusive, ``None`` for the open band. ``0`` is ``(0, 0)``."""

    if label == ZERO_BAND:
        return 0, 0
    if not label.startswith(">"):
        raise ValueError(f"band label {label!r} does not start with '>'")
    lower, _, upper = label[1:].partition("-")
    return _edge(lower), (_edge(upper) if upper else None)


def _table(
    rows: Sequence[Sequence[Any]], heading: str, sheet: str
) -> list[tuple[str, int]]:
    """``(label, value)`` of the ``FINANCIAL_YEAR`` column of one table."""

    starts = [i for i, row in enumerate(rows) if row and _text(row[0]) == heading]
    if len(starts) != 1:
        raise ValueError(f"{sheet}: heading {heading!r} found {len(starts)}x")
    header = [_text(value) for value in rows[starts[0]]]
    if header[-1] != FINANCIAL_YEAR or header.count(FINANCIAL_YEAR) != 1:
        raise ValueError(
            f"{sheet}: {heading!r} does not end with column {FINANCIAL_YEAR}"
        )
    column = len(header) - 1
    out: list[tuple[str, int]] = []
    for row in rows[starts[0] + 1 :]:
        label = _label(row[0]) if row else ""
        if not label or label == END_OF_SHEET or label.startswith("Table "):
            break
        if len(row) <= column:
            raise ValueError(f"{sheet}: row {label!r} has no {FINANCIAL_YEAR} cell")
        out.append((label, _whole(row[column], f"{sheet} {label!r}")))
    if not out:
        raise ValueError(f"{sheet}: {heading!r} has no rows")
    return out


def _title(rows: Sequence[Sequence[Any]], title: str, sheet: str) -> None:
    found = _text(rows[0][0]) if rows and rows[0] else ""
    if found != title:
        raise ValueError(f"{sheet}: title {found!r} is not {title!r}")


def extract_bands(rows: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    _title(rows, BANDS_TITLE, BANDS_SHEET)
    amounts = dict(_table(rows, BANDS_AMOUNT_TABLE, BANDS_SHEET))
    counts = _table(rows, BANDS_COUNT_TABLE, BANDS_SHEET)
    count_by_label = dict(counts)
    for label in (ZERO_BAND, POSITIVE_LABEL, ALL_RETURNS_LABEL):
        if label not in count_by_label:
            raise ValueError(f"{BANDS_SHEET}: Table 9B has no {label!r} row")
    if TOTAL_LABEL not in amounts:
        raise ValueError(f"{BANDS_SHEET}: Table 9A has no {TOTAL_LABEL!r} row")
    bands = [label for label, _ in counts if label.startswith(">")]
    if bands != [label for label in amounts if label != TOTAL_LABEL]:
        raise ValueError(f"{BANDS_SHEET}: Tables 9A and 9B list different bands")
    out: list[tuple[Any, ...]] = [
        (ZERO_BAND, 0, 0, count_by_label[ZERO_BAND], ""),
    ]
    previous_upper: int | None = 0
    for label in bands:
        lower, upper = band_edges(label)
        if lower != previous_upper:
            raise ValueError(f"{BANDS_SHEET}: band {label!r} does not follow on")
        previous_upper = upper
        out.append(
            (
                label,
                lower,
                "" if upper is None else upper,
                count_by_label[label],
                amounts[label],
            )
        )
    if previous_upper is not None:
        raise ValueError(f"{BANDS_SHEET}: the last band is not open")
    out.append(
        (POSITIVE_KEY, "", "", count_by_label[POSITIVE_LABEL], amounts[TOTAL_LABEL])
    )
    out.append((ALL_RETURNS_KEY, "", "", count_by_label[ALL_RETURNS_LABEL], ""))
    return out


def extract_sections(rows: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    _title(rows, SECTIONS_TITLE, SECTIONS_SHEET)
    amounts = _table(rows, SECTIONS_AMOUNT_TABLE, SECTIONS_SHEET)
    counts = _table(rows, SECTIONS_COUNT_TABLE, SECTIONS_SHEET)
    if [label for label, _ in amounts] != [label for label, _ in counts]:
        raise ValueError(f"{SECTIONS_SHEET}: Tables 6A and 6B list different rows")
    labels = [label for label, _ in amounts]
    if labels[-1] != TOTAL_LABEL:
        raise ValueError(f"{SECTIONS_SHEET}: the last row is not {TOTAL_LABEL!r}")
    letters = tuple(label.partition(". ")[0] for label in labels[:-1])
    if letters != SECTIONS:
        raise ValueError(f"{SECTIONS_SHEET}: sections {letters} != {SECTIONS}")
    out: list[tuple[Any, ...]] = []
    for (label, amount), (_, count) in zip(amounts, counts, strict=True):
        if label == TOTAL_LABEL:
            out.append((ALL_KEY, TOTAL_LABEL, count, amount))
        else:
            letter, _, description = label.partition(". ")
            out.append((letter, description, count, amount))
    return out


def to_csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def extract(ods: Path) -> dict[str, bytes]:
    """The two CSVs' bytes, keyed by file name."""

    sys.path.insert(0, str(REPOSITORY_ROOT / "packages" / "microcosm-build" / "src"))
    from microcosm.build.uk_runtime.ods_tables import (  # noqa: PLC0415
        read_ods_tables,
    )

    tables = read_ods_tables(ods, label="HMRC Corporation Tax statistics tables")
    for sheet in (BANDS_SHEET, SECTIONS_SHEET):
        if sheet not in tables:
            raise ValueError(f"the workbook has no sheet {sheet!r}")
    return {
        BANDS_CSV: to_csv_bytes(BANDS_HEADER, extract_bands(tables[BANDS_SHEET].rows)),
        SECTIONS_CSV: to_csv_bytes(
            SECTIONS_HEADER, extract_sections(tables[SECTIONS_SHEET].rows)
        ),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def provenance(name: str, payload: bytes, ods: Path, access_date: str) -> dict:
    source = ods.read_bytes()
    bands = name == BANDS_CSV
    return {
        "kind": "uk_firm_ct_liability_fixture",
        "schema_version": 1,
        "period": f"financial year {FINANCIAL_YEAR} (1 April 2024 to 31 March 2025)",
        "accessed": access_date,
        "url": ODS_URL,
        "landing_page": LANDING_PAGE,
        "collection_page": COLLECTION_PAGE,
        "edition": EDITION,
        "publisher": "HM Revenue and Customs",
        "licence": "Open Government Licence v3.0",
        "sha256": _sha256(source),
        "size_bytes": len(source),
        "source_file": ods.name,
        "sheet": BANDS_SHEET if bands else SECTIONS_SHEET,
        "tables": (
            [BANDS_AMOUNT_TABLE, BANDS_COUNT_TABLE]
            if bands
            else [SECTIONS_AMOUNT_TABLE, SECTIONS_COUNT_TABLE]
        ),
        "column": FINANCIAL_YEAR,
        "method": (
            "Every figure was read from the downloaded workbook on the access "
            "date by tools/extract_uk_firm_ct_tables.py; none was filled from "
            "memory. Counts are as published (rounded by HMRC to the nearest "
            "5) and amounts are GBP million as published (rounded to the "
            "nearest million). "
            + (
                "Band edges are parsed from the band labels (lower exclusive, "
                "upper inclusive, 'm' = million). Table 9A has no row for the "
                "0 band and no liability for all returns, so those cells are "
                "blank. Earlier editions numbered this table 10; in the 2026 "
                "edition Tables 10A,B are trading turnover by band."
                if bands
                else "HMRC section U is 'Overseas; Unclassified, Dormant and "
                "Non-trading', not SIC section U alone. This table is held "
                "out of the profit draw and used only to check it."
            )
        ),
        "liability_concept": (
            "The tables' 'tax liability' total equals 'Corporation Tax "
            "payable' in Table 5A of the same workbook: after double taxation "
            "relief and other charges and reliefs, including ring fence "
            "Corporation Tax and the supplementary charge."
        ),
        "tool": "tools/extract_uk_firm_ct_tables.py",
        "command": (
            "uv run --no-sync python tools/extract_uk_firm_ct_tables.py "
            "--ods <download>/corporation_tax_data_tables_2026.ods"
        ),
        "extraction_date": dt.date.today().isoformat(),
        "file": {
            "name": name,
            "sha256": _sha256(payload),
            "bytes": len(payload),
            "rows": payload.count(b"\n") - 1,
            "columns": list(BANDS_HEADER if bands else SECTIONS_HEADER),
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
