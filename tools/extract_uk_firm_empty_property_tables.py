#!/usr/bin/env python3
"""Extract MHCLG empty property relief tables for the UK firm empty property draw.

Reads three MHCLG *National non-domestic rates collected by councils in
England* files and writes three CSVs with published values only:

- ``mhclg_nndr1_empty_hereditaments.csv``: England hereditaments in receipt of
  mandatory empty premises relief, total and by the six sub-categories, as at
  31 December 2023 (NNDR1 2024-25 supplementary table) and 31 December 2024
  (NNDR1 2025-26 supplementary table), sheet ``NNDR1S Hereditaments Data``;
- ``mhclg_nndr1_empty_relief_value.csv``: England estimated value (£) of
  empty property relief, total and by sub-category, for 2024-25 and 2025-26
  (sheet ``NNDR1S Relief data``; the 2025-26 edition prints the values with a
  minus sign, kept as published);
- ``mhclg_nndr3_2024_25.csv``: 2024-25 outturn items from the NNDR3 Tables
  1-3 workbook: gross rates payable and empty premises relief (£ million),
  hereditaments on the list at 30 September (thousands), aggregate rateable
  value (£ million) and the two multipliers (pence).

Columns are found by MHCLG's variable names (``hdit_mr_empty*``,
``value_empty*``) and row labels, not by position. openpyxl and the ODS reader
are imported only when a file is read, so engine-free CI never needs them.

Usage::

    uv run --no-sync python tools/extract_uk_firm_empty_property_tables.py \\
        --nndr1-2024-25 <download>/NNDR1_2024-25_Supplementary_table.xlsx \\
        --nndr1-2025-26 <download>/NNDR1_2025-26_Supplementary_table__ecomms_.xlsx \\
        --nndr3-ods <download>/NNDR3_2024-25_Tables_1-3_accessible_gov.uk__Values_only__-_3rd_Release.ods

    # Add --check to exit 1 if re-extraction would change a committed CSV.
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
    / "firm_empty_property_2024_25"
)
PROVENANCE_NAME = "provenance.json"
HEREDITAMENTS_CSV = "mhclg_nndr1_empty_hereditaments.csv"
RELIEF_CSV = "mhclg_nndr1_empty_relief_value.csv"
NNDR3_CSV = "mhclg_nndr3_2024_25.csv"
ACCESS_DATE = "2026-11-23"

_GOV = "https://www.gov.uk/government/statistics/"
_ASSETS = "https://assets.publishing.service.gov.uk/media/"
SOURCES = {
    "nndr1_2024_25": {
        "landing_page": _GOV
        + "national-non-domestic-rates-collected-by-councils-in-england-"
        "forecast-2024-to-2025",
        "url": _ASSETS
        + "663b98e474933dccbbb6c431/NNDR1_2024-25_Supplementary_table.xlsx",
        "edition": (
            "National non-domestic rates collected by councils in England: "
            "forecast 2024 to 2025; supplementary data tables (added 9 May 2024)"
        ),
    },
    "nndr1_2025_26": {
        "landing_page": _GOV
        + "national-non-domestic-rates-collected-by-councils-in-england-"
        "forecast-2025-to-2026",
        "url": _ASSETS
        + "67d2c9c4814acba3755e97c1/NNDR1_2025-26_Supplementary_table__ecomms_.xlsx",
        "edition": (
            "National non-domestic rates collected by councils in England: "
            "forecast 2025 to 2026; supplementary data tables (added 13 March 2025)"
        ),
    },
    "nndr3_2024_25": {
        "landing_page": _GOV
        + "national-non-domestic-rates-collected-by-councils-in-england-2024-to-2025",
        "url": _ASSETS
        + "6977f8533fd50ac304b79790/NNDR3_2024-25_Tables_1-3_accessible_gov.uk__"
        "Values_only__-_3rd_Release.ods",
        "edition": (
            "National non-domestic rates collected by councils in England: 2024 "
            "to 2025; Tables 1-3: England, 3rd release (updated 27 January 2026)"
        ),
    },
}

HEREDITAMENT_SHEET = "NNDR1S Hereditaments Data"
RELIEF_SHEET = "NNDR1S Relief data"
#: The six empty sub-categories partition the total (checked on extraction).
HEREDITAMENT_TOTAL = "hdit_mr_empty"
HEREDITAMENT_PARTS = (
    "hdit_mr_emptyind",
    "hdit_mr_emptylisted",
    "hdit_mr_emptycasc",
    "hdit_mr_emptycharity",
    "hdit_mr_emptyoth",
    "hdit_mr_emptyabove",
)
RELIEF_TOTAL = "value_empty"
RELIEF_PARTS = (
    "value_emptyind",
    "value_emptylisted",
    "value_emptycasc",
    "value_emptycharity",
    "value_emptyoth",
    "value_emptyabove",
)
HEREDITAMENT_HEADER = ("snapshot_date", "variable", "label", "england_hereditaments")
RELIEF_HEADER = ("forecast_year", "variable", "label", "england_value_gbp")
NNDR3_HEADER = ("item", "table", "row_label", "year", "value", "unit")
NNDR3_YEAR = "2024 to 2025"
#: item, sheet, row label prefix, unit.
NNDR3_ITEMS = (
    ("gross_rates_payable", "Table_1", "Gross rates payable in year", "GBP million"),
    ("empty_premises_relief", "Table_2", "Of which empty premises", "GBP million"),
    (
        "hereditaments_30_sept",
        "Table_1a",
        "Number of hereditaments on rating list as at 30 Sept",
        "thousand",
    ),
    (
        "aggregate_rateable_value_30_sept",
        "Table_1a",
        "Total aggregate rateable value of all hereditaments",
        "GBP million",
    ),
    (
        "small_business_multiplier",
        "Table_1a",
        "Small business rate multiplier",
        "pence",
    ),
    (
        "standard_multiplier",
        "Table_1a",
        "National non-domestic rates multiplier",
        "pence",
    ),
)
_MONTHS = {"December": 12}


def _text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number, got {value!r}")
    return float(value)


def _format(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(value)


def _variable_row(rows: list[tuple[Any, ...]], marker: str) -> int:
    found = [i for i, row in enumerate(rows[:12]) if marker in map(_text, row)]
    if len(found) != 1:
        raise ValueError(f"variable-name row with {marker!r} found {len(found)}x")
    return found[0]


def _label_row(rows: list[tuple[Any, ...]], at: int, column: int, prefix: str) -> int:
    for candidate in (at - 1, at + 1):
        row = rows[candidate] if 0 <= candidate < len(rows) else ()
        if column < len(row) and _text(row[column]).startswith(prefix):
            return candidate
    raise ValueError(f"no label row starting {prefix!r} next to row {at}")


def _england_row(rows: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    found = [row for row in rows if len(row) > 2 and _text(row[2]) == "England"]
    if len(found) != 1:
        raise ValueError(f"England row found {len(found)}x")
    return found[0]


def _snapshot(title: str) -> str:
    match = re.search(r"as at 31 (December) (\d{4})", title)
    if not match:
        raise ValueError(f"no 'as at 31 December YYYY' in {title!r}")
    return dt.date(int(match.group(2)), _MONTHS[match.group(1)], 31).isoformat()


def _forecast_year(title: str) -> str:
    match = re.search(r"in (\d{4}-\d{2})\b", title)
    if not match:
        raise ValueError(f"no forecast year in {title!r}")
    return match.group(1)


def extract_supplementary(
    hereditaments: list[tuple[Any, ...]], relief: list[tuple[Any, ...]]
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """Hereditament and relief rows from one supplementary workbook's sheets."""

    out_h, out_r = [], []
    for rows, total, parts, prefix, out, kind in (
        (
            hereditaments,
            HEREDITAMENT_TOTAL,
            HEREDITAMENT_PARTS,
            "Mandatory Relief for empty",
            out_h,
            "hereditaments",
        ),
        (
            relief,
            RELIEF_TOTAL,
            RELIEF_PARTS,
            "Estimated value of empty property relief",
            out_r,
            "relief",
        ),
    ):
        at = _variable_row(rows, total)
        names = [_text(cell) for cell in rows[at]]
        labels = rows[_label_row(rows, at, names.index(total), prefix)]
        england = _england_row(rows)
        key = (
            _snapshot(_text(rows[0][0]))
            if kind == "hereditaments"
            else _forecast_year(_text(rows[0][0]))
        )
        values = {}
        for name in (total, *parts):
            if names.count(name) != 1:
                raise ValueError(f"{kind}: variable {name} found {names.count(name)}x")
            column = names.index(name)
            values[name] = _number(england[column], f"{kind} {name}")
            out.append((key, name, _text(labels[column]), _format(values[name])))
        # The hereditament sub-categories partition the total; the relief
        # values do not quite (2024-25: £1,023.9m of £1,040.9m), as published.
        if kind == "hereditaments" and sum(values[p] for p in parts) != values[total]:
            raise ValueError(f"{kind}: sub-categories do not sum to {total}")
    return out_h, out_r


def extract_nndr3(tables: dict[str, Any]) -> list[tuple[Any, ...]]:
    out = []
    for item, sheet, prefix, unit in NNDR3_ITEMS:
        rows = tables[sheet].rows
        headers = [i for i, row in enumerate(rows) if row and _text(row[0]) == "Year"]
        if len(headers) != 1:
            raise ValueError(f"{sheet}: 'Year' header row found {len(headers)}x")
        header = [_text(c) for c in rows[headers[0]]]
        columns = [i for i, h in enumerate(header) if h.startswith(NNDR3_YEAR)]
        if len(columns) != 1:
            raise ValueError(f"{sheet}: column {NNDR3_YEAR!r} found {len(columns)}x")
        found = [row for row in rows if row and _text(row[0]).startswith(prefix)]
        if len(found) != 1:
            raise ValueError(f"{sheet}: row {prefix!r} found {len(found)}x")
        value = _number(found[0][columns[0]], f"{sheet} {prefix}")
        out.append(
            (item, sheet, _text(found[0][0]), header[columns[0]], _format(value), unit)
        )
    return out


def to_csv_bytes(header: Sequence[str], rows: list[tuple[Any, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read_xlsx(path: Path) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    import openpyxl  # noqa: PLC0415 - local-only dependency, kept out of CI imports

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return (
            list(book[HEREDITAMENT_SHEET].iter_rows(values_only=True)),
            list(book[RELIEF_SHEET].iter_rows(values_only=True)),
        )
    finally:
        book.close()


def extract(nndr1_2425: Path, nndr1_2526: Path, nndr3: Path) -> dict[str, bytes]:
    from microcosm.build.uk_runtime.ods_tables import (  # noqa: PLC0415
        read_ods_tables,
    )

    hereditaments: list[tuple[Any, ...]] = []
    relief: list[tuple[Any, ...]] = []
    for path in (nndr1_2425, nndr1_2526):
        h, r = extract_supplementary(*_read_xlsx(path))
        hereditaments += h
        relief += r
    nndr3_rows = extract_nndr3(read_ods_tables(nndr3, label="MHCLG NNDR3 2024-25"))
    return {
        HEREDITAMENTS_CSV: to_csv_bytes(HEREDITAMENT_HEADER, hereditaments),
        RELIEF_CSV: to_csv_bytes(RELIEF_HEADER, relief),
        NNDR3_CSV: to_csv_bytes(NNDR3_HEADER, nndr3_rows),
    }


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
    inputs: dict[str, Path], payloads: dict[str, bytes], access_date: str
) -> dict[str, Any]:
    sources = {}
    for key, path in inputs.items():
        data = path.read_bytes()
        sources[key] = {
            **SOURCES[key],
            "publisher": "Ministry of Housing, Communities and Local Government",
            "file": path.name,
            "size_bytes": len(data),
            "sha256": _sha256(data),
            "licence": "Open Government Licence v3.0",
        }
    sources["nndr1_2024_25"]["sheets"] = [HEREDITAMENT_SHEET, RELIEF_SHEET]
    sources["nndr1_2025_26"]["sheets"] = [HEREDITAMENT_SHEET, RELIEF_SHEET]
    sources["nndr3_2024_25"]["sheets"] = sorted({item[1] for item in NNDR3_ITEMS})
    return {
        "kind": "uk_firm_empty_property_fixture",
        "schema_version": 1,
        "period": (
            "2024-25: relief counts as at 31 December 2023 and 2024, relief "
            "value forecasts for 2024-25 and 2025-26, and the 2024-25 NNDR3 "
            "outturn"
        ),
        "accessed": access_date,
        "method": (
            "Values were read from the downloaded files on the access date by "
            "tools/extract_uk_firm_empty_property_tables.py, from the England "
            "row of each supplementary sheet (columns found by MHCLG variable "
            "name) and the '2024 to 2025' column of NNDR3 Tables 1, 1a and 2; "
            "none was filled from memory."
        ),
        "sources": sources,
        "coverage": (
            "Counts are hereditaments in receipt of mandatory empty premises "
            "relief at the snapshot date, not all empty hereditaments: an "
            "empty property past its 3-month (6 for industrial) exemption "
            "pays full rates and is not counted. The six sub-categories "
            "(industrial above the exemption threshold, listed building, "
            "community amateur sports club, charity, not in the previous "
            "four, non-industrial above the exemption threshold) partition "
            "the total; the relief value sub-categories do not quite sum to "
            "the relief total as published. No source gives the rateable value of empty "
            "property, empty shares by property type, or empty durations."
        ),
        "caveats": [
            "NNDR1 2024-25 supplementary Note 7: the split of empty property "
            "relief by category was first published in 2024-25 and was "
            "validated against earlier unvalidated data; use with caution.",
            "NNDR1 2025-26 Table 4 Note 25: for 2024 the empty premises "
            "hereditament count is under investigation for 2 local "
            "authorities; data are as provided by the authorities.",
            "The NNDR1 2025-26 relief labels read 'to be granted in 2024-25' "
            "as published; the forecast year is taken from the sheet title, "
            "and the total equals Table 2's 2025-26 empty premises forecast "
            "(£1,084.979229 million).",
            "NNDR1 2024-25 supplementary Note 3: Three Rivers and Watford "
            "supplied no supplementary figures for 31 December 2023.",
        ],
        "extraction_date": dt.date.today().isoformat(),
        "tool": "tools/extract_uk_firm_empty_property_tables.py",
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
    parser.add_argument("--nndr1-2024-25", type=Path, required=True)
    parser.add_argument("--nndr1-2025-26", type=Path, required=True)
    parser.add_argument("--nndr3-ods", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--access-date", type=_iso_date, default=ACCESS_DATE)
    args = parser.parse_args(argv)
    inputs = {
        "nndr1_2024_25": args.nndr1_2024_25.resolve(),
        "nndr1_2025_26": args.nndr1_2025_26.resolve(),
        "nndr3_2024_25": args.nndr3_ods.resolve(),
    }
    payloads = extract(*inputs.values())
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
        print("the empty property CSVs match the downloads")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (args.out / name).write_bytes(payload)
        print(f"wrote {args.out / name}")
    record = provenance(inputs, payloads, args.access_date)
    (args.out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out / PROVENANCE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
