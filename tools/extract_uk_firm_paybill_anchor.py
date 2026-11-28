"""Extract the UK paybill anchors (report-only, ``firm_paybill_anchor.py``).

Writes ``packages/microcosm-build/tests/fixtures/uk/firm_paybill_anchor_2024/
paybill_anchors.csv`` from two downloads:

- the ONS time-series CSV for DTWL (UK wages and salaries, ESA D.11, uses,
  current prices, pounds million, seasonally adjusted), Quarterly National
  Accounts dataset: the calendar-year 2024 row;
- the ONS/HMRC PAYE Real Time Information reference table, not seasonally
  adjusted (``rtinsanov2026.xlsx``): sheet "4. Aggregate pay (UK)", the
  twelve months April 2024 to March 2025 summed (no annual figure is
  published), and sheet "1. Payrolled employees (UK)", their mean.

Values are read, never typed. ``--check`` compares a fresh extraction with
the committed CSV.

    uv run --no-sync python tools/extract_uk_firm_paybill_anchor.py \
        --dtwl-csv <download>/dtwl_qna.csv --rti-xlsx <download>/rtinsanov2026.xlsx [--check]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

import openpyxl

FIXTURE = Path("packages/microcosm-build/tests/fixtures/uk/firm_paybill_anchor_2024")
DTWL_SHA256 = "5948ee2d3da0fe42fab8a27768a18eb6b8ee352e712e02e15c817bf5cb8df1a5"
RTI_SHA256 = "8aafec63ce93a6a8b299c438faf4ce54a535d9ed331c2e2dc30106b8c163748c"
TAX_YEAR = [
    "April 2024", "May 2024", "June 2024", "July 2024", "August 2024", "September 2024",
    "October 2024", "November 2024", "December 2024", "January 2025", "February 2025", "March 2025",
]  # fmt: skip
COLUMNS = ("anchor", "period", "value", "unit", "detail")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dtwl_2024(path: Path) -> float:
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip() == "2024":
                return float(row[1])
    raise ValueError("DTWL CSV has no 2024 row")


def _monthly(wb: openpyxl.Workbook, sheet: str) -> dict[str, float]:
    out = {}
    for row in wb[sheet].iter_rows(values_only=True):
        if isinstance(row[0], str) and isinstance(row[1], (int, float)):
            out[row[0].strip()] = float(row[1])
    missing = [m for m in TAX_YEAR if m not in out]
    if missing:
        raise ValueError(f"{sheet} lacks {missing}")
    return out


def extract(dtwl: Path, rti: Path) -> list[tuple]:
    for path, expected in ((dtwl, DTWL_SHA256), (rti, RTI_SHA256)):
        if _sha256(path) != expected:
            raise ValueError(f"{path.name} sha256 is not the pinned {expected}")
    wb = openpyxl.load_workbook(rti, read_only=True, data_only=True)
    pay = _monthly(wb, "4. Aggregate pay (UK)")
    employees = _monthly(wb, "1. Payrolled employees (UK)")
    return [
        ("ons_d11_wages_and_salaries", "2024", _dtwl_2024(dtwl) * 1e6, "GBP",
         "ONS DTWL, UK wages and salaries (D.11), uses, current prices, SA; QNA"),
        ("paye_rti_aggregate_pay", "2024-25", sum(pay[m] for m in TAX_YEAR), "GBP",
         "ONS/HMRC PAYE RTI aggregate pay, UK, NSA; sum of April 2024 to March 2025"),
        ("paye_rti_payrolled_employees", "2024-25", sum(employees[m] for m in TAX_YEAR) / 12, "persons",
         "ONS/HMRC PAYE RTI payrolled employees, UK, NSA; mean of April 2024 to March 2025"),
    ]  # fmt: skip


def _csv(rows: list[tuple]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for anchor, period, value, unit, detail in rows:
        writer.writerow((anchor, period, f"{value:.2f}", unit, detail))
    return buffer.getvalue().encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dtwl-csv", type=Path, required=True)
    parser.add_argument("--rti-xlsx", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = _csv(extract(args.dtwl_csv, args.rti_xlsx))
    target = args.out_dir / "paybill_anchors.csv"
    if args.check:
        if target.read_bytes() != payload:
            print(
                "paybill_anchors.csv differs from a fresh extraction", file=sys.stderr
            )
            return 1
        print("ok: extraction matches the committed CSV")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
