"""Extract UK business capital expenditure by asset and by section (fixture only).

Design note: ``docs/uk-firm-capex-design.md``. Nothing reads this fixture in a
build: profit is drawn after capital allowances, so feeding capex to
policyengine-uk would deduct allowances twice.

Writes ``packages/microcosm-build/tests/fixtures/uk/firm_capex_2024/``:

- ``business_investment_by_asset.csv``: ONS Business investment by asset,
  sheet "CP (Annual)", the 2024 row (pounds million, current prices) for
  NPEK, EDRB, EDWH, EDYP and EDYS;
- ``abs_capex_by_section.csv``: ONS Annual Business Survey 2024 results,
  sheet "Sections A to S", 2024 capital expenditure acquisitions and
  disposals by SIC section.

    uv run --no-sync python tools/extract_uk_firm_capex_tables.py \
        --bi-xlsx <download>/businessinvestmentbyasset.xlsx \
        --abs-xlsx <download>/abssectionsas.xlsx [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import openpyxl
import pandas as pd

FIXTURE = Path("packages/microcosm-build/tests/fixtures/uk/firm_capex_2024")
BI_SHA256 = "f206fa785e4b1f1f04fef31dbcafd4b8b480af4029a20b5f438383fa7c521c2c"
ABS_SHA256 = "9e548c5c410b73e5a7aec1f97db8f92843430cb6ef5457b1459a0e0e9b13e710"
YEAR = 2024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_business_investment(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb["CP (Annual)"].iter_rows(values_only=True))
    titles, ids = rows[2], rows[3]
    row = next(r for r in rows[4:] if r[0] == YEAR)
    out = [
        {
            "series_id": ids[i],
            "asset": str(titles[i]).split(" [")[0],
            "year": YEAR,
            "value_gbp_million": float(row[i]),
        }
        for i in range(1, len(ids))
        if ids[i]
    ]
    frame = pd.DataFrame(out)
    parts = frame[frame.series_id != "NPEK"].value_gbp_million.sum()
    total = frame[frame.series_id == "NPEK"].value_gbp_million.iloc[0]
    if abs(parts - total) > 0.5:
        raise ValueError(f"asset classes sum to {parts}, not NPEK {total}")
    return frame


def extract_abs_capex(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb["Sections A to S"].iter_rows(values_only=True))
    header = rows[6]
    acq = next(
        i
        for i, h in enumerate(header)
        if str(h).startswith("Total capital expenditure acquisitions")
    )
    dis = next(
        i
        for i, h in enumerate(header)
        if str(h).startswith("Total capital expenditure disposals")
    )
    out = [
        {
            "sic_section": r[0].strip(),
            "year": YEAR,
            "acquisitions_gbp_million": float(r[acq]),
            "disposals_gbp_million": float(r[dis]),
        }
        for r in rows[8:]
        if r[2] == YEAR and isinstance(r[0], str)
    ]
    return pd.DataFrame(out)


def _csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    return buffer.getvalue().encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bi-xlsx", type=Path, required=True)
    parser.add_argument("--abs-xlsx", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if _sha256(args.bi_xlsx) != BI_SHA256 or _sha256(args.abs_xlsx) != ABS_SHA256:
        print("a workbook's sha256 differs from the pinned download", file=sys.stderr)
        return 1
    tables = {
        "business_investment_by_asset.csv": extract_business_investment(args.bi_xlsx),
        "abs_capex_by_section.csv": extract_abs_capex(args.abs_xlsx),
    }
    bad = []
    for name, frame in tables.items():
        target = args.out_dir / name
        if args.check:
            if target.read_bytes() != _csv(frame):
                bad.append(name)
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_csv(frame))
            print(f"wrote {name} ({len(frame)} rows)")
    if bad:
        print(f"differ from a fresh extraction: {bad}", file=sys.stderr)
        return 1
    if args.check:
        print("ok: extraction matches the committed CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
