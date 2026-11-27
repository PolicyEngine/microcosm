"""Extract the VAT composition inputs for UK firms (opt-in, ``firm_vat_composition.py``).

Reads two downloaded workbooks and writes the extracted CSVs beside the
hand-written product treatment table in
``packages/microcosm-build/tests/fixtures/uk/firm_vat_composition_2024/``:

- ``sut_intermediate_consumption.csv``: ONS Supply and Use Tables 1997-2024
  (Blue Book 2026), sheet "Table 2 - Int Con 2024", the 'Combined Use'
  intermediate consumption matrix in long form (product, industry, pounds
  million). Industries are written under the product code they make: the
  sheet's 103 industry columns align one-to-one, in order, with its 103
  product rows. Each industry code must equal its product's code, or be one
  of the three spellings in ``INDUSTRY_ALIASES``; the tool refuses the
  workbook otherwise.
- ``sut_dwellings_gfcf.csv``: sheet "Table 4 - GFCF 2024", the Dwellings
  column of the construction product row (``CPA_F41, F42 & F43``) and the
  row's total GFCF.
- ``abs_section_turnover.csv``: ONS Annual Business Survey 2024 results,
  sheet "Sections A to S", the 2024 rows (turnover, approximate GVA and
  total purchases by SIC section, pounds million), and the 2024 division
  rows of sheets "Division 45", "Division 46" and "Division 47" (written
  with their division code in ``sic_section``).
- ``hmrc_vat_relief_costs.csv``: HMRC Tax relief statistics (January 2026),
  sheet "Table_2", the 2024-25 cost (pounds million) of the VAT reliefs
  ``RELIEF_CODES``, with their first forecast year. Checks only.

No value is typed by hand. ``--check`` re-extracts into memory and compares
with the committed CSVs and the sha256 values recorded in
``provenance.json``; it needs the same two downloads.

    uv run --no-sync python tools/extract_uk_firm_vat_composition_tables.py \
        --sut-xlsx <download>/supublicationtablesbb26.xlsx \
        --abs-xlsx <download>/abssectionsas.xlsx \
        --relief-ods <download>/tax_relief_statistics_january_2026.ods [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import openpyxl
import pandas as pd

FIXTURE = Path("packages/microcosm-build/tests/fixtures/uk/firm_vat_composition_2024")
SUT_SHA256 = "716c9e023e95cad60d3ca6affb058540887f436ad21a2f3c0f8f01eec0bbeaa8"
ABS_SHA256 = "9e548c5c410b73e5a7aec1f97db8f92843430cb6ef5457b1459a0e0e9b13e710"
RELIEF_SHA256 = "fa94a61fe6b6a02c19b87a656154f27eaf5f00f35f0cce9339b8101b9ac8ee4a"
RELIEF_SHEET = "Table_2"
RELIEF_YEAR = "2024 to 2025"
RELIEF_CODES = (
    "VAT - NS1",
    "VAT - NS2",
    "VAT - NS3",
    "VAT - NS4",
    "VAT - NS5",
    "VAT - NS6",
    "VAT - NS7",
    "VAT - NS16",
)
#: Industry codes the Int Con sheet spells differently from their product.
INDUSTRY_ALIASES = {
    "C1101T1106 & C12": "C11.01-6 & C12",
    "C241T243": "C241_3",
    "H493T495": "H493_5",
}
ABS_DIVISION_SHEETS = ("Division 45", "Division 46", "Division 47")
INT_CON_SHEET = "Table 2 - Int Con 2024"
GFCF_SHEET = "Table 4 - GFCF 2024"
ABS_SHEET = "Sections A to S"
ABS_YEAR = 2024
CONSTRUCTION = "CPA_F41, F42 & F43"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: object) -> float:
    if value in (None, "", "..", "-"):
        return 0.0
    return float(value)


def _norm(code: object) -> str:
    return " ".join(str(code).replace("  ", " ").split())


def _product_rows(rows: list[tuple]) -> list[tuple]:
    return [r for r in rows if isinstance(r[0], str) and r[0].startswith("CPA_")]


def extract_intermediate_consumption(wb: openpyxl.Workbook) -> pd.DataFrame:
    rows = list(wb[INT_CON_SHEET].iter_rows(values_only=True))
    industries = [c for c in rows[3][2:] if c is not None]
    products = _product_rows(rows)
    codes = [r[0] for r in products]
    if len(industries) != len(codes):
        raise ValueError(f"{len(industries)} industries for {len(codes)} products")
    for industry, product in zip(industries, codes, strict=True):
        code = _norm(industry)
        code = INDUSTRY_ALIASES.get(code, code)
        if code.replace(" ", "") != _norm(product[len("CPA_") :]).replace(" ", ""):
            raise ValueError(f"industry {industry!r} does not align with {product!r}")
    out = []
    for r in products:
        for j, product in enumerate(codes):
            value = _number(r[2 + j])
            if value != 0.0:
                out.append((r[0], product, value))
    return pd.DataFrame(
        out, columns=["product_code", "industry_product_code", "value_gbp_million"]
    )


def extract_dwellings(wb: openpyxl.Workbook) -> pd.DataFrame:
    rows = list(wb[GFCF_SHEET].iter_rows(values_only=True))
    header = rows[4]
    dwellings = [i for i, h in enumerate(header) if h == "Dwellings"]
    total = [i for i, h in enumerate(header) if h == "Total GFCF"]
    if len(dwellings) != 1 or len(total) != 1:
        raise ValueError("GFCF sheet lacks one Dwellings and one Total GFCF column")
    row = [r for r in rows if r[0] == CONSTRUCTION]
    if len(row) != 1:
        raise ValueError(f"GFCF sheet lacks one {CONSTRUCTION} row")
    return pd.DataFrame(
        [
            {
                "product_code": CONSTRUCTION,
                "dwellings_gfcf_gbp_million": _number(row[0][dwellings[0]]),
                "total_gfcf_gbp_million": _number(row[0][total[0]]),
            }
        ]
    )


def extract_abs(wb: openpyxl.Workbook) -> pd.DataFrame:
    rows = list(wb[ABS_SHEET].iter_rows(values_only=True))
    header = rows[6]
    want = {
        "Total turnover": "turnover_gbp_million",
        "Approximate gross value added at basic prices (aGVA)": "agva_gbp_million",
        "Total purchases of goods, materials and services": "purchases_gbp_million",
    }
    index = {v: header.index(k) for k, v in want.items()}
    out = []
    for r in rows[8:]:
        if r[2] == ABS_YEAR and isinstance(r[0], str):
            out.append(
                {
                    "sic_section": r[0].strip(),
                    "year": ABS_YEAR,
                    **{col: _number(r[i]) for col, i in index.items()},
                }
            )
    for sheet in ABS_DIVISION_SHEETS:
        division = sheet.split()[-1]
        rows = list(wb[sheet].iter_rows(values_only=True))
        header = rows[6]
        index = {v: header.index(k) for k, v in want.items()}
        match = [
            r for r in rows[8:] if r[2] == ABS_YEAR and str(r[0]).strip() == division
        ]
        if len(match) != 1:
            raise ValueError(f"{sheet} has no single 2024 division row")
        out.append(
            {
                "sic_section": division,
                "year": ABS_YEAR,
                **{col: _number(match[0][i]) for col, i in index.items()},
            }
        )
    frame = pd.DataFrame(out)
    if not {"G", "45", "46", "47"} <= set(frame["sic_section"]):
        raise ValueError("ABS sheets lack the 2024 section G or division 45-47 rows")
    return frame


def extract_reliefs(path: Path) -> pd.DataFrame:
    table = pd.read_excel(
        path, sheet_name=RELIEF_SHEET, engine="odf", header=None, dtype=str
    )
    header = list(table.iloc[1])
    year = header.index(RELIEF_YEAR)
    out = []
    for code in RELIEF_CODES:
        row = table[table[1] == code]
        if len(row) != 1:
            raise ValueError(f"{RELIEF_SHEET} has no single {code} row")
        r = row.iloc[0]
        out.append(
            {
                "code": code,
                "name": str(r[0]).strip(),
                "first_forecast_year": str(r[4]).strip(),
                "cost_2024_25_gbp_million": float(str(r[year]).replace(",", "")),
            }
        )
    return pd.DataFrame(out)


def extract(sut: Path, abs_path: Path, relief: Path) -> dict[str, pd.DataFrame]:
    for path, expected in (
        (sut, SUT_SHA256),
        (abs_path, ABS_SHA256),
        (relief, RELIEF_SHA256),
    ):
        got = _sha256(path)
        if got != expected:
            raise ValueError(f"{path.name} sha256 {got} is not the pinned {expected}")
    sut_wb = openpyxl.load_workbook(sut, read_only=True, data_only=True)
    abs_wb = openpyxl.load_workbook(abs_path, read_only=True, data_only=True)
    return {
        "sut_intermediate_consumption.csv": extract_intermediate_consumption(sut_wb),
        "sut_dwellings_gfcf.csv": extract_dwellings(sut_wb),
        "abs_section_turnover.csv": extract_abs(abs_wb),
        "hmrc_vat_relief_costs.csv": extract_reliefs(relief),
    }


def _csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.10g")
    return buffer.getvalue().encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sut-xlsx", type=Path, required=True)
    parser.add_argument("--abs-xlsx", type=Path, required=True)
    parser.add_argument("--relief-ods", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    tables = extract(args.sut_xlsx, args.abs_xlsx, args.relief_ods)
    provenance_path = args.out_dir / "provenance.json"
    if args.check:
        provenance = json.loads(provenance_path.read_text())
        bad = []
        for name, frame in tables.items():
            payload = _csv(frame)
            if (args.out_dir / name).read_bytes() != payload:
                bad.append(f"{name} differs from a fresh extraction")
            if (
                provenance["files"][name]["sha256"]
                != hashlib.sha256(payload).hexdigest()
            ):
                bad.append(f"{name} sha256 differs from provenance.json")
        if bad:
            print("\n".join(bad), file=sys.stderr)
            return 1
        print("ok: extraction matches the committed CSVs")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        (args.out_dir / name).write_bytes(_csv(frame))
        print(f"wrote {name} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
