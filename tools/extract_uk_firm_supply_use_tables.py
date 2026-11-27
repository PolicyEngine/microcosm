#!/usr/bin/env python3
"""Extract the ONS Supply and Use Tables (Blue Book 2026) for UK firm consumption links.

Reads the ONS *Input-output supply and use tables* publication workbook
consistent with UK National Accounts 2026 Blue Book (``supublicationtablesbb26
.xlsx``, "Supply and Use Tables, 1997 - 2024"), the year 2024 sheets, and
writes three CSVs of raw values in pounds million (not shares):

``sut_product_uses.csv``
    One row per CPA product (Table 1 and Table 2 rows, 103 products): the
    product's total intermediate consumption (Table 2 - Int Con, column
    "Total intermediate demand"), its final demand by component (Table 2 -
    Final Demand: households, NPISH, central government, local government,
    gross fixed capital formation, valuables, changes in inventories, total
    exports of goods and services), total demand for products, and its supply
    (Table 1: total domestic output at basic prices, total imports of goods
    and services, distributors' trading margins, taxes less subsidies on
    products, total supply at purchasers' prices).

``sut_hhfce_product_coicop.csv``
    Table 3 - HHFCE (domestic concept) in long form: product, COICOP column
    as printed (classes ``01.1`` .. ``09.6``, then divisions ``10``, ``11``,
    ``12``, which the table does not split), pounds million. Zero cells are
    dropped. Table 3 prints Table 1's ``CPA_C27`` and ``CPA_C28`` as one row,
    ``CPA_C27 & C28``, kept under that code.

``sut_product_sic_division.csv``
    Each product's SIC 2007 two-digit division(s). A product code is
    ``CPA_`` then one or more parts joined by ``&`` or ``,``; each part is a
    section letter and a SIC code whose first two digits are the division.
    So ``CPA_C102_3`` (SIC 10.2-3: classes 10.2 and 10.3) is division 10;
    ``CPA_B05 & B06`` is divisions 05 and 06 (two rows); ``CPA_C11.01-6 &
    C12`` is divisions 11 and 12; ``CPA_F41, F42 & F43`` is 41, 42 and 43;
    ``CPA_K65.1-2 & K65.3`` is 65; ``CPA_L68BXL683`` (68B excluding 68.3),
    ``CPA_L68A`` (imputed rent) and ``CPA_L683`` are all 68; ``CPA_C23OTHER``
    is 23. A product spanning several divisions is listed once per division
    with the whole product: the tables give no split. The parse is checked
    against the *Industry and product map* sheet: the same set of divisions
    (the map's 98 and 99, "not explicitly measured", have no product), the
    same section letter per division, and per division as many products as
    the map has distinct A103 (publication level) codes. The map's SIC codes
    and A103 codes of the division are carried for reference.

Identities asserted per product before writing: domestic output + imports +
margins + taxes = total supply; intermediate + final demand = total demand =
total supply; the final demand components sum to total final demand;
Table 3's classes sum to its domestic-concept total, and domestic total +
non-resident expenditure + resident expenditure abroad = the national concept
total = Table 2's households column. Distributors' margins sum to zero.

openpyxl is imported only when a workbook is read, so engine-free CI, which
reads the committed CSVs, never needs it.

Usage::

    uv run --no-sync python tools/extract_uk_firm_supply_use_tables.py \\
        --xlsx <download>/supublicationtablesbb26.xlsx

    # Exit 1 if re-extraction would change the committed CSVs' bytes.
    uv run --no-sync python tools/extract_uk_firm_supply_use_tables.py \\
        --xlsx <download>/supublicationtablesbb26.xlsx --check
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
    / "firm_supply_use_2024"
)
USES_NAME = "sut_product_uses.csv"
HHFCE_NAME = "sut_hhfce_product_coicop.csv"
DIVISION_NAME = "sut_product_sic_division.csv"
PROVENANCE_NAME = "firm_supply_use_2024.provenance.json"

LANDING_PAGE = (
    "https://www.ons.gov.uk/economy/nationalaccounts/supplyandusetables/"
    "datasets/inputoutputsupplyandusetables"
)
XLSX_URL = (
    "https://www.ons.gov.uk/file?uri=/economy/nationalaccounts/"
    "supplyandusetables/datasets/inputoutputsupplyandusetables/current/"
    "supublicationtablesbb26.xlsx"
)
#: The workbook the committed CSVs were extracted from.
XLSX_SHA256 = "716c9e023e95cad60d3ca6affb058540887f436ad21a2f3c0f8f01eec0bbeaa8"
#: The committed fixture's recorded access date: the default for
#: ``--access-date``, which the person who downloads the workbook sets.
ACCESS_DATE = "2026-11-22"

YEAR = 2024
CONTENTS = "Contents"
EDITION_TITLE = "Supply and Use Tables, 1997 - 2024"
EDITION_BASIS = "(consistent with UK National Accounts 2026 Blue Book)"
SUPPLY = f"Table 1 - Supply {YEAR}"
INT_CON = f"Table 2 - Int Con {YEAR}"
FINAL_DEMAND = f"Table 2 - Final Demand {YEAR}"
HHFCE = f"Table 3 - HHFCE {YEAR}"
MAP = "Industry and product map"
SHEETS = (SUPPLY, INT_CON, FINAL_DEMAND, HHFCE, MAP)

#: Table 1 column -> header text (whitespace-normalised) it must sit under.
SUPPLY_COLUMNS = {
    "domestic_output": "Total domestic output of products at basic prices",
    "imports": "Total imports of goods and services",
    "distributors_margins": "Distributors' Trading Margins",
    "taxes_less_subsidies": "Taxes less subsidies on products",
    "total_supply": "Total supply of products at purchasers' prices",
}
#: Table 2 final demand column -> header text.
FINAL_DEMAND_COLUMNS = {
    "households": "Households",
    "npish": "Non-profit institutions serving households",
    "central_government": "Central government",
    "local_government": "Local government",
    "gfcf": "Gross fixed capital formation",
    "valuables": "Valuables",
    "changes_in_inventories": "Changes in inventories",
    "exports": "Total exports of goods and services",
    "total_final_demand": "Total final demand",
    "total_demand": "Total demand for products",
}
INT_CON_TOTAL = "Total intermediate demand"
HHFCE_DOMESTIC = "Total (domestic concept)"
HHFCE_NON_RESIDENT = "Non-resident household expenditure in the UK"
HHFCE_ABROAD = "UK resident household expenditure abroad"
HHFCE_NATIONAL = "Total HHFCE (national concept)"

USES_HEADER = (
    "product_code",
    "product_label",
    "intermediate_consumption",
    "households",
    "npish",
    "central_government",
    "local_government",
    "gfcf",
    "valuables",
    "changes_in_inventories",
    "exports",
    "total_demand",
    "domestic_output",
    "imports",
    "distributors_margins",
    "taxes_less_subsidies",
    "total_supply",
)
HHFCE_HEADER = ("product_code", "coicop", "value_gbp_million")
DIVISION_HEADER = (
    "product_code",
    "sic_division",
    "sic_section",
    "map_sic_codes",
    "map_a103_codes",
)
_FINAL_COMPONENTS = (
    "households",
    "npish",
    "central_government",
    "local_government",
    "gfcf",
    "valuables",
    "changes_in_inventories",
    "exports",
)


def _text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def _number(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    if value != int(value):
        raise ValueError(f"{where}: expected whole pounds million, got {value!r}")
    return int(value)


def _column(header: Sequence[Any], label: str, sheet: str) -> int:
    hits = [i for i, cell in enumerate(header) if _text(cell) == label]
    if len(hits) != 1:
        raise ValueError(f"{sheet}: header {label!r} found {len(hits)}x")
    return hits[0]


def _product_rows(rows: list[tuple[Any, ...]]) -> dict[str, tuple[Any, ...]]:
    out: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        code = _text(row[0])
        if code.startswith("CPA_"):
            if code in out:
                raise ValueError(f"product {code} appears twice")
            out[code] = row
    return out


def _header_row(rows: list[tuple[Any, ...]], label: str, sheet: str) -> int:
    for index, row in enumerate(rows[:8]):
        if any(_text(cell) == label for cell in row):
            return index
    raise ValueError(f"{sheet}: no header row with {label!r}")


def product_divisions(code: str) -> list[tuple[str, str]]:
    """``(section, two-digit division)`` of each part of a CPA product code."""

    if not code.startswith("CPA_"):
        raise ValueError(f"{code!r} is not a CPA product code")
    out: list[tuple[str, str]] = []
    for part in re.split(r"[&,]", code[len("CPA_") :]):
        match = re.fullmatch(r"([A-U])(\d{2})\S*", part.strip())
        if match is None:
            raise ValueError(f"cannot read a SIC division from {part!r} in {code}")
        pair = (match.group(1), match.group(2))
        if pair not in out:
            out.append(pair)
    return out


def table3_products(code: str, products: Sequence[str]) -> list[str]:
    """The Table 1 products a Table 3 row covers.

    Table 3 prints some Table 1 products as one row, ``CPA_C27 & C28`` for
    ``CPA_C27`` and ``CPA_C28``; any other row is its own product.
    """

    if code in products:
        return [code]
    parts = ["CPA_" + part.strip() for part in code[len("CPA_") :].split("&")]
    if len(parts) < 2 or any(part not in products for part in parts):
        raise ValueError(f"Table 3 product {code} is not a union of Table 1 products")
    return parts


def _map_divisions(
    rows: list[tuple[Any, ...]],
) -> dict[str, tuple[str, list[str], list[str]]]:
    """Map sheet: division -> (section, SIC codes, distinct A103 codes)."""

    header = _header_row(rows, "A103", MAP)
    columns = [_text(cell) for cell in rows[header]]
    a21 = columns.index("A21")
    a103 = columns.index("A103")
    section = ""
    last_a103: Any = None
    out: dict[str, tuple[str, list[str], list[str]]] = {}
    for row in rows[header + 2 :]:
        raw = row[0]
        code = _text(raw)
        if not code or code.startswith("*") or not code[:2].isdigit():
            continue
        if isinstance(raw, float):
            # 10.1, 11.07, 33.15: openpyxl gives the printed decimal back.
            code = repr(raw)
        elif isinstance(raw, int):
            code = f"{raw:02d}"
        if _text(row[a21]):
            section = _text(row[a21])
        if row[a103] is not None:
            last_a103 = row[a103]
        if code.endswith("*"):
            # 98*, 99*: "not explicitly measured", no A103 code of their own.
            out.setdefault(code[:2], (section, [code], []))
            continue
        division = code[:2]
        entry = out.setdefault(division, (section, [], []))
        if entry[0] != section:
            raise ValueError(f"division {division} spans sections")
        entry[1].append(code)
        label = str(last_a103)
        if label not in entry[2]:
            entry[2].append(label)
    return out


def extract_tables(
    sheets: dict[str, list[tuple[Any, ...]]],
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """The three CSVs' rows from the five sheets' cell rows."""

    supply_rows = sheets[SUPPLY]
    if f"Supply of products in {YEAR}" not in _text(supply_rows[0][1]):
        raise ValueError(f"{SUPPLY}: unexpected title {supply_rows[0][1]!r}")
    s_head = supply_rows[_header_row(supply_rows, "Product", SUPPLY)]
    s_col = {k: _column(s_head, v, SUPPLY) for k, v in SUPPLY_COLUMNS.items()}
    supply = _product_rows(supply_rows)

    fd_rows = sheets[FINAL_DEMAND]
    fd_head = fd_rows[_header_row(fd_rows, "Product", FINAL_DEMAND)]
    fd_col = {
        k: _column(fd_head, v, FINAL_DEMAND) for k, v in FINAL_DEMAND_COLUMNS.items()
    }
    final = _product_rows(fd_rows)

    ic_rows = sheets[INT_CON]
    ic_head = ic_rows[_header_row(ic_rows, INT_CON_TOTAL, INT_CON)]
    ic_total = _column(ic_head, INT_CON_TOTAL, INT_CON)
    intermediate = _product_rows(ic_rows)

    hh_rows = sheets[HHFCE]
    hh_label_row = _header_row(hh_rows, HHFCE_DOMESTIC, HHFCE)
    hh_labels = hh_rows[hh_label_row]
    hh_codes = [_text(cell) for cell in hh_rows[hh_label_row - 1]]
    hh_domestic = _column(hh_labels, HHFCE_DOMESTIC, HHFCE)
    hh_non_resident = _column(hh_labels, HHFCE_NON_RESIDENT, HHFCE)
    hh_abroad = _column(hh_labels, HHFCE_ABROAD, HHFCE)
    hh_national = _column(hh_labels, HHFCE_NATIONAL, HHFCE)
    coicop_columns = [
        (i, code)
        for i, code in enumerate(hh_codes)
        if i < hh_domestic and re.fullmatch(r"\d{2}(\.\d)?", code)
    ]
    if len({code[:2] for _, code in coicop_columns}) != 12:
        raise ValueError(f"{HHFCE}: expected COICOP divisions 01-12")
    household = _product_rows(hh_rows)

    codes = list(supply)
    for name, table in ((FINAL_DEMAND, final), (INT_CON, intermediate)):
        if sorted(table) != sorted(codes):
            raise ValueError(f"{name} products differ from {SUPPLY}'s")
    # Table 3 merges some Table 1 products into one row (CPA_C27 & C28).
    covered = {code: table3_products(code, codes) for code in household}
    flat = sorted(p for parts in covered.values() for p in parts)
    if flat != sorted(codes):
        raise ValueError(f"{HHFCE} products do not partition {SUPPLY}'s")
    households_by_code: dict[str, int] = {}

    uses: list[tuple[Any, ...]] = []
    hhfce: list[tuple[Any, ...]] = []
    margins = 0
    for code in codes:
        s = {k: _number(supply[code][c], f"{code} {k}") for k, c in s_col.items()}
        f = {k: _number(final[code][c], f"{code} {k}") for k, c in fd_col.items()}
        ic = _number(intermediate[code][ic_total], f"{code} intermediate")
        if (
            s["domestic_output"]
            + s["imports"]
            + s["distributors_margins"]
            + s["taxes_less_subsidies"]
            != s["total_supply"]
        ):
            raise ValueError(f"{code}: supply components do not add up")
        if sum(f[k] for k in _FINAL_COMPONENTS) != f["total_final_demand"]:
            raise ValueError(f"{code}: final demand components do not add up")
        if ic + f["total_final_demand"] != f["total_demand"]:
            raise ValueError(f"{code}: intermediate + final != total demand")
        if f["total_demand"] != s["total_supply"]:
            raise ValueError(f"{code}: total demand != total supply")
        margins += s["distributors_margins"]
        households_by_code[code] = f["households"]
        uses.append(
            (
                code,
                _text(supply[code][1]),
                ic,
                *(f[k] for k in _FINAL_COMPONENTS),
                f["total_demand"],
                *(
                    s[k]
                    for k in (
                        "domestic_output",
                        "imports",
                        "distributors_margins",
                        "taxes_less_subsidies",
                        "total_supply",
                    )
                ),
            )
        )
    if margins != 0:
        raise ValueError(f"distributors' margins sum to {margins}, not 0")

    for code, row in household.items():
        classes = 0
        for column, coicop in coicop_columns:
            value = _number(row[column], f"{code} {coicop}")
            classes += value
            if value != 0:
                hhfce.append((code, coicop, value))
        domestic = _number(row[hh_domestic], f"{code} domestic")
        national = _number(row[hh_national], f"{code} national")
        if classes != domestic:
            raise ValueError(f"{code}: Table 3 classes != domestic total")
        if (
            domestic
            + _number(row[hh_non_resident], f"{code} non-resident")
            + _number(row[hh_abroad], f"{code} abroad")
            != national
        ):
            raise ValueError(f"{code}: Table 3 domestic + adjustments != national")
        if national != sum(households_by_code[p] for p in covered[code]):
            raise ValueError(f"{code}: Table 3 national != Table 2 households")

    mapped = _map_divisions(sheets[MAP])
    divisions: list[tuple[Any, ...]] = []
    products_per_division: dict[str, int] = {}
    for code in codes:
        for section, division in product_divisions(code):
            if division not in mapped:
                raise ValueError(f"{code}: division {division} not in {MAP}")
            map_section, sic_codes, a103 = mapped[division]
            if map_section != section:
                raise ValueError(
                    f"{code}: section {section} but {MAP} has {map_section}"
                )
            products_per_division[division] = products_per_division.get(division, 0) + 1
            divisions.append(
                (code, division, section, ";".join(sic_codes), ";".join(a103))
            )
    for division, (_, _, a103) in mapped.items():
        found = products_per_division.get(division, 0)
        if found != len(a103):
            raise ValueError(
                f"division {division}: {found} products, {MAP} has "
                f"{len(a103)} A103 codes"
            )
    divisions.sort(key=lambda row: (row[1], codes.index(row[0])))
    return uses, hhfce, divisions


def to_csv_bytes(header: Sequence[str], rows: list[tuple[Any, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read(xlsx: Path) -> dict[str, list[tuple[Any, ...]]]:
    import openpyxl  # noqa: PLC0415 - local-only dependency, kept out of CI imports

    book = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        return {
            name: list(book[name].iter_rows(values_only=True))
            for name in (CONTENTS, *SHEETS)
        }
    finally:
        book.close()


def edition(contents: list[tuple[Any, ...]]) -> str:
    """The Contents sheet's title, basis and release month."""

    title = [_text(cell) for row in contents[:6] for cell in row if _text(cell)]
    if EDITION_TITLE not in title or EDITION_BASIS not in title:
        raise ValueError(f"workbook is not {EDITION_TITLE!r} {EDITION_BASIS!r}")
    month = title[title.index(EDITION_BASIS) + 1]
    if not re.fullmatch(r"[A-Z][a-z]+ \d{4}", month):
        raise ValueError(f"no release month after the basis line: {month!r}")
    return f"{EDITION_TITLE} {EDITION_BASIS}, released {month}"


def payloads(
    sheets: dict[str, list[tuple[Any, ...]]],
) -> tuple[dict[str, bytes], dict[str, int]]:
    uses, hhfce, divisions = extract_tables(sheets)
    files = {
        USES_NAME: to_csv_bytes(USES_HEADER, uses),
        HHFCE_NAME: to_csv_bytes(HHFCE_HEADER, hhfce),
        DIVISION_NAME: to_csv_bytes(DIVISION_HEADER, divisions),
    }
    counts = {
        USES_NAME: len(uses),
        HHFCE_NAME: len(hhfce),
        DIVISION_NAME: len(divisions),
    }
    return files, counts


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
    release: str,
    files: dict[str, bytes],
    counts: dict[str, int],
    access_date: str = ACCESS_DATE,
) -> dict[str, Any]:
    workbook = xlsx.read_bytes()
    headers = {
        USES_NAME: USES_HEADER,
        HHFCE_NAME: HHFCE_HEADER,
        DIVISION_NAME: DIVISION_HEADER,
    }
    return {
        "kind": "uk_firm_supply_use_fixture",
        "schema_version": 1,
        "period": f"calendar year {YEAR}",
        "accessed": access_date,
        "method": (
            "Figures were read from the downloaded workbook on the access date "
            "by tools/extract_uk_firm_supply_use_tables.py; none was filled "
            "from memory. Values are pounds million at current prices as "
            "published. Demand is at purchasers' prices (the 'Combined Use' "
            "matrix, domestic output and imports together); domestic output "
            "is at basic prices. The product -> SIC division rows are parsed "
            "from the CPA product codes and checked against the Industry and "
            "product map sheet; a product spanning several divisions is "
            "repeated once per division with the whole product."
        ),
        "identities_checked": [
            "domestic output + imports + distributors' margins + taxes less "
            "subsidies = total supply (each product)",
            "intermediate consumption + total final demand = total demand = "
            "total supply (each product)",
            "final demand components sum to total final demand (each product)",
            "Table 3 COICOP columns sum to Total (domestic concept); domestic + "
            "non-resident expenditure in the UK + resident expenditure abroad "
            "= Total HHFCE (national concept) = Table 2 households (each "
            "product)",
            "distributors' margins sum to zero over products",
            "products per SIC division = distinct A103 codes per division in "
            "the Industry and product map",
        ],
        "notes": [
            "Table 2's households column is the national concept (includes UK "
            "residents' spending abroad, excludes non-residents' spending in "
            "the UK, which sits in exports); Table 3's COICOP columns are the "
            "domestic concept.",
            "Table 3 publishes COICOP classes for divisions 01-09 and only "
            "division totals for 10, 11 and 12 (no 12.6.1 FISIM class).",
            "Distributors' trading margins are negative only on the trade "
            "products CPA_G45, CPA_G46 and CPA_G47 and are spread over the "
            "goods products (and publishing products J58, J59 & J60) as "
            "positive margins; one combined column, not split by wholesale, "
            "retail and motor trade.",
            "Table 3 footnote: COICOP data here may differ from Consumer "
            "Trends because of the CPA to COICOP correlator; at the total "
            "level they are consistent.",
            "Table 3 has 102 product rows: it prints CPA_C27 and CPA_C28 as "
            "one row, CPA_C27 & C28, written under that code.",
            f"Zero cells of Table 3 are not written to {HHFCE_NAME}.",
        ],
        "source": {
            "publisher": "Office for National Statistics",
            "dataset_title": "Input-output supply and use tables",
            "landing_page": LANDING_PAGE,
            "url": XLSX_URL,
            "edition": release,
            "file": xlsx.name,
            "size_bytes": len(workbook),
            "sha256": _sha256(workbook),
            "sheets": list(SHEETS),
            "units": "pounds million, current prices",
            "licence": "Open Government Licence v3.0",
        },
        "extraction_date": dt.date.today().isoformat(),
        "tool": "tools/extract_uk_firm_supply_use_tables.py",
        "command": (
            "uv run --no-sync python tools/extract_uk_firm_supply_use_tables.py "
            "--xlsx <download>/supublicationtablesbb26.xlsx"
        ),
        "files": {
            name: {
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "rows": counts[name],
                "columns": list(headers[name]),
            }
            for name, payload in files.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xlsx", type=Path, required=True)
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
    xlsx = args.xlsx.resolve()
    digest = _sha256(xlsx.read_bytes())
    if digest != XLSX_SHA256:
        print(
            f"warning: workbook sha256 {digest} is not the pinned {XLSX_SHA256}",
            file=sys.stderr,
        )
    sheets = _read(xlsx)
    release = edition(sheets[CONTENTS])
    files, counts = payloads(sheets)
    if args.check:
        stale = [
            name
            for name, payload in files.items()
            if not (args.out / name).is_file()
            or (args.out / name).read_bytes() != payload
        ]
        for name in stale:
            print(f"stale or missing: {name}", file=sys.stderr)
        if stale:
            return 1
        print("supply and use CSVs match the workbook")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (args.out / name).write_bytes(payload)
        print(f"wrote {args.out / name} ({counts[name]} rows)")
    record = provenance(xlsx, release, files, counts, args.access_date)
    (args.out / PROVENANCE_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out / PROVENANCE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
