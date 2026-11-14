#!/usr/bin/env python3
"""Build 2024-25 Band D council tax for England's and Wales's authorities (microcosm#931).

Writes ``uk/council_tax_band_d_england_wales_2024_25.json``: each billing
authority's Band D council tax for 2024-25, keyed by its April 2023 local
authority code on the roster ``local_authority_names.json``, through which the
spine resolves every drawn area. Scotland's 2024-25 Band D is vendored from
the pinned Chronicle feed by ``tools/vendor_uk_ledger_facts.py``; Northern
Ireland has domestic rates and no Band D.

- England: MHCLG, *Council Tax levels set by local authorities in England 2024
  to 2025*, Table 8: area council tax for a dwelling occupied by two adults,
  adult social care and parish precepts included. The sheet is read with the
  standard-library reader ``ods_tables.read_ods_tables``, which gives each
  amount as the binary float the sheet stores. Every amount must lie within
  1e-9 of a whole number of pence and is kept as that pence decimal.
- Wales: StatsWales, *Composition of average band D council tax by billing
  authority*: the 2024-25 rows "Total CT for billing authority area" in pounds
  per band D (the county council element with community councils, plus the
  police element), read with the ``csv`` module and kept as the published
  decimal strings. The 22 authorities are joined to their codes by exact name;
  the all-Wales row "Total Unitary Authorities" is a declared exclusion.

Both files are chronicle#412's raw-only registrations (PR chronicle#413),
pinned here by sha256 and size at the Chronicle commit the UK feed pinned when
this resource was built. Each local file is held to its pin and to its
Chronicle manifest, else refused. The authorities read must be exactly the
roster's English and Welsh codes: a duplicate, a missing authority, an
unmatched name, a Northern Ireland code or a note on a row read is refused,
naming the column, the code or the name, never an amount. Only published
aggregates are read or written.

    uv run --no-sync python tools/build_uk_council_tax_band_d.py \\
        --england-ods <dir>/Table_8_24-25.ods \\
        --england-manifest <chronicle>/db/data/mhclg/council_tax_levels_england_2024_25_table_8/manifest.yaml \\
        --wales-csv <dir>/statswales_band_d_council_tax_composition_by_billing_authority.csv \\
        --wales-manifest <chronicle>/db/data/welshgov/band_d_council_tax_composition_1993_94_to_2026_27/manifest.yaml \\
        --built-on YYYY-MM-DD [--check]

The manifests are the registrations' ``manifest.yaml`` at the pinned Chronicle
commit. ``--check`` rebuilds the resource and compares it with the committed
one, ``built_on`` aside.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml

from microcosm.build.uk_runtime.council_tax_band_d import (
    UK_BAND_D_ENGLAND_WALES_KIND,
    UK_BAND_D_ENGLAND_WALES_METHOD,
    UK_BAND_D_ENGLAND_WALES_RESOURCE,
    UK_BAND_D_ENGLAND_WALES_SCHEMA_VERSION,
    UK_BAND_D_FISCAL_YEAR,
    UK_BAND_D_PENCE_TOLERANCE_TEXT,
    band_d_decimal,
    band_d_pence,
)
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_VINTAGE,
    load_uk_local_authority_names_resource,
)
from microcosm.build.uk_runtime.ods_tables import read_ods_tables

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
DEFAULT_OUT = UK_PACKAGE / UK_BAND_D_ENGLAND_WALES_RESOURCE

#: The Chronicle commit the registrations are read at: the one
#: ``uk/chronicle_feed.json`` pinned when this resource was built. Both
#: manifests and both files are there as their registering merge left them.
CHRONICLE_COMMIT = "80a3482905843d447f8811b500a56dbf3b5f47fa"
_REGISTERED_BY = {
    "issue": "PolicyEngine/chronicle#412",
    "pull_request": "PolicyEngine/chronicle#413",
    "merge_commit": "c51f40d2bdaa66315dd63f334ede5790a8891e27",
}


def _chronicle_manifest(path: str) -> dict[str, str]:
    return {
        "repository": "PolicyEngine/chronicle",
        "path": path,
        "commit": CHRONICLE_COMMIT,
    }


#: Each registration read, as its manifest at ``CHRONICLE_COMMIT`` records it,
#: with what is read from it. The resource carries this mapping as its
#: ``sources``.
SOURCES: dict[str, dict[str, Any]] = {
    "england": {
        "publisher": "Ministry of Housing, Communities and Local Government",
        "chronicle_source_id": "mhclg",
        "chronicle_package_id": "mhclg-council-tax-levels-england-2024-25-table-8",
        "chronicle_year": 2024,
        "chronicle_manifest": _chronicle_manifest(
            "db/data/mhclg/council_tax_levels_england_2024_25_table_8/manifest.yaml"
        ),
        "registration_kind": "raw_only",
        "registered_by": dict(_REGISTERED_BY),
        "table": (
            "Council Tax levels set by local authorities in England 2024 to 2025, "
            "Table 8: Area council tax for a dwelling occupied by 2 adults by band "
            "2024-25"
        ),
        "filename": "Table_8_24-25.ods",
        "url": "https://assets.publishing.service.gov.uk/media/65fad6739316f5001164c3ca/Table_8_24-25.ods",
        "sha256": "249127ae1bff3f37bb335896f801524562de0dd9668c6c9cd64eacd8a8880245",
        "size_bytes": 53_280,
        "selector": {
            "sheet": "Table_8",
            "code_column": "ONS Code",
            "value_column": "Band D",
            "note_column": "Notes",
        },
        "basis": (
            "Area council tax for a Band D dwelling occupied by two adults, in "
            "pounds, adult social care and parish precepts included (the table's "
            "note a). Each amount is stored as a binary float and kept as the pence "
            "decimal it lies within 1e-9 of."
        ),
    },
    "wales": {
        "publisher": "Welsh Government (StatsWales)",
        "chronicle_source_id": "welshgov",
        "chronicle_package_id": (
            "welshgov-band-d-council-tax-composition-1993-94-to-2026-27"
        ),
        "chronicle_year": 2026,
        "chronicle_manifest": _chronicle_manifest(
            "db/data/welshgov/band_d_council_tax_composition_1993_94_to_2026_27/"
            "manifest.yaml"
        ),
        "registration_kind": "raw_only",
        "registered_by": dict(_REGISTERED_BY),
        "table": (
            "Composition of average band D council tax by billing authority "
            "(StatsWales)"
        ),
        "filename": "statswales_band_d_council_tax_composition_by_billing_authority.csv",
        "url": "https://api.stats.gov.wales/v1/3ca971eb-7d9b-4b79-89fa-6d7f77431aad/download/csv",
        "sha256": "b6ea9848bd0fdf4a6310b62a62d614fd4c325b8f8d2e83a4d89eb85886ce552e",
        "size_bytes": 590_478,
        "selector": {
            "columns": [
                "Data values",
                "Data description",
                "Row",
                "Year",
                "Authority",
                "Notes",
            ],
            "rows": {
                "Year": UK_BAND_D_FISCAL_YEAR,
                "Row": "Total CT for billing authority area",
                "Data description": "\N{POUND SIGN} per band D",
            },
            "name_column": "Authority",
            "value_column": "Data values",
            "note_column": "Notes",
            "excluded_names": ["Total Unitary Authorities"],
        },
        "basis": (
            "Average band D council tax of the billing authority area (the county "
            "council element with community councils, plus the police element), in "
            "pounds, unrounded as published. Each authority is joined to its April "
            "2023 code by its exact name on the local authority roster."
        ),
    },
}

#: The sources' licence and the attribution it asks for on derived data.
LICENCE = {
    "licence": "Open Government Licence v3.0",
    "attribution": (
        "Contains public sector information licensed under the Open Government "
        "Licence v3.0. Sources: Ministry of Housing, Communities and Local "
        "Government, Council Tax levels set by local authorities in England 2024 to "
        "2025, Table 8; Welsh Government, StatsWales, Composition of average band D "
        "council tax by billing authority."
    ),
}

DESCRIPTION = (
    "Band D council tax for 2024-25 of every billing authority in England and Wales, "
    "keyed by April 2023 local authority code: England from MHCLG's Table 8 (area "
    "council tax for a dwelling occupied by two adults, at pence), Wales from "
    "StatsWales (the total council tax of the billing authority area, unrounded as "
    "published). Scotland's 2024-25 Band D is vendored from the pinned Chronicle feed "
    "into its own resource; Northern Ireland has domestic rates. Rebuilt by "
    "tools/build_uk_council_tax_band_d.py (microcosm#931)."
)

ENGLAND_LABEL = "MHCLG council tax levels 2024-25, Table 8"
WALES_LABEL = "StatsWales band D council tax composition"


# ----------------------------------------------------------------------------
# Pins


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(
    name: str,
    path: Path,
    manifest_path: Path,
    *,
    pin: Mapping[str, Any] | None = None,
) -> None:
    """Refuse a local file or manifest that is not the pinned registration.

    The manifest must name the pinned source, package, registration kind and
    table, and carry one file entry whose year, filename, url, sha256 and size
    are the pin's; the file's own bytes must have the pinned sha256 and size.
    """

    pin = SOURCES[name] if pin is None else pin
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"The {name} manifest is not a mapping.")
    for field, key in (
        ("source_id", "chronicle_source_id"),
        ("package_id", "chronicle_package_id"),
        ("registration_kind", "registration_kind"),
        ("table", "table"),
    ):
        if manifest.get(field) != pin[key]:
            raise ValueError(f"The {name} manifest's {field} differs from the pin.")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or len(files) != 1:
        raise ValueError(f"The {name} manifest must register exactly one file.")
    ((year, entry),) = files.items()
    if str(year) != str(pin["chronicle_year"]) or not isinstance(entry, Mapping):
        raise ValueError(f"The {name} manifest registers its file under another year.")
    for field, key in (
        ("filename", "filename"),
        ("source_url", "url"),
        ("sha256", "sha256"),
        ("size_bytes", "size_bytes"),
    ):
        if entry.get(field) != pin[key]:
            raise ValueError(
                f"The {name} manifest's files.{year}.{field} differs from the pin."
            )
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"The {name} source must be a regular local file.")
    if path.stat().st_size != pin["size_bytes"] or _sha256(path) != pin["sha256"]:
        raise ValueError(f"The {name} source differs from its pinned sha256 and size.")


# ----------------------------------------------------------------------------
# The roster


def roster() -> tuple[frozenset[str], dict[str, str]]:
    """The roster's English codes, and its Welsh codes by exact name."""

    areas = load_uk_local_authority_names_resource()["areas"]
    english = frozenset(code for code in areas if code.startswith("E"))
    welsh: dict[str, str] = {}
    for code, entry in areas.items():
        if not code.startswith("W"):
            continue
        name = entry["name"]
        if name in welsh:
            raise ValueError(f"The local authority roster repeats the name {name!r}.")
        welsh[name] = code
    return english, welsh


# ----------------------------------------------------------------------------
# England


def _cell(row: tuple[object, ...], column: int) -> object:
    return row[column] if column < len(row) else None


def england_band_d(
    path: Path, *, english_codes: Collection[str]
) -> tuple[dict[str, str], dict[str, int | str]]:
    """Table 8's Band D by ONS code, and how the pence quantisation went.

    Every code in the code column must be one of ``english_codes``, once, and
    every one of them must be present. Rows wholly empty are skipped.
    """

    selector = SOURCES["england"]["selector"]
    sheet = selector["sheet"]
    tables = read_ods_tables(path, label=ENGLAND_LABEL)
    if sheet not in tables:
        raise ValueError(f"{ENGLAND_LABEL} has no sheet {sheet!r}.")
    rows = tables[sheet].rows
    code_column = selector["code_column"]
    value_column = selector["value_column"]
    note_column = selector["note_column"]
    headers = [
        index
        for index, row in enumerate(rows)
        if code_column in row and value_column in row
    ]
    if len(headers) != 1:
        raise ValueError(
            f"{ENGLAND_LABEL} sheet {sheet!r} must hold exactly one header row naming "
            f"{code_column!r} and {value_column!r}."
        )
    header = rows[headers[0]]
    columns: dict[str, int] = {}
    for column in (code_column, value_column, note_column):
        if header.count(column) != 1:
            raise ValueError(
                f"{ENGLAND_LABEL} header must name the column {column!r} exactly once."
            )
        columns[column] = header.index(column)
    english = set(english_codes)
    band_d: dict[str, str] = {}
    exact = 0
    for row in rows[headers[0] + 1 :]:
        if all(cell is None for cell in row):
            continue
        code = _cell(row, columns[code_column])
        if not isinstance(code, str):
            raise ValueError(
                f"{ENGLAND_LABEL}: a row below the header has no {code_column!r}."
            )
        if code.startswith("N"):
            raise ValueError(
                f"{ENGLAND_LABEL}: the column {code_column!r} carries the Northern "
                f"Ireland code {code!r}; Northern Ireland has no council tax."
            )
        if code not in english:
            raise ValueError(
                f"{ENGLAND_LABEL}: the column {code_column!r} carries {code!r}, which "
                "is not an English code on the local authority roster."
            )
        if code in band_d:
            raise ValueError(
                f"{ENGLAND_LABEL}: the column {code_column!r} repeats {code!r}."
            )
        if _cell(row, columns[note_column]) is not None:
            raise ValueError(
                f"{ENGLAND_LABEL}: the column {note_column!r} carries a note for "
                f"{code!r}."
            )
        band_d[code], at_pence = band_d_pence(
            _cell(row, columns[value_column]),
            where=f"{ENGLAND_LABEL}: the column {value_column!r} of {code!r}",
        )
        exact += int(at_pence)
    missing = sorted(english - set(band_d))
    if missing:
        raise ValueError(f"{ENGLAND_LABEL} lacks the roster code(s) {missing}.")
    return band_d, {
        "exact_binary": exact,
        "within_tolerance": len(band_d) - exact,
        "tolerance": UK_BAND_D_PENCE_TOLERANCE_TEXT,
    }


# ----------------------------------------------------------------------------
# Wales


def wales_band_d(
    path: Path, *, welsh_codes_by_name: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """The selected rows' Band D by code, and the declared exclusions applied.

    Each selected row's name must be a declared exclusion or a key of
    ``welsh_codes_by_name``, once, and every key must be present.
    """

    selector = SOURCES["wales"]["selector"]
    name_column = selector["name_column"]
    value_column = selector["value_column"]
    note_column = selector["note_column"]
    excluded = set(selector["excluded_names"])
    selected = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != selector["columns"]:
            raise ValueError(
                f"{WALES_LABEL} must have exactly the columns {selector['columns']}."
            )
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"{WALES_LABEL}: line {reader.line_num} does not have the "
                    "declared columns."
                )
            if all(row[column] == text for column, text in selector["rows"].items()):
                selected.append(row)
    band_d: dict[str, str] = {}
    seen: set[str] = set()
    for row in selected:
        name = row[name_column]
        if name in seen:
            raise ValueError(
                f"{WALES_LABEL}: the column {name_column!r} repeats {name!r} in the "
                "selected rows."
            )
        seen.add(name)
        if name in excluded:
            continue
        code = welsh_codes_by_name.get(name)
        if code is None:
            raise ValueError(
                f"{WALES_LABEL}: the name {name!r} matches no Welsh authority on the "
                "local authority roster."
            )
        if row[note_column]:
            raise ValueError(
                f"{WALES_LABEL}: the column {note_column!r} carries a note for "
                f"{name!r}."
            )
        text = row[value_column]
        band_d_decimal(
            text, where=f"{WALES_LABEL}: the column {value_column!r} of {name!r}"
        )
        band_d[code] = text
    absent = sorted(excluded - seen)
    if absent:
        raise ValueError(
            f"{WALES_LABEL}: the declared exclusion(s) {absent} match no selected row."
        )
    missing = sorted(
        name for name, code in welsh_codes_by_name.items() if code not in band_d
    )
    if missing:
        raise ValueError(
            f"{WALES_LABEL} lacks the roster authority or authorities {missing}."
        )
    return band_d, sorted(excluded)


# ----------------------------------------------------------------------------
# The resource


def band_d_by_code(
    england_ods: Path, wales_csv: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    """England's and Wales's Band D by roster code, and the receipts."""

    english, welsh = roster()
    england, quantisation = england_band_d(england_ods, english_codes=english)
    wales, excluded = wales_band_d(wales_csv, welsh_codes_by_name=welsh)
    band_d = {**england, **wales}
    expected = english | set(welsh.values())
    if set(band_d) != expected or len(band_d) != len(england) + len(wales):
        raise ValueError(
            "The authorities read are not exactly the roster's English and Welsh codes."
        )
    receipts = {
        "authorities": {"england": len(england), "wales": len(wales)},
        "england_values_quantised_to_pence": quantisation,
        "wales_excluded_names": excluded,
    }
    return dict(sorted(band_d.items())), receipts


def _iso_date(text: str) -> str:
    try:
        parsed = dt.date.fromisoformat(text)
    except ValueError:
        parsed = None
    if parsed is None or parsed.isoformat() != text:
        raise ValueError(f"built_on must be a date written YYYY-MM-DD, not {text!r}.")
    return text


def build(
    *,
    england_ods: Path,
    england_manifest: Path,
    wales_csv: Path,
    wales_manifest: Path,
    built_on: str,
) -> dict[str, Any]:
    """The resource payload, after both sources are held to their pins."""

    _iso_date(built_on)
    verify_source("england", england_ods, england_manifest)
    verify_source("wales", wales_csv, wales_manifest)
    band_d, receipts = band_d_by_code(england_ods, wales_csv)
    return {
        "schema_version": UK_BAND_D_ENGLAND_WALES_SCHEMA_VERSION,
        "kind": UK_BAND_D_ENGLAND_WALES_KIND,
        "resource": UK_BAND_D_ENGLAND_WALES_RESOURCE,
        "method": UK_BAND_D_ENGLAND_WALES_METHOD,
        "built_on": built_on,
        "description": DESCRIPTION,
        "fiscal_year": UK_BAND_D_FISCAL_YEAR,
        "local_authority_vintage": UK_LOCAL_AUTHORITY_VINTAGE,
        "sources": json.loads(json.dumps(SOURCES)),
        "licence": dict(LICENCE),
        "receipts": receipts,
        "band_d": band_d,
    }


def render(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def stale(path: Path, payload: Mapping[str, Any]) -> bool:
    """Whether the file at ``path`` is not ``payload`` rendered, ``built_on`` aside."""

    if not path.is_file():
        return True
    text = path.read_text(encoding="utf-8")
    try:
        committed = json.loads(text)
        built_on = _iso_date(committed["built_on"])
    except (ValueError, TypeError, KeyError):
        return True
    return text != render({**payload, "built_on": built_on})


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--england-ods", type=Path, required=True)
    parser.add_argument(
        "--england-manifest",
        type=Path,
        required=True,
        help="the registration's manifest.yaml at the pinned Chronicle commit",
    )
    parser.add_argument("--wales-csv", type=Path, required=True)
    parser.add_argument(
        "--wales-manifest",
        type=Path,
        required=True,
        help="the registration's manifest.yaml at the pinned Chronicle commit",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    def built_on(text: str) -> str:
        try:
            return _iso_date(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    parser.add_argument(
        "--built-on",
        type=built_on,
        required=True,
        help=(
            "build date recorded in the resource (YYYY-MM-DD); passed explicitly so "
            "a rerun reproduces the committed bytes"
        ),
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed resource is stale"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = build(
            england_ods=args.england_ods,
            england_manifest=args.england_manifest,
            wales_csv=args.wales_csv,
            wales_manifest=args.wales_manifest,
            built_on=args.built_on,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f"Refused: {exc}") from exc
    if args.check:
        if stale(args.out, payload):
            raise SystemExit(
                f"{args.out} is stale: rebuild it with "
                "tools/build_uk_council_tax_band_d.py."
            )
        print(f"{args.out.name} is current.", file=sys.stderr)
        return 0
    args.out.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["receipts"], indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
