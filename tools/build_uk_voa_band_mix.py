"""Build the VOA council-tax band mix per England-and-Wales LSOA (microcosm#1240, fix 4).

Reads the VOA ``CTSOP1.1`` zip registered in Chronicle
(``voa-council-tax-stock-by-lad-2025``, properties by band and area at
31 March 2025), verifies it against the manifest pin, and writes
``uk/voa_council_tax_band_mix_lsoa.json``: for every 2021 LSOA the
cumulative share of properties in bands A to I. The post-geography band node
reads a clone's band off its assigned LSOA's cumulative shares at the same
uniform that placed its home value, so value and band agree by construction.

VOA prints ``-`` for a zero or negligible count and ``..`` for a band that
does not apply (band I outside Wales); both read as zero, and the resource
records how many cells each marker covered. Only aggregates of a published
table are written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
DEFAULT_OUT = UK_PACKAGE / "voa_council_tax_band_mix_lsoa.json"
VOA_CSV_MEMBER = "CTSOP1.1/CTSOP1_1_2025_03_31.csv"
BANDS = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
METHOD_VERSION = "uk-voa-band-mix-lsoa-v1"
KIND = "uk_voa_council_tax_band_mix_lsoa"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--voa-zip", type=Path, required=True)
    parser.add_argument("--voa-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--built-on",
        required=True,
        help=(
            "Build date recorded in the resource (YYYY-MM-DD); passed explicitly "
            "so a rerun reproduces the committed bytes."
        ),
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed resource is stale"
    )
    return parser.parse_args(argv)


def _read_table(voa_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(voa_zip) as archive:
        with archive.open(VOA_CSV_MEMBER) as stream:
            return pd.read_csv(
                io.TextIOWrapper(stream, encoding="utf-8-sig"), dtype=str
            )


def band_mix(
    table: pd.DataFrame,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Cumulative band shares per LSOA and the marker receipts."""
    rows = table[table["geography"] == "LSOA"].copy()
    markers = {"-": 0, "..": 0, "blank": 0}
    counts = np.zeros((len(rows), len(BANDS)), dtype=float)
    for column_index, band in enumerate(BANDS):
        raw = rows[f"band_{band.lower()}"].fillna("").astype(str).str.strip()
        for marker in ("-", ".."):
            markers[marker] += int((raw == marker).sum())
        markers["blank"] += int((raw == "").sum())
        numeric = pd.to_numeric(
            raw.str.replace(",", "", regex=False), errors="coerce"
        ).fillna(0.0)
        counts[:, column_index] = numeric.to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    all_properties = pd.to_numeric(
        rows["all_properties"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).to_numpy(dtype=float)
    mismatch = int((np.abs(totals - np.nan_to_num(all_properties)) > 0.5).sum())
    out: dict[str, dict[str, object]] = {}
    empty: list[str] = []
    for index, code in enumerate(rows["ecode"].astype(str)):
        total = totals[index]
        if total <= 0:
            empty.append(code)
            continue
        cumulative = np.cumsum(counts[index]) / total
        cumulative[-1] = 1.0
        out[code] = {
            "all_properties": int(round(total)),
            "cumulative": [float(round(value, 6)) for value in cumulative],
        }
    receipts = {
        "lsoa_rows": int(len(rows)),
        "lsoa_with_properties": len(out),
        "lsoa_without_properties": sorted(empty),
        "markers_read_as_zero": markers,
        "rows_where_band_sum_differs_from_all_properties": mismatch,
        "england_lsoa": int(sum(code.startswith("E") for code in out)),
        "wales_lsoa": int(sum(code.startswith("W") for code in out)),
        "band_i_outside_wales_nonzero": int(
            sum(
                1
                for code, entry in out.items()
                if code.startswith("E") and entry["cumulative"][-2] < 1.0
            )
        ),
    }
    return out, receipts


#: The source's licence and the attribution it asks for on derived data.
LICENCE = {
    "source": "Valuation Office Agency, Council Tax: stock of properties 2025 (CTSOP1.1)",
    "licence": "Open Government Licence v3.0",
    "attribution": (
        "Contains public sector information licensed under the Open Government "
        "Licence v3.0. Source: Valuation Office Agency, Council Tax: stock of "
        "properties, 31 March 2025."
    ),
}


def build(args: argparse.Namespace) -> dict[str, object]:
    manifest = yaml.safe_load(args.voa_manifest.read_text(encoding="utf-8"))
    ((year, entry),) = manifest["files"].items()
    digest = _sha256(args.voa_zip)
    if digest != entry["sha256"] or args.voa_zip.stat().st_size != int(
        entry["size_bytes"]
    ):
        raise ValueError(
            "The VOA CTSOP1.1 zip differs from its Chronicle registration."
        )
    table = _read_table(args.voa_zip)
    mix, receipts = band_mix(table)
    return {
        "schema_version": 1,
        "kind": KIND,
        "resource": args.out.name,
        "built_on": args.built_on,
        "method": METHOD_VERSION,
        "description": (
            "Cumulative share of domestic properties by council-tax band (A to I) for every "
            "England and Wales 2021 LSOA, from VOA CTSOP1.1 at 31 March 2025. Read by the "
            "post-geography band node (microcosm#1240, fix 4)."
        ),
        "source": {
            "chronicle_source_id": manifest.get("source_id"),
            "chronicle_package_id": manifest.get("package_id"),
            "chronicle_year": int(year),
            "filename": entry["filename"],
            "sha256": entry["sha256"],
            "size_bytes": int(entry["size_bytes"]),
            "source_url": entry.get("source_url"),
            "table": manifest.get("table"),
            "as_at": "2025-03-31",
            "csv_member": VOA_CSV_MEMBER,
            "lsoa_vintage": "2021_census",
        },
        "licence": LICENCE,
        "bands": list(BANDS),
        "receipts": receipts,
        "lsoa": mix,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build(args)
    rendered = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        current_payload = json.loads(current) if current else {}
        current_payload.pop("built_on", None)
        payload.pop("built_on", None)
        if current_payload != payload:
            raise SystemExit(f"{args.out} is stale.")
        return 0
    args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps(payload["receipts"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
