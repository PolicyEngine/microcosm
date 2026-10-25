"""Build the UK regional land-values JSON resource from the public CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT / "packages/microcosm-build/tests/fixtures/uk/regional_land_values.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "packages/microcosm-build/src/microcosm/build/uk/regional_land_values.json"
)
EXPECTED_SHA256 = "6dffc75c0bde2906b6a2482c5c6e8cf62284da996e60bcef7eebf427e0cf01f0"
EXPECTED_ROWS = 11


def build_resource(csv_path: Path) -> dict[str, object]:
    if _sha256(csv_path) != EXPECTED_SHA256:
        raise ValueError(f"{csv_path} does not match the reviewed CSV sha256 pin.")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"{csv_path} must contain {EXPECTED_ROWS} rows.")
    values = [
        {
            "region": row["region"],
            "avg_house_price": int(row["avg_house_price"]),
            "dwellings": int(row["dwellings"]),
        }
        for row in rows
    ]
    return {
        "version": 1,
        "country": "uk",
        "policy": (
            "Public regional land-value reference for deterministic post-WAS "
            "property uprating."
        ),
        "source": {
            "provenance": (
                "Incumbent public regional_land_values.csv re-homed as a JSON "
                "country-package resource for Microcosm. Values combine MHCLG "
                "dwellings and the UK House Price Index average price by region, "
                "averaged over the twelve months of calendar 2024 from the July "
                "2026 full file (HM Land Registry, registered raw-only in "
                "Chronicle as hmlr-uk-hpi-full-file-july-2026, sha256 "
                "654a541934ba7b393a741e23bc0c6fb668690f562feb8a7b4fff62d653fd6607)."
            ),
            "valuation_basis": (
                "calendar-2024 average prices: the stored main_residence_value sits "
                "at the dataset period's price level, so the engine's annual "
                "uprating from 2024 projects it once (microcosm#1240 fix 5; the "
                "December 2025 level this resource carried before was projected "
                "from 2024 a second time)"
            ),
            "citation_urls": [
                "https://www.gov.uk/government/collections/dwelling-stock-including-vacants",
                "https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-july-2026",
            ],
        },
        "values": values,
        "chronicle": [
            "2026-08-18: added for microcosm#681 E5 WAS wealth and regional property uprating.",
            "2026-10-13: regional prices moved from the December 2025 level to the calendar-2024 average (microcosm#1240 fix 5, valuation-date contract).",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build_resource(args.input_csv)
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=False,
        ensure_ascii=True,
    )
    rendered += "\n"
    if args.check:
        if args.output_json.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{args.output_json} is stale.")
        return 0
    args.output_json.write_text(rendered, encoding="utf-8")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
