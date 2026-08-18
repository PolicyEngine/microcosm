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
EXPECTED_SHA256 = "088b777d73e71c07890ac9e1add9b683aa962da69ca79c25dd0b54d7e325cfb6"
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
                "dwellings and ONS UK House Price Index average prices for "
                "December 2025."
            ),
            "citation_urls": [
                "https://www.gov.uk/government/collections/dwelling-stock-including-vacants",
                "https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/housepriceindex/december2025",
            ],
        },
        "values": values,
        "chronicle": [
            "2026-08-18: added for microcosm#681 E5 WAS wealth and regional property uprating."
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
