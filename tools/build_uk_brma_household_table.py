"""Build the UK BRMA count-table resource from census private-rented households."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "fd40dae019e5eefb8c976873f69c66f9b74a0747505326e8098b0ad99cc2ae1f"
EXPECTED_ROWS = 936
# The households CSV is a staged input, not repo content. Point
# UK_BRMA_HOUSEHOLDS_CSV at wherever it is staged, or pass --source.
DEFAULT_SOURCE = Path(
    os.environ.get(
        "UK_BRMA_HOUSEHOLDS_CSV",
        ".codex-work/incumbent/storage/brma_private_rented_households.csv",
    )
)
DEFAULT_OUTPUT = Path(
    "packages/microcosm-build/src/microcosm/build/uk/brma_private_rented_households.json"
)
# Census bedroom band whose households weight each LHA category; a region
# published without bedrooms (Northern Ireland, band "all") weights every one.
LHA_CATEGORY_BEDROOMS = {"A": "1", "B": "1", "C": "2", "D": "3", "E": "4+"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_resource(args.source)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_resource(source: Path) -> dict[str, object]:
    digest = _sha256(source)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"{source} sha256 {digest} != expected {EXPECTED_SHA256}.")
    households = pd.read_csv(source, dtype={"bedrooms": str})
    if len(households) != EXPECTED_ROWS:
        raise ValueError(
            f"{source} has {len(households)} rows, expected {EXPECTED_ROWS}."
        )
    if list(households.columns) != ["region", "brma", "bedrooms", "households"]:
        raise ValueError(f"{source} columns are {list(households.columns)}.")
    cells: dict[str, dict[str, dict[str, int]]] = {}
    for region, rows in households.groupby("region", sort=True):
        bands = set(rows.bedrooms)
        for category, band in LHA_CATEGORY_BEDROOMS.items():
            used = "all" if bands == {"all"} else band
            chosen = rows[rows.bedrooms == used]
            if chosen.empty or (chosen.households <= 0).any():
                raise ValueError(f"{region} has no positive {used!r} households.")
            cells.setdefault(str(region), {})[category] = {
                str(row.brma): int(row.households)
                for row in chosen.sort_values("brma").itertuples(index=False)
            }
    return {
        "version": 2,
        "country": "uk",
        "policy": (
            "BRMA assignment samples private-rented households by region and "
            "BRMA from the censuses, using the bedroom band that matches each "
            "LHA category (A and B: one bedroom; C: two; D: three; E: four or "
            "more). Northern Ireland's census has no bedrooms question, so its "
            "counts weight every category."
        ),
        "source": {
            "artifact": source.name,
            "sha256": digest,
            "rows": int(len(households)),
            "provenance": (
                "Private-rented households (private landlord or letting agency "
                "plus other private rented) by region, BRMA and bedrooms: ONS "
                "Census 2021 (England and Wales; TS054 by LSOA with the LSOA's "
                "private-rented-or-rent-free bedroom mix), NRS Scotland's Census "
                "2022 (tenure by bedrooms by ward), NISRA Census 2021 "
                "(households by postcode district times Northern Ireland's "
                "private-rented share). Areas are mapped to BRMAs with the VOA "
                "(May 2020), Rent Officers Wales, Scottish Government and NIHE "
                "BRMA geographies."
            ),
            "census_years": {
                "england_wales": 2021,
                "scotland": 2022,
                "northern_ireland": 2021,
            },
            "unique_brmas": int(households.brma.nunique()),
            "cell_count": sum(len(categories) for categories in cells.values()),
        },
        "chronicle": {
            "status": "registration pending",
            "note": "Provenance registration is intentionally outside PR-CI scope.",
        },
        "cells": cells,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
