"""Build the UK BRMA count-table resource from the staged VOA rent CSV."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "4956a20a5e5a04be5a00abd6b841d29b96da27ca539e522dd570537031120a02"
EXPECTED_ROWS = 1_257_744
DEFAULT_SOURCE = Path(".codex-work/incumbent/storage/lha_list_of_rents.csv.gz")
DEFAULT_OUTPUT = Path(
    "packages/microcosm-build/src/microcosm/build/uk/brma_rent_counts.json"
)


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
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        rents = pd.read_csv(handle)
    if len(rents) != EXPECTED_ROWS:
        raise ValueError(f"{source} has {len(rents)} rows, expected {EXPECTED_ROWS}.")
    required = {"year", "region", "lha_category", "brma"}
    missing = required - set(rents.columns)
    if missing:
        raise ValueError(f"{source} is missing column(s): {sorted(missing)}.")
    grouped = (
        rents.groupby(["region", "lha_category", "brma"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    cells: dict[str, dict[str, dict[str, int]]] = {}
    for row in grouped.sort_values(["region", "lha_category", "brma"]).itertuples(
        index=False
    ):
        cells.setdefault(str(row.region), {}).setdefault(str(row.lha_category), {})[
            str(row.brma)
        ] = int(row.count)
    years = sorted(int(year) for year in rents["year"].unique())
    brmas = sorted(str(value) for value in rents["brma"].unique())
    return {
        "version": 1,
        "country": "uk",
        "policy": (
            "BRMA assignment samples a committed VOA LHA list-of-rents count "
            "table by (region, LHA_category); count-weighted categorical "
            "sampling is distributionally equivalent to uniform row sampling "
            "from the pooled 2019 and 2020 rows."
        ),
        "source": {
            "artifact": source.name,
            "sha256": digest,
            "rows": int(len(rents)),
            "years": years,
            "provenance": (
                "Valuation Office Agency Local Housing Allowance list of rents "
                "CSV staged for the incumbent UK pipeline at rev ebf733c."
            ),
            "unique_brmas": len(brmas),
            "cell_count": int(
                grouped[["region", "lha_category"]].drop_duplicates().shape[0]
            ),
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
