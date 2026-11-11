#!/usr/bin/env python
"""Rebuild the Scottish BRMA x council-area overlap (microcosm#1090, #1092, #1262).

ONS's Price Index of Private Rents publishes Scottish rent levels for the 18
Broad Rental Market Areas, not for the 32 council areas. This tool measures
how the two geographies overlap, in households, from Scotland's Census 2022:

- each 2022 output area's BRMA comes from the committed area-to-BRMA layer
  (``uk/brma_area_layer.json``, built by ``tools/build_uk_brma_area_layer.py``
  with the method microcosm#1092 introduced), so the overlap and the layer are
  one copy of one method;
- its council area comes from the committed Scotland atomic support;
- ``private_rented_households`` and ``all_households`` are NRS's UV404 counts
  by output area (Census 2022, "Private rented: Total" and "All occupied
  households"), read from the pinned Chronicle feed. Output areas sum to each
  intersection exactly, so no ward split is assumed (microcosm#1092 split the
  census ward table, the finest level then registered; PolicyEngine/chronicle#326
  registered UV404 by output area).

UV404 prints ``-`` for a cell with no households, and Chronicle emits no fact
for it; such a cell counts as zero, and the resource records how many. The
census applies record swapping and cell key perturbation, so cells need not
add up exactly to published totals.

    uv run --no-sync python tools/build_uk_scotland_brma_la_overlap.py \\
        --ledger-facts <feed artifact>/consumer_facts.jsonl \\
        --supports <directory of the atomic support NPZs>

``--check`` compares the rebuild with the committed resource instead of
writing it. Only census aggregates and public geography are read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
RUNTIME = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk_runtime"
RESOURCE = UK_PACKAGE / "scotland_brma_la_overlap.json"
LAYER = UK_PACKAGE / "brma_area_layer.json"
SUPPORTS_PROVENANCE = UK_PACKAGE / "uk_atomic_area_supports.provenance.json"
SCOTLAND_SUPPORT = "uk_scotland_output_area_2022"
SCHEMA_VERSION = 2


def _load_layer_module():
    spec = importlib.util.spec_from_file_location(
        "_uk_brma_area_layer", RUNTIME / "brma_area_layer.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layer_module = _load_layer_module()

#: The NRS UV404 output-area table behind the two counts (PolicyEngine/chronicle#326).
UV404 = {
    "package_id": "nrs-census2022-uv404-tenure-output-area",
    "source_sha256": "d6abb0ee1a8f21e17160a517088b1ca62ddcb728a35bf5ecbf82c7f400b6378f",
    "record_sets": {
        "private_rented_households": "nrs.census2022.uv404.oa.private_rented_total",
        "all_households": "nrs.census2022.uv404.oa.all_occupied_households",
    },
}
#: PIPR's own names for the 18 BRMAs (ONS monthly price statistics, Table 1).
PIPR_NAMES = {
    "ABERDEEN_AND_SHIRE": "Aberdeen and Shire",
    "ARGYLL_AND_BUTE": "Argyll and Bute",
    "AYRSHIRES": "Ayrshires",
    "DUMFRIES_AND_GALLOWAY": "Dumfries and Galloway",
    "DUNDEE_AND_ANGUS": "Dundee and Angus",
    "EAST_DUNBARTONSHIRE": "East Dunbartonshire",
    "FIFE": "Fife",
    "FORTH_VALLEY": "Forth Valley",
    "GREATER_GLASGOW": "Greater Glasgow",
    "HIGHLAND_AND_ISLANDS": "Highland and Islands",
    "LOTHIAN": "Lothian",
    "NORTH_LANARKSHIRE": "North Lanarkshire",
    "PERTH_AND_KINROSS": "Perth and Kinross",
    "RENFREWSHIRE_AND_INVERCLYDE": "Renfrewshire/Inverclyde",
    "SCOTTISH_BORDERS": "Scottish Borders",
    "SOUTH_LANARKSHIRE": "South Lanarkshire",
    "WEST_DUNBARTONSHIRE": "West Dunbartonshire",
    "WEST_LOTHIAN": "West Lothian",
}
EXPECTED_COUNCILS = 32
LICENCE = {
    "licence": "Open Government Licence v3.0",
    "attribution": (
        "Contains public sector information licensed under the Open Government "
        "Licence v3.0. Sources: National Records of Scotland, Scotland's Census "
        "2022 UV404 household tenure by output area; the BRMA layer's sources "
        "(Scottish Government broad rental market areas, Rent Service Scotland's "
        "postcode lookup, NRS output-area population-weighted centroids)."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument("--supports", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESOURCE)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def output_area_councils(directory: Path) -> dict[str, str]:
    """Each Scottish output area's council, from the committed support."""

    import numpy as np

    declared = json.loads(SUPPORTS_PROVENANCE.read_text("utf-8"))["supports"][
        SCOTLAND_SUPPORT
    ]
    path = directory / declared["filename"]
    if _sha256(path) != declared["sha256"]:
        raise SystemExit(f"{path} differs from the committed support provenance.")
    with np.load(path, allow_pickle=False) as arrays:
        return dict(
            zip(
                arrays["area"].astype(str).tolist(),
                arrays["local_authority_code"].astype(str).tolist(),
                strict=True,
            )
        )


def uv404_counts(path: Path) -> dict[str, dict[str, int]]:
    """UV404 households by output area, per weight; absent cells are zero."""

    wanted = {record_set: weight for weight, record_set in UV404["record_sets"].items()}
    counts: dict[str, dict[str, int]] = {weight: {} for weight in UV404["record_sets"]}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if "nrs.census2022.uv404.oa." not in line:
                continue
            fact = json.loads(line)
            weight = wanted.get(str((fact.get("layout") or {}).get("record_set_id")))
            if weight is None:
                continue
            if fact["source"]["source_sha256"] != UV404["source_sha256"]:
                raise SystemExit("a UV404 fact comes from another source file")
            value = float(fact["value"])
            if value < 0 or value != int(value):
                raise SystemExit(f"UV404 {weight} is not a household count: {value!r}")
            area = str(fact["geography"]["id"])
            if area in counts[weight]:
                raise SystemExit(f"UV404 {weight} repeats output area {area}")
            counts[weight][area] = int(value)
    return counts


def overlap_cells(
    brma_by_area: Mapping[str, str],
    council_by_area: Mapping[str, str],
    counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Sum output-area counts into BRMA x council cells.

    Every output area lands in exactly one cell, so each weight's cells add up
    to its output-area total, by council and by BRMA. An output area with no
    private-rented count holds none. Cells are sorted by council, then BRMA.
    """

    cells: dict[tuple[str, str], dict[str, int]] = {}
    for area, brma in brma_by_area.items():
        cell = cells.setdefault(
            (council_by_area[area], brma),
            {"output_areas": 0, "private_rented_households": 0, "all_households": 0},
        )
        cell["output_areas"] += 1
        cell["all_households"] += int(counts["all_households"][area])
        cell["private_rented_households"] += int(
            counts["private_rented_households"].get(area, 0)
        )
    return [
        {"brma": brma, "local_authority": council, **cell}
        for (council, brma), cell in sorted(cells.items())
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    layer = layer_module.load_brma_area_layer(LAYER.read_bytes())
    brma_by_area = dict(layer.brma_by_area["scotland_oa22"])
    councils = output_area_councils(args.supports)
    if set(councils) != set(brma_by_area):
        raise SystemExit("the support's output areas and the layer's differ")
    counts = uv404_counts(args.ledger_facts)
    unknown = sorted(
        area for weight in counts for area in counts[weight] if area not in brma_by_area
    )
    if unknown:
        raise SystemExit(f"UV404 names output areas the layer lacks: {unknown[:5]}")
    if set(counts["all_households"]) != set(brma_by_area):
        raise SystemExit("UV404 all occupied households must cover every output area")

    rows = overlap_cells(brma_by_area, councils, counts)
    if len({row["local_authority"] for row in rows}) != EXPECTED_COUNCILS:
        raise SystemExit(f"expected {EXPECTED_COUNCILS} council areas")
    codes = dict(layer.scotland_codes)
    if set(codes) != set(PIPR_NAMES):
        raise SystemExit("the layer's Scottish BRMAs are not PIPR's 18")
    totals = {
        "cells": len(rows),
        "output_areas": len(brma_by_area),
        "output_areas_without_private_renters": len(brma_by_area)
        - len(counts["private_rented_households"]),
        "private_rented_households": sum(
            row["private_rented_households"] for row in rows
        ),
        "all_households": sum(row["all_households"] for row in rows),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "resource": RESOURCE.name,
        "description": (
            "Households in each Scottish Broad Rental Market Area x council area "
            "intersection, from Scotland's Census 2022 UV404 by output area and the "
            "committed area-to-BRMA layer. Rebuilt by "
            "tools/build_uk_scotland_brma_la_overlap.py."
        ),
        "licence": LICENCE,
        "weights": {
            "private_rented_households": (
                "Census 2022 UV404 'Private rented: Total' households (private "
                "landlord or letting agency, plus other private rented; rent-free "
                "households are not private renters) summed over each "
                "intersection's output areas; an output area UV404 prints '-' for "
                "counts zero."
            ),
            "all_households": (
                "Census 2022 UV404 all occupied households summed over each "
                "intersection's output areas."
            ),
        },
        "method": {
            "output_area_to_brma": (
                "The committed area-to-BRMA layer (uk/brma_area_layer.json, "
                "scotland_oa22 grain): population-weighted centroid within the "
                "Scottish Government polygon, corrected by Rent Service Scotland's "
                "postcode lookup, with the two documented Balloch overrides."
            ),
            "output_area_to_council": (
                "The committed Scotland atomic support (2019 council-area code set)."
            ),
            "layer_sha256": hashlib.sha256(LAYER.read_bytes()).hexdigest(),
            "support_sha256": json.loads(SUPPORTS_PROVENANCE.read_text("utf-8"))[
                "supports"
            ][SCOTLAND_SUPPORT]["sha256"],
            "uv404": dict(UV404),
            "history": (
                "microcosm#1092 split Scotland's Census 2022 ward tenure table across "
                "BRMAs by output-area households; UV404 by output area makes the "
                "private-rented cells exact sums (microcosm#1262)."
            ),
        },
        "brmas": {
            brma: {"pipr_area_code": codes[brma], "pipr_area_name": PIPR_NAMES[brma]}
            for brma in sorted(codes)
        },
        "totals": totals,
        "cells": rows,
    }


def _render(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build(args)
    text = _render(payload)
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != text:
            print(f"stale: {args.out} differs from the rebuild", file=sys.stderr)
            return 1
        print(f"current: {args.out}")
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}: {len(payload['cells'])} cells", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
