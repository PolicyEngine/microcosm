#!/usr/bin/env python
"""Build the rent quantiles by BRMA and category from the lists of rents (microcosm#1262).

The post-geography rent node reads each private renter's rent off its BRMA's
distribution of rents for homes of its size. This tool writes those
distributions, ``uk/brma_rent_quantiles.json``: for every Great Britain BRMA
and Local Housing Allowance category B to E (one to four or more bedrooms;
category A, a room in shared accommodation, is never used), 25 quantile
knots of the weekly rent and the mean they imply.

- England: VOA's shadow list of rents for April 2026 (lets from 1 October 2024
  to 30 September 2025).
- Wales: Rent Officers Wales' shadow list for 1 October 2023 to 30 September
  2024 (Table 2 of Welsh Government FOI ATISN 25142).
- Scotland: the Scottish Government's private sector rent statistics for the
  year to September 2025 fix each cell's quartiles and median (Chronicle
  facts); Rent Service Scotland's lettings for the years to September 2016 to
  2021 (FOI 202200303624) give the shape below the lower and above the upper
  quartile.

A cell with fewer than 50 lets takes its own median and its nation and
category's pooled shape; under 20, a log-additive BRMA plus category median
fitted on its nation's cells. No uprating is applied: the rent node scales
each region to its FRS level, so a factor common to a nation's list cancels
within that nation. It does not cancel for the 16 LSOAs whose BRMA crosses
the border: West Cheshire's 11 Welsh LSOAs read England's list under the
Wales scale, and Monmouthshire's 4 and Brecon and Radnor's 1 English LSOAs
read Wales's list, a year older, under English scales.

The three lists are Chronicle raw-only registrations, verified against their
manifests and the digests pinned here; the Scottish statistics are read from a
Chronicle consumer feed and checked against the source file digest pinned
here. Only published rent-officer evidence is read or written.

    uvx --with pandas --with xlrd --with openpyxl --with pyyaml \\
        python tools/build_uk_brma_rent_quantiles.py \\
        --chronicle-data <chronicle>/db/data \\
        --ledger-facts <feed artifact>/consumer_facts.jsonl --built-on YYYY-MM-DD

``--check`` rebuilds and compares with the committed resource.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
RUNTIME = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk_runtime"
DEFAULT_OUT = UK_PACKAGE / "brma_rent_quantiles.json"
LAYER = UK_PACKAGE / "brma_area_layer.json"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layer_module = _load_by_path("_uk_brma_area_layer", RUNTIME / "brma_area_layer.py")
rents_module = _load_by_path(
    "_uk_private_rent_localisation", RUNTIME / "private_rent_localisation.py"
)

REGISTRATIONS: Mapping[str, Mapping[str, str]] = {
    "england_list": {
        "directory": "voa/shadow_list_of_rents_april_2026",
        "package_id": "voa-shadow-list-of-rents-april-2026",
        "filename": "Shadow_List_of_Rents_April_2026.csv",
        "sha256": "4ed4e12b2307a339545271dea3f4d5805cf25cdcb6f757d2f9dfa3c3fe22edbc",
    },
    "wales_list": {
        "directory": "welshgov/shadow_list_of_rents_2023",
        "package_id": "welshgov-shadow-list-of-rents-2023",
        "filename": "atisn25142doc1.xls",
        "sha256": "8ab047dba4b8a6808cadbeb33f81e076a527930d68cb9cb4830be9d25c61c9c2",
    },
    "scotland_lettings": {
        "directory": "scotgov/brma_market_evidence_2016_2021",
        "package_id": "scotgov-brma-market-evidence-2016-2021",
        "filename": "scot_foi_brma_2016_2021.xlsx",
        "sha256": "dba86fa1cce920c8f646460bb13e1d99e16927d57fdf290e874f9631e804ed06",
    },
}
#: The Scottish Government statistics file behind the 2025 facts.
SCOTTISH_STATISTICS = {
    "record_set_prefix": "scotgov.private_rents.2025.",
    "source_sha256": "14f2368b47ec4e9fdc30fc8fa2985511ce37b87307c13a20c38587fa61aae88a",
    "source_file": "private_sector_rents_2010_2025.csv",
    "window": "1 October 2024 to 30 September 2025",
}
WALES_SHEET = "Table 2"
WALES_WINDOW = "1 October 2023 to 30 September 2024"
ENGLAND_WINDOW = "1 October 2024 to 30 September 2025"
SCOTLAND_SHEETS = ("2016", "2017", "2018", "2019", "2020", "2021")
SCOTTISH_BEDROOMS = {
    "1_bedroom": "B",
    "2_bedroom": "C",
    "3_bedroom": "D",
    "4_bedroom": "E",
}
SCOTTISH_MEASURES = ("lower_quartile", "median", "upper_quartile", "mean")
WEEKS_PER_MONTH = 52.0 / 12.0
#: Cell sizes below which a cell borrows its nation's shape, then its median.
OWN_KNOTS_MINIMUM = 50
OWN_MEDIAN_MINIMUM = 20
#: Largest gap allowed between a Scottish cell's implied and published means.
SCOTTISH_MEAN_TOLERANCE = 0.15

LICENCE = {
    "licence": "Open Government Licence v3.0",
    "attribution": (
        "Contains public sector information licensed under the Open Government "
        "Licence v3.0. Sources: Valuation Office Agency, shadow list of rents April "
        "2026; Rent Officers Wales shadow list of rents October 2023 to September "
        "2024 (Welsh Government FOI ATISN 25142); Rent Service Scotland market "
        "evidence 2016 to 2021 (Scottish Government FOI 202200303624); Scottish "
        "Government, private sector rent statistics, Scotland, 2010 to 2025."
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
    parser.add_argument("--chronicle-data", type=Path, required=True)
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="a Chronicle UK consumer_facts.jsonl holding the Scottish 2025 statistics",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--built-on", required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def verify_registrations(chronicle_data: Path) -> dict[str, dict[str, Any]]:
    import yaml

    out: dict[str, dict[str, Any]] = {}
    for name, pin in REGISTRATIONS.items():
        directory = chronicle_data / pin["directory"]
        manifest = yaml.safe_load((directory / "manifest.yaml").read_text("utf-8"))
        if manifest.get("package_id") != pin["package_id"]:
            raise SystemExit(f"{directory}: package id {manifest.get('package_id')!r}.")
        ((year, entry),) = manifest["files"].items()
        if entry["filename"] != pin["filename"] or entry["sha256"] != pin["sha256"]:
            raise SystemExit(f"{directory}: the manifest pins other bytes.")
        path = directory / entry["filename"]
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"{path} must be a regular local file.")
        if (
            path.stat().st_size != int(entry["size_bytes"])
            or _sha256(path) != pin["sha256"]
        ):
            raise SystemExit(f"{path} differs from its Chronicle registration.")
        out[name] = {
            "path": path,
            "chronicle_source_id": manifest.get("source_id"),
            "chronicle_package_id": manifest.get("package_id"),
            "registration_kind": manifest.get("registration_kind"),
            "year": str(year),
            "filename": entry["filename"],
            "sha256": entry["sha256"],
            "size_bytes": int(entry["size_bytes"]),
            "source_url": entry.get("source_url"),
        }
    return out


# ----------------------------------------------------------------------------
# Lists


def _category(value: str) -> str:
    match = re.fullmatch(r"(?:Cat\s*)?([A-E])", str(value).strip())
    if not match:
        raise SystemExit(f"unrecognised LHA category {value!r}")
    return match.group(1)


def england_lets(path: Path):
    import pandas as pd

    frame = pd.read_csv(path, encoding="cp1252", dtype=str)
    if set(frame.columns) != {"NET_RENT", "PERIOD", "BRMA", "LHA_TYPE"}:
        raise SystemExit(f"unexpected England list columns {list(frame.columns)}")
    if set(frame["PERIOD"]) != {"Week"}:
        raise SystemExit("the England list is not all weekly rents")
    return pd.DataFrame(
        {
            "brma": frame["BRMA"].map(layer_module.list_brma_key),
            "category": frame["LHA_TYPE"].map(_category),
            "rent": pd.to_numeric(
                frame["NET_RENT"].str.replace("[£,]", "", regex=True), errors="raise"
            ),
        }
    )


def wales_lets(path: Path):
    import pandas as pd

    raw = pd.read_excel(path, sheet_name=WALES_SHEET, header=None, dtype=str)
    title = str(raw.iloc[0, 0])
    if "1st October 2023 - 30th September 2024" not in title:
        raise SystemExit(f"unexpected Welsh list window: {title!r}")
    frame = raw.iloc[3:, :3]
    frame.columns = [str(value).strip() for value in raw.iloc[2, :3]]
    if list(frame.columns) != ["Category", "BRMA", "Weekly Rent"]:
        raise SystemExit(f"unexpected Welsh list columns {list(frame.columns)}")
    frame = frame.dropna(how="all")
    return pd.DataFrame(
        {
            "brma": frame["BRMA"].map(layer_module.list_brma_key),
            "category": frame["Category"].map(_category),
            "rent": pd.to_numeric(frame["Weekly Rent"], errors="raise"),
        }
    )


def scotland_lettings(path: Path):
    import pandas as pd

    frames = []
    for sheet in SCOTLAND_SHEETS:
        frame = pd.read_excel(path, sheet_name=sheet, dtype=str)
        frame = frame[frame["BRMA"].notna()]
        if set(frame["FREQUENCY"].str.strip()) != {"Weekly"}:
            raise SystemExit(f"Scottish lettings {sheet} are not all weekly")
        expected = f"{int(sheet) - 1}/{sheet[2:]}"
        if set(frame["YEAR ENDING SEPTEMBER"].str.strip()) != {expected}:
            raise SystemExit(f"Scottish lettings sheet {sheet} holds another year")
        frames.append(
            pd.DataFrame(
                {
                    "brma": frame["BRMA"].map(layer_module.list_brma_key),
                    "category": frame["LHA TYPE"].map(_category),
                    "rent": pd.to_numeric(frame["NET RENT"], errors="raise"),
                    "year": sheet,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _usable(lets):
    return lets[(lets["category"] != "A") & (lets["rent"] > 0)].copy()


# ----------------------------------------------------------------------------
# Cells


def _knots(values) -> list[float]:
    import numpy as np

    return [
        float(v)
        for v in np.quantile(np.asarray(values, dtype=float), rents_module.GRID)
    ]


def _shape(lets, by: Iterable[str]) -> list[float]:
    """Quantiles of rents relative to their own cell median, renormalised at 0.5."""

    import numpy as np

    medians = lets.groupby(list(by))["rent"].transform("median")
    relative = (lets["rent"] / medians).to_numpy(dtype=float)
    knots = np.quantile(relative, rents_module.GRID)
    middle = float(np.interp(0.5, rents_module.GRID, knots))
    return [float(v / middle) for v in knots]


def _median_model(medians, counts):
    """log median = BRMA effect + category effect, weighted by lets."""

    import numpy as np

    cells = sorted(medians)
    brmas = sorted({brma for brma, _ in cells})
    categories = sorted({category for _, category in cells})
    rows, targets, weights = [], [], []
    for brma, category in cells:
        row = [1.0 if brma == b else 0.0 for b in brmas] + [
            1.0 if category == c else 0.0 for c in categories[1:]
        ]
        rows.append(row)
        targets.append(np.log(medians[(brma, category)]))
        weights.append(np.sqrt(counts[(brma, category)]))
    design = np.asarray(rows) * np.asarray(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design, np.asarray(targets) * weights, rcond=None)
    brma_effect = dict(zip(brmas, solution[: len(brmas)], strict=True))
    category_effect = {categories[0]: 0.0}
    category_effect.update(zip(categories[1:], solution[len(brmas) :], strict=True))

    def predict(brma: str, category: str) -> float:
        if brma not in brma_effect:
            raise SystemExit(f"{brma} has no category with {OWN_MEDIAN_MINIMUM}+ lets")
        return float(np.exp(brma_effect[brma] + category_effect[category]))

    return predict


def list_cells(lets, nation: str, brmas: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Knots for every BRMA x category of a nation with a list of rents."""

    lets = _usable(lets)
    counts = lets.groupby(["brma", "category"]).size().to_dict()
    medians = lets.groupby(["brma", "category"])["rent"].median().to_dict()
    shapes = {
        category: _shape(lets[lets["category"] == category], ("brma", "category"))
        for category in rents_module.CATEGORIES
    }
    fitted = {
        cell: value
        for cell, value in medians.items()
        if counts[cell] >= OWN_MEDIAN_MINIMUM
    }
    predict = _median_model(fitted, counts)
    out: dict[str, dict[str, Any]] = {}
    for brma in sorted(brmas):
        cells: dict[str, Any] = {"nation": nation}
        for category in rents_module.CATEGORIES:
            n = int(counts.get((brma, category), 0))
            if n >= OWN_KNOTS_MINIMUM:
                rows = lets[(lets["brma"] == brma) & (lets["category"] == category)]
                knots, basis = _knots(rows["rent"]), "own_lets"
            elif n >= OWN_MEDIAN_MINIMUM:
                median = float(medians[(brma, category)])
                knots = [median * v for v in shapes[category]]
                basis = "own_median_nation_shape"
            else:
                median = predict(brma, category)
                knots = [median * v for v in shapes[category]]
                basis = "modelled_median_nation_shape"
            cells[category] = {"lets": n, "basis": basis, "quantiles": _round(knots)}
        out[brma] = cells
    return out


def _round(values: Iterable[float]) -> list[float]:
    return [round(float(v), 4) for v in values]


def scottish_statistics(
    path: Path, codes: Mapping[str, str]
) -> dict[tuple[str, str], dict]:
    """The 2025 quartiles, median and mean by BRMA and category, weekly."""

    names = {code: brma for brma, code in codes.items()}
    out: dict[tuple[str, str], dict[str, float]] = {}
    prefix = SCOTTISH_STATISTICS["record_set_prefix"]
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if prefix not in line:
                continue
            fact = json.loads(line)
            record_set = str((fact.get("layout") or {}).get("record_set_id", ""))
            if not record_set.startswith(prefix):
                continue
            measure, bedrooms = record_set[len(prefix) :].split(".", 1)
            if bedrooms not in SCOTTISH_BEDROOMS or measure not in SCOTTISH_MEASURES:
                continue
            code = str(fact["geography"]["id"])
            if code not in names:
                continue
            if fact["source"]["source_sha256"] != SCOTTISH_STATISTICS["source_sha256"]:
                raise SystemExit("a Scottish rent fact comes from another source file")
            key = (names[code], SCOTTISH_BEDROOMS[bedrooms])
            out.setdefault(key, {})[measure] = float(fact["value"]) / WEEKS_PER_MONTH
    expected = {(brma, c) for brma in codes for c in rents_module.CATEGORIES}
    missing = sorted(
        key for key in expected if set(out.get(key, {})) != set(SCOTTISH_MEASURES)
    )
    if missing:
        raise SystemExit(f"Scottish statistics missing for {missing[:5]}")
    return out


def scotland_cells(lettings, statistics, brmas) -> tuple[dict[str, dict], list[dict]]:
    """Published quartiles, with the lettings' shape outside them."""

    import numpy as np

    grid = np.asarray(rents_module.GRID)
    lettings = _usable(lettings)
    counts = lettings.groupby(["brma", "category"]).size().to_dict()
    national = {
        category: np.asarray(
            _shape(
                lettings[lettings["category"] == category], ("brma", "category", "year")
            )
        )
        for category in rents_module.CATEGORIES
    }
    q25, q50, q75 = (
        int(np.flatnonzero(np.isclose(grid, p))[0]) for p in (0.25, 0.5, 0.75)
    )
    out: dict[str, dict[str, Any]] = {}
    checks: list[dict] = []
    for brma in sorted(brmas):
        cells: dict[str, Any] = {"nation": "scotland"}
        for category in rents_module.CATEGORIES:
            n = int(counts.get((brma, category), 0))
            if n >= OWN_KNOTS_MINIMUM:
                rows = lettings[
                    (lettings["brma"] == brma) & (lettings["category"] == category)
                ]
                shape = np.asarray(_shape(rows, ("year",)))
                shape_basis = "own_lettings"
            else:
                shape = national[category]
                shape_basis = "scotland_category_lettings"
            stats = statistics[(brma, category)]
            low, mid, high = (
                stats["lower_quartile"],
                stats["median"],
                stats["upper_quartile"],
            )
            if not 0 < low <= mid <= high:
                raise SystemExit(f"{brma} {category}: quartiles out of order")
            knots = np.empty(len(grid))
            knots[: q25 + 1] = low * shape[: q25 + 1] / shape[q25]
            knots[q75:] = high * shape[q75:] / shape[q75]
            anchors_p = np.asarray([0.25, 0.5, 0.75])
            anchors_v = np.log([low, mid, high])
            middle = slice(q25, q75 + 1)
            knots[middle] = np.exp(np.interp(grid[middle], anchors_p, anchors_v))
            knots = np.maximum.accumulate(knots)
            implied = rents_module.knot_mean(grid, knots)
            gap = implied / stats["mean"] - 1.0
            checks.append(
                {"brma": brma, "category": category, "mean_gap": round(gap, 4)}
            )
            if abs(gap) > SCOTTISH_MEAN_TOLERANCE:
                raise SystemExit(
                    f"{brma} {category}: implied mean is {gap:+.1%} from the published mean"
                )
            cells[category] = {
                "lets": n,
                "basis": f"published_quartiles_{shape_basis}",
                "quantiles": _round(knots),
                "published_weekly": {
                    key: round(value, 4) for key, value in sorted(stats.items())
                },
            }
        out[brma] = cells
    return out, checks


# ----------------------------------------------------------------------------
# The resource


def build(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    files = verify_registrations(args.chronicle_data)
    layer = layer_module.load_brma_area_layer(LAYER.read_bytes())
    england_wales = set(layer.brma_by_area["ew_lsoa21"].values())
    scotland = set(layer.brma_by_area["scotland_oa22"].values())

    england = england_lets(files["england_list"]["path"])
    wales = wales_lets(files["wales_list"]["path"])
    lettings = scotland_lettings(files["scotland_lettings"]["path"])
    england_names, wales_names = set(england["brma"]), set(wales["brma"])
    if england_names & wales_names:
        raise SystemExit("a BRMA is on both the English and the Welsh list")
    if england_names | wales_names != england_wales:
        raise SystemExit(
            "the English and Welsh lists and the layer name different BRMAs: "
            f"{sorted((england_names | wales_names) ^ england_wales)}"
        )
    if set(lettings["brma"]) != scotland:
        raise SystemExit(
            f"the Scottish lettings and the layer differ: {sorted(set(lettings['brma']) ^ scotland)}"
        )

    brmas = {
        **list_cells(england, "england", england_names),
        **list_cells(wales, "wales", wales_names),
    }
    statistics = scottish_statistics(args.ledger_facts, layer.scotland_codes)
    scottish, checks = scotland_cells(lettings, statistics, scotland)
    brmas.update(scottish)
    grid = np.asarray(rents_module.GRID)
    for cells in brmas.values():
        for category in rents_module.CATEGORIES:
            knots = np.asarray(cells[category]["quantiles"])
            cells[category]["mean"] = round(rents_module.knot_mean(grid, knots), 4)

    def bases(nation: str) -> dict[str, int]:
        found: dict[str, int] = {}
        for cells in brmas.values():
            if cells["nation"] != nation:
                continue
            for category in rents_module.CATEGORIES:
                basis = cells[category]["basis"]
                found[basis] = found.get(basis, 0) + 1
        return dict(sorted(found.items()))

    gaps = [abs(check["mean_gap"]) for check in checks]
    return {
        "schema_version": rents_module.QUANTILES_SCHEMA_VERSION,
        "kind": rents_module.QUANTILES_KIND,
        "method": rents_module.QUANTILES_METHOD,
        "resource": args.out.name,
        "built_on": args.built_on,
        "description": (
            "Weekly rents by Broad Rental Market Area and LHA category B to E from the "
            "rent officers' lists of rents (England April 2026, Wales October 2023 to "
            "September 2024) and, for Scotland, the Scottish Government's 2025 quartiles "
            "with the shape of Rent Service Scotland's 2016 to 2021 lettings. Read by the "
            "post-geography private-rent node (microcosm#1262); rebuilt by "
            "tools/build_uk_brma_rent_quantiles.py."
        ),
        "licence": LICENCE,
        "unit": "gbp_per_week",
        "grid": list(rents_module.GRID),
        "categories": list(rents_module.CATEGORIES),
        "rules": {
            "category_a": "never used: a room in shared accommodation places no household",
            "tails": "flat beyond the end knots (winsorised at the 0.5th and 99.5th percentiles)",
            "own_knots_minimum_lets": OWN_KNOTS_MINIMUM,
            "own_median_minimum_lets": OWN_MEDIAN_MINIMUM,
            "thin_cells": (
                "20 to 49 lets: the cell's median times its nation and category's pooled "
                "shape (lets over their own cell's median); under 20: a log-additive "
                "BRMA plus category median fitted on the nation's cells with 20 or more "
                "lets, weighted by lets, times that shape"
            ),
            "scotland": (
                "published lower quartile, median and upper quartile, log-linear between "
                "them; below the lower and above the upper quartile the shape of the "
                "BRMA and category's 2016 to 2021 lettings over their year's median (the "
                "Scotland-wide category shape under 50 lettings)"
            ),
            "uprating": (
                "none: the rent node scales each FRS region's level, so a factor common "
                "to a nation's list cancels within the nation; Wales's list is a year "
                "older than England's, which the 16 LSOAs of the cross-border BRMAs "
                "(West Cheshire, Monmouthshire, Brecon and Radnor) read under the other "
                "nation's scale"
            ),
            "scotland_mean_tolerance": SCOTTISH_MEAN_TOLERANCE,
        },
        "sources": {
            **{
                name: {key: value for key, value in entry.items() if key != "path"}
                for name, entry in sorted(files.items())
            },
            "scotland_statistics": dict(SCOTTISH_STATISTICS),
        },
        "windows": {
            "england": ENGLAND_WINDOW,
            "wales": WALES_WINDOW,
            "scotland_quartiles": SCOTTISH_STATISTICS["window"],
            "scotland_shape": "years to September 2016 to 2021",
        },
        "receipts": {
            "england_lets": int(len(_usable(england))),
            "wales_lets": int(len(_usable(wales))),
            "scotland_lettings": int(len(_usable(lettings))),
            "brmas": {
                nation: sum(1 for cells in brmas.values() if cells["nation"] == nation)
                for nation in ("england", "wales", "scotland")
            },
            "cell_bases": {
                nation: bases(nation) for nation in ("england", "wales", "scotland")
            },
            "scotland_largest_mean_gap": round(max(gaps), 4),
        },
        "brmas": dict(sorted(brmas.items())),
    }


def _render(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build(args)
    if args.check:
        current = json.loads(args.out.read_text("utf-8")) if args.out.is_file() else {}
        current.pop("built_on", None)
        rebuilt = dict(payload)
        rebuilt.pop("built_on", None)
        if current != json.loads(_render(rebuilt)):
            raise SystemExit(f"{args.out} is stale.")
        print("current")
        return 0
    args.out.write_text(_render(payload), encoding="utf-8")
    print(json.dumps(payload["receipts"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
