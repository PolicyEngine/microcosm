#!/usr/bin/env python
"""Build the area-to-BRMA layer from Chronicle's raw registrations (microcosm#1262).

Every 2021 LSOA of England and Wales and every 2022 output area of Scotland
is placed in one Broad Rental Market Area, so the post-geography node
``uk.full.geography.brma`` can read a placed household's BRMA off its
assigned area. Writes ``uk/brma_area_layer.json`` and its provenance sidecar.

- England and Wales: each LSOA's population-weighted centroid in the VOA
  (May 2020) or Rent Officers Wales (May 2012) BRMA polygons; each output
  area takes its LSOA's BRMA. This is uk-data#516's method
  (``tools/brma_households/build.py``), with its six documented overrides.
- Scotland: each output area's population-weighted centroid in the Scottish
  Government's BRMA polygons (2009, revised 2015), corrected by Rent Service
  Scotland's postcode lookup (FOI 202300368850) where it names one BRMA, with
  the two Balloch output areas it leaves ambiguous overridden. This is
  microcosm#1092's method.

The six publisher files are Chronicle raw-only registrations, verified
against their manifests and against the digests pinned here; the atomic
supports (for the area rosters, regions and councils) are verified against
``uk/uk_atomic_area_supports.provenance.json``. The geography step needs
geopandas, which is not a Microcosm dependency:

    uvx --with geopandas --with pandas --with shapely --with pyogrio \\
        --with openpyxl --with pyyaml python tools/build_uk_brma_area_layer.py \\
        --chronicle-data <chronicle>/db/data \\
        --supports <directory of the atomic support NPZs> --built-on YYYY-MM-DD

``--check`` rebuilds and compares with the committed resources instead of
writing them. Only public geography is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
RUNTIME = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk_runtime"
DEFAULT_OUT = UK_PACKAGE / "brma_area_layer.json"
DEFAULT_PROVENANCE = UK_PACKAGE / "brma_area_layer.provenance.json"
SUPPORTS_PROVENANCE = UK_PACKAGE / "uk_atomic_area_supports.provenance.json"
COUNT_TABLE = UK_PACKAGE / "brma_private_rented_households.json"


def _load_layer_module():
    """``uk_runtime/brma_area_layer.py`` by path: the tool runs outside the workspace."""

    spec = importlib.util.spec_from_file_location(
        "_uk_brma_area_layer", RUNTIME / "brma_area_layer.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layer_module = _load_layer_module()

#: The Chronicle raw-only registrations read, with the bytes pinned here.
REGISTRATIONS: Mapping[str, Mapping[str, str]] = {
    "england_polygons": {
        "directory": "voa/brma_boundaries_2020",
        "package_id": "voa-brma-boundaries-2020",
        "filename": "GML.zip",
        "sha256": "66515b0a1403e4557e6501d2d5ada5e7e17b29deef2da9165ba515a66907f6c6",
    },
    "wales_polygons": {
        "directory": "welshgov/brma_boundaries_2012",
        "package_id": "welshgov-brma-boundaries-2012",
        "filename": "wales_brma_boundaries_2012.zip",
        "sha256": "49de861258ba945eb659648d2a040aef1e9089ac80374f0bf4e85fcb062edd6c",
    },
    "scotland_polygons": {
        "directory": "scotgov/brma_boundaries_2009",
        "package_id": "scotgov-brma-boundaries-2009",
        "filename": "SG_BroadRentalMarketAreas_2009.zip",
        "sha256": "4d121236a35ce896edeb663a7c2c390053c0deebe0cc149c84696bbfd4bddec0",
    },
    "scotland_postcode_lookup": {
        "directory": "scotgov/brma_postcode_lookup_2023",
        "package_id": "scotgov-brma-postcode-lookup-2023",
        "filename": "scot_foi_postcodes_brma_2023.xlsx",
        "sha256": "50366fea2b87bd6b697e7aa8736fbc870838b5e327458213294f31c3de7ede5e",
    },
    "ew_lsoa_centroids": {
        "directory": "ons/lsoa2021_population_weighted_centroids_v4",
        "package_id": "ons-lsoa2021-population-weighted-centroids-v4",
        "filename": "LSOA_PopCentroids_EW_2021_V4.geojson",
        "sha256": "e0872028485e92a8c3bd73d2f995c84663a455ddfddde278d83e24b00fc90a5d",
    },
    "scotland_oa_centroids": {
        "directory": "nrs/oa2022_population_weighted_centroids",
        "package_id": "nrs-oa2022-population-weighted-centroids",
        "filename": "output-area-2022-pwc.zip",
        "sha256": "509d7d937c8cc1477259dcd8e4e0e61240f5cc782721a79e9c279bd589e0ec0c",
    },
}
SUPPORT_FILES = {
    "uk_ew_output_area_2021": "uk_ew_output_area_2021_support.npz",
    "uk_scotland_output_area_2022": "uk_scotland_output_area_2022_support.npz",
}

# Overrides established on 2026-10-02 for uk-data#516 (copied
# verbatim from its tools/brma_households/build.py); live reply pages are
# evidence, not inputs.
LSOA_OVERRIDES = {
    "E01014018": "BRECON_AND_RADNOR",  # Official VOA postcode lookup HR3 5TA, 2026-10-02.
    "E01022272": "MONMOUTHSHIRE",  # Official VOA postcode lookup NP16 7AG, 2026-10-02.
    "E01022273": "MONMOUTHSHIRE",  # Official VOA postcode lookup NP16 7RB, 2026-10-02.
    "E01022274": "MONMOUTHSHIRE",  # Official VOA postcode lookup NP16 7QB, 2026-10-02.
    "E01022275": "MONMOUTHSHIRE",  # Official VOA postcode lookup NP16 7EN, 2026-10-02.
    "W01002024": "CARDIFF",  # Official VOA lookup CF11 0AS (and 12 other postcodes), 2026-10-02.
}
# Two Balloch output areas whose master postcodes Rent Service Scotland's
# lookup lists under two BRMAs, neither the polygon's answer. LHA Direct's
# search for West Dunbartonshire council returned the Argyll and Bute and West
# Dunbartonshire BRMAs on 2026-10-02, and each postcode's candidates meet that
# pair in West Dunbartonshire only (microcosm#1092). Evidence, not an input.
OUTPUT_AREA_OVERRIDES = {
    "S00179382": "WEST_DUNBARTONSHIRE",  # G83 8NQ
    "S00179383": "WEST_DUNBARTONSHIRE",  # G83 8SD
}
OVERRIDE_COUNCIL = "S12000039"
#: The Price Index of Private Rents' codes for the 18 Scottish BRMAs, which the
#: Scottish Government's polygons carry in their ``BRMA`` column.
SCOTTISH_BRMA_CODES = {
    "ABERDEEN_AND_SHIRE": "S33000001",
    "ARGYLL_AND_BUTE": "S33000002",
    "AYRSHIRES": "S33000003",
    "DUMFRIES_AND_GALLOWAY": "S33000004",
    "DUNDEE_AND_ANGUS": "S33000005",
    "EAST_DUNBARTONSHIRE": "S33000006",
    "FIFE": "S33000007",
    "FORTH_VALLEY": "S33000008",
    "GREATER_GLASGOW": "S33000009",
    "HIGHLAND_AND_ISLANDS": "S33000010",
    "LOTHIAN": "S33000011",
    "NORTH_LANARKSHIRE": "S33000012",
    "PERTH_AND_KINROSS": "S33000013",
    "RENFREWSHIRE_AND_INVERCLYDE": "S33000014",
    "SCOTTISH_BORDERS": "S33000015",
    "SOUTH_LANARKSHIRE": "S33000016",
    "WEST_DUNBARTONSHIRE": "S33000017",
    "WEST_LOTHIAN": "S33000018",
}
EXPECTED_ENGLISH_BRMAS = 152
EXPECTED_GB_BRMAS = 192

#: The sources' licences and the attribution they ask for on derived data.
LICENCE = {
    "licence": "Open Government Licence v3.0",
    "attribution": (
        "Contains public sector information licensed under the Open Government "
        "Licence v3.0. Sources: Valuation Office Agency, broad rental market area "
        "boundaries applicable May 2020; Rent Officers Wales, broad rental market "
        "area boundaries applicable May 2012; Scottish Government, broad rental "
        "market areas (2009, revised 2015) and Rent Service Scotland postcode to "
        "BRMA lookup (FOI 202300368850); Office for National Statistics, LSOA "
        "(December 2021) population-weighted centroids; National Records of "
        "Scotland, 2022 output area population-weighted centroids. Contains OS "
        "data (c) Crown copyright and database right 2021 and 2024; contains NRS "
        "data (c) Crown copyright and database right 2024."
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
    parser.add_argument(
        "--chronicle-data",
        type=Path,
        required=True,
        help="the db/data directory of a PolicyEngine/chronicle checkout",
    )
    parser.add_argument(
        "--supports",
        type=Path,
        required=True,
        help="directory holding the England-and-Wales and Scotland support NPZs",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--provenance-out", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--built-on",
        required=True,
        help=(
            "Build date recorded in the resources (YYYY-MM-DD); passed explicitly "
            "so a rerun reproduces the committed bytes."
        ),
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed resources are stale"
    )
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------
# Pins


def verify_registrations(chronicle_data: Path) -> dict[str, dict[str, Any]]:
    """Each registration's file, checked against its manifest and the pin here."""

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


def load_supports(directory: Path) -> dict[str, Any]:
    """The England-and-Wales and Scotland supports, checked against their provenance."""

    import numpy as np

    provenance = json.loads(SUPPORTS_PROVENANCE.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for system, filename in SUPPORT_FILES.items():
        declared = provenance["supports"][system]
        path = directory / filename
        if _sha256(path) != declared["sha256"]:
            raise SystemExit(f"{path} differs from the committed support provenance.")
        with np.load(path, allow_pickle=False) as arrays:
            out[system] = {
                key: arrays[key]
                for key in (
                    "area",
                    "lsoa_code",
                    "frs_region",
                    "local_authority_code",
                    "households",
                )
            }
        out[system]["sha256"] = declared["sha256"]
        out[system]["filename"] = filename
    return out


# ----------------------------------------------------------------------------
# England and Wales


def english_polygons(path: Path):
    """The VOA layer's actual GML geometry (its XSD wrongly declares nine MultiPolygons)."""

    import geopandas as gpd
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon

    ns = {"gml": "http://www.opengis.net/gml", "ogr": "http://ogr.maptools.org/"}

    def ring(node):
        return [tuple(map(float, xy.split(","))) for xy in node.text.split()]

    def polygon(node):
        shell = ring(
            node.find("gml:outerBoundaryIs/gml:LinearRing/gml:coordinates", ns)
        )
        holes = [
            ring(inner)
            for inner in node.findall(
                "gml:innerBoundaryIs/gml:LinearRing/gml:coordinates", ns
            )
        ]
        return Polygon(shell, holes)

    def parts(geometry):
        if geometry.geom_type == "Polygon":
            return [geometry]
        if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
            return [part for child in geometry.geoms for part in parts(child)]
        return []

    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("English BRMA(LHA) Layer.gml"))
    records = []
    for feature in root.findall("gml:featureMember/ogr:English_BRMA_LHA__Layer", ns):
        node = feature.find("ogr:geometryProperty", ns)[0]
        if {n.attrib["srsName"] for n in node.iter() if "srsName" in n.attrib} != {
            "EPSG:27700"
        }:
            raise SystemExit("the VOA layer is not in British National Grid")
        if node.tag.endswith("}Polygon"):
            geometry = polygon(node)
        elif node.tag.endswith("}MultiPolygon"):
            geometry = MultiPolygon(
                [polygon(n) for n in node.findall("gml:polygonMember/gml:Polygon", ns)]
            )
        else:
            raise SystemExit(f"unexpected VOA geometry {node.tag}")
        repaired = parts(make_valid(geometry))
        if not repaired:
            raise SystemExit("a VOA polygon is empty after repair")
        geometry = repaired[0] if len(repaired) == 1 else MultiPolygon(repaired)
        if not geometry.is_valid or geometry.is_empty:
            raise SystemExit("a VOA polygon is invalid after repair")
        name = layer_module.english_welsh_brma_key(
            feature.findtext("ogr:Name", namespaces=ns)
        )
        records.append({"brma": name, "geometry": geometry})
    frame = gpd.GeoDataFrame(records, crs=27700)
    if len(frame) != EXPECTED_ENGLISH_BRMAS or frame["brma"].nunique() != len(frame):
        raise SystemExit(f"expected {EXPECTED_ENGLISH_BRMAS} distinct English BRMAs")
    return frame


def welsh_polygons(path: Path):
    import geopandas as gpd

    frame = gpd.read_file(
        f"zip://{path}!Broad Rental Market Areas/wales_brma.shp"
    ).to_crs(27700)
    frame["geometry"] = frame.geometry.make_valid()
    frame["brma"] = frame["brma_name"].map(layer_module.english_welsh_brma_key)
    if frame["brma"].duplicated().any():
        raise SystemExit("a Welsh BRMA name repeats")
    return frame[["brma", "geometry"]]


def lsoa_regions(support: Mapping[str, Any]) -> dict[str, str]:
    """Each 2021 LSOA's FRS region, from its output areas in the support."""

    import pandas as pd

    frame = pd.DataFrame(
        {
            "lsoa": support["lsoa_code"].astype(str),
            "region": support["frs_region"].astype(str),
        }
    )
    regions = frame.groupby("lsoa")["region"].agg(set)
    split = regions[regions.map(len) > 1]
    if len(split):
        raise SystemExit(f"{len(split)} LSOAs span two FRS regions")
    return {lsoa: next(iter(found)) for lsoa, found in regions.items()}


def england_wales_assignments(files: Mapping[str, Mapping[str, Any]], support) -> tuple:
    """LSOA -> BRMA for England and Wales, with the receipts of the join."""

    import geopandas as gpd

    regions = lsoa_regions(support)
    points = gpd.read_file(files["ew_lsoa_centroids"]["path"]).to_crs(27700)
    points = points[["LSOA21CD", "geometry"]].rename(columns={"LSOA21CD": "lsoa"})
    if not points["lsoa"].is_unique or set(points["lsoa"]) != set(regions):
        raise SystemExit("the LSOA centroids and the support's LSOAs differ")
    assignments: dict[str, str] = {}
    receipts: dict[str, int] = {}
    for nation, prefix, polygons in (
        ("england", "E", english_polygons(files["england_polygons"]["path"])),
        ("wales", "W", welsh_polygons(files["wales_polygons"]["path"])),
    ):
        nation_points = points[points["lsoa"].str.startswith(prefix)]
        joined = gpd.sjoin(nation_points, polygons, how="left", predicate="intersects")
        if not joined["lsoa"].is_unique:
            raise SystemExit(f"a {nation} LSOA centroid sits in two BRMA polygons")
        unmatched = set(joined.loc[joined["brma"].isna(), "lsoa"])
        expected = {code for code in LSOA_OVERRIDES if code.startswith(prefix)}
        if unmatched != expected:
            raise SystemExit(f"the {nation} override set changed: {sorted(unmatched)}")
        for lsoa, brma in zip(joined["lsoa"], joined["brma"], strict=True):
            assignments[str(lsoa)] = (
                LSOA_OVERRIDES[lsoa] if lsoa in expected else str(brma)
            )
        receipts[f"{nation}_lsoa"] = int(len(nation_points))
        receipts[f"{nation}_lsoa_outside_polygons"] = len(unmatched)
        receipts[f"{nation}_polygon_brmas"] = int(polygons["brma"].nunique())
    return assignments, regions, receipts


# ----------------------------------------------------------------------------
# Scotland


def _postcode_key(value: str) -> str:
    # NRS suffixes a split postcode's records with A or B; the lookup does not.
    key = re.sub(r"\s+", "", str(value).upper())
    match = re.fullmatch(r"([A-Z]{1,2}[0-9][0-9A-Z]?[0-9][A-Z]{2})[A-Z]?", key)
    if not match:
        raise SystemExit(f"unrecognised postcode {value!r}")
    return match.group(1)


def scotland_assignments(files: Mapping[str, Mapping[str, Any]], support) -> tuple:
    """Output area -> BRMA for Scotland, with the S33 codes and the receipts."""

    import geopandas as gpd
    import pandas as pd

    brmas = gpd.read_file(f"zip://{files['scotland_polygons']['path']}")
    if len(brmas) != len(SCOTTISH_BRMA_CODES) or not brmas.geometry.is_valid.all():
        raise SystemExit("expected 18 valid Scottish BRMA polygons")
    brmas["brma"] = brmas["BRMAName"].map(layer_module.scottish_brma_key)
    codes = dict(zip(brmas["brma"], brmas["BRMA"].astype(str), strict=True))
    if codes != SCOTTISH_BRMA_CODES:
        raise SystemExit(f"the Scottish BRMA names or codes changed: {codes}")

    path = files["scotland_oa_centroids"]["path"]
    oas = gpd.read_file(
        f"zip://{path}!OutputArea2022_PWC/OutputArea2022_PWC.shp"
    ).rename(columns={"code": "output_area"})
    oas = oas.to_crs(brmas.crs)
    councils = dict(
        zip(
            support["area"].astype(str),
            support["local_authority_code"].astype(str),
            strict=True,
        )
    )
    if not oas["output_area"].is_unique or set(oas["output_area"]) != set(councils):
        raise SystemExit("the output-area centroids and the support's areas differ")

    polygons = brmas[["brma", "geometry"]]
    matches = gpd.sjoin(oas, polygons, how="left", predicate="within")
    outside = matches.loc[matches["brma"].isna(), "output_area"]
    boundary = gpd.sjoin(
        oas[oas["output_area"].isin(outside)],
        polygons,
        how="left",
        predicate="intersects",
    )
    matches = pd.concat(
        [matches[~matches["output_area"].isin(outside)], boundary], ignore_index=True
    )
    hits = matches.groupby("output_area")["brma"].count()
    single = matches[matches["output_area"].map(hits).eq(1)]
    oas = oas.drop(columns="geometry").merge(
        single[["output_area", "brma"]],
        on="output_area",
        how="left",
        validate="one_to_one",
    )
    polygon_brma = oas["brma"].copy()

    lookup = pd.read_excel(
        files["scotland_postcode_lookup"]["path"], usecols=[0, 1], dtype=str
    )
    lookup.columns = ["brma", "postcode"]
    lookup["postcode"] = lookup["postcode"].map(_postcode_key)
    lookup["brma"] = lookup["brma"].map(layer_module.scottish_brma_key)
    if not set(lookup["brma"]) <= set(SCOTTISH_BRMA_CODES):
        raise SystemExit("the postcode lookup names an unknown BRMA")
    candidates = lookup.groupby("postcode")["brma"].agg(set)
    oas["candidates"] = oas["masterpc"].map(_postcode_key).map(candidates)
    unique = oas["candidates"].map(
        lambda found: (
            next(iter(found)) if isinstance(found, set) and len(found) == 1 else None
        )
    )
    filled = oas["brma"].isna() & unique.notna()
    oas["brma"] = oas["brma"].fillna(unique)
    if oas["brma"].isna().any():
        raise SystemExit("an output area outside the polygons has no single BRMA")
    oas["council"] = oas["output_area"].map(councils)

    conflict = pd.Series(
        [
            isinstance(found, set) and pd.notna(polygon) and polygon not in found
            for polygon, found in zip(polygon_brma, oas["candidates"], strict=True)
        ],
        index=oas.index,
    )
    corrected = conflict & unique.notna()
    oas.loc[corrected, "brma"] = unique[corrected]
    unresolved = conflict & unique.isna()
    if set(oas.loc[unresolved, "output_area"]) != set(OUTPUT_AREA_OVERRIDES):
        raise SystemExit("the set of ambiguous Scottish output areas changed")
    if not oas.loc[unresolved, "council"].eq(OVERRIDE_COUNCIL).all():
        raise SystemExit("an overridden output area left West Dunbartonshire")
    oas.loc[unresolved, "brma"] = oas.loc[unresolved, "output_area"].map(
        OUTPUT_AREA_OVERRIDES
    )
    disagree = [
        area
        for area, brma, found in zip(
            oas["output_area"], oas["brma"], oas["candidates"], strict=True
        )
        if isinstance(found, set) and brma not in found
    ]
    if disagree:
        raise SystemExit(f"assignments contradict the postcode lookup: {disagree[:5]}")
    assignments = dict(
        zip(oas["output_area"].astype(str), oas["brma"].astype(str), strict=True)
    )
    receipts = {
        "scotland_output_areas": int(len(oas)),
        "scotland_outside_polygons": int(polygon_brma.isna().sum()),
        "scotland_filled_from_postcode_lookup": int(filled.sum()),
        "scotland_corrected_by_postcode_lookup": int(corrected.sum()),
        "scotland_overridden": int(unresolved.sum()),
    }
    pairs = (
        oas.groupby(["brma", "council"], as_index=False)
        .agg(output_areas=("output_area", "size"), households=("HHcount", "sum"))
        .sort_values(["brma", "council"])
    )
    return assignments, codes, receipts, pairs


# ----------------------------------------------------------------------------
# The resources


def _count_table_brmas() -> dict[str, set[str]]:
    """BRMAs with private-rented households in each region of the count table."""

    table = json.loads(COUNT_TABLE.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for region, categories in table["cells"].items():
        found = out.setdefault(region, set())
        for cells in categories.values():
            found.update(brma for brma, count in cells.items() if float(count) > 0)
    return out


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    files = verify_registrations(args.chronicle_data)
    supports = load_supports(args.supports)
    ew, regions, ew_receipts = england_wales_assignments(
        files, supports["uk_ew_output_area_2021"]
    )
    scotland, codes, scotland_receipts, pairs = scotland_assignments(
        files, supports["uk_scotland_output_area_2022"]
    )

    known = _count_table_brmas()
    known_names = set().union(*known.values())
    gb_names = set(ew.values()) | set(scotland.values())
    unknown = sorted(gb_names - known_names)
    if unknown:
        raise SystemExit(f"BRMAs the engine's count table does not know: {unknown}")
    if len(gb_names) != EXPECTED_GB_BRMAS:
        raise SystemExit(
            f"expected {EXPECTED_GB_BRMAS} Great Britain BRMAs, got {len(gb_names)}"
        )

    layout: dict[str, dict[str, dict[str, list[str]]]] = {
        "ew_lsoa21": {},
        "scotland_oa22": {},
    }
    for lsoa, brma in ew.items():
        layout["ew_lsoa21"].setdefault(regions[lsoa], {}).setdefault(brma, []).append(
            lsoa
        )
    for area, brma in scotland.items():
        layout["scotland_oa22"].setdefault("SCOTLAND", {}).setdefault(brma, []).append(
            area
        )
    for grain in layout.values():
        for by_brma in grain.values():
            for codes_ in by_brma.values():
                codes_.sort()

    agreement: dict[str, Any] = {}
    for region in sorted(set(known) - {"NORTHERN_IRELAND"}):
        grain = "scotland_oa22" if region == "SCOTLAND" else "ew_lsoa21"
        in_layer = set(layout[grain].get(region, {}))
        count_only = sorted(known[region] - in_layer)
        if count_only:
            raise SystemExit(
                f"{region}: the count table has private renters in BRMAs the layer "
                f"places no area of the region in: {count_only}"
            )
        agreement[region] = {"layer_only": sorted(in_layer - known[region])}

    resource = {
        "schema_version": layer_module.SCHEMA_VERSION,
        "kind": layer_module.KIND,
        "method": layer_module.METHOD_VERSION,
        "resource": args.out.name,
        "built_on": args.built_on,
        "description": (
            "The Broad Rental Market Area of every 2021 LSOA in England and Wales and "
            "every 2022 output area in Scotland, by population-weighted centroid in the "
            "published BRMA polygons (Scotland corrected by Rent Service Scotland's "
            "postcode lookup). Read by the post-geography node uk.full.geography.brma "
            "(microcosm#1262); rebuilt by tools/build_uk_brma_area_layer.py."
        ),
        "licence": LICENCE,
        "grains": {
            "ew_lsoa21": {
                "area_system": "uk_ew_output_area_2021",
                "key": "lsoa_code",
                "areas": len(ew),
                "roster_sha256": layer_module.roster_sha256(ew),
            },
            "scotland_oa22": {
                "area_system": "uk_scotland_output_area_2022",
                "key": "oa_code",
                "areas": len(scotland),
                "roster_sha256": layer_module.roster_sha256(scotland),
            },
        },
        "overrides": {
            "ew_lsoa21": dict(sorted(LSOA_OVERRIDES.items())),
            "scotland_oa22": dict(sorted(OUTPUT_AREA_OVERRIDES.items())),
        },
        "scotland_brma_codes": dict(sorted(codes.items())),
        "brmas": layout,
    }
    rendered = _render(resource)
    provenance = {
        "schema_version": 1,
        "kind": "uk_brma_area_layer_provenance",
        "resource": args.out.name,
        "resource_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "built_on": args.built_on,
        "tool": "tools/build_uk_brma_area_layer.py",
        "issues": [
            "microcosm#1262",
            "microcosm#1092",
            "microcosm#1090",
            "uk-data#516",
        ],
        "method": {
            "england_wales": (
                "Each 2021 LSOA's population-weighted centroid joined (intersects) to the "
                "VOA May 2020 polygons in England and the Rent Officers Wales May 2012 "
                "polygons in Wales; centroids outside every polygon take the documented "
                "overrides. Each output area takes its LSOA's BRMA. Ported from "
                "uk-data tools/brma_households/build.py (#516)."
            ),
            "scotland": (
                "Each 2022 output area's population-weighted centroid within the Scottish "
                "Government polygons (then intersects for centroids on a boundary); a "
                "centroid outside every polygon takes its master postcode's single BRMA in "
                "Rent Service Scotland's lookup, and where the lookup names one BRMA that "
                "contradicts the polygon the lookup wins. The two output areas it leaves "
                "ambiguous are overridden. Ported from microcosm#1092."
            ),
            "regions": (
                "Each area's FRS region and council come from the atomic supports, so the "
                "layer and the geography assignment share one roster."
            ),
        },
        "registrations": {
            name: {key: value for key, value in entry.items() if key != "path"}
            for name, entry in sorted(files.items())
        },
        "supports": {
            system: {"filename": entry["filename"], "sha256": entry["sha256"]}
            for system, entry in sorted(supports.items())
        },
        "receipts": {**ew_receipts, **scotland_receipts, "gb_brmas": len(gb_names)},
        "agreement": {
            "uk_data_516_regions": agreement,
            "scotland_brma_council_pairs": [
                {
                    "brma": str(row.brma),
                    "local_authority": str(row.council),
                    "output_areas": int(row.output_areas),
                    "centroid_households": int(row.households),
                }
                for row in pairs.itertuples(index=False)
            ],
        },
    }
    return resource, provenance


def _render(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def _stale(path: Path, payload: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return True
    current = json.loads(path.read_text(encoding="utf-8"))
    current.pop("built_on", None)
    rebuilt = dict(payload)
    rebuilt.pop("built_on", None)
    if "resource_sha256" in rebuilt:
        current.pop("resource_sha256", None)
        rebuilt.pop("resource_sha256", None)
    return current != rebuilt


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    resource, provenance = build(args)
    if args.check:
        stale = [
            str(path)
            for path, payload in (
                (args.out, resource),
                (args.provenance_out, provenance),
            )
            if _stale(path, payload)
        ]
        if stale:
            raise SystemExit(f"stale: {', '.join(stale)}")
        print("current")
        return 0
    args.out.write_text(_render(resource), encoding="utf-8")
    args.provenance_out.write_text(_render(provenance), encoding="utf-8")
    print(json.dumps(provenance["receipts"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
