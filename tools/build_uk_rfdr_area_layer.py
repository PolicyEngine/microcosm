#!/usr/bin/env python
"""Build the Rural Fuel Duty Relief area layer (microcosm#1418).

HMRC Notice 2001 opens the relief to retailers in the Inner and Outer
Hebrides, the Northern Isles, the Islands in the Clyde, the Isles of Scilly,
the post town of Hawes and sixteen mainland postcode districts. The notice
names the island groups without bounding them, so ``RFDR_POSTCODE_DEFINITION``
below declares Microcosm's translation of every part into postcode districts,
unit postcodes and excluded units, with its sources.

The tool reads the ONS National Statistics Postcode Lookup's best-fit 2021
output area of every postcode (Scotland's 2022 output areas under the same
column) and writes ``uk/rfdr_area_layer.json``: the output areas more than
half of whose postcodes lie in the scheme, which the post-geography node
``uk.full.geography.rfdr`` reads. Its provenance records the inputs' digests,
the definition and the counts.

    uv run --no-sync python tools/build_uk_rfdr_area_layer.py \\
        --nspl-zip NSPL_AUG_2026.zip \\
        --nspl-manifest ../chronicle/db/data/ons/nspl_august_2026/manifest.yaml

``--postcode-oa-parquet`` reads the postcode and output-area columns the high
value home stock's stage A already reduced from the same registered zip
(``tools/build_uk_high_value_home_stock.py``), when the zip is not at hand;
the provenance names which input the layer was built from. The committed layer
is built from the registered zip, and ``test_uk_rfdr_area_layer.py`` holds it
to that (Anthony's review of microcosm#1420). The registration is pinned here,
package id, file and digest, as the BRMA tool pins its own, and ``--check``
rebuilds the layer in memory and compares it with the committed files, writing
nothing (the independent review of microcosm#1420):

    uv run --no-sync python tools/build_uk_rfdr_area_layer.py --check \\
        --nspl-zip NSPL_AUG_2026.zip \\
        --nspl-manifest ../chronicle/db/data/ons/nspl_august_2026/manifest.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UK = ROOT / "packages/microcosm-build/src/microcosm/build/uk"
_MODULE = (
    ROOT / "packages/microcosm-build/src/microcosm/build/uk_runtime/rfdr_area_layer.py"
)
_HVH_MODULE = (
    ROOT
    / "packages/microcosm-build/src/microcosm/build/uk_runtime/high_value_home_stock.py"
)
NOTICE_URL = "https://www.gov.uk/guidance/rural-duty-relief-scheme-notice-2001"
#: The Chronicle raw-only registration the layer reads, with the bytes pinned
#: here; the high value home stock's provenance registers the same zip.
NSPL_REGISTRATION = {
    "package_id": "ons-nspl-august-2026",
    "filename": "NSPL_AUG_2026.zip",
    "sha256": "f5d86a518d360cad67f0304f25574d3ce619f242d9ea6385a120abab7c15b15c",
    "member": "Data/NSPL_AUG_2026_UK.csv",
}
#: Microcosm's translation of the scheme's areas (microcosm#1418). SI 2011/2935
#: Schedule 1 (inserted by SI 2015/550) and Notice 2001 name the island groups
#: without listing islands or postcodes, so each group is its postcode
#: districts (bridged islands included: Decision 2011/776/EU covers "all the
#: islands" of the groups and HMRC's 2013 working paper lists stations on Skye
#: and Seil). An island that shares a sector with the mainland joins by its own
#: units, taken from the NRS Census 2022 island groups' output areas; the post
#: town of Hawes is the DL8 3 units HM Land Registry addresses with that post
#: town, and those of the same parish output areas with no sale on record. The
#: Outer Hebrides' pre-1995 PA81 to PA88 codes survive as terminated postcodes in
#: the lookup and are kept, so their output areas are not split. Mainland units
#: inside an island district are excluded.
RFDR_POSTCODE_DEFINITION: dict = {
    "districts": {
        "notice_mainland_districts": [
            "EX35",
            "LA17",
            "NE48",
            "IV14",
            "IV21",
            "IV22",
            "IV26",
            "IV27",
            "IV54",
            "KW12",
            "PA38",
            "PA80",
            "PH19",
            "PH23",
            "PH36",
            "PH41",
        ],
        "outer_hebrides": [
            "HS1",
            "HS2",
            "HS3",
            "HS4",
            "HS5",
            "HS6",
            "HS7",
            "HS8",
            "HS9",
        ],
        "northern_isles": [
            "KW15",
            "KW16",
            "KW17",
            "ZE1",
            "ZE2",
            "ZE3",
        ],
        "isles_of_scilly": [
            "TR21",
            "TR22",
            "TR23",
            "TR24",
            "TR25",
        ],
        "islands_of_the_clyde": [
            "KA27",
            "KA28",
            "PA20",
        ],
        "inner_hebrides": [
            "IV41",
            "IV42",
            "IV43",
            "IV44",
            "IV45",
            "IV46",
            "IV47",
            "IV48",
            "IV49",
            "IV51",
            "IV55",
            "IV56",
            "PA41",
            "PA42",
            "PA43",
            "PA44",
            "PA45",
            "PA46",
            "PA47",
            "PA48",
            "PA49",
            "PA60",
            "PA61",
            "PA62",
            "PA63",
            "PA64",
            "PA65",
            "PA66",
            "PA67",
            "PA68",
            "PA69",
            "PA70",
            "PA71",
            "PA72",
            "PA73",
            "PA74",
            "PA75",
            "PA76",
            "PA77",
            "PA78",
            "PH42",
            "PH43",
            "PH44",
        ],
        "outer_hebrides_withdrawn_before_1995": [
            "PA81",
            "PA82",
            "PA83",
            "PA84",
            "PA85",
            "PA86",
            "PA87",
            "PA88",
        ],
    },
    "sectors": [],
    "units": {
        "raasay": [
            "IV40 8NG",
            "IV40 8NS",
            "IV40 8NT",
            "IV40 8NU",
            "IV40 8NX",
            "IV40 8NY",
            "IV40 8NZ",
            "IV40 8PA",
            "IV40 8PB",
            "IV40 8PD",
            "IV40 8PE",
            "IV40 8PF",
            "IV40 8PG",
            "IV40 8PR",
            "IV40 8YA",
        ],
        "skye_in_iv40_8": [
            "IV40 8DW",
        ],
        "seil": [
            "PA34 4GH",
            "PA34 4QZ",
            "PA34 4RA",
            "PA34 4RB",
            "PA34 4RD",
            "PA34 4RE",
            "PA34 4RF",
            "PA34 4RG",
            "PA34 4RJ",
            "PA34 4RQ",
            "PA34 4TE",
            "PA34 4TF",
            "PA34 4TG",
            "PA34 4TH",
            "PA34 4TJ",
            "PA34 4TL",
            "PA34 4TN",
            "PA34 4TP",
            "PA34 4TR",
            "PA34 4TW",
        ],
        "easdale": [
            "PA34 4TB",
        ],
        "luing_and_shuna": [
            "PA34 4AP",
            "PA34 4SG",
            "PA34 4SZ",
            "PA34 4TU",
            "PA34 4TX",
            "PA34 4TY",
            "PA34 4TZ",
            "PA34 4UA",
            "PA34 4UB",
            "PA34 4UD",
            "PA34 4UE",
            "PA34 4UF",
            "PA34 4UG",
            "PA34 4UH",
            "PA34 4UW",
            "PA34 4WN",
            "PA34 4WZ",
        ],
        "kerrera": [
            "PA34 4PE",
            "PA34 4PF",
            "PA34 4QH",
            "PA34 4QJ",
            "PA34 4SX",
        ],
        "lismore": [
            "PA34 5UG",
            "PA34 5UH",
            "PA34 5UJ",
            "PA34 5UL",
            "PA34 5UN",
            "PA34 5UQ",
        ],
        "hawes_in_land_registry_addresses": [
            "DL8 3LH",
            "DL8 3LJ",
            "DL8 3LL",
            "DL8 3LN",
            "DL8 3LP",
            "DL8 3LQ",
            "DL8 3LR",
            "DL8 3LT",
            "DL8 3LU",
            "DL8 3LW",
            "DL8 3LX",
            "DL8 3LY",
            "DL8 3LZ",
            "DL8 3NA",
            "DL8 3NB",
            "DL8 3NH",
            "DL8 3NJ",
            "DL8 3NL",
            "DL8 3NP",
            "DL8 3NR",
            "DL8 3NS",
            "DL8 3NT",
            "DL8 3NU",
            "DL8 3NW",
            "DL8 3NX",
            "DL8 3PA",
            "DL8 3PB",
            "DL8 3PD",
            "DL8 3PE",
            "DL8 3PF",
            "DL8 3PG",
            "DL8 3PH",
            "DL8 3PJ",
            "DL8 3PL",
            "DL8 3PN",
            "DL8 3PQ",
            "DL8 3PR",
            "DL8 3PU",
            "DL8 3PX",
            "DL8 3PZ",
            "DL8 3QA",
            "DL8 3QF",
            "DL8 3QG",
            "DL8 3QL",
            "DL8 3QP",
            "DL8 3QR",
            "DL8 3QT",
            "DL8 3QU",
            "DL8 3QW",
            "DL8 3QX",
            "DL8 3QY",
            "DL8 3QZ",
            "DL8 3RA",
            "DL8 3RD",
            "DL8 3RE",
            "DL8 3RF",
            "DL8 3RG",
            "DL8 3RH",
            "DL8 3RJ",
            "DL8 3RL",
            "DL8 3RN",
            "DL8 3RP",
            "DL8 3RS",
            "DL8 3RT",
            "DL8 3RU",
            "DL8 3RW",
            "DL8 3RX",
            "DL8 3RY",
            "DL8 3RZ",
            "DL8 3SA",
            "DL8 3SD",
            "DL8 3SF",
            "DL8 3SG",
            "DL8 3SH",
            "DL8 3SJ",
            "DL8 3SN",
            "DL8 3UZ",
        ],
        "hawes_parish_output_areas_without_sales": [
            "DL8 3AU",
            "DL8 3AX",
            "DL8 3AY",
            "DL8 3LS",
            "DL8 3ND",
            "DL8 3NN",
            "DL8 3PP",
            "DL8 3PS",
            "DL8 3PT",
            "DL8 3PW",
            "DL8 3PY",
            "DL8 3QB",
            "DL8 3QD",
            "DL8 3QE",
            "DL8 3QH",
            "DL8 3QJ",
            "DL8 3QN",
            "DL8 3QQ",
            "DL8 3QS",
            "DL8 3RB",
            "DL8 3RQ",
            "DL8 3RR",
            "DL8 3SB",
            "DL8 3SE",
            "DL8 3SL",
            "DL8 3SW",
            "DL8 3UW",
            "DL8 3WY",
            "DL8 3YX",
            "DL8 3YZ",
        ],
    },
    "excluded_units": [
        "IV41 8WA",
        "IV41 8WZ",
        "IV45 8YD",
        "KA27 8SW",
        "KA28 0DA",
        "KA28 0WX",
        "PA87 2BY",
        "TR21 0AA",
        "TR21 0JS",
        "TR21 0PN",
        "TR21 0PR",
        "TR21 0PW",
        "TR21 0WY",
        "TR21 0WZ",
        "TR21 0XN",
        "TR21 0YD",
        "TR21 0YE",
        "TR21 0YH",
        "TR21 0ZS",
        "TR21 0ZT",
        "TR21 0ZU",
        "TR21 0ZW",
        "TR21 0ZX",
    ],
}
#: Places a reading of the notice could include that the definition leaves out:
#: each lies in a mainland output area, so none would change a flag.
RFDR_UNRESOLVED: dict = {
    "PA28 6RS": "Sanda (Islands of the Clyde per Wikipedia; uninhabited at 2011 census). Live unit located on Sanda but NRS puts it in a mainland Kintyre OA; flagging it part-flags that OA.",
    "PA37 1SD": "Eriska (bridged sea-loch island, Loch Creran/Linnhe); listed in Wikipedia's List of Inner Hebrides with the caveat that sea-loch islands are not ordinarily Hebridean; mainland OA.",
    "PA37 1AW": "Eriska, as above.",
    "DL8 3JL": "High Abbotside parish OA but no Land Registry address; the nearest J-range unit (DL8 3JJ) is post town LEYBURN.",
    "Davaar (PA28 6)": "Tidal island in Campbeltown Loch; no live postcode found on it.",
    "Danna and Loch Sween/Crinan islets (PA31 8)": "Sea-loch islands; no postcode found; not normally counted as Inner Hebrides.",
}
#: The sources behind each part of the definition.
RFDR_SOURCES: dict = {
    "hmrc_notice_2001": "https://www.gov.uk/guidance/rural-duty-relief-scheme-notice-2001",
    "si_2011_2935_current": "https://www.legislation.gov.uk/uksi/2011/2935",
    "si_2015_550": "https://www.legislation.gov.uk/uksi/2015/550/made",
    "em_si_2011_2935": "https://legislation.gov.uk/uksi/2011/2935/pdfs/uksiem_20112935_en.pdf",
    "em_si_2015_550": "https://legislation.gov.uk/uksi/2015/550/pdfs/uksiem_20150550_en.pdf",
    "eu_decision_2011_776": "https://legislation.gov.uk/eudn/2011/776/2011-11-24/data.xht",
    "eu_decision_2015_356": "https://www.legislation.gov.uk/eudn/2015/356/adopted",
    "com_2015_4_hawes_villages": "https://www.senat.fr/europe/textes_europeens/e9977.pdf",
    "tiin_2015": "https://assets.publishing.service.gov.uk/media/5a80399a40f0b6230269240a/Rural_Fuel_Duty_Relief.pdf",
    "hmrc_working_paper_15_2013": "https://assets.publishing.service.gov.uk/media/5a7e1837e5274a2e87daf829/rural-fuel-duty-relief.pdf",
    "nrs_census_2022_index_island_groups": "https://www.nrscotland.gov.uk/media/utrbt5ze/census_2022_index.zip",
    "ons_nspl_aug_2026": "https://geoportal.statistics.gov.uk/datasets/9d8364ebae8b4439aa66cda440e54fc8",
    "hm_land_registry_sparql_post_town": "https://landregistry.data.gov.uk/landregistry/query",
    "postcodes_io_live_status_and_coordinates": "https://api.postcodes.io",
    "wikipedia_HS": "https://en.wikipedia.org/wiki/HS_postcode_area",
    "wikipedia_KW": "https://en.wikipedia.org/wiki/KW_postcode_area",
    "wikipedia_ZE": "https://en.wikipedia.org/wiki/ZE_postcode_area",
    "wikipedia_TR": "https://en.wikipedia.org/wiki/TR_postcode_area",
    "wikipedia_KA": "https://en.wikipedia.org/wiki/KA_postcode_area",
    "wikipedia_PA": "https://en.wikipedia.org/wiki/PA_postcode_area",
    "wikipedia_IV": "https://en.wikipedia.org/wiki/IV_postcode_area",
    "wikipedia_PH": "https://en.wikipedia.org/wiki/PH_postcode_area",
    "wikipedia_DL": "https://en.wikipedia.org/wiki/DL_postcode_area",
    "wikipedia_list_of_inner_hebrides": "https://en.wikipedia.org/wiki/List_of_Inner_Hebrides",
    "wikipedia_islands_of_the_clyde": "https://en.wikipedia.org/wiki/Islands_of_the_Clyde",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registered_sha256(manifest: Path) -> str:
    """The NSPL zip's digest, once the Chronicle manifest registers the pin.

    The manifest must be the pinned package's, registering the pinned file and
    bytes alone: another lookup's manifest refuses, whatever its digest.
    """

    import yaml

    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if payload.get("package_id") != NSPL_REGISTRATION["package_id"]:
        raise SystemExit(
            f"error: {manifest} registers package {payload.get('package_id')!r}, "
            f"not {NSPL_REGISTRATION['package_id']!r}."
        )
    entries = [entry for entry in payload.get("files", {}).values()]
    if len(entries) != 1:
        raise SystemExit(f"error: {manifest} registers {len(entries)} files, not one.")
    (entry,) = entries
    if (entry.get("filename"), entry.get("sha256")) != (
        NSPL_REGISTRATION["filename"],
        NSPL_REGISTRATION["sha256"],
    ):
        raise SystemExit(
            f"error: {manifest} registers {entry.get('filename')!r} with sha256 "
            f"{entry.get('sha256')}, not the pinned {NSPL_REGISTRATION['filename']!r}."
        )
    return str(entry["sha256"])


def _manifest_label(manifest: Path) -> str:
    """The manifest's path in the Chronicle repository, as the provenance names it."""

    parts = manifest.resolve().parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("db", "data"):
            return "PolicyEngine/chronicle " + "/".join(parts[index:])
    return str(manifest)


def _postcodes_from_zip(path: Path) -> pd.DataFrame:
    hvh = _load(_HVH_MODULE, "_rfdr_hvh")
    with zipfile.ZipFile(path) as archive:
        with archive.open(NSPL_REGISTRATION["member"]) as stream:
            return hvh.read_nspl_postcode_output_areas(stream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--nspl-zip", type=Path)
    source.add_argument("--postcode-oa-parquet", type=Path)
    parser.add_argument("--nspl-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=UK / "rfdr_area_layer.json")
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=UK / "rfdr_area_layer.provenance.json",
    )
    parser.add_argument(
        "--built-on",
        default=None,
        help=(
            "The layer's build date (default: the committed layer's when nothing "
            "else in it moves, else today)."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Rebuild in memory and compare with --out and --provenance-json, "
            "writing nothing; exits 1 if either would change."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layer_module = _load(_MODULE, "_rfdr_area_layer")
    definition = layer_module.RFDRPostcodeDefinition.from_mapping(
        RFDR_POSTCODE_DEFINITION
    )
    registered = _registered_sha256(args.nspl_manifest)
    if args.nspl_zip is not None:
        digest = _sha256(args.nspl_zip)
        if digest != registered:
            raise SystemExit(
                f"error: {args.nspl_zip} has sha256 {digest}, not the registered "
                f"{registered}."
            )
        postcodes = _postcodes_from_zip(args.nspl_zip)
        source = {"kind": "nspl_zip", "sha256": digest}
    else:
        postcodes = pd.read_parquet(args.postcode_oa_parquet)
        source = {
            "kind": "nspl_postcode_oa21_parquet",
            "sha256": _sha256(args.postcode_oa_parquet),
            "reduced_by": "tools/build_uk_high_value_home_stock.py (stage A)",
        }
    if not {"postcode", "oa21cd"} <= set(postcodes.columns):
        raise SystemExit("error: the postcode lookup lacks postcode or oa21cd.")
    flagged, receipt = layer_module.flag_output_areas(postcodes, definition)
    # The committed layer's date stands unless something else in the layer
    # moves: a rebuild that changes only the date would move its digest, the
    # run-time pin and the node's source digest for nothing (the re-review of
    # microcosm#1420, 15). A missing file is a difference, not an error.
    committed = (
        json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else None
    )
    built_on = args.built_on or (
        committed["built_on"] if committed is not None else date.today().isoformat()
    )
    layer = {
        "schema_version": layer_module.SCHEMA_VERSION,
        "kind": layer_module.KIND,
        "resource": "rfdr_area_layer.json",
        "method": layer_module.METHOD_VERSION,
        "built_on": built_on,
        "description": (
            "The 2021 output areas of England and Wales and 2022 output areas of "
            "Scotland more than half of whose postcodes lie in the Rural Fuel Duty "
            "Relief scheme's areas (HMRC Notice 2001), by the ONS postcode lookup "
            "and Microcosm's declared translation of the notice's island groups "
            "into postcode districts, unit postcodes and excluded units. Read by the "
            "post-geography node "
            "uk.full.geography.rfdr (microcosm#1418); rebuilt by "
            "tools/build_uk_rfdr_area_layer.py."
        ),
        "definition": {
            "notice": NOTICE_URL,
            "districts": sorted(definition.districts),
            "sectors": sorted(definition.sectors),
            "units": sorted(definition.units),
            "excluded_units": sorted(definition.excluded_units),
        },
        "licence": {
            "licence": "Open Government Licence v3.0",
            "attribution": (
                "Contains OS data and Royal Mail data, Crown copyright and database "
                "right; Source: Office for National Statistics, licensed under the "
                "Open Government Licence v3.0."
            ),
        },
        "areas": flagged,
    }
    if (
        args.built_on is None
        and not args.check
        and committed is not None
        and layer != committed
    ):
        built_on = layer["built_on"] = date.today().isoformat()
    layer_text = json.dumps(layer, indent=2, sort_keys=True) + "\n"
    provenance = {
        "resource": "rfdr_area_layer.json",
        "built_on": built_on,
        "layer_sha256": hashlib.sha256(layer_text.encode("utf-8")).hexdigest(),
        "nspl_registration": {
            "manifest": _manifest_label(args.nspl_manifest),
            "zip_sha256": registered,
        },
        "postcode_source": source,
        "definition": RFDR_POSTCODE_DEFINITION,
        "unresolved": RFDR_UNRESOLVED,
        "sources": RFDR_SOURCES,
        "counts": receipt,
    }
    provenance_text = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if args.check:
        moved = [
            str(path)
            for path, text in (
                (args.out, layer_text),
                (args.provenance_json, provenance_text),
            )
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        print(json.dumps({"check": "differs" if moved else "same", "files": moved}))
        return 1 if moved else 0
    args.out.write_text(layer_text)
    args.provenance_json.write_text(provenance_text)
    print(json.dumps({"areas": {k: len(v) for k, v in flagged.items()}, **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
