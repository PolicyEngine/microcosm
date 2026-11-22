"""Build Belgium's atomic geography support and assignment from pinned sources.

Writes ``be/transport_atomic_geography_support.json``, one row per NIS 2025
commune that ``be/precal_references.json`` names as atomic geography support,
and ``be/geography_assignment.json``, one national commune draw followed by four
derived layers. Both stay inactive: the package and transport spec loaders
read and hash them, but Belgium declares no transport graph to consume them.
Every number-shaping choice here is a proposal of METHOD-CARD-BE row BE-MC12,
pending Max's ruling d1902.

The tool fetches nothing. ``--sources DIR`` (or ``BE_GEOGRAPHY_SOURCES``) names a
directory that holds the four files in ``SOURCES``. Each file is refused unless
its size and SHA-256 equal the pin:

- Statbel's population structure zip, for commune, province and region codes
  (the source capture of Chronicle package statbel-population-structure-2025);
- Chronicle's curated Statbel NIS 2025 commune crosswalk (2024 to 2025 codes);
- Eurostat's LAU 2024 to NUTS 2024 table, for each 2024 commune's NUTS3 area;
- Eurostat's NUTS 2024 classification, for the labels of BE335 and BE336.

The two Eurostat files are kept, with their URLs, under
``~/reviews/belgium-2026-10-12/transport-port/sources/eurostat-2026-11-22/``;
Eurostat updates them in place, so a fresh download may not match the pins.

The builder refuses unless the sources agree (``hierarchy_from_sources``,
rules 1 to 7): the crosswalk lists each 2024 code once, Eurostat lists each
LAU once with a NUTS3 code of the pinned NUTS 2024 classification, and the
classification lists each code once, at the NUTS level its length gives; the
crosswalk maps exactly Eurostat's 2024 LAUs onto exactly Statbel's 2025
communes; Statbel's three regions map one to one onto NUTS 2024 level 1, as
02000 BE2, 03000 BE3 and 04000 BE1; its province codes, with 04000 for the 19
Brussels-Capital communes, which belong to no province, map one to one onto
level 2, leaving out the commune 46030, whose
2024 parts lie in BE21 and BE23; the communes whose 2024 parts span two NUTS3
areas are exactly 46029 (BE234 and BE236), 46030 (BE211 and BE236) and 71072
(BE223 and BE224); the BE335 and BE336 labels equal their pins; and NUTS3
BE336, "Bezirk Verviers — Deutschsprachige Gemeinschaft", holds exactly nine
communes that are unchanged from 2024 to 2025. Those nine take the language
region ``german``; every other commune takes its Statbel region's language
region. That reading is inferred, as no pinned file states it: BE336's
communes are the German-speaking Community's territory, that territory is the
German language region, and each Region's territory is its language region or
regions.

``--check`` compares the committed files' bytes with a fresh build and never
writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

SPEC = (
    Path(__file__).resolve().parents[1]
    / "packages/microcosm-build/src/microcosm/build/be"
)
SUPPORT = "transport_atomic_geography_support.json"
ASSIGNMENT = "geography_assignment.json"
PRECAL = "precal_references.json"

SOURCES = {
    "statbel-population-structure-2025": {
        "file": "TF_SOC_POP_STRUCT_2025.zip",
        "sha256": "33910ea36437e39cfffd82ad82c759a2a8d1f2ab7662d45ce71c8ec9f336716c",
        "bytes": 2551751,
    },
    "statbel-nis-2025-commune-crosswalk": {
        "file": "statbel_nis_2025_commune_crosswalk.csv",
        "sha256": "717b0fecc3d78ee072a07f35ada4280f2ae4595ef2366b23d02e96efd872b1ba",
        "bytes": 96285,
    },
    "eurostat-lau-2024-nuts-2024": {
        "file": "EU-27-LAU-2024-NUTS-2024.xlsx",
        "sha256": "1528df47dfc8fa6815e3e7b9fca78b38d40c1142c6b4748ff5fa512f281b5446",
        "bytes": 19299900,
        "url": "https://ec.europa.eu/eurostat/documents/345175/501971/EU-27-LAU-2024-NUTS-2024.xlsx",
    },
    "eurostat-nuts-2021-2024": {
        "file": "NUTS2021-NUTS2024.xlsx",
        "sha256": "1c46016ce209e1cf61cfb66535374dba58851485a72006c6db72676d5dcd9623",
        "bytes": 399811,
        "url": "https://ec.europa.eu/eurostat/documents/345175/629341/NUTS2021-NUTS2024.xlsx",
    },
}

GERMAN_NUTS3 = "BE336"
FRENCH_VERVIERS_NUTS3 = "BE335"
NUTS3_LABELS = {
    GERMAN_NUTS3: "Bezirk Verviers — Deutschsprachige Gemeinschaft",
    FRENCH_VERVIERS_NUTS3: "Arr. Verviers — communes francophones",
}
GERMAN_COMMUNES = (
    "63001",
    "63012",
    "63013",
    "63023",
    "63040",
    "63048",
    "63061",
    "63067",
    "63087",
)
# Each commune whose 2024 parts lie in two NUTS3 areas, with those areas.
NUTS3_STRADDLERS = {
    "46029": ("BE234", "BE236"),
    "46030": ("BE211", "BE236"),
    "71072": ("BE223", "BE224"),
}
NUTS2_STRADDLERS = ("46030",)
NUTS1_BY_REGION = {"02000": "BE2", "03000": "BE3", "04000": "BE1"}
LANGUAGE_BY_REGION = {
    "02000": "dutch",
    "03000": "french",
    "04000": "brussels_bilingual",
}

SYSTEM = "be-commune"
LEVEL = "commune"
CODE_SYSTEM = "be_nis"
VINTAGE = "nis_2025"
STREAM = ["sha256-u53-v1", "be-commune-geography", 0, 0]
OUTPUTS = {
    "area": "be_atomic_area",
    "system": "be_area_system",
    "basis": "be_area_basis",
}
LAYERS = ("commune_nis", "province_nis", "region_nuts1", "community_code")
PRECAL_USE = "atomic_geography_support"


def _pin(key: str) -> str:
    return f"{key}@sha256:{SOURCES[key]['sha256']}"


def _pairs(mapping: dict[str, str]) -> str:
    return ", ".join(f"{key} {value}" for key, value in mapping.items())


_ZIP = _pin("statbel-population-structure-2025")
_CROSSWALK = _pin("statbel-nis-2025-commune-crosswalk")
_LAU = _pin("eurostat-lau-2024-nuts-2024")
_NUTS = _pin("eurostat-nuts-2021-2024")
_FACTS = (
    "Statbel registered population on 1 January 2025 by NIS 2025 commune, "
    "Chronicle record set "
    "statbel.population_structure.cy2025.people.by_commune_nis2025.total"
)

COLUMNS = {
    "area": {
        "kind": "code",
        "source": _FACTS,
        "vintage": VINTAGE,
        "relation": "exact",
    },
    "commune_nis": {
        "kind": "code",
        "source": _FACTS,
        "vintage": VINTAGE,
        "relation": "exact",
    },
    "province_nis": {
        "kind": "code",
        "source": (
            f"{_ZIP}#CD_PROV_REFNIS, Statbel's 2025 province code; the 19 "
            "Brussels-Capital Region communes belong to no province and Statbel "
            "codes them 04000"
        ),
        "vintage": "province_2025",
        "relation": "exact",
    },
    "region_nuts1": {
        "kind": "code",
        "source": (
            f"{_ZIP}#CD_RGN_REFNIS, renamed one to one to NUTS 2024 level 1 "
            f"({_pairs(NUTS1_BY_REGION)}) by {_LAU}#BE/NUTS3 via "
            f"{_CROSSWALK}"
        ),
        "vintage": "nuts1_2025",
        "relation": "exact",
    },
    "community_code": {
        "kind": "code",
        "source": (
            "language region: german for the 9 communes in NUTS 2024 BE336 "
            f"'{NUTS3_LABELS[GERMAN_NUTS3]}' (beside BE335 "
            f"'{NUTS3_LABELS[FRENCH_VERVIERS_NUTS3]}') by {_LAU}#BE/NUTS3 via "
            f"{_CROSSWALK}, labels {_NUTS}#NUTS2024; otherwise from "
            f"{_ZIP}#CD_RGN_REFNIS ({_pairs(LANGUAGE_BY_REGION)}); "
            "inferred, as no pinned file states it: "
            "BE336's communes are the German-speaking Community's territory, that "
            "territory is the German language region, and each Region's territory "
            "is its language region or regions (Constitution art. 4 names the four "
            "language regions)"
        ),
        "vintage": "community_2025",
        "relation": "exact",
    },
    "population": {
        "kind": "weight",
        "source": _FACTS,
        "basis": (
            "Statbel registered population on 1 January 2025, one support row "
            "per commune"
        ),
    },
}


def encode(document) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def _read_pinned(directory: Path, key: str) -> bytes:
    pin = SOURCES[key]
    data = (directory / pin["file"]).read_bytes()
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError(f"{pin['file']} differs from its pin; rebuild deliberately.")
    return data


def read_sources(directory: Path) -> dict:
    """Verify the four pinned files and parse them into plain mappings."""

    import openpyxl

    statbel = []
    raw = _read_pinned(directory, "statbel-population-structure-2025")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        (member,) = archive.namelist()
        with archive.open(member) as handle:
            reader = csv.DictReader(
                io.TextIOWrapper(handle, encoding="utf-8-sig"), delimiter="|"
            )
            for row in reader:
                statbel.append(
                    (
                        row["CD_REFNIS"],
                        row["CD_PROV_REFNIS"],
                        row["CD_RGN_REFNIS"],
                        int(row["MS_POPULATION"]),
                    )
                )
    text = _read_pinned(directory, "statbel-nis-2025-commune-crosswalk").decode("utf-8")
    crosswalk = [
        (row["source_nis"], row["target_nis"])
        for row in csv.DictReader(io.StringIO(text))
    ]
    book = openpyxl.load_workbook(
        io.BytesIO(_read_pinned(directory, "eurostat-lau-2024-nuts-2024")),
        read_only=True,
        data_only=True,
    )
    lau = lau_pairs(list(book["BE"].iter_rows(values_only=True)))
    book = openpyxl.load_workbook(
        io.BytesIO(_read_pinned(directory, "eurostat-nuts-2021-2024")),
        read_only=True,
        data_only=True,
    )
    labels = nuts_labels(list(book["NUTS2024"].iter_rows(values_only=True)))
    return {"statbel": statbel, "crosswalk": crosswalk, "lau": lau, "labels": labels}


def lau_pairs(rows: list[tuple]) -> list[tuple[str, str]]:
    """Read (2024 LAU, NUTS3) pairs from the Eurostat LAU sheet's rows."""

    if rows[0][:2] != ("NUTS3", "LAU CODE"):
        raise ValueError(f"Unexpected Eurostat LAU header {rows[0][:2]}.")
    pairs = []
    for nuts3, code, *_ in rows[1:]:
        if code is None:
            continue
        if not (isinstance(nuts3, str) and len(nuts3) == 5 and nuts3[:2] == "BE"):
            raise ValueError(
                f"Eurostat LAU {code} has NUTS3 {nuts3!r}, not a five-character BE code."
            )
        pairs.append((str(code), nuts3))
    return pairs


def nuts_labels(rows: list[tuple]) -> list[tuple[str, str]]:
    """Read Belgium's (code, label) pairs from the NUTS 2024 sheet's rows.

    Each Belgian row's declared NUTS level must equal its code's length less the
    two-letter country code, so Rule 2 can read a five-character code as NUTS3.
    """

    header = ("Country code", "NUTS Code", "NUTS label", "NUTS level")
    if rows[0][:4] != header:
        raise ValueError(f"Unexpected NUTS 2024 header {rows[0][:4]}.")
    pairs = []
    for country, code, label, level, *_ in rows[1:]:
        if country != "BE":
            continue
        if not (isinstance(code, str) and level == len(code) - 2):
            raise ValueError(
                f"Rule 2: NUTS 2024 gives {code!r} level {level!r}, "
                "which its length does not."
            )
        pairs.append((code, str(label)))
    return pairs


def statbel_population(sources: dict) -> dict[str, int]:
    """Sum Statbel's zip population by commune (all sexes, ages and nationalities)."""

    totals: dict[str, int] = defaultdict(int)
    for commune, _, _, population in sources["statbel"]:
        totals[commune] += population
    return dict(totals)


def _unique(pairs, message: str) -> dict:
    """Return the pairs as a mapping, refusing a key that appears twice."""

    mapping = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(message.format(key))
        mapping[key] = value
    return mapping


def _one_to_one(pairs, rule: str) -> dict:
    forward, backward = defaultdict(set), defaultdict(set)
    for left, right in pairs:
        forward[left].add(right)
        backward[right].add(left)
    if any(len(v) != 1 for v in (*forward.values(), *backward.values())):
        raise ValueError(f"{rule}: {dict(forward)}")
    return {left: next(iter(right)) for left, right in forward.items()}


def hierarchy_from_sources(sources: dict) -> dict[str, dict[str, str]]:
    """Return each NIS 2025 commune's derived codes, refusing any disagreement."""

    statbel = {}
    for commune, province, region, _ in sources["statbel"]:
        if statbel.setdefault(commune, (province, region)) != (province, region):
            raise ValueError(f"Statbel gives {commune} two provinces or regions.")
    # Each key appears once, so no later row can hide a conflicting earlier one.
    targets = _unique(
        sources["crosswalk"], "Rule 2: the crosswalk lists 2024 code {} twice."
    )
    lau = _unique(sources["lau"], "Rule 2: Eurostat lists LAU {} twice.")
    labels = _unique(sources["labels"], "Rule 5: NUTS 2024 lists {} twice.")
    parts = defaultdict(set)
    for source, target in targets.items():
        parts[target].add(source)
    # Rule 1: the crosswalk covers exactly Statbel's 2025 communes.
    if set(parts) != set(statbel):
        raise ValueError("Rule 1: crosswalk targets differ from Statbel's communes.")
    # Rule 2: the crosswalk's 2024 codes are exactly Eurostat's 2024 LAUs, each
    # in a NUTS3 area of the pinned NUTS 2024 classification.
    if set(targets) != set(lau):
        raise ValueError("Rule 2: crosswalk sources differ from Eurostat's LAUs.")
    unknown = sorted(set(lau.values()) - {code for code in labels if len(code) == 5})
    if unknown:
        raise ValueError(f"Rule 2: NUTS 2024 has no NUTS3 area {unknown}.")
    nuts3 = {commune: {lau[part] for part in codes} for commune, codes in parts.items()}
    # Rule 3: one NUTS1 per commune, one to one with Statbel's regions.
    nuts1 = {}
    for commune, cells in nuts3.items():
        levels = {cell[:3] for cell in cells}
        if len(levels) != 1:
            raise ValueError(f"Rule 3: {commune} spans NUTS1 {sorted(levels)}.")
        nuts1[commune] = levels.pop()
    region = _one_to_one(
        ((statbel[c][1], nuts1[c]) for c in statbel),
        "Rule 3: Statbel regions and NUTS1 differ",
    )
    # Rule 4: provinces one to one with NUTS2 where a commune's parts share one.
    straddle3 = {
        c: tuple(sorted(cells)) for c, cells in sorted(nuts3.items()) if len(cells) > 1
    }
    straddle2 = sorted(
        c for c, cells in nuts3.items() if len({x[:4] for x in cells}) > 1
    )
    if (straddle3, tuple(straddle2)) != (NUTS3_STRADDLERS, NUTS2_STRADDLERS):
        raise ValueError(f"Rule 4: NUTS straddlers {straddle3}, {straddle2} differ.")
    _one_to_one(
        (
            (statbel[c][0], next(iter(nuts3[c]))[:4])
            for c in statbel
            if c not in straddle2
        ),
        "Rule 4: Statbel provinces and NUTS2 differ",
    )
    # Rule 5: the Verviers NUTS3 labels are the pinned ones.
    if any(labels.get(code) != label for code, label in NUTS3_LABELS.items()):
        raise ValueError("Rule 5: a Verviers NUTS3 label differs from its pin.")
    # Rule 6: BE336 holds exactly the nine pinned communes, each unchanged.
    german = sorted(c for c, cells in nuts3.items() if GERMAN_NUTS3 in cells)
    if tuple(german) != GERMAN_COMMUNES or any(
        nuts3[c] != {GERMAN_NUTS3} or parts[c] != {c} for c in german
    ):
        raise ValueError(f"Rule 6: BE336 members {german} differ from the pin.")
    # Rule 7: Statbel's three regions, each with its pinned NUTS1 code.
    if region != NUTS1_BY_REGION:
        raise ValueError(f"Rule 7: Statbel regions and NUTS1 {region} differ.")
    return {
        commune: {
            "province_nis": province,
            "region_nuts1": region[statbel_region],
            "community_code": (
                "german" if commune in german else LANGUAGE_BY_REGION[statbel_region]
            ),
        }
        for commune, (province, statbel_region) in sorted(statbel.items())
    }


def geography_rows(precal: dict) -> list[tuple[str, str]]:
    """Return (target, NIS code) for each atomic geography support row."""

    rows = []
    for reference in precal["target_references"]:
        if reference.get("metadata", {}).get("precal_use") != PRECAL_USE:
            continue
        selector = reference["ledger_selector"]
        if (selector["geography_level"], selector["geography_vintage"]) != (
            LEVEL,
            VINTAGE,
        ):
            raise ValueError(f"{reference['name']} is not a {VINTAGE} commune row.")
        rows.append((reference["name"], selector["geography_id"]))
    return sorted(rows, key=lambda row: row[1])


def build_documents(hierarchy: dict, precal: dict) -> dict[str, dict]:
    rows = geography_rows(precal)
    if [code for _, code in rows] != sorted(hierarchy):
        raise ValueError("Rule 1: precal geography rows differ from the communes.")
    support = {
        "version": 1,
        "level": LEVEL,
        "code_system": CODE_SYSTEM,
        "vintage": VINTAGE,
        "share_tolerance": 0,
        "population_geography_level": LEVEL,
        "population_code_column": "commune_nis",
        "columns": COLUMNS,
        "rows": [
            {
                "target": target,
                "codes": {"area": code, "commune_nis": code, **hierarchy[code]},
                "population_share": 1,
            }
            for target, code in rows
        ],
    }
    layers = [
        {
            "input": name,
            "output": name,
            "vintage": COLUMNS[name]["vintage"],
            "relation": COLUMNS[name]["relation"],
            "source": COLUMNS[name]["source"],
        }
        for name in LAYERS
    ]
    assignment = {
        "version": 1,
        "identity": ["household_id"],
        "stream": STREAM,
        "outputs": OUTPUTS,
        "systems": [
            {
                "id": SYSTEM,
                "level": LEVEL,
                "code_system": CODE_SYSTEM,
                "vintage": VINTAGE,
                "source": "be_chronicle_facts",
                "selector": {},
                "constraints": [],
                "observed_area": None,
                "stages": [{"level": "area", "weight": "population"}],
                "layers": layers,
            }
        ],
    }
    return {SUPPORT: support, ASSIGNMENT: assignment}


def stale_files(documents: dict[str, dict], out: Path) -> list[str]:
    return [
        str(out / name)
        for name, document in documents.items()
        if not (out / name).exists()
        or (out / name).read_bytes() != encode(document).encode("utf-8")
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=os.environ.get("BE_GEOGRAPHY_SOURCES"))
    parser.add_argument("--out", type=Path, default=SPEC)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    # An empty or blank value names no directory, not the current one.
    if not (args.sources or "").strip():
        parser.error("--sources or BE_GEOGRAPHY_SOURCES is required")
    precal = json.loads((args.out / PRECAL).read_text(encoding="utf-8"))
    hierarchy = hierarchy_from_sources(read_sources(Path(args.sources)))
    documents = build_documents(hierarchy, precal)
    if args.check:
        stale = stale_files(documents, args.out)
        if stale:
            raise SystemExit(f"Stale generated resource(s): {stale}")
        return
    for name, document in documents.items():
        (args.out / name).write_bytes(encode(document).encode("utf-8"))


if __name__ == "__main__":
    main()
