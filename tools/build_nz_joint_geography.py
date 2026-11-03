"""Generate NZ's inactive TA x region x AS-area atomic support from pinned pages.

This tool performs no network access. ``NZ_AS_AREA_SOURCES`` supplies the
existing G8-m interior-point/centroid placement inputs, and
``NZ_JOINT_GEOGRAPHY_SOURCES`` supplies the separately pinned MBHG REGC pages.
The executable definition has exactly the schema accepted by
``geography.support_from_facts@1``. The separate audit records Census counts,
including outside-TA residents that have no published TA population anchor.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from tools.build_nz_as_area_crosswalk import (
    CONCORDANCE_SA1_CODE,
    CONCORDANCE_TA_CODE,
    DEFAULT_SPEC,
    generate_crosswalk,
)
from tools.fetch_nz_as_area_sources import JOINT_GEOGRAPHY_PROVENANCE, verify

ROOT = Path(__file__).resolve().parents[1]
NZ = ROOT / "packages/microcosm-build/src/microcosm/build/nz"
DEFAULT_OUT = NZ / "atomic_geography_support.json"
DEFAULT_REPORT = ROOT / "docs/nz-joint-geography-report.json"
REGION_CODE = "REGC2025_V1_00"
MESHBLOCK_CODE = "MB2026_V1_00"
ATOMIC_VINTAGE = "ta2026_regc2025_au2017"


def load_joint_sources(source_dir, provenance=None):
    """Verify every pinned page before reading its features; never trust receipts."""
    provenance = JOINT_GEOGRAPHY_PROVENANCE if provenance is None else provenance
    pages = provenance["pages"]
    names = [page["file"] for page in pages]
    if not pages or len(names) != len(set(names)):
        raise ValueError("Joint geography requires nonempty, unique pinned pages")
    payloads = {}
    for page in pages:
        payload = (Path(source_dir) / page["file"]).read_bytes()
        verify(page["file"], payload, page["sha256"], page["bytes"])
        payloads[page["file"]] = payload
    features = []
    for page in sorted(pages, key=lambda page: page["file"]):
        document = json.loads(payloads[page["file"]])
        if len(document["features"]) != page["features"]:
            raise ValueError(f"Feature count mismatch: {page['file']}")
        features.extend(document["features"])
    codes = [feature["attributes"][MESHBLOCK_CODE] for feature in features]
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate meshblock codes in joint-geography pages")
    return sorted(features, key=lambda feature: feature["attributes"][MESHBLOCK_CODE])


def joint_concordance(meshblock_features):
    """Map each official SA1 to one TA x REGC cell, retaining region 99."""
    by_sa1 = defaultdict(set)
    for feature in meshblock_features:
        attributes = feature["attributes"]
        sa1 = attributes[CONCORDANCE_SA1_CODE]
        ta = attributes[CONCORDANCE_TA_CODE]
        region = attributes[REGION_CODE]
        for value, width in ((sa1, 7), (ta, 3), (region, 2)):
            if not isinstance(value, str) or len(value) != width or not value.isdigit():
                raise ValueError("SA1, TA and REGC codes must be zero-padded digits")
        by_sa1[sa1].add((ta, region))
    multiple = {sa1: sorted(cells) for sa1, cells in by_sa1.items() if len(cells) != 1}
    if multiple:
        raise ValueError(f"SA1 with multiple TA x REGC cells: {multiple}")
    if not by_sa1:
        raise ValueError("Joint geography requires an SA1 concordance")
    return {
        sa1: {"ta_code": next(iter(cells))[0], "region_code": next(iter(cells))[1]}
        for sa1, cells in sorted(by_sa1.items())
    }


def _anchors(references):
    if isinstance(references, Mapping):
        references = references["target_references"]
    anchors = {}
    for reference in references:
        if (
            reference.get("metadata", {}).get("precal_use")
            != "atomic_geography_support"
        ):
            continue
        selector = reference["ledger_selector"]
        population_code = selector["geography_id"]
        ta = population_code.removeprefix("nz-ta-")
        if (
            selector["geography_level"] != "local_authority"
            or population_code != f"nz-ta-{ta}"
            or len(ta) != 3
            or not ta.isdigit()
            or ta == "999"
            or ta in anchors
        ):
            raise ValueError("Population anchors must be unique real TA hierarchy ids")
        anchors[ta] = {"target": reference["name"], "population_code": population_code}
    if not anchors:
        raise ValueError("No eligible TA population anchors")
    return anchors


def _definition(rows):
    source = "Stats NZ Meshblock Higher Geographies 2026"

    def code(source, vintage, relation):
        return {
            "kind": "code",
            "source": source,
            "vintage": vintage,
            "relation": relation,
        }

    return {
        "version": 1,
        "level": "ta_region_as_area",
        "code_system": "stats_nz_joint_ta_regc_as",
        "vintage": ATOMIC_VINTAGE,
        "columns": {
            "area": code(
                source + "; G8-m SA1 to AS placement", ATOMIC_VINTAGE, "exact"
            ),
            "ta_code": code(source, "ta_2026", "official_tabulation"),
            "region_code": code(source, "regc_2025", "official_tabulation"),
            "as_area": code(
                "G8-m SA1 interior-point/centroid AU2017 bridge", "au_2017", "best_fit"
            ),
            "population_code": code(
                "Stats NZ TA ERP target hierarchy", "ta_2025", "exact"
            ),
            "population": {
                "kind": "weight",
                "source": "Stats NZ TA ERP at 30 June 2025",
                "basis": "TA ERP apportioned by within-TA 2023 Census SA1 usually resident population shares",
            },
        },
        "rows": rows,
        "share_tolerance": 1e-12,
        "population_geography_level": "local_authority",
        "population_code_column": "population_code",
    }


def aggregate_support(placements, concordance, references, *, report=None):
    """Aggregate observed joint cells, never a product of geography marginals.

    Zero-population cells remain explicit. TA 999 is audit-only because the
    existing surface declares no outside-TA ERP target; its Census population
    is counted separately rather than assigned to a real TA anchor.
    """
    anchors = _anchors(references)
    counts, sa1_counts = Counter(), Counter()
    seen, outside_sa1s = set(), []
    for placement in sorted(placements, key=lambda row: row["sa1_code"]):
        sa1, ta = placement["sa1_code"], placement["ta_code"]
        population, as_area = placement["population"], placement["as_area"]
        if sa1 in seen:
            raise ValueError(f"Duplicate SA1 placement: {sa1}")
        seen.add(sa1)
        if sa1 not in concordance:
            raise ValueError(f"Missing SA1 in TA x REGC concordance: {sa1}")
        joint = concordance[sa1]
        if ta != joint["ta_code"]:
            raise ValueError(f"TA concordance disagreement for SA1 {sa1}")
        if ta != "999" and ta not in anchors:
            raise ValueError(f"No population anchor for real TA {ta}")
        if type(population) is not int or population < 0:
            raise ValueError("Usable SA1 population must be a nonnegative integer")
        if type(as_area) is not int or as_area not in (1, 2, 3, 4):
            raise ValueError("SA1 placement must declare AS area 1, 2, 3 or 4")
        cell = (ta, joint["region_code"], as_area)
        counts[cell] += population
        sa1_counts[cell] += 1
        if ta == "999":
            outside_sa1s.append({**placement, "region_code": joint["region_code"]})
    ta_population = Counter()
    for (ta, _, _), population in counts.items():
        ta_population[ta] += population
    missing = sorted(ta for ta in anchors if ta_population[ta] <= 0)
    if missing:
        raise ValueError(
            f"Population anchors require positive usable SA1 population: {missing}"
        )
    cells, outside_cells, rows = [], [], []
    for (ta, region, as_area), population in sorted(counts.items()):
        area = f"{ta if ta != '999' else 'outside-ta'}|{region}|{as_area}"
        cell = {
            "area": area,
            "ta_code": ta,
            "region_code": region,
            "as_area": as_area,
            "population": population,
            "sa1_count": sa1_counts[ta, region, as_area],
        }
        if ta == "999":
            outside_cells.append(cell)
            continue
        cells.append(cell)
        rows.append(
            {
                "target": anchors[ta]["target"],
                "codes": {
                    "area": area,
                    "ta_code": ta,
                    "region_code": region,
                    "as_area": str(as_area),
                    "population_code": anchors[ta]["population_code"],
                },
                "population_share": population / ta_population[ta],
            }
        )
    region_totals = defaultdict(lambda: Counter())
    ta_region_totals, ta_region_sa1s = Counter(), Counter()
    regions_by_ta = defaultdict(set)
    for (ta, region, as_area), population in counts.items():
        region_totals[region]["population"] += population
        key = "outside_ta_population" if ta == "999" else "anchored_population"
        region_totals[region][key] += population
        if ta != "999":
            regions_by_ta[ta].add(region)
            ta_region_totals[ta, region] += population
            ta_region_sa1s[ta, region] += sa1_counts[ta, region, as_area]
    if report is not None:
        report.clear()
        report.update(
            {
                "description": "Generated by tools/build_nz_joint_geography.py; Census joint-cell audit, not a transport activation",
                "population_basis": "Usable 2023 Census SA1 URP; suppressed/null observations contribute zero under the existing G8-m rule",
                "anchor_basis": f"Existing {len(anchors)} real TA ERP references at 30 June 2025; largest-remainder apportionment preserves each TA ERP exactly",
                "outside_ta_policy": "TA 999 has no declared ERP anchor; its own Census SA1s remain audit-only and are never reassigned to real TAs",
                "sa1_count": len(seen),
                "census_population": sum(counts.values()),
                "anchored_census_population": sum(cell["population"] for cell in cells),
                "outside_ta_population": sum(
                    cell["population"] for cell in outside_cells
                ),
                "atomic_cell_count": len(cells),
                "cells": cells,
                "outside_ta_cells": outside_cells,
                "outside_ta_sa1s": outside_sa1s,
                "ta_totals": [
                    {"ta_code": ta, **anchor, "population": ta_population[ta]}
                    for ta, anchor in sorted(anchors.items())
                ],
                "straddling_tas": [
                    {
                        "ta_code": ta,
                        "region_code": region,
                        "population": population,
                        "sa1_count": ta_region_sa1s[ta, region],
                    }
                    for (ta, region), population in sorted(ta_region_totals.items())
                    if len(regions_by_ta[ta]) > 1
                ],
                "region_totals": [
                    {
                        "region_code": region,
                        **{
                            key: totals[key]
                            for key in (
                                "population",
                                "anchored_population",
                                "outside_ta_population",
                            )
                        },
                    }
                    for region, totals in sorted(region_totals.items())
                ],
            }
        )
    return _definition(rows)


def generate_support(
    source_dir,
    joint_source_dir,
    *,
    spec=None,
    precal_references=None,
    provenance=None,
    report=None,
    placements=None,
):
    """Return executable support and fill optional aggregate/placement evidence."""
    if spec is None:
        spec = json.loads(DEFAULT_SPEC.read_text())
    if precal_references is None:
        precal_references = json.loads((NZ / "precal_references.json").read_text())
    features = load_joint_sources(joint_source_dir, provenance)
    concordance = joint_concordance(features)
    assigned = []
    generate_crosswalk(source_dir, spec, placements=assigned)
    document = aggregate_support(
        assigned, concordance, precal_references, report=report
    )
    if report is not None:
        report["meshblock_count"] = len(features)
        report["concordance_sa1_count"] = len(concordance)
        report["concordance_sa1s_without_census"] = len(concordance) - len(assigned)
        pins = JOINT_GEOGRAPHY_PROVENANCE if provenance is None else provenance
        report["source_provenance"] = {
            **pins,
            "pages": sorted(pins["pages"], key=lambda page: page["file"]),
        }
    if placements is not None:
        placements.clear()
        placements.extend(assigned)
    return document


def _serialized(document):
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=os.environ.get("NZ_AS_AREA_SOURCES")
    )
    parser.add_argument(
        "--joint-sources",
        type=Path,
        default=os.environ.get("NZ_JOINT_GEOGRAPHY_SOURCES"),
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--precal-references", type=Path, default=NZ / "precal_references.json"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Refuse generated-file drift without writing",
    )
    args = parser.parse_args(argv)
    if any(
        path is None or not path.is_dir() for path in (args.sources, args.joint_sources)
    ):
        parser.error(
            "Supply pinned local pages with --sources and --joint-sources or their NZ_*_SOURCES variables"
        )
    report = {}
    document = generate_support(
        args.sources,
        args.joint_sources,
        spec=json.loads(args.spec.read_text()),
        precal_references=json.loads(args.precal_references.read_text()),
        report=report,
    )
    for path, generated in ((args.out, document), (args.report, report)):
        payload = _serialized(generated)
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                parser.exit(1, f"Generated file drift: {path}\n")
        else:
            path.write_bytes(payload)
    print(
        f"{'Checked' if args.check else 'Generated'} {len(document['rows'])} joint atomic cells; "
        f"{report['anchored_census_population']} anchored Census residents; "
        f"{report['outside_ta_population']} audit-only outside-TA residents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
