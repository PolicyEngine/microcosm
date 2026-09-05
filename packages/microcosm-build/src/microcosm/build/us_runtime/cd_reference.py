"""Public CD references and uncertainty, without target or candidate activation.

This inventory retains published observations. Crosswalk shares describe alignment
exposure only; this module neither allocates fiscal cells nor rescales state totals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from .cd_reference_sources import (
    TABLES,
    capture_request,
    census_requests,
    read_census_key,
    strict_json,
    write_json,
)

US_STATE_FIPS = frozenset(
    "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 "
    "27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split()
)
AT_LARGE_117 = frozenset("02 10 11 30 38 46 50 56".split())
SENTINELS = {
    "-666666666": "insufficient_sample",
    "-999999999": "insufficient_geographic_sample",
    "-888888888": "not_applicable_or_available",
    "-222222222": "moe_insufficient_sample",
    "-333333333": "moe_open_interval",
    "-555555555": "controlled_estimate",
}
ANNOTATIONS = {
    "-": "insufficient_sample",
    "N": "insufficient_geographic_sample",
    "(X)": "not_applicable_or_available",
    "**": "moe_insufficient_sample",
    "***": "moe_open_interval",
    "*****": "controlled_estimate",
}
AUTHORITIES = {
    "acs_annotations": "https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-estimate-and-annotation-values.html",
    "acs_annotation_precedence": "https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-api-variable-types.html",
    "acs_geography": "https://www.census.gov/programs-surveys/acs/geography-acs/geography-boundaries-by-year/2024.html",
    "acs_confidence": "https://www.census.gov/programs-surveys/acs/methodology/sample-size-and-data-quality/sample-size-definitions.html",
    "irs_guide": "https://www.irs.gov/pub/irs-soi/22incddocguide.docx",
}


def classify_acs_value(raw, annotation) -> dict:
    """Annotation fields take precedence; retain sentinels and conflicts verbatim."""
    sentinel = SENTINELS.get(str(raw))
    annotation_class = None
    if annotation not in (None, ""):
        if not isinstance(annotation, str):
            raise ValueError("ACS annotation must be text or null")
        annotation_class = ANNOTATIONS.get(annotation)
        if annotation_class is None:
            annotation_class = (
                "open_interval"
                if re.fullmatch(r"[\d,.]+[+-]", annotation)
                else "unknown_annotation"
            )
    classification = annotation_class or sentinel
    numeric = None
    if classification is None:
        if raw in (None, ""):
            classification = "missing"
        else:
            try:
                numeric = float(raw)
                if not math.isfinite(numeric):
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError("unrecognized ACS numeric value") from None
            classification = "numeric"
    return {
        "raw": raw,
        "annotation": annotation,
        "class": classification,
        "numeric_value": numeric,
        "annotation_conflict": bool(
            sentinel and annotation_class and sentinel != annotation_class
        ),
    }


def _district_id(state: str, district: str, congress: int) -> str:
    if not re.fullmatch(r"\d{2}", state) or not re.fullmatch(r"\d{2}", district):
        raise ValueError("invalid district identity")
    return f"500{congress - 100:02d}00US{state}{'00' if district == '98' else district}"


def acs_table_inventory(table: str, metadata: dict, data_rows: list) -> dict:
    if table not in TABLES:
        raise ValueError("undeclared ACS table")
    variables = metadata["variables"]
    stems = sorted(
        key[:-1]
        for key in variables
        if key.startswith(table + "_") and key.endswith("E")
    )
    if not stems or not data_rows:
        raise ValueError("ACS table has no estimate variables or data")
    header = data_rows[0]
    if len(header) != len(set(header)):
        raise ValueError("duplicate ACS header")
    required = {"NAME", "state", "congressional district"}
    required.update(
        stem + suffix for stem in stems for suffix in ("E", "M", "EA", "MA")
    )
    missing = sorted(required - set(header))
    if missing:
        raise ValueError("missing ACS columns: " + ", ".join(missing))
    for stem in stems:
        if any(stem + suffix not in variables for suffix in ("E", "M", "EA", "MA")):
            raise ValueError("missing ACS variable metadata")
    cells, geographies, excluded = [], {}, []
    seen = set()
    counts = {"estimate": Counter(), "moe": Counter()}
    for values in data_rows[1:]:
        if len(values) != len(header):
            raise ValueError("ACS row width mismatch")
        row = dict(zip(header, values, strict=True))
        state, district = row["state"], row["congressional district"]
        identity = _district_id(state, district, 119)
        if identity in seen:
            raise ValueError("duplicate geography in ACS table")
        seen.add(identity)
        if state not in US_STATE_FIPS:
            excluded.append(
                {
                    "state_fips": state,
                    "district": district,
                    "name": row["NAME"],
                    "reason": "outside_us_50_states_dc",
                }
            )
            continue
        geographies[identity] = {
            "geography_id": identity,
            "state_fips": state,
            "published_district": district,
            "name": row["NAME"],
            "published_geo_id": row.get("GEO_ID"),
        }
        for stem in stems:
            estimate = classify_acs_value(row[stem + "E"], row[stem + "EA"])
            moe = classify_acs_value(row[stem + "M"], row[stem + "MA"])
            if moe["numeric_value"] is not None and moe["numeric_value"] < 0:
                raise ValueError("negative numeric ACS MOE")
            counts["estimate"][estimate["class"]] += 1
            counts["moe"][moe["class"]] += 1
            cells.append(
                {
                    "geography_id": identity,
                    "variable": stem,
                    "estimate": estimate,
                    "moe": moe,
                }
            )
    return {
        "table": table,
        "year": 2024,
        "congress": 119,
        "confidence_level": 0.9,
        "uncertainty": {
            "kind": "published_margin_of_error",
            "variance_conversion": "not_performed",
            "controlled_estimates": "raw sentinel retained; no invented variance",
        },
        "grouping": "subject_age_and_sex_published_groups"
        if table == "S0101"
        else "detailed_sex_by_age_published_groups"
        if table == "B01001"
        else "published_detailed_table_cells",
        "reservation": {
            "status": "reserved_same_source_holdout"
            if table in ("B19001", "B25003")
            else "inventory_only",
            "covers_derived_versions": True,
            "targets_allowed": False,
            "tuning_allowed": False,
            "source_independent": False,
            "reason": "ACS tabulations and ACS PUMS share a source; no candidate scoring or tuning here",
        },
        "lineage": "published",
        "geography_ids": sorted(geographies),
        "geographies": [geographies[key] for key in sorted(geographies)],
        "excluded_geographies": excluded,
        "variables": {
            stem: {
                suffix: variables[stem + suffix] for suffix in ("E", "M", "EA", "MA")
            }
            for stem in stems
        },
        "classification_counts": {key: dict(value) for key, value in counts.items()},
        "cells": sorted(
            cells, key=lambda cell: (cell["geography_id"], cell["variable"])
        ),
    }


def irs_csv_inventory(payload: bytes) -> dict:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    header = reader.fieldnames or []
    identity_fields = {"STATEFIPS", "STATE", "CONG_DISTRICT", "agi_stub"}
    if len(header) != len(set(header)) or not identity_fields.issubset(header):
        raise ValueError("IRS CSV identity columns missing or duplicated")
    metrics = [field for field in header if field not in identity_fields]
    counts = {field: Counter() for field in metrics}
    rows, seen = [], set()
    stubs = Counter()
    for row in reader:
        if None in row or None in row.values():
            raise ValueError("IRS row width mismatch")
        state, district, stub = row["STATEFIPS"], row["CONG_DISTRICT"], row["agi_stub"]
        identity = state, district, stub
        if identity in seen:
            raise ValueError("duplicate IRS source row")
        seen.add(identity)
        _district_id(state, district, 117)
        if state not in US_STATE_FIPS | {"00"} or not re.fullmatch(r"\d+", stub):
            raise ValueError("IRS source row outside declared scope")
        if state == "00" and district != "00":
            raise ValueError("national IRS row has district code")
        level = (
            "national"
            if state == "00"
            else "state"
            if district == "00"
            else "congressional_district"
        )
        rows.append(
            {
                "state_fips": state,
                "state": row["STATE"],
                "district": district,
                "agi_stub": stub,
                "published_geography_level": level,
                "published_geography_id": _district_id(state, district, 117)
                if level == "congressional_district"
                else None,
                "at_large_proxy_geography_id": _district_id(state, "00", 117)
                if state in AT_LARGE_117 and district == "00"
                else None,
            }
        )
        stubs[stub] += 1
        for field in metrics:
            raw = row[field]
            if raw in ("", "**"):
                classification = (
                    "missing" if raw == "" else "suppression_or_combination_marker"
                )
            else:
                try:
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError
                except ValueError:
                    raise ValueError("unrecognized IRS metric cell") from None
                classification = (
                    "numeric_zero_disclosure_unknown"
                    if value == 0
                    else "numeric_disclosure_unknown"
                )
            counts[field][classification] += 1
    return {
        "tax_year": 2022,
        "processing_year": 2023,
        "congress": 117,
        "lineage": "published",
        "rows": rows,
        "row_count": len(rows),
        "columns": header,
        "metric_classification_counts": {
            field: dict(count) for field, count in counts.items()
        },
        "agi_stub_inventory": {
            "counts": dict(sorted(stubs.items())),
            "guide_documented_codes": list(range(11)),
            "labels_status": "needs_method_resolution",
            "note": "Retain published codes. Guide section G describes 0–10; fetched CSV contains 0–9. Do not infer income-bin labels.",
        },
        "uncertainty": {
            "variance_status": "not_published_needs_method_resolution",
            "confidence_level": None,
            "note": "Guide section C describes a population; footnote 1 identifies an SOI sample input. No variance is inferred.",
        },
        "disclosure": {
            "numeric_zero_proves_unsuppressed": False,
            "csv_cell_suppression_status": "not_identifiable_from_numeric_cells",
            "published_rules": [
                "fewer than 20 returns: combine adjacent AGI cells",
                "district item totals under 20 excluded",
                "dominant single-return items suppressed using unpublished threshold",
            ],
            "xlsx_markers": "kept separately by sheet and cell; not joined to CSV AGI slices",
        },
        "units": {
            "monetary_amounts": "thousands_of_dollars",
            "rounding_increment": None,
            "rounding_status": "not_established_by_unit_label",
        },
        "coverage_caveats": [
            "tax filers only",
            "tax address may differ from residence",
            "ZIP-based district assignment",
            "US national total is not a congressional district",
            "at-large state totals are explicit proxies, not directly published district observations",
        ],
    }


def crosswalk_alignment(rows: list[dict]) -> dict:
    """Summarize incoming target composition, not fiscal allocation error."""
    incoming, outgoing = defaultdict(list), defaultdict(list)
    seen = set()
    for row in rows:
        source, target = row["source_geography_id"], row["target_geography_id"]
        if not re.fullmatch(r"5001700US\d{4}", source) or not re.fullmatch(
            r"5001900US\d{4}", target
        ):
            raise ValueError("crosswalk geography vintage mismatch")
        if source[9:11] not in US_STATE_FIPS or source[9:11] != target[9:11]:
            raise ValueError("crosswalk outside US50+DC or crosses state")
        if (source, target) in seen:
            raise ValueError("duplicate crosswalk pair")
        seen.add((source, target))
        population, weight = float(row["pair_population"]), float(row["weight"])
        if (
            not math.isfinite(population)
            or population <= 0
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(
                "crosswalk population and weight must be positive and finite"
            )
        incoming[target].append((source, population))
        outgoing[source].append((population, weight))
    if not seen:
        raise ValueError("empty crosswalk")
    for pairs in outgoing.values():
        total = math.fsum(population for population, _ in pairs)
        if not math.isfinite(total) or any(
            not math.isclose(weight, population / total, rel_tol=1e-9, abs_tol=1e-12)
            for population, weight in pairs
        ):
            raise ValueError("crosswalk outgoing weights disagree with pair population")
    targets = []
    for target, pairs in sorted(incoming.items()):
        total = math.fsum(population for _, population in pairs)
        contributions = [
            {
                "source_geography_id": source,
                "pair_population": population,
                "incoming_share": population / total,
            }
            for source, population in sorted(pairs)
        ]
        maximum = max(item["incoming_share"] for item in contributions)
        targets.append(
            {
                "target_geography_id": target,
                "pair_population": total,
                "contributions": contributions,
                "max_incoming_share": maximum,
                "non_dominant_incoming_share": 1 - maximum,
                "known_allocation_error": None,
            }
        )
    return {
        "source_congress": 117,
        "target_congress": 119,
        "population_vintage": "2020_tabulation_blocks",
        "basis": "pair_population_normalized_per_target",
        "interpretation": "alignment exposure, not known allocation error",
        "targets": targets,
    }


def xlsx_disclosure_inventory(payload: bytes) -> dict:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(item.itertext()) for item in shared.findall("x:si", ns)]
        cells = []
        for name in sorted(archive.namelist()):
            if not re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
                continue
            root = ET.fromstring(archive.read(name))
            for cell in root.findall(".//x:c", ns):
                if cell.attrib.get("t") == "s":
                    value = cell.findtext("x:v", namespaces=ns)
                    text = strings[int(value)] if value is not None else ""
                else:
                    text = (
                        "".join(cell.find("x:is", ns).itertext())
                        if cell.find("x:is", ns) is not None
                        else cell.findtext("x:v", default="", namespaces=ns)
                    )
                if "**" in text:
                    cells.append(
                        {
                            "sheet_part": name,
                            "cell": cell.attrib["r"],
                            "text": text,
                            "class": "suppression_or_combination_marker"
                            if text.strip() == "**"
                            else "disclosure_note",
                        }
                    )
    return {
        "markers": cells,
        "csv_cell_mapping": "not_performed",
        "note": "Published workbook cells and disclosure notes remain separate from CSV AGI rows.",
    }


def _pin_local(output: Path, payload: bytes, suffix: str, **metadata) -> dict:
    digest = hashlib.sha256(payload).hexdigest()
    relative = f"raw/{digest}{suffix}"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("local source digest collision")
    if not path.exists():
        path.write_bytes(payload)
    return {**metadata, "sha256": digest, "size_bytes": len(payload), "path": relative}


def build_bundle(
    output: Path,
    irs_authority: Path,
    crosswalk: Path,
    crosswalk_provenance: Path,
    *,
    offline: bool = False,
    api_key: str | None = None,
    fetch=None,
) -> dict:
    """Create or replay an immutable inventory. No candidate values are accepted."""
    output = Path(output)
    captures, tables = [], {}
    for request in census_requests():
        descriptor = capture_request(
            output,
            request,
            offline=offline,
            api_key=api_key,
            **({"fetch": fetch} if fetch else {}),
        )
        captures.append(descriptor)
        tables.setdefault(request["table"], {})[request["kind"]] = strict_json(
            (output / descriptor["path"]).read_bytes()
        )
    acs = {
        table: acs_table_inventory(
            table, tables[table]["metadata"], tables[table]["data"]
        )
        for table in TABLES
    }
    expected = acs["S0101"]["geography_ids"]
    if any(item["geography_ids"] != expected for item in acs.values()):
        raise ValueError("ACS table geography universes differ")
    index_payload = (irs_authority / "source-index.json").read_bytes()
    index = strict_json(index_payload)
    by_name = {item["filename"]: item for item in index}
    if len(by_name) != len(index):
        raise ValueError("duplicate IRS authority source identity")
    local, payloads = [], {}
    for name in ("22incd.csv", "22incdall.xlsx"):
        descriptor = by_name[name]
        if descriptor["url"] != "https://www.irs.gov/pub/irs-soi/" + name:
            raise ValueError("IRS authority URL mismatch")
        payload = (irs_authority / name).read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
            or len(payload) != descriptor["size_bytes"]
        ):
            raise ValueError("IRS authority source digest mismatch")
        payloads[name] = payload
        local.append(
            _pin_local(
                output, payload, Path(name).suffix, url=descriptor["url"], filename=name
            )
        )
    guide = (irs_authority / "22incddocguide.docx").read_bytes()
    if (
        hashlib.sha256(guide).hexdigest()
        != (irs_authority / "22incddocguide.sha256").read_text().split()[0]
    ):
        raise ValueError("IRS guide digest mismatch")
    local.append(
        _pin_local(
            output,
            guide,
            ".docx",
            url=AUTHORITIES["irs_guide"],
            filename="22incddocguide.docx",
        )
    )
    local.append(_pin_local(output, index_payload, ".json", role="irs_source_index"))
    with zipfile.ZipFile(io.BytesIO(guide)) as archive:
        guide_xml = archive.read("word/document.xml")
    guide_text = "\n".join(ET.fromstring(guide_xml).itertext()).encode()
    local.append(
        _pin_local(
            output,
            guide_text,
            ".txt",
            role="irs_guide_extracted_text",
            derived_from_sha256=hashlib.sha256(guide).hexdigest(),
            derivation="word/document.xml text nodes in document order",
        )
    )
    crosswalk_bytes, provenance_bytes = (
        crosswalk.read_bytes(),
        crosswalk_provenance.read_bytes(),
    )
    provenance = strict_json(provenance_bytes)
    if any(
        provenance.get(key) != value
        for key, value in {
            "source_geography_vintage": "117th_congress",
            "target_geography_vintage": "119th_congress",
            "block_vintage": "2020_tabulation_blocks",
        }.items()
    ):
        raise ValueError("crosswalk provenance vintage mismatch")
    if hashlib.sha256(crosswalk_bytes).hexdigest() != provenance["crosswalk_sha256"]:
        raise ValueError("crosswalk provenance digest mismatch")
    local.append(_pin_local(output, crosswalk_bytes, ".csv", role="crosswalk"))
    local.append(
        _pin_local(output, provenance_bytes, ".json", role="crosswalk_provenance")
    )
    alignment = crosswalk_alignment(
        list(csv.DictReader(io.StringIO(crosswalk_bytes.decode())))
    )
    irs = irs_csv_inventory(payloads["22incd.csv"])
    irs["workbook_disclosure"] = xlsx_disclosure_inventory(payloads["22incdall.xlsx"])
    published = sorted(
        {
            row["published_geography_id"]
            for row in irs["rows"]
            if row["published_geography_id"]
        }
    )
    proxies = sorted(
        {
            row["at_large_proxy_geography_id"]
            for row in irs["rows"]
            if row["at_large_proxy_geography_id"]
        }
    )
    crosswalk_sources = {
        item["source_geography_id"]
        for target in alignment["targets"]
        for item in target["contributions"]
    }
    targets = {item["target_geography_id"] for item in alignment["targets"]}
    bundle = {
        "schema_version": 1,
        "kind": "us_cd_reference_inventory",
        "purpose": "reference_inventory_only",
        "candidate_evaluation": "not_performed",
        "target_activation": False,
        "authorities": AUTHORITIES,
        "scope": {
            "country": "us",
            "states": "50_states_and_dc",
            "puerto_rico": "excluded",
            "acs_year": 2024,
            "target_congress": 119,
            "irs_tax_year": 2022,
            "irs_congress": 117,
        },
        "sources": {"census": captures, "local": local},
        "acs": acs,
        "irs": irs,
        "alignment": alignment,
        "reconciliation": {
            "irs_published_district_ids": published,
            "irs_at_large_state_proxy_ids": proxies,
            "irs_missing_crosswalk_source_ids": sorted(
                crosswalk_sources - set(published) - set(proxies)
            ),
            "irs_unmapped_source_ids": sorted(
                (set(published) | set(proxies)) - crosswalk_sources
            ),
            "acs_missing_crosswalk_target_ids": sorted(targets - set(expected)),
            "acs_unmapped_target_ids": sorted(set(expected) - targets),
        },
        "lineage_products": {
            "published": "included",
            "crosswalked_values": "not_generated",
            "state_rescaled_values": "not_generated",
        },
    }
    write_json(output / "reference-inventory.json", bundle)
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--irs-authority-dir", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--crosswalk-provenance", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--census-key-service")
    parser.add_argument("--census-key-account")
    args = parser.parse_args(argv)
    if bool(args.census_key_service) != bool(args.census_key_account):
        parser.error("credential service and account must be supplied together")
    try:
        key = (
            read_census_key(args.census_key_service, args.census_key_account)
            if args.census_key_service and not args.offline
            else None
        )
        bundle = build_bundle(
            args.output_dir,
            args.irs_authority_dir,
            args.crosswalk,
            args.crosswalk_provenance,
            offline=args.offline,
            api_key=key,
        )
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        print(f"Reference bundle failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Reference inventory saved: {args.output_dir / 'reference-inventory.json'}; {len(bundle['acs']['S0101']['geography_ids'])} ACS districts; no candidate evaluation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
