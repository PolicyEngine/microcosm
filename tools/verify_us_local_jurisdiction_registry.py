#!/usr/bin/env python3
"""Verify the US local-jurisdiction registry against Census sources.

Checks the registry (``us_local_jurisdictions.json``) three ways:

- ``--gazetteer-place`` / ``--gazetteer-county``: each registry place and
  county appears in the 2020 Gazetteer with the recorded GEOID, ANSI code,
  name and land/water area, the zip matches the SHA-256 the registry records,
  and a county-rule jurisdiction's county and coextensive city have the same
  land and water area;
- ``--pl-populations``: the registry's ``population_2020`` equals the 2020
  P.L. 94-171 ``P1_001N`` the Census API returns for the place (and, for a
  county rule, the county). The file holds one API response per line;
- ``--block-ladder`` (repeatable): every flag derived from the ladder's blocks
  covers exactly the registry population, and a county-rule flag covers the
  same blocks as its coextensive city
  (:func:`check_us_local_jurisdictions_against_block_ladder`). A ladder the
  registry already records must reproduce the recorded counts.

``--record`` writes each ladder's counts into ``block_ladder_checks`` and
prints the new registry SHA-256, which must then be pinned as
``US_LOCAL_JURISDICTION_REGISTRY_SHA256``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm.build.us_runtime.geography_ladder import load_us_block_ladder
from microcosm.build.us_runtime.local_jurisdictions import (
    JURISDICTION_RUNG_COUNTY,
    US_LOCAL_JURISDICTION_REGISTRY_RESOURCE,
    UsLocalJurisdictionRegistry,
    check_us_local_jurisdictions_against_block_ladder,
    load_us_local_jurisdiction_registry,
)

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "packages/microcosm-build/src/microcosm/build/us_runtime"
    / US_LOCAL_JURISDICTION_REGISTRY_RESOURCE
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gazetteer_rows(path: Path) -> dict[str, dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        (member,) = archive.namelist()
        text = archive.read(member).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows: dict[str, dict[str, str]] = {}
    for row in reader:
        clean = {key.strip(): (value or "").strip() for key, value in row.items()}
        rows[clean["GEOID"]] = clean
    return rows


def check_gazetteer(
    document: Mapping[str, Any],
    *,
    place_zip: Path,
    county_zip: Path,
) -> list[str]:
    """Return failures comparing registry gazetteer records to the zips."""

    failures: list[str] = []
    sources = document["sources"]
    for label, path in (("gazetteer_place", place_zip), ("gazetteer_county", county_zip)):
        observed = _sha256(path)
        if observed != sources[label]["sha256"]:
            failures.append(
                f"{label}: zip SHA-256 {observed} differs from the registry's "
                f"{sources[label]['sha256']}"
            )
    tables = {"place": _gazetteer_rows(place_zip), "county": _gazetteer_rows(county_zip)}
    for item in document["jurisdictions"]:
        records = item["gazetteer"]
        for level, record in records.items():
            row = tables[level].get(record["geoid"])
            if row is None:
                failures.append(f"{item['flag']}: {level} {record['geoid']} not in gazetteer")
                continue
            for key, column in (
                ("ansicode", "ANSICODE"),
                ("name", "NAME"),
                ("aland", "ALAND"),
                ("awater", "AWATER"),
            ):
                if str(record[key]) != row[column]:
                    failures.append(
                        f"{item['flag']}: {level} {record['geoid']} {column} is "
                        f"{row[column]!r}, registry records {record[key]!r}"
                    )
        if item["rule"] == JURISDICTION_RUNG_COUNTY:
            county = records.get("county", {})
            place = records.get("place", {})
            if county.get("geoid") != item["county_fips"]:
                failures.append(f"{item['flag']}: gazetteer county is not county_fips")
            if place.get("geoid") != item["state_fips"] + str(
                item["coextensive_place_fips"]
            ):
                failures.append(f"{item['flag']}: gazetteer place is not the coextensive city")
            if (county.get("aland"), county.get("awater")) != (
                place.get("aland"),
                place.get("awater"),
            ):
                failures.append(
                    f"{item['flag']}: county and city land/water areas differ, so "
                    "they are not the same area"
                )
        elif records.get("place", {}).get("geoid") != item["state_fips"] + item["place_fips"]:
            failures.append(f"{item['flag']}: gazetteer place is not state_fips+place_fips")
    return failures


def check_pl_populations(document: Mapping[str, Any], *, path: Path) -> list[str]:
    """Return failures comparing ``population_2020`` to P.L. 94-171 counts."""

    counts: dict[tuple[str, str], int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        header, *rows = json.loads(line)
        for row in rows:
            record = dict(zip(header, row, strict=True))
            if "place" in record:
                counts[("place", record["state"] + record["place"])] = int(record["P1_001N"])
            elif "county" in record:
                counts[("county", record["state"] + record["county"])] = int(
                    record["P1_001N"]
                )
            elif "us" in record:
                counts[("us", "1")] = int(record["P1_001N"])
    failures: list[str] = []
    national = document["sources"]["population"]["national_total_2020"]
    if counts.get(("us", "1")) not in (None, national):
        failures.append(f"US total {counts[('us', '1')]} differs from the registry's {national}")
    for item in document["jurisdictions"]:
        if item["rule"] == JURISDICTION_RUNG_COUNTY:
            keys = [
                ("county", item["county_fips"]),
                ("place", item["state_fips"] + item["coextensive_place_fips"]),
            ]
        else:
            keys = [("place", item["state_fips"] + item["place_fips"])]
        for key in keys:
            if key not in counts:
                failures.append(f"{item['flag']}: no P1_001N row for {key[0]} {key[1]}")
            elif counts[key] != item["population_2020"]:
                failures.append(
                    f"{item['flag']}: P1_001N for {key[0]} {key[1]} is {counts[key]:,}, "
                    f"registry records {item['population_2020']:,}"
                )
    return failures


def check_block_ladder(
    registry: UsLocalJurisdictionRegistry,
    document: Mapping[str, Any],
    *,
    path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Run the block-ladder check and compare to any recorded entry."""

    ladder_sha256 = _sha256(path)
    ladder = load_us_block_ladder(path)
    report = check_us_local_jurisdictions_against_block_ladder(
        ladder.block_geoid,
        ladder.population,
        ladder.place_fips,
        registry=registry,
    )
    entry = {
        "block_ladder_sha256": ladder_sha256,
        "blocks": int(len(ladder)),
        "population": int(ladder.population.sum()),
        "jurisdictions": report,
    }
    failures: list[str] = []
    national = document["sources"]["population"]["national_total_2020"]
    if entry["population"] != national:
        failures.append(
            f"ladder {ladder_sha256}: total population {entry['population']:,} "
            f"differs from the 2020 national total {national:,}"
        )
    for recorded in document.get("block_ladder_checks", []):
        if recorded["block_ladder_sha256"] == ladder_sha256 and recorded != entry:
            failures.append(
                f"ladder {ladder_sha256}: counts differ from the recorded check"
            )
    return entry, failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--gazetteer-place", type=Path)
    parser.add_argument("--gazetteer-county", type=Path)
    parser.add_argument("--pl-populations", type=Path)
    parser.add_argument("--block-ladder", type=Path, action="append", default=[])
    parser.add_argument(
        "--record",
        action="store_true",
        help="write each ladder's counts into block_ladder_checks",
    )
    args = parser.parse_args(argv)
    if (args.gazetteer_place is None) != (args.gazetteer_county is None):
        parser.error("--gazetteer-place and --gazetteer-county go together")

    document = json.loads(args.registry.read_text())
    registry = load_us_local_jurisdiction_registry(args.registry)
    failures: list[str] = []
    if args.gazetteer_place is not None:
        failures += check_gazetteer(
            document,
            place_zip=args.gazetteer_place,
            county_zip=args.gazetteer_county,
        )
        print("gazetteer: checked", file=sys.stderr)
    if args.pl_populations is not None:
        failures += check_pl_populations(document, path=args.pl_populations)
        print("P.L. 94-171 populations: checked", file=sys.stderr)
    entries = []
    for path in args.block_ladder:
        entry, ladder_failures = check_block_ladder(registry, document, path=path)
        failures += ladder_failures
        entries.append(entry)
        print(json.dumps(entry, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    if args.record and entries:
        recorded = {
            item["block_ladder_sha256"]: item
            for item in document.get("block_ladder_checks", [])
        }
        recorded.update({item["block_ladder_sha256"]: item for item in entries})
        document["block_ladder_checks"] = sorted(
            recorded.values(), key=lambda item: item["block_ladder_sha256"]
        )
        args.registry.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
        print(
            f"recorded; pin US_LOCAL_JURISDICTION_REGISTRY_SHA256 = "
            f"{_sha256(args.registry)}",
            file=sys.stderr,
        )
    print("OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
