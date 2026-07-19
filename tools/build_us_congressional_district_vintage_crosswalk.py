"""Build the 117th->119th CD vintage crosswalk artifact from Census sources.

Downloads (with a local cache) the 119th Congressional District block
equivalency file, the 2020 Block Assignment File ``CD`` layer for each state,
and the 2020 P.L. 94-171 geographic headers (block populations); joins them at
2020-tabulation-block grain weighted by block population; and writes the
``source_geography_id,target_geography_id,weight`` crosswalk that
``congressional_district_vintage.translate_congressional_district_facts_to_current_vintage``
consumes, plus a sidecar provenance JSON recording every source file's SHA-256,
the build metadata, and per-state population conservation.

The method and rationale live in
``populace.build.us_runtime.congressional_district_vintage_crosswalk``. The
crosswalk is a regenerable build artifact, not a Ledger fact (the fact-vs-
computed boundary of PolicyEngine/ledger#71); it feeds the CD geography-vintage
translation that PolicyEngine/populace#205 requires.

Example:
    uv run --python 3.13 --package populace-build --group dev python \
        tools/build_us_congressional_district_vintage_crosswalk.py \
        --out packages/populace-build/src/populace/build/us_runtime/data/\
congressional_district_vintage_crosswalk.csv \
        --cache-dir ~/.cache/populace-us-geography

    # Smoke run over a few states:
    uv run python tools/build_us_congressional_district_vintage_crosswalk.py \
        --out /tmp/cd_xwalk_smoke.csv --states 08,30,11
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from populace.build.us_runtime.block_ladder_sources import (
    US_STATES,
    parse_pl_geo_blocks,
)
from populace.build.us_runtime.congressional_district_vintage import (
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
)
from populace.build.us_runtime.congressional_district_vintage_crosswalk import (
    build_cd_vintage_crosswalk_rows,
    parse_baf_cd_layer,
    parse_national_cd_bef_districts,
)

CD119_BEF_URL = (
    "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/"
    "2025/119-congressional-district-befs/cd119.zip"
)
CD119_NATIONAL_MEMBER = "NationalCD119.txt"
BAF2020_URL_TEMPLATE = (
    "https://www2.census.gov/geo/docs/maps-data/data/baf2020/"
    "BlockAssign_ST{fips}_{usps}.zip"
)
PL94171_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/decennial/2020/data/"
    "01-Redistricting_File--PL_94-171/{dirname}/{usps_lower}2020.pl.zip"
)

SOURCE_CONGRESSIONAL_DISTRICT_VINTAGE = "117th_congress"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 117th->119th CD vintage crosswalk CSV artifact."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "populace-us-geography",
        help="Download cache; re-runs reuse verified files.",
    )
    parser.add_argument(
        "--states",
        help=(
            "Optional comma-separated state FIPS subset (smoke runs only; a "
            "published crosswalk covers all 51)."
        ),
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        help="Path for the provenance sidecar. Defaults to --out with a "
        "'.provenance.json' suffix.",
    )
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / url.rsplit("/", 1)[-1]
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    _log(f"  downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "populace-build (cd vintage crosswalk)"}
    )
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Empty download from {url}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(destination)
    return destination


def _zip_member_lines(archive_path: Path, member: str) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            return io.TextIOWrapper(stream, encoding="latin-1").readlines()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    selected = (
        {value.strip() for value in args.states.split(",")} if args.states else None
    )
    states = [entry for entry in US_STATES if selected is None or entry[0] in selected]
    if selected is not None:
        unknown = selected - {entry[0] for entry in US_STATES}
        if unknown:
            raise SystemExit(f"Unknown state FIPS in --states: {sorted(unknown)}")
    _log(f"Building CD117->CD119 crosswalk for {len(states)} state(s)")

    source_files: dict[str, dict[str, str]] = {}

    cd_zip = _download(CD119_BEF_URL, args.cache_dir)
    source_files["cd119_bef"] = {
        "url": CD119_BEF_URL,
        "member": CD119_NATIONAL_MEMBER,
        "sha256": _sha256(cd_zip),
    }
    _log("  parsing national CD119 BEF (current vintage)")
    current_cd_by_block = parse_national_cd_bef_districts(
        _zip_member_lines(cd_zip, CD119_NATIONAL_MEMBER)
    )
    selected_fips = {fips for fips, _, _ in states}
    if selected is not None:
        current_cd_by_block = {
            block: district
            for block, district in current_cd_by_block.items()
            if f"{block:015d}"[:2] in selected_fips
        }

    old_cd_by_block: dict[int, str] = {}
    block_population: dict[int, int] = {}
    for fips, usps, dirname in states:
        _log(f"  state {fips} {usps}")
        baf_zip = _download(
            BAF2020_URL_TEMPLATE.format(fips=fips, usps=usps), args.cache_dir
        )
        source_files[f"baf2020_cd_{usps.lower()}"] = {
            "url": BAF2020_URL_TEMPLATE.format(fips=fips, usps=usps),
            "member": f"BlockAssign_ST{fips}_{usps}_CD.txt",
            "sha256": _sha256(baf_zip),
        }
        old_cd_by_block.update(
            parse_baf_cd_layer(
                _zip_member_lines(baf_zip, f"BlockAssign_ST{fips}_{usps}_CD.txt"),
                label=f"BlockAssign_ST{fips}_{usps}_CD.txt",
            )
        )

        pl_zip = _download(
            PL94171_URL_TEMPLATE.format(dirname=dirname, usps_lower=usps.lower()),
            args.cache_dir,
        )
        source_files[f"pl94171_{usps.lower()}"] = {
            "url": PL94171_URL_TEMPLATE.format(
                dirname=dirname, usps_lower=usps.lower()
            ),
            "member": f"{usps.lower()}geo2020.pl",
            "sha256": _sha256(pl_zip),
        }
        block_population.update(
            parse_pl_geo_blocks(
                _zip_member_lines(pl_zip, f"{usps.lower()}geo2020.pl"),
                state_fips=fips,
            )
        )

    rows, diagnostics = build_cd_vintage_crosswalk_rows(
        old_cd_by_block=old_cd_by_block,
        current_cd_by_block=current_cd_by_block,
        block_population=block_population,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "source_geography_id",
                "target_geography_id",
                "pair_population",
                "weight",
            ],
            lineterminator="\n",  # Unix newlines; RFC 4180 CRLF trips git checks.
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    crosswalk_sha256 = _sha256(args.out)

    provenance = {
        "schema_version": 1,
        "kind": "us_congressional_district_vintage_crosswalk",
        "source_geography_vintage": SOURCE_CONGRESSIONAL_DISTRICT_VINTAGE,
        "target_geography_vintage": CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
        "block_vintage": "2020_tabulation_blocks",
        "crosswalk_sha256": crosswalk_sha256,
        "method": (
            "Population-weighted overlay of the old (117th, via the 2020 BAF CD "
            "layer) and current (119th BEF) congressional-district assignments on "
            "2020 tabulation blocks, weighted by 2020 P.L. 94-171 block "
            "populations. Each old district's population is redistributed across "
            "current districts; totals conserve by construction."
        ),
        "sources": source_files,
        "diagnostics": diagnostics,
        "states": [fips for fips, _, _ in states],
    }
    provenance_path = args.provenance_json or args.out.with_suffix(
        args.out.suffix + ".provenance.json"
    )
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    _validate_conservation(diagnostics)

    _log(
        f"DONE {len(rows)} rows; "
        f"{diagnostics['source_district_count']} source -> "
        f"{diagnostics['target_district_count']} target districts; "
        f"crosswalk sha256 {crosswalk_sha256}"
    )
    _log(f"  crosswalk:  {args.out}")
    _log(f"  provenance: {provenance_path}")


def _validate_conservation(diagnostics: dict[str, object]) -> None:
    """Fail loudly if any state loses population beyond a small block-coverage gap.

    Populated blocks the BAF/BEF do not cover are a source-coverage gap, not a
    silent redistribution. A published crosswalk should cover essentially every
    populated block; a large gap means a source or field-layout drift.
    """

    state_conservation = diagnostics.get("state_conservation")
    if not isinstance(state_conservation, dict):
        raise ValueError("Crosswalk diagnostics missing state_conservation.")
    offenders: list[str] = []
    for state_fips, record in state_conservation.items():
        state_population = int(record["state_population"])
        unmatched = int(record["unmatched_population"])
        if state_population <= 0:
            continue
        if unmatched / state_population > 0.001:  # >0.1% of a state uncovered
            offenders.append(
                f"{state_fips}: {unmatched:,}/{state_population:,} "
                f"({unmatched / state_population:.2%}) uncovered"
            )
    if offenders:
        raise ValueError(
            "CD vintage crosswalk leaves too much population uncovered "
            "(possible source/layout drift): " + "; ".join(offenders)
        )


if __name__ == "__main__":
    main()
