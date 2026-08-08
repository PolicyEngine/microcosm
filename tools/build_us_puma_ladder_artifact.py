"""Build the US PUMA-ladder artifact from primary Census sources.

Downloads (with a local cache shared with the block-ladder builder) the 2020
Census Tract to 2020 PUMA relationship file, the 119th Congressional District
block equivalency file, and the 2020 P.L. 94-171 geographic headers (block
populations); joins them at 2020-tabulation-block grain and aggregates to the
PUMA anchor plus three ``(PUMA, layer_value, population)`` overlap tables; and
writes one national NPZ artifact whose embedded metadata records a vintage and
source per derived layer (``vintage_policy: error`` — the loader refuses an
artifact missing any of them, or one whose overlaps do not conserve each PUMA's
population). No per-area files, the standing rule.

Because 2020 blocks nest in 2020 tracts nest in 2020 PUMAs, the only source
this builder adds over the block ladder is the small tract-to-PUMA relationship
file; block populations and 119th-CD assignments reuse the block-ladder
parsers unchanged.

The artifact is self-checked by loading it back through
``microcosm.build.us_runtime.load_us_puma_ladder`` before the summary is
written, so a published ladder is by construction a loadable ladder. Every
source's URL + SHA-256, the output SHA-256, and the national conservation
totals are recorded in the summary JSON — the pinned-source manifest.

Example:
    uv run python tools/build_us_puma_ladder_artifact.py \
        --out build/us/us_puma_ladder_2020.npz \
        --cache-dir ~/.cache/populace-us-geography

    # Smoke run over two small states:
    uv run python tools/build_us_puma_ladder_artifact.py \
        --out /tmp/puma_ladder_smoke.npz --states 10,50
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from microcosm.build.us_runtime import load_us_puma_ladder
from microcosm.build.us_runtime.block_ladder_sources import (
    US_STATES,
    parse_national_cd_bef,
    parse_pl_geo_blocks,
)
from microcosm.build.us_runtime.puma_ladder_sources import (
    assemble_us_puma_ladder,
    parse_tract_to_puma_relationship,
)

TRACT_TO_PUMA_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "2020_Census_Tract_to_2020_PUMA.txt"
)
CD119_BEF_URL = (
    "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/"
    "2025/119-congressional-district-befs/cd119.zip"
)
CD119_NATIONAL_MEMBER = "NationalCD119.txt"
PL94171_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/decennial/2020/data/"
    "01-Redistricting_File--PL_94-171/{dirname}/{usps_lower}2020.pl.zip"
)

LAYER_VINTAGES = {
    "congressional_district": {
        "vintage": "119th_congress",
        "source": "Census 119th Congressional District BEF (NationalCD119.txt)",
        "url": CD119_BEF_URL,
    },
    "county": {
        "vintage": "2020_census",
        "source": (
            "Census 2020 Census Tract to 2020 PUMA relationship file "
            "(county is the tract geoid's structural prefix)"
        ),
        "url": TRACT_TO_PUMA_URL,
    },
    "tract": {
        "vintage": "2020_census",
        "source": (
            "Census 2020 Census Tract to 2020 PUMA relationship file, weighted "
            "by 2020 P.L. 94-171 block populations"
        ),
        "url": TRACT_TO_PUMA_URL,
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the national US PUMA-ladder NPZ artifact."
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
            "published ladder covers all 51)."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Path for the build summary. Defaults beside --out.",
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
        url, headers={"User-Agent": "microcosm-build (puma ladder artifact)"}
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


def _text_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig") as stream:
        return stream.readlines()


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
    allowed_state_fips = frozenset(fips for fips, _, _ in states)
    _log(f"Building US PUMA ladder for {len(states)} state(s)")

    source_files: dict[str, dict[str, str]] = {}

    tract_puma_txt = _download(TRACT_TO_PUMA_URL, args.cache_dir)
    source_files["tract_to_puma"] = {
        "url": TRACT_TO_PUMA_URL,
        "sha256": _sha256(tract_puma_txt),
    }
    _log("  parsing tract-to-PUMA relationship")
    tract_to_puma = parse_tract_to_puma_relationship(
        _text_lines(tract_puma_txt), allowed_state_fips=allowed_state_fips
    )
    _log(f"  tract-to-PUMA: {len(tract_to_puma):,} tracts")

    cd_zip = _download(CD119_BEF_URL, args.cache_dir)
    source_files["cd119_bef"] = {"url": CD119_BEF_URL, "sha256": _sha256(cd_zip)}
    _log("  parsing national CD119 BEF")
    cd_by_block = parse_national_cd_bef(
        _zip_member_lines(cd_zip, CD119_NATIONAL_MEMBER)
    )

    block_population: dict[int, int] = {}
    for fips, usps, dirname in states:
        _log(f"  state {fips} {usps}")
        pl_url = PL94171_URL_TEMPLATE.format(dirname=dirname, usps_lower=usps.lower())
        pl_zip = _download(pl_url, args.cache_dir)
        source_files[f"pl94171_{usps.lower()}"] = {
            "url": pl_url,
            "sha256": _sha256(pl_zip),
        }
        geo_member = f"{usps.lower()}geo2020.pl"
        state_blocks = parse_pl_geo_blocks(
            _zip_member_lines(pl_zip, geo_member), state_fips=fips
        )
        block_population.update(state_blocks)

    metadata = {
        "schema_version": 1,
        "kind": "us_puma_ladder",
        "puma_vintage": "2020_puma",
        "sampling_basis": "population",
        "layers": LAYER_VINTAGES,
        "states": [fips for fips, _, _ in states],
        "source_files": source_files,
    }
    _log(f"  aggregating {len(block_population):,} populated blocks")
    payload = assemble_us_puma_ladder(
        block_population=block_population,
        cd_by_block=cd_by_block,
        tract_to_puma=tract_to_puma,
        metadata=metadata,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    ladder = load_us_puma_ladder(args.out)  # self-check: must load cleanly
    summary = {
        "artifact": args.out.name,
        "output_sha256": _sha256(args.out),
        "pumas": int(len(ladder)),
        "population": int(ladder.puma_population.sum()),
        "congressional_district_overlaps": int(len(ladder.cd_overlap_puma)),
        "county_overlaps": int(len(ladder.county_overlap_puma)),
        "tract_overlaps": int(len(ladder.tract_overlap_puma)),
        "congressional_districts": int(np.unique(ladder.cd_overlap_cd).size),
        "counties": int(np.unique(ladder.county_overlap_county).size),
        "layer_vintages": ladder.layer_vintages,
        "states": [fips for fips, _, _ in states],
        "source_files": source_files,
    }
    summary_path = (
        args.summary_json
        if args.summary_json is not None
        else args.out.with_suffix(".summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
