"""Build the US PUMA-ladder artifact from primary Census sources.

Downloads (with a local cache shared with the block-ladder builder) the 2020
Census Tract to 2020 PUMA relationship file, the 119th Congressional District
block equivalency file, and the 2020 P.L. 94-171 geographic headers (block
populations); joins them at 2020-tabulation-block grain and aggregates to the
PUMA anchor, actual joint PUMA/tract/CD cells, and three marginal overlap tables; and
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
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from microcosm.build.us_runtime import (
    US_PUMA_LADDER_SCHEMA_VERSION,
    load_us_puma_ladder,
)
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
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help=(
            "Use only exact local inputs from a schema_version=1 JSON manifest "
            "with sources keyed by tract_to_puma, cd119_bef and pl94171_<state>. "
            "Each row requires path, url, sha256 and bytes. Disables downloads."
        ),
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


def _zip_member_lines(archive_path: Path, member: str) -> Iterator[str]:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            yield from io.TextIOWrapper(stream, encoding="latin-1")


def _text_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8-sig") as stream:
        yield from stream


def _pinned_sources(path: Path, urls: dict[str, str]) -> tuple[dict[str, dict], str]:
    """Validate the input envelope without downloading or reading source payloads."""

    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Pinned source manifest requires schema_version 1.")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(urls):
        raise ValueError("Pinned source manifest must match the exact selected inputs.")
    for name, url in urls.items():
        row = sources[name]
        if not isinstance(row, dict) or set(row) != {"path", "url", "sha256", "bytes"}:
            raise ValueError(f"Pinned source {name} requires path/url/sha256/bytes.")
        if (
            not isinstance(row["path"], str)
            or not Path(row["path"]).is_absolute()
            or row["url"] != url
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["sha256"])
        ):
            raise ValueError(f"Invalid pinned source envelope for {name}.")
    return sources, hashlib.sha256(raw).hexdigest()


def _verify_pinned_source(name: str, row: dict) -> Path:
    path = Path(row["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Pinned source {name} must be a regular local file.")
    if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
        raise ValueError(f"Pinned source identity mismatch for {name}.")
    return path


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

    urls = {"tract_to_puma": TRACT_TO_PUMA_URL, "cd119_bef": CD119_BEF_URL}
    urls.update(
        {
            f"pl94171_{usps.lower()}": PL94171_URL_TEMPLATE.format(
                dirname=dirname, usps_lower=usps.lower()
            )
            for _, usps, dirname in states
        }
    )
    pinned, manifest_sha256 = (
        _pinned_sources(args.source_manifest, urls)
        if args.source_manifest
        else (None, None)
    )

    def source_path(name: str) -> Path:
        if pinned is not None:
            return _verify_pinned_source(name, pinned[name])
        return _download(urls[name], args.cache_dir)

    source_files: dict[str, dict[str, str]] = {}

    tract_puma_txt = source_path("tract_to_puma")
    source_files["tract_to_puma"] = {
        "url": TRACT_TO_PUMA_URL,
        "sha256": _sha256(tract_puma_txt),
    }
    _log("  parsing tract-to-PUMA relationship")
    tract_to_puma = parse_tract_to_puma_relationship(
        _text_lines(tract_puma_txt), allowed_state_fips=allowed_state_fips
    )
    _log(f"  tract-to-PUMA: {len(tract_to_puma):,} tracts")

    cd_zip = source_path("cd119_bef")
    source_files["cd119_bef"] = {"url": CD119_BEF_URL, "sha256": _sha256(cd_zip)}
    _log("  parsing national CD119 BEF")
    cd_by_block = parse_national_cd_bef(
        _zip_member_lines(cd_zip, CD119_NATIONAL_MEMBER)
    )

    block_population: dict[int, int] = {}
    for fips, usps, dirname in states:
        _log(f"  state {fips} {usps}")
        pl_url = PL94171_URL_TEMPLATE.format(dirname=dirname, usps_lower=usps.lower())
        pl_zip = source_path(f"pl94171_{usps.lower()}")
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
        "schema_version": US_PUMA_LADDER_SCHEMA_VERSION,
        "kind": "us_puma_ladder",
        "puma_vintage": "2020_puma",
        "sampling_basis": "population",
        "layers": LAYER_VINTAGES,
        "states": [fips for fips, _, _ in states],
        "source_files": source_files,
    }
    if pinned is not None:
        for name, row in pinned.items():
            _verify_pinned_source(name, row)
        if _sha256(args.source_manifest) != manifest_sha256:
            raise ValueError("Pinned source manifest changed during source parsing.")
        metadata["source_manifest_sha256"] = manifest_sha256
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
        "joint_tract_cd_overlaps": int(len(ladder.joint_overlap_puma)),
        "congressional_districts": int(np.unique(ladder.cd_overlap_cd).size),
        "counties": int(np.unique(ladder.county_overlap_county).size),
        "layer_vintages": ladder.layer_vintages,
        "states": [fips for fips, _, _ in states],
        "source_files": source_files,
    }
    if manifest_sha256 is not None:
        summary["source_manifest_sha256"] = manifest_sha256
    summary_path = (
        args.summary_json
        if args.summary_json is not None
        else args.out.with_suffix(".summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
