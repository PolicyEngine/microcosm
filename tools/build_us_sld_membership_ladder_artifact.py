"""Build the US SLD membership-ladder artifact from primary Census sources.

Downloads (with the cache shared by the other ladder builders) the national
2024 State Legislative District block-equivalency files (NationalSLDU24 /
NationalSLDL24 — the boundary vintage the ACS 2020-2024 5-year SLD tables
tabulate on), the 119th Congressional District BEF, the 2020 tract-to-PUMA
relationship file, and the 2020 P.L. 94-171 geographic headers (block
populations); runs one national block pass; and writes one NPZ carrying the
two overlap tables :mod:`populace.build.us_runtime.sld_membership` draws
from — ``tract x (SLDU, SLDL)`` and ``(PUMA, CD, county) x (SLDU, SLDL)``.
No per-area files, the standing rule.

The artifact is self-checked by loading it back through
``load_us_sld_membership_ladder`` before the summary is written. Every
source's URL + SHA-256, the output SHA-256, and the national totals land in
the provenance JSON beside the NPZ.

Example:
    uv run python tools/build_us_sld_membership_ladder_artifact.py \
        --out build/us/us_sld_membership_ladder_2024.npz \
        --cache-dir ~/.cache/populace-us-geography

    # Smoke run over two states:
    uv run python tools/build_us_sld_membership_ladder_artifact.py \
        --out /tmp/sld_ladder_smoke.npz --states 49,31
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from populace.build.us_runtime.block_ladder_sources import (
    US_STATES,
    parse_national_cd_bef,
    parse_pl_geo_blocks,
)
from populace.build.us_runtime.puma_ladder_sources import (
    parse_tract_to_puma_relationship,
)
from populace.build.us_runtime.sld_membership import (
    assemble_us_sld_membership_ladder,
    load_us_sld_membership_ladder,
    parse_national_sld24_bef,
    write_membership_provenance,
)

SLD24_BEF_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/"
    "2025/2024-state-legislative-bef/{name}.zip"
)
SLD24_NATIONAL_MEMBERS = {
    "upper": ("sldu24", "NationalSLDU24.txt"),
    "lower": ("sldl24", "NationalSLDL24.txt"),
}
CD119_BEF_URL = (
    "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/"
    "2025/119-congressional-district-befs/cd119.zip"
)
CD119_NATIONAL_MEMBER = "NationalCD119.txt"
TRACT_TO_PUMA_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "2020_Census_Tract_to_2020_PUMA.txt"
)
PL94171_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/decennial/2020/data/"
    "01-Redistricting_File--PL_94-171/{dirname}/{usps_lower}2020.pl.zip"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the national US SLD membership-ladder NPZ artifact."
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
        "--provenance-json",
        type=Path,
        help="Path for the provenance summary. Defaults beside --out.",
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
        url,
        headers={"User-Agent": "populace-build (sld membership ladder)"},
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


def _state_filter(
    mapping: dict[int, str],
    allowed_state_fips: frozenset[str],
) -> dict[int, str]:
    return {
        block: value
        for block, value in mapping.items()
        if f"{block // 10**13:02d}" in allowed_state_fips
    }


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
    _log(f"Building US SLD membership ladder for {len(states)} state(s)")

    source_files: dict[str, dict[str, str]] = {}

    sld_by_block: dict[str, dict[int, str]] = {}
    for chamber, (name, member) in SLD24_NATIONAL_MEMBERS.items():
        url = SLD24_BEF_URL_TEMPLATE.format(name=name)
        archive = _download(url, args.cache_dir)
        source_files[f"sld24_{chamber}_bef"] = {
            "url": url,
            "sha256": _sha256(archive),
        }
        _log(f"  parsing national {member}")
        sld_by_block[chamber] = _state_filter(
            parse_national_sld24_bef(
                _zip_member_lines(archive, member),
                chamber=chamber,
            ),
            allowed_state_fips,
        )
        _log(f"  {member}: {len(sld_by_block[chamber]):,} assigned blocks")

    cd_zip = _download(CD119_BEF_URL, args.cache_dir)
    source_files["cd119_bef"] = {"url": CD119_BEF_URL, "sha256": _sha256(cd_zip)}
    _log("  parsing national CD119 BEF")
    cd_by_block = {
        block: cd
        for block, cd in parse_national_cd_bef(
            _zip_member_lines(cd_zip, CD119_NATIONAL_MEMBER)
        ).items()
        if f"{block // 10**13:02d}" in allowed_state_fips
    }

    tract_puma_txt = _download(TRACT_TO_PUMA_URL, args.cache_dir)
    source_files["tract_to_puma"] = {
        "url": TRACT_TO_PUMA_URL,
        "sha256": _sha256(tract_puma_txt),
    }
    _log("  parsing tract-to-PUMA relationship")
    tract_to_puma = parse_tract_to_puma_relationship(
        _text_lines(tract_puma_txt),
        allowed_state_fips=allowed_state_fips,
    )

    block_population: dict[int, int] = {}
    for fips, usps, dirname in states:
        _log(f"  state {fips} {usps}")
        pl_url = PL94171_URL_TEMPLATE.format(
            dirname=dirname,
            usps_lower=usps.lower(),
        )
        pl_zip = _download(pl_url, args.cache_dir)
        source_files[f"pl94171_{usps.lower()}"] = {
            "url": pl_url,
            "sha256": _sha256(pl_zip),
        }
        geo_member = f"{usps.lower()}geo2020.pl"
        block_population.update(
            parse_pl_geo_blocks(
                _zip_member_lines(pl_zip, geo_member),
                state_fips=fips,
            )
        )

    _log("  assembling overlap tables")
    arrays = assemble_us_sld_membership_ladder(
        block_population=block_population,
        sldu_by_block=sld_by_block["upper"],
        sldl_by_block=sld_by_block["lower"],
        cd_by_block=cd_by_block,
        tract_to_puma=tract_to_puma,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    ladder = load_us_sld_membership_ladder(args.out)

    totals = {
        "n_states": len(states),
        "n_tract_rows": int(len(ladder.tract_geoid)),
        "n_cell_rows": int(len(ladder.cell_puma)),
        "population": int(ladder.tract_population.sum()),
        "n_assigned_blocks": int(arrays["n_assigned_blocks"][0]),
        "n_blocks_without_cd": int(arrays["n_blocks_without_cd"][0]),
        "n_upper_districts": sum(
            len(codes) for codes in ladder.district_codes("upper").values()
        ),
        "n_lower_districts": sum(
            len(codes) for codes in ladder.district_codes("lower").values()
        ),
        "artifact_sha256": _sha256(args.out),
    }
    provenance_path = args.provenance_json or args.out.with_suffix(".provenance.json")
    write_membership_provenance(
        provenance_path,
        sources=source_files,
        totals=totals,
    )
    _log(
        f"Wrote {args.out} ({totals['n_tract_rows']:,} tract rows, "
        f"{totals['n_cell_rows']:,} cell rows, population "
        f"{totals['population']:,}); provenance at {provenance_path}"
    )


if __name__ == "__main__":
    main()
