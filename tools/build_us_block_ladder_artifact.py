"""Build the US block-ladder artifact from primary Census/OMB sources.

Downloads (with a local cache) the 2020 P.L. 94-171 geographic headers
(block populations), the 119th Congressional District block equivalency
file, the 2020 Block Assignment Files (SLDU/SLDL/INCPLACE_CDP), and the OMB
CBSA delineation workbook; joins them at 2020-tabulation-block grain; and
writes one national NPZ artifact whose embedded metadata records a vintage
and source per derived layer (``vintage_policy: error`` — the loader refuses
an artifact missing any of them). No per-area files, the standing rule.

The artifact is self-checked by loading it back through
``populace.build.us_runtime.load_us_block_ladder`` before the summary is
written, so a published ladder is by construction a loadable ladder.

Example:
    uv run python tools/build_us_block_ladder_artifact.py \
        --out build/us/us_block_ladder_2020.npz \
        --cache-dir ~/.cache/populace-us-geography

    # Smoke run over two small states:
    uv run python tools/build_us_block_ladder_artifact.py \
        --out /tmp/ladder_smoke.npz --states 10,50
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from populace.build.us_runtime import load_us_block_ladder
from populace.build.us_runtime.block_ladder_sources import (
    US_STATES,
    assemble_us_block_ladder,
    parse_baf_district_file,
    parse_baf_place_file,
    parse_cbsa_delineations,
    parse_national_cd_bef,
    parse_pl_geo_blocks,
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
CBSA_DELINEATIONS_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
    "reference-files/2023/delineation-files/list1_2023.xlsx"
)

LAYER_VINTAGES = {
    "congressional_district": {
        "vintage": "119th_congress",
        "source": "Census 119th Congressional District BEF (NationalCD119.txt)",
        "url": CD119_BEF_URL,
    },
    "sldu": {
        "vintage": "2020_baf",
        "source": (
            "Census 2020 Block Assignment Files, SLDU layer (legislative "
            "plans as of the 2020 P.L. 94-171 release)"
        ),
        "url": BAF2020_URL_TEMPLATE,
    },
    "sldl": {
        "vintage": "2020_baf",
        "source": (
            "Census 2020 Block Assignment Files, SLDL layer (legislative "
            "plans as of the 2020 P.L. 94-171 release)"
        ),
        "url": BAF2020_URL_TEMPLATE,
    },
    "place": {
        "vintage": "2020_census",
        "source": "Census 2020 Block Assignment Files, INCPLACE_CDP layer",
        "url": BAF2020_URL_TEMPLATE,
    },
    "cbsa": {
        "vintage": "omb_2023_delineations",
        "source": (
            "OMB Bulletin No. 23-01 CBSA delineation file (list1_2023.xlsx), "
            "county to CBSA"
        ),
        "url": CBSA_DELINEATIONS_URL,
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the national US block-ladder NPZ artifact."
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


_DOWNLOAD_MAX_ATTEMPTS = 8
_DOWNLOAD_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_DOWNLOAD_MAX_BACKOFF_SECONDS = 60.0


def _fetch(url: str) -> bytes:
    """Fetch ``url`` with a bounded timeout, retrying transient failures.

    The Census/OMB hosts intermittently return 5xx (e.g. a momentary 502 Bad
    Gateway) or drop the connection; without a retry a single blip aborts the
    whole 51-state build after minutes of work. Retries use exponential backoff
    and only cover transient conditions — a 404 or other permanent HTTPError
    still raises immediately.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "populace-build (block ladder artifact)"}
    )
    last_exc: Exception | None = None
    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in _DOWNLOAD_RETRY_STATUS:
                raise
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
        if attempt < _DOWNLOAD_MAX_ATTEMPTS:
            backoff = min(2.0 * (2 ** (attempt - 1)), _DOWNLOAD_MAX_BACKOFF_SECONDS)
            _log(
                f"  transient download failure ({last_exc}); "
                f"retry {attempt}/{_DOWNLOAD_MAX_ATTEMPTS - 1} in {backoff:.0f}s"
            )
            time.sleep(backoff)
    raise RuntimeError(
        f"Download failed after {_DOWNLOAD_MAX_ATTEMPTS} attempts: {url}"
    ) from last_exc


def _download(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / url.rsplit("/", 1)[-1]
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    _log(f"  downloading {url}")
    payload = _fetch(url)
    if not payload:
        raise RuntimeError(f"Empty download from {url}")
    # Write-then-rename: a crash mid-write must not leave a truncated file
    # the next run would treat as a valid cache hit.
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(destination)
    return destination


def _zip_member_lines(archive_path: Path, member: str) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            return io.TextIOWrapper(stream, encoding="latin-1").readlines()


def _optional_baf_layer(archive_path: Path, member: str) -> list[str] | None:
    with zipfile.ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            return None
    return _zip_member_lines(archive_path, member)


def _cbsa_rows(workbook_path: Path):
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "Parsing the OMB delineation workbook requires openpyxl "
            "(workspace dev group; `uv sync --all-packages`)."
        ) from exc
    workbook = openpyxl.load_workbook(workbook_path, read_only=True)
    worksheet = workbook.worksheets[0]
    return worksheet.iter_rows(values_only=True)


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
    _log(f"Building US block ladder for {len(states)} state(s)")

    source_files: dict[str, dict[str, str]] = {}

    cd_zip = _download(CD119_BEF_URL, args.cache_dir)
    source_files["cd119_bef"] = {"path": str(cd_zip), "sha256": _sha256(cd_zip)}
    _log("  parsing national CD119 BEF")
    cd_by_block = parse_national_cd_bef(
        _zip_member_lines(cd_zip, CD119_NATIONAL_MEMBER)
    )

    cbsa_xlsx = _download(CBSA_DELINEATIONS_URL, args.cache_dir)
    source_files["cbsa_delineations"] = {
        "path": str(cbsa_xlsx),
        "sha256": _sha256(cbsa_xlsx),
    }
    cbsa_by_county = parse_cbsa_delineations(_cbsa_rows(cbsa_xlsx))
    _log(f"  OMB delineations: {len(cbsa_by_county):,} counties in a CBSA")

    block_population: dict[int, int] = {}
    sldu_by_block: dict[int, str] = {}
    sldl_by_block: dict[int, str] = {}
    place_by_block: dict[int, int] = {}
    missing_layers: dict[str, list[str]] = {"SLDU": [], "SLDL": [], "INCPLACE_CDP": []}
    for fips, usps, dirname in states:
        _log(f"  state {fips} {usps}")
        pl_zip = _download(
            PL94171_URL_TEMPLATE.format(dirname=dirname, usps_lower=usps.lower()),
            args.cache_dir,
        )
        source_files[f"pl94171_{usps.lower()}"] = {
            "path": str(pl_zip),
            "sha256": _sha256(pl_zip),
        }
        geo_member = f"{usps.lower()}geo2020.pl"
        state_blocks = parse_pl_geo_blocks(
            _zip_member_lines(pl_zip, geo_member), state_fips=fips
        )
        block_population.update(state_blocks)

        baf_zip = _download(
            BAF2020_URL_TEMPLATE.format(fips=fips, usps=usps), args.cache_dir
        )
        source_files[f"baf2020_{usps.lower()}"] = {
            "path": str(baf_zip),
            "sha256": _sha256(baf_zip),
        }
        for layer, target in (
            ("SLDU", sldu_by_block),
            ("SLDL", sldl_by_block),
            ("INCPLACE_CDP", place_by_block),
        ):
            member = f"BlockAssign_ST{fips}_{usps}_{layer}.txt"
            lines = _optional_baf_layer(baf_zip, member)
            if lines is None:
                # A missing layer is a real geography fact (e.g. Nebraska's
                # unicameral legislature has no SLDL), recorded in the
                # summary; blocks stay unassigned for that layer.
                missing_layers[layer].append(usps)
                continue
            if layer == "INCPLACE_CDP":
                target.update(parse_baf_place_file(lines, label=member))
            else:
                target.update(parse_baf_district_file(lines, label=member))

    metadata = {
        "schema_version": 1,
        "kind": "us_block_ladder",
        "block_vintage": "2020_tabulation_blocks",
        "sampling_basis": "population",
        "layers": LAYER_VINTAGES,
        "states": [fips for fips, _, _ in states],
        "missing_baf_layers": {
            layer: sorted(values) for layer, values in missing_layers.items()
        },
        "source_files": source_files,
    }
    _log(f"  assembling {len(block_population):,} populated blocks")
    payload = assemble_us_block_ladder(
        block_population=block_population,
        cd_by_block=cd_by_block,
        sldu_by_block=sldu_by_block,
        sldl_by_block=sldl_by_block,
        place_by_block=place_by_block,
        cbsa_by_county=cbsa_by_county,
        metadata=metadata,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    ladder = load_us_block_ladder(args.out)  # self-check: must load cleanly
    summary = {
        "output": str(args.out.resolve()),
        "output_sha256": _sha256(args.out),
        "blocks": int(len(ladder)),
        "population": int(ladder.population.sum()),
        "congressional_districts": int(
            np.unique(ladder.congressional_district_geoid).size
        ),
        "counties": int(np.unique(ladder.block_geoid // 10**10).size),
        "layer_vintages": ladder.layer_vintages,
        "states": [fips for fips, _, _ in states],
        "missing_baf_layers": metadata["missing_baf_layers"],
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
