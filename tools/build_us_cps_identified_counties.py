"""Build the packaged CPS ASEC identified-county list from Census documentation.

For each ASEC year, downloads (with a local cache) the technical documentation
``cpsmar{yy}.pdf``, renders it with ``pdftotext -layout`` (poppler), parses
"List 4: FIPS County Codes" with
``microcosm.build.us_runtime.cps_identified_county_sources``, and writes one
CSV row per (ASEC year, county) into
``microcosm/build/us_runtime/data/cps_asec_identified_counties.csv``, plus a
sibling ``.provenance.json`` recording each PDF's URL and SHA-256, the
``pdftotext`` version, and per-year counts.

The location draw (``us_runtime.block_location``) rules a county out for a
``GTCO = 0`` record only when this list names it *and* the year's file codes
it; see ``cps_excluded_county_sets``.

Example:
    uv run python tools/build_us_cps_identified_counties.py \
        --asec-year 2023 --asec-year 2024 --asec-year 2025
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from microcosm.build.us_runtime.block_location import (
    default_cps_asec_identified_counties_path,
    load_cps_asec_identified_counties,
)
from microcosm.build.us_runtime.cps_identified_county_sources import (
    LIST_HEADING,
    list_carries_entire_county_guarantee,
    parse_cps_identified_county_list,
)

TECHDOC_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar{yy}.pdf"
)
CSV_COLUMNS = (
    "asec_year",
    "state_fips",
    "county_code",
    "county_name",
    "single_county_micropolitan",
    "entire_county_guarantee",
    "source_url",
    "source_sha256",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--asec-year",
        type=int,
        action="append",
        required=True,
        help="ASEC survey year (repeatable), e.g. 2025 for cpsmar25.pdf.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "microcosm-cps-techdocs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_cps_asec_identified_counties_path(),
    )
    return parser.parse_args(argv)


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
    print(f"  downloading {url}", file=sys.stderr, flush=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "microcosm-build (CPS identified counties)"}
    )
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    if not payload.startswith(b"%PDF"):
        raise RuntimeError(f"{url} did not return a PDF.")
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(destination)
    return destination


def _pdftotext_version() -> str:
    result = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=False
    )
    return (result.stderr or result.stdout).splitlines()[0].strip()


def _render(pdf: Path) -> list[str]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True,
        check=True,
    )
    return io.StringIO(result.stdout.decode("utf-8")).read().splitlines()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rows: list[dict[str, str]] = []
    provenance: dict[str, object] = {
        "kind": "cps_asec_identified_counties",
        "list": LIST_HEADING,
        "note": (
            "Census (ASEC 2023-2025): 'Counties are only included on this list "
            "if the entire county is identified.' The 2026 documentation drops "
            "that sentence; entire_county_guarantee records, per year, whether "
            "the preamble carries it. An asterisk marks a county that is also a "
            "single-county micropolitan area."
        ),
        "pdftotext": _pdftotext_version(),
        "years": {},
    }
    for year in sorted(set(args.asec_year)):
        url = TECHDOC_URL_TEMPLATE.format(yy=f"{year % 100:02d}")
        pdf = _download(url, args.cache_dir)
        sha = _sha256(pdf)
        rendered = _render(pdf)
        counties = parse_cps_identified_county_list(rendered)
        guarantee = list_carries_entire_county_guarantee(rendered)
        provenance["years"][str(year)] = {  # type: ignore[index]
            "url": url,
            "sha256": sha,
            "counties": len(counties),
            "states": len({county.state_fips for county in counties}),
            "entire_county_guarantee": guarantee,
        }
        for county in counties:
            rows.append(
                {
                    "asec_year": str(year),
                    "state_fips": county.state_fips,
                    "county_code": county.county_code,
                    "county_name": county.county_name,
                    "single_county_micropolitan": str(
                        county.single_county_micropolitan
                    ).lower(),
                    "entire_county_guarantee": str(guarantee).lower(),
                    "source_url": url,
                    "source_sha256": sha,
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance["csv_sha256"] = _sha256(args.out)
    provenance_path = args.out.with_name(args.out.name + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    loaded = load_cps_asec_identified_counties(args.out)  # self-check
    print(
        json.dumps(
            {str(year): len(entry.counties) for year, entry in loaded.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
