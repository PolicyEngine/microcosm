#!/usr/bin/env python
"""Generate the UK local authority names resource from the ONS LAD23 lookup.

Writes ``local_authority_names.json``: every April 2023 local authority
district code with its ONS display name and the policyengine-uk
``LocalAuthority`` member name the rowwise build writes as the household
``local_authority`` input (microcosm#953). The source bytes must match
``geography_sources.LAD23_NAMES_SHA256``; the resource records that digest and
the loader asserts it, so a re-published lookup is a reviewed re-pin, never a
silent regeneration.

Run from the repository root::

    uv run python tools/generate_uk_local_authority_names.py \\
        --source-csv build/uk/lad23_names_and_codes.csv

``--check`` compares the generated payload with the committed resource and
exits non-zero on drift; without ``--source-csv`` the lookup is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

import microcosm.build.uk_runtime.geography_sources as geography_sources
from microcosm.build.uk_runtime.geography_sources import (
    LAD23_NAMES_ITEM_ID,
    LAD23_NAMES_URL,
    normalise_lad23_names,
)
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_NAMES_KIND,
    UK_LOCAL_AUTHORITY_NAMES_RESOURCE,
    UK_LOCAL_AUTHORITY_NAMES_SCHEMA_VERSION,
    UK_LOCAL_AUTHORITY_VINTAGE,
    local_authority_engine_key,
)

DEFAULT_SOURCE_CSV = Path("build/uk/lad23_names_and_codes.csv")
DEFAULT_OUTPUT = (
    Path("packages/microcosm-build/src/microcosm/build/uk")
    / UK_LOCAL_AUTHORITY_NAMES_RESOURCE
)
SOURCE_NAME = (
    "Local Authority Districts (April 2023) Names and Codes in the United Kingdom"
)
SOURCE_PUBLISHER = "ONS Open Geography Portal"


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "microcosm-build (UK local authority names)"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Empty download from {url}")
    return payload


def build_local_authority_names(
    *,
    source_csv: Path | None = None,
    url: str = LAD23_NAMES_URL,
) -> dict[str, Any]:
    """Build the resource payload from a local CSV or the published lookup."""

    payload = (
        Path(source_csv).read_bytes() if source_csv is not None else _fetch_bytes(url)
    )
    if not payload:
        raise ValueError("LAD23 names lookup is empty.")
    # The pinned digest, not whatever was downloaded: a re-published lookup
    # is re-pinned in geography_sources first, then regenerated here.
    geography_sources.verify_lad23_names_bytes(payload)
    lookup = normalise_lad23_names(pd.read_csv(io.BytesIO(payload), dtype=str))
    areas: dict[str, dict[str, str]] = {}
    for row in lookup.itertuples(index=False):
        areas[str(row.local_authority_code)] = {
            "name": str(row.local_authority_name),
            "engine_key": local_authority_engine_key(row.local_authority_name),
        }
    keys = [entry["engine_key"] for entry in areas.values()]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(
            f"LAD23 names map two authorities to one engine key: {duplicates}."
        )
    return {
        "schema_version": UK_LOCAL_AUTHORITY_NAMES_SCHEMA_VERSION,
        "country": "uk",
        "kind": UK_LOCAL_AUTHORITY_NAMES_KIND,
        "description": (
            "Every UK local authority district on the April 2023 ONS frame, "
            "with its ONS display name and the policyengine-uk LocalAuthority "
            "member name the rowwise build writes as the household "
            "local_authority input. engine_key is the mechanical rule in "
            "microcosm.build.uk_runtime.local_authority_input (upper-case, "
            "non-alphanumeric runs to one underscore) with the module's "
            "declared aliases; regenerate with "
            "tools/generate_uk_local_authority_names.py (microcosm#953)."
        ),
        "source": {
            "name": SOURCE_NAME,
            "publisher": SOURCE_PUBLISHER,
            "item_id": LAD23_NAMES_ITEM_ID,
            "url": url,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "columns": ["LAD23CD", "LAD23NM"],
            "vintage": UK_LOCAL_AUTHORITY_VINTAGE,
        },
        "area_count": len(areas),
        "areas": dict(sorted(areas.items())),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_csv = args.source_csv
    if source_csv is None and DEFAULT_SOURCE_CSV.exists():
        source_csv = DEFAULT_SOURCE_CSV
    resource = build_local_authority_names(source_csv=source_csv, url=args.url)
    rendered = json.dumps(resource, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        committed = args.output.read_text(encoding="utf-8")
        if committed != rendered:
            print(f"{args.output} is stale; regenerate it.", file=sys.stderr)
            return 1
        print(f"{args.output} is current ({resource['area_count']} authorities).")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {args.output} ({resource['area_count']} authorities, sha256 {resource['source']['sha256'][:12]})"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help=f"local copy of the ONS lookup (default: {DEFAULT_SOURCE_CSV} when present, else download)",
    )
    parser.add_argument("--url", default=LAD23_NAMES_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed resource instead of writing",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
