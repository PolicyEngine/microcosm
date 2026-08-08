#!/usr/bin/env python3
"""Fetch the byte-pinned ACS PUMS archives into the ignored input cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from microcosm.build.us_runtime.acs_sources import (
    fetch_acs_pums_sources,
    load_acs_source_manifest,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INPUTS_DIR = _REPOSITORY_ROOT / "inputs" / "acs_2024_1yr"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=_DEFAULT_INPUTS_DIR,
        help="Ignored repository cache (default: inputs/acs_2024_1yr).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Override the packaged source manifest (primarily for testing).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_acs_source_manifest(args.manifest)
    source = fetch_acs_pums_sources(args.inputs_dir, manifest=manifest)
    print(
        json.dumps(
            {
                "source": asdict(manifest),
                "local_paths": {
                    "household": str(source.household_zip),
                    "person": str(source.person_zip),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
