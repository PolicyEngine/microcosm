"""Resolve the pinned CPS ASEC inputs and print the base builder's arguments.

    uv run python tools/fetch_us_asec_sources.py [--cache-dir DIR] [YEAR ...]

Each requested income year (default: every pinned year) resolves through
``microcosm.build.us_runtime.asec_sources.fetch_asec_source``: a known local
copy whose byte length and SHA-256 match the pin, else a download of the
pinned Hugging Face revision, verified the same way. One line per year is
printed in the form ``tools/build_us_puf_support_base.py`` takes::

    --asec-h5 2022=/path/census_cps_2022.h5 --asec-h5-sha256 2022=7ccca976...

so the output pastes into a base-build invocation and the builder re-verifies
the bytes before any stage runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microcosm.build.us_runtime.asec_sources import (
    ASEC_SOURCE_ARTIFACTS,
    asec_source_artifact,
    fetch_asec_source,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "years",
        nargs="*",
        type=int,
        help="Income years to resolve; the default is every pinned year.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "huggingface_hub cache directory for downloads. When given, the "
            "local convenience copies are not consulted."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    years = args.years or sorted(ASEC_SOURCE_ARTIFACTS)
    try:
        for year in years:
            artifact = asec_source_artifact(year)
            path = fetch_asec_source(year, args.cache_dir)
            print(f"--asec-h5 {year}={path} --asec-h5-sha256 {year}={artifact.sha256}")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    sys.exit(main())
