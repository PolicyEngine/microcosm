"""Resolve the pinned CPS ASEC inputs and print the base builder's arguments.

    uv run python tools/fetch_us_asec_sources.py [--cache-dir DIR] [--all-pinned | YEAR ...]

With no years, the default pool is resolved:
``microcosm.build.us_runtime.asec_sources.ASEC_DEFAULT_POOL_INCOME_YEARS``, the
newest three pinned income years. ``--all-pinned`` resolves every pinned year,
and explicit years resolve exactly those, so income year 2022 stays available
for byte-reproducible historical builds. Each year resolves through
``fetch_asec_source``: a known local copy whose byte length and SHA-256 match
the pin, else a download of that file's pinned Hugging Face revision, verified
the same way. One line per year is printed in the form
``tools/build_us_puf_support_base.py`` takes::

    --asec-h5 2025=/path/census_cps_2025.h5 --asec-h5-sha256 2025=4c5a3218...

so the output pastes into a base-build invocation and the builder re-verifies
the bytes before any stage runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microcosm.build.us_runtime.asec_sources import (
    ASEC_DEFAULT_POOL_INCOME_YEARS,
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
        help=(
            "Income years to resolve; the default is the default pool "
            f"{list(ASEC_DEFAULT_POOL_INCOME_YEARS)}."
        ),
    )
    parser.add_argument(
        "--all-pinned",
        action="store_true",
        help=f"Resolve every pinned year {sorted(ASEC_SOURCE_ARTIFACTS)}.",
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
    args = parser.parse_args(argv)
    if args.all_pinned and args.years:
        parser.error("--all-pinned takes no explicit years.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.all_pinned:
        years = sorted(ASEC_SOURCE_ARTIFACTS)
    else:
        years = args.years or list(ASEC_DEFAULT_POOL_INCOME_YEARS)
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
