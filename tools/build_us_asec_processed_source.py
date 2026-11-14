"""Build a processed CPS ASEC ``census_cps_YYYY.h5`` from its pinned Census archive.

    uv run python tools/build_us_asec_processed_source.py \\
        --income-year 2025 --output /tmp/census_cps_2025.h5 \\
        [--archive asecpub26csv.zip | --cache-dir DIR] \\
        [--compare-to REFERENCE.h5 | --compare-to-pinned] \\
        [--receipt receipt.json] [--overwrite]

The archive for survey year ``income_year + 1`` must match its pin in
``education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES`` (byte
length, SHA-256 and person member). Without ``--archive`` it is taken from the
education-assistance cache, downloaded and verified there if absent. The
build itself is ``microcosm.build.us_runtime.asec_processed_source``.

``--compare-to`` compares the new store with a reference store, table by
table (``compare_asec_processed_stores``). ``--compare-to-pinned`` uses the
income year's pinned processed file from ``asec_sources.fetch_asec_source``,
which verifies its bytes first. A comparison that finds any difference exits
with status 1, after the receipt is written.

Rebuilds reproduce tables, not bytes: HDF5 files are not byte-deterministic.
Write rebuilds to a new path. The tool refuses to replace a file holding a
pinned input's bytes, even with ``--overwrite``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from microcosm.build.us_runtime import asec_processed_source
from microcosm.build.us_runtime.asec_processed_source import (
    build_asec_processed_source,
    compare_asec_processed_stores,
)
from microcosm.build.us_runtime.asec_sources import (
    asec_source_artifact,
    fetch_asec_source,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "microcosm.us.asec_processed_source.build.v1"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--income-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Local asecpubYYcsv.zip; verified against its pin before use.",
    )
    source.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Education-assistance archive cache to read or download into.",
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Reference census_cps store to compare the new store with.",
    )
    reference.add_argument(
        "--compare-to-pinned",
        action="store_true",
        help="Compare with the income year's pinned processed file.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Write a JSON receipt of the build (and comparison) here.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output (never a pinned input's bytes).",
    )
    return parser.parse_args(argv)


def _display_path(path: str | Path) -> str:
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home) :] if text.startswith(home + "/") else text


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _environment() -> dict[str, Any]:
    versions = {}
    for package in ("pandas", "numpy", "tables", "microunit"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    status = _git("status", "--porcelain")
    return {
        "python": platform.python_version(),
        **versions,
        "microcosm_commit": _git("rev-parse", "HEAD"),
        "microcosm_tree_dirty": None if status is None else bool(status),
        # Content digests identify the code that ran even from a dirty tree.
        "builder_sha256": {
            _display_path(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(asec_processed_source.__file__).resolve(),
                Path(__file__).resolve(),
            )
            if path.is_relative_to(ROOT)
        },
    }


def _comparison_summary(report: dict[str, Any]) -> dict[str, Any]:
    tables = {}
    for key, table in report["tables"].items():
        storage = table["hdf5"]
        tables[key] = {
            "equal": table["equal"],
            "frame_equal_exact": table["frame_equal_exact"],
            "detail": table["detail"],
            "shape": table["shape"],
            "column_order_equal": table["column_order_equal"],
            "only_candidate": table["only_candidate"],
            "only_reference": table["only_reference"],
            "dtype_mismatches": table["dtype_mismatches"],
            "value_mismatches": table["value_mismatches"],
            "hdf5_storage": {
                name: storage[name]
                for name in (
                    "equal",
                    "columns",
                    "mismatched_columns",
                    "only_candidate",
                    "only_reference",
                    "axes_equal",
                    "group_attrs_equal",
                    "block_layout_equal",
                )
            },
        }
        if not storage["block_layout_equal"]:
            tables[key]["hdf5_storage"]["block_layout"] = {
                side: [[atom.split("(")[0], len(items)] for atom, items in layout]
                for side, layout in storage["block_layout"].items()
            }
    return {
        "equal": report["equal"],
        "keys_only_candidate": report["keys_only_candidate"],
        "keys_only_reference": report["keys_only_reference"],
        "tables": tables,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    try:
        summary = build_asec_processed_source(
            args.income_year,
            args.output,
            archive=args.archive,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    built = time.monotonic()
    summary["archive"]["path"] = _display_path(summary["archive"]["path"])
    summary["output"]["path"] = _display_path(summary["output"]["path"])
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, **summary}

    reference_path: Path | None = args.compare_to
    reference_pin: str | None = None
    if args.compare_to_pinned:
        try:
            reference_path = fetch_asec_source(args.income_year)
        except ValueError as error:
            raise SystemExit(f"error: {error}") from error
        reference_pin = asec_source_artifact(args.income_year).sha256
    equal = True
    if reference_path is not None:
        report = compare_asec_processed_stores(args.output, reference_path)
        equal = bool(report["equal"])
        receipt["comparison"] = {
            "reference": {
                "path": _display_path(reference_path),
                "pinned_sha256": reference_pin,
                "size_bytes": Path(reference_path).stat().st_size,
            },
            **_comparison_summary(report),
        }
    receipt["elapsed_seconds"] = {
        "build": round(built - started, 1),
        "total": round(time.monotonic() - started, 1),
    }
    receipt["environment"] = _environment()

    text = json.dumps(receipt, indent=1, sort_keys=False) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text, end="")
    if not equal:
        print(
            f"error: {args.output} differs from {reference_path}; see the "
            "comparison above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
