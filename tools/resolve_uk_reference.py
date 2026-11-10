"""Resolve the current build a UK cut compares with, and keep its record.

A UK cut checks and certifies its candidate against the current build of its
own release line, as the Hub's ``releases.json`` names it (microcosm#1095):
the national line against the national, the dense line against the dense.
``tools/run_uk_cut.py`` resolves its reference itself, once, when it fetches
its inputs. This command does the same for runs outside it, such as a dense
build, and checks what a cut would compare with now::

    uv run --no-sync python tools/resolve_uk_reference.py --release microcosm-uk-2024-25-dense --check
    uv run --no-sync python tools/resolve_uk_reference.py --release microcosm-uk-2024-25-dense --output-dir ~/microcosm-uk-work/dense/reference

``--check`` reads the release record, the build's manifest and its
certification evidence, and downloads no dataset. ``--output-dir`` downloads
the dataset, verifies its SHA-256, copies it there and writes
``reference.json`` beside it; pass both paths to the build's
``--reference-*`` options. It refuses a directory that already holds a
reference, so a run keeps the one it started with.

Reading ``policyengine/populace-uk-private`` needs ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from microcosm.build.uk_runtime.cut_reference import (
    UK_CUT_REFERENCE_RECORD_NAME,
    UK_CUT_REFERENCE_RELEASES,
    check_uk_cut_reference,
    fetch_uk_cut_reference,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release", required=True, choices=sorted(UK_CUT_REFERENCE_RELEASES)
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def fetch_reference(release: str, output_dir: Path, *, hub_download=None) -> dict:
    """Download the current build of ``release`` into ``output_dir``."""

    reference = fetch_uk_cut_reference(release, output_dir, hub_download=hub_download)
    return {
        "reference": reference.label,
        "sha256": reference.sha256,
        "dataset": str(output_dir / reference.filename),
        "record": str(output_dir / UK_CUT_REFERENCE_RECORD_NAME),
    }


def main(argv: Sequence[str] | None = None, *, hub_download=None) -> int:
    args = _parse_args(argv)
    try:
        if args.check:
            summary = check_uk_cut_reference(args.release, hub_download=hub_download)
        else:
            summary = fetch_reference(
                args.release, args.output_dir.expanduser(), hub_download=hub_download
            )
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
