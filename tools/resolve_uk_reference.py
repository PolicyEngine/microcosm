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
``reference.json`` beside it, with ``input_mass_reference.json`` (the
build's input-mass totals and carriers, licensed-data aggregates that stay
with the run) and ``reference_surface.json`` (the columns it exports and the
loader inputs it carries, names and flags only). A dense build takes
``--input-mass-reference`` and ``--reference-surface`` from it. It refuses a
directory that already holds a reference, so a run keeps the one it started
with.

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
    write_uk_cut_reference_evidence,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release", required=True, choices=sorted(UK_CUT_REFERENCE_RELEASES)
    )
    parser.add_argument(
        "--build",
        help=(
            "Pin a listed build instead of the line's current one, for a line "
            "with no current build; checked like the current build and recorded "
            "as pinned."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def fetch_reference(
    release: str,
    output_dir: Path,
    *,
    build: str | None = None,
    hub_download=None,
    write_evidence=write_uk_cut_reference_evidence,
) -> dict:
    """Download the current build of ``release`` into ``output_dir``.

    Beside the dataset and its record go the evidence the gates read from it:
    its input-mass totals and carriers (``input_mass_reference.json``) and its
    surface (``reference_surface.json``), which a dense build takes with
    ``--input-mass-reference`` and ``--reference-surface``.
    """

    reference = fetch_uk_cut_reference(
        release, output_dir, build=build, hub_download=hub_download
    )
    written = write_evidence(reference, output_dir / reference.filename, output_dir)
    return {
        "reference": reference.label,
        "sha256": reference.sha256,
        "dataset": str(output_dir / reference.filename),
        "record": str(output_dir / UK_CUT_REFERENCE_RECORD_NAME),
        **{name: str(path) for name, path in written.items()},
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    hub_download=None,
    write_evidence=write_uk_cut_reference_evidence,
) -> int:
    args = _parse_args(argv)
    from microcosm.build.staging_dataset import hub_offline

    if hub_offline():
        # A cached releases.json can name a build that is no longer current
        # (the review of microcosm#1574); run_uk_cut refuses offline too.
        print(
            "refused: unset HF_HUB_OFFLINE: the reference is the line's current "
            "build, which only the Hub's releases.json can say.",
            file=sys.stderr,
        )
        return 1
    try:
        if args.check:
            summary = check_uk_cut_reference(
                args.release, build=args.build, hub_download=hub_download
            )
        else:
            summary = fetch_reference(
                args.release,
                args.output_dir.expanduser(),
                build=args.build,
                hub_download=hub_download,
                write_evidence=write_evidence,
            )
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
