"""Export, validate, and render the append-only Chronicle build archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from populace.build.chronicle import (
    ChronicleRow,
    export_rows,
    load_chronicle_file,
    load_spool_rows,
    render_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "chronicle.jsonl"
DEFAULT_SOURCE = ROOT / "ledger-spool"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate the append-only Populace Chronicle archive."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export",
        help="Append a verified chain suffix to the git archive.",
    )
    export.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Chronicle JSONL archive (default: {DEFAULT_ARCHIVE}).",
    )
    export.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "Completed spool directory or genesis-rooted Chronicle JSONL source "
            f"(default: {DEFAULT_SOURCE})."
        ),
    )

    validate = subparsers.add_parser(
        "validate",
        help="Fail unless every archive row and chain link is valid.",
    )
    validate.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Chronicle JSONL archive (default: {DEFAULT_ARCHIVE}).",
    )

    render = subparsers.add_parser(
        "render",
        help="Render public-safe Chronicle columns as a Markdown table.",
    )
    render.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Chronicle JSONL archive (default: {DEFAULT_ARCHIVE}).",
    )
    render.add_argument(
        "--rung",
        help="Include only this Chronicle rung token (f001, f010, or f100).",
    )
    render.add_argument(
        "--disposition",
        action="append",
        help="Include this disposition; repeat to include more than one.",
    )
    return parser


def _source_rows(path: Path) -> tuple[ChronicleRow, ...]:
    if path.is_dir():
        return load_spool_rows(path)
    return load_chronicle_file(path)


def main(argv: list[str] | None = None) -> int:
    """Run one Chronicle command and return its process exit code."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            rows = load_chronicle_file(args.archive)
            noun = "row" if len(rows) == 1 else "rows"
            tail = rows[-1].row_digest if rows else "none"
            print(f"validated {len(rows)} Chronicle {noun}; tail={tail}")
            return 0

        if args.command == "render":
            rows = load_chronicle_file(args.archive)
            dispositions = (
                set(args.disposition) if args.disposition is not None else None
            )
            print(
                render_markdown(
                    rows,
                    rung=args.rung,
                    dispositions=dispositions,
                ),
                end="",
            )
            return 0

        if args.command == "export":
            candidates = _source_rows(args.source)
            receipt = export_rows(args.archive, candidates)
            print(
                f"exported {receipt.appended} new Chronicle rows "
                f"({receipt.existing} already archived); "
                f"tail={receipt.tail_digest or 'none'}"
            )
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"chronicle {args.command} failed: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
