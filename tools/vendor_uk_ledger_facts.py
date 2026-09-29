#!/usr/bin/env python
"""Vendor pinned Chronicle facts into UK build resources.

Offline authoring tool: it reads an already-exported Chronicle consumer
artifact, checks it against the committed national feed pin, and copies the
facts named by ``uk/ledger_fact_vendor_selections.json`` into per-concern JSON
resources under the UK country package. It never derives a value.

    uv run --no-sync python tools/vendor_uk_ledger_facts.py \\
        --ledger-facts .codex-work/uk-artifact [--check] [--resource NAME ...]
"""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path

from microcosm.build.ledger_artifact import (
    add_ledger_artifact_args,
    resolve_ledger_artifact,
)
from microcosm.build.uk_runtime.chronicle_feed import (
    load_uk_chronicle_feed,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import (
    load_vendor_selections,
    vendor_all,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_ledger_artifact_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write resources into (default: the packaged uk/ directory).",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=None,
        help="Vendor only this resource (repeatable); default is every register entry.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if any committed resource differs from the regeneration.",
    )
    parser.add_argument(
        "--allow-unpinned-feed",
        action="store_true",
        help="Vendor from a feed whose digests differ from the national pin (diagnostic only).",
    )
    parser.add_argument(
        "--report-counts",
        action="store_true",
        help="Print matched row counts per selection without enforcing expected_row_count.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifact = resolve_ledger_artifact(args)
    if artifact is None:
        raise SystemExit("error: --ledger-facts is required.")
    pin = load_uk_chronicle_feed()
    if artifact.facts_sha256 != pin.facts_sha256 or (
        artifact.manifest_sha256 is not None
        and artifact.manifest_sha256 != pin.manifest_sha256
    ):
        message = (
            "Chronicle consumer facts differ from the committed UK national feed pin: "
            f"loaded facts {artifact.facts_sha256}, committed {pin.facts_sha256}."
        )
        if not args.allow_unpinned_feed:
            raise SystemExit(
                f"error: {message} Re-pin first, or pass --allow-unpinned-feed for a diagnostic run."
            )
        print(f"warning: {message} (--allow-unpinned-feed)", file=sys.stderr)
    register = load_vendor_selections()
    output_dir = (
        args.output_dir
        or Path(
            str(files("microcosm.build.uk").joinpath("country_package.json"))
        ).parent
    )
    results = vendor_all(
        register,
        artifact.facts,
        pin=pin,
        only=args.resource,
        strict_counts=not args.report_counts,
    )
    drift = 0
    for result in results:
        target = output_dir / result.resource
        if args.report_counts:
            for selection in result.payload["selections"]:
                print(
                    f"{result.resource} {selection['label']}: {selection['row_count']} rows"
                )
        if args.check:
            current = target.read_bytes() if target.exists() else b""
            status = "ok" if current == result.content else "DRIFT"
            if status == "DRIFT":
                drift += 1
            print(
                f"{status} {result.resource} rows={result.row_count} sha256={result.sha256}"
            )
            continue
        if not args.report_counts:
            target.write_bytes(result.content)
            print(f"wrote {target} rows={result.row_count} sha256={result.sha256}")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
