"""Export, validate, and render the append-only Logbook build archive.

Remote export uses a distinct, read-only ``logbook_exporter`` JWT supplied
as ``POPULACE_LEDGER_EXPORT_KEY`` plus the hosted project's gateway key in
``POPULACE_LEDGER_API_KEY``.  It never reuses the insert-only writer key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request

from microcosm.build.logbook import (
    LOGBOOK_ROW_FIELDS,
    LogbookRow,
    _validate_remote_url,
    export_rows,
    load_logbook_file,
    load_spool_rows,
    order_rows_by_chain,
    render_markdown,
    urlopen,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "logbook.jsonl"
DEFAULT_SOURCE = ROOT / "logbook-spool"
REMOTE_EXPORT_KEY_ENV = "POPULACE_LEDGER_EXPORT_KEY"
REMOTE_API_KEY_ENV = "POPULACE_LEDGER_API_KEY"
REMOTE_PAGE_SIZE = 500


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate the append-only Microcosm Logbook archive."
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
        help=f"Logbook JSONL archive (default: {DEFAULT_ARCHIVE}).",
    )
    source = export.add_mutually_exclusive_group()
    source.add_argument(
        "--source",
        type=Path,
        help=(
            "Completed spool directory or genesis-rooted Logbook JSONL source "
            f"(default: {DEFAULT_SOURCE})."
        ),
    )
    source.add_argument(
        "--remote",
        action="store_true",
        help=(
            "Read the private live store using POPULACE_LEDGER_URL and the "
            f"read-only {REMOTE_EXPORT_KEY_ENV}, authenticated at the gateway "
            f"by {REMOTE_API_KEY_ENV}."
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
        help=f"Logbook JSONL archive (default: {DEFAULT_ARCHIVE}).",
    )

    render = subparsers.add_parser(
        "render",
        help="Render public-safe Logbook columns as a Markdown table.",
    )
    render.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Logbook JSONL archive (default: {DEFAULT_ARCHIVE}).",
    )
    render.add_argument(
        "--rung",
        help="Include only this Logbook rung token (f001, f004, f010, f025, or f100).",
    )
    render.add_argument(
        "--disposition",
        action="append",
        help="Include this disposition; repeat to include more than one.",
    )
    return parser


def _source_rows(path: Path) -> tuple[LogbookRow, ...]:
    if path.is_dir():
        return load_spool_rows(path)
    return load_logbook_file(path)


def _remote_rows() -> tuple[LogbookRow, ...]:
    ledger_url = os.environ.get("POPULACE_LEDGER_URL")
    export_key = os.environ.get(REMOTE_EXPORT_KEY_ENV)
    api_key = os.environ.get(REMOTE_API_KEY_ENV)
    if not ledger_url or not export_key or not api_key:
        raise ValueError(
            "remote export requires POPULACE_LEDGER_URL, "
            f"{REMOTE_EXPORT_KEY_ENV}, and {REMOTE_API_KEY_ENV}"
        )

    rows: list[LogbookRow] = []
    offset = 0
    while True:
        endpoint = _remote_builds_endpoint(
            ledger_url,
            offset=offset,
            limit=REMOTE_PAGE_SIZE,
        )
        request = Request(
            endpoint,
            headers={
                "Accept": "application/json",
                "Accept-Profile": "logbook",
                "apikey": api_key,
                "Authorization": f"Bearer {export_key}",
                "Prefer": "count=exact",
            },
        )
        with urlopen(request, timeout=30.0) as response:
            status = getattr(response, "status", 200)
            if not 200 <= status < 300:
                raise RuntimeError(f"Logbook live store returned HTTP {status}")
            page = _decode_remote_page(response.read())
            total = _content_range_total(getattr(response, "headers", {}))
        rows.extend(LogbookRow.from_mapping(item) for item in page)
        offset += len(page)
        if total is not None and offset >= total:
            break
        if not page:
            if total is not None:
                raise RuntimeError(
                    "Logbook live store ended before its declared row count"
                )
            break
    return order_rows_by_chain(rows)


def _remote_builds_endpoint(url: str, *, offset: int, limit: int) -> str:
    _validate_remote_url(url)
    base = url.rstrip("/")
    if base.endswith("/rest/v1/builds"):
        endpoint = base
    elif base.endswith("/rest/v1"):
        endpoint = f"{base}/builds"
    else:
        endpoint = f"{base}/rest/v1/builds"
    query = urlencode(
        {
            "select": ",".join(sorted(LOGBOOK_ROW_FIELDS)),
            "order": "ts.asc,build_id.asc",
            "limit": str(limit),
            "offset": str(offset),
        }
    )
    return f"{endpoint}?{query}"


def _decode_remote_page(payload: bytes) -> list[dict[str, Any]]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Logbook live store returned invalid JSON: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Logbook live store response must be an array of rows")
    return value


def _content_range_total(headers: Any) -> int | None:
    content_range = headers.get("Content-Range")
    if not isinstance(content_range, str) or "/" not in content_range:
        return None
    total = content_range.rsplit("/", 1)[1]
    if total == "*":
        return None
    try:
        parsed = int(total)
    except ValueError as exc:
        raise ValueError(
            f"Logbook live store returned invalid Content-Range {content_range!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"Logbook live store returned invalid Content-Range {content_range!r}"
        )
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Run one Logbook command and return its process exit code."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            rows = load_logbook_file(args.archive)
            noun = "row" if len(rows) == 1 else "rows"
            tail = rows[-1].row_digest if rows else "none"
            print(f"validated {len(rows)} Logbook {noun}; tail={tail}")
            return 0

        if args.command == "render":
            rows = load_logbook_file(args.archive)
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
            candidates = (
                _remote_rows()
                if args.remote
                else _source_rows(args.source or DEFAULT_SOURCE)
            )
            receipt = export_rows(args.archive, candidates)
            print(
                f"exported {receipt.appended} new Logbook rows "
                f"({receipt.existing} already archived); "
                f"tail={receipt.tail_digest or 'none'}"
            )
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"logbook {args.command} failed: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
