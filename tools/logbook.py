"""Export, validate, and render the append-only Logbook build archives.

Archives are scoped: ``logbook/<country>/<scope>.jsonl`` holds one hash
chain, and chains are never merged across scopes — see ``logbook/README.md``
for why the boundary is permanent once a chain is born. ``export`` extends
exactly one named chain; ``validate`` and ``render`` accept either a single
archive or a directory of them, reporting chain by chain.

Remote export uses a distinct, read-only ``logbook_exporter`` JWT supplied
as ``POPULACE_LEDGER_EXPORT_KEY`` plus the hosted project's gateway key in
``POPULACE_LEDGER_API_KEY``.  It never reuses the insert-only writer key.
The live store is row-oriented and carries every attempt across all scopes;
the per-scope split is an archive convention, not a database partition.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request

from microcosm.build.logbook import (
    LOGBOOK_PROVENANCE_ROW_FIELDS,
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
#: Archives live at ``logbook/<country>/<scope>.jsonl`` — one hash chain per
#: file, never merged. A row's digest covers its predecessor, so a chain can
#: only ever be extended: rows cannot be re-rooted into another archive after
#: the fact without recomputing digests, which is exactly the tampering the
#: chain exists to detect. Scope boundaries are therefore chosen once, when a
#: chain is born, and country is the outermost one.
DEFAULT_ARCHIVE_ROOT = ROOT / "logbook"
DEFAULT_SPOOL_ROOT = ROOT / "logbook-spool"
REMOTE_EXPORT_KEY_ENV = "POPULACE_LEDGER_EXPORT_KEY"
REMOTE_API_KEY_ENV = "POPULACE_LEDGER_API_KEY"
REMOTE_PAGE_SIZE = 500
# Mirror of logbook.chain_scope() in
# supabase/migrations/20260818000000_logbook_chain_scopes.sql. The legacy US
# rows predate the scope split and must continue one mixed `us` chain forever.
LEGACY_US_PIPELINES = (
    "us-2024-release",
    "us-pool-inc2",
    "us-stacked-pool",
)
_PIPELINE_SCOPE_PATTERN = re.compile(
    r"^(?P<country>[a-z]{2})-(?P<line>[a-z0-9_]+)(?:-[a-z0-9_-]+)?$"
)
#: Mirror of logbook.scope_declared() in the same migration: the ratified
#: scope vocabulary, closed-world. Opening a scope is a reviewed diff here,
#: in the migration, and in logbook/README.md -- never a side effect of a
#: well-formed pipeline name.
DECLARED_SCOPES = frozenset({"us", "uk/frs"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate the append-only Microcosm Logbook archive."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export",
        help="Append a verified chain suffix to one scope's git archive.",
    )
    export.add_argument(
        "--archive",
        type=Path,
        required=True,
        help=(
            "The scope archive to extend, e.g. "
            "logbook/uk/<epic>.jsonl. Required and never defaulted: an "
            "export appends to exactly one chain, so which chain a run "
            "belongs to is the operator's declaration, not an inference."
        ),
    )
    source = export.add_mutually_exclusive_group()
    source.add_argument(
        "--source",
        type=Path,
        help=(
            "Completed spool directory or genesis-rooted Logbook JSONL source. "
            "Build spools land beside the run's output artifact, so this is "
            "normally an explicit path."
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
        default=DEFAULT_ARCHIVE_ROOT,
        help=(
            "One scope archive, or a directory of them validated chain by "
            f"chain (default: {DEFAULT_ARCHIVE_ROOT})."
        ),
    )

    render = subparsers.add_parser(
        "render",
        help="Render public-safe Logbook columns as a Markdown table.",
    )
    render.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help=(
            "One scope archive, or a directory of them rendered as a section "
            f"per chain (default: {DEFAULT_ARCHIVE_ROOT})."
        ),
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


def _archive_files(path: Path) -> tuple[Path, ...]:
    """Resolve an archive argument to the chain files it names.

    A directory holds one archive per scope, each an independent chain;
    read commands walk them all so every country's history is visible in
    one invocation without ever merging the chains themselves.
    """

    if not path.is_dir():
        return (path,)
    files = tuple(sorted(path.rglob("*.jsonl")))
    if not files:
        raise ValueError(f"No Logbook scope archives found under {path}.")
    return files


def _scope_label(archive: Path, root: Path | None = None) -> str:
    """Name a chain by its scope path (``us/pool``), not its filename.

    Relative to the directory the command was pointed at, so the country
    stays in the label wherever a set of chains lives; the ratified root is
    the fallback for a single archive named directly.
    """

    resolved = archive.resolve()
    for candidate in (root, DEFAULT_ARCHIVE_ROOT):
        if candidate is None:
            continue
        try:
            relative = resolved.relative_to(candidate.resolve())
        except ValueError:
            continue
        return str(relative.with_suffix(""))
    return resolved.stem


def _chain_scope(pipeline: str) -> str | None:
    """Return the Logbook chain scope declared by a pipeline name."""

    if pipeline in LEGACY_US_PIPELINES:
        return "us"
    match = _PIPELINE_SCOPE_PATTERN.fullmatch(pipeline)
    if match is None:
        return None
    return f"{match.group('country')}/{match.group('line')}"


def _archive_scope(archive: Path) -> str:
    """Derive the export scope from the ratified archive path.

    Every export appends to exactly one scope chain, so the archive path
    must name the scope — for local spools as much as for the live store.
    The chain verifier authenticates row payloads, never filenames: without
    this check a wrong spool would form a perfectly valid chain inside the
    wrong archive, permanently mis-scoping lineage.
    """

    if archive.suffix != ".jsonl":
        raise ValueError(
            f"export archive {archive} must be logbook/us.jsonl or "
            "logbook/<country>/<dataset>.jsonl"
        )
    parts = archive.parts
    scope = None
    if len(parts) >= 2 and parts[-2] == "logbook" and archive.name == "us.jsonl":
        scope = "us"
    elif len(parts) >= 3 and parts[-3] == "logbook":
        country = parts[-2]
        dataset = archive.stem
        if re.fullmatch(r"[a-z]{2}", country) and re.fullmatch(
            r"[a-z0-9_]+",
            dataset,
        ):
            scope = f"{country}/{dataset}"
    if scope is not None:
        if scope not in DECLARED_SCOPES:
            raise ValueError(
                f"export archive {archive} names scope {scope}, which is not "
                "in the ratified scope list; ratify it (migration + this "
                "mirror + README) before exporting"
            )
        return scope
    raise ValueError(
        f"export archive {archive} must be logbook/us.jsonl or "
        "logbook/<country>/<dataset>.jsonl"
    )


def _remote_rows(scope: str) -> tuple[LogbookRow, ...]:
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
            scope=scope,
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
        rows.extend(
            LogbookRow.from_mapping(_normalize_remote_row(item)) for item in page
        )
        offset += len(page)
        if total is not None and offset >= total:
            break
        if not page:
            if total is not None:
                raise RuntimeError(
                    "Logbook live store ended before its declared row count"
                )
            break
    wrong_scope = sorted(
        {
            row.pipeline
            for row in rows
            if _chain_scope(row.pipeline) != scope
        }
    )
    if wrong_scope:
        raise ValueError(
            f"Logbook live store returned pipelines outside scope {scope}: "
            f"{', '.join(wrong_scope)}"
        )
    return order_rows_by_chain(rows)


def _remote_builds_endpoint(
    url: str,
    *,
    scope: str,
    offset: int,
    limit: int,
) -> str:
    _validate_remote_url(url)
    base = url.rstrip("/")
    if base.endswith("/rest/v1/builds"):
        endpoint = base
    elif base.endswith("/rest/v1"):
        endpoint = f"{base}/builds"
    else:
        endpoint = f"{base}/rest/v1/builds"
    query_params = {
        "select": ",".join(sorted(LOGBOOK_PROVENANCE_ROW_FIELDS)),
        "order": "ts.asc,build_id.asc",
        "limit": str(limit),
        "offset": str(offset),
    }
    if scope == "us":
        query_params["pipeline"] = (
            f"in.({','.join(json.dumps(name) for name in LEGACY_US_PIPELINES)})"
        )
    else:
        country, dataset = scope.split("/", 1)
        query_params["pipeline"] = f"like.{country}-{dataset}-*"
    query = urlencode(query_params)
    return f"{endpoint}?{query}"


def _decode_remote_page(payload: bytes) -> list[dict[str, Any]]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Logbook live store returned invalid JSON: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Logbook live store response must be an array of rows")
    return value


def _normalize_remote_row(value: dict[str, Any]) -> dict[str, Any]:
    """Translate the migration's historical SQL NULL back to key absence."""

    normalized = dict(value)
    if normalized.get("run_provenance_identity", object()) is None:
        normalized.pop("run_provenance_identity")
    return normalized


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
            archives = _archive_files(args.archive)
            root = args.archive if args.archive.is_dir() else None
            for archive in archives:
                rows = load_logbook_file(archive)
                noun = "row" if len(rows) == 1 else "rows"
                tail = rows[-1].row_digest if rows else "none"
                scope = _scope_label(archive, root)
                print(f"validated {len(rows)} Logbook {noun} in {scope}; tail={tail}")
            return 0

        if args.command == "render":
            archives = _archive_files(args.archive)
            root = args.archive if args.archive.is_dir() else None
            dispositions = (
                set(args.disposition) if args.disposition is not None else None
            )
            for position, archive in enumerate(archives):
                rows = load_logbook_file(archive)
                table = render_markdown(
                    rows,
                    rung=args.rung,
                    dispositions=dispositions,
                )
                if len(archives) == 1:
                    print(table, end="")
                    continue
                # Chains are independent, so they are shown as separate
                # sections: one table spanning scopes would imply an
                # ordering across them that the archives do not assert.
                if position:
                    print()
                print(f"## {_scope_label(archive, root)}\n")
                print(table, end="")
            return 0

        if args.command == "export":
            if args.archive.is_dir():
                raise ValueError(
                    f"--archive {args.archive} is a directory; an export "
                    "extends exactly one scope chain, so name its archive "
                    "file (e.g. logbook/uk/<epic>.jsonl)."
                )
            if args.source is None and not args.remote:
                raise ValueError(
                    "export needs --source (the completed spool beside the "
                    "run's artifact) or --remote."
                )
            scope = _archive_scope(args.archive)
            if args.remote:
                candidates = _remote_rows(scope)
            else:
                candidates = _source_rows(args.source)
                # The same scope discipline as the remote branch: a wrong
                # spool would chain validly into the wrong archive, and a
                # committed mis-scope can never be re-rooted.
                wrong_scope = sorted(
                    {
                        row.pipeline
                        for row in candidates
                        if _chain_scope(row.pipeline) != scope
                    }
                )
                if wrong_scope:
                    raise ValueError(
                        f"Logbook source {args.source} holds pipelines "
                        f"outside scope {scope}: {', '.join(wrong_scope)}"
                    )
            receipt = export_rows(args.archive, candidates)
            print(
                f"exported {receipt.appended} new Logbook rows into "
                f"{_scope_label(args.archive)} "
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
