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
    DECLARED_LOGBOOK_SCOPES,
    LEGACY_US_PIPELINES,
    LOGBOOK_ROW_FIELDS,
    LogbookRow,
    _validate_remote_url,
    export_rows,
    load_logbook_file,
    load_spool_rows,
    logbook_chain_scope,
    order_rows_by_chain,
    render_markdown,
    spool_build_rows,
    urlopen,
)
from microcosm.build.logbook_family import (
    FamilyAction,
    FamilyArchiveRecords,
    FamilyMember,
    LogbookFamily,
    export_family_scope,
    import_family_scope,
    load_family_archive_records,
    load_family_spool,
    reconcile_logbook_spool,
    validate_family_membership,
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
FAMILY_ARCHIVE_DIRECTORIES = frozenset({"families", "family_members", "family_actions"})
#: Mirror of logbook.scope_declared() in the database migrations: the ratified
#: scope vocabulary, closed-world. Opening a scope is a reviewed diff here,
#: in the migration, and in logbook/README.md -- never a side effect of a
#: well-formed pipeline name.
DECLARED_SCOPES = DECLARED_LOGBOOK_SCOPES


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

    family_export = subparsers.add_parser(
        "family-export",
        help="Append family records to the three archives for one scope.",
    )
    family_export.add_argument(
        "--scope", choices=sorted(DECLARED_SCOPES), required=True
    )
    family_export.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
    )
    family_source = family_export.add_mutually_exclusive_group(required=True)
    family_source.add_argument(
        "--source",
        type=Path,
        help="A Logbook spool containing family record subdirectories.",
    )
    family_source.add_argument(
        "--remote",
        action="store_true",
        help="Read family records for the scope from the live store.",
    )

    family_import = subparsers.add_parser(
        "family-import",
        help="Copy one scope's family archives into a durable local spool.",
    )
    family_import.add_argument(
        "--scope", choices=sorted(DECLARED_SCOPES), required=True
    )
    family_import.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
    )
    family_import.add_argument(
        "--spool",
        type=Path,
        default=DEFAULT_SPOOL_ROOT,
    )

    reconcile = subparsers.add_parser(
        "reconcile",
        help="Send queued builds and family records in dependency order.",
    )
    reconcile.add_argument(
        "--spool",
        type=Path,
        default=DEFAULT_SPOOL_ROOT,
    )

    list_families = subparsers.add_parser(
        "list-families",
        help="List archived dataset families.",
    )
    list_families.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
    )
    list_families.add_argument("--scope", choices=sorted(DECLARED_SCOPES))

    list_builds = subparsers.add_parser(
        "list-family-builds",
        help="List archived builds associated with one family.",
    )
    list_builds.add_argument("--family-id", required=True)
    list_builds.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
    )

    show_history = subparsers.add_parser(
        "show-family-history",
        help="Show archived revocations and replacements for one family.",
    )
    show_history.add_argument("--family-id", required=True)
    show_history.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
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

    if _within_family_archive_directory(path):
        raise ValueError(f"No Logbook build archives found under {path}.")
    if not path.is_dir():
        return (path,)
    files = tuple(
        sorted(
            candidate
            for candidate in path.rglob("*.jsonl")
            if not _is_family_archive(candidate, root=path)
        )
    )
    if not files:
        raise ValueError(f"No Logbook scope archives found under {path}.")
    return files


def _is_family_archive(path: Path, *, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] in FAMILY_ARCHIVE_DIRECTORIES


def _within_family_archive_directory(path: Path) -> bool:
    candidates = (path, *path.parents)
    return any(candidate.name in FAMILY_ARCHIVE_DIRECTORIES for candidate in candidates)


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

    return logbook_chain_scope(pipeline)


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
        rows.extend(LogbookRow.from_database_mapping(item) for item in page)
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
        {row.pipeline for row in rows if _chain_scope(row.pipeline) != scope}
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
        "select": ",".join(sorted(LOGBOOK_ROW_FIELDS)),
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


def _remote_family_records(scope: str) -> FamilyArchiveRecords:
    families = tuple(
        LogbookFamily.from_mapping(row)
        for row in _remote_table_rows(
            table="families",
            fields=("family_id", "chain_scope", "source_pool_sha256"),
            scope=scope,
        )
    )
    members = tuple(
        FamilyMember.from_mapping(row)
        for row in _remote_table_rows(
            table="family_members_public",
            fields=("family_id", "build_id"),
            scope=scope,
        )
    )
    actions = tuple(
        FamilyAction.from_mapping(row)
        for row in _remote_table_rows(
            table="family_actions_public",
            fields=(
                "action_id",
                "family_id",
                "build_id",
                "action_type",
                "related_build_id",
                "recorded_at",
                "actor",
                "reason",
                "evidence_location",
            ),
            scope=scope,
        )
    )
    return FamilyArchiveRecords(families, members, actions)


def _remote_table_rows(
    *,
    table: str,
    fields: tuple[str, ...],
    scope: str,
) -> tuple[dict[str, Any], ...]:
    ledger_url = os.environ.get("POPULACE_LEDGER_URL")
    export_key = os.environ.get(REMOTE_EXPORT_KEY_ENV)
    api_key = os.environ.get(REMOTE_API_KEY_ENV)
    if not ledger_url or not export_key or not api_key:
        raise ValueError(
            "remote export requires POPULACE_LEDGER_URL, "
            f"{REMOTE_EXPORT_KEY_ENV}, and {REMOTE_API_KEY_ENV}"
        )

    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        endpoint = _remote_table_endpoint(
            ledger_url,
            table=table,
            fields=fields,
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
        rows.extend(page)
        offset += len(page)
        if total is not None and offset >= total:
            break
        if not page:
            if total is not None:
                raise RuntimeError(
                    "Logbook live store ended before its declared row count"
                )
            break
    return tuple(rows)


def _remote_table_endpoint(
    url: str,
    *,
    table: str,
    fields: tuple[str, ...],
    scope: str,
    offset: int,
    limit: int,
) -> str:
    _validate_remote_url(url)
    base = url.rstrip("/")
    if base.endswith("/rest/v1/builds"):
        base = base[: -len("/builds")]
    elif not base.endswith("/rest/v1"):
        base = f"{base}/rest/v1"
    query = urlencode(
        {
            "select": ",".join(fields),
            "chain_scope": f"eq.{scope}",
            "order": ",".join(f"{field}.asc" for field in fields[:2]),
            "limit": str(limit),
            "offset": str(offset),
        }
    )
    return f"{base}/{table}?{query}"


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


def _archived_family_records(
    archive_root: Path,
    *,
    scope: str | None = None,
) -> FamilyArchiveRecords:
    scopes = (scope,) if scope is not None else tuple(sorted(DECLARED_SCOPES))
    families: list[LogbookFamily] = []
    members: list[FamilyMember] = []
    actions: list[FamilyAction] = []
    for candidate_scope in scopes:
        records = load_family_archive_records(archive_root, candidate_scope)
        families.extend(records.families)
        members.extend(records.family_members)
        actions.extend(records.family_actions)
    family_ids: dict[str, LogbookFamily] = {}
    for family in families:
        previous = family_ids.get(family.family_id)
        if previous is not None and previous != family:
            raise ValueError(
                f"Family {family.family_id} has conflicting archived records."
            )
        family_ids[family.family_id] = family
    return FamilyArchiveRecords(tuple(families), tuple(members), tuple(actions))


def _archived_builds(archive_root: Path) -> dict[str, LogbookRow]:
    return {
        row.build_id: row
        for archive in _archive_files(archive_root)
        for row in load_logbook_file(archive)
    }


def _scope_build_archive(archive_root: Path, scope: str) -> Path:
    return archive_root / Path(*scope.split("/")).with_suffix(".jsonl")


def _print_json_lines(values: list[dict[str, Any]]) -> None:
    for value in values:
        print(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )


def _public_build_mapping(row: LogbookRow) -> dict[str, Any]:
    mapping = row.to_mapping()
    mapping.pop("gate_verdicts")
    mapping.pop("cost_usd")
    mapping.pop("row_format_version", None)
    if row.disposition not in {"published", "certified"}:
        mapping["artifact_location"] = None
    mapping.setdefault("requested_k", None)
    mapping.setdefault("realized_k", None)
    mapping.setdefault("record_unit", None)
    return mapping


def main(argv: list[str] | None = None) -> int:
    """Run one Logbook command and return its process exit code."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "family-export":
            if args.remote:
                records = _remote_family_records(args.scope)
                builds = _remote_rows(args.scope)
            else:
                records = load_family_spool(args.source)
                build_archive = _scope_build_archive(args.archive_root, args.scope)
                archived_builds = (
                    load_logbook_file(build_archive) if build_archive.is_file() else ()
                )
                builds = (*archived_builds, *load_spool_rows(args.source))
            receipts = export_family_scope(
                args.archive_root,
                scope=args.scope,
                builds=builds,
                families=records.families,
                family_members=records.family_members,
                family_actions=records.family_actions,
            )
            print(
                "exported family records for "
                f"{args.scope}: "
                + ", ".join(
                    f"{record_type}={receipt.appended} new/{receipt.existing} existing"
                    for record_type, receipt in receipts.items()
                )
            )
            return 0

        if args.command == "family-import":
            records = load_family_archive_records(
                args.archive_root,
                args.scope,
            )
            build_archive = _scope_build_archive(args.archive_root, args.scope)
            if records.family_members and not build_archive.is_file():
                raise ValueError(
                    f"Family members for {args.scope} require build archive "
                    f"{build_archive}."
                )
            builds = load_logbook_file(build_archive) if build_archive.is_file() else ()
            builds_by_id = {build.build_id: build for build in builds}
            families_by_id = {family.family_id: family for family in records.families}
            for member in records.family_members:
                try:
                    family = families_by_id[member.family_id]
                    build = builds_by_id[member.build_id]
                except KeyError as exc:
                    raise ValueError(
                        f"Family member import is missing archived record "
                        f"{exc.args[0]}."
                    ) from exc
                validate_family_membership(family, member, build)
            spool_build_rows(builds, spool_dir=args.spool)
            imported = import_family_scope(
                args.archive_root,
                scope=args.scope,
                spool_dir=args.spool,
                builds=builds,
            )
            print(
                f"imported family records for {args.scope}: "
                f"builds={len(builds)}, families={len(imported.families)}, "
                f"members={len(imported.family_members)}, "
                f"actions={len(imported.family_actions)}"
            )
            return 0

        if args.command == "reconcile":
            receipt = reconcile_logbook_spool(args.spool)
            print(
                "reconciled Logbook spool: "
                f"builds={receipt.builds.posted} posted/"
                f"{receipt.builds.retained} retained; "
                f"family records={receipt.families.posted} posted/"
                f"{receipt.families.retained} retained"
            )
            errors = (*receipt.builds.errors, *receipt.families.errors)
            if errors:
                raise RuntimeError("; ".join(errors))
            return 0

        if args.command == "list-families":
            records = _archived_family_records(
                args.archive_root,
                scope=args.scope,
            )
            _print_json_lines(
                [
                    family.to_mapping()
                    for family in sorted(
                        records.families,
                        key=lambda value: (value.chain_scope, value.family_id),
                    )
                ]
            )
            return 0

        if args.command == "list-family-builds":
            records = _archived_family_records(args.archive_root)
            family = next(
                (
                    value
                    for value in records.families
                    if value.family_id == args.family_id
                ),
                None,
            )
            if family is None:
                raise ValueError(f"Unknown family_id {args.family_id}.")
            builds = _archived_builds(args.archive_root)
            selected: list[LogbookRow] = []
            for member in records.family_members:
                if member.family_id != family.family_id:
                    continue
                try:
                    selected.append(builds[member.build_id])
                except KeyError as exc:
                    raise ValueError(
                        f"Family member {member.build_id} has no archived build."
                    ) from exc
            selected.sort(
                key=lambda value: (
                    value.requested_k is None,
                    value.requested_k or 0,
                    value.build_id,
                )
            )
            _print_json_lines([_public_build_mapping(row) for row in selected])
            return 0

        if args.command == "show-family-history":
            records = _archived_family_records(args.archive_root)
            if not any(
                family.family_id == args.family_id for family in records.families
            ):
                raise ValueError(f"Unknown family_id {args.family_id}.")
            actions = sorted(
                (
                    action
                    for action in records.family_actions
                    if action.family_id == args.family_id
                ),
                key=lambda value: (value.recorded_at, value.action_id),
            )
            _print_json_lines([action.to_mapping() for action in actions])
            return 0

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
