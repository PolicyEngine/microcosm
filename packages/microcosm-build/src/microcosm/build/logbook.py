"""Durable, append-only Logbook client for terminal build attempts.

``record_build_attempt`` is the adoption seam owned by microcosm#665/#666. Callers
provide a complete terminal attempt receipt and the digest of the chain head
they are extending. Logbook validates and hashes the receipt, durably writes
``<row_digest>.json`` beneath the caller's spool directory, and only then makes
a best-effort Supabase REST insert.

Remote availability is never part of build correctness. If either
``LOGBOOK_URL`` or ``LOGBOOK_KEY`` is absent, the validated row stays in
spool-only mode without error. Network and HTTP failures are returned as
receipt data and never raised. Local validation and durable-spool failures
remain fatal: callers must not claim an attempt was recorded if its local row
was invalid or could not be persisted.

The ledger-era spellings (``POPULACE_LEDGER_URL``, ``POPULACE_LEDGER_KEY``,
``POPULACE_LEDGER_API_KEY``) are still honored for the Logbook dual-read
window (microcosm#632) and warn once per process; see
:mod:`microcosm.build.logbook_env`.

The caller owns chain coordination and must provide ``prev_row_digest`` for
the ledger head it extends. The row digest is

``sha256(canonical JSON of all non-chain fields || prev_row_digest)``

where the predecessor is lowercase ASCII, or the empty string for genesis.
Writes use the house pattern: file fsync, atomic rename, then containing-
directory fsync. ``reconcile_spool`` retries rows in predecessor order with
PostgREST ignore-duplicate semantics and removes only server-acknowledged rows.
Export local rows before reconciliation; if that ordering is missed,
``tools/logbook.py export --remote`` recovers the private rows with a
distinct read-only exporter credential.

The Supabase key must identify the migration's ``logbook_writer`` role, not
the service role. Hosted Supabase projects should additionally provide the
project gateway key as ``LOGBOOK_API_KEY``; single-key deployments may
omit it. The ``logbook`` schema must also be enabled in the hosted project's
PostgREST exposed-schema setting. The US stacked driver and the three UK
drivers record through this seam; this module remains driver-agnostic.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from microcosm.build.logbook_env import (
    LEGACY_API_KEY_ENV,
    LOGBOOK_API_KEY_ENV,
    LOGBOOK_KEY_ENV,
    LOGBOOK_URL_ENV,
    logbook_env,
)

__all__ = [
    "BUILD_DISPOSITIONS",
    "LOGBOOK_ROW_FIELDS",
    "LOGBOOK_RUNGS",
    "LogbookExportResult",
    "LogbookRow",
    "LogbookWriteResult",
    "ReconcileResult",
    "canonical_json_bytes",
    "compute_row_digest",
    "export_rows",
    "load_logbook_file",
    "load_spool_rows",
    "load_logbook_row",
    "order_rows_by_chain",
    "reconcile_spool",
    "record_build_attempt",
    "render_markdown",
]


BUILD_DISPOSITIONS = frozenset(
    {
        "iterating",
        "billed",
        "published",
        "certified",
        "failed",
        "superseded",
        "discarded",
    }
)
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BUILD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
LOGBOOK_RUNGS = frozenset({"f001", "f004", "f010", "f025", "f100"})
#: Ledger-era name for the project gateway key, kept because callers and
#: operator runbooks still name it. Reads go through
#: :func:`microcosm.build.logbook_env.logbook_env`, which prefers
#: ``LOGBOOK_API_KEY`` and warns once when it falls back to this one.
LEDGER_API_KEY_ENV = LEGACY_API_KEY_ENV
LOGBOOK_ROW_FIELDS = frozenset(
    {
        "build_id",
        "ts",
        "pipeline",
        "rung",
        "seed",
        "code_pin",
        "input_pins_digest",
        "identity_digest",
        "phases_reached",
        "gate_verdicts",
        "wall_seconds",
        "cost_usd",
        "artifact_location",
        "disposition",
        "prediction_id",
        "prev_row_digest",
        "row_digest",
    }
)
_HASH_EXCLUDED_FIELDS = frozenset({"prev_row_digest", "row_digest"})
_ARCHIVE_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_ARCHIVE_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class LogbookRow:
    """A validated terminal build-attempt row with a verified chain digest."""

    build_id: str
    ts: str
    pipeline: str
    rung: str
    seed: int | None
    code_pin: str
    input_pins_digest: str
    identity_digest: str
    phases_reached: tuple[str, ...]
    gate_verdicts: dict[str, Any]
    wall_seconds: int | float | None
    cost_usd: int | float | None
    artifact_location: str | None
    disposition: str
    prediction_id: str | None
    prev_row_digest: str | None
    row_digest: str

    @classmethod
    def create(
        cls,
        *,
        build_id: str,
        ts: str | datetime,
        pipeline: str,
        rung: str,
        seed: int | None,
        code_pin: str,
        input_pins_digest: str,
        identity_digest: str,
        phases_reached: Sequence[str],
        gate_verdicts: Mapping[str, Any],
        wall_seconds: int | float | None,
        cost_usd: int | float | None,
        artifact_location: str | None,
        disposition: str,
        prediction_id: str | None,
        prev_row_digest: str | None,
        row_digest: str | None = None,
    ) -> LogbookRow:
        """Validate, normalize, and hash a complete attempt receipt."""

        normalized_build_id = _nonempty_text(build_id, "build_id")
        if not _BUILD_ID_PATTERN.fullmatch(normalized_build_id):
            raise ValueError(
                "build_id must use only letters, digits, '.', '_', ':', or '-'."
            )
        normalized_pipeline = _nonempty_text(pipeline, "pipeline")
        normalized_rung = _validate_rung(rung)
        normalized_seed = _validate_seed(seed)
        normalized_code_pin = _nonempty_text(code_pin, "code_pin")
        normalized_input_digest = _validate_digest(
            input_pins_digest,
            "input_pins_digest",
            nullable=False,
        )
        normalized_identity_digest = _validate_digest(
            identity_digest,
            "identity_digest",
            nullable=False,
        )
        normalized_phases = _validate_phases(phases_reached)
        normalized_gates = _validate_gate_verdicts(gate_verdicts)
        normalized_wall = _validate_nonnegative_number(wall_seconds, "wall_seconds")
        normalized_cost = _validate_nonnegative_number(cost_usd, "cost_usd")
        normalized_artifact = _optional_text(artifact_location, "artifact_location")
        if disposition not in BUILD_DISPOSITIONS:
            raise ValueError(
                f"disposition must be one of {sorted(BUILD_DISPOSITIONS)}, "
                f"got {disposition!r}."
            )
        if disposition in {"published", "certified"} and normalized_artifact is None:
            raise ValueError(
                f"artifact_location is required for disposition {disposition!r}."
            )
        normalized_prediction = _optional_text(prediction_id, "prediction_id")
        normalized_prev = _validate_digest(
            prev_row_digest,
            "prev_row_digest",
            nullable=True,
        )

        values: dict[str, Any] = {
            "build_id": normalized_build_id,
            "ts": _normalize_timestamp(ts, "ts"),
            "pipeline": normalized_pipeline,
            "rung": normalized_rung,
            "seed": normalized_seed,
            "code_pin": normalized_code_pin,
            "input_pins_digest": normalized_input_digest,
            "identity_digest": normalized_identity_digest,
            "phases_reached": list(normalized_phases),
            "gate_verdicts": normalized_gates,
            "wall_seconds": normalized_wall,
            "cost_usd": normalized_cost,
            "artifact_location": normalized_artifact,
            "disposition": disposition,
            "prediction_id": normalized_prediction,
            "prev_row_digest": normalized_prev,
        }
        calculated = compute_row_digest(values)
        if row_digest is not None:
            supplied = _validate_digest(row_digest, "row_digest", nullable=False)
            if supplied != calculated:
                raise ValueError(
                    f"row_digest for build {normalized_build_id!r} does not match "
                    f"the canonical row: {supplied} != {calculated}."
                )

        return cls(
            build_id=normalized_build_id,
            ts=values["ts"],
            pipeline=normalized_pipeline,
            rung=normalized_rung,
            seed=normalized_seed,
            code_pin=normalized_code_pin,
            input_pins_digest=normalized_input_digest,
            identity_digest=normalized_identity_digest,
            phases_reached=normalized_phases,
            gate_verdicts=normalized_gates,
            wall_seconds=normalized_wall,
            cost_usd=normalized_cost,
            artifact_location=normalized_artifact,
            disposition=disposition,
            prediction_id=normalized_prediction,
            prev_row_digest=normalized_prev,
            row_digest=calculated,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LogbookRow:
        """Load a row object, rejecting missing, extra, or tampered fields."""

        if not isinstance(value, Mapping):
            raise ValueError(
                f"Logbook row must be an object, got {type(value).__name__}."
            )
        keys = frozenset(value)
        if keys != LOGBOOK_ROW_FIELDS:
            missing = sorted(LOGBOOK_ROW_FIELDS - keys)
            extra = sorted(keys - LOGBOOK_ROW_FIELDS)
            raise ValueError(
                f"Logbook row schema mismatch; missing={missing}, extra={extra}."
            )
        return cls.create(**dict(value))

    def to_mapping(self) -> dict[str, Any]:
        """Return the normalized JSON row in #628 database-column order."""

        mapping = {
            "build_id": self.build_id,
            "ts": self.ts,
            "pipeline": self.pipeline,
            "rung": self.rung,
            "seed": self.seed,
            "code_pin": self.code_pin,
            "input_pins_digest": self.input_pins_digest,
            "identity_digest": self.identity_digest,
            "phases_reached": list(self.phases_reached),
            "gate_verdicts": deepcopy(self.gate_verdicts),
            "wall_seconds": self.wall_seconds,
            "cost_usd": self.cost_usd,
            "artifact_location": self.artifact_location,
            "disposition": self.disposition,
            "prediction_id": self.prediction_id,
            "prev_row_digest": self.prev_row_digest,
            "row_digest": self.row_digest,
        }
        # ``frozen=True`` prevents attribute replacement but a caller can
        # still mutate nested JSON containers.  Re-authenticate immediately
        # before every serialization boundary so neither such a mutation nor
        # direct dataclass construction can persist an invalid row.
        LogbookRow.from_mapping(mapping)
        return mapping

    def to_json_line(self) -> str:
        """Serialize one portable compact JSON row."""

        return (
            json.dumps(
                self.to_mapping(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(frozen=True)
class LogbookWriteResult:
    """Receipt for one durable local write and best-effort remote insert."""

    row: LogbookRow
    spool_path: Path
    posted: bool = False
    remote_error: str | None = None


@dataclass(frozen=True)
class ReconcileResult:
    """Receipt for one ordered spool reconciliation pass."""

    attempted: int
    posted: int
    retained: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class LogbookExportResult:
    """Receipt for one content-append to the git archive."""

    existing: int
    appended: int
    tail_digest: str | None


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON using #628's deterministic Python/PostgreSQL form.

    Object keys are lexicographically sorted, whitespace is omitted, Unicode
    remains UTF-8, and every number is rendered in plain base-10 form without
    insignificant trailing zeroes. Booleans remain distinct from integers.
    """

    try:
        return _canonical_json_text(value).encode("utf-8")
    except (UnicodeEncodeError, InvalidOperation) as exc:
        raise ValueError(f"Value is not canonical JSON: {exc}.") from exc


def compute_row_digest(value: Mapping[str, Any]) -> str:
    """Compute SHA-256(canonical non-chain fields || predecessor)."""

    missing = (LOGBOOK_ROW_FIELDS - {"row_digest"}) - frozenset(value)
    if missing:
        raise ValueError(f"Cannot hash Logbook row; missing fields: {sorted(missing)}.")
    predecessor = _validate_digest(
        value.get("prev_row_digest"),
        "prev_row_digest",
        nullable=True,
    )
    payload = {
        key: value[key] for key in sorted(value) if key not in _HASH_EXCLUDED_FIELDS
    }
    material = canonical_json_bytes(payload) + (predecessor or "").encode("ascii")
    return hashlib.sha256(material).hexdigest()


def record_build_attempt(
    *,
    build_id: str,
    ts: str | datetime,
    pipeline: str,
    rung: str,
    seed: int | None,
    code_pin: str,
    input_pins_digest: str,
    identity_digest: str,
    phases_reached: Sequence[str],
    gate_verdicts: Mapping[str, Any],
    wall_seconds: int | float | None,
    cost_usd: int | float | None,
    artifact_location: str | None,
    disposition: str,
    prediction_id: str | None,
    prev_row_digest: str | None,
    row_digest: str | None = None,
    spool_dir: str | Path = "logbook-spool",
    timeout: float = 10.0,
) -> LogbookWriteResult:
    """Validate, durably spool, then best-effort insert one terminal attempt.

    The complete row, including ``prev_row_digest``, is supplied by the caller
    so chain coordination is explicit. A missing URL or key selects spool-only
    mode. Remote failures are reported in ``remote_error`` and never raised;
    validation and local persistence errors do raise.
    """

    row = LogbookRow.create(
        build_id=build_id,
        ts=ts,
        pipeline=pipeline,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        input_pins_digest=input_pins_digest,
        identity_digest=identity_digest,
        phases_reached=phases_reached,
        gate_verdicts=gate_verdicts,
        wall_seconds=wall_seconds,
        cost_usd=cost_usd,
        artifact_location=artifact_location,
        disposition=disposition,
        prediction_id=prediction_id,
        prev_row_digest=prev_row_digest,
        row_digest=row_digest,
    )
    spool_path = Path(spool_dir) / f"{row.row_digest}.json"
    _atomic_write_row(spool_path, row)
    config = _remote_config()
    if config is None:
        return LogbookWriteResult(row=row, spool_path=spool_path)
    posted, error = _post_build_row(
        row,
        ledger_url=config[0],
        ledger_key=config[1],
        ledger_api_key=config[2],
        timeout=timeout,
    )
    return LogbookWriteResult(
        row=row,
        spool_path=spool_path,
        posted=posted,
        remote_error=error,
    )


def load_logbook_row(path: str | Path) -> LogbookRow:
    """Load one completed spool row and authenticate its digest filename."""

    spool_path = Path(path)
    try:
        value = json.loads(spool_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Logbook spool row {spool_path}: {exc}.") from exc
    row = LogbookRow.from_mapping(value)
    if spool_path.suffix != ".json" or spool_path.stem != row.row_digest:
        raise ValueError(
            "Logbook spool filename does not match row_digest for "
            f"{row.build_id}: {spool_path.name}."
        )
    return row


def load_spool_rows(spool_dir: str | Path) -> tuple[LogbookRow, ...]:
    """Load and chain-order every completed JSON row in a spool directory."""

    directory = Path(spool_dir)
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise ValueError(f"Logbook spool path is not a directory: {directory}.")
    rows = tuple(load_logbook_row(path) for path in sorted(directory.glob("*.json")))
    return order_rows_by_chain(rows)


def order_rows_by_chain(
    rows: Sequence[LogbookRow],
) -> tuple[LogbookRow, ...]:
    """Return one complete chain or suffix, rejecting forks and disconnection."""

    if not rows:
        return ()
    by_digest: dict[str, LogbookRow] = {}
    build_ids: set[str] = set()
    successors: dict[str, LogbookRow] = {}
    for row in rows:
        if row.row_digest in by_digest:
            raise ValueError(f"Duplicate Logbook row_digest: {row.row_digest}.")
        if row.build_id in build_ids:
            raise ValueError(f"Duplicate Logbook build_id: {row.build_id}.")
        by_digest[row.row_digest] = row
        build_ids.add(row.build_id)
        predecessor = row.prev_row_digest
        if predecessor is not None:
            if predecessor in successors:
                first = successors[predecessor]
                raise ValueError(
                    "Logbook chain forks after "
                    f"{predecessor}: {first.build_id}, {row.build_id}."
                )
            successors[predecessor] = row

    roots = [
        row
        for row in rows
        if row.prev_row_digest is None or row.prev_row_digest not in by_digest
    ]
    if len(roots) != 1:
        raise ValueError(
            "Logbook rows must form one connected chain; "
            f"found {len(roots)} roots among {len(rows)} rows."
        )

    ordered: list[LogbookRow] = []
    current = roots[0]
    while True:
        ordered.append(current)
        successor = successors.get(current.row_digest)
        if successor is None:
            break
        current = successor
    if len(ordered) != len(rows):
        unseen = sorted(set(by_digest) - {row.row_digest for row in ordered})
        raise ValueError(
            "Logbook rows contain a cycle or disconnected component; "
            f"unseen digests={unseen}."
        )
    return tuple(ordered)


def reconcile_spool(
    spool_dir: str | Path = "logbook-spool",
    *,
    timeout: float = 10.0,
) -> ReconcileResult:
    """Retry a spool suffix in chain order and remove acknowledged rows only.

    Export the spool to the git archive first.  A read-only live-store export
    remains available to operators if reconciliation happens first.
    """

    rows = load_spool_rows(spool_dir)
    config = _remote_config()
    if config is None or not rows:
        return ReconcileResult(
            attempted=0,
            posted=0,
            retained=len(rows),
            errors=(),
        )

    directory = Path(spool_dir)
    attempted = 0
    posted = 0
    removed = 0
    errors: list[str] = []
    for row in rows:
        attempted += 1
        success, error = _post_build_row(
            row,
            ledger_url=config[0],
            ledger_key=config[1],
            ledger_api_key=config[2],
            timeout=timeout,
        )
        if not success:
            errors.append(f"{row.build_id}: {error or 'remote insert failed'}")
            break
        posted += 1
        path = directory / f"{row.row_digest}.json"
        try:
            path.unlink()
            removed += 1
            _fsync_parent_directory(directory)
        except OSError as exc:
            errors.append(
                f"{row.build_id}: remote insert succeeded but spool cleanup "
                f"failed: {exc}"
            )
            break
    return ReconcileResult(
        attempted=attempted,
        posted=posted,
        retained=len(rows) - removed,
        errors=tuple(errors),
    )


def load_logbook_file(path: str | Path) -> tuple[LogbookRow, ...]:
    """Load and validate a genesis-rooted Logbook JSONL archive."""

    archive = Path(path)
    if not archive.exists():
        raise ValueError(f"Logbook archive does not exist: {archive}.")
    rows = _load_jsonl_rows(archive)
    _validate_ordered_chain(rows, expected_predecessor=None)
    return rows


def export_rows(
    path: str | Path,
    candidates: Sequence[LogbookRow],
) -> LogbookExportResult:
    """Content-append a verified chain suffix, failing closed on divergence."""

    archive = Path(path)
    with _exclusive_archive_lock(archive):
        return _export_rows_locked(archive, candidates)


def _export_rows_locked(
    archive: Path,
    candidates: Sequence[LogbookRow],
) -> LogbookExportResult:
    """Append while the archive's stable parent-directory lock is held."""

    if archive.exists():
        original = archive.read_bytes()
        existing = _load_jsonl_bytes(archive, original)
        _validate_ordered_chain(existing, expected_predecessor=None)
    else:
        existing = ()
        original = b""

    ordered = order_rows_by_chain(candidates)
    existing_by_id = {row.build_id: row for row in existing}
    new_rows: list[LogbookRow] = []
    saw_new = False
    for row in ordered:
        archived = existing_by_id.get(row.build_id)
        if archived is not None:
            if saw_new:
                raise ValueError(
                    f"Logbook candidate chain returns to archived row {row.build_id} "
                    "after its new suffix begins."
                )
            if archived != row:
                raise ValueError(
                    f"Logbook candidate diverges from archived row {row.build_id}."
                )
            continue
        saw_new = True
        new_rows.append(row)

    predecessor = existing[-1].row_digest if existing else None
    _validate_ordered_chain(tuple(new_rows), expected_predecessor=predecessor)
    if new_rows:
        if original and not original.endswith(b"\n"):
            raise ValueError(f"Logbook archive {archive} does not end with a newline.")
        addition = "".join(row.to_json_line() for row in new_rows).encode("utf-8")
        _atomic_write_bytes(archive, original + addition)
    elif archive.exists():
        # A previous atomic replacement may have become visible before its
        # parent-directory fsync failed.  An exact retry must complete that
        # durability boundary instead of returning early.
        _fsync_file_and_parent(archive)

    tail = new_rows[-1].row_digest if new_rows else predecessor
    return LogbookExportResult(
        existing=len(existing),
        appended=len(new_rows),
        tail_digest=tail,
    )


def render_markdown(
    rows: Sequence[LogbookRow],
    *,
    rung: str | None = None,
    dispositions: set[str] | None = None,
) -> str:
    """Render the public-safe Logbook projection as a Markdown table."""

    if rung is not None:
        _validate_rung(rung)
    if dispositions is not None:
        invalid = dispositions - BUILD_DISPOSITIONS
        if invalid:
            raise ValueError(f"Unknown Logbook dispositions: {sorted(invalid)}.")
    selected = [
        row
        for row in rows
        if (rung is None or row.rung == rung)
        and (dispositions is None or row.disposition in dispositions)
    ]
    lines = [
        "| Timestamp (UTC) | Build | Pipeline | Rung | Disposition | Artifact |",
        "|---|---|---|---|---|---|",
    ]
    for row in selected:
        public_artifact = (
            row.artifact_location
            if row.disposition in {"published", "certified"}
            else None
        )
        values = (
            row.ts,
            row.build_id,
            row.pipeline,
            row.rung,
            row.disposition,
            public_artifact or "—",
        )
        lines.append(
            "| " + " | ".join(_markdown_cell(value) for value in values) + " |"
        )
    return "\n".join(lines) + "\n"


def _load_jsonl_rows(path: Path) -> tuple[LogbookRow, ...]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read Logbook rows from {path}: {exc}.") from exc
    return _load_jsonl_bytes(path, content)


def _load_jsonl_bytes(path: Path, content: bytes) -> tuple[LogbookRow, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Cannot read Logbook rows from {path}: {exc}.") from exc
    if not text:
        return ()
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    rows: list[LogbookRow] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(
                f"Invalid Logbook row {line_number} (blank) in {path}: "
                "blank lines are not allowed."
            )
        build_id = "unknown"
        try:
            value = json.loads(line)
            if isinstance(value, Mapping):
                build_id = str(value.get("build_id", build_id))
            rows.append(LogbookRow.from_mapping(value))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Logbook row {line_number} ({build_id}) in {path}: {exc}"
            ) from exc
    return tuple(rows)


def _validate_ordered_chain(
    rows: Sequence[LogbookRow],
    *,
    expected_predecessor: str | None,
) -> None:
    predecessor = expected_predecessor
    build_ids: set[str] = set()
    for position, row in enumerate(rows, start=1):
        if row.build_id in build_ids:
            raise ValueError(
                f"Logbook chain repeats build_id {row.build_id} at row {position}."
            )
        if row.prev_row_digest != predecessor:
            context = "genesis" if predecessor is None else "current archive tail"
            raise ValueError(
                f"Logbook chain diverges at row {position} ({row.build_id}): "
                f"prev_row_digest={row.prev_row_digest!r}, expected {context} "
                f"{predecessor!r}."
            )
        build_ids.add(row.build_id)
        predecessor = row.row_digest


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


@contextmanager
def _exclusive_archive_lock(path: Path) -> Iterator[None]:
    """Serialize atomic archive replacements without leaving a lock artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.parent.resolve()
    with _ARCHIVE_THREAD_LOCKS_GUARD:
        thread_lock = _ARCHIVE_THREAD_LOCKS.setdefault(
            directory,
            threading.Lock(),
        )
    with thread_lock:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuse redirects so credential headers never cross an origin boundary."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


_NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler())


def urlopen(request: Request, *, timeout: float) -> Any:
    """Open one Logbook request without following HTTP redirects."""

    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def _remote_config() -> tuple[str, str, str] | None:
    url = logbook_env(LOGBOOK_URL_ENV)
    key = logbook_env(LOGBOOK_KEY_ENV)
    if not url or not key:
        return None
    api_key = logbook_env(LOGBOOK_API_KEY_ENV) or key
    return url, key, api_key


def _validate_remote_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Logbook URL must have a host and no embedded credentials")
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("Logbook URL must use HTTPS (HTTP is loopback-only)")


def _builds_endpoint(url: str) -> str:
    _validate_remote_url(url)
    base = url.rstrip("/")
    if base.endswith("/rest/v1/builds"):
        endpoint = base
    elif base.endswith("/rest/v1"):
        endpoint = f"{base}/builds"
    else:
        endpoint = f"{base}/rest/v1/builds"
    return f"{endpoint}?{urlencode({'on_conflict': 'build_id'})}"


def _post_build_row(
    row: LogbookRow,
    *,
    ledger_url: str,
    ledger_key: str,
    ledger_api_key: str,
    timeout: float,
) -> tuple[bool, str | None]:
    if timeout <= 0:
        return False, "timeout must be greater than zero"
    try:
        request = Request(
            _builds_endpoint(ledger_url),
            data=row.to_json_line().rstrip("\n").encode("utf-8"),
            method="POST",
            headers={
                "apikey": ledger_api_key,
                "Authorization": f"Bearer {ledger_key}",
                "Content-Profile": "logbook",
                "Content-Type": "application/json",
                "Prefer": "resolution=ignore-duplicates,return=minimal",
            },
        )
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if not 200 <= status < 300:
                return False, f"Supabase returned HTTP {status}"
    except HTTPError as exc:
        try:
            body = exc.read(1_024).decode("utf-8", errors="replace").strip()
        except OSError:
            body = ""
        detail = f": {body}" if body else ""
        return False, f"Supabase returned HTTP {exc.code}{detail}"
    except OSError as exc:
        return False, f"Supabase insert failed: {exc}"
    except Exception as exc:  # pragma: no cover - defensive build-safety boundary
        return False, f"Supabase insert failed: {type(exc).__name__}: {exc}"
    return True, None


def _canonical_json_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON cannot contain non-finite numbers.")
        return _canonical_decimal(Decimal(str(value)))
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(
                "Canonical JSON cannot contain NUL characters; PostgreSQL "
                "text and jsonb cannot store them."
            )
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Canonical JSON object keys must be strings.")
        return (
            "{"
            + ",".join(
                f"{_canonical_json_text(key)}:{_canonical_json_text(value[key])}"
                for key in sorted(value)
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_json_text(item) for item in value) + "]"
    raise ValueError(
        f"Canonical JSON does not support values of type {type(value).__name__}."
    )


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("Canonical JSON cannot contain non-finite numbers.")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _normalize_timestamp(value: str | datetime, field: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"{field} must be an ISO-8601 timestamp: {value!r}."
            ) from exc
    else:
        raise ValueError(
            f"{field} must be an ISO-8601 string or datetime, got "
            f"{type(value).__name__}."
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset.")
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _validate_rung(value: str) -> str:
    if not isinstance(value, str) or value not in LOGBOOK_RUNGS:
        raise ValueError(
            "rung must be a #624 fraction token: 'f001', 'f004', 'f010', 'f025', or 'f100'."
        )
    return value


def _validate_seed(value: int | None) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**63 - 1
    ):
        raise ValueError(
            f"seed must be a non-negative signed 64-bit integer or null, got {value!r}."
        )
    return value


def _validate_phases(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("phases_reached must be a non-empty array of phase names.")
    phases = tuple(_nonempty_text(item, "phases_reached item") for item in value)
    if not phases:
        raise ValueError("phases_reached must contain at least one phase.")
    if len(set(phases)) != len(phases):
        raise ValueError("phases_reached must not contain duplicate phase names.")
    return phases


def _validate_gate_verdicts(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("gate_verdicts must be a non-empty JSON object.")
    normalized = _normalize_json_value(value)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise AssertionError("gate_verdicts normalization lost its object shape")
    for gate, receipt in normalized.items():
        _nonempty_text(gate, "gate_verdicts gate name")
        if not isinstance(receipt, Mapping):
            raise ValueError(f"gate_verdicts[{gate!r}] must be an object.")
        _nonempty_text(receipt.get("verdict"), f"gate_verdicts[{gate!r}].verdict")
        _nonempty_text(receipt.get("receipt"), f"gate_verdicts[{gate!r}].receipt")
    canonical_json_bytes(normalized)
    return normalized


def _normalize_json_value(value: Any) -> Any:
    """Return JSON containers with tuples and mappings normalized for retries."""

    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Canonical JSON object keys must be strings.")
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(
        f"Canonical JSON does not support values of type {type(value).__name__}."
    )


def _validate_nonnegative_number(
    value: int | float | None,
    field: str,
) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-negative JSON number or null.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative, got {value!r}.")
    return value


def _validate_digest(
    value: Any,
    field: str,
    *,
    nullable: bool,
) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        suffix = " or null" if nullable else ""
        raise ValueError(f"{field} must be a lowercase SHA-256 digest{suffix}.")
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL characters.")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace.")
    return value


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, field)


def _atomic_write_row(path: Path, row: LogbookRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_logbook_row(path)
        if existing != row:
            raise ValueError(
                f"Logbook spool digest collision or divergent retry at {path}."
            )
        # Complete a possibly interrupted replacement whose file became
        # visible before the parent-directory fsync succeeded.
        _fsync_file_and_parent(path)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(row.to_json_line())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably replace one file with exact bytes using the house pattern."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file_and_parent(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_parent_directory(path.parent)


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
