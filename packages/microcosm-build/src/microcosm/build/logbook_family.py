"""Typed, durable records for Logbook dataset families and later decisions."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request

from .logbook import (
    DECLARED_LOGBOOK_SCOPES,
    LogbookRow,
    ReconcileResult,
    _atomic_write_bytes,
    _fsync_file_and_parent,
    _fsync_parent_directory,
    _nonempty_text,
    _normalize_timestamp,
    _optional_text,
    _remote_config,
    _validate_digest,
    _validate_remote_url,
    load_logbook_file,
    load_spool_rows,
    logbook_chain_scope,
    reconcile_spool,
    urlopen,
)

__all__ = [
    "FAMILY_ACTION_TYPES",
    "FamilyAction",
    "FamilyArchiveRecords",
    "FamilyExportResult",
    "FamilyMember",
    "FamilyReconcileResult",
    "FamilyWriteResult",
    "LogbookFamily",
    "LogbookReconcileResult",
    "derive_family_id",
    "export_family_records",
    "export_family_scope",
    "family_archive_path",
    "import_family_scope",
    "load_family_actions",
    "load_family_archive_records",
    "load_family_members",
    "load_family_spool",
    "load_families",
    "reconcile_family_spool",
    "reconcile_logbook_spool",
    "record_family",
    "record_family_action",
    "record_family_member",
    "validate_family_action",
    "validate_family_archive_records",
    "validate_family_membership",
    "validate_family_source",
]


FAMILY_ACTION_TYPES = frozenset({"revokes", "supersedes"})
# uuid5(NAMESPACE_URL, "https://policyengine.org/microcosm/logbook-family")
_FAMILY_ID_NAMESPACE = uuid.UUID("67c736a3-4a56-5c31-9cb6-37ef0a014645")
_BUILD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_SPOOL_DIRECTORIES = {
    "families": "families",
    "family_members": "family_members",
    "family_actions": "family_actions",
}


class FamilyRecord(Protocol):
    def to_mapping(self) -> dict[str, Any]: ...

    def to_json_line(self) -> str: ...


@dataclass(frozen=True)
class LogbookFamily:
    family_id: str
    chain_scope: str
    source_pool_sha256: str

    @classmethod
    def create(
        cls,
        *,
        chain_scope: str,
        source_pool_sha256: str,
        family_id: str | None = None,
    ) -> LogbookFamily:
        parsed_scope = _nonempty_text(chain_scope, "chain_scope")
        if parsed_scope not in DECLARED_LOGBOOK_SCOPES:
            raise ValueError(
                f"chain_scope must be one of {sorted(DECLARED_LOGBOOK_SCOPES)}, "
                f"got {parsed_scope!r}."
            )
        parsed_source = _validate_digest(
            source_pool_sha256,
            "source_pool_sha256",
            nullable=False,
        )
        assert parsed_source is not None
        derived_id = derive_family_id(parsed_scope, parsed_source)
        if family_id is not None:
            parsed_id = _canonical_uuid(family_id, "family_id")
            if parsed_id != derived_id:
                raise ValueError(
                    f"family_id must be {derived_id} for chain_scope "
                    f"{parsed_scope!r} and source_pool_sha256 {parsed_source}."
                )
        return cls(
            family_id=derived_id,
            chain_scope=parsed_scope,
            source_pool_sha256=parsed_source,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LogbookFamily:
        _exact_keys(
            value,
            {"family_id", "chain_scope", "source_pool_sha256"},
            "family",
        )
        return cls.create(**dict(value))

    def to_mapping(self) -> dict[str, Any]:
        mapping = {
            "family_id": self.family_id,
            "chain_scope": self.chain_scope,
            "source_pool_sha256": self.source_pool_sha256,
        }
        LogbookFamily.from_mapping(mapping)
        return mapping

    def to_json_line(self) -> str:
        return _json_line(self.to_mapping())


def derive_family_id(chain_scope: str, source_pool_sha256: str) -> str:
    """Return the stable UUID for one scope and prepared-input manifest."""
    parsed_scope = _nonempty_text(chain_scope, "chain_scope")
    if parsed_scope not in DECLARED_LOGBOOK_SCOPES:
        raise ValueError(
            f"chain_scope must be one of {sorted(DECLARED_LOGBOOK_SCOPES)}, "
            f"got {parsed_scope!r}."
        )
    parsed_source = _validate_digest(
        source_pool_sha256,
        "source_pool_sha256",
        nullable=False,
    )
    assert parsed_source is not None
    return str(
        uuid.uuid5(
            _FAMILY_ID_NAMESPACE,
            f"{parsed_scope}\0{parsed_source}",
        )
    )


@dataclass(frozen=True)
class FamilyMember:
    family_id: str
    build_id: str

    @classmethod
    def create(cls, *, family_id: str, build_id: str) -> FamilyMember:
        parsed_build = _nonempty_text(build_id, "build_id")
        if not _BUILD_ID_PATTERN.fullmatch(parsed_build):
            raise ValueError(
                "build_id must use only letters, digits, '.', '_', ':', or '-'."
            )
        return cls(
            family_id=_canonical_uuid(family_id, "family_id"),
            build_id=parsed_build,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FamilyMember:
        _exact_keys(value, {"family_id", "build_id"}, "family member")
        return cls.create(**dict(value))

    def to_mapping(self) -> dict[str, Any]:
        mapping = {
            "family_id": self.family_id,
            "build_id": self.build_id,
        }
        FamilyMember.from_mapping(mapping)
        return mapping

    def to_json_line(self) -> str:
        return _json_line(self.to_mapping())


@dataclass(frozen=True)
class FamilyAction:
    action_id: str
    family_id: str
    build_id: str
    action_type: str
    related_build_id: str | None
    recorded_at: str
    actor: str
    reason: str
    evidence_location: str | None

    @classmethod
    def create(
        cls,
        *,
        action_id: str,
        family_id: str,
        build_id: str,
        action_type: str,
        related_build_id: str | None,
        recorded_at: str,
        actor: str,
        reason: str,
        evidence_location: str | None,
    ) -> FamilyAction:
        parsed_type = _nonempty_text(action_type, "action_type")
        if parsed_type not in FAMILY_ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {sorted(FAMILY_ACTION_TYPES)}, "
                f"got {parsed_type!r}."
            )
        parsed_build = FamilyMember.create(
            family_id=family_id,
            build_id=build_id,
        )
        parsed_related = (
            None
            if related_build_id is None
            else FamilyMember.create(
                family_id=family_id,
                build_id=related_build_id,
            ).build_id
        )
        if parsed_type == "revokes" and parsed_related is not None:
            raise ValueError("revokes action must not contain related_build_id.")
        if parsed_type == "supersedes":
            if parsed_related is None:
                raise ValueError("supersedes action requires related_build_id.")
            if parsed_related == parsed_build.build_id:
                raise ValueError("A build cannot supersede itself.")
        return cls(
            action_id=_canonical_uuid(action_id, "action_id"),
            family_id=parsed_build.family_id,
            build_id=parsed_build.build_id,
            action_type=parsed_type,
            related_build_id=parsed_related,
            recorded_at=_normalize_timestamp(recorded_at, "recorded_at"),
            actor=_nonempty_text(actor, "actor"),
            reason=_nonempty_text(reason, "reason"),
            evidence_location=_optional_text(
                evidence_location,
                "evidence_location",
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FamilyAction:
        _exact_keys(
            value,
            {
                "action_id",
                "family_id",
                "build_id",
                "action_type",
                "related_build_id",
                "recorded_at",
                "actor",
                "reason",
                "evidence_location",
            },
            "family action",
        )
        return cls.create(**dict(value))

    def to_mapping(self) -> dict[str, Any]:
        mapping = {
            "action_id": self.action_id,
            "family_id": self.family_id,
            "build_id": self.build_id,
            "action_type": self.action_type,
            "related_build_id": self.related_build_id,
            "recorded_at": self.recorded_at,
            "actor": self.actor,
            "reason": self.reason,
            "evidence_location": self.evidence_location,
        }
        FamilyAction.from_mapping(mapping)
        return mapping

    def to_json_line(self) -> str:
        return _json_line(self.to_mapping())


@dataclass(frozen=True)
class FamilyWriteResult:
    record: LogbookFamily | FamilyMember | FamilyAction
    spool_path: Path
    posted: bool = False
    remote_error: str | None = None


@dataclass(frozen=True)
class FamilyReconcileResult:
    attempted: int
    posted: int
    retained: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class LogbookReconcileResult:
    builds: ReconcileResult
    families: FamilyReconcileResult


@dataclass(frozen=True)
class FamilyExportResult:
    existing: int
    appended: int


@dataclass(frozen=True)
class FamilyArchiveRecords:
    families: tuple[LogbookFamily, ...]
    family_members: tuple[FamilyMember, ...]
    family_actions: tuple[FamilyAction, ...]


def validate_family_source(
    family: LogbookFamily,
    source_pool_sha256: str,
) -> None:
    supplied = _validate_digest(
        source_pool_sha256,
        "source_pool_sha256",
        nullable=False,
    )
    if supplied != family.source_pool_sha256:
        raise ValueError(
            f"Family {family.family_id} identifies source "
            f"{family.source_pool_sha256}, got {supplied}."
        )


def validate_family_membership(
    family: LogbookFamily,
    member: FamilyMember,
    build: LogbookRow,
) -> None:
    if member.family_id != family.family_id:
        raise ValueError(
            f"Membership family {member.family_id} does not match "
            f"family {family.family_id}."
        )
    if member.build_id != build.build_id:
        raise ValueError(
            f"Membership build {member.build_id} does not match build {build.build_id}."
        )
    build_scope = logbook_chain_scope(build.pipeline)
    if build_scope != family.chain_scope:
        raise ValueError(
            f"Build {build.build_id} scope {build_scope!r} does not match "
            f"family scope {family.chain_scope!r}."
        )


def validate_family_action(
    action: FamilyAction,
    *,
    members: Sequence[FamilyMember],
    builds: Mapping[str, LogbookRow],
) -> None:
    member_ids = {
        member.build_id for member in members if member.family_id == action.family_id
    }
    if action.build_id not in member_ids:
        raise ValueError(
            f"Build {action.build_id} is not a member of family {action.family_id}."
        )
    if action.action_type == "revokes":
        return
    assert action.related_build_id is not None
    if action.related_build_id not in member_ids:
        raise ValueError(
            f"Build {action.related_build_id} is not a member of family "
            f"{action.family_id}."
        )
    try:
        replacement = builds[action.build_id]
        replaced = builds[action.related_build_id]
    except KeyError as exc:
        raise ValueError(
            f"Missing build data for family action: {exc.args[0]}."
        ) from exc
    if (
        replacement.requested_k != replaced.requested_k
        or replacement.record_unit != replaced.record_unit
        or (
            replacement.requested_k is None
            and replacement.rung != replaced.rung
        )
    ):
        raise ValueError(
            "Superseding builds must have matching requested_k and record_unit; "
            "builds without cardinality must also have matching rung."
        )


def record_family(
    family: LogbookFamily,
    *,
    spool_dir: str | Path = "logbook-spool",
    timeout: float = 10.0,
    post_remote: bool = True,
) -> FamilyWriteResult:
    for existing in load_family_spool(spool_dir).families:
        if existing.family_id == family.family_id and existing != family:
            raise ValueError(
                f"Family spool contains a divergent retry for {family.family_id}."
            )
        if (
            existing.family_id != family.family_id
            and existing.chain_scope == family.chain_scope
            and existing.source_pool_sha256 == family.source_pool_sha256
        ):
            raise ValueError(
                f"Source {family.source_pool_sha256} in scope "
                f"{family.chain_scope} already belongs to family "
                f"{existing.family_id}."
            )
    return _record_family_value(
        family,
        record_type="families",
        key=family.family_id,
        conflict_fields=("family_id",),
        spool_dir=spool_dir,
        timeout=timeout,
        post_remote=post_remote,
    )


def record_family_member(
    member: FamilyMember,
    *,
    spool_dir: str | Path = "logbook-spool",
    timeout: float = 10.0,
    post_remote: bool = True,
) -> FamilyWriteResult:
    return _record_family_value(
        member,
        record_type="family_members",
        key=f"{member.family_id}--{member.build_id}",
        conflict_fields=("family_id", "build_id"),
        spool_dir=spool_dir,
        timeout=timeout,
        post_remote=post_remote,
    )


def record_family_action(
    action: FamilyAction,
    *,
    spool_dir: str | Path = "logbook-spool",
    timeout: float = 10.0,
    post_remote: bool = True,
) -> FamilyWriteResult:
    return _record_family_value(
        action,
        record_type="family_actions",
        key=action.action_id,
        conflict_fields=("action_id",),
        spool_dir=spool_dir,
        timeout=timeout,
        post_remote=post_remote,
    )


def reconcile_logbook_spool(
    spool_dir: str | Path = "logbook-spool",
    *,
    timeout: float = 10.0,
) -> LogbookReconcileResult:
    build_rows = load_spool_rows(spool_dir)
    family_records = load_family_spool(spool_dir)
    if any(
        (
            family_records.families,
            family_records.family_members,
            family_records.family_actions,
        )
    ) and _family_records_are_self_contained(family_records):
        validate_family_archive_records(family_records, builds=build_rows or None)
    build_result = reconcile_spool(spool_dir, timeout=timeout)
    if build_result.errors:
        family_result = _retained_family_result(spool_dir)
    else:
        family_result = reconcile_family_spool(
            spool_dir,
            timeout=timeout,
            builds=build_rows or None,
        )
    return LogbookReconcileResult(
        builds=build_result,
        families=family_result,
    )


def reconcile_family_spool(
    spool_dir: str | Path = "logbook-spool",
    *,
    timeout: float = 10.0,
    builds: Sequence[LogbookRow] | None = None,
) -> FamilyReconcileResult:
    directory = Path(spool_dir)
    queued = _load_spooled_family_records(directory)
    retained = sum(len(records) for _, records in queued)
    records = load_family_spool(directory)
    available_builds = (
        tuple(builds) if builds is not None else load_spool_rows(directory)
    )
    if retained and _family_records_are_self_contained(records):
        validate_family_archive_records(
            records,
            builds=available_builds or None,
        )
    config = _remote_config()
    if config is None or retained == 0:
        return FamilyReconcileResult(0, 0, retained, ())

    attempted = 0
    posted = 0
    removed = 0
    errors: list[str] = []
    for record_type, records in queued:
        for path, record in records:
            attempted += 1
            success, error = _post_family_value(
                record,
                record_type=record_type,
                conflict_fields=_conflict_fields(record_type),
                ledger_url=config[0],
                ledger_key=config[1],
                ledger_api_key=config[2],
                timeout=timeout,
            )
            if not success:
                errors.append(
                    f"{_record_key(record)}: {error or 'remote insert failed'}"
                )
                return FamilyReconcileResult(
                    attempted,
                    posted,
                    retained - removed,
                    tuple(errors),
                )
            posted += 1
            try:
                path.unlink()
                removed += 1
                _fsync_parent_directory(path.parent)
            except OSError as exc:
                errors.append(
                    f"{_record_key(record)}: remote insert succeeded but "
                    f"spool cleanup failed: {exc}"
                )
                return FamilyReconcileResult(
                    attempted,
                    posted,
                    retained - removed,
                    tuple(errors),
                )
    return FamilyReconcileResult(
        attempted,
        posted,
        retained - removed,
        tuple(errors),
    )


def load_families(path: str | Path) -> tuple[LogbookFamily, ...]:
    return _load_family_file(path, LogbookFamily)


def load_family_members(path: str | Path) -> tuple[FamilyMember, ...]:
    return _load_family_file(path, FamilyMember)


def load_family_actions(path: str | Path) -> tuple[FamilyAction, ...]:
    return _load_family_file(path, FamilyAction)


def load_family_spool(spool_dir: str | Path) -> FamilyArchiveRecords:
    queued = dict(_load_spooled_family_records(Path(spool_dir)))
    return FamilyArchiveRecords(
        families=tuple(
            record
            for _, record in queued["families"]
            if isinstance(record, LogbookFamily)
        ),
        family_members=tuple(
            record
            for _, record in queued["family_members"]
            if isinstance(record, FamilyMember)
        ),
        family_actions=tuple(
            record
            for _, record in queued["family_actions"]
            if isinstance(record, FamilyAction)
        ),
    )


def export_family_records[RecordT: (LogbookFamily, FamilyMember, FamilyAction)](
    path: str | Path,
    candidates: Sequence[RecordT],
) -> FamilyExportResult:
    archive = Path(path)
    if archive.exists():
        original = archive.read_bytes()
        if candidates:
            record_class = type(candidates[0])
            existing = _load_family_file(archive, record_class)
        elif original:
            raise ValueError(
                "Cannot infer family record type for a nonempty archive "
                "without candidates."
            )
        else:
            existing = ()
    else:
        original = b""
        existing = ()
    if original and not original.endswith(b"\n"):
        raise ValueError(f"Family archive {archive} does not end with a newline.")

    existing_by_key = {_record_key(record): record for record in existing}
    appended: list[RecordT] = []
    seen_new: dict[str, RecordT] = {}
    for record in candidates:
        key = _record_key(record)
        previous = existing_by_key.get(key)
        if previous is not None:
            if previous != record:
                raise ValueError(
                    f"Family archive record {key} conflicts with existing content."
                )
            continue
        current = seen_new.get(key)
        if current is not None:
            if current != record:
                raise ValueError(
                    f"Family archive candidates reuse {key} with different content."
                )
            continue
        seen_new[key] = record
        appended.append(record)

    if appended:
        addition = "".join(record.to_json_line() for record in appended).encode("utf-8")
        _atomic_write_bytes(archive, original + addition)
    elif archive.exists():
        _fsync_file_and_parent(archive)
    return FamilyExportResult(
        existing=len(existing),
        appended=len(appended),
    )


def family_archive_path(
    archive_root: str | Path,
    record_type: str,
    scope: str,
) -> Path:
    if record_type not in _SPOOL_DIRECTORIES:
        raise ValueError(f"Unknown family record type: {record_type}.")
    parsed_scope = _nonempty_text(scope, "scope")
    if parsed_scope not in DECLARED_LOGBOOK_SCOPES:
        raise ValueError(
            f"scope must be one of {sorted(DECLARED_LOGBOOK_SCOPES)}, "
            f"got {parsed_scope!r}."
        )
    scope_parts = parsed_scope.split("/")
    return Path(archive_root) / record_type / Path(*scope_parts).with_suffix(".jsonl")


def load_family_archive_records(
    archive_root: str | Path,
    scope: str,
) -> FamilyArchiveRecords:
    paths = {
        record_type: family_archive_path(archive_root, record_type, scope)
        for record_type in _SPOOL_DIRECTORIES
    }
    records = FamilyArchiveRecords(
        families=_load_optional_family_file(paths["families"], LogbookFamily),
        family_members=_load_optional_family_file(
            paths["family_members"],
            FamilyMember,
        ),
        family_actions=_load_optional_family_file(
            paths["family_actions"],
            FamilyAction,
        ),
    )
    validate_family_archive_records(records, scope=scope)
    return records


def export_family_scope(
    archive_root: str | Path,
    *,
    scope: str,
    builds: Sequence[LogbookRow],
    families: Sequence[LogbookFamily] = (),
    family_members: Sequence[FamilyMember] = (),
    family_actions: Sequence[FamilyAction] = (),
) -> dict[str, FamilyExportResult]:
    existing = load_family_archive_records(archive_root, scope)
    combined = FamilyArchiveRecords(
        families=_merge_archive_records(existing.families, families),
        family_members=_merge_archive_records(
            existing.family_members,
            family_members,
        ),
        family_actions=_merge_archive_records(
            existing.family_actions,
            family_actions,
        ),
    )
    validate_family_archive_records(combined, scope=scope, builds=builds)
    candidates_by_type = {
        "families": families,
        "family_members": family_members,
        "family_actions": family_actions,
    }
    existing_counts = {
        "families": len(existing.families),
        "family_members": len(existing.family_members),
        "family_actions": len(existing.family_actions),
    }
    return {
        record_type: (
            export_family_records(
                family_archive_path(archive_root, record_type, scope),
                candidates,
            )
            if candidates
            else FamilyExportResult(existing_counts[record_type], 0)
        )
        for record_type, candidates in candidates_by_type.items()
    }


def import_family_scope(
    archive_root: str | Path,
    *,
    scope: str,
    spool_dir: str | Path = "logbook-spool",
    builds: Sequence[LogbookRow] | None = None,
) -> FamilyArchiveRecords:
    records = load_family_archive_records(archive_root, scope)
    available_builds = tuple(builds) if builds is not None else _load_scope_builds(
        archive_root,
        scope,
        required=bool(records.family_members),
    )
    validate_family_archive_records(
        records,
        scope=scope,
        builds=available_builds,
    )
    for family in records.families:
        record_family(family, spool_dir=spool_dir, post_remote=False)
    for member in records.family_members:
        record_family_member(member, spool_dir=spool_dir, post_remote=False)
    for action in records.family_actions:
        record_family_action(action, spool_dir=spool_dir, post_remote=False)
    return records


def _record_family_value(
    record: LogbookFamily | FamilyMember | FamilyAction,
    *,
    record_type: str,
    key: str,
    conflict_fields: tuple[str, ...],
    spool_dir: str | Path,
    timeout: float,
    post_remote: bool,
) -> FamilyWriteResult:
    path = Path(spool_dir) / _SPOOL_DIRECTORIES[record_type] / f"{key}.json"
    _atomic_write_family_record(path, record)
    config = _remote_config()
    if config is None or not post_remote:
        return FamilyWriteResult(record=record, spool_path=path)
    posted, error = _post_family_value(
        record,
        record_type=record_type,
        conflict_fields=conflict_fields,
        ledger_url=config[0],
        ledger_key=config[1],
        ledger_api_key=config[2],
        timeout=timeout,
    )
    return FamilyWriteResult(
        record=record,
        spool_path=path,
        posted=posted,
        remote_error=error,
    )


def _post_family_value(
    record: LogbookFamily | FamilyMember | FamilyAction,
    *,
    record_type: str,
    conflict_fields: tuple[str, ...],
    ledger_url: str,
    ledger_key: str,
    ledger_api_key: str,
    timeout: float,
) -> tuple[bool, str | None]:
    if timeout <= 0:
        return False, "timeout must be greater than zero"
    try:
        endpoint = _table_endpoint(
            ledger_url,
            record_type,
            conflict_fields=conflict_fields,
        )
        request = Request(
            endpoint,
            data=record.to_json_line().rstrip("\n").encode("utf-8"),
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
    except Exception as exc:  # pragma: no cover
        return False, f"Supabase insert failed: {type(exc).__name__}: {exc}"
    return True, None


def _table_endpoint(
    url: str,
    table: str,
    *,
    conflict_fields: tuple[str, ...],
) -> str:
    _validate_remote_url(url)
    base = url.rstrip("/")
    if base.endswith("/rest/v1/builds"):
        base = base[: -len("/builds")]
    elif not base.endswith("/rest/v1"):
        base = f"{base}/rest/v1"
    query = urlencode({"on_conflict": ",".join(conflict_fields)})
    return f"{base}/{table}?{query}"


def _atomic_write_family_record(
    path: Path,
    record: LogbookFamily | FamilyMember | FamilyAction,
) -> None:
    content = record.to_json_line().encode("utf-8")
    if path.exists():
        try:
            existing = type(record).from_mapping(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid family spool record {path}: {exc}.") from exc
        if existing != record:
            raise ValueError(
                f"Family spool key collision or divergent retry at {path}."
            )
        _fsync_file_and_parent(path)
        return
    _atomic_write_bytes(path, content)


def _load_spooled_family_records(
    spool_dir: Path,
) -> tuple[
    tuple[
        str,
        tuple[
            tuple[Path, LogbookFamily | FamilyMember | FamilyAction],
            ...,
        ],
    ],
    ...,
]:
    definitions = (
        ("families", LogbookFamily),
        ("family_members", FamilyMember),
        ("family_actions", FamilyAction),
    )
    result = []
    for record_type, record_class in definitions:
        directory = spool_dir / _SPOOL_DIRECTORIES[record_type]
        records = []
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    record = record_class.from_mapping(value)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid family spool record {path}: {exc}."
                    ) from exc
                if path.stem != _record_key(record):
                    raise ValueError(
                        f"Family spool filename does not match record key: {path}."
                    )
                records.append((path, record))
        result.append((record_type, tuple(records)))
    return tuple(result)


def _retained_family_result(spool_dir: str | Path) -> FamilyReconcileResult:
    queued = _load_spooled_family_records(Path(spool_dir))
    retained = sum(len(records) for _, records in queued)
    return FamilyReconcileResult(0, 0, retained, ())


def _load_family_file[RecordT: (LogbookFamily, FamilyMember, FamilyAction)](
    path: str | Path,
    record_class: type[RecordT],
) -> tuple[RecordT, ...]:
    archive = Path(path)
    if not archive.exists():
        raise ValueError(f"Family archive does not exist: {archive}.")
    try:
        text = archive.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read family archive {archive}: {exc}.") from exc
    if not text:
        return ()
    lines = text.splitlines()
    records: list[RecordT] = []
    seen: dict[str, RecordT] = {}
    for position, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(
                f"Invalid blank family record at line {position} in {archive}."
            )
        try:
            record = record_class.from_mapping(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Invalid family record at line {position} in {archive}: {exc}."
            ) from exc
        key = _record_key(record)
        previous = seen.get(key)
        if previous is not None:
            if previous != record:
                raise ValueError(f"Family archive reuses {key} with different content.")
            raise ValueError(f"Family archive repeats record {key}.")
        seen[key] = record
        records.append(record)
    return tuple(records)


def _load_optional_family_file[RecordT: (LogbookFamily, FamilyMember, FamilyAction)](
    path: Path,
    record_class: type[RecordT],
) -> tuple[RecordT, ...]:
    if not path.exists():
        return ()
    return _load_family_file(path, record_class)


def _merge_archive_records[RecordT: (LogbookFamily, FamilyMember, FamilyAction)](
    existing: Sequence[RecordT],
    candidates: Sequence[RecordT],
) -> tuple[RecordT, ...]:
    merged = list(existing)
    by_key = {_record_key(record): record for record in existing}
    for record in candidates:
        key = _record_key(record)
        previous = by_key.get(key)
        if previous is not None:
            if previous != record:
                raise ValueError(
                    f"Family archive record {key} conflicts with existing content."
                )
            continue
        by_key[key] = record
        merged.append(record)
    return tuple(merged)


def validate_family_archive_records(
    records: FamilyArchiveRecords,
    *,
    scope: str | None = None,
    builds: Sequence[LogbookRow] | None = None,
) -> None:
    expected_scope = None
    if scope is not None:
        expected_scope = _nonempty_text(scope, "scope")
        if expected_scope not in DECLARED_LOGBOOK_SCOPES:
            raise ValueError(
                f"scope must be one of {sorted(DECLARED_LOGBOOK_SCOPES)}, "
                f"got {expected_scope!r}."
            )

    families: dict[str, LogbookFamily] = {}
    families_by_source: dict[tuple[str, str], LogbookFamily] = {}
    for family in records.families:
        family.to_mapping()
        previous = families.get(family.family_id)
        if previous is not None and previous != family:
            raise ValueError(
                f"Family {family.family_id} has conflicting records."
            )
        families[family.family_id] = family
        source_key = (family.chain_scope, family.source_pool_sha256)
        previous_source = families_by_source.get(source_key)
        if (
            previous_source is not None
            and previous_source.family_id != family.family_id
        ):
            raise ValueError(
                f"Source {family.source_pool_sha256} in scope "
                f"{family.chain_scope} belongs to multiple families."
            )
        families_by_source[source_key] = family

    wrong_scope = sorted(
        family.family_id
        for family in records.families
        if expected_scope is not None and family.chain_scope != expected_scope
    )
    if wrong_scope:
        raise ValueError(
            f"Family archive for scope {expected_scope} contains families "
            f"from another scope: {', '.join(wrong_scope)}."
        )

    members: dict[tuple[str, str], FamilyMember] = {}
    family_by_build: dict[str, str] = {}
    for member in records.family_members:
        member.to_mapping()
        if member.family_id not in families:
            raise ValueError(
                f"Family member {member.build_id} references missing family "
                f"{member.family_id}."
            )
        member_key = (member.family_id, member.build_id)
        previous = members.get(member_key)
        if previous is not None and previous != member:
            raise ValueError(
                f"Family member {member.family_id}/{member.build_id} has "
                "conflicting records."
            )
        members[member_key] = member
        previous_family = family_by_build.get(member.build_id)
        if previous_family is not None and previous_family != member.family_id:
            raise ValueError(
                f"Build {member.build_id} belongs to more than one family: "
                f"{previous_family} and {member.family_id}."
            )
        family_by_build[member.build_id] = member.family_id

    builds_by_id: dict[str, LogbookRow] | None = None
    if builds is not None:
        builds_by_id = {}
        for build in builds:
            previous = builds_by_id.get(build.build_id)
            if previous is not None and previous != build:
                raise ValueError(
                    f"Build {build.build_id} has conflicting records available "
                    "for family export."
                )
            builds_by_id[build.build_id] = build
        for member in records.family_members:
            build = builds_by_id.get(member.build_id)
            if build is None:
                raise ValueError(
                    f"Family member {member.family_id}/{member.build_id} "
                    f"references missing build {member.build_id}."
                )
            validate_family_membership(families[member.family_id], member, build)
    actions: dict[str, FamilyAction] = {}
    replacements: dict[tuple[str, str], FamilyAction] = {}
    replacement_edges: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for action in records.family_actions:
        action.to_mapping()
        previous_action = actions.get(action.action_id)
        if previous_action is not None:
            if previous_action != action:
                raise ValueError(
                    f"Family action {action.action_id} has conflicting records."
                )
            continue
        actions[action.action_id] = action
        if (action.family_id, action.build_id) not in members:
            raise ValueError(
                f"Family action {action.action_id} references missing member "
                f"{action.family_id}/{action.build_id}."
            )
        if (
            action.related_build_id is not None
            and (
                action.family_id,
                action.related_build_id,
            )
            not in members
        ):
            raise ValueError(
                f"Family action {action.action_id} references missing related "
                f"member {action.family_id}/{action.related_build_id}."
            )
        if action.action_type == "supersedes":
            assert action.related_build_id is not None
            replacement_key = (action.family_id, action.related_build_id)
            previous_replacement = replacements.get(replacement_key)
            if previous_replacement is not None:
                raise ValueError(
                    f"Build {action.related_build_id} has more than one direct "
                    f"replacement in family {action.family_id}."
                )
            replacements[replacement_key] = action
            replacement_edges.setdefault(
                (action.family_id, action.build_id),
                set(),
            ).add(replacement_key)
        if builds_by_id is not None:
            validate_family_action(
                action,
                members=tuple(members.values()),
                builds=builds_by_id,
            )

    _validate_replacement_graph(replacement_edges)


def _validate_replacement_graph(
    edges: Mapping[tuple[str, str], set[tuple[str, str]]],
) -> None:
    states: dict[tuple[str, str], int] = {}

    def visit(node: tuple[str, str]) -> None:
        state = states.get(node, 0)
        if state == 1:
            raise ValueError(
                f"Family {node[0]} contains a replacement cycle involving "
                f"build {node[1]}."
            )
        if state == 2:
            return
        states[node] = 1
        for related in edges.get(node, ()):
            visit(related)
        states[node] = 2

    for build in edges:
        visit(build)


def _family_records_are_self_contained(records: FamilyArchiveRecords) -> bool:
    family_ids = {family.family_id for family in records.families}
    members = {
        (member.family_id, member.build_id) for member in records.family_members
    }
    return all(
        member.family_id in family_ids for member in records.family_members
    ) and all(
        (
            (action.family_id, action.build_id) in members
            and (
                action.related_build_id is None
                or (action.family_id, action.related_build_id) in members
            )
        )
        for action in records.family_actions
    )


def _load_scope_builds(
    archive_root: str | Path,
    scope: str,
    *,
    required: bool,
) -> tuple[LogbookRow, ...]:
    build_archive = Path(archive_root) / Path(*scope.split("/")).with_suffix(".jsonl")
    if not build_archive.is_file():
        if required:
            raise ValueError(
                f"Family members for {scope} require build archive {build_archive}."
            )
        return ()
    return load_logbook_file(build_archive)


def _record_key(record: LogbookFamily | FamilyMember | FamilyAction) -> str:
    if isinstance(record, LogbookFamily):
        return record.family_id
    if isinstance(record, FamilyMember):
        return f"{record.family_id}--{record.build_id}"
    return record.action_id


def _conflict_fields(record_type: str) -> tuple[str, ...]:
    if record_type == "families":
        return ("family_id",)
    if record_type == "family_members":
        return ("family_id", "build_id")
    if record_type == "family_actions":
        return ("action_id",)
    raise ValueError(f"Unknown family record type: {record_type}.")


def _canonical_uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical UUID string.")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        raise ValueError(f"{field} must be a canonical UUID string.") from None
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(f"{field} must use canonical lowercase UUID text.")
    return canonical


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{label} schema mismatch; missing={missing}, extra={extra}.")


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
