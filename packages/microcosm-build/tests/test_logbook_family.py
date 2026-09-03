"""Contracts for Logbook dataset-family records."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import microcosm.build.logbook as logbook
import microcosm.build.logbook_family as family_module
from microcosm.build.logbook import LogbookRow, record_build_attempt
from microcosm.build.logbook_family import (
    FamilyAction,
    FamilyArchiveRecords,
    FamilyMember,
    LogbookFamily,
    derive_family_id,
    export_family_records,
    export_family_scope,
    family_archive_path,
    import_family_scope,
    load_families,
    load_family_archive_records,
    reconcile_family_spool,
    reconcile_logbook_spool,
    record_family,
    record_family_action,
    record_family_member,
    validate_family_action,
    validate_family_archive_records,
    validate_family_membership,
    validate_family_source,
)

FAMILY_ID = "86c298e7-a71a-5000-999f-cbf4aa993dac"
ACTION_ID = "87654321-4321-4321-8321-cba987654321"


@pytest.fixture(autouse=True)
def _spool_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)


def _family(**overrides: object) -> LogbookFamily:
    values = {
        "chain_scope": "us",
        "source_pool_sha256": "a" * 64,
    }
    values.update(overrides)
    if "family_id" not in overrides:
        values["family_id"] = derive_family_id(
            str(values["chain_scope"]),
            str(values["source_pool_sha256"]),
        )
    return LogbookFamily.create(**values)


def _build(
    build_id: str = "family-build-1",
    *,
    pipeline: str = "us-stacked-pool",
    requested_k: int | None = 20_000,
    rung: str = "f100",
) -> LogbookRow:
    return LogbookRow.create(
        build_id=build_id,
        ts="2026-08-21T12:00:00Z",
        pipeline=pipeline,
        rung=rung,
        seed=17,
        code_pin="abc1234",
        input_pins_digest="1" * 64,
        identity_digest="2" * 64,
        phases_reached=["built"],
        gate_verdicts={
            "build_validation": {
                "verdict": "passed",
                "receipt": "receipt://fixture.json",
            }
        },
        wall_seconds=1.0,
        cost_usd=None,
        artifact_location=None,
        disposition="iterating",
        prediction_id=None,
        prev_row_digest=None,
        row_format_version=2,
        requested_k=requested_k,
        realized_k=requested_k,
        record_unit=None if requested_k is None else "household",
    )


def _action(**overrides: object) -> FamilyAction:
    values = {
        "action_id": ACTION_ID,
        "family_id": FAMILY_ID,
        "build_id": "family-build-2",
        "action_type": "supersedes",
        "related_build_id": "family-build-1",
        "recorded_at": "2026-08-21T12:30:00+00:00",
        "actor": "anthony",
        "reason": "Corrected build",
        "evidence_location": None,
    }
    values.update(overrides)
    return FamilyAction.create(**values)


def test_family_and_membership_have_exact_fields() -> None:
    family = _family()
    member = FamilyMember.create(
        family_id=family.family_id,
        build_id="family-build-1",
    )

    assert family.to_mapping() == {
        "family_id": FAMILY_ID,
        "chain_scope": "us",
        "source_pool_sha256": "a" * 64,
    }
    assert member.to_mapping() == {
        "family_id": FAMILY_ID,
        "build_id": "family-build-1",
    }


def test_family_id_is_derived_from_scope_and_source() -> None:
    family = LogbookFamily.create(
        chain_scope="us",
        source_pool_sha256="a" * 64,
    )

    assert family.family_id == FAMILY_ID
    assert derive_family_id("uk/frs", "a" * 64) != FAMILY_ID
    with pytest.raises(ValueError, match="family_id must be"):
        LogbookFamily.from_mapping(
            {
                **family.to_mapping(),
                "family_id": "12345678-1234-4234-9234-123456789abc",
            }
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"family_id": "not-a-uuid"}, "canonical UUID"),
        ({"family_id": FAMILY_ID.upper()}, "canonical lowercase"),
        ({"chain_scope": "uk/locals"}, "chain_scope"),
        ({"source_pool_sha256": "A" * 64}, "source_pool_sha256"),
    ],
)
def test_family_validation_rejects_invalid_identity(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _family(**overrides)


def test_membership_validates_family_build_and_scope() -> None:
    family = _family()
    member = FamilyMember.create(
        family_id=FAMILY_ID,
        build_id="family-build-1",
    )

    validate_family_source(family, "a" * 64)
    validate_family_membership(family, member, _build())

    with pytest.raises(ValueError, match="identifies source"):
        validate_family_source(family, "b" * 64)
    with pytest.raises(ValueError, match="does not match family scope"):
        validate_family_membership(
            family,
            member,
            _build(pipeline="uk-frs-staging"),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"action_type": "publishes"}, "action_type"),
        (
            {"action_type": "revokes", "related_build_id": "family-build-1"},
            "must not contain",
        ),
        ({"related_build_id": None}, "requires related_build_id"),
        ({"related_build_id": "family-build-2"}, "cannot supersede itself"),
        ({"actor": " "}, "actor"),
        ({"recorded_at": "2026-08-21T12:30:00"}, "UTC offset"),
    ],
)
def test_family_action_shape_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _action(**overrides)


def test_supersession_requires_members_and_matching_cardinality() -> None:
    action = _action()
    members = (
        FamilyMember.create(family_id=FAMILY_ID, build_id="family-build-1"),
        FamilyMember.create(family_id=FAMILY_ID, build_id="family-build-2"),
    )
    builds = {
        "family-build-1": _build("family-build-1"),
        "family-build-2": _build("family-build-2"),
    }

    validate_family_action(action, members=members, builds=builds)

    mismatched = {
        **builds,
        "family-build-2": _build("family-build-2", requested_k=57_240),
    }
    with pytest.raises(ValueError, match="matching requested_k"):
        validate_family_action(action, members=members, builds=mismatched)

    mismatched_fraction = {
        "family-build-1": _build(
            "family-build-1",
            requested_k=None,
            rung="f100",
        ),
        "family-build-2": _build(
            "family-build-2",
            requested_k=None,
            rung="f010",
        ),
    }
    with pytest.raises(ValueError, match="must also have matching rung"):
        validate_family_action(
            action,
            members=members,
            builds=mismatched_fraction,
        )


def test_family_archive_validation_matches_database_relationship_constraints() -> None:
    family = _family()
    other_family = _family(source_pool_sha256="b" * 64)
    builds = {
        build_id: _build(build_id)
        for build_id in ("family-build-1", "family-build-2", "family-build-3")
    }
    members = tuple(
        FamilyMember.create(family_id=FAMILY_ID, build_id=build_id)
        for build_id in builds
    )

    duplicate_membership = FamilyArchiveRecords(
        families=(family, other_family),
        family_members=(
            members[0],
            FamilyMember.create(
                family_id=other_family.family_id,
                build_id=members[0].build_id,
            ),
        ),
        family_actions=(),
    )
    with pytest.raises(ValueError, match="belongs to more than one family"):
        validate_family_archive_records(
            duplicate_membership,
            scope="us",
            builds=tuple(builds.values()),
        )

    replacement = _action()
    second_replacement = _action(
        action_id="97654321-4321-4321-8321-cba987654321",
        build_id="family-build-3",
    )
    with pytest.raises(ValueError, match="more than one direct replacement"):
        validate_family_archive_records(
            FamilyArchiveRecords(
                families=(family,),
                family_members=members,
                family_actions=(replacement, second_replacement),
            ),
            scope="us",
            builds=tuple(builds.values()),
        )

    reverse_replacement = _action(
        action_id="a7654321-4321-4321-8321-cba987654321",
        build_id="family-build-1",
        related_build_id="family-build-2",
    )
    with pytest.raises(ValueError, match="replacement cycle"):
        validate_family_archive_records(
            FamilyArchiveRecords(
                families=(family,),
                family_members=members,
                family_actions=(replacement, reverse_replacement),
            ),
            scope="us",
            builds=tuple(builds.values()),
        )


def test_family_import_rejects_invalid_relationship_before_spooling(
    tmp_path: Path,
) -> None:
    family = _family()
    members = (
        FamilyMember.create(family_id=FAMILY_ID, build_id="family-build-1"),
        FamilyMember.create(family_id=FAMILY_ID, build_id="family-build-2"),
    )
    action = _action()
    builds = (
        _build("family-build-1", requested_k=57_240),
        _build("family-build-2", requested_k=20_000),
    )
    archive_root = tmp_path / "logbook"
    export_family_records(
        family_archive_path(archive_root, "families", "us"),
        (family,),
    )
    export_family_records(
        family_archive_path(archive_root, "family_members", "us"),
        members,
    )
    export_family_records(
        family_archive_path(archive_root, "family_actions", "us"),
        (action,),
    )
    spool = tmp_path / "spool"

    with pytest.raises(ValueError, match="matching requested_k"):
        import_family_scope(
            archive_root,
            scope="us",
            spool_dir=spool,
            builds=builds,
        )

    assert not spool.exists()


def test_family_spool_is_durable_and_idempotent(tmp_path: Path) -> None:
    family = _family()

    first = record_family(family, spool_dir=tmp_path)
    second = record_family(family, spool_dir=tmp_path)

    assert first == second
    assert first.spool_path == tmp_path / "families" / f"{FAMILY_ID}.json"
    assert json.loads(first.spool_path.read_text()) == family.to_mapping()

    with pytest.raises(ValueError, match="family_id must be"):
        _family(family_id=FAMILY_ID, source_pool_sha256="b" * 64)


def test_family_spool_retry_completes_interrupted_parent_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_sync = logbook._fsync_parent_directory
    calls = 0

    def fail_first(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory sync failure")
        real_sync(path)

    monkeypatch.setattr(logbook, "_fsync_parent_directory", fail_first)

    with pytest.raises(OSError, match="injected directory sync failure"):
        record_family(_family(), spool_dir=tmp_path)
    persisted = tmp_path / "families" / f"{FAMILY_ID}.json"
    assert persisted.exists()

    retry = record_family(_family(), spool_dir=tmp_path)

    assert retry.spool_path == persisted
    assert calls == 2


class _Response:
    status = 201

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_remote_family_insert_occurs_after_local_spool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-key")
    requests = []

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        assert timeout == 3.0
        assert (tmp_path / "families" / f"{FAMILY_ID}.json").exists()
        requests.append(request)
        return _Response()

    monkeypatch.setattr(family_module, "urlopen", fake_urlopen)

    result = record_family(_family(), spool_dir=tmp_path, timeout=3.0)

    assert result.posted is True
    request = requests[0]
    parsed = urlparse(request.full_url)
    assert parsed.path == "/rest/v1/families"
    assert parse_qs(parsed.query)["on_conflict"] == ["family_id"]
    assert request.headers["Content-profile"] == "logbook"
    assert json.loads(request.data) == _family().to_mapping()


def test_combined_reconciliation_preserves_dependency_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_result = record_build_attempt(
        **{
            key: value
            for key, value in _build().to_mapping().items()
            if key not in {"row_digest", "row_format_version"}
        },
        row_format_version=2,
        spool_dir=tmp_path,
    )
    family = _family()
    member = FamilyMember.create(
        family_id=FAMILY_ID,
        build_id=build_result.row.build_id,
    )
    action = FamilyAction.create(
        action_id=ACTION_ID,
        family_id=FAMILY_ID,
        build_id=build_result.row.build_id,
        action_type="revokes",
        related_build_id=None,
        recorded_at="2026-08-21T12:30:00Z",
        actor="anthony",
        reason="Invalid artifact",
        evidence_location=None,
    )
    record_family(family, spool_dir=tmp_path, post_remote=False)
    record_family_member(member, spool_dir=tmp_path, post_remote=False)
    record_family_action(action, spool_dir=tmp_path, post_remote=False)
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")
    posted_paths: list[str] = []

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        del timeout
        posted_paths.append(urlparse(request.full_url).path)
        return _Response()

    monkeypatch.setattr(logbook, "urlopen", fake_urlopen)
    monkeypatch.setattr(family_module, "urlopen", fake_urlopen)

    receipt = reconcile_logbook_spool(tmp_path)

    assert receipt.builds.errors == ()
    assert receipt.families.errors == ()
    assert posted_paths == [
        "/rest/v1/builds",
        "/rest/v1/families",
        "/rest/v1/family_members",
        "/rest/v1/family_actions",
    ]
    assert list(tmp_path.rglob("*.json")) == []


def test_family_reconciliation_stops_and_retains_dependents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    family = _family()
    member = FamilyMember.create(
        family_id=FAMILY_ID,
        build_id="family-build-1",
    )
    record_family(family, spool_dir=tmp_path, post_remote=False)
    record_family_member(member, spool_dir=tmp_path, post_remote=False)
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")

    def fail(*_args: object, **_kwargs: object) -> _Response:
        raise OSError("offline")

    monkeypatch.setattr(family_module, "urlopen", fail)

    receipt = reconcile_family_spool(tmp_path)

    assert receipt.attempted == 1
    assert receipt.posted == 0
    assert receipt.retained == 2
    assert "offline" in receipt.errors[0]
    assert len(list(tmp_path.rglob("*.json"))) == 2


def test_family_archive_append_is_idempotent(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "families" / "us.jsonl"
    family = _family()

    first = export_family_records(archive, [family])
    second = export_family_records(archive, [family])

    assert (first.existing, first.appended) == (0, 1)
    assert (second.existing, second.appended) == (1, 0)
    assert load_families(archive) == (family,)


@pytest.mark.parametrize(
    ("scope", "relative_path"),
    [
        ("us", "families/us.jsonl"),
        ("uk/frs", "families/uk/frs.jsonl"),
    ],
)
def test_family_scope_archives_use_scope_specific_paths(
    tmp_path: Path,
    scope: str,
    relative_path: str,
) -> None:
    assert family_archive_path(tmp_path, "families", scope) == (
        tmp_path / relative_path
    )


def test_family_scope_archive_exports_and_imports_in_dependency_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    family = _family()
    members = (
        FamilyMember.create(family_id=FAMILY_ID, build_id="family-build-1"),
        FamilyMember.create(family_id=FAMILY_ID, build_id="family-build-2"),
    )
    action = _action()
    builds = (_build("family-build-1"), _build("family-build-2"))
    archive_root = tmp_path / "logbook"

    first = export_family_scope(
        archive_root,
        scope="us",
        builds=builds,
        families=(family,),
        family_members=members,
        family_actions=(action,),
    )
    second = export_family_scope(
        archive_root,
        scope="us",
        builds=builds,
        families=(family,),
        family_members=members,
        family_actions=(action,),
    )
    no_new_actions = export_family_scope(
        archive_root,
        scope="us",
        builds=builds,
        families=(family,),
        family_members=members,
    )

    assert {key: value.appended for key, value in first.items()} == {
        "families": 1,
        "family_members": 2,
        "family_actions": 1,
    }
    assert all(value.appended == 0 for value in second.values())
    assert no_new_actions["family_actions"] == family_module.FamilyExportResult(
        existing=1,
        appended=0,
    )
    assert load_family_archive_records(archive_root, "us").families == (family,)

    calls: list[str] = []
    real_family = family_module.record_family
    real_member = family_module.record_family_member
    real_action = family_module.record_family_action

    def track_family(*args: object, **kwargs: object):
        calls.append("families")
        return real_family(*args, **kwargs)

    def track_member(*args: object, **kwargs: object):
        calls.append("family_members")
        return real_member(*args, **kwargs)

    def track_action(*args: object, **kwargs: object):
        calls.append("family_actions")
        return real_action(*args, **kwargs)

    monkeypatch.setattr(family_module, "record_family", track_family)
    monkeypatch.setattr(family_module, "record_family_member", track_member)
    monkeypatch.setattr(family_module, "record_family_action", track_action)

    imported = import_family_scope(
        archive_root,
        scope="us",
        spool_dir=tmp_path / "spool",
        builds=builds,
    )

    assert imported.family_actions == (action,)
    assert calls == [
        "families",
        "family_members",
        "family_members",
        "family_actions",
    ]


def test_family_scope_archive_rejects_cross_scope_and_missing_dependencies(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="another scope"):
        export_family_scope(
            tmp_path,
            scope="us",
            builds=(),
            families=(_family(chain_scope="uk/frs"),),
        )
    assert not list(tmp_path.rglob("*.jsonl"))

    member = FamilyMember.create(
        family_id=FAMILY_ID,
        build_id="family-build-1",
    )
    with pytest.raises(ValueError, match="missing family"):
        export_family_scope(
            tmp_path,
            scope="us",
            builds=(_build(),),
            family_members=(member,),
        )
    assert not list(tmp_path.rglob("*.jsonl"))

    with pytest.raises(ValueError, match="missing build"):
        export_family_scope(
            tmp_path,
            scope="us",
            builds=(),
            families=(_family(),),
            family_members=(member,),
        )
    assert not list(tmp_path.rglob("*.jsonl"))

    with pytest.raises(ValueError, match="does not match family scope"):
        export_family_scope(
            tmp_path,
            scope="us",
            builds=(_build(pipeline="uk-frs-staging"),),
            families=(_family(),),
            family_members=(member,),
        )
    assert not list(tmp_path.rglob("*.jsonl"))
