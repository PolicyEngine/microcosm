"""Checkouts at different microcosm versions sharing one telemetry spool.

Every checkout and worktree on a host opens the same spool, so a build can find
it migrated by a newer checkout, or by a branch with a different migration.
These tests open one spool as several checkouts at once: the packaged migration
history, and temporary copies of it that carry synthetic later revisions. They
check that an older checkout keeps using a spool a newer one migrated, that
events queued by either version are delivered by the other, that a checkout
whose history diverges from the spool's is refused without the write lock, and
that every revision is additive, which is what lets older code rely on the
migrated schema.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import as_file, files
from pathlib import Path

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun
from microcosm.build.telemetry_emitter_constants import TELEMETRY_SERVICE_MODULE
from microcosm.build.telemetry_emitter_service import collector as collector_module
from microcosm.build.telemetry_emitter_service import migrations as migrations_module
from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DATABASE_TIMEOUT_SECONDS,
    SPOOL_LINEAGE_TABLE,
)
from microcosm.build.telemetry_emitter_service.database import create_spool_engine
from microcosm.build.telemetry_emitter_service.migrations import (
    IncompatibleSpoolRevisionError,
    MigrationHistory,
    SpoolRevisionState,
    alembic_config,
    classify_spool_revision,
    current_database_revision,
    migration_head_revision,
    migration_history,
    recorded_spool_lineage,
    upgrade_spool_database,
)
from microcosm.build.telemetry_emitter_service.spool import EventSpool
from microcosm.build.telemetry_protocol import (
    BUILD_COMPLETED_MESSAGE,
    BUILD_STARTED_MESSAGE,
)

_SERVICE_PACKAGE = "microcosm.build.telemetry_emitter_service"
_INITIAL_REVISION = "20261007_01"
_LOOPBACK_COLLECTOR = "http://127.0.0.1:9"
_BOOKKEEPING_TABLES = frozenset({"alembic_version", SPOOL_LINEAGE_TABLE})


@functools.cache
def _packaged_head() -> str:
    head = migration_head_revision()
    assert head is not None
    return head


def _registration(run_id: str, producer_id: str) -> dict[str, object]:
    return TelemetryRun(
        run_id=run_id,
        country_code="US",
        pipeline="test-pipeline",
        producer_id=producer_id,
    ).as_registration()


def _event(stage_id: str) -> dict[str, object]:
    return {"event_type": "stage", "stage_id": stage_id, "status": "started"}


def _close(spool: EventSpool) -> None:
    spool._engine.dispose()


@contextlib.contextmanager
def _write_lock(path: Path):
    """Hold the spool's write lock from another connection, as a writer would."""

    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def _tables(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()


def _stamp(path: Path, revision: str, lineage: frozenset[str] | None) -> None:
    """Stamp a spool as another checkout's migration would have left it."""

    connection = sqlite3.connect(path)
    try:
        connection.execute("UPDATE alembic_version SET version_num = ?", (revision,))
        connection.execute(f"DROP TABLE IF EXISTS {SPOOL_LINEAGE_TABLE}")
        if lineage is not None:
            connection.execute(
                f"CREATE TABLE {SPOOL_LINEAGE_TABLE} "
                "(revision TEXT NOT NULL PRIMARY KEY)"
            )
            connection.executemany(
                f"INSERT INTO {SPOOL_LINEAGE_TABLE} (revision) VALUES (?)",
                [(name,) for name in sorted(lineage)],
            )
        connection.commit()
    finally:
        connection.close()


# --- Other checkouts' migration histories ------------------------------------

_REVISION_SOURCE = '''"""Synthetic revision {revision} in a test checkout's history."""

import sqlalchemy as sa
from alembic import op

revision = {revision!r}
down_revision = {down_revision!r}
branch_labels = None
depends_on = None


def upgrade() -> None:
{upgrade}


def downgrade() -> None:
    raise NotImplementedError
'''


@dataclass(frozen=True)
class _Revision:
    revision: str
    down_revision: str
    upgrade: str

    def source(self) -> str:
        return _REVISION_SOURCE.format(
            revision=self.revision,
            down_revision=self.down_revision,
            upgrade=textwrap.indent(textwrap.dedent(self.upgrade).strip(), "    "),
        )


def _additive_revision(revision: str, down_revision: str | None = None) -> _Revision:
    """A later revision like the next one in flight, which adds a table (#1151).

    It also adds a nullable column, a defaulted NOT NULL column and an index to
    the tables older code writes. Names carry the revision, so sibling and
    descendant revisions never collide.
    """

    return _Revision(
        revision,
        down_revision or _packaged_head(),
        f"""
        op.create_table(
            "graph_publication_jobs_{revision}",
            sa.Column("publication_id", sa.Text(), primary_key=True),
            sa.Column("status", sa.Text(), nullable=False),
        )
        op.add_column(
            "telemetry_runs",
            sa.Column("region_{revision}", sa.Text(), nullable=True),
        )
        op.add_column(
            "telemetry_events",
            sa.Column(
                "attempts_{revision}",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        op.create_index(
            "telemetry_events_created_{revision}",
            "telemetry_events",
            ["created_at"],
        )
        """,
    )


def _checkout(root: Path, *revisions: _Revision) -> Path:
    """Write one checkout's migration history: the packaged one plus more."""

    with as_file(files(_SERVICE_PACKAGE).joinpath("alembic")) as packaged:
        shutil.copytree(packaged, root, ignore=shutil.ignore_patterns("__pycache__"))
    for revision in revisions:
        (root / "versions" / f"v{revision.revision}.py").write_text(revision.source())
    return root


class _Collector:
    """Accept everything a CollectorDelivery sends, recording the events."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.registrations: list[dict[str, object]] = []

    def post(self, url, payload, token, *, timeout=5.0):
        if url.endswith("/v1/auth/huggingface/exchange"):
            return 200, {"access_token": "collector-token", "expires_in": 3600}
        if url.endswith("/v1/runs"):
            self.registrations.append(dict(payload))
            return 201, {"registered": True}
        self.events.extend(payload["events"])
        return 202, {"accepted": len(payload["events"]), "duplicates": 0}


@pytest.fixture
def collector(monkeypatch) -> _Collector:
    fake = _Collector()
    monkeypatch.setattr(collector_module, "_http_post", fake.post)
    monkeypatch.setattr(collector_module, "_huggingface_token", lambda: "hf-token")
    return fake


def _deliver(spool: EventSpool) -> None:
    CollectorDelivery(spool, development_collector_url=_LOOPBACK_COLLECTOR).flush_once()


# --- An older checkout keeps using a spool a newer one migrated --------------


def test_an_older_checkout_uses_a_spool_a_newer_checkout_migrated(tmp_path) -> None:
    spool_path = tmp_path / "events.sqlite3"
    newer = _checkout(tmp_path / "newer", _additive_revision("future"))
    _close(EventSpool(spool_path, script_location=newer))
    assert current_database_revision(spool_path) == "future"
    assert recorded_spool_lineage(spool_path) == migration_history(newer).revisions

    # The older checkout only reads, so another writer's lock does not delay it.
    with _write_lock(spool_path):
        started = time.monotonic()
        older = EventSpool(spool_path)
        opened_in = time.monotonic() - started
    try:
        assert opened_in < DATABASE_TIMEOUT_SECONDS
        registration = _registration("older-run", "older-producer")
        older.register(registration)
        queued = [older.append(registration, _event(stage)) for stage in "abc"]
        assert older.batch("older-run", "older-producer") == queued
        older.acknowledge([queued[0]["event_id"]])
        older.prune()
        assert older.pending_runs() == [registration]
        assert older.batch("older-run", "older-producer") == queued[1:]
    finally:
        _close(older)

    engine = create_spool_engine(spool_path)
    try:
        assert upgrade_spool_database(engine) is SpoolRevisionState.AHEAD
    finally:
        engine.dispose()
    assert current_database_revision(spool_path) == "future"
    assert "graph_publication_jobs_future" in _tables(spool_path)


def test_events_queued_by_either_version_are_delivered_by_the_other(
    tmp_path, collector
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    newer = EventSpool(
        spool_path,
        script_location=_checkout(tmp_path / "newer", _additive_revision("future")),
    )
    older = EventSpool(spool_path)
    try:
        older_run = _registration("older-run", "older-producer")
        older.register(older_run)
        from_older = [older.append(older_run, _event(stage)) for stage in "ab"]
        _deliver(newer)
        assert collector.events == from_older
        assert not older.has_pending()

        newer_run = _registration("newer-run", "newer-producer")
        newer.register(newer_run)
        from_newer = [newer.append(newer_run, _event(stage)) for stage in "ab"]
        _deliver(older)
        assert collector.events == from_older + from_newer
        assert not newer.has_pending()
        assert collector.registrations == [older_run, newer_run]
    finally:
        _close(older)
        _close(newer)


def _stored_rows(path: Path) -> dict[str, list[dict[str, object]]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return {
            table: sorted(
                (dict(row) for row in connection.execute(f"SELECT * FROM {table}")),
                key=repr,
            )
            for table in ("telemetry_runs", "telemetry_events")
        }
    finally:
        connection.close()


def test_upgrading_keeps_every_queued_row_and_its_delivery(tmp_path, collector) -> None:
    spool_path = tmp_path / "events.sqlite3"
    older = EventSpool(spool_path)
    try:
        pending = _registration("pending-run", "producer-a")
        older.register(pending)
        queued = [older.append(pending, _event(stage)) for stage in "abc"]
        local = _registration("local-run", "producer-b")
        older.register(local)
        older.append(local, _event("a"))
        older.make_local_only("local-run", "producer-b", "missing_credential")
    finally:
        _close(older)
    before = _stored_rows(spool_path)

    newer = EventSpool(
        spool_path,
        script_location=_checkout(tmp_path / "newer", _additive_revision("future")),
    )
    try:
        after = _stored_rows(spool_path)
        for table, rows in before.items():
            kept = ({column: row[column] for column in rows[0]} for row in after[table])
            assert sorted(kept, key=repr) == rows
        assert {row["region_future"] for row in after["telemetry_runs"]} == {None}
        assert {row["attempts_future"] for row in after["telemetry_events"]} == {0}
        _deliver(newer)
    finally:
        _close(newer)
    assert collector.events == queued


def test_a_live_older_service_keeps_queuing_across_a_newer_migration(tmp_path) -> None:
    """SQLite re-prepares an open connection's statements after the change."""

    spool_path = tmp_path / "events.sqlite3"
    older = EventSpool(spool_path)
    try:
        registration = _registration("older-run", "older-producer")
        older.register(registration)
        older.append(registration, _event("a"))
        _close(
            EventSpool(
                spool_path,
                script_location=_checkout(
                    tmp_path / "newer", _additive_revision("future")
                ),
            )
        )
        assert current_database_revision(spool_path) == "future"
        older.append(registration, _event("b"))
        older.prune()
        events = older.batch("older-run", "older-producer")
    finally:
        _close(older)
    assert [(event["sequence"], event["stage_id"]) for event in events] == [
        (1, "a"),
        (2, "b"),
    ]


def test_an_opener_that_waited_out_a_newer_migration_changes_nothing(
    tmp_path, monkeypatch
) -> None:
    """It found the spool behind, then got the lock after a newer checkout.

    Without the second look under the lock, Alembic would try to upgrade a
    spool stamped with a revision this opener does not know, and fail.
    """

    spool_path = tmp_path / "events.sqlite3"
    middle = _additive_revision("middle")
    opener = _checkout(tmp_path / "opener", middle)
    newer = _checkout(
        tmp_path / "newer", middle, _additive_revision("future", "middle")
    )
    _close(EventSpool(spool_path))
    results: list[SpoolRevisionState | BaseException] = []
    # The opener builds its locking engine only after its read-only look found
    # the spool behind, so the migration below starts after that look.
    found_behind = threading.Event()
    create_engine_for_migrations = migrations_module.create_spool_engine

    def create_spool_engine_and_signal(path, **options):
        if options.get("immediate_transactions"):
            found_behind.set()
        return create_engine_for_migrations(path, **options)

    monkeypatch.setattr(
        migrations_module, "create_spool_engine", create_spool_engine_and_signal
    )

    def open_spool() -> None:
        engine = create_spool_engine(spool_path)
        try:
            results.append(
                upgrade_spool_database(
                    engine, busy_timeout_seconds=lambda: 60, script_location=opener
                )
            )
        except BaseException as error:
            results.append(error)
        finally:
            engine.dispose()

    waiting = threading.Thread(target=open_spool, daemon=True)
    migrator = create_spool_engine(spool_path, immediate_transactions=True)
    try:
        with migrator.begin() as connection:
            waiting.start()
            assert found_behind.wait(timeout=60), results
            with alembic_config(connection=connection, script_location=newer) as config:
                command.upgrade(config, "head")
            migrations_module._record_lineage(connection, migration_history(newer))
        waiting.join(timeout=90)
    finally:
        migrator.dispose()

    assert results == [SpoolRevisionState.AHEAD]
    assert current_database_revision(spool_path) == "future"
    # The opener left the newer checkout's lineage alone. Had it written its
    # own, the spool's revision would be missing from it, and every older
    # checkout would be refused from then on.
    assert recorded_spool_lineage(spool_path) == migration_history(newer).revisions
    for script_location in (None, opener, newer):
        _close(EventSpool(spool_path, script_location=script_location))


def test_an_older_append_waits_out_a_newer_migration_in_progress(tmp_path) -> None:
    """The event waits in SQLite's busy handler while the migration holds the lock.

    One that outlasts that 5 s wait is lost, as behind any long writer.
    """

    spool_path = tmp_path / "events.sqlite3"
    newer = _checkout(tmp_path / "newer", _additive_revision("future"))
    older = EventSpool(spool_path)
    registration = _registration("older-run", "older-producer")
    older.register(registration)
    older.append(registration, _event("a"))
    appended: list[dict[str, object] | BaseException] = []

    def append() -> None:
        try:
            appended.append(older.append(registration, _event("b")))
        except BaseException as error:
            appended.append(error)

    appending = threading.Thread(target=append, daemon=True)
    migrator = create_spool_engine(spool_path, immediate_transactions=True)
    try:
        with migrator.begin() as connection:
            with alembic_config(connection=connection, script_location=newer) as config:
                command.upgrade(config, "head")
            migrations_module._record_lineage(connection, migration_history(newer))
            appending.start()
            time.sleep(0.5)
            assert appending.is_alive(), appended
        appending.join(timeout=60)
        events = older.batch("older-run", "older-producer")
    finally:
        migrator.dispose()
        _close(older)

    assert [type(result) for result in appended] == [dict], appended
    assert [(event["sequence"], event["stage_id"]) for event in events] == [
        (1, "a"),
        (2, "b"),
    ]
    assert current_database_revision(spool_path) == "future"


def test_a_failed_upgrade_leaves_the_schema_the_stamp_and_the_lineage(
    tmp_path, monkeypatch
) -> None:
    """They commit together, so an upgrade that fails last undoes all three."""

    spool_path = tmp_path / "events.sqlite3"
    _close(EventSpool(spool_path))
    lineage = recorded_spool_lineage(spool_path)
    record_lineage = migrations_module._record_lineage

    def record_lineage_then_fail(connection, history) -> None:
        record_lineage(connection, history)
        raise RuntimeError("interrupted after the lineage")

    monkeypatch.setattr(migrations_module, "_record_lineage", record_lineage_then_fail)
    with pytest.raises(RuntimeError, match="interrupted after the lineage"):
        EventSpool(
            spool_path,
            script_location=_checkout(tmp_path / "newer", _additive_revision("future")),
        )

    assert current_database_revision(spool_path) == _packaged_head()
    assert recorded_spool_lineage(spool_path) == lineage
    assert "graph_publication_jobs_future" not in _tables(spool_path)
    assert "region_future" not in _runs_columns(spool_path)


@pytest.mark.parametrize(
    "created_by", ["a lineage-recording runner", "an older runner"]
)
def test_a_checkout_at_head_records_the_lineage_an_older_runner_left_out(
    tmp_path, created_by
) -> None:
    """A checkout from before lineage was recorded migrated the spool.

    Until a lineage-recording checkout at that revision opens it, an older
    checkout cannot tell that the revision descends from its own head.
    """

    spool_path = tmp_path / "events.sqlite3"
    newer = _checkout(tmp_path / "newer", _additive_revision("future"))
    if created_by == "a lineage-recording runner":
        _close(EventSpool(spool_path))
    state = _open_as_runner_without_lineage(spool_path, newer)
    assert state is SpoolRevisionState.BEHIND
    assert current_database_revision(spool_path) == "future"
    assert recorded_spool_lineage(spool_path) == (
        migration_history().revisions
        if created_by == "a lineage-recording runner"
        else None
    )
    with pytest.raises(IncompatibleSpoolRevisionError):
        EventSpool(spool_path)

    _close(EventSpool(spool_path, script_location=newer))
    assert recorded_spool_lineage(spool_path) == migration_history(newer).revisions
    # Recorded once: with the lock held elsewhere, the next opens only read.
    with _write_lock(spool_path):
        started = time.monotonic()
        _close(EventSpool(spool_path, script_location=newer))
        older = EventSpool(spool_path)
        assert time.monotonic() - started < DATABASE_TIMEOUT_SECONDS
    try:
        registration = _registration("older-run", "older-producer")
        older.register(registration)
        assert older.append(registration, _event("a"))["sequence"] == 1
    finally:
        _close(older)


def _initial_checkout(root: Path) -> Path:
    """A checkout whose history is the initial revision alone."""

    _checkout(root)
    with alembic_config(script_location=root) as config:
        for script in ScriptDirectory.from_config(config).walk_revisions():
            if script.down_revision is not None:
                Path(script.path).unlink()
    return root


def test_a_spool_at_the_initial_revision_needs_no_lineage(tmp_path) -> None:
    """Every history starts there, so nothing is written and no lock is taken.

    Every spool that exists when lineage-recording checkouts arrive is in
    this state.
    """

    spool_path = tmp_path / "events.sqlite3"
    initial = _initial_checkout(tmp_path / "initial")
    state = _open_as_runner_without_lineage(spool_path, initial)
    assert state is SpoolRevisionState.BEHIND
    assert current_database_revision(spool_path) == _INITIAL_REVISION
    assert recorded_spool_lineage(spool_path) is None

    with _write_lock(spool_path):
        started = time.monotonic()
        spool = EventSpool(spool_path, script_location=initial)
        assert time.monotonic() - started < DATABASE_TIMEOUT_SECONDS
    _close(spool)
    assert recorded_spool_lineage(spool_path) is None


def test_the_lineage_table_is_not_part_of_the_schema_alembic_checks(tmp_path) -> None:
    spool_path = tmp_path / "events.sqlite3"
    _close(EventSpool(spool_path))
    assert recorded_spool_lineage(spool_path) == migration_history().revisions
    engine = create_spool_engine(spool_path)
    try:
        with (
            engine.begin() as connection,
            alembic_config(connection=connection) as config,
        ):
            command.check(config)
    finally:
        engine.dispose()


# --- Diverged and unknown histories are refused without the lock -------------


def test_a_branch_with_a_different_migration_is_refused_without_the_lock(
    tmp_path,
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    branch_a = _checkout(tmp_path / "a", _additive_revision("branch_a"))
    branch_b = _checkout(tmp_path / "b", _additive_revision("branch_b"))
    _close(EventSpool(spool_path, script_location=branch_a))
    lineage = recorded_spool_lineage(spool_path)

    with _write_lock(spool_path):
        started = time.monotonic()
        with pytest.raises(
            IncompatibleSpoolRevisionError,
            match=(
                "revision 'branch_a' is not in this microcosm's migration history, "
                "whose head is 'branch_b'"
            ),
        ):
            EventSpool(spool_path, script_location=branch_b)
        assert time.monotonic() - started < DATABASE_TIMEOUT_SECONDS

    assert current_database_revision(spool_path) == "branch_a"
    assert recorded_spool_lineage(spool_path) == lineage
    assert "graph_publication_jobs_branch_b" not in _tables(spool_path)
    # The revision both branches descend from still uses the spool.
    _close(EventSpool(spool_path))


@pytest.mark.parametrize(
    ("lineage", "usable"),
    [
        pytest.param(None, False, id="no-recorded-lineage"),
        pytest.param({"<head>"}, False, id="lineage-without-the-stamp"),
        pytest.param({"future"}, False, id="lineage-without-this-head"),
        pytest.param({"<head>", "future"}, True, id="descends-from-this-head"),
    ],
)
def test_a_spool_stamped_with_a_future_revision_is_used_only_if_it_descends(
    tmp_path, lineage, usable
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    _close(EventSpool(spool_path))
    if lineage is not None:
        lineage = frozenset(
            _packaged_head() if name == "<head>" else name for name in lineage
        )
    _stamp(spool_path, "future", lineage)

    if not usable:
        with pytest.raises(IncompatibleSpoolRevisionError, match="'future'"):
            EventSpool(spool_path)
        return
    spool = EventSpool(spool_path)
    try:
        registration = _registration("run", "producer")
        spool.register(registration)
        assert spool.append(registration, _event("a"))["sequence"] == 1
    finally:
        _close(spool)
    assert current_database_revision(spool_path) == "future"


# --- End to end: the real client and service processes ------------------------


class _CollectorHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict[str, object]]]

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.requests.append((self.path, payload))
        if self.path.endswith("/v1/auth/huggingface/exchange"):
            status, response = 200, {"access_token": "token", "expires_in": 3600}
        elif self.path == "/v1/runs":
            status, response = 201, {"registered": True}
        else:
            status = 202
            response = {"accepted": len(payload["events"]), "duplicates": 0}
        body = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


@contextlib.contextmanager
def _collector_server():
    requests: list[tuple[str, dict[str, object]]] = []
    handler = type("Handler", (_CollectorHandler,), {"requests": requests})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_an_older_build_delivers_its_own_and_newer_events_through_a_migrated_spool(
    tmp_path, monkeypatch, capfd, real_local_telemetry
) -> None:
    """The real client starts the real (older) service on a newer spool."""

    spool_path = tmp_path / "events.sqlite3"
    newer = EventSpool(
        spool_path,
        script_location=_checkout(tmp_path / "newer", _additive_revision("future")),
    )
    try:
        newer_run = _registration("newer-run", "newer-producer")
        newer.register(newer_run)
        from_newer = [newer.append(newer_run, _event(stage)) for stage in "ab"]
    finally:
        _close(newer)

    monkeypatch.setenv("HF_TOKEN", "hf-ambient-test-token")
    with _collector_server() as (collector_url, requests):
        emitter = LocalTelemetryEmitter.start(
            run_id="older-build",
            country_code="US",
            pipeline="test-pipeline",
            development_collector_url=collector_url,
            spool_path=spool_path,
            heartbeat_seconds=60,
            startup_timeout_seconds=60,
        )
        assert emitter.available
        emitter.stage("compile", message="Compiling.")
        emitter.complete()
        assert emitter._process is not None
        assert emitter._process.wait(timeout=60) == 0

    delivered: dict[str, list[dict[str, object]]] = {}
    for path, payload in requests:
        if path.endswith("/events"):
            for event in payload["events"]:
                delivered.setdefault(str(event["run_id"]), []).append(event)
    assert delivered["newer-run"] == from_newer
    older_events = delivered["older-build"]
    assert [event["sequence"] for event in older_events] == list(
        range(1, len(older_events) + 1)
    )
    assert older_events[0]["message"] == BUILD_STARTED_MESSAGE
    assert older_events[-1]["message"] == BUILD_COMPLETED_MESSAGE
    assert current_database_revision(spool_path) == "future"
    # On a loaded host the client may warn that one send outlived its 0.2 s
    # wait; the service still queued it. It must never have refused the spool.
    stderr = capfd.readouterr().err
    assert "emitter service stopped" not in stderr
    assert "Traceback" not in stderr


def test_the_service_refuses_a_diverged_spool_in_one_line(tmp_path) -> None:
    """The real process: status 1 and one stderr line naming both revisions."""

    spool_path = tmp_path / "events.sqlite3"
    _close(EventSpool(spool_path))
    _stamp(spool_path, "elsewhere", frozenset({"other_root", "elsewhere"}))
    # The service refuses the spool before it would bind this.
    socket_path = tmp_path / "emitter.sock"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            TELEMETRY_SERVICE_MODULE,
            "--socket",
            str(socket_path),
            "--spool",
            str(spool_path),
            "--registration-json",
            json.dumps(_registration("run", "producer")),
            "--parent-pid",
            "1",
            "--development-collector-url",
            _LOOPBACK_COLLECTOR,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 1
    lines = [line for line in completed.stderr.splitlines() if line.strip()]
    assert lines == [
        "warning: the local telemetry emitter service stopped: "
        "IncompatibleSpoolRevisionError: telemetry spool schema revision "
        f"'elsewhere' is not in this microcosm's migration history, whose head "
        f"is {_packaged_head()!r}, and the spool does not record it as "
        f"descending from {_packaged_head()!r}"
    ]
    assert not socket_path.exists()
    assert current_database_revision(spool_path) == "elsewhere"


# --- Properties over revision trees -------------------------------------------


@st.composite
def _revision_trees(draw, max_revisions: int = 4) -> list[int | None]:
    """Parent indices of a revision tree; revision 0 is the packaged head."""

    count = draw(st.integers(min_value=1, max_value=max_revisions))
    return [None] + [
        draw(st.integers(min_value=0, max_value=index - 1))
        for index in range(1, count + 1)
    ]


def _ancestry(parents: list[int | None], node: int) -> frozenset[int]:
    """``node`` and every revision it descends from."""

    lineage = {node}
    while (parent := parents[node]) is not None:
        lineage.add(parent)
        node = parent
    return frozenset(lineage)


def _expected_state(
    parents: list[int | None], stamp: int | None, head: int
) -> SpoolRevisionState:
    """The model: which way the spool lies from a checkout, by ancestry alone."""

    if stamp == head:
        return SpoolRevisionState.AT_HEAD
    if stamp is None or stamp in _ancestry(parents, head):
        return SpoolRevisionState.BEHIND
    if head in _ancestry(parents, stamp):
        return SpoolRevisionState.AHEAD
    return SpoolRevisionState.INCOMPATIBLE


@settings(max_examples=300, deadline=None)
@given(
    parents=_revision_trees(max_revisions=6),
    head=st.integers(min_value=0, max_value=6),
    stamp=st.one_of(st.none(), st.integers(min_value=0, max_value=6)),
    recorded=st.sampled_from(["exact", "absent", "without-stamp", "without-root"]),
)
def test_classification_follows_ancestry_alone(parents, head, stamp, recorded) -> None:
    head %= len(parents)
    names = [f"r{index}" for index in range(len(parents))]
    history = MigrationHistory(
        head=names[head],
        revisions=frozenset(names[node] for node in _ancestry(parents, head)),
        bases=frozenset({names[0]}),
    )
    if stamp is None:
        state = classify_spool_revision(None, None, history)
        assert state is SpoolRevisionState.BEHIND
        return
    stamp %= len(parents)
    lineage = frozenset(names[node] for node in _ancestry(parents, stamp))
    lineage = {
        "exact": lineage,
        "absent": None,
        "without-stamp": lineage - {names[stamp]},
        "without-root": lineage - {names[0]},
    }[recorded]
    state = classify_spool_revision(names[stamp], lineage, history)

    expected = _expected_state(parents, stamp, head)
    # Being ahead by ancestry is not enough: the spool must say so, by
    # recording both its own revision and this checkout's head.
    if expected is SpoolRevisionState.AHEAD and (
        lineage is None or not {names[stamp], names[head]} <= lineage
    ):
        expected = SpoolRevisionState.INCOMPATIBLE
    assert state is expected


def _data_version(connection: sqlite3.Connection) -> int:
    return connection.execute("PRAGMA data_version").fetchone()[0]


def _runs_columns(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            row[1] for row in connection.execute("PRAGMA table_info(telemetry_runs)")
        }
    finally:
        connection.close()


def _open_as_runner_without_lineage(
    path: Path, script_location: Path | None
) -> SpoolRevisionState:
    """Open as a checkout from before lineage was recorded, like #1168's runner.

    It refuses a revision it does not know. Otherwise it upgrades to its head
    in one ``BEGIN IMMEDIATE`` transaction, and records no lineage.
    """

    history = migration_history(script_location)
    revision = current_database_revision(path)
    if revision == history.head:
        return SpoolRevisionState.AT_HEAD
    if revision is not None and revision not in history.revisions:
        return SpoolRevisionState.INCOMPATIBLE
    engine = create_spool_engine(path, immediate_transactions=True)
    try:
        with (
            engine.begin() as connection,
            alembic_config(
                connection=connection, script_location=script_location
            ) as config,
        ):
            command.upgrade(config, "head")
    finally:
        engine.dispose()
    return SpoolRevisionState.BEHIND


@dataclass
class _ModelSpool:
    stamp: int | None = None
    lineage: frozenset[int] | None = None


def _model_open(
    parents: list[int | None],
    spool: _ModelSpool,
    node: int,
    records_lineage: bool,
    *,
    needs_no_lineage: frozenset[int],
) -> tuple[SpoolRevisionState, bool]:
    """What one open should find and whether it should write; updates ``spool``.

    A runner from before lineage was recorded upgrades without recording it and
    refuses every revision it does not know. A lineage-recording runner records
    it on each upgrade, repairs it at its own head unless that revision is one
    every history starts from (``needs_no_lineage``), and accepts an unknown
    revision only on the recorded lineage.
    """

    ancestry = _ancestry(parents, node)
    if spool.stamp == node:
        repair = (
            records_lineage
            and node not in needs_no_lineage
            and spool.lineage != ancestry
        )
        if repair:
            spool.lineage = ancestry
        return SpoolRevisionState.AT_HEAD, repair
    if spool.stamp is None or spool.stamp in ancestry:
        spool.stamp = node
        if records_lineage:
            spool.lineage = ancestry
        return SpoolRevisionState.BEHIND, True
    if (
        records_lineage
        and spool.lineage is not None
        and {spool.stamp, node} <= spool.lineage
    ):
        return SpoolRevisionState.AHEAD, False
    return SpoolRevisionState.INCOMPATIBLE, False


def _open_in_order(
    root: Path, parents: list[int | None], opens: list[tuple[int, bool]]
) -> tuple[list[SpoolRevisionState], int | None]:
    """Open one spool as each checkout in turn, checking the model after each.

    Each open is a revision and whether that checkout's runner records
    lineage. Returns each open's state and the revision the spool ends at.
    """

    names = [_packaged_head()] + [f"node_{index}" for index in range(1, len(parents))]
    revisions = {
        node: _additive_revision(names[node], names[parents[node]])
        for node in range(1, len(parents))
    }
    checkouts: dict[int, Path | None] = {0: None}
    for node in range(1, len(parents)):
        lineage = sorted(_ancestry(parents, node) - {0})
        checkouts[node] = _checkout(
            root / f"checkout_{node}", *(revisions[ancestor] for ancestor in lineage)
        )
    spool_path = root / "events.sqlite3"
    sqlite3.connect(spool_path).close()
    observer = sqlite3.connect(spool_path)
    model = _ModelSpool()
    states: list[SpoolRevisionState] = []
    # Revision 0 stands for the whole packaged history, whatever its length.
    packaged = migration_history()
    needs_no_lineage = frozenset({0} if names[0] in packaged.bases else ())
    try:
        for node, records_lineage in opens:
            before = _data_version(observer)
            expected, writes = _model_open(
                parents,
                model,
                node,
                records_lineage,
                needs_no_lineage=needs_no_lineage,
            )
            if records_lineage:
                engine = create_spool_engine(spool_path)
                try:
                    state = upgrade_spool_database(
                        engine, script_location=checkouts[node]
                    )
                except IncompatibleSpoolRevisionError:
                    state = SpoolRevisionState.INCOMPATIBLE
                finally:
                    engine.dispose()
            else:
                state = _open_as_runner_without_lineage(spool_path, checkouts[node])
            assert state is expected
            runner = "a lineage-recording" if records_lineage else "an older"
            event(f"{runner} runner found the spool {state}")
            if writes and state is SpoolRevisionState.AT_HEAD:
                event("a checkout at head recorded a missing or stale lineage")
            states.append(state)
            # Only an upgrade or a lineage repair writes; every other open reads.
            assert (_data_version(observer) != before) == writes
            assert model.stamp is not None
            assert current_database_revision(spool_path) == names[model.stamp]
            assert recorded_spool_lineage(spool_path) == (
                None
                if model.lineage is None
                else packaged.revisions | {names[other] for other in model.lineage}
            )
            # The schema is exactly that of the spool's revision: refused
            # checkouts added nothing.
            applied = _ancestry(parents, model.stamp) - {0}
            synthetic = range(1, len(parents))
            tables = _tables(spool_path)
            columns = _runs_columns(spool_path)
            assert {
                other
                for other in synthetic
                if f"graph_publication_jobs_{names[other]}" in tables
            } == applied
            assert {
                other for other in synthetic if f"region_{names[other]}" in columns
            } == applied
            if state is not SpoolRevisionState.INCOMPATIBLE:
                # Everything this checkout's schema has is in the spool.
                assert _ancestry(parents, node) - {0} <= applied
    finally:
        observer.close()
    return states, model.stamp


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    parents=_revision_trees(max_revisions=3),
    opens=st.lists(
        st.tuples(st.integers(min_value=0, max_value=3), st.booleans()),
        min_size=1,
        max_size=8,
    ),
)
def test_opens_in_any_order_by_checkouts_on_a_revision_tree_match_the_model(
    tmp_path_factory, parents, opens
) -> None:
    """Including checkouts whose runner predates lineage, and never records it."""

    states, _ = _open_in_order(
        tmp_path_factory.mktemp("tree"),
        parents,
        [(node % len(parents), records_lineage) for node, records_lineage in opens],
    )
    assert len(states) == len(opens)


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    length=st.integers(min_value=1, max_value=4),
    opens=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=5),
)
def test_lineage_recording_checkouts_on_one_line_of_history_are_never_refused(
    tmp_path_factory, length, opens
) -> None:
    parents = [None] + list(range(length))
    opens = [node % len(parents) for node in opens]
    states, stamp = _open_in_order(
        tmp_path_factory.mktemp("line"), parents, [(node, True) for node in opens]
    )
    assert SpoolRevisionState.INCOMPATIBLE not in states
    # The spool ends at the newest revision any checkout brought.
    assert stamp == max(opens)


# --- Every revision is additive ------------------------------------------------
#
# The check is a guard, not a proof that older code keeps working. It holds a
# migration to a short list of operations on the tables that existed before
# it, and confirms that rows the real spool code stored survive it and can
# still be written naming only the old columns.


@dataclass(frozen=True)
class _TableShape:
    sql: str
    columns: dict[str, tuple[object, ...]]
    indexes: frozenset[tuple[object, ...]]
    triggers: frozenset[tuple[str, str]]


def _shape(path: Path) -> dict[str, _TableShape]:
    """Every table's stored definition, columns, indexes and triggers."""

    connection = sqlite3.connect(path)
    try:
        shapes: dict[str, _TableShape] = {}
        for table, sql in connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            if table in _BOOKKEEPING_TABLES:
                continue
            columns = {
                name: (declared.upper(), notnull, default, primary_key, hidden)
                for _, name, declared, notnull, default, primary_key, hidden in (
                    connection.execute(f'PRAGMA table_xinfo("{table}")')
                )
            }
            indexes = set()
            for _, index, unique, origin, partial in connection.execute(
                f'PRAGMA index_list("{table}")'
            ).fetchall():
                keys = tuple(
                    (name, descending, collation)
                    for _, _, name, descending, collation, key in connection.execute(
                        f'PRAGMA index_xinfo("{index}")'
                    )
                    if key
                )
                (index_sql,) = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                    (index,),
                ).fetchone()
                # SQLite numbers the indexes behind PRIMARY KEY and UNIQUE
                # itself, so they are compared by definition, not name.
                indexes.add(
                    (index if origin == "c" else None, unique, origin, partial)
                    + (keys, index_sql)
                )
            shapes[table] = _TableShape(
                sql=sql,
                columns=columns,
                indexes=frozenset(indexes),
                triggers=frozenset(
                    connection.execute(
                        "SELECT name, sql FROM sqlite_master "
                        "WHERE type = 'trigger' AND tbl_name = ?",
                        (table,),
                    )
                ),
            )
        return shapes
    finally:
        connection.close()


def _inserted_text(before: str, after: str) -> str | None:
    """What ``after`` adds to ``before`` in one place, or ``None`` otherwise.

    ``ALTER TABLE … ADD COLUMN`` inserts the column into the stored definition
    after the last column, ahead of any table constraints. A rename, a drop
    or a rebuilt table changes the definition some other way.
    """

    prefix = len(os.path.commonprefix([before, after]))
    suffix = len(os.path.commonprefix([before[::-1], after[::-1]]))
    suffix = min(suffix, len(before) - prefix, len(after) - prefix)
    if prefix + suffix != len(before):
        return None
    return after[prefix : len(after) - suffix]


_CONSTRAINT_OR_EXPRESSION = re.compile(
    r"\b(CHECK|UNIQUE|PRIMARY|GENERATED|AS)\b", re.IGNORECASE
)


def _shape_violations(
    before: dict[str, _TableShape], after: dict[str, _TableShape]
) -> list[str]:
    problems: list[str] = []
    for table, old in before.items():
        new = after.get(table)
        if new is None:
            problems.append(f"{table}: removed")
            continue
        added = _inserted_text(old.sql, new.sql)
        if added is None:
            problems.append(f"{table}: definition changed, not only extended")
        elif _CONSTRAINT_OR_EXPRESSION.search(added):
            problems.append(f"{table}: new columns carry a constraint or expression")
        for column in old.columns.keys() - new.columns.keys():
            problems.append(f"{table}.{column}: removed")
        for column, definition in new.columns.items():
            if column in old.columns:
                if old.columns[column] != definition:
                    problems.append(f"{table}.{column}: changed")
                continue
            _, notnull, default, primary_key, hidden = definition
            if hidden:
                problems.append(f"{table}.{column}: new column is generated")
            if primary_key:
                problems.append(f"{table}.{column}: new column is in the primary key")
            if notnull and default is None:
                problems.append(
                    f"{table}.{column}: new column is NOT NULL without a default"
                )
        for index in sorted(old.indexes - new.indexes, key=repr):
            problems.append(f"{table}: index {index[0] or index[4]} removed or changed")
        for name, unique, _, partial, keys, _ in sorted(
            new.indexes - old.indexes, key=repr
        ):
            if unique:
                problems.append(f"{table}: new index {name or keys} is unique")
            if partial:
                problems.append(f"{table}: new index {name or keys} is partial")
            if any(column is None for column, _, _ in keys):
                problems.append(
                    f"{table}: new index {name or keys} is on an expression"
                )
        if new.triggers != old.triggers:
            problems.append(f"{table}: triggers changed")
    return problems


_NAME = r'(?:["`\[]?\w+["`\]]?\s*\.\s*)?["`\[]?(\w+)["`\]]?'
_ROW_STATEMENTS = (
    (
        "INSERT",
        re.compile(rf"(?:INSERT|REPLACE)(?:\s+OR\s+\w+)?\s+INTO\s+{_NAME}", re.I),
    ),
    ("UPDATE", re.compile(rf"UPDATE(?:\s+OR\s+\w+)?\s+{_NAME}", re.I)),
    ("DELETE", re.compile(rf"DELETE\s+FROM\s+{_NAME}", re.I)),
)
_ALTER_TABLE = re.compile(rf"ALTER\s+TABLE\s+{_NAME}\s+(\w+)", re.I)
_DROP = re.compile(rf"DROP\s+(TABLE|INDEX|TRIGGER)\s+(?:IF\s+EXISTS\s+)?{_NAME}", re.I)
_CREATE_TRIGGER = re.compile(
    rf"CREATE\s+(?:TEMP\s+|TEMPORARY\s+)?TRIGGER\b.*?\bON\s+{_NAME}", re.I | re.S
)
_HARMLESS_STATEMENT = re.compile(
    r"(BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE|PRAGMA|SELECT|ANALYZE|"
    r"CREATE\s+(?:UNIQUE\s+)?INDEX|CREATE\s+TABLE|CREATE\s+VIEW)\b",
    re.I,
)


def _statement_violations(
    statements: list[str], before: dict[str, _TableShape]
) -> list[str]:
    """Statements the migration ran that the policy does not allow.

    On a table that already existed, only ``ALTER TABLE … ADD COLUMN`` and
    ``CREATE INDEX`` are allowed: no inserted, updated or deleted rows, no
    other ``ALTER``, no ``DROP`` and no trigger. Whatever a statement would
    spare of the rows this check seeds, it is refused for its kind.
    """

    indexes = {
        index[0]: table
        for table, shape in before.items()
        for index in shape.indexes
        if index[0]
    }
    triggers = {
        name: table for table, shape in before.items() for name, _ in shape.triggers
    }
    refused = "statement not allowed on an existing table"
    problems: list[str] = []
    for statement in statements:
        statement = statement.strip()
        if statement.startswith("--"):
            continue  # SQLite's own checks, traced as comments
        rows = [
            (kind, found.group(1))
            for kind, pattern in _ROW_STATEMENTS
            # A WITH clause can precede the verb, so look past the start.
            for found in pattern.finditer(statement)
            if statement.upper().startswith(("WITH", kind, "REPLACE"))
        ]
        if rows:
            problems += [
                f"{table}: {refused}: {kind}" for kind, table in rows if table in before
            ]
        elif found := _ALTER_TABLE.match(statement):
            table, action = found.groups()
            if table in before and action.upper() != "ADD":
                problems.append(f"{table}: {refused}: ALTER TABLE {action.upper()}")
        elif found := _DROP.match(statement):
            kind, name = found.group(1).upper(), found.group(2)
            table = {"TABLE": name, "INDEX": indexes.get(name)}.get(
                kind, triggers.get(name)
            )
            if table in before:
                problems.append(f"{table}: {refused}: DROP {kind} {name}")
        elif found := _CREATE_TRIGGER.match(statement):
            if found.group(1) in before:
                problems.append(f"{found.group(1)}: {refused}: CREATE TRIGGER")
        elif not _HARMLESS_STATEMENT.match(statement):
            problems.append(f"statement to review by hand: {statement[:60]}")
    return problems


@functools.cache
def _stored_by_the_spool(label: str) -> dict[str, list[dict[str, object]]]:
    """The rows this checkout's spool code stores for one build's runs.

    A pending run with two events, a local-only run with one and a registered
    run with none, read back as stored: valid JSON, both upload states.
    """

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.sqlite3"
        spool = EventSpool(path)
        try:
            pending = _registration(f"{label}-pending", f"{label}-producer")
            spool.register(pending)
            for stage in "ab":
                spool.append(pending, _event(stage))
            local = _registration(f"{label}-local", f"{label}-producer")
            spool.register(local)
            spool.append(local, _event("a"))
            spool.make_local_only(f"{label}-local", f"{label}-producer", "test")
            spool.register(_registration(f"{label}-idle", f"{label}-producer"))
        finally:
            _close(spool)
        return _stored_rows(path)


# Rows for tables the spool code above does not fill, by table name: a
# function from 0 (a row to seed) or 1 (a row to write afterwards) to a valid
# row. A revision that adds a table whose constraints refuse arbitrary values
# gives it rows here, so that the revision after it can be checked.
_ROWS_FOR_OTHER_TABLES: dict[str, Callable[[int], dict[str, object]]] = {}


def _sample(declared: str, index: int) -> object:
    """A value of a column's affinity, distinct for each ``index``."""

    if "INT" in declared:
        return index + 1
    if any(part in declared for part in ("CHAR", "CLOB", "TEXT")):
        return f"value-{index}"
    if not declared or "BLOB" in declared:
        return f"value-{index}".encode()
    if any(part in declared for part in ("REAL", "FLOA", "DOUB")):
        return index + 0.5
    return index + 1


def _writable_columns(shape: _TableShape) -> list[str]:
    return [name for name, definition in shape.columns.items() if not definition[4]]


def _rows(table: str, shape: _TableShape, variant: int) -> list[list[object]]:
    """Rows for one table, each in the order of its writable columns."""

    columns = _writable_columns(shape)
    stored = _stored_by_the_spool(("seeded", "written")[variant])
    if table in stored:
        return [[row[column] for column in columns] for row in stored[table]]
    if table in _ROWS_FOR_OTHER_TABLES:
        row = _ROWS_FOR_OTHER_TABLES[table](variant)
        return [[row[column] for column in columns]]
    return [[_sample(shape.columns[column][0], variant) for column in columns]]


def _insert(
    connection: sqlite3.Connection, table: str, shape: _TableShape, row: list[object]
) -> None:
    columns = _writable_columns(shape)
    connection.execute(
        f'INSERT INTO "{table}" ({", ".join(f'"{name}"' for name in columns)}) '
        f"VALUES ({', '.join('?' for _ in columns)})",
        row,
    )


def _old_rows(
    connection: sqlite3.Connection, table: str, shape: _TableShape
) -> list[tuple[object, ...]]:
    columns = ", ".join(f'"{name}"' for name in _writable_columns(shape))
    return sorted(connection.execute(f'SELECT {columns} FROM "{table}"'), key=repr)


def _write_violations(path: Path, before: dict[str, _TableShape]) -> list[str]:
    """Insert, update and delete rows in each table naming only old columns."""

    problems: list[str] = []
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        for table, shape in before.items():
            columns = _writable_columns(shape)
            key = [name for name in columns if shape.columns[name][3]] or columns
            others = [name for name in columns if name not in key]
            where = " AND ".join(f'"{name}" = ?' for name in key)
            try:
                for row in _rows(table, shape, 1):
                    values = dict(zip(columns, row, strict=True))
                    key_values = [values[name] for name in key]
                    _insert(connection, table, shape, row)
                    if others:
                        connection.execute(
                            f'UPDATE "{table}" SET '
                            + ", ".join(f'"{name}" = ?' for name in others)
                            + f" WHERE {where}",
                            [values[name] for name in others] + key_values,
                        )
                    connection.execute(
                        f'DELETE FROM "{table}" WHERE {where}', key_values
                    )
            except sqlite3.Error as error:
                problems.append(
                    f"{table}: a write naming only its existing columns failed: {error}"
                )
    finally:
        connection.close()
    return problems


def _additivity_violations(
    path: Path, migrate: Callable[[Callable[[str], None]], None]
) -> list[str]:
    """Why ``migrate`` breaks the additive-only rule on ``path``, if it does.

    ``migrate`` runs the migration in one transaction and passes every
    statement SQLite executes to the function it is given. Before it, each
    table is given the rows the real spool code stores, or arbitrary rows for
    tables that code does not fill.
    """

    before = _shape(path)
    connection = sqlite3.connect(path)
    try:
        for table, shape in before.items():
            for row in _rows(table, shape, 0):
                try:
                    _insert(connection, table, shape, row)
                except sqlite3.Error as error:
                    raise AssertionError(
                        f"give table {table!r} valid rows in _ROWS_FOR_OTHER_TABLES: "
                        f"it refuses arbitrary ones ({error})"
                    ) from error
        connection.commit()
        seeded = {
            table: _old_rows(connection, table, shape)
            for table, shape in before.items()
        }
    finally:
        connection.close()

    statements: list[str] = []
    try:
        migrate(statements.append)
    except Exception as error:
        reason = (str(error).strip().splitlines() or [type(error).__name__])[0]
        return [f"the migration failed in its transaction: {reason}"]

    after = _shape(path)
    problems = _statement_violations(statements, before)
    problems += _shape_violations(before, after)
    connection = sqlite3.connect(path)
    try:
        for table, shape in before.items():
            try:
                rows = _old_rows(connection, table, shape)
            except sqlite3.Error:
                continue  # the table or a column went, which is reported above
            if rows != seeded[table]:
                problems.append(f"{table}: existing rows changed")
    finally:
        connection.close()
    surviving = {
        table: shape
        for table, shape in before.items()
        if table in after and set(shape.columns) <= set(after[table].columns)
    }
    return problems + _write_violations(path, surviving)


def _revision_additivity_violations(
    script_location: Path | None, revision: str, parent: str, directory: Path
) -> list[str]:
    directory.mkdir(parents=True)
    path = directory / "events.sqlite3"
    # As upgrade_spool_database migrates: one BEGIN IMMEDIATE transaction.
    engine = create_spool_engine(path, immediate_transactions=True)

    def upgrade_to(target: str, trace: Callable[[str], None] | None = None) -> None:
        with (
            engine.begin() as connection,
            alembic_config(
                connection=connection, script_location=script_location
            ) as config,
        ):
            driver_connection = connection.connection.driver_connection
            driver_connection.set_trace_callback(trace)
            try:
                command.upgrade(config, target)
            finally:
                driver_connection.set_trace_callback(None)

    try:
        upgrade_to(parent)
        return _additivity_violations(path, lambda trace: upgrade_to(revision, trace))
    finally:
        engine.dispose()


def _parents(down_revision: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if down_revision is None:
        return ()
    return (down_revision,) if isinstance(down_revision, str) else down_revision


def test_every_packaged_revision_after_the_initial_one_is_additive(tmp_path) -> None:
    """The initial revision creates the schema or adopts a pre-Alembic spool."""

    with alembic_config() as config:
        scripts = list(ScriptDirectory.from_config(config).walk_revisions())
    roots = {script.revision for script in scripts if script.down_revision is None}
    assert roots == {_INITIAL_REVISION}
    for script in scripts:
        for parent in _parents(script.down_revision):
            directory = tmp_path / f"{script.revision}-from-{parent}"
            assert (
                _revision_additivity_violations(
                    None, script.revision, parent, directory
                )
                == []
            ), script.revision


def test_a_later_additive_revision_passes_the_same_check(tmp_path) -> None:
    newer = _checkout(tmp_path / "newer", _additive_revision("future"))
    assert (
        _revision_additivity_violations(
            newer, "future", _packaged_head(), tmp_path / "check"
        )
        == []
    )


# The generated column and the conditional statements are the ones an earlier
# version of this check passed: its seeded rows held no JSON and no real
# upload state, so they neither tripped the expression nor matched the WHERE.
_OVERFLOW_ON_VALID_JSON = (
    "ALTER TABLE telemetry_events ADD COLUMN synthetic_overflow INTEGER GENERATED ALWAYS "
    "AS (CASE WHEN json_valid(payload_json) THEN abs(-9223372036854775808) "
    "ELSE 0 END) VIRTUAL"
)
_NULL_FOR_PENDING_RUNS = (
    "ALTER TABLE telemetry_runs ADD COLUMN synthetic_gate INT GENERATED ALWAYS AS "
    "(CASE WHEN upload_state = 'pending' THEN NULL ELSE 1 END) VIRTUAL NOT NULL"
)
_DELETE_VALID_EVENTS = "DELETE FROM telemetry_events WHERE json_valid(payload_json)"
_HOLD_PENDING_RUNS = (
    "UPDATE telemetry_runs SET upload_state = 'held' WHERE upload_state = 'pending'"
)

_BREAKING_REVISIONS = {
    "drop-column": (
        'op.drop_column("telemetry_runs", "local_only_reason")',
        "telemetry_runs.local_only_reason: removed",
    ),
    "unique-index": (
        'op.create_index("synthetic_unique_index", "telemetry_runs", '
        '["run_id", "updated_at"], unique=True)',
        "telemetry_runs: new index synthetic_unique_index is unique",
    ),
    "required-column": (
        """
        with op.batch_alter_table("telemetry_runs", recreate="always") as batch:
            batch.add_column(
                sa.Column("synthetic_note", sa.Text(), nullable=False, server_default="x")
            )
        with op.batch_alter_table("telemetry_runs", recreate="always") as batch:
            batch.alter_column("synthetic_note", server_default=None)
        """,
        "telemetry_runs.synthetic_note: new column is NOT NULL without a default",
    ),
    "type-change": (
        """
        with op.batch_alter_table("telemetry_events") as batch:
            batch.alter_column("sequence", type_=sa.Text())
        """,
        "telemetry_events.sequence: changed",
    ),
    "rewritten-rows": (
        "op.execute(\"UPDATE telemetry_runs SET upload_state = 'held'\")",
        "telemetry_runs: existing rows changed",
    ),
    "conditional-update": (
        f"op.execute({_HOLD_PENDING_RUNS!r})",
        "telemetry_runs: statement not allowed on an existing table: UPDATE",
    ),
    "conditional-delete": (
        f"op.execute({_DELETE_VALID_EVENTS!r})",
        "telemetry_events: statement not allowed on an existing table: DELETE",
    ),
    "generated-column": (
        f"op.execute({_OVERFLOW_ON_VALID_JSON!r})",
        "telemetry_events.synthetic_overflow: new column is generated",
    ),
    "vacuum": (
        """
        op.create_table("synthetic_table", sa.Column("body", sa.Text()))
        op.execute("VACUUM")
        """,
        "the migration failed in its transaction",
    ),
}


@pytest.mark.parametrize("name", sorted(_BREAKING_REVISIONS))
def test_the_additivity_check_rejects_a_breaking_alembic_revision(
    tmp_path, name
) -> None:
    body, problem = _BREAKING_REVISIONS[name]
    branch = _checkout(
        tmp_path / "branch", _Revision("breaking", _packaged_head(), body)
    )
    problems = _revision_additivity_violations(
        branch, "breaking", _packaged_head(), tmp_path / "check"
    )
    assert any(found.startswith(problem) for found in problems), problems


def _recreate(
    connection: sqlite3.Connection,
    table: str,
    edit: Callable[[str], str],
    *,
    leading_values: str = "",
) -> None:
    """SQLite's documented twelve-step table change, as Alembic's batch mode."""

    (sql,) = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    indexes = [
        index_sql
        for (index_sql,) in connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL",
            (table,),
        )
    ]
    # SQLite quotes the name in the definition it stores after a rename.
    renamed, count = re.subn(
        rf'^CREATE TABLE "?{table}"? \(', f"CREATE TABLE {table}_new (", edit(sql)
    )
    assert count == 1, sql
    connection.execute(renamed)
    connection.execute(f"INSERT INTO {table}_new SELECT {leading_values}* FROM {table}")
    connection.execute(f"DROP TABLE {table}")
    connection.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
    for index_sql in indexes:
        connection.execute(index_sql)


def _edited(old: str, new: str) -> Callable[[str], str]:
    def edit(sql: str) -> str:
        assert old in sql, sql
        return sql.replace(old, new, 1)

    return edit


def _sql(*statements: str) -> Callable[[sqlite3.Connection], object]:
    def apply(connection: sqlite3.Connection) -> None:
        for statement in statements:
            connection.execute(statement)

    return apply


_ALLOWED_CHANGES: dict[str, Callable[[sqlite3.Connection], object]] = {
    "new-table": _sql(
        "CREATE TABLE synthetic_jobs "
        "(publication_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
    ),
    "new-table-with-check": _sql(
        "CREATE TABLE synthetic_states (state TEXT NOT NULL "
        "CHECK (state IN ('queued', 'done')))"
    ),
    "new-table-referencing-runs": _sql(
        "CREATE TABLE synthetic_receipts (run_id TEXT, producer_id TEXT, "
        "FOREIGN KEY (run_id, producer_id) "
        "REFERENCES telemetry_runs (run_id, producer_id))"
    ),
    "new-table-filled-from-old-rows": _sql(
        "CREATE TABLE synthetic_notes (run_id TEXT, note TEXT)",
        "INSERT INTO synthetic_notes (run_id) SELECT run_id FROM telemetry_runs",
    ),
    "nullable-column": _sql(
        "ALTER TABLE telemetry_runs ADD COLUMN synthetic_region TEXT"
    ),
    "defaulted-column": _sql(
        "ALTER TABLE telemetry_events ADD COLUMN synthetic_attempts INTEGER NOT NULL DEFAULT 0"
    ),
    "plain-index": _sql(
        "CREATE INDEX synthetic_plain_index ON telemetry_runs (updated_at)"
    ),
}

# Each maps to a problem it must be reported as when applied alone. They are
# applied in this order, after any allowed changes.
_FORBIDDEN_CHANGES: dict[str, tuple[Callable[[sqlite3.Connection], object], str]] = {
    "drop-column": (
        _sql("ALTER TABLE telemetry_runs DROP COLUMN local_only_reason"),
        "telemetry_runs.local_only_reason: removed",
    ),
    "rename-column": (
        _sql("ALTER TABLE telemetry_events RENAME COLUMN created_at TO queued_at"),
        "telemetry_events: statement not allowed on an existing table: "
        "ALTER TABLE RENAME",
    ),
    "drop-index": (
        _sql("DROP INDEX telemetry_events_run_sequence"),
        "telemetry_events: index telemetry_events_run_sequence removed",
    ),
    "replaced-index": (
        _sql(
            "DROP INDEX telemetry_events_run_sequence",
            "CREATE INDEX telemetry_events_run_sequence "
            "ON telemetry_events (run_id, producer_id, sequence)",
        ),
        "telemetry_events: statement not allowed on an existing table: DROP INDEX",
    ),
    "unique-index": (
        _sql(
            "CREATE UNIQUE INDEX synthetic_unique_index ON telemetry_runs (run_id, updated_at)"
        ),
        "telemetry_runs: new index synthetic_unique_index is unique",
    ),
    "partial-index": (
        _sql(
            "CREATE INDEX synthetic_partial_index ON telemetry_events (created_at) "
            "WHERE sequence > 0"
        ),
        "telemetry_events: new index synthetic_partial_index is partial",
    ),
    "expression-index": (
        _sql(
            "CREATE INDEX synthetic_expression_index ON telemetry_events (length(payload_json))"
        ),
        "telemetry_events: new index synthetic_expression_index is on an expression",
    ),
    "trigger": (
        _sql(
            "CREATE TRIGGER synthetic_trigger AFTER INSERT ON telemetry_runs "
            "BEGIN SELECT 1; END"
        ),
        "telemetry_runs: statement not allowed on an existing table: CREATE TRIGGER",
    ),
    "rewritten-rows": (
        _sql("UPDATE telemetry_runs SET upload_state = 'held'"),
        "telemetry_runs: existing rows changed",
    ),
    "conditional-update": (
        _sql(_HOLD_PENDING_RUNS),
        "telemetry_runs: existing rows changed",
    ),
    "deleted-rows": (
        _sql("DELETE FROM telemetry_events"),
        "telemetry_events: existing rows changed",
    ),
    "conditional-delete": (
        _sql(_DELETE_VALID_EVENTS),
        "telemetry_events: existing rows changed",
    ),
    "delete-through-a-with-clause": (
        _sql(
            "WITH doomed AS (SELECT event_id FROM telemetry_events) "
            "DELETE FROM telemetry_events WHERE event_id IN (SELECT event_id FROM doomed)"
        ),
        "telemetry_events: statement not allowed on an existing table: DELETE",
    ),
    "inserted-rows": (
        _sql(
            "INSERT INTO telemetry_runs "
            "(run_id, producer_id, registration_json, updated_at) "
            "VALUES ('extra', 'extra', '{}', '2026-10-09T00:00:00+00:00')"
        ),
        "telemetry_runs: statement not allowed on an existing table: INSERT",
    ),
    "generated-column": (
        _sql(_OVERFLOW_ON_VALID_JSON),
        "telemetry_events.synthetic_overflow: new column is generated",
    ),
    "constrained-generated-column": (
        _sql(_NULL_FOR_PENDING_RUNS),
        "the migration failed in its transaction: NOT NULL constraint failed",
    ),
    "column-with-check": (
        _sql(
            "ALTER TABLE telemetry_events ADD COLUMN synthetic_flag INTEGER DEFAULT 0 "
            "CHECK (synthetic_flag = 0)"
        ),
        "telemetry_events: new columns carry a constraint or expression",
    ),
    "required-column": (
        lambda connection: _recreate(
            connection,
            "telemetry_runs",
            _edited("(\n\t", "(\n\tsynthetic_note TEXT NOT NULL, \n\t"),
            leading_values="'x', ",
        ),
        "telemetry_runs.synthetic_note: new column is NOT NULL without a default",
    ),
    "type-change": (
        lambda connection: _recreate(
            connection,
            "telemetry_events",
            _edited("sequence INTEGER NOT NULL", "sequence TEXT NOT NULL"),
        ),
        "telemetry_events.sequence: changed",
    ),
    "check-constraint": (
        lambda connection: _recreate(
            connection,
            "telemetry_events",
            lambda sql: sql[: sql.rindex(")")] + ", \n\tCHECK (sequence > 0)\n)",
        ),
        "telemetry_events: definition changed, not only extended",
    ),
    "collation": (
        lambda connection: _recreate(
            connection,
            "telemetry_runs",
            _edited("upload_state TEXT", "upload_state TEXT COLLATE RTRIM"),
        ),
        "telemetry_runs: definition changed, not only extended",
    ),
    "identical-rebuild": (
        lambda connection: _recreate(connection, "telemetry_runs", lambda sql: sql),
        "telemetry_runs: statement not allowed on an existing table: DROP TABLE",
    ),
    "drop-table": (
        _sql("DROP TABLE telemetry_events"),
        "telemetry_events: removed",
    ),
    "vacuum": (
        _sql("VACUUM"),
        "the migration failed in its transaction: cannot VACUUM",
    ),
}


@pytest.fixture(scope="module")
def spool_at_packaged_head(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("template") / "events.sqlite3"
    _close(EventSpool(path))
    return path


def _change(
    path: Path,
    allowed: list[str],
    forbidden: list[str],
    trace: Callable[[str], None] | None = None,
) -> None:
    """Apply the named changes in one ``BEGIN IMMEDIATE`` transaction."""

    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.set_trace_callback(trace)
        connection.execute("BEGIN IMMEDIATE")
        try:
            for name in _ALLOWED_CHANGES:
                if name in allowed:
                    _ALLOWED_CHANGES[name](connection)
            for name, (apply, _) in _FORBIDDEN_CHANGES.items():
                if name in forbidden:
                    apply(connection)
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()


@pytest.mark.parametrize("name", list(_FORBIDDEN_CHANGES))
def test_the_additivity_check_names_each_forbidden_change(
    tmp_path, spool_at_packaged_head, name
) -> None:
    path = tmp_path / "events.sqlite3"
    shutil.copyfile(spool_at_packaged_head, path)
    problems = _additivity_violations(
        path, lambda trace: _change(path, [], [name], trace)
    )
    assert any(found.startswith(_FORBIDDEN_CHANGES[name][1]) for found in problems), (
        problems
    )


@settings(max_examples=80, deadline=None)
@given(
    allowed=st.lists(st.sampled_from(sorted(_ALLOWED_CHANGES)), unique=True),
    forbidden=st.lists(st.sampled_from(sorted(_FORBIDDEN_CHANGES)), unique=True),
)
def test_the_additivity_check_flags_exactly_the_forbidden_changes(
    tmp_path_factory, spool_at_packaged_head, allowed, forbidden
) -> None:
    """And on whatever it accepts, this checkout's spool code still works."""

    directory = tmp_path_factory.mktemp("changes")
    checked = directory / "checked.sqlite3"
    shutil.copyfile(spool_at_packaged_head, checked)
    problems = _additivity_violations(
        checked, lambda trace: _change(checked, allowed, forbidden, trace)
    )
    assert bool(problems) == bool(forbidden), problems
    if forbidden:
        return

    # A newer checkout stamped this spool after making exactly these changes.
    used = directory / "used.sqlite3"
    shutil.copyfile(spool_at_packaged_head, used)
    _change(used, allowed, [])
    _stamp(used, "future", migration_history().revisions | {"future"})
    spool = EventSpool(used)
    try:
        registration = _registration("run", "producer")
        spool.register(registration)
        queued = [spool.append(registration, _event(stage)) for stage in "ab"]
        assert spool.pending_runs() == [registration]
        assert spool.batch("run", "producer") == queued
        spool.acknowledge([queued[0]["event_id"]])
        spool.make_local_only("run", "producer", "test")
        spool.prune()
        assert spool.batch("run", "producer") == queued[1:]
        assert not spool.has_deliverable()
    finally:
        _close(spool)
