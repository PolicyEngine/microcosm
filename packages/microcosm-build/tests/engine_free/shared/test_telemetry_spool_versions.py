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
from sqlalchemy import create_engine, inspect

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun
from microcosm.build.telemetry_emitter_constants import TELEMETRY_SERVICE_MODULE
from microcosm.build.telemetry_emitter_service import collector as collector_module
from microcosm.build.telemetry_emitter_service import migrations as migrations_module
from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DATABASE_TIMEOUT_SECONDS,
    SPOOL_LINEAGE_TABLE,
)
from microcosm.build.telemetry_emitter_service.database import (
    create_spool_engine,
    sqlite_database_url,
)
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
                    engine, busy_timeout_seconds=60, script_location=opener
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
    socket_path = Path(tempfile.mkdtemp(prefix="microcosm-test-", dir="/tmp")) / "s"
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


def _open_in_order(
    root: Path, parents: list[int | None], opens: list[int]
) -> tuple[list[SpoolRevisionState], int | None]:
    """Open one spool as each checkout in turn, checking the model after each.

    Returns each open's state and the revision the spool ends at.
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
    stamp: int | None = None
    states: list[SpoolRevisionState] = []
    try:
        for node in opens:
            before = _data_version(observer)
            engine = create_spool_engine(spool_path)
            try:
                state = upgrade_spool_database(engine, script_location=checkouts[node])
            except IncompatibleSpoolRevisionError:
                state = SpoolRevisionState.INCOMPATIBLE
            finally:
                engine.dispose()
            assert state is _expected_state(parents, stamp, node)
            event(f"an open found the spool {state}")
            states.append(state)
            if state is SpoolRevisionState.BEHIND:
                stamp = node
            else:
                # Only an upgrade writes; every other open just reads.
                assert _data_version(observer) == before
            assert current_database_revision(spool_path) == names[stamp]
            spool_lineage = _ancestry(parents, stamp)
            assert recorded_spool_lineage(spool_path) == {
                names[ancestor] for ancestor in spool_lineage
            }
            # The schema is exactly the spool's lineage's: refused checkouts
            # added nothing.
            synthetic = range(1, len(parents))
            tables = _tables(spool_path)
            columns = _runs_columns(spool_path)
            assert {
                node
                for node in synthetic
                if f"graph_publication_jobs_{names[node]}" in tables
            } == spool_lineage - {0}
            assert {
                node for node in synthetic if f"region_{names[node]}" in columns
            } == spool_lineage - {0}
            if state is not SpoolRevisionState.INCOMPATIBLE:
                # Everything this checkout's schema has is in the spool.
                needed = _ancestry(parents, node) - {0}
                assert {f"graph_publication_jobs_{names[n]}" for n in needed} <= tables
                assert {f"region_{names[n]}" for n in needed} <= columns
    finally:
        observer.close()
    return states, stamp


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    parents=_revision_trees(max_revisions=4),
    opens=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=5),
)
def test_opens_in_any_order_by_checkouts_on_a_revision_tree_match_the_model(
    tmp_path_factory, parents, opens
) -> None:
    states, _ = _open_in_order(
        tmp_path_factory.mktemp("tree"),
        parents,
        [node % len(parents) for node in opens],
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
def test_checkouts_on_one_line_of_history_are_never_refused(
    tmp_path_factory, length, opens
) -> None:
    parents = [None] + list(range(length))
    opens = [node % len(parents) for node in opens]
    states, stamp = _open_in_order(tmp_path_factory.mktemp("line"), parents, opens)
    assert SpoolRevisionState.INCOMPATIBLE not in states
    # The spool ends at the newest revision any checkout brought.
    assert stamp == max(opens)


# --- Every revision is additive ------------------------------------------------


@dataclass(frozen=True)
class _TableShape:
    columns: dict[str, tuple[object, ...]]
    indexes: frozenset[tuple[object, ...]]
    foreign_keys: frozenset[tuple[object, ...]]
    checks: frozenset[str]
    triggers: frozenset[tuple[str, str]]


def _shape(path: Path) -> dict[str, _TableShape]:
    """Every table's columns, indexes, foreign keys, checks and triggers."""

    connection = sqlite3.connect(path)
    engine = create_engine(sqlite_database_url(path))
    try:
        reflection = inspect(engine)
        shapes: dict[str, _TableShape] = {}
        for (table,) in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ):
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
            ):
                keys = tuple(
                    (name, descending, collation)
                    for _, _, name, descending, collation, key in connection.execute(
                        f'PRAGMA index_xinfo("{index}")'
                    )
                    if key
                )
                (sql,) = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                    (index,),
                ).fetchone()
                # SQLite numbers the indexes behind PRIMARY KEY and UNIQUE
                # itself, so they are compared by definition, not name.
                indexes.add(
                    (index if origin == "c" else None, unique, origin, partial)
                    + (keys, sql)
                )
            references: dict[int, list[object]] = {}
            for (
                identifier,
                _,
                target,
                source_column,
                target_column,
                on_update,
                on_delete,
                match,
            ) in connection.execute(f'PRAGMA foreign_key_list("{table}")'):
                reference = references.setdefault(
                    identifier, [target, (), (), on_update, on_delete, match]
                )
                reference[1] += (source_column,)
                reference[2] += (target_column,)
            shapes[table] = _TableShape(
                columns=columns,
                indexes=frozenset(indexes),
                foreign_keys=frozenset(tuple(item) for item in references.values()),
                checks=frozenset(
                    str(check["sqltext"])
                    for check in reflection.get_check_constraints(table)
                ),
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
        engine.dispose()
        connection.close()


def _shape_violations(
    before: dict[str, _TableShape], after: dict[str, _TableShape]
) -> list[str]:
    problems: list[str] = []
    for table, old in before.items():
        new = after.get(table)
        if new is None:
            problems.append(f"{table}: removed")
            continue
        for column, definition in old.columns.items():
            if column not in new.columns:
                problems.append(f"{table}.{column}: removed")
            elif new.columns[column] != definition:
                problems.append(
                    f"{table}.{column}: changed from {definition} "
                    f"to {new.columns[column]}"
                )
        for column, (_, notnull, default, primary_key, hidden) in new.columns.items():
            if column in old.columns:
                continue
            if primary_key:
                problems.append(f"{table}.{column}: new column is in the primary key")
            if notnull and default is None and not hidden:
                problems.append(
                    f"{table}.{column}: new column is NOT NULL without a default"
                )
        for index in sorted(old.indexes - new.indexes, key=repr):
            problems.append(f"{table}: index {index[0] or index[4]} removed or changed")
        for index in sorted(new.indexes - old.indexes, key=repr):
            if index[1]:
                problems.append(f"{table}: new unique index {index[0] or index[4]}")
        if new.foreign_keys != old.foreign_keys:
            problems.append(f"{table}: foreign keys changed")
        if new.checks != old.checks:
            problems.append(f"{table}: CHECK constraints changed")
        if new.triggers != old.triggers:
            problems.append(f"{table}: triggers changed")
    return problems


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


def _insert(
    connection: sqlite3.Connection, table: str, shape: _TableShape, index: int
) -> None:
    columns = _writable_columns(shape)
    connection.execute(
        f'INSERT INTO "{table}" ({", ".join(f'"{name}"' for name in columns)}) '
        f"VALUES ({', '.join('?' for _ in columns)})",
        [_sample(shape.columns[name][0], index) for name in columns],
    )


def _old_rows(
    connection: sqlite3.Connection, table: str, shape: _TableShape
) -> list[tuple[object, ...]]:
    columns = ", ".join(f'"{name}"' for name in _writable_columns(shape))
    return sorted(connection.execute(f'SELECT {columns} FROM "{table}"'), key=repr)


def _write_violations(path: Path, before: dict[str, _TableShape]) -> list[str]:
    """Insert, update and delete a row in each table naming only old columns."""

    problems: list[str] = []
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        for table, shape in before.items():
            columns = _writable_columns(shape)
            key = [name for name in columns if shape.columns[name][3]] or columns
            others = [name for name in columns if name not in key]
            where = " AND ".join(f'"{name}" = ?' for name in key)
            key_values = [_sample(shape.columns[name][0], 1) for name in key]
            try:
                _insert(connection, table, shape, 1)
                if others:
                    connection.execute(
                        f'UPDATE "{table}" SET '
                        + ", ".join(f'"{name}" = ?' for name in others)
                        + f" WHERE {where}",
                        [_sample(shape.columns[name][0], 2) for name in others]
                        + key_values,
                    )
                connection.execute(f'DELETE FROM "{table}" WHERE {where}', key_values)
            except sqlite3.Error as error:
                problems.append(
                    f"{table}: a write naming only its existing columns failed: {error}"
                )
    finally:
        connection.close()
    return problems


def _additivity_violations(path: Path, migrate: Callable[[], None]) -> list[str]:
    """Why older code could not use ``path`` after ``migrate``, if it could not.

    Every table gets a row first. Afterwards each table must keep its columns,
    indexes, foreign keys, checks, triggers and rows; new columns must be
    nullable or defaulted and outside the primary key; new indexes on them
    must not be unique; and a row written naming only the old columns must
    insert, update and delete.
    """

    before = _shape(path)
    connection = sqlite3.connect(path)
    try:
        for table, shape in before.items():
            _insert(connection, table, shape, 0)
        connection.commit()
        seeded = {
            table: _old_rows(connection, table, shape)
            for table, shape in before.items()
        }
    finally:
        connection.close()

    migrate()

    after = _shape(path)
    problems = _shape_violations(before, after)
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
    engine = create_spool_engine(path)

    def upgrade_to(target: str) -> None:
        with (
            engine.begin() as connection,
            alembic_config(
                connection=connection, script_location=script_location
            ) as config,
        ):
            command.upgrade(config, target)

    try:
        upgrade_to(parent)
        return _additivity_violations(path, lambda: upgrade_to(revision))
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


_BREAKING_REVISIONS = {
    "drop-column": (
        'op.drop_column("telemetry_runs", "local_only_reason")',
        "telemetry_runs.local_only_reason: removed",
    ),
    "unique-index": (
        'op.create_index("runs_updated", "telemetry_runs", ["updated_at"], '
        "unique=True)",
        "telemetry_runs: new unique index runs_updated",
    ),
    "required-column": (
        """
        with op.batch_alter_table("telemetry_runs", recreate="always") as batch:
            batch.add_column(
                sa.Column("note", sa.Text(), nullable=False, server_default="x")
            )
        with op.batch_alter_table("telemetry_runs", recreate="always") as batch:
            batch.alter_column("note", server_default=None)
        """,
        "telemetry_runs.note: new column is NOT NULL without a default",
    ),
    "type-change": (
        """
        with op.batch_alter_table("telemetry_events") as batch:
            batch.alter_column("sequence", type_=sa.Text())
        """,
        "telemetry_events.sequence: changed from ('INTEGER'",
    ),
    "rewritten-rows": (
        "op.execute(\"UPDATE telemetry_runs SET upload_state = 'held'\")",
        "telemetry_runs: existing rows changed",
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
    changed = edit(sql)
    assert changed != sql
    # SQLite quotes the name in the definition it stores after a rename.
    renamed, count = re.subn(
        rf'^CREATE TABLE "?{table}"? \(', f"CREATE TABLE {table}_new (", changed
    )
    assert count == 1, changed
    connection.execute(renamed)
    connection.execute(f"INSERT INTO {table}_new SELECT {leading_values}* FROM {table}")
    connection.execute(f"DROP TABLE {table}")
    connection.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
    for index_sql in indexes:
        connection.execute(index_sql)


def _sql(statement: str) -> Callable[[sqlite3.Connection], object]:
    return lambda connection: connection.execute(statement)


_ADDITIVE_CHANGES: dict[str, Callable[[sqlite3.Connection], object]] = {
    "new-table": _sql(
        "CREATE TABLE graph_publication_jobs "
        "(publication_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
    ),
    "new-table-referencing-runs": _sql(
        "CREATE TABLE run_receipts (run_id TEXT, producer_id TEXT, "
        "FOREIGN KEY (run_id, producer_id) "
        "REFERENCES telemetry_runs (run_id, producer_id))"
    ),
    "nullable-column": _sql("ALTER TABLE telemetry_runs ADD COLUMN region TEXT"),
    "defaulted-column": _sql(
        "ALTER TABLE telemetry_events ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
    ),
    "plain-index": _sql(
        "CREATE INDEX telemetry_runs_updated ON telemetry_runs (updated_at)"
    ),
}

# Applied in this order, after any additive changes, so every combination
# applies cleanly. Each maps to the problem it alone must be reported as.
_BREAKING_CHANGES: dict[str, tuple[Callable[[sqlite3.Connection], object], str]] = {
    "drop-column": (
        _sql("ALTER TABLE telemetry_runs DROP COLUMN local_only_reason"),
        "telemetry_runs.local_only_reason: removed",
    ),
    "rename-column": (
        _sql("ALTER TABLE telemetry_events RENAME COLUMN created_at TO queued_at"),
        "telemetry_events.created_at: removed",
    ),
    "drop-index": (
        _sql("DROP INDEX telemetry_events_run_sequence"),
        "telemetry_events: index telemetry_events_run_sequence removed",
    ),
    "unique-index": (
        _sql("CREATE UNIQUE INDEX runs_updated ON telemetry_runs (updated_at)"),
        "telemetry_runs: new unique index runs_updated",
    ),
    "trigger": (
        _sql(
            "CREATE TRIGGER runs_touch AFTER INSERT ON telemetry_runs "
            "BEGIN SELECT 1; END"
        ),
        "telemetry_runs: triggers changed",
    ),
    "rewritten-rows": (
        _sql("UPDATE telemetry_runs SET upload_state = 'held'"),
        "telemetry_runs: existing rows changed",
    ),
    "deleted-rows": (
        _sql("DELETE FROM telemetry_events"),
        "telemetry_events: existing rows changed",
    ),
    "required-column": (
        lambda connection: _recreate(
            connection,
            "telemetry_runs",
            lambda sql: sql.replace("(\n\t", "(\n\tnote TEXT NOT NULL, \n\t", 1),
            leading_values="'x', ",
        ),
        "telemetry_runs.note: new column is NOT NULL without a default",
    ),
    "type-change": (
        lambda connection: _recreate(
            connection,
            "telemetry_events",
            lambda sql: sql.replace(
                "sequence INTEGER NOT NULL", "sequence TEXT NOT NULL"
            ),
        ),
        "telemetry_events.sequence: changed from ('INTEGER'",
    ),
    "check-constraint": (
        lambda connection: _recreate(
            connection,
            "telemetry_events",
            lambda sql: sql[: sql.rindex(")")] + ", \n\tCHECK (sequence > 0)\n)",
        ),
        "telemetry_events: CHECK constraints changed",
    ),
    "drop-table": (
        _sql("DROP TABLE telemetry_events"),
        "telemetry_events: removed",
    ),
}


@pytest.fixture(scope="module")
def spool_at_packaged_head(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("template") / "events.sqlite3"
    _close(EventSpool(path))
    return path


def _change(path: Path, additive: list[str], breaking: list[str]) -> None:
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("BEGIN")
        for name in _ADDITIVE_CHANGES:
            if name in additive:
                _ADDITIVE_CHANGES[name](connection)
        for name, (apply, _) in _BREAKING_CHANGES.items():
            if name in breaking:
                apply(connection)
        connection.execute("COMMIT")
    finally:
        connection.close()


@pytest.mark.parametrize("name", list(_BREAKING_CHANGES))
def test_the_additivity_check_names_each_breaking_change(
    tmp_path, spool_at_packaged_head, name
) -> None:
    path = tmp_path / "events.sqlite3"
    shutil.copyfile(spool_at_packaged_head, path)
    problems = _additivity_violations(path, lambda: _change(path, [], [name]))
    assert any(found.startswith(_BREAKING_CHANGES[name][1]) for found in problems), (
        problems
    )


@settings(max_examples=60, deadline=None)
@given(
    additive=st.lists(st.sampled_from(sorted(_ADDITIVE_CHANGES)), unique=True),
    breaking=st.lists(st.sampled_from(sorted(_BREAKING_CHANGES)), unique=True),
)
def test_the_additivity_check_flags_exactly_the_changes_that_break(
    tmp_path_factory, spool_at_packaged_head, additive, breaking
) -> None:
    """And on whatever it accepts, this checkout's spool code still works."""

    directory = tmp_path_factory.mktemp("changes")
    checked = directory / "checked.sqlite3"
    shutil.copyfile(spool_at_packaged_head, checked)
    problems = _additivity_violations(
        checked, lambda: _change(checked, additive, breaking)
    )
    assert bool(problems) == bool(breaking), problems
    if breaking:
        return

    # A newer checkout stamped this spool after making exactly these changes.
    used = directory / "used.sqlite3"
    shutil.copyfile(spool_at_packaged_head, used)
    _change(used, additive, [])
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
