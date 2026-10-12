"""The telemetry emitter service under lock contention on its shared spool.

Every concurrent build on a host shares one SQLite spool. These tests hold
its write lock from a second connection (``BEGIN IMMEDIATE``), the way another
build's service does while it writes, and check that a starting service waits
the lock out instead of exiting, that concurrent first opens migrate one at a
time, and that nothing it prints on the build's stderr is a traceback.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import alembic.op
import pytest
from alembic import command
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError, OperationalError

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun
from microcosm.build.telemetry_emitter_constants import TELEMETRY_SERVICE_MODULE
from microcosm.build.telemetry_emitter_service import collector as collector_module
from microcosm.build.telemetry_emitter_service import database as database_module
from microcosm.build.telemetry_emitter_service import main as main_module
from microcosm.build.telemetry_emitter_service import runtime as runtime_module
from microcosm.build.telemetry_emitter_service import spool as spool_module
from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DATABASE_TIMEOUT_SECONDS,
    PRUNE_INTERVAL_SECONDS,
    PRUNE_STEP_SECONDS,
    READY_DEADLINE_MARGIN_SECONDS,
    RETENTION_DAYS,
    SPOOL_LOCKED_EXIT_STATUS,
    SPOOL_RETRY_INITIAL_SECONDS,
    SPOOL_RETRY_MAX_SECONDS,
    STARTUP_RETRY_LIMIT_SECONDS,
)
from microcosm.build.telemetry_emitter_service.contention import (
    is_transient_spool_error,
    retry_spool_contention,
)
from microcosm.build.telemetry_emitter_service.database import create_spool_engine
from microcosm.build.telemetry_emitter_service.diagnostics import (
    describe_error,
    write_warning,
)
from microcosm.build.telemetry_emitter_service.migrations import (
    alembic_config,
    current_database_revision,
    migration_head_revision,
)
from microcosm.build.telemetry_emitter_service.models import (
    SpoolModel,
    TelemetryEventRecord,
    TelemetryRunRecord,
)
from microcosm.build.telemetry_emitter_service.runtime import EmitterService
from microcosm.build.telemetry_emitter_service.spool import EventSpool
from microcosm.build.telemetry_protocol import (
    BUILD_COMPLETED_MESSAGE,
    BUILD_STARTED_MESSAGE,
    LOCAL_PING_MESSAGE,
)

_LOOPBACK_COLLECTOR = "http://127.0.0.1:9"
_SPOOL_STATES = ("fresh", "at_head", "at_head_with_expired_rows")


def _registration(run_id: str = "run-a") -> dict[str, object]:
    return TelemetryRun(
        run_id=run_id,
        country_code="US",
        pipeline="test-pipeline",
        producer_id="producer-a",
    ).as_registration()


def _event(stage_id: str = "compile") -> dict[str, object]:
    return {"event_type": "stage", "stage_id": stage_id, "status": "started"}


def _close(spool: EventSpool) -> None:
    spool._engine.dispose()


def _prepare_spool(path: Path, state: str) -> None:
    """Leave ``path`` as a new file, a spool at head, or one with expired rows."""

    if state == "fresh":
        sqlite3.connect(path).close()
        return
    spool = EventSpool(path)
    if state == "at_head_with_expired_rows":
        registration = _registration("expired-run")
        spool.register(registration)
        spool.append(registration, _event())
        expired = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS + 1)).isoformat()
        with spool._session_factory.begin() as session:
            session.query(TelemetryEventRecord).update({"created_at": expired})
    _close(spool)


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


_SOCKET_DIRECTORIES: list[Path] = []


def _short_socket_path() -> Path:
    # macOS limits AF_UNIX paths to roughly 100 bytes, too few for tmp_path.
    directory = Path(tempfile.mkdtemp(prefix="microcosm-test-", dir="/tmp"))
    _SOCKET_DIRECTORIES.append(directory)
    return directory / "e.sock"


@pytest.fixture(autouse=True)
def _remove_socket_directories():
    yield
    while _SOCKET_DIRECTORIES:
        shutil.rmtree(_SOCKET_DIRECTORIES.pop(), ignore_errors=True)


def _run_main(arguments: list[str], *, timeout: float) -> int:
    """Run the service's main(), failing rather than hanging if it never returns."""

    status: list[int] = []
    thread = threading.Thread(
        target=lambda: status.append(main_module.main(arguments)), daemon=True
    )
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"main() still running after {timeout} s"
    return status[0]


def _ping(socket_path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.5)
            client.connect(str(socket_path))
            client.sendall(LOCAL_PING_MESSAGE)
            return client.recv(16) == b"ok\n"
    except OSError:
        return False


def _close_service(socket_path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2)
        client.connect(str(socket_path))
        client.sendall(b'{"action":"close"}\n')
        assert client.recv(16) == b"ok\n"


def _service_arguments(
    socket_path: Path,
    spool_path: Path,
    *,
    ready_deadline: float | None,
) -> list[str]:
    arguments = [
        "--socket",
        str(socket_path),
        "--spool",
        str(spool_path),
        "--registration-json",
        json.dumps(_registration()),
        "--parent-pid",
        str(os.getpid()),
        "--development-collector-url",
        _LOOPBACK_COLLECTOR,
    ]
    if ready_deadline is not None:
        arguments.append(f"--ready-deadline={ready_deadline!r}")
    return arguments


def _lock_error(code: int = sqlite3.SQLITE_BUSY) -> OperationalError:
    driver_error = sqlite3.OperationalError("database is locked")
    driver_error.sqlite_errorcode = code
    return OperationalError("INSERT INTO telemetry_runs ...", {}, driver_error)


# --- Startup waits out a held lock ------------------------------------------


@pytest.mark.parametrize("state", _SPOOL_STATES)
def test_service_registers_once_a_held_spool_lock_is_released(
    tmp_path, monkeypatch, capsys, state
) -> None:
    """In process: the lock outlasts several SQLite waits, so startup retries.

    Each wait is capped at 0.1 s here, and the lock is held until two attempts
    have failed on it.
    """

    monkeypatch.setattr(spool_module, "DATABASE_TIMEOUT_SECONDS", 0.1)
    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, state)
    socket_path = _short_socket_path()
    outcomes: list[tuple[str, float, BaseException | None]] = []

    def recording(name, operation):
        def wrapped(*args, **kwargs):
            try:
                result = operation(*args, **kwargs)
            except BaseException as error:
                outcomes.append((name, time.monotonic(), error))
                raise
            outcomes.append((name, time.monotonic(), None))
            return result

        return wrapped

    monkeypatch.setattr(EventSpool, "__init__", recording("open", EventSpool.__init__))
    monkeypatch.setattr(
        EventSpool, "register", recording("register", EventSpool.register)
    )
    status: list[int] = []
    arguments = _service_arguments(
        socket_path, spool_path, ready_deadline=time.time() + 60
    )
    service = threading.Thread(
        target=lambda: status.append(main_module.main(arguments)), daemon=True
    )

    def failures() -> list[BaseException]:
        return [error for _, _, error in outcomes if error is not None]

    try:
        with _write_lock(spool_path):
            service.start()
            # Hold the lock until startup has failed on it twice, however long
            # the host takes, so one SQLite wait cannot explain the success.
            deadline = time.monotonic() + 60
            while len(failures()) < 2:
                assert time.monotonic() < deadline, "startup never retried"
                assert service.is_alive(), f"service exited with {status}"
                time.sleep(0.02)
            assert not socket_path.exists()
            released_at = time.monotonic()
        deadline = time.monotonic() + 60
        while not _ping(socket_path):
            assert time.monotonic() < deadline, "service never became ready"
            assert service.is_alive(), f"service exited with {status}"
            time.sleep(0.02)
        ready_at = time.monotonic()
    finally:
        # Stop the service even when an assertion above failed.
        if socket_path.exists():
            _close_service(socket_path)
        service.join(timeout=60)

    assert not service.is_alive()
    assert status == [0]
    assert all(is_transient_spool_error(error) for error in failures())
    name, finished_at, error = outcomes[-1]
    assert (name, error) == ("register", None)
    assert finished_at >= released_at
    assert ready_at >= released_at
    assert "Traceback" not in capsys.readouterr().err


def test_service_exits_75_with_one_line_before_the_build_stops_waiting(
    tmp_path, capsys
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, "at_head")
    socket_path = _short_socket_path()
    arguments = _service_arguments(
        socket_path,
        spool_path,
        ready_deadline=time.time() + READY_DEADLINE_MARGIN_SECONDS + 0.5,
    )

    with _write_lock(spool_path):
        started = time.monotonic()
        status = _run_main(arguments, timeout=30)
        waited = time.monotonic() - started

    assert status == SPOOL_LOCKED_EXIT_STATUS
    # The build would give up at margin + 0.5 s; the service has already
    # said why and exited.
    assert waited < READY_DEADLINE_MARGIN_SECONDS + 0.5
    assert not socket_path.exists()
    error_output = capsys.readouterr().err
    assert error_output.count("\n") == 1
    assert error_output.startswith(
        "warning: the local telemetry emitter service could not register"
    )
    assert str(spool_path) in error_output
    assert "(database is locked)" in error_output
    assert "Traceback" not in error_output


def test_startup_waits_for_the_lock_as_long_as_the_build_waits(tmp_path) -> None:
    """Never worse than one long SQLite wait.

    With a 3 s build budget, the client's default before #1166, a lock held for
    1.5 s is waited out, as the old service's single 5 s busy wait did.
    """

    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, "at_head")
    socket_path = _short_socket_path()
    arguments = _service_arguments(
        socket_path, spool_path, ready_deadline=time.time() + 3.0
    )
    status: list[int] = []
    service = threading.Thread(
        target=lambda: status.append(main_module.main(arguments)), daemon=True
    )
    try:
        with _write_lock(spool_path):
            service.start()
            time.sleep(1.5)
        deadline = time.monotonic() + 10
        while not _ping(socket_path):
            assert time.monotonic() < deadline, f"service never ready: {status}"
            assert service.is_alive(), f"service exited with {status}"
            time.sleep(0.02)
    finally:
        if socket_path.exists():
            _close_service(socket_path)
        service.join(timeout=60)
    assert not service.is_alive()
    assert status == [0]


def test_unexpected_service_failure_is_one_line_with_status_1(tmp_path, capsys) -> None:
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("")
    arguments = _service_arguments(
        _short_socket_path(),
        occupied / "events.sqlite3",
        ready_deadline=time.time() + 60,
    )

    assert _run_main(arguments, timeout=30) == 1

    error_output = capsys.readouterr().err
    assert error_output.count("\n") == 1
    assert error_output.startswith(
        "warning: the local telemetry emitter service stopped: "
    )
    assert "Traceback" not in error_output


def _parse_ready_deadline(value: str):
    return main_module.build_parser().parse_args(
        [
            "--socket",
            "s",
            "--spool",
            "q",
            "--registration-json",
            "{}",
            "--parent-pid",
            "1",
            "--ready-deadline",
            value,
        ]
    )


def test_ready_deadline_rejects_nan_and_caps_infinity(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        _parse_ready_deadline("nan")
    assert raised.value.code == 2
    # Invalid arguments are one line on the build's stderr, not a usage block.
    assert capsys.readouterr().err == (
        "warning: the local telemetry emitter service could not start: "
        "argument --ready-deadline: ready deadline must be a number of seconds\n"
    )

    arguments = _parse_ready_deadline("inf")
    deadline = main_module.startup_deadline(
        arguments.ready_deadline, now_unix=time.time(), now_monotonic=100.0
    )
    assert deadline == (
        100.0 + STARTUP_RETRY_LIMIT_SECONDS - READY_DEADLINE_MARGIN_SECONDS
    )


def test_startup_waits_end_by_the_deadline_and_the_spool_then_waits_normally(
    tmp_path,
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, "at_head")
    spool = EventSpool(spool_path, busy_deadline=time.monotonic() + 0.3)
    try:
        with _write_lock(spool_path):
            started = time.monotonic()
            with pytest.raises(OperationalError):
                spool.register(_registration())
            waited = time.monotonic() - started
        assert waited < 1.0
        # A far deadline still waits no longer than a normal statement, and a
        # past one does not wait at all.
        spool.busy_deadline = time.monotonic() + 100
        assert spool.busy_timeout_seconds() == DATABASE_TIMEOUT_SECONDS
        spool.busy_deadline = time.monotonic() - 1
        assert spool.busy_timeout_seconds() == 0
    finally:
        _close(spool)

    registered = main_module.open_registered_spool(
        spool_path, _registration(), deadline=time.monotonic() + 10
    )
    try:
        assert registered.busy_deadline is None
        with registered._engine.connect() as connection:
            milliseconds = connection.exec_driver_sql("PRAGMA busy_timeout").scalar()
        assert milliseconds == DATABASE_TIMEOUT_SECONDS * 1000
    finally:
        _close(registered)


def test_without_a_deadline_startup_makes_one_normal_attempt(tmp_path) -> None:
    """A build that passes no deadline gets the old behaviour: one 5 s wait."""

    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, "at_head")
    opened: list[EventSpool] = []
    original_init = EventSpool.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        opened.append(self)

    with mock.patch.object(EventSpool, "__init__", recording_init):
        spool = main_module.open_registered_spool(
            spool_path, _registration(), deadline=None
        )
    try:
        assert [item.busy_deadline for item in opened] == [None]
        assert spool.busy_timeout_seconds() == DATABASE_TIMEOUT_SECONDS
    finally:
        _close(spool)


def test_build_keeps_telemetry_when_the_spool_is_locked_past_a_busy_wait(
    tmp_path, capfd, real_local_telemetry
) -> None:
    """End to end: the real client starts the real service process."""

    spool_path = tmp_path / "spool" / "events.sqlite3"
    spool_directory = spool_path.parent
    _prepare_spool(spool_path, "at_head")
    # Opening the spool sets its directory to 0o700, which tells this test
    # when the service has reached the spool without a process inspector.
    spool_directory.chmod(0o750)
    started: dict[str, object] = {}

    def start() -> None:
        started["emitter"] = LocalTelemetryEmitter.start(
            run_id="locked-spool-run",
            country_code="US",
            pipeline="test-pipeline",
            development_collector_url=_LOOPBACK_COLLECTOR,
            spool_path=spool_path,
            heartbeat_seconds=60,
            startup_timeout_seconds=90,
        )
        started["returned_at"] = time.monotonic()

    client = threading.Thread(target=start, daemon=True)
    with _write_lock(spool_path):
        client.start()
        deadline = time.monotonic() + 60
        while spool_directory.stat().st_mode & 0o777 != 0o700:
            assert time.monotonic() < deadline, "service never opened the spool"
            assert client.is_alive(), "client returned before the service opened"
            time.sleep(0.02)
        # One full SQLite busy wait, and then some, passes with the lock held.
        time.sleep(DATABASE_TIMEOUT_SECONDS + 1.0)
        assert client.is_alive(), "client stopped waiting while the spool was locked"
        released_at = time.monotonic()
    client.join(timeout=90)

    emitter = started["emitter"]
    assert isinstance(emitter, LocalTelemetryEmitter)
    try:
        assert emitter.available
        assert started["returned_at"] > released_at
        emitter.stage("compile", message="Compiling.")
        emitter.complete()
        assert emitter._process is not None
        assert emitter._process.wait(timeout=30) == 0
    finally:
        # A failed assertion must not leave the service running.
        if emitter._process is not None and emitter._process.poll() is None:
            emitter._process.terminate()
            emitter._process.wait(timeout=10)

    spool = EventSpool(spool_path)
    try:
        events = spool.batch("locked-spool-run", emitter.run.producer_id)
    finally:
        _close(spool)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["message"] == BUILD_STARTED_MESSAGE
    assert events[-1]["message"] == BUILD_COMPLETED_MESSAGE
    assert "Traceback" not in capfd.readouterr().err


def test_service_process_reports_a_locked_spool_in_one_line(tmp_path) -> None:
    """The real process: exit 75 and a single stderr line, no traceback."""

    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, "at_head")
    socket_path = _short_socket_path()
    with _write_lock(spool_path):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                TELEMETRY_SERVICE_MODULE,
                # A deadline already past allows one attempt.
                *_service_arguments(
                    socket_path, spool_path, ready_deadline=time.time()
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )

    assert completed.returncode == SPOOL_LOCKED_EXIT_STATUS
    lines = [line for line in completed.stderr.splitlines() if line.strip()]
    assert len(lines) == 1, completed.stderr
    assert lines[0].startswith(
        "warning: the local telemetry emitter service could not register"
    )
    assert not socket_path.exists()


def test_an_interrupted_service_exits_without_a_traceback(tmp_path) -> None:
    """SIGINT ends the real process as SIGTERM does, with nothing on stderr."""

    spool_path = tmp_path / "spool" / "events.sqlite3"
    _prepare_spool(spool_path, "at_head")
    # Opening the spool sets its directory to 0o700: the service is retrying.
    spool_path.parent.chmod(0o750)
    socket_path = _short_socket_path()
    with _write_lock(spool_path):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                TELEMETRY_SERVICE_MODULE,
                *_service_arguments(
                    socket_path, spool_path, ready_deadline=time.time() + 60
                ),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 120
            while spool_path.parent.stat().st_mode & 0o777 != 0o700:
                assert time.monotonic() < deadline, "service never opened the spool"
                assert process.poll() is None, "service exited before opening"
                time.sleep(0.02)
            process.send_signal(signal.SIGINT)
            _, error_output = process.communicate(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    assert process.returncode == -signal.SIGINT
    assert error_output == ""


def test_client_passes_its_wait_as_a_wall_clock_ready_deadline(
    tmp_path, monkeypatch, real_local_telemetry
) -> None:
    commands: list[list[str]] = []

    class ExitedProcess:
        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    def popen(command, **kwargs):
        commands.append(command)
        return ExitedProcess()

    monkeypatch.setattr("microcosm.build.telemetry_emitter.subprocess.Popen", popen)
    before = time.time()
    LocalTelemetryEmitter.start(
        run_id="deadline",
        country_code="US",
        pipeline="test-pipeline",
        development_collector_url=_LOOPBACK_COLLECTOR,
        spool_path=tmp_path / "events.sqlite3",
        startup_timeout_seconds=42,
    )
    after = time.time()

    (command,) = commands
    (argument,) = [part for part in command if part.startswith("--ready-deadline=")]
    ready_deadline = float(argument.removeprefix("--ready-deadline="))
    assert before + 42 <= ready_deadline <= after + 42
    # One token, so the service parses even a deadline that is already past.
    assert (
        main_module.build_parser()
        .parse_args(
            ["--socket", "s", "--spool", "q", "--registration-json", "{}"]
            + ["--parent-pid", "1", "--ready-deadline=-inf"]
        )
        .ready_deadline
        == -math.inf
    )


@given(
    ready_deadline=st.floats(-1e6, 1e6),
    now_unix=st.floats(-1e6, 1e6),
    now_monotonic=st.floats(0, 1e6),
)
def test_startup_gives_up_before_the_build_and_within_the_limit(
    ready_deadline, now_unix, now_monotonic
) -> None:
    deadline = main_module.startup_deadline(
        ready_deadline, now_unix=now_unix, now_monotonic=now_monotonic
    )
    waits = deadline - now_monotonic
    budget = ready_deadline - now_unix
    # It stops at least the margin before the build gives up, and within the
    # limit however far away the build's deadline is.
    assert waits <= budget - READY_DEADLINE_MARGIN_SECONDS + 1e-6
    assert waits <= STARTUP_RETRY_LIMIT_SECONDS - READY_DEADLINE_MARGIN_SECONDS + 1e-6
    # And it uses all of that time: nothing is left on the table.
    assert waits >= min(budget, STARTUP_RETRY_LIMIT_SECONDS) - (
        READY_DEADLINE_MARGIN_SECONDS + 1e-6
    )
    assert (
        main_module.startup_deadline(
            None, now_unix=now_unix, now_monotonic=now_monotonic
        )
        is None
    )


# --- Migrations run one process at a time, atomically -----------------------


def test_opening_a_spool_at_head_takes_no_write_lock(tmp_path) -> None:
    """Retention runs in the worker, so expired rows don't lock the opener."""

    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, "at_head_with_expired_rows")

    with _write_lock(spool_path):
        started = time.monotonic()
        spool = EventSpool(spool_path)
        opened_in = time.monotonic() - started
    try:
        assert opened_in < DATABASE_TIMEOUT_SECONDS
        assert spool.has_pending()
    finally:
        _close(spool)


def test_a_failed_migration_leaves_no_partial_schema(tmp_path, monkeypatch) -> None:
    spool_path = tmp_path / "events.sqlite3"

    def fail(*args, **kwargs):
        raise RuntimeError("migration interrupted")

    monkeypatch.setattr(alembic.op, "create_index", fail)
    with pytest.raises(RuntimeError, match="migration interrupted"):
        EventSpool(spool_path)

    connection = sqlite3.connect(spool_path)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    finally:
        connection.close()
    assert tables == []


def test_an_opener_waits_for_a_migration_in_progress(tmp_path, monkeypatch) -> None:
    """A second opener waits, then finds the spool at head and changes nothing.

    Before migrations took the write lock first, the waiting opener had
    already read an empty schema and failed with "table already exists".
    """

    spool_path = tmp_path / "events.sqlite3"
    sqlite3.connect(spool_path).close()
    migrator = create_spool_engine(spool_path, immediate_transactions=True)
    opened: list[EventSpool] = []
    failures: list[BaseException] = []

    def open_spool() -> None:
        try:
            opened.append(EventSpool(spool_path))
        except BaseException as error:
            failures.append(error)

    second = threading.Thread(target=open_spool, daemon=True)
    waiting = threading.Event()
    begin_immediate = database_module._begin_immediate

    def signalling_begin_immediate(connection) -> None:
        waiting.set()
        begin_immediate(connection)

    try:
        with migrator.begin() as connection:
            # Only the opener's migration engine is created after this point.
            monkeypatch.setattr(
                database_module, "_begin_immediate", signalling_begin_immediate
            )
            second.start()
            # The opener has asked for the write lock this migration holds.
            assert waiting.wait(timeout=30), "the opener never began its migration"
            time.sleep(0.2)
            assert second.is_alive()
            SpoolModel.metadata.create_all(connection)
            connection.exec_driver_sql(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL, "
                "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
            )
            connection.exec_driver_sql(
                "INSERT INTO alembic_version (version_num) VALUES (?)",
                (migration_head_revision(),),
            )
        second.join(timeout=DATABASE_TIMEOUT_SECONDS + 5)
    finally:
        migrator.dispose()

    assert failures == []
    (spool,) = opened
    _close(spool)
    assert current_database_revision(spool_path) == migration_head_revision()
    check = create_spool_engine(spool_path)
    try:
        with (
            check.begin() as connection,
            alembic_config(connection=connection) as config,
        ):
            command.check(config)
        assert set(inspect(check).get_table_names()) == {
            "alembic_version",
            "telemetry_events",
            "telemetry_runs",
        }
    finally:
        check.dispose()


_CONCURRENT_OPEN = """
import sys, time
from pathlib import Path
from microcosm.build.telemetry_emitter_service.spool import EventSpool
barrier, path, index = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
(barrier / ("ready-" + index)).touch()
while not (barrier / "go").exists():
    time.sleep(0.001)
spool = EventSpool(path)
spool.register({"run_id": "run-" + index, "producer_id": "producer-" + index})
"""


def test_concurrent_first_opens_of_a_new_spool_all_succeed(tmp_path) -> None:
    """Before migrations took the write lock, about half of these failed."""

    spool_path = tmp_path / "events.sqlite3"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    count = 6
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _CONCURRENT_OPEN, str(barrier), str(spool_path)]
            + [str(index)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(count)
    ]
    try:
        # Release every process at once, after all of them have imported.
        deadline = time.monotonic() + 120
        while len(list(barrier.glob("ready-*"))) < count:
            assert time.monotonic() < deadline, "workers never became ready"
            assert all(process.poll() is None for process in processes)
            time.sleep(0.01)
        (barrier / "go").touch()
        results = [process.communicate(timeout=120) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()

    assert [process.returncode for process in processes] == [0] * count, results
    assert current_database_revision(spool_path) == migration_head_revision()
    connection = sqlite3.connect(spool_path)
    try:
        (registered,) = connection.execute(
            "SELECT count(*) FROM telemetry_runs"
        ).fetchone()
    finally:
        connection.close()
    assert registered == count


# --- Retention moved off the readiness and event paths -----------------------


def test_appends_never_prune(tmp_path, monkeypatch) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    pruned: list[float] = []
    monkeypatch.setattr(spool, "prune", lambda: pruned.append(time.monotonic()))
    registration = _registration()
    spool.register(registration)
    try:
        for _ in range(3):
            spool.append(registration, _event())
        assert pruned == []
    finally:
        _close(spool)


@given(
    gaps=st.lists(st.floats(0, 3 * PRUNE_INTERVAL_SECONDS), max_size=30),
    results=st.lists(
        st.sampled_from(["finished", "unfinished", "failed"]), max_size=30
    ),
)
# A call exactly one interval after a finished prune is due.
@example(gaps=[PRUNE_INTERVAL_SECONDS], results=["finished", "finished"])
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_prune_runs_once_per_interval_until_a_backlog_is_drained(
    monkeypatch, gaps, results
) -> None:
    clock = SimpleNamespace(now=1_000.0)
    monkeypatch.setattr(
        spool_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    attempts: list[tuple[float, str]] = []
    outcomes = iter(results)

    def prune(*, time_budget_seconds):
        assert time_budget_seconds == PRUNE_STEP_SECONDS
        result = next(outcomes, "finished")
        attempts.append((clock.now, result))
        if result == "failed":
            raise _lock_error()
        return result == "finished"

    # prune_if_due reads only the schedule and prune, so no database is needed.
    spool = EventSpool.__new__(EventSpool)
    spool._last_prune_at = None
    spool.prune = prune
    calls = [clock.now]
    for gap in gaps:
        calls.append(calls[-1] + gap)
    for now in calls:
        clock.now = now
        with contextlib.suppress(OperationalError):
            spool.prune_if_due()

    # Reference schedule: the first call prunes. A later call prunes at once
    # when the previous prune ran out of its step; otherwise once the interval
    # has passed since the previous attempt, whether it finished or failed.
    expected: list[float] = []
    for now in calls:
        previous = attempts[len(expected) - 1] if expected else None
        if (
            previous is None
            or previous[1] == "unfinished"
            or now - previous[0] >= PRUNE_INTERVAL_SECONDS
        ):
            expected.append(now)
    assert [at for at, _ in attempts] == expected


# --- Lock-error classification ------------------------------------------------


def test_only_busy_and_locked_result_codes_are_transient() -> None:
    """Every primary code with several extensions, raw and wrapped."""

    busy_or_locked = {5, 6}  # SQLITE_BUSY and SQLITE_LOCKED, from the SQLite docs
    for primary in range(256):
        for extension in (0, 1, 2, 3, 255):
            code = primary | extension << 8
            error = sqlite3.OperationalError("message")
            error.sqlite_errorcode = code
            expected = primary in busy_or_locked
            assert is_transient_spool_error(error) is expected, code
            assert is_transient_spool_error(_lock_error(code)) is expected, code


def test_real_sqlite_errors_are_classified_by_what_retrying_can_fix(tmp_path) -> None:
    """Differential check against errors SQLite itself raises."""

    path = tmp_path / "errors.sqlite3"
    engine = create_spool_engine(path)
    impatient = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            connection.exec_driver_sql("INSERT INTO t VALUES (1)")
        with _write_lock(path):
            with pytest.raises(sqlite3.OperationalError) as locked:
                impatient.execute("INSERT INTO t VALUES (2)")
        with pytest.raises(OperationalError) as exists:
            with engine.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE t (id INTEGER)")
        with pytest.raises(IntegrityError) as duplicate:
            with engine.begin() as connection:
                connection.exec_driver_sql("INSERT INTO t VALUES (1)")
    finally:
        impatient.close()
        engine.dispose()

    assert is_transient_spool_error(locked.value)
    assert not is_transient_spool_error(exists.value)
    assert not is_transient_spool_error(duplicate.value)
    assert not is_transient_spool_error(RuntimeError("database is locked"))
    assert describe_error(exists.value) == "table t already exists"


# --- The retry helper ---------------------------------------------------------

# Lock errors are weighted up and attempts are often short, so most examples
# retry several times and many give up within one wait of the deadline.
_OUTCOME = st.sampled_from(["locked", "locked", "locked", "ok", "fatal"])
_ATTEMPT_SECONDS = st.one_of(st.just(0.0), st.floats(0, 0.3), st.floats(0, 6))
_LOCKED_THEN_OK = [("locked", 0.0)] * 6 + [("ok", 0.0)]


@given(
    outcomes=st.lists(
        st.tuples(_OUTCOME, _ATTEMPT_SECONDS),
        min_size=1,
        max_size=40,
    ),
    window=st.one_of(st.just(-math.inf), st.floats(-1, 3), st.floats(-5, 60)),
    jitter_fraction=st.floats(0, 1),
    # How far each sleep overruns, as sleeps do on a loaded host.
    oversleeps=st.lists(st.one_of(st.just(0.0), st.floats(0, 3)), max_size=8),
)
# The first retry's full wait (0.05 s) would cross a deadline 0.04 s away, or
# start exactly on one 0.05 s away.
@example(outcomes=_LOCKED_THEN_OK, window=0.04, jitter_fraction=1.0, oversleeps=[])
@example(outcomes=_LOCKED_THEN_OK, window=0.05, jitter_fraction=1.0, oversleeps=[])
# Six retries fit, so the delays must double up to the cap.
@example(outcomes=_LOCKED_THEN_OK, window=10.0, jitter_fraction=0.5, oversleeps=[])
# The first sleep fits a deadline 1 s away but wakes 3 s late, past it.
@example(outcomes=_LOCKED_THEN_OK, window=1.0, jitter_fraction=0.5, oversleeps=[3.0])
def test_retry_respects_its_deadline_and_backoff(
    outcomes, window, jitter_fraction, oversleeps
):
    clock = SimpleNamespace(now=100.0)
    deadline = clock.now + window
    attempt_starts: list[float] = []
    jitters: list[tuple[float, float, float, float]] = []  # (at, low, high, wait)
    waits: list[tuple[float, float]] = []  # (started, seconds)
    wakes: list[float] = []
    late = iter(oversleeps)

    def operation():
        index = len(attempt_starts)
        attempt_starts.append(clock.now)
        kind, duration = outcomes[min(index, len(outcomes) - 1)]
        clock.now += duration
        if kind == "locked":
            raise _lock_error()
        if kind == "fatal":
            raise ValueError("schema error")
        return index

    def jitter(low, high):
        wait = low + jitter_fraction * (high - low)
        jitters.append((clock.now, low, high, wait))
        return wait

    def sleep(seconds):
        waits.append((clock.now, seconds))
        clock.now += seconds + next(late, 0.0)
        wakes.append(clock.now)

    try:
        result = retry_spool_contention(
            operation,
            deadline=deadline,
            clock=lambda: clock.now,
            sleep=sleep,
            jitter=jitter,
        )
    except ValueError:
        outcome = "fatal"
    except OperationalError:
        outcome = "locked"
    else:
        outcome = "ok"

    kinds = [outcomes[min(i, len(outcomes) - 1)][0] for i in range(len(attempt_starts))]
    # Every attempt but the last was a lock error, and the result is the last's.
    assert all(kind == "locked" for kind in kinds[:-1])
    assert kinds[-1] == outcome
    if outcome == "ok":
        assert result == len(attempt_starts) - 1
    # The first attempt always runs; every later one starts before the deadline,
    # even after a sleep that overran, and no wait is asked to reach it.
    assert all(start < deadline for start in attempt_starts[1:])
    assert all(started + seconds < deadline for started, seconds in waits)
    overran = bool(wakes) and wakes[-1] >= deadline
    # Each lock error draws one wait in [delay / 2, delay]; delays double to a
    # cap; every wait drawn is slept except one that would reach the deadline,
    # and every sleep is followed by an attempt unless it woke too late.
    assert len(jitters) == kinds.count("locked")
    assert len(waits) == len(attempt_starts) - 1 + overran
    for index, (_, low, high, wait) in enumerate(jitters):
        assert high == min(
            SPOOL_RETRY_MAX_SECONDS, SPOOL_RETRY_INITIAL_SECONDS * 2**index
        )
        assert low == high / 2
        assert low <= wait <= high
    assert [seconds for _, seconds in waits] == [
        wait for _, _, _, wait in jitters[: len(waits)]
    ]
    # A lock error escapes only when its retry could not start before the
    # deadline: the wait would reach it, or the sleep woke past it.
    if outcome == "locked":
        at, _, _, wait = jitters[-1]
        assert at + wait >= deadline or overran
    else:
        assert not overran


def test_retry_with_no_deadline_makes_one_attempt() -> None:
    attempts: list[int] = []

    def operation():
        attempts.append(1)
        raise _lock_error()

    with pytest.raises(OperationalError):
        retry_spool_contention(operation, deadline=-math.inf, sleep=pytest.fail)
    assert attempts == [1]


# --- The delivery worker keeps running ----------------------------------------


class _Outcomes:
    """Hands each worker step the next scripted outcome, recording failures."""

    def __init__(self, script: list[str]) -> None:
        self._script = iter(script)
        self.raised: list[str] = []

    def next(self, value=None):
        kind = next(self._script, "ok")
        if kind == "locked":
            self.raised.append("OperationalError")
            raise _lock_error()
        if kind == "runtime":
            self.raised.append("RuntimeError")
            raise RuntimeError("collector bug")
        if kind == "value":
            self.raised.append("ValueError")
            raise ValueError("bad response")
        return value


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    script=st.lists(
        st.sampled_from(["ok", "ok", "locked", "runtime", "value"]), max_size=200
    ),
    ticks=st.integers(1, 30),
    parent_dies_at=st.one_of(st.none(), st.integers(1, 30)),
    parent_check_raises=st.booleans(),
    closed_at=st.one_of(st.none(), st.integers(1, 30)),
    heartbeat_seconds=st.sampled_from([1.0, 2.0, 3.5]),
)
def test_worker_survives_any_step_failure(
    monkeypatch,
    script,
    ticks,
    parent_dies_at,
    parent_check_raises,
    closed_at,
    heartbeat_seconds,
) -> None:
    """The worker never dies, checks its build every tick, and tells a crash
    from a clean close.

    ``closed_at`` is the tick on which the build's close arrives while a step
    runs, as it does when the accept thread handles it mid-tick; the build may
    then exit before the tick ends. ``parent_check_raises`` makes the check of
    a dead build raise instead of returning False.
    """

    clock = SimpleNamespace(now=0.0, tick=0)
    monkeypatch.setattr(
        runtime_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    outcomes = _Outcomes(script)
    appended: list[str] = []
    parent_checks: list[int] = []

    class Stop:
        def __init__(self) -> None:
            self.flag = False

        def wait(self, timeout):
            clock.now += timeout
            clock.tick += 1
            return self.flag or clock.tick > ticks

        def set(self):
            self.flag = True

        def is_set(self):
            return self.flag

    def append_many(registration, pairs, *, before_write=None, **waits):
        # The worker only queues events; the drain's writer stores them.
        before_write()
        appended.extend(
            event["event_type"] + ":" + event["status"] for event, _ in pairs
        )

    def parent_alive():
        parent_checks.append(clock.tick)
        alive = parent_dies_at is None or clock.tick < parent_dies_at
        if not alive and parent_check_raises:
            raise OverflowError("signed integer is greater than maximum")
        return alive

    def flush_once():
        if clock.tick == closed_at:
            service._stop.set()
        return outcomes.next(False)

    service = EmitterService(
        socket_path=Path("/unused"),
        registration=_registration(),
        spool=SimpleNamespace(
            append_many=append_many,
            prune_if_due=outcomes.next,
            has_deliverable=lambda: outcomes.next(False),
        ),
        delivery=SimpleNamespace(flush_once=flush_once),
        sampler=SimpleNamespace(
            sample=lambda: outcomes.next({}), parent_alive=parent_alive
        ),
        heartbeat_seconds=heartbeat_seconds,
        drain_seconds=2.0,
    )
    service._stop = Stop()
    heartbeat_attempts: list[tuple[int, float, bool]] = []
    queue_heartbeat = service._queue_heartbeat

    def recorded_heartbeat() -> None:
        try:
            queue_heartbeat()
        except Exception:
            heartbeat_attempts.append((clock.tick, clock.now, False))
            raise
        heartbeat_attempts.append((clock.tick, clock.now, True))

    service._queue_heartbeat = recorded_heartbeat
    exit_records: list[tuple[int, bool]] = []
    queue_unexpected_exit = service._queue_unexpected_exit

    def recorded_unexpected_exit() -> None:
        try:
            queue_unexpected_exit()
        except Exception:
            exit_records.append((clock.tick, False))
            raise
        exit_records.append((clock.tick, True))

    service._queue_unexpected_exit = recorded_unexpected_exit
    error_output = io.StringIO()

    with contextlib.redirect_stderr(error_output):
        service._worker()  # never raises

    closed = closed_at is not None and closed_at <= ticks
    last_tick = min(
        tick for tick in (ticks, parent_dies_at, closed_at) if tick is not None
    )
    # The parent is checked on every tick until it is found dead, except on the
    # tick the build closed.
    assert parent_checks == [
        tick for tick in range(1, last_tick + 1) if not (closed and tick == closed_at)
    ]
    # A build that died without closing is recorded once, on the tick it is
    # found, which stops the loop; a build that closed first never is.
    crashed = (
        parent_dies_at is not None
        and parent_dies_at <= ticks
        and (not closed or parent_dies_at < closed_at)
    )
    assert [tick for tick, _ in exit_records] == ([parent_dies_at] if crashed else [])
    assert service._stop.is_set() is (crashed or closed)
    # Reference heartbeat schedule: due one interval after the start or the
    # last success; a failed heartbeat stays due and is retried next tick.
    due = heartbeat_seconds
    expected_ticks: list[int] = []
    attempts = iter(heartbeat_attempts)
    for tick in range(1, last_tick + 1):
        now = float(tick)
        if now >= due:
            expected_ticks.append(tick)
            attempt_tick, _, succeeded = next(attempts)
            assert attempt_tick == tick
            if succeeded:
                due = now + heartbeat_seconds
    assert [tick for tick, _, _ in heartbeat_attempts] == expected_ticks
    # The drain stores what the worker queued, in order: every heartbeat that
    # was queued, then the record of a dead build.
    queued_heartbeats = sum(succeeded for _, _, succeeded in heartbeat_attempts)
    recorded_exit = any(succeeded for _, succeeded in exit_records)
    assert appended == ["heartbeat:progress"] * queued_heartbeats + (
        ["run:failed"] if recorded_exit else []
    )
    # One line per error type other than lock contention, which is silent.
    reported = {
        name for name in outcomes.raised if name in {"RuntimeError", "ValueError"}
    }
    if crashed and parent_check_raises:
        reported.add("OverflowError")
    lines = error_output.getvalue().splitlines()
    assert len(lines) == len(reported)
    assert {line.split(" hit ")[1].split(" ")[0] for line in lines} == reported
    assert not any("Traceback" in line for line in lines)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    script=st.lists(st.sampled_from(["ok", "locked", "runtime", "value"]), max_size=60),
    deliverable=st.lists(st.booleans(), max_size=60),
    drain_seconds=st.sampled_from([0.0, 0.4, 2.0, 5.0]),
)
def test_drain_survives_any_failure_and_keeps_its_deadline(
    monkeypatch, script, deliverable, drain_seconds
) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        runtime_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    outcomes = _Outcomes(script)
    answers = iter(deliverable)
    checks: list[float] = []

    def has_deliverable() -> bool:
        checks.append(clock.now)
        outcomes.next()
        return next(answers, False)

    def flush_once() -> bool:
        clock.now += 0.01
        outcomes.next()
        return False

    service = EmitterService(
        socket_path=Path("/unused"),
        registration=_registration(),
        spool=SimpleNamespace(has_deliverable=has_deliverable),
        delivery=SimpleNamespace(flush_once=flush_once),
        sampler=SimpleNamespace(),
        heartbeat_seconds=60,
        drain_seconds=drain_seconds,
    )
    error_output = io.StringIO()

    with contextlib.redirect_stderr(error_output):
        service._drain()  # never raises

    # It never runs past its deadline, and checks again after every failure
    # until nothing is deliverable or the deadline passes.
    assert clock.now <= drain_seconds + 0.01 + 1e-9
    assert all(check < drain_seconds for check in checks)
    reported = {
        name for name in outcomes.raised if name in {"RuntimeError", "ValueError"}
    }
    assert len(error_output.getvalue().splitlines()) == len(reported)


def test_unexpected_exit_waits_out_lock_contention(monkeypatch) -> None:
    """The drain's writer stores a killed build's record once the lock clears."""

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        runtime_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    attempts: list[float] = []
    stored: list[str] = []

    def append_many(registration, pairs, *, before_write=None, **waits):
        before_write()
        attempts.append(clock.now)
        clock.now += 0.2
        if len(attempts) < 4:
            raise _lock_error()
        stored.extend(event["stage_id"] for event, _ in pairs)

    service = EmitterService(
        socket_path=Path("/unused"),
        registration=_registration(),
        spool=SimpleNamespace(append_many=append_many, has_deliverable=lambda: False),
        delivery=SimpleNamespace(),
        sampler=SimpleNamespace(sample=dict),
        heartbeat_seconds=60,
        drain_seconds=15,
    )
    service._handle({"action": "event", "event": _event("compile")})

    assert service._attempt(service._queue_unexpected_exit)
    service._drain()

    assert len(attempts) == 4
    assert attempts[-1] < 15
    # Behind every event the build sent, and last.
    assert stored == ["compile", "failed"]
    assert service.event_queue.closed


def test_service_stops_serving_when_its_worker_dies(tmp_path) -> None:
    """Only the worker notices a dead build, so serving must not outlive it."""

    spool = EventSpool(tmp_path / "events.sqlite3")
    spool.register(_registration())
    socket_path = _short_socket_path()
    service = EmitterService(
        socket_path=socket_path,
        registration=_registration(),
        spool=spool,
        delivery=SimpleNamespace(flush_once=lambda: False),
        sampler=SimpleNamespace(sample=dict, parent_alive=lambda: True),
        heartbeat_seconds=60,
        drain_seconds=0,
    )
    service._worker = lambda: None  # returns at once without stopping the service
    serving = threading.Thread(target=service.run, daemon=True)
    try:
        serving.start()
        serving.join(timeout=10)
        assert not serving.is_alive()
        assert not socket_path.exists()
        # The service drained in the worker's place and stopped its writer.
        assert not service.writer.is_alive()
    finally:
        # A failing run must not leave a server thread behind.
        service._stop.set()
        serving.join(timeout=5)
        _close(spool)


class _GoneStream:
    """A stderr whose reader has exited, like a build piped through ``tee``."""

    def write(self, text):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def test_warnings_never_raise_when_the_builds_stderr_is_gone(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stderr", _GoneStream())
    write_warning("the service keeps running")
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stderr", closed)
    write_warning("the service keeps running")

    monkeypatch.setattr(sys, "stderr", _GoneStream())
    service = EmitterService(
        socket_path=Path("/unused"),
        registration=_registration(),
        spool=SimpleNamespace(),
        delivery=SimpleNamespace(),
        sampler=SimpleNamespace(),
        heartbeat_seconds=60,
    )

    def bug() -> None:
        raise RuntimeError("collector bug")

    assert service._attempt(bug) is False


def test_delivery_backs_off_after_an_unexpected_error(monkeypatch) -> None:
    """The worker survives the error, so it must not turn into a request a tick."""

    requests: list[str] = []

    def post(url, payload, bearer_token, **kwargs):
        requests.append(url)
        return 200, {"access_token": "collector-token", "expires_in": None}

    monkeypatch.setattr(collector_module, "_http_post", post)
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    delivery = CollectorDelivery(
        SimpleNamespace(pending_runs=lambda: [_registration()]),
        development_collector_url=_LOOPBACK_COLLECTOR,
    )

    with pytest.raises(TypeError):
        delivery.flush_once()
    assert delivery.flush_once() is False
    assert len(requests) == 1


def test_delivery_retries_a_locked_spool_on_the_next_tick(monkeypatch) -> None:
    attempts: list[int] = []

    def pending_runs():
        attempts.append(1)
        raise _lock_error()

    delivery = CollectorDelivery(
        SimpleNamespace(pending_runs=pending_runs),
        development_collector_url=_LOOPBACK_COLLECTOR,
    )
    for _ in range(3):
        with pytest.raises(OperationalError):
            delivery.flush_once()
    assert len(attempts) == 3


def test_a_spool_from_an_unknown_migration_is_refused_without_the_write_lock(
    tmp_path,
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, "at_head")
    connection = sqlite3.connect(spool_path)
    connection.execute("UPDATE alembic_version SET version_num = 'from_elsewhere'")
    connection.commit()
    connection.close()

    with _write_lock(spool_path):
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="'from_elsewhere' is not in"):
            EventSpool(spool_path)
        assert time.monotonic() - started < DATABASE_TIMEOUT_SECONDS


# --- Retention in short batches, matching the single-pass rule ----------------

_STAMP_ORIGIN = datetime(2026, 10, 9, 12, tzinfo=UTC)


def _stamp(age_days: float, index: int) -> str:
    # The index keeps stamps distinct, so the oldest-first order is unambiguous.
    return (
        _STAMP_ORIGIN - timedelta(days=age_days) + timedelta(microseconds=index)
    ).isoformat()


@settings(max_examples=40, deadline=None)
@given(
    events=st.lists(
        st.tuples(
            st.integers(0, 2),  # run
            st.floats(0, 2 * RETENTION_DAYS),  # age in days
            st.integers(0, 60),  # padding characters
        ),
        max_size=18,
    ),
    run_ages=st.lists(st.floats(0, 2 * RETENTION_DAYS), min_size=3, max_size=3),
    # None leaves the real 100 MiB cap, so only age removes rows.
    cap_fraction=st.one_of(st.none(), st.floats(0, 1.2)),
    batch_rows=st.integers(1, 4),
    step_budget=st.sampled_from([0.0, math.inf]),
)
# More expired rows than one batch, with no size pressure to hide a leftover.
@example(
    events=[(0, 10.0, 5)] * 6,
    run_ages=[1.0, 1.0, 1.0],
    cap_fraction=None,
    batch_rows=2,
    step_budget=math.inf,
)
# Removing the oldest live event (20 characters of 50) fits the cap exactly.
@example(
    events=[(0, 1.0, 10), (0, 1.0, 20)],
    run_ages=[1.0, 1.0, 1.0],
    cap_fraction=0.6,
    batch_rows=2,
    step_budget=math.inf,
)
def test_batched_prune_matches_the_single_pass_retention_rule(
    tmp_path_factory, events, run_ages, cap_fraction, batch_rows, step_budget
) -> None:
    """Differential: the batched prune keeps exactly what one pass would.

    The reference is the retention rule the single-transaction prune applied:
    drop events past ``RETENTION_DAYS``; then, oldest first, drop events until
    the stored payloads fit ``MAX_QUEUED_BYTES``; then drop expired runs that
    have no events left.
    """

    spool = EventSpool(tmp_path_factory.mktemp("prune") / "events.sqlite3")
    keys = [(f"run-{index}", f"producer-{index}") for index in range(3)]
    rows: list[dict[str, object]] = []
    try:
        with spool._session_factory.begin() as session:
            for index, (run_id, producer_id) in enumerate(keys):
                session.add(
                    TelemetryRunRecord(
                        run_id=run_id,
                        producer_id=producer_id,
                        registration={"run_id": run_id, "producer_id": producer_id},
                        next_sequence=1,
                        upload_state="pending",
                        local_only_reason=None,
                        updated_at=_stamp(run_ages[index], index),
                    )
                )
            for index, (run, age, padding) in enumerate(events):
                run_id, producer_id = keys[run]
                payload = {"pad": "x" * padding}
                row = {
                    "event_id": f"event-{index:02d}",
                    "run": keys[run],
                    "created_at": _stamp(age, index),
                    "sequence": index,
                    "length": len(
                        json.dumps(payload, separators=(",", ":"), sort_keys=True)
                    ),
                }
                rows.append(row)
                session.add(
                    TelemetryEventRecord(
                        event_id=row["event_id"],
                        run_id=run_id,
                        producer_id=producer_id,
                        sequence=index,
                        payload=payload,
                        created_at=row["created_at"],
                    )
                )
        cutoff = (_STAMP_ORIGIN - timedelta(days=RETENTION_DAYS)).isoformat()
        live = [row for row in rows if row["created_at"] >= cutoff]
        cap = (
            spool_module.MAX_QUEUED_BYTES
            if cap_fraction is None
            else int(cap_fraction * sum(row["length"] for row in live))
        )
        batches: list[int] = []
        delete = spool._delete

        def recording_delete(model, criterion) -> int:
            deleted = delete(model, criterion)
            batches.append(deleted)
            return deleted

        spool._delete = recording_delete
        frozen_now = mock.Mock(wraps=datetime)
        frozen_now.now.return_value = _STAMP_ORIGIN
        with (
            mock.patch.object(spool_module, "MAX_QUEUED_BYTES", cap),
            mock.patch.object(spool_module, "PRUNE_BATCH_ROWS", batch_rows),
            mock.patch.object(spool_module, "PRUNE_BATCH_PAUSE_SECONDS", 0.0),
            mock.patch.object(spool_module, "datetime", frozen_now),
        ):
            # A zero budget stops after every batch, so the prune is resumed
            # call by call until it reports that it finished.
            calls = 1
            while not spool.prune(time_budget_seconds=step_budget):
                calls += 1
                assert calls <= len(events) + 3
        with spool._session_factory() as session:
            kept_events = set(
                session.scalars(select(TelemetryEventRecord.event_id)).all()
            )
            kept_runs = set(
                session.execute(
                    select(
                        TelemetryRunRecord.run_id,
                        TelemetryRunRecord.producer_id,
                    )
                ).all()
            )
    finally:
        _close(spool)

    expected_events = {row["event_id"] for row in live}
    excess = sum(row["length"] for row in live) - cap
    for row in sorted(live, key=lambda row: (row["created_at"], row["sequence"])):
        if excess <= 0:
            break
        expected_events.discard(row["event_id"])
        excess -= row["length"]
    expected_runs = {
        key
        for index, key in enumerate(keys)
        if _stamp(run_ages[index], index) >= cutoff
        or any(row["run"] == key and row["event_id"] in expected_events for row in rows)
    }
    assert kept_events == expected_events
    assert kept_runs == expected_runs
    # Each transaction deleted at most one batch.
    assert all(deleted <= batch_rows for deleted in batches)


def test_a_prune_with_nothing_to_remove_only_reads(tmp_path) -> None:
    """An idle prune runs every minute in every service; it takes no lock."""

    spool_path = tmp_path / "events.sqlite3"
    spool = EventSpool(spool_path)
    try:
        registration = _registration()
        spool.register(registration)
        spool.append(registration, _event())
        with _write_lock(spool_path):
            started = time.monotonic()
            assert spool.prune() is True
            assert time.monotonic() - started < 1.0
        assert spool.has_pending()
    finally:
        _close(spool)


def test_prune_pauses_between_batches_outside_the_process_lock(
    tmp_path, monkeypatch
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    pauses: list[bool] = []
    try:
        registration = _registration()
        spool.register(registration)
        for _ in range(5):
            spool.append(registration, _event())
        expired = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS + 1)).isoformat()
        with spool._session_factory.begin() as session:
            session.query(TelemetryEventRecord).update({"created_at": expired})

        def pause(seconds: float) -> None:
            # This build's event path takes the same lock, so it must be free.
            pauses.append(spool._lock._is_owned())

        monkeypatch.setattr(spool_module, "PRUNE_BATCH_ROWS", 2)
        monkeypatch.setattr(
            spool_module,
            "time",
            SimpleNamespace(monotonic=time.monotonic, sleep=pause),
        )
        assert spool.prune() is True
        assert not spool.has_pending()
    finally:
        _close(spool)

    # Five rows in batches of two: a pause after each full batch, never while
    # this process holds its lock.
    assert pauses == [False, False]


def test_a_prune_out_of_time_stops_between_batches_and_resumes(
    tmp_path, monkeypatch
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    try:
        registration = _registration()
        spool.register(registration)
        for _ in range(5):
            spool.append(registration, _event())
        expired = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS + 1)).isoformat()
        with spool._session_factory.begin() as session:
            session.query(TelemetryEventRecord).update({"created_at": expired})
        monkeypatch.setattr(spool_module, "PRUNE_BATCH_ROWS", 2)
        monkeypatch.setattr(spool_module, "PRUNE_BATCH_PAUSE_SECONDS", 0.0)

        remaining = []
        while not spool.prune(time_budget_seconds=0.0):
            remaining.append(len(spool.batch("run-a", "producer-a", limit=10)))
        remaining.append(len(spool.batch("run-a", "producer-a", limit=10)))
    finally:
        _close(spool)

    # One batch per call: five rows go two, two, then one.
    assert remaining == [3, 1, 0]


def test_a_service_that_closes_within_its_first_tick_still_prunes(tmp_path) -> None:
    """Short builds must not leave retention unenforced on a host forever."""

    spool_path = tmp_path / "events.sqlite3"
    _prepare_spool(spool_path, "at_head_with_expired_rows")
    spool = EventSpool(spool_path)
    registration = _registration()
    spool.register(registration)
    socket_path = _short_socket_path()
    service = EmitterService(
        socket_path=socket_path,
        registration=registration,
        spool=spool,
        delivery=SimpleNamespace(flush_once=lambda: False),
        sampler=SimpleNamespace(sample=dict, parent_alive=lambda: True),
        heartbeat_seconds=60,
        drain_seconds=0,
    )
    serving = threading.Thread(target=service.run, daemon=True)
    try:
        assert spool.has_pending()
        serving.start()
        deadline = time.monotonic() + 30
        while not _ping(socket_path):
            assert time.monotonic() < deadline, "service never became ready"
            time.sleep(0.01)
        # Closed at once, well inside the worker's first one-second tick.
        _close_service(socket_path)
        serving.join(timeout=30)
        assert not serving.is_alive()
        assert not spool.has_pending()
    finally:
        service._stop.set()
        serving.join(timeout=5)
        _close(spool)


def test_expired_runs_are_deleted_in_batches(tmp_path, monkeypatch) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    deleted: list[tuple[str, int]] = []
    pauses: list[float] = []
    try:
        for index in range(5):
            spool.register({"run_id": f"run-{index}", "producer_id": "producer-a"})
        expired = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS + 1)).isoformat()
        with spool._session_factory.begin() as session:
            session.query(TelemetryRunRecord).update({"updated_at": expired})
        delete = spool._delete

        def recording_delete(model, criterion) -> int:
            count = delete(model, criterion)
            deleted.append((model.__tablename__, count))
            return count

        spool._delete = recording_delete
        monkeypatch.setattr(spool_module, "PRUNE_BATCH_ROWS", 2)
        monkeypatch.setattr(
            spool_module,
            "time",
            SimpleNamespace(monotonic=time.monotonic, sleep=pauses.append),
        )
        assert spool.prune() is True
        with spool._session_factory() as session:
            remaining = session.scalars(select(TelemetryRunRecord.run_id)).all()
    finally:
        _close(spool)

    # Five runs in batches of two, with a pause after each full batch.
    assert deleted == [
        ("telemetry_runs", 2),
        ("telemetry_runs", 2),
        ("telemetry_runs", 1),
    ]
    assert len(pauses) == 2
    assert remaining == []


def test_a_spool_under_the_size_cap_is_not_scanned(tmp_path, monkeypatch) -> None:
    """The size check reads the file's page counts, not every payload."""

    spool = EventSpool(tmp_path / "events.sqlite3")
    statements: list[str] = []
    try:
        registration = _registration()
        spool.register(registration)
        for _ in range(3):
            spool.append(registration, _event())
        sqlalchemy_event.listen(
            spool._engine,
            "before_cursor_execute",
            lambda conn, cursor, statement, *rest: statements.append(statement),
        )
        assert spool.prune() is True
        assert not [text for text in statements if "sum(" in text.lower()]
        assert len(spool.batch("run-a", "producer-a")) == 3

        # Over the cap, the payloads are summed and the oldest events go.
        statements.clear()
        monkeypatch.setattr(spool_module, "MAX_QUEUED_BYTES", 1)
        assert spool.prune() is True
        assert [text for text in statements if "sum(" in text.lower()]
        assert spool.batch("run-a", "producer-a") == []
    finally:
        _close(spool)
