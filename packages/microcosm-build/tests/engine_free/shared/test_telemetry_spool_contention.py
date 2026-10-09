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

import alembic.op
import pytest
from alembic import command
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError, OperationalError

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun
from microcosm.build.telemetry_emitter_constants import TELEMETRY_SERVICE_MODULE
from microcosm.build.telemetry_emitter_service import database as database_module
from microcosm.build.telemetry_emitter_service import main as main_module
from microcosm.build.telemetry_emitter_service import runtime as runtime_module
from microcosm.build.telemetry_emitter_service import spool as spool_module
from microcosm.build.telemetry_emitter_service.constants import (
    DATABASE_TIMEOUT_SECONDS,
    PRUNE_INTERVAL_SECONDS,
    READY_DEADLINE_MARGIN_SECONDS,
    RETENTION_DAYS,
    SPOOL_LOCKED_EXIT_STATUS,
    SPOOL_RETRY_INITIAL_SECONDS,
    SPOOL_RETRY_MAX_SECONDS,
)
from microcosm.build.telemetry_emitter_service.contention import (
    describe_error,
    is_transient_spool_error,
    retry_spool_contention,
)
from microcosm.build.telemetry_emitter_service.database import create_spool_engine
from microcosm.build.telemetry_emitter_service.migrations import (
    alembic_config,
    current_database_revision,
    migration_head_revision,
)
from microcosm.build.telemetry_emitter_service.models import (
    SpoolModel,
    TelemetryEventRecord,
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


def _short_socket_path() -> Path:
    # macOS limits AF_UNIX paths to roughly 100 bytes.
    return Path(tempfile.mkdtemp(prefix="microcosm-test-", dir="/tmp")) / "e.sock"


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
        arguments += ["--ready-deadline", repr(ready_deadline)]
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
    """In process, with a short busy timeout so the lock outlasts many waits."""

    monkeypatch.setattr(database_module, "DATABASE_TIMEOUT_SECONDS", 0.1)
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
        target=lambda: status.append(main_module.main(arguments))
    )

    with _write_lock(spool_path):
        service.start()
        time.sleep(1.0)
        assert not socket_path.exists()
        assert service.is_alive()
        released_at = time.monotonic()
    deadline = time.monotonic() + 20
    while not _ping(socket_path):
        assert time.monotonic() < deadline, "service never became ready"
        assert service.is_alive(), f"service exited with {status}"
        time.sleep(0.02)
    ready_at = time.monotonic()
    _close_service(socket_path)
    service.join(timeout=10)

    assert status == [0]
    failures = [error for _, _, error in outcomes if error is not None]
    # Ten busy timeouts fit in the hold, so a single SQLite wait cannot
    # explain the success: startup retried.
    assert len(failures) >= 2
    assert all(is_transient_spool_error(error) for error in failures)
    name, finished_at, error = outcomes[-1]
    assert (name, error) == ("register", None)
    assert finished_at >= released_at
    assert ready_at >= released_at
    assert "Traceback" not in capsys.readouterr().err


def test_service_exits_75_with_one_line_while_the_spool_stays_locked(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(database_module, "DATABASE_TIMEOUT_SECONDS", 0.1)
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
        status = main_module.main(arguments)
        waited = time.monotonic() - started

    assert status == SPOOL_LOCKED_EXIT_STATUS
    assert waited < 0.5 + 2 * DATABASE_TIMEOUT_SECONDS
    assert not socket_path.exists()
    error_output = capsys.readouterr().err
    assert error_output.count("\n") == 1
    assert error_output.startswith(
        "warning: the local telemetry emitter service could not register"
    )
    assert str(spool_path) in error_output
    assert "(database is locked)" in error_output
    assert "Traceback" not in error_output


def test_unexpected_service_failure_is_one_line_with_status_1(tmp_path, capsys) -> None:
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("")
    arguments = _service_arguments(
        _short_socket_path(),
        occupied / "events.sqlite3",
        ready_deadline=time.time() + 60,
    )

    assert main_module.main(arguments) == 1

    error_output = capsys.readouterr().err
    assert error_output.count("\n") == 1
    assert error_output.startswith(
        "warning: the local telemetry emitter service stopped: "
    )
    assert "Traceback" not in error_output


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_ready_deadline_must_be_finite(value) -> None:
    with pytest.raises(SystemExit) as raised:
        main_module.build_parser().parse_args(
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
    assert raised.value.code == 2


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

    client = threading.Thread(target=start)
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
    assert emitter.available
    assert started["returned_at"] > released_at
    emitter.stage("compile", message="Compiling.")
    emitter.complete()
    assert emitter._process is not None
    assert emitter._process.wait(timeout=30) == 0

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
                # A deadline already past allows one attempt: SQLite's own wait.
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
    ready_deadline = float(command[command.index("--ready-deadline") + 1])
    assert before + 42 <= ready_deadline <= after + 42


@given(
    ready_deadline=st.floats(-1e6, 1e6),
    now_unix=st.floats(-1e6, 1e6),
    now_monotonic=st.floats(0, 1e6),
)
def test_startup_deadline_is_the_ready_deadline_less_the_margin_on_this_clock(
    ready_deadline, now_unix, now_monotonic
) -> None:
    deadline = main_module.startup_deadline(
        ready_deadline, now_unix=now_unix, now_monotonic=now_monotonic
    )
    remaining = ready_deadline - now_unix - READY_DEADLINE_MARGIN_SECONDS
    assert deadline - now_monotonic == pytest.approx(remaining, abs=1e-6)
    assert (
        main_module.startup_deadline(
            None, now_unix=now_unix, now_monotonic=now_monotonic
        )
        == -math.inf
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


def test_an_opener_waits_for_a_migration_in_progress(tmp_path) -> None:
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

    second = threading.Thread(target=open_spool)
    try:
        with migrator.begin() as connection:
            second.start()
            time.sleep(0.5)
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
    failures=st.lists(st.booleans(), max_size=30),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_prune_runs_at_most_once_per_interval_counting_failures(
    monkeypatch, gaps, failures
) -> None:
    clock = SimpleNamespace(now=1_000.0)
    monkeypatch.setattr(
        spool_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    attempts: list[float] = []
    outcomes = iter(failures)

    def prune() -> None:
        attempts.append(clock.now)
        if next(outcomes, False):
            raise _lock_error()

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

    # Reference schedule: the first call prunes; later calls prune once the
    # interval has passed since the previous attempt, whether or not it failed.
    expected: list[float] = []
    for now in calls:
        if not expected or now - expected[-1] >= PRUNE_INTERVAL_SECONDS:
            expected.append(now)
    assert attempts == expected


# --- Lock-error classification ------------------------------------------------


@given(code=st.integers(0, 2**16))
def test_only_busy_and_locked_result_codes_are_transient(code) -> None:
    error = sqlite3.OperationalError("message")
    error.sqlite_errorcode = code
    transient = code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    assert is_transient_spool_error(error) is transient
    assert is_transient_spool_error(_lock_error(code)) is transient


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

_OUTCOME = st.sampled_from(["ok", "locked", "fatal"])


@given(
    outcomes=st.lists(
        st.tuples(_OUTCOME, st.floats(0, 6)),
        min_size=1,
        max_size=40,
    ),
    window=st.one_of(st.just(-math.inf), st.floats(-5, 60)),
    jitter_fraction=st.floats(0, 1),
)
def test_retry_respects_its_deadline_and_backoff(outcomes, window, jitter_fraction):
    clock = SimpleNamespace(now=100.0)
    deadline = clock.now + window
    attempt_starts: list[float] = []
    jitters: list[tuple[float, float, float, float]] = []  # (at, low, high, wait)
    waits: list[tuple[float, float]] = []  # (started, seconds)

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
        clock.now += seconds

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
    # The first attempt always runs; every later one starts before the deadline
    # and no wait reaches it.
    assert all(start < deadline for start in attempt_starts[1:])
    assert all(started + seconds < deadline for started, seconds in waits)
    # Each lock error draws one wait in [delay / 2, delay]; delays double to a
    # cap; every wait drawn is slept except one that would reach the deadline.
    assert len(jitters) == kinds.count("locked")
    assert len(waits) == len(attempt_starts) - 1
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
    # deadline.
    if outcome == "locked":
        at, _, _, wait = jitters[-1]
        assert at + wait >= deadline


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
    heartbeat_seconds=st.sampled_from([1.0, 2.0, 3.5]),
)
def test_worker_survives_any_step_failure(
    monkeypatch, script, ticks, parent_dies_at, heartbeat_seconds
) -> None:
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
    appended: list[tuple[int, str]] = []
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

    def append(registration, event, *, resources=None):
        outcomes.next()
        appended.append((clock.tick, event["event_type"] + ":" + event["status"]))

    def parent_alive():
        parent_checks.append(clock.tick)
        return parent_dies_at is None or clock.tick < parent_dies_at

    service = EmitterService(
        socket_path=Path("/unused"),
        registration=_registration(),
        spool=SimpleNamespace(
            append=append,
            prune_if_due=outcomes.next,
            has_deliverable=lambda: outcomes.next(False),
        ),
        delivery=SimpleNamespace(flush_once=lambda: outcomes.next(False)),
        sampler=SimpleNamespace(
            sample=lambda: outcomes.next({}), parent_alive=parent_alive
        ),
        heartbeat_seconds=heartbeat_seconds,
        drain_seconds=2.0,
    )
    service._stop = Stop()
    heartbeat_attempts: list[tuple[int, float, bool]] = []
    append_heartbeat = service._append_heartbeat

    def recorded_heartbeat() -> None:
        try:
            append_heartbeat()
        except Exception:
            heartbeat_attempts.append((clock.tick, clock.now, False))
            raise
        heartbeat_attempts.append((clock.tick, clock.now, True))

    service._append_heartbeat = recorded_heartbeat
    exit_records: list[int] = []
    append_unexpected_exit = service._append_unexpected_exit

    def recorded_unexpected_exit() -> None:
        exit_records.append(clock.tick)
        append_unexpected_exit()

    service._append_unexpected_exit = recorded_unexpected_exit
    error_output = io.StringIO()

    with contextlib.redirect_stderr(error_output):
        service._worker()  # never raises

    last_tick = ticks if parent_dies_at is None else min(ticks, parent_dies_at)
    # The parent is checked on every tick until it is found dead.
    assert parent_checks == list(range(1, last_tick + 1))
    # A dead parent is recorded once, on the tick it is found, and stops the
    # loop.
    parent_died = parent_dies_at is not None and parent_dies_at <= ticks
    assert exit_records == ([parent_dies_at] if parent_died else [])
    assert service._stop.is_set() is parent_died
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
    # One line per error type other than lock contention, which is silent.
    reported = {
        name for name in outcomes.raised if name in {"RuntimeError", "ValueError"}
    }
    lines = error_output.getvalue().splitlines()
    assert len(lines) == len(reported)
    assert {line.split(" hit ")[1].split(" ")[0] for line in lines} == reported
    assert not any("Traceback" in line for line in lines)


def test_unexpected_exit_waits_out_lock_contention(monkeypatch) -> None:
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

    def append(registration, event, *, resources=None):
        attempts.append(clock.now)
        clock.now += 0.2
        if len(attempts) < 4:
            raise _lock_error()

    service = EmitterService(
        socket_path=Path("/unused"),
        registration=_registration(),
        spool=SimpleNamespace(append=append),
        delivery=SimpleNamespace(),
        sampler=SimpleNamespace(sample=dict),
        heartbeat_seconds=60,
        drain_seconds=15,
    )

    assert service._attempt(service._append_unexpected_exit)
    assert len(attempts) == 4
    assert attempts[-1] < 15
