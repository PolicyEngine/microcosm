"""The emitter service's event path, decoupled from writes to its spool.

The service acknowledges an event once it is queued in memory; one writer
thread appends queued events to the shared spool, oldest first, retrying lock
contention. These tests state the queue's and the writer's invariants as
properties, then hold the spool's write lock from a second connection
(``BEGIN IMMEDIATE``), the way another build's service does while it writes,
while a build sends events.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import socket
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from sqlalchemy.exc import OperationalError

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun
from microcosm.build.telemetry_emitter_constants import DEFAULT_SEND_TIMEOUT_SECONDS
from microcosm.build.telemetry_emitter_service import runtime as runtime_module
from microcosm.build.telemetry_emitter_service.constants import (
    DATABASE_TIMEOUT_SECONDS,
    QUEUE_MAX_BYTES,
    QUEUE_MAX_EVENTS,
    QUEUE_RESERVED_BYTES,
    QUEUE_RESERVED_EVENTS,
    SPOOL_RETRY_MAX_SECONDS,
    WRITER_BATCH_EVENTS,
    WRITER_BUSY_TIMEOUT_SECONDS,
)
from microcosm.build.telemetry_emitter_service.contention import (
    SpoolBusyError,
    is_transient_spool_error,
    retry_spool_contention,
)
from microcosm.build.telemetry_emitter_service.event_queue import (
    EventQueue,
    QueueClosedError,
    QueuedEvent,
    QueueFullError,
    SpoolWriter,
)
from microcosm.build.telemetry_emitter_service.runtime import EmitterService
from microcosm.build.telemetry_emitter_service.spool import EventSpool
from microcosm.build.telemetry_protocol import (
    BUILD_COMPLETED_MESSAGE,
    BUILD_STARTED_MESSAGE,
    LOCAL_ACKNOWLEDGEMENT_ERROR,
    LOCAL_ACKNOWLEDGEMENT_OK,
    MAX_TELEMETRY_DETAILS_BYTES,
    MAX_TELEMETRY_MESSAGE_CHARS,
    MAX_TELEMETRY_TEXT_CHARS,
    UNEXPECTED_PROCESS_EXIT_MESSAGE,
)
from microcosm.build.telemetry_sanitization import sanitize_details, sanitize_text

_LOOPBACK_COLLECTOR = "http://127.0.0.1:9"


def _registration(run_id: str = "run-a") -> dict[str, object]:
    return TelemetryRun(
        run_id=run_id,
        country_code="US",
        pipeline="test-pipeline",
        producer_id="producer-a",
    ).as_registration()


def _event(stage_id: str, **details: object) -> dict[str, object]:
    return {
        "event_type": "stage",
        "stage_id": stage_id,
        "status": "started",
        "details": details,
    }


def _lock_error() -> OperationalError:
    driver_error = sqlite3.OperationalError("database is locked")
    driver_error.sqlite_errorcode = sqlite3.SQLITE_BUSY
    return OperationalError("INSERT INTO telemetry_events ...", {}, driver_error)


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


def _stored_stages(path: Path) -> list[str]:
    """Stage ids on disk in sequence order, read without the service's spool.

    A reader is not blocked by another connection's ``BEGIN IMMEDIATE``.
    """

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT payload_json FROM telemetry_events ORDER BY sequence"
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(payload)["stage_id"] for (payload,) in rows]


def _stored_events(path: Path, run_id: str, producer_id: str) -> list[dict]:
    spool = EventSpool(path)
    try:
        return spool.batch(run_id, producer_id, limit=100_000)
    finally:
        spool._engine.dispose()


class _Clock:
    """A fake monotonic clock; sleeping advances it, by ``oversleep`` too long."""

    def __init__(self, now: float = 1_000.0, oversleep: float = 0.0) -> None:
        self.now = now
        self.oversleep = oversleep
        self.sleeps: list[tuple[float, float]] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append((self.now, seconds))
        self.now += seconds + self.oversleep


def _service(spool, **overrides) -> EmitterService:
    arguments = {
        "socket_path": Path("/unused"),
        "registration": _registration(),
        "spool": spool,
        "delivery": SimpleNamespace(flush_once=lambda: False),
        "sampler": SimpleNamespace(sample=dict, parent_alive=lambda: True),
        "heartbeat_seconds": 60,
    }
    return EmitterService(**(arguments | overrides))


def _send(service: EmitterService, event: dict[str, object]) -> bool:
    """Hand one event to the service as the socket thread would; True for ``ok``."""

    try:
        service._handle({"action": "event", "event": event})
    except Exception:
        return False
    return True


# --- The queue's bound ----------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(
    max_events=st.integers(1, 12),
    reserved_events=st.integers(0, 6),
    max_bytes=st.integers(1, 400),
    reserved_bytes=st.integers(0, 200),
    operations=st.lists(
        st.one_of(
            st.tuples(
                st.just("offer"),
                st.integers(1, 80),  # encoded size
                st.booleans(),  # run event
                st.sampled_from([False, False, False, True]),  # close behind it
            ),
            st.tuples(st.just("remove"), st.integers(0, 4)),
        ),
        max_size=80,
    ),
)
def test_queue_takes_exactly_what_fits_keeps_order_and_never_exceeds_its_bound(
    max_events, reserved_events, max_bytes, reserved_bytes, operations
) -> None:
    queue = EventQueue(
        max_events=max_events,
        max_bytes=max_bytes,
        reserved_events=reserved_events,
        reserved_bytes=reserved_bytes,
    )
    model: list[QueuedEvent] = []
    closed = False
    for operation in operations:
        if operation[0] == "remove":
            count = min(operation[1], len(model))
            queue.remove(count)
            del model[:count]
        else:
            _, size, run_event, close = operation
            event = QueuedEvent(b"x" * size, run_event)
            # Reference rule: a run event may use the reserve; nothing else may.
            limit_events = max_events - (0 if run_event else reserved_events)
            limit_bytes = max_bytes - (0 if run_event else reserved_bytes)
            fits = (
                len(model) < limit_events
                and sum(e.size for e in model) + size <= limit_bytes
            )
            expected = (
                QueueClosedError if closed else (None if fits else QueueFullError)
            )
            try:
                queue.offer(event, close=close)
            except (QueueClosedError, QueueFullError) as error:
                assert type(error) is expected
            else:
                assert expected is None
                model.append(event)
                if not run_event:
                    # An ordinary event never takes room held for run events.
                    assert len(model) <= max_events - reserved_events
                    assert sum(e.size for e in model) <= max_bytes - reserved_bytes
            closed = closed or close
        # The bound holds after every step, and the queue is exactly the model.
        assert len(queue) == len(model) <= max_events
        assert queue.queued_bytes == sum(e.size for e in model) <= max_bytes
        oldest = queue.oldest(len(model) + 1)
        assert len(oldest) == len(model)
        assert all(a is b for a, b in zip(oldest, model, strict=True))
        assert queue.closed is closed


def test_the_reserve_holds_every_run_event_of_a_build_at_the_clients_limits() -> None:
    """Started, one ending (completed, failed or blocked), and the service's
    record of a killed build."""

    largest_message = sanitize_text(
        "\U0001f600" * 10_000, limit=MAX_TELEMETRY_MESSAGE_CHARS
    )
    largest_details = sanitize_details(
        {f"key-{n}": "\U0001f600" * 50 for n in range(500)}
    )
    run_event = {
        "timestamp": "2026-10-09T12:00:00.000000+00:00",
        "event_type": "run",
        "stage_id": "complete",
        "status": "completed",
        "message": largest_message,
        "details": largest_details,
    }
    sample = {
        "cpu_user_seconds": 1e12,
        "cpu_system_seconds": 1e12,
        "rss_bytes": 2**62,
        "peak_rss_bytes": 2**62,
    }
    largest = QueuedEvent.encode(run_event, sample).size
    assert largest > MAX_TELEMETRY_DETAILS_BYTES
    # The service writes the killed-build record itself, around the longest
    # stage id it keeps.
    killed = runtime_module._unexpected_exit_event(
        "\U0001f600" * MAX_TELEMETRY_TEXT_CHARS
    )
    largest_killed = QueuedEvent.encode(killed, sample).size
    assert QUEUE_RESERVED_EVENTS >= 3
    assert QUEUE_RESERVED_BYTES >= 2 * largest + largest_killed
    assert QUEUE_RESERVED_EVENTS < QUEUE_MAX_EVENTS
    assert QUEUE_RESERVED_BYTES < QUEUE_MAX_BYTES


def test_a_long_stage_id_cannot_crowd_out_the_killed_build_record() -> None:
    """The client does not limit a stage id; the service's copy of it is cut.

    Progress events with a 300,000-character stage id, each under the 1 MiB
    message limit, fill the room outside the reserve. The killed-build record
    names the last stage, and must still fit the reserve.
    """

    service = _service(SimpleNamespace())
    stage = "s" * 300_000
    accepted = 0
    error_output = io.StringIO()
    with contextlib.redirect_stderr(error_output):
        while _send(service, _event(stage + str(accepted))):
            accepted += 1
        assert accepted > 0
        ordinary_room = (
            service.event_queue.max_bytes
            - service.event_queue.reserved_bytes
            - service.event_queue.queued_bytes
        )
        assert ordinary_room < len(stage)

        assert service._attempt(service._queue_unexpected_exit)

    assert service.event_queue.closed
    record, _ = service.event_queue.oldest(accepted + 1)[-1].decode()
    assert record["message"] == UNEXPECTED_PROCESS_EXIT_MESSAGE
    assert record["details"]["failed_during"] == stage[:MAX_TELEMETRY_TEXT_CHARS]
    # A heartbeat carries the same bounded copy.
    assert len(runtime_module._heartbeat_event(service._last_stage)["stage_id"]) == (
        MAX_TELEMETRY_TEXT_CHARS
    )


def test_queued_bytes_count_the_encoded_event_and_its_sample() -> None:
    sampler_reading = {"cpu_user_seconds": 1.5, "rss_bytes": 10}
    service = _service(
        SimpleNamespace(),
        sampler=SimpleNamespace(sample=lambda: sampler_reading),
    )
    events = [_event(f"stage-{index}", note="x" * index) for index in range(5)]
    for event in events:
        assert _send(service, event)

    queued = service.event_queue.oldest(10)
    assert [item.decode() for item in queued] == [(e, sampler_reading) for e in events]
    assert service.event_queue.queued_bytes == sum(
        len(json.dumps([e, sampler_reading], separators=(",", ":")).encode())
        for e in events
    )


# --- What the socket acknowledges ---------------------------------------------


def _reply(service: EmitterService, message: bytes) -> bytes:
    """Run one connection through the service's socket handling."""

    ours, theirs = socket.socketpair()
    with ours, theirs:
        ours.sendall(message)
        theirs.settimeout(1)
        return service._serve_connection(theirs)


def test_acknowledgement_means_queued_and_error_means_not_queued(capsys) -> None:
    service = _service(
        SimpleNamespace(),
        event_queue=EventQueue(
            max_events=3, max_bytes=10_000, reserved_events=1, reserved_bytes=500
        ),
    )

    def message(event) -> bytes:
        return json.dumps({"action": "event", "event": event}).encode() + b"\n"

    # Malformed events are refused and never queued.
    assert _reply(service, message({"stage_id": "x"})) == LOCAL_ACKNOWLEDGEMENT_ERROR
    assert _reply(service, message([1, 2])) == LOCAL_ACKNOWLEDGEMENT_ERROR
    nan = b'{"action":"event","event":{"event_type":"stage","status":"started",'
    nan += b'"details":{"x":NaN}}}\n'
    assert _reply(service, nan) == LOCAL_ACKNOWLEDGEMENT_ERROR
    assert len(service.event_queue) == 0

    # Two ordinary events fill the room outside the reserve.
    assert _reply(service, message(_event("a"))) == LOCAL_ACKNOWLEDGEMENT_OK
    assert _reply(service, message(_event("b"))) == LOCAL_ACKNOWLEDGEMENT_OK
    for _ in range(3):
        assert _reply(service, message(_event("c"))) == LOCAL_ACKNOWLEDGEMENT_ERROR
    # A run event still fits: the build's outcome is not crowded out.
    completed = {"event_type": "run", "stage_id": "complete", "status": "completed"}
    assert _reply(service, message(completed)) == LOCAL_ACKNOWLEDGEMENT_OK
    assert [item.decode()[0]["stage_id"] for item in service.event_queue.oldest(9)] == [
        "a",
        "b",
        "complete",
    ]
    # One warning line however many updates were refused.
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("warning: the local telemetry emitter service's queue")

    # Close refuses everything after it, heartbeats included.
    assert _reply(service, b'{"action":"close"}\n') == LOCAL_ACKNOWLEDGEMENT_OK
    service.event_queue.remove(3)
    assert _reply(service, message(_event("late"))) == LOCAL_ACKNOWLEDGEMENT_ERROR
    assert not service._attempt(service._queue_heartbeat)
    assert len(service.event_queue) == 0
    assert capsys.readouterr().err == ""


# --- The writer: order, no loss under contention, a bounded drain -------------


class _ScriptedSpool:
    """Appends like the real spool: in order, all or nothing, numbered.

    Each attempt first takes the next scripted lock outcome; an event whose
    details carry ``refused`` makes its whole transaction fail for a reason
    other than contention.
    """

    def __init__(self, lock_errors) -> None:
        self._lock_errors = iter(lock_errors)
        self.stored: list[tuple[str, int]] = []
        self.locked_attempts = 0
        self.busy_timeouts: set[float | None] = set()

    def append_many(
        self,
        registration,
        pairs,
        *,
        busy_timeout_seconds=None,
        lock_timeout_seconds=None,
        before_write=None,
    ):
        if before_write is not None:
            before_write()
        self.busy_timeouts.add(busy_timeout_seconds)
        if next(self._lock_errors, False):
            self.locked_attempts += 1
            raise _lock_error()
        if any(event["details"].get("refused") for event, _ in pairs):
            raise ValueError("the spool refused this event")
        for event, _ in pairs:
            self.stored.append((event["stage_id"], len(self.stored) + 1))


@settings(max_examples=200, deadline=None)
@given(
    operations=st.lists(
        st.one_of(
            st.tuples(
                st.just("send"),
                st.booleans(),  # a run event
                st.sampled_from([False] * 5 + [True]),  # the spool refuses it
            ),
            st.just(("write",)),
        ),
        max_size=60,
    ),
    lock_errors=st.lists(st.booleans(), max_size=150),
    max_events=st.integers(1, 15),
    reserved_events=st.integers(0, 4),
)
@example(
    operations=[("send", False, False)] * 3 + [("write",)],
    lock_errors=[True] * 40,
    max_events=5,
    reserved_events=0,
)
def test_every_acknowledged_event_is_stored_once_in_acknowledgement_order(
    operations, lock_errors, max_events, reserved_events
) -> None:
    clock = _Clock()
    spool = _ScriptedSpool(lock_errors)
    service = _service(
        spool,
        event_queue=EventQueue(
            max_events=max_events,
            max_bytes=10**9,
            reserved_events=reserved_events,
            reserved_bytes=0,
        ),
    )
    service.writer = SpoolWriter(
        spool,
        service.registration,
        service.event_queue,
        report_dropped=service._report_dropped,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    acknowledged: list[tuple[str, bool]] = []
    queue_was_full = False
    error_output = io.StringIO()
    with contextlib.redirect_stderr(error_output):
        for index, operation in enumerate(operations):
            if operation[0] == "write":
                # While serving, the writer outlasts any run of lock errors.
                assert service.writer.write_queued()
                assert len(service.event_queue) == 0
                continue
            _, run_event, refused = operation
            stage = f"event-{index}"
            event = _event(stage, refused=refused)
            if run_event:
                event["event_type"] = "run"
            if _send(service, event):
                acknowledged.append((stage, refused))
            else:
                queue_was_full = True
        unwritten = service.writer.finish(math.inf)

    kept = [stage for stage, refused in acknowledged if not refused]
    # Sequence order is acknowledgement order; nothing acknowledged is lost or
    # stored twice, whatever lock errors the spool raised.
    assert [stage for stage, _ in spool.stored] == kept
    assert [sequence for _, sequence in spool.stored] == list(range(1, len(kept) + 1))
    assert unwritten == 0
    # Only events the spool itself refused are dropped, each once. Each kind of
    # warning is one line however often it happens.
    assert service.writer.dropped == len(acknowledged) - len(kept)
    lines = error_output.getvalue().splitlines()
    assert sorted(line.split(" ")[6] for line in lines) == sorted(
        (["spool"] if service.writer.dropped else [])
        + (["queue"] if queue_was_full else [])
    )
    # Lock errors were retried with backoff, never reported.
    assert all(seconds <= SPOOL_RETRY_MAX_SECONDS for _, seconds in clock.sleeps)
    assert len(clock.sleeps) == spool.locked_attempts
    assert spool.busy_timeouts <= {WRITER_BUSY_TIMEOUT_SECONDS}


class _LockedUntil:
    """A spool another process keeps locked until ``lock_until`` on a fake clock.

    Like the real spool, an attempt runs ``before_write`` once it holds the
    spool, then its statements together wait up to ``busy_timeout_seconds``
    for the lock and succeed as soon as it is released within that. Only
    attempts that reach the database are recorded.
    """

    def __init__(self, clock: _Clock, lock_until: float, insert_seconds: float):
        self.clock = clock
        self.lock_until = lock_until
        self.insert_seconds = insert_seconds
        self.attempt_starts: list[float] = []
        self.attempt_ends: list[float] = []
        self.stored: list[str] = []

    def append_many(
        self,
        registration,
        pairs,
        *,
        busy_timeout_seconds=None,
        lock_timeout_seconds=None,
        before_write=None,
    ):
        if before_write is not None:
            before_write()
        started = self.clock.now
        self.attempt_starts.append(started)
        if self.lock_until > started + busy_timeout_seconds:
            self.clock.now = started + busy_timeout_seconds
            self.attempt_ends.append(self.clock.now)
            raise _lock_error()
        self.clock.now = max(started, self.lock_until) + self.insert_seconds
        self.attempt_ends.append(self.clock.now)
        self.stored.extend(event["stage_id"] for event, _ in pairs)

    def has_deliverable(self) -> bool:
        return False


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    events=st.integers(1, 450),
    locked_for=st.one_of(st.just(0.0), st.floats(0, 25)),
    drain_seconds=st.one_of(st.just(0.0), st.floats(0, 20)),
    insert_seconds=st.floats(0, 0.05),
    oversleep=st.one_of(st.just(0.0), st.floats(0, 1.0)),
)
@example(
    events=450, locked_for=3.0, drain_seconds=15.0, insert_seconds=0.05, oversleep=0.0
)
@example(
    events=10, locked_for=30.0, drain_seconds=15.0, insert_seconds=0.0, oversleep=0.0
)
# A backoff that overruns its plan by more than the margin before the deadline.
@example(
    events=10, locked_for=30.0, drain_seconds=1.0, insert_seconds=0.0, oversleep=0.9
)
def test_the_drain_writes_in_order_until_its_deadline_and_reports_the_rest(
    monkeypatch, events, locked_for, drain_seconds, insert_seconds, oversleep
) -> None:
    clock = _Clock(oversleep=oversleep)
    monkeypatch.setattr(runtime_module, "time", clock)
    spool = _LockedUntil(clock, clock.now + locked_for, insert_seconds)
    service = _service(spool, drain_seconds=drain_seconds)
    sent = [f"event-{index}" for index in range(events)]
    for stage in sent:
        assert _send(service, _event(stage))
    service._stop.set()
    started = clock.now
    deadline = started + drain_seconds
    error_output = io.StringIO()

    with contextlib.redirect_stderr(error_output):
        service._drain()

    last_start = deadline - WRITER_BUSY_TIMEOUT_SECONDS
    # No append touches the database within one lock-wait budget of the
    # deadline, however late a backoff wakes, so every append ends by the
    # deadline plus its own inserts.
    assert all(start < last_start for start in spool.attempt_starts)
    assert all(end <= deadline + insert_seconds + 1e-9 for end in spool.attempt_ends)
    # What was written is the oldest events, in order.
    assert spool.stored == sent[: len(spool.stored)]
    unwritten = events - len(spool.stored)
    lines = error_output.getvalue().splitlines()
    if unwritten:
        assert len(lines) == 1
        assert f"could not write {unwritten} queued update(s)" in lines[0]
    else:
        assert lines == []
    # Contention that clears with room to spare loses nothing: an attempt in
    # its busy wait succeeds at the release, a sleeping one within a backoff
    # and its overrun.
    batches = math.ceil(events / WRITER_BATCH_EVENTS)
    release = started + locked_for
    wakes_by = release + SPOOL_RETRY_MAX_SECONDS + oversleep
    if wakes_by + batches * insert_seconds < last_start:
        assert unwritten == 0
    # A lock held through the whole drain writes nothing and loses all.
    if release >= deadline:
        assert spool.stored == []


@pytest.mark.parametrize("window", [2.0, 0.1, 0.0])
def test_a_deadline_that_arrives_mid_retry_stops_every_later_attempt(window) -> None:
    """The cutoff is checked in every attempt, not once before the first.

    ``window`` is how far ahead the deadline is when it arrives, during a
    backoff sleep of a writer retrying with no deadline. At 0.1 s and 0 s the
    cutoff has already passed, so no further attempt may touch the database.
    """

    clock = _Clock()
    queue = EventQueue()
    queue.offer(QueuedEvent.encode(_event("a"), None))
    spool = _LockedUntil(clock, math.inf, 0.0)
    writer = SpoolWriter(
        spool,
        _registration(),
        queue,
        report_dropped=pytest.fail,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    arrives_at = clock.now + 30.0
    advance = clock.sleep

    def sleep(seconds):
        advance(seconds)
        # The service stops while the writer sleeps between attempts.
        if clock.now >= arrives_at and writer._drain_deadline == math.inf:
            writer.begin_drain(clock.now + window)
        assert clock.now < arrives_at + 60, "the writer ignored the drain deadline"

    writer._sleep = sleep
    assert not writer.write_queued()
    assert is_transient_spool_error(writer.stopped_by)
    assert len(queue) == 1
    cutoff = writer._drain_deadline - WRITER_BUSY_TIMEOUT_SECONDS
    assert spool.attempt_starts, "the writer never reached the spool"
    assert all(start < cutoff for start in spool.attempt_starts[-3:])
    if window < WRITER_BUSY_TIMEOUT_SECONDS:
        # The cutoff had passed when the deadline arrived: nothing ran after.
        assert spool.attempt_starts[-1] < arrives_at


# --- The retry helper with a deadline that moves ------------------------------


@settings(deadline=None)
@given(
    outcomes=st.lists(
        st.sampled_from(["locked", "locked", "locked", "ok", "fatal"]),
        min_size=1,
        max_size=40,
    ),
    finite_from=st.integers(0, 40),
    window=st.floats(-1, 5),
    jitter_fraction=st.floats(0, 1),
)
def test_retry_reads_a_moving_deadline_after_every_lock_error(
    outcomes, finite_from, window, jitter_fraction
) -> None:
    clock = SimpleNamespace(now=100.0)
    reads: list[float] = []
    decisions: list[tuple[float, float, float]] = []  # (at, wait, deadline read)
    attempts: list[float] = []
    brought_forward: list[float] = []

    def deadline() -> float:
        # Infinite until the caller brings it forward, then fixed.
        if len(reads) >= finite_from:
            if not brought_forward:
                brought_forward.append(clock.now + window)
            value = brought_forward[0]
        else:
            value = math.inf
        reads.append(value)
        return value

    def operation():
        attempts.append(clock.now)
        # Once the deadline is finite, at most window / (shortest wait) more
        # attempts fit: 5 s / 0.025 s. A helper that kept an old reading of
        # infinity would retry a spool that stays locked forever.
        assert len(attempts) <= finite_from + 250, "the deadline was not re-read"
        kind = outcomes[min(len(attempts) - 1, len(outcomes) - 1)]
        clock.now += 0.01
        if kind == "locked":
            raise _lock_error()
        if kind == "fatal":
            raise ValueError("schema error")
        return len(attempts)

    def jitter(low, high):
        return low + jitter_fraction * (high - low)

    def sleep(seconds):
        decisions.append((clock.now, seconds, reads[-1]))
        clock.now += seconds

    try:
        retry_spool_contention(
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

    kinds = [outcomes[min(i, len(outcomes) - 1)] for i in range(len(attempts))]
    assert kinds[-1] == outcome
    # The deadline is read once per lock error and never before the first try.
    assert len(reads) == kinds.count("locked")
    # No wait reaches the deadline read just before it.
    assert all(at + wait < read for at, wait, read in decisions)
    # While it reads infinity the helper never gives up on contention.
    if outcome == "locked":
        assert reads[-1] < math.inf
    if all(read == math.inf for read in reads):
        assert outcome != "locked"


# --- Batched appends ------------------------------------------------------------


@settings(max_examples=25, deadline=None)
@given(
    stages=st.lists(st.text("abc", min_size=1, max_size=3), min_size=1, max_size=12),
    cuts=st.lists(st.integers(0, 12), max_size=6),
)
def test_appending_in_batches_matches_appending_one_at_a_time(stages, cuts) -> None:
    """Differential: batching changes how many transactions, not what is stored."""

    boundaries = sorted({0, len(stages), *(min(cut, len(stages)) for cut in cuts)})
    batches = [
        stages[start:end]
        for start, end in zip(boundaries, boundaries[1:], strict=False)
    ]
    registration = _registration()
    stored = []
    with tempfile.TemporaryDirectory() as directory:
        for name, write in (
            (
                "one-at-a-time",
                lambda spool: [
                    spool.append(registration, _event(s), resources={"n": 1})
                    for s in stages
                ],
            ),
            (
                "batched",
                lambda spool: [
                    spool.append_many(
                        registration, [(_event(s), {"n": 1}) for s in batch]
                    )
                    for batch in batches
                ],
            ),
        ):
            spool = EventSpool(Path(directory) / f"{name}.sqlite3")
            try:
                spool.register(registration)
                write(spool)
                stored.append(
                    [
                        {
                            key: value
                            for key, value in payload.items()
                            if key not in {"event_id", "timestamp"}
                        }
                        for payload in spool.batch("run-a", "producer-a", limit=100)
                    ]
                )
            finally:
                spool._engine.dispose()
    one_at_a_time, batched = stored
    assert batched == one_at_a_time
    assert [p["sequence"] for p in batched] == list(range(1, len(stages) + 1))
    assert [p["stage_id"] for p in batched] == stages


def test_a_batch_is_stored_whole_or_not_at_all_and_waits_only_its_own_wait(
    tmp_path,
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    spool = EventSpool(spool_path)
    registration = _registration()
    spool.register(registration)
    pairs = [(_event(f"s{index}"), None) for index in range(5)]
    try:
        with _write_lock(spool_path):
            started = time.monotonic()
            with pytest.raises(OperationalError) as locked:
                spool.append_many(
                    registration,
                    pairs,
                    busy_timeout_seconds=WRITER_BUSY_TIMEOUT_SECONDS,
                )
            waited = time.monotonic() - started
        assert is_transient_spool_error(locked.value)
        assert waited < DATABASE_TIMEOUT_SECONDS / 2
        assert _stored_stages(spool_path) == []
        # The failed call changed nothing, so repeating it numbers from 1.
        payloads = spool.append_many(registration, pairs)
        assert [p["sequence"] for p in payloads] == [1, 2, 3, 4, 5]
        assert _stored_stages(spool_path) == [f"s{index}" for index in range(5)]
        # The shorter wait applied to that call only.
        with spool._lock, spool._engine.connect() as connection:
            milliseconds = connection.exec_driver_sql("PRAGMA busy_timeout").scalar()
        assert milliseconds == DATABASE_TIMEOUT_SECONDS * 1000
    finally:
        spool._engine.dispose()


def test_a_commit_a_reader_blocks_stores_nothing_so_a_retry_cannot_duplicate(
    tmp_path,
) -> None:
    """The lock error can come at COMMIT, after the inserts, and still undoes them."""

    spool_path = tmp_path / "events.sqlite3"
    spool = EventSpool(spool_path)
    registration = _registration()
    spool.register(registration)
    pairs = [(_event(f"s{index}"), None) for index in range(3)]
    reader = sqlite3.connect(spool_path, isolation_level=None)
    try:
        # A read transaction holds a SHARED lock, which a commit must wait out.
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM telemetry_events").fetchone()
        with pytest.raises(OperationalError) as locked:
            spool.append_many(
                registration, pairs, busy_timeout_seconds=WRITER_BUSY_TIMEOUT_SECONDS
            )
        assert is_transient_spool_error(locked.value)
        reader.execute("COMMIT")
        assert _stored_stages(spool_path) == []
        payloads = spool.append_many(registration, pairs)
        assert [p["sequence"] for p in payloads] == [1, 2, 3]
        assert _stored_stages(spool_path) == ["s0", "s1", "s2"]
    finally:
        reader.close()
        spool._engine.dispose()


def test_an_appends_lock_waits_share_one_budget(tmp_path) -> None:
    """Waiting for the write lock and again at COMMIT must not add up.

    Another connection holds the write lock for 1.5 s while a reader holds a
    SHARED lock throughout. With a 2 s budget the append gets the write lock
    after 1.5 s and then has 0.5 s, not another 2 s, to wait at COMMIT.
    """

    spool_path = tmp_path / "events.sqlite3"
    spool = EventSpool(spool_path)
    registration = _registration()
    spool.register(registration)
    reader = sqlite3.connect(spool_path, isolation_level=None)
    holding = threading.Event()

    def hold_write_lock() -> None:
        with _write_lock(spool_path):
            holding.set()
            time.sleep(1.5)

    holder = threading.Thread(target=hold_write_lock, daemon=True)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM telemetry_events").fetchone()
        holder.start()
        assert holding.wait(10)
        started = time.monotonic()
        with pytest.raises(OperationalError) as locked:
            spool.append_many(
                registration, [(_event("a"), None)], busy_timeout_seconds=2.0
            )
        waited = time.monotonic() - started
        assert is_transient_spool_error(locked.value)
        # Past the other writer's 1.5 s, so the wait at COMMIT happened, and
        # well short of the 3.5 s that a full second wait would take.
        assert 1.5 <= waited < 2.9
        reader.execute("COMMIT")
        assert _stored_stages(spool_path) == []
    finally:
        holder.join(timeout=10)
        reader.close()
        spool._engine.dispose()


def test_this_processes_own_spool_lock_cannot_carry_a_write_past_the_cutoff(
    tmp_path,
) -> None:
    """The worker can hold the spool's in-process lock for a whole busy wait.

    Another thread holds it for 3 s; the drain deadline is 1 s away. The
    writer must give up at the cutoff, not write when the lock comes free.
    """

    spool_path = tmp_path / "events.sqlite3"
    spool = EventSpool(spool_path)
    registration = _registration()
    spool.register(registration)
    queue = EventQueue()
    for stage in ("a", "b"):
        queue.offer(QueuedEvent.encode(_event(stage), None))
    writer = SpoolWriter(spool, registration, queue, report_dropped=pytest.fail)
    holding = threading.Event()

    def hold_spool_lock() -> None:
        with spool._lock:
            holding.set()
            time.sleep(3.0)

    holder = threading.Thread(target=hold_spool_lock, daemon=True)
    try:
        holder.start()
        assert holding.wait(10)
        started = time.monotonic()
        writer.begin_drain(started + 1.0)
        assert not writer.write_queued()
        gave_up_after = time.monotonic() - started
        assert isinstance(writer.stopped_by, SpoolBusyError)
        assert is_transient_spool_error(writer.stopped_by)
        # At the cutoff, well before the other thread lets go.
        assert gave_up_after < 2.5
        holder.join(timeout=10)
        assert len(queue) == 2
        assert _stored_stages(spool_path) == []
        # Asked directly, the spool refuses within the caller's wait.
        with spool._lock:
            refused = []

            def ask() -> None:
                try:
                    spool.append_many(
                        registration, [(_event("c"), None)], lock_timeout_seconds=0.1
                    )
                except Exception as error:
                    refused.append(error)

            asking = threading.Thread(target=ask, daemon=True)
            asking.start()
            asking.join(timeout=10)
        assert [type(error) for error in refused] == [SpoolBusyError]
    finally:
        holder.join(timeout=10)
        spool._engine.dispose()


def test_before_write_runs_with_the_spool_held_and_can_abandon_the_append(
    tmp_path,
) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    registration = _registration()
    spool.register(registration)
    held: list[bool] = []

    def refuse() -> None:
        # RLock._is_owned is true only for the thread that holds it.
        held.append(spool._lock._is_owned())
        raise RuntimeError("too late")

    try:
        with pytest.raises(RuntimeError, match="too late"):
            spool.append_many(registration, [(_event("a"), None)], before_write=refuse)
        assert held == [True]
        assert _stored_stages(spool.path) == []
        assert not spool._lock._is_owned()
    finally:
        spool._engine.dispose()


# --- Real lock contention -----------------------------------------------------


class _Sampler:
    def __init__(self) -> None:
        self.alive = True

    def sample(self) -> dict[str, object]:
        return {"rss_bytes": 100}

    def parent_alive(self) -> bool:
        return self.alive


def _start_service(spool_path: Path, **overrides):
    spool = EventSpool(spool_path)
    registration = _registration()
    spool.register(registration)
    # No collector here, so the shutdown drain stops once the spool is written.
    spool.has_deliverable = lambda: False
    socket_path = _short_socket_path()
    service = EmitterService(
        **(
            {
                "socket_path": socket_path,
                "registration": registration,
                "spool": spool,
                "delivery": SimpleNamespace(flush_once=lambda: False),
                "sampler": _Sampler(),
                "heartbeat_seconds": 60,
                "drain_seconds": 5,
            }
            | overrides
        )
    )
    serving = threading.Thread(target=service.run, daemon=True)
    serving.start()
    deadline = time.monotonic() + 10
    while not socket_path.exists():
        assert time.monotonic() < deadline, "service never bound its socket"
        time.sleep(0.01)
    client = LocalTelemetryEmitter(
        run=TelemetryRun("run-a", "US", "test-pipeline", producer_id="producer-a"),
        process=None,
        socket_path=socket_path,
        runtime_dir=None,
    )
    return service, serving, client


def _stop(service: EmitterService, serving: threading.Thread, client) -> None:
    client.close()
    service._stop.set()
    serving.join(timeout=30)
    service.spool._engine.dispose()


def test_a_held_spool_lock_neither_delays_nor_loses_a_builds_events(
    tmp_path, capsys
) -> None:
    """The lock outlasts SQLite's whole 5 s busy wait, which used to drop events."""

    spool_path = tmp_path / "events.sqlite3"
    service, serving, client = _start_service(spool_path)
    spool = service.spool
    lock_errors: list[float] = []
    append_many = spool.append_many

    def recording(*args, **kwargs):
        try:
            return append_many(*args, **kwargs)
        except Exception as error:
            if is_transient_spool_error(error):
                lock_errors.append(time.monotonic())
            raise

    spool.append_many = recording
    sent: list[str] = []
    send_seconds: list[float] = []
    try:
        with _write_lock(spool_path):
            held_from = time.monotonic()
            while time.monotonic() - held_from < DATABASE_TIMEOUT_SECONDS + 1.0:
                stage = f"stage-{len(sent)}"
                started = time.monotonic()
                client.stage(stage, message="Working.")
                send_seconds.append(time.monotonic() - started)
                sent.append(stage)
                time.sleep(0.02)
            # A burst, as a calibration loop sends them.
            for _ in range(200):
                stage = f"stage-{len(sent)}"
                started = time.monotonic()
                client.stage(stage)
                send_seconds.append(time.monotonic() - started)
                sent.append(stage)
            # Every reply came while the spool was locked, before any write.
            assert _stored_stages(spool_path) == []
            released_at = time.monotonic()
        deadline = time.monotonic() + 20
        while len(_stored_stages(spool_path)) < len(sent):
            assert time.monotonic() < deadline, "queued events never reached the spool"
            time.sleep(0.05)
    finally:
        _stop(service, serving, client)

    # Not lost, not reordered, not duplicated.
    assert _stored_stages(spool_path) == sent
    events = _stored_events(spool_path, "run-a", "producer-a")
    assert [event["sequence"] for event in events] == list(range(1, len(sent) + 1))
    assert all(event["resources"] == {"rss_bytes": 100} for event in events)
    # Not delayed: the client never hit its own timeout or printed its warning.
    assert not client._warned
    assert max(send_seconds) < DEFAULT_SEND_TIMEOUT_SECONDS
    assert "could not queue" not in capsys.readouterr().err
    # The writer met the lock repeatedly while it was held.
    assert len([at for at in lock_errors if at < released_at]) >= 2
    assert not serving.is_alive()


def test_close_drains_events_queued_behind_a_held_lock(tmp_path, capsys) -> None:
    spool_path = tmp_path / "events.sqlite3"
    service, serving, client = _start_service(spool_path, drain_seconds=10)
    sent = [f"stage-{index}" for index in range(30)]
    try:
        with _write_lock(spool_path):
            for stage in sent:
                client.stage(stage)
            client.close()
            closed_at = time.monotonic()
            time.sleep(1.0)
            # Still locked, still serving its drain.
            assert serving.is_alive()
        serving.join(timeout=20)
        stopped_after = time.monotonic() - closed_at
    finally:
        _stop(service, serving, client)

    assert not serving.is_alive()
    assert stopped_after < 10 + 1
    assert _stored_stages(spool_path) == sent
    assert capsys.readouterr().err == ""


def test_a_lock_held_through_the_drain_loses_only_the_queue_and_says_so(
    tmp_path, capsys
) -> None:
    spool_path = tmp_path / "events.sqlite3"
    drain_seconds = 1.0
    service, serving, client = _start_service(spool_path, drain_seconds=drain_seconds)
    try:
        with _write_lock(spool_path):
            for index in range(5):
                client.stage(f"stage-{index}")
            client.close()
            closed_at = time.monotonic()
            serving.join(timeout=drain_seconds + 5)
            stopped_after = time.monotonic() - closed_at
    finally:
        _stop(service, serving, client)

    assert not serving.is_alive()
    # The drain gives up on time: its last attempt starts a statement wait
    # before the deadline, and the socket thread's accept wait adds at most
    # its own timeout.
    assert stopped_after < drain_seconds + 1.5
    assert _stored_stages(spool_path) == []
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert "could not write 5 queued update(s)" in lines[0]
    assert "(database is locked)" in lines[0]


class _HeldDelivery:
    """A delivery step that blocks, as a collector request or a busy wait does."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def flush_once(self) -> bool:
        self.entered.set()
        self.release.wait(60)
        return False


@pytest.mark.parametrize("worker", ["returns after the window", "never returns"])
def test_run_does_not_return_while_the_writer_may_still_write(
    tmp_path, capsys, worker
) -> None:
    """One shutdown window, opened at the close, whatever the worker is doing.

    The worker is inside a delivery step when the build closes, and the spool
    stays locked. The writer must still get its deadline at the close, and
    ``run`` must not return before the writer has stopped and the events it
    left are reported: not earlier with the writer still retrying, and not a
    second window later.
    """

    spool_path = tmp_path / "events.sqlite3"
    drain_seconds = 2.0
    delivery = _HeldDelivery()
    service, serving, client = _start_service(
        spool_path, drain_seconds=drain_seconds, delivery=delivery
    )
    try:
        assert delivery.entered.wait(15), "the worker never reached delivery"
        with _write_lock(spool_path):
            for index in range(3):
                client.stage(f"stage-{index}")
            client.close()
            closed_at = time.monotonic()
            if worker == "returns after the window":
                threading.Timer(drain_seconds + 0.5, delivery.release.set).start()
            serving.join(timeout=30)
            returned_after = time.monotonic() - closed_at
            # Read at the moment run returned, with the spool still locked.
            writer_alive = service.writer.is_alive()
            lines = capsys.readouterr().err.splitlines()
    finally:
        delivery.release.set()
        _stop(service, serving, client)

    assert not serving.is_alive()
    assert not writer_alive
    assert len(lines) == 1
    assert "could not write 3 queued update(s)" in lines[0]
    assert "(database is locked)" in lines[0]
    # The window, the worker's interval, a statement wait and the accept wait;
    # a second window would add drain_seconds.
    assert returned_after < drain_seconds + 1.0 + 0.25 + 0.5 + 0.75
    assert _stored_stages(spool_path) == []


def test_a_killed_build_is_recorded_after_every_event_it_sent(tmp_path) -> None:
    spool_path = tmp_path / "events.sqlite3"
    service, serving, client = _start_service(spool_path, drain_seconds=10)
    sent = [f"stage-{index}" for index in range(10)]
    try:
        with _write_lock(spool_path):
            for stage in sent:
                client.stage(stage)
            # The build dies without closing; the worker notices on its tick.
            service.sampler.alive = False
            deadline = time.monotonic() + 10
            while not service.event_queue.closed:
                assert time.monotonic() < deadline, "worker never noticed"
                time.sleep(0.05)
            time.sleep(0.5)
        serving.join(timeout=20)
    finally:
        _stop(service, serving, client)

    events = _stored_events(spool_path, "run-a", "producer-a")
    assert [event["stage_id"] for event in events[:-1]] == sent
    assert events[-1]["message"] == UNEXPECTED_PROCESS_EXIT_MESSAGE
    assert events[-1]["details"]["failed_during"] == sent[-1]
    assert [event["sequence"] for event in events] == list(range(1, len(sent) + 2))


def test_service_stops_serving_when_its_writer_dies(tmp_path, capsys) -> None:
    """Acknowledged events would never reach the spool without the writer."""

    spool_path = tmp_path / "events.sqlite3"
    service, serving, client = _start_service(spool_path, drain_seconds=0)

    def bug() -> bool:
        raise RuntimeError("writer bug")

    service.writer.write_queued = bug
    try:
        client.stage("wakes the writer")
        serving.join(timeout=10)
        assert not serving.is_alive()
    finally:
        _stop(service, serving, client)

    err = capsys.readouterr().err
    assert "spool writer stopped: RuntimeError: writer bug" in err
    assert "Traceback" not in err


def test_build_sends_through_a_held_spool_lock_end_to_end(
    tmp_path, capfd, real_local_telemetry
) -> None:
    """The real client and the real service process."""

    spool_path = tmp_path / "spool" / "events.sqlite3"
    emitter = LocalTelemetryEmitter.start(
        run_id="held-lock-run",
        country_code="US",
        pipeline="test-pipeline",
        development_collector_url=_LOOPBACK_COLLECTOR,
        spool_path=spool_path,
        heartbeat_seconds=60,
        startup_timeout_seconds=60,
    )
    assert emitter.available
    send_seconds: list[float] = []
    with _write_lock(spool_path):
        held_from = time.monotonic()
        done = 0
        while time.monotonic() - held_from < DATABASE_TIMEOUT_SECONDS + 1.0:
            started = time.monotonic()
            emitter.progress("calibrating", done=done, total=1000)
            send_seconds.append(time.monotonic() - started)
            done += 1
            time.sleep(0.05)
    emitter.complete()
    assert emitter._process is not None
    assert emitter._process.wait(timeout=60) == 0

    events = _stored_events(spool_path, "held-lock-run", emitter.run.producer_id)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["message"] == BUILD_STARTED_MESSAGE
    assert [event["details"]["done"] for event in events[1:-1]] == list(range(done))
    assert events[-1]["message"] == BUILD_COMPLETED_MESSAGE
    assert not emitter._warned
    assert max(send_seconds) < DEFAULT_SEND_TIMEOUT_SECONDS
    err = capfd.readouterr().err
    assert "could not queue" not in err
    assert "Traceback" not in err
