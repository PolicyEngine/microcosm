"""Unix-socket runtime for the telemetry emitter service."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DEFAULT_DRAIN_SECONDS,
    DRAIN_RETRY_SECONDS,
    DRAIN_TIME_REASON,
    EVENT_DROPPED_WARNING,
    EVENT_FIELDS_ERROR,
    EVENT_OBJECT_ERROR,
    FAILURE_CLASS_UNEXPECTED_PROCESS_EXIT,
    LOCAL_MESSAGE_TOO_LARGE_ERROR,
    MINIMUM_HEARTBEAT_SECONDS,
    QUEUE_FULL_WARNING,
    SOCKET_ACCEPT_TIMEOUT_SECONDS,
    SOCKET_CONNECTION_TIMEOUT_SECONDS,
    SOCKET_LISTEN_BACKLOG,
    UNSUPPORTED_ACTION_ERROR,
    UNWRITTEN_EVENTS_WARNING,
    WORKER_INTERVAL_SECONDS,
    WORKER_STEP_WARNING,
)
from microcosm.build.telemetry_emitter_service.contention import (
    is_transient_spool_error,
)
from microcosm.build.telemetry_emitter_service.diagnostics import (
    describe_error,
    write_warning,
)
from microcosm.build.telemetry_emitter_service.event_queue import (
    DrainDeadlineError,
    EventQueue,
    EventRefusedError,
    QueuedEvent,
    QueueFullError,
    SpoolWriter,
)
from microcosm.build.telemetry_emitter_service.resources import ProcessTreeSampler
from microcosm.build.telemetry_emitter_service.spool import EventSpool
from microcosm.build.telemetry_emitter_service.timestamps import utc_now
from microcosm.build.telemetry_protocol import (
    ACTION_CLOSE,
    ACTION_EVENT,
    ACTION_PING,
    EVENT_TYPE_HEARTBEAT,
    EVENT_TYPE_RUN,
    LOCAL_ACKNOWLEDGEMENT_ERROR,
    LOCAL_ACKNOWLEDGEMENT_OK,
    LOCAL_MESSAGE_DELIMITER,
    LOCAL_SOCKET_READ_BYTES,
    MAX_LOCAL_MESSAGE_BYTES,
    STAGE_COMPLETE,
    STAGE_CREATED,
    STAGE_FAILED,
    STATUS_FAILED,
    STATUS_PROGRESS,
    UNEXPECTED_PROCESS_EXIT_MESSAGE,
)


def _heartbeat_event(stage_id: str) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "event_type": EVENT_TYPE_HEARTBEAT,
        "stage_id": stage_id,
        "status": STATUS_PROGRESS,
        "message": None,
        "details": {},
    }


def _unexpected_exit_event(stage_id: str) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "event_type": EVENT_TYPE_RUN,
        "stage_id": STAGE_FAILED,
        "status": STATUS_FAILED,
        "message": UNEXPECTED_PROCESS_EXIT_MESSAGE,
        "details": {
            "failure_class": FAILURE_CLASS_UNEXPECTED_PROCESS_EXIT,
            "failed_during": stage_id,
        },
    }


class EmitterService:
    """Private local socket server with a spool writer and a delivery worker.

    Three threads share the work. The socket thread validates each message,
    queues events in memory and replies. The writer thread appends queued
    events to the spool, oldest first. The worker samples resources, queues
    heartbeats, prunes, delivers to the collector and notices a dead build.

    The reply is the acknowledgement contract with the build:

    - ``ok`` to an event means the event was valid and is queued in memory
      behind every event queued before it, this service's heartbeats
      included. The writer appends queued events to the spool in that order,
      and the spool assigns their producer sequences as it does, so the
      sequence order of stored events is acknowledgement order.
    - ``ok`` does not mean the event is on disk, and the reply never waits for
      the spool. When the spool is free the event is written moments later.
      While another process holds the spool's lock the event waits in the
      queue, and the writer retries for as long as the contention lasts.
    - An acknowledged event is lost only if this process is killed before
      writing it; if the spool refuses it for a reason other than lock
      contention (one warning line per error type); if the writer thread
      fails (one line, and the service stops serving); or if the shutdown
      drain ends before the writer reaches it, as when the spool stays locked
      throughout the drain (one line with the count).
    - ``error`` means the message was malformed, or the queue was full or
      closed. The event was not queued.
    """

    def __init__(
        self,
        *,
        socket_path: Path,
        registration: Mapping[str, Any],
        spool: EventSpool,
        delivery: CollectorDelivery,
        sampler: ProcessTreeSampler,
        heartbeat_seconds: float,
        drain_seconds: float = DEFAULT_DRAIN_SECONDS,
        event_queue: EventQueue | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.registration = dict(registration)
        self.spool = spool
        self.delivery = delivery
        self.sampler = sampler
        self.heartbeat_seconds = max(
            MINIMUM_HEARTBEAT_SECONDS,
            heartbeat_seconds,
        )
        self.drain_seconds = max(0.0, drain_seconds)
        self._stop = threading.Event()
        self._last_stage = STAGE_CREATED
        # Warnings already printed, shared by the three threads.
        self._warning_lock = threading.Lock()
        self._reported_error_types: set[str] = set()
        self._dropped_error_types: set[str] = set()
        self._warned_queue_full = False
        self._drain_lock = threading.Lock()
        self._drain_started = False
        self.event_queue = event_queue if event_queue is not None else EventQueue()
        # The writer reads this module's clock when it runs, so a test that
        # replaces the module's time drives the writer too.
        self.writer = SpoolWriter(
            spool,
            self.registration,
            self.event_queue,
            report_dropped=self._report_dropped,
            clock=lambda: time.monotonic(),
            sleep=lambda seconds: time.sleep(seconds),
        )

    def run(self) -> None:
        """Serve local messages until the client closes or exits.

        The spool must already hold this producer's registration. Queued
        events are written to the spool before this returns, within
        ``drain_seconds`` of the stop.
        """

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink(missing_ok=True)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(self.socket_path))
                os.chmod(self.socket_path, 0o600)
                server.listen(SOCKET_LISTEN_BACKLOG)
                server.settimeout(SOCKET_ACCEPT_TIMEOUT_SECONDS)
                self.writer.start()
                worker = threading.Thread(target=self._worker, daemon=True)
                worker.start()
                self._serve(server, worker)
                self._stop.set()
                worker.join(timeout=self.drain_seconds + WORKER_INTERVAL_SECONDS)
                # The worker drains on its way out. One that died or hung
                # before then leaves the queue to this thread.
                self._drain()
        finally:
            self.socket_path.unlink(missing_ok=True)
            try:
                self.socket_path.parent.rmdir()
            except OSError:
                pass

    def _serve(self, server: socket.socket, worker: threading.Thread) -> None:
        # Only the worker notices that the build died without closing, so a
        # service whose worker has stopped must stop too, or it would serve a
        # dead build forever. Without the writer, acknowledged events would
        # never reach the spool.
        while not self._stop.is_set() and worker.is_alive() and self.writer.is_alive():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(SOCKET_CONNECTION_TIMEOUT_SECONDS)
                response = self._serve_connection(connection)
                try:
                    connection.sendall(response)
                except OSError:
                    pass

    def _serve_connection(self, connection: socket.socket) -> bytes:
        """Read and act on one message, returning the reply the class describes."""

        try:
            message = self._read_message(connection)
            self._handle(message)
        except Exception:
            return LOCAL_ACKNOWLEDGEMENT_ERROR
        return LOCAL_ACKNOWLEDGEMENT_OK

    @staticmethod
    def _read_message(connection: socket.socket) -> dict[str, Any]:
        chunks = bytearray()
        while len(chunks) <= MAX_LOCAL_MESSAGE_BYTES:
            data = connection.recv(LOCAL_SOCKET_READ_BYTES)
            if not data:
                break
            chunks.extend(data)
            if LOCAL_MESSAGE_DELIMITER in data:
                break
        if len(chunks) > MAX_LOCAL_MESSAGE_BYTES:
            raise ValueError(LOCAL_MESSAGE_TOO_LARGE_ERROR)
        return json.loads(bytes(chunks).split(LOCAL_MESSAGE_DELIMITER, 1)[0])

    def _handle(self, message: Mapping[str, Any]) -> None:
        """Act on one message; raising makes the reply ``error``.

        An event is validated here and queued, never written: the reply must
        not wait for the spool.
        """

        action = message.get("action")
        if action == ACTION_EVENT:
            event = message.get("event")
            if not isinstance(event, Mapping):
                raise ValueError(EVENT_OBJECT_ERROR)
            if "event_type" not in event or "status" not in event:
                raise ValueError(EVENT_FIELDS_ERROR)
            stage_id = event.get("stage_id")
            if isinstance(stage_id, str) and stage_id not in {
                STAGE_COMPLETE,
                STAGE_FAILED,
            }:
                self._last_stage = stage_id
            self._enqueue(event)
        elif action == ACTION_CLOSE:
            # Nothing, not even a heartbeat, is queued behind the build's last
            # event.
            self.event_queue.close()
            self._stop.set()
        elif action == ACTION_PING:
            return
        else:
            raise ValueError(UNSUPPORTED_ACTION_ERROR)

    def _enqueue(self, event: Mapping[str, Any], *, close: bool = False) -> None:
        """Queue an event with a resource sample taken now."""

        queued = QueuedEvent.encode(event, self.sampler.sample())
        try:
            self.event_queue.offer(queued, close=close)
        except QueueFullError:
            with self._warning_lock:
                first = not self._warned_queue_full
                self._warned_queue_full = True
            if first:
                write_warning(
                    QUEUE_FULL_WARNING.format(
                        events=len(self.event_queue),
                        megabytes=self.event_queue.queued_bytes / (1024 * 1024),
                    )
                )
            raise

    def _worker(self) -> None:
        # A failed step is skipped for this tick only. The parent check below
        # must keep running: it is the only thing that stops a service whose
        # build died without closing it.
        next_heartbeat = time.monotonic() + self.heartbeat_seconds
        while not self._stop.wait(WORKER_INTERVAL_SECONDS):
            now = time.monotonic()
            # Sample every worker iteration so short-lived build children are
            # much less likely to disappear between stage and heartbeat events.
            self._attempt(self.sampler.sample)
            # A heartbeat that fails stays due, so the next tick retries it.
            if now >= next_heartbeat and self._attempt(self._queue_heartbeat):
                next_heartbeat = now + self.heartbeat_seconds
            self._attempt(self.spool.prune_if_due)
            self._attempt(self.delivery.flush_once)
            # A build that closed while this tick ran may already have exited
            # cleanly; only a build that never closed died unexpectedly.
            if not self._stop.is_set() and not self._parent_alive():
                self._attempt(self._queue_unexpected_exit)
                # Closed even when the record could not be queued: nothing
                # arrives from a dead build.
                self.event_queue.close()
                self._stop.set()
                break
        self._drain()

    def _parent_alive(self) -> bool:
        # A parent that cannot be checked is treated as gone: the alternative
        # is a service that may outlive its build.
        try:
            return self.sampler.parent_alive()
        except Exception as error:
            self._report(error)
            return False

    def _queue_heartbeat(self) -> None:
        self._enqueue(_heartbeat_event(self._last_stage))

    def _queue_unexpected_exit(self) -> None:
        # This is the only record of a build that was killed. It goes behind
        # every event the build sent and closes the queue, so it is the last
        # event of the run; as a run event it may use the reserved room.
        self._enqueue(_unexpected_exit_event(self._last_stage), close=True)

    def _attempt(self, step: Callable[[], object]) -> bool:
        """Run one worker step, reporting rather than raising its failure."""

        try:
            step()
        except Exception as error:
            self._report(error)
            return False
        return True

    def _report(self, error: Exception) -> None:
        # Lock contention clears by itself and the step runs again on a later
        # tick, so it is not worth a line in the build's log. Nor is an event
        # the queue refused: a full queue has printed its own line. Anything
        # else is reported once per error type.
        if is_transient_spool_error(error) or isinstance(error, EventRefusedError):
            return
        error_type = type(error).__name__
        with self._warning_lock:
            if error_type in self._reported_error_types:
                return
            self._reported_error_types.add(error_type)
        write_warning(
            WORKER_STEP_WARNING.format(
                error_type=error_type,
                error=describe_error(error),
            )
        )

    def _report_dropped(self, error: Exception) -> None:
        # An event the spool refused for a reason other than lock contention,
        # reported once per error type.
        error_type = type(error).__name__
        with self._warning_lock:
            if error_type in self._dropped_error_types:
                return
            self._dropped_error_types.add(error_type)
        write_warning(
            EVENT_DROPPED_WARNING.format(
                error_type=error_type,
                error=describe_error(error),
            )
        )

    def _drain(self) -> None:
        """Write the queue to the spool, then deliver, within ``drain_seconds``.

        Runs once: from the worker on its way out, or from ``run`` when the
        worker stopped without draining.
        """

        with self._drain_lock:
            if self._drain_started:
                return
            self._drain_started = True
        deadline = time.monotonic() + self.drain_seconds
        unwritten = self.writer.finish(deadline)
        if unwritten:
            stopped_by = self.writer.stopped_by
            write_warning(
                UNWRITTEN_EVENTS_WARNING.format(
                    count=unwritten,
                    seconds=self.drain_seconds,
                    reason=(
                        describe_error(stopped_by)
                        if stopped_by is not None
                        and not isinstance(stopped_by, DrainDeadlineError)
                        else DRAIN_TIME_REASON
                    ),
                )
            )
        while time.monotonic() < deadline:
            progressed = False
            try:
                if not self.spool.has_deliverable():
                    return
                progressed = self.delivery.flush_once()
            except Exception as error:
                self._report(error)
            if not progressed:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(DRAIN_RETRY_SECONDS, remaining))
