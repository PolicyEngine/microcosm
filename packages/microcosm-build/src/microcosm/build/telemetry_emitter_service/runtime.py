"""Existing local service with independent telemetry and graph workers."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DEFAULT_DRAIN_SECONDS,
    DRAIN_RETRY_SECONDS,
    EVENT_OBJECT_ERROR,
    FAILURE_CLASS_UNEXPECTED_PROCESS_EXIT,
    LOCAL_MESSAGE_TOO_LARGE_ERROR,
    MINIMUM_HEARTBEAT_SECONDS,
    SOCKET_ACCEPT_TIMEOUT_SECONDS,
    SOCKET_CONNECTION_TIMEOUT_SECONDS,
    SOCKET_LISTEN_BACKLOG,
    UNSUPPORTED_ACTION_ERROR,
    WORKER_INTERVAL_SECONDS,
)
from microcosm.build.telemetry_emitter_service.graph_publication import (
    GRAPH_PUBLICATION_ACTION,
    GraphPublicationDelivery,
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
    """Private local socket server with a concurrent delivery worker."""

    def __init__(
        self,
        *,
        socket_path: Path,
        registration: Mapping[str, Any] | None,
        spool: EventSpool,
        delivery: CollectorDelivery,
        sampler: ProcessTreeSampler,
        heartbeat_seconds: float,
        drain_seconds: float = DEFAULT_DRAIN_SECONDS,
        graph_delivery: GraphPublicationDelivery | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.registration = dict(registration) if registration is not None else None
        self.graph_delivery = graph_delivery
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

    def run(self) -> None:
        """Serve local messages until the client closes or exits."""

        if self.registration is not None:
            self.spool.register(self.registration)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink(missing_ok=True)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(self.socket_path))
                os.chmod(self.socket_path, 0o600)
                server.listen(SOCKET_LISTEN_BACKLOG)
                server.settimeout(SOCKET_ACCEPT_TIMEOUT_SECONDS)
                workers = []
                if self.registration is not None:
                    workers.append(threading.Thread(target=self._worker, daemon=True))
                if self.graph_delivery is not None:
                    workers.append(
                        threading.Thread(target=self._graph_worker, daemon=True)
                    )
                for worker in workers:
                    worker.start()
                self._serve(server)
                deadline = (
                    time.monotonic() + self.drain_seconds + WORKER_INTERVAL_SECONDS
                )
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
        finally:
            self._stop.set()
            self.socket_path.unlink(missing_ok=True)
            try:
                self.socket_path.parent.rmdir()
            except OSError:
                pass

    def _serve(self, server: socket.socket) -> None:
        while not self._stop.is_set():
            if not self.sampler.parent_alive():
                if self.registration is not None:
                    self.spool.append(
                        self.registration,
                        _unexpected_exit_event(self._last_stage),
                        resources=self.sampler.sample(),
                    )
                self._stop.set()
                break
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
        action = message.get("action")
        if action == GRAPH_PUBLICATION_ACTION and self.graph_delivery is not None:
            self.graph_delivery.queue.enqueue(
                Path(message["directory"]), message["inventory"]
            )
        elif action == ACTION_EVENT and self.registration is not None:
            event = message.get("event")
            if not isinstance(event, Mapping):
                raise ValueError(EVENT_OBJECT_ERROR)
            stage_id = event.get("stage_id")
            if isinstance(stage_id, str) and stage_id not in {
                STAGE_COMPLETE,
                STAGE_FAILED,
            }:
                self._last_stage = stage_id
            self.spool.append(
                self.registration,
                event,
                resources=self.sampler.sample(),
            )
        elif action == ACTION_CLOSE:
            self._stop.set()
        elif action == ACTION_PING:
            return
        else:
            raise ValueError(UNSUPPORTED_ACTION_ERROR)

    def _worker(self) -> None:
        next_heartbeat = time.monotonic() + self.heartbeat_seconds
        while not self._stop.wait(WORKER_INTERVAL_SECONDS):
            now = time.monotonic()
            # Sample every worker iteration so short-lived build children are
            # much less likely to disappear between stage and heartbeat events.
            self.sampler.sample()
            if now >= next_heartbeat:
                self.spool.append(
                    self.registration,
                    _heartbeat_event(self._last_stage),
                    resources=self.sampler.sample(),
                )
                next_heartbeat = now + self.heartbeat_seconds
            self.delivery.flush_once()
        self._drain()

    def _graph_worker(self) -> None:
        # Uploads may take much longer than a heartbeat. They never run on the
        # telemetry worker or the local socket thread. Pending jobs survive exit.
        while not self._stop.wait(WORKER_INTERVAL_SECONDS):
            self.graph_delivery.flush_once()

    def _drain(self) -> None:
        deadline = time.monotonic() + self.drain_seconds
        while self.spool.has_deliverable() and time.monotonic() < deadline:
            if not self.delivery.flush_once():
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(DRAIN_RETRY_SECONDS, remaining))
