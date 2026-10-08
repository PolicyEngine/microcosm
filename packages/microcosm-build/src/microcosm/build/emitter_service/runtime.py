"""Component-neutral Unix-socket transport and worker lifecycle."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from microcosm.build.emitter_protocol import (
    ACTION_CLOSE,
    ACTION_PING,
    LOCAL_ACKNOWLEDGEMENT_ERROR,
    LOCAL_ACKNOWLEDGEMENT_OK,
    LOCAL_MESSAGE_DELIMITER,
    LOCAL_SOCKET_READ_BYTES,
    MAX_LOCAL_MESSAGE_BYTES,
)

from .constants import (
    DEFAULT_DRAIN_SECONDS,
    SOCKET_ACCEPT_TIMEOUT_SECONDS,
    SOCKET_CONNECTION_TIMEOUT_SECONDS,
    SOCKET_LISTEN_BACKLOG,
    WORKER_INTERVAL_SECONDS,
)


class EmitterComponent(Protocol):
    """Add a publisher by implementing this interface, not by changing runtime."""

    name: str

    def start(self) -> None: ...
    def handle(self, message: Mapping[str, Any]) -> bool: ...
    def run(self, stop: threading.Event) -> None: ...
    def parent_exited(self) -> None: ...


class EmitterService:
    """One private socket and independently scheduled publishing components."""

    def __init__(
        self,
        *,
        socket_path: Path,
        components: Sequence[EmitterComponent],
        parent_alive: Callable[[], bool],
        shutdown_seconds: float = DEFAULT_DRAIN_SECONDS,
    ):
        self.socket_path = socket_path
        self.components = tuple(components)
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("Emitter component names must be unique.")
        self.parent_alive = parent_alive
        self.shutdown_seconds = max(0, shutdown_seconds)
        self._stop = threading.Event()

    def run(self) -> None:
        for component in self.components:
            component.start()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        workers = []
        try:
            self.socket_path.unlink(missing_ok=True)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(self.socket_path))
                os.chmod(self.socket_path, 0o600)
                server.listen(SOCKET_LISTEN_BACKLOG)
                server.settimeout(SOCKET_ACCEPT_TIMEOUT_SECONDS)
                for component in self.components:
                    worker = threading.Thread(
                        target=self._run_component,
                        args=(component,),
                        name=component.name,
                        daemon=True,
                    )
                    workers.append(worker)
                    worker.start()
                self._serve(server)
        finally:
            self._stop.set()
            # A slow component must not delay another component's shutdown.
            deadline = time.monotonic() + self.shutdown_seconds
            for worker in workers:
                worker.join(timeout=max(0, deadline - time.monotonic()))
            self.socket_path.unlink(missing_ok=True)
            try:
                self.socket_path.parent.rmdir()
            except OSError:
                pass

    def _run_component(self, component: EmitterComponent) -> None:
        while not self._stop.is_set():
            try:
                component.run(self._stop)
                return
            except Exception:
                # Durable jobs remain retryable; one worker's failure must not
                # stop the socket server or other publishing components.
                self._stop.wait(WORKER_INTERVAL_SECONDS)

    def _serve(self, server: socket.socket) -> None:
        while not self._stop.is_set():
            if not self.parent_alive():
                for component in self.components:
                    try:
                        component.parent_exited()
                    except Exception:
                        pass
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
            self._handle(self._read_message(connection))
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
            raise ValueError("Local emitter message exceeds size limit.")
        value = json.loads(bytes(chunks).split(LOCAL_MESSAGE_DELIMITER, 1)[0])
        if not isinstance(value, dict):
            raise ValueError("Local emitter message must be an object.")
        return value

    def _handle(self, message: Mapping[str, Any]) -> None:
        if message.get("action") == ACTION_CLOSE:
            self._stop.set()
        elif message.get("action") == ACTION_PING:
            return
        else:
            for component in self.components:
                if component.handle(message):
                    return
            raise ValueError("Unsupported emitter action.")
