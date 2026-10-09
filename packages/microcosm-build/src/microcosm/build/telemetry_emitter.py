"""Best-effort client for the local telemetry emitter service.

The build process only writes small messages to a private local socket.  A
separate process owns retry, authentication, resource sampling, and network
I/O so collector availability cannot delay or fail a dataset build.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microcosm.build.telemetry_emitter_constants import (
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_SEND_TIMEOUT_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    INVALID_ACKNOWLEDGEMENT_ERROR,
    NO_LOCAL_SOCKET_WARNING,
    QUEUE_WARNING,
    RUNTIME_DIRECTORY_PREFIX,
    SERVICE_NOT_READY_WARNING,
    SERVICE_PROCESS_EXIT_TIMEOUT_SECONDS,
    SERVICE_READY_TIMEOUT_SECONDS,
    SERVICE_START_WARNING,
    SOCKET_FILENAME,
    STARTUP_POLL_SECONDS,
    TELEMETRY_CACHE_PARTS,
    TELEMETRY_SERVICE_MODULE,
    TELEMETRY_SPOOL_FILENAME,
    TEMPORARY_DIRECTORY_ALIAS,
)
from microcosm.build.telemetry_identity import runtime_identity
from microcosm.build.telemetry_protocol import (
    ACTION_CLOSE,
    ACTION_EVENT,
    BUILD_COMPLETED_MESSAGE,
    BUILD_STARTED_MESSAGE,
    CALIBRATION_EVENT_KIND,
    EVENT_TYPE_CALIBRATION,
    EVENT_TYPE_PROGRESS,
    EVENT_TYPE_RUN,
    EVENT_TYPE_STAGE,
    LOCAL_ACKNOWLEDGEMENT_OK,
    LOCAL_ACKNOWLEDGEMENT_READ_BYTES,
    LOCAL_MESSAGE_DELIMITER,
    LOCAL_PING_MESSAGE,
    MAX_TELEMETRY_MESSAGE_CHARS,
    SEQUENTIAL_STATUS_MAP,
    STAGE_CALIBRATING,
    STAGE_COMPLETE,
    STAGE_CREATED,
    STAGE_FAILED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROGRESS,
    STATUS_STARTED,
    TelemetryEventType,
    TelemetryStatus,
)
from microcosm.build.telemetry_sanitization import (
    sanitize_details,
    sanitize_text,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _cache_dir() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return root.joinpath(*TELEMETRY_CACHE_PARTS)


@dataclass(frozen=True)
class TelemetryRun:
    """Identity registered with the hosted collector for one build."""

    run_id: str
    country_code: str
    pipeline: str
    candidate_id: str | None = None
    release_id: str | None = None
    run_kind: str = "build"
    producer_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def as_registration(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "producer_id": self.producer_id,
            "country_code": self.country_code,
            "pipeline": self.pipeline,
            "candidate_id": self.candidate_id,
            "release_id": self.release_id,
            "run_kind": self.run_kind,
        }


class LocalTelemetryEmitter:
    """Non-blocking handle owned by the instrumented build process."""

    def __init__(
        self,
        *,
        run: TelemetryRun,
        process: subprocess.Popen[bytes] | None,
        socket_path: Path | None,
        runtime_dir: Path | None,
        send_timeout_seconds: float = DEFAULT_SEND_TIMEOUT_SECONDS,
    ) -> None:
        self.run = run
        self._process = process
        self._socket_path = socket_path
        self._runtime_dir = runtime_dir
        self._send_timeout_seconds = send_timeout_seconds
        self._closed = False
        self._warned = False
        self._transition_stage: str | None = None

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        country_code: str,
        pipeline: str,
        candidate_id: str | None = None,
        release_id: str | None = None,
        run_kind: str = "build",
        development_collector_url: str | None = None,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        spool_path: Path | str | None = None,
    ) -> LocalTelemetryEmitter:
        """Start the service, returning a harmless disabled handle on failure."""

        run = TelemetryRun(
            run_id=run_id,
            country_code=country_code,
            pipeline=pipeline,
            candidate_id=candidate_id,
            release_id=release_id,
            run_kind=run_kind,
        )
        if not hasattr(socket, "AF_UNIX"):
            print(NO_LOCAL_SOCKET_WARNING, file=sys.stderr)
            return cls(run=run, process=None, socket_path=None, runtime_dir=None)

        runtime_dir: Path | None = None
        socket_path: Path | None = None
        process: subprocess.Popen[bytes] | None = None
        try:
            # macOS limits AF_UNIX paths to roughly 100 bytes. Its default
            # temporary directory is already long, so use the short system alias.
            temporary_root = (
                TEMPORARY_DIRECTORY_ALIAS
                if TEMPORARY_DIRECTORY_ALIAS.is_dir()
                else None
            )
            runtime_dir = Path(
                tempfile.mkdtemp(prefix=RUNTIME_DIRECTORY_PREFIX, dir=temporary_root)
            )
            runtime_dir.chmod(0o700)
            socket_path = runtime_dir / SOCKET_FILENAME
            queue_path = (
                Path(spool_path)
                if spool_path
                else _cache_dir() / TELEMETRY_SPOOL_FILENAME
            )
            command = [
                sys.executable,
                "-m",
                TELEMETRY_SERVICE_MODULE,
                "--socket",
                str(socket_path),
                "--spool",
                str(queue_path),
                "--registration-json",
                json.dumps(run.as_registration(), separators=(",", ":")),
                "--parent-pid",
                str(os.getpid()),
                "--heartbeat-seconds",
                str(heartbeat_seconds),
                # The service retries spool lock contention until just before
                # this wall-clock time, when the wait below gives up.
                "--ready-deadline",
                repr(datetime.now(UTC).timestamp() + startup_timeout_seconds),
            ]
            if development_collector_url is not None:
                command.extend(
                    ["--development-collector-url", development_collector_url]
                )
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                start_new_session=True,
            )
            deadline = time.monotonic() + startup_timeout_seconds
            while time.monotonic() < deadline:
                if socket_path.exists() and _service_ready(socket_path):
                    emitter = cls(
                        run=run,
                        process=process,
                        socket_path=socket_path,
                        runtime_dir=runtime_dir,
                    )
                    emitter.emit(
                        event_type=EVENT_TYPE_RUN,
                        stage_id=STAGE_CREATED,
                        status=STATUS_STARTED,
                        message=BUILD_STARTED_MESSAGE,
                        details={"identity": runtime_identity()},
                    )
                    return emitter
                if process.poll() is not None:
                    break
                time.sleep(STARTUP_POLL_SECONDS)
        except Exception as error:
            print(
                SERVICE_START_WARNING.format(
                    error_type=type(error).__name__,
                    error=error,
                ),
                file=sys.stderr,
            )
        else:
            print(SERVICE_NOT_READY_WARNING, file=sys.stderr)
        if process is not None:
            _terminate_process(process)
        try:
            if socket_path is not None:
                socket_path.unlink(missing_ok=True)
            if runtime_dir is not None:
                runtime_dir.rmdir()
        except OSError:
            pass
        return cls(run=run, process=None, socket_path=None, runtime_dir=runtime_dir)

    @property
    def available(self) -> bool:
        return self._socket_path is not None and not self._closed

    def emit(
        self,
        *,
        event_type: TelemetryEventType,
        status: TelemetryStatus,
        stage_id: str | None = None,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Queue one event with the local service, never raising into the build.

        The service acknowledges the event once it is queued in the service's
        memory, behind every event acknowledged before it, without waiting for
        the shared spool; see ``_send``.
        """

        if not self.available:
            return
        self._send(
            {
                "action": ACTION_EVENT,
                "event": {
                    "timestamp": _now(),
                    "event_type": event_type,
                    "stage_id": stage_id,
                    "status": status,
                    "message": (
                        sanitize_text(message, limit=MAX_TELEMETRY_MESSAGE_CHARS)
                        if message
                        else None
                    ),
                    "details": sanitize_details(details or {}),
                },
            }
        )

    def stage(
        self,
        stage_id: str,
        *,
        status: TelemetryStatus = STATUS_STARTED,
        message: str | None = None,
        **details: Any,
    ) -> None:
        self.emit(
            event_type=EVENT_TYPE_STAGE,
            stage_id=stage_id,
            status=status,
            message=message,
            details=details,
        )

    def transition_stage(
        self,
        stage_id: str,
        *,
        status: str = "running",
        message: str | None = None,
        **details: Any,
    ) -> None:
        """Translate a sequential stage update into explicit lifecycle events."""

        collector_status = SEQUENTIAL_STATUS_MAP.get(status, STATUS_PROGRESS)
        if collector_status == STATUS_STARTED:
            if self._transition_stage == stage_id:
                self.stage(
                    stage_id,
                    status=STATUS_PROGRESS,
                    message=message,
                    **details,
                )
                return
            self._close_transition_stage()
            self._transition_stage = stage_id
        elif self._transition_stage == stage_id:
            self._transition_stage = None
        else:
            self._close_transition_stage()
        self.stage(
            stage_id,
            status=collector_status,
            message=message,
            **details,
        )

    def progress(
        self,
        stage_id: str,
        *,
        done: int,
        total: int,
        unit: str | None = None,
        **details: Any,
    ) -> None:
        self.emit(
            event_type=EVENT_TYPE_PROGRESS,
            stage_id=stage_id,
            status=STATUS_PROGRESS,
            details={"done": done, "total": total, "unit": unit, **details},
        )

    def calibration_progress(self, event: Mapping[str, Any]) -> None:
        if event.get("kind") != CALIBRATION_EVENT_KIND:
            return
        self.emit(
            event_type=EVENT_TYPE_CALIBRATION,
            stage_id=STAGE_CALIBRATING,
            status=STATUS_PROGRESS,
            details=event,
        )

    def transition_calibration_progress(self, event: Mapping[str, Any]) -> None:
        """Enter the sequential calibration stage, then report one epoch."""

        if event.get("kind") != CALIBRATION_EVENT_KIND:
            return
        if self._transition_stage != STAGE_CALIBRATING:
            self.transition_stage(STAGE_CALIBRATING)
        self.calibration_progress(event)

    def fail(
        self,
        error: BaseException,
        *,
        failed_during: str | None = None,
        failure_class: str = "build_failure",
    ) -> None:
        failed_stage = failed_during or self._transition_stage
        message = str(error)[:MAX_TELEMETRY_MESSAGE_CHARS]
        details = {
            "error_type": type(error).__name__,
            "failure_class": failure_class,
            "failed_during": failed_stage,
        }
        self._close_transition_stage(status=STATUS_FAILED, message=message, **details)
        self.emit(
            event_type=EVENT_TYPE_RUN,
            stage_id=STAGE_FAILED,
            status=STATUS_FAILED,
            message=message,
            details=details,
        )
        self.close()

    def complete(self) -> None:
        self._close_transition_stage()
        self.emit(
            event_type=EVENT_TYPE_RUN,
            stage_id=STAGE_COMPLETE,
            status=STATUS_COMPLETED,
            message=BUILD_COMPLETED_MESSAGE,
        )
        self.close()

    def close(self) -> None:
        """Ask the service to flush in the background, then release the handle."""

        if self._closed:
            return
        self._send({"action": ACTION_CLOSE})
        self._closed = True

    def _close_transition_stage(
        self,
        *,
        status: TelemetryStatus = STATUS_COMPLETED,
        message: str | None = None,
        **details: Any,
    ) -> None:
        if self._transition_stage is None:
            return
        stage_id = self._transition_stage
        self._transition_stage = None
        self.stage(stage_id, status=status, message=message, **details)

    def _send(self, payload: Mapping[str, Any]) -> None:
        """Send one message and wait at most the send timeout for its reply.

        ``ok`` to an event means the service has queued it in memory behind
        the events acknowledged before it; a writer thread then appends it to
        the spool in that order, retrying while another build holds the
        spool's lock. The reply never waits for the spool, so a locked spool
        does not delay the build. ``error``, or no reply within the timeout,
        means the event may not be queued: the first such failure prints one
        warning, and the build continues either way.
        """

        if self._socket_path is None:
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self._send_timeout_seconds)
                client.connect(str(self._socket_path))
                client.sendall(
                    json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
                    + LOCAL_MESSAGE_DELIMITER
                )
                acknowledgement = client.recv(LOCAL_ACKNOWLEDGEMENT_READ_BYTES)
                if acknowledgement != LOCAL_ACKNOWLEDGEMENT_OK:
                    raise OSError(INVALID_ACKNOWLEDGEMENT_ERROR)
        except Exception as error:
            if not self._warned:
                print(
                    QUEUE_WARNING.format(error_type=type(error).__name__),
                    file=sys.stderr,
                )
                self._warned = True


def start_local_telemetry_emitter_service(
    **run: Any,
) -> LocalTelemetryEmitter:
    """Named construction seam for build entrypoints and tests."""

    return LocalTelemetryEmitter.start(**run)


def _service_ready(socket_path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(SERVICE_READY_TIMEOUT_SECONDS)
            client.connect(str(socket_path))
            client.sendall(LOCAL_PING_MESSAGE)
            return (
                client.recv(LOCAL_ACKNOWLEDGEMENT_READ_BYTES)
                == LOCAL_ACKNOWLEDGEMENT_OK
            )
    except OSError:
        return False


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate and reap an emitter service that did not become usable."""

    try:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=SERVICE_PROCESS_EXIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=SERVICE_PROCESS_EXIT_TIMEOUT_SECONDS)
    except OSError:
        pass
