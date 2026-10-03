"""Best-effort client for the local telemetry emitter service.

The build process only writes small messages to a private local socket.  A
separate process owns retry, authentication, resource sampling, and network
I/O so collector availability cannot delay or fail a dataset build.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from platform import platform
from typing import Any, Literal
from urllib.parse import urlsplit

DEFAULT_COLLECTOR_URL = "https://microcosm-telemetry-389282473430.us-central1.run.app"
COLLECTOR_URL_ENV = "MICROCOSM_TELEMETRY_COLLECTOR_URL"
_SENSITIVE_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
    "traceback",
)
_SECRET_TEXT = re.compile(
    r"(?i)(?:bearer\s+[^\s]+|(?:token|secret|password|credential)\s*[:=]\s*[^\s]+|hf_[A-Za-z0-9_-]{8,})"
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _cache_dir() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return root / "microcosm" / "telemetry"


def _collector_url(value: str) -> str:
    parsed = urlsplit(value)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not parsed.hostname or (
        parsed.scheme != "https" and not (local and parsed.scheme == "http")
    ):
        raise ValueError("collector URL must use HTTPS except on localhost")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("collector URL must not contain credentials or query data")
    return value.rstrip("/")


def _safe_text(value: str, *, limit: int = 2_000) -> str:
    return _SECRET_TEXT.sub("[redacted]", value)[:limit]


def _safe_json(value: Any, *, depth: int = 0) -> Any:
    """Return JSON-compatible telemetry without retaining model objects."""

    if depth >= 6:
        return "[maximum depth]"
    if isinstance(value, Mapping):
        result = {}
        for key, item in list(value.items())[:200]:
            name = str(key)
            if any(part in name.lower() for part in _SENSITIVE_KEY_PARTS):
                result[name] = "[redacted]"
            else:
                result[name] = _safe_json(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_json(item, depth=depth + 1) for item in value[:200]]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, (int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _safe_json(item())
    return _safe_text(str(value))


def _safe_details(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = _safe_json(value)
    assert isinstance(sanitized, dict)
    if len(json.dumps(sanitized, separators=(",", ":")).encode()) <= 8_192:
        return sanitized
    compact: dict[str, Any] = {"telemetry_details_truncated": True}
    for key, item in sanitized.items():
        if not (isinstance(item, (str, int, float, bool)) or item is None):
            continue
        candidate = {**compact, key: item}
        if len(json.dumps(candidate, separators=(",", ":")).encode()) > 8_192:
            break
        compact[key] = item
    return compact


def _run_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        memory_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
    except (AttributeError, OSError, ValueError):
        memory_bytes = None
    versions = {}
    for distribution in (
        "microcosm-build",
        "microcosm-graph",
        "policyengine-us",
        "policyengine-uk",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return {
        "git_commit": commit,
        "host": {
            "platform": platform(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": memory_bytes,
        },
        "runtime": versions,
    }


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
        send_timeout_seconds: float = 0.2,
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
        collector_url: str | None = None,
        heartbeat_seconds: float = 60.0,
        startup_timeout_seconds: float = 3.0,
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
        raw_url = (
            collector_url
            or os.environ.get(COLLECTOR_URL_ENV, "").strip()
            or DEFAULT_COLLECTOR_URL
        )
        try:
            url = _collector_url(raw_url)
        except ValueError as error:
            print(
                f"warning: Microcosm telemetry is unavailable: {error}.",
                file=sys.stderr,
            )
            return cls(run=run, process=None, socket_path=None, runtime_dir=None)
        if not hasattr(socket, "AF_UNIX"):
            print(
                "warning: Microcosm telemetry is unavailable because this "
                "platform has no local Unix sockets.",
                file=sys.stderr,
            )
            return cls(run=run, process=None, socket_path=None, runtime_dir=None)

        # macOS limits AF_UNIX paths to roughly 100 bytes.  Its default
        # temporary directory is already long, so use the short system alias.
        temporary_root = Path("/tmp") if Path("/tmp").is_dir() else None
        runtime_dir = Path(
            tempfile.mkdtemp(prefix="microcosm-telemetry-", dir=temporary_root)
        )
        runtime_dir.chmod(0o700)
        socket_path = runtime_dir / "emitter.sock"
        queue_path = Path(spool_path) if spool_path else _cache_dir() / "events.sqlite3"
        command = [
            sys.executable,
            "-m",
            "microcosm.build.telemetry_emitter_service",
            "--socket",
            str(socket_path),
            "--spool",
            str(queue_path),
            "--collector-url",
            url,
            "--registration-json",
            json.dumps(run.as_registration(), separators=(",", ":")),
            "--parent-pid",
            str(os.getpid()),
            "--heartbeat-seconds",
            str(heartbeat_seconds),
        ]
        try:
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
                        event_type="run",
                        stage_id="created",
                        status="started",
                        message="Microcosm build started.",
                        details={"identity": _run_identity()},
                    )
                    return emitter
                if process.poll() is not None:
                    break
                time.sleep(0.02)
        except Exception as error:
            print(
                "warning: the local telemetry emitter service could not start: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
        else:
            print(
                "warning: the local telemetry emitter service did not become ready; "
                "the build will continue without hosted telemetry.",
                file=sys.stderr,
            )
        try:
            socket_path.unlink(missing_ok=True)
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
        event_type: Literal["run", "stage", "progress", "calibration", "heartbeat"],
        status: Literal["started", "progress", "completed", "failed"],
        stage_id: str | None = None,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Queue one event locally, never raising into the build."""

        if not self.available:
            return
        self._send(
            {
                "action": "event",
                "event": {
                    "timestamp": _now(),
                    "event_type": event_type,
                    "stage_id": stage_id,
                    "status": status,
                    "message": _safe_text(message, limit=500) if message else None,
                    "details": _safe_details(details or {}),
                },
            }
        )

    def stage(
        self,
        stage_id: str,
        *,
        status: Literal["started", "progress", "completed", "failed"] = "started",
        message: str | None = None,
        **details: Any,
    ) -> None:
        self.emit(
            event_type="stage",
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

        collector_status = {
            "failed": "failed",
            "passed": "completed",
            "running": "started",
        }.get(status, "progress")
        if collector_status == "started":
            if self._transition_stage == stage_id:
                self.stage(
                    stage_id,
                    status="progress",
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
            event_type="progress",
            stage_id=stage_id,
            status="progress",
            details={"done": done, "total": total, "unit": unit, **details},
        )

    def calibration_progress(self, event: Mapping[str, Any]) -> None:
        if event.get("kind") != "calibration_epoch":
            return
        self.emit(
            event_type="calibration",
            stage_id="calibrating",
            status="progress",
            details=event,
        )

    def transition_calibration_progress(self, event: Mapping[str, Any]) -> None:
        """Enter the sequential calibration stage, then report one epoch."""

        if event.get("kind") != "calibration_epoch":
            return
        if self._transition_stage != "calibrating":
            self.transition_stage("calibrating")
        self.calibration_progress(event)

    def fail(
        self,
        error: BaseException,
        *,
        failed_during: str | None = None,
        failure_class: str = "build_failure",
    ) -> None:
        failed_stage = failed_during or self._transition_stage
        self._close_transition_stage()
        self.emit(
            event_type="run",
            stage_id="failed",
            status="failed",
            message=str(error)[:500],
            details={
                "error_type": type(error).__name__,
                "failure_class": failure_class,
                "failed_during": failed_stage,
            },
        )
        self.close()

    def complete(self) -> None:
        self._close_transition_stage()
        self.emit(
            event_type="run",
            stage_id="complete",
            status="completed",
            message="Microcosm build completed.",
        )
        self.close()

    def close(self) -> None:
        """Ask the service to flush in the background, then release the handle."""

        if self._closed:
            return
        self._send({"action": "close"})
        self._closed = True

    def _close_transition_stage(self) -> None:
        if self._transition_stage is None:
            return
        stage_id = self._transition_stage
        self._transition_stage = None
        self.stage(stage_id, status="completed")

    def _send(self, payload: Mapping[str, Any]) -> None:
        if self._socket_path is None:
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self._send_timeout_seconds)
                client.connect(str(self._socket_path))
                client.sendall(
                    json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
                    + b"\n"
                )
                acknowledgement = client.recv(16)
                if acknowledgement != b"ok\n":
                    raise OSError("invalid acknowledgement")
        except Exception as error:
            if not self._warned:
                print(
                    "warning: the local telemetry emitter service could not queue "
                    f"an update ({type(error).__name__}); the build will continue.",
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
            client.settimeout(0.1)
            client.connect(str(socket_path))
            client.sendall(b'{"action":"ping"}\n')
            return client.recv(16) == b"ok\n"
    except OSError:
        return False
