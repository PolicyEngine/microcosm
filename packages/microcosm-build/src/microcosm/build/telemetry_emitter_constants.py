"""Configuration and user-facing messages for the telemetry emitter client."""

from __future__ import annotations

from pathlib import Path
from typing import Final

TELEMETRY_SERVICE_MODULE: Final = "microcosm.build.telemetry_emitter_service"
TELEMETRY_CACHE_PARTS: Final = ("microcosm", "telemetry")
TELEMETRY_SPOOL_FILENAME: Final = "events.sqlite3"
RUNTIME_DIRECTORY_PREFIX: Final = "microcosm-telemetry-"
SOCKET_FILENAME: Final = "emitter.sock"
TEMPORARY_DIRECTORY_ALIAS: Final = Path("/tmp")

DEFAULT_SEND_TIMEOUT_SECONDS: Final = 0.2
DEFAULT_HEARTBEAT_SECONDS: Final = 60.0
# start() returns as soon as the service answers its first ping or exits, so
# only a live service that never answers costs the whole wait, while giving up
# early discards hosted telemetry for the entire build. Startup is interpreter
# start plus the SQLAlchemy spool: 0.4-0.8 s on a build host at load average
# 80-160 on 2026-10-09, where single imports also stalled for up to 24 s.
DEFAULT_STARTUP_TIMEOUT_SECONDS: Final = 30.0
STARTUP_POLL_SECONDS: Final = 0.02
SERVICE_READY_TIMEOUT_SECONDS: Final = 0.1
SERVICE_PROCESS_EXIT_TIMEOUT_SECONDS: Final = 1.0

NO_LOCAL_SOCKET_WARNING: Final = (
    "warning: Microcosm telemetry is unavailable because this platform has no "
    "local Unix sockets."
)
SERVICE_START_WARNING: Final = (
    "warning: the local telemetry emitter service could not start: "
    "{error_type}: {error}"
)
SERVICE_EXITED_WARNING: Final = (
    "warning: the local telemetry emitter service exited with status "
    "{returncode} before it became ready; the build will continue without "
    "hosted telemetry."
)
SERVICE_NOT_READY_WARNING: Final = (
    "warning: the local telemetry emitter service did not become ready "
    "within {timeout_seconds:g} s; the build will continue without hosted "
    "telemetry."
)
QUEUE_WARNING: Final = (
    "warning: the local telemetry emitter service could not queue an update "
    "({error_type}); the build will continue."
)
INVALID_ACKNOWLEDGEMENT_ERROR: Final = "invalid acknowledgement"
