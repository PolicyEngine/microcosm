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
DEFAULT_STARTUP_TIMEOUT_SECONDS: Final = 3.0
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
SERVICE_NOT_READY_WARNING: Final = (
    "warning: the local telemetry emitter service did not become ready; "
    "the build will continue without hosted telemetry."
)
QUEUE_WARNING: Final = (
    "warning: the local telemetry emitter service could not queue an update "
    "({error_type}); the build will continue."
)
INVALID_ACKNOWLEDGEMENT_ERROR: Final = "invalid acknowledgement"
