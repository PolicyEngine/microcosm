"""Process and transport configuration independent of publishing components."""

from typing import Final

DEFAULT_DRAIN_SECONDS: Final = 15.0
SOCKET_LISTEN_BACKLOG: Final = 16
SOCKET_ACCEPT_TIMEOUT_SECONDS: Final = 0.5
SOCKET_CONNECTION_TIMEOUT_SECONDS: Final = 0.25
WORKER_INTERVAL_SECONDS: Final = 1.0
