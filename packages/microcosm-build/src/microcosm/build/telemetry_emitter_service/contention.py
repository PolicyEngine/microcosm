"""Bounded retry for lock contention on the shared telemetry spool."""

from __future__ import annotations

import random
import sqlite3
import time
from collections.abc import Callable

from sqlalchemy.exc import DBAPIError

from microcosm.build.telemetry_emitter_service.constants import (
    SPOOL_BUSY_ERROR,
    SPOOL_RETRY_INITIAL_SECONDS,
    SPOOL_RETRY_MAX_SECONDS,
)

# SQLite reports extended result codes; the low byte is the primary code.
_PRIMARY_RESULT_CODE_MASK = 0xFF
_TRANSIENT_RESULT_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})


class SpoolBusyError(sqlite3.OperationalError):
    """Another thread of this process kept the spool past the caller's wait.

    It carries SQLite's busy code, so it is retried like a lock another
    process holds.
    """

    def __init__(self) -> None:
        super().__init__(SPOOL_BUSY_ERROR)
        self.sqlite_errorcode = sqlite3.SQLITE_BUSY
        self.sqlite_errorname = "SQLITE_BUSY"


def is_transient_spool_error(error: BaseException) -> bool:
    """Return whether an error is SQLite lock contention that a retry can clear.

    That is SQLITE_BUSY ("database is locked") or SQLITE_LOCKED, raised
    directly by ``sqlite3`` or wrapped by SQLAlchemy. Schema, constraint and
    I/O errors are not transient.
    """

    if isinstance(error, DBAPIError):
        error = error.orig
    if not isinstance(error, sqlite3.Error):
        return False
    code = getattr(error, "sqlite_errorcode", None)
    return (
        isinstance(code, int)
        and code & _PRIMARY_RESULT_CODE_MASK in _TRANSIENT_RESULT_CODES
    )


def retry_spool_contention[T](
    operation: Callable[[], T],
    *,
    deadline: float | Callable[[], float],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> T:
    """Run ``operation``, retrying spool lock contention until ``deadline``.

    ``deadline`` is a ``clock`` reading, or a function returning one. A
    function is read again after every lock error and after every wait, so its
    caller can bring the deadline forward, from infinity say, while this
    waits. The first attempt always runs. After a transient lock error the
    next attempt waits a jittered backoff that starts at
    ``SPOOL_RETRY_INITIAL_SECONDS`` and doubles up to
    ``SPOOL_RETRY_MAX_SECONDS``. When that attempt would not start before the
    deadline, or the wait overran it, the last lock error is raised instead.
    Any other error is raised at once. Each attempt may itself wait in
    SQLite's busy handler, so this bounds when attempts start, not when the
    last one ends.
    """

    current_deadline = deadline if callable(deadline) else lambda: deadline
    delay = SPOOL_RETRY_INITIAL_SECONDS
    while True:
        try:
            return operation()
        except Exception as error:
            if not is_transient_spool_error(error):
                raise
            wait = jitter(delay / 2, delay)
            if clock() + wait >= current_deadline():
                raise
            contended = error
        sleep(wait)
        if clock() >= current_deadline():
            # The sleep overran, as it can on a loaded host, or the deadline
            # moved forward during it.
            raise contended
        delay = min(SPOOL_RETRY_MAX_SECONDS, delay * 2)
