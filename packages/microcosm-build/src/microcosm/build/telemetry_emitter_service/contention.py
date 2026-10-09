"""Bounded retry for lock contention on the shared telemetry spool."""

from __future__ import annotations

import random
import sqlite3
import time
from collections.abc import Callable

from sqlalchemy.exc import DBAPIError

from microcosm.build.telemetry_emitter_service.constants import (
    SPOOL_RETRY_INITIAL_SECONDS,
    SPOOL_RETRY_MAX_SECONDS,
)

# SQLite reports extended result codes; the low byte is the primary code.
_PRIMARY_RESULT_CODE_MASK = 0xFF
_TRANSIENT_RESULT_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})


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
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> T:
    """Run ``operation``, retrying spool lock contention until ``deadline``.

    ``deadline`` is a ``clock`` reading. The first attempt always runs. After a
    transient lock error the next attempt waits a jittered backoff that starts
    at ``SPOOL_RETRY_INITIAL_SECONDS`` and doubles up to
    ``SPOOL_RETRY_MAX_SECONDS``. When that attempt would not start before
    ``deadline``, the last lock error is raised instead. Any other error is
    raised at once. Each attempt may itself wait in SQLite's busy handler, so
    this bounds when attempts start, not when the last one ends.
    """

    delay = SPOOL_RETRY_INITIAL_SECONDS
    while True:
        try:
            return operation()
        except Exception as error:
            if not is_transient_spool_error(error):
                raise
            wait = jitter(delay / 2, delay)
            if clock() + wait >= deadline:
                raise
        sleep(wait)
        delay = min(SPOOL_RETRY_MAX_SECONDS, delay * 2)
