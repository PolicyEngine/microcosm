"""One-line warnings on the build's stderr, which the service inherits."""

from __future__ import annotations

import sys

from sqlalchemy.exc import DBAPIError

from microcosm.build.telemetry_emitter_service.constants import (
    MAX_WARNING_ERROR_CHARS,
)


def describe_error(error: BaseException) -> str:
    """Return one short line describing an error, for a warning on stderr.

    SQLAlchemy's message repeats the SQL statement and its parameters, which
    can hold a registration or event payload, and a documentation link on later
    lines, so a wrapped driver error is described by the driver's own message.
    """

    if isinstance(error, DBAPIError) and error.orig is not None:
        error = error.orig
    lines = str(error).strip().splitlines()
    return (lines[0] if lines else type(error).__name__)[:MAX_WARNING_ERROR_CHARS]


def write_warning(message: str) -> None:
    """Print one line to stderr without ever raising.

    The service runs in its own session, so it can outlive a build whose
    stderr was a pipe that has since closed; a failed write must not stop it.
    """

    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass
