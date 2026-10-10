"""Turn SIGTERM into an exception the build's own failure path records.

A supervisor, a budget stop or ``kill`` ends a process with SIGTERM, which by
default kills it without running any ``finally`` or ``except``: the staging run
stays ``running`` and no Logbook row is written. :func:`raise_on_sigterm`
raises :class:`BuildTerminatedError` instead, so the run closes as a failed run
with ``TERMINATED`` and the attempt is recorded (discarded, like an operator's
Ctrl-C).

``BuildTerminatedError`` subclasses :class:`KeyboardInterrupt`, not
:class:`Exception`: the existing interrupt arms record it, and graph kernels'
``except Exception`` handlers cannot swallow it. Python reports the
signal-style exit status only for an exact ``KeyboardInterrupt``, so a command
re-raises it as ``SystemExit(BuildTerminatedError.exit_code)`` (143) after its
close-out, and supervisors still see a termination rather than a crash.

SIGKILL and an out-of-memory kill cannot be caught; the hosted emitter's
heartbeat and its ``unexpected_process_exit`` event cover those.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Iterator

__all__ = ["BuildTerminatedError", "raise_on_sigterm"]


class BuildTerminatedError(KeyboardInterrupt):
    """SIGTERM ended the build (a supervisor, a budget stop, ``kill``)."""

    exit_code = 128 + int(signal.SIGTERM)

    def __str__(self) -> str:
        return "The build was terminated by SIGTERM."


@contextlib.contextmanager
def raise_on_sigterm() -> Iterator[None]:
    """Raise :class:`BuildTerminatedError` on the first SIGTERM inside the block.

    Off the main thread (where Python cannot install handlers) it does nothing.
    The handler fires once and restores the default, so a second SIGTERM kills
    the process at once: a supervisor's escalation still works, even while the
    graph store settles after the first. The previous handler is restored when
    the block exits.
    """

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handle(signum, frame):  # noqa: ARG001 - the signal handler signature
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        raise BuildTerminatedError()

    try:
        previous = signal.signal(signal.SIGTERM, handle)
    except ValueError:  # pragma: no cover - not the main interpreter thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(
            signal.SIGTERM, previous if previous is not None else signal.SIG_DFL
        )
