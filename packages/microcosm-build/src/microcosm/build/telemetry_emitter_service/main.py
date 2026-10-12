"""Command-line construction for the telemetry emitter service."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import NoReturn

from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DEFAULT_HEARTBEAT_SECONDS,
    READY_DEADLINE_ERROR,
    READY_DEADLINE_MARGIN_SECONDS,
    SERVICE_ARGUMENTS_WARNING,
    SERVICE_FAILED_WARNING,
    SPOOL_LOCKED_EXIT_STATUS,
    SPOOL_LOCKED_WARNING,
    STARTUP_RETRY_LIMIT_SECONDS,
)
from microcosm.build.telemetry_emitter_service.contention import (
    is_transient_spool_error,
    retry_spool_contention,
)
from microcosm.build.telemetry_emitter_service.diagnostics import (
    describe_error,
    write_warning,
)
from microcosm.build.telemetry_emitter_service.resources import ProcessTreeSampler
from microcosm.build.telemetry_emitter_service.runtime import EmitterService
from microcosm.build.telemetry_emitter_service.spool import EventSpool


def _seconds(value: str) -> float:
    seconds = float(value)
    if math.isnan(seconds):
        raise argparse.ArgumentTypeError(READY_DEADLINE_ERROR)
    return seconds


class _ArgumentParser(argparse.ArgumentParser):
    """Report invalid arguments in one line, like every other startup failure."""

    def error(self, message: str) -> NoReturn:
        write_warning(SERVICE_ARGUMENTS_WARNING.format(error=message))
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    """Create the service command-line parser."""

    parser = _ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--development-collector-url")
    parser.add_argument("--registration-json", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
    )
    parser.add_argument(
        "--ready-deadline",
        type=_seconds,
        help=(
            "Unix time at which the build stops waiting for readiness. Until "
            "shortly before then, startup retries spool lock contention; "
            "without it, startup makes one attempt."
        ),
    )
    return parser


def startup_deadline(
    ready_deadline: float | None,
    *,
    now_unix: float,
    now_monotonic: float,
) -> float | None:
    """Return the ``time.monotonic()`` reading at which startup gives up.

    The build passes its own give-up time as Unix time, the clock both
    processes share. The margin leaves time to bind the socket and answer the
    build's ping before the build gives up. The limit bounds the wait however
    far the deadline or the wall clock moves, so a service whose build has died
    still gives up. Without a deadline from the build, there is none here.
    """

    if ready_deadline is None:
        return None
    remaining = min(ready_deadline - now_unix, STARTUP_RETRY_LIMIT_SECONDS)
    return now_monotonic + remaining - READY_DEADLINE_MARGIN_SECONDS


def open_registered_spool(
    path: Path,
    registration: dict[str, object],
    *,
    deadline: float | None,
) -> EventSpool:
    """Open the spool and register this producer, waiting out lock contention.

    With a ``deadline``, startup waits for another process's lock until then:
    each SQLite wait ends by it, and an attempt that fails on the lock is
    retried while one can still start before it. Without one, a single attempt
    waits as long as any statement does, ``DATABASE_TIMEOUT_SECONDS``. The spool
    returned waits normally.
    """

    opened: list[EventSpool] = []

    def attempt() -> EventSpool:
        if not opened:
            opened.append(EventSpool(path, busy_deadline=deadline))
        opened[0].register(registration)
        return opened[0]

    spool = retry_spool_contention(
        attempt,
        deadline=-math.inf if deadline is None else deadline,
    )
    spool.busy_deadline = None
    return spool


def main(argv: list[str] | None = None) -> int:
    """Run the telemetry emitter service until the build disconnects.

    The socket is bound only once this build is registered, so a build that
    finds the service ready can queue events. The service's stderr is the
    build's, so a failure is one line, never a traceback. Exit status 75 means
    another process kept the spool locked past the startup deadline, 2 that
    the arguments were invalid, and 1 any other failure.
    """

    args = build_parser().parse_args(argv)
    started_at = time.monotonic()
    deadline = startup_deadline(
        args.ready_deadline,
        now_unix=time.time(),
        now_monotonic=started_at,
    )
    try:
        registration = json.loads(args.registration_json)
        spool = open_registered_spool(args.spool, registration, deadline=deadline)
        service = EmitterService(
            socket_path=args.socket,
            registration=registration,
            spool=spool,
            delivery=CollectorDelivery(
                spool,
                development_collector_url=args.development_collector_url,
            ),
            sampler=ProcessTreeSampler(args.parent_pid),
            heartbeat_seconds=args.heartbeat_seconds,
        )
        service.run()
    except Exception as error:
        if is_transient_spool_error(error):
            write_warning(
                SPOOL_LOCKED_WARNING.format(
                    spool=args.spool,
                    waited_seconds=time.monotonic() - started_at,
                    error=describe_error(error),
                )
            )
            return SPOOL_LOCKED_EXIT_STATUS
        write_warning(
            SERVICE_FAILED_WARNING.format(
                error_type=type(error).__name__,
                error=describe_error(error),
            )
        )
        return 1
    return 0
