"""Command-line construction for the telemetry emitter service."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DEFAULT_HEARTBEAT_SECONDS,
    READY_DEADLINE_ERROR,
    READY_DEADLINE_MARGIN_SECONDS,
    SERVICE_FAILED_WARNING,
    SPOOL_LOCKED_EXIT_STATUS,
    SPOOL_LOCKED_WARNING,
)
from microcosm.build.telemetry_emitter_service.contention import (
    describe_error,
    is_transient_spool_error,
    retry_spool_contention,
)
from microcosm.build.telemetry_emitter_service.resources import ProcessTreeSampler
from microcosm.build.telemetry_emitter_service.runtime import EmitterService
from microcosm.build.telemetry_emitter_service.spool import EventSpool


def _finite_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds):
        raise argparse.ArgumentTypeError(READY_DEADLINE_ERROR)
    return seconds


def build_parser() -> argparse.ArgumentParser:
    """Create the service command-line parser."""

    parser = argparse.ArgumentParser()
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
        type=_finite_seconds,
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
) -> float:
    """Return the ``time.monotonic()`` reading at which startup stops retrying.

    The build passes its own give-up time as Unix time, the clock both
    processes share. The margin leaves time to bind the socket and answer the
    build's ping before the build gives up.
    """

    if ready_deadline is None:
        return -math.inf
    return now_monotonic + (ready_deadline - now_unix) - READY_DEADLINE_MARGIN_SECONDS


def main(argv: list[str] | None = None) -> int:
    """Run the telemetry emitter service until the build disconnects.

    Its stderr is the build's, so a failure is one line, never a traceback.
    Exit status 75 means another process kept the spool locked past the
    startup deadline; 1 means any other failure.
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
        spool = retry_spool_contention(
            lambda: EventSpool(args.spool),
            deadline=deadline,
        )
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
            startup_deadline=deadline,
        )
        service.run()
    except Exception as error:
        if is_transient_spool_error(error):
            print(
                SPOOL_LOCKED_WARNING.format(
                    spool=args.spool,
                    waited_seconds=time.monotonic() - started_at,
                    error=describe_error(error),
                ),
                file=sys.stderr,
                flush=True,
            )
            return SPOOL_LOCKED_EXIT_STATUS
        print(
            SERVICE_FAILED_WARNING.format(
                error_type=type(error).__name__,
                error=describe_error(error),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0
