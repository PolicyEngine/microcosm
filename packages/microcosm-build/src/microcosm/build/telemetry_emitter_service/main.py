"""Command-line construction for the telemetry emitter service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DEFAULT_HEARTBEAT_SECONDS,
    OWN_LEASE_UNAVAILABLE_MESSAGE,
)
from microcosm.build.telemetry_emitter_service.leases import OwnLeaseUnavailableError
from microcosm.build.telemetry_emitter_service.resources import ProcessTreeSampler
from microcosm.build.telemetry_emitter_service.runtime import EmitterService
from microcosm.build.telemetry_emitter_service.spool import EventSpool


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the telemetry emitter service until the build disconnects."""

    args = build_parser().parse_args(argv)
    registration = json.loads(args.registration_json)
    spool = EventSpool(args.spool)
    # The delivery takes this producer's lease, which must happen before the
    # service is ready: from then on the run can have events. Without the
    # lease the service does not start, and the build goes on without it.
    try:
        delivery = CollectorDelivery(
            spool,
            registration,
            development_collector_url=args.development_collector_url,
        )
    except OwnLeaseUnavailableError:
        print(OWN_LEASE_UNAVAILABLE_MESSAGE, file=sys.stderr, flush=True)
        return 1
    service = EmitterService(
        socket_path=args.socket,
        registration=registration,
        spool=spool,
        delivery=delivery,
        sampler=ProcessTreeSampler(args.parent_pid),
        heartbeat_seconds=args.heartbeat_seconds,
    )
    try:
        service.run()
    finally:
        delivery.close()
    return 0
