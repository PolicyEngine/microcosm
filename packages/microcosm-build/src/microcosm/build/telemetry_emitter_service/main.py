"""Command-line construction for the telemetry emitter service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DEFAULT_HEARTBEAT_SECONDS,
)
from microcosm.build.telemetry_emitter_service.graph_publication import (
    DEFAULT_GRAPH_PUBLICATION_ORIGIN,
    GraphPublicationDelivery,
    GraphPublicationQueue,
)
from microcosm.build.telemetry_emitter_service.resources import ProcessTreeSampler
from microcosm.build.telemetry_emitter_service.runtime import EmitterService
from microcosm.build.telemetry_emitter_service.spool import EventSpool


def build_parser() -> argparse.ArgumentParser:
    """Create the service command-line parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--development-collector-url")
    parser.add_argument("--registration-json", help="Omit for graph-only delivery.")
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
    registration = (
        json.loads(args.registration_json) if args.registration_json else None
    )
    spool = EventSpool(args.spool)
    delivery = CollectorDelivery(
        spool, development_collector_url=args.development_collector_url
    )
    service = EmitterService(
        socket_path=args.socket,
        registration=registration,
        spool=spool,
        delivery=delivery,
        graph_delivery=GraphPublicationDelivery(
            GraphPublicationQueue(args.spool),
            credential=delivery.session.credential,
            invalidate_credential=delivery.session.invalidate,
            origin=DEFAULT_GRAPH_PUBLICATION_ORIGIN,
        ),
        sampler=ProcessTreeSampler(args.parent_pid),
        heartbeat_seconds=args.heartbeat_seconds,
    )
    service.run()
    return 0
