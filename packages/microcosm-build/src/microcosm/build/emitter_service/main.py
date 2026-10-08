"""Compose telemetry and Orrery publishing in one extensible local service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microcosm.build.emitter_service.auth import (
    CollectorSession,
    _collector_origin,
    _development_collector_url,
)
from microcosm.build.emitter_service.components.graph_publication import (
    DEFAULT_GRAPH_PUBLICATION_ORIGIN,
    GraphPublicationDelivery,
    GraphPublicationQueue,
)
from microcosm.build.emitter_service.components.orrery import OrreryPublicationComponent
from microcosm.build.emitter_service.components.telemetry import TelemetryComponent
from microcosm.build.emitter_service.runtime import EmitterService
from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.constants import (
    DEFAULT_HEARTBEAT_SECONDS,
    PRODUCTION_COLLECTOR_URL,
)
from microcosm.build.telemetry_emitter_service.resources import ProcessTreeSampler
from microcosm.build.telemetry_emitter_service.spool import EventSpool


def build_parser() -> argparse.ArgumentParser:
    """Create the service command-line parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--development-collector-url")
    parser.add_argument(
        "--registration-json",
        help="Optional telemetry registration; omit for graph-only publishing.",
    )
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run all configured publishing components until the build disconnects."""

    args = build_parser().parse_args(argv)
    session = CollectorSession(
        _development_collector_url(args.development_collector_url)
        if args.development_collector_url
        else _collector_origin(PRODUCTION_COLLECTOR_URL)
    )
    sampler = ProcessTreeSampler(args.parent_pid)
    components = []
    if args.registration_json is not None:
        spool = EventSpool(args.spool)
        components.append(
            TelemetryComponent(
                registration=json.loads(args.registration_json),
                spool=spool,
                delivery=CollectorDelivery(
                    spool,
                    development_collector_url=args.development_collector_url,
                    session=session,
                ),
                sampler=sampler,
                heartbeat_seconds=args.heartbeat_seconds,
            )
        )
    components.append(
        OrreryPublicationComponent(
            GraphPublicationDelivery(
                GraphPublicationQueue(args.spool),
                credential=session.credential,
                invalidate_credential=session.invalidate,
                origin=DEFAULT_GRAPH_PUBLICATION_ORIGIN,
            )
        )
    )
    service = EmitterService(
        socket_path=args.socket,
        components=components,
        parent_alive=sampler.parent_alive,
    )
    service.run()
    return 0
