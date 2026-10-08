"""Compatibility constructor for the telemetry-only service introduced in #1099.

New callers compose components through microcosm.build.emitter_service.
"""

from microcosm.build.emitter_service.components.orrery import OrreryPublicationComponent
from microcosm.build.emitter_service.components.telemetry import TelemetryComponent
from microcosm.build.emitter_service.runtime import EmitterService as PublishingService


class EmitterService(PublishingService):
    def __init__(
        self,
        *,
        socket_path,
        registration,
        spool,
        delivery,
        sampler,
        heartbeat_seconds,
        drain_seconds=15.0,
        graph_delivery=None,
    ):
        components = [
            TelemetryComponent(
                registration=registration,
                spool=spool,
                delivery=delivery,
                sampler=sampler,
                heartbeat_seconds=heartbeat_seconds,
                drain_seconds=drain_seconds,
            )
        ]
        if graph_delivery is not None:
            components.append(OrreryPublicationComponent(graph_delivery))
        super().__init__(
            socket_path=socket_path,
            components=components,
            parent_alive=sampler.parent_alive,
            shutdown_seconds=drain_seconds + 1.0,
        )
