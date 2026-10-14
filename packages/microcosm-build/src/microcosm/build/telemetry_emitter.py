"""Microcosm metadata adapter for the independently packaged local emitter."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm_emitter.host.launcher import start_services
from microcosm_emitter.orrery.client import (
    GraphPublicationClient,
    default_spool_path,
    publish_graph,
)
from microcosm_emitter.orrery.contracts import (
    PublicationFileRole,
    PublicationInventory,
    PublicationReceipt,
)
from microcosm_emitter.telemetry.client import (
    LocalTelemetryEmitter as EmitterTelemetryClient,
)
from microcosm_emitter.telemetry.client import (
    TelemetryRun,
)
from microcosm_emitter.telemetry.constants import (
    DEFAULT_HEARTBEAT_SECONDS,
    SERVICE_START_WARNING,
)

from microcosm.build.telemetry_identity import runtime_identity


class LocalTelemetryEmitter(EmitterTelemetryClient):
    """Supply build identity and select both workers in one emitter process."""

    def __init__(
        self,
        *,
        run,
        transport,
        handle=None,
        graph_client=None,
        close_service_on_finish: bool = True,
    ):
        super().__init__(
            run=run,
            transport=transport,
            handle=handle,
            close_service_on_finish=close_service_on_finish,
        )
        self._graph_client = (
            graph_client if graph_client is not None else GraphPublicationClient()
        )

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        country_code: str,
        pipeline: str,
        candidate_id: str | None = None,
        release_id: str | None = None,
        run_kind: str = "build",
        development_collector_url: str | None = None,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        # The service took about four seconds to become ready on an M1 Max,
        # past the provider's three-second default; this is only a ceiling.
        startup_timeout_seconds: float = 15.0,
        spool_path: Path | str | None = None,
        close_service_on_finish: bool = True,
    ) -> LocalTelemetryEmitter:
        run = TelemetryRun(
            run_id=run_id,
            country_code=country_code,
            pipeline=pipeline,
            candidate_id=candidate_id,
            release_id=release_id,
            run_kind=run_kind,
        )
        try:
            path = Path(spool_path) if spool_path else default_spool_path()
            identity = runtime_identity()
            shared = {
                "spool_path": str(path),
                "development_collector_url": development_collector_url,
            }
            handle = start_services(
                {
                    "telemetry": {
                        **shared,
                        "registration": run.as_registration(),
                        "identity": identity,
                        "heartbeat_seconds": heartbeat_seconds,
                    },
                    "orrery": shared,
                },
                startup_timeout=startup_timeout_seconds,
            )
        except Exception as error:
            print(
                SERVICE_START_WARNING.format(error_type=type(error).__name__),
                file=sys.stderr,
            )
            return cls(
                run=run,
                transport=None,
                close_service_on_finish=close_service_on_finish,
            )
        return cls(
            run=run,
            transport=handle.client.for_module("telemetry"),
            handle=handle,
            graph_client=GraphPublicationClient(handle.client.for_module("orrery")),
            close_service_on_finish=close_service_on_finish,
        )

    def publication_inventory(
        self, directory: Path, roles: Mapping[str, PublicationFileRole]
    ) -> PublicationInventory | None:
        """Ask the existing graph worker to measure selected public files."""
        return self._graph_client.inventory(directory, roles)

    def publish_graph(
        self,
        directory: Path,
        inventory: PublicationInventory,
        *,
        wait_seconds: float = 30.0,
    ) -> PublicationReceipt:
        return publish_graph(
            directory,
            inventory,
            client=self._graph_client,
            wait_seconds=wait_seconds,
        )


def start_local_telemetry_emitter_service(**run: Any) -> LocalTelemetryEmitter:
    """Construct the build adapter without importing emitter service internals."""
    return LocalTelemetryEmitter.start(**run)
