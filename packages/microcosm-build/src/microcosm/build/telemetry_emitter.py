"""Microcosm metadata adapter for the independently packaged local provider."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from microcosm_provider_client.launcher import start_services
from microcosm_provider_orrery.client import default_spool_path, publish_graph
from microcosm_provider_orrery.contracts import PublicationInventory, PublicationReceipt
from microcosm_provider_telemetry.client import (
    LocalTelemetryEmitter as ProviderTelemetryEmitter,
)
from microcosm_provider_telemetry.client import (
    TelemetryRun,
)
from microcosm_provider_telemetry.constants import (
    DEFAULT_HEARTBEAT_SECONDS,
    SERVICE_START_WARNING,
)

from microcosm.build.telemetry_identity import runtime_identity


class LocalTelemetryEmitter(ProviderTelemetryEmitter):
    """Supply build identity and select both workers in one provider process."""

    def __init__(self, *, run, transport, handle=None, spool_path=None):
        super().__init__(run=run, transport=transport, handle=handle)
        self._spool_path = Path(spool_path) if spool_path else None

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
        startup_timeout_seconds: float = 3.0,
        spool_path: Path | str | None = None,
    ) -> LocalTelemetryEmitter:
        run = TelemetryRun(
            run_id=run_id,
            country_code=country_code,
            pipeline=pipeline,
            candidate_id=candidate_id,
            release_id=release_id,
            run_kind=run_kind,
        )
        path = None
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
            return cls(run=run, transport=None, spool_path=path)
        emitter = cls(
            run=run,
            transport=handle.client.for_module("telemetry"),
            handle=handle,
            spool_path=path,
        )
        emitter.emit(
            event_type="run",
            stage_id="created",
            status="started",
            message="Build started",
            details={"identity": identity},
        )
        return emitter

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
            spool_path=self._spool_path,
            available=lambda: self.available,
            wait_seconds=wait_seconds,
        )


def start_local_telemetry_emitter_service(**run: Any) -> LocalTelemetryEmitter:
    """Construct the build adapter without importing provider service internals."""
    return LocalTelemetryEmitter.start(**run)
