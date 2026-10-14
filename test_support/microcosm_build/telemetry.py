"""In-memory telemetry and synthetic graph requests for build tests."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm_emitter.orrery.contracts import (
    PublicationFileRole,
    PublicationInventory,
    PublicationReceipt,
)
from microcosm_emitter.telemetry.service.lifecycle import (
    LifecycleState,
    process_request,
)

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun


def fake_publication_inventory(
    directory: Path, roles: Mapping[str, PublicationFileRole]
) -> PublicationInventory:
    """Measure only synthetic test files without starting an emitter service."""
    return {
        "version": 1,
        "publication_id": uuid.uuid4().hex,
        "files": [
            {
                "name": name,
                "role": role,
                "bytes": (directory / name).stat().st_size,
                "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
            }
            for name, role in sorted(roles.items())
        ],
    }


def fake_publish_graph(
    directory: Path,
    inventory: PublicationInventory,
    *,
    wait_seconds: float = 30.0,
) -> PublicationReceipt:
    """Return an in-memory receipt without persistence, credentials or HTTP."""
    return {
        "version": 1,
        "publication_id": inventory["publication_id"],
        "status": "pending",
        "error_code": None,
    }


class FakeTelemetryEmitter(LocalTelemetryEmitter):
    """Exercise real lifecycle methods without starting a service or socket."""

    def __init__(
        self, run: TelemetryRun, *, close_service_on_finish: bool = True
    ) -> None:
        self.messages: list[Mapping[str, Any]] = []
        self._lifecycle = LifecycleState()
        super().__init__(
            run=run,
            transport=self,
            close_service_on_finish=close_service_on_finish,
        )

    def send(self, payload: Mapping[str, Any]) -> None:
        self._lifecycle, events = process_request(self._lifecycle, payload)
        self.messages.extend({"action": "event", "event": event} for event in events)

    def close(self) -> None:
        self._closed = True

    def publication_inventory(
        self, directory: Path, roles: Mapping[str, PublicationFileRole]
    ) -> PublicationInventory:
        return fake_publication_inventory(directory, roles)

    def publish_graph(
        self,
        directory: Path,
        inventory: PublicationInventory,
        *,
        wait_seconds: float = 30.0,
    ) -> PublicationReceipt:
        self.messages.append(
            {
                "action": "graph_publication",
                "directory": str(directory),
                "inventory": inventory,
            }
        )
        return fake_publish_graph(directory, inventory, wait_seconds=wait_seconds)

    @property
    def events(self) -> list[Mapping[str, Any]]:
        return [message["event"] for message in self.messages if "event" in message]
