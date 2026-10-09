"""In-memory telemetry and isolated opt-in service startup for tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun


class FakeTelemetryEmitter(LocalTelemetryEmitter):
    """Exercise real lifecycle methods without starting a service or socket."""

    def __init__(self, run: TelemetryRun) -> None:
        self.messages: list[Mapping[str, Any]] = []
        super().__init__(run=run, transport=self)

    def send(self, payload: Mapping[str, Any]) -> None:
        self.messages.append({"action": "event", "event": payload})

    def close(self) -> None:
        self._closed = True

    def publish_graph(
        self, directory: Path, inventory: dict, *, wait_seconds: float = 30.0
    ) -> dict:
        self.messages.append(
            {
                "action": "graph_publication",
                "directory": str(directory),
                "inventory": inventory,
            }
        )
        return {
            "version": 1,
            "publication_id": inventory["publication_id"],
            "status": "pending",
            "error_code": None,
        }

    @property
    def events(self) -> list[Mapping[str, Any]]:
        return [message["event"] for message in self.messages if "event" in message]
