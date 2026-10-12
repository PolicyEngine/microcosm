"""In-memory telemetry and isolated opt-in service startup for tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun


class FakeTelemetryEmitter(LocalTelemetryEmitter):
    """Exercise real lifecycle methods without starting a service or socket."""

    def __init__(self, run: TelemetryRun) -> None:
        super().__init__(
            run=run,
            process=None,
            socket_path=Path("/unused-test-telemetry-socket"),
            runtime_dir=None,
        )
        self.messages: list[Mapping[str, Any]] = []

    def _send(self, payload: Mapping[str, Any]) -> None:
        self.messages.append(payload)

    @property
    def events(self) -> list[Mapping[str, Any]]:
        return [message["event"] for message in self.messages if "event" in message]
