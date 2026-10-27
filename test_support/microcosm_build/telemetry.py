"""In-memory telemetry and synthetic graph requests for build tests.

``FakeTelemetryEmitter`` is Microcosm's real adapter over an in-memory
``EmitterClient`` (:mod:`test_support.microcosm_build.emitter`). Its
``events`` apply the rules the emitter's telemetry service documents for
turning requests into events, so tests can read run outcomes without the
emitter installed:

- a stage transition finishes the current transition stage, if it is a
  different one, and starts the new one;
- the first calibration epoch sent with ``transition`` moves to the
  ``calibrating`` stage;
- ``complete`` and ``block`` finish the current stage, ``fail`` fails it,
  and each then records the run's outcome;
- requests after the outcome make no events.

Events carry no timestamps and are not sanitized; the emitter's own tests
cover both.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm_emitter_interface.orrery import (
    PublicationFileRole,
    PublicationInventory,
    PublicationReceipt,
)

from microcosm.build.telemetry_emitter import LocalTelemetryEmitter, TelemetryRun
from test_support.microcosm_build.emitter import (
    FakeEmitterClient,
)
from test_support.microcosm_build.emitter import (
    fake_publication_inventory as _measured_inventory,
)

BUILD_COMPLETED_MESSAGE = "Microcosm build completed."
BUILD_BLOCKED_MESSAGE = "Validation checks refused the candidate at {phase}."
#: How the service reads a transition's status.
_TRANSITION_STATUSES = {
    "failed": "failed",
    "completed": "completed",
    "passed": "completed",
    "progress": "progress",
    "started": "started",
    "running": "started",
}


def fake_publication_inventory(
    directory: Path, roles: Mapping[str, PublicationFileRole]
) -> PublicationInventory:
    """Measure only synthetic test files without starting an emitter."""
    return _measured_inventory(directory, roles, uuid.uuid4().hex)


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


class _Lifecycle:
    """The emitter service's request-to-event rules, without its storage."""

    def __init__(self) -> None:
        self.transition_stage: str | None = None
        self.last_stage = "created"
        self.finished = False
        self.events: list[dict[str, Any]] = []

    def apply(self, request: Mapping[str, Any]) -> None:
        if self.finished:
            return
        command = request["command"]
        if command == "emit":
            self._add(
                request["event_type"],
                request["status"],
                request.get("stage_id"),
                request.get("message"),
                request.get("details") or {},
            )
        elif command == "stage":
            self._add(
                "stage",
                request.get("status", "started"),
                request["stage_id"],
                request.get("message"),
                request.get("details") or {},
            )
        elif command == "transition_stage":
            self._transition(request)
        elif command == "progress":
            self._add(
                "progress",
                "progress",
                request["stage_id"],
                details={
                    "done": request["done"],
                    "total": request["total"],
                    "unit": request.get("unit"),
                    **(request.get("details") or {}),
                },
            )
        elif command in {"calibration_progress", "transition_calibration_progress"}:
            event = request["event"]
            if event.get("kind") == "calibration_epoch":
                if (
                    command == "transition_calibration_progress"
                    and self.transition_stage != "calibrating"
                ):
                    self._transition({"stage_id": "calibrating"})
                self._add("calibration", "progress", "calibrating", details=event)
        elif command == "fail":
            details = {
                "error_type": request.get("error_type"),
                "failure_class": request.get("failure_class", "build_failure"),
                "failed_during": request.get("failed_during")
                or self.transition_stage
                or self.last_stage,
            }
            if request.get("error_code") is not None:
                details["error_code"] = request["error_code"]
            message = request.get("message")
            self._close_stage("failed", message, details)
            self._add("run", "failed", "failed", message, details)
        elif command == "block":
            check_ids = list(request["blocking_gate_ids"])
            count = request.get("blocking_failure_count")
            details: dict[str, Any] = {
                "phase": request["phase"],
                "blocking_gate_ids": check_ids,
                "blocking_failure_count": len(check_ids) if count is None else count,
            }
            if request.get("gate_statuses") is not None:
                details["gate_statuses"] = dict(request["gate_statuses"])
            self._close_stage()
            self._add(
                "run",
                "blocked",
                "blocked",
                BUILD_BLOCKED_MESSAGE.format(phase=details["phase"]),
                details,
            )
        elif command == "complete":
            self._close_stage()
            self._add("run", "completed", "complete", BUILD_COMPLETED_MESSAGE)
        else:
            raise ValueError(f"unsupported telemetry command {command!r}")

    def _add(
        self,
        event_type: str,
        status: str,
        stage_id: str | None = None,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "stage_id": stage_id,
                "status": status,
                "message": message or None,
                "details": dict(details or {}),
            }
        )
        if event_type == "run" and status in {"completed", "failed", "blocked"}:
            self.finished = True
            self.transition_stage = None
        elif stage_id is not None:
            self.last_stage = stage_id

    def _transition(self, request: Mapping[str, Any]) -> None:
        stage = request["stage_id"]
        status = _TRANSITION_STATUSES.get(request.get("status", "running"), "progress")
        if status == "started":
            if self.transition_stage == stage:
                status = "progress"
            else:
                self._close_stage()
                self.transition_stage = stage
        elif self.transition_stage == stage:
            if status in {"completed", "failed"}:
                self.transition_stage = None
        else:
            self._close_stage()
        self._add(
            "stage",
            status,
            stage,
            request.get("message"),
            request.get("details") or {},
        )

    def _close_stage(
        self,
        status: str = "completed",
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if self.transition_stage is not None:
            stage, self.transition_stage = self.transition_stage, None
            self._add("stage", status, stage, message, details)


def telemetry_events(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The events the emitter's telemetry service would make of ``messages``."""
    lifecycle = _Lifecycle()
    for message in messages:
        lifecycle.apply(message)
    return lifecycle.events


#: Makes the in-memory client of each ``FakeTelemetryEmitter`` built without
#: one; :func:`share_fake_hub` replaces it for one test.
client_factory: Callable[[], FakeEmitterClient] = FakeEmitterClient


def share_fake_hub(monkeypatch, hub: Any) -> None:
    """Commit the run files every fake emitter receives into ``hub.files``.

    ``hub`` is a fake Hub keyed by ``(repository, path)``; the build's other
    uploads (staged dataset bundles, releases) still go to it directly, so a
    test sees both in one place.
    """

    monkeypatch.setattr(
        sys.modules[__name__],
        "client_factory",
        lambda: FakeEmitterClient(hub_files=hub.files),
    )


class FakeTelemetryEmitter(LocalTelemetryEmitter):
    """Microcosm's adapter over an in-memory emitter client."""

    def __init__(
        self,
        run: TelemetryRun,
        *,
        close_service_on_finish: bool = True,
        client: FakeEmitterClient | None = None,
    ) -> None:
        super().__init__(
            run=run,
            client=client if client is not None else client_factory(),
            close_service_on_finish=close_service_on_finish,
        )

    @property
    def events(self) -> list[dict[str, Any]]:
        return telemetry_events(self.client.telemetry_messages)

    @property
    def messages(self) -> list[Mapping[str, Any]]:
        """The events as event messages, then the graph hand-overs."""
        return [
            *({"action": "event", "event": event} for event in self.events),
            *self.client.orrery_requests,
        ]
