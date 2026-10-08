"""Best-effort telemetry: event queue, sampling, heartbeat, and bounded drain."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from microcosm.build.telemetry_emitter_service.constants import (
    DRAIN_RETRY_SECONDS,
    FAILURE_CLASS_UNEXPECTED_PROCESS_EXIT,
    MINIMUM_HEARTBEAT_SECONDS,
    WORKER_INTERVAL_SECONDS,
)
from microcosm.build.telemetry_emitter_service.timestamps import utc_now
from microcosm.build.telemetry_protocol import (
    ACTION_EVENT,
    EVENT_TYPE_HEARTBEAT,
    EVENT_TYPE_RUN,
    STAGE_COMPLETE,
    STAGE_CREATED,
    STAGE_FAILED,
    STATUS_FAILED,
    STATUS_PROGRESS,
    UNEXPECTED_PROCESS_EXIT_MESSAGE,
)


class TelemetryComponent:
    name = "telemetry"

    def __init__(
        self,
        *,
        registration,
        spool,
        delivery,
        sampler,
        heartbeat_seconds: float,
        drain_seconds: float = 15.0,
    ):
        self.registration = dict(registration)
        self.spool, self.delivery, self.sampler = spool, delivery, sampler
        self.heartbeat_seconds = max(MINIMUM_HEARTBEAT_SECONDS, heartbeat_seconds)
        self.drain_seconds = max(0, drain_seconds)
        self._last_stage = STAGE_CREATED

    def start(self) -> None:
        self.spool.register(self.registration)

    def handle(self, message: Mapping[str, Any]) -> bool:
        if message.get("action") != ACTION_EVENT:
            return False
        event = message.get("event")
        if not isinstance(event, Mapping):
            raise ValueError("Telemetry event must be an object.")
        stage_id = event.get("stage_id")
        if isinstance(stage_id, str) and stage_id not in {STAGE_COMPLETE, STAGE_FAILED}:
            self._last_stage = stage_id
        self.spool.append(self.registration, event, resources=self.sampler.sample())
        return True

    def parent_exited(self) -> None:
        self.spool.append(
            self.registration,
            {
                "timestamp": utc_now(),
                "event_type": EVENT_TYPE_RUN,
                "stage_id": STAGE_FAILED,
                "status": STATUS_FAILED,
                "message": UNEXPECTED_PROCESS_EXIT_MESSAGE,
                "details": {
                    "failure_class": FAILURE_CLASS_UNEXPECTED_PROCESS_EXIT,
                    "failed_during": self._last_stage,
                },
            },
            resources=self.sampler.sample(),
        )

    def run(self, stop: threading.Event) -> None:
        next_heartbeat = time.monotonic() + self.heartbeat_seconds
        while not stop.wait(WORKER_INTERVAL_SECONDS):
            self.sampler.sample()
            now = time.monotonic()
            if now >= next_heartbeat:
                self.spool.append(
                    self.registration,
                    {
                        "timestamp": utc_now(),
                        "event_type": EVENT_TYPE_HEARTBEAT,
                        "stage_id": self._last_stage,
                        "status": STATUS_PROGRESS,
                        "message": None,
                        "details": {},
                    },
                    resources=self.sampler.sample(),
                )
                next_heartbeat = now + self.heartbeat_seconds
            self.delivery.flush_once()
        deadline = time.monotonic() + self.drain_seconds
        while self.spool.has_deliverable() and time.monotonic() < deadline:
            if not self.delivery.flush_once():
                time.sleep(DRAIN_RETRY_SECONDS)
