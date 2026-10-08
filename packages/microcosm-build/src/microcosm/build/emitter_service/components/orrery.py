"""Durable completed-graph jobs; no telemetry registration or retention policy."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.build.emitter_service.components.graph_publication import (
    GRAPH_PUBLICATION_ACTION,
    GraphPublicationDelivery,
)
from microcosm.build.emitter_service.constants import WORKER_INTERVAL_SECONDS


class OrreryPublicationComponent:
    name = "orrery_publication"

    def __init__(self, delivery: GraphPublicationDelivery):
        self.delivery = delivery

    def start(self) -> None:
        pass

    def handle(self, message: Mapping[str, Any]) -> bool:
        if message.get("action") != GRAPH_PUBLICATION_ACTION:
            return False
        self.delivery.queue.enqueue(Path(message["directory"]), message["inventory"])
        return True

    def parent_exited(self) -> None:
        # Jobs and source bytes are already durable; keep them for retry.
        pass

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(WORKER_INTERVAL_SECONDS):
            self.delivery.flush_once()
