"""Durable local queue for telemetry events."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from microcosm.build.telemetry_emitter_service.constants import (
    BATCH_SIZE,
    DATABASE_TIMEOUT_SECONDS,
    LOCAL_ONLY_PRE_ELIGIBILITY,
    MAX_QUEUED_BYTES,
    PRUNE_INTERVAL_SECONDS,
    RETENTION_DAYS,
    UPLOAD_STATE_LOCAL_ONLY,
    UPLOAD_STATE_PENDING,
)
from microcosm.build.telemetry_emitter_service.timestamps import utc_now
from microcosm.build.telemetry_protocol import TELEMETRY_SCHEMA_VERSION

_DATABASE_SCHEMA: Final = f"""
CREATE TABLE IF NOT EXISTS telemetry_runs (
    run_id TEXT NOT NULL,
    producer_id TEXT NOT NULL,
    registration_json TEXT NOT NULL,
    next_sequence INTEGER NOT NULL DEFAULT 1,
    upload_state TEXT NOT NULL DEFAULT '{UPLOAD_STATE_PENDING}',
    local_only_reason TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(run_id, producer_id)
);
CREATE TABLE IF NOT EXISTS telemetry_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    producer_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, producer_id, sequence),
    FOREIGN KEY(run_id, producer_id)
        REFERENCES telemetry_runs(run_id, producer_id)
);
CREATE INDEX IF NOT EXISTS telemetry_events_run_sequence
    ON telemetry_events(run_id, producer_id, sequence);
"""


class EventSpool:
    """Small SQLite queue shared by successive emitter service processes."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self._connection = sqlite3.connect(
            self.path,
            timeout=DATABASE_TIMEOUT_SECONDS,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._last_prune_at = 0.0
        self._initialize_database()
        self.prune()

    def _initialize_database(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.executescript(_DATABASE_SCHEMA)
            columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(telemetry_runs)"
                ).fetchall()
            }
            if "upload_state" not in columns:
                self._connection.execute(
                    "ALTER TABLE telemetry_runs ADD COLUMN "
                    f"upload_state TEXT NOT NULL DEFAULT '{UPLOAD_STATE_PENDING}'"
                )
                self._connection.execute(
                    "UPDATE telemetry_runs SET upload_state = ?",
                    (UPLOAD_STATE_LOCAL_ONLY,),
                )
            if "local_only_reason" not in columns:
                self._connection.execute(
                    "ALTER TABLE telemetry_runs ADD COLUMN local_only_reason TEXT"
                )
                self._connection.execute(
                    "UPDATE telemetry_runs SET local_only_reason = ? "
                    "WHERE upload_state = ?",
                    (LOCAL_ONLY_PRE_ELIGIBILITY, UPLOAD_STATE_LOCAL_ONLY),
                )

    def register(self, registration: Mapping[str, Any]) -> None:
        """Create or refresh a producer registration."""

        run_id = str(registration["run_id"])
        producer_id = str(registration["producer_id"])
        encoded = json.dumps(registration, separators=(",", ":"), sort_keys=True)
        with self._lock, self._connection:
            existing = self._connection.execute(
                """
                SELECT registration_json FROM telemetry_runs
                WHERE run_id = ? AND producer_id = ?
                """,
                (run_id, producer_id),
            ).fetchone()
            if existing is not None:
                old = json.loads(existing["registration_json"])
                if old.get("producer_id") != registration.get("producer_id"):
                    raise ValueError(f"run_id {run_id!r} already has another producer")
            self._connection.execute(
                """
                INSERT INTO telemetry_runs (
                    run_id, producer_id, registration_json, next_sequence, updated_at
                ) VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(run_id, producer_id) DO UPDATE SET
                    registration_json = excluded.registration_json,
                    updated_at = excluded.updated_at
                """,
                (run_id, producer_id, encoded, utc_now()),
            )

    def append(
        self,
        registration: Mapping[str, Any],
        event: Mapping[str, Any],
        *,
        resources: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an event and assign its stable producer sequence."""

        run_id = str(registration["run_id"])
        producer_id = str(registration["producer_id"])
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT next_sequence FROM telemetry_runs
                WHERE run_id = ? AND producer_id = ?
                """,
                (run_id, producer_id),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            sequence = int(row["next_sequence"])
            event_id = uuid.uuid4().hex
            payload = {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "event_id": event_id,
                "run_id": run_id,
                "producer_id": producer_id,
                "sequence": sequence,
                "timestamp": event.get("timestamp") or utc_now(),
                "event_type": event["event_type"],
                "stage_id": event.get("stage_id"),
                "status": event["status"],
                "message": event.get("message"),
                "details": event.get("details") or {},
                "resources": resources,
            }
            encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
            self._connection.execute(
                """
                INSERT INTO telemetry_events (
                    event_id, run_id, producer_id, sequence,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, run_id, producer_id, sequence, encoded, utc_now()),
            )
            self._connection.execute(
                """
                UPDATE telemetry_runs
                SET next_sequence = ?, updated_at = ?
                WHERE run_id = ? AND producer_id = ?
                """,
                (sequence + 1, utc_now(), run_id, producer_id),
            )
        if time.monotonic() - self._last_prune_at >= PRUNE_INTERVAL_SECONDS:
            self.prune()
        return payload

    def pending_runs(self) -> list[dict[str, Any]]:
        """Return registrations that have events eligible for delivery."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT DISTINCT r.registration_json
                FROM telemetry_runs r
                JOIN telemetry_events e
                  ON e.run_id = r.run_id AND e.producer_id = r.producer_id
                WHERE r.upload_state = ?
                ORDER BY r.updated_at
                """,
                (UPLOAD_STATE_PENDING,),
            ).fetchall()
        return [json.loads(row["registration_json"]) for row in rows]

    def make_local_only(
        self,
        run_id: str,
        producer_id: str,
        reason: str,
    ) -> None:
        """Permanently exclude one producer's queued events from upload."""

        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE telemetry_runs
                SET upload_state = ?, local_only_reason = ?, updated_at = ?
                WHERE run_id = ? AND producer_id = ?
                """,
                (
                    UPLOAD_STATE_LOCAL_ONLY,
                    reason,
                    utc_now(),
                    run_id,
                    producer_id,
                ),
            )

    def has_deliverable(self) -> bool:
        """Return whether any queued event remains eligible for delivery."""

        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1
                FROM telemetry_events e
                JOIN telemetry_runs r
                  ON r.run_id = e.run_id AND r.producer_id = e.producer_id
                WHERE r.upload_state = ?
                LIMIT 1
                """,
                (UPLOAD_STATE_PENDING,),
            ).fetchone()
        return row is not None

    def batch(
        self,
        run_id: str,
        producer_id: str,
        limit: int = BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """Return the next ordered batch for one producer."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json FROM telemetry_events
                WHERE run_id = ? AND producer_id = ?
                ORDER BY sequence LIMIT ?
                """,
                (run_id, producer_id, limit),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def acknowledge(self, event_ids: list[str]) -> None:
        """Remove events acknowledged by the collector."""

        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        with self._lock, self._connection:
            self._connection.execute(
                f"DELETE FROM telemetry_events WHERE event_id IN ({placeholders})",
                event_ids,
            )

    def has_pending(self) -> bool:
        """Return whether any event remains in local storage."""

        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM telemetry_events LIMIT 1"
            ).fetchone()
        return row is not None

    def prune(self) -> None:
        """Enforce the age and total-size retention limits."""

        cutoff = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS)).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM telemetry_events WHERE created_at < ?",
                (cutoff,),
            )
            size_row = self._connection.execute(
                "SELECT COALESCE(SUM(LENGTH(payload_json)), 0) AS bytes "
                "FROM telemetry_events"
            ).fetchone()
            excess = int(size_row["bytes"]) - MAX_QUEUED_BYTES
            if excess > 0:
                self._remove_oldest_bytes(excess)
            self._connection.execute(
                """
                DELETE FROM telemetry_runs
                WHERE updated_at < ? AND NOT EXISTS (
                    SELECT 1 FROM telemetry_events e
                    WHERE e.run_id = telemetry_runs.run_id
                      AND e.producer_id = telemetry_runs.producer_id
                )
                """,
                (cutoff,),
            )
        self._last_prune_at = time.monotonic()

    def _remove_oldest_bytes(self, excess: int) -> None:
        rows = self._connection.execute(
            """
            SELECT event_id, LENGTH(payload_json) AS bytes
            FROM telemetry_events ORDER BY created_at, sequence
            """
        ).fetchall()
        removed = 0
        event_ids: list[str] = []
        for row in rows:
            event_ids.append(row["event_id"])
            removed += int(row["bytes"])
            if removed >= excess:
                break
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        self._connection.execute(
            f"DELETE FROM telemetry_events WHERE event_id IN ({placeholders})",
            event_ids,
        )
