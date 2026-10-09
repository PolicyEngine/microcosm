"""Durable local queue for telemetry events."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, delete, func, select

from microcosm.build.telemetry_emitter_service.constants import (
    BATCH_SIZE,
    DATABASE_TIMEOUT_SECONDS,
    MAX_QUEUED_BYTES,
    PRUNE_BATCH_ROWS,
    PRUNE_INTERVAL_SECONDS,
    RETENTION_DAYS,
    UPLOAD_STATE_LOCAL_ONLY,
    UPLOAD_STATE_PENDING,
)
from microcosm.build.telemetry_emitter_service.database import (
    create_spool_engine,
    create_spool_session_factory,
)
from microcosm.build.telemetry_emitter_service.migrations import (
    upgrade_spool_database,
)
from microcosm.build.telemetry_emitter_service.models import (
    TelemetryEventRecord,
    TelemetryRunRecord,
)
from microcosm.build.telemetry_emitter_service.timestamps import utc_now
from microcosm.build.telemetry_protocol import TELEMETRY_SCHEMA_VERSION


class EventSpool:
    """Small SQLite queue shared by successive emitter service processes."""

    def __init__(
        self,
        path: Path | str,
        *,
        busy_timeout_seconds: float = DATABASE_TIMEOUT_SECONDS,
        script_location: Traversable | None = None,
    ) -> None:
        """Open the spool, migrating it to this checkout's schema if behind.

        ``script_location`` replaces the packaged migration history, so tests
        can open one spool as checkouts at different versions.
        """

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        # How long one statement waits for another process's lock. Startup
        # lowers it so its own retry loop decides when to give up; it applies
        # from each connection's next checkout.
        self.busy_timeout_seconds = busy_timeout_seconds
        self._engine = create_spool_engine(
            self.path,
            busy_timeout_seconds=lambda: self.busy_timeout_seconds,
        )
        try:
            upgrade_spool_database(
                self._engine,
                busy_timeout_seconds=busy_timeout_seconds,
                script_location=script_location,
            )
        except BaseException:
            self._engine.dispose()
            raise
        self._session_factory = create_spool_session_factory(self._engine)
        self._lock = threading.RLock()
        # Retention is enforced by the service's delivery worker through
        # prune_if_due, not here: pruning needs the write lock whenever a row
        # has expired, and this constructor runs before the build's readiness
        # ping is answered.
        self._last_prune_at: float | None = None

    def register(self, registration: Mapping[str, Any]) -> None:
        """Create or refresh a producer registration."""

        run_id = str(registration["run_id"])
        producer_id = str(registration["producer_id"])
        with self._lock, self._session_factory.begin() as session:
            existing = session.get(TelemetryRunRecord, (run_id, producer_id))
            if existing is not None:
                if existing.registration.get("producer_id") != registration.get(
                    "producer_id"
                ):
                    raise ValueError(f"run_id {run_id!r} already has another producer")
                existing.registration = dict(registration)
                existing.updated_at = utc_now()
                return
            session.add(
                TelemetryRunRecord(
                    run_id=run_id,
                    producer_id=producer_id,
                    registration=dict(registration),
                    next_sequence=1,
                    upload_state=UPLOAD_STATE_PENDING,
                    local_only_reason=None,
                    updated_at=utc_now(),
                )
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
        with self._lock, self._session_factory.begin() as session:
            run = session.get(TelemetryRunRecord, (run_id, producer_id))
            if run is None:
                raise KeyError(run_id)
            sequence = run.next_sequence
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
            session.add(
                TelemetryEventRecord(
                    event_id=event_id,
                    run_id=run_id,
                    producer_id=producer_id,
                    sequence=sequence,
                    payload=payload,
                    created_at=utc_now(),
                )
            )
            run.next_sequence = sequence + 1
            run.updated_at = utc_now()
        return payload

    def pending_runs(self) -> list[dict[str, Any]]:
        """Return registrations that have events eligible for delivery."""

        statement = (
            select(TelemetryRunRecord)
            .join(TelemetryRunRecord.events)
            .where(TelemetryRunRecord.upload_state == UPLOAD_STATE_PENDING)
            .order_by(TelemetryRunRecord.updated_at)
            .distinct()
        )
        with self._lock, self._session_factory() as session:
            runs = session.scalars(statement).all()
            return [dict(run.registration) for run in runs]

    def make_local_only(
        self,
        run_id: str,
        producer_id: str,
        reason: str,
    ) -> None:
        """Permanently exclude one producer's queued events from upload."""

        with self._lock, self._session_factory.begin() as session:
            run = session.get(TelemetryRunRecord, (run_id, producer_id))
            if run is None:
                return
            run.upload_state = UPLOAD_STATE_LOCAL_ONLY
            run.local_only_reason = reason
            run.updated_at = utc_now()

    def has_deliverable(self) -> bool:
        """Return whether any queued event remains eligible for delivery."""

        statement = (
            select(TelemetryEventRecord)
            .join(TelemetryEventRecord.run)
            .where(TelemetryRunRecord.upload_state == UPLOAD_STATE_PENDING)
            .limit(1)
        )
        with self._lock, self._session_factory() as session:
            return session.scalar(statement) is not None

    def batch(
        self,
        run_id: str,
        producer_id: str,
        limit: int = BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """Return the next ordered batch for one producer."""

        statement = (
            select(TelemetryEventRecord)
            .where(
                TelemetryEventRecord.run_id == run_id,
                TelemetryEventRecord.producer_id == producer_id,
            )
            .order_by(TelemetryEventRecord.sequence)
            .limit(limit)
        )
        with self._lock, self._session_factory() as session:
            events = session.scalars(statement).all()
            return [dict(event.payload) for event in events]

    def acknowledge(self, event_ids: list[str]) -> None:
        """Remove events acknowledged by the collector."""

        if not event_ids:
            return
        statement = select(TelemetryEventRecord).where(
            TelemetryEventRecord.event_id.in_(event_ids)
        )
        with self._lock, self._session_factory.begin() as session:
            for event in session.scalars(statement):
                session.delete(event)

    def has_pending(self) -> bool:
        """Return whether any event remains in local storage."""

        statement = select(TelemetryEventRecord).limit(1)
        with self._lock, self._session_factory() as session:
            return session.scalar(statement) is not None

    def prune_if_due(self) -> None:
        """Prune unless an attempt began less than ``PRUNE_INTERVAL_SECONDS`` ago.

        The first call always prunes. A failed attempt still counts, so a spool
        that another process keeps locked is not rescanned every worker tick.
        """

        now = time.monotonic()
        if (
            self._last_prune_at is not None
            and now - self._last_prune_at < PRUNE_INTERVAL_SECONDS
        ):
            return
        self._last_prune_at = now
        self.prune()

    def prune(self) -> None:
        """Enforce the age and total-size retention limits.

        Rows go in batches of ``PRUNE_BATCH_ROWS``, each its own short
        transaction, so a large backlog never holds the spool's write lock, or
        the lock this process's event path shares, for long.
        """

        cutoff = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS)).isoformat()
        expired = (
            select(TelemetryEventRecord.event_id)
            .where(TelemetryEventRecord.created_at < cutoff)
            .limit(PRUNE_BATCH_ROWS)
        )
        while self._delete_events(expired) == PRUNE_BATCH_ROWS:
            pass
        oldest_first = self._oldest_events_over_size_limit()
        for start in range(0, len(oldest_first), PRUNE_BATCH_ROWS):
            self._delete_events(oldest_first[start : start + PRUNE_BATCH_ROWS])
        with self._lock, self._session_factory.begin() as session:
            session.execute(
                delete(TelemetryRunRecord)
                .where(
                    TelemetryRunRecord.updated_at < cutoff,
                    ~TelemetryRunRecord.events.any(),
                )
                .execution_options(synchronize_session=False)
            )

    def _delete_events(self, event_ids) -> int:
        """Delete the selected or listed events in one transaction."""

        with self._lock, self._session_factory.begin() as session:
            return session.execute(
                delete(TelemetryEventRecord)
                .where(TelemetryEventRecord.event_id.in_(event_ids))
                .execution_options(synchronize_session=False)
            ).rowcount

    def _oldest_events_over_size_limit(self) -> list[str]:
        """Return the oldest events whose removal brings storage under the cap."""

        stored = func.length(TelemetryEventRecord.payload, type_=Integer)
        with self._lock, self._session_factory() as session:
            excess = (
                session.scalar(select(func.coalesce(func.sum(stored), 0)))
                - MAX_QUEUED_BYTES
            )
            doomed: list[str] = []
            if excess <= 0:
                return doomed
            oldest = select(TelemetryEventRecord.event_id, stored).order_by(
                TelemetryEventRecord.created_at,
                TelemetryEventRecord.sequence,
                TelemetryEventRecord.event_id,
            )
            for event_id, length in session.execute(oldest):
                doomed.append(event_id)
                excess -= length
                if excess <= 0:
                    break
            return doomed
