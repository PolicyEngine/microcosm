"""Durable local queue for telemetry events."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, delete, func, select, tuple_
from sqlalchemy.orm import Session

from microcosm.build.telemetry_emitter_service.constants import (
    BATCH_SIZE,
    DATABASE_TIMEOUT_SECONDS,
    MAX_QUEUED_BYTES,
    PRUNE_BATCH_PAUSE_SECONDS,
    PRUNE_BATCH_ROWS,
    PRUNE_INTERVAL_SECONDS,
    PRUNE_STEP_SECONDS,
    RETENTION_DAYS,
    UPLOAD_STATE_LOCAL_ONLY,
    UPLOAD_STATE_PENDING,
)
from microcosm.build.telemetry_emitter_service.contention import SpoolBusyError
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


class _LockWaitBudget:
    """Caps the time one transaction's statements together wait for locks.

    SQLite's busy timeout applies to each lock a statement needs, so a
    transaction that waits for the write lock and then again at COMMIT can
    wait twice as long as the timeout. ``apply``, called before each statement
    that can wait, sets the timeout to what is left of ``seconds``. The next
    checkout of the connection restores the spool's own timeout.
    """

    def __init__(self, session: Session, seconds: float | None) -> None:
        self._session = session
        self._deadline = None if seconds is None else time.monotonic() + seconds

    def apply(self) -> None:
        if self._deadline is None:
            return
        milliseconds = max(0, round((self._deadline - time.monotonic()) * 1000))
        self._session.connection().exec_driver_sql(
            f"PRAGMA busy_timeout = {milliseconds}"
        )


class EventSpool:
    """Small SQLite queue shared by successive emitter service processes."""

    def __init__(
        self,
        path: Path | str,
        *,
        busy_deadline: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        # A time.monotonic() reading that startup sets: until it is cleared,
        # no statement waits past it for another process's lock.
        self.busy_deadline = busy_deadline
        self._engine = create_spool_engine(
            self.path,
            busy_timeout_seconds=self.busy_timeout_seconds,
        )
        try:
            upgrade_spool_database(
                self._engine,
                busy_timeout_seconds=self.busy_timeout_seconds,
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

    def busy_timeout_seconds(self) -> float:
        """How long a statement may wait for another process's lock.

        At most ``DATABASE_TIMEOUT_SECONDS``, and never past ``busy_deadline``
        while one is set. Applied each time a connection is checked out.
        """

        if self.busy_deadline is None:
            return DATABASE_TIMEOUT_SECONDS
        remaining = self.busy_deadline - time.monotonic()
        return min(DATABASE_TIMEOUT_SECONDS, max(0.0, remaining))

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

        return self.append_many(registration, [(event, resources)])[0]

    def append_many(
        self,
        registration: Mapping[str, Any],
        events: Sequence[tuple[Mapping[str, Any], Mapping[str, Any] | None]],
        *,
        busy_timeout_seconds: float | None = None,
        lock_timeout_seconds: float | None = None,
        before_write: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Append ``(event, resources)`` pairs in one transaction, in order.

        The events take consecutive producer sequences in the order given, or,
        when the transaction fails, none of them is stored and the producer's
        next sequence is unchanged, so a failed call can simply be repeated.

        Three controls bound how long the call can take, for a caller with a
        deadline:

        - ``lock_timeout_seconds`` caps the wait for another thread of this
          process to finish its own spool operation. Past it the call raises
          ``SpoolBusyError``, which counts as lock contention.
        - ``before_write`` runs once this call holds the spool, before it
          touches the database. Raising from it abandons the call.
        - ``busy_timeout_seconds`` caps the time all of this call's statements
          together wait for another process's lock: reading the producer's
          row, taking the write lock, and committing past other processes'
          readers. Without it each statement waits ``busy_timeout_seconds`` on
          the spool.
        """

        run_id = str(registration["run_id"])
        producer_id = str(registration["producer_id"])
        payloads: list[dict[str, Any]] = []
        with self._held(lock_timeout_seconds):
            if before_write is not None:
                before_write()
            with self._session_factory.begin() as session:
                lock_waits = _LockWaitBudget(session, busy_timeout_seconds)
                lock_waits.apply()
                run = session.get(TelemetryRunRecord, (run_id, producer_id))
                if run is None:
                    raise KeyError(run_id)
                for event, resources in events:
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
                    payloads.append(payload)
                run.updated_at = utc_now()
                # The inserts take the write lock; the commit that follows
                # waits for other processes' readers.
                lock_waits.apply()
                session.flush()
                lock_waits.apply()
        return payloads

    @contextmanager
    def _held(self, timeout_seconds: float | None) -> Iterator[None]:
        """Hold this process's spool lock, waiting at most ``timeout_seconds``."""

        if timeout_seconds is None:
            self._lock.acquire()
        elif not self._lock.acquire(timeout=max(0.0, timeout_seconds)):
            raise SpoolBusyError()
        try:
            yield
        finally:
            self._lock.release()

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
        """Prune when due, at most ``PRUNE_STEP_SECONDS`` of work per call.

        The first call prunes. After a prune finishes, or fails, the next waits
        ``PRUNE_INTERVAL_SECONDS``: a failed attempt still counts, so a spool
        that another process keeps locked is not rescanned every worker tick.
        A prune that runs out of its step continues on the next call, so a
        large backlog drains over several ticks.
        """

        now = time.monotonic()
        if (
            self._last_prune_at is not None
            and now - self._last_prune_at < PRUNE_INTERVAL_SECONDS
        ):
            return
        self._last_prune_at = now
        if not self.prune(time_budget_seconds=PRUNE_STEP_SECONDS):
            self._last_prune_at = None

    def prune(self, *, time_budget_seconds: float = math.inf) -> bool:
        """Enforce the age and total-size retention limits.

        Rows go in batches of ``PRUNE_BATCH_ROWS``, each its own short
        transaction, with a ``PRUNE_BATCH_PAUSE_SECONDS`` pause between batches
        outside this process's lock, so other processes' writers and this
        build's events get in while a backlog drains. A spool with nothing to
        remove is only read. Returns whether the prune finished; it stops
        between batches once ``time_budget_seconds`` have passed.
        """

        deadline = time.monotonic() + time_budget_seconds
        cutoff = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS)).isoformat()
        is_expired = TelemetryEventRecord.created_at < cutoff
        if self._any(select(TelemetryEventRecord.event_id).where(is_expired)):
            expired = (
                select(TelemetryEventRecord.event_id)
                .where(is_expired)
                .limit(PRUNE_BATCH_ROWS)
            )
            in_batch = TelemetryEventRecord.event_id.in_(expired)
            while self._delete(TelemetryEventRecord, in_batch) == PRUNE_BATCH_ROWS:
                if not self._pause_between_batches(deadline):
                    return False
        oldest_first = self._oldest_events_over_size_limit()
        for start in range(0, len(oldest_first), PRUNE_BATCH_ROWS):
            if start and not self._pause_between_batches(deadline):
                return False
            batch = oldest_first[start : start + PRUNE_BATCH_ROWS]
            self._delete(TelemetryEventRecord, TelemetryEventRecord.event_id.in_(batch))
        is_expired_run = (
            TelemetryRunRecord.updated_at < cutoff,
            ~TelemetryRunRecord.events.any(),
        )
        if self._any(select(TelemetryRunRecord.run_id).where(*is_expired_run)):
            expired_runs = (
                select(TelemetryRunRecord.run_id, TelemetryRunRecord.producer_id)
                .where(*is_expired_run)
                .limit(PRUNE_BATCH_ROWS)
            )
            in_batch = tuple_(
                TelemetryRunRecord.run_id, TelemetryRunRecord.producer_id
            ).in_(expired_runs)
            while self._delete(TelemetryRunRecord, in_batch) == PRUNE_BATCH_ROWS:
                if not self._pause_between_batches(deadline):
                    return False
        return True

    def _any(self, statement) -> bool:
        """Return whether a read-only query matches any row."""

        with self._lock, self._session_factory() as session:
            return session.scalar(statement.limit(1)) is not None

    @staticmethod
    def _pause_between_batches(deadline: float) -> bool:
        """Pause before the next batch, or return False when out of time."""

        if time.monotonic() + PRUNE_BATCH_PAUSE_SECONDS >= deadline:
            return False
        time.sleep(PRUNE_BATCH_PAUSE_SECONDS)
        return True

    def _delete(self, model, criterion) -> int:
        """Delete one batch of matching rows in its own transaction."""

        with self._lock, self._session_factory.begin() as session:
            return session.execute(
                delete(model)
                .where(criterion)
                .execution_options(synchronize_session=False)
            ).rowcount

    def _used_bytes(self) -> int:
        """Return the bytes of the pages the spool file uses, without a scan."""

        with self._lock, self._engine.connect() as connection:
            page_size = connection.exec_driver_sql("PRAGMA page_size").scalar()
            pages = connection.exec_driver_sql("PRAGMA page_count").scalar()
            free_pages = connection.exec_driver_sql("PRAGMA freelist_count").scalar()
        return (pages - free_pages) * page_size

    def _oldest_events_over_size_limit(self) -> list[str]:
        """Return the oldest events whose removal brings storage under the cap."""

        # Every stored payload lies on a used page, so a file that uses no more
        # than the cap cannot hold more than the cap. That spares the usual
        # prune a read of every payload; other writers cannot commit while
        # such a read runs.
        if self._used_bytes() <= MAX_QUEUED_BYTES:
            return []
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
