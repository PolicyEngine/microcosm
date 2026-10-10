"""Ordered, bounded hand-off of accepted events from the socket to the spool.

The service's socket thread validates each event, queues it here and replies
at once. One writer thread appends queued events to the shared spool, oldest
first, and is the only code that appends this producer's events, so the spool
assigns sequences in queue order. Lock contention on the spool therefore
delays only the writer, never a build's send.
"""

from __future__ import annotations

import itertools
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from microcosm.build.telemetry_emitter_service.constants import (
    QUEUE_CLOSED_ERROR,
    QUEUE_FULL_ERROR,
    QUEUE_MAX_BYTES,
    QUEUE_MAX_EVENTS,
    QUEUE_RESERVED_BYTES,
    QUEUE_RESERVED_EVENTS,
    WRITER_BATCH_EVENTS,
    WRITER_BUSY_TIMEOUT_SECONDS,
    WRITER_STOPPED_WARNING,
)
from microcosm.build.telemetry_emitter_service.contention import (
    is_transient_spool_error,
    retry_spool_contention,
)
from microcosm.build.telemetry_emitter_service.diagnostics import (
    describe_error,
    write_warning,
)
from microcosm.build.telemetry_protocol import EVENT_TYPE_RUN


class EventRefusedError(Exception):
    """The queue did not take an event."""


class QueueFullError(EventRefusedError):
    """Taking the event would exceed the queue's bound."""

    def __init__(self) -> None:
        super().__init__(QUEUE_FULL_ERROR)


class QueueClosedError(EventRefusedError):
    """The queue takes no more events: the service is shutting down."""

    def __init__(self) -> None:
        super().__init__(QUEUE_CLOSED_ERROR)


class DrainDeadlineError(Exception):
    """The writer may not start another append before the drain deadline."""


@dataclass(frozen=True, slots=True)
class QueuedEvent:
    """One accepted event and its resource sample, held as compact JSON.

    Holding bytes rather than parsed objects makes the queue's byte bound a
    bound on the memory its events occupy.
    """

    encoded: bytes
    run_event: bool

    @classmethod
    def encode(
        cls,
        event: Mapping[str, Any],
        resources: Mapping[str, Any] | None,
    ) -> QueuedEvent:
        """Encode an event; raises ``ValueError`` for values JSON cannot hold."""

        encoded = json.dumps(
            [event, resources], separators=(",", ":"), allow_nan=False
        ).encode()
        return cls(encoded, event.get("event_type") == EVENT_TYPE_RUN)

    @property
    def size(self) -> int:
        return len(self.encoded)

    def decode(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        event, resources = json.loads(self.encoded)
        return event, resources


class EventQueue:
    """Accepted events, oldest first, bounded in count and in encoded bytes.

    Safe to share between threads. Its lock is held only for in-memory work,
    never while the spool is written. Only the writer removes events, and only
    from the front, so the events it reads with ``oldest`` are still the oldest
    when it removes them.
    """

    def __init__(
        self,
        *,
        max_events: int = QUEUE_MAX_EVENTS,
        max_bytes: int = QUEUE_MAX_BYTES,
        reserved_events: int = QUEUE_RESERVED_EVENTS,
        reserved_bytes: int = QUEUE_RESERVED_BYTES,
    ) -> None:
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.reserved_events = reserved_events
        self.reserved_bytes = reserved_bytes
        self._events: deque[QueuedEvent] = deque()
        self._bytes = 0
        self._closed = False
        self._changed = threading.Condition()

    def __len__(self) -> int:
        with self._changed:
            return len(self._events)

    @property
    def queued_bytes(self) -> int:
        with self._changed:
            return self._bytes

    @property
    def closed(self) -> bool:
        with self._changed:
            return self._closed

    def offer(self, event: QueuedEvent, *, close: bool = False) -> None:
        """Queue ``event`` behind every event queued before it.

        A run event may use the reserved room; any other event must leave it
        free. Raises ``QueueFullError`` when the event does not fit and
        ``QueueClosedError`` once the queue is closed. With ``close`` the queue
        closes after this offer, whether or not it took the event, so nothing
        can be queued behind it.
        """

        with self._changed:
            try:
                if self._closed:
                    raise QueueClosedError()
                if not self._fits(event):
                    raise QueueFullError()
                self._events.append(event)
                self._bytes += event.size
            finally:
                if close:
                    self._closed = True
                self._changed.notify_all()

    def _fits(self, event: QueuedEvent) -> bool:
        max_events = self.max_events
        max_bytes = self.max_bytes
        if not event.run_event:
            max_events -= self.reserved_events
            max_bytes -= self.reserved_bytes
        return len(self._events) < max_events and self._bytes + event.size <= max_bytes

    def close(self) -> None:
        """Refuse every later offer; queued events stay for the writer."""

        with self._changed:
            self._closed = True
            self._changed.notify_all()

    def wait_for_events(self) -> bool:
        """Block until an event is queued, or return False once closed and empty."""

        with self._changed:
            while not self._events and not self._closed:
                self._changed.wait()
            return bool(self._events)

    def oldest(self, limit: int) -> list[QueuedEvent]:
        """Return up to ``limit`` of the oldest events, leaving them queued."""

        with self._changed:
            return list(itertools.islice(self._events, limit))

    def remove(self, count: int) -> None:
        """Remove the ``count`` oldest events, once they are written or dropped."""

        with self._changed:
            for _ in range(count):
                self._bytes -= self._events.popleft().size
            self._changed.notify_all()


class SpoolWriter:
    """Appends one producer's queued events to the spool, in order.

    While the service serves, a lock error is retried for as long as it lasts:
    an event leaves the queue only once the spool has stored it, or once the
    spool has refused it for another reason.

    ``begin_drain`` closes the queue and sets a deadline. From then on no
    append touches the database after ``deadline -
    WRITER_BUSY_TIMEOUT_SECONDS``: every attempt checks that once it holds the
    spool, having waited no longer than that for another thread to release it.
    An append's statements together wait at most
    ``WRITER_BUSY_TIMEOUT_SECONDS`` for other processes' locks, so the last
    append ends by the deadline plus the time its own inserts take.
    """

    def __init__(
        self,
        spool: Any,
        registration: Mapping[str, Any],
        queue: EventQueue,
        *,
        report_dropped: Callable[[Exception], None],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spool = spool
        self.registration = dict(registration)
        self.queue = queue
        self._report_dropped = report_dropped
        self._clock = clock
        self._sleep = sleep
        self._drain_deadline = math.inf
        self._thread: threading.Thread | None = None
        # Why the writer last stopped with events still queued.
        self.stopped_by: BaseException | None = None
        # Events the spool refused for a reason other than lock contention.
        self.dropped = 0

    def start(self) -> None:
        """Write queued events on a daemon thread until the drain ends."""

        self._thread = threading.Thread(
            target=self._run, name="telemetry-spool-writer", daemon=True
        )
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def begin_drain(self, deadline: float) -> None:
        """Close the queue and give the writer until ``deadline`` to empty it.

        ``deadline`` is a reading of this writer's clock. Returns at once; a
        writer that is retrying a lock error picks the deadline up at its next
        attempt.
        """

        self._drain_deadline = deadline
        self.queue.close()

    def wait_drained(self) -> int:
        """Wait for the drain to end and return how many events are unwritten.

        The writer's last append ends by the deadline plus its inserts; this
        waits one statement wait past the deadline for it. A writer whose
        thread was never started, such as one a test drives step by step,
        writes on the calling thread.
        """

        if self._thread is None:
            if len(self.queue):
                self.write_queued()
        else:
            self._thread.join(
                timeout=max(0.0, self._drain_deadline - self._clock())
                + WRITER_BUSY_TIMEOUT_SECONDS
            )
        return len(self.queue)

    def finish(self, deadline: float) -> int:
        """Drain until ``deadline`` and return how many events are unwritten."""

        self.begin_drain(deadline)
        return self.wait_drained()

    def _run(self) -> None:
        try:
            while self.queue.wait_for_events():
                if not self.write_queued():
                    return
        except Exception as error:
            # A bug, not contention: one line, never a traceback, on the
            # build's stderr. The service stops serving once this thread ends.
            self.stopped_by = error
            write_warning(
                WRITER_STOPPED_WARNING.format(
                    error_type=type(error).__name__, error=describe_error(error)
                )
            )

    def write_queued(self) -> bool:
        """Append queued events, oldest first and in batches, until none is left.

        Returns False when the drain deadline stopped it with events queued.
        """

        while events := self.queue.oldest(WRITER_BATCH_EVENTS):
            try:
                self._write_oldest(events)
            except Exception as error:
                if not (
                    isinstance(error, DrainDeadlineError)
                    or is_transient_spool_error(error)
                ):
                    raise
                # A deadline that passed between attempts ends the drain with
                # no error of its own; keep the lock error that explains it.
                if self.stopped_by is None or is_transient_spool_error(error):
                    self.stopped_by = error
                return False
        return True

    def _write_oldest(self, events: list[QueuedEvent]) -> None:
        """Append the queue's oldest ``events`` in one transaction, then remove them.

        A lock error is retried until the drain deadline and then raised, with
        the events still queued. An event the spool refuses for any other
        reason is reported and dropped on its own, so it cannot take the rest
        of its batch with it.
        """

        try:
            self._append(events)
        except Exception as error:
            if isinstance(error, DrainDeadlineError) or is_transient_spool_error(error):
                raise
            if len(events) > 1:
                for event in events:
                    self._write_oldest([event])
                return
            self.dropped += 1
            self._report_dropped(error)
        self.queue.remove(len(events))

    def _append(self, events: list[QueuedEvent]) -> None:
        pairs = [event.decode() for event in events]

        def attempt() -> None:
            # Checked at the start of every attempt, not once before the first:
            # a backoff can outlast its plan, and the deadline can arrive
            # while the writer sleeps.
            left = self._last_start() - self._clock()
            if left <= 0:
                raise DrainDeadlineError()
            try:
                self.spool.append_many(
                    self.registration,
                    pairs,
                    busy_timeout_seconds=WRITER_BUSY_TIMEOUT_SECONDS,
                    lock_timeout_seconds=min(left, WRITER_BUSY_TIMEOUT_SECONDS),
                    before_write=self._check_may_write,
                )
            except Exception as error:
                if is_transient_spool_error(error):
                    self.stopped_by = error
                raise

        retry_spool_contention(
            attempt,
            deadline=self._last_start,
            clock=self._clock,
            sleep=self._sleep,
        )
        # Written: an earlier lock error no longer explains anything.
        self.stopped_by = None

    def _check_may_write(self) -> None:
        # Runs once the spool is held, so time spent waiting for another
        # thread to release it cannot carry database work past the deadline.
        if self._clock() >= self._last_start():
            raise DrainDeadlineError()

    def _last_start(self) -> float:
        # The latest an append may begin its database work. Infinite until
        # begin_drain sets a deadline.
        return self._drain_deadline - WRITER_BUSY_TIMEOUT_SECONDS
