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
    spool has refused it for another reason. ``finish`` sets a drain deadline
    and closes the queue; the writer then starts no append after
    ``deadline - WRITER_BUSY_TIMEOUT_SECONDS``, and since each statement waits
    at most that long for a lock, its last append ends by the deadline plus the
    time its own inserts take.
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
        """Write queued events on a daemon thread until ``finish``."""

        self._thread = threading.Thread(
            target=self._run, name="telemetry-spool-writer", daemon=True
        )
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def finish(self, deadline: float) -> int:
        """Close the queue and write what it holds, until ``deadline``.

        ``deadline`` is a reading of this writer's clock. Returns how many
        events were left unwritten. A writer whose thread was never started,
        such as one a test drives step by step, writes on the calling thread.
        """

        self._drain_deadline = deadline
        self.queue.close()
        if self._thread is None:
            if len(self.queue):
                self.write_queued()
        else:
            # The thread's last append ends by the deadline plus its inserts;
            # this waits for that and gives up a statement wait later.
            self._thread.join(
                timeout=max(0.0, deadline - self._clock()) + WRITER_BUSY_TIMEOUT_SECONDS
            )
        return len(self.queue)

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
        if self._clock() >= self._last_attempt_start():
            raise DrainDeadlineError()
        pairs = [event.decode() for event in events]
        retry_spool_contention(
            lambda: self.spool.append_many(
                self.registration,
                pairs,
                busy_timeout_seconds=WRITER_BUSY_TIMEOUT_SECONDS,
            ),
            deadline=self._last_attempt_start,
            clock=self._clock,
            sleep=self._sleep,
        )

    def _last_attempt_start(self) -> float:
        # Infinite until finish sets a drain deadline.
        return self._drain_deadline - WRITER_BUSY_TIMEOUT_SECONDS
