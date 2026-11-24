"""A pool of spawned worker processes for household-batched engine work.

The US build runs policyengine-us over its households one batch at a time.
This pool runs those batches, or any other independent units of work, in
separate worker processes and hands their results back in unit order, so a
caller can join them (concatenate arrays, add totals) in exactly the order a
single process would use and get byte-identical outputs.

How it works:

* **Spawned workers.** Each worker is a fresh Python process started with the
  ``spawn`` start method. Engine objects never cross between processes; each
  worker builds its own. Workers must be processes, not threads, because every
  policyengine-us construction sets a back-reference on the engine's shared
  default rules (``us_runtime/engine_lifecycle.py``).
* **Start-up data sent once per worker.** ``initializer(*initargs)`` runs once
  in each worker, and whatever it returns is passed to every unit that worker
  runs as ``unit_function(state, unit)``. Spawn pickles ``initargs`` once per
  worker start, so a frame passed there crosses once per worker, not once per
  unit. A unit itself should be small (indices, a request descriptor).
* **Worker replacement.** With ``max_units_per_worker``, a worker is stopped
  after that many units and a fresh one takes its place, so memory that an
  engine leaves behind does not build up. The pool starts the replacement only
  after the old worker has exited, so no more than ``workers`` processes are
  ever alive at once. The pool manages its own processes, so worker
  replacement has none of the interpreter restrictions of
  ``ProcessPoolExecutor(max_tasks_per_child=...)``.
* **A memory cap per worker.** With ``memory_cap_bytes``, a thread in the
  parent reads each worker's memory every ``memory_poll_seconds`` and stops a
  worker that goes over the cap. The reading is the process's footprint: its
  resident memory plus pages the operating system has compressed (macOS) or
  swapped out (Linux), from :func:`process_memory_bytes`.
* **Running a failed unit again.** A unit whose worker exits before returning
  its result (a crash, the operating system killing it, the memory cap) or
  whose unit function raises ``MemoryError`` runs again in a fresh worker, up
  to ``max_attempts`` attempts in all. Any other exception from the unit
  function stops the run at once: the same inputs would raise it again.
* **Results in unit order.** :meth:`WorkerPool.map` yields results in the
  order of its units, whatever order the workers finish in. At most
  ``max_units_ahead`` units, counted from the oldest result not yet handed
  back, are running or waiting with a finished result, which bounds the
  results the parent holds.
* **A worker environment.** ``worker_environment`` sets environment variables
  in each worker as it starts, before the worker imports anything, so a caller
  can, for example, give workers different numeric thread counts from the
  parent's. The pool chooses no values; without it, workers inherit the
  parent's environment.

The pool decides nothing about sizing: the caller sets the number of workers
and the memory cap. A unit function must not start ``multiprocessing`` child
processes, because the workers are daemonic.
"""

from __future__ import annotations

import ctypes
import heapq
import math
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_any
from multiprocessing.reduction import ForkingPickler

__all__ = [
    "FailedAttempt",
    "WorkerPool",
    "WorkerPoolError",
    "WorkerPoolReport",
    "WorkerPoolSettings",
    "WorkerPoolUnitError",
    "process_memory_bytes",
]

#: The longest the parent waits for a worker message before it checks the
#: workers again.
_WAIT_SECONDS = 0.5
#: How long :meth:`WorkerPool.close` waits for a stopped worker to exit before
#: killing it.
_SHUTDOWN_SECONDS = 10.0

# Environment changes for a worker start are made in this process's own
# environment, so two pools starting workers at once must take turns.
_ENVIRONMENT_LOCK = threading.Lock()


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")


@dataclass(frozen=True)
class WorkerPoolSettings:
    """How many workers to run and the limits each one runs under.

    Attributes:
        workers: Worker processes alive at most at any time.
        memory_cap_bytes: Stop a worker whose memory footprint goes over this,
            and run its unit again. ``None`` sets no cap.
        max_units_per_worker: Replace a worker after this many units. ``None``
            keeps each worker for the whole run.
        max_attempts: Attempts in all for a unit whose worker fails (see the
            module docstring); the run stops when a unit fails this many times.
        max_units_ahead: Units, counted from the oldest result not yet handed
            back, that may be running or waiting with a finished result.
            ``None`` means twice ``workers``.
        memory_poll_seconds: Seconds between readings of each worker's memory.
        worker_environment: Environment variables set in each worker as it
            starts. ``None`` leaves the parent's environment unchanged.
    """

    workers: int
    memory_cap_bytes: int | None = None
    max_units_per_worker: int | None = None
    max_attempts: int = 2
    max_units_ahead: int | None = None
    memory_poll_seconds: float = 0.5
    worker_environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _require_positive_integer("workers", self.workers)
        for name in ("memory_cap_bytes", "max_units_per_worker", "max_units_ahead"):
            value = getattr(self, name)
            if value is not None:
                _require_positive_integer(name, value)
        _require_positive_integer("max_attempts", self.max_attempts)
        seconds = self.memory_poll_seconds
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(
                f"memory_poll_seconds must be a positive number; got {seconds!r}."
            )
        if self.worker_environment is not None:
            environment = dict(self.worker_environment)
            for key, value in environment.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or "=" in key
                    or "\0" in key
                    or not isinstance(value, str)
                    or "\0" in value
                ):
                    raise ValueError(
                        "worker_environment must map variable names to string "
                        f"values; got {key!r}: {value!r}."
                    )
            object.__setattr__(self, "worker_environment", environment)

    @property
    def units_ahead(self) -> int:
        """The window of units in flight that ``max_units_ahead`` sets."""
        if self.max_units_ahead is None:
            return 2 * self.workers
        return self.max_units_ahead


@dataclass(frozen=True)
class FailedAttempt:
    """One attempt at a unit that ended without a result."""

    unit_index: int
    label: str
    attempt: int
    worker_pid: int | None
    cause: str


@dataclass(frozen=True)
class WorkerPoolReport:
    """What a pool has done so far.

    Attributes:
        worker_starts: Worker processes started, replacements included.
        units_completed: Units whose result came back.
        failed_attempts: Every attempt at a unit that ended without a result,
            in the order the parent saw them.
        max_live_workers: The most worker processes alive at once.
        peak_worker_memory_bytes: The highest memory footprint read from any
            worker, or ``None`` before the first reading.
    """

    worker_starts: int
    units_completed: int
    failed_attempts: tuple[FailedAttempt, ...]
    max_live_workers: int
    peak_worker_memory_bytes: int | None


class WorkerPoolError(RuntimeError):
    """The pool could not run its units."""


class WorkerPoolUnitError(WorkerPoolError):
    """A unit failed, so the run stopped.

    Attributes:
        unit_index: The unit's position in the sequence given to ``map``.
        label: The unit's label, from ``describe_unit``.
        causes: Why each attempt at the unit ended without a result.
    """

    def __init__(
        self, message: str, *, unit_index: int, label: str, causes: Sequence[str]
    ) -> None:
        super().__init__(message)
        self.unit_index = unit_index
        self.label = label
        self.causes = tuple(causes)


class _RemoteTracebackError(Exception):
    """The formatted traceback of an exception raised in a worker."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Reading a process's memory
# ---------------------------------------------------------------------------


class _RusageInfoV0(ctypes.Structure):
    # struct rusage_info_v0, <sys/resource.h> on macOS.
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


_RUSAGE_INFO_V0 = 0
_LIBPROC: ctypes.CDLL | None = None


def _darwin_footprint(pid: int) -> int | None:
    global _LIBPROC
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _LIBPROC.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_RusageInfoV0),
        ]
        _LIBPROC.proc_pid_rusage.restype = ctypes.c_int
    info = _RusageInfoV0()
    if _LIBPROC.proc_pid_rusage(pid, _RUSAGE_INFO_V0, ctypes.byref(info)) != 0:
        return None
    return int(info.ri_phys_footprint)


def _linux_footprint(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as status:
            lines = status.read().splitlines()
    except OSError:
        return None
    kibibytes = {}
    for line in lines:
        name, _, rest = line.partition(":")
        if name in ("VmRSS", "VmSwap"):
            kibibytes[name] = int(rest.split()[0])
    if "VmRSS" not in kibibytes:
        # A process that has exited but not been reaped reports no memory.
        return None
    return (kibibytes["VmRSS"] + kibibytes.get("VmSwap", 0)) * 1024


def _memory_reading_supported() -> bool:
    return sys.platform == "darwin" or sys.platform.startswith("linux")


def process_memory_bytes(pid: int) -> int | None:
    """A process's memory footprint in bytes, or ``None`` if it can't be read.

    This is the memory the process holds whether or not it is in RAM: on Linux
    its resident memory plus its swapped-out pages (``VmRSS + VmSwap`` from
    ``/proc/<pid>/status``); on macOS its physical footprint
    (``proc_pid_rusage``'s ``ri_phys_footprint``), which counts compressed
    pages. Resident memory alone falls when the system is short of memory and
    compresses or swaps pages out, so it would let a worker pass its cap.
    Returns ``None`` for a process that has exited, one this process may not
    read, and on other platforms.
    """
    if sys.platform == "darwin":
        return _darwin_footprint(pid)
    if sys.platform.startswith("linux"):
        return _linux_footprint(pid)
    return None


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------


def _summary(error: BaseException) -> str:
    return "".join(traceback.format_exception_only(error)).strip()


def _formatted(error: BaseException) -> str:
    return "".join(traceback.format_exception(error))


def _send_quietly(connection, message) -> bool:
    """Send ``message``; False if the parent has gone."""
    try:
        connection.send(message)
    except OSError:
        return False
    return True


def _worker_main(connection, initializer, initargs, unit_function) -> None:
    """Run in each worker: start up once, then run units until told to stop."""
    try:
        state = None if initializer is None else initializer(*initargs)
    except Exception as error:
        _send_quietly(connection, ("start_error", _summary(error), _formatted(error)))
        return
    if not _send_quietly(connection, ("ready", os.getpid())):
        return
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        index, unit = message
        try:
            result = unit_function(state, unit)
        except MemoryError as error:
            # The worker's memory may be in a bad state: report, then exit so
            # a fresh worker runs the unit again.
            _send_quietly(
                connection, ("memory_error", index, _summary(error), _formatted(error))
            )
            return
        except Exception as error:
            if not _send_quietly(
                connection, ("error", index, _summary(error), _formatted(error))
            ):
                return
            continue
        try:
            connection.send(("result", index, result))
        except OSError:
            return
        except Exception as error:
            summary = f"its result could not be sent to the parent: {_summary(error)}"
            if not _send_quietly(
                connection, ("error", index, summary, _formatted(error))
            ):
                return
        del message, unit, result


# ---------------------------------------------------------------------------
# The parent's side
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _Worker:
    process: multiprocessing.process.BaseProcess
    connection: object
    ready: bool = False
    unit_index: int | None = None
    units_run: int = 0
    retiring: bool = False
    connection_lost: bool = False

    @property
    def pid(self) -> int | None:
        return self.process.pid


@dataclass(eq=False)
class _Run:
    """One ``map``'s units and what has happened to each."""

    units: list[object]
    attempts: list[int]
    causes: list[list[str]]
    results: dict[int, object]
    retries: list[int]
    undispatched: deque[int]
    start_failures: list[str]


@contextmanager
def _environment_for_start(environment: Mapping[str, str] | None) -> Iterator[None]:
    """Hold ``environment`` in this process's environment while a worker starts.

    A spawned worker gets a copy of the parent's environment when its
    interpreter starts, before it imports the parent's main module or anything
    else, so this is the only point at which every library in the worker sees
    the values. They are restored as soon as the worker has started.
    """
    if not environment:
        yield
        return
    with _ENVIRONMENT_LOCK:
        previous = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _require_importable(name: str, function: Callable) -> None:
    if not callable(function):
        raise TypeError(f"{name} must be callable; got {function!r}.")
    try:
        pickle.dumps(function)
    except (pickle.PicklingError, AttributeError, TypeError) as error:
        raise TypeError(
            f"{name} must be importable by name in a spawned worker, such as a "
            f"module-level function; got {function!r} ({error})."
        ) from None


def _exit_cause(exitcode: int | None) -> str:
    if exitcode is None:
        return "the worker's connection to the parent closed"
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = f"signal {-exitcode}"
        return f"the worker was killed by {name}"
    if exitcode == 0:
        return "the worker exited without returning a result"
    return f"the worker exited with code {exitcode}"


def _gibibytes(count: int) -> str:
    return f"{count / 1024**3:.2f} GiB"


def _stop_processes(workers: list[_Worker]) -> None:
    """Kill any workers still alive; run when a pool is collected or at exit."""
    for worker in list(workers):
        try:
            if worker.process.exitcode is None:
                worker.process.kill()
                worker.process.join(_SHUTDOWN_SECONDS)
        except (OSError, ValueError, AssertionError):
            pass


class WorkerPool:
    """Spawned worker processes that run units and return results in order.

    Use it as a context manager, so its workers stop when the block ends::

        with WorkerPool(
            WorkerPoolSettings(workers=4, max_units_per_worker=2),
            initializer=load_batches,
            initargs=(h5_path,),
            unit_function=score_request,
        ) as pool:
            for result in pool.map(requests):
                ...

    ``initializer`` and ``unit_function`` must be importable by name in a
    fresh interpreter (module-level functions). ``describe_unit(index, unit)``
    labels a unit in errors; it runs only in the parent. ``map`` may be called
    again once the previous ``map`` has finished, and reuses the live workers.
    A ``map`` that does not run to the end, because a unit failed or the
    caller stopped iterating, closes the pool.
    """

    def __init__(
        self,
        settings: WorkerPoolSettings,
        *,
        unit_function: Callable[[object, object], object],
        initializer: Callable[..., object] | None = None,
        initargs: Sequence[object] = (),
        describe_unit: Callable[[int, object], str] | None = None,
    ) -> None:
        if not isinstance(settings, WorkerPoolSettings):
            raise TypeError("settings must be a WorkerPoolSettings.")
        _require_importable("unit_function", unit_function)
        if initializer is not None:
            _require_importable("initializer", initializer)
        if settings.memory_cap_bytes is not None and not _memory_reading_supported():
            raise ValueError(
                f"memory_cap_bytes needs worker memory readings, which "
                f"{sys.platform} does not support here."
            )
        self._settings = settings
        self._unit_function = unit_function
        self._initializer = initializer
        self._initargs = tuple(initargs)
        self._describe_unit = describe_unit
        self._context = multiprocessing.get_context("spawn")
        # Every reap, start and kill of a worker process happens under this
        # lock, so the memory thread never signals a process the parent has
        # already reaped.
        self._lock = threading.Lock()
        self._workers: list[_Worker] = []
        self._cap_overruns: dict[int, int] = {}
        self._closed = False
        self._mapping = False
        self._worker_starts = 0
        self._units_completed = 0
        self._failed_attempts: list[FailedAttempt] = []
        self._max_live_workers = 0
        self._peak_memory: int | None = None
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._finalizer = weakref.finalize(self, _stop_processes, self._workers)

    # -- public ------------------------------------------------------------

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def report(self) -> WorkerPoolReport:
        with self._lock:
            return WorkerPoolReport(
                worker_starts=self._worker_starts,
                units_completed=self._units_completed,
                failed_attempts=tuple(self._failed_attempts),
                max_live_workers=self._max_live_workers,
                peak_worker_memory_bytes=self._peak_memory,
            )

    def map(self, units: Iterable[object]) -> Iterator[object]:
        """Run ``unit_function(state, unit)`` for each unit; yield results in order."""
        if self._closed:
            raise WorkerPoolError("The worker pool is closed.")
        if self._mapping:
            raise WorkerPoolError("The worker pool is already running a map.")
        self._mapping = True
        finished = False
        try:
            yield from self._run(list(units))
            finished = True
        finally:
            self._mapping = False
            if not finished:
                self.close()

    def close(self) -> None:
        """Stop every worker: idle ones are told to exit, busy ones are killed."""
        if self._closed:
            return
        self._closed = True
        self._monitor_stop.set()
        if self._monitor is not None:
            self._monitor.join()
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            if (
                worker.ready
                and worker.unit_index is None
                and not worker.connection_lost
            ):
                if not worker.retiring:
                    self._send_stop(worker)
            else:
                with self._lock:
                    worker.process.kill()
        deadline = time.monotonic() + _SHUTDOWN_SECONDS
        for worker in workers:
            with self._lock:
                worker.process.join(max(0.0, deadline - time.monotonic()))
                if worker.process.exitcode is None:
                    worker.process.kill()
                    worker.process.join(_SHUTDOWN_SECONDS)
            self._forget(worker)

    # -- running units ------------------------------------------------------

    def _label(self, units: Sequence[object], index: int) -> str:
        if self._describe_unit is not None:
            return str(self._describe_unit(index, units[index]))
        return f"unit {index + 1}/{len(units)}"

    def _run(self, units: list[object]) -> Iterator[object]:
        count = len(units)
        if count == 0:
            return
        settings = self._settings
        self._start_monitor()
        run = _Run(
            units=units,
            attempts=[0] * count,
            causes=[[] for _ in range(count)],
            results={},
            retries=[],
            undispatched=deque(range(count)),
            start_failures=[],
        )
        next_index = 0
        while next_index < count:
            while next_index in run.results:
                result = run.results.pop(next_index)
                next_index += 1
                yield result
                del result
            if next_index >= count:
                break
            window_end = next_index + settings.units_ahead
            waiting = len(run.retries) + sum(
                1 for index in run.undispatched if index < window_end
            )
            self._start_workers(waiting)
            for worker in list(self._workers):
                if (
                    not worker.ready
                    or worker.retiring
                    or worker.connection_lost
                    or worker.unit_index is not None
                ):
                    continue
                if run.retries:
                    index = heapq.heappop(run.retries)
                elif run.undispatched and run.undispatched[0] < window_end:
                    index = run.undispatched.popleft()
                else:
                    break
                if not self._send_unit(worker, index, run):
                    heapq.heappush(run.retries, index)
            self._wait_and_handle(run)

    def _unit_failed(self, run: _Run, index: int, pid: int | None, cause: str) -> None:
        """Record a failed attempt; queue the unit again or stop the run."""
        label = self._label(run.units, index)
        run.causes[index].append(cause)
        with self._lock:
            self._failed_attempts.append(
                FailedAttempt(index, label, run.attempts[index], pid, cause)
            )
        if run.attempts[index] >= self._settings.max_attempts:
            detail = "; ".join(
                f"attempt {number}: {text}"
                for number, text in enumerate(run.causes[index], start=1)
            )
            raise WorkerPoolUnitError(
                f"{label} failed on all {run.attempts[index]} attempts: {detail}",
                unit_index=index,
                label=label,
                causes=run.causes[index],
            )
        heapq.heappush(run.retries, index)

    def _start_workers(self, waiting: int) -> None:
        available = sum(
            1
            for worker in self._workers
            if not worker.retiring
            and not worker.connection_lost
            and worker.unit_index is None
        )
        while available < waiting and len(self._workers) < self._settings.workers:
            self._start_worker()
            available += 1

    def _start_worker(self) -> None:
        parent_end, child_end = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child_end, self._initializer, self._initargs, self._unit_function),
            name=f"worker-pool-{self._worker_starts + 1}",
            daemon=True,
        )
        try:
            with self._lock, _environment_for_start(self._settings.worker_environment):
                process.start()
                self._workers.append(_Worker(process=process, connection=parent_end))
                self._worker_starts += 1
                self._max_live_workers = max(self._max_live_workers, len(self._workers))
        except Exception as error:
            parent_end.close()
            raise WorkerPoolError(f"A worker could not be started: {error}") from error
        finally:
            child_end.close()

    def _send_unit(self, worker: _Worker, index: int, run: _Run) -> bool:
        """Send a unit to an idle worker; False if the worker has gone."""
        try:
            payload = ForkingPickler.dumps((index, run.units[index]))
        except Exception as error:
            label = self._label(run.units, index)
            cause = f"the unit could not be sent to a worker: {_summary(error)}"
            raise WorkerPoolUnitError(
                f"{label} failed: {cause}",
                unit_index=index,
                label=label,
                causes=[cause],
            ) from error
        try:
            worker.connection.send_bytes(payload)
        except OSError:
            worker.connection_lost = True
            return False
        run.attempts[index] += 1
        worker.unit_index = index
        return True

    def _send_stop(self, worker: _Worker) -> None:
        worker.retiring = True
        try:
            worker.connection.send(None)
        except OSError:
            worker.connection_lost = True

    def _wait_and_handle(self, run: _Run) -> None:
        by_connection = {
            worker.connection: worker
            for worker in self._workers
            if not worker.connection_lost
        }
        by_sentinel = {worker.process.sentinel: worker for worker in self._workers}
        ready = wait_for_any([*by_connection, *by_sentinel], timeout=_WAIT_SECONDS)
        touched: list[_Worker] = []
        for item in ready:
            worker = by_connection.get(item) or by_sentinel.get(item)
            if worker is not None and worker not in touched:
                touched.append(worker)
        for worker in touched:
            self._drain(worker, run)
        for worker in list(self._workers):
            with self._lock:
                exitcode = worker.process.exitcode
            if exitcode is None:
                continue
            # Messages sent just before the worker exited come first.
            self._drain(worker, run)
            self._worker_exited(worker, exitcode, run)

    def _drain(self, worker: _Worker, run: _Run) -> None:
        while not worker.connection_lost:
            try:
                if not worker.connection.poll():
                    return
                message = worker.connection.recv()
            except (EOFError, OSError):
                worker.connection_lost = True
                return
            self._handle(worker, message, run)

    def _handle(self, worker: _Worker, message: tuple, run: _Run) -> None:
        kind = message[0]
        if kind == "ready":
            worker.ready = True
            run.start_failures.clear()
        elif kind == "start_error":
            _, summary, text = message
            raise WorkerPoolError(
                f"A worker failed while starting: {summary}"
            ) from _RemoteTracebackError(text)
        elif kind == "result":
            _, index, result = message
            if index != worker.unit_index:
                raise WorkerPoolError(
                    f"Worker {worker.pid} returned unit {index} while running "
                    f"unit {worker.unit_index}."
                )
            run.results[index] = result
            worker.unit_index = None
            worker.units_run += 1
            with self._lock:
                self._units_completed += 1
            limit = self._settings.max_units_per_worker
            if limit is not None and worker.units_run >= limit:
                self._send_stop(worker)
        elif kind == "error":
            _, index, summary, text = message
            label = self._label(run.units, index)
            run.causes[index].append(summary)
            with self._lock:
                self._failed_attempts.append(
                    FailedAttempt(
                        index, label, run.attempts[index], worker.pid, summary
                    )
                )
            raise WorkerPoolUnitError(
                f"{label} failed in worker {worker.pid}: {summary}",
                unit_index=index,
                label=label,
                causes=run.causes[index],
            ) from _RemoteTracebackError(text)
        elif kind == "memory_error":
            _, index, summary, _text = message
            worker.unit_index = None
            worker.retiring = True
            self._unit_failed(run, index, worker.pid, summary)
        else:  # pragma: no cover - the worker sends only the kinds above
            raise WorkerPoolError(f"Worker {worker.pid} sent {kind!r}.")

    def _worker_exited(self, worker: _Worker, exitcode: int, run: _Run) -> None:
        pid = worker.pid
        with self._lock:
            overrun = self._cap_overruns.pop(pid, None)
        if overrun is not None:
            cause = (
                f"the worker went over its memory cap of "
                f"{_gibibytes(self._settings.memory_cap_bytes)} "
                f"({_gibibytes(overrun)} read)"
            )
        else:
            cause = _exit_cause(exitcode)
        index = worker.unit_index
        was_starting = not worker.ready and not worker.retiring
        self._forget(worker)
        if index is not None:
            self._unit_failed(run, index, pid, cause)
        elif was_starting:
            run.start_failures.append(cause)
            if len(run.start_failures) >= self._settings.max_attempts:
                raise WorkerPoolError(
                    f"Workers failed {len(run.start_failures)} times in a row "
                    f"while starting: {'; '.join(run.start_failures)}"
                )

    def _forget(self, worker: _Worker) -> None:
        with self._lock:
            if worker in self._workers:
                self._workers.remove(worker)
            try:
                worker.process.join(0)
                if worker.process.exitcode is not None:
                    worker.process.close()
            except ValueError:
                pass
        try:
            worker.connection.close()
        except OSError:
            pass

    # -- the memory thread --------------------------------------------------

    def _start_monitor(self) -> None:
        if self._monitor is not None or not _memory_reading_supported():
            return
        self._monitor = threading.Thread(
            target=self._watch_memory, name="worker-pool-memory", daemon=True
        )
        self._monitor.start()

    def _watch_memory(self) -> None:
        cap = self._settings.memory_cap_bytes
        while not self._monitor_stop.wait(self._settings.memory_poll_seconds):
            with self._lock:
                targets = [(worker, worker.process.pid) for worker in self._workers]
            for worker, pid in targets:
                if pid is None:
                    continue
                observed = process_memory_bytes(pid)
                if observed is None:
                    continue
                with self._lock:
                    # A worker still listed has not been reaped or closed, so
                    # its pid is still its own.
                    if (
                        worker not in self._workers
                        or worker.process.exitcode is not None
                    ):
                        continue
                    if self._peak_memory is None or observed > self._peak_memory:
                        self._peak_memory = observed
                    if cap is not None and observed > cap:
                        self._cap_overruns.setdefault(pid, observed)
                        worker.process.kill()
