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
  unit; while a worker starts, the parent briefly holds that pickled copy too.
  A unit itself should be small (indices, a request descriptor).
* **Worker replacement.** With ``max_units_per_worker``, a worker is told to
  stop after that many units and a fresh one takes its place, so memory that
  an engine leaves behind does not build up. The replacement starts only after
  the old worker has exited, so no more than ``workers`` processes are ever
  alive at once; a worker that has not exited ``_STOP_SECONDS`` after being
  told to stop is killed. The pool manages its own processes, so worker
  replacement has none of the interpreter restrictions of
  ``ProcessPoolExecutor(max_tasks_per_child=...)``.
* **A memory cap per worker.** With ``memory_cap_bytes``, a thread in the
  parent reads each worker's memory every ``memory_poll_seconds`` and kills a
  worker that goes over the cap. The parent also reads a worker's memory when
  it returns a result, and replaces a worker that is over the cap then, before
  giving it another unit, so memory one unit leaves behind is not charged to
  the next. The reading is the process's footprint: its resident memory plus
  pages the operating system has compressed (macOS) or swapped out (Linux),
  from :func:`process_memory_bytes`.
* **Running a failed unit again.** A unit whose worker exits before returning
  its result (a crash, the operating system killing it, the memory cap) or
  whose unit function raises ``MemoryError`` runs again in a fresh worker, up
  to ``max_attempts`` attempts in all. Any other exception from the unit
  function, and a unit or result that cannot be pickled or unpickled, stops
  the run at once: the same inputs would fail again.
* **Results in unit order.** :meth:`WorkerPool.map` yields results in the
  order of its units, whatever order the workers finish in. At most
  ``max_units_ahead`` units, counted from the oldest result not yet handed
  back, are running or waiting with a finished result, which bounds the
  results the parent holds. Workers are given new units only while the caller
  is asking ``map`` for its next result.
* **A worker environment.** ``worker_environment`` sets environment variables
  in each worker as it starts, before the worker imports anything, so a caller
  can, for example, give workers different numeric thread counts from the
  parent's. The pool chooses no values; without it, workers inherit the
  parent's environment. The values are held in the parent's own environment
  while each worker starts, so another thread of the parent that reads the
  environment, or starts a subprocess, at that moment sees them too.
* **No orphaned workers.** Each worker exits as soon as its parent process
  dies, even in the middle of a unit; a pool that is garbage-collected or left
  open at interpreter exit kills its workers.

The pool decides nothing about sizing: the caller sets the number of workers
and the memory cap. A unit function must not start ``multiprocessing`` child
processes, because the workers are daemonic. A worker start imports this
module, and so the ``microcosm.build`` package and what it imports.

On Linux a worker is killed through a process file descriptor (``pidfd``), so
a kill can never reach another process that has been given the same process
ID. On macOS the pool kills by process ID: it checks first that the worker has
not exited, and the parent reaps workers only under the lock the memory thread
holds, but other code in the parent that reaps child processes (another
``multiprocessing.Process.start``, ``multiprocessing.active_children()``)
leaves a window of microseconds in which the ID could be reused.
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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
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
#: How long a worker told to stop may take to exit before it is killed.
_STOP_SECONDS = 10.0
#: How long :meth:`WorkerPool.close` waits for its workers to exit before it
#: kills them.
_SHUTDOWN_SECONDS = 10.0
#: The exit code of a worker that stopped because its parent died.
_PARENT_GONE_EXIT_CODE = 75

# Environment changes for a worker start are made in this process's own
# environment, so two pools starting workers at once must take turns.
_ENVIRONMENT_LOCK = threading.Lock()


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")


def _environment_entry_is_valid(key: object, value: object) -> bool:
    if not isinstance(key, str) or not isinstance(value, str):
        return False
    if not key or "=" in key or "\0" in key or "\0" in value:
        return False
    try:
        os.fsencode(key)
        os.fsencode(value)
    except UnicodeEncodeError:
        return False
    return True


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
            Also the number of worker starts in a row that may fail before the
            run stops.
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
                if not _environment_entry_is_valid(key, value):
                    raise ValueError(
                        "worker_environment must map variable names to string "
                        "values the operating system can encode; got "
                        f"{key!r}: {value!r}."
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
        workers_stopped_over_cap: Workers replaced because they were over the
            memory cap after returning a result, without a unit being charged.
    """

    worker_starts: int
    units_completed: int
    failed_attempts: tuple[FailedAttempt, ...]
    max_live_workers: int
    peak_worker_memory_bytes: int | None
    workers_stopped_over_cap: int


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


def _exit_when_parent_dies() -> None:
    """End this worker as soon as its parent process dies.

    A spawned worker holds the read end of a pipe whose write end only its
    parent holds, so the pipe reads as closed exactly when the parent has
    gone. Without this, a busy worker would run its unit to the end after its
    parent was killed.
    """
    parent = multiprocessing.parent_process()
    if parent is None:  # pragma: no cover - a pool worker always has a parent
        return

    def watch() -> None:
        wait_for_any([parent.sentinel])
        os._exit(_PARENT_GONE_EXIT_CODE)

    threading.Thread(target=watch, name="worker-pool-parent", daemon=True).start()


def _worker_main(connection, initializer, initargs, unit_function) -> None:
    """Run in each worker: start up once, then run units until told to stop."""
    _exit_when_parent_dies()
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
        except Exception as error:
            # The unit could not be unpickled here; the parent knows which.
            if not _send_quietly(
                connection, ("receive_error", _summary(error), _formatted(error))
            ):
                return
            continue
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
    pid: int
    pidfd: int | None = None
    ready: bool = False
    unit_index: int | None = None
    units_run: int = 0
    retiring: bool = False
    stop_deadline: float | None = None
    connection_lost: bool = False
    stopped_over_cap: bool = False
    exited: bool = False


@dataclass(eq=False)
class _PoolState:
    """What the pool, its memory thread and its finalizer share.

    The memory thread and the finalizer hold this, not the pool, so a pool
    nobody refers to any more can be garbage-collected and its finalizer can
    stop the workers. Every change to ``workers``, every kill and every reap
    happens under ``lock``.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    workers: list[_Worker] = field(default_factory=list)
    cap_overruns: dict[int, int] = field(default_factory=dict)
    stop: threading.Event = field(default_factory=threading.Event)
    peak_memory: int | None = None
    workers_stopped_over_cap: int = 0


@dataclass(eq=False)
class _Run:
    """One ``map``'s units and what has happened to each."""

    units: list[object]
    attempts: list[int]
    causes: list[list[str]]
    results: dict[int, object]
    retries: list[int]
    start_failures: list[str]


def _has_exited(worker: _Worker) -> bool:
    """True once the worker's process has exited, whoever reaped it."""
    return bool(wait_for_any([worker.process.sentinel], timeout=0))


def _kill(worker: _Worker) -> None:
    """Kill a worker that has not exited; call with the state's lock held."""
    if worker.exited or _has_exited(worker):
        return
    try:
        if worker.pidfd is not None:
            signal.pidfd_send_signal(worker.pidfd, signal.SIGKILL)
        else:
            os.kill(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _shut_down(state: _PoolState) -> None:
    """Kill every worker; run when a pool is collected or at interpreter exit."""
    state.stop.set()
    with state.lock:
        workers = list(state.workers)
        for worker in workers:
            _kill(worker)
    for worker in workers:
        try:
            worker.process.join(_SHUTDOWN_SECONDS)
        except (OSError, ValueError, AssertionError):
            pass


def _watch_memory(state: _PoolState, cap: int | None, interval: float) -> None:
    """The memory thread: read each worker's footprint, kill one over the cap."""
    while not state.stop.wait(interval):
        with state.lock:
            targets = [worker for worker in state.workers if not worker.exited]
        for worker in targets:
            observed = process_memory_bytes(worker.pid)
            if observed is None:
                continue
            with state.lock:
                # A worker still listed has not been reaped by the pool, so its
                # process ID is still its own.
                if worker not in state.workers or worker.exited or _has_exited(worker):
                    continue
                if state.peak_memory is None or observed > state.peak_memory:
                    state.peak_memory = observed
                if cap is not None and observed > cap:
                    state.cap_overruns.setdefault(worker.pid, observed)
                    _kill(worker)


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
        try:
            for name, value in environment.items():
                os.environ[name] = value
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
    problem = None
    if getattr(function, "__module__", None) == "__main__":
        main = sys.modules.get("__main__")
        spec = getattr(main, "__spec__", None)
        if (
            getattr(main, "__file__", None) is None
            and getattr(spec, "name", None) is None
        ):
            # Spawn re-imports the parent's main module from its file or its
            # module name; a ``python -c`` command or a REPL has neither.
            problem = "its module is a __main__ that spawned workers cannot import"
    if problem is None:
        try:
            pickle.dumps(function)
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            problem = str(error)
    if problem is not None:
        raise TypeError(
            f"{name} must be importable by name in a spawned worker, such as a "
            f"module-level function in a module file; got {function!r} "
            f"({problem})."
        )


def _exit_cause(exitcode: int | None) -> str:
    if exitcode is None:
        return "the worker exited; its exit status was collected elsewhere"
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
    fresh interpreter (module-level functions in a module file).
    ``describe_unit(index, unit)`` labels a unit in errors; it runs only in the
    parent. ``map`` may be called again once the previous ``map`` has handed
    back its last result, and reuses the live workers. A ``map`` stopped
    before its last result, because a unit failed or the caller stopped
    iterating, closes the pool.
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
        self._shared = _PoolState()
        self._closed = False
        self._mapping = False
        self._worker_starts = 0
        self._units_completed = 0
        self._failed_attempts: list[FailedAttempt] = []
        self._max_live_workers = 0
        self._monitor: threading.Thread | None = None
        self._finalizer = weakref.finalize(self, _shut_down, self._shared)

    # -- public ------------------------------------------------------------

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def report(self) -> WorkerPoolReport:
        with self._shared.lock:
            return WorkerPoolReport(
                worker_starts=self._worker_starts,
                units_completed=self._units_completed,
                failed_attempts=tuple(self._failed_attempts),
                max_live_workers=self._max_live_workers,
                peak_worker_memory_bytes=self._shared.peak_memory,
                workers_stopped_over_cap=self._shared.workers_stopped_over_cap,
            )

    def map(self, units: Iterable[object]) -> Iterator[object]:
        """Run ``unit_function(state, unit)`` for each unit; yield results in order."""
        if self._closed:
            raise WorkerPoolError("The worker pool is closed.")
        if self._mapping:
            raise WorkerPoolError("The worker pool is already running a map.")
        units = list(units)
        self._mapping = True
        delivered_all = False
        try:
            for index, result in self._run(units):
                if index == len(units) - 1:
                    # Every result is out: the map is over even if the caller
                    # never asks for more, so the pool stays usable.
                    delivered_all = True
                    self._mapping = False
                yield result
            delivered_all = True
        finally:
            self._mapping = False
            if not delivered_all:
                self.close()

    def close(self) -> None:
        """Stop every worker: idle ones are told to exit, busy ones are killed."""
        if self._closed:
            return
        self._closed = True
        shared = self._shared
        shared.stop.set()
        if self._monitor is not None:
            self._monitor.join()
        with shared.lock:
            workers = list(shared.workers)
        for worker in workers:
            if (
                worker.ready
                and worker.unit_index is None
                and not worker.retiring
                and not worker.connection_lost
            ):
                self._send_stop(worker)
            elif not worker.retiring:
                with shared.lock:
                    _kill(worker)
        deadline = time.monotonic() + _SHUTDOWN_SECONDS
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            wait_for_any([worker.process.sentinel], timeout=remaining)
            with shared.lock:
                _kill(worker)
            wait_for_any([worker.process.sentinel], timeout=_SHUTDOWN_SECONDS)
            self._forget(worker)
        self._finalizer.detach()

    # -- running units ------------------------------------------------------

    def _label(self, units: Sequence[object], index: int) -> str:
        if self._describe_unit is not None:
            return str(self._describe_unit(index, units[index]))
        return f"unit {index + 1}/{len(units)}"

    def _run(self, units: list[object]) -> Iterator[tuple[int, object]]:
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
            start_failures=[],
        )
        next_new = 0  # The first unit not yet sent to any worker.
        next_index = 0  # The next result to hand back.
        while next_index < count:
            if self._closed:
                raise WorkerPoolError("The worker pool is closed.")
            if next_index in run.results:
                result = run.results.pop(next_index)
                next_index += 1
                yield next_index - 1, result
                del result
                continue
            window_end = min(count, next_index + settings.units_ahead)
            waiting = len(run.retries) + max(0, window_end - next_new)
            self._kill_workers_past_their_stop_deadline()
            self._start_workers(waiting)
            for worker in list(self._shared.workers):
                if (
                    not worker.ready
                    or worker.retiring
                    or worker.connection_lost
                    or worker.unit_index is not None
                ):
                    continue
                if run.retries:
                    index = heapq.heappop(run.retries)
                elif next_new < window_end:
                    index = next_new
                    next_new += 1
                else:
                    break
                if not self._send_unit(worker, index, run):
                    heapq.heappush(run.retries, index)
            self._wait_and_handle(run)

    def _unit_failed(self, run: _Run, index: int, pid: int | None, cause: str) -> None:
        """Record a failed attempt; queue the unit again or stop the run."""
        label = self._label(run.units, index)
        run.causes[index].append(cause)
        with self._shared.lock:
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

    def _stop_the_run(
        self, run: _Run, index: int, pid: int | None, summary: str, text: str | None
    ) -> None:
        """Stop at once on a failure the same inputs would repeat."""
        label = self._label(run.units, index)
        run.causes[index].append(summary)
        with self._shared.lock:
            self._failed_attempts.append(
                FailedAttempt(index, label, run.attempts[index], pid, summary)
            )
        error = WorkerPoolUnitError(
            f"{label} failed in worker {pid}: {summary}",
            unit_index=index,
            label=label,
            causes=run.causes[index],
        )
        if text is None:
            raise error
        raise error from _RemoteTracebackError(text)

    def _start_workers(self, waiting: int) -> None:
        available = sum(
            1
            for worker in self._shared.workers
            if not worker.retiring
            and not worker.connection_lost
            and worker.unit_index is None
        )
        while (
            available < waiting and len(self._shared.workers) < self._settings.workers
        ):
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
        # Not under the state's lock: with spawn, ``start`` blocks until the
        # worker has read its start-up data, which can take as long as the
        # worker's imports, and the memory thread needs the lock meanwhile.
        try:
            with _environment_for_start(self._settings.worker_environment):
                process.start()
        except Exception as error:
            parent_end.close()
            raise WorkerPoolError(f"A worker could not be started: {error}") from error
        finally:
            child_end.close()
        pidfd = None
        if hasattr(os, "pidfd_open"):
            try:
                pidfd = os.pidfd_open(process.pid)
            except OSError:
                pidfd = None
        worker = _Worker(
            process=process, connection=parent_end, pid=process.pid, pidfd=pidfd
        )
        with self._shared.lock:
            self._shared.workers.append(worker)
            self._worker_starts += 1
            self._max_live_workers = max(
                self._max_live_workers, len(self._shared.workers)
            )

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
        worker.stop_deadline = time.monotonic() + _STOP_SECONDS
        try:
            worker.connection.send(None)
        except OSError:
            worker.connection_lost = True

    def _kill_workers_past_their_stop_deadline(self) -> None:
        now = time.monotonic()
        with self._shared.lock:
            for worker in self._shared.workers:
                if worker.stop_deadline is not None and now >= worker.stop_deadline:
                    _kill(worker)

    def _wait_and_handle(self, run: _Run) -> None:
        workers = list(self._shared.workers)
        timeout = _WAIT_SECONDS
        now = time.monotonic()
        for worker in workers:
            if worker.stop_deadline is not None:
                timeout = min(timeout, max(0.0, worker.stop_deadline - now))
        connections = [
            worker.connection for worker in workers if not worker.connection_lost
        ]
        sentinels = [worker.process.sentinel for worker in workers]
        ready = wait_for_any([*connections, *sentinels], timeout=timeout)
        ready_connections = [item for item in ready if not isinstance(item, int)]
        ready_sentinels = {item for item in ready if isinstance(item, int)}
        for worker in workers:
            if any(item is worker.connection for item in ready_connections):
                self._drain(worker, run)
        for worker in workers:
            if worker.process.sentinel in ready_sentinels:
                # Messages sent just before the worker exited come first.
                self._drain(worker, run)
                self._worker_exited(worker, self._reap(worker), run)

    def _reap(self, worker: _Worker) -> int | None:
        """The exit code of a worker whose process has exited, once reaped."""
        with self._shared.lock:
            worker.process.join(0.1)
            worker.exited = True
            return worker.process.exitcode

    def _drain(self, worker: _Worker, run: _Run) -> None:
        while not worker.connection_lost:
            try:
                if not worker.connection.poll():
                    return
                message = worker.connection.recv()
            except (EOFError, OSError):
                worker.connection_lost = True
                return
            except Exception as error:
                worker.connection_lost = True
                if worker.unit_index is None:
                    raise WorkerPoolError(
                        f"A message from worker {worker.pid} could not be read: "
                        f"{_summary(error)}"
                    ) from error
                self._stop_the_run(
                    run,
                    worker.unit_index,
                    worker.pid,
                    "it returned a result that could not be read in the parent: "
                    f"{_summary(error)}",
                    _formatted(error),
                )
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
            with self._shared.lock:
                self._units_completed += 1
            self._replace_if_over_cap(worker)
            limit = self._settings.max_units_per_worker
            if not worker.retiring and limit is not None and worker.units_run >= limit:
                self._send_stop(worker)
        elif kind == "receive_error":
            _, summary, text = message
            if worker.unit_index is None:
                raise WorkerPoolError(
                    f"Worker {worker.pid} could not read a message: {summary}"
                ) from _RemoteTracebackError(text)
            self._stop_the_run(
                run,
                worker.unit_index,
                worker.pid,
                f"the worker could not unpickle the unit: {summary}",
                text,
            )
        elif kind == "error":
            _, index, summary, text = message
            self._stop_the_run(run, index, worker.pid, summary, text)
        elif kind == "memory_error":
            _, index, summary, _text = message
            worker.unit_index = None
            worker.retiring = True
            worker.stop_deadline = time.monotonic() + _STOP_SECONDS
            self._unit_failed(run, index, worker.pid, summary)
        else:  # pragma: no cover - the worker sends only the kinds above
            raise WorkerPoolError(f"Worker {worker.pid} sent {kind!r}.")

    def _replace_if_over_cap(self, worker: _Worker) -> None:
        """Retire a worker over the cap now, before it is given another unit."""
        cap = self._settings.memory_cap_bytes
        if cap is None:
            return
        observed = process_memory_bytes(worker.pid)
        if observed is None or observed <= cap:
            return
        shared = self._shared
        with shared.lock:
            if shared.peak_memory is None or observed > shared.peak_memory:
                shared.peak_memory = observed
            if not worker.stopped_over_cap:
                worker.stopped_over_cap = True
                shared.workers_stopped_over_cap += 1
            worker.retiring = True
            worker.stop_deadline = time.monotonic()
            _kill(worker)

    def _worker_exited(self, worker: _Worker, exitcode: int | None, run: _Run) -> None:
        pid = worker.pid
        with self._shared.lock:
            overrun = self._shared.cap_overruns.pop(pid, None)
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
        with self._shared.lock:
            if worker in self._shared.workers:
                self._shared.workers.remove(worker)
            worker.exited = True
            self._shared.cap_overruns.pop(worker.pid, None)
            if worker.pidfd is not None:
                os.close(worker.pidfd)
                worker.pidfd = None
            try:
                worker.process.join(0)
                if worker.process.exitcode is not None:
                    worker.process.close()
                elif _has_exited(worker):
                    # Something else collected its exit status (for example a
                    # SIGCHLD handler set to SIG_IGN), so multiprocessing can
                    # never learn it and would list the process as running
                    # for ever.
                    children = getattr(multiprocessing.process, "_children", None)
                    if isinstance(children, set):
                        children.discard(worker.process)
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
            target=_watch_memory,
            args=(
                self._shared,
                self._settings.memory_cap_bytes,
                self._settings.memory_poll_seconds,
            ),
            name="worker-pool-memory",
            daemon=True,
        )
        self._monitor.start()
