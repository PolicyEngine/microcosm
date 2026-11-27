"""The units, results and worker entry points of pooled US post-export scoring.

The release tool (``tools/build_us_fiscal_refresh_release.py``) scores the
written release H5 in household batches. With pooled scoring it runs every
baseline pass in its own process, then sends the reform passes to worker
processes of the shared pool (:mod:`microcosm.build.worker_pool`) and runs
each post-export consumer again in its own process over the results:

- the engine-free dry run records every request each consumer makes
  (:mod:`microcosm.build.us_runtime.post_export_requests`), and which of a
  reformed simulation's requests the serial scorer scores together in one
  pass;
- each :class:`PostExportUnit` is every pass of one reformed simulation over
  the H5's household batches (a baseline unit is refused: each worker's
  memory cap is sized for reform passes);
- a worker reads each batch from the file the release tool's scorer wrote it
  to, builds one engine per batch, scores the pass's keys on it in ascending
  period order, applies the same refusals as the serial scorer, and returns
  each key's values and weights per batch (:class:`PostExportPassValues`);
- the release tool joins each key's parts in batch order, as the serial scorer
  does, and serves them to the consumer it runs again.

A failure inside a pass is returned as data (:class:`PostExportUnitFailure`),
not raised, so the pool keeps running and the release tool raises it where the
serial scorer would have: when the consumer asks for that pass.
:exc:`MemoryError` is raised, so the pool runs the unit again in a fresh
worker.

The types live here, in an importable package, so a spawned worker can
unpickle them however the release tool was loaded. The worker entry points
(:func:`start_post_export_worker`, :func:`score_post_export_unit`) live here
for the same reason; the worker finds the release tool, or loads it from its
file, when it starts (:func:`release_tool`).

:func:`plan_post_export_workers` chooses how many workers pooled scoring
starts and the limits each runs under, from the machine and the operator's
``MICROCOSM_POST_EXPORT_WORKERS`` (tracker decision 1.4 · 6, approved by
Anthony on 27 November 2026, until task 2.6 · 2's planner replaces it).
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import os
import sys
import traceback
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microcosm.build import host_resources
from microcosm.build.us_runtime.post_export_requests import (
    PostExportReform,
    check_post_export_reform,
)
from microcosm.build.worker_pool import WorkerPoolSettings

__all__ = [
    "MEMORY_BUDGET_FRACTION",
    "PARENT_PEAK_BYTES",
    "POST_EXPORT_WORKERS_VARIABLE",
    "UNITS_PER_WORKER",
    "WORKER_PEAK_BYTES",
    "PostExportPass",
    "PostExportPassTiming",
    "PostExportPassValues",
    "PostExportUnit",
    "PostExportUnitFailure",
    "PostExportUnitResult",
    "PostExportWorkerPlan",
    "PostExportWorkersError",
    "plan_post_export_workers",
    "release_tool",
    "score_post_export_unit",
    "start_post_export_worker",
]

#: One scored request: (variable, period, map_to), the release tool's
#: ``PostExportKey``.
type PostExportKey = tuple[str, int, str | None]

#: Where a pass failed: building the reform's rules, the check that the reform
#: keeps the behavioral-response parameters at baseline, or scoring a batch.
FAILURE_STAGES = ("rules", "responses", "scoring")


def _check_key(key: object) -> None:
    if (
        not isinstance(key, tuple)
        or len(key) != 3
        or not isinstance(key[0], str)
        or not key[0]
        or isinstance(key[1], bool)
        or not isinstance(key[1], int)
        or not (key[2] is None or (isinstance(key[2], str) and key[2]))
    ):
        raise ValueError(
            "A post-export key is (variable, period, map_to): a variable name, "
            f"an integer period and an entity name or None; got {key!r}."
        )


@dataclass(frozen=True)
class PostExportPass:
    """One batch-outer pass: each batch engine scores ``keys`` in this order.

    Attributes:
        label: The pass's name in errors and progress, as the serial scorer
            names it (``"<consumer> reform <variable>@<period>, ..."``).
        keys: Unique keys in ascending period order.
    """

    label: str
    keys: tuple[PostExportKey, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("A post-export pass needs a label.")
        keys = tuple(self.keys)
        if not keys:
            raise ValueError(f"Post-export pass {self.label!r} has no keys.")
        for key in keys:
            _check_key(key)
        if len(set(keys)) != len(keys):
            raise ValueError(f"Post-export pass {self.label!r} repeats a key.")
        periods = [key[1] for key in keys]
        if periods != sorted(periods):
            raise ValueError(
                f"Post-export pass {self.label!r} must hold its keys in "
                "ascending period order; each engine refuses an earlier period "
                "after a later one."
            )
        object.__setattr__(self, "keys", keys)


@dataclass(frozen=True)
class PostExportUnit:
    """A reformed simulation's passes, which a worker runs over the household
    batches ``[batch_start, batch_stop)``.

    Baseline passes run in the certifying process, never in a worker (a
    worker's memory cap is sized for reform passes; in the 5% export probe a
    baseline pass over one 2,000-household batch went over it), so a unit
    without a reform is refused.

    Attributes:
        consumer: The consumer whose passes these are.
        reformed_simulation: Which of the consumer's ``simulate(reform)``
            calls made the simulation, numbered from 0.
        reform: The description the worker rebuilds the reform from
            (:func:`~microcosm.build.us_runtime.post_export_requests.build_post_export_reform`).
            The worker builds the reform's rules once and scores every pass
            with them, as the serial scorer does for one reformed simulation.
        passes: The passes, in the order the serial scorer runs them.
        batch_start: The first batch, numbered from 0.
        batch_stop: One past the last batch.
        n_batches: The H5's batch count. A pass over more than one batch of
            the H5 refuses the formulas that aggregate over their whole
            simulation, whatever part of the H5 this unit scores.
    """

    consumer: str
    reformed_simulation: int
    reform: PostExportReform
    passes: tuple[PostExportPass, ...]
    batch_start: int
    batch_stop: int
    n_batches: int

    # Not hashable: most reform descriptions hold dictionaries.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        passes = tuple(self.passes)
        if not passes or not all(isinstance(item, PostExportPass) for item in passes):
            raise ValueError("A post-export unit needs one or more PostExportPass.")
        object.__setattr__(self, "passes", passes)
        for name in ("batch_start", "batch_stop", "n_batches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"PostExportUnit.{name} must be an integer.")
        if not 0 <= self.batch_start < self.batch_stop <= self.n_batches:
            raise ValueError(
                "A post-export unit's batches must be a non-empty range within "
                f"the H5's {self.n_batches}; got [{self.batch_start}, "
                f"{self.batch_stop})."
            )
        if self.consumer is not None and (
            not isinstance(self.consumer, str) or not self.consumer
        ):
            raise ValueError("PostExportUnit.consumer must be a name.")
        if self.reform is None:
            raise ValueError(
                "A post-export pool unit needs a reform: baseline passes run in "
                "the certifying process, never in a worker."
            )
        check_post_export_reform(self.reform)
        if self.consumer is None:
            raise ValueError("A reform unit names the consumer that asked for it.")
        simulation = self.reformed_simulation
        if isinstance(simulation, bool) or not isinstance(simulation, int):
            raise TypeError("A reform unit names its reformed simulation's number.")
        if simulation < 0:
            raise ValueError("A reformed simulation's number is 0 or more.")

    def describe(self) -> str:
        """The unit's name in pool errors."""
        if self.batch_stop - self.batch_start == self.n_batches:
            batches = f"all {self.n_batches} batch(es)"
        else:
            batches = (
                f"batches {self.batch_start + 1}-{self.batch_stop} of {self.n_batches}"
            )
        return (
            f"post-export {self.consumer} reformed simulation "
            f"{self.reformed_simulation} ({len(self.passes)} pass(es), first "
            f"{self.passes[0].label!r}), {batches}"
        )


@dataclass(frozen=True)
class PostExportPassValues:
    """One pass's values: for each key of the pass, in order, one
    ``(values, weights)`` pair per batch of the unit, in batch order."""

    parts: tuple[tuple[tuple[Any, Any], ...], ...]


@dataclass(frozen=True)
class PostExportUnitFailure:
    """Why a pass of a unit has no values.

    Attributes:
        stage: Where it failed (:data:`FAILURE_STAGES`).
        error_name: The exception's class name.
        error_module: The module of the exception's class.
        builtin_base: The first built-in exception class among the class's
            bases (``KeyError``, ``RuntimeError``, ``Exception``, ...).
        message: ``str()`` of the exception.
        traceback: The exception's traceback, formatted in the worker.
        batch_file_error: Whether it is the release tool's
            ``PostExportBatchFileError`` (a batch file that changed or cannot
            be read), which the release tool raises as its own.
    """

    stage: str
    error_name: str
    error_module: str
    builtin_base: str
    message: str
    traceback: str
    batch_file_error: bool = False

    @classmethod
    def from_exception(
        cls, stage: str, error: BaseException, *, batch_file_error: bool = False
    ) -> PostExportUnitFailure:
        if stage not in FAILURE_STAGES:
            raise ValueError(f"Unknown post-export failure stage {stage!r}.")
        base = next(
            kind
            for kind in type(error).__mro__
            if kind.__module__ == "builtins" and issubclass(kind, BaseException)
        )
        try:
            message = str(error)
        except Exception:  # an exception whose str raises
            message = f"<str() of the {type(error).__name__} raised>"
        return cls(
            stage=stage,
            error_name=type(error).__name__,
            error_module=type(error).__module__,
            builtin_base=base.__name__,
            message=message,
            traceback="".join(traceback.format_exception(error)),
            batch_file_error=batch_file_error,
        )

    def exception(self) -> BaseException:
        """An exception with the worker's exception's class name and ``str()``.

        Its class is made here, named as the worker's and derived from the
        built-in exception class the worker's derived from, so ``except
        KeyError`` still catches a ``KeyError`` and a report that records
        ``f"{type(error).__name__}: {error}"`` records what the serial scorer
        would have. The worker's traceback is the exception's
        ``worker_traceback``.
        """
        base = getattr(builtins, self.builtin_base, None)
        if not (isinstance(base, type) and issubclass(base, Exception)):
            base = Exception
        message = self.message

        def describe(error) -> str:
            return message

        kind = type(
            self.error_name,
            (base,),
            {"__module__": self.error_module, "__str__": describe},
        )
        try:
            error = kind(message)
        except Exception:
            # A built-in base that does not take one message argument.
            error = type(self.error_name, (Exception,), {"__str__": describe})(message)
        error.worker_traceback = self.traceback
        return error


@dataclass(frozen=True)
class PostExportPassTiming:
    """What one pass cost its worker.

    Attributes:
        wall_seconds: Wall time of the pass.
        cpu_seconds: The worker's CPU time during the pass.
        footprint_bytes: The worker's memory footprint when the pass ended
            (:func:`microcosm.build.worker_pool.process_memory_bytes`), or
            ``None`` where it cannot be read.
        peak_rss_bytes: The worker's peak resident memory since it started.
    """

    wall_seconds: float
    cpu_seconds: float
    footprint_bytes: int | None
    peak_rss_bytes: int


@dataclass(frozen=True)
class PostExportUnitResult:
    """A unit's outcome: per pass, its values or its failure, and its timing."""

    outcomes: tuple[PostExportPassValues | PostExportUnitFailure, ...]
    timings: tuple[PostExportPassTiming, ...]
    worker_pid: int


# ---------------------------------------------------------------------------
# The worker's entry points
# ---------------------------------------------------------------------------


def _source_sha256(module) -> str | None:
    """The SHA-256 of the bytes ``module`` ran, or of its file now."""
    recorded = getattr(module, "__loaded_sha256__", None)
    if recorded is not None:
        return str(recorded)
    try:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def _refuse_other_bytes(module, path: Path, sha256: str | None) -> None:
    if sha256 is None:
        return
    observed = _source_sha256(module)
    if observed != sha256:
        raise RuntimeError(
            f"The release tool at {path} has SHA-256 {observed}; the scoring "
            f"process runs {sha256}. A worker scores only with the code the "
            "scoring process runs, so a tool file changed while scoring ran is "
            "refused."
        )


def release_tool(module_name: str, path: str | Path, *, sha256: str | None = None):
    """The release tool in this process: loaded already, or from ``path``.

    A worker spawned from a release run has run the parent's main script again
    as ``__mp_main__``. That script is either the release tool itself or
    certification's script, which imports the tool, so the tool is usually
    loaded already; it is loaded from ``path`` only when it is not (the
    parent loaded it by path, as the tests and the export probe do). Either
    way the tool's module body runs once per worker.

    With ``sha256``, the parent's digest of the tool's bytes, a tool whose
    bytes differ is refused: the bytes this function executes, or the file of
    a tool found loaded already. (The microcosm packages the tool imports are
    not checked.)
    """
    path = Path(path).resolve()
    for candidate in (
        sys.modules.get(module_name),
        sys.modules.get("__mp_main__"),
        sys.modules.get("__main__"),
    ):
        file = getattr(candidate, "__file__", None)
        if file is not None and Path(file).resolve() == path:
            _refuse_other_bytes(candidate, path, sha256)
            return candidate
    name = module_name
    if name in {"__main__", "__mp_main__"} or name in sys.modules:
        name = f"_post_export_release_tool_{zlib.crc32(str(path).encode()):08x}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the release tool from {path}")
    module = importlib.util.module_from_spec(spec)
    # Execute the bytes that are checked, not a second read of the file.
    source = path.read_bytes()
    module.__loaded_sha256__ = hashlib.sha256(source).hexdigest()
    _refuse_other_bytes(module, path, sha256)
    # Registered before it runs, as ``dataclasses`` resolves annotations
    # through ``sys.modules``.
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def start_post_export_worker(
    module_name: str, path: str, sha256: str | None, *arguments: Any
):
    """The pool's initializer: the release tool's worker over its batch files."""
    return release_tool(module_name, path, sha256=sha256)._PostExportWorker(*arguments)


def score_post_export_unit(worker, unit: PostExportUnit) -> PostExportUnitResult:
    """The pool's unit function: score ``unit`` (``worker.score``)."""
    return worker.score(unit)


# ---------------------------------------------------------------------------
# How many workers, and their limits (tracker decision 1.4 · 6)
# ---------------------------------------------------------------------------

_GIB = 2**30

#: The variable an operator sets: unset, blank or ``auto`` for the formula, a
#: positive whole number for that many pool workers, or ``serial`` to score in
#: the certifying process alone.
POST_EXPORT_WORKERS_VARIABLE = "MICROCOSM_POST_EXPORT_WORKERS"

#: The share of the machine's memory a build may plan to use (decision 1.1 · 2).
MEMORY_BUDGET_FRACTION = 0.75

#: The certifying process's planned peak while workers score, after its
#: baseline passes: the release tool's imports (about 1.1-1.2 GiB) and the
#: merged baseline arrays it holds (up to about 1.1 GiB at national size).
#: Not measured: what its baseline passes leave resident (freed memory is not
#: always returned to the system), and, after a failed merged pass, every
#: stage's own baseline arrays, held at once (a key several stages plan is
#: held once per stage). The execution record reports the process's footprint
#: when the workers start.
PARENT_PEAK_BYTES = 3 * _GIB

#: One worker's planned peak over its replacement interval (1.4 · 6's one-batch
#: check: about 3.5-4.7 GiB over two reform passes).
WORKER_PEAK_BYTES = 5 * _GIB

#: Units a worker runs before a fresh worker takes its place. A reform unit is
#: one reformed simulation's passes (one pass for every shipped consumer).
UNITS_PER_WORKER = 2


class PostExportWorkersError(ValueError):
    """``MICROCOSM_POST_EXPORT_WORKERS`` is not a setting, or the machine is
    too small for one worker and the operator did not say to run anyway."""


def _gib(count: int | float) -> str:
    return f"{count / _GIB:.2f} GiB"


@dataclass(frozen=True)
class PostExportWorkerPlan:
    """How post-export scoring runs on this machine.

    Attributes:
        serial: Score in the certifying process alone (the operator's
            ``serial``).
        workers: The most pool workers; ``None`` when serial. The scorer
            starts no more than a pool run has units.
        chosen_by: ``"formula"``, or ``MICROCOSM_POST_EXPORT_WORKERS=<value>``.
        total_memory_bytes: The machine's or container's memory.
        usable_cores: The cores this process may use.
        memory_cap_bytes: Each worker's memory-footprint cap; ``None`` when
            serial.
        max_units_per_worker: Units a worker runs before it is replaced;
            ``None`` when serial.
    """

    serial: bool
    workers: int | None
    chosen_by: str
    total_memory_bytes: int
    usable_cores: int
    memory_cap_bytes: int | None
    max_units_per_worker: int | None

    @property
    def budget_bytes(self) -> int:
        """75% of the memory (decision 1.1 · 2), in bytes."""
        return self.total_memory_bytes * 3 // 4

    @property
    def planned_peak_bytes(self) -> int | None:
        """This process's planned peak plus every worker's."""
        if self.workers is None:
            return None
        return PARENT_PEAK_BYTES + self.workers * WORKER_PEAK_BYTES

    def pool_settings(self) -> WorkerPoolSettings | None:
        """The shared pool's settings; ``None`` when serial. Workers inherit
        this process's environment, thread variables included."""
        if self.serial:
            return None
        return WorkerPoolSettings(
            workers=self.workers,
            memory_cap_bytes=self.memory_cap_bytes,
            max_units_per_worker=self.max_units_per_worker,
        )

    def record(self) -> dict[str, object]:
        """The plan as run metadata, for the manifests' scoring block."""
        return {
            "mode": "serial" if self.serial else "pooled",
            "chosen_by": self.chosen_by,
            "workers": self.workers,
            "total_memory_bytes": self.total_memory_bytes,
            "usable_cores": self.usable_cores,
            "budget_bytes": self.budget_bytes,
            "planned_peak_bytes": self.planned_peak_bytes,
            "memory_cap_bytes": self.memory_cap_bytes,
            "max_units_per_worker": self.max_units_per_worker,
        }

    def describe(self) -> str:
        """The line certification and the export probe print."""
        machine = (
            f"{self.usable_cores} usable core(s), {_gib(self.total_memory_bytes)} "
            "of memory"
        )
        if self.serial:
            return (
                f"Post-export scoring runs in this process alone ({self.chosen_by}; "
                f"{machine})."
            )
        planned, budget = self.planned_peak_bytes, self.budget_bytes
        against = "above" if planned > budget else "within"
        return (
            f"Post-export scoring: up to {self.workers} pool worker"
            f"{'' if self.workers == 1 else 's'} ({self.chosen_by}; {machine}). "
            f"Planned peak {_gib(planned)} ({_gib(PARENT_PEAK_BYTES)} for this "
            f"process, {_gib(WORKER_PEAK_BYTES)} per worker), {against} the "
            f"budget of {_gib(budget)} (75% of memory). Each worker is capped at "
            f"{_gib(self.memory_cap_bytes)} and replaced after "
            f"{self.max_units_per_worker} units."
        )


def _requested_workers(value: str) -> int | str | None:
    """``None`` for the formula, ``"serial"``, or a count."""
    if value in ("", "auto"):
        return None
    if value == "serial":
        return "serial"
    if value.isascii() and value.isdigit():
        try:
            count = int(value)
        except ValueError:  # longer than int() will parse
            count = 0
        if count > 0:
            return count
    raise PostExportWorkersError(
        f"{POST_EXPORT_WORKERS_VARIABLE} must be auto, serial or a positive whole "
        f"number of post-export scoring workers, got {value!r}."
    )


def plan_post_export_workers(
    environment: Mapping[str, str] = os.environ,
    *,
    total_memory_bytes: int | None = None,
    usable_cores: int | None = None,
) -> PostExportWorkerPlan:
    """Choose the post-export pool's workers and limits for this machine.

    - Workers: ``min(usable cores - 1, floor((0.75 * memory - 3 GiB) / 5 GiB))``,
      at least one, unless ``MICROCOSM_POST_EXPORT_WORKERS`` names a count
      (``1`` is one pool worker, a process of its own) or ``serial``.
    - A memory term below one means no room for this process and one worker
      within the budget, which is refused unless the variable names a count
      or ``serial``.
    - Each worker is capped at ``max(5 GiB, (0.75 * memory - 3 GiB) /
      workers)`` of memory footprint and replaced after
      :data:`UNITS_PER_WORKER` units; it inherits this process's thread
      variables.

    ``total_memory_bytes`` and ``usable_cores`` replace the machine's readings
    (:mod:`microcosm.build.host_resources`), for tests.

    Raises:
        PostExportWorkersError: For a value that is not a setting, or a
            machine too small for one worker.
    """
    requested = _requested_workers(environment.get(POST_EXPORT_WORKERS_VARIABLE, ""))
    memory = int(
        total_memory_bytes
        if total_memory_bytes is not None
        else host_resources.total_memory_bytes()
    )
    cores = int(
        usable_cores if usable_cores is not None else host_resources.usable_cores()
    )
    room = memory * 3 // 4 - PARENT_PEAK_BYTES
    if requested == "serial":
        return PostExportWorkerPlan(
            serial=True,
            workers=None,
            chosen_by=f"{POST_EXPORT_WORKERS_VARIABLE}=serial",
            total_memory_bytes=memory,
            usable_cores=cores,
            memory_cap_bytes=None,
            max_units_per_worker=None,
        )
    if requested is None:
        memory_term = room // WORKER_PEAK_BYTES
        if memory_term < 1:
            raise PostExportWorkersError(
                f"There is no room for post-export scoring on this machine: "
                f"this process and one pool worker need "
                f"{_gib(PARENT_PEAK_BYTES + WORKER_PEAK_BYTES)} within the memory "
                f"budget, 75% of {_gib(memory)}, which is {_gib(memory * 3 // 4)}. "
                f"Set {POST_EXPORT_WORKERS_VARIABLE} to a number of workers, or "
                "to serial to score in this process alone, to run it anyway."
            )
        workers = min(max(1, cores - 1), memory_term)
        chosen_by = "formula"
    else:
        workers = requested
        chosen_by = f"{POST_EXPORT_WORKERS_VARIABLE}={requested}"
    return PostExportWorkerPlan(
        serial=False,
        workers=workers,
        chosen_by=chosen_by,
        total_memory_bytes=memory,
        usable_cores=cores,
        memory_cap_bytes=max(WORKER_PEAK_BYTES, room // workers),
        max_units_per_worker=UNITS_PER_WORKER,
    )
