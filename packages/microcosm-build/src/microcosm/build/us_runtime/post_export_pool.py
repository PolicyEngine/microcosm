"""The units, results and worker entry points of pooled US post-export scoring.

The release tool (``tools/build_us_fiscal_refresh_release.py``) scores the
written release H5 in household batches. With pooled scoring it sends the
scoring to worker processes of the shared pool
(:mod:`microcosm.build.worker_pool`) and runs each post-export consumer again
in its own process over the results:

- the engine-free dry run records every request each consumer makes
  (:mod:`microcosm.build.us_runtime.post_export_requests`), and which of a
  reformed simulation's requests the serial scorer scores together in one
  pass;
- each :class:`PostExportUnit` is a run of one or more such passes over a range
  of the H5's household batches: the merged baseline plan over a range of
  batches, or every pass of one reformed simulation over all of them;
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
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microcosm.build.us_runtime.post_export_requests import (
    PostExportReform,
    check_post_export_reform,
)

__all__ = [
    "PostExportPass",
    "PostExportPassTiming",
    "PostExportPassValues",
    "PostExportUnit",
    "PostExportUnitFailure",
    "PostExportUnitResult",
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
            names it (``"merged baseline plan"``, ``"<consumer> baseline
            plan"``, ``"<consumer> reform <variable>@<period>, ..."``).
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
    """Passes a worker runs over the household batches ``[batch_start, batch_stop)``.

    Attributes:
        consumer: The consumer whose passes these are, or ``None`` for the
            merged baseline plan of every consumer.
        reformed_simulation: For a reform unit, which of the consumer's
            ``simulate(reform)`` calls made the simulation, numbered from 0;
            ``None`` for a baseline unit.
        reform: ``None`` for a baseline unit; otherwise the description the
            worker rebuilds the reform from
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

    consumer: str | None
    reformed_simulation: int | None
    reform: PostExportReform | None
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
            raise ValueError("PostExportUnit.consumer must be a name or None.")
        if self.reform is None:
            if self.reformed_simulation is not None:
                raise ValueError("A baseline unit names no reformed simulation.")
            return
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
        if self.reform is None:
            return f"post-export {self.passes[0].label}, {batches}"
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


def release_tool(module_name: str, path: str | Path):
    """The release tool in this process: loaded already, or from ``path``.

    A worker spawned from a release run has run the parent's main script again
    as ``__mp_main__``. That script is either the release tool itself or
    certification's script, which imports the tool, so the tool is usually
    loaded already; it is loaded from ``path`` only when it is not (the
    parent loaded it by path, as the tests and the export probe do). Either
    way the tool's module body runs once per worker.
    """
    path = Path(path).resolve()
    for candidate in (
        sys.modules.get(module_name),
        sys.modules.get("__mp_main__"),
        sys.modules.get("__main__"),
    ):
        file = getattr(candidate, "__file__", None)
        if file is not None and Path(file).resolve() == path:
            return candidate
    name = module_name
    if name in {"__main__", "__mp_main__"} or name in sys.modules:
        name = f"_post_export_release_tool_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the release tool from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, as ``dataclasses`` resolves annotations
    # through ``sys.modules``.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def start_post_export_worker(module_name: str, path: str, *arguments: Any):
    """The pool's initializer: the release tool's worker over its batch files."""
    return release_tool(module_name, path)._PostExportWorker(*arguments)


def score_post_export_unit(worker, unit: PostExportUnit) -> PostExportUnitResult:
    """The pool's unit function: score ``unit`` (``worker.score``)."""
    return worker.score(unit)
