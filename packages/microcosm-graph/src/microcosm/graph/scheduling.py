"""Concurrent kernel execution behind canonical admission (amendment 26).

The executor admits nodes one at a time, in ``compiled.order``. Admission is
everything that touches shared state: validating a result, patching the
cumulative population, invoking the population observer, writing the store
and recording the receipt. None of that moves here, and none of it becomes
concurrent.

What this module adds is the one step that does not touch shared state: a
kernel's ``run(context)``. Once every declared predecessor of a node has been
admitted, the coordinator can project that node's context and hand the kernel
call to a worker thread, ahead of the node's canonical turn. When the turn
arrives, the executor's unchanged sequential path decides whether that
precomputed call is the call it would have made:

* the population the context was projected from is the same object the turn
  reads, and the writer-derived numeric scopes are equal -- then the context
  is the turn's context, and is reused as it stands; or
* otherwise the turn projects its own context, and the precomputed outcome is
  used only when the two context digests are equal. A different digest
  discards it and the turn runs the kernel on its own context.

Either way the kernel's result enters the same validation, patch,
observation, persistence and receipt code, in the same order, as a sequential
run. Node keys, receipts, cache records and every stored byte are therefore
those of the sequential run, provided each kernel honours the contract it
already declares: its output is a function of its projected context, its
declared sources and its node-key seed, and nothing else. Running kernels on
worker threads adds one obligation that is the caller's, like amendment 25's
live-observer promise: a registered kernel instance must be safe to call from a
worker thread while other kernels run. The executor cannot check that.

Worker threads never publish to the store, never touch the cumulative
populations, and are joined before ``run_graph`` settles a failure or
performs its run-end source pass. Nothing about scheduling enters a key, a
receipt, a cache record or the portable manifest.
"""

from __future__ import annotations

import contextvars
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from .kernel import KernelContext, KernelRole

__all__ = [
    "MAX_WORKERS_ENVIRONMENT",
    "THREAD_NAME_PREFIX",
    "KernelOutcome",
    "PreparedKernel",
    "Speculation",
    "invoke_kernel",
    "resolve_max_workers",
]

#: Read when ``run_graph(max_workers=None)``: lets a country runtime run its
#: nested graphs concurrently without threading a parameter through source
#: modules whose bytes are pinned into kernel implementation hashes.
MAX_WORKERS_ENVIRONMENT = "MICROCOSM_GRAPH_MAX_WORKERS"

#: Worker thread names start with this, so a test or a caller can prove that no
#: worker outlives the run that started it.
THREAD_NAME_PREFIX = "microcosm-graph-worker"

_DECIMAL = re.compile(r"[0-9]+")


def resolve_max_workers(max_workers: object) -> int:
    """The worker count one ``run_graph`` call uses.

    An explicit ``max_workers`` wins. ``None`` reads
    :data:`MAX_WORKERS_ENVIRONMENT`; unset or blank means 1, the sequential
    executor. Anything else that is not a positive integer is refused before
    the run touches its store.
    """

    if max_workers is None:
        raw = os.environ.get(MAX_WORKERS_ENVIRONMENT, "").strip()
        if not raw:
            return 1
        if _DECIMAL.fullmatch(raw) is None:
            raise ValueError(
                f"{MAX_WORKERS_ENVIRONMENT}={raw!r} must be a positive decimal integer."
            )
        value = int(raw)
        label = f"{MAX_WORKERS_ENVIRONMENT}={raw!r}"
    else:
        if type(max_workers) is not int:
            raise TypeError("max_workers must be an int or None.")
        value = max_workers
        label = f"max_workers={max_workers!r}"
    if value < 1:
        raise ValueError(f"{label} must be at least 1.")
    return value


@dataclass
class KernelOutcome:
    """What one ``kernel.run`` call produced, carried from where it ran.

    ``error`` is an ordinary exception the kernel raised; the executor decides
    what it means (a failed gate verdict or a refusal) exactly as it does for
    an inline call. ``after`` is the context digest taken immediately after
    the call, on the same context object the kernel received, and is skipped
    in the one case the sequential path skips it: a non-gate kernel that
    raised, which is refused before any digest is compared.
    """

    result: object = None
    error: Exception | None = None
    after: bytes | None = None
    after_error: Exception | None = None
    started: float = 0.0
    finished: float = 0.0
    thread: str = ""


def invoke_kernel(
    kernel: object,
    context: KernelContext,
    digest: Callable[[KernelContext], bytes],
) -> KernelOutcome:
    """Run one kernel on one context and digest that context afterwards."""

    outcome = KernelOutcome(
        started=time.perf_counter(), thread=threading.current_thread().name
    )
    try:
        outcome.result = kernel.run(context)  # type: ignore[attr-defined]
    except Exception as error:
        outcome.error = error
    outcome.finished = time.perf_counter()
    role = kernel.capabilities.role  # type: ignore[attr-defined]
    if outcome.error is None or role is KernelRole.GATE:
        try:
            outcome.after = digest(context)
        except Exception as error:
            outcome.after_error = error
    return outcome


@dataclass(eq=False)
class PreparedKernel:
    """A node's context, projected once its predecessors were all admitted.

    ``incumbent``, ``tolerances`` and ``numerics`` are what the projection
    read that its node's canonical turn could see differently; the turn
    compares them before reusing ``context`` as its own.
    """

    node_id: str
    kernel: object
    context: KernelContext
    before: bytes
    incumbent: object
    tolerances: Mapping[object, object]
    numerics: Mapping[object, object]
    future: Future[KernelOutcome] | None = field(default=None, repr=False)


class Speculation:
    """A bounded thread pool that runs ready kernels ahead of admission.

    ``max_workers`` is the pool's thread count, so it bounds the kernel calls
    running at once; the coordinator thread runs none while this is active.
    It also bounds how many prepared contexts may wait for their turn with
    their results, so memory beyond the sequential run's is at most
    ``max_workers`` early contexts and their results, plus any discarded call
    still finishing on its worker -- and whatever the kernels themselves
    allocate, which no worker count can bound.

    Every method runs on the coordinator thread. Workers run only
    :func:`invoke_kernel`.
    """

    def __init__(
        self,
        max_workers: int,
        *,
        order: Sequence[str],
        predecessors: Mapping[str, Sequence[str]],
        admitted: Mapping[str, object],
        prepare: Callable[[str], PreparedKernel | None],
        digest: Callable[[KernelContext], bytes],
        record: MutableMapping[str, object],
    ) -> None:
        if type(max_workers) is not int or max_workers < 2:
            raise ValueError("Speculation needs at least two workers.")
        self.max_workers = max_workers
        self._order = tuple(order)
        self._predecessors = predecessors
        self._admitted = admitted
        self._prepare = prepare
        self._digest = digest
        self._record = record
        self._timings: dict[str, dict[str, object]] = record.setdefault(  # type: ignore[assignment]
            "kernels", {}
        )
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=THREAD_NAME_PREFIX
        )
        self._prepared: dict[str, PreparedKernel] = {}
        self._seen: set[str] = set()
        self._running: set[Future[KernelOutcome]] = set()
        self._position = 0

    # -- bookkeeping -----------------------------------------------------

    def _count(self, name: str, increment: int = 1) -> None:
        self._record[name] = int(self._record.get(name, 0)) + increment  # type: ignore[arg-type]

    def _busy(self) -> int:
        self._running = {future for future in self._running if not future.done()}
        return len(self._running)

    def _submit(self, kernel: object, context: KernelContext) -> Future[KernelOutcome]:
        # Each call runs in a copy of the coordinator's context variables, so a
        # kernel that reads a scope its caller opened sees it on a worker too.
        snapshot = contextvars.copy_context()
        future = self._pool.submit(
            snapshot.run, invoke_kernel, kernel, context, self._digest
        )
        self._running.add(future)
        busy = len(self._running)
        if busy > int(self._record.get("peak_submitted", 0)):  # type: ignore[arg-type]
            self._record["peak_submitted"] = busy
        return future

    # -- dispatch --------------------------------------------------------

    def fill(self, position: int) -> None:
        """Dispatch ready, not yet admitted nodes from ``position`` onward.

        ``position`` is the canonical head. The scan runs in canonical order,
        so the head, whose predecessors are always admitted, is offered the
        first free worker. A node is ready when every declared predecessor has
        a receipt; it is then offered once. A node whose preparation declines
        (a probable cache hit, an unreached node, or anything that raised) is
        not offered again: its turn does the work exactly as it would without
        this scheduler, including raising whatever preparation raised.
        """

        self._position = position
        for node_id in self._order[position:]:
            if (
                self._busy() >= self.max_workers
                or len(self._prepared) >= self.max_workers
            ):
                return
            if node_id in self._seen or node_id in self._admitted:
                continue
            if any(
                parent not in self._admitted for parent in self._predecessors[node_id]
            ):
                continue
            self._seen.add(node_id)
            try:
                prepared = self._prepare(node_id)
            except Exception:
                prepared = None
            if prepared is None:
                self._count("declined")
                continue
            prepared.future = self._submit(prepared.kernel, prepared.context)
            self._prepared[node_id] = prepared
            self._count("dispatched")

    def claim(self, node_id: str) -> PreparedKernel | None:
        """Hand a node's prepared call to its canonical turn, if there is one."""

        self._seen.add(node_id)
        return self._prepared.pop(node_id, None)

    def outcome(self, prepared: PreparedKernel, *, reused: bool) -> KernelOutcome:
        """The prepared call's outcome, once its turn has accepted its context."""

        self._count("reused_context" if reused else "verified_context")
        assert prepared.future is not None
        return self._await(prepared.node_id, prepared.future, "speculative")

    def discard(self, prepared: PreparedKernel) -> None:
        """Drop a prepared call whose context is not the turn's context.

        A call already running is left to finish on its worker and its result
        is never read; it still counts against ``max_workers`` until it ends.
        """

        self._count("discarded")
        assert prepared.future is not None
        prepared.future.cancel()

    def run(
        self, node_id: str, kernel: object, context: KernelContext
    ) -> KernelOutcome:
        """Run a turn's own context on a worker and wait for it."""

        self._seen.add(node_id)
        self._count("submitted_at_turn")
        return self._await(node_id, self._submit(kernel, context), "turn")

    def retire(self, node_id: str) -> None:
        """Forget a node its turn admitted without claiming its prepared call."""

        stale = self._prepared.pop(node_id, None)
        if stale is not None:
            self.discard(stale)

    def _await(
        self, node_id: str, future: Future[KernelOutcome], path: str
    ) -> KernelOutcome:
        # While the turn waits, every worker that frees up is offered the next
        # ready node, so the coordinator's wait is not the pool's idle time.
        while not future.done():
            wait(tuple(self._running | {future}), return_when=FIRST_COMPLETED)
            self.fill(self._position)
        outcome = future.result()
        self._timings[node_id] = {
            "path": path,
            "thread": outcome.thread,
            "started": outcome.started,
            "finished": outcome.finished,
        }
        return outcome

    def close(self) -> None:
        """Cancel every call not yet started and join every call that did.

        Called on every exit from the node loop, including by exception, and
        therefore before ``run_graph`` settles a failed run or performs its
        run-end source pass: no worker is left running when either reads the
        sources or the store's write ledger.
        """

        for prepared in self._prepared.values():
            if prepared.future is not None:
                prepared.future.cancel()
        self._prepared.clear()
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._running.clear()
