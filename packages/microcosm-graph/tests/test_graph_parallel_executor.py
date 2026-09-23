"""Amendment 26: concurrent kernel workers behind canonical admission.

``run_graph(max_workers=N)`` lets a node whose declared predecessors are all
admitted have its kernel called on a worker thread ahead of its canonical
turn, while admission -- validation, patching, the observer, every store
write and every receipt -- stays on the calling thread in ``compiled.order``.

The claim these tests hold the executor to is byte identity with the
sequential run: the manifest key, every content-addressed node receipt,
every node key and seed, every attached population, and every relative path
and byte of the store's object tree, at every worker count and under seeded
perturbations of which worker finishes first. Failures are held to the same
standard: the exception and everything published before it are the
sequential run's, and a worker whose node is never admitted publishes
nothing. Scheduling itself is observable only through the private
``_concurrency_record`` and never enters a key, receipt or stored byte.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.graph.executor as graph_executor
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    NodeRejected,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.scheduling import (
    MAX_WORKERS_ENVIRONMENT,
    THREAD_NAME_PREFIX,
    PreparedKernel,
    Speculation,
    resolve_max_workers,
)

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

WORKER_COUNTS = (2, 3, 4, 8)


# ----------------------------------------------------------------------
# Probes: seeded pauses that reorder completions, and what they observed
# ----------------------------------------------------------------------


class Probe:
    """What the kernels of one run saw: overlap, threads, completion order."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.threads: set[str] = set()
        self.started: list[str] = []
        self.finished: list[str] = []

    @contextmanager
    def running(self, node_id: str):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(threading.current_thread().name)
            self.started.append(node_id)
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1
                self.finished.append(node_id)


class Jittered:
    """A registered kernel that pauses for a seeded, per-node time, then delegates.

    ``ref``, ``capabilities`` and ``implementation_hash`` are the wrapped
    kernel's, so node keys are exactly those of the unwrapped registry; only
    when each call finishes moves.
    """

    def __init__(
        self,
        inner: object,
        probe: Probe,
        delay: Callable[[str], float],
    ) -> None:
        self.ref = inner.ref  # type: ignore[attr-defined]
        self.capabilities = inner.capabilities  # type: ignore[attr-defined]
        self._inner = inner
        self._probe = probe
        self._delay = delay

    def implementation_hash(self) -> str:
        return self._inner.implementation_hash()  # type: ignore[attr-defined]

    def run(self, context: KernelContext) -> KernelResult:
        with self._probe.running(context.node.id):
            pause = self._delay(context.node.id)
            if pause:
                time.sleep(pause)
            return self._inner.run(context)  # type: ignore[attr-defined]


def seeded_delay(seed: int, *, scale: float = 0.012) -> Callable[[str], float]:
    """A pause per node id, drawn from ``seed`` alone: the same seed, the same run."""

    def delay(node_id: str) -> float:
        return random.Random(f"{seed}/{node_id}").uniform(0.0, scale)

    return delay


def jittered(
    registry: KernelRegistry, probe: Probe, delay: Callable[[str], float]
) -> KernelRegistry:
    wrapped = KernelRegistry()
    for kernel in registry.as_mapping().values():
        wrapped.register(Jittered(kernel, probe, delay))
    return wrapped


# ----------------------------------------------------------------------
# Test-local kernels
# ----------------------------------------------------------------------


class FnKernel(KernelBase):
    """A test kernel around a plain function; its hash is this module's source."""

    def __init__(
        self,
        ref: str,
        capabilities: Capabilities,
        fn: Callable[[KernelContext], KernelResult],
    ) -> None:
        self.ref = ref
        self.capabilities = capabilities
        self._fn = fn
        self._lock = threading.Lock()
        self.calls = 0

    def run(self, context: KernelContext) -> KernelResult:
        with self._lock:
            self.calls += 1
        return self._fn(context)


_PLAIN = Capabilities(determinism=Determinism.DETERMINISTIC)
FOREST = ArtifactType("qrf.forest", 1)


def _person_series(context: KernelContext, values: np.ndarray) -> pd.Series:
    table = context.tables["person"]
    return pd.Series(
        values,
        index=pd.Index(table["person_id"], name="person_id"),
        dtype="float64",
    )


def _fit(context: KernelContext) -> KernelResult:
    ages = context.tables["person"]["age"].to_numpy(dtype=np.float64)
    payload = canonical_json({"mean_age": float(ages.mean()), "n": int(len(ages))})
    return KernelResult(
        columns={("person", "fitted"): _person_series(context, ages * 0.5)},
        artifacts={"forest": payload},
        receipt={"trees": 3},
    )


def _consume(context: KernelContext) -> KernelResult:
    donor = context.artifacts["donor"]
    table = context.tables["person"]
    drawn = context.rng.uniform(size=len(table)) * float(len(donor.payload))
    return KernelResult(
        columns={("person", "drawn"): _person_series(context, drawn)},
        receipt={"payload_bytes": len(donor.payload)},
    )


def artifact_registry() -> KernelRegistry:
    registry = toy.toy_registry()
    registry.register(FnKernel("fit.forest@1", _PLAIN, _fit))
    registry.register(
        FnKernel(
            "consume.forest@1",
            Capabilities(
                determinism=Determinism.SEEDED,
                numeric=Numeric.BITWISE,
                seed_source=SeedSource.EXECUTOR,
            ),
            _consume,
        )
    )
    return registry


# ----------------------------------------------------------------------
# The graphs: the acceptance suite's own toy graphs, plus wider ones
# ----------------------------------------------------------------------


def _tolerance_gate(node_id: str, column: str, *, population: str = "survey") -> Node:
    verdict = f"{node_id}_verdict"
    return Node(
        node_id,
        "gate.tolerance@1",
        inputs=(Slice("person", (column,)),),
        outputs=(Owned("release", verdict, "string"),),
        params={"entity": "person", "column": column, "verdict_column": verdict},
        population=population,
    )


def tolerance_graph() -> Graph:
    """Charter C5's scoped numerics, carried through a structural version."""

    bounded = Node(
        "bounded",
        "derive.tolerant@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "bounded_value", "float64"),),
        params={
            "entity": "person",
            "columns": ("age",),
            "target": "bounded_value",
            "scale": 1.0,
        },
        population="survey",
    )
    bitwise = toy.derive("bitwise", ("age",), "bitwise_value")
    carrier = toy.select_node("adults_view", base="survey", policy="free")
    return Graph(
        "toy",
        (toy.SOURCE,),
        (
            toy.CREATE,
            bounded,
            bitwise,
            _tolerance_gate("bounded_gate", "bounded_value"),
            _tolerance_gate("bitwise_gate", "bitwise_value"),
            carrier,
            _tolerance_gate("carried_gate", "bounded_value", population=carrier.id),
            _tolerance_gate(
                "carried_bitwise_gate", "bitwise_value", population=carrier.id
            ),
        ),
    )


def structural_graph() -> Graph:
    """Three structural successors of one base, each with its own siblings.

    Both EXPAND kinds and a FILTER read ``survey``; every ordinary node sits on
    a structural version, so an entrant EXPAND needs no claim beyond its own.
    """

    expand, claim = toy.entrant_expand_node()
    cohort, cohort_claim = toy.entrant_person_node()
    return Graph(
        "toy",
        (toy.SOURCE,),
        (
            toy.CREATE,
            toy.select_node("adults", base="survey"),
            toy.derive(
                "adult_resources",
                ("age", "income"),
                "adult_resources",
                scale=1.5,
                population="adults",
            ),
            toy.draw("adult_draw", "adult_noise", population="adults"),
            toy.impute(
                "adult_target", ("age", "income"), "adult_target", population="adults"
            ),
            expand,
            claim,
            toy.draw("entry_draw", "entry_noise", population=expand.id),
            toy.derive("entry_sum", ("age",), "entry_sum", population=expand.id),
            cohort,
            cohort_claim,
            toy.draw("cohort_draw", "cohort_noise", population=cohort.id),
        ),
    )


def wide_graph(width: int = 10) -> Graph:
    """The end-to-end toy country with many independent same-version siblings.

    Every leaf on ``survey`` shares one population version with the chained
    targets, so the coordinator folds their results into one cumulative
    population in canonical order however they finish; the REWEIGHT successor
    depends on all of them, and more siblings hang off the calibrated version.
    """

    full = toy.full_graph()
    survey_leaves = tuple(
        toy.draw(f"leaf_{index:02d}", f"leaf_{index:02d}_value") for index in range(width)
    )
    derived_leaves = tuple(
        toy.derive(
            f"sum_{index:02d}", ("age", "income"), f"sum_{index:02d}", scale=index + 1.0
        )
        for index in range(width // 2)
    )
    calibrated_leaves = tuple(
        toy.derive(
            f"cal_{index:02d}",
            ("age",),
            f"cal_{index:02d}_value",
            scale=float(index),
            population="calibrated",
        )
        for index in range(width // 2)
    )
    return Graph(
        "toy",
        (toy.SOURCE,),
        (*full.nodes, *survey_leaves, *derived_leaves, *calibrated_leaves),
    )


def artifact_graph() -> Graph:
    """A typed byte edge: the consumer is ready only once its producer is admitted."""

    fit = Node(
        "fit",
        "fit.forest@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "fitted", "float64"),),
        population="survey",
        artifact_outputs=(ArtifactOutput("forest", FOREST),),
    )
    consume = Node(
        "consume",
        "consume.forest@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "drawn", "float64"),),
        population="survey",
        artifact_inputs=(ArtifactInput("donor", "fit", "forest", FOREST),),
    )
    leaves = tuple(toy.draw(f"leaf_{name}", f"leaf_{name}_value") for name in "abcd")
    return Graph("toy", (toy.SOURCE,), (toy.CREATE, fit, consume, *leaves))


@dataclass(frozen=True)
class Fixture:
    graph: Callable[[], Graph]
    registry: Callable[[], KernelRegistry] = toy.toy_registry


FIXTURES: dict[str, Fixture] = {
    "small": Fixture(toy.small_graph),
    "chained": Fixture(lambda: toy.chained_graph(("a", "b", "c", "d", "e", "f"))),
    "full": Fixture(toy.full_graph),
    "full_failed_gate": Fixture(lambda: toy.full_graph(gate_low=1e12)),
    "tolerance": Fixture(tolerance_graph),
    "structural": Fixture(structural_graph),
    "wide": Fixture(wide_graph),
    "artifacts": Fixture(artifact_graph, artifact_registry),
}


# ----------------------------------------------------------------------
# Running and fingerprinting
# ----------------------------------------------------------------------


def store_objects(store: ContentStore) -> dict[str, bytes]:
    """Every file in the store's object tree, by relative path, with its bytes."""

    root = store.objects
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def store_litter(store: ContentStore) -> list[str]:
    return sorted(path.name for path in store.tmp.iterdir())


def population_fingerprint(manifest: object) -> dict[str, object]:
    """Every attached population: tables (values, dtypes, column order), weights, strata."""

    found: dict[str, object] = {}
    for version in sorted(manifest.populations):  # type: ignore[attr-defined]
        view = manifest.populations[version]  # type: ignore[attr-defined]
        tables = {}
        for entity in view.entities:
            table = view.entity(entity)
            tables[entity] = (
                tuple(str(column) for column in table.columns),
                tuple(str(dtype) for dtype in table.dtypes),
                pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes(),
            )
        weights = {
            entity: (
                view.weights_for(entity).kind.value,
                np.asarray(view.weights_for(entity).values).tobytes(),
            )
            for entity in view.weighted_entities
        }
        strata = pd.util.hash_pandas_object(view.strata, index=True).to_numpy().tobytes()
        found[version] = (tables, weights, strata)
    return found


@dataclass(frozen=True)
class Outcome:
    """Everything the identity claim covers, for one run."""

    manifest_key: str
    content_addressed: bytes
    receipts: dict[str, bytes]
    keys: dict[str, str]
    seeds: dict[str, int]
    hits: dict[str, bool]
    populations: dict[str, object]
    objects: dict[str, bytes]
    litter: list[str]

    def assert_identical(self, other: Outcome, *, label: str) -> None:
        assert other.manifest_key == self.manifest_key, label
        assert other.content_addressed == self.content_addressed, label
        assert other.receipts == self.receipts, label
        assert other.keys == self.keys, label
        assert other.seeds == self.seeds, label
        assert other.hits == self.hits, label
        assert other.populations == self.populations, label
        assert sorted(other.objects) == sorted(self.objects), label
        for path, content in self.objects.items():
            assert other.objects[path] == content, f"{label}: {path}"
        assert other.litter == self.litter == [], label


def fingerprint(manifest: object, store: ContentStore) -> Outcome:
    nodes = manifest.nodes  # type: ignore[attr-defined]
    return Outcome(
        manifest_key=manifest.key,  # type: ignore[attr-defined]
        content_addressed=canonical_json(manifest.content_addressed),  # type: ignore[attr-defined]
        receipts={
            node_id: canonical_json(receipt._content_payload())
            for node_id, receipt in nodes.items()
        },
        keys={node_id: receipt.key for node_id, receipt in nodes.items()},
        seeds={node_id: receipt.seed for node_id, receipt in nodes.items()},
        hits={node_id: receipt.hit for node_id, receipt in nodes.items()},
        populations=population_fingerprint(manifest),
        objects=store_objects(store),
        litter=store_litter(store),
    )


@dataclass
class Run:
    manifest: object
    store: ContentStore
    outcome: Outcome
    record: dict[str, object]
    probe: Probe


def execute(
    graph: Graph,
    root: Path,
    *,
    workers: int | None,
    registry: KernelRegistry | None = None,
    seed: int | None = None,
    source: Path | None = None,
    store: ContentStore | None = None,
    resume: str = "auto",
    observer: Callable[[str, object], None] | None = None,
) -> Run:
    probe = Probe()
    registry = toy.toy_registry() if registry is None else registry
    delay = seeded_delay(seed) if seed is not None else (lambda _node_id: 0.0)
    registry = jittered(registry, probe, delay)
    if source is None:
        source = toy.copy_source(root / "source")
    store = ContentStore(root / "store") if store is None else store
    record: dict[str, object] = {}
    manifest = run_graph(
        compile_graph(graph),
        sources={"survey": source},
        store=store,
        kernels=registry,
        resume=resume,  # type: ignore[arg-type]
        max_workers=workers,
        _population_observer=observer,
        _concurrency_record=record,
    )
    return Run(manifest, store, fingerprint(manifest, store), record, probe)


def live_workers() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(THREAD_NAME_PREFIX)
    ]


@pytest.fixture(autouse=True)
def _no_worker_outlives_its_test():
    assert live_workers() == []
    yield
    assert live_workers() == [], "a graph worker thread outlived run_graph"


# ----------------------------------------------------------------------
# Byte identity on every fixture, at every worker count
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_worker_count_reproduces_the_sequential_run_byte_for_byte(
    tmp_path: Path, name: str
) -> None:
    """Manifest key, receipts, keys, seeds, populations and store bytes all agree."""

    fixture = FIXTURES[name]
    baseline = execute(
        fixture.graph(), tmp_path / "w1", workers=1, registry=fixture.registry()
    )
    assert baseline.record["max_workers"] == 1
    assert baseline.probe.peak == 1
    assert {entry["path"] for entry in baseline.record["kernels"].values()} == {
        "inline"
    }
    assert not any(baseline.outcome.hits.values())
    assert baseline.outcome.objects, "the sequential run published nothing"

    for workers in WORKER_COUNTS:
        run = execute(
            fixture.graph(),
            tmp_path / f"w{workers}",
            workers=workers,
            registry=fixture.registry(),
            seed=workers,
        )
        baseline.outcome.assert_identical(run.outcome, label=f"{name} at {workers}")
        assert run.record["max_workers"] == workers
        # Every kernel call ran on a worker thread, never on the coordinator.
        assert run.probe.threads and all(
            thread.startswith(THREAD_NAME_PREFIX) for thread in run.probe.threads
        )
        assert run.probe.peak <= workers
        paths = {entry["path"] for entry in run.record["kernels"].values()}
        assert paths <= {"speculative", "turn"}
        assert run.record.get("discarded", 0) == 0


def test_independent_siblings_actually_overlap(tmp_path: Path) -> None:
    """The wide graph runs more than one kernel at once, on more than one thread."""

    run = execute(wide_graph(), tmp_path / "w4", workers=4, seed=7)
    assert run.probe.peak >= 2
    assert len(run.probe.threads) >= 2
    assert run.record["dispatched"] >= 10
    # Early projections were used both ways: as they stood when no sibling had
    # been admitted in between, and after re-projection proved them equal.
    assert run.record["reused_context"] >= 1
    assert run.record["verified_context"] >= 1


def _barrier_graph() -> Graph:
    def node(node_id: str) -> Node:
        return Node(
            node_id,
            f"meet.{node_id}@1",
            inputs=(Slice("person", ("age",)),),
            outputs=(Owned("person", f"{node_id}_age", "float64"),),
            population="survey",
        )

    return Graph("toy", (toy.SOURCE,), (toy.CREATE, node("left"), node("right")))


def _barrier_registry(barrier: threading.Barrier) -> KernelRegistry:
    def meet(target: str) -> Callable[[KernelContext], KernelResult]:
        def run(context: KernelContext) -> KernelResult:
            barrier.wait()
            return _ages(target)(context)

        return run

    registry = toy.toy_registry()
    registry.register(FnKernel("meet.left@1", _PLAIN, meet("left_age")))
    registry.register(FnKernel("meet.right@1", _PLAIN, meet("right_age")))
    return registry


def test_kernels_run_concurrently_not_merely_on_another_thread(tmp_path: Path) -> None:
    """Two independent kernels meet at a barrier neither can pass alone."""

    concurrent = execute(
        _barrier_graph(),
        tmp_path / "w2",
        workers=2,
        registry=_barrier_registry(threading.Barrier(2, timeout=30)),
    )
    assert concurrent.probe.peak == 2
    # Sequentially the first kernel waits alone until its barrier times out,
    # and the run refuses on that kernel: the pass above needed real overlap.
    with pytest.raises(NodeRejected, match="left"):
        execute(
            _barrier_graph(),
            tmp_path / "w1",
            workers=1,
            registry=_barrier_registry(threading.Barrier(2, timeout=0.2)),
        )


@pytest.mark.parametrize("name", ["wide", "structural", "artifacts"])
def test_seeded_completion_orders_all_fold_to_the_sequential_run(
    tmp_path: Path, name: str
) -> None:
    """Seeded pauses reorder which worker finishes first; admission never moves."""

    fixture = FIXTURES[name]
    baseline = execute(
        fixture.graph(), tmp_path / "w1", workers=1, registry=fixture.registry()
    )
    orders = set()
    for seed in range(6):
        run = execute(
            fixture.graph(),
            tmp_path / f"seed{seed}",
            workers=4,
            registry=fixture.registry(),
            seed=100 + seed,
        )
        baseline.outcome.assert_identical(run.outcome, label=f"{name} seed {seed}")
        orders.add(tuple(run.probe.finished))
    # The sweep is only evidence if the pauses did reorder completions.
    assert len(orders) >= 2


# ----------------------------------------------------------------------
# Resume: warm, partial, required and forbidden reuse
# ----------------------------------------------------------------------


def test_warm_reruns_hit_every_record_and_dispatch_nothing(tmp_path: Path) -> None:
    graph = wide_graph()
    source = toy.copy_source(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    cold = execute(graph, tmp_path / "cold", workers=4, source=source, store=store, seed=1)
    assert not any(cold.outcome.hits.values())

    warm = execute(graph, tmp_path / "warm", workers=4, source=source, store=store, seed=2)
    assert all(warm.outcome.hits.values())
    assert warm.manifest.key == cold.manifest.key
    assert warm.outcome.objects == cold.outcome.objects
    assert warm.probe.started == []
    assert warm.record.get("dispatched", 0) == 0

    required = execute(
        graph, tmp_path / "req", workers=4, source=source, store=store, resume="require"
    )
    assert all(required.outcome.hits.values())
    assert required.manifest.key == cold.manifest.key
    assert required.probe.started == []
    # resume="require" restores every node and runs sequentially by design.
    assert "dispatched" not in required.record


def test_a_partial_store_resumes_to_the_same_bytes(tmp_path: Path) -> None:
    """A parallel run over a store holding a prefix equals the sequential one."""

    prefix = toy.small_graph()
    graph = wide_graph()
    outcomes = {}
    for workers in (1, 4):
        root = tmp_path / f"w{workers}"
        source = toy.copy_source(root / "source")
        store = ContentStore(root / "store")
        execute(prefix, root / "prefix", workers=1, source=source, store=store)
        run = execute(graph, root / "full", workers=workers, source=source, store=store, seed=3)
        assert any(run.outcome.hits.values()) and not all(run.outcome.hits.values())
        outcomes[workers] = run.outcome
    outcomes[1].assert_identical(outcomes[4], label="partial resume")


def test_forbidden_reuse_rewrites_the_same_bytes(tmp_path: Path) -> None:
    outcomes = {}
    for workers in (1, 4):
        root = tmp_path / f"w{workers}"
        source = toy.copy_source(root / "source")
        store = ContentStore(root / "store")
        execute(wide_graph(), root / "cold", workers=1, source=source, store=store)
        run = execute(
            wide_graph(),
            root / "forbid",
            workers=workers,
            source=source,
            store=store,
            resume="forbid",
            seed=4,
        )
        assert not any(run.outcome.hits.values())
        outcomes[workers] = run.outcome
    outcomes[1].assert_identical(outcomes[4], label="resume=forbid")


# ----------------------------------------------------------------------
# Failures: the same exception, the same publications, no late writes
# ----------------------------------------------------------------------


def _slow(seconds: float, fn: Callable[[KernelContext], KernelResult]):
    def run(context: KernelContext) -> KernelResult:
        time.sleep(seconds)
        return fn(context)

    return run


def _ages(target: str) -> Callable[[KernelContext], KernelResult]:
    def run(context: KernelContext) -> KernelResult:
        ages = context.tables["person"]["age"].to_numpy(dtype=np.float64)
        return KernelResult(columns={("person", target): _person_series(context, ages)})

    return run


def _explodes(context: KernelContext) -> KernelResult:
    raise RuntimeError(f"{context.node.id} exploded on purpose")


def _person_node(node_id: str, kernel: str, target: str) -> Node:
    return Node(
        node_id,
        kernel,
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", target, "float64"),),
        population="survey",
    )


def _failure_graph() -> Graph:
    """``a_fails`` is first in canonical order; its siblings finish before it."""

    return Graph(
        "toy",
        (toy.SOURCE,),
        (
            toy.CREATE,
            _person_node("a_fails", "fail.slow@1", "a_value"),
            _person_node("b_done", "ok.fast@1", "b_value"),
            _person_node("c_done", "ok.fast2@1", "c_value"),
            toy.derive("d_after", ("b_value", "c_value"), "d_value"),
        ),
    )


def _failure_registry() -> tuple[KernelRegistry, dict[str, FnKernel]]:
    registry = toy.toy_registry()
    kernels = {
        "a": FnKernel("fail.slow@1", _PLAIN, _slow(0.3, _explodes)),
        "b": FnKernel("ok.fast@1", _PLAIN, _ages("b_value")),
        "c": FnKernel("ok.fast2@1", _PLAIN, _ages("c_value")),
    }
    for kernel in kernels.values():
        registry.register(kernel)
    return registry, kernels


def test_a_failure_raises_the_sequential_refusal_and_publishes_its_prefix_only(
    tmp_path: Path,
) -> None:
    """Workers that finished for nodes after the failure publish nothing."""

    results = {}
    for workers in (1, 4):
        root = tmp_path / f"w{workers}"
        registry, kernels = _failure_registry()
        store = ContentStore(root / "store")
        record: dict[str, object] = {}
        with pytest.raises(NodeRejected) as raised:
            run_graph(
                compile_graph(_failure_graph()),
                sources={"survey": toy.copy_source(root / "source")},
                store=store,
                kernels=registry,
                max_workers=workers,
                _concurrency_record=record,
            )
        assert live_workers() == []
        results[workers] = (
            type(raised.value),
            str(raised.value),
            type(raised.value.__cause__),
            str(raised.value.__cause__),
            store_objects(store),
            store_litter(store),
            {name: kernel.calls for name, kernel in kernels.items()},
            record,
        )

    sequential, parallel = results[1], results[4]
    assert "a_fails" in sequential[1]
    assert parallel[:6] == sequential[:6]
    # Sequentially, nothing after the failing node ever ran...
    assert sequential[6] == {"a": 1, "b": 0, "c": 0}
    # ...while in parallel both later siblings ran to completion on workers,
    # and still published nothing: the stores above are equal.
    assert parallel[6] == {"a": 1, "b": 1, "c": 1}
    assert parallel[7]["dispatched"] >= 3


def test_a_speculative_sibling_failure_surfaces_only_at_its_own_turn(
    tmp_path: Path,
) -> None:
    """A sibling that fails first on a worker still waits for earlier admissions."""

    graph = Graph(
        "toy",
        (toy.SOURCE,),
        (
            toy.CREATE,
            _person_node("a_slow", "ok.slow@1", "a_value"),
            _person_node("b_fails", "fail.fast@1", "b_value"),
        ),
    )
    results = {}
    for workers in (1, 3):
        root = tmp_path / f"w{workers}"
        registry = toy.toy_registry()
        registry.register(FnKernel("ok.slow@1", _PLAIN, _slow(0.3, _ages("a_value"))))
        registry.register(FnKernel("fail.fast@1", _PLAIN, _explodes))
        store = ContentStore(root / "store")
        with pytest.raises(NodeRejected, match="b_fails") as raised:
            run_graph(
                compile_graph(graph),
                sources={"survey": toy.copy_source(root / "source")},
                store=store,
                kernels=registry,
                max_workers=workers,
            )
        results[workers] = (str(raised.value), store_objects(store))
    assert results[3] == results[1]
    # a_slow was admitted and persisted before b_fails refused, in both runs.
    assert len(results[1][1]) > 0


def test_a_gate_that_raises_on_a_worker_is_still_a_failed_verdict(tmp_path: Path) -> None:
    graph = replace_gate(toy.full_graph(), kernel="bad.gate_raise@1")
    baseline = execute(graph, tmp_path / "w1", workers=1)
    run = execute(graph, tmp_path / "w4", workers=4, seed=5)
    baseline.outcome.assert_identical(run.outcome, label="raising gate")
    assert run.manifest.nodes["gate_tax"].receipt["outcome"] == "fail"


def replace_gate(graph: Graph, *, kernel: str) -> Graph:
    return toy.replace_node(graph, replace(toy.gate_node(), kernel=kernel))


def test_a_worker_that_mutates_its_context_is_refused_as_sequentially(
    tmp_path: Path,
) -> None:
    mutating = Node(
        "mutator",
        "bad.mutate@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "mutated", "float64"),),
        params={"column": "age", "target": "mutated"},
        population="survey",
    )
    graph = toy.small_graph(extra=(mutating,))
    messages = {}
    for workers in (1, 4):
        root = tmp_path / f"w{workers}"
        store = ContentStore(root / "store")
        with pytest.raises(NodeRejected) as raised:
            run_graph(
                compile_graph(graph),
                sources={"survey": toy.copy_source(root / "source")},
                store=store,
                kernels=toy.toy_registry(),
                max_workers=workers,
            )
        messages[workers] = (str(raised.value), store_objects(store))
    assert messages[4] == messages[1]


def test_an_observer_failure_stops_at_the_same_node_with_the_same_store(
    tmp_path: Path,
) -> None:
    observed: dict[int, list[str]] = {}

    def observer_for(workers: int):
        seen = observed.setdefault(workers, [])

        def observe(node_id: str, population: object) -> None:
            seen.append(node_id)
            if node_id == "target_a":
                raise RuntimeError("observer refused target_a")

        return observe

    results = {}
    for workers in (1, 4):
        root = tmp_path / f"w{workers}"
        store = ContentStore(root / "store")
        with pytest.raises(RuntimeError, match="observer refused"):
            execute(
                wide_graph(),
                root,
                workers=workers,
                store=store,
                observer=observer_for(workers),
                seed=6,
            )
        results[workers] = store_objects(store)
    assert observed[4] == observed[1]
    assert results[4] == results[1]


def test_the_observer_sees_the_canonical_order_on_the_calling_thread(
    tmp_path: Path,
) -> None:
    calls: dict[int, list[tuple[str, str, tuple[str, ...]]]] = {}

    def observer_for(workers: int):
        seen = calls.setdefault(workers, [])

        def observe(node_id: str, population: object) -> None:
            seen.append(
                (
                    node_id,
                    threading.current_thread().name,
                    tuple(population.frame.table("person").columns),  # type: ignore[attr-defined]
                )
            )

        return observe

    order = None
    for workers in (1, 4):
        run = execute(
            wide_graph(), tmp_path / f"w{workers}", workers=workers, observer=observer_for(workers)
        )
        order = run.manifest.nodes.keys()
    assert calls[4] == calls[1]
    assert [entry[0] for entry in calls[4]] == list(order)
    assert {entry[1] for entry in calls[4]} == {threading.current_thread().name}


def test_a_source_changed_by_a_worker_evicts_the_run_as_sequentially(
    tmp_path: Path,
) -> None:
    """A kernel that rewrites its declared source is refused; the run's writes go."""

    def rewrites_source(context: KernelContext) -> KernelResult:
        path = Path(context.sources["survey"]) / "person.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        return _ages("touched")(context)

    graph = Graph(
        "toy",
        (toy.SOURCE,),
        (
            toy.CREATE,
            toy.draw("leaf_a", "leaf_a_value"),
            Node(
                "toucher",
                "touch.source@1",
                sources=("survey",),
                inputs=(Slice("person", ("age",)),),
                outputs=(Owned("person", "touched", "float64"),),
                population="survey",
            ),
            toy.draw("leaf_z", "leaf_z_value"),
        ),
    )
    results = {}
    for workers in (1, 4):
        root = tmp_path / f"w{workers}"
        registry = toy.toy_registry()
        registry.register(FnKernel("touch.source@1", _PLAIN, rewrites_source))
        store = ContentStore(root / "store")
        with pytest.raises(NodeRejected) as raised:
            run_graph(
                compile_graph(graph),
                sources={"survey": toy.copy_source(root / "source")},
                store=store,
                kernels=registry,
                max_workers=workers,
            )
        results[workers] = (str(raised.value), store_objects(store), store_litter(store))
    assert "survey" in results[1][0]
    assert results[4] == results[1]


# ----------------------------------------------------------------------
# A precomputed call is used only when its context is the turn's context
# ----------------------------------------------------------------------


def test_a_stale_early_projection_is_discarded_and_the_turn_recomputes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault injection: the early projection of one node is wrong on purpose.

    ``leaf_01`` shares its version with ``leaf_00``, admitted between its
    early projection and its turn, so the turn re-projects, finds a different
    digest, discards the precomputed call and runs the kernel on its own
    context. The run still equals the sequential one.
    """

    graph = Graph(
        "toy",
        (toy.SOURCE,),
        (
            toy.CREATE,
            _person_node("leaf_00", "ok.slow@1", "leaf_00_value"),
            toy.derive("leaf_01", ("age",), "leaf_01_value"),
        ),
    )

    def registry() -> KernelRegistry:
        kernels = toy.toy_registry()
        kernels.register(FnKernel("ok.slow@1", _PLAIN, _slow(0.2, _ages("leaf_00_value"))))
        return kernels

    baseline = execute(graph, tmp_path / "w1", workers=1, registry=registry())

    original = graph_executor._project_context
    poisoned: list[str] = []

    def project(node: Node, population: object, **kwargs: object):
        context = original(node, population, **kwargs)
        if node.id == "leaf_01" and not poisoned:
            poisoned.append(node.id)
            tables = dict(context.tables)
            person = tables["person"].copy()
            person["age"] = person["age"] + 1
            tables["person"] = person
            return replace(context, tables=tables)
        return context

    monkeypatch.setattr(graph_executor, "_project_context", project)
    run = execute(graph, tmp_path / "w2", workers=2, registry=registry())
    assert poisoned == ["leaf_01"]
    assert run.record["discarded"] == 1
    assert run.record["kernels"]["leaf_01"]["path"] == "turn"
    baseline.outcome.assert_identical(run.outcome, label="discarded projection")


# ----------------------------------------------------------------------
# The flag
# ----------------------------------------------------------------------


def test_the_worker_count_resolves_from_the_argument_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MAX_WORKERS_ENVIRONMENT, raising=False)
    assert resolve_max_workers(None) == 1
    assert resolve_max_workers(3) == 3
    monkeypatch.setenv(MAX_WORKERS_ENVIRONMENT, "  ")
    assert resolve_max_workers(None) == 1
    monkeypatch.setenv(MAX_WORKERS_ENVIRONMENT, " 6 ")
    assert resolve_max_workers(None) == 6
    assert resolve_max_workers(1) == 1  # an explicit count wins
    for bad in ("0", "-2", "+2", "2.0", "two", "0x2"):
        monkeypatch.setenv(MAX_WORKERS_ENVIRONMENT, bad)
        with pytest.raises(ValueError, match=MAX_WORKERS_ENVIRONMENT):
            resolve_max_workers(None)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            resolve_max_workers(bad)
    for bad in (True, 2.0, "2"):
        with pytest.raises(TypeError, match="max_workers"):
            resolve_max_workers(bad)


def test_an_invalid_worker_count_is_refused_before_the_store_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentStore(tmp_path / "store")
    source = toy.copy_source(tmp_path / "source")
    for bad, error in ((0, ValueError), (True, TypeError), (2.0, TypeError)):
        with pytest.raises(error, match="max_workers"):
            run_graph(
                compile_graph(toy.small_graph()),
                sources={"survey": source},
                store=store,
                kernels=toy.toy_registry(),
                max_workers=bad,
            )
    monkeypatch.setenv(MAX_WORKERS_ENVIRONMENT, "many")
    with pytest.raises(ValueError, match=MAX_WORKERS_ENVIRONMENT):
        run_graph(
            compile_graph(toy.small_graph()),
            sources={"survey": source},
            store=store,
            kernels=toy.toy_registry(),
        )
    assert store_objects(store) == {}


def test_the_environment_opts_a_nested_run_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MAX_WORKERS_ENVIRONMENT, "3")
    run = execute(wide_graph(), tmp_path / "env", workers=None, seed=8)
    assert run.record["max_workers"] == 3
    assert run.record["dispatched"] > 0
    explicit = execute(wide_graph(), tmp_path / "explicit", workers=1)
    assert explicit.record["max_workers"] == 1
    explicit.outcome.assert_identical(run.outcome, label="environment opt-in")


def test_scheduling_never_enters_the_portable_manifest(tmp_path: Path) -> None:
    """Portable JSON differs only in run-level fields (timestamps, wall times)."""

    import json

    documents = {}
    for workers in (1, 4):
        run = execute(wide_graph(), tmp_path / f"w{workers}", workers=workers, seed=9)
        document = json.loads(run.manifest.to_json())
        for field in ("started_at", "finished_at", "host"):
            document.pop(field)
        for node in document["nodes"].values():
            node.pop("wall_time")
        documents[workers] = document
    assert documents[4] == documents[1]
    assert "worker" not in json.dumps(documents[4])


# ----------------------------------------------------------------------
# The scheduler on its own
# ----------------------------------------------------------------------


class _Sleeper:
    def __init__(self, probe: Probe, pause: float = 0.02) -> None:
        self.probe = probe
        self.pause = pause
        self.capabilities = Capabilities(determinism=Determinism.DETERMINISTIC)

    def run(self, context: object) -> str:
        with self.probe.running(str(context)):
            time.sleep(self.pause)
            return f"ran {context}"


def _speculation(
    order: Sequence[str],
    predecessors: Mapping[str, Sequence[str]],
    admitted: dict[str, object],
    *,
    workers: int,
    probe: Probe,
    decline: frozenset[str] = frozenset(),
    record: dict[str, object] | None = None,
) -> Speculation:
    kernel = _Sleeper(probe)

    def prepare(node_id: str) -> PreparedKernel | None:
        if node_id in decline:
            return None
        return PreparedKernel(
            node_id=node_id,
            kernel=kernel,
            context=node_id,  # type: ignore[arg-type]
            before=b"",
            incumbent=None,
            tolerances={},
            numerics={},
        )

    return Speculation(
        workers,
        order=order,
        predecessors=predecessors,
        admitted=admitted,
        prepare=prepare,
        digest=lambda _context: b"",
        record={} if record is None else record,
    )


def test_the_scheduler_bounds_running_and_waiting_calls_by_the_worker_count() -> None:
    order = tuple(f"n{index}" for index in range(12))
    predecessors = {node_id: () for node_id in order}
    admitted: dict[str, object] = {}
    probe = Probe()
    record: dict[str, object] = {}
    speculation = _speculation(
        order, predecessors, admitted, workers=3, probe=probe, record=record
    )
    try:
        for position, node_id in enumerate(order):
            speculation.fill(position)
            prepared = speculation.claim(node_id)
            assert prepared is not None
            outcome = speculation.outcome(prepared, reused=True)
            assert outcome.result == f"ran {node_id}"
            admitted[node_id] = object()
    finally:
        speculation.close()
    assert probe.peak <= 3
    assert record["dispatched"] == 12
    assert record["peak_submitted"] <= 3
    assert live_workers() == []


def test_the_scheduler_offers_only_nodes_whose_predecessors_are_admitted() -> None:
    order = ("root", "left", "right", "join")
    predecessors = {
        "root": (),
        "left": ("root",),
        "right": ("root",),
        "join": ("left", "right"),
    }
    admitted: dict[str, object] = {}
    probe = Probe()
    speculation = _speculation(
        order, predecessors, admitted, workers=4, probe=probe, decline=frozenset({"right"})
    )
    try:
        speculation.fill(0)
        assert set(speculation._prepared) == {"root"}
        root = speculation.claim("root")
        speculation.outcome(root, reused=True)
        admitted["root"] = object()
        speculation.fill(1)
        # "right" declined, so its turn runs it; "join" waits for both parents.
        assert set(speculation._prepared) == {"left"}
        assert speculation.claim("right") is None
        admitted["left"] = object()
        speculation.fill(2)
        assert "join" not in speculation._prepared
        outcome = speculation.run("right", _Sleeper(probe), "right")
        assert outcome.result == "ran right"
        admitted["right"] = object()
        speculation.fill(3)
        assert set(speculation._prepared) == {"left", "join"}
        speculation.retire("left")
        assert "left" not in speculation._prepared
    finally:
        speculation.close()
    assert live_workers() == []


def test_closing_the_scheduler_joins_calls_still_running() -> None:
    order = ("a", "b")
    probe = Probe()
    speculation = _speculation(
        order, {"a": (), "b": ()}, {}, workers=2, probe=probe
    )
    speculation.fill(0)
    speculation.close()
    assert live_workers() == []
    assert probe.active == 0
    speculation.close()  # idempotent


def test_a_single_worker_is_not_a_scheduler() -> None:
    with pytest.raises(ValueError, match="two workers"):
        _speculation(("a",), {"a": ()}, {}, workers=1, probe=Probe())
