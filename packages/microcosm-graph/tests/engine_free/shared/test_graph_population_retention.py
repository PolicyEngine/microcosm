"""A run that names the versions it retains frees every other one.

``run_graph(retain_populations=...)`` lets a caller say which population
versions it will read from the manifest. The default, ``None``, keeps every
version to the end of the run and attaches all of them, as before. Given a
collection, each version it does not name leaves the executor once the last
node that reads it has applied its result, so its frame can be freed. These
tests pin that contract:

- the store, the node keys, the receipts and the portable manifest are the
  same with and without the keyword, cold, warm and under ``resume="require"``,
  over CREATE, FILTER, EXPAND and REWEIGHT versions, a gate and a release;
- only the named versions, and their mass ledgers, are attached;
- a dead version's tables are freed by its last reader, before that reader is
  observed or writes its own frame, and nothing a node held outlives it into
  the run-end pass;
- a version read by several members and several structural nodes is kept
  until the last of them has run, and a version nothing reads goes with the
  node that made it;
- an unreached last reader still releases its version;
- a name that is not a version of the graph is refused before the run reads a
  source, runs a kernel or writes to the store.

Synthetic graphs only: the small graphs below and the toy country's fixture
tables. No data file, no country package, no engine.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import weakref
from collections.abc import Callable, Collection, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.graph.executor as graph_executor
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.decl import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.executor import run_graph
from microcosm.graph.kernel import (
    Capabilities,
    Determinism,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
)
from microcosm.graph.manifest import NodeReceipt, PopulationView, RunManifest
from microcosm.graph.population import Population
from microcosm.graph.store import ContentStore
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-graph")

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", _TEST_PATHS.tests / "_toy.py"
    )
    assert _SPEC is not None and _SPEC.loader is not None
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

SOURCE = SourceRef("survey", "csv-tables")
EVIDENCE = ArtifactType("gate.evidence", 1)


class _Kernel:
    def __init__(self, ref: str, capabilities: Capabilities, compute) -> None:
        self.ref = ref
        self.capabilities = capabilities
        self.compute = compute
        self.calls = 0

    def implementation_hash(self) -> str:
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        self.calls += 1
        return self.compute(context)


def _source_path(root: Path) -> Path:
    root.mkdir()
    (root / "value.txt").write_text("0", encoding="utf-8")
    return root


def _source_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20], dtype=np.int64),
            "age": np.asarray([10, 20, 30], dtype=np.int64),
            "selected": pd.Series([True, False, True], dtype="boolean"),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([10, 20], dtype=np.int64),
            "size": np.asarray([2, 1], dtype=np.int64),
        }
    )
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray([1.0, 2.0]), WeightKind.DESIGN)},
        pd.Series(["a", "a", "b"], name="stratum"),
    )


def _source(context: KernelContext) -> KernelResult:
    return KernelResult(frame=_source_frame())


def _double(context: KernelContext) -> KernelResult:
    table = context.tables["person"]
    values = table[str(context.params["source"])].to_numpy(dtype=np.float64)
    return KernelResult(
        columns={
            ("person", str(context.params["target"])): pd.Series(
                values * 2.0,
                index=pd.Index(table["person_id"], name="person_id"),
                dtype="float64",
            )
        }
    )


def _keep(context: KernelContext) -> KernelResult:
    table = context.tables["person"]
    if context.params["keep"] == "selected":
        keep = table["selected"].to_numpy(dtype=np.bool_)
    else:
        keep = np.ones(len(table), dtype=np.bool_)
    return KernelResult(keep=pd.Series(keep, index=table["person_id"], dtype="bool"))


def _refuse(context: KernelContext) -> KernelResult:
    raise RuntimeError("evidence unavailable")


def _registry() -> KernelRegistry:
    deterministic = Capabilities(Determinism.DETERMINISTIC)
    registry = KernelRegistry()
    for kernel in (
        _Kernel(
            "source@1",
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
            _source,
        ),
        _Kernel("double@1", deterministic, _double),
        _Kernel(
            "keep@1",
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER),
            _keep,
        ),
        _Kernel(
            "gate@1",
            Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE),
            _refuse,
        ),
        _Kernel("use@1", deterministic, _double),
    ):
        registry.register(kernel)
    return registry


CREATE = Node(
    "survey",
    "source@1",
    sources=("survey",),
    structural=StructuralDelta.CREATE,
    outputs=(
        Owned("person", "age", "int64"),
        Owned("person", "selected", "boolean"),
        Owned("household", "size", "int64"),
    ),
)


def _member(node_id: str, version: str, source: str) -> Node:
    """An ordinary member of ``version`` that writes the column named ``node_id``."""

    return Node(
        node_id,
        "double@1",
        inputs=(Slice("person", (source,)),),
        outputs=(Owned("person", node_id, "float64"),),
        params={"source": source, "target": node_id},
        population=version,
    )


def _boundary(
    node_id: str, base: str, reads: str, *, keep_selected: bool = False
) -> Node:
    """A FILTER on ``base``: an identity boundary, or one that drops a person."""

    return Node(
        node_id,
        "keep@1",
        structural=StructuralDelta.FILTER,
        base=base,
        inputs=(Slice("person", (reads,)),),
        params={"keep": "selected" if keep_selected else "all"},
        mass="free" if keep_selected else "conserve",
    )


def _chain() -> Graph:
    """``survey`` -> ``kept`` -> ``final``, with one member on each version.

    ``kept`` drops a person, so the versions differ in rows as well as in
    columns; ``final`` keeps everyone, as an identity boundary does. The run
    order is survey, a, kept, b, final, c.
    """

    return Graph(
        "toy",
        (SOURCE,),
        (
            CREATE,
            _member("a", "survey", "age"),
            _boundary("kept", "survey", "selected", keep_selected=True),
            _member("b", "kept", "a"),
            _boundary("final", "kept", "b"),
            _member("c", "final", "b"),
        ),
    )


def _shared_base() -> Graph:
    """``survey`` read by two members and by three structural nodes.

    ``first``, ``leaf`` and ``second`` all build on ``survey`` at one depth, so
    ``second`` is the last of its five readers. Nothing reads ``leaf``'s own
    version, so ``leaf`` is the last to use it. The run order is survey, a, b,
    first, leaf, second, c, d.
    """

    return Graph(
        "toy",
        (SOURCE,),
        (
            CREATE,
            _member("a", "survey", "age"),
            _member("b", "survey", "age"),
            _boundary("first", "survey", "a"),
            _boundary("second", "survey", "selected", keep_selected=True),
            _boundary("leaf", "survey", "age"),
            _member("c", "first", "a"),
            _member("d", "second", "b"),
        ),
    )


def _unreached_reader() -> Graph:
    """``survey``'s last reader is a boundary that never runs.

    The gate raises, so the bytes ``use`` reads are never produced: ``use`` is
    unreached, and so is ``boundary``, which reads ``use``'s column. ``beside``
    runs. The run order is survey, beside, gate, use, boundary.
    """

    gate = Node(
        "gate",
        "gate@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("household", "gate_verdict", "string"),),
        population="survey",
        artifact_outputs=(ArtifactOutput("evidence", EVIDENCE),),
    )
    use = Node(
        "use",
        "use@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "use", "float64"),),
        params={"source": "age", "target": "use"},
        population="survey",
        artifact_inputs=(ArtifactInput("evidence", "gate", "evidence", EVIDENCE),),
    )
    return Graph(
        "toy",
        (SOURCE,),
        (
            CREATE,
            gate,
            use,
            _boundary("boundary", "survey", "use"),
            _member("beside", "survey", "age"),
        ),
    )


def _toy_expand() -> Graph:
    """The toy country's entrant EXPAND on ``survey``, and the claim of its cells."""

    return toy.small_graph(nodes=(toy.CREATE, *toy.entrant_expand_node()))


#: Each graph the neutrality test runs: how to build it, write its source and
#: register its kernels. The chain has CREATE and FILTER versions. The toy
#: country adds two REWEIGHT versions, a gate and a release, and its entrant
#: graph an EXPAND version.
_GRAPHS: dict[
    str,
    tuple[Callable[[], Graph], Callable[[Path], Path], Callable[[], KernelRegistry]],
] = {
    "chain": (_chain, _source_path, _registry),
    "toy": (toy.full_graph, toy.copy_source, toy.toy_registry),
    "toy-expand": (_toy_expand, toy.copy_source, toy.toy_registry),
}


class _Liveness:
    """An observer that records which versions' tables are still alive.

    It holds only weak references, to the tables of the latest population
    admitted in each version, so it never keeps a version alive itself. At
    each node it first takes the population just admitted as its version's
    latest, then records which of the versions seen so far are alive. By then
    the executor has made that node's releases and dropped its incumbent, so a
    version the node was the last to read is already dead. The executor must
    hand it the live population (``_population_observer_detach=False``): a
    detached snapshot is a copy that dies at once.

    It runs no garbage collection, and the tests that use it turn the cyclic
    collector off (``no_cyclic_collector``). So a released version's tables
    die by reference counting alone, at once, not at some later collection.
    """

    def __init__(self) -> None:
        self.tables: dict[str, dict[str, weakref.ref[pd.DataFrame]]] = {}
        self.alive: dict[str, dict[str, bool]] = {}

    def __call__(self, node_id: str, population: Population) -> None:
        self.tables[population.version] = {
            entity: weakref.ref(population.frame.table(entity))
            for entity in population.frame.entities
        }
        self.alive[node_id] = self.alive_now()

    def alive_now(self) -> dict[str, bool]:
        return {
            version: any(table() is not None for table in tables.values())
            for version, tables in self.tables.items()
        }


@pytest.fixture
def no_cyclic_collector() -> Iterator[None]:
    """Turn the cyclic garbage collector off for one test.

    A released version must be freed by reference counting alone. With the
    collector on, a version kept alive only by a reference cycle could be
    freed by an incidental collection, and a liveness test would pass for the
    wrong reason.
    """

    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _run(
    graph: Graph,
    source: Path,
    store: ContentStore,
    *,
    retain: Collection[str] | None = None,
    resume: str = "auto",
    observer: _Liveness | None = None,
    registry: KernelRegistry | None = None,
) -> RunManifest:
    return run_graph(
        compile_graph(graph),
        sources={"survey": source},
        store=store,
        kernels=_registry() if registry is None else registry,
        resume=resume,  # type: ignore[arg-type]
        retain_populations=retain,
        _population_observer=observer,
        _population_observer_detach=observer is None,
    )


def _object_bytes(store: ContentStore) -> dict[str, bytes]:
    return {
        path.relative_to(store.objects).as_posix(): path.read_bytes()
        for path in sorted(store.objects.rglob("*"))
        if path.is_file()
    }


def _portable(manifest: RunManifest) -> dict[str, object]:
    """The manifest's JSON less the fields that move between two runs.

    Those are the two timestamps, the host, and the run-level fields of each
    receipt (``NodeReceipt.RUN_LEVEL_FIELDS``). None of them reaches the key.
    """

    payload = json.loads(manifest.to_json())
    for name in ("started_at", "finished_at", "host"):
        del payload[name]
    for receipt in payload["nodes"].values():
        for name in sorted(NodeReceipt.RUN_LEVEL_FIELDS):
            del receipt[name]
    return payload


def _assert_same_version(actual: PopulationView, expected: PopulationView) -> None:
    assert actual.entities == expected.entities
    for entity in expected.entities:
        pd.testing.assert_frame_equal(actual.table(entity), expected.table(entity))
    assert actual.weighted_entities == expected.weighted_entities
    for entity in expected.weighted_entities:
        assert actual.weights_for(entity).kind is expected.weights_for(entity).kind
        np.testing.assert_array_equal(
            actual.weights_for(entity).values, expected.weights_for(entity).values
        )
    pd.testing.assert_series_equal(actual.strata, expected.strata)
    assert actual.metadata == expected.metadata
    assert actual.mass_log == expected.mass_log


# --------------------------------------------------------------------------
# What retention must not move.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("graph", "retain"),
    [
        ("chain", ("final",)),
        ("chain", ("survey", "final")),
        ("chain", frozenset({"kept"})),
        ("chain", ()),
        ("toy", ("calibrated",)),
        ("toy", ()),
        ("toy-expand", ("survey",)),
        ("toy-expand", ()),
    ],
    ids=(
        "chain-final",
        "chain-survey-and-final",
        "chain-kept",
        "chain-none",
        "toy-calibrated",
        "toy-none",
        "toy-expand-survey",
        "toy-expand-none",
    ),
)
@pytest.mark.parametrize(
    ("warm", "resume"),
    [(False, "auto"), (True, "auto"), (True, "require")],
    ids=("cold", "warm", "require"),
)
def test_retention_moves_no_store_object_key_receipt_or_manifest_field(
    tmp_path: Path, warm: bool, resume: str, graph: str, retain: Collection[str]
) -> None:
    """Two runs of one graph, with and without the keyword, into two stores."""

    build, write_source, registry = _GRAPHS[graph]
    source = write_source(tmp_path / "source")
    every_store = ContentStore(tmp_path / "every")
    named_store = ContentStore(tmp_path / "named")
    if warm:
        for store in (every_store, named_store):
            _run(build(), source, store, registry=registry())
    every = _run(build(), source, every_store, resume=resume, registry=registry())
    named = _run(
        build(),
        source,
        named_store,
        resume=resume,
        retain=retain,
        registry=registry(),
    )

    assert _object_bytes(named_store) == _object_bytes(every_store)
    assert all(receipt.hit is warm for receipt in named.nodes.values())
    assert named.key == every.key
    assert _portable(named) == _portable(every)
    assert dict(named.source_identities) == dict(every.source_identities)
    assert dict(named.verification_epoch) == dict(every.verification_epoch) == {}
    assert set(named.populations) == set(named.mass_ledgers) == set(retain)


@pytest.mark.parametrize(
    "retain",
    [("final",), ("survey", "final"), ["kept"], ()],
    ids=("final", "survey-and-final", "kept", "none"),
)
def test_only_the_named_versions_and_their_mass_ledgers_are_attached(
    tmp_path: Path, retain: Collection[str]
) -> None:
    """A named version is attached exactly as the default run attaches it."""

    source = _source_path(tmp_path / "source")
    every = _run(_chain(), source, ContentStore(tmp_path / "every"))
    named = _run(_chain(), source, ContentStore(tmp_path / "named"), retain=retain)

    versions = {"survey", "kept", "final"}
    assert set(every.populations) == set(every.mass_ledgers) == versions
    assert set(named.populations) == set(named.mass_ledgers) == set(retain)
    for version in retain:
        _assert_same_version(named.population(version), every.population(version))
        assert named.mass_ledger(version) == every.mass_ledger(version)
    for version in versions - set(retain):
        with pytest.raises(KeyError, match="is not attached to this manifest"):
            named.population(version)
        with pytest.raises(KeyError, match="is not attached to this manifest"):
            named.mass_ledger(version)


# --------------------------------------------------------------------------
# When a version is released.
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("no_cyclic_collector")
@pytest.mark.parametrize("warm", (False, True), ids=("cold", "warm"))
def test_a_dead_version_is_released_by_its_last_reader(
    tmp_path: Path, warm: bool
) -> None:
    """On a structural chain each version goes with the boundary built on it.

    ``survey`` is last read by ``kept`` and ``kept`` by ``final``. With
    ``final`` retained, ``survey`` is dead by the time ``kept`` is observed,
    and ``kept`` by the time ``final`` is. Each was still there for its last
    reader to read, or the run would have failed. The default run keeps both
    to the end.
    """

    source = _source_path(tmp_path / "source")
    every_store = ContentStore(tmp_path / "every")
    named_store = ContentStore(tmp_path / "named")
    if warm:
        for store in (every_store, named_store):
            _run(_chain(), source, store)

    every = _Liveness()
    _run(_chain(), source, every_store, observer=every)
    named = _Liveness()
    manifest = _run(_chain(), source, named_store, observer=named, retain=("final",))

    assert all(receipt.hit is warm for receipt in manifest.nodes.values())
    assert list(named.alive) == list(compile_graph(_chain()).order)
    assert named.alive["kept"] == {"survey": False, "kept": True}
    assert named.alive["b"] == {"survey": False, "kept": True}
    assert named.alive["final"] == {"survey": False, "kept": False, "final": True}
    assert named.alive["c"] == {"survey": False, "kept": False, "final": True}
    assert every.alive["kept"] == {"survey": True, "kept": True}
    assert every.alive["c"] == {"survey": True, "kept": True, "final": True}
    assert named.alive_now() == {"survey": False, "kept": False, "final": True}
    # What is still alive is the attached version itself.
    attached = manifest.population("final")
    for entity, table in named.tables["final"].items():
        assert table() is attached.table(entity)


@pytest.mark.usefixtures("no_cyclic_collector")
@pytest.mark.parametrize("warm", (False, True), ids=("cold", "warm"))
def test_a_version_read_by_several_nodes_is_kept_until_the_last(
    tmp_path: Path, warm: bool
) -> None:
    """Two members and three structural nodes read ``survey``; it waits for all.

    Dropping it after a member, after ``first`` or after ``leaf`` would leave
    ``second`` with no base, and the run would fail. Nothing reads ``leaf``'s
    own version, so it goes with ``leaf``: it is dead by the next node.
    """

    source = _source_path(tmp_path / "source")
    every_store = ContentStore(tmp_path / "every")
    named_store = ContentStore(tmp_path / "named")
    if warm:
        for store in (every_store, named_store):
            _run(_shared_base(), source, store)

    every = _Liveness()
    every_manifest = _run(_shared_base(), source, every_store, observer=every)
    named = _Liveness()
    manifest = _run(
        _shared_base(), source, named_store, observer=named, retain=("first", "second")
    )

    assert all(receipt.hit is warm for receipt in manifest.nodes.values())
    assert list(named.alive) == list(compile_graph(_shared_base()).order)
    assert named.alive["first"]["survey"] is True
    assert named.alive["leaf"] == {"survey": True, "first": True, "leaf": True}
    assert named.alive["second"] == {
        "survey": False,
        "first": True,
        "leaf": False,
        "second": True,
    }
    assert named.alive["d"] == named.alive["second"]
    assert every.alive["d"] == {
        "survey": True,
        "first": True,
        "leaf": True,
        "second": True,
    }
    assert set(every_manifest.populations) == {"survey", "first", "leaf", "second"}
    assert set(manifest.populations) == {"first", "second"}
    assert manifest.population("second").table("person")["person_id"].tolist() == [
        1,
        3,
    ]


@pytest.mark.usefixtures("no_cyclic_collector")
@pytest.mark.parametrize(
    ("retain", "alive"), [(None, True), (("final",), False)], ids=("every", "final")
)
def test_a_dead_base_is_freed_before_its_last_reader_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retain: Collection[str] | None,
    alive: bool,
) -> None:
    """``kept`` writes with ``survey`` already gone, and ``final`` with ``kept``.

    A structural write copies every column of the node's own frame. A base
    still held there would add a whole version to that copy's peak. The
    default run keeps both bases, so they are alive there.
    """

    liveness = _Liveness()
    writes: dict[str, dict[str, bool]] = {}
    write = graph_executor._write_node

    def observed(store, node, **kwargs):
        writes[node.id] = liveness.alive_now()
        return write(store, node, **kwargs)

    monkeypatch.setattr(graph_executor, "_write_node", observed)
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    _run(_chain(), source, store, observer=liveness, retain=retain)

    assert list(writes) == list(compile_graph(_chain()).order)
    assert writes["kept"] == {"survey": alive, "kept": True}
    assert writes["final"] == {"survey": alive, "kept": alive, "final": True}


@pytest.mark.usefixtures("no_cyclic_collector")
@pytest.mark.parametrize(
    ("graph", "retain", "alive"),
    [
        (_chain, ("final",), {"survey": False, "kept": False, "final": True}),
        (_chain, (), {"survey": False, "kept": False, "final": False}),
        (_unreached_reader, (), {"survey": False}),
    ],
    ids=("chain-final", "chain-none", "unreached-none"),
)
def test_nothing_a_node_held_outlives_it_into_the_run_end_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graph: Callable[[], Graph],
    retain: Collection[str],
    alive: dict[str, bool],
) -> None:
    """The run-end source pass finds only the versions the caller retains.

    That pass re-reads every source in full. ``final``'s last user is its
    member ``c``, the last node to run. ``survey``'s, in the unreached graph,
    is a boundary that never runs, after the gate that last admitted it.
    Neither leaves its population behind for the pass.
    """

    liveness = _Liveness()
    passes: list[dict[str, bool]] = []
    derive = graph_executor._SourceIdentities.derive

    def observed(self, name, path):
        passes.append(liveness.alive_now())
        return derive(self, name, path)

    monkeypatch.setattr(graph_executor._SourceIdentities, "derive", observed)
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    _run(graph(), source, store, observer=liveness, retain=retain)

    assert passes == [alive]


@pytest.mark.parametrize(
    "retain", [(), ("boundary",)], ids=("none", "the-unreached-version")
)
def test_an_unreached_last_reader_still_releases_its_version(
    tmp_path: Path, retain: Collection[str]
) -> None:
    """``boundary`` never runs, but it is ``survey``'s last reader all the same.

    The default run attaches ``survey`` and has no ``boundary`` to attach. A run
    that does not retain ``survey`` attaches nothing, even when it names the
    version ``boundary`` would have opened.
    """

    source = _source_path(tmp_path / "source")
    every = _run(_unreached_reader(), source, ContentStore(tmp_path / "every"))
    named = _run(
        _unreached_reader(), source, ContentStore(tmp_path / "named"), retain=retain
    )

    assert named.nodes["boundary"].receipt["execution"]["state"] == "unreached"
    assert set(every.populations) == {"survey"}
    assert dict(named.populations) == {}
    assert dict(named.mass_ledgers) == {}
    assert named.key == every.key


# --------------------------------------------------------------------------
# What is refused.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retain", "error", "message"),
    [
        (("missing",), ValueError, r"\['missing'\], which are not population"),
        (("final", "a"), ValueError, r"\['a'\], which are not population"),
        ("final", TypeError, "a collection of version names"),
        (b"final", TypeError, "a collection of version names"),
        (iter(("final",)), TypeError, "a collection of version names"),
        (("final", 1), TypeError, "each version as a string"),
    ],
    ids=("unknown", "a-member", "a-string", "bytes", "an-iterator", "not-a-string"),
)
def test_a_name_that_is_not_a_version_is_refused_before_any_work(
    tmp_path: Path, retain: object, error: type[Exception], message: str
) -> None:
    """The source does not exist, so a refusal after any read would differ."""

    registry = _registry()
    store = ContentStore(tmp_path / "store")
    with pytest.raises(error, match=message):
        _run(
            _chain(),
            tmp_path / "absent",
            store,
            retain=retain,  # type: ignore[arg-type]
            registry=registry,
        )
    assert _object_bytes(store) == {}
    assert all(kernel.calls == 0 for kernel in registry.as_mapping().values())
