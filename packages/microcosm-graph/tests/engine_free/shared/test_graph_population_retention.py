"""A run that names the versions it retains frees every other one.

``run_graph(retain_populations=...)`` lets a caller say which population
versions it will read from the manifest. The default, ``None``, keeps every
version to the end of the run and attaches all of them, as before. Given a
collection, each version it does not name leaves the executor once the last
node that reads it has run, so its frame can be freed. These tests pin that
contract:

- the store, the node keys, the receipts and the portable manifest are the
  same with and without the keyword, cold, warm and under ``resume="require"``;
- only the named versions, and their mass ledgers, are attached;
- a dead version's tables are released after its last reader, and not before;
- a version read by several members and several structural nodes is kept
  until the last of them has run;
- an unreached last reader still releases its version;
- a name that is not a version of the graph is refused before the run reads a
  source, runs a kernel or writes to the store.

Synthetic toy graph only: no data file, no country package, no engine.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    """``survey`` read by two members and by two structural nodes.

    ``first`` and ``second`` both build on ``survey`` at one depth, so
    ``second`` is the last of its four readers. The run order is survey, a, b,
    first, second, c, d.
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


class _Liveness:
    """An observer that records which versions' tables are still alive.

    It holds only weak references, to the tables of the latest population
    admitted in each version, so it never keeps a version alive itself. At each
    node it first records which of the versions seen so far are alive. By then
    the executor has made its releases for every earlier node, and the node
    being admitted still holds its own incumbent. The executor must hand it the
    live population (``_population_observer_detach=False``): a detached
    snapshot is a copy that dies at once.

    It runs no garbage collection. A released version's tables die by
    reference counting alone, so they are freed at once, not at some later
    collection.
    """

    def __init__(self) -> None:
        self.tables: dict[str, dict[str, weakref.ref[pd.DataFrame]]] = {}
        self.alive: dict[str, dict[str, bool]] = {}

    def __call__(self, node_id: str, population: Population) -> None:
        self.alive[node_id] = self.alive_now()
        self.tables[population.version] = {
            entity: weakref.ref(population.frame.table(entity))
            for entity in population.frame.entities
        }

    def alive_now(self) -> dict[str, bool]:
        return {
            version: any(table() is not None for table in tables.values())
            for version, tables in self.tables.items()
        }


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
    "retain",
    [("final",), ("survey", "final"), frozenset({"kept"}), ()],
    ids=("final", "survey-and-final", "kept", "none"),
)
@pytest.mark.parametrize(
    ("warm", "resume"),
    [(False, "auto"), (True, "auto"), (True, "require")],
    ids=("cold", "warm", "require"),
)
def test_retention_moves_no_store_object_key_receipt_or_manifest_field(
    tmp_path: Path, warm: bool, resume: str, retain: Collection[str]
) -> None:
    """Two runs of one graph, with and without the keyword, into two stores."""

    source = _source_path(tmp_path / "source")
    every_store = ContentStore(tmp_path / "every")
    named_store = ContentStore(tmp_path / "named")
    if warm:
        for store in (every_store, named_store):
            _run(_chain(), source, store)
    every = _run(_chain(), source, every_store, resume=resume)
    named = _run(_chain(), source, named_store, resume=resume, retain=retain)

    assert _object_bytes(named_store) == _object_bytes(every_store)
    assert all(receipt.hit is warm for receipt in named.nodes.values())
    assert named.key == every.key
    assert _portable(named) == _portable(every)
    assert dict(named.source_identities) == dict(every.source_identities)
    assert dict(named.verification_epoch) == dict(every.verification_epoch) == {}


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


@pytest.mark.parametrize("warm", (False, True), ids=("cold", "warm"))
def test_a_dead_version_is_released_after_its_last_reader(
    tmp_path: Path, warm: bool
) -> None:
    """On a structural chain each version goes once the next boundary has run.

    ``survey`` is last read by ``kept`` and ``kept`` by ``final``. With
    ``final`` retained, ``survey`` is gone by the time ``b`` is admitted and
    ``kept`` by the time ``c`` is. The default run keeps both to the end.
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
    assert named.alive["kept"] == {"survey": True}
    assert named.alive["b"] == {"survey": False, "kept": True}
    assert named.alive["final"] == {"survey": False, "kept": True}
    assert named.alive["c"] == {"survey": False, "kept": False, "final": True}
    assert every.alive["c"] == {"survey": True, "kept": True, "final": True}
    assert named.alive_now() == {"survey": False, "kept": False, "final": True}
    # What is still alive is the attached version itself.
    attached = manifest.population("final")
    for entity, table in named.tables["final"].items():
        assert table() is attached.table(entity)


@pytest.mark.parametrize("warm", (False, True), ids=("cold", "warm"))
def test_a_version_read_by_several_nodes_is_kept_until_the_last(
    tmp_path: Path, warm: bool
) -> None:
    """Two members and two structural nodes read ``survey``; it waits for all four.

    Dropping it after a member or after ``first`` would leave ``second`` with
    no base, and the run would fail.
    """

    source = _source_path(tmp_path / "source")
    every_store = ContentStore(tmp_path / "every")
    named_store = ContentStore(tmp_path / "named")
    if warm:
        for store in (every_store, named_store):
            _run(_shared_base(), source, store)

    every = _Liveness()
    _run(_shared_base(), source, every_store, observer=every)
    named = _Liveness()
    manifest = _run(
        _shared_base(), source, named_store, observer=named, retain=("first", "second")
    )

    assert all(receipt.hit is warm for receipt in manifest.nodes.values())
    assert list(named.alive) == list(compile_graph(_shared_base()).order)
    assert named.alive["first"]["survey"] is True
    assert named.alive["second"]["survey"] is True
    assert named.alive["c"]["survey"] is False
    assert named.alive["d"] == {"survey": False, "first": True, "second": True}
    assert every.alive["d"]["survey"] is True
    assert set(manifest.populations) == {"first", "second"}
    assert manifest.population("second").table("person")["person_id"].tolist() == [
        1,
        3,
    ]


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
