"""A kernel's report of a changed identity is never filed as a gate verdict.

Amendment 7 turns a raising gate into a ``fail`` verdict, and the executor
files that verdict under the node key, so every later run that derives the
key is served it. Keys are derived from each kernel's ``implementation_hash``
before any node runs. The review of #1125 (finding 1) found a kernel whose
identity can move during a run: ``gates.battery@1`` raises once its binding
registry, or a source file in a binding's closure, changed after the kernel
was built. So an edit made while a build ran stored ``fail`` under the
unedited registry's key, and a later run with the correct registry on that
store was served the ``fail``.

Amendment 29: the kernel raises ``KernelIdentityChangedError``, and the
executor never files it as a verdict. The run refuses, nothing for the node is
stored, and the nodes it already completed keep their stored results, since
their keys still name what computed them. The executor does not re-derive a
hash itself: most gate hashes re-read module files, which move when a file is
rewritten under a process whose imported code did not change, so only the
kernel can tell a real move from that.

Invariants, each executed below:

- **No stale outcome.** After any first run, a run with the unchanged kernel
  on the same store gets the same gate outcome as a run on a cold store (a
  differential between the shared store and a fresh one), for a kernel whose
  identity is in-memory state it checks and for one whose hash only re-reads
  files, under ``resume="auto"`` and ``"forbid"``.
- **Refusal is exactly a reported move.** The first run refuses if and only
  if the kernel reports that its identity moved. A hash that moves without a
  report refuses nothing. Otherwise the outcome is the one the unchanged
  kernel produces (amendment 7 untouched), and it is filed and reused.
- **A refusal stores nothing for the gate** and keeps every earlier node's
  record, so the next run reuses that work, computes the gate afresh, and
  then computes what follows it.

Synthetic toy graph only: no data file, no country package, no engine.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import microcosm.graph.executor as graph_executor
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import KernelIdentityChangedError, NodeRejectedError
from microcosm.graph.decl import (
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
from microcosm.graph.manifest import RunManifest
from microcosm.graph.population import Population
from microcosm.graph.store import ContentStore, StoreMiss

SOURCE = SourceRef("survey", "csv-tables")
CREATE = Node(
    "survey",
    "source@1",
    structural=StructuralDelta.CREATE,
    sources=("survey",),
    outputs=(Owned("person", "age", "int64"), Owned("household", "size", "int64")),
)
GATE = Node(
    "gate",
    "gate@1",
    inputs=(Slice("person", ("age",)),),
    outputs=(Owned("household", "gate_verdict", "string"),),
    population="survey",
)
AFTER = Node(
    "after",
    "after@1",
    inputs=(Slice("household", ("gate_verdict",)),),
    outputs=(Owned("household", "gate_copy", "string"),),
    population="survey",
)
GRAPH = Graph("toy", (SOURCE,), (CREATE, GATE, AFTER))
COMPILED = compile_graph(GRAPH)

#: What a gate does: return a verdict, or raise an ordinary exception, which
#: amendment 7 makes a ``fail`` verdict.
BEHAVIOURS = ("pass", "fail", "raise")
#: When the gate's identity moves: never, from the population observer before
#: the gate runs (the review's mid-run binding edit), or inside the gate's own
#: ``run`` after it started evaluating.
MOVES = ("none", "observer", "during")
#: ``memory``: the identity is state the kernel runs with and checks, as
#: ``gates.battery@1``'s bindings are. ``disk``: only the hash moves, as a
#: source hash does when a file is rewritten under code already imported.
KINDS = ("memory", "disk")
BASE = "base"
GATE_MESSAGE = "gate evidence unavailable"
MOVED_MESSAGE = "gate@1 is no longer the kernel it was built as"


class _Kernel:
    """A plain kernel with a fixed identity that counts its own executions."""

    def __init__(
        self,
        ref: str,
        capabilities: Capabilities,
        compute: Callable[[KernelContext], KernelResult],
    ) -> None:
        self.ref = ref
        self.capabilities = capabilities
        self.compute = compute
        self.calls = 0

    def implementation_hash(self) -> str:
        return hashlib.sha256(f"{self.ref}/{BASE}".encode()).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        self.calls += 1
        return self.compute(context)


class _Gate:
    """A gate whose identity is a mutable attribute.

    A ``memory`` gate behaves as its current identity (``behaviours``, else
    ``edited``) and keeps the battery's contract: its hash and its ``run``
    raise ``KernelIdentityChangedError`` once the identity left the one it
    was built as, ``run`` checking before it evaluates, before it returns and
    when evaluation raises. A ``disk`` gate always runs the behaviour it was
    built with and never reports; only its hash follows the identity.
    """

    ref = "gate@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE)

    def __init__(self, base: str, *, kind: str = "memory", edited: str = "pass"):
        self.kind = kind
        self.identity = BASE
        self.behaviours = {BASE: base}
        self.edited = edited
        self.during_run: Callable[[_Gate], None] | None = None
        self.calls = 0

    def _require_identity(self, cause: Exception | None = None) -> None:
        if self.kind == "memory" and self.identity != BASE:
            raise KernelIdentityChangedError(MOVED_MESSAGE) from cause

    def implementation_hash(self) -> str:
        self._require_identity()
        return hashlib.sha256(f"{self.ref}/{self.identity}".encode()).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        self.calls += 1
        self._require_identity()
        try:
            result = self._evaluate(context)
        except Exception as error:
            self._require_identity(cause=error)
            raise
        self._require_identity()
        return result

    def _evaluate(self, context: KernelContext) -> KernelResult:
        current = self.identity if self.kind == "memory" else BASE
        behaviour = self.behaviours.get(current, self.edited)
        if self.during_run is not None:
            self.during_run(self)
        if behaviour == "raise":
            raise LookupError(GATE_MESSAGE)
        ids = context.tables["household"]["household_id"]
        return KernelResult(
            columns={
                ("household", "gate_verdict"): pd.Series(
                    behaviour, index=ids, dtype="string"
                )
            },
            receipt={"outcome": behaviour, "evidence": {"identity": current}},
        )


class _Reports(_Gate):
    """A gate that reports a changed identity its hash cannot see."""

    def run(self, context: KernelContext) -> KernelResult:
        self.calls += 1
        raise KernelIdentityChangedError(MOVED_MESSAGE)


def _source(context: KernelContext) -> KernelResult:
    offset = int((context.sources["survey"] / "value.txt").read_text("utf-8"))
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20], dtype=np.int64),
            "age": np.asarray([10, 20, 30], dtype=np.int64) + offset,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([10, 20], dtype=np.int64),
            "size": np.asarray([2, 1], dtype=np.int64),
        }
    )
    return KernelResult(
        frame=Frame(
            {"person": person, "household": household},
            EntitySchema(group_entities=("household",)),
            {"household": Weights(np.asarray([1.0, 2.0]), WeightKind.DESIGN)},
            pd.Series(["a", "a", "b"], name="stratum"),
        )
    )


def _after(context: KernelContext) -> KernelResult:
    table = context.tables["household"]
    return KernelResult(
        columns={
            ("household", "gate_copy"): pd.Series(
                table["gate_verdict"].array.copy(),
                index=table["household_id"],
                dtype="string",
            )
        }
    )


@dataclass
class _Toy:
    gate: _Gate
    registry: KernelRegistry

    def calls(self) -> dict[str, int]:
        return {ref: k.calls for ref, k in self.registry.as_mapping().items()}


def _toy(gate: _Gate, after: _Kernel | None = None) -> _Toy:
    registry = KernelRegistry()
    registry.register(
        _Kernel(
            "source@1",
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
            _source,
        )
    )
    registry.register(gate)
    registry.register(
        after or _Kernel("after@1", Capabilities(Determinism.DETERMINISTIC), _after)
    )
    return _Toy(gate, registry)


def _source_dir(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "value.txt").write_text("0", encoding="utf-8")
    return root


def _run(
    toy: _Toy,
    source: Path,
    store: ContentStore,
    observer: Callable[[str, Population], None] | None = None,
    *,
    resume: str = "auto",
) -> RunManifest:
    return run_graph(
        COMPILED,
        sources={"survey": source},
        store=store,
        kernels=toy.registry,
        resume=resume,  # type: ignore[arg-type]
        _population_observer=observer,
    )


def _move(gate: _Gate, identity: str) -> None:
    gate.identity = identity


def _move_before_gate(gate: _Gate, identity: str) -> Callable[[str, Population], None]:
    """An observer that edits the gate's identity once ``survey`` is admitted."""

    def observe(node_id: str, population: Population) -> None:
        if node_id == CREATE.id:
            _move(gate, identity)

    return observe


def _record_stored(store: ContentStore, key: str) -> bool:
    return store.has(graph_executor._cache_record_key(key))


def _clean_keys(tmp_path: Path, source: Path) -> dict[str, str]:
    manifest = _run(_toy(_Gate("pass")), source, ContentStore(tmp_path / "keys"))
    return {node_id: receipt.key for node_id, receipt in manifest.nodes.items()}


def test_the_reviewers_probe_a_mid_run_edit_stores_no_verdict(tmp_path: Path) -> None:
    """Review of #1125, finding 1, on the toy graph: run 1, run 2, run 3.

    Before amendment 29, run 1 filed ``fail`` under the unedited key and run 2,
    with a correct kernel on the same store, was served it as a hit; only the
    cold store in run 3 passed.
    """
    source = _source_dir(tmp_path / "src")
    store = ContentStore(tmp_path / "store")
    edited = _toy(_Gate("pass"))

    with pytest.raises(KernelIdentityChangedError, match=MOVED_MESSAGE) as refused:
        _run(edited, source, store, _move_before_gate(edited.gate, "edited"))

    assert isinstance(refused.value, NodeRejectedError)
    assert edited.calls() == {"source@1": 1, "gate@1": 1, "after@1": 0}

    correct = _toy(_Gate("pass"))
    resumed = _run(correct, source, store)
    gate = resumed.nodes[GATE.id]
    assert gate.receipt["outcome"] == "pass"
    assert not gate.hit
    assert resumed.nodes[CREATE.id].hit
    assert correct.calls() == {"source@1": 0, "gate@1": 1, "after@1": 1}

    cold = _run(_toy(_Gate("pass")), source, ContentStore(tmp_path / "cold"))
    assert cold.nodes[GATE.id].receipt["outcome"] == "pass"
    assert cold.nodes[GATE.id].key == gate.key
    assert cold.key == resumed.key


def test_a_refused_gate_leaves_no_record_and_keeps_the_runs_earlier_work(
    tmp_path: Path,
) -> None:
    source = _source_dir(tmp_path / "src")
    keys = _clean_keys(tmp_path, source)
    for resume in ("auto", "forbid"):
        store = ContentStore(tmp_path / f"store-{resume}")
        edited = _toy(_Gate("pass"))
        with pytest.raises(KernelIdentityChangedError):
            _run(
                edited,
                source,
                store,
                _move_before_gate(edited.gate, "edited"),
                resume=resume,
            )

        assert _record_stored(store, keys[CREATE.id])
        assert not _record_stored(store, keys[GATE.id])
        assert not _record_stored(store, keys[AFTER.id])
        with pytest.raises(StoreMiss):
            _run(_toy(_Gate("pass")), source, store, resume="require")


def test_an_identity_moved_inside_the_gates_own_run_is_refused(
    tmp_path: Path,
) -> None:
    """A move after the first check is caught on the way out, pass or raise."""
    source = _source_dir(tmp_path / "src")
    for behaviour in BEHAVIOURS:
        store = ContentStore(tmp_path / f"store-{behaviour}")
        edited = _toy(_Gate(behaviour))
        edited.gate.during_run = lambda gate: _move(gate, "edited")
        with pytest.raises(KernelIdentityChangedError) as refused:
            _run(edited, source, store)
        if behaviour == "raise":
            assert isinstance(refused.value.__cause__, LookupError)
        else:
            assert refused.value.__cause__ is None

        again = _toy(_Gate(behaviour))
        resumed = _run(again, source, store)
        expected = "fail" if behaviour == "raise" else behaviour
        assert resumed.nodes[GATE.id].receipt["outcome"] == expected
        assert not resumed.nodes[GATE.id].hit


def test_a_kernels_report_is_never_a_verdict_whatever_its_role(
    tmp_path: Path,
) -> None:
    """The report is final even when the hash holds, and keeps its type."""
    source = _source_dir(tmp_path / "src")
    keys = _clean_keys(tmp_path, source)

    store = ContentStore(tmp_path / "gate")
    reporting = _toy(_Reports("pass"))
    with pytest.raises(KernelIdentityChangedError, match=MOVED_MESSAGE):
        _run(reporting, source, store)
    assert reporting.calls() == {"source@1": 1, "gate@1": 1, "after@1": 0}
    assert not _record_stored(store, keys[GATE.id])

    def report(context: KernelContext) -> KernelResult:
        raise KernelIdentityChangedError("after@1 is not the kernel it was built as")

    compute = _toy(
        _Gate("pass"),
        _Kernel("after@1", Capabilities(Determinism.DETERMINISTIC), report),
    )
    store = ContentStore(tmp_path / "compute")
    with pytest.raises(KernelIdentityChangedError, match="after@1 is not") as refused:
        _run(compute, source, store)
    # Re-raised as the kernel raised it, not wrapped in a generic rejection.
    assert refused.value.__cause__ is None
    assert _record_stored(store, keys[GATE.id])
    assert not _record_stored(store, keys[AFTER.id])


def test_a_gate_whose_identity_holds_still_files_its_failure(tmp_path: Path) -> None:
    """Amendment 7 is unchanged: a genuine failure is a verdict, filed and reused."""
    source = _source_dir(tmp_path / "src")
    store = ContentStore(tmp_path / "store")
    first = _run(_toy(_Gate("raise")), source, store)
    gate = first.nodes[GATE.id]
    assert gate.receipt["outcome"] == "fail"
    assert gate.receipt["evidence"] == {
        "exception_type": "LookupError",
        "message": GATE_MESSAGE,
    }
    assert _record_stored(store, gate.key)

    again = _toy(_Gate("raise"))
    replay = _run(again, source, store)
    assert all(receipt.hit for receipt in replay.nodes.values())
    assert sum(again.calls().values()) == 0


def test_a_hash_that_moves_without_a_report_is_not_second_guessed(
    tmp_path: Path,
) -> None:
    """The executor does not re-derive hashes, so a disk-hashed gate is kept.

    Most gate hashes re-read their module files. A file rewritten under a
    running process moves that hash without changing the code that runs, so
    the outcome is still the keyed computation's, and refusing it would abort
    a correct build. The gate runs as built, files its verdict under the key
    derived at run start, and a later run of the same kernel reuses it.
    """
    source = _source_dir(tmp_path / "src")
    store = ContentStore(tmp_path / "store")
    drifted = _toy(_Gate("pass", kind="disk"))

    first = _run(drifted, source, store, _move_before_gate(drifted.gate, "rewritten"))
    assert drifted.gate.implementation_hash() != first.nodes[GATE.id].kernel_impl_hash
    assert first.nodes[GATE.id].receipt["outcome"] == "pass"
    assert drifted.calls() == {"source@1": 1, "gate@1": 1, "after@1": 1}

    again = _toy(_Gate("pass", kind="disk"))
    replay = _run(again, source, store)
    assert replay.nodes[GATE.id].hit
    assert replay.nodes[GATE.id].key == first.nodes[GATE.id].key


def _clean_outcome(behaviour: str) -> str:
    """The outcome an unperturbed gate with this behaviour files."""
    return "fail" if behaviour == "raise" else behaviour


def _outcome(toy: _Toy, source: Path, store: ContentStore) -> RunManifest | None:
    try:
        return _run(toy, source, store)
    except KernelIdentityChangedError:
        return None


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    kind=st.sampled_from(KINDS),
    base=st.sampled_from(BEHAVIOURS),
    edited=st.sampled_from(BEHAVIOURS),
    move=st.sampled_from(MOVES),
    identity=st.one_of(st.just(BASE), st.text(min_size=1, max_size=6)),
    resume=st.sampled_from(("auto", "forbid")),
)
# The review's probe, under both resume policies.
@example(
    kind="memory", base="pass", edited="raise", move="observer", identity="x",
    resume="auto",
)  # fmt: skip
@example(
    kind="memory", base="pass", edited="raise", move="observer", identity="x",
    resume="forbid",
)  # fmt: skip
# Its stale-pass twin, a move inside the run, and a disk-only hash move.
@example(
    kind="memory", base="fail", edited="pass", move="observer", identity="x",
    resume="auto",
)  # fmt: skip
@example(
    kind="memory", base="raise", edited="pass", move="during", identity="x",
    resume="auto",
)  # fmt: skip
@example(
    kind="disk", base="raise", edited="pass", move="observer", identity="x",
    resume="auto",
)  # fmt: skip
# An edit that leaves the identity where it was is not a move.
@example(
    kind="memory", base="raise", edited="pass", move="observer", identity=BASE,
    resume="auto",
)  # fmt: skip
def test_the_store_never_serves_a_gate_outcome_its_key_does_not_name(
    tmp_path: Path,
    kind: str,
    base: str,
    edited: str,
    move: str,
    identity: str,
    resume: str,
) -> None:
    root = Path(tempfile.mkdtemp(dir=tmp_path))
    source = _source_dir(root / "src")
    store = ContentStore(root / "store")

    first = _toy(_Gate(base, kind=kind, edited=edited))
    observer = None
    if move == "observer":
        observer = _move_before_gate(first.gate, identity)
    elif move == "during":
        first.gate.during_run = lambda gate: _move(gate, identity)
    reported = kind == "memory" and move != "none" and identity != BASE

    try:
        manifest: RunManifest | None = _run(
            first, source, store, observer, resume=resume
        )
    except KernelIdentityChangedError:
        manifest = None
    # Refusal is exactly a reported move.
    assert (manifest is None) == reported
    if manifest is not None:
        assert manifest.nodes[GATE.id].receipt["outcome"] == _clean_outcome(base)

    # No stale outcome: the shared store answers as a cold store does.
    shared = _toy(_Gate(base, kind=kind, edited=edited))
    served = _outcome(shared, source, store)
    cold = _outcome(_toy(_Gate(base, kind=kind)), source, ContentStore(root / "c"))
    assert served is not None and cold is not None
    assert served.nodes[GATE.id].receipt["outcome"] == _clean_outcome(base)
    assert served.nodes[GATE.id].receipt == cold.nodes[GATE.id].receipt
    assert served.key == cold.key
    # A refusal stored nothing for the gate and kept the earlier work; a filed
    # outcome is reused.
    assert served.nodes[CREATE.id].hit
    assert served.nodes[GATE.id].hit == (manifest is not None)
    assert shared.calls()["gate@1"] == (0 if manifest is not None else 1)
