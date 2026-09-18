"""Integrated execution, reuse, projection, and rejection contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.graph.executor as graph_executor
from microcosm.frame import (
    EntitySchema,
    Frame,
    LinkSpec,
    MassChangeRecord,
    WeightKind,
    Weights,
)
from microcosm.graph.availability import EXECUTION_SCHEMA
from microcosm.graph.decl import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Graph,
    GraphError,
    Node,
    Owned,
    Ownership,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
)
from microcosm.graph.errors import NodeRejectedError
from microcosm.graph.executor import NodeRejected, run_graph
from microcosm.graph.explain import explain_html
from microcosm.graph.kernel import (
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Numeric,
    NumericScope,
    Tolerance,
)
from microcosm.graph.keys import opaque_artifact_key, platform_fingerprint
from microcosm.graph.manifest import Decision, RunManifest
from microcosm.graph.population import MassRecord, Population
from microcosm.graph.store import (
    ContentStore,
    StoreCorrupt,
    StoreMiss,
    StoreUnavailable,
)

if "_toy" not in sys.modules:
    _TOY_SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    assert _TOY_SPEC is not None and _TOY_SPEC.loader is not None
    sys.modules["_toy"] = importlib.util.module_from_spec(_TOY_SPEC)
    _TOY_SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

SOURCE = SourceRef("survey", "csv-tables", description="toy bytes")
CREATE = Node(
    "survey",
    "source@1",
    sources=("survey",),
    structural=StructuralDelta.CREATE,
    outputs=(
        Owned("person", "age", "int64"),
        Owned("person", "income", "float64"),
        Owned("person", "selected", "boolean"),
        Owned("household", "size", "int64"),
    ),
)


class _Kernel:
    def __init__(
        self,
        ref: str,
        capabilities: Capabilities,
        compute: Callable[[KernelContext], KernelResult],
        *,
        implementation: str = "base",
    ) -> None:
        self.ref = ref
        self.capabilities = capabilities
        self.compute = compute
        self.implementation = implementation
        self.calls = 0

    def implementation_hash(self) -> str:
        return hashlib.sha256(f"{self.ref}/{self.implementation}".encode()).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        self.calls += 1
        return self.compute(context)


def _source_path(root: Path, value: int = 0) -> Path:
    root.mkdir()
    (root / "value.txt").write_text(str(value), encoding="utf-8")
    return root


def _source_frame(path: Path) -> Frame:
    offset = int((path / "value.txt").read_text(encoding="utf-8"))
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20], dtype=np.int64),
            "age": np.asarray([10, 20, 30], dtype=np.int64) + offset,
            "income": np.asarray([-0.0, 2.0, 3.0], dtype=np.float64),
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
    frame = _source_frame(context.sources["survey"])
    return KernelResult(frame=frame, receipt={"rows": frame.n("person")})


def _add(context: KernelContext) -> KernelResult:
    table = context.tables["person"]
    source = str(context.params["source"])
    target = str(context.params["target"])
    scale = float(context.params["scale"])
    values = table[source].to_numpy(dtype=np.float64) * scale
    return KernelResult(
        columns={
            ("person", target): pd.Series(
                values,
                index=pd.Index(table["person_id"], name="person_id"),
                dtype="float64",
            )
        },
        artifacts={"diagnostic": b"add-kernel"},
        receipt={"rows": len(values)},
    )


def _draw(context: KernelContext) -> KernelResult:
    table = context.tables["person"]
    target = str(context.params["target"])
    values = context.rng.uniform(size=len(table))
    return KernelResult(
        columns={
            ("person", target): pd.Series(
                values,
                index=pd.Index(table["person_id"], name="person_id"),
                dtype="float64",
            )
        },
        receipt={"draws": len(values)},
    )


def _registry(
    *,
    extra: _Kernel | None = None,
) -> KernelRegistry:
    deterministic = Capabilities(Determinism.DETERMINISTIC)
    registry = KernelRegistry()
    kernels = (
        _Kernel(
            "source@1",
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
            _source,
        ),
        _Kernel("a@1", deterministic, _add),
        _Kernel("b@1", deterministic, _add),
        _Kernel("leaf@1", deterministic, _draw),
    )
    for kernel in kernels:
        registry.register(kernel)
    if extra is not None:
        registry.register(extra)
    return registry


def _ordinary(
    node_id: str,
    kernel: str,
    source: str,
    target: str,
    *,
    scale: float = 1.0,
    description: str = "",
) -> Node:
    return Node(
        node_id,
        kernel,
        inputs=(Slice("person", (source,)),),
        outputs=(Owned("person", target, "float64"),),
        params={"source": source, "target": target, "scale": scale},
        population="survey",
        description=description,
    )


def _graph(*, scale: float = 2.0, leaf: bool = True, description: str = "") -> Graph:
    nodes = [
        replace(CREATE, description=description),
        _ordinary("a", "a@1", "age", "a", scale=scale, description=description),
        _ordinary("b", "b@1", "a", "b", scale=3.0, description=description),
    ]
    if leaf:
        nodes.append(
            Node(
                "leaf",
                "leaf@1",
                inputs=(Slice("person", ("age",)),),
                outputs=(Owned("person", "leaf", "float64"),),
                params={"target": "leaf"},
                population="survey",
                description=description,
            )
        )
    return Graph("toy", (SOURCE,), tuple(nodes))


def _calls(registry: KernelRegistry) -> dict[str, int]:
    return {ref: kernel.calls for ref, kernel in registry.as_mapping().items()}


def _object_bytes(store: ContentStore) -> dict[str, bytes]:
    return {
        path.relative_to(store.objects).as_posix(): path.read_bytes()
        for path in sorted(store.objects.rglob("*"))
        if path.is_file()
    }


def _run(
    graph: Graph,
    source: Path,
    store: ContentStore,
    registry: KernelRegistry,
    *,
    resume: str = "auto",
    decisions: tuple[Decision | Mapping[str, object], ...] = (),
):
    return run_graph(
        compile_graph(graph),
        sources={"survey": source},
        store=store,
        kernels=registry,
        resume=resume,  # type: ignore[arg-type]
        decisions=decisions,  # type: ignore[arg-type]
    )


def _release_graph(
    *, gate_outcome: str, tier_answer: str, requires: tuple[str, ...] = ()
) -> Graph:
    gate = Node(
        "gate",
        "gate@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("household", "gate_verdict", "string"),),
        params={"outcome": gate_outcome},
        population="survey",
    )
    bridge = Node(
        "bridge",
        "bridge@1",
        inputs=(Slice("household", ("gate_verdict",)),),
        outputs=(Owned("household", "gate_copy", "string"),),
        population="survey",
    )
    release = Node(
        "release",
        "release@1",
        inputs=(Slice("household", ("gate_copy",)),),
        outputs=(Owned("household", "tier", "string"),),
        params={"answer": tier_answer, "requires_decisions": requires},
        population="survey",
    )
    return Graph("toy", (SOURCE,), (CREATE, gate, bridge, release))


def _release_registry() -> KernelRegistry:
    def gate(context: KernelContext) -> KernelResult:
        ids = context.tables["household"]["household_id"]
        outcome = str(context.params["outcome"])
        return KernelResult(
            columns={
                ("household", "gate_verdict"): pd.Series(
                    outcome, index=ids, dtype="string"
                )
            },
            receipt={"outcome": outcome, "evidence": {"fixture": True}},
        )

    def bridge(context: KernelContext) -> KernelResult:
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

    def release(context: KernelContext) -> KernelResult:
        ids = context.tables["household"]["household_id"]
        answer = str(context.params["answer"])
        return KernelResult(
            columns={
                ("household", "tier"): pd.Series(answer, index=ids, dtype="string")
            },
            receipt={"tier": answer, "outcome": "kernel-answer"},
        )

    registry = _registry()
    registry.register(
        _Kernel(
            "gate@1",
            Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE),
            gate,
        )
    )
    registry.register(
        _Kernel("bridge@1", Capabilities(Determinism.DETERMINISTIC), bridge)
    )
    registry.register(
        _Kernel(
            "release@1",
            Capabilities(Determinism.DETERMINISTIC, role=KernelRole.RELEASE),
            release,
        )
    )
    return registry


def test_determinism_across_stores_and_zero_kernel_memoization(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    first_store = ContentStore(tmp_path / "first")
    second_store = ContentStore(tmp_path / "second")
    first_registry = _registry()
    second_registry = _registry()

    first = _run(_graph(), source, first_store, first_registry)
    second = _run(_graph(), source, second_store, second_registry)

    assert {node: item.key for node, item in first.nodes.items()} == {
        node: item.key for node, item in second.nodes.items()
    }
    assert _object_bytes(first_store) == _object_bytes(second_store)
    assert first.key == second.key
    assert all(not item.hit for item in first.nodes.values())
    assert first.nodes["survey"].frame_key is not None
    assert first.nodes["a"].frame_key is None
    assert first.nodes["a"].receipt["capabilities"]["determinism"] == (  # type: ignore[index]
        "deterministic"
    )
    diagnostic_key = first.nodes["a"].opaque_artifacts["diagnostic"]
    assert first_store.load_bytes(diagnostic_key) == b"add-kernel"
    for node_id in first.nodes:
        for coordinate, key in first.nodes[node_id].artifacts.items():
            pd.testing.assert_series_equal(
                first_store.load_column(key),
                second_store.load_column(second.nodes[node_id].artifacts[coordinate]),
                check_exact=True,
            )

    calls_before = _calls(first_registry)
    warm = _run(_graph(), source, first_store, first_registry)
    assert _calls(first_registry) == calls_before
    assert all(item.hit for item in warm.nodes.values())
    assert warm.key == first.key
    assert warm.population("survey").table("person")["b"].tolist() == [60, 120, 180]


def test_exact_param_kernel_source_invalidation_and_decision_node_invariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_path(tmp_path / "source")

    param_store = ContentStore(tmp_path / "param")
    baseline = _run(_graph(), source, param_store, _registry())
    changed_param = _run(_graph(scale=4.0), source, param_store, _registry())
    assert {node for node, item in changed_param.nodes.items() if not item.hit} == {
        "a",
        "b",
    }
    assert baseline.nodes["leaf"].key == changed_param.nodes["leaf"].key

    code_store = ContentStore(tmp_path / "code")
    code_baseline = _run(_graph(), source, code_store, _registry())
    changed_registry = _registry()
    monkeypatch.setattr(
        changed_registry.get("a@1"),
        "implementation_hash",
        lambda: "changed",
    )
    changed_code = _run(_graph(), source, code_store, changed_registry)
    assert {node for node, item in changed_code.nodes.items() if not item.hit} == {
        "a",
        "b",
    }
    assert code_baseline.nodes["leaf"].key == changed_code.nodes["leaf"].key

    source_store = ContentStore(tmp_path / "bytes")
    source_baseline = _run(_graph(), source, source_store, _registry())
    (source / "value.txt").write_text("1", encoding="utf-8")
    changed_source = _run(_graph(), source, source_store, _registry())
    assert all(not item.hit for item in changed_source.nodes.values())
    assert source_baseline.nodes["survey"].key != changed_source.nodes["survey"].key

    decision = {
        "name": "publish",
        "owner": "reviewer",
        "signature": "toy-signature-0001",
    }
    decided = _run(_graph(), source, source_store, _registry(), decisions=(decision,))
    assert all(item.hit for item in decided.nodes.values())
    assert {node: item.key for node, item in decided.nodes.items()} == {
        node: item.key for node, item in changed_source.nodes.items()
    }
    assert decided.key == replace(decided, decisions=()).key
    assert [dict(record) for record in decided.decisions] == [decision]


def test_decisions_supplied_to_a_run_do_not_move_the_manifest_key(
    tmp_path: Path,
) -> None:
    """F5: a release node's decision-derived outcome is provenance, not identity."""

    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    graph = _release_graph(
        gate_outcome="pass", tier_answer="certified", requires=("publish",)
    )
    undecided = _run(graph, source, store, _release_registry())
    decision = {
        "name": "publish",
        "owner": "reviewer",
        "signature": "toy-signature-0001",
    }
    decided = _run(graph, source, store, _release_registry(), decisions=(decision,))
    assert undecided.nodes["release"].receipt["outcome"] == "unreached"
    assert decided.nodes["release"].receipt["outcome"] == "pass"
    assert undecided.key == decided.key
    assert undecided.to_json() != decided.to_json()


def test_inert_fields_order_and_leaf_removal_do_not_move_survivors(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    baseline = _run(_graph(), source, store, _registry())
    described_graph = _graph(description="provenance only")
    described_graph = Graph(
        "renamed label",
        (replace(SOURCE, description="source prose"),),
        tuple(reversed(described_graph.nodes)),
    )
    described = _run(described_graph, source, store, _registry())
    assert all(item.hit for item in described.nodes.values())
    assert {node: item.key for node, item in baseline.nodes.items()} == {
        node: item.key for node, item in described.nodes.items()
    }

    removed = _run(_graph(leaf=False), source, store, _registry())
    assert all(item.hit for item in removed.nodes.values())
    assert {node: baseline.nodes[node].key for node in removed.nodes} == {
        node: item.key for node, item in removed.nodes.items()
    }


def test_resume_policies_preflight_and_forbid_reads_no_cached_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    _run(_graph(), source, store, _registry())

    required_registry = _registry()
    required = _run(_graph(), source, store, required_registry, resume="require")
    assert all(item.hit for item in required.nodes.values())
    assert sum(_calls(required_registry).values()) == 0

    missing_registry = _registry()
    with pytest.raises(StoreMiss, match="before execution"):
        _run(
            _graph(),
            source,
            ContentStore(tmp_path / "empty"),
            missing_registry,
            resume="require",
        )
    assert sum(_calls(missing_registry).values()) == 0

    forbid_registry = _registry()
    with monkeypatch.context() as write_only:

        def reject_cache_read(*args: object, **kwargs: object) -> None:
            raise AssertionError("resume='forbid' read an existing store object")

        write_only.setattr("microcosm.graph.store._verified_meta", reject_cache_read)
        forbidden = _run(_graph(), source, store, forbid_registry, resume="forbid")
    assert all(not item.hit for item in forbidden.nodes.values())
    assert sum(_calls(forbid_registry).values()) == len(forbidden.nodes)

    warm_again = _run(_graph(), source, store, _registry())
    assert all(item.hit for item in warm_again.nodes.values())


def _bad_node(kernel: str, *, dtype: str = "float64", absent: bool = False) -> Node:
    return Node(
        "bad",
        kernel,
        inputs=(Slice("person", ("selected",), rows="selected"),),
        outputs=(
            Owned(
                "person",
                "bad",
                dtype,
                rows="selected",
                ownership=Ownership.ABSENT if absent else Ownership.PRODUCED,
            ),
        ),
        population="survey",
    )


def _outside(context: KernelContext) -> KernelResult:
    return KernelResult(
        columns={
            ("person", "bad"): pd.Series(
                [1.0, 2.0, 3.0], index=[1, 2, 3], dtype="float64"
            )
        }
    )


def _dense_bool(context: KernelContext) -> KernelResult:
    ids = context.tables["person"]["person_id"]
    return KernelResult(
        columns={("person", "bad"): pd.Series(True, index=ids, dtype="bool")}
    )


def _nonnull_absent(context: KernelContext) -> KernelResult:
    ids = context.tables["person"]["person_id"]
    return KernelResult(
        columns={("person", "bad"): pd.Series(1.0, index=ids, dtype="float64")}
    )


@pytest.mark.parametrize(
    ("node", "compute", "match"),
    [
        (_bad_node("outside@1"), _outside, "owned ids"),
        (_bad_node("dense@1", dtype="boolean"), _dense_bool, "dtype"),
        (
            _bad_node("absent@1", absent=True),
            _nonnull_absent,
            "ABSENT",
        ),
    ],
)
def test_executor_rejects_ownership_dtype_and_absent_breaches(
    tmp_path: Path,
    node: Node,
    compute: Callable[[KernelContext], KernelResult],
    match: str,
) -> None:
    source = _source_path(tmp_path / "source")
    bad_kernel = _Kernel(node.kernel, Capabilities(Determinism.DETERMINISTIC), compute)
    registry = _registry(extra=bad_kernel)
    graph = Graph("toy", (SOURCE,), (CREATE, node))
    with pytest.raises(NodeRejected, match=match):
        _run(graph, source, ContentStore(tmp_path / "store"), registry)


def test_executor_detects_mutation_even_when_pandas_replaces_a_buffer(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def mutate(context: KernelContext) -> KernelResult:
        table = context.tables["person"]
        table["age"] = table["age"] + 1
        ids = table["person_id"]
        return KernelResult(
            columns={
                ("person", "bad"): pd.Series(
                    np.zeros(len(ids)), index=ids, dtype="float64"
                )
            }
        )

    node = Node(
        "bad",
        "mutate@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "bad", "float64"),),
        population="survey",
    )
    registry = _registry(
        extra=_Kernel(node.kernel, Capabilities(Determinism.DETERMINISTIC), mutate)
    )
    with pytest.raises(NodeRejected, match="mutated"):
        _run(
            Graph("toy", (SOURCE,), (CREATE, node)),
            source,
            ContentStore(tmp_path / "store"),
            registry,
        )


def test_context_contains_only_declared_entity_slices(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")

    def inspect_context(context: KernelContext) -> KernelResult:
        assert set(context.tables) == {"person"}
        assert set(context.tables["person"]) == {
            "person_id",
            "person_household_id",
            "age",
        }
        ids = context.tables["person"]["person_id"]
        return KernelResult(
            columns={
                ("person", "isolated"): pd.Series(
                    np.zeros(len(ids)), index=ids, dtype="float64"
                )
            }
        )

    node = Node(
        "isolated",
        "isolated@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "isolated", "float64"),),
        population="survey",
    )
    registry = _registry(
        extra=_Kernel(
            node.kernel, Capabilities(Determinism.DETERMINISTIC), inspect_context
        )
    )
    _run(
        Graph("toy", (SOURCE,), (CREATE, node)),
        source,
        ContentStore(tmp_path / "store"),
        registry,
    )


def test_output_entity_receives_only_its_structural_id_view(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")

    def cross_entity(context: KernelContext) -> KernelResult:
        assert set(context.tables) == {"person", "household"}
        assert set(context.weights) == {"person", "household"}
        assert set(context.tables["person"]) == {
            "person_id",
            "person_household_id",
            "age",
        }
        assert set(context.tables["household"]) == {"household_id"}
        ids = context.tables["household"]["household_id"]
        return KernelResult(
            columns={
                ("household", "score"): pd.Series(
                    np.zeros(len(ids)), index=ids, dtype="float64"
                )
            }
        )

    node = Node(
        "cross",
        "cross@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("household", "score", "float64"),),
        population="survey",
    )
    registry = _registry(
        extra=_Kernel(
            node.kernel, Capabilities(Determinism.DETERMINISTIC), cross_entity
        )
    )
    _run(
        Graph("toy", (SOURCE,), (CREATE, node)),
        source,
        ContentStore(tmp_path / "store"),
        registry,
    )


@pytest.mark.parametrize("explicit_rewrite_input", [False, True])
def test_rewrite_incumbent_is_projected_from_its_owned_declaration(
    tmp_path: Path,
    explicit_rewrite_input: bool,
) -> None:
    source = _source_path(tmp_path / "source")
    boundary_tolerance = Tolerance(atol=1e-6)

    def keep_all(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        return KernelResult(
            keep=pd.Series(True, index=person["person_id"], dtype="bool")
        )

    def rewrite(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        assert set(person) == {
            "person_id",
            "person_household_id",
            "age",
            "income",
        }
        # The FILTER that opens this version carries ``age`` and ``income``
        # from CREATE without writing them, so the tolerances a reader (and
        # a rewrite, for its incumbent) sees are the producer's — bitwise
        # here — not the tolerance-bound carrier's (charter C5).
        assert context.tolerances == {
            ("person", "age"): None,
            ("person", "income"): None,
        }

        return KernelResult(
            columns={
                ("person", "income"): pd.Series(
                    person["income"].to_numpy(copy=True) + 1.0,
                    index=pd.Index(person["person_id"], name="person_id"),
                    dtype="float64",
                )
            }
        )

    boundary = Node(
        "rewrite_boundary",
        "identity.filter@1",
        inputs=(Slice("person", ("selected",)),),
        structural=StructuralDelta.FILTER,
        base="survey",
    )
    rewriter = Node(
        "rewrite_income",
        "rewrite.income@1",
        inputs=(
            Slice("person", ("age", "income") if explicit_rewrite_input else ("age",)),
        ),
        outputs=(Owned("person", "income", "float64", rewrite=True),),
        population=boundary.id,
    )
    registry = _registry()
    registry.register(
        _Kernel(
            boundary.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                numeric=Numeric.TOLERANCE_BOUND,
                structural=StructuralDelta.FILTER,
                tolerance=boundary_tolerance,
            ),
            keep_all,
        )
    )
    registry.register(
        _Kernel(
            rewriter.kernel,
            Capabilities(Determinism.DETERMINISTIC),
            rewrite,
        )
    )

    manifest = _run(
        Graph("toy", (SOURCE,), (CREATE, boundary, rewriter)),
        source,
        ContentStore(tmp_path / f"store-{explicit_rewrite_input}"),
        registry,
    )

    assert manifest.population(boundary.id).table("person")["income"].tolist() == [
        1.0,
        3.0,
        4.0,
    ]


def test_masked_writer_preserves_upstream_tolerance_provenance(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    producer_tolerance = Tolerance(rtol=1e-5)

    def keep_all(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        return KernelResult(
            keep=pd.Series(True, index=person["person_id"], dtype="bool")
        )

    def patch_selected(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        selected = person["selected"].to_numpy(dtype=np.bool_)
        ids = person.loc[selected, "person_id"]
        return KernelResult(
            columns={
                ("person", "income"): pd.Series(
                    np.array([10.0, 30.0], dtype=np.float64),
                    index=pd.Index(ids, name="person_id"),
                    dtype="float64",
                )
            }
        )

    def audit_income(context: KernelContext) -> KernelResult:
        assert context.tolerances == {("person", "income"): producer_tolerance}
        return KernelResult(
            receipt={
                "outcome": "pass",
                "evidence": {"tolerance": {"rtol": 1e-5, "atol": 0.0, "ulps": 0}},
            }
        )

    create = replace(CREATE, kernel="masked.source@1")
    boundary = Node(
        "masked_boundary",
        "masked.filter@1",
        inputs=(Slice("person", ("selected",)),),
        structural=StructuralDelta.FILTER,
        base=create.id,
        mass="free",
    )
    masked = Node(
        "masked_income",
        "masked.writer@1",
        inputs=(Slice("person", ("selected",)),),
        outputs=(Owned("person", "income", "float64", rows="selected"),),
        population=boundary.id,
    )
    reader = Node(
        "audit_masked_income",
        "masked.gate@1",
        inputs=(Slice("person", ("income",)),),
        population=boundary.id,
    )
    registry = _registry()
    for kernel in (
        _Kernel(
            create.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                numeric=Numeric.TOLERANCE_BOUND,
                structural=StructuralDelta.CREATE,
                tolerance=producer_tolerance,
            ),
            _source,
        ),
        _Kernel(
            boundary.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                structural=StructuralDelta.FILTER,
            ),
            keep_all,
        ),
        _Kernel(
            masked.kernel,
            Capabilities(Determinism.DETERMINISTIC),
            patch_selected,
        ),
        _Kernel(
            reader.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                role=KernelRole.GATE,
            ),
            audit_income,
        ),
    ):
        registry.register(kernel)

    manifest = _run(
        Graph("toy", (SOURCE,), (create, boundary, masked, reader)),
        source,
        ContentStore(tmp_path / "store"),
        registry,
    )
    receipt = json.loads(manifest.to_json())["nodes"][reader.id]["receipt"]
    assert receipt["outcome"] == "pass"
    assert receipt["evidence"]["tolerance_writers"] == {
        "person.income": [create.id, masked.id]
    }


@pytest.mark.parametrize(
    ("first_numeric", "second_numeric"),
    [
        (Numeric.TOLERANCE_BOUND, Numeric.PLATFORM_BITWISE),
        (Numeric.PLATFORM_BITWISE, Numeric.TOLERANCE_BOUND),
    ],
    ids=["bounded-then-platform", "platform-then-bounded"],
)
def test_input_numeric_scopes_preserve_platform_across_masked_writer_order(
    tmp_path: Path,
    first_numeric: Numeric,
    second_numeric: Numeric,
) -> None:
    source = _source_path(tmp_path / "source")
    bound = Tolerance(rtol=1e-6, atol=2e-6, ulps=3)
    current_platform = platform_fingerprint()

    def write_full(target: str) -> Callable[[KernelContext], KernelResult]:
        def compute(context: KernelContext) -> KernelResult:
            person = context.tables["person"]
            return KernelResult(
                columns={
                    ("person", target): pd.Series(
                        person["age"].to_numpy(dtype=np.float64),
                        index=pd.Index(person["person_id"], name="person_id"),
                        dtype="float64",
                    )
                }
            )

        return compute

    def rewrite_selected(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        selected = person["selected"].to_numpy(dtype=np.bool_)
        return KernelResult(
            columns={
                ("person", "mixed_value"): pd.Series(
                    person.loc[selected, "mixed_value"].to_numpy(dtype=np.float64),
                    index=pd.Index(person.loc[selected, "person_id"], name="person_id"),
                    dtype="float64",
                )
            }
        )

    def keep_all(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        return KernelResult(
            keep=pd.Series(True, index=person["person_id"], dtype="bool")
        )

    expected = {
        ("person", "age"): NumericScope(),
        ("person", "mixed_value"): NumericScope(
            numeric=Numeric.TOLERANCE_BOUND,
            tolerance=bound,
            platform=current_platform,
        ),
        ("person", "platform_value"): NumericScope(
            numeric=Numeric.PLATFORM_BITWISE,
            platform=current_platform,
        ),
    }

    def audit(context: KernelContext) -> KernelResult:
        assert context.numerics == expected
        assert set(context.tolerances) == set(expected)
        for coordinate, scope in context.numerics.items():
            assert context.tolerances[coordinate] == scope.tolerance
        return KernelResult(receipt={"outcome": "pass", "evidence": {"scopes": 3}})

    def numeric_capabilities(numeric: Numeric) -> Capabilities:
        return Capabilities(
            Determinism.DETERMINISTIC,
            numeric=numeric,
            tolerance=bound if numeric is Numeric.TOLERANCE_BOUND else None,
        )

    create = replace(CREATE, kernel="numeric.source@1")
    boundary = Node(
        "numeric_boundary",
        "numeric.filter@1",
        inputs=(Slice("person", ("selected", "mixed_value", "platform_value")),),
        structural=StructuralDelta.FILTER,
        base=create.id,
        mass="free",
    )
    platform_only = Node(
        "platform_only",
        "numeric.platform-only@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "platform_value", "float64"),),
        population=create.id,
    )
    first = Node(
        "mixed_first",
        "numeric.mixed-first@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "mixed_value", "float64"),),
        population=create.id,
    )
    second = Node(
        "mixed_second",
        "numeric.mixed-second@1",
        inputs=(Slice("person", ("selected",)),),
        outputs=(
            Owned("person", "mixed_value", "float64", rows="selected", rewrite=True),
        ),
        population=boundary.id,
    )
    gate = Node(
        "numeric_gate",
        "numeric.gate@1",
        inputs=(Slice("person", ("age", "mixed_value", "platform_value")),),
        population=boundary.id,
    )
    registry = _registry()
    for kernel in (
        _Kernel(
            create.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                structural=StructuralDelta.CREATE,
            ),
            _source,
        ),
        _Kernel(
            platform_only.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                numeric=Numeric.PLATFORM_BITWISE,
            ),
            write_full("platform_value"),
        ),
        _Kernel(
            boundary.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                structural=StructuralDelta.FILTER,
            ),
            keep_all,
        ),
        _Kernel(
            first.kernel, numeric_capabilities(first_numeric), write_full("mixed_value")
        ),
        _Kernel(second.kernel, numeric_capabilities(second_numeric), rewrite_selected),
        _Kernel(
            gate.kernel,
            Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE),
            audit,
        ),
    ):
        registry.register(kernel)

    manifest = _run(
        Graph(
            "numeric-scopes",
            (SOURCE,),
            (create, boundary, platform_only, first, second, gate),
        ),
        source,
        ContentStore(tmp_path / "store"),
        registry,
    )
    assert manifest.nodes[gate.id].receipt["outcome"] == "pass"


def test_tolerance_gate_refuses_cross_platform_numeric_scope() -> None:
    coordinate = ("person", "income")
    current_platform = platform_fingerprint()
    gate = toy.GateReportsTolerance(
        "gate.platform-scope@1",
        Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE),
    )
    node = Node(
        "platform_scope_gate",
        gate.ref,
        inputs=(Slice("person", ("income",)),),
        outputs=(Owned("release", "platform_verdict", "string"),),
        params={
            "entity": "person",
            "column": "income",
            "verdict_column": "platform_verdict",
            "comparison_platform": "different/platform",
        },
    )
    context = KernelContext(
        node=node,
        tables={
            "person": pd.DataFrame(
                {"person_id": [1], "income": np.array([1.0], dtype=np.float64)}
            ),
            "release": pd.DataFrame({"release_id": [1]}),
        },
        weights={},
        strata=pd.Series(["a"], name="stratum"),
        params=node.params,
        rng=np.random.default_rng(0),
        tolerances={coordinate: None},
        numerics={
            coordinate: NumericScope(
                numeric=Numeric.PLATFORM_BITWISE,
                platform=current_platform,
            )
        },
    )

    result = gate.run(context)

    assert result.receipt["outcome"] == "evidence_absent"
    assert result.receipt["evidence"] == {
        "observed": 1.0,
        "tolerance": None,
        "numeric": "platform_bitwise",
        "platform": current_platform,
        "comparison_platform": "different/platform",
        "reason": "numeric contract is scoped to a different platform",
    }
    assert result.columns[("release", "platform_verdict")].tolist() == [
        "evidence_absent"
    ]


def test_filter_mask_result_is_applied_to_the_base_frame(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")

    def select_rows(context: KernelContext) -> KernelResult:
        table = context.tables["person"]
        return KernelResult(
            keep=pd.Series(
                table["selected"].to_numpy(dtype=np.bool_),
                index=table["person_id"],
                dtype="bool",
            )
        )

    node = Node(
        "selected",
        "filter@1",
        structural=StructuralDelta.FILTER,
        base="survey",
        inputs=(Slice("person", ("selected",)),),
        mass="free",
    )
    kernel = _Kernel(
        node.kernel,
        Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER),
        select_rows,
    )
    manifest = _run(
        Graph("toy", (SOURCE,), (CREATE, node)),
        source,
        ContentStore(tmp_path / "store"),
        _registry(extra=kernel),
    )
    assert manifest.population("selected").table("person")["person_id"].tolist() == [
        1,
        3,
    ]
    assert manifest.mass_ledger("selected")[-1].operation == "filter"


@pytest.mark.parametrize(
    (
        "producer_tolerance",
        "expand_tolerance",
        "claim_tolerance",
        "claim_expected",
        "reader_expected",
    ),
    [
        (
            Tolerance(rtol=1e-6),
            None,
            None,
            Tolerance(rtol=1e-6),
            Tolerance(rtol=1e-6),
        ),
        (
            None,
            Tolerance(atol=2e-6),
            Tolerance(ulps=3),
            Tolerance(atol=2e-6),
            Tolerance(atol=2e-6, ulps=3),
        ),
        (
            Tolerance(rtol=1e-6),
            Tolerance(atol=2e-6),
            Tolerance(ulps=3),
            Tolerance(rtol=1e-6, atol=2e-6),
            Tolerance(rtol=1e-6, atol=2e-6, ulps=3),
        ),
    ],
    ids=("producer-bound", "expand-and-claim-bound", "componentwise-maximum"),
)
def test_entrant_expand_aggregates_all_coordinate_writer_tolerances(
    tmp_path: Path,
    producer_tolerance: Tolerance | None,
    expand_tolerance: Tolerance | None,
    claim_tolerance: Tolerance | None,
    claim_expected: Tolerance,
    reader_expected: Tolerance,
) -> None:
    source = _source_path(tmp_path / "source")

    def capabilities(
        *,
        structural: StructuralDelta = StructuralDelta.NONE,
        tolerance: Tolerance | None = None,
        role: KernelRole = KernelRole.COMPUTE,
    ) -> Capabilities:
        if tolerance is None:
            return Capabilities(
                Determinism.DETERMINISTIC,
                structural=structural,
                role=role,
            )
        return Capabilities(
            Determinism.DETERMINISTIC,
            numeric=Numeric.TOLERANCE_BOUND,
            structural=structural,
            role=role,
            tolerance=tolerance,
        )

    def expand_household(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1], index=pd.Index([4], name="person_id"), dtype="int64"
                ),
                "household": pd.Series(
                    [pd.NA],
                    index=pd.Index([30], name="household_id"),
                    dtype="Int64",
                ),
            },
            columns={
                ("person", "person_household_id"): pd.Series(
                    [10, 10, 20, 30],
                    index=pd.Index([1, 2, 3, 4], name="person_id"),
                    dtype="int64",
                ),
                ("household", "size"): pd.Series(
                    [2, 1, 1],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="int64",
                ),
            },
            weights=Weights(
                np.array([1.0, 2.0, 1.0], dtype=np.float64), WeightKind.DESIGN
            ),
        )

    def claim_size(context: KernelContext) -> KernelResult:
        assert context.tolerances == {("household", "size"): claim_expected}
        household = context.tables["household"]
        return KernelResult(
            columns={
                ("household", "size"): pd.Series(
                    household["size"].array.copy(),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="int64",
                )
            }
        )

    def report_tolerance(context: KernelContext) -> KernelResult:
        assert context.tolerances == {("household", "size"): reader_expected}
        tolerance_payload = {
            "rtol": reader_expected.rtol,
            "atol": reader_expected.atol,
            "ulps": reader_expected.ulps,
        }
        household = context.tables["household"]
        return KernelResult(
            columns={
                ("household", "tolerance_verdict"): pd.Series(
                    ["pass"] * len(household),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="string",
                )
            },
            receipt={
                "outcome": "pass",
                "evidence": {"tolerance": tolerance_payload},
            },
        )

    create = replace(CREATE, kernel="writer.source@1")
    expand = Node(
        "admit_household",
        "writer.expand@1",
        structural=StructuralDelta.EXPAND,
        base=create.id,
        params={
            "expand_cells": (
                ("person", "person_household_id", "int64"),
                ("household", "size", "int64"),
            ),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
        entrants=True,
    )
    claim = Node(
        "claim_size",
        "writer.claim@1",
        outputs=(Owned("household", "size", "int64"),),
        params={"materialized_expand_outputs": ("household.size",)},
        population=expand.id,
    )
    reader = Node(
        "report_tolerance",
        "writer.gate@1",
        inputs=(Slice("household", ("size",)),),
        outputs=(Owned("household", "tolerance_verdict", "string"),),
        population=expand.id,
    )
    registry = _registry()
    for kernel in (
        _Kernel(
            create.kernel,
            capabilities(
                structural=StructuralDelta.CREATE,
                tolerance=producer_tolerance,
            ),
            _source,
        ),
        _Kernel(
            expand.kernel,
            capabilities(
                structural=StructuralDelta.EXPAND,
                tolerance=expand_tolerance,
            ),
            expand_household,
        ),
        _Kernel(
            claim.kernel,
            capabilities(tolerance=claim_tolerance),
            claim_size,
        ),
        _Kernel(
            reader.kernel,
            capabilities(role=KernelRole.GATE),
            report_tolerance,
        ),
    ):
        registry.register(kernel)

    manifest = _run(
        Graph("toy", (SOURCE,), (create, expand, claim, reader)),
        source,
        ContentStore(tmp_path / "store"),
        registry,
    )
    document = json.loads(manifest.to_json())
    expand_receipt = document["nodes"][expand.id]["receipt"]
    assert expand_receipt["expand_writes"]["household.size"] == ["entrant"]
    claim_receipt = document["nodes"][claim.id]["receipt"]
    assert claim_receipt["capabilities"]["tolerance_writers"] == {
        "household.size": [create.id, expand.id]
    }
    reader_receipt = document["nodes"][reader.id]["receipt"]
    assert reader_receipt["evidence"]["tolerance"] == {
        "rtol": reader_expected.rtol,
        "atol": reader_expected.atol,
        "ulps": reader_expected.ulps,
    }
    assert reader_receipt["evidence"]["tolerance_writers"] == {
        "household.size": [create.id, expand.id, claim.id]
    }


def test_expand_lineage_receipt_and_materialized_cell_survive_cache(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def expand(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1, 2],
                    index=pd.Index([4, 5], name="person_id"),
                    dtype="int64",
                ),
                "household": pd.Series(
                    [10],
                    index=pd.Index([30], name="household_id"),
                    dtype="int64",
                ),
            },
            columns={
                ("household", "is_clone"): pd.Series(
                    [False, False, True],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="bool",
                )
            },
            weights=Weights(
                np.array([0.5, 2.0, 0.5], dtype=np.float64),
                WeightKind.IMPORTANCE,
            ),
        )

    def claim(context: KernelContext) -> KernelResult:
        assert context.tolerances == {("household", "is_clone"): Tolerance(atol=9e-6)}
        household = context.tables["household"]
        assert set(household) == {"household_id", "is_clone"}
        return KernelResult(
            columns={
                ("household", "is_clone"): pd.Series(
                    household["is_clone"].array.copy(),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="bool",
                )
            }
        )

    clone = Node(
        "clone",
        "expand@1",
        structural=StructuralDelta.EXPAND,
        base="survey",
        params={
            "expand_cells": (("household", "is_clone", "bool"),),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="free",
        entrants=True,
    )
    claim_clone = Node(
        "claim_clone",
        "claim@1",
        outputs=(Owned("household", "is_clone", "bool"),),
        params={"materialized_expand_outputs": ("household.is_clone",)},
        population=clone.id,
    )
    expand_kernel = _Kernel(
        clone.kernel,
        Capabilities(
            Determinism.DETERMINISTIC,
            numeric=Numeric.TOLERANCE_BOUND,
            structural=StructuralDelta.EXPAND,
            tolerance=Tolerance(atol=9e-6),
        ),
        expand,
    )
    claim_kernel = _Kernel(
        claim_clone.kernel,
        Capabilities(Determinism.DETERMINISTIC),
        claim,
    )
    registry = _registry()
    registry.register(expand_kernel)
    registry.register(claim_kernel)
    graph = Graph("toy", (SOURCE,), (CREATE, clone, claim_clone))
    store = ContentStore(tmp_path / "store")

    cold = _run(graph, source, store, registry, resume="forbid")
    assert cold.nodes[clone.id].receipt["expand_writes"] == {
        "household.is_clone": ("new-column",)
    }
    assert dict(cold.nodes[clone.id].receipt["expand"]) == {
        "household": ((30, 10),),
        "person": ((4, 1), (5, 2)),
    }
    assert cold.nodes[claim_clone.id].receipt["capabilities"]["tolerance_writers"] == {
        "household.is_clone": (clone.id,)
    }
    assert cold.population(clone.id).table("person")[
        "person_household_id"
    ].tolist() == [10, 10, 20, 30, 30]
    assert cold.population(clone.id).table("household")["is_clone"].tolist() == [
        False,
        False,
        True,
    ]
    assert cold.mass_ledger(clone.id)[-1].operation == "expand"

    warm = _run(graph, source, store, registry)
    assert warm.nodes[clone.id].hit
    assert warm.nodes[claim_clone.id].hit
    assert (
        warm.nodes[clone.id].receipt["expand"] == cold.nodes[clone.id].receipt["expand"]
    )
    assert (
        warm.population(clone.id)
        .table("person")
        .equals(cold.population(clone.id).table("person"))
    )
    assert expand_kernel.calls == 1
    assert claim_kernel.calls == 1


@pytest.mark.parametrize(
    "coordinate",
    [("house.hold", "new_value"), ("household", "new.value")],
    ids=["entity", "column"],
)
def test_dotted_expand_cell_is_rejected_before_cold_or_warm_execution(
    tmp_path: Path,
    coordinate: tuple[str, str],
) -> None:
    source = _source_path(tmp_path / "source")
    entity, column = coordinate

    def expand(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1, 2],
                    index=pd.Index([4, 5], name="person_id"),
                    dtype="int64",
                ),
                "household": pd.Series(
                    [10],
                    index=pd.Index([30], name="household_id"),
                    dtype="int64",
                ),
            },
            columns={
                coordinate: pd.Series(
                    [False, False, True],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="bool",
                )
            },
            weights=Weights(
                np.array([0.5, 2.0, 0.5], dtype=np.float64),
                WeightKind.IMPORTANCE,
            ),
        )

    clone = Node(
        "dotted_clone",
        "expand.dotted@1",
        structural=StructuralDelta.EXPAND,
        base="survey",
        params={
            "expand_cells": ((entity, column, "bool"),),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="free",
        entrants=True,
    )
    expand_kernel = _Kernel(
        clone.kernel,
        Capabilities(
            Determinism.DETERMINISTIC,
            structural=StructuralDelta.EXPAND,
        ),
        expand,
    )
    registry = _registry()
    registry.register(expand_kernel)
    graph = Graph("toy", (SOURCE,), (CREATE, clone))
    store = ContentStore(tmp_path / "store")

    for resume in ("forbid", "auto"):
        with pytest.raises(NodeRejected, match=r"expand_cells.*dot-free"):
            _run(graph, source, store, registry, resume=resume)
    assert expand_kernel.calls == 0
    assert _object_bytes(store) == {}


def test_materialized_expand_claim_rejects_filter_population(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")

    def keep_all(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        return KernelResult(
            keep=pd.Series(True, index=person["person_id"], dtype="bool")
        )

    def claim_size(context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        return KernelResult(
            columns={
                ("household", "size"): pd.Series(
                    household["size"].array.copy(),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="int64",
                )
            }
        )

    filtered = Node(
        "filtered",
        "filter.claim-boundary@1",
        inputs=(Slice("person", ("selected",)),),
        structural=StructuralDelta.FILTER,
        base="survey",
        mass="free",
    )
    claimant = Node(
        "claim_filtered_size",
        "claim.filtered-size@1",
        outputs=(Owned("household", "size", "int64"),),
        params={"materialized_expand_outputs": ("household.size",)},
        population=filtered.id,
    )
    filter_kernel = _Kernel(
        filtered.kernel,
        Capabilities(
            Determinism.DETERMINISTIC,
            structural=StructuralDelta.FILTER,
        ),
        keep_all,
    )
    claim_kernel = _Kernel(
        claimant.kernel,
        Capabilities(Determinism.DETERMINISTIC),
        claim_size,
    )
    registry = _registry()
    registry.register(filter_kernel)
    registry.register(claim_kernel)
    graph = Graph("toy", (SOURCE,), (CREATE, filtered, claimant))
    store = ContentStore(tmp_path / "store")

    for _ in range(2):
        with pytest.raises(
            NodeRejected,
            match="claim_filtered_size.*filtered.*FILTER",
        ):
            _run(graph, source, store, registry)
    assert filter_kernel.calls == 1
    assert claim_kernel.calls == 0


def test_materialized_expand_claim_rejects_non_materialized_coordinate(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def copy_household(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1, 2],
                    index=pd.Index([4, 5], name="person_id"),
                    dtype="int64",
                ),
                "household": pd.Series(
                    [10],
                    index=pd.Index([30], name="household_id"),
                    dtype="int64",
                ),
            },
            weights=Weights(
                np.array([0.5, 2.0, 0.5], dtype=np.float64),
                WeightKind.IMPORTANCE,
            ),
        )

    def claim_size(context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        return KernelResult(
            columns={
                ("household", "size"): pd.Series(
                    household["size"].array.copy(),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="int64",
                )
            }
        )

    clone = Node(
        "clone",
        "expand.unmaterialized@1",
        structural=StructuralDelta.EXPAND,
        base="survey",
        params={
            "expand_cells": (),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="free",
    )
    claimant = Node(
        "claim_unmaterialized_size",
        "claim.unmaterialized-size@1",
        outputs=(Owned("household", "size", "int64"),),
        params={"materialized_expand_outputs": ("household.size",)},
        population=clone.id,
    )
    expand_kernel = _Kernel(
        clone.kernel,
        Capabilities(
            Determinism.DETERMINISTIC,
            structural=StructuralDelta.EXPAND,
        ),
        copy_household,
    )
    claim_kernel = _Kernel(
        claimant.kernel,
        Capabilities(Determinism.DETERMINISTIC),
        claim_size,
    )
    registry = _registry()
    registry.register(expand_kernel)
    registry.register(claim_kernel)
    graph = Graph("toy", (SOURCE,), (CREATE, clone, claimant))
    store = ContentStore(tmp_path / "store")

    for _ in range(2):
        with pytest.raises(
            NodeRejected,
            match=r"claim_unmaterialized_size.*household\.size.*clone",
        ):
            _run(graph, source, store, registry)
    assert expand_kernel.calls == 1
    assert claim_kernel.calls == 0


@pytest.mark.parametrize("lineage_dtype", ["int64", "Int64"])
def test_entrant_expand_with_zero_entrants_allows_declared_materialized_claim(
    tmp_path: Path, lineage_dtype: str
) -> None:
    """A declared-entrants node may carry a nullable lineage dtype even when it
    admits no entrant on this run."""

    source = _source_path(tmp_path / "source")

    def admit_no_entrants(context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        return KernelResult(
            expand={
                "person": pd.Series(
                    [],
                    index=pd.Index([], name="person_id", dtype="int64"),
                    dtype=lineage_dtype,
                ),
                "household": pd.Series(
                    [],
                    index=pd.Index([], name="household_id", dtype="int64"),
                    dtype=lineage_dtype,
                ),
            },
            columns={
                ("household", "size"): pd.Series(
                    household["size"].array.copy(),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="int64",
                )
            },
            weights=context.weights["household"],
        )

    def claim_size(context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        return KernelResult(
            columns={
                ("household", "size"): pd.Series(
                    household["size"].array.copy(),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="int64",
                )
            }
        )

    expand = Node(
        "zero_entrant_expand",
        "expand.zero-entrants@1",
        inputs=(Slice("household", ("size",)),),
        structural=StructuralDelta.EXPAND,
        base=CREATE.id,
        params={
            "expand_cells": (("household", "size", "int64"),),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
        entrants=True,
    )
    claimant = Node(
        "claim_zero_entrant_size",
        "claim.zero-entrant-size@1",
        outputs=(Owned("household", "size", "int64"),),
        params={"materialized_expand_outputs": ("household.size",)},
        population=expand.id,
    )
    expand_kernel = _Kernel(
        expand.kernel,
        Capabilities(
            Determinism.DETERMINISTIC,
            structural=StructuralDelta.EXPAND,
        ),
        admit_no_entrants,
    )
    claim_kernel = _Kernel(
        claimant.kernel,
        Capabilities(Determinism.DETERMINISTIC),
        claim_size,
    )
    registry = _registry()
    registry.register(expand_kernel)
    registry.register(claim_kernel)
    graph = Graph("toy", (SOURCE,), (CREATE, expand, claimant))
    store = ContentStore(tmp_path / "store")

    cold = _run(graph, source, store, registry)
    warm = _run(graph, source, store, registry)

    for manifest in (cold, warm):
        assert manifest.nodes[expand.id].receipt["expand_declared"] == (
            "household.size",
        )
        assert manifest.nodes[expand.id].receipt["expand_writes"] == {}
        assert manifest.nodes[claimant.id].receipt["capabilities"][
            "tolerance_writers"
        ] == {"household.size": (CREATE.id,)}
    assert not cold.nodes[expand.id].hit
    assert not cold.nodes[claimant.id].hit
    assert warm.nodes[expand.id].hit
    assert warm.nodes[claimant.id].hit
    assert expand_kernel.calls == 1
    assert claim_kernel.calls == 1


def test_entrant_expand_rejects_mutated_copied_carried_values(
    tmp_path: Path,
) -> None:
    expand, claim = toy.entrant_person_node(
        "mutated_copy", strata_mode="mutates_copied_income"
    )
    graph = toy.small_graph(nodes=(toy.CREATE, expand, claim))
    sources = {"survey": toy.copy_source(tmp_path / "source")}
    registry = toy.toy_registry()
    store = ContentStore(tmp_path / "store")
    source_person = toy.read_toy_frame(sources["survey"]).table("person")
    source_id = int(source_person["person_id"].iloc[0])
    copied_id = int(source_person["person_id"].max()) + 2

    for attempt in range(2):
        with pytest.raises(
            NodeRejected, match=r"mutated_copy.*person\.income"
        ) as error:
            toy.run_toy(
                graph,
                tmp_path / f"attempt-{attempt}",
                sources=sources,
                registry=registry,
                store=store,
            )
        assert f"({copied_id}, {source_id})" in str(error.value)

    calls = toy.calls_by_ref(registry)
    assert calls["source.csv@1"] == 1
    assert calls[expand.kernel] == 2
    assert calls[claim.kernel] == 0


def test_expand_allows_copied_value_declared_as_same_version_rewrite(
    tmp_path: Path,
) -> None:
    expand_tolerance = Tolerance(rtol=7e-6)

    def copy_with_rewritten_income(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        person_ids = pd.Index(person["person_id"], name="person_id")
        source_id = int(person_ids[0])
        copy_id = int(person_ids.max()) + 1
        target_ids = person_ids.append(
            pd.Index([copy_id], dtype="int64", name="person_id")
        )
        income = pd.concat(
            [
                person["income"].reset_index(drop=True),
                pd.Series([float(person["income"].iloc[0]) + 1.0]),
            ],
            ignore_index=True,
        )
        return KernelResult(
            expand={
                "person": pd.Series(
                    [source_id],
                    index=pd.Index([copy_id], dtype="int64", name="person_id"),
                    dtype="int64",
                ),
                "household": pd.Series(
                    [],
                    index=pd.Index([], dtype="int64", name="household_id"),
                    dtype="int64",
                ),
            },
            columns={
                ("person", "income"): pd.Series(
                    income.array, index=target_ids, dtype="float64"
                )
            },
            weights=context.weights["household"],
        )

    def claim_rewritten_income(context: KernelContext) -> KernelResult:
        assert context.tolerances == {("person", "income"): expand_tolerance}
        person = context.tables["person"]
        return KernelResult(
            columns={
                ("person", "income"): pd.Series(
                    person["income"].array.copy(),
                    index=pd.Index(person["person_id"], name="person_id"),
                    dtype="float64",
                )
            }
        )

    expand = Node(
        "copy_rewritten_income",
        "copy.rewritten_income@1",
        inputs=(
            Slice("person", ("income",)),
            Slice("household", ("size",)),
        ),
        structural=StructuralDelta.EXPAND,
        base="survey",
        params={
            "expand_cells": (("person", "income", "float64"),),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
    )
    claim = Node(
        "claim_rewritten_income",
        "claim.rewritten_income@1",
        outputs=(Owned("person", "income", "float64", rewrite=True),),
        population=expand.id,
    )
    registry = _registry()
    registry.register(
        _Kernel(
            expand.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                numeric=Numeric.TOLERANCE_BOUND,
                structural=StructuralDelta.EXPAND,
                tolerance=expand_tolerance,
            ),
            copy_with_rewritten_income,
        )
    )
    registry.register(
        _Kernel(
            claim.kernel,
            Capabilities(Determinism.DETERMINISTIC),
            claim_rewritten_income,
        )
    )
    graph = Graph("toy", (SOURCE,), (CREATE, expand, claim))
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")

    cold = _run(graph, source, store, registry, resume="forbid")
    warm = _run(graph, source, store, registry)

    for run in (cold, warm):
        assert run.population(expand.id).table("person")["income"].iloc[-1] == 1.0
        assert run.nodes[expand.id].receipt["expand_writes"] == {
            "person.income": ("copied-rewrite",)
        }
        assert run.nodes[claim.id].receipt["capabilities"]["tolerance_writers"] == {
            "person.income": (CREATE.id, expand.id)
        }
    assert not cold.nodes[expand.id].hit
    assert warm.nodes[expand.id].hit
    assert warm.nodes[claim.id].hit


def test_group_entrant_expand_manifest_json_carries_mass_record(
    tmp_path: Path,
) -> None:
    expand, claim = toy.entrant_expand_node()
    graph = toy.small_graph(nodes=(toy.CREATE, expand, claim))
    sources = {"survey": toy.copy_source(tmp_path / "source")}
    registry = toy.toy_registry()
    store = ContentStore(tmp_path / "store")
    cold = toy.run_toy(
        graph,
        tmp_path / "cold",
        sources=sources,
        registry=registry,
        store=store,
    )
    warm = toy.run_toy(
        graph,
        tmp_path / "warm",
        sources=sources,
        registry=registry,
        store=store,
    )

    for run in (cold, warm):
        ledger = run.manifest.mass_ledger(expand.id)[-1]
        document = json.loads(run.manifest.to_json())
        mass = document["nodes"][expand.id]["receipt"]["mass"]
        assert mass == {
            "policy": ledger.policy,
            "before": ledger.before_total,
            "after": ledger.after_total,
            "stratum_before": {
                str(key): value for key, value in ledger.before_by_stratum
            },
            "stratum_after": {
                str(key): value for key, value in ledger.after_by_stratum
            },
        }
        assert mass["after"] - mass["before"] == 125.0
    assert warm.manifest.nodes[expand.id].hit


def test_expand_id_overlay_is_rejected_without_committing_cache(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")

    def replace_lineage_id(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1], index=pd.Index([4], name="person_id"), dtype="int64"
                ),
                "household": pd.Series(
                    [10], index=pd.Index([30], name="household_id"), dtype="int64"
                ),
            },
            columns={
                ("person", "person_household_id"): pd.Series(
                    [10, 10, 20, 40],
                    index=pd.Index([1, 2, 3, 4], name="person_id"),
                    dtype="int64",
                ),
                ("household", "household_id"): pd.Series(
                    [10, 20, 40],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="int64",
                ),
            },
            weights=Weights(
                np.array([1.0, 2.0, 1.0], dtype=np.float64), WeightKind.DESIGN
            ),
        )

    expand = Node(
        "replace_lineage_id",
        "bad.expand@1",
        structural=StructuralDelta.EXPAND,
        base="survey",
        params={
            "expand_cells": (
                ("person", "person_household_id", "int64"),
                ("household", "household_id", "int64"),
            ),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
    )
    kernel = _Kernel(
        expand.kernel,
        Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.EXPAND),
        replace_lineage_id,
    )
    registry = _registry(extra=kernel)
    graph = Graph("toy", (SOURCE,), (CREATE, expand))
    store = ContentStore(tmp_path / "store")

    for _ in range(2):
        with pytest.raises(NodeRejected, match="cannot overlay entity id column"):
            _run(graph, source, store, registry)
        if kernel.calls == 1:
            first_store_bytes = _object_bytes(store)
        else:
            assert _object_bytes(store) == first_store_bytes
    assert kernel.calls == 2


def test_partitioned_graph_accepts_structural_entrant_partition_values(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def create_partitioned(context: KernelContext) -> KernelResult:
        original = _source_frame(context.sources["survey"])
        tables = {entity: original.table(entity).copy() for entity in original.entities}
        tables["household"]["period"] = np.array([2024, 2025], dtype=np.int64)
        return KernelResult(
            frame=Frame(
                tables,
                original.schema,
                {"household": original.weights_for("household")},
                original.strata,
            )
        )

    def admit_household(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1], index=pd.Index([4], name="person_id"), dtype="int64"
                ),
                "household": pd.Series(
                    [pd.NA],
                    index=pd.Index([30], name="household_id"),
                    dtype="Int64",
                ),
            },
            columns={
                ("person", "person_household_id"): pd.Series(
                    [10, 10, 20, 30],
                    index=pd.Index([1, 2, 3, 4], name="person_id"),
                    dtype="int64",
                ),
                ("household", "size"): pd.Series(
                    [2, 1, 1],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="int64",
                ),
                ("household", "period"): pd.Series(
                    [2024, 2025, 2026],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="int64",
                ),
            },
            weights=Weights(
                np.array([1.0, 2.0, 1.0], dtype=np.float64), WeightKind.DESIGN
            ),
        )

    def pass_through(column: str) -> Callable[[KernelContext], KernelResult]:
        def run(context: KernelContext) -> KernelResult:
            household = context.tables["household"]
            return KernelResult(
                columns={
                    ("household", column): pd.Series(
                        household[column].array.copy(),
                        index=pd.Index(household["household_id"], name="household_id"),
                        dtype="int64",
                    )
                }
            )

        return run

    partition_tolerance = Tolerance(atol=4e-6)

    def audit_period(context: KernelContext) -> KernelResult:
        assert context.tolerances == {("household", "period"): partition_tolerance}
        return KernelResult(
            receipt={
                "outcome": "pass",
                "evidence": {"tolerance": {"rtol": 0.0, "atol": 4e-6, "ulps": 0}},
            }
        )

    create = replace(
        CREATE,
        kernel="partition.source@1",
        outputs=(*CREATE.outputs, Owned("household", "period", "int64")),
    )
    expand = Node(
        "admit_household",
        "partition.expand@1",
        structural=StructuralDelta.EXPAND,
        base="survey",
        params={
            "expand_cells": (
                ("person", "person_household_id", "int64"),
                ("household", "size", "int64"),
                ("household", "period", "int64"),
            ),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
        entrants=True,
    )
    claim_size = Node(
        "claim_size",
        "claim.size@1",
        outputs=(Owned("household", "size", "int64"),),
        params={"materialized_expand_outputs": ("household.size",)},
        population=expand.id,
    )
    period_gate = Node(
        "audit_period",
        "gate.partition-tolerance@1",
        inputs=(Slice("household", ("period",)),),
        population=expand.id,
    )
    kernels = (
        _Kernel(
            create.kernel,
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
            create_partitioned,
        ),
        _Kernel(
            expand.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                numeric=Numeric.TOLERANCE_BOUND,
                structural=StructuralDelta.EXPAND,
                tolerance=partition_tolerance,
            ),
            admit_household,
        ),
        _Kernel(
            claim_size.kernel,
            Capabilities(Determinism.DETERMINISTIC),
            pass_through("size"),
        ),
        _Kernel(
            period_gate.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                role=KernelRole.GATE,
            ),
            audit_period,
        ),
    )

    def registry(*extra: _Kernel) -> KernelRegistry:
        result = _registry()
        for kernel in (*kernels, *extra):
            result.register(kernel)
        return result

    graph = Graph(
        "toy",
        (SOURCE,),
        (create, expand, claim_size, period_gate),
        mass_partition=("household", "period"),
    )
    store = ContentStore(tmp_path / "store")
    first_registry = registry()
    cold = _run(graph, source, store, first_registry)
    warm = _run(graph, source, store, first_registry)

    for manifest in (cold, warm):
        assert manifest.population(expand.id).table("household")["period"].tolist() == [
            2024,
            2025,
            2026,
        ]
        partition = manifest.nodes[expand.id].receipt["mass"]["partition"]  # type: ignore[index]
        assert (partition["entity"], partition["column"]) == ("household", "period")
        document = json.loads(manifest.to_json())
        period_receipt = document["nodes"][period_gate.id]["receipt"]
        assert period_receipt["capabilities"]["tolerance_writers"] == {
            "household.period": [create.id, expand.id]
        }
        assert period_receipt["evidence"]["tolerance_writers"] == {
            "household.period": [create.id, expand.id]
        }
    assert warm.nodes[expand.id].hit
    assert warm.nodes[claim_size.id].hit
    assert warm.nodes[period_gate.id].hit

    claim_period = Node(
        "claim_period",
        "claim.period@1",
        outputs=(Owned("household", "period", "int64"),),
        params={"materialized_expand_outputs": ("household.period",)},
        population=expand.id,
    )
    period_kernel = _Kernel(
        claim_period.kernel,
        Capabilities(Determinism.DETERMINISTIC),
        pass_through("period"),
    )
    # Compilation refuses the owner before the executor's own guard can.
    with pytest.raises(GraphError, match="owns mass partition"):
        _run(
            Graph(
                "toy",
                (SOURCE,),
                (create, expand, claim_size, claim_period),
                mass_partition=("household", "period"),
            ),
            source,
            ContentStore(tmp_path / "ordinary-owner"),
            registry(period_kernel),
        )


def test_create_rejects_undeclared_frame_columns(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")

    def create_extra(context: KernelContext) -> KernelResult:
        original = _source_frame(context.sources["survey"])
        tables = {entity: original.table(entity).copy() for entity in original.entities}
        tables["person"]["hidden"] = np.ones(original.n("person"), dtype=np.int64)
        frame = Frame(
            tables,
            original.schema,
            {
                entity: original.weights_for(entity)
                for entity in original.weighted_entities
            },
            original.strata,
        )
        return KernelResult(frame=frame)

    create = replace(CREATE, kernel="source-extra@1")
    kernel = _Kernel(
        create.kernel,
        Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
        create_extra,
    )
    with pytest.raises(NodeRejected, match="exactly equal"):
        _run(
            Graph("toy", (SOURCE,), (create,)),
            source,
            ContentStore(tmp_path / "store"),
            _registry(extra=kernel),
        )


def test_malformed_kernel_result_containers_are_node_rejections(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def malformed(context: KernelContext) -> KernelResult:
        return KernelResult(columns=None)  # type: ignore[arg-type]

    node = _ordinary("malformed", "malformed@1", "age", "bad")
    kernel = _Kernel(node.kernel, Capabilities(Determinism.DETERMINISTIC), malformed)
    with pytest.raises(NodeRejected, match="columns is not a mapping"):
        _run(
            Graph("toy", (SOURCE,), (CREATE, node)),
            source,
            ContentStore(tmp_path / "store"),
            _registry(extra=kernel),
        )


def test_corrupt_cache_and_unavailable_codec_abort_before_recompute(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    manifest = _run(_graph(), source, store, _registry())
    artifact = manifest.nodes["a"].artifacts[("person", "a")]
    payload = store.object_path(artifact) / "values.npy"
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)

    registry = _registry()
    with pytest.raises(StoreCorrupt):
        _run(_graph(), source, store, registry)
    assert sum(_calls(registry).values()) == 0

    missing_codec_graph = Graph(
        "toy",
        (SourceRef("survey", "missing-codec"),),
        (CREATE,),
    )
    missing_registry = _registry()
    with pytest.raises(StoreUnavailable):
        _run(
            missing_codec_graph,
            source,
            ContentStore(tmp_path / "unavailable"),
            missing_registry,
        )
    assert sum(_calls(missing_registry).values()) == 0

    isolated_registry = _registry()
    with pytest.raises(StoreUnavailable, match="csv-tables"):
        _run(
            _graph(),
            source,
            ContentStore(tmp_path / "isolated", codecs={}),
            isolated_registry,
        )
    assert sum(_calls(isolated_registry).values()) == 0


def test_entrants_need_a_design_anchor_after_a_reweight(tmp_path: Path) -> None:
    """B6 / amendment 11: an entrant is admitted only while weights are design."""

    source = _source_path(tmp_path / "source")

    def reweight(context: KernelContext) -> KernelResult:
        before = context.weights["household"].values
        return KernelResult(
            weights=Weights(before * 2, WeightKind.IMPORTANCE),
            receipt={
                "mass": {
                    "policy": "free",
                    "before": 4.0,
                    "after": 8.0,
                    "stratum_before": {"a": 2.0, "b": 2.0},
                    "stratum_after": {"a": 4.0, "b": 4.0},
                }
            },
        )

    def admit(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1], index=pd.Index([4], name="person_id"), dtype="int64"
                ),
                "household": pd.Series(
                    [pd.NA], index=pd.Index([30], name="household_id"), dtype="Int64"
                ),
            },
            columns={
                ("person", "person_household_id"): pd.Series(
                    [10, 10, 20, 30],
                    index=pd.Index([1, 2, 3, 4], name="person_id"),
                    dtype="int64",
                ),
                ("household", "size"): pd.Series(
                    [2, 1, 1],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="int64",
                ),
            },
            weights=Weights(
                np.array([2.0, 4.0, 1.0], dtype=np.float64), WeightKind.IMPORTANCE
            ),
        )

    pool = Node(
        "pool",
        "reweight@1",
        structural=StructuralDelta.REWEIGHT,
        base="survey",
        inputs=(Slice("household", ("size",)),),
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    entrant = Node(
        "admit_household",
        "admit@1",
        structural=StructuralDelta.EXPAND,
        base=pool.id,
        params={
            "expand_cells": (
                ("person", "person_household_id", "int64"),
                ("household", "size", "int64"),
            ),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="free",
        entrants=True,
    )
    registry = _registry(
        extra=_Kernel(
            pool.kernel,
            Capabilities(
                Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
            ),
            reweight,
        )
    )
    registry.register(
        _Kernel(
            entrant.kernel,
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.EXPAND),
            admit,
        )
    )

    def claim_size(context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        return KernelResult(
            columns={
                ("household", "size"): pd.Series(
                    household["size"].array.copy(),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="int64",
                )
            }
        )

    claim = Node(
        "claim_size",
        "claim@1",
        outputs=(Owned("household", "size", "int64"),),
        params={"materialized_expand_outputs": ("household.size",)},
        population=entrant.id,
    )
    registry.register(
        _Kernel(claim.kernel, Capabilities(Determinism.DETERMINISTIC), claim_size)
    )
    with pytest.raises(NodeRejected, match=r"cannot anchor new 'household' ids"):
        _run(
            Graph("toy", (SOURCE,), (CREATE, pool, entrant, claim)),
            source,
            ContentStore(tmp_path / "store"),
            registry,
        )


def test_expand_cannot_regress_the_base_weight_kind(tmp_path: Path) -> None:
    """B6 / D1: weight kinds only move forward; declaring design on a reweighted base is refused."""

    source = _source_path(tmp_path / "source")

    def reweight(context: KernelContext) -> KernelResult:
        before = context.weights["household"].values
        return KernelResult(
            weights=Weights(before * 2, WeightKind.IMPORTANCE),
            receipt={
                "mass": {
                    "policy": "free",
                    "before": 4.0,
                    "after": 8.0,
                    "stratum_before": {"a": 2.0, "b": 2.0},
                    "stratum_after": {"a": 4.0, "b": 4.0},
                }
            },
        )

    def admit(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1], index=pd.Index([4], name="person_id"), dtype="int64"
                ),
                "household": pd.Series(
                    [pd.NA], index=pd.Index([30], name="household_id"), dtype="Int64"
                ),
            },
            columns={
                ("person", "person_household_id"): pd.Series(
                    [10, 10, 20, 30],
                    index=pd.Index([1, 2, 3, 4], name="person_id"),
                    dtype="int64",
                ),
                ("household", "size"): pd.Series(
                    [2, 1, 1],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="int64",
                ),
            },
            weights=Weights(
                np.array([1.0, 2.0, 1.0], dtype=np.float64), WeightKind.DESIGN
            ),
        )

    pool = Node(
        "pool",
        "reweight@1",
        structural=StructuralDelta.REWEIGHT,
        base="survey",
        inputs=(Slice("household", ("size",)),),
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    entrant = Node(
        "admit_household",
        "admit@1",
        structural=StructuralDelta.EXPAND,
        base=pool.id,
        params={
            "expand_cells": (
                ("person", "person_household_id", "int64"),
                ("household", "size", "int64"),
            ),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
        entrants=True,
    )
    registry = _registry(
        extra=_Kernel(
            pool.kernel,
            Capabilities(
                Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
            ),
            reweight,
        )
    )
    registry.register(
        _Kernel(
            entrant.kernel,
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.EXPAND),
            admit,
        )
    )

    def claim_size(context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        return KernelResult(
            columns={
                ("household", "size"): pd.Series(
                    household["size"].array.copy(),
                    index=pd.Index(household["household_id"], name="household_id"),
                    dtype="int64",
                )
            }
        )

    claim = Node(
        "claim_size",
        "claim@1",
        outputs=(Owned("household", "size", "int64"),),
        params={"materialized_expand_outputs": ("household.size",)},
        population=entrant.id,
    )
    registry.register(
        _Kernel(claim.kernel, Capabilities(Determinism.DETERMINISTIC), claim_size)
    )
    with pytest.raises(
        NodeRejected,
        match=r"cannot regress 'household' weights from 'importance' to 'design'",
    ):
        _run(
            Graph("toy", (SOURCE,), (CREATE, pool, entrant, claim)),
            source,
            ContentStore(tmp_path / "store"),
            registry,
        )


def test_structural_reweight_uses_explicit_kind_and_mass_receipt(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def reweight(context: KernelContext) -> KernelResult:
        before = context.weights["household"].values
        after = before * 2
        assert set(context.tables) == {"household"}
        assert set(context.tables["household"]) == {"household_id", "size"}
        return KernelResult(
            weights=Weights(after, WeightKind.IMPORTANCE),
            receipt={
                "mass": {
                    "policy": "free",
                    "before": 4.0,
                    "after": 8.0,
                    "stratum_before": {"a": 2.0, "b": 2.0},
                    "stratum_after": {"a": 4.0, "b": 4.0},
                }
            },
        )

    pool = Node(
        "pool",
        "reweight@1",
        structural=StructuralDelta.REWEIGHT,
        base="survey",
        inputs=(Slice("household", ("size",)),),
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    kernel = _Kernel(
        pool.kernel,
        Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT),
        reweight,
    )
    manifest = _run(
        Graph("toy", (SOURCE,), (CREATE, pool)),
        source,
        ContentStore(tmp_path / "store"),
        _registry(extra=kernel),
    )
    assert manifest.population("pool").weights_for("household").kind is (
        WeightKind.IMPORTANCE
    )
    assert manifest.nodes["pool"].receipt["mass"]["after"] == 8.0  # type: ignore[index]
    assert manifest.nodes["pool"].frame_key is not None
    assert manifest.nodes["pool"].weight_key is not None
    assert manifest.mass_ledger("pool")[-1].after_total == 8.0


@pytest.mark.parametrize("column", ["person_id", "person_household_id"])
def test_ordinary_nodes_cannot_own_implicit_structural_columns(
    tmp_path: Path, column: str
) -> None:
    source = _source_path(tmp_path / column)
    node = Node(
        "rewrite_structure",
        "rewrite.structure@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", column, "int64"),),
        population="survey",
    )
    kernel = _Kernel(
        node.kernel,
        Capabilities(Determinism.DETERMINISTIC),
        lambda context: KernelResult(
            columns={
                ("person", column): pd.Series(
                    context.tables["person"]["person_id"].to_numpy(copy=True),
                    index=context.tables["person"]["person_id"],
                    dtype="int64",
                )
            }
        ),
    )

    with pytest.raises(NodeRejected, match="structural column"):
        _run(
            Graph("toy", (SOURCE,), (CREATE, node)),
            source,
            ContentStore(tmp_path / f"{column}-store"),
            _registry(extra=kernel),
        )
    assert kernel.calls == 0


def test_weight_artifact_cannot_collide_with_a_data_column(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def create_with_weight_named_column(context: KernelContext) -> KernelResult:
        frame = _source_frame(context.sources["survey"])
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["household"]["__weights__"] = np.array([10.0, 20.0])
        return KernelResult(
            frame=Frame(
                tables,
                frame.schema,
                {
                    entity: frame.weights_for(entity)
                    for entity in frame.weighted_entities
                },
                frame.strata,
            )
        )

    def reweight(context: KernelContext) -> KernelResult:
        before = context.weights["household"].values
        after = before * 2
        return KernelResult(
            weights=Weights(after, WeightKind.IMPORTANCE),
            receipt={
                "mass": {
                    "policy": "free",
                    "before": 4.0,
                    "after": 8.0,
                    "stratum_before": {"a": 2.0, "b": 2.0},
                    "stratum_after": {"a": 4.0, "b": 4.0},
                }
            },
        )

    create = Node(
        "survey",
        "source.weights@1",
        sources=("survey",),
        structural=StructuralDelta.CREATE,
        outputs=(*CREATE.outputs, Owned("household", "__weights__", "float64")),
    )
    pool = Node(
        "pool",
        "reweight.weights@1",
        structural=StructuralDelta.REWEIGHT,
        base="survey",
        inputs=(Slice("household", ("size",)),),
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    graph = Graph("toy", (SOURCE,), (create, pool))

    def registry() -> KernelRegistry:
        result = _registry()
        result.register(
            _Kernel(
                create.kernel,
                Capabilities(
                    Determinism.DETERMINISTIC,
                    structural=StructuralDelta.CREATE,
                ),
                create_with_weight_named_column,
            )
        )
        result.register(
            _Kernel(
                pool.kernel,
                Capabilities(
                    Determinism.DETERMINISTIC,
                    structural=StructuralDelta.REWEIGHT,
                ),
                reweight,
            )
        )
        return result

    store = ContentStore(tmp_path / "store")
    cold = _run(graph, source, store, registry())
    warm = _run(graph, source, store, registry())

    for manifest in (cold, warm):
        np.testing.assert_array_equal(
            manifest.population("pool").weights_for("household").values,
            np.array([2.0, 4.0]),
        )
        receipt = manifest.nodes["pool"]
        assert receipt.weight_key != receipt.artifacts[("household", "__weights__")]
    assert all(item.hit for item in warm.nodes.values())


@pytest.mark.parametrize(
    ("gate_outcome", "tier"),
    [
        ("pass", "certified"),
        ("not_applicable", "certified"),
        ("fail", "evidence"),
        ("evidence_absent", "evidence"),
        ("unreached", "evidence"),
    ],
)
def test_release_tier_uses_transitive_gate_ancestry(
    tmp_path: Path, gate_outcome: str, tier: str
) -> None:
    source = _source_path(tmp_path / "source")
    manifest = _run(
        _release_graph(gate_outcome=gate_outcome, tier_answer=tier),
        source,
        ContentStore(tmp_path / "store"),
        _release_registry(),
    )

    receipt = manifest.nodes["release"].receipt
    assert receipt["tier"] == tier
    assert receipt["outcome"] == ("pass" if tier == "certified" else "fail")
    assert receipt["gate_ancestry"] == ("gate",)


def test_release_rejects_a_kernel_tier_that_disagrees_with_ancestry(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    with pytest.raises(NodeRejected, match="gate ancestry derives 'certified'"):
        _run(
            _release_graph(gate_outcome="pass", tier_answer="evidence"),
            source,
            ContentStore(tmp_path / "store"),
            _release_registry(),
        )


def test_release_decision_changes_only_the_manifest_outcome(tmp_path: Path) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    graph = _release_graph(
        gate_outcome="pass", tier_answer="certified", requires=("publish",)
    )
    missing = _run(graph, source, store, _release_registry())
    signed = _run(
        graph,
        source,
        store,
        _release_registry(),
        decisions=({"name": "publish", "owner": "reviewer", "signature": "signed"},),
    )

    assert missing.nodes["release"].receipt["outcome"] == "unreached"
    assert signed.nodes["release"].receipt["outcome"] == "pass"
    assert signed.nodes["release"].hit
    assert signed.nodes["release"].key == missing.nodes["release"].key
    assert signed.nodes["release"].artifacts == missing.nodes["release"].artifacts
    assert signed.nodes["release"].receipt["tier"] == "certified"
    assert signed.nodes["release"].receipt["requires_decisions"] == ("publish",)


def test_certified_loader_revalidates_authenticated_required_decisions(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    graph = _release_graph(
        gate_outcome="pass",
        tier_answer="certified",
        requires=("publish",),
    )
    signed = _run(
        graph,
        source,
        store,
        _release_registry(),
        decisions=({"name": "publish", "owner": "reviewer", "signature": "signed"},),
    )
    path = tmp_path / "certified.json"
    signed.save(path)
    document = json.loads(path.read_text())
    authenticated_body = document["content_addressed"]
    authenticated_key = document["key"]
    document["decisions"] = []
    assert document["content_addressed"] == authenticated_body
    assert document["key"] == authenticated_key
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(NodeRejected, match=r"unreached.*publish"):
        RunManifest.load_certified(path, store)


@pytest.mark.parametrize(
    "replacement",
    [
        {"name": "publish", "owner": "", "signature": "signed"},
        {"name": "publish", "owner": "reviewer", "signature": ""},
        {
            "owner": "",
            "kind": "publish",
            "text": "approved",
            "signed_at": "2026-09-03",
        },
        {
            "owner": "reviewer",
            "kind": "publish",
            "text": "",
            "signed_at": "2026-09-03",
        },
        {
            "owner": "reviewer",
            "kind": "publish",
            "text": "approved",
            "signed_at": "",
        },
    ],
    ids=[
        "legacy-owner",
        "legacy-signature",
        "current-owner",
        "current-text",
        "current-signed-at",
    ],
)
def test_certified_loader_revalidates_signed_decision_fields(
    tmp_path: Path,
    replacement: dict[str, str],
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    graph = _release_graph(
        gate_outcome="pass",
        tier_answer="certified",
        requires=("publish",),
    )
    signed = _run(
        graph,
        source,
        store,
        _release_registry(),
        decisions=({"name": "publish", "owner": "reviewer", "signature": "signed"},),
    )
    path = tmp_path / "invalid-signed-record.json"
    signed.save(path)
    document = json.loads(path.read_text())
    authenticated_body = document["content_addressed"]
    authenticated_key = document["key"]
    document["decisions"] = [replacement]
    assert document["content_addressed"] == authenticated_body
    assert document["key"] == authenticated_key
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(NodeRejected, match="non-empty signed decision"):
        RunManifest.load_certified(path, store)


def test_gate_outcome_is_closed_and_gate_exceptions_become_failures(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    invalid_graph = _release_graph(gate_outcome="maybe", tier_answer="evidence")
    with pytest.raises(NodeRejected, match="expected one of"):
        _run(
            invalid_graph,
            source,
            ContentStore(tmp_path / "invalid"),
            _release_registry(),
        )

    gate = Node(
        "gate",
        "gate.raise@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("household", "gate_verdict", "string"),),
        population="survey",
    )

    def explode(context: KernelContext) -> KernelResult:
        raise LookupError("gate evidence unavailable")

    registry = _registry(
        extra=_Kernel(
            gate.kernel,
            Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE),
            explode,
        )
    )
    store = ContentStore(tmp_path / "exploding")
    failed = _run(Graph("toy", (SOURCE,), (CREATE, gate)), source, store, registry)
    receipt = failed.nodes["gate"].receipt
    assert receipt["outcome"] == "fail"
    assert receipt["evidence"] == {
        "exception_type": "LookupError",
        "message": "gate evidence unavailable",
    }
    verdict = store.load_column(
        failed.nodes["gate"].artifacts[("household", "gate_verdict")]
    )
    assert verdict.tolist() == ["fail", "fail"]

    compute = replace(gate, id="compute", kernel="compute.raise@1")
    compute_registry = _registry(
        extra=_Kernel(
            compute.kernel,
            Capabilities(Determinism.DETERMINISTIC),
            explode,
        )
    )
    with pytest.raises(NodeRejected, match="gate evidence unavailable"):
        _run(
            Graph("toy", (SOURCE,), (CREATE, compute)),
            source,
            ContentStore(tmp_path / "compute"),
            compute_registry,
        )


def test_cache_misses_receipt_without_tolerance_writer_provenance(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    graph = _graph(leaf=False)
    registry = _registry()
    cold = _run(graph, source, store, registry)

    node = graph.node("a")
    kernel = registry.get(node.kernel)
    assert isinstance(kernel, _Kernel)
    assert kernel.calls == 1
    key = cold.nodes[node.id].key
    record_key = graph_executor._cache_record_key(key)
    record = store.load_json(record_key)
    raw_receipt = record["receipt"]
    assert isinstance(raw_receipt, dict)
    receipt_capabilities = raw_receipt["capabilities"]
    assert isinstance(receipt_capabilities, dict)
    assert receipt_capabilities.pop("tolerance_writers") == {"person.age": ["survey"]}
    store.put_json(record_key, record, node_key=key, verify_existing=False)

    with pytest.raises(StoreMiss, match=r"cache misses.*'a'"):
        _run(graph, source, store, registry, resume="require")
    assert kernel.calls == 1

    warm = _run(graph, source, store, registry)
    assert not warm.nodes[node.id].hit
    assert kernel.calls == 2
    repaired = store.load_json(record_key)
    repaired_receipt = repaired["receipt"]
    assert isinstance(repaired_receipt, dict)
    repaired_capabilities = repaired_receipt["capabilities"]
    assert isinstance(repaired_capabilities, dict)
    assert repaired_capabilities["tolerance_writers"] == {"person.age": ["survey"]}


def test_cache_load_misses_when_stored_capabilities_disagree(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    graph = _graph()
    registry = _registry()
    manifest = _run(graph, source, store, registry)

    node = graph.node("a")
    key = manifest.nodes["a"].key
    record_key = graph_executor._cache_record_key(key)
    record = store.load_json(record_key)
    stored_capabilities = record["capabilities"]
    assert isinstance(stored_capabilities, dict)
    record["capabilities"] = {
        **stored_capabilities,
        "seed_source": "param",
    }
    store.put_json(record_key, record, node_key=key, verify_existing=False)

    kernel = registry.get(node.kernel)
    with pytest.raises(StoreMiss, match="capabilities"):
        graph_executor._load_record(
            store,
            node,
            key=key,
            kernel_impl_hash=kernel.implementation_hash(),
            capabilities=kernel.capabilities,
        )


def test_cache_load_names_legacy_capabilities_without_tolerance(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")
    store = ContentStore(tmp_path / "store")
    graph = _graph()
    registry = _registry()
    manifest = _run(graph, source, store, registry)

    node = graph.node("a")
    key = manifest.nodes["a"].key
    record_key = graph_executor._cache_record_key(key)
    record = store.load_json(record_key)
    stored_capabilities = record["capabilities"]
    assert isinstance(stored_capabilities, dict)
    del stored_capabilities["tolerance"]
    store.put_json(record_key, record, node_key=key, verify_existing=False)

    kernel = registry.get(node.kernel)
    with pytest.raises(StoreMiss, match=r"legacy_capabilities.*tolerance"):
        graph_executor._load_record(
            store,
            node,
            key=key,
            kernel_impl_hash=kernel.implementation_hash(),
            capabilities=kernel.capabilities,
        )


def test_fit_qrf_seed_source_change_misses_a_shared_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from microcosm.fit.kernels import QRF_EXECUTOR_KERNEL, QRF_PARAM_KERNEL
    from microcosm.graph import graph_from_json
    from tools.graph_parity_fixtures import FIXTURES, ParityCsvSource

    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    case = FIXTURES / "fit.qrf"
    compiled = compile_graph(graph_from_json((case / "graph.json").read_text()))
    store = ContentStore(tmp_path / "store")

    param_registry = KernelRegistry()
    param_registry.register(ParityCsvSource())
    param_registry.register(QRF_PARAM_KERNEL)
    cold = run_graph(
        compiled,
        sources={"fixture": case / "inputs.csv"},
        store=store,
        kernels=param_registry,
    )
    assert not cold.nodes["fit_qrf"].hit

    assert QRF_PARAM_KERNEL.ref == QRF_EXECUTOR_KERNEL.ref
    assert (
        QRF_PARAM_KERNEL.implementation_hash()
        == QRF_EXECUTOR_KERNEL.implementation_hash()
    )
    assert (
        QRF_PARAM_KERNEL.capabilities.seed_source
        is not QRF_EXECUTOR_KERNEL.capabilities.seed_source
    )

    executor_registry = KernelRegistry()
    executor_registry.register(ParityCsvSource())
    executor_registry.register(QRF_EXECUTOR_KERNEL)
    with pytest.raises(NodeRejected, match="EXECUTOR-seeded.*must omit"):
        run_graph(
            compiled,
            sources={"fixture": case / "inputs.csv"},
            store=store,
            kernels=executor_registry,
        )


def test_fit_qrf_model_artifact_is_canonical_across_runtime_worker_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pickle

    from microcosm.fit.kernels import QRF_PARAM_KERNEL
    from microcosm.graph import graph_from_json
    from tools.graph_parity_fixtures import FIXTURES, ParityCsvSource

    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    case = FIXTURES / "fit.qrf"
    compiled = compile_graph(graph_from_json((case / "graph.json").read_text()))
    registry = KernelRegistry()
    registry.register(ParityCsvSource())
    registry.register(QRF_PARAM_KERNEL)

    manifests = []
    stores = []
    for n_jobs in ("1", "2"):
        monkeypatch.setenv("POPULACE_FIT_N_JOBS", n_jobs)
        store = ContentStore(tmp_path / f"store-{n_jobs}")
        stores.append(store)
        manifests.append(
            run_graph(
                compiled,
                sources={"fixture": case / "inputs.csv"},
                store=store,
                kernels=registry,
                resume="forbid",
            )
        )

    first_node = manifests[0].nodes["fit_qrf"]
    second_node = manifests[1].nodes["fit_qrf"]
    assert first_node.key == second_node.key
    first_artifact = first_node.opaque_artifacts["model"]
    second_artifact = second_node.opaque_artifacts["model"]
    assert first_artifact == second_artifact
    first_bytes = stores[0].load_bytes(first_artifact)
    assert first_bytes == stores[1].load_bytes(second_artifact)

    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "2")
    loaded = pickle.loads(first_bytes)  # noqa: S301 - trusted graph-store artifact
    forests = [
        forest
        for target in loaded._target_models.values()
        for forest in (target.positive, target.negative)
        if forest is not None
    ]
    assert forests
    assert {forest.model.n_jobs for forest in forests} == {2}


def test_fit_qrf_tolerance_source_hash_pin_is_current() -> None:
    import json

    from microcosm.fit.kernels import QRF_PARAM_KERNEL, QRFKernel
    from tools.graph_parity_fixtures import FIXTURES

    pins = json.loads((FIXTURES / "fit.qrf" / "pins.json").read_text())
    # fit.qrf@1 is platform-bitwise (amendment 16): no tolerance, and the
    # implementation identity must not depend on the capability declaration.
    assert QRF_PARAM_KERNEL.capabilities.numeric is Numeric.PLATFORM_BITWISE
    assert QRF_PARAM_KERNEL.capabilities.tolerance is None
    assert pins["implementation_hash"] == QRF_PARAM_KERNEL.implementation_hash()
    changed_tolerance = QRFKernel(QRF_PARAM_KERNEL.capabilities.seed_source)
    changed_tolerance.capabilities = replace(
        changed_tolerance.capabilities,
        numeric=Numeric.TOLERANCE_BOUND,
        tolerance=Tolerance(ulps=2),
    )
    assert changed_tolerance.implementation_hash() == pins["implementation_hash"]


def test_entrant_materialization_rejects_a_masked_claimant(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def admit_household(context: KernelContext) -> KernelResult:
        return KernelResult(
            expand={
                "person": pd.Series(
                    [1],
                    index=pd.Index([4], name="person_id"),
                    dtype="int64",
                ),
                "household": pd.Series(
                    [pd.NA],
                    index=pd.Index([30], name="household_id"),
                    dtype="Int64",
                ),
            },
            columns={
                ("person", "person_household_id"): pd.Series(
                    [10, 10, 20, 30],
                    index=pd.Index([1, 2, 3, 4], name="person_id"),
                    dtype="int64",
                ),
                ("household", "size"): pd.Series(
                    [2, 1, 1],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="int64",
                ),
                ("household", "claim_mask"): pd.Series(
                    [True, True, False],
                    index=pd.Index([10, 20, 30], name="household_id"),
                    dtype="boolean",
                ),
            },
            weights=Weights(
                np.array([1.0, 2.0, 1.0], dtype=np.float64),
                WeightKind.DESIGN,
            ),
        )

    expand = Node(
        "admit_household",
        "masked.expand@1",
        structural=StructuralDelta.EXPAND,
        base="survey",
        params={
            "expand_cells": (
                ("person", "person_household_id", "int64"),
                ("household", "size", "int64"),
                ("household", "claim_mask", "boolean"),
            ),
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
        },
        mass="free",
        entrants=True,
    )
    claim_mask = Node(
        "claim_mask",
        "claim.mask@1",
        outputs=(Owned("household", "claim_mask", "boolean"),),
        params={"materialized_expand_outputs": ("household.claim_mask",)},
        population=expand.id,
    )
    claim_size = Node(
        "claim_size",
        "claim.masked-size@1",
        inputs=(Slice("household", ("claim_mask",)),),
        outputs=(Owned("household", "size", "int64", rows="claim_mask"),),
        params={"materialized_expand_outputs": ("household.size",)},
        population=expand.id,
    )

    def must_not_run(context: KernelContext) -> KernelResult:
        raise AssertionError(f"claimant {context.node.id} should not run")

    registry = _registry()
    registry.register(
        _Kernel(
            expand.kernel,
            Capabilities(
                Determinism.DETERMINISTIC,
                structural=StructuralDelta.EXPAND,
            ),
            admit_household,
        )
    )
    mask_kernel = _Kernel(
        claim_mask.kernel,
        Capabilities(Determinism.DETERMINISTIC),
        must_not_run,
    )
    size_kernel = _Kernel(
        claim_size.kernel,
        Capabilities(Determinism.DETERMINISTIC),
        must_not_run,
    )
    registry.register(mask_kernel)
    registry.register(size_kernel)

    with pytest.raises(NodeRejected, match="household.size.*rows='all'"):
        _run(
            Graph("toy", (SOURCE,), (CREATE, expand, claim_mask, claim_size)),
            source,
            ContentStore(tmp_path / "store"),
            registry,
        )
    assert mask_kernel.calls == size_kernel.calls == 0


# --- Amendment 19: typed opaque artifacts -----------------------------------

FOREST = ArtifactType("qrf.forest", 1)


def _artifact_graph(
    *,
    declare_output: bool = True,
    consumer_type: ArtifactType = FOREST,
) -> Graph:
    fit = Node(
        "fit",
        "fit@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "fitted", "float64"),),
        params={"source": "age", "target": "fitted", "scale": 1.0},
        population="survey",
        artifact_outputs=(ArtifactOutput("forest", FOREST),) if declare_output else (),
    )
    draw = Node(
        "draw",
        "consume@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "drawn", "float64"),),
        population="survey",
        artifact_inputs=(ArtifactInput("donor", "fit", "forest", consumer_type),),
    )
    return Graph("toy", (SOURCE,), (CREATE, fit, draw))


def _artifact_registry(
    *,
    seen: list[Mapping[str, object]] | None = None,
    payload: bytes = b"forest-bytes",
    emit: bool = True,
    producer: Capabilities | None = None,
    consumer: Capabilities | None = None,
) -> KernelRegistry:
    def fit(context: KernelContext) -> KernelResult:
        table = context.tables["person"]
        return KernelResult(
            columns={
                ("person", "fitted"): pd.Series(
                    table["age"].to_numpy(dtype=np.float64),
                    index=pd.Index(table["person_id"], name="person_id"),
                    dtype="float64",
                )
            },
            artifacts=({"forest": payload} if emit else {}) | {"notes": b"diagnostic"},
            receipt={"trees": 3},
        )

    def consume(context: KernelContext) -> KernelResult:
        if seen is not None:
            seen.append(dict(context.artifacts))
        donor = context.artifacts["donor"]
        table = context.tables["person"]
        return KernelResult(
            columns={
                ("person", "drawn"): pd.Series(
                    np.full(len(table), float(len(donor.payload))),
                    index=pd.Index(table["person_id"], name="person_id"),
                    dtype="float64",
                )
            },
            receipt={"payload_bytes": len(donor.payload)},
        )

    registry = _registry()
    registry.register(
        _Kernel("fit@1", producer or Capabilities(Determinism.DETERMINISTIC), fit)
    )
    registry.register(
        _Kernel(
            "consume@1", consumer or Capabilities(Determinism.DETERMINISTIC), consume
        )
    )
    return registry


def test_a_declared_artifact_reaches_its_consumer_verified(tmp_path: Path) -> None:
    """Amendment 19: the executor hands over the producer's exact bytes."""
    seen: list[Mapping[str, object]] = []
    store = ContentStore(tmp_path / "store")
    registry = _artifact_registry(seen=seen)
    manifest = _run(_artifact_graph(), _source_path(tmp_path / "src"), store, registry)
    assert len(seen) == 1
    donor = seen[0]["donor"]
    assert set(seen[0]) == {"donor"}
    assert donor.payload == b"forest-bytes"
    assert donor.type == FOREST
    producer_key = manifest.nodes["fit"].key
    assert donor.producer_key == producer_key
    assert donor.key == opaque_artifact_key(producer_key, "forest")
    assert donor.numerics == NumericScope(numeric=Numeric.BITWISE)
    assert manifest.nodes["fit"].opaque_artifacts["forest"] == donor.key
    # Every person row carries the payload length the consumer measured.
    drawn = manifest.populations["survey"].person["drawn"]
    assert set(drawn.to_numpy()) == {float(len(b"forest-bytes"))}


def test_a_kernel_that_omits_a_declared_artifact_is_rejected(tmp_path: Path) -> None:
    """Amendment 19: a declared output is required of the kernel."""
    store = ContentStore(tmp_path / "store")
    registry = _artifact_registry(emit=False)
    with pytest.raises(NodeRejected, match="missing declared artifact 'forest'"):
        _run(_artifact_graph(), _source_path(tmp_path / "src"), store, registry)


def test_undeclared_opaque_bytes_stay_legal_and_unaddressable(tmp_path: Path) -> None:
    """Amendment 19: only declared outputs become artifact edges."""
    store = ContentStore(tmp_path / "store")
    manifest = _run(
        _artifact_graph(), _source_path(tmp_path / "src"), store, _artifact_registry()
    )
    assert set(manifest.nodes["fit"].opaque_artifacts) == {"forest", "notes"}
    assert set(manifest.nodes["fit"].typed_artifacts["outputs"]) == {"forest"}
    graph = _artifact_graph()
    bad = Graph(
        graph.country,
        graph.sources,
        tuple(
            node
            if node.id != "draw"
            else replace(
                node,
                artifact_inputs=(ArtifactInput("donor", "fit", "notes", FOREST),),
            )
            for node in graph.nodes
        ),
    )
    with pytest.raises(GraphError, match="no declared artifact 'notes'"):
        compile_graph(bad)


def test_an_artifact_edge_is_memoized_like_every_other_input(tmp_path: Path) -> None:
    """Amendment 19: a second run hits the store and executes no kernel."""
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    first = _artifact_registry()
    _run(_artifact_graph(), source, store, first)
    second = _artifact_registry()
    manifest = _run(_artifact_graph(), source, store, second)
    assert all(receipt.hit for receipt in manifest.nodes.values())
    assert _calls(second)["fit@1"] == 0
    assert _calls(second)["consume@1"] == 0


def test_a_cached_typed_contract_that_disagrees_with_the_graph_is_refused(
    tmp_path: Path,
) -> None:
    """Amendment 19: the cached record pins the typed contract it was run under."""
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    manifest = _run(_artifact_graph(), source, store, _artifact_registry())
    # Rewriting a cached consumer record's contract is corruption, not a miss.
    key = manifest.nodes["draw"].key
    raw = store.load_json(graph_executor._cache_record_key(key))
    raw["typed_artifacts"]["inputs"]["donor"]["type"]["schema_version"] = 99
    store.put_json(
        graph_executor._cache_record_key(key), raw, node_key=key, verify_existing=False
    )
    with pytest.raises(StoreCorrupt, match="typed artifact contracts disagree"):
        _run(_artifact_graph(), source, store, _artifact_registry())


def test_the_manifest_records_typed_provenance_and_round_trips(tmp_path: Path) -> None:
    """Amendment 19: typed edges are portable provenance at manifest schema 3."""
    store = ContentStore(tmp_path / "store")
    manifest = _run(
        _artifact_graph(), _source_path(tmp_path / "src"), store, _artifact_registry()
    )
    binding = manifest.nodes["draw"].typed_artifacts["inputs"]["donor"]
    assert binding["producer"] == "fit" and binding["artifact"] == "forest"
    assert binding["producer_key"] == manifest.nodes["fit"].key
    text = manifest.to_json()
    assert '"schema_version":3' in text
    restored = RunManifest.from_json(text)
    assert restored.key == manifest.key
    assert (
        restored.nodes["draw"].typed_artifacts == manifest.nodes["draw"].typed_artifacts
    )


def test_a_typed_edge_refuses_to_launder_its_producer_numeric_scope(
    tmp_path: Path,
) -> None:
    """Amendment 19 x 16/17: bytes carry their producer's numeric contract."""
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    registry = _artifact_registry(
        producer=Capabilities(Determinism.SEEDED, numeric=Numeric.PLATFORM_BITWISE)
    )
    with pytest.raises(NodeRejectedError, match="platform_bitwise artifact requires"):
        _run(_artifact_graph(), source, store, registry)
    bounded = _artifact_registry(
        producer=Capabilities(
            Determinism.SEEDED,
            numeric=Numeric.TOLERANCE_BOUND,
            tolerance=Tolerance(rtol=1e-6),
        )
    )
    with pytest.raises(NodeRejectedError, match="tolerance_bound artifact requires"):
        _run(_artifact_graph(), source, store, bounded)
    # A consumer that declares the producer's class reads the bytes.
    matched = _artifact_registry(
        producer=Capabilities(Determinism.SEEDED, numeric=Numeric.PLATFORM_BITWISE),
        consumer=Capabilities(Determinism.SEEDED, numeric=Numeric.PLATFORM_BITWISE),
    )
    manifest = _run(_artifact_graph(), source, store, matched)
    scope = manifest.nodes["draw"].typed_artifacts["inputs"]["donor"]["numerics"]
    assert scope["numeric"] == "platform_bitwise"
    assert scope["platform"] == platform_fingerprint()


def test_an_artifact_payload_enters_the_input_context_digest(tmp_path: Path) -> None:
    """Amendment 19: artifact bytes are input state, so mutation is detectable."""
    node = Node("draw", "consume@1")
    scope = NumericScope()
    base = dict(
        node=node,
        tables={},
        weights={},
        strata=pd.Series(dtype="int64"),
        params={},
        rng=np.random.default_rng(0),
    )
    first = KernelContext(
        **base,
        artifacts={"donor": ArtifactValue(b"one", FOREST, "a" * 64, "b" * 64, scope)},
    )
    second = KernelContext(
        **base,
        artifacts={"donor": ArtifactValue(b"two", FOREST, "a" * 64, "b" * 64, scope)},
    )
    bare = KernelContext(**base)
    digest = graph_executor._context_digest
    assert digest(first) != digest(second)
    assert digest(first) != digest(bare)
    assert digest(first) == digest(
        KernelContext(
            **base,
            artifacts={
                "donor": ArtifactValue(b"one", FOREST, "a" * 64, "b" * 64, scope)
            },
        )
    )


EVIDENCE = ArtifactType("gate.evidence", 1)


def _gate_artifact_graph(*, release_behind: bool, answer: str) -> Graph:
    """A gate declaring typed evidence, its byte consumer, and a cell consumer.

    ``release`` reads the cell consumer's column when ``release_behind`` is
    true (so it sits behind the byte edge) and the gate's own verdict column
    otherwise (so it sits beside it).
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
        outputs=(Owned("person", "used", "float64"),),
        population="survey",
        artifact_inputs=(ArtifactInput("evidence", "gate", "evidence", EVIDENCE),),
    )
    after = Node(
        "after",
        "after@1",
        inputs=(Slice("person", ("used",)),),
        outputs=(Owned("person", "after", "float64"),),
        population="survey",
    )
    release = Node(
        "release",
        "release@1",
        inputs=(
            (Slice("person", ("after",)),)
            if release_behind
            else (Slice("household", ("gate_verdict",)),)
        ),
        outputs=(Owned("household", "tier", "string"),),
        params={"answer": answer, "requires_decisions": ()},
        population="survey",
    )
    return Graph("toy", (SOURCE,), (CREATE, gate, use, after, release))


def _gate_artifact_registry(*, raising: bool) -> KernelRegistry:
    def failing_gate(context: KernelContext) -> KernelResult:
        raise RuntimeError("evidence unavailable")

    def passing_gate(context: KernelContext) -> KernelResult:
        ids = context.tables["household"]["household_id"]
        return KernelResult(
            columns={
                ("household", "gate_verdict"): pd.Series(
                    "pass", index=ids, dtype="string"
                )
            },
            artifacts={"evidence": b"evidence-bytes"},
            receipt={"outcome": "pass", "evidence": {"fixture": True}},
        )

    def use(context: KernelContext) -> KernelResult:
        table = context.tables["person"]
        payload = context.artifacts["evidence"].payload
        return KernelResult(
            columns={
                ("person", "used"): pd.Series(
                    np.full(len(table), float(len(payload))),
                    index=pd.Index(table["person_id"], name="person_id"),
                    dtype="float64",
                )
            }
        )

    def after(context: KernelContext) -> KernelResult:
        table = context.tables["person"]
        return KernelResult(
            columns={
                ("person", "after"): pd.Series(
                    table["used"].to_numpy(dtype=np.float64) * 2.0,
                    index=pd.Index(table["person_id"], name="person_id"),
                    dtype="float64",
                )
            }
        )

    def release(context: KernelContext) -> KernelResult:
        ids = context.tables["household"]["household_id"]
        answer = str(context.params["answer"])
        return KernelResult(
            columns={
                ("household", "tier"): pd.Series(answer, index=ids, dtype="string")
            },
            receipt={"outcome": "pass"},
        )

    deterministic = Capabilities(Determinism.DETERMINISTIC)
    registry = _registry()
    registry.register(
        _Kernel(
            "gate@1",
            Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE),
            failing_gate if raising else passing_gate,
        )
    )
    registry.register(_Kernel("use@1", deterministic, use))
    registry.register(_Kernel("after@1", deterministic, after))
    registry.register(
        _Kernel(
            "release@1",
            Capabilities(Determinism.DETERMINISTIC, role=KernelRole.RELEASE),
            release,
        )
    )
    return registry


def test_a_gate_that_declares_evidence_and_raises_leaves_its_consumers_unreached(
    tmp_path: Path,
) -> None:
    """A gate exception is still a verdict (amendment 7); its outputs are absent.

    The gate records ``fail`` with a ``gate_exception`` execution state naming
    the outputs it could not produce; the byte consumer and everything causally
    behind it are ``unreached`` with their blockers named by node key, no
    kernel behind the edge runs, nothing is invented for them, the release
    behind the edge stays evidence-tier, and the manifest serializes at schema
    4, round-trips, and replays as hits under every resume policy.
    """
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    graph = _gate_artifact_graph(release_behind=True, answer="evidence")
    registry = _gate_artifact_registry(raising=True)
    manifest = _run(graph, source, store, registry)

    gate = manifest.nodes["gate"]
    assert gate.receipt["outcome"] == "fail"
    assert gate.receipt["execution"] == {
        "schema": EXECUTION_SCHEMA,
        "state": "gate_exception",
        "unavailable_artifacts": ("evidence",),
    }
    assert gate.receipt["evidence"]["exception_type"] == "RuntimeError"
    assert not gate.opaque_artifacts
    assert set(gate.typed_artifacts["outputs"]) == {"evidence"}
    verdict = manifest.populations["survey"].household["gate_verdict"]
    assert set(verdict.to_numpy()) == {"fail"}
    assert "used" not in manifest.populations["survey"].person.columns

    use = manifest.nodes["use"]
    assert use.receipt["outcome"] == "unreached"
    assert use.receipt["execution"] == {
        "schema": EXECUTION_SCHEMA,
        "state": "unreached",
        "blocked_by": {"gate": gate.key},
    }
    assert not use.artifacts and use.frame_key is None and not use.opaque_artifacts
    after = manifest.nodes["after"]
    assert after.receipt["execution"]["blocked_by"] == {"use": use.key}
    release = manifest.nodes["release"]
    assert release.receipt["execution"]["blocked_by"] == {"after": after.key}
    assert release.receipt["tier"] == "evidence"
    assert release.receipt["gate_ancestry"] == ("gate",)
    assert manifest.tier == "evidence"
    calls = _calls(registry)
    assert calls["gate@1"] == 1
    assert calls["use@1"] == calls["after@1"] == calls["release@1"] == 0

    text = manifest.to_json()
    assert '"schema_version":4' in text
    restored = RunManifest.from_json(text)
    assert restored.key == manifest.key
    assert restored.nodes["use"].receipt == use.receipt
    assert restored.tier == "evidence"

    def assert_explained(outcome: RunManifest, cache: str) -> None:
        rendered = explain_html(compile_graph(graph), outcome)
        for node_id in ("use", "after", "release"):
            role = "release" if node_id == "release" else "compute"
            assert (
                f'aria-label="{node_id}; {node_id}@1; {role}; none; '
                f'{cache} · unreached"'
            ) in rendered
        assert rendered.count('execution-unreached" data-node-detail=') == 3
        assert f'status-{cache} gate-fail execution-gate_exception"' in rendered
        assert f"{cache} · gate fail · exception" in rendered

    assert_explained(manifest, "miss")

    for resume in ("auto", "require"):
        again = _gate_artifact_registry(raising=True)
        replay = _run(graph, source, store, again, resume=resume)
        assert all(receipt.hit for receipt in replay.nodes.values())
        assert replay.key == manifest.key
        assert sum(_calls(again).values()) == 0
        assert_explained(replay, "hit")


def test_unreached_gate_cannot_certify_a_downstream_release(tmp_path: Path) -> None:
    base = _gate_artifact_graph(release_behind=True, answer="certified")
    second_gate = Node(
        "second_gate",
        "second_gate@1",
        inputs=(Slice("person", ("used",)),),
        outputs=(Owned("household", "second_verdict", "string"),),
        population="survey",
    )
    release = replace(
        base.node("release"), inputs=(Slice("household", ("second_verdict",)),)
    )
    graph = Graph(
        "toy",
        (SOURCE,),
        (CREATE, base.node("gate"), base.node("use"), second_gate, release),
    )

    def forbidden_gate(context: KernelContext) -> KernelResult:
        raise AssertionError("An unreached gate must not run")

    def registry_with_second_gate() -> KernelRegistry:
        registry = _gate_artifact_registry(raising=True)
        registry.register(
            _Kernel(
                "second_gate@1",
                Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE),
                forbidden_gate,
            )
        )
        return registry

    source = _source_path(tmp_path / "src")
    store = ContentStore(tmp_path / "store")
    cold_key = None
    for resume in ("auto", "require"):
        registry = registry_with_second_gate()
        manifest = _run(graph, source, store, registry, resume=resume)
        gate = manifest.nodes["second_gate"]
        assert gate.receipt["outcome"] == "unreached"
        assert gate.receipt["execution"] == {
            "schema": EXECUTION_SCHEMA,
            "state": "unreached",
            "blocked_by": {"use": manifest.nodes["use"].key},
        }
        assert not gate.artifacts and gate.frame_key is None
        assert not gate.opaque_artifacts
        assert "second_verdict" not in manifest.population("survey").household
        assert manifest.nodes["release"].receipt["execution"]["blocked_by"] == {
            "second_gate": gate.key
        }
        assert manifest.nodes["release"].receipt["tier"] == "evidence"
        assert manifest.tier == "evidence"
        calls = _calls(registry)
        assert calls["second_gate@1"] == calls["release@1"] == 0
        restored = RunManifest.from_json(manifest.to_json())
        assert restored.nodes["second_gate"].receipt == gate.receipt
        assert restored.key == manifest.key and restored.tier == "evidence"
        rendered = explain_html(compile_graph(graph), manifest)
        cache = "hit" if resume == "require" else "miss"
        assert f"{cache} · gate unreached · unreached" in rendered
        if resume == "require":
            assert manifest.key == cold_key
            assert all(node.hit for node in manifest.nodes.values())
            assert sum(calls.values()) == 0
        else:
            cold_key = manifest.key
            assert calls["gate@1"] == 1


def test_unreached_propagates_through_a_structural_node_and_its_version(
    tmp_path: Path,
) -> None:
    """A FILTER whose input is unreached is unreached, and so is its version.

    The version the filter would have opened has no population, so a node
    placed on it is blocked by the filter itself (its version is one of its
    compiled predecessors), while a node beside the edge still runs.
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
        outputs=(Owned("person", "used", "float64"),),
        population="survey",
        artifact_inputs=(ArtifactInput("evidence", "gate", "evidence", EVIDENCE),),
    )
    boundary = Node(
        "boundary",
        "keep@1",
        inputs=(Slice("person", ("used",)),),
        structural=StructuralDelta.FILTER,
        base="survey",
    )
    on_boundary = Node(
        "on_boundary",
        "after@1",
        inputs=(Slice("person", ("used",)),),
        outputs=(Owned("person", "after", "float64"),),
        population="boundary",
    )
    beside = Node(
        "beside",
        "a@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "a", "float64"),),
        params={"source": "age", "target": "a", "scale": 1.0},
        population="survey",
    )
    graph = Graph("toy", (SOURCE,), (CREATE, gate, use, boundary, on_boundary, beside))

    def keep(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        return KernelResult(
            keep=pd.Series(True, index=person["person_id"], dtype="bool")
        )

    def registry_with_filter() -> KernelRegistry:
        registry = _gate_artifact_registry(raising=True)
        registry.register(
            _Kernel(
                "keep@1",
                Capabilities(
                    Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
                ),
                keep,
            )
        )
        return registry

    registry = registry_with_filter()
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    manifest = _run(graph, source, store, registry)
    nodes = manifest.nodes
    assert nodes["boundary"].receipt["execution"]["blocked_by"] == {
        "use": nodes["use"].key
    }
    assert nodes["on_boundary"].receipt["execution"]["blocked_by"] == {
        "boundary": nodes["boundary"].key
    }
    assert nodes["boundary"].frame_key is None
    assert "boundary" not in manifest.populations
    assert "execution" not in nodes["beside"].receipt
    assert set(manifest.populations["survey"].person["a"]) == {10.0, 20.0, 30.0}
    calls = _calls(registry)
    assert calls["keep@1"] == calls["after@1"] == 0 and calls["a@1"] == 1
    restored = RunManifest.from_json(manifest.to_json())
    assert restored.key == manifest.key
    replay = _run(graph, source, store, registry_with_filter(), resume="require")
    assert all(receipt.hit for receipt in replay.nodes.values())


def test_a_gate_that_declares_evidence_and_passes_produces_it(
    tmp_path: Path,
) -> None:
    """The same declaration on a gate that succeeds is an ordinary byte edge."""
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    graph = _gate_artifact_graph(release_behind=True, answer="certified")
    manifest = _run(graph, source, store, _gate_artifact_registry(raising=False))
    gate = manifest.nodes["gate"]
    assert gate.receipt["outcome"] == "pass"
    assert "execution" not in gate.receipt
    assert gate.opaque_artifacts["evidence"] == opaque_artifact_key(
        gate.key, "evidence"
    )
    assert store.load_bytes(gate.opaque_artifacts["evidence"]) == b"evidence-bytes"
    used = manifest.populations["survey"].person["used"]
    assert set(used.to_numpy()) == {float(len(b"evidence-bytes"))}
    assert manifest.nodes["release"].receipt["gate_ancestry"] == ("gate",)
    assert manifest.tier == "certified"
    assert '"schema_version":3' in manifest.to_json()


def test_a_kernel_may_not_author_the_executor_execution_state(
    tmp_path: Path,
) -> None:
    """The execution state is executor evidence; a kernel returning one is rejected."""

    def authoring(context: KernelContext) -> KernelResult:
        table = context.tables["person"]
        return KernelResult(
            columns={
                ("person", "a"): pd.Series(
                    np.zeros(len(table)),
                    index=pd.Index(table["person_id"], name="person_id"),
                    dtype="float64",
                )
            },
            receipt={
                "execution": {
                    "schema": EXECUTION_SCHEMA,
                    "state": "unreached",
                    "blocked_by": {},
                }
            },
        )

    registry = KernelRegistry()
    registry.register(
        _Kernel(
            "source@1",
            Capabilities(Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE),
            _source,
        )
    )
    registry.register(
        _Kernel("a@1", Capabilities(Determinism.DETERMINISTIC), authoring)
    )
    graph = Graph("toy", (SOURCE,), (CREATE, _ordinary("a", "a@1", "age", "a")))
    store = ContentStore(tmp_path / "store")
    with pytest.raises(NodeRejected, match="may not author executor execution"):
        _run(graph, _source_path(tmp_path / "src"), store, registry)
    # A free-form "execution" diagnostic that does not claim the executor's
    # schema is still just a receipt field.
    assert (
        graph_executor.has_execution({"execution": {"literal_full_scans": 1}}) is False
    )


def test_a_cached_unreached_record_is_refused_once_its_inputs_exist(
    tmp_path: Path,
) -> None:
    """An unreached record is a hit only while the same inputs are unavailable."""
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    graph = _gate_artifact_graph(release_behind=True, answer="certified")
    manifest = _run(graph, source, store, _gate_artifact_registry(raising=False))
    gate_key = manifest.nodes["gate"].key
    use_key = manifest.nodes["use"].key
    record_key = graph_executor._cache_record_key(use_key)
    raw = store.load_json(record_key)
    raw.update(
        schema_version=3,
        receipt={
            "outcome": "unreached",
            "execution": {
                "schema": EXECUTION_SCHEMA,
                "state": "unreached",
                "blocked_by": {"gate": gate_key},
            },
            "evidence": {"reason": "Required graph inputs are unavailable."},
            "capabilities": raw["capabilities"],
        },
        columns=[],
        frame_key=None,
        weight=None,
        opaque=[],
    )
    store.put_json(record_key, raw, node_key=use_key, verify_existing=False)
    for resume in ("auto", "require"):
        with pytest.raises(StoreCorrupt, match="has no unavailable input blocker"):
            _run(
                graph,
                source,
                store,
                _gate_artifact_registry(raising=False),
                resume=resume,
            )


def test_a_manifest_authenticates_its_unreached_blockers(tmp_path: Path) -> None:
    """Portable provenance names each blocker by key and the manifest checks it."""
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    graph = _gate_artifact_graph(release_behind=True, answer="evidence")
    manifest = _run(graph, source, store, _gate_artifact_registry(raising=True))
    payload = json.loads(manifest.to_json())

    forged = json.loads(json.dumps(payload))
    forged["nodes"]["use"]["receipt"]["execution"]["blocked_by"] = {"gate": "0" * 64}
    with pytest.raises(ValueError, match="missing or has a different key"):
        RunManifest.from_json(json.dumps(forged))

    forged = json.loads(json.dumps(payload))
    forged["nodes"]["after"]["receipt"]["execution"]["blocked_by"] = {
        "gate": payload["nodes"]["gate"]["key"]
    }
    with pytest.raises(ValueError, match="not one of its declared typed inputs"):
        RunManifest.from_json(json.dumps(forged))

    forged = json.loads(json.dumps(payload))
    forged["schema_version"] = 3
    with pytest.raises(ValueError, match="require manifest schema 4"):
        RunManifest.from_json(json.dumps(forged))


def test_the_private_population_observer_sees_every_admitted_population(
    tmp_path: Path,
) -> None:
    """``_population_observer`` runs per node, cold and on hits, before persistence.

    It is an integration seam for verifiers, not a kernel capability: nothing
    it does enters a key or a receipt, and an exception it raises refuses the
    run.
    """
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    seen: list[tuple[str, int]] = []

    def observe(node_id: str, population: Population) -> None:
        seen.append((node_id, population.frame.n("person")))

    cold = run_graph(
        compile_graph(_graph()),
        sources={"survey": source},
        store=store,
        kernels=_registry(),
        _population_observer=observe,
    )
    assert [node_id for node_id, _ in seen] == list(cold.nodes)
    assert {n for _, n in seen} == {3}
    plain = _run(_graph(), source, store, _registry())
    assert plain.key == cold.key

    seen.clear()
    warm = run_graph(
        compile_graph(_graph()),
        sources={"survey": source},
        store=store,
        kernels=_registry(),
        _population_observer=observe,
    )
    assert all(receipt.hit for receipt in warm.nodes.values())
    assert [node_id for node_id, _ in seen] == list(warm.nodes)

    def refuse(node_id: str, population: Population) -> None:
        raise RuntimeError(f"verifier refused {node_id}")

    with pytest.raises(RuntimeError, match="verifier refused survey"):
        run_graph(
            compile_graph(_graph()),
            sources={"survey": source},
            store=ContentStore(tmp_path / "other"),
            kernels=_registry(),
            _population_observer=refuse,
        )


@pytest.mark.parametrize("warm", (False, True))
def test_mutating_and_retained_observers_cannot_change_execution_or_cache(
    tmp_path: Path, warm: bool
) -> None:
    source = _source_path(tmp_path / "source")
    compiled = compile_graph(_graph(leaf=False))
    plain = run_graph(
        compiled,
        sources={"survey": source},
        store=ContentStore(tmp_path / "plain"),
        kernels=_registry(),
    )
    store = ContentStore(tmp_path / "observed")
    if warm:
        run_graph(
            compiled, sources={"survey": source}, store=store, kernels=_registry()
        )
    retained = []

    def mutate(population):
        population.frame.person.loc[:, "age"] += 100
        weights = population.frame.weights_for("household").values
        weights.setflags(write=True)
        weights[:] = 999
        population.frame._metadata = {"observer": "changed"}
        object.__setattr__(population, "owners", {})

    def observe(node_id, population):
        # Also mutate earlier snapshots during later callbacks, after a simple
        # before/after check around their own callback would have completed.
        retained.append(population)
        for previous in retained:
            mutate(previous)

    observed = run_graph(
        compiled,
        sources={"survey": source},
        store=store,
        kernels=_registry(),
        _population_observer=observe,
    )
    for population in retained:
        mutate(population)  # retained references remain harmless after return
    replay = run_graph(
        compiled, sources={"survey": source}, store=store, kernels=_registry()
    )
    assert all(receipt.hit for receipt in observed.nodes.values()) is warm
    assert all(receipt.hit for receipt in replay.nodes.values())
    for actual in (observed, replay):
        assert actual.key == plain.key
        assert {name: item.key for name, item in actual.nodes.items()} == {
            name: item.key for name, item in plain.nodes.items()
        }
        for entity in plain.populations["survey"].entities:
            pd.testing.assert_frame_equal(
                actual.populations["survey"].table(entity),
                plain.populations["survey"].table(entity),
            )
        np.testing.assert_array_equal(
            actual.populations["survey"].weights_for("household").values,
            plain.populations["survey"].weights_for("household").values,
        )
        assert (
            actual.populations["survey"].metadata
            == plain.populations["survey"].metadata
        )
    assert observed.populations["survey"].person["b"].tolist() == [60.0, 120.0, 180.0]


def test_observer_snapshot_detaches_complete_population_storage(tmp_path: Path) -> None:
    original = _source_frame(_source_path(tmp_path / "source"))
    person = original.person.copy()
    person.index = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 3)], names=["part", "row"]
    )
    person["object_cell"] = pd.Series(
        [{"nested": [1]}, {"nested": [2]}, {"nested": [3]}],
        index=person.index,
        dtype=object,
    )
    person["category"] = pd.Categorical(["x", "y", "x"])
    person["selected"] = pd.array([True, pd.NA, False], dtype="boolean")
    person["selected"].array._data[1] = True  # preserve storage beneath the mask
    person.attrs["nested"] = {"values": [1, 2]}
    schema = EntitySchema(
        group_entities=("household",),
        links=(LinkSpec("relations", "person", "household"),),
    )
    link = pd.DataFrame({"person_id": [1, 2, 3], "household_id": [10, 10, 20]})
    mass_log = (MassChangeRecord("household", 3.0, 3.0, 1.0, "unchanged"),)
    frame = Frame(
        {"person": person, "household": original.table("household"), "relations": link},
        schema,
        {"household": original.weights_for("household")},
        pd.Series(["a", "a", "b"], index=person.index, name="stratum"),
        metadata={"nested": [{"source": "fixture"}], "signed_zero": -0.0},
        mass_log=mass_log,
    )
    ledger = (
        MassRecord(
            "fixture",
            "reweight",
            "conserve",
            3.0,
            3.0,
            (("a", 3.0),),
            (("a", 3.0),),
            entity="household",
        ),
    )
    population = Population.from_frame(frame, "fixture", mass_ledger=ledger)
    snapshot = graph_executor._observer_snapshot(population)
    assert snapshot.frame.schema == population.frame.schema
    assert snapshot.frame.mass_log == population.frame.mass_log
    assert snapshot.mass_ledger == population.mass_ledger
    assert dict(snapshot.owners) == dict(population.owners)
    assert dict(snapshot.weight_kind) == dict(population.weight_kind)
    assert snapshot.frame.metadata == population.frame.metadata
    assert np.signbit(snapshot.frame.metadata["signed_zero"])
    assert snapshot.frame.metadata is not frame.metadata
    assert snapshot.frame.metadata["nested"][0] is not frame.metadata["nested"][0]
    for name in frame.entities:
        pd.testing.assert_frame_equal(snapshot.frame.table(name), frame.table(name))
    pd.testing.assert_frame_equal(
        snapshot.frame.link("relations"), frame.link("relations")
    )
    pd.testing.assert_series_equal(snapshot.frame.strata, frame.strata)
    np.testing.assert_array_equal(
        snapshot.design_weights["household"], population.design_weights["household"]
    )
    assert not np.shares_memory(
        snapshot.design_weights["household"], population.design_weights["household"]
    )
    np.testing.assert_array_equal(
        snapshot.frame.person["selected"].array._data,
        frame.person["selected"].array._data,
    )

    snapshot.frame.person.at[("a", 1), "object_cell"]["nested"].append(99)
    snapshot.frame.person.attrs["nested"]["values"].append(99)
    snapshot.frame.person.index.set_names(["changed", "row"], inplace=True)
    level = snapshot.frame.person.index.levels[0].to_numpy(copy=False)
    level.setflags(write=True)
    level[0] = "changed"
    categories = snapshot.frame.person["category"].cat.categories.to_numpy(copy=False)
    categories.setflags(write=True)
    categories[0] = "changed"
    snapshot.frame.link("relations").iloc[0, 0] = 999
    snapshot.frame.strata.iloc[0] = "changed"
    snapshot.frame.person["selected"].array._data[1] = False
    snapshot.frame.weights_for("household").values.setflags(write=True)
    snapshot.frame.weights_for("household").values[:] = 999
    captured_metadata = snapshot.frame.metadata["nested"][0]
    object.__setattr__(captured_metadata, "_items", (("source", "changed"),))
    assert frame.metadata["nested"][0]["source"] == "fixture"
    snapshot.frame._metadata = {"changed": True}
    object.__setattr__(snapshot.frame.schema.links[0], "name", "changed")
    object.__setattr__(snapshot.frame.schema, "group_entities", ("changed",))
    object.__setattr__(snapshot.frame.mass_log[0], "reason", "changed")
    object.__setattr__(snapshot.mass_ledger[0], "policy", "changed")
    object.__setattr__(snapshot, "owners", {})
    assert frame.person.at[("a", 1), "object_cell"] == {"nested": [1]}
    assert frame.person.attrs["nested"] == {"values": [1, 2]}
    assert frame.person.index.names == ["part", "row"]
    assert frame.person.index.levels[0].tolist() == ["a", "b"]
    assert frame.person["category"].cat.categories.tolist() == ["x", "y"]
    assert frame.link("relations").iloc[0, 0] == 1
    assert frame.strata.iloc[0] == "a"
    assert bool(frame.person["selected"].array._data[1]) is True
    assert frame.weights_for("household").values.tolist() == [1.0, 2.0]
    assert frame.metadata["nested"][0]["source"] == "fixture"
    assert frame.schema.links[0].name == "relations"
    assert frame.schema.group_entities == ("household",)
    assert frame.mass_log[0].reason == "unchanged"
    assert population.mass_ledger[0].policy == "conserve"
    assert population.owners

    # Record annotations do not freeze nested members. Even a caller-supplied
    # container inside a record must not remain an alias across the seam.
    nested_record = replace(ledger[0], before_by_stratum=((["mutable"], 3.0),))
    nested = replace(population, mass_ledger=(nested_record,))
    nested_snapshot = graph_executor._observer_snapshot(nested)
    nested_snapshot.mass_ledger[0].before_by_stratum[0][0].append("changed")
    assert nested.mass_ledger[0].before_by_stratum[0][0] == ["mutable"]


def test_absent_observer_allocates_no_snapshot(tmp_path: Path, monkeypatch) -> None:
    def forbidden(population):
        raise AssertionError("snapshot without observer")

    monkeypatch.setattr(graph_executor, "_observer_snapshot", forbidden)
    source = _source_path(tmp_path / "source")
    _run(_graph(), source, ContentStore(tmp_path / "store"), _registry())


def test_a_gate_reached_only_through_bytes_still_derives_the_tier(
    tmp_path: Path,
) -> None:
    """Amendment 19 cannot route around F2: a byte edge is a real ancestor.

    ``gate`` fails and owns a column only ``fit`` reads; ``fit``'s bytes are
    the only path from that subgraph to ``release``. The release's gate
    ancestry must still name the gate, so its tier is evidence.
    """
    gate = Node(
        "gate",
        "gate@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("household", "gate_verdict", "string"),),
        params={"outcome": "fail"},
        population="survey",
    )
    fit = Node(
        "fit",
        "fit@1",
        inputs=(Slice("household", ("gate_verdict",)),),
        outputs=(Owned("person", "fitted", "float64"),),
        params={"source": "age", "target": "fitted", "scale": 1.0},
        population="survey",
        artifact_outputs=(ArtifactOutput("forest", FOREST),),
    )
    release = Node(
        "release",
        "release@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("household", "tier", "string"),),
        params={"answer": "evidence", "requires_decisions": ()},
        population="survey",
        artifact_inputs=(ArtifactInput("donor", "fit", "forest", FOREST),),
    )
    graph = Graph("toy", (SOURCE,), (CREATE, gate, fit, release))
    compiled = compile_graph(graph)
    assert "fit" in compiled.predecessors["release"]

    registry = _release_registry()
    registry.register(
        _Kernel(
            "fit@1",
            Capabilities(Determinism.DETERMINISTIC),
            lambda context: KernelResult(
                columns={
                    ("person", "fitted"): pd.Series(
                        np.zeros(len(context.tables["person"])),
                        index=pd.Index(
                            context.tables["person"]["person_id"], name="person_id"
                        ),
                        dtype="float64",
                    )
                },
                artifacts={"forest": b"forest-bytes"},
            ),
        )
    )
    store = ContentStore(tmp_path / "store")
    manifest = _run(graph, _source_path(tmp_path / "src"), store, registry)
    assert manifest.nodes["release"].receipt["gate_ancestry"] == ("gate",)
    assert manifest.tier == "evidence"
    restored = RunManifest.from_json(manifest.to_json())
    assert restored.tier == "evidence"


def test_an_artifact_edge_crosses_population_versions_and_resume_policies(
    tmp_path: Path,
) -> None:
    """Amendment 19: a byte edge is not confined to one population version.

    ``fit`` lives in the ``survey`` version; ``draw`` lives in the version a
    FILTER opens. The bytes cross the boundary, the producer is still a
    predecessor, and all three resume policies agree.
    """

    def keep_all(context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        return KernelResult(
            keep=pd.Series(True, index=person["person_id"], dtype="bool")
        )

    fit = Node(
        "fit",
        "fit@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "fitted", "float64"),),
        params={"source": "age", "target": "fitted", "scale": 1.0},
        population="survey",
        artifact_outputs=(ArtifactOutput("forest", FOREST),),
    )
    boundary = Node(
        "boundary",
        "identity.filter@1",
        inputs=(Slice("person", ("selected",)),),
        structural=StructuralDelta.FILTER,
        base="survey",
    )
    draw = Node(
        "draw",
        "consume@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "drawn", "float64"),),
        population="boundary",
        artifact_inputs=(ArtifactInput("donor", "fit", "forest", FOREST),),
    )
    graph = Graph("toy", (SOURCE,), (CREATE, fit, boundary, draw))
    compiled = compile_graph(graph)
    assert "fit" in compiled.predecessors["draw"]
    assert compiled.versions["draw"] == "boundary"

    source = _source_path(tmp_path / "src")
    store = ContentStore(tmp_path / "store")

    def registry() -> KernelRegistry:
        built = _artifact_registry()
        built.register(
            _Kernel(
                "identity.filter@1",
                Capabilities(
                    Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
                ),
                keep_all,
            )
        )
        return built

    cold = _run(graph, source, store, registry())
    assert not any(receipt.hit for receipt in cold.nodes.values())
    warm = _run(graph, source, store, registry(), resume="require")
    assert all(receipt.hit for receipt in warm.nodes.values())
    assert warm.nodes["draw"].key == cold.nodes["draw"].key

    forbidden = _run(graph, source, store, registry(), resume="forbid")
    assert not any(receipt.hit for receipt in forbidden.nodes.values())
    assert forbidden.nodes["draw"].key == cold.nodes["draw"].key
    assert forbidden.nodes["draw"].typed_artifacts == cold.nodes["draw"].typed_artifacts


def test_a_cached_record_missing_a_declared_artifact_is_a_miss(tmp_path: Path) -> None:
    """Amendment 19: the miss decision lives inside the recompute fallback.

    Dropping the producer's stored artifact entry makes its cached record a
    miss, so ``auto`` re-executes the kernel and rewrites the bytes, and
    ``require`` reports the miss rather than a corrupt store.
    """
    store = ContentStore(tmp_path / "store")
    source = _source_path(tmp_path / "src")
    first = _run(_artifact_graph(), source, store, _artifact_registry())
    key = first.nodes["fit"].key
    record_key = graph_executor._cache_record_key(key)
    record = store.load_json(record_key)
    record["opaque"] = [
        entry for entry in record["opaque"] if entry["name"] != "forest"
    ]
    store.put_json(record_key, record, node_key=key, verify_existing=False)

    with pytest.raises(StoreMiss, match="cache misses before execution: 'fit'"):
        _run(_artifact_graph(), source, store, _artifact_registry(), resume="require")

    recovered = _artifact_registry()
    second = _run(_artifact_graph(), source, store, recovered)
    assert not second.nodes["fit"].hit
    assert _calls(recovered)["fit@1"] == 1
    assert second.nodes["fit"].key == key
    assert second.nodes["fit"].opaque_artifacts["forest"] == opaque_artifact_key(
        key, "forest"
    )


def test_a_cache_hit_authenticates_its_artifact_edges_without_reading_them(
    tmp_path: Path,
) -> None:
    """Amendment 19: identity is checked from receipts; bytes are read to run.

    A node that hits its cached record never runs a kernel, so its declared
    inputs' payloads are not read — but the producer receipt and the
    descriptor are still authenticated.
    """

    class _CountingStore(ContentStore):
        loads: list[str] = []

        def load_bytes(self, key: str) -> bytes:
            type(self).loads.append(key)
            return super().load_bytes(key)

    source = _source_path(tmp_path / "src")
    _CountingStore.loads = []
    store = _CountingStore(tmp_path / "store")
    cold = _run(_artifact_graph(), source, store, _artifact_registry())
    artifact = cold.nodes["fit"].opaque_artifacts["forest"]
    assert artifact in _CountingStore.loads  # the consumer ran, so it read them

    _CountingStore.loads = []
    warm = _run(_artifact_graph(), source, store, _artifact_registry())
    assert all(receipt.hit for receipt in warm.nodes.values())
    # The producer's own restore still reads its stored artifacts; no consumer
    # read happens on top of that.
    assert _CountingStore.loads.count(artifact) == 1

    # Tampering with the producer's recorded identity is still caught on a
    # hit, without any payload being read for the consumer. The guard that
    # fires is the record-shape contract check on the producer's own restore,
    # which is why the consumer's later receipt comparison never has to.
    _CountingStore.loads = []
    graph = _artifact_graph()
    record_key = graph_executor._cache_record_key(cold.nodes["fit"].key)
    record = store.load_json(record_key)
    record["typed_artifacts"]["outputs"]["forest"]["key"] = "0" * 64
    store.put_json(
        record_key, record, node_key=cold.nodes["fit"].key, verify_existing=False
    )
    with pytest.raises(StoreCorrupt, match="typed artifact contracts disagree"):
        _run(graph, source, store, _artifact_registry())
