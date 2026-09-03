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
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.decl import (
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
from microcosm.graph.executor import NodeRejected, run_graph
from microcosm.graph.kernel import (
    Capabilities,
    Determinism,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Numeric,
    Tolerance,
)
from microcosm.graph.manifest import Decision
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
    assert warm.key != first.key
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


def test_entrant_expand_with_zero_entrants_allows_declared_materialized_claim(
    tmp_path: Path,
) -> None:
    source = _source_path(tmp_path / "source")

    def admit_no_entrants(context: KernelContext) -> KernelResult:
        household = context.tables["household"]
        return KernelResult(
            expand={
                "person": pd.Series(
                    [],
                    index=pd.Index([], name="person_id", dtype="int64"),
                    dtype="int64",
                ),
                "household": pd.Series(
                    [],
                    index=pd.Index([], name="household_id", dtype="int64"),
                    dtype="int64",
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
        sources={"fixture": case},
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
            sources={"fixture": case},
            store=store,
            kernels=executor_registry,
        )


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
