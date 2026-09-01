"""The one-screen graph-node view contract."""

from __future__ import annotations

from microcosm.graph.decl import (
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.kernel import Capabilities, Determinism, SeedSource
from microcosm.graph.keys import seed
from microcosm.graph.manifest import NodeReceipt, RunManifest
from microcosm.graph.view import describe


def _compiled_graph():
    create = Node(
        "survey",
        "source.frame@1",
        sources=("survey",),
        outputs=(Owned("person", "age", "int64"),),
        structural=StructuralDelta.CREATE,
    )
    model = Node(
        "model",
        "toy.model@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "income", "float64"),),
        params={"trees": 100},
    )
    return compile_graph(
        Graph("toy", (SourceRef("survey", "csv-tables"),), (model, create))
    )


def _receipt(key: str, structural: StructuralDelta, *, hit: bool) -> NodeReceipt:
    return NodeReceipt(
        key=key,
        hit=hit,
        seed=seed(key),
        kernel_ref=("source.frame@1" if structural else "toy.model@1"),
        kernel_impl_hash="f" * 64,
        capabilities=Capabilities(
            determinism=Determinism.SEEDED,
            seed_source=SeedSource.EXECUTOR,
            structural=structural,
        ),
        receipt={"rows": 3},
        artifacts={("person", "age"): "a" * 64},
        wall_time=0.125,
    )


def test_describe_uses_manifest_for_runtime_identity_and_receipt() -> None:
    compiled = _compiled_graph()
    manifest = RunManifest(
        "toy",
        {
            "survey": _receipt("1" * 64, StructuralDelta.CREATE, hit=False),
            "model": _receipt("2" * 64, StructuralDelta.NONE, hit=True),
        },
    )
    rendered = describe(compiled, "model", manifest)
    assert len(rendered.splitlines()) < 40
    assert "Node: model" in rendered
    assert "Version: survey" in rendered
    assert f"survey [{'1' * 64}]" in rendered
    assert "Inputs: person[age] rows=all" in rendered
    assert "Owned cells: person.income:float64 rows=all" in rendered
    assert 'Parameters: {"trees":100}' in rendered
    assert "Kernel: toy.model@1" in rendered
    assert f"Implementation hash: {'f' * 64}" in rendered
    assert f"= {seed('2' * 64)}" in rendered
    assert "Store: hit" in rendered
    assert 'Receipt: {"rows":3}' in rendered


def test_describe_graph_alone_marks_runtime_facts_unavailable() -> None:
    rendered = describe(_compiled_graph(), "model")
    assert len(rendered.splitlines()) < 40
    assert "<available at run time>" in rendered
    assert "seed\\0" in rendered
    assert "Store:" not in rendered
