"""Charter group C: seeds and factorization.

Commit ``0347a009`` removed five person targets from the late PUF-transfer
surface. The generic max-eight packer then repacked the 32 survivors from five
batches into four; 31 of them moved, drew from a different RNG position, and
conditioned on a different predecessor prefix. Eight regressed in the by-origin
battery. Nothing about any of those 31 targets had been edited.

These four properties are the shape that makes that impossible: a node's
identity, its predecessors, and its RNG stream come from its own declaration
and nowhere else. The declaration-order half of C1 and the predecessor half of
C3 are already covered at compile time in ``test_graph_decl.py``; what is
asserted here is that the executor honours them in keys, seeds, and bytes.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    Node,
    Numeric,
    Owned,
    Slice,
    Tolerance,
    compile_graph,
)

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]


def _dotted(node: ast.expr) -> str:
    """The dotted name of an expression, or "" when it is not a plain name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_c1_order_invariance(tmp_path: Path) -> None:
    """Declaration order carries no meaning: no key, no seed, no byte moves.

    The frozen API exposes no batch-packing knob, which is the point: there is
    no container whose repacking could move a node. What can still be permuted
    is the order the nodes are declared in, so that is what is permuted.
    """
    graph = toy.small_graph()
    shuffled = Graph("toy", graph.sources, tuple(reversed(graph.nodes)))
    assert compile_graph(graph).order == compile_graph(shuffled).order

    forward = toy.run_toy(graph, tmp_path / "forward")
    registry = toy.toy_registry()
    backward = toy.run_toy(
        shuffled,
        tmp_path / "forward",
        sources=forward.sources,
        registry=registry,
        store=ContentStore(tmp_path / "forward" / "store"),
    )
    assert backward.keys() == forward.keys()
    assert backward.seeds() == forward.seeds()
    assert backward.all_bytes() == forward.all_bytes()
    assert toy.total_calls(registry) == 0


def test_c2_removal_invariance(tmp_path: Path) -> None:
    """Removing a leaf, or adding one, moves no other node's key or output.

    This is the ``0347a009`` shape stated as a general property: five targets
    removed, zero survivors re-modelled; and the mirror, one target added,
    zero incumbents re-modelled.
    """
    five = toy.chained_graph(("a", "b", "c", "d", "e"))
    full = toy.run_toy(five, tmp_path / "run")

    registry = toy.toy_registry()
    trimmed = toy.run_toy(
        toy.chained_graph(()),
        tmp_path / "run",
        sources=full.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    survivors = set(trimmed.compiled.order)
    assert survivors == set(full.compiled.order) - {f"leaf_{n}" for n in "abcde"}
    assert trimmed.keys() == {k: v for k, v in full.keys().items() if k in survivors}
    assert trimmed.seeds() == {k: v for k, v in full.seeds().items() if k in survivors}
    assert trimmed.all_bytes() == {
        k: v for k, v in full.all_bytes().items() if k in survivors
    }
    assert trimmed.misses() == set()
    assert toy.total_calls(registry) == 0

    grown_registry = toy.toy_registry()
    grown = toy.run_toy(
        toy.chained_graph(("a", "b", "c", "d", "e", "f")),
        tmp_path / "run",
        sources=full.sources,
        registry=grown_registry,
        store=ContentStore(tmp_path / "run" / "store"),
    )
    assert grown.misses() == {"leaf_f"}
    assert toy.total_calls(grown_registry) == 1
    for node_id in full.compiled.order:
        assert grown.keys()[node_id] == full.keys()[node_id]
        assert grown.all_bytes()[node_id] == full.all_bytes()[node_id]


def test_c3_declared_predecessors_only(tmp_path: Path) -> None:
    """A kernel is handed its declared slices and nothing else.

    An undeclared read is impossible rather than merely detected: the column
    is not in the projection, so reading it raises inside the kernel.
    """
    run = toy.run_toy(toy.small_graph(), tmp_path / "run")
    receipt = run.manifest.nodes["target_b"].receipt
    assert receipt["predictors"] == ("age", "target_a")

    structural = {"person_id", "person_household_id", "person_release_id"}
    seen = set(receipt["columns_seen"])
    assert seen - structural == {"age", "target_a"}
    assert receipt["entities_seen"] == ("person",)
    # Undeclared columns are not withheld from the kernel by policy; they are
    # simply not in the projection, so an undeclared read cannot compile away.
    assert {"income", "noise_a"}.isdisjoint(seen)
    assert run.compiled.predecessors["target_b"] == ("survey", "target_a")


def test_c4_seed_from_identity(tmp_path: Path) -> None:
    """A node's seed is a pure function of its key, in any graph.

    Two graphs that differ elsewhere but declare the same node, over the same
    inputs and the same kernel, give it the same key, the same seed, and
    byte-identical draws. And no positional RNG consumption exists anywhere in
    the shard, so no container could supply a stream instead.
    """
    for path in toy.graph_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        dotted = {
            _dotted(call.func) for call in ast.walk(tree) if isinstance(call, ast.Call)
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "RandomState" not in names, f"{path.name} uses legacy global RNG state"
        assert not any(name.endswith("random.seed") for name in dotted), (
            f"{path.name} seeds the process-global generator"
        )
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and _dotted(call.func).endswith(
                "default_rng"
            ):
                assert len(call.args) + len(call.keywords) == 1, (
                    f"{path.name} seeds a generator from more than one value"
                )

    here = toy.run_toy(toy.small_graph(), tmp_path / "here")
    elsewhere = toy.run_toy(
        toy.small_graph(extra=(toy.draw("draw_z", "noise_z"),)), tmp_path / "elsewhere"
    )
    assert elsewhere.keys()["draw_a"] == here.keys()["draw_a"]
    assert elsewhere.seeds()["draw_a"] == here.seeds()["draw_a"]
    assert elsewhere.bytes_of("draw_a") == here.bytes_of("draw_a")

    assert elsewhere.keys()["draw_z"] != here.keys()["draw_a"]
    assert elsewhere.seeds()["draw_z"] != here.seeds()["draw_a"]
    assert len(set(elsewhere.seeds().values())) == len(elsewhere.seeds())


def test_c5_tolerance_is_declared(tmp_path: Path) -> None:
    """Receipts and readers carry an owner's exact declared tolerance.

    Capability receipts encode a tolerance as ``rtol``, ``atol``, and ``ulps``;
    bitwise owners encode it as ``None``. A gate reads that same owner mapping
    by coordinate and reports the JSON-safe value in its evidence.
    """
    with pytest.raises(ValueError, match="must declare its Tolerance"):
        Capabilities(
            determinism=Determinism.DETERMINISTIC,
            numeric=Numeric.TOLERANCE_BOUND,
        )
    with pytest.raises(ValueError, match="bitwise kernel declares no Tolerance"):
        Capabilities(
            determinism=Determinism.DETERMINISTIC,
            numeric=Numeric.BITWISE,
            tolerance=Tolerance(rtol=1e-6),
        )

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

    def tolerance_gate(node_id: str, column: str) -> Node:
        verdict = f"{node_id}_verdict"
        return Node(
            node_id,
            "gate.tolerance@1",
            inputs=(Slice("person", (column,)),),
            outputs=(Owned("release", verdict, "string"),),
            params={
                "entity": "person",
                "column": column,
                "verdict_column": verdict,
            },
            population="survey",
        )

    bounded_gate = tolerance_gate("bounded_gate", "bounded_value")
    bitwise_gate = tolerance_gate("bitwise_gate", "bitwise_value")
    graph = Graph(
        "toy",
        (toy.SOURCE,),
        (toy.CREATE, bounded, bitwise, bounded_gate, bitwise_gate),
    )
    run = toy.run_toy(graph, tmp_path / "run")
    bound = {"rtol": 1e-6, "atol": 0.0, "ulps": 0}

    bounded_receipt = run.manifest.nodes[bounded.id].receipt
    assert bounded_receipt["capabilities"]["tolerance"] == bound
    bounded_evidence = run.manifest.nodes[bounded_gate.id].receipt
    assert bounded_evidence["outcome"] == "pass"
    assert bounded_evidence["evidence"]["tolerance"] == bound

    bitwise_receipt = run.manifest.nodes[bitwise.id].receipt
    assert bitwise_receipt["capabilities"]["tolerance"] is None
    bitwise_evidence = run.manifest.nodes[bitwise_gate.id].receipt
    assert bitwise_evidence["outcome"] == "pass"
    assert bitwise_evidence["evidence"]["tolerance"] is None
