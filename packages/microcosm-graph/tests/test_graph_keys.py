"""Node-key factorization, invalidation, and seed contracts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.decl import (
    CompiledGraph,
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.keys import (
    artifact_key,
    frame_key,
    node_key,
    seed,
    source_content_key,
    weights_key,
)

SOURCE = SourceRef("survey", "csv-tables")
CREATE = Node(
    "survey",
    "source.frame@1",
    structural=StructuralDelta.CREATE,
    sources=("survey",),
    outputs=(
        Owned("person", "age", "int64"),
        Owned("person", "keep", "boolean"),
    ),
)


def _ordinary(
    node_id: str,
    inputs: tuple[str, ...],
    output: str,
    *,
    parameter: int = 1,
    description: str = "",
) -> Node:
    return Node(
        node_id,
        "toy.model@1",
        inputs=(Slice("person", inputs),),
        outputs=(Owned("person", output, "float64"),),
        params={"parameter": parameter},
        description=description,
    )


def _graph(*, parameter: int = 1, leaf: bool = True) -> Graph:
    a = _ordinary("a", ("age",), "a", parameter=parameter)
    b = _ordinary("b", ("a",), "b")
    nodes = [CREATE, a, b]
    if leaf:
        nodes.append(_ordinary("leaf", ("age",), "leaf"))
    return Graph("toy", (SOURCE,), tuple(nodes))


def _all_keys(
    graph: Graph,
    *,
    hashes: dict[str, str] | None = None,
    source_key: str = "1" * 64,
) -> tuple[CompiledGraph, dict[str, str]]:
    compiled = compile_graph(graph)
    implementation_hashes = {
        "source.frame@1": "a" * 64,
        "toy.model@1": "b" * 64,
        **(hashes or {}),
    }
    keys: dict[str, str] = {}
    for node_id in compiled.order:
        node = graph.node(node_id)
        keys[node_id] = node_key(
            compiled,
            node_id,
            keys,
            implementation_hashes[node.kernel],
            {"survey": source_key},
        )
    return compiled, keys


def test_source_key_is_path_invariant_and_content_sensitive(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "renamed.csv"
    left.write_bytes(b"same bytes\n")
    right.write_bytes(b"same bytes\n")
    expected = sha256_domain(
        "source",
        canonical_json(
            (
                "survey",
                hashlib.sha256(b"same bytes\n").hexdigest(),
                len(b"same bytes\n"),
            )
        ),
    )
    assert source_content_key("survey", left) == expected
    assert source_content_key("survey", right) == expected
    right.write_bytes(b"different\n")
    assert source_content_key("survey", right) != expected


def test_directory_source_identity_ignores_root_path(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        root.mkdir()
        (root / "schema.json").write_text("{}")
        (root / "person.csv").write_text("id\n1\n")
    assert source_content_key("survey", left) == source_content_key("survey", right)
    (right / "person.csv").write_text("id\n2\n")
    assert source_content_key("survey", left) != source_content_key("survey", right)


def test_artifact_domains_and_seed_are_exact() -> None:
    key = "f" * 64
    assert artifact_key(key, "person", "age") == sha256_domain(
        "artifact", canonical_json((key, "person", "age"))
    )
    assert frame_key(key) == sha256_domain("frame", canonical_json((key,)))
    assert weights_key(key, "person") == sha256_domain(
        "weights", canonical_json((key, "person"))
    )
    assert artifact_key(key, "person", "age") != frame_key(key)
    assert weights_key(key, "person") != artifact_key(key, "person", "__weights__")
    expected_seed = int.from_bytes(
        hashlib.sha256(b"seed\0" + key.encode()).digest()[:8], "little"
    )
    assert seed(key) == expected_seed


def test_declaration_order_and_unrelated_leaf_are_invariant() -> None:
    full = _graph()
    shuffled = Graph("toy", (SOURCE,), tuple(reversed(full.nodes)))
    _, full_keys = _all_keys(full)
    _, shuffled_keys = _all_keys(shuffled)
    _, without_leaf = _all_keys(_graph(leaf=False))
    assert shuffled_keys == full_keys
    assert {node_id: full_keys[node_id] for node_id in without_leaf} == without_leaf


def test_parameter_and_kernel_hash_invalidate_exact_descendants() -> None:
    _, baseline = _all_keys(_graph())
    _, changed_parameter = _all_keys(_graph(parameter=2))
    assert {
        node_id
        for node_id in baseline
        if baseline[node_id] != changed_parameter[node_id]
    } == {"a", "b"}

    changed_a_kernel = Graph(
        "toy",
        (SOURCE,),
        tuple(
            replace(node, kernel="toy.changed@1") if node.id == "a" else node
            for node in _graph().nodes
        ),
    )
    _, changed_code = _all_keys(changed_a_kernel, hashes={"toy.changed@1": "c" * 64})
    assert {
        node_id for node_id in baseline if baseline[node_id] != changed_code[node_id]
    } == {"a", "b"}


def test_descriptive_fields_change_no_key() -> None:
    graph = _graph()
    described = Graph(
        "a different country label",
        (replace(SOURCE, description="source prose"),),
        tuple(
            replace(node, description="node prose", citation="paper")
            for node in graph.nodes
        ),
    )
    assert _all_keys(graph)[1] == _all_keys(described)[1]


def test_carried_columns_resolve_to_the_structural_version() -> None:
    subset = Node(
        "adults",
        "toy.filter@1",
        structural=StructuralDelta.FILTER,
        base="survey",
        inputs=(Slice("person", ("keep",)),),
    )
    model = replace(_ordinary("model", ("age",), "modeled"), population="adults")
    graph = Graph("toy", (SOURCE,), (CREATE, subset, model))
    compiled = compile_graph(graph)
    keys = {"survey": "a" * 64, "adults": "b" * 64}
    baseline = node_key(compiled, "model", keys, "c" * 64, {})
    changed_unreachable_base = node_key(
        compiled,
        "model",
        {"survey": "d" * 64, "adults": "b" * 64},
        "c" * 64,
        {},
    )
    assert baseline == changed_unreachable_base


def test_structural_key_binds_every_patch_in_its_base_version() -> None:
    patched = replace(_ordinary("patched", ("age",), "patched"), population="survey")
    subset = Node(
        "adults",
        "toy.filter@1",
        structural=StructuralDelta.FILTER,
        base="survey",
        inputs=(Slice("person", ("keep",)),),
    )
    graph = Graph("toy", (SOURCE,), (CREATE, patched, subset))
    compiled = compile_graph(graph)
    baseline = node_key(
        compiled,
        "adults",
        {"survey": "a" * 64, "patched": "b" * 64},
        "c" * 64,
        {},
    )
    changed_patch = node_key(
        compiled,
        "adults",
        {"survey": "a" * 64, "patched": "d" * 64},
        "c" * 64,
        {},
    )
    assert baseline != changed_patch


def test_non_create_source_consumers_bind_their_declared_source_bytes() -> None:
    consumer = replace(_ordinary("consumer", ("age",), "value"), sources=("survey",))
    graph = Graph("toy", (SOURCE,), (CREATE, consumer))
    compiled = compile_graph(graph)
    baseline = node_key(
        compiled,
        "consumer",
        {"survey": "a" * 64},
        "b" * 64,
        {"survey": "c" * 64},
    )
    changed = node_key(
        compiled,
        "consumer",
        {"survey": "a" * 64},
        "b" * 64,
        {"survey": "d" * 64},
    )
    assert baseline != changed
