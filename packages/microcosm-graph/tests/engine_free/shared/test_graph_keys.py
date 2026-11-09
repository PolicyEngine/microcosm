"""Node-key factorization, invalidation, and seed contracts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.decl import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    CompiledGraph,
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.kernel import (
    Capabilities,
    Determinism,
    KernelRole,
    Numeric,
    SeedSource,
    Tolerance,
)
from microcosm.graph.keys import (
    artifact_key,
    frame_key,
    node_key,
    opaque_artifact_key,
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


def _capabilities(
    structural: StructuralDelta = StructuralDelta.NONE,
) -> Capabilities:
    return Capabilities(Determinism.DETERMINISTIC, structural=structural)


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
        "toy.fit@1": "1" * 64,
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
            kernel_capabilities=_capabilities(node.structural),
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
    baseline = node_key(
        compiled,
        "model",
        keys,
        "c" * 64,
        {},
        kernel_capabilities=_capabilities(),
    )
    changed_unreachable_base = node_key(
        compiled,
        "model",
        {"survey": "d" * 64, "adults": "b" * 64},
        "c" * 64,
        {},
        kernel_capabilities=_capabilities(),
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
        kernel_capabilities=_capabilities(StructuralDelta.FILTER),
    )
    changed_patch = node_key(
        compiled,
        "adults",
        {"survey": "a" * 64, "patched": "d" * 64},
        "c" * 64,
        {},
        kernel_capabilities=_capabilities(StructuralDelta.FILTER),
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
        kernel_capabilities=_capabilities(),
    )
    changed = node_key(
        compiled,
        "consumer",
        {"survey": "a" * 64},
        "b" * 64,
        {"survey": "d" * 64},
        kernel_capabilities=_capabilities(),
    )
    assert baseline != changed


def test_every_capability_field_changes_the_node_key() -> None:
    compiled = compile_graph(_graph())
    base = Capabilities(
        determinism=Determinism.SEEDED,
        numeric=Numeric.TOLERANCE_BOUND,
        seed_source=SeedSource.EXECUTOR,
        role=KernelRole.COMPUTE,
        consumes_se=False,
        dependencies=("numpy",),
        tolerance=Tolerance(rtol=1e-6, atol=2e-6, ulps=1),
    )

    def key(capabilities: Capabilities) -> str:
        return node_key(
            compiled,
            "a",
            {"survey": "a" * 64},
            "b" * 64,
            {},
            kernel_capabilities=capabilities,
        )

    baseline = key(base)
    variants = (
        replace(base, determinism=Determinism.DETERMINISTIC),
        replace(base, numeric=Numeric.BITWISE, tolerance=None),
        replace(base, seed_source=SeedSource.PARAM),
        replace(base, structural=StructuralDelta.FILTER),
        replace(base, role=KernelRole.GATE),
        replace(base, consumes_se=True),
        replace(base, dependencies=("numpy", "pandas")),
        replace(base, tolerance=Tolerance(rtol=3e-6, atol=2e-6, ulps=1)),
    )
    assert all(key(capabilities) != baseline for capabilities in variants)

    positive_zero = replace(base, tolerance=Tolerance(rtol=0.0, atol=2e-6, ulps=1))
    negative_zero = replace(base, tolerance=Tolerance(rtol=-0.0, atol=2e-6, ulps=1))
    assert key(positive_zero) == key(negative_zero)


def test_platform_bitwise_keys_carry_the_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Amendment 16: a platform-bitwise kernel's key differs across platforms;
    other kernels' keys do not depend on the platform at all."""
    from microcosm.graph import keys as keys_module

    fingerprints = iter(["arm64/darwin/py3.14", "x86_64/linux/py3.14"])
    monkeypatch.setattr(keys_module, "platform_fingerprint", lambda: next(fingerprints))
    compiled = compile_graph(_graph())
    bound = Capabilities(Determinism.SEEDED, numeric=Numeric.PLATFORM_BITWISE)
    first = node_key(
        compiled, "a", {"survey": "s" * 64}, "impl", {}, kernel_capabilities=bound
    )
    second = node_key(
        compiled, "a", {"survey": "s" * 64}, "impl", {}, kernel_capabilities=bound
    )
    assert first != second
    plain = Capabilities(Determinism.DETERMINISTIC)
    monkeypatch.setattr(
        keys_module, "platform_fingerprint", lambda: "never/called/py0.0"
    )
    assert node_key(
        compiled, "a", {"survey": "s" * 64}, "impl", {}, kernel_capabilities=plain
    ) == node_key(
        compiled, "a", {"survey": "s" * 64}, "impl", {}, kernel_capabilities=plain
    )


_FOREST = ArtifactType("qrf.forest", 1)


def _artifact_graph(
    *,
    declare: bool = True,
    type_: ArtifactType = _FOREST,
    alias: str = "donor",
    output: str = "forest",
) -> Graph:
    producer = Node(
        "fit",
        "toy.fit@1",
        inputs=(Slice("person", ("age",)),),
        artifact_outputs=(ArtifactOutput(output, type_),) if declare else (),
    )
    consumer = Node(
        "draw",
        "toy.model@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "drawn", "float64"),),
        artifact_inputs=(
            (ArtifactInput(alias, "fit", output, type_),) if declare else ()
        ),
    )
    return Graph("toy", (SOURCE,), (CREATE, producer, consumer))


def test_opaque_artifact_key_is_the_node_key_and_the_output_name() -> None:
    """Amendment 19: a typed output keeps the pre-existing opaque identity."""
    key = "c" * 64
    expected = sha256_domain("node-artifact", canonical_json((key, "forest")))
    assert opaque_artifact_key(key, "forest") == expected
    assert opaque_artifact_key(key, "forest") != opaque_artifact_key(key, "other")
    assert opaque_artifact_key(key, "forest") != opaque_artifact_key("d" * 64, "forest")


def test_a_declared_artifact_edge_enters_both_ends_of_the_key() -> None:
    """Amendment 19: the output declaration is normative and the input adds a term."""
    _, declared = _all_keys(_artifact_graph())
    _, undeclared = _all_keys(_artifact_graph(declare=False))
    assert declared["survey"] == undeclared["survey"]
    assert declared["fit"] != undeclared["fit"]
    assert declared["draw"] != undeclared["draw"]
    _, again = _all_keys(_artifact_graph())
    assert again == declared
    # And so the output's own store identity moves with its producer: typing
    # existing bytes does not preserve the identity they were filed under.
    assert opaque_artifact_key(declared["fit"], "forest") != opaque_artifact_key(
        undeclared["fit"], "forest"
    )


def test_a_node_declaring_no_artifacts_keeps_its_pre_amendment_projection() -> None:
    """Amendment 19: empty declarations are elided, so they add no key term."""
    compiled = compile_graph(_artifact_graph(declare=False))
    for node_id in compiled.order:
        projection = compiled.graph.node(node_id).normative()
        assert "artifact_inputs" not in projection
        assert "artifact_outputs" not in projection
    declared = compile_graph(_artifact_graph()).graph.node("draw").normative()
    assert set(declared) - set(compiled.graph.node("draw").normative()) == {
        "artifact_inputs"
    }


def test_every_part_of_an_artifact_declaration_is_normative() -> None:
    """Amendment 19: no field of a declared edge is inert."""
    _, base = _all_keys(_artifact_graph())
    _, retyped = _all_keys(_artifact_graph(type_=ArtifactType("qrf.forest", 2)))
    _, renamed_type = _all_keys(_artifact_graph(type_=ArtifactType("other", 1)))
    _, realiased = _all_keys(_artifact_graph(alias="teacher"))
    _, renamed_output = _all_keys(_artifact_graph(output="trees"))
    # A type or output rename moves both ends; the consumer-local alias moves
    # only the consumer, because the producer never sees it.
    for moved in (retyped, renamed_type, renamed_output):
        assert moved["fit"] != base["fit"]
        assert moved["draw"] != base["draw"]
    assert realiased["fit"] == base["fit"]
    assert realiased["draw"] != base["draw"]
    assert base["survey"] == realiased["survey"] == retyped["survey"]


def test_a_byte_edge_carries_descendant_exact_invalidation() -> None:
    """Amendment 19 keeps A3 over bytes: the producer's key is in the consumer's.

    The producer has its own kernel ref, so re-hashing only its
    implementation isolates the byte edge: the consumer reads no cell of the
    producer, and its key moves anyway.
    """
    graph = _artifact_graph()
    fit_hashes = {"toy.fit@1": "1" * 64}
    _, base = _all_keys(graph, hashes=fit_hashes)
    _, moved = _all_keys(graph, hashes={"toy.fit@1": "e" * 64})
    assert moved["survey"] == base["survey"]
    assert moved["fit"] != base["fit"]
    assert moved["draw"] != base["draw"]
    # Without the declaration the same producer edit leaves the consumer alone.
    plain = _artifact_graph(declare=False)
    _, plain_base = _all_keys(plain, hashes=fit_hashes)
    _, plain_moved = _all_keys(plain, hashes={"toy.fit@1": "e" * 64})
    assert plain_moved["fit"] != plain_base["fit"]
    assert plain_moved["draw"] == plain_base["draw"]
