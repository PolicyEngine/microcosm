"""A failed typed gate remains evidence without inventing downstream data."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelResult,
    Node,
    NodeRejectedError,
    Owned,
    RunManifest,
    StoreCorruptError,
    StoreMissError,
    compile_graph,
    describe,
    run_graph,
)
from microcosm.graph.availability import EXECUTION_SCHEMA
from microcosm.graph.kernel import KernelRole

spec = importlib.util.spec_from_file_location(
    "_gate_artifact_toy", Path(__file__).with_name("_toy.py")
)
toy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = toy
spec.loader.exec_module(toy)
EVIDENCE = ArtifactType("test.gate-evidence", 1)


class Gate(KernelBase):
    ref = "test.typed-gate@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE)
    calls = 0

    def run(self, context):
        self.calls += 1
        raise RuntimeError("evidence could not be computed")


class Consume(KernelBase):
    ref = "test.consume-evidence@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)
    calls = 0

    def run(self, context):
        self.calls += 1
        raise AssertionError("An unavailable typed value must never reach a kernel.")


def fixture(tmp_path, *, release="structural"):
    gate = Node(
        "gate",
        Gate.ref,
        population="survey",
        outputs=(Owned("release", "gate_verdict", "string"),),
        artifact_outputs=(ArtifactOutput("evidence", EVIDENCE),),
    )
    byte_input = ArtifactInput("evidence", "gate", "evidence", EVIDENCE)
    nodes = [toy.CREATE, gate, toy.derive("independent", ("age",), "age_copy")]
    if release == "structural":
        nodes.extend(
            [
                Node(
                    "consumer",
                    Consume.ref,
                    population="survey",
                    outputs=(Owned("person", "unavailable", "float64"),),
                    artifact_inputs=(byte_input,),
                    artifact_outputs=(ArtifactOutput("transformed", EVIDENCE),),
                ),
                Node(
                    "byte_child",
                    Consume.ref,
                    population="survey",
                    artifact_inputs=(
                        ArtifactInput("model", "consumer", "transformed", EVIDENCE),
                    ),
                ),
                toy.derive("column_child", ("unavailable",), "unavailable_copy"),
                toy.select_node(),
                toy.derive("structural_child", ("age",), "later", population="adults"),
                toy.release_node(population="adults"),
            ]
        )
    else:
        final = toy.release_node(population="survey")
        nodes.append(
            replace(final, artifact_inputs=(byte_input,))
            if release == "byte"
            else final
        )
    compiled = compile_graph(Graph("toy", (toy.SOURCE,), tuple(nodes)))
    registry = toy.toy_registry()
    gate_kernel, consumer = Gate(), Consume()
    registry.register(gate_kernel)
    registry.register(consumer)
    return (
        compiled,
        toy.toy_sources(tmp_path),
        ContentStore(tmp_path / "store"),
        registry,
        gate_kernel,
        consumer,
    )


@pytest.mark.parametrize("release", ["structural", "byte", "verdict"])
def test_exception_evidence_and_unreached_dependents_replay(tmp_path, release):
    compiled, sources, store, registry, gate, consumer = fixture(
        tmp_path, release=release
    )
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    assert cold.tier == "evidence"
    assert cold.nodes["gate"].receipt["outcome"] == "fail"
    assert cold.nodes["gate"].receipt["evidence"]["exception_type"] == "RuntimeError"
    assert cold.nodes["gate"].receipt["execution"]["unavailable_artifacts"] == (
        "evidence",
    )
    assert not cold.nodes["gate"].opaque_artifacts
    assert "age_copy" in cold.population("survey").table("person")
    assert consumer.calls == 0
    if release != "verdict":
        assert cold.nodes["release"].receipt["outcome"] == "unreached"
        assert not cold.nodes["release"].artifacts
    if release == "structural":
        for node_id in (
            "consumer",
            "byte_child",
            "column_child",
            "adults",
            "structural_child",
            "release",
        ):
            receipt = cold.nodes[node_id]
            assert receipt.receipt["execution"]["state"] == "unreached"
            assert not receipt.artifacts and not receipt.opaque_artifacts
            assert receipt.frame_key is receipt.weight_key is None
        assert "adults" not in cold.populations
        assert "unavailable" not in cold.population("survey").table("person")
        assert "unreached" in describe(compiled, "consumer", cold)
    calls = toy.total_calls(registry)
    for policy in ("auto", "require"):
        replay = run_graph(
            compiled, sources=sources, store=store, kernels=registry, resume=policy
        )
        assert replay.key == cold.key and all(
            node.hit for node in replay.nodes.values()
        )
        assert gate.calls == 1 and consumer.calls == 0
        assert toy.total_calls(registry) == calls
    path = tmp_path / "manifest.json"
    cold.save(path)
    assert json.loads(path.read_text())["schema_version"] == 4
    assert RunManifest.load(path, store).key == cold.key
    with pytest.raises(NodeRejectedError, match="unreached|evidence"):
        RunManifest.load_certified(path, store)


def test_require_refuses_missing_unreached_receipt_before_any_execution(tmp_path):
    import shutil

    from microcosm.graph.executor import _cache_record_key

    compiled, sources, store, registry, gate, consumer = fixture(tmp_path)
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    shutil.rmtree(store.object_path(_cache_record_key(cold.nodes["consumer"].key)))
    before = toy.total_calls(registry)
    with pytest.raises(StoreMissError, match="consumer"):
        run_graph(
            compiled, sources=sources, store=store, kernels=registry, resume="require"
        )
    assert (
        toy.total_calls(registry) == before and gate.calls == 1 and consumer.calls == 0
    )


@pytest.mark.parametrize("parent", ["missing", "survey", "consumer"])
def test_forged_dangling_or_cyclic_blocker_refused(tmp_path, parent):
    compiled, sources, store, registry, _, _ = fixture(tmp_path)
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    nodes = dict(cold.nodes)
    receipt = dict(nodes["consumer"].receipt)
    receipt["execution"] = {
        "schema": EXECUTION_SCHEMA,
        "state": "unreached",
        "blocked_by": {
            parent: cold.nodes[parent].key if parent in cold.nodes else "a" * 64,
        },
    }
    nodes["consumer"] = replace(nodes["consumer"], receipt=receipt)
    with pytest.raises(ValueError, match="block|unavailable|cycle"):
        replace(cold, nodes=nodes)


def test_exception_manifest_cannot_downgrade_or_claim_available_bytes(tmp_path):
    compiled, sources, store, registry, _, _ = fixture(tmp_path)
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    raw = json.loads(cold.to_json())
    raw["schema_version"] = 3
    with pytest.raises(ValueError, match="schema 4"):
        RunManifest.from_json(json.dumps(raw))
    nodes = dict(cold.nodes)
    with pytest.raises(ValueError, match="unavailable|artifact"):
        nodes["gate"] = replace(nodes["gate"], opaque_artifacts={"evidence": "a" * 64})
        replace(cold, nodes=nodes)


def test_forbid_reexecutes_gate_but_never_unavailable_consumers(tmp_path):
    compiled, sources, store, registry, gate, consumer = fixture(tmp_path)
    first = run_graph(compiled, sources=sources, store=store, kernels=registry)
    again = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    assert again.key == first.key
    assert not any(node.hit for node in again.nodes.values())
    assert gate.calls == 2 and consumer.calls == 0


def test_artifact_only_gate_exception_returns_evidence_receipt(tmp_path):
    compiled, sources, store, registry, gate, _ = fixture(tmp_path)
    original = compiled.graph
    compiled = compile_graph(
        replace(
            original,
            nodes=(
                original.node("survey"),
                replace(original.node("gate"), outputs=()),
            ),
        )
    )
    first = run_graph(compiled, sources=sources, store=store, kernels=registry)
    assert first.nodes["gate"].receipt["outcome"] == "fail"
    assert (
        not first.nodes["gate"].artifacts and not first.nodes["gate"].opaque_artifacts
    )
    again = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="require"
    )
    assert again.key == first.key and gate.calls == 1


@pytest.mark.parametrize("resume", ["auto", "require"])
def test_cached_blockers_are_rederived_from_actual_dependencies(tmp_path, resume):
    from microcosm.graph.executor import _cache_record_key

    compiled, sources, store, registry, gate, consumer = fixture(tmp_path)
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    key = cold.nodes["consumer"].key
    record_key = _cache_record_key(key)
    record = store.load_json(record_key)
    record["receipt"]["execution"]["blocked_by"] = {"survey": cold.nodes["survey"].key}
    store.put_json(record_key, record, node_key=key, verify_existing=False)
    with pytest.raises(StoreCorruptError, match="blocked provenance"):
        run_graph(
            compiled, sources=sources, store=store, kernels=registry, resume=resume
        )
    assert gate.calls == 1 and consumer.calls == 0


@pytest.mark.parametrize(
    "diagnostic", ["legacy", {"state": "unreached", "detail": "legacy diagnostic"}]
)
def test_legacy_free_form_execution_metadata_remains_portable(tmp_path, diagnostic):
    from microcosm.graph import NodeReceipt

    node = NodeReceipt(
        key="a" * 64,
        hit=False,
        seed=1,
        kernel_ref="legacy@1",
        kernel_impl_hash="b" * 64,
        capabilities=Capabilities(Determinism.DETERMINISTIC),
        receipt={"execution": diagnostic},
    )
    manifest = RunManifest(country="toy", nodes={"legacy": node})
    raw = manifest.to_json()
    assert json.loads(raw)["schema_version"] == 2
    assert RunManifest.from_json(raw).key == manifest.key


@pytest.mark.parametrize("field", ["frame_key", "weight_key"])
def test_unreached_cannot_claim_an_empty_product_key(tmp_path, field):
    compiled, sources, store, registry, _, _ = fixture(tmp_path)
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    with pytest.raises(ValueError, match="products"):
        replace(cold.nodes["consumer"], **{field: ""})


class InvalidReceipt(KernelBase):
    ref = "test.invalid-receipt@1"

    def __init__(self, role, receipt):
        self.capabilities = Capabilities(Determinism.DETERMINISTIC, role=role)
        self.receipt = receipt

    def run(self, context):
        return KernelResult(artifacts={"evidence": b"computed"}, receipt=self.receipt)


@pytest.mark.parametrize("role", [KernelRole.GATE, KernelRole.COMPUTE])
@pytest.mark.parametrize("shape", ["namespace", "nonmapping"])
def test_invalid_returned_receipt_rejects_without_caching_failed_evidence(
    tmp_path, role, shape
):
    from microcosm.graph.executor import _all_node_keys, _cache_record_key
    from microcosm.graph.keys import source_content_key

    _, sources, store, registry, _, _ = fixture(tmp_path)
    receipt = (
        {"execution": {"schema": EXECUTION_SCHEMA, "state": "unreached"}}
        if shape == "namespace"
        else []
    )
    kernel = InvalidReceipt(role, receipt)
    registry.register(kernel)
    node = Node(
        "invalid",
        kernel.ref,
        population="survey",
        artifact_outputs=(ArtifactOutput("evidence", EVIDENCE),),
    )
    compiled = compile_graph(Graph("toy", (toy.SOURCE,), (toy.CREATE, node)))
    keys, _ = _all_node_keys(
        compiled,
        registry,
        {name: source_content_key(name, path) for name, path in sources.items()},
    )
    message = (
        "executor execution metadata"
        if shape == "namespace"
        else "receipt is not a mapping"
    )
    with pytest.raises(NodeRejectedError, match=message):
        run_graph(compiled, sources=sources, store=store, kernels=registry)
    assert not store.has(_cache_record_key(keys[node.id]))


@pytest.mark.parametrize("stale", ["legacy", "different"])
def test_auto_replaces_stale_unreached_receipt_before_required_replay(tmp_path, stale):
    from microcosm.graph.executor import _cache_record_key

    compiled, sources, store, registry, gate, consumer = fixture(tmp_path)
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    key = cold.nodes["consumer"].key
    record_key = _cache_record_key(key)
    record = store.load_json(record_key)
    correct_capabilities = dict(record["capabilities"])
    if stale == "legacy":
        del record["capabilities"]["tolerance"]
    else:
        record["capabilities"]["role"] = "gate"
    store.put_json(record_key, record, node_key=key, verify_existing=False)
    repaired = run_graph(compiled, sources=sources, store=store, kernels=registry)
    assert not repaired.nodes["consumer"].hit
    assert store.load_json(record_key)["capabilities"] == correct_capabilities
    required = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="require"
    )
    assert required.key == cold.key
    assert all(node.hit for node in required.nodes.values())
    assert gate.calls == 1 and consumer.calls == 0


class TransientGate(KernelBase):
    ref = "test.transient-typed-gate@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE)
    unavailable = True
    calls = 0

    def run(self, context):
        self.calls += 1
        if self.unavailable:
            raise RuntimeError("Temporary evidence failure")
        return KernelResult(
            artifacts={"evidence": b"available"},
            receipt={"outcome": "pass", "evidence": {"checked": True}},
        )


class AvailableConsumer(KernelBase):
    ref = "test.available-evidence-consumer@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)
    calls = 0

    def run(self, context):
        self.calls += 1
        assert context.artifacts["evidence"].payload == b"available"
        return KernelResult(receipt={"outcome": "pass"})


def test_full_forbid_replay_recovers_after_transient_gate_failure(tmp_path):
    import shutil

    from microcosm.graph.executor import _cache_record_key

    _, sources, store, registry, _, _ = fixture(tmp_path)
    gate, consumer = TransientGate(), AvailableConsumer()
    registry.register(gate)
    registry.register(consumer)
    compiled = compile_graph(
        Graph(
            "toy",
            (toy.SOURCE,),
            (
                toy.CREATE,
                Node(
                    "gate",
                    gate.ref,
                    population="survey",
                    artifact_outputs=(ArtifactOutput("evidence", EVIDENCE),),
                ),
                Node(
                    "consumer",
                    consumer.ref,
                    population="survey",
                    artifact_inputs=(
                        ArtifactInput("evidence", "gate", "evidence", EVIDENCE),
                    ),
                ),
            ),
        )
    )
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    gate.unavailable = False
    cached = run_graph(compiled, sources=sources, store=store, kernels=registry)
    assert cached.nodes["gate"].receipt["outcome"] == "fail"
    assert gate.calls == 1 and consumer.calls == 0
    # Selective eviction cannot make cached descendants appear available.
    shutil.rmtree(store.object_path(_cache_record_key(cold.nodes["gate"].key)))
    with pytest.raises(StoreCorruptError, match="no unavailable input blocker"):
        run_graph(compiled, sources=sources, store=store, kernels=registry)
    fresh = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    assert fresh.nodes["gate"].receipt["outcome"] == "pass"
    assert fresh.nodes["consumer"].receipt["outcome"] == "pass"
    required = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="require"
    )
    assert all(node.hit for node in required.nodes.values())
    assert required.key == fresh.key
    assert gate.calls == 3 and consumer.calls == 1
