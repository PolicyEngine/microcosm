"""Typed model edges preserve ownership, reuse and numeric provenance."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    GraphError,
    KernelBase,
    KernelResult,
    Node,
    NodeRejectedError,
    Numeric,
    Owned,
    RunManifest,
    Slice,
    Tolerance,
    compile_graph,
    describe,
    graph_from_json,
    graph_to_json,
    run_graph,
)
from microcosm.graph.decl import StructuralDelta
from microcosm.graph.kernel import KernelRole, SeedSource
from microcosm.graph.keys import node_key, seed

spec = importlib.util.spec_from_file_location(
    "_artifact_toy", Path(__file__).with_name("_toy.py")
)
toy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = toy
spec.loader.exec_module(toy)
MODEL = ArtifactType("test.scalar-model", 1)


class Train(KernelBase):
    ref = "artifact.train@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)
    calls = 0

    def run(self, context):
        self.calls += 1
        value = float(context.tables["person"]["income"].mean())
        return KernelResult(artifacts={"model": str(value).encode(), "debug": b"extra"})


class Apply(KernelBase):
    ref = "artifact.apply@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)
    calls = 0

    def run(self, context):
        self.calls += 1
        assert set(context.artifacts) == {"fitted"}
        fitted = context.artifacts["fitted"]
        assert isinstance(fitted, ArtifactValue) and fitted.type == MODEL
        with pytest.raises(TypeError):
            context.artifacts["unrelated"] = fitted
        assert isinstance(fitted.payload, bytes)
        table = context.tables["person"]
        values = table.age.to_numpy() + float(fitted.payload)
        return KernelResult(
            columns={
                ("person", "predicted"): pd.Series(
                    values,
                    index=pd.Index(table.person_id, name="person_id"),
                    dtype="float64",
                )
            }
        )


def graph():
    return Graph(
        "toy",
        (toy.SOURCE,),
        (
            toy.CREATE,
            replace(toy.CREATE, id="recipient"),
            Node(
                "train",
                "artifact.train@1",
                population="survey",
                inputs=(Slice("person", ("income",)),),
                artifact_outputs=(ArtifactOutput("model", MODEL),),
            ),
            Node(
                "apply",
                "artifact.apply@1",
                population="recipient",
                inputs=(Slice("person", ("age",)),),
                outputs=(Owned("person", "predicted", "float64"),),
                artifact_inputs=(ArtifactInput("fitted", "train", "model", MODEL),),
            ),
        ),
    )


def registry(*, producer=Numeric.BITWISE, consumer=Numeric.BITWISE):
    reg = toy.toy_registry()
    train = Train()
    apply = Apply()
    for obj, numeric in ((train, producer), (apply, consumer)):
        obj.capabilities = Capabilities(
            Determinism.DETERMINISTIC,
            numeric=numeric,
            tolerance=Tolerance(rtol=1e-5)
            if numeric is Numeric.TOLERANCE_BOUND
            else None,
        )
        reg.register(obj)
    return reg, train, apply


def test_cross_population_cold_warm_and_roundtrip(tmp_path):
    original = graph()
    compiled = compile_graph(original)
    assert "train" in compiled.predecessors["apply"]
    assert graph_from_json(graph_to_json(original)) == original
    sources = toy.toy_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    reg, train, apply = registry()
    cold = run_graph(compiled, sources=sources, store=store, kernels=reg)
    warm = run_graph(compiled, sources=sources, store=store, kernels=reg)
    assert train.calls == apply.calls == 1
    assert cold.key == warm.key
    assert all(node.hit for node in warm.nodes.values())
    assert "model" in describe(compiled, "apply", warm)
    restored = RunManifest.from_json(warm.to_json())
    assert restored.key == warm.key
    assert restored.to_json() == warm.to_json()
    assert json.loads(warm.to_json())["schema_version"] == 3


@pytest.mark.parametrize(
    "change",
    [
        {"producer": "missing"},
        {"artifact": "debug"},
        {"type": ArtifactType("other", 1)},
        {"type": ArtifactType("test.scalar-model", 2)},
        {"producer": "apply"},
    ],
)
def test_invalid_edges_rejected(change):
    g = graph()
    app = g.node("apply")
    bad = replace(app, artifact_inputs=(replace(app.artifact_inputs[0], **change),))
    with pytest.raises(GraphError):
        compile_graph(toy.replace_node(g, bad))


def test_duplicate_alias_and_bad_type_rejected():
    app = graph().node("apply")
    with pytest.raises(GraphError):
        replace(app, artifact_inputs=app.artifact_inputs * 2)
    with pytest.raises((GraphError, TypeError)):
        ArtifactType("model", True)
    with pytest.raises((GraphError, TypeError)):
        ArtifactType("model", 0)


def test_artifact_cycle():
    g = graph()
    train = g.node("train")
    app = g.node("apply")
    train = replace(
        train, artifact_inputs=(ArtifactInput("back", "apply", "back", MODEL),)
    )
    app = replace(app, artifact_outputs=(ArtifactOutput("back", MODEL),))
    with pytest.raises(GraphError, match="Cycle"):
        compile_graph(toy.replace_node(g, train, app))


@pytest.mark.parametrize(
    "producer,consumer,allowed",
    [
        (Numeric.BITWISE, Numeric.BITWISE, True),
        (Numeric.BITWISE, Numeric.PLATFORM_BITWISE, True),
        (Numeric.BITWISE, Numeric.TOLERANCE_BOUND, True),
        (Numeric.PLATFORM_BITWISE, Numeric.BITWISE, False),
        (Numeric.PLATFORM_BITWISE, Numeric.PLATFORM_BITWISE, True),
        (Numeric.PLATFORM_BITWISE, Numeric.TOLERANCE_BOUND, False),
        (Numeric.TOLERANCE_BOUND, Numeric.BITWISE, False),
        (Numeric.TOLERANCE_BOUND, Numeric.PLATFORM_BITWISE, False),
        (Numeric.TOLERANCE_BOUND, Numeric.TOLERANCE_BOUND, True),
    ],
)
def test_numeric_contract_table(tmp_path, producer, consumer, allowed):
    reg, train, apply = registry(producer=producer, consumer=consumer)
    args = dict(
        sources=toy.toy_sources(tmp_path),
        store=ContentStore(tmp_path / "store"),
        kernels=reg,
    )
    if not allowed:
        with pytest.raises(
            NodeRejectedError, match="numeric|Numeric|platform|tolerance"
        ):
            run_graph(compile_graph(graph()), **args)
        assert apply.calls == 0
    else:
        cold = run_graph(compile_graph(graph()), **args)
        warm = run_graph(compile_graph(graph()), **args)
        assert cold.key == warm.key and apply.calls == 1


def test_legacy_pinned_keys_and_json():
    baseline = json.loads(
        (Path(__file__).parent / "fixtures/legacy-graph-key-baseline.json").read_text()
    )
    g = graph_from_json(baseline["graph_json"])
    compiled = compile_graph(g)
    keys = {}
    assert graph_to_json(g) == baseline["graph_json"]
    for node_id in compiled.order:
        node = g.node(node_id)
        raw = baseline["kernel_capabilities"][node.kernel]
        cap = Capabilities(
            Determinism(raw["determinism"]),
            numeric=Numeric(raw["numeric"]),
            seed_source=SeedSource(raw["seed_source"]),
            structural=StructuralDelta(raw["structural"]),
            role=KernelRole(raw["role"]),
            consumes_se=raw["consumes_se"],
            dependencies=tuple(raw["dependencies"]),
        )
        keys[node_id] = node_key(
            compiled,
            node_id,
            keys,
            baseline["kernel_implementation_hashes"][node.kernel],
            baseline["source_keys"],
            kernel_capabilities=cap,
        )
    assert keys == baseline["node_keys"]
    assert {n: seed(k) for n, k in keys.items()} == baseline["node_seeds"]


@pytest.mark.parametrize("artifacts", [{}, {"model": bytearray(b"2")}, {"model": "2"}])
def test_missing_or_mutable_declared_output(tmp_path, artifacts):
    reg, train, apply = registry()
    train.run = lambda context: KernelResult(artifacts=artifacts)
    with pytest.raises(NodeRejectedError, match="artifact|bytes"):
        run_graph(
            compile_graph(graph()),
            sources=toy.toy_sources(tmp_path),
            store=ContentStore(tmp_path / "store"),
            kernels=reg,
        )
    assert apply.calls == 0


def test_artifact_inputs_enter_keys_without_recipient_refit(tmp_path):
    original = graph()
    reg, _, _ = registry()
    sources = toy.toy_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    first = run_graph(
        compile_graph(original), sources=sources, store=store, kernels=reg
    )
    changed = toy.replace_node(
        original, replace(original.node("apply"), params={"application": "changed"})
    )
    second = run_graph(
        compile_graph(changed), sources=sources, store=store, kernels=reg
    )
    assert second.nodes["train"].hit and not second.nodes["apply"].hit
    assert first.nodes["train"].key == second.nodes["train"].key
    changed = toy.replace_node(
        original, replace(original.node("train"), params={"training": "changed"})
    )
    third = run_graph(compile_graph(changed), sources=sources, store=store, kernels=reg)
    assert not third.nodes["train"].hit and not third.nodes["apply"].hit
    assert third.nodes["recipient"].hit


def test_tampered_model_refused_even_when_consumer_cached(tmp_path):
    from microcosm.graph import StoreCorruptError

    reg, train, apply = registry()
    sources = toy.toy_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    first = run_graph(compile_graph(graph()), sources=sources, store=store, kernels=reg)
    key = first.nodes["train"].opaque_artifacts["model"]
    (store.object_path(key) / "payload.bin").write_bytes(b"tampered")
    with pytest.raises(StoreCorruptError):
        run_graph(compile_graph(graph()), sources=sources, store=store, kernels=reg)
    assert train.calls == apply.calls == 1


def test_require_preflights_missing_model_and_missing_codec(tmp_path):
    import shutil

    from microcosm.graph import StoreMissError, StoreUnavailableError

    reg, _, _ = registry()
    sources = toy.toy_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    first = run_graph(compile_graph(graph()), sources=sources, store=store, kernels=reg)
    key = first.nodes["train"].opaque_artifacts["model"]
    shutil.rmtree(store.object_path(key))
    fresh, train, apply = registry()
    with pytest.raises(StoreMissError):
        run_graph(
            compile_graph(graph()),
            sources=sources,
            store=store,
            kernels=fresh,
            resume="require",
        )
    assert train.calls == apply.calls == toy.total_calls(fresh) == 0
    with pytest.raises(StoreUnavailableError):
        run_graph(
            compile_graph(graph()),
            sources=sources,
            store=ContentStore(tmp_path / "store", codecs={}),
            kernels=fresh,
            resume="require",
        )
    assert toy.total_calls(fresh) == 0


def test_cached_contract_and_missing_declared_output_refused(tmp_path, monkeypatch):
    from microcosm.graph import StoreCorruptError, StoreMissError

    reg, _, _ = registry()
    sources = toy.toy_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    first = run_graph(compile_graph(graph()), sources=sources, store=store, kernels=reg)
    real = store.load_json
    mode = ["type"]

    def altered(key):
        record = real(key)
        if record.get("node_id") == "train":
            if mode[0] == "type":
                record["typed_artifacts"]["outputs"]["model"]["type"][
                    "schema_version"
                ] = 2
            else:
                record["opaque"] = [
                    item for item in record["opaque"] if item["name"] != "model"
                ]
        return record

    monkeypatch.setattr(store, "load_json", altered)
    with pytest.raises(StoreCorruptError, match="contracts"):
        run_graph(
            compile_graph(graph()),
            sources=sources,
            store=store,
            kernels=reg,
            resume="require",
        )
    mode[0] = "missing"
    with pytest.raises(StoreMissError):
        run_graph(
            compile_graph(graph()),
            sources=sources,
            store=store,
            kernels=reg,
            resume="require",
        )
    assert first.nodes["train"].typed_artifacts


class ReportScope(KernelBase):
    ref = "artifact.scope@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE)

    def run(self, context):
        scope = context.numerics[("person", "predicted")]
        return KernelResult(
            receipt={
                "outcome": "pass",
                "evidence": {
                    "numeric": scope.numeric.value,
                    "platform": scope.platform,
                },
            }
        )


def test_model_scope_reaches_downstream_gate_on_cold_warm(tmp_path):
    reg, _, _ = registry(
        producer=Numeric.PLATFORM_BITWISE, consumer=Numeric.PLATFORM_BITWISE
    )
    reg.register(ReportScope())
    g = graph()
    gate = Node(
        "scope",
        "artifact.scope@1",
        population="recipient",
        inputs=(Slice("person", ("predicted",)),),
    )
    g = replace(g, nodes=(*g.nodes, gate))
    sources = toy.toy_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    first = run_graph(compile_graph(g), sources=sources, store=store, kernels=reg)
    second = run_graph(compile_graph(g), sources=sources, store=store, kernels=reg)
    for result in (first, second):
        evidence = result.nodes["scope"].receipt["evidence"]
        assert evidence["numeric"] == "platform_bitwise" and evidence["platform"]
    assert second.nodes["scope"].hit


def test_mixed_numeric_inputs_and_combined_scope_refused(tmp_path):
    from microcosm.graph import NumericScope
    from microcosm.graph.artifact_edges import require_compatible_scope

    with pytest.raises(NodeRejectedError, match="platform and tolerance"):
        require_compatible_scope(
            NumericScope(Numeric.TOLERANCE_BOUND, Tolerance(rtol=1e-5), "arm64/test"),
            Capabilities(
                Determinism.DETERMINISTIC,
                numeric=Numeric.TOLERANCE_BOUND,
                tolerance=Tolerance(rtol=1e-5),
            ),
        )
    reg, train, _ = registry(
        producer=Numeric.PLATFORM_BITWISE, consumer=Numeric.PLATFORM_BITWISE
    )
    other = Train()
    other.ref = "artifact.other@1"
    other.capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.TOLERANCE_BOUND,
        tolerance=Tolerance(rtol=1e-5),
    )
    reg.register(other)
    g = graph()
    second = replace(g.node("train"), id="other", kernel=other.ref)
    app = replace(
        g.node("apply"),
        artifact_inputs=(
            *g.node("apply").artifact_inputs,
            ArtifactInput("second", "other", "model", MODEL),
        ),
    )
    g = toy.replace_node(replace(g, nodes=(*g.nodes, second)), app)
    with pytest.raises(NodeRejectedError, match="tolerance"):
        run_graph(
            compile_graph(g),
            sources=toy.toy_sources(tmp_path),
            store=ContentStore(tmp_path / "store"),
            kernels=reg,
        )
    assert train.calls == 0


class EvidenceGate(KernelBase):
    ref = "artifact.evidence@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE)

    def run(self, context):
        return KernelResult(
            artifacts={"evidence": b"fail"},
            receipt={"outcome": "fail", "evidence": {"reason": "synthetic failure"}},
        )


class EvidenceRelease(KernelBase):
    ref = "artifact.release@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, role=KernelRole.RELEASE)

    def run(self, context):
        table = context.tables["release"]
        assert context.artifacts["gate"].payload == b"fail"
        return KernelResult(
            columns={
                ("release", "tier"): pd.Series(
                    ["evidence"],
                    index=pd.Index(table.release_id, name="release_id"),
                    dtype="string",
                )
            },
            receipt={"outcome": "fail", "tier": "evidence"},
        )


def test_artifact_only_gate_ancestry_and_manifest_tamper(tmp_path):
    evidence_type = ArtifactType("test.evidence", 1)
    g = Graph(
        "toy",
        (toy.SOURCE,),
        (
            toy.CREATE,
            Node(
                "gate",
                "artifact.evidence@1",
                population="survey",
                artifact_outputs=(ArtifactOutput("evidence", evidence_type),),
            ),
            Node(
                "release",
                "artifact.release@1",
                population="survey",
                outputs=(Owned("release", "tier", "string"),),
                params={"requires_decisions": ()},
                artifact_inputs=(
                    ArtifactInput("gate", "gate", "evidence", evidence_type),
                ),
            ),
        ),
    )
    reg = toy.toy_registry()
    reg.register(EvidenceGate())
    reg.register(EvidenceRelease())
    sources = toy.toy_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(compile_graph(g), sources=sources, store=store, kernels=reg)
    assert manifest.tier == "evidence" and manifest.nodes["release"].receipt[
        "gate_ancestry"
    ] == ("gate",)
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert RunManifest.load(path, store).key == manifest.key
    raw = json.loads(manifest.to_json())
    raw["nodes"]["release"]["typed_artifacts"]["inputs"]["gate"]["type"][
        "schema_version"
    ] = 3
    with pytest.raises(ValueError, match="producer|artifact|content"):
        RunManifest.from_json(json.dumps(raw))
    raw = json.loads(manifest.to_json())
    raw["schema_version"] = 2
    with pytest.raises(ValueError, match="schema 3"):
        RunManifest.from_json(json.dumps(raw))


def test_shared_typed_ancestry_is_memoized():
    """A layered shared DAG must not enumerate exponentially many paths."""
    import hashlib

    from microcosm.graph import NodeReceipt
    from microcosm.graph.artifact_edges import descriptor

    nodes = {}
    prior = []
    caps = Capabilities(Determinism.DETERMINISTIC)
    for layer in range(30):
        current = []
        for column in range(2):
            node_id = f"layer{layer}_{column}"
            key = hashlib.sha256(node_id.encode()).hexdigest()
            output = descriptor(
                producer=node_id,
                artifact="model",
                type_=MODEL,
                producer_key=key,
                capabilities=caps,
            )
            nodes[node_id] = NodeReceipt(
                key=key,
                hit=False,
                seed=0,
                kernel_ref="toy@1",
                kernel_impl_hash="a" * 64,
                capabilities=caps,
                opaque_artifacts={"model": output["key"]},
                typed_artifacts={
                    "inputs": {
                        parent: nodes[parent].typed_artifacts["outputs"]["model"]
                        for parent in prior
                    },
                    "outputs": {"model": output},
                },
            )
            current.append(node_id)
        prior = current
    assert len(RunManifest(country="toy", nodes=nodes).nodes) == 60
