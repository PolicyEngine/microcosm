"""Derived model rows cross typed edges without a new population or index edit."""

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import QRF
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit.graph_legacy_apply_matrix import (
    LegacyQRFApplyMatrixKernel,
    decode_matrix_apply_state,
)
from microcosm.fit.graph_legacy_qrf import (
    LegacyQRFTrainKernel,
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.fit.model_input import (
    RECIPIENT_MATRIX_TYPE,
    decode_recipient_matrix,
    encode_recipient_matrix,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    NumericScope,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    platform_fingerprint,
    run_graph,
)


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")


def population(values, ids):
    table = values.copy()
    table.insert(0, "household_id", ids)
    return Frame(
        {
            "household": table,
            "person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
    )


def frames():
    x = np.arange(42, dtype=float) % 7
    donor = population(
        pd.DataFrame(
            {
                "x": x,
                "indicator": np.arange(42, dtype=float) % 2,
                "y": 1 + x,
                "z": 2 + x * x,
            }
        ),
        np.arange(42, dtype=np.int64) + 10,
    )
    recipient = population(
        pd.DataFrame({"x": [1.0, 5.0, 2.0], "indicator": [0.0, 1.0, 0.0]}),
        np.array([11, 97, 301], dtype=np.int64),
    )
    return donor, recipient


class Source(KernelBase):
    ref = "test.matrix.source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources[context.node.sources[0]])
        )


class Prepare(KernelBase):
    ref = "test.matrix.prepare@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
    )

    def run(self, context):
        table = context.tables["household"]
        ids = table.household_id.to_numpy()
        features = table[["x", "indicator"]].copy()
        features.index = pd.Index(ids)
        return KernelResult(
            artifacts={
                "matrix": encode_recipient_matrix(
                    features, entity="household", entity_ids=ids
                )
            }
        )


def setup(tmp_path, donor, recipient):
    store = ContentStore(tmp_path)
    sources, source_nodes = {}, []
    for name, frame in (("donor", donor), ("recipient", recipient)):
        table = frame.table("household")
        key = hashlib.sha256(
            table.to_json(orient="table").encode()
            + frame.resolve_weights("household").values.tobytes()
        ).hexdigest()
        sources[name] = store.put_frame(key, frame)
        source_nodes.append(
            Node(
                name,
                Source.ref,
                sources=(name,),
                structural=StructuralDelta.CREATE,
                outputs=tuple(
                    Owned("household", col, "float64")
                    for col in table
                    if col != "household_id"
                ),
            )
        )
    fits = legacy_qrf_train_nodes(
        "fit",
        population="donor",
        entity="household",
        predictors=("x", "indicator"),
        targets=("y", "z"),
        seed=31,
        n_estimators=2,
        phase="early_gap_fill",
    )
    matrix = Node(
        "matrix",
        Prepare.ref,
        population="recipient",
        inputs=(Slice("household", ("x", "indicator")),),
        artifact_outputs=(ArtifactOutput("matrix", RECIPIENT_MATRIX_TYPE),),
    )
    applies = legacy_qrf_apply_matrix_nodes(
        "apply",
        population="recipient",
        fit_nodes=fits,
        matrix_producer="matrix",
        matrix_artifact="matrix",
        seed=31,
        phase="early_gap_fill",
    )
    graph = Graph(
        "us",
        (SourceRef("donor", "frame-store"), SourceRef("recipient", "frame-store")),
        (*source_nodes, matrix, *fits, *applies),
    )
    kernels = KernelRegistry()
    for kernel in (
        Source(),
        Prepare(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        kernels.register(kernel)
    return compile_graph(graph), store, kernels, sources, fits, applies


def context_for(manifest, store, node):
    artifacts = {}
    for edge in node.artifact_inputs:
        producer = manifest.node(edge.producer)
        key = producer.opaque_artifacts[edge.artifact]
        artifacts[edge.name] = ArtifactValue(
            store.load_bytes(key),
            edge.type,
            key,
            producer.key,
            NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
        )
    return KernelContext(
        node,
        {},
        {},
        pd.Series([], dtype=str),
        node.params,
        np.random.default_rng(0),
        artifacts=artifacts,
    )


def test_real_matrix_chain_matches_legacy_and_replays_without_population_index_change(
    tmp_path,
):
    donor, recipient = frames()
    compiled, store, kernels, sources, fits, applies = setup(tmp_path, donor, recipient)
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    matrix = decode_recipient_matrix(
        store.load_bytes(manifest.node("matrix").opaque_artifacts["matrix"])
    )
    assert isinstance(
        manifest.population("recipient").table("household").index, pd.RangeIndex
    )
    assert type(matrix.features.index) is pd.Index
    model = QRF(seed=31, n_estimators=2)
    state = model.start_chain(donor, ["x", "indicator"], ["y", "z"], weights="design")
    raw = pd.DataFrame(index=matrix.features.index)
    for node in applies:
        expected = model.fit_draw_next(
            donor, matrix.features, raw, state=state, weights="design"
        )
        payload = store.load_bytes(manifest.node(node.id).opaque_artifacts["raw_draw"])
        np.testing.assert_array_equal(
            codec.read_raw_target(payload, target=expected.target, index=raw.index),
            expected.raw_draw,
        )
        with pytest.raises(ValueError):
            codec.read_raw_target(
                payload,
                target=expected.target,
                index=recipient.table("household").index,
            )
        packet = decode_matrix_apply_state(
            store.load_bytes(manifest.node(node.id).opaque_artifacts["apply_state"])
        )
        assert packet["application"]["state"] == expected.state.to_dict()
        assert packet["matrix_producer_key"] == manifest.node("matrix").key
        state = expected.state
        raw[expected.target] = expected.raw_draw
    warm = run_graph(
        compiled,
        sources=sources,
        store=ContentStore(tmp_path),
        kernels=kernels,
        resume="require",
    )
    assert warm.key == manifest.key and all(node.hit for node in warm.nodes.values())
    for node in applies:
        for name in ("raw_draw", "apply_state"):
            assert store.load_bytes(
                warm.node(node.id).opaque_artifacts[name]
            ) == store.load_bytes(manifest.node(node.id).opaque_artifacts[name])


@pytest.mark.parametrize(
    "mutation", ("features", "membership", "weights", "source_pin")
)
def test_recipient_mutations_reuse_fits_but_rekey_matrix_chain(
    tmp_path, monkeypatch, mutation
):
    import microcosm.fit.graph_legacy_train as train

    donor, recipient = frames()
    compiled, store, kernels, sources, fits, applies = setup(tmp_path, donor, recipient)
    calls = []
    original = train.fit_target

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(train, "fit_target", spy)
    first = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    assert len(calls) == 2
    if mutation == "features":
        recipient.table("household").loc[0, "x"] += 1
    elif mutation == "membership":
        recipient = population(
            pd.DataFrame({"x": [1.0, 2.0], "indicator": [0.0, 1.0]}),
            np.array([20, 70], dtype=np.int64),
        )
    elif mutation == "weights":
        recipient = Frame(
            {entity: recipient.table(entity) for entity in recipient.entities},
            recipient.schema,
            {"household": Weights(np.full(3, 7.0), WeightKind.DESIGN)},
        )
    else:
        recipient.table("household")["unconsumed_pin"] = 7.0
    second_args = setup(tmp_path, donor, recipient)
    second = run_graph(
        second_args[0], sources=second_args[3], store=store, kernels=kernels
    )
    assert len(calls) == 2
    for node in fits:
        assert second.node(node.id).hit
        assert second.node(node.id).key == first.node(node.id).key
        assert (
            second.node(node.id).opaque_artifacts
            == first.node(node.id).opaque_artifacts
        )
    assert second.node("matrix").key != first.node("matrix").key
    for node in applies:
        assert not second.node(node.id).hit
        assert second.node(node.id).key != first.node(node.id).key
        assert (
            second.node(node.id).opaque_artifacts["apply_state"]
            != first.node(node.id).opaque_artifacts["apply_state"]
        )
    # A weights/pin-only change keeps actual prepared values and raw draws equal,
    # while the state still binds the new real producer ancestry.
    if mutation in {"weights", "source_pin"}:
        assert store.load_bytes(
            second.node("matrix").opaque_artifacts["matrix"]
        ) == store.load_bytes(first.node("matrix").opaque_artifacts["matrix"])
        assert all(
            store.load_bytes(second.node(n.id).opaque_artifacts["raw_draw"])
            == store.load_bytes(first.node(n.id).opaque_artifacts["raw_draw"])
            for n in applies
        )
    required = run_graph(
        second_args[0],
        sources=second_args[3],
        store=ContentStore(tmp_path),
        kernels=kernels,
        resume="require",
    )
    assert all(node.hit for node in required.nodes.values())


def test_declaration_pins_one_matrix_across_chain(tmp_path):
    donor, recipient = frames()
    _, _, _, _, fits, applies = setup(tmp_path, donor, recipient)
    for node in applies:
        assert node.population == "recipient"
        assert not node.inputs and not node.outputs and not node.sources
        assert node.params["phase"] == "early_gap_fill"
        edge = [edge for edge in node.artifact_inputs if edge.name == "matrix"]
        assert edge == [
            ArtifactInput("matrix", "matrix", "matrix", RECIPIENT_MATRIX_TYPE)
        ]
    with pytest.raises(ValueError, match="complete target chain"):
        legacy_qrf_apply_matrix_nodes(
            "bad",
            population="recipient",
            fit_nodes=fits[:1],
            matrix_producer="matrix",
            seed=31,
            phase="early_gap_fill",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "values",
        "rows",
        "predictor_order",
        "predictor_name",
        "entity",
        "producer",
        "state_producer",
        "state_sha",
        "raw_prior",
        "model_history",
        "draw_seed",
        "missing_prior",
        "state_type",
        "truncated_matrix",
        "matrix_platform",
        "phase",
    ),
)
def test_matrix_route_refuses_changed_chain_bindings(tmp_path, mutation):
    donor, recipient = frames()
    compiled, store, kernels, sources, _, applies = setup(tmp_path, donor, recipient)
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    context = context_for(manifest, store, applies[1])
    artifacts = dict(context.artifacts)
    value = artifacts["matrix"]
    decoded = decode_recipient_matrix(value.payload)
    if mutation in {"values", "rows", "predictor_order", "predictor_name", "entity"}:
        features = decoded.features.copy()
        entity = decoded.entity
        if mutation == "values":
            features.iloc[0, 0] += 1
        elif mutation == "rows":
            features = features.iloc[::-1]
        elif mutation == "predictor_order":
            features = features[["indicator", "x"]]
        elif mutation == "predictor_name":
            features = features.rename(columns={"x": "other"})
        else:
            entity = "tax_unit"
        payload = encode_recipient_matrix(
            features, entity=entity, entity_ids=features.index.to_numpy()
        )
        artifacts["matrix"] = replace(value, payload=payload)
    elif mutation == "producer":
        artifacts["matrix"] = replace(value, producer_key="f" * 64)
    elif mutation in {"state_producer", "state_sha", "model_history"}:
        packet = decode_matrix_apply_state(artifacts["apply_state"].payload)
        if mutation == "model_history":
            packet["application"]["models"][0]["sha256"] = "f" * 64
        else:
            packet[
                "matrix_producer_key"
                if mutation == "state_producer"
                else "matrix_sha256"
            ] = "f" * 64
        artifacts["apply_state"] = replace(
            artifacts["apply_state"], payload=codec.encode_json(packet)
        )
    elif mutation == "raw_prior":
        raw = codec.read_raw_target(
            artifacts["prior_000"].payload, target="y", index=decoded.features.index
        ).copy()
        raw[0] += 1
        artifacts["prior_000"] = replace(
            artifacts["prior_000"],
            payload=codec.encode_raw_target(
                raw, target="y", index=decoded.features.index
            ),
        )
    elif mutation == "draw_seed":
        context = replace(context, params={**context.params, "seed": 67})
    elif mutation == "missing_prior":
        del artifacts["prior_000"]
    elif mutation == "state_type":
        artifacts["apply_state"] = replace(
            artifacts["apply_state"], type=codec.APPLY_STATE_TYPE
        )
    elif mutation == "truncated_matrix":
        artifacts["matrix"] = replace(value, payload=value.payload[:-1])
    elif mutation == "matrix_platform":
        artifacts["matrix"] = replace(
            value, numerics=NumericScope(Numeric.PLATFORM_BITWISE, platform="foreign")
        )
    else:
        context = replace(context, params={**context.params, "phase": ""})
    with pytest.raises(ValueError):
        LegacyQRFApplyMatrixKernel().run(replace(context, artifacts=artifacts))


@pytest.mark.parametrize(
    "mutation", ("model_bytes", "model_producer", "envelope_platform")
)
def test_matrix_model_refusal_precedes_unpickle(tmp_path, monkeypatch, mutation):
    import json

    from microcosm.fit import qrf_target

    donor, recipient = frames()
    compiled, store, kernels, sources, _, applies = setup(tmp_path, donor, recipient)
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    context = context_for(manifest, store, applies[0])
    artifacts = dict(context.artifacts)
    value = artifacts["model"]
    if mutation == "model_bytes":
        artifacts["model"] = replace(value, payload=value.payload + b"foreign")
    elif mutation == "model_producer":
        artifacts["model"] = replace(value, producer_key="f" * 64)
    else:
        body = value.payload[len(qrf_target._MAGIC) :]
        length = int.from_bytes(body[:8], "big")
        header = json.loads(body[8 : 8 + length])
        header["implementation"]["platform"] = "foreign"
        encoded = codec.encode_json(header)
        payload = (
            qrf_target._MAGIC
            + len(encoded).to_bytes(8, "big")
            + encoded
            + body[8 + length :]
        )
        artifacts["model"] = replace(value, payload=payload)
        packet = codec.decode_json(artifacts["training_state"].payload)
        packet["models"][0]["sha256"] = codec.sha(payload)
        artifacts["training_state"] = replace(
            artifacts["training_state"], payload=codec.encode_json(packet)
        )

    def forbidden(_):
        pytest.fail("unpickle ran before model envelope validation")

    monkeypatch.setattr(qrf_target.pickle, "loads", forbidden)
    with pytest.raises(ValueError):
        LegacyQRFApplyMatrixKernel().run(replace(context, artifacts=artifacts))


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema_version", True),
        ("matrix_sha256", "bad"),
        ("matrix_producer_key", "bad"),
        ("application", {}),
    ),
)
def test_public_matrix_state_parser_refuses_malformed_packets(tmp_path, field, value):
    donor, recipient = frames()
    compiled, store, kernels, sources, _, applies = setup(tmp_path, donor, recipient)
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    packet = decode_matrix_apply_state(
        store.load_bytes(manifest.node(applies[0].id).opaque_artifacts["apply_state"])
    )
    packet[field] = value
    with pytest.raises(ValueError):
        decode_matrix_apply_state(codec.encode_json(packet))


def test_both_kernel_keys_change_when_shared_implementation_changes(
    tmp_path, monkeypatch
):
    from microcosm.fit import _graph_legacy_apply as shared
    from microcosm.fit import model_input
    from microcosm.fit.graph_legacy_apply import LegacyQRFApplyKernel

    original = shared.__file__
    source = tmp_path / "shared.py"
    source.write_text(open(original).read() + "\n# implementation identity probe\n")
    kernels = (LegacyQRFApplyKernel(), LegacyQRFApplyMatrixKernel())
    before = [kernel.implementation_hash() for kernel in kernels]
    monkeypatch.setattr(shared, "__file__", str(source))
    after = [kernel.implementation_hash() for kernel in kernels]
    assert all(left != right for left, right in zip(before, after, strict=True))
    monkeypatch.setattr(shared, "__file__", original)
    matrix_source = tmp_path / "matrix.py"
    matrix_source.write_text(
        open(model_input.__file__).read() + "\n# codec identity probe\n"
    )
    monkeypatch.setattr(model_input, "__file__", str(matrix_source))
    assert kernels[0].implementation_hash() == before[0]
    assert kernels[1].implementation_hash() != before[1]
