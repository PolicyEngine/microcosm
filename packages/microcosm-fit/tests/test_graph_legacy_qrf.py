"""Graph-native target chains preserve the existing weighted QRF protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import QRF
from microcosm.fit.graph_legacy_qrf import (
    LegacyQRFApplyKernel,
    LegacyQRFTrainKernel,
    legacy_qrf_apply_nodes,
    legacy_qrf_train_nodes,
    read_raw_target,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
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
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    platform_fingerprint,
    run_graph,
)


@pytest.fixture(autouse=True)
def serial_workers(monkeypatch):
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")


def model_frame(values, *, entity="tax_unit", weights=None):
    values = values.copy()
    ids = np.arange(len(values), dtype=np.int64) + 10
    values.insert(0, f"{entity}_id", ids)
    if entity == "person":
        values.insert(1, "person_model_unit_id", ids)
        schema = EntitySchema(group_entities=("model_unit",))
        tables = {"person": values, "model_unit": pd.DataFrame({"model_unit_id": ids})}
    else:
        schema = EntitySchema(group_entities=(entity,))
        tables = {
            entity: values,
            "person": pd.DataFrame({"person_id": ids, f"person_{entity}_id": ids}),
        }
    return Frame(
        tables,
        schema,
        {
            entity: Weights(
                np.ones(len(values)) if weights is None else weights, WeightKind.DESIGN
            )
        },
    )


def fixture_frames(*, index=None, entity="tax_unit"):
    x = np.arange(42, dtype=float) % 7
    donor = pd.DataFrame(
        {"x": x, "y": 1.0 + x + np.arange(42) % 3, "z": 3.0 + x * x + np.arange(42) % 5}
    )
    recipient = pd.DataFrame({"x": [1.0, 5.0, 2.0, 6.0]}, index=index)
    return model_frame(
        donor, entity=entity, weights=1.0 + np.arange(42) % 4
    ), model_frame(recipient, entity=entity)


class SourceKernel(KernelBase):
    ref = "test.legacy.source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources[context.node.sources[0]])
        )


def source_node(name, frame):
    return Node(
        name,
        SourceKernel.ref,
        sources=(name,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(entity, column, str(table[column].dtype))
            for entity in frame.entities
            for table in (frame.table(entity),)
            for column in table
            if not column.endswith("_id")
        ),
    )


def source_path(store, frame):
    digest = hashlib.sha256()
    for entity in frame.entities:
        digest.update(entity.encode())
        digest.update(frame.table(entity).to_json(orient="table").encode())
    for entity in frame.weighted_entities:
        digest.update(frame.weights_for(entity).values.tobytes())
    return store.put_frame(digest.hexdigest(), frame)


def setup_graph(
    tmp_path, donor, recipient, *, seed=31, draw_seed=None, predictors=("x",)
):
    store = ContentStore(tmp_path)
    entity = next(entity for entity in donor.weighted_entities)
    fits = legacy_qrf_train_nodes(
        "fit",
        population="donor",
        entity=entity,
        predictors=predictors,
        targets=("y", "z"),
        seed=seed,
        n_estimators=2,
        phase="early_gap_fill",
    )
    applies = legacy_qrf_apply_nodes(
        "apply",
        population="recipient",
        fit_nodes=fits,
        seed=seed if draw_seed is None else draw_seed,
        phase="early_gap_fill",
    )
    graph = Graph(
        country="us",
        sources=(
            SourceRef("donor", "frame-store"),
            SourceRef("recipient", "frame-store"),
        ),
        nodes=(
            source_node("donor", donor),
            source_node("recipient", recipient),
            *fits,
            *applies,
        ),
    )
    kernels = KernelRegistry()
    for kernel in (SourceKernel(), LegacyQRFTrainKernel(), LegacyQRFApplyKernel()):
        kernels.register(kernel)
    sources = {
        "donor": source_path(store, donor),
        "recipient": source_path(store, recipient),
    }
    return compile_graph(graph), store, kernels, sources, fits, applies


@pytest.mark.parametrize("entity", ("person", "tax_unit"))
@pytest.mark.parametrize(
    "index",
    (
        pd.Index([11, 19, 31, 97], name="original_row"),
        pd.Index(["alpha", "beta", "gamma", "delta"], name="original_row"),
    ),
)
def test_exact_legacy_parity_and_fresh_store_restart(tmp_path, entity, index):
    donor, recipient = fixture_frames(index=index, entity=entity)
    compiled, store, kernels, sources, fits, applies = setup_graph(
        tmp_path, donor, recipient
    )
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    model = QRF(seed=31, n_estimators=2)
    state = model.start_chain(donor, ["x"], ["y", "z"], weights="design")
    raw = pd.DataFrame(index=recipient.table(entity).index)
    for fit, apply in zip(fits, applies, strict=True):
        expected = model.fit_draw_next(
            donor, recipient, raw, state=state, weights="design"
        )
        actual = read_raw_target(
            store.load_bytes(manifest.node(apply.id).opaque_artifacts["raw_draw"]),
            target=expected.target,
            index=raw.index,
        )
        np.testing.assert_array_equal(actual, expected.raw_draw)
        state = expected.state
        raw[expected.target] = expected.raw_draw
        checkpoint = json.loads(
            store.load_bytes(manifest.node(apply.id).opaque_artifacts["apply_state"])
        )
        assert checkpoint["state"] == state.to_dict()
        training = json.loads(
            store.load_bytes(manifest.node(fit.id).opaque_artifacts["training_state"])
        )
        assert "recipient_index" not in training["state"]
        assert "draw_rng_state" not in training["state"]
    # Reopen the store rather than reusing any in-memory populations/models.
    again = run_graph(
        compiled,
        sources=sources,
        store=ContentStore(tmp_path),
        kernels=kernels,
        resume="require",
    )
    assert again.key == manifest.key
    assert all(node.hit for node in again.nodes.values())


@pytest.mark.parametrize("entity", ("person", "tax_unit"))
def test_unsupported_multiindex_refuses_without_silent_index_conversion(
    tmp_path, entity
):
    index = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=("arm", "row")
    )
    donor, recipient = fixture_frames(index=index, entity=entity)
    with pytest.raises(TypeError, match="ContentStore does not support MultiIndex"):
        setup_graph(tmp_path, donor, recipient)
    assert recipient.table(entity).index.equals(index)


@pytest.mark.parametrize(
    "mutation", ("membership", "weights", "source_pin", "draw_seed")
)
def test_recipient_changes_reuse_every_fitted_target(tmp_path, monkeypatch, mutation):
    import microcosm.fit.graph_legacy_train as train_module

    donor, recipient = fixture_frames()
    args = setup_graph(tmp_path, donor, recipient)
    compiled, store, kernels, sources, fits, applies = args
    calls = []
    original = train_module.fit_target

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "fit_target", spy)
    first = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    assert len(calls) == 2
    training_only = Graph(
        country="us",
        sources=(SourceRef("donor", "frame-store"),),
        nodes=(source_node("donor", donor), *fits),
    )
    pinned_fits = run_graph(
        compile_graph(training_only),
        sources={"donor": sources["donor"]},
        store=ContentStore(tmp_path),
        kernels=kernels,
        resume="require",
    )
    assert all(node.hit for node in pinned_fits.nodes.values())
    assert all(
        pinned_fits.node(node.id).key == first.node(node.id).key for node in fits
    )
    if mutation == "membership":
        recipient = model_frame(
            pd.DataFrame({"x": [1.0, 3.0, 6.0]}, index=[101, 102, 103])
        )
    elif mutation == "weights":
        recipient = model_frame(
            recipient.table("tax_unit")[["x"]], weights=np.full(4, 9.0)
        )
    elif mutation == "source_pin":
        recipient.table("tax_unit")["unconsumed_source_receipt"] = 7
    second_args = setup_graph(
        tmp_path, donor, recipient, draw_seed=67 if mutation == "draw_seed" else None
    )
    second = run_graph(
        second_args[0], sources=second_args[3], store=store, kernels=kernels
    )
    assert len(calls) == 2
    for node in fits:
        assert second.node(node.id).key == first.node(node.id).key
        assert (
            second.node(node.id).opaque_artifacts["model"]
            == first.node(node.id).opaque_artifacts["model"]
        )
        assert second.node(node.id).hit
    assert all(not second.node(node.id).hit for node in applies)
    required = run_graph(
        second_args[0],
        sources=second_args[3],
        store=ContentStore(tmp_path),
        kernels=kernels,
        resume="require",
    )
    assert all(node.hit for node in required.nodes.values())


@pytest.mark.parametrize(
    "mutation",
    ("predictor", "observed_prior", "weights", "pattern_seed", "optional_pattern"),
)
def test_real_donor_or_fit_policy_change_invalidates_training(tmp_path, mutation):
    donor, recipient = fixture_frames()
    compiled, store, kernels, sources, fits, _ = setup_graph(tmp_path, donor, recipient)
    first = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    seed = 31
    predictors = ("x",)
    if mutation == "predictor":
        donor.table("tax_unit").loc[0, "x"] += 20
    elif mutation == "observed_prior":
        donor.table("tax_unit").loc[0, "y"] += 20
    elif mutation == "weights":
        donor = model_frame(
            donor.table("tax_unit")[["x", "y", "z"]], weights=np.full(42, 2.0)
        )
    elif mutation == "pattern_seed":
        seed = 71
    elif mutation == "optional_pattern":
        donor.table("tax_unit")["optional"] = np.arange(42) % 2
        recipient.table("tax_unit")["optional"] = np.arange(4) % 2
        predictors = ("x", "optional")
    second_args = setup_graph(
        tmp_path, donor, recipient, seed=seed, predictors=predictors
    )
    second = run_graph(
        second_args[0], sources=second_args[3], store=store, kernels=kernels
    )
    for node in fits:
        assert second.node(node.id).key != first.node(node.id).key
        assert not second.node(node.id).hit


def apply_context(manifest, store, node, frame):
    entity = node.inputs[0].entity
    scope = NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint())
    artifacts = {}
    for item in node.artifact_inputs:
        producer = manifest.node(item.producer)
        key = producer.opaque_artifacts[item.artifact]
        artifacts[item.name] = ArtifactValue(
            store.load_bytes(key), item.type, key, producer.key, scope
        )
    return KernelContext(
        node=node,
        tables={entity: frame.table(entity)},
        weights={entity: frame.resolve_weights(entity)},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts=artifacts,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "raw_bytes",
        "raw_target",
        "row_order",
        "draw_seed",
        "model_history",
        "missing_prior",
    ),
)
def test_apply_refuses_changed_raw_prefix_or_chain_identity(tmp_path, mutation):
    from microcosm.fit import _graph_legacy_qrf as codec

    donor, recipient = fixture_frames(index=pd.Index([11, 19, 31, 97], name="row"))
    compiled, store, kernels, sources, _, applies = setup_graph(
        tmp_path, donor, recipient
    )
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    context = apply_context(manifest, store, applies[1], recipient)
    artifacts = dict(context.artifacts)
    if mutation in {"raw_bytes", "raw_target"}:
        raw = read_raw_target(
            artifacts["prior_000"].payload,
            target="y",
            index=recipient.table("tax_unit").index,
        ).copy()
        raw[0] += 1
        payload = codec.encode_raw_target(
            raw,
            target="z" if mutation == "raw_target" else "y",
            index=recipient.table("tax_unit").index,
        )
        artifacts["prior_000"] = replace(artifacts["prior_000"], payload=payload)
    elif mutation == "row_order":
        context = replace(
            context, tables={"tax_unit": recipient.table("tax_unit").iloc[::-1]}
        )
    elif mutation == "draw_seed":
        context = replace(context, params={**context.params, "seed": 67})
    elif mutation == "model_history":
        state = codec.decode_json(artifacts["apply_state"].payload)
        state["models"][0]["sha256"] = "f" * 64
        artifacts["apply_state"] = replace(
            artifacts["apply_state"], payload=codec.encode_json(state)
        )
    elif mutation == "missing_prior":
        del artifacts["prior_000"]
    with pytest.raises(ValueError):
        LegacyQRFApplyKernel().run(replace(context, artifacts=artifacts))


@pytest.mark.parametrize(
    "mutation",
    ("model_bytes", "model_platform", "model_producer", "envelope_implementation"),
)
def test_graph_model_validation_precedes_unpickle(tmp_path, monkeypatch, mutation):
    from microcosm.fit import _graph_legacy_qrf as codec
    from microcosm.fit import qrf_target

    donor, recipient = fixture_frames()
    compiled, store, kernels, sources, _, applies = setup_graph(
        tmp_path, donor, recipient
    )
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    context = apply_context(manifest, store, applies[0], recipient)
    artifacts = dict(context.artifacts)
    value = artifacts["model"]
    if mutation == "model_bytes":
        artifacts["model"] = replace(value, payload=value.payload + b"bad")
    elif mutation == "model_platform":
        artifacts["model"] = replace(
            value,
            numerics=NumericScope(Numeric.PLATFORM_BITWISE, platform="other-platform"),
        )
    elif mutation == "model_producer":
        artifacts["model"] = replace(value, producer_key="f" * 64)
    else:
        # Update the trusted sibling digest to reach the inner envelope check.
        body = value.payload[len(qrf_target._MAGIC) :]
        length = int.from_bytes(body[:8], "big")
        metadata = json.loads(body[8 : 8 + length])
        metadata["implementation"]["platform"] = "other-platform"
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
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
        pytest.fail("unpickle ran before envelope validation")

    monkeypatch.setattr(qrf_target.pickle, "loads", forbidden)
    with pytest.raises(ValueError):
        LegacyQRFApplyKernel().run(replace(context, artifacts=artifacts))


def test_zero_target_keeps_legacy_no_draw_rng_transition(tmp_path):
    donor, recipient = fixture_frames()
    donor.table("tax_unit")["y"] = 0.0
    donor.table("tax_unit")["z"] = 2.5
    compiled, store, kernels, sources, _, applies = setup_graph(
        tmp_path, donor, recipient
    )
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    first = json.loads(
        store.load_bytes(manifest.node(applies[0].id).opaque_artifacts["apply_state"])
    )
    state = QRF(seed=31, n_estimators=2).start_chain(
        donor, ["x"], ["y", "z"], weights="design"
    )
    assert first["state"]["draw_rng_state"] == state.to_dict()["draw_rng_state"]
    raw = read_raw_target(
        store.load_bytes(manifest.node(applies[0].id).opaque_artifacts["raw_draw"]),
        target="y",
        index=recipient.table("tax_unit").index,
    )
    np.testing.assert_array_equal(raw, np.zeros(4))


def test_application_seed_controls_the_actual_second_seedsequence_stream(tmp_path):
    from microcosm.fit import qrf

    donor, recipient = fixture_frames()
    compiled, store, kernels, sources, _, applies = setup_graph(
        tmp_path, donor, recipient, draw_seed=67
    )
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    model = QRF(seed=31, n_estimators=2)
    initial = model.start_chain(donor, ["x"], ["y", "z"], weights="design")
    _, draw_seed = np.random.SeedSequence(67).spawn(2)
    state = replace(
        initial,
        draw_rng_state_json=qrf._rng_state_json(np.random.default_rng(draw_seed)),
    )
    raw = pd.DataFrame(index=recipient.table("tax_unit").index)
    for apply in applies:
        expected = model.fit_draw_next(
            donor, recipient, raw, state=state, weights="design"
        )
        actual = read_raw_target(
            store.load_bytes(manifest.node(apply.id).opaque_artifacts["raw_draw"]),
            target=expected.target,
            index=raw.index,
        )
        np.testing.assert_array_equal(actual, expected.raw_draw)
        packet = json.loads(
            store.load_bytes(manifest.node(apply.id).opaque_artifacts["apply_state"])
        )
        assert packet["state"] == expected.state.to_dict()
        state = expected.state
        raw[expected.target] = expected.raw_draw


@pytest.mark.parametrize("case", ("positive", "hurdle", "zero_constant"))
def test_slice_adapter_retains_pre_refactor_raw_and_state_bytes(tmp_path, case):
    from microcosm.fit import _graph_legacy_qrf as codec

    golden = json.loads(
        (Path(__file__).parent / "golden/legacy-slice-apply-v1.json").read_text()
    )
    donor, recipient = fixture_frames(index=pd.Index([11, 19, 31, 97], name="row"))
    if case == "hurdle":
        donor.table("tax_unit").loc[:20, "y"] = 0.0
    elif case == "zero_constant":
        donor.table("tax_unit")["y"] = 0.0
        donor.table("tax_unit")["z"] = 2.5
    compiled, store, kernels, sources, fits, applies = setup_graph(
        tmp_path, donor, recipient
    )
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    for fit, apply, captured in zip(fits, applies, golden["cases"][case], strict=True):
        assert store.load_bytes(
            manifest.node(apply.id).opaque_artifacts["raw_draw"]
        ) == bytes.fromhex(captured["raw_draw"])
        # Forest identities are platform/version-bound. Only those independently
        # supplied input records vary; every state/RNG/raw-history byte is frozen.
        expected = codec.decode_json(bytes.fromhex(captured["apply_state"]))
        inputs = codec.decode_json(
            store.load_bytes(manifest.node(fit.id).opaque_artifacts["training_state"])
        )
        assert [item["target"] for item in expected["models"]] == [
            item["target"] for item in inputs["models"]
        ]
        expected["models"] = inputs["models"]
        assert store.load_bytes(
            manifest.node(apply.id).opaque_artifacts["apply_state"]
        ) == codec.encode_json(expected)
