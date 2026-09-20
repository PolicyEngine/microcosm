"""Invented real fit/probability graph, cache replay and typed-edge refusals."""

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import categorical as cat
from microcosm.fit import graph_categorical as graph
from microcosm.fit import model_input
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
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
    SeedSource,
    SourceRef,
    StoreCorruptError,
    StructuralDelta,
    compile_graph,
    load_source,
    platform_fingerprint,
    run_graph,
)
from microcosm.graph.keys import opaque_artifact_key

SOURCE_TYPE = ArtifactType("test.categorical_source", 1)
CLASSES = ("survivors", "retirement", "disability", "dependents")
DONOR_SOURCE = b"invented categorical donor projection"
RECIPIENT_SOURCE = b"invented unresolved original recipients"
CONFIG = cat.CategoricalConfig(max_iter=2)


def table(donor=True):
    count = 8 if donor else 3
    result = pd.DataFrame(
        dict(
            person_id=np.arange(count, dtype=np.int64) + (2**53 + (1 if donor else 21)),
            age=np.zeros(count),
            total=np.zeros(count),
        )
    )
    if donor:
        result["category"] = pd.array(CLASSES * 2, dtype="string")
    return result


def matrix(values=None):
    values = table(False) if values is None else values
    features = values[["age", "total"]].copy(deep=True)
    features.index = pd.Index(values.person_id.to_numpy(copy=True), name="person_id")
    return cat.matrix_bytes(features, entity="person")


def fit_node(**changes):
    opts = dict(
        population="donors",
        entity="person",
        predictors=("age", "total"),
        label="category",
        classes=CLASSES,
        seed=101,
        config=CONFIG,
        source_projection=ArtifactInput(
            "source_projection", "donors", "projection", SOURCE_TYPE
        ),
        source_sha256=cat._sha(DONOR_SOURCE),
    )
    opts.update(changes)
    return graph.categorical_fit_node("fit", **opts)


def apply_node(fit=None, **changes):
    opts = dict(
        population="recipients",
        fit_node=fit_node() if fit is None else fit,
        matrix=ArtifactInput(
            "matrix", "recipients", "matrix", model_input.RECIPIENT_MATRIX_TYPE
        ),
        matrix_sha256=cat._sha(matrix()),
        source_projection=ArtifactInput(
            "source_projection", "recipients", "projection", SOURCE_TYPE
        ),
        source_sha256=cat._sha(RECIPIENT_SOURCE),
    )
    opts.update(changes)
    return graph.categorical_probability_node("probabilities", **opts)


def artifact(payload, type_, name, *, key="b" * 64, platform=False):
    return ArtifactValue(
        payload,
        type_,
        opaque_artifact_key(key, name),
        key,
        NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint())
        if platform
        else NumericScope(),
    )


def fit_context():
    node = fit_node()
    return KernelContext(
        node,
        {"person": table()},
        {"person": Weights(np.array([1.0, 2.0, 3.0, 4.0] * 2), WeightKind.DESIGN)},
        pd.Series(dtype=str),
        node.params,
        np.random.default_rng(0),
        artifacts={
            "source_projection": artifact(DONOR_SOURCE, SOURCE_TYPE, "projection")
        },
    )


@pytest.fixture(scope="module")
def fitted():
    return graph.CategoricalFitKernel().run(fit_context())


def application_context(fitted):
    node = apply_node()
    return KernelContext(
        node,
        {},
        {},
        pd.Series(dtype=str),
        node.params,
        np.random.default_rng(0),
        artifacts={
            "model": artifact(
                fitted.artifacts["model"], cat.MODEL_TYPE, "model", platform=True
            ),
            "model_metadata": artifact(
                fitted.artifacts["model_metadata"],
                graph.MODEL_METADATA_TYPE,
                "model_metadata",
                platform=True,
            ),
            "matrix": artifact(
                matrix(), model_input.RECIPIENT_MATRIX_TYPE, "matrix", key="c" * 64
            ),
            "source_projection": artifact(
                RECIPIENT_SOURCE, SOURCE_TYPE, "projection", key="c" * 64
            ),
        },
    )


def binding(context):
    doc = cat.codec.decode_json(context.artifacts["model_metadata"].payload)
    return dict(
        model_sha256=doc["model_sha256"],
        model_producer_key=context.artifacts["model"].producer_key,
        training_id=doc["training"]["training_id"],
        donor_source_sha256=cat._sha(DONOR_SOURCE),
        matrix_sha256=cat._sha(context.artifacts["matrix"].payload),
        matrix_producer_key=context.artifacts["matrix"].producer_key,
        source_sha256=cat._sha(RECIPIENT_SOURCE),
        source_producer_key=context.artifacts["source_projection"].producer_key,
        classes=list(CLASSES),
    )


def test_real_probability_adapter_preserves_declared_order_and_rng(fitted):
    context = application_context(fitted)
    before = cat.codec.encode_json(context.rng.bit_generator.state)
    result = graph.CategoricalProbabilityKernel().run(context)
    parsed = graph.read_probabilities(
        result.artifacts["probabilities"],
        expected_binding=binding(context),
        recipient_matrix=matrix(),
    )
    np.testing.assert_allclose(
        parsed.to_numpy(), np.tile([0.1, 0.2, 0.3, 0.4], (3, 1)), rtol=0, atol=1e-12
    )
    assert tuple(parsed.columns) == CLASSES
    assert cat.codec.encode_json(context.rng.bit_generator.state) == before
    assert not result.columns and result.frame is None
    assert (
        graph.CategoricalProbabilityKernel.capabilities.seed_source is SeedSource.NONE
    )
    assert graph.CategoricalFitKernel.capabilities.seed_source is SeedSource.PARAM


@pytest.mark.parametrize(
    "defect",
    [
        "params",
        "type",
        "key",
        "producer",
        "model_bytes",
        "metadata",
        "matrix_bytes",
        "source_bytes",
        "numeric_scope",
        "fit_seed",
        "predictor_order",
    ],
)
def test_typed_inputs_refuse_before_probability_evaluation(fitted, monkeypatch, defect):
    context = application_context(fitted)
    values = dict(context.artifacts)
    if defect == "params":
        context = replace(context, params={**context.params, "extra": True})
    elif defect == "type":
        values["model"] = replace(values["model"], type=SOURCE_TYPE)
    elif defect == "key":
        values["matrix"] = replace(values["matrix"], key="f" * 64)
    elif defect == "producer":
        values["model_metadata"] = artifact(
            values["model_metadata"].payload,
            graph.MODEL_METADATA_TYPE,
            "model_metadata",
            key="d" * 64,
            platform=True,
        )
    elif defect == "model_bytes":
        values["model"] = replace(
            values["model"], payload=values["model"].payload + b"bad"
        )
    elif defect == "metadata":
        metadata = cat.codec.decode_json(values["model_metadata"].payload)
        metadata["training"]["source_sha256"] = "f" * 64
        values["model_metadata"] = replace(
            values["model_metadata"], payload=cat.codec.encode_json(metadata)
        )
    elif defect == "matrix_bytes":
        values["matrix"] = replace(
            values["matrix"], payload=values["matrix"].payload + b"bad"
        )
    elif defect == "source_bytes":
        values["source_projection"] = replace(
            values["source_projection"], payload=b"changed"
        )
    elif defect == "numeric_scope":
        values["model"] = replace(values["model"], numerics=NumericScope())
    else:
        new_fit = (
            fit_node(seed=102)
            if defect == "fit_seed"
            else fit_node(predictors=("total", "age"))
        )
        node = apply_node(fit=new_fit)
        context = replace(context, node=node, params=node.params)
    calls = []

    def forbidden(*a, **k):
        calls.append(True)
        raise AssertionError("invalid edge reached probability evaluator")

    monkeypatch.setattr(cat.CategoricalModel, "probabilities", forbidden)
    with pytest.raises(ValueError):
        graph.CategoricalProbabilityKernel().run(replace(context, artifacts=values))
    assert not calls


@pytest.mark.parametrize(
    "defect",
    ["classes", "matrix_sha256", "source_sha256", "model_producer_key", "training_id"],
)
def test_probability_reader_refuses_changed_expected_identity(fitted, defect):
    context = application_context(fitted)
    result = graph.CategoricalProbabilityKernel().run(context)
    expected = binding(context)
    expected[defect] = list(CLASSES[::-1]) if defect == "classes" else "e" * 64
    with pytest.raises(ValueError, match="BINDING"):
        graph.read_probabilities(
            result.artifacts["probabilities"],
            expected_binding=expected,
            recipient_matrix=matrix(),
        )


class SourceKernel(KernelBase):
    ref = "test.categorical.source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        frame = load_source("frame-store", context.sources[context.node.id])
        donor = context.node.id == "donors"
        artifacts = {"projection": DONOR_SOURCE if donor else RECIPIENT_SOURCE}
        if not donor:
            artifacts["matrix"] = matrix(frame.person)
        return KernelResult(frame=frame, artifacts=artifacts)


def source_frame(values, *, donor):
    person = values.copy(deep=True)
    person["person_household_id"] = person.person_id.to_numpy(copy=True)
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": person.person_id.to_numpy(copy=True)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                np.array([1.0, 2.0, 3.0, 4.0] * 2) if donor else np.ones(len(person)),
                WeightKind.DESIGN,
            )
        },
        metadata={"invented_categorical_fixture": True},
    )


def declarations(fit=None, apply=None):
    creates = []
    for name, donor in (("donors", True), ("recipients", False)):
        values = table(donor)
        creates.append(
            Node(
                name,
                SourceKernel.ref,
                sources=(name,),
                structural=StructuralDelta.CREATE,
                outputs=tuple(
                    Owned("person", c, str(values[c].dtype))
                    for c in values
                    if c != "person_id"
                ),
                artifact_outputs=(ArtifactOutput("projection", SOURCE_TYPE),)
                + (
                    ()
                    if donor
                    else (ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),)
                ),
            )
        )
    fitted = fit_node() if fit is None else fit
    applied = apply_node(fitted) if apply is None else apply
    return compile_graph(
        Graph(
            "us",
            sources=(
                SourceRef("donors", "frame-store"),
                SourceRef("recipients", "frame-store"),
            ),
            nodes=(*creates, fitted, applied),
        )
    )


def test_genuine_tiny_graph_cold_required_keys_and_corruption(tmp_path):
    compiled = declarations()
    registry = KernelRegistry()
    for kernel in (
        SourceKernel(),
        graph.CategoricalFitKernel(),
        graph.CategoricalProbabilityKernel(),
    ):
        registry.register(kernel)
    store = ContentStore(tmp_path)
    frames = {
        name: source_frame(table(donor), donor=donor)
        for name, donor in (("donors", True), ("recipients", False))
    }
    refs = {
        name: store.put_frame(hashlib.sha256(name.encode()).hexdigest(), frame)
        for name, frame in frames.items()
    }
    before = {name: frame.person.copy(deep=True) for name, frame in frames.items()}
    cold = run_graph(compiled, sources=refs, store=store, kernels=registry)
    assert len(cold.nodes) == 4 and not any(n.hit for n in cold.nodes.values())
    warm = run_graph(
        compiled,
        sources=refs,
        store=ContentStore(tmp_path),
        kernels=registry,
        resume="require",
    )
    assert cold.key == warm.key and all(n.hit for n in warm.nodes.values())
    assert (
        cold.node("probabilities").opaque_artifacts
        == warm.node("probabilities").opaque_artifacts
    )
    values = cold.node("probabilities")
    payload = store.load_bytes(values.opaque_artifacts["probabilities"])
    fitted = cold.node("fit")
    recipient = cold.node("recipients")
    model_bytes = store.load_bytes(fitted.opaque_artifacts["model"])
    model_metadata = cat.codec.decode_json(
        store.load_bytes(fitted.opaque_artifacts["model_metadata"])
    )
    donor_source = store.load_bytes(cold.node("donors").opaque_artifacts["projection"])
    recipient_source = store.load_bytes(recipient.opaque_artifacts["projection"])
    recipient_matrix = store.load_bytes(recipient.opaque_artifacts["matrix"])
    assert donor_source == DONOR_SOURCE and recipient_source == RECIPIENT_SOURCE
    assert recipient_matrix == matrix()
    assert model_metadata["model_sha256"] == cat._sha(model_bytes)
    # Bind the documented fields to the actual retained producer nodes and
    # artifacts; an executor receipt also contains unrelated capability data.
    expected = dict(
        model_sha256=cat._sha(model_bytes),
        model_producer_key=fitted.key,
        training_id=model_metadata["training"]["training_id"],
        donor_source_sha256=cat._sha(donor_source),
        matrix_sha256=cat._sha(recipient_matrix),
        matrix_producer_key=recipient.key,
        source_sha256=cat._sha(recipient_source),
        source_producer_key=recipient.key,
        classes=list(CLASSES),
    )
    parsed = graph.read_probabilities(
        payload, expected_binding=expected, recipient_matrix=recipient_matrix
    )
    np.testing.assert_allclose(
        parsed.to_numpy(), np.tile([0.1, 0.2, 0.3, 0.4], (3, 1)), atol=1e-12, rtol=0
    )
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame.person, before[name])
    changed_fit = fit_node(seed=102)
    changed = run_graph(
        declarations(changed_fit), sources=refs, store=store, kernels=registry
    )
    assert changed.node("donors").hit and changed.node("recipients").hit
    assert not changed.node("fit").hit and not changed.node("probabilities").hit
    assert changed.node("fit").key != cold.node("fit").key
    # A recipient-only matrix identity change cannot reuse application bytes.
    changed_table = table(False)
    changed_table.loc[0, "total"] = 1.0
    changed_frame = source_frame(changed_table, donor=False)
    new_refs = {**refs, "recipients": store.put_frame("a" * 64, changed_frame)}
    new_apply = apply_node(matrix_sha256=cat._sha(matrix(changed_table)))
    updated = run_graph(
        declarations(apply=new_apply), sources=new_refs, store=store, kernels=registry
    )
    assert updated.node("fit").hit and not updated.node("probabilities").hit
    corrupt = (
        store.object_path(values.opaque_artifacts["probabilities"]) / "payload.bin"
    )
    corrupt.write_bytes(payload + b" ")
    with pytest.raises(StoreCorruptError):
        run_graph(
            compiled, sources=refs, store=store, kernels=registry, resume="require"
        )
