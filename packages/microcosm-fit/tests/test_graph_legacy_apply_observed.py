"""Real three-target application with separately traced qualified conditioning."""

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import graph_legacy_apply_observed as observed
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.fit.graph_legacy_apply_matrix import (
    LegacyQRFApplyMatrixKernel,
    decode_matrix_apply_state,
)
from microcosm.fit.graph_legacy_qrf import (
    LegacyQRFTrainKernel,
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
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


class Source(KernelBase):
    ref = "test.observed.source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources[context.node.sources[0]])
        )


class Prepare(KernelBase):
    ref = "test.observed.prepare@1"
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
                "matrix": model_input.encode_recipient_matrix(
                    features, entity="household", entity_ids=ids
                )
            }
        )


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


class QualifiedFixture(KernelBase):
    """Invented qualified values, deliberately no claim of survey authority."""

    ref = "test.observed.fixture@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
    )

    def run(self, context):
        matrix_value = context.artifacts["matrix"]
        matrix = model_input.decode_recipient_matrix(matrix_value.payload)
        artifacts = {}
        for target in ("a", "b"):
            artifacts[target] = observed.encode_observed_target(
                np.array(context.params[target], dtype="<f8"),
                np.array(context.params[target + "_known"], dtype=bool),
                target=target,
                index=matrix.features.index,
                matrix_sha256=codec.sha(matrix_value.payload),
                matrix_producer_key=matrix_value.producer_key,
                source_sha256="c" * 64,
            )
        return KernelResult(artifacts=artifacts)


def setup(root, *, mask_b=(True, False, True)):
    a, b = np.tile(np.repeat([1.0, 2.0, 3.0], 3), 8), np.tile([10.0, 20.0, 30.0], 24)
    donor = population(
        pd.DataFrame(
            {"x": np.ones(72), "indicator": np.zeros(72), "a": a, "b": b, "c": b * 10}
        ),
        np.arange(72, dtype="int64") + 10,
    )
    recipient = population(
        pd.DataFrame({"x": [1.0, 1.0, 1.0], "indicator": [0.0, 0.0, 0.0]}),
        np.array([11, 97, 301], dtype="int64"),
    )
    store, sources, nodes = ContentStore(root), {}, []
    for name, frame in (("donor", donor), ("recipient", recipient)):
        key = hashlib.sha256(
            frame.table("household").to_json(orient="table").encode()
        ).hexdigest()
        sources[name] = store.put_frame(key, frame)
        nodes.append(
            Node(
                name,
                Source.ref,
                sources=(name,),
                structural=StructuralDelta.CREATE,
                outputs=tuple(
                    Owned("household", c, "float64")
                    for c in frame.table("household")
                    if c != "household_id"
                ),
            )
        )
    fits = legacy_qrf_train_nodes(
        "fit",
        population="donor",
        entity="household",
        predictors=("x", "indicator"),
        targets=("a", "b", "c"),
        seed=31,
        n_estimators=2,
        zero_atol=0,
        phase="invented",
    )
    matrix = Node(
        "matrix",
        Prepare.ref,
        population="recipient",
        inputs=(Slice("household", ("x", "indicator")),),
        artifact_outputs=(ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),),
    )
    supplied = Node(
        "qualified",
        QualifiedFixture.ref,
        population="recipient",
        params={
            "a": (3.0, 1.0, 2.0),
            "a_known": (True, True, True),
            "b": (30.0, -999.0, 10.0),
            "b_known": mask_b,
        },
        artifact_inputs=(
            ArtifactInput(
                "matrix", "matrix", "matrix", model_input.RECIPIENT_MATRIX_TYPE
            ),
        ),
        artifact_outputs=tuple(
            ArtifactOutput(t, observed.OBSERVED_TARGET_TYPE) for t in ("a", "b")
        ),
    )
    common = dict(
        population="recipient",
        fit_nodes=fits,
        matrix_producer="matrix",
        seed=31,
        phase="invented",
    )
    applies = observed.legacy_qrf_apply_observed_matrix_nodes(
        "conditioned",
        observed_producer="qualified",
        observed_artifacts={"a": "a", "b": "b"},
        **common,
    )
    plain = observed.legacy_qrf_apply_observed_matrix_nodes("plain", **common)
    legacy = legacy_qrf_apply_matrix_nodes("legacy", **common)
    declaration = Graph(
        "us",
        (SourceRef("donor", "frame-store"), SourceRef("recipient", "frame-store")),
        (*nodes, matrix, supplied, *fits, *applies, *plain, *legacy),
    )
    kernels = KernelRegistry()
    for kernel in (
        Source(),
        Prepare(),
        QualifiedFixture(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
        observed.LegacyQRFApplyObservedMatrixKernel(),
    ):
        kernels.register(kernel)
    return (
        compile_graph(declaration),
        store,
        kernels,
        sources,
        fits,
        applies,
        plain,
        legacy,
        donor,
    )


def blob(manifest, store, node, name):
    return store.load_bytes(
        manifest.node(node.id if isinstance(node, Node) else node).opaque_artifacts[
            name
        ]
    )


@pytest.fixture(scope="module")
def applied(tmp_path_factory):
    # Explicit worker env is also needed during module-fixture setup.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPULACE_FIT_N_JOBS", "1")
        patch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
        parts = setup(tmp_path_factory.mktemp("observed-matrix"))
        compiled, store, kernels, sources, *_ = parts
        manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
        yield parts, manifest


def test_three_real_targets_condition_before_successors_and_preserve_draws(applied):
    parts, manifest = applied
    compiled, store, kernels, sources, fits, applies, plain, legacy, donor = parts
    matrix_payload = blob(manifest, store, "matrix", "matrix")
    matrix = model_input.decode_recipient_matrix(matrix_payload)
    prior = pd.DataFrame(index=matrix.features.index)
    state = qrf.RegimeGatedQRF(seed=31, n_estimators=2, zero_atol=0).start_chain(
        donor, ["x", "indicator"], ["a", "b", "c"], weights="design"
    )
    for i, (fit, node) in enumerate(zip(fits, applies, strict=True)):
        model_payload = blob(manifest, store, fit, "model")
        fitted = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
            model_payload, expected_sha256=codec.sha(model_payload)
        )
        expected = qrf_target.apply_target(fitted, matrix.features, prior, state=state)
        raw = blob(manifest, store, node, "raw_draw")
        assert raw == codec.encode_raw_target(
            expected.raw_draw, target=expected.target, index=matrix.features.index
        )
        merged = expected.raw_draw.copy()
        if i < 2:
            supplied, known = observed.read_observed_target(
                blob(manifest, store, "qualified", expected.target),
                target=expected.target,
                index=matrix.features.index,
                matrix_sha256=codec.sha(matrix_payload),
                matrix_producer_key=manifest.node("matrix").key,
            )
            merged[known] = supplied[known]
        conditioning = blob(manifest, store, node, "conditioning")
        assert conditioning == codec.encode_raw_target(
            merged, target=expected.target, index=matrix.features.index
        )
        packet = observed.decode_observed_matrix_apply_state(
            blob(manifest, store, node, "apply_state")
        )
        record = packet["application"]["raw_targets"][-1]
        assert record["draw_sha256"] == codec.sha(raw)
        assert record["conditioning_sha256"] == codec.sha(conditioning)
        assert record["observed_rows"] == (3, 2, 0)[i]
        assert record["model_producer_key"] == manifest.node(fit.id).key
        assert record["observed_producer_key"] == (
            manifest.node("qualified").key if i < 2 else None
        )
        assert packet["application"]["prior_producer_keys"] == [
            manifest.node(n.id).key for n in applies[:i]
        ]
        prior[expected.target], state = merged, expected.state
    np.testing.assert_array_equal(prior.a, [3.0, 1.0, 2.0])
    np.testing.assert_array_equal(prior.b.iloc[[0, 2]], [30.0, 10.0])
    np.testing.assert_array_equal(prior.c, prior.b * 10)
    # Fully unknown final target is identical byte-for-byte across its outputs.
    assert blob(manifest, store, applies[-1], "raw_draw") == blob(
        manifest, store, applies[-1], "conditioning"
    )
    for new, old in zip(plain, legacy, strict=True):
        assert blob(manifest, store, new, "raw_draw") == blob(
            manifest, store, old, "raw_draw"
        )
        assert blob(manifest, store, new, "conditioning") == blob(
            manifest, store, old, "raw_draw"
        )
        old_state = decode_matrix_apply_state(
            blob(manifest, store, old, "apply_state")
        )["application"]["state"]
        assert (
            observed.decode_observed_matrix_apply_state(
                blob(manifest, store, new, "apply_state")
            )["application"]["state"]
            == old_state
        )
    warm = run_graph(
        compiled, sources=sources, store=store, kernels=kernels, resume="require"
    )
    assert warm.key == manifest.key and all(n.hit for n in warm.nodes.values())
    for node in applies:
        for name in ("raw_draw", "conditioning", "apply_state"):
            assert blob(warm, store, node, name) == blob(manifest, store, node, name)


def test_known_mask_does_not_skip_draws_or_change_rng_position(applied):
    parts, before = applied
    compiled, store, kernels, sources, fits, applies, *_ = parts
    nodes = tuple(
        replace(n, params={**n.params, "b_known": (False, False, False)})
        if n.id == "qualified"
        else n
        for n in compiled.graph.nodes
    )
    changed = run_graph(
        compile_graph(replace(compiled.graph, nodes=nodes)),
        sources=sources,
        store=store,
        kernels=kernels,
    )
    assert all(changed.node(n.id).hit for n in fits)
    for node in applies:
        previous = observed.decode_observed_matrix_apply_state(
            blob(before, store, node, "apply_state")
        )
        current = observed.decode_observed_matrix_apply_state(
            blob(changed, store, node, "apply_state")
        )
        assert (
            current["application"]["state"]["draw_rng_state"]
            == previous["application"]["state"]["draw_rng_state"]
        )
    assert blob(before, store, applies[1], "raw_draw") == blob(
        changed, store, applies[1], "raw_draw"
    )
    assert blob(before, store, applies[1], "conditioning") != blob(
        changed, store, applies[1], "conditioning"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "prior_bytes",
        "prior_producer",
        "last_producer",
        "matrix_bytes",
        "matrix_producer",
        "model_producer",
        "model_bytes",
        "state_matrix",
        "state_seed",
        "state_schema",
        "missing_prior",
    ],
)
def test_conditioned_kernel_refuses_broken_chain_bindings(applied, mutation):
    parts, manifest = applied
    _, store, _, _, _, nodes, *_ = parts
    context = context_for(manifest, store, nodes[-1])
    artifacts = dict(context.artifacts)

    def change(name, **kwargs):
        artifacts[name] = replace(artifacts[name], **kwargs)

    if mutation == "prior_bytes":
        matrix = model_input.decode_recipient_matrix(artifacts["matrix"].payload)
        change(
            "prior_000",
            payload=codec.encode_raw_target(
                np.zeros(3), target="a", index=matrix.features.index
            ),
        )
    elif mutation == "prior_producer":
        change("prior_000", producer_key="d" * 64)
    elif mutation == "last_producer":
        change("prior_001", producer_key="d" * 64)
    elif mutation == "matrix_bytes":
        matrix = model_input.decode_recipient_matrix(artifacts["matrix"].payload)
        features = matrix.features.copy()
        features.iloc[0, 0] += 1
        change(
            "matrix",
            payload=model_input.encode_recipient_matrix(
                features, entity=matrix.entity, entity_ids=matrix.entity_ids
            ),
        )
    elif mutation == "matrix_producer":
        change("matrix", producer_key="d" * 64)
    elif mutation == "model_producer":
        change("model", producer_key="d" * 64)
    elif mutation == "model_bytes":
        change("model", payload=b"not a model")
    elif mutation == "missing_prior":
        artifacts.pop("prior_000")
    else:
        packet = observed.decode_observed_matrix_apply_state(
            artifacts["apply_state"].payload
        )
        if mutation == "state_matrix":
            packet["matrix_sha256"] = "d" * 64
        elif mutation == "state_seed":
            packet["application"]["seed"] += 1
        else:
            packet["schema_version"] = 1
        change("apply_state", payload=codec.encode_json(packet))
    with pytest.raises(ValueError):
        observed.LegacyQRFApplyObservedMatrixKernel().run(
            replace(context, artifacts=artifacts)
        )


def test_observed_codec_preserves_exact_values_mask_and_unknown_bits():
    index = pd.Index(np.array([11, 97, 301], dtype="int64"), name="household_id")
    values = np.array([-0.0, np.nan, -17.25], dtype="<f8")
    known = np.array([True, False, True])
    binding = dict(
        target="y", index=index, matrix_sha256="a" * 64, matrix_producer_key="b" * 64
    )
    payload = observed.encode_observed_target(
        values, known, source_sha256="c" * 64, **binding
    )
    restored, mask = observed.read_observed_target(payload, **binding)
    assert restored.tobytes() == values.tobytes()
    assert mask.tobytes() == known.tobytes()
    assert not restored.flags.writeable and not mask.flags.writeable
    with pytest.raises(ValueError):
        observed.read_observed_target(payload, **{**binding, "index": index[::-1]})
    with pytest.raises(ValueError):
        observed.encode_observed_target(
            values, np.ones(3, dtype=bool), source_sha256="c" * 64, **binding
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "float32",
        "list",
        "integer_mask",
        "noncanonical_bool",
        "shape",
        "known_nan",
        "known_inf",
    ],
)
def test_observed_codec_refuses_coercion_and_nonfinite_known_cells(mutation):
    values, mask = np.array([1.0, 2.0]), np.array([True, False])
    if mutation == "float32":
        values = values.astype("float32")
    elif mutation == "list":
        values = list(values)
    elif mutation == "integer_mask":
        mask = mask.astype("int64")
    elif mutation == "noncanonical_bool":
        mask = np.frombuffer(bytes([2, 0]), dtype=bool)
    elif mutation == "shape":
        values, mask = values[:, None], mask[:, None]
    elif mutation == "known_nan":
        values[0] = np.nan
    else:
        values[0] = np.inf
    with pytest.raises(ValueError):
        observed.encode_observed_target(
            values,
            mask,
            target="a",
            index=pd.Index([11, 97], dtype="int64"),
            matrix_sha256="a" * 64,
            matrix_producer_key="b" * 64,
            source_sha256="c" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "target",
        "index",
        "matrix_hash",
        "matrix_producer",
        "wrong_type",
        "wrong_scope",
        "truncated",
        "bad_mask",
        "known_inf",
        "schema_bool",
    ],
)
def test_observed_typed_edge_refuses_changed_representation_before_draw(
    applied, monkeypatch, mutation
):
    parts, manifest = applied
    _, store, _, _, _, nodes, *_ = parts
    context = context_for(manifest, store, nodes[0])
    artifacts = dict(context.artifacts)
    value = artifacts["observed"]
    payload = value.payload
    offset = len(observed._MAGIC)
    length = int.from_bytes(payload[offset : offset + 4], "big")
    start = offset + 4
    header = codec.decode_json(payload[start : start + length])
    body = payload[start + length :]
    if mutation in {"target", "index", "matrix_hash", "matrix_producer", "schema_bool"}:
        if mutation == "target":
            header["target"] = "b"
        elif mutation == "index":
            header["index"] = qrf._index_identity(
                pd.Index([301, 97, 11], dtype="int64")
            ).to_dict()
        elif mutation == "matrix_hash":
            header["matrix_sha256"] = "d" * 64
        elif mutation == "matrix_producer":
            header["matrix_producer_key"] = "d" * 64
        else:
            header["schema_version"] = True
        encoded = codec.encode_json(header)
        payload = observed._MAGIC + len(encoded).to_bytes(4, "big") + encoded + body
    elif mutation == "wrong_type":
        value = replace(value, type=codec.RAW_TARGET_TYPE)
    elif mutation == "wrong_scope":
        value = replace(
            value, numerics=NumericScope(Numeric.PLATFORM_BITWISE, platform="other")
        )
    elif mutation == "truncated":
        payload = payload[:-1]
    elif mutation == "bad_mask":
        payload = payload[: start + length] + bytes([2]) + body[1:]
    else:
        values = np.frombuffer(body[3:], dtype="<f8").copy()
        values[0] = np.inf
        payload = payload[: start + length] + body[:3] + values.tobytes()
    artifacts["observed"] = replace(value, payload=payload)

    def must_not_draw(*args, **kwargs):
        pytest.fail("Malformed observed conditioning reached model application")

    monkeypatch.setattr(qrf_target, "apply_target", must_not_draw)
    with pytest.raises(ValueError):
        observed.LegacyQRFApplyObservedMatrixKernel().run(
            replace(context, artifacts=artifacts)
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "producer_missing",
        "producer_bad",
        "model_bad",
        "partial_provenance",
        "unknown_changed",
        "negative_count",
        "too_many_known",
        "bool_count",
    ],
)
def test_v2_state_refuses_incoherent_provenance(applied, mutation):
    parts, manifest = applied
    _, store, _, _, _, nodes, *_ = parts
    packet = observed.decode_observed_matrix_apply_state(
        blob(manifest, store, nodes[-1], "apply_state")
    )
    application = packet["application"]
    record = application["raw_targets"][-1]
    if mutation == "extra":
        packet["authority"] = True
    elif mutation == "producer_missing":
        application["prior_producer_keys"].pop()
    elif mutation == "producer_bad":
        application["prior_producer_keys"][0] = "not-a-key"
    elif mutation == "model_bad":
        record["model_producer_key"] = None
    elif mutation == "partial_provenance":
        record["observed_sha256"] = "a" * 64
    elif mutation == "unknown_changed":
        record["conditioning_sha256"] = "a" * 64
    elif mutation == "negative_count":
        record["observed_rows"] = -1
    elif mutation == "too_many_known":
        application["raw_targets"][0]["observed_rows"] = 4
    else:
        record["observed_rows"] = False
    with pytest.raises(ValueError):
        observed.decode_observed_matrix_apply_state(codec.encode_json(packet))


def test_legacy_and_observed_state_decoders_do_not_accept_each_other(applied):
    parts, manifest = applied
    _, store, _, _, _, nodes, _, legacy, _ = parts
    with pytest.raises(ValueError):
        decode_matrix_apply_state(blob(manifest, store, nodes[-1], "apply_state"))
    with pytest.raises(ValueError):
        observed.decode_observed_matrix_apply_state(
            blob(manifest, store, legacy[-1], "apply_state")
        )


@pytest.mark.parametrize(
    "mapping,producer",
    [
        ({"absent": "a"}, "qualified"),
        ({"a": "a", "b": "a"}, "qualified"),
        ({"a": "a"}, None),
    ],
)
def test_observed_declaration_refuses_ambiguous_target_roster(
    applied, mapping, producer
):
    parts, _ = applied
    fits = parts[4]
    with pytest.raises(ValueError):
        observed.legacy_qrf_apply_observed_matrix_nodes(
            "invalid",
            population="recipient",
            fit_nodes=fits,
            matrix_producer="matrix",
            observed_producer=producer,
            observed_artifacts=mapping,
            seed=31,
            phase="test",
        )
