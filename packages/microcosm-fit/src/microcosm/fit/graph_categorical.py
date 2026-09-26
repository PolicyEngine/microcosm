"""Typed donor-only categorical fit and recipient-only probability operations.

Source projection bytes are supplied by the owning host's typed edge. Their
hashes bind dependencies; neither those hashes nor detached model/probability
artifacts issue source authority. No population columns or RNG draws are owned.
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import categorical, model_input, qrf, qrf_target
from microcosm.fit import model as weight_model
from microcosm.frame import WeightKind
from microcosm.frame import weights as frame_weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    SeedSource,
    Slice,
    platform_fingerprint,
    source_hash,
)
from microcosm.graph.keys import opaque_artifact_key

MODEL_METADATA_TYPE = ArtifactType("microcosm.fit.categorical_model_metadata", 1)
PROBABILITY_TYPE = ArtifactType("microcosm.fit.categorical_probabilities", 1)
_MAGIC = b"microcosm.fit.categorical_probabilities/1\n"


def _require(condition, reason):
    if not condition:
        raise ValueError("GRAPH_CATEGORICAL_" + reason)


def _edge(value, name):
    _require(type(value) is ArtifactInput and value.name == name, "EDGE:" + name)
    return value


def categorical_fit_node(
    node_id,
    *,
    population,
    entity,
    predictors,
    label,
    classes,
    seed,
    config,
    source_projection,
    source_sha256,
):
    """Declare a DESIGN-weighted fit on a separately qualified donor population."""
    categorical._names(predictors)
    categorical._classes(classes)
    categorical._names((entity, label))
    _require(
        label not in predictors and f"{entity}_id" not in (*predictors, label),
        "COLUMN_COLLISION",
    )
    _require(
        type(config) is categorical.CategoricalConfig
        and categorical._hash(source_sha256),
        "FIT_CONFIG",
    )
    config.parameters(seed)
    return Node(
        node_id,
        CategoricalFitKernel.ref,
        population=population,
        inputs=(Slice(entity, (*predictors, label)),),
        params=dict(
            entity=entity,
            predictors=predictors,
            label=label,
            classes=classes,
            seed=seed,
            config_json=codec.encode_json(asdict(config)).decode(),
            source_sha256=source_sha256,
            weight_kind=WeightKind.DESIGN.value,
        ),
        artifact_inputs=(_edge(source_projection, "source_projection"),),
        artifact_outputs=(
            ArtifactOutput("model", categorical.MODEL_TYPE),
            ArtifactOutput("model_metadata", MODEL_METADATA_TYPE),
        ),
        description="Fit a DESIGN-weighted categorical classifier; no recipient fitting or survey authority.",
    )


def categorical_probability_node(
    node_id,
    *,
    population,
    fit_node,
    matrix,
    matrix_sha256,
    source_projection,
    source_sha256,
):
    """Apply a fitted model without fitting recipients or consuming draw RNG."""
    _require(
        type(fit_node) is Node and fit_node.kernel == CategoricalFitKernel.ref,
        "FIT_NODE",
    )
    _require(
        all(categorical._hash(v) for v in (matrix_sha256, source_sha256)),
        "APPLICATION_HASH",
    )
    _edge(matrix, "matrix")
    _require(matrix.type == model_input.RECIPIENT_MATRIX_TYPE, "MATRIX_TYPE")
    p = dict(fit_node.params)
    return Node(
        node_id,
        CategoricalProbabilityKernel.ref,
        population=population,
        params=dict(
            fit_params_json=codec.encode_json(p).decode(),
            matrix_sha256=matrix_sha256,
            source_sha256=source_sha256,
        ),
        artifact_inputs=(
            ArtifactInput("model", fit_node.id, "model", categorical.MODEL_TYPE),
            ArtifactInput(
                "model_metadata", fit_node.id, "model_metadata", MODEL_METADATA_TYPE
            ),
            matrix,
            _edge(source_projection, "source_projection"),
        ),
        artifact_outputs=(ArtifactOutput("probabilities", PROBABILITY_TYPE),),
        description="Evaluate actual fitted category probabilities in declared class order; no draw or canonical amount attachment.",
    )


def _artifact(context, name, *, platform=False):
    edges = [edge for edge in context.node.artifact_inputs if edge.name == name]
    _require(len(edges) == 1 and name in context.artifacts, "ARTIFACT_ROSTER")
    edge, value = edges[0], context.artifacts[name]
    _require(
        type(value) is ArtifactValue
        and value.type == edge.type
        and value.key == opaque_artifact_key(value.producer_key, edge.artifact),
        "ARTIFACT_IDENTITY:" + name,
    )
    if platform:
        _require(
            value.numerics.numeric is Numeric.PLATFORM_BITWISE
            and value.numerics.platform == platform_fingerprint(),
            "ARTIFACT_PLATFORM:" + name,
        )
    return value


def _source(context):
    value = _artifact(context, "source_projection")
    _require(
        categorical._sha(value.payload) == context.params["source_sha256"],
        "SOURCE_HASH",
    )
    return value


def _fit_from_params(node_id, population, params, edge):
    params = dict(params)
    _require(
        params.pop("weight_kind") == WeightKind.DESIGN.value,
        "DESIGN_WEIGHT_DECLARATION",
    )
    config = categorical.CategoricalConfig(
        **codec.decode_json(params.pop("config_json").encode())
    )
    params["predictors"] = tuple(params["predictors"])
    params["classes"] = tuple(params["classes"])
    return categorical_fit_node(
        node_id, population=population, config=config, source_projection=edge, **params
    )


def _metadata(model, payload):
    return dict(model_sha256=categorical._sha(payload), training=model.metadata)


class CategoricalFitKernel(KernelBase):
    ref = "fit.categorical.fit@1"
    capabilities = Capabilities(
        Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        dependencies=categorical.DEPENDENCIES,
    )

    def implementation_hash(self):
        return _implementation(self)

    def run(self, context):
        node = context.node
        _require(dict(context.params) == dict(node.params), "PARAMETERS")
        _require(len(node.artifact_inputs) == 1, "FIT_EDGES")
        expected = _fit_from_params(
            node.id, node.population, context.params, node.artifact_inputs[0]
        )
        _require(node.normative() == expected.normative(), "FIT_DECLARATION")
        p = context.params
        _require(
            set(context.artifacts) == {"source_projection"}
            and set(context.tables) == {p["entity"]}
            and not context.sources,
            "FIT_CONTEXT",
        )
        _source(context)
        table = context.tables[p["entity"]]
        identity = f"{p['entity']}_id"
        _require(
            table.columns.is_unique
            and {identity, *p["predictors"], p["label"]} <= set(table),
            "FIT_COLUMNS",
        )
        _require(
            table[identity].dtype == np.dtype("int64") and table[identity].is_unique,
            "DONOR_IDS",
        )
        index = pd.Index(table[identity].to_numpy(copy=True), name=identity)
        features = table.loc[:, list(p["predictors"])].copy(deep=True)
        features.index = index
        labels = pd.Series(table[p["label"]].to_numpy(copy=True), index=index)
        model = categorical.fit_categorical(
            features,
            labels,
            weights=context.weights[p["entity"]],
            classes=p["classes"],
            entity=p["entity"],
            source_sha256=p["source_sha256"],
            seed=p["seed"],
            config=categorical.CategoricalConfig(
                **codec.decode_json(p["config_json"].encode())
            ),
        )
        payload = model.to_bytes()
        metadata = _metadata(model, payload)
        return KernelResult(
            artifacts=dict(model=payload, model_metadata=codec.encode_json(metadata)),
            receipt=dict(
                model_sha256=metadata["model_sha256"],
                training_id=model.metadata["training_id"],
                rows=len(table),
                weight_kind=WeightKind.DESIGN.value,
            ),
        )


def _implementation(kernel):
    return source_hash(
        sys.modules[__name__],
        categorical,
        model_input,
        codec,
        qrf,
        qrf_target,
        weight_model,
        frame_weights,
        dependencies=kernel.capabilities.dependencies,
    )


class CategoricalProbabilityKernel(KernelBase):
    ref = "fit.categorical.probabilities@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=categorical.DEPENDENCIES,
    )

    def implementation_hash(self):
        return _implementation(self)

    def run(self, context):
        node, p = context.node, context.params
        _require(
            dict(p) == dict(node.params)
            and set(p) == {"fit_params_json", "matrix_sha256", "source_sha256"},
            "PARAMETERS",
        )
        _require(len(node.artifact_inputs) == 4, "PROBABILITY_EDGES")
        model_edge, metadata_edge, matrix_edge, source_edge = node.artifact_inputs
        _require(model_edge.producer == metadata_edge.producer, "MODEL_PRODUCER")
        # Reconstruct the expected declared fit shape; its population is not
        # used here. The actual producer/key is bound by the typed graph edges.
        fit = _fit_from_params(
            model_edge.producer,
            node.population,
            codec.decode_json(p["fit_params_json"].encode()),
            source_edge,
        )
        expected = categorical_probability_node(
            node.id,
            population=node.population,
            fit_node=fit,
            matrix=matrix_edge,
            matrix_sha256=p["matrix_sha256"],
            source_projection=source_edge,
            source_sha256=p["source_sha256"],
        )
        _require(node.normative() == expected.normative(), "PROBABILITY_DECLARATION")
        _require(
            not context.tables
            and not context.sources
            and set(context.artifacts)
            == {"model", "model_metadata", "matrix", "source_projection"},
            "PROBABILITY_CONTEXT",
        )
        source = _source(context)
        model_value = _artifact(context, "model", platform=True)
        metadata_value = _artifact(context, "model_metadata", platform=True)
        matrix_value = _artifact(context, "matrix")
        _require(
            model_value.producer_key == metadata_value.producer_key,
            "MODEL_PRODUCER_KEY",
        )
        _require(
            categorical._sha(matrix_value.payload) == p["matrix_sha256"], "MATRIX_HASH"
        )
        metadata = codec.decode_json(metadata_value.payload)
        _require(
            set(metadata) == {"model_sha256", "training"}
            and metadata["model_sha256"] == categorical._sha(model_value.payload),
            "MODEL_DIGEST",
        )
        training = metadata["training"]
        categorical._metadata(training)
        fp = codec.decode_json(p["fit_params_json"].encode())
        fp["config"] = codec.decode_json(fp["config_json"].encode())
        _require(
            codec.encode_json(
                {
                    k: training[k]
                    for k in (
                        "entity",
                        "classes",
                        "predictors",
                        "seed",
                        "config",
                        "source_sha256",
                        "weight_kind",
                    )
                }
            )
            == codec.encode_json(
                {
                    k: fp[k]
                    for k in (
                        "entity",
                        "classes",
                        "predictors",
                        "seed",
                        "config",
                        "source_sha256",
                        "weight_kind",
                    )
                }
            ),
            "TRAINING_DECLARATION",
        )
        model = categorical.CategoricalModel.from_trusted_bytes(
            model_value.payload, expected_sha256=metadata["model_sha256"]
        )
        _require(
            codec.encode_json(_metadata(model, model_value.payload))
            == metadata_value.payload,
            "MODEL_METADATA",
        )
        matrix = model_input.decode_recipient_matrix(matrix_value.payload)
        probabilities = model.probabilities(matrix.features, entity=matrix.entity)
        body = categorical.matrix_bytes(probabilities, entity=matrix.entity)
        binding = dict(
            protocol=PROBABILITY_TYPE.name + "/1",
            model_sha256=metadata["model_sha256"],
            model_producer_key=model_value.producer_key,
            training_id=training["training_id"],
            donor_source_sha256=training["source_sha256"],
            matrix_sha256=p["matrix_sha256"],
            matrix_producer_key=matrix_value.producer_key,
            source_sha256=p["source_sha256"],
            source_producer_key=source.producer_key,
            classes=list(fp["classes"]),
            probability_matrix_sha256=categorical._sha(body),
        )
        header = codec.encode_json(binding)
        _require(len(header) <= categorical._HEADER_LIMIT, "PROBABILITY_HEADER_SIZE")
        payload = _MAGIC + len(header).to_bytes(4, "big") + header + body
        return KernelResult(
            artifacts={"probabilities": payload},
            receipt={**binding, "rows": len(probabilities)},
        )


def read_probabilities(payload, *, expected_binding, recipient_matrix):
    """Check exact identity/shape/value binding, not source or scientific authority.

    The owner obtains expected bindings from its actual retained graph edges.
    Detached matching bytes alone cannot establish that a source was qualified.
    """
    offset = len(_MAGIC)
    _require(
        type(payload) is bytes
        and payload.startswith(_MAGIC)
        and len(payload) > offset + 4,
        "PROBABILITY_ENVELOPE",
    )
    size = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    _require(
        0 < size <= categorical._HEADER_LIMIT and offset + size < len(payload),
        "PROBABILITY_HEADER_SIZE",
    )
    header = codec.decode_json(payload[offset : offset + size])
    fields = {
        "protocol",
        "model_sha256",
        "model_producer_key",
        "training_id",
        "donor_source_sha256",
        "matrix_sha256",
        "matrix_producer_key",
        "source_sha256",
        "source_producer_key",
        "classes",
        "probability_matrix_sha256",
    }
    _require(
        set(header) == fields and header["protocol"] == PROBABILITY_TYPE.name + "/1",
        "PROBABILITY_SCHEMA",
    )
    _require(
        type(expected_binding) is dict
        and set(expected_binding) == fields - {"protocol", "probability_matrix_sha256"},
        "EXPECTED_BINDING",
    )
    _require(
        codec.encode_json({k: header[k] for k in expected_binding})
        == codec.encode_json(expected_binding),
        "PROBABILITY_BINDING",
    )
    _require(
        all(categorical._hash(header[k]) for k in fields - {"protocol", "classes"}),
        "PROBABILITY_HASHES",
    )
    _require(type(header["classes"]) is list, "PROBABILITY_CLASSES")
    categorical._classes(tuple(header["classes"]))
    body = payload[offset + size :]
    _require(
        categorical._sha(body) == header["probability_matrix_sha256"]
        and categorical._sha(recipient_matrix) == header["matrix_sha256"],
        "PROBABILITY_MATRIX_HASH",
    )
    result, recipient = (
        model_input.decode_recipient_matrix(body),
        model_input.decode_recipient_matrix(recipient_matrix),
    )
    _require(
        result.entity == recipient.entity
        and result.features.index.equals(recipient.features.index)
        and result.features.index.name == recipient.features.index.name
        and tuple(result.features.columns) == tuple(header["classes"]),
        "PROBABILITY_INDEX_SCHEMA",
    )
    values = result.features.to_numpy(copy=False)
    _require(
        (values >= 0).all()
        and (values <= 1).all()
        and np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-12),
        "PROBABILITY_VALUES",
    )
    return result.features
