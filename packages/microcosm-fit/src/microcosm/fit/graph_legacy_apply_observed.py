"""Apply every fitted target, then condition successors on qualified values.

This is a separate protocol from the unchanged legacy raw-draw chain. The
producer owns qualification of supplied values; a source digest is descriptive
provenance, never source authority or a domain/scientific acceptance decision.
"""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import graph_legacy_qrf as declarations
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.fit.kernels import FIT_QRF_DEPENDENCIES
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)

OBSERVED_TARGET_TYPE = ArtifactType("microcosm.fit.qrf.legacy_observed_target", 1)
CONDITIONING_TARGET_TYPE = ArtifactType(
    "microcosm.fit.qrf.legacy_conditioning_target", 1
)
OBSERVED_MATRIX_APPLY_STATE_TYPE = ArtifactType(
    "microcosm.fit.legacy_matrix_apply_state", 2
)
_MAGIC = b"microcosm.fit.qrf.legacy_observed_target/1\n"
_HEADER_MAX = model_input.MATRIX_HEADER_MAX_BYTES


def _binding(*, target, index, matrix_sha256, matrix_producer_key):
    codec.names((target,), "observed target")
    if type(index) is not pd.Index or index.dtype != np.dtype("int64"):
        raise ValueError("Observed target requires the matrix's ordinary int64 index.")
    model_input._index(index.to_numpy(copy=False), index.name)
    if not codec._hash(matrix_sha256) or not codec._hash(matrix_producer_key):
        raise ValueError(
            "Observed target requires exact matrix byte/producer identities."
        )
    return {
        "schema_version": 1,
        "target": target,
        "rows": len(index),
        "index": qrf._index_identity(index).to_dict(),
        "matrix_sha256": matrix_sha256,
        "matrix_producer_key": matrix_producer_key,
    }


def encode_observed_target(
    values, known, *, target, index, matrix_sha256, matrix_producer_key, source_sha256
) -> bytes:
    """Encode exact float64 values and bool knownness without claiming authority.

    Only known cells must be finite. Unknown backing bits are retained exactly
    but never used for conditioning. Neither values nor the mask are coerced.
    """
    metadata = _binding(
        target=target,
        index=index,
        matrix_sha256=matrix_sha256,
        matrix_producer_key=matrix_producer_key,
    )
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype("<f8")
        or not isinstance(known, np.ndarray)
        or known.dtype != np.dtype("bool")
        or values.shape != known.shape
        or values.shape != (len(index),)
        or not np.isin(known.view("u1"), (0, 1)).all()
        or not np.isfinite(values[known]).all()
        or not codec._hash(source_sha256)
    ):
        raise ValueError(
            "Observed target requires aligned exact values, mask and finite known cells."
        )
    header = codec.encode_json({**metadata, "source_sha256": source_sha256})
    if len(header) > _HEADER_MAX:
        raise ValueError("Observed target header exceeds its bound.")
    return (
        _MAGIC
        + len(header).to_bytes(4, "big")
        + header
        + known.tobytes()
        + values.tobytes()
    )


def _read_observed(payload, **binding):
    expected = _binding(**binding)
    if not isinstance(payload, bytes) or not payload.startswith(_MAGIC):
        raise ValueError("Invalid observed target envelope.")
    offset = len(_MAGIC)
    if len(payload) < offset + 4:
        raise ValueError("Truncated observed target header.")
    length = int.from_bytes(payload[offset : offset + 4], "big")
    start = offset + 4
    if length < 1 or length > _HEADER_MAX or len(payload) < start + length:
        raise ValueError("Invalid observed target header length.")
    header = codec.decode_json(payload[start : start + length])
    if (
        set(header) != set(expected) | {"source_sha256"}
        or not codec._hash(header["source_sha256"])
        or any(
            codec.encode_json(header[k]) != codec.encode_json(v)
            for k, v in expected.items()
        )
    ):
        raise ValueError("Observed target target/index/matrix binding changed.")
    body = memoryview(payload)[start + length :]
    rows = expected["rows"]
    if len(body) != rows * 9:
        raise ValueError("Observed target body length differs from its row axis.")
    raw_mask = np.frombuffer(body[:rows], dtype="u1")
    if not np.isin(raw_mask, (0, 1)).all():
        raise ValueError("Observed known mask must contain only boolean bytes.")
    known = raw_mask.view(np.bool_)
    values = np.frombuffer(body[rows:], dtype="<f8")
    if not np.isfinite(values[known]).all():
        raise ValueError("Known observed target values must be finite.")
    return header, values, known


def read_observed_target(
    payload: bytes,
    *,
    target: str,
    index: pd.Index,
    matrix_sha256: str,
    matrix_producer_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read exact readonly arrays from a typed edge; no source authority is issued."""
    _, values, known = _read_observed(
        payload,
        target=target,
        index=index,
        matrix_sha256=matrix_sha256,
        matrix_producer_key=matrix_producer_key,
    )
    return values, known


def decode_observed_matrix_apply_state(payload: bytes) -> dict:
    """Validate the new draw/conditioning history; legacy decoders stay strict.

    Current output producer keys are only available to the next consumer.
    prior_producer_keys records all earlier conditioning producers, while that
    consumer binds the last conditioning and state to the same actual producer.
    """
    packet = codec.decode_json(payload)
    if (
        set(packet)
        != {"schema_version", "matrix_sha256", "matrix_producer_key", "application"}
        or type(packet["schema_version"]) is not int
        or packet["schema_version"] != 2
        or not codec._hash(packet["matrix_sha256"])
        or not codec._hash(packet["matrix_producer_key"])
    ):
        raise ValueError("Invalid observed matrix application checkpoint.")
    application = packet["application"]
    if (
        not isinstance(application, dict)
        or set(application)
        != {
            "schema_version",
            "state",
            "seed",
            "models",
            "raw_targets",
            "prior_producer_keys",
        }
        or type(application["schema_version"]) is not int
        or application["schema_version"] != 2
        or type(application["seed"]) is not int
        or application["seed"] < 0
    ):
        raise ValueError("Invalid observed application state.")
    state = qrf.QRFChainState.from_dict(application["state"])
    if not state.completed_targets or state.recipient_index is None:
        raise ValueError(
            "Observed application requires a completed target and recipient identity."
        )
    codec.history(application["models"], state.completed_targets, models=True)
    history = application["raw_targets"]
    keys = application["prior_producer_keys"]
    if (
        not isinstance(history, list)
        or len(history) != len(state.completed_targets)
        or not isinstance(keys, list)
        or len(keys) != len(history) - 1
        or not all(codec._hash(key) for key in keys)
    ):
        raise ValueError("Observed application history/producer length changed.")
    fields = {
        "target",
        "draw_sha256",
        "conditioning_sha256",
        "model_producer_key",
        "observed_sha256",
        "observed_producer_key",
        "source_sha256",
        "observed_rows",
    }
    for record, target in zip(history, state.completed_targets, strict=True):
        if (
            not isinstance(record, dict)
            or set(record) != fields
            or record["target"] != target
            or not all(
                codec._hash(record[k])
                for k in ("draw_sha256", "conditioning_sha256", "model_producer_key")
            )
            or type(record["observed_rows"]) is not int
            or not 0 <= record["observed_rows"] <= state.recipient_index.length
        ):
            raise ValueError("Invalid observed target history.")
        provenance = [
            record[k]
            for k in ("observed_sha256", "observed_producer_key", "source_sha256")
        ]
        if not (
            all(v is None for v in provenance)
            or all(codec._hash(v) for v in provenance)
        ):
            raise ValueError("Incomplete observed target provenance.")
        if record["observed_sha256"] is None and record["observed_rows"] != 0:
            raise ValueError("Absent observation cannot supply known rows.")
        if (
            record["observed_rows"] == 0
            and record["conditioning_sha256"] != record["draw_sha256"]
        ):
            raise ValueError("Unknown target must retain its untouched draw.")
    return packet


def legacy_qrf_apply_observed_matrix_nodes(
    prefix,
    *,
    population,
    fit_nodes,
    matrix_producer,
    matrix_artifact="matrix",
    observed_producer=None,
    observed_artifacts=None,
    seed,
    phase,
):
    """Declare real draws and optional identity-bound observed conditioning edges."""
    mapping = {} if observed_artifacts is None else observed_artifacts
    if not isinstance(mapping, Mapping):
        raise ValueError("Observed artifacts must map target names to artifact names.")
    mapping = dict(mapping)
    legacy = declarations.legacy_qrf_apply_matrix_nodes(
        prefix,
        population=population,
        fit_nodes=fit_nodes,
        matrix_producer=matrix_producer,
        matrix_artifact=matrix_artifact,
        seed=seed,
        phase=phase,
    )
    targets = tuple(node.params["target"] for node in legacy)
    if (
        not set(mapping) <= set(targets)
        or any(not isinstance(v, str) or not v for v in mapping.values())
        or len(set(mapping.values())) != len(mapping)
        or (
            mapping
            and (not isinstance(observed_producer, str) or not observed_producer)
        )
    ):
        raise ValueError("Observed artifact target roster or producer is invalid.")
    result = []
    for node in legacy:
        edges = tuple(
            replace(edge, type=OBSERVED_MATRIX_APPLY_STATE_TYPE)
            if edge.name == "apply_state"
            else replace(edge, artifact="conditioning", type=CONDITIONING_TARGET_TYPE)
            if edge.name.startswith("prior_")
            else edge
            for edge in node.artifact_inputs
        )
        target = node.params["target"]
        if target in mapping:
            edges += (
                ArtifactInput(
                    "observed", observed_producer, mapping[target], OBSERVED_TARGET_TYPE
                ),
            )
        result.append(
            replace(
                node,
                kernel=LegacyQRFApplyObservedMatrixKernel.ref,
                artifact_inputs=edges,
                artifact_outputs=(
                    ArtifactOutput("raw_draw", codec.RAW_TARGET_TYPE),
                    ArtifactOutput("conditioning", CONDITIONING_TARGET_TYPE),
                    ArtifactOutput("apply_state", OBSERVED_MATRIX_APPLY_STATE_TYPE),
                ),
            )
        )
    return tuple(result)


class LegacyQRFApplyObservedMatrixKernel(KernelBase):
    """Keep actual draws intact; merge qualified cells only for later conditioning."""

    ref = "fit.qrf.legacy_target.apply_matrix_observed@1"
    capabilities = Capabilities(
        Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        dependencies=FIT_QRF_DEPENDENCIES,
    )

    def implementation_hash(self):
        return source_hash(
            type(self),
            codec,
            declarations,
            model_input,
            qrf_target,
            qrf,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        node = context.node
        if node.kernel != self.ref or node.inputs or node.outputs or node.sources:
            raise ValueError("Observed matrix apply reads typed artifacts only.")
        if (
            set(context.params) != {"target", "seed", "phase"}
            or type(context.params["seed"]) is not int
            or context.params["seed"] < 0
            or not isinstance(context.params["phase"], str)
            or not context.params["phase"]
        ):
            raise ValueError(
                "Observed apply requires exact target, nonnegative seed and phase."
            )
        if node.artifact_outputs != (
            ArtifactOutput("raw_draw", codec.RAW_TARGET_TYPE),
            ArtifactOutput("conditioning", CONDITIONING_TARGET_TYPE),
            ArtifactOutput("apply_state", OBSERVED_MATRIX_APPLY_STATE_TYPE),
        ):
            raise ValueError(
                "Observed apply owns draw, conditioning and version2 state."
            )
        edges = {edge.name: edge for edge in node.artifact_inputs}
        if not {"model", "training_state", "matrix"} <= set(edges):
            raise ValueError("Observed apply lacks model/training/matrix edges.")
        model_value = codec.artifact(
            context, "model", qrf_target.LEGACY_QRF_TARGET_TYPE
        )
        training_value = codec.artifact(
            context, "training_state", codec.TRAINING_STATE_TYPE
        )
        if (
            edges["model"].producer != edges["training_state"].producer
            or model_value.producer_key != training_value.producer_key
        ):
            raise ValueError(
                "Model and training state require the same actual fit producer."
            )
        training, next_training = codec.read_training(training_value.payload)
        target, seed = context.params["target"], context.params["seed"]
        if (
            not training["models"]
            or training["models"][-1]["target"] != target
            or training["models"][-1]["sha256"] != codec.sha(model_value.payload)
        ):
            raise ValueError("Model bytes/target differ from the training checkpoint.")
        # Only graph-verified sibling outputs make this trusted model transport.
        # A caller-supplied digest alone cannot authenticate arbitrary pickle.
        fitted = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
            model_value.payload, expected_sha256=training["models"][-1]["sha256"]
        )
        if (
            fitted.next_training_state != next_training
            or fitted.training_id != training["models"][-1]["training_id"]
        ):
            raise ValueError("Model envelope differs from its training checkpoint.")
        matrix_value = codec.artifact(
            context, "matrix", model_input.RECIPIENT_MATRIX_TYPE
        )
        matrix = model_input.decode_recipient_matrix(matrix_value.payload)
        binding = {
            "matrix_sha256": codec.sha(matrix_value.payload),
            "matrix_producer_key": matrix_value.producer_key,
        }
        before = fitted.training_state.to_dict()
        if matrix.entity != before["entity"] or tuple(matrix.features) != tuple(
            before["predictors"]
        ):
            raise ValueError(
                "Recipient matrix differs from the fitted entity/predictors."
            )
        prior_targets = before["completed_targets"]
        expected = {"model", "training_state", "matrix"} | (
            {"observed"} if "observed" in edges else set()
        )
        if prior_targets:
            expected |= {
                "apply_state",
                *(f"prior_{i:03d}" for i in range(len(prior_targets))),
            }
        if set(edges) != expected or set(context.artifacts) != expected:
            raise ValueError("Observed apply requires the exact conditioning prefix.")
        if prior_targets:
            previous_value = codec.artifact(
                context, "apply_state", OBSERVED_MATRIX_APPLY_STATE_TYPE
            )
            previous = decode_observed_matrix_apply_state(previous_value.payload)
            if any(previous[k] != v for k, v in binding.items()):
                raise ValueError(
                    "Recipient matrix bytes/producer changed within the chain."
                )
            application = previous["application"]
            state = qrf.QRFChainState.from_dict(application["state"])
            if (
                application["seed"] != seed
                or application["models"] != training["models"][:-1]
                or tuple(state.completed_targets) != tuple(prior_targets)
            ):
                raise ValueError("Observed application seed/model history changed.")
            history = application["raw_targets"]
        else:
            _, draw_seed = np.random.SeedSequence(seed).spawn(2)
            state = qrf.QRFChainState.from_dict(
                {
                    **before,
                    "recipient_index": None,
                    "draw_rng_state": np.random.default_rng(
                        draw_seed
                    ).bit_generator.state,
                }
            )
            history = []
        prior, producer_keys = pd.DataFrame(index=matrix.features.index), []
        for i, name in enumerate(prior_targets):
            value = codec.artifact(context, f"prior_{i:03d}", CONDITIONING_TARGET_TYPE)
            if codec.sha(value.payload) != history[i]["conditioning_sha256"]:
                raise ValueError("Conditioning bytes differ from recorded history.")
            producer_keys.append(value.producer_key)
            prior[name] = codec.read_raw_target(
                value.payload, target=name, index=matrix.features.index
            )
        if prior_targets and (
            producer_keys[:-1] != application["prior_producer_keys"]
            or producer_keys[-1] != previous_value.producer_key
            or edges[f"prior_{len(prior_targets) - 1:03d}"].producer
            != edges["apply_state"].producer
        ):
            raise ValueError("Conditioning prefix producer lineage changed.")
        observation = None
        if "observed" in edges:
            observation = codec.artifact(context, "observed", OBSERVED_TARGET_TYPE)
            metadata, supplied, known = _read_observed(
                observation.payload,
                target=target,
                index=matrix.features.index,
                **binding,
            )
        # Always perform the actual full-row draw before considering knownness.
        result = qrf_target.apply_target(fitted, matrix.features, prior, state=state)
        raw_payload = codec.encode_raw_target(
            result.raw_draw, target=target, index=matrix.features.index
        )
        conditioning = raw_payload
        if observation is not None:
            merged = result.raw_draw.copy()
            merged[known] = supplied[known]
            conditioning = codec.encode_raw_target(
                merged, target=target, index=matrix.features.index
            )
        record = {
            "target": target,
            "draw_sha256": codec.sha(raw_payload),
            "conditioning_sha256": codec.sha(conditioning),
            "model_producer_key": model_value.producer_key,
            "observed_sha256": None
            if observation is None
            else codec.sha(observation.payload),
            "observed_producer_key": None
            if observation is None
            else observation.producer_key,
            "source_sha256": None if observation is None else metadata["source_sha256"],
            "observed_rows": 0 if observation is None else int(known.sum()),
        }
        packet = {
            "schema_version": 2,
            **binding,
            "application": {
                "schema_version": 2,
                "state": result.state.to_dict(),
                "seed": seed,
                "models": training["models"],
                "raw_targets": [*history, record],
                "prior_producer_keys": producer_keys,
            },
        }
        state_payload = codec.encode_json(packet)
        decode_observed_matrix_apply_state(state_payload)
        return KernelResult(
            artifacts={
                "raw_draw": raw_payload,
                "conditioning": conditioning,
                "apply_state": state_payload,
            },
            receipt={
                "phase": context.params["phase"],
                "target": target,
                "entity": matrix.entity,
                "recipient_rows": len(matrix.features),
                **binding,
                **record,
                "regime": result.regime,
            },
        )
