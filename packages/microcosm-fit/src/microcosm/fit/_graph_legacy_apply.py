"""One shared legacy application body for Slice and typed-matrix adapters."""

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import qrf
from microcosm.fit.qrf_target import (
    LEGACY_QRF_TARGET_TYPE,
    LegacyQRFTargetArtifact,
    apply_target,
)
from microcosm.graph import ArtifactOutput, KernelResult


def run_application(
    context,
    *,
    entity,
    predictors,
    table,
    state_type=codec.APPLY_STATE_TYPE,
    state_decoder=codec.read_application,
    state_encoder=codec.encode_json,
    extra_inputs=frozenset(),
):
    if set(context.params) != {"target", "seed", "phase"}:
        raise ValueError(
            "Legacy apply requires exact target, seed and phase parameters."
        )
    seed = context.params["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Legacy apply seed must be nonnegative.")
    if context.node.artifact_outputs != (
        ArtifactOutput("raw_draw", codec.RAW_TARGET_TYPE),
        ArtifactOutput("apply_state", state_type),
    ):
        raise ValueError(
            "Legacy apply owns one raw target and a small application state."
        )
    model_value = codec.artifact(context, "model", LEGACY_QRF_TARGET_TYPE)
    training_value = codec.artifact(
        context, "training_state", codec.TRAINING_STATE_TYPE
    )
    input_by_name = {item.name: item for item in context.node.artifact_inputs}
    if (
        input_by_name["model"].producer != input_by_name["training_state"].producer
        or model_value.producer_key != training_value.producer_key
    ):
        raise ValueError("Legacy model and training state must share one fit producer.")
    training, next_training = codec.read_training(training_value.payload)
    target = context.params["target"]
    if (
        not training["models"]
        or training["models"][-1]["target"] != target
        or training["models"][-1]["sha256"] != codec.sha(model_value.payload)
    ):
        raise ValueError(
            "Legacy model bytes/target disagree with the training checkpoint."
        )
    # Expected SHA is taken from the independently verified sibling output.
    # The graph store/producer is trusted; a digest cannot make foreign pickle safe.
    fitted = LegacyQRFTargetArtifact.from_trusted_bytes(
        model_value.payload, expected_sha256=training["models"][-1]["sha256"]
    )
    if (
        fitted.next_training_state != next_training
        or fitted.training_id != training["models"][-1]["training_id"]
    ):
        raise ValueError("Legacy model training envelope differs from its checkpoint.")
    before = fitted.training_state.to_dict()
    if entity != before["entity"] or predictors != tuple(before["predictors"]):
        raise ValueError(
            "Legacy recipient Slice differs from the model entity/predictors."
        )
    prior_targets = before["completed_targets"]
    expected_inputs = {"model", "training_state"} | set(extra_inputs)
    if prior_targets:
        expected_inputs.add("apply_state")
        expected_inputs.update(f"prior_{i:03d}" for i in range(len(prior_targets)))
    if (
        set(input_by_name) != expected_inputs
        or set(context.artifacts) != expected_inputs
    ):
        raise ValueError(
            "Legacy apply inputs must contain the exact ordered raw prefix."
        )
    if prior_targets:
        previous, state = state_decoder(
            codec.artifact(context, "apply_state", state_type).payload
        )
        if previous["seed"] != seed or previous["models"] != training["models"][:-1]:
            raise ValueError("Legacy application seed/model history changed.")
        raw_history = previous["raw_targets"]
    else:
        # Exactly start_chain's second SeedSequence stream, not keyed draws.
        _, draw_seed = np.random.SeedSequence(seed).spawn(2)
        state = qrf.QRFChainState.from_dict(
            {
                **before,
                "recipient_index": None,
                "draw_rng_state": np.random.default_rng(draw_seed).bit_generator.state,
            }
        )
        raw_history = []
    raw = pd.DataFrame(index=table.index)
    for i, prior in enumerate(prior_targets):
        value = codec.artifact(context, f"prior_{i:03d}", codec.RAW_TARGET_TYPE)
        if codec.sha(value.payload) != raw_history[i]["sha256"]:
            raise ValueError("Legacy raw prior bytes differ from application history.")
        raw[prior] = codec.read_raw_target(
            value.payload, target=prior, index=table.index
        )
    result = apply_target(fitted, table, raw, state=state)
    payload = codec.encode_raw_target(result.raw_draw, target=target, index=table.index)
    next_state = state_encoder(
        {
            "schema_version": 1,
            "state": result.state.to_dict(),
            "seed": seed,
            "models": training["models"],
            "raw_targets": [
                *raw_history,
                {"target": target, "sha256": codec.sha(payload)},
            ],
        }
    )
    return KernelResult(
        artifacts={"raw_draw": payload, "apply_state": next_state},
        receipt={
            "phase": context.params["phase"],
            "target": target,
            "recipient_rows": len(table),
            "entity": entity,
            "model_sha256": codec.sha(model_value.payload),
            "raw_sha256": codec.sha(payload),
            "regime": result.regime,
        },
    )
