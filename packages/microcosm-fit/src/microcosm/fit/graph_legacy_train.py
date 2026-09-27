"""Donor-only fitted legacy QRF targets with no recipient graph dependency."""

from __future__ import annotations

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model as weight_module
from microcosm.fit import qrf, qrf_target
from microcosm.fit.kernels import FIT_QRF_DEPENDENCIES
from microcosm.fit.qrf_target import (
    LEGACY_QRF_TARGET_TYPE,
    LegacyQRFTrainingState,
    fit_target,
)
from microcosm.graph import (
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)


class LegacyQRFTrainKernel(KernelBase):
    ref = "fit.qrf.legacy_target.train@1"
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
            qrf_target,
            qrf,
            weight_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        declared, table = codec.table(context, self.ref)
        expected_params = {
            "predictors",
            "targets",
            "target",
            "seed",
            "n_estimators",
            "zero_atol",
            "max_samples_leaf",
            "phase",
        }
        if set(context.params) != expected_params:
            raise ValueError(
                "Legacy QRF training parameters must be explicit and exact."
            )
        predictors = codec.names(context.params["predictors"], "predictors")
        targets = codec.names(context.params["targets"], "targets")
        if set(predictors) & set(targets) or declared.columns != (
            *predictors,
            *targets,
        ):
            raise ValueError(
                "Legacy training Slice must contain predictors then disjoint ordered targets."
            )
        if context.node.artifact_outputs != (
            ArtifactOutput("model", LEGACY_QRF_TARGET_TYPE),
            ArtifactOutput("training_state", codec.TRAINING_STATE_TYPE),
        ):
            raise ValueError(
                "Legacy training owns model and small training-state artifacts."
            )
        names = {item.name for item in context.node.artifact_inputs}
        if names not in (set(), {"training_state"}) or set(context.artifacts) != names:
            raise ValueError(
                "Legacy training accepts only the prior training checkpoint."
            )
        seed, trees = context.params["seed"], context.params["n_estimators"]
        if type(seed) is not int or seed < 0 or type(trees) is not int or trees < 1:
            raise ValueError(
                "Legacy fit requires nonnegative seed and positive tree count."
            )
        model = qrf.RegimeGatedQRF(
            seed=seed,
            n_estimators=trees,
            zero_atol=context.params["zero_atol"],
            max_samples_leaf=context.params["max_samples_leaf"],
        )
        donor = codec.model_frame(context, declared, table)
        weight_kind = context.weights[declared.entity].kind.value
        if names:
            packet, state = codec.read_training(
                codec.artifact(
                    context, "training_state", codec.TRAINING_STATE_TYPE
                ).payload
            )
            models = packet["models"]
        else:
            state = LegacyQRFTrainingState.from_chain(
                model.start_chain(
                    donor, list(predictors), list(targets), weights=weight_kind
                )
            )
            models = []
        state_values = state.to_dict()
        if (
            tuple(state_values["predictors"]) != predictors
            or tuple(state_values["targets"]) != targets
            or state.next_target != context.params["target"]
        ):
            raise ValueError(
                "Legacy training state/target order differs from declaration."
            )
        fitted = fit_target(model, donor, state=state, weights=weight_kind)
        payload = fitted.to_bytes()
        next_models = [
            *models,
            {
                "target": fitted.target,
                "sha256": codec.sha(payload),
                "training_id": fitted.training_id,
            },
        ]
        next_state = codec.encode_json(
            {
                "schema_version": 1,
                "state": fitted.next_training_state.to_dict(),
                "models": next_models,
            }
        )
        return KernelResult(
            artifacts={"model": payload, "training_state": next_state},
            receipt={
                "phase": context.params["phase"],
                "target": fitted.target,
                "training_id": fitted.training_id,
                "model_sha256": codec.sha(payload),
                "donor_rows": len(table),
                "entity": declared.entity,
                "weight_kind": weight_kind,
                "regime": fitted.regime,
            },
        )
