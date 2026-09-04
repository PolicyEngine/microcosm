"""Train reusable QRF models independently of application adapter code."""

from __future__ import annotations

import microcosm.fit._graph_qrf as shared_module
import microcosm.fit.model as fit_model_module
import microcosm.fit.qrf as qrf_module
from microcosm.fit import fit as fit_qrf
from microcosm.fit._graph_qrf import QRF_MODEL_TYPE, _encode_model, _names, _table
from microcosm.fit.kernels import FIT_QRF_DEPENDENCIES
from microcosm.fit.qrf import DEFAULT_N_ESTIMATORS, DEFAULT_ZERO_ATOL
from microcosm.graph import (
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)

_TRAIN_PARAMS = {
    "predictors",
    "targets",
    "seed",
    "n_estimators",
    "zero_atol",
    "max_samples_leaf",
}


class QRFTrainKernel(KernelBase):
    """Fit a weighted donor model; produce no recipient columns or draws."""

    ref = "fit.qrf.train@1"
    capabilities = Capabilities(
        Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        dependencies=FIT_QRF_DEPENDENCIES,
    )

    def implementation_hash(self):
        return source_hash(
            type(self),
            shared_module,
            fit_qrf,
            fit_model_module,
            qrf_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        declared, table, _ = _table(context, self.ref)
        node = context.node
        if (
            node.outputs
            or node.artifact_inputs
            or node.artifact_outputs != (ArtifactOutput("model", QRF_MODEL_TYPE),)
        ):
            raise ValueError("QRF training owns exactly one typed model artifact.")
        if set(context.params) - _TRAIN_PARAMS:
            raise ValueError("Unknown QRF training parameters.")
        predictors = _names(context.params.get("predictors"), "predictors")
        targets = _names(context.params.get("targets"), "targets")
        if set(predictors) & set(targets) or declared.columns != (
            *predictors,
            *targets,
        ):
            raise ValueError(
                "Training Slice must contain predictors then disjoint targets."
            )
        seed = context.params.get("seed")
        trees = context.params.get("n_estimators", DEFAULT_N_ESTIMATORS)
        if type(seed) is not int or seed < 0 or type(trees) is not int or trees < 1:
            raise ValueError(
                "QRF training requires a nonnegative seed and positive tree count."
            )
        weights = context.weights[declared.entity]
        if len(weights) != len(table):
            raise ValueError("QRF donor weights must align to the input rows.")
        support = weights.values > 0
        if not support.any():
            raise ValueError("QRF training requires positive donor weight mass.")
        # Zero-mass rows are outside this training population's support. Filter
        # before regime detection to avoid bootstrapping an empty sign class.
        donor = table.loc[support, [*predictors, *targets]].copy()
        model = fit_qrf(
            donor,
            list(predictors),
            list(targets),
            weights=weights.values[support],
            n_estimators=trees,
            zero_atol=context.params.get("zero_atol", DEFAULT_ZERO_ATOL),
            max_samples_leaf=context.params.get("max_samples_leaf"),
            seed=seed,
        )
        return KernelResult(
            artifacts={"model": _encode_model(model, weights.kind.value)},
            receipt={
                "entity": declared.entity,
                "predictors": predictors,
                "targets": targets,
                "donor_rows": len(donor),
                "excluded_zero_weight_rows": int((~support).sum()),
                "donor_weight_sum": float(weights.values[support].sum()),
                "source_weight_kind": weights.kind.value,
                "fit_weight_kind": model.weight_kind,
                "regimes": model.regimes(),
                "seed": seed,
                "n_estimators": trees,
                "seed_source": "param",
            },
        )
