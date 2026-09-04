"""Apply reusable QRF models independently of training adapter code."""

from __future__ import annotations

import numpy as np
import pandas as pd

import microcosm.fit._graph_qrf as shared_module
import microcosm.fit.qrf as qrf_module
import microcosm.graph.canonical as canonical_module
import microcosm.graph.randomness as randomness_module
from microcosm.fit._graph_qrf import QRF_MODEL_TYPE, _table, load_qrf_model
from microcosm.fit.kernels import FIT_QRF_DEPENDENCIES
from microcosm.graph import (
    ROWS_ALL,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Numeric,
    SeedSource,
    keyed_uniform,
    source_hash,
)

_APPLY_PARAMS = {"random_stream", "period"}


class QRFApplyKernel(KernelBase):
    """Reuse one trained model and draw targets at stable recipient coordinates."""

    ref = "fit.qrf.apply@1"
    capabilities = Capabilities(
        Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.KEYED,
        dependencies=FIT_QRF_DEPENDENCIES,
    )

    def implementation_hash(self):
        return source_hash(
            type(self),
            shared_module,
            qrf_module,
            randomness_module,
            canonical_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        declared, table, id_column = _table(context, self.ref)
        node = context.node
        if (
            len(node.artifact_inputs) != 1
            or node.artifact_inputs[0].name != "model"
            or node.artifact_inputs[0].type != QRF_MODEL_TYPE
            or node.artifact_outputs
            or set(context.artifacts) != {"model"}
        ):
            raise ValueError("QRF apply requires exactly one typed model input.")
        if set(context.params) != _APPLY_PARAMS:
            raise ValueError(
                "QRF apply requires only random_stream and period parameters."
            )
        period = context.params["period"]
        if type(period) is not int:
            raise ValueError("QRF apply period must be an integer.")
        stream = context.params["random_stream"]
        # Validate even an empty recipient table before loading the model.
        keyed_uniform(stream=stream, keys=[])
        artifact = context.artifacts["model"]
        model = load_qrf_model(artifact)
        if declared.columns != tuple(model.predictors):
            raise ValueError("QRF apply Slice does not match the model predictors.")
        if tuple(o.column for o in node.outputs) != tuple(model.targets) or any(
            o.entity != declared.entity or o.dtype != "float64" or o.rows != ROWS_ALL
            for o in node.outputs
        ):
            raise ValueError(
                "QRF apply must own every model target as all-row float64."
            )
        arrays = {
            kind: {
                target: keyed_uniform(
                    stream=stream,
                    keys=[(i, "qrf", target, period, kind) for i in table[id_column]],
                )
                for target in model.targets
            }
            for kind in ("quantiles", "sign_uniforms")
        }
        drawn = model.predict_from_uniforms(table, **arrays)
        index = pd.Index(table[id_column].to_numpy(copy=True), name=id_column)
        return KernelResult(
            columns={
                (declared.entity, target): pd.Series(
                    drawn[target].to_numpy(dtype=np.float64, copy=True),
                    index=index,
                    name=target,
                    dtype="float64",
                )
                for target in model.targets
            },
            receipt={
                "entity": declared.entity,
                "recipient_rows": len(table),
                "model_key": artifact.key,
                "model_producer_key": artifact.producer_key,
                "random_stream": stream,
                "period": period,
                "seed_source": "keyed",
            },
        )
