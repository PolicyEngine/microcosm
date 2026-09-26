"""Apply one frozen legacy QRF target with exact raw-prefix/RNG progression."""

from microcosm.fit import _graph_legacy_apply as shared
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import qrf, qrf_target
from microcosm.fit.kernels import FIT_QRF_DEPENDENCIES
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    Numeric,
    SeedSource,
    source_hash,
)


class LegacyQRFApplyKernel(KernelBase):
    ref = "fit.qrf.legacy_target.apply@1"
    capabilities = Capabilities(
        Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        dependencies=FIT_QRF_DEPENDENCIES,
    )

    def implementation_hash(self):
        return source_hash(
            type(self),
            shared,
            codec,
            qrf_target,
            qrf,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        declared, table = codec.table(context, self.ref)
        return shared.run_application(
            context, entity=declared.entity, predictors=declared.columns, table=table
        )
