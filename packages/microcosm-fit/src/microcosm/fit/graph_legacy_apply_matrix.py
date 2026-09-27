"""Apply the shared legacy protocol to a derived, identity-bound matrix."""

from microcosm.fit import _graph_legacy_apply as shared
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.fit.kernels import FIT_QRF_DEPENDENCIES
from microcosm.graph import (
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)

MATRIX_APPLY_STATE_TYPE = ArtifactType("microcosm.fit.legacy_matrix_apply_state", 1)


def decode_matrix_apply_state(payload: bytes) -> dict:
    """Validate the wrapper and unchanged legacy packet for a downstream owner.

    A finalizer must additionally bind the matrix SHA and actual producer key
    to its placement edge and verify the last raw SHA against its raw artifact.
    This parser alone does not grant population or placement authority.
    """
    packet = codec.decode_json(payload)
    if (
        set(packet)
        != {"schema_version", "matrix_sha256", "matrix_producer_key", "application"}
        or type(packet["schema_version"]) is not int
        or packet["schema_version"] != 1
        or not codec._hash(packet["matrix_sha256"])
        or not codec._hash(packet["matrix_producer_key"])
    ):
        raise ValueError("Invalid matrix application checkpoint.")
    _, state = codec.read_application(codec.encode_json(packet["application"]))
    if not state.completed_targets:
        raise ValueError("Matrix application checkpoint has no completed target.")
    return packet


class LegacyQRFApplyMatrixKernel(KernelBase):
    ref = "fit.qrf.legacy_target.apply_matrix@1"
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
            model_input,
            qrf_target,
            qrf,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        node = context.node
        if node.kernel != self.ref or node.inputs or node.outputs or node.sources:
            raise ValueError(
                "Matrix apply reads typed artifacts only, with no Slice or source."
            )
        if (
            not isinstance(context.params.get("phase"), str)
            or not context.params["phase"]
        ):
            raise ValueError("Matrix apply requires its explicit outer phase.")
        value = codec.artifact(context, "matrix", model_input.RECIPIENT_MATRIX_TYPE)
        matrix = model_input.decode_recipient_matrix(value.payload)
        binding = {
            "matrix_sha256": codec.sha(value.payload),
            "matrix_producer_key": value.producer_key,
        }

        def read_state(payload):
            packet = decode_matrix_apply_state(payload)
            if any(packet[key] != expected for key, expected in binding.items()):
                raise ValueError(
                    "Matrix application input bytes/producer changed within the chain."
                )
            return codec.read_application(codec.encode_json(packet["application"]))

        def write_state(application):
            return codec.encode_json(
                {"schema_version": 1, **binding, "application": application}
            )

        result = shared.run_application(
            context,
            entity=matrix.entity,
            predictors=tuple(matrix.features.columns),
            table=matrix.features,
            state_type=MATRIX_APPLY_STATE_TYPE,
            state_decoder=read_state,
            state_encoder=write_state,
            extra_inputs=frozenset({"matrix"}),
        )
        return KernelResult(
            artifacts=result.artifacts, receipt={**result.receipt, **binding}
        )
