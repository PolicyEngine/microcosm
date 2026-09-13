"""Graph owner of the additive complete PUF return-source artifact."""

from __future__ import annotations

import hashlib
from importlib import import_module

from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import load_source_bytes

from . import puf_full_source as full
from . import puf_raw_source as raw

FULL_RETURN_SOURCE_TYPE = ArtifactType("microcosm.us.puf_2015_full_return_source", 3)


class FullPufReturnSourceKernel(KernelBase):
    ref = "us.puf.full_return_source@3"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.BITWISE, dependencies=("numpy",)
    )

    def __init__(self, *, definition=None, source_codecs=None):
        self.definition = (
            raw.packaged_definition() if definition is None else definition
        )
        self.source_codecs = (
            raw.puf_raw_source_codecs(self.definition)
            if source_codecs is None
            else source_codecs
        )

    def implementation_hash(self):
        base = source_hash(
            import_module(__name__),
            full,
            raw,
            dependencies=self.capabilities.dependencies,
        )
        return hashlib.sha256(
            base.encode("ascii") + b"\0" + canonical_json(raw.csv_acceptance_profile())
        ).hexdigest()

    def run(self, context):
        d = self.definition
        if (
            context.node.kernel != self.ref
            or context.node.structural is not StructuralDelta.NONE
            or context.node.sources != (d.main.source_name, d.demographic.source_name)
            or context.node.inputs
            or context.node.outputs
            or context.node.artifact_inputs
            or context.node.artifact_outputs
            != (ArtifactOutput("full_return_source", FULL_RETURN_SOURCE_TYPE),)
            or dict(context.params) != {"definition": d.params_text}
        ):
            raise ValueError("FULL_SOURCE_NODE_DECLARATION")
        main = load_source_bytes(
            d.main.codec,
            context.sources[d.main.source_name],
            registry=self.source_codecs,
        )
        demo = load_source_bytes(
            d.demographic.codec,
            context.sources[d.demographic.source_name],
            registry=self.source_codecs,
        )
        decoded = full.decode_full_puf_source(main, demo, d)
        payload = full.encode_full_puf_source(decoded)
        return KernelResult(
            artifacts={"full_return_source": payload},
            receipt={
                "full_puf_return_source": {
                    "definition_sha256": d.sha256,
                    "definition_route": d.route,
                    "source_sha256": dict(decoded.source_sha256),
                    "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                    "artifact_bytes": len(payload),
                    "rows": len(decoded.status["RECID"]),
                    "ordinary_rows": int(decoded.ordinary.sum()),
                    "projected_monetary_fields": len(full.MONEY_COLUMNS),
                    "projected_count_fields": len(full.COUNT_COLUMNS),
                    "amount_units": "source_whole_us_dollars",
                    "weight_units": "S006_integer_hundredths",
                    "period": "raw_FLPDYR_preserved",
                    "canonical_modeling": False,
                    "target_year_growth": False,
                    "donor_admission": False,
                }
            },
        )


def full_puf_return_source_node(
    *, population, definition=None, node_id="us_puf_full_source.return_source"
):
    if not isinstance(population, str) or not population:
        raise ValueError("FULL_SOURCE_POPULATION")
    d = raw.packaged_definition() if definition is None else definition
    return Node(
        node_id,
        FullPufReturnSourceKernel.ref,
        population=population,
        sources=(d.main.source_name, d.demographic.source_name),
        params={"definition": d.params_text},
        artifact_outputs=(
            ArtifactOutput("full_return_source", FULL_RETURN_SOURCE_TYPE),
        ),
        description="Authenticate and type 52 monetary fields, seven exemption/dependent count fields, and complete PUF source status; no model or growth.",
    )
