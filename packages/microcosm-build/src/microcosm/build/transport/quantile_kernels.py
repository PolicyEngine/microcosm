"""``transport.quantile_map@2``: a quantile map whose open top fits its own alpha.

The kernel estimates the Pareto alpha of the open top band from the declared
anchor facts on the same compiled band surface the map's shares come from,
then maps exactly as ``transport.quantile_map@1`` does with that alpha as a
literal. It delegates the map to ``@1``'s own ``run``, so no second
interpolation path exists, and it lives in its own module because
``source_hash`` digests a kernel's whole defining module: a class added to
``population_kernels`` would re-key ``@1``.

The interpolation parameter is ``{"method": "pareto", "estimator": {"id":
"grouped_pareto_mle@1", "anchor_references": [...], "bracket": [low, high],
"iterations": n}}``. The anchor references are the top contiguous suffix of
the band list, ending with the open top band, and their counts are the
surface facts the band rows reference. The receipt records the fit.
"""

from __future__ import annotations

from dataclasses import replace

import microcosm.build.tail_estimators as tail_estimators_module
import microcosm.frame.transport as transport_module
from microcosm.build.tail_estimators import (
    grouped_pareto_mle,
    parse_grouped_pareto_declaration,
)
from microcosm.graph import (
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import graph_inputs, population_kernels, target_kernels
from .artifact_types import TARGET_SURFACE_TYPE
from .graph_inputs import canonical_document_param, require_params, string_param
from .population_kernels import TRANSPORT_QUANTILE_MAP, _surface_bands
from .target_kernels import decode_target_surface

__all__ = [
    "QUANTILE_MAP_V2_REF",
    "TRANSPORT_QUANTILE_MAP_V2",
    "TransportQuantileMapV2Kernel",
    "estimated_alpha",
    "register_quantile_kernels",
]

QUANTILE_MAP_V2_REF = "transport.quantile_map@2"


def estimated_alpha(context: KernelContext, ref: str = QUANTILE_MAP_V2_REF):
    """Fit the declared estimator to the anchor counts on the node's surface.

    Returns the fit and the literal interpolation document ``@1`` reads.
    """

    interpolation = canonical_document_param(context, ref, "interpolation")
    if set(interpolation) != {"method", "estimator"} or (
        interpolation["method"] != "pareto"
    ):
        raise ValueError(
            f"{ref} interpolation is {{method: pareto, estimator: ...}}; a literal "
            "alpha belongs to transport.quantile_map@1."
        )
    references, bracket, iterations = parse_grouped_pareto_declaration(
        interpolation["estimator"]
    )
    document = canonical_document_param(context, ref, "bands")
    if set(document) != {"bands"}:
        raise ValueError(
            f"{ref} fits one alpha to a bare band list; signed or component "
            "bands are refused."
        )
    artifact = context.artifacts.get("surface")
    if artifact is None or artifact.type != TARGET_SURFACE_TYPE:
        raise ValueError(f"{ref} requires a target surface under alias 'surface'.")
    surface = decode_target_surface(artifact.payload)
    entity = string_param(context, ref, "entity")
    unit = (
        string_param(context, ref, "aggregate_entity")
        if "aggregate_entity" in context.params
        else entity
    )
    # The same validation and the same facts the map's shares use.
    _surface_bands(document, surface, unit=unit)
    rows = document["bands"]
    if [row["reference"] for row in rows[-len(references) :]] != list(references):
        raise ValueError(
            f"{ref} anchor_references must be the top contiguous bands of the "
            "band list, in order, ending with the open top."
        )
    specs = {spec.name: spec for spec in surface.registry.specs}
    fit = grouped_pareto_mle(
        [
            (
                row["reference"],
                row["lower"],
                row["upper"],
                specs[row["reference"]].value,
            )
            for row in rows[-len(references) :]
        ],
        bracket=bracket,
        iterations=iterations,
    )
    literal = canonical_json({"method": "pareto", "alpha": fit.alpha_float}).decode(
        "utf-8"
    )
    return fit, literal


class TransportQuantileMapV2Kernel(KernelBase):
    """``transport.quantile_map@2``: ``@1`` with a fitted open-top alpha."""

    ref = QUANTILE_MAP_V2_REF
    capabilities = TRANSPORT_QUANTILE_MAP.capabilities

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            tail_estimators_module,
            population_kernels,
            graph_inputs,
            target_kernels,
            transport_module,
            *target_kernels._COMPILE_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context,
            self.ref,
            required=frozenset(
                {"entity", "column", "bands", "interpolation", "bands_sha256"}
            ),
            optional=frozenset({"component_column", "aggregate_entity"}),
        )
        fit, literal = estimated_alpha(context, self.ref)
        delegated = replace(
            context, params={**context.params, "interpolation": literal}
        )
        result = TRANSPORT_QUANTILE_MAP.run(delegated)
        return replace(
            result, receipt={**result.receipt, "tail_estimator": fit.receipt()}
        )


TRANSPORT_QUANTILE_MAP_V2 = TransportQuantileMapV2Kernel()


def register_quantile_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Register ``transport.quantile_map@2`` explicitly and idempotently."""
    registry.register(TRANSPORT_QUANTILE_MAP_V2)
    return registry
