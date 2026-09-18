"""Typed byte edges: numeric refusal, portable descriptors, and their values.

The declaration side lives in :mod:`microcosm.graph.decl` (``ArtifactType``,
``ArtifactOutput``, ``ArtifactInput``) and the kernel side in
:mod:`microcosm.graph.kernel` (``ArtifactValue``). This module is the
executor-side glue between them: it derives the numeric scope a producer's
bytes carry, refuses an edge that would launder that scope, renders the
portable descriptor a receipt records, and rebuilds a verified
``ArtifactValue`` from one (amendment 19).
"""

from __future__ import annotations

from collections.abc import Mapping

from .decl import ArtifactType, CompiledGraph, Node
from .errors import NodeRejectedError
from .kernel import (
    ArtifactValue,
    Capabilities,
    KernelRegistry,
    Numeric,
    NumericScope,
    Tolerance,
)
from .keys import opaque_artifact_key, platform_fingerprint

__all__ = [
    "descriptor",
    "numeric_scope",
    "require_compatible_scope",
    "scope_from_payload",
    "scope_payload",
    "typed_contracts",
    "value_from_descriptor",
]


def numeric_scope(capabilities: Capabilities) -> NumericScope:
    """The scope a producer's bytes carry, from its declared capabilities."""

    return NumericScope(
        numeric=capabilities.numeric,
        tolerance=capabilities.tolerance,
        platform=platform_fingerprint()
        if capabilities.numeric is Numeric.PLATFORM_BITWISE
        else None,
    )


def scope_payload(scope: NumericScope) -> dict[str, object]:
    """Render a scope as the canonical JSON a receipt records."""

    return {
        "numeric": scope.numeric.value,
        "tolerance": None
        if scope.tolerance is None
        else {
            "rtol": scope.tolerance.rtol,
            "atol": scope.tolerance.atol,
            "ulps": scope.tolerance.ulps,
        },
        "platform": scope.platform,
    }


def scope_from_payload(raw: object) -> NumericScope:
    """Rebuild a scope from its recorded payload, refusing a malformed one."""

    if not isinstance(raw, Mapping) or set(raw) != {"numeric", "tolerance", "platform"}:
        raise ValueError("Malformed artifact numeric scope.")
    tolerance = raw["tolerance"]
    if tolerance is not None:
        if not isinstance(tolerance, Mapping) or set(tolerance) != {
            "rtol",
            "atol",
            "ulps",
        }:
            raise ValueError("Malformed artifact tolerance.")
        tolerance = Tolerance(**tolerance)
    return NumericScope(
        Numeric(raw["numeric"]), tolerance=tolerance, platform=raw["platform"]
    )


def require_compatible_scope(scope: NumericScope, consumer: Capabilities) -> None:
    """Refuse scope laundering across a byte edge.

    A cell's numeric class reaches its reader through
    ``KernelContext.numerics`` (amendment 17), which a gate consults. Opaque
    bytes carry no per-cell coordinates, so the only honest rule is that a
    consumer of scoped bytes declares the same class: a platform-bitwise
    payload may not be read by a kernel claiming to hold on every platform,
    and a tolerance-bound payload may not be read by one claiming bitwise
    reproduction. The consumer's own declared tolerance remains its own
    output contract.
    """

    if scope.platform is not None and scope.numeric is Numeric.TOLERANCE_BOUND:
        raise NodeRejectedError(
            "Typed artifacts combining platform and tolerance scopes are unsupported."
        )
    if (
        scope.numeric is Numeric.PLATFORM_BITWISE
        and consumer.numeric is not Numeric.PLATFORM_BITWISE
    ):
        raise NodeRejectedError(
            "A platform_bitwise artifact requires a platform_bitwise consumer."
        )
    if (
        scope.numeric is Numeric.TOLERANCE_BOUND
        and consumer.numeric is not Numeric.TOLERANCE_BOUND
    ):
        raise NodeRejectedError(
            "A tolerance_bound artifact requires a tolerance_bound consumer."
        )


def descriptor(
    *,
    producer: str,
    artifact: str,
    type_: ArtifactType,
    producer_key: str,
    capabilities: Capabilities,
) -> dict[str, object]:
    """The portable provenance of one typed artifact, as canonical JSON."""

    return {
        "producer": producer,
        "artifact": artifact,
        "producer_key": producer_key,
        "key": opaque_artifact_key(producer_key, artifact),
        "type": {"name": type_.name, "schema_version": type_.schema_version},
        "numerics": scope_payload(numeric_scope(capabilities)),
    }


def typed_contracts(
    compiled: CompiledGraph,
    node: Node,
    keys: Mapping[str, str],
    kernels: KernelRegistry,
) -> dict[str, object]:
    """Every typed edge of one node, resolved against the compiled graph.

    Empty for a node that declares none, so nothing is recorded and no
    cached record shape changes for a graph that predates amendment 19.
    """

    if not node.artifact_inputs and not node.artifact_outputs:
        return {}
    consumer = kernels.get(node.kernel).capabilities
    inputs = {}
    for binding in node.artifact_inputs:
        producer = compiled.graph.node(binding.producer)
        capabilities = kernels.get(producer.kernel).capabilities
        require_compatible_scope(numeric_scope(capabilities), consumer)
        inputs[binding.name] = descriptor(
            producer=producer.id,
            artifact=binding.artifact,
            type_=binding.type,
            producer_key=keys[producer.id],
            capabilities=capabilities,
        )
    return {
        "inputs": inputs,
        "outputs": {
            output.name: descriptor(
                producer=node.id,
                artifact=output.name,
                type_=output.type,
                producer_key=keys[node.id],
                capabilities=consumer,
            )
            for output in node.artifact_outputs
        },
    }


def value_from_descriptor(payload: bytes, raw: object) -> ArtifactValue:
    """Rebuild a verified :class:`ArtifactValue` from a recorded descriptor."""

    if not isinstance(raw, Mapping) or set(raw) != {
        "producer",
        "artifact",
        "producer_key",
        "key",
        "type",
        "numerics",
    }:
        raise ValueError("Malformed typed artifact descriptor.")
    if any(
        not isinstance(raw[name], str) or not raw[name]
        for name in ("producer", "artifact")
    ):
        raise ValueError("Typed artifact producer/name must be nonempty strings.")
    type_raw = raw["type"]
    if not isinstance(type_raw, Mapping) or set(type_raw) != {"name", "schema_version"}:
        raise ValueError("Malformed typed artifact type.")
    value = ArtifactValue(
        payload,
        ArtifactType(type_raw["name"], type_raw["schema_version"]),
        raw["key"],
        raw["producer_key"],
        scope_from_payload(raw["numerics"]),
    )
    if value.key != opaque_artifact_key(value.producer_key, raw["artifact"]):
        raise ValueError("Typed artifact identity does not match its producer.")
    return value
