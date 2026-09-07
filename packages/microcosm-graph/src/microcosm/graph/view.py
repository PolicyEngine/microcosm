"""Compact human-readable views of compiled graph nodes."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from .canonical import canonical_json
from .decl import CompiledGraph, StructuralDelta
from .manifest import RunManifest

__all__ = ["describe"]

_UNAVAILABLE = "<available at run time>"


def _value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _input_text(compiled: CompiledGraph, node_id: str) -> str:
    node = compiled.graph.node(node_id)
    if not node.inputs:
        return "(none)"
    return "; ".join(
        f"{slice_.entity}[{', '.join(slice_.columns)}] rows={slice_.rows}"
        for slice_ in node.inputs
    )


def _owned_text(compiled: CompiledGraph, node_id: str) -> str:
    node = compiled.graph.node(node_id)
    if not node.outputs:
        if node.structural is not StructuralDelta.NONE:
            return "(none; structural frame carries base columns)"
        return "(none)"
    return "; ".join(
        f"{owned.entity}.{owned.column}:{owned.dtype} rows={owned.rows} "
        f"ownership={owned.ownership.value}"
        for owned in node.outputs
    )


def _predecessor_text(
    compiled: CompiledGraph, node_id: str, manifest: RunManifest | None
) -> str:
    predecessors = compiled.predecessors[node_id]
    if not predecessors:
        return "(none)"
    rendered: list[str] = []
    for predecessor in predecessors:
        receipt = None if manifest is None else manifest.nodes.get(predecessor)
        key = _UNAVAILABLE if receipt is None else receipt.key
        rendered.append(f"{predecessor} [{key}]")
    return "; ".join(rendered)


def describe(
    compiled: CompiledGraph,
    node_id: str,
    manifest: RunManifest | None = None,
) -> str:
    """Render a node's declaration and optional run facts in under 40 lines."""

    node = compiled.graph.node(node_id)
    version = compiled.versions[node_id]
    run_receipt = None if manifest is None else manifest.nodes.get(node_id)
    node_identity = _UNAVAILABLE if run_receipt is None else run_receipt.key
    implementation = (
        _UNAVAILABLE if run_receipt is None else run_receipt.kernel_impl_hash
    )
    lines = [
        f"Node: {node.id}",
        f"Version: {version} (structural={node.structural.value})",
    ]
    if node.base is not None:
        lines.append(f"Base version: {node.base}")
    lines.extend(
        [
            f"Node key: {node_identity}",
            f"Predecessors: {_predecessor_text(compiled, node_id, manifest)}",
            f"Inputs: {_input_text(compiled, node_id)}",
            f"Owned cells: {_owned_text(compiled, node_id)}",
            "Parameters: " + canonical_json(node.params).decode("utf-8"),
            f"Sources: {', '.join(node.sources) if node.sources else '(none)'}",
            f"Kernel: {node.kernel}",
            f"Implementation hash: {implementation}",
        ]
    )
    if node.artifact_inputs:
        lines.append(
            "Artifact inputs: "
            + "; ".join(
                f"{item.name} <- {item.producer}.{item.artifact} ({item.type.name}@{item.type.schema_version})"
                for item in node.artifact_inputs
            )
        )
    if node.artifact_outputs:
        lines.append(
            "Artifact outputs: "
            + "; ".join(
                f"{item.name} ({item.type.name}@{item.type.schema_version})"
                for item in node.artifact_outputs
            )
        )
    if run_receipt is None:
        lines.append(
            'Seed: int.from_bytes(sha256(b"seed\\0" + node_key)[:8], "little")'
        )
    else:
        capabilities = run_receipt.capabilities
        if isinstance(capabilities, Mapping):
            tolerance_text = "<legacy_capabilities: tolerance unavailable>"
            determinism = capabilities["determinism"]
            numeric = capabilities["numeric"]
            seed_source = capabilities["seed_source"]
            structural = capabilities["structural"]
            consumes_se = capabilities["consumes_se"]
        else:
            tolerance = capabilities.tolerance
            tolerance_text = canonical_json(
                None
                if tolerance is None
                else {
                    "rtol": tolerance.rtol,
                    "atol": tolerance.atol,
                    "ulps": tolerance.ulps,
                }
            ).decode("utf-8")
            determinism = capabilities.determinism
            numeric = capabilities.numeric
            seed_source = capabilities.seed_source
            structural = capabilities.structural
            consumes_se = capabilities.consumes_se
        lines.extend(
            [
                'Seed: int.from_bytes(sha256(b"seed\\0" + node_key)[:8], '
                f'"little") = {run_receipt.seed}',
                f"Store: {'hit' if run_receipt.hit else 'miss'}; "
                f"wall_time={run_receipt.wall_time:.6g}s",
                "Capabilities: "
                f"determinism={_value(determinism)}, "
                f"numeric={_value(numeric)}, "
                f"seed={_value(seed_source)}, "
                f"structural={_value(structural)}, "
                f"consumes_se={consumes_se}, "
                f"tolerance={tolerance_text}",
                "Receipt: " + canonical_json(run_receipt.receipt).decode("utf-8"),
            ]
        )
    if len(lines) >= 40:  # Defensive: additions must preserve charter G1.
        raise AssertionError("describe() exceeded the 39-line display contract")
    return "\n".join(lines)
