"""Candidate-only inspection export past a structural full-build gate block.

``microcosm-build-uk --release-role dense --inspection-export`` exports,
packages and stages an unreleasable H5 for inspection when a structural gate
blocks the artifact, so a candidate can be studied rather than discarded. The
CLI refuses the flag for release candidates and for the national role; the
manifest stays unreleasable and the exit status still reports the block.

The kernel lives in its own module: the gate, holdout and export kernels hash
:mod:`graph_terminal`'s source, so keeping the inspection path out of it
leaves every default node key unchanged.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

from microcosm.graph import (
    Graph,
    KernelContext,
    KernelRegistry,
    KernelResult,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import graph_terminal
from .graph_population import context_frame

#: The export description node the inspection kernel replaces.
EXPORT_PREPARE_NODE = "uk.full.export.prepare"


class UKInspectionExportPrepareKernel(graph_terminal.UKExportPrepareKernel):
    """Describe the H5 as the export kernel does, without the gate refusal."""

    ref = "uk.full-export.prepare-inspection@1"

    def implementation_hash(self) -> str:
        return source_hash(sys.modules[__name__], graph_terminal)

    def run(self, context: KernelContext) -> KernelResult:
        if context.params.get("inspection_export") is not True:
            raise ValueError("The inspection export kernel requires inspection_export.")
        bindings = json.loads(str(context.params["bindings"]))
        if bindings.get("inspection_export") is not True:
            raise ValueError(
                "An inspection export must record inspection_export in its request."
            )
        bindings["artifacts"] = {
            name: value.key for name, value in context.artifacts.items()
        }
        descriptor = graph_terminal.describe_uk_export(
            context_frame(context), bindings=bindings
        )
        return KernelResult(artifacts={"export_descriptor": canonical_json(descriptor)})


def use_inspection_export(graph: Graph) -> Graph:
    """Swap the export description node onto the inspection kernel."""

    nodes = []
    swapped = 0
    for node in graph.nodes:
        if node.id == EXPORT_PREPARE_NODE:
            node = replace(
                node,
                kernel=UKInspectionExportPrepareKernel.ref,
                params={**dict(node.params), "inspection_export": True},
            )
            swapped += 1
        nodes.append(node)
    if swapped != 1:
        raise ValueError(f"Expected one {EXPORT_PREPARE_NODE} node, found {swapped}.")
    return replace(graph, nodes=tuple(nodes))


def register_uk_inspection_export_kernel(registry: KernelRegistry) -> None:
    registry.register(UKInspectionExportPrepareKernel())
