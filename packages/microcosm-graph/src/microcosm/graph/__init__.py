"""microcosm-graph: a content-addressed DAG of cell-ownership nodes.

Public API. ``decl`` and ``kernel`` are the frozen interfaces; the runtime
modules implement execution, storage, provenance, and inspection against them.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from .decl import (
    DESCRIPTIVE_FIELDS,
    DTYPES,
    GATE_OUTCOMES,
    MASS_POLICIES,
    PARTITION_DTYPES,
    ROWS_ALL,
    WEIGHT_KINDS,
    CompiledGraph,
    Graph,
    GraphError,
    Node,
    Owned,
    Ownership,
    Param,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
)
from .errors import (
    GraphRuntimeError,
    NodeRejectedError,
    StoreCorruptError,
    StoreMissError,
    StoreUnavailableError,
)
from .kernel import (
    Capabilities,
    Determinism,
    Kernel,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Numeric,
    SeedSource,
    Tolerance,
    source_hash,
)

__all__ = [
    "DESCRIPTIVE_FIELDS",
    "DTYPES",
    "GATE_OUTCOMES",
    "MASS_POLICIES",
    "PARTITION_DTYPES",
    "ROWS_ALL",
    "WEIGHT_KINDS",
    "Capabilities",
    "CompiledGraph",
    "ContentStore",
    "Decision",
    "Determinism",
    "Graph",
    "GraphError",
    "GraphRuntimeError",
    "Kernel",
    "KernelBase",
    "KernelContext",
    "KernelRegistry",
    "KernelResult",
    "KernelRole",
    "MassRecord",
    "Node",
    "NodeReceipt",
    "NodeRejected",
    "NodeRejectedError",
    "Numeric",
    "Owned",
    "Ownership",
    "Param",
    "Population",
    "PopulationView",
    "PopulationError",
    "ResumePolicy",
    "RunManifest",
    "SOURCE_CODECS",
    "SeedSource",
    "Tolerance",
    "Slice",
    "SourceCodec",
    "SourceCodecRegistry",
    "SourceRef",
    "StoreCorruptError",
    "StoreMissError",
    "StoreUnavailableError",
    "StoreCorrupt",
    "StoreError",
    "StoreMiss",
    "StoreUnavailable",
    "StructuralDelta",
    "WeightTransition",
    "compile_graph",
    "describe",
    "explain_html",
    "graph_from_json",
    "graph_to_json",
    "load_source",
    "run_graph",
    "source_hash",
]

_FRAME_SERIES = "0.1"


def _check_frame_version() -> None:
    try:
        version = _metadata.version("microcosm-frame")
    except _metadata.PackageNotFoundError:
        return
    if not version.startswith(_FRAME_SERIES + "."):
        raise ImportError(
            f"microcosm-graph requires microcosm-frame {_FRAME_SERIES}.x, found "
            f"{version}."
        )


_check_frame_version()

from .codecs import (  # noqa: E402 - check dependency series before runtime import
    SOURCE_CODECS,
    SourceCodec,
    SourceCodecRegistry,
    load_source,
)
from .executor import NodeRejected, run_graph  # noqa: E402
from .explain import explain_html  # noqa: E402
from .manifest import Decision, NodeReceipt, PopulationView, RunManifest  # noqa: E402
from .population import MassRecord, Population, PopulationError  # noqa: E402
from .serialize import graph_from_json, graph_to_json  # noqa: E402
from .store import (  # noqa: E402
    ContentStore,
    ResumePolicy,
    StoreCorrupt,
    StoreError,
    StoreMiss,
    StoreUnavailable,
)
from .view import describe  # noqa: E402
