"""microcosm-graph: a content-addressed DAG of cell-ownership nodes.

Public API. ``decl`` and ``kernel`` are the frozen interfaces; the runtime
modules implement execution, storage, provenance, and inspection against them.

Declarations, the compiler schema and the Orrery projection need only the
standard library, so they import eagerly. Exports from the runtime modules,
which need numpy, pandas and microcosm-frame, load on first access: a process
that only projects saved schemas and execution overlays imports
``microcosm.graph.orrery`` without those dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib import metadata as _metadata

from .decl import (
    DESCRIPTIVE_FIELDS,
    DTYPES,
    GATE_OUTCOMES,
    MASS_POLICIES,
    PARTITION_DTYPES,
    ROWS_ALL,
    WEIGHT_KINDS,
    WEIGHT_UPDATE_MASS_POLICIES,
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
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
    WeightUpdate,
    compile_graph,
)
from .errors import (
    GraphRuntimeError,
    NodeRejectedError,
    StoreCorruptError,
    StoreMissError,
    StoreUnavailableError,
)
from .schema import graph_schema
from .serialize import graph_from_json, graph_to_json
from .weight_update import WEIGHT_UPDATE_AXIS_SCHEMA, weight_update_receipt

__all__ = [
    "platform_fingerprint",
    "DESCRIPTIVE_FIELDS",
    "DTYPES",
    "GATE_OUTCOMES",
    "MASS_POLICIES",
    "PARTITION_DTYPES",
    "ROWS_ALL",
    "WEIGHT_KINDS",
    "ArtifactInput",
    "ArtifactOutput",
    "ArtifactType",
    "ArtifactValue",
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
    "NumericScope",
    "Tolerance",
    "Slice",
    "SourceBytesCodec",
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
    "WeightUpdate",
    "WEIGHT_UPDATE_AXIS_SCHEMA",
    "WEIGHT_UPDATE_MASS_POLICIES",
    "compile_graph",
    "describe",
    "explain_html",
    "graph_from_json",
    "graph_schema",
    "graph_to_json",
    "keyed_uniform",
    "load_source",
    "load_source_bytes",
    "run_graph",
    "source_hash",
    "weight_update_receipt",
    "orrery_document",
    "orrery_json",
    "ArtifactSummaryContext",
    "RecordedRun",
    "checkpoint_references",
    "collect_execution_evidence",
    "load_run_evidence",
    "publish_run_evidence",
    "record_run_binding",
    "save_graph_schema",
    "save_run_evidence",
]

#: The runtime modules and the exports each one supplies on first access.
_RUNTIME_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "codecs": (
        "SOURCE_CODECS",
        "SourceBytesCodec",
        "SourceCodec",
        "SourceCodecRegistry",
        "load_source",
        "load_source_bytes",
    ),
    "evidence": (
        "ArtifactSummaryContext",
        "RecordedRun",
        "checkpoint_references",
        "collect_execution_evidence",
        "load_run_evidence",
        "publish_run_evidence",
        "record_run_binding",
        "save_graph_schema",
        "save_run_evidence",
    ),
    "executor": ("NodeRejected", "run_graph"),
    "explain": ("explain_html",),
    "kernel": (
        "ArtifactValue",
        "Capabilities",
        "Determinism",
        "Kernel",
        "KernelBase",
        "KernelContext",
        "KernelRegistry",
        "KernelResult",
        "KernelRole",
        "Numeric",
        "NumericScope",
        "SeedSource",
        "Tolerance",
        "source_hash",
    ),
    "keys": ("platform_fingerprint",),
    "manifest": ("Decision", "NodeReceipt", "PopulationView", "RunManifest"),
    "population": ("MassRecord", "Population", "PopulationError"),
    "randomness": ("keyed_uniform",),
    "store": (
        "ContentStore",
        "ResumePolicy",
        "StoreCorrupt",
        "StoreError",
        "StoreMiss",
        "StoreUnavailable",
    ),
    "view": ("describe",),
}
_RUNTIME_EXPORTS = {
    export: module_name
    for module_name, exports in _RUNTIME_MODULE_EXPORTS.items()
    for export in exports
}

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


# The installed series is checked before any runtime module can be imported.
_check_frame_version()


def __getattr__(name: str) -> object:
    """Import a runtime export's defining module the first time it is used."""

    module_name = _RUNTIME_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_RUNTIME_EXPORTS})


def orrery_document(
    graph: Graph | CompiledGraph,
    *,
    title: str | None = None,
    extensions: Mapping[str, object] | None = None,
    execution: object | None = None,
) -> dict[str, object]:
    """Compile and transform a graph into a complete Orrery document."""

    from .orrery import orrery_document as export

    return export(graph, title=title, extensions=extensions, execution=execution)


def orrery_json(
    graph: Graph | CompiledGraph,
    *,
    title: str | None = None,
    extensions: Mapping[str, object] | None = None,
    execution: object | None = None,
) -> str:
    """Compile and return deterministic UTF-8-ready Orrery JSON."""

    from .orrery import orrery_json as export

    return export(graph, title=title, extensions=extensions, execution=execution)
