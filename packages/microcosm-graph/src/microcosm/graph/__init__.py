"""microcosm-graph: a content-addressed DAG of cell-ownership nodes.

Public API. ``decl`` and ``kernel`` are the frozen interfaces; the executor,
store, manifest, and view are implemented against them and land through
``docs/graph-acceptance.md``. Until each lands, its name here raises
``NotImplementedError`` so the red acceptance suite imports cleanly.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from .decl import (
    DESCRIPTIVE_FIELDS,
    DTYPES,
    GATE_OUTCOMES,
    MASS_POLICIES,
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
    NodeRejected,
    StoreCorrupt,
    StoreMiss,
    StoreUnavailable,
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
    source_hash,
)

__all__ = [
    "DESCRIPTIVE_FIELDS",
    "DTYPES",
    "GATE_OUTCOMES",
    "MASS_POLICIES",
    "ROWS_ALL",
    "WEIGHT_KINDS",
    "Capabilities",
    "CompiledGraph",
    "ContentStore",
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
    "Node",
    "NodeRejected",
    "Numeric",
    "Owned",
    "Ownership",
    "Param",
    "RunManifest",
    "SeedSource",
    "Slice",
    "SourceRef",
    "StoreCorrupt",
    "StoreMiss",
    "StoreUnavailable",
    "StructuralDelta",
    "WeightTransition",
    "compile_graph",
    "describe",
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

_PENDING = (
    "microcosm-graph: {name} is not implemented yet; see "
    "docs/graph-acceptance.md for the property that lands it."
)


class ContentStore:
    """Pending: content-addressed artifact store (charter group E)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_PENDING.format(name="ContentStore"))


class RunManifest:
    """Pending: run provenance record (charter E5, A7)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_PENDING.format(name="RunManifest"))


def run_graph(*args: object, **kwargs: object) -> RunManifest:
    """Pending: the executor (charter groups A–E)."""
    raise NotImplementedError(_PENDING.format(name="run_graph"))


def describe(*args: object, **kwargs: object) -> str:
    """Pending: the one-screen node view (charter G1)."""
    raise NotImplementedError(_PENDING.format(name="describe"))
