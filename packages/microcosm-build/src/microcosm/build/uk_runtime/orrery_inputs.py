"""The collector's Orrery inputs for a UK build (not written yet).

The build's graph step still lives in :mod:`.orrery_publication`; this module
only names it, and its command writes nothing.
"""

from __future__ import annotations

from .orrery_publication import (
    EVIDENCE_MANIFEST_NAME,
    EXECUTION_OVERLAY_NAME,
    FAILURE_BUNDLE_DIRECTORY,
    FAILURE_MARKER_NAME,
    finalize_graph,
    stage_graph_evidence,
)

__all__ = [
    "EVIDENCE_MANIFEST_NAME",
    "EXECUTION_OVERLAY_NAME",
    "FAILURE_BUNDLE_DIRECTORY",
    "FAILURE_MARKER_NAME",
    "SCHEMA_NAME",
    "finalize_graph",
    "main",
    "stage_graph_evidence",
]

SCHEMA_NAME = "graph.schema.json"


def main(argv: list[str] | None = None) -> int:
    """Write the collector's Orrery inputs into a new directory (not yet)."""

    return 0
