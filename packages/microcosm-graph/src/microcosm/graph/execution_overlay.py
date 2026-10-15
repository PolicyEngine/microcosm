"""The execution overlay's protocol, shared by evidence capture and projection.

:func:`microcosm.graph.evidence.collect_execution_evidence` verifies recorded
runs against the content store and returns the overlay;
:mod:`microcosm.graph.orrery` projects it. This module needs only the standard
library, so a process that only projects needs neither the store nor numpy,
pandas or microcosm-frame.
"""

from __future__ import annotations

__all__ = ["EXECUTION_PROTOCOL"]

EXECUTION_PROTOCOL = "microcosm.graph.execution-evidence.v1"
