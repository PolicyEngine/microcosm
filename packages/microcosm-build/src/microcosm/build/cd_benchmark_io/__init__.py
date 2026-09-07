"""Explicit filesystem boundary for CD v4; no scoring or default admissions.

The frozen ``cd_benchmark`` package remains pure. This package performs local
metadata and ledger I/O only when called explicitly. Genuine source/scientific
producer adapters and approvals are deliberately not supplied here.
"""

from .artifacts import (
    BoundaryError,
    FilePin,
    ProducerEvidence,
    ResolvedCandidate,
    load_authority,
    load_candidate,
    read_pinned,
)
from .durable import DurableLedger

__all__ = [
    "BoundaryError",
    "FilePin",
    "ProducerEvidence",
    "ResolvedCandidate",
    "load_authority",
    "load_candidate",
    "read_pinned",
    "DurableLedger",
]
