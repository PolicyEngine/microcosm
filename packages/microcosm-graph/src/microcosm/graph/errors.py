"""Exceptions the runtime raises. Names are part of the public contract.

The acceptance suite asserts on these types, so the executor and the store
raise exactly them: a kernel that breaks its declaration is ``NodeRejected``;
an artifact whose bytes do not match its key is ``StoreCorrupt``; a codec
or dependency missing at load time is ``StoreUnavailable`` (never a rebuild);
a node with no store hit under ``resume="require"`` is ``StoreMiss``.
"""

from __future__ import annotations

__all__ = [
    "GraphRuntimeError",
    "NodeRejected",
    "StoreCorrupt",
    "StoreMiss",
    "StoreUnavailable",
]


class GraphRuntimeError(RuntimeError):
    """Base class for every runtime failure of the node graph."""


class NodeRejected(GraphRuntimeError):
    """A kernel's result violated its node's declaration; nothing was applied."""


class StoreCorrupt(GraphRuntimeError):
    """A stored artifact or manifest does not match the key it is filed under."""


class StoreUnavailable(GraphRuntimeError):
    """A codec or dependency needed to load or verify an artifact is absent."""


class StoreMiss(GraphRuntimeError):
    """``resume="require"`` found a node with no stored result."""
