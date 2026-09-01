"""Exceptions the runtime raises. Names are part of the public contract.

The acceptance suite asserts on these types, so the executor and the store
raise exactly them: a kernel that breaks its declaration is
``NodeRejectedError``; an artifact whose bytes do not match its key is
``StoreCorruptError``; a codec or dependency missing at load time is
``StoreUnavailableError`` (never a rebuild); a node with no store hit under
``resume="require"`` is ``StoreMissError``.
"""

from __future__ import annotations

__all__ = [
    "GraphRuntimeError",
    "NodeRejectedError",
    "StoreCorruptError",
    "StoreMissError",
    "StoreUnavailableError",
]


class GraphRuntimeError(RuntimeError):
    """Base class for every runtime failure of the node graph."""


class NodeRejectedError(GraphRuntimeError):
    """A kernel's result violated its node's declaration; nothing was applied."""


class StoreCorruptError(GraphRuntimeError):
    """A stored artifact or manifest does not match the key it is filed under."""


class StoreUnavailableError(GraphRuntimeError):
    """A codec or dependency needed to load or verify an artifact is absent."""


class StoreMissError(GraphRuntimeError):
    """``resume="require"`` found a node with no stored result."""
