"""Exceptions the runtime raises. Names are part of the public contract.

The acceptance suite asserts on these types, so the executor and the store
raise exactly them: a kernel that breaks its declaration is
``NodeRejectedError``; an artifact whose bytes do not match its key is
``StoreCorruptError``; a codec or dependency missing at load time is
``StoreUnavailableError`` (never a rebuild); a node with no store hit under
``resume="require"`` is ``StoreMissError``; a kernel that is no longer the
computation its node key names is ``KernelIdentityChangedError``, a
``NodeRejectedError`` the executor never files as a gate verdict.
"""

from __future__ import annotations

__all__ = [
    "GraphRuntimeError",
    "KernelIdentityChangedError",
    "NodeRejectedError",
    "StoreCorruptError",
    "StoreMissError",
    "StoreUnavailableError",
]


class GraphRuntimeError(RuntimeError):
    """Base class for every runtime failure of the node graph."""


class NodeRejectedError(GraphRuntimeError):
    """A kernel's result violated its node's declaration; nothing was applied."""


class KernelIdentityChangedError(NodeRejectedError):
    """A kernel is no longer the computation its implementation hash named.

    A kernel raises it when state its ``implementation_hash`` describes moved
    after that hash was fixed, so the node keys derived from it misname the
    kernel. The executor never files it as a gate verdict, unlike any other
    gate exception (amendment 7): the run refuses and nothing for the node is
    stored (amendment 29), because a verdict filed under the key would be
    served to every later run of the unchanged kernel. The executor cannot
    detect the move itself, since a hash re-read from disk also moves when a
    file changes under code already imported; a kernel whose identity can move
    in process must check it in ``run`` and raise this.
    """


class StoreCorruptError(GraphRuntimeError):
    """A stored artifact or manifest does not match the key it is filed under."""


class StoreUnavailableError(GraphRuntimeError):
    """A codec or dependency needed to load or verify an artifact is absent."""


class StoreMissError(GraphRuntimeError):
    """``resume="require"`` found a node with no stored result."""
