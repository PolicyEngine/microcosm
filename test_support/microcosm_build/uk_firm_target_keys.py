"""The forbidden-key walk UK firm target declarations must pass, for tests.

A target declaration carries no observed value and no runtime hook: none of
``FORBIDDEN_VALUE_KEYS`` or ``FORBIDDEN_RUNTIME_KEYS`` may appear at any depth,
except ``value`` in a filter predicate and ``operation`` in resolution
defaults.
"""

from __future__ import annotations

from collections.abc import Mapping

FORBIDDEN_VALUE_KEYS = {"aggregation", "operation", "registry", "target_value", "value"}
FORBIDDEN_RUNTIME_KEYS = {
    "callable",
    "command",
    "execute",
    "executable",
    "function",
    "import",
    "imports",
    "module",
    "python_code",
    "runtime_code",
    "script",
    "solver",
}


def is_filter_predicate(node: Mapping) -> bool:
    return (
        "value" in node
        and "operator" in node
        and ("concept" in node or "variable" in node)
    )


def carried_forbidden_keys(node, *, in_defaults: bool = False) -> set[str]:
    """The forbidden keys ``node`` carries at any depth."""

    carried: set[str] = set()
    if isinstance(node, Mapping):
        forbidden = set(FORBIDDEN_VALUE_KEYS | FORBIDDEN_RUNTIME_KEYS)
        if is_filter_predicate(node):
            forbidden.discard("value")
        if in_defaults:
            forbidden.discard("operation")
        carried.update(forbidden.intersection(node))
        for child in node.values():
            carried.update(carried_forbidden_keys(child))
    elif isinstance(node, (list, tuple)):
        for child in node:
            carried.update(carried_forbidden_keys(child))
    return carried
