"""Shared helpers for the US gate battery's shadow tests (microcosm#1585 item 11).

Each US scope is evaluated by one build tool whose gate functions the
bindings call through the tool's own globals, so the tests load the tools
the way their own tests do: by path, without engines.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

from test_support.paths import paths_for

#: Scope to the build tool that evaluates it.
US_GATE_SCOPE_TOOLS = {
    "us_base": "build_us_puf_support_base",
    "us_pool": "build_us_multispine_pool",
    "us_national": "build_us_fiscal_refresh_release",
    "us_local": "build_us_acs_local_release",
}


def us_gate_tool_path(scope: str) -> Path:
    return (
        paths_for("microcosm-build").repository
        / "tools"
        / f"{US_GATE_SCOPE_TOOLS[scope]}.py"
    )


@cache
def load_us_gate_tool(scope: str) -> ModuleType:
    """The scope's build tool, loaded once per process."""

    path = us_gate_tool_path(scope)
    name = f"us_gate_shadow_{US_GATE_SCOPE_TOOLS[scope]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _literal_string(node: ast.AST) -> str | None:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _lambda_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Lambda) and isinstance(node.body, ast.Call):
        func = node.body.func
        if isinstance(func, ast.Name):
            return func.id
    return None


#: Package modules where a scope observes a gate inside a runtime function
#: rather than in its tool.
US_GATE_SCOPE_RUNTIME_SITES = {
    "us_pool": ("microcosm.build.us_runtime.multispine_pool",),
}

_HELPERS = frozenset(
    {
        "shadow_gate",
        "shadow_failures",
        "shadow_check",
        "shadow_not_run",
        "_shadow_frame_gate",
        "_observe_frame_gate",
    }
)


def _module_path(module: str) -> Path:
    relative = module.removeprefix("microcosm.build.").replace(".", "/")
    return (
        paths_for("microcosm-build").repository
        / "packages/microcosm-build/src/microcosm/build"
        / f"{relative}.py"
    )


def _sites(path: Path, namespace: str) -> list[tuple[str, str | None, str]]:
    tree = ast.parse(path.read_text())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in _HELPERS:
            continue
        entry_id = _literal_string(node.args[0])
        if entry_id is None:
            continue
        gate = _lambda_call_name(node.args[1]) if len(node.args) > 1 else None
        reference = None if gate is None else f"{namespace}:{gate}"
        sites.append((entry_id, reference, f"{path.name}:{node.lineno}"))
    return sites


def shadow_wiring(scope: str) -> list[tuple[str, str | None, str]]:
    """Every shadow call site of a scope: (entry id, gate reference, where).

    Reads the scope's tool and the runtime modules it observes gates in: a
    call to one of the shadow helpers whose first argument is an entry id
    literal. The gate reference names the function today's lambda calls at
    that site, as a binding names it (``tool:<name>`` in the tool,
    ``<module>:<name>`` in a runtime module), when the site has one.
    """

    sites = _sites(us_gate_tool_path(scope), "tool")
    for module in US_GATE_SCOPE_RUNTIME_SITES.get(scope, ()):
        sites.extend(_sites(_module_path(module), module))
    return sites
