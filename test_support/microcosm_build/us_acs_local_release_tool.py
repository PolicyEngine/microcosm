"""Loaders for the ACS local release chain's two tools.

``tools/`` is not a package, so tests in more than one category load the
tools by path through these helpers.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

from test_support.paths import paths_for

_TOOLS = paths_for("microcosm-build").repository / "tools"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tool_module() -> ModuleType:
    """``tools/build_us_acs_local_release.py``, freshly executed."""

    return _load("build_us_acs_local_release")


def load_staging_builder_module() -> ModuleType:
    """``tools/build_us_acs_multispine_base.py``, the staging shim."""

    return _load("build_us_acs_multispine_base")
