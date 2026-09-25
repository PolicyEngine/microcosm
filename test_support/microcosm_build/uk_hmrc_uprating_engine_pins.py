"""The declared UK uprating appliers read vendored engine values, not the engine.

The fast CI tiers carry no country package, so the compile of the fixture
subset must run without policyengine-uk; the pins file carries the values and
the engine version they came from, and the ``requires_uk`` test holds it in
lockstep with the installed engine.
"""

# ruff: noqa: F401

from __future__ import annotations

import json

import pytest

from microcosm.build.uk_runtime import hmrc_uprating as hu
from test_support.paths import paths_for

_PINS_PATH = (
    paths_for("microcosm-build").package
    / "src"
    / "microcosm"
    / "build"
    / "uk"
    / hu.UK_ENGINE_PINS_RESOURCE
)

__all__ = [name for name in globals() if not name.startswith("__")]
