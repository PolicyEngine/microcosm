# ruff: noqa: F401
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


ROOT = _TEST_PATHS.repository
UK_PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build/uk"


def _load(name: str) -> dict:
    return json.loads((UK_PACKAGE / name).read_text(encoding="utf-8"))


__all__ = [name for name in globals() if not name.startswith("__")]
