"""UK release input-coverage manifest derivation and candidate evidence."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


_REPO_ROOT = _TEST_PATHS.repository
_GENERATOR = _REPO_ROOT / "tools" / "build_uk_release_input_coverage_manifest.py"
_UK_PACKAGE = "microcosm.build.uk"


def _resource(name: str) -> dict:
    return json.loads(files(_UK_PACKAGE).joinpath(name).read_text(encoding="utf-8"))


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_uk_release_input_coverage_manifest", _GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = [name for name in globals() if not name.startswith("__")]
