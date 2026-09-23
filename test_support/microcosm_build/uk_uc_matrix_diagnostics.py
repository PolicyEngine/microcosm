"""Target comparisons preserve source ancestry and their declared controls."""

# ruff: noqa: F401

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


@pytest.fixture(scope="module")
def diagnostic_tool():
    root = _TEST_PATHS.repository
    tools = root / "tools"
    sys.path.insert(0, str(tools))
    try:
        spec = importlib.util.spec_from_file_location(
            "uc_matrix_diagnostics", tools / "diagnose_uk_uc_matrix.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(tools))


__all__ = [name for name in globals() if not name.startswith("__")]
