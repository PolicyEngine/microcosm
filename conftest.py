"""Repo-root pytest configuration.

CI invokes pytest as a console script, which does not place the working
directory on ``sys.path``. Tests that exercise the F0 migration tooling
import the ``tools.us_bundle_generation`` package from the repository
root, so the root joins the path here explicitly rather than by the
accident of ``python -m pytest``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def pytest_collection_modifyitems(config, items):
    engine_markers = {
        "requires_us": "policyengine_us",
        "requires_uk": "policyengine_uk",
    }
    for marker_name, module_name in engine_markers.items():
        if importlib.util.find_spec(module_name) is not None:
            continue
        skip = pytest.mark.skip(
            reason=f"requires {module_name.replace('_', '-')} extra"
        )
        for item in items:
            if item.get_closest_marker(marker_name):
                item.add_marker(skip)
