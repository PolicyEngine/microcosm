"""Repo-root pytest configuration.

CI invokes pytest as a console script, which does not place the working
directory on ``sys.path``. Tests that exercise the F0 migration tooling
import the ``tools.us_bundle_generation`` package from the repository
root, so the root joins the path here explicitly rather than by the
accident of ``python -m pytest``.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
