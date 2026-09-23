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

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="collect tests under tests/integration/",
    )


def _test_category(path: Path) -> tuple[str, str] | None:
    """Return the execution environment and scope encoded in a test path."""

    parts = path.parts
    try:
        tests_index = parts.index("tests")
        return parts[tests_index + 1], parts[tests_index + 2]
    except (ValueError, IndexError):
        return None


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Exclude unavailable test environments before importing their modules."""

    category = _test_category(Path(collection_path))
    if category is None:
        return None
    environment, scope = category
    if environment == "integration" and not config.getoption("--run-integration"):
        return True
    if environment in {"engine", "integration"}:
        module_name = {"us": "policyengine_us", "uk": "policyengine_uk"}.get(scope)
        if module_name is not None and importlib.util.find_spec(module_name) is None:
            return True
    return None


def pytest_collection_modifyitems(items) -> None:
    """Expose directory-derived country requirements as pytest markers."""

    for item in items:
        category = _test_category(Path(item.path))
        if category is None:
            continue
        environment, scope = category
        if environment == "integration":
            item.add_marker("integration")
        if environment in {"engine", "integration"} and scope in {"us", "uk"}:
            item.add_marker(f"requires_{scope}")
