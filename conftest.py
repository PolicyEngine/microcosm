"""Repo-root pytest configuration.

CI invokes pytest as a console script, which does not place the working
directory on ``sys.path``. Tests that exercise the F0 migration tooling
import the ``tools.us_bundle_generation`` package from the repository
root, so the root joins the path here explicitly rather than by the
accident of ``python -m pytest``.

A suite started by a hook, ``rebase --exec`` command or shell alias in a
linked worktree inherits an absolute ``GIT_DIR``, which would send every
fixture's git command to that worktree's repository. The variables that
pick git's repository are removed here, before any test runs.
"""

import importlib.util
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from test_support.git_isolation import (  # noqa: E402
    drop_inherited_git_repository,
)
from tools.ci_test_plan import (  # noqa: E402
    TEST_GROUPS,
    TestGroup,
    group_name_for_collection_path,
)

drop_inherited_git_repository(os.environ)


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="collect tests under tests/integration/",
    )


def _test_group(path: Path) -> TestGroup | None:
    """Return registry metadata for the group containing a collection path."""

    group = group_name_for_collection_path(path)
    return TEST_GROUPS.get(group) if group is not None else None


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Exclude unavailable test environments before importing their modules."""

    group = _test_group(Path(collection_path))
    if group is None:
        return None
    if group.integration and not config.getoption("--run-integration"):
        return True
    if (
        group.engine_module is not None
        and importlib.util.find_spec(group.engine_module) is None
    ):
        return True
    return None


def pytest_collection_modifyitems(items) -> None:
    """Expose directory-derived country requirements as pytest markers."""

    for item in items:
        group = _test_group(Path(item.path))
        if group is None:
            continue
        if group.integration:
            item.add_marker("integration")
        if group.engine_module is not None:
            item.add_marker(f"requires_{group.country}")
