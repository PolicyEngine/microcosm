"""The acceptance-suite ratchet: counting, and refusing a rise.

``tools/graph_acceptance_burndown.py`` is the executable half of the charter's
second process rule — the number of red properties only goes down. These tests
cover the two things that rule depends on: counting markers from the syntax
tree rather than from text, and noticing when a file's count rose against a
baseline. Both run against source held in a temporary file, so nothing here
depends on the state of the real suite.
"""

# ruff: noqa: F401

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_support.paths import paths_for
from tools import graph_acceptance_burndown as burndown

_TEST_PATHS = paths_for("microcosm-graph")

# Imported as a module, not by name: pytest collects any imported callable whose
# name matches ``test*``, and ``tests_in`` would be collected as a test.
charter_ids = burndown.charter_ids
dotted = burndown.dotted
markers_in = burndown.markers_in

TOOL = _TEST_PATHS.repository / "tools" / "graph_acceptance_burndown.py"

TWO_RED_PROPERTIES = '''
"""A module whose docstring mentions @pytest.mark.xfail and must not count."""

import pytest


@pytest.mark.xfail(strict=True, reason="charter A1: pending")
def test_a1_one() -> None:
    assert False


# @pytest.mark.xfail(strict=True, reason="charter A2: pending")
def test_a2_commented_out_marker_is_green() -> None:
    assert True


@pytest.mark.xfail(strict=True, reason="charter A3: pending")
def test_a3_three() -> None:
    assert False


def helper_that_is_not_a_test() -> str:
    return "@pytest.mark.xfail(strict=True, reason='charter A9: pending')"
'''

ONE_RED_PROPERTY = TWO_RED_PROPERTIES.replace(
    '@pytest.mark.xfail(strict=True, reason="charter A3: pending")\ndef test_a3_three',
    "def test_a3_three",
).replace(
    "def test_a3_three() -> None:\n    assert False",
    "def test_a3_three() -> None:\n    assert True",
)

SLOPPY_MARKERS = """
import pytest


@pytest.mark.xfail(reason="charter B1: pending")
def test_b1_not_strict() -> None:
    assert False


@pytest.mark.xfail(strict=True, reason="because I said so")
def test_b2_no_charter_id() -> None:
    assert False
"""


def _run(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the *copy* of the tool inside ``repository``.

    The tool locates the repository from its own ``__file__``, not from the
    working directory, so running the original would score the real suite.
    """
    return subprocess.run(
        [
            sys.executable,
            str(repository / "tools" / "graph_acceptance_burndown.py"),
            *arguments,
        ],
        cwd=repository,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path: Path, sources: dict[str, str]) -> Path:
    """A throwaway git repository shaped like this one, with one commit."""
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "packages" / "microcosm-graph" / "tests").mkdir(parents=True)
    (root / "tools" / "graph_acceptance_burndown.py").write_text(TOOL.read_text())
    (root / "docs" / "graph-acceptance.md").write_text(
        "| Id | Property |\n|---|---|\n| A1 | one |\n| A3 | three |\n"
    )
    for name, text in sources.items():
        (root / "packages" / "microcosm-graph" / "tests" / name).write_text(text)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=root,
        check=True,
    )
    return root


NEW_PROPERTY_STARTS_RED = (
    ONE_RED_PROPERTY
    + """

@pytest.mark.xfail(strict=True, reason="charter A9: pending")
def test_a9_nine() -> None:
    assert False
"""
)


A1_GREEN_A3_RED = """
import pytest


def test_a1_one() -> None:
    assert True


@pytest.mark.xfail(strict=True, reason="charter A3: pending")
def test_a3_three() -> None:
    assert False
"""


MISLABELLED_MARKER = """
import pytest


@pytest.mark.xfail(strict=True, reason="charter A3: pending")
def test_a1_one() -> None:
    assert False


@pytest.mark.xfail(strict=True, reason="charter A3: pending")
def test_a3_three() -> None:
    assert False
"""

__all__ = [name for name in globals() if not name.startswith("__")]
