"""The acceptance-suite ratchet: counting, and refusing a rise.

``tools/graph_acceptance_burndown.py`` is the executable half of the charter's
second process rule — the number of red properties only goes down. These tests
cover the two things that rule depends on: counting markers from the syntax
tree rather than from text, and noticing when a file's count rose against a
baseline. Both run against source held in a temporary file, so nothing here
depends on the state of the real suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import graph_acceptance_burndown as burndown

# Imported as a module, not by name: pytest collects any imported callable whose
# name matches ``test*``, and ``tests_in`` would be collected as a test.
charter_ids = burndown.charter_ids
dotted = burndown.dotted
markers_in = burndown.markers_in

TOOL = Path(__file__).resolve().parents[3] / "tools" / "graph_acceptance_burndown.py"

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


def test_markers_are_counted_from_the_syntax_tree_not_the_text() -> None:
    """A docstring, a comment, and a string literal all name ``xfail``.

    Only the two real decorators count; grep would have found five.
    """
    assert TWO_RED_PROPERTIES.count("mark.xfail") == 5
    markers = markers_in(TWO_RED_PROPERTIES, "sample.py")
    assert [marker.test for marker in markers] == ["test_a1_one", "test_a3_three"]
    assert [marker.charter_id for marker in markers] == ["A1", "A3"]
    assert all(marker.strict for marker in markers)
    assert all(marker.file == "sample.py" for marker in markers)


def test_a_marker_records_whether_it_is_strict_and_which_property_it_claims() -> None:
    markers = {marker.test: marker for marker in markers_in(SLOPPY_MARKERS)}
    assert markers["test_b1_not_strict"].strict is False
    assert markers["test_b1_not_strict"].charter_id == "B1"
    assert markers["test_b2_no_charter_id"].strict is True
    assert markers["test_b2_no_charter_id"].charter_id == ""


def test_a_property_going_green_lowers_the_count() -> None:
    """Deleting a marker is the only way the number moves, and it moves down."""
    before = markers_in(TWO_RED_PROPERTIES)
    after = markers_in(ONE_RED_PROPERTY)
    assert len(before) == 2
    assert len(after) == 1
    assert {marker.charter_id for marker in after} == {"A1"}


def test_tests_are_indexed_by_the_charter_id_in_their_name() -> None:
    """A green property keeps its test, so state survives losing the marker."""
    named = burndown.tests_in(TWO_RED_PROPERTIES)
    assert named == {
        "A1": "test_a1_one",
        "A2": "test_a2_commented_out_marker_is_green",
        "A3": "test_a3_three",
    }


def test_charter_ids_come_from_the_charter_tables() -> None:
    text = "\n".join(
        [
            "| Id | Property |",
            "|---|---|",
            "| A1 | **Determinism.** ... |",
            "| A1 | a repeat that must not double-count |",
            "| H3 | **US post-transfer parity.** ... |",
            "| V1 | **Graph explorer.** another lane's property |",
            "not a table row at all",
        ]
    )
    assert charter_ids(text) == ("A1", "H3")


def test_dotted_names_resolve_through_attributes() -> None:
    """The decorator matcher works on any spelling of the same marker."""
    import ast

    def func(source: str) -> ast.expr:
        return ast.parse(source).body[0].value.func

    assert dotted(func("pytest.mark.xfail(strict=True)")) == "pytest.mark.xfail"
    assert dotted(func("xfail(strict=True)")) == "xfail"
    assert dotted(func("mark.xfail(strict=True)")) == "mark.xfail"
    assert dotted(ast.parse("(a + b)(1)").body[0].value.func) == ""

    aliased = markers_in(
        "from pytest import mark\n\n\n"
        '@mark.xfail(strict=True, reason="charter A1: pending")\n'
        "def test_a1_aliased() -> None:\n    assert False\n"
    )
    assert [marker.charter_id for marker in aliased] == ["A1"]


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
    subprocess.run(["git", "init", "-q", "-b", "node-graph"], cwd=root, check=True)
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
        ["git", "update-ref", "refs/remotes/origin/node-graph", "HEAD"],
        cwd=root,
        check=True,
    )
    return root


def test_verify_passes_when_a_property_goes_green(tmp_path: Path) -> None:
    root = _repository(tmp_path, {"test_acceptance_a.py": TWO_RED_PROPERTIES})
    assert _run(root, "--verify").returncode == 0

    target = root / "packages" / "microcosm-graph" / "tests" / "test_acceptance_a.py"
    target.write_text(ONE_RED_PROPERTY)
    done = _run(root, "--verify")
    assert done.returncode == 0
    assert "2 -> 1" in done.stdout
    assert "verification=ok" in done.stdout


def test_verify_fails_when_a_file_re_reds_a_property(tmp_path: Path) -> None:
    root = _repository(tmp_path, {"test_acceptance_a.py": ONE_RED_PROPERTY})
    target = root / "packages" / "microcosm-graph" / "tests" / "test_acceptance_a.py"
    target.write_text(TWO_RED_PROPERTIES)

    risen = _run(root, "--verify")
    assert risen.returncode == 1
    assert "1 -> 2" in risen.stdout
    assert "re-reds 1 property" in risen.stdout
    assert "verification=failed" in risen.stdout


def test_verify_refuses_a_marker_that_is_not_strict_or_names_no_property(
    tmp_path: Path,
) -> None:
    """A non-strict marker hides an ``xpass``; an id-less reason hides a swap."""
    root = _repository(tmp_path, {"test_acceptance_a.py": SLOPPY_MARKERS})
    sloppy = _run(root, "--verify")
    assert sloppy.returncode == 1
    assert "is not strict" in sloppy.stdout
    assert "names no charter id" in sloppy.stdout


def test_verify_reports_a_charter_property_with_no_test(tmp_path: Path) -> None:
    root = _repository(tmp_path, {"test_acceptance_a.py": ONE_RED_PROPERTY})
    (root / "docs" / "graph-acceptance.md").write_text(
        "| Id | Property |\n|---|---|\n| A1 | one |\n| A3 | three |\n| A9 | nine |\n"
    )
    missing = _run(root, "--verify")
    assert missing.returncode == 1
    assert "charter A9 has no test in the suite" in missing.stdout


def test_json_carries_every_property_with_its_state(tmp_path: Path) -> None:
    """The shape the visuals lane renders as charter V2."""
    root = _repository(tmp_path, {"test_acceptance_a.py": ONE_RED_PROPERTY})
    data = json.loads(_run(root, "--json").stdout)
    assert data["total"] == 1
    assert data["charter_ids"] == ["A1", "A3"]
    states = {entry["id"]: entry["state"] for entry in data["properties"]}
    assert states == {"A1": "red", "A3": "green"}
    assert data["groups"] == [
        {
            "group": "A",
            "markers": 1,
            "ids": ["A1"],
            "files": ["packages/microcosm-graph/tests/test_acceptance_a.py"],
        }
    ]


def test_the_real_suite_is_all_strict_and_all_accounted_for() -> None:
    """The tool's own checks, run against the suite it exists to score."""
    data = burndown.report(burndown.counts(burndown.suite_files()))
    root = burndown.ROOT
    assert data["total"] == 4
    assert not [entry for entry in data["properties"] if entry["state"] == "missing"]
    states = {entry["id"]: entry["state"] for entry in data["properties"]}
    pending = {"B6", "B7", "C5", "D6"}  # amendments 11-14, flipped by their lanes
    assert {
        identifier for identifier, state in states.items() if state == "red"
    } == pending
    assert all(
        state == "green"
        for identifier, state in states.items()
        if identifier not in pending
    )
    for entry in data["files"]:
        source = (root / entry["file"]).read_text()
        for marker in markers_in(source, entry["file"]):
            assert marker.strict, f"{marker.test} is not strict"
            assert marker.charter_id, f"{marker.test} names no charter id"


@pytest.mark.parametrize("flag", ["--json", ""])
def test_the_tool_runs_from_the_command_line(flag: str) -> None:
    """``--verify`` is exercised by the temp-repo tests and by the lint job;
    here the plain and JSON forms are run against the real suite."""
    arguments = [sys.executable, str(TOOL)] + ([flag] if flag else [])
    completed = subprocess.run(
        arguments, cwd=TOOL.parents[1], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout


NEW_PROPERTY_STARTS_RED = (
    ONE_RED_PROPERTY
    + """

@pytest.mark.xfail(strict=True, reason="charter A9: pending")
def test_a9_nine() -> None:
    assert False
"""
)


def test_verify_lets_a_property_new_to_the_charter_start_red(tmp_path: Path) -> None:
    """The charter's meta-TDD rule: a new property is committed red first.

    A marker on an id the baseline charter never listed is not a re-red; a
    marker on an id it did list still is.
    """
    root = _repository(tmp_path, {"test_acceptance_a.py": ONE_RED_PROPERTY})
    (root / "docs" / "graph-acceptance.md").write_text(
        "| Id | Property |\n|---|---|\n| A1 | one |\n| A3 | three |\n| A9 | nine |\n"
    )
    target = root / "packages" / "microcosm-graph" / "tests" / "test_acceptance_a.py"
    target.write_text(NEW_PROPERTY_STARTS_RED)
    admitted = _run(root, "--verify")
    assert admitted.returncode == 0, admitted.stdout
    assert "1 -> 1 (+1 new: A9)" in admitted.stdout
    assert "verification=ok" in admitted.stdout

    # The same marker on a property the baseline charter already listed
    # (A3, green there) is a re-red and still fails.
    target.write_text(
        NEW_PROPERTY_STARTS_RED.replace(
            "def test_a3_three() -> None:\n    assert True",
            '@pytest.mark.xfail(strict=True, reason="charter A3: pending")\n'
            "def test_a3_three() -> None:\n    assert False",
        )
    )
    refused = _run(root, "--verify")
    assert refused.returncode == 1
    assert "re-reds 1 property" in refused.stdout
