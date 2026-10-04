"""What the export sampler and probe record about the code they ran.

``tools/sample_us_export_households.py`` and ``tools/probe_us_post_export.py``
each record the repository state they loaded from (``_git_state``: HEAD,
whether ``tools/`` or ``packages/`` differ from it, and a digest of those
differences) and compare it with the state when they write their receipt or
report. Invariants, on real git repositories:

- the digest is ``None`` exactly when the watched tree equals HEAD's, and any
  change to the watched content (tracked or untracked, text or binary) changes
  ``(HEAD, digest)``: equal records mean equal watched trees;
- changes outside ``tools/`` and ``packages/`` leave the record unchanged, and
  reading it twice gives the same record (determinism);
- the two tools' implementations agree on every state (differential);
- ``moved_since_load`` is exactly "the state differs from the one at load",
  ``None`` when either moment has no git state;
- a sibling tool the probe loads runs the bytes it hashes, and keeps that hash
  after its file changes; a module passed in is hashed when the record is
  written, and says so.

These tests load both tools without the engine.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from test_support.microcosm_build.us_export_subsample import *

WATCHED = ("tools/", "packages/")
PATHS = (
    "tools/a.py",
    "tools/b.bin",
    "packages/p/src/m.py",
    "packages/p/tests/t.py",
    "docs/x.md",
    "README.md",
)
CONTENTS = (
    b"",
    b"one\n",
    b"two\n",
    b"one\ntwo\n",
    b"\x00\xff\x10binary",
    b"\x00\xfe\x11other",
)


def _watched(path: str) -> bool:
    return path.startswith(WATCHED)


@pytest.fixture(scope="module")
def git_env():
    """Git without the user's or the system's configuration."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
        patch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        for role in ("AUTHOR", "COMMITTER"):
            patch.setenv(f"GIT_{role}_NAME", "test")
            patch.setenv(f"GIT_{role}_EMAIL", "test@example.invalid")
        yield


class Repo:
    """A git repository and a model of its watched content."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q", "-b", "main")
        self.files: dict[str, bytes] = {"tools/a.py": b"one\n", "docs/x.md": b"x\n"}
        for path, content in self.files.items():
            self._write(path, content)
        self.commit()

    def git(self, *args: str) -> None:
        subprocess.run(
            ["git", "-C", str(self.root), *args], check=True, capture_output=True
        )

    def _write(self, path: str, content: bytes) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def write(self, path: str, content: bytes) -> None:
        self._write(path, content)
        self.files[path] = content

    def delete(self, path: str) -> None:
        if path in self.files:
            (self.root / path).unlink()
            del self.files[path]

    def commit(self) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", "state")
        self.committed = dict(self.files)

    def stage(self) -> None:
        self.git("add", "-A")

    def watched(self, files: dict[str, bytes]) -> dict[str, bytes]:
        return {path: data for path, data in files.items() if _watched(path)}

    def snapshot(self) -> tuple:
        return tuple(sorted(self.watched(self.files).items()))


operations = st.lists(
    st.one_of(
        st.tuples(st.just("write"), st.sampled_from(PATHS), st.sampled_from(CONTENTS)),
        st.tuples(st.just("delete"), st.sampled_from(PATHS)),
        st.tuples(st.just("stage")),
        st.tuples(st.just("commit")),
    ),
    min_size=1,
    max_size=6,
)


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(steps=operations)
# Edits that each need a part of the digest, pinned so every run covers them:
# an untracked file edited twice; a tracked file edited twice within a dirty
# tree; a tracked binary file edited twice (needs --binary); an edit after
# staging; an edit under packages/; an edit outside the watched paths.
@example(steps=[("write", "tools/b.bin", b"one\n"), ("write", "tools/b.bin", b"two\n")])
@example(
    steps=[("write", "tools/a.py", b"two\n"), ("write", "tools/a.py", b"one\ntwo\n")]
)
@example(
    steps=[
        ("write", "tools/b.bin", CONTENTS[4]),
        ("commit",),
        ("write", "tools/b.bin", CONTENTS[5]),
        ("write", "tools/b.bin", CONTENTS[4] + CONTENTS[5]),
    ]
)
@example(
    steps=[
        ("write", "packages/p/src/m.py", b"one\n"),
        ("stage",),
        ("write", "packages/p/src/m.py", b"two\n"),
    ]
)
@example(steps=[("write", "docs/x.md", b"two\n"), ("delete", "docs/x.md")])
def test_git_state_records_the_watched_tree(
    sampler, probe_tool, git_env, tmp_path_factory, steps
) -> None:
    repo = Repo(tmp_path_factory.mktemp("repo"))
    seen: dict[tuple, tuple] = {}
    previous = None
    for step in [("start",), *steps]:
        kind = step[0]
        if kind == "write":
            repo.write(step[1], step[2])
        elif kind == "delete":
            repo.delete(step[1])
        elif kind == "stage":
            repo.stage()
        elif kind == "commit":
            repo.commit()
        state = sampler._git_state(repo.root)
        # Differential: the probe's copy reads the same state; determinism.
        assert probe_tool._git_state(repo.root) == state
        assert sampler._git_state(repo.root) == state
        head, dirty, changes = state
        assert head is not None
        expected_dirty = repo.watched(repo.files) != repo.watched(repo.committed)
        assert dirty is expected_dirty
        assert (changes is None) is (not dirty)
        # Equal records mean equal watched trees.
        key = (head, changes)
        snapshot = repo.snapshot()
        assert seen.setdefault(key, snapshot) == snapshot
        # A change outside tools/ and packages/ leaves the record unchanged.
        if (
            previous is not None
            and kind in ("write", "delete")
            and not _watched(step[1])
        ):
            assert state == previous
        previous = state


def test_git_state_without_a_repository(sampler, probe_tool, tmp_path) -> None:
    for tool in (sampler, probe_tool):
        assert tool._git_state(tmp_path) == (None, None, None)


@pytest.mark.parametrize("tool_name", ["sampler", "probe_tool"])
def test_moved_since_load_compares_the_two_states(
    tool_name, request, monkeypatch
) -> None:
    tool = request.getfixturevalue(tool_name)
    monkeypatch.setattr(
        tool,
        "_TOOL_SOURCE",
        {"commit": "a" * 40, "dirty": True, "changes_sha256": "1" * 64, "sha256": "x"},
    )
    assert tool._moved("a" * 40, "1" * 64) is False
    assert tool._moved("b" * 40, "1" * 64) is True  # HEAD moved
    assert tool._moved("a" * 40, "2" * 64) is True  # an edit within a dirty tree
    assert tool._moved("a" * 40, None) is True  # the edits were undone
    assert tool._moved(None, None) is None  # no git when writing
    monkeypatch.setitem(tool._TOOL_SOURCE, "commit", None)
    assert tool._moved("a" * 40, "1" * 64) is None  # no git at load


@pytest.mark.parametrize("tool_name", ["sampler", "probe_tool"])
def test_the_record_keeps_the_load_state(tool_name, request, monkeypatch) -> None:
    tool = request.getfixturevalue(tool_name)
    move_head_after_load(monkeypatch)
    record = (
        tool._tool_source_record()
        if tool_name == "sampler"
        else tool._tool_source_record({})
    )
    assert {key: record[key] for key in LOAD_STATE_FIELDS} == tool._TOOL_SOURCE
    assert record["commit_at_write"] == LATER_HEAD
    assert record["dirty_at_write"] is False
    assert record["changes_sha256_at_write"] is None
    expected = True if tool._TOOL_SOURCE["commit"] is not None else None
    assert record["moved_since_load"] is expected


_SIBLING = b"""from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Point:
    x: int


VALUE = Point(1).x
"""


def test_load_tool_runs_the_bytes_it_hashes(probe_tool, tmp_path, monkeypatch) -> None:
    """The probe's ``_load_tool`` executes the bytes it hashes (a dataclass
    with string annotations resolves through ``sys.modules``), and the module
    keeps that hash after its file changes."""
    monkeypatch.setattr(probe_tool, "_TOOLS", tmp_path)
    (tmp_path / "sibling.py").write_bytes(_SIBLING)
    name = "probe_tool_source_sibling"
    try:
        module = probe_tool._load_tool(name, "sibling.py")
        loaded = hashlib.sha256(_SIBLING).hexdigest()
        assert module.VALUE == 1
        assert sys.modules[name] is module
        assert module.__loaded_sha256__ == loaded
        (tmp_path / "sibling.py").write_bytes(_SIBLING.replace(b"(1)", b"(2)"))
        assert probe_tool._module_sha256(module) == {
            "sha256": loaded,
            "hashed": "at load",
        }
        # A second load reuses the module that ran.
        assert probe_tool._load_tool(name, "sibling.py") is module
    finally:
        sys.modules.pop(name, None)


def test_load_tool_leaves_no_module_behind_on_failure(
    probe_tool, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(probe_tool, "_TOOLS", tmp_path)
    (tmp_path / "broken.py").write_bytes(b"raise RuntimeError('boom')\n")
    with pytest.raises(RuntimeError, match="boom"):
        probe_tool._load_tool("probe_tool_source_broken", "broken.py")
    assert "probe_tool_source_broken" not in sys.modules


def test_a_module_passed_in_is_hashed_at_write(probe_tool, tmp_path) -> None:
    path = tmp_path / "passed.py"
    path.write_bytes(b"x = 1\n")
    passed = SimpleNamespace(__file__=str(path))
    assert probe_tool._module_sha256(passed) == {
        "sha256": hashlib.sha256(b"x = 1\n").hexdigest(),
        "hashed": "at write",
    }
    assert probe_tool._module_sha256(SimpleNamespace()) == {
        "sha256": None,
        "hashed": None,
    }
