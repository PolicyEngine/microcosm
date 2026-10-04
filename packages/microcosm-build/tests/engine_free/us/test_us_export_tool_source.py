"""What the export sampler and probe record about the code they ran.

``tools/sample_us_export_households.py`` and ``tools/probe_us_post_export.py``
each record the repository state they loaded from (``_git_state``: HEAD,
whether the working tree under ``tools/`` or ``packages/`` differs from HEAD's
tree, and a digest of those differences) and compare it with the state when
they write their receipt or report. Invariants, on real git repositories:

- the state is a function of HEAD and the watched working tree's bytes and
  modes: the digest is ``None`` exactly when they equal HEAD's, equal states
  mean equal watched trees and equal watched trees give equal states, so
  staging, rewriting a file with the same bytes, ignored files and changes
  outside ``tools/`` and ``packages/`` leave it unchanged;
- reading it never writes the index, and reading it twice gives the same
  state; the two tools' implementations agree on every state (differential);
- ``moved_since_load`` is exactly "the state differs from the one at load",
  ``None`` when either moment has no git state, and the record keeps every
  load-time field;
- a sibling tool the probe loads runs the bytes it hashes, even if its file
  changes right after that read, and its tracebacks quote those bytes; a
  module passed in is hashed when the record is written, and says so.

These tests load both tools without the engine.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import traceback
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
#: Ignored by the repository's .gitignore, under a watched path.
IGNORED = "tools/__pycache__/a.cpython-314.pyc"
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
def git_env(tmp_path_factory):
    """Git without the user's or the system's configuration or excludes."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
        patch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        patch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
        for role in ("AUTHOR", "COMMITTER"):
            patch.setenv(f"GIT_{role}_NAME", "test")
            patch.setenv(f"GIT_{role}_EMAIL", "test@example.invalid")
        yield


class Repo:
    """A git repository and a model of its files: path -> (bytes, executable)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q")
        self.files: dict[str, tuple[bytes, bool]] = {}
        self.write(".gitignore", b"__pycache__/\n")
        self.write("tools/a.py", b"one\n")
        self.write("docs/x.md", b"x\n")
        self.commit()

    def git(self, *args: str) -> None:
        subprocess.run(
            ["git", "-C", str(self.root), *args], check=True, capture_output=True
        )

    def write(self, path: str, content: bytes) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        executable = self.files.get(path, (b"", False))[1]
        self.files[path] = (content, executable)

    def touch(self, path: str) -> None:
        """Rewrite a file with its own bytes: new stat data, same content."""
        if path in self.files:
            (self.root / path).write_bytes(self.files[path][0])

    def chmod(self, path: str) -> None:
        if path in self.files:
            content, executable = self.files[path]
            (self.root / path).chmod(0o644 if executable else 0o755)
            self.files[path] = (content, not executable)

    def delete(self, path: str) -> None:
        if path in self.files:
            (self.root / path).unlink()
            del self.files[path]

    def write_ignored(self, content: bytes) -> None:
        target = self.root / IGNORED
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def stage(self) -> None:
        self.git("add", "-A")

    def commit(self) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", "state")
        self.committed = dict(self.files)

    def watched(self, files) -> tuple:
        return tuple(sorted((p, v) for p, v in files.items() if _watched(p)))

    def index_stat(self) -> tuple:
        info = (self.root / ".git" / "index").stat()
        return (info.st_ino, info.st_mtime_ns, info.st_size)


NEUTRAL = ("stage", "touch", "write_ignored")

operations = st.lists(
    st.one_of(
        st.tuples(st.just("write"), st.sampled_from(PATHS), st.sampled_from(CONTENTS)),
        st.tuples(st.just("delete"), st.sampled_from(PATHS)),
        st.tuples(st.just("touch"), st.sampled_from(PATHS)),
        st.tuples(st.just("chmod"), st.sampled_from(PATHS)),
        st.tuples(st.just("write_ignored"), st.sampled_from(CONTENTS)),
        st.tuples(st.just("stage")),
        st.tuples(st.just("commit")),
    ),
    min_size=1,
    max_size=7,
)


def _apply(repo: Repo, step: tuple) -> None:
    kind = step[0]
    if kind == "write":
        repo.write(step[1], step[2])
    elif kind == "delete":
        repo.delete(step[1])
    elif kind == "touch":
        repo.touch(step[1])
    elif kind == "chmod":
        repo.chmod(step[1])
    elif kind == "write_ignored":
        repo.write_ignored(step[1])
    elif kind == "stage":
        repo.stage()
    elif kind == "commit":
        repo.commit()


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(steps=operations)
# Pinned so every run covers each case that once broke or could break the
# state: an untracked file edited twice; a tracked file edited twice in a
# dirty tree; a tracked binary file edited twice; an edit after staging; two
# fully staged edits; a staged edit undone in the working tree; a new file
# staged and then deleted; two deletion-only trees; a mode change and its
# undoing; a same-bytes rewrite beside a real edit (which porcelain
# `git diff` answers by rewriting the index); edits outside the watched paths.
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
@example(
    steps=[
        ("write", "tools/a.py", b"two\n"),
        ("stage",),
        ("write", "tools/a.py", b"one\ntwo\n"),
        ("stage",),
    ]
)
@example(
    steps=[
        ("write", "tools/a.py", b"two\n"),
        ("stage",),
        ("write", "tools/a.py", b"one\n"),
    ]
)
@example(steps=[("write", "tools/b.bin", b""), ("stage",), ("delete", "tools/b.bin")])
@example(
    steps=[
        ("write", "packages/p/src/m.py", b"one\n"),
        ("commit",),
        ("delete", "tools/a.py"),
        ("delete", "packages/p/src/m.py"),
    ]
)
@example(steps=[("chmod", "tools/a.py"), ("chmod", "tools/a.py")])
@example(
    steps=[
        ("write", "packages/p/src/m.py", b"one\n"),
        ("commit",),
        ("write", "tools/a.py", b"two\n"),
        ("touch", "packages/p/src/m.py"),
    ]
)
@example(
    steps=[
        ("write", "docs/x.md", b"two\n"),
        ("write_ignored", b"one\n"),
        ("delete", "docs/x.md"),
    ]
)
def test_git_state_is_a_function_of_the_watched_tree(
    sampler, probe_tool, git_env, tmp_path_factory, steps
) -> None:
    repo = Repo(tmp_path_factory.mktemp("repo"))
    by_state: dict[tuple, tuple] = {}
    by_tree: dict[tuple, tuple] = {}
    previous = None
    for step in [("start",), *steps]:
        _apply(repo, step)
        index_before = repo.index_stat()
        state = sampler._git_state(repo.root)
        # Differential: the probe's copy reads the same state; determinism.
        assert probe_tool._git_state(repo.root) == state
        assert sampler._git_state(repo.root) == state
        # Reading the state never writes the index.
        assert repo.index_stat() == index_before
        head, dirty, changes = state
        assert head is not None
        tree = repo.watched(repo.files)
        assert dirty is (tree != repo.watched(repo.committed))
        assert (changes is None) is (not dirty)
        # Equal states mean equal watched trees, and the converse.
        assert by_state.setdefault((head, changes), (head, tree)) == (head, tree)
        assert by_tree.setdefault((head, tree), (head, changes)) == (head, changes)
        neutral = step[0] in NEUTRAL or (
            step[0] in ("write", "delete") and not _watched(step[1])
        )
        if previous is not None and neutral:
            assert state == previous, step
        previous = state


def test_git_state_without_a_repository(sampler, probe_tool, tmp_path) -> None:
    for tool in (sampler, probe_tool):
        assert tool._git_state(tmp_path) == (None, None, None)


def test_git_state_reads_an_untracked_symlink(sampler, git_env, tmp_path) -> None:
    repo = Repo(tmp_path / "repo")
    (repo.root / "tools" / "link").symlink_to("a.py")
    first = sampler._git_state(repo.root)
    (repo.root / "tools" / "link").unlink()
    (repo.root / "tools" / "link").symlink_to("b.py")
    second = sampler._git_state(repo.root)
    assert first[1] and second[1] and first[2] != second[2]


@pytest.mark.parametrize("tool_name", ["sampler", "probe_tool"])
def test_moved_since_load_compares_the_two_states(
    tool_name, request, monkeypatch
) -> None:
    tool = request.getfixturevalue(tool_name)
    monkeypatch.setattr(
        tool,
        "_TOOL_SOURCE",
        {**tool._TOOL_SOURCE, "commit": "a" * 40, "changes_sha256": "1" * 64},
    )
    assert tool._moved("a" * 40, "1" * 64) is False
    assert tool._moved("b" * 40, "1" * 64) is True  # HEAD moved
    assert tool._moved("a" * 40, "2" * 64) is True  # an edit within a dirty tree
    assert tool._moved("a" * 40, None) is True  # the edits were undone
    assert tool._moved(None, None) is None  # no git when writing
    monkeypatch.setitem(tool._TOOL_SOURCE, "commit", None)
    assert tool._moved("a" * 40, "1" * 64) is None  # no git at load


def _record(tool, tool_name):
    return (
        tool._tool_source_record()
        if tool_name == "sampler"
        else tool._tool_source_record({})
    )


@pytest.mark.parametrize("tool_name", ["sampler", "probe_tool"])
def test_the_record_keeps_every_load_field(tool_name, request, monkeypatch) -> None:
    """The load fields come from load time whatever the write-time state is,
    and a changed digest under the same HEAD counts as a move."""
    tool = request.getfixturevalue(tool_name)
    loaded = {
        "commit": "a" * 40,
        "dirty": True,
        "changes_sha256": "1" * 64,
        "sha256": "s" * 64,
        "installed_distributions": {"count": 1, "sha256": "d" * 64},
    }
    monkeypatch.setattr(tool, "_TOOL_SOURCE", loaded)
    for at_write, moved in (
        (("a" * 40, True, "1" * 64), False),
        (("a" * 40, True, "2" * 64), True),
        (("a" * 40, False, None), True),
        (("b" * 40, True, "1" * 64), True),
    ):
        monkeypatch.setattr(tool, "_git_state", lambda *a, state=at_write: state)
        record = _record(tool, tool_name)
        assert {key: record[key] for key in LOAD_STATE_FIELDS} == loaded
        assert (
            record["commit_at_write"],
            record["dirty_at_write"],
            record["changes_sha256_at_write"],
        ) == at_write
        assert record["moved_since_load"] is moved


@pytest.mark.parametrize("tool_name", ["sampler", "probe_tool"])
def test_the_load_hash_is_of_the_bytes_read_before_the_heavy_imports(
    tool_name, request
) -> None:
    tool = request.getfixturevalue(tool_name)
    expected = hashlib.sha256(tool._SOURCE_AT_LOAD).hexdigest()
    assert tool._TOOL_SOURCE["sha256"] == expected
    assert tool._SOURCE_AT_LOAD == Path(tool.__file__).read_bytes()
    environment = tool._installed_distributions()
    assert environment == tool._TOOL_SOURCE["installed_distributions"]
    assert environment["count"] > 0


@pytest.mark.parametrize("tool_name", ["sampler", "probe_tool"])
def test_the_record_keeps_the_load_state_after_head_moves(
    tool_name, request, monkeypatch
) -> None:
    tool = request.getfixturevalue(tool_name)
    move_head_after_load(monkeypatch)
    record = _record(tool, tool_name)
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


def boom():
    raise RuntimeError("original line")
"""


def test_load_tool_runs_the_bytes_it_hashes(probe_tool, tmp_path, monkeypatch) -> None:
    """The probe's ``_load_tool`` executes the bytes it hashed, even when the
    file changes right after that read; tracebacks quote those bytes; and the
    module keeps that hash. A loader that hashed one read and executed a
    second would run ``Point(2)`` here."""
    monkeypatch.setattr(probe_tool, "_TOOLS", tmp_path)
    sibling = tmp_path / "sibling.py"
    sibling.write_bytes(_SIBLING)
    edited = _SIBLING.replace(b"(1)", b"(2)").replace(b"original line", b"edited line")
    real_read = Path.read_bytes

    def read_then_edit(self):
        data = real_read(self)
        if self == sibling:
            sibling.write_bytes(edited)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_edit)
    name = "probe_tool_source_sibling"
    try:
        module = probe_tool._load_tool(name, "sibling.py")
        monkeypatch.setattr(Path, "read_bytes", real_read)
        assert sibling.read_bytes() == edited
        loaded = hashlib.sha256(_SIBLING).hexdigest()
        assert module.VALUE == 1
        assert sys.modules[name] is module
        assert module.__loaded_sha256__ == loaded
        assert probe_tool._module_sha256(module) == {
            "sha256": loaded,
            "file": str(sibling),
            "hashed": "at load",
        }
        with pytest.raises(RuntimeError) as raised:
            module.boom()
        quoted = "".join(traceback.format_exception(raised.value))
        assert 'raise RuntimeError("original line")' in quoted
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
    assert probe_tool._module_sha256(SimpleNamespace(__file__=str(path))) == {
        "sha256": hashlib.sha256(b"x = 1\n").hexdigest(),
        "file": str(path),
        "hashed": "at write",
    }
    assert probe_tool._module_sha256(SimpleNamespace()) == {
        "sha256": None,
        "file": None,
        "hashed": None,
    }
    missing = SimpleNamespace(__file__=str(tmp_path / "gone.py"))
    gone = probe_tool._module_sha256(missing)
    assert gone["sha256"] is None
    assert gone["hashed"].startswith("unreadable at write")
