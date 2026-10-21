"""What the export sampler and probe record about the code they ran.

``tools/sample_us_export_households.py`` and ``tools/probe_us_post_export.py``
each record, before their heavy imports, their own bytes and the repository
state (``_git_state``: HEAD, whether the working tree under ``tools/`` or
``packages/`` differs from HEAD's tree, and a digest of the differences), and
compare that state with the one when they write their receipt or report.
Invariants, on real git repositories:

- the state is a function of HEAD and the watched working tree's bytes and
  modes: the digest is ``None`` exactly when they equal HEAD's, equal states
  mean equal watched trees and equal watched trees give equal states, so
  staging, rewriting a file with the same bytes, ignored files and changes
  outside ``tools/`` and ``packages/`` leave it unchanged, while index flags
  and ``core.filemode`` hide nothing;
- reading it never writes the index, and reading it twice gives the same
  state; the two tools' implementations agree on every state (differential);
- a file that keeps changing while read, a tracked path replaced by a
  special file, an unreadable file and a nested repository at another commit
  each count as a difference, never as HEAD's bytes;
- the load fields are read before the heavy imports, the record keeps them,
  and ``moved_since_load`` is exactly "the state differs from the one at
  load", ``None`` when either moment has no git state;
- a sibling tool the probe loads runs the bytes it hashes, compiled without
  the probe's own future flags, and tracebacks quote the bytes that ran.

These tests load both tools without the engine.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import linecache
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


def _both(sampler, probe_tool, root) -> tuple:
    state = sampler._git_state(root)
    assert probe_tool._git_state(root) == state
    return state


def test_git_state_reads_past_the_index(sampler, probe_tool, git_env, tmp_path) -> None:
    """The assume-unchanged and skip-worktree bits and core.filemode=false make
    git skip a file; here its bytes and mode are still read. A skip-worktree
    file missing from the working tree counts as unchanged, as git counts
    it."""
    repo = Repo(tmp_path / "repo")
    repo.write("tools/b.py", b"b\n")
    repo.write("packages/p/src/m.py", b"m\n")
    repo.commit()
    clean = _both(sampler, probe_tool, repo.root)
    assert clean[1] is False

    repo.git("update-index", "--assume-unchanged", "tools/a.py")
    (repo.root / "tools/a.py").write_bytes(b"edited\n")
    assert _both(sampler, probe_tool, repo.root)[1] is True
    (repo.root / "tools/a.py").write_bytes(b"one\n")
    assert _both(sampler, probe_tool, repo.root) == clean

    repo.git("update-index", "--skip-worktree", "tools/b.py")
    (repo.root / "tools/b.py").write_bytes(b"edited\n")
    assert _both(sampler, probe_tool, repo.root)[1] is True
    (repo.root / "tools/b.py").unlink()
    assert _both(sampler, probe_tool, repo.root) == clean

    repo.git("config", "core.filemode", "false")
    (repo.root / "packages/p/src/m.py").chmod(0o755)
    assert _both(sampler, probe_tool, repo.root)[1] is True


def test_git_state_reads_nested_repositories(
    sampler, probe_tool, git_env, tmp_path
) -> None:
    """A submodule counts as unchanged when it is not checked out or is at
    the commit HEAD records, and as a difference at any other commit; an
    untracked clone under a watched path is recorded by its commit."""
    nested = Repo(tmp_path / "nested")
    first = _head(nested)
    repo = Repo(tmp_path / "repo")
    repo.git("update-index", "--add", "--cacheinfo", f"160000,{first},tools/sub")
    repo.git("commit", "-q", "-m", "submodule")
    clean = _both(sampler, probe_tool, repo.root)
    assert clean[1] is False  # recorded, not checked out
    subprocess.run(
        ["git", "clone", "-q", str(nested.root), str(repo.root / "tools/sub")],
        check=True,
        capture_output=True,
    )
    assert _both(sampler, probe_tool, repo.root) == clean  # at the recorded commit
    moved = Repo.__new__(Repo)
    moved.root = repo.root / "tools/sub"
    moved.git("commit", "-q", "--allow-empty", "-m", "later")
    assert _both(sampler, probe_tool, repo.root)[1] is True

    untracked = repo.root / "packages/clone"
    subprocess.run(
        ["git", "clone", "-q", str(nested.root), str(untracked)],
        check=True,
        capture_output=True,
    )
    before = _both(sampler, probe_tool, repo.root)
    clone = Repo.__new__(Repo)
    clone.root = untracked
    clone.git("commit", "-q", "--allow-empty", "-m", "later")
    after = _both(sampler, probe_tool, repo.root)
    assert before[1] is after[1] is True
    assert before[2] != after[2]


def _blob(object_format: str, content: bytes) -> str:
    return hashlib.new(object_format, b"blob %d\0" % len(content) + content).hexdigest()


def test_a_file_changing_while_read_is_read_again_or_unstable(
    sampler, tmp_path, monkeypatch
) -> None:
    target = tmp_path / "f.py"
    target.write_bytes(b"x\n")
    real_lstat = Path.lstat
    changing = {"calls": 0, "until": 0}

    def lstat(self, *args, **kwargs):
        info = real_lstat(self, *args, **kwargs)
        if self != target:
            return info
        changing["calls"] += 1
        if changing["calls"] > changing["until"]:
            return info
        return SimpleNamespace(
            st_ino=info.st_ino,
            st_dev=info.st_dev,
            st_size=info.st_size,
            st_mode=info.st_mode,
            st_mtime_ns=info.st_mtime_ns + changing["calls"],
        )

    monkeypatch.setattr(Path, "lstat", lstat)
    changing["until"] = 2  # the first read sees a change; the second does not
    entry = sampler._worktree_entry(target, "sha1")
    assert entry == (
        "100644",
        _blob("sha1", b"x\n"),
        hashlib.sha256(b"x\n").hexdigest(),
    )
    changing.update(calls=0, until=10**6)  # it never settles
    assert sampler._worktree_entry(target, "sha1") == (
        "unstable",
        "",
        "changed while read",
    )


def test_git_state_never_reads_special_or_unreadable_files_as_head(
    sampler, probe_tool, git_env, tmp_path
) -> None:
    repo = Repo(tmp_path / "repo")
    clean = _both(sampler, probe_tool, repo.root)
    if hasattr(os, "mkfifo"):
        # git cannot track a FIFO, so an untracked one is not listed; a
        # tracked path replaced by one is read as special, never opened.
        untracked = repo.root / "tools" / "pipe"
        os.mkfifo(untracked)
        assert sampler._worktree_entry(untracked, "sha1")[0] == "special"
        assert _both(sampler, probe_tool, repo.root) == clean
        untracked.unlink()
        tracked = repo.root / "tools" / "a.py"
        tracked.unlink()
        os.mkfifo(tracked)
        assert _both(sampler, probe_tool, repo.root)[1] is True
        tracked.unlink()
        tracked.write_bytes(b"one\n")
    assert _both(sampler, probe_tool, repo.root) == clean
    if os.geteuid() != 0:
        tracked = repo.root / "tools" / "a.py"
        tracked.chmod(0)
        try:
            entry = sampler._worktree_entry(tracked, "sha1")
            assert entry[0] == "unreadable"
            assert _both(sampler, probe_tool, repo.root)[1] is True
        finally:
            tracked.chmod(0o644)
    assert _both(sampler, probe_tool, repo.root) == clean


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


TOOL_FILES = {
    "sampler": "sample_us_export_households.py",
    "probe_tool": "probe_us_post_export.py",
}


def _head(repo: Repo) -> str:
    return subprocess.run(
        ["git", "-C", str(repo.root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.parametrize("tool_name", ["sampler", "probe_tool"])
def test_the_load_state_is_read_before_the_heavy_imports(
    tool_name, request, git_env, tmp_path, monkeypatch
) -> None:
    """A copy of the tool in its own repository is edited and committed while
    it imports numpy. Its record keeps the bytes that run, the commit and the
    clean tree from before that import, and tracebacks quote those bytes;
    the state at write shows the move."""
    tool = request.getfixturevalue(tool_name)
    filename = TOOL_FILES[tool_name]
    original = Path(tool.__file__).read_bytes()
    repo = Repo(tmp_path / "repo")
    repo.write(f"tools/{filename}", original)
    repo.commit()
    loaded_commit = _head(repo)
    copy = repo.root / "tools" / filename
    name = f"tool_source_copy_{tool_name}"
    real_import = builtins.__import__
    fired: list[bool] = []

    def import_then_edit(module_name, globals=None, *args, **kwargs):
        if (
            module_name == "numpy"
            and not fired
            and (globals or {}).get("__name__") == name
        ):
            fired.append(True)
            repo.write(f"tools/{filename}", original + b"\n# edited during import\n")
            repo.commit()
        return real_import(module_name, globals, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(name, copy)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with monkeypatch.context() as during_import:
            during_import.setattr(builtins, "__import__", import_then_edit)
            spec.loader.exec_module(module)
        assert fired
        source = module._TOOL_SOURCE
        assert source["sha256"] == hashlib.sha256(original).hexdigest()
        assert source["commit"] == loaded_commit
        assert source["dirty"] is False
        assert source["changes_sha256"] is None
        assert linecache.getlines(str(copy)) == original.decode().splitlines(True)
        record = _record(module, tool_name)
        assert record["commit_at_write"] == _head(repo) != loaded_commit
        assert record["moved_since_load"] is True
        inventory = module._installed_distributions()
        assert inventory == source["installed_distributions"]
        assert inventory["count"] > 0
    finally:
        sys.modules.pop(name, None)
        linecache.cache.pop(str(copy), None)


@pytest.mark.parametrize("tool_name", ["sampler", "probe_tool"])
def test_the_distribution_inventory_never_raises(
    tool_name, request, monkeypatch
) -> None:
    """A distribution whose metadata cannot be read costs only its own line,
    and a failure to list distributions is recorded, not raised."""
    import importlib.metadata as metadata

    tool = request.getfixturevalue(tool_name)

    class Unreadable:
        _path = "/nowhere/broken.dist-info"
        version = "1"

        @property
        def metadata(self):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    readable = SimpleNamespace(metadata={"Name": "Readable"}, version="2.0")
    monkeypatch.setattr(metadata, "distributions", lambda: [readable, Unreadable()])
    inventory = tool._installed_distributions()
    assert inventory["count"] == 2
    assert inventory["unreadable"] == 1
    assert len(inventory["sha256"]) == 64

    def unavailable():
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(metadata, "distributions", unavailable)
    inventory = tool._installed_distributions()
    assert inventory["count"] is None
    assert inventory["error"].startswith("OSError")


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


_SIBLING_WITHOUT_FUTURE = b"""def f(x: int) -> int:
    return x


ANNOTATION = f.__annotations__["x"]
"""


def test_load_tool_compiles_without_the_probes_future_flags(
    probe_tool, tmp_path, monkeypatch
) -> None:
    """The probe has ``from __future__ import annotations``; a sibling without
    it keeps real annotations, as the spec loader would give it."""
    monkeypatch.setattr(probe_tool, "_TOOLS", tmp_path)
    (tmp_path / "plain.py").write_bytes(_SIBLING_WITHOUT_FUTURE)
    name = "probe_tool_source_plain"
    try:
        module = probe_tool._load_tool(name, "plain.py")
        assert module.ANNOTATION is int
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
