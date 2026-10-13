"""The suite never runs git against a repository it inherited.

Git exports an absolute ``GIT_DIR`` to the hooks, ``rebase --exec`` commands
and shell aliases it runs in a linked worktree. The root ``conftest.py``
removes it, and the other variables that pick git's repository, before any
test runs (``test_support.git_isolation``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from test_support.git_isolation import (
    GIT_CONFIG_CARRIERS,
    drop_inherited_git_repository,
)
from test_support.paths import REPOSITORY_ROOT

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not installed"
)

_GIT_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", "/"),
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=_GIT_ENV if env is None else env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_variables() -> list[str]:
    if shutil.which("git") is None:
        return []
    listed = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return sorted(set(listed) - GIT_CONFIG_CARRIERS)


def test_repository_variables_are_dropped_and_the_rest_kept() -> None:
    environ = {
        "PATH": "/usr/bin",
        "GIT_DIR": "/elsewhere/.git/worktrees/x",
        "GIT_INDEX_FILE": "/elsewhere/.git/worktrees/x/index",
        "GIT_WORK_TREE": "/elsewhere",
        "GIT_CONFIG_PARAMETERS": "'core.quotepath'='false'",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_TERMINAL_PROMPT": "0",
    }
    dropped = drop_inherited_git_repository(environ)
    assert dropped == ["GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"]
    assert environ == {
        "PATH": "/usr/bin",
        "GIT_CONFIG_PARAMETERS": "'core.quotepath'='false'",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_TERMINAL_PROMPT": "0",
    }


@settings(max_examples=50, deadline=None)
@given(
    environ=st.dictionaries(
        st.sampled_from(
            [
                *_repository_variables(),
                *sorted(GIT_CONFIG_CARRIERS),
                "PATH",
                "GIT_TERMINAL_PROMPT",
                "GIT_AUTHOR_NAME",
            ]
        ),
        st.text(max_size=6),
        max_size=10,
    )
)
def test_no_repository_variable_survives_and_nothing_else_changes(environ) -> None:
    # For any inherited environment: every variable git calls
    # repository-local is removed and named, bar the ``git -c`` carriers,
    # and every other variable keeps its value. A second call removes nothing.
    before = dict(environ)
    variables = set(_repository_variables())
    dropped = drop_inherited_git_repository(environ)
    assert dropped == sorted(variables & set(before))
    assert not variables & set(environ)
    assert environ == {k: v for k, v in before.items() if k not in variables}
    assert drop_inherited_git_repository(environ) == []


def test_a_missing_git_removes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    environ = {"GIT_DIR": "/elsewhere/.git"}
    assert drop_inherited_git_repository(environ) == []
    assert environ == {"GIT_DIR": "/elsewhere/.git"}


def test_the_root_conftest_keeps_a_fixture_s_commit_out_of_an_inherited_repository(
    tmp_path: Path,
) -> None:
    # What a hook in a linked worktree hands the suite, and what a fixture
    # then does: ``git init`` and ``commit`` in a temporary directory, with
    # the environment it inherited. Under GIT_DIR alone that commit lands on
    # the bystander's checked-out branch.
    bystander = tmp_path / "bystander"
    bystander.mkdir()
    _git(bystander, "init", "-q", "-b", "main")
    _git(bystander, "commit", "-q", "--allow-empty", "-m", "bystander")
    linked = tmp_path / "linked"
    _git(bystander, "worktree", "add", "-q", str(linked), "-b", "side")
    linked_git_dir = _git(linked, "rev-parse", "--absolute-git-dir")
    refs = _git(bystander, "for-each-ref")
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    script = (
        "import conftest, subprocess\n"
        "for args in (['init', '-q', '-b', 'main'],\n"
        "             ['commit', '-q', '--allow-empty', '-m', 'fixture'],\n"
        "             ['update-ref', 'refs/remotes/origin/main', 'HEAD']):\n"
        f"    subprocess.run(['git', *args], cwd={str(fixture)!r}, check=True)\n"
    )
    hook_env = {
        **_GIT_ENV,
        "GIT_DIR": linked_git_dir,
        "GIT_INDEX_FILE": f"{linked_git_dir}/index",
    }
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=hook_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert _git(bystander, "for-each-ref") == refs
    assert _git(fixture, "log", "--format=%s") == "fixture"
    # Without the conftest, the same commands move the bystander's branch.
    unguarded = script.replace("import conftest, subprocess", "import subprocess")
    subprocess.run(
        [sys.executable, "-c", unguarded],
        cwd=REPOSITORY_ROOT,
        env=hook_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert _git(bystander, "for-each-ref") != refs
