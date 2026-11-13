"""The transport CLI runs when started as a module or through its driver.

``python -m microcosm.build.transport.cli`` once imported the module and exited
0 without calling ``main`` (microcosm#1604): the module had no ``__main__``
block. A wrapper that reads only the exit code would take that for a finished
build.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from test_support.paths import paths_for

ROOT = paths_for("microcosm-build").repository
ENTRY_FORMS = pytest.mark.parametrize(
    "command",
    [["-m", "microcosm.build.transport.cli"], ["tools/build_transport.py"]],
    ids=["module", "driver"],
)


def _run(command: list[str], *arguments: str) -> subprocess.CompletedProcess[str]:
    # Prefer this worktree's sources to editable installs in another checkout.
    pythonpath = os.pathsep.join(
        [
            *(str(path) for path in sorted((ROOT / "packages").glob("*/src"))),
            str(ROOT),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    return subprocess.run(
        [sys.executable, *command, *arguments],
        cwd=ROOT,
        # argparse colours its usage text when the caller's shell asks for colour.
        env={**os.environ, "PYTHONPATH": pythonpath, "PYTHON_COLORS": "0"},
        capture_output=True,
        text=True,
        check=False,
    )


@ENTRY_FORMS
def test_each_entry_form_prints_the_clis_usage(command: list[str]) -> None:
    result = _run(command, "--help")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: "), result.stdout
    assert "--country" in result.stdout and "--resume" in result.stdout
    # Started as a module, cli must not already be imported by its package.
    assert "RuntimeWarning" not in result.stderr, result.stderr


@ENTRY_FORMS
def test_each_entry_form_exits_one_on_refused_arguments(command: list[str]) -> None:
    result = _run(command)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "transport build refused: invalid command-line arguments." in result.stderr
