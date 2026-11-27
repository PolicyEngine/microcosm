"""A transport build records the code and runtime it ran on (tracker 5b · 13).

The release validator requires each build to record its Microcosm commit,
from a clean checkout, and its Python version. US and UK builds record them
while building, and their release steps read them from the build's records.
``build-status.json`` carries the same record for a transport build, so the
spec release assembler reads it there.
"""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata
from pathlib import Path

from microcosm.build.transport.cli import (
    TRANSPORT_RUNTIME_DISTRIBUTIONS,
    transport_build_provenance,
)
from test_support.git_repositories import NO_BACKGROUND_MAINTENANCE


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *NO_BACKGROUND_MAINTENANCE, *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "config", "user.name", "Fixture")
    (root / "module.py").write_text("VALUE = 1\n")
    _git(root, "add", "module.py")
    _git(root, "commit", "--quiet", "-m", "Fixture commit")
    return root


def test_a_clean_checkout_records_its_commit_as_clean(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "checkout")

    provenance = transport_build_provenance(checkout=repository)

    assert provenance["code"] == {
        "repository": "PolicyEngine/microcosm",
        "git_commit": _git(repository, "rev-parse", "HEAD"),
        "git_dirty": False,
    }


def test_a_tracked_change_records_the_checkout_as_dirty(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "checkout")
    (repository / "module.py").write_text("VALUE = 2\n")

    provenance = transport_build_provenance(checkout=repository)

    assert provenance["code"]["git_dirty"] is True


def test_a_directory_outside_git_records_the_commit_as_unmeasured(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "not-a-checkout"
    outside.mkdir()

    provenance = transport_build_provenance(checkout=outside)

    # Unmeasured, never asserted clean: the assembler refuses it.
    assert provenance["code"]["git_commit"] is None
    assert provenance["code"]["git_dirty"] is None


def test_the_runtime_records_python_and_each_distribution(tmp_path: Path) -> None:
    provenance = transport_build_provenance(checkout=tmp_path)

    runtime = provenance["runtime"]
    assert runtime.get("python") == platform.python_version()
    assert set(runtime) == {"python", *TRANSPORT_RUNTIME_DISTRIBUTIONS}
    assert runtime["microcosm-build"] == metadata.version("microcosm-build")
    assert "axiom-rules-engine" in TRANSPORT_RUNTIME_DISTRIBUTIONS
