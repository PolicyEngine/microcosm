"""The one-time build-folder switch tool is gone.

The tool moved a dataset repository's builds, tagged or in untagged folders,
into build folders listed in ``releases.json``, deleted its pointer files, and
then deleted the old layout's files. The US, UK
and BE release repositories have all moved and had their old layout deleted,
so Microcosm no longer has it, and nothing tells an operator to run it or
configures anything for it. ``changelog.d/`` and the root journals are history
and may still name it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from test_support.paths import paths_for

#: The removed module's name, as an import, a ``python -m`` command or a path
#: would spell it.
REMOVED_MODULE_NAME = "build_folder_switch"
#: The file suffixes read under the scanned directories: the kinds of file a
#: source, test, tool, document or configuration is written in.
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".sh", ".toml", ".json", ".yml", ".yaml", ".txt", ".ipynb"}
)


def _scanned_directories(repository: Path) -> list[Path]:
    """The package sources and tests, the shared test support, the tools,
    the docs and the CI workflows."""
    packages = repository / "packages"
    return [
        *sorted(packages.glob("*/src")),
        *sorted(packages.glob("*/tests")),
        repository / "test_support",
        repository / "tools",
        repository / "docs",
        repository / ".github" / "workflows",
    ]


def _scanned_files(repository: Path) -> list[Path]:
    """Every file with a suffix in ``TEXT_SUFFIXES`` under the scanned
    directories, plus the READMEs, the ``pyproject.toml`` files (whose lint
    settings name modules) and the root ``conftest.py``."""
    files = [
        candidate
        for directory in _scanned_directories(repository)
        for candidate in directory.rglob("*")
        if candidate.is_file()
        and candidate.suffix in TEXT_SUFFIXES
        and "__pycache__" not in candidate.parts
    ]
    packages = repository / "packages"
    files.extend(
        [
            repository / "README.md",
            *sorted(packages.glob("*/README.md")),
            repository / "pyproject.toml",
            *sorted(packages.glob("*/pyproject.toml")),
            repository / "conftest.py",
        ]
    )
    return files


def test_the_build_folder_switch_module_is_removed() -> None:
    assert importlib.util.find_spec("microcosm.build.build_folder_switch") is None


def test_the_scan_reads_directories_and_files_that_exist() -> None:
    """A wrong root would make the scan below pass without reading anything."""
    repository = paths_for("microcosm-build").repository
    directories = _scanned_directories(repository)

    assert len([directory for directory in directories if directory.name == "src"]) > 1
    assert (
        len([directory for directory in directories if directory.name == "tests"]) > 1
    )
    assert [directory for directory in directories if not directory.is_dir()] == []
    assert [
        candidate for candidate in _scanned_files(repository) if not candidate.is_file()
    ] == []


def test_no_source_test_tool_document_or_setting_names_the_switch_tool() -> None:
    repository = paths_for("microcosm-build").repository
    this_file = Path(__file__).resolve()
    naming_files = sorted(
        str(candidate.relative_to(repository))
        for candidate in _scanned_files(repository)
        if candidate.resolve() != this_file
        and REMOVED_MODULE_NAME
        in candidate.read_text(encoding="utf-8", errors="replace")
    )

    assert naming_files == []
