"""The one-time build-folder switch tool is gone.

The tool moved each dataset repository from tags and pointer files to build
folders and ``releases.json``, then deleted the old layout's files. The US, UK
and BE release repositories have all moved and had their old layout deleted,
so Microcosm no longer has it, and nothing tells an operator to run it.
``changelog.d/`` and the root journals are history and may still name it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from test_support.paths import paths_for

#: The removed module's name, as an import, a ``python -m`` command or a path
#: would spell it.
REMOVED_MODULE_NAME = "build_folder_switch"
#: The kinds of file a source, test, tool or document is written in.
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".sh", ".toml", ".json", ".yml", ".yaml", ".txt", ".ipynb"}
)


def _current_files(repository: Path) -> list[Path]:
    """Every text file of the package sources and tests, the shared test
    support, the tools, the docs and the READMEs."""
    roots = [
        *sorted((repository / "packages").glob("*/src")),
        *sorted((repository / "packages").glob("*/tests")),
        repository / "test_support",
        repository / "tools",
        repository / "docs",
    ]
    files = [
        candidate
        for root in roots
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.suffix in TEXT_SUFFIXES
        and "__pycache__" not in candidate.parts
    ]
    files.append(repository / "README.md")
    files.extend(sorted((repository / "packages").glob("*/README.md")))
    return files


def test_the_build_folder_switch_module_is_removed() -> None:
    assert importlib.util.find_spec("microcosm.build.build_folder_switch") is None


def test_no_source_test_tool_or_document_names_the_switch_tool() -> None:
    repository = paths_for("microcosm-build").repository
    this_file = Path(__file__).resolve()
    naming_files = sorted(
        str(candidate.relative_to(repository))
        for candidate in _current_files(repository)
        if candidate.resolve() != this_file
        and REMOVED_MODULE_NAME
        in candidate.read_text(encoding="utf-8", errors="replace")
    )

    assert naming_files == []
