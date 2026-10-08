"""Source-citation validators shared by the country spec package tests.

A line number names text only at a fixed commit. A country spec resource
therefore cites this repository's own sources by symbol (a dotted
``microcosm.*`` name, a function or class name, a ``# ---`` section header,
or a key or flag name), because their line numbers move under unrelated
edits. It cites another repository's source by line only when the citation
is pinned at a commit, in one of two ways:

- the same string names the commit inline: ``commit <sha>``, ``at <sha>``,
  ``@<sha>`` or a GitHub ``/blob/<sha>/`` permalink, where ``<sha>`` is 7-40
  lowercase hex digits with at least one digit and one letter, so a decimal
  amount ("at 2500000") or a word ("at defaced") pins nothing;
- the resource holds a mapping, at any depth, whose ``path`` names the file
  and whose ``commit`` is a sha of the same form (``source``-style pins,
  matched by file name across the whole resource).

An inline commit pins every citation in its string. A cited path is
in-repository when it resolves in this checkout, or when it has at least two
components and is the tail of a file here (``us_runtime/asec_pool.py``); a
line citation of it is refused even when pinned. A bare file name cannot be
told apart from another repository's, so it must be pinned.

The recognised spellings are the ones ``_LINE_CITATIONS`` documents. A symbol
between the file and its lines (``x.py some_function, lines 12-19``) is not
recognised.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from microcosm.build.spec_engine import load_yaml12
from test_support.paths import REPOSITORY_ROOT

#: Every country spec package: a build directory with a country_package.json.
COUNTRY_PACKAGE_ROOT = REPOSITORY_ROOT / "packages/microcosm-build/src/microcosm/build"

#: Text source kinds a line citation can name.
_SOURCE_SUFFIXES = (
    "py|pyi|ipynb|yaml|yml|jsonl|json|toml|cfg|ini|sh|md|rst|txt|csv|sql"
    "|rs|tsx|ts|mjs|js"
)
#: A source file named in prose: ``concepts.py``, or a path ending in one.
_SOURCE = (
    r"(?P<file>[A-Za-z0-9_./-]*[A-Za-z0-9_]\.(?:" + _SOURCE_SUFFIXES + r"))"
    r"(?![A-Za-z0-9_])"
)
#: A line locator into a source, in each spelling the packages have used or
#: GitHub produces: ``x.py:12``, ``x.py:12-19``, ``x.py:12,19``, ``x.py#L12``,
#: ``x.py L12``, ``x.py, L12``, ``x.py (L12)``, ``x.py line 12``,
#: ``x.py, lines 12-19``, ``x.py: lines 12-19``, ``x.py (lines 12-19)``, and
#: the reverse ``lines 12-19 of x.py`` and ``line 12 in x.py``. The file name
#: may be quoted or backticked (```x.py` lines 12-19``).
_LINE_CITATIONS = (
    re.compile(
        _SOURCE + r"[`'\"]?(?::|#L|,?\s*\(?\s*L|,?\s*:?\s*\(?\s*[Ll]ines?\s+)(?=\d)"
    ),
    re.compile(
        r"\b[Ll]ines?\s+\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*\s+(?:of|in)\s+[`'\"]?"
        + _SOURCE
    ),
)
#: A commit sha: 7-40 lowercase hex with at least one digit and one letter,
#: so a decimal amount ("2500000"), a word ("defaced") or a branch name such
#: as ``main`` is not one.
_SHA = r"(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}"
#: A commit pin's value in a ``path``/``commit`` mapping.
_COMMIT_PIN = re.compile(_SHA)
#: A commit named inline. The sha must not run on into a file name
#: (``at a1b2c3d.py:12`` names a file, not a commit).
INLINE_COMMIT = re.compile(
    r"(?:(?i:\bcommit\s+|\bat\s+)|@|/(?:blob|tree)/)"
    + _SHA
    + r"(?![0-9A-Za-z_]|\.[0-9A-Za-z_])"
)
#: Top-level directories whose files are this repository's sources.
_SOURCE_ROOTS = ("packages", "tools", "experiments", "test_support", "docs")
_SKIPPED_DIRECTORIES = {"__pycache__", ".venv", "node_modules", ".git"}
#: A dotted Microcosm symbol named in prose, such as
#: ``microcosm.graph.executor._structural_columns``.
DOTTED_MICROCOSM_SYMBOL = re.compile(
    r"(?<![A-Za-z0-9_./-])microcosm(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
)
#: The ``microcosm.<shard>`` namespaces in this checkout. A dotted name under
#: any other second component, such as the ``microcosm.axiom_input_closure.v1``
#: format id, names no Python symbol.
MICROCOSM_SHARDS = frozenset(
    path.name
    for path in REPOSITORY_ROOT.glob("packages/microcosm-*/src/microcosm/*")
    if path.is_dir() and path.name != "__pycache__"
)


class LineCitation(NamedTuple):
    """A line citation: the source as cited, and the string carrying it."""

    file: str
    text: str

    @property
    def name(self) -> str:
        return PurePosixPath(self.file).name


def nested_strings(value: object) -> Iterator[str]:
    """Every string in a payload, mapping keys included, at any depth."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_strings(child)
    elif isinstance(value, str):
        yield value


def line_citations(value: object) -> list[LineCitation]:
    """Each line citation of a source in a payload, in string order."""

    return [
        LineCitation(match["file"], text)
        for text in nested_strings(value)
        for pattern in _LINE_CITATIONS
        for match in pattern.finditer(text)
    ]


def names_a_commit(text: str) -> bool:
    """Whether a string names a commit inline."""

    return INLINE_COMMIT.search(text) is not None


def named_commits(text: str) -> list[str]:
    """The commits a string names inline, in order."""

    return [
        re.sub(r"^(?:(?i:commit|at)\s+|@|/(?:blob|tree)/)", "", match.group(0))
        for match in INLINE_COMMIT.finditer(text)
    ]


def commit_pinned_sources(value: object) -> set[str]:
    """File names of the sources a payload pins in ``path``/``commit`` maps."""

    pinned: set[str] = set()
    if isinstance(value, Mapping):
        path, commit = value.get("path"), value.get("commit")
        if (
            isinstance(path, str)
            and isinstance(commit, str)
            and _COMMIT_PIN.fullmatch(commit)
        ):
            pinned.add(PurePosixPath(path).name)
        for child in value.values():
            pinned |= commit_pinned_sources(child)
    elif isinstance(value, list):
        for child in value:
            pinned |= commit_pinned_sources(child)
    return pinned


def unpinned_line_citations(document: object) -> list[LineCitation]:
    """Line citations that neither their string nor the document pins."""

    pinned = commit_pinned_sources(document)
    return [
        citation
        for citation in line_citations(document)
        if citation.name not in pinned and not names_a_commit(citation.text)
    ]


@cache
def _repository_path_tails(root: Path) -> frozenset[str]:
    """Every tail of two or more components of a source file in ``root``."""

    tails: set[str] = set()
    for top in _SOURCE_ROOTS:
        for directory, subdirectories, files in os.walk(root / top):
            subdirectories[:] = [
                name for name in subdirectories if name not in _SKIPPED_DIRECTORIES
            ]
            parts = PurePosixPath(Path(directory).relative_to(root).as_posix()).parts
            for name in files:
                path = (*parts, name)
                tails.update("/".join(path[cut:]) for cut in range(len(path) - 1))
    return frozenset(tails)


def resolves_in_repository(file: str, root: Path = REPOSITORY_ROOT) -> bool:
    """Whether a cited path names a file in this checkout.

    A path of two or more components resolves when it is the tail of a source
    file here: ``tools/x.py``, ``packages/microcosm-build/src/...``,
    ``microcosm/graph/executor.py`` and ``us_runtime/asec_pool.py`` all do.
    """

    relative = PurePosixPath(file)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
        return False
    return relative.as_posix() in _repository_path_tails(root)


def in_repository_line_citations(
    document: object, root: Path = REPOSITORY_ROOT
) -> list[LineCitation]:
    """Line citations of this repository's files, pinned or not."""

    return [
        citation
        for citation in line_citations(document)
        if resolves_in_repository(citation.file, root)
    ]


def dotted_microcosm_symbols(value: object) -> set[str]:
    """Every dotted ``microcosm.<shard>.*`` name in a payload's strings."""

    return {
        match
        for text in nested_strings(value)
        for match in DOTTED_MICROCOSM_SYMBOL.findall(text)
        if match.split(".")[1] in MICROCOSM_SHARDS
    }


def resolve_dotted_symbol(dotted: str) -> object:
    """Import the longest module prefix of a dotted name, then get the rest."""

    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        name = ".".join(parts[:cut])
        try:
            target: object = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            continue
        for attribute in parts[cut:]:
            target = getattr(target, attribute)
        return target
    raise ModuleNotFoundError(dotted, name=dotted)


def country_packages() -> list[str]:
    """Each country spec package directory name, sorted."""

    return sorted(
        path.parent.name for path in COUNTRY_PACKAGE_ROOT.glob("*/country_package.json")
    )


def package_payloads(country: str) -> dict[str, object]:
    """Every resource in a country package, parsed, by relative path.

    JSON and YAML resources are parsed; any other file is one string.
    """

    root = COUNTRY_PACKAGE_ROOT / country
    payloads: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        name = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            payloads[name] = json.loads(text)
        elif path.suffix in {".yaml", ".yml"}:
            payloads[name] = load_yaml12(text, source=name)
        else:
            payloads[name] = text
    return payloads
