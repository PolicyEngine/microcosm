"""Source-citation validators shared by the country spec package tests.

A line number names text only at a fixed commit. A country spec resource
therefore cites this repository's own sources by symbol (a dotted
``microcosm.*`` name, a function or class name, a ``# ---`` section header,
or a key or flag name), because their line numbers move under unrelated
edits. It cites another repository's source by line only when the citation
is pinned at a commit, in one of two ways:

- the string carrying the citation names the commit inline: ``commit <sha>``,
  ``at <sha>``, ``@<sha>``, a GitHub ``/blob/``, ``/tree/``, ``/raw/``,
  ``/blame/``, ``/commit/`` or ``/commits/`` URL segment, or a
  ``raw.githubusercontent.com/<owner>/<repo>/<sha>/`` URL, where ``<sha>`` is
  7-40 lowercase hex digits with at least one digit and one letter, so a
  decimal amount ("at 2500000"), a word ("at defaced") or a truncated hash
  ("at e2de2b41…") pins nothing;
- the resource holds a mapping, at any depth, whose ``path`` ends with the
  cited path and whose ``commit`` is a sha of the same form (``source``-style
  pins): a pin of ``upstream_data/datasets/cps/cps.py`` covers
  ``cps.py`` and ``datasets/cps/cps.py``, not ``other/cps.py``.

Each citation binds to one inline commit (``citing_commit``): the nearest one
in the citation's own clause, else the nearest one before it in the string,
unless the citation's clause names a moving ref (``on main``, ``@HEAD``,
``/blob/master/``). A clause ends at a ``;`` or a sentence end. So
"obr.py:302 on main; compare us-data at 42ed5d4" and "us-data at 42ed5d4;
obr.py:302 on main" pin nothing, and in "x.py:12 at <a>; y.py:30 at <b>" each
file binds its own commit. A commit-like token in no recognised
spelling, in a citation's clause or between the citation and the commit it
binds to ("archived 42ed5d45 cps.py L197"), is refused as a misspelled pin
(``misspelled_commit_pins``), so a stray sha cannot sit beside a citation
that some other commit pins.

A cited path is in-repository when, after any prefix naming this repository
is removed (a ``github.com/PolicyEngine/microcosm/blob/<ref>/`` URL, or
``microcosm/`` before a top-level source directory), it is a file here or one
of the path forms ``_repository_path_tails`` lists. A line citation of it is
refused even when pinned. A bare file name cannot be told apart from another
repository's, so it must be pinned.

The recognised spellings are the ones ``_LINE_CITATIONS`` documents. A source
name followed by a line locator in no recognised spelling, with no ``.``,
``;`` or line break between (``x.py some_function, lines 12-19``), is refused
too (``unrecognised_line_locators``), so a new spelling fails loudly instead
of passing unscanned. A dotted name with lines
(``build_loss_matrix.create_target_matrix:93-103``) is out of scope: that
spelling is also how ledger fact keys such as
``ledger.aggregate_fact.v2:5973fe86…`` read.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path, PurePosixPath

from microcosm.build.spec_engine import load_yaml12
from test_support.paths import REPOSITORY_ROOT

#: Every country spec package: a build directory with a country_package.json.
COUNTRY_PACKAGE_ROOT = REPOSITORY_ROOT / "packages/microcosm-build/src/microcosm/build"

#: Text source kinds a line citation can name.
_SOURCE_SUFFIXES = (
    "py|pyi|ipynb|yaml|yml|jsonl|json|toml|cfg|ini|sh|md|rst|txt|csv|sql"
    "|rs|tsx|ts|mjs|js"
)
#: A source file named in prose: ``concepts.py``, a path ending in one, or a
#: URL of one.
_SOURCE = (
    r"(?P<file>(?:https?:)?[A-Za-z0-9_./-]*[A-Za-z0-9_]\.(?:"
    + _SOURCE_SUFFIXES
    + r"))(?![A-Za-z0-9_])"
)
#: Quotes or backticks around a file name or a locator.
_QUOTE = r"[`'\"‘’“”]*"
#: A line number, plain or GitHub-style (``L12``).
_NUMBER = r"L?\s*\d+(?!\w)"
#: One line or a range of them: ``12-19``, ``L12-L19``, ``12–19``,
#: ``12 to 19``, ``12 through 19``.
_SPAN = rf"{_NUMBER}(?:\s*(?:[-–—]|(?i:to|through))\s*{_NUMBER})?"
#: Several lines or ranges: ``12-19, 30`` or ``12-19 and 30``.
_NUMBERS = rf"{_SPAN}(?:\s*(?:,|,?\s*(?i:and))\s*{_SPAN})*"
#: A line label: ``line``, ``lines`` (any case), ``l.`` or ``ll.``.
_LABEL = r"(?i:lines?\b\s+|ll?\.\s*)"
#: A line locator into a source, in each spelling the packages have used or
#: GitHub produces, followed by its line numbers:
#:
#: - forward: ``x.py:12``, ``x.py:L12``, ``x.py#L12``, ``x.py L12``,
#:   ``x.py line 12``, ``x.py: lines 12``, ``x.py @ L12``, and a separator
#:   (``,``, ``;``, ``—``, ``–`` or a spaced ``-``), a bracket and ``at`` or
#:   ``on`` before a label or an ``L`` number: ``x.py, lines 12-19``,
#:   ``x.py (lines 12-19)``, ``x.py [L12]``, ``x.py at line 12``,
#:   ``x.py — ll. 12-19``;
#: - reverse: ``lines 12-19 of x.py``, ``line 12 in x.py``, ``L12 of x.py``.
#:
#: The numbers are any ``_NUMBERS``; the file name and the locator may be
#: quoted or backticked (```x.py` lines 12-19``).
_LINE_CITATIONS = (
    re.compile(
        _SOURCE
        + _QUOTE
        + r"(?::(?=L?\d)"
        + rf"|\s*[:：]\s*{_QUOTE}{_LABEL}"
        + rf"|\s*@\s*{_QUOTE}(?:{_LABEL}|(?=L\d))"
        + r"|#(?=L\d)"
        + r"|\s*(?:[,;—–]|\s-(?=\s))?\s*[\[(]?\s*(?:(?i:at|on)\s+)?"
        + rf"{_QUOTE}(?:{_LABEL}|(?=L\d)))"
        + _QUOTE
        + _NUMBERS
    ),
    re.compile(
        rf"\b(?:{_LABEL}|(?=L\d)){_QUOTE}{_NUMBERS}{_QUOTE}\s+(?i:of|in)\s+{_QUOTE}"
        + _SOURCE
    ),
)
#: A source name followed by a labelled line locator (``L12``, ``line 12``,
#: ``ll. 12``) within 24 characters, with no ``.``, ``;`` or line break
#: between: the shape of every forward spelling, recognised or not. A bare
#: number after a colon is left out: ``x.json: 63 rows`` is a count.
_LOCATOR_NEAR_A_SOURCE = re.compile(
    _SOURCE
    + _QUOTE
    + r"[^.;\n]{0,24}?(?<![A-Za-z0-9])(?=L\d|(?i:lines?)\s+\d|(?i:ll?)\.\s*\d)"
)
#: A commit sha: 7-40 lowercase hex with at least one digit and one letter,
#: so a decimal amount ("2500000"), a word ("defaced") or a branch name such
#: as ``main`` is not one.
_SHA = r"(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}"
#: A commit pin's value in a ``path``/``commit`` mapping.
_COMMIT_PIN = re.compile(_SHA)
#: A commit named inline. The sha must not run on into a file name
#: (``at a1b2c3d.py:12`` names a file), a longer token (``a1b2c3d-x``) or an
#: ellipsis (``e2de2b41…`` and ``e2de2b41...`` are truncated hashes).
INLINE_COMMIT = re.compile(
    r"(?:(?i:\bcommit\s+|\bat\s+)|@|/(?:blob|tree|raw|blame|commits?)/"
    r"|githubusercontent\.com/[^/\s]+/[^/\s]+/)"
    + r"(?P<sha>"
    + _SHA
    + r")(?![0-9A-Za-z_…-]|\.[0-9A-Za-z_.])"
)
#: A commit-like token in any spelling: not part of a path, a URL, an
#: identifier, a longer token or a truncated hash.
_COMMIT_LIKE = re.compile(
    r"(?<![0-9A-Za-z_/@.…-])" + _SHA + r"(?![0-9A-Za-z_/…-]|\.[0-9A-Za-z_.])"
)
#: A moving ref named in prose: ``on main``, ``at HEAD``, ``@master`` or a
#: ``/blob/main/`` URL segment.
_MOVING_REF = re.compile(
    r"\b(?i:on|at)\s+(?:main|master|HEAD)\b|@(?:main|master|HEAD)\b"
    r"|/(?:blob|tree|raw|blame)/(?:main|master|HEAD)/"
)
#: A clause break: a semicolon, or the space after a sentence end before a
#: capital, a bracket or a backtick.
_CLAUSE_BREAK = re.compile(r";|(?<=[.!?])\s+(?=[A-Z(\[`])")
#: Top-level directories whose files are this repository's sources.
_SOURCE_ROOTS = ("packages", "tools", "experiments", "test_support", "docs")
_SKIPPED_DIRECTORIES = {"__pycache__", ".venv", "node_modules", ".git"}
#: A prefix naming this repository before a path in it: a GitHub URL of a
#: file here at any ref, or ``microcosm/`` (after any ``./`` or ``../`` and
#: an optional ``PolicyEngine/``) before a top-level source directory.
_THIS_REPOSITORY = re.compile(
    r"(?:https?:)?/*(?:www\.)?github\.com/(?i:policyengine/microcosm)/"
    r"(?:blob|tree|raw|blame)/[^/]+/"
    r"|(?:https?:)?/*raw\.githubusercontent\.com/(?i:policyengine/microcosm)/[^/]+/"
    r"|(?:\.\.?/)*(?:(?i:policyengine)/)?microcosm/(?=(?:"
    + "|".join(_SOURCE_ROOTS)
    + r")/)"
)
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


@dataclass(frozen=True)
class LineCitation:
    """A line citation: the source as cited, and the string carrying it.

    ``start`` and ``end`` locate the citation in ``text`` (-1 when built by
    hand); they do not take part in equality.
    """

    file: str
    text: str
    start: int = field(default=-1, compare=False)
    end: int = field(default=-1, compare=False)

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


def _citation_matches(text: str) -> list[re.Match[str]]:
    return sorted(
        (match for pattern in _LINE_CITATIONS for match in pattern.finditer(text)),
        key=lambda match: match.start(),
    )


def line_citations(value: object) -> list[LineCitation]:
    """Each line citation of a source in a payload, in string order."""

    return [
        LineCitation(match["file"], text, match.start(), match.end())
        for text in nested_strings(value)
        for match in _citation_matches(text)
    ]


def unrecognised_line_locators(value: object) -> list[LineCitation]:
    """Source names followed by a line locator in no recognised spelling."""

    found: list[LineCitation] = []
    for text in nested_strings(value):
        matches = _citation_matches(text)
        files = {match.start("file") for match in matches}
        spans = [(match.start(), match.end()) for match in matches]
        for near in _LOCATOR_NEAR_A_SOURCE.finditer(text):
            if near.start("file") in files or any(
                start <= near.end() < end for start, end in spans
            ):
                continue
            found.append(LineCitation(near["file"], text, near.start(), near.end()))
    return found


def names_a_commit(text: str) -> bool:
    """Whether a string names a commit inline."""

    return INLINE_COMMIT.search(text) is not None


def named_commits(text: str) -> list[str]:
    """The commits a string names inline, in order."""

    return [match["sha"] for match in INLINE_COMMIT.finditer(text)]


def _clause(citation: LineCitation) -> tuple[int, int]:
    """The bounds of the clauses of a citation's string that it touches.

    A break inside the citation (``x.py; lines 12-19``) does not split it.
    """

    if citation.start < 0:
        raise ValueError("only a citation from line_citations carries its position")
    first, last = 0, len(citation.text)
    for clause_break in _CLAUSE_BREAK.finditer(citation.text):
        if clause_break.end() <= citation.start:
            first = clause_break.end()
        elif clause_break.start() >= citation.end:
            last = clause_break.start()
            break
    return first, last


def _binding(citation: LineCitation) -> re.Match[str] | None:
    """The inline commit match a line citation binds to, if any."""

    commits = list(INLINE_COMMIT.finditer(citation.text))
    start, end = _clause(citation)
    in_clause = [match for match in commits if start <= match.start() < end]
    if in_clause:
        return min(
            in_clause,
            key=lambda match: max(
                match.start() - citation.end, citation.start - match.end()
            ),
        )
    if _MOVING_REF.search(citation.text, start, end):
        return None
    before = [match for match in commits if match.start() < start]
    return before[-1] if before else None


def citing_commit(citation: LineCitation) -> str | None:
    """The inline commit a line citation binds to, if any.

    That is the nearest commit in the citation's own clause, else the nearest
    one before the citation in its string, unless the citation's clause names
    a moving ref.
    """

    binding = _binding(citation)
    return None if binding is None else binding["sha"]


def misspelled_commit_pins(value: object) -> list[tuple[LineCitation, str]]:
    """Commit-like tokens in no pin spelling beside a line citation.

    A token counts when it sits in the citation's own clause, or between the
    citation and an earlier commit the citation binds to.
    """

    found: list[tuple[LineCitation, str]] = []
    for citation in line_citations(value):
        text = citation.text
        pinned = {match.start("sha") for match in INLINE_COMMIT.finditer(text)}
        start, end = _clause(citation)
        binding = _binding(citation)
        if binding is not None and binding.end() < start:
            start = binding.end()
        found.extend(
            (citation, token.group(0))
            for token in _COMMIT_LIKE.finditer(text, start, end)
            if token.start() not in pinned
        )
    return found


def commit_pinned_sources(value: object) -> set[str]:
    """Paths of the sources a payload pins in ``path``/``commit`` mappings."""

    pinned: set[str] = set()
    if isinstance(value, Mapping):
        path, commit = value.get("path"), value.get("commit")
        if (
            isinstance(path, str)
            and isinstance(commit, str)
            and _COMMIT_PIN.fullmatch(commit)
        ):
            pinned.add(PurePosixPath(path).as_posix())
        for child in value.values():
            pinned |= commit_pinned_sources(child)
    elif isinstance(value, list):
        for child in value:
            pinned |= commit_pinned_sources(child)
    return pinned


def pin_covers(cited: str, pinned: str) -> bool:
    """Whether a ``path`` pin covers a cited path: the cited components end it."""

    cited_parts = PurePosixPath(cited).parts
    pinned_parts = PurePosixPath(pinned).parts
    return (
        0 < len(cited_parts) <= len(pinned_parts)
        and pinned_parts[-len(cited_parts) :] == cited_parts
    )


def unpinned_line_citations(document: object) -> list[LineCitation]:
    """Line citations that neither an inline commit nor a path pin covers."""

    pinned = commit_pinned_sources(document)
    return [
        citation
        for citation in line_citations(document)
        if citing_commit(citation) is None
        and not any(pin_covers(citation.file, pin) for pin in pinned)
    ]


@cache
def _repository_path_tails(root: Path) -> frozenset[str]:
    """The path forms that name a source file in ``root``.

    A file is named by its path from the root (``tools/x.py``) and, under
    ``packages``, by its path from its distribution directory
    (``microcosm-build/src/...``). A file in a package's ``src/microcosm/``
    is also named by its path from ``src/``, from the ``microcosm`` namespace
    (``microcosm/graph/executor.py``), from its shard (``graph/executor.py``)
    and from a first-level subpackage of its shard
    (``us_runtime/asec_pool.py``), never by a deeper tail, so an unrelated
    ``utils/loss.py`` stays another repository's.
    """

    tails: set[str] = set()
    for top in _SOURCE_ROOTS:
        for directory, subdirectories, files in os.walk(root / top):
            subdirectories[:] = [
                name for name in subdirectories if name not in _SKIPPED_DIRECTORIES
            ]
            parts = PurePosixPath(Path(directory).relative_to(root).as_posix()).parts
            for name in files:
                path = (*parts, name)
                cuts = {0}
                if path[0] == "packages":
                    cuts.add(1)
                    if path[2:4] == ("src", "microcosm"):
                        cuts.update(range(2, min(6, len(path) - 1)))
                tails.update(
                    "/".join(path[cut:]) for cut in cuts if cut < len(path) - 1
                )
    return frozenset(tails)


def resolves_in_repository(file: str, root: Path = REPOSITORY_ROOT) -> bool:
    """Whether a cited path names a file in this checkout.

    A prefix naming this repository is removed first; what is left resolves
    when it is a file here (after such a prefix) or one of the path forms
    ``_repository_path_tails`` lists: ``tools/x.py``,
    ``microcosm/tools/x.py``, ``packages/microcosm-build/src/...``,
    ``microcosm/graph/executor.py`` and ``us_runtime/asec_pool.py`` all do.
    """

    named = _THIS_REPOSITORY.match(file)
    relative = PurePosixPath(file[named.end() :] if named else file)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    if named is not None and (root / relative).is_file():
        return True
    return len(relative.parts) >= 2 and relative.as_posix() in _repository_path_tails(
        root
    )


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
