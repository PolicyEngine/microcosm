"""Country spec packages cite sources by symbol or at a commit, never drifting.

A line number names text only at a fixed commit, so a citation such as
``asec_pool.py lines 61-75`` silently points at other code after an unrelated
edit. The validators live in ``test_support.microcosm_build.
spec_source_citations``; these tests state the invariants over every country
spec package (every build directory with a ``country_package.json``) and check
each refusal branch with Hypothesis, so a vacuous pass on today's data cannot
hide a broken check:

- no resource cites a line of a file in this repository, pinned or not
  (``in_repository_line_citations``). A cited path is this repository's when,
  after any prefix naming this repository (a ``PolicyEngine/microcosm``
  GitHub URL, or ``microcosm/`` before a top-level source directory), it is a
  file here or an anchored path form of one: its path from the root, from its
  distribution directory, from ``src/``, from the ``microcosm`` namespace,
  from its shard, or from a first-level subpackage of its shard, the run of
  path characters before a cited name included (``period=2025-01.json``).
  A bare file name cannot be resolved and falls to the next rule;
- no resource cites a line in a string that holds a dot segment (``..`` as
  a path segment beside ``/``, ``\\`` or ``%2f``, either dot perhaps
  ``%2e``), pinned or not (``dot_segment_line_citations``): whitespace, a
  backslash or markup in the segment it removes can make a renderer remove
  another one than a reading does, so which file the citation names is
  unknown;
- no resource links a line of this repository by its GitHub path, however a
  reading takes the link (``line_links_of_this_repository``): no string's
  source holds ``PolicyEngine/microcosm/``, read as GitHub reads a path,
  with ``#L<n>`` at the first ``#`` after it;
- no resource cites a line of any other source unless the citation binds to
  an inline commit (the nearest one in its own clause, else the nearest one
  before it in its string; ``citing_commit``), or the resource pins a path
  that the cited path ends in, in a ``path``/``commit`` mapping
  (``unpinned_line_citations``);
- no commit-like token in an unrecognised spelling sits in a citation's
  clause or between the citation and the commit it binds to
  (``misspelled_commit_pins``);
- no source name is followed by a line locator in an unrecognised spelling
  (``unrecognised_line_locators``);
- every dotted ``microcosm.*`` name in a resource resolves through import
  plus getattr (``resolve_dotted_symbol``).

Each string is read with its URLs as a browser reads them and, where that
differs, as Markdown delimits them (each URL also ending at a quote, a
parenthesis or an angle bracket); every rule applies to both readings.

Differential checks compare two records of one fact: the commit each UK
citation binds to with the incumbent commit its resource declares, and the US
known gaps' symbol citations with the code they name (the Census person-column
restore list, the relationship fallback, and the PUF support tool's
functions).

New Zealand's binding guard is the stricter grammar in
``test_nz_spec_package.py`` (#1282); this guard scans the NZ package too.
If an NZ resource starts failing here, ask the NZ hub before exempting it.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import string
import time
from collections.abc import Callable
from html.entities import html5
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest
from hypothesis import assume, event, example, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import asec_pool
from microcosm.build.us_runtime.asec_census_person_columns import (
    ASEC_CENSUS_PERSON_COLUMN_NAMES,
    ASEC_CENSUS_PERSON_COLUMNS,
)
from test_support.microcosm_build import spec_source_citations as citations
from test_support.microcosm_build.spec_source_citations import (
    COUNTRY_PACKAGE_ROOT,
    MICROCOSM_SHARDS,
    REPOSITORY_ROOT,
    LineCitation,
    citing_commit,
    commit_pinned_sources,
    country_packages,
    dot_segment_line_citations,
    dotted_microcosm_symbols,
    in_repository_line_citations,
    line_citations,
    line_links_of_this_repository,
    misspelled_commit_pins,
    named_commits,
    names_a_commit,
    package_payloads,
    pin_covers,
    resolve_dotted_symbol,
    resolves_in_repository,
    unpinned_line_citations,
    unrecognised_line_locators,
)

PACKAGES = ("am", "be", "nz", "uk", "us")


# ---------------------------------------------------------------------------
# Committed resources
# ---------------------------------------------------------------------------


def test_every_country_package_is_scanned() -> None:
    assert tuple(country_packages()) == PACKAGES
    for country in PACKAGES:
        assert package_payloads(country)


@pytest.mark.parametrize("country", PACKAGES)
def test_no_resource_cites_a_line_it_does_not_pin(country: str) -> None:
    offenders = {
        name: hits
        for name, payload in package_payloads(country).items()
        if (hits := unpinned_line_citations(payload))
    }
    assert offenders == {}


@pytest.mark.parametrize("country", PACKAGES)
def test_no_resource_cites_a_line_of_this_repository(country: str) -> None:
    offenders = {
        name: hits
        for name, payload in package_payloads(country).items()
        if (hits := in_repository_line_citations(payload))
    }
    assert offenders == {}


@pytest.mark.parametrize("country", PACKAGES)
def test_no_resource_cites_a_line_in_a_string_with_a_dot_segment(country: str) -> None:
    offenders = {
        name: hits
        for name, payload in package_payloads(country).items()
        if (hits := dot_segment_line_citations(payload))
    }
    assert offenders == {}


@pytest.mark.parametrize("country", PACKAGES)
def test_no_resource_links_a_line_of_this_repository(country: str) -> None:
    offenders = {
        name: hits
        for name, payload in package_payloads(country).items()
        if (hits := line_links_of_this_repository(payload))
    }
    assert offenders == {}


@pytest.mark.parametrize("country", PACKAGES)
def test_no_resource_misspells_a_commit_pin(country: str) -> None:
    offenders = {
        name: hits
        for name, payload in package_payloads(country).items()
        if (hits := misspelled_commit_pins(payload))
    }
    assert offenders == {}


@pytest.mark.parametrize("country", PACKAGES)
def test_no_resource_cites_a_line_in_an_unrecognised_spelling(country: str) -> None:
    offenders = {
        name: hits
        for name, payload in package_payloads(country).items()
        if (hits := unrecognised_line_locators(payload))
    }
    assert offenders == {}


@pytest.mark.parametrize(
    ("country", "resource", "cited"),
    [
        ("nz", "as_rate_bridge.json", {"run.py"}),
        ("uk", "uk_data_target_parity.json", {"obr.py", "voa_council_tax.py"}),
        ("uk", "uk_population_targets.json", {"income.py", "hmrc_salary_sacrifice.py"}),
        (
            "us",
            "source_stages.json",
            {"cps.py", "puf.py", "disaggregate_puf.py", "sipp.py"},
        ),
        (
            "us",
            "spec/sources.yaml",
            {"cps.py", "puf.py", "disaggregate_puf.py", "sipp.py"},
        ),
        ("us", "ecps_parity_known_gaps.json", {"cps.py", "census_cps.py"}),
        ("us", "target_parity_manifest.json", {"loss.py", "cps.py"}),
    ],
)
def test_the_pinned_line_citations_are_scanned(
    country: str, resource: str, cited: set[str]
) -> None:
    payload = package_payloads(country)[resource]
    assert cited <= {citation.name for citation in line_citations(payload)}


@pytest.mark.parametrize("country", PACKAGES)
def test_every_dotted_microcosm_symbol_resolves(country: str) -> None:
    named = set().union(
        *map(dotted_microcosm_symbols, package_payloads(country).values())
    )
    unresolved = []
    for dotted in sorted(named):
        try:
            resolve_dotted_symbol(dotted)
        except (ImportError, AttributeError) as error:
            unresolved.append(f"{dotted}: {error}")
    assert unresolved == []


@pytest.mark.parametrize(
    ("country", "cited"),
    [
        (
            "nz",
            {
                "microcosm.graph.executor._structural_columns",
                "microcosm.graph.executor._validate_population_declaration",
            },
        ),
        (
            "us",
            {
                "microcosm.build.us_runtime.asec_pool.load_asec_h5_tables",
                "microcosm.build.us_runtime.asec_pool._prepare_year_input",
                "microcosm.build.us_runtime.asec_pool._with_relationship_recode",
            },
        ),
    ],
)
def test_the_dotted_symbol_citations_are_scanned(country: str, cited: set[str]) -> None:
    named = set().union(
        *map(dotted_microcosm_symbols, package_payloads(country).values())
    )
    assert cited <= named


def test_only_names_under_a_shard_are_symbol_citations() -> None:
    assert {"build", "frame", "graph"} <= MICROCOSM_SHARDS
    text = (
        "format microcosm.axiom_input_closure.v1, read by "
        "microcosm.frame.concepts.concept and microcosm.build.us/source_stages.json"
    )
    assert dotted_microcosm_symbols({"note": text}) == {
        "microcosm.frame.concepts.concept",
        "microcosm.build.us",
    }


@pytest.mark.parametrize(
    ("dotted", "error"),
    [
        ("microcosm.graph.executor._no_such_symbol", AttributeError),
        ("microcosm.no_such_shard.module.symbol", ModuleNotFoundError),
    ],
)
def test_an_unresolvable_dotted_symbol_raises(
    dotted: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        resolve_dotted_symbol(dotted)


# ---------------------------------------------------------------------------
# Validator properties
# ---------------------------------------------------------------------------


_STEM = st.from_regex(r"[a-z_][a-z0-9_]{0,15}", fullmatch=True)
_SUFFIX = st.sampled_from(("py", "yaml", "json", "sh", "csv", "toml"))
#: Directories that do not exist in this checkout, so a path under one is
#: another repository's (an empty directory leaves a bare file name).
_FOREIGN_DIRECTORY = st.sampled_from(
    (
        "",
        "archived_data/datasets/cps/",
        "datasets/cps/",
        "targets/sources/",
        "nz-lane/emtr_reproduction/",
    )
)
#: Leading directories a pin's path may carry beyond the cited path.
_PIN_ROOT = st.sampled_from(("", "upstream_data/", "src/lane/", "a/b/c/"))
#: Line-locator spellings: {file} is the cited source, {a} and {b} lines.
_LINE_LOCATORS = (
    "{file}:{a}",
    "{file}:{a}-{b}",
    "{file}:{a},{b}",
    "{file}:L{a}",
    "{file}#L{a}",
    "{file}#L{a}-L{b}",
    "{file} L{a}",
    "{file} L{a}-L{b}",
    "{file} [L{a}]",
    "{file} @ L{a}",
    "{file} line {a}",
    "{file} lines {a}-{b}",
    "{file} lines {a}–{b}",
    "{file} lines {a} to {b}",
    "{file} lines {a} through {b}",
    "{file} lines {a}-{b} and {a}",
    "{file}, lines {a}-{b}",
    "{file}; lines {a}-{b}",
    "{file} — lines {a}-{b}",
    "{file} - lines {a}-{b}",
    "{file} (lines {a}-{b})",
    "{file} at line {a}",
    "{file} on lines {a}-{b}",
    "{file} ll. {a}-{b}",
    "{file} (l. {a})",
    "{file}: lines {a}-{b}",
    "{file}: line {a}",
    "{file}, L{a}",
    "{file} (L{a})",
    "`{file}` lines {a}-{b}",
    "“{file}” lines {a}-{b}",
    "Lines {a}-{b} of {file}",
    "lines {a} to {b} of {file}",
    "lines {a}-{b} and {a} of {file}",
    "ll. {a}-{b} of {file}",
    "L{a} of {file}",
    "line {a} in {file}",
    "line {a} of `{file}`",
)
#: The spellings that name the file before its lines.
_FORWARD_LOCATORS = tuple(
    locator
    for locator in _LINE_LOCATORS
    if locator.index("{file}") < locator.index("{a}")
)
#: Symbol-citation spellings, which carry no line number.
_SYMBOL_LOCATORS = (
    "{file} {symbol}",
    '{file} _pointer("{symbol}")',
    '{file}, the "# --- Relationships" block',
    "{file}: {symbol}",
    "{symbol} in {file} at main",
)
#: Line locators in no recognised spelling: a symbol sits between the file
#: and its lines.
_UNRECOGNISED_LOCATORS = (
    "{file} {symbol}, lines {a}-{b}",
    "{file} {symbol} L{a}",
    "{file} {symbol} (line {a})",
    "{file}, {symbol} ll. {a}-{b}",
)
#: Symbols that would complete a recognised spelling in those locators.
_LOCATOR_WORDS = {"at", "on", "l", "ll", "line", "lines"}
#: Inline-commit spellings.
_INLINE_PINS = (
    "commit {sha}",
    "Commit {sha}",
    "at {sha}",
    "At {sha}",
    "uk-data@{sha}",
    "https://github.com/PolicyEngine/archived-data/blob/{sha}/x",
    "https://github.com/PolicyEngine/archived-data/tree/{sha}/x",
    "https://github.com/PolicyEngine/archived-data/raw/{sha}/x",
    "https://github.com/PolicyEngine/archived-data/blame/{sha}/x",
    "https://github.com/PolicyEngine/archived-data/commit/{sha}",
    "https://github.com/PolicyEngine/archived-data/commits/{sha}",
    "https://raw.githubusercontent.com/PolicyEngine/archived-data/{sha}/x",
)
#: Prose around a citation. It has no digit, so it can neither extend a
#: planted line number nor name a commit of its own.
_PROSE = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,;()'\"-_",
    max_size=40,
)
#: Prose within one clause: no ``;`` and no sentence end.
_CLAUSE_PROSE = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,()'\"-_",
    max_size=40,
)
#: Moving refs a clause can name.
_MOVING_REFS = (
    "on main",
    "On main",
    "at HEAD",
    "@master",
    "on `main`",
    "at 'HEAD'",
    "@“master”",
    "on refs/heads/main",
    "https://github.com/PolicyEngine/upstream-data/blob/main/x",
    "https://github.com/PolicyEngine/upstream-data/blob/refs/heads/master/x",
    "https://raw.githubusercontent.com/PolicyEngine/upstream-data/main/x",
    "https://raw.githubusercontent.com/PolicyEngine/upstream-data/refs/heads/main/x",
)
#: URL prefixes of another repository's files at a moving ref.
_MOVING_URLS = (
    "https://github.com/PolicyEngine/upstream-data/blob/main/",
    "https://github.com/PolicyEngine/upstream-data/blob/refs/heads/master/",
    "https://raw.githubusercontent.com/PolicyEngine/upstream-data/main/",
    "https://raw.githubusercontent.com/PolicyEngine/upstream-data/HEAD/",
    "https://raw.githubusercontent.com/PolicyEngine/upstream-data/refs/heads/main/",
)
#: Quote marks a file name or a locator may carry.
_QUOTE_MARKS = st.sampled_from(("", "`", "'", '"', "‘", "’", "“", "”"))
#: Locators with quote slots before and after the file ({q1}, {q2}) and
#: before and after its line numbers ({q3}, {q4}).
_QUOTABLE_LOCATORS = (
    "{q1}{file}{q2}:{q3}{a}-{b}{q4}",
    "{q1}{file}{q2}:{q3}L{a}{q4}",
    "{q1}{file}{q2}#{q3}L{a}{q4}",
    "{q1}{file}{q2} {q3}lines {a}-{b}{q4}",
    "{q1}{file}{q2}, {q3}lines {a}-{b}{q4}",
    "{q1}{file}{q2} @ {q3}L{a}{q4}",
    "{q1}{file}{q2}: {q3}lines {a}-{b}{q4}",
    "{q3}lines {a}-{b}{q4} of {q1}{file}{q2}",
)
#: Clause breaks, each followed by text that opens the next clause.
_CLAUSE_BREAKS = ("; ", ";", ". See ", "! See ", "? (", ". `")
_HEX = st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)
_BRANCH_NAMES = ("main", "HEAD", "launch/ga-path-execution-gate", "ba6e7d", "")
#: Files in this checkout, in each path form a citation can use.
_IN_REPOSITORY_FILES = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py",
    "microcosm-build/src/microcosm/build/us_runtime/asec_pool.py",
    "src/microcosm/build/us_runtime/asec_pool.py",
    "build/us_runtime/asec_pool.py",
    "us_runtime/asec_pool.py",
    "graph/executor.py",
    "microcosm-frame/src/microcosm/frame/concepts.py",
    "microcosm/graph/executor.py",
    "tools/build_us_puf_support_base.py",
    "microcosm/tools/build_us_puf_support_base.py",
    "../microcosm/tools/build_us_puf_support_base.py",
    "PolicyEngine/microcosm/experiments/build_j_recert/buildj_base.sh",
    "experiments/build_j_recert/base_j.summary.json",
    "https://github.com/PolicyEngine/microcosm/blob/main/tools/build_us_puf_support_base.py",
    "https://github.com/PolicyEngine/microcosm/blob/0a1b2c3d4e/packages/microcosm-build/"
    "src/microcosm/build/us_runtime/asec_pool.py",
    "https://raw.githubusercontent.com/PolicyEngine/microcosm/main/tools/"
    "build_us_puf_support_base.py",
    "https://raw.githubusercontent.com/PolicyEngine/microcosm/refs/heads/main/tools/"
    "build_us_puf_support_base.py",
    "https://github.com/PolicyEngine/microcosm/blob/refs/heads/main/tools/"
    "build_us_puf_support_base.py",
    "https://github.com/PolicyEngine/microcosm/tree/feature/citations/packages/"
    "microcosm-build/src/microcosm/build/us_runtime/asec_pool.py",
)
#: URL prefixes of this repository, up to the ref.
_THIS_REPOSITORY_URLS = (
    "https://github.com/PolicyEngine/microcosm/blob/",
    "https://github.com/policyengine/microcosm/tree/",
    "github.com/PolicyEngine/microcosm/raw/",
    "https://www.github.com/PolicyEngine/microcosm/blame/",
    "https://raw.githubusercontent.com/PolicyEngine/microcosm/",
)
#: Files in this checkout, from the repository root.
_ROOTED_FILES = (
    "tools/build_us_puf_support_base.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py",
    "experiments/build_j_recert/buildj_base.sh",
)
#: One segment of a git ref.
_REF_SEGMENT = st.from_regex(r"[A-Za-z0-9_-]{1,12}", fullmatch=True)
#: A URL up to its ref: the host, and the owner, repository and route (its
#: scheme is dropped; ``_URL_LEADS`` writes one).
_URL_PARTS = re.compile(r"(?:https?://)?(?P<host>[^/]+)(?P<route>/.*)")
#: Ways to write the start of a URL, each with the start it is read as.
_URL_LEADS = (
    ("https://", "https://"),
    ("http://", "http://"),
    ("https:", "https://"),
    ("https:/", "https://"),
    ("https:///", "https://"),
    ("https:\\\\", "https://"),
    ("http:\\/", "http://"),
    ("//", "//"),
    ("\\\\", "//"),
    ("/\\", "//"),
    ("///", "//"),
    ("", ""),
)
#: Authorities around a host: a user (up to the last ``@``) and a port.
_AUTHORITIES = (
    "{host}",
    "{host}:443",
    "{host}:",
    "{host}:0443",
    "user@{host}",
    "u:p@{host}:443",
    "a@b@{host}",
)
#: Detours in a URL path that GitHub resolves away, quotes and brackets in
#: them included.
_DOT_DETOURS = (
    "",
    "./",
    "%2e/",
    "docs/../",
    "x/%2E%2e/",
    "a/b/../../",
    "tools/./../",
    "'x'/../",
    '"/../',
    "<x>/../",
    "`x`/../",
)
#: A URL query, without its ``?``: anything but a fragment or a space, quotes
#: and brackets included.
_URL_QUERY = st.text(
    alphabet=string.ascii_letters
    + string.digits
    + "=&.;:,%+-_~?/'\"`‘’“”<>!$()*[]{}|^",
    max_size=120,
)


def _plant(text: str, depth: int, as_key: bool) -> object:
    planted: object = {text: "cited"} if as_key else text
    for level in range(depth):
        planted = {"level": level, "note": [planted]}
    return planted


def _escape(text: str, chosen: set[int], upper: bool, keep: str = "") -> str:
    """``text`` with its characters at ``chosen`` written as ``%`` escapes.

    A character in ``keep``, or in an escape already there, stays as written:
    escaping the ``2`` of ``%2E`` would name another path (``%%32E``).
    """

    escaped = {
        index
        for escape in re.finditer(r"%[0-9A-Fa-f]{2}", text)
        for index in range(escape.start(), escape.end())
    }
    return "".join(
        (f"%{ord(char):02X}" if upper else f"%{ord(char):02x}")
        if index in chosen and index not in escaped and char not in keep
        else char
        for index, char in enumerate(text)
    )


#: The full stops a host may be written with: full-width, ideographic and
#: half-width ideographic. The standard reads each as ``.`` (bun 1.3).
_FULL_STOPS = ("\uff0e", "\u3002", "\uff61")


def _spell_host(
    host: str, wide: set[int], chosen: set[int], upper: bool, stop: str
) -> str:
    """``host`` with its characters at ``wide`` written full-width (``ｇ``),
    a dot there as ``stop``, and those at ``chosen`` as ``%`` escapes of
    their UTF-8 bytes."""

    spelled = []
    for index, char in enumerate(host):
        if index in wide:
            char = stop if char == "." else chr(ord(char) + 0xFEE0)
        if index in chosen:
            char = "".join(
                f"%{byte:02X}" if upper else f"%{byte:02x}" for byte in char.encode()
            )
        spelled.append(char)
    return "".join(spelled)


class _CountingPath(type(REPOSITORY_ROOT)):  # type: ignore[misc]
    """A path that counts the ``is_file`` checks made through it."""

    checks = 0

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        type(self).checks += 1
        return super().is_file(follow_symlinks=follow_symlinks)


def _is_commit_like(sha: str) -> bool:
    return any(c.isdigit() for c in sha) and any(c.isalpha() for c in sha)


_SHA = _HEX.filter(_is_commit_like)


@settings(max_examples=300, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    locator=st.sampled_from(_LINE_LOCATORS),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    prefix=_PROSE,
    tail=_PROSE,
    depth=st.integers(0, 3),
    as_key=st.booleans(),
)
# The reverse spellings come last, so generation draws them least: a worded
# reverse range is found only if ``to`` reads as a range.
@example(
    stem="x",
    suffix="py",
    directory="",
    locator="lines {a} to {b} of {file}",
    a=12,
    b=19,
    prefix="",
    tail="",
    depth=0,
    as_key=False,
)
@example(
    stem="x",
    suffix="py",
    directory="datasets/cps/",
    locator="{file}#L{a}",
    a=12,
    b=12,
    prefix="",
    tail="",
    depth=2,
    as_key=True,
)
@example(
    stem="x",
    suffix="py",
    directory="",
    locator="ll. {a}-{b} of {file}",
    a=12,
    b=19,
    prefix="",
    tail="",
    depth=0,
    as_key=False,
)
def test_an_unpinned_line_citation_is_refused_at_any_depth(
    stem: str,
    suffix: str,
    directory: str,
    locator: str,
    a: int,
    b: int,
    prefix: str,
    tail: str,
    depth: int,
    as_key: bool,
) -> None:
    file = f"{directory}{stem}.{suffix}"
    text = f"{prefix} {locator.format(file=file, a=a, b=b)} {tail}"
    document = _plant(text, depth, as_key)
    assert LineCitation(file, text) in unpinned_line_citations(document)
    assert unrecognised_line_locators(document) == []
    assert not resolves_in_repository(file)


@settings(max_examples=300, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    here=st.one_of(st.none(), st.sampled_from(_IN_REPOSITORY_FILES)),
    locator=st.sampled_from(_QUOTABLE_LOCATORS),
    q1=_QUOTE_MARKS,
    q2=_QUOTE_MARKS,
    q3=_QUOTE_MARKS,
    q4=_QUOTE_MARKS,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    depth=st.integers(0, 3),
    as_key=st.booleans(),
)
@example(
    stem="x",
    suffix="py",
    directory="",
    here="https://github.com/PolicyEngine/microcosm/blob/main/tools/"
    "build_us_puf_support_base.py",
    locator="{q1}{file}{q2}#{q3}L{a}{q4}",
    q1="",
    q2="",
    q3="`",
    q4="",
    a=1,
    b=1,
    depth=0,
    as_key=False,
)
def test_a_quoted_line_locator_is_refused_as_a_value_or_a_key(
    stem: str,
    suffix: str,
    directory: str,
    here: str | None,
    locator: str,
    q1: str,
    q2: str,
    q3: str,
    q4: str,
    a: int,
    b: int,
    depth: int,
    as_key: bool,
) -> None:
    file = here or f"{directory}{stem}.{suffix}"
    text = locator.format(file=file, q1=q1, q2=q2, q3=q3, q4=q4, a=a, b=b)
    document = _plant(text, depth, as_key)
    # A quote after a URL's ``#`` ends the URL in the string's second reading,
    # which finds the same file's locator again (``x.py'L12``).
    found = line_citations(document)
    assert [citation for citation in found if citation.text == text] == [
        LineCitation(file, text)
    ]
    assert {citation.file for citation in found} == {file}
    assert unrecognised_line_locators(document) == []
    if here is None:
        assert LineCitation(file, text) in unpinned_line_citations(document)
    else:
        assert in_repository_line_citations(document) == found


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    root=_PIN_ROOT,
    locator=st.sampled_from(_LINE_LOCATORS),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    depth=st.integers(0, 3),
    as_key=st.booleans(),
    commit=_SHA,
)
def test_a_line_citation_of_a_file_pinned_at_a_commit_passes(
    stem: str,
    suffix: str,
    directory: str,
    root: str,
    locator: str,
    a: int,
    b: int,
    depth: int,
    as_key: bool,
    commit: str,
) -> None:
    citation = locator.format(file=f"{directory}{stem}.{suffix}", a=a, b=b)
    document = {
        "source": {
            "repository": "TheAxiomFoundation/ops",
            "path": f"{root}{directory}{stem}.{suffix}",
            "commit": commit,
        },
        "notes": [_plant(citation, depth, as_key)],
    }
    assert line_citations(document)
    assert unpinned_line_citations(document) == []


@settings(max_examples=300, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    cited_directory=_FOREIGN_DIRECTORY,
    pin_directory=_FOREIGN_DIRECTORY,
    root=_PIN_ROOT,
    locator=st.sampled_from(_LINE_LOCATORS),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    commit=_SHA,
)
@example(
    stem="x",
    suffix="py",
    cited_directory="archived_data/datasets/cps/",
    pin_directory="",
    root="",
    locator="{file}:{a}",
    a=12,
    b=12,
    commit="a1b2c3d",
)
def test_a_path_pin_covers_exactly_the_citations_its_path_ends_in(
    stem: str,
    suffix: str,
    cited_directory: str,
    pin_directory: str,
    root: str,
    locator: str,
    a: int,
    b: int,
    commit: str,
) -> None:
    cited = f"{cited_directory}{stem}.{suffix}"
    pinned = f"{root}{pin_directory}{stem}.{suffix}"
    citation = locator.format(file=cited, a=a, b=b)
    document = {"source": {"path": pinned, "commit": commit}, "note": citation}
    covered = pinned == cited or pinned.endswith("/" + cited)
    assert pin_covers(cited, pinned) is covered
    assert (unpinned_line_citations(document) == []) is covered


@settings(max_examples=300, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    prefix=_PROSE,
    middle=_PROSE,
    clause_middle=_CLAUSE_PROSE,
    tail=_PROSE,
    pin_first=st.booleans(),
    depth=st.integers(0, 3),
    as_key=st.booleans(),
)
# Only a commit in an earlier clause tests the fallback to the nearest commit
# before a citation, and a whole run can draw no ``;`` between them.
@example(
    stem="x",
    suffix="py",
    directory="",
    locator="{file}:{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
    prefix="",
    middle=";",
    clause_middle="",
    tail="",
    pin_first=True,
    depth=0,
    as_key=False,
)
@example(
    stem="x",
    suffix="py",
    directory="",
    locator="{file}:{a}",
    pin="https://github.com/PolicyEngine/archived-data/raw/{sha}/x",
    sha="a1b2c3d",
    a=12,
    b=12,
    prefix="",
    middle="",
    clause_middle="",
    tail="",
    pin_first=False,
    depth=0,
    as_key=False,
)
def test_a_commit_before_a_citation_or_later_in_its_clause_pins_it(
    stem: str,
    suffix: str,
    directory: str,
    locator: str,
    pin: str,
    sha: str,
    a: int,
    b: int,
    prefix: str,
    middle: str,
    clause_middle: str,
    tail: str,
    pin_first: bool,
    depth: int,
    as_key: bool,
) -> None:
    citation = locator.format(file=f"{directory}{stem}.{suffix}", a=a, b=b)
    named = pin.format(sha=sha)
    if pin_first:
        text = f"{prefix} {named} {middle} {citation} {tail}"
    else:
        text = f"{prefix} {citation} {clause_middle} {named} {tail}"
    document = _plant(text, depth, as_key)
    assert names_a_commit(text)
    assert [citing_commit(found) for found in line_citations(document)] == [sha]
    assert unpinned_line_citations(document) == []


@settings(max_examples=300, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    prefix=_PROSE,
    middle=_CLAUSE_PROSE,
    clause_break=st.sampled_from(_CLAUSE_BREAKS),
    tail=_PROSE,
)
@example(
    stem="x",
    suffix="py",
    directory="",
    locator="{file}:{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
    prefix="",
    middle="",
    clause_break="; ",
    tail="",
)
@example(
    stem="x",
    suffix="py",
    directory="",
    locator="{file}:{a}",
    pin="Commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
    prefix="",
    middle="",
    clause_break=". See ",
    tail="",
)
def test_a_commit_only_in_a_later_clause_does_not_pin_a_citation(
    stem: str,
    suffix: str,
    directory: str,
    locator: str,
    pin: str,
    sha: str,
    a: int,
    b: int,
    prefix: str,
    middle: str,
    clause_break: str,
    tail: str,
) -> None:
    file = f"{directory}{stem}.{suffix}"
    citation = locator.format(file=file, a=a, b=b)
    text = f"{prefix} {citation} {middle}{clause_break}{pin.format(sha=sha)} {tail}"
    assert names_a_commit(text)
    assert [citing_commit(found) for found in line_citations(text)] == [None]
    assert LineCitation(file, text) in unpinned_line_citations({"note": text})


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    moving=st.sampled_from(_MOVING_REFS),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    middle=_CLAUSE_PROSE,
    ref_first=st.booleans(),
)
@example(
    stem="x",
    suffix="py",
    directory="",
    locator="{file}:{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    moving="on refs/heads/main",
    a=12,
    b=12,
    middle="",
    ref_first=False,
)
def test_an_earlier_commit_does_not_pin_a_citation_whose_clause_names_a_moving_ref(
    stem: str,
    suffix: str,
    directory: str,
    locator: str,
    pin: str,
    sha: str,
    moving: str,
    a: int,
    b: int,
    middle: str,
    ref_first: bool,
) -> None:
    file = f"{directory}{stem}.{suffix}"
    citation = locator.format(file=file, a=a, b=b)
    clause = (
        f"{moving} {middle} {citation}"
        if ref_first
        else f"{citation} {middle} {moving}"
    )
    text = f"{pin.format(sha=sha)}; {clause}"
    assert [citing_commit(found) for found in line_citations(text)] == [None]
    assert LineCitation(file, text) in unpinned_line_citations({"note": text})


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    url=st.sampled_from(_MOVING_URLS),
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    clause_break=st.sampled_from(_CLAUSE_BREAKS),
)
def test_an_earlier_commit_does_not_pin_a_cited_url_at_a_moving_ref(
    stem: str,
    suffix: str,
    directory: str,
    url: str,
    locator: str,
    pin: str,
    sha: str,
    a: int,
    b: int,
    clause_break: str,
) -> None:
    file = f"{url}{directory}{stem}.{suffix}"
    citation = locator.format(file=file, a=a, b=b)
    text = f"{pin.format(sha=sha)}{clause_break}{citation}"
    assert [citing_commit(found) for found in line_citations(text)] == [None]
    assert LineCitation(file, text) in unpinned_line_citations({"note": text})


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    other=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    first_sha=_SHA,
    second_sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    middle=_CLAUSE_PROSE,
)
def test_each_citation_binds_the_commit_in_its_own_clause(
    stem: str,
    other: str,
    suffix: str,
    locator: str,
    first_sha: str,
    second_sha: str,
    a: int,
    b: int,
    middle: str,
) -> None:
    assume(stem != other)
    first = locator.format(file=f"{stem}.{suffix}", a=a, b=b)
    second = locator.format(file=f"{other}.{suffix}", a=b, b=a)
    text = f"{first} at {first_sha} {middle}; {second} at {second_sha}"
    assert {
        citation.name: citing_commit(citation) for citation in line_citations(text)
    } == {f"{stem}.{suffix}": first_sha, f"{other}.{suffix}": second_sha}


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
)
@example(
    stem="x",
    suffix="py",
    locator="{file}:{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
def test_a_commit_named_in_another_string_does_not_pin_a_citation(
    stem: str,
    suffix: str,
    locator: str,
    pin: str,
    sha: str,
    a: int,
    b: int,
) -> None:
    citation = locator.format(file=f"{stem}.{suffix}", a=a, b=b)
    document = {"provenance": pin.format(sha=sha), "note": citation}
    assert LineCitation(f"{stem}.{suffix}", citation) in unpinned_line_citations(
        document
    )


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    keyword=st.sampled_from(("commit ", "at ", "At ", "@", "/blob/")),
    not_a_commit=st.one_of(
        st.text(alphabet="0123456789", min_size=7, max_size=40),
        st.text(alphabet="abcdef", min_size=7, max_size=40),
        st.text(alphabet="0123456789abcdef", min_size=1, max_size=6),
        st.text(alphabet="0123456789abcdef", min_size=41, max_size=64),
        st.sampled_from(_BRANCH_NAMES),
        _SHA.map(lambda sha: f"{sha}…"),
        _SHA.map(lambda sha: f"{sha}..."),
        _SHA.map(lambda sha: f"{sha}-dirty"),
    ),
    a=st.integers(1, 99_999),
)
@example(
    stem="x",
    suffix="py",
    locator="{file}:{a}",
    keyword="commit ",
    not_a_commit="1234567",
    a=12,
)
@example(
    stem="x",
    suffix="py",
    locator="{file}:{a}",
    keyword="commit ",
    not_a_commit="e2de2b41…",
    a=12,
)
def test_a_decimal_a_word_a_digest_a_truncation_or_a_branch_names_no_commit(
    stem: str,
    suffix: str,
    locator: str,
    keyword: str,
    not_a_commit: str,
    a: int,
) -> None:
    citation = locator.format(file=f"{stem}.{suffix}", a=a, b=a)
    text = f"{citation} {keyword}{not_a_commit}"
    assert not names_a_commit(f"{keyword}{not_a_commit}")
    assert LineCitation(f"{stem}.{suffix}", text) in unpinned_line_citations(
        {"note": text}
    )


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    deeper=_PIN_ROOT,
    locator=st.sampled_from(_FORWARD_LOCATORS),
    keyword=st.sampled_from(("commit ", "Commit ", "at ", "At ", "uk-data@")),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
)
def test_a_hex_directory_after_a_pin_word_names_no_commit(
    stem: str,
    suffix: str,
    deeper: str,
    locator: str,
    keyword: str,
    sha: str,
    a: int,
    b: int,
) -> None:
    file = f"{sha}/{deeper}{stem}.{suffix}"
    text = locator.format(file=f"{keyword}{file}", a=a, b=b)
    assert not names_a_commit(text)
    assert LineCitation(file, text) in unpinned_line_citations({"note": text})
    assert misspelled_commit_pins({"note": text}) == []


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    other=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    commit=_SHA,
    branch=st.sampled_from(_BRANCH_NAMES),
    pin_other_file=st.booleans(),
)
@example(
    stem="x",
    other="y",
    suffix="py",
    locator="{file}:{a}",
    a=12,
    b=12,
    commit="a1b2c3d",
    branch="main",
    pin_other_file=True,
)
@example(
    stem="x",
    other="y",
    suffix="py",
    locator="{file}:{a}",
    a=12,
    b=12,
    commit="a1b2c3d",
    branch="main",
    pin_other_file=False,
)
def test_a_pin_of_another_file_or_at_a_branch_does_not_cover_a_citation(
    stem: str,
    other: str,
    suffix: str,
    locator: str,
    a: int,
    b: int,
    commit: str,
    branch: str,
    pin_other_file: bool,
) -> None:
    if pin_other_file:
        assume(other != stem)
        pin = {"path": f"{other}.{suffix}", "commit": commit}
    else:
        pin = {"path": f"{stem}.{suffix}", "commit": branch}
    citation = locator.format(file=f"{stem}.{suffix}", a=a, b=b)
    hits = unpinned_line_citations({"source": pin, "note": citation})
    assert LineCitation(f"{stem}.{suffix}", citation) in hits
    assert not any(
        pin_covers(f"{stem}.{suffix}", pinned)
        for pinned in commit_pinned_sources({"source": pin})
    )


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    not_a_commit=st.one_of(
        st.text(alphabet="0123456789", min_size=7, max_size=40),
        st.text(alphabet="abcdef", min_size=7, max_size=40),
        st.text(alphabet="0123456789ABCDEF", min_size=7, max_size=40).filter(
            lambda value: any(c.isalpha() for c in value)
        ),
    ),
    a=st.integers(1, 99_999),
)
@example(stem="x", suffix="py", locator="{file}:{a}", not_a_commit="1234567", a=12)
def test_a_path_pin_whose_commit_is_no_sha_pins_nothing(
    stem: str, suffix: str, locator: str, not_a_commit: str, a: int
) -> None:
    pin = {"path": f"{stem}.{suffix}", "commit": not_a_commit}
    citation = locator.format(file=f"{stem}.{suffix}", a=a, b=a)
    assert commit_pinned_sources({"source": pin}) == set()
    assert LineCitation(f"{stem}.{suffix}", citation) in unpinned_line_citations(
        {"source": pin, "note": citation}
    )


@settings(max_examples=300, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    stray=_SHA,
    word=st.sampled_from(("archived ", "rev ", "sha ", "(", "")),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    middle=_CLAUSE_PROSE,
    place=st.sampled_from(("gap", "clause_start", "clause_end")),
)
@example(
    stem="x",
    suffix="py",
    locator="{file}:{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    stray="e4f5a6b",
    word="rev ",
    a=12,
    b=12,
    middle="",
    place="gap",
)
@example(
    stem="x",
    suffix="py",
    locator="{file}:{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    stray="e4f5a6b",
    word="rev ",
    a=12,
    b=12,
    middle="",
    place="clause_end",
)
def test_a_commit_in_no_pin_spelling_beside_a_citation_is_refused(
    stem: str,
    suffix: str,
    locator: str,
    pin: str,
    sha: str,
    stray: str,
    word: str,
    a: int,
    b: int,
    middle: str,
    place: str,
) -> None:
    file = f"{stem}.{suffix}"
    citation = locator.format(file=file, a=a, b=b)
    named = pin.format(sha=sha)
    if place == "gap":
        # Between the commit the citation binds to and the citation's clause.
        text = f"{named} {word}{stray}; {middle} {citation}"
    elif place == "clause_start":
        # In the citation's own clause, before it.
        text = f"{named}; {word}{stray} {middle} {citation}"
    else:
        # In the citation's own clause, after it.
        text = f"{named} {middle} {citation} {word}{stray}"
    assert unpinned_line_citations({"note": text}) == []
    assert (LineCitation(file, text), stray) in misspelled_commit_pins({"note": text})


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    sha=_SHA,
    stray=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    middle=_CLAUSE_PROSE,
    stray_first=st.booleans(),
)
@example(
    stem="x",
    suffix="py",
    locator="{file}:{a}",
    sha="a1b2c3d",
    stray="e4f5a6b",
    a=12,
    b=12,
    middle="",
    stray_first=False,
)
def test_a_commit_like_token_in_another_clause_is_no_misspelled_pin(
    stem: str,
    suffix: str,
    locator: str,
    sha: str,
    stray: str,
    a: int,
    b: int,
    middle: str,
    stray_first: bool,
) -> None:
    citation = locator.format(file=f"{stem}.{suffix}", a=a, b=b)
    if stray_first:
        text = f"archived {stray} {middle}; {citation} at {sha}"
    else:
        text = f"{citation} at {sha}; archived {stray} {middle}"
    assert unpinned_line_citations({"note": text}) == []
    assert misspelled_commit_pins({"note": text}) == []


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    locator=st.sampled_from(_SYMBOL_LOCATORS),
    symbol=_STEM,
    prefix=_PROSE,
    tail=_PROSE,
)
def test_a_symbol_citation_is_not_a_line_citation(
    stem: str,
    suffix: str,
    directory: str,
    locator: str,
    symbol: str,
    prefix: str,
    tail: str,
) -> None:
    citation = locator.format(file=f"{directory}{stem}.{suffix}", symbol=symbol)
    document = {"rule": f"{prefix} {citation} {tail}"}
    assert line_citations(document) == []
    assert unrecognised_line_locators(document) == []


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    locator=st.sampled_from(_UNRECOGNISED_LOCATORS),
    symbol=_STEM,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    prefix=_PROSE,
    tail=_PROSE,
    depth=st.integers(0, 3),
    as_key=st.booleans(),
)
@example(
    stem="x",
    suffix="py",
    directory="",
    locator="{file} {symbol}, lines {a}-{b}",
    symbol="helper",
    a=12,
    b=19,
    prefix="",
    tail="",
    depth=0,
    as_key=False,
)
def test_a_line_locator_in_an_unrecognised_spelling_is_refused(
    stem: str,
    suffix: str,
    directory: str,
    locator: str,
    symbol: str,
    a: int,
    b: int,
    prefix: str,
    tail: str,
    depth: int,
    as_key: bool,
) -> None:
    assume(symbol not in _LOCATOR_WORDS)
    file = f"{directory}{stem}.{suffix}"
    text = f"{prefix} {locator.format(file=file, symbol=symbol, a=a, b=b)} {tail}"
    document = _plant(text, depth, as_key)
    assert line_citations(document) == []
    assert LineCitation(file, text) in unrecognised_line_locators(document)


@settings(max_examples=300, deadline=None)
@given(
    file=st.sampled_from(_IN_REPOSITORY_FILES),
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    pinned_inline=st.booleans(),
    pinned_by_path=st.booleans(),
)
@example(
    file=_IN_REPOSITORY_FILES[0],
    locator="{file}:{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
    pinned_inline=True,
    pinned_by_path=True,
)
def test_a_line_citation_of_this_repository_is_refused_even_when_pinned(
    file: str,
    locator: str,
    pin: str,
    sha: str,
    a: int,
    b: int,
    pinned_inline: bool,
    pinned_by_path: bool,
) -> None:
    text = locator.format(file=file, a=a, b=b)
    if pinned_inline:
        text = f"{text} {pin.format(sha=sha)}"
    document: dict[str, object] = {"note": text}
    if pinned_by_path:
        document["source"] = {"path": file, "commit": sha}
    assert resolves_in_repository(file)
    assert LineCitation(file, text) in in_repository_line_citations(document)


@settings(max_examples=200, deadline=None)
@given(
    url=st.sampled_from(_THIS_REPOSITORY_URLS),
    ref=st.lists(_REF_SEGMENT, min_size=1, max_size=4).map("/".join),
    path=st.sampled_from(_ROOTED_FILES),
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
)
def test_a_url_of_this_repository_is_refused_at_a_ref_of_any_depth(
    url: str,
    ref: str,
    path: str,
    locator: str,
    pin: str,
    sha: str,
    a: int,
    b: int,
) -> None:
    file = f"{url}{ref}/{path}"
    text = f"{locator.format(file=file, a=a, b=b)} {pin.format(sha=sha)}"
    assert resolves_in_repository(file)
    assert in_repository_line_citations({"note": text}) == [LineCitation(file, text)]


def test_in_repository_paths_resolve_from_an_anchor_or_a_repository_prefix() -> None:
    for file in (
        *_IN_REPOSITORY_FILES,
        "github.com/PolicyEngine/microcosm/blob/x/pyproject.toml",
        "tools/x/../build_us_puf_support_base.py",
        "a/../b/../tools/build_us_puf_support_base.py",
    ):
        assert resolves_in_repository(file), file
    for file in (
        "concepts.py",
        "pyproject.toml",
        "datasets/cps/cps.py",
        "src/lib.rs",
        "archived_data/datasets/cps/cps.py",
        "targets/sources/obr.py",
        "build_j_recert/buildj_base.sh",
        "../tools/build_us_puf_support_base.py",
        "tools/../../tools/build_us_puf_support_base.py",
        "tools/build_us_puf_support_base.py/..",
        "/tools/build_us_puf_support_base.py",
        "https://github.com/PolicyEngine/upstream-data/blob/main/tools/"
        "build_us_puf_support_base.py",
        "https://github.com/PolicyEngine/microcosm/blob/main/no/such/file.py",
        "microcosm/graph",
    ):
        assert not resolves_in_repository(file), file
    assert (REPOSITORY_ROOT / _IN_REPOSITORY_FILES[0]).is_file()
    assert COUNTRY_PACKAGE_ROOT.is_dir()


def test_every_source_path_here_is_read_whole() -> None:
    # The in-repository rule relies on it: a reading ends a path at any
    # character outside ``_PATH_HERE`` (whitespace, a backslash, a ``+``), so a
    # path here holding one could be cited by a name no reading resolves. An
    # ``=``, which evidence files here hold, is read through by a citation's
    # ``path``.
    tails = citations._repository_path_tails(REPOSITORY_ROOT)
    assert "tools/ci_test_plan.py" in tails
    assert sorted(tail for tail in tails if not set(tail) <= citations._PATH_HERE) == []


def test_a_path_here_that_holds_an_equals_sign_is_read_whole(tmp_path: Path) -> None:
    # ``_SOURCE`` starts a name after an ``=`` (``href=x.py`` names ``x.py``),
    # so a cited name's run of path characters is tried too, forward and
    # reverse, in prose and in a URL.
    cited = "docs/evidence/margins/period=2025-01.json"
    (tmp_path / cited).parent.mkdir(parents=True)
    (tmp_path / cited).write_text("{}", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/x.py").write_text("", encoding="utf-8")
    for written in (
        f"{cited} line 3 at a1b2c3d",
        f"microcosm/{cited}:3 at a1b2c3d",
        f"lines 3-4 of {cited} at a1b2c3d",
        f"https://github.com/PolicyEngine/microcosm/blob/a1b2c3d/{cited}#L3",
    ):
        found = in_repository_line_citations({"note": written}, tmp_path)
        assert {citation.file for citation in found} == {"2025-01.json"}, written
    found = in_repository_line_citations(
        {"note": "<a href=tools/x.py#L3>x</a>"}, tmp_path
    )
    assert [(citation.file, citation.path) for citation in found] == [
        ("tools/x.py", "href=tools/x.py")
    ]
    # Another path's run names nothing here, and a reverse locator reads a
    # name after an ``=`` as a forward one does, pinned or not.
    assert (
        in_repository_line_citations({"note": f"x={cited[5:]} line 3"}, tmp_path) == []
    )
    assert [
        citation.file
        for citation in unpinned_line_citations(
            {"note": "lines 3 of vendor/foo=bar.py"}
        )
    ] == ["bar.py"]


def test_only_anchored_tails_name_a_file_here(tmp_path: Path) -> None:
    files = (
        "packages/microcosm-build/src/microcosm/build/us_runtime/utils/loss.py",
        "packages/microcosm-build/tests/utils/loss.py",
        "tools/utils/loss.py",
    )
    for file in files:
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / file).write_text("", encoding="utf-8")
    tails = {
        "/".join(PurePosixPath(file).parts[cut:])
        for file in files
        for cut in range(len(PurePosixPath(file).parts))
    }
    assert {tail for tail in tails if resolves_in_repository(tail, tmp_path)} == {
        "packages/microcosm-build/src/microcosm/build/us_runtime/utils/loss.py",
        "microcosm-build/src/microcosm/build/us_runtime/utils/loss.py",
        "src/microcosm/build/us_runtime/utils/loss.py",
        "microcosm/build/us_runtime/utils/loss.py",
        "build/us_runtime/utils/loss.py",
        "us_runtime/utils/loss.py",
        "packages/microcosm-build/tests/utils/loss.py",
        "microcosm-build/tests/utils/loss.py",
        "tools/utils/loss.py",
    }
    for file in (
        "microcosm/tools/utils/loss.py",
        "../microcosm/tools/utils/loss.py",
        "https://github.com/PolicyEngine/microcosm/blob/main/tools/utils/loss.py",
        "https://github.com/policyengine/microcosm/tree/a1b2c3d/packages/"
        "microcosm-build/tests/utils/loss.py",
    ):
        assert resolves_in_repository(file, tmp_path), file


def test_the_round_one_review_counterexamples_are_refused() -> None:
    obr = "targets/sources/obr.py"
    for text in (
        f"uk-data {obr}:302 on main; compare us-data at 42ed5d4",
        f"us-data at 42ed5d4; uk-data {obr}:302 on main",
    ):
        assert unpinned_line_citations({"note": text}) == [LineCitation(obr, text)]
    origin = (
        "verified at archived commit 2e044571, loss.py line 33. The input was "
        "imputed (archived 42ed5d45 cps.py L197-271)."
    )
    assert [
        (citation.file, token) for citation, token in misspelled_commit_pins(origin)
    ] == [("cps.py", "42ed5d45")]
    pinned_elsewhere = {
        "source": {
            "path": "upstream_data/datasets/cps/cps.py",
            "commit": "a1b2c3d",
        },
        "note": "archived_data/datasets/cps/cps.py:12",
    }
    assert [
        citation.file for citation in unpinned_line_citations(pinned_elsewhere)
    ] == ["archived_data/datasets/cps/cps.py"]
    tool = "tools/build_us_puf_support_base.py"
    for text in (
        f"https://github.com/PolicyEngine/microcosm/blob/a1b2c3d/{tool}#L12",
        f"microcosm/{tool}:12 at a1b2c3d",
    ):
        assert len(in_repository_line_citations({"note": text})) == 1, text
    assert unpinned_line_citations({"note": "x.py:12 at e2de2b41…"})
    for text in (
        "x.py:L12",
        "x.py at line 12",
        "x.py at lines 12-19",
        "x.py; lines 12-19",
        "x.py — lines 12-19",
        "x.py [L12]",
        "lines 12 to 19 of x.py",
        "lines 12 through 19 in x.py",
        "lines 12-19 and 30 of x.py",
        "lines 12—19 of x.py",
        "x.py ll. 12-19",
    ):
        assert line_citations(text) == [LineCitation("x.py", text)], text


def test_the_round_two_review_counterexamples_are_refused() -> None:
    tool = "tools/build_us_puf_support_base.py"
    for document in ({"note": f'{tool}:"12-19"'}, {f'{tool}:"12-19"': "cited"}):
        assert [
            citation.file for citation in in_repository_line_citations(document)
        ] == [tool]
    cps = (
        "https://raw.githubusercontent.com/PolicyEngine/upstream-data/main/"
        "upstream_data/datasets/cps/cps.py"
    )
    for text in (f"commit a1b2c3d; {cps}#L12", "commit a1b2c3d; x.py:12 on `main`"):
        assert [citing_commit(found) for found in line_citations(text)] == [None]
        assert len(unpinned_line_citations({"note": text})) == 1, text
    text = "look at a1b2c3d/foo.py:12"
    assert unpinned_line_citations({"note": text}) == [
        LineCitation("a1b2c3d/foo.py", text)
    ]
    # A URL pin is still followed by its path.
    pinned = "https://github.com/o/upstream-data/blob/a1b2c3d/foo.py#L12"
    assert [citing_commit(found) for found in line_citations(pinned)] == ["a1b2c3d"]


def test_the_round_three_review_counterexample_is_refused() -> None:
    tool = "tools/build_us_puf_support_base.py"
    for url in (
        "https://raw.githubusercontent.com/PolicyEngine/microcosm/refs/heads/main/",
        "https://github.com/PolicyEngine/microcosm/blob/refs/heads/main/",
    ):
        text = f"{url}{tool}#L12 at a1b2c3d"
        assert in_repository_line_citations({"note": text}) == [
            LineCitation(f"{url}{tool}", text)
        ]


@settings(max_examples=300, deadline=None)
@given(
    url=st.sampled_from(_THIS_REPOSITORY_URLS),
    ref=st.one_of(_SHA, st.lists(_REF_SEGMENT, min_size=1, max_size=3).map("/".join)),
    detour=st.sampled_from(_DOT_DETOURS),
    path=st.sampled_from(_ROOTED_FILES),
    chosen=st.sets(st.integers(0, 160), max_size=12),
    upper=st.booleans(),
    case=st.sampled_from((str.lower, str.upper, str.title)),
    leads=st.sampled_from(_URL_LEADS),
    authority=st.sampled_from(_AUTHORITIES),
    wide=st.sets(st.integers(0, 24), max_size=6),
    stop=st.sampled_from(_FULL_STOPS),
    escaped_host=st.sets(st.integers(0, 24), max_size=6),
    dot=st.booleans(),
    backslashed=st.sets(st.integers(0, 160), max_size=6),
    query=st.one_of(st.just(""), _URL_QUERY.map("?".__add__)),
    fragment=st.sampled_from(("#L{a}", "#L{a}-L{b}")),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
)
# One spelling each: an escape in the route, an escaped slash after the ref,
# a ``./`` and a ``..`` detour, an escaped, a backslashed and a full-width
# host, a user, and a port on a host written without a scheme.
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="",
    path="tools/build_us_puf_support_base.py",
    chosen={1},
    upper=False,
    case=str.lower,
    leads=("https://", "https://"),
    authority="{host}",
    wide=set(),
    stop=_FULL_STOPS[0],
    escaped_host=set(),
    dot=False,
    backslashed=set(),
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="",
    path="tools/build_us_puf_support_base.py",
    chosen={36},
    upper=False,
    case=str.lower,
    leads=("https://", "https://"),
    authority="{host}",
    wide=set(),
    stop=_FULL_STOPS[0],
    escaped_host=set(),
    dot=False,
    backslashed=set(),
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="./",
    path="tools/build_us_puf_support_base.py",
    chosen=set(),
    upper=False,
    case=str.lower,
    leads=("https://", "https://"),
    authority="{host}",
    wide=set(),
    stop=_FULL_STOPS[0],
    escaped_host=set(),
    dot=False,
    backslashed=set(),
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="docs/../",
    path="tools/build_us_puf_support_base.py",
    chosen=set(),
    upper=False,
    case=str.lower,
    leads=("https://", "https://"),
    authority="{host}",
    wide=set(),
    stop=_FULL_STOPS[0],
    escaped_host=set(),
    dot=False,
    backslashed=set(),
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="",
    path="tools/build_us_puf_support_base.py",
    chosen=set(),
    upper=False,
    case=str.lower,
    leads=("https://", "https://"),
    authority="{host}",
    wide=set(),
    stop=_FULL_STOPS[0],
    escaped_host={0},
    dot=False,
    backslashed=set(),
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="",
    path="tools/build_us_puf_support_base.py",
    chosen=set(),
    upper=False,
    case=str.lower,
    leads=("https://", "https://"),
    authority="{host}",
    wide=set(),
    stop=_FULL_STOPS[0],
    escaped_host=set(),
    dot=False,
    backslashed={0},
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="",
    path="tools/build_us_puf_support_base.py",
    chosen=set(),
    upper=False,
    case=str.lower,
    leads=("https://", "https://"),
    authority="{host}",
    wide={0},
    stop=_FULL_STOPS[0],
    escaped_host=set(),
    dot=False,
    backslashed=set(),
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="",
    path="tools/build_us_puf_support_base.py",
    chosen=set(),
    upper=False,
    case=str.lower,
    leads=("https://", "https://"),
    authority="user@{host}",
    wide=set(),
    stop=_FULL_STOPS[0],
    escaped_host=set(),
    dot=False,
    backslashed=set(),
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    url="https://github.com/PolicyEngine/microcosm/blob/",
    ref="a1b2c3d",
    detour="",
    path="tools/build_us_puf_support_base.py",
    chosen=set(),
    upper=False,
    case=str.lower,
    leads=("", ""),
    authority="{host}:443",
    wide=set(),
    stop=_FULL_STOPS[0],
    escaped_host=set(),
    dot=False,
    backslashed=set(),
    query="",
    fragment="#L{a}",
    pin="commit {sha}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
def test_a_url_of_this_repository_is_refused_however_github_lets_it_be_written(
    url: str,
    ref: str,
    detour: str,
    path: str,
    chosen: set[int],
    upper: bool,
    case: Callable[[str], str],
    leads: tuple[str, str],
    authority: str,
    wide: set[int],
    stop: str,
    escaped_host: set[int],
    dot: bool,
    backslashed: set[int],
    query: str,
    fragment: str,
    pin: str,
    sha: str,
    a: int,
    b: int,
) -> None:
    parts = _URL_PARTS.fullmatch(url)
    assert parts is not None
    host, route = parts["host"], parts["route"]
    written_lead, lead = leads
    # A browser reads a host's escapes and full-width letters; written without
    # a scheme or slashes, a host is GitHub's only as spelled.
    spelled = case(host)
    if written_lead:
        spelled = _spell_host(
            spelled,
            {index % len(host) for index in wide},
            {index % len(host) for index in escaped_host},
            upper,
            stop,
        )
    # GitHub 404s a ``%2F`` before the ref; a browser reads ``\`` as ``/``.
    route_and_path = _escape(route, chosen, upper, keep="/") + _escape(
        f"{ref}/{detour}{path}", {index - len(route) for index in chosen}, upper
    )
    slashes = [index for index, char in enumerate(route_and_path) if char == "/"]
    backslashed = {slashes[index % len(slashes)] for index in backslashed}
    route_and_path = "".join(
        "\\" if index in backslashed else char
        for index, char in enumerate(route_and_path)
    )
    written = (
        case(written_lead)
        + authority.format(host=spelled + ("." if dot else ""))
        + route_and_path
    )
    text = f"{written}{query}{fragment.format(a=a, b=b)} {pin.format(sha=sha)}"
    # A quote or a parenthesis in the query ends the URL in the string's
    # second reading, which may read what follows as a quoted locator of the
    # same file (``x.py‘#L12``): the first reading is the URL's.
    read = next(citations._readable_strings(text))
    found = [
        citation
        for citation in in_repository_line_citations({"note": text})
        if citation.text == read
    ]
    assert [citation.file for citation in found] == [f"{lead}{host}{route}{ref}/{path}"]
    # A citation's text is already read: reading it again changes nothing.
    assert [
        citation
        for citation in in_repository_line_citations({"note": read})
        if citation.text == read
    ] == found


@settings(max_examples=200, deadline=None)
@given(
    directory=_FOREIGN_DIRECTORY,
    stem=_STEM,
    suffix=_SUFFIX,
    query=_URL_QUERY,
    fragment=st.sampled_from(("#L{a}", "#L{a}-L{b}")),
    sha=_SHA,
    pinned=st.booleans(),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
)
@example(
    directory="",
    stem="x",
    suffix="py",
    query="plain=1",
    fragment="#L{a}",
    sha="a1b2c3d",
    pinned=False,
    a=12,
    b=12,
)
def test_a_query_does_not_hide_a_line_citation(
    directory: str,
    stem: str,
    suffix: str,
    query: str,
    fragment: str,
    sha: str,
    pinned: bool,
    a: int,
    b: int,
) -> None:
    ref = sha if pinned else "main"
    url = f"https://github.com/PolicyEngine/upstream-data/blob/{ref}/{directory}{stem}.{suffix}"
    document = {"note": f"{url}?{query}{fragment.format(a=a, b=b)}"}
    # A quote or a parenthesis in the query ends the URL in the string's
    # second reading, which reads the rest of the query as prose: only the
    # first reading is the URL's.
    read = next(citations._readable_strings(document["note"]))
    found = [citation for citation in line_citations(document) if citation.text == read]
    assert [(citation.file, citing_commit(citation)) for citation in found] == [
        (url, sha if pinned else None)
    ]
    unpinned = [
        citation
        for citation in unpinned_line_citations(document)
        if citation.text == read
    ]
    assert unpinned == ([] if pinned else found)


#: A dot segment's two dots: either perhaps ``%2e``, written as a Markdown or
#: an HTML escape, or split by a tab, which the URL Standard deletes.
_DOT_DOTS = (
    "..",
    ".%2e",
    "%2E.",
    "%2e%2E",
    "&#46;&#46;",
    ".&period;",
    "\\.\\.",
    ".\t.",
)
#: What is no dot segment where one would be.
_NOT_DOT_DOTS = (".", "...", "x", "%2e", ".x", "x..", "&#46;")
#: Strings with a dot segment ({dots} beside a {separator}) and a line
#: citation ({url}), in prose, in markup, or in the cited URL's own path
#: ({inside}).
_DOT_SEGMENT_PLACES = (
    "{url} see docs{separator}{dots}",
    "{url} see {dots}{separator}docs",
    "{dots}{separator}docs{separator}notes: {url}",
    "{url} (docs{separator}{dots}{separator}notes)",
    '<a href="{dots}{separator}x">{url}</a>',
    "[a](<{dots}{separator}x>) {url}",
    "{inside}",
)


@settings(max_examples=300, deadline=None)
@given(
    place=st.sampled_from(_DOT_SEGMENT_PLACES),
    dots=st.sampled_from(_DOT_DOTS),
    not_dots=st.sampled_from(_NOT_DOT_DOTS),
    # GitHub reads ``%2f`` as ``/`` before it resolves a path.
    separator=st.sampled_from(("/", "\\", "%2f", "%2F")),
    directory=_FOREIGN_DIRECTORY,
    stem=_STEM,
    suffix=_SUFFIX,
    pinned=st.booleans(),
    sha=_SHA,
    a=st.integers(1, 99_999),
    depth=st.integers(0, 2),
    as_key=st.booleans(),
)
@example(
    place="{inside}",
    dots="..",
    not_dots=".",
    separator="%2f",
    directory="",
    stem="x",
    suffix="py",
    pinned=True,
    sha="a1b2c3d",
    a=12,
    depth=0,
    as_key=False,
)
@example(
    place="{dots}{separator}docs{separator}notes: {url}",
    dots="..",
    not_dots=".",
    separator="%2F",
    directory="",
    stem="x",
    suffix="py",
    pinned=False,
    sha="a1b2c3d",
    a=12,
    depth=0,
    as_key=False,
)
@example(
    place="{inside}",
    dots="..",
    not_dots=".",
    separator="/",
    directory="",
    stem="x",
    suffix="py",
    pinned=True,
    sha="a1b2c3d",
    a=12,
    depth=0,
    as_key=False,
)
@example(
    place="{url} see docs{separator}{dots}",
    dots="%2e%2E",
    not_dots="...",
    separator="\\",
    directory="datasets/cps/",
    stem="x",
    suffix="py",
    pinned=False,
    sha="a1b2c3d",
    a=12,
    depth=1,
    as_key=True,
)
@example(
    place='<a href="{dots}{separator}x">{url}</a>',
    dots="&#46;&#46;",
    not_dots="&#46;",
    separator="/",
    directory="",
    stem="x",
    suffix="py",
    pinned=True,
    sha="a1b2c3d",
    a=12,
    depth=0,
    as_key=False,
)
@example(
    place="{dots}{separator}docs{separator}notes: {url}",
    dots=".\t.",
    not_dots="x..",
    separator="/",
    directory="",
    stem="x",
    suffix="py",
    pinned=True,
    sha="a1b2c3d",
    a=12,
    depth=0,
    as_key=False,
)
def test_a_dot_segment_refuses_every_line_citation_in_its_string_pinned_or_not(
    place: str,
    dots: str,
    not_dots: str,
    separator: str,
    directory: str,
    stem: str,
    suffix: str,
    pinned: bool,
    sha: str,
    a: int,
    depth: int,
    as_key: bool,
) -> None:
    blob = "https://github.com/PolicyEngine/upstream-data/blob/"
    blob += f"{sha if pinned else 'main'}/{directory}"

    def written(dots: str) -> str:
        return place.format(
            url=f"{blob}{stem}.{suffix}#L{a}",
            # A letter after the separator: ``\_`` would be an escaped ``_``.
            inside=f"{blob}x{separator}{dots}{separator}docs/{stem}.{suffix}#L{a}",
            dots=dots,
            separator=separator,
        )

    document = _plant(written(dots), depth, as_key)
    assert dot_segment_line_citations(document) == line_citations(document) != []
    # Without the dot segment, the same string's citations pass this rule.
    document = _plant(written(not_dots), depth, as_key)
    assert line_citations(document)
    assert dot_segment_line_citations(document) == []


#: Markup that holds a URL ({url}) with whitespace in it.
_URL_ENCLOSURES = (
    "[a](<{url}>)",
    "[a]( <{url}>)",
    "[a](\t<{url}>)",
    "[a](\n<{url}>)",
    "[a]\n\n[a]: <{url}>",
    '<a href="{url}">a</a>',
    "<a href='{url}'>a</a>",
    'Use `prefix="` here.\n\n<a href="{url}">a</a>',
)


@settings(max_examples=200, deadline=None)
@given(
    enclosure=st.sampled_from(_URL_ENCLOSURES),
    path=st.sampled_from(_ROOTED_FILES),
    sha=_SHA,
    query=st.text(alphabet="ab=&+% \t\u00a0\u3000", max_size=60),
    fragment=st.sampled_from(("#L{a}", "#L{a}-L{b}")),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
)
@example(
    enclosure='Use `prefix="` here.\n\n<a href="{url}">a</a>',
    path="tools/build_us_puf_support_base.py",
    sha="a1b2c3d",
    query="a" + " " * 30 + "b",
    fragment="#L{a}",
    a=12,
    b=12,
)
@example(
    enclosure="[a]( <{url}>)",
    path="tools/build_us_puf_support_base.py",
    sha="a1b2c3d",
    query="a\tb" * 15,
    fragment="#L{a}-L{b}",
    a=3,
    b=9,
)
def test_a_line_link_of_this_repository_is_refused_whatever_whitespace_its_query_holds(
    enclosure: str,
    path: str,
    sha: str,
    query: str,
    fragment: str,
    a: int,
    b: int,
) -> None:
    # Each renders as a link to the file at its lines: GitHub ignores the
    # query, however long a run of spaces, tabs or Unicode spaces it holds.
    cited = f"https://github.com/PolicyEngine/microcosm/blob/{sha}/{path}"
    url = f"{cited}?{query}{fragment.format(a=a, b=b)}"
    document = {"note": enclosure.format(url=url)}
    assert cited in {
        citation.file for citation in in_repository_line_citations(document)
    }


def test_the_round_four_review_counterexamples_are_refused() -> None:
    blob = "https://github.com/PolicyEngine/microcosm/blob/"
    blob += "1840bc92f8d155adb685f1090245058ebb990df0/"
    tool = "tools/build_us_puf_support_base.py"
    # GitHub served each of these (HTTP 200) on 2026-10-26.
    for written in (
        f"{blob}%74ools/build_us_puf_support_base.py#L12",
        f"{blob}{tool}?plain=1&review={'x' * 32}#L12",
        f"{blob}tools%2Fbuild_us_puf_support_base.py#L12",
        f"{blob}tools/build_us_puf_support_base%2Epy#L12",
        f"{blob}docs/../{tool}#L12",
        f"{blob}docs/%2e%2e/{tool}#L12",
        f"{blob}./{tool}#L12",
        blob.replace("github.com", "GitHub.com") + f"{tool}#L12",
        blob.replace("github.com", "github.com:443") + f"{tool}#L12",
        blob.replace("github.com", "user@github.com") + f"{tool}#L12",
        blob.replace("/blob/", "/%62lob/") + f"{tool}#L12",
    ):
        assert [
            citation.file
            for citation in in_repository_line_citations({"note": written})
        ] == [f"{blob}{tool}"], written
    path = blob.removeprefix("https://github.com")
    for written, file in (
        (f"see GitHub.com{path}{tool}#L12.", f"github.com{path}{tool}"),
        (
            f"//github.com{path}{tool}?plain=1&review={'x' * 32}#L12",
            f"//github.com{path}{tool}",
        ),
    ):
        assert [
            citation.file
            for citation in in_repository_line_citations({"note": written})
        ] == [file], written
    # GitHub reads %23 and %3F as part of a file name (HTTP 404).
    for written in (f"{blob}{tool}%23L12", f"{blob}{tool}%3Fplain=1"):
        assert line_citations({"note": written}) == [], written
    # Punctuation after a URL is prose, so the clause still ends at the ``;``.
    assert unpinned_line_citations(
        {"note": "x.py:12, see https://example.org/doc?page=1; at a1b2c3d"}
    ) == [LineCitation("x.py", "x.py:12, see https://example.org/doc; at a1b2c3d")]
    assert (
        misspelled_commit_pins(
            {"note": "x.py:12 at a1b2c3d, see https://example.org/doc?page=1; 9f8e7d6c"}
        )
        == []
    )
    gitlab = "https://gitlab.com/o/upstream-data/-/blob/a1b2c3d/x.py"
    text = f"{gitlab}?ref_type=heads&view=plain&page=2 some_function, lines 12-19"
    assert unrecognised_line_locators({"note": text}) == [
        LineCitation(gitlab, f"{gitlab} some_function, lines 12-19")
    ]


def test_the_round_five_review_counterexamples_are_refused() -> None:
    route = "/PolicyEngine/microcosm/blob/c67030f75e07737f6aeb34841d2190a8009ec351/"
    tool = "tools/ci_test_plan.py"
    escaped = "%74ools/ci_test_plan.py"
    backslashed = (route + tool).replace("/", "\\")
    for written, lead in (
        (f"https://%67ithub.com{route}{tool}#L12", "https://"),
        (f"//github.com:443{route}{tool}#L12", "//"),
        (f"//user@github.com{route}{escaped}#L12", "//"),
        (f"https:github.com{route}{tool}#L12", "https://"),
        (f"HTTPS:\\\\GitHub.com{backslashed}#L12", "https://"),
        (f"\\\\github.com{route}{tool}#L12", "//"),
        (f"///github.com{route}{tool}#L12", "//"),
        (f"https://ｇithub.com{route}{tool}#L12", "https://"),
        (f"https://raw.githubu\ua7f1ercontent.com{route}{tool}#L12", "https://"),
        (f"https://github.com.{route}{tool}#L12", "https://"),
        (f"https://u:p@GITHUB.COM:0443{route}{tool}#L12", "https://"),
        (f"a@b@github.com:{route}{escaped}#L12", ""),
    ):
        host = "raw.githubusercontent.com" if "raw." in written else "github.com"
        # To Markdown ``\\`` is an escaped ``\``, so a decoded reading follows
        # the URL Standard's, which comes first.
        read = next(citations._readable_strings(written))
        assert [
            citation.file
            for citation in in_repository_line_citations({"note": written})
            if citation.text == read
        ] == [f"{lead}{host}{route}{tool}"], written
    # The URL Standard refuses each of these authorities, so no browser opens
    # them: each is read as written, and its path names no file here.
    for written in (
        f"https://%2567ithub.com{route}{tool}#L12",
        f"https://user%40github.com{route}{tool}#L12",
        f"https://github.com:44x{route}{tool}#L12",
        f"https://github.com:65536{route}{tool}#L12",
    ):
        assert in_repository_line_citations({"note": written}) == [], written


def test_the_round_six_review_counterexample_is_refused() -> None:
    blob = "https://github.com/PolicyEngine/microcosm/blob/"
    blob += "29983852e5182266c9a4e565731920ea972fe695/"
    tool = "tools/ci_test_plan.py"
    # A browser reads a quote or a bracket inside a URL as part of it (bun
    # 1.3's ``URL``), and GitHub served each of these, escaped as a browser
    # sends them (HTTP 200 on 2026-10-27).
    for written in (
        f"{blob}{tool}?value='{'x' * 40}'#L12",
        f'{blob}{tool}?value="x"#L12',
        f"{blob}{tool}?value=`x`#L12",
        f"{blob}{tool}?value=‘x’“x”#L12",
        f"{blob}{tool}?value=<x>#L12",
        f"{blob}tools/'x'/../ci_test_plan.py#L12",
        f"{blob}tools/<x>/../ci_test_plan.py#L12",
        f"'{blob}{tool}#L12'",
        f"“{blob}{tool}?value='x'#L12”",
        f"<{blob}{tool}#L12>",
        f"({blob}{tool}?value=(x)#L12).",
    ):
        assert [
            citation.file
            for citation in in_repository_line_citations({"note": written})
        ] == [f"{blob}{tool}"], written


def test_the_round_seven_review_counterexamples_are_refused() -> None:
    blob = "https://github.com/PolicyEngine/microcosm/blob/"
    blob += "29983852e5182266c9a4e565731920ea972fe695/"
    tool = "tools/ci_test_plan.py"
    cited = blob + tool
    network = cited.removeprefix("https:")
    bare = cited.removeprefix("https://")
    # CommonMark renders each of these with a link to the file at line 12,
    # however the URL before it ends.
    for written, file in (
        (f"[https://example.org/?x](<{cited}#L12>)", cited),
        (f"<https://example.org/?x><{cited}#L12>", cited),
        # With no query, only a reading that ends the first URL at its ``>``
        # reads the host a browser lower-cases.
        (f"<https://example.org/x><{cited.replace('github', 'GitHub')}#L12>", cited),
        (f"[https://example.org/?x]({cited}#L12)", cited),
        (f"[https://example.org/?x]({network}#L12)", network),
        (f"[https://example.org/?x]({bare}#L12)", bare),
        (f"[https://example.org/?x]({tool}#L12)", tool),
        (f'<a href="https://example.org/?x">{cited}#L12</a>', cited),
        (f"'https://example.org/?x','{cited}#L12'", cited),
    ):
        # Read through the first URL's query, the string cites the file again.
        assert {
            citation.file
            for citation in in_repository_line_citations({"note": written})
        } == {file}, written


def test_a_string_is_read_as_a_browser_and_as_markdown_delimit_its_urls() -> None:
    # As the URL Standard reads it (bun 1.3's ``URL``), a URL written straight
    # after another, with only a quote between, is part of the first one's
    # path, so its host is read as written; ended at the quote, as prose ends
    # it, it is read on its own.
    written = "https://example.org/a'https://GitHub.com/o/r/blob/a1b2c3d/x.py#L12"
    assert [citation.text for citation in line_citations({"note": written})] == [
        written,
        "https://example.org/a'https://github.com/o/r/blob/a1b2c3d/x.py#L12",
    ]
    # Ended at a quote in its query, a URL leaves the rest of the query to be
    # read as prose: a quoted locator (``x.py‘#L1``) of the same file, or,
    # with more between, a locator in no recognised spelling.
    url = "https://github.com/o/r/blob/a1b2c3d/x.py"
    assert [
        (citation.file, citing_commit(citation))
        for citation in line_citations({"note": f"{url}?0‘#L1"})
    ] == [(url, "a1b2c3d"), (url, "a1b2c3d")]
    assert unrecognised_line_locators({"note": f"{url}?q='v'#L1"}) == [
        LineCitation(url, f"{url}'v'#L1")
    ]
    # A string read the same way both times is read once.
    for written in (
        "https://GitHub.com/o/r/blob/a1b2c3d/x.py#L12",
        "(https://github.com/o/r/blob/a1b2c3d/x.py#L12).",
        "https://[::1]:8080/o/r/blob/a1b2c3d/x.py#L12",
    ):
        assert len(line_citations({"note": written})) == 1, written


def test_the_round_eight_review_counterexamples_are_refused() -> None:
    blob = "https://github.com/PolicyEngine/microcosm/blob/"
    blob += "29983852e5182266c9a4e565731920ea972fe695/"
    tool = "tools/ci_test_plan.py"
    cited = blob + tool
    # A range can end where another starts. Tried only from a list's head, a
    # reverse locator missed the one starting at the list's last number
    # (``L2-3`` after ``L1-L2``), or found a later, shorter one.
    for written in (
        f"L1-L2-3 of {tool}",
        f"L1 to L2 through 3 of {tool}",
        f"lines 1-L2-3 in {tool}",
        f"L1-L2, L3-L4-5 of {tool}",
    ):
        assert [
            citation.file
            for citation in in_repository_line_citations({"note": written})
        ] == [tool], written
    assert [
        (citation.start, citation.end)
        for citation in line_citations({"note": "L1-L2-L3 of x.py"})
    ] == [(3, 16)]
    # CommonMark (markdown-it-py 4.2) renders each Markdown link here, and
    # HTML decodes each attribute, as a link to line 12 of the file.
    for written in (
        f"[a]({blob}tools/ci_test_plan\\.py#L12)",
        f"[a]({blob}tools/ci\\_test\\_plan.py#L12)",
        f"[a](<{cited}\\#L12>)",
        f"[a]({cited}&#35;L12)",
        f"[a]({cited}#&#76;12)",
        f"[a]({blob}tools/ci_test_plan&#46;py#L12)",
        f"[a]({cited}&#x23;L12)",
        f"[a]({cited}&#0000035;L12)",
        f"[a]({cited}&num;L12)",
        f"[a]({blob}tools/ci_test_plan&period;py#L12)",
        f"[a][r]\n\n[r]: {cited}&#35;L12",
        f"[a]({cited.replace('github.com', 'github&#46;com')}#L12)",
        f'<a href="{cited}&#35;L12">a</a>',
        f"<a href='{cited}&#35L12'>a</a>",
        f'<a href="{cited}&num;L12">a</a>',
    ):
        assert {
            citation.file
            for citation in in_repository_line_citations({"note": written})
        } == {cited}, written
    # The URL Standard deletes each ASCII tab and line break from a URL, so
    # bun 1.3's ``URL`` reads each of these attributes as the same link.
    for written in (
        f'<a href="{blob}tools/ci_test_plan.p\ny#L12">a</a>',
        f'<a href="{blob}tools/ci_test_plan.p&#10;y#L12">a</a>',
        f'<a href="{cited}#L1&#13;\n2">a</a>',
        f'<a href="ht\ttps://git\nhub.com/{cited.removeprefix("https://github.com/")}#L12">a</a>',
    ):
        assert {
            citation.file
            for citation in in_repository_line_citations({"note": written})
        } == {cited}, written


def test_the_round_nine_review_counterexamples_are_refused() -> None:
    blob = "https://github.com/PolicyEngine/microcosm/blob/"
    blob += "29983852e5182266c9a4e565731920ea972fe695/"
    cited = blob + "tools/ci_test_plan.py"
    # parse5 7.3 decodes each attribute, markdown-it-py 4.2 renders each
    # Markdown link, and bun 1.3's ``URL`` then reads each as a link to line
    # 12 of the file.
    for written in (
        # HTML decodes the reference and leaves the backslash, which the URL
        # Standard reads as ``/``. Markdown's one pass took ``\&`` as an
        # escaped ``&``, so no reading decoded the reference.
        f'<a href="{blob}tools\\&#99;i_test_plan.py#L12">a</a>',
        f"<a href='{blob}tools\\&#x63;i_test_plan.py#L12'>a</a>",
        f"<a href={blob}tools\\&#99i_test_plan.py#L12>a</a>",
        f'<a href="{blob}tools\\&#0099;i_test_plan.py#L12">a</a>',
        f'<a href="{blob[:-1]}\\&#116;ools/ci_test_plan.py#L12">a</a>',
        f'<a href="{blob}tools\\&Tab;ci_test_plan.py#L12">a</a>',
        # Whitespace in a segment that a later ``..`` removes. An attribute's
        # quoted value and a destination in angle brackets hold any; a plain
        # destination and an unquoted value hold a Unicode space, or one a
        # reference decodes to. Each URL was read only up to it.
        f'<a href="{blob}tools/ x/../ci_test_plan.py#L12">a</a>',
        f"<a href='{blob}tools/x\x0c/../ci_test_plan.py#L12'>a</a>",
        f'<a href=" {blob}tools/a b/../ci_test_plan.py#L12 ">a</a>',
        f'<a href="{blob}tools/&quot; x/../ci_test_plan.py#L12">a</a>',
        f"[a](<{blob}tools/ x/../ci_test_plan.py#L12>)",
        f"[a]: <{blob}tools/ x/../ci_test_plan.py#L12>\n\n[a]",
        f"[a]({blob}tools/\u00a0x/../ci_test_plan.py#L12)",
        f"[a]({blob}tools/&#32;x/../ci_test_plan.py#L12)",
        f"[a]({blob}tools/&nbsp;x/../ci\\_test\\_plan.py#L12)",
        f"[https://example.org/?x]({blob}tools/&quot;/../ci_test_plan.py#L12)",
        f"<a href={blob}tools/x&#32;/../ci_test_plan.py#L12>a</a>",
        f"<a href={blob}tools/x\u00a0/../ci_test_plan.py#L12>a</a>",
        f"<{blob}tools/x\u00a0/../ci_test_plan.py#L12>",
        # An attribute's value is read after whitespace around its ``=``, and
        # with the tabs and line breaks in it deleted before it is stripped.
        f'<a href = " {blob}tools/a b/../ci_test_plan.py#L12 ">a</a>',
        f'<a href=\n"{blob}tools/ x/../ci_test_plan.py#L12">a</a>',
        f'<a href="ht\ttps://{blob[8:]}tools/a b/../ci_test_plan.py#L12">a</a>',
        f'<a href="\thttps://{blob[8:]}tools/a b/../ci_test_plan.py#L12">a</a>',
        # HTML decodes a reference before the URL Standard strips the value's
        # ends, and markdown-it strips whitespace from a destination's ends
        # before following it.
        f'<a href="&#32;{blob}tools/a b/../ci_test_plan.py#L12">a</a>',
        f'<a href="&#12;{blob}tools/a b/../ci_test_plan.py#L12&#32;">a</a>',
        f"[a](< {blob}tools/x /../ci_test_plan.py#L12 >)",
        f"[a](<\u3000{blob}tools/x /../ci_test_plan.py#L12\u00a0>)",
        f"[a](<&#32;{blob}tools/x /../ci_test_plan.py#L12>)",
        # HTML reads ``&#1;`` as U+0001. Python's ``html.unescape`` dropped it,
        # so ``.&#1;.`` became a ``..`` that removed ``tools``.
        f'<a href="{blob}tools/.&#1;./../ci_test_plan.py#L12">a</a>',
        # In an attribute, a legacy name before a letter is left as written.
        f'<a href="{blob}tools/&notx/../&#99;i_test_plan.py#L12">a</a>',
    ):
        assert {
            citation.file
            for citation in in_repository_line_citations({"note": written})
        } == {cited}, written
    # A relative path's inner ``..`` is resolved too.
    pinned = (
        "tools/x/../ci_test_plan.py#L12 at 29983852e5182266c9a4e565731920ea972fe695"
    )
    assert [
        citation.file for citation in in_repository_line_citations({"note": pinned})
    ] == ["tools/x/../ci_test_plan.py"]
    # Reading through a Unicode space or an attribute's value refuses
    # nothing written plainly: a pinned citation with a no-break space, a
    # title holding a URL and a space, and prose after a URL.
    sha = "29983852e5182266c9a4e565731920ea972fe695"
    for written in (
        f"https://github.com/o/r/blob/{sha}/x.py\u00a0line 12",
        f'<a title="https://example.org/a b" href="https://github.com/o/r/blob/{sha}/x.py#L12">',
        f"see https://example.org/a\u00a0b and x.py line 12 at {sha}",
    ):
        document = {"note": written}
        assert in_repository_line_citations(document) == [], written
        assert unpinned_line_citations(document) == [], written
        assert misspelled_commit_pins(document) == [], written
        assert unrecognised_line_locators(document) == [], written


def test_the_round_ten_review_counterexamples_are_refused() -> None:
    blob = "https://github.com/PolicyEngine/microcosm/blob/"
    blob += "29983852e5182266c9a4e565731920ea972fe695/"
    tools = blob + "tools/"
    cited = tools + "ci_test_plan.py"
    network = tools.removeprefix("https:")
    code_span = 'Use `prefix="` before the value.\n\n'
    # markdown-it-py 4.2 and parse5 7.3 render each of these, and bun 1.3's
    # ``URL`` reads each, as a link to line 12 of the file: the ``..``
    # removes a segment that holds whitespace or a backslash. Before round 11
    # no rule refused them.
    for written, in_repository in (
        # CommonMark allows spaces, tabs and one line break before ``<``.
        (f"[a]( <{tools}a b/../ci_test_plan.py#L12>)", True),
        (f"[a](\t<{tools}a b/../ci_test_plan.py#L12>)", True),
        (f"[a](\n<{tools}a b/../ci_test_plan.py#L12>)", True),
        (f"[a]( \t\n \t<{tools}a b/../ci_test_plan.py#L12>)", True),
        (f"[a]\n\n[a]: \n<{tools}a b/../ci_test_plan.py#L12>", True),
        # The ``="`` in the code span starts an enclosure that ends at the
        # attribute's opening quote, so no reading holds its value.
        (f'{code_span}<a href="{tools}a b/../ci_test_plan.py#L12">a</a>', False),
        (f"{code_span}<a href='{tools}a b/../ci_test_plan.py#L12'>a</a>", True),
        # Markdown writes a backslash it leaves as ``%5C``, so ``a\b`` is one
        # segment; a reading takes the backslash as ``/``.
        (f"[a](<{tools}a\\b/../ci_test_plan.py#L12>)", False),
        (f"[a]({tools}a\\b/../ci_test_plan.py#L12)", False),
        # markdown-it decodes the references, then strips the destination.
        (f"[a](&#32;{network}a&#32;b/../ci_test_plan.py#L12)", False),
        (f"[a](&#32;{tools}a&#32;b/../ci_test_plan.py#L12)", True),
    ):
        document = {"note": written}
        assert dot_segment_line_citations(document), written
        assert {
            citation.file for citation in in_repository_line_citations(document)
        } == ({cited} if in_repository else set()), written
    # GitHub ignores a query, so whitespace in one changes no file: each of
    # these links to line 12 (or 3-9) of the file, and a reading through the
    # query finds it however far the whitespace runs.
    query = "?" + "a b=c d&" * 4
    no_break, tab = "a\u00a0b" * 10, "a\tb" * 10
    for written in (
        f"[a]( <{cited}{query}#L12>)",
        f"[a](\t<{cited}{query}#L12>)",
        f"[a](\n<{cited}{query}#L12>)",
        f"[a](<{cited}{query}#L12>)",
        f'{code_span}<a href="{cited}{query}#L12">a</a>',
        f"<a href='{cited}{query}#L12'>a</a>",
        f'<a href="{cited}?{"a b " * 8}#L12">a</a>',
        f"[a](<{cited}?{'a&#32;b' * 6}#L12>)",
        f"[a](<{cited}?{no_break}#L12>)",
        f"[a](<{cited}?{tab}#L12>)",
        f"[a](<{cited}?{'a b' * 10}#L3-L9>)",
        # Where no enclosure holds the link, a query read through to a ``#``
        # swallows a later URL, so one reading ends a query at each
        # enclosure's end: a ``"``, a ``'`` and an angle bracket.
        f'Use `prefix="` here. <a title=https://example.org/?x href="{cited}{query}#L12">a</a>',
        f"Use `prefix='` here. <a title=https://example.org/?x href='{cited}{query}#L12'>a</a>",
        f'Use `prefix="` here [x](<https://example.org/?x>) [a](<{cited}{query}#L12>) "',
    ):
        assert {
            citation.file
            for citation in in_repository_line_citations({"note": written})
        } == {cited}, written


def test_a_dot_segment_refuses_the_line_citations_of_its_string_only() -> None:
    pinned = "https://github.com/PolicyEngine/upstream-data/blob/a1b2c3d/x.py#L12"
    # A run of dots that is no whole segment, and a lone ``.``, remove nothing.
    for written in (
        f"{pinned}; rows 1 .. 9",
        f"{pinned} via x/./y",
        f"{pinned} at sha...",
        f"{pinned}; a.b..c",
        "x.py:12 at a1b2c3d; see docs/.../notes",
        "see ../docs and x/../y; no line here",
    ):
        assert dot_segment_line_citations({"note": written}) == [], written
    # A dot segment anywhere in the string refuses each citation in it, and
    # none in another string.
    for written, count in (
        (f"{pinned}; see ../docs", 1),
        ("x.py:12 at a1b2c3d; see docs/%2e%2E/notes", 1),
        ('x.py:12 at a1b2c3d; <a href="..\\x">x</a>', 1),
        ("x.py:12 and y.py:30 at a1b2c3d; see ../docs", 2),
        ("x.py:12 at a1b2c3d; see ..%2fdocs", 1),
        ("x.py:12 at a1b2c3d; see docs%2F.%2e%2fnotes", 1),
    ):
        document = {"note": written, "other": pinned}
        assert dot_segment_line_citations(document) == line_citations(written)
        assert len(dot_segment_line_citations(document)) == count, written


def test_the_round_eleven_review_counterexamples_are_refused() -> None:
    sha = "2822aff1ce521b2db42593ac38c7d81b44ca98a2"
    blob = "https://github.com/PolicyEngine/microcosm/blob/"
    path = f"/PolicyEngine/microcosm/blob/{sha}/tools/ci_test_plan.py"
    code_span = 'Use `prefix="` before the value.\n\n'
    # markdown-it-py 4.2 and parse5 7.3 render each of these, and bun 1.3's
    # ``URL`` reads each, as a link to line 12 of ``tools/ci_test_plan.py`` on
    # github.com; at ``2822aff1`` no rule refused them. A ref can hold
    # punctuation at which a reading starts a name, so the citation it finds
    # (``foo/tools/ci_test_plan.py``) names no file here.
    written = []
    for ref in (
        "feature+foo",
        "feature@foo",
        "release!1",
        "v1,2;3=4",
        "it's",
        "a$b",
        "a=b",
    ):
        written.append(f"[a](<{blob}{ref}/tools/ci_test_plan.py#L12>) at {sha}")
        written.append(
            f'<a href="{blob}{ref}/tools/ci_test_plan.py#L12">a</a> at {sha}'
        )
    # Where a code span's ``="`` has taken an attribute's opening quote, a
    # reading ends the URL at the whitespace in its user, and finds no host
    # in what follows; so too with a query longer than the backstop's
    # window, and with the owner escaped and its slash doubled, which GitHub
    # reads through.
    for authority in (
        "a b@%67ithub.com",
        "a b@ｇithub.com",
        "a b@github。com",
        "a b@%67ithub.com:443",
        "a\tb c@%47ITHUB.com",
    ):
        written.append(f'{code_span}<a href="https://{authority}{path}#L12">a</a>')
    written += (
        f'{code_span}<a href="https://a b@%67ithub.com{path}?{"a" * 30}=1#L12">a</a>',
        f'{code_span}<a href="https://a b@%67ithub.com/%50olicyEngine//microcosm/./'
        f'blob/{sha}/tools/ci_test_plan.py#L12">a</a>',
    )
    for text in written:
        assert line_links_of_this_repository({"note": text}) == [text], text
    # GitHub reads ``%2f`` as ``/`` and then resolves the path: github.com
    # serves ``tools/ci_test_plan.py`` for this, though no source held a dot
    # segment by round 11's definition.
    text = (
        f'{code_span}<a href="{blob}{sha}/tools/a b%2f..%2fci_test_plan.py#L12">a</a>'
    )
    assert line_links_of_this_repository({"note": text}) == [text]
    assert dot_segment_line_citations({"note": text}) == line_citations(text) != []


def test_a_line_link_of_this_repository_is_refused_by_its_path_alone() -> None:
    # Whatever file it names: through markup and ``%`` escapes, backslashes,
    # repeated slashes, ``.`` segments, a tab and the owner's case, as a
    # browser and then GitHub read a path. A Makefile or a workflow outside
    # the source directories drifts too.
    for written in (
        "https://github.com/PolicyEngine/microcosm/blob/main/Makefile#L3",
        "https://github.com/PolicyEngine/microcosm/blob/a1b2c3d/.github/workflows/ci.yml#L30",
        "github.com/PolicyEngine&#47;microcosm/blob/x/a.py&#35;L12",
        "PolicyEngine\\microcosm\\blob\\x\\a.py#L12",
        "policyengine//MICROCOSM/blob/x/a.py#L3-L9",
        "PolicyEngine/./microcosm/blob/x/a.py#L3",
        "PolicyEngine/micro\tcosm/blob/x/a.py#L3",
        "%50olicyEngine/%6Dicrocosm/blob/x/a.py#L3",
        # Prose between the path and a later line fragment is read as part of
        # the link: refusing the string is intended, as a dot segment's is.
        "see github.com/PolicyEngine/microcosm/tree/main/docs and cps.py#L197 at a1b2c3d",
    ):
        assert line_links_of_this_repository({"note": written}) == [written], written
    # An issue, a comment, a heading or a bare ``L`` is no line; another
    # repository's path is not this one's; a ``#`` between the path and a line
    # fragment starts the link's fragment first; and the path in another
    # string links nothing in this one.
    for written in (
        "PolicyEngine/microcosm#38 and x.py#L12 at a1b2c3d",
        "https://github.com/PolicyEngine/microcosm/pull/939#issuecomment-1; x.py#L12 at a1b2c3d",
        "https://github.com/PolicyEngine/microcosm/blob/main/README.md#readme",
        "https://github.com/PolicyEngine/microcosm/blob/main/a.py#L",
        "https://github.com/PolicyEngine/microcosm-dynamics/blob/a1b2c3d/x.py#L12",
        "https://github.com/NotPolicyEngine/microcosm/blob/a1b2c3d/x.py#L12",
        "github.com/PolicyEngine/microcosm/issues/1 (#2) and x.py#L12 at a1b2c3d",
        "x.py#L12 at a1b2c3d; see github.com/PolicyEngine/microcosm/",
    ):
        assert line_links_of_this_repository({"note": written}) == [], written
    document = {"a": "github.com/PolicyEngine/microcosm/", "b": "x.py#L12 at a1b2c3d"}
    assert line_links_of_this_repository(document) == []


#: A ref segment holding punctuation at which a reading starts a name, though
#: a URL's path holds it: ``feature+foo``, ``release!1``, ``v1,2;3=4``.
_PUNCTUATED_REF_SEGMENT = st.from_regex(
    r"[A-Za-z0-9_+@!$,;=()-]{1,12}", fullmatch=True
).filter(lambda segment: "microcosm" not in segment.lower())
#: A user and a host of a link to github.com: whitespace in a user ends a URL
#: in a reading, and a browser reads a host's escapes, full-width letters and
#: ideographic full stop.
_GITHUB_AUTHORITIES = (
    "github.com",
    "user@github.com",
    "a b@%67ithub.com",
    "a\tb c@ｇithub.com",
    "x y@github。com:443",
)
#: This repository's owner and name as GitHub routes them, and other
#: repositories' of the same shape.
_THIS_REPOSITORY_PATHS = (
    "PolicyEngine/microcosm",
    "policyengine/MICROCOSM",
    "%50olicyEngine/%6Dicrocosm",
    "PolicyEngine//microcosm",
    "PolicyEngine/./microcosm",
)
_OTHER_REPOSITORY_PATHS = (
    "PolicyEngine/microcosm-dynamics",
    "PolicyEngine/microcosms",
    "NotPolicyEngine/microcosm",
    "PolicyEngine/micro",
)


@settings(max_examples=300, deadline=None)
@given(
    enclosure=st.sampled_from(_URL_ENCLOSURES),
    authority=st.sampled_from(_GITHUB_AUTHORITIES),
    repository=st.sampled_from(_THIS_REPOSITORY_PATHS),
    other=st.sampled_from(_OTHER_REPOSITORY_PATHS),
    route=st.sampled_from(("blob", "tree", "blame")),
    ref=st.lists(_PUNCTUATED_REF_SEGMENT, min_size=1, max_size=3).map("/".join),
    path=st.sampled_from(_ROOTED_FILES),
    query=st.sampled_from(("", "?plain=1", "?" + "a" * 30, "?a b=c")),
    fragment=st.sampled_from(("#L{a}", "#L{a}-L{b}")),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
)
@example(
    enclosure="[a](<{url}>)",
    authority="github.com",
    repository="PolicyEngine/microcosm",
    other="PolicyEngine/microcosm-dynamics",
    route="blob",
    ref="feature+foo",
    path="tools/build_us_puf_support_base.py",
    query="",
    fragment="#L{a}",
    sha="a1b2c3d",
    a=12,
    b=12,
)
@example(
    enclosure='Use `prefix="` here.\n\n<a href="{url}">a</a>',
    authority="a b@%67ithub.com",
    repository="%50olicyEngine/%6Dicrocosm",
    other="NotPolicyEngine/microcosm",
    route="tree",
    ref="release!1/x",
    path="tools/build_us_puf_support_base.py",
    query="?" + "a" * 30,
    fragment="#L{a}-L{b}",
    sha="a1b2c3d",
    a=3,
    b=9,
)
def test_a_line_link_of_this_repository_is_refused_whatever_its_ref_or_authority_holds(
    enclosure: str,
    authority: str,
    repository: str,
    other: str,
    route: str,
    ref: str,
    path: str,
    query: str,
    fragment: str,
    sha: str,
    a: int,
    b: int,
) -> None:
    # markdown-it-py 4.2, parse5 7.3 and bun 1.3 render each of these as a
    # link to the file's lines on github.com, which routes the path, except a
    # full-width host in a Markdown destination, which markdown-it writes as
    # no URL bun reads; each is refused, pinned or not, and another
    # repository's link of the same shape is not.
    def written(repository: str) -> str:
        line = fragment.format(a=a, b=b)
        url = f"https://{authority}/{repository}/{route}/{ref}/{path}{query}{line}"
        return f"Pinned at {sha}. " + enclosure.format(url=url)

    here = written(repository)
    assert line_links_of_this_repository({"note": here}) == [here]
    assert line_links_of_this_repository({"note": written(other)}) == []


def test_a_string_is_read_again_with_its_markup_escapes_decoded() -> None:
    def readings(text: str) -> list[str]:
        return list(citations._readable_strings(text))

    # Decoded in one pass, as CommonMark decodes: an escaped ``&`` starts no
    # reference, and a decoded one starts no second. HTML leaves the
    # backslash as written and decodes the reference after it.
    assert readings("x\\&amp;y") == ["x\\&amp;y", "x&amp;y", "x\\&y"]
    assert readings("x&amp;#35;y") == ["x&amp;#35;y", "x&#35;y"]
    # A backslash before anything but ASCII punctuation stays.
    assert readings("x\\a\\é\\1") == ["x\\a\\é\\1"]
    # HTML decodes a numeric reference without its ``;``.
    assert readings("&#35L1") == ["&#35L1", "#L1"]
    # A numeric reference is read past its leading zeros, and one past
    # U+10FFFF is U+FFFD, however many digits it has: converting 5,000
    # digits whole raises.
    assert readings("&#" + "0" * 5_000 + "35;L1") == [
        "&#" + "0" * 5_000 + "35;L1",
        "#L1",
    ]
    # So is one to 0 or to a surrogate.
    for written in (
        "&#" + "9" * 5_000 + ";",
        "&#x" + "f" * 5_000 + ";",
        "&#x110000;",
        "&#0;",
        "&#xD800;",
    ):
        assert readings(written) == [written, "\ufffd"], written[:12]
    # HTML reads a reference to a control as the control (Python's
    # ``html.unescape`` drops it) and one to a C1 control as windows-1252
    # does. A legacy name before a letter or ``=`` is decoded in text; in a
    # URL, as in an attribute, it is left as written.
    assert readings("&#1;&#128;&notit;&amp=x") == [
        "&#1;&#128;&notit;&amp=x",
        "\x01\u20ac\u00acit;&=x",
        "\x01\u20ac&notit;&amp=x",
    ]
    # In a URL, whitespace a reference decodes to is held as a browser holds
    # it: percent-encoded, or deleted where it is a tab or a line break.
    assert readings("x&#32;y&nbsp;z&#9;") == [
        "x&#32;y&nbsp;z&#9;",
        "x y\xa0z\t",
        "x y\xa0z",
        "x%20y%C2%A0z",
    ]
    # The escapes are decoded before a URL is read: ``%`` decoded from
    # ``&#37;`` starts an escape the URL's path decodes.
    url = "https://github.com/o/r/blob/a1b2c3d/x&#37;2Epy#L1"
    assert readings(url)[-1] == "https://github.com/o/r/blob/a1b2c3d/x.py#L1"
    # A tab or a line break, as written or decoded, is deleted in a reading
    # of its own.
    assert readings("a\tb\nc\r\nd") == ["a\tb\nc\r\nd", "abcd"]
    assert readings("x&#10;y") == ["x&#10;y", "x\ny", "xy"]
    # An attribute's quoted value and a destination in angle brackets are
    # read as the URL a browser follows: stripped at its ends, with what it
    # holds held. A value that is no URL is left as written.
    assert readings('<a title="a b" href=" https://example.org/a b/../x.py#L1 ">') == [
        '<a title="a b" href=" https://example.org/a b/../x.py#L1 ">',
        '<a title="a b" href="https://example.org/x.py#L1">',
    ]
    assert readings("[a](<https://example.org/a b/../x\\.py#L1>)") == [
        "[a](<https://example.org/a b/../x\\.py#L1>)",
        "[a](<https://example.org/x.py#L1>)",
        "[a](<https://example.org/a b/../x.py#L1>)",
    ]
    # Where a string holds a Unicode space, its URLs are read through it too.
    assert readings("https://example.org/a\xa0b/../x.py#L1 c") == [
        "https://example.org/a\xa0b/../x.py#L1 c",
        "https://example.org/x.py#L1 c",
    ]
    # A string with nothing to decode is read as before.
    for written, read in (
        ("x.py:12", "x.py:12"),
        ("a & b; c\\d", "a & b; c\\d"),
        ("https://example.org/?a=1&b=2", "https://example.org/"),
    ):
        assert readings(written) == [read], written


#: Each ASCII character's named character references (``&num;`` for ``#``),
#: from the HTML standard's table, which CommonMark reads too.
_NAMED_REFERENCES = {
    char: tuple(
        sorted(
            name
            for name, value in html5.items()
            if value == char and name.endswith(";")
        )
    )
    for char in (*string.printable, "\u00a0")
}
#: Markdown and HTML around a link destination: whether a backslash escape is
#: decoded there (Markdown) or not (HTML), and the whitespace a URL there can
#: hold as written. An HTML attribute's URL may hold a tab or a line break,
#: which the URL Standard deletes; a Markdown destination may not.
_MARKUP = (
    ("[a]({})", True, "\u00a0"),
    ("[a](<{}>)", True, " \u00a0"),
    ("[a][r]\n\n[r]: {}", True, "\u00a0"),
    ("See [the plan]({} 'title').", True, "\u00a0"),
    ('<a href="{}">a</a>', False, " \u00a0\x0c\t\n\r"),
    ("<a href='{}'>a</a>", False, " \u00a0\x0c\t\n\r"),
    ("<a href={}>a</a>", False, "\u00a0"),
)


def _escape_markup(
    text: str,
    chosen: dict[int, int],
    backslash: bool,
    zeros: int,
    breaks: dict[int, str] | None = None,
) -> str:
    """``text`` with its characters at ``chosen`` written as Markdown or HTML
    escape them: a backslash before ASCII punctuation (where ``backslash``), a
    decimal reference with ``zeros`` leading zeros, a hexadecimal one, or a
    named one, each chosen by its number, falling back to the next. Each of
    ``breaks`` goes before the character at its index."""

    def spell(char: str, kind: int) -> str:
        for option in range(kind, kind + 4):
            option %= 4
            if option == 0 and backslash and char in string.punctuation:
                return "\\" + char
            if option == 1:
                return f"&#{'0' * zeros}{ord(char)};"
            if option == 2:
                return f"&#x{ord(char):X};"
            if option == 3 and _NAMED_REFERENCES.get(char):
                names = _NAMED_REFERENCES[char]
                return "&" + names[zeros % len(names)]
        raise AssertionError(char)

    return "".join(
        (breaks or {}).get(index, "")
        + (spell(char, chosen[index]) if index in chosen else char)
        for index, char in enumerate(text)
    )


#: An ASCII tab or line break, as written or as an HTML reference.
_URL_BREAK_SPELLINGS = ("\n", "\t", "\r\n", "&#10;", "&#x9;", "&NewLine;")


@settings(max_examples=300, deadline=None)
@given(
    url=st.sampled_from(_THIS_REPOSITORY_URLS),
    sha=_SHA,
    path=st.sampled_from(_ROOTED_FILES),
    fragment=st.sampled_from(("#L{a}", "#L{a}-L{b}")),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    markup=st.sampled_from(_MARKUP),
    chosen=st.dictionaries(
        st.integers(0, 200), st.integers(0, 3), min_size=1, max_size=24
    ),
    zeros=st.integers(0, 3),
    breaks=st.dictionaries(
        st.integers(0, 200), st.sampled_from(_URL_BREAK_SPELLINGS), max_size=3
    ),
    detour=st.integers(-1, 5),
    backslashes=st.sets(st.integers(0, 20), max_size=2),
)
@example(
    url=_THIS_REPOSITORY_URLS[0],
    sha="a1b2c3d",
    path=_ROOTED_FILES[0],
    fragment="#L{a}",
    a=12,
    b=12,
    markup=_MARKUP[4],
    chosen={0: 0},
    zeros=0,
    breaks={},
    detour=0,
    backslashes={7},
)
def test_a_line_link_of_this_repository_is_refused_however_markup_escapes_it(
    url: str,
    sha: str,
    path: str,
    fragment: str,
    a: int,
    b: int,
    markup: tuple[str, bool, str],
    chosen: dict[int, int],
    zeros: int,
    breaks: dict[int, str],
    detour: int,
    backslashes: set[int],
) -> None:
    template, backslash, holds = markup
    link = f"{url}{sha}/{path}{fragment.format(a=a, b=b)}"
    plain = {
        citation.file
        for citation in in_repository_line_citations({"note": template.format(link)})
    }
    assert plain
    # The link may go through a segment holding whitespace that a ``..``
    # then removes.
    if detour >= 0:
        link = f"{url}{sha}/{holds[detour % len(holds)]}x/../{path}"
        link += fragment.format(a=a, b=b)
    chosen = {index % len(link): kind for index, kind in chosen.items()}
    # In HTML a slash after the authority may be written as a backslash,
    # which the URL Standard reads as a slash, with a reference after it.
    slashes = [
        index
        for index, char in enumerate(link)
        if char == "/" and index > link.find("//") + 1
    ]
    if not backslash:
        for index in {slashes[choice % len(slashes)] for choice in backslashes}:
            link = link[:index] + "\\" + link[index + 1 :]
            chosen.setdefault(index + 1, 1)
    # A URL in an HTML attribute may break anywhere: the URL Standard deletes
    # its tabs and line breaks. A Markdown destination cannot hold one, and an
    # unquoted value only one a reference decodes to.
    breaks = {
        index % len(link): spelled
        for index, spelled in breaks.items()
        if spelled.startswith("&") or set(spelled) <= set(holds)
    }
    escaped = template.format(
        _escape_markup(link, chosen, backslash, zeros, None if backslash else breaks)
    )
    event(f"escapes: {min(len(chosen), 3)}")
    event(f"breaks: {0 if backslash else min(len(breaks), 2)}")
    event(f"detour: {'none' if detour < 0 else repr(holds[detour % len(holds)])}")
    event(f"backslashes: {0 if backslash else len(backslashes)}")
    assert plain <= {
        citation.file for citation in in_repository_line_citations({"note": escaped})
    }


def test_a_long_string_is_scanned_in_linear_time() -> None:
    # Each of these once took time quadratic in its length: a run of path
    # characters was rescanned from each of its characters, and a user was
    # looked for before a GitHub host from each character of a run that can
    # be in one. At these lengths the quadratic scans took seconds each.
    started = time.process_time()
    for text in (
        ":" * 60_000,
        "//" * 30_000,
        "a" * 60_000,
        "github.com/" * 6_000,
        "https://a/" + "%74" * 20_000,
    ):
        line_citations(text)
        unrecognised_line_locators(text)
    # So did a string holding thousands of citations: each one's clause,
    # commits and commit-like tokens were found again from the whole string.
    for text in (
        "x.py:1 " * 8_000,
        "x.py:1 at a1b2c3d " * 4_000,
        "a1b2c3d; " + "x.py:1; " * 8_000,
        "x.py:1 y.py foo L1 " * 4_000,
    ):
        misspelled_commit_pins(text)
        unpinned_line_citations(text)
        unrecognised_line_locators(text)
    assert time.process_time() - started < 2
    # Each of these took 4 to 28 seconds: a list of line numbers that failed
    # was retried once for each way to share its whitespace out (exponential
    # in its length), a failed locator once for each way to share a run of
    # spaces or quotes, a failed range once for each way to share the spaces
    # after its dash, a reverse locator was tried from each number of a
    # failed list, and a long host label was encoded in Punycode (quadratic).
    # The exponential ones come first, so a regression fails before the
    # slower ones run.
    for text in (
        "L1 and " * 24,
        "lines " + "1  and  " * 8 + "in x.py",
        "lines " + "1 and     " * 10 + "in x.py",
        "lines " + "1, " * 24 + "in x.py",
        "x.py" + " " * 1_024 + "x",
        "x.py" + "'" * 16_000 + "L1x",
        "line " + " " * 32_768 + "1x in x.py",
        "lines L1-" + " " * 32_768 + "2x in x.py",
        "x.py line" + " " * 65_536 + "1x",
        "lines L1 to" + " " * 32_768 + "2x in x.py",
        "L1 and " * 8_000,
        "https://" + "".join(map(chr, range(0x4000, 0x4000 + 8_000))) + "/a.py#L12",
    ):
        line_citations(text)
        unrecognised_line_locators(text)
        assert time.process_time() - started < 2, text[:24]
    # Since round 9: ranges that each end where the next starts, a list read
    # again from its last number, and runs of what a markup escape starts with.
    for text in (
        "L1-" * 16_000,
        "L1-L2-L3" + ", 4" * 8_000 + " x",
        "&#" + "0" * 100_000 + "35;",
        "&#x" * 30_000,
        "\\" * 100_000,
        "&a" * 50_000,
    ):
        line_citations(text)
        unrecognised_line_locators(text)
        assert time.process_time() - started < 2, text[:24]
    # Since round 10, in 2 s of their own: runs of what an attribute's value,
    # a destination in angle brackets, a long legacy-looking name and a URL
    # read through Unicode spaces start with.
    started = time.process_time()
    for text in (
        '="https:' * 20_000,
        "='" + "x" * 100_000,
        "](<https:" * 20_000,
        "](<https:" + "\\x" * 50_000,
        ("&amp" + "x" * 28) * 3_000,
        "https:" + "\u00a0" * 100_000,
        "https://a\u00a0" * 20_000,
    ):
        line_citations(text)
        unrecognised_line_locators(text)
        assert time.process_time() - started < 2, text[:24]
    # Since round 11, in 2 s of their own: runs of what a dot segment, a URL
    # read through its query and a destination after whitespace start with.
    started = time.process_time()
    for text in (
        "/.." * 30_000,
        "=%2e." * 20_000,
        "\\..\\" * 25_000,
        "https://a?" + " " * 100_000 + "#",
        "https:" * 20_000 + " ?#",
        "https://a?b #" * 10_000,
        "?#https://a#" + "." * 100_000,
        "](" + " " * 100_000 + "\n<",
        "](\t \t<https:" * 15_000,
    ):
        line_citations(text)
        unrecognised_line_locators(text)
        dot_segment_line_citations(text)
        assert time.process_time() - started < 2, text[:24]
    # Since round 12, in 2 s of their own: runs of what this repository's path,
    # a line fragment, a run of slashes or ``.`` segments, a dot segment beside
    # a ``%2f``, and the run of path characters before a name start with.
    started = time.process_time()
    for text in (
        "policyengine/microcosm" * 2_500 + "#L1",
        "PolicyEngine/microcosm/#L1" * 2_000,
        "#L1" * 20_000,
        "/./" * 20_000 + "#L1",
        "/" * 60_000 + "#L1",
        "%2f..%2f" * 8_000,
        "=..%2f" * 10_000,
        "a=" * 30_000 + "x.py:1",
        "=x.py:1" * 8_000,
        "lines 1 of " + "a=" * 30_000,
    ):
        line_citations(text)
        in_repository_line_citations(text)
        dot_segment_line_citations(text)
        line_links_of_this_repository(text)
        assert time.process_time() - started < 2, text[:24]


#: Pieces of text around source names and line locators.
_LOCATOR_PIECES = (
    "x",
    "a/b",
    "tools",
    ".py",
    ".json",
    ".",
    "/",
    "-",
    "_",
    ":",
    "12",
    "L3",
    " ",
    "lines ",
    "l.",
    "https:",
    "//",
    "'",
    "`",
    ", ",
    "of ",
    "#",
    "@",
    ";",
    "\n",
    "at ",
    "(",
    "—",
    "x.py:12",
    "a/b.json#L3",
    "x.py foo L9",
    "x.py, lines 4-5",
    "lines 4-5 of y.py",
    "L1 and L2 in a/b.py",
    "L1, ",
    " and ",
    "L1-L2",
    "-3",
    "-L4",
    " to L2",
    " through 3",
    "L1-L2-3 of x.py",
)


@settings(max_examples=500, deadline=None)
@given(text=st.lists(st.sampled_from(_LOCATOR_PIECES), max_size=40).map("".join))
@example(text="x.py:12/y.py:13")
@example(text="xhttps://a/b.py:12")
@example(text="L1 and L2 x L3 of y.py")
@example(text="lines 1, L2 of x.py")
@example(text="L1 and L2 and L3 in L4 of x.py")
@example(text="L1-L2-3 of tools/ci_test_plan.py")
@example(text="L1 to L2 through 3 of x.py")
@example(text="L1-L2-L3 of x.py")
@example(text="L1-L2, L3-L4-5 of x.py")
def test_source_names_are_found_where_finditer_finds_them(text: str) -> None:
    for name, pattern in (
        ("forward", citations._FORWARD_LOCATOR),
        ("reverse", citations._REVERSE_LOCATOR),
        ("near", citations._LOCATOR_NEAR_A_SOURCE),
    ):
        found = [
            (match.span(), match["file"])
            for match in citations._finditer(pattern, text)
        ]
        assert found == [
            (match.span(), match["file"]) for match in pattern.finditer(text)
        ]
        event(f"{name} matches: {min(len(found), 3)}")


#: The locator patterns as ``ea912094`` wrote them, before their whitespace and
#: quotes were read possessively: the reference the current ones must agree
#: with.
_REFERENCE_QUOTE = r"[`'\"‘’“”]*"
_REFERENCE_NUMBER = r"L?\s*\d+(?!\w)"
_REFERENCE_SPAN = (
    rf"{_REFERENCE_NUMBER}(?:\s*(?:[-–—]|(?i:to|through))\s*{_REFERENCE_NUMBER})?"
)
_REFERENCE_NUMBERS = (
    rf"{_REFERENCE_SPAN}(?:\s*(?:,|,?\s*(?i:and))\s*{_REFERENCE_SPAN})*"
)
_REFERENCE_LABEL = r"(?i:lines?\b\s+|ll?\.\s*)"
_REFERENCE_FORWARD = re.compile(
    citations._SOURCE
    + _REFERENCE_QUOTE
    + rf"(?::(?={_REFERENCE_QUOTE}L?\d)"
    + rf"|\s*[:：]\s*{_REFERENCE_QUOTE}{_REFERENCE_LABEL}"
    + rf"|\s*@\s*{_REFERENCE_QUOTE}(?:{_REFERENCE_LABEL}|(?=L\d))"
    + rf"|#(?={_REFERENCE_QUOTE}L\d)"
    + r"|\s*(?:[,;—–]|\s-(?=\s))?\s*[\[(]?\s*(?:(?i:at|on)\s+)?"
    + rf"{_REFERENCE_QUOTE}(?:{_REFERENCE_LABEL}|(?=L\d)))"
    + _REFERENCE_QUOTE
    + _REFERENCE_NUMBERS
)
_REFERENCE_REVERSE = re.compile(
    rf"\b(?:{_REFERENCE_LABEL}|(?=L\d)){_REFERENCE_QUOTE}{_REFERENCE_NUMBERS}"
    rf"{_REFERENCE_QUOTE}\s+(?i:of|in)\s+{_REFERENCE_QUOTE}" + citations._SOURCE
)
#: Pieces of line locators, heavy in what reading possessively changed: runs
#: of whitespace and quotes, separators, ranges and labels. The reference
#: backtracks exponentially on a long list that fails, so a text holds at most
#: 24 pieces.
_LIST_PIECES = (
    "x.py",
    "a/b.json",
    ":",
    "：",
    "#",
    "@",
    " ",
    "  ",
    "\t",
    "\n",
    ",",
    ";",
    "—",
    "–",
    "-",
    " - ",
    "(",
    "[",
    "at ",
    "on ",
    "line ",
    "Lines",
    "l.",
    "ll. ",
    "L",
    "L1",
    "12",
    " to ",
    "through",
    " and ",
    "AND",
    ", and ",
    " of ",
    " in ",
    "'",
    "`",
    "“",
    "x",
    "1x",
    "x.py:12",
    "x.py L3",
    "x.py, lines 1-2",
    "x.py @ L5",
    "x.py (ll. 6–7)",
    "lines 4 of y.py",
    "L1 and L2 in a/b.py",
    "L1-L2",
    "-3",
    "-L4",
    "L1-L2-3 of x.py",
)


@settings(max_examples=1000, deadline=None)
@given(text=st.lists(st.sampled_from(_LIST_PIECES), max_size=24).map("".join))
@example(text="x.py - L12")
@example(text="x.py- L12")
@example(text="x.py -L12")
@example(text="lines  '  12 ,  and  13 ' of x.py")
@example(text="x.py ''L1 and L2")
@example(text="x.py: lines 1 through  2, AND 3")
@example(text="L1-L2-3 of x.py")
@example(text="L1 to L2 through 3 of x.py")
def test_the_locators_match_as_before_their_whitespace_was_read_possessively(
    text: str,
) -> None:
    for name, current, reference in (
        ("forward", citations._FORWARD_LOCATOR, _REFERENCE_FORWARD),
        ("reverse", citations._REVERSE_LOCATOR, _REFERENCE_REVERSE),
    ):
        found = [
            (match.span(), match.span("file"))
            for match in citations._finditer(current, text)
        ]
        assert found == [
            (match.span(), match.span("file")) for match in reference.finditer(text)
        ]
        event(f"{name} matches: {min(len(found), 3)}")


def _clause_from_the_whole_string(citation: LineCitation) -> tuple[int, int]:
    first, last = 0, len(citation.text)
    for clause_break in citations._CLAUSE_BREAK.finditer(citation.text):
        if clause_break.end() <= citation.start:
            first = clause_break.end()
        elif clause_break.start() >= citation.end:
            last = clause_break.start()
            break
    return first, last


def _binding_from_the_whole_string(citation: LineCitation) -> re.Match[str] | None:
    commits = list(citations.INLINE_COMMIT.finditer(citation.text))
    start, end = _clause_from_the_whole_string(citation)
    in_clause = [match for match in commits if start <= match.start() < end]
    if in_clause:
        return min(
            in_clause,
            key=lambda match: max(
                match.start() - citation.end, citation.start - match.end()
            ),
        )
    if citations._MOVING_REF.search(citation.text, start, end):
        return None
    before = [match for match in commits if match.start() < start]
    return before[-1] if before else None


def _misspelled_pins_from_the_whole_string(text: str) -> list[tuple[LineCitation, str]]:
    found: list[tuple[LineCitation, str]] = []
    for citation in line_citations(text):
        pinned = {
            match.start("sha")
            for match in citations.INLINE_COMMIT.finditer(citation.text)
        }
        start, end = _clause_from_the_whole_string(citation)
        binding = _binding_from_the_whole_string(citation)
        if binding is not None and binding.end() < start:
            start = binding.end()
        found.extend(
            (citation, token.group(0))
            for token in citations._COMMIT_LIKE.finditer(citation.text, start, end)
            if token.start() not in pinned
        )
    return found


def _unrecognised_from_the_whole_string(text: str) -> list[LineCitation]:
    found: list[LineCitation] = []
    for readable in citations._readable_strings(text):
        matches = citations._citation_matches(readable)
        files = {match.start("file") for match in matches}
        spans = [(match.start(), match.end()) for match in matches]
        for near in citations._LOCATOR_NEAR_A_SOURCE.finditer(readable):
            if near.start("file") in files or any(
                start <= near.end() < end for start, end in spans
            ):
                continue
            found.append(LineCitation(near["file"], readable, near.start(), near.end()))
    return found


#: Pieces of text around citations: pins in each spelling, commit-like tokens
#: in none, moving refs and clause breaks.
_BINDING_PIECES = (
    "x.py:12",
    "tools/x.py L3",
    "lines 4-5 of y.json",
    "x.py foo L9",
    "x.py, lines 1-2",
    "x.py; lines 6-7",
    " at a1b2c3d",
    " commit 1a2b3c4d",
    "@b2c3d4e",
    "/blob/c3d4e5f/",
    "a1b2c3d",
    " 9f8e7d6c",
    " on main",
    "@HEAD",
    "/blob/main/",
    "https://github.com/o/r/blob/d4e5f6a/z.py#L7",
    "https://github.com/o/r/blob/main/z.py#L7",
    "; ",
    ";",
    ". See ",
    "! Then ",
    " ",
    ", ",
    " and ",
    "(",
    "'",
)


@settings(max_examples=500, deadline=None)
@given(text=st.lists(st.sampled_from(_BINDING_PIECES), max_size=30).map("".join))
@example(text="at a1b2c3d x.py:12 at b2c3d4e")
@example(text=";x.py:12 at a1b2c3d")
@example(text="https://github.com/o/r/blob/d4e5f6a/z.py#L7 at a1b2c3d")
@example(text="a1b2c3d; x.py:12; 9f8e7d6c y.py:3 on main")
@example(text="x.py:1 y.py foo L1 x.py, lines 1-2 z.py foo L2")
@example(text="at a1b2c3d; on main;x.py:12")
@example(text="9f8e7d6c;x.py:12")
def test_bindings_are_found_as_a_scan_of_the_whole_string_finds_them(
    text: str,
) -> None:
    found = line_citations(text)
    assert [citing_commit(citation) for citation in found] == [
        None if binding is None else binding["sha"]
        for binding in map(_binding_from_the_whole_string, found)
    ]
    assert misspelled_commit_pins(text) == _misspelled_pins_from_the_whole_string(text)
    assert [
        (citation, citation.start, citation.end)
        for citation in unrecognised_line_locators(text)
    ] == [
        (citation, citation.start, citation.end)
        for citation in _unrecognised_from_the_whole_string(text)
    ]


#: URLs and how they are read: the WHATWG URL Standard's reading (bun 1.3's
#: ``URL``, against an ``https`` page for a network-path reference) without
#: its query, a network-path reference kept as one. Where the standard refuses
#: a URL it is read as written.
_WHATWG_READINGS = (
    ("https://%67ithub.com/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("//github.com:443/o/r/blob/a1b2c3d/x.py#L12", "//github.com"),
    ("\\\\user@GitHub.com\\o\\r\\blob\\a1b2c3d\\x.py#L12", "//github.com"),
    ("https:github.com/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("HTTP:///github.com/o/r/blob/a1b2c3d/x.py?plain=1#L12", "http://github.com"),
    ("https://ｇithub．com/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("https://github\u3002com/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("https://github\uff61com/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("//github%E3%80%82com/o/r/blob/a1b2c3d/x.py#L12", "//github.com"),
    ("https://%EF%BD%87ithub.com/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("https://gi%C2%ADthub.com/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("https://github.c\u2092m/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("https://git\u210cub.com/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("https://a@b@github.com:/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("https://GITHUB.COM:0443/o/r/blob/a1b2c3d/x.py#L12", "https://github.com"),
    ("https://Ex%61mple.ORG:8443/o/r/blob/a1b2c3d/x.py#L12", "https://example.org"),
    ("https://[::1]:8080/o/r/blob/a1b2c3d/x.py#L12", "https://[::1]"),
    ("https://%2567ithub.com/o/r/blob/a1b2c3d/x.py#L12", None),
    ("https://user%40github.com/o/r/blob/a1b2c3d/x.py#L12", None),
    ("https://github.com%2F/o/r/blob/a1b2c3d/x.py#L12", None),
    ("https://gi%FFthub.com/o/r/blob/a1b2c3d/x.py#L12", None),
    ("https://github.com:44x/o/r/blob/a1b2c3d/x.py#L12", None),
    ("https://github.com:65536/o/r/blob/a1b2c3d/x.py#L12", None),
)


@pytest.mark.parametrize(("written", "authority"), _WHATWG_READINGS)
def test_a_url_is_read_as_the_url_standard_reads_it(
    written: str, authority: str | None
) -> None:
    reading = written if authority is None else f"{authority}/o/r/blob/a1b2c3d/x.py#L12"
    # The URL Standard's reading comes first. To Markdown ``\\`` is an escaped
    # ``\``, so a decoded reading may follow.
    assert next(citations._readable_strings(written)) == reading
    assert [
        citation.text
        for citation in line_citations({"note": written})
        if citation.text == reading
    ] == [reading]


def test_a_trailing_dot_is_dropped_from_a_host() -> None:
    # The standard keeps it, but DNS ignores it and GitHub redirects
    # ``github.com.`` to ``github.com`` (HTTP 301 on 2026-10-27).
    for stop in (".", *_FULL_STOPS):
        written = f"https://github.com{stop}/o/r/blob/a1b2c3d/x.py#L12"
        assert [citation.text for citation in line_citations({"note": written})] == [
            "https://github.com/o/r/blob/a1b2c3d/x.py#L12"
        ], written


def test_a_label_longer_than_dns_allows_is_left_unencoded() -> None:
    # Up to 63 characters a label is encoded as the standard encodes it (bun
    # 1.3); a longer one names no host either way, so it is left as mapped.
    for length, reading in (
        (63, "https://xn--td" + "a" * 63 + ".com"),
        (64, "https://" + "ü" * 64 + ".com"),
    ):
        written = f"https://{'Ü' * length}.com/o/r/blob/a1b2c3d/x.py#L12"
        assert [citation.text for citation in line_citations({"note": written})] == [
            f"{reading}/o/r/blob/a1b2c3d/x.py#L12"
        ], length


def test_a_break_inside_a_citation_does_not_end_its_clause() -> None:
    for text in ("x.py; lines 12-19 at a1b2c3d", "x.py; lines 12-19, commit a1b2c3d"):
        assert [citing_commit(found) for found in line_citations(text)] == [
            "a1b2c3d"
        ], text


def test_a_long_ref_is_searched_only_as_deep_as_the_deepest_path_here(
    tmp_path: Path,
) -> None:
    deepest = "tools/lanes/emtr/check.py"
    (tmp_path / deepest).parent.mkdir(parents=True)
    (tmp_path / deepest).write_text("", encoding="utf-8")
    root = _CountingPath(tmp_path)
    url = "https://github.com/PolicyEngine/microcosm/blob/" + "x/" * 500
    assert resolves_in_repository(url + deepest, root)
    _CountingPath.checks = 0
    assert not resolves_in_repository(url + "tools/lanes/emtr/missing.py", root)
    # One check per suffix of at most four segments, not one per ref segment.
    assert _CountingPath.checks == 4


@settings(max_examples=200, deadline=None)
@given(
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    prefix=_PROSE,
    tail=_PROSE,
)
def test_named_commits_returns_the_bare_sha(
    pin: str, sha: str, prefix: str, tail: str
) -> None:
    assert named_commits(f"{prefix} {pin.format(sha=sha)} {tail}") == [sha]


# ---------------------------------------------------------------------------
# Differential: citations against the records and code they name
# ---------------------------------------------------------------------------


def _citing_commits(payload: object) -> set[str | None]:
    citations = line_citations(payload)
    assert citations
    return {citing_commit(citation) for citation in citations}


def test_uk_data_parity_citations_bind_the_incumbent_commit() -> None:
    parity = package_payloads("uk")["uk_data_target_parity.json"]
    incumbent = parity["reference"]["incumbent_commit"]
    assert _citing_commits(parity) == {incumbent}


def test_uk_population_citations_bind_the_registry_pinned_ref() -> None:
    contract = package_payloads("uk")["uk_population_targets.json"]
    pinned_ref = contract["registry_parity"]["pinned_ref"]
    commits = _citing_commits(contract)
    assert pinned_ref in commits
    assert all(
        commit is not None and pinned_ref.startswith(commit) for commit in commits
    )


#: The us-data commit the archived PUF and CPS pipelines are cited at.
_ARCHIVED_US_DATA = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"


@pytest.mark.parametrize("resource", ["source_stages.json", "spec/sources.yaml"])
def test_the_puf_note_binds_its_disclosure_citations_to_the_archived_commit(
    resource: str,
) -> None:
    bound = {
        citation.text[citation.start : citation.end]: citing_commit(citation)
        for citation in line_citations(package_payloads("us")[resource])
    }
    assert bound["datasets/puf/puf.py lines 1378-1398"] == _ARCHIVED_US_DATA
    assert (
        bound["datasets/puf/disaggregate_puf.py lines 55-60 and 85-123"]
        == _ARCHIVED_US_DATA
    )


def test_the_esi_origin_binds_each_citation_to_its_own_commit() -> None:
    manifest = package_payloads("us")["target_parity_manifest.json"]
    origin = manifest["families"]["bea_nipa.private_group_health_insurance"]["fence"][
        "origin"
    ]
    assert {
        (citation.file, citing_commit(citation)) for citation in line_citations(origin)
    } == {("loss.py", "2e044571"), ("datasets/cps/cps.py", _ARCHIVED_US_DATA[:8])}


def _known_gaps() -> dict[str, dict[str, object]]:
    return package_payloads("us")["ecps_parity_known_gaps.json"]["known_gaps"]


#: Each US known gap whose build contract names Census person columns that
#: the reviewed restore leaves out.
_UNRESTORED_GAP_COLUMNS = {
    "financial_assistance": ("FIN_VAL", "FIN_YN", "I_FINVAL"),
    "is_unmarried_partner_of_household_head": ("PERRP", "PECOHAB", "A_FAMREL"),
    "survivor_benefits": ("SRVS_VAL",),
}


@pytest.mark.parametrize(("gap", "columns"), sorted(_UNRESTORED_GAP_COLUMNS.items()))
def test_a_known_gap_names_columns_the_census_restore_leaves_out(
    gap: str, columns: tuple[str, ...]
) -> None:
    contract = _known_gaps()[gap]["evidence"]["hermetic_build_contract"]
    assert "microcosm.build.us_runtime.asec_pool._prepare_year_input" in contract
    for column in columns:
        assert column in contract
        assert column not in ASEC_CENSUS_PERSON_COLUMN_NAMES


def test_the_restore_runs_before_the_relationship_fallback() -> None:
    contract = _known_gaps()["is_unmarried_partner_of_household_head"]["evidence"][
        "hermetic_build_contract"
    ]
    assert "restores the 2022 and 2023 inputs' missing A_EXPRRP" in contract
    (exprrp,) = [
        column for column in ASEC_CENSUS_PERSON_COLUMNS if column.name == "A_EXPRRP"
    ]
    assert exprrp.domain is not None and 13 in exprrp.domain
    source = inspect.getsource(asec_pool._prepare_year_input)
    restore = source.index("restore_asec_census_person_columns(")
    assert restore < source.index("_with_relationship_recode(person)")
    assert "Build J predates #720" in contract
    summary = json.loads(
        (REPOSITORY_ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    assert {
        source["year"]: source["relationship_recode_source"]
        for source in summary["base_source"]["metadata"]["sources"]
    } == {
        2022: "derived:line_spouse_parent",
        2023: "derived:line_spouse_parent",
        2024: "source:A_EXPRRP",
    }


@settings(max_examples=100, deadline=None)
@given(
    households=st.lists(
        st.lists(
            st.tuples(
                st.integers(0, 3),
                st.integers(1, 2),
                st.integers(0, 3),
                st.integers(0, 3),
            ),
            min_size=1,
            max_size=4,
        ),
        min_size=1,
        max_size=4,
    )
)
def test_the_relationship_fallback_never_emits_code_13(
    households: list[list[tuple[int, int, int, int]]],
) -> None:
    rows = [
        {
            "PH_SEQ": household,
            "A_LINENO": line,
            "A_SPOUSE": spouse,
            "A_SEX": sex,
            "PEPAR1": parent1,
            "PEPAR2": parent2,
        }
        for household, members in enumerate(households, start=1)
        for line, (spouse, sex, parent1, parent2) in enumerate(members, start=1)
    ]
    person, source = asec_pool._with_relationship_recode(pd.DataFrame(rows))
    assert source == "derived:line_spouse_parent"
    assert 13 not in set(person["A_EXPRRP"])
    with_source = person.assign(A_EXPRRP=13)
    kept, kept_source = asec_pool._with_relationship_recode(with_source)
    assert kept is with_source
    assert kept_source == "source:A_EXPRRP"


def test_the_known_gaps_name_functions_the_puf_support_tool_defines() -> None:
    tool = REPOSITORY_ROOT / "tools/build_us_puf_support_base.py"
    module = ast.parse(tool.read_text(encoding="utf-8"))
    defined = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    named = ("_parse_args", "_pooled_asec_sources_from_args", "_asec_sources_from_args")
    contract = _known_gaps()["financial_assistance"]["evidence"][
        "hermetic_build_contract"
    ]
    assert all(name in contract for name in named)
    assert set(named) <= set(defined)
    assert '"--asec-h5"' in ast.get_source_segment(
        tool.read_text(encoding="utf-8"), defined["_parse_args"]
    )
