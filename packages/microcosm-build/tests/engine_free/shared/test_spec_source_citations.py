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
  from its shard, or from a first-level subpackage of its shard. A bare file
  name cannot be resolved and falls to the next rule;
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
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import asec_pool
from microcosm.build.us_runtime.asec_census_person_columns import (
    ASEC_CENSUS_PERSON_COLUMN_NAMES,
    ASEC_CENSUS_PERSON_COLUMNS,
)
from test_support.microcosm_build.spec_source_citations import (
    COUNTRY_PACKAGE_ROOT,
    MICROCOSM_SHARDS,
    REPOSITORY_ROOT,
    LineCitation,
    citing_commit,
    commit_pinned_sources,
    country_packages,
    dotted_microcosm_symbols,
    in_repository_line_citations,
    line_citations,
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
#: A URL up to its ref: the scheme, the host, and the owner, repository and
#: route.
_URL_PARTS = re.compile(r"(?P<scheme>(?:https?://)?)(?P<host>[^/]+)(?P<route>/.*)")
#: Detours in a URL path that GitHub resolves away.
_DOT_DETOURS = ("", "./", "%2e/", "docs/../", "x/%2E%2e/", "a/b/../../", "tools/./../")
#: A URL query, without its ``?``: anything but a fragment, a quote or a space.
_URL_QUERY = st.text(
    alphabet=string.ascii_letters + string.digits + "=&.;:,%+-_~?/", max_size=120
)


def _plant(text: str, depth: int, as_key: bool) -> object:
    planted: object = {text: "cited"} if as_key else text
    for level in range(depth):
        planted = {"level": level, "note": [planted]}
    return planted


def _escape(text: str, chosen: set[int], upper: bool, keep: str) -> str:
    """``text`` with its characters at ``chosen`` written as ``%`` escapes."""

    return "".join(
        (f"%{ord(char):02X}" if upper else f"%{ord(char):02x}")
        if index in chosen and char not in keep
        else char
        for index, char in enumerate(text)
    )


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
    assert line_citations(document) == [LineCitation(file, text)]
    assert unrecognised_line_locators(document) == []
    if here is None:
        assert LineCitation(file, text) in unpinned_line_citations(document)
    else:
        assert in_repository_line_citations(document) == [LineCitation(file, text)]


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
        "/tools/build_us_puf_support_base.py",
        "https://github.com/PolicyEngine/upstream-data/blob/main/tools/"
        "build_us_puf_support_base.py",
        "https://github.com/PolicyEngine/microcosm/blob/main/no/such/file.py",
        "microcosm/graph",
    ):
        assert not resolves_in_repository(file), file
    assert (REPOSITORY_ROOT / _IN_REPOSITORY_FILES[0]).is_file()
    assert COUNTRY_PACKAGE_ROOT.is_dir()


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
    authority=st.sampled_from(
        ("{host}", "{host}:443", "user@{host}", "u:p@{host}:443")
    ),
    query=st.one_of(st.just(""), _URL_QUERY.map("?".__add__)),
    fragment=st.sampled_from(("#L{a}", "#L{a}-L{b}")),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_SHA,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
)
def test_a_url_of_this_repository_is_refused_however_github_lets_it_be_written(
    url: str,
    ref: str,
    detour: str,
    path: str,
    chosen: set[int],
    upper: bool,
    case: Callable[[str], str],
    authority: str,
    query: str,
    fragment: str,
    pin: str,
    sha: str,
    a: int,
    b: int,
) -> None:
    parts = _URL_PARTS.fullmatch(url)
    assert parts is not None
    scheme, host, route = parts["scheme"], parts["host"], parts["route"]
    # A user or a port needs a scheme; GitHub 404s a ``%2F`` before the ref.
    written = (
        case(scheme)
        + (authority.format(host=case(host)) if scheme else case(host))
        + _escape(route, chosen, upper, keep="/%")
        + _escape(
            f"{ref}/{detour}{path}",
            {index - len(route) for index in chosen},
            upper,
            keep="%",
        )
    )
    text = f"{written}{query}{fragment.format(a=a, b=b)} {pin.format(sha=sha)}"
    found = in_repository_line_citations({"note": text})
    assert [citation.file for citation in found] == [f"{url}{ref}/{path}"]
    # A citation's text is already read: reading it again changes nothing.
    assert in_repository_line_citations({"note": found[0].text}) == found


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
    assert [
        (citation.file, citing_commit(citation))
        for citation in line_citations(document)
    ] == [(url, sha if pinned else None)]
    assert len(unpinned_line_citations(document)) == (0 if pinned else 1)


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
    gitlab = "https://gitlab.com/o/upstream-data/-/blob/a1b2c3d/x.py"
    text = f"{gitlab}?ref_type=heads&view=plain&page=2 some_function, lines 12-19"
    assert unrecognised_line_locators({"note": text}) == [
        LineCitation(gitlab, f"{gitlab} some_function, lines 12-19")
    ]


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
