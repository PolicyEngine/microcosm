"""Country spec packages cite sources by symbol or at a commit, never drifting.

A line number names text only at a fixed commit, so a citation such as
``asec_pool.py lines 61-75`` silently points at other code after an unrelated
edit. The validators live in ``test_support.microcosm_build.
spec_source_citations``; these tests state the invariants over every country
spec package (every build directory with a ``country_package.json``) and check
each refusal branch with Hypothesis, so a vacuous pass on today's data cannot
hide a broken check:

- no resource cites a line of a file in this repository, pinned or not
  (``in_repository_line_citations``): this repository's sources are cited by
  symbol;
- no resource cites a line of any other source unless the citation's own
  string names a commit, or the resource pins that file in a
  ``path``/``commit`` mapping (``unpinned_line_citations``);
- every dotted ``microcosm.*`` name in a resource resolves through import
  plus getattr (``resolve_dotted_symbol``).

Differential checks compare two records of one fact: the UK citations' commits
with the incumbent commit each resource declares, and the US known gaps'
symbol citations with the code they name (the Census person-column restore
list, the relationship fallback, and the PUF support tool's functions).
"""

from __future__ import annotations

import ast
import inspect

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
    REPOSITORY_ROOT,
    LineCitation,
    commit_pinned_sources,
    country_packages,
    dotted_microcosm_symbols,
    in_repository_line_citations,
    line_citations,
    named_commits,
    names_a_commit,
    package_payloads,
    resolve_dotted_symbol,
    resolves_in_repository,
    unpinned_line_citations,
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


@pytest.mark.parametrize(
    ("country", "resource", "cited"),
    [
        ("nz", "as_rate_bridge.json", {"run.py"}),
        ("uk", "uk_data_target_parity.json", {"obr.py", "voa_council_tax.py"}),
        ("uk", "uk_population_targets.json", {"income.py", "hmrc_salary_sacrifice.py"}),
        ("us", "source_stages.json", {"cps.py", "census_cps.py", "sipp.py"}),
        ("us", "spec/sources.yaml", {"cps.py", "census_cps.py", "sipp.py"}),
        ("us", "ecps_parity_known_gaps.json", {"cps.py", "census_cps.py"}),
        ("us", "target_parity_manifest.json", {"loss.py"}),
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
        "targets/sources/",
        "nz-lane/emtr_reproduction/",
    )
)
#: Line-locator spellings: {file} is the cited source, {a} and {b} lines.
_LINE_LOCATORS = (
    "{file}:{a}",
    "{file}:{a}-{b}",
    "{file}:{a},{b}",
    "{file}#L{a}",
    "{file}#L{a}-L{b}",
    "{file} L{a}",
    "{file} line {a}",
    "{file} lines {a}-{b}",
    "{file}, lines {a}-{b}",
    "{file} (lines {a}-{b})",
    "Lines {a}-{b} of {file}",
    "line {a} in {file}",
)
#: Symbol-citation spellings, which carry no line number.
_SYMBOL_LOCATORS = (
    "{file} {symbol}",
    '{file} _pointer("{symbol}")',
    '{file}, the "# --- Relationships" block',
    "{file}: {symbol}",
    "{symbol} in {file} at main",
)
#: Inline-commit spellings.
_INLINE_PINS = (
    "commit {sha}",
    "Commit {sha}",
    "at {sha}",
    "At {sha}",
    "uk-data@{sha}",
    "https://github.com/PolicyEngine/archived-data/blob/{sha}/x",
)
#: Prose around a citation. It has no digit, so it can neither extend a
#: planted line number nor name a commit of its own.
_PROSE = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,;()'\"-_",
    max_size=40,
)
_HEX = st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)
_BRANCH_NAMES = ("main", "HEAD", "launch/ga-path-execution-gate", "ba6e7d", "")
#: Files in this checkout, in each path form a citation can use.
_IN_REPOSITORY_FILES = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py",
    "microcosm-frame/src/microcosm/frame/concepts.py",
    "microcosm/graph/executor.py",
    "tools/build_us_puf_support_base.py",
    "experiments/build_j_recert/buildj_base.sh",
    "experiments/build_j_recert/base_j.summary.json",
)


def _plant(text: str, depth: int, as_key: bool) -> object:
    planted: object = {text: "cited"} if as_key else text
    for level in range(depth):
        planted = {"level": level, "note": [planted]}
    return planted


def _is_commit_like(sha: str) -> bool:
    return any(c.isdigit() for c in sha) and any(c.isalpha() for c in sha)


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
    hits = unpinned_line_citations(_plant(text, depth, as_key))
    assert LineCitation(file, text) in hits
    assert not resolves_in_repository(file)


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    directory=_FOREIGN_DIRECTORY,
    pin_directory=_FOREIGN_DIRECTORY,
    locator=st.sampled_from(_LINE_LOCATORS),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    depth=st.integers(0, 3),
    as_key=st.booleans(),
    commit=_HEX,
)
def test_a_line_citation_of_a_file_pinned_at_a_commit_passes(
    stem: str,
    suffix: str,
    directory: str,
    pin_directory: str,
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
            "path": f"{pin_directory}{stem}.{suffix}",
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
    directory=_FOREIGN_DIRECTORY,
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_HEX,
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    prefix=_PROSE,
    middle=_PROSE,
    tail=_PROSE,
    pin_first=st.booleans(),
    depth=st.integers(0, 3),
    as_key=st.booleans(),
)
def test_a_commit_named_in_the_citing_string_pins_it(
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
    tail: str,
    pin_first: bool,
    depth: int,
    as_key: bool,
) -> None:
    assume(_is_commit_like(sha))
    citation = locator.format(file=f"{directory}{stem}.{suffix}", a=a, b=b)
    named = pin.format(sha=sha)
    first, second = (named, citation) if pin_first else (citation, named)
    text = f"{prefix} {first} {middle} {second} {tail}"
    document = _plant(text, depth, as_key)
    assert names_a_commit(text)
    assert line_citations(document)
    assert unpinned_line_citations(document) == []


@settings(max_examples=200, deadline=None)
@given(
    stem=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_HEX,
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
    assume(_is_commit_like(sha))
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
    keyword=st.sampled_from(("commit ", "at ", "At ", "@")),
    not_a_commit=st.one_of(
        st.text(alphabet="0123456789", min_size=7, max_size=40),
        st.text(alphabet="abcdef", min_size=7, max_size=40),
        st.text(alphabet="0123456789abcdef", min_size=1, max_size=6),
        st.text(alphabet="0123456789abcdef", min_size=41, max_size=64),
        st.sampled_from(_BRANCH_NAMES),
    ),
    a=st.integers(1, 99_999),
)
def test_a_decimal_a_word_a_digest_or_a_branch_names_no_commit(
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
    other=_STEM,
    suffix=_SUFFIX,
    locator=st.sampled_from(_LINE_LOCATORS),
    a=st.integers(1, 99_999),
    b=st.integers(1, 99_999),
    commit=_HEX,
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
    assert commit_pinned_sources({"source": pin}).isdisjoint({f"{stem}.{suffix}"})


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
    assert line_citations({"rule": f"{prefix} {citation} {tail}"}) == []


@settings(max_examples=200, deadline=None)
@given(
    file=st.sampled_from(_IN_REPOSITORY_FILES),
    locator=st.sampled_from(_LINE_LOCATORS),
    pin=st.sampled_from(_INLINE_PINS),
    sha=_HEX,
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
    assume(_is_commit_like(sha))
    text = locator.format(file=file, a=a, b=b)
    if pinned_inline:
        text = f"{text} {pin.format(sha=sha)}"
    document: dict[str, object] = {"note": text}
    if pinned_by_path:
        document["source"] = {"path": file, "commit": sha}
    assert resolves_in_repository(file)
    assert LineCitation(file, text) in in_repository_line_citations(document)


def test_in_repository_paths_resolve_from_the_root_packages_or_a_shard() -> None:
    for file in _IN_REPOSITORY_FILES:
        assert resolves_in_repository(file), file
    for file in (
        "concepts.py",
        "archived_data/datasets/cps/cps.py",
        "targets/sources/obr.py",
        "../microcosm/tools/build_us_puf_support_base.py",
        "/tools/build_us_puf_support_base.py",
        "microcosm/graph",
    ):
        assert not resolves_in_repository(file), file
    assert (REPOSITORY_ROOT / _IN_REPOSITORY_FILES[0]).is_file()
    assert COUNTRY_PACKAGE_ROOT.is_dir()


@settings(max_examples=200, deadline=None)
@given(
    pin=st.sampled_from(_INLINE_PINS),
    sha=_HEX,
    prefix=_PROSE,
    tail=_PROSE,
)
def test_named_commits_returns_the_bare_sha(
    pin: str, sha: str, prefix: str, tail: str
) -> None:
    assume(_is_commit_like(sha))
    assert named_commits(f"{prefix} {pin.format(sha=sha)} {tail}") == [sha]


# ---------------------------------------------------------------------------
# Differential: citations against the records and code they name
# ---------------------------------------------------------------------------


def _line_citing_commits(payload: object) -> set[str]:
    texts = {citation.text for citation in line_citations(payload)}
    assert texts
    return {commit for text in texts for commit in named_commits(text)}


def test_uk_data_parity_citations_name_the_incumbent_commit() -> None:
    parity = package_payloads("uk")["uk_data_target_parity.json"]
    incumbent = parity["reference"]["incumbent_commit"]
    commits = _line_citing_commits(parity)
    assert commits == {"8629dbbfe82727278d47ee5ff09fd1da4acefa38"}
    assert all(commit.startswith(incumbent) for commit in commits)


def test_uk_population_citations_name_the_registry_pinned_ref() -> None:
    contract = package_payloads("uk")["uk_population_targets.json"]
    pinned_ref = contract["registry_parity"]["pinned_ref"]
    commits = _line_citing_commits(contract)
    assert pinned_ref in commits
    assert all(pinned_ref.startswith(commit) for commit in commits)


def _known_gaps() -> dict[str, dict[str, object]]:
    return package_payloads("us")["ecps_parity_known_gaps.json"]["known_gaps"]


#: Each US known gap whose build contract names Census person columns that
#: the reviewed restore leaves out.
_UNRESTORED_GAP_COLUMNS = {
    "employer_sponsored_insurance_premiums": (
        "NOW_OWNGRP",
        "NOW_HIPAID",
        "NOW_GRPFTYP",
    ),
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
    for gap in ("employer_sponsored_insurance_premiums", "financial_assistance"):
        contract = _known_gaps()[gap]["evidence"]["hermetic_build_contract"]
        assert all(name in contract for name in named), gap
    assert set(named) <= set(defined)
    assert '"--asec-h5"' in ast.get_source_segment(
        tool.read_text(encoding="utf-8"), defined["_parse_args"]
    )
