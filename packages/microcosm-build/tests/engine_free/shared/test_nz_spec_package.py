"""The New Zealand spec package, graph era: invariants over its resources.

The NZ package is spec data only. These tests state the invariants the graph
build relies on and check them on the committed resources. Each validator
also has Hypothesis cases that break its input and require a refusal, at
least one per refusal branch, so a vacuous pass on today's data cannot hide a
broken check:

- the three reference sets (calibration, pre-calibration, hold-out) are
  separate resources, active rows are cell-pinned, parked rows refuse, and no
  calibration or pre-calibration reference shares a hold-out reference's
  name, Ledger identifier or family, or has a selector that one Ledger fact
  could satisfy together with a hold-out selector (``holdout_leaks``);
- no resource names a rulespec commit other than the two reviewed on
  rulespec-nz main, the rules-binding pin and the commit the committed Axiom
  input surface was generated at (``rulespec_commit_mentions``);
- every crosswalk territorial authority's area shares sum to 1 within 1e-9
  (``crosswalk_errors``), and the S3 modal area refuses ties
  (``modal_area``);
- every scenario names exactly one method-card row and a tier from
  {entitlement, calibration}, and changes exactly one knob from the centre
  (``scenario_errors``);
- no resource cites a line of a repository file, and foreign-file line
  citations require a same-resource commit pin (``unpinned_line_citations``);
- JSON citations refuse index steps into arrays consisting of mappings.
  Instead, ``file.json#/entries[input=target]`` selects exactly one record.
  A selector extends an RFC 6901 token naming an array; ordinary pointer
  steps may follow it, e.g. ``/entries[input=target]/state_binding/columns/0``.
  Comma-separated terms are a conjunction, e.g. ``[input=target,module=nz]``.
  Fields are identifiers and values are nonempty strings; brackets, commas,
  equals signs and whitespace are reserved. Parse the selector before
  decoding RFC 6901 ``~0`` (tilde) and ``~1`` (slash) escapes. No match or
  multiple matches is an error. Indexes into scalar arrays remain valid.

Differential checks compare two records of one fact: the loader's target
references with a direct parse, the rules bindings' module digests with the
committed Axiom input surface, and the export contract's formula-owned list
with the bound variables, and source citations with the code and JSON nodes
they identify.

The line guard catches drift-prone citations that an author (human or model)
would plausibly write in NZ spec prose. It is not a defence against deliberate
obfuscation, such as spelling out line numbers or splitting them across keys.

Recognized line citations (for any extension, or known checkout filenames
such as LICENSE and .gitignore):

- The filename may be plain, backtick-wrapped, single/double/curly-quoted,
  an inline Markdown link's destination or text (including backticks), an
  image link, or reference-link text. A link attributes its locator to the
  destination (without query or fragment) and every filename in its label;
  all must be pinned foreign files for an exemption.
- Forward: file:12, file:L12, file：L12 (full-width colon), file#L12,
  file @ L12, file L12, file, L12, file (L12), file [L12], file line 12,
  file lines L12-L20, file l.12, file ll.12-20, and file at/on line 12.
  Locators may themselves be quoted or backtick-wrapped.
- Reverse: line 12 in file, lines 12-20 of file, ll.12-20 in file,
  and L12-L20 of/in file. Words ignore case; the L prefix is uppercase.
- Numbers may form comma lists or ranges with hyphen, en dash, em dash,
  or "to", with optional spaces and an L on either endpoint.
- HTTP(S) URLs may be bare or Markdown destinations, inside parentheses,
  quotes, angle brackets or HTML hrefs, or in Markdown reference definitions.
  GitHub blob/raw/tree/blame/<ref>/path and raw.githubusercontent.com URLs
  accept #L12, #L12-L20, #L12-20 and #L12C3-L20C9 fragments (also lowercase
  l/c). Query strings before fragments and percent-encoded URL paths work.
  Trailing punctuation and symbols, including Unicode dashes and ellipses,
  cannot hide a complete line fragment. Ununderstood fragments stay scannable.
  PolicyEngine/microcosm URLs at any ref, including commit permalinks, name
  local files. Foreign URLs require a same-resource {path, commit} pin for
  the exact repository-relative path; basename matching is prose-only.
  Paths normalize leading ./; a trailing slash names a directory, not a file.

In-scope examples:
    file:12
    file:L12
    file：L12
    file#L12
    file @ L12
    file L12
    file, L12
    file (L12)
    file [L12]
    file line 12
    file lines L12-L20
    file l.12
    file ll.12-20
    file at line 12
    file on line 12
    line 12 in file
    lines 12-20 of file
    ll.12-20 in file
    L12-L20 of file
    file:12,14-20
    file lines 12–20
    file lines 12—20
    file lines 12 to 20
    `file` lines 12-20
    'file' lines 12-20
    "file" lines 12-20
    ‘file’ lines 12-20
    “file” lines 12-20
    file: `L12`
    file: 'line 12'
    [file](file) lines 12-20
    [`file`](file) L12
    [file][source] lines 12-20
    [the source](file) lines 12-20
    lines 12-20 of [the source](file)
    [upstream.rs](file) L12
    [the source](file#L12)
    [the source](file:L12)
    [file:L12](upstream.rs)
    [the source](file?raw=1) L12
    ![x](file) L5
    LICENSE:12
    .gitignore:12
    [the source](https://github.com/PolicyEngine/microcosm/blob/main/file#L12)
    https://github.com/PolicyEngine/microcosm/raw/main/file#L12-L20
    https://raw.githubusercontent.com/PolicyEngine/microcosm/main/file#L12C3-L20C9
    <https://github.com/PolicyEngine/microcosm/blame/main/file?plain=1#l5>
    https://github.com/PolicyEngine/microcosm/tree/main/file#L5-10: see the field.

Out of scope: bare prose numbers ("line 12" without a file), unknown
extensionless names ("unknown_source:12"), spelled-out numbers ("file line
twelve"), locators split across keys or values ({"file": "line", "n": 12}),
HTML/entity-encoded markup ("file&#58;12"), encoded locator syntax (#%4C5 or
#L%35), repeatedly encoded URL paths, Unicode lookalike locator letters,
zero-width characters inserted into locators,
escaped/nested Markdown labels, link titles, and arbitrary Markdown rendering
or reference-link destination lookup. The guard scans parsed resource strings
for explicit file/line locators.
In plain prose, lowercase l12 is ambiguous with a Python identifier; use
uppercase L12 or an explicit line/l./ll. label. Symbols, heading anchors,
JSON Pointers and keyed selectors remain opaque, including locator-like tokens.
Versioned symbols such as takeup.assign@1 also remain valid.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import re
import subprocess
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cache
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from microcosm.build.chronicle_epoch import (
    is_accepted_consumer_artifact_schema_version,
)
from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_artifact import chronicle_feed_identity
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
    reference_fact_selectors,
)
from microcosm.build.spec_engine import load_yaml12
from microcosm.frame import concepts as frame_concepts
from microcosm.frame.input_closure import INPUT_CLOSURE_FORMAT
from test_support.paths import paths_for

_BUILD = paths_for("microcosm-build")
NZ_ROOT = _BUILD.package / "src/microcosm/build/nz"
NZ_INPUT_SURFACE = (
    paths_for("microcosm-frame").tests / "fixtures/axiom_input_surfaces/nz.json"
)

REFERENCE_FILES = {
    "calibration": "target_references.json",
    "precal": "precal_references.json",
    "holdout": "holdout_references.json",
}
HELD_OUT_FAMILIES = {"accommodation_supplement", "working_for_families"}
FORBIDDEN_VALUE_KEYS = {"value", "values", "observed", "observed_value"}
#: The head of rulespec-nz#117, the unmerged Budget-transport draft that
#: microcosm#821's WFF stage pinned. It must never re-enter the package.
UNMERGED_RULESPEC_COMMIT = "3b663b3e6eb6408351154990be0c4b92d42c92da"
METHOD_CARD_ROW = re.compile(r"MC(?:[1-9]|1[0-5])")
METHOD_CARD_ROWS = tuple(f"MC{number}" for number in range(1, 16))
TIERS = {"entitlement", "calibration"}
SHARE_TOLERANCE = 1e-9
AREAS = {1, 2, 3, 4}
HOLDOUT_LEAK_KINDS = ("name", "ledger identifier", "ledger selector", "family")


def _load(name: str) -> dict[str, Any]:
    return json.loads((NZ_ROOT / name).read_text(encoding="utf-8"))


def _rows(reference_set: str) -> list[dict[str, Any]]:
    return _load(REFERENCE_FILES[reference_set])["target_references"]


def _references(reference_set: str) -> list[LedgerTargetReference]:
    return [LedgerTargetReference(**row) for row in _rows(reference_set)]


def _nested_keys(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def _nested_strings(value: object) -> Iterator[str]:
    """Every string in a payload, mapping keys included, at any depth."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_strings(child)
    elif isinstance(value, str):
        yield value


def _package_payloads() -> dict[str, object]:
    payloads: dict[str, object] = {}
    for path in sorted(NZ_ROOT.rglob("*")):
        if not path.is_file() or not (
            path.suffix == ".json"
            or (path.parent == NZ_ROOT / "spec" and path.suffix == ".yaml")
        ):
            continue
        name = path.relative_to(NZ_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        payloads[name] = (
            json.loads(text)
            if path.suffix == ".json"
            else load_yaml12(text, source=name)
        )
    return payloads


def _bindings() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in _load("axiom_rules_bindings.json")["bindings"]}


def _input_surface() -> dict[str, dict[str, Any]]:
    surface = json.loads(NZ_INPUT_SURFACE.read_text(encoding="utf-8"))
    return {module["path"]: module for module in surface["modules"]}


def _reporting_parameter_paths(value: object, path: str) -> Iterator[str]:
    """Each executable parameter leaf needs its own method citation."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _reporting_parameter_paths(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _reporting_parameter_paths(child, f"{path}/{index}")
    else:
        yield path


def _reporting_citations(value: object) -> Iterator[Mapping[str, Any]]:
    """Include ancillary method citations as well as executable parameters."""
    if isinstance(value, Mapping):
        if "item" in value and "source" in value:
            yield value
        for child in value.values():
            yield from _reporting_citations(child)
    elif isinstance(value, list):
        for child in value:
            yield from _reporting_citations(child)


# ---------------------------------------------------------------------------
# Validators (exercised by Hypothesis below, applied to the package here)
# ---------------------------------------------------------------------------

_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{7,40}(?![0-9a-fA-F])")


def _hex_runs(text: str) -> set[str]:
    return {run.lower() for run in _HEX.findall(text)}


def rulespec_commit_mentions(value: object) -> set[str]:
    """Return every commit (7-40 hex) a payload may attribute to rulespec.

    The scan errs towards reporting: a reviewed commit costs nothing, and an
    unreviewed one is what the scan exists to catch. A hex run of 7-40
    characters (any case) counts as a mention when it is in:

    - a string (a value or a mapping key) that contains "rulespec" in any
      case, wherever in the string the run sits;
    - any key or value, at any depth, below a key that contains "rulespec"
      or in a rulespec record (its ``repository`` or ``source_repo`` contains
      "rulespec" in any case);
    - the value, at any depth, of a key that contains "commit" in any case,
      unless the key's mapping is a foreign record (its repository
      attribution lacks "rulespec") or the key is one of the seven fields
      of a complete Chronicle identity below ``feed_identity``, and neither
      the key nor an enclosing key contains "rulespec".

    A foreign record cancels the context it sits in, except that its commit
    keys still count below a rulespec key. Integers count where hex strings
    count: YAML reads an all-digit short SHA as an integer. A short SHA that
    YAML reads as a float (``1234e56``) is not caught; quote SHAs.
    """

    found: set[str] = set()

    def visit(
        node: object,
        *,
        rulespec_context: bool,
        commit_value: bool,
        feed_identity_record: bool = False,
    ) -> None:
        if isinstance(node, Mapping):
            repositories = [
                repository
                for field in ("repository", "source_repo")
                if isinstance(repository := node.get(field), str)
            ]
            named = bool(repositories)
            rulespec_record = any(
                "rulespec" in repository.lower() for repository in repositories
            )
            artifact_identity_fields: set[str] = set()
            if feed_identity_record and is_accepted_consumer_artifact_schema_version(
                node.get("artifact_schema_version")
            ):
                try:
                    artifact_identity_fields = set(chronicle_feed_identity(node))
                except ValueError:
                    pass
            foreign = named and not rulespec_record
            if rulespec_record:
                commit_value = True
            elif foreign:
                commit_value = False
            for key, child in node.items():
                name = str(key).lower()
                identity_field = key in artifact_identity_fields
                field_commit_value = commit_value and (
                    rulespec_context or not identity_field
                )
                commit_key = "commit" in name and (
                    rulespec_context or (not foreign and not identity_field)
                )
                visit(
                    str(key),
                    rulespec_context=rulespec_context,
                    commit_value=field_commit_value,
                )
                visit(
                    child,
                    rulespec_context=rulespec_context or "rulespec" in name,
                    commit_value=field_commit_value or commit_key or "rulespec" in name,
                    feed_identity_record=name == "feed_identity",
                )
        elif isinstance(node, list):
            for child in node:
                visit(
                    child, rulespec_context=rulespec_context, commit_value=commit_value
                )
        elif isinstance(node, str):
            if commit_value or "rulespec" in node.lower():
                found.update(_hex_runs(node))
        elif isinstance(node, int) and not isinstance(node, bool) and commit_value:
            found.update(_hex_runs(str(node)))

    visit(value, rulespec_context=False, commit_value=False)
    return found


def reviewed_rulespec_commits() -> set[str]:
    """Full rulespec-nz commits reviewed as on its main branch: there are two.

    The rules-binding pin (verified with ``git branch -r --contains`` when it
    was set) and the commit the committed Axiom input surface was generated
    at (an ancestor of the pin).
    """

    pin = _load("axiom_rules_bindings.json")["rulespec"]["commit"]
    surface = json.loads(NZ_INPUT_SURFACE.read_text(encoding="utf-8"))
    return {pin, surface["rulespec"]["commit"]}


def unreviewed_mentions(mentions: set[str], reviewed: set[str]) -> set[str]:
    """Mentions that do not abbreviate a reviewed commit."""

    return {
        mention
        for mention in mentions
        if not any(commit.startswith(mention) for commit in reviewed)
    }


#: Selector fields that name one Ledger fact. The compiler resolves a
#: reference's ledger_fact_key and ledger_source_record_id through one index
#: over a fact's aggregate, semantic, plain and legacy fact keys and its source
#: record id (ledger_targets._ledger_fact_index), so identifiers are compared
#: across fields.
_IDENTIFIER_SELECTOR_FIELDS = (
    "aggregate_fact_key",
    "semantic_fact_key",
    "legacy_fact_key",
    "source_record_id",
)
#: Selector fields matched as structures; here they never rule out an overlap.
_STRUCTURED_SELECTOR_FIELDS = {"dimensions", "dimension_values"}


def _selector_values(value: object) -> frozenset[str] | None:
    """The values a selector field admits, or None when it admits any.

    Mirrors ledger_targets._fact_matches_selector: an empty value is a
    wildcard and a list matches by membership.
    """

    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return frozenset(str(item) for item in value)
    return frozenset({str(value)})


def _ledger_identifiers(reference: LedgerTargetReference) -> set[str]:
    selector = reference.ledger_selector
    fields = [reference.ledger_fact_key, reference.ledger_source_record_id]
    fields += [selector.get(field) for field in _IDENTIFIER_SELECTOR_FIELDS]
    return {value for field in fields for value in _selector_values(field) or ()}


def _selectors_overlap(
    first: Mapping[str, object], second: Mapping[str, object]
) -> bool:
    """Whether one Ledger fact could satisfy both selectors.

    Every field both selectors pin must admit a common value. A field that
    either side leaves out or empty rules nothing out. This assumes a fact's
    alternative fields for one selector key agree: the compiler reads
    source_measure_id, for one, from two places in a fact.
    """

    if not first or not second:
        return False
    for field in set(first) & set(second) - _STRUCTURED_SELECTOR_FIELDS:
        admitted = _selector_values(first[field]), _selector_values(second[field])
        if None not in admitted and not admitted[0] & admitted[1]:
            return False
    return True


def holdout_leaks(
    cal: Sequence[LedgerTargetReference],
    precal: Sequence[LedgerTargetReference],
    holdout: Sequence[LedgerTargetReference],
) -> list[str]:
    """Name the routes from a hold-out fact into calibration or pre-calibration.

    These are the routes the references themselves show. An upstream
    (calibration or pre-calibration) reference leaks when it:

    - has a hold-out reference's name;
    - shares a Ledger identifier with one, in any of the identifier fields;
    - resolves through a selector (its own, or a scaled_by_ratio operand's,
      per ledger_targets.reference_fact_selectors) that one Ledger fact
      could satisfy together with a hold-out reference's selector;
    - belongs to a family the hold-out set covers.

    Sharing only a source is not a leak when both selectors pin different
    measures: one source publishes many facts. Without the Ledger feed this
    cannot match an identifier on one side against a selector on the other;
    the graph-level ancestry test (package G6) is to cover that.
    """

    names = {reference.name for reference in holdout}
    identifiers = {
        identifier
        for reference in holdout
        for identifier in _ledger_identifiers(reference)
    }
    families = {reference.family for reference in holdout}
    leaks: list[str] = []
    for reference_set, references in (("calibration", cal), ("precal", precal)):
        for reference in references:
            label = f"{reference_set} reference {reference.name!r}"
            if reference.name in names:
                leaks.append(f"{label}: hold-out name {reference.name!r}")
            for identifier in sorted(_ledger_identifiers(reference) & identifiers):
                leaks.append(f"{label}: hold-out ledger identifier {identifier!r}")
            for held in holdout:
                if any(
                    _selectors_overlap(selector, held_selector)
                    for selector in reference_fact_selectors(reference)
                    for held_selector in reference_fact_selectors(held)
                ):
                    leaks.append(f"{label}: hold-out ledger selector of {held.name!r}")
            if reference.family in families:
                leaks.append(f"{label}: hold-out family {reference.family!r}")
    return leaks


def crosswalk_errors(rows: list[Mapping[str, Any]]) -> list[str]:
    """Name every crosswalk row that breaks the share contract."""

    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        code = row.get("ta_code")
        if not isinstance(code, str) or not code:
            errors.append("row without ta_code")
            continue
        if code in seen:
            errors.append(f"{code}: duplicate territorial authority")
        seen.add(code)
        shares = row.get("shares")
        if not isinstance(shares, list) or not shares:
            errors.append(f"{code}: no shares")
            continue
        areas = [share.get("as_area") for share in shares]
        if any(isinstance(area, bool) or not isinstance(area, int) for area in areas):
            errors.append(f"{code}: area must be an integer")
            continue
        if len(set(areas)) != len(areas):
            errors.append(f"{code}: repeated area")
        if not set(areas) <= AREAS:
            errors.append(f"{code}: area outside 1-4")
        values = [share.get("population_share") for share in shares]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in values
        ):
            errors.append(f"{code}: share outside [0, 1]")
            continue
        if abs(math.fsum(values) - 1.0) > SHARE_TOLERANCE:
            errors.append(f"{code}: shares sum to {math.fsum(values)!r}, not 1")
    return errors


def modal_area(shares: list[Mapping[str, Any]]) -> int:
    """The S3 alternative: the area with the largest share; ties refuse."""

    best = max(share["population_share"] for share in shares)
    winners = [
        share["as_area"] for share in shares if share["population_share"] == best
    ]
    if len(winners) != 1:
        raise ValueError(f"tied modal areas {sorted(winners)}")
    return int(winners[0])


def scenario_errors(document: Mapping[str, Any]) -> list[str]:
    """Name every scenario that breaks the declared-grid contract."""

    errors: list[str] = []
    central = document["central_knobs"]
    knob_rows = document["knob_method_card_rows"]
    domains = document["knob_domains"]
    ids = [scenario.get("id") for scenario in document["scenarios"]]
    if len(set(ids)) != len(ids):
        errors.append("duplicate scenario ids")
    for scenario in document["scenarios"]:
        label = scenario.get("id")
        row = scenario.get("method_card_row")
        if not isinstance(row, str) or not METHOD_CARD_ROW.fullmatch(row):
            errors.append(f"{label}: must name exactly one method-card row")
        if scenario.get("tier") not in TIERS:
            errors.append(f"{label}: tier outside {sorted(TIERS)}")
        knobs = scenario.get("knobs")
        if not isinstance(knobs, Mapping):
            errors.append(f"{label}: knobs must be a mapping")
            continue
        if not knobs:
            if row != document["method_card_row"]:
                errors.append(f"{label}: the central scenario names the grid row")
            continue
        if len(knobs) != 1:
            errors.append(f"{label}: must change exactly one knob")
            continue
        ((knob, value),) = knobs.items()
        if knob not in central:
            errors.append(f"{label}: unknown knob {knob!r}")
            continue
        if value == central[knob]:
            errors.append(f"{label}: knob {knob!r} equals the central value")
        if knob_rows.get(knob) != row:
            errors.append(f"{label}: knob {knob!r} belongs to {knob_rows.get(knob)}")
        domain = domains[knob]
        if isinstance(domain, list):
            if value not in domain:
                errors.append(f"{label}: {value!r} outside the {knob!r} domain")
        elif value is None:
            if "status" not in scenario:
                errors.append(f"{label}: a pending value needs a status")
        elif not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            errors.append(f"{label}: {knob!r} needs a positive number")
    return errors


#: A file citation, with any extension, optionally qualified by directories.
_SOURCE_FILE = (
    r"(?P<file>[A-Za-z0-9_./-]*[A-Za-z0-9_-]\.[A-Za-z0-9][A-Za-z0-9_-]*)"
    r"(?![A-Za-z0-9_.-])"
)


def _line_citation_patterns(source: str) -> tuple[re.Pattern[str], ...]:
    """Share the documented locator grammar with extensionless filenames."""

    quotes = r"[`'\"‘’“”]*"
    # Keep wrappers attached to their filename or complete Markdown link.
    wrapped = (
        rf"\[?{quotes}{source}{quotes}"
        r"(?:\](?:\((?:[^()\n]|\([^()\n]*\))*\)|\[[^\]\n]*\])?)?"
    )
    number = r"L?\s*\d+(?!\w)"
    span = rf"{number}(?:\s*(?:[-–—]|(?i:to))\s*{number})?"
    numbers = rf"{span}(?:\s*,\s*{span})*"
    label = r"(?i:lines?\b\s+|ll?\.\s*)"
    forward = (
        wrapped
        + rf"(?:\s*[:：]\s*{quotes}(?:{label})?|\s*@\s*{quotes}(?:{label}|(?=L))|"
        rf"\s*#(?=L)|\s*,?\s*[\[(]?\s*"
        rf"(?:(?i:at|on)\s+)?{quotes}(?:{label}|(?=L))){quotes}{numbers}"
    )
    reverse = rf"\b(?:{label}|(?=L)){quotes}{numbers}{quotes}\s+(?i:of|in)\s+" + wrapped
    # Lowercase URL fragments must also survive raw fallback scanning. Keep
    # bare l12 and file:l12 ambiguous with symbols, as in the prose grammar.
    lower_fragment = wrapped + r"\s*#l\d+(?:[Cc]\d+)?(?!\w)"
    return (re.compile(forward), re.compile(reverse), re.compile(lower_fragment))


#: Keep every locator spelling from #1141, now for any file extension.
_LINE_CITATIONS = _line_citation_patterns(_SOURCE_FILE)
_MARKDOWN_LINK = re.compile(
    r"(?P<link>!?\[(?P<label>[^\]\n]*)\]"
    r"\((?P<destination>(?:[^()\n]|\([^()\n]*\))*)\))"
)
_MARKDOWN_LINE_CITATIONS = _line_citation_patterns(_MARKDOWN_LINK.pattern)
_SOURCE_URL = re.compile(r"(?P<url>(?i:https?)://[^\s<>\[\]()`\"'‘’“”]+)")
_URL_LINE_CITATIONS = _line_citation_patterns(r"[<(]?" + _SOURCE_URL.pattern + r"[>)]?")
_URL_LINE_FRAGMENT = re.compile(r"[Ll]\d+(?:[Cc]\d+)?(?:-[Ll]?\d+(?:[Cc]\d+)?)?")
_COMMIT_PIN = re.compile(r"[0-9a-f]{7,40}")


@dataclasses.dataclass(frozen=True)
class _CitationFile:
    path: str
    # URLs declare repository identity; plain filenames use checkout lookup.
    repository: bool | None = None


def _citation_destination(destination: str) -> tuple[_CitationFile, bool] | None:
    """Parse a path or HTTP(S) file URL and retain its repository identity."""

    def complete(match: re.Match[str] | None, text: str) -> bool:
        return match is not None and all(
            unicodedata.category(char)[0] in {"P", "S"} for char in text[match.end() :]
        )

    destination = destination.strip().removeprefix("<").removesuffix(">")
    path, separator, fragment = destination.partition("#")
    path = path.split("?", 1)[0]
    repository = None
    if re.match(r"(?i:https?)://", destination):
        try:
            url = urlsplit(destination)
        except ValueError:
            return None
        if not url.hostname:
            return None
        path, fragment = unquote(url.path).lstrip("/"), url.fragment
        file_prefix = re.match(r".*[^/.]\.[A-Za-z0-9][A-Za-z0-9_-]*", path)
        if not path.endswith("/") and complete(file_prefix, path):
            path = file_prefix[0]
        repository = False
        parts = path.split("/")
        if url.hostname.lower() == "github.com":
            if len(parts) < 5 or parts[2] not in {"blob", "raw", "tree", "blame"}:
                return None
            repository = "/".join(parts[:2]).lower() == "policyengine/microcosm"
            path = "/".join(parts[4:])
        elif url.hostname.lower() == "raw.githubusercontent.com":
            if len(parts) < 4:
                return None
            repository = "/".join(parts[:2]).lower() == "policyengine/microcosm"
            path = "/".join(parts[3:])
        if repository:
            # A branch may contain slashes. Prefer the longest known checkout
            # suffix, while keeping files absent from this ref local as well.
            path = max(
                (
                    tracked
                    for tracked in _repository_files()
                    if path == tracked or path.endswith("/" + tracked)
                ),
                key=len,
                default=path,
            )
    if (
        re.fullmatch(_SOURCE_FILE, path)
        or _extensionless_file_pattern().fullmatch(path)
        # Decoded URL names may contain spaces, brackets or Unicode. Falling
        # back to ASCII prose matching would lose their repository identity.
        or (
            repository is not None
            and re.search(r"[^/.]\.[A-Za-z0-9][A-Za-z0-9_-]*$", path)
        )
    ):
        # Take the longest complete line/column span. Surrounding punctuation
        # is prose, including Unicode punctuation and ASCII symbol characters.
        # A heading such as L12_symbol still has a non-punctuation suffix.
        located = complete(_URL_LINE_FRAGMENT.match(fragment), fragment)
        if (
            not located
            and fragment.strip()
            and not (
                complete(re.match(r"[\w.-]+", fragment), fragment)
                or fragment.startswith("/")
            )
        ):
            # Do not mask a fragment we could not understand. Raw scanning
            # can still catch a locator in a malformed URL or pasted query.
            return None
        return _CitationFile(str(PurePosixPath(path)), repository), located
    if repository is None and not separator:
        for pattern in (*_LINE_CITATIONS, *_extensionless_line_patterns()):
            if located := pattern.fullmatch(path):
                return _CitationFile(str(PurePosixPath(located["file"]))), True
    return None


def _mask_json_pointer_citations(text: str) -> str:
    # Share the pointer parser's spans; fragment tokens are data, never prose.
    # Retain the file so an independent locator outside a link still cites it.
    return _JSON_POINTER_CITATION.sub(
        lambda match: match["file"] + "#" + " " * len(match["pointer"]), text
    )


def _markdown_link_files(
    match: re.Match[str], destination: _CitationFile
) -> frozenset[_CitationFile]:
    """Attribute a link to its destination and every filename in its label."""

    files = {
        _CitationFile(str(PurePosixPath(named["file"])))
        for pattern in (re.compile(_SOURCE_FILE), _extensionless_file_pattern())
        for named in pattern.finditer(match["label"])
    }
    files.add(destination)
    return frozenset(files)


def _source_line_citation_groups(
    value: object,
) -> Iterator[tuple[frozenset[_CitationFile], str]]:
    """Keep all files sharing a link's locator together for the pin check."""

    patterns = (*_LINE_CITATIONS, *_extensionless_line_patterns())
    for text in _nested_strings(value):
        prose = _mask_json_pointer_citations(text)
        located_links = {
            match.span("link")
            for pattern in _MARKDOWN_LINE_CITATIONS
            for match in pattern.finditer(prose)
        }
        parsed_links = set()
        for link in _MARKDOWN_LINK.finditer(prose):
            destination = _citation_destination(link["destination"])
            if destination is None:
                continue
            file, located = destination
            parsed_links.add(link.span())
            if (
                located
                or link.span("link") in located_links
                or any(pattern.search(link["label"]) for pattern in patterns)
            ):
                yield _markdown_link_files(link, file), text
        # Unparseable destinations must leave the full link available to scan.
        prose = _MARKDOWN_LINK.sub(
            lambda match, spans=parsed_links: (
                " " * len(match[0]) if match.span() in spans else match[0]
            ),
            prose,
        )
        located_urls = {
            match.span("url")
            for pattern in _URL_LINE_CITATIONS
            for match in pattern.finditer(prose)
        }
        parsed_urls = set()
        for url in _SOURCE_URL.finditer(prose):
            destination = _citation_destination(url[0])
            if destination is None:
                continue
            file, located = destination
            parsed_urls.add(url.span())
            if located or url.span() in located_urls:
                yield frozenset({file}), text
        prose = _SOURCE_URL.sub(
            lambda match, spans=parsed_urls: (
                " " * len(match[0]) if match.span() in spans else match[0]
            ),
            prose,
        )
        for pattern in patterns:
            for match in pattern.finditer(prose):
                yield (
                    frozenset({_CitationFile(str(PurePosixPath(match["file"])))}),
                    text,
                )


def source_line_citations(value: object) -> list[tuple[str, str]]:
    """Each (file path, string) citing a line, including nested mapping keys."""

    return [
        (name, text)
        for files, text in _source_line_citation_groups(value)
        for name in sorted({file.path for file in files})
    ]


@cache
def _repository_files() -> frozenset[str]:
    """Tracked and newly added checkout files, without traversing environments."""

    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=_BUILD.repository,
        check=True,
        capture_output=True,
    )
    return frozenset(result.stdout.decode().split("\0")) - {""}


@cache
def _extensionless_file_pattern() -> re.Pattern[str]:
    # Bare words in prose are not file names. Recognize extensionless names
    # only when the checkout actually contains such a file (also .gitignore).
    names = {
        PurePosixPath(path).name
        for path in _repository_files()
        if not PurePosixPath(path).suffix
    }
    alternatives = "|".join(re.escape(name) for name in sorted(names))
    source = rf"(?P<file>[A-Za-z0-9_./-]*(?:{alternatives}))(?![A-Za-z0-9_.-])"
    return re.compile(source if names else r"(?P<file>(?!))")


@cache
def _extensionless_line_patterns() -> tuple[re.Pattern[str], ...]:
    return _line_citation_patterns(_extensionless_file_pattern().pattern)


def _repository_source(path: str) -> bool:
    root = _BUILD.repository.resolve()
    candidate = (root / path).resolve()
    if (
        candidate.is_relative_to(root)
        and candidate.relative_to(root).as_posix() in _repository_files()
    ):
        return True
    name = str(PurePosixPath(path))
    return any(
        tracked == name or tracked.endswith("/" + name)
        for tracked in _repository_files()
    )


def commit_pinned_sources(value: object) -> set[str]:
    """Paths named by same-mapping lowercase 7–40-hex commit pins."""

    pinned: set[str] = set()
    if isinstance(value, Mapping):
        path, commit = value.get("path"), value.get("commit")
        if (
            isinstance(path, str)
            and re.fullmatch(r"[A-Za-z0-9_./-]+", path)
            and not path.endswith("/")
            and PurePosixPath(path).name not in {"", ".", ".."}
            and isinstance(commit, str)
            and _COMMIT_PIN.fullmatch(commit)
        ):
            pinned.add(str(PurePosixPath(path)))
        for child in value.values():
            pinned |= commit_pinned_sources(child)
    elif isinstance(value, list):
        for child in value:
            pinned |= commit_pinned_sources(child)
    return pinned


def _pin_covers(cited: str, pinned: str) -> bool:
    # The existing ops resource cites bare run.py but pins its complete path.
    # Qualified citations cannot borrow a pin from another directory's file.
    return cited == pinned or ("/" not in cited and PurePosixPath(pinned).name == cited)


def unpinned_line_citations(document: object) -> list[tuple[str, str]]:
    """Refuse all local-file lines and foreign-file lines without a file pin."""

    pinned = commit_pinned_sources(document)
    return [
        (name, text)
        for files, text in _source_line_citation_groups(document)
        if not all(
            not (
                file.repository
                if file.repository is not None
                else _repository_source(file.path)
            )
            and any(
                file.path == pin
                if file.repository is not None
                else _pin_covers(file.path, pin)
                for pin in pinned
            )
            for file in files
        )
        for name in sorted({file.path for file in files})
    ]


# Retain #1141's public guard spelling while widening its scope to all files.
unpinned_python_line_citations = unpinned_line_citations


_DOTTED_MICROCOSM_SYMBOL = re.compile(
    r"(?<![A-Za-z0-9_./-])microcosm(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
)
_JSON_POINTER_CITATION = re.compile(_SOURCE_FILE + r"#(?P<pointer>(?:/[^\s;()\"']*)+)")
_JSON_RECORD_SELECTOR = re.compile(r"(?P<key>[^\[\]]+)\[(?P<terms>[^\[\]]+)\]")
_JSON_SELECTOR_TERM = re.compile(
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\[\],=\s]+)"
)


def resolve_dotted_symbol(dotted: str) -> object:
    """Import the longest module prefix, then resolve its attributes."""

    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        name = ".".join(parts[:cut])
        try:
            target: object = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name is None or not (
                name == error.name or name.startswith(error.name + ".")
            ):
                raise
            continue
        if cut == 1:
            raise ModuleNotFoundError(dotted, name=dotted)
        for attribute in parts[cut:]:
            target = getattr(target, attribute)
        return target
    raise ModuleNotFoundError(dotted, name=dotted)


def dotted_symbol_citations(value: object) -> set[str]:
    """Find Python symbols, distinguishing the existing serialized format ID."""

    if isinstance(value, Mapping):
        return {
            dotted
            for key, child in value.items()
            for nested in (
                [str(key)]
                if key == "format" and child == INPUT_CLOSURE_FORMAT
                else [str(key), child]
            )
            for dotted in dotted_symbol_citations(nested)
        }
    if isinstance(value, list):
        return {dotted for child in value for dotted in dotted_symbol_citations(child)}
    if isinstance(value, str):
        return set(_DOTTED_MICROCOSM_SYMBOL.findall(value))
    return set()


def json_pointer_citations(value: object) -> list[tuple[str, str]]:
    """Find and validate every citation, including nested mapping keys."""

    citations = [
        (match["file"], match["pointer"].rstrip(","))
        for text in _nested_strings(value)
        for match in _JSON_POINTER_CITATION.finditer(text)
    ]
    for citation in citations:
        resolve_json_pointer(*citation)
    return citations


def resolve_json_pointer(file: str, pointer: str) -> object:
    """Resolve a pointer with exactly-one keyed selectors; refuse record indexes."""

    relative = str(PurePosixPath(file))
    candidates = {
        (_BUILD.repository / candidate).resolve()
        for candidate in _repository_files()
        if candidate == relative or candidate.endswith("/" + relative)
    }
    if len(candidates) != 1:
        raise ValueError(f"JSON citation needs one checkout file: {file}: {candidates}")
    (path,) = candidates
    if not path.is_relative_to(_BUILD.repository.resolve()) or not path.is_file():
        raise ValueError(f"JSON citation leaves checkout or names no file: {file}")
    if pointer and not pointer.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {pointer}")
    content = path.read_text(encoding="utf-8")
    selected: object = (
        load_yaml12(content, source=relative)
        if path.suffix in {".yaml", ".yml"}
        else json.loads(content)
    )
    for token in pointer.split("/")[1:]:
        if re.search(r"~(?![01])", token):
            raise ValueError(f"Invalid JSON pointer escape: {pointer}")
        selector: dict[str, str] | None = None
        key = token
        if "[" in token or "]" in token:
            match = _JSON_RECORD_SELECTOR.fullmatch(token)
            if match is None:
                raise ValueError(f"Invalid JSON record selector: {token}")
            key = match["key"]
            selector = {}
            for term in match["terms"].split(","):
                pair = _JSON_SELECTOR_TERM.fullmatch(term)
                if pair is None or pair["field"] in selector:
                    raise ValueError(f"Invalid JSON record selector: {token}")
                selector[pair["field"]] = (
                    pair["value"].replace("~1", "/").replace("~0", "~")
                )
        key = key.replace("~1", "/").replace("~0", "~")
        if isinstance(selected, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", key):
                raise ValueError(f"Invalid JSON array index: {key}")
            if all(isinstance(record, Mapping) for record in selected):
                raise ValueError(
                    f"JSON citation indexes a record array: {file}#{pointer}"
                )
            selected = selected[int(key)]
        elif isinstance(selected, Mapping):
            selected = selected[key]
        else:
            raise TypeError(f"JSON pointer traverses a scalar: {file}#{pointer}")
        if selector is not None:
            if not isinstance(selected, list) or not all(
                isinstance(record, Mapping) for record in selected
            ):
                raise ValueError(
                    f"JSON record selector needs an array of mappings: {token}"
                )
            matches = [
                record
                for record in selected
                if all(record.get(field) == value for field, value in selector.items())
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"JSON selector needs one matching record, found {len(matches)}: {token}"
                )
            selected = matches[0]
    return selected


# ---------------------------------------------------------------------------
# Reference sets
# ---------------------------------------------------------------------------


_IDENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=24
)


#: Every place a reference can carry a Ledger identifier.
_IDENTIFIER_SLOTS = (
    "ledger_fact_key",
    "ledger_source_record_id",
    *(f"ledger_selector.{field}" for field in _IDENTIFIER_SELECTOR_FIELDS),
)
#: Ways an upstream selector can reach the fact a hold-out selector names.
_OVERLAPPING_SELECTORS = {
    "same": lambda selector: dict(selector),
    "measure in a list": lambda selector: {
        **selector,
        "source_measure_id": ["another_measure", selector["source_measure_id"]],
    },
    "measure left empty": lambda selector: {**selector, "source_measure_id": ""},
    "source only": lambda selector: {"source_name": selector["source_name"]},
    "no geography": lambda selector: {
        field: value for field, value in selector.items() if field != "geography_level"
    },
}

_Plant = Callable[[LedgerTargetReference], LedgerTargetReference]


def _with_identifier(
    reference: LedgerTargetReference, slot: str, identifier: str
) -> LedgerTargetReference:
    if slot.startswith("ledger_selector."):
        field = slot.removeprefix("ledger_selector.")
        selector = {**reference.ledger_selector, field: identifier}
        return dataclasses.replace(reference, ledger_selector=selector)
    return dataclasses.replace(reference, **{slot: identifier})


def _as_ratio_numerator(
    reference: LedgerTargetReference, numerator: Mapping[str, object]
) -> LedgerTargetReference:
    """``reference`` as a scaled_by_ratio whose numerator reads ``numerator``."""

    operands = (
        {"role": "base"},
        {"role": "numerator", "period_type": "", "period_value": "", **numerator},
        {"role": "denominator", "source_name": "unrelated_source"},
    )
    return dataclasses.replace(
        reference, value_operation="scaled_by_ratio", value_operands=operands
    )


def _fresh_hold_out_and_plant(
    held: LedgerTargetReference, kind: str, identity: str, data: st.DataObject
) -> tuple[LedgerTargetReference, _Plant]:
    """Give ``held`` a fresh ``kind`` identity; return it and its upstream plant."""

    if kind == "name":
        return (
            dataclasses.replace(held, name=identity),
            lambda reference: dataclasses.replace(reference, name=identity),
        )
    if kind == "ledger identifier":
        held_slot, planted_slot = (
            data.draw(st.sampled_from(_IDENTIFIER_SLOTS)) for _ in range(2)
        )
        return (
            _with_identifier(held, held_slot, identity),
            lambda reference: _with_identifier(reference, planted_slot, identity),
        )
    if kind == "ledger selector":
        selector = {
            **held.ledger_selector,
            "source_name": f"{identity}_source",
            "source_measure_id": f"{identity}_measure",
        }
        variant = data.draw(st.sampled_from(sorted(_OVERLAPPING_SELECTORS)))
        overlapping = _OVERLAPPING_SELECTORS[variant](selector)
        route = data.draw(st.sampled_from(["own", "upstream ratio", "hold-out ratio"]))
        if route == "hold-out ratio":
            # The hold-out reads the fresh fact as a scaled_by_ratio numerator.
            return (
                _as_ratio_numerator(held, selector),
                lambda reference: dataclasses.replace(
                    reference, ledger_selector=overlapping
                ),
            )
        held = dataclasses.replace(held, ledger_selector=selector)
        if route == "upstream ratio":
            return held, lambda reference: _as_ratio_numerator(reference, overlapping)
        return held, lambda reference: dataclasses.replace(
            reference, ledger_selector=overlapping
        )
    assert kind == "family"
    return (
        dataclasses.replace(held, family=identity),
        lambda reference: dataclasses.replace(reference, family=identity),
    )


def test_chronicle_feed_is_a_loaded_legacy_json_resource() -> None:
    spec = load_country_spec("nz")
    row = next(row for row in spec.resource_rows if row.path == "chronicle_feed.json")

    assert row.kind == "legacy_json"
    assert row.schema_id == "legacy_json"
    assert (
        spec.resource_hashes[row.path]
        == hashlib.sha256((NZ_ROOT / row.path).read_bytes()).hexdigest()
    )
    assert _load(row.path)["country"] == spec.country
    assert _load(row.path)["scope"] == "nz_calibration"


class TestReferenceSets:
    @pytest.mark.parametrize("reference_set", sorted(REFERENCE_FILES))
    def test_each_set_is_a_self_identifying_nz_reference_resource(
        self, reference_set: str
    ) -> None:
        document = _load(REFERENCE_FILES[reference_set])
        assert document["country"] == "nz"
        assert document["reference_set"] == reference_set
        expected_operations = (
            [
                "identity",
                "rolled_forward_by_ratio",
                "allocated_residual_by_ratio",
                "linear_combination",
            ]
            if reference_set == "calibration"
            else ["identity"]
        )
        assert document["allowed_value_operations"] == expected_operations
        rows = document["target_references"]
        assert rows
        assert FORBIDDEN_VALUE_KEYS.isdisjoint(_nested_keys(rows))
        names = [row["name"] for row in rows]
        assert len(set(names)) == len(names)
        assert {row["metadata"]["reference_set"] for row in rows} == {reference_set}
        for row in rows:
            assert METHOD_CARD_ROW.fullmatch(row["metadata"]["method_card_row"])
        # The declared operations hold row by row, so every fact a reference
        # reads comes through its own selector or identifiers.
        for reference in _references(reference_set):
            assert reference.value_operation in expected_operations
            if reference.value_operation == "identity":
                assert reference.value_operands == ()
            else:
                assert {operand["role"] for operand in reference.value_operands} == {
                    "base",
                    "numerator",
                    "denominator",
                }

    @pytest.mark.parametrize("reference_set", sorted(REFERENCE_FILES))
    def test_every_parked_reference_names_its_blocker_and_refuses_compilation(
        self, reference_set: str
    ) -> None:
        references = _references(reference_set)
        parked = [
            row for row in references if row.metadata["activation_status"] != "active"
        ]
        assert parked
        for reference in parked:
            assert reference.metadata["activation_status"].startswith("requires_")
            if reference_set != "holdout":
                assert reference.metadata["parking_reason"].strip()
            with pytest.raises(ValueError, match="non-executable placeholder"):
                compile_ledger_target_references([], [reference], country="nz")
        assert (
            sum(row.metadata["activation_status"] == "active" for row in references)
            == {"calibration": 77, "precal": 80, "holdout": 2}[reference_set]
        )

    def test_pre_calibration_rows_are_not_weight_calibration_targets(self) -> None:
        roles = {
            reference.metadata["target_role"] for reference in _references("precal")
        }
        assert roles == {"pre_calibration"}

    def test_loader_and_direct_parse_agree_on_the_calibration_set(self) -> None:
        # Differential: the country-spec loader validates target_references.json;
        # the pre-calibration and hold-out files use the direct parse.
        loaded = load_country_spec("nz").target_references
        assert loaded == tuple(_references("calibration"))

    def test_calibration_set_is_the_method_card_mc5_list(self) -> None:
        references = _references("calibration")
        assert {
            reference.metadata["contract_target_id"] for reference in references
        } == {
            "stats_nz_population_by_age_region",
            "stats_nz_census_households_by_size_region",
            "stats_nz_census_families_by_type_region",
            "stats_nz_census_households_by_tenure_region",
            "ird_taxable_income_people_by_band",
            "ird_taxable_income_total_by_band",
            "msd_main_benefit_recipients_by_program_age",
            "msd_nz_super_recipients",
            "msd_veterans_pension_recipients",
        }
        assert len(references) == 773
        assert {reference.metadata["target_role"] for reference in references} == {
            "calibration"
        }
        assert {reference.metadata["method_card_row"] for reference in references} == {
            "MC5"
        }

    def test_holdout_set_is_validation_only(self) -> None:
        references = _references("holdout")
        assert {reference.name for reference in references} == {
            "msd_accommodation_supplement_recipients_national",
            "msd_annual_report_accommodation_assistance_expenditure",
            "treasury_an24_01_accommodation_supplement_fiscal_total",
            "ird_wff_recipient_families",
            "ird_iwtc_recipient_families",
            "ird_wff_recipient_families_by_family_income_band",
        }
        for reference in references:
            assert reference.metadata["target_role"] == "validation"
            assert reference.metadata["criticality"] == "diagnostic"
            assert reference.family in HELD_OUT_FAMILIES

    @pytest.mark.parametrize(
        "reference_name",
        [
            "ird_wff_recipient_families",
            "ird_iwtc_recipient_families",
            "ird_wff_recipient_families_by_family_income_band",
        ],
    )
    def test_selected_parked_wff_reference_is_named_by_preflight(
        self, reference_name: str
    ) -> None:
        from microcosm.build.transport.compose import validate_transport_activation

        reference = next(
            row for row in _rows("holdout") if row["name"] == reference_name
        )
        assert reference["metadata"]["activation_status"].startswith("requires_")
        spec = {
            "country": "nz",
            "resources": {
                "target_references": {"target_references": []},
                "precal_references": {"target_references": []},
                "holdout_references": {"target_references": [reference]},
                "transport_graph": {
                    "nodes": [
                        {
                            "params": {
                                "reference_document": {
                                    "resource": "holdout_references",
                                    "encoding": "json",
                                }
                            }
                        }
                    ]
                },
            },
        }
        # Other unimplemented producers may also be named. This selected
        # parked reference must remain an independent preflight blocker.
        with pytest.raises(
            ValueError,
            match=f"placeholder reference holdout_references\\.{reference_name}",
        ):
            validate_transport_activation(spec)

    def test_no_held_out_fact_reaches_calibration_or_pre_calibration(self) -> None:
        held = _references("holdout")
        upstream = _references("calibration") + _references("precal")
        assert {reference.family for reference in held} == HELD_OUT_FAMILIES
        # The AS recipients hold-out and the main-benefit calibration row share
        # the source msd_quarterly_benefit_facts but not a measure, so they
        # name different facts (the plan's "filter by fact key").
        assert (
            holdout_leaks(_references("calibration"), _references("precal"), held) == []
        )
        assert HELD_OUT_FAMILIES.isdisjoint(reference.family for reference in upstream)

    @pytest.mark.parametrize(
        "name",
        [
            "msd_accommodation_supplement_recipients_national",
            "msd_annual_report_accommodation_assistance_expenditure",
        ],
    )
    def test_source_only_ratio_plant_reaches_a_differently_pinned_holdout(
        self, name: str
    ) -> None:
        held = next(row for row in _references("holdout") if row.name == name)
        donor = _references("calibration")[0]
        planted = _as_ratio_numerator(
            donor, {"source_name": held.ledger_selector["source_name"]}
        )
        assert any(
            ": hold-out ledger selector " in leak
            for leak in holdout_leaks([planted], [], [held])
        )

    # A stalled shared host once made one draw take 36.9 s (load1 ~300), which
    # Hypothesis reports as too_slow; the strategy itself is cheap.
    @settings(
        max_examples=300,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(
        data=st.data(),
        kind=st.sampled_from(HOLDOUT_LEAK_KINDS),
        target=st.sampled_from(("calibration", "precal")),
        identity=_IDENT,
        insert=st.booleans(),
    )
    def test_a_hold_out_identity_planted_upstream_is_refused(
        self, data, kind: str, target: str, identity: str, insert: bool
    ) -> None:
        sets = {name: _references(name) for name in REFERENCE_FILES}
        held = data.draw(st.integers(0, len(sets["holdout"]) - 1))
        sets["holdout"][held], plant = _fresh_hold_out_and_plant(
            sets["holdout"][held], kind, identity, data
        )
        # A fresh identity: before the plant, no upstream reference carries it.
        assume(not holdout_leaks(sets["calibration"], sets["precal"], sets["holdout"]))
        upstream = sets[target]
        if insert:
            donor = data.draw(st.sampled_from(upstream))
            position = data.draw(st.integers(0, len(upstream)))
            upstream.insert(position, plant(donor))
        else:
            position = data.draw(st.integers(0, len(upstream) - 1))
            upstream[position] = plant(upstream[position])
        leaks = holdout_leaks(sets["calibration"], sets["precal"], sets["holdout"])
        assert any(
            leak.startswith(f"{target} ") and f": hold-out {kind} " in leak
            for leak in leaks
        )

    def test_a_reference_in_two_upstream_sets_names_one_fact(self) -> None:
        calibration = {row["name"]: row for row in _rows("calibration")}
        shared = [row for row in _rows("precal") if row["name"] in calibration]
        assert len(shared) == 14
        assert {row["metadata"]["contract_target_id"] for row in shared} == {
            "msd_main_benefit_recipients_by_program_age",
            "msd_nz_super_recipients",
            "msd_veterans_pension_recipients",
        }
        fact_fields = (
            "ledger_selector",
            "entity",
            "measure",
            "filter",
            "period",
            "family",
        )
        for row in shared:
            twin = calibration[row["name"]]
            assert {field: row.get(field) for field in fact_fields} == {
                field: twin.get(field) for field in fact_fields
            }
            if (
                row["metadata"]["contract_target_id"]
                == "msd_main_benefit_recipients_by_program_age"
            ):
                assert row["filter"]
                assert row["metadata"]["activation_status"] == "active"
            assert row["ledger_selector"]["period_value"] == "2026-06"
            assert row["period"] == "2026-06"

    def test_gate_profile_requires_exactly_the_calibration_families(self) -> None:
        spec = load_country_spec("nz")
        gate = next(
            gate for gate in spec.gates.gates if gate.gate == "target_profile_coverage"
        )
        required = set(gate.parameters["required_families"])
        assert required == {
            reference.family for reference in _references("calibration")
        }
        assert HELD_OUT_FAMILIES.isdisjoint(required)


# ---------------------------------------------------------------------------
# Rulespec pin
# ---------------------------------------------------------------------------


_HEX40_TEXT = st.text(alphabet="0123456789abcdef", min_size=40, max_size=40)
_FILLER = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20)
#: A suffix that cannot extend the planted hex run (41 hex digits is no commit).
_SUFFIX = _FILLER.filter(lambda text: text[:1].lower() not in set("0123456789abcdef"))
#: Prose between "rulespec" and a planted commit, of any length, with no hex
#: digit that could extend the planted run.
_GAP = st.text(
    alphabet="ghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ '`-_@:/.,()#",
    min_size=1,
    max_size=300,
)
_KEY_AFFIX = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_-.", max_size=12)
#: Spellings of the rulespec-nz repository a record may carry.
_RULESPEC_REPOSITORIES = (
    "TheAxiomFoundation/rulespec-nz",
    "TheAxiomFoundation/rulespec-nz.git",
    "https://github.com/TheAxiomFoundation/rulespec-nz/",
    "https://github.com/TheAxiomFoundation/rulespec-nz/tree/main",
    "TheAxiomFoundation/RuleSpec-NZ",
)


class TestRulespecPin:
    @pytest.mark.parametrize(
        "case",
        [
            "canonical",
            "unattributed",
            "incomplete",
            "explicit_commit",
            "extra_commit",
            "rulespec_extra_revision",
            "prose",
            "rulespec_context",
            "wrong_schema",
        ],
    )
    def test_chronicle_identity_scope_preserves_rulespec_mentions(self, case) -> None:
        identity = _load("reference_aging.json")["feed_identity"]
        commit = identity["source_commit"]
        payload = {"feed_identity": identity}
        expected = set()
        if case == "unattributed":
            payload = {"other": identity}
            expected = {commit}
        elif case == "incomplete":
            identity.pop("manifest_sha256")
            expected = {commit}
        elif case == "explicit_commit":
            identity["rulespec_nz_commit"] = "c0ffee1"
            expected = {"c0ffee1"}
        elif case == "extra_commit":
            identity["commit"] = "deadbee"
            expected = {"deadbee"}
        elif case == "rulespec_extra_revision":
            identity["revision"] = "deadbee"
            payload = {"rulespec": payload}
            expected = {commit, "deadbee"}
        elif case == "prose":
            identity["notes"] = "RuleSpec-nz deadbee"
            expected = {"deadbee"}
        elif case == "rulespec_context":
            payload = {"rulespec": payload}
            expected = {commit}
        elif case == "wrong_schema":
            identity["artifact_schema_version"] = "unknown"
            expected = {commit}
        assert rulespec_commit_mentions(payload) == expected

    def test_every_rulespec_commit_named_anywhere_was_reviewed_on_main(self) -> None:
        pin = _load("axiom_rules_bindings.json")["rulespec"]
        mentions = rulespec_commit_mentions(_package_payloads())
        assert pin["commit"] in mentions
        assert unreviewed_mentions(mentions, reviewed_rulespec_commits()) == set()
        assert pin["branch"] == "main"
        # The branch claim is the lane's git receipt, not checkable offline.
        assert "git branch -r --contains" in pin["verification"]

    def test_the_unmerged_budget_transport_commit_is_absent(self) -> None:
        texts = [
            path.read_text(encoding="utf-8")
            for path in NZ_ROOT.rglob("*")
            if path.is_file()
        ]
        assert not any(UNMERGED_RULESPEC_COMMIT[:7] in text for text in texts)

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT, prefix=_FILLER, suffix=_SUFFIX, nest=st.integers(0, 3))
    def test_scanner_finds_an_attributed_commit_at_any_depth(
        self, commit: str, prefix: str, suffix: str, nest: int
    ) -> None:
        planted: object = f"{prefix}rulespec-nz@{commit}{suffix}"
        for depth in range(nest):
            planted = {"level": depth, "note": [planted]}
        assert commit in rulespec_commit_mentions(planted)
        record = {"repository": "TheAxiomFoundation/rulespec-nz", "commit": commit}
        assert rulespec_commit_mentions({"wrapper": [record]}) == {commit}

    @settings(max_examples=200, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        repository_field=st.sampled_from(("repository", "source_repo")),
        repository_name=st.sampled_from(
            ("TheAxiomFoundation/axiom-oracles", "PolicyEngine/chronicle")
        ),
        commit_field=st.sampled_from(("commit", "source_commit")),
    )
    def test_scanner_ignores_commits_of_other_repositories(
        self,
        commit: str,
        repository_field: str,
        repository_name: str,
        commit_field: str,
    ) -> None:
        other = {repository_field: repository_name, commit_field: commit}
        assert rulespec_commit_mentions(other) == set()
        assert rulespec_commit_mentions({"rulespec": other}) == {commit}

    @settings(max_examples=200, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        style=st.sampled_from(
            [
                "pinned to rulespec-nz commit {}.",
                "at rulespec-nz commit `{}`",
                "generated at rulespec-nz {} today",
                "rulespec-nz@{}:nz/statutes",
            ]
        ),
    )
    def test_scanner_catches_abbreviated_prose_and_flags_unreviewed_commits(
        self, commit: str, length: int, style: str
    ) -> None:
        short = commit[:length]
        mentions = rulespec_commit_mentions({"notes": style.format(short)})
        assert short in mentions
        reviewed = reviewed_rulespec_commits()
        if not any(full.startswith(short) for full in reviewed):
            assert unreviewed_mentions(mentions, reviewed) == {short}

    def test_scanner_catches_the_821_wording(self) -> None:
        wording = "the rules boundary names the contract at rulespec-nz commit 3b663b3."
        mentions = rulespec_commit_mentions({"status": wording})
        assert unreviewed_mentions(mentions, reviewed_rulespec_commits()) == {"3b663b3"}

    def test_scanner_catches_the_phrasings_the_reviews_planted(self) -> None:
        # Each edit passed an earlier version of the scanner: the first two in
        # the microcosm#1119 review (finding 3), the last three in the review
        # of its fix-up.
        payloads = _package_payloads()
        payloads["spec/bundle.yaml"]["status"] += (
            " Uses rulespec-nz's commit deadbeefcafe1."
        )
        payloads["currency_bridge.json"]["rulespec_nz_commit"] = "deadbeef" * 5
        surface = payloads["axiom_rules_bindings.json"]["input_surface"]
        assert "git diff 6fe181fc 8dc2507" in surface["notes"]
        surface["notes"] = surface["notes"].replace(
            "git diff 6fe181fc 8dc2507", "git diff 6fe181fc c0ffee1"
        )
        # The IRS rate's source record lives in money_normalization.json (w2a).
        payloads["money_normalization.json"]["rate"]["citation"]["rules"] = {
            "repository": "TheAxiomFoundation/rulespec-nz.git",
            "revision": "facade9876543",
        }
        payloads["scenarios.json"]["rulespec"] = {"head": "abad1dea"}
        mentions = rulespec_commit_mentions(payloads)
        assert unreviewed_mentions(mentions, reviewed_rulespec_commits()) == {
            "deadbeefcafe1",
            "deadbeef" * 5,
            "c0ffee1",
            "facade9876543",
            "abad1dea",
        }

    def test_a_rulespec_commit_planted_in_the_input_closure_is_caught(self) -> None:
        # The closure names the axiom-rules-engine commit that generated the
        # input surface in a foreign record (one naming that repository), so
        # the scan passes it over; test_nz_axiom_input_closure.py pins that
        # commit to the surface. A rulespec commit planted beside it is caught.
        reviewed = reviewed_rulespec_commits()
        clean = _package_payloads()
        engine = clean["axiom_input_closure.json"]["input_surface"]["engine"]
        assert engine["repository"] == "TheAxiomFoundation/axiom-rules-engine"
        assert unreviewed_mentions(rulespec_commit_mentions(clean), reviewed) == set()

        def planted(edit: Callable[[dict[str, Any]], object]) -> set[str]:
            payloads = copy.deepcopy(clean)
            edit(payloads["axiom_input_closure.json"])
            return unreviewed_mentions(rulespec_commit_mentions(payloads), reviewed)

        assert planted(
            lambda closure: closure["input_surface"].update(rulespec_commit="c0ffee1")
        ) == {"c0ffee1"}
        assert planted(
            lambda closure: closure["input_surface"]["engine"].update(
                rulespec_nz_commit="abad1dea"
            )
        ) == {"abad1dea"}
        assert planted(
            lambda closure: closure["entries"][0].update(
                reason="Set at rulespec-nz deadbee."
            )
        ) == {"deadbee"}
        # The record escapes the scan only while it names a non-rulespec
        # repository.
        assert planted(
            lambda closure: closure["input_surface"]["engine"].update(
                repository="TheAxiomFoundation/rulespec-nz"
            )
        ) == {engine["commit"]}
        # The engine evidence cites the engine's src/rulespec.rs, so the
        # engine commit's hex in that prose reads as a rulespec mention.
        short = engine["commit"][:8]
        assert planted(
            lambda closure: closure.update(
                engine_optional_evidence=(
                    f"axiom-rules-engine {short}: "
                    + closure["engine_optional_evidence"]
                )
            )
        ) == {short}

    @settings(max_examples=300, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        gap=_GAP,
        before=st.booleans(),
        upper=st.booleans(),
        anchor=st.sampled_from(["rulespec", "Rulespec", "RuleSpec", "RULESPEC"]),
    )
    def test_any_hex_run_in_a_string_that_names_rulespec_is_a_mention(
        self,
        commit: str,
        length: int,
        gap: str,
        before: bool,
        upper: bool,
        anchor: str,
    ) -> None:
        short = commit[:length]
        planted = short.upper() if upper else short
        text = f"{planted}{gap}{anchor}" if before else f"{anchor}{gap}{planted}"
        assert short in rulespec_commit_mentions({"notes": text})
        assert short in rulespec_commit_mentions({text: "a mapping key"})

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT, length=st.integers(7, 40), filler=_FILLER)
    def test_a_string_that_does_not_name_rulespec_is_not_a_mention(
        self, commit: str, length: int, filler: str
    ) -> None:
        text = f"{filler}{commit[:length]}{filler}"
        assume("rulespec" not in text.lower())
        assert rulespec_commit_mentions({"notes": text}) == set()

    @settings(max_examples=200, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        repository=st.sampled_from(_RULESPEC_REPOSITORIES),
        repository_field=st.sampled_from(("repository", "source_repo")),
        key=st.sampled_from(["commit", "revision", "sha", "ref", "head", "pinned"]),
    )
    def test_every_hex_value_of_a_rulespec_record_is_a_mention(
        self, commit: str, length: int, repository: str, repository_field: str, key: str
    ) -> None:
        short = commit[:length]
        record = {repository_field: repository, key: short}
        assert rulespec_commit_mentions({"source": record}) == {short}
        assert rulespec_commit_mentions({"rulespec": {key: short}}) == {short}
        # A value straight under a rulespec key, as a string or in a list.
        for rulespec_key in ("rulespec", f"rulespec_{key}"):
            assert rulespec_commit_mentions({rulespec_key: short}) == {short}
            assert rulespec_commit_mentions({rulespec_key: [short]}) == {short}

    @settings(max_examples=200, deadline=None)
    @given(digits=st.integers(10**6, 10**40 - 1))
    def test_an_all_digit_short_sha_that_yaml_reads_as_an_integer_counts(
        self, digits: int
    ) -> None:
        payload = load_yaml12(f"rulespec_nz_commit: {digits}\n", source="t.yaml")
        assert payload == {"rulespec_nz_commit": digits}
        assert rulespec_commit_mentions(payload) == {str(digits)}

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT, length=st.integers(7, 40))
    def test_a_hex_mapping_key_under_a_commit_key_is_a_mention(
        self, commit: str, length: int
    ) -> None:
        short = commit[:length]
        assert short in rulespec_commit_mentions({"reviewed_commits": {short: "main"}})

    @settings(max_examples=300, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        prefix=_KEY_AFFIX,
        suffix=_KEY_AFFIX,
        upper=st.booleans(),
        shape=st.sampled_from(["string", "list", "mapping"]),
        nest=st.integers(0, 3),
    )
    def test_hex_under_any_commit_key_outside_a_foreign_record_is_a_mention(
        self,
        commit: str,
        length: int,
        prefix: str,
        suffix: str,
        upper: bool,
        shape: str,
        nest: int,
    ) -> None:
        short = commit[:length]
        key = f"{prefix}{'COMMIT' if upper else 'commit'}{suffix}"
        value = {"string": short, "list": [short], "mapping": {"sha": short}}[shape]
        planted: object = {key: value}
        for depth in range(nest):
            planted = {"level": depth, "pins": [planted]}
        assert short in rulespec_commit_mentions(planted)

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT, prefix=_KEY_AFFIX, suffix=_KEY_AFFIX)
    def test_a_foreign_record_keeps_its_commits_unless_the_key_names_rulespec(
        self, commit: str, prefix: str, suffix: str
    ) -> None:
        key = f"{prefix}commit{suffix}"
        assume("rulespec" not in key)
        record = {"repository": "TheAxiomFoundation/ops", key: commit}
        assert rulespec_commit_mentions(record) == set()
        # It stays foreign under another commit key, but not under rulespec.
        assert rulespec_commit_mentions({"pinned_commits": [record]}) == set()
        assert rulespec_commit_mentions({"rulespec": record}) == {commit}
        record["rulespec_nz_commit"] = commit
        assert rulespec_commit_mentions(record) == {commit}


# ---------------------------------------------------------------------------
# Rules bindings and the reg 17/18 bridge
# ---------------------------------------------------------------------------


class TestRulesBindings:
    def test_bound_module_digests_match_the_committed_input_surface(self) -> None:
        # Differential: two independent records of the same module bytes.
        surface = _input_surface()
        for binding in _bindings().values():
            module = surface[binding["rulespec_path"]]
            assert module["status"] == "compiled"
            assert module["sha256"] == binding["sha256"]
            assert len(module["canonical_inputs"]) == binding["root_input_count"]

    def test_requested_variables_are_outputs_not_root_inputs(self) -> None:
        surface = _input_surface()
        for binding in _bindings().values():
            inputs = set(surface[binding["rulespec_path"]]["canonical_inputs"])
            variables = binding["variables"]
            assert variables and len(set(variables)) == len(variables)
            assert inputs.isdisjoint(variables)

    def test_bindings_declare_relation_free_modules_on_mapped_entities(self) -> None:
        document = _load("axiom_rules_bindings.json")
        entity_names = document["entity_names"]
        assert set(document["periods"]) == {"2026-27"}
        for binding in document["bindings"]:
            assert binding["relations_declared"] is False
            assert binding["imports_declared"] is False
            assert entity_names[binding["entity"]] == binding["engine_entity"]
            assert binding["period"] in document["periods"]
            assert binding["rulespec_path"].startswith("nz/")
        assert "nz.scn." not in json.dumps(document)
        period = document["periods"]["2026-27"]
        assert (period["kind"], period["start"], period["end"]) == (
            "tax_year",
            "2026-04-01",
            "2027-03-31",
        )

    def test_export_contract_excludes_exactly_the_bound_outputs(self) -> None:
        # Differential: the export gate's formula-owned list and the bindings.
        contract = _load("export_contract.json")
        bound = {v for binding in _bindings().values() for v in binding["variables"]}
        assert set(contract["formula_owned_excluded"]) == bound
        assert contract["closed"] is True
        assert set(contract["forbidden"]) == {"person_weight", "family_weight"}
        assert (
            contract["_period"]["label"]
            in _load("axiom_rules_bindings.json")["periods"]
        )

    def test_bridge_evaluates_only_bound_variables_under_real_input_names(self) -> None:
        bridge = _load("as_rate_bridge.json")
        bindings = _bindings()
        surface = _input_surface()
        assert set(bridge["rate_bindings"]) <= set(bindings)
        evaluations = []
        for output in bridge["outputs"]:
            evaluations += output.get("terms", [])
            for branch in output.get("branches", []):
                evaluations += branch["evaluations"]
        assert evaluations
        for evaluation in evaluations:
            binding = bindings[evaluation["binding"]]
            assert evaluation["binding"] in bridge["rate_bindings"]
            assert evaluation["variable"] in binding["variables"]
            assert evaluation["engine_entity"] == binding["engine_entity"]
            inputs = set(surface[binding["rulespec_path"]]["canonical_inputs"])
            overrides = [row["variable"] for row in evaluation["input_overrides"]]
            assert len(set(overrides)) == len(overrides)
            assert set(overrides) <= inputs

    def test_bridge_targets_are_the_two_as_inputs_the_module_leaves_open(self) -> None:
        bridge = _load("as_rate_bridge.json")
        surface = _input_surface()
        as_inputs = set(
            surface[_bindings()["accommodation_supplement"]["rulespec_path"]][
                "canonical_inputs"
            ]
        )
        targets = {output["target_input"] for output in bridge["outputs"]}
        assert targets == {
            "accommodation_supplement_base_rate_weekly_amount",
            "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount",
        }
        assert targets <= as_inputs
        (case,) = bridge["differential_expectations"]["cases"]
        assert case["request_id"] == "golden-08"
        assert set(case) >= targets

    def test_bridge_reads_only_unit_attributes_the_unit_rule_declares(self) -> None:
        declared = set(_load("benefit_unit_rule.json")["unit_attributes"]["columns"])
        bridge_text = json.dumps(_load("as_rate_bridge.json"))
        used = {
            f"family.{name}" for name in re.findall(r"family\.([a-z_]+)", bridge_text)
        }
        assert used
        assert used <= declared

    def test_reg17_family_tax_credit_term_uses_the_statutory_divisor(self) -> None:
        (base_rate,) = [
            output
            for output in _load("as_rate_bridge.json")["outputs"]
            if output["id"] == "reg17_base_rate"
        ]
        (ftc,) = [term for term in base_rate["terms"] if "divisor" in term]
        assert ftc["divisor"] == 52


# ---------------------------------------------------------------------------
# Crosswalk (MC9)
# ---------------------------------------------------------------------------


@st.composite
def _crosswalk_rows(draw) -> list[dict[str, Any]]:
    count = draw(st.integers(1, 8))
    rows = []
    for index in range(count):
        areas = draw(
            st.lists(
                st.sampled_from(sorted(AREAS)), min_size=1, max_size=4, unique=True
            )
        )
        weights = draw(
            st.lists(
                st.floats(1e-6, 1e6, allow_nan=False, allow_infinity=False),
                min_size=len(areas),
                max_size=len(areas),
            )
        )
        total = math.fsum(weights)
        rows.append(
            {
                "ta_code": f"{index:03d}",
                "shares": [
                    {"as_area": area, "population_share": weight / total}
                    for area, weight in zip(areas, weights, strict=True)
                ],
            }
        )
    return rows


# Each breaks one valid crosswalk row in one way and returns the exact error
# crosswalk_errors must report. The share-sum branch has its own drift test.
_Breaker = Callable[[list[dict[str, Any]], st.DataObject], str]
_MISSING = object()


def _pick_row(rows: list[dict[str, Any]], data: st.DataObject) -> dict[str, Any]:
    return rows[data.draw(st.integers(0, len(rows) - 1))]


def _blank_ta_code(rows, data) -> str:
    row = _pick_row(rows, data)
    blank = data.draw(st.sampled_from([_MISSING, "", None, 0, 42]))
    if blank is _MISSING:
        del row["ta_code"]
    else:
        row["ta_code"] = blank
    return "row without ta_code"


def _duplicate_ta(rows, data) -> str:
    row = _pick_row(rows, data)
    rows.insert(data.draw(st.integers(0, len(rows))), copy.deepcopy(row))
    return f"{row['ta_code']}: duplicate territorial authority"


def _empty_shares(rows, data) -> str:
    row = _pick_row(rows, data)
    shares = data.draw(st.sampled_from([_MISSING, [], None, {}, "1", 1]))
    if shares is _MISSING:
        del row["shares"]
    else:
        row["shares"] = shares
    return f"{row['ta_code']}: no shares"


def _non_integer_area(rows, data) -> str:
    row = _pick_row(rows, data)
    share = data.draw(st.sampled_from(row["shares"]))
    share["as_area"] = data.draw(st.sampled_from([True, False, None, "1", 1.0, 2.5]))
    return f"{row['ta_code']}: area must be an integer"


def _repeated_area(rows, data) -> str:
    row = _pick_row(rows, data)
    index = data.draw(st.integers(0, len(row["shares"]) - 1))
    share = row["shares"][index]
    half = share["population_share"] / 2
    # Split one share in two under the same area, so the sum still holds.
    row["shares"][index : index + 1] = [
        {**share, "population_share": half},
        {**share, "population_share": share["population_share"] - half},
    ]
    return f"{row['ta_code']}: repeated area"


def _area_outside_one_to_four(rows, data) -> str:
    row = _pick_row(rows, data)
    share = data.draw(st.sampled_from(row["shares"]))
    share["as_area"] = data.draw(st.integers().filter(lambda area: area not in AREAS))
    return f"{row['ta_code']}: area outside 1-4"


def _share_outside_unit_interval(rows, data) -> str:
    row = _pick_row(rows, data)
    share = data.draw(st.sampled_from(row["shares"]))
    share["population_share"] = data.draw(
        st.one_of(
            st.floats(max_value=0, exclude_max=True),
            st.floats(min_value=1, exclude_min=True),
            st.sampled_from([math.nan, None, "0.5", True]),
        )
    )
    return f"{row['ta_code']}: share outside [0, 1]"


CROSSWALK_BREAKERS: dict[str, _Breaker] = {
    "row without ta_code": _blank_ta_code,
    "duplicate territorial authority": _duplicate_ta,
    "no shares": _empty_shares,
    "area must be an integer": _non_integer_area,
    "repeated area": _repeated_area,
    "area outside 1-4": _area_outside_one_to_four,
    "share outside [0, 1]": _share_outside_unit_interval,
}


class TestCrosswalk:
    def test_committed_rows_satisfy_the_share_contract(self) -> None:
        document = _load("as_area_crosswalk.json")
        assert crosswalk_errors(document["rows"]) == []
        assert document["method"]["share_tolerance"] == SHARE_TOLERANCE
        # An empty crosswalk can only ship as an explicit harvest item.
        assert (document["status"] == "requires_harvest") == (not document["rows"])

    def test_area_definitions_are_cited_and_list_each_unit_once(self) -> None:
        definitions = _load("as_area_crosswalk.json")["area_definitions"]
        assert definitions["source"]["url"].startswith(
            "https://www.workandincome.govt.nz/"
        )
        assert definitions["source"]["retrieved_on"]
        assert "residual" in definitions["areas"]["4"]
        listed = [
            name
            for area in ("1", "2", "3")
            for location in definitions["areas"][area]["locations"]
            for name in location["statistical_area_units_2017"]
        ]
        assert listed
        assert len(set(listed)) == len(listed)

    @settings(max_examples=200, deadline=None)
    @given(_crosswalk_rows())
    def test_normalised_population_shares_pass(self, rows) -> None:
        assert crosswalk_errors(rows) == []

    @settings(max_examples=200, deadline=None)
    @given(_crosswalk_rows(), st.floats(1e-7, 0.5), st.booleans())
    def test_any_share_drift_beyond_tolerance_fails(self, rows, drift, upward) -> None:
        share = rows[0]["shares"][0]
        moved = share["population_share"] + (drift if upward else -drift)
        if not 0 <= moved <= 1:
            moved = share["population_share"] - (drift if upward else -drift)
        if not 0 <= moved <= 1:
            moved = share["population_share"] / 2 if share["population_share"] else 0.5
        share["population_share"] = moved
        total = math.fsum(item["population_share"] for item in rows[0]["shares"])
        assert abs(total - 1.0) > SHARE_TOLERANCE
        assert any(error.startswith("000:") for error in crosswalk_errors(rows))

    @settings(max_examples=50, deadline=None)
    @given(_crosswalk_rows())
    def test_boolean_areas_are_refused(self, rows) -> None:
        rows[0]["shares"][0]["as_area"] = True
        assert any("integer" in error for error in crosswalk_errors(rows))

    @pytest.mark.parametrize("branch", sorted(CROSSWALK_BREAKERS))
    @settings(max_examples=100, deadline=None)
    @given(rows=_crosswalk_rows(), data=st.data())
    def test_every_refusal_branch_names_the_broken_row(
        self, branch: str, rows, data
    ) -> None:
        expected = CROSSWALK_BREAKERS[branch](rows, data)
        assert expected in crosswalk_errors(rows)

    @settings(max_examples=200, deadline=None)
    @given(data=st.data())
    def test_a_tied_modal_area_refuses(self, data) -> None:
        areas = data.draw(
            st.lists(
                st.sampled_from(sorted(AREAS)), min_size=2, max_size=4, unique=True
            )
        )
        weights = data.draw(
            st.lists(st.floats(1e-6, 1e6), min_size=len(areas), max_size=len(areas))
        )
        first, second = data.draw(
            st.lists(
                st.integers(0, len(areas) - 1), min_size=2, max_size=2, unique=True
            )
        )
        weights[first] = weights[second] = max(weights)
        total = math.fsum(weights)
        shares = [
            {"as_area": area, "population_share": weight / total}
            for area, weight in zip(areas, weights, strict=True)
        ]
        # A tie is a valid crosswalk row; only the S3 alternative refuses it.
        assert crosswalk_errors([{"ta_code": "001", "shares": shares}]) == []
        with pytest.raises(ValueError, match="tied modal areas"):
            modal_area(shares)

    @settings(max_examples=200, deadline=None)
    @given(_crosswalk_rows())
    def test_modal_alternative_is_a_listed_area_or_refuses(self, rows) -> None:
        for row in rows:
            try:
                area = modal_area(row["shares"])
            except ValueError:
                continue
            assert area in {share["as_area"] for share in row["shares"]}
            assert (
                max(row["shares"], key=lambda s: s["population_share"])["as_area"]
                == area
            )


# ---------------------------------------------------------------------------
# Scenarios (§2.4, MC3a)
# ---------------------------------------------------------------------------


# Each breaks the committed grid in one way and returns the exact error
# scenario_errors must report. The method-card-row, tier and two-knob branches
# have their own tests below.
_ScenarioBreaker = Callable[[dict[str, Any], st.DataObject], str]


def _changed_scenario(document, data) -> dict[str, Any]:
    return data.draw(st.sampled_from([s for s in document["scenarios"] if s["knobs"]]))


def _open_knobs(document) -> list[str]:
    """Knobs whose domain is a description rather than a list of values."""

    return sorted(
        knob
        for knob, domain in document["knob_domains"].items()
        if not isinstance(domain, list)
    )


def _duplicate_scenario_id(document, data) -> str:
    first, second = data.draw(
        st.lists(
            st.integers(0, len(document["scenarios"]) - 1),
            min_size=2,
            max_size=2,
            unique=True,
        )
    )
    document["scenarios"][second]["id"] = document["scenarios"][first]["id"]
    return "duplicate scenario ids"


def _knobs_not_a_mapping(document, data) -> str:
    scenario = data.draw(st.sampled_from(document["scenarios"]))
    scenario["knobs"] = data.draw(
        st.sampled_from([None, [], ["asset_test"], "asset_test", 1])
    )
    return f"{scenario['id']}: knobs must be a mapping"


def _central_scenario_off_the_grid_row(document, data) -> str:
    scenario = data.draw(st.sampled_from(document["scenarios"]))
    scenario["knobs"] = {}
    scenario["method_card_row"] = data.draw(
        st.sampled_from(
            [row for row in METHOD_CARD_ROWS if row != document["method_card_row"]]
        )
    )
    return f"{scenario['id']}: the central scenario names the grid row"


def _unknown_knob(document, data) -> str:
    scenario = _changed_scenario(document, data)
    (value,) = scenario["knobs"].values()
    name = data.draw(_IDENT.filter(lambda name: name not in document["central_knobs"]))
    scenario["knobs"] = {name: value}
    return f"{scenario['id']}: unknown knob {name!r}"


def _knob_at_its_central_value(document, data) -> str:
    scenario = _changed_scenario(document, data)
    (knob,) = scenario["knobs"]
    scenario["knobs"][knob] = document["central_knobs"][knob]
    return f"{scenario['id']}: knob {knob!r} equals the central value"


def _knob_under_another_method_card_row(document, data) -> str:
    scenario = _changed_scenario(document, data)
    (knob,) = scenario["knobs"]
    owner = document["knob_method_card_rows"][knob]
    scenario["method_card_row"] = data.draw(
        st.sampled_from([row for row in METHOD_CARD_ROWS if row != owner])
    )
    return f"{scenario['id']}: knob {knob!r} belongs to {owner}"


def _value_outside_a_listed_domain(document, data) -> str:
    scenario = _changed_scenario(document, data)
    listed = sorted(set(document["knob_domains"]) - set(_open_knobs(document)))
    knob = data.draw(st.sampled_from(listed))
    domain = document["knob_domains"][knob]
    value = data.draw(
        st.one_of(st.text(max_size=16), st.integers(), st.none(), st.booleans()).filter(
            lambda value: value not in domain
        )
    )
    scenario["knobs"] = {knob: value}
    scenario["method_card_row"] = document["knob_method_card_rows"][knob]
    return f"{scenario['id']}: {value!r} outside the {knob!r} domain"


def _pending_value_without_a_status(document, data) -> str:
    scenario = _changed_scenario(document, data)
    knob = data.draw(st.sampled_from(_open_knobs(document)))
    scenario["knobs"] = {knob: None}
    scenario.pop("status", None)
    return f"{scenario['id']}: a pending value needs a status"


def _open_knob_not_a_positive_number(document, data) -> str:
    scenario = _changed_scenario(document, data)
    knob = data.draw(st.sampled_from(_open_knobs(document)))
    scenario["knobs"] = {
        knob: data.draw(
            st.one_of(
                st.floats(max_value=0),
                st.integers(max_value=0),
                st.sampled_from([math.inf, math.nan, True, "1.5", [1.5]]),
            )
        )
    }
    return f"{scenario['id']}: {knob!r} needs a positive number"


SCENARIO_BREAKERS: dict[str, _ScenarioBreaker] = {
    "duplicate scenario ids": _duplicate_scenario_id,
    "knobs must be a mapping": _knobs_not_a_mapping,
    "the central scenario names the grid row": _central_scenario_off_the_grid_row,
    "unknown knob": _unknown_knob,
    "equals the central value": _knob_at_its_central_value,
    "belongs to another method-card row": _knob_under_another_method_card_row,
    "outside a listed domain": _value_outside_a_listed_domain,
    "a pending value needs a status": _pending_value_without_a_status,
    "needs a positive number": _open_knob_not_a_positive_number,
}


class TestScenarios:
    def test_committed_grid_satisfies_the_scenario_contract(self) -> None:
        document = _load("scenarios.json")
        assert scenario_errors(document) == []
        assert [scenario["id"] for scenario in document["scenarios"]] == [
            "S0",
            "S1",
            "S2",
            "S3",
            "S5",
            "V1",
            "V2",
            "V3",
            "V4",
        ]
        branch_points = {
            tier: row["branch_point"] for tier, row in document["tiers"].items()
        }
        assert branch_points == {"entitlement": "nz.as", "calibration": "nz.open"}
        assert [s["id"] for s in document["scenarios"] if not s["knobs"]] == ["S0"]

    def test_input_overrides_name_real_accommodation_supplement_inputs(self) -> None:
        surface = _input_surface()
        as_inputs = set(
            surface[_bindings()["accommodation_supplement"]["rulespec_path"]][
                "canonical_inputs"
            ]
        )
        for scenario in _load("scenarios.json")["scenarios"]:
            for override in scenario.get("input_overrides", []):
                assert override["variable"] in as_inputs

    def test_annualisation_is_the_declared_mc12_convention(self) -> None:
        conventions = _load("scenarios.json")["gap_conventions"]
        assert conventions["annualisation"]["weeks_per_year"] == 52
        assert conventions["annualisation"]["method_card_row"] == "MC12"

    @settings(max_examples=200, deadline=None)
    @given(st.data())
    def test_a_scenario_with_two_or_no_method_card_rows_is_refused(self, data) -> None:
        document = _load("scenarios.json")
        index = data.draw(st.integers(0, len(document["scenarios"]) - 1))
        bad_row = data.draw(
            st.sampled_from(["", "MC", "MC0", "MC16", "MC1, MC5", "MC1 MC5", "mc3"])
        )
        document["scenarios"][index]["method_card_row"] = bad_row
        assert any(
            "exactly one method-card row" in error
            for error in scenario_errors(document)
        )

    @settings(max_examples=200, deadline=None)
    @given(st.data())
    def test_a_scenario_outside_the_two_tiers_is_refused(self, data) -> None:
        document = _load("scenarios.json")
        index = data.draw(st.integers(0, len(document["scenarios"]) - 1))
        tier = data.draw(st.text(max_size=12).filter(lambda value: value not in TIERS))
        document["scenarios"][index]["tier"] = tier
        assert any("tier outside" in error for error in scenario_errors(document))

    @settings(max_examples=200, deadline=None)
    @given(st.data())
    def test_a_scenario_changing_two_knobs_is_refused(self, data) -> None:
        document = _load("scenarios.json")
        changed = [s for s in document["scenarios"] if s["knobs"]]
        scenario = data.draw(st.sampled_from(changed))
        (knob,) = scenario["knobs"]
        other = data.draw(
            st.sampled_from(sorted(set(document["central_knobs"]) - {knob}))
        )
        scenario["knobs"][other] = document["central_knobs"][other]
        assert any("exactly one knob" in error for error in scenario_errors(document))

    @pytest.mark.parametrize("branch", sorted(SCENARIO_BREAKERS))
    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_every_refusal_branch_names_the_broken_scenario(
        self, branch: str, data
    ) -> None:
        document = _load("scenarios.json")
        expected = SCENARIO_BREAKERS[branch](document, data)
        assert expected in scenario_errors(document)


# ---------------------------------------------------------------------------
# Source citations
# ---------------------------------------------------------------------------


_SOURCE_STEM = st.from_regex(r"[a-z_][a-z0-9_]{0,15}", fullmatch=True)
_SOURCE_EXTENSION = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9_]{0,7}", fullmatch=True)
_SOURCE_DIRECTORY = st.sampled_from(
    (
        "",
        "packages/microcosm-frame/src/microcosm/frame/",
        "microcosm/graph/",
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
    "{file}:L{a}",
    "{file}：L{a}-L{b}",
    "{file} @ L{a}",
    "{file}, L{a}-L{b}",
    "{file} (L{a}-L{b})",
    "{file} [L{a}-L{b}]",
    "{file} lines L{a}-L{b}",
    "{file} ll.{a}-{b}",
    "{file} at line {a}",
    "{file} on lines {a}-{b}",
    "lines L{a}-L{b} of {file}",
    "L{a}-L{b} in {file}",
    "ll.{a}-{b} in {file}",
    "lines {a}, {b} in {file}",
)
_FILE_WRAPPERS = (
    "{file}",
    "`{file}`",
    "'{file}'",
    '"{file}"',
    "‘{file}’",
    "“{file}”",
    "[{file}](source)",
    "[`{file}`](source)",
    "[{file}][source]",
)
_LINE_LOCATOR_SPELLINGS = st.builds(
    lambda locator, wrapper, dash: locator.replace("-", dash).replace(
        "{file}", wrapper
    ),
    st.sampled_from(_LINE_LOCATORS),
    st.sampled_from(_FILE_WRAPPERS),
    st.sampled_from(("-", "–", "—", " to ")),
)
#: Symbol-citation spellings, which carry no line number.
_SYMBOL_LOCATORS = (
    "{file} {symbol}",
    '{file} _pointer("{symbol}")',
    '{file}, the "# --- Relationships" block',
    "{file}: {symbol}",
    "{symbol} in {file} at main",
)
#: Prose around a citation. It has no digit, so it can neither extend a
#: planted line number nor plant a citation of its own.
_PROSE = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,;()'\"-_",
    max_size=40,
)
_BRANCH_NAMES = (
    "main",
    "HEAD",
    "launch/ga-path-execution-gate",
    "ba6e7d",
    "BA6E7D6",
    "",
)
_SOURCE_URL_FORMS = (
    "https://github.com/{repository}/blob/{ref}/{path}",
    "http://github.com/{repository}/raw/{ref}/{path}",
    "https://raw.githubusercontent.com/{repository}/{ref}/{path}",
)


def _plant(text: str, depth: int, as_key: bool) -> object:
    planted: object = {text: "cited"} if as_key else text
    for level in range(depth):
        planted = {"level": level, "note": [planted]}
    return planted


@pytest.fixture
def citation_checkout(tmp_path, monkeypatch):
    """An isolated file lookup; each write replaces that scratch JSON payload."""

    files: set[str] = set()

    def write(file: str, payload: object) -> None:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        files.add(file)
        monkeypatch.setattr(__name__ + "._BUILD", SimpleNamespace(repository=tmp_path))
        monkeypatch.setattr(__name__ + "._repository_files", lambda: files)

    return write


class TestSourceCitations:
    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    def test_every_documented_in_scope_spelling_is_refused(self, as_key: bool) -> None:
        assert __doc__ is not None
        examples = __doc__.split("In-scope examples:\n", 1)[1].split("\n\n", 1)[0]
        spellings = [line.strip() for line in examples.splitlines()]
        assert spellings
        for spelling in spellings:
            citation = spelling.replace("file", "concepts.py")
            assert unpinned_line_citations(_plant(citation, 3, as_key)), citation

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize(
        ("citation", "files", "exempt"),
        [
            ("[the source](concepts.py) lines 889-928", {"concepts.py"}, False),
            ("lines 889-928 of [the source](concepts.py)", {"concepts.py"}, False),
            (
                "[upstream.rs](concepts.py) lines 889-928",
                {"upstream.rs", "concepts.py"},
                False,
            ),
            (
                "lines 889-928 of [upstream.rs](concepts.py)",
                {"upstream.rs", "concepts.py"},
                False,
            ),
            ("[upstream.rs](upstream.rs) lines 889-928", {"upstream.rs"}, True),
            ("[the source](concepts.py#L12)", {"concepts.py"}, False),
            ("[the source](concepts.py:L12)", {"concepts.py"}, False),
            ("[upstream.rs](concepts.py#L12)", {"upstream.rs", "concepts.py"}, False),
            ("[the source](concepts.py#heading) L12", {"concepts.py"}, False),
            ("[the source](concepts.py?raw=1) L12", {"concepts.py"}, False),
            ("[the source](concepts.py?raw=1#L12)", {"concepts.py"}, False),
            ("![x](concepts.py) L5", {"concepts.py"}, False),
            (
                "[concepts.py:L12](upstream.rs)",
                {"concepts.py", "upstream.rs"},
                False,
            ),
            (
                "[upstream.rs lines 12-20](concepts.py)",
                {"concepts.py", "upstream.rs"},
                False,
            ),
            (
                "[concepts.py and upstream.rs](other.rs) L12",
                {"concepts.py", "upstream.rs", "other.rs"},
                False,
            ),
            (
                "[upstream.rs](other.rs) L12",
                {"upstream.rs", "other.rs"},
                False,
            ),
        ],
    )
    def test_markdown_link_lines_are_attributed_to_every_named_file(
        self, citation: str, files: set[str], exempt: bool, as_key: bool
    ) -> None:
        document = {
            "source": {"path": "upstream.rs", "commit": "abcdef0"},
            "notes": _plant(citation, 3, as_key),
        }
        assert set(source_line_citations(document)) == {
            (file, citation) for file in files
        }
        assert (unpinned_line_citations(document) == []) is exempt

    @pytest.mark.parametrize("url_form", [None, *_SOURCE_URL_FORMS])
    @settings(max_examples=200, deadline=None)
    @given(
        label=st.sampled_from(("concepts.py", "the source", "upstream.rs")),
        destination=st.sampled_from(("concepts.py", "upstream.rs", "other.rs")),
        reverse=st.booleans(),
        depth=st.integers(0, 4),
        as_key=st.booleans(),
        fragment=st.sampled_from(("", "#L12", "#L12-L20", "#L12C3-L20C9")),
        query=st.sampled_from(("", "?raw=1")),
    )
    def test_markdown_link_exemption_requires_every_file_to_be_pinned_and_foreign(
        self,
        url_form: str | None,
        label: str,
        destination: str,
        reverse: bool,
        depth: int,
        as_key: bool,
        fragment: str,
        query: str,
    ) -> None:
        path = destination
        if url_form is not None:
            if destination == "concepts.py":
                path = "packages/microcosm-frame/src/microcosm/frame/concepts.py"
            destination = url_form.format(
                repository="PolicyEngine/microcosm"
                if destination == "concepts.py"
                else "Other/repository",
                ref="a" * 40,
                path=path,
            )
        link = f"[{label}]({destination}{query}{fragment})"
        citation = f"lines 12-20 of {link}" if reverse else f"{link} lines 12-20"
        # Even an explicit pin for the local file cannot exempt it.
        document = {
            "pins": [
                {"path": file, "commit": "abcdef0"}
                for file in ("concepts.py", path, "upstream.rs")
                if file != "other.rs"
            ],
            "notes": _plant(citation, depth, as_key),
        }
        attributed = {path} | ({label} if label != "the source" else set())
        assert set(source_line_citations(document)) == {
            (file, citation) for file in attributed
        }
        assert (unpinned_line_citations(document) == []) is (
            attributed <= {"upstream.rs"}
        )

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize("url_form", _SOURCE_URL_FORMS)
    @pytest.mark.parametrize("ref", ["main", "feature/citations", "a" * 40])
    @pytest.mark.parametrize(
        "spelling",
        [
            "[the source]({url}#L889-L928)",
            "[upstream.rs]({url}#L889-L928)",
            "{url}#L889-L928",
            "[x]({url}) lines 889-928",
            "lines 889-928 of [x]({url})",
            "[upstream.rs]({url}) lines 889-928",
            "lines 889-928 of [upstream.rs]({url})",
            "{url} lines 889-928",
            "lines 889-928 of {url}",
            "[the source]({url}#L12)",
            "{url}#L12",
            "[the source]({url}#L12C3-L20C9)",
            "{url}#L12C3-L20C9",
        ],
    )
    def test_this_repository_url_lines_are_never_exempt(
        self, url_form: str, ref: str, spelling: str, as_key: bool
    ) -> None:
        path = "packages/microcosm-frame/src/microcosm/frame/concepts.py"
        url = url_form.format(repository="PolicyEngine/microcosm", ref=ref, path=path)
        citation = spelling.format(url=url)
        document = {
            "pins": [
                {"path": file, "commit": "abcdef0"} for file in (path, "upstream.rs")
            ],
            "notes": _plant(citation, 3, as_key),
        }
        expected = {path} | ({"upstream.rs"} if "[upstream.rs]" in citation else set())
        assert set(source_line_citations(document)) == {
            (file, citation) for file in expected
        }
        assert set(unpinned_line_citations(document)) == {
            (file, citation) for file in expected
        }

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize("url_form", _SOURCE_URL_FORMS)
    @pytest.mark.parametrize("pin", [None, "concepts.py", "other.rs"])
    @pytest.mark.parametrize("bare", [False, True], ids=["link", "bare"])
    def test_foreign_url_lines_require_the_same_resource_file_pin(
        self, url_form: str, pin: str | None, bare: bool, as_key: bool
    ) -> None:
        # A foreign file remains foreign even when its name exists locally.
        url = url_form.format(
            repository="Other/repository", ref="a" * 40, path="concepts.py"
        )
        citation = f"{url}#L12" if bare else f"[the source]({url}#L12)"
        document = {"notes": _plant(citation, 3, as_key)}
        if pin is not None:
            document["source"] = {"path": pin, "commit": "abcdef0"}
        assert source_line_citations(document) == [("concepts.py", citation)]
        assert unpinned_line_citations(document) == (
            [] if pin == "concepts.py" else [("concepts.py", citation)]
        )

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize("url_form", _SOURCE_URL_FORMS)
    @pytest.mark.parametrize("bare", [False, True], ids=["link", "bare"])
    @pytest.mark.parametrize(
        ("path", "pin"),
        [("upstream.rs", "src/upstream.rs"), ("src/upstream.rs", "upstream.rs")],
    )
    def test_url_files_cannot_borrow_a_different_directory_pin(
        self, url_form: str, path: str, pin: str, bare: bool, as_key: bool
    ) -> None:
        url = url_form.format(repository="Other/repository", ref="main", path=path)
        citation = f"{url}#L12" if bare else f"[the source]({url}#L12)"
        document = {
            "source": {"path": pin, "commit": "abcdef0"},
            "notes": _plant(citation, 3, as_key),
        }
        assert source_line_citations(document) == [(path, citation)]
        assert unpinned_line_citations(document) == [(path, citation)]
        document["source"]["path"] = path
        assert unpinned_line_citations(document) == []

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize(
        "punctuation",
        list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        + [
            "…",
            "—",
            "–",
            "‐",
            "‑",
            "‒",
            "―",
            "。",
            "！",
            "？",
            "：",
            "；",
            "’",
            "”",
            "»",
            "】",
        ],
    )
    def test_url_line_fragments_survive_trailing_punctuation(
        self, punctuation: str, as_key: bool
    ) -> None:
        path = "packages/microcosm-frame/src/microcosm/frame/concepts.py"
        citation = (
            f"See https://github.com/PolicyEngine/microcosm/blob/main/{path}"
            f"#L12{punctuation} this defines the field."
        )
        assert unpinned_line_citations(_plant(citation, 3, as_key)) == [
            (path, citation)
        ]

    @settings(max_examples=300, deadline=None)
    @given(
        url_form=st.sampled_from(_SOURCE_URL_FORMS),
        stem=_SOURCE_STEM,
        extension=_SOURCE_EXTENSION,
        directory=_SOURCE_DIRECTORY,
        ref=st.sampled_from(("main", "feature/citations", "a" * 40)),
        query=st.sampled_from(("", "?plain=1")),
        fragment=st.sampled_from(
            ("L5", "L5-L10", "L5-10", "l5", "l5-l10", "L5C2-L10C3")
        ),
        punctuation=st.text(
            st.characters(categories=("P", "S")), min_size=1, max_size=8
        ),
        suffix=st.one_of(st.just(""), _PROSE.map(lambda text: " " + text)),
        depth=st.integers(0, 4),
        as_key=st.booleans(),
    )
    def test_local_url_line_fragments_are_refused_with_any_punctuation(
        self,
        url_form: str,
        stem: str,
        extension: str,
        directory: str,
        ref: str,
        query: str,
        fragment: str,
        punctuation: str,
        suffix: str,
        depth: int,
        as_key: bool,
    ) -> None:
        path = f"{directory}{stem}.{extension}"
        url = url_form.format(repository="PolicyEngine/microcosm", ref=ref, path=path)
        citation = f"{url}{query}#{fragment}{punctuation}{suffix}"
        document = {
            "source": {"path": path, "commit": "abcdef0"},
            "notes": _plant(citation, depth, as_key),
        }
        assert unpinned_line_citations(document), citation

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize(
        "spelling",
        [
            "({url}#L5)",
            "'{url}#L5'",
            '"{url}#L5"',
            "‘{url}#L5’",
            "“{url}#L5”",
            "<{url}#L5>",
            "[x][1]\n\n[1]: {url}#L5",
            '<a href="{url}#L5">source</a>',
            "{url}?plain=1#L5",
            "[source]({url}?plain=1#L5)",
            "{url}#L5-10",
            "[source]({url}#L5-10)",
            "{url}#l5",
            "[source]({url}#l5)",
            "{url}#l5-l10",
            "{url}#l5c2-l10c3",
        ],
    )
    def test_authored_url_wrappers_and_fragment_spellings_are_refused(
        self, spelling: str, as_key: bool
    ) -> None:
        path = "packages/microcosm-frame/src/microcosm/frame/concepts.py"
        url = f"https://github.com/PolicyEngine/microcosm/blob/main/{path}"
        citation = spelling.format(url=url)
        assert unpinned_line_citations(_plant(citation, 3, as_key)) == [
            (path, citation)
        ]

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize("route", ["blob", "raw", "tree", "blame"])
    @pytest.mark.parametrize("bare", [False, True], ids=["link", "bare"])
    @pytest.mark.parametrize("encoded", [False, True], ids=["plain", "percent-encoded"])
    def test_url_paths_keep_repository_identity_at_any_commit(
        self, route: str, bare: bool, encoded: bool, as_key: bool
    ) -> None:
        # Use an absent file so checkout suffix lookup cannot supply identity.
        path = "retired/source.rs"
        url_path = "retired%2Fsource%2Ers" if encoded else path
        url = f"https://github.com/PolicyEngine/microcosm/{route}/{'a' * 40}/{url_path}#L5"
        citation = url if bare else f"[source]({url})"
        document = {
            "source": {"path": path, "commit": "abcdef0"},
            "notes": _plant(citation, 3, as_key),
        }
        assert unpinned_line_citations(document) == [(path, citation)]

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize("bare", [False, True], ids=["link", "bare"])
    @pytest.mark.parametrize(
        "fragment",
        ["L12?plain=1", "L12:see-field", "l12!explanation", "l12—explanation"],
    )
    def test_ununderstood_url_fragments_remain_available_to_raw_scanning(
        self, fragment: str, bare: bool, as_key: bool
    ) -> None:
        url = f"https://github.com/PolicyEngine/microcosm/blob/main/concepts.py#{fragment}"
        citation = url if bare else f"[source]({url})"
        assert unpinned_line_citations(_plant(citation, 3, as_key))

    @pytest.mark.parametrize(
        "host_path",
        [
            "example.org/src/upstream.rs",
            "gitlab.com/Other/repository/-/blob/main/src/upstream.rs",
        ],
    )
    def test_other_hosts_accept_short_ranges_but_require_an_exact_pin(
        self, host_path: str
    ) -> None:
        path = host_path.split("/", 1)[1]
        citation = f"https://{host_path}#L5-10"
        document = {"source": {"path": path, "commit": "abcdef0"}, "note": citation}
        assert source_line_citations(document) == [(path, citation)]
        assert unpinned_line_citations(document) == []
        document["source"]["path"] = "upstream.rs"
        assert unpinned_line_citations(document) == [(path, citation)]

    @pytest.mark.parametrize(
        "pin", ["./src/upstream.rs", "src/upstream.rs/", "./src/upstream.rs/"]
    )
    def test_url_pins_normalize_dot_prefixes_but_refuse_directories(
        self, pin: str
    ) -> None:
        citation = "https://github.com/Other/repository/blob/main/src/upstream.rs#L5"
        document = {"source": {"path": pin, "commit": "abcdef0"}, "note": citation}
        assert unpinned_line_citations(document) == (
            [("src/upstream.rs", citation)] if pin.endswith("/") else []
        )

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize("bare", [False, True], ids=["link", "bare"])
    @pytest.mark.parametrize(
        "repository", ["PolicyEngine/microcosm", "Other/repository"]
    )
    @pytest.mark.parametrize(
        ("encoded", "decoded", "wrong_pin"),
        [
            ("weird%20upstream.rs", "weird upstream.rs", "20upstream.rs"),
            ("weird%5Bupstream.rs", "weird[upstream.rs", "5Bupstream.rs"),
            ("%E2%9C%93upstream.rs", "✓upstream.rs", "93upstream.rs"),
        ],
    )
    def test_encoded_url_filenames_cannot_borrow_a_prose_suffix_pin(
        self,
        encoded: str,
        decoded: str,
        wrong_pin: str,
        repository: str,
        bare: bool,
        as_key: bool,
    ) -> None:
        url = f"https://github.com/{repository}/blob/main/{encoded}#L5"
        citation = url if bare else f"[source]({url})"
        document = {
            "source": {"path": wrong_pin, "commit": "abcdef0"},
            "notes": _plant(citation, 3, as_key),
        }
        assert source_line_citations(document) == [(decoded, citation)]
        assert unpinned_line_citations(document) == [(decoded, citation)]

    @pytest.mark.parametrize("punctuation", list(".,;!?:") + ["…", "—"])
    def test_bare_url_file_punctuation_keeps_independent_lines_visible(
        self, punctuation: str
    ) -> None:
        path = "packages/microcosm-frame/src/microcosm/frame/concepts.py"
        url = f"https://github.com/PolicyEngine/microcosm/blob/main/{path}{punctuation}"
        assert source_line_citations(url) == []
        for source in (url, f"({url})", f"<{url}>"):
            for citation in (f"{source} L99", f"L99 in {source}"):
                assert unpinned_line_citations(citation) == [(path, citation)]

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize(
        "citation",
        [
            "[the source](not a path concepts.py#L12)",
            "[the source](https://[broken concepts.py#L12)",
        ],
    )
    def test_unparseable_link_destinations_cannot_hide_line_citations(
        self, citation: str, as_key: bool
    ) -> None:
        assert ("concepts.py", citation) in unpinned_line_citations(
            _plant(citation, 3, as_key)
        )

    @pytest.mark.parametrize("url_form", _SOURCE_URL_FORMS)
    def test_repository_urls_keep_identity_for_files_absent_from_the_checkout(
        self, url_form: str
    ) -> None:
        path = "retired/source.rs"
        url = url_form.format(
            repository="policyengine/MICROCOSM", ref="a" * 40, path=path
        )
        citation = f"[the source]({url}#L12)"
        document = {"source": {"path": path, "commit": "abcdef0"}, "note": citation}
        assert unpinned_line_citations(document) == [(path, citation)]

    @pytest.mark.parametrize(
        "fragment",
        [
            "heading12",
            "L12_symbol",
            "/key/concepts.py:L12",
            "heading!",
            "heading:",
            "heading…",
            "heading—",
        ],
    )
    def test_url_stable_fragments_keep_independent_line_locators_visible(
        self, fragment: str
    ) -> None:
        path = "packages/microcosm-frame/src/microcosm/frame/concepts.py"
        url = f"https://github.com/PolicyEngine/microcosm/blob/main/{path}?raw=1#{fragment}"
        for source in (
            url,
            f"[the source]({url})",
            f"({url})",
            f"<{url}>",
            f"‘{url}’",
            f"“{url}”",
        ):
            assert source_line_citations(source) == []
            for citation in (f"{source} L99", f"L99 in {source}"):
                assert unpinned_line_citations(citation) == [(path, citation)]

    @pytest.mark.parametrize("scheme", ["http", "https"])
    def test_other_http_file_urls_use_the_existing_foreign_pin_rule(
        self, scheme: str
    ) -> None:
        path = "src/upstream.rs"
        url = f"{scheme}://example.org/{path}?raw=1#L12C3-L20C9"
        for citation in (url, f"[the source]({url})"):
            document = {"source": {"path": path, "commit": "abcdef0"}, "note": citation}
            assert source_line_citations(document) == [(path, citation)]
            assert unpinned_line_citations(document) == []
            assert unpinned_line_citations(citation) == [(path, citation)]

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize("selector", [False, True], ids=["pointer", "selector"])
    @pytest.mark.parametrize("token", ["concepts.py:L12", "concepts.py#L12"])
    def test_json_pointer_and_selector_fragments_are_opaque_to_line_scanning(
        self, citation_checkout, token: str, selector: bool, as_key: bool
    ) -> None:
        record = {"input": token, "value": "stable record"}
        citation_checkout("records.json", {"entries": [record], token: record})
        pointer = f"/entries[input={token}]" if selector else f"/{token}"
        citation = f"records.json#{pointer}"
        document = _plant(citation, 3, as_key)
        assert json_pointer_citations(document) == [("records.json", pointer)]
        assert resolve_json_pointer("records.json", pointer) == record
        assert source_line_citations(document) == []
        assert unpinned_line_citations(document) == []
        # Masking must leave independent prose visible and retain the input text.
        prose = f"{citation}; concepts.py:L99"
        assert unpinned_line_citations(_plant(prose, 3, as_key)) == [
            ("concepts.py", prose)
        ]
        # A stable destination fragment cannot hide a locator outside its link.
        for linked in (
            f"[the source]({citation}) L99",
            f"L99 in [the source]({citation})",
        ):
            assert source_line_citations(_plant(linked, 3, as_key)) == [
                ("records.json", linked)
            ]
            assert unpinned_line_citations(_plant(linked, 3, as_key)) == [
                ("records.json", linked)
            ]

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize("source", ["concepts.py", "LICENSE"])
    @pytest.mark.parametrize(
        "spelling",
        [
            pytest.param("`{file}` lines 889-928", id="backtick-forward"),
            pytest.param("lines 889-928 of `{file}`", id="backtick-reverse"),
            pytest.param("{file}, L889-L928", id="l-comma"),
            pytest.param("{file} (L889-L928)", id="l-parentheses"),
            pytest.param("line 12 in `{file}`", id="line-in-backticks"),
            pytest.param("{file}:L12", id="l-colon"),
            pytest.param("{file} @ L12", id="l-at"),
            pytest.param("{file}#L12-L20", id="l-fragment"),
            pytest.param("{file} lines 12–20", id="en-dash-forward"),
            pytest.param("{file} ll. 12-20", id="ll-forward"),
            pytest.param("see {file}, line 12", id="see-comma-line"),
            pytest.param("{file}：12", id="fullwidth-colon"),
            pytest.param("{file}：L12–L20", id="fullwidth-l-range"),
            pytest.param("'{file}' lines 12-20", id="single-quote-forward"),
            pytest.param("lines 12-20 of '{file}'", id="single-quote-reverse"),
            pytest.param('"{file}" lines 12-20', id="double-quote-forward"),
            pytest.param('lines 12-20 of "{file}"', id="double-quote-reverse"),
            pytest.param("‘{file}’ lines 12-20", id="curly-single-forward"),
            pytest.param("lines 12-20 of ‘{file}’", id="curly-single-reverse"),
            pytest.param("“{file}” lines 12-20", id="curly-double-forward"),
            pytest.param("lines 12-20 of “{file}”", id="curly-double-reverse"),
            pytest.param("[{file}]({file}) lines 12-20", id="link-forward"),
            pytest.param("lines 12-20 of [{file}]({file})", id="link-reverse"),
            pytest.param("[`{file}`]({file}) L12", id="code-link-forward"),
            pytest.param("L12 in [`{file}`]({file})", id="code-link-reverse"),
            pytest.param(
                "[{file}](docs/source(v1)) lines 12-20", id="link-parentheses"
            ),
            pytest.param("[{file}][source] lines 12-20", id="reference-link"),
            pytest.param("{file}, L12", id="l-single"),
            pytest.param("{file} [L12-L20]", id="l-brackets"),
            pytest.param("L12-L20 in {file}", id="l-reverse"),
            pytest.param("lines L12–L20 of `{file}`", id="l-en-dash-reverse"),
            pytest.param("{file} lines L12-L20", id="l-labelled"),
            pytest.param("{file} lines 12—20", id="em-dash-forward"),
            pytest.param("lines 12—20 of {file}", id="em-dash-reverse"),
            pytest.param("lines 12–20 in {file}", id="en-dash-reverse"),
            pytest.param("lines 12, 14-20 of {file}", id="reverse-list"),
            pytest.param("ll.12–20 in {file}", id="ll-reverse"),
            pytest.param("{file} l.12", id="l-abbreviation"),
            pytest.param("{file} at line 12", id="at-line"),
            pytest.param("{file} on lines 12 to 20", id="on-lines-to"),
            pytest.param("lines 12 to 20 of {file}", id="reverse-to"),
            pytest.param("{file}, L12–L20", id="l-en-dash-forward"),
            pytest.param("LINES 12-20 OF `{file}`", id="uppercase-reverse"),
            pytest.param("`{file}`: `L12`", id="quoted-locator"),
            pytest.param("{file} `L12`", id="quoted-bare-locator"),
            pytest.param("{file} `lines 12-20`", id="quoted-labelled-locator"),
            pytest.param("{file}: lines 12-20", id="colon-labelled"),
            pytest.param("lines `12–20` in `{file}`", id="quoted-reverse-numbers"),
        ],
    )
    def test_formatted_line_citation_is_refused_at_depth(
        self, spelling: str, source: str, as_key: bool
    ) -> None:
        citation = spelling.format(file=source)
        # A local file cannot borrow an exemption from a same-document pin.
        document = {
            "source": {"path": source, "commit": "abcdef0"},
            "notes": _plant(citation, 3, as_key),
        }
        assert source_line_citations(document) == [(source, citation)]
        assert unpinned_line_citations(document) == [(source, citation)]

    @pytest.mark.parametrize(
        ("pointer", "records", "message"),
        [
            ("/entries[input=absent]", [{"input": "target"}], "one matching record"),
            ("/entries[unknown=target]", [{"input": "target"}], "one matching record"),
            ("/entries[input=target]", [{"other": "target"}], "one matching record"),
            (
                "/entries[input=target]",
                [{"input": "target"}, {"input": "target"}],
                "one matching record",
            ),
            ("/entries[]", [{"input": "target"}], "selector"),
            ("/entries[input]", [{"input": "target"}], "selector"),
            ("/entries[=target]", [{"input": "target"}], "selector"),
            ("/entries[input=]", [{"input": "target"}], "selector"),
            ("/entries[input=target", [{"input": "target"}], "selector"),
            ("/entries[input=target]]", [{"input": "target"}], "selector"),
            ("/entries[[input=target]]", [{"input": "target"}], "selector"),
            ("/entries[input=target]junk", [{"input": "target"}], "selector"),
            ("/entries[input=target,]", [{"input": "target"}], "selector"),
            ("/entries[input=target,input=target]", [{"input": "target"}], "selector"),
            ("/entries/0", [{"input": "target"}], "record array"),
            ("/entries/0/input", [{"input": "target"}], "record array"),
        ],
    )
    def test_invalid_record_selectors_and_record_array_indexes_are_refused(
        self, citation_checkout, pointer: str, records: list, message: str
    ) -> None:
        citation_checkout("records.json", {"entries": records})
        with pytest.raises(ValueError, match=message):
            resolve_json_pointer("records.json", pointer)
        with pytest.raises(ValueError, match=message):
            json_pointer_citations(_plant(f"records.json#{pointer}", 2, True))

    def test_keyed_selector_extends_json_pointers_and_supports_conjunctions(
        self, citation_checkout
    ) -> None:
        target = {"input": "target", "module": "nz/core~v1", "values": [7, 42]}
        citation_checkout(
            "records.json",
            {"entries": [{"input": "target", "module": "other"}, target]},
        )
        pointer = "/entries[input=target,module=nz~1core~0v1]/values/1"
        assert resolve_json_pointer("records.json", pointer) == 42
        assert json_pointer_citations(f"records.json#{pointer}; following prose") == [
            ("records.json", pointer)
        ]
        assert json_pointer_citations(f"records.json#{pointer}, following prose") == [
            ("records.json", pointer)
        ]

    @pytest.mark.parametrize(
        "entries", [{"input": "target"}, ["target"], [{"input": "target"}, 1]]
    )
    def test_record_selector_requires_an_array_of_mappings(
        self, citation_checkout, entries: object
    ) -> None:
        citation_checkout("records.json", {"entries": entries})
        pointer = "/entries[input=target]"
        with pytest.raises(ValueError, match="array of mappings"):
            resolve_json_pointer("records.json", pointer)
        with pytest.raises(ValueError, match="array of mappings"):
            json_pointer_citations(f"records.json#{pointer}")

    def test_scalar_array_indexes_still_check_bounds(self, citation_checkout) -> None:
        citation_checkout("records.json", {"values": [7, 42]})
        assert resolve_json_pointer("records.json", "/values/1") == 42
        with pytest.raises(IndexError):
            resolve_json_pointer("records.json", "/values/2")

    def test_keyed_selector_is_independent_of_record_position(
        self, citation_checkout
    ) -> None:
        @settings(max_examples=100, deadline=None)
        @given(keys=st.lists(_SOURCE_STEM, unique=True, max_size=25), data=st.data())
        def check(keys: list[str], data) -> None:
            position = data.draw(st.integers(0, len(keys)))
            records = [{"input": f"other_{key}", "value": key} for key in keys]
            target = {"input": "target", "value": {"answer": 42}}
            records.insert(position, target)
            citation_checkout("records.json", {"entries": records})
            pointer = "/entries[input=target]"
            assert resolve_json_pointer("records.json", pointer) == target
            assert resolve_json_pointer("records.json", pointer + "/value/answer") == 42
            assert json_pointer_citations(f"records.json#{pointer}") == [
                ("records.json", pointer)
            ]
            with pytest.raises(ValueError, match="record array"):
                resolve_json_pointer("records.json", f"/entries/{position}")
            with pytest.raises(ValueError, match="record array"):
                json_pointer_citations(f"records.json#/entries/{position}")

        check()

    @pytest.mark.parametrize(
        "disclosure", ["sharing_home", "non_beneficiary", "market_income"]
    )
    def test_keyed_citations_survive_a_record_inserted_before_the_target(
        self, citation_checkout, disclosure: str
    ) -> None:
        reporting = _load("reporting_classifiers.json")
        if disclosure == "market_income":
            file = "packages/microcosm-frame/src/microcosm/frame/adapters/axiom_concept_mappings/nz.json"
            original = json.loads(
                (_BUILD.repository / file).read_text(encoding="utf-8")
            )
            array, field = "bindings", "canonical_input"
            expected_key = "nz:statutes/social_security/accommodation_supplement/core#input.accommodation_supplement_relevant_weekly_income"
            evidence = reporting["disclosures"][1]["evidence"]
        else:
            file, array, field = "axiom_input_closure.json", "entries", "input"
            original = _load(file)
            if disclosure == "sharing_home":
                expected_key = (
                    "nz_super_shares_residence_with_person_at_least_adult_age"
                )
                evidence = reporting["disclosures"][0]["evidence"]
            else:
                expected_key = "accommodation_supplement_non_beneficiary"
                evidence = reporting["as_non_beneficiary_audit"]["evidence"]
        (expected,) = [row for row in original[array] if row[field] == expected_key]
        original[array].insert(0, {field: "dummy_record"})
        citation_checkout(file, original)
        (citation,) = json_pointer_citations(evidence)
        assert resolve_json_pointer(*citation) == expected

    def test_local_file_pins_cannot_hide_absolute_or_parent_path_aliases(self) -> None:
        relative = "packages/microcosm-frame/src/microcosm/frame/concepts.py"
        aliases = [
            str(_BUILD.repository / relative),
            f"../{_BUILD.repository.name}/{relative}",
            "packages/microcosm-frame/src/microcosm/frame/../frame/concepts.py",
        ]
        for source in aliases:
            citation = f"{source}:889-928"
            assert unpinned_line_citations(
                {
                    "source": {"path": source, "commit": "abcdef0"},
                    "note": citation,
                }
            ) == [(source, citation)]

    @pytest.mark.parametrize("source", ["LICENSE", ".gitignore", "Dockerfile"])
    def test_extensionless_repository_files_are_also_refused(self, source: str) -> None:
        citation = f"{source}:1-3"
        assert unpinned_line_citations(
            {
                "source": {"path": source, "commit": "abcdef0"},
                "note": citation,
            }
        ) == [(source, citation)]

    def test_the_input_closure_format_identifier_is_not_a_python_symbol(self) -> None:
        assert dotted_symbol_citations({"format": INPUT_CLOSURE_FORMAT}) == set()
        assert dotted_symbol_citations({"note": INPUT_CLOSURE_FORMAT}) == {
            INPUT_CLOSURE_FORMAT
        }

    def test_nested_dotted_attributes_resolve_after_the_module_prefix(self) -> None:
        assert (
            resolve_dotted_symbol("microcosm.frame.concepts.TransportRule.CARRY")
            is frame_concepts.TransportRule.CARRY
        )

    def test_json_pointer_keys_use_rfc6901_slash_escaping(self) -> None:
        selected = resolve_json_pointer(
            "reporting_classifiers.json",
            "/parameter_provenance/~1income_decile~1entity",
        )
        assert selected["item"] == "A1"

    def test_create_and_donor_reader_citations_still_describe_unit_inputs(self) -> None:
        from microcosm.build.transport import population_kernels, unit_declarations
        from microcosm.build.transport.population_kernels import TransportCreateKernel
        from microcosm.frame import unit_construction
        from microcosm.frame.transport import read_populace_us_donor
        from microcosm.graph.decl import StructuralDelta

        rule = _load("benefit_unit_rule.json")
        reporting = _load("reporting_classifiers.json")
        for document in [rule, reporting["tie_key_evidence"]]:
            named = dotted_symbol_citations(document)
            assert (
                "microcosm.build.transport.population_kernels.TransportCreateKernel"
                in named
            )
        assert (
            "microcosm.frame.transport.read_populace_us_donor"
            in dotted_symbol_citations(reporting)
        )
        assert TransportCreateKernel.capabilities.structural is StructuralDelta.CREATE
        # Differential source evidence: CREATE reads the verified concept donor,
        # then delegates its unchanged NZ rule through the canonical parser and
        # the registered benefit-unit membership constructor.
        create_source = inspect.getsource(TransportCreateKernel)
        assert (
            create_source.index("read_populace_us_donor(")
            < create_source.index("create_unit_declaration(")
            < create_source.index("build_declared_units(")
        )
        assert "tables, dropped = split_for_transport(donor.tables)" in create_source
        assert 'tables["person"], tables["household"], declaration' in create_source
        assert (
            population_kernels.create_unit_declaration
            is unit_declarations.create_unit_declaration
        )
        assert TransportCreateKernel.ref == unit_declarations.CREATE_V1_REF
        parser_source = inspect.getsource(unit_declarations.create_unit_declaration)
        assert "if kernel == CREATE_V1_REF:" in parser_source
        assert "rule = BenefitUnitRule.from_spec(document)" in parser_source
        assert "return MultiUnitRule.from_benefit_unit(rule)" in parser_source
        declaration = unit_declarations.create_unit_declaration(
            TransportCreateKernel.ref, rule
        )
        (unit,) = declaration.units
        assert unit.kind == "benefit_unit"
        assert (
            unit.rule.to_dict()
            == unit_construction.BenefitUnitRule.from_spec(rule).to_dict()
        )
        rule_type, constructor = unit_construction._UNIT_KINDS[unit.kind]
        assert rule_type is unit_construction.BenefitUnitRule
        constructor_source = inspect.getsource(constructor)
        assert "_heads(rows, _dependents(rows, rule), rule)" in constructor_source
        framework_source = inspect.getsource(unit_construction.build_declared_units)
        assert "_UNIT_KINDS[item.kind][1](rows, person, item.rule)" in framework_source
        assert (
            "_build_partition(person, household, item.rule, rows, head)"
            in framework_source
        )
        legacy_source = inspect.getsource(unit_construction.build_benefit_units)
        assert "_dependents(rows, rule)" in legacy_source
        assert "_heads(rows, dependent, rule)" in legacy_source
        assert "_build_partition(person, household, rule, rows, head)" in legacy_source
        donor_source = inspect.getsource(read_populace_us_donor)
        assert "tables = POLICYENGINE_US_CONCEPT_MAPPING.decode(raw)" in donor_source
        assert 'tables["person"][name] = values' in donor_source
        assert 'tables["household"]["reference_person_id"]' in donor_source
        assert 'tables["household"][SOURCE_HOUSEHOLD_ID_COLUMN]' in donor_source

    def test_financial_independence_citations_still_bind_carried_usual_hours(
        self,
    ) -> None:
        from microcosm.frame.adapters.policyengine_us_concepts import (
            POLICYENGINE_US_CONCEPT_MAPPING,
        )
        from microcosm.frame.concept_mapping import Identity

        rule = _load("benefit_unit_rule.json")["dependent_child"][
            "financial_independence"
        ]
        hours = frame_concepts.concept(rule["hours_concept"])
        assert hours.id == "fact:person.usual_weekly_hours"
        assert "across all jobs and self-employment" in hours.definition
        assert "in the weeks the person worked" in hours.definition
        assert "reference year" in hours.definition
        assert hours.transport is frame_concepts.TransportRule.CARRY
        (binding,) = [
            binding
            for binding in POLICYENGINE_US_CONCEPT_MAPPING.bindings
            if binding.engine_input == rule["donor_variable"]
        ]
        assert binding.engine_entity == "person"
        assert binding.concepts == (hours.id,)
        assert isinstance(binding.transform, Identity)
        assert binding.relation is frame_concepts.AlignmentRelation.EXACT
        evidence = rule["concept_evidence"]
        assert 'concept("fact:person.usual_weekly_hours")' in evidence
        assert (
            "docs/concept-coverage/policyengine-us.md#inputs-each-concept-feeds"
            in evidence
        )
        assert (
            "microcosm.frame.adapters.policyengine_us_concepts.POLICYENGINE_US_CONCEPT_MAPPING"
            in evidence
        )
        assert "microcosm.frame.transport.read_populace_us_donor" in evidence
        coverage = _BUILD.repository / "docs/concept-coverage/policyengine-us.md"
        source = coverage.read_text(encoding="utf-8")
        assert "## Inputs each concept feeds" in source
        (row,) = [
            row for row in source.splitlines() if row.startswith(f"| `{hours.id}` |")
        ]
        assert f"`{rule['donor_variable']}`" in row

    def test_reporting_disclosure_cites_the_sharing_home_default_record(self) -> None:
        disclosure = _load("reporting_classifiers.json")["disclosures"][0]
        (citation,) = json_pointer_citations(disclosure["evidence"])
        record = resolve_json_pointer(*citation)
        assert (
            record["input"]
            == "nz_super_shares_residence_with_person_at_least_adult_age"
        )
        assert record["class"] == "defaulted"
        assert record["value"] is False
        assert "overstates NZS for singles who share a home" in record["reason"]
        assert "overstated for single people who share a home" in disclosure["note"]

    def test_reporting_non_beneficiary_audit_cites_its_exact_state_binding(
        self,
    ) -> None:
        document = _load("reporting_classifiers.json")
        (citation,) = json_pointer_citations(
            document["as_non_beneficiary_audit"]["evidence"]
        )
        record = resolve_json_pointer(*citation)
        assert record["input"] == "accommodation_supplement_non_beneficiary"
        assert record["class"] == "encoded"
        binding = record["state_binding"]
        assert binding["test"] == "any_true"
        assert binding["group_rule"] == "any_member"
        assert binding["negate"] is True
        assert binding["columns"] == [
            "person.receives_jobseeker_support",
            "person.receives_sole_parent_support",
            "person.receives_supported_living_payment",
            "person.receives_nz_super",
        ]
        assert "person.receives_veterans_pension" not in binding["columns"]
        assert (
            "The hub must extend both together"
            in document["as_non_beneficiary_audit"]["veterans_pension"]
        )

    def test_reporting_market_income_disclosure_cites_the_mapping_it_describes(
        self,
    ) -> None:
        document = _load("reporting_classifiers.json")
        disclosure = document["disclosures"][1]
        (citation,) = json_pointer_citations(disclosure["evidence"])
        binding = resolve_json_pointer(*citation)
        assert (
            binding["engine_input"] == "accommodation_supplement_relevant_weekly_income"
        )
        assert binding["engine_entity"] == "Family"
        assert binding["group_rule"] == "sum_over_members"
        market = {
            row["column"]
            for row in document["income_decile"]["components"]
            if row["kind"] == "market"
        }
        assert set(binding["concepts"]) == {
            f"fact:person.{column}" for column in market
        }
        assert binding["transform"] == {"kind": "scaled_sum", "factor": 1 / 52}
        assert "annual gross income" in binding["note"]
        rental = frame_concepts.concept("fact:person.rental_income")
        pension = frame_concepts.concept("fact:person.private_pension_income")
        assert callable(frame_concepts._amount)
        assert "net of expenses" in rental.definition
        assert "before tax" not in rental.definition.lower()
        assert "before tax" not in pension.definition.lower()
        assert '_amount("rental_income")' in disclosure["evidence"]
        assert '_amount("private_pension_income")' in disclosure["evidence"]

    def test_no_resource_cites_a_repository_line_or_unpinned_foreign_line(self) -> None:
        offenders = {
            name: hits
            for name, payload in _package_payloads().items()
            if (hits := unpinned_line_citations(payload))
        }
        assert offenders == {}

    def test_the_foreign_line_citations_are_scanned_and_covered_by_their_pins(
        self,
    ) -> None:
        bridge = _load("as_rate_bridge.json")
        assert {name for name, _ in source_line_citations(bridge)} == {"run.py"}
        assert commit_pinned_sources(bridge) == {
            "nz-lane/emtr_reproduction/run.py",
            "conformance/executable/nz-treasury-incomeexplorer/requests.json",
        }
        assert bridge["source"]["commit"] == "ba6e7d6e4f488fdb2762dfb9d91c90f3238b719c"
        assert unpinned_line_citations(bridge) == []
        # The prose-only plan_citation names no file, so it remains out of scope.
        assert source_line_citations(bridge["source"]["plan_citation"]) == []
        closure = _load("axiom_input_closure.json")
        expected = {
            "src/rulespec.rs",
            "src/dense.rs",
            "src/formula.rs",
            "python-ext/src/lib.rs",
        }
        assert {name for name, _ in source_line_citations(closure)} == expected
        assert commit_pinned_sources(closure) == expected
        assert unpinned_line_citations(closure) == []
        for pin in closure["engine_source_pins"]["pins"]:
            assert pin["commit"] == closure["input_surface"]["engine"]["commit"]
            assert re.fullmatch(r"[0-9a-f]{40}", pin["commit"])
        # Each exemption is required: removing that file's pin exposes a citation.
        for name in expected:
            unpinned = copy.deepcopy(closure)
            unpinned["engine_source_pins"]["pins"] = [
                pin
                for pin in unpinned["engine_source_pins"]["pins"]
                if pin["path"] != name
            ]
            assert name in {path for path, _ in unpinned_line_citations(unpinned)}

    @pytest.mark.parametrize(
        "replaced",
        [
            "packages/microcosm-frame/src/microcosm/frame/concepts.py:880-919",
            "a lone co-resident parent is always parent 1 (the pointer "
            "definition, concepts.py lines 892-901; the pointer checks ...)",
            "(microcosm.graph.executor._structural_columns, executor.py lines 636-643)",
            "(_validate_population_declaration, executor.py lines 1550-1573)",
            "the conformance harness uses 365/7 weeks (ops main run.py:168)",
            "concepts.py:889-928",
            "executor.py:638-645",
            "executor.py:1549-1554",
            "transport/population_kernels.py:208-232",
            "concepts.py:900-907",
            "concepts.py:1069-1083",
            "adapters/policyengine_us_concepts.py:352-362",
            "transport.py:843-849",
            "docs/concept-coverage/policyengine-us.md:31",
            "packages/microcosm-build/src/microcosm/build/transport/population_kernels.py:209-228",
            "packages/microcosm-frame/src/microcosm/frame/transport.py:843-856",
            "axiom_input_closure.json:1045-1052",
            "axiom_input_closure.json:1183-1204",
            "packages/microcosm-frame/src/microcosm/frame/adapters/axiom_concept_mappings/nz.json:670-688",
            "concepts.py:990-1000,1040-1051",
        ],
    )
    def test_the_line_citations_the_package_carried_are_refused(
        self, replaced: str
    ) -> None:
        assert unpinned_line_citations({"rule": replaced})

    @pytest.mark.parametrize("locator", _LINE_LOCATORS)
    @pytest.mark.parametrize(
        "extension", ["py", "json", "yaml", "rs", "md", "toml", "csv", "ts", "h5"]
    )
    def test_every_locator_spelling_covers_every_file_extension(
        self, locator: str, extension: str
    ) -> None:
        citation = locator.format(file=f"source.{extension}", a=12, b=34)
        assert source_line_citations(citation) == [(f"source.{extension}", citation)]

    @settings(max_examples=300, deadline=None)
    @given(
        stem=_SOURCE_STEM,
        extension=_SOURCE_EXTENSION,
        directory=_SOURCE_DIRECTORY,
        locator=_LINE_LOCATOR_SPELLINGS,
        a=st.integers(1, 99_999),
        b=st.integers(1, 99_999),
        prefix=_PROSE,
        suffix=_PROSE,
        depth=st.integers(0, 3),
        as_key=st.booleans(),
    )
    def test_an_unpinned_line_citation_is_refused_at_any_depth(
        self,
        stem: str,
        extension: str,
        directory: str,
        locator: str,
        a: int,
        b: int,
        prefix: str,
        suffix: str,
        depth: int,
        as_key: bool,
    ) -> None:
        source = f"{directory}{stem}.{extension}"
        citation = locator.format(file=source, a=a, b=b)
        text = f"{prefix} {citation} {suffix}"
        assert (source, text) in unpinned_line_citations(_plant(text, depth, as_key))

    @settings(max_examples=200, deadline=None)
    @given(
        stem=_SOURCE_STEM,
        extension=_SOURCE_EXTENSION,
        locator=_LINE_LOCATOR_SPELLINGS,
        a=st.integers(1, 99_999),
        b=st.integers(1, 99_999),
        depth=st.integers(0, 3),
        as_key=st.booleans(),
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        bare_name=st.booleans(),
    )
    def test_a_foreign_file_pin_at_a_commit_covers_its_line_citation(
        self,
        stem: str,
        extension: str,
        locator: str,
        a: int,
        b: int,
        depth: int,
        as_key: bool,
        commit: str,
        length: int,
        bare_name: bool,
    ) -> None:
        source = f"foreign_citation_fixture_{stem}.{extension}"
        path = f"another-repository/src/{source}"
        citation = locator.format(file=source if bare_name else path, a=a, b=b)
        document = {
            "source": {
                "repository": "Other/repository",
                "path": path,
                "commit": commit[:length],
            },
            "notes": [_plant(citation, depth, as_key)],
        }
        assert source_line_citations(document)
        assert unpinned_line_citations(document) == []

    @settings(max_examples=200, deadline=None)
    @given(
        stem=_SOURCE_STEM,
        other=_SOURCE_STEM,
        extension=_SOURCE_EXTENSION,
        locator=_LINE_LOCATOR_SPELLINGS,
        a=st.integers(1, 99_999),
        b=st.integers(1, 99_999),
        commit=_HEX40_TEXT,
        branch=st.sampled_from(_BRANCH_NAMES),
        pin_other_file=st.booleans(),
    )
    def test_another_file_or_invalid_commit_does_not_cover_a_citation(
        self,
        stem: str,
        other: str,
        extension: str,
        locator: str,
        a: int,
        b: int,
        commit: str,
        branch: str,
        pin_other_file: bool,
    ) -> None:
        source = f"foreign_citation_fixture_{stem}.{extension}"
        if pin_other_file:
            assume(other != stem)
            pin = {
                "path": f"foreign_citation_fixture_{other}.{extension}",
                "commit": commit,
            }
        else:
            pin = {"path": source, "commit": branch}
        citation = locator.format(file=source, a=a, b=b)
        assert (source, citation) in unpinned_line_citations(
            {"source": pin, "note": citation}
        )

    def test_a_different_directory_does_not_pin_the_same_basename(self) -> None:
        citation = "other/src/foreign_citation_fixture.rs:12-34"
        assert unpinned_line_citations(
            {
                "source": {
                    "path": "first/src/foreign_citation_fixture.rs",
                    "commit": "abcdef0",
                },
                "note": citation,
            }
        ) == [("other/src/foreign_citation_fixture.rs", citation)]

    @pytest.mark.parametrize(
        "source",
        [
            "concepts.py",
            "packages/microcosm-frame/src/microcosm/frame/concepts.py",
            "axiom_input_closure.json",
            "docs/concept-coverage/policyengine-us.md",
        ],
    )
    def test_a_pin_never_exempts_a_repository_file(self, source: str) -> None:
        citation = f"{source}:12-34"
        assert unpinned_line_citations(
            {
                "source": {"path": source, "commit": "abcdef0"},
                "note": citation,
            }
        ) == [(source, citation)]

    @settings(max_examples=200, deadline=None)
    @given(
        stem=_SOURCE_STEM,
        extension=_SOURCE_EXTENSION,
        directory=_SOURCE_DIRECTORY,
        locator=st.sampled_from(_SYMBOL_LOCATORS),
        wrapper=st.sampled_from(_FILE_WRAPPERS),
        symbol=_SOURCE_STEM,
        prefix=_PROSE,
        suffix=_PROSE,
    )
    def test_a_symbol_citation_is_not_a_line_citation(
        self,
        stem: str,
        extension: str,
        directory: str,
        locator: str,
        wrapper: str,
        symbol: str,
        prefix: str,
        suffix: str,
    ) -> None:
        source = wrapper.format(file=f"{directory}{stem}.{extension}")
        citation = locator.format(file=source, symbol=symbol)
        assert source_line_citations({"rule": f"{prefix} {citation} {suffix}"}) == []

    @pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
    @pytest.mark.parametrize(
        "citation",
        [
            "`microcosm.graph.executor._structural_columns`",
            "takeup.assign@1",
            "`takeup.assign` @ 1",
            "`concepts.py`: `_pointer`",
            "concepts.py: L12_symbol",
            "concepts.py: l12",
            "concepts.py: line12",
            "docs/method.md#heading12",
            "records.json#/entries/12",
            "records.json#/entries[input=L12,module=nz~1core~0v1]/values/0",
            "records.json#/entries[input=line12]",
        ],
    )
    def test_symbols_and_stable_records_are_not_line_citations(
        self, citation: str, as_key: bool
    ) -> None:
        assert source_line_citations(_plant(citation, 3, as_key)) == []
        assert unpinned_line_citations(_plant(citation, 3, as_key)) == []

    def test_a_json_pointer_citation_is_not_a_line_citation(
        self, citation_checkout
    ) -> None:
        @settings(max_examples=200, deadline=None)
        @given(
            stem=_SOURCE_STEM,
            key=_SOURCE_STEM,
            index=st.integers(0, 9999),
            depth=st.integers(0, 3),
            as_key=st.booleans(),
        )
        def check(stem: str, key: str, index: int, depth: int, as_key: bool) -> None:
            citation_checkout(
                f"{stem}.json", {"entries": {key: list(range(index + 1))}}
            )
            pointer = f"/entries/{key}/{index}"
            citation = f"{stem}.json#{pointer}"
            assert source_line_citations(_plant(citation, depth, as_key)) == []
            assert json_pointer_citations(_plant(citation, depth, as_key)) == [
                (f"{stem}.json", pointer)
            ]
            assert resolve_json_pointer(f"{stem}.json", pointer) == index

        check()

    def test_every_json_pointer_citation_in_the_package_resolves(self) -> None:
        citations = {
            citation
            for payload in _package_payloads().values()
            for citation in json_pointer_citations(payload)
        }
        assert {
            (
                "axiom_input_closure.json",
                "/entries[input=nz_super_shares_residence_with_person_at_least_adult_age]",
            ),
            (
                "axiom_input_closure.json",
                "/entries[input=accommodation_supplement_non_beneficiary]",
            ),
            (
                "packages/microcosm-frame/src/microcosm/frame/adapters/axiom_concept_mappings/nz.json",
                "/bindings[canonical_input=nz:statutes~1social_security~1accommodation_supplement~1core#input.accommodation_supplement_relevant_weekly_income]",
            ),
        } <= citations
        for file, pointer in sorted(citations):
            resolve_json_pointer(file, pointer)

    @pytest.mark.parametrize(
        ("file", "pointer", "error"),
        [
            ("axiom_input_closure.json", "/entries/no_such_record", ValueError),
            ("axiom_input_closure.json", "/no_such_key", KeyError),
            ("no_such_citation_file.json", "/entries", ValueError),
            ("axiom_input_closure.json", "/entries/01", ValueError),
            ("axiom_input_closure.json", "/entries/999999", ValueError),
            ("axiom_input_closure.json", "/bad~2escape", ValueError),
        ],
    )
    def test_a_missing_or_invalid_json_pointer_raises(
        self, file: str, pointer: str, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            resolve_json_pointer(file, pointer)

    def test_executor_symbols_still_make_unit_ids_and_memberships_structural(
        self,
    ) -> None:
        from microcosm.graph import executor

        schema = SimpleNamespace(
            entities=("person", "family"),
            person_entity="person",
            group_entities=("family",),
            entity_id_column=lambda entity: f"{entity}_id",
            membership_column=lambda entity: f"person_{entity}_id",
        )
        frame = SimpleNamespace(schema=schema)
        assert executor._structural_columns(frame, "family") == ["family_id"]
        assert executor._structural_columns(frame, "person") == [
            "person_id",
            "person_family_id",
        ]
        for entity, column in [("family", "family_id"), ("person", "person_family_id")]:
            node = SimpleNamespace(
                id="citation-fixture",
                weights=None,
                structural=executor.StructuralDelta.NONE,
                outputs=[SimpleNamespace(entity=entity, column=column)],
            )
            with pytest.raises(
                executor.NodeRejected, match="cannot own structural column"
            ):
                executor._validate_population_declaration(
                    node, SimpleNamespace(frame=frame)
                )

    def test_benefit_unit_rule_cites_the_relationship_pointers_by_symbol(
        self,
    ) -> None:
        rule = _load("benefit_unit_rule.json")
        source = inspect.getsource(frame_concepts)
        block = source.split("# --- Relationships", 1)[1].split("# --- ", 1)[0]
        block_pointers = re.findall(r'_pointer\(\s*"(\w+)"', block)
        assert block_pointers == [
            "partner_person_id",
            "parent_1_person_id",
            "parent_2_person_id",
            "reference_person_id",
        ]
        pointers = rule["pointer_concepts"]
        named = [
            pointers["partner"],
            *pointers["parents"],
            pointers["household_reference_person"],
        ]
        assert named == block_pointers
        assert pointers["citation"] == (
            "packages/microcosm-frame/src/microcosm/frame/concepts.py, the "
            f'"# --- Relationships" block: _pointer("{block_pointers[0]}") '
            f'through _pointer("{block_pointers[-1]}")'
        )
        cited = {
            name
            for text in _nested_strings(rule)
            for name in re.findall(r'_pointer\("(\w+)"\)', text)
        }
        assert cited == {
            "partner_person_id",
            "parent_1_person_id",
            "reference_person_id",
        }
        (child,) = [
            row for row in rule["composition"] if row["member"] == "dependent_child"
        ]
        assert 'concepts.py _pointer("parent_1_person_id")' in child["rule"]
        parent_1 = frame_concepts.concept("fact:person.parent_1_person_id")
        assert "always parent 1" in parent_1.definition
        assert callable(frame_concepts._check_pointers)

    def test_every_dotted_microcosm_symbol_in_the_package_resolves(self) -> None:
        named = {
            dotted
            for payload in _package_payloads().values()
            for dotted in dotted_symbol_citations(payload)
        }
        assert {
            "microcosm.graph.executor._structural_columns",
            "microcosm.graph.executor._validate_population_declaration",
        } <= named
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
        self, dotted: str, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            resolve_dotted_symbol(dotted)


# ---------------------------------------------------------------------------
# Remaining resources
# ---------------------------------------------------------------------------


class TestDeclaredResources:
    def test_d1376_fit_and_admin_policy_is_explicit_and_active(self) -> None:
        spec = load_country_spec("nz")
        gates = {gate.id: gate for gate in spec.gates.gates}
        families = (
            "demography",
            "household_structure",
            "housing_tenure",
            "income",
            "main_benefits",
            "superannuation",
        )
        for gate_id, within, selected in (
            ("calibration_per_family_fit", 0.05, families),
            ("demography_per_family_fit", 0.02, ("demography",)),
        ):
            gate = gates[gate_id]
            assert dict(gate.parameters) == {
                "within": within,
                "min_family_share": 1,
                "hard_within": within,
                "min_hard_family_share": 1,
                "min_family_size": 1,
                "families": selected,
            }
        admin = gates["national_and_region_admin_aggregates"]
        assert dict(admin.parameters) == {
            "default_rtol": 0.05,
            "families": families,
            "geography_levels": ("country", "region"),
        }
        for gate_id in (
            "calibration_per_family_fit",
            "demography_per_family_fit",
            "national_and_region_admin_aggregates",
        ):
            gate = gates[gate_id]
            assert gate.not_applicable is None
            assert gate.criticality == "release_blocking"
            assert gate.evidence_absent_blocks
            assert "d1376, 2026-10-12; g8-method-addendum B7:" in gate.notes
        assert gates["macro_realism_nz"].not_applicable
        # A reference tolerance is absolute, so it cannot abbreviate d1376's
        # percentage policy. The gate owns the numerical thresholds.
        assert all(
            reference.tolerance is None for reference in _references("calibration")
        )

    def test_mc15_ruled_thresholds_and_tracked_targets_are_declared(self) -> None:
        spec = load_country_spec("nz")
        gates = {gate.id: gate for gate in spec.gates.gates}
        ess = gates["weight_effective_sample_size"]
        ratio = gates["weight_max_to_median_ratio"]
        support = gates["reported_cell_support"]
        target = gates["weight_effective_sample_size_target_a"]
        cell_target = gates["reported_cell_support_target_a"]
        assert ess.parameters["minimum_effective_sample_size"] == 4000
        assert "minimum_ess_fraction" not in ess.parameters
        assert ratio.parameters["maximum_max_to_median_ratio"] == 5
        assert support.parameters["minimum_cell_ess"] == 100
        assert support.parameters["maximum_contribution_cv"] == 2
        assert support.parameters["maximum_single_donor_dollar_share"] == 0.1
        assert target.parameters["minimum_effective_sample_size"] == 16000
        assert cell_target.parameters["minimum_cell_ess"] == 400
        assert cell_target.parameters["maximum_single_donor_dollar_share"] == 0.05
        # The support contract's copy of target (a) drives the per-cell flags;
        # it must match the tracked target gate that reports them.
        assert support.parameters["target_minimum_cell_ess"] == 400
        assert support.parameters["target_maximum_single_donor_dollar_share"] == 0.05
        assert set(cell_target.parameters["required_cells"]) == set(
            support.parameters["required_cells"]
        )
        assert set(cell_target.parameters["required_scenarios"]) == set(
            support.parameters["required_scenarios"]
        )
        for gate in (ess, ratio, support, target, cell_target):
            assert gate.not_applicable is None
            assert gate.parameters["aggregation_column"] == "source_household_id"
            assert gate.parameters["required_weight_entity"] == "household"
            assert gate.parameters["required_weight_kind"] == "calibrated"
        assert all(
            g.criticality == "release_blocking" and g.evidence_absent_blocks
            for g in (ess, ratio, support)
        )
        assert target.criticality == cell_target.criticality == "diagnostic"
        scenarios = _load("scenarios.json")["scenarios"]
        assert support.parameters["required_scenarios"] == tuple(
            s["id"] for s in scenarios
        )
        from microcosm.build.transport.takeup_kernels import reporting_cell_declarations

        grid = _load("scenarios.json")["gap_conventions"]["reporting_cells"]
        assert set(support.parameters["required_cells"]) == {
            row["name"] for row in reporting_cell_declarations(grid)
        }

    def test_currency_bridge_is_one_declared_multiplication(self) -> None:
        # The executable rate lives once, as cited decimal text, in
        # money_normalization.json; the MC4 record points to it.
        bridge = _load("currency_bridge.json")
        assert (bridge["from_currency"], bridge["to_currency"]) == ("USD", "NZD")
        assert bridge["operation"] == "multiply"
        assert bridge["rate"] == {
            "resource": "money_normalization.json",
            "path": ["rate"],
        }
        money = _load("money_normalization.json")
        rate = money[bridge["rate"]["path"][0]]
        assert money["quote"] == "to_per_from"
        assert money["rounding"] == {"digits": 2, "mode": "half_even"}
        assert rate["form"] == "decimal_text"
        assert isinstance(rate["text"], str) and float(rate["text"]) > 0
        assert rate["citation"]["url"].startswith("https://www.irs.gov/")
        assert rate["citation"]["column"] == str(bridge["rate_year"])
        assert "awaiting D1" not in bridge["status"]

    def test_benefit_unit_rule_declares_the_settled_dependent_child_age(self) -> None:
        rule = _load("benefit_unit_rule.json")
        assert rule["entity"] == "family" and rule["engine_entity"] == "Family"
        assert rule["dependent_child"]["age_limit_years"] == 18
        assert "UNVERIFIED" not in json.dumps(rule)
        assert set(rule["membership_columns"]) == {
            "family_id",
            "family_household_id",
            "person_family_id",
        }

    def test_support_contract_receipts_match_the_package_files(self) -> None:
        # Differential: sources.yaml records the bytes of the two contracts.
        sources = load_yaml12(
            (NZ_ROOT / "spec/sources.yaml").read_text(encoding="utf-8"),
            source="nz/spec/sources.yaml",
        )
        contracts = {
            "nz_populace_us_support_contract": "source_stages.json",
            "nz_region_assignment_contract": "geography_spine.json",
        }
        rows = {row["id"]: row for row in sources["sources"]}
        assert set(rows) == set(contracts)
        for source_id, name in contracts.items():
            raw = (NZ_ROOT / name).read_bytes()
            assert rows[source_id]["sha256"] == hashlib.sha256(raw).hexdigest()
            assert rows[source_id]["byte_size"] == len(raw)

    def test_donor_pin_and_us_support_provenance_are_explicit(self) -> None:
        stages = load_country_spec("nz").sources.stage_map()
        assert tuple(stages) == ("load_populace_us_support_pool",)
        donor = stages["load_populace_us_support_pool"]
        (artifact,) = donor.artifacts
        assert artifact["revision"] == (
            "populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z"
        )
        assert artifact["sha256"] == (
            "48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e"
        )
        assert artifact["size_bytes"] == 462915783
        assert {"donor_country_code", "support_stratum"} <= set(donor.outputs)
        assert "US donor support records" in donor.survey


class TestReportingClassifiers:
    def test_reporting_resource_is_packaged_and_loader_visible(self) -> None:
        from microcosm.build.transport.compose import load_transport_spec

        resources = load_transport_spec("nz")["resources"]
        assert resources["reporting_classifiers"] == _load("reporting_classifiers.json")

    def test_every_executable_parameter_cites_its_archived_method_item(self) -> None:
        document = _load("reporting_classifiers.json")
        paths = {
            path: item
            for classifier, item in (("income_decile", "A1"), ("receipt_group", "A2"))
            for path in _reporting_parameter_paths(
                document[classifier], f"/{classifier}"
            )
        }
        assert set(document["parameter_provenance"]) == set(paths)
        anchors = {"A1": "a1-income-decile-provisional", "A2": "a2-receipt-group"}
        for path, item in paths.items():
            citation = document["parameter_provenance"][path]
            assert citation["item"] == item
            assert citation["source"] == (
                f"docs/nz-reporting-classifiers-method.md#{anchors[item]}"
            )

    def test_every_provenance_path_and_anchor_resolves_in_checkout(self) -> None:
        citations = list(_reporting_citations(_load("reporting_classifiers.json")))
        assert citations
        for citation in citations:
            relative_path, separator, anchor = citation["source"].partition("#")
            assert separator and anchor, citation
            source_path = (_BUILD.repository / relative_path).resolve()
            assert source_path.is_relative_to(_BUILD.repository), citation
            assert source_path.is_file(), citation
            headings = re.findall(
                r"^#{1,6} (.+)$",
                source_path.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            heading_anchors = {
                re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
                for heading in headings
            }
            assert anchor in heading_anchors, citation

    def test_income_names_assigned_receipt_and_gross_components_only(self) -> None:
        document = _load("reporting_classifiers.json")
        classifier = document["income_decile"]
        assert classifier["entity"] == "family"
        assert classifier["membership_column"] == "person_family_id"
        assert classifier["weight_entity"] == "household"
        assert classifier["expected_weight_kind"] == "calibrated"
        assert classifier["tie_key"] == classifier["unit_id_column"] == "family_id"
        assert classifier["n_bins"] == 10
        assert classifier["rank_method"] == "weighted_midpoint"
        market = [row for row in classifier["components"] if row["kind"] == "market"]
        assert {row["column"] for row in market} == {
            "employment_income",
            "nonfarm_self_employment_income",
            "farm_self_employment_income",
            "interest_income",
            "dividend_income",
            "rental_income",
            "private_pension_income",
        }
        assert all(row["annualisation_factor"] == 1 for row in market)
        benefits = [row for row in classifier["components"] if row["kind"] == "benefit"]
        assert all(row["annualisation_factor"] == 52 for row in benefits)
        assert all(row["tax_basis"] == "before_tax" for row in classifier["components"])
        super_income = next(
            row for row in benefits if row["receipt_column"] == "receives_nz_super"
        )
        assert super_income["column"] == "nz_super_weekly_amount_before_tax"
        assert super_income["column"] in _bindings()["nz_super"]["variables"]
        assert all("entitled" not in row["receipt_column"] for row in benefits)

    def test_before_tax_paid_amounts_resolve_without_blocked_inputs(self) -> None:
        from microcosm.frame.reporting import validate_income_classifier

        document = _load("reporting_classifiers.json")
        classifier = document["income_decile"]
        missing = {
            row["receipt_column"]
            for row in classifier["components"]
            if row["column"] is None
        }
        assert missing == set()
        assert document["blocked_inputs"] == []
        validate_income_classifier(classifier)
        assert {
            row["column"]
            for row in classifier["components"]
            if row["kind"] == "benefit" and row["receipt_column"] != "receives_nz_super"
        } == {
            f"{program}_gross_weekly_payment"
            for program in (
                "jobseeker_support",
                "sole_parent_support",
                "supported_living_payment",
            )
        }
        paid = _bindings()["main_benefits_rates"]["variables"]
        assert {
            "jobseeker_support_net_weekly_payment",
            "sole_parent_support_net_weekly_payment",
            "supported_living_payment_net_weekly_payment",
        } <= set(paid)
        assert not any("before_tax" in name for name in paid)

    @pytest.mark.parametrize(
        ("entity", "column"),
        [
            ("person", "jobseeker_support_net_weekly_payment"),
            ("person", "sole_parent_support_net_weekly_payment"),
            ("person", "supported_living_payment_net_weekly_payment"),
            ("family", "accommodation_supplement"),
            ("family", "accommodation_supplement_rounded_weekly_payment"),
            ("family", "working_for_families"),
        ],
    )
    def test_spec_refuses_net_benefit_and_outcome_income_columns(
        self, entity: str, column: str
    ) -> None:
        from microcosm.frame.reporting import validate_income_classifier

        classifier = _load("reporting_classifiers.json")["income_decile"]
        # Resolve only the unavailable amount declarations for this contract
        # test; these fixture names make no claim about available engine outputs.
        for index, component in enumerate(classifier["components"]):
            if component["column"] is None:
                component["column"] = f"fixture_before_tax_amount_{index}"
        classifier["components"].append(
            {
                "entity": entity,
                "column": column,
                "kind": "market",
                "tax_basis": "before_tax",
                "annualisation_factor": 1,
            }
        )
        with pytest.raises(ValueError, match="Excluded income component"):
            validate_income_classifier(classifier)

    def test_mc11_and_receipt_group_select_one_superannuitant_predicate(self) -> None:
        from microcosm.build.transport.compose import load_transport_spec
        from microcosm.frame.reporting import validate_receipt_classifier

        document = _load("reporting_classifiers.json")
        classifier = validate_receipt_classifier(document["receipt_group"])
        reference = _load("as_rate_bridge.json")["superannuitant_units"]["predicate"]
        selected = load_transport_spec("nz")["resources"][reference["resource"]]
        for key in reference["path"]:
            selected = selected[key]
        assert selected == classifier["predicates"]["superannuitant"]
        assert selected == {
            "entity": "person",
            "columns": ["receives_nz_super"],
            "reduction": "any",
        }
        assert classifier["priority"] == ["superannuitant", "beneficiary"]
        assert classifier["default"] == "other"
        assert classifier["predicates"]["beneficiary"]["columns"] == [
            "receives_jobseeker_support",
            "receives_sole_parent_support",
            "receives_supported_living_payment",
        ]

    def test_unsupported_veterans_receipt_is_explicit_and_excluded(self) -> None:
        document = _load("reporting_classifiers.json")
        declared = _load("axiom_input_closure.json")["receipt_flags"]["columns"]
        (unsupported,) = document["unsupported_receipt_inputs"]
        assert unsupported["column"] == "receives_veterans_pension"
        assert f"person.{unsupported['column']}" not in declared
        assert (
            unsupported["column"]
            not in document["receipt_group"]["predicates"]["superannuitant"]["columns"]
        )
        assert "unsupported" in unsupported["required_behavior"].lower()
        assert document["mixed_unit_audit"]["per_scenario"] is True

    def test_as_non_beneficiary_binding_agrees_for_supported_receipt_flags(
        self,
    ) -> None:
        document = _load("reporting_classifiers.json")
        classifier = document["receipt_group"]
        expected = {
            f"person.{column}"
            for predicate in classifier["predicates"].values()
            for column in predicate["columns"]
        }
        entries = _load("axiom_input_closure.json")["entries"]
        (entry,) = [
            row
            for row in entries
            if row["input"] == "accommodation_supplement_non_beneficiary"
        ]
        binding = entry["state_binding"]
        assert binding["test"] == "any_true"
        assert binding["group_rule"] == "any_member"
        assert binding["negate"] is True
        assert "person.receives_nz_super" in binding["columns"]
        assert expected == set(binding["columns"])
        assert "unsupported" in document["as_non_beneficiary_audit"]["veterans_pension"]

    def test_baseline_cells_are_declared_for_every_scenario_and_variant(self) -> None:
        document = _load("reporting_classifiers.json")
        contract = document["baseline_cell_contract"]
        assert contract["scenario_ids"] == [
            row["id"] for row in _load("scenarios.json")["scenarios"]
        ]
        assert (
            contract["boundary_scenario"]
            == document["income_decile"]["boundary_scenario"]
            == "S0"
        )
        assert contract["join_column"] == "family_id"
        assert contract["variant_rule"] == "carry_baseline_cell_by_unit_id"
        assert contract["diagnostics"] == "report_each_scenario_weighted_decile_shares"

    def test_pinned_donor_unit_ids_survive_rebuilds_and_row_permutations(self) -> None:
        import pandas as pd

        from microcosm.frame.unit_construction import (
            BenefitUnitRule,
            DependentChildRule,
            build_benefit_units,
        )

        person = pd.DataFrame(
            {
                "person_id": [102, 101, 202, 201],
                "person_household_id": [10, 10, 20, 20],
                "age": [30, 30, 30, 30],
                "partner_person_id": pd.array([101, 102, None, None], dtype="Int64"),
                "parent_1_person_id": pd.array([None] * 4, dtype="Int64"),
                "parent_2_person_id": pd.array([None] * 4, dtype="Int64"),
            }
        )
        household = pd.DataFrame(
            {
                "household_id": [10, 20],
                "reference_person_id": [102, 202],
                "source_household_id": ["same_source", "same_source"],
            }
        )
        rule = BenefitUnitRule(
            entity="family",
            dependent_child=DependentChildRule(max_age=17, financial_independence=None),
            unparented_child="reference_person_unit",
            split_parents="first_parent_unit",
            provenance={"purpose": "Adult-only stable-id reporting fixture"},
        )
        units, membership = build_benefit_units(person, household, rule)
        repeated, repeated_membership = build_benefit_units(person, household, rule)
        permuted, permuted_membership = build_benefit_units(
            person.iloc[[3, 1, 0, 2]], household.iloc[::-1], rule
        )
        pd.testing.assert_frame_equal(units, repeated)
        pd.testing.assert_frame_equal(units, permuted)
        pd.testing.assert_series_equal(membership, repeated_membership)
        pd.testing.assert_series_equal(membership, permuted_membership.sort_index())
        assert units["family_id"].tolist() == [102, 201, 202]
        assert units["family_source_household_id"].tolist() == ["same_source"] * 3

    def test_s0_through_s3_structural_zero_cells_are_listed_on_support_entry(
        self,
    ) -> None:
        document = _load("reporting_classifiers.json")
        zeros = document["structural_zero_cells"]
        assert zeros == [
            {
                "scenario": f"S{index}",
                "cell": "beneficiary_superannuitant_other/superannuitant",
            }
            for index in range(4)
        ]
        support = next(
            row
            for row in _load("gates.json")["gates"]
            if row["id"] == "reported_cell_support"
        )
        for row in zeros:
            assert f"{row['scenario']}/{row['cell']}" in support["notes"]
        assert (
            "reported amounts determine actual structural-zero diagnostics"
            in (support["notes"])
        )
