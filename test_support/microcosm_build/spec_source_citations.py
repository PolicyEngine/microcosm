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
  decimal amount ("at 2500000"), a word ("at defaced"), a truncated hash
  ("at e2de2b41…") or a hex directory ("at a1b2c3d/foo.py") pins nothing;
- the resource holds a mapping, at any depth, whose ``path`` ends with the
  cited path and whose ``commit`` is a sha of the same form (``source``-style
  pins): a pin of ``upstream_data/datasets/cps/cps.py`` covers
  ``cps.py`` and ``datasets/cps/cps.py``, not ``other/cps.py``.

Each citation binds to one inline commit (``citing_commit``): the nearest one
in the citation's own clause, else the nearest one before it in the string,
unless the citation's clause names a moving ref (``on main``, ``@HEAD``,
``on `main` ``, ``/blob/master/``, ``raw.githubusercontent.com/<o>/<r>/main/``).
A clause ends at a ``;`` or a sentence end. So
"obr.py:302 on main; compare us-data at 42ed5d4" and "us-data at 42ed5d4;
obr.py:302 on main" pin nothing, and in "x.py:12 at <a>; y.py:30 at <b>" each
file binds its own commit. A commit-like token in no recognised
spelling, in a citation's clause or between the citation and the commit it
binds to ("archived 42ed5d45 cps.py L197"), is refused as a misspelled pin
(``misspelled_commit_pins``), so a stray sha cannot sit beside a citation
that some other commit pins.

Each URL is read the way a browser and then GitHub read it before anything
is scanned (``_read_url``). Its authority is parsed as the WHATWG URL
Standard parses it: ``\\`` read as ``/``, any number of slashes after the
scheme, a user and a port dropped, and the host percent-decoded, mapped to
ASCII (IDNA) and lower-cased, so ``%67ithub.com``, ``ｇithub.com`` and
``user@GitHub.com:443`` are all ``github.com``. Then, as GitHub does, ``%``
escapes of unreserved characters and ``/`` in its path are decoded, ``.`` and
``..`` segments resolved, and the query dropped. So
``HTTPS://GitHub.com/o/r/blob/<sha>/docs/../%74ools/x%2Epy?plain=1#L12``
cites ``https://github.com/o/r/blob/<sha>/tools/x.py`` at line 12. GitHub
reads ``%23`` and ``%3F`` as part of a file name, so they stay escaped. A URL
runs to whitespace: a quote or a bracket inside it is part of it, as it is to
a browser, so ``x.py?q='a'#L12`` still cites line 12 of ``x.py``. Markdown
and prose end a URL at a quote, a parenthesis or an angle bracket too, so a
string is also read that way where it reads differently
(``_readable_strings``): in ``[https://a/?x](https://github.com/...#L12)``
the label's query does not hide the link. Markdown and HTML decode a
backslash before punctuation and a character reference before a link is
followed, so a string holding one is read again with them decoded:
``[a](https://github.com/.../x\\.py&#35;L12)`` links to line 12 of
``x.py``. They are decoded as Markdown decodes them and, apart, as HTML does,
which leaves the backslash as written (``\\&#99;`` is ``\\c``, so
``href="https:\\&#47;github.com/..."`` names github.com); and each of those
once as text and once in a URL, where whitespace a reference decodes to is
percent-encoded, or deleted where it is a tab or a line break. A reference
is decoded as the HTML standard decodes it: a numeric one may lack its
``;``, one to 0, a surrogate or past U+10FFFF is U+FFFD, one to a C1
control is read as windows-1252 reads it, and in a URL, as in an attribute,
a legacy name without ``;`` before a letter, a digit or ``=`` stays as
written. The URL Standard deletes ASCII tabs and line breaks from a URL
wherever they are, so each reading is read again without them: in an HTML
attribute, ``.../x.p&#10;y#L12`` links to line 12 of ``x.py``. A browser
follows a quoted attribute value, and a Markdown destination in angle
brackets, as one URL: its escapes decoded, then its ends stripped (as the
URL Standard strips a URL, and markdown-it a destination) and the spaces
inside it percent-encoded. So a string holding one that starts a URL is read
again with it read so: ``href="&#32;https://github.com/.../a b/../x.py#L12"``
links to line 12 of ``x.py``. The URL Standard ends no URL at a Unicode space such as
U+00A0 (it percent-encodes one), so where a string holds one, its URLs are
read through it too. A citation any reading finds is checked.

A cited path is in-repository when, after any prefix naming this repository
is removed (a ``github.com/PolicyEngine/microcosm/blob/<ref>/`` URL, at a ref
of any depth such as ``refs/heads/main``, or ``microcosm/`` before a
top-level source directory) and its inner ``..`` segments resolved
(``tools/x/../y.py`` is ``tools/y.py``), it is a file here or one of the path
forms ``_repository_path_tails`` lists. A line citation of it is
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

import html
import importlib
import json
import os
import re
import string
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache, lru_cache
from html.entities import html5
from itertools import accumulate
from pathlib import Path, PurePosixPath
from urllib.parse import unquote_to_bytes

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
#: ``_QUOTE``, never given back (``*+``). Where it is used, what follows it
#: cannot start with a quote, or is another ``_QUOTES`` that would only take
#: the same quotes, so giving one back never helps a match.
_QUOTES = r"[`'\"‘’“”]*+"
#: A line number, plain or GitHub-style (``L12``).
_NUMBER = r"L?\s*+\d++(?!\w)"
#: One line or a range of them: ``12-19``, ``L12-L19``, ``12–19``,
#: ``12 to 19``, ``12 through 19``.
_SPAN = rf"{_NUMBER}(?:\s*+(?:[-–—]|(?i:to|through))\s*+{_NUMBER})?"
#: Several lines or ranges: ``12-19, 30``, ``12-19 and 30`` or
#: ``12, and 30``. Each run of whitespace is read by one possessive ``\s*+``
#: and each separator can be read only one way, so a list that fails is given
#: up in time linear in its length, not retried once for each way to share
#: its whitespace out.
_NUMBERS = rf"{_SPAN}(?:\s*+(?:,\s*+(?:(?i:and)\s*+)?|(?i:and)\s*+){_SPAN})*"
#: A line label: ``line``, ``lines`` (any case), ``l.`` or ``ll.``.
_LABEL = r"(?i:lines?\b\s++|ll?\.\s*+)"
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
#: quoted or backticked (```x.py` lines 12-19``, ``x.py:"12-19"``). Each run
#: of whitespace is read by one possessive ``\s*+``, and the space before a
#: spaced ``-`` is looked behind for (``(?<=\s)``), so a locator that fails
#: does so in time linear in its length.
_FORWARD_LOCATOR = re.compile(
    _SOURCE
    + _QUOTES
    + rf"(?::(?={_QUOTE}L?\d)"
    + rf"|\s*+[:：]\s*+{_QUOTES}{_LABEL}"
    + rf"|\s*+@\s*+{_QUOTES}(?:{_LABEL}|(?=L\d))"
    + rf"|#(?={_QUOTE}L\d)"
    + r"|\s*+(?:[,;—–]|(?<=\s)-(?=\s))?\s*+[\[(]?\s*+(?:(?i:at|on)\s++)?"
    + rf"{_QUOTES}(?:{_LABEL}|(?=L\d)))"
    + _QUOTES
    + _NUMBERS
)
_REVERSE_LOCATOR = re.compile(
    rf"\b(?:{_LABEL}|(?=L\d)){_QUOTES}{_NUMBERS}{_QUOTES}\s++(?i:of|in)\s++{_QUOTES}"
    + _SOURCE
)
_LINE_CITATIONS = (_FORWARD_LOCATOR, _REVERSE_LOCATOR)
#: How a reverse locator starts: a label or an ``L`` number, and the list of
#: line numbers after it. What the locator needs after its list (`` of ``,
#: `` in ``) cannot continue the list, so it matches after the whole list or
#: not at all.
_REVERSE_HEAD = re.compile(rf"\b(?:{_LABEL}|(?=L\d)){_QUOTES}{_NUMBERS}")
#: The last number of a list, where it is an ``L`` number a reverse locator
#: can start at (``L2`` in ``L1-L2``), searched for between a list's first
#: character and its end.
_LAST_L_NUMBER = re.compile(r"(?<!\w)L\d++\Z")
#: A source name followed by a labelled line locator (``L12``, ``line 12``,
#: ``ll. 12``) within 24 characters, with no ``.``, ``;`` or line break
#: between: the shape of every forward spelling, recognised or not. A bare
#: number after a colon is left out: ``x.json: 63 rows`` is a count.
_LOCATOR_NEAR_A_SOURCE = re.compile(
    _SOURCE
    + _QUOTE
    + r"[^.;\n]{0,24}?(?<![A-Za-z0-9])(?=L\d|(?i:lines?)\s+\d|(?i:ll?)\.\s*\d)"
)
#: The patterns that start with a ``_SOURCE``, each tried only where a source
#: name can start a match (see ``_finditer``): after a character no path holds,
#: or at ``http(s):``.
_STARTING_A_RUN = {
    pattern: re.compile(r"(?:(?<![A-Za-z0-9_./-])|(?=https?:))" + pattern.pattern)
    for pattern in (_FORWARD_LOCATOR, _LOCATOR_NEAR_A_SOURCE)
}
#: Where a URL starts: at ``http(s):``, at a network-path reference
#: (``//host/...``, any two of ``/`` and ``\``), or at a GitHub host written
#: without a scheme (``github.com/...``, before an optional port; a user
#: before it is left as prose). A GitHub host is matched where it starts, so
#: a long run of punctuation is scanned once.
_URL_START = (
    r"(?:(?i:https?):"
    r"|(?<![\w./\\@-])[/\\]{2}"
    r"|(?<![\w./\\-])"
    r"(?i:(?:www\.)?github\.com|raw\.githubusercontent\.com)\.?(?::[0-9]*)?(?=[/\\]))"
)
#: A URL as a browser reads it: it runs to whitespace, so a quote or an angle
#: bracket inside it is part of it, and it does not end in a quote, a bracket
#: or punctuation that closes the sentence around it (``'``, ``>``, ``.``,
#: ``,``, ``;``, ``)``, ...).
_URL = re.compile(_URL_START + r"\S*[^\s\"'`‘’“”<>.,;:!?)\]]")
#: A URL as Markdown and prose delimit it: as ``_URL``, but it also ends at a
#: quote, a parenthesis or an angle bracket, so the label and the destination
#: of ``[https://a/?x](https://b/...)``, the two autolinks of
#: ``<https://a/?x><https://b/...>`` and a URL quoted straight after another
#: are each read on their own.
_DELIMITED_URL = re.compile(_URL_START + r"[^\s\"'`‘’“”<>()]*[^\s\"'`‘’“”<>()\].,;:!?]")
_URLS = (_URL, _DELIMITED_URL)
#: ``_URLS`` as a Markdown destination, an HTML attribute and the URL Standard
#: hold a URL: only ASCII whitespace (``[\t\n\f\r ]``) ends one, so a no-break
#: space or another Unicode space inside it is part of it
#: (``.../tools/\u00a0x/../x.py``). They are tried where a string holds one.
_URLS_THROUGH_UNICODE_SPACES = (
    re.compile(_URL_START + r"[^\t\n\f\r ]*[^\s\"'`‘’“”<>.,;:!?)\]]"),
    re.compile(_URL_START + r"[^\t\n\f\r \"'`‘’“”<>()]*[^\s\"'`‘’“”<>()\].,;:!?]"),
)
#: Whitespace other than ASCII whitespace: what ``_URLS_THROUGH_UNICODE_SPACES``
#: reads a URL through.
_UNICODE_SPACE = re.compile(r"[^\S\t\n\f\r ]")
#: A character reference: decimal (``&#35;``) or hexadecimal (``&#x23;``),
#: with or without its ``;``, or named (``&num;``, or a legacy name such as
#: ``&amp`` without one).
_REFERENCE = (
    r"&#(?:[xX](?P<hexadecimal>[0-9A-Fa-f]++)|(?P<decimal>[0-9]++));?"
    r"|&(?P<name>[A-Za-z][A-Za-z0-9]{0,31};?)"
)
#: What Markdown decodes in text and in a link destination: a backslash
#: before ASCII punctuation (``x\.py``) and a character reference.
_MARKDOWN_ESCAPE = re.compile(r"\\(?P<punctuation>[!-/:-@\[-`{-~])|" + _REFERENCE)
#: What HTML decodes in text and in an attribute: a character reference. A
#: backslash is left as written, so in ``tools\&#99;i.py`` the reference is
#: decoded and the URL Standard then reads the backslash as ``/``.
_HTML_REFERENCE = re.compile(_REFERENCE)
#: What a numeric reference to a C1 control reads as in HTML: the character
#: windows-1252 puts there, where it puts one (``&#128;`` is ``€``).
_C1_REPLACEMENTS = {
    number: bytes([number]).decode("cp1252")
    for number in range(0x80, 0xA0)
    if number not in (0x81, 0x8D, 0x8F, 0x90, 0x9D)
}
#: What a URL can hold but a reading would end it at: whitespace, a quote, an
#: angle bracket or a parenthesis. A browser deletes an ASCII tab or line
#: break from a URL and percent-encodes the rest (``_hold``).
_HELD_IN_A_URL = re.compile(r"[\s\"'`‘’“”<>()]")
#: Where a URL can hold any whitespace: an HTML attribute's quoted value
#: (``href="..."``, ``href='...'``), to its closing quote, and a Markdown
#: destination in angle brackets (``[a](<...>)``, ``[a]: <...>``), to an
#: unescaped ``>``.
_ENCLOSED_URL = re.compile(
    r"=[\t\n\f\r ]*+(?:\"(?P<double>[^\"]*+)\"|'(?P<single>[^']*+)')"
    r"|\](?:\(|:[\t ]*+(?:\r\n?|\n)?[\t ]*+)<(?P<angle>(?:\\[^\n\r]|[^<>\n\r\\])*+)>"
)
#: A URL start at the start of a string.
_STARTS_A_URL = re.compile(_URL_START)
#: What the URL Standard strips from both ends of a URL: C0 controls and
#: spaces.
_C0_OR_SPACE = "".join(map(chr, range(0x21)))
#: What the URL Standard deletes from a URL, wherever it is, before reading
#: it: an ASCII tab or line break.
_URL_BREAKS = re.compile(r"[\t\n\r]")
#: The scheme a URL starts with, when it has one.
_SCHEME = re.compile(r"(?i:https?):")
#: A URL up to its query or fragment.
_BEFORE_QUERY = re.compile(r"[^?#]*")
#: A port after a host: ASCII digits, if any.
_PORT = re.compile(r"(?::(?P<number>[0-9]*))?")
#: The characters the WHATWG URL Standard refuses in a host once it is
#: decoded and mapped to ASCII (its forbidden domain code points).
_FORBIDDEN_IN_HOST = re.compile(r"[\x00-\x20#%/:<>?@\[\\\]^|\x7f]")
#: The default-ignorable code points, which UTS46 drops from a host or
#: refuses in one.
_IGNORABLE_IN_HOST = re.compile(
    "[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f"
    "\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff0-\ufff8"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff]"
)
#: The code points UTS46 maps into ASCII that this Python's Unicode database
#: (16.0) does not know: Unicode 17's U+A7F1 MODIFIER LETTER CAPITAL S.
_MAPPED_AFTER_UNICODE_16 = str.maketrans({"\ua7f1": "s"})
#: The most characters a DNS label holds (RFC 1035 section 2.3.4).
_LONGEST_LABEL = 63
#: A ``%`` escape in a URL.
_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
#: The characters a ``%`` escape in a URL path is read as: the unreserved ones
#: (RFC 3986 section 2.3), and ``/``, which GitHub reads as a separator after
#: the ref (``tools%2Fx.py``). Any other escape stays as written.
_DECODED = frozenset(string.ascii_letters + string.digits + "-._~/")
#: A commit sha: 7-40 lowercase hex with at least one digit and one letter,
#: so a decimal amount ("2500000"), a word ("defaced") or a branch name such
#: as ``main`` is not one.
_SHA = r"(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}"
#: A commit pin's value in a ``path``/``commit`` mapping.
_COMMIT_PIN = re.compile(_SHA)
#: A commit named inline. The sha must not run on into a file name
#: (``at a1b2c3d.py:12`` names a file), a longer token (``a1b2c3d-x``) or an
#: ellipsis (``e2de2b41…`` and ``e2de2b41...`` are truncated hashes). After a
#: prose spelling (``commit``, ``at``, ``@``) it must not run on into a path
#: either: in ``at a1b2c3d/foo.py:12`` the hex is a directory. A URL segment
#: is always followed by one.
INLINE_COMMIT = re.compile(
    r"(?:(?P<prose>(?i:\bcommit\s+|\bat\s+)|@)|/(?:blob|tree|raw|blame|commits?)/"
    r"|githubusercontent\.com/[^/\s]+/[^/\s]+/)"
    + r"(?P<sha>"
    + _SHA
    + r")(?![0-9A-Za-z_…-]|\.[0-9A-Za-z_.])(?(prose)(?!/))"
)
#: A commit-like token in any spelling: not part of a path, a URL, an
#: identifier, a longer token or a truncated hash.
_COMMIT_LIKE = re.compile(
    r"(?<![0-9A-Za-z_/@.…-])" + _SHA + r"(?![0-9A-Za-z_/…-]|\.[0-9A-Za-z_.])"
)
#: A branch a moving ref names.
_BRANCH = r"(?:refs/heads/)?(?:main|master|HEAD)"
#: A moving ref named in prose, the branch quoted or not (``on main``,
#: ``at `HEAD` ``, ``@master``), or a URL at one: a ``/blob/main/`` segment or
#: ``raw.githubusercontent.com/<owner>/<repo>/main/``, either also under
#: ``refs/heads/``.
_MOVING_REF = re.compile(
    rf"\b(?i:on|at)\s+{_QUOTE}{_BRANCH}\b|@{_QUOTE}{_BRANCH}\b"
    rf"|/(?:blob|tree|raw|blame)/{_BRANCH}/"
    rf"|githubusercontent\.com/[^/\s]+/[^/\s]+/{_BRANCH}/"
)
#: A clause break: a semicolon, or the space after a sentence end before a
#: capital, a bracket or a backtick.
_CLAUSE_BREAK = re.compile(r";|(?<=[.!?])\s+(?=[A-Z(\[`])")
#: Top-level directories whose files are this repository's sources.
_SOURCE_ROOTS = ("packages", "tools", "experiments", "test_support", "docs")
_SKIPPED_DIRECTORIES = {"__pycache__", ".venv", "node_modules", ".git"}
#: A prefix naming this repository before a path in it: a GitHub URL of a
#: file here, up to its ref (``url``), or ``microcosm/`` (after any ``./`` or
#: ``../`` and an optional ``PolicyEngine/``) before a top-level source
#: directory.
_THIS_REPOSITORY = re.compile(
    r"(?P<url>(?:https?:)?/*(?:www\.)?github\.com/(?i:policyengine/microcosm)/"
    r"(?:blob|tree|raw|blame)/"
    r"|(?:https?:)?/*raw\.githubusercontent\.com/(?i:policyengine/microcosm)/)"
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

    Both read each URL as ``_read_url`` reads it. ``start`` and ``end``
    locate the citation in ``text`` (-1 when built by hand); they do not take
    part in equality.
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


def _decode(path: str) -> str:
    """A URL path with each escape of a ``_DECODED`` character decoded."""

    def read(escape: re.Match[str]) -> str:
        character = chr(int(escape[1], 16))
        return character if character in _DECODED else escape[0]

    return _ESCAPE.sub(read, path)


def _remove_dot_segments(path: str) -> str:
    """A URL path with its ``.`` and ``..`` segments resolved (RFC 3986 5.2.4)."""

    segments: list[str] = []
    for segment in path.split("/"):
        if segment == "..":
            if len(segments) > 1:
                segments.pop()
        elif segment != ".":
            segments.append(segment)
    if path.endswith(("/.", "/..")):
        segments.append("")
    return "/".join(segments)


def _to_ascii(domain: str) -> str:
    """A decoded host mapped to ASCII as UTS46 (IDNA) maps it.

    Default-ignorable code points are dropped, the rest NFKC case-folded
    (``ｇ`` and ``ₒ`` are ``g`` and ``o``), ``。`` read as ``.``, and a label
    left outside ASCII written in Punycode. Where UTS46 refuses a character
    this reads it anyway, so it never misses a host UTS46 reads. A label
    longer than DNS allows (``_LONGEST_LABEL``) is left unencoded: it names
    no host GitHub serves either way, and Python's Punycode encoder takes
    time quadratic in a label's length.
    """

    domain = _IGNORABLE_IN_HOST.sub("", domain).translate(_MAPPED_AFTER_UNICODE_16)
    folded = unicodedata.normalize(
        "NFKC", unicodedata.normalize("NFKC", domain).casefold()
    ).replace("\u3002", ".")
    return ".".join(
        label
        if label.isascii() or len(label) > _LONGEST_LABEL
        else "xn--" + label.encode("punycode").decode("ascii")
        for label in folded.split(".")
    )


def _read_host(authority: str) -> str | None:
    """The host in a URL's authority, as the WHATWG URL Standard reads it.

    The user (up to the last ``@``) and the port are dropped; the host is
    percent-decoded and mapped to ASCII by ``_to_ascii``. A trailing dot,
    which DNS ignores, is dropped too: GitHub redirects ``github.com.`` to
    ``github.com``. ``None`` when the standard refuses the authority: a port
    that is not a number up to 65535, an empty host, or a host that is not
    UTF-8 or holds a ``_FORBIDDEN_IN_HOST`` character once read.
    """

    host = authority.rpartition("@")[2]
    if host.startswith("["):
        # An IPv6 address, kept as written: GitHub serves no URL by address.
        address, bracket, port = host.partition("]")
        host = address + bracket if bracket else ""
    else:
        host, colon, port = host.partition(":")
        port = colon + port
    number = _PORT.fullmatch(port)
    if not host or number is None:
        return None
    digits = (number["number"] or "").lstrip("0")
    if len(digits) > 5 or int(digits or 0) > 65535:
        return None
    if host.startswith("["):
        return host.lower()
    try:
        host = _to_ascii(unquote_to_bytes(host).decode("utf-8"))
    except UnicodeError:
        return None
    if _FORBIDDEN_IN_HOST.search(host):
        return None
    return host.removesuffix(".")


def _read_url(url: re.Match[str]) -> str:
    """A URL as a browser and then GitHub read it, without its query.

    An ``http(s)`` URL, or a network-path reference read against an
    ``https`` page, is parsed as the WHATWG URL Standard parses it: ``\\``
    before the query is ``/``, any number of slashes may come before the
    authority, and ``_read_host`` reads the host. A GitHub URL written
    without its scheme is read as if ``https://`` came first. Then, as GitHub
    does, the path's ``_DECODED`` escapes are decoded and its dot segments
    resolved; the fragment (``#L12``) is kept. A URL the standard refuses is
    left as written.
    """

    text = url[0]
    scheme = _SCHEME.match(text)
    rest = text[scheme.end() :] if scheme else text
    before = _BEFORE_QUERY.match(rest)[0].replace("\\", "/")
    slashes = len(before) - len(before.lstrip("/"))
    authority, slash, path = before[slashes:].partition("/")
    host = _read_host(authority)
    if host is None:
        return text
    lead = f"{scheme[0].lower()}//" if scheme else "//" if slashes else ""
    path = _remove_dot_segments(_decode(slash + path))
    fragment = rest.partition("#")[2]
    return f"{lead}{host}{path}{'#' + fragment if fragment else ''}"


def _code_point(digits: str, base: int) -> str:
    """The character a numeric reference names, as HTML reads it.

    Leading zeros are dropped first, so a long run of digits is never
    converted whole. Zero, a surrogate and a number past U+10FFFF are U+FFFD,
    a C1 control is its ``_C1_REPLACEMENTS`` character where it has one, and
    any other code point, a control included, is itself.
    """

    digits = digits.lstrip("0")
    if len(digits) > (6 if base == 16 else 7):
        return "\ufffd"
    number = int(digits or "0", base)
    if number == 0 or number > 0x10FFFF or 0xD800 <= number <= 0xDFFF:
        return "\ufffd"
    return _C1_REPLACEMENTS.get(number, chr(number))


def _named_in_an_attribute(reference: re.Match[str]) -> str:
    """A named reference as HTML decodes it in an attribute.

    The longest name in HTML's table that the reference starts with is read,
    as in text, but a legacy name read without its ``;`` and followed by a
    letter, a digit or ``=`` is left as written (``&notit;``, ``&amp=``).
    """

    name = reference["name"]
    for length in range(len(name), 1, -1):
        if name[:length] in html5:
            following = (
                name[length : length + 1]
                or reference.string[reference.end() : reference.end() + 1]
            )
            if name[length - 1] == ";" or not (
                following == "=" or following.isascii() and following.isalnum()
            ):
                return html5[name[:length]] + name[length:]
            break
    return reference[0]


def _decode_in_text(escape: re.Match[str]) -> str:
    """A ``_MARKDOWN_ESCAPE`` or ``_HTML_REFERENCE`` as text reads it.

    A named reference is read as ``html.unescape`` reads one in text, and a
    numeric one as ``_code_point`` reads it.
    """

    kind = escape.lastgroup
    if kind == "punctuation":
        return escape[kind]
    if kind == "name":
        return html.unescape(escape[0])
    return _code_point(escape[kind], 16 if kind == "hexadecimal" else 10)


def _hold(character: re.Match[str]) -> str:
    """A ``_HELD_IN_A_URL`` character as a browser holds it in a URL.

    An ASCII tab or line break is deleted and anything else percent-encoded.
    """

    if character[0] in "\t\n\r":
        return ""
    return "".join(f"%{byte:02X}" for byte in character[0].encode())


def _decode_in_an_attribute(escape: re.Match[str]) -> str:
    """A ``_MARKDOWN_ESCAPE`` or ``_HTML_REFERENCE`` as an attribute decodes it.

    A named reference is read as ``_named_in_an_attribute`` reads it, and any
    other escape as ``_decode_in_text`` reads it.
    """

    if escape.lastgroup == "name":
        return _named_in_an_attribute(escape)
    return _decode_in_text(escape)


def _decode_in_a_url(escape: re.Match[str]) -> str:
    """A ``_MARKDOWN_ESCAPE`` or ``_HTML_REFERENCE`` inside a URL.

    It is decoded as ``_decode_in_an_attribute`` decodes it, and what it
    stands for is held as a browser holds it (``_hold``): in
    ``href=https://.../x&#32;/../y.py`` the space neither ends the URL nor
    stops ``..`` from removing its segment.
    """

    return _HELD_IN_A_URL.sub(_hold, _decode_in_an_attribute(escape))


def _read_enclosed(enclosed: re.Match[str]) -> str:
    """An ``_ENCLOSED_URL`` with its URL as a browser takes it.

    An attribute's value has its references decoded
    (``_decode_in_an_attribute``), then the C0 controls and spaces at its ends
    stripped and its tabs and line breaks deleted, as HTML and then the URL
    Standard read it: ``href="&#32;https://..."`` is a URL. A destination's
    backslash escapes and references are decoded, as Markdown decodes them,
    and then the whitespace at its ends stripped, as markdown-it strips it
    (Python's ``str.strip``, in mdurl) before following a link:
    ``[a](< https://... >)`` is one. What the URL then still holds that a
    reading would end it at is held (``_hold``). A value that does not then
    start a URL is left as written.
    """

    kind = enclosed.lastgroup
    if kind == "angle":
        url = _MARKDOWN_ESCAPE.sub(_decode_in_an_attribute, enclosed[kind]).strip()
    else:
        url = _HTML_REFERENCE.sub(_decode_in_an_attribute, enclosed[kind])
        url = _URL_BREAKS.sub("", url.strip(_C0_OR_SPACE))
    if not _STARTS_A_URL.match(url):
        return enclosed[0]
    start, end = (index - enclosed.start() for index in enclosed.span(kind))
    return enclosed[0][:start] + _HELD_IN_A_URL.sub(_hold, url) + enclosed[0][end:]


def _readable_strings(value: object) -> Iterator[str]:
    """``nested_strings``, with the URLs in them read as ``_read_url`` reads them.

    Each string is read from several sources, each the string as one kind of
    reader takes it before following a link:

    - as written;
    - with each HTML attribute's quoted value and each Markdown destination
      in angle brackets read as a browser takes its URL (``_read_enclosed``),
      so a space inside one (``href=".../tools/ x/../y.py#L12"``) neither ends
      it nor stops ``..`` from removing its segment;
    - where it holds a backslash or a ``&``, decoded as Markdown decodes it
      (``_MARKDOWN_ESCAPE``) and as HTML decodes it (``_HTML_REFERENCE``),
      each as text reads it (``_decode_in_text``) and as a URL holds it
      (``_decode_in_a_url``): ``[a](https://github.com/.../x\\.py&#35;L12)``
      links to line 12 of ``x.py``, and so does
      ``<a href="https://github.com/...\\&#120;.py#L12">``, where HTML decodes
      the reference and leaves the backslash.

    Each source is read again with its ``_URL_BREAKS`` deleted, as the URL
    Standard reads a URL (``href=".../x.p&#10;y#L12"``). Each is read with each
    URL running to whitespace (``_URL``), as a browser reads it, and with each
    URL also ending at a quote, a parenthesis or an angle bracket
    (``_DELIMITED_URL``), as Markdown and prose delimit one, so a link's
    destination is not lost in the query of a URL before it. Where a source
    holds a Unicode space, its URLs are also read through Unicode spaces
    (``_URLS_THROUGH_UNICODE_SPACES``). Each reading that differs from the
    others is yielded once.
    """

    for text in nested_strings(value):
        sources = [text, _ENCLOSED_URL.sub(_read_enclosed, text)]
        if "&" in text or "\\" in text:
            sources += (
                escapes.sub(decode, text)
                for escapes in (_MARKDOWN_ESCAPE, _HTML_REFERENCE)
                for decode in (_decode_in_text, _decode_in_a_url)
            )
        yield from dict.fromkeys(
            pattern.sub(_read_url, joined)
            for source in dict.fromkeys(sources)
            for joined in dict.fromkeys((source, _URL_BREAKS.sub("", source)))
            for pattern in (
                _URLS + _URLS_THROUGH_UNICODE_SPACES
                if _UNICODE_SPACE.search(joined)
                else _URLS
            )
        )


def _finditer(pattern: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    """``pattern.finditer(text)``, in time linear in ``text``.

    A ``_SOURCE`` match that starts inside a run of the characters a path is
    made of (``[A-Za-z0-9_./-]``), other than at ``http(s):``, could start
    where the run does, so ``finditer`` starts each match at the start of a
    run, at ``http(s):``, or where its previous match ended. Trying only
    those starts, rather than every character of a run, finds the same
    matches.

    A reverse locator starts with a label or an ``L`` number and a list of
    line numbers (``_REVERSE_HEAD``). Inside a list it can start only at an
    ``L`` number. From one that starts a span (``L12`` in ``1, L12-19``), or
    ends a range a separator follows (``L19`` in ``L12-L19, 30``), it reads
    the rest of the list as the list's head did, to the same end, and what
    ends a reverse locator (`` of ``, `` in ``) cannot continue a list, so
    when it fails from the head it fails from there too. Only the list's last
    number can be read another way: when it ends a range and another dash
    follows (``L1-L2-3 of x.py``), from there it starts a range of its own
    (``L2-3``). So when a reverse locator fails from a list's head, it is
    tried next from the list's last number (``_LAST_L_NUMBER``), and
    otherwise after the list. Each number is read in at most two lists,
    rather than once from each number of a list.
    """

    if pattern is _REVERSE_LOCATOR:
        position = 0
        while head := _REVERSE_HEAD.search(text, position):
            match = pattern.match(text, head.start())
            if match is not None:
                yield match
                position = match.end()
                continue
            last = _LAST_L_NUMBER.search(text, head.start() + 1, head.end())
            position = head.end() if last is None else last.start()
        return
    anchored = _STARTING_A_RUN.get(pattern)
    if anchored is None:
        yield from pattern.finditer(text)
        return
    position = 0
    while match := pattern.match(text, position) or anchored.search(text, position):
        yield match
        position = match.end()


def _citation_matches(text: str) -> list[re.Match[str]]:
    return sorted(
        (match for pattern in _LINE_CITATIONS for match in _finditer(pattern, text)),
        key=lambda match: match.start(),
    )


def line_citations(value: object) -> list[LineCitation]:
    """Each line citation of a source in a payload, in string order.

    A string read more than one way (``_readable_strings``) gives the
    citations of each reading, so one that several readings find is listed
    once for each.
    """

    return [
        LineCitation(match["file"], text, match.start(), match.end())
        for text in _readable_strings(value)
        for match in _citation_matches(text)
    ]


def unrecognised_line_locators(value: object) -> list[LineCitation]:
    """Source names followed by a line locator in no recognised spelling."""

    found: list[LineCitation] = []
    for text in _readable_strings(value):
        matches = _citation_matches(text)
        files = {match.start("file") for match in matches}
        # A near match ending inside a citation is part of it: the citations
        # starting at or before its end reach past it.
        starts = [match.start() for match in matches]
        reach = list(accumulate((match.end() for match in matches), max))
        for near in _finditer(_LOCATOR_NEAR_A_SOURCE, text):
            started = bisect_right(starts, near.end())
            if near.start("file") in files or (
                started and reach[started - 1] > near.end()
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


@dataclass(frozen=True)
class _Marks:
    """Where a string's clause breaks, inline commits and commit-like tokens
    in no pin spelling are, found once for all the citations in it."""

    break_starts: list[int]
    break_ends: list[int]
    commits: list[re.Match[str]]
    commit_starts: list[int]
    commit_ends: list[int]
    loose: list[re.Match[str]]
    loose_starts: list[int]


@lru_cache(maxsize=1024)
def _marks(text: str) -> _Marks:
    breaks = list(_CLAUSE_BREAK.finditer(text))
    commits = list(INLINE_COMMIT.finditer(text))
    pinned = {match.start("sha") for match in commits}
    loose = [
        token for token in _COMMIT_LIKE.finditer(text) if token.start() not in pinned
    ]
    return _Marks(
        [match.start() for match in breaks],
        [match.end() for match in breaks],
        commits,
        [match.start() for match in commits],
        [match.end() for match in commits],
        loose,
        [token.start() for token in loose],
    )


@lru_cache(maxsize=1024)
def _names_a_moving_ref(text: str, start: int, end: int) -> bool:
    return _MOVING_REF.search(text, start, end) is not None


def _clause(citation: LineCitation) -> tuple[int, int]:
    """The bounds of the clauses of a citation's string that it touches.

    A break inside the citation (``x.py; lines 12-19``) does not split it.
    """

    if citation.start < 0:
        raise ValueError("only a citation from line_citations carries its position")
    marks = _marks(citation.text)
    before = bisect_right(marks.break_ends, citation.start)
    after = bisect_left(marks.break_starts, citation.end)
    first = marks.break_ends[before - 1] if before else 0
    last = (
        marks.break_starts[after]
        if after < len(marks.break_starts)
        else len(citation.text)
    )
    return first, last


def _binding(citation: LineCitation) -> re.Match[str] | None:
    """The inline commit match a line citation binds to, if any."""

    marks = _marks(citation.text)
    start, end = _clause(citation)
    # Commits do not overlap, so the nearest ones in the clause are among the
    # last that ends before the citation, those overlapping it and the first
    # that starts after it.
    first = bisect_right(marks.commit_ends, citation.start)
    after = bisect_left(marks.commit_starts, citation.end)
    nearest = marks.commits[max(first - 1, 0) : after + 1]
    in_clause = [match for match in nearest if start <= match.start() < end]
    if in_clause:
        return min(
            in_clause,
            key=lambda match: max(
                match.start() - citation.end, citation.start - match.end()
            ),
        )
    if _names_a_moving_ref(citation.text, start, end):
        return None
    before = bisect_left(marks.commit_starts, start)
    return marks.commits[before - 1] if before else None


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
        marks = _marks(citation.text)
        start, end = _clause(citation)
        binding = _binding(citation)
        if binding is not None and binding.end() < start:
            start = binding.end()
        # A token cannot cross a clause break or end a commit, so the ones in
        # the span are the string's that start in it.
        first = bisect_left(marks.loose_starts, start)
        last = bisect_left(marks.loose_starts, end)
        found.extend((citation, token.group(0)) for token in marks.loose[first:last])
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


@cache
def _repository_depth(root: Path) -> int:
    """The most components a path form in ``root`` has."""

    return max(
        (tail.count("/") + 1 for tail in _repository_path_tails(root)), default=0
    )


def resolves_in_repository(file: str, root: Path = REPOSITORY_ROOT) -> bool:
    """Whether a cited path names a file in this checkout.

    A prefix naming this repository is removed first, and each ``..`` with a
    segment before it removes that segment (``tools/x/../y.py`` is
    ``tools/y.py``); a path that climbs above where it starts (``../y.py``)
    names nothing here. What is left resolves when it is a file here (after
    such a prefix) or one of the path forms ``_repository_path_tails`` lists:
    ``tools/x.py``, ``microcosm/tools/x.py``,
    ``packages/microcosm-build/src/...``, ``microcosm/graph/executor.py`` and
    ``us_runtime/asec_pool.py`` all do.
    """

    named = _THIS_REPOSITORY.match(file)
    relative = PurePosixPath(file[named.end() :] if named else file)
    if relative.is_absolute():
        return False
    parts: list[str] = []
    for part in relative.parts:
        if part != "..":
            parts.append(part)
        elif parts:
            parts.pop()
        else:
            # It climbs out of a directory it does not name.
            return False
    if not parts:
        return False
    relative = PurePosixPath(*parts)
    candidates: Iterable[PurePosixPath] = [relative]
    if named is not None and named["url"]:
        # A URL's ref comes first and may span segments (``main``,
        # ``refs/heads/main``, ``feature/x``), so try the path after each.
        # No path form here is longer than ``_repository_depth`` segments, so
        # only the paths that short are tried, however long the ref.
        parts = relative.parts
        first = max(1, len(parts) - _repository_depth(root))
        candidates = (PurePosixPath(*parts[cut:]) for cut in range(first, len(parts)))
    tails = _repository_path_tails(root)
    return any(
        (named is not None and (root / candidate).is_file())
        or (len(candidate.parts) >= 2 and candidate.as_posix() in tails)
        for candidate in candidates
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
