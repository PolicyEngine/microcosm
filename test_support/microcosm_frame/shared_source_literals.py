"""Find country and donor-digest literals in a shared module's source.

A shared transport module selects donor behaviour through its parameters. It
holds no country name or code, no country-scoped node id, and no donor digest
or digest prefix, so no branch can key on one.

The check is a heuristic over the module's string and bytes constants
(f-string parts included). It flags a string that holds:

- one of these country names or parts of names, in any letter case,
  anywhere in the string: ``zealand``, ``belgi`` (Belgium, Belgian),
  ``united kingdom`` (with a space, hyphen, underscore or nothing between
  the words), ``britain``, ``british``, ``england``, ``scotland`` or
  ``aotearoa``;
- the code ``nz``, ``nzl``, ``bel``, ``uk`` or ``gbr`` standing alone, or
  ``be`` or ``gb`` as the whole string (both are also English, as in "must
  be" and "32 GB");
- a country-scoped node id such as ``nz.create``, in any letter case;
- a run of hex characters 64 or more long, whatever its composition, or 8 or
  more long with at least one letter, except the hex alphabet itself.

It does not catch a branch on the donor's own code, ``"us"``, which it allows
because the donor is a US survey and the reader labels its strata with it. It
passes runs of fewer than 64 digits (dates and counts look like them), so a
digest prefix made only of digits passes, as does any prefix shorter than
eight characters. It does not see a literal assembled at run time from
shorter parts or read from a parameter or file, a name in an identifier,
comment or non-string constant, ``be`` or ``gb`` inside a longer string, any
country name or spelling not listed above (Wales, Canada, a French or German
name for New Zealand), or a country named by a numeric code.
"""

from __future__ import annotations

import ast
import re

# The hex alphabet is not a digest.
_HEX_ALPHABET = "0123456789abcdef"
# A maximal run of eight or more hex characters, in either letter case.
_HEX_RUN = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{8,}")
# A SHA-256 is 64 hex characters.
_SHA256_LENGTH = 64
_COUNTRY_NAME = re.compile(
    r"(?i)zealand|belgi|united[\s_-]*kingdom|britain|british|england"
    r"|scotland|aotearoa"
)
# Codes that are not English words, wherever they stand in a string.
_COUNTRY_CODE = re.compile(r"(?i)(?<![a-z0-9])(nz|nzl|bel|uk|gbr)(?![a-z0-9])")
# "be" and "gb" are also words, so they count only as a whole string.
_BARE_CODE = re.compile(r"(?i)(?:be|gb)")
# A country-scoped node id such as "nz.create" or "BE.create".
_NODE_ID = re.compile(r"(?i)\b(nz|be|uk|us)\.[a-z_]")


def _digest_like(run: str) -> bool:
    """Whether a hex run could be a digest, or a prefix long enough to name
    a file. Runs of fewer than 64 digits are dates and counts."""
    if run.lower() == _HEX_ALPHABET:
        return False
    return len(run) >= _SHA256_LENGTH or not run.isdigit()


def country_or_digest_literals(source: str) -> list[str]:
    """String constants in ``source`` that name a country or a digest.

    The module docstring says what the check flags and what it misses.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bytes):
            text = node.value.decode("latin-1")
        elif isinstance(node.value, str):
            text = node.value
        else:
            continue
        if (
            any(_digest_like(match.group(0)) for match in _HEX_RUN.finditer(text))
            or _COUNTRY_NAME.search(text)
            or _COUNTRY_CODE.search(text)
            or _BARE_CODE.fullmatch(text)
            or _NODE_ID.search(text)
        ):
            found.append(text)
    return found
