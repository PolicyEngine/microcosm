"""Find country and donor-digest literals in a shared module's source.

A shared transport module selects donor behaviour through its parameters. It
holds no country name or code, no country-scoped node id, and no donor digest
or digest prefix, so no branch can key on one.
"""

from __future__ import annotations

import ast
import re

# The hex alphabet is not a digest.
_HEX_ALPHABET = "0123456789abcdef"
# Eight or more hex characters holding a digit and a letter: a SHA-256, or a
# prefix of one long enough to name a file.
_DIGEST = re.compile(
    r"(?i)(?<![0-9a-f])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{8,}"
)
_COUNTRY_NAME = re.compile(r"(?i)zealand|belgi")
# Codes that are not English words, wherever they stand in a string.
_COUNTRY_CODE = re.compile(r"(?i)(?<![a-z0-9])(nz|nzl|bel|uk|gb|gbr)(?![a-z0-9])")
# "be" is also a word, so it counts only as a whole string.
_BARE_CODE = re.compile(r"(?i)be")
# A country-scoped node id such as "nz.create".
_NODE_ID = re.compile(r"\b(nz|be|uk|us)\.[a-z_]")


def country_or_digest_literals(source: str) -> list[str]:
    """String constants in ``source`` that name a country or a digest.

    The donor's own country code, ``"us"``, is allowed: the donor is a US
    survey, and the reader labels its strata with it.
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
        digests = [
            match.group(0)
            for match in _DIGEST.finditer(text)
            if match.group(0).lower() != _HEX_ALPHABET
        ]
        if (
            digests
            or _COUNTRY_NAME.search(text)
            or _COUNTRY_CODE.search(text)
            or _BARE_CODE.fullmatch(text)
            or _NODE_ID.search(text)
        ):
            found.append(text)
    return found
