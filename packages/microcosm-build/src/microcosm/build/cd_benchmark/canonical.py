"""Canonical JSON bytes and digests, exactly as the protocol defines them.

``/identity/encoding`` and ``/exposure_ledger/format`` require UTF-8 canonical
JSON with sorted keys, compact separators, no BOM, no nonfinite values and no
trailing LF inside the hashed bytes, and they additionally reject duplicate
object keys. The shared aggregate-JSON reader in
:mod:`microcosm.build.candidate_quality` enforces the first set but accepts
duplicate keys (``json.loads`` keeps the last one), so this module supplies the
stricter loader the contract names rather than relaxing the contract to match an
existing reader. The protocol also fixes ``ensure_ascii=false``, which that
reader leaves at the escaping default; for ASCII payloads the two encoders agree
byte for byte apart from its trailing newline, and a contract test pins that.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

__all__ = [
    "CanonicalJsonError",
    "SHA256_HEX",
    "canonical_bytes",
    "digest",
    "is_sha256_hex",
    "strict_load",
]

SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


class CanonicalJsonError(ValueError):
    """Bytes are not the canonical JSON the protocol requires."""


def is_sha256_hex(value: object) -> bool:
    """Return whether ``value`` is a lowercase 64-character hex digest."""
    return isinstance(value, str) and SHA256_HEX.fullmatch(value) is not None


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise CanonicalJsonError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _nonfinite(token: str) -> None:
    raise CanonicalJsonError(f"Nonfinite JSON value {token}.")


def strict_load(data: bytes) -> Any:
    """Parse canonical JSON bytes, refusing BOM, duplicate keys and nonfinite.

    Raises:
        CanonicalJsonError: On any of those, or on invalid UTF-8/JSON.
    """
    if not isinstance(data, bytes):
        raise CanonicalJsonError("Canonical JSON input must be bytes.")
    if data.startswith(b"\xef\xbb\xbf"):
        raise CanonicalJsonError("Canonical JSON must not begin with a BOM.")
    try:
        text = data.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except CanonicalJsonError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise CanonicalJsonError(f"Invalid canonical JSON: {error}") from None
    _assert_finite(value)
    return value


def _assert_finite(value: object) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float | int):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise CanonicalJsonError("JSON numbers must be finite.")
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite(child)


def canonical_bytes(value: object) -> bytes:
    """Serialize to the protocol's canonical UTF-8 JSON, with no trailing LF."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise CanonicalJsonError(f"Value is not canonical JSON: {error}") from None
    return text.encode("utf-8")


def digest(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of exact bytes."""
    if not isinstance(data, bytes):
        raise CanonicalJsonError("Digest input must be bytes.")
    return hashlib.sha256(data).hexdigest()
