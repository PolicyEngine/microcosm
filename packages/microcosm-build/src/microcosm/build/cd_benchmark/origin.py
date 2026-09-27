"""Typed original-source household origin keys, exactly as ``v4`` encodes them.

``/source_household_origin/key_encoding`` fixes the encoding:
``SHA256(UTF8(domain) + one NUL byte + canonical payload bytes)``, lowercase
hex, where the payload is a canonical UTF-8 JSON array of four ordered typed
pairs. Each pair is a two-element array whose first element is the literal type
name. Integers are JSON integers, never booleans, and raw strings receive no
normalization, trimming or coercion.

The two arms differ in their third and fourth components, and this module names
them per arm rather than generically, so a survey vintage cannot be filed as an
income cohort year:

* ACS — ``arm=ACS_PUMS``, authenticated original household *archive* member
  identity, **survey vintage**, and the **raw ``SERIALNO`` string** with its
  leading and lexical characters preserved.
* ASEC — ``arm=ASEC``, authenticated original household member identity,
  **income cohort year**, and the **native integer ``H_SEQ``**.

Two limits carried from the document. A generated ``source_household_id`` or a
normalized ``household_id`` is positional and cannot replace this identity
(``/source_household_origin/ACS/not_equivalent``). Cohort or vintage
distinguishes statistical source records; it does not prove different real
households across years, and an origin key does not establish independent
households across surveys or independent imputation donors.

Physical staging paths, recompressed archive hashes and clone or cache
filenames are not identities: the authenticated source manifest supplies
``canonical_member_id`` and the original member bytes hash, and aliases must
resolve to that same member before hashing. Typed components stay private in
authenticated lineage; public diagnostics use counts and digests only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from microcosm.build.cd_benchmark.canonical import canonical_bytes, is_sha256_hex

__all__ = [
    "AcsHouseholdOrigin",
    "AsecHouseholdOrigin",
    "HouseholdOrigin",
    "ORIGIN_DOMAIN",
    "OriginKeyError",
    "SourceArm",
    "SourceMember",
    "origin_key",
    "origin_payload",
    "origin_payload_bytes",
]

#: ``/source_household_origin/key_encoding/domain``.
ORIGIN_DOMAIN = "microcosm.statistical_source_household.v1"


class OriginKeyError(ValueError):
    """The typed origin components are not what the encoding accepts."""


class SourceArm(StrEnum):
    """The two arms with a registered origin definition.

    ``/source_household_origin/other_arms``: any other arm needs its own
    registered, authenticated source-specific definition; there is deliberately
    no default to a synthetic row id here.
    """

    ACS_PUMS = "ACS_PUMS"
    ASEC = "ASEC"


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OriginKeyError(f"{label} must be a nonempty string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OriginKeyError(f"{label} must be a JSON integer, not a boolean.")
    return value


@dataclass(frozen=True)
class SourceMember:
    """Canonical original member identity from the authenticated manifest."""

    canonical_member_id: str
    member_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.canonical_member_id, "canonical_member_id")
        if not is_sha256_hex(self.member_sha256):
            raise OriginKeyError(
                "member_sha256 must be a lowercase 64-character hex digest."
            )

    def as_payload(self) -> dict[str, str]:
        """The member pair's object value, with lexicographically sorted keys."""
        return {
            "canonical_member_id": self.canonical_member_id,
            "member_sha256": self.member_sha256,
        }


@dataclass(frozen=True)
class AcsHouseholdOrigin:
    """ACS PUMS origin: archive member identity, survey vintage, raw SERIALNO."""

    member: SourceMember
    survey_vintage: int
    raw_serialno: str

    arm = SourceArm.ACS_PUMS

    def __post_init__(self) -> None:
        if not isinstance(self.member, SourceMember):
            raise OriginKeyError("member must be an authenticated SourceMember.")
        _integer(self.survey_vintage, "survey_vintage")
        # No normalization, trimming or coercion: an integer SERIALNO would be a
        # coercion of the published string, and leading characters are meaning.
        if not isinstance(self.raw_serialno, str) or not self.raw_serialno:
            raise OriginKeyError("raw_serialno must be the nonempty published string.")


@dataclass(frozen=True)
class AsecHouseholdOrigin:
    """ASEC origin: member identity, income cohort year, native integer H_SEQ."""

    member: SourceMember
    income_cohort_year: int
    native_h_seq: int

    arm = SourceArm.ASEC

    def __post_init__(self) -> None:
        if not isinstance(self.member, SourceMember):
            raise OriginKeyError("member must be an authenticated SourceMember.")
        _integer(self.income_cohort_year, "income_cohort_year")
        # The shipped ASEC adapter's NATIVE_KEY invariant is H_SEQ > 0; a zero or
        # negative sequence is not a native household key.
        if _integer(self.native_h_seq, "native_h_seq") <= 0:
            raise OriginKeyError("native_h_seq must be a positive native key.")


HouseholdOrigin = AcsHouseholdOrigin | AsecHouseholdOrigin


def origin_payload(origin: HouseholdOrigin) -> list[list[object]]:
    """Return the four ordered typed pairs for ``origin``.

    Raises:
        OriginKeyError: If ``origin`` is not a registered arm's origin.
    """
    if isinstance(origin, AcsHouseholdOrigin):
        return [
            ["string", str(SourceArm.ACS_PUMS)],
            ["member", origin.member.as_payload()],
            ["integer", origin.survey_vintage],
            ["string", origin.raw_serialno],
        ]
    if isinstance(origin, AsecHouseholdOrigin):
        return [
            ["string", str(SourceArm.ASEC)],
            ["member", origin.member.as_payload()],
            ["integer", origin.income_cohort_year],
            ["integer", origin.native_h_seq],
        ]
    raise OriginKeyError(
        "Only registered ACS and ASEC origins have a defined encoding; another "
        "arm requires its own registered source-specific origin definition."
    )


def origin_payload_bytes(origin: HouseholdOrigin) -> bytes:
    """Canonical payload bytes: sorted object keys, compact, no BOM or LF."""
    return canonical_bytes(origin_payload(origin))


def origin_key(origin: HouseholdOrigin) -> str:
    """Return the lowercase hex origin key for one original source household."""
    payload = origin_payload_bytes(origin)
    return hashlib.sha256(ORIGIN_DOMAIN.encode("utf-8") + b"\x00" + payload).hexdigest()
