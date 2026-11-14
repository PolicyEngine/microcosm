"""2024-25 Band D council tax for Great Britain, and the statutory band formula (microcosm#931).

Council-tax localisation prices a household's bill at the local authority the
spine draws for it. That needs each authority's Band D for the build's fiscal
year and the statute that turns a Band D into the bill of a band. Band D comes
from two committed resources, both keyed by the April 2023 codes of the local
authority roster ``local_authority_names.json``, through which the spine
resolves every drawn area:

- ``council_tax_band_d_england_wales_2024_25.json``: England's and Wales's
  2024-25 Band D, built from two raw-only Chronicle registrations by
  ``tools/build_uk_council_tax_band_d.py``. England's amounts are at pence;
  Wales's are the published decimals, unrounded.
- ``scotgov_band_d_council_tax_2024_25.json``: Scotland's 2024-25 average
  Band D for its 32 council areas (CTAS 2026, the record set
  ``scotgov.ctas2026.band_d.fy2024``), vendored from the pinned Chronicle feed
  by ``tools/vendor_uk_ledger_facts.py``. Its rows are read through
  ``vendored_rows``, which refuses a copy taken from another feed before it
  reads a row. Each amount is the loaded binary number; it must lie within
  1e-9 of a whole number of pence and is kept as that pence.

:func:`load_uk_council_tax_band_d` merges the two as exact rationals, each read
as a :class:`fractions.Fraction` through :class:`decimal.Decimal`. It refuses a
code outside the roster's Great Britain codes, a Northern Ireland code
(Northern Ireland levies domestic rates, not council tax), a missing or
repeated authority, and a value that is not a positive finite decimal. Every
value is used as published: none is overridden.

The statute (:func:`uk_statutory_council_tax`) multiplies an authority's Band
D by its band's proportion, by nation. England and Wales take 6, 7, 8, 9, 11,
13, 15 and 18 ninths for bands A to H (Local Government Finance Act 1992,
section 5), and Wales 21 ninths for band I (the Council Tax (Valuation Bands)
(Wales) Order 2003, from 2005-06). Scotland takes 240, 280, 320, 360, 473, 585,
705 and 882 three-hundred-and-sixtieths for A to H (section 74, as the Council
Tax (Substitution of Proportion) (Scotland) Order 2016 substituted it from
2017-18). A dwelling with a single adult resident takes the 25 per cent
discount (sections 11 and 79). The nations and each nation's bands are
``council_tax_constants``' ``UK_COUNCIL_TAX_BANDS_BY_NATION``, read here and
never edited.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from microcosm.build.uk_runtime.council_tax_constants import (
    UK_COUNCIL_TAX_BANDS_BY_NATION,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import (
    VENDORED_RESOURCE_KIND,
    load_vendored_resource,
    vendored_rows,
    verify_vendored_resource_feed_identity,
)
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_VINTAGE,
    load_uk_local_authority_names_resource,
)

__all__ = [
    "SCOTLAND_BAND_D_ARTIFACT_ROLE",
    "SCOTLAND_BAND_D_CONCEPT",
    "SCOTLAND_BAND_D_GEOGRAPHY_VINTAGE",
    "SCOTLAND_BAND_D_RECORD_SETS",
    "SCOTLAND_BAND_D_RESOURCE",
    "UK_BAND_D_ARTIFACT_ROLE",
    "UK_BAND_D_BUILD_PERIOD",
    "UK_BAND_D_ENGLAND_WALES_KIND",
    "UK_BAND_D_ENGLAND_WALES_METHOD",
    "UK_BAND_D_ENGLAND_WALES_RESOURCE",
    "UK_BAND_D_ENGLAND_WALES_SCHEMA_VERSION",
    "UK_BAND_D_FISCAL_YEAR",
    "UK_BAND_D_PENCE_TOLERANCE",
    "UK_BAND_D_PENCE_TOLERANCE_TEXT",
    "UK_COUNCIL_TAX_BAND_RATIOS",
    "UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT",
    "UKCouncilTaxBandD",
    "UKCouncilTaxBandDError",
    "band_d_decimal",
    "band_d_pence",
    "check_scotland_band_d_declaration",
    "load_uk_council_tax_band_d",
    "uk_statutory_council_tax",
]

#: England's and Wales's Band D: built by ``tools/build_uk_council_tax_band_d.py``
#: from the raw-only registrations, never edited by hand.
UK_BAND_D_ENGLAND_WALES_RESOURCE = "council_tax_band_d_england_wales_2024_25.json"
UK_BAND_D_ENGLAND_WALES_KIND = "uk_council_tax_band_d_england_wales"
UK_BAND_D_ENGLAND_WALES_SCHEMA_VERSION = 1
UK_BAND_D_ENGLAND_WALES_METHOD = "uk-council-tax-band-d-v1"
#: The artifact role under which a stage declares the England-and-Wales resource.
UK_BAND_D_ARTIFACT_ROLE = "council_tax_band_d"

#: Scotland's Band D: vendored from the pinned Chronicle feed by
#: ``tools/vendor_uk_ledger_facts.py``, never edited by hand.
SCOTLAND_BAND_D_RESOURCE = "scotgov_band_d_council_tax_2024_25.json"
#: The artifact role under which a stage declares the Scottish resource.
SCOTLAND_BAND_D_ARTIFACT_ROLE = "scotland_band_d_facts"
#: Every Chronicle record set read from the Scottish resource, as a stage's
#: ``verify_vendored_fact_resource`` operation declares them.
SCOTLAND_BAND_D_RECORD_SETS: tuple[str, ...] = ("scotgov.ctas2026.band_d.fy2024",)
SCOTLAND_BAND_D_CONCEPT = "scotgov.council_tax.average_band_d"
#: The council-area vintage of the Scottish rows: the 2019 council areas, which
#: the April 2023 roster carries for Scotland.
SCOTLAND_BAND_D_GEOGRAPHY_VINTAGE = "ca_2019"
_SCOTLAND_UNIT = "gbp"
_SCOTLAND_LEVEL = "local_authority"

#: The fiscal year of both resources, and the build period (its opening year)
#: the stages map them to.
UK_BAND_D_FISCAL_YEAR = "2024-25"
UK_BAND_D_BUILD_PERIOD = 2024

#: The farthest, in pounds, a stored binary amount may sit from a whole number
#: of pence and still be kept as that pence.
UK_BAND_D_PENCE_TOLERANCE_TEXT = "1e-9"
UK_BAND_D_PENCE_TOLERANCE = Fraction(Decimal(UK_BAND_D_PENCE_TOLERANCE_TEXT))

#: A GSS code's first letter names its nation. The England-and-Wales resource
#: covers E and W codes, the Scottish resource S codes, and Northern Ireland's N
#: codes have no Band D.
_ENGLAND_WALES_LETTERS = "EW"
_SCOTLAND_LETTERS = "S"
_NORTHERN_IRELAND_LETTER = "N"

_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")


class UKCouncilTaxBandDError(ValueError):
    """Raised when the Band D resources or the statute's inputs are refused."""


# ----------------------------------------------------------------------------
# The statute


def _proportions(
    nation: str, numerators: tuple[int, ...], denominator: int
) -> Mapping[str, Fraction]:
    """A nation's band proportions to Band D, over the bands it uses."""

    return MappingProxyType(
        {
            band: Fraction(numerator, denominator)
            for band, numerator in zip(
                UK_COUNCIL_TAX_BANDS_BY_NATION[nation], numerators, strict=True
            )
        }
    )


_NINTHS = (6, 7, 8, 9, 11, 13, 15, 18)

#: Each nation's statutory band proportions to Band D, by the engine's
#: ``country`` member: England's and Wales's ninths (Wales's band I since
#: 2005-06), and Scotland's three-hundred-and-sixtieths since 2017-18.
UK_COUNCIL_TAX_BAND_RATIOS: Mapping[str, Mapping[str, Fraction]] = MappingProxyType(
    {
        "ENGLAND": _proportions("ENGLAND", _NINTHS, 9),
        "WALES": _proportions("WALES", (*_NINTHS, 21), 9),
        "SCOTLAND": _proportions(
            "SCOTLAND", (240, 280, 320, 360, 473, 585, 705, 882), 360
        ),
    }
)

#: The discount on the bill of a dwelling with a single adult resident, in every
#: nation that levies council tax.
UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT = Fraction(1, 4)


def _exact_band_d(value: object) -> Fraction:
    exact = isinstance(value, int | Fraction | Decimal) and not isinstance(value, bool)
    if not exact:
        raise UKCouncilTaxBandDError(
            "Band D must be an exact rational (an int, Fraction or Decimal), "
            f"not a {type(value).__name__}."
        )
    if isinstance(value, Decimal) and not value.is_finite():
        raise UKCouncilTaxBandDError("Band D must be finite.")
    amount = Fraction(value)
    if amount <= 0:
        raise UKCouncilTaxBandDError("Band D must be positive.")
    return amount


def uk_statutory_council_tax(
    band_d: Fraction | int | Decimal, nation: str, band: str, *, single_adult: bool
) -> Fraction:
    """A dwelling's annual council tax under the statute, before any reduction.

    ``band_d`` is its authority's Band D in pounds, as an exact rational;
    ``nation`` is the engine's ``country`` member and ``band`` one of that
    nation's bands. The bill is Band D times the band's proportion, less
    ``UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT`` of it where ``single_adult``.
    """

    proportions = UK_COUNCIL_TAX_BAND_RATIOS.get(nation)
    if proportions is None:
        if nation in UK_COUNCIL_TAX_BANDS_BY_NATION:
            raise UKCouncilTaxBandDError(
                f"{nation} levies domestic rates, not council tax."
            )
        raise UKCouncilTaxBandDError(f"{nation!r} is not a council-tax nation.")
    if band not in proportions:
        raise UKCouncilTaxBandDError(f"{nation} has no council-tax band {band!r}.")
    if type(single_adult) is not bool:
        raise UKCouncilTaxBandDError("single_adult must be a bool.")
    bill = _exact_band_d(band_d) * proportions[band]
    if single_adult:
        bill *= 1 - UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT
    return bill


# ----------------------------------------------------------------------------
# Amounts


def band_d_decimal(text: object, *, where: str) -> Fraction:
    """A Band D written as a decimal, as an exact rational.

    The text must be digits with an optional fraction (no sign, exponent or
    separator), and the amount positive. ``where`` names the amount in a
    refusal, which never quotes it.
    """

    if not isinstance(text, str) or not _DECIMAL.fullmatch(text):
        raise UKCouncilTaxBandDError(f"{where} is not a positive decimal amount.")
    amount = Fraction(Decimal(text))
    if amount <= 0:
        raise UKCouncilTaxBandDError(f"{where} is not a positive decimal amount.")
    return amount


def band_d_pence(value: object, *, where: str) -> tuple[str, bool]:
    """A stored binary amount as its pence decimal, and whether it was exactly that.

    The amount must be a positive finite number within
    ``UK_BAND_D_PENCE_TOLERANCE`` of a whole number of pence. ``where`` names
    the amount in a refusal, which never quotes it.
    """

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise UKCouncilTaxBandDError(f"{where} is not a number.")
    if not math.isfinite(value) or value <= 0:
        raise UKCouncilTaxBandDError(f"{where} is not a positive finite amount.")
    exact = Fraction(value)
    pence = round(exact * 100)
    if pence <= 0 or abs(exact - Fraction(pence, 100)) > UK_BAND_D_PENCE_TOLERANCE:
        raise UKCouncilTaxBandDError(
            f"{where} is not within {UK_BAND_D_PENCE_TOLERANCE_TEXT} of a whole "
            "number of pence."
        )
    return str(Decimal(pence).scaleb(-2)), exact == Fraction(pence, 100)


def _nation_code(code: object, letters: str, *, resource: str) -> str:
    """A code the resource may carry: one of ``letters``' nations, never N."""

    if not isinstance(code, str) or not code:
        raise UKCouncilTaxBandDError(
            f"{resource} carries a Band D without a local authority code."
        )
    if code.startswith(_NORTHERN_IRELAND_LETTER):
        raise UKCouncilTaxBandDError(
            f"{resource} carries the Northern Ireland code {code!r}; Northern "
            "Ireland levies domestic rates, not council tax."
        )
    if code[0] not in letters:
        raise UKCouncilTaxBandDError(
            f"{resource} carries {code!r}, a code of a nation it does not cover."
        )
    return code


# ----------------------------------------------------------------------------
# The resources


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object's members, refusing a key written twice."""

    members: dict[str, Any] = {}
    for key, value in pairs:
        if key in members:
            raise UKCouncilTaxBandDError(
                f"{UK_BAND_D_ENGLAND_WALES_RESOURCE} repeats the key {key!r}."
            )
        members[key] = value
    return members


def _england_wales_payload(
    payload: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], str]:
    if payload is not None:
        return payload, "supplied"
    data = (
        files("microcosm.build.uk")
        .joinpath(UK_BAND_D_ENGLAND_WALES_RESOURCE)
        .read_bytes()
    )
    return (
        json.loads(data, object_pairs_hook=_unique_keys),
        hashlib.sha256(data).hexdigest(),
    )


def _england_wales_band_d(payload: Mapping[str, Any]) -> dict[str, Fraction]:
    resource = UK_BAND_D_ENGLAND_WALES_RESOURCE
    for key, expected in (
        ("schema_version", UK_BAND_D_ENGLAND_WALES_SCHEMA_VERSION),
        ("kind", UK_BAND_D_ENGLAND_WALES_KIND),
        ("resource", UK_BAND_D_ENGLAND_WALES_RESOURCE),
        ("fiscal_year", UK_BAND_D_FISCAL_YEAR),
        ("local_authority_vintage", UK_LOCAL_AUTHORITY_VINTAGE),
    ):
        actual = payload.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise UKCouncilTaxBandDError(f"{resource}: {key} must be {expected!r}.")
    band_d = payload.get("band_d")
    if not isinstance(band_d, Mapping) or not band_d:
        raise UKCouncilTaxBandDError(f"{resource}: band_d must be a non-empty mapping.")
    amounts: dict[str, Fraction] = {}
    for code, text in band_d.items():
        _nation_code(code, _ENGLAND_WALES_LETTERS, resource=resource)
        where = f"{resource}: the Band D of {code!r}"
        amounts[code] = band_d_decimal(text, where=where)
    return amounts


def _scotland_rows(
    payload: Mapping[str, Any] | None, *, verify_feed_identity: bool
) -> tuple[list[Mapping[str, Any]], str, Mapping[str, Any]]:
    """The Scottish rows, the resource's sha256 and the feed it was taken from.

    The packaged resource is read through ``vendored_rows``, which checks its
    feed against the committed pin before it reads a row; a supplied payload
    (tests) is checked against the pin first (unless ``verify_feed_identity``
    is false), then for the vendored header.
    """

    resource = SCOTLAND_BAND_D_RESOURCE
    if payload is not None:
        if verify_feed_identity:
            verify_vendored_resource_feed_identity(payload)
        rows = payload.get("rows")
        if (
            payload.get("kind") != VENDORED_RESOURCE_KIND
            or payload.get("resource") != resource
            or not isinstance(rows, list)
            or payload.get("row_count") != len(rows)
        ):
            raise UKCouncilTaxBandDError(
                f"The supplied payload is not the vendored resource {resource}."
            )
        feed = payload.get("source_fact_feed") or {}
        return rows, "supplied", MappingProxyType(dict(feed))
    if verify_feed_identity:
        rows = vendored_rows(resource)
    else:
        rows = list(load_vendored_resource(resource)["rows"])
    data = files("microcosm.build.uk").joinpath(resource).read_bytes()
    feed = json.loads(data)["source_fact_feed"]
    return rows, hashlib.sha256(data).hexdigest(), MappingProxyType(dict(feed))


def _scotland_band_d(rows: list[Mapping[str, Any]]) -> dict[str, Fraction]:
    """Each council area's Band D, its row held to the declared record set."""

    resource = SCOTLAND_BAND_D_RESOURCE
    band_d: dict[str, Fraction] = {}
    for row in rows:
        record = str(row.get("source_record_id"))
        layout = row.get("layout") or {}
        period = row.get("period") or {}
        geography = row.get("geography") or {}
        if layout.get("record_set_id") not in SCOTLAND_BAND_D_RECORD_SETS:
            raise UKCouncilTaxBandDError(
                f"{resource}: {record!r} is not of the record sets "
                f"{list(SCOTLAND_BAND_D_RECORD_SETS)}."
            )
        if (
            row.get("concept") != SCOTLAND_BAND_D_CONCEPT
            or row.get("unit") != _SCOTLAND_UNIT
        ):
            raise UKCouncilTaxBandDError(
                f"{resource}: {record!r} is not {SCOTLAND_BAND_D_CONCEPT!r} in "
                f"{_SCOTLAND_UNIT!r}."
            )
        if period.get("type") != "fiscal_year" or str(period.get("value")) != str(
            UK_BAND_D_BUILD_PERIOD
        ):
            raise UKCouncilTaxBandDError(
                f"{resource}: {record!r} is not of fiscal year {UK_BAND_D_FISCAL_YEAR}."
            )
        if (
            geography.get("level") != _SCOTLAND_LEVEL
            or geography.get("vintage") != SCOTLAND_BAND_D_GEOGRAPHY_VINTAGE
        ):
            raise UKCouncilTaxBandDError(
                f"{resource}: {record!r} is not a {_SCOTLAND_LEVEL} row of vintage "
                f"{SCOTLAND_BAND_D_GEOGRAPHY_VINTAGE!r}."
            )
        code = _nation_code(geography.get("id"), _SCOTLAND_LETTERS, resource=resource)
        if code in band_d:
            raise UKCouncilTaxBandDError(f"{resource} repeats {code!r}.")
        where = f"{resource}: the Band D of {code!r}"
        text, _ = band_d_pence(row.get("value"), where=where)
        band_d[code] = band_d_decimal(text, where=where)
    return band_d


@dataclass(frozen=True)
class UKCouncilTaxBandD:
    """2024-25 Band D council tax by Great Britain local authority code.

    ``band_d`` maps every England, Wales and Scotland code of the April 2023
    roster to its Band D in pounds, as an exact rational; Northern Ireland has
    none. The digests are the two packaged resources' sha256 (``"supplied"``
    for a payload a test supplied).
    """

    band_d: Mapping[str, Fraction]
    england_wales_sha256: str
    scotland_sha256: str
    scotland_source_fact_feed: Mapping[str, Any]

    def evidence(self) -> dict[str, object]:
        return {
            "fiscal_year": UK_BAND_D_FISCAL_YEAR,
            "resources": {
                UK_BAND_D_ENGLAND_WALES_RESOURCE: self.england_wales_sha256,
                SCOTLAND_BAND_D_RESOURCE: self.scotland_sha256,
            },
            "source_commit": self.scotland_source_fact_feed.get("source_commit"),
        }


def load_uk_council_tax_band_d(
    *,
    england_wales_payload: Mapping[str, Any] | None = None,
    scotland_payload: Mapping[str, Any] | None = None,
    verify_feed_identity: bool = True,
) -> UKCouncilTaxBandD:
    """Merge both Band D resources over the roster's Great Britain codes.

    ``england_wales_payload`` and ``scotland_payload`` replace the packaged
    resources (tests); the Scottish one is checked against the committed feed
    pin first (unless ``verify_feed_identity`` is false). The merged codes must
    be exactly the roster's England, Wales and Scotland codes.
    """

    england_wales_data, england_wales_sha256 = _england_wales_payload(
        england_wales_payload
    )
    rows, scotland_sha256, feed = _scotland_rows(
        scotland_payload, verify_feed_identity=verify_feed_identity
    )
    england_wales = _england_wales_band_d(england_wales_data)
    scotland = _scotland_band_d(rows)
    band_d = {**england_wales, **scotland}
    roster = load_uk_local_authority_names_resource()["areas"]
    great_britain = {
        code for code in roster if not code.startswith(_NORTHERN_IRELAND_LETTER)
    }
    off_roster = sorted(set(band_d) - great_britain)
    if off_roster:
        raise UKCouncilTaxBandDError(
            f"The Band D codes {off_roster} are not on the local authority roster "
            f"{UK_LOCAL_AUTHORITY_VINTAGE!r}."
        )
    missing = sorted(great_britain - set(band_d))
    if missing:
        raise UKCouncilTaxBandDError(
            f"The Band D resources lack the roster code(s) {missing}."
        )
    return UKCouncilTaxBandD(
        band_d=MappingProxyType(dict(sorted(band_d.items()))),
        england_wales_sha256=england_wales_sha256,
        scotland_sha256=scotland_sha256,
        scotland_source_fact_feed=feed,
    )


def check_scotland_band_d_declaration(
    parameters: Mapping[str, Any], *, stage: str, period: int
) -> None:
    """The runtime arm of a stage's ``verify_vendored_fact_resource`` operation.

    The declaration names the Scottish resource, its artifact role and record
    set, and maps the facts to the frame's build period.
    """

    expected = {
        "artifact_role": SCOTLAND_BAND_D_ARTIFACT_ROLE,
        "resource": SCOTLAND_BAND_D_RESOURCE,
        "record_sets": list(SCOTLAND_BAND_D_RECORD_SETS),
        "mapped_build_period": UK_BAND_D_BUILD_PERIOD,
    }
    declared = {key: parameters.get(key) for key in expected}
    if declared != expected:
        raise UKCouncilTaxBandDError(
            f"{stage}'s verify_vendored_fact_resource must declare {expected}, "
            f"not {declared}."
        )
    if int(period) != UK_BAND_D_BUILD_PERIOD:
        raise UKCouncilTaxBandDError(
            f"{stage} maps Scotland's Band D to build period "
            f"{UK_BAND_D_BUILD_PERIOD}, but the frame's period is {period}."
        )
