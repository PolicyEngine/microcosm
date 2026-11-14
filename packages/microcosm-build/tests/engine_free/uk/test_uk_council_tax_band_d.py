"""2024-25 Band D for Great Britain and the statutory band formula (microcosm#931).

The committed resources load and cover exactly the roster's Great Britain
codes; the loader refuses a payload outside that closed world (refusals are
built by mutating the committed payloads with invented amounts); the runtime
arm refuses a drifted declaration; and the statute's proportions are restated
here, so a ratio typo or Scotland's pre-2017 proportions fail.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from importlib.resources import files
from types import SimpleNamespace
from typing import Any

import pytest

from microcosm.build.uk_runtime import council_tax_band_d
from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed
from microcosm.build.uk_runtime.council_tax_band_d import (
    SCOTLAND_BAND_D_RECORD_SETS,
    SCOTLAND_BAND_D_RESOURCE,
    UK_BAND_D_ENGLAND_WALES_RESOURCE,
    UK_COUNCIL_TAX_BAND_RATIOS,
    UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT,
    UKCouncilTaxBandDError,
    check_scotland_band_d_declaration,
    load_uk_council_tax_band_d,
    uk_statutory_council_tax,
)
from microcosm.build.uk_runtime.council_tax_constants import (
    UK_COUNCIL_TAX_BANDS_BY_NATION,
    UK_NATION_BY_REGION,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import (
    feed_identity,
    load_vendored_resource,
)
from microcosm.build.uk_runtime.local_authority_input import (
    load_uk_local_authority_names_resource,
)
from tools import build_uk_council_tax_band_d as tool

#: An invented amount for the mutations below; a refusal must not quote it.
INVENTED = "4321.07"


def _packaged(name: str) -> dict[str, Any]:
    return json.loads(
        files("microcosm.build.uk").joinpath(name).read_text(encoding="utf-8")
    )


def _sha256(name: str) -> str:
    return hashlib.sha256(
        files("microcosm.build.uk").joinpath(name).read_bytes()
    ).hexdigest()


def _great_britain_codes() -> set[str]:
    return {
        code
        for code in load_uk_local_authority_names_resource()["areas"]
        if code[0] in "EWS"
    }


def _assert_quotes_no_amount(message: str) -> None:
    assert "4321" not in message
    assert not re.search(r"\d\.\d", message), message


# ----------------------------------------------------------------------------
# The committed resources


def test_the_resources_load_over_exactly_the_great_britain_codes() -> None:
    loaded = load_uk_council_tax_band_d()

    assert set(loaded.band_d) == _great_britain_codes()
    assert len(loaded.band_d) == 350
    assert {
        nation: sum(code.startswith(nation) for code in loaded.band_d)
        for nation in "EWSN"
    } == {"E": 296, "W": 22, "S": 32, "N": 0}
    assert all(
        isinstance(value, Fraction) and value > 0 for value in loaded.band_d.values()
    )
    assert loaded.evidence() == {
        "fiscal_year": "2024-25",
        "resources": {
            UK_BAND_D_ENGLAND_WALES_RESOURCE: _sha256(UK_BAND_D_ENGLAND_WALES_RESOURCE),
            SCOTLAND_BAND_D_RESOURCE: _sha256(SCOTLAND_BAND_D_RESOURCE),
        },
        "source_commit": load_uk_chronicle_feed().source_commit,
    }


def test_england_and_scotland_are_at_pence_and_wales_unrounded() -> None:
    """England's and Scotland's amounts are at pence, and Wales's as published.

    A kept float artefact, an amount off pence, a Welsh amount past six
    decimals or Welsh amounts rounded to pence fail here.
    """

    payload = _packaged(UK_BAND_D_ENGLAND_WALES_RESOURCE)
    loaded = load_uk_council_tax_band_d().band_d
    for code, text in payload["band_d"].items():
        if code.startswith("E"):
            assert re.fullmatch(r"[0-9]+\.[0-9]{2}", text), code
        else:
            assert code.startswith("W"), code
            assert re.fullmatch(r"[0-9]+(?:\.[0-9]{1,6})?", text), code
        assert loaded[code] == Fraction(Decimal(text)), code
    welsh = [text for code, text in payload["band_d"].items() if code.startswith("W")]
    assert len(welsh) == 22
    # Kept unrounded, some Welsh amount carries a third decimal; rounded to
    # pence, none would.
    assert any(len(text.partition(".")[2]) > 2 for text in welsh)
    scotland = [code for code in loaded if code.startswith("S")]
    assert len(scotland) == 32
    assert all((loaded[code] * 100).denominator == 1 for code in scotland)


def test_the_england_wales_resource_records_the_tool_pins() -> None:
    payload = _packaged(UK_BAND_D_ENGLAND_WALES_RESOURCE)

    assert payload["sources"] == tool.SOURCES
    assert payload["licence"] == tool.LICENCE
    assert {
        key: payload[key]
        for key in (
            "schema_version",
            "kind",
            "resource",
            "method",
            "fiscal_year",
            "local_authority_vintage",
        )
    } == {
        "schema_version": 1,
        "kind": "uk_council_tax_band_d_england_wales",
        "resource": "council_tax_band_d_england_wales_2024_25.json",
        "method": "uk-council-tax-band-d-v1",
        "fiscal_year": "2024-25",
        "local_authority_vintage": "2023_april_lad",
    }
    assert payload["receipts"]["authorities"] == {"england": 296, "wales": 22}
    assert payload["receipts"]["wales_excluded_names"] == ["Total Unitary Authorities"]


def test_the_scottish_resource_is_vendored_from_the_feed_pin() -> None:
    payload = load_vendored_resource(SCOTLAND_BAND_D_RESOURCE)

    assert payload["source_fact_feed"] == feed_identity(load_uk_chronicle_feed())
    assert payload["consumers"] == ["uk_runtime.council_tax_band_d"]
    assert payload["licence"]["licence"] == "Open Government Licence v3.0"
    assert [selection["row_count"] for selection in payload["selections"]] == [32]
    assert {row["layout"]["record_set_id"] for row in payload["rows"]} == {
        "scotgov.ctas2026.band_d.fy2024"
    }
    assert SCOTLAND_BAND_D_RECORD_SETS == ("scotgov.ctas2026.band_d.fy2024",)


# ----------------------------------------------------------------------------
# Loader refusals: England and Wales


def _set_band_d(code: str, value: object) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        payload["band_d"][code] = value

    return mutate


def _set_field(key: str, value: object) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        payload[key] = value

    return mutate


def _drop_band_d(code: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        del payload["band_d"][code]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set_band_d("N09000001", INVENTED), "Northern Ireland code 'N09000001'"),
        (_set_band_d("S12000005", INVENTED), "a nation it does not cover"),
        (_set_band_d("E06999999", INVENTED), "not on the local authority roster"),
        (_drop_band_d("W06000015"), r"lack the roster code\(s\) \['W06000015'\]"),
        (_set_band_d("E06000001", "4,321.07"), "not a positive decimal amount"),
        (_set_band_d("E06000001", "4.32107e3"), "not a positive decimal amount"),
        (_set_band_d("E06000001", "-4321.07"), "not a positive decimal amount"),
        (_set_band_d("E06000001", "0.00"), "not a positive decimal amount"),
        (_set_band_d("E06000001", ""), "not a positive decimal amount"),
        (_set_band_d("E06000001", 4321.07), "not a positive decimal amount"),
        (_set_field("kind", "uk_council_tax_band_d"), "kind must be"),
        (_set_field("fiscal_year", "2025-26"), "fiscal_year must be"),
        (_set_field("local_authority_vintage", "2024_lad"), "vintage must be"),
        (_set_field("schema_version", 2), "schema_version must be"),
        (_set_field("schema_version", True), "schema_version must be"),
        (_set_field("resource", "band_d.json"), "resource must be"),
        (_set_field("band_d", {}), "band_d must be a non-empty mapping"),
    ],
    ids=[
        "northern-ireland",
        "scottish-code",
        "off-roster",
        "missing",
        "thousands-separator",
        "exponent",
        "negative",
        "zero",
        "blank",
        "json-number",
        "kind",
        "fiscal-year",
        "vintage",
        "schema-version",
        "schema-version-bool",
        "resource",
        "empty",
    ],
)
def test_refuses_an_england_wales_payload_outside_the_closed_world(
    mutate, message
) -> None:
    payload = copy.deepcopy(_packaged(UK_BAND_D_ENGLAND_WALES_RESOURCE))
    mutate(payload)

    with pytest.raises(UKCouncilTaxBandDError, match=message) as refusal:
        load_uk_council_tax_band_d(england_wales_payload=payload)
    _assert_quotes_no_amount(str(refusal.value))


def test_a_code_written_twice_in_the_packaged_resource_is_refused(
    monkeypatch,
) -> None:
    """JSON keeps the last of two equal keys; the loader refuses the second."""

    package = files("microcosm.build.uk")
    text = package.joinpath(UK_BAND_D_ENGLAND_WALES_RESOURCE).read_text(
        encoding="utf-8"
    )
    doubled = text.replace(
        '"band_d": {', f'"band_d": {{\n  "E06000001": "{INVENTED}",', 1
    )
    assert doubled != text

    class Package:
        def joinpath(self, name: str):
            if name == UK_BAND_D_ENGLAND_WALES_RESOURCE:
                return SimpleNamespace(read_bytes=doubled.encode)
            return package.joinpath(name)

    monkeypatch.setattr(council_tax_band_d, "files", lambda _: Package())
    with pytest.raises(
        UKCouncilTaxBandDError, match="repeats the key 'E06000001'"
    ) as refusal:
        load_uk_council_tax_band_d()
    _assert_quotes_no_amount(str(refusal.value))


# ----------------------------------------------------------------------------
# Loader refusals: Scotland


def _set_row(path: tuple[str, ...], value: object) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        target = payload["rows"][0]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _repeat_row(payload: dict[str, Any]) -> None:
    payload["rows"].append(copy.deepcopy(payload["rows"][0]))
    payload["row_count"] += 1


def _drop_row(payload: dict[str, Any]) -> None:
    payload["rows"].pop(0)
    payload["row_count"] -= 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            _set_row(("layout", "record_set_id"), "scotgov.ctas2026.band_d.fy2023"),
            "is not of the record sets",
        ),
        (_set_row(("period", "value"), 2023), "is not of fiscal year 2024-25"),
        (_set_row(("period", "type"), "calendar_year"), "is not of fiscal year"),
        (_set_row(("concept",), "scotgov.council_tax.band_d"), "average_band_d"),
        (_set_row(("unit",), "gbp_thousands"), "average_band_d"),
        (_set_row(("geography", "level"), "country"), "local_authority row"),
        (_set_row(("geography", "vintage"), "ca_2011"), "local_authority row"),
        (_set_row(("geography", "id"), "N09000001"), "Northern Ireland code"),
        (_set_row(("geography", "id"), "E06000001"), "a nation it does not cover"),
        (_set_row(("geography", "id"), "S12999999"), "not on the local authority"),
        (_repeat_row, "repeats"),
        (_drop_row, "lack the roster code"),
        (_set_row(("value",), 4321.075), "within 1e-9 of a whole number of pence"),
        (_set_row(("value",), 4321.0700000025), "within 1e-9 of a whole number"),
        (_set_row(("value",), float("nan")), "not a positive finite amount"),
        (_set_row(("value",), float("inf")), "not a positive finite amount"),
        (_set_row(("value",), -4321.07), "not a positive finite amount"),
        (_set_row(("value",), 0), "not a positive finite amount"),
        (_set_row(("value",), True), "is not a number"),
        (_set_row(("value",), INVENTED), "is not a number"),
    ],
    ids=[
        "record-set",
        "period",
        "period-type",
        "concept",
        "unit",
        "level",
        "vintage",
        "northern-ireland",
        "english-code",
        "off-roster",
        "duplicate",
        "missing",
        "half-penny",
        "off-pence",
        "nan",
        "inf",
        "negative",
        "zero",
        "bool",
        "text",
    ],
)
def test_refuses_a_scottish_row_outside_the_declared_record_set(
    mutate, message
) -> None:
    payload = copy.deepcopy(_packaged(SCOTLAND_BAND_D_RESOURCE))
    mutate(payload)

    with pytest.raises(UKCouncilTaxBandDError, match=message) as refusal:
        load_uk_council_tax_band_d(scotland_payload=payload)
    _assert_quotes_no_amount(str(refusal.value))


def test_scottish_amounts_are_kept_as_the_pence_they_lie_within_the_guard_of() -> None:
    payload = copy.deepcopy(_packaged(SCOTLAND_BAND_D_RESOURCE))
    code = payload["rows"][0]["geography"]["id"]
    for stored in (4321.07, 4321.07 + 5e-10, 4321.07 - 5e-10, 4321):
        payload["rows"][0]["value"] = stored
        loaded = load_uk_council_tax_band_d(scotland_payload=payload)
        assert loaded.band_d[code] == round(Fraction(stored) * 100) / Fraction(100)
    assert loaded.scotland_sha256 == "supplied"
    assert loaded.england_wales_sha256 == _sha256(UK_BAND_D_ENGLAND_WALES_RESOURCE)


def test_a_scottish_payload_from_another_feed_is_refused_before_its_rows() -> None:
    payload = copy.deepcopy(_packaged(SCOTLAND_BAND_D_RESOURCE))
    payload["source_fact_feed"]["source_commit"] = "0" * 40
    payload["rows"][0]["value"] = "not read"

    with pytest.raises(ValueError, match="differs from the committed UK pin"):
        load_uk_council_tax_band_d(scotland_payload=payload)

    payload["rows"][0]["value"] = 4321.07
    loaded = load_uk_council_tax_band_d(
        scotland_payload=payload, verify_feed_identity=False
    )
    assert loaded.evidence()["source_commit"] == "0" * 40


@pytest.mark.parametrize(
    ("key", "value"),
    [("kind", "uk_ledger_facts"), ("resource", "band_d.json"), ("row_count", 31)],
)
def test_refuses_a_scottish_payload_that_is_not_the_vendored_resource(
    key, value
) -> None:
    payload = copy.deepcopy(_packaged(SCOTLAND_BAND_D_RESOURCE))
    payload[key] = value

    with pytest.raises(UKCouncilTaxBandDError, match="is not the vendored resource"):
        load_uk_council_tax_band_d(scotland_payload=payload)


# ----------------------------------------------------------------------------
# The runtime arm

DECLARATION = {
    "kind": "verify_vendored_fact_resource",
    "artifact_role": "scotland_band_d_facts",
    "resource": "scotgov_band_d_council_tax_2024_25.json",
    "feed_pin": "chronicle_feed.json",
    "record_sets": ["scotgov.ctas2026.band_d.fy2024"],
    "mapped_build_period": 2024,
}


def test_the_runtime_arm_accepts_the_declaration_at_2024() -> None:
    check_scotland_band_d_declaration(DECLARATION, stage="a_stage", period=2024)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("artifact_role", "band_d_facts"),
        ("resource", "scotgov_chargeable_dwellings_by_band.json"),
        ("record_sets", ["scotgov.ctas2026.band_d.fy2025"]),
        ("record_sets", []),
        ("mapped_build_period", 2025),
    ],
)
def test_the_runtime_arm_refuses_a_drifted_declaration(key, value) -> None:
    with pytest.raises(UKCouncilTaxBandDError, match="a_stage's verify_vendored"):
        check_scotland_band_d_declaration(
            {**DECLARATION, key: value}, stage="a_stage", period=2024
        )


def test_the_runtime_arm_refuses_another_build_period() -> None:
    with pytest.raises(UKCouncilTaxBandDError, match="frame's period is 2025"):
        check_scotland_band_d_declaration(DECLARATION, stage="a_stage", period=2025)


# ----------------------------------------------------------------------------
# The statute

NINTHS = {
    band: Fraction(numerator, 9)
    for band, numerator in zip(
        "ABCDEFGHI", (6, 7, 8, 9, 11, 13, 15, 18, 21), strict=True
    )
}
#: Each nation's proportions, restated: England A to H, Wales A to I, Scotland
#: A to H at the proportions in force from 2017-18.
PROPORTIONS = {
    "ENGLAND": {band: NINTHS[band] for band in "ABCDEFGH"},
    "WALES": dict(NINTHS),
    "SCOTLAND": {
        band: Fraction(numerator, 360)
        for band, numerator in zip(
            "ABCDEFGH", (240, 280, 320, 360, 473, 585, 705, 882), strict=True
        )
    },
}
#: An invented Band D.
BAND_D = Fraction(Decimal("1987.65"))


def test_the_statute_prices_every_band_of_every_nation() -> None:
    assert {
        nation: dict(proportions)
        for nation, proportions in UK_COUNCIL_TAX_BAND_RATIOS.items()
    } == PROPORTIONS
    for nation, proportions in PROPORTIONS.items():
        for band, proportion in proportions.items():
            bill = uk_statutory_council_tax(BAND_D, nation, band, single_adult=False)
            assert bill == BAND_D * proportion
            single = uk_statutory_council_tax(BAND_D, nation, band, single_adult=True)
            assert single == BAND_D * proportion * Fraction(3, 4)


def test_scotland_takes_the_2017_proportions_for_e_to_h() -> None:
    before_2017 = {band: NINTHS[band] for band in "EFGH"}
    for band, proportion in before_2017.items():
        assert UK_COUNCIL_TAX_BAND_RATIOS["SCOTLAND"][band] != proportion
    assert uk_statutory_council_tax(360, "SCOTLAND", "H", single_adult=False) == 882
    assert uk_statutory_council_tax(9, "WALES", "I", single_adult=False) == 21
    assert uk_statutory_council_tax(9, "ENGLAND", "H", single_adult=False) == 18


def test_the_proportions_follow_council_tax_constants() -> None:
    nations = set(UK_NATION_BY_REGION.values())
    assert set(UK_COUNCIL_TAX_BANDS_BY_NATION) == nations
    assert set(UK_COUNCIL_TAX_BAND_RATIOS) == nations - {"NORTHERN_IRELAND"}
    for nation, proportions in UK_COUNCIL_TAX_BAND_RATIOS.items():
        assert tuple(proportions) == UK_COUNCIL_TAX_BANDS_BY_NATION[nation]
        assert proportions["D"] == 1
    assert UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT == Fraction(1, 4)


def test_the_statute_takes_any_exact_band_d() -> None:
    expected = Fraction(198765, 100) * Fraction(11, 9)
    for band_d in (Fraction(198765, 100), Decimal("1987.65"), Decimal("1987.650")):
        assert uk_statutory_council_tax(band_d, "ENGLAND", "E", single_adult=False) == (
            expected
        )
    assert uk_statutory_council_tax(1987, "WALES", "D", single_adult=True) == Fraction(
        1987 * 3, 4
    )


@pytest.mark.parametrize(
    ("band_d", "nation", "band", "single_adult", "message"),
    [
        (BAND_D, "NORTHERN_IRELAND", "D", False, "levies domestic rates"),
        (BAND_D, "ENGLAND", "I", False, "ENGLAND has no council-tax band 'I'"),
        (BAND_D, "SCOTLAND", "I", False, "SCOTLAND has no council-tax band 'I'"),
        (BAND_D, "LONDON", "D", False, "'LONDON' is not a council-tax nation"),
        (1987.65, "ENGLAND", "D", False, "an exact rational .* not a float"),
        (True, "ENGLAND", "D", False, "an exact rational .* not a bool"),
        ("1987.65", "ENGLAND", "D", False, "an exact rational .* not a str"),
        (0, "ENGLAND", "D", False, "must be positive"),
        (Fraction(-1), "WALES", "D", False, "must be positive"),
        (Decimal("NaN"), "SCOTLAND", "D", False, "must be finite"),
        (BAND_D, "ENGLAND", "D", 1, "single_adult must be a bool"),
    ],
    ids=[
        "northern-ireland",
        "england-band-i",
        "scotland-band-i",
        "region",
        "float",
        "bool",
        "text",
        "zero",
        "negative",
        "nan",
        "single-adult-int",
    ],
)
def test_the_statute_refuses_what_it_does_not_cover(
    band_d, nation, band, single_adult, message
) -> None:
    with pytest.raises(UKCouncilTaxBandDError, match=message):
        uk_statutory_council_tax(band_d, nation, band, single_adult=single_adult)
