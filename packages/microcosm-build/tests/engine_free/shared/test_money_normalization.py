"""Exact donor money normalization: declaration, arithmetic, receipts, factors.

The property tests compare against an independent oracle built only from
``fractions.Fraction`` and the built-in ``round`` (ties to even); it never
calls a production helper.
"""

from __future__ import annotations

import copy
import math
from decimal import localcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.money_normalization import (
    MONEY_FACTOR_PURPOSE,
    check_money_factor,
    money_column_rosters,
    money_factor_inputs,
    normalize_money_tables,
    normalized_column_families,
    resolve_money_rate,
    selected_money_factors,
    validate_money_normalization,
)
from microcosm.build.reference_factors import FACTOR_PURPOSES
from test_support.microcosm_build.money_normalization import (
    DIMENSION,
    NORMALIZED,
    factor_id,
    money_factor_context,
    oracle_factor,
    rate_feed_form,
    toy_declaration,
    write_money_feed,
)

PROPERTY = settings(
    max_examples=200,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
_FIELDS = (
    "schema_version",
    "from_currency",
    "to_currency",
    "rate",
    "quote",
    "rounding",
    "donor_money_period",
    "columns",
    "index_family_factors",
    "factor_profile",
    "cents_suffix",
    "quantile_mapped",
    "pending",
)


def _oracle(value: float, rate: Fraction, quote: str, factor: Fraction, digits=2):
    """Two roundings from first principles: currency, then factor."""
    x = Fraction(repr(float(value)))
    scale = 10**digits
    converted = x * rate * scale if quote == "to_per_from" else x * scale / rate
    first = round(converted)
    return first, round(first * factor), round(converted * factor)


def _identity_factor():
    return {"identity": True, "reason": "test"}


def _factor(numerator: int, digits: int = 6, name: str = "f") -> dict:
    return {
        "factor_id": name,
        "numerator": numerator,
        "digits": digits,
        "declaration_sha256": "d" * 64,
        "receipt_sha256": "e" * 64,
    }


def _one_column_declaration(column="rental_income", entity="person", **options):
    """The toy declaration with one normalized column and the rest pending."""
    declaration = toy_declaration(**options)
    validated = validate_money_normalization(copy.deepcopy(declaration))
    family = normalized_column_families(validated)[entity, column]
    others = {
        key: value
        for key, value in money_column_rosters(validated).items()
        if value[0] == "normalized" and key != (entity, column)
    }
    declaration["columns"] = {entity: [column]}
    declaration["index_family_factors"] = {
        family: declaration["index_family_factors"][family]
    }
    pending = {}
    for other_entity, other_column in others:
        pending.setdefault(other_entity, {})[other_column] = {
            "awaits": ["test"],
            "reason": "test",
        }
    declaration["pending"] = pending
    return validate_money_normalization(declaration), family


def _normalize(values, *, rate="1.25", quote="to_per_from", factor=None, column=None):
    column = column or "rental_income"
    declaration, family = _one_column_declaration(column, rate=rate, quote=quote)
    tables = {
        "person": pd.DataFrame(
            {"person_id": np.arange(len(values)), column: np.asarray(values, float)}
        )
    }
    arrays, receipts = normalize_money_tables(
        tables,
        declaration,
        rate=Fraction(rate),
        rate_text=rate,
        factors={family: factor or _identity_factor()},
    )
    return arrays, receipts, declaration, tables


# ---------------------------------------------------------------------------
# Declaration validation
# ---------------------------------------------------------------------------


def test_toy_declaration_validates_and_rosters_partition_every_money_concept():
    declaration = validate_money_normalization(toy_declaration())
    rosters = money_column_rosters(declaration)
    assert {key for key, (route, _) in rosters.items() if route == "normalized"} == {
        (entity, column) for entity, names in NORMALIZED.items() for column in names
    }
    assert {key for key, (route, _) in rosters.items() if route != "normalized"} == {
        ("person", "employment_income"),
        ("person", "liquid_financial_assets"),
        ("household", "rent"),
    }
    assert ("person", "public_pension_income") not in rosters
    assert set(normalized_column_families(declaration).values()) == {
        "mixed_income",
        "capital_income",
        "pensions",
        "owner_housing_costs",
    }


@pytest.mark.parametrize("field", _FIELDS)
def test_every_field_is_required(field):
    declaration = toy_declaration()
    declaration.pop(field)
    with pytest.raises(ValueError, match=f"missing fields \\['{field}'\\]"):
        validate_money_normalization(declaration)


def test_unknown_fields_refuse_and_descriptive_text_rides_along():
    declaration = toy_declaration()
    declaration["fallback_rate"] = "1"
    with pytest.raises(ValueError, match="unknown fields"):
        validate_money_normalization(declaration)
    declaration.pop("fallback_rate")
    declaration["notes"] = "Reviewed prose."
    assert validate_money_normalization(declaration)["notes"] == "Reviewed prose."


@pytest.mark.parametrize("value", [1.25, 1, None, True])
def test_a_rate_given_as_a_json_number_refuses(value):
    declaration = toy_declaration()
    declaration["rate"]["text"] = value
    with pytest.raises(ValueError, match="decimal text, never a JSON number"):
        validate_money_normalization(declaration)


@pytest.mark.parametrize("text", ["1.2e3", "-1.25", "0", "0.000", " 1.25", "1,25", ""])
def test_rate_text_must_be_plain_positive_decimal(text):
    declaration = toy_declaration()
    declaration["rate"]["text"] = text
    with pytest.raises(ValueError, match="rate text"):
        validate_money_normalization(declaration)


@pytest.mark.parametrize(
    "edit",
    [
        lambda rate: rate.pop("citation"),
        lambda rate: rate["citation"].pop("url"),
        lambda rate: rate.update(citation="IRS"),
        lambda rate: rate.update(form="literal"),
        lambda rate: rate.update(extra="x"),
    ],
)
def test_decimal_rate_needs_its_citation_and_closed_form(edit):
    declaration = toy_declaration()
    edit(declaration["rate"])
    with pytest.raises(ValueError, match="rate"):
        validate_money_normalization(declaration)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("quote",), "multiply", "quote"),
        (("quote",), None, "quote"),
        (("rounding", "mode"), "half_up", "half_even"),
        (("rounding", "digits"), 7, "digits"),
        (("rounding", "digits"), 2.0, "digits"),
        (("donor_money_period", "type"), "fiscal_year", "period type"),
        (("donor_money_period", "value"), "2024", "year"),
        (("from_currency",), "usd", "ISO 4217"),
        (("to_currency",), "USD", "distinct currencies"),
        (("cents_suffix",), "cents", "cents_suffix"),
        (("schema_version",), 2, "schema_version"),
    ],
)
def test_closed_values_refuse(path, value, message):
    declaration = toy_declaration()
    target = declaration
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_money_normalization(declaration)


def test_rounding_and_period_objects_are_closed():
    declaration = toy_declaration()
    declaration["rounding"]["places"] = 2
    with pytest.raises(ValueError, match="rounding has unknown"):
        validate_money_normalization(declaration)
    declaration = toy_declaration()
    declaration["donor_money_period"].pop("value")
    with pytest.raises(ValueError, match="donor_money_period is missing"):
        validate_money_normalization(declaration)


@pytest.mark.parametrize("field", ["type", "value", "basis", "evidence"])
def test_every_donor_period_field_is_required(field):
    declaration = toy_declaration()
    declaration["donor_money_period"].pop(field)
    with pytest.raises(ValueError, match=f"missing fields \\['{field}'\\]"):
        validate_money_normalization(declaration)


@pytest.mark.parametrize("basis", ["declared_convention", "authenticated"])
def test_donor_period_basis_is_explicit_with_evidence(basis):
    declaration = toy_declaration()
    declaration["donor_money_period"]["basis"] = basis
    validated = validate_money_normalization(declaration)
    assert validated["donor_money_period"] == declaration["donor_money_period"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("basis", "model_period", "period basis"),
        ("basis", None, "period basis"),
        ("evidence", [], "evidence"),
        ("evidence", "not a citation list", "evidence"),
        ("evidence", [""], "evidence"),
        ("evidence", [None], "evidence"),
    ],
)
def test_donor_period_refuses_an_unknown_basis_or_missing_evidence(
    field, value, message
):
    declaration = toy_declaration()
    declaration["donor_money_period"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_money_normalization(declaration)


def test_each_column_appears_in_exactly_one_roster():
    declaration = toy_declaration()
    declaration["columns"]["person"].append("employment_income")
    with pytest.raises(ValueError, match="both normalized and quantile_mapped"):
        validate_money_normalization(declaration)
    declaration = toy_declaration()
    declaration["pending"] = {
        "household": {"rent": {"awaits": ["x"], "reason": "x"}},
    }
    with pytest.raises(ValueError, match="both quantile_mapped and pending"):
        validate_money_normalization(declaration)


def test_every_transported_money_concept_must_be_declared():
    declaration = toy_declaration()
    declaration["columns"]["household"].remove("property_tax")
    with pytest.raises(ValueError, match=r"unclassified: \['household.property_tax'\]"):
        validate_money_normalization(declaration)


def test_rosters_name_money_concepts_and_their_awaited_references():
    declaration = toy_declaration()
    declaration["columns"]["person"].append("age")
    with pytest.raises(ValueError, match="not a money concept"):
        validate_money_normalization(declaration)
    for awaits in ([], [""], "toy bands", ["x", "x"]):
        declaration = toy_declaration()
        declaration["quantile_mapped"]["household"]["rent"]["awaits"] = awaits
        with pytest.raises(ValueError, match="awaits"):
            validate_money_normalization(declaration)


def test_a_quantile_mapped_drop_default_column_is_admitted():
    # Belgium condition d: the declaration does not consult transport rules.
    declaration = toy_declaration()
    declaration["quantile_mapped"]["person"]["public_pension_income"] = {
        "awaits": ["toy pension bands"],
        "reason": "A country may quantile-map a column whose default is drop.",
    }
    rosters = money_column_rosters(validate_money_normalization(declaration))
    assert rosters["person", "public_pension_income"][0] == "quantile_mapped"


def test_an_unmapped_family_refuses_rather_than_defaulting_to_one():
    declaration = toy_declaration()
    declaration["index_family_factors"].pop("pensions")
    with pytest.raises(ValueError, match=r"\['pensions'\] have no declared factor"):
        validate_money_normalization(declaration)


def test_unused_and_unknown_families_refuse():
    declaration = toy_declaration()
    declaration["index_family_factors"]["rents"] = {
        "identity": True,
        "reason": "unused",
    }
    with pytest.raises(ValueError, match="that no normalized column uses"):
        validate_money_normalization(declaration)
    declaration = toy_declaration()
    declaration["index_family_factors"]["wages"] = {"identity": True, "reason": "x"}
    with pytest.raises(ValueError, match="unknown index families"):
        validate_money_normalization(declaration)


@pytest.mark.parametrize(
    "entry",
    [
        {"identity": True},
        {"identity": True, "reason": ""},
        {"identity": "yes", "reason": "x"},
        {"identity": False, "reason": "x"},
        {"identity": True, "reason": "x", "factor_id": "f"},
    ],
)
def test_an_identity_family_is_explicit_with_a_reason(entry):
    declaration = toy_declaration()
    declaration["index_family_factors"]["pensions"] = entry
    with pytest.raises(ValueError, match="identity"):
        validate_money_normalization(declaration)


def test_explicit_identity_is_admitted_and_selected_as_none():
    declaration = validate_money_normalization(
        toy_declaration(identity_families=("pensions",))
    )
    _, factors = selected_money_factors(declaration, {})
    assert factors["pensions"] is None
    assert factors["capital_income"] == factor_id("central")


def test_profiled_families_name_one_factor_per_profile_and_a_disclosure():
    declaration = toy_declaration()
    declaration["index_family_factors"]["pensions"]["factors"].pop("high")
    with pytest.raises(ValueError, match="every declared profile"):
        validate_money_normalization(declaration)
    declaration = toy_declaration()
    declaration["index_family_factors"]["pensions"].pop("disclosure")
    with pytest.raises(ValueError, match="missing fields \\['disclosure'\\]"):
        validate_money_normalization(declaration)


def test_unprofiled_declarations_use_one_factor_id_per_family():
    declaration = toy_declaration()
    declaration["factor_profile"] = None
    for entry in declaration["index_family_factors"].values():
        entry.pop("factors")
        entry["factor_id"] = factor_id("central")
    validated = validate_money_normalization(declaration)
    assert selected_money_factors(validated, {DIMENSION: "low"}) == (
        None,
        dict.fromkeys(validated["index_family_factors"], factor_id("central")),
    )
    declaration["factor_profile"] = {
        "dimension": DIMENSION,
        "profiles": ["central"],
        "central_profile": "central",
    }
    with pytest.raises(ValueError, match="unknown fields \\['factor_id'\\]"):
        validate_money_normalization(declaration)


def test_profile_selection_follows_the_knob_and_refuses_unknown_profiles():
    declaration = validate_money_normalization(toy_declaration())
    for selection, expected in (
        ({}, "central"),
        ({DIMENSION: "low"}, "low"),
        ({DIMENSION: "high", "other_knob": "x"}, "high"),
    ):
        profile, factors = selected_money_factors(declaration, selection)
        assert profile == expected
        assert set(factors.values()) == {factor_id(expected)}
    with pytest.raises(ValueError, match="Unknown money normalization profile"):
        selected_money_factors(declaration, {DIMENSION: "bogus"})


def test_feed_rate_form_is_closed_and_needs_its_expected_decimal():
    identity = {
        "source_commit": "c" * 40,
        "facts_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "artifact_schema_version": "policyengine_ledger.consumer_artifact.v2",
        "consumer_fact_schema_versions": ["chronicle.consumer_fact.v4"],
        "consumer_fact_schema_sha256": "d" * 64,
        "fact_row_count": 5,
    }
    declaration = toy_declaration()
    declaration["rate"] = rate_feed_form(identity, "1.0824")
    validated = validate_money_normalization(declaration)
    assert validated["rate"]["expected_decimal"] == "1.0824"
    for edit in (
        lambda rate: rate.pop("expected_decimal"),
        lambda rate: rate.update(expected_decimal=1.0824),
        lambda rate: rate.update(assertion_policy="anything"),
        lambda rate: rate["selector"].update(not_a_selector_field=1),
        lambda rate: rate.update(citation={"publisher": "x"}),
    ):
        broken = copy.deepcopy(declaration)
        edit(broken["rate"])
        with pytest.raises(ValueError):
            validate_money_normalization(broken)


# ---------------------------------------------------------------------------
# Exact arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "rate", "quote", "factor", "expected"),
    [
        # 3 * 1.25 = 3.75 -> 375 cents; 375 * 1.1 = 412.5 ties to even 412.
        (3.0, "1.25", "to_per_from", 1100000, 412),
        # 5 * 1.25 * 1.1 = 6.875 -> 687.5 ties to even 688.
        (5.0, "1.25", "to_per_from", 1100000, 688),
        # Negative signed amount: -2.675 * 125 = -334.375 -> -334; * 1.1 = -367.4.
        (-2.675, "1.25", "to_per_from", 1100000, -367),
        # Binary 2.675 is below 2.675; the decimal tie still rounds to even 268.
        (2.675, "1", "to_per_from", 1000000, 268),
        # A negative half-cent tie rounds to even: -0.125 -> -12.5 -> -12.
        (-0.125, "1", "to_per_from", 1000000, -12),
        (-0.135, "1", "to_per_from", 1000000, -14),
        # Float multiplication gives 57.49999999999999 rather than the exact tie.
        (0.575, "1", "to_per_from", 1000000, 58),
        # from_per_to divides by the exact rate: 3 / 0.6 = 5.00 -> 500 cents.
        (3.0, "0.6", "from_per_to", 1000000, 500),
        # A large amount stays exact: 1e15 * 1.654 = 1.654e15 exactly.
        (1e15, "1.654", "to_per_from", 1000000, 165400000000000000),
        # And 1e9 / 0.6 = 1666666666.666... -> 166666666667 cents.
        (1e9, "0.6", "from_per_to", 1000000, 166666666667),
    ],
)
def test_hand_cases_round_half_even_at_each_step(value, rate, quote, factor, expected):
    arrays, _, declaration, _ = _normalize(
        [value],
        rate=rate,
        quote=quote,
        factor=None if factor == 1000000 else _factor(factor),
    )
    cents = arrays["person", "rental_income_cents"]
    assert cents.dtype == np.int64 and int(cents[0]) == expected
    assert arrays["person", "rental_income"][0] == expected / 100


def test_two_roundings_are_recorded_separately_from_one_rounding():
    # 0.004 * 100 = 0.4 -> 0 cents before ageing; one rounding of 0.4 * 1.5 = 0.6 -> 1.
    first, two, one = _oracle(0.004, Fraction(1), "to_per_from", Fraction(3, 2))
    assert (first, two, one) == (0, 0, 1)
    arrays, *_ = _normalize([0.004], rate="1", factor=_factor(1500000))
    assert int(arrays["person", "rental_income_cents"][0]) == 0


def test_four_cents_rounds_to_seven_fx_then_eight_indexed_cents():
    rate, factor = Fraction("1.654"), Fraction("1.079073")
    assert _oracle(0.04, rate, "to_per_from", factor) == (7, 8, 7)
    arrays, receipts, *_ = _normalize([0.04], rate="1.654", factor=_factor(1079073))
    assert int(arrays["person", "rental_income_cents"][0]) == 8
    assert receipts[0]["sum_after_fx"] == "0.07"
    assert receipts[0]["sum_after"] == "0.08"


def test_decimal_context_does_not_round_products_quotients_or_receipt_sums():
    values = [1e15, 0.04, -1e15, 0.125, -0.135]
    expected, expected_receipts, *_ = _normalize(
        values, rate="1.654", factor=_factor(1079073)
    )
    with localcontext() as context:
        context.prec = 2
        actual, receipts, *_ = _normalize(values, rate="1.654", factor=_factor(1079073))
    assert {key: values.tolist() for key, values in actual.items()} == {
        key: values.tolist() for key, values in expected.items()
    }
    assert receipts == expected_receipts
    assert receipts[0]["sum_before"] == "0.03"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nan_and_infinite_amounts_refuse(bad):
    with pytest.raises(ValueError, match="NaN and infinite"):
        _normalize([1.0, bad])


def test_an_unsigned_column_refuses_a_negative_amount_and_signed_keeps_sign():
    with pytest.raises(
        ValueError, match="Unsigned money column person.interest_income"
    ):
        _normalize([-1.0], column="interest_income")
    arrays, *_ = _normalize([-1.0, 1.0])
    assert arrays["person", "rental_income"].tolist() == [-1.25, 1.25]


@pytest.mark.parametrize("value", [1e17, -1e17])
def test_int64_minor_units_bound_refuses_overflow(value):
    with pytest.raises(ValueError, match="exceeds int64"):
        _normalize([value], rate="1")


def test_int64_bound_is_checked_at_both_rounding_steps():
    # The FX cents overflow even though the factor would shrink them into range.
    with pytest.raises(ValueError, match="exceeds int64"):
        _normalize([1e17], rate="1", factor=_factor(1, digits=1))
    # The FX cents fit, but the factor would overflow the companion column.
    with pytest.raises(ValueError, match="exceeds int64"):
        _normalize([1e16], rate="1", factor=_factor(10, digits=0))


def test_int64_lower_endpoint_is_a_valid_signed_cent_amount():
    arrays, *_ = _normalize([-1.0], rate="1", factor=_factor(2**63, digits=2))
    assert int(arrays["person", "rental_income_cents"][0]) == -(2**63)


@pytest.mark.parametrize(
    ("entity", "column"),
    [(entity, column) for entity, names in NORMALIZED.items() for column in names],
)
def test_every_declared_cents_column_checks_int64_overflow(entity, column):
    declaration, family = _one_column_declaration(column, entity)
    with pytest.raises(ValueError, match=f"{entity}.{column} exceeds int64"):
        normalize_money_tables(
            {entity: pd.DataFrame({column: [1e17]})},
            declaration,
            rate=Fraction(1),
            rate_text="1",
            factors={family: _identity_factor()},
        )


def test_normalizing_twice_refuses():
    arrays, _, declaration, tables = _normalize([1.0, 2.0])
    table = tables["person"].copy()
    for (_, column), values in arrays.items():
        table[column] = values
    with pytest.raises(ValueError, match="normalized exactly once"):
        normalize_money_tables(
            {"person": table},
            declaration,
            rate=Fraction(5, 4),
            rate_text="1.25",
            factors={"capital_income": _identity_factor()},
        )


def test_receipt_records_exact_sums_residuals_and_identities():
    _, receipts, *_ = _normalize(
        [3.0, 5.0, -2.675, 0.1], factor=_factor(1100000, name="toy")
    )
    (receipt,) = receipts
    assert receipt["n"] == 4
    assert receipt["sum_before"] == "5.425"
    # Currency cents: 375, 625, -334 (from -334.375), 12 (from 12.5).
    assert receipt["sum_after_fx"] == "6.78"
    # Aged cents: 412 (412.5), 688 (687.5), -367 (-367.4), 13 (13.2).
    assert receipt["sum_after"] == "7.46"
    assert (
        Fraction(**receipt["fx_rounding_residual_minor_units"])
        == Fraction(678) - Fraction("5.425") * 125
    )
    assert Fraction(**receipt["factor_rounding_residual_minor_units"]) == Fraction(
        746
    ) - Fraction(678) * Fraction(11, 10)
    assert receipt["factor"] == {
        "factor_id": "toy",
        "declaration_sha256": "d" * 64,
        "receipt_sha256": "e" * 64,
        "operational_factor": {"numerator": 1100000, "denominator": 1000000},
    }
    assert receipt["rate_text"] == "1.25"
    assert receipt["quote"] == "to_per_from"
    assert receipt["donor_money_period"] == toy_declaration()["donor_money_period"]
    assert receipt["cents_column"] == "rental_income_cents"
    assert receipt["index_family"] == "capital_income"


def test_identity_receipt_names_its_reason_and_has_no_factor_residual():
    _, receipts, *_ = _normalize([0.5])
    assert receipts[0]["factor"] == {"identity": True, "reason": "test"}
    assert receipts[0]["factor_rounding_residual_minor_units"] == {
        "numerator": 0,
        "denominator": 1,
    }


# ---------------------------------------------------------------------------
# Rates and factors
# ---------------------------------------------------------------------------


def test_decimal_rate_is_exact_and_never_passes_through_float():
    rate, evidence = resolve_money_rate(
        validate_money_normalization(toy_declaration(rate="1.654"))
    )
    assert rate == Fraction(827, 500) != Fraction(float("1.654"))
    assert evidence["text"] == "1.654"


def test_feed_rate_replays_against_the_authenticated_feed(tmp_path):
    _, pin, identity, facts = write_money_feed(tmp_path / "feed", rate="1.0824")
    declaration = toy_declaration()
    declaration["rate"] = rate_feed_form(identity, "1.0824")
    validated = validate_money_normalization(declaration)
    rate, evidence = resolve_money_rate(validated, facts, pin)
    assert rate == Fraction("1.0824")
    assert evidence["operand"]["decimal_operand"] == "1.0824"
    with pytest.raises(ValueError, match="authenticated feed facts"):
        resolve_money_rate(validated)
    declaration["rate"]["expected_decimal"] = "1.0825"
    with pytest.raises(ValueError, match="replay differs"):
        resolve_money_rate(validate_money_normalization(declaration), facts, pin)
    declaration["rate"] = rate_feed_form({**identity, "fact_row_count": 1}, "1.0824")
    with pytest.raises(ValueError, match="feed identity mismatch"):
        resolve_money_rate(validate_money_normalization(declaration), facts, pin)


def test_money_factors_carry_their_purpose_and_cover_the_donor_period(tmp_path):
    assert MONEY_FACTOR_PURPOSE in FACTOR_PURPOSES
    _, _, identity, facts = write_money_feed(tmp_path / "feed")
    declaration = validate_money_normalization(toy_declaration())
    context = money_factor_context(identity, facts)
    for entry, profile in zip(
        context["factors"], ("low", "central", "high"), strict=True
    ):
        numerator, digits = check_money_factor(entry, declaration)
        assert Fraction(numerator, 10**digits) == oracle_factor(profile)
        assert money_factor_inputs(entry, declaration)["factor_id"] == factor_id(
            profile
        )
    profile_context = money_factor_context(identity, facts, purpose="profile_input")
    with pytest.raises(ValueError, match="applies only money_normalization_input"):
        check_money_factor(profile_context["factors"][1], declaration)
    other = copy.deepcopy(declaration)
    other["donor_money_period"]["value"] = 2023
    with pytest.raises(ValueError, match="not the declared donor money period"):
        check_money_factor(context["factors"][1], other)


# ---------------------------------------------------------------------------
# Properties against the independent oracle
# ---------------------------------------------------------------------------

_AMOUNTS = st.one_of(
    st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-(10**9), max_value=10**9).map(lambda value: value / 1000),
    st.integers(min_value=-(10**6), max_value=10**6).map(lambda value: value / 8),
)
_RATES = st.decimals(min_value="0.001", max_value="99.999", places=3).map(str)
_FACTORS = st.integers(min_value=500000, max_value=2500000)


@PROPERTY
@example(values=[2.675, -0.125, 3.0], rate="1.000", factor=1100000, quote="to_per_from")
@example(values=[1e9], rate="0.600", factor=1000000, quote="from_per_to")
@given(
    values=st.lists(_AMOUNTS, min_size=1, max_size=12),
    rate=_RATES,
    factor=_FACTORS,
    quote=st.sampled_from(["to_per_from", "from_per_to"]),
)
def test_normalization_equals_the_fraction_oracle(values, rate, factor, quote):
    arrays, receipts, *_ = _normalize(
        values, rate=rate, quote=quote, factor=_factor(factor)
    )
    exact_rate = Fraction(rate)
    oracle = [
        _oracle(value, exact_rate, quote, Fraction(factor, 10**6))[1]
        for value in values
    ]
    cents = arrays["person", "rental_income_cents"].tolist()
    assert cents == oracle
    # The float column is exactly the cents over 100, and cents times one
    # hundredth reproduces the oracle's decimal amount.
    assert arrays["person", "rental_income"].tolist() == [
        float(Fraction(units, 100)) for units in oracle
    ]
    assert Fraction(receipts[0]["sum_after"]) == Fraction(sum(oracle), 100)
    assert Fraction(receipts[0]["sum_before"]) == sum(
        Fraction(repr(float(value))) for value in values
    )


@PROPERTY
@given(values=st.lists(_AMOUNTS, min_size=1, max_size=12), rate=_RATES)
def test_normalizing_twice_is_refused(values, rate):
    arrays, _, declaration, tables = _normalize(values, rate=rate)
    table = tables["person"].copy()
    for (_, column), result in arrays.items():
        table[column] = result
    with pytest.raises(ValueError, match="normalized exactly once"):
        normalize_money_tables(
            {"person": table},
            declaration,
            rate=Fraction(rate),
            rate_text=rate,
            factors={"capital_income": _identity_factor()},
        )


@PROPERTY
@given(
    values=st.lists(_AMOUNTS, min_size=2, max_size=12),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    factor=_FACTORS,
)
def test_permuting_rows_commutes_with_normalization(values, seed, factor):
    order = np.random.default_rng(seed).permutation(len(values))
    arrays, receipts, *_ = _normalize(values, factor=_factor(factor))
    shuffled, shuffled_receipts, *_ = _normalize(
        [values[index] for index in order], factor=_factor(factor)
    )
    for key in arrays:
        assert shuffled[key].tolist() == arrays[key][order].tolist()
    assert shuffled_receipts == receipts


@PROPERTY
@example(value=0.004, rate="1.000", factor=1500000, quote="to_per_from")
@given(
    value=_AMOUNTS,
    rate=_RATES,
    factor=_FACTORS,
    quote=st.sampled_from(["to_per_from", "from_per_to"]),
)
def test_two_step_rounding_differs_from_one_step_by_at_most_one_cent(
    value, rate, factor, quote
):
    exact_factor = Fraction(factor, 10**6)
    _, two, one = _oracle(value, Fraction(rate), quote, exact_factor)
    assert abs(two - one) <= math.floor(1 + exact_factor / 2)
    if exact_factor < 2:
        assert abs(two - one) <= 1
    arrays, *_ = _normalize([value], rate=rate, quote=quote, factor=_factor(factor))
    assert int(arrays["person", "rental_income_cents"][0]) == two


@PROPERTY
@example(value=0.04, factor=1079073)
@given(
    value=_AMOUNTS,
    factor=st.sampled_from([1077090, 1079073, 1081052]),
)
def test_the_three_declared_factors_bound_two_step_difference_to_one_cent(
    value, factor
):
    # For y = x * rate * 100, half-even r satisfies |r(y) - y| <= 1/2.
    # Thus |r(r(y)*f) - r(y*f)| <= f/2 + 1 < 2 for each declared f < 2.
    # The difference is an integer, so its magnitude is at most one cent.
    exact_factor = Fraction(factor, 10**6)
    assert exact_factor / 2 + 1 < 2
    _, two, one = _oracle(value, Fraction("1.654"), "to_per_from", exact_factor)
    assert abs(two - one) <= 1
    arrays, *_ = _normalize([value], rate="1.654", factor=_factor(factor))
    assert int(arrays["person", "rental_income_cents"][0]) == two
