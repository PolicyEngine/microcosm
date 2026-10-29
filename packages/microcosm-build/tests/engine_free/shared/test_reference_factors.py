"""Exact factor arithmetic, evidence and replay without country assumptions."""

from copy import deepcopy
from datetime import date
from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.build.ledger_artifact import LexicalFloat
from microcosm.build.reference_factors import (
    FACTOR_PURPOSES,
    certified_nth_root_bounds,
    factor_document_sha256,
    replay_reference_factor,
    require_current_factors,
    resolve_reference_factor,
    validate_factor_context,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.reference_factors import (
    factor_context_fixture,
    factor_fixture,
)


def _fraction(row):
    return Fraction(row["numerator"], row["denominator"])


def _oracle_root(radicand, degree, places=50):
    """Independent integer bisection oracle; no production arithmetic helpers."""
    denominator = 10**places
    low, high = (
        0,
        max(2, (radicand.numerator // radicand.denominator + 2)) * denominator,
    )
    while high - low > 1:
        middle = (low + high) // 2
        if (
            middle**degree * radicand.denominator
            <= radicand.numerator * denominator**degree
        ):
            low = middle
        else:
            high = middle
    lower = Fraction(low, denominator)
    if lower**degree == radicand:
        return lower, lower
    return lower, Fraction(high, denominator)


def _oracle_round(number, digits):
    scaled = number * 10**digits
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if remainder * 2 > scaled.denominator or (
        remainder * 2 == scaled.denominator and quotient % 2
    ):
        quotient += 1
    return quotient


def _oracle_factor(source, anchor, growth, frequency, digits, *, places=50):
    while places <= 4096:
        low, high = _oracle_root(1 + Fraction(growth) / 100, frequency, places)
        lower = (
            Fraction(anchor)
            / Fraction(source)
            * sum(low**index for index in range(frequency))
            / frequency
        )
        upper = (
            Fraction(anchor)
            / Fraction(source)
            * sum(high**index for index in range(frequency))
            / frequency
        )
        if _oracle_round(lower, digits) == _oracle_round(upper, digits):
            return _oracle_round(lower, digits), (low, high)
        places = max(1, places * 2)
    raise AssertionError("Independent oracle could not enclose a rounded factor.")


def test_independent_oracle_refines_ambiguous_rounding():
    coarse = _oracle_root(Fraction(103, 100), 2, places=1)
    assert _oracle_round((1 + coarse[0]) / 2, 6) != _oracle_round(
        (1 + coarse[1]) / 2, 6
    )
    expected, bounds = _oracle_factor("1", "1", "3", 2, 6, places=1)
    assert expected == 1007445
    assert bounds[1] - bounds[0] < coarse[1] - coarse[0]


def test_factor_resolves_exact_operands_once_and_replays():
    context, facts, identity = factor_context_fixture()
    entry = context["factors"][0]
    receipt = entry["receipt"]
    assert entry["operational_factor"] == {"numerator": 1100000, "denominator": 1000000}
    assert _fraction(receipt["denominator_mean"]) == 100
    assert receipt["root_bounds"]["exact"] is True
    assert [slot["status"] for slot in receipt["target_schedule"]] == [
        "observed",
        "projected",
    ]
    assert all(
        slot["numeric_token_proof"]["decimal_values_equal"]
        for slot in receipt["source_schedule"]
    )
    replay_reference_factor(
        entry["declaration"], facts, identity, entry["operational_factor"], receipt
    )
    assert validate_factor_context(context, ["toy_factor"]) == {"toy_factor": entry}


def test_currency_check_without_selected_factors_is_a_noop():
    require_current_factors({}, today=date(2026, 11, 5))


def test_factor_resolution_and_replay_ignore_caller_feed_pin_prose():
    declaration, facts, identity = factor_fixture()
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    pin = {
        **identity,
        "build": "Reviewed prose edit",
        "source_repo": "Reviewed repository description",
    }

    revised_factor, revised_receipt = resolve_reference_factor(declaration, facts, pin)

    assert revised_factor == factor
    assert revised_receipt == receipt
    assert revised_receipt["feed_identity"] == identity
    assert declaration["feed_identity"] == identity
    replay_reference_factor(declaration, facts, pin, factor, receipt)


@pytest.mark.parametrize("container", ["declaration", "context"])
@pytest.mark.parametrize(
    "field",
    [
        "source_commit",
        "facts_sha256",
        "manifest_sha256",
        "artifact_schema_version",
        "consumer_fact_schema_versions",
        "consumer_fact_schema_sha256",
        "fact_row_count",
    ],
)
def test_factor_identity_requires_every_authenticated_field(container, field):
    context, facts, identity = factor_context_fixture()
    declaration = context["factors"][0]["declaration"]
    document = declaration if container == "declaration" else context
    document["feed_identity"] = {
        key: value for key, value in document["feed_identity"].items() if key != field
    }

    with pytest.raises(ValueError, match="missing fields"):
        if container == "declaration":
            resolve_reference_factor(declaration, facts, identity)
        else:
            validate_factor_context(context)


@pytest.mark.parametrize("container", ["declaration", "context"])
@pytest.mark.parametrize("field", ["build", "source_repo"])
def test_factor_declared_identity_refuses_prose_fields(container, field):
    context, facts, identity = factor_context_fixture()
    declaration = context["factors"][0]["declaration"]
    document = declaration if container == "declaration" else context
    document["feed_identity"] = {**document["feed_identity"], field: "Prose"}

    with pytest.raises(ValueError, match="Unknown factor .*feed identity fields"):
        if container == "declaration":
            resolve_reference_factor(declaration, facts, identity)
        else:
            validate_factor_context(context)


def test_irrational_root_is_an_enclosure_never_an_exact_rational():
    expected, _ = _oracle_factor("100", "110", "3", 4, 6)
    declaration, facts, identity = factor_fixture(
        frequency=4, growth="3", expected_numerator=expected
    )
    _, receipt = resolve_reference_factor(declaration, facts, identity)
    bounds = receipt["root_bounds"]
    lower, upper = _fraction(bounds["lower"]), _fraction(bounds["upper"])
    assert bounds["exact"] is False
    assert lower < upper
    assert lower**4 < Fraction(103, 100) < upper**4
    factor_bounds = receipt["unrounded_factor_bounds"]
    assert _oracle_round(_fraction(factor_bounds["lower"]), 6) == expected
    assert _oracle_round(_fraction(factor_bounds["upper"]), 6) == expected


def test_exact_root_is_recognized_without_false_irrationality():
    lower, upper, exact = certified_nth_root_bounds(Fraction(121, 100), 2, 8)
    assert lower == upper == Fraction(11, 10)
    assert exact is True


@pytest.mark.parametrize(
    "anchor,expected", [("2469135", 1234568), ("2469133", 1234566)]
)
def test_single_final_rounding_uses_nearest_half_even(anchor, expected):
    declaration, facts, identity = factor_fixture(
        frequency=1,
        source_level="2000000",
        anchor_level=anchor,
        expected_numerator=expected,
    )
    factor, _ = resolve_reference_factor(declaration, facts, identity)
    assert factor["numerator"] == expected


def test_observed_only_diagnostic_needs_no_projection_operand():
    declaration, facts, identity = factor_fixture()
    observed = deepcopy(facts[-2])
    observed["aggregate_fact_key"] = "target-1"
    observed["period"]["value"] = declaration["target_schedule"][1]["selector"][
        "period_value"
    ]
    observed["period_coverage"] = declaration["target_schedule"][1]["validation"][
        "period_coverage"
    ]
    facts.append(observed)
    declaration["purpose"] = "diagnostic"
    declaration["growth"] = None
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert factor["numerator"] == 1100000
    assert receipt["growth"] is None
    assert all(slot["status"] == "observed" for slot in receipt["target_schedule"])


@pytest.mark.parametrize(
    "field,mutation,message",
    [
        ("operation", "expression_eval", "operation"),
        ("operation_version", 2, "operation"),
        ("schema_version", 2, "schema_version"),
        ("unknown", "field", "Unknown"),
        ("subperiods_per_year", True, "subperiods_per_year"),
        ("subperiods_per_year", 0, "subperiods_per_year"),
        ("projection_policy", "june_anchor", "projection policy"),
        ("purpose", "execute", "purpose"),
        ("source_period", True, "period labels"),
        ("target_period", None, "period labels"),
        ("source_period", "", "period labels"),
    ],
)
def test_closed_declaration_guards(field, mutation, message):
    declaration, facts, identity = factor_fixture()
    declaration[field] = mutation
    with pytest.raises(ValueError, match=message):
        resolve_reference_factor(declaration, facts, identity)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing", "exactly once"),
        ("duplicate", "Duplicate"),
        ("unit", "unit mismatch"),
        ("source_sha256", "source_sha256 mismatch"),
        ("source_vintage", "source_vintage mismatch"),
        ("entity_role", "entity_role mismatch"),
        ("coverage", "coverage mismatch"),
        ("boolean", "Boolean"),
        ("nan", "finite"),
        ("float_without_token", "lexical_floats=True"),
        ("lossy_float", "does not reproduce"),
        ("negative", "positive"),
    ],
)
def test_operand_guards(mutation, message):
    declaration, facts, identity = factor_fixture()
    if mutation == "missing":
        facts.pop(0)
    elif mutation == "duplicate":
        duplicate = deepcopy(facts[0])
        duplicate["aggregate_fact_key"] += "-duplicate"
        facts.append(duplicate)
    elif mutation == "unit":
        facts[0]["observed_measure"]["unit"] = "other"
    elif mutation == "source_sha256":
        facts[0]["source"]["source_sha256"] = "b" * 64
    elif mutation == "source_vintage":
        facts[0]["source"]["vintage"] = "revised"
    elif mutation == "entity_role":
        facts[0]["entity"]["role"] = "other"
    elif mutation == "coverage":
        facts[0]["period_coverage"]["end_date"] = "2001-02-01"
        # The fixture shares this dictionary with its declaration; keep the pin.
        declaration["source_schedule"][0]["validation"]["period_coverage"][
            "end_date"
        ] = "2001-01-01"
        facts[0]["period_coverage"] = {
            "start_date": "2001-01-01",
            "end_date": "2001-02-01",
        }
    elif mutation == "boolean":
        facts[0]["value"] = True
    elif mutation == "nan":
        facts[0]["value"] = Decimal("NaN")
    elif mutation == "float_without_token":
        facts[0]["value"] = 100.0
    elif mutation == "lossy_float":
        facts[0]["value"] = LexicalFloat("100.000000000000000001")
    elif mutation == "negative":
        facts[0]["value"] = Decimal("-1")
    with pytest.raises(ValueError, match=message):
        resolve_reference_factor(declaration, facts, identity)


def test_lexical_float_conversion_is_proved_and_used_exactly():
    declaration, facts, identity = factor_fixture(
        source_level="100.25", anchor_level="110.275", expected_numerator=1100000
    )
    for fact in facts:
        fact["value"] = LexicalFloat(str(fact["value"]))
    _, receipt = resolve_reference_factor(declaration, facts, identity)
    proof = receipt["source_schedule"][0]["numeric_token_proof"]
    assert proof == {
        "numeric_representation": "binary_float_with_lexical_token",
        "lexical_token": "100.25",
        "recovered_decimal": "100.25",
        "decimal_values_equal": True,
    }
    assert _fraction(receipt["denominator_mean"]) == Fraction(401, 4)


def test_integer_operands_are_exact_beyond_binary_float_precision():
    source = 10**80 + 10
    anchor = source * 11 // 10
    declaration, facts, identity = factor_fixture(
        source_level=str(source),
        anchor_level=str(anchor),
        expected_numerator=1100000,
    )
    for fact in facts:
        fact["value"] = int(fact["value"])

    factor, receipt = resolve_reference_factor(declaration, facts, identity)

    assert factor == {"numerator": 1100000, "denominator": 1000000}
    assert _fraction(receipt["denominator_mean"]) == source
    assert _fraction(receipt["numerator_mean_bounds"]["lower"]) == anchor
    assert _fraction(receipt["numerator_mean_bounds"]["upper"]) == anchor
    operands = [
        *receipt["source_schedule"],
        *(slot for slot in receipt["target_schedule"] if slot["status"] == "observed"),
        receipt["growth"],
    ]
    for operand in operands:
        token = operand["decimal_operand"]
        assert operand["numeric_token_proof"] == {
            "numeric_representation": "exact_integer",
            "lexical_token": token,
            "recovered_decimal": token,
            "decimal_values_equal": True,
        }
    assert receipt["source_schedule"][0]["decimal_operand"] == str(source)
    assert receipt["target_schedule"][0]["decimal_operand"] == str(anchor)


def test_source_projection_requires_existing_allow_guard_even_with_exact_key():
    declaration, facts, identity = factor_fixture()
    declaration["growth"]["assertion_policy"] = "observed_only"
    with pytest.raises(ValueError, match="allow_source_projection"):
        resolve_reference_factor(declaration, facts, identity)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("schedule_order", "misordered"),
        ("duplicate_period", "Duplicate"),
        ("schedule_gap", "noncontiguous"),
        ("negative_weight", "positive"),
        ("offset", "projection offsets"),
        ("unknown_selector", "selector"),
        ("growth_floor", "positive 1"),
        ("sensitivity", "sensitivity provenance"),
        ("expired", "expired"),
        ("wrong_expected", "expected factor mismatch"),
        ("wrong_feed", "feed identity"),
        ("wrong_growth_period", "exactly once"),
        ("source_column", "source_column_id mismatch"),
    ],
)
def test_numeric_and_schedule_guards(mutation, message):
    declaration, facts, identity = factor_fixture()
    if mutation == "schedule_order":
        declaration["source_schedule"].reverse()
    elif mutation == "duplicate_period":
        declaration["source_schedule"][1]["selector"]["period_value"] = "2001-01-01"
    elif mutation == "schedule_gap":
        declaration["source_schedule"][1]["validation"]["period_coverage"][
            "start_date"
        ] = "2001-01-04"
    elif mutation == "negative_weight":
        declaration["source_schedule"][0]["weight"] = "-1"
    elif mutation == "offset":
        declaration["target_schedule"][1]["projection_offset"] = 0
    elif mutation == "unknown_selector":
        declaration["source_schedule"][0]["selector"]["unit"] = "money_per_week"
    elif mutation == "growth_floor":
        facts[-1]["value"] = Decimal("-100")
    elif mutation == "sensitivity":
        declaration["growth"]["sensitivity_provenance"] = []
    elif mutation == "expired":
        declaration["validity"]["valid_until"] = "2001-12-31"
    elif mutation == "wrong_expected":
        declaration["expected_operational_factor"]["numerator"] += 1
    elif mutation == "wrong_feed":
        identity = {**identity, "facts_sha256": "wrong"}
    elif mutation == "wrong_growth_period":
        declaration["growth"]["selector"]["period_value"] = "2003"
    elif mutation == "source_column":
        declaration["growth"]["validation"]["source_column_id"] = "wrong"
    with pytest.raises(ValueError, match=message):
        resolve_reference_factor(declaration, facts, identity)


def test_observed_preferred_requires_redeclaration_and_reanchors_projection():
    declaration, facts, identity = factor_fixture(frequency=4)
    observed = deepcopy(facts[-2])
    observed["aggregate_fact_key"] = "target-1"
    observed["value"] = Decimal("120")
    observed["period"]["value"] = declaration["target_schedule"][1]["selector"][
        "period_value"
    ]
    observed["period_coverage"] = declaration["target_schedule"][1]["validation"][
        "period_coverage"
    ]
    facts.append(observed)
    with pytest.raises(ValueError, match="expected factor mismatch"):
        resolve_reference_factor(declaration, facts, identity)
    # Growth zero makes the latest-anchor formula transparent: (110+120+120+120)/4.
    declaration["expected_operational_factor"]["numerator"] = 1175000
    _, receipt = resolve_reference_factor(declaration, facts, identity)
    assert _fraction(receipt["numerator_mean_bounds"]["lower"]) == Fraction(470, 4)
    assert [slot["projection_offset"] for slot in receipt["target_schedule"]] == [
        0,
        0,
        1,
        2,
    ]
    assert receipt["target_schedule"][2]["projection_anchor"] == {
        "aggregate_fact_key": "target-1"
    }


def test_observed_gap_refuses_instead_of_guessing_anchor():
    declaration, facts, identity = factor_fixture(frequency=4)
    observed = deepcopy(facts[-2])
    observed["period"]["value"] = declaration["target_schedule"][2]["selector"][
        "period_value"
    ]
    observed["period_coverage"] = declaration["target_schedule"][2]["validation"][
        "period_coverage"
    ]
    facts.append(observed)
    with pytest.raises(ValueError, match="contiguous from the anchor"):
        resolve_reference_factor(declaration, facts, identity)


def test_self_consistent_receipt_cannot_replace_compile_replay():
    context, facts, identity = factor_context_fixture()
    entry = context["factors"][0]
    entry["declaration"]["expected_operational_factor"]["numerator"] = 1200000
    entry["operational_factor"]["numerator"] = 1200000
    entry["receipt"]["operational_factor"]["numerator"] = 1200000
    entry["receipt"]["declaration_sha256"] = factor_document_sha256(
        entry["declaration"]
    )
    entry["receipt"]["receipt_sha256"] = factor_document_sha256(
        {
            key: value
            for key, value in entry["receipt"].items()
            if key != "receipt_sha256"
        }
    )
    assert validate_factor_context(context, ["toy_factor"])
    with pytest.raises(ValueError, match="expected factor mismatch"):
        replay_reference_factor(
            entry["declaration"],
            facts,
            identity,
            entry["operational_factor"],
            entry["receipt"],
        )


def test_receipt_tampering_refuses_and_unselected_edits_are_local():
    context, _, _ = factor_context_fixture()
    other = deepcopy(context["factors"][0])
    other["declaration"]["id"] = "unused"
    other["receipt"]["receipt_sha256"] = "invalid"
    context["factors"].append(other)
    assert list(validate_factor_context(context, ["toy_factor"])) == ["toy_factor"]
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_factor_context(context, ["unused"])


@settings(max_examples=70, deadline=None)
@example(source=32, anchor=1, growth=0, frequency=2, digits=4)
@given(
    source=st.integers(1, 10**9),
    anchor=st.integers(1, 10**9),
    growth=st.integers(-49, 49),
    frequency=st.sampled_from([1, 2, 4, 12]),
    digits=st.sampled_from([4, 6, 8]),
)
def test_certified_rounding_and_root_bounds_match_independent_fraction_oracle(
    source, anchor, growth, frequency, digits
):
    source_text, anchor_text = str(Decimal(source) / 100), str(Decimal(anchor) / 100)
    expected, oracle_bounds = _oracle_factor(
        source_text, anchor_text, str(growth), frequency, digits
    )
    declaration, facts, identity = factor_fixture(
        source_level=source_text,
        anchor_level=anchor_text,
        growth=str(growth),
        frequency=frequency,
        digits=digits,
        expected_numerator=expected,
    )
    if expected == 0:
        with pytest.raises(
            ValueError, match="Expected operational factor must be positive"
        ):
            resolve_reference_factor(declaration, facts, identity)
        return
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert factor["numerator"] == expected
    lower, upper = (
        _fraction(receipt["root_bounds"]["lower"]),
        _fraction(receipt["root_bounds"]["upper"]),
    )
    radicand = 1 + Fraction(growth, 100)
    assert lower**frequency <= radicand <= upper**frequency
    assert lower <= oracle_bounds[1] and upper >= oracle_bounds[0]
    if receipt["root_bounds"]["exact"]:
        assert lower == upper and lower**frequency == radicand
    else:
        assert lower < upper


def test_very_large_integer_cross_products_do_not_overflow():
    huge = 10**200
    declaration, facts, identity = factor_fixture(
        source_level=str(huge),
        anchor_level=str(3 * huge),
        frequency=12,
        expected_numerator=3000000,
    )
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert factor["numerator"] == 3000000
    assert _fraction(receipt["denominator_mean"]) == huge


@pytest.mark.parametrize(
    "frequency,digits,message",
    [
        (13, 6, "subperiods_per_year.*1 to 12.*serializable"),
        (365, 6, "subperiods_per_year.*1 to 12.*serializable"),
        (12, 19, "rounding digits.*0 to 18.*serializable"),
    ],
)
def test_unsupported_exact_receipt_schedule_refuses_at_declaration(
    frequency, digits, message
):
    declaration, facts, identity = factor_fixture()
    declaration["subperiods_per_year"] = frequency
    declaration["rounding"]["digits"] = digits
    # A malformed schedule is intentional: range validation must precede it.
    declaration["source_schedule"] = []
    with pytest.raises(ValueError, match=message):
        resolve_reference_factor(declaration, facts, identity)


def test_largest_supported_schedule_and_rounding_have_serializable_exact_receipt():
    expected, _ = _oracle_factor("100", "110", "3", 12, 18)
    declaration, facts, identity = factor_fixture(
        frequency=12, growth="3", digits=18, expected_numerator=expected
    )
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert factor == {"numerator": expected, "denominator": 10**18}
    encoded = canonical_json(receipt)
    assert encoded and factor_document_sha256(receipt)
    replay_reference_factor(declaration, facts, identity, factor, receipt)


def test_root_refinement_stays_within_exact_receipt_precision_budget(monkeypatch):
    precisions = []

    def ambiguous_bounds(value, degree, decimal_places):
        precisions.append(decimal_places)
        return Fraction(1), Fraction(2), False

    monkeypatch.setattr(
        "microcosm.build.reference_factors.certified_nth_root_bounds", ambiguous_bounds
    )
    declaration, facts, identity = factor_fixture(frequency=12, digits=18)
    with pytest.raises(ValueError, match="supported exact receipt precision"):
        resolve_reference_factor(declaration, facts, identity)
    assert precisions == [26, 52, 104, 208]


@pytest.mark.parametrize("purpose", sorted(FACTOR_PURPOSES))
def test_factor_purposes_are_closed_and_additive(purpose):
    assert FACTOR_PURPOSES == {
        "profile_input",
        "diagnostic",
        "money_normalization_input",
    }
    declaration, facts, identity = factor_fixture()
    declaration["purpose"] = purpose
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert factor == declaration["expected_operational_factor"]
    declaration["purpose"] = "money"
    with pytest.raises(ValueError, match="Unknown factor purpose"):
        resolve_reference_factor(declaration, facts, identity)
