"""Exact factor arithmetic, evidence and replay without country assumptions."""

import hashlib
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
from microcosm.build.reference_profiles import validate_reference_factor_evidence
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.reference_factors import (
    factor_context_fixture,
    factor_fixture,
    published_projection_fixture,
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


# ---------------------------------------------------------------------------
# Version 1 keeps its behaviour beside the published-projection operation
# ---------------------------------------------------------------------------


def test_version_1_refuses_a_projection_source_slot_with_todays_message():
    declaration, facts, identity = factor_fixture()
    declaration["source_schedule"][0]["validation"]["assertion"] = "source_projection"
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(declaration, facts, identity)
    assert str(error.value) == "Source schedule requires observations."


def test_version_1_refuses_a_projection_target_slot_with_todays_message():
    declaration, facts, identity = factor_fixture()
    declaration["target_schedule"][0]["validation"]["assertion"] = "source_projection"
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(declaration, facts, identity)
    assert str(error.value) == "Target schedules prefer observations only."


def test_version_1_refuses_the_new_field_set_under_its_own_name():
    declaration, facts, identity = factor_fixture()
    del declaration["growth"], declaration["projection_policy"]
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(declaration, facts, identity)
    assert str(error.value) == (
        "Factor declaration missing fields: ['growth', 'projection_policy']."
    )


# The identity factor_fixture() builds at frequency 2, as literals, so a later
# change to the fixture's schema constants cannot move a version-1 pin.
_PINNED_IDENTITY = {
    "source_commit": "c" * 40,
    "facts_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
    "artifact_schema_version": "policyengine_ledger.consumer_artifact.v1",
    "consumer_fact_schema_versions": ["chronicle.consumer_fact.v4"],
    "consumer_fact_schema_sha256": "d" * 64,
    "fact_row_count": 4,
}


def _version_1_case(case):
    if case == "growth_zero":
        return factor_fixture(feed_identity=_PINNED_IDENTITY)
    if case == "growth_none":
        declaration, facts, identity = factor_fixture(
            feed_identity=_PINNED_IDENTITY, frequency=1
        )
        declaration["growth"] = None
        return declaration, facts, identity
    if case == "growth_three_inexact":
        return factor_fixture(
            feed_identity=_PINNED_IDENTITY,
            growth="3",
            expected_numerator=_oracle_factor("100", "110", "3", 2, 6)[0],
        )
    assert case == "diagnostic_all_observed"
    declaration, facts, identity = factor_fixture(feed_identity=_PINNED_IDENTITY)
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
    return declaration, facts, identity


# Copied from a baseline frozen at microcosm 17fecf5e3, before the
# published-projection operation was added, and never regenerated from a later
# tree (rf-projection-op-20261117 baseline/at-17fecf5e3, SHA256SUMS digest
# f9bc283354ccb9532bf573be86f6958a0040e09dc889fb9a7a03f340446339e3). Per case:
# the operational numerator over 10**6, receipt declaration_sha256, the sha256
# of the canonical receipt bytes, receipt_sha256, and the raw declaration's
# document hash.
_VERSION_1_BYTES = {
    "growth_zero": (
        1100000,
        "657f24fe7b87c12b5f2f4cc2fd494f2a4e1b9c154de08e41cbf304c094ae8638",
        "1bb03c0fb36cb0371a1339b083c6bd75b7bb822f49ad8e97602e137ddbc518e3",
        "cb48d93f9a0ae6611813bde4ec018826ab309fac2d55dafd6080459343332ee2",
        "657f24fe7b87c12b5f2f4cc2fd494f2a4e1b9c154de08e41cbf304c094ae8638",
    ),
    "growth_none": (
        1100000,
        "628263adb6a682a8f1585aa14264361dcf093becec9ac2da646cda7d17f16abc",
        "be2e0b141858796bb71a942e955aeaea390fcabd4480de59a1b53e7b15c8c858",
        "7e6cfea668e1e5a993da7a112ccf78a1d27f287a2b37617ae919bc121a3a4e87",
        "628263adb6a682a8f1585aa14264361dcf093becec9ac2da646cda7d17f16abc",
    ),
    "growth_three_inexact": (
        1108189,
        "5077f75703a82ca339e54fa0890fd701cfba35d6c93c950fcfe9f0aa4878e38e",
        "03a1c32a62e1de682a08524182dbb6eb81e7f4e7bca5ea41e18acdd988fefc3c",
        "db7f13a047fe39ca2409cba9135541f668eac8121cb8f138cf1080d42685b1f3",
        "5077f75703a82ca339e54fa0890fd701cfba35d6c93c950fcfe9f0aa4878e38e",
    ),
    "diagnostic_all_observed": (
        1100000,
        "e80da08ee25c80cd032df855fe18af83d3e12932b5c2a69fd830b15f86c5301f",
        "6db434da6062283ded02c39a472be38371ca97193ebffb64322796d7898dbdc0",
        "c3dbba1e129c79a99244eaea9e6037f4169784ffd731880e5997775722a6708c",
        "e80da08ee25c80cd032df855fe18af83d3e12932b5c2a69fd830b15f86c5301f",
    ),
}


@pytest.mark.parametrize("case", sorted(_VERSION_1_BYTES))
def test_version_1_bytes_match_the_frozen_baseline(case):
    numerator, declaration_sha, receipt_bytes_sha, receipt_sha, raw_sha = (
        _VERSION_1_BYTES[case]
    )
    declaration, facts, identity = _version_1_case(case)
    assert factor_document_sha256(declaration) == raw_sha
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert factor == {"numerator": numerator, "denominator": 10**6}
    assert receipt["declaration_sha256"] == declaration_sha
    assert hashlib.sha256(canonical_json(receipt)).hexdigest() == receipt_bytes_sha
    assert receipt["receipt_sha256"] == receipt_sha


# ---------------------------------------------------------------------------
# The published-projection operation
# ---------------------------------------------------------------------------

_PUBLISHED = "published_projection_period_average_ratio"


def _rehashed(receipt):
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return {**body, "receipt_sha256": factor_document_sha256(body)}


def _entry(declaration, factor, receipt):
    return {
        "declaration": declaration,
        "operational_factor": factor,
        "receipt": receipt,
    }


def test_the_new_operation_refuses_version_1_fields():
    declaration, facts, identity = factor_fixture()
    declaration["operation"] = _PUBLISHED
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(declaration, facts, identity)
    assert str(error.value) == (
        "Unknown factor declaration fields: ['growth', 'projection_policy']."
    )


def test_published_projection_resolves_and_records_the_projection():
    declaration, facts, identity = published_projection_fixture()
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert factor == {"numerator": 1100000, "denominator": 1000000}
    assert set(receipt) == {
        "schema_version",
        "operation",
        "operation_version",
        "factor_id",
        "declaration_sha256",
        "feed_identity",
        "source_period",
        "target_period",
        "target_assertion",
        "source_schedule",
        "target_schedule",
        "subperiods_per_year",
        "denominator_mean",
        "numerator_mean",
        "unrounded_factor",
        "rounding",
        "operational_factor",
        "validity",
        "evidence",
        "decision_references",
        "receipt_sha256",
    }
    assert receipt["target_assertion"] == "source_projection"
    assert [slot["assertion"] for slot in receipt["target_schedule"]] == [
        "source_projection"
    ]
    assert [slot["assertion"] for slot in receipt["source_schedule"]] == ["observation"]
    assert _fraction(receipt["unrounded_factor"]) == Fraction(11, 10)
    assert (receipt["operation"], receipt["operation_version"]) == (_PUBLISHED, 1)
    assert receipt["declaration_sha256"] == factor_document_sha256(declaration)
    replay_reference_factor(declaration, facts, identity, factor, receipt)

    pin = {**identity, "country": "toy", "build": "prose"}
    revised_factor, revised_receipt = resolve_reference_factor(declaration, facts, pin)
    assert revised_factor == factor
    assert revised_receipt == receipt
    assert revised_receipt["feed_identity"] == identity
    replay_reference_factor(declaration, facts, pin, factor, receipt)


def test_published_projection_refuses_a_projection_source():
    declaration, facts, identity = published_projection_fixture()
    entry = _entry(declaration, *resolve_reference_factor(declaration, facts, identity))
    slot = declaration["source_schedule"][0]
    slot["validation"]["assertion"] = "source_projection"
    slot["selector"]["assertion"] = "source_projection"
    facts[0]["assertion"] = "source_projection"
    message = "Source schedule requires observations."
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(declaration, facts, identity)
    assert str(error.value) == message
    with pytest.raises(ValueError) as error:
        validate_factor_context({"schema_version": 1, "factors": [entry]})
    assert str(error.value) == message


def test_published_projection_refuses_an_observation_target():
    declaration, facts, identity = published_projection_fixture()
    entry = _entry(declaration, *resolve_reference_factor(declaration, facts, identity))
    declared = deepcopy(declaration)
    declared["target_schedule"][0]["validation"]["assertion"] = "observation"
    message = "Published-projection target schedules require source_projection slots."
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(declared, facts, identity)
    assert str(error.value) == message
    with pytest.raises(ValueError) as error:
        validate_factor_context(
            {"schema_version": 1, "factors": [{**entry, "declaration": declared}]}
        )
    assert str(error.value) == message

    # Declared a projection, but the selector and the fact are observations.
    observed_facts = deepcopy(facts)
    observed_facts[-1]["assertion"] = "observation"
    mismatched = deepcopy(declaration)
    mismatched["target_schedule"][0]["selector"]["assertion"] = "observation"
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(mismatched, observed_facts, identity)
    assert str(error.value) == "Factor operand assertion mismatch."
    # A selector's assertion filters on the fact's, so it then finds nothing.
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(declaration, observed_facts, identity)
    assert str(error.value) == "Factor selector must resolve exactly once; found 0."


@pytest.mark.parametrize("purpose", [*sorted(FACTOR_PURPOSES), "money"])
def test_published_projection_admits_money_only(purpose):
    declaration, facts, identity = published_projection_fixture()
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    if purpose == "money_normalization_input":
        assert factor == declaration["expected_operational_factor"]
        context = {
            "schema_version": 1,
            "factors": [_entry(declaration, factor, receipt)],
        }
        assert list(validate_factor_context(context, [declaration["id"]])) == [
            declaration["id"]
        ]
        return
    message = (
        "Unknown factor purpose."
        if purpose == "money"
        else f"Factor purpose {purpose!r} is not admitted by {_PUBLISHED}, "
        "which ages money only."
    )
    refused = {**declaration, "purpose": purpose}
    with pytest.raises(ValueError) as error:
        resolve_reference_factor(refused, facts, identity)
    assert str(error.value) == message
    # The decode path never resolves: the declaration check alone refuses, before
    # the stale factor and receipt are read.
    entry, entry_id = _entry(refused, factor, receipt), refused["id"]
    with pytest.raises(ValueError) as error:
        validate_factor_context({"schema_version": 1, "factors": [entry]}, [entry_id])
    assert str(error.value) == message
    with pytest.raises(ValueError) as error:
        validate_reference_factor_evidence({"selected_factors": {entry_id: entry}})
    assert str(error.value) == message


def _drop(key):
    def mutate(declaration):
        del declaration[key]

    return mutate


def _set(key, value):
    def mutate(declaration):
        declaration[key] = value

    return mutate


def _target_offset(declaration):
    declaration["target_schedule"][0]["projection_offset"] = 0


def _digits(declaration):
    declaration["rounding"]["digits"] = 19


def _expected_denominator(declaration):
    declaration["expected_operational_factor"] = {
        "numerator": 11000000,
        "denominator": 10**7,
    }


def _expired(declaration):
    declaration["validity"]["valid_until"] = "2001-12-31"


def _prose_identity(declaration):
    declaration["feed_identity"] = {**declaration["feed_identity"], "build": "Prose"}


_PUBLISHED_FIELDS = (
    "schema_version",
    "id",
    "operation",
    "operation_version",
    "purpose",
    "feed_identity",
    "source_period",
    "target_period",
    "source_schedule",
    "target_schedule",
    "subperiods_per_year",
    "rounding",
    "expected_operational_factor",
    "validity",
    "evidence",
    "decision_references",
)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (_set("growth", None), "Unknown factor declaration fields"),
        (
            _set("projection_policy", "observed_preferred_latest_contiguous"),
            "Unknown factor declaration fields",
        ),
        (_set("unknown", "field"), "Unknown factor declaration fields"),
        *((_drop(key), "missing fields") for key in _PUBLISHED_FIELDS),
        (_target_offset, "Unknown factor schedule slot fields"),
        (_set("operation_version", 2), "Unknown factor operation or operation version"),
        (
            _set("operation_version", True),
            "Unknown factor operation or operation version",
        ),
        (_set("target_period", "2002"), "requires an integer target_period"),
        (_set("target_period", True), "requires an integer target_period"),
        (_set("target_period", None), "requires an integer target_period"),
        (_set("schema_version", 2), "schema_version 1"),
        (_set("subperiods_per_year", 0), "subperiods_per_year"),
        (_set("subperiods_per_year", 13), "subperiods_per_year"),
        (_digits, "rounding digits"),
        (_expected_denominator, r"positive p / 10\*\*digits"),
        (_expired, "expired"),
        (_prose_identity, "Unknown factor declaration feed identity fields"),
    ],
)
def test_published_projection_field_set_is_closed(mutate, message):
    declaration, facts, identity = published_projection_fixture()
    mutate(declaration)
    with pytest.raises(ValueError, match=message):
        resolve_reference_factor(declaration, facts, identity)


@pytest.mark.parametrize(
    "target,expected", [("2000001", 1000000), ("2000003", 1000002)]
)
def test_published_projection_rounds_once_half_even(target, expected):
    declaration, facts, identity = published_projection_fixture(
        source_levels=("2000000",),
        target_levels=(target,),
        expected_numerator=expected,
    )
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert factor == {"numerator": expected, "denominator": 10**6}
    assert _fraction(receipt["unrounded_factor"]) == Fraction(int(target), 2000000)


def test_published_projection_weighted_period_means():
    # (100*1 + 104*3) / 4 = 103 and (110*1 + 118*3) / 4 = 116.
    declaration, facts, identity = published_projection_fixture(
        frequency=2,
        source_levels=("100", "104"),
        target_levels=("110", "118"),
        weights=("1", "3"),
        expected_numerator=1126214,
    )
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    assert _fraction(receipt["denominator_mean"]) == Fraction(103)
    assert _fraction(receipt["numerator_mean"]) == Fraction(116)
    assert _fraction(receipt["unrounded_factor"]) == Fraction(116, 103)
    assert factor == {"numerator": 1126214, "denominator": 10**6}
    assert [slot["weight"] for slot in receipt["target_schedule"]] == ["1", "3"]


def test_published_projection_refuses_a_wrong_expected_factor():
    declaration, facts, identity = published_projection_fixture(
        expected_numerator=1100001
    )
    with pytest.raises(ValueError) as refused:
        resolve_reference_factor(declaration, facts, identity)
    assert str(refused.value) == (
        "Declared expected factor mismatch: computed "
        "{'numerator': 1100000, 'denominator': 1000000}; re-declaration is required."
    )


def test_published_projection_refuses_another_feed():
    declaration, facts, identity = published_projection_fixture()
    pin = {**identity, "fact_row_count": identity["fact_row_count"] + 1}
    with pytest.raises(ValueError) as refused:
        resolve_reference_factor(declaration, facts, pin)
    assert str(refused.value) == (
        "Factor feed identity mismatch; re-declaration is required."
    )


@pytest.mark.parametrize(
    "source,target,message",
    [
        ("0", "110", "Factor denominator operands must be positive."),
        ("-100", "-110", "Factor denominator operands must be positive."),
        ("100", "0", "Published-projection target operands must be positive."),
        ("100", "-110", "Published-projection target operands must be positive."),
    ],
)
def test_published_projection_refuses_non_positive_operands(source, target, message):
    # A positive expectation, so the declaration check cannot refuse first. Two
    # negative levels would otherwise give a positive factor that validates.
    declaration, facts, identity = published_projection_fixture(
        source_levels=(source,), target_levels=(target,), expected_numerator=1100000
    )
    with pytest.raises(ValueError) as refused:
        resolve_reference_factor(declaration, facts, identity)
    assert str(refused.value) == message


@settings(max_examples=70, deadline=None)
@example(source=2, target=1, digits=0)
@example(source=2000000, target=2000001, digits=6)
@given(
    source=st.integers(1, 10**12),
    target=st.integers(1, 10**12),
    digits=st.integers(0, 18),
)
def test_published_projection_matches_fraction_oracle(source, target, digits):
    # The built-in round on a Fraction takes ties to even; no production helper.
    expected = round(Fraction(target, source) * 10**digits)
    declaration, facts, identity = published_projection_fixture(
        source_levels=(str(source),),
        target_levels=(str(target),),
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
    assert factor == {"numerator": expected, "denominator": 10**digits}
    assert receipt == _rehashed(receipt)
    assert canonical_json(receipt)
    replay_reference_factor(declaration, facts, identity, factor, receipt)


def test_published_projection_validates_and_replays_beside_version_1():
    context, facts, identity = factor_context_fixture()
    declaration, projection_facts, _ = published_projection_fixture(
        feed_identity=identity
    )
    factor, receipt = resolve_reference_factor(declaration, projection_facts, identity)
    entry = _entry(declaration, factor, receipt)
    context["factors"].append(entry)
    selected = validate_factor_context(context)
    assert selected == {"toy_factor": context["factors"][0], declaration["id"]: entry}
    version_1 = context["factors"][0]
    replay_reference_factor(
        version_1["declaration"],
        facts,
        identity,
        version_1["operational_factor"],
        version_1["receipt"],
    )
    replay_reference_factor(declaration, projection_facts, identity, factor, receipt)
    require_current_factors(selected, today=date(2050, 1, 1))
    with pytest.raises(ValueError, match="expired or not yet valid"):
        require_current_factors(selected, today=date(2100, 1, 1))

    relabelled = _entry(
        declaration,
        factor,
        _rehashed({**receipt, "operation": "projected_period_average_ratio"}),
    )
    with pytest.raises(ValueError) as error:
        validate_factor_context({"schema_version": 1, "factors": [relabelled]})
    assert str(error.value) == "Factor declaration and receipt are incompatible."

    tampered = _entry(
        declaration, factor, _rehashed({**receipt, "target_assertion": "observation"})
    )
    assert validate_factor_context({"schema_version": 1, "factors": [tampered]})
    with pytest.raises(ValueError) as error:
        replay_reference_factor(
            declaration, projection_facts, identity, factor, tampered["receipt"]
        )
    assert str(error.value) == "Factor replay differs from declared result or evidence."
