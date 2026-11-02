"""Summed-band money rows in ``scale_money_and_intervals`` (G8-r2c hunk A).

A coarse money row sums several published bands: the resolver turns its
``identity`` row into one weighted operand over the member set, and its count
companion keeps ``sum`` semantics over the same members. Compilation runs the
real ledger, so these tests check the arithmetic the matrix receives.
"""

from __future__ import annotations

import copy
import json
import math
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import _validate_target_references
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.reference_profiles import (
    resolve_reference_profiles,
    validate_reference_resolution,
)
from test_support.microcosm_build.reference_intervals import (
    SUMMED_MEASUREMENT,
    band_fact,
    proof_for,
    summed_profile_fixture,
    summed_row,
)

TEN = [f"n{index:02d}" for index in range(10)]


def _facts(labels, *, count=lambda i: 100 + i, money=lambda i: 1_000 * (i + 1)):
    facts = []
    for index, label in enumerate(labels):
        facts.append(band_fact("count", label, count(index)))
        facts.append(band_fact("money", label, money(index)))
    return facts


def _resolve(inputs, proofs=None, selection=None):
    base, profiles, context, intervals = inputs
    return resolve_reference_profiles(
        base,
        profiles,
        selection or {},
        intervals,
        factor_context=context,
        measurement_proofs=proofs,
    )


def _compile(effective, facts):
    references = _validate_target_references(effective, country=effective["country"])
    registry = compile_ledger_target_references(
        facts, references, country=effective["country"]
    )
    return {spec.name: spec for spec in registry.specs}


def _rows(document):
    return {row["name"]: row for row in document["target_references"]}


def test_ten_member_pair_resolves_to_one_operand_with_count_ten():
    inputs = summed_profile_fixture([TEN])
    effective, _, receipt = _resolve(inputs, proof_for())
    rows = _rows(effective)
    money, count = rows["toy_money_b0"], rows["toy_count_b0"]
    assert money["value_operation"] == "linear_combination"
    assert money["value_operands"] == [
        {
            "dimension_values": {"band": TEN},
            "expected_member_count": 10,
            "weight": 1.1,
        }
    ]
    assert count["value_operation"] == "sum"
    assert count["expected_member_count"] == 10
    assert "value_operands" not in count or count["value_operands"] == []
    for row in (money, count):
        assert row["metadata"]["activation_status"] == "active"
        assert json.loads(row["metadata"]["reference_prerequisites_remaining"]) == []
        assert "parking_reason" not in row["metadata"]
    assert receipt["measurement_proofs"] == proof_for()
    assert validate_reference_resolution(effective, receipt) == receipt


def test_summed_money_operand_resolves_amount_facts_only():
    """Counts and amounts share one record set and one band dimension."""
    inputs = summed_profile_fixture([TEN])
    effective, _, _ = _resolve(inputs, proof_for())
    facts = _facts(TEN)
    specs = _compile(effective, facts)
    money_keys = json.loads(specs["toy_money_b0"].metadata["ledger_member_fact_keys"])
    count_keys = json.loads(specs["toy_count_b0"].metadata["ledger_member_fact_keys"])
    assert sorted(money_keys) == sorted(
        f"ledger.aggregate_fact.v2:toy-money-{label}" for label in TEN
    )
    assert sorted(count_keys) == sorted(
        f"ledger.aggregate_fact.v2:toy-count-{label}" for label in TEN
    )
    assert specs["toy_count_b0"].value == sum(100 + i for i in range(10))
    assert math.isclose(
        specs["toy_money_b0"].value,
        1.1 * sum(1_000 * (i + 1) for i in range(10)),
        rel_tol=1e-15,
    )


def test_without_a_proof_rows_stay_parked_and_compilation_refuses():
    inputs = summed_profile_fixture([TEN])
    effective, _, receipt = _resolve(inputs)
    for row in effective["target_references"]:
        assert row["metadata"]["activation_status"] == "requires_prerequisites"
        assert json.loads(row["metadata"]["reference_prerequisites_remaining"]) == [
            f"measurement:{SUMMED_MEASUREMENT}"
        ]
    assert "measurement_proofs" not in receipt
    assert validate_reference_resolution(effective, receipt) == receipt
    with pytest.raises(ValueError, match="non-executable placeholder"):
        _compile(effective, _facts(TEN))


def test_a_proof_for_another_measurement_does_not_satisfy_this_one():
    inputs = summed_profile_fixture([TEN])
    other = {
        "other": {
            "measurement_id": "other",
            "producers": {"toy.calibrate": "toy.other"},
        }
    }
    effective, _, receipt = _resolve(inputs, other)
    assert {
        row["metadata"]["activation_status"] for row in effective["target_references"]
    } == {"requires_prerequisites"}
    assert "measurement_proofs" not in receipt


def test_an_active_row_without_its_proof_in_the_receipt_refuses():
    inputs = summed_profile_fixture([TEN])
    effective, _, receipt = _resolve(inputs, proof_for())
    parked, _, parked_receipt = _resolve(inputs)
    with pytest.raises(ValueError):
        validate_reference_resolution(effective, parked_receipt)
    forged = copy.deepcopy(parked)
    for row in forged["target_references"]:
        row["metadata"]["activation_status"] = "active"
    with pytest.raises(ValueError):
        validate_reference_resolution(forged, parked_receipt)
    assert validate_reference_resolution(effective, receipt) == receipt


def _mutate(mutation):
    groups = [TEN[:5], TEN[5:]]
    base, profiles, context, intervals = summed_profile_fixture(groups)
    rows = _rows(base)
    if mutation == "money_member_count":
        rows["toy_money_b0"]["expected_member_count"] = 4
    elif mutation == "count_member_count":
        rows["toy_count_b0"]["expected_member_count"] = 6
    elif mutation == "non_identity_money":
        rows["toy_money_b0"]["value_operation"] = "sum"
    elif mutation == "mismatched_members":
        rows["toy_count_b0"]["ledger_selector"]["dimension_values"]["band"] = [
            *TEN[:4],
            "elsewhere",
        ]
    elif mutation == "overlapping_members":
        for kind in ("count", "money"):
            selector = rows[f"toy_{kind}_b1"]["ledger_selector"]
            selector["dimension_values"]["band"] = [TEN[4], *TEN[6:]]
    elif mutation == "identity_count":
        rows["toy_count_b0"]["value_operation"] = "identity"
    elif mutation == "fact_key_and_members":
        rows["toy_money_b0"]["ledger_selector"]["aggregate_fact_key"] = "toy-money-x"
    elif mutation == "repeated_member":
        for kind in ("count", "money"):
            selector = rows[f"toy_{kind}_b0"]["ledger_selector"]
            selector["dimension_values"]["band"] = [*TEN[:4], TEN[0]]
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(mutation)
    return base, profiles, context, intervals


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("money_member_count", "lists 5 members but declares expected_member_count 4"),
        ("count_member_count", "lists 5 members but declares expected_member_count 6"),
        ("non_identity_money", "untransformed source reference"),
        ("mismatched_members", "different member sets"),
        ("overlapping_members", "share a summed member"),
        ("identity_count", "value_operation 'sum'"),
        ("fact_key_and_members", "singleton fact key and a member set"),
        ("repeated_member", "repeat or are not scalars"),
    ],
)
def test_summed_rows_refuse_each_inconsistency(mutation, message):
    with pytest.raises(ValueError, match=message):
        _resolve(_mutate(mutation), proof_for())


def test_receipt_states_the_binary64_rounding_of_the_weight():
    inputs = summed_profile_fixture([TEN])
    _, _, receipt = _resolve(inputs, proof_for())
    record = receipt["factor_weight_rounding"]["toy_central"]
    exact = Fraction(record["exact_numerator"], record["exact_denominator"])
    assert record["weight"] == 1.1
    assert Fraction(record["binary64_relative_rounding"]) == (
        (Fraction(1.1) - exact) / exact
    )
    assert abs(Fraction(record["binary64_relative_rounding"])) <= Fraction(1, 2**53)


def _partition(draw_sizes, labels):
    groups, start = [], 0
    for size in draw_sizes:
        groups.append(labels[start : start + size])
        start += size
    return groups


@given(
    sizes=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=5),
    values=st.lists(
        st.integers(min_value=0, max_value=10**9), min_size=30, max_size=30
    ),
    counts=st.lists(st.integers(min_value=0, max_value=10**6), min_size=30, max_size=30),
    numerator=st.integers(min_value=500_000, max_value=2_000_000),
)
@settings(
    max_examples=25,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_compiled_coarse_rows_equal_factor_times_member_sums(
    sizes, values, counts, numerator
):
    labels = [f"m{index:02d}" for index in range(sum(sizes))]
    groups = _partition(sizes, labels)
    anchor = f"{numerator // 10_000}.{numerator % 10_000:04d}"
    inputs = summed_profile_fixture(groups, anchors={"central": anchor})
    effective, _, receipt = _resolve(inputs, proof_for())
    rational = receipt["selected_factors"]["toy_central"]["operational_factor"]
    factor = Fraction(rational["numerator"], rational["denominator"])
    assert factor == Fraction(numerator, 10**6)
    facts = _facts(labels, count=lambda i: counts[i], money=lambda i: values[i])
    specs = _compile(effective, facts)
    for band, members in enumerate(groups):
        indices = [labels.index(label) for label in members]
        assert specs[f"toy_count_b{band}"].value == sum(counts[i] for i in indices)
        exact = factor * sum(values[i] for i in indices)
        assert math.isclose(
            specs[f"toy_money_b{band}"].value, float(exact), rel_tol=1e-12, abs_tol=1e-6
        )


def _singleton_inputs(labels, anchors=None):
    """Today's path: one singleton count/money pair per native band."""
    base, profiles, context, intervals = summed_profile_fixture(
        [labels], anchors=anchors
    )
    rows = []
    for label in labels:
        for kind in ("count", "money"):
            row = summed_row(kind, f"single_{kind}_{label}", [label], indicator="i")
            row.pop("expected_member_count")
            row["filter"] = f"single_{label}"
            selector = row["ledger_selector"]
            selector["dimension_values"] = {"band": label}
            selector["aggregate_fact_key"] = f"ledger.aggregate_fact.v2:toy-{kind}-{label}"
            if kind == "count":
                row["value_operation"] = "identity"
            rows.append(row)
    base["target_references"] = rows
    transform = profiles["dimensions"]["index"]["profiles"]["central"][
        "transformations"
    ][0]
    for profile in profiles["dimensions"]["index"]["profiles"].values():
        (transform,) = profile["transformations"]
        transform["reference_names"] = [
            row["name"] for row in rows if row["name"].startswith("single_money")
        ]
        transform["count_reference_names"] = [
            row["name"] for row in rows if row["name"].startswith("single_count")
        ]
        transform["interval_table_ids"] = []
    return base, profiles, context, None


@pytest.mark.parametrize("anchor", ["105", "110", "117.1109"])
def test_summed_path_equals_the_sum_of_singleton_rows_aged_today(anchor):
    facts = _facts(TEN, money=lambda i: 123_457 * (i + 3) + 11)
    summed, _, _ = _resolve(
        summed_profile_fixture([TEN], anchors={"central": anchor}), proof_for()
    )
    singles, _, _ = _resolve(_singleton_inputs(TEN, {"central": anchor}), proof_for())
    summed_specs = _compile(summed, facts)
    single_specs = _compile(singles, facts)
    assert summed_specs["toy_count_b0"].value == sum(
        single_specs[f"single_count_{label}"].value for label in TEN
    )
    assert math.isclose(
        summed_specs["toy_money_b0"].value,
        sum(single_specs[f"single_money_{label}"].value for label in TEN),
        rel_tol=1e-13,
    )
    for label in TEN:
        operand = _rows(singles)[f"single_money_{label}"]["value_operands"][0]
        assert set(operand) == {"aggregate_fact_key", "expected_member_count", "weight"}
        assert operand["expected_member_count"] == 1
