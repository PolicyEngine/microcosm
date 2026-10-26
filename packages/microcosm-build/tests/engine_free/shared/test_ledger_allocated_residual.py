"""Country-neutral exact partitions and proportional residual allocation."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
    fact_matches_selector,
    reconcile_signed_partition,
    reference_fact_selectors,
    target_spec_from_ledger_reference,
)

DIMENSION = "census.partition_member"


def _fact(key, value, member=None, *, period=2023, growth=False, geography="01"):
    return {
        "aggregate_fact_key": f"fact:{key}",
        "value": value,
        "period": {"type": "calendar_year", "value": period},
        "geography": {"level": "region", "id": geography, "vintage": "current"},
        "entity": {"name": "person" if growth else "household"},
        "aggregation": {"method": "sum"},
        "source": {"source_name": "publisher", "source_sha256": "a" * 64},
        "source_release_key": "release:one",
        "observed_measure": {
            "source_measure_id": "population" if growth else "count",
            "source_concept": "population" if growth else "households",
        },
        "layout": {
            "record_set_id": "population"
            if growth
            else "total"
            if member is None
            else "partition"
        },
        "dimensions": {} if member is None else {DIMENSION: member},
        "universe_constraints": {"domain": "people" if growth else "households"},
    }


def _selector(fact):
    result = {
        "aggregate_fact_key": fact["aggregate_fact_key"],
        "source_name": fact["source"]["source_name"],
        "source_measure_id": fact["observed_measure"]["source_measure_id"],
        "source_concept": fact["observed_measure"]["source_concept"],
        "entity_name": fact["entity"]["name"],
        "record_set_id": fact["layout"]["record_set_id"],
        "period_type": fact["period"]["type"],
        "period_value": fact["period"]["value"],
        "geography_level": fact["geography"]["level"],
        "geography_id": fact["geography"]["id"],
        "geography_vintage": fact["geography"]["vintage"],
        "dimensions": list(fact["dimensions"]),
    }
    if fact["dimensions"]:
        result["dimension_values"] = dict(fact["dimensions"])
    return result


def _case(
    counts=(72, 87, 24, 24, 9, 6, 3, 0),
    residual=15,
    recipients=None,
    group=(1,),
    growth=True,
):
    atoms = [
        _fact(f"atom:{index}", value, index) for index, value in enumerate(counts, 1)
    ]
    total = _fact("total", math.fsum(counts) + residual)
    recipients = tuple(range(1, len(counts) + 1)) if recipients is None else recipients
    selector = _selector(atoms[group[0] - 1])
    if len(group) > 1:
        selector.pop("aggregate_fact_key")
        selector["dimension_values"] = {DIMENSION: list(group)}
    operands = [
        {"role": "base", "expected_member_count": len(group)},
        {"role": "total", **_selector(total)},
        {
            "role": "partition",
            "member_selectors": [_selector(fact) for fact in atoms],
            "allocation_member_keys": [
                atoms[index - 1]["aggregate_fact_key"] for index in recipients
            ],
        },
    ]
    facts = [*atoms, total]
    if growth:
        for role, period, value in (
            ("numerator", 2025, 109),
            ("denominator", 2023, 100),
        ):
            fact = _fact(role, value, period=period, growth=True)
            operands.append({"role": role, **_selector(fact)})
            facts.append(fact)
    reference = LedgerTargetReference(
        name="allocated group",
        ledger_selector=selector,
        value_operation="allocated_residual_by_ratio",
        value_operands=tuple(operands),
        entity="household",
        measure="indicator",
        period=2025 if growth else 2023,
        metadata={
            "allocation_ruling": "reviewed-convention",
            "census_residual_allocation": "proportional",
        },
    )
    return facts, reference


def _compile(facts, reference, direct=False):
    if direct:
        return target_spec_from_ledger_reference(tuple(facts), reference)
    return compile_ledger_target_references(
        facts, [reference], country="fixture"
    ).specs[0]


@pytest.mark.parametrize("growth", [False, True])
def test_mar_closes_preserves_shares_zero_leaves_and_applies_growth_once(growth):
    counts = (72, 87, 24, 24, 9, 6, 3, 0)
    values = [
        _compile(*_case(counts, group=(index,), growth=growth)).value
        for index in range(1, 9)
    ]
    ratio = 109 / 100 if growth else 1
    assert math.isclose(math.fsum(values), 240 * ratio, rel_tol=1e-12, abs_tol=1e-7)
    assert values[0] == pytest.approx(76.8 * ratio, rel=1e-12, abs=1e-7)
    assert values[-1] == 0
    for i in range(8):
        for j in range(8):
            assert values[i] * counts[j] == pytest.approx(
                values[j] * counts[i], rel=1e-12, abs=1e-7
            )


def test_sensitivity_exact_recipients_and_increment_ratio():
    counts = (72, 87, 24, 24, 9, 6, 3, 0)
    small = [
        _compile(*_case(counts, recipients=(1, 2), group=(i,), growth=False)).value
        for i in range(1, 9)
    ]
    mar = [
        _compile(*_case(counts, group=(i,), growth=False)).value for i in range(1, 9)
    ]
    assert small[:2] == pytest.approx([78.79245283018868, 95.20754716981132])
    assert small[2:] == list(counts[2:])
    assert all(small[i] < mar[i] for i in range(2, 7))
    assert math.isclose(math.fsum(small), 240, rel_tol=1e-12, abs_tol=1e-7)
    assert (small[0] - counts[0]) * counts[1] == pytest.approx(
        (small[1] - counts[1]) * counts[0]
    )


def test_grouping_preserves_the_third_atomic_member():
    facts, reference = _case((41, 19, 7), 3, group=(1, 3))
    value = _compile(facts, reference).value
    assert value == pytest.approx(
        float(Fraction(48) * Fraction(70, 67) * Fraction(109, 100))
    )
    singleton_sum = math.fsum(
        _compile(*_case((41, 19, 7), 3, group=(i,))).value for i in (1, 3)
    )
    assert value == pytest.approx(singleton_sum)


def test_receipt_labels_keys_and_preserves_base_geography_and_raw_facts():
    facts, reference = _case()
    before = deepcopy(facts)
    spec = _compile(facts, reference)
    receipt = json.loads(spec.metadata["ledger_allocation_receipt"])
    assert receipt["operation"] == "allocated_residual_by_ratio"
    assert receipt["ruling"] == "reviewed-convention"
    assert receipt["selected_profile"] == "proportional"
    assert receipt["T"] == 240 and receipt["L"] == 225 and receipt["R"] == 15
    assert receipt["B"] == 72 and receipt["A"] == 225 and receipt["W"] == 72
    assert receipt["allocated_increment"] == pytest.approx(4.8)
    assert receipt["Q"] == 1.09 and receipt["N"] == 109 and receipt["D"] == 100
    assert receipt["zero_residual"] is False
    assert receipt["source_period"] == 2023 and receipt["target_period"] == 2025
    assert receipt["final_target"] == spec.value
    assert set(receipt["member_digests"]) == {"base", "partition", "recipients"}
    assert spec.metadata["ledger_geography_id"] == "01"
    assert facts == before


def test_prefilter_keeps_every_auxiliary_selector_and_unrelated_rows_are_inert():
    facts, reference = _case(group=(1, 3))
    unrelated = _fact("unrelated", 1_000_000, 1, geography="02")
    selectors = reference_fact_selectors(reference)
    filtered = [
        fact
        for fact in [*reversed(facts), unrelated]
        if any(fact_matches_selector(fact, selector) for selector in selectors)
    ]
    assert len(filtered) == len(facts)
    assert _compile(filtered, reference).value == _compile(facts, reference).value
    operands = deepcopy(reference.value_operands)
    operands[2]["member_selectors"].reverse()
    operands[2]["allocation_member_keys"].reverse()
    reordered = replace(reference, value_operands=operands)
    assert (
        _compile(filtered, reordered).metadata["ledger_allocation_receipt"]
        == _compile(facts, reference).metadata["ledger_allocation_receipt"]
    )


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "substituted",
        "string",
        "float",
        "wrong_universe",
        "wrong_snapshot",
        "wrong_period",
        "negative",
        "nonfinite",
    ],
)
def test_refuses_invalid_atomic_partitions_on_both_compilation_paths(change, direct):
    facts, reference = _case((41, 19, 7), 3, group=(1, 3))
    if change == "missing":
        facts.pop(2)
    elif change == "duplicate":
        facts.append(deepcopy(facts[2]))
    elif change == "substituted":
        original_value = facts[2]["value"]
        facts[2] = deepcopy(facts[0])
        facts[2]["value"] = original_value
        facts[2]["aggregate_fact_key"] = "fact:atom:3"
    elif change in {"string", "float"}:
        facts[2]["dimensions"][DIMENSION] = "3" if change == "string" else 3.0
    elif change == "wrong_universe":
        facts[3]["universe_constraints"]["domain"] = "families"
    elif change == "wrong_snapshot":
        facts[2]["source_release_key"] = "release:other"
    elif change == "wrong_period":
        facts[2]["period"]["value"] = 2024
    else:
        facts[2]["value"] = -1 if change == "negative" else math.nan
    with pytest.raises(ValueError):
        _compile(facts, reference, direct)


def test_indexed_catalog_resolution_preserves_duplicate_occurrences_outside_base():
    facts, reference = _case((41, 19, 7), residual=3, group=(1,), growth=False)
    facts.append(facts[2])
    with pytest.raises(ValueError, match="selector must resolve exactly once, got 2"):
        _compile(facts, reference)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "predicate", ["range", "different_equality", "wrong_typed_equality", "other_role"]
)
def test_partition_dimension_filters_cannot_change_the_total_universe(
    predicate, direct
):
    facts, reference = _case((0, 3, 5), residual=2, group=(2,), growth=False)
    for code, fact in enumerate(facts[:3], 1):
        cell = {
            "variable": DIMENSION,
            "operator": "==",
            "value": code,
            "role": "filter",
        }
        extra = dict(cell)
        if predicate == "range":
            extra.update(operator=">=", value=2)
        elif predicate == "different_equality":
            extra["value"] = code + 1
        elif predicate == "wrong_typed_equality":
            extra["value"] = float(code)
        else:
            extra["role"] = "measure"
        fact["universe_constraints"]["constraints"] = [cell, extra]
    with pytest.raises(ValueError, match="count universe"):
        _compile(facts, reference, direct)


@pytest.mark.parametrize("direct", [False, True])
def test_only_exact_typed_category_equalities_are_removed_from_universe(direct):
    facts, reference = _case((0, 3, 5), residual=2, group=(2,), growth=False)
    for code, fact in enumerate(facts[:3], 1):
        fact["universe_constraints"]["constraints"] = [
            {"variable": DIMENSION, "operator": "==", "value": code, "role": "filter"}
        ]
    assert _compile(facts, reference, direct).value == pytest.approx(3.75)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "roster,count",
    [([1, 3, 4], 2), ([1, "3"], 1), ([1, 3, 4], 3), ([1, "3"], 2), ([1, 3], 1)],
)
def test_declared_base_roster_must_match_its_count_and_partition(roster, count, direct):
    facts, reference = _case((41, 19, 7), residual=3, group=(1, 3))
    # Simulate a direct caller supplying an already-created, subsequently
    # altered declaration; compilation must repeat the constructor's guards.
    reference.ledger_selector["dimension_values"][DIMENSION] = roster
    reference.value_operands[0]["expected_member_count"] = count
    with pytest.raises(ValueError, match="typed roster"):
        _compile(facts, reference, direct)


def test_constructor_rejects_base_roster_members_outside_partition():
    _, reference = _case((41, 19, 7), residual=3, group=(1, 3))
    selector = deepcopy(reference.ledger_selector)
    selector["dimension_values"][DIMENSION] = [1, "3"]
    with pytest.raises(ValueError, match="typed roster"):
        replace(reference, ledger_selector=selector)


@pytest.mark.parametrize("direct", [False, True])
def test_negative_residual_is_refused_without_clipping(direct):
    with pytest.raises(ValueError, match="negative.*residual"):
        _compile(*_case(residual=-1), direct)


@pytest.mark.parametrize("direct", [False, True])
def test_zero_residual_with_zero_recipient_mass_is_identity(direct):
    facts, reference = _case((0, 0, 7), residual=0, recipients=(1, 2), group=(3,))
    spec = _compile(facts, reference, direct)
    assert spec.value == pytest.approx(7 * 109 / 100)
    assert (
        json.loads(spec.metadata["ledger_allocation_receipt"])["zero_residual"] is True
    )


def test_positive_residual_with_zero_recipient_mass_refuses():
    with pytest.raises(ValueError, match="positive recipient mass"):
        _compile(*_case((0, 0, 7), recipients=(1, 2)))


@pytest.mark.parametrize("direct", [False, True])
def test_future_count_period_cannot_bypass_consumer_period_policy(direct):
    facts, reference = _case(growth=False)
    reference = replace(reference, period=2022)
    with pytest.raises(ValueError, match="at or before target period"):
        _compile(facts, reference, direct)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "marker",
    [
        {},
        {"reference_profiles_resolved": "1"},
        {"reference_selection_sha256": "a" * 64},
    ],
)
def test_raw_reference_profile_requirement_cannot_be_promoted_by_status(direct, marker):
    facts, reference = _case()
    reference = replace(
        reference,
        metadata={
            "activation_status": "active",
            "required_reference_profile": "allocation",
            **marker,
        },
    )
    with pytest.raises(ValueError, match="requires reference profile resolution"):
        _compile(facts, reference, direct)


def test_total_and_atoms_require_same_typed_shared_dimension_values():
    _, reference = _case()
    operands = deepcopy(reference.value_operands)
    operands[1]["dimensions"] = ["sex"]
    operands[1]["dimension_values"] = {"sex": "F"}
    for selector in operands[2]["member_selectors"]:
        selector["dimensions"].append("sex")
        selector["dimension_values"]["sex"] = "M"
    with pytest.raises(ValueError, match="shared total dimension values"):
        replace(reference, value_operands=operands)


@pytest.mark.parametrize(
    "change",
    [
        "unpaired",
        "outside_recipient",
        "total_in_partition",
        "duplicate_partition",
        "duplicate_cell_other_record_set",
        "missing_pin",
    ],
)
def test_reference_contract_refuses_unpaired_or_malformed_operands(change):
    _, reference = _case()
    operands = deepcopy(reference.value_operands)
    if change == "unpaired":
        operands = operands[:-1]
    elif change == "outside_recipient":
        operands[2]["allocation_member_keys"].append("fact:outside")
    elif change == "total_in_partition":
        _allocation_total = {
            key: value for key, value in operands[1].items() if key != "role"
        }
        operands[2]["member_selectors"].append(_allocation_total)
        operands[2]["allocation_member_keys"].append(
            _allocation_total["aggregate_fact_key"]
        )
    elif change == "duplicate_partition":
        operands[2]["member_selectors"][1] = deepcopy(
            operands[2]["member_selectors"][0]
        )
    elif change == "duplicate_cell_other_record_set":
        selector = operands[2]["member_selectors"][1]
        selector["record_set_id"] = "another.table"
        selector["dimension_values"] = deepcopy(
            operands[2]["member_selectors"][0]["dimension_values"]
        )
    else:
        del operands[2]["member_selectors"][0]["record_set_id"]
    with pytest.raises(ValueError):
        replace(reference, value_operands=operands)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "zero_denominator",
        "negative_denominator",
        "low_ratio",
        "high_ratio",
        "wrong_series",
        "wrong_period",
        "undeclared_proxy",
    ],
)
def test_growth_operand_refusals_on_both_paths(change, direct):
    facts, reference = _case()
    if change == "zero_denominator":
        facts[-1]["value"] = 0
    elif change == "negative_denominator":
        facts[-1]["value"] = -1
    elif change == "low_ratio":
        facts[-2]["value"] = 50
    elif change == "high_ratio":
        facts[-2]["value"] = 201
    else:
        operands = deepcopy(reference.value_operands)
        if change == "wrong_series":
            facts[-1]["entity"]["name"] = "different_entity"
            operands[-1]["entity_name"] = "different_entity"
        elif change == "wrong_period":
            facts[-1]["period"]["value"] = 2024
            operands[-1]["period_value"] = 2024
        else:
            for fact, operand in zip(facts[-2:], operands[-2:], strict=True):
                fact["geography"]["id"] = "all"
                operand["geography_id"] = "all"
        reference = replace(reference, value_operands=operands)
    with pytest.raises(ValueError):
        _compile(facts, reference, direct)


def test_explicit_proxy_preserves_output_geography():
    facts, reference = _case()
    operands = deepcopy(reference.value_operands)
    for fact, operand in zip(facts[-2:], operands[-2:], strict=True):
        fact["geography"]["id"] = "all"
        operand["geography_id"] = "all"
    reference = replace(
        reference,
        value_operands=operands,
        metadata={"growth_geography_proxy": "declared wider-area growth"},
    )
    spec = _compile(facts, reference)
    assert spec.metadata["ledger_geography_id"] == "01"
    assert json.loads(spec.metadata["ledger_allocation_receipt"])["Q"] == 1.09


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("denominator_role", ["total", "atom"])
def test_growth_roles_may_reuse_total_or_atom_without_double_counting(
    direct, denominator_role
):
    facts, reference = _case(growth=False)
    denominator = facts[-1] if denominator_role == "total" else facts[0]
    numerator = deepcopy(denominator)
    numerator["aggregate_fact_key"] = "fact:later_count"
    numerator["period"]["value"] = 2025
    numerator["value"] *= 1.1
    facts.append(numerator)
    reference = replace(
        reference,
        period=2025,
        value_operands=(
            *reference.value_operands,
            {"role": "numerator", **_selector(numerator)},
            {"role": "denominator", **_selector(denominator)},
        ),
    )
    spec = _compile(facts, reference, direct)
    assert spec.value == pytest.approx(76.8 * 1.1)
    receipt = json.loads(spec.metadata["ledger_allocation_receipt"])
    assert receipt["L"] == 225 and receipt["T"] == 240
    assert receipt["fact_keys"]["denominator"] == denominator["aggregate_fact_key"]
    keys = json.loads(spec.metadata["ledger_member_fact_keys"])
    assert len(keys) == len(set(keys)) == len(facts)
    # The same declared role overlap cannot excuse duplicate source atoms.
    with pytest.raises(ValueError, match="exactly once|multiple Ledger facts"):
        _compile([*facts, deepcopy(facts[0])], reference, direct)


@pytest.mark.parametrize("direct", [False, True])
def test_distinct_fact_reusing_an_atom_key_cannot_replace_the_atom(direct):
    # Review counterexample: a growth numerator that reuses atom 1's key
    # passes its own selector, so key-only deduplication would swap the
    # 72-household atom for the 66-person growth fact.
    facts, reference = _case(group=(1, 2))
    facts[-2]["value"] = 66
    facts[-1]["value"] = 60
    assert _compile(facts, reference, direct).value == pytest.approx(186.56)
    facts[-2]["aggregate_fact_key"] = facts[0]["aggregate_fact_key"]
    operands = deepcopy(list(reference.value_operands))
    operands[-2]["aggregate_fact_key"] = facts[0]["aggregate_fact_key"]
    reference = replace(reference, value_operands=tuple(operands))
    match = "distinct facts share" if direct else "distinct facts share|exactly once"
    with pytest.raises(ValueError, match=match):
        _compile(facts, reference, direct)


@pytest.mark.parametrize("direct", [False, True])
def test_distinct_base_fact_reusing_an_atom_key_cannot_replace_the_atom(direct):
    # Delta-review counterexample: a base selector pinned to a different
    # record that reuses atom 1's key and carries 66 instead of 72 would
    # otherwise displace the validated atom (76.8 became 72.33).
    facts, reference = _case(growth=False)
    assert _compile(facts, reference, direct).value == pytest.approx(76.8)
    facts[0]["source_record_id"] = "original-atom-record"
    alternate = deepcopy(facts[0])
    alternate["source_record_id"] = "different-base-record"
    alternate["value"] = 66
    facts.append(alternate)
    selector = deepcopy(reference.ledger_selector)
    selector["source_record_id"] = alternate["source_record_id"]
    operands = deepcopy(list(reference.value_operands))
    operands[2]["member_selectors"][0]["source_record_id"] = facts[0][
        "source_record_id"
    ]
    reference = replace(
        reference, ledger_selector=selector, value_operands=tuple(operands)
    )
    match = "distinct facts share" if direct else "distinct facts share|exactly once"
    with pytest.raises(ValueError, match=match):
        _compile(facts, reference, direct)


# Cold local-constant discovery on the shared build machine can dominate data
# generation; retain every generated example and numerical assertion.
@settings(
    max_examples=100,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    counts=st.lists(st.integers(0, 100_000), min_size=1, max_size=8),
    residual=st.integers(0, 10_000),
    recipient_mask=st.integers(1, 255),
    group_mask=st.integers(1, 255),
    numerator=st.integers(51, 200),
)
# Independently scaled totals leave a rounding residual of each sign.
@example(counts=[1, 2], residual=0, recipient_mask=3, group_mask=1, numerator=55)
@example(counts=[1, 2], residual=0, recipient_mask=3, group_mask=1, numerator=58)
def test_exact_rational_oracle_for_partitions_recipients_groups_and_growth(
    counts, residual, recipient_mask, group_mask, numerator
):
    recipients = tuple(
        i for i in range(1, len(counts) + 1) if recipient_mask & (1 << (i - 1))
    ) or (1,)
    group = tuple(
        i for i in range(1, len(counts) + 1) if group_mask & (1 << (i - 1))
    ) or (1,)
    mass = sum(counts[i - 1] for i in recipients)
    facts, reference = _case(counts, residual, recipients, group)
    facts[-2]["value"] = numerator
    if residual and not mass:
        with pytest.raises(ValueError, match="positive recipient mass"):
            _compile(facts, reference)
        return
    base = sum(counts[i - 1] for i in group)
    weight = sum(counts[i - 1] for i in group if i in recipients)
    allocated = Fraction(base) + (Fraction(residual * weight, mass) if residual else 0)
    expected = allocated * Fraction(numerator, 100)
    value = _compile(
        [*reversed(facts), _fact("other", 999, 1, geography="02")], reference
    ).value
    assert math.isclose(value, float(expected), rel_tol=1e-12, abs_tol=1e-7)
    # The generated group and its complement form an exact output partition.
    complement = tuple(i for i in range(1, len(counts) + 1) if i not in group)
    partition_values = [value]
    if complement:
        other_facts, other_reference = _case(counts, residual, recipients, complement)
        other_facts[-2]["value"] = numerator
        other_value = _compile(other_facts, other_reference).value
        other_base = sum(counts[i - 1] for i in complement)
        other_weight = sum(counts[i - 1] for i in complement if i in recipients)
        other_oracle = (
            Fraction(other_base)
            + (Fraction(residual * other_weight, mass) if residual else 0)
        ) * Fraction(numerator, 100)
        assert math.isclose(
            other_value, float(other_oracle), rel_tol=1e-12, abs_tol=1e-7
        )
        partition_values.append(other_value)
    assert math.isclose(
        math.fsum(partition_values),
        float(Fraction(sum(counts) + residual) * Fraction(numerator, 100)),
        rel_tol=1e-12,
        abs_tol=1e-7,
    )
    # Scaling the Census total and leaves commutes with allocation in exact
    # arithmetic. The scaled total is computed independently as q*T, not from
    # the scaled leaves, and the separate growth pair is omitted so scaling
    # happens once. In floating point a zero residual can round to a tiny
    # negative one (counts 1, 2 at q=0.55); that must still refuse, never clip.
    q = Fraction(numerator, 100)
    scaled_counts = [float(x * q) for x in counts]
    scaled_facts, scaled_reference = _case(
        scaled_counts, 0, recipients, group, growth=False
    )
    scaled_facts[len(counts)]["value"] = float((sum(counts) + residual) * q)
    scaled_residual = scaled_facts[len(counts)]["value"] - math.fsum(scaled_counts)
    if scaled_residual < 0:
        with pytest.raises(ValueError, match="negative.*residual"):
            _compile(scaled_facts, scaled_reference)
    elif scaled_residual > 0 and not mass:
        with pytest.raises(ValueError, match="positive recipient mass"):
            _compile(scaled_facts, scaled_reference)
    else:
        assert math.isclose(
            _compile(scaled_facts, scaled_reference).value,
            value,
            rel_tol=1e-12,
            abs_tol=1e-7,
        )


@settings(
    max_examples=50,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    counts=st.lists(st.integers(0, 100_000), min_size=2, max_size=8),
    residual=st.integers(0, 10_000),
    change=st.sampled_from(
        (
            "missing",
            "duplicate",
            "substitution",
            "type",
            "recipient_outside",
            "total_universe",
            "period",
            "vintage",
            "snapshot",
            "negative_residual",
            "nonfinite",
            "unpaired_growth",
            "wrong_growth_series",
            "growth_bounds",
        )
    ),
    direct=st.booleans(),
)
def test_generated_invalid_partition_rosters_refuse(counts, residual, change, direct):
    facts, reference = _case(counts, residual)
    with pytest.raises(ValueError):
        if change == "missing":
            facts.pop(1)
        elif change == "duplicate":
            facts.insert(1, deepcopy(facts[1]))
        elif change == "substitution":
            facts[1]["dimensions"][DIMENSION] = 1
        elif change == "type":
            facts[1]["dimensions"][DIMENSION] = 2.0
        elif change == "recipient_outside":
            operands = deepcopy(reference.value_operands)
            operands[2]["allocation_member_keys"] = ["fact:outside"]
            reference = replace(reference, value_operands=operands)
        elif change == "total_universe":
            facts[len(counts)]["universe_constraints"]["domain"] = "other_universe"
        elif change == "period":
            facts[1]["period"]["value"] = 2024
        elif change == "vintage":
            facts[1]["geography"]["vintage"] = "another_vintage"
        elif change == "snapshot":
            facts[1]["source_release_key"] = "release:other"
        elif change == "negative_residual":
            if not sum(counts):
                facts[0]["value"] = 1
            facts[len(counts)]["value"] = max(0, sum(counts) - 1)
        elif change == "nonfinite":
            facts[1]["value"] = math.inf
        elif change == "unpaired_growth":
            reference = replace(reference, value_operands=reference.value_operands[:-1])
        elif change == "wrong_growth_series":
            operands = deepcopy(reference.value_operands)
            operands[-1]["entity_name"] = "another_entity"
            facts[-1]["entity"]["name"] = "another_entity"
            reference = replace(reference, value_operands=operands)
        else:
            facts[-2]["value"] = 301
        _compile(facts, reference, direct)


SIGNED_POLICY = {
    "kind": "signed_proportional_reconciliation",
    "max_abs": 3,
    "max_relative": "0.001",
    "combine": "either",
    "ruling": "d1932",
    "disclosure": "Independently rounded published counts; a modelling adjustment.",
}


def _signed_case(counts=(41, 19, 7), residual=3, *, sensitivity=False, growth=False):
    facts, reference = _case(counts, residual, growth=growth)
    operands = deepcopy(reference.value_operands)
    partition = operands[2]
    partition["discrepancy_policy"] = deepcopy(SIGNED_POLICY)
    if sensitivity:
        partition["positive_allocation_member_keys"] = ["fact:atom:3"]
    return facts, replace(
        reference,
        value_operands=operands,
        metadata={
            "allocation_ruling": "d1932",
            "family_discrepancy": "family_discrepancy_positive_to_type3"
            if sensitivity
            else "central",
        },
    )


def _fraction(value):
    return Fraction(value["numerator"], value["denominator"])


def _signed_receipt(*args, **kwargs):
    facts, reference = _signed_case(*args, **kwargs)
    return json.loads(_compile(facts, reference).metadata["ledger_allocation_receipt"])


def test_signed_region_99_passes_absolute_bound_with_either_combination():
    facts, reference = _signed_case((72, 60, 15), residual=3)
    operands = deepcopy(reference.value_operands)
    selector = deepcopy(reference.ledger_selector)
    selector["geography_id"] = "99"
    for fact in facts:
        fact["geography"]["id"] = "99"
    operands[1]["geography_id"] = "99"
    for member in operands[2]["member_selectors"]:
        member["geography_id"] = "99"
    reference = replace(reference, ledger_selector=selector, value_operands=operands)
    receipt = json.loads(
        _compile(facts, reference).metadata["ledger_allocation_receipt"]
    )
    assert receipt["region"] == "99"
    assert receipt["bounds_passed"] == ["absolute"]
    assert _fraction(receipt["L"]) == 147
    assert _fraction(receipt["R"]) == 3
    assert sum(map(_fraction, receipt["reconciled"].values())) == 150


def test_signed_relative_only_bound_admits_more_than_three():
    receipt = _signed_receipt((5000, 4000, 1000), residual=4)
    assert receipt["bounds_passed"] == ["relative"]
    assert sum(map(_fraction, receipt["reconciled"].values())) == 10004


@pytest.mark.parametrize("residual", [-4, 4])
def test_signed_relative_bound_is_inclusive(residual):
    # |R| / L == max_relative exactly: 4 / 4000 is 0.001, past max_abs=3.
    receipt = _signed_receipt((2000, 1500, 500), residual=residual)
    assert receipt["bounds_passed"] == ["relative"]
    assert sum(map(_fraction, receipt["reconciled"].values())) == 4000 + residual
    with pytest.raises(ValueError, match=r"R=5, L=4000.*exceeds either bound"):
        _signed_receipt((2000, 1500, 500), residual=5)


def test_signed_relative_bound_compares_exact_rationals():
    # Fraction("0.1") is exactly 1/10; Fraction(float("0.1")) exceeds it by
    # about 5.6e-18, so a gap 1e-18 past the bound tells the two apart.
    policy = {**SIGNED_POLICY, "max_abs": 0, "max_relative": "0.1"}
    partition = {"a": Fraction(6), "b": Fraction(4)}
    _, _, bounds = reconcile_signed_partition(
        partition, Fraction(11), policy, region="01"
    )
    assert bounds == ("relative",)
    with pytest.raises(ValueError, match="exceeds either bound"):
        reconcile_signed_partition(
            partition, Fraction(11) + Fraction(1, 10**18), policy, region="01"
        )


@pytest.mark.parametrize("value", [41.5, "41.5", "83/2"])
def test_signed_count_reader_refuses_non_integer_counts(value):
    facts, reference = _signed_case()
    facts[0]["value"] = value
    with pytest.raises(ValueError, match="exact nonnegative integer count"):
        _compile(facts, reference)


@pytest.mark.parametrize("residual", [-4, 4])
def test_signed_outside_either_bound_refuses_and_names_region_gap_leaves(residual):
    with pytest.raises(
        ValueError,
        match=rf"region 01: R={residual}, L=100.*max_abs=3.*max_relative=0.001",
    ):
        _signed_receipt((30, 40, 30), residual=residual)


def test_signed_zero_partition_nonzero_total_refuses():
    with pytest.raises(ValueError, match="region 01: R=1, L=0.*L=0"):
        _signed_receipt((0, 0, 0), residual=1)


def test_signed_zero_partition_and_total_close_to_zero():
    receipt = _signed_receipt((0, 0, 0), residual=0)
    assert all(_fraction(value) == 0 for value in receipt["reconciled"].values())


@pytest.mark.parametrize(
    "key,value",
    [
        ("unknown", True),
        ("kind", "unknown"),
        ("max_abs", -1),
        ("max_abs", True),
        ("max_relative", 0.001),
        ("max_relative", "1e-3"),
        ("max_relative", "NaN"),
        ("max_relative", "Infinity"),
        ("max_relative", "1/1000"),
        ("max_relative", "-0.001"),
        ("max_relative", ""),
        ("max_relative", True),
        ("combine", "both"),
        ("ruling", ""),
        ("disclosure", ""),
    ],
)
def test_signed_policy_closed_keys_and_exact_decimal_refusals(key, value):
    _, reference = _signed_case()
    operands = deepcopy(reference.value_operands)
    operands[2]["discrepancy_policy"][key] = value
    with pytest.raises(ValueError):
        replace(reference, value_operands=operands)


@pytest.mark.parametrize(
    "change",
    [
        "subset_central",
        "positive_without_policy",
        "positive_outside",
        "multiple_positive",
        "positive_null",
    ],
)
def test_signed_recipient_contract_refuses_incompatible_declarations(change):
    _, reference = _signed_case(sensitivity=True)
    operands = deepcopy(reference.value_operands)
    if change == "subset_central":
        operands[2]["allocation_member_keys"] = ["fact:atom:3"]
    elif change == "positive_without_policy":
        del operands[2]["discrepancy_policy"]
    elif change == "positive_outside":
        operands[2]["positive_allocation_member_keys"] = ["fact:outside"]
    elif change == "positive_null":
        operands[2]["positive_allocation_member_keys"] = None
    else:
        operands[2]["positive_allocation_member_keys"] = ["fact:atom:2", "fact:atom:3"]
    with pytest.raises(ValueError):
        replace(reference, value_operands=operands)


def test_signed_negative_discrepancy_scales_down_without_clipping():
    receipt = _signed_receipt(residual=-3)
    assert _fraction(receipt["R"]) == -3
    assert _fraction(receipt["reconciled"]["fact:atom:1"]) == Fraction(41 * 64, 67)
    assert all(_fraction(increment) < 0 for increment in receipt["increments"].values())


@pytest.mark.parametrize("residual", [-3, 0])
def test_signed_sensitivity_on_nonpositive_discrepancy_matches_central(residual):
    central = _signed_receipt(residual=residual)
    sensitivity = _signed_receipt(residual=residual, sensitivity=True)
    assert sensitivity["reconciled"] == central["reconciled"]
    assert sensitivity["increments"] == central["increments"]
    assert sensitivity["fact_keys"]["recipients"] == central["fact_keys"]["recipients"]


def test_signed_positive_sensitivity_assigns_all_to_zero_mass_type_three():
    receipt = _signed_receipt((1, 2, 0), residual=3, sensitivity=True)
    assert {key: _fraction(value) for key, value in receipt["increments"].items()} == {
        "fact:atom:1": 0,
        "fact:atom:2": 0,
        "fact:atom:3": 3,
    }
    assert receipt["fact_keys"]["recipients"] == ["fact:atom:3"]
    assert sum(map(_fraction, receipt["reconciled"].values())) == 6


def test_signed_receipt_is_exact_complete_and_byte_deterministic():
    facts, reference = _signed_case(growth=True, sensitivity=True)
    first = _compile(facts, reference).metadata["ledger_allocation_receipt"]
    operands = deepcopy(reference.value_operands)
    operands[2]["member_selectors"].reverse()
    operands[2]["allocation_member_keys"].reverse()
    second = _compile(
        list(reversed(facts)), replace(reference, value_operands=operands)
    ).metadata["ledger_allocation_receipt"]
    assert first == second
    receipt = json.loads(first)
    assert receipt["policy"] == SIGNED_POLICY
    assert receipt["ruling"] == "d1932"
    assert receipt["x"] == {
        f"fact:atom:{i}": {"numerator": value, "denominator": 1}
        for i, value in enumerate((41, 19, 7), 1)
    }
    assert _fraction(receipt["Q"]) == Fraction(109, 100)
    assert _fraction(receipt["growth_operands"]["numerator"]["value"]) == 109
    assert _fraction(receipt["growth_operands"]["denominator"]["value"]) == 100
    assert set(receipt["fact_keys"]) == {
        "base",
        "partition",
        "recipients",
        "total",
        "numerator",
        "denominator",
    }
    assert _fraction(receipt["final_target"]) == Fraction(41 * 109, 100)


@pytest.mark.parametrize("direct", [False, True])
def test_legacy_policy_absent_receipt_is_byte_identical(direct):
    import hashlib

    facts, reference = _case()
    receipt = _compile(facts, reference, direct).metadata["ledger_allocation_receipt"]
    assert (
        hashlib.sha256(receipt.encode()).hexdigest()
        == "98efc6e6578dcc681d25fd8fcc5811c0f2f595f9f347e5aa71c7fe43caba2307"
    )


@st.composite
def _admissible_signed_counts(draw):
    counts = tuple(draw(st.lists(st.integers(0, 100_000), min_size=3, max_size=3)))
    leaves = sum(counts)
    if not leaves:
        return counts, 0
    bound = max(3, leaves // 1000)
    return counts, draw(st.integers(-min(leaves, bound), bound))


@settings(
    max_examples=80,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(case=_admissible_signed_counts())
def test_signed_exact_closure_nonnegativity_and_preserved_shares(case):
    from microcosm.build.ledger_targets import reconcile_signed_partition

    counts, residual = case
    raw = {str(i): Fraction(value) for i, value in enumerate(counts, 1)}
    adjusted, increments, _ = reconcile_signed_partition(
        raw, Fraction(sum(counts) + residual), SIGNED_POLICY, region="generated"
    )
    assert all(
        isinstance(value, Fraction) and value >= 0 for value in adjusted.values()
    )
    assert sum(adjusted.values(), Fraction()) == sum(counts) + residual
    assert sum(increments.values(), Fraction()) == residual
    for left in raw:
        for right in raw:
            assert adjusted[left] * raw[right] == adjusted[right] * raw[left]


@settings(
    max_examples=80,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(case=_admissible_signed_counts())
def test_signed_sensitivity_property_positive_only_type_three_and_exact_closure(case):
    from microcosm.build.ledger_targets import reconcile_signed_partition

    counts, residual = case
    raw = {str(i): Fraction(value) for i, value in enumerate(counts, 1)}
    total = Fraction(sum(counts) + residual)
    central, _, _ = reconcile_signed_partition(
        raw, total, SIGNED_POLICY, region="generated"
    )
    adjusted, increments, _ = reconcile_signed_partition(
        raw, total, SIGNED_POLICY, region="generated", positive_recipient_keys=("3",)
    )
    assert sum(adjusted.values(), Fraction()) == total
    assert all(
        isinstance(value, Fraction) and value >= 0 for value in adjusted.values()
    )
    if residual > 0:
        assert increments == {"1": 0, "2": 0, "3": residual}
        assert adjusted["1"] == raw["1"] and adjusted["2"] == raw["2"]
    else:
        assert adjusted == central


@settings(
    max_examples=50,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    case=_admissible_signed_counts(),
    numerator=st.integers(51, 200),
    sensitivity=st.booleans(),
)
def test_signed_compiled_erp_bridge_scales_exact_closure_once(
    case, numerator, sensitivity
):
    counts, residual = case
    facts, reference = _signed_case(
        counts, residual, growth=True, sensitivity=sensitivity
    )
    facts[-2]["value"] = numerator
    receipt = json.loads(
        _compile(facts, reference).metadata["ledger_allocation_receipt"]
    )
    ratio = Fraction(numerator, 100)
    assert _fraction(receipt["Q"]) == ratio
    assert (
        sum(map(_fraction, receipt["final_targets"].values()), Fraction())
        == Fraction(sum(counts) + residual) * ratio
    )
    assert _fraction(receipt["final_target"]) == _fraction(
        receipt["final_targets"]["fact:atom:1"]
    )


@settings(
    max_examples=30,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    cases=st.lists(_admissible_signed_counts(), min_size=2, max_size=4),
    ordering=st.data(),
)
def test_signed_permuting_regions_and_fact_order_preserves_every_value(cases, ordering):
    facts, references = [], []
    for index, (counts, residual) in enumerate(cases):
        members, reference = _signed_case(counts, residual, growth=True)
        region = str(index)
        selector = deepcopy(reference.ledger_selector)
        selector["geography_id"] = region
        selector["aggregate_fact_key"] += f":{region}"
        operands = deepcopy(reference.value_operands)
        for member in members:
            member["geography"]["id"] = region
            member["aggregate_fact_key"] += f":{region}"
        for operand in [operands[1], *operands[2]["member_selectors"], *operands[3:]]:
            operand["geography_id"] = region
            operand["aggregate_fact_key"] += f":{region}"
        operands[2]["allocation_member_keys"] = [
            key + f":{region}" for key in operands[2]["allocation_member_keys"]
        ]
        facts.extend(members)
        references.append(
            replace(
                reference,
                name=f"region-{region}",
                ledger_selector=selector,
                value_operands=operands,
            )
        )

    def values(selected_facts, selected_references):
        result = compile_ledger_target_references(
            selected_facts, selected_references, country="fixture"
        )
        return {
            spec.name: (spec.value, spec.metadata["ledger_allocation_receipt"])
            for spec in result.specs
        }

    expected = values(facts, references)
    permutation = ordering.draw(st.permutations(tuple(range(len(references)))))
    assert (
        values(list(reversed(facts)), [references[index] for index in permutation])
        == expected
    )
