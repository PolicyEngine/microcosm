"""Exact member rosters for density-preserving Census roll-forwards."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
    reference_fact_selectors,
    target_spec_from_ledger_reference,
)

TENURE = "census.household_tenure"


def _fact(
    key: str,
    value: float,
    *,
    period: int = 2023,
    tenure: object = 1,
    geography: str = "01",
    population: bool = False,
) -> dict:
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{key}",
        "value": value,
        "period": {"type": "calendar_year", "value": period},
        "geography": {
            "level": "region",
            "id": geography,
            "name": geography,
            "vintage": "regc_2025" if population else "unstated",
        },
        "entity": {"name": "person" if population else "household"},
        "aggregation": {"method": "sum"},
        "source": {"source_name": "official_statistics"},
        "observed_measure": {
            "source_measure_id": "population" if population else "count",
            "source_concept": "resident_population" if population else "households",
            "unit": "people" if population else "households",
        },
        "layout": {
            "record_set_id": f"erp.{period}.all_ages"
            if population
            else "census.tenure",
        },
        "dimensions": {} if population else {TENURE: tenure},
    }


def _reference(*, grouped: bool = True) -> LedgerTargetReference:
    return LedgerTargetReference(
        name="owned including trust" if grouped else "not owned",
        ledger_selector={
            "source_name": "official_statistics",
            "source_measure_id": "count",
            "source_concept": "households",
            "entity_name": "household",
            "record_set_id": "census.tenure",
            "geography_level": "region",
            "geography_id": "01",
            "geography_vintage": "unstated",
            "period_type": "calendar_year",
            "period_value": 2023,
            "dimensions": [TENURE],
            "dimension_values": {TENURE: [1, 3] if grouped else 2},
        },
        value_operation="rolled_forward_by_ratio",
        value_operands=(
            {"role": "base", **({"expected_member_count": 2} if grouped else {})},
            *(
                {
                    "role": role,
                    "source_measure_id": "population",
                    "source_concept": "resident_population",
                    "entity_name": "person",
                    "record_set_id": f"erp.{period}.all_ages",
                    "geography_level": "region",
                    "geography_id": "01",
                    "geography_vintage": "regc_2025",
                    "period_type": "calendar_year",
                    "period_value": period,
                    "dimensions": [],
                }
                for role, period in (("numerator", 2025), ("denominator", 2023))
            ),
        ),
        entity="household",
        measure="households",
        filter="owned_indicator" if grouped else "not_owned_indicator",
        period=2025,
        family="housing_tenure",
    )


def _facts(
    *,
    owned: float = 42_546,
    trust: float = 7_806,
    numerator: float = 201_100,
    denominator: float = 198_500,
) -> list[dict]:
    return [
        _fact("owned", owned, tenure=1),
        _fact("trust", trust, tenure=3),
        _fact("not_owned", 20_118, tenure=2),
        _fact("erp_2025", numerator, period=2025, population=True),
        _fact("erp_2023", denominator, population=True),
    ]


def _compile(facts=None, reference=None):
    return compile_ledger_target_references(
        _facts() if facts is None else facts,
        [_reference() if reference is None else reference],
        country="fixture",
    ).specs[0]


def test_owned_and_trust_scale_together_with_base_only_count_and_keys():
    spec = _compile()
    assert spec.value == pytest.approx((42_546 + 7_806) * 201_100 / 198_500)
    assert spec.value != round(spec.value)
    assert spec.metadata["ledger_value_base_cells"] == "2"
    assert json.loads(spec.metadata["ledger_value_base_fact_keys"]) == [
        "ledger.aggregate_fact.v2:owned",
        "ledger.aggregate_fact.v2:trust",
    ]
    assert spec.metadata["ledger_member_fact_count"] == "4"
    assert len(json.loads(spec.metadata["ledger_member_fact_keys"])) == 4
    assert all(
        "expected_member_count" not in selector
        for selector in reference_fact_selectors(_reference())
    )


def test_scalar_base_retains_its_entity_and_vintage_with_person_growth_operands():
    spec = _compile(reference=_reference(grouped=False))
    assert spec.value == pytest.approx(20_118 * 201_100 / 198_500)
    assert spec.metadata["ledger_value_base_cells"] == "1"


@pytest.mark.parametrize(
    "codes", [(1, 1), (3, 3), (1,), (3,), (1, 3, 3), ("1", "3"), (1.0, 3.0)]
)
def test_grouped_base_requires_each_integer_member_once(codes):
    facts = [
        _fact(f"base_{index}", 20, tenure=code) for index, code in enumerate(codes)
    ] + _facts()[-2:]
    with pytest.raises(
        ValueError, match="typed member|expected 2 base facts|base matched no"
    ):
        _compile(facts)


def test_pre_resolved_grouped_facts_cannot_bypass_exact_roster_validation():
    facts = _facts()
    duplicate_owned = deepcopy(facts[0])
    duplicate_owned["aggregate_fact_key"] = "ledger.aggregate_fact.v2:duplicate_owned"
    with pytest.raises(ValueError, match="each declared typed member"):
        target_spec_from_ledger_reference(
            (facts[0], duplicate_owned, *facts[-2:]), _reference()
        )


@pytest.mark.parametrize(
    "change",
    [
        "extra_dimension",
        "mixed_period",
        "wrong_record_set",
        "wrong_geography",
        "wrong_source",
    ],
)
def test_grouped_member_cannot_cross_dimension_period_or_source_cell(change):
    facts = _facts()
    trust = facts[1]
    if change == "extra_dimension":
        trust["dimensions"]["unselected"] = "extra"
    elif change == "mixed_period":
        trust["period"]["value"] = 2024
    elif change == "wrong_record_set":
        trust["layout"]["record_set_id"] = "another.table"
    elif change == "wrong_geography":
        trust["geography"]["id"] = "02"
    else:
        trust["source"]["source_name"] = "another_publisher"
    with pytest.raises(ValueError, match="expected 2 base facts"):
        _compile(facts)


@pytest.mark.parametrize(
    "pin",
    [
        "source_name",
        "source_measure_id",
        "source_concept",
        "entity_name",
        "record_set_id",
        "period_type",
        "period_value",
        "geography_level",
        "geography_id",
        "geography_vintage",
        "dimensions",
    ],
)
def test_grouped_base_contract_requires_closed_source_and_dimension_pins(pin):
    reference = _reference()
    selector = {
        key: value for key, value in reference.ledger_selector.items() if key != pin
    }
    with pytest.raises(ValueError, match="must fix one source"):
        replace(reference, ledger_selector=selector)


@pytest.mark.parametrize("roster", [[1, 1], [], [1, 3, 2]])
def test_grouped_base_rejects_duplicate_or_count_mismatched_rosters(roster):
    reference = _reference()
    with pytest.raises(ValueError, match="unique scalars|roster length"):
        replace(
            reference,
            ledger_selector={
                **reference.ledger_selector,
                "dimension_values": {TENURE: roster},
            },
        )


@pytest.mark.parametrize("count", [True, 0, -1, 2.0])
def test_grouped_count_is_a_positive_integer(count):
    reference = _reference()
    with pytest.raises(ValueError, match="positive integer"):
        replace(
            reference,
            value_operands=(
                {"role": "base", "expected_member_count": count},
                *reference.value_operands[1:],
            ),
        )


def test_top_level_member_count_also_counts_only_base_and_disagreement_refuses():
    reference = replace(_reference(), expected_member_count=2)
    assert _compile(reference=reference).metadata["ledger_value_base_cells"] == "2"
    with pytest.raises(ValueError, match="declarations disagree"):
        replace(reference, expected_member_count=4)


@pytest.mark.parametrize("operand_count", ["absent", None])
def test_absent_or_null_base_count_keeps_the_top_level_count(operand_count):
    reference = _reference(grouped=False)
    base = {"role": "base"}
    if operand_count is None:
        base["expected_member_count"] = None
    reference = replace(
        reference,
        expected_member_count=2,
        value_operands=(base, *reference.value_operands[1:]),
    )
    with pytest.raises(ValueError, match="expected 2 base facts, got 1"):
        _compile(reference=reference)


@pytest.mark.parametrize("operand_count", ["absent", None])
@pytest.mark.parametrize("shape", ["reference_selector", "base_operand"])
def test_null_base_count_cannot_admit_a_short_geography_list(shape, operand_count):
    reference = _reference(grouped=False)
    selector = dict(reference.ledger_selector)
    base = {"role": "base"}
    if shape == "reference_selector":
        selector["geography_id"] = ["01", "02"]
    else:
        base["geography_id"] = ["01", "02"]
    if operand_count is None:
        base["expected_member_count"] = None
    reference = replace(
        reference,
        ledger_selector=selector,
        expected_member_count=3,
        value_operands=(base, *reference.value_operands[1:]),
    )
    facts = [
        _fact("region01", 10, tenure=2),
        _fact("region02", 20, tenure=2, geography="02"),
        *_facts()[-2:],
    ]
    with pytest.raises(ValueError, match="expected 3 base facts, got 2"):
        _compile(facts, reference)


def test_explicit_multiple_geographies_still_sum_one_scalar_per_geography():
    reference = _reference(grouped=False)
    selector = {
        key: value
        for key, value in reference.ledger_selector.items()
        if key != "record_set_id"
    }
    operands = (
        {"role": "base", "geography_id": ["01", "02"], "expected_member_count": 2},
        *reference.value_operands[1:],
    )
    reference = replace(reference, ledger_selector=selector, value_operands=operands)
    facts = [
        _fact("region01", 10, tenure=2),
        _fact("region02", 20, tenure=2, geography="02"),
        *_facts()[-2:],
    ]
    assert _compile(facts, reference).value == pytest.approx(30 * 201_100 / 198_500)


@pytest.mark.parametrize(
    "ratio,passes",
    [
        (0.5, False),
        (math.nextafter(0.5, 1), True),
        (2, True),
        (math.nextafter(2, 3), False),
    ],
)
def test_ratio_interval_is_open_at_half_and_closed_at_two(ratio, passes):
    facts = _facts(numerator=ratio, denominator=1)
    if passes:
        assert _compile(facts).value == pytest.approx((42_546 + 7_806) * ratio)
    else:
        with pytest.raises(ValueError, match="outside"):
            _compile(facts)


@pytest.mark.parametrize("denominator", [0, -1, math.inf, -math.inf, math.nan])
def test_ratio_denominator_must_be_finite_and_positive(denominator):
    with pytest.raises(ValueError, match="finite positive|invalid value"):
        _compile(_facts(denominator=denominator))


@settings(max_examples=40, database=None, deadline=None)
@given(
    owned=st.integers(min_value=1, max_value=1_000_000),
    trust=st.integers(min_value=1, max_value=1_000_000),
    denominator=st.integers(min_value=1, max_value=10_000_000),
    ratio=st.floats(
        min_value=0.500001, max_value=2, allow_nan=False, allow_infinity=False
    ),
)
def test_positive_growth_preserves_tenure_shares_and_households_per_person(
    owned, trust, denominator, ratio
):
    numerator = denominator * ratio
    facts = _facts(
        owned=owned, trust=trust, numerator=numerator, denominator=denominator
    )
    grouped = _compile(facts).value
    scalar_reference = _reference(grouped=False)
    owned_reference = replace(
        scalar_reference,
        ledger_selector={
            **scalar_reference.ledger_selector,
            "dimension_values": {TENURE: 1},
        },
    )
    grown_owned = _compile(facts, owned_reference).value
    assert grouped == pytest.approx(math.fsum([owned, trust]) * numerator / denominator)
    assert grouped / numerator == pytest.approx((owned + trust) / denominator)
    assert grown_owned / grouped == pytest.approx(owned / (owned + trust))


def test_growth_operands_require_all_age_same_series_and_declared_periods():
    reference = _reference()
    for field, replacement in (
        ("period", {"type": "calendar_year", "value": 2024}),
        ("dimensions", {"age_band": "0_14"}),
        ("entity", {"name": "household"}),
    ):
        facts = deepcopy(_facts())
        facts[-1][field] = replacement
        with pytest.raises(ValueError, match="must resolve exactly once"):
            _compile(facts, reference)
