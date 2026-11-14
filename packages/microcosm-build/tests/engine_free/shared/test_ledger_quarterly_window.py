"""The quarterly window: the mean of one series over declared quarters (Y3, chronicle#420 item 1).

ONS's NEET Table 1 gives each Labour Force Survey quarter its own record set
(``people_nsa.2025_q1`` and so on). The window reads the four quarters inside
calendar 2025 as one series, and that reading must not move any other
reference's series identity.
"""

from __future__ import annotations

import json

import pytest

from microcosm.build.ledger_targets import (
    QUARTERLY_WINDOW_AVERAGE,
    LedgerTargetReference,
    _selector_period_invariant_key,
    compile_ledger_target_references,
)

QUARTERS = ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"]


def _neet_fact(quarter: str, value: float, *, measure: str = "neet_16_to_24"):
    stamp = quarter.lower().replace("-", "_")
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:neet-{measure}-{stamp}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:neet-{measure}",
        "label": f"United Kingdom {quarter} young people who were NEET",
        "source_release_key": "ons_neet_2026_08_27",
        "value": value,
        "period": {"type": "quarter", "value": quarter},
        "geography": {
            "level": "country",
            "id": "K02000001",
            "name": "United Kingdom",
            "vintage": "current",
        },
        "entity": {"name": "person", "role": "young_person"},
        "observed_measure": {
            "source_name": "ons",
            "source_measure_id": measure,
            "source_concept": "ons.lfs_neet_people",
            "unit": "people",
        },
        "concept_alignment": {
            "source_concept": "ons.lfs_neet_people",
            "canonical_concept": "ons.lfs_neet_people",
            "relation": "source_label",
            "authority": "ons",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": "ons",
            "source_table": "NEET Table 1",
            "source_file": "notineducationemploymentortrainingaj26.xlsx",
            "source_sha256": "b" * 64,
            "vintage": "neet_2026_08_27",
        },
        "dimensions": {
            "age_band": "16_to_24",
            "economic_activity": "all",
            "seasonal_adjustment": "not_seasonally_adjusted",
            "sex": "all",
        },
        "universe_constraints": {"domain": "employment"},
        "layout": {
            "record_set_id": f"ons.neet.august_2026.people_nsa.{stamp}",
            "record_set_spec_id": f"ons.neet.august_2026.people_nsa.{stamp}.v1",
            "groupby_dimension": "ons.neet_quarter",
            "groupby_value_id": stamp,
            "measure_id": measure,
        },
    }


def _reference(**overrides) -> LedgerTargetReference:
    fields = {
        "name": "ons/neet/people_aged_16_to_24",
        "ledger_selector": {
            "source_name": "ons",
            "source_concept": ["ons.lfs_neet_people"],
            "source_measure_id": "neet_16_to_24",
            "period_type": "quarter",
            "period_value": QUARTERS,
        },
        "value_operation": QUARTERLY_WINDOW_AVERAGE,
        "period_match_policy": "source_window",
        "entity": "person",
        "measure": "people",
        "period": 2025,
    }
    fields.update(overrides)
    return LedgerTargetReference(**fields)


def test_the_window_is_the_mean_of_its_declared_quarters() -> None:
    values = (900.0, 940.0, 1000.0, 960.0)
    facts = [_neet_fact(q, v) for q, v in zip(QUARTERS, values, strict=True)]
    # Quarters outside the window are left out.
    facts += [_neet_fact("2024-Q4", 5_000.0), _neet_fact("2026-Q1", 5_000.0)]

    (spec,) = compile_ledger_target_references(
        facts[::-1], [_reference()], country="uk"
    ).specs

    assert spec.period == 2025
    assert spec.value == pytest.approx(950.0)
    assert spec.metadata["ledger_value_operation"] == QUARTERLY_WINDOW_AVERAGE
    assert spec.metadata["ledger_member_fact_count"] == "4"
    assert json.loads(spec.metadata["ledger_source_quarters"]) == QUARTERS
    assert spec.metadata["ledger_source_quarter_count"] == "4"
    assert spec.metadata["ledger_value_formula"] == (
        "sum_of_declared_quarters / declared_quarter_count"
    )
    assert spec.metadata["ledger_source_release_key"] == "ons_neet_2026_08_27"


@pytest.mark.parametrize("defect", ["missing", "duplicate", "other_series", "month"])
def test_the_window_refuses_an_incomplete_or_mixed_set_of_quarters(defect) -> None:
    facts = [_neet_fact(q, 1.0) for q in QUARTERS]
    if defect == "missing":
        facts.pop()
    elif defect == "duplicate":
        facts.append(_neet_fact(QUARTERS[0], 2.0))
        facts[-1]["aggregate_fact_key"] += "-again"
    elif defect == "other_series":
        facts[2]["layout"]["record_set_id"] = "ons.neet.august_2026.men_nsa.2025_q3"
    else:
        facts[1]["period"] = {"type": "month", "value": "2025-05"}

    with pytest.raises(ValueError, match="quarterly window"):
        compile_ledger_target_references(facts, [_reference()], country="uk")


@pytest.mark.parametrize(
    "fields",
    [
        {"period_match_policy": "exact"},
        {"period": None},
        {"value_operands": ({"dimension_values": {"sex": "all"}},)},
        {"ledger_selector": {"period_type": "month", "period_value": QUARTERS}},
        {"ledger_selector": {"period_type": "quarter", "period_value": []}},
        {"ledger_selector": {"period_type": "quarter", "period_value": "2025-Q1"}},
        {"ledger_selector": {"period_type": "quarter", "period_value": ["2025Q1"]}},
        {"ledger_selector": {"period_type": "quarter", "period_value": ["2025-Q5"]}},
        {"ledger_selector": {"period_type": "quarter", "period_value": QUARTERS[::-1]}},
        {
            "ledger_selector": {
                "period_type": "quarter",
                "period_value": ["2025-Q1", "2025-Q1"],
            }
        },
    ],
)
def test_the_window_needs_an_explicit_ordered_list_of_quarters(fields) -> None:
    with pytest.raises(ValueError):
        _reference(**fields)


def test_only_the_quarterly_window_reads_quarters_as_one_series() -> None:
    first, second = _neet_fact("2025-Q1", 1.0), _neet_fact("2025-Q2", 1.0)

    assert _selector_period_invariant_key(first) != _selector_period_invariant_key(
        second
    )
    assert _selector_period_invariant_key(
        first, strip_quarters=True
    ) == _selector_period_invariant_key(second, strip_quarters=True)
