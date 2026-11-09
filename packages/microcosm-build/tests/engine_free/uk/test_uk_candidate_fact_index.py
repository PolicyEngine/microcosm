"""The national compile's source index selects exactly the scanned facts (#1123)."""

from __future__ import annotations

import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.ledger_targets import (
    _candidate_facts_for_reference,
    _fact_indices_by_source,
)


def _fact(source: str, concept: str, value: float) -> dict:
    return {
        "source": {"source_name": source},
        "observed_measure": {"source_concept": concept},
        "geography": {"id": "K02000001", "level": "country"},
        "period": {"type": "calendar_year", "value": 2025},
        "value": value,
    }


_FACTS = (
    _fact("ons", "ons.a", 1.0),
    _fact("hmrc", "hmrc.a", 2.0),
    _fact("ons", "ons.b", 3.0),
    _fact("lps", "lps.a", 4.0),
    _fact("hmrc", "hmrc.a", 5.0),
)


def _reference(**overrides) -> LedgerTargetReference:
    fields = {
        "name": "t",
        "ledger_selector": {"source_name": "ons"},
        "entity": "household",
        "measure": "m",
        "period": 2025,
    }
    fields.update(overrides)
    return LedgerTargetReference(**fields)


@pytest.mark.parametrize(
    "reference",
    [
        _reference(),
        _reference(ledger_selector={"source_name": ["hmrc", "lps"]}),
        _reference(ledger_selector={"source_concept": "hmrc.a"}),
        _reference(ledger_selector={"source_name": "absent"}),
        _reference(
            ledger_selector={"source_name": "ons", "source_concept": "ons.a"},
            value_operation="rolled_forward_by_ratio",
            value_operands=(
                {"role": "base"},
                {
                    "role": "numerator",
                    "source_name": "lps",
                    "source_concept": "lps.a",
                    "period_type": "calendar_year",
                    "period_value": 2025,
                },
                {
                    "role": "denominator",
                    "source_name": "lps",
                    "source_concept": "lps.a",
                    "period_type": "calendar_year",
                    "period_value": 2021,
                },
            ),
        ),
    ],
    ids=["one_source", "source_list", "no_source_pin", "absent_source", "operands"],
)
def test_the_index_keeps_exactly_the_facts_a_full_scan_keeps(reference):
    indexed = _candidate_facts_for_reference(
        _FACTS, reference, fact_indices_by_source=_fact_indices_by_source(_FACTS)
    )
    assert indexed == _candidate_facts_for_reference(_FACTS, reference)


def test_the_index_keeps_feed_order_across_sources():
    reference = _reference(ledger_selector={"source_name": ["lps", "ons"]})
    indexed = _candidate_facts_for_reference(
        _FACTS, reference, fact_indices_by_source=_fact_indices_by_source(_FACTS)
    )
    assert [fact["value"] for fact in indexed] == [1.0, 3.0, 4.0]
