"""The reviewed PolicyEngine-US helpers against the installed engine's source.

Building the static index already fails on a helper with no review or with
changed code. These checks close the other direction: no review is stale, and
every review's reads are recorded where the installed formulas call it.
"""

from pathlib import Path

import policyengine_us

from microcosm.frame.adapters._policyengine_us_helper_reviews import (
    POLICYENGINE_US_HELPER_REVIEWS,
)
from microcosm.frame.adapters._policyengine_us_source_index import (
    _index_policyengine_us_sources,
)

_INDEX = _index_policyengine_us_sources(
    Path(policyengine_us.__file__).parent / "variables"
)


def test_every_review_is_for_a_helper_the_formulas_call() -> None:
    assert set(_INDEX.reviewed_helpers) == set(POLICYENGINE_US_HELPER_REVIEWS)


def test_every_reviewed_read_is_recorded_for_a_caller() -> None:
    helper_call_targets = {
        target
        for target, receipts in _INDEX.consumers.items()
        if any(receipt.kind.endswith("helper_call") for receipt in receipts)
    }
    reviewed_reads = {
        name
        for review in POLICYENGINE_US_HELPER_REVIEWS.values()
        for name in review.reads
    }

    assert reviewed_reads <= helper_call_targets


def test_tax_unit_non_dep_add_reads_reach_their_formulas() -> None:
    """policyengine-us 2.60 moved many tax-unit sums onto this helper."""

    assert {
        receipt.consumer
        for receipt in _INDEX.consumers["investment_income_elected_form_4952"]
        if receipt.kind == "helper_call"
    } == {"net_capital_gain", "schedule_d_tax_worksheet_form_4952_lines"}
