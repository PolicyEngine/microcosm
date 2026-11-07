"""The recertification scorer copies three release-tool rules; hold them equal.

``microcosm.data.compatibility_recertification`` scores SOI rows with the
release tool's signed-component rule and its register of SOI concepts that
are one engine variable less another, and refuses the release tool's
population-aggregate formulas in batch engines. The data shard cannot import
the tool, so it carries copies, and these tests are what keeps them the
tool's.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.data import compatibility_recertification as recert
from test_support.microcosm_build.us_post_export_scoring import _load_builder_module


@pytest.fixture(scope="module")
def release_tool():
    return _load_builder_module()


def test_population_aggregate_variables_are_the_release_tools(release_tool):
    assert recert.POPULATION_AGGREGATE_VARIABLES == (
        release_tool.US_POPULATION_AGGREGATE_VARIABLES
    )
    assert set(recert.STATE_AGGREGATE_VARIABLES) <= set(
        release_tool.US_POPULATION_AGGREGATE_VARIABLES
    )


_SOURCES = sorted(
    {
        "adjusted_gross_income",
        "rent_and_royalty_net_income",
        "capital_gains_losses",
        "business_net_losses",
        "estate_losses",
        "partnership_and_s_corp_losses",
        "rent_and_royalty_net_losses",
        "miscellaneous_losses",
        "income_tax",
        "income_tax_after_nonrefundable_credits",
        "eitc",
        "ctc",
        "employment_income",
        "taxable_social_security",
    }
)


@settings(max_examples=300, deadline=None)
@given(
    values=st.lists(
        st.floats(-1e12, 1e12, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=16,
    ),
    source=st.sampled_from(_SOURCES),
)
def test_signed_component_is_the_release_tools(release_tool, values, source):
    np.testing.assert_array_equal(
        recert.signed_component(values, source),
        release_tool._signed_component(np.asarray(values), source),
    )


def test_soi_difference_concepts_are_the_release_tools(release_tool):
    assert recert.SOI_DIFFERENCE_CONCEPTS == release_tool.SOI_DIFFERENCE_CONCEPTS
    assert not set(recert.SOI_DIFFERENCE_CONCEPTS) & set(release_tool.SOI_VARIABLE_MAP)


_AMOUNTS = st.floats(-1e12, 1e12, allow_nan=False, allow_infinity=False)


@settings(max_examples=300, deadline=None)
@given(
    pairs=st.lists(st.tuples(_AMOUNTS, _AMOUNTS), min_size=1, max_size=16),
    concept=st.sampled_from(sorted(recert.SOI_DIFFERENCE_CONCEPTS)),
)
def test_difference_rows_score_as_the_release_tool_builds_them(
    release_tool, pairs, concept
):
    """Both sides take the minuend less the subtrahend, then its positive part."""

    minuend, subtrahend = (np.asarray(column) for column in zip(*pairs, strict=True))
    built = release_tool._signed_component(
        release_tool.soi_difference_concept_values(minuend, subtrahend), concept
    )
    scored = recert.signed_component(
        recert.combine_metric_values("difference", [minuend, subtrahend]), concept
    )
    np.testing.assert_array_equal(scored, built)


_NONNEGATIVE = st.floats(0.0, 1e9, allow_nan=False, allow_infinity=False)
_TAX_UNITS = st.lists(
    st.tuples(_NONNEGATIVE, _NONNEGATIVE, _NONNEGATIVE, st.booleans()),
    min_size=1,
    max_size=32,
)


def _line_22(release_tool, units):
    """Each unit's SOI income tax after credits, as the release tool builds it.

    A unit is ``(tax before credits, nonrefundable credits, weight, in mask)``.
    policyengine-us caps the credits at the tax before credits
    (``income_tax_capped_non_refundable_credits``), an engine premise the
    scenario test ``test_us_soi_income_tax_concept`` pins.
    """

    before, credits, weight, mask = (
        np.asarray(column) for column in zip(*units, strict=True)
    )
    capped = np.minimum(before, credits)
    minuend, subtrahend = release_tool.SOI_DIFFERENCE_CONCEPTS[
        "income_tax_after_nonrefundable_credits"
    ]
    assert (minuend, subtrahend) == (
        "income_tax_before_credits",
        "income_tax_capped_non_refundable_credits",
    )
    line_22 = release_tool._signed_component(
        release_tool.soi_difference_concept_values(before, capped),
        "income_tax_after_nonrefundable_credits",
    )
    before_credits = release_tool._signed_component(before, "income_tax_before_credits")
    return line_22, before_credits, credits, weight, mask.astype(bool)


@settings(max_examples=300, deadline=None)
@given(units=_TAX_UNITS)
def test_income_tax_after_credits_lies_between_zero_and_tax_before_credits(
    release_tool, units
):
    line_22, before, credits, _, _ = _line_22(release_tool, units)
    assert np.all(line_22 >= 0.0)
    assert np.all(line_22 <= before)
    np.testing.assert_array_equal(line_22, np.maximum(before - credits, 0.0))


@settings(max_examples=300, deadline=None)
@given(units=_TAX_UNITS)
def test_returns_with_tax_after_credits_never_outnumber_returns_with_tax_before(
    release_tool, units
):
    """N06500 <= N05800 holds unit by unit, so on every mask and weighting."""

    line_22, before, _, weight, mask = _line_22(release_tool, units)
    assert np.all(~(line_22 > 0.0) | (before > 0.0))
    after_returns = release_tool._soi_component_row(
        line_22, "income_tax_after_nonrefundable_credits", indicator=True
    )
    before_returns = release_tool._soi_component_row(
        before, "income_tax_before_credits", indicator=True
    )
    assert np.dot(weight, after_returns * mask) <= np.dot(weight, before_returns * mask)
    after_amount = release_tool._soi_component_row(
        line_22, "income_tax_after_nonrefundable_credits", indicator=False
    )
    assert np.dot(weight, after_amount * mask) <= np.dot(weight, before * mask)
