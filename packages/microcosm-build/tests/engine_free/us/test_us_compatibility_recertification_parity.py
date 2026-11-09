"""The recertification scorer copies two release-tool rules; hold them equal.

``microcosm.data.compatibility_recertification`` scores SOI rows with the
release tool's signed-component rule and refuses the release tool's
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
