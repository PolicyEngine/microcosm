"""The release tool and the recertification scorer share three rules; check them.

``microcosm.data.compatibility_recertification`` holds the signed-component
rule that scores SOI rows, the population-aggregate formulas a batch engine
refuses, and the walker that reads which of them an engine computed. The data
shard cannot import the tool, so the tool reads all three from it
(microcosm#1585). These tests hold the tool's names to the shared rules, and
the walker to each tool's former behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.data import compatibility_recertification as recert
from test_support.microcosm_build.us_post_export_scoring import _load_builder_module


@pytest.fixture(scope="module")
def release_tool():
    return _load_builder_module()


@pytest.fixture(scope="module")
def recertification_tool():
    import tools.build_us_compatibility_recertification as tool

    return tool


def test_population_aggregate_variables_are_the_release_tools(release_tool):
    assert recert.POPULATION_AGGREGATE_VARIABLES == (
        release_tool.US_POPULATION_AGGREGATE_VARIABLES
    )
    assert set(recert.STATE_AGGREGATE_VARIABLES) <= set(
        release_tool.US_POPULATION_AGGREGATE_VARIABLES
    )


def test_the_release_tool_reads_the_shared_rules(release_tool):
    assert (
        release_tool.US_POPULATION_AGGREGATE_VARIABLES
        is recert.POPULATION_AGGREGATE_VARIABLES
    )
    assert release_tool._signed_component is recert.signed_component
    assert release_tool._engine_known_periods is recert.engine_known_periods


def test_each_tool_walks_in_its_own_mode(release_tool, recertification_tool):
    """The release tool walks every engine it reaches and raises on a variable
    an engine does not define; the recertification tool reads the root alone
    and skips such a variable, as their copies did at 90e4d94e7."""

    aggregates = ("household_income_decile", "medicaid_slcsp_state_denominator")

    def engine(known, **links):
        def get_holder(name):
            if name not in aggregates:
                raise KeyError(name)
            return SimpleNamespace(get_known_periods=lambda: known.get(name, []))

        return SimpleNamespace(
            get_holder=get_holder,
            tax_benefit_system=SimpleNamespace(variables=set(aggregates)),
            **links,
        )

    branch = engine({"household_income_decile": [2025]})
    root = engine({"medicaid_slcsp_state_denominator": [2024]}, branches={"b": branch})
    assert release_tool._engine_known_periods(root, aggregates) == {
        ("household_income_decile", "2025"),
        ("medicaid_slcsp_state_denominator", "2024"),
    }
    with pytest.raises(KeyError):
        release_tool._engine_known_periods(root, (*aggregates, "undefined"))
    assert recertification_tool._engine_known_periods(
        root, (*aggregates, "undefined")
    ) == {("medicaid_slcsp_state_denominator", "2024")}


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
