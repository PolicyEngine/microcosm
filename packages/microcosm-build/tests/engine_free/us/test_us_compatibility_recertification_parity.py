"""The release tool and the recertification scorer share three rules; check them.

``microcosm.data.compatibility_recertification`` holds the signed-component
rule that scores SOI rows, the population-aggregate formulas a batch engine
refuses, and the walker that reads which of them an engine computed. The data
shard cannot import the tool, so the tool reads all three from it
(microcosm#1585). These tests hold the tool's names to the shared rules, and
the walker to each tool's former behaviour.

Two SOI rules run the other way: the scorer keeps its own copies of the
tool's no-op ledger filter rule and of the EITC masks the tool's SOI slice
applies (microcosm#880), so these tests also hold each copy to the tool's
function over generated rows.
"""

from __future__ import annotations

import ast
import inspect
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


_FILTER_VALUES = (
    "",
    " ",
    "all",
    "All",
    " ALL ",
    "_all",
    "all_",
    "all statuses",
    "all_returns",
    "none",
    "0",
    "3plus",
)


@settings(max_examples=300, deadline=None)
@given(value=st.one_of(st.sampled_from(_FILTER_VALUES), st.text(max_size=12)))
def test_noop_ledger_filter_rule_is_the_release_tools(release_tool, value):
    assert recert._is_noop_ledger_filter_value(value) == (
        release_tool._is_noop_ledger_filter_value(value)
    )


def _eitc_literals(function, method):
    """The string literals ``function`` passes to ``str.<method>``."""

    tree = ast.parse(inspect.getsource(function))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def test_eitc_child_count_patterns_are_the_release_tools(release_tool):
    """A record set or measure the tool learns to read as a child count must
    reach the scorer's copy, or the scorer would score that row unmasked."""

    tool = release_tool._soi_eitc_child_count_filter
    assert _eitc_literals(tool, "endswith") == set(recert._EITC_CHILD_COUNT_RECORD_SETS)
    assert _eitc_literals(tool, "startswith") == set(recert._EITC_CHILD_COUNT_MEASURES)


_RECORD_SETS = (
    "",
    "irs_soi.table_2_5",
    "irs_soi.table_2_5.eitc_by_agi",
    "eitc_by_agi_children.one_qualifying_child",
    "irs_soi.eitc_by_agi_children.two_qualifying_children.amount",
    *(f"irs_soi.table_2_5{suffix}" for suffix in recert._EITC_CHILD_COUNT_RECORD_SETS),
)
_MEASURES = (
    "",
    "eitc_total",
    "eitc_no_children",
    "returns_eitc_one_child_",
    "itemized_deductions_amount",
    *(f"{prefix}amount" for prefix in recert._EITC_CHILD_COUNT_MEASURES),
)
_DOMAINS = (
    recert._POSITIVE_EITC_DOMAIN,
    "individual_income_tax_returns",
    f"{recert._POSITIVE_EITC_DOMAIN}_by_state",
)


def _optional(values):
    return st.one_of(st.none(), st.sampled_from(values), st.text(max_size=8))


@settings(max_examples=500, deadline=None)
@given(
    domain=_optional(_DOMAINS),
    child_count=_optional(_FILTER_VALUES),
    record_set=_optional(_RECORD_SETS),
    measure=_optional(_MEASURES),
)
def test_eitc_mask_is_the_release_tools(
    release_tool, domain, child_count, record_set, measure
):
    """The scorer refuses a row exactly when the tool's SOI slice would narrow
    it to positive-EITC returns or to one EITC child count."""

    fields = {
        "ledger_domain": domain,
        "ledger_filter_eitc_child_count": child_count,
        "ledger_layout_record_set_id": record_set,
        "source_measure_id": measure,
    }
    metadata = {key: value for key, value in fields.items() if value is not None}
    assert recert._soi_eitc_masked(metadata) == (
        release_tool._soi_eitc_child_count_filter(metadata) is not None
        or release_tool._soi_requires_positive_eitc_filter(metadata)
    )
