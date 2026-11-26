"""The input columns of a household-batched policyengine-us pass.

``us_household_pass_inputs`` lists every variable a pass's formulas can read
(the engine's static dependency graph of the variable, the variable itself
included) with the table it is stored in and the value an absent column
reads as. ``microcosm.build.household_pass_reuse`` digests these columns per
household to decide which households a later pass can take from an earlier
one.
"""

from __future__ import annotations

import pytest

from microcosm.build.household_pass_reuse import PassInputColumn
from microcosm.build.us_runtime.household_pass_inputs import us_household_pass_inputs
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSVariableMetadataIndex


@pytest.fixture(scope="module")
def metadata_index() -> PolicyEngineUSVariableMetadataIndex:
    return PolicyEngineUSVariableMetadataIndex()


@pytest.fixture(scope="module")
def system():
    from policyengine_us.system import system

    return system


@pytest.mark.parametrize("variable", ["uncapped_ssi", "is_medicaid_eligible"])
def test_pass_inputs_are_every_engine_variable_in_the_graph(
    variable, metadata_index, system
) -> None:
    closure = metadata_index.variable_dependency_closure(variable)
    inputs = us_household_pass_inputs(
        variable, metadata_index=metadata_index, system=system
    )

    # The graph also names a few parameter paths; the engine reads no column
    # by those names.
    expected = sorted(
        name
        for name in {*closure.input_leaves, *closure.formula_nodes}
        if name in system.variables
    )
    assert inputs.variable == variable
    assert [column.name for column in inputs.columns] == expected
    assert variable in expected
    assert closure.sha256 in inputs.description
    for column in inputs.columns:
        assert column.entity == system.variables[column.name].entity.key


def test_only_plain_inputs_with_neutral_defaults_read_as_absent(
    metadata_index, system
) -> None:
    """A stored value equal to the default is the same as no column only for
    an input without a formula whose default survives the engine's monthly
    division and uprating: a non-numeric value, a boolean, or zero."""

    columns = {
        column.name: column
        for variable in ("uncapped_ssi", "is_medicaid_eligible")
        for column in us_household_pass_inputs(
            variable, metadata_index=metadata_index, system=system
        ).columns
    }

    assert columns["employment_income_before_lsr"] == PassInputColumn(
        "employment_income_before_lsr", "person", default=0, absent_reads_default=True
    )
    assert columns["takes_up_snap_if_eligible"] == PassInputColumn(
        "takes_up_snap_if_eligible", "spm_unit", default=True, absent_reads_default=True
    )
    # Enum defaults are stored by member name.
    assert columns["immigration_status_str"] == PassInputColumn(
        "immigration_status_str", "person", default="CITIZEN", absent_reads_default=True
    )
    assert columns["county_fips"] == PassInputColumn(
        "county_fips", "household", default="", absent_reads_default=True
    )
    # Nonzero numeric defaults: a stored default is uprated or divided.
    assert columns["age"].absent_reads_default is False
    assert columns["state_fips"] == PassInputColumn(
        "state_fips", "household", default=6, absent_reads_default=False
    )
    # A stored value replaces a formula, including an ``adds`` sum.
    assert columns["uncapped_ssi"].absent_reads_default is False
    medically_needy_expenses = columns["medicaid_medically_needy_medical_expenses"]
    assert medically_needy_expenses.absent_reads_default is False
