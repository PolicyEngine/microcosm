"""The input columns of a household-batched policyengine-us pass.

``us_household_pass_inputs`` lists every variable a pass's formulas can
read, the variable itself included, with the table it is stored in and the
value an absent column reads as: the engine's static dependency graph, plus
the reads that graph does not follow (names in strings, such as a branch
simulation's ``calculate("income_tax")``, ``get_holder("has_itin")`` and
``all_of_variables([...])``, and ``defined_for`` state flags).
``microcosm.build.household_pass_reuse`` digests these columns per household
to decide which households a later pass can take from an earlier one.
``engine_workflow/us/test_us_post_calibration_take_up_inputs.py`` checks the
list against the names the engine reads while it computes each variable.
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
    graph = {
        name
        for name in {*closure.input_leaves, *closure.formula_nodes}
        if name in system.variables
    }
    names = [column.name for column in inputs.columns]
    assert inputs.variable == variable
    assert names == sorted(set(names))
    assert graph <= set(names)
    assert variable in names
    assert closure.sha256 in inputs.description
    for column in inputs.columns:
        assert column.entity == system.variables[column.name].entity.key


def test_pass_inputs_add_the_reads_the_static_graph_does_not_follow(
    metadata_index, system
) -> None:
    def added(variable: str) -> set[str]:
        closure = metadata_index.variable_dependency_closure(variable)
        listed = {
            column.name
            for column in us_household_pass_inputs(
                variable, metadata_index=metadata_index, system=system
            ).columns
        }
        return listed - {*closure.input_leaves, *closure.formula_nodes}

    assert added("uncapped_ssi") == set()
    medicaid_added = added("is_medicaid_eligible")
    # tax_liability_if_itemizing computes income_tax in a branch simulation;
    # has_tin reads a stored has_itin through get_holder; is_adult_for_medicaid
    # combines its _fc and _nfc variables with all_of_variables; state
    # programs are defined_for a state flag.
    for name in (
        "income_tax",
        "has_itin",
        "is_adult_for_medicaid_fc",
        "is_military",
        "refundable_payroll_tax_credit",
        "CA",
        "NY",
    ):
        assert name in medicaid_added, name


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


#: Variables that module-level helper functions of listed formulas read and
#: the list leaves out. Each is a behavioral-response measurement, read by
#: ``get_behavioral_response_measurements``, ``calculate_income_lsr_effect`` or
#: ``calculate_substitution_lsr_effect`` only when the simulation has a
#: baseline branch and nonzero elasticities; the passes build no reform, and
#: the engine-workflow trace finds none of them read.
_REVIEWED_HELPER_READS = {
    "household_net_income",
    "income_elasticity",
    "marginal_tax_rate",
    "marginal_tax_rate_on_capital_gains",
    "substitution_elasticity",
}


def _code_strings_and_names(code):
    strings, names = set(), set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, str):
            strings.add(constant)
        elif isinstance(constant, type(code)):
            nested_strings, nested_names = _code_strings_and_names(constant)
            strings |= nested_strings
            names |= nested_names
    return strings, names


@pytest.mark.parametrize("variable", ["uncapped_ssi", "is_medicaid_eligible"])
def test_helper_functions_read_no_unreviewed_variable_outside_the_list(
    variable, metadata_index, system
) -> None:
    """The list does not follow module-level helpers; this pins what they read.

    Walks every policyengine-us function a listed formula calls by global
    name, and the functions those call, and collects the strings in their
    code that name engine variables.
    """

    import types

    listed = {
        column.name
        for column in us_household_pass_inputs(
            variable, metadata_index=metadata_index, system=system
        ).columns
    }
    helper_reads: set[str] = set()
    for name in listed:
        for formula in system.variables[name].formulas.values():
            _, names = _code_strings_and_names(formula.__code__)
            pending = [formula.__globals__.get(global_name) for global_name in names]
            seen = set()
            while pending:
                helper = pending.pop()
                if (
                    not isinstance(helper, types.FunctionType)
                    or not helper.__module__.startswith("policyengine_us")
                    or helper in seen
                ):
                    continue
                seen.add(helper)
                strings, names = _code_strings_and_names(helper.__code__)
                helper_reads |= {
                    value for value in strings if value in system.variables
                }
                pending.extend(
                    helper.__globals__.get(global_name) for global_name in names
                )

    assert helper_reads - listed == _REVIEWED_HELPER_READS
