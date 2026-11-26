"""The columns a household-batched policyengine-us pass can read.

``microcosm.build.household_pass_reuse`` reuses a pass's values for
households whose input columns are unchanged. For policyengine-us those
columns are every variable in the static dependency graph of the pass's
variable (``PolicyEngineUSVariableMetadataIndex.variable_dependency_closure``:
every variable its formulas can read, on any branch and in any period), the
variable itself included, because a stored column replaces a formula.
"""

from __future__ import annotations

import enum

from microcosm.build.household_pass_reuse import HouseholdPassInputs, PassInputColumn
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSVariableMetadataIndex

__all__ = ["us_household_pass_inputs"]


def _stored_default(definition) -> object:
    """The default as a dataset stores it: Enum members by name."""

    default = definition.default_value
    if isinstance(default, enum.Enum):
        return default.name
    return default


def _absent_reads_default(definition) -> bool:
    """Whether a stored default gives the engine the same input as no column.

    Not for a variable the engine computes (a formula, or an ``adds`` or
    ``subtracts`` sum): a stored value replaces it. Not for a nonzero numeric
    default either: the engine uprates a stored value it reads at a later
    period and divides a yearly value into months, while an absent column
    reads the default unchanged.
    """

    computed = (
        bool(definition.formulas)
        or bool(getattr(definition, "adds", None))
        or bool(getattr(definition, "subtracts", None))
    )
    if computed:
        return False
    if definition.value_type in (int, float):
        return definition.default_value == 0
    return True


def us_household_pass_inputs(
    variable: str,
    *,
    metadata_index: PolicyEngineUSVariableMetadataIndex,
    system,
) -> HouseholdPassInputs:
    """Every engine variable ``variable``'s pass can read, with its table.

    ``system`` is the installed policyengine-us tax-benefit system; it gives
    each variable's entity, default and whether it has a formula. Graph nodes
    the system does not define (a few parameter paths the static index reads
    as names) are left out, because the engine reads no column by those names.
    """

    closure = metadata_index.variable_dependency_closure(variable)
    names = sorted(
        name
        for name in {*closure.input_leaves, *closure.formula_nodes}
        if name in system.variables
    )
    columns = tuple(
        PassInputColumn(
            name=name,
            entity=system.variables[name].entity.key,
            default=_stored_default(system.variables[name]),
            absent_reads_default=_absent_reads_default(system.variables[name]),
        )
        for name in names
    )
    return HouseholdPassInputs(
        variable=variable,
        columns=columns,
        description=(
            f"policyengine-us {closure.engine_version} dependency graph of "
            f"{variable}: {len(columns)} variables, sha256 {closure.sha256}"
        ),
    )
