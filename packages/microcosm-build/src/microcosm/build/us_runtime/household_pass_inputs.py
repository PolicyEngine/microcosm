"""The columns a household-batched policyengine-us pass can read.

``microcosm.build.household_pass_reuse`` reuses a pass's values for
households whose input columns are unchanged. For policyengine-us those
columns are every variable the pass's variable can read, the variable itself
included, because a stored column replaces a formula. The list starts from
the static dependency graph
(``PolicyEngineUSVariableMetadataIndex.variable_dependency_closure``: every
variable its formulas read through ``entity(...)``, ``add`` and similar
calls, on any branch and in any period). That graph does not follow some
reads, so the list adds, until nothing new appears:

* every string in a listed variable's class source, or among its formula
  functions' constants and closure values, that names an engine variable,
  with each such variable's own graph. That covers
  ``simulation.get_branch(...).calculate("...")``,
  ``simulation.get_holder("...")``, ``all_of_variables([...])`` and formulas
  generated at import without a class source (the state flags such as ``CA``
  read ``state_code_str`` this way);
* each listed variable's ``defined_for`` variable (state programs name their
  state flag as a ``StateCode`` member, not a string).

A string that names a variable without reading it only adds a column to the
digest, which can rerun a household but never reuses a wrong value.

Reads inside module-level helper functions are not followed: in
policyengine-us 2.2.1 the only variables those helpers read outside the list
are behavioral-response measurements (``household_net_income``,
``marginal_tax_rate`` and similar), which a simulation without a reform does
not compute. ``test_us_household_pass_inputs.py`` pins that set, so an engine
whose helpers read anything else fails until it is reviewed.
"""

from __future__ import annotations

import ast
import enum
import inspect
import textwrap
import types
from functools import cache

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
    names = _readable_variables(variable, metadata_index=metadata_index, system=system)
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
            f"policyengine-us {closure.engine_version}: {len(columns)} variables "
            f"{variable} can read (dependency graph sha256 {closure.sha256}, "
            "with string and defined_for reads)"
        ),
    )


def _graph_variables(
    name: str, *, metadata_index: PolicyEngineUSVariableMetadataIndex, system
) -> set[str]:
    try:
        closure = metadata_index.variable_dependency_closure(name)
    except ValueError:
        # A variable the static index does not define (one generated at
        # import) has no graph to add; its own name is still listed.
        return {name}
    return {
        node
        for node in (*closure.input_leaves, *closure.formula_nodes)
        if node in system.variables
    }


@cache
def _class_string_constants(variable_class: type) -> frozenset[str]:
    try:
        source = inspect.getsource(variable_class)
    except (OSError, TypeError):
        # A class generated at import has no class source; its formula
        # functions' constants are read in _formula_strings.
        return frozenset()
    return frozenset(
        node.value
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _code_strings(code: types.CodeType) -> set[str]:
    strings = set()
    for constant in code.co_consts:
        if isinstance(constant, str):
            strings.add(constant)
        elif isinstance(constant, types.CodeType):
            strings |= _code_strings(constant)
    return strings


def _formula_strings(definition) -> set[str]:
    """Strings among the formula functions' constants and closure values."""

    strings: set[str] = set()
    for formula in definition.formulas.values():
        code = getattr(formula, "__code__", None)
        if code is not None:
            strings |= _code_strings(code)
        for cell in getattr(formula, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, str):
                strings.add(value)
            elif isinstance(value, (list, tuple, set, frozenset)):
                strings |= {item for item in value if isinstance(item, str)}
    return strings


def _extra_reads(definition, system) -> set[str]:
    """Variables ``definition`` reads that its static graph may not show."""

    reads = {
        value
        for value in (
            _class_string_constants(type(definition)) | _formula_strings(definition)
        )
        if value in system.variables
    }
    defined_for = getattr(definition, "defined_for", None)
    if defined_for is not None:
        reads.add(getattr(defined_for, "name", defined_for))
    reads.discard(definition.name)
    return {read for read in reads if read in system.variables}


def _readable_variables(
    variable: str, *, metadata_index: PolicyEngineUSVariableMetadataIndex, system
) -> list[str]:
    names = _graph_variables(variable, metadata_index=metadata_index, system=system)
    names.add(variable)
    pending = sorted(names)
    while pending:
        definition = system.variables[pending.pop()]
        for read in sorted(_extra_reads(definition, system) - names):
            added = {read} | _graph_variables(
                read, metadata_index=metadata_index, system=system
            )
            pending.extend(sorted(added - names))
            names |= added
    return sorted(names)
