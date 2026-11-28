"""A rule engine's entity inputs, read from source, and checks against them.

A firm table loads into a country model only if every exported column is one
of the model's formula-less variables on the firm entity, with the right value
type and enum members. This module reads those variables from a
policyengine-core country package's source with ``ast`` (it never imports the
package), so engine-free tests can check an export against a committed
snapshot. It also holds the input-status vocabulary a build uses to record how
it fills each input, and the checks that keep that record complete.

- :func:`read_entity_variables` reads every ``Variable`` subclass whose
  ``entity`` is a given name: value type, unit, default, enum members (the
  ``possible_values`` class resolved through the defining module's imports),
  definition period, whether it has a formula (a ``formula`` or
  ``formula_*`` method or attribute, ``adds`` or ``subtracts``) and its file.
  A variable created at runtime is not in the source and so not read.
- :func:`input_status_problems` checks a build's input-status record against
  the engine's inputs.
- :func:`export_schema_problems` checks an exported table's columns, dtypes
  and enum values against the variables read.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

__all__ = [
    "EXPORTED_INPUT_STATUSES",
    "INPUT_STATUSES",
    "EntityVariables",
    "export_schema_problems",
    "input_status_problems",
    "read_entity_variables",
]

#: How a build fills an engine input:
#:
#: - ``set``: mapped directly from a generator column or a source draw;
#: - ``derived``: computed from other columns by a rule;
#: - ``assumed``: carries a data assumption beyond the source;
#: - ``not_set``: not exported; the engine's default applies.
INPUT_STATUSES: Final[tuple[str, ...]] = ("set", "derived", "assumed", "not_set")

#: The statuses of inputs a build exports.
EXPORTED_INPUT_STATUSES: Final[tuple[str, ...]] = ("set", "derived", "assumed")

_DTYPE_KINDS: Final[Mapping[str, str]] = {"float": "f", "int": "i", "bool": "b"}


@dataclass(frozen=True)
class EntityVariables:
    """The variables a country package defines, as read from its source.

    ``variables`` maps each variable on the entity to its record (see
    :func:`read_entity_variables`); ``input_names`` lists the formula-less
    ones, sorted; ``variable_names`` is every variable the package defines,
    on any entity.
    """

    variables: dict[str, dict[str, Any]]
    input_names: list[str]
    variable_names: list[str]


@dataclass
class _Module:
    name: str
    path: Path
    tree: ast.Module
    #: Local name -> (module, attribute) for ``from module import attribute``.
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: Local alias -> module for ``import module`` and ``import module as m``.
    module_aliases: dict[str, str] = field(default_factory=dict)
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)


def _module_name(package_dir: Path, path: Path) -> str:
    parts = list(path.relative_to(package_dir.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(module: _Module, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = module.name.split(".")
    if module.path.name != "__init__.py":
        package = package[:-1]
    if node.level > 1:
        package = package[: len(package) - (node.level - 1)]
    return ".".join([*package, *([node.module] if node.module else [])])


def _read_modules(package_dir: Path) -> dict[str, _Module]:
    modules: dict[str, _Module] = {}
    tests_dir = package_dir / "tests"
    for path in sorted(package_dir.rglob("*.py")):
        if tests_dir in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _Module(name=_module_name(package_dir, path), path=path, tree=tree)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                source = _resolve_relative(module, node)
                for alias in node.names:
                    if alias.name != "*":
                        module.imports[alias.asname or alias.name] = (
                            source,
                            alias.name,
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module.module_aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ClassDef):
                module.classes[node.name] = node
        modules[module.name] = module
    return modules


def _base_names(node: ast.ClassDef) -> list[str]:
    names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _assignments(node: ast.ClassDef) -> dict[str, ast.expr]:
    values: dict[str, ast.expr] = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            if isinstance(statement.target, ast.Name):
                values[statement.target.id] = statement.value
    return values


def _is_formula_name(name: str) -> bool:
    return name == "formula" or name.startswith("formula_")


def _has_formula(node: ast.ClassDef) -> bool:
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_formula_name(statement.name):
                return True
    return any(
        _is_formula_name(name) or name in ("adds", "subtracts")
        for name in _assignments(node)
    )


def _enum_members(node: ast.ClassDef) -> list[str]:
    members = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    members.append(target.id)
    return members


def _resolve_class(
    modules: dict[str, _Module],
    module: _Module,
    name: str,
    seen: frozenset = frozenset(),
) -> tuple[_Module, ast.ClassDef]:
    """The class ``name`` refers to in ``module``, following its imports."""

    key = (module.name, name)
    if key in seen:
        raise ValueError(f"Import cycle resolving {name} in {module.name}.")
    if name in module.classes:
        return module, module.classes[name]
    if name in module.imports:
        source, attribute = module.imports[name]
        if source in modules:
            return _resolve_class(modules, modules[source], attribute, seen | {key})
        submodule = f"{source}.{attribute}"
        if submodule in modules:
            raise ValueError(f"{name} in {module.name} is a module, not a class.")
    raise ValueError(f"Cannot resolve class {name} from {module.name}.")


def _resolve_enum(
    modules: dict[str, _Module], module: _Module, expression: ast.expr
) -> tuple[str, str, list[str]]:
    if isinstance(expression, ast.Name):
        owner, node = _resolve_class(modules, module, expression.id)
    elif isinstance(expression, ast.Attribute) and isinstance(
        expression.value, ast.Name
    ):
        alias = module.module_aliases.get(expression.value.id)
        target = modules.get(alias) if alias else None
        if target is None:
            source = module.imports.get(expression.value.id)
            target = modules.get(".".join(source)) if source else None
        if target is None:
            raise ValueError(f"Cannot resolve {ast.unparse(expression)}.")
        owner, node = _resolve_class(modules, target, expression.attr)
    else:
        raise ValueError(f"Cannot resolve possible_values {ast.unparse(expression)}.")
    if "Enum" not in _base_names(node):
        raise ValueError(f"{node.name} in {owner.name} is not an Enum.")
    return node.name, owner.name, _enum_members(node)


def _literal_or_source(expression: ast.expr) -> Any:
    try:
        return ast.literal_eval(expression)
    except ValueError:
        return ast.unparse(expression)


def _definition_period(expression: ast.expr | None) -> str | None:
    if expression is None:
        return None
    if isinstance(expression, ast.Name):
        return expression.id.lower()
    return _literal_or_source(expression)


def _variable_record(
    modules: dict[str, _Module],
    module: _Module,
    node: ast.ClassDef,
    package_dir: Path,
) -> dict[str, Any]:
    values = _assignments(node)
    enum = None
    members: list[str] | None = None
    if "possible_values" in values:
        enum_name, enum_module, members = _resolve_enum(
            modules, module, values["possible_values"]
        )
        enum = {"name": enum_name, "module": enum_module}
    default = values.get("default_value")
    if default is None:
        default_value = None
    elif (
        members is not None
        and isinstance(default, ast.Attribute)
        and default.attr in members
    ):
        default_value = default.attr
    else:
        default_value = _literal_or_source(default)
    unit = values.get("unit")
    return {
        "default_value": default_value,
        "definition_period": _definition_period(values.get("definition_period")),
        "enum": enum,
        "enum_members": members,
        "file": module.path.relative_to(package_dir).as_posix(),
        "has_formula": _has_formula(node),
        "name": node.name,
        "unit": None if unit is None else _literal_or_source(unit),
        "value_type": ast.unparse(values["value_type"])
        if "value_type" in values
        else None,
    }


def read_entity_variables(package_dir: Path, *, entity: str) -> EntityVariables:
    """Read the variables on ``entity`` from a country package's source.

    ``package_dir`` is the package directory itself (for example
    ``policyengine-uk/policyengine_uk``); its ``tests/`` directory is skipped.
    ``entity`` is the entity class name the variables assign (for example
    ``Firm``). Each record holds ``default_value``, ``definition_period``,
    ``enum`` (``name`` and ``module``, or None), ``enum_members``, ``file``
    (relative to the package), ``has_formula``, ``name``, ``unit`` and
    ``value_type``.

    Raises ``ValueError`` when the package defines no variable on the entity,
    defines one twice, or names an enum it cannot resolve.
    """

    package_dir = Path(package_dir)
    if not package_dir.is_dir():
        raise ValueError(f"{package_dir} is not a package directory.")
    modules = _read_modules(package_dir)
    variable_names: set[str] = set()
    variables: dict[str, dict[str, Any]] = {}
    for module in modules.values():
        for node in module.classes.values():
            if "Variable" not in _base_names(node):
                continue
            variable_names.add(node.name)
            assigned = _assignments(node).get("entity")
            if isinstance(assigned, ast.Name) and assigned.id == entity:
                if node.name in variables:
                    raise ValueError(f"{entity} variable {node.name} is defined twice.")
                variables[node.name] = _variable_record(
                    modules, module, node, package_dir
                )
    if not variables:
        raise ValueError(f"{package_dir} defines no {entity} variables.")
    inputs = sorted(
        name for name, variable in variables.items() if not variable["has_formula"]
    )
    return EntityVariables(
        variables=variables,
        input_names=inputs,
        variable_names=sorted(variable_names),
    )


def input_status_problems(
    status: Mapping[str, str],
    engine_inputs: Iterable[str],
    *,
    exported_columns: Iterable[str] | None = None,
) -> list[str]:
    """What is wrong with a build's input-status record; empty when nothing.

    ``status`` maps every engine input to one of :data:`INPUT_STATUSES`. It
    must cover exactly ``engine_inputs``. With ``exported_columns`` (the
    engine inputs the fullest export writes), the inputs whose status is
    exported (:data:`EXPORTED_INPUT_STATUSES`) must be exactly those columns.
    """

    problems = []
    unknown = sorted(
        {value for value in status.values() if value not in INPUT_STATUSES}
    )
    if unknown:
        problems.append(
            f"unknown input statuses {unknown}; choose from {INPUT_STATUSES}"
        )
    inputs = set(engine_inputs)
    missing = sorted(inputs - set(status))
    if missing:
        problems.append(f"engine inputs without a status: {missing}")
    extra = sorted(set(status) - inputs)
    if extra:
        problems.append(f"statuses for names that are not engine inputs: {extra}")
    if exported_columns is not None:
        exported = set(exported_columns)
        claimed = {
            name for name, value in status.items() if value in EXPORTED_INPUT_STATUSES
        }
        unexported = sorted(claimed - exported)
        if unexported:
            problems.append(
                f"inputs marked exported that no export writes: {unexported}"
            )
        unclaimed = sorted(exported - claimed)
        if unclaimed:
            problems.append(f"exported inputs marked not_set or missing: {unclaimed}")
    return problems


def export_schema_problems(
    table: pd.DataFrame,
    variables: EntityVariables,
    *,
    columns: Iterable[str] | None = None,
) -> list[str]:
    """What in ``table`` would not load as engine inputs; empty when nothing.

    Each checked column (``columns``, else every column of ``table``) must be
    a formula-less variable in ``variables``. A float, int or bool variable
    needs a column of that dtype kind; an enum variable needs string values
    that are members of its enum.
    """

    inputs = set(variables.input_names)
    problems = []
    for column in table.columns if columns is None else columns:
        if column not in table.columns:
            problems.append(f"{column}: not in the table")
            continue
        if column not in inputs:
            problems.append(f"{column}: not a formula-less engine input")
            continue
        variable = variables.variables[column]
        value_type = variable["value_type"]
        if value_type == "Enum":
            values = table[column].to_numpy()
            if not all(isinstance(value, str) for value in values):
                problems.append(f"{column}: enum values must be member names")
                continue
            outside = sorted(set(values) - set(variable["enum_members"] or ()))
            if outside:
                problems.append(f"{column}: values {outside[:5]} are not enum members")
        elif value_type in _DTYPE_KINDS:
            kind = table[column].dtype.kind
            if kind != _DTYPE_KINDS[value_type]:
                problems.append(
                    f"{column}: dtype {table[column].dtype} for value type {value_type}"
                )
        else:
            problems.append(f"{column}: unchecked value type {value_type!r}")
    return problems
