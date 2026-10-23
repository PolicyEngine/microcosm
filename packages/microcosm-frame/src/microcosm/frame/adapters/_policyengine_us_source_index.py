"""Import-free PolicyEngine-US source inventory and static-value IR.

This private module owns the single AST inventory used by both metadata and
consumer queries.  It deliberately does not import :mod:`policyengine_us`:
ordinary variable classes and their static reference surfaces are read from
the installed wheel's source tree.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from microcosm.frame.adapters._policyengine_us_helper_reviews import (
    POLICYENGINE_US_HELPER_REVIEWS,
    HelperReview,
)
from microcosm.frame.schema import VariableMetadata

_ENTITY_KEY_BY_SOURCE_NAME: dict[str, str] = {
    "Person": "person",
    "Household": "household",
    "TaxUnit": "tax_unit",
    "SPMUnit": "spm_unit",
    "Family": "family",
    "MaritalUnit": "marital_unit",
}
_DTYPE_KIND_BY_SOURCE_NAME: dict[str, str] = {
    "float": "float",
    "int": "int",
    "bool": "bool",
    "str": "str",
}
_PERIOD_BY_SOURCE_NAME: dict[str, str] = {
    "YEAR": "year",
    "MONTH": "month",
    "ETERNITY": "point",
    "DAY": "point",
}


@dataclass(frozen=True, order=True)
class ConsumerReceipt:
    """One statically authenticated PolicyEngine-US variable reference site."""

    consumer: str
    path: str
    line: int
    kind: str
    # Exact entity object on which the variable is requested or aggregated.
    # Empty only when the static reference site exposes no resolvable entity
    # receiver; class-level ``adds`` use the owning variable's entity.
    receiver_entity: str = ""
    # Enclosing group aggregation when it differs from the direct receiver.
    # For ``spm_unit.any(person(...))``, for example, these fields are
    # ``person`` and ``spm_unit`` respectively.
    aggregation_entity: str = ""


@dataclass(frozen=True)
class _StringValue:
    """One exact string, retaining how the string was obtained."""

    value: str
    parameter_backed: bool = False
    constructed: bool = False


@dataclass(frozen=True)
class _SequenceValue:
    """An exact finite sequence (also used for finite set domains)."""

    items: tuple[_ExactValue, ...]
    set_like: bool = False


@dataclass(frozen=True)
class _MapValue:
    """An exact finite string-keyed mapping."""

    items: tuple[tuple[str, _ExactValue], ...]


@dataclass(frozen=True)
class _ParameterPathValue:
    """A parameter path that is materialized only when its values are read."""

    parts: tuple[str, ...]


@dataclass(frozen=True)
class _AlternativeValue:
    """Finite exact alternatives from an unresolved but bounded branch."""

    values: tuple[_ExactValue, ...]


@dataclass(frozen=True)
class _BoolValue:
    value: bool


@dataclass(frozen=True)
class _IntValue:
    value: int
    source: _SequenceValue | None = None


@dataclass(frozen=True)
class _EntityValue:
    """A symbolic PolicyEngine population/entity object."""

    name: str


@dataclass(frozen=True)
class _OpaqueValue:
    """A known non-string value whose internals are irrelevant to the index."""

    label: str = "opaque"


@dataclass(frozen=True)
class _UnknownValue:
    """A monotone unknown: operations must preserve, never erase, its origin."""

    origins: tuple[str, ...]


@dataclass(frozen=True)
class _LocalFunctionValue:
    """A function defined inside the function being interpreted.

    A call to it is interpreted in place, in the caller's scope (Python
    closures read enclosing names when called), so what it reads belongs to
    the enclosing consumer.
    """

    node: ast.FunctionDef | ast.AsyncFunctionDef


_ExactValue = (
    _StringValue
    | _SequenceValue
    | _MapValue
    | _ParameterPathValue
    | _AlternativeValue
    | _BoolValue
    | _IntValue
    | _EntityValue
    | _OpaqueValue
    | _UnknownValue
    | _LocalFunctionValue
)


def _unknown(*origins: str) -> _UnknownValue:
    return _UnknownValue(tuple(dict.fromkeys(origins)))


#: ``None`` as the evaluator spells a ``None`` constant. It is opaque, so it
#: still fails closed at a sink; ``is None`` tests and one-argument
#: ``dict.get`` misses resolve against it exactly.
_NONE_VALUE = _OpaqueValue("None")


def _contains_unknown(value: _ExactValue) -> bool:
    """Return whether an exact value contains any unresolved component."""

    if isinstance(value, _UnknownValue):
        return True
    if isinstance(value, _SequenceValue):
        return any(_contains_unknown(item) for item in value.items)
    if isinstance(value, _MapValue):
        return any(_contains_unknown(item) for _key, item in value.items)
    if isinstance(value, _AlternativeValue):
        return any(_contains_unknown(item) for item in value.values)
    return False


@dataclass(frozen=True)
class _SourceVariableDefinition:
    metadata: VariableMetadata
    always_computed: bool
    formula_starts: tuple[tuple[int, int, int], ...]

    @property
    def formula_owned(self) -> bool:
        return self.always_computed or bool(self.formula_starts)

    def computed_at(self, period: int | str) -> bool:
        if self.always_computed:
            return True
        at = _period_start(period)
        return any(start <= at for start in self.formula_starts)


@dataclass(frozen=True)
class _PolicyEngineUSSourceIndex:
    definitions: Mapping[str, _SourceVariableDefinition]
    consumers: Mapping[str, tuple[ConsumerReceipt, ...]]
    #: Qualified names of the package helpers the formulas call, each
    #: answered from its review.
    reviewed_helpers: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ImportBinding:
    source_module: str
    source_name: str
    bound_name: str


@dataclass(frozen=True)
class _ModuleImportBinding:
    source_module: str
    bound_path: tuple[str, ...]


@dataclass(frozen=True)
class _ModuleIR:
    """One parsed source module and its module-scoped static surfaces."""

    source_path: Path
    display_path: str
    module_name: str
    is_package: bool
    tree: ast.Module
    module_imports: tuple[_ModuleImportBinding, ...]
    imports: tuple[_ImportBinding, ...]
    star_imports: tuple[str, ...]
    classes: tuple[ast.ClassDef, ...]
    functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]


class _ParameterListResolver:
    """Read all historical list-valued parameter instants without engine import."""

    def __init__(self, parameters_root: Path) -> None:
        self._root = parameters_root
        self._cache: dict[tuple[str, ...], _ExactValue] = {}

    def resolve(
        self,
        path: tuple[str, ...],
        *,
        source_path: Path,
        line: int,
        consumer: str,
    ) -> _ExactValue:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        parameter_path = self._root.joinpath(*path).with_suffix(".yaml")
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PolicyEngine parameter-backed consumer indexing requires "
                "PyYAML from the PolicyEngine-US installation."
            ) from exc
        try:
            # BaseLoader preserves the valid PolicyEngine instant ``0000-01-01``
            # as text instead of trying to construct an invalid datetime.date.
            data = yaml.load(parameter_path.read_text(), Loader=yaml.BaseLoader)
        except (OSError, yaml.YAMLError) as exc:
            unresolved = _unknown(
                f"parameter:{'.'.join(path)}:{source_path}:{line}:{exc}"
            )
            self._cache[path] = unresolved
            return unresolved
        if not isinstance(data, Mapping):
            opaque = _OpaqueValue(f"parameter:{'.'.join(path)}")
            self._cache[path] = opaque
            return opaque
        history = data.get("values", data)
        if not isinstance(history, Mapping):
            opaque = _OpaqueValue(f"parameter:{'.'.join(path)}")
            self._cache[path] = opaque
            return opaque

        alternatives: list[_SequenceValue] = []
        for _instant, raw_value in history.items():
            if raw_value == "expected":
                continue
            value = raw_value
            if isinstance(value, Mapping):
                if value.get("expected"):
                    continue
                if "value" in value:
                    value = value["value"]
            if value is None:
                continue
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                opaque = _OpaqueValue(f"parameter:{'.'.join(path)}")
                self._cache[path] = opaque
                return opaque
            alternatives.append(
                _SequenceValue(
                    tuple(_StringValue(item, parameter_backed=True) for item in value)
                )
            )
        unique = tuple(dict.fromkeys(alternatives))
        if not unique:
            resolved: _ExactValue = _SequenceValue(())
        elif len(unique) == 1:
            resolved = unique[0]
        else:
            resolved = _AlternativeValue(unique)
        self._cache[path] = resolved
        return resolved


def _source_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _attribute_path(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        base = _attribute_path(node.value)
        if base is not None:
            return (*base, node.attr)
    return None


def _assigned_name(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target.id, statement.value
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        if statement.value is not None:
            return statement.target.id, statement.value
    return None


def _is_nonempty_source_value(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.keys)
    return True


def _formula_start(name: str) -> tuple[int, int, int] | None:
    if not name.startswith("formula_"):
        return None
    parts = name.removeprefix("formula_").split("_")
    if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise RuntimeError(f"Unsupported PolicyEngine formula name {name!r}.")
    year, *rest = (int(part) for part in parts)
    month = rest[0] if rest else 1
    day = rest[1] if len(rest) == 2 else 1
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise RuntimeError(f"Unsupported PolicyEngine formula date {name!r}.")
    return year, month, day


def _period_start(period: int | str) -> tuple[int, int, int]:
    parts = str(period).split("-", 2)
    if not parts[0].isdigit():
        raise ValueError(f"PolicyEngine period must begin with a year, got {period!r}.")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    return year, month, day


def _variable_definition(
    node: ast.ClassDef,
    *,
    source_path: Path,
) -> _SourceVariableDefinition:
    assignments = dict(
        assigned
        for statement in node.body
        if (assigned := _assigned_name(statement)) is not None
    )
    required = ("entity", "value_type", "definition_period")
    missing = [name for name in required if name not in assignments]
    if missing:
        raise RuntimeError(
            f"PolicyEngine variable {node.name!r} in {source_path} has no static "
            f"assignment(s) for {missing}."
        )
    entity_name = _source_name(assignments["entity"])
    value_type_name = _source_name(assignments["value_type"])
    period_name = _source_name(assignments["definition_period"])
    if entity_name not in _ENTITY_KEY_BY_SOURCE_NAME:
        raise RuntimeError(
            f"PolicyEngine variable {node.name!r} in {source_path} has unsupported "
            f"entity metadata {entity_name!r}."
        )
    if value_type_name is None:
        raise RuntimeError(
            f"PolicyEngine variable {node.name!r} in {source_path} has dynamic "
            "value_type metadata."
        )
    if period_name not in _PERIOD_BY_SOURCE_NAME:
        raise RuntimeError(
            f"PolicyEngine variable {node.name!r} in {source_path} has unsupported "
            f"definition_period metadata {period_name!r}."
        )

    always_computed = False
    formula_starts: list[tuple[int, int, int]] = []
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "formula":
                always_computed = True
            elif statement.name.startswith("formula_"):
                start = _formula_start(statement.name)
                if start is not None:
                    formula_starts.append(start)
            continue
        assigned = _assigned_name(statement)
        if assigned is None:
            continue
        name, value = assigned
        if name in {"formula", "formulas", "adds", "subtracts"}:
            always_computed |= _is_nonempty_source_value(value)

    return _SourceVariableDefinition(
        metadata=VariableMetadata(
            name=node.name,
            entity=_ENTITY_KEY_BY_SOURCE_NAME[entity_name],
            dtype=_DTYPE_KIND_BY_SOURCE_NAME.get(value_type_name, "str"),
            period=_PERIOD_BY_SOURCE_NAME[period_name],
        ),
        always_computed=always_computed,
        formula_starts=tuple(sorted(formula_starts)),
    )


def _is_policyengine_variable(node: ast.ClassDef) -> bool:
    return any(_source_name(base) == "Variable" for base in node.bases)


def _resolve_import_module(
    module_name: str,
    statement: ast.ImportFrom,
    *,
    is_package: bool,
) -> str | None:
    if statement.level == 0:
        return statement.module
    package = module_name.split(".") if is_package else module_name.split(".")[:-1]
    keep = len(package) - statement.level + 1
    if keep < 0:
        return None
    prefix = package[:keep]
    if statement.module:
        prefix.extend(statement.module.split("."))
    return ".".join(prefix)


def _module_ir(
    source_path: Path,
    *,
    root: Path,
    package: str,
    display_prefix: str,
) -> _ModuleIR:
    """Parse one source module below ``root``, named inside ``package``."""

    try:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(
            f"Could not index PolicyEngine-US source {source_path}."
        ) from exc
    relative_path = source_path.relative_to(root).as_posix()
    is_package = relative_path.endswith("/__init__.py") or relative_path == (
        "__init__.py"
    )
    relative_module = relative_path.removesuffix(".py")
    if relative_module == "__init__":
        relative_module = ""
    elif relative_module.endswith("/__init__"):
        relative_module = relative_module.removesuffix("/__init__")
    relative_module = relative_module.replace("/", ".")
    module_name = package
    if relative_module:
        module_name += f".{relative_module}"
    module_imports: list[_ModuleImportBinding] = []
    imports: list[_ImportBinding] = []
    star_imports: list[str] = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                bound_path = (
                    (alias.asname,)
                    if alias.asname is not None
                    else tuple(alias.name.split("."))
                )
                module_imports.append(
                    _ModuleImportBinding(
                        source_module=alias.name,
                        bound_path=bound_path,
                    )
                )
            continue
        if not isinstance(statement, ast.ImportFrom):
            continue
        source_module = _resolve_import_module(
            module_name,
            statement,
            is_package=is_package,
        )
        if source_module is None:
            continue
        for alias in statement.names:
            if alias.name == "*":
                star_imports.append(source_module)
            else:
                module_imports.append(
                    _ModuleImportBinding(
                        source_module=f"{source_module}.{alias.name}",
                        bound_path=(alias.asname or alias.name,),
                    )
                )
                imports.append(
                    _ImportBinding(
                        source_module=source_module,
                        source_name=alias.name,
                        bound_name=alias.asname or alias.name,
                    )
                )
    return _ModuleIR(
        source_path=source_path,
        display_path=f"{display_prefix}{relative_path}",
        module_name=module_name,
        is_package=is_package,
        tree=tree,
        module_imports=tuple(module_imports),
        imports=tuple(imports),
        star_imports=tuple(star_imports),
        classes=tuple(node for node in tree.body if isinstance(node, ast.ClassDef)),
        functions=tuple(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
    )


def _inventory_modules(variables_root: Path) -> dict[str, _ModuleIR]:
    modules: dict[str, _ModuleIR] = {}
    for source_path in sorted(variables_root.rglob("*.py")):
        module = _module_ir(
            source_path,
            root=variables_root,
            package="policyengine_us.variables",
            display_prefix="variables/",
        )
        modules[module.module_name] = module
    return modules


def _is_helper_module_name(name: str) -> bool:
    return (name == "policyengine_us" or name.startswith("policyengine_us.")) and not (
        name == "policyengine_us.variables"
        or name.startswith("policyengine_us.variables.")
    )


def _helper_module_path(package_root: Path, module_name: str) -> Path | None:
    parts = module_name.split(".")[1:]
    candidates = (
        (package_root.joinpath(*parts).with_suffix(".py"),) if parts else ()
    ) + (package_root.joinpath(*parts, "__init__.py"),)
    return next((path for path in candidates if path.is_file()), None)


def _imported_module_names(module: _ModuleIR) -> set[str]:
    names = set(module.star_imports)
    names.update(binding.source_module for binding in module.imports)
    # ``import a.b`` and ``from a import b`` may each name a submodule.
    names.update(binding.source_module for binding in module.module_imports)
    return names


def _inventory_helper_modules(
    package_root: Path,
    modules: Mapping[str, _ModuleIR],
) -> dict[str, _ModuleIR]:
    """Package modules outside ``variables/`` that formula modules import.

    Formulas reach PolicyEngine-US helpers (``model_api``, ``tools.general``,
    the HUD parameter loaders) through star and explicit imports, possibly
    through further package modules.  This follows those imports from every
    formula module, transitively.  The modules are inventoried to bind helper
    names and to digest helper code; their functions are never interpreted.
    """

    helper_modules: dict[str, _ModuleIR] = {}
    pending = sorted(
        {name for module in modules.values() for name in _imported_module_names(module)}
    )
    while pending:
        name = pending.pop()
        if (
            name in helper_modules
            or name in modules
            or not _is_helper_module_name(name)
        ):
            continue
        path = _helper_module_path(package_root, name)
        if path is None:
            continue
        module = _module_ir(
            path,
            root=package_root,
            package="policyengine_us",
            display_prefix="",
        )
        helper_modules[module.module_name] = module
        pending.extend(sorted(_imported_module_names(module)))
    return helper_modules


def _is_none(left: _ExactValue, right: _ExactValue) -> bool | None:
    """``left is right`` when one side is None and the other is decidable."""

    exact = (_StringValue, _SequenceValue, _MapValue, _BoolValue, _IntValue)
    for value, other in ((left, right), (right, left)):
        if value == _NONE_VALUE:
            if other == _NONE_VALUE:
                return True
            if isinstance(other, exact):
                return False
    return None


def _sequence_display(
    node: ast.List | ast.Tuple | ast.Set,
    evaluate: Callable[[ast.expr], _ExactValue],
    materialize: Callable[[_ExactValue], _ExactValue] | None = None,
) -> _ExactValue:
    """Evaluate a list, tuple or set display, splicing ``*name`` items.

    A starred item contributes its items only when it evaluates to exact
    sequences (policyengine-us 2.60 writes ``["adjusted_gross_income",
    *TAX_UNIT_ADDITIONS]`` and ``["ssi_earned_income", *ssi_sources.unearned]``);
    ``materialize`` resolves a parameter list first. A starred value with
    several exact alternatives gives one display per alternative; anything
    else leaves the display unknown.
    """
    displays: list[list[_ExactValue]] = [[]]
    for element in node.elts:
        if isinstance(element, ast.Starred):
            spliced = evaluate(element.value)
            if materialize is not None:
                spliced = materialize(spliced)
            alternatives = _alternatives(spliced)
            if _contains_unknown(spliced) or not all(
                isinstance(value, _SequenceValue) for value in alternatives
            ):
                return _unknown(f"starred:{node.lineno}")
            displays = [
                [*display, *value.items]
                for display in displays
                for value in alternatives
                if isinstance(value, _SequenceValue)
            ]
            if len(displays) > _STATE_CAP:
                return _unknown(f"starred-cap:{node.lineno}")
        else:
            item = evaluate(element)
            for display in displays:
                display.append(item)
    set_like = isinstance(node, ast.Set)
    return _make_alternative(
        tuple(_SequenceValue(tuple(display), set_like=set_like) for display in displays)
    )


def _eval_module_value(
    node: ast.expr,
    env: Mapping[str, _ExactValue],
) -> _ExactValue:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return _StringValue(node.value)
        if isinstance(node.value, bool):
            return _BoolValue(node.value)
        return _OpaqueValue(repr(node.value))
    if isinstance(node, ast.Name):
        return env.get(node.id, _unknown(f"name:{node.id}"))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return _sequence_display(node, lambda item: _eval_module_value(item, env))
    if isinstance(node, ast.Dict):
        items: list[tuple[str, _ExactValue]] = []
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                return _unknown(f"dict-unpack:{node.lineno}")
            key = _eval_module_value(key_node, env)
            if not isinstance(key, _StringValue):
                return _unknown(f"dict-key:{node.lineno}")
            items.append((key.value, _eval_module_value(value_node, env)))
        return _MapValue(tuple(items))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_module_value(node.left, env)
        right = _eval_module_value(node.right, env)
        if _contains_unknown(left) or _contains_unknown(right):
            return _unknown(f"add:{node.lineno}")
        if isinstance(left, _StringValue) and isinstance(right, _StringValue):
            return _StringValue(
                left.value + right.value,
                parameter_backed=(left.parameter_backed or right.parameter_backed),
                constructed=True,
            )
        if isinstance(left, _SequenceValue) and isinstance(right, _SequenceValue):
            return _SequenceValue(left.items + right.items)
        return _unknown(f"add:{node.lineno}")
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "split"
            and not node.args
        ):
            base = _eval_module_value(node.func.value, env)
            if isinstance(base, _StringValue):
                return _SequenceValue(
                    tuple(_StringValue(part) for part in base.value.split())
                )
        return _unknown(f"call:{node.lineno}")
    return _unknown(f"{type(node).__name__}:{getattr(node, 'lineno', 0)}")


def _module_environments(
    modules: Mapping[str, _ModuleIR],
    resolver: _ParameterListResolver,
) -> dict[str, Mapping[str, _ExactValue]]:
    """Resolve explicit source imports without sharing bare names across modules."""

    environments: dict[str, Mapping[str, _ExactValue]] = {
        name: MappingProxyType({}) for name in modules
    }
    evaluator = _ExactEvaluator()
    for _ in range(len(modules) + 1):
        changed = False
        for module_name in sorted(modules):
            module = modules[module_name]
            env: dict[str, _ExactValue] = {}
            for source_module in module.star_imports:
                if source_module in modules:
                    source_env = environments[source_module]
                    exported = source_env.get("__all__")
                    if isinstance(exported, _SequenceValue) and all(
                        isinstance(item, _StringValue) for item in exported.items
                    ):
                        export_names = {
                            item.value
                            for item in exported.items
                            if isinstance(item, _StringValue)
                        }
                    else:
                        export_names = {
                            name for name in source_env if not name.startswith("_")
                        }
                    env.update(
                        (name, value)
                        for name, value in source_env.items()
                        if name in export_names
                    )
            for binding in module.imports:
                source_env = environments.get(binding.source_module)
                if source_env is None:
                    continue
                value = source_env.get(binding.source_name)
                if value is not None:
                    env[binding.bound_name] = value
            # Python assignments shadow imports and earlier assignments.  A
            # lexical pass is exact; the outer import fixed point handles files
            # whose imported constants live in later-sorted modules.
            for statement in module.tree.body:
                assigned = _assigned_name(statement)
                if assigned is None:
                    continue
                name, expression = assigned
                env[name] = evaluator.evaluate(
                    expression,
                    {},
                    _EvaluationFrame(
                        module=module,
                        module_env=env,
                        resolver=resolver,
                        consumer="<module>",
                    ),
                )
            frozen = MappingProxyType(env)
            if dict(environments[module_name]) != env:
                environments[module_name] = frozen
                changed = True
        if not changed:
            return environments
    raise RuntimeError("PolicyEngine-US static module imports did not converge.")


@dataclass(frozen=True)
class _EvaluationFrame:
    module: _ModuleIR
    module_env: Mapping[str, _ExactValue]
    resolver: _ParameterListResolver
    consumer: str
    defer_unresolved_sinks: bool = False
    enqueue_helpers: bool = True
    # Local functions the enclosing function calls directly by name; any
    # other local function is interpreted where it is defined.
    called_local_functions: frozenset[str] = frozenset()


def _make_alternative(values: tuple[_ExactValue, ...]) -> _ExactValue:
    unique = tuple(dict.fromkeys(values))
    if not unique:
        return _SequenceValue(())
    if len(unique) == 1:
        return unique[0]
    if any(_contains_unknown(value) for value in unique):
        origins = tuple(
            origin
            for value in unique
            if isinstance(value, _UnknownValue)
            for origin in value.origins
        )
        return _unknown(*(origins or ("alternative",)))
    return _AlternativeValue(unique)


def _alternatives(value: _ExactValue) -> tuple[_ExactValue, ...]:
    if isinstance(value, _AlternativeValue):
        return value.values
    return (value,)


def _materialize_value(
    value: _ExactValue,
    *,
    frame: _EvaluationFrame,
    line: int,
) -> _ExactValue:
    if isinstance(value, _ParameterPathValue):
        return frame.resolver.resolve(
            value.parts,
            source_path=frame.module.source_path,
            line=line,
            consumer=frame.consumer,
        )
    if isinstance(value, _SequenceValue):
        return _SequenceValue(
            tuple(
                _materialize_value(item, frame=frame, line=line) for item in value.items
            ),
            set_like=value.set_like,
        )
    if isinstance(value, _MapValue):
        return _MapValue(
            tuple(
                (key, _materialize_value(item, frame=frame, line=line))
                for key, item in value.items
            )
        )
    if isinstance(value, _AlternativeValue):
        return _make_alternative(
            tuple(
                _materialize_value(item, frame=frame, line=line)
                for item in value.values
            )
        )
    return value


def _constructed_string(value: _StringValue, text: str) -> _StringValue:
    return _StringValue(
        text,
        parameter_backed=value.parameter_backed,
        constructed=True,
    )


def _value_identity(value: _ExactValue) -> object:
    """Return Python-like identity while ignoring receipt provenance flags."""

    if isinstance(value, _StringValue):
        return ("str", value.value)
    if isinstance(value, _SequenceValue):
        return ("seq", tuple(_value_identity(item) for item in value.items))
    if isinstance(value, _MapValue):
        return (
            "map",
            tuple((key, _value_identity(item)) for key, item in value.items),
        )
    if isinstance(value, _BoolValue):
        return ("bool", value.value)
    if isinstance(value, _IntValue):
        return ("int", value.value)
    return value


def _parameter_tainted(value: _ExactValue) -> _ExactValue:
    if isinstance(value, _StringValue):
        return _StringValue(
            value.value,
            parameter_backed=True,
            constructed=value.constructed,
        )
    if isinstance(value, _SequenceValue):
        return _SequenceValue(
            tuple(_parameter_tainted(item) for item in value.items),
            set_like=value.set_like,
        )
    if isinstance(value, _MapValue):
        return _MapValue(
            tuple((key, _parameter_tainted(item)) for key, item in value.items)
        )
    if isinstance(value, _AlternativeValue):
        return _make_alternative(
            tuple(_parameter_tainted(item) for item in value.values)
        )
    return value


def _combine_values(
    left: _ExactValue,
    right: _ExactValue,
    operation: str,
) -> _ExactValue:
    if _contains_unknown(left) or _contains_unknown(right):
        return _unknown(operation)
    results: list[_ExactValue] = []
    for left_value in _alternatives(left):
        for right_value in _alternatives(right):
            if operation == "add":
                if isinstance(left_value, _StringValue) and isinstance(
                    right_value, _StringValue
                ):
                    results.append(
                        _StringValue(
                            left_value.value + right_value.value,
                            parameter_backed=(
                                left_value.parameter_backed
                                or right_value.parameter_backed
                            ),
                            constructed=True,
                        )
                    )
                    continue
                if isinstance(left_value, _SequenceValue) and isinstance(
                    right_value, _SequenceValue
                ):
                    results.append(_SequenceValue(left_value.items + right_value.items))
                    continue
            elif (
                operation == "subtract"
                and isinstance(left_value, _SequenceValue)
                and isinstance(right_value, _SequenceValue)
            ):
                excluded = {_value_identity(item) for item in right_value.items}
                results.append(
                    _SequenceValue(
                        tuple(
                            item
                            for item in left_value.items
                            if _value_identity(item) not in excluded
                        ),
                        set_like=True,
                    )
                )
                continue
            results.append(_unknown(operation))
    return _make_alternative(tuple(results))


def _is_exact_data(value: _ExactValue) -> bool:
    """Whether ``value`` is plain data whose equality the evaluator decides."""

    if isinstance(value, (_StringValue, _IntValue, _BoolValue)):
        return True
    if isinstance(value, _SequenceValue):
        return all(_is_exact_data(item) for item in value.items)
    if isinstance(value, _MapValue):
        return all(_is_exact_data(item) for _key, item in value.items)
    return value == _NONE_VALUE


def _truth(value: _ExactValue) -> bool | None:
    if isinstance(value, _BoolValue):
        return value.value
    if isinstance(value, _SequenceValue):
        return bool(value.items)
    if isinstance(value, _StringValue):
        return bool(value.value)
    return None


class _ExactEvaluator:
    """Syntax-directed evaluator for the pinned wheel's finite string shapes."""

    def evaluate(
        self,
        node: ast.expr,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return _StringValue(node.value)
            if isinstance(node.value, bool):
                return _BoolValue(node.value)
            if isinstance(node.value, int):
                return _IntValue(node.value)
            return _OpaqueValue(repr(node.value))
        if isinstance(node, ast.Name):
            return local_env.get(
                node.id,
                frame.module_env.get(node.id, _unknown(f"name:{node.id}")),
            )
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return _sequence_display(
                node,
                lambda item: self.evaluate(item, local_env, frame),
                lambda value: _materialize_value(value, frame=frame, line=node.lineno),
            )
        if isinstance(node, ast.Dict):
            items: list[tuple[str, _ExactValue]] = []
            for key_node, value_node in zip(node.keys, node.values, strict=True):
                if key_node is None:
                    return _unknown(f"dict-unpack:{node.lineno}")
                key = self.evaluate(key_node, local_env, frame)
                if not isinstance(key, _StringValue):
                    return _unknown(f"dict-key:{node.lineno}")
                items.append((key.value, self.evaluate(value_node, local_env, frame)))
            return _MapValue(tuple(items))
        if isinstance(node, ast.Attribute):
            base = self.evaluate(node.value, local_env, frame)
            if isinstance(base, _ParameterPathValue):
                return _ParameterPathValue((*base.parts, node.attr))
            if isinstance(base, _EntityValue):
                if node.attr == "members":
                    return _EntityValue("person")
                if node.attr in _ENTITY_CALL_NAMES:
                    return _EntityValue(node.attr)
                return _OpaqueValue(f"entity-attribute:{node.attr}")
            return _OpaqueValue(f"attribute:{node.attr}")
        if isinstance(node, ast.Subscript):
            return self._subscript(node, local_env, frame)
        if isinstance(node, ast.BinOp):
            left = _materialize_value(
                self.evaluate(node.left, local_env, frame),
                frame=frame,
                line=node.lineno,
            )
            right = _materialize_value(
                self.evaluate(node.right, local_env, frame),
                frame=frame,
                line=node.lineno,
            )
            if isinstance(node.op, ast.Add):
                return _combine_values(left, right, "add")
            if isinstance(node.op, ast.Sub):
                return _combine_values(left, right, "subtract")
            return _OpaqueValue(type(node.op).__name__)
        if isinstance(node, ast.IfExp):
            condition = self.evaluate(node.test, local_env, frame)
            truth = _truth(condition)
            if truth is True:
                return self.evaluate(node.body, local_env, frame)
            if truth is False:
                return self.evaluate(node.orelse, local_env, frame)
            return _make_alternative(
                (
                    self.evaluate(node.body, local_env, frame),
                    self.evaluate(node.orelse, local_env, frame),
                )
            )
        if isinstance(node, ast.JoinedStr):
            return self._joined_string(node, local_env, frame)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self._comprehension(node, local_env, frame)
        if isinstance(node, ast.DictComp):
            return self._dict_comprehension(node, local_env, frame)
        if isinstance(node, ast.Compare):
            return self._compare(node, local_env, frame)
        if isinstance(node, ast.BoolOp):
            values = [self.evaluate(item, local_env, frame) for item in node.values]
            truths = [_truth(item) for item in values]
            if all(item is not None for item in truths):
                if isinstance(node.op, ast.And):
                    return _BoolValue(all(bool(item) for item in truths))
                return _BoolValue(any(bool(item) for item in truths))
            return _unknown(f"bool:{node.lineno}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = _truth(self.evaluate(node.operand, local_env, frame))
            return (
                _BoolValue(not value)
                if value is not None
                else _unknown(f"not:{node.lineno}")
            )
        if isinstance(node, ast.Call):
            return self._call(node, local_env, frame)
        return _OpaqueValue(type(node).__name__)

    def _call(
        self,
        node: ast.Call,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue:
        if isinstance(node.func, ast.Name) and node.func.id == "parameters":
            return _ParameterPathValue(())
        function_name = _source_name(node.func)
        if function_name in {"list", "tuple", "set", "sorted"} and node.args:
            value = _materialize_value(
                self.evaluate(node.args[0], local_env, frame),
                frame=frame,
                line=node.lineno,
            )
            return self._convert_collection(function_name, value, node.lineno)
        if function_name == "dict":
            return self._dict_call(node, local_env, frame)
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            base = _materialize_value(
                self.evaluate(node.func.value, local_env, frame),
                frame=frame,
                line=node.lineno,
            )
            if method in {"keys", "values", "items"} and not node.args:
                return self._mapping_view(method, base, node.lineno)
            if method in {"lower", "upper"} and not node.args:
                return self._change_case(method, base, node.lineno)
            if method == "split" and not node.args:
                return self._split(base, node.lineno)
            if method == "replace" and len(node.args) == 2:
                old = self.evaluate(node.args[0], local_env, frame)
                new = self.evaluate(node.args[1], local_env, frame)
                return self._replace(base, old, new, node.lineno)
            if method == "index" and len(node.args) == 1:
                needle = self.evaluate(node.args[0], local_env, frame)
                return self._sequence_index(base, needle, node.lineno)
            if method == "get" and len(node.args) == 2 and not node.keywords:
                key = self.evaluate(node.args[0], local_env, frame)
                default = self.evaluate(node.args[1], local_env, frame)
                return self._mapping_get(base, key, default, node.lineno)
            if method == "get" and len(node.args) == 1 and not node.keywords:
                key = self.evaluate(node.args[0], local_env, frame)
                return self._mapping_get(base, key, _NONE_VALUE, node.lineno)
        # Engine reads and numerical/model-api calls do not themselves produce
        # a string domain; sink scanning authenticates their arguments.
        return _OpaqueValue(function_name or "call")

    def _sequence_index(
        self,
        base: _ExactValue,
        needle: _ExactValue,
        line: int,
    ) -> _ExactValue:
        if _contains_unknown(base) or _contains_unknown(needle):
            return _unknown(f"index:{line}")
        results: list[_ExactValue] = []
        for base_value in _alternatives(base):
            for needle_value in _alternatives(needle):
                if not isinstance(base_value, _SequenceValue):
                    results.append(_unknown(f"index:{line}"))
                    continue
                identity = _value_identity(needle_value)
                position = next(
                    (
                        index
                        for index, item in enumerate(base_value.items)
                        if _value_identity(item) == identity
                    ),
                    None,
                )
                results.append(
                    _IntValue(position, source=base_value)
                    if position is not None
                    else _unknown(f"index-missing:{line}")
                )
        return _make_alternative(tuple(results))

    def _convert_collection(
        self,
        function_name: str,
        value: _ExactValue,
        line: int,
    ) -> _ExactValue:
        if _contains_unknown(value):
            return _unknown(f"{function_name}:{line}")
        results: list[_ExactValue] = []
        for alternative in _alternatives(value):
            if not isinstance(alternative, _SequenceValue):
                results.append(_unknown(f"{function_name}:{line}"))
                continue
            items = alternative.items
            if function_name in {"set", "sorted"}:
                items = tuple(dict.fromkeys(items))
                if function_name == "sorted" and all(
                    isinstance(item, _StringValue) for item in items
                ):
                    items = tuple(
                        sorted(
                            items,
                            key=lambda item: (
                                item.value if isinstance(item, _StringValue) else ""
                            ),
                        )
                    )
            results.append(_SequenceValue(items, set_like=function_name == "set"))
        return _make_alternative(tuple(results))

    def _dict_call(
        self,
        node: ast.Call,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue:
        items: list[tuple[str, _ExactValue]] = []
        if node.args:
            base = self.evaluate(node.args[0], local_env, frame)
            if not isinstance(base, _MapValue):
                return _unknown(f"dict:{node.lineno}")
            items.extend(base.items)
        for keyword in node.keywords:
            if keyword.arg is None:
                return _unknown(f"dict-unpack:{node.lineno}")
            items.append((keyword.arg, self.evaluate(keyword.value, local_env, frame)))
        return _MapValue(tuple(items))

    def _mapping_view(
        self,
        method: str,
        value: _ExactValue,
        line: int,
    ) -> _ExactValue:
        if _contains_unknown(value):
            return _unknown(f"{method}:{line}")
        results: list[_ExactValue] = []
        for alternative in _alternatives(value):
            if not isinstance(alternative, _MapValue):
                results.append(_unknown(f"{method}:{line}"))
                continue
            if method == "keys":
                items: tuple[_ExactValue, ...] = tuple(
                    _StringValue(key) for key, _value in alternative.items
                )
            elif method == "values":
                items = tuple(item for _key, item in alternative.items)
            else:
                items = tuple(
                    _SequenceValue((_StringValue(key), item))
                    for key, item in alternative.items
                )
            results.append(_SequenceValue(items))
        return _make_alternative(tuple(results))

    def _mapping_get(
        self,
        base: _ExactValue,
        key: _ExactValue,
        default: _ExactValue,
        line: int,
    ) -> _ExactValue:
        """Resolve ``mapping.get(key, default)`` over an exact finite mapping.

        This is :meth:`_subscript`'s mapping arm with the missing-key case
        replaced by the caller's explicit default instead of an unknown, which
        is exactly what the two-argument form means.  The one-argument form
        passes :data:`_NONE_VALUE`, which stays opaque at a sink; an ``is
        None`` test on it is exact (policyengine-us 2.60's
        ``de_itemized_deductions_indv`` skips a deduction whose
        ``EXPENSE_SOURCES.get(deduction)`` is ``None``).

        PolicyEngine-US 2.2.1 reaches this through
        ``snap_individual_utility_allowance``, whose loop over the
        parameter-backed ``utility_types`` list maps each entry through a
        module-level override table before reading it as an SPM-unit variable
        (``EXPENSE_VARIABLE_OVERRIDES.get(expense, expense)``).  The loop is
        unrolled per entry, so every lookup is a single exact key and the
        recorded consumer set is exact, not an over-approximation.
        """

        if _contains_unknown(base) or _contains_unknown(key):
            return _unknown(f"get:{line}")
        results: list[_ExactValue] = []
        for base_value in _alternatives(base):
            for key_value in _alternatives(key):
                if not (
                    isinstance(base_value, _MapValue)
                    and isinstance(key_value, _StringValue)
                ):
                    results.append(_unknown(f"get:{line}"))
                    continue
                mapping = dict(base_value.items)
                selected = mapping.get(key_value.value, default)
                results.append(
                    _parameter_tainted(selected)
                    if key_value.parameter_backed
                    else selected
                )
        return _make_alternative(tuple(results))

    def _change_case(
        self,
        method: str,
        value: _ExactValue,
        line: int,
    ) -> _ExactValue:
        if _contains_unknown(value):
            return _unknown(f"{method}:{line}")
        results: list[_ExactValue] = []
        for alternative in _alternatives(value):
            if not isinstance(alternative, _StringValue):
                results.append(_unknown(f"{method}:{line}"))
                continue
            transformed = (
                alternative.value.lower()
                if method == "lower"
                else alternative.value.upper()
            )
            results.append(_constructed_string(alternative, transformed))
        return _make_alternative(tuple(results))

    def _split(self, value: _ExactValue, line: int) -> _ExactValue:
        if _contains_unknown(value):
            return _unknown(f"split:{line}")
        results: list[_ExactValue] = []
        for alternative in _alternatives(value):
            if not isinstance(alternative, _StringValue):
                results.append(_unknown(f"split:{line}"))
                continue
            results.append(
                _SequenceValue(
                    tuple(
                        _constructed_string(alternative, part)
                        for part in alternative.value.split()
                    )
                )
            )
        return _make_alternative(tuple(results))

    def _replace(
        self,
        base: _ExactValue,
        old: _ExactValue,
        new: _ExactValue,
        line: int,
    ) -> _ExactValue:
        if any(_contains_unknown(value) for value in (base, old, new)):
            return _unknown(f"replace:{line}")
        results: list[_ExactValue] = []
        for base_value in _alternatives(base):
            for old_value in _alternatives(old):
                for new_value in _alternatives(new):
                    if not all(
                        isinstance(value, _StringValue)
                        for value in (base_value, old_value, new_value)
                    ):
                        results.append(_unknown(f"replace:{line}"))
                        continue
                    assert isinstance(base_value, _StringValue)
                    assert isinstance(old_value, _StringValue)
                    assert isinstance(new_value, _StringValue)
                    results.append(
                        _StringValue(
                            base_value.value.replace(old_value.value, new_value.value),
                            parameter_backed=(
                                base_value.parameter_backed
                                or old_value.parameter_backed
                                or new_value.parameter_backed
                            ),
                            constructed=True,
                        )
                    )
        return _make_alternative(tuple(results))

    def _subscript(
        self,
        node: ast.Subscript,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue:
        base = self.evaluate(node.value, local_env, frame)
        if isinstance(base, _ParameterPathValue):
            key = self.evaluate(node.slice, local_env, frame)
            if isinstance(key, _StringValue):
                return _ParameterPathValue((*base.parts, key.value))
        base = _materialize_value(base, frame=frame, line=node.lineno)
        if isinstance(node.slice, ast.Slice):
            return self._slice(base, node.slice, local_env, frame, node.lineno)
        key = self.evaluate(node.slice, local_env, frame)
        if (
            _contains_unknown(key)
            and isinstance(base, _MapValue)
            and base.items
            and all(isinstance(value, _StringValue) for _key, value in base.items)
        ):
            # The key is decided at run time (policyengine-us 2.60's
            # ``CAPITAL_GAINS_RESPONSE_FORM_VARIABLES[form]``, where ``form``
            # depends on parameter values), but the result is one of the
            # map's values or a KeyError, so every value is a possible read.
            return _make_alternative(tuple(value for _key, value in base.items))
        if _contains_unknown(base) or _contains_unknown(key):
            return _unknown(f"subscript:{node.lineno}")
        results: list[_ExactValue] = []
        for base_value in _alternatives(base):
            for key_value in _alternatives(key):
                if isinstance(base_value, _MapValue) and isinstance(
                    key_value, _StringValue
                ):
                    mapping = dict(base_value.items)
                    selected = mapping.get(
                        key_value.value,
                        _unknown(f"missing-key:{key_value.value}"),
                    )
                    results.append(
                        _parameter_tainted(selected)
                        if key_value.parameter_backed
                        else selected
                    )
                elif isinstance(base_value, _SequenceValue) and isinstance(
                    key_value, _IntValue
                ):
                    try:
                        results.append(base_value.items[key_value.value])
                    except IndexError:
                        results.append(_unknown(f"index:{node.lineno}"))
                elif isinstance(base_value, _StringValue) and isinstance(
                    key_value, _IntValue
                ):
                    try:
                        results.append(
                            _constructed_string(
                                base_value, base_value.value[key_value.value]
                            )
                        )
                    except IndexError:
                        results.append(_unknown(f"index:{node.lineno}"))
                else:
                    results.append(_unknown(f"subscript:{node.lineno}"))
        return _make_alternative(tuple(results))

    def _slice(
        self,
        base: _ExactValue,
        slice_node: ast.Slice,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        line: int,
    ) -> _ExactValue:
        bounds: list[tuple[_IntValue | None, _IntValue | None, _IntValue | None]] = [
            (None, None, None)
        ]
        for position, expression in enumerate(
            (slice_node.lower, slice_node.upper, slice_node.step)
        ):
            if expression is None:
                continue
            evaluated = self.evaluate(expression, local_env, frame)
            expanded: list[
                tuple[_IntValue | None, _IntValue | None, _IntValue | None]
            ] = []
            for alternative in _alternatives(evaluated):
                if not isinstance(alternative, _IntValue):
                    return _unknown(f"slice:{line}")
                for current in bounds:
                    parts = list(current)
                    parts[position] = alternative
                    expanded.append((parts[0], parts[1], parts[2]))
            bounds = expanded
        results: list[_ExactValue] = []
        for base_value in _alternatives(base):
            for lower, upper, step in bounds:
                correlated = (lower, upper, step)
                if isinstance(base_value, _SequenceValue) and any(
                    bound is not None
                    and bound.source is not None
                    and bound.source != base_value
                    for bound in correlated
                ):
                    continue
                selection = slice(
                    lower.value if lower is not None else None,
                    upper.value if upper is not None else None,
                    step.value if step is not None else None,
                )
                if isinstance(base_value, _SequenceValue):
                    results.append(
                        _SequenceValue(
                            base_value.items[selection],
                            set_like=base_value.set_like,
                        )
                    )
                elif isinstance(base_value, _StringValue):
                    results.append(
                        _constructed_string(base_value, base_value.value[selection])
                    )
                else:
                    results.append(_unknown(f"slice:{line}"))
        return _make_alternative(tuple(results))

    def _joined_string(
        self,
        node: ast.JoinedStr,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue:
        values: tuple[_ExactValue, ...] = (_StringValue(""),)
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                next_value: _ExactValue = _StringValue(part.value)
            elif (
                isinstance(part, ast.FormattedValue)
                and part.conversion == -1
                and part.format_spec is None
            ):
                next_value = self.evaluate(part.value, local_env, frame)
            else:
                return _unknown(f"f-string:{node.lineno}")
            combined: list[_ExactValue] = []
            for left in values:
                result = _combine_values(left, next_value, "add")
                combined.extend(_alternatives(result))
            values = tuple(combined)
        return _make_alternative(values)

    def _compare(
        self,
        node: ast.Compare,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return _unknown(f"compare:{node.lineno}")
        left = _materialize_value(
            self.evaluate(node.left, local_env, frame),
            frame=frame,
            line=node.lineno,
        )
        right = _materialize_value(
            self.evaluate(node.comparators[0], local_env, frame),
            frame=frame,
            line=node.lineno,
        )
        if _contains_unknown(left) or _contains_unknown(right):
            return _unknown(f"compare:{node.lineno}")
        results: list[_ExactValue] = []
        for left_value in _alternatives(left):
            for right_value in _alternatives(right):
                if isinstance(
                    node.ops[0], (ast.In, ast.NotIn, ast.Eq, ast.NotEq)
                ) and not (_is_exact_data(left_value) and _is_exact_data(right_value)):
                    # An array, engine object or other opaque value compared
                    # with anything (``period.start.year != 2021``) is not
                    # decided here: both outcomes stay possible.
                    results.append(_unknown(f"compare:{node.lineno}"))
                    continue
                if isinstance(node.ops[0], (ast.In, ast.NotIn)):
                    if isinstance(right_value, _SequenceValue):
                        result = any(
                            _value_identity(left_value) == _value_identity(item)
                            for item in right_value.items
                        )
                    elif isinstance(right_value, _MapValue) and isinstance(
                        left_value, _StringValue
                    ):
                        result = left_value.value in dict(right_value.items)
                    else:
                        results.append(_unknown(f"membership:{node.lineno}"))
                        continue
                    if isinstance(node.ops[0], ast.NotIn):
                        result = not result
                    results.append(_BoolValue(result))
                elif isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                    result = _value_identity(left_value) == _value_identity(right_value)
                    if isinstance(node.ops[0], ast.NotEq):
                        result = not result
                    results.append(_BoolValue(result))
                elif isinstance(node.ops[0], (ast.Is, ast.IsNot)):
                    # Only identity with None is decidable: None is the one
                    # opaque value this lattice names, and an exact string,
                    # number, Boolean or collection is never None.
                    result = _is_none(left_value, right_value)
                    if result is None:
                        results.append(_unknown(f"identity:{node.lineno}"))
                        continue
                    if isinstance(node.ops[0], ast.IsNot):
                        result = not result
                    results.append(_BoolValue(result))
                else:
                    results.append(_unknown(f"compare:{node.lineno}"))
        return _make_alternative(tuple(results))

    def _comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue:
        states: list[dict[str, _ExactValue]] = [dict(local_env)]
        for generator in node.generators:
            expanded: list[dict[str, _ExactValue]] = []
            for state in states:
                iterable = _materialize_value(
                    self.evaluate(generator.iter, state, frame),
                    frame=frame,
                    line=generator.iter.lineno,
                )
                for sequence in _alternatives(iterable):
                    if not isinstance(sequence, _SequenceValue):
                        return _unknown(f"comprehension:{generator.iter.lineno}")
                    for item in sequence.items:
                        bound = dict(state)
                        if not _bind_target(generator.target, item, bound):
                            return _unknown(
                                f"comprehension-bind:{generator.target.lineno}"
                            )
                        include = True
                        for condition in generator.ifs:
                            condition_value = self.evaluate(condition, bound, frame)
                            truth = _truth(condition_value)
                            if (
                                truth is None
                                and isinstance(condition_value, _AlternativeValue)
                                and all(
                                    isinstance(item, _BoolValue)
                                    for item in condition_value.values
                                )
                            ):
                                truth = any(
                                    item.value
                                    for item in condition_value.values
                                    if isinstance(item, _BoolValue)
                                )
                            if truth is None:
                                return _unknown(
                                    f"comprehension-filter:{condition.lineno}"
                                )
                            if not truth:
                                include = False
                                break
                        if include:
                            expanded.append(bound)
            states = expanded
        values = tuple(self.evaluate(node.elt, state, frame) for state in states)
        if any(_contains_unknown(value) for value in values):
            return _unknown(f"comprehension-result:{node.lineno}")
        return _SequenceValue(
            tuple(dict.fromkeys(values)) if isinstance(node, ast.SetComp) else values,
            set_like=isinstance(node, ast.SetComp),
        )

    def _dict_comprehension(
        self,
        node: ast.DictComp,
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue:
        states = self._comprehension_states(
            node.generators,
            local_env,
            frame,
        )
        if isinstance(states, _UnknownValue):
            return states
        items: list[tuple[str, _ExactValue]] = []
        for state in states:
            key = self.evaluate(node.key, state, frame)
            if not isinstance(key, _StringValue):
                return _unknown(f"dict-comprehension-key:{node.lineno}")
            items.append((key.value, self.evaluate(node.value, state, frame)))
        return _MapValue(tuple(items))

    def _comprehension_states(
        self,
        generators: list[ast.comprehension],
        local_env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> list[dict[str, _ExactValue]] | _UnknownValue:
        states: list[dict[str, _ExactValue]] = [dict(local_env)]
        for generator in generators:
            expanded: list[dict[str, _ExactValue]] = []
            for state in states:
                iterable = _materialize_value(
                    self.evaluate(generator.iter, state, frame),
                    frame=frame,
                    line=generator.iter.lineno,
                )
                for sequence in _alternatives(iterable):
                    if not isinstance(sequence, _SequenceValue):
                        return _unknown(f"comprehension:{generator.iter.lineno}")
                    for item in sequence.items:
                        bound = dict(state)
                        if not _bind_target(generator.target, item, bound):
                            return _unknown(
                                f"comprehension-bind:{generator.target.lineno}"
                            )
                        include = True
                        for condition in generator.ifs:
                            truth = _truth(self.evaluate(condition, bound, frame))
                            if truth is None:
                                return _unknown(
                                    f"comprehension-filter:{condition.lineno}"
                                )
                            if not truth:
                                include = False
                                break
                        if include:
                            expanded.append(bound)
            states = expanded
        return states


def _bind_target(
    target: ast.expr,
    value: _ExactValue,
    env: dict[str, _ExactValue],
) -> bool:
    if isinstance(target, ast.Name):
        env[target.id] = value
        return True
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, _SequenceValue):
        if len(target.elts) != len(value.items):
            return False
        return all(
            _bind_target(element, item, env)
            for element, item in zip(target.elts, value.items, strict=True)
        )
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
        value, _AlternativeValue
    ):
        # Each position takes the alternatives of that position; this keeps
        # every possible value of each name, not which ones occur together.
        if not all(
            isinstance(item, _SequenceValue) and len(item.items) == len(target.elts)
            for item in value.values
        ):
            return False
        return all(
            _bind_target(
                element,
                _make_alternative(
                    tuple(
                        item.items[position]
                        for item in value.values
                        if isinstance(item, _SequenceValue)
                    )
                ),
                env,
            )
            for position, element in enumerate(target.elts)
        )
    return False


_ENTITY_CALL_NAMES = frozenset(
    {"person", "household", "tax_unit", "spm_unit", "family", "marital_unit"}
)
_ENTITY_AGGREGATION_METHODS = frozenset({"all", "any", "max", "min", "sum"})
_REFERENCE_HELPER_ARGUMENTS: dict[str, int] = {
    # PolicyEngine-Core logical aggregators share ``add``'s
    # ``(entity, period, variables)`` signature.  PolicyEngine-US's own
    # helpers are answered from POLICYENGINE_US_HELPER_REVIEWS instead.
    "and_": 2,
    "any_": 2,
    "for_each_variable": 2,
    "multiply": 2,
    "or_": 2,
}
#: Simulation and population methods that read a variable named by their
#: first argument: ``branch.calculate(name, period)``, ``get_holder(name)``,
#: and ``simulation.get_array(name, period)`` (a holder's own
#: ``get_array(period, branch)`` is skipped; its ``get_holder`` call names it).
_SIMULATION_READ_METHODS = frozenset({"calculate", "get_array", "get_holder"})
_STATE_CAP = 20_000


@dataclass(frozen=True, order=True)
class _FunctionID:
    module_name: str
    function_name: str


@dataclass(frozen=True)
class _FunctionIR:
    identifier: _FunctionID
    module: _ModuleIR
    node: ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class _FunctionContext:
    identifier: _FunctionID
    args: tuple[_ExactValue, ...]
    keywords: tuple[tuple[str, _ExactValue], ...]


@dataclass
class _Flow:
    env: dict[str, _ExactValue]
    signal: str = "normal"
    # What a ``return`` flow returns; None on every other flow.
    value: _ExactValue | None = None


def _function_parameters(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.arg, ...]:
    return tuple((*function.args.posonlyargs, *function.args.args))


def _is_local_name(call: ast.Call, env: Mapping[str, _ExactValue]) -> bool:
    """Whether a call's bare callee name is a local, not a module function."""

    return isinstance(call.func, ast.Name) and call.func.id in env


def _bound_parameters(
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> tuple[str, ...]:
    arguments = function.args
    names = [
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    ]
    for argument in (arguments.vararg, arguments.kwarg):
        if argument is not None:
            names.append(argument.arg)
    return tuple(names)


def _directly_called_local_functions(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Functions defined inside ``function`` that it only ever calls by name.

    Those are interpreted at each call.  A local function that is also passed
    or returned as a value, or never called, is interpreted where it is
    defined instead, with unknown arguments.
    """

    local = {
        node.name
        for node in ast.walk(function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node is not function
    }
    callees = {
        id(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    loads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in local
    ]
    called = {node.id for node in loads if id(node) in callees}
    escaping = {node.id for node in loads if id(node) not in callees}
    return frozenset(called - escaping)


def _function_bindings(
    modules: Mapping[str, _ModuleIR],
    module_envs: Mapping[str, Mapping[str, _ExactValue]],
) -> tuple[
    dict[_FunctionID, _FunctionIR],
    dict[str, Mapping[str, _FunctionID]],
]:
    functions: dict[_FunctionID, _FunctionIR] = {}
    for module in modules.values():
        for node in module.functions:
            identifier = _FunctionID(module.module_name, node.name)
            functions[identifier] = _FunctionIR(identifier, module, node)

    bindings: dict[str, Mapping[str, _FunctionID]] = {
        module_name: MappingProxyType(
            {
                function.node.name: identifier
                for identifier, function in functions.items()
                if identifier.module_name == module_name
            }
        )
        for module_name in modules
    }
    for _ in range(len(modules) + 1):
        changed = False
        for module_name in sorted(modules):
            module = modules[module_name]
            current: dict[str, _FunctionID] = {}
            for source_module in module.star_imports:
                source_bindings = bindings.get(source_module, {})
                exported = module_envs.get(source_module, {}).get("__all__")
                if isinstance(exported, _SequenceValue) and all(
                    isinstance(item, _StringValue) for item in exported.items
                ):
                    export_names = {
                        item.value
                        for item in exported.items
                        if isinstance(item, _StringValue)
                    }
                else:
                    export_names = {
                        name for name in source_bindings if not name.startswith("_")
                    }
                current.update(
                    (name, identifier)
                    for name, identifier in source_bindings.items()
                    if name in export_names
                )
            for binding in module.imports:
                source_identifier = bindings.get(binding.source_module, {}).get(
                    binding.source_name
                )
                if source_identifier is not None:
                    current[binding.bound_name] = source_identifier
            for function in module.functions:
                current[function.name] = _FunctionID(module_name, function.name)
            if dict(bindings[module_name]) != current:
                bindings[module_name] = MappingProxyType(current)
                changed = True
        if not changed:
            return functions, bindings
    raise RuntimeError("PolicyEngine-US source function imports did not converge.")


def _code_dump(node: ast.AST) -> str:
    """``node``'s code without docstrings, which do not change what it reads."""

    stripped = copy.deepcopy(node)
    for item in ast.walk(stripped):
        if isinstance(
            item, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = item.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                item.body = body[1:] or [ast.Pass()]
    return ast.dump(stripped, include_attributes=False)


def _module_assignment(
    modules: Mapping[str, _ModuleIR],
    module_name: str,
    name: str,
    seen: set[tuple[str, str]],
) -> list[str]:
    """The top-level assignment of ``name`` that ``module_name`` sees."""

    if (module_name, name) in seen or module_name not in modules:
        return []
    seen.add((module_name, name))
    module = modules[module_name]
    found: list[str] = []
    for statement in module.tree.body:
        assigned = _assigned_name(statement)
        if assigned is not None and assigned[0] == name:
            found = [f"{module_name}.{name}\n{_code_dump(statement)}"]
    if found:
        return found
    for binding in module.imports:
        if binding.bound_name == name:
            return _module_assignment(
                modules, binding.source_module, binding.source_name, seen
            )
    for source_module in reversed(module.star_imports):
        found = _module_assignment(modules, source_module, name, seen)
        if found:
            return found
    return []


def _helper_digest(
    identifier: _FunctionID,
    modules: Mapping[str, _ModuleIR],
    functions: Mapping[_FunctionID, _FunctionIR],
    bindings: Mapping[str, Mapping[str, _FunctionID]],
) -> str:
    """SHA-256 of a helper's code, the package functions it uses and their constants.

    Every function the helper names, directly or through another, and every
    module-level constant those functions read is included, so a helper whose
    reads could change gets a new digest.
    """

    parts: set[str] = set()
    seen_constants: set[tuple[str, str]] = set()
    seen: set[_FunctionID] = set()
    pending = [identifier]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        function = functions.get(current)
        if function is None:
            continue
        parts.add(
            f"{current.module_name}.{current.function_name}\n"
            f"{_code_dump(function.node)}"
        )
        names = bindings.get(current.module_name, {})
        for node in ast.walk(function.node):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            callee = names.get(node.id)
            if callee is not None:
                pending.append(callee)
                continue
            parts.update(
                _module_assignment(
                    modules, current.module_name, node.id, seen_constants
                )
            )
    return hashlib.sha256("\n\n".join(sorted(parts)).encode()).hexdigest()


class _ConsumerInterpreter:
    """Sequential exact-state interpreter for PolicyEngine consumer receipts."""

    def __init__(
        self,
        modules: Mapping[str, _ModuleIR],
        module_envs: Mapping[str, Mapping[str, _ExactValue]],
        resolver: _ParameterListResolver,
        definitions: Mapping[str, _SourceVariableDefinition],
        *,
        helper_modules: Mapping[str, _ModuleIR] | None = None,
        helper_reviews: Mapping[str, HelperReview] | None = None,
    ) -> None:
        self._modules = modules
        self._module_envs = module_envs
        self._resolver = resolver
        self._definitions = definitions
        self._evaluator = _ExactEvaluator()
        self._helper_modules = MappingProxyType(dict(helper_modules or {}))
        self._helper_reviews = (
            POLICYENGINE_US_HELPER_REVIEWS
            if helper_reviews is None
            else MappingProxyType(dict(helper_reviews))
        )
        functions, bindings = _function_bindings(
            {**modules, **self._helper_modules},
            module_envs,
        )
        # Formula-module functions are interpreted; package helpers outside
        # ``variables/`` are bound by name and answered from their reviews.
        self._functions = {
            identifier: function
            for identifier, function in functions.items()
            if identifier.module_name in modules
        }
        self._bindings = {
            module_name: MappingProxyType(
                {
                    name: identifier
                    for name, identifier in names.items()
                    if identifier.module_name in modules
                }
            )
            for module_name, names in bindings.items()
            if module_name in modules
        }
        self._all_functions = functions
        self._helper_bindings = {
            module_name: MappingProxyType(
                {
                    name: identifier
                    for name, identifier in names.items()
                    if identifier.module_name in self._helper_modules
                }
            )
            for module_name, names in bindings.items()
        }
        self._all_bindings = bindings
        self._checked_helpers: set[_FunctionID] = set()
        self.reviewed_helpers: set[str] = set()
        self._helper_digests: dict[_FunctionID, str] = {}
        self._local_calls: set[tuple[int, object]] = set()
        self._receipts: dict[str, set[ConsumerReceipt]] = {}
        self._queue: list[_FunctionContext] = []
        self._seen_contexts: set[_FunctionContext] = set()
        self._return_values: dict[_FunctionContext, _ExactValue] = {}
        self._returning: set[_FunctionContext] = set()

    def run(self) -> Mapping[str, tuple[ConsumerReceipt, ...]]:
        for module_name in sorted(self._modules):
            module = self._modules[module_name]
            for function in module.functions:
                args = tuple(
                    self._root_argument(argument.arg)
                    for argument in _function_parameters(function)
                )
                self._analyze_function(
                    module=module,
                    function=function,
                    consumer=function.name,
                    owner_name=None,
                    args=args,
                    keywords=(),
                    defer_unresolved_sinks=True,
                    enqueue_helpers=False,
                )
            for owner in module.classes:
                if not _is_policyengine_variable(owner):
                    continue
                self._scan_class_attributes(module, owner)
                for function in owner.body:
                    if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = tuple(
                            self._root_argument(argument.arg)
                            for argument in _function_parameters(function)
                        )
                        self._analyze_function(
                            module=module,
                            function=function,
                            consumer=owner.name,
                            owner_name=owner.name,
                            args=args,
                            keywords=(),
                        )

        queue_index = 0
        while queue_index < len(self._queue):
            context = self._queue[queue_index]
            queue_index += 1
            function = self._functions[context.identifier]
            self._analyze_function(
                module=function.module,
                function=function.node,
                consumer=function.node.name,
                owner_name=None,
                args=context.args,
                keywords=context.keywords,
            )
        return MappingProxyType(
            {target: tuple(sorted(items)) for target, items in self._receipts.items()}
        )

    @staticmethod
    def _root_argument(name: str) -> _ExactValue:
        if name in _ENTITY_CALL_NAMES:
            return _EntityValue(name)
        if name in {"period", "parameters"}:
            return _OpaqueValue(name)
        return _unknown(f"root-argument:{name}")

    def _frame(self, module: _ModuleIR, consumer: str) -> _EvaluationFrame:
        return _EvaluationFrame(
            module=module,
            module_env=self._module_envs[module.module_name],
            resolver=self._resolver,
            consumer=consumer,
        )

    def _scan_class_attributes(
        self,
        module: _ModuleIR,
        owner: ast.ClassDef,
    ) -> None:
        frame = self._frame(module, owner.name)
        for statement in owner.body:
            assigned = _assigned_name(statement)
            if assigned is None:
                continue
            name, expression = assigned
            if name in {"adds", "subtracts"}:
                if (
                    isinstance(expression, ast.Constant)
                    and isinstance(expression.value, str)
                    and "." in expression.value
                ):
                    value = self._resolver.resolve(
                        tuple(expression.value.split(".")),
                        source_path=module.source_path,
                        line=expression.lineno,
                        consumer=owner.name,
                    )
                else:
                    value = self._evaluator.evaluate(expression, {}, frame)
                self._record_receipts(
                    value,
                    expression=expression,
                    base_kind=name,
                    module=module,
                    consumer=owner.name,
                    owner_name=owner.name,
                    frame=frame,
                    receiver_entity=self._definitions[owner.name].metadata.entity,
                    aggregation_entity=(self._definitions[owner.name].metadata.entity),
                )
            elif (
                name == "defined_for"
                and isinstance(expression, ast.Constant)
                and isinstance(expression.value, str)
            ):
                self._record_receipts(
                    _StringValue(expression.value),
                    expression=expression,
                    base_kind="defined_for",
                    module=module,
                    consumer=owner.name,
                    owner_name=owner.name,
                    frame=frame,
                )
            else:
                self._scan_class_attribute_calls(module, owner, expression, frame)

    def _scan_class_attribute_calls(
        self,
        module: _ModuleIR,
        owner: ast.ClassDef,
        expression: ast.expr,
        frame: _EvaluationFrame,
    ) -> None:
        """Record helper factories a class body calls (``formula = f(...)``).

        ``all_of_variables([...])`` and ``sum_among_non_dependents(name)``
        build the variable's formula from variable names.  A class-body call
        to a formula-module function would build one the index cannot see, so
        it fails closed.
        """

        for node in ast.walk(expression):
            if not isinstance(node, ast.Call):
                continue
            helper = self._package_helper(node, {}, frame)
            if helper is not None:
                self._record_package_helper(
                    helper,
                    node,
                    {},
                    frame,
                    owner.name,
                    self._definitions[owner.name].metadata.entity,
                )
            elif self._helper_identifier(node, frame) is not None:
                self._raise_unresolved(node, frame)

    def _analyze_function(
        self,
        *,
        module: _ModuleIR,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        consumer: str,
        owner_name: str | None,
        args: tuple[_ExactValue, ...],
        keywords: tuple[tuple[str, _ExactValue], ...],
        defer_unresolved_sinks: bool = False,
        enqueue_helpers: bool = True,
    ) -> list[_Flow]:
        env: dict[str, _ExactValue] = {}
        parameters = _function_parameters(function)
        keyword_values = dict(keywords)
        for index, parameter in enumerate(parameters):
            if index < len(args):
                env[parameter.arg] = args[index]
            elif parameter.arg in keyword_values:
                env[parameter.arg] = keyword_values[parameter.arg]
            else:
                env[parameter.arg] = self._root_argument(parameter.arg)
        for parameter in function.args.kwonlyargs:
            env[parameter.arg] = keyword_values.get(
                parameter.arg, self._root_argument(parameter.arg)
            )
        frame = _EvaluationFrame(
            module=module,
            module_env=self._module_envs[module.module_name],
            resolver=self._resolver,
            consumer=consumer,
            defer_unresolved_sinks=defer_unresolved_sinks,
            enqueue_helpers=enqueue_helpers,
            called_local_functions=_directly_called_local_functions(function),
        )
        return self._analyze_block(
            function.body,
            [_Flow(env)],
            frame=frame,
            owner_name=owner_name,
        )

    def _analyze_block(
        self,
        statements: list[ast.stmt],
        flows: list[_Flow],
        *,
        frame: _EvaluationFrame,
        owner_name: str | None,
    ) -> list[_Flow]:
        current = flows
        for statement in statements:
            next_flows: list[_Flow] = []
            for flow in current:
                if flow.signal != "normal":
                    next_flows.append(flow)
                    continue
                next_flows.extend(
                    self._analyze_statement(
                        statement,
                        flow,
                        frame=frame,
                        owner_name=owner_name,
                    )
                )
            current = self._dedupe_flows(next_flows, frame)
        return current

    def _analyze_statement(
        self,
        statement: ast.stmt,
        flow: _Flow,
        *,
        frame: _EvaluationFrame,
        owner_name: str | None,
    ) -> list[_Flow]:
        env = flow.env
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            expression = statement.value
            if expression is None:
                return [flow]
            self._scan_expression(expression, env, frame, owner_name)
            value = self._evaluator.evaluate(expression, env, frame)
            if isinstance(expression, ast.Call) and isinstance(value, _OpaqueValue):
                returned = self._helper_return_value(expression, env, frame)
                if returned is not None:
                    value = returned
            updated = dict(env)
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                if not self._assign_value(target, value, updated, frame):
                    self._assign_unknown(target, updated, statement.lineno)
            return [_Flow(updated)]
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            updated = dict(env)
            updated[statement.name] = _LocalFunctionValue(statement)
            if statement.name not in frame.called_local_functions:
                self._analyze_local_function(
                    statement,
                    {
                        name: _unknown(f"local-argument:{name}")
                        for name in _bound_parameters(statement)
                    },
                    updated,
                    frame,
                    owner_name,
                )
            return [_Flow(updated)]
        if isinstance(statement, ast.AugAssign):
            self._scan_expression(statement.value, env, frame, owner_name)
            previous = self._evaluator.evaluate(statement.target, env, frame)
            value = self._evaluator.evaluate(statement.value, env, frame)
            if isinstance(statement.op, ast.Add):
                updated_value = _combine_values(previous, value, "add")
            elif isinstance(statement.op, ast.Sub):
                updated_value = _combine_values(previous, value, "subtract")
            else:
                updated_value = _OpaqueValue(type(statement.op).__name__)
            updated = dict(env)
            if not self._assign_value(statement.target, updated_value, updated, frame):
                self._assign_unknown(statement.target, updated, statement.lineno)
            return [_Flow(updated)]
        if isinstance(statement, ast.Expr):
            self._scan_expression(statement.value, env, frame, owner_name)
            return [
                _Flow(
                    self._apply_mutation(statement.value, env, frame),
                )
            ]
        if isinstance(statement, ast.If):
            self._scan_expression(statement.test, env, frame, owner_name)
            truth = _truth(self._evaluator.evaluate(statement.test, env, frame))
            if truth is True:
                return self._analyze_block(
                    statement.body,
                    [_Flow(dict(env))],
                    frame=frame,
                    owner_name=owner_name,
                )
            if truth is False:
                return self._analyze_block(
                    statement.orelse,
                    [_Flow(dict(env))],
                    frame=frame,
                    owner_name=owner_name,
                )
            return [
                *self._analyze_block(
                    statement.body,
                    [_Flow(dict(env))],
                    frame=frame,
                    owner_name=owner_name,
                ),
                *self._analyze_block(
                    statement.orelse,
                    [_Flow(dict(env))],
                    frame=frame,
                    owner_name=owner_name,
                ),
            ]
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            return self._analyze_for(statement, flow, frame, owner_name)
        if isinstance(statement, ast.While):
            self._scan_expression(statement.test, env, frame, owner_name)
            return [
                _Flow(dict(env)),
                *self._analyze_block(
                    statement.body,
                    [_Flow(dict(env))],
                    frame=frame,
                    owner_name=owner_name,
                ),
            ]
        if isinstance(statement, (ast.Return, ast.Yield, ast.YieldFrom)):
            value = getattr(statement, "value", None)
            if value is not None:
                self._scan_expression(value, env, frame, owner_name)
            returned = (
                self._evaluator.evaluate(value, env, frame)
                if value is not None
                else _NONE_VALUE
            )
            if not isinstance(statement, ast.Return):
                returned = _unknown(f"yield:{statement.lineno}")
            return [_Flow(dict(env), "return", returned)]
        if isinstance(statement, ast.Break):
            return [_Flow(dict(env), "break")]
        if isinstance(statement, ast.Continue):
            return [_Flow(dict(env), "continue")]
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                self._scan_expression(item.context_expr, env, frame, owner_name)
            return self._analyze_block(
                statement.body,
                [_Flow(dict(env))],
                frame=frame,
                owner_name=owner_name,
            )
        if isinstance(statement, ast.Try):
            branches = [
                self._analyze_block(
                    statement.body,
                    [_Flow(dict(env))],
                    frame=frame,
                    owner_name=owner_name,
                )
            ]
            branches.extend(
                self._analyze_block(
                    handler.body,
                    [_Flow(dict(env))],
                    frame=frame,
                    owner_name=owner_name,
                )
                for handler in statement.handlers
            )
            merged = [flow for branch in branches for flow in branch]
            if statement.finalbody:
                merged = self._analyze_block(
                    statement.finalbody,
                    merged,
                    frame=frame,
                    owner_name=owner_name,
                )
            return merged
        return [flow]

    def _analyze_for(
        self,
        statement: ast.For | ast.AsyncFor,
        flow: _Flow,
        frame: _EvaluationFrame,
        owner_name: str | None,
    ) -> list[_Flow]:
        self._scan_expression(statement.iter, flow.env, frame, owner_name)
        iterable = _materialize_value(
            self._evaluator.evaluate(statement.iter, flow.env, frame),
            frame=frame,
            line=statement.iter.lineno,
        )
        results: list[_Flow] = []
        alternatives = _alternatives(iterable)
        if any(not isinstance(value, _SequenceValue) for value in alternatives):
            alternatives = (_SequenceValue((_unknown(f"loop:{statement.lineno}"),)),)
        for alternative in alternatives:
            assert isinstance(alternative, _SequenceValue)
            active = [_Flow(dict(flow.env))]
            stopped: list[_Flow] = []
            for item in alternative.items:
                iteration: list[_Flow] = []
                for current in active:
                    bound = dict(current.env)
                    if not _bind_target(statement.target, item, bound):
                        self._assign_unknown(
                            statement.target, bound, statement.target.lineno
                        )
                    body_flows = self._analyze_block(
                        statement.body,
                        [_Flow(bound)],
                        frame=frame,
                        owner_name=owner_name,
                    )
                    for body_flow in body_flows:
                        if body_flow.signal == "break":
                            stopped.append(_Flow(body_flow.env))
                        elif body_flow.signal == "continue":
                            iteration.append(_Flow(body_flow.env))
                        else:
                            iteration.append(body_flow)
                active = self._dedupe_flows(iteration, frame)
            if statement.orelse:
                active = self._analyze_block(
                    statement.orelse,
                    active,
                    frame=frame,
                    owner_name=owner_name,
                )
            results.extend((*active, *stopped))
        return results

    def _scan_expression(
        self,
        expression: ast.expr,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
        aggregation_entity: str = "",
    ) -> None:
        if isinstance(expression, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            self._scan_comprehension(
                expression,
                env,
                frame,
                owner_name,
                aggregation_entity,
            )
            return
        if isinstance(expression, ast.Lambda):
            # Whatever the body reads is read by the enclosing consumer; the
            # parameters are unknown, so a read named by one fails closed.
            lambda_env = dict(env)
            for parameter in _bound_parameters(expression):
                lambda_env[parameter] = _unknown(f"lambda-argument:{parameter}")
            self._scan_expression(
                expression.body,
                lambda_env,
                frame,
                owner_name,
                aggregation_entity,
            )
            return
        if isinstance(expression, ast.Call):
            function_name = _source_name(expression.func)
            callee = self._evaluator.evaluate(expression.func, env, frame)
            nested_aggregation_entity = aggregation_entity
            if (
                isinstance(expression.func, ast.Attribute)
                and expression.func.attr in _ENTITY_AGGREGATION_METHODS
            ):
                aggregation_receiver = self._evaluator.evaluate(
                    expression.func.value,
                    env,
                    frame,
                )
                if isinstance(aggregation_receiver, _EntityValue):
                    nested_aggregation_entity = aggregation_receiver.name
            if (
                function_name in _ENTITY_CALL_NAMES or isinstance(callee, _EntityValue)
            ) and expression.args:
                receiver_entity = (
                    callee.name
                    if isinstance(callee, _EntityValue)
                    else function_name or ""
                )
                self._record_expression(
                    expression.args[0],
                    env=env,
                    frame=frame,
                    owner_name=owner_name,
                    base_kind="entity_call",
                    receiver_entity=receiver_entity,
                    aggregation_entity=aggregation_entity,
                )
            elif isinstance(expression.func, ast.Name) and function_name in {
                "add",
                "subtract",
            }:
                variable_expression = (
                    expression.args[2]
                    if len(expression.args) >= 3
                    else next(
                        (
                            keyword.value
                            for keyword in expression.keywords
                            if keyword.arg == "variables"
                        ),
                        None,
                    )
                )
                if variable_expression is None:
                    self._raise_unresolved(expression, frame)
                receiver = (
                    self._evaluator.evaluate(expression.args[0], env, frame)
                    if expression.args
                    else _unknown(f"receiver:{expression.lineno}")
                )
                self._record_expression(
                    variable_expression,
                    env=env,
                    frame=frame,
                    owner_name=owner_name,
                    base_kind=function_name,
                    receiver_entity=(
                        receiver.name if isinstance(receiver, _EntityValue) else ""
                    ),
                    aggregation_entity=(
                        receiver.name if isinstance(receiver, _EntityValue) else ""
                    ),
                )
            elif (
                package_helper := self._package_helper(expression, env, frame)
            ) is not None:
                self._record_package_helper(
                    package_helper,
                    expression,
                    env,
                    frame,
                    owner_name,
                    aggregation_entity,
                )
            elif isinstance(callee, _LocalFunctionValue):
                self._call_local_function(callee, expression, env, frame, owner_name)
            elif (
                isinstance(expression.func, ast.Attribute)
                and expression.func.attr in _SIMULATION_READ_METHODS
            ):
                self._record_simulation_read(expression, env, frame, owner_name)
            else:
                helper_argument = _REFERENCE_HELPER_ARGUMENTS.get(function_name or "")
                if helper_argument is not None and len(expression.args) > (
                    helper_argument
                ):
                    self._record_expression(
                        expression.args[helper_argument],
                        env=env,
                        frame=frame,
                        owner_name=owner_name,
                        base_kind="helper_call",
                        aggregation_entity=aggregation_entity,
                    )
            self._enqueue_helper(expression, env, frame)
        else:
            nested_aggregation_entity = aggregation_entity
        for child in ast.iter_child_nodes(expression):
            if isinstance(child, ast.expr):
                self._scan_expression(
                    child,
                    env,
                    frame,
                    owner_name,
                    nested_aggregation_entity,
                )

    def _scan_comprehension(
        self,
        expression: ast.ListComp | ast.SetComp | ast.GeneratorExp,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
        aggregation_entity: str = "",
    ) -> None:
        states: list[dict[str, _ExactValue]] = [dict(env)]
        for generator in expression.generators:
            expanded: list[dict[str, _ExactValue]] = []
            for state in states:
                self._scan_expression(
                    generator.iter,
                    state,
                    frame,
                    owner_name,
                    aggregation_entity,
                )
                iterable = _materialize_value(
                    self._evaluator.evaluate(generator.iter, state, frame),
                    frame=frame,
                    line=generator.iter.lineno,
                )
                for sequence in _alternatives(iterable):
                    if not isinstance(sequence, _SequenceValue):
                        sequence = _SequenceValue(
                            (_unknown(f"comprehension:{generator.iter.lineno}"),)
                        )
                    for item in sequence.items:
                        bound = dict(state)
                        if not _bind_target(generator.target, item, bound):
                            self._assign_unknown(
                                generator.target,
                                bound,
                                generator.target.lineno,
                            )
                        include = True
                        for condition in generator.ifs:
                            self._scan_expression(
                                condition,
                                bound,
                                frame,
                                owner_name,
                                aggregation_entity,
                            )
                            truth = _truth(
                                self._evaluator.evaluate(condition, bound, frame)
                            )
                            if truth is False:
                                include = False
                                break
                        if include:
                            expanded.append(bound)
            states = expanded
        for state in states:
            self._scan_expression(
                expression.elt,
                state,
                frame,
                owner_name,
                aggregation_entity,
            )

    def _call_local_function(
        self,
        function: _LocalFunctionValue,
        call: ast.Call,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
    ) -> None:
        """Interpret a local function's body for one call, in the caller's scope."""

        node = function.node
        parameters = _bound_parameters(node)
        bound: dict[str, _ExactValue] = {
            name: _unknown(f"local-argument:{name}") for name in parameters
        }
        if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
            keyword.arg is None for keyword in call.keywords
        ):
            self._raise_unresolved(call, frame)
        positional = (*node.args.posonlyargs, *node.args.args)
        for argument, value in zip(positional, call.args, strict=False):
            bound[argument.arg] = self._evaluator.evaluate(value, env, frame)
        for keyword in call.keywords:
            if keyword.arg in bound:
                bound[keyword.arg] = self._evaluator.evaluate(keyword.value, env, frame)
        self._analyze_local_function(node, bound, env, frame, owner_name)

    def _analyze_local_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        arguments: Mapping[str, _ExactValue],
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
    ) -> None:
        key = (
            id(node),
            tuple(
                (name, _value_identity(value))
                for name, value in sorted(arguments.items())
            ),
        )
        if key in self._local_calls:
            # The same call is already being interpreted further up (a
            # recursive local function); its reads are recorded there.
            return
        self._local_calls.add(key)
        try:
            scope = dict(env)
            scope[node.name] = _LocalFunctionValue(node)
            scope.update(arguments)
            self._analyze_block(
                node.body,
                [_Flow(scope)],
                frame=frame,
                owner_name=owner_name,
            )
        finally:
            self._local_calls.discard(key)

    def _record_simulation_read(
        self,
        call: ast.Call,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
    ) -> None:
        """Record ``calculate``/``get_holder``/``get_array`` reads by name."""

        assert isinstance(call.func, ast.Attribute)
        if call.func.attr == "get_array":
            receiver = self._evaluator.evaluate(call.func.value, env, frame)
            if isinstance(receiver, _OpaqueValue) and receiver.label == "get_holder":
                return
        keyword_name = "variable_name" if call.func.attr == "calculate" else "variable"
        name = (
            call.args[0]
            if call.args and not isinstance(call.args[0], ast.Starred)
            else next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == keyword_name
                ),
                None,
            )
        )
        if name is None:
            self._raise_unresolved(call, frame)
        self._record_expression(
            name,
            env=env,
            frame=frame,
            owner_name=owner_name,
            base_kind="simulation_read",
        )

    def _package_helper(
        self,
        call: ast.Call,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _FunctionID | None:
        """The PolicyEngine-US helper outside ``variables/`` a call invokes."""

        module_name = frame.module.module_name
        if isinstance(call.func, ast.Name):
            name = call.func.id
            if (
                name in env
                or name in frame.module_env
                or name in self._bindings.get(module_name, {})
            ):
                return None
            return self._helper_bindings.get(module_name, {}).get(name)
        if not isinstance(call.func, ast.Attribute):
            return None
        base_path = _attribute_path(call.func.value)
        if base_path is None or base_path[0] in env:
            return None
        for binding in reversed(frame.module.module_imports):
            if base_path[: len(binding.bound_path)] != binding.bound_path:
                continue
            suffix = base_path[len(binding.bound_path) :]
            source_module = ".".join((binding.source_module, *suffix))
            if source_module not in self._helper_modules:
                return None
            return self._helper_bindings[source_module].get(call.func.attr)
        return None

    def _helper_review(
        self,
        identifier: _FunctionID,
        call: ast.Call,
        frame: _EvaluationFrame,
    ) -> HelperReview:
        qualified_name = f"{identifier.module_name}.{identifier.function_name}"
        digest = self._helper_digests.get(identifier)
        if digest is None:
            digest = _helper_digest(
                identifier,
                {**self._modules, **self._helper_modules},
                self._all_functions,
                self._all_bindings,
            )
            self._helper_digests[identifier] = digest
        review = self._helper_reviews.get(qualified_name)
        location = (
            f"called by {frame.consumer!r} at {frame.module.display_path}:{call.lineno}"
        )
        if review is None:
            raise RuntimeError(
                f"Unreviewed PolicyEngine-US helper {qualified_name} (digest "
                f"{digest}) {location}. Review which variables it reads and "
                "add it to POLICYENGINE_US_HELPER_REVIEWS."
            )
        if review.digest != digest:
            raise RuntimeError(
                f"PolicyEngine-US helper {qualified_name} changed (digest "
                f"{digest}, reviewed {review.digest}) {location}. Review which "
                "variables it now reads and update POLICYENGINE_US_HELPER_REVIEWS."
            )
        if identifier not in self._checked_helpers:
            unknown = sorted(
                name
                for name in (
                    *review.reads,
                    *(
                        read
                        for _position, _keyword, names in review.reads_when_omitted
                        for read in names
                    ),
                )
                if name not in self._definitions
            )
            if unknown:
                raise RuntimeError(
                    f"PolicyEngine-US helper review {qualified_name} names reads "
                    f"that are not variables: {unknown}."
                )
            self._checked_helpers.add(identifier)
            self.reviewed_helpers.add(qualified_name)
        return review

    def _record_package_helper(
        self,
        identifier: _FunctionID,
        call: ast.Call,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
        aggregation_entity: str,
    ) -> None:
        """Record a reviewed helper's reads as the calling consumer's reads."""

        review = self._helper_review(identifier, call, frame)
        for position, keyword in review.arguments:
            if any(
                isinstance(argument, ast.Starred)
                for argument in call.args[: position + 1]
            ):
                self._raise_unresolved(call, frame)
            if position < len(call.args):
                argument = call.args[position]
            else:
                argument = next(
                    (item.value for item in call.keywords if item.arg == keyword),
                    None,
                )
                if argument is None and any(item.arg is None for item in call.keywords):
                    self._raise_unresolved(call, frame)
            if argument is None:
                continue
            self._record_expression(
                argument,
                env=env,
                frame=frame,
                owner_name=owner_name,
                base_kind="helper_call",
                aggregation_entity=aggregation_entity,
            )
        reads = list(review.reads)
        for position, keyword, names in review.reads_when_omitted:
            argument = (
                call.args[position]
                if position < len(call.args)
                else next(
                    (item.value for item in call.keywords if item.arg == keyword),
                    None,
                )
            )
            if argument is None or (
                self._evaluator.evaluate(argument, env, frame) == _NONE_VALUE
            ):
                reads.extend(names)
        if reads:
            self._record_receipts(
                _SequenceValue(tuple(_StringValue(name) for name in reads)),
                expression=call,
                base_kind="helper_call",
                module=frame.module,
                consumer=frame.consumer,
                owner_name=owner_name,
                frame=frame,
                aggregation_entity=aggregation_entity,
            )

    def _enqueue_helper(
        self,
        call: ast.Call,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> None:
        if not frame.enqueue_helpers or _is_local_name(call, env):
            return
        identifier = self._helper_identifier(call, frame)
        if identifier is None:
            return
        context = _FunctionContext(
            identifier=identifier,
            args=tuple(
                self._evaluator.evaluate(argument, env, frame) for argument in call.args
            ),
            keywords=tuple(
                sorted(
                    (
                        keyword.arg,
                        self._evaluator.evaluate(keyword.value, env, frame),
                    )
                    for keyword in call.keywords
                    if keyword.arg is not None
                )
            ),
        )
        if context not in self._seen_contexts:
            self._seen_contexts.add(context)
            self._queue.append(context)

    def _helper_return_value(
        self,
        call: ast.Call,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> _ExactValue | None:
        """What a source helper returns for this call's exact arguments.

        policyengine-us 2.60 formulas unpack helper results before reading
        them (``income, deductions = dependent_income_and_deductions(...)``).
        The helper body is interpreted with the call's arguments, once per
        distinct context, and its ``return`` values (``None`` when it can
        fall off the end) become the assigned value's alternatives. A
        recursive context is unknown. Returns None for a call that names no
        source helper, which keeps the evaluator's opaque result.
        """

        if _is_local_name(call, env):
            return None
        identifier = self._helper_identifier(call, frame)
        if identifier is None or identifier not in self._functions:
            return None
        if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
            keyword.arg is None for keyword in call.keywords
        ):
            return _unknown(f"helper-unpack:{call.lineno}")
        context = _FunctionContext(
            identifier=identifier,
            args=tuple(
                self._evaluator.evaluate(argument, env, frame) for argument in call.args
            ),
            keywords=tuple(
                sorted(
                    (keyword.arg, self._evaluator.evaluate(keyword.value, env, frame))
                    for keyword in call.keywords
                    if keyword.arg is not None
                )
            ),
        )
        if context in self._return_values:
            return self._return_values[context]
        if context in self._returning:
            return _unknown(f"helper-recursion:{identifier.function_name}")
        self._returning.add(context)
        try:
            function = self._functions[identifier]
            # Only the value is wanted here. The helper's own receipts and
            # its strict sink checks belong to the context the call enqueues
            # (when its frame enqueues helpers), so this pass defers
            # unresolved sinks and enqueues nothing.
            flows = self._analyze_function(
                module=function.module,
                function=function.node,
                consumer=function.node.name,
                owner_name=None,
                args=context.args,
                keywords=context.keywords,
                defer_unresolved_sinks=True,
                enqueue_helpers=False,
            )
        finally:
            self._returning.discard(context)
        values = [
            flow.value
            for flow in flows
            if flow.signal == "return" and flow.value is not None
        ]
        if any(flow.signal == "normal" for flow in flows):
            values.append(_NONE_VALUE)
        result = (
            _make_alternative(tuple(values))
            if values
            else _unknown(f"helper-no-return:{identifier.function_name}")
        )
        self._return_values[context] = result
        return result

    def _helper_identifier(
        self,
        call: ast.Call,
        frame: _EvaluationFrame,
    ) -> _FunctionID | None:
        if isinstance(call.func, ast.Name):
            return self._bindings[frame.module.module_name].get(call.func.id)
        if not isinstance(call.func, ast.Attribute):
            return None
        base_path = _attribute_path(call.func.value)
        if base_path is None:
            return None
        for binding in reversed(frame.module.module_imports):
            if base_path[: len(binding.bound_path)] != binding.bound_path:
                continue
            suffix = base_path[len(binding.bound_path) :]
            source_module = ".".join((binding.source_module, *suffix))
            if source_module not in self._modules:
                return None
            identifier = self._bindings[source_module].get(call.func.attr)
            if identifier is None:
                self._raise_unresolved(call, frame)
            return identifier
        return None

    def _record_expression(
        self,
        expression: ast.expr,
        *,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
        base_kind: str,
        receiver_entity: str = "",
        aggregation_entity: str = "",
    ) -> None:
        value = self._evaluator.evaluate(expression, env, frame)
        self._record_receipts(
            value,
            expression=expression,
            base_kind=base_kind,
            module=frame.module,
            consumer=frame.consumer,
            owner_name=owner_name,
            frame=frame,
            receiver_entity=receiver_entity,
            aggregation_entity=aggregation_entity,
        )

    def _record_receipts(
        self,
        value: _ExactValue,
        *,
        expression: ast.expr,
        base_kind: str,
        module: _ModuleIR,
        consumer: str,
        owner_name: str | None,
        frame: _EvaluationFrame,
        receiver_entity: str = "",
        aggregation_entity: str = "",
    ) -> None:
        value = _materialize_value(
            value,
            frame=frame,
            line=expression.lineno,
        )
        if frame.defer_unresolved_sinks and not self._is_receipt_value(value):
            return
        strings = self._receipt_strings(value, expression, frame)
        for item in strings:
            if owner_name is not None and item.value == owner_name:
                continue
            if item.parameter_backed and item.constructed:
                kind = f"constructed_parameter_{base_kind}"
            elif item.parameter_backed:
                kind = f"parameter_{base_kind}"
            elif item.constructed:
                kind = f"constructed_{base_kind}"
            else:
                kind = base_kind
            self._receipts.setdefault(item.value, set()).add(
                ConsumerReceipt(
                    consumer=consumer,
                    path=module.display_path,
                    line=expression.lineno,
                    kind=kind,
                    receiver_entity=receiver_entity,
                    aggregation_entity=aggregation_entity,
                )
            )

    def _receipt_strings(
        self,
        value: _ExactValue,
        expression: ast.expr,
        frame: _EvaluationFrame,
    ) -> list[_StringValue]:
        if isinstance(value, _StringValue):
            return [value]
        if isinstance(value, _SequenceValue):
            result: list[_StringValue] = []
            for item in value.items:
                result.extend(self._receipt_strings(item, expression, frame))
            return result
        if isinstance(value, _AlternativeValue):
            result = []
            for item in value.values:
                result.extend(self._receipt_strings(item, expression, frame))
            return result
        self._raise_unresolved(expression, frame)

    @classmethod
    def _is_receipt_value(cls, value: _ExactValue) -> bool:
        if isinstance(value, _StringValue):
            return True
        if isinstance(value, _SequenceValue):
            return all(cls._is_receipt_value(item) for item in value.items)
        if isinstance(value, _AlternativeValue):
            return all(cls._is_receipt_value(item) for item in value.values)
        return False

    @staticmethod
    def _raise_unresolved(
        expression: ast.expr,
        frame: _EvaluationFrame,
    ) -> None:
        raise RuntimeError(
            "Unresolved dynamic PolicyEngine consumer aggregation for "
            f"{frame.consumer!r} at {frame.module.display_path}:"
            f"{expression.lineno}: {ast.unparse(expression)}."
        )

    def _apply_mutation(
        self,
        expression: ast.expr,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> dict[str, _ExactValue]:
        updated = dict(env)
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and isinstance(expression.func.value, ast.Name)
            and expression.func.attr in {"append", "extend"}
            and expression.args
        ):
            return updated
        name = expression.func.value.id
        previous = env.get(name, _unknown(f"mutation:{name}"))
        argument = self._evaluator.evaluate(expression.args[0], env, frame)
        if expression.func.attr == "append":
            extension: _ExactValue = _SequenceValue((argument,))
        else:
            extension = _materialize_value(
                argument,
                frame=frame,
                line=expression.lineno,
            )
        updated[name] = _combine_values(previous, extension, "add")
        return updated

    @staticmethod
    def _assign_unknown(
        target: ast.expr,
        env: dict[str, _ExactValue],
        line: int,
    ) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = _unknown(f"assignment:{line}:{target.id}")
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            env[target.value.id] = _unknown(f"assignment:{line}:{target.value.id}")
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                _ConsumerInterpreter._assign_unknown(item, env, line)

    def _assign_value(
        self,
        target: ast.expr,
        value: _ExactValue,
        env: dict[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> bool:
        if _bind_target(target, value, env):
            return True
        if not (
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
        ):
            return False
        base = env.get(
            target.value.id,
            frame.module_env.get(target.value.id, _unknown(f"name:{target.value.id}")),
        )
        key = self._evaluator.evaluate(target.slice, env, frame)
        if not isinstance(base, _MapValue) or not isinstance(key, _StringValue):
            return False
        mapping = dict(base.items)
        mapping[key.value] = value
        env[target.value.id] = _MapValue(tuple(mapping.items()))
        return True

    @staticmethod
    def _dedupe_flows(
        flows: list[_Flow],
        frame: _EvaluationFrame,
    ) -> list[_Flow]:
        unique: dict[
            tuple[str, tuple[tuple[str, _ExactValue], ...], _ExactValue | None], _Flow
        ] = {}
        for flow in flows:
            key = (flow.signal, tuple(sorted(flow.env.items())), flow.value)
            unique[key] = flow
        if len(unique) > _STATE_CAP:
            raise RuntimeError(
                "PolicyEngine-US static consumer state cap exceeded for "
                f"{frame.consumer!r} in {frame.module.display_path}."
            )
        return list(unique.values())


def _index_policyengine_us_sources(
    variables_root: Path,
    *,
    parameters_root: Path | None = None,
    helper_reviews: Mapping[str, HelperReview] | None = None,
) -> _PolicyEngineUSSourceIndex:
    """Build the shared source index from one parsed module inventory.

    Consumer interpretation is layered on this exact IR; this checkpoint keeps
    the metadata surface live while the sequential consumer pass is added.
    """

    if not variables_root.is_dir():
        raise RuntimeError(
            "The installed PolicyEngine-US variable source tree is unavailable "
            f"at {variables_root}."
        )
    modules = _inventory_modules(variables_root)
    resolver = _ParameterListResolver(
        parameters_root
        if parameters_root is not None
        else variables_root.parent / "parameters"
    )
    module_envs = _module_environments(modules, resolver)

    definitions: dict[str, _SourceVariableDefinition] = {}
    for module_name in sorted(modules):
        module = modules[module_name]
        for node in module.classes:
            if not _is_policyengine_variable(node):
                continue
            if node.name in definitions:
                raise RuntimeError(
                    f"Duplicate PolicyEngine variable class {node.name!r} in "
                    f"{module.source_path}."
                )
            definitions[node.name] = _variable_definition(
                node,
                source_path=module.source_path,
            )
    if not definitions:
        raise RuntimeError(
            f"No PolicyEngine variable classes found below {variables_root}."
        )
    interpreter = _ConsumerInterpreter(
        modules,
        module_envs,
        resolver,
        definitions,
        helper_modules=_inventory_helper_modules(variables_root.parent, modules),
        helper_reviews=helper_reviews,
    )
    consumers = interpreter.run()
    return _PolicyEngineUSSourceIndex(
        definitions=MappingProxyType(definitions),
        consumers=consumers,
        reviewed_helpers=tuple(sorted(interpreter.reviewed_helpers)),
    )


def _index_policyengine_us_variable_sources(
    variables_root: Path,
) -> Mapping[str, _SourceVariableDefinition]:
    """Compatibility metadata view over the one combined source index."""

    return _index_policyengine_us_sources(variables_root).definitions


__all__ = [
    "ConsumerReceipt",
    "_PolicyEngineUSSourceIndex",
    "_SourceVariableDefinition",
    "_index_policyengine_us_sources",
    "_index_policyengine_us_variable_sources",
]
