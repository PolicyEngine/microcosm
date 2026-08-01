"""Import-free PolicyEngine-US source inventory and static-value IR.

This private module owns the single AST inventory used by both metadata and
consumer queries.  It deliberately does not import :mod:`policyengine_us`:
ordinary variable classes and their static reference surfaces are read from
the installed wheel's source tree.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from populace.frame.schema import VariableMetadata

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
)


def _unknown(*origins: str) -> _UnknownValue:
    return _UnknownValue(tuple(dict.fromkeys(origins)))


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


@dataclass(frozen=True)
class _ImportBinding:
    source_module: str
    source_name: str
    bound_name: str


@dataclass(frozen=True)
class _ModuleIR:
    """One parsed source module and its module-scoped static surfaces."""

    source_path: Path
    display_path: str
    module_name: str
    is_package: bool
    tree: ast.Module
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


def _inventory_modules(variables_root: Path) -> dict[str, _ModuleIR]:
    modules: dict[str, _ModuleIR] = {}
    for source_path in sorted(variables_root.rglob("*.py")):
        try:
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
        except (OSError, SyntaxError) as exc:
            raise RuntimeError(
                f"Could not index PolicyEngine variable source {source_path}."
            ) from exc
        relative_path = source_path.relative_to(variables_root).as_posix()
        is_package = relative_path.endswith("/__init__.py") or relative_path == (
            "__init__.py"
        )
        relative_module = relative_path.removesuffix(".py")
        if relative_module == "__init__":
            relative_module = ""
        elif relative_module.endswith("/__init__"):
            relative_module = relative_module.removesuffix("/__init__")
        relative_module = relative_module.replace("/", ".")
        module_name = "policyengine_us.variables"
        if relative_module:
            module_name += f".{relative_module}"
        imports: list[_ImportBinding] = []
        star_imports: list[str] = []
        for statement in tree.body:
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
                    imports.append(
                        _ImportBinding(
                            source_module=source_module,
                            source_name=alias.name,
                            bound_name=alias.asname or alias.name,
                        )
                    )
        modules[module_name] = _ModuleIR(
            source_path=source_path,
            display_path=f"variables/{relative_path}",
            module_name=module_name,
            is_package=is_package,
            tree=tree,
            imports=tuple(imports),
            star_imports=tuple(star_imports),
            classes=tuple(node for node in tree.body if isinstance(node, ast.ClassDef)),
            functions=tuple(
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
        )
    return modules


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
        return _SequenceValue(
            tuple(_eval_module_value(item, env) for item in node.elts),
            set_like=isinstance(node, ast.Set),
        )
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
            return _SequenceValue(
                tuple(self.evaluate(item, local_env, frame) for item in node.elts),
                set_like=isinstance(node, ast.Set),
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
    return False


_ENTITY_CALL_NAMES = frozenset(
    {"person", "household", "tax_unit", "spm_unit", "family", "marital_unit"}
)
_REFERENCE_HELPER_ARGUMENTS: dict[str, int] = {"tax_unit_non_dep_sum": 0}
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


def _function_parameters(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.arg, ...]:
    return tuple((*function.args.posonlyargs, *function.args.args))


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


class _ConsumerInterpreter:
    """Sequential exact-state interpreter for PolicyEngine consumer receipts."""

    def __init__(
        self,
        modules: Mapping[str, _ModuleIR],
        module_envs: Mapping[str, Mapping[str, _ExactValue]],
        resolver: _ParameterListResolver,
    ) -> None:
        self._modules = modules
        self._module_envs = module_envs
        self._resolver = resolver
        self._evaluator = _ExactEvaluator()
        self._functions, self._bindings = _function_bindings(modules, module_envs)
        self._receipts: dict[str, set[ConsumerReceipt]] = {}
        self._queue: list[_FunctionContext] = []
        self._seen_contexts: set[_FunctionContext] = set()

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
    ) -> None:
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
        )
        self._analyze_block(
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
            return [_Flow(dict(env), "return")]
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
    ) -> None:
        if isinstance(expression, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            self._scan_comprehension(expression, env, frame, owner_name)
            return
        if isinstance(expression, ast.Lambda):
            return
        if isinstance(expression, ast.Call):
            function_name = _source_name(expression.func)
            callee = self._evaluator.evaluate(expression.func, env, frame)
            if (
                function_name in _ENTITY_CALL_NAMES or isinstance(callee, _EntityValue)
            ) and expression.args:
                self._record_expression(
                    expression.args[0],
                    env=env,
                    frame=frame,
                    owner_name=owner_name,
                    base_kind="entity_call",
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
                self._record_expression(
                    variable_expression,
                    env=env,
                    frame=frame,
                    owner_name=owner_name,
                    base_kind=function_name,
                )
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
                    )
            self._enqueue_helper(expression, env, frame)
        for child in ast.iter_child_nodes(expression):
            if isinstance(child, ast.expr):
                self._scan_expression(child, env, frame, owner_name)

    def _scan_comprehension(
        self,
        expression: ast.ListComp | ast.SetComp | ast.GeneratorExp,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
    ) -> None:
        states: list[dict[str, _ExactValue]] = [dict(env)]
        for generator in expression.generators:
            expanded: list[dict[str, _ExactValue]] = []
            for state in states:
                self._scan_expression(generator.iter, state, frame, owner_name)
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
                            self._scan_expression(condition, bound, frame, owner_name)
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
            self._scan_expression(expression.elt, state, frame, owner_name)

    def _enqueue_helper(
        self,
        call: ast.Call,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
    ) -> None:
        if not frame.enqueue_helpers:
            return
        if not isinstance(call.func, ast.Name):
            return
        identifier = self._bindings[frame.module.module_name].get(call.func.id)
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

    def _record_expression(
        self,
        expression: ast.expr,
        *,
        env: Mapping[str, _ExactValue],
        frame: _EvaluationFrame,
        owner_name: str | None,
        base_kind: str,
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
        unique: dict[tuple[str, tuple[tuple[str, _ExactValue], ...]], _Flow] = {}
        for flow in flows:
            key = (flow.signal, tuple(sorted(flow.env.items())))
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
    consumers = _ConsumerInterpreter(modules, module_envs, resolver).run()
    return _PolicyEngineUSSourceIndex(
        definitions=MappingProxyType(definitions),
        consumers=consumers,
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
