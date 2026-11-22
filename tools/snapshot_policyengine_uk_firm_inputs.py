"""Snapshot policyengine-uk's firm variables for engine-free tests.

Reads a policyengine-uk checkout's source with ``ast`` (it never imports
``policyengine_uk``) and writes, as sorted JSON:

- every ``Variable`` subclass whose ``entity`` is ``Firm``, anywhere in the
  package except its ``tests/`` directory, with its value type, unit, default
  value, enum members (the ``possible_values`` class resolved through the
  defining module's imports, so ``Region`` comes from
  ``variables/household/demographic/geography.py``), definition period,
  whether it has a formula (a ``formula`` or ``formula_*`` method or
  attribute, ``adds`` or ``subtracts``) and its file relative to the package;
- the formula-less firm variables, which a dataset sets as inputs;
- every policyengine-uk variable name, and which of Microcosm's own firm
  columns (``MICROCOSM_ONLY_COLUMNS``) collide with one;
- the checkout's HEAD commit, whether its working tree is dirty, the pull
  request the firm entity comes from and the snapshot date.

A variable created at runtime (by a reform or a loop) is not in the source
and so not in the snapshot.

``--check`` regenerates the snapshot and fails if anything but the snapshot
date differs from ``--out``, so a moved policyengine-uk HEAD or a changed firm
variable shows up before the fixture is refreshed.

Usage::

    uv run --no-sync python tools/snapshot_policyengine_uk_firm_inputs.py \\
        --policyengine-uk-root ../policyengine-uk \\
        --out packages/microcosm-build/tests/fixtures/uk/policyengine_uk_firm_inputs.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PACKAGE_NAME = "policyengine_uk"
FIRM_ENTITY = "Firm"
#: The policyengine-uk pull request that adds the firm entity.
POLICYENGINE_UK_PULL_REQUEST = "PolicyEngine/policyengine-uk#2940"

#: Columns Microcosm's firm table carries that are not policyengine-uk inputs.
#: None may share a name with a policyengine-uk variable, or a dataset load
#: would read it as one.
MICROCOSM_ONLY_COLUMNS = (
    "in_frame",
    "vat_scope",
    "vat_registered",
    "base_weight",
    "data_vintage",
    "source_firm_id",
    "source_firm_key",
    "ons_legal_status",
    "ons_region_code",
)

#: Fields ``--check`` ignores: the date the snapshot was taken.
_VOLATILE_FIELDS = ("snapshot_date",)


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


def _is_variable(node: ast.ClassDef) -> bool:
    return "Variable" in _base_names(node)


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


def _firm_variable(
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


def _git(root: Path, *args: str) -> str:
    environment = dict(os.environ)
    local = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for name in local:
        environment.pop(name, None)
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout


def build_snapshot(root: Path, *, snapshot_date: str) -> dict[str, Any]:
    """The firm-input snapshot of the policyengine-uk checkout at ``root``."""

    package_dir = root / PACKAGE_NAME
    if not package_dir.is_dir():
        raise ValueError(f"{root} holds no {PACKAGE_NAME} package.")
    modules = _read_modules(package_dir)
    variable_names: set[str] = set()
    firm_variables: dict[str, dict[str, Any]] = {}
    for module in modules.values():
        for node in module.classes.values():
            if not _is_variable(node):
                continue
            variable_names.add(node.name)
            entity = _assignments(node).get("entity")
            if isinstance(entity, ast.Name) and entity.id == FIRM_ENTITY:
                if node.name in firm_variables:
                    raise ValueError(f"Firm variable {node.name} is defined twice.")
                firm_variables[node.name] = _firm_variable(
                    modules, module, node, package_dir
                )
    if not firm_variables:
        raise ValueError(f"{package_dir} defines no {FIRM_ENTITY} variables.")
    inputs = sorted(
        name for name, variable in firm_variables.items() if not variable["has_formula"]
    )
    return {
        "collisions": sorted(set(MICROCOSM_ONLY_COLUMNS) & variable_names),
        "firm_input_names": inputs,
        "firm_variables": firm_variables,
        "microcosm_only_columns_checked": sorted(MICROCOSM_ONLY_COLUMNS),
        "policyengine_uk": {
            "dirty": bool(_git(root, "status", "--porcelain").strip()),
            "head_commit": _git(root, "rev-parse", "HEAD").strip(),
            "pull_request": POLICYENGINE_UK_PULL_REQUEST,
        },
        "policyengine_uk_variable_names": sorted(variable_names),
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": snapshot_date,
    }


def render(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def _stable(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in snapshot.items() if key not in _VOLATILE_FIELDS
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policyengine-uk-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if --out differs from a fresh snapshot (ignoring its date).",
    )
    args = parser.parse_args(argv)
    snapshot = build_snapshot(
        args.policyengine_uk_root.resolve(),
        snapshot_date=datetime.now(UTC).date().isoformat(),
    )
    if args.check:
        committed = json.loads(args.out.read_text(encoding="utf-8"))
        if _stable(committed) != _stable(snapshot):
            changed = sorted(
                key
                for key in set(_stable(committed)) | set(_stable(snapshot))
                if committed.get(key) != snapshot.get(key)
            )
            print(f"{args.out} is stale; changed fields: {changed}", file=sys.stderr)
            return 1
        print(f"{args.out} matches the policyengine-uk checkout.")
        return 0
    args.out.write_text(render(snapshot), encoding="utf-8")
    inputs = snapshot["firm_input_names"]
    print(
        f"Wrote {args.out}: {len(snapshot['firm_variables'])} firm variables, "
        f"{len(inputs)} inputs, HEAD {snapshot['policyengine_uk']['head_commit']}"
        f"{' (dirty)' if snapshot['policyengine_uk']['dirty'] else ''}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
