"""The first-party modules a module imports, read from its source.

A kernel's implementation hash covers the files of the modules it lists, and
``source_hash`` does not follow imports. A module list that stands for "the
code this kernel runs" is therefore complete only if every first-party module
a listed module imports is listed too, or excluded with a reason. These
helpers find those imports, so a test can pin the list's import closure.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import ModuleType

#: Every workspace package imports as ``microcosm.<shard>``.
FIRST_PARTY_PREFIX = "microcosm."


def _is_package(module: ModuleType) -> bool:
    return Path(module.__file__).name == "__init__.py"


def _import_stem(module_name: str, level: int, stem: str | None) -> str:
    """The absolute module name an ``import`` statement's ``from`` part names."""

    if not level:
        return stem or ""
    module = importlib.import_module(module_name)
    package = module_name if _is_package(module) else module_name.rsplit(".", 1)[0]
    base = package.rsplit(".", level - 1)[0]
    return f"{base}.{stem}" if stem else base


def defining_module(package_name: str, attribute: str) -> str:
    """The module that defines ``attribute`` as the package re-exports it.

    A submodule, class or function names its own module. A constant is traced
    through the package ``__init__``'s ``from … import`` statements. A name
    the ``__init__`` defines itself resolves to the package.
    """

    package = importlib.import_module(package_name)
    value = getattr(package, attribute, None)
    if isinstance(value, ModuleType):
        return value.__name__
    if inspect.isclass(value) or inspect.isfunction(value):
        return inspect.getmodule(value).__name__
    tree = ast.parse(Path(package.__file__).read_text(encoding="utf-8"))
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        for alias in statement.names:
            if (alias.asname or alias.name) != attribute:
                continue
            stem = _import_stem(package_name, statement.level, statement.module)
            if _is_package(importlib.import_module(stem)):
                return defining_module(stem, alias.name)
            return stem
    return package_name


def first_party_imports(module_name: str) -> set[str]:
    """Every first-party module ``module_name`` imports, at any depth of its AST.

    Function-level imports count: they run when the function does. A name
    imported from a package resolves to the module that defines it, so
    ``from microcosm.calibrate import TargetRegistry`` names
    ``microcosm.calibrate.registry``.
    """

    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            stem = _import_stem(module_name, node.level, node.module)
            if not stem.startswith(FIRST_PARTY_PREFIX):
                continue
            if not _is_package(importlib.import_module(stem)):
                found.add(stem)
                continue
            found.update(defining_module(stem, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(
                alias.name
                for alias in node.names
                if alias.name.startswith(FIRST_PARTY_PREFIX)
                and not _is_package(importlib.import_module(alias.name))
            )
    found.discard(module_name)
    return found


def uncovered_imports(
    listed: tuple[str, ...], excluded: set[str] | frozenset[str]
) -> set[str]:
    """``listed -> imported`` for each import that is neither listed nor excluded.

    Excluded modules are not followed, so an exclusion stops the walk; every
    listed module is checked, so the closure holds transitively.
    """

    covered = set(listed) | set(excluded)
    return {
        f"{name} -> {imported}"
        for name in listed
        for imported in first_party_imports(name)
        if imported not in covered
    }
