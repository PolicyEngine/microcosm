"""The first-party modules a module imports, read from its source.

A kernel's implementation hash covers the files of the modules it passes to
``source_hash``, and ``source_hash`` does not follow imports. A kernel's
hashed modules therefore cover the code it runs only if every first-party
module they import is hashed too, or excluded with a reason. These helpers
find those imports, so a test can pin that closure.

What the scan sees: ``import a.b`` and ``import a.b as c`` (for a package,
through the attribute chains the module reads from the name it binds), and
``from a.b import c``, absolute or relative, at any depth of the module,
function bodies included. A name imported from a package resolves to the
module that defines it.

What it cannot see: a module named by a string (``importlib.import_module``,
``__import__``, ``getattr`` with a computed name), a module object passed in
as a value, and data files read by path (``importlib.resources``, ``open``).
Data files are not source, so a kernel that reads one must cover it in its
hash some other way, as the UK target kernels do with the country spec's
resource fingerprint.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from types import ModuleType

#: Every workspace package imports as ``microcosm.<shard>``.
FIRST_PARTY_PREFIX = "microcosm."


def _is_package(module: ModuleType) -> bool:
    return Path(module.__file__).name == "__init__.py"


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _import_stem(module_name: str, level: int, stem: str | None) -> str:
    """The absolute module name an ``import`` statement's ``from`` part names."""

    if not level:
        return stem or ""
    module = importlib.import_module(module_name)
    package = module_name if _is_package(module) else module_name.rsplit(".", 1)[0]
    base = package.rsplit(".", level - 1)[0]
    return f"{base}.{stem}" if stem else base


def _defining_module_of(value: object) -> str | None:
    """The module that defines a class or function, through any wrapper."""

    if inspect.isclass(value):
        return value.__module__
    unwrapped = inspect.unwrap(value) if callable(value) else value
    if inspect.isroutine(unwrapped):
        module = inspect.getmodule(unwrapped)
        return module.__name__ if module is not None else None
    return None


def defining_module(package_name: str, attribute: str) -> str:
    """The module that defines ``attribute`` as the package exposes it.

    A submodule resolves to itself whether or not it has been imported yet,
    so the answer does not depend on which tests ran first. A class or
    function names its own module. A constant is traced through the package
    ``__init__``'s ``from … import`` statements. A name the ``__init__``
    defines itself resolves to the package.
    """

    package = importlib.import_module(package_name)
    value = getattr(package, attribute, None)
    if isinstance(value, ModuleType):
        return value.__name__
    submodule = f"{package_name}.{attribute}"
    if value is None and _module_exists(submodule):
        return submodule
    defined_in = _defining_module_of(value)
    if defined_in is not None:
        return defined_in
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


def _resolve_dotted(name: str) -> str:
    """The module a dotted attribute chain such as ``a.b.C`` reads from."""

    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        if not _module_exists(prefix):
            continue
        module = importlib.import_module(prefix)
        if _is_package(module) and end < len(parts):
            return defining_module(prefix, parts[end])
        return prefix
    return name


def _attribute_chains(tree: ast.AST) -> list[str]:
    """Every outermost dotted chain rooted at a plain name (``a.b.c``)."""

    inner = {
        id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    chains = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in inner:
            continue
        parts = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            chains.append(".".join([value.id, *reversed(parts)]))
    return chains


def _package_import_reads(binding: str, target: str, chains: list[str]) -> set[str]:
    """The modules read through ``import target`` bound to the name ``binding``.

    ``import a.b`` binds ``a`` and ``import a.b as c`` binds ``c``; either way
    each chain read from the bound name resolves to the module defining what
    it names. With no chain into ``target``, the package itself is read.
    """

    prefix = target if binding == target.split(".")[0] else binding
    reads = {
        _resolve_dotted(target + chain[len(prefix) :])
        for chain in chains
        if chain == prefix or chain.startswith(prefix + ".")
    }
    return {name for name in reads if name.startswith(target)} or {target}


def first_party_imports(module_name: str) -> set[str]:
    """Every first-party module ``module_name`` imports, at any depth of its AST.

    Function-level imports count: they run when the function does. A name
    imported from a package resolves to the module that defines it, so
    ``from microcosm.calibrate import TargetRegistry`` names
    ``microcosm.calibrate.registry``.
    """

    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    chains: list[str] | None = None
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
            for alias in node.names:
                if not alias.name.startswith(FIRST_PARTY_PREFIX):
                    continue
                if not _is_package(importlib.import_module(alias.name)):
                    found.add(alias.name)
                    continue
                if chains is None:
                    chains = _attribute_chains(tree)
                binding = alias.asname or alias.name.split(".")[0]
                found |= _package_import_reads(binding, alias.name, chains)
    found.discard(module_name)
    return found


def unlisted_imports(listed: Sequence[str], excluded: AbstractSet[str]) -> set[str]:
    """``listed -> imported`` for each import of a listed module neither
    listed nor excluded. The check of a shared module list."""

    covered = set(listed) | set(excluded)
    return {
        f"{name} -> {imported}"
        for name in listed
        for imported in first_party_imports(name)
        if imported not in covered
    }


def unhashed_imports(
    hashed: AbstractSet[str], *, listed: Sequence[str], excluded: AbstractSet[str]
) -> set[str]:
    """``importer -> imported`` for each import a kernel's hash leaves uncovered.

    The walk starts at every module the kernel hashes. An import is covered
    when the kernel hashes it, or when the importer is in the shared list
    ``listed`` and the import is one of that list's reviewed exclusions
    ``excluded``. The walk goes on through every covered import, so an
    excluded module's own imports must be covered too, and its imports are
    not covered by the list's exclusions, whose reasons were written for the
    listed importers alone. An uncovered import is reported and not followed.
    """

    listed_names = set(listed)
    uncovered: set[str] = set()
    pending = sorted(hashed)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for imported in sorted(first_party_imports(name)):
            if imported in hashed or (name in listed_names and imported in excluded):
                pending.append(imported)
            else:
                uncovered.add(f"{name} -> {imported}")
    return uncovered


def edges(unclassified: Mapping[str, Sequence[str]]) -> set[str]:
    """``importer -> imported`` for a mapping of each import to its importers."""

    return {
        f"{importer} -> {imported}"
        for imported, importers in unclassified.items()
        for importer in importers
    }


def hashed_module_names(kernel: object, module: ModuleType) -> frozenset[str]:
    """Every module ``kernel.implementation_hash()`` passes to ``source_hash``.

    ``module`` is the module whose ``source_hash`` name the kernel calls; the
    recorder stands in for it during one call, then the original is restored.
    """

    original = module.source_hash
    recorded: list[object] = []

    def record(*objects: object, **dependencies: object) -> str:
        recorded.extend(objects)
        return original(*objects, **dependencies)

    module.source_hash = record
    try:
        kernel.implementation_hash()
    finally:
        module.source_hash = original
    names = set()
    for item in recorded:
        defining = item if isinstance(item, ModuleType) else inspect.getmodule(item)
        if defining is None:
            raise ValueError(f"{item!r} has no module whose file can be hashed.")
        names.add(defining.__name__)
    return frozenset(names)


def assert_an_edit_rekeys(
    kernel: object, module: ModuleType, tmp_path: Path, monkeypatch
) -> None:
    """``kernel``'s implementation hash changes when ``module``'s file does.

    The file is copied with one more line and ``module.__file__`` points at
    the copy, which ``source_hash`` reads.
    """

    before = kernel.implementation_hash()
    source = Path(module.__file__)
    edited = tmp_path / source.name
    edited.write_bytes(source.read_bytes() + b"\n# an edited line\n")
    monkeypatch.setattr(module, "__file__", str(edited))
    assert kernel.implementation_hash() != before, (kernel.ref, module.__name__)
