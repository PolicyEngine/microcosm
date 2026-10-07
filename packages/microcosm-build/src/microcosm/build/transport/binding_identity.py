"""The identity of a gate binding: a closed vocabulary that fails closed.

``gates.battery@1`` takes its binding registry as a constructor argument, so
the bindings' behaviour must enter its implementation hash: otherwise a
changed binding would be served from the store under the old node key.
Describing arbitrary Python behaviour exactly is not possible by reflection
(decorators rewrite names, lambdas share a qualified name, C objects expose no
state), so a binding must be built from a closed vocabulary that can be
described exactly, and everything else is refused:

- the binding itself is an instance of a frozen dataclass defined at module
  top level (as :class:`~microcosm.build.gate_battery.FunctionBinding` is),
  with no state outside its fields;
- every field value is plain data (``None``, ``bool``, ``int``, a finite
  ``float``, ``str``, ``bytes``; exact ``tuple``, ``list``, ``dict``,
  ``MappingProxyType``, ``frozenset`` or ``set`` of plain data, with string
  mapping keys), an enum member, a class, another such dataclass, or a
  function;
- a function or class is defined at module top level in a module with a
  source file, its ``__module__`` names the module it was defined in, and a
  function has no closure and no ``__wrapped__`` decorator layer; its
  defaults are described by these same rules.

:func:`describe` returns a canonical-JSON description that keeps types (a
tuple is not a list), mapping order and every value, plus the SHA-256 of the
source of the defining modules of the binding's functions and classes (with
their parent packages), and, transitively, of every first-party module
(``microcosm.*``, ``test_support.*``) named by an ``import`` or
``from ... import`` statement anywhere in those files, function-local imports
included. Import statements are read from the source (``ast``) and resolved
with the import system's path finder, never executed, so resolving the
closure imports nothing.

Not covered, and so not allowed to carry behaviour: data a module reads from
disk (package JSON, registers), state a module mutates at run time, and code
reached without an import statement (``importlib.import_module`` or
``__import__``, even with a literal name). A binding takes its values through
the gate manifest's parameters or through evidence artifacts.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import hashlib
import importlib.machinery
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import FunctionType, MappingProxyType

from microcosm.graph.canonical import canonical_json

__all__ = [
    "FIRST_PARTY_PACKAGES",
    "UndescribableBindingError",
    "describe",
    "source_closure",
]

#: Top-level packages whose modules a binding's source closure follows.
FIRST_PARTY_PACKAGES = ("microcosm", "test_support")

_CONTAINERS = {
    tuple: "tuple",
    list: "list",
    dict: "dict",
    MappingProxyType: "mappingproxy",
    set: "set",
    frozenset: "frozenset",
}


class UndescribableBindingError(ValueError):
    """A binding holds a value outside the describable vocabulary."""


def _defined_in(obj: FunctionType | type, path: str) -> str:
    """The name of the module ``obj`` is defined at top level of."""

    if isinstance(obj, FunctionType):
        name = obj.__globals__.get("__name__")
        if name != obj.__module__:
            raise UndescribableBindingError(
                f"{path}: {obj.__qualname__} claims module {obj.__module__!r} but "
                f"was defined in {name!r} (a decorator rewrote it)."
            )
    name = obj.__module__
    module = sys.modules.get(name) if isinstance(name, str) else None
    if module is None or getattr(module, "__file__", None) is None:
        raise UndescribableBindingError(
            f"{path}: {obj.__qualname__} has no source file to bind."
        )
    if (
        obj.__qualname__ != obj.__name__
        or getattr(module, obj.__name__, None) is not obj
    ):
        raise UndescribableBindingError(
            f"{path}: {obj.__qualname__} is not a module-level definition of "
            f"{name!r}; bind a function or class defined at top level."
        )
    return name


def _slots(kind: type) -> set[str]:
    names: set[str] = set()
    for klass in kind.__mro__:
        declared = klass.__dict__.get("__slots__", ())
        names.update((declared,) if isinstance(declared, str) else declared)
    return names - {"__dict__", "__weakref__"}


class _Describer:
    def __init__(self) -> None:
        self.modules: set[str] = set()
        self._active: set[int] = set()

    def _code(self, obj: FunctionType | type, path: str) -> str:
        module = _defined_in(obj, path)
        self.modules.add(module)
        return f"{module}.{obj.__qualname__}"

    def describe(self, value: object, path: str) -> object:
        kind = type(value)
        if value is None or kind in (bool, int, str):
            return {"type": kind.__name__, "value": value}
        if kind is float:
            if not math.isfinite(value):
                raise UndescribableBindingError(f"{path}: non-finite float {value!r}.")
            return {"type": "float", "value": value}
        if kind is bytes:
            return {"type": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
        if isinstance(value, enum.Enum):
            return {
                "enum": self._code(kind, path),
                "member": value.name,
                "value": self.describe(value.value, f"{path}.value"),
            }
        if isinstance(value, type):
            return {"class": self._code(value, path)}
        if kind is FunctionType:
            return self._function(value, path)
        if kind in _CONTAINERS:
            return self._guarded(value, path, self._container)
        if dataclasses.is_dataclass(value):
            return self._guarded(value, path, self._dataclass)
        raise UndescribableBindingError(
            f"{path}: a {kind.__module__}.{kind.__qualname__} is outside the "
            "describable vocabulary (plain data, enums, classes, top-level "
            "functions, frozen dataclasses)."
        )

    def _guarded(self, value, path: str, describe_inner) -> object:
        if id(value) in self._active:
            raise UndescribableBindingError(f"{path}: reference cycle.")
        self._active.add(id(value))
        try:
            return describe_inner(value, path)
        finally:
            self._active.discard(id(value))

    def _container(self, value, path: str) -> object:
        kind = _CONTAINERS[type(value)]
        if kind in ("dict", "mappingproxy"):
            if any(type(key) is not str for key in value):
                raise UndescribableBindingError(
                    f"{path}: mapping keys must be strings."
                )
            return {
                kind: [
                    [key, self.describe(item, f"{path}[{key!r}]")]
                    for key, item in value.items()
                ]
            }
        if kind in ("set", "frozenset"):
            items = [self.describe(item, f"{path}{{}}") for item in value]
            return {kind: sorted(items, key=canonical_json)}
        return {
            kind: [
                self.describe(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        }

    def _function(self, value: FunctionType, path: str) -> object:
        if value.__closure__:
            raise UndescribableBindingError(
                f"{path}: {value.__qualname__} closes over variables; bind a "
                "top-level function and pass values through gates.json."
            )
        if hasattr(value, "__wrapped__"):
            raise UndescribableBindingError(
                f"{path}: {value.__qualname__} is a decorated wrapper; bind the "
                "undecorated function."
            )
        return {
            "function": self._code(value, path),
            "defaults": self.describe(value.__defaults__ or (), f"{path}.defaults"),
            "kwdefaults": self.describe(
                dict(value.__kwdefaults__ or {}), f"{path}.kwdefaults"
            ),
        }

    def _dataclass(self, value: object, path: str) -> object:
        kind = type(value)
        if not kind.__dataclass_params__.frozen:
            raise UndescribableBindingError(
                f"{path}: {kind.__qualname__} must be a frozen dataclass, so its "
                "identity cannot change after the kernel is built."
            )
        fields = dataclasses.fields(value)
        names = {field.name for field in fields}
        extra = (set(getattr(value, "__dict__", {})) | _slots(kind)) - names
        if extra:
            raise UndescribableBindingError(
                f"{path}: {kind.__qualname__} carries state outside its fields "
                f"{sorted(extra)}."
            )
        return {
            "dataclass": self._code(kind, path),
            "fields": [
                [
                    field.name,
                    self.describe(getattr(value, field.name), f"{path}.{field.name}"),
                ]
                for field in fields
            ],
        }


# ---------------------------------------------------------------------------
# The first-party source closure
# ---------------------------------------------------------------------------

#: (module, path, mtime_ns, size) -> the module names its imports name.
_IMPORTS: dict[tuple[str, str, int, int], tuple[str, ...]] = {}
#: Module name -> source file, for names found without importing them.
_RESOLVED: dict[str, Path] = {}
#: (path, mtime_ns, size) -> the file's SHA-256. A rewrite changes the size or
#: the modification time, so the key moves with the content.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _digest(path: Path) -> str:
    key = _file_key(path)
    digest = _DIGESTS.get(key)
    if digest is None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _DIGESTS[key] = digest
    return digest


def _first_party(name: str) -> bool:
    return any(
        name == top or name.startswith(top + ".") for top in FIRST_PARTY_PACKAGES
    )


def _module_file(name: str) -> Path | None:
    """The source file of a module, found without importing it.

    An imported module reports its own file. Otherwise the import system's
    path finder looks the name up in its parent package's directories without
    executing anything; it matches file names case-exactly, so a class name
    such as ``QRF`` never resolves to ``qrf.py`` on a case-insensitive disk.
    """

    module = sys.modules.get(name)
    if module is not None:
        file = getattr(module, "__file__", None)
        return Path(file) if file is not None else None
    cached = _RESOLVED.get(name)
    if cached is not None:
        return cached
    parent_name, _, _ = name.rpartition(".")
    if not parent_name:
        return None
    parent = sys.modules.get(parent_name)
    if parent is not None:
        search = list(getattr(parent, "__path__", ()))
    else:
        parent_file = _module_file(parent_name)
        search = (
            [str(parent_file.parent)]
            if parent_file is not None and parent_file.name == "__init__.py"
            else []
        )
    if not search:
        return None
    spec = importlib.machinery.PathFinder.find_spec(name, search)
    origin = getattr(spec, "origin", None)
    if not isinstance(origin, str) or not origin.endswith(".py"):
        return None
    # Only a found file is remembered: a module created later is still found.
    _RESOLVED[name] = Path(origin)
    return _RESOLVED[name]


def _imported_names(name: str, path: Path) -> tuple[str, ...]:
    """Every module an import statement in ``path`` can name, with parents."""

    key = (name, *_file_key(path))
    cached = _IMPORTS.get(key)
    if cached is not None:
        return cached
    tree = ast.parse(path.read_bytes(), filename=str(path))
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            named.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, node.module] if node.module else anchor)
            else:
                base = node.module or ""
            named.add(base)
            # ``from pkg import name`` may name a submodule.
            named.update(f"{base}.{alias.name}" for alias in node.names)
    result = tuple(sorted(_with_parents(named)))
    _IMPORTS[key] = result
    return result


def _with_parents(names: Iterable[str]) -> set[str]:
    """Each name and its parent packages, whose ``__init__`` an import runs."""

    return {
        ".".join(parts[:index])
        for item in names
        if item
        for parts in (item.split("."),)
        for index in range(1, len(parts) + 1)
    }


def source_closure(modules: Iterable[str]) -> dict[str, str]:
    """SHA-256 of the given modules and every first-party module they import.

    The given (defining) modules are always included, first-party or not,
    with their parent packages. From there the walk reads import statements
    from source and follows first-party names only; a name that is not a
    module file (an imported function or constant) is skipped once its module
    has been added.
    """

    starting = set(modules)
    found: dict[str, Path] = {}
    seen: set[str] = set()
    pending = sorted(_with_parents(starting))
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in starting and not _first_party(name):
            continue
        path = _module_file(name)
        if path is None:
            continue
        found[name] = path
        pending.extend(item for item in _imported_names(name, path) if item not in seen)
    return {name: _digest(found[name]) for name in sorted(found)}


def describe(
    binding: object, *, path: str = "binding"
) -> tuple[object, Mapping[str, str]]:
    """Describe a binding exactly and digest the source its code can run.

    Returns:
        ``(description, sources)``: a canonical-JSON value, and the SHA-256,
        by module name, of the binding's defining modules and every
        first-party module in their source closure.

    Raises:
        UndescribableBindingError: For a binding that is not a frozen
            top-level dataclass, or any value outside the vocabulary.
    """

    if not dataclasses.is_dataclass(binding) or isinstance(binding, type):
        raise UndescribableBindingError(
            f"{path}: a binding must be a frozen dataclass instance."
        )
    describer = _Describer()
    try:
        description = describer.describe(binding, path)
        canonical_json(description)
    except UndescribableBindingError:
        raise
    except (AttributeError, KeyError, RecursionError, TypeError, ValueError) as error:
        raise UndescribableBindingError(
            f"{path}: cannot be described exactly ({type(error).__name__}: {error})."
        ) from error
    return description, source_closure(describer.modules)
