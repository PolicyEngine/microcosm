"""The first-party modules a function can reach, read statically from source.

A kernel's implementation hash covers the modules its list names. A trace of
one run sees the modules whose functions that run calls. It can't see a
module the run only reads constants from, or a branch the run didn't take.
This walk reads source instead, and over-approximates. From a root function
it follows every branch of each body it reaches, with default argument
values, decorators, class bases, class attributes and module-level constant
values. It resolves each name through imports, at module or function level,
to the module that defines it. An attribute of an object whose type the walk
can't tell reaches every same-named member of each class the walk has
referenced.

It imports nothing from the repository and runs none of its code: it parses
each first-party module's file under ``packages/*/src``.

What it can't see: a name built at run time (``getattr`` with a computed
name, ``importlib``), a package's lazy ``__getattr__`` exports, and code that
an argument carries in from outside the walk (the executor's context, for
one). Annotations stay out: a ``def`` evaluates them at import, or never
under ``from __future__ import annotations``. A :class:`Cut` leaves out one
guarded branch, so a test names each exception it relies on rather than
skipping a module.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from test_support.paths import REPOSITORY_ROOT

#: Dunder methods Python calls without naming them. A referenced class
#: reaches each one it defines.
IMPLICIT_METHODS = frozenset(
    {
        "__init__",
        "__post_init__",
        "__new__",
        "__init_subclass__",
        "__set_name__",
        "__class_getitem__",
        "__call__",
        "__len__",
        "__iter__",
        "__next__",
        "__getitem__",
        "__contains__",
        "__bool__",
        "__eq__",
        "__ne__",
        "__hash__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__str__",
        "__repr__",
        "__format__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__enter__",
        "__exit__",
    }
)

#: One reached thing: (defining module, kind, qualified name or None).
Item = tuple[str, str, str | None]


@dataclass(frozen=True)
class Cut:
    """One guarded branch the walk leaves out: an ``if`` in one function.

    In ``module``'s ``function`` (a qualname such as ``Target.__post_init__``)
    the walk skips the test and the body of each ``if`` whose test's source
    contains ``guard``. It still walks the ``else`` branch.
    """

    module: str
    function: str
    guard: str


def module_paths(root: Path = REPOSITORY_ROOT) -> dict[str, Path]:
    """Each first-party module's dotted name -> its file under ``packages/*/src``."""

    paths = {}
    for path in sorted(root.glob("packages/*/src/microcosm/**/*.py")):
        source_root = next(parent for parent in path.parents if parent.name == "src")
        parts = path.relative_to(source_root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        paths[".".join(parts)] = path
    return paths


def _statements(body: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    """Module-level statements, through ``if``, ``try`` and ``with`` blocks."""

    for node in body:
        yield node
        if isinstance(node, ast.If | ast.With | ast.Try | ast.TryStar):
            yield from _statements(node.body)
            yield from _statements(getattr(node, "orelse", ()))
            yield from _statements(getattr(node, "finalbody", ()))
            for handler in getattr(node, "handlers", ()):
                yield from _statements(handler.body)


class _Module:
    """One parsed module's top-level names: imports, functions, classes, values."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.imports: dict[str, tuple[str, str | None]] = {}
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.constants: dict[str, list[ast.AST]] = defaultdict(list)
        for node in _statements(tree.body):
            if isinstance(node, ast.Import | ast.ImportFrom):
                self.imports.update(_bindings(self, node))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if node.value is None:
                    continue
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            self.constants[name.id].append(node.value)

    def absolute(self, level: int, module: str | None) -> str:
        """The module a ``from`` import names, relative imports resolved."""

        if not level:
            return module or ""
        parts = self.name.split(".")
        if not self.is_package:
            parts = parts[:-1]
        parts = parts[: len(parts) - (level - 1)]
        base = ".".join(parts)
        return f"{base}.{module}" if module else base


def _bindings(
    module: _Module, node: ast.Import | ast.ImportFrom
) -> dict[str, tuple[str, str | None]]:
    """The names one import statement binds -> (module, attribute or None)."""

    if isinstance(node, ast.Import):
        return {
            (alias.asname or alias.name.split(".")[0]): (
                alias.name if alias.asname else alias.name.split(".")[0],
                None,
            )
            for alias in node.names
        }
    stem = module.absolute(node.level, node.module)
    return {(alias.asname or alias.name): (stem, alias.name) for alias in node.names}


def _annotations(node: ast.AST) -> Iterator[ast.AST]:
    """The annotation expressions inside ``node``."""

    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            yield from (item for item in [child.returns] if item is not None)
            arguments = child.args
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            ):
                if argument is not None and argument.annotation is not None:
                    yield argument.annotation
        elif isinstance(child, ast.AnnAssign):
            yield child.annotation


def _walk(node: ast.AST, skipped: frozenset[int]) -> Iterator[ast.AST]:
    """Every node under ``node``, pruning each subtree whose id is skipped."""

    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in skipped:
            continue
        yield current
        stack.extend(ast.iter_child_nodes(current))


class _Walk:
    def __init__(self, paths: Mapping[str, Path], cuts: Iterable[Cut]) -> None:
        self.paths = paths
        # Every module name and each dotted prefix of one, so a namespace
        # package (``microcosm``) resolves on the way to its modules.
        self.prefixes = {
            ".".join(name.split(".")[:size])
            for name in paths
            for size in range(1, name.count(".") + 2)
        }
        self.cuts = tuple(cuts)
        self.modules: dict[str, _Module] = {}
        self.reached: dict[Item, set[str]] = defaultdict(set)
        self.pending: list[Item] = []
        self.classes: set[tuple[str, str]] = set()
        self.attributes: set[str] = set()
        self.cuts_used: set[Cut] = set()

    def module(self, name: str) -> _Module | None:
        if name not in self.paths:
            return None
        if name not in self.modules:
            self.modules[name] = _Module(name, self.paths[name])
        return self.modules[name]

    def is_module(self, name: str) -> bool:
        """A first-party module, or a namespace package that holds one."""

        return name in self.prefixes

    def push(self, item: Item, site: str) -> None:
        # A namespace package has no file, and so nothing to hash.
        if item[0] not in self.paths:
            return
        if item not in self.reached:
            self.pending.append(item)
        self.reached[item].add(site)

    def resolve_expression(self, module: str, node: ast.AST) -> Item | None:
        """The item a name or a ``module.attribute`` chain names, if first-party."""

        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        item = self.resolve(module, node.id)
        for attribute in reversed(chain):
            if item is None or item[1] != "module":
                return None
            target = item[0]
            if self.is_module(f"{target}.{attribute}"):
                item = (f"{target}.{attribute}", "module", None)
            else:
                item = self.resolve(target, attribute)
        return item

    def resolve(self, module: str, name: str, depth: int = 0) -> Item | None:
        """Follow ``name`` in ``module``'s scope to the item that defines it."""

        found = self.module(module)
        if found is None or depth > 32:
            return None
        if name in found.functions:
            return (module, "function", name)
        if name in found.classes:
            return (module, "class", name)
        if name in found.constants:
            return (module, "constant", name)
        if name in found.imports:
            return self.resolve_import(*found.imports[name], depth=depth + 1)
        return None

    def resolve_import(
        self, target: str, attribute: str | None, *, depth: int = 0
    ) -> Item | None:
        if attribute is None:
            return (target, "module", None) if self.is_module(target) else None
        if self.is_module(f"{target}.{attribute}"):
            return (f"{target}.{attribute}", "module", None)
        if target in self.paths:
            return self.resolve(target, attribute, depth)
        return None

    def skipped(self, module: str, qualname: str, node: ast.AST) -> frozenset[int]:
        """Annotations, and the subtrees each cut in this function leaves out."""

        ids = {id(child) for item in _annotations(node) for child in ast.walk(item)}
        for cut in self.cuts:
            if (cut.module, cut.function) != (module, qualname):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.If) and cut.guard in ast.unparse(child.test):
                    self.cuts_used.add(cut)
                    for part in (child.test, *child.body):
                        ids.update(id(item) for item in ast.walk(part))
        return frozenset(ids)

    def scan(
        self,
        module: str,
        node: ast.AST,
        site: str,
        *,
        qualname: str = "",
        owner: str | None = None,
    ) -> None:
        """Push every first-party item ``node``'s code can name."""

        found = self.module(module)
        skipped = self.skipped(module, qualname, node)
        bound: set[str] = set()
        local_imports: dict[str, tuple[str, str | None]] = {}
        for child in _walk(node, skipped):
            if isinstance(child, ast.Import | ast.ImportFrom):
                local_imports.update(_bindings(found, child))
            elif isinstance(child, ast.Name) and isinstance(
                child.ctx, ast.Store | ast.Del
            ):
                bound.add(child.id)
            elif isinstance(child, ast.arg):
                bound.add(child.arg)
            elif (
                isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and child is not node
            ):
                bound.add(child.name)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                bound.add(child.name)

        def lookup(name: str) -> Item | None:
            if name in local_imports:
                return self.resolve_import(*local_imports[name])
            if name in bound:
                return None
            return self.resolve(module, name)

        for child in _walk(node, skipped):
            line = getattr(child, "lineno", "?")
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                item = lookup(child.id)
                if item is not None:
                    self.push(item, f"{site}:{line}")
            elif isinstance(child, ast.Attribute):
                self.attributes.add(child.attr)
                chain = []
                current: ast.AST = child
                while isinstance(current, ast.Attribute):
                    chain.append(current.attr)
                    current = current.value
                if not isinstance(current, ast.Name):
                    continue
                if owner is not None and current.id in {"self", "cls"}:
                    self.push(
                        (module, "member", f"{owner}.{chain[-1]}"), f"{site}:{line}"
                    )
                    continue
                item = lookup(current.id)
                if item is None or item[1] != "module":
                    continue
                target = item[0]
                for attribute in reversed(chain):
                    if self.is_module(f"{target}.{attribute}"):
                        target = f"{target}.{attribute}"
                        self.push((target, "module", None), f"{site}:{line}")
                        continue
                    member = self.resolve(target, attribute)
                    if member is not None:
                        self.push(member, f"{site}:{line}")
                    break

    def members(self, module: str, name: str) -> dict[str, ast.AST]:
        """A class body's methods and attribute values, by name."""

        out: dict[str, ast.AST] = {}
        for node in self.module(module).classes[name].body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                out[node.name] = node
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = node
        return out

    def process(self, item: Item) -> None:
        module, kind, name = item
        found = self.module(module)
        if found is None:
            return
        if kind == "function":
            self.scan(module, found.functions[name], f"{module}.{name}", qualname=name)
        elif kind == "constant":
            for value in found.constants[name]:
                self.scan(module, value, f"{module}.{name}=")
        elif kind == "class":
            self.classes.add((module, name))
            node = found.classes[name]
            for part in (*node.bases, *node.keywords, *node.decorator_list):
                self.scan(module, part, f"{module}.{name}(class)")
            for member, body in self.members(module, name).items():
                if not isinstance(body, ast.FunctionDef | ast.AsyncFunctionDef) or (
                    member in IMPLICIT_METHODS
                ):
                    self.push(
                        (module, "member", f"{name}.{member}"), f"{module}.{name}"
                    )
        elif kind == "member":
            owner, member = name.split(".", 1)
            if owner not in found.classes:
                return
            members = self.members(module, owner)
            if member in members:
                self.scan(
                    module,
                    members[member],
                    f"{module}.{name}",
                    qualname=name,
                    owner=owner,
                )
                return
            # Inherited: the same name on each first-party base class.
            for base in found.classes[owner].bases:
                resolved = self.resolve_expression(module, base)
                if resolved is not None and resolved[1] == "class":
                    self.push(
                        (resolved[0], "member", f"{resolved[2]}.{member}"),
                        f"{module}.{name}(base)",
                    )

    def run(self, module: str, qualname: str) -> None:
        if "." in qualname:
            self.push((module, "member", qualname), "root")
        else:
            self.push((module, "function", qualname), "root")
        while True:
            while self.pending:
                self.process(self.pending.pop())
            added = False
            for module_name, class_name in sorted(self.classes):
                for member in self.members(module_name, class_name):
                    item = (module_name, "member", f"{class_name}.{member}")
                    if member in self.attributes and item not in self.reached:
                        self.push(item, f"{module_name}.{class_name}(attribute)")
                        added = True
            if not added:
                return


@dataclass(frozen=True)
class Reach:
    """What a static walk from one root reached."""

    #: Each reached item -> the sites (``module.qualname:line``) that named it.
    items: Mapping[Item, frozenset[str]]
    #: The cuts whose guard the walk found and left out.
    cuts_used: frozenset[Cut]

    @property
    def modules(self) -> frozenset[str]:
        return frozenset(module for module, _, _ in self.items)

    def names(self, module: str) -> list[str]:
        """The reached items that ``module`` defines, for failure messages."""

        return sorted(
            f"{kind}:{name}" for found, kind, name in self.items if found == module
        )


def reach(
    module: str,
    qualname: str,
    *,
    cuts: Iterable[Cut] = (),
    paths: Mapping[str, Path] | None = None,
) -> Reach:
    """Statically walk ``module``'s ``qualname`` (a function or ``Class.method``)."""

    walk = _Walk(module_paths() if paths is None else paths, cuts)
    walk.run(module, qualname)
    return Reach(
        {item: frozenset(sites) for item, sites in walk.reached.items()},
        frozenset(walk.cuts_used),
    )
