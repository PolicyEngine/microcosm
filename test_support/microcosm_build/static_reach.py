"""The first-party modules a function can reach, read statically from source.

A kernel's implementation hash covers the modules its list names. A trace of
one run sees the modules whose functions that run calls. It can't see a
module the run only reads constants from, or a branch the run didn't take.
This walk reads source instead, and over-approximates.

From a root, a function or a method with its class, the walk follows every
branch of each body it reaches, with default argument values, decorators,
class bases, class attributes and module-level constant values. It resolves
each name to every definition that can bind it, through imports at module or
function level, star imports and a package's lazy exports. Each scope binds
only its own names, as Python scopes them: a comprehension, lambda or nested
function keeps its names to itself, a class body's names stay out of its
methods, and ``global`` and ``nonlocal`` send a name outward.

Where code reaches a definition without naming it, the walk counts it too:

- A module-level constant stands for each module, class or function its
  value names (``_helper = helper_module``).
- A module-level statement that changes a reached name in place
  (``_TABLE.update(...)``, ``_TABLE[key] = value``) runs with it, in any
  module the walk has read.
- An attribute of an object whose type the walk can't tell reaches every
  same-named member of each class it has referenced. It also reaches every
  same-named definition in each module the walk has seen used as a value,
  such as an argument or a dict entry.
- ``getattr``, ``hasattr``, ``setattr`` and ``delattr`` with a literal name
  read that attribute, and so do the literal names that ``attrgetter`` and
  ``methodcaller`` take.
- ``import_module`` and ``__import__`` with a literal name reach that module
  as a value.
- A referenced class reaches every dunder and sunder method it defines,
  since Python calls them without naming them: operators, conversions,
  containers, descriptors, copying, pickling and Enum's ``_missing_``.
- A name that a package doesn't bind but exports lazily, through a
  module-level ``__getattr__``, reaches that function and the name in each of
  the package's modules.

It imports nothing from the repository and runs none of its code: it parses
each first-party module's file under ``packages/*/src``.

What it can't see:

- a name built at run time (``getattr`` or ``import_module`` with a computed
  name, ``globals()``, ``exec``);
- code reached by registration rather than by name: a function that a
  decorator records for a framework to call (``singledispatch``, pydantic's
  validators), or a statement in a module the walk never reads that fills a
  reached registry on import;
- code that an argument carries in from outside the walk (the executor's
  context, for one);
- a module that only an annotation names. Before Python 3.14 a ``def``
  statement evaluates its annotations as it runs; from 3.14 they wait until
  something reads them (``typing.get_type_hints``); and
  ``from __future__ import annotations`` keeps them as strings.

A :class:`Cut` leaves out one guarded branch, so a test names each exception
it relies on rather than skipping a module.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from test_support.paths import REPOSITORY_ROOT

#: Builtins that read or write an attribute. With a literal name, the walk
#: reads the call as that attribute.
ATTRIBUTE_BUILTINS = frozenset({"getattr", "hasattr", "setattr", "delattr"})
#: Callables whose literal string arguments name attributes.
ATTRIBUTE_NAMERS = frozenset({"attrgetter", "methodcaller"})
#: Callables whose literal first argument names a module to import.
IMPORTERS = frozenset({"import_module", "__import__"})

#: One reached thing: (defining module, kind, qualified name or None).
Item = tuple[str, str, str | None]
#: What one name binds in a scope: None for a plain local, ``("import",
#: module, attribute or None)``, or ``("member", class)`` for a name in a
#: top-level class's body.
Binding = tuple[str, ...] | None
Scope = Mapping[str, Binding]

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPES = (*_FUNCTIONS, ast.ClassDef, *_COMPREHENSIONS)


def is_implicit(member: str) -> bool:
    """Whether Python can call ``member`` without naming it.

    Every dunder (``__add__``, ``__float__``) counts, and so does every
    sunder, the single-underscore names Enum calls (``_missing_``).
    """

    core = member.strip("_")
    return bool(core) and member in {f"__{core}__", f"_{core}_"}


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
    """The statements that run in ``body``'s own scope, through every block."""

    for node in body:
        yield node
        if isinstance(node, _DEFS):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from _statements(getattr(node, field, ()))
        for handler in getattr(node, "handlers", ()):
            yield from _statements(handler.body)
        for case in getattr(node, "cases", ()):
            yield from _statements(case.body)


def _stored(target: ast.AST) -> list[str]:
    """The names an assignment target binds."""

    return [
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]


def _root_name(node: ast.AST) -> str | None:
    """The name an attribute, subscript or call chain starts from."""

    while isinstance(node, ast.Attribute | ast.Subscript | ast.Call | ast.Starred):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _add(values: list[ast.AST], node: ast.AST) -> None:
    if not any(value is node for value in values):
        values.append(node)


def _assignments(node: ast.stmt) -> Iterator[tuple[ast.AST, ast.AST]]:
    """The (target, value) pairs one statement binds in its own scope."""

    if isinstance(node, ast.Assign):
        for target in node.targets:
            yield target, node.value
    elif isinstance(node, ast.AnnAssign | ast.AugAssign) and node.value is not None:
        yield node.target, node.value
    elif isinstance(node, ast.For | ast.AsyncFor):
        yield node.target, node.iter
    elif isinstance(node, ast.With | ast.AsyncWith):
        for item in node.items:
            if item.optional_vars is not None:
                yield item.optional_vars, item.context_expr
    if not isinstance(node, _DEFS):
        for child in ast.walk(node):
            if isinstance(child, ast.NamedExpr):
                yield child.target, child.value


def _mutations(node: ast.stmt) -> Iterator[str]:
    """The names one statement changes in place.

    ``_TABLE`` for ``_TABLE.update(...)``, ``_TABLE[key] = value`` or
    ``_TABLE.field += 1``.
    """

    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        if isinstance(func, ast.Attribute):
            name = _root_name(func.value)
            if name is not None:
                yield name
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets = [node.target]
    else:
        targets = []
    for target in targets:
        for part in ast.walk(target):
            if isinstance(part, ast.Attribute | ast.Subscript) and isinstance(
                part.ctx, ast.Store
            ):
                name = _root_name(part.value)
                if name is not None:
                    yield name


class _Module:
    """One parsed module's top-level bindings, by name."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        functions: dict[str, list[ast.AST]] = defaultdict(list)
        classes: dict[str, list[ast.ClassDef]] = defaultdict(list)
        constants: dict[str, list[ast.AST]] = defaultdict(list)
        self.star_imports: list[str] = []
        #: (name, statement): each module-level statement that changes a
        #: name in place, once for each name it changes.
        self.mutations: list[tuple[str, ast.stmt]] = []
        for node in _statements(tree.body):
            if isinstance(node, ast.Import | ast.ImportFrom):
                for bound, binding in _bindings(self, node).items():
                    if bound == "*":
                        self.star_imports.append(binding[0])
                    else:
                        imports[bound].append(binding)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions[node.name].append(node)
            elif isinstance(node, ast.ClassDef):
                classes[node.name].append(node)
            else:
                for target, value in _assignments(node):
                    for bound in _stored(target):
                        _add(constants[bound], value)
                for bound in dict.fromkeys(_mutations(node)):
                    self.mutations.append((bound, node))
        self.imports = dict(imports)
        self.functions = dict(functions)
        self.classes = dict(classes)
        self.constants = dict(constants)

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


def _members(module: _Module, node: ast.ClassDef) -> dict[str, list[ast.AST]]:
    """One class body's names, each with the statements that bind it."""

    out: dict[str, list[ast.AST]] = defaultdict(list)
    for child in _statements(node.body):
        if isinstance(child, _DEFS):
            out[child.name].append(child)
        elif isinstance(child, ast.Import | ast.ImportFrom):
            for bound in _bindings(module, child):
                out[bound].append(child)
        elif isinstance(child, ast.AnnAssign) and child.value is None:
            # A field with no default, such as a dataclass's ``rows: int``.
            for bound in _stored(child.target):
                _add(out[bound], child)
        else:
            for target, _ in _assignments(child):
                for bound in _stored(target):
                    _add(out[bound], child)
    return dict(out)


def _annotations(node: ast.AST) -> Iterator[ast.AST]:
    """The annotation expressions inside ``node``."""

    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            if child.returns is not None:
                yield child.returns
            for argument in _arguments(child.args):
                if argument.annotation is not None:
                    yield argument.annotation
        elif isinstance(child, ast.AnnAssign):
            yield child.annotation


def _arguments(arguments: ast.arguments) -> Iterator[ast.arg]:
    for argument in (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        arguments.vararg,
        arguments.kwarg,
    ):
        if argument is not None:
            yield argument


def _outer_parts(node: ast.AST) -> list[ast.AST]:
    """The parts of a nested scope that its enclosing scope evaluates."""

    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        decorators = [] if isinstance(node, ast.Lambda) else node.decorator_list
        defaults = [item for item in node.args.kw_defaults if item is not None]
        return [*decorators, *node.args.defaults, *defaults]
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *node.keywords]
    if isinstance(node, _COMPREHENSIONS):
        return [node.generators[0].iter]
    return []


def _inner_parts(node: ast.AST) -> list[ast.AST]:
    """The parts of a scope that the scope itself evaluates."""

    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return list(node.body)
    if isinstance(node, ast.Lambda):
        return [node.body]
    if isinstance(node, _COMPREHENSIONS):
        first, *rest = node.generators
        parts: list[ast.AST] = [first.target, *first.ifs]
        for generator in rest:
            parts.extend([generator.iter, generator.target, *generator.ifs])
        if isinstance(node, ast.DictComp):
            parts.extend([node.key, node.value])
        else:
            parts.append(node.elt)
        return parts
    return []


def _same_scope(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Every node under ``nodes`` that runs in their scope, not a nested one's."""

    stack = list(nodes)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _SCOPES):
            stack.extend(_outer_parts(node))
        else:
            stack.extend(ast.iter_child_nodes(node))


def _walrus_targets(node: ast.AST) -> Iterator[str]:
    """A comprehension's walrus targets, which bind in its enclosing scope."""

    for inner in _same_scope(_inner_parts(node)):
        if isinstance(inner, ast.NamedExpr) and isinstance(inner.target, ast.Name):
            yield inner.target.id
        elif isinstance(inner, _COMPREHENSIONS):
            yield from _walrus_targets(inner)


def _own_names(module: _Module, scope: ast.AST) -> tuple[dict[str, Binding], set[str]]:
    """The names ``scope`` binds for itself, and the names it declares global."""

    names: dict[str, Binding] = {}
    if isinstance(scope, _COMPREHENSIONS):
        for generator in scope.generators:
            names.update(dict.fromkeys(_stored(generator.target)))
        return names, set()
    if isinstance(scope, _FUNCTIONS):
        names.update(dict.fromkeys(argument.arg for argument in _arguments(scope.args)))
    declared_global: set[str] = set()
    declared_nonlocal: set[str] = set()
    for node in _same_scope(_inner_parts(scope)):
        if isinstance(node, ast.Global):
            declared_global.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            declared_nonlocal.update(node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.setdefault(node.id, None)
        elif isinstance(node, _DEFS):
            names.setdefault(node.name, None)
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar):
            if node.name:
                names.setdefault(node.name, None)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.setdefault(node.rest, None)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for bound, binding in _bindings(module, node).items():
                names[bound] = ("import", *binding)
        elif isinstance(node, _COMPREHENSIONS):
            for target in _walrus_targets(node):
                names.setdefault(target, None)
    for name in declared_global | declared_nonlocal:
        names.pop(name, None)
    return names, declared_global


def _visits(
    module: _Module,
    node: ast.AST,
    skipped: frozenset[int],
    here: Scope,
    closure: Scope,
) -> Iterator[tuple[ast.AST, Scope]]:
    """Every node under ``node`` with the names local where it runs.

    ``here`` holds the names local at ``node``. ``closure`` holds those a
    nested function there would see, which leave out a class body's. A node
    comes before the nodes under it.
    """

    stack: list[tuple[ast.AST, Scope, Scope]] = [(node, here, closure)]
    while stack:
        current, here, closure = stack.pop()
        if id(current) in skipped:
            continue
        yield current, here
        if isinstance(current, _SCOPES):
            own, declared_global = _own_names(module, current)
            inner = {
                name: binding
                for name, binding in {**closure, **own}.items()
                if name not in declared_global
            }
            # A class body sees its own names; its methods and lambdas don't.
            after = closure if isinstance(current, ast.ClassDef) else inner
            stack.extend(
                (part, inner, after) for part in reversed(_inner_parts(current))
            )
            stack.extend(
                (part, here, closure) for part in reversed(_outer_parts(current))
            )
        else:
            children = list(ast.iter_child_nodes(current))
            stack.extend((child, here, closure) for child in reversed(children))


def _named(value: ast.AST) -> Iterator[ast.AST]:
    """The names and dotted names an expression's value can be."""

    if isinstance(value, ast.Name | ast.Attribute):
        yield value
    elif isinstance(value, ast.IfExp):
        yield from _named(value.body)
        yield from _named(value.orelse)
    elif isinstance(value, ast.BoolOp):
        for part in value.values:
            yield from _named(part)
    elif isinstance(value, ast.NamedExpr | ast.Starred):
        yield from _named(value.value)
    elif isinstance(value, ast.Tuple | ast.List | ast.Set):
        # ``a, b = first, second``: over-approximate by element.
        for part in value.elts:
            yield from _named(part)


def _callee(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


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
        #: Modules the walk has seen used as a value, not as an attribute's root.
        self.values: set[str] = set()
        self.attributes: set[str] = set()
        self.cuts_used: set[Cut] = set()
        self.mutations_run: set[int] = set()
        self._members: dict[tuple[str, str], dict[str, list[ast.AST]]] = {}
        self._inside: dict[str, list[str]] = {}
        # The lookups in progress. A cycle of re-exports or aliases adds
        # nothing the outer lookup doesn't already see.
        self._resolving: set[tuple[str, str]] = set()
        self._lazy: set[tuple[str, str]] = set()
        self._aliasing: set[Item] = set()

    def module(self, name: str) -> _Module | None:
        if name not in self.paths:
            return None
        if name not in self.modules:
            self.modules[name] = _Module(name, self.paths[name])
        return self.modules[name]

    def is_module(self, name: str) -> bool:
        """A first-party module, or a namespace package that holds one."""

        return name in self.prefixes

    def inside(self, package: str) -> list[str]:
        """The first-party modules inside ``package``, at any depth."""

        if package not in self._inside:
            prefix = f"{package}."
            self._inside[package] = sorted(
                name for name in self.paths if name.startswith(prefix)
            )
        return self._inside[package]

    def push(self, item: Item, site: str) -> None:
        # A namespace package has no file, and so nothing to hash.
        if item[0] not in self.paths:
            return
        if item not in self.reached:
            self.pending.append(item)
        self.reached[item].add(site)

    def members(self, module: str, name: str) -> dict[str, list[ast.AST]]:
        """A top-level class's body names, across every ``class`` that defines it."""

        key = (module, name)
        if key not in self._members:
            found = self.module(module)
            out: dict[str, list[ast.AST]] = defaultdict(list)
            for node in found.classes.get(name, ()):
                for member, nodes in _members(found, node).items():
                    out[member].extend(nodes)
            self._members[key] = dict(out)
        return self._members[key]

    def resolve(self, module: str, name: str) -> list[Item]:
        """Every definition ``name`` can bind in ``module``'s top-level scope."""

        found = self.module(module)
        key = (module, name)
        if found is None or key in self._resolving:
            return []
        self._resolving.add(key)
        try:
            items: list[Item] = [
                (module, kind, name)
                for kind, table in (
                    ("function", found.functions),
                    ("class", found.classes),
                    ("constant", found.constants),
                )
                if name in table
            ]
            for binding in found.imports.get(name, ()):
                items.extend(self.resolve_import(*binding))
            for star in found.star_imports:
                items.extend(self.resolve(star, name))
        finally:
            self._resolving.discard(key)
        return items

    def resolve_import(self, target: str, attribute: str | None) -> list[Item]:
        """What ``import target`` or ``from target import attribute`` binds."""

        if attribute is None:
            return [(target, "module", None)] if self.is_module(target) else []
        return self.exported(target, attribute)

    def exported(self, module: str, name: str) -> list[Item]:
        """What ``module.name`` can be, read from outside ``module``."""

        items: list[Item] = []
        if self.is_module(f"{module}.{name}"):
            items.append((f"{module}.{name}", "module", None))
        items.extend(self.resolve(module, name))
        found = self.module(module)
        if items or found is None or "__getattr__" not in found.functions:
            return items
        # A lazy export: the module's __getattr__ and, for a package, the name
        # wherever the package's own modules bind it.
        items.append((module, "function", "__getattr__"))
        key = (module, name)
        if found.is_package and key not in self._lazy:
            self._lazy.add(key)
            try:
                for inner in self.inside(module):
                    items.extend(self.resolve(inner, name))
            finally:
                self._lazy.discard(key)
        return items

    def resolve_expression(self, module: str, node: ast.AST) -> list[Item]:
        """What a name or dotted name in ``module``'s top-level scope names."""

        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return []
        items = self.resolve(module, node.id)
        for attribute in reversed(chain):
            items = [
                found
                for item in items
                for start in self.aliased(item)
                if start[1] == "module"
                for found in self.exported(start[0], attribute)
            ]
        return items

    def aliased(self, item: Item) -> list[Item]:
        """``item``; for a constant, what each of its values names.

        ``_helper = helper_module`` makes ``_helper`` stand for that module.
        A constant whose values name nothing stands for itself.
        """

        if item[1] != "constant" or item in self._aliasing:
            return [item]
        self._aliasing.add(item)
        found: list[Item] = []
        try:
            for value in self.module(item[0]).constants[item[2]]:
                for named in _named(value):
                    for resolved in self.resolve_expression(item[0], named):
                        found.extend(self.aliased(resolved))
        finally:
            self._aliasing.discard(item)
        return found or [item]

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
        """Push every first-party item ``node``'s code can name.

        With an ``owner``, ``node`` sits in that top-level class's body, so
        the body's names resolve to the class's members.
        """

        found = self.module(module)
        skipped = self.skipped(module, qualname, node)
        start: Scope = (
            dict.fromkeys(self.members(module, owner), ("member", owner))
            if owner is not None
            else {}
        )
        # Names that root an attribute chain; any other module name is a value.
        rooted: set[int] = set()

        def lookup(name: str, here: Scope) -> list[Item]:
            if name not in here:
                return self.resolve(module, name)
            binding = here[name]
            if binding is None:
                return []
            if binding[0] == "member":
                return [(module, "member", f"{binding[1]}.{name}")]
            return self.resolve_import(binding[1], binding[2])

        def attribute(node: ast.Attribute, here: Scope, at: str) -> None:
            self.attributes.add(node.attr)
            chain = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if not isinstance(current, ast.Name):
                return
            rooted.add(id(current))
            if owner is not None and current.id in {"self", "cls"}:
                self.push((module, "member", f"{owner}.{chain[-1]}"), at)
                return
            starts = [
                start
                for item in lookup(current.id, here)
                for start in self.aliased(item)
                if start[1] == "module"
            ]
            for name in reversed(chain):
                following = []
                for start in starts:
                    for item in self.exported(start[0], name):
                        self.push(item, at)
                        following.extend(
                            found
                            for found in self.aliased(item)
                            if found[1] == "module"
                        )
                starts = following

        for child, here in _visits(found, node, skipped, start, {}):
            at = f"{site}:{getattr(child, 'lineno', '?')}"
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                for item in lookup(child.id, here):
                    self.push(item, at)
                    if item[1] == "module" and id(child) not in rooted:
                        self.values.add(item[0])
            elif isinstance(child, ast.Attribute):
                attribute(child, here, at)
            elif isinstance(child, ast.Import | ast.ImportFrom):
                for bound, binding in _bindings(found, child).items():
                    if bound != "*":
                        for item in self.resolve_import(*binding):
                            self.push(item, at)
            elif isinstance(child, ast.Call):
                callee = _callee(child.func)
                texts = [_literal(argument) for argument in child.args]
                if callee in ATTRIBUTE_BUILTINS and len(texts) >= 2 and texts[1]:
                    read = ast.Attribute(child.args[0], texts[1], ast.Load())
                    attribute(ast.copy_location(read, child), here, at)
                elif callee in ATTRIBUTE_NAMERS:
                    for text in filter(None, texts):
                        self.attributes.update(text.split("."))
                elif callee in IMPORTERS and texts and texts[0]:
                    # import_module returns the module it names; __import__
                    # returns the top package, whose attributes reach the rest.
                    name = texts[0]
                    if callee == "__import__":
                        name = name.partition(".")[0]
                    if self.is_module(name):
                        self.push((texts[0], "module", None), at)
                        self.push((name, "module", None), at)
                        self.values.add(name)

    def process(self, item: Item) -> None:
        module, kind, name = item
        found = self.module(module)
        if found is None:
            return
        if kind == "function":
            for node in found.functions.get(name, ()):
                self.scan(module, node, f"{module}.{name}", qualname=name)
        elif kind == "constant":
            for value in found.constants.get(name, ()):
                self.scan(module, value, f"{module}.{name}=")
        elif kind == "class":
            self.classes.add((module, name))
            for node in found.classes.get(name, ()):
                for part in (*node.decorator_list, *node.bases, *node.keywords):
                    self.scan(module, part, f"{module}.{name}(class)")
            for member, nodes in self.members(module, name).items():
                methods = all(
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    for node in nodes
                )
                if is_implicit(member) or not methods:
                    self.push(
                        (module, "member", f"{name}.{member}"), f"{module}.{name}"
                    )
        elif kind == "member":
            owner, member = name.split(".", 1)
            if owner not in found.classes:
                return
            nodes = self.members(module, owner).get(member)
            for node in nodes or ():
                self.scan(module, node, f"{module}.{name}", qualname=name, owner=owner)
            if nodes:
                return
            # Inherited: the same name on each first-party base class.
            for node in found.classes[owner]:
                for base in node.bases:
                    for resolved in self.resolve_expression(module, base):
                        for start in self.aliased(resolved):
                            if start[1] == "class":
                                self.push(
                                    (start[0], "member", f"{start[2]}.{member}"),
                                    f"{module}.{name}(base)",
                                )

    def is_reached(self, item: Item) -> bool:
        if item[1] == "module":
            return any(found[0] == item[0] for found in self.reached)
        return item in self.reached

    def settle(self) -> bool:
        """Push what reached names imply; whether anything new came."""

        added = False
        for module_name, class_name in sorted(self.classes):
            for member in self.members(module_name, class_name):
                item = (module_name, "member", f"{class_name}.{member}")
                if member in self.attributes and item not in self.reached:
                    self.push(item, f"{module_name}.{class_name}(attribute)")
                    added = True
        for module_name in sorted(self.values):
            for name in sorted(self.attributes):
                for item in self.exported(module_name, name):
                    if item not in self.reached:
                        self.push(item, f"{module_name}(value).{name}")
                        added = True
                    if item[1] == "module" and item[0] not in self.values:
                        self.values.add(item[0])
                        added = True
        for found in list(self.modules.values()):
            for bound, node in found.mutations:
                if id(node) in self.mutations_run:
                    continue
                if any(
                    self.is_reached(item) for item in self.resolve(found.name, bound)
                ):
                    self.mutations_run.add(id(node))
                    self.scan(found.name, node, f"{found.name}:{node.lineno}(changes)")
                    added = True
        return added

    def run(self, module: str, qualname: str) -> None:
        if "." in qualname:
            # A method's class, with its bases, decorators and dunders.
            self.push((module, "class", qualname.split(".", 1)[0]), "root")
            self.push((module, "member", qualname), "root")
        else:
            self.push((module, "function", qualname), "root")
        while True:
            while self.pending:
                self.process(self.pending.pop())
            if not self.settle() and not self.pending:
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
