"""The first-party modules a function can reach, read statically from source.

A kernel's implementation hash covers the modules its list names. A trace of
one run sees the modules whose functions that run calls. It can't see a
module the run only reads constants from, or a branch the run didn't take.
This walk reads source instead, and over-approximates.

It parses each first-party module's file under ``packages/*/src`` and imports
nothing under ``packages/``, so it runs none of the code it walks.

What the walk follows from a root, a function or a method with its class:

- Bodies. Every branch of each body it reaches, with default argument values,
  decorators, class bases and keywords, class attributes and the values of
  module-level constants.
- Bindings. A name resolves to every definition that can bind it: each
  alternative of a ``try`` or an ``if``, at module or function level, star
  imports, re-exports, and a package's lazy exports through a module-level
  ``__getattr__``, defined or assigned. Every form of assignment binds: a
  ``for`` or ``with`` target stands for what the loop iterates or the
  statement enters, a ``match`` capture for the subject, and a walrus
  target for its value.
- Scopes. Each scope binds only its own names, as Python scopes them. A
  comprehension, lambda or nested function keeps its names to itself, a class
  body's names stay out of its methods, and ``global`` and ``nonlocal`` send a
  name outward. A class body reads a name it binds from its own namespace
  only once a statement at the body's top level has bound it. Before that, or
  when only a branch, a loop or a bare annotation binds it, Python reads the
  module's binding, so the walk follows that one too.
- Aliases. A module-level constant stands for each module, class or function
  its value names: ``_helper = helper_module``, ``_Base = pkg.Base``,
  ``pkg.Base[int]``, either branch of a conditional, each operand of ``and``
  or ``or``, a walrus's value, each element of a tuple, list or set (starred
  ones too), and ``getattr``, ``import_module`` or ``__import__`` with a
  literal name (for ``__import__``, the top package as well).
- Inheritance. A member that a class doesn't define resolves on each
  first-party base, through aliases and subscripts (``Base[int]``).
- Self and cls. In a method, a chain from ``self`` or ``cls`` continues
  through a module the class body imports (``self.mod.fn()``). The
  attribute fallback reaches the member itself.
- Implicit calls. A referenced class reaches every dunder and sunder method
  it defines, since Python calls them without naming them: operators,
  conversions, containers, descriptors, copying, pickling and Enum's
  ``_missing_``.
- Import-time code. Importing a module runs its module-level code, and
  creating a class runs its body. In each module that holds an item the walk
  reaches, and in each class it reaches, the walk follows every statement
  that runs only for its effect, in any block and whatever it names: an
  expression statement (``_setup()``, ``setattr(Kernel, ...)``), a ``del``,
  and a store to an item or an attribute. It also follows each compound
  statement's header there: an ``if`` or ``while`` test, what a ``for``
  iterates or a ``with`` enters, a ``match`` statement's subject, patterns
  and guards, and each ``except`` or ``except*`` clause's types.
- Changes by name. In any module the walk has read, a statement at module
  level or in any class body that changes a reached name in place runs with
  it: a method call on the name (``_TABLE.update(...)``, also inside an
  assignment), a store to or a ``del`` of its item or attribute, or a call
  statement that takes it as an argument (``_fill(_TABLE)``), also through a
  constant that aliases it (``_VIEW = _TABLE``).
- Attribute fallback. An attribute of an object whose type the walk can't
  tell reaches every same-named member of each class the walk has
  referenced, and every same-named definition in each module it has seen
  used as a value: an argument, a dict entry, or a dotted name such as
  ``pkg.helper``.
- Literal names. ``getattr``, ``hasattr``, ``setattr`` and ``delattr`` read
  the attribute a literal names, and so do ``attrgetter``, ``methodcaller``
  and a ``match`` class pattern (its keyword names, and an explicit
  ``__match_args__`` for positional patterns). ``import_module`` and
  ``__import__`` reach the module a literal names as a value:
  ``import_module`` relative to a literal package, ``__package__`` or
  ``__name__``, and ``__import__`` with its top package as well. A
  module-level constant that holds only strings counts as a literal.

What it can't see:

- a name built at run time (``getattr`` or ``import_module`` with a computed
  name, ``globals()``, ``exec``);
- code reached by registration rather than by name: a function that a
  decorator records for a framework or a registry to call
  (``singledispatch``, pydantic's validators);
- the import-time code of a module that holds no item the walk reaches, such
  as the ``__init__`` of a package above a reached module, or a module
  imported only for its effects, unless a statement there changes a reached
  name in place in a module the walk has read;
- the side effects of a call inside a statement that binds a name or tests a
  value (``_X = _setup()``, ``assert _setup()``), unless the walk reaches that
  statement another way;
- the import-time code of a class or function the walk doesn't reach: its
  decorators, default values, class bases and class attributes, unless a
  statement in the class body changes a reached name in place;
- a member inherited from a base that the ``class`` statement computes with a
  call;
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
#: The most rounds a walk may take to settle. Each round reaches a new item or
#: value module, or runs a new statement, so a walk that needs more has a bug.
MAX_ROUNDS = 10_000

#: One reached thing: (defining module, kind, qualified name or None).
Item = tuple[str, str, str | None]
#: One way a scope can bind a name: ``("import", module, attribute or None)``,
#: ``("member", class)`` for a name in a top-level class's body, or
#: ``("module",)`` for a class-body name that can also come from the module.
Binding = tuple[str, ...]
#: A scope's names -> every way each can be bound; ``()`` for a plain local.
Scope = Mapping[str, tuple[Binding, ...]]

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPES = (*_FUNCTIONS, ast.ClassDef, *_COMPREHENSIONS)
_COMPOUND = (
    ast.If,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.Try,
    ast.TryStar,
)
_FROM_MODULE: tuple[Binding, ...] = (("module",),)


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


def _header(node: ast.stmt) -> list[ast.AST]:
    """What a compound statement evaluates itself, outside its bodies."""

    if isinstance(node, ast.If | ast.While):
        return [node.test]
    if isinstance(node, ast.For | ast.AsyncFor):
        return [node.iter]
    if isinstance(node, ast.With | ast.AsyncWith):
        return [item.context_expr for item in node.items]
    if isinstance(node, ast.Match):
        parts: list[ast.AST] = [node.subject]
        for case in node.cases:
            parts.append(case.pattern)
            if case.guard is not None:
                parts.append(case.guard)
        return parts
    if isinstance(node, ast.Try | ast.TryStar):
        return [handler.type for handler in node.handlers if handler.type is not None]
    return []


def _parts(node: ast.stmt) -> list[ast.AST]:
    """What one statement runs itself: a compound statement's header, or all of it."""

    return _header(node) if isinstance(node, _COMPOUND) else [node]


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


def _captures(pattern: ast.AST) -> Iterator[str]:
    """The names a ``match`` pattern captures."""

    for part in ast.walk(pattern):
        if isinstance(part, ast.MatchAs | ast.MatchStar) and part.name:
            yield part.name
        elif isinstance(part, ast.MatchMapping) and part.rest:
            yield part.rest


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
    elif isinstance(node, ast.Match):
        # A capture holds the subject or a part of it.
        for case in node.cases:
            for name in _captures(case.pattern):
                yield ast.Name(name, ast.Store()), node.subject
    if not isinstance(node, _DEFS):
        for part in _parts(node):
            for child in ast.walk(part):
                if isinstance(child, ast.NamedExpr):
                    yield child.target, child.value


def _stores_into(node: ast.stmt) -> bool:
    """Whether a statement stores to an item or an attribute."""

    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets = [node.target]
    else:
        return False
    return any(
        isinstance(part, ast.Attribute | ast.Subscript)
        and isinstance(part.ctx, ast.Store)
        for target in targets
        for part in ast.walk(target)
    )


def _effects(body: Iterable[ast.stmt]) -> Iterator[ast.AST]:
    """What a module or class body runs for its effects, through every block.

    Each statement that runs only for its effect (an expression statement, a
    ``del``, a store to an item or an attribute), and each compound
    statement's header.
    """

    for node in _statements(body):
        if isinstance(node, _COMPOUND):
            yield from _header(node)
        elif isinstance(node, ast.Expr | ast.Delete) or _stores_into(node):
            yield node


def _mutations(part: ast.AST) -> Iterator[str]:
    """The names one statement (or one header) can change in place.

    ``_TABLE`` for ``_TABLE.update(...)``, ``_ROW = _TABLE.setdefault(...)``,
    ``_TABLE[key] = value``, ``_TABLE.field += 1``, ``del _TABLE[key]``, and
    for a call statement each name its arguments start from:
    ``_fill(_TABLE)`` or ``setattr(_TABLE, ...)``.
    """

    calls = []
    if isinstance(part, ast.Expr) and isinstance(part.value, ast.Call):
        calls.append(part.value)
    for node in _same_scope([part]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = _root_name(node.func.value)
            if name is not None:
                yield name
        elif isinstance(node, ast.Attribute | ast.Subscript) and isinstance(
            node.ctx, ast.Store | ast.Del
        ):
            name = _root_name(node.value)
            if name is not None:
                yield name
    for call in calls:
        for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
            name = _root_name(argument)
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
        #: (name, part, class body or None): each module-level or class-level
        #: statement (or compound header) that changes a name in place, once
        #: for each name it changes.
        self.mutations: list[tuple[str, ast.AST, ast.ClassDef | None]] = []
        #: What the module runs at import beyond its bindings.
        self.effects: list[ast.AST] = list(_effects(tree.body))
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
                for child in _statements(node.body):
                    if not isinstance(child, (*_DEFS, ast.Import, ast.ImportFrom)):
                        self._changes(child, node)
            else:
                for target, value in _assignments(node):
                    for bound in _stored(target):
                        _add(constants[bound], value)
                self._changes(node, None)
        self.imports = dict(imports)
        self.functions = dict(functions)
        self.classes = dict(classes)
        self.constants = dict(constants)

    def _changes(self, node: ast.stmt, body: ast.ClassDef | None) -> None:
        for part in _parts(node):
            for bound in dict.fromkeys(_mutations(part)):
                self.mutations.append((bound, part, body))

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

    @property
    def package(self) -> str:
        """``__package__``: the module's own name for a package, else its parent's."""

        return self.name if self.is_package else self.name.rpartition(".")[0]


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
    """One class body's names, each with the statements that bind it.

    A name the body declares ``global`` binds at module level instead.
    """

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
    declared = {
        name
        for child in _same_scope(_inner_parts(node))
        if isinstance(child, ast.Global)
        for name in child.names
    }
    return {name: nodes for name, nodes in out.items() if name not in declared}


def _binds(node: ast.stmt) -> set[str]:
    """The names a statement binds whenever it runs to its end."""

    if isinstance(node, _DEFS):
        return {node.name}
    if isinstance(node, ast.Import | ast.ImportFrom):
        return {
            alias.asname or alias.name.split(".")[0]
            for alias in node.names
            if alias.name != "*"
        }
    if isinstance(node, ast.Assign):
        return {name for target in node.targets for name in _stored(target)}
    if isinstance(node, ast.AnnAssign | ast.AugAssign) and node.value is not None:
        return set(_stored(node.target))
    if isinstance(node, ast.With | ast.AsyncWith):
        return {
            name
            for item in node.items
            if item.optional_vars is not None
            for name in _stored(item.optional_vars)
        }
    return set()


def _bound_before(body: ast.ClassDef, name: str, node: ast.AST) -> bool:
    """Whether the class ``body`` has surely bound ``name`` when ``node`` runs.

    Only a statement at the body's top level that ends before ``node`` starts
    binds for sure: a branch, a loop or a ``try`` can skip its bindings, a bare
    annotation binds nothing, and a ``del`` can undo a binding.
    """

    if not hasattr(node, "lineno"):
        return False
    for child in _same_scope(_inner_parts(body)):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Del):
            if child.id == name:
                return False
    start = (node.lineno, node.col_offset)
    for statement in body.body:
        if (statement.end_lineno, statement.end_col_offset) > start:
            return False
        if name in _binds(statement):
            return True
    return False


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


def _own_names(
    module: _Module, scope: ast.AST
) -> tuple[dict[str, tuple[Binding, ...]], set[str]]:
    """The names ``scope`` binds for itself, and the names it declares global.

    Each name maps to every import that binds it there: a name that a ``try``
    or an ``if`` imports two ways keeps both.
    """

    names: dict[str, tuple[Binding, ...]] = {}

    def local(name: str) -> None:
        names.setdefault(name, ())

    if isinstance(scope, _COMPREHENSIONS):
        for generator in scope.generators:
            for name in _stored(generator.target):
                local(name)
        return names, set()
    if isinstance(scope, _FUNCTIONS):
        for argument in _arguments(scope.args):
            local(argument.arg)
    declared_global: set[str] = set()
    declared_nonlocal: set[str] = set()
    for node in _same_scope(_inner_parts(scope)):
        if isinstance(node, ast.Global):
            declared_global.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            declared_nonlocal.update(node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            local(node.id)
        elif isinstance(node, _DEFS):
            local(node.name)
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar):
            if node.name:
                local(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            local(node.rest)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for bound, binding in _bindings(module, node).items():
                names[bound] = (*names.get(bound, ()), ("import", *binding))
        elif isinstance(node, _COMPREHENSIONS):
            for target in _walrus_targets(node):
                local(target)
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
            if isinstance(current, ast.ClassDef):
                # Python reads a name the class body binds from the class
                # namespace or, before the body binds it, from the module:
                # never from an enclosing function.
                own = {
                    name: (*bindings, *_FROM_MODULE) for name, bindings in own.items()
                }
            inner = {
                name: bindings
                for name, bindings in {**closure, **own}.items()
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
    """The names and dotted names an expression's value can be.

    ``getattr(pkg, "mod")`` counts as ``pkg.mod``, and a literal that
    ``import_module`` or ``__import__`` takes comes back as that constant.
    """

    if isinstance(value, ast.Name | ast.Attribute):
        yield value
    elif isinstance(value, ast.Subscript):
        yield from _named(value.value)
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
    elif isinstance(value, ast.Call):
        callee = _callee(value.func)
        texts = [_literal(argument) for argument in value.args]
        if callee == "getattr" and len(texts) >= 2 and texts[1]:
            read = ast.Attribute(value.args[0], texts[1], ast.Load())
            yield ast.copy_location(read, value)
        elif callee in IMPORTERS and texts and texts[0]:
            yield ast.Constant(texts[0])
            if callee == "__import__":
                yield ast.Constant(texts[0].partition(".")[0])


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
        self.reached_modules: set[str] = set()
        self.pending: list[Item] = []
        self.classes: set[tuple[str, str]] = set()
        #: Modules the walk has seen used as a value, not as an attribute's root.
        self.values: set[str] = set()
        self.attributes: set[str] = set()
        self.cuts_used: set[Cut] = set()
        #: The import-time statements and changes in place the walk has run.
        self.ran: set[int] = set()
        self._members: dict[tuple[str, str], dict[str, list[ast.AST]]] = {}
        #: A top-level class member's binding statement -> its class body.
        self._body_of: dict[int, ast.ClassDef] = {}
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

    def push(self, item: Item, site: str) -> bool:
        """Record that ``site`` names ``item``; whether ``item`` is new."""

        # A namespace package has no file, and so nothing to hash.
        if item[0] not in self.paths:
            return False
        new = item not in self.reached
        if new:
            self.pending.append(item)
            self.reached_modules.add(item[0])
        self.reached[item].add(site)
        return new

    def members(self, module: str, name: str) -> dict[str, list[ast.AST]]:
        """A top-level class's body names, across every ``class`` that defines it."""

        key = (module, name)
        if key not in self._members:
            found = self.module(module)
            out: dict[str, list[ast.AST]] = defaultdict(list)
            for node in found.classes.get(name, ()):
                for member, nodes in _members(found, node).items():
                    out[member].extend(nodes)
                    for child in nodes:
                        self._body_of[id(child)] = node
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
                items.extend(self.exported(star, name))
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
        if items or found is None:
            return items
        # A lazy export: whatever binds the module's __getattr__ and, for a
        # package, the name wherever the package's own modules bind it.
        getters = self.resolve(module, "__getattr__")
        if not getters:
            return items
        items.extend(getters)
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
                    if isinstance(named, ast.Constant):
                        if self.is_module(named.value):
                            found.append((named.value, "module", None))
                        continue
                    for resolved in self.resolve_expression(item[0], named):
                        found.extend(self.aliased(resolved))
        finally:
            self._aliasing.discard(item)
        return found or [item]

    def member_imports(self, module: str, owner: str, member: str) -> list[Item]:
        """What a top-level class member is where the class body imports it.

        A member that an assignment binds needs nothing here: the walk reaches
        a class's non-method members with the class, and a module that a value
        names counts as a value.
        """

        found = self.module(module)
        return [
            item
            for node in self.members(module, owner).get(member, ())
            if isinstance(node, ast.Import | ast.ImportFrom)
            for item in self.resolve_import(*_bindings(found, node)[member])
        ]

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
        body: ast.ClassDef | None = None,
    ) -> None:
        """Push every first-party item ``node``'s code can name.

        With an ``owner``, ``node`` sits in that top-level class's ``body``, so
        the body's names resolve to the class's members, and also to the
        module's binding wherever the body may not have bound them yet.
        """

        found = self.module(module)
        skipped = self.skipped(module, qualname, node)
        start: Scope = (
            dict.fromkeys(self.members(module, owner), (("member", owner),))
            if owner is not None
            else {}
        )
        # Names that root an attribute chain; any other module name is a value.
        rooted: set[int] = set()
        # Attributes that a longer chain extends.
        prefixes: set[int] = set()

        def lookup(name: str, here: Scope, read: ast.AST) -> list[Item]:
            if name not in here:
                return self.resolve(module, name)
            items: list[Item] = []
            for binding in here[name]:
                if binding[0] == "import":
                    items.extend(self.resolve_import(binding[1], binding[2]))
                elif binding[0] == "module":
                    items.extend(self.resolve(module, name))
                else:
                    items.append((module, "member", f"{binding[1]}.{name}"))
                    if body is None or not _bound_before(body, name, read):
                        items.extend(self.resolve(module, name))
            return items

        def literal_names(node: ast.AST, here: Scope) -> list[str]:
            """The strings ``node`` can be: a literal, or a constant of literals."""

            text = _literal(node)
            if text is not None:
                return [text]
            if not isinstance(node, ast.Name) or node.id in here:
                return []
            values = [
                value
                for item in self.resolve(module, node.id)
                if item[1] == "constant"
                for value in self.module(item[0]).constants[item[2]]
            ]
            texts = [_literal(value) for value in values]
            return texts if texts and None not in texts else []

        def named_items(node: ast.AST, here: Scope) -> list[Item]:
            """What a name or dotted name names where it runs, without pushing."""

            chain = []
            while isinstance(node, ast.Attribute):
                chain.append(node.attr)
                node = node.value
            if not isinstance(node, ast.Name):
                return []
            items = lookup(node.id, here, node)
            for attribute in reversed(chain):
                items = [
                    item
                    for start in items
                    for module_item in self.aliased(start)
                    if module_item[1] == "module"
                    for item in self.exported(module_item[0], attribute)
                ]
            return items

        def attribute(node: ast.Attribute, here: Scope, at: str) -> None:
            self.attributes.add(node.attr)
            if isinstance(node.value, ast.Attribute):
                prefixes.add(id(node.value))
            chain = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if not isinstance(current, ast.Name):
                return
            rooted.add(id(current))
            if owner is not None and current.id in {"self", "cls"}:
                # The member itself comes from the attribute fallback.
                member = chain.pop()
                found_items = self.member_imports(module, owner, member)
            else:
                found_items = lookup(current.id, here, current)
            starts = [
                start
                for item in found_items
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
            if chain and id(node) not in prefixes:
                # A dotted name for a module, used as a value.
                self.values.update(start[0] for start in starts)

        def call(node: ast.Call, here: Scope, at: str) -> None:
            callee = _callee(node.func)
            if callee in ATTRIBUTE_BUILTINS and len(node.args) >= 2:
                for text in literal_names(node.args[1], here):
                    read = ast.Attribute(node.args[0], text, ast.Load())
                    attribute(ast.copy_location(read, node), here, at)
            elif callee in ATTRIBUTE_NAMERS:
                for argument in node.args:
                    for text in literal_names(argument, here):
                        self.attributes.update(text.split("."))
            elif callee in IMPORTERS and node.args:
                for text in literal_names(node.args[0], here):
                    for name in self.absolute(module, text, node, here):
                        # import_module returns the module it names;
                        # __import__ returns the top package, or with a
                        # fromlist the module itself.
                        named = [name]
                        if callee == "__import__":
                            named.append(name.partition(".")[0])
                        for value in named:
                            if self.is_module(value):
                                self.push((value, "module", None), at)
                                self.values.add(value)

        def match_class(node: ast.MatchClass, here: Scope) -> None:
            self.attributes.update(node.kwd_attrs)
            if not node.patterns:
                return
            for item in named_items(node.cls, here):
                for start in self.aliased(item):
                    if start[1] == "class":
                        self.attributes.update(self.match_args(start))

        for child, here in _visits(found, node, skipped, start, {}):
            at = f"{site}:{getattr(child, 'lineno', '?')}"
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                for item in lookup(child.id, here, child):
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
                call(child, here, at)
            elif isinstance(child, ast.MatchClass):
                match_class(child, here)

    def absolute(
        self, module: str, text: str, node: ast.Call, here: Scope
    ) -> list[str]:
        """The module names an import call's literal can mean."""

        if not text.startswith("."):
            return [text]
        level = len(text) - len(text.lstrip("."))
        rest = text[level:]
        if len(node.args) > 1:
            package_node = node.args[1]
        else:
            package_node = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "package"
                ),
                None,
            )
        if package_node is None:
            return []
        found = self.module(module)
        if isinstance(package_node, ast.Name) and package_node.id == "__package__":
            packages = [found.package]
        elif isinstance(package_node, ast.Name) and package_node.id == "__name__":
            packages = [module]
        else:
            text_node = _literal(package_node)
            packages = [text_node] if text_node is not None else []
        names = []
        for package in packages:
            parts = package.split(".")
            if level > len(parts):
                continue
            base = ".".join(parts[: len(parts) - (level - 1)])
            names.append(f"{base}.{rest}" if rest else base)
        return names

    def match_args(self, item: Item) -> list[str]:
        """The attributes a class's explicit ``__match_args__`` names.

        A positional ``match`` class pattern reads them. A dataclass's own
        ``__match_args__`` names its fields, which hold data, not code.
        """

        module, _, name = item
        return [
            text
            for node in self.members(module, name).get("__match_args__", ())
            for _, value in _assignments(node)
            for element in getattr(value, "elts", ())
            if (text := _literal(element)) is not None
        ]

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
                # Creating the class runs its body.
                for part in _effects(node.body):
                    if id(part) not in self.ran:
                        self.ran.add(id(part))
                        self.scan(
                            module,
                            part,
                            f"{module}.{name}(runs)",
                            owner=name,
                            body=node,
                        )
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
                self.scan(
                    module,
                    node,
                    f"{module}.{name}",
                    qualname=name,
                    owner=owner,
                    body=self._body_of.get(id(node)),
                )
            if nodes:
                return
            # Inherited: the same name on each first-party base class.
            for node in found.classes[owner]:
                for base in node.bases:
                    while isinstance(base, ast.Subscript):
                        base = base.value
                    for resolved in self.resolve_expression(module, base):
                        for start in self.aliased(resolved):
                            if start[1] == "class":
                                self.push(
                                    (start[0], "member", f"{start[2]}.{member}"),
                                    f"{module}.{name}(base)",
                                )

    def is_reached(self, item: Item) -> bool:
        if item[1] == "module":
            return item[0] in self.reached_modules
        return item in self.reached

    def changes_reached(
        self, found: _Module, bound: str, body: ast.ClassDef | None
    ) -> bool:
        """Whether changing ``bound`` where ``found`` binds it changes a reached item."""

        items = self.resolve(found.name, bound)
        if body is not None and bound in _members(found, body):
            items.append((found.name, "member", f"{body.name}.{bound}"))
        return any(
            self.is_reached(alias)
            for item in items
            for alias in dict.fromkeys((item, *self.aliased(item)))
        )

    def settle(self) -> bool:
        """Push what reached names imply; whether anything new came."""

        added = False
        for module_name, class_name in sorted(self.classes):
            for member in self.members(module_name, class_name):
                item = (module_name, "member", f"{class_name}.{member}")
                if (
                    member in self.attributes
                    and item not in self.reached
                    and self.push(item, f"{module_name}.{class_name}(attribute)")
                ):
                    added = True
        for module_name in sorted(self.values):
            for name in sorted(self.attributes):
                for item in self.exported(module_name, name):
                    if item not in self.reached and self.push(
                        item, f"{module_name}(value).{name}"
                    ):
                        added = True
                    if item[1] == "module" and item[0] not in self.values:
                        self.values.add(item[0])
                        added = True
        # Import-time code, in each module that holds a reached item.
        for module_name in sorted(self.reached_modules):
            for part in self.module(module_name).effects:
                if id(part) not in self.ran:
                    self.ran.add(id(part))
                    self.scan(
                        module_name,
                        part,
                        f"{module_name}:{getattr(part, 'lineno', '?')}(runs)",
                    )
                    added = True
        # Changes in place, in every module the walk has read.
        for found in list(self.modules.values()):
            for bound, part, body in found.mutations:
                if id(part) in self.ran or not self.changes_reached(found, bound, body):
                    continue
                self.ran.add(id(part))
                self.scan(
                    found.name,
                    part,
                    f"{found.name}:{getattr(part, 'lineno', '?')}(changes)",
                    owner=None if body is None else body.name,
                    body=body,
                )
                added = True
        return added

    def run(self, module: str, qualname: str) -> None:
        if "." in qualname:
            # A method's class, with its bases, decorators and dunders.
            self.push((module, "class", qualname.split(".", 1)[0]), "root")
            self.push((module, "member", qualname), "root")
        else:
            self.push((module, "function", qualname), "root")
        for _ in range(MAX_ROUNDS):
            while self.pending:
                self.process(self.pending.pop())
            if not self.settle() and not self.pending:
                return
        raise RuntimeError(
            f"the walk from {module}.{qualname} didn't settle in {MAX_ROUNDS} rounds"
        )


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
