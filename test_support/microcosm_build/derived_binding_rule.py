"""The derived binding rule's walk, shared by every country's graph tests.

``source_hash`` does not follow imports, so a node's hash must name every
module whose code can change what the node computes (microcosm#931). The rule
derives those modules from what a stage's transform, or a graph kernel's own
code, names, and follows each named definition into the module that defines
it, through re-exports, lazy exports and module attributes, to a fixed point.

The machinery here is country-neutral. A country declares where its rule
stops in a ``RuleTables`` (its core modules, refusal-only checks, kernel
stops and declaration modules), each entry with a reason its own tests
check, and walks through one ``BindingRule`` built from it. Its tests hold
each node's key to the walks. test_uk_graph.py holds the UK stages and
kernels to it; tests/engine_free/shared/test_derived_binding_rule.py holds
the controls.

What the rule cannot see: a call through any other object whose class the
code never names, a name looked up by a computed string, and data files read
by path.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import NamedTuple

#: The packages the rule follows: every first-party package, each under the
#: microcosm namespace (microcosm-build, -calibrate, -data, -diagnostics,
#: -fit, -frame and -graph).
FIRST_PARTY_SCOPE = "microcosm."
#: The Frame's package. A country lists the Frame's modules among its core
#: modules, and the reason it gives them covers each definition a kernel
#: names there, which kernels build and read frames with, so ``findings``
#: reports the core definitions kernels name outside it.
FRAME_SCOPE = "microcosm.frame."
#: Packages that export names lazily, through a module ``__getattr__``, each
#: with the table, of name to the submodule defining it, that ``__getattr__``
#: reads. The walk resolves such a name to the module that defines it and
#: follows it like any other (microcosm#931 ruling 8). microcosm.graph imports
#: its runtime modules, which need numpy, pandas and microcosm-frame, only on
#: first access. A fact about the first-party packages, not about a country:
#: test_every_lazy_export_resolves_to_the_object_python_returns checks it.
LAZY_EXPORT_TABLES: Mapping[str, str] = MappingProxyType(
    {"microcosm.graph": "_RUNTIME_EXPORTS"}
)
#: First-party modules that define a module ``__getattr__`` but export
#: nothing through it, each with why.
NO_LAZY_EXPORTS: Mapping[str, str] = MappingProxyType(
    {
        "microcosm.build.us_runtime.sources": (
            "its __getattr__ raises for every name: it only words the error for "
            "the names it removed"
        ),
    }
)
#: Method and function names that change their receiver or an argument in
#: place.
IN_PLACE_CALLS = frozenset(
    {
        "__delitem__",
        "__setattr__",
        "__setitem__",
        "add",
        "append",
        "appendleft",
        "at",
        "clear",
        "copyto",
        "difference_update",
        "discard",
        "extend",
        "extendleft",
        "fill",
        "fill_diagonal",
        "insert",
        "intersection_update",
        "itemset",
        "place",
        "pop",
        "popitem",
        "put",
        "putmask",
        "remove",
        "resize",
        "reverse",
        "rotate",
        "setdefault",
        "setflags",
        "shuffle",
        "sort",
        "symmetric_difference_update",
        "update",
    }
)

#: The builtins whose call returns a new container (``local_containers``).
FRESH_CONTAINER_CALLS = frozenset({"dict", "list", "set"})
#: The graph's records, enums and constants that kernels' code names, each
#: with why naming it binds nothing a kernel's key misses. They are facts
#: about the graph, the same for every country, so a country's table of the
#: core definitions its kernels name takes these reasons from here and gives
#: its own only to definitions whose reason rests on its own checks.
GRAPH_RECORD_REASONS: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        ("microcosm.graph.decl", "ArtifactType"): (
            "a frozen record of an artifact type's name and version, which a "
            "kernel names to declare or check its artifacts' types; node_key "
            "hashes each node's declared artifact types as part of its "
            "normative declaration"
        ),
        ("microcosm.graph.decl", "ArtifactOutput"): (
            "a frozen record pairing an artifact's name with its type, which a "
            "kernel names to declare and check its outputs, as the shared atomic "
            "geography gate does; node_key hashes each node's declared artifact "
            "outputs"
        ),
        ("microcosm.graph.decl", "StructuralDelta"): (
            "the enum a kernel's capabilities declare its row-set change in; the "
            "executor refuses a kernel whose declared change differs from its "
            "node's, and node_key binds the declared value with the rest of the "
            "kernel's capabilities"
        ),
        ("microcosm.graph.decl", "GATE_OUTCOMES"): (
            "the closed set of outcomes a gate may record, to which the executor "
            "holds every gate node's outcome; a kernel names it to refuse a gate "
            "report recording another, and an edit to it changes what every gate "
            "may record alike"
        ),
        ("microcosm.graph.decl", "Node"): (
            "the frozen record of a node's declaration, which a kernel names to "
            "annotate or check the node its context carries; node_key hashes "
            "each node's normative declaration"
        ),
        ("microcosm.graph.decl", "Ownership"): (
            "the enum of what a node promises about the cells it owns, which a "
            "kernel names to check its node's declared outputs; node_key hashes "
            "each node's declared outputs with their ownership"
        ),
        ("microcosm.graph.decl", "ROWS_ALL"): (
            "the rows token of a slice or output that covers every row, which a "
            "kernel names to refuse a node that declares fewer; node_key hashes "
            "each node's slices and outputs with their rows"
        ),
        ("microcosm.graph.decl", "Slice"): (
            "the frozen record of what a node reads, one entity's columns under "
            "a row mask, which a kernel names to annotate or check its node's "
            "slices; it only checks its fields, and node_key hashes each node's "
            "declared slices"
        ),
        ("microcosm.graph.kernel", "ArtifactValue"): (
            "the frozen record of an artifact's verified bytes and typed "
            "provenance that the executor hands a kernel, which a kernel names "
            "to annotate its inputs; it only checks and freezes its fields, and "
            "node_key binds each typed artifact input's producer key and type"
        ),
        ("microcosm.graph.kernel", "KernelBase"): (
            "the base class of the kernels, whose default implementation hash "
            "hashes a kernel class's own module; a kernel's code names it as a "
            "class's base, and it computes nothing a node's values depend on"
        ),
        ("microcosm.graph.kernel", "KernelRole"): (
            "the enum a kernel's capabilities declare its role in, by which the "
            "executor holds a gate's outcome to the gate outcomes and derives a "
            "release's tier from its gate ancestors; node_key binds the declared "
            "value with the rest of the kernel's capabilities"
        ),
        ("microcosm.graph.kernel", "Capabilities"): (
            "the frozen record of a kernel's declared contract, which a kernel "
            "names to declare its capabilities; it only checks the types and "
            "consistency of its fields, and node_key binds every field it holds "
            "(keys._capabilities_projection)"
        ),
        ("microcosm.graph.kernel", "Determinism"): (
            "the enum a kernel's capabilities declare its determinism in, "
            "whether its output is a function of its inputs and seed; node_key "
            "binds the declared value with the rest of the kernel's capabilities"
        ),
        ("microcosm.graph.kernel", "Numeric"): (
            "the enum a kernel's capabilities declare its numbers' "
            "reproducibility in, from which the executor resolves the numeric "
            "scope of each column a reader is given; node_key binds the declared "
            "value with the rest of the kernel's capabilities"
        ),
        ("microcosm.graph.kernel", "SeedSource"): (
            "the enum a kernel's capabilities declare its seed source in, where "
            "a seeded kernel takes its randomness from; node_key binds the "
            "declared value with the rest of the kernel's capabilities"
        ),
    }
)


@dataclass(frozen=True)
class RuleTables:
    """One country's declaration of where its rule stops, each entry with the
    reason its tests check.

    - ``core_modules``: the modules every node runs through, which the rule
      neither follows nor asks a node to bind (``BindingRule.walk``).
    - ``refusal_only``: checks that read values only to decide whether to
      refuse, which the rule does not follow; a kernel's method is named
      ``Class.method``, and the kernel rule leaves it out of what the kernel
      runs (``BindingRule.kernel_body``).
    - ``stops``: where the kernel rule stops by default
      (``BindingRule.kernel_walk``).
    - ``declaration_modules``: modules the rule neither follows nor returns,
      because a node's key already hashes what it reads from them; none by
      default.
    - ``lazy_export_tables``: the packages' lazy exports
      (``LAZY_EXPORT_TABLES`` by default).
    - ``scope``: the packages the rule follows (``FIRST_PARTY_SCOPE``).
    """

    core_modules: Mapping[str, str]
    refusal_only: Mapping[tuple[str, str], str] = MappingProxyType({})
    stops: Mapping[tuple[str, str], str] = MappingProxyType({})
    declaration_modules: tuple[str, ...] = ()
    lazy_export_tables: Mapping[str, str] = LAZY_EXPORT_TABLES
    scope: str = FIRST_PARTY_SCOPE


class ModuleDefinitions(NamedTuple):
    """What one module imports, anywhere in it, and defines at its top level."""

    imports: dict[str, tuple[str, str]]
    modules: dict[str, str]
    stars: tuple[str, ...]
    definitions: dict[str, object]


class Walk(NamedTuple):
    """What a root can compute through (``BindingRule.helper_walk``)."""

    reach: dict[str, frozenset[str]]
    stops: frozenset[tuple[str, str]]
    core: frozenset[str]


class KernelWalk(NamedTuple):
    """What a kernel's own code can compute through
    (``BindingRule.kernel_walk``)."""

    reach: dict[str, frozenset[str]]
    stops: frozenset[tuple[str, str]]
    core: frozenset[str]
    met: dict[tuple[str, str], frozenset]
    core_names: frozenset[tuple[str, str]]


class KeyedWalk(NamedTuple):
    """A walk the kernel rule holds a key to, and the modules that key
    hashes."""

    walk: KernelWalk
    key: frozenset[str]


class Findings(NamedTuple):
    """What a country's keyed walks find (``findings``): by label, the modules
    a walk reaches that its key does not hash; the names the rule stops at
    outside each key; the listed stops met; and the core definitions named
    outside the Frame's modules."""

    unhashed: dict[str, list[str]]
    unlisted: frozenset[tuple[str, str]]
    met: frozenset[tuple[str, str]]
    core_names: frozenset[tuple[str, str]]


@functools.cache
def is_module(name: str) -> bool:
    """Whether a first-party dotted name is a module. The rule never follows
    another package, so it never imports one to find out."""

    if not name.startswith(FIRST_PARTY_SCOPE):
        return False
    parent = name.rpartition(".")[0]
    try:
        if parent:
            spec = importlib.util.find_spec(parent)
            if spec is None or spec.submodule_search_locations is None:
                return False
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@functools.cache
def module_definitions(name: str, scope: str) -> ModuleDefinitions | None:
    if not name.startswith(scope):
        return None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not (spec.origin or "").endswith(".py"):
        return None
    package = name if spec.submodule_search_locations else name.rpartition(".")[0]
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    imports: dict[str, tuple[str, str]] = {}
    modules: dict[str, str] = {}
    stars: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                origin = ".".join(part for part in (base, node.module) if part)
            else:
                origin = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    stars.append(origin)
                elif is_module(f"{origin}.{alias.name}"):
                    modules[alias.asname or alias.name] = f"{origin}.{alias.name}"
                else:
                    imports[alias.asname or alias.name] = (origin, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    head = alias.name.partition(".")[0]
                    modules[head] = head
    definitions: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            definitions[node.name] = node
        elif isinstance(node, ast.TypeAlias):
            definitions[node.name.id] = node
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        definitions[sub.id] = node
    return ModuleDefinitions(imports, modules, tuple(stars), definitions)


def _parameters(node) -> set[str]:
    arguments = node.args
    return {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *(a for a in (arguments.vararg, arguments.kwarg) if a is not None),
        )
    }


def _fresh_container(value, calls: frozenset[str]) -> bool:
    """A new container: a dict, list or set display or comprehension, or a
    call by one of the names ``calls``."""

    if isinstance(
        value, ast.Dict | ast.List | ast.Set | ast.DictComp | ast.ListComp | ast.SetComp
    ):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in calls
    )


def _own_scope(node) -> set[int]:
    """The nodes of ``node``'s own scope, by id: everything in it but the
    functions, lambdas and classes nested in it."""

    found: set[int] = set()
    pending = list(ast.iter_child_nodes(node))
    while pending:
        sub = pending.pop()
        found.add(id(sub))
        if not isinstance(
            sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef
        ):
            pending.extend(ast.iter_child_nodes(sub))
    return found


def local_containers(node) -> frozenset[str]:
    """The names a function binds only to containers it creates itself.

    A name is local when the function's own scope binds it to a new
    container (``name = <fresh>`` or ``name: T = <fresh>``, where fresh is a
    dict, list or set display or comprehension, or a call of ``dict``,
    ``list`` or ``set``), every other binding of it anywhere in the function,
    nested scopes included, is one too, and it is a parameter of neither the
    function nor anything nested in it. A bare annotation binds nothing, and
    an augmented assignment to the name itself keeps it local, since a
    container's augmented operators change it in place or refuse. A call
    counts only where the function binds no name ``dict``, ``list`` or
    ``set`` of its own; one its module binds at its top level is not seen."""

    parameters: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            parameters |= _parameters(sub)
    # Each single-name binding with its value, and the name targets they and
    # the bindings of no new object (a bare annotation, an augmented
    # assignment to a name) account for.
    valued: dict[int, tuple[str, ast.expr]] = {}
    accounted: set[int] = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Assign)
            and len(sub.targets) == 1
            and isinstance(sub.targets[0], ast.Name)
        ):
            valued[id(sub)] = (sub.targets[0].id, sub.value)
            accounted.add(id(sub.targets[0]))
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            if sub.value is not None:
                valued[id(sub)] = (sub.target.id, sub.value)
            accounted.add(id(sub.target))
        elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
            accounted.add(id(sub.target))
    # Every other binding: loop, with, walrus, unpacking and comprehension
    # targets, deletions, imports, nested definitions, exception and match
    # captures.
    other: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store | ast.Del):
            if id(sub) not in accounted:
                other.add(sub.id)
        elif isinstance(sub, ast.alias):
            other.add((sub.asname or sub.name).partition(".")[0])
        elif (
            isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and sub is not node
        ):
            other.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler | ast.MatchAs | ast.MatchStar):
            if sub.name is not None:
                other.add(sub.name)
        elif isinstance(sub, ast.MatchMapping) and sub.rest is not None:
            other.add(sub.rest)
    stored = {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store | ast.Del)
    }
    calls = FRESH_CONTAINER_CALLS - (parameters | other | stored)
    own = _own_scope(node)
    fresh = {
        name
        for key, (name, value) in valued.items()
        if key in own and _fresh_container(value, calls)
    }
    stale = {
        name for name, value in valued.values() if not _fresh_container(value, calls)
    }
    return frozenset(fresh - stale - other - parameters)


def _stored(target):
    """The parts of an assignment or deletion target that receive a value or
    lose one: the target, or each element of a tuple or list target, starred
    or not, at any depth."""

    if isinstance(target, ast.Tuple | ast.List):
        for element in target.elts:
            yield from _stored(element)
    elif isinstance(target, ast.Starred):
        yield from _stored(target.value)
    else:
        yield target


def _item_of(target, local: frozenset[str]) -> bool:
    """``target`` is one item of a local container: ``name[...]``."""

    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id in local
    )


def writes_nothing(node) -> bool:
    """Whether a function can only refuse: it returns nothing but None, or its
    sole parameter unrebound, and writes nothing outside itself: no store
    through a subscript or attribute, no augmented assignment, no ``out=`` or
    ``inplace=`` argument, and no call by a name that changes its receiver or
    an argument in place.

    A container the function creates is its own (``local_containers``): it
    may store into or delete one item of it, call an in-place method on it,
    or augment it as a whole. A nested store (into an item of an item, or an
    attribute of one), an augmented store into an item, which can change in
    place an object the function was handed, and a store through any other
    name, an alias of the container included, are still writes, and
    returning the container returns a value."""

    parameters = _parameters(node)
    rebound = {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store | ast.Del)
    }
    returnable = parameters - rebound if len(parameters) == 1 else set()
    local = local_containers(node)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Return) and sub.value is not None:
            value = sub.value
            if not (
                (isinstance(value, ast.Constant) and value.value is None)
                or (isinstance(value, ast.Name) and value.id in returnable)
            ):
                return False
        elif isinstance(sub, ast.AugAssign):
            if not (isinstance(sub.target, ast.Name) and sub.target.id in local):
                return False
        elif isinstance(sub, ast.Yield | ast.YieldFrom | ast.Global | ast.Nonlocal):
            return False
        elif isinstance(sub, ast.Assign | ast.AnnAssign | ast.Delete):
            targets = (
                sub.targets
                if isinstance(sub, ast.Assign | ast.Delete)
                else [sub.target]
            )
            if any(
                isinstance(inner, ast.Subscript | ast.Attribute)
                for target in targets
                for part in _stored(target)
                if not _item_of(part, local)
                for inner in ast.walk(part)
            ):
                return False
        elif isinstance(sub, ast.keyword) and sub.arg in {"inplace", "out"}:
            return False
        elif isinstance(sub, ast.Call):
            function = sub.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr in IN_PLACE_CALLS
                and not (
                    isinstance(function.value, ast.Name) and function.value.id in local
                )
            ):
                return False
            if isinstance(function, ast.Name) and function.id in {
                *IN_PLACE_CALLS,
                "delattr",
                "eval",
                "exec",
                "setattr",
            }:
                return False
    return True


def string_expression(node) -> bool:
    """A string literal, an f-string of names or of one call on names, or a
    concatenation of them."""

    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return string_expression(node.left) and string_expression(node.right)
    if not isinstance(node, ast.JoinedStr):
        return False
    for part in node.values:
        value = part.value if isinstance(part, ast.FormattedValue) else part
        if isinstance(value, ast.Call):
            if not isinstance(value.func, ast.Name) or value.keywords:
                return False
            value = value.args[0] if len(value.args) == 1 else None
        if not isinstance(value, ast.Name | ast.Constant):
            return False
    return True


def calls_super(node) -> bool:
    return any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "super"
        for sub in ast.walk(node)
    )


def statement_names(statement) -> list[str]:
    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
        return [statement.name]
    if isinstance(statement, ast.Assign | ast.AnnAssign):
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        return [
            sub.id
            for target in targets
            for sub in ast.walk(target)
            if isinstance(sub, ast.Name)
        ]
    return []


def self_attributes(statement) -> set[str]:
    """The attributes ``statement`` reads on ``self``, ``cls`` or
    ``type(self)``."""

    found = set()
    for sub in ast.walk(statement):
        if not isinstance(sub, ast.Attribute):
            continue
        value = sub.value
        if (isinstance(value, ast.Name) and value.id in {"self", "cls"}) or (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "type"
        ):
            found.add(sub.attr)
    return found


def key_only_methods(body, keyed) -> frozenset[str]:
    """The methods of a kernel's ``body`` that only its key computation calls:
    named on ``self``, ``cls`` or ``type(self)`` by its ``keyed`` statements
    (``implementation_hash``) or by another such method, and by nothing else
    the kernel runs. They compute the key as the hash does."""

    methods = {
        name
        for _, statement, name in body
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    reads: dict[str, set[str]] = {}
    for _, statement, name in body:
        reads.setdefault(name, set()).update(self_attributes(statement))
    key = set().union(*(self_attributes(statement) for _, statement, _ in keyed))
    key_only: set[str] = set()
    while True:
        found = {
            method
            for method in methods - key_only
            if method in key.union(*(reads[item] for item in key_only))
            and not any(
                method in names
                for other, names in reads.items()
                if other != method and other not in key_only
            )
        }
        if not found:
            return frozenset(key_only)
        key_only |= found


def merged(first: Walk, second: Walk) -> Walk:
    reach = dict(first.reach)
    for module, names in second.reach.items():
        reach[module] = reach.get(module, frozenset()) | names
    return Walk(reach, first.stops | second.stops, first.core | second.core)


def class_named(ref: tuple[str, str]) -> type:
    module, name = ref
    return getattr(importlib.import_module(module), name)


def hashed_by(compute: Callable[[], object]) -> frozenset[str]:
    """Every module ``compute()`` hashes, recorded at each ``source_hash`` it
    reaches, in whichever module calls it: every loaded module's
    ``source_hash`` name stands in for the recorder during one call, after a
    first call has imported what ``compute`` imports."""

    from microcosm.graph import kernel as graph_kernel

    original = graph_kernel.source_hash
    recorded: list[object] = []

    def record(*objects, **dependencies):
        recorded.extend(objects)
        return original(*objects, **dependencies)

    compute()
    patched = [
        module
        for module in list(sys.modules.values())
        if isinstance(module, ModuleType)
        and getattr(module, "__dict__", {}).get("source_hash") is original
    ]
    for module in patched:
        module.source_hash = record
    try:
        compute()
    finally:
        for module in patched:
            module.source_hash = original
    return frozenset(
        (item if isinstance(item, ModuleType) else inspect.getmodule(item)).__name__
        for item in recorded
    )


def hashed_modules(kernel) -> frozenset[str]:
    """Every module ``kernel.implementation_hash()`` hashes (``hashed_by``)."""

    return hashed_by(kernel.implementation_hash)


def kernel_classes_defined(sources: Iterable[tuple[str, Path]]) -> set[type]:
    """Every kernel class the given module files define: each class whose body
    declares ``ref``, or that stores a ``ref`` attribute. ``sources`` pairs
    each module's dotted name with its file; a module is imported only where
    it defines one."""

    defined = set()
    for module, path in sources:
        for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef):
                continue
            declares_ref = any(
                "ref" in statement_names(statement) for statement in node.body
            ) or any(
                isinstance(sub, ast.Attribute)
                and sub.attr == "ref"
                and isinstance(sub.ctx, ast.Store)
                for sub in ast.walk(node)
            )
            if declares_ref:
                defined.add(getattr(importlib.import_module(module), node.name))
    return defined


def findings(walks: Mapping[str, Mapping[str, KeyedWalk]]) -> Findings:
    """What the keyed walks find, by kernel ref and label (``Findings``): each
    walk's modules outside its key, every name it stops at outside its key,
    the listed stops it meets and the core definitions it names outside the
    Frame's modules (``FRAME_SCOPE``)."""

    unhashed: dict[str, list[str]] = {}
    unlisted: set[tuple[str, str]] = set()
    met: set[tuple[str, str]] = set()
    core_names: set[tuple[str, str]] = set()
    for keyed_walks in walks.values():
        for label, (walk, key) in keyed_walks.items():
            missing = sorted(set(walk.reach) - key)
            if missing:
                unhashed[label] = missing
            unlisted.update(item for item in walk.stops if item[0] not in key)
            met |= set(walk.met)
            core_names.update(
                item for item in walk.core_names if not item[0].startswith(FRAME_SCOPE)
            )
    return Findings(
        unhashed, frozenset(unlisted), frozenset(met), frozenset(core_names)
    )


class BindingRule:
    """The rule's walk under one country's declaration (``RuleTables``).

    Each method takes the arguments of the function it was in
    test_uk_graph.py, the optional ``scope`` among them, which is the
    declaration's unless given. A rule keeps its own caches of each
    definition's kind and of the lazy exports, since both depend on its
    tables."""

    def __init__(self, tables: RuleTables) -> None:
        self.tables = tables
        self.scope = tables.scope
        self._kinds: dict[tuple[str, str, str], str | None] = {}
        self._lazy: dict[str, Mapping[str, str]] = {}

    def lazy_exports(self, module: str) -> Mapping[str, str]:
        """Each name ``module`` exports lazily, to the module defining it, read
        from the table its ``__getattr__`` reads (``lazy_export_tables``)."""

        if module not in self._lazy:
            table = self.tables.lazy_export_tables.get(module)
            if table is None:
                self._lazy[module] = MappingProxyType({})
            else:
                exports = getattr(importlib.import_module(module), table)
                self._lazy[module] = MappingProxyType(
                    {
                        name: f"{module}.{submodule}"
                        for name, submodule in exports.items()
                    }
                )
        return self._lazy[module]

    def resolve(
        self, module: str, name: str, scope: str | None = None
    ) -> tuple[str, str] | None:
        """The top-level definition ``name`` refers to in ``module``, through
        re-exports and lazy exports (``lazy_export_tables``); None for a
        module or anything outside ``scope``."""

        scope = self.scope if scope is None else scope
        pending, seen = [(module, name)], set()
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            module, name = item
            facts = module_definitions(module, scope)
            if facts is None:
                continue
            if name in facts.definitions:
                return item
            if name in facts.imports:
                pending.append(facts.imports[name])
                continue
            # As in Python, the module's __getattr__ supplies a name only where
            # none of its own attributes, its star imports' included, is one.
            lazy = self.lazy_exports(module).get(name)
            if lazy is not None:
                pending.append((lazy, name))
            pending.extend((star, name) for star in facts.stars)
        return None

    def named(
        self, module: str, node, scope: str | None = None
    ) -> set[tuple[str, str]]:
        """The top-level definitions ``node`` names, annotations included: a
        bare name, or a name read by attribute on a module, each resolved
        through re-exports."""

        scope = self.scope if scope is None else scope
        facts = module_definitions(module, scope)
        found = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                hit = self.resolve(module, sub.id, scope)
            elif isinstance(sub, ast.Attribute):
                chain = []
                value = sub
                while isinstance(value, ast.Attribute):
                    chain.insert(0, value.attr)
                    value = value.value
                target = (
                    facts.modules.get(value.id) if isinstance(value, ast.Name) else None
                )
                while target and chain and is_module(f"{target}.{chain[0]}"):
                    target = f"{target}.{chain.pop(0)}"
                hit = (
                    self.resolve(target, chain[0], scope) if target and chain else None
                )
            else:
                continue
            if hit is not None:
                found.add(hit)
        return found

    def helper_kind(self, module: str, name: str, scope: str | None = None) -> str:
        """How a top-level definition can bear on a value.

        ``value`` unless it is a listed refusal-only ``check``
        (``refusal_only``) or provably one of: a ``name`` (one string: a
        literal, an f-string or concatenation of names, or another name of
        these kinds); a ``column name`` (a function returning one such string
        built from its arguments); a ``record`` (a class of undefaulted fields
        whose types are of these kinds, no methods). A collection of strings
        is a value, since it selects rows, columns or codes, as is a number. A
        cycle reads as ``value``.
        """

        scope = self.scope if scope is None else scope
        if (module, name) in self.tables.refusal_only:
            return "check"
        key = (module, name, scope)
        if key in self._kinds:
            return self._kinds[key] or "value"
        self._kinds[key] = None
        self._kinds[key] = self._classify(module, name, scope)
        return self._kinds[key]

    def _names_only_harmless(self, module: str, node, scope: str) -> bool:
        return all(
            found[0].startswith(self.tables.declaration_modules)
            or self.helper_kind(*found, scope) != "value"
            for found in self.named(module, node, scope)
        )

    def _classify(self, module: str, name: str, scope: str) -> str:
        node = module_definitions(module, scope).definitions[name]
        # A type alias statement (``type X = ...``) reads as an assignment of
        # its value.
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.TypeAlias):
            if node.value is None or (
                (
                    isinstance(node.value, ast.Name | ast.Attribute)
                    or string_expression(node.value)
                )
                and self._names_only_harmless(module, node.value, scope)
            ):
                return "name"
            return "value"
        if isinstance(node, ast.ClassDef):
            fields = [item for item in node.body if isinstance(item, ast.AnnAssign)]
            fields_only = all(
                (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
                or isinstance(item, ast.Pass)
                or (isinstance(item, ast.AnnAssign) and item.value is None)
                for item in node.body
            )
            if fields_only and all(
                self._names_only_harmless(module, item, scope)
                for item in (
                    *node.bases,
                    *node.keywords,
                    *node.decorator_list,
                    *(field.annotation for field in fields),
                )
            ):
                return "record"
            return "value"
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            body = body[1:]
        if (
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and string_expression(body[0].value)
            and self._names_only_harmless(module, body[0].value, scope)
        ):
            return "column name"
        return "value"

    def helper_walk(
        self,
        root: tuple[str, str],
        scope: str | None = None,
        *,
        follow_core: bool = False,
    ) -> Walk:
        """What ``root`` can compute through, where the rule stops, and the
        core modules it meets.

        ``reach`` maps every module holding a definition that can change what
        ``root`` computes to those definitions: ``root`` itself, and each
        ``value`` definition it names, and they name, to a fixed point.
        ``stops`` is every ``name``, ``column name`` or ``record`` those
        definitions name, which the rule does not follow. ``core`` is every
        listed core module (``core_modules``) whose definitions they name,
        which the rule neither follows nor returns in ``reach``, unless
        ``follow_core``. The declaration modules and the listed refusal-only
        checks are neither followed nor returned."""

        scope = self.scope if scope is None else scope
        module, name = root
        node = module_definitions(module, scope).definitions[name]
        walk, _, _ = self.walk(
            {root},
            [(found, root) for found in self.named(module, node, scope)],
            scope,
            follow_core=follow_core,
        )
        return walk

    def walk(
        self,
        roots: set[tuple[str, str]],
        seeds: list[tuple[tuple[str, str], object]],
        scope: str | None = None,
        *,
        follow_core: bool = False,
        halts: Mapping[tuple[str, str], str] = MappingProxyType({}),
    ) -> tuple[Walk, dict[tuple[str, str], frozenset], frozenset[tuple[str, str]]]:
        """The walk behind ``helper_walk`` and ``kernel_walk``: ``roots`` are
        reached as given, and each ``(definition, namer)`` seed is followed as
        ``helper_walk`` follows what its root names. A definition in ``halts``
        is neither followed nor returned; the second result maps each one met
        to what named it, and the third holds every core module's definition
        met."""

        scope = self.scope if scope is None else scope
        declarations = self.tables.declaration_modules
        core_modules = self.tables.core_modules
        reached: dict[str, set[str]] = {}
        for module, name in roots:
            reached.setdefault(module, set()).add(name)
        stops: set[tuple[str, str]] = set()
        core: set[str] = set()
        core_names: set[tuple[str, str]] = set()
        met: dict[tuple[str, str], set] = {}
        pending, seen = list(seeds), set(roots)
        while pending:
            item, namer = pending.pop()
            if item in halts:
                met.setdefault(item, set()).add(namer)
                continue
            if item in seen:
                continue
            seen.add(item)
            module, name = item
            if module.startswith(declarations):
                continue
            if module in core_modules:
                core.add(module)
                core_names.add(item)
                if not follow_core:
                    continue
            kind = self.helper_kind(module, name, scope)
            if kind != "value":
                if kind != "check":
                    stops.add(item)
                continue
            reached.setdefault(module, set()).add(name)
            node = module_definitions(module, scope).definitions[name]
            pending.extend((found, item) for found in self.named(module, node, scope))
        walk = Walk(
            {module: frozenset(names) for module, names in reached.items()},
            frozenset(stops),
            frozenset(core),
        )
        return (
            walk,
            {item: frozenset(namers) for item, namers in met.items()},
            frozenset(core_names),
        )

    def helper_reach(
        self, root: tuple[str, str], scope: str | None = None
    ) -> dict[str, frozenset[str]]:
        """Every module holding a definition that can change what ``root``
        computes, with those definitions (``helper_walk``)."""

        return self.helper_walk(root, scope).reach

    def kernel_classes(self, kernel_class: type) -> list[type]:
        """The first-party classes a kernel's code comes from, in its method
        resolution order, without the core modules' (``KernelBase``)."""

        return [
            cls
            for cls in kernel_class.__mro__
            if cls.__module__.startswith(self.scope)
            and cls.__module__ not in self.tables.core_modules
        ]

    def kernel_body(self, kernel_class: type) -> list[tuple[type, object, str]]:
        """What a kernel of ``kernel_class`` runs: each method or class
        attribute with its class and name, from the first class in the method
        resolution order that defines it, as Python resolves it, and a base's
        definition too where the one that wins calls ``super()``.
        ``implementation_hash`` is left out, since it computes the key: an
        edit to what it calls that changes what it returns changes the key. So
        is a method only the key computation calls (``key_only_methods``),
        such as the UK full gates' ``_country_resources``, and so are the
        listed refusal-only methods."""

        chosen: dict[str, object | None] = {}
        body = []
        keyed = []
        for cls in self.kernel_classes(kernel_class):
            definitions = module_definitions(cls.__module__, self.scope).definitions
            for statement in definitions[cls.__name__].body:
                for name in statement_names(statement):
                    if name in chosen:
                        previous = chosen[name]
                        if previous is not None and calls_super(previous):
                            chosen[name] = statement
                            (keyed if name == "implementation_hash" else body).append(
                                (cls, statement, name)
                            )
                        continue
                    if (
                        cls.__module__,
                        f"{cls.__name__}.{name}",
                    ) in self.tables.refusal_only:
                        chosen[name] = None
                        continue
                    chosen[name] = statement
                    (keyed if name == "implementation_hash" else body).append(
                        (cls, statement, name)
                    )
        key_only = key_only_methods(body, keyed)
        return [item for item in body if item[2] not in key_only]

    def kernel_walk(
        self,
        kernel_class: type,
        scope: str | None = None,
        *,
        halts: Mapping[tuple[str, str], str] | None = None,
    ) -> KernelWalk:
        """What a kernel's own code can compute through: the rule followed
        from what ``kernel_body`` names, as from a stage's transform, stopping
        also at ``halts`` (``stops`` unless given). Its classes are in
        ``reach``; ``met`` maps each listed stop it meets to what names it
        there (a definition, or ``Class.attribute`` for the kernel's own
        code), and ``core_names`` holds the core modules' definitions it
        names."""

        scope = self.scope if scope is None else scope
        classes = self.kernel_classes(kernel_class)
        seeds = [
            (found, (cls.__module__, f"{cls.__name__}.{name}"))
            for cls, statement, name in self.kernel_body(kernel_class)
            for found in self.named(cls.__module__, statement, scope)
        ]
        walk, met, core_names = self.walk(
            {(cls.__module__, cls.__name__) for cls in classes},
            seeds,
            scope,
            halts=self.tables.stops if halts is None else halts,
        )
        return KernelWalk(walk.reach, walk.stops, walk.core, met, core_names)

    def class_attributes(self, cls: type) -> dict[str, list[tuple[type, object]]]:
        """Each attribute ``cls`` defines, with the class and statement that
        define it, as ``kernel_body`` chooses them by method resolution
        order."""

        body: dict[str, list[tuple[type, object]]] = {}
        for owner, statement, name in self.kernel_body(cls):
            body.setdefault(name, []).append((owner, statement))
        return body

    def engine_methods(self, engine_class: type, walk: Walk) -> frozenset[str]:
        """The attributes of ``engine_class`` that the code ``walk`` reaches
        names: read by attribute, on any receiver, or through ``getattr`` or
        ``hasattr`` with a constant name. It matches by name alone, so it sees
        the engine under any name the code holds it by, and can only
        over-count."""

        found: set[object] = set()
        for module, names in walk.reach.items():
            definitions = module_definitions(module, self.scope).definitions
            for name in names:
                for sub in ast.walk(definitions[name]):
                    if isinstance(sub, ast.Attribute):
                        found.add(sub.attr)
                    elif (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id in {"getattr", "hasattr"}
                        and len(sub.args) > 1
                        and isinstance(sub.args[1], ast.Constant)
                    ):
                        found.add(sub.args[1].value)
        return frozenset(found & set(self.class_attributes(engine_class)))

    def engine_roots(self, engine_class: type) -> frozenset[tuple[str, str]]:
        """The classes an engine of ``engine_class`` comes from, the roots of
        its walk. Their definitions name every method, so the engine methods a
        walk names are read outside them (``engine_reach``)."""

        return frozenset(
            (cls.__module__, cls.__name__) for cls in self.kernel_classes(engine_class)
        )

    def engine_seeds(
        self,
        engine_class: type,
        methods: frozenset[str],
        scope: str | None = None,
    ) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        """What ``methods`` of ``engine_class``, the constructor the builds call
        and every method they call on ``self`` name, each with the method
        naming it: the seeds of the engine walk (``engine_walk``)."""

        scope = self.scope if scope is None else scope
        body = self.class_attributes(engine_class)
        pending, chosen = sorted({"__init__", *methods} & set(body)), set()
        while pending:
            name = pending.pop()
            if name in chosen:
                continue
            chosen.add(name)
            for _, statement in body[name]:
                pending.extend(
                    sub.attr
                    for sub in ast.walk(statement)
                    if isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in {"self", "cls"}
                    and sub.attr in body
                )
        return [
            (found, (owner.__module__, f"{owner.__name__}.{name}"))
            for name in sorted(chosen)
            for owner, statement in body[name]
            for found in self.named(owner.__module__, statement, scope)
        ]

    def engine_walk(
        self,
        engine_class: type,
        methods: frozenset[str],
        scope: str | None = None,
    ) -> Walk:
        """What an engine of ``engine_class`` computes through for code that
        calls ``methods`` on it: the rule's walk, at ``scope`` with the same
        tables, from those methods, the constructor the builds call, and every
        method they call on ``self``, to a fixed point."""

        scope = self.scope if scope is None else scope
        walk, _, _ = self.walk(
            set(self.engine_roots(engine_class)),
            self.engine_seeds(engine_class, methods, scope),
            scope,
        )
        return walk

    def engine_methods_beyond(self, engine_class: type, walk: Walk) -> frozenset[str]:
        """The methods of ``engine_class`` that ``walk`` names outside the
        engine's own classes (``engine_methods``, ``engine_roots``)."""

        roots = self.engine_roots(engine_class)
        return self.engine_methods(
            engine_class,
            Walk(
                {
                    module: frozenset(
                        name for name in names if (module, name) not in roots
                    )
                    for module, names in walk.reach.items()
                },
                walk.stops,
                walk.core,
            ),
        )

    def engine_reach(
        self, engine_class: type, walk: Walk, scope: str | None = None
    ) -> Walk:
        """What an engine of ``engine_class`` computes through for code whose
        walk is ``walk``: the engine walk (``engine_walk``) from the methods
        ``walk`` names, and from those the engine walk's own reach names on
        any receiver outside the engine's classes, to a fixed point. So a
        helper the engine hands itself to, which calls another of its methods
        on that argument, is followed into that method too."""

        methods = self.engine_methods(engine_class, walk)
        while True:
            reached = self.engine_walk(engine_class, methods, scope)
            more = methods | self.engine_methods_beyond(engine_class, reached)
            if more == methods:
                return reached
            methods = more

    def keyed_walk(
        self,
        kernel,
        *,
        halts: Mapping[tuple[str, str], str] | None = None,
        handed_classes: Iterable[tuple[str, str]] = (),
        handed_values: Iterable[tuple[str, str]] = (),
        key: Iterable[str] = (),
    ) -> KeyedWalk:
        """What the kernel rule holds one kernel's key to: the walk from what
        it runs (``kernel_walk``), with what the objects its build hands it
        compute through, and the modules its implementation hash hashes
        (``hashed_modules``) with ``key``, those its key binds beyond them.

        A handed value is a definition the build hands the kernel, such as a
        gate registry, which the walk follows from its definition as from
        what the kernel names. A handed class is the class of an object the
        build hands the kernel whose class its code need not name, such as a
        rules engine: from the methods the walks so far name on any receiver,
        the walk follows them, the constructor the build calls and every
        method they call on ``self``, and the methods its own reach names
        outside the object's classes, to a fixed point, as the engine root
        does (``engine_reach``). Each is walked on its own with the kernel's
        ``halts`` (``stops`` unless given) and joins the kernel's walk, its
        stops, listed stops met and core names included, so it can only add
        to what the kernel walk finds."""

        halts = self.tables.stops if halts is None else halts
        own = self.kernel_walk(type(kernel), halts=halts)
        parts = [(Walk(own.reach, own.stops, own.core), own.met, own.core_names)]
        for value in handed_values:
            module, name = value
            node = module_definitions(module, self.scope).definitions[name]
            parts.append(
                self.walk(
                    {value},
                    [(found, value) for found in self.named(module, node)],
                    halts=halts,
                )
            )
        named = functools.reduce(merged, (walk for walk, _, _ in parts))
        for ref in handed_classes:
            handed = class_named(ref)
            methods = self.engine_methods(handed, named)
            while True:
                part = self.walk(
                    set(self.engine_roots(handed)),
                    self.engine_seeds(handed, methods),
                    halts=halts,
                )
                more = methods | self.engine_methods_beyond(handed, part[0])
                if more == methods:
                    break
                methods = more
            parts.append(part)
        walk = functools.reduce(merged, (walk for walk, _, _ in parts))
        met: dict[tuple[str, str], frozenset] = {}
        for _, part_met, _ in parts:
            for item, namers in part_met.items():
                met[item] = met.get(item, frozenset()) | namers
        return KeyedWalk(
            KernelWalk(
                walk.reach,
                walk.stops,
                walk.core,
                met,
                frozenset().union(*(names for _, _, names in parts)),
            ),
            hashed_modules(kernel) | frozenset(key),
        )

    def refusal_only_violations(self) -> list[tuple[str, str]]:
        """The listed refusal-only checks (``refusal_only``) whose definition
        can return a value or write (``writes_nothing``)."""

        violations = []
        for module, name in self.tables.refusal_only:
            owner, _, method = name.rpartition(".")
            node = module_definitions(module, self.scope).definitions[owner or name]
            if owner:
                (node,) = [
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == method
                ]
            if not writes_nothing(node):
                violations.append((module, name))
        return sorted(violations)

    def core_modules_off_run_graph_path(self) -> list[str]:
        """The listed core modules (``core_modules``) the walk from run_graph,
        which keys, runs and patches every node, does not meet."""

        executor = self.helper_walk(
            ("microcosm.graph.executor", "run_graph"), follow_core=True
        )
        on_path = set(executor.reach) | set(executor.core)
        return sorted(set(self.tables.core_modules) - on_path)
