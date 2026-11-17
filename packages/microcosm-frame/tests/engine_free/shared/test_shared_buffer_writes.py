"""No raw writes into memory that pandas copy-on-write does not protect.

:meth:`Frame.replace_tables` and :meth:`Frame.with_columns` build frames that
share the memory of every unchanged column with their source frame
(:meth:`Frame._adopt`). pandas copy-on-write keeps a write made through pandas
(``.loc``, column assignment, in-place methods) on one frame from reaching
the other, but it does not see a raw write into the shared memory. Checked
with pandas 3.0.3 and NumPy 2.4.6, each write below changes the other frame's
data without an error, so this test reads the scanned source with :mod:`ast`
and refuses it. Each rule's name is what a failure reports.

``makes-array-writable``
    ``setflags(write=...)`` with anything but a false constant, or assigning
    ``flags.writeable``, ``flags["W"]`` or ``flags["WRITEABLE"]``. The arrays
    ``to_numpy()`` and ``.values`` return for NumPy-typed columns are
    read-only views of the shared memory; this makes them writable.
``writes-through-writable-view``
    A write into a writable view of pandas' memory: ``.array`` (of a Series
    or Index), ``.base`` (the writable array behind a read-only view) or
    ``.asi8`` (datetime data). Writes are item assignment, augmented
    assignment, ``__setitem__``, ``fill``, ``sort`` and the other in-place
    ndarray methods, ``fillna(copy=False)``, ``out=``, ``shuffle`` and the
    in-place NumPy functions below. It counts when the written expression
    reaches the view through indexing, ``np.asarray``, ``np.flip`` and the
    other NumPy view functions, ``to_numpy()``, ``astype(copy=False)`` and
    the other view methods, a list, tuple, set or dict holding it, ``or``
    and conditional expressions, or a name the same function binds to any
    of these (by assignment, unpacking, a ``for`` loop or a ``with``
    statement). Workspace classes' own ``self.array`` is exempt.
``pandas-private-attribute``
    ``._ndarray``, ``._mgr``, ``._get_column_array`` or
    ``._iter_column_arrays`` on anything, and ``._values``, ``._data``,
    ``._mask`` or ``._codes`` on anything but ``self`` or ``cls``. These
    reach pandas' internal arrays, which are always writable.
``numpy-copyto``
    Any use of ``numpy.copyto``.
``writes-into-existing-array``
    A write into an array the same function did not create:
    ``out=`` given to a NumPy function, ufunc, ndarray method or NumPy random
    ``Generator`` method (by keyword, or by position for NumPy functions and
    ufuncs), the first argument of ``numpy.<ufunc>.at`` (which writes even
    into a read-only array), the first argument of ``numpy.put``,
    ``numpy.putmask``, ``numpy.place``, ``numpy.put_along_axis``,
    ``numpy.fill_diagonal``, ``numpy.random.shuffle``, any ``.shuffle``
    method and ``numpy.nan_to_num(copy=False)``. NumPy functions are also
    followed through a local name bound to them (``plus = np.add``).
    Created means one of: the result of a ufunc, a ``numpy.random`` function
    or a NumPy function that always allocates
    (``_ALLOCATING_NUMPY_FUNCTIONS``: ``np.zeros``, ``np.where``,
    ``np.concatenate``, ...), when not given an output array; ``np.array``
    without ``copy=``; ``to_numpy(copy=True)``; a subscript of one of these;
    a ``.copy()``, arithmetic, a comparison, ``list(...)`` and the other
    container builtins, or a list, set or dict display (a slice of these is
    new, an item is not, because a shallow copy or a container shares its
    items); or a local name whose every binding in the function, other than
    to ``None``, is one of these.

``setflags``, ``Series.array[i] = value``, ``._ndarray``, ``._mgr``,
``np.copyto`` and ``out=`` are the writes first identified as unprotected.
The other rules, views, functions and methods were added after checking that
they write into shared memory too.

Scope: every Python file under ``packages/*/src/`` (the source of every
workspace package, including microcosm-frame and the microcosm-build code that
builds US and UK frames) and under ``tools/`` (the release and build tools),
except files in ``node_modules`` or hidden directories. Tests and
``test_support/`` are not scanned.

What source reading cannot see, so is not refused here:

- the arrays ``to_numpy()`` and ``.values`` return for extension-typed
  columns (``Int64``, ``Float64``, ``boolean``, ``category``, python-backed
  strings) are writable views of the shared memory;
- a pandas object built with ``copy=False`` over another object's array
  writes into it when it is written through pandas;
- names reached dynamically (``getattr(np, name)``, ``**{"out": array}``,
  ``*arguments``), positional ``out`` given to ndarray methods, writes
  through closures, and items of a container a function did not build.

A finding that cannot write into memory a pandas object shares goes in
``_REVIEWED_FINDINGS`` with the reason.
"""

from __future__ import annotations

import ast
import bisect
import functools
import hashlib
import inspect
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from test_support.paths import paths_for

_REPOSITORY = paths_for("microcosm-frame").repository

#: Reviewed findings that cannot write into memory a pandas object shares,
#: keyed by (repository-relative path, enclosing function, rule, the
#: expression's source, a digest of the enclosing function's source) as a
#: failure prints it, with the reason. The function is ``Class.method`` or
#: ``outer.inner``, or ``<module>`` for module-level code. Any change to the
#: enclosing function changes the digest and needs a new review.
_REVIEWED_FINDINGS: dict[tuple[str, str, str, str, str], str] = {
    (
        "packages/microcosm-build/src/microcosm/build/uk_runtime/frs_relationships.py",
        "derive_frs_relationships",
        "writes-into-existing-array",
        "np.fill_diagonal(matrix, 0)",
        "f439ecc6ea13",
    ): (
        "np.fill_diagonal writes into matrix = grid[rows][:, columns]; indexing "
        "with integer arrays returns a new array, and grid is to_numpy() of a "
        "DataFrame this function computes from the raw FRS columns."
    ),
    (
        "tools/verify_uk_identity_stability.py",
        "_pre_split_household_weights",
        "writes-into-existing-array",
        "np.add.at(weights, lineage.root_positions, weights[lineage.copy_positions])",
        "6cc4521d9bc0",
    ): (
        "np.add.at writes into the array _pre_clone_household_weights returns, "
        "which _pre_residential_household_weights creates with "
        "np.asarray(...).copy() from the household weights."
    ),
}

_MAKES_ARRAY_WRITABLE = "makes-array-writable"
_WRITES_THROUGH_WRITABLE_VIEW = "writes-through-writable-view"
_PANDAS_PRIVATE_ATTRIBUTE = "pandas-private-attribute"
_NUMPY_COPYTO = "numpy-copyto"
_WRITES_INTO_EXISTING_ARRAY = "writes-into-existing-array"

#: Private attributes refused on any object.
_ALWAYS_PRIVATE_ATTRIBUTES = frozenset(
    {"_ndarray", "_mgr", "_get_column_array", "_iter_column_arrays"}
)
#: Private attributes refused unless read from ``self`` or ``cls``, because
#: workspace classes use the same names for their own state.
_FOREIGN_PRIVATE_ATTRIBUTES = frozenset({"_values", "_data", "_mask", "_codes"})
_OWN_OBJECT_NAMES = frozenset({"self", "cls"})

#: Attributes that return a writable view of pandas' memory.
_WRITABLE_VIEW_ATTRIBUTES = frozenset({"array", "base", "asi8"})

#: NumPy functions whose result is always a new array (unless given ``out=``).
#: A ufunc's result and a ``numpy.random`` function's result are new too.
_ALLOCATING_NUMPY_FUNCTIONS = frozenset(
    f"numpy.{name}"
    for name in (
        "where",
        "select",
        "sort",
        "argsort",
        "unique",
        "searchsorted",
        "isin",
        "bincount",
        "cumsum",
        "cumprod",
        "clip",
        "take",
        "round",
        "around",
        "flatnonzero",
        "outer",
        "dot",
        "empty",
        "zeros",
        "ones",
        "full",
        "empty_like",
        "zeros_like",
        "ones_like",
        "full_like",
        "arange",
        "linspace",
        "eye",
        "identity",
        "copy",
        "concatenate",
        "stack",
        "hstack",
        "vstack",
        "column_stack",
        "tile",
        "repeat",
    )
)

#: Builtins whose result is a new container (whose items may be shared).
_CONTAINER_BUILTINS = frozenset(
    {"list", "dict", "set", "frozenset", "tuple", "sorted", "bytearray"}
)

#: Expressions whose value is a new array: arithmetic and comparisons.
_NEW_VALUE_EXPRESSIONS = (ast.BinOp, ast.UnaryOp, ast.Compare)
#: Displays and comprehensions: a new container whose items may be shared.
_CONTAINER_EXPRESSIONS = (
    ast.List,
    ast.ListComp,
    ast.Tuple,
    ast.Set,
    ast.SetComp,
    ast.Dict,
    ast.DictComp,
    ast.GeneratorExp,
)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

#: ndarray methods and NumPy random Generator methods that take an ``out``
#: array.
_METHODS_WITH_OUT = frozenset(
    {
        "all",
        "any",
        "argmax",
        "argmin",
        "choose",
        "clip",
        "compress",
        "cumprod",
        "cumsum",
        "dot",
        "max",
        "mean",
        "min",
        "prod",
        "round",
        "std",
        "sum",
        "take",
        "trace",
        "var",
        "random",
        "standard_normal",
        "standard_exponential",
        "standard_gamma",
        "permuted",
    }
)

#: NumPy functions that write into their first argument.
_IN_PLACE_NUMPY_FUNCTIONS = frozenset(
    {
        "numpy.put",
        "numpy.putmask",
        "numpy.place",
        "numpy.put_along_axis",
        "numpy.fill_diagonal",
        "numpy.random.shuffle",
    }
)

#: ndarray methods that write into the array they are called on.
_IN_PLACE_ARRAY_METHODS = frozenset(
    {
        "__setitem__",
        "fill",
        "sort",
        "put",
        "partition",
        "resize",
        "itemset",
        "setfield",
    }
)

#: NumPy functions that can return a view of an argument.
_VIEW_NUMPY_FUNCTIONS = frozenset(
    f"numpy.{name}"
    for name in (
        "asarray",
        "asanyarray",
        "ascontiguousarray",
        "asfortranarray",
        "ravel",
        "reshape",
        "squeeze",
        "transpose",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "frombuffer",
        "flip",
        "fliplr",
        "flipud",
        "rot90",
        "moveaxis",
        "swapaxes",
        "rollaxis",
        "expand_dims",
        "real",
        "imag",
        "lib.stride_tricks.as_strided",
    )
)

#: Methods and attributes that can return a view of the object they are read
#: from.
_VIEW_METHODS = frozenset(
    {"to_numpy", "view", "reshape", "ravel", "squeeze", "transpose", "__array__"}
)
_VIEW_ATTRIBUTES = frozenset({"T", "real", "imag", "flat"})

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

#: How new the value of an expression is (higher is newer).
_NOT_NEW = 0
#: A new object whose items may be shared: a container, or a ``.copy()``
#: that may be a shallow copy of a container.
_NEW_OBJECT = 1
#: A new array: every item and view of it is new memory.
_NEW_ARRAY = 2


@dataclass(frozen=True)
class Finding:
    """One refused expression: where it is, which rule, and its source."""

    path: str
    line: int
    function: str
    rule: str
    code: str
    function_digest: str

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (self.path, self.function, self.rule, self.code, self.function_digest)

    def describe(self) -> str:
        return f"{self.path}:{self.line} in {self.function}: {self.rule}: {self.code}"


# Bindings ------------------------------------------------------------------

#: One binding of a name: the value expression (``None`` when it cannot be
#: followed, as for an import), and whether the name is bound to exactly that
#: value (``False`` when it is bound to a part of it, as a loop target is
#: bound to an item of its iterable).
_Binding = tuple[ast.expr | None, bool]


class _Scope:
    """One function, class or module, and the names it binds.

    The bindings are collected on first use, because most scopes are never
    asked.
    """

    def __init__(self, qualified_name: str, node: ast.AST) -> None:
        self.qualified_name = qualified_name
        self.node = node
        self._parameters: frozenset[str] = frozenset()
        self._declared_elsewhere: set[str] = set()
        self._bindings: defaultdict[str, list[_Binding]] | None = None

    def child_name(self, name: str) -> str:
        if self.qualified_name == "<module>":
            return name
        return f"{self.qualified_name}.{name}"

    def bindings_of(self, name: str) -> list[_Binding] | None:
        """The bindings of ``name`` here, or ``None`` when it is not local.

        A parameter, a name declared ``global`` or ``nonlocal``, and a name
        this scope never binds are not local.
        """

        self._collect()
        if name in self._parameters or name in self._declared_elsewhere:
            return None
        return self._bindings.get(name) or None

    def is_parameter(self, name: str) -> bool:
        self._collect()
        return name in self._parameters

    def _bind(self, target: ast.expr, value: ast.expr | None, exact: bool) -> None:
        if isinstance(target, ast.Name):
            self._bindings[target.id].append((value, exact))
        elif isinstance(target, ast.Starred):
            self._bind(target.value, value, False)
        elif isinstance(target, (ast.Tuple, ast.List)):
            pairs = (
                exact
                and isinstance(value, (ast.Tuple, ast.List))
                and len(value.elts) == len(target.elts)
                and not any(
                    isinstance(element, ast.Starred)
                    for element in (*target.elts, *value.elts)
                )
            )
            for index, element in enumerate(target.elts):
                if pairs:
                    self._bind(element, value.elts[index], True)
                else:
                    self._bind(element, value, False)

    def _collect(self) -> None:
        if self._bindings is not None:
            return
        self._bindings = defaultdict(list)
        node = self.node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            self._parameters = frozenset(
                argument.arg
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    arguments.vararg,
                    arguments.kwarg,
                )
                if argument is not None
            )
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        for child in _nodes_in_scope(body):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    self._bind(target, child.value, True)
            elif isinstance(child, (ast.AnnAssign, ast.NamedExpr)):
                if child.value is not None:
                    self._bind(child.target, child.value, True)
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                self._bind(child.target, child.iter, False)
            elif isinstance(child, ast.withitem) and child.optional_vars is not None:
                self._bind(child.optional_vars, child.context_expr, False)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                self._bindings[child.name].append((None, False))
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    self._bindings[bound].append((None, False))
            elif isinstance(child, _NESTED_SCOPES) and not isinstance(
                child, ast.Lambda
            ):
                self._bindings[child.name].append((None, False))
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                self._declared_elsewhere.update(child.names)
            elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                self._bindings[child.name].append((None, False))
            elif isinstance(child, ast.MatchMapping) and child.rest:
                self._bindings[child.rest].append((None, False))


class _ComprehensionScope:
    """A comprehension: its loop targets shadow the enclosing scope's names."""

    def __init__(self, enclosing: _Scope | _ComprehensionScope, node: ast.AST):
        self._enclosing = enclosing
        self.qualified_name = enclosing.qualified_name
        self.node = enclosing.node
        self._targets: defaultdict[str, list[_Binding]] = defaultdict(list)
        for generator in node.generators:
            for name in ast.walk(generator.target):
                if isinstance(name, ast.Name):
                    self._targets[name.id].append((generator.iter, False))

    def child_name(self, name: str) -> str:
        return self._enclosing.child_name(name)

    def bindings_of(self, name: str) -> list[_Binding] | None:
        if name in self._targets:
            return self._targets[name]
        return self._enclosing.bindings_of(name)

    def is_parameter(self, name: str) -> bool:
        return name in self._targets or self._enclosing.is_parameter(name)


def _nodes_in_scope(roots: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Yield the nodes of one scope, without entering nested scopes.

    A nested function, class or lambda is yielded (its name is bound here)
    but not entered. Comprehensions are entered except for their loop
    targets, which bind in the comprehension's own scope.
    """

    stack = list(reversed(list(roots)))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _NESTED_SCOPES):
            continue
        children = list(ast.iter_child_nodes(node))
        if isinstance(node, ast.comprehension):
            children = [child for child in children if child is not node.target]
        stack.extend(reversed(children))


def _statements(tree: ast.Module) -> Iterator[ast.AST]:
    """Yield every statement at any depth, without entering expressions."""

    stack: list[ast.AST] = list(reversed(tree.body))
    while stack:
        node = stack.pop()
        yield node
        for field_name in ("body", "orelse", "finalbody", "handlers", "cases"):
            children = getattr(node, field_name, None)
            if isinstance(children, list):
                stack.extend(reversed(children))


class _NumpyNames:
    """The names one module binds to NumPy, to resolve ``np.add.at`` and the like.

    ``derived_names`` are names some statement assigns a NumPy object to
    (``plus = np.add``); the scanner follows them through the bindings of
    the scope that uses them.
    """

    def __init__(self, tree: ast.Module) -> None:
        self._aliases: dict[str, str] = {}
        statements = list(_statements(tree))
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "numpy" or alias.name.startswith("numpy."):
                        if alias.asname:
                            self._aliases[alias.asname] = alias.name
                        else:
                            self._aliases["numpy"] = "numpy"
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
                and (node.module == "numpy" or node.module.startswith("numpy."))
            ):
                for alias in node.names:
                    self._aliases[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        derived: set[str] = set()
        for node in statements:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                if self.resolve(node.value) is None:
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    derived.update(
                        name.id
                        for name in ast.walk(target)
                        if isinstance(name, ast.Name)
                    )
        self.derived_names = frozenset(derived)

    @property
    def bound_names(self) -> frozenset[str]:
        return frozenset(self._aliases)

    def resolve(self, expression: ast.expr) -> str | None:
        """The dotted NumPy name of ``expression`` (``numpy.add.at``), if any."""

        attributes: list[str] = []
        while isinstance(expression, ast.Attribute):
            attributes.append(expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name):
            return None
        root = self._aliases.get(expression.id)
        if root is None:
            return None
        return ".".join([root, *reversed(attributes)])


# Which statements can hold a finding ---------------------------------------

#: A dot before an attribute, allowing spaces and backslash continuations.
_DOT = r"\.(?:\s|\\)*"
#: Text found in every refused expression, except NumPy names, names bound to
#: NumPy objects, and writes through a name bound to a writable view.
_TRIGGER_TEXT = (
    r"setflags|writeable|WRITEABLE|\bflags\b|copyto|shuffle|\bout\s*="
    rf"|{_DOT}_(?:ndarray|mgr|values|data|mask|codes|get_column_array"
    r"|iter_column_arrays)\b"
)
_WRITABLE_VIEW_TEXT = re.compile(rf"{_DOT}(?:array|base|asi8)\b")
#: A module without any of this text cannot hold a finding and is not parsed.
_FILE_TRIGGERS = re.compile(rf"{_TRIGGER_TEXT}|numpy|{_WRITABLE_VIEW_TEXT.pattern}")


class _TriggerLines:
    """The lines of one module that hold text a refused expression contains.

    A statement whose lines hold none of it cannot hold a finding, so the
    scan skips it. This needs line numbers counted as the parser counts
    them, so a module with a lone carriage return gets no ``_TriggerLines``.
    """

    def __init__(self, source: str, names: Iterable[str]) -> None:
        alternatives = [_TRIGGER_TEXT]
        alternatives.extend(rf"\b{re.escape(name)}\b" for name in sorted(names))
        pattern = re.compile("|".join(alternatives))
        newlines = [match.start() for match in re.finditer("\n", source)]
        self._lines = sorted(
            {
                bisect.bisect_left(newlines, match.start()) + 1
                for match in pattern.finditer(source)
            }
        )

    def may_hold_finding(self, statement: ast.stmt) -> bool:
        decorators = getattr(statement, "decorator_list", ())
        first = min([statement.lineno, *(item.lineno for item in decorators)])
        index = bisect.bisect_left(self._lines, first)
        last = statement.end_lineno or statement.lineno
        return index < len(self._lines) and self._lines[index] <= last


# NumPy lookups -------------------------------------------------------------


@functools.cache
def _numpy_object(dotted_name: str) -> object | None:
    found: object = np
    for part in dotted_name.split(".")[1:]:
        found = getattr(found, part, None)
        if found is None:
            return None
    return found


@functools.cache
def _positional_outputs(dotted_name: str) -> slice | None:
    """Which positional arguments of a NumPy function or ufunc are outputs."""

    function = _numpy_object(dotted_name)
    if isinstance(function, np.ufunc):
        return slice(function.nin, None)
    if not callable(function):
        return None
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return None
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    for index, parameter in enumerate(parameters):
        if parameter.name == "out" and parameter.kind in positional:
            return slice(index, index + 1)
    return None


def _is_false_constant(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Constant) and not expression.value


def _is_none(expression: ast.expr | None) -> bool:
    return isinstance(expression, ast.Constant) and expression.value is None


def _keyword(call: ast.Call, name: str) -> ast.keyword | None:
    return next((keyword for keyword in call.keywords if keyword.arg == name), None)


def _is_true_constant(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Constant) and expression.value is True


def _copy_is_off(call: ast.Call) -> bool:
    """Whether a call passes ``copy=`` with anything but ``True``."""

    copy = _keyword(call, "copy")
    return copy is not None and not _is_true_constant(copy.value)


def _is_own_object(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Name) and expression.id in _OWN_OBJECT_NAMES


# The scanner ---------------------------------------------------------------


class _Scanner:
    def __init__(self, source: str, path: str, tree: ast.Module) -> None:
        self._source = source
        self._path = path
        self._tree = tree
        self._numpy = _NumpyNames(tree)
        # A write through a name bound to a writable view holds no trigger
        # text of its own, so a module that reads one is scanned in full.
        lone_carriage_return = "\r" in source and "\r" in source.replace("\r\n", "")
        # Without ``.array``, ``.base`` or ``.asi8`` text, nothing in the
        # module can reach a writable view.
        self._has_writable_views = bool(_WRITABLE_VIEW_TEXT.search(source))
        self._trigger_lines = (
            None
            if lone_carriage_return or self._has_writable_views
            else _TriggerLines(
                source, self._numpy.bound_names | self._numpy.derived_names
            )
        )
        self._module_scope = _Scope("<module>", tree)
        self._scope: _Scope | _ComprehensionScope = self._module_scope
        self._digests: dict[int, str] = {}
        self._checks = {
            ast.Call: self._check_call,
            ast.Assign: self._check_assign,
            ast.AnnAssign: self._check_annotated_assign,
            ast.AugAssign: self._check_augmented_assign,
            ast.Attribute: self._check_attribute,
            ast.ImportFrom: self._check_import_from,
        }
        self.findings: list[Finding] = []

    def scan(self) -> list[Finding]:
        stack: list[tuple[ast.AST, _Scope | _ComprehensionScope]] = [
            (node, self._module_scope) for node in reversed(self._tree.body)
        ]
        while stack:
            node, scope = stack.pop()
            if (
                self._trigger_lines is not None
                and isinstance(node, ast.stmt)
                and not self._trigger_lines.may_hold_finding(node)
            ):
                continue
            self._scope = scope
            check = self._checks.get(type(node))
            if check is not None:
                check(node)
            children = list(ast.iter_child_nodes(node))
            if isinstance(node, _NESTED_SCOPES):
                # Decorators, defaults, annotations and base classes belong
                # to the enclosing scope; only the body is inside.
                name = "<lambda>" if isinstance(node, ast.Lambda) else node.name
                inner = _Scope(scope.child_name(name), node)
                body = [node.body] if isinstance(node, ast.Lambda) else node.body
                inside = {id(statement) for statement in body}
                stack.extend(
                    (child, inner if id(child) in inside else scope)
                    for child in reversed(children)
                )
            elif isinstance(node, _COMPREHENSIONS):
                inner = _ComprehensionScope(scope, node)
                stack.extend((child, inner) for child in reversed(children))
            else:
                stack.extend((child, scope) for child in reversed(children))
        return self.findings

    # Reporting -----------------------------------------------------------

    def _function_digest(self) -> str:
        node = self._scope.node
        digest = self._digests.get(id(node))
        if digest is None:
            if isinstance(node, ast.Module):
                segment = self._source
            else:
                segment = ast.get_source_segment(self._source, node) or ""
            digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()[:12]
            self._digests[id(node)] = digest
        return digest

    def _report(self, node: ast.AST, rule: str) -> None:
        segment = ast.get_source_segment(self._source, node) or ast.unparse(node)
        code = " ".join(segment.split())
        self.findings.append(
            Finding(
                path=self._path,
                line=node.lineno,
                function=self._scope.qualified_name,
                rule=rule,
                code=code if len(code) <= 120 else code[:117] + "...",
                function_digest=self._function_digest(),
            )
        )

    # Following names -------------------------------------------------------

    def _bindings(self, name: str) -> list[_Binding] | None:
        """The bindings of ``name`` in the current scope, if it is local."""

        return self._scope.bindings_of(name)

    def _resolve(self, expression: ast.expr) -> str | None:
        """The dotted NumPy name of ``expression``, following ``plus = np.add``."""

        name = self._numpy.resolve(expression)
        if name is not None:
            return name
        attributes: list[str] = []
        root = expression
        while isinstance(root, ast.Attribute):
            attributes.append(root.attr)
            root = root.value
        if not isinstance(root, ast.Name) or root.id not in self._numpy.derived_names:
            return None
        bindings = self._bindings(root.id)
        if bindings is None and not self._scope.is_parameter(root.id):
            bindings = self._module_scope.bindings_of(root.id)
        if not bindings or not all(exact for _value, exact in bindings):
            return None
        resolved = {self._numpy.resolve(value) for value, _exact in bindings if value}
        if len(resolved) != 1 or None in resolved:
            return None
        return ".".join([resolved.pop(), *reversed(attributes)])

    def _call_newness(self, call: ast.Call) -> int:
        if self._output_arguments(call):
            return _NOT_NEW
        name = self._resolve(call.func)
        if name is not None:
            if name in _ALLOCATING_NUMPY_FUNCTIONS:
                return _NEW_ARRAY
            if isinstance(_numpy_object(name), np.ufunc):
                return _NEW_ARRAY
            if (
                name.startswith("numpy.random.")
                and name not in _IN_PLACE_NUMPY_FUNCTIONS
            ):
                return _NEW_ARRAY
            if name == "numpy.array":
                return _NOT_NEW if _copy_is_off(call) else _NEW_ARRAY
            return _NOT_NEW
        function = call.func
        if isinstance(function, ast.Name) and function.id in _CONTAINER_BUILTINS:
            return _NEW_OBJECT
        if not isinstance(function, ast.Attribute):
            return _NOT_NEW
        if function.attr == "to_numpy":
            copy = _keyword(call, "copy")
            return _NEW_ARRAY if copy and _is_true_constant(copy.value) else _NOT_NEW
        if function.attr == "copy":
            deep = _keyword(call, "deep")
            if deep is not None and not _is_true_constant(deep.value):
                return _NOT_NEW
            if not call.args:
                return _NEW_OBJECT
            first = call.args[0]
            if isinstance(first, ast.Constant) and (
                first.value is True or isinstance(first.value, str)
            ):
                return _NEW_OBJECT
        return _NOT_NEW

    def _newness(
        self, expression: ast.expr, assumed: Mapping[str, int] | None = None
    ) -> int:
        """How new the value of ``expression`` is (``_NOT_NEW`` to ``_NEW_ARRAY``)."""

        assumed = assumed or {}
        if isinstance(expression, ast.Subscript):
            container = self._newness(expression.value, assumed)
            if container == _NEW_ARRAY:
                return _NEW_ARRAY
            if container == _NEW_OBJECT and isinstance(expression.slice, ast.Slice):
                return _NEW_OBJECT
            return _NOT_NEW
        if isinstance(expression, ast.IfExp):
            return min(
                self._newness(expression.body, assumed),
                self._newness(expression.orelse, assumed),
            )
        if isinstance(expression, ast.BoolOp):
            return min(self._newness(value, assumed) for value in expression.values)
        if isinstance(expression, ast.Call):
            return self._call_newness(expression)
        if isinstance(expression, _NEW_VALUE_EXPRESSIONS):
            return _NEW_ARRAY
        if isinstance(expression, _CONTAINER_EXPRESSIONS):
            return _NEW_OBJECT
        if not isinstance(expression, ast.Name):
            return _NOT_NEW
        name = expression.id
        if name in assumed:
            return assumed[name]
        # A binding to None is a placeholder that holds no array.
        bindings = [
            (value, exact)
            for value, exact in self._bindings(name) or []
            if not (exact and _is_none(value))
        ]
        if not bindings or any(value is None or not exact for value, exact in bindings):
            return _NOT_NEW
        # A binding may refer back to the name (values = values[:n]), so
        # start from the newest level and lower it until it holds.
        level = _NEW_ARRAY
        while True:
            inner = {**assumed, name: level}
            result = min(self._newness(value, inner) for value, _exact in bindings)
            if result >= level:
                return level
            level = result

    def _reaches_writable_view(
        self, expression: ast.expr, following: frozenset[str] = frozenset()
    ) -> bool:
        """Whether ``expression`` can be, or hold, a writable view of pandas' memory."""

        if not self._has_writable_views:
            return False
        if isinstance(expression, ast.Attribute):
            if expression.attr in _WRITABLE_VIEW_ATTRIBUTES:
                return not _is_own_object(expression.value)
            if expression.attr in _VIEW_ATTRIBUTES:
                return self._reaches_writable_view(expression.value, following)
            return False
        if isinstance(expression, (ast.Subscript, ast.Starred)):
            return self._reaches_writable_view(expression.value, following)
        if isinstance(expression, ast.IfExp):
            parts: list[ast.expr] = [expression.body, expression.orelse]
        elif isinstance(expression, ast.BoolOp):
            parts = list(expression.values)
        elif isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
            parts = list(expression.elts)
        elif isinstance(expression, ast.Dict):
            parts = list(expression.values)
        elif isinstance(expression, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            parts = [expression.elt]
        elif isinstance(expression, ast.DictComp):
            parts = [expression.value]
        elif isinstance(expression, ast.Call):
            parts = self._view_call_sources(expression)
        elif isinstance(expression, ast.Name) and expression.id not in following:
            parts = [
                value for value, _exact in self._bindings(expression.id) or [] if value
            ]
            following = following | {expression.id}
        else:
            return False
        return any(self._reaches_writable_view(part, following) for part in parts)

    def _view_call_sources(self, call: ast.Call) -> list[ast.expr]:
        """The expressions a call's result can be a view of."""

        name = self._resolve(call.func)
        if name in _VIEW_NUMPY_FUNCTIONS or (
            name == "numpy.array" and _copy_is_off(call)
        ):
            return [*call.args, *(keyword.value for keyword in call.keywords)]
        function = call.func
        if isinstance(function, ast.Attribute):
            if function.attr in _VIEW_METHODS:
                copy = _keyword(call, "copy")
                if copy is not None and _is_true_constant(copy.value):
                    return []
                return [function.value]
            if function.attr == "astype" and _copy_is_off(call):
                return [function.value]
        return []

    def _check_written_array(self, node: ast.AST, written: ast.expr) -> None:
        """Report a write into ``written`` unless it is new in this function."""

        if self._reaches_writable_view(written):
            self._report(node, _WRITES_THROUGH_WRITABLE_VIEW)
        elif self._newness(written) == _NOT_NEW:
            self._report(node, _WRITES_INTO_EXISTING_ARRAY)

    # Rules -----------------------------------------------------------------

    def _output_arguments(self, call: ast.Call) -> list[ast.expr]:
        """The arrays a call writes into: ``out=``, positional outputs, ``ufunc.at``."""

        written: list[ast.expr] = []
        name = self._resolve(call.func)
        function = call.func
        method = function.attr if isinstance(function, ast.Attribute) else None
        output = _keyword(call, "out")
        # ``out=`` means an output array only to NumPy and to array and random
        # Generator methods; workspace functions use it for output paths.
        if output is not None and (name is not None or method in _METHODS_WITH_OUT):
            if isinstance(output.value, (ast.Tuple, ast.List)):
                written.extend(
                    element for element in output.value.elts if not _is_none(element)
                )
            elif not _is_false_constant(output.value):
                written.append(output.value)
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            return written
        if name is None:
            if method == "shuffle":
                written.extend(call.args[:1])
            return written
        if name in _IN_PLACE_NUMPY_FUNCTIONS:
            return [*written, *call.args[:1]]
        if name == "numpy.nan_to_num" and _copy_is_off(call):
            return [*written, *call.args[:1]]
        if name.endswith(".at") and isinstance(_numpy_object(name[:-3]), np.ufunc):
            return [*written, *call.args[:1]]
        outputs = _positional_outputs(name)
        return written if outputs is None else [*written, *call.args[outputs]]

    def _writes_into_receiver(self, call: ast.Call, method: str) -> bool:
        if method in _IN_PLACE_ARRAY_METHODS:
            return True
        if method == "fillna":
            return _copy_is_off(call)
        if method == "byteswap":
            in_place = _keyword(call, "inplace")
            flag = in_place.value if in_place is not None else None
            if flag is None and call.args:
                flag = call.args[0]
            return flag is not None and not _is_false_constant(flag)
        return False

    def _check_call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr == "setflags":
                write = _keyword(node, "write")
                flag = write.value if write is not None else None
                if flag is None and node.args:
                    flag = node.args[0]
                if flag is not None and not _is_false_constant(flag):
                    self._report(node, _MAKES_ARRAY_WRITABLE)
            elif self._writes_into_receiver(node, function.attr):
                if self._reaches_writable_view(function.value):
                    self._report(node, _WRITES_THROUGH_WRITABLE_VIEW)
        for written in self._output_arguments(node):
            self._check_written_array(node, written)

    def _check_assigned_target(
        self, node: ast.AST, target: ast.expr, value: ast.expr | None
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._check_assigned_target(node, element, None)
        elif isinstance(target, ast.Starred):
            self._check_assigned_target(node, target.value, None)
        elif isinstance(target, ast.Subscript):
            flags = target.value
            key = target.slice
            if (
                isinstance(flags, ast.Attribute)
                and flags.attr == "flags"
                and isinstance(key, ast.Constant)
                and key.value in ("W", "WRITEABLE")
            ):
                if value is None or not _is_false_constant(value):
                    self._report(node, _MAKES_ARRAY_WRITABLE)
            elif self._reaches_writable_view(target.value):
                self._report(node, _WRITES_THROUGH_WRITABLE_VIEW)
        elif (
            isinstance(target, ast.Attribute)
            and target.attr == "writeable"
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "flags"
        ):
            if value is None or not _is_false_constant(value):
                self._report(node, _MAKES_ARRAY_WRITABLE)

    def _check_assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_assigned_target(node, target, node.value)

    def _check_annotated_assign(self, node: ast.AnnAssign) -> None:
        self._check_assigned_target(node, node.target, node.value)

    def _check_augmented_assign(self, node: ast.AugAssign) -> None:
        # ``values += 1`` writes into the array ``values`` names.
        if isinstance(node.target, ast.Name) and self._reaches_writable_view(
            node.target
        ):
            self._report(node, _WRITES_THROUGH_WRITABLE_VIEW)
        else:
            self._check_assigned_target(node, node.target, None)

    def _check_attribute(self, node: ast.Attribute) -> None:
        if node.attr in _ALWAYS_PRIVATE_ATTRIBUTES:
            self._report(node, _PANDAS_PRIVATE_ATTRIBUTE)
        elif node.attr in _FOREIGN_PRIVATE_ATTRIBUTES and not _is_own_object(
            node.value
        ):
            self._report(node, _PANDAS_PRIVATE_ATTRIBUTE)
        elif node.attr == "copyto":
            self._report(node, _NUMPY_COPYTO)

    def _check_import_from(self, node: ast.ImportFrom) -> None:
        if node.module is not None and (
            node.module == "numpy" or node.module.startswith("numpy.")
        ):
            if any(alias.name == "copyto" for alias in node.names):
                self._report(node, _NUMPY_COPYTO)


def scan_source(source: str, path: str) -> list[Finding]:
    """Every refused expression in one module's source."""

    if not _FILE_TRIGGERS.search(source):
        return []
    tree = ast.parse(source, filename=path)
    return _Scanner(source, path, tree).scan()


def scanned_files(repository: Path) -> list[Path]:
    """The Python files the scan reads: ``packages/*/src/`` and ``tools/``."""

    roots = [*sorted(repository.glob("packages/*/src")), repository / "tools"]
    files: list[Path] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            parts = path.relative_to(root).parts[:-1]
            if any(part == "node_modules" or part.startswith(".") for part in parts):
                continue
            files.append(path)
    return files


@dataclass(frozen=True)
class _WorkspaceScan:
    paths: frozenset[str]
    findings: tuple[Finding, ...]


@pytest.fixture(scope="module")
def workspace_scan() -> _WorkspaceScan:
    paths: set[str] = set()
    findings: list[Finding] = []
    for path in scanned_files(_REPOSITORY):
        relative = path.relative_to(_REPOSITORY).as_posix()
        paths.add(relative)
        # Read the bytes as they are, so line endings reach the parser.
        source = path.read_bytes().decode("utf-8")
        findings.extend(scan_source(source, relative))
    return _WorkspaceScan(frozenset(paths), tuple(findings))


def test_the_scanned_source_has_no_unprotected_buffer_writes(
    workspace_scan: _WorkspaceScan,
) -> None:
    unreviewed = [
        finding
        for finding in workspace_scan.findings
        if finding.key not in _REVIEWED_FINDINGS
    ]

    assert not unreviewed, (
        "These expressions can write into memory that frames share under "
        "pandas copy-on-write (see this module's docstring for each rule). "
        "Write through pandas instead (.loc, column assignment), or copy the "
        "array first (np.array(values, copy=True)). A finding that cannot "
        "write into memory a pandas object shares goes in "
        "_REVIEWED_FINDINGS, under the key shown, with the reason.\n"
        + "\n".join(
            f"{finding.describe()}\n    key: {finding.key!r}" for finding in unreviewed
        )
    )


def test_every_reviewed_finding_still_occurs(workspace_scan: _WorkspaceScan) -> None:
    present = {finding.key for finding in workspace_scan.findings}

    assert not sorted(set(_REVIEWED_FINDINGS) - present), (
        "These reviewed findings no longer occur as reviewed. The expression "
        "or its enclosing function changed, so review it again and update "
        "its key, or remove the entry."
    )


def test_the_scan_covers_every_package_and_the_us_frame_building_code(
    workspace_scan: _WorkspaceScan,
) -> None:
    packages_with_source = {
        directory.name
        for directory in (_REPOSITORY / "packages").iterdir()
        if (directory / "src").is_dir()
    }
    scanned_packages = {
        path.split("/")[1]
        for path in workspace_scan.paths
        if path.startswith("packages/")
    }

    assert scanned_packages == packages_with_source
    assert {
        "packages/microcosm-frame/src/microcosm/frame/bundle.py",
        "packages/microcosm-build/src/microcosm/build/frame_checkpoint.py",
        "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py",
        "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py",
        "tools/build_us_fiscal_refresh_release.py",
        "tools/build_us_multispine_pool.py",
    } <= workspace_scan.paths


def test_the_scan_skips_node_modules_and_hidden_directories(tmp_path: Path) -> None:
    for relative in (
        "packages/microcosm-example/src/microcosm/example/module.py",
        "packages/microcosm-example/tests/test_module.py",
        "tools/tool.py",
        "tools/route_a/helper.py",
        "tools/orrery-contract/node_modules/node-gyp/gyp/pylib/gyp/input.py",
        "tools/.cache/generated.py",
    ):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("", encoding="utf-8")

    scanned = [
        path.relative_to(tmp_path).as_posix() for path in scanned_files(tmp_path)
    ]

    assert scanned == [
        "packages/microcosm-example/src/microcosm/example/module.py",
        "tools/route_a/helper.py",
        "tools/tool.py",
    ]


_NUMPY = "import numpy as np\n"


def _in_function(*body_lines: str, parameters: str = "series") -> str:
    """A module importing NumPy with one function holding ``body_lines``."""

    body = "".join(f"    {line}\n" for line in body_lines)
    return f"{_NUMPY}def change({parameters}):\n{body}"


#: Snippets the scan must refuse, with the one rule each must report.
_REFUSED_SNIPPETS = {
    # makes-array-writable
    "setflags_write_true": (
        "values = series.to_numpy()\nvalues.setflags(write=True)\nvalues[0] = 1.0\n",
        _MAKES_ARRAY_WRITABLE,
    ),
    "setflags_positional": ("values.setflags(True)\n", _MAKES_ARRAY_WRITABLE),
    "setflags_variable": ("values.setflags(write=flag)\n", _MAKES_ARRAY_WRITABLE),
    "flags_writeable_attribute": (
        "values.flags.writeable = True\n",
        _MAKES_ARRAY_WRITABLE,
    ),
    "flags_writeable_key": (
        'values.flags["WRITEABLE"] = True\n',
        _MAKES_ARRAY_WRITABLE,
    ),
    "flags_w_key": ('values.flags["W"] = 1\n', _MAKES_ARRAY_WRITABLE),
    "flags_key_parenthesized": (
        _in_function('(values.flags)["W"] = True', parameters="values"),
        _MAKES_ARRAY_WRITABLE,
    ),
    # writes-through-writable-view
    "series_array_item": (
        'frame.person["income"].array[0] = 1.0\n',
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_slice_augmented": (
        "series.array[mask] += 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_annotated": (
        "series.array[0]: float = 1.0\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_unpacked": (
        "series.array[0], other = 1.0, 2.0\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_setitem_call": (
        "series.array.__setitem__(0, 1.0)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "index_array_item": (
        "table.index.array[0] = 7\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_backslash_continuation": (
        "series.\\\narray[0] = 1.0\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_asarray": (
        _in_function("values = np.asarray(series.array)", "values[0] = 0.0"),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_asarray_keyword": (
        _NUMPY + "np.asarray(a=series.array)[0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_flip": (
        _NUMPY + "np.flip(np.asarray(series.array))[0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_to_numpy": (
        "series.array.to_numpy()[0] = 1.0\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_astype_without_copy": (
        'series.array.astype("float64", copy=False)[0] = 1.0\n',
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_alias_chain": (
        _in_function("backing = series.array", "values = backing", "values[0] = 0.0"),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_conditional": (
        _in_function(
            "values = series.array if flag else other",
            "values[0] = 1",
            parameters="series, other, flag",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_either_operand": (
        _in_function(
            "values = override or series.array",
            "values[0] = 1",
            parameters="override, series",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_unpacking": (
        _in_function(
            "left, right = first.array, second.array",
            "left[0] = 1",
            parameters="first, second",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_loop_target": (
        _in_function(
            "for values in (frame[name].array for name in names):",
            "    values[0] = 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_in_dict_comprehension": (
        _in_function(
            "arrays = {name: frame[name].array for name in names}",
            'arrays["income"][0] = 1.0',
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_in_list_written_by_out": (
        _in_function(
            "arrays = [np.asarray(series.array)]",
            "np.add(arrays[0], 1, out=arrays[0])",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_augmented_name": (
        _in_function("values = np.asarray(series.array)", "values += 1"),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_to_numpy_augmented_name": (
        _in_function("values = series.array.to_numpy()", "values *= 2"),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_fill": (
        _NUMPY + "np.asarray(series.array).fill(0)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_sort": (
        _NUMPY + "np.asarray(series.array).sort()\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_put": (
        _NUMPY + "np.asarray(series.array).put([0], 1)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_partition": (
        _NUMPY + "np.asarray(series.array).partition(1)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_resize": (
        _NUMPY + "np.asarray(series.array).resize(3)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_itemset": (
        _NUMPY + "np.asarray(series.array).itemset(0, 1)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_setfield": (
        _NUMPY + "np.asarray(series.array).setfield(1, np.int64)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_byteswap_in_place": (
        _NUMPY + "np.asarray(series.array).byteswap(inplace=True)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_fillna_without_copy": (
        "series.array.fillna(0.0, copy=False)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_transposed_view": (
        "series.array.T[0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_reshaped_view": (
        "series.array.reshape(-1)[0] = 1.0\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_as_out": (
        _NUMPY + "np.add(series.array, 1, out=series.array)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_in_place_function": (
        _NUMPY + "np.putmask(np.asarray(series.array), mask, 0)\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_shuffled_by_generator": (
        _in_function("rng.shuffle(np.asarray(series.array))", parameters="rng, series"),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_no_copy_array": (
        _NUMPY + "np.array(series.array, copy=False)[0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "read_only_view_base": (
        "series.to_numpy().base[0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "datetime_asi8": ("series.array.asi8[0] = 0\n", _WRITES_THROUGH_WRITABLE_VIEW),
    "index_asi8": ("frame.index.asi8[0] = 0\n", _WRITES_THROUGH_WRITABLE_VIEW),
    # pandas-private-attribute
    "ndarray_attribute": (
        "values = series.array._ndarray\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "manager_attribute": (
        "blocks = frame._mgr.blocks\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "manager_attribute_on_self": (
        "class Table:\n    def blocks(self):\n        return self._mgr.blocks\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "series_values_attribute": (
        "raw = series._values\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "masked_data": ("raw = series.array._data\n", _PANDAS_PRIVATE_ATTRIBUTE),
    "masked_mask": ("series.array._mask[:] = False\n", _PANDAS_PRIVATE_ATTRIBUTE),
    "categorical_codes": (
        "codes = series.cat.codes.array._codes\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "column_array_method": (
        "column = frame._get_column_array(0)\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "column_arrays_iterator": (
        "for column in frame._iter_column_arrays():\n    pass\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    # numpy-copyto
    "copyto": (_NUMPY + "np.copyto(target, source)\n", _NUMPY_COPYTO),
    "copyto_through_numpy": (
        "import numpy\nnumpy.copyto(target, source)\n",
        _NUMPY_COPYTO,
    ),
    "copyto_imported": (
        "from numpy import copyto\ncopyto(target, source)\n",
        _NUMPY_COPYTO,
    ),
    "copyto_imported_as_alias": (
        "from numpy import copyto as overwrite\n",
        _NUMPY_COPYTO,
    ),
    "copyto_passed_as_function": (
        "apply_each(np.copyto, pairs)\n",
        _NUMPY_COPYTO,
    ),
    # writes-into-existing-array
    "out_parameter": (
        _in_function("np.add(values, 1, out=values)", parameters="values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_attribute": (
        _NUMPY + "np.multiply(scale, 2, out=self.buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_to_numpy": (
        _NUMPY + "np.add(scale, 1, out=series.to_numpy())\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_method_call": (
        "def clip_in_place(values):\n    values.clip(0, 1, out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_array_method_on_parameter": (
        "def running(values):\n    values.cumsum(out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_generator_method": (
        _in_function("rng.random(out=values)", parameters="rng, values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_generator_permuted": (
        _in_function("rng.permuted(values, out=values)", parameters="rng, values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_rebound_to_existing": (
        _in_function(
            "buffer = np.zeros(3)",
            "buffer = series.to_numpy()",
            "np.add(buffer, 1, out=buffer)",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_slice_of_existing": (
        _in_function(
            "buffer = series.to_numpy()",
            "buffer = buffer[1:]",
            "np.add(buffer, 1, out=buffer)",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_slice_of_parameter": (
        _in_function(
            "values = values[1:]",
            "np.add(values, 1, out=values)",
            parameters="values",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_item_of_new_list_rebound_to_itself": (
        _in_function(
            "values = [series.to_numpy()]",
            "values = values[0]",
            "np.add(values, 1, out=values)",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_tuple": (
        _in_function("np.divmod(7, 2, out=(left, right))", parameters="left, right"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_positional_ufunc": (
        _in_function("np.add(values, 1, values)", parameters="values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_positional_function": (
        _in_function("np.cumsum(values, 0, None, totals)", parameters="values, totals"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_from_imported_ufunc": (
        "from numpy import add\ndef add_one(values):\n    add(values, 1, values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_through_local_function_alias": (
        _NUMPY
        + "plus = np.add\n"
        + "def add_one(values):\n"
        + "    plus(values, 1, values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_keyword_through_function_alias": (
        _in_function(
            "plus = np.add", "plus(values, 1, out=values)", parameters="values"
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at": (
        _in_function("np.add.at(values, [0], 1)", parameters="values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_imported": (
        "from numpy import add\ndef count(values):\n    add.at(values, [0], 1)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_to_numpy": (
        _NUMPY + "np.maximum.at(series.to_numpy(), positions, values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_item_of_new_list": (
        _in_function(
            "columns = [frame[name].to_numpy() for name in names]",
            "np.add.at(columns[0], [0], 1)",
            parameters="frame, names",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_item_of_shallow_dict_copy": (
        _in_function(
            "arrays = arrays_by_name.copy()",
            'np.multiply(arrays["x"], 2, out=arrays["x"])',
            parameters="arrays_by_name",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "numpy_put": (
        _in_function("np.put(values, [0], 1)", parameters="values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "numpy_place": (
        _in_function("np.place(values, mask, 0)", parameters="values, mask"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "numpy_put_along_axis": (
        _in_function(
            "np.put_along_axis(values, positions, 0, axis=0)",
            parameters="values, positions",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "numpy_nan_to_num_without_copy": (
        _in_function("np.nan_to_num(values, copy=False)", parameters="values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "numpy_fill_diagonal": (
        _in_function("np.fill_diagonal(matrix, 0)", parameters="matrix"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "numpy_random_shuffle": (
        _in_function("np.random.shuffle(values)", parameters="values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_module_array_from_function": (
        _NUMPY + "buffer = np.zeros(3)\ndef add_one():\n    np.add(1, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_global_declared": (
        _in_function(
            "global buffer",
            "buffer = np.zeros(3)",
            "np.add(1, 1, out=buffer)",
            parameters="",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_loop_target": (
        _in_function(
            "for buffer in arrays:",
            "    np.add(buffer, 1, out=buffer)",
            parameters="arrays",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_comprehension_target": (
        _in_function("[np.add(a, 1, out=a) for a in arrays]", parameters="arrays"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_comprehension_target_shadowing_a_new_array": (
        _in_function(
            "a = np.zeros(3)",
            "return [np.add(a, 1, out=a) for a in arrays]",
            parameters="arrays",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_lambda_parameter": (
        _NUMPY + "add_one = lambda values: np.add(values, 1, out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_in_a_decorator": (
        _NUMPY
        + "@register(np.add(1, 1, out=buffer))\n"
        + "def change():\n"
        + "    return 1\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_in_a_decorator_over_a_function_binding_the_name": (
        _NUMPY
        + "@register(np.add(1, 1, out=buffer))\n"
        + "def change():\n"
        + "    buffer = np.zeros(3)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_shallow_copy": (
        _in_function(
            "shallow = table.copy(deep=False)",
            "np.add(1, 1, out=shallow)",
            parameters="table",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_positional_shallow_copy": (
        _in_function(
            "shallow = table.copy(False)",
            "np.add(1, 1, out=shallow)",
            parameters="table",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_array_without_copy": (
        _in_function(
            "view = np.array(values, copy=None)",
            "np.add(view, 1, out=view)",
            parameters="values",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_concatenate_into_existing": (
        _in_function(
            "joined = np.concatenate(parts, out=existing)",
            "np.add(joined, 1, out=joined)",
            parameters="parts, existing",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_one_branch_existing": (
        _in_function(
            "buffer = np.zeros(3) if fresh else values",
            "np.add(buffer, 1, out=buffer)",
            parameters="values, fresh",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_either_operand": (
        _in_function(
            "buffer = given or np.zeros(3)",
            "np.add(buffer, 1, out=buffer)",
            parameters="given",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_ufunc_result_with_output": (
        _in_function(
            "total = np.add(values, 1, existing)",
            "np.add(total, 1, out=total)",
            parameters="values, existing",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_with_target": (
        _in_function(
            "with opener() as buffer:",
            "    np.add(buffer, 1, out=buffer)",
            parameters="opener",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_except_target": (
        _in_function(
            "try:",
            "    pass",
            "except Exception as buffer:",
            "    np.add(1, 1, out=buffer)",
            parameters="",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_imported_name": (
        _in_function("import buffer", "np.add(1, 1, out=buffer)", parameters=""),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_match_capture": (
        _in_function(
            "match value:",
            "    case [buffer]:",
            "        np.add(1, 1, out=buffer)",
            parameters="value",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_after_lone_carriage_returns": (
        "import numpy as np\rdef change(values):\r    np.add(values, 1, out=values)\r",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
}


@pytest.mark.parametrize(
    ("source", "rule"), list(_REFUSED_SNIPPETS.values()), ids=list(_REFUSED_SNIPPETS)
)
def test_the_scanner_refuses_each_unprotected_write(source: str, rule: str) -> None:
    findings = scan_source(source, "snippet.py")

    assert {finding.rule for finding in findings} == {rule}, findings


#: Snippets the scan must accept: reads, read-only flags, and writes into
#: arrays the same function created.
_ACCEPTED_SNIPPETS = {
    "setflags_read_only": "values.setflags(write=False)\nvalues.setflags(write=0)\n",
    "setflags_alignment_only": "values.setflags(align=True)\n",
    "flags_read_only": (
        'values.flags.writeable = False\nvalues.flags["WRITEABLE"] = False\n'
    ),
    "out_new_array_inline": (
        _NUMPY
        + "share = np.divide(part, total, out=np.zeros_like(part), where=total > 0)\n"
    ),
    "out_new_array_slice": _in_function(
        "totals = np.empty(len(weights) + 1)",
        "totals[0] = 0.0",
        "np.cumsum(weights, out=totals[1:])",
        parameters="weights",
    ),
    "out_new_array_row": _in_function(
        "grid = np.zeros((2, 3))", "np.add(1, 1, out=grid[0])", parameters=""
    ),
    "out_deep_copy": _in_function(
        "total = first.copy()",
        "np.add(total, later, out=total)",
        parameters="first, later",
    ),
    "out_deep_copy_in_order": _in_function(
        "total = first.copy('C')",
        "np.add(total, later, out=total)",
        parameters="first, later",
    ),
    "out_copied_to_numpy": _in_function(
        "values = series.to_numpy(dtype=float, copy=True)",
        "np.add(values, 1, out=values)",
    ),
    "out_same_array_as_input": _in_function(
        "draws = np.empty(count)",
        "np.minimum(draws, 1.0, out=draws)",
        "return draws",
        parameters="count",
    ),
    "out_chained_and_conditional_names": _in_function(
        "first = second = np.zeros(3)",
        "alias = second",
        "chosen = np.ones(3) if wide else alias",
        "chosen += 1",
        "np.add(values, 1, out=chosen)",
        "np.add(values, 1, out=first)",
        parameters="values, wide",
    ),
    "out_array_with_copy": _in_function(
        "owned = np.array(values)",
        "copied = np.array(values, copy=True)",
        "np.add(owned, 1, out=owned)",
        "np.add(copied, 1, out=copied)",
        parameters="values",
    ),
    "out_none": _NUMPY + "np.add(left, right, out=None)\n",
    "out_none_pair": _NUMPY + "np.divmod(left, 2, out=(None, None))\n",
    "ufunc_at_new_array": _in_function(
        "counts = np.zeros(size, dtype=np.int64)",
        "np.add.at(counts, positions, 1)",
        parameters="positions, size",
    ),
    "in_place_function_new_array": _in_function(
        "matrix = np.ones((size, size))",
        "np.fill_diagonal(matrix, 0)",
        parameters="size",
    ),
    "ufunc_without_out": _NUMPY + "total = np.add(left, right)\n",
    "array_function_not_attribute": (
        _NUMPY
        + "import pandas as pd\n"
        + "values = np.array(series)\n"
        + "values[0] = 1\n"
        + "codes = pd.array([1, 2])\n"
        + "codes[0] = 3\n"
    ),
    "series_array_reads": (
        "for value in series.array:\n"
        + "    total += value\n"
        + "index = pd.Index(series.array, name='id')\n"
        + "first = series.array[0]\n"
    ),
    "series_array_deep_copy_written": (
        "values = series.array.to_numpy(copy=True)\nvalues[0] = 1\n"
    ),
    "series_array_byteswap_copy": (
        _NUMPY + "swapped = np.asarray(series.array).byteswap()\n"
    ),
    "own_array_attribute": (
        "class Grid:\n    def set(self, row, value):\n        self.array[row] = value\n"
    ),
    "own_private_state": (
        "class Catalog:\n"
        + "    def add(self, key, value):\n"
        + "        self._values[key] = value\n"
        + "        self._data.append(value)\n"
        + "    @classmethod\n"
        + "    def empty(cls):\n"
        + "        return cls._mask\n"
    ),
    "subscript_write_into_new_array": _in_function(
        "marks = np.zeros(size)", "marks[0] = 1", parameters="size"
    ),
    "out_keyword_of_workspace_functions": (
        "launch(pool_manifest=manifest, out=args.out)\n"
        + "report.write(rows, out=destination)\n"
    ),
    "none_placeholder_then_copy": _in_function(
        "total: np.ndarray | None = None",
        "for current in slices:",
        "    if total is None:",
        "        total = current.copy()",
        "    else:",
        "        np.add(total, current, out=total)",
        parameters="slices",
    ),
    "list_display_shuffled": _in_function(
        "values: list[float] = []",
        "values.extend(draws)",
        "values = values[:3]",
        "np.random.shuffle(values)",
        parameters="draws",
    ),
    "list_builtin_shuffled": _in_function(
        "order = list(range(size))", "np.random.shuffle(order)", parameters="size"
    ),
    "random_and_rounded_results": _in_function(
        "order = np.random.permutation(size)",
        "np.add(order, 1, out=order)",
        "rounded = np.round(values)",
        "np.add(rounded, 1, out=rounded)",
        parameters="size, values",
    ),
    "generator_method_into_new_array": _in_function(
        "draws = np.empty(3)", "rng.random(out=draws)", parameters="rng"
    ),
    "where_ufunc_arithmetic_and_comparison_results": _in_function(
        "out = np.zeros(3)",
        "np.add.at(out, positions, values)",
        "out = np.where(mask, out, 0.0)",
        "np.multiply(out, 2, out=out)",
        "scaled = np.exp(values)",
        "np.multiply(scaled, 2, out=scaled)",
        "total = left + right",
        "np.add(total, 1, out=total)",
        "positive = left > 0",
        "np.logical_not(positive, out=positive)",
        parameters="positions, values, mask, left, right",
    ),
}


@pytest.mark.parametrize(
    "source", list(_ACCEPTED_SNIPPETS.values()), ids=list(_ACCEPTED_SNIPPETS)
)
def test_the_scanner_accepts_reads_and_writes_into_new_arrays(source: str) -> None:
    assert scan_source(source, "snippet.py") == []


def test_a_finding_names_its_file_line_function_and_digest() -> None:
    method = (
        "    def add(self, values):\n"
        + "        def inner(target):\n"
        + "            np.add(target, 1, out=target)\n"
        + "        inner(values)\n"
    )
    source = _NUMPY + "class Aggregator:\n" + method

    (finding,) = scan_source(source, "pkg/module.py")

    assert finding.describe() == (
        "pkg/module.py:5 in Aggregator.add.inner: writes-into-existing-array: "
        "np.add(target, 1, out=target)"
    )
    inner_source = "def inner(target):\n            np.add(target, 1, out=target)"
    assert (
        finding.function_digest
        == hashlib.sha256(inner_source.encode("utf-8")).hexdigest()[:12]
    )


def test_a_changed_function_changes_the_finding_key() -> None:
    original = _in_function("np.fill_diagonal(matrix, 0)", parameters="matrix")
    changed = _in_function(
        "matrix = matrix.T", "np.fill_diagonal(matrix, 0)", parameters="matrix"
    )

    (before,) = scan_source(original, "pkg/module.py")
    (after,) = scan_source(changed, "pkg/module.py")

    assert before.code == after.code
    assert before.key != after.key
