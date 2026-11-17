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
``writes-through-series-array``
    A write into ``.array`` (of a Series or Index): item assignment,
    ``__setitem__``, ``fill``, ``sort``, ``put`` and the other in-place
    ndarray methods, ``out=``, or the in-place NumPy functions below. It
    counts when the written expression reaches ``.array`` through indexing,
    ``np.asarray`` and the other NumPy view functions, ``to_numpy()`` and
    the other view methods, or a name the same function binds to such an
    expression. ``ser.array`` is writable for float, integer and boolean
    columns, and so are ``np.asarray(ser.array)`` and
    ``ser.array.to_numpy()``.
``pandas-private-attribute``
    ``._ndarray`` or ``._mgr`` on anything, and ``._values``, ``._data``,
    ``._mask`` or ``._codes`` on anything but ``self`` or ``cls``. These
    reach pandas' internal arrays, which are always writable.
``numpy-copyto``
    Any use of ``numpy.copyto``.
``writes-into-existing-array``
    ``out=`` given to a NumPy function, ufunc or ndarray method (by keyword,
    or by position for NumPy functions and ufuncs), the first argument of
    ``numpy.<ufunc>.at`` (which writes even into a read-only array), and the
    first argument of ``numpy.put``, ``numpy.putmask``, ``numpy.place``,
    ``numpy.fill_diagonal`` and ``numpy.random.shuffle``, unless the array
    was created in the same function. Created means one of: the result of a
    ufunc or of a NumPy function that always allocates
    (``_ALLOCATING_NUMPY_FUNCTIONS``: ``np.zeros``, ``np.where``,
    ``np.concatenate``, ...), when not given an output array; ``np.array``
    without ``copy=``; a deep ``.copy()``; arithmetic, a comparison, or a
    list, set or dict display; a subscript of one of these; or a local name
    whose every binding in the function, other than to ``None``, is one of
    these.

``setflags``, ``Series.array[i] = value``, ``._ndarray``, ``._mgr``,
``np.copyto`` and ``out=`` are the writes first identified as unprotected.
``ufunc.at``, the other in-place NumPy functions and methods, the other
writes through ``.array``, and the other private attributes were added after
checking that they write into shared memory too.

Scope: every Python file under ``packages/*/src/`` (the source of every
workspace package, including microcosm-frame and the microcosm-build code that
builds US and UK frames) and under ``tools/`` (the release and build tools),
except files in ``node_modules`` or hidden directories. Tests and
``test_support/`` are not scanned.

What source reading cannot see: the arrays ``to_numpy()`` and ``.values``
return for nullable extension types (``Int64``, ``boolean``, ``Float64``)
are writable views of the shared memory, and a pandas object built with
``copy=False`` over another object's array writes into it when it is written
through pandas. Neither is refused here.

A finding that cannot write into memory a pandas object shares goes in
``_REVIEWED_FINDINGS`` with the reason.
"""

from __future__ import annotations

import ast
import bisect
import functools
import inspect
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from test_support.paths import paths_for

_REPOSITORY = paths_for("microcosm-frame").repository

#: Reviewed findings that cannot write into memory a pandas object shares,
#: keyed by (repository-relative path, enclosing function, rule, the
#: expression's source as a failure reports it), with the reason. The
#: function is ``Class.method`` or ``outer.inner``, or ``<module>`` for
#: module-level code. A changed expression needs a new review.
_REVIEWED_FINDINGS: dict[tuple[str, str, str, str], str] = {
    (
        "packages/microcosm-build/src/microcosm/build/uk_runtime/frs_relationships.py",
        "derive_frs_relationships",
        "writes-into-existing-array",
        "np.fill_diagonal(matrix, 0)",
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
    ): (
        "np.add.at writes into the array _pre_clone_household_weights returns, "
        "which _pre_residential_household_weights creates with "
        "np.asarray(...).copy() from the household weights."
    ),
}

_MAKES_ARRAY_WRITABLE = "makes-array-writable"
_WRITES_THROUGH_SERIES_ARRAY = "writes-through-series-array"
_PANDAS_PRIVATE_ATTRIBUTE = "pandas-private-attribute"
_NUMPY_COPYTO = "numpy-copyto"
_WRITES_INTO_EXISTING_ARRAY = "writes-into-existing-array"

#: Private attributes refused on any object.
_ALWAYS_PRIVATE_ATTRIBUTES = frozenset({"_ndarray", "_mgr"})
#: Private attributes refused unless read from ``self`` or ``cls``, because
#: workspace classes use the same names for their own state.
_FOREIGN_PRIVATE_ATTRIBUTES = frozenset({"_values", "_data", "_mask", "_codes"})
_OWN_OBJECT_NAMES = frozenset({"self", "cls"})

#: NumPy functions whose result is always a new array (unless given ``out=``).
#: A ufunc's result is new too.
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
        "diff",
        "cumsum",
        "cumprod",
        "clip",
        "take",
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

#: Expressions whose value is always a new object: arithmetic, comparisons,
#: and list, set and dict displays and comprehensions.
_NEW_OBJECT_EXPRESSIONS = (
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.List,
    ast.ListComp,
    ast.Set,
    ast.SetComp,
    ast.Dict,
    ast.DictComp,
)

#: ndarray methods that take an ``out`` array.
_ARRAY_METHODS_WITH_OUT = frozenset(
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
    }
)

#: NumPy functions that write into their first argument.
_IN_PLACE_NUMPY_FUNCTIONS = frozenset(
    {
        "numpy.put",
        "numpy.putmask",
        "numpy.place",
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
        "byteswap",
    }
)

#: NumPy functions that can return a view of their first argument.
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
    )
)

#: Methods and attributes that can return a view of the object they are read
#: from.
_VIEW_METHODS = frozenset(
    {"to_numpy", "view", "reshape", "ravel", "squeeze", "transpose", "__array__"}
)
_VIEW_ATTRIBUTES = frozenset({"T", "real", "imag", "flat"})

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


@dataclass(frozen=True)
class Finding:
    """One refused expression: where it is, which rule, and its source."""

    path: str
    line: int
    function: str
    rule: str
    code: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.path, self.function, self.rule, self.code)

    def describe(self) -> str:
        return f"{self.path}:{self.line} in {self.function}: {self.rule}: {self.code}"


class _Scope:
    """One function, class or module, and the names it binds.

    The bindings are collected on first use, because most scopes are never
    asked. A name maps to the value of each assignment to it; ``None`` stands
    for a binding whose value cannot be followed (a loop target, tuple
    unpacking, an import, a ``with`` target, ...).
    """

    def __init__(self, qualified_name: str, node: ast.AST) -> None:
        self.qualified_name = qualified_name
        self._node = node
        self._parameters: frozenset[str] = frozenset()
        self._declared_elsewhere: set[str] = set()
        self._bindings: defaultdict[str, list[ast.expr | None]] | None = None

    def child_name(self, name: str) -> str:
        if self.qualified_name == "<module>":
            return name
        return f"{self.qualified_name}.{name}"

    def bindings_of(self, name: str) -> list[ast.expr | None] | None:
        """The values ``name`` is bound to here, or ``None`` when it is not local.

        A parameter, a name declared ``global`` or ``nonlocal``, and a name
        this scope never binds are not local.
        """

        if self._bindings is None:
            self._bindings = defaultdict(list)
            self._collect_bindings()
        if name in self._parameters or name in self._declared_elsewhere:
            return None
        return self._bindings.get(name) or None

    def _bind(self, target: ast.expr, value: ast.expr | None) -> None:
        if isinstance(target, ast.Name):
            self._bindings[target.id].append(value)
        elif isinstance(target, ast.Starred):
            self._bind(target.value, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind(element, None)

    def _collect_bindings(self) -> None:
        node = self._node
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
                    self._bind(target, child.value)
            elif isinstance(child, (ast.AnnAssign, ast.NamedExpr)):
                if child.value is not None:
                    self._bind(child.target, child.value)
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                self._bind(child.target, None)
            elif isinstance(child, ast.withitem) and child.optional_vars is not None:
                self._bind(child.optional_vars, None)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                self._bindings[child.name].append(None)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    self._bindings[bound].append(None)
            elif isinstance(child, _NESTED_SCOPES) and not isinstance(
                child, ast.Lambda
            ):
                self._bindings[child.name].append(None)
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                self._declared_elsewhere.update(child.names)
            elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                self._bindings[child.name].append(None)
            elif isinstance(child, ast.MatchMapping) and child.rest:
                self._bindings[child.rest].append(None)


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
    """The names one module binds to NumPy, to resolve ``np.add.at`` and the like."""

    def __init__(self, tree: ast.Module) -> None:
        self._aliases: dict[str, str] = {}
        for node in _statements(tree):
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


#: Text found in every refused expression, except NumPy names and writes
#: through a name bound to ``.array``.
_TRIGGER_TEXT = (
    r"setflags|writeable|WRITEABLE|flags\s*\[|copyto"
    r"|\.\s*_(?:ndarray|mgr|values|data|mask|codes)\b|\bout\s*="
)
#: A module without any of this text cannot hold a finding and is not parsed.
_FILE_TRIGGERS = re.compile(_TRIGGER_TEXT + r"|numpy|\.\s*array\b")
_ARRAY_ATTRIBUTE = re.compile(r"\.\s*array\b")


class _TriggerLines:
    """The lines of one module that hold text a refused expression contains.

    A statement whose lines hold none of it cannot hold a finding, so the
    scan skips it. This needs line numbers counted as the parser counts
    them, so a module with a lone carriage return gets no ``_TriggerLines``.
    """

    def __init__(self, source: str, numpy_names: Iterable[str]) -> None:
        alternatives = [_TRIGGER_TEXT]
        alternatives.extend(rf"\b{re.escape(name)}\b" for name in sorted(numpy_names))
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


def _keyword(call: ast.Call, name: str) -> ast.keyword | None:
    return next((keyword for keyword in call.keywords if keyword.arg == name), None)


def _is_true_constant(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Constant) and expression.value is True


class _Scanner:
    def __init__(self, source: str, path: str, tree: ast.Module) -> None:
        self._source = source
        self._path = path
        self._tree = tree
        self._numpy = _NumpyNames(tree)
        # A write through a name bound to ``.array`` holds no trigger text of
        # its own, so a module that reads ``.array`` is scanned in full.
        lone_carriage_return = "\r" in source and "\r" in source.replace("\r\n", "")
        self._trigger_lines = (
            None
            if lone_carriage_return or _ARRAY_ATTRIBUTE.search(source)
            else _TriggerLines(source, self._numpy.bound_names)
        )
        self._scope = _Scope("<module>", tree)
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
        stack = [(node, self._scope) for node in reversed(self._tree.body)]
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
            if isinstance(node, _NESTED_SCOPES):
                name = "<lambda>" if isinstance(node, ast.Lambda) else node.name
                scope = _Scope(scope.child_name(name), node)
            children = list(ast.iter_child_nodes(node))
            stack.extend((child, scope) for child in reversed(children))
        return self.findings

    # Reporting -----------------------------------------------------------

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
            )
        )

    # Following arrays ------------------------------------------------------

    def _bindings(self, name: str) -> list[ast.expr | None] | None:
        """The values ``name`` is bound to in the current scope, if it is local."""

        return self._scope.bindings_of(name)

    def _allocates_new_array(self, call: ast.Call) -> bool:
        if self._output_arguments(call):
            return False
        name = self._numpy.resolve(call.func)
        if name in _ALLOCATING_NUMPY_FUNCTIONS:
            return True
        if name is not None and isinstance(_numpy_object(name), np.ufunc):
            return True
        if name == "numpy.array":
            copy = _keyword(call, "copy")
            return copy is None or _is_true_constant(copy.value)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "copy":
            deep = _keyword(call, "deep")
            if deep is not None and not _is_true_constant(deep.value):
                return False
            if not call.args:
                return True
            first = call.args[0]
            return isinstance(first, ast.Constant) and (
                first.value is True or isinstance(first.value, str)
            )
        return False

    def _is_new_local_array(
        self, expression: ast.expr, following: frozenset[str] = frozenset()
    ) -> bool:
        if isinstance(expression, ast.Subscript):
            return self._is_new_local_array(expression.value, following)
        if isinstance(expression, ast.IfExp):
            return self._is_new_local_array(
                expression.body, following
            ) and self._is_new_local_array(expression.orelse, following)
        if isinstance(expression, ast.Call):
            return self._allocates_new_array(expression)
        if isinstance(expression, _NEW_OBJECT_EXPRESSIONS):
            return True
        if isinstance(expression, ast.Name) and expression.id in following:
            # A binding that refers back to the name being followed (such as
            # values = values[:n]) is new when the name's other bindings are.
            return True
        if isinstance(expression, ast.Name):
            # A binding to None is a placeholder that holds no array.
            values = [
                value
                for value in self._bindings(expression.id) or []
                if not (isinstance(value, ast.Constant) and value.value is None)
            ]
            return bool(values) and all(
                value is not None
                and self._is_new_local_array(value, following | {expression.id})
                for value in values
            )
        return False

    def _reaches_series_array(
        self, expression: ast.expr, following: frozenset[str] = frozenset()
    ) -> bool:
        if isinstance(expression, ast.Attribute):
            if expression.attr == "array":
                return True
            if expression.attr in _VIEW_ATTRIBUTES:
                return self._reaches_series_array(expression.value, following)
            return False
        if isinstance(expression, ast.Subscript):
            return self._reaches_series_array(expression.value, following)
        if isinstance(expression, ast.IfExp):
            return self._reaches_series_array(
                expression.body, following
            ) or self._reaches_series_array(expression.orelse, following)
        if isinstance(expression, ast.Call):
            name = self._numpy.resolve(expression.func)
            copy = _keyword(expression, "copy")
            makes_view = name in _VIEW_NUMPY_FUNCTIONS or (
                name == "numpy.array"
                and copy is not None
                and not _is_true_constant(copy.value)
            )
            if makes_view:
                return bool(expression.args) and self._reaches_series_array(
                    expression.args[0], following
                )
            function = expression.func
            if isinstance(function, ast.Attribute) and function.attr in _VIEW_METHODS:
                if copy is not None and _is_true_constant(copy.value):
                    return False
                return self._reaches_series_array(function.value, following)
            return False
        if isinstance(expression, ast.Name) and expression.id not in following:
            values = self._bindings(expression.id) or []
            return any(
                value is not None
                and self._reaches_series_array(value, following | {expression.id})
                for value in values
            )
        return False

    def _check_written_array(self, node: ast.AST, written: ast.expr) -> None:
        """Report a write into ``written`` unless it is a new local array."""

        if self._reaches_series_array(written):
            self._report(node, _WRITES_THROUGH_SERIES_ARRAY)
        elif not self._is_new_local_array(written):
            self._report(node, _WRITES_INTO_EXISTING_ARRAY)

    # Rules -----------------------------------------------------------------

    def _output_arguments(self, call: ast.Call) -> list[ast.expr]:
        """The arrays a call writes into: ``out=``, positional outputs, ``ufunc.at``."""

        written: list[ast.expr] = []
        name = self._numpy.resolve(call.func)
        output = _keyword(call, "out")
        # ``out=`` means an output array only to NumPy and to ndarray methods;
        # workspace functions use it for output paths.
        array_method = (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in _ARRAY_METHODS_WITH_OUT
        )
        if output is not None and (name is not None or array_method):
            if isinstance(output.value, (ast.Tuple, ast.List)):
                written.extend(output.value.elts)
            elif not _is_false_constant(output.value):
                written.append(output.value)
        if name is None or any(isinstance(arg, ast.Starred) for arg in call.args):
            return written
        if name in _IN_PLACE_NUMPY_FUNCTIONS:
            return [*written, *call.args[:1]]
        if name.endswith(".at") and isinstance(_numpy_object(name[:-3]), np.ufunc):
            return [*written, *call.args[:1]]
        outputs = _positional_outputs(name)
        return written if outputs is None else [*written, *call.args[outputs]]

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
            elif function.attr in _IN_PLACE_ARRAY_METHODS:
                if self._reaches_series_array(function.value):
                    self._report(node, _WRITES_THROUGH_SERIES_ARRAY)
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
            elif self._reaches_series_array(target.value):
                self._report(node, _WRITES_THROUGH_SERIES_ARRAY)
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
        self._check_assigned_target(node, node.target, None)

    def _check_attribute(self, node: ast.Attribute) -> None:
        if node.attr in _ALWAYS_PRIVATE_ATTRIBUTES:
            self._report(node, _PANDAS_PRIVATE_ATTRIBUTE)
        elif node.attr in _FOREIGN_PRIVATE_ATTRIBUTES and not (
            isinstance(node.value, ast.Name) and node.value.id in _OWN_OBJECT_NAMES
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
        findings.extend(scan_source(path.read_text(encoding="utf-8"), relative))
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
        "_REVIEWED_FINDINGS with the reason.\n"
        + "\n".join(finding.describe() for finding in unreviewed)
    )


def test_every_reviewed_finding_still_occurs(workspace_scan: _WorkspaceScan) -> None:
    present = {finding.key for finding in workspace_scan.findings}

    assert not sorted(set(_REVIEWED_FINDINGS) - present)


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

#: Snippets the scan must refuse, with the one rule each must report.
_REFUSED_SNIPPETS = {
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
    "series_array_item": (
        'frame.person["income"].array[0] = 1.0\n',
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_slice_augmented": (
        "series.array[mask] += 1\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_annotated": (
        "series.array[0]: float = 1.0\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_unpacked": (
        "series.array[0], other = 1.0, 2.0\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_setitem_call": (
        "series.array.__setitem__(0, 1.0)\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "index_array_item": ("table.index.array[0] = 7\n", _WRITES_THROUGH_SERIES_ARRAY),
    "series_array_through_asarray": (
        _NUMPY
        + "def zero_first(series):\n"
        + "    values = np.asarray(series.array)\n"
        + "    values[0] = 0.0\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_through_to_numpy": (
        "series.array.to_numpy()[0] = 1.0\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_through_alias_chain": (
        "def zero_first(series):\n"
        + "    backing = series.array\n"
        + "    values = backing\n"
        + "    values[0] = 0.0\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_fill": (
        _NUMPY + "np.asarray(series.array).fill(0)\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_sort": (
        _NUMPY + "np.asarray(series.array).sort()\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_reshaped_view": (
        "series.array.reshape(-1)[0] = 1.0\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_as_out": (
        _NUMPY + "np.add(series.array, 1, out=series.array)\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_in_place_function": (
        _NUMPY + "np.putmask(np.asarray(series.array), mask, 0)\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_no_copy_array": (
        _NUMPY + "np.array(series.array, copy=False)[0] = 1\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "ndarray_attribute": (
        "values = series.array._ndarray\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "manager_attribute": ("blocks = frame._mgr.blocks\n", _PANDAS_PRIVATE_ATTRIBUTE),
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
    "out_parameter": (
        _NUMPY + "def add_one(values):\n    np.add(values, 1, out=values)\n",
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
    "out_rebound_to_existing": (
        _NUMPY
        + "def add_one(series):\n"
        + "    buffer = np.zeros(3)\n"
        + "    buffer = series.to_numpy()\n"
        + "    np.add(buffer, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_slice_of_existing": (
        _NUMPY
        + "def add_one(series):\n"
        + "    buffer = series.to_numpy()\n"
        + "    buffer = buffer[1:]\n"
        + "    np.add(buffer, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_slice_of_parameter": (
        _NUMPY
        + "def add_one(values):\n"
        + "    values = values[1:]\n"
        + "    np.add(values, 1, out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_tuple": (
        _NUMPY + "def split(left, right):\n    np.divmod(7, 2, out=(left, right))\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_positional_ufunc": (
        _NUMPY + "def add_one(values):\n    np.add(values, 1, values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_positional_function": (
        _NUMPY
        + "def running(values, totals):\n    np.cumsum(values, 0, None, totals)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_from_imported_ufunc": (
        "from numpy import add\ndef add_one(values):\n    add(values, 1, values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at": (
        _NUMPY + "def count(values):\n    np.add.at(values, [0], 1)\n",
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
    "numpy_put": (
        _NUMPY + "def set_first(values):\n    np.put(values, [0], 1)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "numpy_fill_diagonal": (
        _NUMPY + "def clear(matrix):\n    np.fill_diagonal(matrix, 0)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "numpy_random_shuffle": (
        _NUMPY + "def mix(values):\n    np.random.shuffle(values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_module_array_from_function": (
        _NUMPY + "buffer = np.zeros(3)\ndef add_one():\n    np.add(1, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_global_declared": (
        _NUMPY
        + "def add_one():\n"
        + "    global buffer\n"
        + "    buffer = np.zeros(3)\n"
        + "    np.add(1, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_loop_target": (
        _NUMPY
        + "def add_one(arrays):\n"
        + "    for buffer in arrays:\n"
        + "        np.add(buffer, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_comprehension_target": (
        _NUMPY + "def add_one(arrays):\n    [np.add(a, 1, out=a) for a in arrays]\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_lambda_parameter": (
        _NUMPY + "add_one = lambda values: np.add(values, 1, out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_shallow_copy": (
        _NUMPY
        + "def add_one(table):\n"
        + "    shallow = table.copy(deep=False)\n"
        + "    np.add(1, 1, out=shallow)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_positional_shallow_copy": (
        _NUMPY
        + "def add_one(table):\n"
        + "    shallow = table.copy(False)\n"
        + "    np.add(1, 1, out=shallow)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_array_without_copy": (
        _NUMPY
        + "def add_one(values):\n"
        + "    view = np.array(values, copy=None)\n"
        + "    np.add(view, 1, out=view)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_concatenate_into_existing": (
        _NUMPY
        + "def join(parts, existing):\n"
        + "    joined = np.concatenate(parts, out=existing)\n"
        + "    np.add(joined, 1, out=joined)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_one_branch_existing": (
        _NUMPY
        + "def add_one(values, fresh):\n"
        + "    buffer = np.zeros(3) if fresh else values\n"
        + "    np.add(buffer, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_with_target": (
        _NUMPY
        + "def add_one(opener):\n"
        + "    with opener() as buffer:\n"
        + "        np.add(buffer, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_array_method_on_parameter": (
        "def running(values):\n    values.cumsum(out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_either_operand": (
        _NUMPY
        + "def add_one(given):\n"
        + "    buffer = given or np.zeros(3)\n"
        + "    np.add(buffer, 1, out=buffer)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_ufunc_result_with_output": (
        _NUMPY
        + "def add_one(values, existing):\n"
        + "    total = np.add(values, 1, existing)\n"
        + "    np.add(total, 1, out=total)\n",
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
    "out_new_array_slice": (
        _NUMPY
        + "def running(weights):\n"
        + "    totals = np.empty(len(weights) + 1)\n"
        + "    totals[0] = 0.0\n"
        + "    np.cumsum(weights, out=totals[1:])\n"
    ),
    "out_deep_copy": (
        _NUMPY
        + "def accumulate(first, later):\n"
        + "    total = first.copy()\n"
        + "    np.add(total, later, out=total)\n"
    ),
    "out_deep_copy_in_order": (
        _NUMPY
        + "def accumulate(first, later):\n"
        + "    total = first.copy('C')\n"
        + "    np.add(total, later, out=total)\n"
    ),
    "out_same_array_as_input": (
        _NUMPY
        + "def capped(count):\n"
        + "    draws = np.empty(count)\n"
        + "    np.minimum(draws, 1.0, out=draws)\n"
        + "    return draws\n"
    ),
    "out_chained_and_conditional_names": (
        _NUMPY
        + "def add_one(values, wide):\n"
        + "    first = second = np.zeros(3)\n"
        + "    alias = second\n"
        + "    chosen = np.ones(3) if wide else alias\n"
        + "    chosen += 1\n"
        + "    np.add(values, 1, out=chosen)\n"
        + "    np.add(values, 1, out=first)\n"
    ),
    "out_array_with_copy": (
        _NUMPY
        + "def add_one(values):\n"
        + "    owned = np.array(values)\n"
        + "    copied = np.array(values, copy=True)\n"
        + "    np.add(owned, 1, out=owned)\n"
        + "    np.add(copied, 1, out=copied)\n"
    ),
    "out_none": _NUMPY + "np.add(left, right, out=None)\n",
    "ufunc_at_new_array": (
        _NUMPY
        + "def count(positions, size):\n"
        + "    counts = np.zeros(size, dtype=np.int64)\n"
        + "    np.add.at(counts, positions, 1)\n"
    ),
    "in_place_function_new_array": (
        _NUMPY
        + "def clear(size):\n"
        + "    matrix = np.ones((size, size))\n"
        + "    np.fill_diagonal(matrix, 0)\n"
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
    "own_private_state": (
        "class Catalog:\n"
        + "    def add(self, key, value):\n"
        + "        self._values[key] = value\n"
        + "        self._data.append(value)\n"
        + "    @classmethod\n"
        + "    def empty(cls):\n"
        + "        return cls._mask\n"
    ),
    "subscript_write_into_new_array": (
        _NUMPY + "def flags(size):\n    marks = np.zeros(size)\n    marks[0] = 1\n"
    ),
    "out_keyword_of_workspace_functions": (
        "launch(pool_manifest=manifest, out=args.out)\n"
        + "report.write(rows, out=destination)\n"
    ),
    "none_placeholder_then_copy": (
        _NUMPY
        + "def accumulate(slices):\n"
        + "    total: np.ndarray | None = None\n"
        + "    for current in slices:\n"
        + "        if total is None:\n"
        + "            total = current.copy()\n"
        + "        else:\n"
        + "            np.add(total, current, out=total)\n"
    ),
    "list_display_shuffled": (
        _NUMPY
        + "def sizes(draws):\n"
        + "    values: list[float] = []\n"
        + "    values.extend(draws)\n"
        + "    values = values[:3]\n"
        + "    np.random.shuffle(values)\n"
    ),
    "where_ufunc_arithmetic_and_comparison_results": (
        _NUMPY
        + "def totals(positions, values, mask, left, right):\n"
        + "    out = np.zeros(3)\n"
        + "    np.add.at(out, positions, values)\n"
        + "    out = np.where(mask, out, 0.0)\n"
        + "    np.multiply(out, 2, out=out)\n"
        + "    scaled = np.exp(values)\n"
        + "    np.multiply(scaled, 2, out=scaled)\n"
        + "    total = left + right\n"
        + "    np.add(total, 1, out=total)\n"
        + "    positive = left > 0\n"
        + "    np.logical_not(positive, out=positive)\n"
    ),
}


@pytest.mark.parametrize(
    "source", list(_ACCEPTED_SNIPPETS.values()), ids=list(_ACCEPTED_SNIPPETS)
)
def test_the_scanner_accepts_reads_and_writes_into_new_arrays(source: str) -> None:
    assert scan_source(source, "snippet.py") == []


def test_a_finding_names_its_file_line_and_function() -> None:
    source = (
        _NUMPY
        + "class Aggregator:\n"
        + "    def add(self, values):\n"
        + "        def inner(target):\n"
        + "            np.add(target, 1, out=target)\n"
        + "        inner(values)\n"
    )

    (finding,) = scan_source(source, "pkg/module.py")

    assert finding.describe() == (
        "pkg/module.py:5 in Aggregator.add.inner: writes-into-existing-array: "
        "np.add(target, 1, out=target)"
    )
