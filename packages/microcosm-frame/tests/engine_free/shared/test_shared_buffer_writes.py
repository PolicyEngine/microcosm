"""A syntactic check for raw writes that pandas copy-on-write does not protect.

:meth:`Frame.replace_tables`, :meth:`Frame.with_columns`,
:meth:`Frame.with_weights`, :meth:`Frame.place` and :meth:`Frame.select` build
frames that share the memory of unchanged columns with their source frame
(:meth:`Frame.adopt`). pandas copy-on-write keeps a write made through pandas
(``.loc``, column assignment, in-place methods) on one frame from reaching
the other, but it does not see a raw write into the shared memory. Checked
with pandas 3.0.3 and NumPy 2.4.6, each form below writes into the shared
memory without an error, so this test reads the scanned source with
:mod:`ast` and refuses it.

This is a check on how code is written, not an analysis of where arrays come
from. It looks at one expression at a time and, for the last rule only,
follows a name to its own assignments in the same function. Each rule's name
is what a failure reports.

``makes-array-writable``
    ``x.setflags(write=...)`` with anything but a false constant (a
    positional first argument counts too), and assigning anything but a
    false constant to ``x.flags.writeable``, ``x.flags["W"]`` or
    ``x.flags["WRITEABLE"]``. The arrays ``to_numpy()`` and ``.values``
    return for NumPy-typed columns are read-only views of the shared memory;
    this makes them writable.
``writes-through-series-array``
    Item assignment, plain or augmented (``x.array[i] = v``,
    ``x.array[mask] += v``), into ``x.array``, and ``x.array.__setitem__(...)``,
    where ``x`` is not ``self`` or ``cls``. ``Series.array`` is the column's
    own array.
``pandas-private-attribute``
    Any use of ``._ndarray`` or ``._mgr``, and of ``._values``, ``._data``,
    ``._mask`` or ``._codes`` on anything but ``self`` or ``cls`` (workspace
    classes use those names for their own state). These reach pandas'
    internal arrays.
``numpy-copyto``
    Any use of ``copyto`` as an attribute (``np.copyto``) or imported from
    NumPy.
``writes-into-existing-array``
    ``out=`` given by keyword to a NumPy function or ufunc (named through
    ``import numpy as np`` or ``from numpy import ...``) or to an ndarray
    method that takes it (``_METHODS_WITH_OUT``), and the first argument of
    ``np.<ufunc>.at``, which writes even into a read-only array, unless that
    array is created in the function. Created means:

    - a call, not itself given ``out=``, of a NumPy function that allocates
      (``_ALLOCATING_NUMPY_FUNCTIONS``: ``np.zeros``, ``np.empty_like``,
      ``np.where``, ...), of a ufunc, of ``np.array`` without ``copy=`` or
      with ``copy=True``, of ``.to_numpy(copy=True)``, or of a random
      ``Generator`` draw (``_RANDOM_DRAW_METHODS``: ``rng.random(n)``, ...);
    - arithmetic or a comparison (``a - b``, ``a > 0``);
    - ``.copy()`` with no arguments (except of ``.index`` or ``.columns``, or
      a subscript of them: ``Index.copy()`` is shallow, and a slice of an
      index shares its memory), or ``+``, ``*`` or ``|`` (which also
      join lists and dicts), or a slice of these, but not an item of them,
      because a copy or join of a dict or list shares its items;
    - a subscript of the others;
    - a local name, or a subscript of one, whose every assignment in the
      function is one of these or ``None``, with at least one that is not
      ``None``. A parameter, a name declared ``global`` or ``nonlocal``, and
      a name bound any other way in the function (a loop, comprehension or
      ``with`` target, unpacking, an import) are not created.

    ``.astype()`` does not count as created: a pandas object's conversion
    to its own type shares its memory.

``setflags``, ``Series.array[i] = value``, ``._ndarray``, ``._mgr``,
``np.copyto`` and ``out=`` are the writes first identified as unprotected.
``ufunc.at`` and the other private attributes were added after probes showed
that they write into shared memory too.

Scope: every Python file under ``packages/*/src/`` (the source of every
workspace package, including microcosm-frame and the microcosm-build code that
builds US and UK frames) and under ``tools/`` (the release and build tools),
except files in ``node_modules`` or hidden directories. Tests and
``test_support/`` are not scanned.

What this check does not see, so does not refuse:

- arrays reached in any other way: through another name
  (``values = x.array; values[0] = 1``), ``np.asarray(x.array)``,
  ``x.array.to_numpy()``, ``.base`` or ``.asi8``, containers, closures, and
  other functions' arguments and return values;
- positional ``out`` arguments, SciPy ufuncs, and the other functions and
  methods that write in place (``np.put``, ``np.putmask``,
  ``np.fill_diagonal``, ``np.random.shuffle``, ``arr.sort()``, ``arr.fill()``);
- what ``.values`` and ``to_numpy()`` return for extension-typed columns:
  ``.values`` returns the column's own extension array (for example for the
  nullable integer, float and boolean types, ``category`` and python-backed
  strings), and ``to_numpy()`` a writable view of the column's data (for
  example for the nullable types without missing values, python-backed
  strings, and sparse columns without gaps);
- pandas objects built with ``copy=False`` over another object's array, and
  ``np.asarray(index)``, which is writable although the index is shared even
  between deep copies.

Catching these needs a check at run time, such as hashing each shared column
when a frame is built and again before its outputs are written.

A finding that cannot write into memory a pandas object shares goes in
``_REVIEWED_FINDINGS`` with the reason.
"""

from __future__ import annotations

import ast
import bisect
import hashlib
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
#: expression's source, a digest of the enclosing function's source) as a
#: failure prints it, with the reason. The function is ``Class.method`` or
#: ``outer.inner``, or ``<module>`` for module-level code. Any change to the
#: enclosing function changes the digest and needs a new review; a reason
#: that rests on other functions names them, and changing them needs one too.
_REVIEWED_FINDINGS: dict[tuple[str, str, str, str, str], str] = {
    (
        "tools/verify_uk_identity_stability.py",
        "_pre_split_household_weights",
        "writes-into-existing-array",
        "np.add.at(weights, lineage.root_positions, weights[lineage.copy_positions])",
        "88cad28e8a13",
    ): (
        "np.add.at writes into the array _pre_clone_household_weights returns, "
        "_pre_clone_fold's first array, which _pre_residential_household_weights "
        "creates with np.asarray(...).copy() from the household weights. Review "
        "again if any of those functions changes."
    ),
    (
        "tools/verify_uk_identity_stability.py",
        "_pre_high_value_household_weights",
        "writes-into-existing-array",
        "np.add.at(weights, lineage.root_positions, weights[lineage.copy_positions])",
        "c6d258cb6a6b",
    ): (
        "np.add.at writes into the array _pre_split_household_weights returns, "
        "the same copy of the household weights its own entry describes "
        "(microcosm#931). Review again if it or the functions it rests on "
        "change."
    ),
    (
        "packages/microcosm-frame/src/microcosm/frame/shared_columns.py",
        "_column_buffers",
        "pandas-private-attribute",
        "values._data",
        "64ca7d79dc01",
    ): (
        "Read only, so the shared-column check can hash a nullable column's "
        "values. pandas has no public access to them when values are missing."
    ),
    (
        "packages/microcosm-frame/src/microcosm/frame/shared_columns.py",
        "_column_buffers",
        "pandas-private-attribute",
        "values._mask",
        "64ca7d79dc01",
    ): (
        "Read only, so the shared-column check can hash a nullable column's "
        "missing-value mask. pandas exposes only a copy of it."
    ),
    (
        "packages/microcosm-graph/src/microcosm/graph/executor.py",
        "_read_only_masked",
        "pandas-private-attribute",
        "values._data",
        "f08675645fa1",
    ): (
        "Read only: the graph executor passes a nullable column's values to "
        "_read_only_view, which builds a read-only array whose base is a "
        "_ReadOnlyMemory, so a kernel's item write into it, into its .base or "
        "after setflags(write=True) raises instead of reaching the stored "
        "population (graph amendment 31). Review again if either changes."
    ),
    (
        "packages/microcosm-graph/src/microcosm/graph/executor.py",
        "_read_only_masked",
        "pandas-private-attribute",
        "values._mask",
        "f08675645fa1",
    ): (
        "Read only: the graph executor passes a nullable column's mask to "
        "_read_only_view, which builds a read-only array whose base is a "
        "_ReadOnlyMemory, so a kernel's item write into it, into its .base or "
        "after setflags(write=True) raises instead of reaching the stored "
        "population (graph amendment 31). Review again if either changes."
    ),
}

#: The functions a reviewed reason rests on, by repository-relative path and
#: name, with a digest of each one's source. A change to one needs the
#: reasons that name it reviewed again.
_REVIEWED_HELPERS: dict[tuple[str, str], str] = {
    (
        "tools/verify_uk_identity_stability.py",
        "_pre_clone_fold",
    ): "bb389467cc29",
    (
        "tools/verify_uk_identity_stability.py",
        "_pre_clone_household_weights",
    ): "9d4045e667d2",
    (
        "tools/verify_uk_identity_stability.py",
        "_pre_residential_household_weights",
    ): "6cdb2cd9265e",
    (
        "tools/verify_uk_identity_stability.py",
        "_pre_split_household_weights",
    ): "88cad28e8a13",
    (
        "packages/microcosm-graph/src/microcosm/graph/executor.py",
        "_read_only_view",
    ): "9e403641598e",
    (
        "packages/microcosm-graph/src/microcosm/graph/executor.py",
        "_ReadOnlyMemory",
    ): "fa358510ff88",
}

_MAKES_ARRAY_WRITABLE = "makes-array-writable"
_WRITES_THROUGH_SERIES_ARRAY = "writes-through-series-array"
_PANDAS_PRIVATE_ATTRIBUTE = "pandas-private-attribute"
_NUMPY_COPYTO = "numpy-copyto"
_WRITES_INTO_EXISTING_ARRAY = "writes-into-existing-array"

#: Private attributes refused on any object.
_ALWAYS_PRIVATE_ATTRIBUTES = frozenset({"_ndarray", "_mgr"})
#: Private attributes refused unless read from ``self`` or ``cls``.
_FOREIGN_PRIVATE_ATTRIBUTES = frozenset({"_values", "_data", "_mask", "_codes"})
_OWN_OBJECT_NAMES = frozenset({"self", "cls"})

#: NumPy functions whose result is new memory (unless given ``out=``),
#: checked for ndarrays and for integer, float and boolean Series.
#: ``np.clip``, ``np.tile``, ``np.round`` and ``np.take`` are left out:
#: given a pandas object, their result can share its memory.
_ALLOCATING_NUMPY_FUNCTIONS = frozenset(
    {
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
        "interp",
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
        "repeat",
    }
)

#: NumPy random Generator methods that draw a new array.
_RANDOM_DRAW_METHODS = frozenset(
    {
        "random",
        "normal",
        "uniform",
        "integers",
        "permutation",
        "standard_normal",
        "exponential",
        "poisson",
        "binomial",
    }
)

#: ndarray methods, and NumPy random Generator methods, that take ``out``.
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

#: Text every refused expression contains. A module without any of it is not
#: parsed, and a statement whose lines hold none of it is not visited.
_TRIGGER_TEXT = re.compile(
    r"setflags|writeable|WRITEABLE|\bflags\b|copyto|\.\s*array\b|\bout\s*="
    r"|\.\s*at\s*\(|\.\s*_(?:ndarray|mgr|values|data|mask|codes)\b"
)

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

#: What ``_created_kind`` returns: a new array, or a ``.copy()``.
_NEW_ARRAY = "new array"
_COPY = "copy"


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
        code = self.code if len(self.code) <= 120 else self.code[:117] + "..."
        return f"{self.path}:{self.line} in {self.function}: {self.rule}: {code}"


def _is_false_constant(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Constant) and not expression.value


def _is_none(expression: ast.expr | None) -> bool:
    return isinstance(expression, ast.Constant) and expression.value is None


def _keyword(call: ast.Call, name: str) -> ast.keyword | None:
    return next((keyword for keyword in call.keywords if keyword.arg == name), None)


def _is_true_constant(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Constant) and expression.value is True


def _is_slice(index: ast.expr) -> bool:
    """Whether a subscript's index is a slice (``a[1:]``, ``a[:, 0:2]``)."""

    if isinstance(index, ast.Tuple):
        return any(isinstance(part, ast.Slice) for part in index.elts)
    return isinstance(index, ast.Slice)


def _is_own_object(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Name) and expression.id in _OWN_OBJECT_NAMES


def _is_series_array(expression: ast.expr) -> bool:
    return (
        isinstance(expression, ast.Attribute)
        and expression.attr == "array"
        and not _is_own_object(expression.value)
    )


def _statements(tree: ast.Module) -> Iterator[ast.AST]:
    """Every statement at any depth, without entering expressions."""

    stack: list[ast.AST] = list(reversed(tree.body))
    while stack:
        node = stack.pop()
        yield node
        for field_name in ("body", "orelse", "finalbody", "handlers", "cases"):
            children = getattr(node, field_name, None)
            if isinstance(children, list):
                stack.extend(reversed(children))


class _NumpyNames:
    """The names one module binds to NumPy (``np``, ``from numpy import add``)."""

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


def _numpy_object(dotted_name: str) -> object | None:
    found: object = np
    for part in dotted_name.split(".")[1:]:
        found = getattr(found, part, None)
        if found is None:
            return None
    return found


def _nodes_in_scope(roots: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """The nodes of one function or module, without entering nested ones."""

    stack = list(reversed(list(roots)))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _NESTED_SCOPES):
            stack.extend(reversed(list(ast.iter_child_nodes(node))))


class _Scope:
    """One function, class or module and its assignments, collected on first use.

    ``assignments`` maps a name to the value of each plain assignment to it,
    with ``None`` for a binding of any other kind.
    """

    def __init__(self, qualified_name: str, node: ast.AST) -> None:
        self.qualified_name = qualified_name
        self.node = node
        self._assignments: defaultdict[str, list[ast.expr | None]] | None = None
        self._not_local: set[str] = set()

    def assignments_of(self, name: str) -> list[ast.expr | None] | None:
        """The assignments of a local ``name``, or ``None`` if it is not local."""

        if self._assignments is None:
            self._collect()
        if name in self._not_local:
            return None
        return self._assignments.get(name) or None

    def _collect(self) -> None:
        assignments: defaultdict[str, list[ast.expr | None]] = defaultdict(list)
        node = self.node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            self._not_local.update(
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

        def bind_other(target: ast.AST) -> None:
            # Only names the target binds: ``a[0], b = ...`` binds ``b`` and
            # writes an item of ``a``.
            if isinstance(target, ast.Name):
                assignments[target.id].append(None)
            elif isinstance(target, ast.Starred):
                bind_other(target.value)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    bind_other(element)

        for child in _nodes_in_scope(body):
            if isinstance(child, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                if child.value is None:
                    continue
                targets = (
                    child.targets if isinstance(child, ast.Assign) else [child.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id].append(child.value)
                    elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
                        bind_other(target)
            elif isinstance(child, (ast.For, ast.AsyncFor, ast.comprehension)):
                bind_other(child.target)
            elif isinstance(child, ast.withitem) and child.optional_vars is not None:
                bind_other(child.optional_vars)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                assignments[child.name].append(None)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    assignments[alias.asname or alias.name.split(".", 1)[0]].append(
                        None
                    )
            elif isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                assignments[child.name].append(None)
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                self._not_local.update(child.names)
            elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                assignments[child.name].append(None)
            elif isinstance(child, ast.MatchMapping) and child.rest:
                assignments[child.rest].append(None)
        self._assignments = assignments


def _trigger_lines(source: str) -> list[int] | None:
    """The numbers of the lines holding trigger text, or ``None`` when lines
    cannot be counted as the parser counts them (a lone carriage return)."""

    if "\r" in source and "\r" in source.replace("\r\n", ""):
        return None
    newlines = [match.start() for match in re.finditer("\n", source)]
    return sorted(
        {
            bisect.bisect_left(newlines, match.start()) + 1
            for match in _TRIGGER_TEXT.finditer(source)
        }
    )


class _Scanner(ast.NodeVisitor):
    def __init__(self, source: str, path: str, tree: ast.Module) -> None:
        self._source = source
        self._path = path
        self._numpy = _NumpyNames(tree)
        self._scopes = [_Scope("<module>", tree)]
        self._trigger_lines = _trigger_lines(source)
        self.findings: list[Finding] = []

    def visit(self, node: ast.AST) -> None:
        # Every rule looks at text inside the statement it reports, so a
        # statement without trigger text cannot hold a finding.
        if self._trigger_lines is not None and isinstance(node, ast.stmt):
            decorators = getattr(node, "decorator_list", ())
            first = min([node.lineno, *(item.lineno for item in decorators)])
            index = bisect.bisect_left(self._trigger_lines, first)
            last = node.end_lineno or node.lineno
            if index >= len(self._trigger_lines) or self._trigger_lines[index] > last:
                return
        super().visit(node)

    # Scopes and reporting ------------------------------------------------

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        enclosing = self._scopes[-1].qualified_name
        qualified = name if enclosing == "<module>" else f"{enclosing}.{name}"
        # Decorators, defaults and base classes belong to the enclosing scope.
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        inside = {id(statement) for statement in body}
        for child in ast.iter_child_nodes(node):
            if id(child) not in inside:
                self.visit(child)
        self._scopes.append(_Scope(qualified, node))
        for statement in body:
            self.visit(statement)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_scope(node, "<lambda>")

    def _report(self, node: ast.AST, rule: str) -> None:
        scope = self._scopes[-1]
        if isinstance(scope.node, ast.Module):
            function_source = self._source
        else:
            function_source = ast.get_source_segment(self._source, scope.node) or ""
        segment = ast.get_source_segment(self._source, node) or ast.unparse(node)
        self.findings.append(
            Finding(
                path=self._path,
                line=node.lineno,
                function=scope.qualified_name,
                rule=rule,
                code=" ".join(segment.split()),
                function_digest=hashlib.sha256(
                    function_source.encode("utf-8")
                ).hexdigest()[:12],
            )
        )

    # Created arrays --------------------------------------------------------

    def _created_kind(self, expression: ast.expr) -> str | None:
        """``_NEW_ARRAY`` for a new array; ``_COPY`` for a new object whose
        items may be shared (a ``.copy()``, or ``+``, ``*`` or ``|``, which
        also join containers); ``None`` for anything else."""

        items_taken = False
        while isinstance(expression, ast.Subscript):
            items_taken = items_taken or not _is_slice(expression.slice)
            expression = expression.value
        kind = self._whole_created_kind(expression)
        if kind == _COPY and items_taken:
            return None
        return kind

    def _whole_created_kind(self, expression: ast.expr) -> str | None:
        if isinstance(expression, ast.BinOp):
            joins = isinstance(expression.op, (ast.Add, ast.Mult, ast.BitOr))
            return _COPY if joins else _NEW_ARRAY
        if isinstance(expression, (ast.UnaryOp, ast.Compare)):
            return _NEW_ARRAY
        if not isinstance(expression, ast.Call) or _keyword(expression, "out"):
            return None
        function = expression.func
        name = self._numpy.resolve(function)
        if name is None and isinstance(function, ast.Attribute):
            if function.attr == "copy":
                # ``Index.copy()`` is shallow by default, and a slice of an
                # index shares the index's memory.
                plain = not expression.args and not expression.keywords
                copied = function.value
                while isinstance(copied, ast.Subscript):
                    copied = copied.value
                of_an_index = isinstance(copied, ast.Attribute) and (
                    copied.attr in ("index", "columns")
                )
                return _COPY if plain and not of_an_index else None
            if function.attr == "to_numpy":
                copy = _keyword(expression, "copy")
                return _NEW_ARRAY if copy and _is_true_constant(copy.value) else None
            if function.attr in _RANDOM_DRAW_METHODS:
                return _NEW_ARRAY
        if name is None:
            return None
        if name == "numpy.array":
            copy = _keyword(expression, "copy")
            return _NEW_ARRAY if copy is None or _is_true_constant(copy.value) else None
        if name.removeprefix("numpy.") in _ALLOCATING_NUMPY_FUNCTIONS:
            return _NEW_ARRAY
        return _NEW_ARRAY if isinstance(_numpy_object(name), np.ufunc) else None

    def _is_created(self, expression: ast.expr) -> bool:
        items_taken = False
        while isinstance(expression, ast.Subscript):
            items_taken = items_taken or not _is_slice(expression.slice)
            expression = expression.value
        if isinstance(expression, ast.Name):
            assignments = self._scopes[-1].assignments_of(expression.id)
            if assignments is None:
                return False
            values = [value for value in assignments if not _is_none(value)]
            if not values or any(value is None for value in values):
                return False
            kinds = [self._created_kind(value) for value in values]
        else:
            kinds = [self._created_kind(expression)]
        if None in kinds:
            return False
        return not (items_taken and _COPY in kinds)

    # Rules -----------------------------------------------------------------

    def _written_arrays(self, call: ast.Call) -> list[ast.expr]:
        """``out=`` given to NumPy or an array method, and ``ufunc.at``'s array."""

        written: list[ast.expr] = []
        name = self._numpy.resolve(call.func)
        function = call.func
        output = _keyword(call, "out")
        array_method = (
            isinstance(function, ast.Attribute) and function.attr in _METHODS_WITH_OUT
        )
        if output is not None and (name is not None or array_method):
            if isinstance(output.value, (ast.Tuple, ast.List)):
                written.extend(
                    element for element in output.value.elts if not _is_none(element)
                )
            elif not _is_false_constant(output.value):
                written.append(output.value)
        if name is not None and name.endswith(".at") and call.args:
            if isinstance(_numpy_object(name[: -len(".at")]), np.ufunc):
                written.append(call.args[0])
        return written

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr == "setflags":
                write = _keyword(node, "write")
                flag = write.value if write is not None else None
                if flag is None and node.args:
                    flag = node.args[0]
                if flag is not None and not _is_false_constant(flag):
                    self._report(node, _MAKES_ARRAY_WRITABLE)
            elif function.attr == "__setitem__" and _is_series_array(function.value):
                self._report(node, _WRITES_THROUGH_SERIES_ARRAY)
        for written in self._written_arrays(node):
            if not self._is_created(written):
                self._report(node, _WRITES_INTO_EXISTING_ARRAY)
        self.generic_visit(node)

    def _check_target(
        self, node: ast.AST, target: ast.expr, value: ast.expr | None
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._check_target(node, element, None)
        elif isinstance(target, ast.Starred):
            self._check_target(node, target.value, None)
        elif isinstance(target, ast.Subscript):
            container = target.value
            if (
                isinstance(container, ast.Attribute)
                and container.attr == "flags"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value in ("W", "WRITEABLE")
            ):
                if value is None or not _is_false_constant(value):
                    self._report(node, _MAKES_ARRAY_WRITABLE)
            elif _is_series_array(container):
                self._report(node, _WRITES_THROUGH_SERIES_ARRAY)
        elif (
            isinstance(target, ast.Attribute)
            and target.attr == "writeable"
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "flags"
        ):
            if value is None or not _is_false_constant(value):
                self._report(node, _MAKES_ARRAY_WRITABLE)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(node, target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # An annotation without a value writes nothing.
        if node.value is not None:
            self._check_target(node, node.target, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node, node.target, None)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _ALWAYS_PRIVATE_ATTRIBUTES or (
            node.attr in _FOREIGN_PRIVATE_ATTRIBUTES and not _is_own_object(node.value)
        ):
            self._report(node, _PANDAS_PRIVATE_ATTRIBUTE)
        elif node.attr == "copyto":
            self._report(node, _NUMPY_COPYTO)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None and (
            node.module == "numpy" or node.module.startswith("numpy.")
        ):
            if any(alias.name == "copyto" for alias in node.names):
                self._report(node, _NUMPY_COPYTO)
        self.generic_visit(node)


def scan_source(source: str, path: str) -> list[Finding]:
    """Every refused expression in one module's source."""

    if not _TRIGGER_TEXT.search(source):
        return []
    tree = ast.parse(source, filename=path)
    scanner = _Scanner(source, path, tree)
    scanner.visit(tree)
    return scanner.findings


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
        # utf-8-sig drops a byte-order mark, which ast.parse refuses.
        findings.extend(scan_source(path.read_text(encoding="utf-8-sig"), relative))
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


def _definition(tree: ast.Module, qualified_name: str) -> ast.AST:
    """The one function or class whose qualified name (``Class.method``) is
    ``qualified_name``."""

    found: list[ast.AST] = []
    stack: list[tuple[str, ast.AST]] = [("", node) for node in tree.body]
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{prefix}{node.name}"
            if name == qualified_name:
                found.append(node)
            stack.extend((f"{name}.", child) for child in node.body)
    assert len(found) == 1, f"{qualified_name} is defined {len(found)} times"
    return found[0]


def test_the_functions_reviewed_reasons_rest_on_are_unchanged() -> None:
    changed: list[str] = []
    for (relative, name), digest in _REVIEWED_HELPERS.items():
        source = (_REPOSITORY / relative).read_text(encoding="utf-8-sig")
        function = _definition(ast.parse(source), name)
        segment = ast.get_source_segment(source, function) or ""
        current = hashlib.sha256(segment.encode("utf-8")).hexdigest()[:12]
        if current != digest:
            changed.append(f"{relative}:{name} now has digest {current!r}")

    assert not changed, (
        "A reviewed reason in _REVIEWED_FINDINGS rests on these functions, "
        "which changed. Review the reasons that name them again, then update "
        "_REVIEWED_HELPERS.\n" + "\n".join(changed)
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


def _in_function(*body_lines: str, parameters: str = "values") -> str:
    """A module importing NumPy with one function holding ``body_lines``."""

    body = "".join(f"    {line}\n" for line in body_lines)
    return f"{_NUMPY}def change({parameters}):\n{body}"


#: Snippets the scan must refuse, with the one rule each must report.
_REFUSED_SNIPPETS = {
    # makes-array-writable
    "setflags_write_true": (
        "values.setflags(write=True)\nvalues[0] = 1.0\n",
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
    "flags_writeable_from_a_name": (
        "values.flags.writeable = flag\n",
        _MAKES_ARRAY_WRITABLE,
    ),
    "flags_w_key_from_a_name": ('values.flags["W"] = flag\n', _MAKES_ARRAY_WRITABLE),
    "flags_writeable_from_an_expression": (
        "values.flags.writeable = not frozen\n",
        _MAKES_ARRAY_WRITABLE,
    ),
    # writes-through-series-array
    "series_array_item": (
        'frame.person["income"].array[0] = 1.0\n',
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_augmented": (
        "series.array[mask] += 1\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "series_array_annotated": (
        "series.array[0]: float = 1.0\n",
        _WRITES_THROUGH_SERIES_ARRAY,
    ),
    "out_parameter_rebound_on_one_path": (
        _in_function(
            "if flag:",
            "    values = np.zeros(3)",
            "np.add.at(values, [0], 1)",
            parameters="values, flag",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_lambda_parameter_shadowing_a_module_array": (
        _NUMPY
        + "values = np.zeros(3)\n"
        + "add_one = lambda values: np.add(values, 1, out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_name_only_ever_none": (
        _in_function("buffer = None", "np.add(1, 1, out=buffer)", parameters=""),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_nonlocal_name": (
        _NUMPY
        + "def outer():\n"
        + "    buffer = np.zeros(3)\n"
        + "    def inner():\n"
        + "        nonlocal buffer\n"
        + "        buffer = np.zeros(3)\n"
        + "        np.add(1, 1, out=buffer)\n"
        + "    return inner\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_imported_name_shadowing_a_new_array": (
        _in_function(
            "buffer = np.zeros(3)",
            "import buffer",
            "np.add(1, 1, out=buffer)",
            parameters="",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_an_item_of_a_merged_dict": (
        _in_function(
            'merged = columns | {"w": weights}',
            'np.add.at(merged["income"], [0], 100)',
            parameters="columns, weights",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_an_item_of_joined_lists": (
        _in_function(
            "joined = parts + extra",
            "np.add.at(joined[0], [0], 1)",
            parameters="parts, extra",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_an_index_copy": (
        _in_function(
            "ids = table.index.copy()",
            "np.add.at(ids, positions, 1)",
            parameters="table, positions",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_a_copy_of_an_index_slice": (
        _in_function(
            "ids = table.index[1:].copy()",
            "np.add.at(ids, positions, 1)",
            parameters="table, positions",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_standard_library_random_choice": (
        _in_function(
            "chosen = random.choice(arrays)",
            "np.add(chosen, 1, out=chosen)",
            parameters="random, arrays",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_an_item_of_a_copied_dict": (
        _in_function(
            "updated = columns.copy()",
            'np.add.at(updated["income"], [0], 100)',
            parameters="columns",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_astype": (
        _in_function(
            "converted = series.astype(float)",
            "np.add(converted, 1, out=converted)",
            parameters="series",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
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
    # pandas-private-attribute
    "ndarray_attribute": (
        "values = series.array._ndarray\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "manager_attribute": ("blocks = frame._mgr.blocks\n", _PANDAS_PRIVATE_ATTRIBUTE),
    "manager_attribute_on_self": (
        "class Table:\n    def blocks(self):\n        return self._mgr.blocks\n",
        _PANDAS_PRIVATE_ATTRIBUTE,
    ),
    "series_values_attribute": ("raw = series._values\n", _PANDAS_PRIVATE_ATTRIBUTE),
    "masked_data": ("raw = series.array._data\n", _PANDAS_PRIVATE_ATTRIBUTE),
    "masked_mask": ("series.array._mask[:] = False\n", _PANDAS_PRIVATE_ATTRIBUTE),
    "categorical_codes": (
        "codes = series.cat.codes.array._codes\n",
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
    "copyto_passed_as_function": ("apply_each(np.copyto, pairs)\n", _NUMPY_COPYTO),
    # writes-into-existing-array
    "out_parameter": (
        _in_function("np.add(values, 1, out=values)"),
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
    "out_tuple": (
        _in_function("np.divmod(7, 2, out=(left, right))", parameters="left, right"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_array_method": (
        _in_function("values.clip(0, 1, out=values)"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_generator_method": (
        _in_function("rng.random(out=values)", parameters="rng, values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_through_imported_ufunc": (
        "from numpy import add\ndef change(values):\n    add(1, 1, out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_name_bound_to_a_view": (
        _in_function(
            "buffer = series.to_numpy()",
            "np.add(buffer, 1, out=buffer)",
            parameters="series",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_name_rebound_to_a_view": (
        _in_function(
            "buffer = np.zeros(3)",
            "buffer = series.to_numpy()",
            "np.add(buffer, 1, out=buffer)",
            parameters="series",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_name_bound_to_another_name": (
        _in_function(
            "buffer = np.zeros(3)",
            "alias = buffer",
            "np.add(alias, 1, out=alias)",
            parameters="",
        ),
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
    "out_array_without_copy": (
        _in_function(
            "view = np.array(values, copy=None)",
            "np.add(view, 1, out=view)",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_allocation_given_an_output": (
        _in_function(
            "joined = np.concatenate(parts, out=values)",
            "np.add(joined, 1, out=joined)",
            parameters="parts, values",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_module_array_used_in_a_function": (
        _NUMPY + "buffer = np.zeros(3)\ndef change():\n    np.add(1, 1, out=buffer)\n",
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
    "out_loop_target_shadowing_a_new_array": (
        _in_function(
            "buffer = np.zeros(3)",
            "for buffer in arrays:",
            "    np.add(buffer, 1, out=buffer)",
            parameters="arrays",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_comprehension_target_shadowing_a_new_array": (
        _in_function(
            "buffer = np.zeros(3)",
            "return [np.add(buffer, 1, out=buffer) for buffer in arrays]",
            parameters="arrays",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_with_target_shadowing_a_new_array": (
        _in_function(
            "buffer = np.zeros(3)",
            "with opener() as buffer:",
            "    np.add(buffer, 1, out=buffer)",
            parameters="opener",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_unpacked_shadowing_a_new_array": (
        _in_function(
            "buffer = np.zeros(3)",
            "buffer, other = pair",
            "np.add(buffer, 1, out=buffer)",
            parameters="pair",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_lambda_parameter": (
        _NUMPY + "add_one = lambda values: np.add(values, 1, out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_in_a_decorator_over_a_function_binding_the_name": (
        _NUMPY
        + "@register(np.add(1, 1, out=buffer))\n"
        + "def change():\n"
        + "    buffer = np.zeros(3)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at": (
        _in_function("np.add.at(values, [0], 1)"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_imported": (
        "from numpy import add\ndef count(values):\n    add.at(values, [0], 1)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_after_lone_carriage_returns": (
        "import numpy as np\rdef change(values):\r    np.add(values, 1, out=values)\r",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_in_a_decorator_over_a_plain_function": (
        _NUMPY
        + "@register(np.add(1, 1, out=buffer))\n"
        + "def change():\n"
        + "    return 1\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_to_numpy": (
        _NUMPY + "np.maximum.at(series.to_numpy(), positions, weights)\n",
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
#: arrays the function created.
_ACCEPTED_SNIPPETS = {
    "setflags_read_only": "values.setflags(write=False)\nvalues.setflags(write=0)\n",
    "setflags_alignment_only": "values.setflags(align=True)\n",
    "flags_read_only": (
        'values.flags.writeable = False\nvalues.flags["WRITEABLE"] = False\n'
    ),
    "series_array_reads": (
        "for value in series.array:\n"
        + "    total += value\n"
        + "index = pd.Index(series.array, name='id')\n"
        + "first = series.array[0]\n"
    ),
    "own_array_and_private_state": (
        "class Grid:\n"
        + "    def set(self, row, value):\n"
        + "        self.array[row] = value\n"
        + "        self._values[row] = value\n"
        + "        self._data.append(value)\n"
        + "    @classmethod\n"
        + "    def empty(cls):\n"
        + "        return cls._mask\n"
    ),
    "array_function_not_attribute": (
        _NUMPY
        + "import pandas as pd\n"
        + "values = np.array(series)\n"
        + "values[0] = 1\n"
        + "codes = pd.array([1, 2])\n"
        + "codes[0] = 3\n"
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
    "out_copy": _in_function(
        "total = first.copy()",
        "np.add(total, later, out=total)",
        parameters="first, later",
    ),
    "out_same_array_as_input": _in_function(
        "draws = np.empty(count)",
        "np.minimum(draws, 1.0, out=draws)",
        "return draws",
        parameters="count",
    ),
    "out_rebound_between_new_arrays": _in_function(
        "out = np.zeros(3)",
        "np.add.at(out, positions, weights)",
        "out = np.where(mask, out, 0.0)",
        "np.multiply(out, 2, out=out)",
        parameters="positions, weights, mask",
    ),
    "out_none_placeholder_then_copy": _in_function(
        "total: np.ndarray | None = None",
        "for current in slices:",
        "    if total is None:",
        "        total = current.copy()",
        "    else:",
        "        np.add(total, current, out=total)",
        parameters="slices",
    ),
    "out_ufunc_and_array_results": _in_function(
        "scaled = np.exp(values)",
        "np.multiply(scaled, 2, out=scaled)",
        "owned = np.array(values)",
        "np.add(owned, 1, out=owned)",
    ),
    "out_none": _NUMPY + "np.add(left, right, out=None)\n",
    "out_none_pair": _NUMPY + "np.divmod(left, 2, out=(None, None))\n",
    "out_keyword_of_workspace_functions": (
        "launch(pool_manifest=manifest, out=args.out)\n"
        + "report.write(rows, out=destination)\n"
    ),
    "ufunc_at_new_array": _in_function(
        "counts = np.zeros(size, dtype=np.int64)",
        "np.add.at(counts, positions, 1)",
        parameters="positions, size",
    ),
    "ufunc_without_out": _NUMPY + "total = np.add(left, right)\n",
    "out_copies_the_failure_message_suggests": _in_function(
        "weights = np.array(frame.w, copy=True)",
        "np.add.at(weights, positions, 1)",
        "values = frame.x.to_numpy(copy=True)",
        "np.add(values, 1, out=values)",
        parameters="frame, positions",
    ),
    "out_arithmetic_and_draws": _in_function(
        "product = left * right",
        "np.add(product, 1, out=product)",
        "positive = left > 0",
        "np.logical_not(positive, out=positive)",
        "draws = rng.random(size)",
        "np.multiply(draws, 2, out=draws)",
        parameters="left, right, rng, size",
    ),
    "out_numpy_copy": _in_function(
        "buffer = np.copy(values)",
        "np.add(buffer, 1, out=buffer)",
        "np.add.at(np.copy(values), [0], 1)",
    ),
    "out_names_bound_to_subscripts_of_new_arrays": _in_function(
        "ordered = np.sort(values)[::-1]",
        "np.cumsum(ordered, out=ordered)",
        "row = np.zeros((2, size))[0]",
        "np.add(row, 1, out=row)",
        parameters="values, size",
    ),
    "out_items_of_a_new_array_assigned_by_unpacking": _in_function(
        "bounds = np.empty(2)",
        "bounds[0], bounds[1] = low, high",
        "np.clip(bounds, 0, 1, out=bounds)",
        "order = np.argsort(values)",
        "order[0], order[-1] = order[-1], order[0]",
        "np.add(order, 1, out=order)",
        parameters="low, high, values",
    ),
    "out_multidimensional_slice_of_a_copy": _in_function(
        "result = matrix.copy()",
        "np.clip(result[:, 1:], 0, None, out=result[:, 1:])",
        parameters="matrix",
    ),
    "out_slice_of_a_copy": _in_function(
        "totals = first.copy()",
        "np.cumsum(later, out=totals[1:])",
        parameters="first, later",
    ),
    "annotation_without_a_value": ('series.array[0]: float\nvalues.flags["W"]: bool\n'),
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
    original = _in_function("np.add.at(values, [0], 1)")
    changed = _in_function("values = values.T", "np.add.at(values, [0], 1)")

    (before,) = scan_source(original, "pkg/module.py")
    (after,) = scan_source(changed, "pkg/module.py")

    assert before.code == after.code
    assert before.key != after.key


def test_every_output_in_an_out_tuple_is_checked() -> None:
    both_existing = _in_function(
        "np.divmod(7, 2, out=(left, right))", parameters="left, right"
    )
    second_existing = _in_function(
        "np.divmod(7, 2, out=(np.zeros(1), right))", parameters="right"
    )

    assert len(scan_source(both_existing, "snippet.py")) == 2
    assert len(scan_source(second_existing, "snippet.py")) == 1
