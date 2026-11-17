"""No raw writes into memory that pandas copy-on-write does not protect.

:meth:`Frame.replace_tables` and :meth:`Frame.with_columns` build frames that
share the memory of every unchanged column with their source frame
(:meth:`Frame._adopt`). pandas copy-on-write keeps a write made through pandas
(``.loc``, column assignment, in-place methods) on one frame from reaching
the other, but it does not see a raw write into the shared memory. Checked
with pandas 3.0.3 and NumPy 2.4.6, the writes below change the other frame's
data without an error, so this test reads the scanned source with :mod:`ast`
and refuses them. Each rule's name is what a failure reports.

``makes-array-writable``
    ``setflags(write=...)`` with anything but a false constant, or assigning
    ``flags.writeable``, ``flags["W"]`` or ``flags["WRITEABLE"]``. The arrays
    ``to_numpy()`` and ``.values`` return for NumPy-typed columns are
    read-only views of the shared memory; this makes them writable.
``writes-through-writable-view``
    A write into a writable view of pandas' memory: ``.array`` (of a Series
    or Index), ``.asi8`` (datetime data), or ``.base`` read from a call's
    result (such as ``to_numpy()``) or from ``.values`` or ``.array``, which
    is the writable array behind a read-only view.

    Writes are item assignment, ``__setitem__``, ``fill``, ``sort`` and the
    other in-place ndarray methods, ``fillna(copy=False)``, ``out=``,
    ``shuffle`` and the in-place NumPy functions below, into the view or
    into an item of it (a row of a two-dimensional view is a view too).
    ``+=`` and the other augmented assignments count only into a view
    itself, because an item of a one-dimensional view is a scalar, which
    ``+=`` rebinds; and not into a name bound only to ``.array``, an
    extension array, for which ``+=`` builds a new array.

    The written expression reaches the view through:

    - indexing of any kind;
    - the NumPy functions that can return a view (``np.asarray``,
      ``np.flip``, ``np.split``, ... in ``_VIEW_NUMPY_FUNCTIONS`` and
      ``_SPLITTING_NUMPY_FUNCTIONS``), ``to_numpy()``,
      ``astype(copy=False)`` and the other view methods and attributes,
      unless given ``copy=True``;
    - a list, tuple, set or dict display or comprehension holding it (with
      ``*``/``**`` unpacking), containers joined with ``+`` or ``|``,
      ``list``, ``tuple``, ``sorted``, ``dict``, ``zip``, ``enumerate``,
      ``next``, ``.items()``, ``.values()``, ``.copy()`` of a container,
      and ``.get()``, ``.pop()`` and ``.setdefault()`` (with a default);
    - ``or`` and conditional expressions;
    - a name the same function binds to any of these, by assignment,
      unpacking, a ``with`` statement, or a ``for`` loop or comprehension
      (which pair ``zip``, ``enumerate`` and ``.items()`` targets with
      their own sources, also when nested or wrapped in ``list`` or
      ``sorted``), or fills with one (``append``, ``extend``, ``insert``,
      ``add``, ``update``, ``setdefault``, ``name[key] = ...``, ``+=``).

    Workspace classes' own ``self.array`` is exempt.
``pandas-private-attribute``
    ``._ndarray``, ``._mgr``, ``._get_column_array`` or
    ``._iter_column_arrays`` on anything, and ``._values``, ``._data``,
    ``._mask`` or ``._codes`` on anything but ``self`` or ``cls``. These
    reach pandas' internal arrays, which are always writable.
``numpy-copyto``
    Any use of ``numpy.copyto``.
``writes-into-existing-array``
    A write into an array the same function did not create: ``out=``
    given to a NumPy or SciPy function or ufunc, an ndarray method or a
    NumPy random ``Generator`` method (by keyword, or by position for NumPy
    and SciPy functions and ufuncs), the first argument of
    ``numpy.<ufunc>.at`` (which writes even into a read-only array), and the
    first argument of ``numpy.put``, ``numpy.putmask``, ``numpy.place``,
    ``numpy.put_along_axis``, ``numpy.fill_diagonal``,
    ``numpy.random.shuffle``, any ``.shuffle`` method and
    ``numpy.nan_to_num(copy=False)``. Library functions are also followed
    through a local name bound to them (``plus = np.add``).

    Created means one of:

    - the result of a ufunc or ufunc reduction, a ``numpy.random``
      function, a random ``Generator`` draw (``rng.normal``, ...),
      ``.flatten()``, or a NumPy function that always allocates
      (``_ALLOCATING_NUMPY_FUNCTIONS``: ``np.zeros``, ``np.where``,
      ``np.concatenate``, ...), when not given an output array;
    - ``np.array`` without ``copy=``, or ``to_numpy(copy=True)``;
    - a view, piece (``np.split``), subscript or item (a loop target,
      also in a comprehension, or unpacking) of one of these. ``np.clip``,
      ``np.round`` and ``np.take`` count only as views here: given a
      pandas object they can return its memory;
    - arithmetic or a comparison;
    - a ``.copy()``, ``list(...)`` and the other container builtins, ``+``,
      ``*`` or ``|`` (which also join containers), or a list, set or dict
      display. A slice of these is new; an item is not, because a shallow
      copy or a container shares its items;
    - a local name whose every binding in the function, other than to
      ``None``, is one of these.

``setflags``, ``Series.array[i] = value``, ``._ndarray``, ``._mgr``,
``np.copyto`` and ``out=`` are the writes first identified as unprotected.
The other rules, views, functions and methods were added after checking that
they write into shared memory too.

Scope: every Python file under ``packages/*/src/`` (the source of every
workspace package, including microcosm-frame and the microcosm-build code that
builds US and UK frames) and under ``tools/`` (the release and build tools),
except files in ``node_modules`` or hidden directories. Tests and
``test_support/`` are not scanned.

The scan follows arrays only through the forms listed above. In particular,
it does not refuse:

- Writes it cannot see from the source:
  - into the extension arrays of extension-typed columns: ``.values``
    returns the shared array for ``Int64``, ``Float64``, ``boolean``,
    ``category`` and python-backed string columns, and ``to_numpy()`` a
    writable view of it for all of these but ``category``;
  - through a pandas object built with ``copy=False`` over another object's
    array;
  - into ``np.asarray(index)``, which is writable although the index is
    shared even between deep copies;
  - ``+=`` on a datetime column's ``.array``, which writes in place.
- Arrays reached in other ways:
  - through other functions' return values or arguments, closures and
    object arrays;
  - names reached dynamically (``getattr(np, name)``, ``**{"out": array}``,
    ``*arguments``), and positional ``out`` given to ndarray methods;
  - items of a container the function did not build;
  - array libraries other than NumPy and SciPy (``torch.from_numpy`` of a
    read-only view returns a writable tensor).
- Differences of order or kind within a function:
  - Neither rule is flow-sensitive: a name counts as created only if every
    binding in its function does, and as a view if any binding is one. So
    a parameter rebound to a new array, and a name rebound from a view to a
    new array, count as before.
  - A boolean or integer array index gives a copy, but counts as indexing.
  - ``.astype()`` counts as not created, because a pandas object's
    same-type conversion shares its memory.

A finding that cannot write into memory a pandas object shares goes in
``_REVIEWED_FINDINGS`` with the reason.
"""

from __future__ import annotations

import ast
import bisect
import functools
import hashlib
import importlib
import inspect
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
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
        "np.asarray(...).copy() from the household weights. Review again if "
        "either of those functions changes."
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
#: ``.base`` counts too, when read from an array expression.
_WRITABLE_VIEW_ATTRIBUTES = frozenset({"array", "asi8"})

#: Libraries whose functions the scan resolves through imports.
_LIBRARIES = ("numpy", "scipy")

#: NumPy functions whose result is new memory (unless given ``out=``),
#: checked for ndarrays and for integer, float and boolean Series.
#: ``np.clip``, ``np.tile``, ``np.round`` and ``np.take`` are treated as
#: views instead: given a pandas object, their result can share its memory.
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
    )
)

#: ufunc methods whose result is a new array (unless given ``out=``).
_ALLOCATING_UFUNC_METHODS = frozenset({"reduce", "accumulate", "reduceat", "outer"})

#: NumPy random Generator methods that draw a new array.
_RANDOM_DRAW_METHODS = frozenset(
    {
        "random",
        "normal",
        "uniform",
        "integers",
        "choice",
        "permutation",
        "standard_normal",
        "standard_exponential",
        "standard_gamma",
        "exponential",
        "gamma",
        "beta",
        "binomial",
        "poisson",
        "lognormal",
        "multinomial",
        "multivariate_normal",
        "dirichlet",
        "laplace",
        "logistic",
        "triangular",
        "geometric",
    }
)

#: Builtins whose result is a new container (whose items may be shared).
_CONTAINER_BUILTINS = frozenset(
    {"list", "dict", "set", "frozenset", "tuple", "sorted", "bytearray"}
)
#: Builtins and methods whose result holds the items of their arguments or
#: receiver, for following views through iteration.
_SAME_ITEMS_BUILTINS = frozenset(
    {"reversed", "iter", "list", "tuple", "sorted", "set", "frozenset"}
)
_TUPLE_YIELDING_BUILTINS = frozenset({"zip", "enumerate"})
#: Builtins whose result has exactly the items of their one argument.
_ITEM_KEEPING_WRAPPERS = frozenset({"list", "tuple", "sorted", "reversed", "iter"})
_SAME_ITEMS_METHODS = frozenset({"values", "popitem"})
_TUPLE_YIELDING_METHODS = frozenset({"items"})
#: Methods that return one item of the container they are called on.
_ITEM_TAKING_METHODS = frozenset({"get", "pop", "setdefault"})
#: Methods that put one item, or the items of an iterable, into the
#: container they are called on.
_ITEM_ADDING_METHODS = frozenset(
    {"append", "appendleft", "insert", "add", "setdefault"}
)
_ITEMS_ADDING_METHODS = frozenset({"extend", "update"})

#: Arithmetic and comparisons give a new array; ``+`` and ``*`` also join or
#: repeat lists, whose items are shared, so they give a new container.
_NEW_ARRAY_EXPRESSIONS = (ast.UnaryOp, ast.Compare)
_JOINING_OPERATORS = (ast.Add, ast.Mult, ast.BitOr)
#: Operators that join containers holding views (``*`` repeats a list, but
#: with an array it is arithmetic, so it is left out).
_VIEW_JOINING_OPERATORS = (ast.Add, ast.BitOr)
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
        "numpy.random.mtrand.shuffle",
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

#: NumPy functions that can return a view of an argument (or views, as
#: ``np.split`` does).
_VIEW_NUMPY_FUNCTIONS = frozenset(
    f"numpy.{name}"
    for name in (
        "asarray",
        "asanyarray",
        "ascontiguousarray",
        "asfortranarray",
        "asarray_chkfinite",
        "require",
        "ravel",
        "reshape",
        "squeeze",
        "transpose",
        "permute_dims",
        "matrix_transpose",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "frombuffer",
        "from_dlpack",
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
        "trim_zeros",
        "tile",
        "real_if_close",
        "asmatrix",
        "lib.stride_tricks.as_strided",
    )
)

#: NumPy functions that return new memory for an ndarray or an extension
#: array, but, given a Series or DataFrame, call its method, whose result can
#: share the object's memory: never counted as created.
_PANDAS_SHARING_NUMPY_FUNCTIONS = frozenset(
    f"numpy.{name}" for name in ("clip", "round", "around", "take")
)

#: NumPy functions that return a list or tuple of views of an argument.
_SPLITTING_NUMPY_FUNCTIONS = frozenset(
    f"numpy.{name}"
    for name in (
        "split",
        "array_split",
        "hsplit",
        "vsplit",
        "dsplit",
        "unstack",
        "broadcast_arrays",
    )
)

#: Methods and attributes that can return a view of the object they are read
#: from.
_VIEW_METHODS = frozenset(
    {
        "to_numpy",
        "view",
        "reshape",
        "ravel",
        "squeeze",
        "transpose",
        "swapaxes",
        "__array__",
    }
)
_VIEW_ATTRIBUTES = frozenset({"T", "mT", "real", "imag", "flat"})

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

#: How new the value of an expression is (higher is newer).
_NOT_NEW = 0
#: A new object whose items may be shared: a container, or a ``.copy()``
#: that may be a shallow copy of a container.
_NEW_OBJECT = 1
#: A new array: every item and view of it is new memory.
_NEW_ARRAY = 2

#: Whether, and how deep in containers, a value holds a writable view of
#: pandas' memory. ``_IS_VIEW`` is a view. ``_ELEMENT`` is an item of a view:
#: a row (itself a view) of a two-dimensional view, but a scalar of a
#: one-dimensional one. Each container level adds 2, so 2 is a container of
#: views and 3 a container of elements. ``_NO_VIEW`` is none, or deeper than
#: tracked.
_IS_VIEW = 0
_ELEMENT = 1
_NO_VIEW = 11


def _item_of(level: int) -> int:
    """An item of a view is an element; an item of a container, one level less."""

    if level >= _NO_VIEW:
        return _NO_VIEW
    if level <= _ELEMENT:
        return _ELEMENT
    return level - 2


def _items_taken(level: int, count: int) -> int:
    for _step in range(count):
        level = _item_of(level)
    return level


def _holder_of(level: int) -> int:
    """A container holding a value holds its view one level deeper."""

    return _NO_VIEW if level >= _NO_VIEW else min(level + 2, _NO_VIEW)


def _items_holder_of(level: int) -> int:
    """A new container of a value's items (``list(value)``, ``extend``)."""

    return _holder_of(_item_of(level))


def _view_of(level: int) -> int:
    """A NumPy view of a view or element stays one; of a container it is a copy."""

    return level if level <= _ELEMENT else _NO_VIEW


def _bound_level(items_taken: int, level: int) -> int:
    """The level of a name bound ``items_taken`` items deep into a value."""

    return _items_taken(level, items_taken)


def _held_level(items_taken: int, one_item: bool, level: int) -> int:
    """The level of a container filled with a value (see ``_HeldValue``)."""

    level = _items_taken(level, items_taken)
    return _holder_of(level) if one_item else _items_holder_of(level)


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


# Bindings ------------------------------------------------------------------

#: One binding of a name: the value expression (``None`` when it cannot be
#: followed, as for an import), and how many items deep into that value the
#: name is bound: 0 for the value itself, 1 for a loop target (an item of
#: its iterable), 2 for a name unpacked from a loop target, and so on.
_Binding = tuple[ast.expr | None, int]
#: What a container is filled with: the value, how many items deep into it,
#: and whether it goes in as one item (``append``, ``name[key] = value``)
#: rather than as its items (``extend``, ``update``, ``+=``).
_HeldValue = tuple[ast.expr, int, bool]


def _target_names(
    target: ast.expr, value: ast.expr | None, items_taken: int
) -> Iterator[tuple[ast.expr, ast.expr | None, int]]:
    """The names and subscripts an assignment target binds, with their values.

    Each is yielded with the value it takes items from and how many items
    deep it is bound.
    """

    if isinstance(target, (ast.Name, ast.Subscript)):
        yield target, value, items_taken
    elif isinstance(target, ast.Starred):
        # A starred target collects items into a new list: the same depth.
        yield from _target_names(target.value, value, items_taken)
    elif isinstance(target, (ast.Tuple, ast.List)):
        pairs = (
            items_taken == 0
            and isinstance(value, (ast.Tuple, ast.List))
            and len(value.elts) == len(target.elts)
            and not any(
                isinstance(element, ast.Starred)
                for element in (*target.elts, *value.elts)
            )
        )
        for index, element in enumerate(target.elts):
            if pairs:
                yield from _target_names(element, value.elts[index], 0)
            elif isinstance(element, ast.Starred):
                yield from _target_names(element.value, value, items_taken)
            else:
                yield from _target_names(element, value, items_taken + 1)


def _loop_target_names(
    target: ast.expr, iterable: ast.expr
) -> Iterator[tuple[ast.expr, ast.expr | None, int]]:
    """What a ``for`` target binds, pairing each name with its own source.

    ``for a, b in zip(x, y)`` binds ``a`` to an item of ``x`` and ``b`` to
    an item of ``y``; ``enumerate`` and ``.items()`` pair the second name
    with the iterable's items. Pairs nest (``enumerate(zip(...))``), and
    ``list``, ``sorted`` and the other wrappers that keep the items change
    nothing.
    """

    while (
        isinstance(iterable, ast.Call)
        and isinstance(iterable.func, ast.Name)
        and iterable.func.id in _ITEM_KEEPING_WRAPPERS
        and len(iterable.args) == 1
        and not isinstance(iterable.args[0], ast.Starred)
    ):
        iterable = iterable.args[0]
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and not any(isinstance(element, ast.Starred) for element in target.elts)
        and isinstance(iterable, ast.Call)
        and not any(isinstance(argument, ast.Starred) for argument in iterable.args)
    ):
        function = iterable.func
        elements = target.elts
        arguments = iterable.args
        if (
            isinstance(function, ast.Name)
            and function.id == "zip"
            and len(arguments) == len(elements)
        ):
            for element, argument in zip(elements, arguments, strict=True):
                yield from _loop_target_names(element, argument)
            return
        if (
            isinstance(function, ast.Name)
            and function.id == "enumerate"
            and len(elements) == 2
            and arguments
        ):
            yield from _target_names(elements[0], None, 1)
            yield from _loop_target_names(elements[1], arguments[0])
            return
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "items"
            and len(elements) == 2
        ):
            yield from _target_names(elements[0], None, 1)
            yield from _loop_target_names(elements[1], function.value)
            return
    yield from _target_names(target, iterable, 1)


class _Scope:
    """One function, class or module, and the names it binds.

    The bindings are collected on first use, because most scopes are never
    asked. ``held_values`` are what a name's container is filled with
    (``name.append(value)``, ``name[key] = value``, ``name += value``).
    """

    def __init__(self, qualified_name: str, node: ast.AST) -> None:
        self.qualified_name = qualified_name
        self.node = node
        self._parameters: frozenset[str] = frozenset()
        self._declared_elsewhere: set[str] = set()
        self._bindings: defaultdict[str, list[_Binding]] | None = None
        self._held: defaultdict[str, list[_HeldValue]] = defaultdict(list)
        # Filled by the scanner on first use, held here so that they live
        # exactly as long as the scope.
        self.newness_table: dict[str, int] | None = None
        self.view_table: dict[str, int] | None = None

    def child_name(self, name: str) -> str:
        if self.qualified_name == "<module>":
            return name
        return f"{self.qualified_name}.{name}"

    def is_local(self, name: str) -> bool:
        self._collect()
        return (
            name not in self._parameters
            and name not in self._declared_elsewhere
            and bool(self._bindings.get(name))
        )

    def bindings_of(self, name: str) -> list[_Binding] | None:
        """The bindings of ``name`` here, or ``None`` when it is not local.

        A parameter, a name declared ``global`` or ``nonlocal``, and a name
        this scope never binds are not local.
        """

        return self._bindings[name] if self.is_local(name) else None

    def is_parameter(self, name: str) -> bool:
        self._collect()
        return name in self._parameters

    def local_bindings(self) -> dict[str, list[_Binding]]:
        self._collect()
        return {
            name: bindings
            for name, bindings in self._bindings.items()
            if self.is_local(name)
        }

    def held_names(self) -> list[str]:
        self._collect()
        return list(self._held)

    def held_values(self, name: str) -> list[_HeldValue]:
        self._collect()
        return self._held.get(name, [])

    def _record(self, bound: Iterable[tuple[ast.expr, ast.expr | None, int]]) -> None:
        for target, value, items_taken in bound:
            if isinstance(target, ast.Name):
                self._bindings[target.id].append((value, items_taken))
            elif isinstance(target.value, ast.Name) and value is not None:
                # ``name[key] = value`` puts one item into ``name``.
                self._held[target.value.id].append((value, items_taken, True))

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
                    self._record(_target_names(target, child.value, 0))
            elif isinstance(child, (ast.AnnAssign, ast.NamedExpr)):
                if child.value is not None:
                    self._record(_target_names(child.target, child.value, 0))
            elif isinstance(child, ast.AugAssign):
                # ``name += values`` adds the items of ``values`` to a list;
                # ``name[key] += value`` changes an item and adds none.
                if isinstance(child.target, ast.Name):
                    self._held[child.target.id].append((child.value, 0, False))
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                self._record(_loop_target_names(child.target, child.iter))
            elif isinstance(child, ast.withitem) and child.optional_vars is not None:
                self._record(_target_names(child.optional_vars, child.context_expr, 1))
            elif isinstance(child, ast.ExceptHandler) and child.name:
                self._bindings[child.name].append((None, 1))
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    self._bindings[bound].append((None, 1))
            elif isinstance(child, _NESTED_SCOPES) and not isinstance(
                child, ast.Lambda
            ):
                self._bindings[child.name].append((None, 1))
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                self._declared_elsewhere.update(child.names)
            elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                self._bindings[child.name].append((None, 1))
            elif isinstance(child, ast.MatchMapping) and child.rest:
                self._bindings[child.rest].append((None, 1))
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
            ):
                method = child.func.attr
                if method in _ITEM_ADDING_METHODS or method in _ITEMS_ADDING_METHODS:
                    one_item = method in _ITEM_ADDING_METHODS
                    self._held[child.func.value.id].extend(
                        (argument, 0, one_item)
                        for argument in (
                            *child.args,
                            *(keyword.value for keyword in child.keywords),
                        )
                    )


class _ComprehensionScope:
    """A comprehension: its loop targets shadow the enclosing scope's names.

    ``upto`` limits it to its first generators, the ones whose targets a
    later generator's iterable can see.
    """

    def __init__(
        self,
        enclosing: _Scope | _ComprehensionScope,
        node: ast.AST,
        upto: int | None = None,
    ) -> None:
        self.enclosing = enclosing
        self.qualified_name = enclosing.qualified_name
        self.node = enclosing.node
        self.comprehension = node
        self.upto = upto
        self._targets: defaultdict[str, list[tuple[ast.expr | None, int, int]]] = (
            defaultdict(list)
        )
        generators = node.generators if upto is None else node.generators[:upto]
        for index, generator in enumerate(generators):
            for target, value, items_taken in _loop_target_names(
                generator.target, generator.iter
            ):
                if isinstance(target, ast.Name):
                    self._targets[target.id].append((value, items_taken, index))

    def before(self, index: int) -> _ComprehensionScope:
        """This comprehension as seen from its generator ``index``."""

        return _ComprehensionScope(self.enclosing, self.comprehension, upto=index)

    def child_name(self, name: str) -> str:
        return self.enclosing.child_name(name)

    def target_sources(
        self, name: str
    ) -> list[tuple[ast.expr | None, int, int]] | None:
        """What ``name`` takes items from, if it is a loop target here: each
        source with its item count and the index of its generator."""

        return self._targets.get(name)

    def bindings_of(self, name: str) -> list[_Binding] | None:
        if name in self._targets:
            return [(value, items) for value, items, _index in self._targets[name]]
        return self.enclosing.bindings_of(name)

    def is_parameter(self, name: str) -> bool:
        return name in self._targets or self.enclosing.is_parameter(name)


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


def _is_library_module(module: str) -> bool:
    return any(
        module == library or module.startswith(f"{library}.") for library in _LIBRARIES
    )


class _LibraryNames:
    """The names one module binds to NumPy and SciPy, to resolve ``np.add.at``.

    ``derived_names`` are names some statement assigns a library object to
    (``plus = np.add``); the scanner follows them through the bindings of
    the scope that uses them.
    """

    def __init__(self, tree: ast.Module) -> None:
        self._aliases: dict[str, str] = {}
        statements = list(_statements(tree))
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_library_module(alias.name):
                        continue
                    if alias.asname:
                        self._aliases[alias.asname] = alias.name
                    else:
                        root = alias.name.split(".", 1)[0]
                        self._aliases[root] = root
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
                and _is_library_module(node.module)
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
        """The dotted library name of ``expression`` (``numpy.add.at``), if any."""

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
#: Text found in every refused expression, except library names, names bound
#: to library objects, and writes through a name bound to a writable view.
_TRIGGER_TEXT = (
    r"setflags|writeable|WRITEABLE|\bflags\b|copyto|shuffle|\bout\s*="
    rf"|{_DOT}_(?:ndarray|mgr|values|data|mask|codes|get_column_array"
    r"|iter_column_arrays)\b"
)
_WRITABLE_VIEW_TEXT = re.compile(rf"{_DOT}(?:array|base|asi8)\b")
#: A module without any of this text cannot hold a finding and is not parsed.
_FILE_TRIGGERS = re.compile(
    rf"{_TRIGGER_TEXT}|numpy|scipy|{_WRITABLE_VIEW_TEXT.pattern}"
)


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


# Library lookups -----------------------------------------------------------


@functools.cache
def _library_object(dotted_name: str) -> object | None:
    """The NumPy or SciPy object a dotted name names, importing its module."""

    parts = dotted_name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            found: object = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for part in parts[split:]:
            found = getattr(found, part, None)
            if found is None:
                return None
        return found
    return None


@functools.cache
def _positional_outputs(dotted_name: str) -> slice | None:
    """Which positional arguments of a library function or ufunc are outputs."""

    function = _library_object(dotted_name)
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


def _items_newness(level: int, items_taken: int) -> int:
    """How new a name bound ``items_taken`` items deep into a value is: an
    item of a new array is new memory, an item of a container may not be."""

    if items_taken == 0:
        return level
    return _NEW_ARRAY if level == _NEW_ARRAY else _NOT_NEW


def _is_ufunc_method(dotted_name: str, methods: Iterable[str]) -> bool:
    owner, _, method = dotted_name.rpartition(".")
    return method in methods and isinstance(_library_object(owner), np.ufunc)


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

#: How a name is looked up while a scope's table is computed.
_NameLookup = Callable[[str], object]


class _Scanner:
    def __init__(self, source: str, path: str, tree: ast.Module) -> None:
        self._source = source
        self._path = path
        self._tree = tree
        self._libraries = _LibraryNames(tree)
        lone_carriage_return = "\r" in source and "\r" in source.replace("\r\n", "")
        # Without ``.array``, ``.base`` or ``.asi8`` text, nothing in the
        # module can reach a writable view. A write through a name bound to
        # one holds no trigger text of its own, so a module with that text
        # is scanned in full.
        self._has_writable_views = bool(_WRITABLE_VIEW_TEXT.search(source))
        self._trigger_lines = (
            None
            if lone_carriage_return or self._has_writable_views
            else _TriggerLines(
                source, self._libraries.bound_names | self._libraries.derived_names
            )
        )
        self._module_scope = _Scope("<module>", tree)
        self._scope: _Scope | _ComprehensionScope = self._module_scope
        self._digests: dict[int, str] = {}
        self._target_levels: dict[tuple[bool, int, int | None, str], int] = {}
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
        self.findings.append(
            Finding(
                path=self._path,
                line=node.lineno,
                function=self._scope.qualified_name,
                rule=rule,
                code=" ".join(segment.split()),
                function_digest=self._function_digest(),
            )
        )

    # Following names -------------------------------------------------------

    def _bindings(self, name: str) -> list[_Binding] | None:
        """The bindings of ``name`` in the current scope, if it is local."""

        return self._scope.bindings_of(name)

    def _owning_scope(self) -> _Scope:
        scope = self._scope
        while isinstance(scope, _ComprehensionScope):
            scope = scope.enclosing
        return scope

    def _resolve(self, expression: ast.expr) -> str | None:
        """The dotted library name of ``expression``, following ``plus = np.add``."""

        name = self._libraries.resolve(expression)
        if name is not None:
            return name
        attributes: list[str] = []
        root = expression
        while isinstance(root, ast.Attribute):
            attributes.append(root.attr)
            root = root.value
        if (
            not isinstance(root, ast.Name)
            or root.id not in self._libraries.derived_names
        ):
            return None
        bindings = self._bindings(root.id)
        if bindings is None and not self._scope.is_parameter(root.id):
            bindings = self._module_scope.bindings_of(root.id)
        if not bindings or any(items_taken for _value, items_taken in bindings):
            return None
        resolved = {
            self._libraries.resolve(value) for value, _items in bindings if value
        }
        if len(resolved) != 1 or None in resolved:
            return None
        return ".".join([resolved.pop(), *reversed(attributes)])

    # How new a value is ----------------------------------------------------

    def _name_newness(self, name: str) -> int:
        """How new ``name``'s value is where it is used, in the current scope."""

        scope = self._scope
        while isinstance(scope, _ComprehensionScope):
            sources = scope.target_sources(name)
            if sources is not None:
                return self._comprehension_target(scope, name, sources, newness=True)
            scope = scope.enclosing
        return self._newness_table(scope).get(name, _NOT_NEW)

    def _comprehension_target(
        self,
        scope: _ComprehensionScope,
        name: str,
        sources: list[tuple[ast.expr | None, int, int]],
        *,
        newness: bool,
    ) -> int:
        """How new, or how close to a view, a comprehension's loop target is.

        Each source is measured as the comprehension sees it from that
        target's generator. Results are cached: they depend only on the
        comprehension and the tables of the scopes around it.
        """

        key = (newness, id(scope.comprehension), scope.upto, name)
        cached = self._target_levels.get(key)
        if cached is not None:
            return cached
        saved_scope = self._scope
        levels: list[int] = []
        try:
            for value, items_taken, index in sources:
                if value is None:
                    if newness:
                        levels.append(_NOT_NEW)
                    continue
                self._scope = scope.before(index)
                if newness:
                    levels.append(_items_newness(self._newness(value), items_taken))
                else:
                    levels.append(_items_taken(self._view_level(value), items_taken))
        finally:
            self._scope = saved_scope
        result = min(levels, default=_NOT_NEW if newness else _NO_VIEW)
        self._target_levels[key] = result
        return result

    def _newness_table(self, scope: _Scope) -> dict[str, int]:
        """How new each local name of ``scope`` is.

        Every name starts at the newest level and is lowered to the oldest
        level any of its bindings gives, until nothing changes, so a name
        bound to itself (``values = values[:n]``) is as new as its other
        bindings make it. A binding to None is a placeholder that holds no
        array.
        """

        if scope.newness_table is not None:
            return scope.newness_table
        saved_scope, self._scope = self._scope, scope
        levels: dict[str, int] = {}
        candidates: dict[str, list[_Binding]] = {}
        for name, bindings in scope.local_bindings().items():
            kept = [
                (value, items_taken)
                for value, items_taken in bindings
                if not (items_taken == 0 and _is_none(value))
            ]
            if not kept or any(value is None for value, _items in kept):
                levels[name] = _NOT_NEW
            else:
                levels[name] = _NEW_ARRAY
                candidates[name] = kept

        def lookup(name: str) -> int:
            return levels.get(name, _NOT_NEW)

        changed = True
        try:
            while changed:
                changed = False
                for name, values in candidates.items():
                    if levels[name] == _NOT_NEW:
                        continue
                    level = min(
                        _items_newness(self._newness(value, lookup), items_taken)
                        for value, items_taken in values
                    )
                    if level < levels[name]:
                        levels[name] = level
                        changed = True
        finally:
            self._scope = saved_scope
        scope.newness_table = levels
        return levels

    def _call_newness(self, call: ast.Call, lookup: _NameLookup | None) -> int:
        if self._output_arguments(call):
            return _NOT_NEW
        name = self._resolve(call.func)
        function = call.func
        if name is not None:
            if name in _ALLOCATING_NUMPY_FUNCTIONS:
                return _NEW_ARRAY
            if isinstance(_library_object(name), np.ufunc):
                return _NEW_ARRAY
            if _is_ufunc_method(name, _ALLOCATING_UFUNC_METHODS):
                return _NEW_ARRAY
            if (
                name.startswith("numpy.random.")
                and name not in _IN_PLACE_NUMPY_FUNCTIONS
            ):
                return _NEW_ARRAY
            if name == "numpy.array":
                return _NOT_NEW if _copy_is_off(call) else _NEW_ARRAY
            if call.args and (
                name in _VIEW_NUMPY_FUNCTIONS
                or name in _SPLITTING_NUMPY_FUNCTIONS
                or name in _PANDAS_SHARING_NUMPY_FUNCTIONS
            ):
                # A view of a new array, or an array made from a new container.
                source = self._newness(call.args[0], lookup)
                return _NEW_ARRAY if source >= _NEW_OBJECT else _NOT_NEW
            return _NOT_NEW
        if isinstance(function, ast.Name) and function.id in _CONTAINER_BUILTINS:
            return _NEW_OBJECT
        if not isinstance(function, ast.Attribute):
            return _NOT_NEW
        method = function.attr
        if method == "to_numpy":
            copy = _keyword(call, "copy")
            copy_value = copy.value if copy is not None else None
            if copy_value is None and len(call.args) >= 2:
                copy_value = call.args[1]
            return (
                _NEW_ARRAY
                if copy_value is not None and _is_true_constant(copy_value)
                else _NOT_NEW
            )
        if method == "flatten" or method in _RANDOM_DRAW_METHODS:
            return _NEW_ARRAY
        if method in _VIEW_METHODS:
            receiver = self._newness(function.value, lookup)
            return _NEW_ARRAY if receiver == _NEW_ARRAY else _NOT_NEW
        if method == "copy":
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

    def _newness(self, expression: ast.expr, lookup: _NameLookup | None = None) -> int:
        """How new the value of ``expression`` is (``_NOT_NEW`` to ``_NEW_ARRAY``)."""

        if isinstance(expression, ast.Attribute):
            # ``.T`` and the other view attributes of a new array are new.
            if expression.attr in _VIEW_ATTRIBUTES:
                source = self._newness(expression.value, lookup)
                return _NEW_ARRAY if source == _NEW_ARRAY else _NOT_NEW
            return _NOT_NEW
        if isinstance(expression, ast.Subscript):
            container = self._newness(expression.value, lookup)
            if container == _NEW_ARRAY:
                return _NEW_ARRAY
            if container == _NEW_OBJECT and isinstance(expression.slice, ast.Slice):
                return _NEW_OBJECT
            return _NOT_NEW
        if isinstance(expression, ast.IfExp):
            return min(
                self._newness(expression.body, lookup),
                self._newness(expression.orelse, lookup),
            )
        if isinstance(expression, ast.BoolOp):
            return min(self._newness(value, lookup) for value in expression.values)
        if isinstance(expression, ast.Call):
            return self._call_newness(expression, lookup)
        if isinstance(expression, ast.BinOp):
            if isinstance(expression.op, _JOINING_OPERATORS):
                return _NEW_OBJECT
            return _NEW_ARRAY
        if isinstance(expression, _NEW_ARRAY_EXPRESSIONS):
            return _NEW_ARRAY
        if isinstance(expression, _CONTAINER_EXPRESSIONS):
            return _NEW_OBJECT
        if isinstance(expression, ast.Name):
            if lookup is not None:
                return lookup(expression.id)
            return self._name_newness(expression.id)
        return _NOT_NEW

    # Whether a value can be a writable view --------------------------------

    def _name_view_level(self, name: str) -> int:
        """Whether, and how deep, ``name`` holds a writable view where it is used."""

        scope = self._scope
        while isinstance(scope, _ComprehensionScope):
            sources = scope.target_sources(name)
            if sources is not None:
                return self._comprehension_target(scope, name, sources, newness=False)
            scope = scope.enclosing
        return self._view_table(scope).get(name, _NO_VIEW)

    def _view_table(self, scope: _Scope) -> dict[str, int]:
        """Whether, and how deep, each name of ``scope`` holds a writable view.

        Starts every name at ``_NO_VIEW`` and lowers it to the closest level
        any of its bindings or held values gives, until nothing changes. A
        binding that takes items of its value (a loop target, unpacking)
        takes one level per item; a container filled with a value holds it
        one level deeper.
        """

        if scope.view_table is not None:
            return scope.view_table
        saved_scope, self._scope = self._scope, scope
        sources: defaultdict[str, list[tuple[ast.expr, Callable[[int], int]]]] = (
            defaultdict(list)
        )
        for name, bindings in scope.local_bindings().items():
            for value, items_taken in bindings:
                if value is not None:
                    sources[name].append(
                        (value, functools.partial(_bound_level, items_taken))
                    )
        # A container filled here may be a parameter, so held values count
        # for any name.
        for name in scope.held_names():
            for value, items_taken, one_item in scope.held_values(name):
                sources[name].append(
                    (value, functools.partial(_held_level, items_taken, one_item))
                )
        levels: dict[str, int] = dict.fromkeys(sources, _NO_VIEW)

        def lookup(name: str) -> int:
            return levels.get(name, _NO_VIEW)

        changed = True
        try:
            while changed:
                changed = False
                for name, parts in sources.items():
                    if levels[name] == _IS_VIEW:
                        continue
                    level = min(
                        convert(self._view_level(value, lookup))
                        for value, convert in parts
                    )
                    if level < levels[name]:
                        levels[name] = level
                        changed = True
        finally:
            self._scope = saved_scope
        scope.view_table = levels
        return levels

    def _is_container(self, expression: ast.expr) -> bool:
        # A chain of joins (``[a] + [b] + rest``) is a container if any of
        # its operands is one.
        while isinstance(expression, ast.BinOp) and isinstance(
            expression.op, _VIEW_JOINING_OPERATORS
        ):
            if self._is_container(expression.right):
                return True
            expression = expression.left
        if isinstance(expression, _CONTAINER_EXPRESSIONS):
            return True
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            return expression.func.id in _CONTAINER_BUILTINS
        if isinstance(expression, ast.Name):
            bindings = self._bindings(expression.id) or []
            return bool(bindings) and all(
                value is not None
                and items_taken == 0
                and isinstance(value, _CONTAINER_EXPRESSIONS)
                for value, items_taken in bindings
            )
        return False

    def _reaches_writable_view(self, expression: ast.expr) -> bool:
        """Whether a write into ``expression`` can reach pandas' memory."""

        return self._view_level(expression) <= _ELEMENT

    def _view_level(
        self, expression: ast.expr, lookup: _NameLookup | None = None
    ) -> int:
        """Whether, and how deep in containers, ``expression`` holds a writable view."""

        if not self._has_writable_views:
            return _NO_VIEW
        if isinstance(expression, ast.Attribute):
            if expression.attr in _WRITABLE_VIEW_ATTRIBUTES:
                return _NO_VIEW if _is_own_object(expression.value) else _IS_VIEW
            if expression.attr == "base":
                # ``.base`` of an array expression is the array behind a view;
                # on other objects it is an ordinary attribute.
                array_expression = isinstance(expression.value, ast.Call) or (
                    isinstance(expression.value, ast.Attribute)
                    and expression.value.attr in ("values", "array")
                )
                return _IS_VIEW if array_expression else _NO_VIEW
            if expression.attr in _VIEW_ATTRIBUTES:
                return _view_of(self._view_level(expression.value, lookup))
            return _NO_VIEW
        if isinstance(expression, ast.Subscript):
            return _item_of(self._view_level(expression.value, lookup))
        if isinstance(expression, ast.Starred):
            return self._view_level(expression.value, lookup)
        if isinstance(expression, ast.Name):
            if lookup is not None:
                return lookup(expression.id)
            return self._name_view_level(expression.id)
        if isinstance(expression, ast.IfExp):
            return min(
                self._view_level(expression.body, lookup),
                self._view_level(expression.orelse, lookup),
            )
        if isinstance(expression, ast.BoolOp):
            return min(self._view_level(value, lookup) for value in expression.values)
        if isinstance(expression, ast.BinOp):
            return self._joined_view_level(expression, lookup)
        if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
            # A starred element puts its items into the new container.
            items = [
                _item_of(self._view_level(element.value, lookup))
                if isinstance(element, ast.Starred)
                else self._view_level(element, lookup)
                for element in expression.elts
            ]
            return _holder_of(min(items, default=_NO_VIEW))
        if isinstance(expression, ast.Dict):
            # ``**mapping`` puts the mapping's values into the new dict.
            items = [
                _item_of(self._view_level(value, lookup))
                if key is None
                else self._view_level(value, lookup)
                for key, value in zip(expression.keys, expression.values, strict=True)
            ]
            return _holder_of(min(items, default=_NO_VIEW))
        if isinstance(expression, _COMPREHENSIONS):
            return self._comprehension_view_level(expression, lookup)
        if isinstance(expression, ast.Call):
            return self._call_view_level(expression, lookup)
        return _NO_VIEW

    def _joined_view_level(
        self, expression: ast.BinOp, lookup: _NameLookup | None
    ) -> int:
        """``[a.array] + [b.array] + rest`` and ``mapping | other`` join
        containers; arithmetic, and any operator with an array operand, give
        new memory."""

        if not isinstance(expression.op, _VIEW_JOINING_OPERATORS):
            return _NO_VIEW
        operands: list[ast.expr] = []
        node: ast.expr = expression
        while isinstance(node, ast.BinOp) and isinstance(
            node.op, _VIEW_JOINING_OPERATORS
        ):
            operands.append(node.right)
            node = node.left
        operands.append(node)
        if not any(self._is_container(operand) for operand in operands):
            return _NO_VIEW
        level = min(self._view_level(operand, lookup) for operand in operands)
        return _NO_VIEW if level <= _ELEMENT else level

    def _comprehension_view_level(
        self, expression: ast.expr, lookup: _NameLookup | None
    ) -> int:
        """A comprehension holds what its element is, its loop targets taking
        items of their iterables."""

        targets: dict[str, int] = {}

        def inner(name: str) -> int:
            if name in targets:
                return targets[name]
            if lookup is not None:
                return lookup(name)
            return self._name_view_level(name)

        for generator in expression.generators:
            for target, value, items_taken in _loop_target_names(
                generator.target, generator.iter
            ):
                if isinstance(target, ast.Name):
                    level = (
                        _NO_VIEW
                        if value is None
                        else _items_taken(self._view_level(value, inner), items_taken)
                    )
                    targets[target.id] = min(targets.get(target.id, _NO_VIEW), level)
        element = (
            expression.value if isinstance(expression, ast.DictComp) else expression.elt
        )
        return _holder_of(self._view_level(element, inner))

    def _call_view_level(self, call: ast.Call, lookup: _NameLookup | None) -> int:
        arguments = [*call.args, *(keyword.value for keyword in call.keywords)]

        def nearest(expressions: Iterable[ast.expr]) -> int:
            return min(
                (self._view_level(expression, lookup) for expression in expressions),
                default=_NO_VIEW,
            )

        name = self._resolve(call.func)
        if name in _SPLITTING_NUMPY_FUNCTIONS:
            return _holder_of(_view_of(nearest(arguments)))
        if name in _VIEW_NUMPY_FUNCTIONS:
            copy = _keyword(call, "copy")
            if copy is not None and _is_true_constant(copy.value):
                return _NO_VIEW
            return _view_of(nearest(arguments))
        if name in ("numpy.array", "numpy.astype") and _copy_is_off(call):
            return _view_of(nearest(arguments))
        function = call.func
        if isinstance(function, ast.Name):
            if function.id == "next":
                return _item_of(nearest(arguments))
            if function.id == "dict":
                # ``dict(mapping)`` holds the mapping's values, ``dict(pairs)``
                # the pairs' second items, and ``dict(key=value)`` each value.
                positional = nearest(call.args)
                from_items = (
                    _NO_VIEW
                    if positional <= _ELEMENT
                    else min(positional, _holder_of(_items_taken(positional, 2)))
                )
                return min(
                    from_items,
                    _holder_of(nearest(keyword.value for keyword in call.keywords)),
                )
            if function.id in _SAME_ITEMS_BUILTINS:
                return _items_holder_of(nearest(arguments))
            if function.id in _TUPLE_YIELDING_BUILTINS:
                # A container of tuples of the arguments' items.
                return _holder_of(_items_holder_of(nearest(arguments)))
            return _NO_VIEW
        if not isinstance(function, ast.Attribute):
            return _NO_VIEW
        method = function.attr
        receiver = self._view_level(function.value, lookup)
        if method in _VIEW_METHODS:
            copy = _keyword(call, "copy")
            copy_value = copy.value if copy is not None else None
            if copy_value is None and method == "to_numpy" and len(call.args) >= 2:
                copy_value = call.args[1]
            if copy_value is not None and _is_true_constant(copy_value):
                return _NO_VIEW
            return _view_of(receiver)
        if method == "astype" and _copy_is_off(call):
            return _view_of(receiver)
        if method == "copy":
            # A copy of an array is new memory; a copy of a container holds
            # the same items.
            return _NO_VIEW if receiver <= _ELEMENT else receiver
        if method in _ITEM_TAKING_METHODS:
            # ``get(key, default)`` may also return its default.
            return min(_item_of(receiver), nearest(call.args[1:]))
        if method in _SAME_ITEMS_METHODS:
            return _items_holder_of(receiver)
        if method in _TUPLE_YIELDING_METHODS:
            return _holder_of(_items_holder_of(receiver))
        return _NO_VIEW

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
        # ``out=`` means an output array only to the libraries and to array
        # and random Generator methods; workspace functions use it for paths.
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
        if _is_ufunc_method(name, ("at",)):
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
        # ``values += 1`` writes into the array ``values`` names. An element
        # of a view may be a scalar, which ``+=`` rebinds, so only a view
        # itself counts; ``holder[key] += 1`` writes into a view the holder
        # holds.
        target = node.target
        if isinstance(target, ast.Name):
            if self._view_level(
                target
            ) == _IS_VIEW and not self._bound_only_to_extension_arrays(target.id):
                self._report(node, _WRITES_THROUGH_WRITABLE_VIEW)
        elif isinstance(target, ast.Subscript):
            container = self._view_level(target.value)
            if container <= _ELEMENT or _item_of(container) == _IS_VIEW:
                self._report(node, _WRITES_THROUGH_WRITABLE_VIEW)

    def _bound_only_to_extension_arrays(self, name: str) -> bool:
        """Whether every binding of ``name`` is ``x.array`` itself (or a slice).

        That is a pandas extension array, for which ``+=`` builds a new
        array and rebinds the name instead of writing, except for datetime
        data.
        """

        bindings = self._bindings(name) or []

        def is_extension_array(value: ast.expr) -> bool:
            while isinstance(value, ast.Subscript):
                value = value.value
            return (
                isinstance(value, ast.Attribute)
                and value.attr == "array"
                and not _is_own_object(value.value)
            )

        return bool(bindings) and all(
            items_taken == 0 and value is not None and is_extension_array(value)
            for value, items_taken in bindings
        )

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
    # Round-four cases.
    "series_array_through_dict_values": (
        _in_function(
            "by_name = {name: frame[name].array for name in names}",
            "for values in by_name.values():",
            "    values[0] = 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_dict_item_pairs": (
        _in_function(
            "by_name = {name: frame[name].array for name in names}",
            "for pair in by_name.items():",
            "    pair[1][0] = 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_comprehension_target_named_like_its_source": (
        _in_function(
            "values = [frame[name].array for name in names]",
            "[values.fill(0) for values in values]",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_in_a_three_way_join": (
        _in_function(
            "pair = [first.array] + [second.array] + extra",
            "pair[0][0] = 1",
            parameters="first, second, extra",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_in_a_long_join": (
        _in_function(
            "parts = " + " + ".join(["[series.array]"] * 600), "parts[0][0] = 1"
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_dict_of_zip": (
        _in_function(
            "arrays = [frame[name].array for name in names]",
            "by_name = dict(zip(names, arrays))",
            'by_name["income"][0] = 1',
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_dict_keyword": (
        _in_function(
            'by_name = dict(income=frame["income"].array)',
            'by_name["income"][0] = 1',
            parameters="frame",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_enumerate_of_zip": (
        _in_function(
            "arrays = [frame[name].array for name in names]",
            "for position, (name, values) in enumerate(zip(names, arrays)):",
            "    values[0] = position",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "out_loop_target_over_a_list_of_existing_arrays": (
        _in_function(
            "for buffer in [series.to_numpy()]:",
            "    np.add(buffer, 1, out=buffer)",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    # Round-three cases.
    "series_array_through_a_dict_comprehension_of_loop_targets": (
        _in_function(
            "arrays = [frame[name].array for name in names]",
            "by_name = {name: values for name, values in zip(names, arrays)}",
            'by_name["income"][0] = 1',
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_an_identity_comprehension": (
        _in_function(
            "arrays = [frame[name].array for name in names]",
            "[values for values in arrays][0][0] = 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_filled_in_a_nested_comprehension": (
        _in_function(
            "groups = [[frame[name].array for name in names]]",
            "[values.fill(0) for group in groups for values in group]",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_next": (
        _in_function(
            "arrays = [frame[name].array for name in names]",
            "next(values for values in arrays)[0] = 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_augmented_through_a_loop_target": (
        _in_function(
            "arrays = [np.asarray(frame[name].array) for name in names]",
            "for values in arrays:",
            "    values += 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_augmented_through_an_item": (
        _in_function(
            "arrays = [np.asarray(frame[name].array) for name in names]",
            "arrays[0] += 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_augmented_after_split": (
        _in_function(
            "left, right = np.split(np.asarray(series.array), 2)", "left += 1"
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_starred_display": (
        _in_function(
            "arrays = [frame[name].array for name in names]",
            "[*arrays, other][0][0] = 1",
            parameters="frame, names, other",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_dict_unpacking": (
        _in_function(
            "by_name = {name: frame[name].array for name in names}",
            '{**by_name}["income"][0] = 1',
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_dict_union": (
        _in_function(
            "by_name = {name: frame[name].array for name in names}",
            '(by_name | {})["income"][0] = 1',
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_get_default": (
        _in_function(
            'values = overrides.get("income", frame["income"].array)',
            "values[0] = 1",
            parameters="frame, overrides",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_unpacked_into_subscripts": (
        _in_function(
            "columns = {}",
            'columns["a"], columns["b"] = np.split(np.asarray(series.array), 2)',
            'columns["a"][0] = 1',
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_real_if_close": (
        _NUMPY + "np.real_if_close(np.asarray(series.array))[0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "out_item_of_a_dict_union": (
        _in_function(
            "merged = by_name | {}",
            'np.add(merged["x"], 1, out=merged["x"])',
            parameters="by_name",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_take_of_a_frame": (
        _in_function(
            "np.add.at(np.take(frame, [0], axis=1), ([0], [0]), 100)",
            parameters="frame",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    # Round-two cases: views reached through iteration and filled containers.
    "series_array_through_zip": (
        _in_function(
            "arrays = [frame[name].array for name in names]",
            "for name, values in zip(names, arrays):",
            "    values[0] = 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_enumerate": (
        _in_function(
            "arrays = [frame[name].array for name in names]",
            "for position, values in enumerate(arrays):",
            "    values[0] = position",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_dict_items": (
        _in_function(
            "arrays = {name: frame[name].array for name in names}",
            "for name, values in arrays.items():",
            "    values[0] = 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_through_dict_get": (
        _in_function(
            "arrays = {name: frame[name].array for name in names}",
            'arrays.get("income")[0] = 1',
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_appended": (
        _in_function(
            "arrays = []",
            "for name in names:",
            "    arrays.append(frame[name].array)",
            "arrays[0][0] = 1",
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_appended_to_a_parameter": (
        _in_function(
            "arrays.append(series.array)",
            "arrays[0][0] = 1",
            parameters="series, arrays",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_stored_by_key": (
        _in_function(
            "columns = {}",
            "for name in names:",
            "    columns[name] = frame[name].array",
            'columns["income"][0] = 1',
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_added_with_augmented_assignment": (
        _in_function(
            "arrays = []",
            "arrays += [series.array]",
            "arrays[0][0] = 1",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_in_joined_lists": (
        _in_function(
            "pair = [first.array] + [second.array]",
            "pair[0][0] = 1",
            parameters="first, second",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_split": (
        _in_function(
            "parts = np.split(np.asarray(series.array), 2)", "parts[0][0] = 1"
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_broadcast": (
        _in_function(
            "left, right = np.broadcast_arrays(np.asarray(series.array), 0)",
            "left[0] = 1",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_required": (
        _NUMPY + "np.require(series.array)[0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_swapped_axes_method": (
        _NUMPY + "np.asarray(series.array).swapaxes(0, 0)[0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "series_array_matrix_transpose_attribute": (
        _NUMPY + "np.atleast_2d(series.array).mT[0, 0] = 1\n",
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    # Round-two cases: arrays that are not new.
    "ufunc_at_on_item_of_joined_new_list": (
        _in_function(
            "columns = [frame[name].to_numpy() for name in names] + []",
            "np.add.at(columns[0], [0], 1)",
            parameters="frame, names",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_clip_of_a_series": (
        _in_function(
            'np.add.at(np.clip(frame["f"], 0, None), positions, weights)',
            parameters="frame, positions, weights",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "ufunc_at_on_tile_of_a_series": (
        _in_function("np.add.at(np.tile(series, 1), [0], 1)"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_round_of_a_parameter": (
        _in_function(
            "rounded = np.round(values)",
            "np.add(rounded, 1, out=rounded)",
            parameters="values",
        ),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_scipy_ufunc_keyword": (
        "import scipy.special\n"
        + "def change(values):\n"
        + "    scipy.special.expit(values, out=values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "out_scipy_ufunc_positional": (
        "from scipy.special import expit\n"
        + "def change(values):\n"
        + "    expit(values, values)\n",
        _WRITES_INTO_EXISTING_ARRAY,
    ),
    "series_array_in_a_shallow_dict_copy": (
        _in_function(
            "arrays = {name: frame[name].array for name in names}",
            "copied = arrays.copy()",
            'copied["income"][0] = 1',
            parameters="frame, names",
        ),
        _WRITES_THROUGH_WRITABLE_VIEW,
    ),
    "numpy_random_mtrand_shuffle": (
        _in_function("np.random.mtrand.shuffle(values)", parameters="values"),
        _WRITES_INTO_EXISTING_ARRAY,
    ),
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
    # Round-four cases.
    "zip_rows_paired_inside_enumerate": _in_function(
        'for position, (row, value) in enumerate(zip(matrix, frame["x"].array)):',
        "    row[:] = value",
        parameters="matrix, frame",
    ),
    "zip_rows_paired_inside_list": _in_function(
        'for row, value in list(zip(matrix, frame["x"].array)):',
        "    row[:] = value",
        parameters="matrix, frame",
    ),
    "comprehension_target_over_a_new_array": _in_function(
        "grid = np.zeros((3, 3))",
        "[np.add(row, 1, out=row) for row in grid]",
        parameters="",
    ),
    "pieces_of_a_new_array": _in_function(
        "buffer = np.zeros(size)",
        "for chunk in np.array_split(buffer, parts):",
        "    np.multiply(chunk, 2, out=chunk)",
        parameters="size, parts",
    ),
    "extension_array_name_augmented": _in_function(
        'total = frame["a"].array',
        'total += frame["b"].array',
        parameters="frame",
    ),
    "scalar_of_a_view_augmented": _in_function(
        'for value in frame["x"].array:',
        "    value -= 1",
        parameters="frame",
    ),
    "clip_of_a_view_written": _in_function(
        'clipped = np.clip(frame["x"].array, 0, None)',
        "clipped[clipped > 10] = 10",
        parameters="frame",
    ),
    "deep_chain_of_comprehension_generators": _in_function(
        "chain_0 = series.array",
        "["
        + "chain_24.fill(0) "
        + " ".join(
            f"for chain_{index} in zip(chain_{index - 1}, chain_{index - 1})"
            for index in range(1, 25)
        )
        + "]",
    ),
    # Round-three cases.
    "accumulating_scalars_of_views_into_a_dict": (
        "from collections import defaultdict\n"
        + _in_function(
            "totals = defaultdict(float)",
            'for group, weight in zip(frame["group"].array, frame["weight"].array):',
            "    totals[group] += weight",
            parameters="frame",
        )
    ),
    "running_total_of_view_columns": _in_function(
        "total = np.zeros(size)",
        "for column in columns:",
        "    total += frame[column].array",
        "total[total < 0] = 0",
        "np.maximum(total, 0, out=total)",
        parameters="frame, columns, size",
    ),
    "zip_pairs_rows_of_a_parameter_with_a_view": _in_function(
        'for row, value in zip(matrix, frame["x"].array):',
        "    row[:] = value",
        parameters="matrix, frame",
    ),
    "views_and_items_of_new_arrays": _in_function(
        "grid = np.zeros((3, 3)).T",
        "np.add(grid, 1, out=grid)",
        "matrix = np.ones((3, 3))",
        "np.fill_diagonal(matrix.T, 0)",
        "for row in np.zeros((3, 3)):",
        "    np.add(row, 1, out=row)",
        "first, second = np.zeros((2, 3))",
        "np.add(first, 1, out=first)",
        parameters="",
    ),
    "copies_given_by_keyword_and_position": _in_function(
        "values = np.asarray(series.array, copy=True)",
        "values[0] = 1",
        "other = series.array.to_numpy(float, True)",
        "other[0] = 1",
    ),
    "starred_rest_replaced": _in_function(
        "arrays = [frame[name].array for name in names]",
        "first, *rest = arrays",
        "rest[0] = np.zeros(3)",
        parameters="frame, names",
    ),
    "view_times_a_list": _in_function(
        "weights = [1.0, 2.0]",
        "scaled = series.array * weights",
        "scaled[0] = 0",
    ),
    "ordinary_base_attribute": _in_function(
        "columns = holder.base", 'columns["x"] = 1', parameters="holder"
    ),
    # Round-two cases.
    "long_chain_of_views_of_a_new_array": _in_function(
        "chain_0 = np.zeros(64)",
        *(
            f"chain_{index} = chain_{index - 1}[1:] if flag else chain_{index - 1}[:-1]"
            for index in range(1, 40)
        ),
        "np.add(chain_39, 1, out=chain_39)",
        parameters="flag",
    ),
    "generator_draws": _in_function(
        "draws = rng.normal(size=size)",
        "np.multiply(draws, 2, out=draws)",
        "sample = np.random.default_rng(seed).random(size)",
        "np.add(sample, 1, out=sample)",
        parameters="rng, seed, size",
    ),
    "flattened_reshaped_and_listed_new_arrays": _in_function(
        "flat = values.flatten()",
        "np.add(flat, 1, out=flat)",
        "grid = np.empty(6).reshape(2, 3)",
        "np.add(grid, 1, out=grid)",
        "listed = np.asarray([1.0, 2.0])",
        "np.add(listed, 1, out=listed)",
        parameters="values",
    ),
    "ufunc_reductions_and_interpolation": _in_function(
        "peaks = np.maximum.accumulate(values)",
        "np.add(peaks, 1, out=peaks)",
        "curve = np.interp(points, xp, fp)",
        "np.add(curve, 1, out=curve)",
        parameters="values, points, xp, fp",
    ),
    "positional_copy_to_numpy": _in_function(
        "values = series.to_numpy(float, True)", "np.add(values, 1, out=values)"
    ),
    "copy_of_a_view_written": _in_function(
        "encoded = series.array.asi8.copy()", "encoded[0] = 0"
    ),
    "list_of_a_view_reassigned": _in_function(
        "items = list(series.array)", "items[0] = 1"
    ),
    "frame_holding_a_view_written_by_column": _in_function(
        "carried = table.copy()",
        "carried['id'] = replacement.array",
        "tables = {}",
        "tables['person'] = carried",
        "tables['person']['income'] = aligned.array",
        parameters="table, replacement, aligned",
    ),
    "arithmetic_on_a_view_written": _in_function(
        "doubled = np.asarray(series.array) * 2", "doubled[0] = 0"
    ),
    "scipy_ufunc_into_new_array": (
        "import numpy as np\n"
        + "import scipy.special\n"
        + "def change(values):\n"
        + "    result = np.empty(3)\n"
        + "    scipy.special.expit(values, out=result)\n"
    ),
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
    "random_results": _in_function(
        "order = np.random.permutation(size)",
        "np.add(order, 1, out=order)",
        "chosen = np.random.choice(values, 3)",
        "np.add(chosen, 1, out=chosen)",
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
