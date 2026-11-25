"""Deep copies of tables that copy each column once.

``DataFrame.copy(deep=True)`` copies each pandas block of a table and then, when
a NumPy dtype is held in more than one block, stacks the copies of that dtype
into one new block and reorders its rows into column order, which makes two
further copies. The graph executor hands each kernel tables that hold one block
per column (amendment 31), so a kernel that deep-copies its input tables briefly
holds about three copies of them. :func:`copy_table` builds the same result with
each column copied once, straight into its row of the new block, and the caller
can pass the copies to :meth:`~microcosm.frame.bundle.Frame.adopt`, which stores
them without copying again.

The result matches ``table.copy(deep=True)`` block for block: the same blocks
with the same dtype, columns, shape and strides, so each column lies at the same
offset within its block, and BLAS products such as ``np.dot``, whose rounding on
some platforms depends on how an operand is aligned, see the same layout. The
values, index and column labels, ``attrs`` and flags are equal. Three things
differ, and each leaves the copy further from its source than pandas' copy does:

- a categorical column's categories are copied too, where pandas' copy shares
  them with the source, so a write into them (``.cat.categories.values[0] =``,
  for example) changes only the copy;
- a ``RangeIndex`` is rebuilt, because a view of one shares the cache that holds
  its values once materialized, and ``np.asarray(index)`` returns that cached
  array writable; other indexes are views of the source's, as in pandas' copy;
- the order of pandas' internal list of blocks follows the order its
  consolidation gives (one-dimensional extension blocks, then NumPy blocks, each
  by dtype name), which is the order of ``copy(deep=True)`` for a table holding
  one block per column and may differ for others. Nothing reads a table block by
  block in that order: values, layout and every column's memory are the same.

A table this module cannot build block for block is copied with
``copy(deep=True)``, and then its categorical columns' categories are copied and
a ``RangeIndex`` is rebuilt, so only the third difference does not apply. These
are a pandas subclass, whose type and metadata only its own methods keep; a
table with a column whose block pandas stores in two dimensions but builds from
an extension array (``datetime64``, ``timedelta64``, timezone-aware datetimes
and periods); a table with two NumPy dtypes of one name; and any table under a
pandas without ``pandas.api.internals.create_dataframe_from_blocks`` (added in
pandas 3.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

try:
    from pandas.api.internals import create_dataframe_from_blocks
except ImportError:  # pandas before 3.0
    create_dataframe_from_blocks = None

__all__ = ["copy_series", "copy_table"]

#: NumPy dtype kinds pandas stores in one two-dimensional block per dtype:
#: boolean, signed and unsigned integer, float, complex and object.
_BLOCK_KINDS = frozenset("biufcO")
#: Extension dtypes pandas stores in two-dimensional blocks that a table can
#: hold several columns of.
_TWO_DIMENSIONAL_EXTENSION_DTYPES = (pd.DatetimeTZDtype, pd.PeriodDtype)


def copy_table(table: pd.DataFrame) -> pd.DataFrame:
    """``table.copy(deep=True)``, built with one copy of each column.

    The copy shares no column memory with ``table``. See the module docstring
    for what matches pandas' copy and the three ways the copy differs.

    Args:
        table: The table to copy. It is only read.

    Returns:
        A new table that no other object refers to, so a pandas write into it
        is made in place, as into the result of ``copy(deep=True)``.
    """

    if create_dataframe_from_blocks is None or type(table) is not pd.DataFrame:
        return _pandas_copy(table)
    numpy_positions: dict[np.dtype, list[int]] = {}
    extension_blocks: list[tuple[str, ExtensionArray, int]] = []
    columns = list(table.items())
    for position, (_, column) in enumerate(columns):
        dtype = column.dtype
        if isinstance(dtype, np.dtype):
            if dtype.kind not in _BLOCK_KINDS:
                return _pandas_copy(table)
            numpy_positions.setdefault(dtype, []).append(position)
        elif isinstance(dtype, ExtensionDtype) and not isinstance(
            dtype, _TWO_DIMENSIONAL_EXTENSION_DTYPES
        ):
            extension_blocks.append((dtype.name, _copied_values(column), position))
        else:
            return _pandas_copy(table)
    if len({dtype.name for dtype in numpy_positions}) != len(numpy_positions):
        return _pandas_copy(table)

    blocks: list[tuple[np.ndarray | ExtensionArray, np.ndarray]] = [
        (values, np.array([position], dtype=np.intp))
        for _, values, position in sorted(
            extension_blocks, key=lambda block: (block[0], block[2])
        )
    ]
    for dtype in sorted(numpy_positions, key=lambda dtype: dtype.name):
        positions = numpy_positions[dtype]
        values = np.empty((len(positions), len(table)), dtype=dtype)
        for row, position in enumerate(positions):
            values[row] = columns[position][1].to_numpy()
        blocks.append((values, np.array(positions, dtype=np.intp)))
    copied = create_dataframe_from_blocks(
        blocks, index=_index_of_copy(table.index), columns=table.columns.view()
    )
    return copied.__finalize__(table, method="copy")


def copy_series(series: pd.Series) -> pd.Series:
    """``series.copy(deep=True)``, with categories copied and a new ``RangeIndex``.

    The same two differences from pandas' copy as :func:`copy_table` makes, for
    the same reasons. A series of a type :func:`copy_table` leaves to pandas is
    copied with ``copy(deep=True)``, and its ``RangeIndex`` rebuilt.

    Args:
        series: The series to copy. It is only read.

    Returns:
        A new series that shares no memory with ``series``.
    """

    dtype = series.dtype
    if (
        type(series) is not pd.Series
        or (isinstance(dtype, np.dtype) and dtype.kind not in _BLOCK_KINDS)
        or isinstance(dtype, _TWO_DIMENSIONAL_EXTENSION_DTYPES)
    ):
        return _pandas_copy(series)
    values = _copied_values(series)
    copied = pd.Series(
        values,
        index=_index_of_copy(series.index),
        name=series.name,
        dtype=values.dtype,
        copy=False,
    )
    return copied.__finalize__(series, method="copy")


def _copied_values(column: pd.Series) -> np.ndarray | ExtensionArray:
    """A copy of one column's values, as its block's own copy makes it."""

    if isinstance(column.dtype, np.dtype):
        return column.to_numpy().copy()
    values = column.array
    if isinstance(values, pd.Categorical):
        dtype = pd.CategoricalDtype(
            values.categories.copy(deep=True), ordered=values.ordered
        )
        return pd.Categorical.from_codes(
            values.codes.copy(), dtype=dtype, validate=False
        )
    return values.copy()


def _pandas_copy(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """``data.copy(deep=True)``, with categories copied and a new ``RangeIndex``."""

    copied = data.copy(deep=True)
    if isinstance(copied, pd.DataFrame):
        for position, (_, column) in enumerate(copied.items()):
            if isinstance(column.dtype, pd.CategoricalDtype):
                copied.isetitem(position, _copied_values(column))
    if type(data.index) is pd.RangeIndex:
        copied.index = _index_of_copy(data.index)
    return copied


def _index_of_copy(index: pd.Index) -> pd.Index:
    """The index a copy carries: a view of ``index``, or a new ``RangeIndex``."""

    if type(index) is pd.RangeIndex:
        return pd.RangeIndex(index.start, index.stop, index.step, name=index.name)
    return index.view()
