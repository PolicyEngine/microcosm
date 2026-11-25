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

The result matches ``table.copy(deep=True)`` block for block when each block of
``table`` lists its columns in position order, as the blocks of tables built by
pandas' own operations and the executor's inputs do: the same blocks with the
same dtype, columns, shape and strides, so each column lies at the same offset
within its block, and BLAS products such as ``np.dot``, whose rounding on some
platforms depends on how an operand is aligned, see the same layout. The values,
index and column labels, ``attrs`` and flags are equal. Three things differ, and
each leaves the copy further from its source than pandas' copy does:

- A categorical column's categories are copied too, where pandas' copy shares
  them with the source, so a write into them (``.cat.categories.values[0] =``,
  for example) changes only the copy. The codes keep their integer type.
- The index and the column labels are copied, where pandas' copy views them.
  ``np.<ufunc>.at`` writes even into a read-only array, so a view of the
  executor's read-only index would let it reach the context, and through a
  node that reads every row the stored population. A ``RangeIndex``, as an
  index, a level of a ``MultiIndex`` or categories, is rebuilt, because even a
  deep copy of one shares the cache holding its values once materialized, and
  ``np.asarray`` returns that cached array writable; a ``CategoricalIndex``
  gets copied categories. This costs one array per table whose index is not a
  ``RangeIndex``.
- pandas' internal list of blocks is in the order its consolidation gives
  (one-dimensional extension blocks, then NumPy blocks, each by dtype name and
  then column position). That is the order of ``copy(deep=True)`` whenever
  pandas has to merge the blocks of a table that holds them in column order, as
  for a kernel input with two or more columns of one NumPy dtype, and may differ
  otherwise. Nothing reads a table block by block in that order: values, layout
  and every column's memory are the same.

A pyarrow-backed column is copied as pandas copies it, sharing its Arrow
buffers with the source. pyarrow treats them as immutable, but ``np.<ufunc>.at``
writes through the read-only array ``to_numpy()`` returns for a numeric one.
The graph store accepts no pyarrow-backed column, so no kernel input holds one.

Some tables are copied with ``copy(deep=True)``, and then their categories and
labels are copied as above, so only the third difference does not apply:

- a pandas subclass, whose type and metadata only its own methods keep;
- a table with a ``datetime64``, ``timedelta64``, timezone-aware datetime or
  period column, whose blocks pandas builds from extension arrays in two
  dimensions; on pandas 3.0.3 the one-copy build would give the same blocks,
  but this module leaves them to pandas rather than depend on how a pandas
  version merges them;
- a table with two NumPy dtypes of one name, which pandas merges into one
  block;
- any table under a pandas without
  ``pandas.api.internals.create_dataframe_from_blocks`` (added in pandas 3.0).
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

    The copy shares no memory with ``table`` other than a pyarrow-backed
    column's buffers. See the module docstring for what matches pandas' copy
    and the three ways the copy differs.

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
        blocks, index=_copied_index(table.index), columns=_copied_index(table.columns)
    )
    return copied.__finalize__(table, method="copy")


def copy_series(series: pd.Series) -> pd.Series:
    """``series.copy(deep=True)``, with its categories and index copied.

    The same two differences from pandas' copy as :func:`copy_table` makes, for
    the same reasons. A series of a type :func:`copy_table` leaves to pandas is
    copied with ``copy(deep=True)``, and then its categories and index copied.

    Args:
        series: The series to copy. It is only read.

    Returns:
        A new series that shares no memory with ``series`` other than
        pyarrow buffers.
    """

    if type(series) is not pd.Series:
        return _pandas_copy(series)
    dtype = series.dtype
    if (isinstance(dtype, np.dtype) and dtype.kind not in _BLOCK_KINDS) or isinstance(
        dtype, _TWO_DIMENSIONAL_EXTENSION_DTYPES
    ):
        return _pandas_copy(series)
    values = _copied_values(series)
    copied = pd.Series(
        values,
        index=_copied_index(series.index),
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
        return _copied_categorical(values)
    return values.copy()


def _copied_categorical(values: pd.Categorical) -> pd.Categorical:
    """A copy of ``values`` with its categories copied too.

    ``rename_categories`` copies the codes in their own integer type and keeps
    the categories it is given; ``from_codes`` would narrow the codes.
    """

    return values.rename_categories(_copied_index(values.categories))


def _pandas_copy(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """``data.copy(deep=True)``, with its categories and labels copied as above."""

    copied = data.copy(deep=True)
    if isinstance(copied, pd.DataFrame):
        for position, (_, column) in enumerate(copied.items()):
            if isinstance(column.dtype, pd.CategoricalDtype):
                copied.isetitem(position, _copied_values(column))
        copied.columns = _copied_index(data.columns)
    elif isinstance(copied.dtype, pd.CategoricalDtype):
        # The series' own constructor keeps a subclass's type, and
        # __finalize__ its metadata, as copy(deep=True) does.
        copied = data._constructor(
            _copied_values(data), index=copied.index, name=copied.name, copy=False
        ).__finalize__(data, method="copy")
    copied.index = _copied_index(data.index)
    return copied


def _copied_index(index: pd.Index) -> pd.Index:
    """An index equal to ``index`` that shares no memory with it.

    ``copy(deep=True)`` would do, except that it keeps a ``RangeIndex``'s cache
    of materialized values, also as a level of a ``MultiIndex``, and a
    ``CategoricalIndex``'s categories; those are rebuilt.
    """

    if type(index) is pd.RangeIndex:
        return pd.RangeIndex(index.start, index.stop, index.step, name=index.name)
    if type(index) is pd.MultiIndex:
        return pd.MultiIndex(
            levels=[_copied_index(level) for level in index.levels],
            codes=[np.array(codes, copy=True) for codes in index.codes],
            names=index.names,
            sortorder=index.sortorder,
            verify_integrity=False,
        )
    if type(index) is pd.CategoricalIndex:
        return pd.CategoricalIndex(_copied_categorical(index.array), name=index.name)
    return index.copy(deep=True)
