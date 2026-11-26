"""Describe a pandas table's block layout and the memory behind its columns.

Shared by the tests of :mod:`microcosm.frame.table_copies` and of the UK
kernels that copy their input tables with it. These read pandas internals
(``_mgr`` and the arrays behind extension columns), which the source tree's
write scan refuses outside tests.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

#: Attributes of pandas arrays that hold a column's NumPy memory.
_ARRAY_ATTRIBUTES = ("_ndarray", "_data", "_mask", "_codes")


def block_layout(table: pd.DataFrame) -> list[tuple[object, ...]]:
    """Each pandas block of ``table``, in pandas' order of its blocks.

    An entry holds the block's dtype, the column positions it holds in its
    order, and for each NumPy array behind it the array's shape, strides and
    whether it is C-contiguous. Equal entries put each column at the same
    offset in an equally shaped block.
    """

    return [
        (
            str(block.dtype),
            tuple(int(position) for position in block.mgr_locs.as_array),
            _array_layouts(block.values),
        )
        for block in table._mgr.blocks
    ]


def _array_layouts(values: object) -> tuple[object, ...]:
    if isinstance(values, np.ndarray):
        return (("ndarray", values.shape, values.strides, values.flags.c_contiguous),)
    return (
        type(values).__name__,
        *(
            (name, array.shape, array.strides, array.flags.c_contiguous)
            for name in _ARRAY_ATTRIBUTES
            if isinstance(array := getattr(values, name, None), np.ndarray)
        ),
    )


def column_arrays(column: pd.Series) -> list[np.ndarray]:
    """The NumPy arrays holding one column's values, its mask and its categories.

    pyarrow-backed columns hold immutable buffers and contribute none.
    """

    values = column.array
    if isinstance(column.dtype, np.dtype):
        return [np.asarray(values)]
    arrays = [
        array
        for name in _ARRAY_ATTRIBUTES
        if isinstance(array := getattr(values, name, None), np.ndarray)
    ]
    if isinstance(values, pd.Categorical):
        categories = values.categories.array
        arrays.extend(
            array
            for name in _ARRAY_ATTRIBUTES
            if isinstance(array := getattr(categories, name, None), np.ndarray)
        )
    return arrays


def shares_memory(first: pd.DataFrame, second: pd.DataFrame) -> list[str]:
    """Columns of ``first`` holding an array that shares memory with ``second``."""

    other = [array for _, column in second.items() for array in column_arrays(column)]
    return [
        str(name)
        for name, column in first.items()
        if any(
            np.shares_memory(array, other_array)
            for array in column_arrays(column)
            for other_array in other
        )
    ]


def memory_digest(table: pd.DataFrame | pd.Series) -> str:
    """A digest of the values, the raw bytes and the labels behind ``table``.

    Values are hashed through pandas (so a categorical column's categories
    count), and every NumPy array behind each column is hashed byte for byte
    (so a write behind a missing value's mask counts too).
    """

    digest = hashlib.sha256()
    frame = table.to_frame() if isinstance(table, pd.Series) else table
    digest.update(repr(list(frame.columns)).encode())
    digest.update(repr((frame.index.names, frame.columns.names)).encode())
    digest.update(
        np.asarray(frame.index).tobytes()
        if frame.index.dtype.kind in "biuf"
        else repr(list(frame.index)).encode()
    )
    for name, column in frame.items():
        digest.update(str(name).encode() + b"\0")
        digest.update(
            pd.util.hash_pandas_object(column, index=False).to_numpy().tobytes()
        )
        if isinstance(column.dtype, pd.CategoricalDtype):
            digest.update(repr(list(column.cat.categories)).encode())
        for array in column_arrays(column):
            digest.update(
                np.ascontiguousarray(array).view(np.uint8).tobytes()
                if array.dtype != object
                else repr(array.tolist()).encode()
            )
    return digest.hexdigest()
