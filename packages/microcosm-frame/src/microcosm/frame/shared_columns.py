"""Find in-place changes to column memory that frames share.

Under pandas copy-on-write, frame operations such as
:meth:`~microcosm.frame.bundle.Frame.with_weights` share the memory of unchanged
columns. A write made through pandas stays on one frame, but a raw write into
the memory, for example through ``np.add.at`` or the array behind a nullable
column, changes every frame sharing it. This module detects that afterwards. It
does not prevent the write or identify the code that made it.

**Recording.** When an operation starts sharing a frame's columns,
:func:`record_shared_columns` stores a SHA-256 digest of each array
:func:`_column_buffers` lists for the column. A digest belongs to the array's
memory, not to a frame, so a column is hashed once however many frames go on to
share it. Columns that are never shared are never hashed. Digests live only in
this process and are dropped when the array that owns the memory is freed.

**Comparing.** :func:`check_shared_columns` hashes each recorded array again and
raises :class:`SharedColumnChangedError` naming every column that differs. The
:class:`~microcosm.frame.bundle.Frame` constructor, ``engine_tables`` and the
writers of checkpoints and outputs call it before copying or writing.

A column is compared only while it is still stored in the memory that was
recorded. Replacing, adding or dropping a column through pandas moves it to
other memory, and so does a pandas write while the memory is still shared, so
none of these trip the check. Once no other table refers to the memory, an
in-place pandas write such as ``.loc``, ``fillna(inplace=True)`` or ``update``
changes the recorded memory and is reported, because frame tables are documented
as read-only.

Not covered:

- pyarrow-backed columns, whose buffers are immutable, and any other extension
  type :func:`_column_buffers` does not list;
- a copy taken after the change, for example ``table.copy()`` or a ``select``
  that drops rows, which carries the changed values into new memory.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Iterable, Iterator, Mapping
from functools import partial
from typing import NamedTuple

import numpy as np
import pandas as pd

_NULLABLE_ARRAY_TYPES = (
    pd.arrays.BooleanArray,
    pd.arrays.FloatingArray,
    pd.arrays.IntegerArray,
)
_DATETIME_LIKE_ARRAY_TYPES = (
    pd.arrays.DatetimeArray,
    pd.arrays.PeriodArray,
    pd.arrays.TimedeltaArray,
)

_BufferKey = tuple[int, int, int, str, int]


class SharedColumnChangedError(RuntimeError):
    """Memory that frames shared was changed in place.

    Attributes:
        changed_columns: ``(table, column)`` pairs, in table and column order.
    """

    def __init__(self, changed_columns: Iterable[tuple[str, str]]) -> None:
        self.changed_columns = tuple(changed_columns)
        super().__init__(self.changed_columns)

    def __str__(self) -> str:
        columns = ", ".join(
            f"column {column!r} of table {table!r}"
            for table, column in self.changed_columns
        )
        return (
            "Refusing to continue: column memory was changed in place after "
            f"frames began sharing it, in {columns}. The usual cause is a raw "
            "write that bypassed pandas, which changes every frame still "
            "holding that memory. An in-place pandas write made after the "
            "other tables sharing the memory were freed is reported too."
        )


class _StoredArray(NamedTuple):
    """One array holding part of a column.

    ``hidden_positions`` marks positions whose stored values mean nothing,
    because a nullable column's mask says they are missing. pandas itself
    overwrites them, for example in ``cumsum``, so they are left out of the
    digest.
    """

    values: np.ndarray
    hidden_positions: np.ndarray | None = None


class _OwnerRecord:
    """The digests of the arrays stored in one owning array's memory."""

    __slots__ = ("digests", "reference")

    def __init__(self, reference: weakref.ref) -> None:
        self.reference = reference
        self.digests: dict[_BufferKey, bytes] = {}


#: Records by ``id`` of the owning array. Each entry's weak reference removes it
#: when the owner is freed, so a later array at the same address or with the
#: same ``id`` is never compared with it.
_owner_records: dict[int, _OwnerRecord] = {}


def _forget_owner(owner_id: int, _reference: weakref.ref) -> None:
    _owner_records.pop(owner_id, None)


def _tracked_owner_count() -> int:
    """How many owning arrays currently have recorded digests."""

    return len(_owner_records)


def _column_buffers(column: pd.Series | pd.Index) -> tuple[_StoredArray, ...]:
    """The arrays holding a column's data, without copying them.

    Returns an empty tuple for storage this module does not track.
    """

    if isinstance(column.dtype, np.dtype):
        return (_StoredArray(column.to_numpy()),)
    values = column.array
    if isinstance(values, _NULLABLE_ARRAY_TYPES):
        # pandas has no public way to reach these arrays when the column has
        # missing values: to_numpy() then returns a converted copy. They are
        # only read here.
        mask = values._mask
        return (_StoredArray(values._data, mask), _StoredArray(mask))
    if isinstance(values, pd.Categorical):
        return (_StoredArray(values.codes), *_column_buffers(values.categories))
    if isinstance(values, pd.arrays.StringArray):
        return (_StoredArray(values.to_numpy()),)
    if isinstance(values, _DATETIME_LIKE_ARRAY_TYPES):
        return (_StoredArray(values.asi8),)
    if isinstance(values, pd.arrays.SparseArray):
        return (
            _StoredArray(values.sp_values),
            _StoredArray(values.sp_index.to_int_index().indices),
        )
    return ()


def _owner_of(array: np.ndarray) -> np.ndarray:
    """The array at the root of ``array``'s chain of views."""

    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    return owner


def _array_address(array: np.ndarray) -> int:
    return array.__array_interface__["data"][0]


def _buffer_key(stored: _StoredArray) -> _BufferKey:
    values = stored.values
    hidden_positions = stored.hidden_positions
    return (
        _array_address(values),
        values.shape[0],
        values.strides[0],
        values.dtype.str,
        0 if hidden_positions is None else _array_address(hidden_positions),
    )


def _buffer_digest(stored: _StoredArray) -> bytes:
    """Digest the meaningful contents of one stored array.

    An object array stores references, so its digest covers which object each
    position refers to. That needs no hashable values, so columns of lists work,
    and replacing an element with another object changes it.
    """

    values = stored.values
    hidden_positions = stored.hidden_positions
    if hidden_positions is not None and hidden_positions.any():
        values = values.copy()
        values[hidden_positions] = 0
    if values.dtype.hasobject:
        contents = np.fromiter(map(id, values), dtype=np.uintp, count=len(values))
    elif values.flags.c_contiguous:
        contents = values
    else:
        contents = values.copy(order="C")
    return hashlib.sha256(contents.view(np.uint8)).digest()


def _named_columns(
    tables: Mapping[str, pd.DataFrame], strata: pd.Series | None
) -> Iterator[tuple[str, str, pd.Series]]:
    """Each column of the DataFrames given; anything else is left to its caller."""

    for table_name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            for column_name, column in table.items():
                yield str(table_name), str(column_name), column
    if isinstance(strata, pd.Series):
        yield "strata", "strata", strata


def _record_buffer(stored: _StoredArray) -> None:
    owner = _owner_of(stored.values)
    owner_id = id(owner)
    record = _owner_records.get(owner_id)
    if record is None:
        reference = weakref.ref(owner, partial(_forget_owner, owner_id))
        record = _owner_records.setdefault(owner_id, _OwnerRecord(reference))
    key = _buffer_key(stored)
    if key not in record.digests:
        record.digests[key] = _buffer_digest(stored)


def _buffer_changed(stored: _StoredArray) -> bool:
    record = _owner_records.get(id(_owner_of(stored.values)))
    if record is None:
        return False
    recorded_digest = record.digests.get(_buffer_key(stored))
    return recorded_digest is not None and recorded_digest != _buffer_digest(stored)


def record_shared_columns(
    tables: Mapping[str, pd.DataFrame], *, strata: pd.Series | None = None
) -> None:
    """Record digests of the columns of ``tables`` that are about to be shared.

    A column whose memory already has a digest keeps it, so a change made
    before a later frame starts sharing the column is still found.

    Args:
        tables: DataFrames by table name.
        strata: Per-person stratum labels, if they are shared too.
    """

    for _table_name, _column_name, column in _named_columns(tables, strata):
        for stored in _column_buffers(column):
            _record_buffer(stored)


def check_shared_columns(
    tables: Mapping[str, pd.DataFrame], *, strata: pd.Series | None = None
) -> None:
    """Refuse tables whose recorded column memory has changed.

    Columns with no recorded digest are skipped, so this is safe to call on
    any table.

    Args:
        tables: DataFrames by table name, used to name changed columns.
        strata: Per-person stratum labels, reported as table ``"strata"``.

    Raises:
        SharedColumnChangedError: If any recorded column has changed.
    """

    if not _owner_records:
        return
    changed_columns = [
        (table_name, column_name)
        for table_name, column_name, column in _named_columns(tables, strata)
        if any(_buffer_changed(stored) for stored in _column_buffers(column))
    ]
    if changed_columns:
        raise SharedColumnChangedError(changed_columns)
