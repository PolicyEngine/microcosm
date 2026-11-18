"""Find raw writes into column memory that frames share.

Under pandas copy-on-write, frame operations such as
:meth:`~microcosm.frame.bundle.Frame.with_weights` share the memory of unchanged
columns. A write made through pandas stays on one frame, but a raw write into
the memory, for example through ``np.add.at`` or the array behind a nullable
column, changes every frame sharing it. This module detects that afterwards. It
does not prevent the write or identify the code that made it.

**Recording.** When an operation starts sharing a column,
:func:`record_shared_columns` stores a SHA-256 digest of each array holding the
column's data. A digest belongs to the array's memory, not to a frame, so a
column is hashed once however many frames go on to share it. Columns that are
never shared are never hashed. Digests live only in this process and are dropped
when the array that owns the memory is freed.

**Comparing.** :func:`check_shared_columns` hashes each recorded array again and
raises :class:`SharedColumnChangedError` naming every column that differs. The
writers of checkpoints and outputs call it before writing.

A column is compared only while it is still stored in the memory that was
recorded. Replacing, adding or dropping a column through pandas moves the table
to other memory, and so does a pandas write while the memory is shared, so none
of these trip the check. A pandas write into a single cell of memory no other
table refers to any more happens in place and is reported, because it changes
the recorded memory of a frame whose tables are documented as read-only.

Not covered: pyarrow-backed columns, whose buffers are immutable, and any other
extension type :func:`_column_buffers` does not list.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Iterable, Iterator, Mapping
from functools import partial

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

_BufferKey = tuple[int, int, int, str]


class SharedColumnChangedError(RuntimeError):
    """A column's shared memory changed without going through pandas.

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
            "Refusing to continue: memory shared between frames changed after "
            f"sharing began, in {columns}. Something wrote into that memory "
            "directly instead of through pandas, so every frame sharing it now "
            "holds the changed values."
        )


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


def _column_buffers(column: pd.Series) -> tuple[np.ndarray, ...]:
    """The arrays holding a column's data, without copying them.

    Returns an empty tuple for storage this module does not track.
    """

    if isinstance(column.dtype, np.dtype):
        return (column.to_numpy(),)
    values = column.array
    if isinstance(values, _NULLABLE_ARRAY_TYPES):
        # pandas has no public way to reach these arrays when the column has
        # missing values: to_numpy() then returns a converted copy. They are
        # only read here.
        return (values._data, values._mask)
    if isinstance(values, pd.Categorical):
        return (values.codes,)
    if isinstance(values, pd.arrays.StringArray):
        return (values.to_numpy(),)
    if isinstance(values, _DATETIME_LIKE_ARRAY_TYPES):
        return (values.asi8,)
    if isinstance(values, pd.arrays.SparseArray):
        return (values.sp_values,)
    return ()


def _owner_of(buffer: np.ndarray) -> np.ndarray:
    """The array at the root of ``buffer``'s chain of views."""

    owner = buffer
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    return owner


def _buffer_key(buffer: np.ndarray) -> _BufferKey:
    return (
        buffer.__array_interface__["data"][0],
        buffer.shape[0],
        buffer.strides[0],
        buffer.dtype.str,
    )


def _buffer_digest(buffer: np.ndarray) -> bytes:
    """Digest the bytes stored in ``buffer``.

    An object array stores references, so its digest covers which object each
    position refers to. That needs no hashable values, so columns of lists work,
    and replacing an element with another object changes it.
    """

    if buffer.dtype.hasobject:
        contents = np.fromiter(map(id, buffer), dtype=np.uintp, count=len(buffer))
    elif buffer.flags.c_contiguous:
        contents = buffer
    else:
        contents = buffer.copy(order="C")
    return hashlib.sha256(contents.view(np.uint8)).digest()


def _named_columns(
    tables: Mapping[str, pd.DataFrame], strata: pd.Series | None
) -> Iterator[tuple[str, str, pd.Series]]:
    for table_name, table in tables.items():
        for column_name, column in table.items():
            yield str(table_name), str(column_name), column
    if strata is not None:
        yield "strata", "strata", strata


def _record_buffer(buffer: np.ndarray) -> None:
    owner = _owner_of(buffer)
    owner_id = id(owner)
    record = _owner_records.get(owner_id)
    if record is None:
        reference = weakref.ref(owner, partial(_forget_owner, owner_id))
        record = _owner_records.setdefault(owner_id, _OwnerRecord(reference))
    key = _buffer_key(buffer)
    if key not in record.digests:
        record.digests[key] = _buffer_digest(buffer)


def _buffer_changed(buffer: np.ndarray) -> bool:
    record = _owner_records.get(id(_owner_of(buffer)))
    if record is None:
        return False
    recorded_digest = record.digests.get(_buffer_key(buffer))
    return recorded_digest is not None and recorded_digest != _buffer_digest(buffer)


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
        for buffer in _column_buffers(column):
            _record_buffer(buffer)


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
        if any(_buffer_changed(buffer) for buffer in _column_buffers(column))
    ]
    if changed_columns:
        raise SharedColumnChangedError(changed_columns)
