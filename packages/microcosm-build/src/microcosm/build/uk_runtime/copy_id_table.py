"""Fixed decimal fields for the ids of UK copy rows (microcosm#931).

Every UK copy layer used to offset its copies' ids by a power of ten one digit
above the largest id it saw. Once the spine clones households at its start,
that largest id grows with the clone count K, so every later offset would move
with K and the K=1 spine would stop being the clone-0 slice of the K=15 spine.
Here each layer owns a decimal field of the id instead, declared up front:
copy ``k`` of a row in field ``f`` takes ``id + k x radix(f)`` on all five id
columns (``household_id``; ``person_id``, ``person_household_id`` and
``person_benunit_id``; ``benunit_id``), so every copy id is a pure function of
its source id and its copy path.

The fields are listed least significant first, in the order the spine applies
them, each with its ordinal bound (ordinals ``0 .. bound - 1``, 0 meaning "not
a copy in this field") and its radix:

- ``vintage``: reserved for pooled FRS vintages (the plan's P3). The spine
  reads one FRS release today, so nothing mints it; raw FRS ids
  (``household_id`` the sernum, ``benunit_id`` sernum x 100 + b,
  ``person_id`` sernum x 1000 + p) must stay below its radix.
- ``geographic_support``: the geographic clones the clone-first spine will
  mint at its start (K at most 100). Nothing mints it yet.
- ``spi_support_channel``: the SPI support copy (ordinal 1, stage 16) and the
  income band donor (ordinal 2, stage 17).
- ``high_value_support``: the high-value home copies (stage 26).
- ``cgt_support_split``: the CGT support copies (stage 42), four digits.
- ``cgt_incidence_clone``: the CGT incidence twin (stage 43).
- ``cgt_residential_split``: the residential arms (stage 46), the top field,
  so arms follow every incumbent in id order.

A field must sit above every field filled before it runs, which keeps a copy's
parent below the field's radix: a copy's ordinal is then ``id // radix``
modulo the field's width, and its parent ``id - ordinal x radix``. The table
is UK-local; a shared, country-neutral table comes with a later stage. This
module imports nothing from the UK runtime, so any module can read it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

__all__ = [
    "UK_COPY_ID_BASE_RADIX",
    "UK_COPY_ID_FIELDS",
    "UK_COPY_ID_LARGEST_ID",
    "CopyIdField",
    "mint",
    "ordinal",
    "ordinal_bound",
    "parent",
    "radix",
    "require_below",
]

_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class CopyIdField:
    """One decimal field of a copy id: ordinals ``0 .. bound - 1`` at ``radix``."""

    name: str
    bound: int
    radix: int

    @property
    def width(self) -> int:
        """The field's span, ``10 ** digits(bound - 1)``."""

        return 10 ** len(str(self.bound - 1))


#: Raw ids sit below this radix, the first field's.
UK_COPY_ID_BASE_RADIX = 10**8
#: The fields in the order the spine applies them, least significant first.
UK_COPY_ID_FIELDS: tuple[CopyIdField, ...] = (
    CopyIdField("vintage", 10, 10**8),
    CopyIdField("geographic_support", 100, 10**9),
    CopyIdField("spi_support_channel", 3, 10**11),
    CopyIdField("high_value_support", 6, 10**12),
    CopyIdField("cgt_support_split", 10_000, 10**13),
    CopyIdField("cgt_incidence_clone", 2, 10**17),
    CopyIdField("cgt_residential_split", 8, 10**18),
)


def _largest_id(fields: Sequence[CopyIdField], base_radix: int) -> int:
    """The largest id the fields can mint: every ordinal at its bound less one."""

    return sum((field.bound - 1) * field.radix for field in fields) + base_radix - 1


def _validate_fields(fields: Sequence[CopyIdField], base_radix: int) -> int:
    """Refuse a malformed table; return its largest mintable id.

    The first radix is the base radix; every later radix is the previous one
    times the previous field's width, so the fields are contiguous decimal
    digits and none overlaps another; the largest mintable id fits int64.
    """

    names = [field.name for field in fields]
    if not names or len(set(names)) != len(names):
        raise ValueError("The copy-id fields must be named, each once.")
    expected = base_radix
    for field in fields:
        if type(field.bound) is not int or field.bound < 2:
            raise ValueError(
                f"Copy-id field {field.name!r} needs an integer bound of at least 2."
            )
        if field.radix != expected:
            raise ValueError(
                f"Copy-id field {field.name!r} sits at radix {field.radix}, not "
                f"{expected}, so the fields are not contiguous decimal digits."
            )
        expected = field.radix * field.width
    largest = _largest_id(fields, base_radix)
    if largest >= _INT64_MAX:
        raise ValueError(
            f"The copy-id fields can mint ids up to {largest}, which int64 cannot hold."
        )
    return largest


#: About 7.2e18, below int64's 2**63 - 1.
UK_COPY_ID_LARGEST_ID = _validate_fields(UK_COPY_ID_FIELDS, UK_COPY_ID_BASE_RADIX)
_FIELDS = MappingProxyType({field.name: field for field in UK_COPY_ID_FIELDS})


def _field(name: str) -> CopyIdField:
    try:
        return _FIELDS[name]
    except KeyError as error:
        raise ValueError(f"No UK copy-id field is named {name!r}.") from error


def radix(field: str) -> int:
    """The declared radix of ``field``."""

    return _field(field).radix


def ordinal_bound(field: str) -> int:
    """The declared bound of ``field``: its ordinals run below it."""

    return _field(field).bound


def _ids(values: object, *, what: str) -> np.ndarray:
    """Exact int64 ids, refusing floats, booleans and negative ids by count."""

    array = np.asarray(values)
    if array.dtype.kind == "O":
        items = array.ravel().tolist()
        if any(
            isinstance(item, bool | np.bool_) or not isinstance(item, int | np.integer)
            for item in items
        ):
            raise TypeError(f"{what} must be integers.")
        if any(not -_INT64_MAX - 1 <= int(item) <= _INT64_MAX for item in items):
            raise ValueError(f"{what} must fit int64.")
        array = np.asarray([int(item) for item in items], dtype=np.int64).reshape(
            array.shape
        )
    elif array.dtype.kind == "u":
        if array.size and int(array.max()) > _INT64_MAX:
            raise ValueError(f"{what} must fit int64.")
    elif array.dtype.kind != "i":
        raise TypeError(f"{what} must be integers, not {array.dtype}.")
    result = array.astype(np.int64, copy=False)
    negative = int((result < 0).sum())
    if negative:
        raise ValueError(f"{negative} of the {what} are negative.")
    return result


def require_below(ids: object, field: str, *, what: str = "ids") -> None:
    """Refuse any id at or above ``field``'s radix, printing only the count.

    A stage that mints ``field`` copies rows whose ids must all sit below it,
    so that every copy's parent is recoverable from the copy's id.
    """

    values = _ids(ids, what=what)
    reached = int((values >= radix(field)).sum())
    if reached:
        raise ValueError(
            f"{reached} of the {what} reach the {field!r} copy-id field "
            f"(radix {radix(field)}), which only a later stage may fill "
            "(microcosm#931)."
        )


def ordinal(ids: object, field: str) -> np.ndarray:
    """Each id's ordinal in ``field``: ``(id // radix) % width``."""

    spec = _field(field)
    return (_ids(ids, what="ids") // spec.radix) % spec.width


def parent(ids: object, field: str) -> np.ndarray:
    """Each id with its ``field`` digits cleared: the row it was copied from."""

    values = _ids(ids, what="ids")
    spec = _field(field)
    return values - ((values // spec.radix) % spec.width) * spec.radix


def mint(ids: object, field: str, ordinals: object) -> np.ndarray:
    """``ids + ordinals x radix(field)`` as int64, refusing every malformed copy.

    Refuses an ordinal outside ``[1, bound)`` in a fixed sentence (it names
    the field and its declared bound, never how many copies a row needed),
    an id whose ``field`` digits are already set (a copy of a copy in the same
    field), and a result int64 cannot hold. ``ordinals`` is one integer or one
    per id. The other messages carry counts and declared constants only,
    never an id.
    """

    spec = _field(field)
    values = _ids(ids, what="ids")
    steps = _ids(ordinals, what=f"{field!r} ordinals")
    try:
        steps = np.broadcast_to(steps, values.shape)
    except ValueError as error:
        raise ValueError(
            f"{field!r} ordinals must be one integer or one per id."
        ) from error
    if ((steps < 1) | (steps >= spec.bound)).any():
        raise ValueError(
            f"A {field!r} copy ordinal falls outside [1, {spec.bound}), the "
            "field's declared bound (microcosm#931)."
        )
    occupied = int((((values // spec.radix) % spec.width) != 0).sum())
    if occupied:
        raise ValueError(
            f"{occupied} id(s) already carry a {field!r} ordinal; a copy is minted "
            "from a row that is not a copy in its own field (microcosm#931)."
        )
    offsets = steps * spec.radix
    overflow = int((values > _INT64_MAX - offsets).sum())
    if overflow:
        raise ValueError(
            f"{overflow} {field!r} copy id(s) would overflow int64 (microcosm#931)."
        )
    return values + offsets
