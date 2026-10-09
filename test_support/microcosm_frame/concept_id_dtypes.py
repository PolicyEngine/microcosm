"""Concept frames whose id and pointer columns take every accepted integer dtype.

The concept-frame contract accepts ids and pointers of any integer dtype:
NumPy or pandas nullable, signed or unsigned, any width. Matching ids across
those dtypes is where values wrap. A bare int64 cast turns ``2**64 - 1`` into
-1, and pandas matches against a narrower unsigned index by casting the other
side down to it (261 onto 5 for uint8). :func:`id_typed_frames` relabels a
valid frame's ids onto values at those edges, may plant pointers on the
values a wrap would collide with, and gives each id column a dtype drawn from
those that hold its values. A consumer must then either refuse the frame or
treat it exactly as the same ids typed int64.
"""

# ruff: noqa: F401

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from hypothesis import strategies as st

from test_support.microcosm_frame.concept_frames import concept_frames

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
UINT64_MAX = 2**64 - 1

#: Every column holding a person or household id, as (entity, column), in the
#: order a refusal names them.
ID_COLUMNS: tuple[tuple[str, str], ...] = (
    ("person", "person_id"),
    ("person", "person_household_id"),
    ("person", "partner_person_id"),
    ("person", "parent_1_person_id"),
    ("person", "parent_2_person_id"),
    ("household", "household_id"),
    ("household", "reference_person_id"),
)
_PERSON_POINTERS = ("partner_person_id", "parent_1_person_id", "parent_2_person_id")
#: The id columns a frame requires, never null; the rest are nullable pointers.
_REQUIRED = {
    ("person", "person_id"),
    ("person", "person_household_id"),
    ("household", "household_id"),
}

#: Every integer dtype the contract accepts for an id, NumPy then nullable.
ID_DTYPES = (
    *(np.dtype(f"{sign}int{bits}") for sign in ("", "u") for bits in (8, 16, 32, 64)),
    *(
        pd.api.types.pandas_dtype(f"{sign}Int{bits}")
        for sign in ("", "U")
        for bits in (8, 16, 32, 64)
    ),
)

#: Values within one of a power of two where some integer dtype wraps.
_EDGES = sorted(
    {
        sign * 2**power + offset
        for power in (7, 8, 15, 16, 31, 32, 63, 64)
        for sign in (1, -1)
        for offset in (-1, 0, 1)
        if INT64_MIN <= sign * 2**power + offset <= UINT64_MAX
    }
)

#: The shifts a wrap applies to a value: a multiple of a dtype's span.
_WRAPS = tuple(sign * 2**power for power in (8, 16, 32, 64) for sign in (1, -1))


def _ids(unsigned: bool) -> st.SearchStrategy[int]:
    """Ids near zero, at the dtype edges, or anywhere int64 or uint64 holds."""

    low, high = (0, UINT64_MAX) if unsigned else (INT64_MIN, INT64_MAX)
    return st.one_of(
        st.integers(max(low, -300), 300),
        st.sampled_from([value for value in _EDGES if low <= value <= high]),
        st.integers(low, high),
    )


def holds(dtype, values: list[int | None]) -> bool:
    """Whether ``dtype`` can hold every one of ``values`` (``None`` is null)."""

    if isinstance(dtype, np.dtype):
        if any(value is None for value in values):
            return False
        info = np.iinfo(dtype)
    else:
        info = np.iinfo(dtype.numpy_dtype)
    return all(info.min <= value <= info.max for value in values if value is not None)


def typed(
    values: list[int | None], dtype
) -> np.ndarray | pd.api.extensions.ExtensionArray:
    """``values`` as a column of ``dtype``."""

    if isinstance(dtype, np.dtype):
        return np.array(values, dtype=dtype)
    return pd.array(values, dtype=dtype)


class IdTypedFrame(NamedTuple):
    """A frame with retyped ids, its int64 twin, and the columns int64 cannot hold.

    ``twin`` holds the same values with every required id column typed
    ``int64`` and every pointer typed ``Int64``, or is ``None`` when ``wide``
    is not empty. ``wide`` names, as ``"entity.column"`` in
    :data:`ID_COLUMNS` order, every column holding a value above
    ``2**63 - 1``.
    """

    tables: dict[str, pd.DataFrame]
    twin: dict[str, pd.DataFrame] | None
    wide: list[str]


def _frame(
    base: dict[str, pd.DataFrame], columns: dict[tuple[str, str], object]
) -> dict[str, pd.DataFrame]:
    out = {entity: table.copy() for entity, table in base.items()}
    for (entity, column), values in columns.items():
        out[entity][column] = values
    return out


@st.composite
def id_typed_frames(draw, *, dangling_households: bool = False) -> IdTypedFrame:
    """A concept frame whose ids sit at the dtype edges, in any accepted dtype.

    Starting from a valid frame, person and household ids are relabelled
    (each set drawn wholly from the int64 range or wholly from the uint64
    range, so some dtype holds them). Each pointer column may then gain one
    planted value: an id shifted by a dtype's span, the value a wrap would
    land on an existing id, or any id. With ``dangling_households`` the
    household pointer may be planted too, which leaves a person naming no
    household. Each id column then takes a dtype drawn from the accepted ones
    that hold its values.
    """

    base = draw(concept_frames(max_households=3, max_members=4))
    person, household = base["person"], base["household"]
    old_persons = person["person_id"].tolist()
    old_households = household["household_id"].tolist()
    new_persons = draw(
        st.lists(
            _ids(draw(st.booleans(), label="unsigned person ids")),
            min_size=len(old_persons),
            max_size=len(old_persons),
            unique=True,
        ),
        label="person ids",
    )
    new_households = draw(
        st.lists(
            _ids(draw(st.booleans(), label="unsigned household ids")),
            min_size=len(old_households),
            max_size=len(old_households),
            unique=True,
        ),
        label="household ids",
    )
    to_person = dict(zip(old_persons, new_persons, strict=True))
    to_household = dict(zip(old_households, new_households, strict=True))

    def relabel(values: pd.Series, to: dict[int, int]) -> list[int | None]:
        return [None if pd.isna(value) else to[int(value)] for value in values]

    values: dict[tuple[str, str], list[int | None]] = {
        ("person", "person_id"): list(new_persons),
        ("person", "person_household_id"): relabel(
            person["person_household_id"], to_household
        ),
        **{
            ("person", column): relabel(person[column], to_person)
            for column in _PERSON_POINTERS
        },
        ("household", "household_id"): list(new_households),
        ("household", "reference_person_id"): relabel(
            household["reference_person_id"], to_person
        ),
    }

    planted = [("person", column) for column in _PERSON_POINTERS]
    planted.append(("household", "reference_person_id"))
    if dangling_households:
        planted.append(("person", "person_household_id"))
    for key in planted:
        if not draw(st.booleans(), label=f"plant {key[1]}"):
            continue
        column = values[key]
        targets = new_households if key[1] == "person_household_id" else new_persons
        row = draw(st.integers(0, len(column) - 1), label="row")
        if draw(st.booleans(), label="plant a wrap image"):
            value = draw(st.sampled_from(targets)) + draw(st.sampled_from(_WRAPS))
        else:
            value = draw(_ids(draw(st.booleans())))
        if not INT64_MIN <= value <= UINT64_MAX:
            continue
        before, column[row] = column[row], value
        if not any(holds(dtype, column) for dtype in ID_DTYPES):
            column[row] = before

    columns = {
        key: typed(
            values[key],
            draw(
                st.sampled_from(
                    [dtype for dtype in ID_DTYPES if holds(dtype, values[key])]
                ),
                label=f"{key[1]} dtype",
            ),
        )
        for key in ID_COLUMNS
    }
    wide = [
        f"{entity}.{column}"
        for entity, column in ID_COLUMNS
        if any(
            value is not None and value > INT64_MAX
            for value in values[(entity, column)]
        )
    ]
    twin = (
        None
        if wide
        else _frame(
            base,
            {
                key: typed(
                    values[key],
                    np.dtype(np.int64) if key in _REQUIRED else pd.Int64Dtype(),
                )
                for key in ID_COLUMNS
            },
        )
    )
    return IdTypedFrame(_frame(base, columns), twin, wide)


def three_person_frame() -> dict[str, pd.DataFrame]:
    """Persons 10 and 20, partners, are the parents of 30; 10 heads household 1."""

    person = pd.DataFrame(
        {
            "person_id": np.array([10, 20, 30], dtype=np.int64),
            "person_household_id": np.array([1, 1, 1], dtype=np.int64),
            "partner_person_id": pd.array([20, 10, None], dtype="Int64"),
            "parent_1_person_id": pd.array([None, None, 10], dtype="Int64"),
            "parent_2_person_id": pd.array([None, None, 20], dtype="Int64"),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array([1], dtype=np.int64),
            "reference_person_id": pd.array([10], dtype="Int64"),
        }
    )
    return {"person": person, "household": household}


#: Stands for the id int64 cannot hold in :data:`WIDE_CASES`.
WIDE = object()

#: For each way an id int64 cannot hold reaches :func:`three_person_frame`,
#: the columns that then hold it, null-free so a NumPy dtype can hold them.
#: The household id appears on both tables, so both are named.
WIDE_CASES: dict[str, dict[tuple[str, str], list[object]]] = {
    "person_id": {("person", "person_id"): [10, 20, WIDE]},
    "household_id": {
        ("person", "person_household_id"): [WIDE, WIDE, WIDE],
        ("household", "household_id"): [WIDE],
    },
    "partner_person_id": {("person", "partner_person_id"): [20, 10, WIDE]},
    "parent_1_person_id": {("person", "parent_1_person_id"): [WIDE, WIDE, 10]},
    "parent_2_person_id": {("person", "parent_2_person_id"): [WIDE, WIDE, 20]},
    "reference_person_id": {("household", "reference_person_id"): [WIDE]},
}


def wide_id_frame(
    case: str, value: int, dtype
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """:func:`three_person_frame` with ``value`` placed as ``WIDE_CASES[case]``
    says, in columns of ``dtype``; and the columns a refusal must name."""

    columns = WIDE_CASES[case]
    tables = _frame(
        three_person_frame(),
        {
            key: typed([value if item is WIDE else item for item in items], dtype)
            for key, items in columns.items()
        },
    )
    return tables, [
        f"{entity}.{column}"
        for entity, column in ID_COLUMNS
        if (entity, column) in columns
    ]


__all__ = [name for name in globals() if not name.startswith("__")]
