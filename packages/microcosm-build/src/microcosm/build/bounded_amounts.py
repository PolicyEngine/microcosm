"""Whether amounts stay within a bound each record carries.

A check on a finished frame may need to know whether an amount never exceeds
a bound held on the same record, such as a reduction within the bill it
reduces, where the amount sits on the record itself or on its members, summed
to it through their membership column (microcosm#931). This module answers
that for any tables: the caller names the bound's entity, its id and bound
columns, the amount columns on the same table, and each member table's amount
and membership columns.

It names no country and imports nothing first-party. Exclusions are the
caller's: a check that leaves records out drops them, and their members,
before calling. Its results are booleans and its texts name tables and
columns only: never an id, a value or a count of rows, so a check that prints
them prints nothing of a record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["FINITE_NONNEGATIVE", "amounts_within_bound"]

#: The result's key for whether every amount and bound read is finite and
#: nonnegative; no amount column may take the name.
FINITE_NONNEGATIVE = "finite_nonnegative"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Bounded amounts: " + message)


def _name(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _names(value: object) -> bool:
    """A sequence of names that is not itself a string."""
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(_name(name) for name in value)
    )


def _table(tables: Mapping[str, pd.DataFrame], entity: str) -> pd.DataFrame:
    _require(
        isinstance(tables.get(entity), pd.DataFrame),
        f"the {entity} table is not a DataFrame",
    )
    return tables[entity]


def _column(table: pd.DataFrame, entity: str, column: str) -> pd.Series:
    _require(column in table.columns, f"the {entity} table has no {column} column")
    _require(
        not table.columns[table.columns == column].duplicated().any(),
        f"the {entity} table repeats its {column} column",
    )
    return table[column]


def _amounts(table: pd.DataFrame, entity: str, column: str) -> np.ndarray:
    """A numeric column as float64, a missing value as NaN."""
    series = _column(table, entity, column)
    _require(
        getattr(series.dtype, "kind", None) in ("i", "u", "f"),
        f"{entity}.{column} is not a numeric column",
    )
    return series.to_numpy(dtype=np.float64, na_value=np.nan)


def amounts_within_bound(
    tables: Mapping[str, pd.DataFrame],
    *,
    entity: str,
    id_column: str,
    bound_column: str,
    same_entity_columns: Sequence[str],
    member_columns: Mapping[str, Sequence[str]],
    membership_columns: Mapping[str, str],
    absolute_tolerance: float,
) -> dict[str, bool]:
    """Per named column, whether every record's amount is within its bound.

    ``tables[entity]`` holds one row per record, its ids in ``id_column`` and
    its bound in ``bound_column``. Each of ``same_entity_columns`` is an
    amount on the same table. Each of ``member_columns[member]`` is an amount
    on ``tables[member]``, each row of which names its record's id in
    ``membership_columns[member]``; a record's amount is the sum of its
    members' (zero where it has none), summed as float64 in row order.

    Returns one boolean per named column, keyed by the column, in the order
    named (the same table's, then each member table's), each whether every
    record's amount is at most its bound plus ``absolute_tolerance``; then
    :data:`FINITE_NONNEGATIVE`, whether every value read of the bound and of
    each named column, member rows included, is finite and nonnegative, so
    any NaN, infinite or negative value read fails it. A missing value is
    NaN: a NaN amount fails its column, and a NaN bound fails every column.
    An amount of plus infinity fails its column unless its bound is plus
    infinity too; minus infinity, like any amount at most its bound, passes
    it. A table without rows passes.

    Exclusions are the caller's: it drops the records a check leaves out,
    and their members, before calling.

    Refused, with texts that name tables and columns only: tables that are
    not a mapping, or a table that is not a DataFrame; a name that is not a
    nonempty string without surrounding whitespace, or a column list that is
    a string; a bound column that is the id column; no named column, a
    column named twice or named :data:`FINITE_NONNEGATIVE`, or an amount
    that is the bound or id column or its own table's membership column; a
    member table that is the record's own, or membership columns not given
    for exactly the member tables; a column a table lacks or repeats; a
    bound or amount column that is not of an integer or float dtype; ids
    that are missing or repeated; a member row whose membership names no
    record; and a tolerance that is not a finite nonnegative number.
    """
    _require(isinstance(tables, Mapping), "tables must map entities to DataFrames")
    _require(
        _name(entity) and _name(id_column) and _name(bound_column),
        "entity, id_column and bound_column must be names",
    )
    _require(id_column != bound_column, "the bound column must not be the id column")
    _require(_names(same_entity_columns), "same_entity_columns must list columns")
    _require(
        isinstance(member_columns, Mapping)
        and all(
            _name(member) and _names(columns)
            for member, columns in member_columns.items()
        ),
        "member_columns must map member tables to lists of columns",
    )
    _require(
        entity not in member_columns,
        "a member table must not be the record's own table",
    )
    _require(
        isinstance(membership_columns, Mapping)
        and set(membership_columns) == set(member_columns)
        and all(_name(column) for column in membership_columns.values()),
        "membership_columns must name one column for each member table",
    )
    named = [
        *same_entity_columns,
        *(column for columns in member_columns.values() for column in columns),
    ]
    _require(bool(named), "no amount column is named")
    _require(len(set(named)) == len(named), "an amount column is named twice")
    _require(
        FINITE_NONNEGATIVE not in named,
        f"no amount column may be named {FINITE_NONNEGATIVE}",
    )
    _require(
        not {id_column, bound_column} & set(same_entity_columns),
        "an amount column is the id or the bound column",
    )
    _require(
        all(
            membership_columns[member] not in columns
            for member, columns in member_columns.items()
        ),
        "an amount column is its table's membership column",
    )
    _require(
        isinstance(absolute_tolerance, int | float)
        and not isinstance(absolute_tolerance, bool)
        and math.isfinite(absolute_tolerance)
        and absolute_tolerance >= 0,
        "absolute_tolerance must be a finite nonnegative number",
    )

    table = _table(tables, entity)
    ids = pd.Index(_column(table, entity, id_column).to_numpy())
    _require(
        not ids.hasnans and ids.is_unique,
        f"{entity}.{id_column} must name each row once",
    )
    bound = _amounts(table, entity, bound_column)
    limit = bound + float(absolute_tolerance)
    read = [bound]
    result: dict[str, bool] = {}
    for column in same_entity_columns:
        amount = _amounts(table, entity, column)
        read.append(amount)
        result[column] = bool(np.all(amount <= limit))
    for member, columns in member_columns.items():
        rows = _table(tables, member)
        membership = _column(rows, member, membership_columns[member])
        position = ids.get_indexer(membership.to_numpy())
        _require(
            not membership.isna().any() and bool(np.all(position >= 0)),
            f"a {member} row's {membership_columns[member]} names no {entity} row",
        )
        for column in columns:
            amount = _amounts(rows, member, column)
            read.append(amount)
            total = np.bincount(position, weights=amount, minlength=len(ids))
            result[column] = bool(np.all(total <= limit))
    result[FINITE_NONNEGATIVE] = all(
        bool(np.all(np.isfinite(values) & (values >= 0))) for values in read
    )
    return result
