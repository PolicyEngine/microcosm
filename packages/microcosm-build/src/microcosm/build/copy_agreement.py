"""Whether copied rows still equal the rows they were copied from.

A copying step makes each copy a whole-row copy of its parent and rewrites
only what it declares; a later step that re-draws a copy's columns gives the
copy values of its own. A check on a finished frame can then ask, column by
column, whether every copy of a layer still carries its immediate parent's
values on the columns that layer must not move. This module answers that for
any table, any id column and any copy layer: the caller says which rows are
the layer's copies and which row each was copied from, as its own copy-id
rules decode them, and names the columns.

It names no country, reads no frame and imports nothing first-party. Its
results are booleans and its texts name columns only: never an id, a value
or a count of rows, so a check that prints them prints nothing of a record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

__all__ = ["CopyAgreement", "rows_agree_with_parents"]


@dataclass(frozen=True)
class CopyAgreement:
    """Whether every checked copy row equals its parent, column by column.

    ``columns`` maps each named column to whether every checked row's value
    equals its parent's (missing on both counts as equal; a column the table
    lacks reads ``False``). ``parents_present`` says whether every checked
    row's parent is a row of the table. ``copies_checked`` says whether any
    row was flagged as a copy. ``failures`` carries one count-free text per
    failing column or condition.
    """

    columns: Mapping[str, bool] = field(default_factory=dict)
    parents_present: bool = True
    copies_checked: bool = False
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures


def _same(child: pd.Series, parent: pd.Series) -> np.ndarray:
    """Row by row, whether the two values are equal or both missing."""

    child = child.reset_index(drop=True)
    parent = parent.reset_index(drop=True)
    child_missing = child.isna().to_numpy(dtype=bool)
    parent_missing = parent.isna().to_numpy(dtype=bool)
    present = ~child_missing & ~parent_missing
    equal = np.zeros(len(child), dtype=bool)
    if present.any():
        # Only rows where both values are present are compared, so no
        # missing marker reaches the comparison.
        equal[present] = child[present].eq(parent[present]).to_numpy(dtype=bool)
    return equal | (child_missing & parent_missing)


def rows_agree_with_parents(
    table: pd.DataFrame,
    *,
    id_column: str,
    parent_ids: Sequence[object] | np.ndarray,
    rows: Sequence[bool] | np.ndarray,
    columns: Iterable[str],
) -> CopyAgreement:
    """Whether each of ``rows`` equals its parent on each of ``columns``.

    ``parent_ids`` gives, for every row of ``table`` in order, the
    ``id_column`` value of the row it was copied from; only the rows
    ``rows`` flags are read. Each flagged row is compared with the table's
    row whose id is its parent's, column by column. A flagged row whose
    parent is not in the table fails the check, as do ids that do not name
    each row once and a named column the table lacks.

    Refused, by a fixed text: ``parent_ids`` or ``rows`` that do not give
    one entry per row, and an ``id_column`` the table lacks.
    """

    names = tuple(str(column) for column in columns)
    flagged = np.asarray(rows, dtype=bool)
    parents = np.asarray(parent_ids)
    if flagged.shape != (len(table),) or parents.shape != (len(table),):
        raise ValueError(
            "rows_agree_with_parents reads one copy flag and one parent id per "
            "table row."
        )
    if id_column not in table.columns:
        raise ValueError(f"The table has no {id_column!r} column to find parents by.")
    failures: list[str] = []
    agree = dict.fromkeys(names, False)
    ids = pd.Index(table[id_column].to_numpy())
    if ids.hasnans or not ids.is_unique:
        failures.append(f"{id_column}: the ids do not name each row once.")
        return CopyAgreement(
            columns=MappingProxyType(agree),
            parents_present=False,
            copies_checked=bool(flagged.any()),
            failures=tuple(failures),
        )
    positions = ids.get_indexer(parents[flagged])
    found = positions >= 0
    parents_present = bool(found.all())
    if not parents_present:
        failures.append(f"{id_column}: a copy row's parent is not in the table.")
    child_positions = np.flatnonzero(flagged)[found]
    parent_positions = positions[found]
    for column in names:
        if column not in table.columns:
            failures.append(f"{column}: the table has no such column.")
            continue
        values = table[column]
        same = _same(values.iloc[child_positions], values.iloc[parent_positions])
        agree[column] = bool(same.all())
        if not agree[column]:
            failures.append(f"{column}: a copy row differs from its parent.")
    return CopyAgreement(
        columns=MappingProxyType(agree),
        parents_present=parents_present,
        copies_checked=bool(flagged.any()),
        failures=tuple(failures),
    )
