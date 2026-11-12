"""The shared copy-agreement check (microcosm#931): copies against parents.

Invented tables only. A copy's parent is named by id, as a country's copy-id
rules decode it; the check compares each flagged copy with its parent,
column by column, and its texts name columns only.
"""

from __future__ import annotations

import ast
import inspect
import re

import numpy as np
import pandas as pd
import pytest

from microcosm.build import copy_agreement
from microcosm.build.copy_agreement import CopyAgreement, rows_agree_with_parents


def _table() -> pd.DataFrame:
    """Two parents (ids 1 and 2) and three copies of them (11, 12 and 21)."""

    return pd.DataFrame(
        {
            "row_id": np.array([1, 2, 11, 12, 21], dtype=np.int64),
            "area": pd.array(["a", "b", "a", "a", "b"], dtype="string"),
            "region": ["north", "south", "north", "north", "south"],
            "flag": [True, False, True, True, False],
            "value": [1.5, np.nan, 1.5, 1.5, np.nan],
            "kind": pd.Categorical(["x", "y", "x", "x", "y"]),
            "own": ["p", "q", "r", "s", "t"],
        },
        index=[50, 40, 30, 20, 10],
    )


_PARENTS = np.array([1, 2, 1, 1, 2], dtype=np.int64)
_COPIES = np.array([False, False, True, True, True])
_HELD = ("area", "region", "flag", "value", "kind")


def _no_digit(texts) -> bool:
    return not any(re.search(r"\d", text) for text in texts)


def test_copies_that_carry_their_parents_values_agree() -> None:
    result = rows_agree_with_parents(
        _table(), id_column="row_id", parent_ids=_PARENTS, rows=_COPIES, columns=_HELD
    )

    assert isinstance(result, CopyAgreement)
    assert result.passed and result.failures == ()
    assert dict(result.columns) == dict.fromkeys(_HELD, True)
    assert result.parents_present and result.copies_checked
    # Missing on both sides counts as equal: the value column is NaN on the
    # second parent and its copy.


def test_a_column_the_copies_redraw_differs_and_is_named_alone() -> None:
    result = rows_agree_with_parents(
        _table(),
        id_column="row_id",
        parent_ids=_PARENTS,
        rows=_COPIES,
        columns=(*_HELD, "own"),
    )

    assert not result.passed
    assert dict(result.columns) == {**dict.fromkeys(_HELD, True), "own": False}
    assert result.failures == ("own: a copy row differs from its parent.",)
    assert _no_digit(result.failures)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("area", pd.NA),
        ("area", "c"),
        ("region", None),
        ("value", 2.5),
        ("value", np.nan),
        ("flag", False),
        ("kind", "y"),
    ],
)
def test_one_copy_that_moved_fails_its_column(column, value) -> None:
    table = _table()
    # The copy 12, at position 3, whose parent is 1.
    table.iloc[3, table.columns.get_loc(column)] = value

    result = rows_agree_with_parents(
        table, id_column="row_id", parent_ids=_PARENTS, rows=_COPIES, columns=_HELD
    )

    assert not result.passed
    assert {name for name, agrees in result.columns.items() if not agrees} == {column}
    assert result.failures == (f"{column}: a copy row differs from its parent.",)


def test_a_missing_value_on_both_sides_agrees_in_object_columns_too() -> None:
    table = _table()
    table["region"] = pd.Series(
        ["north", pd.NA, "north", "north", pd.NA], index=table.index, dtype=object
    )

    result = rows_agree_with_parents(
        table, id_column="row_id", parent_ids=_PARENTS, rows=_COPIES, columns=_HELD
    )

    assert result.passed, result.failures


def test_rows_not_flagged_are_never_read() -> None:
    table = _table()
    table.loc[table["row_id"] == 2, "own"] = "changed"
    flagged = np.array([False, False, True, True, False])

    result = rows_agree_with_parents(
        table, id_column="row_id", parent_ids=_PARENTS, rows=flagged, columns=_HELD
    )

    assert result.passed
    # No copy flagged: every column agrees, vacuously.
    empty = rows_agree_with_parents(
        table,
        id_column="row_id",
        parent_ids=_PARENTS,
        rows=np.zeros(5, dtype=bool),
        columns=(*_HELD, "own"),
    )
    assert empty.passed and not empty.copies_checked
    assert set(empty.columns.values()) == {True}


def test_a_copy_whose_parent_is_absent_fails_by_the_id_column() -> None:
    parents = _PARENTS.copy()
    parents[4] = 3

    result = rows_agree_with_parents(
        _table(), id_column="row_id", parent_ids=parents, rows=_COPIES, columns=_HELD
    )

    assert not result.passed and not result.parents_present
    assert result.failures == ("row_id: a copy row's parent is not in the table.",)
    # The copies whose parents are present are still compared.
    assert set(result.columns.values()) == {True}


def test_ids_that_do_not_name_each_row_once_fail() -> None:
    table = _table()
    table.loc[table.index[4], "row_id"] = 12

    result = rows_agree_with_parents(
        table, id_column="row_id", parent_ids=_PARENTS, rows=_COPIES, columns=_HELD
    )

    assert not result.passed
    assert result.failures == ("row_id: the ids do not name each row once.",)
    assert set(result.columns.values()) == {False}


def test_a_column_the_table_lacks_fails_by_name() -> None:
    result = rows_agree_with_parents(
        _table(),
        id_column="row_id",
        parent_ids=_PARENTS,
        rows=_COPIES,
        columns=("area", "absent"),
    )

    assert not result.passed
    assert dict(result.columns) == {"area": True, "absent": False}
    assert result.failures == ("absent: the table has no such column.",)


def test_misaligned_inputs_are_refused_by_a_fixed_text() -> None:
    with pytest.raises(ValueError, match="one copy flag and one parent id per"):
        rows_agree_with_parents(
            _table(),
            id_column="row_id",
            parent_ids=_PARENTS[:4],
            rows=_COPIES,
            columns=_HELD,
        )
    with pytest.raises(ValueError, match="one copy flag and one parent id per"):
        rows_agree_with_parents(
            _table(),
            id_column="row_id",
            parent_ids=_PARENTS,
            rows=_COPIES[:4],
            columns=_HELD,
        )
    with pytest.raises(ValueError, match="no 'missing' column"):
        rows_agree_with_parents(
            _table(),
            id_column="missing",
            parent_ids=_PARENTS,
            rows=_COPIES,
            columns=_HELD,
        )


def test_the_check_reads_rows_by_position_whatever_the_index() -> None:
    table = _table()
    order = [4, 2, 0, 3, 1]
    shuffled = table.iloc[order]

    result = rows_agree_with_parents(
        shuffled,
        id_column="row_id",
        parent_ids=_PARENTS[order],
        rows=_COPIES[order],
        columns=(*_HELD, "own"),
    )

    assert dict(result.columns) == {**dict.fromkeys(_HELD, True), "own": False}


def test_the_module_names_no_country_and_imports_nothing_first_party() -> None:
    source = inspect.getsource(copy_agreement)
    tree = ast.parse(source)
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "microcosm" not in imported
    for country in ("uk", "us", "nz", "frs", "spi", "cgt"):
        assert not re.search(rf"\b{country}\b", source, flags=re.IGNORECASE), country
